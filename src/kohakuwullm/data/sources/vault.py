"""Record sources over the KohakuVault caption/tag databases.

``DanbooruRecords`` is keyed by post id, ``PathKeyedRecords`` by image path;
both yield the same normalized record. Two rules any new source must follow:
reopen database handles on PID change, and never scan with a defaulted
``limit``. See docs/internals/data.md.
"""

import os

import kohakuvault as kv
import numpy as np

DEFAULT_ROOT = "/xg7/caption-datasets"

# dataset -> (caption db, tagger db or None)
PATH_KEYED: dict[str, tuple[str, str | None]] = {
    "cc12m": ("cc12m.db", "cc12m_tags.db"),
    "coyo11m": ("coyo11m.db", "coyo11m_tags.db"),
    "laion_coco": ("laion_coco.db", "laion_coco_tags.db"),
    "nozomi": ("nozomi.db", "nozomi_tags.db"),
    "imagenet": ("imagenet.db", None),
    "danbooru_tagger": ("danbooru.db", "danbooru_tags.db"),
}

TAG_CATEGORIES = ["general", "artist", "character", "copyright", "meta"]
DANBOORU_KEY_PREFIX = "/mp34-1/danbooru2023/1_full7.62M"

# Score -> quality bucket. Here, not baked into the export, so it stays a knob.
QUALITY_THRESHOLDS = [
    (150, "masterpiece"),
    (100, "best quality"),
    (75, "great quality"),
    (50, "good quality"),
    (25, "normal quality"),
    (10, "below average quality"),
    (0, "low quality"),
    (-(1 << 60), "worst quality"),
]


def quality_of(score: int | None) -> str | None:
    """Bucket a danbooru score into its quality word."""
    if score is None:
        return None
    for low, label in QUALITY_THRESHOLDS:
        if score >= low:
            return label
    return None


def open_vault(path: str):
    """Open a KVault with auto-pack on."""
    vault = kv.KVault(path)
    vault.enable_auto_pack()
    return vault


def empty_record() -> dict:
    """Every field a normalized record can have, all absent.

    Built in full, so asking any source for any field gives ``None`` or ``[]``.
    """
    return {
        "id": None,
        "key": None,
        "source": None,
        **{c: [] for c in TAG_CATEGORIES},
        "rating": [],
        "quality": None,
        "width": None,
        "height": None,
        "short": None,
        "long": None,
        "title": None,
        "score": None,
        "year": None,
    }


def _select(tag_conf, threshold: float) -> list[str]:
    """``{tag: confidence}`` -> sorted list of tags above ``threshold``."""
    if not tag_conf:
        return []
    if isinstance(tag_conf, dict):
        return sorted(t for t, c in tag_conf.items() if c is None or c >= threshold)
    return sorted(tag_conf)


def _argmax(tag_conf):
    """The highest-confidence tag, for the categories that are single-valued."""
    if not tag_conf or not isinstance(tag_conf, dict):
        return None
    return max(tag_conf.items(), key=lambda item: item[1])[0]


class _ForkSafeVaults:
    """Lazily opens vault handles and reopens them after a fork."""

    def __init__(self, **paths: str | None) -> None:
        self._paths = paths
        self._handles: dict[str, object] | None = None
        self._pid: int | None = None

    def get(self) -> dict:
        """The handles for this process, reopening them if the PID changed."""
        pid = os.getpid()
        if self._handles is None or self._pid != pid:
            self._handles = {
                name: (open_vault(path) if path else None)
                for name, path in self._paths.items()
            }
            self._pid = pid
        return self._handles


class DanbooruRecords:
    """Ground-truth danbooru metadata, optionally joined with a Qwen caption.

    Args:
        root: dataset root holding ``raw/`` and ``qwen-caption/``.
        with_caption: join ``qwen-caption/danbooru.db`` by image path.
        image_only: keep only posts with a local image (a one-off full scan;
            the result is cached next to ``ids.npy``).
    """

    name = "danbooru"

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        with_caption: bool = True,
        image_only: bool = False,
        ids_path: str | None = None,
    ) -> None:
        self.root = root
        meta_path = os.path.join(root, "raw/danbooru_meta.db")
        cap_path = os.path.join(root, "qwen-caption/danbooru.db")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        self.with_caption = with_caption and os.path.exists(cap_path)
        self._vaults = _ForkSafeVaults(
            meta=meta_path, cap=cap_path if self.with_caption else None
        )
        self.ids = np.load(ids_path or os.path.join(root, "raw/ids.npy"))
        if image_only:
            self.ids = self._image_only_ids()

    def _image_only_ids(self) -> np.ndarray:
        """Post ids that have a local image, from the cache or a one-off scan."""
        cache = os.path.join(self.root, "raw/ids_with_image.npy")
        if os.path.exists(cache):
            return np.load(cache)
        meta = self._vaults.get()["meta"]
        keep = [
            i
            for i in self.ids
            if (meta.get(f"{int(i):08d}".encode()) or {}).get("has_image")
        ]
        ids = np.asarray(keep, dtype=np.int64)
        np.save(cache, ids)
        return ids

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, post_id: int) -> dict | None:
        """Normalized record for one post, or ``None`` if the metadata is absent.

        A missing caption is not missing data: such posts are tag-only.
        """
        handles = self._vaults.get()
        meta_rec = handles["meta"].get(f"{int(post_id):08d}".encode())
        if meta_rec is None:
            return None
        rec = empty_record()
        rec["source"] = "danbooru"
        rec["id"] = meta_rec["id"]
        rec["key"] = f"{int(post_id):08d}"
        for category in TAG_CATEGORIES:
            rec[category] = list(meta_rec.get(category) or [])
        rec["rating"] = [meta_rec["rating"]] if meta_rec.get("rating") else []
        rec["score"] = meta_rec.get("score")
        rec["quality"] = quality_of(meta_rec.get("score"))
        rec["width"] = meta_rec.get("width")
        rec["height"] = meta_rec.get("height")
        rec["year"] = meta_rec.get("year")

        cap = handles["cap"]
        if cap is not None:
            key = f"{DANBOORU_KEY_PREFIX}/{int(post_id)}.webp".encode()
            cap_rec = cap.get(key)
            if cap_rec:
                rec["short"] = cap_rec.get("brief") or None
                rec["long"] = cap_rec.get("description") or None
                rec["title"] = cap_rec.get("title") or None
        return rec

    def __getitem__(self, index: int) -> dict | None:
        return self.get(int(self.ids[index]))


class KeyIndex:
    """Cached key list for a path-keyed vault, so it can be indexed positionally.

    Cached next to the db as a newline-delimited file. See docs/internals/data.md.
    """

    def __init__(self, db_path: str, cache_path: str | None = None) -> None:
        self.db_path = db_path
        self.cache_path = cache_path or (db_path + ".keys")
        self._keys: list[bytes] | None = None

    def build(self) -> list[bytes]:
        """Scan every key once and write the cache, replacing it atomically."""
        vault = open_vault(self.db_path)
        # Explicit limit: keys() is bounded by `limit`, which defaults to 10_000.
        keys = list(vault.keys(limit=len(vault) + 1))
        vault.close()
        tmp = self.cache_path + ".part"
        with open(tmp, "wb") as handle:
            for key in keys:
                handle.write(key + b"\n")
        os.replace(tmp, self.cache_path)
        return keys

    def keys(self) -> list[bytes]:
        """The key list, from the cache if it exists and from a scan if not."""
        if self._keys is None:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as handle:
                    self._keys = handle.read().split(b"\n")[:-1]
            else:
                self._keys = self.build()
        return self._keys


class PathKeyedRecords:
    """``qwen-caption/<name>.db`` joined with ``tagger/<name>_tags.db`` by path."""

    def __init__(
        self,
        name: str,
        root: str = DEFAULT_ROOT,
        threshold: float = 0.35,
    ) -> None:
        if name not in PATH_KEYED:
            raise KeyError(
                f"unknown path-keyed dataset {name!r}; choices: {sorted(PATH_KEYED)}"
            )
        cap_name, tag_name = PATH_KEYED[name]
        self.name = name
        self.threshold = threshold
        self.cap_path = os.path.join(root, "qwen-caption", cap_name)
        tag_path = os.path.join(root, "tagger", tag_name) if tag_name else None
        if not os.path.exists(self.cap_path):
            raise FileNotFoundError(self.cap_path)
        if tag_path and not os.path.exists(tag_path):
            tag_path = None  # fall back to tags embedded in the caption record
        self.tag_path = tag_path
        self._vaults = _ForkSafeVaults(cap=self.cap_path, tag=tag_path)
        self._index = KeyIndex(self.cap_path)

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> list[bytes]:
        return self._index.keys()

    def get(self, key) -> dict | None:
        """Normalized record for one image path; ``None`` only if neither db has it.

        Either half alone is usable, and supports a subset of the tasks.
        """
        handles = self._vaults.get()
        if isinstance(key, str):
            key = key.encode()
        cap_rec = handles["cap"].get(key)
        tag_rec = handles["tag"].get(key) if handles["tag"] is not None else None
        if cap_rec is None and tag_rec is None:
            return None

        rec = empty_record()
        rec["key"] = key.decode("utf-8", "replace")
        rec["id"] = os.path.splitext(os.path.basename(rec["key"]))[0]
        rec["source"] = self.name
        if cap_rec:
            rec["short"] = cap_rec.get("brief") or None
            rec["long"] = cap_rec.get("description") or None
            rec["title"] = cap_rec.get("title") or None

        tags = tag_rec if tag_rec else (cap_rec or {}).get("tags") or {}
        rec["general"] = _select(tags.get("general"), self.threshold)
        rec["character"] = _select(tags.get("character"), self.threshold)
        rating = _argmax(tags.get("rating"))
        rec["rating"] = [rating] if rating else []
        return rec

    def __getitem__(self, index: int) -> dict | None:
        return self.get(self.keys()[index])


def load_records(name: str, root: str = DEFAULT_ROOT, **kwargs):
    """``name`` -> a record source. ``"danbooru"`` is the ground-truth one.

    A ``"<category>/<dataset>"`` name resolves to the general-pretrain corpus.
    """
    if "/" in name:
        from kohakuwullm.data.sources.corpus import CorpusRecords

        return CorpusRecords(name, **kwargs)
    if name == "danbooru":
        return DanbooruRecords(root=root, **kwargs)
    return PathKeyedRecords(name, root=root, **kwargs)
