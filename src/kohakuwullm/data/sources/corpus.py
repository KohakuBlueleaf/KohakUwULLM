"""Record source over the general-pretrain text corpus in KVault shards.

Each dataset is a set of ``<category>__<name>.NNNN.db`` vaults whose keys are
sequential big-endian int64, so a positional index resolves by arithmetic with
no key cache. Handles reopen on PID change, as every source here must.
See docs/internals/data.md.
"""

import bisect
import json
import os

import kohakuvault as kv

DEFAULT_ROOT = "/Iolite/text-dataset/_vault"


def open_vault(path: str):
    vault = kv.KVault(path)
    vault.enable_auto_pack()
    return vault


def shard_paths(name: str, root: str) -> list[str]:
    """Vault shards of one dataset, in index order."""
    prefix = name.replace("/", "__") + "."
    found = [
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.startswith(prefix) and f.endswith(".db")
    ]
    return sorted(found)


def shard_count(path: str) -> int:
    """Document count of one shard, from its marker rather than a scan."""
    marker = path + ".done"
    if os.path.exists(marker):
        with open(marker) as fh:
            text = fh.read().strip()
            if text.isdigit():
                return int(text)
    vault = open_vault(path)
    try:
        return len(vault)
    finally:
        vault.close()


class CorpusRecords:
    """Positional view over one dataset's vault shards, pinned at construction.

    Shards added to ``root`` after construction are invisible to this view.
    See docs/internals/data.md.

    Args:
        name: ``"<category>/<dataset>"``, e.g. ``"en/nemotron-cc-v2.1-hq"``.
        root: directory holding the shards.
        field: key the rendered record exposes the document text under.
        snapshot: ``[(shard basename, document count)]`` to use verbatim.
    """

    def __init__(
        self,
        name: str,
        root: str = DEFAULT_ROOT,
        field: str = "text",
        snapshot: list | None = None,
    ) -> None:
        self.name = name
        self.field = field
        self.root = root
        if snapshot:
            self.paths = [os.path.join(root, base) for base, _ in snapshot]
            missing = [p for p in self.paths if not os.path.exists(p)]
            if missing:
                raise FileNotFoundError(
                    f"{name!r} snapshot names {len(missing)} absent shard(s), "
                    f"first {missing[0]}"
                )
            counts = [int(n) for _, n in snapshot]
        else:
            self.paths = shard_paths(name, root)
            if not self.paths:
                raise FileNotFoundError(f"no vault shards for {name!r} under {root}")
            counts = [shard_count(p) for p in self.paths]
        bounds = [0]
        for n in counts:
            bounds.append(bounds[-1] + n)
        self.bounds = bounds
        self._handles: list | None = None
        self._pid: int | None = None

    def snapshot(self) -> list:
        """``[(shard basename, document count)]`` for exactly this view."""
        return [
            (os.path.basename(path), self.bounds[i + 1] - self.bounds[i])
            for i, path in enumerate(self.paths)
        ]

    def __len__(self) -> int:
        return self.bounds[-1]

    def _vaults(self) -> list:
        pid = os.getpid()
        if self._handles is None or self._pid != pid:
            self._handles = [open_vault(p) for p in self.paths]
            self._pid = pid
        return self._handles

    def __getitem__(self, index: int) -> dict | None:
        if index < 0:
            index += len(self)
        shard = bisect.bisect_right(self.bounds, index) - 1
        offset = index - self.bounds[shard]
        blob = self._vaults()[shard].get(offset.to_bytes(8, "big"))
        if not blob:
            return None
        return {self.field: blob.decode("utf-8", "replace"), "source": self.name}


def available(root: str = DEFAULT_ROOT) -> dict:
    """``{name: documents}`` from the conversion manifest, or from the shards."""
    manifest = os.path.join(root, "manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as fh:
            return {k: v["documents"] for k, v in json.load(fh).items()}
    names: dict[str, int] = {}
    for f in sorted(os.listdir(root)):
        if not f.endswith(".db"):
            continue
        name = f.rsplit(".", 2)[0].replace("__", "/", 1)
        names[name] = names.get(name, 0) + shard_count(os.path.join(root, f))
    return names


def snapshot_sources(sources: list[dict], root: str = DEFAULT_ROOT) -> dict:
    """``{name: snapshot}`` for every corpus source in a config ``SOURCES``."""
    out = {}
    for spec in sources:
        name = spec["name"]
        if "/" not in name or name in out:
            continue
        out[name] = CorpusRecords(name, root=root).snapshot()
    return out


def write_manifest(sources: list[dict], path: str, root: str = DEFAULT_ROOT) -> dict:
    """Record the shard view a run is using, next to its checkpoints."""
    manifest = {"root": root, "sources": snapshot_sources(sources, root)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def read_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
