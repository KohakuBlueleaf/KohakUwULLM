"""Data package: record sources, prompt rendering, and varlen packing.

``sources`` -> ``renderers`` -> ``packing``, over the ``loader`` plumbing that
registers ``map`` / ``iterative`` / ``ddp`` / ``pipeline``. :func:`build_dataset`
and :func:`build_loader` wire them from a config-shaped spec. See docs/internals/data.md.
"""

import bisect

from kohakuwullm.data.loader.fast_loader import (
    BatchRenderedDataset,
    GroupSampler,
    build_fast_loader,
)
from kohakuwullm.data.loader.iterative import (
    TokenBudgetIterableDataset,
    build_iterative_loader,
    pack_to_budget,
)
from kohakuwullm.data.loader.microbatch import (
    MicroBatchedDataset,
    MicroBatchedStep,
    build_pipeline_loader,
)
from kohakuwullm.data.loader.padded import (
    collate_padded,
    padding_fraction,
    split_padded,
)
from kohakuwullm.data.loader.permute import ShardIndices, held_out, shard_indices
from kohakuwullm.data.loader.resume import (
    ResumableLoader,
    ResumablePackedDataset,
    build_ddp_loader,
    resolve_rank_world,
)
from kohakuwullm.data.packing import (
    IGNORE_INDEX,
    PackedBatch,
    RenderedDataset,
    WeightedConcatDataset,
    collate_packed,
    encode_sample,
    split_packed,
)
from kohakuwullm.data.renderers.chatml import ChatMLRenderer, QAChatMLRenderer
from kohakuwullm.data.renderers.plain import ChatRenderer, PlainRenderer
from kohakuwullm.data.renderers.tipo import SPECIAL_TOKENS, TIPORenderer
from kohakuwullm.data.sources.corpus import CorpusRecords
from kohakuwullm.data.sources.vault import (
    DEFAULT_ROOT,
    PATH_KEYED,
    DanbooruRecords,
    PathKeyedRecords,
    load_records,
)
from kohakuwullm.registry import LOADER, RENDERER, build


def build_dataset(
    sources: list[dict],
    tokenizer,
    renderer="tipo",
    root: str = DEFAULT_ROOT,
    max_length: int = 2048,
    seed: int = 0,
):
    """Build the training dataset from a list of source specs.

    Each entry is ``{"name": "danbooru", "repeat": 3, **source_kwargs}``, where
    the repeat is a re-render rather than a duplicate.
    """
    render_fn = build(renderer, RENDERER)
    datasets, repeats = [], []
    for spec in sources:
        opts = dict(spec)
        name = opts.pop("name")
        repeat = int(opts.pop("repeat", 1))
        records = load_records(name, root=root, **opts)
        datasets.append(
            RenderedDataset(
                records, render_fn, tokenizer, max_length=max_length, seed=seed
            )
        )
        repeats.append(repeat)
    if len(datasets) == 1 and repeats[0] == 1:
        return datasets[0]
    return WeightedConcatDataset(datasets, repeats)


RENDER_KEY = "__render__"


@RENDERER.register("per_source")
class PerSourceRenderer:
    """Render each record with the renderer its source spec named.

    ``build_records`` stamps the resolved callable under ``RENDER_KEY``; records
    from a source that named none fall back to ``default``.
    """

    def __init__(self, default="plain") -> None:
        self.default = build(default, RENDERER)

    def __call__(self, rec: dict, rng=None, **kwargs) -> tuple[str, str]:
        render_fn = rec.get(RENDER_KEY, self.default)
        return render_fn(rec, rng=rng, **kwargs)


class _Fraction:
    """The evenly-strided first ``frac`` of a source, as a view.

    Strided rather than truncated, so a partial pass spans the whole source
    instead of whichever shards happen to sort first.
    """

    def __init__(self, source, frac: float) -> None:
        self.source = source
        self.count = max(1, int(len(source) * frac))
        self.step = len(source) / self.count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        return self.source[min(int(index * self.step), len(self.source) - 1)]


class _ConcatRepeated:
    """Indexable view over several record sources with per-source repeats.

    A view, not a list: the sources are lazy and must stay that way. A repeat
    lists the source again rather than duplicating its records, and a fractional
    repeat takes a strided share of one pass. See docs/internals/data.md.
    """

    def __init__(self, sources: list[dict], root: str, snapshot=None) -> None:
        self.parts = []
        self.renderers = []
        snapshot = snapshot or {}
        bounds = [0]
        for spec in sources:
            opts = dict(spec)
            name = opts.pop("name")
            repeat = float(opts.pop("repeat", 1))
            # Resolved here, never per record: one callable per part.
            render = opts.pop("renderer", None)
            render_fn = build(render, RENDERER) if render else None
            if repeat <= 0:
                continue
            if name in snapshot:
                opts["snapshot"] = snapshot[name]
            # A spec's own root wins: a corpus and a vault do not share one.
            source = load_records(name, root=opts.pop("root", root), **opts)
            whole, frac = int(repeat), repeat - int(repeat)
            for _ in range(whole):
                self.parts.append(source)
                self.renderers.append(render_fn)
                bounds.append(bounds[-1] + len(source))
            if frac > 1e-9:
                part = _Fraction(source, frac)
                self.parts.append(part)
                self.renderers.append(render_fn)
                bounds.append(bounds[-1] + len(part))
        self.bounds = bounds
        self.mixed = any(fn is not None for fn in self.renderers)

    def __len__(self) -> int:
        return self.bounds[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        part = bisect.bisect_right(self.bounds, index) - 1
        record = self.parts[part][index - self.bounds[part]]
        render_fn = self.renderers[part]
        if render_fn is None or record is None:
            return record
        # Copied, not mutated: a source may hand back a cached dict.
        tagged = dict(record)
        tagged[RENDER_KEY] = render_fn
        return tagged


def build_records(sources: list[dict], root: str = DEFAULT_ROOT, snapshot=None):
    """Indexable record view for the iterative loader, honouring repeats."""
    return _ConcatRepeated(sources, root, snapshot)


def build_loader(
    kind,
    sources: list[dict],
    tokenizer,
    renderer="tipo",
    root: str = DEFAULT_ROOT,
    **kwargs,
):
    """Resolve a loader by name and build it against ``sources``.

    ``kind`` may be a registered name or a callable. Not ``build``: these are
    factories to call, not components to instantiate from a spec.
    """
    factory = LOADER.get(kind) if isinstance(kind, str) else kind
    return factory(sources, tokenizer, renderer=renderer, root=root, **kwargs)


@LOADER.register("iterative")
def _iterative_loader(
    sources, tokenizer, renderer="tipo", root=DEFAULT_ROOT, snapshot=None, **kwargs
):
    """Token-budget varlen batches, single process."""
    return build_iterative_loader(
        build_records(sources, root=root, snapshot=snapshot),
        build(renderer, RENDERER),
        tokenizer,
        **kwargs,
    )


@LOADER.register("ddp")
def _ddp_loader(
    sources, tokenizer, renderer="tipo", root=DEFAULT_ROOT, snapshot=None, **kwargs
):
    """Token-budget varlen batches, one disjoint shard per rank, resumable."""
    return build_ddp_loader(
        build_records(sources, root=root, snapshot=snapshot),
        build(renderer, RENDERER),
        tokenizer,
        **kwargs,
    )


@LOADER.register("pipeline")
def _pipeline_loader(
    sources, tokenizer, renderer="tipo", root=DEFAULT_ROOT, snapshot=None, **kwargs
):
    """Fixed-size microbatch steps for a pipeline schedule, resumable."""
    return build_pipeline_loader(
        build_records(sources, root=root, snapshot=snapshot),
        build(renderer, RENDERER),
        tokenizer,
        **kwargs,
    )


@LOADER.register("map")
def _map_loader(
    sources,
    tokenizer,
    renderer="tipo",
    root=DEFAULT_ROOT,
    snapshot=None,
    max_length: int = 2048,
    seed: int = 0,
    layout: str = "packed",
    **kwargs,
):
    """Map-style, fixed sample count per batch; the measured-best online loader."""
    return build_fast_loader(
        build_records(sources, root=root, snapshot=snapshot),
        build(renderer, RENDERER),
        tokenizer,
        max_length=max_length,
        seed=seed,
        layout=layout,
        **kwargs,
    )


__all__ = [
    "build_dataset",
    "build_loader",
    "ChatRenderer",
    "CorpusRecords",
    "PlainRenderer",
    "build_records",
    "ChatMLRenderer",
    "QAChatMLRenderer",
    "PerSourceRenderer",
    "RENDER_KEY",
    "load_records",
    "DanbooruRecords",
    "PathKeyedRecords",
    "PATH_KEYED",
    "DEFAULT_ROOT",
    "TIPORenderer",
    "SPECIAL_TOKENS",
    "RenderedDataset",
    "WeightedConcatDataset",
    "PackedBatch",
    "collate_packed",
    "collate_padded",
    "build_fast_loader",
    "build_iterative_loader",
    "build_ddp_loader",
    "build_pipeline_loader",
    "ResumableLoader",
    "ResumablePackedDataset",
    "resolve_rank_world",
    "MicroBatchedDataset",
    "MicroBatchedStep",
    "TokenBudgetIterableDataset",
    "pack_to_budget",
    "shard_indices",
    "ShardIndices",
    "held_out",
    "BatchRenderedDataset",
    "GroupSampler",
    "split_padded",
    "padding_fraction",
    "encode_sample",
    "split_packed",
    "IGNORE_INDEX",
]
