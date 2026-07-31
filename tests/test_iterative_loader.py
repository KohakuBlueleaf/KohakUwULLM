"""Invariants of the token-budget iterative loader.

The negative cases are the point here. A batcher that drops the tail of a shard,
duplicates a document across two workers, or lets one long document blow a hole
in the budget still trains -- it just quietly changes the effective batch size or
the number of epochs, and neither shows up in a loss curve. The same goes double
for a resume: one that replays two hundred documents, or skips them, trains
perfectly happily.
"""

import random

import numpy as np
import pytest
import torch
import torch.utils.data as data

from kohakuwullm.data import build_loader, build_records
from kohakuwullm.data.loader.iterative import (
    TokenBudgetIterableDataset,
    _passthrough,
    pack_to_budget,
    shard_indices,
)
from kohakuwullm.data.loader.microbatch import (
    MicroBatchedDataset,
    build_pipeline_loader,
)
from kohakuwullm.data.loader.resume import build_ddp_loader, resolve_rank_world
from kohakuwullm.data.packing import IGNORE_INDEX, RenderedDataset
from kohakuwullm.registry import LOADER

K = 4096
M = 128
CTX_MAX = 1024
# The pipeline path is exercised at a small budget so a test epoch holds enough
# steps to interrupt one; the shape of the claim does not depend on the size.
MICRO_K = 512
MICRO_M = 64
MICRO_CTX = 256


def _sample(length: int, tag: int) -> dict:
    return {"input_ids": [tag] * length, "labels": [tag] * length}


def _totals(groups) -> list[int]:
    return [sum(len(s["input_ids"]) for s in group) for group in groups]


def _tags(groups) -> list[list[int]]:
    return [[s["input_ids"][0] for s in group] for group in groups]


def _skewed_stream(n: int, seed: int = 0) -> list[dict]:
    """Mostly short documents with a heavy tail, like the rendered corpus."""
    rng = random.Random(seed)
    lengths = [
        rng.randint(M + 1, CTX_MAX) if rng.random() < 0.05 else rng.randint(8, M)
        for _ in range(n)
    ]
    return [_sample(length, i) for i, length in enumerate(lengths)]


def test_budget_is_never_exceeded_and_deficit_is_bounded_by_m():
    totals = _totals(list(pack_to_budget(_skewed_stream(4000), K, M)))

    assert all(total <= K for total in totals)
    # Every batch but the trailing one ends on a document of at most `m` and so
    # cannot be more than `m` short. This is the whole claim of the design.
    assert all(K - M < total <= K for total in totals[:-1])


def test_retry_is_what_bounds_the_deficit():
    """Without the long-document retry the deficit is bounded only by ctx_max."""
    samples = _skewed_stream(4000)
    with_retry = _totals(list(pack_to_budget(samples, K, M)))[:-1]
    # m >= k can never fire the retry, which reduces this to a plain greedy fill.
    greedy = _totals(list(pack_to_budget(samples, K, K)))[:-1]

    assert min(with_retry) > K - M
    assert min(greedy) <= K - M
    assert np.std(with_retry) < np.std(greedy)


def test_no_sample_is_dropped_or_duplicated():
    samples = _skewed_stream(3000)
    groups = list(pack_to_budget(samples, K, M))

    assert sorted(tag for group in _tags(groups) for tag in group) == list(
        range(len(samples))
    )


def test_queued_documents_open_the_next_batch_in_draw_order():
    # The first 600 does not fit and is long, so the retry queues it and keeps
    # drawing; the second is queued behind it. Both open the next batch.
    samples = [_sample(2000, 0), _sample(2000, 1), _sample(600, 2), _sample(600, 3)]

    assert _tags(list(pack_to_budget(samples, K, M))) == [[0, 1], [2, 3]]


def test_document_longer_than_the_budget_is_emitted_alone():
    samples = [_sample(64, 0), _sample(K + 500, 1), _sample(64, 2)]
    groups = list(pack_to_budget(samples, K, M))

    assert max(_totals(groups)) == K + 500
    assert sorted(tag for group in _tags(groups) for tag in group) == [0, 1, 2]


def test_a_stream_of_only_long_documents_terminates():
    # Every rejection is long, so the retry never finds its exit and only
    # `max_retry` ends the batch. The deficit bound is forfeited here.
    samples = [_sample(CTX_MAX, i) for i in range(200)]
    groups = list(pack_to_budget(samples, K, M, max_retry=4))

    assert sorted(tag for group in _tags(groups) for tag in group) == list(range(200))
    assert all(total <= K for total in _totals(groups))


def test_drop_last_drops_at_most_the_under_budget_tail():
    samples = _skewed_stream(2000)
    kept = list(pack_to_budget(samples, K, M))
    dropped = list(pack_to_budget(samples, K, M, drop_last=True))

    assert len(dropped) == len(kept) - (1 if _totals(kept)[-1] <= K - M else 0)
    assert all(total > K - M for total in _totals(dropped))


class _FakeRecords:
    def __init__(self, n: int, missing: set[int] | None = None) -> None:
        self.n = n
        self.missing = missing or set()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        return None if index in self.missing else {"id": index}


class _FakeRenderer:
    def __call__(self, rec, rng=None):
        # Every token spells the record id, so a dropped or duplicated record is
        # visible in the emitted token ids rather than only in a count.
        return "", " ".join([str(rec["id"])] * (5 + rec["id"] % 400))


class _FakeTokenizer:
    """Whitespace tokenizer mapping a word to the integer it spells."""

    bos_token_id = None
    eos_token_id = None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(word) for word in text.split()]}


def _iter_dataset(n: int = 2000, missing: set[int] | None = None, **kwargs):
    options = dict(k=K, m=M, ctx_max=CTX_MAX, seed=7)
    options.update(kwargs)
    return TokenBudgetIterableDataset(
        _FakeRecords(n, missing), _FakeRenderer(), _FakeTokenizer(), **options
    )


def _doc_ids(batches) -> list[int]:
    """The record id of every document emitted, in order."""
    out = []
    for batch in batches:
        starts = batch.seq_info.cu_seqlens[:-1].tolist()
        out.extend(int(batch.tokens[start]) for start in starts)
    return out


def test_truncation_at_ctx_max():
    lengths = [
        length
        for batch in _iter_dataset(ctx_max=64)
        for length in batch.seq_info.seqlens.tolist()
    ]

    assert max(lengths) == 64


def test_ctx_max_above_the_budget_is_rejected():
    with pytest.raises(ValueError):
        _iter_dataset(ctx_max=K + 1)


def test_batches_are_packed_and_labels_never_cross_documents():
    batch = next(iter(_iter_dataset(n=500)))

    assert batch.seq_info.packed
    assert batch.num_tokens == int(batch.seq_info.cu_seqlens[-1])
    assert K - M < batch.num_tokens <= K
    # The last position of every document predicts nothing.
    ends = (batch.seq_info.cu_seqlens[1:] - 1).tolist()
    assert all(batch.labels[end] == IGNORE_INDEX for end in ends)


def test_missing_records_are_skipped_not_emitted_empty():
    dataset = _iter_dataset(n=500, missing=set(range(0, 500, 2)))
    ids = _doc_ids(dataset)

    assert sorted(ids) == list(range(1, 500, 2))


def _loader_doc_ids(num_workers: int, n: int = 3000) -> list[int]:
    loader = data.DataLoader(
        _iter_dataset(n=n),
        batch_size=None,
        num_workers=num_workers,
        collate_fn=_passthrough,
    )
    return _doc_ids(loader)


def test_worker_sharding_yields_disjoint_data():
    # An IterableDataset is copied wholesale into every worker: one that does
    # not shard emits its whole shard N times over.
    assert sorted(_loader_doc_ids(4)) == list(range(3000))


def test_shards_are_disjoint_and_complete():
    shards = [shard_indices(1000, i, 4, seed=3, epoch=0) for i in range(4)]

    assert sorted(np.concatenate(shards).tolist()) == list(range(1000))
    assert len({len(shard) for shard in shards}) == 1


def test_shard_order_is_seeded_and_epoch_dependent():
    a = shard_indices(1000, 0, 2, seed=3, epoch=0)
    b = shard_indices(1000, 0, 2, seed=3, epoch=0)
    c = shard_indices(1000, 0, 2, seed=3, epoch=1)
    d = shard_indices(1000, 0, 2, seed=4, epoch=0)

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)


def test_repeated_iteration_advances_the_epoch():
    dataset = _iter_dataset(n=800)
    first = _doc_ids(dataset)
    second = _doc_ids(dataset)
    dataset.set_epoch(0)
    reset = _doc_ids(dataset)

    # A persistent worker never sees set_epoch, so a second pass over the same
    # object must not replay the first pass.
    assert first != second
    assert sorted(first) == sorted(second)
    assert first == reset


def test_distributed_ranks_see_disjoint_data():
    ids = [_doc_ids(_iter_dataset(n=1200, rank=r, world_size=3)) for r in range(3)]

    assert sorted(i for shard in ids for i in shard) == list(range(1200))


def test_ranks_disagree_on_batch_count_unless_it_is_pinned():
    # Shards hold equal document counts but not equal token counts, so the
    # number of batches is data-dependent. Under DDP the short rank leaves the
    # collective early and the run hangs, which is why batches_per_epoch exists.
    free = [
        sum(1 for _ in _iter_dataset(n=1200, rank=r, world_size=3)) for r in range(3)
    ]
    pinned = [
        sum(
            1
            for _ in _iter_dataset(
                n=1200, rank=r, world_size=3, batches_per_epoch=min(free)
            )
        )
        for r in range(3)
    ]

    assert len(set(free)) > 1
    assert pinned == [min(free)] * 3


def test_rank_and_worker_sharding_compose():
    # The two shardings are nested: shard `rank * num_workers + worker_id` of
    # `world_size * num_workers`. Get the product wrong -- shard by rank only, or
    # by worker only -- and every document is emitted once per worker or once per
    # rank, which trains fine and quietly multiplies the epoch.
    ids = []
    for rank in range(2):
        loader = data.DataLoader(
            _iter_dataset(n=2400, rank=rank, world_size=2),
            batch_size=None,
            num_workers=3,
            collate_fn=_passthrough,
        )
        ids.extend(_doc_ids(loader))

    assert sorted(ids) == list(range(2400))


def test_persistent_workers_advance_the_epoch_themselves():
    dataset = _iter_dataset(n=1500)
    loader = data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        collate_fn=_passthrough,
        persistent_workers=True,
    )
    first = _doc_ids(loader)
    second = _doc_ids(loader)

    # set_epoch cannot reach a live worker, so the worker's own pass counter is
    # the only thing standing between a second epoch and a replay of the first.
    assert sorted(first) == sorted(second) == list(range(1500))
    assert first != second


# -------------------------------------------------------------------- resume


class _CountingStream:
    """Records how many samples the packer has drawn, which is the one thing a
    caller cannot see from the outside."""

    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples
        self.drawn = 0

    def __iter__(self):
        for sample in self.samples:
            self.drawn += 1
            yield sample


def _resume_from(samples: list[dict], groups: list[list[dict]], drawn: int):
    """The packer's state after ``groups``: everything drawn but not emitted."""
    emitted = {tag for group in _tags(groups) for tag in group}
    carry = [s for s in samples[:drawn] if s["input_ids"][0] not in emitted]
    return carry + samples[drawn:]


INTERLEAVED = [3900, 500, 100, 200, 50, 100, 300, 300, 400, 700]


def test_pack_to_budget_resumes_from_the_documents_it_still_holds():
    samples = _skewed_stream(2000)
    uninterrupted = _tags(list(pack_to_budget(samples, K, M)))

    source = _CountingStream(samples)
    stream = pack_to_budget(source, K, M)
    before = [next(stream) for _ in range(40)]
    resumed = _tags(
        list(pack_to_budget(_resume_from(samples, before, source.drawn), K, M))
    )

    # Drawn-minus-emitted is the packer's whole state, which is what the loader's
    # position records instead of serialising tokenized documents.
    assert _tags(before) == uninterrupted[:40]
    assert resumed == uninterrupted[40:]


def test_a_carry_is_not_simply_the_tail_of_what_was_drawn():
    # 1 and 3 are long enough to fire the retry, so the batch carries them and
    # goes on to emit 4. Any cursor-shaped resume gets this case wrong; the
    # loader records the documents themselves for exactly this reason.
    samples = [_sample(length, i) for i, length in enumerate(INTERLEAVED)]
    uninterrupted = _tags(list(pack_to_budget(samples, K, M)))

    source = _CountingStream(samples)
    stream = pack_to_budget(source, K, M)
    before = [next(stream)]
    resumed = _tags(
        list(pack_to_budget(_resume_from(samples, before, source.drawn), K, M))
    )

    assert _tags(before) == [[0, 2, 4]] == uninterrupted[:1]
    assert resumed == uninterrupted[1:]


def test_a_resume_that_forgets_the_carry_loses_those_documents():
    samples = _skewed_stream(2000)
    uninterrupted = _tags(list(pack_to_budget(samples, K, M)))

    source = _CountingStream(samples)
    stream = pack_to_budget(source, K, M)
    before = [next(stream) for _ in range(5)]
    # Resuming from the read cursor alone -- the obvious implementation -- drops
    # every document the packer had drawn and not yet emitted.
    naive = _tags(list(pack_to_budget(samples[source.drawn :], K, M)))

    lost = {tag for group in uninterrupted for tag in group} - {
        tag for group in _tags(before) + naive for tag in group
    }
    assert lost
    assert naive != uninterrupted[5:]


def test_a_resume_from_the_last_emitted_document_skips_the_ones_behind_it():
    samples = [_sample(length, i) for i, length in enumerate(INTERLEAVED)]
    stream = pack_to_budget(_CountingStream(samples), K, M)
    before = [next(stream)]
    # The furthest document emitted, the other tempting cursor. It sits *ahead*
    # of part of the carry, so resuming there drops 1 and 3 for good.
    last = max(tag for group in _tags(before) for tag in group)
    after = _tags(list(pack_to_budget(samples[last + 1 :], K, M)))

    seen = sorted(tag for group in _tags(before) + after for tag in group)
    assert seen != list(range(len(samples)))
    assert 1 not in seen and 3 not in seen


def _ddp_loader(n: int = 2000, num_workers: int = 0, **kwargs):
    options = dict(
        k=K,
        m=M,
        ctx_max=CTX_MAX,
        seed=7,
        rank=0,
        world_size=1,
        num_workers=num_workers,
        pin_memory=False,
    )
    options.update(kwargs)
    return build_ddp_loader(
        _FakeRecords(n), _FakeRenderer(), _FakeTokenizer(), **options
    )


def _prefetched(loader):
    """Lightning's fetcher for a loader with no ``__len__``: one batch ahead.

    ``_PrefetchDataFetcher`` pre-pulls one batch so it can see the end of the
    iterator coming, and refills after every pop -- so while the trainer works on
    a batch, the one after it has already been drawn.
    """
    iterator = iter(loader)
    queue = [next(iterator, None)]
    while queue[0] is not None:
        batch = queue.pop(0)
        queue.append(next(iterator, None))
        yield batch


def _consumer(loader):
    """Iterate the way the trainer will: pinned loaders are not prefetched."""
    return iter(loader) if loader.steps_per_epoch is not None else _prefetched(loader)


def _run_and_checkpoint(loader, batches: int) -> tuple[list[int], dict]:
    """Consume ``batches`` batches, then checkpoint as a crash would."""
    ids: list[int] = []
    for seen, batch in enumerate(_consumer(loader), start=1):
        ids.extend(_doc_ids([batch]))
        if seen == batches:
            break
    return ids, loader.state_dict()


@pytest.mark.parametrize(
    "num_workers, batches", [(0, 4), (2, 4), (2, 5), (3, 4), (3, 8)]
)
def test_a_resumed_run_continues_identically(num_workers, batches):
    # batches_per_epoch pinned, as DDP needs it for lockstep anyway: a shard that
    # runs dry first drops out of the round-robin, and past that point delivery
    # order is no longer a rotation of arrival order.
    def loader():
        return _ddp_loader(num_workers=num_workers, batches_per_epoch=20)

    uninterrupted = _doc_ids(loader())

    before, state = _run_and_checkpoint(loader(), batches)
    resumed = loader()
    resumed.load_state_dict(state)
    after = _doc_ids(resumed)

    # Document for document, in order -- not merely the same set. The cases where
    # `batches` is not a multiple of the worker count are the interesting ones:
    # the restarted round-robin has to be rotated back into phase.
    assert len(before) > 0 and len(after) > 0
    assert before + after == uninterrupted


@pytest.mark.parametrize("batches", [3, 4])
def test_a_resumed_run_is_exact_when_the_consumer_prefetches(batches):
    # Unpinned, so the loader has no __len__ and the trainer's fetcher keeps one
    # batch in flight. One worker, so the cycle rotation is not in play and the
    # only claim under test is the prefetch lag: committing on hand-off here would
    # describe a batch still queued, and the resume would skip it.
    uninterrupted = _doc_ids(_prefetched(_ddp_loader(n=800)))

    before, state = _run_and_checkpoint(_ddp_loader(n=800), batches)
    resumed = _ddp_loader(n=800)
    resumed.load_state_dict(state)

    assert before + _doc_ids(_prefetched(resumed)) == uninterrupted


def test_load_state_dict_is_idempotent():
    # Lightning restores the loader state and the trainer restores it again, so
    # applying it twice must not move the resume point.
    _, state = _run_and_checkpoint(_ddp_loader(n=800, batches_per_epoch=20), 3)

    once = _ddp_loader(n=800, batches_per_epoch=20)
    once.load_state_dict(state)
    twice = _ddp_loader(n=800, batches_per_epoch=20)
    twice.load_state_dict(state)
    twice.load_state_dict(state)

    assert _doc_ids(_consumer(twice)) == _doc_ids(_consumer(once))


def test_the_position_survives_a_weights_only_load(tmp_path):
    _, state = _run_and_checkpoint(_ddp_loader(n=800, num_workers=2), 2)
    path = tmp_path / "state.pt"
    torch.save(state, path)

    # Lightning's checkpoint loader passes weights_only=None, which is True on
    # torch 2.13: an ndarray or an rng object in the position would refuse to
    # load, and the failure would land on the resume rather than on the save.
    assert torch.load(path, weights_only=True) == state


@pytest.mark.parametrize("num_workers", [0, 2, 3])
def test_a_resumed_run_neither_replays_nor_skips(num_workers):
    # Unpinned, so the shards run dry at different times and the tail is not a
    # clean rotation. Order is what that costs; completeness is not. Unpinned also
    # means the consumer prefetches, which the loader's one-behind commit answers.
    before, state = _run_and_checkpoint(_ddp_loader(num_workers=num_workers), 3)
    resumed = _ddp_loader(num_workers=num_workers)
    resumed.load_state_dict(state)
    seen = before + _doc_ids(_prefetched(resumed))

    assert sorted(seen) == list(range(2000))
    assert len(seen) == len(set(seen))


def test_resume_restores_each_rank_into_its_own_shard():
    world = 2

    def loader(rank):
        return _ddp_loader(n=1200, rank=rank, world_size=world, batches_per_epoch=10)

    uninterrupted = [_doc_ids(loader(rank)) for rank in range(world)]

    befores, states = [], []
    for rank in range(world):
        before, state = _run_and_checkpoint(loader(rank), 3)
        befores.append(before)
        states.append(state)
    # state_dict all-gathers because only rank 0's checkpoint is written; with no
    # process group up each rank fills its own slot, so they are merged here.
    merged = {**states[0], "ranks": [s["ranks"][r] for r, s in enumerate(states)]}

    for rank in range(world):
        resumed = loader(rank)
        resumed.load_state_dict(merged)
        # Restoring rank 0's cursor on rank 1 would land at a plausible offset in
        # the wrong shard, which is the failure with no symptom.
        assert befores[rank] + _doc_ids(resumed) == uninterrupted[rank]


def test_a_resumed_run_does_not_re_resume_the_next_epoch():
    _, state = _run_and_checkpoint(_ddp_loader(n=800), 3)
    resumed = _ddp_loader(n=800)
    resumed.load_state_dict(state)
    tail = _doc_ids(resumed)
    second = _doc_ids(resumed)

    # The position is popped as the iterator starts, so the epoch after a
    # resumed one is a full pass in its own order -- not the resume point again.
    assert len(tail) < 800
    assert sorted(second) == list(range(800))


@pytest.mark.parametrize("batches_per_epoch", [None, 20])
def test_the_position_is_the_last_batch_consumed_not_produced(batches_per_epoch):
    loader = _ddp_loader(n=800, num_workers=2, batches_per_epoch=batches_per_epoch)
    before, state = _run_and_checkpoint(loader, 2)
    positions = state["ranks"][0]

    # Two consumed batches with two workers means one each, whatever the workers
    # and the fetcher had run ahead to produce. Recording on arrival instead would
    # count those, and the resume would skip them -- which is why the loader
    # commits one behind exactly when the consumer prefetches.
    assert sorted(positions) == [0, 1]
    assert [positions[shard]["emitted"] for shard in (0, 1)] == [1, 1]
    assert len(before) > 0


def test_resume_rejects_a_changed_rank_or_worker_count():
    _, state = _run_and_checkpoint(_ddp_loader(n=800, num_workers=2), 2)

    with pytest.raises(ValueError, match="workers"):
        _ddp_loader(n=800, num_workers=3).load_state_dict(state)
    with pytest.raises(ValueError, match="rank run"):
        _ddp_loader(
            n=800, num_workers=2, world_size=2, batches_per_epoch=2
        ).load_state_dict(state)


def test_a_shard_count_that_moved_is_caught_by_the_dataset_too():
    # Last line of defence: the loader checks what it knows about, but the shard
    # that reads the position is the one that can compare it against the shuffle
    # it is actually striding.
    dataset = _iter_dataset(n=800)
    dataset.set_resume_state({0: {"shards": 4, "pass": 0, "cursor": 0, "carry": ()}})

    with pytest.raises(ValueError, match="shards"):
        next(iter(dataset))


def test_state_dict_is_rank_local_and_says_which_rank_wrote_it():
    _, state = _run_and_checkpoint(_ddp_loader(n=800), 2)

    # Under DDP only rank 0's checkpoint is written, so a consumer has to be able
    # to tell a rank-local cursor from a global one before trusting it.
    assert state["world_size"] == 1
    assert state["num_workers"] == 0
    assert list(state["ranks"][0]) == [0]
    assert state["ranks"][0][0]["emitted"] == 2


def test_length_is_per_rank_and_only_exists_once_the_count_is_pinned():
    pinned = _ddp_loader(n=800, batches_per_epoch=3)

    assert len(pinned) == 3
    assert sum(1 for _ in pinned) == 3
    with pytest.raises(TypeError):
        len(_ddp_loader(n=800))


def test_ddp_loader_warns_when_the_batch_count_is_not_pinned():
    # Silence is the dangerous outcome here: the ranks disagree on when the epoch
    # ends and the run hangs in the next collective, with nothing in the log.
    with pytest.warns(UserWarning, match="batches_per_epoch"):
        _ddp_loader(n=200, rank=0, world_size=2)


def test_resolve_rank_world_prefers_explicit_then_the_launcher(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert resolve_rank_world() == (2, 4)
    assert resolve_rank_world(rank=0, world_size=1) == (0, 1)

    monkeypatch.delenv("RANK")
    monkeypatch.delenv("WORLD_SIZE")
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert resolve_rank_world() == (3, 1)


# ------------------------------------------------------------------ pipeline


def _pipeline_dataset(n: int = 3000, num_microbatches: int = 4, **kwargs):
    options = dict(k=MICRO_K, m=MICRO_M, ctx_max=MICRO_CTX, seed=7)
    options.update(kwargs)
    return MicroBatchedDataset(
        _FakeRecords(n),
        _FakeRenderer(),
        _FakeTokenizer(),
        num_microbatches=num_microbatches,
        **options,
    )


def _unpadded_groups(n: int, **kwargs):
    """The same shard through the plain packer, for comparison."""
    return _iter_dataset(n=n, k=MICRO_K, m=MICRO_M, ctx_max=MICRO_CTX, **kwargs)


def _documents(batch) -> list[tuple[int, int]]:
    """``(record id, length)`` per document, dropping the masked filler.

    A filler is the document whose labels are *entirely* ignored; a real one
    always has at least one target, since only its last position predicts
    nothing.
    """
    out = []
    for start, length in zip(
        batch.seq_info.cu_seqlens[:-1].tolist(), batch.seq_info.seqlens.tolist()
    ):
        labels = batch.labels[start : start + length]
        if bool((labels == IGNORE_INDEX).all()):
            continue
        out.append((int(batch.tokens[start]), length))
    return out


def test_every_pipeline_microbatch_carries_exactly_k_tokens():
    steps = [step for step, _ in _pipeline_dataset()]

    assert len(steps) > 2
    for step in steps:
        # PipelineStage declares the boundary activation's shape once, at build
        # time, so a microbatch of any other width is a shape error at best and a
        # deadlock at worst.
        assert step.num_microbatches == 4
        assert step.tokens.numel() == step.labels.numel() == 4 * MICRO_K
        assert [chunk.num_tokens for chunk in step.microbatches] == [MICRO_K] * 4
        assert [int(info.cu_seqlens[-1]) for info in step.seq_infos] == [MICRO_K] * 4


def test_pipeline_pads_the_tail_rather_than_splitting_a_document():
    steps = [step for step, _ in _pipeline_dataset(n=1500, num_microbatches=1)]
    plain = list(_unpadded_groups(1500))

    assert len(steps) > 2
    padded = 0
    for step, batch in zip(steps, plain):
        chunk = step.microbatches[0]
        # Documents intact and in the same order as the unpadded packer emits
        # them: a TIPO example is a prompt and its completion, so half of one is
        # a different training example.
        assert _documents(chunk) == _documents(batch)
        padded += chunk.num_tokens > batch.num_tokens
    assert padded


def test_pipeline_pad_is_masked_so_the_loss_scale_is_untouched():
    steps = [step for step, _ in _pipeline_dataset(n=1500)]

    for step in steps:
        for chunk in step.microbatches:
            assert chunk.num_trained == int((chunk.labels != IGNORE_INDEX).sum())
        assert step.num_trained == sum(c.num_trained for c in step.microbatches)
        # The filler tokens carry no target at all, so trained tokens stay below
        # the padded width and the trainer's denominator never counts pad.
        assert step.num_trained < step.num_tokens


def test_pipeline_microbatches_are_views_of_the_step_tensors():
    step, _ = next(iter(_pipeline_dataset(n=800)))
    stride = MICRO_K * step.tokens.element_size()

    # The flat tensors are what a torch pipeline schedule chunks along dim 0, and
    # the per-microbatch views are the same storage: neither costs a copy.
    assert [chunk.tokens.data_ptr() for chunk in step.microbatches] == [
        step.tokens.data_ptr() + i * stride for i in range(4)
    ]


def test_a_step_survives_a_device_move():
    step, _ = next(iter(_pipeline_dataset(n=800)))
    moved = step.to(torch.device("cpu"))

    assert torch.equal(moved.tokens, step.tokens)
    assert torch.equal(moved.labels, step.labels)
    assert (moved.num_tokens, moved.num_trained) == (step.num_tokens, step.num_trained)
    assert [int(info.cu_seqlens[-1]) for info in moved.seq_infos] == [MICRO_K] * 4


def test_pipeline_drops_a_trailing_partial_step():
    groups = sum(1 for _ in _unpadded_groups(1200))
    steps = sum(1 for _ in _pipeline_dataset(n=1200, num_microbatches=8))

    # A short step would leave the schedule waiting on microbatches that never
    # arrive, so the remainder is dropped rather than emitted narrow.
    assert groups % 8 != 0
    assert steps == groups // 8


def test_pipeline_batches_per_epoch_counts_steps_not_microbatches():
    free = sum(1 for _ in _pipeline_dataset(n=1200))
    pinned = sum(1 for _ in _pipeline_dataset(n=1200, batches_per_epoch=3))

    assert free > 3
    assert pinned == 3


def _step_documents(source) -> list[tuple[int, int]]:
    return [
        doc
        for step in source
        for chunk in step.microbatches
        for doc in _documents(chunk)
    ]


def test_pipeline_steps_hold_no_document_twice():
    docs = _step_documents(step for step, _ in _pipeline_dataset(n=1200))

    assert len({doc[0] for doc in docs}) == len(docs)


@pytest.mark.parametrize("batches_per_epoch", [None, 8])
def test_a_resumed_pipeline_run_continues_identically(batches_per_epoch):
    def loader():
        return build_pipeline_loader(
            _FakeRecords(1500),
            _FakeRenderer(),
            _FakeTokenizer(),
            k=MICRO_K,
            num_microbatches=4,
            m=MICRO_M,
            ctx_max=MICRO_CTX,
            seed=7,
            num_workers=0,
            pin_memory=False,
            batches_per_epoch=batches_per_epoch,
        )

    uninterrupted = _step_documents(_consumer(loader()))

    live = loader()
    before = []
    for seen, step in enumerate(_consumer(live), start=1):
        before.extend(_step_documents([step]))
        if seen == 3:
            break
    state = live.state_dict()

    resumed = loader()
    resumed.load_state_dict(state)

    # Document for document across the interruption, whether the step count is
    # pinned (no prefetch) or not (one step in flight the whole time).
    assert before + _step_documents(_consumer(resumed)) == uninterrupted


# ------------------------------------------------------------------ registry


def test_loader_registry_exposes_every_loader():
    assert set(LOADER.keys()) >= {"iterative", "map", "ddp", "pipeline"}


def test_map_loader_builds_against_a_record_source(monkeypatch):
    monkeypatch.setattr(
        "kohakuwullm.data.load_records",
        lambda name, root=None, **kw: [{"id": i} for i in range(64)],
    )

    loader = build_loader(
        "map",
        [{"name": "x"}],
        _FakeTokenizer(),
        renderer=_FakeRenderer(),
        batch_size=4,
        num_workers=0,
    )

    # It takes records, a renderer and a tokenizer -- it builds the RenderedDataset
    # itself, so handing it a dataset raises before a run ever starts.
    assert isinstance(loader.dataset, RenderedDataset)
    assert len(loader.dataset) == 64


def test_build_loader_accepts_a_callable_so_a_config_can_supply_its_own():
    seen = {}

    def custom(sources, tokenizer, renderer=None, root=None, **kwargs):
        seen.update(sources=sources, renderer=renderer, root=root, kwargs=kwargs)
        return "loader"

    result = build_loader(
        custom, [{"name": "x"}], _FakeTokenizer(), renderer="tipo", root="/r", k=8
    )

    assert result == "loader"
    assert seen["sources"] == [{"name": "x"}]
    assert seen["root"] == "/r"
    assert seen["kwargs"] == {"k": 8}


def test_build_loader_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        build_loader("no-such-loader", [{"name": "x"}], _FakeTokenizer())


def test_build_records_repeats_a_source_without_dropping_any(monkeypatch):
    monkeypatch.setattr(
        "kohakuwullm.data.load_records", lambda name, root=None, **kw: [name] * 3
    )

    records = build_records([{"name": "a", "repeat": 2}, {"name": "b"}], root="/r")

    # A repeat lists the source again rather than duplicating in place, so the
    # iterative loader's own shuffle decides how repeats are spaced. Indexed
    # rather than compared as a list: the view never materializes its sources,
    # which is what keeps a lazy database-backed source lazy.
    assert len(records) == 9
    assert [records[i] for i in range(9)] == ["a"] * 3 + ["a"] * 3 + ["b"] * 3
    assert records[-1] == "b"


def test_build_records_leaves_the_source_spec_untouched(monkeypatch):
    monkeypatch.setattr(
        "kohakuwullm.data.load_records", lambda name, root=None, **kw: [name]
    )
    sources = [{"name": "a", "repeat": 2}]

    build_records(sources, root="/r")

    # `name` and `repeat` are popped from a copy; mutating the caller's spec
    # would break a second call with the same config.
    assert sources == [{"name": "a", "repeat": 2}]
