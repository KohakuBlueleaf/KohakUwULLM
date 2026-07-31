"""Checks for the synthetic packed batches every pipeline benchmark measures on.

These ran as two copies -- one in ``step_throughput.py``, one in
``stage_balance.py`` -- and a benchmark comparing a stage split against a step
time is only meaningful while both see the same documents. The agreement test
below is what stops them drifting apart again.
"""

import torch

from kohakuwullm.bench.core.batches import (
    IGNORE_INDEX,
    make_info,
    make_packs,
    make_tokens,
    trained_tokens,
)


def test_every_microbatch_spends_the_token_budget_exactly():
    infos = make_packs(8192, 1024, 128, 512, torch.device("cpu"), seed=0)
    assert len(infos) == 8
    for info in infos:
        assert int(info.cu_seqlens[-1]) == 1024
        lengths = info.cu_seqlens.diff().tolist()
        # Only the document trimmed at the boundary may fall below `lo`; a
        # short document anywhere else means the fill loop stopped early, and
        # a pipeline stage built for 1024 tokens would then reshape mid-run.
        assert all(128 <= n <= 512 for n in lengths[:-1])
        assert 0 < lengths[-1] <= 512


def test_single_and_multi_microbatch_builders_agree():
    device = torch.device("cpu")
    for seed in (0, 7, 1234):
        solo = make_info(2048, 512, 4096, device, seed=seed)
        first = make_packs(2048, 2048, 512, 4096, device, seed=seed)[0]
        assert solo.cu_seqlens.tolist() == first.cu_seqlens.tolist()


def test_microbatches_differ_from_each_other():
    # One generator across the whole list, not one reseeded per microbatch:
    # reseeding would hand every microbatch identical documents and make a
    # multi-microbatch step measure the same shape N times.
    infos = make_packs(4096, 1024, 64, 512, torch.device("cpu"), seed=3)
    shapes = {tuple(i.cu_seqlens.tolist()) for i in infos}
    assert len(shapes) > 1


def test_only_document_final_positions_are_masked():
    device = torch.device("cpu")
    infos = make_packs(4096, 1024, 128, 512, device, seed=11)
    ids, labels = make_tokens(infos, 65536, device, seed=11)

    assert ids.shape == labels.shape == (4096,)
    masked = (labels == IGNORE_INDEX).nonzero().flatten().tolist()
    ends, offset = [], 0
    for info in infos:
        ends += (info.cu_seqlens[1:].long() + offset - 1).tolist()
        offset += int(info.cu_seqlens[-1])
    assert masked == sorted(ends)
    assert trained_tokens(labels) == 4096 - len(ends)


def test_token_ids_are_not_drawn_from_the_length_stream():
    device = torch.device("cpu")
    infos = make_packs(2048, 1024, 128, 512, device, seed=5)
    same = make_tokens(infos, 65536, device, seed=5)[0]
    other = make_tokens(infos, 65536, device, seed=6)[0]
    assert not torch.equal(same, other)
