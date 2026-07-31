"""Packing and padding: the label rules, and that the two layouts agree.

These test ``data/packing.py`` and ``data/padded.py`` rather than the model, and
lived in ``test_models.py`` only because the loss they produce is what the model
consumes.

The label rules are where a silent bug lives. Labels are shifted *inside* each
document, so the last position of one never predicts the first token of its
neighbour; the user half and the BOS are context, not targets. Getting either
wrong trains a model that works, on a slightly different objective than intended.
"""

import pytest
import torch
from model_fixtures import tiny_config

from kohakuwullm import LMBackbone
from kohakuwullm.data.loader.padded import collate_padded, split_padded
from kohakuwullm.data.packing import IGNORE_INDEX, collate_packed, split_packed

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------- packing


def test_collate_shifts_labels_within_documents():
    """The last position of a document must not predict the next document."""
    samples = [
        {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
        {"input_ids": [5, 6, 7], "labels": [IGNORE_INDEX, 6, 7]},
    ]
    batch = collate_packed(samples)
    assert batch.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert batch.labels.tolist() == [2, 3, 4, IGNORE_INDEX, 6, 7, IGNORE_INDEX]
    assert batch.seq_info.cu_seqlens.tolist() == [0, 4, 7]
    assert batch.num_tokens == 7
    assert batch.num_trained == 5


def test_collate_position_ids_restart_per_document():
    samples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [4, 5]},
    ]
    batch = collate_packed(samples)
    assert batch.seq_info.position_ids.tolist() == [0, 1, 2, 0, 1]


def test_collate_drops_degenerate_samples():
    samples = [
        {"input_ids": [], "labels": []},
        {"input_ids": [9], "labels": [9]},
        {"input_ids": [1, 2], "labels": [1, 2]},
    ]
    batch = collate_packed(samples)
    assert batch.seq_info.num_seqs == 1


def test_split_packed_preserves_tokens_and_boundaries():
    samples = [
        {"input_ids": list(range(n)), "labels": list(range(n))}
        for n in (10, 40, 25, 5, 60)
    ]
    batch = collate_packed(samples)
    chunks = split_packed(batch, 3)
    assert sum(c.num_tokens for c in chunks) == batch.num_tokens
    assert sum(c.num_trained for c in chunks) == batch.num_trained
    # Every chunk is itself a valid packed batch: no document was cut in half.
    for chunk in chunks:
        lengths = chunk.seq_info.seqlens.tolist()
        assert sum(lengths) == chunk.num_tokens
        assert all(n > 0 for n in lengths)


def test_pad_to_multiple_adds_only_masked_tokens():
    samples = [{"input_ids": list(range(10)), "labels": list(range(10))}]
    batch = collate_packed(samples, pad_to_multiple=64)
    assert batch.num_tokens == 64
    assert batch.num_trained == 9  # the filler contributes no gradient


# --------------------------------------------------------------------- layouts


def test_padded_collate_shapes_and_masking():
    samples = [
        {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
        {"input_ids": [5, 6], "labels": [IGNORE_INDEX, 6]},
    ]
    batch = collate_padded(samples, pad_to_multiple=4)
    assert batch.tokens.shape == (2, 4)
    assert not batch.seq_info.packed
    # Row 1 has 2 real tokens; position 0 predicts token 1, position 1 predicts
    # nothing, and the two pad positions must contribute no gradient.
    assert batch.labels[1].tolist() == [6, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    # num_tokens counts computed positions (padding included), so a padded run's
    # throughput cannot be mistaken for a packed one's.
    assert batch.num_tokens == 8


@cuda_only
def test_padded_and_packed_give_the_same_loss():
    """The two layouts must be numerically equivalent, differing only in waste."""
    torch.manual_seed(0)
    samples = [
        {"input_ids": list(range(2, n + 2)), "labels": list(range(2, n + 2))}
        for n in (10, 40, 25, 60)
    ]
    packed = collate_packed(samples)
    padded = collate_padded(samples)
    assert packed.num_trained == padded.num_trained

    model = LMBackbone(tiny_config()).cuda().to(torch.bfloat16).eval()
    loss_p, logs_p = model.loss(
        packed.tokens.cuda(), packed.labels.cuda(), packed.seq_info.to("cuda")
    )
    loss_d, logs_d = model.loss(
        padded.tokens.cuda(), padded.labels.cuda(), padded.seq_info.to("cuda")
    )
    per_token_p = loss_p.item() / int(logs_p["n_tokens"])
    per_token_d = loss_d.item() / int(logs_d["n_tokens"])
    assert abs(per_token_p - per_token_d) < 1e-2


def test_split_padded_preserves_rows():
    samples = [{"input_ids": list(range(20)), "labels": list(range(20))}] * 6
    batch = collate_padded(samples)
    chunks = split_padded(batch, 3)
    assert sum(c.tokens.shape[0] for c in chunks) == 6
    assert sum(c.num_trained for c in chunks) == batch.num_trained
