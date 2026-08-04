"""Preview prompts drawn from real corpus documents.

Two sources: :func:`prompts_from_batch` cuts the batch being trained on at a
turn boundary read out of its own loss mask, and :func:`corpus_prompts` cuts
held-out documents at a token fraction. Both draws are seeded.
See docs/guides/training.md.
"""

import random

from kohakuwullm.data.packing import IGNORE_INDEX
from kohakuwullm.data.sources.corpus import CorpusRecords

FILLER_MIN_TOKENS = 32


def _batch_fields(batch):
    """``(tokens, labels, [SeqInfo])`` from a packed batch, a step, or a pipeline step.

    Accepts all three field spellings the trainers use: ``PackedBatch.seq_info``,
    ``MicroBatchedStep.seq_infos``, and ``PipelineStep``'s
    ``inputs`` / ``target`` / ``layout``.
    """
    tokens = getattr(batch, "tokens", None)
    if tokens is None:
        tokens, labels = batch.inputs, batch.target
    else:
        labels = batch.labels
    infos = getattr(batch, "seq_infos", None)
    if infos is None:
        infos = getattr(batch, "layout", None)
    if infos is None:
        infos = batch.seq_info
    if not isinstance(infos, (list, tuple)):
        infos = [infos]
    return tokens, labels, list(infos)


def _documents(batch):
    """``[(tokens, labels)]`` per document, from a packed batch or a step."""
    tokens, labels, infos = _batch_fields(batch)
    width = int(tokens.numel()) // len(infos)
    out = []
    for micro, info in enumerate(infos):
        base = micro * width
        bounds = info.cu_seqlens.tolist()
        for start, stop in zip(bounds, bounds[1:]):
            out.append(
                (
                    tokens[base + start : base + stop],
                    labels[base + start : base + stop],
                )
            )
    return out


def _trained_spans(labels) -> list[tuple[int, int]]:
    """``[(start, stop)]`` per run of trained tokens, in unshifted positions.

    ``labels`` is a document's shifted row, where a target at position ``i``
    means token ``i + 1`` is the one being predicted.
    """
    flags = (labels != IGNORE_INDEX).tolist()
    spans: list[tuple[int, int]] = []
    start = None
    for i, trained in enumerate(flags):
        if trained and start is None:
            start = i
        elif not trained and start is not None:
            spans.append((start + 1, i + 1))
            start = None
    if start is not None:
        spans.append((start + 1, len(flags)))
    return spans


def _pick_turn(
    spans: list[tuple[int, int]],
    max_prefix_tokens: int,
    min_prompt_tokens: int,
    min_reference_tokens: int,
) -> tuple[int, int] | None:
    """The deepest trained span whose context still fits the prompt cap."""
    fits = [
        (start, stop)
        for start, stop in spans
        if min_prompt_tokens <= start <= max_prefix_tokens
        and stop - start >= min_reference_tokens
    ]
    return fits[-1] if fits else None


def prompts_from_batch(
    batch,
    tokenizer,
    count: int = 4,
    prefix_frac: float = 0.25,
    max_prefix_tokens: int = 768,
    max_reference_tokens: int = 256,
    min_doc_tokens: int = 64,
    min_prompt_tokens: int = 16,
    min_reference_tokens: int = 8,
    turns_only: bool = True,
    seed: int | None = None,
) -> list[tuple[str, str, list[int], str]]:
    """``[(name, prompt, prompt_ids, reference)]`` cut from the batch being trained on.

    The cut lands on the first token of a trained span, so the prompt is the
    document's own context up to and including ``<|im_start|>assistant\\n`` and
    the reference is that one turn rather than the document remainder. A
    document whose loss covers every token carries no boundary and is skipped;
    with ``turns_only`` off it is cut at ``prefix_frac`` instead.
    """
    docs = _documents(batch)
    rng = random.Random(seed) if seed is not None else random
    order = list(range(len(docs)))
    rng.shuffle(order)

    rows: list[tuple[str, str, list[int], str]] = []
    for index in order:
        if len(rows) >= count:
            break
        tokens, labels = docs[index]
        length = int(tokens.numel())
        turn = _pick_turn(
            _trained_spans(labels),
            max_prefix_tokens,
            min_prompt_tokens,
            min_reference_tokens,
        )
        if turn is None:
            if turns_only or length < min_doc_tokens:
                continue
            if int((labels != IGNORE_INDEX).sum()) < FILLER_MIN_TOKENS:
                continue
            cut = min(max(int(length * prefix_frac), 16), max_prefix_tokens, length - 8)
            turn = (cut, length)
        start, stop = turn
        prompt_ids = tokens[:start].tolist()
        rows.append(
            (
                f"batch[{index}]",
                tokenizer.decode(prompt_ids, skip_special_tokens=False),
                prompt_ids,
                tokenizer.decode(
                    tokens[start : min(stop, start + max_reference_tokens)].tolist(),
                    skip_special_tokens=True,
                ),
            )
        )
    return rows


def corpus_prompts(
    source: str,
    tokenizer,
    count: int = 8,
    prefix_frac: float = 0.25,
    max_prefix_tokens: int = 256,
    min_doc_tokens: int = 128,
    seed: int = 20090220,
    holdout: int = 100_000,
    field: str = "text",
) -> tuple[list[str], list[str]]:
    """``(prompts, references)`` cut from ``holdout`` documents at the tail.

    Args:
        source: ``"<category>/<dataset>"``.
        prefix_frac: fraction of each document's tokens used as the prompt.
        max_prefix_tokens: cap on the prompt, so previews stay cheap.
        min_doc_tokens: skip documents too short to split usefully.
        holdout: draw only from the last this-many documents, which the
            training mixture should exclude.
    """
    records = CorpusRecords(source, field=field)
    total = len(records)
    lo = max(0, total - holdout)
    rng = random.Random(seed)

    prompts: list[str] = []
    references: list[str] = []
    seen = 0
    while len(prompts) < count and seen < count * 50:
        seen += 1
        rec = records[rng.randrange(lo, total)]
        if not rec:
            continue
        ids = tokenizer(rec[field], add_special_tokens=False)["input_ids"]
        if len(ids) < min_doc_tokens:
            continue
        cut = min(max(int(len(ids) * prefix_frac), 16), max_prefix_tokens)
        prompts.append(tokenizer.decode(ids[:cut]))
        references.append(tokenizer.decode(ids[cut : cut + max_prefix_tokens]))
    return prompts, references
