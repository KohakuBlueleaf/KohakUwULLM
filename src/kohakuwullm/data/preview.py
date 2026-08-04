"""Preview prompts drawn from real corpus documents.

Two sources: :func:`prompts_from_batch` cuts the batch being trained on at a
turn boundary read out of its own loss mask, and :func:`corpus_prompts` cuts
held-out documents at a token fraction. Both draws are seeded.
See docs/guides/training.md.
"""

import random

from kohakuwullm.data.packing import IGNORE_INDEX
from kohakuwullm.data.renderers.chatml import ASSISTANT_OPEN
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
    tokens,
    header: list[int] | None,
    max_prefix_tokens: int,
    min_prompt_tokens: int,
    min_reference_tokens: int,
) -> tuple[int, int] | None:
    """The deepest trained span whose context fits the prompt cap.

    With ``header``, a span also has to open an assistant turn: the ids right
    before it must be ``<|im_start|>assistant\\n``. Without it, any span whose
    context is long enough qualifies, which includes a masked TIPO prompt.
    """
    fits = []
    for start, stop in spans:
        if not min_prompt_tokens <= start <= max_prefix_tokens:
            continue
        if stop - start < min_reference_tokens:
            continue
        if header is not None and (
            start < len(header)
            or tokens[start - len(header) : start].tolist() != header
        ):
            continue
        fits.append((start, stop))
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

    With ``turns_only`` the cut lands on the first token after an
    ``<|im_start|>assistant\\n`` header, so the prompt is the document's own
    context and the reference is that one reply rather than the document
    remainder. Documents with no such header are skipped. Without it, any
    mask boundary qualifies and a document with none is cut at ``prefix_frac``.
    """
    docs = _documents(batch)
    rng = random.Random(seed) if seed is not None else random
    order = list(range(len(docs)))
    rng.shuffle(order)
    header = (
        tokenizer(ASSISTANT_OPEN, add_special_tokens=False)["input_ids"]
        if turns_only
        else None
    )

    rows: list[tuple[str, str, list[int], str]] = []
    for index in order:
        if len(rows) >= count:
            break
        tokens, labels = docs[index]
        length = int(tokens.numel())
        turn = _pick_turn(
            _trained_spans(labels),
            tokens,
            header,
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
