"""Preview prompts drawn from real corpus documents.

Takes a fixed sample of documents, cuts each at a token fraction, and returns
the prefixes as prompts plus the true continuations. The draw is seeded, so the
same documents appear at every preview and progress across a run is visible on
identical inputs. See docs/guides/training.md.
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


def prompts_from_batch(
    batch,
    tokenizer,
    count: int = 4,
    prefix_frac: float = 0.25,
    max_prefix_tokens: int = 256,
    max_reference_tokens: int = 256,
    min_doc_tokens: int = 64,
    seed: int | None = None,
) -> list[tuple[str, str, str]]:
    """``[(name, prompt, reference)]`` reconstructed from the batch being trained on.

    Cuts each document at its loss-mask boundary when it has one, so a chat
    example is prompted with its real context and compared against its real
    reply; otherwise at ``prefix_frac`` of the document. Fully-masked filler
    documents are skipped.
    """
    docs = _documents(batch)
    rng = random.Random(seed) if seed is not None else random
    order = list(range(len(docs)))
    rng.shuffle(order)

    rows: list[tuple[str, str, str]] = []
    for index in order:
        if len(rows) >= count:
            break
        tokens, labels = docs[index]
        length = int(tokens.numel())
        if length < min_doc_tokens:
            continue
        trained = labels != IGNORE_INDEX
        if int(trained.sum()) < FILLER_MIN_TOKENS:
            continue
        # A masked prefix is the example's own context; use it verbatim.
        first = int(trained.nonzero()[0]) if bool(trained.any()) else 0
        cut = (
            first
            if first >= 16
            else min(max(int(length * prefix_frac), 16), max_prefix_tokens)
        )
        cut = min(cut, length - 8)
        prompt = tokenizer.decode(tokens[:cut].tolist(), skip_special_tokens=False)
        reference = tokenizer.decode(
            tokens[cut : cut + max_reference_tokens].tolist(), skip_special_tokens=True
        )
        rows.append((f"batch[{index}]", prompt, reference))
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
