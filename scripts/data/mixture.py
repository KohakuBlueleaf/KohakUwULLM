"""Derive per-source ``repeat`` weights for a target token mixture.

Measures each dataset's mean document length by sampling, converts the target
token shares into document counts, and prints a ``SOURCES`` block. Sampling is
bounded, so this reads a few thousand documents rather than the corpus.

    python scripts/data/mixture.py --budget 105e9
    python scripts/data/mixture.py --sample 400 --tokenizer models/tokenizer
"""

import argparse
import sys

# Target token counts, from the 100B row of
# internal/general-pretrain-datasets.md section 2.
# Source that soaks up any shortfall from a too-small dataset.
ABSORB = "en/nemotron-cc-v2.1-hq-syn"
# Sources whose vault rows are JSON conversations, not document text.
CHAT_SOURCES = {"sft/smoltalk"}

TARGET = {
    "en/nemotron-cc-v2.1-hq": 26.0e9,
    "en/nemotron-cc-v2.1-hq-syn": 21.0e9,
    "en/nemotron-cc-v2.1-mhq": 13.0e9,
    "ja/fineweb-2-edu-japanese-10bt": 10.0e9,
    "zh-tw/finepdfs-zh-zhtw": 5.0e9,
    "zh-tw/ultra-fineweb-l3": 3.0e9,
    "zh-tw/wiki-zhtw": 1.0e9,
    "zh-tw/ptt-zhtw": 1.0e9,
    "en/nemotron-cc-v2.1-hq-dqa": 8.0e9,
    "ko/hplt3-kor-hang": 5.0e9,
    "stem/nemotron-specialized": 4.0e9,
    "sft/smoltalk": 1.0e9,
}


def mean_tokens(
    source: str, tokenizer, sample: int, seed: int = 0
) -> tuple[float, int]:
    """``(mean tokens per rendered document, document count)`` from a bounded sample.

    Measured after rendering, so a ChatML source is counted with its turn
    overhead included.
    """
    import random

    from kohakuwullm.data.packing import nonempty_segments
    from kohakuwullm.data.renderers.chatml import ChatRenderer
    from kohakuwullm.data.renderers.plain import PlainRenderer
    from kohakuwullm.data.sources.corpus import CorpusRecords

    chat = source in CHAT_SOURCES
    records = CorpusRecords(source, schema="json" if chat else "text")
    render = ChatRenderer() if chat else PlainRenderer()
    total = len(records)
    rng = random.Random(seed)
    texts = []
    for _ in range(sample * 3):
        if len(texts) >= sample:
            break
        rec = records[rng.randrange(total)]
        if not rec:
            continue
        text = "".join(t for t, _ in nonempty_segments(render(rec)))
        if text:
            texts.append(text)
    if not texts:
        return 0.0, total
    ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return sum(len(x) for x in ids) / len(ids), total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="models/tokenizer")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--ctx-max", type=int, default=2048)
    ap.add_argument(
        "--budget",
        type=float,
        default=0.0,
        help="rescale every target so they sum to this",
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    targets = dict(TARGET)
    if args.budget:
        scale = args.budget / sum(targets.values())
        targets = {k: v * scale for k, v in targets.items()}

    print(
        f"{'source':40s} {'docs':>13s} {'tok/doc':>8s} {'1 pass':>9s} "
        f"{'target':>9s} {'repeat':>7s}"
    )
    print("-" * 92)
    rows = []
    for source, target in targets.items():
        try:
            mean, docs = mean_tokens(source, tokenizer, args.sample)
        except FileNotFoundError:
            print(f"{source:40s} {'NOT CONVERTED':>13s}")
            continue
        # ctx_max truncates, so a document never contributes more than that.
        effective = min(mean, args.ctx_max)
        one_pass = effective * docs
        repeat = target / one_pass if one_pass else 0.0
        rows.append((source, docs, effective, one_pass, target, repeat))
        print(
            f"{source:40s} {docs:13,d} {effective:8.0f} {one_pass / 1e9:8.2f}B "
            f"{target / 1e9:8.2f}B {repeat:7.2f}"
        )

    # Never repeat a source: a short one is taken whole and its shortfall is
    # absorbed elsewhere. See internal/general-pretrain-datasets.md.
    print("\nSOURCES = [")
    shortfall = 0.0
    plan = []
    for source, _, _, one_pass, target, repeat in rows:
        weight = min(repeat, 1.0)
        if repeat > 1.0:
            shortfall += target - one_pass
        plan.append((source, weight, one_pass))
    absorb = ABSORB if ABSORB in {s for s, _, _ in plan} else None
    for source, weight, one_pass in plan:
        if source == absorb and shortfall > 0:
            weight = min(1.0, weight + shortfall / one_pass)
        print(f'    {{"name": "{source}", "repeat": {weight:.3f}}},')
    print("]")

    got = sum(
        (min(1.0, w + (shortfall / p if s == absorb and shortfall > 0 else 0.0))) * p
        for s, w, p in plan
    )
    print(
        f"\n  delivered: {got / 1e9:.1f}B tokens   target {sum(targets.values()) / 1e9:.1f}B"
    )
    if shortfall > 0:
        print(f"  {shortfall / 1e9:.1f}B of shortfall absorbed by {absorb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
