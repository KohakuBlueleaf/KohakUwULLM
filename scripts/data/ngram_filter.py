"""Score documents by their longest run of repeated n-grams.

A document is rejected when some n-gram, for any n in ``1..max_n`` words,
repeats consecutively at least ``threshold`` times. OLMo 2 traced training loss
spikes to exactly this shape, and filtering it also cuts generation repetition.
See internal/training-health-monitoring.md.

    python scripts/data/ngram_filter.py --source en/nemotron-cc-v2.1-hq --sample 20000
    python scripts/data/ngram_filter.py --source zh-tw/ptt-zhtw --show 3
"""

import argparse
import re
import sys

DEFAULT_MAX_N = 13
DEFAULT_THRESHOLD = 32
# CJK has no spaces, so fall back to characters when whitespace is scarce.
CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")


def units(text: str) -> list[str]:
    """Words, or characters wherever whitespace fails to segment the text.

    Covers CJK and equally the unspaced ASCII runs (base64, hex dumps) that a
    word split collapses into a single token.
    """
    words = text.split()
    if len(words) >= 8 and len(text) / max(len(words), 1) < 12:
        return words
    return list(text)


def max_repeat_run(seq: list[str], max_n: int = DEFAULT_MAX_N) -> tuple[int, int]:
    """``(longest consecutive repeat count, its n)`` over n-grams of size 1..max_n."""
    best = (0, 0)
    length = len(seq)
    for n in range(1, min(max_n, length // 2) + 1):
        run = 1
        i = n
        while i + n <= length:
            if seq[i - n : i] == seq[i : i + n]:
                run += 1
                if run > best[0]:
                    best = (run, n)
            else:
                run = 1
            i += n
    return best


def is_degenerate(
    text: str,
    max_n: int = DEFAULT_MAX_N,
    threshold: int = DEFAULT_THRESHOLD,
) -> bool:
    seq = units(text)
    if len(seq) < threshold:
        return False
    return max_repeat_run(seq, max_n)[0] >= threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="<category>/<dataset>")
    ap.add_argument("--root", default="/Iolite/text-dataset/_vault")
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--max-n", type=int, default=DEFAULT_MAX_N)
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    ap.add_argument("--show", type=int, default=0, help="print this many rejects")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random

    from kohakuwullm.data.sources.corpus import CorpusRecords

    records = CorpusRecords(args.source)
    total = len(records)
    rng = random.Random(args.seed)
    picks = [rng.randrange(total) for _ in range(min(args.sample, total))]

    seen = rejected = 0
    shown = 0
    worst = (0, 0, "")
    for index in picks:
        rec = records[index]
        if not rec:
            continue
        text = rec["text"]
        seen += 1
        seq = units(text)
        if len(seq) < args.threshold:
            continue
        run, n = max_repeat_run(seq, args.max_n)
        if run > worst[0]:
            worst = (run, n, text)
        if run >= args.threshold:
            rejected += 1
            if shown < args.show:
                shown += 1
                print(f"\n  REJECT run={run} n={n}: {text[:200]!r}")

    print(f"\n{args.source}: {total:,} docs, sampled {seen:,}")
    print(
        f"  rejected {rejected:,} ({100 * rejected / max(seen, 1):.3f}%) "
        f"at threshold {args.threshold}, max_n {args.max_n}"
    )
    print(f"  worst run seen: {worst[0]} repeats of a {worst[1]}-gram")
    return 0


if __name__ == "__main__":
    sys.exit(main())
