"""Document filters applied at load time, before rendering.

A rejected document is skipped and the packer draws the next one, so it costs
its read and nothing else. See docs/internals/data.md.
"""

import re

DEFAULT_MAX_N = 13
DEFAULT_THRESHOLD = 32
CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")


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
    """Whether some n-gram repeats consecutively at least ``threshold`` times."""
    seq = units(text)
    if len(seq) < threshold:
        return False
    return max_repeat_run(seq, max_n)[0] >= threshold


def ngram_repeat(field: str = "text", **kwargs):
    """Filter rejecting documents with long consecutive n-gram repeats."""

    def predicate(rec: dict) -> bool:
        text = rec.get(field)
        return bool(text) and is_degenerate(text, **kwargs)

    return predicate


FILTERS = {"ngram": ngram_repeat}


def build_filter(spec):
    """Resolve ``None`` / a name / a ``{"name": ..., **kwargs}`` dict / a callable."""
    if spec is None or callable(spec):
        return spec
    if isinstance(spec, str):
        return FILTERS[spec]()
    options = dict(spec)
    return FILTERS[options.pop("name")](**options)
