"""Profile a JSONL corpus: field shape, content markers and token budget.

Complements ``dataset_census.py``, which counts rows. This answers what the rows
contain and what they would cost to train on.

Tokens are estimated two ways. Every row contributes characters, and a bounded
sample per source is really tokenized to fix that source's own characters-per-
token ratio. Extrapolating from a per-source ratio rather than a global constant
matters here, because code and prose differ by a factor approaching two.

    .venv/bin/python scripts/data/dataset_profile.py \\
        --root /xg7/datasets/distill-codex/data --out internal/profile \\
        --tokenizer models/tokenizer --sample 2000

Streaming, so memory is the counters and one tokenizer batch.
See docs/internals/data.md.
"""

import argparse
import collections
import json
import os
import re
import statistics
import sys

TRUNCATION_CAP = 4000
BATCH = 256

MARKERS = {
    "fenced_code": re.compile(r"```"),
    "chatml": re.compile(r"<\|im_start\|>|<\|im_end\|>"),
    "markdown_head": re.compile(r"^#{1,6} ", re.M),
    "diff": re.compile(r"^(\+\+\+|---|@@) ", re.M),
    "html": re.compile(r"</[a-zA-Z][^>]*>"),
    "json_like": re.compile(r"^\s*[{\[]"),
    "url": re.compile(r"https?://"),
    "refusal": re.compile(r"\b(I can't|I cannot|I'm sorry|As an AI)\b", re.I),
}
CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")


def shards(root: str):
    for category in sorted(os.listdir(root)):
        cat = os.path.join(root, category)
        if not os.path.isdir(cat):
            continue
        for source in sorted(os.listdir(cat)):
            src = os.path.join(cat, source)
            if not os.path.isdir(src):
                continue
            for name in sorted(os.listdir(src)):
                if name.endswith(".jsonl"):
                    yield category, source, os.path.join(src, name)


def field(row: dict, names) -> str:
    for key in names:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


PROMPT_KEYS = ("instruction", "prompt", "input", "question", "query", "user")
REPLY_KEYS = (
    "response",
    "output",
    "completion",
    "answer",
    "content",
    "text",
    "assistant",
)


def new_entry(category: str, source: str) -> dict:
    return {
        "category": category,
        "source": source,
        "rows": 0,
        "chars": 0,
        "prompt_chars": 0,
        "fields": collections.Counter(),
        "populated": collections.Counter(),
        "markers": collections.Counter(),
        "lengths": [],
        "cjk_rows": 0,
        "truncated": 0,
        "sample_chars": 0,
        "sample_tokens": 0,
        "empty_rows": 0,
    }


def shard_counts(root: str) -> dict:
    """Shards per source, so the token sample can be spread over all of them."""
    counts: collections.Counter = collections.Counter()
    for category, source, _ in shards(root):
        counts[f"{category}/{source}"] += 1
    return counts


def profile(root: str, tokenizer, sample: int, keep_lengths: int) -> dict:
    stats: dict = {}
    batch: list[str] = []
    owner: list[dict] = []
    per_shard = shard_counts(root)

    def flush() -> None:
        if not batch:
            return
        for entry, ids in zip(
            owner, tokenizer(batch, add_special_tokens=False)["input_ids"]
        ):
            entry["sample_tokens"] += len(ids)
        batch.clear()
        owner.clear()

    for category, source, path in shards(root):
        key = f"{category}/{source}"
        entry = stats.setdefault(key, new_entry(category, source))
        quota = max(1, sample // max(per_shard[key], 1))
        taken = 0
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry["rows"] += 1
                if not field(row, REPLY_KEYS).strip():
                    entry["empty_rows"] += 1
                entry["fields"].update(row.keys())
                for name, value in row.items():
                    if isinstance(value, str) and value.strip():
                        entry["populated"][name] += 1

                prompt = field(row, PROMPT_KEYS)
                reply = field(row, REPLY_KEYS)
                text = prompt + "\n" + reply
                entry["chars"] += len(reply)
                entry["prompt_chars"] += len(prompt)
                if len(reply) >= TRUNCATION_CAP or len(prompt) >= TRUNCATION_CAP:
                    entry["truncated"] += 1
                if len(entry["lengths"]) < keep_lengths:
                    entry["lengths"].append(len(reply))
                if CJK.search(text):
                    entry["cjk_rows"] += 1
                for name, pattern in MARKERS.items():
                    if pattern.search(text):
                        entry["markers"][name] += 1

                if tokenizer is not None and taken < quota and reply.strip():
                    taken += 1
                    entry["sample_chars"] += len(text)
                    batch.append(text)
                    owner.append(entry)
                    if len(batch) >= BATCH:
                        flush()
        print(f"  {key:52s} {entry['rows']:>10,}", file=sys.stderr, flush=True)
    flush()
    return stats


def report(stats: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for key, e in stats.items():
        rows = max(e["rows"], 1)
        lengths = sorted(e["lengths"]) or [0]
        ratio = e["sample_chars"] / e["sample_tokens"] if e["sample_tokens"] else None
        total_chars = e["chars"] + e["prompt_chars"]
        payload[key] = {
            "category": e["category"],
            "source": e["source"],
            "rows": e["rows"],
            "total_chars": total_chars,
            "chars_per_token": round(ratio, 3) if ratio else None,
            "est_tokens": int(total_chars / ratio) if ratio else int(total_chars / 4),
            "est_tokens_basis": "measured" if ratio else "chars/4",
            "median_response_chars": statistics.median(lengths),
            "p95_response_chars": lengths[
                min(len(lengths) - 1, int(0.95 * len(lengths)))
            ],
            "empty_frac": round(e["empty_rows"] / rows, 4),
            "truncated_frac": round(e["truncated"] / rows, 4),
            "cjk_frac": round(e["cjk_rows"] / rows, 4),
            "fields": sorted(e["fields"]),
            "populated_frac": {
                k: round(v / rows, 3) for k, v in e["populated"].most_common(12)
            },
            "marker_frac": {
                k: round(v / rows, 3) for k, v in e["markers"].most_common()
            },
        }
    with open(os.path.join(out_dir, "profile.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    rows = sorted(payload.values(), key=lambda r: -r["est_tokens"])
    total = sum(r["est_tokens"] for r in rows)
    lines = [
        f"# Profile: {len(rows)} sources, ~{total / 1e9:.2f}B estimated tokens",
        "",
        "| source | rows | est tokens | share | c/tok | med resp | empty | trunc | code | cjk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['source']}` | {r['rows']:,} | {r['est_tokens'] / 1e6:.1f}M | "
            f"{100 * r['est_tokens'] / max(total, 1):.1f}% | "
            f"{r['chars_per_token'] or 0:.2f} | {r['median_response_chars']:.0f} | "
            f"{100 * r['empty_frac']:.0f}% | {100 * r['truncated_frac']:.1f}% | "
            f"{100 * r['marker_frac'].get('fenced_code', 0):.0f}% | "
            f"{100 * r['cjk_frac']:.1f}% |"
        )
    with open(os.path.join(out_dir, "profile.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out_dir}/profile.json and profile.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="internal/profile")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--keep-lengths", type=int, default=20000)
    args = ap.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report(profile(args.root, tokenizer, args.sample, args.keep_lengths), args.out)


if __name__ == "__main__":
    main()
