"""Census a JSONL corpus laid out as ``<root>/<category>/<source>/*.jsonl``.

Answers the questions a mixture decision needs and a dataset card cannot: how
many rows each source really holds, what fields it really carries, how much of
it is truncated, and how much of it is duplicated somewhere else.

    .venv/bin/python scripts/data/dataset_census.py \\
        --root /xg7/datasets/distill-codex/data --out out/data/census

Streams every shard, so peak memory is the dedup table rather than the corpus.
See docs/internals/data.md.
"""

import argparse
import collections
import hashlib
import json
import os
import sys

# The card caps instruction/response here, so a row landing exactly on it was cut.
TRUNCATION_CAP = 4000


def shards(root: str):
    """``(category, source, path)`` for every JSONL under the two-level layout."""
    for category in sorted(os.listdir(root)):
        cat_dir = os.path.join(root, category)
        if not os.path.isdir(cat_dir):
            continue
        for source in sorted(os.listdir(cat_dir)):
            src_dir = os.path.join(cat_dir, source)
            if not os.path.isdir(src_dir):
                continue
            for name in sorted(os.listdir(src_dir)):
                if name.endswith(".jsonl"):
                    yield category, source, os.path.join(src_dir, name)


def row_text(row: dict) -> str:
    """The response-ish field, whatever this source happens to call it."""
    for key in ("response", "output", "completion", "answer", "content", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def prompt_text(row: dict) -> str:
    for key in ("instruction", "prompt", "input", "question", "query"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def fingerprint(row: dict) -> str:
    """Hash of prompt+response, normalized, for cross-source duplicate detection."""
    blob = (prompt_text(row).strip() + "\x00" + row_text(row).strip()).lower()
    return hashlib.blake2b(blob.encode("utf-8", "replace"), digest_size=16).hexdigest()


def census(root: str, dedup: bool, limit: int) -> dict:
    """Per-source counts, fields, truncation and duplicate share."""
    stats: dict = {}
    seen: dict[str, str] = {} if dedup else {}
    for category, source, path in shards(root):
        key = f"{category}/{source}"
        entry = stats.setdefault(
            key,
            {
                "category": category,
                "source": source,
                "rows": 0,
                "shards": 0,
                "bytes": 0,
                "fields": collections.Counter(),
                "truncated": 0,
                "empty_response": 0,
                "dup_rows": 0,
                "dup_against": collections.Counter(),
                "source_dataset": collections.Counter(),
                "chars": 0,
            },
        )
        entry["shards"] += 1
        entry["bytes"] += os.path.getsize(path)
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    entry["rows"] += 1
                    continue
                entry["rows"] += 1
                if entry["rows"] <= limit:
                    entry["fields"].update(row.keys())
                    if (tag := row.get("source_dataset")) is not None:
                        entry["source_dataset"][str(tag)] += 1
                body = row_text(row)
                entry["chars"] += len(body)
                if not body:
                    entry["empty_response"] += 1
                if (
                    len(body) >= TRUNCATION_CAP
                    or len(prompt_text(row)) >= TRUNCATION_CAP
                ):
                    entry["truncated"] += 1
                if dedup:
                    mark = fingerprint(row)
                    owner = seen.get(mark)
                    if owner is None:
                        seen[mark] = key
                    elif owner != key:
                        entry["dup_rows"] += 1
                        entry["dup_against"][owner] += 1
                    else:
                        entry["dup_rows"] += 1
                        entry["dup_against"]["<self>"] += 1
        print(f"  {key:52s} {entry['rows']:>10,} rows", file=sys.stderr, flush=True)
    return stats


def report(stats: dict, out_dir: str) -> None:
    """Write the JSON census and a markdown table ranked by row count."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for key, e in stats.items():
        payload[key] = {
            "category": e["category"],
            "source": e["source"],
            "rows": e["rows"],
            "shards": e["shards"],
            "gib": round(e["bytes"] / 2**30, 3),
            "mean_response_chars": round(e["chars"] / max(e["rows"], 1), 1),
            "truncated_frac": round(e["truncated"] / max(e["rows"], 1), 4),
            "empty_response_frac": round(e["empty_response"] / max(e["rows"], 1), 4),
            "dup_frac": round(e["dup_rows"] / max(e["rows"], 1), 4),
            "dup_against": dict(e["dup_against"].most_common(5)),
            "fields": sorted(e["fields"]),
            "source_dataset": dict(e["source_dataset"].most_common(5)),
        }
    with open(os.path.join(out_dir, "census.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    rows = sorted(payload.values(), key=lambda r: -r["rows"])
    total = sum(r["rows"] for r in rows)
    lines = [
        f"# Census: {len(rows)} sources, {total:,} rows",
        "",
        "| source | category | rows | share | GiB | trunc | dup | mean chars |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['source']}` | {r['category']} | {r['rows']:,} | "
            f"{100 * r['rows'] / max(total, 1):.1f}% | {r['gib']:.2f} | "
            f"{100 * r['truncated_frac']:.1f}% | {100 * r['dup_frac']:.1f}% | "
            f"{r['mean_response_chars']:.0f} |"
        )
    with open(os.path.join(out_dir, "census.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"\nwrote {out_dir}/census.json and census.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="out/data/census")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument(
        "--field-sample",
        type=int,
        default=2000,
        help="rows per source inspected for field names",
    )
    args = ap.parse_args()
    report(census(args.root, not args.no_dedup, args.field_sample), args.out)


if __name__ == "__main__":
    main()
