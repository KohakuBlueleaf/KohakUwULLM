"""Convert a conversation dataset into KVault shards the training loader can index.

Values are JSON ``{"messages": [{"role", "content"}, ...]}`` rather than the
UTF-8 document text the plain corpus stores, so read them with
``CorpusRecords(..., schema="json")`` and render them with ``chat``.
Keys are sequential big-endian int64, as everywhere else in the vault.

    python scripts/data/to_vault_chat.py --src /Iolite/text-dataset/sft/smoltalk \\
        --name sft/smoltalk
    python scripts/data/to_vault_chat.py --src ... --name ... --dry-run

See internal/corpus-data-setup.md.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

VAULT = "/Iolite/text-dataset/_vault"
STAGE = "/xg7/vault-stage"
FILES_PER_SHARD = 3
MIN_REPLY_CHARS = 8
ROLES = ("system", "user", "assistant", "tool")
MESSAGE_FIELDS = ("messages", "conversations", "conversation")
PROMPT_FIELDS = ("instruction", "prompt", "question", "query", "input")
REPLY_FIELDS = ("response", "output", "completion", "answer")
# Per-worker budgets, as in _meta/to_vault.py: both multiply by the worker count.
BATCH_ROWS = 200_000
BATCH_BYTES = 512 << 20
CACHE_KB = 256_000
MMAP_BYTES = 1 << 30


def parquet_files(src: str) -> list[str]:
    """Every parquet under ``src``, in sorted order, skipping the test split."""
    out = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.endswith(".parquet") and not name.startswith("test-"):
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def as_messages(row: dict) -> list[dict]:
    """One parquet row -> ``[{"role", "content"}]``, or ``[]`` if unusable.

    Accepts a message list under any of :data:`MESSAGE_FIELDS`, or a flat
    prompt / reply pair. Blank turns are dropped; an unknown role becomes
    ``user``.
    """
    raw = None
    for field in MESSAGE_FIELDS:
        if row.get(field):
            raw = row[field]
            break
    if raw is None:
        prompt = next((row[f] for f in PROMPT_FIELDS if row.get(f)), "")
        reply = next((row[f] for f in REPLY_FIELDS if row.get(f)), "")
        raw = []
        if row.get("system"):
            raw.append({"role": "system", "content": row["system"]})
        if prompt:
            raw.append({"role": "user", "content": prompt})
        if reply:
            raw.append({"role": "assistant", "content": reply})

    messages = []
    for message in raw:
        content = (message.get("content") or message.get("value") or "").strip()
        if not content:
            continue
        role = (message.get("role") or message.get("from") or "user").lower()
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        messages.append({"role": role if role in ROLES else "user", "content": content})
    return messages


def usable(messages: list[dict]) -> bool:
    """A conversation is usable when a non-trivial reply follows some context."""
    for i, message in enumerate(messages):
        if (
            message["role"] == "assistant"
            and i > 0
            and len(message["content"]) >= MIN_REPLY_CHARS
        ):
            return True
    return False


def rows_parquet(path: str):
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    for group in range(handle.metadata.num_row_groups):
        for row in handle.read_row_group(group).to_pylist():
            yield row


def convert(job) -> tuple[int, int, int, bool]:
    """Write one vault shard from a list of input files."""
    import kohakuvault as kv

    name, files, index, vault_dir, stage_dir = job
    stem = name.replace("/", "__")
    final = os.path.join(vault_dir, f"{stem}.{index:04d}.db")
    marker = final + ".done"
    if os.path.exists(marker):
        with open(marker) as handle:
            return index, int(handle.read().strip() or 0), 0, True

    os.makedirs(stage_dir, exist_ok=True)
    out = os.path.join(stage_dir, f"{stem}.{index:04d}.db")
    for stale in (out, out + "-wal", out + "-shm"):
        if os.path.exists(stale):
            os.remove(stale)

    vault = kv.KVault(out, page_size=65536, cache_kb=CACHE_KB, mmap_size=MMAP_BYTES)
    vault.enable_auto_pack()
    count = skipped = pending = 0
    batch = []
    began = time.time()
    for file_no, path in enumerate(files, start=1):
        for row in rows_parquet(path):
            messages = as_messages(row)
            if not usable(messages):
                skipped += 1
                continue
            record = {"messages": messages}
            if row.get("source"):
                record["subset"] = row["source"]
            blob = json.dumps(record, ensure_ascii=False).encode("utf-8")
            batch.append((count.to_bytes(8, "big"), blob))
            pending += len(blob)
            count += 1
            if len(batch) >= BATCH_ROWS or pending >= BATCH_BYTES:
                vault.update(iter(batch))
                batch.clear()
                pending = 0
        elapsed = time.time() - began
        print(
            f"  PROGRESS {stem}.{index:04d} file {file_no}/{len(files)} "
            f"{count:>10,} rows {skipped:>8,} skipped {elapsed:6.0f}s "
            f"{count / max(elapsed, 1):>7,.0f} rows/s",
            flush=True,
        )
    if batch:
        vault.update(iter(batch))
    vault.close()
    for tail in ("-wal", "-shm"):
        if os.path.exists(out + tail):
            os.remove(out + tail)
    shutil.move(out, final)
    with open(marker, "w") as handle:
        handle.write(str(count))
    return index, count, skipped, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of parquet files")
    ap.add_argument("--name", required=True, help='"<category>/<dataset>"')
    ap.add_argument("--vault", default=VAULT)
    ap.add_argument("--stage", default=STAGE)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--files-per-shard", type=int, default=FILES_PER_SHARD)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = parquet_files(args.src)
    if not files:
        print(f"no parquet under {args.src}")
        return 1
    jobs = [
        (
            args.name,
            files[i : i + args.files_per_shard],
            i // args.files_per_shard,
            args.vault,
            args.stage,
        )
        for i in range(0, len(files), args.files_per_shard)
    ]
    print(f"{args.name}: {len(files)} files -> {len(jobs)} shards", flush=True)
    if args.dry_run:
        for _, group, index, _, _ in jobs:
            print(
                f"  {index:04d}: {len(group)} files, first {os.path.basename(group[0])}"
            )
        return 0

    os.makedirs(args.vault, exist_ok=True)
    total = dropped = 0
    began = time.time()
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for index, count, skipped, cached in pool.imap_unordered(convert, jobs):
            total += count
            dropped += skipped
            state = "cached" if cached else "written"
            print(f"  shard {index:04d} {state}: {count:,} rows", flush=True)
    print(
        f"\n{args.name}: {total:,} rows, {dropped:,} unusable, "
        f"{time.time() - began:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
