"""Build the KohakUwULLM tokenizer: DeepSeek-V4 pruned to 64k + 1536 special slots.

The result is 65536 ids exactly:

    [0     , 64000)  ordinary BPE tokens, kept in merge order from DeepSeek-V4
    [64000 , 64000+n) named specials -- BOS/EOS/PAD plus the TIPO task vocabulary
    [.     , 65536)  <|reserved_N|> placeholders

A power-of-two vocabulary keeps the head GEMM's N dimension tile-aligned, and
the reserved block means a new control token later is an id assignment rather
than a re-embedding.

After writing, the script re-tokenizes a sample of the real corpus with both the
original and the pruned tokenizer and reports the change in tokens-per-sample --
pruning that quietly inflates sequence length by 20% would cost more than the
embedding it saved, so it is measured rather than assumed.

Usage:
    .venv/bin/python scripts/tokenizer/build_tokenizer.py --out models/tokenizer
    .venv/bin/python scripts/tokenizer/build_tokenizer.py --out models/tokenizer \\
        --check-samples 2000
"""

import argparse
import json
import os
import random

from kohakuwullm.data.renderers.tipo import SPECIAL_TOKENS, TIPORenderer
from kohakuwullm.tokenizer.prune import load_tokenizer_json, prune_bpe, summarize

SOURCE = "deepseek-ai/DeepSeek-V4-Flash"
CORE_SPECIALS = ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]


def build(out_dir: str, source: str, keep: int, total: int) -> dict:
    spec = load_tokenizer_json(source)
    named = CORE_SPECIALS + SPECIAL_TOKENS
    pruned = prune_bpe(
        spec,
        keep=keep,
        named_specials=named,
        reserved_slots=total - keep - len(named),
        total_size=total,
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tokenizer.json"), "w", encoding="utf-8") as handle:
        json.dump(pruned, handle, ensure_ascii=False)

    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 131072,
        "clean_up_tokenization_spaces": False,
    }
    with open(
        os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(config, handle, indent=2)
    return pruned


def check_roundtrip(out_dir: str, source: str, n_samples: int, root: str) -> None:
    """Compare tokens-per-sample before and after pruning, on real records."""
    from transformers import AutoTokenizer

    from kohakuwullm.data.sources.vault import DanbooruRecords

    original = AutoTokenizer.from_pretrained(source)
    pruned = AutoTokenizer.from_pretrained(out_dir)
    renderer = TIPORenderer()

    try:
        records = DanbooruRecords(root=root)
    except FileNotFoundError:
        print(f"no corpus at {root}; skipping the length check")
        return

    rng = random.Random(0)
    before = after = 0
    unk_id = pruned.unk_token_id
    unk_hits = 0
    checked = 0
    for _ in range(n_samples):
        rec = records[rng.randrange(len(records))]
        if rec is None:
            continue
        user, output = renderer(rec, rng=rng)
        text = user + output
        a = original(text, add_special_tokens=False)["input_ids"]
        b = pruned(text, add_special_tokens=False)["input_ids"]
        before += len(a)
        after += len(b)
        unk_hits += sum(1 for t in b if t == unk_id)
        checked += 1

    if not checked:
        print("no records fetched; skipping the length check")
        return
    print(
        f"length check over {checked} samples:\n"
        f"  original  {before / checked:8.1f} tokens/sample\n"
        f"  pruned    {after / checked:8.1f} tokens/sample "
        f"({100 * (after - before) / max(before, 1):+.2f}%)\n"
        f"  unk       {unk_hits} token(s) total"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="models/tokenizer")
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--keep", type=int, default=64000, help="ordinary tokens to retain")
    ap.add_argument("--total", type=int, default=65536, help="final vocabulary size")
    ap.add_argument("--check-samples", type=int, default=1000)
    ap.add_argument("--corpus-root", default="/xg7/caption-datasets")
    args = ap.parse_args()

    pruned = build(args.out, args.source, args.keep, args.total)
    stats = summarize(pruned)
    print(
        f"wrote {args.out}\n"
        f"  vocab   {stats['vocab_size']}\n"
        f"  merges  {stats['merges']}\n"
        f"  special {stats['added_tokens']}"
    )
    if args.check_samples > 0:
        check_roundtrip(args.out, args.source, args.check_samples, args.corpus_root)


if __name__ == "__main__":
    main()
