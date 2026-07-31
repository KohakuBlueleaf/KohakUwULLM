"""Prune a BPE tokenizer to a smaller vocabulary, then reserve special slots.

Truncating to the first ``N`` ids yields a closed vocabulary, because a trained
BPE assigns ids in merge order. Special slots are appended *after* the kept
tokens, so ids stay stable across later additions.

See docs/internals/data.md.
"""

import json
import os

SPECIAL_SLOT_TEMPLATE = "<|reserved_{index}|>"


def load_tokenizer_json(name_or_path: str) -> dict:
    """Read a fast-tokenizer ``tokenizer.json`` from a local path or the Hub."""
    if os.path.isdir(name_or_path):
        path = os.path.join(name_or_path, "tokenizer.json")
    elif os.path.isfile(name_or_path):
        path = name_or_path
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(name_or_path, "tokenizer.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def prune_bpe(
    spec: dict,
    keep: int,
    named_specials: list[str] | None = None,
    reserved_slots: int = 0,
    total_size: int | None = None,
) -> dict:
    """Truncate a BPE vocab to ``keep`` tokens, then append the special block.

    Args:
        spec: a parsed ``tokenizer.json``.
        keep: how many ordinary tokens to retain (ids ``0..keep-1``).
        named_specials: special tokens to add first, in order.
        reserved_slots: unnamed ``<|reserved_N|>`` placeholders after them.
        total_size: assert the final vocabulary is exactly this size.

    Returns a new spec; the input is not modified.
    """
    model = spec["model"]
    if model.get("type") != "BPE":
        raise ValueError(f"expected a BPE tokenizer, got {model.get('type')!r}")

    vocab: dict[str, int] = model["vocab"]
    kept = {token: idx for token, idx in vocab.items() if idx < keep}
    if len(kept) != keep:
        raise ValueError(
            f"vocab is not densely numbered below {keep}: kept {len(kept)} tokens"
        )

    # A merge survives only if both parents and the result are all still present.
    merges = model["merges"]
    kept_tokens = set(kept)
    new_merges = []
    for merge in merges:
        left, right = merge if isinstance(merge, (list, tuple)) else merge.split(" ", 1)
        if (
            left in kept_tokens
            and right in kept_tokens
            and (left + right) in kept_tokens
        ):
            new_merges.append(merge)

    specials = list(named_specials or [])
    specials += [SPECIAL_SLOT_TEMPLATE.format(index=i) for i in range(reserved_slots)]

    added = []
    next_id = keep
    for token in specials:
        if token in kept:
            continue
        kept[token] = next_id
        added.append(
            {
                "id": next_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        next_id += 1

    if total_size is not None and next_id != total_size:
        raise ValueError(
            f"final vocab is {next_id}, expected {total_size} "
            f"(keep={keep} + {len(added)} special)"
        )

    out = json.loads(json.dumps(spec))  # deep copy without touching the input
    out["model"]["vocab"] = kept
    out["model"]["merges"] = new_merges
    # Keep any original special that survived truncation, then our new block.
    surviving = [t for t in out.get("added_tokens", []) if t["id"] < keep]
    out["added_tokens"] = surviving + added
    return out


def summarize(spec: dict) -> dict:
    """Counts for a log line / a test assertion."""
    return {
        "vocab_size": len(spec["model"]["vocab"]),
        "merges": len(spec["model"]["merges"]),
        "added_tokens": len(spec.get("added_tokens", [])),
    }
