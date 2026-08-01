"""Sample from a trained checkpoint, on one GPU, with the run's own config.

The model comes from the config's ``PRESET`` / ``ARCH_OVERRIDES`` and the prompts
from its ``SAMPLE_PROMPTS``, so what this prints is what the in-training preview
would print. Generation runs until every row emits EOS or the context is full.

    kogine run scripts/tools/sample.py --config configs/lm/tipo_moe_1b_uwupipe.py \
        --set CKPT=out/ckpt/tipo-moe-1b-uwupipe/step-64000.ckpt \
        --set TEMPERATURE=1.0 --set MIN_P=0.1

See docs/guides/generation.md.
"""

import torch
from transformers import AutoTokenizer

from kohakuwullm.data.renderers.tipo import build_prompt
from kohakuwullm.generation.engine import LocalGenerator
from kohakuwullm.models import LMBackbone
from kohakuwullm.models.presets import get_preset

CKPT = ""
TOKENIZER = "models/tokenizer"
VOCAB_SIZE = 65536
PRESET = "Kohaku-MoE-1B"
ARCH_OVERRIDES: dict = {}
SAMPLE_PROMPTS: list | None = None
SAMPLE_COUNT = 4
# None runs every row to EOS or to the model's context limit.
MAX_NEW_TOKENS: int | None = None
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
MIN_P = 0.1
SEED = 20090220
DEVICE = "cuda"
DTYPE = "bfloat16"
ATTENTION: str | None = None


def load_weights(model: LMBackbone, path: str) -> None:
    """Load a trainer checkpoint's ``state_dict`` into a whole-model backbone."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)}: {missing[:5]}")
    if unexpected:
        print(f"[warn] ignored {len(unexpected)} extra keys, e.g. {unexpected[:3]}")


def prompts() -> list[tuple[str, str]]:
    """``(name, text)`` for each entry of ``SAMPLE_PROMPTS``."""
    specs = SAMPLE_PROMPTS or [{"tags": "1girl"}]
    out = []
    for index, spec in enumerate(specs):
        spec = dict(spec)
        name = spec.pop("name", None) or f"prompt{index}"
        out.append((name, build_prompt(**spec)))
    return out


def main() -> None:
    if not CKPT:
        raise SystemExit("set CKPT to a checkpoint path")
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    overrides = dict(ARCH_OVERRIDES)
    if ATTENTION:
        overrides["attention"] = ATTENTION
    config = get_preset(PRESET, vocab_size=VOCAB_SIZE, **overrides)
    model = LMBackbone(config)
    load_weights(model, CKPT)
    model = model.to(DEVICE, dtype=getattr(torch, DTYPE)).eval()

    generator = LocalGenerator(model, seed=SEED)
    eos = tokenizer.eos_token_id
    for name, text in prompts():
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        batch = ids.unsqueeze(0).expand(SAMPLE_COUNT, -1).contiguous().to(DEVICE)
        with torch.autocast(DEVICE, dtype=getattr(torch, DTYPE)):
            out = generator.generate(
                batch,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                min_p=MIN_P,
                eos_token_id=eos,
            )
        print(f"\n{'=' * 78}\n[{name}] prompt:\n{text}\n{'-' * 78}")
        for index, row in enumerate(out):
            body = row[ids.shape[0] :]
            hit_eos = bool((body == eos).any()) if eos is not None else False
            kept = body.tolist()
            if eos is not None and hit_eos:
                kept = kept[: kept.index(eos)]
            print(
                f"[{index}] {len(kept)} tokens, eos={hit_eos}\n"
                f"{tokenizer.decode(kept, skip_special_tokens=True)}\n"
            )


if __name__ == "__main__":
    main()
