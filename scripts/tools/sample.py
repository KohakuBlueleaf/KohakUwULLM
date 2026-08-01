"""Sample from a checkpoint on one GPU, with the run's own config.

The model comes from the config's ``PRESET`` / ``ARCH_OVERRIDES`` and the prompts
from its ``SAMPLE_PROMPTS``, so what this prints is what the in-training preview
prints. Rows finish independently: a row that emits EOS is held there while the
rest keep decoding, and the batch stops when all of them are done or the context
is full.

    kogine run scripts/tools/sample.py --config configs/lm/tipo_moe_1b_uwupipe.py \
        --set CKPT=out/ckpt/tipo-moe-1b-uwupipe/step-64000.ckpt \
        --set TEMPERATURE=1.0 --set MIN_P=0.1

``--set COMPILE=reduce-overhead`` compiles the decode step; ``--set BENCH=True``
times it against eager on the same prompts. See docs/guides/generation.md.
"""

import time

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
# Read from the training config, so sampling runs in the dtype the model was
# trained in rather than a default of its own.
PARAM_DTYPE = "bf16"
AUTOCAST_DTYPE = ""
MXFP8 = False
# Attention backend override; None keeps the config's.
ATTENTION: str | None = None
# torch.compile mode for the decode step; "" disables it.
COMPILE = ""
# Timing mode: greedy, a fixed step count and no EOS stop, so every arm of a
# comparison does identical work. 0 disables.
BENCH_STEPS = 0


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def param_dtype() -> torch.dtype:
    """The dtype the checkpoint's parameters are held in."""
    return _DTYPES[PARAM_DTYPE]


def autocast_dtype() -> torch.dtype:
    """The dtype the forward runs in; falls back to the parameter dtype."""
    return _DTYPES[AUTOCAST_DTYPE] if AUTOCAST_DTYPE else param_dtype()


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


def build_model():
    """The checkpoint's model, on device, in eval, optionally compiled."""
    overrides = dict(ARCH_OVERRIDES)
    if ATTENTION:
        overrides["attn"] = ATTENTION
    config = get_preset(PRESET, vocab_size=VOCAB_SIZE, **overrides)
    model = LMBackbone(config)
    load_weights(model, CKPT)
    model = model.to(DEVICE, dtype=param_dtype()).eval()
    if MXFP8:
        model._swap_to_mxfp8()
    if COMPILE:
        # In place and over the whole backbone: one graph per decode step.
        model.compile(mode=COMPILE)
    return model


def run(generator, batch, eos):
    """One generate call under the script's sampling settings."""
    with torch.autocast(DEVICE, dtype=autocast_dtype()):
        return generator.generate(
            batch,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            min_p=MIN_P,
            eos_token_id=eos,
        )


def timed(generator, batch):
    """Best-of-3 wall time for `BENCH_STEPS` greedy steps, EOS stop disabled."""
    with torch.autocast(DEVICE, dtype=autocast_dtype()):

        def once():
            return generator.generate(
                batch,
                max_new_tokens=BENCH_STEPS,
                temperature=0.0,
                eos_token_id=None,
            )

        out = once()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            once()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - start)
    return best, out


def report(tokenizer, name, text, ids, out, eos) -> None:
    """Print one prompt's rows, each trimmed at its own EOS, and where each stopped."""
    print(f"\n{'=' * 78}\n[{name}] prompt:\n{text}\n{'-' * 78}")
    stops = []
    for index, row in enumerate(out):
        body = row[ids.shape[0] :].tolist()
        hit = eos is not None and eos in body
        kept = body[: body.index(eos)] if hit else body
        stops.append(len(kept) if hit else None)
        print(f"[{index}] {len(kept)} tokens, eos={hit}")
        print(f"{tokenizer.decode(kept, skip_special_tokens=True)}\n")
    done = [s for s in stops if s is not None]
    print(
        f"[{name}] batch ran {out.shape[1] - ids.shape[0]} steps; rows stopped at "
        f"{stops} ({len(done)}/{len(stops)} on EOS, "
        f"spread {max(done) - min(done) if done else 0} tokens)"
    )


def main() -> None:
    if not CKPT:
        raise SystemExit("set CKPT to a checkpoint path")
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tokenizer.eos_token_id
    model = build_model()
    generator = LocalGenerator(model, seed=SEED)

    for name, text in prompts():
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        batch = ids.unsqueeze(0).expand(SAMPLE_COUNT, -1).contiguous().to(DEVICE)
        if BENCH_STEPS:
            wall, out = timed(generator, batch)
            print(
                f"RESULT rows={SAMPLE_COUNT} steps={BENCH_STEPS} "
                f"dtype={PARAM_DTYPE}/{AUTOCAST_DTYPE or PARAM_DTYPE} "
                f"mxfp8={int(MXFP8)} compile={COMPILE or 'off'} "
                f"{wall / BENCH_STEPS * 1e3:.2f} ms/step "
                f"{SAMPLE_COUNT * BENCH_STEPS / wall:.0f} tok/s "
                f"first_token={int(out[0, ids.shape[0]])}",
                flush=True,
            )
            return
        out = run(generator, batch, eos)
        report(tokenizer, name, text, ids, out, eos)


if __name__ == "__main__":
    main()
