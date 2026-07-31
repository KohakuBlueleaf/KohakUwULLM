"""Per-stage memory for a preset under a given precision and optimizer.

Answers one question: does this preset fit on this card, and with how much left
for activations? Builds each pipeline stage in turn, allocates its optimizer
state by taking a real step, and reports the peak.

Only one stage exists at a time, so an 8B model is measured on a single GPU.

Usage:
    .venv/bin/python scripts/bench/e2e/stage_memory.py --preset MoE-8B-A1B
    .venv/bin/python scripts/bench/e2e/stage_memory.py --preset MoE-8B-A1B \
        --param-dtype bf16 --optimizer muon
"""

import argparse
import json
import os

import torch

from kohakuwullm.bench import device_name
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.training.optim.build import build_optimizer
from kohakuwullm.training.optim.lowbit import cast_parameters_
from kohakuwullm.training.parallel.pipeline import LMStage, plan_for

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def measure(cfg, plan, args, device):
    backbone = LMBackbone(cfg)
    stage = LMStage(backbone, plan).to(device=device, dtype=torch.float32)
    cast = {"cast": 0, "kept_fp32": 0}
    if args.param_dtype != "fp32":
        cast = cast_parameters_(stage, DTYPES[args.param_dtype])

    params = sum(p.numel() for p in stage.parameters())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    weights_gib = torch.cuda.memory_allocated() / 2**30

    opt = build_optimizer(stage, name=args.optimizer, lr=1e-4)
    lengths = torch.full((1,), args.micro_tokens, dtype=torch.int32)
    info = SeqInfo.from_lengths(lengths, device)
    stage.set_seq_info([info])

    if plan.has_embed:
        inp = torch.randint(0, cfg.vocab_size, (args.micro_tokens,), device=device)
    else:
        inp = torch.randn(
            args.micro_tokens,
            cfg.dim,
            device=device,
            dtype=DTYPES[args.param_dtype],
            requires_grad=True,
        )
    targets = torch.randint(0, cfg.vocab_size, (args.micro_tokens,), device=device)

    try:
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.autocast):
                out = stage(inp)
                if plan.has_head:
                    out, _ = stage.loss(out, targets)
            (out if plan.has_head else out.float().sum()).backward()
            opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        ok = True
        # Optimizer state usually inherits the parameter dtype from
        # zeros_like, so "full bf16" is a claim to verify, not assume.
        seen = {
            str(v.dtype).replace("torch.", "")
            for st in opt.state.values()
            for v in st.values()
            if torch.is_tensor(v) and v.is_floating_point()
        }
        state_dtypes = ",".join(sorted(seen)) or "none"
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in str(
            exc
        ):
            raise
        peak, ok = float("nan"), False
        state_dtypes = "oom"
        torch.cuda.empty_cache()

    row = dict(
        stage=plan.index,
        layers=f"{plan.start_layer}..{plan.end_layer}",
        params_m=params / 1e6,
        weights_gib=weights_gib,
        peak_gib=peak,
        state_dtypes=state_dtypes,
        cast=cast["cast"],
        kept_fp32=cast["kept_fp32"],
        ok=ok,
    )
    del stage, opt, backbone
    torch.cuda.empty_cache()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/train/stage_memory")
    ap.add_argument("--preset", default="MoE-8B-A1B")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--micro-tokens", type=int, default=4096)
    ap.add_argument("--param-dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--autocast", action="store_true", default=True)
    ap.add_argument("--no-autocast", dest="autocast", action="store_false")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda")
    cfg = get_preset(args.preset, vocab_size=args.vocab, tie_embeddings=False)
    cfg.max_position = max(cfg.max_position, args.micro_tokens)
    plans = plan_for(cfg, args.stages, seq_len=args.micro_tokens)

    print(
        f"{device_name()} | {args.preset} | params {args.param_dtype} | "
        f"optimizer {args.optimizer} | micro {args.micro_tokens}",
        flush=True,
    )
    rows = []
    for plan in plans:
        row = measure(cfg, plan, args, device)
        rows.append(row)
        status = (
            f"weights {row['weights_gib']:6.2f}  peak {row['peak_gib']:6.2f} GiB"
            if row["ok"]
            else "OOM"
        )
        print(
            f"  stage {row['stage']} {row['layers']:>8} {row['params_m']:8.1f}M  "
            f"{status}  state={row['state_dtypes']}  "
            f"(cast {row['cast']}, fp32 {row['kept_fp32']})",
            flush=True,
        )

    good = [r for r in rows if r["ok"]]
    if good:
        worst = max(good, key=lambda r: r["peak_gib"])
        print(
            f"\nbinding stage {worst['stage']}: {worst['peak_gib']:.2f} GiB peak"
            f"{'' if len(good) == len(rows) else '  (some stages OOM)'}",
            flush=True,
        )

    name = f"{args.preset}_{args.param_dtype}_{args.optimizer}_{args.micro_tokens}.json"
    with open(os.path.join(args.out, name), "w") as handle:
        json.dump(dict(config=vars(args), rows=rows), handle, indent=2)


if __name__ == "__main__":
    main()
