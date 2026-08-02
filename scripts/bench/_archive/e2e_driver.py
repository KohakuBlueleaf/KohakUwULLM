"""Launch one isolated process group per (preset, strategy), then aggregate + plot.

Every combination gets its own process. Two reasons, both learned by failure:

* Mixing DDP's collective communicators with the pipeline's lazily-created P2P
  communicators inside one process throws an internal NCCL error, even though
  each works alone.
* Peak-memory numbers are meaningless if a previous strategy left the caching
  allocator fragmented.

A combination that fails leaves the others intact; its part file records the
error so the plot can show a gap rather than silently omitting it.

Usage:
    .venv/bin/python scripts/bench/_archive/e2e_driver.py --out out/bench/train/e2e
    .venv/bin/python scripts/bench/_archive/e2e_driver.py --presets Nano-500M \\
        --strategies ddp pipeline --gpus 4
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time


def run_one(preset, strategy, args) -> bool:
    cmd = [
        sys.executable,
        "scripts/bench/_archive/e2e.py",
        "--gpus",
        str(args.gpus),
        "--out",
        args.out,
        "--presets",
        preset,
        "--strategies",
        strategy,
        "--run-id",
        f"{preset}:{strategy}",
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--microbatches",
        str(args.microbatches),
        "--tokens-per-step",
        str(args.tokens_per_step),
        "--pp-schedule",
        args.pp_schedule,
    ]
    label = f"{preset:16s} {strategy:10s}"
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.time() - start
    line = next(
        (ln for ln in proc.stdout.splitlines() if "tok/s" in ln or "FAILED" in ln), ""
    )
    ok = proc.returncode == 0 and "tok/s" in line
    print(
        f"  {label} {'OK ' if ok else 'ERR'} ({elapsed:5.0f}s) {line.strip()}",
        flush=True,
    )
    if not ok:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        for ln in tail:
            print(f"      {ln[:150]}", flush=True)
    return ok


def aggregate(args):
    """Merge part files, add the static memory model, and plot."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import e2e as E

    rows = []
    for path in sorted(glob.glob(os.path.join(args.out, "parts", "*.json"))):
        with open(path) as handle:
            rows.extend(json.load(handle))

    from kohakuwullm.models import get_preset

    for preset in args.presets:
        config = get_preset(preset, vocab_size=args.vocab)
        stats = E.estimate_memory(config)
        rows.append(
            {
                "preset": preset,
                "strategy": "memory_model",
                "ok": True,
                "params": stats["params"],
                "ddp_state_gib": stats["ddp_state_gib"],
            }
        )

    with open(os.path.join(args.out, "e2e.json"), "w") as handle:
        json.dump(rows, handle, indent=2)
    args.world = args.gpus
    E.plot(rows, args.out, args)
    print(f"\nwrote {args.out}/e2e.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/train/e2e")
    ap.add_argument(
        "--presets", nargs="+", default=["Nano-200M", "Nano-500M", "MoE-1B-A200M"]
    )
    ap.add_argument(
        "--strategies", nargs="+", default=["ddp", "ddp+ckpt", "pipeline", "pp+ckpt"]
    )
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--tokens-per-step", type=int, default=16384)
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pp-schedule", default="1f1b")
    # Generous: the first run of a (preset, strategy) pays Triton autotuning for
    # every kernel at that shape, on every stage. An earlier 300s budget killed
    # a pipeline run that was working -- all four GPUs at 100% -- purely because
    # it had not finished warming up.
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--gpu-gib", type=float, default=31.4)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)

    if not args.plot_only:
        for preset in args.presets:
            print(f"\n=== {preset} ===", flush=True)
            for strategy in args.strategies:
                try:
                    run_one(preset, strategy, args)
                except subprocess.TimeoutExpired:
                    print(f"  {preset:16s} {strategy:10s} TIMEOUT", flush=True)
    aggregate(args)


if __name__ == "__main__":
    main()
