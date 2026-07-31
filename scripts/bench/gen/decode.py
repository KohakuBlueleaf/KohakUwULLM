"""Decode throughput against the DRAM roofline.

    .venv/bin/python scripts/bench/gen/decode.py --preset Kohaku-MoE-1B

Reports tokens/s and the fraction of the measured 1791 GB/s a decode step achieves.
See docs/guides/generation.md.
"""

import argparse
import json
import os
import time

import torch

from kohakuwullm.bench.model.ladder import census
from kohakuwullm.generation import LocalGenerator
from kohakuwullm.models import LMBackbone, get_preset

PEAK_GBPS = 1791.0


def decode_bytes(rung, batch: int, context: int, elem: int) -> int:
    """Bytes a single decode step must read: active weights plus the KV prefix."""
    weights = rung.compute_active * elem
    kv = 2 * batch * context * rung.kv_heads * rung.head_dim * rung.depth * elem
    return weights + kv


def bench(model, rung, batch, prompt, new_tokens, device, dtype, warmup=2, iters=3):
    """Median tokens/s over ``iters`` cached-decode runs."""
    gen = LocalGenerator(model)
    ids = torch.randint(0, rung_vocab(rung), (batch, prompt), device=device)
    samples = []
    for step in range(warmup + iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype):
            gen.generate(ids, max_new_tokens=new_tokens, temperature=0.0)
        torch.cuda.synchronize()
        if step >= warmup:
            samples.append(time.perf_counter() - start)
    samples.sort()
    seconds = samples[len(samples) // 2]
    return batch * new_tokens / seconds, seconds


def rung_vocab(rung) -> int:
    return get_preset(rung.name).vocab_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="Kohaku-MoE-1B")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--prompt", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--out", default="out/bench/gen/decode")
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    elem = 2
    overrides = {"max_position": 4096}
    if args.depth:
        overrides["depth"] = args.depth
    config = get_preset(args.preset, **overrides)
    rung = census(args.preset, **overrides)
    model = LMBackbone(config).to(device=device, dtype=dtype).eval()

    rows = []
    print(f"{args.preset}  active {rung.compute_active / 1e6:.0f}M  depth {rung.depth}")
    print(f"{'batch':>6} {'tok/s':>9} {'GB/s':>8} {'roofline':>9} {'% peak':>7}")
    for batch in args.batches:
        context = args.prompt + args.new_tokens // 2
        per_step = decode_bytes(rung, batch, context, elem)
        ceiling = batch * PEAK_GBPS * 1e9 / per_step
        try:
            rate, seconds = bench(
                model, rung, batch, args.prompt, args.new_tokens, device, dtype
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{batch:6d}  OOM")
            continue
        achieved = per_step * (rate / batch) / 1e9
        rows.append(
            dict(
                batch=batch,
                tokens_per_s=rate,
                gbps=achieved,
                roofline_tokens_per_s=ceiling,
                frac_peak=achieved / PEAK_GBPS,
                seconds=seconds,
            )
        )
        print(
            f"{batch:6d} {rate:9.0f} {achieved:8.0f} {ceiling:9.0f} "
            f"{100 * achieved / PEAK_GBPS:6.1f}%"
        )

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.preset}.json")
    with open(path, "w") as handle:
        json.dump(
            dict(
                preset=args.preset,
                active_params=rung.compute_active,
                peak_gbps=PEAK_GBPS,
                device=torch.cuda.get_device_name(),
                rows=rows,
            ),
            handle,
            indent=2,
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
