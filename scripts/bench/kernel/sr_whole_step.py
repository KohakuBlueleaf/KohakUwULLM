"""Does grouping the SR writeback help a *whole step*, or only an idle device?

``scripts/bench/kernel/stochastic_round.py`` times the writeback on its own and finds the
grouping worth 6.25x at 200M and 1.01x at 3B: grouped runs at DRAM peak so it
scales with bytes, per-parameter scales with tensor count at ~33 us a launch, and
they cross near 7.3M parameters per tensor. That measurement has the device doing
nothing else, which is the one condition a training step never satisfies.

The open question is whether the freed host issue **overlaps the forward and
backward** rather than being saved outright. If it does, the large rungs gain even
where the isolated sweep says they do not, because the per-parameter launches were
never on the critical path -- the host was ahead of the device anyway. If it does
not, the isolated number stands and the speedup is a small-model effect.

So both arms are timed inside one un-synchronized loop, with events bracketing the
forward-backward and the writeback separately. Nothing synchronizes between them,
which is the point: a timer that empties the queue first cannot see pipelining,
and that is exactly how this repo has drawn a wrong whole-loop conclusion before.

**What this measures and what it leaves out.** The two arms differ only in how the
writeback is dispatched; the forward, backward, data and model are identical. The
optimizer's *own* arithmetic -- moment updates, Muon's Newton-Schulz -- is excluded,
because it is common to both arms and would add a constant. That biases the result
*against* grouping: a real optimizer issues its own launches, so the host is under
more pressure than it is here, and freeing 173-494 launches can only matter more.
Read a gain here as a floor and a null result as genuinely null.

Measured at Kohaku-1.5B, 8192 tokens: **1.0006x**, with the writeback 1.59% of the
step. The isolated number stands and the freed host issue buys nothing here, so
grouping is a small-model effect. Null under a construction biased toward a gain,
which is why the bias paragraph above is not decoration.

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/bench/kernel/sr_whole_step.py \\
        --presets Kohaku-1.5B,Kohaku-MoE-3B --tokens 2048
"""

import argparse
import json
import math
import os
import statistics
import time

import torch

from kohakuwullm.bench import make_info, make_tokens
from kohakuwullm.bench.core.contention import sample_spread
from kohakuwullm.bench.core.timing import device_name
from kohakuwullm.kernels.optim.stochastic_round import stochastic_round_update_
from kohakuwullm.kernels.optim.stochastic_round_grouped import GroupedWriteback
from kohakuwullm.models import LMBackbone, get_preset

# Bytes per parameter held live: the 16-bit weight, its 16-bit gradient, and one
# fp32 update buffer. The optimizer's moments are deliberately absent -- see the
# module docstring -- which is what lets a 2.9B rung fit at all.
LIVE_BYTES = 8


def arm_loop(fwd_bwd, writeback, iters: int) -> tuple[float, list[float], list[float]]:
    """One arm's own un-synchronized loop. Returns (wall per step, model, writeback).

    Synchronized *before* the loop and not inside it, so the queue starts empty and
    then stays as full as training keeps it.
    """
    marks = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(iters)
    ]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for begin, middle, end in marks:
        begin.record()
        fwd_bwd()
        middle.record()
        writeback()
        end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) * 1e3 / iters
    return (
        wall,
        [a.elapsed_time(b) for a, b, _ in marks],
        [b.elapsed_time(c) for _, b, c in marks],
    )


def rotated(arms: dict, fwd_bwd, iters: int, warmup: int, reps: int) -> dict:
    """Time each arm in its own loop, repeating the sequence in rotated orders.

    Two confounds had to be designed out, and each one produced a nonsense number
    first:

    * **Blocked order.** All of arm A then all of arm B puts thermal drift in
      lockstep with arm order; that reported the grouped arm 2% slower while its
      device span was 35% shorter. Rotating the order across reps and taking the
      median removes it.
    * **Queue coupling.** Running the arms round-robin *inside* one loop was worse:
      the host blocks on a full launch queue, so how long an arm appears to take
      depends on how many launches the previous arm left queued. That reported the
      grouped arm **faster than doing no writeback at all**, which is impossible.
      Each arm now gets its own loop with a synchronize before it.

    Only the wall time and the event spans are reported. A host/device split is
    deliberately absent: host wall time here includes blocking on a full queue, so
    "the host took as long as the whole step" is equally true when the device is
    the bottleneck, and cannot be read as host-bound.
    """
    names = list(arms)
    for _ in range(warmup):
        for name in names:
            fwd_bwd()
            arms[name]()
    torch.cuda.synchronize()

    walls = {name: [] for name in names}
    models = {name: [] for name in names}
    writes = {name: [] for name in names}
    for rep in range(reps):
        order = names[rep % len(names) :] + names[: rep % len(names)]
        for name in order:
            wall, model_ms, write_ms = arm_loop(fwd_bwd, arms[name], iters)
            walls[name].append(wall)
            models[name] += model_ms
            writes[name] += write_ms
    return {
        name: {
            "wall_ms": statistics.median(walls[name]),
            "wall_samples": walls[name],
            "model_ms": statistics.median(models[name]),
            "writeback_ms": statistics.median(writes[name]),
            "spread": sample_spread(walls[name]),
        }
        for name in names
    }


def measure(
    preset: str, dtype: torch.dtype, tokens: int, iters: int, reps: int
) -> dict:
    config = get_preset(preset)
    config.grad_ckpt = True  # the 2.9B rung does not fit without it
    model = LMBackbone(config).to("cuda").to(dtype)
    params = [p for p in model.parameters() if p.dtype is dtype]
    updates = [
        torch.empty(p.shape, dtype=torch.float32, device="cuda").normal_(0, 1e-6)
        for p in params
    ]
    detached = [p.detach() for p in params]
    grouped = GroupedWriteback(detached, updates)

    info = make_info(tokens, 64, 512, "cuda")
    ids, labels = make_tokens([info], config.vocab_size, "cuda")

    def fwd_bwd():
        loss, _ = model.loss(ids, labels, info, reduction="mean")
        loss.backward()
        for param in model.parameters():
            param.grad = None

    def per_param():
        offset = 0
        for param, update in zip(detached, updates):
            stochastic_round_update_(param, update, 11, decay=1e-4, rng_offset=offset)
            offset += param.numel()

    arms = rotated(
        {
            "no-writeback": lambda: None,
            "per-parameter": per_param,
            "grouped": lambda: grouped(11, decay=1e-4),
        },
        fwd_bwd,
        iters,
        4,
        reps,
    )

    numel = sum(p.numel() for p in params)
    per, one = arms["per-parameter"], arms["grouped"]
    out = {
        "preset": preset,
        "dtype": str(dtype).replace("torch.", ""),
        "tie_embeddings": bool(config.tie_embeddings),
        "tokens": tokens,
        "tensors": len(params),
        "numel": numel,
        "chunks": grouped.n_chunks,
        "arms": arms,
        "whole_step_speedup": per["wall_ms"] / one["wall_ms"],
        "writeback_share_per_param": (
            (per["wall_ms"] - arms["no-writeback"]["wall_ms"]) / per["wall_ms"]
        ),
    }
    print(
        f"  {preset} {len(params)} tensors {numel / 1e9:.3f}B, {tokens} tokens\n"
        f"    fwd/bwd only     {arms['no-writeback']['wall_ms']:8.2f} ms/step  "
        f"spread {arms['no-writeback']['spread']:5.1%}\n"
        f"    + per-parameter  {per['wall_ms']:8.2f} ms/step  "
        f"adds {per['wall_ms'] - arms['no-writeback']['wall_ms']:7.2f}  "
        f"device span {per['writeback_ms']:6.2f}  spread {per['spread']:5.1%}\n"
        f"    + grouped        {one['wall_ms']:8.2f} ms/step  "
        f"adds {one['wall_ms'] - arms['no-writeback']['wall_ms']:7.2f}  "
        f"device span {one['writeback_ms']:6.2f}  spread {one['spread']:5.1%}\n"
        f"    whole-step speedup {out['whole_step_speedup']:.3f}x, "
        f"writeback is {100 * out['writeback_share_per_param']:.1f}% of the "
        f"per-parameter step"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", default="Kohaku-1.5B,Kohaku-MoE-3B")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="out/bench/kernel/sr")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{device_name()} CUDA_VISIBLE_DEVICES={visible} {total:.0f} GiB")

    rows = []
    for preset in args.presets.split(","):
        with torch.device("meta"):
            probe = LMBackbone(get_preset(preset))
        need = (
            sum(math.prod(p.shape) for p in probe.parameters() if p.is_floating_point())
            * LIVE_BYTES
            / 2**30
        )
        if need > 0.75 * total:
            print(f"  {preset} needs ~{need:.1f} GiB live, skipping")
            continue
        rows.append(measure(preset, torch.bfloat16, args.tokens, args.iters, args.reps))
        torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "device": device_name(),
        "cuda_visible_devices": visible,
        "rows": rows,
    }
    with open(os.path.join(args.out, "sr_whole_step.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {args.out}/sr_whole_step.json")


if __name__ == "__main__":
    main()
