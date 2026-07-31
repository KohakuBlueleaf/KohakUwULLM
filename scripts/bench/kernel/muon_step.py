"""Where Muon's per-step cost goes: dispatch, elementwise, or Newton-Schulz FLOPs.

The Newton-Schulz iteration's arithmetic does not scale with the batch. It is
``~20 m^2 n`` per hidden matrix per optimizer step whatever the token count, so
whether Muon is expensive is a question about the *model*, not the step size, and
whether it is launch-bound or FLOP-bound cannot be read off the parameter count.

Four independent readings, because each alone is ambiguous:

* the whole-loop split -- fwd/bwd and ``opt.step()`` bracketed by CUDA events
  inside one un-synchronized loop, so the optimizer is measured in the pipelined
  state a real training step leaves it in;
* the analytic Newton-Schulz FLOP count against the measured bf16 ceiling;
* a **matmul-only floor**: the same matmuls at the same shapes with the
  elementwise terms deleted, which is the fastest the current schedule could
  possibly run;
* the host issue ladder, over several loop lengths, because a host measurement
  that grows with ``iters`` has stopped measuring the host (the launch queue
  filled and the loop began blocking on the device).

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench/kernel/muon_step.py \
        --preset Nano-500M --tokens 16384
"""

import argparse
import json
import os
import statistics
import time

import torch

from kohakuwullm.bench import make_info, make_tokens
from kohakuwullm.bench.core.timing import (
    TENSOR_MAMF_TFLOPS,
    contended_fraction_limit,
    device_op_counts,
    graph_ms,
    sample_contended_fraction,
    sample_spread,
)
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.training.optim.build import build_optimizer, group_parameters


def matrix_shapes(optimizer) -> list[tuple[tuple[int, ...], int]]:
    """``(shape, count)`` for every parameter in a ``use_muon`` group."""
    counts: dict[tuple[int, ...], int] = {}
    for group in optimizer.param_groups:
        if not group.get("use_muon"):
            continue
        for param in group["params"]:
            key = tuple(param.shape)
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1] * _numel(kv[0]))


def _numel(shape: tuple[int, ...]) -> int:
    out = 1
    for size in shape:
        out *= size
    return out


def ns_flops(shape: tuple[int, ...], phases: tuple[int, ...]) -> float:
    """Newton-Schulz matmul FLOPs for one matrix of ``shape``.

    The iteration runs on the short side (the polar factor of the transpose is
    the transpose), so ``m`` is ``min`` of the last two dimensions -- charging
    the long side would overstate a skinny k/v projection by its aspect ratio
    cubed.

    ``P`` phase groups over ``S`` steps cost ``4 P m^2 n + 6 (S - P) m^3``: each
    group forms one gram and applies one accumulated product, and the steps
    inside it are three ``m x m`` matmuls each.
    """
    m, n = sorted(shape[-2:])
    batch = _numel(shape) // (m * n)
    groups, steps = len(phases), sum(phases)
    return batch * 2.0 * (2 * groups * m * m * n + 3 * (steps - groups) * m**3)


def total_ns_flops(shapes, phases_of) -> float:
    return sum(count * ns_flops(shape, phases_of(shape)) for shape, count in shapes)


def loop_split(fwd_bwd, opt_step, iters: int, warmup: int) -> dict:
    """Per-step wall time and the fwd/bwd vs optimizer split, from one loop.

    No synchronization inside the loop, so the launch queue stays full exactly as
    it does in training and the event spans measure the device timeline rather
    than a queue-empty cold start. A microbenchmark of ``opt.step()`` alone
    cannot see that state, and this repo has twice drawn a whole-loop conclusion
    from one that could not.
    """
    marks = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(iters)
    ]
    for _ in range(warmup):
        fwd_bwd()
        opt_step()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for begin, middle, end in marks:
        begin.record()
        fwd_bwd()
        middle.record()
        opt_step()
        end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) * 1e3 / iters

    model_ms = [a.elapsed_time(b) for a, b, _ in marks]
    opt_ms = [b.elapsed_time(c) for _, b, c in marks]
    step_ms = [a.elapsed_time(c) for a, _, c in marks]
    return {
        "wall_ms": wall,
        "model_ms": statistics.median(model_ms),
        "opt_ms": statistics.median(opt_ms),
        "step_ms": statistics.median(step_ms),
        "opt_share": statistics.median(opt_ms) / statistics.median(step_ms),
        "spread": sample_spread(step_ms),
        "contended_fraction": sample_contended_fraction(step_ms),
        "samples": step_ms,
    }


def host_ladder(fn, ladder=(1, 2, 4, 8, 16, 32), warmup: int = 3) -> dict:
    """Per-call host issue time at several loop lengths.

    ``cpu_enqueue_ms`` measures host issue *only while the launch queue has
    room*. Past that the loop blocks on the device and the number silently
    becomes device time -- which is how a 93% "host share" was once read off a
    step that was not host-bound at all. A genuine host measurement is flat in
    ``iters``, so the ladder is returned rather than a single number.
    """
    out = {}
    for iters in ladder:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        elapsed = (time.perf_counter() - start) * 1e3 / iters
        torch.cuda.synchronize()
        out[iters] = elapsed
    return out


def host_is_flat(ladder: dict, tolerance: float = 0.25) -> bool:
    """Whether the ladder's growth is small enough to still be host time."""
    values = [ladder[k] for k in sorted(ladder)]
    return max(values) <= min(values) * (1.0 + tolerance)


def matmul_floor(shapes, phases_of, batch_elems: int, dtype) -> dict:
    """Time the Newton-Schulz matmuls alone, at the shapes and batching Muon uses.

    The floor the arm's arithmetic cannot beat: same matmul sequence, same
    chunking and same phase grouping, with every elementwise term and every
    momentum op deleted. Muon's measured time over this is the share that is
    *not* the matmuls, and it is the only way to tell "the GEMMs are slow at
    these shapes" from "the GEMMs are fine and the overhead is elsewhere".
    """
    chunks = []
    for shape, count in shapes:
        per_chunk = max(1, batch_elems // max(_numel(shape), 1))
        left = count
        while left > 0:
            take = min(per_chunk, left)
            m, n = sorted(shape[-2:])
            lead = (take,) + shape[:-2] if take > 1 else shape[:-2]
            chunks.append(
                (
                    torch.randn(lead + (m, n), dtype=dtype, device="cuda"),
                    phases_of(shape),
                )
            )
            left -= take

    def run():
        for x, phases in chunks:
            for length in phases:
                gram = x @ x.mT
                product = None
                for offset in range(length):
                    product = gram if product is None else gram @ product
                    if offset + 1 < length:
                        gram = gram @ gram @ gram
                x = product @ x

    ms = _median_ms(run)
    chunks.clear()
    torch.cuda.empty_cache()
    return {"ms": ms}


def _median_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        begin.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(begin.elapsed_time(end))
    return statistics.median(times)


def _square_matmul_ms(size: int, dtype, iters: int) -> float:
    """Time one square matmul, in a frame that owns its operands.

    Separate from its caller so the two 128 MiB operands are freed by the frame
    exiting. Dropping them with ``del`` (or rebinding them to ``None``) inside the
    caller unbinds names the timing closure captures, which is a ruff ``F821`` and
    a real hazard: the closure would raise if it outlived the free.
    """
    left = torch.randn(size, size, dtype=dtype, device="cuda")
    right = torch.randn(size, size, dtype=dtype, device="cuda")
    return _median_ms(lambda: left @ right, iters=iters)


def bf16_gemm_ceiling(size: int = 8192, iters: int = 20) -> float:
    """Measured bf16 matmul TFLOP/s on this card, for the % of peak denominator.

    Measured rather than taken from ``TENSOR_MAMF_TFLOPS``: the ratio a Muon
    result is judged against decides whether "FLOP-bound" is a verdict or a
    guess, and a datasheet-derived ceiling has already produced two impossible
    numbers in this repo.
    """
    ms = _square_matmul_ms(size, torch.bfloat16, iters)
    torch.cuda.empty_cache()
    return 2.0 * size**3 / (ms / 1e3) / 1e12


def muon_factory(args, compile_ns: bool | None, cls=None, **extra):
    """An optimizer factory for one Muon arm, for :func:`measure_arm`.

    ``cls`` is a class, not a registry name, so an A/B can hold a second
    implementation that deliberately does not claim the ``muon`` name. It goes
    through ``group_parameters`` the same way ``build_optimizer`` does, so the
    two arms differ only in the optimizer.
    """
    kwargs = {"ns_variant": args.ns_variant, "compile_ns": compile_ns, **extra}
    if args.ns_batch_elems:
        kwargs["ns_batch_elems"] = args.ns_batch_elems

    def factory(model):
        if cls is None:
            return build_optimizer(
                model, name="muon", lr=args.lr, muon_lr=args.muon_lr, **kwargs
            )
        groups = group_parameters(
            model, args.weight_decay, lr=args.lr, muon_lr=args.muon_lr
        )
        return cls(groups, lr=args.lr, **kwargs)

    return factory


def adamw_factory(args):
    def factory(model):
        return build_optimizer(model, name="adamw", lr=args.lr)

    return factory


def measure_arm(cfg, args, factory, tag: str) -> dict:
    """One arm: a real step loop, then the optimizer alone from that same state."""
    torch.manual_seed(1234)
    model = LMBackbone(cfg).to(device="cuda", dtype=torch.float32)
    opt = factory(model)
    info = make_info(args.tokens, args.len_min, args.len_max, "cuda", args.seed)
    ids, labels = make_tokens([info], cfg.vocab_size, "cuda", args.seed)

    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model.loss(ids, labels, info, reduction="mean")
        loss.backward()

    row = {"arm": tag, "optimizer": type(opt).__name__}
    row.update(loop_split(fwd_bwd, opt.step, args.iters, args.warmup))

    # Grads are already populated and shapes never change, so the optimizer can
    # be re-timed on its own without another backward.
    row["host_ladder"] = host_ladder(opt.step)
    row["host_flat"] = host_is_flat(row["host_ladder"])
    row["host_ms"] = min(row["host_ladder"].values())
    row["graph_ms"] = graph_ms(opt.step, warmup=3, iters=20)
    try:
        counts = device_op_counts(opt.step)
        row["launches"] = counts["total"]
        row["kernels"] = counts["kernel"]
    except RuntimeError as exc:
        row["launches"] = None
        row["launch_error"] = str(exc)

    shapes = matrix_shapes(opt)
    row["shapes"] = [
        {"shape": list(shape), "count": count, "elems": _numel(shape)}
        for shape, count in shapes
    ]
    row["muon_params"] = sum(count * _numel(shape) for shape, count in shapes)
    if shapes:
        # The phases an arm actually uses, read off the optimizer rather than
        # recomputed from its arguments: an arm whose selection rule differs from
        # this script's idea of it would otherwise be scored against the wrong
        # floor and look inefficient.
        phases = getattr(opt, "_phases", {})
        row["phases"] = {str(list(k)): list(v) for k, v in phases.items()}

        def phases_of(shape, phases=phases):
            return phases.get(shape, (1,) * args.ns_steps)

        flops = total_ns_flops(shapes, phases_of)
        row["ns_flops"] = flops
        row["ns_tflops"] = flops / (row["opt_ms"] / 1e3) / 1e12
        row["matmul_floor_ms"] = matmul_floor(
            shapes, phases_of, opt.ns_batch_elems, torch.bfloat16
        )["ms"]
        row["floor_tflops"] = flops / (row["matmul_floor_ms"] / 1e3) / 1e12
    return row


def report(row: dict, ceiling: float) -> None:
    print(f"  {row['arm']:>22}: step {row['step_ms']:7.1f} ms  ", end="")
    print(
        f"model {row['model_ms']:7.1f}  opt {row['opt_ms']:7.1f} "
        f"({100 * row['opt_share']:4.1f}% of step)"
    )
    print(
        f"  {'':>22}  host {row['host_ms']:7.1f} ms"
        f" ({'flat' if row['host_flat'] else 'NOT FLAT -- queue filled'})"
        f"  graph {row['graph_ms']:7.1f} ms  launches {row['launches']}"
    )
    if "ns_tflops" in row:
        print(
            f"  {'':>22}  NS {row['ns_flops'] / 1e12:6.2f} TFLOP -> "
            f"{row['ns_tflops']:6.1f} TF/s ({100 * row['ns_tflops'] / ceiling:4.1f}% "
            f"of {ceiling:.0f}T)  matmul floor {row['matmul_floor_ms']:7.1f} ms "
            f"({row['floor_tflops']:6.1f} TF/s)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/train/muon")
    ap.add_argument("--preset", default="Nano-500M")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--len-min", type=int, default=50)
    ap.add_argument("--len-max", type=int, default=600)
    ap.add_argument("--ns-variant", default="cubic5", choices=["cubic5", "quintic"])
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--ns-batch-elems", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--arms", nargs="+", default=["adamw", "eager", "compiled"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = get_preset(args.preset, vocab_size=args.vocab, tie_embeddings=True)
    cfg.max_position = max(cfg.max_position, args.tokens)
    cfg.grad_ckpt = args.grad_ckpt
    ceiling = bf16_gemm_ceiling()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        f"{args.preset}  {args.tokens} tok/step  ns={args.ns_variant}  "
        f"CUDA_VISIBLE_DEVICES={visible}\n"
        f"  measured bf16 gemm ceiling {ceiling:.1f} TF/s "
        f"(TENSOR_MAMF_TFLOPS says {TENSOR_MAMF_TFLOPS:.0f})",
        flush=True,
    )

    plans = {
        "adamw": adamw_factory(args),
        "eager": muon_factory(args, False),
        "compiled": muon_factory(args, True),
        # Grouping disabled: isolates the elementwise fusions from the change in
        # the iteration's arithmetic.
        "ungrouped": muon_factory(args, True, gram_aspect=float("inf")),
    }
    rows = []
    for arm in args.arms:
        row = measure_arm(cfg, args, plans[arm], arm)
        row["contended_limit"] = contended_fraction_limit()
        rows.append(row)
        report(row, ceiling)
        torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"{args.preset}_{args.tokens}"
    path = os.path.join(args.out, f"mechanism_{tag}.json")
    with open(path, "w") as handle:
        json.dump(
            {
                "preset": args.preset,
                "tokens": args.tokens,
                "ns_variant": args.ns_variant,
                "gemm_ceiling_tflops": ceiling,
                "visible_devices": visible,
                "rows": rows,
            },
            handle,
            indent=2,
        )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
