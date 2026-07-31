"""Does the MXFP8 training speedup grow with model size? Three presets, one card.

At Nano-200M the swap wins 1.071x end to end against a 1.203x matmul ceiling. Three
mechanisms predict that both numbers rise with size, and they are separable, which is
the point of measuring all three sizes rather than asserting the trend:

1. **The tied LM head stays bf16 and shrinks as a share.** It costs ``dim * vocab``
   while the body costs ``~dim^2 * depth``, so its 29.9% of matmul MAC/token at 200M
   roughly halves by 1B and the fp8-covered fraction rises from 68.1%. This moves the
   *ceiling*, and is isolated here by holding the per-GEMM speedup fixed at the
   measured 1.59 -- so any ceiling growth reported is this effect alone.
2. **Wider GEMMs favour fp8 specifically.** The vendor path measures 518 TF/s at
   N=1280 against 610 at N=5120 while bf16 sits flat at 229-237, so the per-GEMM
   *ratio* improves 2.26x -> 2.57x. This moves the measured speedup without moving a
   fixed-1.59 ceiling, so it shows up as a rising fraction-of-ceiling.
3. **Dispatch amortises.** Device work per launch grows with width, so a step spends
   proportionally less time issuing. Shows up as a rising fraction-of-ceiling, and is
   told apart from (2) by ``ops_per_step`` against MAC per step.

So the attribution is: ceiling growth is (1); measured-over-ceiling growth is (2)
and/or (3). **A flat curve falsifies (1)** and is worth as much as a rising one, so
nothing here is written to make the trend appear.

**There is no host/device split here, and that is a correction rather than an
omission.** The natural instrument, ``timing_profile``, reported a 98-99% host share
for every row -- which would have marked every speedup inadmissible, for a reason that
turned out not to exist. ``cpu_enqueue_ms`` measures host issue only while the
pending-launch queue has room, and a whole training step issues 1900-2700 ops, so its
30-iteration unsynchronised loop fills the queue and starts blocking on the device. Its
per-iteration "host" time for Nano-200M bf16 rises 69.6 -> 104.9 -> 126.6 -> 133.9 ->
138.9 ms as ``iters`` goes 1 -> 30, converging on the 141.5 ms steady state; a real
host measurement is flat in ``iters``. So ``wall - host`` was not a device time. What
is reported instead:

* ``serialized_ms`` -- one sync per iteration. Models the *current* training loop,
  which spends three ``.item()`` calls per step, and reproduces the A/B's separately
  measured 1.070x at Nano-200M from a different harness.
* ``steady_ms`` -- queue saturated, i.e. a loop whose logging does not sync. fp8 gains
  much more here, because it issues ~39% more ops and so has more host work to hide.
  The serialized-to-steady gap is therefore throughput the *harness* is discarding,
  not throughput the kernels lack.
* ``ops_per_step`` -- a count, immune to queue depth, and the unit a fusion moves.

**One card, recorded.** ``device_name()`` reads index 0 of the *visible* set and
returns a byte-identical string on all four cards, so the JSON records
``CUDA_VISIBLE_DEVICES`` from the environment. The 200M point is **re-measured here**
rather than taken from ``out/bench/train/fp8_training/``, whose card is unrecoverable: that
point is the anchor every ratio is read against, so an unknown ~1% card offset would
land where it does the most damage.

Usage::

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/bench/fp8/fp8_sizes.py
"""

import argparse
import json
import os
import time

import torch

from kohakuwullm.bench.core.batches import make_info
from kohakuwullm.bench.core.timing import (
    bench_ms,
    device_name,
    device_op_counts,
    stream_bandwidths,
)
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.mxfp8_swap import (
    refresh_mxfp8_weights,
    swap_mxfp8,
)

PRESETS = ("Nano-200M", "Nano-500M", "Nano-1B")
VOCAB_SIZE = 65536
TOKENS = 16384
# Document lengths, chosen so the packed batch's mean document lands near the real
# danbooru/TIPO cache's 238.25. Drawing uniform on [50, 426] gives a nominal mean of
# 238; packing to an exact token count and trimming the straddler shifts the achieved
# mean to ~248, which is what the JSON records and what the ceiling is computed from.
# Attention is ~2% of matmul MAC here, so a 4% error in the mean cannot move the
# conclusion -- but the ceiling is derived from the achieved lengths, not the nominal
# ones, so it does not have to.
DOC_LO, DOC_HI = 50, 426
SEED = 20090220
# Per-GEMM MXFP8 speedup over bf16 on sm_120, from scripts/bench/kernel/mxfp8.py. Held
# *fixed* across sizes on purpose: it makes the reported ceiling move only with the
# fp8 MAC share, which isolates mechanism (1) from mechanism (2).
GEMM_SPEEDUP = 1.59
TRIAD_FLOOR_GBPS = 1700.0


def _require_linear_only(swapped: set[str], projections: dict[str, int]) -> None:
    """Refuse a preset whose fp8 matmul this accounting cannot see.

    The census below walks ``nn.Linear`` modules, which is every tensor the swap could
    reach when it was written. A sparse preset's routed experts are *declared* rather
    than type-matched, so they are absent here and their MAC -- 42% of a MoE model's
    per-token arithmetic -- would silently drop out of both the share and the ceiling.
    Raising is the point: an under-reported denominator makes an fp8 speedup look
    better than it is, which is the error a benchmark must not make quietly.
    """
    missing = swapped - set(projections)
    if missing:
        raise RuntimeError(
            f"{len(missing)} fp8 tensor(s) are not `nn.Linear` projections, so this "
            f"accounting cannot charge them: {sorted(missing)[:4]}... Take the MAC "
            "from the swap report's own census instead of a second walk over the model."
        )


def matmul_accounting(config, info) -> dict:
    """MAC per token by category, and the ceiling the swap could reach.

    Attention pairs come from the *timed* batch's own document lengths rather than a
    nominal figure, so the ceiling describes the workload that was actually measured.
    """
    model = LMBackbone(config)
    swapped = set(swap_mxfp8(LMBackbone(config)).swapped)
    projections = {
        name: module.in_features * module.out_features
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    _require_linear_only(swapped, projections)
    fp8 = sum(v for k, v in projections.items() if k in swapped)
    other = sum(v for k, v in projections.items() if k not in swapped)
    head = config.dim * config.vocab_size

    lengths = (info.cu_seqlens[1:] - info.cu_seqlens[:-1]).to(torch.float64)
    # Causal attention over a document of length L costs L(L+1)/2 pairs, so per-token
    # cost is the length-weighted mean of (L+1)/2 -- not (mean L)/2, which understates
    # a packed batch with a long tail.
    pairs = float(((lengths * (lengths + 1) / 2).sum() / lengths.sum()).item())
    attention = 2 * pairs * config.head_dim * config.heads * config.depth
    total = fp8 + other + head + attention
    # fwd + dgrad + wgrad is three products; the swap moves two of the three, and only
    # for its own projections. WGRAD stays high precision.
    ceiling = (3 * total) / (2 * fp8 / GEMM_SPEEDUP + fp8 + 3 * (total - fp8))
    return {
        "mac_per_token": {
            "fp8_projections": fp8,
            "other_projections": other,
            "lm_head": head,
            "attention": attention,
            "total": total,
        },
        "fp8_share_of_matmul": fp8 / total,
        "lm_head_share_of_matmul": head / total,
        "causal_pairs_per_token": pairs,
        "mean_document_tokens": float(lengths.mean().item()),
        "matmul_speedup_ceiling": ceiling,
        "swapped_projections": len(swapped),
    }


def l2_residency(config, tokens: int) -> dict:
    """Which activations fit the 96 MiB L2, per size.

    Recorded because it predicts a kink rather than explaining one afterwards. ``w_in``
    is a *fused* gate+up projection, so the tensor that feeds ``w_out`` is half its
    output width -- and at 16384 tokens that post-SwiGLU intermediate is 79.7 MB at
    Nano-200M and 113.2 MB at Nano-500M, i.e. **the boundary falls between the two
    smallest sizes**, and Nano-200M is the anchor every ratio here is read against.

    The MoE ladder measured the same boundary biting hardest where the *whole*
    intermediate fits (75% of the traffic bound against 81-82% elsewhere), so a
    discontinuity between 200M and 500M is expected. Its direction is not predictable
    from here -- an L2-resident tensor helps whichever dtype is more bandwidth-bound --
    so this is a flag to report the kink, not a correction to apply.
    """
    l2_bytes = 96 * 1024**2
    dim, wide = config.dim, None
    for module in LMBackbone(config).blocks[0].modules():
        if isinstance(module, torch.nn.Linear) and module.out_features not in (
            dim,
            config.vocab_size,
        ):
            wide = max(wide or 0, module.out_features)
    gate_up = tokens * (wide or 0) * 2
    # Fused gate+up: the SwiGLU output, which is w_out's input, is half as wide.
    post_swiglu = gate_up // 2
    return {
        "l2_bytes": l2_bytes,
        "gate_up_act_bytes": gate_up,
        "post_swiglu_act_bytes": post_swiglu,
        "residual_act_bytes": tokens * dim * 2,
        "post_swiglu_fits_l2": post_swiglu <= l2_bytes,
        "gate_up_fits_l2": gate_up <= l2_bytes,
    }


def build(preset: str, arm: str, grad_ckpt: bool, device) -> tuple[LMBackbone, int]:
    """One arm's model, and how many MXFP8 modules it must refresh per step."""
    torch.manual_seed(SEED)
    config = get_preset(preset, vocab_size=VOCAB_SIZE, grad_ckpt=grad_ckpt)
    model = LMBackbone(config, head_kwargs={"kernel": "chunked_ce"}).to(device)
    model.train()
    if arm == "bf16":
        return model, 0
    report = swap_mxfp8(model)
    # `blocking`, not `skipped`: declared-but-unconverted matmul and an unclaimed
    # matmul-shaped parameter are mixtures with zero skipped projections.
    if report.blocking:
        raise RuntimeError(f"{preset}/{arm}: {report.summary()}")
    # Modules, not tensors: an MoE layer's two expert stacks share one fp8 cache.
    return model, len(report.modules)


def steady_state_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    """Queue-saturated step time: no sync inside the loop, only at its ends.

    What a training loop achieves when host issue and device execution overlap.
    :func:`bench_ms` cannot report it -- it synchronizes *before* recording its start
    event, which serializes the two and charges host issue as wall time.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def measure(preset: str, arm: str, tokens: int, grad_ckpt: bool, device) -> dict:
    """One fwd+bwd+refresh, timed in **both** regimes, plus its launch count.

    Two regimes because the answer differs between them and the difference is the
    finding, not noise:

    * ``serialized_ms`` (:func:`bench_ms`) syncs once per iteration, so host issue and
      device execution do not overlap. This models the *current* training loop, which
      calls ``grad_norm.item()`` and ``logs["ce"].item()`` every step -- three syncs
      per step -- and it reproduces the A/B's independently measured 1.070x.
    * ``steady_ms`` lets the queue saturate, which is what a loop without per-step
      syncs achieves. fp8 gains far more here because it issues ~39% more ops, so it
      has more host work to hide under the device.

    **There is deliberately no host/device split.** ``cpu_enqueue_ms`` measures host
    issue only while the pending-launch queue has room; a whole training step issues
    ~1900-2700 ops, so a 30-iteration unsynchronised loop fills the queue and the
    "host" number becomes device time. Measured on Nano-200M bf16, per-iteration
    "host" time rises 69.6 -> 104.9 -> 126.6 -> 133.9 -> 138.9 ms as ``iters`` goes
    1 -> 30, converging on the 141.5 ms steady state; a real host measurement would be
    flat. So ``wall - host`` here is not a device time and a host share computed from
    it is not a host share. ``ops_per_step`` is reported instead -- it is a count, it
    cannot be contaminated this way, and it is the unit a fusion actually moves.
    """
    model, expect_refresh = build(preset, arm, grad_ckpt, device)
    info = make_info(tokens, DOC_LO, DOC_HI, device, seed=SEED)
    ids = torch.randint(0, VOCAB_SIZE, (tokens,), device=device)
    labels = ids.clone()

    def step():
        with torch.autocast("cuda", torch.bfloat16):
            loss, _ = model.loss(ids, labels, info, reduction="sum")
        (loss / tokens).backward()
        model.zero_grad(set_to_none=True)
        refreshed = refresh_mxfp8_weights(model)
        if refreshed != expect_refresh:
            raise RuntimeError(f"refreshed {refreshed}, expected {expect_refresh}")

    torch.cuda.reset_peak_memory_stats()
    counts = device_op_counts(step)
    ops = int(counts["total"]) if isinstance(counts, dict) else int(counts)
    return {
        "arm": arm,
        "serialized_ms": float(bench_ms(step, warmup=3, iters=10)),
        "steady_ms": steady_state_ms(step),
        "ops_per_step": ops,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "mxfp8_modules": expect_refresh,
    }


def token_sweep(preset: str, counts: tuple[int, ...], out_dir: str) -> dict:
    """Step time against tokens/step for one preset, both arms.

    The mechanism evidence, and the reason the size trend exists at all. fp8's step
    time is *flat* in tokens until the device work exceeds the launch cost -- 2048 to
    8192 tokens costs it 1.4% more time -- so below that point it is dispatch-bound and
    a 4x larger batch is free. bf16, issuing ~750 fewer ops, is device-bound much
    earlier. What gates MXFP8 is therefore device work per launch, which both more
    tokens and more width increase; size scaling and token scaling are one axis.

    Deliberately reports the measured bracket rather than a fitted crossover. A least
    squares ``fixed + marginal * tokens`` through a flat-then-rising curve is the wrong
    model and returned a crossover of 16,479 tokens -- close enough to the shipped
    16,296 to look like a finding, and an artifact of the fit.
    """
    device = torch.device("cuda")

    def sweep_arm(arm: str) -> tuple[dict[int, float], dict | None]:
        """One arm's curve. A function so its model is local and freed on return --
        holding two 1B models alive across the loop would OOM the larger presets."""
        model, expect = build(preset, arm, False, device)
        refresh = None
        if expect:
            ops = device_op_counts(lambda: refresh_mxfp8_weights(model))
            refresh = {
                "ms": steady_state_ms(lambda: refresh_mxfp8_weights(model)),
                "ops": int(ops["total"]) if isinstance(ops, dict) else int(ops),
                "modules": expect,
            }
        times: dict[int, float] = {}
        for tokens in counts:
            info = make_info(tokens, DOC_LO, DOC_HI, device, seed=SEED)
            ids = torch.randint(0, VOCAB_SIZE, (tokens,), device=device)

            def step():
                with torch.autocast("cuda", torch.bfloat16):
                    loss, _ = model.loss(ids, ids, info, reduction="sum")
                (loss / tokens).backward()
                model.zero_grad(set_to_none=True)
                refresh_mxfp8_weights(model)

            try:
                times[tokens] = steady_state_ms(step)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            print(
                f"  {arm:<8}{tokens:>7} tokens {times[tokens]:8.2f} ms "
                f"{times[tokens] / tokens * 1e3:7.2f} us/token",
                flush=True,
            )
        return times, refresh

    rows: dict[str, dict[int, float]] = {}
    refresh = None
    for arm in ("bf16", "fp8_up"):
        rows[arm], arm_refresh = sweep_arm(arm)
        refresh = arm_refresh or refresh
        torch.cuda.empty_cache()

    shared = sorted(set(rows["bf16"]) & set(rows["fp8_up"]))
    ratios = {t: rows["bf16"][t] / rows["fp8_up"][t] for t in shared}
    below = [t for t in shared if ratios[t] < 1.0]
    above = [t for t in shared if ratios[t] >= 1.0]
    report = {
        "preset": preset,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "steady_ms": {
            arm: {str(k): v for k, v in data.items()} for arm, data in rows.items()
        },
        "speedup": {str(k): v for k, v in ratios.items()},
        "refresh_alone": refresh,
        "crossover_bracket": [
            max(below) if below else None,
            min(above) if above else None,
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "token_sweep.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    lo, hi = report["crossover_bracket"]
    print(
        f"\n{preset}: fp8 loses at or below {lo} tokens/step and wins at or above {hi}"
        if lo and hi
        else f"\n{preset}: no crossover inside {shared}"
    )
    for t in shared:
        print(f"  {t:>7} tokens: {ratios[t]:.4f}x")
    if refresh:
        print(
            f"  weight refresh alone: {refresh['ms']:.2f} ms over {refresh['ops']} ops "
            f"for {refresh['modules']} modules"
        )
    return report


def run(
    tokens: int, grad_ckpt: bool, out_dir: str, presets: tuple[str, ...] = PRESETS
) -> dict:
    device = torch.device("cuda")
    bandwidths = stream_bandwidths()
    if bandwidths["triad"] < TRIAD_FLOOR_GBPS:
        raise RuntimeError(
            f"triad {bandwidths['triad']:.0f} GB/s below {TRIAD_FLOOR_GBPS:.0f}: "
            "the card is contended, so nothing measured here would be a measurement"
        )

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        f"{device_name()} | CUDA_VISIBLE_DEVICES={visible} | {tokens} tokens "
        f"| grad_ckpt={grad_ckpt} | triad {bandwidths['triad']:.0f} GB/s",
        flush=True,
    )
    rows = []
    for preset in presets:
        config = get_preset(preset, vocab_size=VOCAB_SIZE)
        info = make_info(tokens, DOC_LO, DOC_HI, torch.device("cpu"), seed=SEED)
        row = {
            "preset": preset,
            "dim": config.dim,
            "depth": config.depth,
            "tokens": tokens,
            "accounting": matmul_accounting(config, info),
            "l2": l2_residency(config, tokens),
        }
        for arm in ("bf16", "fp8_up"):
            try:
                row[arm] = measure(preset, arm, tokens, grad_ckpt, device)
            except torch.cuda.OutOfMemoryError:
                # Recorded rather than raised: one size failing to fit must not
                # discard the sizes that did, and "OOM at this width" is itself a
                # ladder fact worth keeping.
                torch.cuda.empty_cache()
                row[arm] = {"arm": arm, "oom": True}
                print(f"  {preset}/{arm}: OOM at {tokens} tokens", flush=True)
                continue
            torch.cuda.empty_cache()
            m = row[arm]
            print(
                f"  {preset:<11}{arm:<8}serial {m['serialized_ms']:7.2f}  "
                f"steady {m['steady_ms']:7.2f} ms  ops {m['ops_per_step']:5d}  "
                f"peak {m['peak_memory_gib']:.2f} GiB",
                flush=True,
            )
        rows.append(summarize(row))

    report = {
        "device": device_name(),
        "visible_devices": visible,
        "tokens": tokens,
        "grad_ckpt": grad_ckpt,
        "gemm_speedup_assumed": GEMM_SPEEDUP,
        "triad_gbps": bandwidths["triad"],
        "sizes": rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fp8_sizes.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    return report


def summarize(row: dict) -> dict:
    """Attach both speedups and the ceiling fractions, or the reason there are none."""
    bf16, fp8 = row.get("bf16", {}), row.get("fp8_up", {})
    if bf16.get("oom") or fp8.get("oom"):
        row["speedup_serialized"] = None
        row["speedup_steady"] = None
        row["reason"] = "OOM"
        return row
    ceiling = row["accounting"]["matmul_speedup_ceiling"]
    for name, key in (("serialized", "serialized_ms"), ("steady", "steady_ms")):
        ratio = bf16[key] / fp8[key]
        row[f"speedup_{name}"] = ratio
        row[f"fraction_of_ceiling_{name}"] = (ratio - 1) / (ceiling - 1)
    # Launch count per unit of device work is the dispatch-amortisation evidence, and
    # unlike a host share it is a count that cannot be contaminated by queue depth.
    row["ops_ratio_fp8_over_bf16"] = fp8["ops_per_step"] / bf16["ops_per_step"]
    row["mac_per_op_bf16"] = (
        row["accounting"]["mac_per_token"]["total"]
        * row["tokens"]
        / bf16["ops_per_step"]
    )
    row["reason"] = ""
    return row


def report_table(report: dict) -> None:
    print(
        f"\n{'preset':<12}{'fp8 MAC':>9}{'head':>7}{'ceil':>7}"
        f"{'serial':>9}{'of ceil':>9}{'steady':>9}{'of ceil':>9}"
        f"{'ops bf16':>10}{'ops fp8':>9}{'L2':>6}"
    )
    for row in report["sizes"]:
        acc = row["accounting"]
        if row["speedup_serialized"] is None:
            print(f"{row['preset']:<12}{'':>32}{row['reason']:>9}")
            continue
        print(
            f"{row['preset']:<12}"
            f"{acc['fp8_share_of_matmul']:>8.1%}{acc['lm_head_share_of_matmul']:>7.1%}"
            f"{acc['matmul_speedup_ceiling']:>7.3f}"
            f"{row['speedup_serialized']:>9.4f}{row['fraction_of_ceiling_serialized']:>9.1%}"
            f"{row['speedup_steady']:>9.4f}{row['fraction_of_ceiling_steady']:>9.1%}"
            f"{row['bf16']['ops_per_step']:>10d}{row['fp8_up']['ops_per_step']:>9d}"
            f"{'fits' if row['l2']['post_swiglu_fits_l2'] else '--':>6}"
        )
    usable = [r for r in report["sizes"] if r["speedup_serialized"] is not None]
    if len(usable) >= 2:
        first, last = usable[0], usable[-1]
        print(
            f"\n{first['preset']} -> {last['preset']}: "
            f"serialized {first['speedup_serialized']:.4f}x -> "
            f"{last['speedup_serialized']:.4f}x, "
            f"steady {first['speedup_steady']:.4f}x -> {last['speedup_steady']:.4f}x, "
            f"ceiling {first['accounting']['matmul_speedup_ceiling']:.3f} -> "
            f"{last['accounting']['matmul_speedup_ceiling']:.3f}"
        )
        print(
            "  ceiling growth is the shrinking bf16 LM head (mechanism 1); "
            "fraction-of-ceiling growth is wider GEMMs and/or dispatch amortising; "
            "the serialized/steady gap is overlap fp8 can exploit and bf16 cannot, "
            "which the current training loop throws away with three .item() per step"
        )
        # Only a *split* matters. When every rung is L2-resident, or none is, the
        # boundary is not inside the comparison and saying "read this as a
        # discontinuity" would invent one -- which the per-row version of this line did
        # at 8192 tokens, where all three fit and it fired three times.
        fits = [r for r in usable if r["l2"]["post_swiglu_fits_l2"]]
        if fits and len(fits) < len(usable):
            names = ", ".join(
                f"{r['preset']} ({r['l2']['post_swiglu_act_bytes'] / 1e6:.0f} MB)"
                for r in fits
            )
            print(
                f"  L2 boundary falls inside this sweep: {names} fit the 96 MiB L2 and "
                "the larger rungs do not, so that step is a discontinuity, not a trend"
            )
        elif fits:
            print(
                f"  all {len(usable)} rungs are L2-resident at this token count "
                "(post-SwiGLU under 96 MiB), so the boundary is not inside the sweep"
            )
        else:
            print(
                "  no rung is L2-resident at this token count, so the boundary is not "
                "inside the sweep"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument(
        "--grad-ckpt",
        action="store_true",
        help="uniformly across all sizes, never per-size: it changes the fwd/bwd "
        "balance, so a mixed setting would make the sizes incomparable",
    )
    parser.add_argument("--out", default="out/bench/train/fp8_sizes")
    parser.add_argument(
        "--token-sweep",
        metavar="PRESET",
        help="instead of the size sweep, measure step time vs tokens/step for one "
        "preset -- the mechanism evidence behind the size trend",
    )
    parser.add_argument(
        "--token-counts",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192, 16384, 24576],
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(PRESETS),
        help="subset to run, for probing the memory limit of the largest cheaply",
    )
    args = parser.parse_args()
    if args.token_sweep:
        token_sweep(args.token_sweep, tuple(args.token_counts), args.out)
        return
    report_table(run(args.tokens, args.grad_ckpt, args.out, tuple(args.presets)))


if __name__ == "__main__":
    main()
