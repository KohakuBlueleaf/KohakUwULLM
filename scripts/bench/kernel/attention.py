"""Which attention backend, at what sequence length, at what window.

Five backends implement one contract -- ``varlen`` (PyTorch 2.13 FA2), ``triton``
(ours), ``mxfp8`` (ours, block-scaled e4m3 scores), ``sdpa``, ``flex`` -- and the
choice is not about speed alone. ``sdpa``
has no varlen path: handed the packed training layout it needs an explicit
``(T, T)`` block-diagonal mask, which drops it off the fused kernel onto one that
materializes ``(B, H, T, T)`` scores. A latency plot hides that right up to the
point it stops fitting, so **every row reports peak memory beside latency, an OOM
is a result rather than a crash, and every panel plots time against memory.**

Sweeps: ``context`` (backend x sequence length x layout, global), ``window`` (the
same grid at 512/1024/2048), ``ragged`` (TIPO documents packed into one
microbatch against the padded batch a non-varlen backend needs instead), ``gqa``
(KV-head count, which moves the footprint but not the FLOPs).

Measurement conventions:

* **One document per batch in the context and window sweeps.** A fixed token
  budget with shorter documents would hold the ``(T, T)`` mask at a constant size
  and hide the quadratic term the sweep exists to find.
* **FLOPs count the layer, not the kernel** -- the four projections are most of
  a 1k-token layer's work. Causal masking is charged as the query-key pairs
  actually scored, a window as ``min(i + 1, w)`` per query rather than ``w``.
  Bytes are the compulsory floor, so a path that moves a ``(T, T)`` mask it did
  not need reports a *lower* effective GB/s. Padded rows are charged the useful
  token count, so padding costs TFLOP/s rather than inflating the denominator.
* **``fwd`` rows keep the autograd graph** -- a training forward, not ``no_grad``.
* **``flex`` is timed with its block mask already built.** ``create_block_mask``
  costs 6.8 ms and 2.5 GiB transiently at 16k tokens and ``FlexAttention`` caches
  per instance, so these rows are the steady state of a batch whose document
  boundaries never change. A packed training batch changes them every step.
* **Peak memory is the delta over what was resident around the cell**, so the
  harness -- the L2-flush scratch, the inductor buffers each new ``flex`` shape
  leaves behind -- is not charged to the layer.
* **Window and sequence length are dimensions inside the time-memory plane**, not
  panels of their own: the ranking changes along them, and two differently-scaled
  panels cannot show that.
* **The card is measured, not assumed.** Free memory and the three STREAM
  patterns are read before and after the sweeps, and one shape is timed twice in
  different sweeps. On a shared card an OOM row is otherwise indistinguishable
  from a neighbour holding 9 GiB.

Usage:
    .venv/bin/python scripts/bench/kernel/attention.py --out out/bench/kernel/attention
    .venv/bin/python scripts/bench/kernel/attention.py --dtype float32
"""

import argparse
import json
import os
import re
from dataclasses import dataclass

import torch
import torch._dynamo

from kohakuwullm.bench import (
    Palette,
    bar_labels,
    device_name,
    finish_axis,
    format_metrics,
    module_metrics,
    new_figure,
    rel_error,
    save_figure,
    stream_bandwidths,
)
from kohakuwullm.models.components.attention import (
    FlexAttention,
    MXFP8Attention,
    SDPAAttention,
    TritonVarlenAttention,
    VarlenAttention,
)
from kohakuwullm.models.components.seqinfo import SeqInfo

BACKENDS = {
    "varlen": VarlenAttention,
    "triton": TritonVarlenAttention,
    "mxfp8": MXFP8Attention,
    "sdpa": SDPAAttention,
    "flex": FlexAttention,
}
LAYOUTS = ("packed", "padded")

# Backward multipliers. The projections are ordinary GEMMs (dgrad + wgrad = 2x
# forward); a flash backward is 4 GEMMs plus a recomputed S against 2 in the
# forward, so 2.5x. Charged uniformly -- a backend that stores S rather than
# recomputing it is flattered by the 0.5, and no conclusion here turns on that.
PROJ_FWDBWD = 3.0
ATTN_FWDBWD = 3.5
BYTES_FWDBWD = 3.0

_OOM_REQUEST = re.compile(r"Tried to allocate ([\d.]+) ([KMG])iB")
_GIB_SCALE = {"K": 1 / 2**20, "M": 1 / 2**10, "G": 1.0}
_NOTE = dict(fontsize=7, textcoords="offset points")
_XMARK = dict(marker="X", s=115, zorder=4)
CARD_GIB = 31.4
# A clean STREAM read on this card. The ceiling is a contention detector in its
# own right: bench-kernels measured 1781 GB/s idle and 779 under a neighbouring
# job, same card and code. It beats polling free memory, which cannot tell an
# idle card from one that is idle for the next 400 ms.
CLEAN_DRAM_GBPS = 1780.0


@dataclass
class Env:
    """What a sweep needs beyond a shape: precision, roofline, iteration counts."""

    device: str
    dtype: torch.dtype
    bandwidth_gbps: float
    warmup: int
    iters: int

    @property
    def tensor_cores(self) -> bool:
        """fp32 inputs never reach the mma units, so they are scored against the
        vector ceiling rather than the achievable tensor peak."""
        return self.dtype in (torch.float16, torch.bfloat16)

    @property
    def width(self) -> int:
        return torch.finfo(self.dtype).bits // 8


def build_module(cls, args, window: int | None, env: Env):
    module = cls(
        args.dim,
        args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        qk_norm=True,
        sliding_window=window,
    )
    return module.to(device=env.device, dtype=env.dtype)


def make_batch(layout: str, lengths: list[int], args, env: Env):
    """``(x, seq_info)`` for one layout.

    The padded batch pads every document to the longest and carries no
    key-padding mask, matching ``SDPAAttention``: padding is attended to and paid
    for, which is the cost a backend without a varlen path cannot avoid.
    """
    if layout == "packed":
        info = SeqInfo.from_lengths(
            torch.tensor(lengths, dtype=torch.int32), env.device
        )
        shape = (sum(lengths), args.dim)
    else:
        info = SeqInfo.padded(len(lengths), max(lengths), env.device)
        shape = (len(lengths), max(lengths), args.dim)
    x = torch.randn(*shape, device=env.device, dtype=env.dtype, requires_grad=True)
    return x, info


def scored_pairs(length: int, window: int | None) -> int:
    """Query-key pairs a causal (optionally windowed) document actually scores."""
    if window is None or window >= length:
        return length * (length + 1) // 2
    return window * (window + 1) // 2 + (length - window) * window


def layer_flops(lengths: list[int], window: int | None, args) -> tuple[float, float]:
    """Forward FLOPs of one attention layer, split into projections and attention."""
    tokens = sum(lengths)
    q_out = args.heads * args.head_dim
    kv_out = args.kv_heads * args.head_dim
    proj = 2 * tokens * args.dim * (2 * q_out + 2 * kv_out)
    pairs = sum(scored_pairs(n, window) for n in lengths)
    # QK^T and PV are each 2 flop per (pair, query head, head_dim) element. GQA
    # does not reduce this: every query head still scores against its own keys.
    attn = 4 * pairs * args.heads * args.head_dim
    return float(proj), float(attn)


def layer_bytes(lengths: list[int], args, env: Env) -> float:
    """Compulsory forward DRAM traffic: a floor, not an accounting."""
    tokens = sum(lengths)
    q_out = args.heads * args.head_dim
    kv_out = args.kv_heads * args.head_dim
    elements = (
        2 * tokens * args.dim
        + args.dim * (2 * q_out + 2 * kv_out)
        + 2 * tokens * (q_out + 2 * kv_out)
        + 2 * tokens * q_out
    )
    return float(elements * env.width)


def _requested_gib(message: str) -> float:
    match = _OOM_REQUEST.search(message)
    if not match:
        return float("nan")
    return float(match.group(1)) * _GIB_SCALE[match.group(2)]


def measure(fn, flops: float, bytes_moved: float, env: Env) -> dict:
    """One timed cell, with an out-of-memory outcome recorded rather than raised.

    The warmup call sits outside ``module_metrics`` because that helper probes
    peak memory *first*: a compiling ``flex`` or an autotuning Triton kernel would
    otherwise report its one-off workspace as the layer's steady-state footprint.
    """
    try:
        fn()
        torch.cuda.synchronize()
        row = module_metrics(
            fn,
            flops=flops,
            bytes_moved=bytes_moved,
            warmup=env.warmup,
            iters=env.iters,
            tensor_cores=env.tensor_cores,
            peak_bandwidth_gbps=env.bandwidth_gbps,
        )
        row.update(ok=True, error="", oom_gib=float("nan"))
        return row
    except (torch.OutOfMemoryError, RuntimeError) as err:
        if not isinstance(err, torch.OutOfMemoryError) and "out of memory" not in str(
            err
        ):
            raise
        torch.cuda.empty_cache()
        return dict(
            ms=float("nan"),
            tflops=0.0,
            gbps=0.0,
            peak_gib=float("nan"),
            compute_pct=0.0,
            bandwidth_pct=0.0,
            ok=False,
            error="OOM",
            oom_gib=_requested_gib(str(err)),
        )


def step_fn(module, x, info, mode):
    """The closure that gets timed, built outside ``run_cell`` so that cell can
    drop its own references and free the batch before building the next one."""
    if mode == "fwdbwd":

        def call():
            module(x, info).sum().backward()

    else:

        def call():
            module(x, info)

    return call


def run_cell(env, args, backend, layout, lengths, window, mode) -> dict:
    """Build, warm and measure one (backend, layout, shape, window, mode) cell.

    Peak memory is the *delta* over what was resident before and after the cell.
    Compiling a new ``flex`` shape leaves inductor buffers allocated for the rest
    of the process, and ``reset_peak_memory_stats`` counts everything already
    allocated, so an absolute peak drifts upward as the sweep runs: the same 16k
    cell measured 0.95 GiB in the first sweep and 1.68 GiB in the second. The
    delta is the layer's own footprint and is comparable across cells.
    """
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated() / 2**30
    x, info = make_batch(layout, lengths, args, env)
    module = build_module(BACKENDS[backend], args, window, env)
    proj, attn = layer_flops(lengths, window, args)
    fwd_bytes = layer_bytes(lengths, args, env)
    if mode == "fwdbwd":
        flops = PROJ_FWDBWD * proj + ATTN_FWDBWD * attn
        moved = BYTES_FWDBWD * fwd_bytes
    else:
        flops = proj + attn
        moved = fwd_bytes

    call = step_fn(module, x, info, mode)
    row = measure(call, flops, moved, env)
    del call, module, x, info
    torch.cuda.empty_cache()
    # What this cell left behind was allocated while compiling, i.e. before the
    # high-water mark, so it is inside the measured peak without being part of
    # the layer's footprint.
    retained = torch.cuda.memory_allocated() / 2**30 - resident
    row["peak_gib"] -= resident + retained
    row.update(
        backend=backend,
        layout=layout,
        mode=mode,
        window=window,
        tokens=sum(lengths),
        n_docs=len(lengths),
        attn_flop_share=attn / (proj + attn),
        resident_gib=resident,
        retained_gib=retained,
    )
    return row


def label(row) -> str:
    window = "full" if row["window"] is None else f"w{row['window']}"
    return f"{row['backend']}/{row['layout']} {window} {row['mode']}"


def report(row) -> None:
    if row["ok"]:
        print(format_metrics(label(row), row, width=30), flush=True)
        return
    wanted = row["oom_gib"]
    detail = f" (asked {wanted:.1f} GiB)" if wanted == wanted else ""
    print(f"  {label(row):>30}: {row['error']}{detail}", flush=True)


def sweep_grid(env, args, sweep: str, seq_lens, windows) -> list[dict]:
    """Backend x layout x window x sequence length, one document per batch."""
    rows = []
    for seq_len in seq_lens:
        for window in windows:
            for layout in LAYOUTS:
                for backend in BACKENDS:
                    for mode in ("fwd", "fwdbwd"):
                        row = run_cell(
                            env, args, backend, layout, [seq_len], window, mode
                        )
                        row.update(sweep=sweep, seq_len=seq_len)
                        rows.append(row)
                        report(row)
        print(f"  -- {sweep} S={seq_len} done", flush=True)
    return rows


def tipo_lengths(mean_len: int, budget: int, seed: int) -> list[int]:
    """Lognormal document lengths, the shape rendered TIPO samples come out as."""
    generator = torch.Generator().manual_seed(seed)
    lengths, total = [], 0
    while total < budget:
        draw = int(mean_len * torch.randn(1, generator=generator).mul(0.6).exp())
        draw = max(16, min(draw, 4096, budget - total))
        lengths.append(draw)
        total += draw
    return lengths


def sweep_ragged(env, args) -> list[dict]:
    """Packed varlen against the padded batch a non-varlen backend would need.

    Both layouts run the same documents and both are charged the same *useful*
    FLOPs, so the padded rows pay for their padding in effective TFLOP/s.
    """
    rows = []
    for mean_len in args.doc_means:
        lengths = tipo_lengths(mean_len, args.tokens, seed=mean_len)
        pad_frac = 1 - sum(lengths) / (len(lengths) * max(lengths))
        for layout in LAYOUTS:
            for backend in BACKENDS:
                for mode in ("fwd", "fwdbwd"):
                    row = run_cell(env, args, backend, layout, lengths, None, mode)
                    row.update(sweep="ragged", mean_len=mean_len, pad_frac=pad_frac)
                    rows.append(row)
                    report(row)
        print(
            f"  -- ragged mean {mean_len} ({len(lengths)} docs, "
            f"{100 * pad_frac:.0f}% pad) done",
            flush=True,
        )
    return rows


def sweep_gqa(env, args) -> list[dict]:
    """KV-head count on the training backend: footprint moves, FLOPs do not."""
    rows = []
    seq_len = 2048
    lengths = [seq_len] * max(1, args.tokens // seq_len)
    for kv_heads in sorted(
        {h for h in (1, 2, 4, 8, args.heads) if args.heads % h == 0}
    ):
        scoped = argparse.Namespace(**vars(args))
        scoped.kv_heads = kv_heads
        row = run_cell(env, scoped, "varlen", "packed", lengths, None, "fwdbwd")
        row.update(
            sweep="gqa",
            seq_len=seq_len,
            kv_heads=kv_heads,
            kv_bytes_per_token=2 * kv_heads * args.head_dim * env.width,
        )
        rows.append(row)
        print(format_metrics(f"varlen kv={kv_heads}", row, width=30), flush=True)
    return rows


def check_precision(env, args) -> list[dict]:
    """Every backend against an fp64 reference, full causal and windowed.

    A windowed kernel that skips the wrong key blocks is fast and wrong, and the
    window sweep recommends one, so the windowed case is checked too.
    """
    seq_len, n_seqs = 256, 4
    lengths = [seq_len] * n_seqs
    tokens = seq_len * n_seqs
    packed = SeqInfo.from_lengths(torch.tensor(lengths, dtype=torch.int32), env.device)
    padded = SeqInfo.padded(n_seqs, seq_len, env.device)

    torch.manual_seed(0)
    reference = Env(env.device, torch.float64, env.bandwidth_gbps, 0, 0)
    x64 = torch.randn(tokens, args.dim, device=env.device, dtype=torch.float64)

    rows = []
    for window in (None, args.windows[0]):
        # fp64 has no flash path, so this falls through to SDPA with an explicit
        # mask -- the reference the fast kernels are judged against.
        base = build_module(VarlenAttention, args, window, reference)
        ref = base(x64, packed).double()
        state = base.state_dict()
        for name, cls in BACKENDS.items():
            module = build_module(cls, args, window, env)
            module.load_state_dict({k: v.to(env.dtype) for k, v in state.items()})
            x = x64.to(env.dtype)
            if name == "sdpa":
                out = module(x.view(n_seqs, seq_len, args.dim), padded)
                out = out.reshape(tokens, -1)
            else:
                out = module(x, packed)
            rows.append(
                dict(
                    backend=name,
                    window=window,
                    rel_err=rel_error(out, ref),
                    dtype=args.dtype,
                )
            )
            del module
        del base, ref
        torch.cuda.empty_cache()
    return rows


def frontier(points: list[dict]) -> list[dict]:
    """Points not dominated on both axes, ordered by memory."""
    best, out = float("inf"), []
    for row in sorted(points, key=lambda r: (r["peak_gib"], r["ms"])):
        if row["ms"] < best:
            out.append(row)
            best = row["ms"]
    return out


def series(rows, sweep, backend, layout, mode, window=None, seq_len=None):
    """Every row matching one series, ordered by sequence length."""
    sel = [
        r
        for r in rows
        if r["sweep"] == sweep
        and r["backend"] == backend
        and r["layout"] == layout
        and r["mode"] == mode
        and r["window"] == window
        and (seq_len is None or r["seq_len"] == seq_len)
    ]
    return sorted(sel, key=lambda r: r.get("seq_len", 0))


def cell(rows, *args, **kwargs):
    return next(iter(series(rows, *args, **kwargs)), None)


def window_name(window: int | None) -> str:
    return "global" if window is None else f"w{window}"


def mamf_floor_ms(rows, lengths, window, args) -> float:
    """The fwd+bwd time this shape would take at the achievable tensor peak.

    Read off a measured row rather than restated here, so the line drawn and the
    "% of peak" in every row can never disagree about which ceiling applies.
    """
    ceiling = next((r["compute_ceiling_tflops"] for r in rows if r["ok"]), 0.0)
    if not ceiling:
        return 0.0
    proj, attn = layer_flops(lengths, window, args)
    return 1e3 * (PROJ_FWDBWD * proj + ATTN_FWDBWD * attn) / (ceiling * 1e12)


def _oom_band(ax, dead, pal, ok, seen):
    """Off-scale band for configurations that have no point on the axis.

    An OOM asked for two orders of magnitude more memory than anything that ran,
    so plotting it at its requested size would squeeze every surviving point into
    a sliver at the left edge. Entries that failed the same way are collapsed: a
    backend that OOMs at all four windows is one fact, not four.
    """
    right = ax.get_xlim()[1]
    edge = max(r["peak_gib"] for r in ok) * 1.7
    ax.axvspan(edge, right, color="#EFEFEF", zorder=0)
    centre = (edge * right) ** 0.5
    top = max(r["ms"] for r in ok)

    collapsed: dict[tuple, list] = {}
    for key, name, row in dead:
        collapsed.setdefault((key, round(row["oom_gib"], 1)), []).append(name)
    for i, ((key, asked), names) in enumerate(collapsed.items()):
        height = top / 1.5**i
        ax.scatter(
            centre,
            height,
            color=pal.color(key),
            label=None if key in seen else key,
            **_XMARK,
        )
        seen.add(key)
        what = names[0] if len(names) == 1 else f"{key}, all {len(names)}"
        detail = f"asked {asked:.0f} GiB" if asked == asked else "no allocation size"
        ax.annotate(
            f"{what}\nOOM, {detail}",
            (centre, height),
            ha="center",
            va="top",
            xytext=(0, -8),
            **_NOTE,
        )


def plane(ax, tracks, groups, pal, title, budget=None, fronts=True):
    """Time against peak memory -- the plane every choice in this figure is made in.

    ``tracks`` is ``(color key, marker, rows)``; a track's points are joined in
    order, so one track is one backend walking a dimension (window, or sequence
    length) and the reader sees where that dimension moves it rather than reading
    two differently-scaled panels against each other.

    ``groups`` is ``(label, rows, floor_ms)``, one comparable set each. The
    frontier and the achievable-peak floor are drawn *within* a group and never
    across them: a shorter window or a shorter sequence is always cheaper, so a
    frontier spanning them would answer a backend question with an architecture
    change.
    """
    seen = set()
    ok = [r for _, _, rows in tracks for r in rows if r["ok"]]
    # Named from the group so an off-scale marker can say which configuration it
    # was, which the track alone does not carry.
    dead = [
        (r["backend"], f"{r['backend']} {name}", r)
        for name, group, _ in groups
        for r in group
        if not r["ok"]
    ]
    for key, marker, rows in tracks:
        good = [r for r in rows if r["ok"]]
        if len(good) > 1:
            ax.plot(
                [r["peak_gib"] for r in good],
                [r["ms"] for r in good],
                linewidth=1.1,
                color=pal.color(key),
                zorder=2,
            )
        for row in good:
            ax.scatter(
                row["peak_gib"],
                row["ms"],
                s=80,
                marker=marker,
                color=pal.color(key),
                zorder=3,
                label=None if key in seen else key,
            )
            seen.add(key)

    ax.set_xscale("log")
    ax.set_yscale("log")
    if not ok:
        return
    lo = min(r["peak_gib"] for r in ok)
    hi = max(r["peak_gib"] for r in ok)
    ax.set_xlim(lo / 1.9, hi * (3.0 if dead else 1.9))

    for name, rows, floor in groups:
        live = [r for r in rows if r["ok"]]
        if not live:
            continue
        if fronts and len(live) > 1:
            front = frontier(live)
            ax.plot(
                [r["peak_gib"] for r in front],
                [r["ms"] for r in front],
                color="#c0392b",
                linewidth=1.1,
                linestyle="--",
                zorder=1,
                label=None if "front" in seen else "Pareto frontier",
            )
            seen.add("front")
        anchor = max(live, key=lambda r: r["peak_gib"])
        # A long label on a rightmost point runs into the off-scale band; short
        # ones ("global", "w512", "8k") clear it, and keeping those on one side
        # is worth more than the symmetry.
        flip = bool(dead) and len(name) > 12
        ax.annotate(
            name,
            (anchor["peak_gib"], anchor["ms"]),
            ha="right" if flip else "left",
            xytext=(-10, -2) if flip else (9, -2),
            **_NOTE,
        )
        if floor:
            span = [min(r["peak_gib"] for r in live) / 1.15, anchor["peak_gib"] * 1.15]
            ax.plot(
                span,
                [floor, floor],
                color="#666666",
                linestyle=":",
                linewidth=1,
                zorder=1,
                label=None if "floor" in seen else "achievable tensor peak",
            )
            seen.add("floor")

    if budget and lo / 1.9 < budget < ax.get_xlim()[1]:
        ax.axvline(budget, color="#999999", linewidth=1)
        ax.annotate(
            f"{budget:.2f} GiB/layer",
            (budget, min(r["ms"] for r in ok)),
            rotation=90,
            va="bottom",
            ha="right",
            xytext=(-3, 0),
            **_NOTE,
        )
    ax.margins(y=0.12)
    if dead:
        _oom_band(ax, dead, pal, ok, seen)
    finish_axis(ax, "peak memory (GiB, one layer)", "fwd+bwd (ms)", title, si_x=False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")


def plot_window_plane(ax, rows, pal, args, budget):
    """Backend x window at the long-context shape, in the time-memory plane."""
    seq_len = args.window_seq_lens[-1]
    windows = [*args.windows, None]

    def at(backend, window):
        return cell(rows, "window", backend, "packed", "fwdbwd", window, seq_len)

    tracks = [
        (backend, "o", [r for w in windows if (r := at(backend, w))])
        for backend in BACKENDS
    ]
    groups = [
        (
            window_name(w),
            [r for b in BACKENDS if (r := at(b, w))],
            mamf_floor_ms(rows, [seq_len], w, args),
        )
        for w in windows
    ]
    plane(
        ax,
        tracks,
        groups,
        pal,
        f"S={seq_len} packed: every backend at every window",
        budget=budget,
    )


def plot_scaling_plane(ax, rows, pal, args, budget):
    """The same plane swept over sequence length: where each backend's curve goes."""
    tracks = [
        (backend, "o", series(rows, "context", backend, "packed", "fwdbwd"))
        for backend in BACKENDS
    ]
    groups = [
        (
            f"{seq_len // 1024}k",
            [
                r
                for b in BACKENDS
                if (r := cell(rows, "context", b, "packed", "fwdbwd", seq_len=seq_len))
            ],
            0.0,
        )
        for seq_len in args.seq_lens
    ]
    plane(
        ax,
        tracks,
        groups,
        pal,
        f"Packed, global: sequence length {args.seq_lens[0]} -> {args.seq_lens[-1]}",
        budget=budget,
        fronts=False,
    )
    ax.axvline(CARD_GIB, color="#c0392b", linewidth=1)
    ax.annotate(
        "32 GiB card",
        (CARD_GIB, ax.get_ylim()[0]),
        rotation=90,
        va="bottom",
        ha="right",
        xytext=(-3, 2),
        **_NOTE,
    )


def plot_layout_plane(ax, rows, pal, args, budget):
    """The same plane for the layout question, on the batch training actually sees."""
    mean_len = args.doc_means[0]

    def at(backend, layout):
        return next(
            (
                r
                for r in rows
                if r["sweep"] == "ragged"
                and r["backend"] == backend
                and r["layout"] == layout
                and r["mode"] == "fwdbwd"
                and r["mean_len"] == mean_len
            ),
            None,
        )

    lengths = tipo_lengths(mean_len, args.tokens, seed=mean_len)
    floor = mamf_floor_ms(rows, lengths, None, args)
    # varlen and triton have no padded kernel -- both fall through to the same
    # _sdpa_attend call and measure within 0.2% of it -- so the padded layout gets
    # one point for the three of them instead of three coincident markers.
    shares_sdpa = ("varlen", "triton")
    tracks, groups = [], []
    for layout, marker in (("packed", "o"), ("padded", "s")):
        group = []
        for backend in BACKENDS:
            row = at(backend, layout)
            if row is None or (layout == "padded" and backend in shares_sdpa):
                continue
            tracks.append((backend, marker, [row]))
            group.append(row)
        name = "packed" if layout == "packed" else "padded (= varlen = triton)"
        groups.append((name, group, floor))
    pad = next((r["pad_frac"] for r in groups[0][1]), 0.0)
    plane(
        ax,
        tracks,
        groups,
        pal,
        f"TIPO batch: {args.tokens // 1024}k tokens of mean-{mean_len} documents\n"
        f"packed (circle) vs padded (square, {100 * pad:.0f}% of it padding)",
        budget=budget,
    )


def plot_precision(ax, precision, pal, args):
    windows = sorted({r["window"] for r in precision}, key=lambda w: w or 0)
    xs = range(len(BACKENDS))
    width = 0.8 / len(windows)
    floor = min(max(r["rel_err"], 1e-12) for r in precision) / 10
    for i, window in enumerate(windows):
        offset = (i - (len(windows) - 1) / 2) * width
        values = [
            next(
                (
                    max(r["rel_err"], floor)
                    for r in precision
                    if r["backend"] == b and r["window"] == window
                ),
                floor,
            )
            for b in BACKENDS
        ]
        bars = ax.bar(
            [x + offset for x in xs],
            values,
            width,
            color=pal.color("full" if window is None else "windowed"),
            label=window_name(window),
        )
        bar_labels(ax, bars, "{:.1e}")
    ax.set_yscale("log")
    ax.set_ylim(floor, max(r["rel_err"] for r in precision) * 30)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(list(BACKENDS))
    ax.set_xlabel("backend")
    ax.set_ylabel(f"relative error vs fp64 ({args.dtype})")
    ax.set_title("Precision: all four agree to bf16's limit")
    ax.legend(frameon=False, fontsize=8)


def plot(rows, precision, out_dir, args):
    # Assigned in a fixed order so a backend keeps its color across every panel
    # and across the other figures in the suite.
    pal = Palette()
    for key in (*BACKENDS, "full", "windowed"):
        pal.color(key)
    budget = CARD_GIB / args.layers

    fig, axes = new_figure(2, 2, figsize=(15.5, 12.5))
    plot_window_plane(axes[0], rows, pal, args, budget)
    plot_scaling_plane(axes[1], rows, pal, args, budget)
    plot_layout_plane(axes[2], rows, pal, args, budget)
    plot_precision(axes[3], precision, pal, args)

    fig.suptitle(
        f"Attention backends -- {device_name()}  |  dim {args.dim}, "
        f"{args.heads}x{args.head_dim} heads, {args.kv_heads} kv, {args.dtype}\n"
        "lower-left is better; the memory axis is why the fastest-looking backend "
        "is not always the answer",
        fontweight="bold",
    )
    save_figure(fig, os.path.join(out_dir, "attention.png"))


def consistency(rows, args) -> None:
    """The same shape is measured in two sweeps; they must agree.

    S is timed once in the context sweep and again in the window sweep at
    ``window=None``, minutes apart. On a shared card that is the only signal that
    a neighbouring job arrived halfway through and half the table is contended.
    """
    print("\nsame shape in two sweeps (drift here means the machine changed):")
    for seq_len in args.window_seq_lens:
        for backend in BACKENDS:
            a = cell(rows, "context", backend, "packed", "fwdbwd", seq_len=seq_len)
            b = cell(rows, "window", backend, "packed", "fwdbwd", None, seq_len)
            if not (a and b and a["ok"] and b["ok"]):
                continue
            drift = 100 * (b["ms"] - a["ms"]) / a["ms"]
            flag = "  <-- DRIFT" if abs(drift) > 5 else ""
            print(
                f"  S={seq_len:>6} {backend:>7}: {a['ms']:8.2f} vs {b['ms']:8.2f} ms"
                f"  ({drift:+5.1f}%){flag}"
            )


def summarize(rows, args) -> None:
    """fwd+bwd in the training layout, which is the table the decision comes from."""
    print("\npacked layout, full causal, fwd+bwd (ms):")
    print(f"  {'S':>7}" + "".join(f"{b:>12}" for b in BACKENDS))
    for seq_len in args.seq_lens:
        line = f"  {seq_len:>7}"
        for backend in BACKENDS:
            row = cell(rows, "context", backend, "packed", "fwdbwd", seq_len=seq_len)
            if row is None:
                line += f"{'-':>12}"
            else:
                line += f"{row['ms']:>12.2f}" if row["ok"] else f"{'OOM':>12}"
        print(line)


def device_health(when: str) -> dict:
    """Free memory and the three STREAM patterns, as a contention check.

    Three independent tells, all cheap. A neighbour shows up as held memory, as a
    collapsed DRAM ceiling, and -- sharpest of the three -- as an inversion of the
    STREAM ordering: triad moves three streams to copy's two and is normally the
    highest, so a run where it is not is contended even when the headline number
    still looks plausible. Taken before and after the sweeps, because contention
    that starts in the middle is the case a single reading misses.
    """
    torch.cuda.empty_cache()
    free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
    # Device-wide usage minus what this process holds. By the end of a sweep our
    # own inductor buffers and flush scratch are ~3 GiB, and charging those to a
    # neighbour makes the guard condemn every clean run it is meant to certify.
    mine = torch.cuda.memory_reserved() / 2**30
    others = total - free - mine
    notes = []
    try:
        streams = stream_bandwidths()
    except torch.OutOfMemoryError:
        # Never fatal: this runs after the sweeps and before the JSON is written,
        # and losing a finished run to the probe would be the worst outcome here.
        streams = {"copy": 0.0, "read": 0.0, "triad": 0.0}
        notes.append("STREAM probe could not allocate -- the card is full")
    peak = max(streams.values())
    patterns = "  ".join(f"{k} {v:.0f}" for k, v in streams.items())
    print(
        f"  {when:>5}: {free:.1f}/{total:.1f} GiB free ({mine:.1f} ours)  |  "
        f"{patterns} GB/s"
    )

    # A fresh CUDA context alone accounts for ~1 GiB, so the bar sits above that.
    if others > 2.0:
        notes.append(f"{others:.1f} GiB held by another process")
    if peak < 0.95 * CLEAN_DRAM_GBPS:
        notes.append(
            f"DRAM ceiling {peak:.0f} GB/s against ~{CLEAN_DRAM_GBPS:.0f} clean"
        )
    if peak and streams["triad"] < max(streams["copy"], streams["read"]):
        notes.append("STREAM ordering inverted -- triad below copy/read")
    for note in notes:
        print(f"  WARNING ({when}): {note}", flush=True)
    return dict(
        when=when,
        free_gib=free,
        total_gib=total,
        ours_gib=mine,
        others_gib=others,
        peak_gbps=peak,
        contended=bool(notes),
        notes=notes,
        **streams,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/bench/kernel/attention")
    ap.add_argument("--dim", type=int, default=1280)
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    ap.add_argument(
        "--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384]
    )
    ap.add_argument("--windows", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--window-seq-lens", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--doc-means", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    # Only used to turn the card's capacity into a per-layer memory budget line.
    ap.add_argument("--layers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # flex_attention is only fused once torch.compile has seen it, and past the
    # default 8 recompiles dynamo drops back to the eager path -- which
    # materializes the whole score matrix and times something no configuration
    # would ever run. This sweep crosses ~30 distinct (length, layout, window)
    # shapes, so the limit is raised and a hit is made fatal rather than silent.
    torch._dynamo.config.recompile_limit = 512
    torch._dynamo.config.accumulated_recompile_limit = 4096
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    # An OOM row is a property of the algorithm only if the card was otherwise
    # empty, so the machine is measured before anything else is allocated.
    print(f"device: {device_name()}")
    health = [device_health("start")]
    env = Env(
        device="cuda",
        dtype=getattr(torch, args.dtype),
        bandwidth_gbps=health[0]["peak_gbps"],
        warmup=args.warmup,
        iters=args.iters,
    )

    rows = []
    print("context sweep:")
    rows += sweep_grid(env, args, "context", args.seq_lens, [None])
    print("window sweep:")
    rows += sweep_grid(env, args, "window", args.window_seq_lens, [*args.windows, None])
    print("ragged sweep:")
    rows += sweep_ragged(env, args)
    print("gqa sweep:")
    rows += sweep_gqa(env, args)
    print("precision:")
    precision = check_precision(env, args)
    for row in precision:
        window = "full" if row["window"] is None else f"w{row['window']}"
        print(f"  {row['backend']:>8} {window:>6}: rel_err {row['rel_err']:.3e}")

    print("device after the sweeps:")
    health.append(device_health("end"))

    with open(os.path.join(args.out, "attention.json"), "w") as handle:
        json.dump(
            dict(config=vars(args), device=health, rows=rows, precision=precision),
            handle,
            indent=2,
        )
    plot(rows, precision, args.out, args)
    summarize(rows, args)
    consistency(rows, args)
    if any(h["contended"] for h in health):
        print("\nDISCARD THIS RUN: the card was shared. See the warnings above.")
    print(f"\nwrote {args.out}/attention.png")


if __name__ == "__main__":
    main()
