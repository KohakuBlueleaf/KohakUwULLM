"""Does a 16-bit weight actually freeze under round-to-nearest, and when?

The argument for stochastic rounding is that round-to-nearest discards any update
below half an ulp of the weight it lands on, so the coordinate stops moving while
its gradient is still nonzero. That is a calculation. This measures it, as the
**never-moved fraction**: the share of weight elements still bit-identical to
their initial value after N steps. A frozen coordinate is directly observable;
inferring it from a ratio is not the same thing.

**The learning rate is the instrument, not the training stage.** The freeze
condition per element is ``|dw| < 0.5 * 2^-m * |w|`` with ``m`` the stored
mantissa bits -- 7 for bf16, 10 for fp16, so an 8x gap in resolution. Sweeping
``lr`` walks the update across that threshold in minutes where a decaying
schedule would take a full run, and it is the same phenomenon: the writeback
cannot tell why the update is small.

Two things the measurement corrected about that reasoning, both worth keeping:

* ``|dw|`` is **not** ``lr``. Adam's per-element ``|m| / sqrt(v)`` is well under 1
  for a noisy gradient, so the median update here is 0.12 half-ulps at lr=3e-4 in
  bf16, not the ~15 a naive ``lr / (0.5 * 2^-m * |w|)`` predicts.
* The **median is not the criterion** -- the tail is. An element needs one
  surviving step out of N, so what predicts the freeze is the *fraction* of
  updates above half an ulp per step, which the second panel plots. Reporting the
  median alone produced a panel that contradicted the never-moved curve beside it.

Measured on Kohaku-200M over 200 steps, the freeze onset sits **one decade apart**
in the two formats, and this repo's own cosine schedule lands between them: a
floor of 0.1 x 3e-4 = **3e-5 freezes 47% of bf16 hidden weights under RTN and
0.95% of fp16's**. fp16 does not break down until 3e-6. So under this schedule
bf16 needs stochastic rounding and fp16 does not -- which is the mantissa argument
holding up, with the margin being a decade of learning rate rather than the
factor of 2.5 the closed form predicts.

Four arms, differing in **one** thing: the rounding rule on the writeback.
Everything else -- fp32 moments, fp32 update arithmetic, the optimizer, the data,
the seed -- is shared, so a difference between arms cannot come from anywhere
else.

Secondary and closed-form: decoupled weight decay is a relative shrink of
``lr * wd`` per step, so it survives RTN only when ``lr * wd > 0.5 * 2^-m``. At
3e-4 and 0.1 that is 3e-5 against 2.0e-3 (bf16) and 2.4e-4 (fp16) -- **discarded
in both formats**, which makes SR a correctness requirement for weight decay
rather than an accuracy improvement. Reported per arm as ``decay_ratio``.

This is the *why* of the three-script set: ``stochastic_round.py`` measures what
the SR writeback costs in isolation and ``sr_whole_step.py`` whether that cost
survives a pipelined training step.

Usage:
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/bench/kernel/lost_update.py
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/bench/kernel/lost_update.py \\
        --preset Kohaku-200M --steps 300 --out out/bench/kernel/sr
"""

import argparse
import json
import os
import statistics

import torch

from kohakuwullm.bench import make_info, make_tokens
from kohakuwullm.bench.core.plotting import finish_axis, new_figure, save_figure
from kohakuwullm.bench.core.timing import device_name
from kohakuwullm.kernels.optim.stochastic_round import stochastic_round_
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.training.optim.lowbit import (
    KEEP_FP32_DEFAULT,
    StochasticAdamW,
    cast_parameters_,
)

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
MANTISSA = {torch.bfloat16: 7, torch.float16: 10}
# Peak, the cosine floor at 0.1x, and two decades below it. The first two are this
# repo's tuned schedule; the rest bracket both formats' thresholds so the curve
# has points on either side of each rather than one point and an extrapolation.
LR_GRID = [3e-4, 3e-5, 3e-6, 3e-7]
WEIGHT_DECAY = 0.1
# Distinct batches cycled through the run, so the gradient carries minibatch noise
# and the model cannot memorise its way to a zero gradient.
BATCH_POOL = 32


def half_ulp(param: torch.Tensor) -> torch.Tensor:
    """Half the distance to the next representable value, elementwise, in fp32."""
    away = torch.nextafter(param, torch.full_like(param, float("inf")))
    return 0.5 * (away.float() - param.float()).abs()


def sr_writeback(counter: list[int]):
    """A ``(dst, src)`` writeback whose seed advances on every call.

    ``StochasticAdamW`` resolves its rounding rule once into ``_round``, so
    replacing that attribute swaps the rule and changes nothing else -- which is
    what makes the four arms differ in exactly one thing. The counter is a list
    because the closure has to mutate it, and a seed that does not advance is the
    failure mode ``tests/test_stochastic_round.py`` pins.
    """

    def write(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        counter[0] += 1
        return stochastic_round_(dst, src, counter[0])

    return write


# Token-indexed axes are excluded from the freeze metric. Only the rows of tokens
# present in a batch receive any gradient, so with a 65536 vocab and 4096 tokens
# most of the embedding legitimately never moves -- and the embedding is 134M of
# this preset's 204M parameters, so counting it measures the batch's token
# coverage rather than the rounding rule. An earlier version of this script did
# exactly that and reported 24% "frozen" at a learning rate nothing freezes at.
TOKEN_INDEXED = ("embed", "head")


def tracked(model: LMBackbone, dtype: torch.dtype) -> list[torch.Tensor]:
    """Dense-gradient parameters held in ``dtype`` -- the ones the question applies to.

    The *actual* ``Parameter`` objects, not ``detach()`` copies: ``opt.state`` is
    keyed by object identity, so a detached view looks up as a miss and every
    state-derived metric comes back empty.

    ``KEEP_FP32_DEFAULT`` already leaves norms, the router and every vector in
    fp32, so those are excluded by the dtype test.
    """
    return [
        param
        for name, param in model.named_parameters()
        if param.dtype is dtype and not any(t in name for t in TOKEN_INDEXED)
    ]


def run_arm(
    preset: str, dtype: torch.dtype, rounding: str, lr: float, steps: int, tokens: int
) -> dict:
    torch.manual_seed(0)
    config = get_preset(preset)
    model = LMBackbone(config).to("cuda")
    cast_parameters_(model, dtype, keep_fp32=KEEP_FP32_DEFAULT)
    params = tracked(model, dtype)
    initial = [p.clone() for p in params]

    opt = StochasticAdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, rounding="nearest"
    )
    if rounding == "stochastic":
        opt._round = sr_writeback([0])

    # A pool of distinct batches, cycled -- not one batch repeated. With a single
    # fixed batch the gradient is noiseless, so Adam's |m| / sqrt(v) goes to 1 and
    # the model memorises it: at lr=3e-4 the loss reached 0.000 and the gradient
    # vanished, which shrinks the update for a reason that has nothing to do with
    # precision. That confounded the ratio columns badly enough that they did not
    # scale with the learning rate at all.
    pool = []
    for index in range(BATCH_POOL):
        info = make_info(tokens, 64, 512, "cuda", seed=index)
        ids, labels = make_tokens([info], config.vocab_size, "cuda", seed=index)
        pool.append((info, ids, labels))

    ratios: list[tuple[float, float]] = []
    losses: list[float] = []
    for step in range(steps):
        info, ids, labels = pool[step % BATCH_POOL]
        loss, _ = model.loss(ids, labels, info, reduction="mean")
        loss.backward()
        opt.step()
        # After the step, never before it: the optimizer state does not exist
        # until the first `step()`, and one nan poisons a mean.
        if step and step % max(steps // 6, 1) == 0:
            ratios.append(_update_stats(opt, params, lr))
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

    frozen, covered = _never_moved(params, initial, opt)
    threshold = 0.5 * 2.0 ** -MANTISSA[dtype]
    out = {
        "preset": preset,
        "dtype": str(dtype).replace("torch.", ""),
        "rounding": rounding,
        "lr": lr,
        "steps": steps,
        "tensors": len(params),
        "numel": sum(p.numel() for p in params),
        "never_moved": frozen,
        "gated_fraction": covered,
        "update_ratio_median": (
            statistics.median(r[0] for r in ratios) if ratios else float("nan")
        ),
        # The statistic that actually predicts the freeze. The median ratio does
        # not: at lr=3e-4 it reads 0.045 half-ulps while only 1% of weights
        # freeze, because an element needs just *one* surviving step out of 200
        # and the distribution is heavy-tailed. Reporting the median alone made
        # the panel contradict the measurement beside it.
        "survives_rtn": (
            statistics.median(r[1] for r in ratios) if ratios else float("nan")
        ),
        # lr*wd against the relative half-ulp: below 1 the decay is discarded.
        "decay_ratio": lr * WEIGHT_DECAY / threshold,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
    del model, opt, params, initial
    torch.cuda.empty_cache()
    return out


def _update_stats(opt, params, lr: float) -> tuple[float, float]:
    """``(median ratio, fraction above half an ulp)`` from Adam's live state.

    Estimated as ``lr * |m| / (sqrt(v) + eps)`` rather than by diffing weights,
    because a diff cannot separate "the update was small" from "the update was
    discarded", and separating those is the point.

    Two numbers because the median is not the predictive one. Adam's per-element
    ratio ``|m| / sqrt(v)`` is well below 1 for a noisy gradient, so the *typical*
    update is a small fraction of an ulp at every learning rate here; what decides
    whether a coordinate ever moves is the tail, and the second return value is
    that tail.
    """
    medians, survive = [], []
    for param in params[:: max(len(params) // 8, 1)]:
        state = opt.state.get(param)
        if not state or "exp_avg" not in state:
            continue  # pragma: no cover -- only before the first step()
        step = max(state["step"], 1)
        denom = (state["exp_avg_sq"].float() / (1 - 0.95**step)).sqrt() + 1e-8
        update = (state["exp_avg"].float() / (1 - 0.9**step)).abs() * (lr / denom)
        ratio = update / half_ulp(param).clamp_min(1e-30)
        medians.append(ratio.median().item())
        survive.append((ratio > 1.0).float().mean().item())
    if not medians:
        return float("nan"), float("nan")
    return statistics.median(medians), statistics.median(survive)


def _never_moved(params, initial, opt) -> tuple[float, float]:
    """Fraction still bit-identical to init, over elements that had an update.

    Gated on a nonzero first moment. Without the gate an element the optimizer
    never pushed counts as frozen, which conflates "no gradient" with "update
    discarded" -- and only the second is the freeze. Returns the gated fraction
    too, so a run where the gate excluded almost everything is visible rather
    than silently reported as a clean number.
    """
    same = total = eligible = seen = 0
    for param, start in zip(params, initial):
        state = opt.state.get(param, {})
        moment = state.get("exp_avg")
        gate = moment.abs() > 0 if moment is not None else torch.ones_like(param) > 0
        stuck = (param == start) & gate
        same += int(stuck.sum().item())
        total += int(gate.sum().item())
        eligible += int(gate.sum().item())
        seen += param.numel()
    return (same / max(total, 1), eligible / max(seen, 1))


def figure(rows: list[dict], path: str) -> None:
    fig, axes = new_figure(1, 2, figsize=(12.0, 4.4))
    styles = {
        ("bfloat16", "nearest"): ("#D55E00", "o", "-"),
        ("bfloat16", "stochastic"): ("#D55E00", "s", "--"),
        ("float16", "nearest"): ("#0072B2", "o", "-"),
        ("float16", "stochastic"): ("#0072B2", "s", "--"),
    }
    for (dtype, rounding), (color, marker, dashes) in styles.items():
        series = [r for r in rows if r["dtype"] == dtype and r["rounding"] == rounding]
        if not series:
            continue
        series.sort(key=lambda r: r["lr"])
        axes[0].plot(
            [r["lr"] for r in series],
            [100 * r["never_moved"] for r in series],
            color=color,
            marker=marker,
            ls=dashes,
            label=f"{dtype} {rounding}",
        )
        axes[1].plot(
            [r["lr"] for r in series],
            [100 * r["survives_rtn"] for r in series],
            color=color,
            marker=marker,
            ls=dashes,
            label=f"{dtype} {rounding}",
        )
    axes[0].set_xscale("log")
    finish_axis(
        axes[0],
        "learning rate",
        "weights never moved (%)",
        "the freeze, observed",
        legend=True,
        si_x=False,
    )
    axes[1].set_xscale("log")
    finish_axis(
        axes[1],
        "learning rate",
        "updates above half an ulp (%)",
        "why it freezes: the tail, not the median",
        legend=True,
        si_x=False,
    )
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="Kohaku-200M")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--out", default="out/bench/kernel/sr")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    config = get_preset(args.preset)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(
        f"{device_name()} CUDA_VISIBLE_DEVICES={visible} {args.preset} "
        f"tie_embeddings={config.tie_embeddings} steps={args.steps}"
    )
    print(
        "predicted freeze threshold at |w|=0.02: "
        f"bf16 lr<{0.5 * 2.0**-7 * 0.02:.2e}, fp16 lr<{0.5 * 2.0**-10 * 0.02:.2e}"
    )

    rows = []
    for name, dtype in DTYPES.items():
        for rounding in ("nearest", "stochastic"):
            for lr in LR_GRID:
                row = run_arm(args.preset, dtype, rounding, lr, args.steps, args.tokens)
                rows.append(row)
                print(
                    f"  {name} {rounding:10s} lr={lr:7.1e}  "
                    f"never-moved {100 * row['never_moved']:6.2f}%  "
                    f"|dw|/half-ulp med {row['update_ratio_median']:7.3f} "
                    f"survives {100 * row['survives_rtn']:6.2f}%  "
                    f"decay/half-ulp {row['decay_ratio']:7.4f}  "
                    f"loss {row['loss_first']:.3f} -> {row['loss_last']:.3f}"
                )

    os.makedirs(args.out, exist_ok=True)
    payload = {
        "device": device_name(),
        "cuda_visible_devices": visible,
        "preset": args.preset,
        "tie_embeddings": bool(config.tie_embeddings),
        "steps": args.steps,
        "tokens": args.tokens,
        "weight_decay": WEIGHT_DECAY,
        "rows": rows,
    }
    with open(os.path.join(args.out, "lost_update.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    figure(rows, os.path.join(args.out, "lost_update.png"))
    print(f"wrote {args.out}/lost_update.json and .png")


if __name__ == "__main__":
    main()
