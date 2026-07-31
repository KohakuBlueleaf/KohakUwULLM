"""Does MXFP8 training hold up? A four-arm A/B on the real corpus.

``MXFP8Linear`` is 1.58-1.60x bf16 on forward+backward and its per-GEMM relative
error is 3.77e-02 -- 23x bf16's. The premise of MX training is that round-up block
scaling keeps *end-to-end loss* on top of bf16 anyway, and the one published
counterexample is an 843M model diverging at 300B tokens under round-to-nearest.
Nothing in a per-GEMM error number predicts which side a run lands on, so this
runs the training.

Five arms, and the last two are the ones that make the first three readable:

1. ``bf16``             -- unmodified.
2. ``fp8_up``           -- MXFP8 everywhere the swap reaches, round-up.
3. ``fp8_rtn``          -- the same, with the scale exponent rounded to nearest.
4. ``bf16_ctrl``        -- ``bf16`` again, bit-identical setup.
5. ``fp8_linears_only`` -- fp8 in the ``nn.Linear`` projections but **not** in an MoE
   layer's routed experts. Sparse presets only; on a dense preset it is arm 2.
6. ``fp8_moe_unfused`` -- arm 2 with the routed experts on the composed expert path
   in :mod:`kohakuwullm.kernels.mxfp8.moe_unfused` rather than the fused one. Same
   projections, same quantizer, same fp8 share; only the expert kernels differ, so
   the gap against arm 2 is attributable to them.

Arm 4 is the noise floor. Two runs of the same recipe on the same data do *not*
produce the same curve: the embedding backward and the flash-attention backward
both accumulate with atomics, and a training run amplifies that divergence.
Without arm 4 there is no way to say whether a 0.005 loss gap is fp8 or is the
same gap bf16 has with itself, and every conclusion below is a statement about
which of the two it is.

Arm 5 splits a gap that arm 2 can only report. At ``MoE-1B-A280M`` the full fp8 arm sat
0.2125 nats above ``bf16`` -- 41x the floor -- while arm 5, at 32.7% of matmul in fp8
against arm 2's 75.7%, came in at 1x the floor. So the ``nn.Linear`` swap is free and the
routed experts carry all of it, which is a different finding from "MoE fp8 costs 0.21
nats" and points at a different kernel. It is a *diagnostic* arm: its swap report is
deliberately blocking, because a model that is 43% bf16 is not an fp8 run.

The data is cached first and *replayed* from a memmap, so all four arms see the
same tokens in the same order. Live loaders were rejected for this: the renderer
draws a fresh task/length split per visit, so two runs of the same seed are close
but not identical, and "close" is exactly the resolution the experiment needs.

Usage::

    .venv/bin/python scripts/bench/fp8/fp8_training.py --stage cache --steps 6000
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench/fp8/fp8_training.py \\
        --stage train --arm bf16 --steps 6000
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from anyschedule import AnySchedule

from kohakuwullm.bench.analysis.runlog import HEADER, write_step_rows
from kohakuwullm.bench.core.timing import device_name, stream_bandwidths
from kohakuwullm.bench.vendor.mxfp8_rounding import load_round_to_nearest
from kohakuwullm.data import build_loader
from kohakuwullm.kernels.mxfp8.linear import MXFP8Linear
from kohakuwullm.models import LMBackbone, get_preset
from kohakuwullm.models.components.moe import MoEMLP
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.models.mxfp8_swap import (
    refresh_mxfp8_weights,
    swap_mxfp8,
)
from kohakuwullm.training.optim.build import build_optimizer
from kohakuwullm.utils import autofill_schedule_steps

# The repo's default recipe, unchanged. A divergence test is only interesting
# against the recipe that would actually be shipped, and an lr picked to make one
# arm unstable would put the answer in the lr rather than in the dtype.
PRESET = "Nano-200M"
VOCAB_SIZE = 65536
TOKENIZER = "models/tokenizer"
DATA_ROOT = "/xg7/caption-datasets"
SOURCES = [{"name": "danbooru", "repeat": 1}]
TOKEN_BUDGET = 16384
MAX_DOCS = 256
CTX_MAX = 2048
SEED = 20090220

OPTIMIZER = "adamw"
LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
EPS = 1e-8
GRAD_CLIP = 1.0
SCHEDULE = {"lr": {"mode": "cosine", "min_value": 0.1, "end": -1}}
WARMUP_RATIO = 0.01

ARMS = (
    "bf16",
    "fp8_up",
    "fp8_rtn",
    "bf16_ctrl",
    "fp8_linears_only",
    "fp8_moe_unfused",
)
# Steps buffered between device transfers, i.e. how often the per-step `.item()` syncs.
# Buffering was expected to be worth several percent and measured 0.3%: at 16,296
# tokens/step, 400 steps, buffering 50 instead of 1 moved bf16 152.722 -> 152.846 ms and
# fp8 144.244 -> 143.846. This step is device-bound, so a sync placed after the work it
# waits on costs only the slack. `bench_ms` predicted 8.6 ms because it drains the queue
# before each iteration and so cannot see cross-step pipelining -- a property of that
# harness, not of this loop.
#
# 1 is mandatory rather than merely adequate: per-step `elapsed` is device-accurate only
# when a sync happens each step, and `bench.runlog.contended_fraction` reads exactly
# those diffs. Above 1, `step_ms` (an endpoint mean) is still exact but the contention
# gate is not.
LOG_EVERY = 1
# Per-GEMM MXFP8 speedup over bf16 on sm_120, from scripts/bench/kernel/mxfp8.py. Only
# an input to the accounting below, never to a measurement.
GEMM_SPEEDUP = 1.59
# Below this the triad reading means another process is on the card, and a
# throughput number from a contended run is not a measurement.
TRIAD_FLOOR_GBPS = 1700.0


# ------------------------------------------------------------------ data cache


def cache_batches(
    directory: str, steps: int, num_workers: int = 16, tokens: int = TOKEN_BUDGET
) -> None:
    """Render ``steps`` packed batches once and write them as a replayable memmap.

    int32 on disk, widened to int64 at replay: the vocabulary is 65536 and the
    ignore index is -100, so int32 is lossless here and halves a multi-GB file
    that three later processes each map.

    ``tokens`` is the per-step budget. It is a parameter rather than the module constant
    because the token count is an *operating point*, not a detail: MXFP8 is 1.139x at
    16384 tokens and 0.625x at 8192 on Nano-200M, so a benchmark that reports one and
    not the other is reporting the half that flatters it. Nano-1B additionally does not
    fit 16384, so a second cache is the only way to include it.

    The render parameters are written to ``meta.json`` beside the memmap. Two caches
    that differ only in token budget are otherwise indistinguishable on disk, and
    replaying the wrong one silently changes tokens per step -- which changes both the
    loss curve and every throughput number derived from it.
    """
    from transformers import AutoTokenizer

    os.makedirs(directory, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    loader = build_loader(
        "ddp",
        SOURCES,
        tokenizer,
        renderer="tipo",
        root=DATA_ROOT,
        k=tokens,
        m=MAX_DOCS,
        ctx_max=CTX_MAX,
        num_workers=num_workers,
        seed=SEED,
        batches_per_epoch=max(1, steps // max(num_workers, 1) + 1),
    )

    tokens_path = os.path.join(directory, "tokens.i32")
    labels_path = os.path.join(directory, "labels.i32")
    token_offsets = [0]
    cu_offsets = [0]
    cu_flat: list[np.ndarray] = []
    trained = []
    start = time.perf_counter()
    with open(tokens_path, "wb") as tf, open(labels_path, "wb") as lf:
        iterator = iter(loader)
        for step in range(steps):
            batch = next(iterator)
            tf.write(batch.tokens.numpy().astype(np.int32).tobytes())
            lf.write(batch.labels.numpy().astype(np.int32).tobytes())
            cu = batch.seq_info.cu_seqlens.numpy().astype(np.int32)
            cu_flat.append(cu)
            token_offsets.append(token_offsets[-1] + int(batch.tokens.numel()))
            cu_offsets.append(cu_offsets[-1] + int(cu.size))
            trained.append(int(batch.num_trained))
            if (step + 1) % 500 == 0:
                rate = (step + 1) / (time.perf_counter() - start)
                print(f"  cached {step + 1}/{steps} batches ({rate:.1f}/s)", flush=True)

    np.savez(
        os.path.join(directory, "index.npz"),
        token_offsets=np.array(token_offsets, dtype=np.int64),
        cu_offsets=np.array(cu_offsets, dtype=np.int64),
        cu_flat=np.concatenate(cu_flat),
        trained=np.array(trained, dtype=np.int64),
    )
    total = int(token_offsets[-1])
    with open(os.path.join(directory, "meta.json"), "w") as handle:
        json.dump(
            {
                "steps": steps,
                "token_budget": tokens,
                "max_docs": MAX_DOCS,
                "ctx_max": CTX_MAX,
                "seed": SEED,
                "sources": SOURCES,
                "tokens_total": total,
                "tokens_per_step": total / max(steps, 1),
                "trained_fraction": sum(trained) / max(total, 1),
            },
            handle,
            indent=2,
        )
    print(
        f"cached {steps} batches, {total} tokens "
        f"({total / max(steps, 1):.0f}/step), "
        f"{sum(trained) / max(total, 1):.1%} trained"
    )


class BatchReplay:
    """The cached batches, mapped rather than loaded.

    The three training arms run as separate processes against one file, so a
    memmap keeps the corpus in the page cache once instead of three times.
    """

    def __init__(self, directory: str) -> None:
        index = np.load(os.path.join(directory, "index.npz"))
        self.token_offsets = index["token_offsets"]
        self.cu_offsets = index["cu_offsets"]
        self.cu_flat = index["cu_flat"]
        self.trained = index["trained"]
        total = int(self.token_offsets[-1])
        self.tokens = np.memmap(
            os.path.join(directory, "tokens.i32"),
            dtype=np.int32,
            mode="r",
            shape=(total,),
        )
        self.labels = np.memmap(
            os.path.join(directory, "labels.i32"),
            dtype=np.int32,
            mode="r",
            shape=(total,),
        )

    def __len__(self) -> int:
        return len(self.trained)

    @property
    def tokens_per_step(self) -> float:
        """Achieved mean, not the requested budget -- the packer trims the straddling
        document, so a 8192-token budget delivers 8104 and a 16384 one delivers 16296.
        Every per-token figure divides by this, so it is read from the cache."""
        return int(self.token_offsets[-1]) / max(len(self.trained), 1)

    def step(self, index: int, device) -> tuple[torch.Tensor, torch.Tensor, SeqInfo]:
        lo, hi = int(self.token_offsets[index]), int(self.token_offsets[index + 1])
        clo, chi = int(self.cu_offsets[index]), int(self.cu_offsets[index + 1])
        # astype widens *and* copies, which is what takes the tensors off the
        # read-only mapping; torch.from_numpy on a memmap slice would otherwise
        # hand back a tensor it warns about writing to.
        cu = torch.from_numpy(self.cu_flat[clo:chi].astype(np.int32))
        info = SeqInfo.from_lengths(cu[1:] - cu[:-1], device)
        tokens = torch.from_numpy(self.tokens[lo:hi].astype(np.int64))
        labels = torch.from_numpy(self.labels[lo:hi].astype(np.int64))
        return tokens.to(device), labels.to(device), info


# ----------------------------------------------------------------------- arms


def build_arm(arm: str, device, preset: str = PRESET) -> tuple[LMBackbone, int]:
    """The model for one arm, and how many MXFP8 modules it must refresh per step.

    The swap happens *after* construction so every arm starts from bit-identical
    weights: ``LMBackbone.initialize_weights`` runs on ``nn.Linear`` and the
    surgery copies what it produced.
    """
    torch.manual_seed(SEED)
    config = get_preset(preset, vocab_size=VOCAB_SIZE)
    model = LMBackbone(config, head_kwargs={"kernel": "chunked_ce"}).to(device)

    match arm:
        case "bf16" | "bf16_ctrl":
            return model, 0
        case "fp8_up" | "fp8_linears_only" | "fp8_moe_unfused":
            linear_cls = MXFP8Linear
        case "fp8_rtn":
            linear_cls = load_round_to_nearest().linear.MXFP8Linear
        case _:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    if arm == "fp8_moe_unfused":
        # Set before the swap, because `enable_mxfp8` reads it once and binds the
        # routed path there. Everything else about this arm is `fp8_up`: same
        # projections, same quantizer, same share of matmul in fp8 -- so the gap
        # between the two is attributable to the expert kernels and to nothing else.
        for module in model.modules():
            if isinstance(module, MoEMLP):
                module.mxfp8_expert_path = "unfused"
        if not any(isinstance(m, MoEMLP) for m in model.modules()):
            raise RuntimeError(
                f"{arm}: preset {preset!r} has no MoE layer, so this arm is identical "
                "to fp8_up and would isolate nothing. Use a sparse preset."
            )

    attribution = arm == "fp8_linears_only"
    report = swap_mxfp8(model, linear_cls=linear_cls, convert_declared=not attribution)
    # `blocking`, not `skipped`: a declared matmul nothing converted and a
    # matmul-shaped parameter nobody claimed are the two ways an arm is a mixture
    # without a single skipped projection, which is how a MoE preset used to reach a
    # benchmark with 42% of its arithmetic in bf16.
    #
    # The attribution arm is a mixture on purpose, so it tolerates `unreached` -- and
    # only that. A shape refusal or an unclaimed parameter still means the arm is not
    # the configuration its name describes, which is a different failure from the one
    # it is built to isolate.
    fatal = (
        report.skipped or report.unaccounted or (report.unreached and not attribution)
    )
    if fatal:
        raise RuntimeError(
            f"{arm}: {report.summary()}\n"
            f"Only {report.share('fp8'):.1%} of the per-token matmul reached fp8, "
            "which makes this arm a mixture rather than an fp8 run."
        )
    print(report.summary())
    if attribution:
        if not report.unreached:
            raise RuntimeError(
                f"{arm}: nothing was left in bf16 on preset {preset!r}, so this arm is "
                "identical to fp8_up and would attribute nothing. Use a sparse preset."
            )
        print(
            f"[{arm}] ATTRIBUTION ARM: {report.share('unreached'):.1%} of matmul left "
            "in bf16 on purpose. Not an fp8 run; only the gap against fp8_up is a result."
        )
    # Modules, not tensors: `refresh_mxfp8_weights` counts caches, and an MoE layer
    # holds two expert stacks behind one.
    return model, len(report.modules)


def matmul_accounting(directory: str, out_dir: str, preset: str = PRESET) -> dict:
    """How much of the model's matmul the swap actually moves, and the ceiling.

    Needed to read the throughput half of the result at all. ``MXFP8Linear``
    measures 1.59x bf16 *per GEMM*, but two things cut that before it reaches a
    step: WGRAD stays bf16, so only two of a layer's three products move; and the
    tied LM head is 30% of the model's MACs per token and correctly stays bf16.
    Quoting the per-GEMM figure as the training speedup would overstate it by
    more than the whole measured win.

    Runs on CPU and touches no weights, so it never competes with an arm.
    """
    config = get_preset(preset, vocab_size=VOCAB_SIZE)
    # Two models: the swap reports which names it took, and the untouched one is
    # where the shapes are read. Reading them off the swapped model would need a
    # second shape convention -- MXFP8Linear keeps no `in_features`.
    swapped = set(swap_mxfp8(LMBackbone(config)).swapped)
    model = LMBackbone(config)
    projections = {
        name: module.in_features * module.out_features
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    # A sparse preset's routed experts are *declared* rather than type-matched, so they
    # never appear in `projections` and 42% of a MoE model's per-token arithmetic would
    # drop out of the denominator -- flattering the fp8 share and the ceiling both.
    missing = swapped - set(projections)
    if missing:
        raise RuntimeError(
            f"{len(missing)} fp8 tensor(s) are not `nn.Linear` projections, so this "
            f"accounting cannot charge them: {sorted(missing)[:4]}. Take the MAC from "
            "the swap report's own census instead of a second walk over the model."
        )
    fp8 = sum(v for k, v in projections.items() if k in swapped)
    other = sum(v for k, v in projections.items() if k not in swapped)
    head = config.dim * config.vocab_size

    index = np.load(os.path.join(directory, "index.npz"))
    # Every document in the cache, not a sample of one batch. `cu_flat` is the
    # per-batch arrays concatenated, so each batch boundary shows up as a single
    # negative diff (back to 0); the positive diffs are exactly the lengths.
    diffs = np.diff(index["cu_flat"].astype(np.int64))
    lengths = diffs[diffs > 0].astype(np.float64)
    # Causal attention over a document of length L costs L(L+1)/2 pairs, so the
    # per-token cost is the length-weighted mean of (L+1)/2 -- not (mean L)/2,
    # which would understate it on a packed batch with a long tail.
    pairs = (lengths * (lengths + 1) / 2).sum() / lengths.sum()
    attention = 2 * pairs * config.head_dim * config.heads * config.depth
    total = fp8 + other + head + attention

    # fwd + dgrad + wgrad is three products; the swap moves two of the three, and
    # only for its own projections.
    ceiling = (3 * total) / (2 * fp8 / GEMM_SPEEDUP + fp8 + 3 * (total - fp8))
    report = {
        "preset": preset,
        "mean_document_tokens": float(lengths.mean()),
        "causal_pairs_per_token": float(pairs),
        "mac_per_token": {
            "fp8_projections": fp8,
            "other_projections": other,
            "lm_head": head,
            "attention": attention,
            "total": total,
        },
        "fp8_share_of_matmul": fp8 / total,
        "per_gemm_speedup": GEMM_SPEEDUP,
        "matmul_speedup_ceiling": ceiling,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "accounting.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return report


def train(
    arm: str,
    directory: str,
    steps: int,
    out_dir: str,
    lr: float = LR,
    tag: str | None = None,
    log_every: int = LOG_EVERY,
    preset: str = PRESET,
) -> None:
    """One arm. ``lr`` and ``tag`` exist for the stability sweep, which reruns
    the same two arms at multiples of the base rate; both default to the main
    experiment's values so a plain invocation is unchanged."""
    tag = tag or arm
    device = torch.device("cuda")
    bandwidths = stream_bandwidths()
    if bandwidths["triad"] < TRIAD_FLOOR_GBPS:
        raise RuntimeError(
            f"triad {bandwidths['triad']:.0f} GB/s is below {TRIAD_FLOOR_GBPS:.0f}: "
            "the card is contended, so nothing measured here would be a measurement"
        )

    replay = BatchReplay(directory)
    if len(replay) < steps:
        raise ValueError(f"cache holds {len(replay)} batches, asked for {steps}")

    model, expect_refresh = build_arm(arm, device, preset=preset)
    optimizer = build_optimizer(
        model,
        name=OPTIMIZER,
        lr=lr,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        eps=EPS,
    )
    schedule = dict(SCHEDULE)
    autofill_schedule_steps(schedule["lr"], steps, warmup_ratio=WARMUP_RATIO)
    scheduler = AnySchedule(optimizer, config=schedule)
    parameters = [p for p in model.parameters() if p.requires_grad]

    print(
        f"[{tag}] {device_name()} | {preset} | {steps} steps x ~{replay.tokens_per_step:.0f} tokens "
        f"| lr {lr:.2e} | triad {bandwidths['triad']:.0f} GB/s",
        flush=True,
    )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.csv")
    model.train()
    # int64: this run passes 2^31 tokens, and a float counter stops resolving
    # single tokens at 2^24.
    seen = 0
    pending: list[tuple] = []
    start = time.perf_counter()
    with open(path, "w") as handle:
        handle.write(HEADER + "\n")
        for step in range(steps):
            tokens, labels, info = replay.step(step, device)
            n_trained = int(replay.trained[step])
            with torch.autocast("cuda", torch.bfloat16):
                loss, logs = model.loss(tokens, labels, info, reduction="sum")
            # The head already reduced in fp32; dividing the fp32 sum by the
            # token count here is what makes the gradient a per-token mean.
            (loss / n_trained).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            # After *every* step, not lazily. A stale cache trains the model on
            # the previous step's weights and nothing in the loss says so.
            refreshed = refresh_mxfp8_weights(model)
            if refreshed != expect_refresh:
                raise RuntimeError(
                    f"refreshed {refreshed} MXFP8 caches, expected {expect_refresh}"
                )
            # No-op and sync-free on a dense preset (`moe_blocks` is empty). On a sparse
            # one it is part of the shipped recipe, not an extra: aux-loss-free
            # balancing is the only thing holding the load even, and an arm that skips
            # it is not the recipe whose loss is being compared.
            model.update_router_bias()

            seen += int(tokens.numel())
            # Buffered on *device*. Reading either scalar here would sync, and the
            # sync is what serializes host issue against device execution -- which
            # costs fp8 far more than bf16, because fp8 issues ~39% more ops and so
            # has more host work that could otherwise hide under the device.
            pending.append(
                (
                    step,
                    int(tokens.numel()),
                    n_trained,
                    logs["ce"].detach(),
                    grad_norm.detach(),
                    optimizer.param_groups[0]["lr"],
                    time.perf_counter() - start,
                )
            )
            if len(pending) >= log_every:
                last = write_step_rows(handle, pending, time.perf_counter() - start)
                pending.clear()
                if (step + 1) % 100 < log_every:
                    print(
                        f"  [{tag}] step {step + 1}/{steps} loss {last[0]:.4f} "
                        f"|g| {last[1]:.3f} "
                        f"{seen / last[2] / 1e3:.1f}k tok/s "
                        f"{last[2] / (step + 1) * 1e3:.0f} ms/step",
                        flush=True,
                    )
        if pending:
            write_step_rows(handle, pending, time.perf_counter() - start)

    elapsed = time.perf_counter() - start
    summary = {
        "arm": arm,
        "tag": tag,
        "learning_rate": lr,
        "preset": preset,
        "steps": steps,
        "tokens": seen,
        "seconds": elapsed,
        "tokens_per_sec": seen / elapsed,
        "ms_per_step": elapsed / steps * 1e3,
        "log_every": log_every,
        "mxfp8_modules": expect_refresh,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "triad_gbps": bandwidths["triad"],
        "device": device_name(),
    }
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cache", "train", "account"], required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--cache", default="out/bench/train/fp8_training/cache")
    parser.add_argument("--out", default="out/bench/train/fp8_training")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--preset",
        default=PRESET,
        help="architecture preset; the cache fixes tokens/step, this fixes the model",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=TOKEN_BUDGET,
        help="per-step token budget for --stage cache; the operating point, not a detail",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--tag", default=None, help="output basename; defaults to --arm"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=LOG_EVERY,
        help="steps buffered between device transfers; 1 reproduces published runs "
        "bit-exactly, higher removes the per-step sync and is faster (see LOG_EVERY)",
    )
    args = parser.parse_args()

    if args.stage == "cache":
        cache_batches(
            args.cache, args.steps, num_workers=args.num_workers, tokens=args.tokens
        )
        return
    if args.stage == "account":
        matmul_accounting(args.cache, args.out, preset=args.preset)
        return
    if args.arm is None:
        parser.error("--stage train needs --arm")
    train(
        args.arm,
        args.cache,
        args.steps,
        args.out,
        lr=args.lr,
        tag=args.tag,
        log_every=args.log_every,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
