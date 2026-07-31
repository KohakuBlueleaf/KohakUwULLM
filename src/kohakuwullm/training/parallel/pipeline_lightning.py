"""Pipeline-parallel training on ``torch.distributed.pipelining`` under Lightning.

``PipelineOnlyStrategy`` launches and initializes the process group; the loop
stays here. Stages need explicit ``input_args`` and their own autocast.
See docs/internals/pipeline.md.
"""

import warnings

import torch
import torch.distributed as dist
from torch.distributed.pipelining import (
    PipelineStage,
    Schedule1F1B,
    ScheduleGPipe,
    ScheduleInterleaved1F1B,
)

from kohakuwullm.models import LMBackbone
from kohakuwullm.training.optim.build import build_optimizer
from kohakuwullm.training.optim.lowbit import cast_parameters_
from kohakuwullm.training.parallel.pipeline import (
    HEAD_INEFFICIENCY,
    AutocastStage,
    LMStage,
    measure_head_scale,
    plan_for,
)

SCHEDULES = {
    "1f1b": Schedule1F1B,
    "gpipe": ScheduleGPipe,
    "interleaved1f1b": ScheduleInterleaved1F1B,
}


def build_stage(
    config,
    rank: int,
    world: int,
    device: torch.device,
    microbatch_tokens: int,
    autocast_dtype: torch.dtype | None = torch.bfloat16,
    param_dtype: torch.dtype = torch.float32,
    seq_len: int = 2048,
    calibrate: bool = True,
    decode: bool = False,
    head_kwargs: dict | None = None,
):
    """Build this rank's stage and the matching ``PipelineStage``.

    ``microbatch_tokens`` must be identical on every rank; it fixes the boundary
    shape ``input_args`` declares. See docs/internals/pipeline.md.
    """
    head_scale = HEAD_INEFFICIENCY
    if calibrate:
        scale = torch.zeros(1, device=device)
        if rank == 0:
            try:
                scale[0] = measure_head_scale(config, device, microbatch_tokens)
                # Return the probe's memory before the stage is allocated.
                torch.cuda.empty_cache()
            except Exception as exc:
                warnings.warn(
                    f"head calibration failed ({exc}); falling back to the "
                    f"default correction {HEAD_INEFFICIENCY}",
                    stacklevel=2,
                )
                scale[0] = HEAD_INEFFICIENCY
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(scale, src=0)
            head_scale = float(scale.item())
        elif rank == 0:
            head_scale = float(scale.item())

    backbone = LMBackbone(config, head_kwargs=head_kwargs)
    plans = plan_for(config, world, seq_len=seq_len, head_scale=head_scale)
    plan = plans[rank]

    module = LMStage(backbone, plan, boundary_dtype=param_dtype).to(
        device=device, dtype=torch.float32
    )
    # Cast before `PipelineStage`, which freezes the boundary metadata.
    if param_dtype is not torch.float32:
        cast_parameters_(module, param_dtype)
    if autocast_dtype is not None:
        module = AutocastStage(module, dtype=autocast_dtype)

    example = boundary_example(
        plan,
        config,
        microbatch_tokens,
        device,
        param_dtype,
        decode,
        router_stream=module.router_stream,
    )
    stage = PipelineStage(
        module,
        rank,
        world,
        device,
        input_args=example if isinstance(example, tuple) else (example,),
        group=pipeline_group(),
    )
    return module, stage, plan, plans


_PIPELINE_GROUP = None
_PIPELINE_GROUP_WORLD = None


def pipeline_group():
    """A process group carrying only the boundary sends, created once.

    Collective: every rank must reach this the same number of times.
    See docs/internals/pipeline.md.
    """
    global _PIPELINE_GROUP, _PIPELINE_GROUP_WORLD
    if not (dist.is_available() and dist.is_initialized()):
        return None
    world = dist.get_world_size()
    # A torn-down and rebuilt default group leaves the cached handle dangling.
    if _PIPELINE_GROUP is None or _PIPELINE_GROUP_WORLD != world:
        _PIPELINE_GROUP = dist.new_group(ranks=list(range(world)), backend="nccl")
        _PIPELINE_GROUP_WORLD = world
    return _PIPELINE_GROUP


def boundary_example(
    plan,
    config,
    microbatch_tokens: int,
    device: torch.device,
    param_dtype: torch.dtype,
    decode: bool = False,
    router_stream: bool = False,
):
    """The zero tensor whose shape and dtype ``PipelineStage`` freezes as the boundary.

    Packed ``(T,)`` / ``(T, dim)`` for training, padded ``(rows, 1)`` /
    ``(rows, 1, dim)`` for cached decode. ``router_stream`` appends the ``(1,)``
    auxiliary-loss accumulator, making the boundary a tuple.
    See docs/internals/pipeline.md and docs/internals/moe-router-loss.md.
    """
    if decode:
        if plan.has_embed:
            return torch.zeros(microbatch_tokens, 1, dtype=torch.long, device=device)
        return torch.zeros(
            microbatch_tokens, 1, config.dim, dtype=param_dtype, device=device
        )
    if plan.has_embed:
        return torch.zeros(microbatch_tokens, dtype=torch.long, device=device)
    # The boundary carries `param_dtype`, not `autocast_dtype`.
    hidden = torch.zeros(
        microbatch_tokens, config.dim, dtype=param_dtype, device=device
    )
    if not router_stream:
        return hidden
    return hidden, torch.zeros(1, dtype=torch.float32, device=device)


def decode_stage(
    module,
    plan,
    config,
    rank: int,
    world: int,
    device: torch.device,
    rows_per_microbatch: int,
    param_dtype: torch.dtype = torch.float32,
):
    """A decode-shaped ``PipelineStage`` over an already-built stage module.

    Shares ``module`` rather than rebuilding it; only the boundary differs.
    See docs/guides/generation.md.
    """
    example = boundary_example(
        plan, config, rows_per_microbatch, device, param_dtype, decode=True
    )
    return PipelineStage(
        module, rank, world, device, input_args=(example,), group=pipeline_group()
    )


def build_schedule(
    stage,
    num_microbatches: int,
    loss_fn=None,
    kind: str = "1f1b",
    scale_grads: bool = False,
):
    """Wrap a ``PipelineStage`` in the requested schedule.

    ``scale_grads`` defaults to False: ``loss_fn`` owns normalization.
    """
    if kind not in SCHEDULES:
        raise ValueError(f"unknown schedule {kind!r}; have {sorted(SCHEDULES)}")
    return SCHEDULES[kind](
        stage,
        n_microbatches=num_microbatches,
        loss_fn=loss_fn,
        scale_grads=scale_grads,
    )


def run_step(schedule, rank: int, world: int, tokens=None, targets=None, losses=None):
    """One optimizer step's worth of microbatches.

    Every rank must call this exactly once per step, in the same order.
    """
    if world == 1:
        raise ValueError("pipeline schedule requires world_size > 1")
    if rank == 0:
        schedule.step(tokens)
    elif rank == world - 1:
        schedule.step(target=targets, losses=losses)
    else:
        schedule.step()


def stage_local_optimizer(module, lr: float = 3e-4, **kwargs):
    """AdamW over this stage's parameters only, which are disjoint per rank."""
    return build_optimizer(module, lr=lr, **kwargs)
