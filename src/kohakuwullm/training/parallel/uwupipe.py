"""KohakUwULLM on kohakuwupipe: the adapter between the model and the trainer.

Everything model-specific lives here -- the cost model the split uses, the fp8
refresh and the router-bias update -- so ``kohakuwupipe`` stays architecture
free. See docs/internals/pipeline.md.
"""

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from anyschedule import AnySchedule

from kohakuwullm.compile_utils import apply_compile
from kohakuwullm.models import LMBackbone
from kohakuwullm.models.components.moe import MoEMLP
from kohakuwullm.models.mxfp8_swap import refresh_mxfp8_weights, swap_mxfp8
from kohakuwullm.training.optim.build import build_optimizer
from kohakuwullm.training.optim.lowbit import cast_parameters_
from kohakuwullm.training.parallel.pipeline import (
    HEAD_INEFFICIENCY,
    AutocastStage,
    LMStage,
    _block_cost,
    _block_params,
)
from kohakuwupipe import PipelineModule, plan_from_layers, plan_stages

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class PipelineStep:
    """One optimizer step, in the shape ``PipelineLoop`` drives.

    ``inputs`` is the packed token ids the first stage embeds and ``target`` the
    shifted labels the head compares against; ``layout`` is one :class:`SeqInfo`
    per microbatch, and ``trained`` is the step's real-label count as a host int.
    See docs/internals/pipeline.md.
    """

    inputs: torch.Tensor
    target: torch.Tensor
    layout: Any
    trained: int


def corpus_steps(loader, device):
    """Adapt a :class:`MicroBatchedStep` loader to :class:`PipelineStep`.

    The loader names the layout ``seq_infos`` and reports ``trained`` per
    microbatch; the loop wants ``layout`` and one host int.
    """
    for step in loader:
        step = step.to(device)
        yield PipelineStep(step.tokens, step.labels, step.seq_infos, step.num_trained)


def batch_signature(step: PipelineStep) -> torch.Tensor:
    """Four fp64 numbers that identify a step's content."""
    return torch.tensor(
        [
            float(step.inputs.shape[0]),
            float(step.trained),
            float(step.inputs.to(torch.float64).sum()),
            float(step.target.clamp_min(0).to(torch.float64).sum()),
        ],
        dtype=torch.float64,
        device=step.inputs.device,
    )


def first_mismatch(signatures) -> int | None:
    """The first rank whose signature differs from rank 0's, or ``None``."""
    for rank, other in enumerate(signatures[1:], start=1):
        if not torch.equal(other, signatures[0]):
            return rank
    return None


def verify_same_batch(step: PipelineStep) -> None:
    """Raise unless every rank drew the identical step.

    Each rank builds its own loader, so a divergence pairs the first stage's
    tokens with the last stage's labels from a different batch -- a loss that is
    wrong and entirely plausible. Collective; one sync, on one step.
    See docs/internals/pipeline.md.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    signature = batch_signature(step)
    gathered = [torch.zeros_like(signature) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, signature)
    rank = first_mismatch(gathered)
    if rank is not None:
        raise RuntimeError(
            f"rank {rank} drew a different first step than rank 0 "
            f"({gathered[rank].tolist()} vs {gathered[0].tolist()}); every rank "
            "must iterate an identical stream or the stages train on mismatched data"
        )


def plan_for(
    config,
    num_stages: int,
    seq_len: int,
    head_scale: float | None = None,
    layers=None,
    costs=None,
):
    """The stage split for ``config``.

    ``costs`` is a measured :class:`~kohakuwullm.training.parallel.autotune.LayerCosts`
    and replaces the analytic model outright; ``layers`` pins an explicit
    per-stage count and skips both. See docs/internals/pipeline.md.
    """
    scale = HEAD_INEFFICIENCY if head_scale is None else head_scale
    vocab_params = float(config.dim * config.vocab_size)
    layer_cost = costs.layers if costs is not None else _block_cost(config, seq_len)
    head_cost = costs.head if costs is not None else vocab_params * scale
    if layers is not None:
        if isinstance(layers, str) or not hasattr(layers, "__iter__"):
            raise TypeError(f"layers must be a sequence of ints, got {layers!r}")
        counts = [int(n) for n in layers]
        if sum(counts) != config.depth:
            raise ValueError(
                f"layers {counts} sum to {sum(counts)}, not depth {config.depth}"
            )
        if len(counts) != num_stages:
            raise ValueError(f"layers {counts} is not {num_stages} stages")
        return plan_from_layers(counts, layer_cost=layer_cost, head_cost=head_cost)
    return plan_stages(
        depth=config.depth,
        num_stages=num_stages,
        layer_cost=layer_cost,
        head_cost=head_cost,
        layer_params=(costs.params if costs is not None else _block_params(config)),
        head_params=(
            costs.head_params
            if costs is not None
            else (0.0 if config.tie_embeddings else vocab_params)
        ),
        embed_params=costs.embed_params if costs is not None else vocab_params,
    )


def with_router_losses(
    config, aux_loss_weight: float = 0.0, z_loss_weight: float = 0.0
):
    """``config`` with the MoE router's auxiliary weights overridden.

    Merges into whatever the preset already set, and returns ``config`` itself
    when both weights are zero. See docs/internals/moe-router-loss.md.
    """
    if not (aux_loss_weight or z_loss_weight):
        return config
    return replace(
        config,
        moe_router_kwargs={
            **config.moe_router_kwargs,
            "aux_loss_weight": aux_loss_weight,
            "z_loss_weight": z_loss_weight,
        },
    )


class LMPipelineModule(PipelineModule):
    """An :class:`LMStage` behind the ``kohakuwupipe`` module contract.

    Args:
        config: the arch config; every rank must pass the same one.
        optimizer / optimizer_kwargs: as ``build_optimizer`` takes them.
        param_dtype / autocast_dtype: stage storage and forward autocast.
        micro_tokens: fixes the boundary shape.
    """

    def __init__(
        self,
        config,
        micro_tokens: int,
        optimizer: str = "adamw",
        optimizer_kwargs: dict | None = None,
        learning_rate: float = 3e-4,
        param_dtype: str = "bf16",
        autocast_dtype: str | None = "bf16",
        head_kwargs: dict | None = None,
        mxfp8: bool = False,
        mxfp8_moe: str = "fused",
        mxfp8_scope: tuple[str, ...] | None = None,
        compile_spec: dict | None = None,
        scheduler_config: dict | None = None,
        loader=None,
        verify_data: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.micro_tokens = micro_tokens
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.learning_rate = learning_rate
        self.param_dtype = _DTYPES[param_dtype]
        self.autocast_dtype = (
            None if autocast_dtype is None else _DTYPES[autocast_dtype]
        )
        self.head_kwargs = head_kwargs
        self.mxfp8 = mxfp8
        self.mxfp8_moe = mxfp8_moe
        self.mxfp8_scope = mxfp8_scope
        # "module" mode only: "model" prefixes every checkpoint key with
        # `_orig_mod.`, which the stage-slice loader indexes by name.
        self.compile_spec = compile_spec
        self.scheduler_config = scheduler_config
        # Held so its stream position joins the checkpoint; a resume that restores
        # the weights but not the position re-trains tokens it has already seen.
        self.loader = loader
        self.verify_data = verify_data
        self.inner: LMStage | None = None
        self.mxfp8_modules = 0

    def configure_model(self, plan, rank, world, device):
        """Build this rank's slice, cast it, then wrap it in its autocast."""
        backbone = LMBackbone(self.config, head_kwargs=self.head_kwargs)
        inner = LMStage(backbone, plan, boundary_dtype=self.param_dtype).to(
            device=device, dtype=torch.float32
        )
        if self.param_dtype is not torch.float32:
            cast_parameters_(inner, self.param_dtype)
        # Bind the expert path before the swap reads it, then convert this
        # stage and refuse a partial result. See docs/internals/mxfp8.md.
        if self.mxfp8:
            for child in inner.modules():
                if isinstance(child, MoEMLP):
                    child.mxfp8_expert_path = self.mxfp8_moe
            report = swap_mxfp8(inner, scope=self.mxfp8_scope)
            if report.blocking:
                raise RuntimeError(f"mxfp8 swap incomplete: {report.blocking}")
            self.mxfp8_modules = len(report.modules)
        if self.compile_spec and self.compile_spec.get("mode", "module") == "model":
            raise ValueError(
                "compile mode 'model' renames checkpoint keys; use 'module'"
            )
        apply_compile(inner, self.compile_spec)
        self.inner = inner
        if self.autocast_dtype is None:
            return inner
        return AutocastStage(inner, dtype=self.autocast_dtype)

    @property
    def block_prefix(self) -> str:
        """Where the blocks live in the stage's state dict, wrapper included."""
        return "blocks" if self.autocast_dtype is None else "module.blocks"

    def boundary_example(self, plan, device):
        """The boundary this rank receives; call after :meth:`configure_model`.

        A tuple when the stage carries the router-loss accumulator.
        See docs/internals/moe-router-loss.md.
        """
        if plan.has_embed:
            return torch.zeros(self.micro_tokens, dtype=torch.long, device=device)
        hidden = torch.zeros(
            self.micro_tokens, self.config.dim, dtype=self.param_dtype, device=device
        )
        if not self.inner.router_stream:
            return hidden
        return hidden, torch.zeros(1, dtype=torch.float32, device=device)

    def configure_optimizers(self):
        """The stage's optimizer, plus its LR schedule when one is configured."""
        optimizer = build_optimizer(
            self.stage_module,
            self.optimizer_name,
            lr=self.learning_rate,
            **self.optimizer_kwargs,
        )
        if not self.scheduler_config:
            return optimizer
        return optimizer, AnySchedule(optimizer, config=self.scheduler_config)

    def set_seq_info(self, layout) -> None:
        self.stage_module.set_seq_info(layout)

    def training_step(self, batch, batch_idx: int) -> None:
        """Check once that every rank drew the same step."""
        if batch_idx == 0 and self.verify_data:
            verify_same_batch(batch)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Add the loader's stream position. Collective: every rank must call."""
        if self.loader is not None:
            checkpoint["loader"] = self.loader.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Arm the loader to continue where the checkpoint stopped."""
        if self.loader is not None and "loader" in checkpoint:
            self.loader.load_state_dict(checkpoint["loader"])

    def post_step(self, stage_module) -> dict[str, torch.Tensor]:
        """Refresh the fp8 copies and advance the routers; both per step."""
        if self.mxfp8_modules:
            refresh_mxfp8_weights(self.inner)
        return self.inner.update_router_bias()
