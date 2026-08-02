"""One launch for a whole model's stochastic-rounded writeback.

One chunk descriptor per (tensor, chunk), built once from the shapes and reused
every step. The RNG offset of element ``i`` of tensor ``t`` is ``cu_numel[t] + i``,
so the result is bit-identical to the per-parameter loop it replaces.

See docs/internals/kernels.md.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.optim.stochastic_round import _format_of, _sr_round

# Elements per program, larger than the per-tensor kernel's 1024: the table costs
# 16 bytes per chunk.
_GROUP_BLOCK = 4096
_GROUP_WARPS = 4


@triton.jit
def _sr_grouped_kernel(
    param_ptrs,
    update_ptrs,
    chunk_tensor,
    chunk_start,
    chunk_len,
    chunk_rng,
    keep,
    seed,
    PARAM_TY: tl.constexpr,
    K_NORMAL: tl.constexpr,
    MIN_EXP: tl.constexpr,
    TINY_BITS: tl.constexpr,
    TAIL_SCALE: tl.constexpr,
    SUBNORMAL_GAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    which = tl.load(chunk_tensor + pid)
    start = tl.load(chunk_start + pid)
    count = tl.load(chunk_len + pid)
    rng = tl.load(chunk_rng + pid)

    # The pointee dtype cannot be read off an int64 handle the way it can off a
    # real pointer argument, so the parameter dtype arrives as a constexpr.
    param = tl.load(param_ptrs + which).to(tl.pointer_type(PARAM_TY))
    update = tl.load(update_ptrs + which).to(tl.pointer_type(tl.float32))

    offs = tl.arange(0, BLOCK)
    mask = offs < count
    w = tl.load(param + start + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(update + start + offs, mask=mask, other=0.0)
    # `rng` is int64: the whole-model counter passes 2^31.
    draw = tl.randint(seed, rng + offs).to(tl.int32, bitcast=True)
    y = _sr_round(
        w * keep + u, draw, K_NORMAL, MIN_EXP, TINY_BITS, TAIL_SCALE, SUBNORMAL_GAP
    )
    tl.store(param + start + offs, y.to(PARAM_TY), mask=mask)


_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


class GroupedWriteback:
    """A reusable one-launch ``param = SR(param * (1 - decay) + update)``.

    Construct once per parameter list and call every step. The table holds raw
    pointers, so anything that reallocates a parameter or an update buffer
    invalidates it; call ``rebuild`` then. All parameters must share one dtype,
    since the rounding constants and the store type are compile-time arguments.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        updates: list[torch.Tensor],
        block: int = _GROUP_BLOCK,
    ) -> None:
        if len(params) != len(updates):
            raise ValueError(f"{len(params)} params against {len(updates)} updates")
        if not params:
            raise ValueError("GroupedWriteback needs at least one parameter")
        self.dtype = params[0].dtype
        self._fmt = _format_of(self.dtype)
        self.block = block
        self._validate(params, updates)
        self._build(params, updates)

    def _validate(self, params, updates) -> None:
        device = params[0].device
        for param, update in zip(params, updates):
            if param.dtype is not self.dtype:
                raise ValueError(
                    f"mixed parameter dtypes: {self.dtype} and {param.dtype}"
                )
            if update.dtype is not torch.float32:
                raise ValueError(f"updates must be float32, got {update.dtype}")
            if param.shape != update.shape:
                raise ValueError(
                    f"shape mismatch: {tuple(param.shape)} vs {tuple(update.shape)}"
                )
            if not param.is_contiguous() or not update.is_contiguous():
                raise ValueError("grouped writeback needs contiguous tensors")
            if param.device != device or update.device != device:
                raise ValueError("every tensor must live on one device")

    def _build(self, params, updates) -> None:
        device = params[0].device
        numels = [p.numel() for p in params]
        total = sum(numels)
        # Per-tensor addressing is 32-bit; the whole-model RNG counter is int64.
        oversized = [n for n in numels if n >= 2**31]
        if oversized:
            raise ValueError(
                f"tensor of {oversized[0]} elements exceeds 32-bit addressing"
            )

        numel = torch.tensor(numels, dtype=torch.int64)
        cu = torch.cat([torch.zeros(1, dtype=torch.int64), numel.cumsum(0)])
        per_tensor = (numel + self.block - 1) // self.block
        which = torch.repeat_interleave(torch.arange(len(params)), per_tensor)
        # Index of the chunk *within* its tensor: the global chunk index minus
        # where that tensor's chunks began.
        chunk_base = torch.cat(
            [torch.zeros(1, dtype=torch.int64), per_tensor.cumsum(0)]
        )
        within = torch.arange(int(per_tensor.sum())) - chunk_base[which]
        start = within * self.block
        length = torch.clamp(numel[which] - start, max=self.block)

        self.n_chunks = int(per_tensor.sum())
        self.numel = total
        self._cu_numel = [int(v) for v in cu.tolist()]
        self.param_ptrs = torch.tensor(
            [p.data_ptr() for p in params], dtype=torch.int64, device=device
        )
        self.update_ptrs = torch.tensor(
            [u.data_ptr() for u in updates], dtype=torch.int64, device=device
        )
        self.chunk_tensor = which.to(torch.int32).to(device)
        self.chunk_start = start.to(torch.int32).to(device)
        self.chunk_len = length.to(torch.int32).to(device)
        self.chunk_rng = (cu[which] + start).to(torch.int64).to(device)

    def rebuild(self, params: list[torch.Tensor], updates: list[torch.Tensor]) -> None:
        """Re-read every address. Call after anything that may have reallocated."""
        self._validate(params, updates)
        self._build(params, updates)

    def __call__(self, seed: int, *, decay: float = 0.0) -> None:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        _sr_grouped_kernel[(self.n_chunks,)](
            self.param_ptrs,
            self.update_ptrs,
            self.chunk_tensor,
            self.chunk_start,
            self.chunk_len,
            self.chunk_rng,
            1.0 - decay,
            seed,
            PARAM_TY=_TL_DTYPE[self.dtype],
            K_NORMAL=self._fmt.k_normal,
            MIN_EXP=self._fmt.min_exp,
            TINY_BITS=self._fmt.tiny_bits,
            TAIL_SCALE=self._fmt.tail_scale,
            SUBNORMAL_GAP=self._fmt.subnormal_gap,
            BLOCK=self.block,
            num_warps=_GROUP_WARPS,
        )

    def rng_offsets(self) -> list[int]:
        """Per-tensor RNG offsets, so the per-parameter path can be made to match."""
        return list(self._cu_numel[:-1])
