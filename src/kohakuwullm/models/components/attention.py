"""Causal self-attention: four interchangeable kernels behind one contract.

All take ``(x, seq_info, posenc)`` and return the same shape as ``x``; only the
inner kernel call differs. See docs/concepts/architecture.md for which to select where.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.varlen import AuxRequest, varlen_attn

from kohakuwullm.kernels.attention.flash_attn import triton_varlen_attn
from kohakuwullm.kernels.mxfp8.attention import BLOCK_SCALE, mxfp8_varlen_attn
from kohakuwullm.models.components.norm import RMSNorm
from kohakuwullm.registry import ATTENTION


class BaseAttention(nn.Module):
    """Shared projections / GQA layout / QK-norm / sinks.

    Args:
        dim: model width.
        heads: number of query heads.
        kv_heads: number of key-value heads (``heads`` -> MHA, 1 -> MQA).
        head_dim: per-head width; defaults to ``dim // heads``.
        qk_norm: RMSNorm over ``q`` and ``k`` before the kernel.
        qk_norm_affine: give the QK norms a learnable scale.
        bias: bias on qkv/out projections.
        sliding_window: attend only this many tokens back; ``None`` -> full causal.
        sink: learnable per-head attention sink.
        softmax_scale: override the default ``head_dim ** -0.5``.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int | None = None,
        head_dim: int | None = None,
        qk_norm: bool = True,
        qk_norm_affine: bool = True,
        bias: bool = False,
        sliding_window: int | None = None,
        sink: bool = False,
        softmax_scale: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        if self.heads % self.kv_heads != 0:
            raise ValueError(
                f"heads ({self.heads}) must be divisible by kv_heads ({self.kv_heads})"
            )
        self.head_dim = head_dim or (dim // heads)
        self.sliding_window = sliding_window
        self.scale = softmax_scale or self.head_dim**-0.5

        q_out = self.heads * self.head_dim
        kv_out = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_out, bias=bias)
        self.k_proj = nn.Linear(dim, kv_out, bias=bias)
        self.v_proj = nn.Linear(dim, kv_out, bias=bias)
        self.o_proj = nn.Linear(q_out, dim, bias=bias)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=eps, affine=qk_norm_affine)
            self.k_norm = RMSNorm(self.head_dim, eps=eps, affine=qk_norm_affine)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.sink = nn.Parameter(torch.zeros(self.heads)) if sink else None

    def project(self, x: torch.Tensor, posenc):
        """``x`` -> ``q, k, v`` each ``(..., H, Dh)`` with the position axis intact.

        QK-norm and RoPE run in fp32, so ``q`` and ``k`` are cast back to ``v``'s
        dtype at the end: the flash kernels reject a mixed-precision triple.
        """
        lead = x.shape[:-1]
        q = self.q_norm(self.q_proj(x).view(*lead, self.heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(*lead, self.kv_heads, self.head_dim))
        v = self.v_proj(x).view(*lead, self.kv_heads, self.head_dim)
        if posenc is not None:
            q, k = posenc.apply(q, k)
        return q.to(v.dtype), k.to(v.dtype), v

    def apply_sink(self, out: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        """Fold a learnable sink logit into an already-normalized attention output.

        ``out * sigmoid(lse - sink)``. See docs/concepts/architecture.md.
        """
        if self.sink is None:
            return out
        sink = self.sink.to(lse.dtype)
        # lse is (..., H) after the transpose done by each backend.
        return out * torch.sigmoid(lse - sink).unsqueeze(-1).to(out.dtype)

    def merge(self, out: torch.Tensor) -> torch.Tensor:
        """``(..., H, Dh)`` -> ``(..., dim)`` and out-project."""
        return self.o_proj(out.reshape(*out.shape[:-2], self.heads * self.head_dim))

    def attend_cached(self, x: torch.Tensor, posenc, cache) -> torch.Tensor:
        """Attend padded ``(B, S, dim)`` ``x`` over a :class:`LayerKVCache` prefix.

        Every backend shares this path. See docs/concepts/architecture.md.
        """
        offset = cache.offset
        q, k, v = self.project(x, posenc)
        k, v = cache.append(k, v)
        if cache.cache.static:
            mask = cache.cache.key_mask(q.shape[1])
            if self.sliding_window is not None:
                mask = mask & cache.cache.window_mask(q.shape[1], self.sliding_window)
            return self.merge(
                _attend_with_mask(
                    self, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask
                )
            )
        if q.shape[1] == 1:
            return self.merge(_decode_attend(self, q, k, v))
        return self.merge(_sdpa_attend(self, q, k, v, q_offset=offset))


@ATTENTION.register("varlen")
class VarlenAttention(BaseAttention):
    """FlashAttention-2 varlen via ``torch.nn.attention.varlen.varlen_attn``.

    The training default. Consumes the packed ``(T, D)`` layout directly, so
    cross-document attention is impossible by construction. Falls back to SDPA on
    a padded batch or a dtype the kernel does not take.
    """

    def forward(
        self, x: torch.Tensor, seq_info, posenc=None, cache=None
    ) -> torch.Tensor:
        if cache is not None:
            return self.attend_cached(x, posenc, cache)
        q, k, v = self.project(x, posenc)
        # The flash kernel takes fp16/bf16 packed only; anything else -> SDPA.
        if not seq_info.packed or v.dtype not in (torch.float16, torch.bfloat16):
            if seq_info.packed:
                doc_id = _doc_ids(seq_info, q.shape[0], q.device)
                q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
                return self.merge(_sdpa_attend(self, q, k, v, doc_id).squeeze(0))
            return self.merge(_sdpa_attend(self, q, k, v))

        window = (
            (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, 0)
        )
        cu = seq_info.cu_seqlens
        need_lse = self.sink is not None
        result = varlen_attn(
            q,
            k,
            v,
            cu,
            cu,
            seq_info.max_seqlen,
            seq_info.max_seqlen,
            scale=self.scale,
            window_size=window,
            enable_gqa=self.kv_heads != self.heads,
            return_aux=AuxRequest(lse=True) if need_lse else None,
        )
        if need_lse:
            # varlen_attn returns out as (T, H, Dh) but lse as (H, T).
            out, lse = result
            out = self.apply_sink(out, lse.transpose(0, 1))
        else:
            out = result
        return self.merge(out)


def _causal_mask(
    q_len: int,
    kv_len: int,
    device,
    window: int | None,
    doc_id: torch.Tensor | None,
    q_offset: int = 0,
) -> torch.Tensor:
    """``(q_len, kv_len)`` bool mask: causal, optionally windowed / per-document.

    Query row ``i`` sits at absolute position ``q_offset + i``; key column ``j``
    at ``j``. ``doc_id`` is only defined when the two lengths agree and
    ``q_offset`` is 0.
    """
    q_idx = torch.arange(q_len, device=device) + q_offset
    kv_idx = torch.arange(kv_len, device=device)
    mask = q_idx[:, None] >= kv_idx[None, :]
    if window is not None:
        mask &= q_idx[:, None] - kv_idx[None, :] < window
    if doc_id is not None:
        mask &= doc_id[:, None] == doc_id[None, :]
    return mask


def _doc_ids(seq_info, seq_len: int, device) -> torch.Tensor:
    """``(T,)`` document index per token, for block-diagonal masking."""
    doc_id = torch.zeros(seq_len, dtype=torch.int32, device=device)
    boundaries = seq_info.cu_seqlens[1:-1].long()
    if boundaries.numel():
        doc_id[boundaries] = 1
    return doc_id.cumsum(0)


def _sdpa_attend(module: BaseAttention, q, k, v, doc_id=None, q_offset=0):
    """Causal SDPA over ``(B, S, H, Dh)``, with optional window / sink / doc mask.

    ``k``/``v`` may be longer than ``q``, with ``q`` starting at absolute
    position ``q_offset`` -- the cached-generation case.

    ``doc_id`` is **required** whenever the ``(1, T, ...)`` batch is really a
    packed set of documents; without it ``is_causal`` attends across boundaries.
    """
    q = q.transpose(1, 2)  # (B, H, S, Dh)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = module.kv_heads != module.heads
    q_len, kv_len = q.shape[-2], k.shape[-2]

    if (
        module.sliding_window is None
        and module.sink is None
        and doc_id is None
        and q_len == kv_len
    ):
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=module.scale, enable_gqa=enable_gqa
        )
        return out.transpose(1, 2)

    # A window, a sink, document boundaries or a cached prefix need an explicit mask.
    mask = _causal_mask(
        q_len, kv_len, q.device, module.sliding_window, doc_id, q_offset=q_offset
    )
    return _attend_with_mask(module, q, k, v, mask)


def _attend_with_mask(module: BaseAttention, q, k, v, mask):
    """Attend ``(B, H, S, Dh)`` under ``mask``; ``None`` lets every query see all keys."""
    enable_gqa = module.kv_heads != module.heads
    if module.sink is None:
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=module.scale, enable_gqa=enable_gqa
        )
        return out.transpose(1, 2)

    if enable_gqa:
        repeat = module.heads // module.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scores = (q.float() @ k.float().transpose(-1, -2)) * module.scale
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    sink = module.sink.to(scores.dtype).view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
    probs = torch.cat([scores, sink], dim=-1).softmax(-1)[..., :-1]
    out = (probs.to(v.dtype) @ v).transpose(1, 2)
    return out


def _decode_attend(module: BaseAttention, q, k, v):
    """One query row per sequence against the cached prefix, mask-free.

    A window becomes a slice of the prefix rather than a mask.
    """
    if module.sliding_window is not None:
        k = k[:, -module.sliding_window :]
        v = v[:, -module.sliding_window :]
    return _attend_with_mask(
        module, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None
    )


@ATTENTION.register("triton")
class TritonVarlenAttention(BaseAttention):
    """Varlen flash attention from our own Triton kernel (``kernels/flash_attn.py``).

    Same contract and numerics as ``varlen``, with tile sizes autotuned for
    sm_120's ~100 KB shared-memory budget. See docs/concepts/architecture.md.
    """

    def forward(
        self, x: torch.Tensor, seq_info, posenc=None, cache=None
    ) -> torch.Tensor:
        if cache is not None:
            return self.attend_cached(x, posenc, cache)
        q, k, v = self.project(x, posenc)
        if not seq_info.packed or v.dtype not in (torch.float16, torch.bfloat16):
            if seq_info.packed:
                doc_id = _doc_ids(seq_info, q.shape[0], q.device)
                q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
                return self.merge(_sdpa_attend(self, q, k, v, doc_id).squeeze(0))
            return self.merge(_sdpa_attend(self, q, k, v))

        need_lse = self.sink is not None
        result = triton_varlen_attn(
            q,
            k,
            v,
            seq_info.cu_seqlens,
            seq_info.max_seqlen,
            sm_scale=self.scale,
            causal=True,
            window=self.sliding_window,
            return_lse=need_lse,
        )
        if need_lse:
            # The kernel returns lse as (H, T); out is (T, H, Dh).
            out, lse = result
            out = self.apply_sink(out, lse.transpose(0, 1))
        else:
            out = result
        return self.merge(out)


@ATTENTION.register("mxfp8")
class MXFP8Attention(BaseAttention):
    """Varlen flash attention with `QK^T` in block-scaled e4m3.

    Same contract as ``triton``; ``PV`` and both gradient GEMMs stay in the input
    dtype. Falls back to SDPA for a padded batch, a non-16-bit dtype, or a
    head_dim that is not a multiple of the 32-element scale block.
    See docs/internals/mxfp8-attention.md.
    """

    def forward(
        self, x: torch.Tensor, seq_info, posenc=None, cache=None
    ) -> torch.Tensor:
        if cache is not None:
            return self.attend_cached(x, posenc, cache)
        q, k, v = self.project(x, posenc)
        usable = (
            seq_info.packed
            and v.dtype in (torch.float16, torch.bfloat16)
            and q.shape[-1] % BLOCK_SCALE == 0
        )
        if not usable:
            if seq_info.packed:
                doc_id = _doc_ids(seq_info, q.shape[0], q.device)
                q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
                return self.merge(_sdpa_attend(self, q, k, v, doc_id).squeeze(0))
            return self.merge(_sdpa_attend(self, q, k, v))

        need_lse = self.sink is not None
        result = mxfp8_varlen_attn(
            q,
            k,
            v,
            seq_info.cu_seqlens,
            seq_info.max_seqlen,
            sm_scale=self.scale,
            causal=True,
            window=self.sliding_window,
            return_lse=need_lse,
        )
        if need_lse:
            out, lse = result
            out = self.apply_sink(out, lse.transpose(0, 1))
        else:
            out = result
        return self.merge(out)


@ATTENTION.register("sdpa")
class SDPAAttention(BaseAttention):
    """Padded causal attention on ``F.scaled_dot_product_attention``.

    Correct for both layouts: a packed batch gets an explicit block-diagonal
    causal mask, so it matches ``varlen`` numerically. That mask is ``(T, T)``,
    so this is the reference and the eval path, not the training path.
    """

    def forward(
        self, x: torch.Tensor, seq_info, posenc=None, cache=None
    ) -> torch.Tensor:
        if cache is not None:
            return self.attend_cached(x, posenc, cache)
        q, k, v = self.project(x, posenc)
        if seq_info.packed:
            doc_id = _doc_ids(seq_info, q.shape[0], q.device)
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
            return self.merge(_sdpa_attend(self, q, k, v, doc_id).squeeze(0))
        return self.merge(_sdpa_attend(self, q, k, v))


@ATTENTION.register("flex")
class FlexAttention(BaseAttention):
    """FlexAttention with a compiled document + causal (+ window) block mask.

    The most general backend: arbitrary masking without materializing scores,
    and slower than ``varlen`` for plain causal work on sm_120.
    """

    _compiled = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mask_cache: dict = {}

    @classmethod
    def _kernel(cls):
        """Compiled ``flex_attention``, once per process and shared across layers.

        Compilation is mandatory, not an optimization. See docs/concepts/architecture.md.
        """
        if cls._compiled is None:
            from torch.nn.attention.flex_attention import flex_attention

            cls._compiled = torch.compile(flex_attention, dynamic=False)
        return cls._compiled

    def _block_mask(self, seq_info, q_len: int, device):
        from torch.nn.attention.flex_attention import create_block_mask

        window = self.sliding_window
        key = (q_len, seq_info.packed, window, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        if seq_info.packed:
            # doc_id[t] = index of the document token t belongs to
            doc_id = torch.zeros(q_len, dtype=torch.int32, device=device)
            doc_id[seq_info.cu_seqlens[1:-1].long()] = 1
            doc_id = doc_id.cumsum(0)

            def mask_mod(b, h, qi, kv):
                causal = qi >= kv
                same_doc = doc_id[qi] == doc_id[kv]
                if window is None:
                    return causal & same_doc
                return causal & same_doc & (qi - kv < window)

            block_mask = create_block_mask(
                mask_mod, None, None, q_len, q_len, device=device
            )
        else:

            def mask_mod(b, h, qi, kv):
                causal = qi >= kv
                if window is None:
                    return causal
                return causal & (qi - kv < window)

            block_mask = create_block_mask(
                mask_mod, None, None, q_len, q_len, device=device
            )
        self._mask_cache[key] = block_mask
        return block_mask

    def forward(
        self, x: torch.Tensor, seq_info, posenc=None, cache=None
    ) -> torch.Tensor:
        if cache is not None:
            return self.attend_cached(x, posenc, cache)
        q, k, v = self.project(x, posenc)
        packed = seq_info.packed
        if packed:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        block_mask = self._block_mask(seq_info, q.shape[-2], q.device)
        out = self._kernel()(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=self.scale,
            enable_gqa=self.kv_heads != self.heads,
        )
        out = out.transpose(1, 2)
        if packed:
            out = out.squeeze(0)
        return self.merge(out)
