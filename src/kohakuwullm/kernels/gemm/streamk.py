"""Stream-K bf16/fp16 GEMM for sm_120, driven by a `Plan`.

`gemm(a, b, dev)` picks a plan for the shape and runs it. Tiles that divide
evenly across the CTAs are computed whole and stored directly; the leftover
tiles are split along K into an fp32 scratch that a fixup pass casts into C.

See docs/performance/gemm.md section 5d.
"""

import torch
import triton
import triton.language as tl

from kohakuwullm.kernels.gemm.device import Device
from kohakuwullm.kernels.gemm.plan import Plan, plan


@triton.jit
def _coords(tile, grid_m, grid_n, GROUP_M: tl.constexpr):
    """Flat tile index to (pid_m, pid_n) under group-M rasterization."""
    if GROUP_M == 1:
        return tile // grid_n, tile % grid_n
    width = GROUP_M * grid_n
    group = tile // width
    rows = min(grid_m - group * GROUP_M, GROUP_M)
    return group * GROUP_M + (tile % rows), (tile % width) // rows


@triton.jit
def _mac(
    a_ptr,
    b_ptr,
    pid_m,
    pid_n,
    k0,
    k1,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Accumulate one output tile over k-iterations ``[k0, k1)``; returns fp32."""
    offs_m = tl.max_contiguous(
        tl.multiple_of((pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M, BLOCK_M), BLOCK_M
    )
    offs_n = tl.max_contiguous(
        tl.multiple_of((pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N, BLOCK_N), BLOCK_N
    )
    offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(k0, k1):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            # Zeroing A alone kills the tail; B only needs to stay in bounds.
            mk = (kk * BLOCK_K + tl.arange(0, BLOCK_K)) < K
            a = tl.load(a_ptrs, mask=mk[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mk[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    return acc


@triton.jit
def _streamk(
    a_ptr,
    b_ptr,
    c_ptr,
    p_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    iters_per_tile,
    sk_tiles,
    dp_tiles,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SK_CTAS: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """A K-split share of the leftover tiles, then whole tiles stored to C."""
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    sk_total = sk_tiles * iters_per_tile
    it = pid * sk_total // SK_CTAS
    end = (pid + 1) * sk_total // SK_CTAS
    while pid < SK_CTAS and it < end:
        tile = it // iters_per_tile
        k0 = it - tile * iters_per_tile
        k1 = min(iters_per_tile, k0 + (end - it))
        pid_m, pid_n = _coords(tile, grid_m, grid_n, GROUP_M)
        acc = _mac(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            k0,
            k1,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            EVEN_K,
        )
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        tl.atomic_add(
            p_ptr
            + tile * BLOCK_M * BLOCK_N
            + offs_m[:, None] * BLOCK_N
            + offs_n[None, :],
            acc,
            sem="relaxed",
        )
        it += k1 - k0

    for i in range(pid, dp_tiles, NUM_CTAS):
        tile = sk_tiles + i
        pid_m, pid_n = _coords(tile, grid_m, grid_n, GROUP_M)
        acc = _mac(
            a_ptr,
            b_ptr,
            pid_m,
            pid_n,
            0,
            iters_per_tile,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            EVEN_K,
        )
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc.to(c_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


@triton.jit
def _fixup(
    c_ptr,
    p_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    grid_m,
    grid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Cast the split tiles' fp32 partials into C."""
    tile = tl.program_id(0)
    pid_m, pid_n = _coords(tile, grid_m, grid_n, GROUP_M)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    p = p_ptr + tile * BLOCK_M * BLOCK_N + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    acc = tl.load(p)
    om = pid_m * BLOCK_M + offs_m
    on = pid_n * BLOCK_N + offs_n
    tl.store(
        c_ptr + om[:, None] * stride_cm + on[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=(om[:, None] < M) & (on[None, :] < N),
    )


class StreamKGemm:
    """A planned GEMM bound to one shape, holding its fixup scratch."""

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dev: Device,
        elem_bytes: int = 2,
        group_m: int = 8,
        p: Plan | None = None,
    ):
        self.plan = p or plan(m, n, k, dev, elem_bytes)
        self.m, self.n, self.k = m, n, k
        self.group_m = group_m
        bm, bn, bk = self.plan.tile
        self.grid_m, self.grid_n = triton.cdiv(m, bm), triton.cdiv(n, bn)
        self.dp_tiles = self.plan.tiles - self.plan.sk_tiles
        self.scratch = torch.zeros(
            max(self.plan.sk_tiles, 1) * bm * bn, device="cuda", dtype=torch.float32
        )
        self.handle = None

    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None
    ) -> torch.Tensor:
        p = self.plan
        bm, bn, bk = p.tile
        if c is None:
            c = torch.empty((self.m, self.n), device=a.device, dtype=a.dtype)
        extra = {"maxnreg": p.maxnreg} if p.maxnreg else {}
        if p.sk_tiles:
            self.scratch.zero_()
        self.handle = _streamk[(p.ctas,)](
            a,
            b,
            c,
            self.scratch,
            self.m,
            self.n,
            self.k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            p.iters,
            p.sk_tiles,
            self.dp_tiles,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=self.group_m,
            NUM_CTAS=p.ctas,
            SK_CTAS=max(p.sk_ctas, 1),
            EVEN_K=(self.k % bk == 0),
            num_warps=p.warps,
            num_stages=p.stages,
            **extra,
        )
        if p.sk_tiles:
            _fixup[(p.sk_tiles,)](
                c,
                self.scratch,
                self.m,
                self.n,
                c.stride(0),
                c.stride(1),
                self.grid_m,
                self.grid_n,
                BLOCK_M=bm,
                BLOCK_N=bn,
                GROUP_M=self.group_m,
                num_warps=4,
            )
        return c


def gemm(a: torch.Tensor, b: torch.Tensor, dev: Device) -> torch.Tensor:
    """``a @ b`` under a freshly computed plan; use `StreamKGemm` to reuse one."""
    m, k = a.shape
    _, n = b.shape
    return StreamKGemm(m, n, k, dev, a.element_size())(a, b)
