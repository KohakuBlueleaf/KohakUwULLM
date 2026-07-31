"""Every kernel that advertises a CPU fallback must actually take it.

Its own file because it is the one part of the kernel suite that must **not**
carry the ``requires CUDA`` module mark. These tests used to sit under that mark
in ``test_kernels.py``, which meant the CPU path was only ever exercised on a
box that had a GPU -- i.e. never in the situation the fallback exists for. A
fallback that is only tested where it is not used is not tested.
"""

import torch
import torch.nn.functional as F

from kohakuwullm.kernels.elementwise.rmsnorm import rms_norm as triton_rms_norm
from kohakuwullm.kernels.elementwise.swiglu import swiglu_mul


def test_rmsnorm_falls_back_on_cpu():
    x = torch.randn(8, 64)
    w = torch.ones(64)
    assert torch.allclose(triton_rms_norm(x, w), F.rms_norm(x, (64,), w, 1e-6))


def test_swiglu_falls_back_on_cpu():
    g = torch.randn(8, 64)
    v = torch.randn(8, 64)
    assert torch.allclose(swiglu_mul(g, v), F.silu(g) * v)
