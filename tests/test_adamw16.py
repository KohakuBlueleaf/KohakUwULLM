"""Tests for the 16-bit-state fused AdamW.

Separate file rather than appended to ``test_kernels.py`` only because that file
had three concurrent authors when this was written; it belongs there.

Every test here runs under ``TRITON_INTERPRET=1``, which executes the real
Triton kernel on the CPU. That is not a substitute for hardware -- the
interpreter does not enforce every constraint the compiler does, and a
non-``constexpr`` module global inside ``@jit`` has passed interpretation and
then failed to compile -- but it does catch the Triton-semantics bugs a pure
torch reference reproduces *differently* and therefore agrees with.

**In a subprocess, because the interpreter is process-global and latched at
decoration time.** ``@triton.jit`` chooses ``InterpretedFunction`` over
``JITFunction`` when ``TRITON_INTERPRET`` is set *at the moment the kernel module is
imported*, and pytest imports every test module during collection -- so setting the
variable at this module's top level re-decorated every kernel module imported after
it. That silently ran the whole suite's Triton surface on the CPU:
``test_swiglu_matches_eager`` went from 15s to past 120s, and the ULP tests were
measuring numpy's arithmetic rather than the tensor cores they exist to check.
Spawning a child keeps the variable out of the parent session entirely, which is the
only isolation that works when the choice is made at import.
"""

import os
import subprocess
import sys

import pytest
import torch

# Set only in the child. The parent must never define it -- see the module docstring.
INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"

if INTERPRETED:
    from kohakuwullm.kernels.optim.adamw16 import adamw16_step

# The real tests below run only in the child; the parent contributes the one test
# that spawns it and holds it to a count, so a child that collects nothing fails
# rather than passing vacuously.
interpreted_only = pytest.mark.skipif(
    not INTERPRETED,
    reason="runs in the child spawned by test_the_interpreted_suite_runs_and_passes",
)
REAL_TESTS = 5


def test_the_interpreted_suite_runs_and_passes():
    """Run this file again with the interpreter on, in a process of its own."""
    if INTERPRETED:
        pytest.skip("already the child")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider"],
        env={**os.environ, "TRITON_INTERPRET": "1"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # Pinning the count, not just the exit status: a collection error or a rename
    # that skips everything also exits 0 with "no tests ran".
    assert f"{REAL_TESTS} passed" in proc.stdout, proc.stdout


def _reference(p, g, m, r, step, lr, betas, eps, wd, store_root=True):
    """fp64 AdamW. ``store_root=False`` is the bug the design exists to avoid."""
    b1, b2 = betas
    p64, g64 = p.double(), g.double()
    m64 = b1 * m.double() + (1 - b1) * g64
    v_prev = r.double() ** 2 if store_root else r.double()
    v64 = b2 * v_prev + (1 - b2) * g64 * g64
    mhat = m64 / (1 - b1**step)
    vhat = v64 / (1 - b2**step)
    upd = mhat / (vhat.sqrt() + eps)
    return p64 - lr * (upd + wd * p64), m64, (v64.sqrt() if store_root else v64)


@interpreted_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_adamw16_tracks_an_fp64_reference(dtype):
    """Several steps against fp64, so bias correction is exercised, not just one step."""
    torch.manual_seed(0)
    n = 4096
    p = (torch.randn(n) * 0.02).to(dtype)
    m = torch.zeros(n, dtype=dtype)
    r = torch.zeros(n, dtype=dtype)
    p64 = p.double()
    m64 = torch.zeros(n, dtype=torch.float64)
    r64 = torch.zeros(n, dtype=torch.float64)

    lr, betas, eps, wd = 1e-3, (0.9, 0.95), 1e-8, 0.1
    for step in range(1, 6):
        g = (torch.randn(n) * 1e-3).to(dtype)
        adamw16_step(p, g, m, r, step, lr, betas, eps, wd)
        p64, m64, r64 = _reference(p64, g, m64, r64, step, lr, betas, eps, wd)

    # 16-bit storage bounds this: three tensors are rounded every step, so the
    # tolerance is the dtype's, not the algorithm's.
    tol = 4e-2 if dtype is torch.bfloat16 else 6e-3
    rel = (p.double() - p64).norm() / p64.norm()
    assert rel < tol, f"{dtype} drifted {rel:.2e} from fp64 over 5 steps"


@interpreted_only
def test_storing_v_instead_of_its_root_underflows_in_fp16():
    """The control for the whole design: ``sqrt(v)`` is not a stylistic choice.

    fp16's smallest subnormal is ~6e-8, so a gradient of 1e-4 gives ``v = 1e-8``
    which flushes to zero -- and then the update divides by ``sqrt(0) + eps``,
    i.e. multiplies by 1e8. Storing the root keeps the same quantity at 1e-4,
    comfortably normal. This test fails if anyone "simplifies" the state back to
    ``v``, which no accuracy comparison against fp32 would reveal because fp32
    has the exponent range to hide it.
    """
    g = torch.full((256,), 1e-4, dtype=torch.float16)
    v = (g.double() ** 2).to(torch.float16)
    assert (v == 0).all(), "premise changed: grad**2 no longer underflows in fp16"
    assert (g != 0).all(), "sqrt(v) must stay representable where v does not"

    p = torch.zeros(256, dtype=torch.float16)
    m = torch.zeros(256, dtype=torch.float16)
    r = torch.zeros(256, dtype=torch.float16)
    adamw16_step(p, g, m, r, 1, 1e-3, (0.9, 0.95), 1e-8, 0.0)

    # One step from zero state gives |update| ~= lr regardless of gradient scale --
    # that scale-invariance is what Adam is for. A v-storing kernel would divide
    # by eps here and move the parameter by ~1e5 * lr instead.
    assert torch.isfinite(p).all()
    assert (p.abs() < 2e-3).all(), f"update blew up: max {p.abs().max().item():.3e}"


@interpreted_only
def test_stochastic_writeback_refuses_a_dtype_its_construction_does_not_fit():
    """``+rand16 & 0xFFFF0000`` is exact for bf16 and a no-op for fp16.

    Exact for bf16 because the 16 random bits *are* the discarded bits. For fp16
    the discarded width is ``max(13, -1-e)`` and varies with the exponent, so the
    same mask truncates to a finer grid than fp16's and the following cast
    re-rounds -- leaving plain round-to-nearest wearing an SR costume. Refusing is
    the only honest option until the fp16 construction is derived separately.
    """
    kw = dict(step=1, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    for dtype in (torch.float16, torch.float32):
        t = torch.zeros(64, dtype=dtype)
        with pytest.raises(ValueError, match="bf16-only"):
            adamw16_step(t, t.clone(), t.clone(), t.clone(), stochastic=True, **kw)


@interpreted_only
def test_mismatched_state_shape_is_rejected_rather_than_read_past():
    """A short state tensor would otherwise be indexed to the param's numel."""
    p = torch.zeros(128, dtype=torch.bfloat16)
    short = torch.zeros(64, dtype=torch.bfloat16)
    kw = dict(step=1, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    with pytest.raises(ValueError, match="exp_avg"):
        adamw16_step(p, p.clone(), short, p.clone(), **kw)
