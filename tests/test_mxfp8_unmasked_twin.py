"""The scale-column negative controls must be *armed*, and that is checkable on CPU.

Two suites prove their kernels never read past the MX scale width by running a copy
of the kernel with the mask removed and requiring it to return NaN. Those probes
need a device. What does not need a device is the question that actually went wrong
today: **does the twin still get built from the right modules at all?**

When ``kernels/mxfp8_experts.py`` became ``kernels/mxfp8/experts.py``, the loader's
hardcoded module names stopped resolving. On a GPU box that surfaces as an error; on
any box, it means the control is disarmed. And the quieter variant is worse -- a
rewrite that matched *nothing* leaves the twin byte-identical to the fixed kernel, so
the probe runs, returns finite output, and the assertion that was supposed to catch
an out-of-bounds read reports that there is none.

So this file pins the machinery rather than the numerics, and carries no CUDA mark:
importing a ``triton.jit`` module does not compile it, so every check below runs
where no card exists. What it deliberately does **not** prove is that the unmasked
kernel returns non-finite output -- that is the probe's job, in
``test_kernels_mxfp8_experts_masking.py`` and ``test_kernels_mxfp8_quantize.py``.
"""

import pathlib
import sys

from mxfp8_unmasked import (
    _EXPECTED_MASKS,
    _EXPERTS_MODULES,
    _EXPERTS_PACKAGE,
    SCALE_COLUMN_MASK,
    load_unmasked_pq,
    unmasked_expert_module,
)

from kohakuwullm.kernels.mxfp8 import grouped, quantize


def _source_of(module) -> str:
    return pathlib.Path(module.__file__).read_text()


def test_every_expert_twin_is_built_and_actually_unmasked():
    """Real module has N masks, twin has zero, for every module in the chain.

    ``N`` is asserted against ``_EXPECTED_MASKS`` inside the loader already; the
    half that only this test covers is the ``-> 0``. A loader that copied the file
    without substituting would satisfy the count check and still hand back the
    fixed kernel.
    """
    for name in _EXPERTS_MODULES:
        twin = unmasked_expert_module(name)
        real = sys.modules[f"{_EXPERTS_PACKAGE}.{name}"]
        assert _source_of(real).count(SCALE_COLUMN_MASK) == _EXPECTED_MASKS[name]
        assert _source_of(twin).count(SCALE_COLUMN_MASK) == 0, f"{name} kept a mask"


def test_the_twin_is_a_separate_module_not_an_alias():
    """Identity, not just source text: the probe must call a different function.

    A twin that re-exported the real kernel would pass the source check above --
    the copied *file* has no mask -- while every call still ran the fixed code.
    """
    twin = unmasked_expert_module("grouped")
    assert twin is not grouped
    assert twin.grouped_mxfp8_gemm is not grouped.grouped_mxfp8_gemm

    pq = load_unmasked_pq()
    assert pq.mxfp8_matmul_pq is not quantize.mxfp8_matmul_pq
    assert _source_of(pq).count(SCALE_COLUMN_MASK) == 0


def test_no_twin_module_imports_its_siblings_from_the_real_package():
    """``moe`` reaches its kernels by name, so one missed rewrite disarms the rest.

    The entry point carries no mask of its own and is copied only to be repointed.
    If that repoint fails it still imports and still runs -- against the fixed
    kernels -- which is the failure this asserts away.
    """
    for name in _EXPERTS_MODULES:
        leaked = [
            line.strip()
            for line in _source_of(unmasked_expert_module(name)).splitlines()
            if f"{_EXPERTS_PACKAGE}." in line and "import" in line
        ]
        assert (
            not leaked
        ), f"{name} still imports siblings from the real package: {leaked}"
