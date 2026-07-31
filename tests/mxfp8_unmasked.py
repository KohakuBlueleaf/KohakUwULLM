"""Rebuild a kernel module with its scale-column mask removed, as a control.

Not a test module. Two suites need the same negative control -- the dense
pre-quantized GEMM and the four fused expert kernels -- and the mask they strip
is one string, so it is defined once here rather than restated in each.

The control has to be *executed*, not asserted in a docstring: the probe in
either suite is only evidence that the mask works if the unmasked twin is shown
to fail the same probe. Rewriting module source is how
:mod:`kohakuwullm.bench.vendor.mxfp8_rounding` builds its own variant and for the same
reason -- ``triton.jit`` reads a kernel through ``inspect.getsource``, so the
twin has to be a real file rather than an ``exec`` of a string.

Every substitution count is asserted. A rewrite that matched nothing would
silently make the control a second copy of the fixed code, which is the one
failure mode that turns both tests green while proving nothing.
"""

import importlib
import importlib.util
import pathlib
import sys
import tempfile
import types

from kohakuwullm.kernels.mxfp8 import mxfp8_matmul_pq

_UNMASKED_PQ = "kohakuwullm.kernels._mxfp8_unmasked_scale_columns"
SCALE_COLUMN_MASK = " & mask_g[None, :]"
_unmasked_pq_cache: types.ModuleType | None = None


def load_unmasked_pq() -> types.ModuleType:
    """A copy of ``kernels.mxfp8`` with the scale-column mask removed again.

    The negative control for a *silent* out-of-bounds read has to be executed, not
    asserted in a docstring: the probe below is only evidence that the mask works
    if the unmasked kernel is shown to fail the same probe. Rewriting the module
    source is how :mod:`kohakuwullm.bench.vendor.mxfp8_rounding` builds its own variant
    and for the same reason -- ``triton.jit`` reads a kernel through
    ``inspect.getsource``, so the twin has to be a real file rather than an
    ``exec`` of a string.

    Cached per process because the copy carries its own ``triton.autotune`` state,
    and the substitution count is asserted because a rewrite that matched nothing
    would make the control a second copy of the fixed kernel -- the one failure
    mode that would turn this test green while proving nothing.
    """
    global _unmasked_pq_cache
    if _unmasked_pq_cache is not None:
        return _unmasked_pq_cache
    path = pathlib.Path(sys.modules[mxfp8_matmul_pq.__module__].__file__)
    text = path.read_text()
    found = text.count(SCALE_COLUMN_MASK)
    assert found == 2, (
        f"{path.name}: expected 2 scale-column masks, found {found}. The unmasked "
        "control would be identical to the fixed kernel."
    )
    copy = pathlib.Path(tempfile.mkdtemp(prefix="mxfp8_unmasked_")) / "mxfp8.py"
    copy.write_text(text.replace(SCALE_COLUMN_MASK, ""))
    spec = importlib.util.spec_from_file_location(_UNMASKED_PQ, copy)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_UNMASKED_PQ] = module
    spec.loader.exec_module(module)
    _unmasked_pq_cache = module
    return module


_UNMASKED_EXPERTS = "kohakuwullm_unmasked_experts"
_EXPERTS_PACKAGE = "kohakuwullm.kernels.mxfp8"
# The modules whose kernels carry a scale-column mask, in dependency order. The
# twins must be built as a *set*: `moe` imports the kernels by name, so rewriting
# one file and leaving its importer pointing at the real one would produce a
# control that runs the fixed kernels. `moe` carries no mask of its own and is
# copied only because it is the entry point that has to reach the twins.
_EXPERTS_MODULES = ("grouped", "experts", "experts_bwd", "moe")
_EXPECTED_MASKS = {"grouped": 3, "experts": 5, "experts_bwd": 4, "moe": 0}
_unmasked_experts_cache: types.ModuleType | None = None


def load_unmasked_experts() -> types.ModuleType:
    """The fused expert path rebuilt with every scale-column mask removed.

    The negative control for a *silent* out-of-bounds read has to be executed. The
    single-file trick :func:`load_unmasked_pq` uses does not reach here, because the
    autograd wiring imports the kernels from a sibling module -- so the whole
    four-module chain is copied and its internal imports repointed at the copies.

    ``kernels.mxfp8`` is deliberately **not** copied. Its own masks have their own
    test, and leaving it shared keeps this control isolated to the grouped and expert
    kernels rather than varying two things at once.

    Every per-file substitution count is asserted, because a rewrite that matched
    nothing would silently make the control a second copy of the fixed code -- the
    one failure mode that turns this test green while proving nothing.
    """
    global _unmasked_experts_cache
    if _unmasked_experts_cache is not None:
        return _unmasked_experts_cache
    root = (
        pathlib.Path(tempfile.mkdtemp(prefix="unmasked_experts_")) / _UNMASKED_EXPERTS
    )
    root.mkdir()
    (root / "__init__.py").write_text("")
    for name in _EXPERTS_MODULES:
        source = pathlib.Path(
            importlib.import_module(f"{_EXPERTS_PACKAGE}.{name}").__file__
        ).read_text()
        found = source.count(SCALE_COLUMN_MASK)
        assert found == _EXPECTED_MASKS[name], (
            f"{name}.py: expected {_EXPECTED_MASKS[name]} scale-column masks, found "
            f"{found}. The unmasked control would not differ from the fixed code."
        )
        source = source.replace(SCALE_COLUMN_MASK, "")
        # Only the dotted paths *into* the package are repointed. A bare
        # `from kohakuwullm.kernels.mxfp8 import ...` keeps reaching the real
        # package, which is what leaves the quantizer shared.
        for sibling in _EXPERTS_MODULES:
            source = source.replace(
                f"{_EXPERTS_PACKAGE}.{sibling}", f"{_UNMASKED_EXPERTS}.{sibling}"
            )
        (root / f"{name}.py").write_text(source)

    # Appended, not prepended. The twin package name is unique, so the front of the
    # path buys nothing, and a temp directory sitting at position 0 for the rest of
    # the session is the shape of hazard that let `TRITON_INTERPRET` leak out of one
    # test module and re-decorate every kernel imported after it: process-global
    # import state, set by a test, outliving the test that set it.
    sys.path.append(str(root.parent))
    _unmasked_experts_cache = importlib.import_module(f"{_UNMASKED_EXPERTS}.moe")
    return _unmasked_experts_cache


def unmasked_expert_module(name: str) -> types.ModuleType:
    """One module of the twin package, e.g. ``"grouped"``.

    The whole set is built together -- ``moe`` imports its siblings by name, so a
    twin that reached the real ones would run the fixed kernels -- so this goes
    through :func:`load_unmasked_experts` rather than copying one file.
    """
    if name not in _EXPERTS_MODULES:
        raise KeyError(f"{name!r} is not one of {_EXPERTS_MODULES}")
    load_unmasked_experts()
    return importlib.import_module(f"{_UNMASKED_EXPERTS}.{name}")
