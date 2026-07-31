"""A round-to-nearest twin of the MXFP8 quantizer, for A/B against round-up.

Built by rewriting the source of :mod:`kohakuwullm.kernels.mxfp8.quantize` onto disk
and importing the copy under a second module name; the shipped file is never
modified. Both rewrites assert their substitution count. See docs/performance/ab-testing.md.
"""

import importlib.util
import pathlib
import sys
import tempfile
import types
from typing import NamedTuple

from kohakuwullm.kernels.mxfp8 import linear as mxfp8_linear
from kohakuwullm.kernels.mxfp8 import quantize as mxfp8

# Module names the rewritten copies are registered under.
RTN_QUANTIZER = "kohakuwullm.kernels._mxfp8_round_to_nearest"
RTN_LINEAR = "kohakuwullm.kernels._mxfp8_linear_round_to_nearest"

_CEIL = "tl.ceil("
_NEAREST = "tl.floor(0.5 + "
_QUANTIZER_IMPORT = "from kohakuwullm.kernels.mxfp8 import"


class RoundToNearest(NamedTuple):
    """The two rewritten modules. ``linear.MXFP8Linear`` is the drop-in class."""

    quantizer: types.ModuleType
    linear: types.ModuleType


_cache: RoundToNearest | None = None


def _rewrite(path: pathlib.Path, old: str, new: str, expected: int) -> str:
    """``path``'s source with ``old`` replaced, or a refusal if the count is wrong."""
    source = path.read_text()
    found = source.count(old)
    if found != expected:
        raise RuntimeError(
            f"{path}: expected {expected} occurrence(s) of {old!r}, found {found}. "
            "The round-to-nearest variant would be identical to round-up."
        )
    return source.replace(old, new)


def _load(name: str, path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so the linear copy can import it by name.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_round_to_nearest() -> RoundToNearest:
    """Build and import the round-to-nearest twins of both MXFP8 modules.

    Cached per process; the copies carry their own ``triton.autotune`` state.
    """
    global _cache
    if _cache is not None:
        return _cache

    root = pathlib.Path(tempfile.mkdtemp(prefix="mxfp8_rtn_"))
    quantizer_src = _rewrite(pathlib.Path(mxfp8.__file__), _CEIL, _NEAREST, 1)
    quantizer_path = root / "mxfp8_round_to_nearest.py"
    quantizer_path.write_text(quantizer_src)
    quantizer = _load(RTN_QUANTIZER, quantizer_path)

    linear_src = _rewrite(
        pathlib.Path(mxfp8_linear.__file__),
        _QUANTIZER_IMPORT,
        f"from {RTN_QUANTIZER} import",
        1,
    )
    linear_path = root / "mxfp8_linear_round_to_nearest.py"
    linear_path.write_text(linear_src)
    _cache = RoundToNearest(quantizer, _load(RTN_LINEAR, linear_path))
    return _cache
