"""Declaration protocol for matmul a module holds as bare parameters.

See docs/internals/mxfp8.md.
"""

from dataclasses import dataclass

# Attribute names `swap_mxfp8` looks up on a module.
DECLARE = "mxfp8_matmul"
CONVERT = "enable_mxfp8"


@dataclass(frozen=True)
class Matmul:
    """One matmul a module holds as a bare parameter.

    Args:
        mac_per_token: multiply-accumulates one token spends against this tensor,
            counting only the experts that run for it.
        refusal: why fp8 must not take this tensor, or ``None`` if it may.
        never: ``True`` if the refusal follows from the tensor's job, ``False`` if it
            follows from this model's shapes.
    """

    mac_per_token: int
    refusal: str | None = None
    never: bool = False
