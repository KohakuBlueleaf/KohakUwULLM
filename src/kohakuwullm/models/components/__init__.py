"""Swappable transformer components.

Importing this package registers every built-in into its registry (``NORM``,
``MLP``, ``ATTENTION``, ``POSENC``, ``ROUTER``) as a side effect.
"""

from kohakuwullm.models.components import (
    attention,
    mlp,
    moe,
    moe_variants,
    ndrope,
    norm,
    posenc,
)
from kohakuwullm.models.components.seqinfo import SeqInfo

__all__ = [
    "attention",
    "mlp",
    "moe",
    # Imported for its registry side effects, not for a name.
    "moe_variants",
    "ndrope",
    "norm",
    "posenc",
    "SeqInfo",
]
