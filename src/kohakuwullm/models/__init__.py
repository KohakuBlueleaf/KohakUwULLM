"""Model package.

Importing it registers all built-in components into their registries.
"""

from kohakuwullm.models import components  # noqa: F401  (registers built-ins)
from kohakuwullm.models.backbone import LMBackbone
from kohakuwullm.models.block import DecoderBlock
from kohakuwullm.models.cache import KVCache
from kohakuwullm.models.components.seqinfo import SeqInfo
from kohakuwullm.models.flops import FlopCounter, document_lengths
from kohakuwullm.models.head import LMHead
from kohakuwullm.models.presets import (
    KOHAKU_LADDER,
    PRESETS,
    LMArchConfig,
    get_preset,
)

__all__ = [
    "LMBackbone",
    "DecoderBlock",
    "KVCache",
    "FlopCounter",
    "document_lengths",
    "LMHead",
    "LMArchConfig",
    "KOHAKU_LADDER",
    "PRESETS",
    "get_preset",
    "SeqInfo",
    "components",
]
