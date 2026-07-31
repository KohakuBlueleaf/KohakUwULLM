"""KohakUwULLM -- an extensible decoder-only LLM training framework.

Importing the package registers all built-in components into their registries,
so ``build(spec, REGISTRY)`` works immediately. The Lightning training stack
lives in ``kohakuwullm.training`` and must be imported explicitly (it pulls in
Lightning); the data stack lives in ``kohakuwullm.data``.

See ``docs/`` for usage and ``configs/lm/`` for runnable recipes.
"""

from kohakuwullm import models  # noqa: F401  (registers built-ins)
from kohakuwullm.compile_utils import apply_compile
from kohakuwullm.models import (
    KOHAKU_LADDER,
    PRESETS,
    DecoderBlock,
    KVCache,
    LMArchConfig,
    LMBackbone,
    LMHead,
    SeqInfo,
    get_preset,
)
from kohakuwullm.registry import (
    ATTENTION,
    DATASET,
    MLP,
    NORM,
    OPTIMIZER,
    POSENC,
    RENDERER,
    ROUTER,
    Registry,
    build,
)
from kohakuwullm.utils import count_active_parameters, count_parameters, human_count

__version__ = "0.2.0"

__all__ = [
    "build",
    "Registry",
    "NORM",
    "MLP",
    "ATTENTION",
    "POSENC",
    "ROUTER",
    "DATASET",
    "RENDERER",
    "OPTIMIZER",
    "LMBackbone",
    "LMArchConfig",
    "LMHead",
    "DecoderBlock",
    "KVCache",
    "SeqInfo",
    "KOHAKU_LADDER",
    "PRESETS",
    "get_preset",
    "apply_compile",
    "count_parameters",
    "count_active_parameters",
    "human_count",
    "__version__",
]
