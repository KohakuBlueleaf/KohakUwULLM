"""torchao's quantized-state AdamW variants, behind the optional torchao dependency.

Quantizes the Adam moments only. See docs/internals/optimizers.md.
"""

from functools import partial
from typing import Any

from kohakuwullm.registry import OPTIMIZER

# Registry key -> torchao class name.
QUANTIZED_ADAMW: dict[str, str] = {
    "adamw8bit": "AdamW8bit",
    "adamw4bit": "AdamW4bit",
    "adamwfp8": "AdamWFp8",
}


def build_quantized_adamw(class_name: str, params: Any, **kwargs: Any):
    """Construct one of torchao's quantized AdamW variants, importing it here."""
    try:
        from torchao import optim as ao_optim
    except ImportError as exc:
        raise ImportError(
            f"optimizer {class_name!r} needs torchao: `uv pip install torchao`"
        ) from exc
    return getattr(ao_optim, class_name)(params, **kwargs)


for _key, _class_name in QUANTIZED_ADAMW.items():
    OPTIMIZER.register(_key)(partial(build_quantized_adamw, _class_name))
