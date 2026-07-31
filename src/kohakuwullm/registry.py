"""Build-time component dispatch: named registries and a dotted-path escape hatch.

See docs/guides/extending.md for the accepted spec forms.
"""

from collections.abc import Callable
from typing import Any

from kohakuwullm.utils import import_class


class Registry:
    """A named string -> class/callable map with a decorator-based ``register``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str | None = None) -> Callable[[Callable], Callable]:
        """A decorator registering the wrapped object under ``key`` or its ``__name__``."""

        def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            name = key or obj.__name__
            if name in self._items:
                raise KeyError(f"{self.name!r} already has an entry named {name!r}")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        """The entry under ``key``, or a ``KeyError`` listing what is registered."""
        if key not in self._items:
            raise KeyError(f"unknown {self.name} {key!r}; registered: {self.keys()}")
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> list[str]:
        return sorted(self._items)


# One registry per swappable axis; built-ins register into these at import time.
NORM = Registry("norm")
MLP = Registry("mlp")
ATTENTION = Registry("attention")
POSENC = Registry("posenc")
ROUTER = Registry("router")  # MoE token -> expert assignment
DATASET = Registry("dataset")  # named record sources
RENDERER = Registry("renderer")  # record -> (user_text, output_text)
OPTIMIZER = Registry("optimizer")
LOADER = Registry("loader")  # dataset -> batches


def resolve(name: str, registry: Registry | None = None) -> Callable[..., Any]:
    """The class or callable ``name`` selects, **without** constructing it.

    For callers that supply constructor arguments themselves.
    """
    return _resolve(name, registry)


def _resolve(name: str, registry: Registry | None) -> Callable[..., Any]:
    """Dotted paths import; anything else is a registry key."""
    if "." in name:
        return import_class(name)
    if registry is None:
        raise ValueError(f"{name!r} is not a dotted path and no registry was provided")
    return registry.get(name)


def build(spec: Any, registry: Registry | None = None, **kwargs: Any) -> Any:
    """Resolve ``spec`` to a concrete object, passing ``kwargs`` at construction.

    Build-time only -- never call this inside a training loop.
    """
    match spec:
        case None:
            return None
        case dict():
            opts = dict(spec)
            name = opts.pop("name")
            return _resolve(name, registry)(**opts, **kwargs)
        case str():
            return _resolve(spec, registry)(**kwargs)
        case type():
            return spec(**kwargs)
        # A typed spec dataclass.
        case _ if hasattr(spec, "build") and callable(spec.build):
            return spec.build()
        # An already-built instance.
        case _:
            return spec
