"""Rank-aware structured logging: ``[HH:MM:SS] [rankN] [module] [LEVEL] message``.

Only rank 0 writes to stderr by default, so a four-rank run does not print
everything four times; ``KWP_LOG_ALL_RANKS=1`` turns the others back on.
See docs/kohakuwupipe/logging.md.
"""

import logging
import os
import sys
from typing import Any

COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[95m",
    "RESET": "\033[0m",
}

ROOT = "kohakuwupipe"
_handler: logging.Handler | None = None


def _supports_color() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class RankFilter(logging.Filter):
    """Tags every record with its rank, and drops non-zero ranks unless asked."""

    def __init__(self, rank: int, all_ranks: bool) -> None:
        super().__init__()
        self.rank = rank
        self.all_ranks = all_ranks

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        return self.all_ranks or self.rank == 0


class Formatter(logging.Formatter):
    """Colored, with any extra kwargs appended as ``key=value``."""

    SKIP = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
        "message",
        "asctime",
        "taskName",
    }

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color and _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        extras = [
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in record.__dict__.items()
            if k not in self.SKIP and k != "rank"
        ]
        message = record.getMessage()
        if extras:
            message = f"{message} [{', '.join(extras)}]"
        name = record.name.replace(f"{ROOT}.", "")
        head = (
            f"[{self.formatTime(record, '%H:%M:%S')}] "
            f"[rank{getattr(record, 'rank', 0)}] [{name}] [{record.levelname}]"
        )
        if not self.use_color:
            return f"{head} {message}"
        color = COLORS.get(record.levelname, "")
        return f"{color}{head} {message}{COLORS['RESET']}"


class Logger(logging.Logger):
    """Accepts extra fields as keyword arguments: ``log.info("x", step=3)``."""

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            extra = {**(extra or {}), **kwargs}
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel + 1)


logging.setLoggerClass(Logger)


def configure(rank: int = 0, level: int | str = logging.INFO) -> None:
    """Install the stderr handler. Idempotent; call once per process."""
    global _handler
    root = logging.getLogger(ROOT)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if _handler is None:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(Formatter())
        root.addHandler(_handler)
        root.propagate = False
    all_ranks = bool(os.environ.get("KWP_LOG_ALL_RANKS"))
    for existing in list(_handler.filters):
        _handler.removeFilter(existing)
    _handler.addFilter(RankFilter(rank, all_ranks))
    _handler.setLevel(level)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """A logger under the package root; configure() decides where it goes."""
    return logging.getLogger(name if name.startswith(ROOT) else f"{ROOT}.{name}")
