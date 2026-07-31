"""Shared plot styling for ``scripts/bench/*``. Headless (Agg), writes PNG + SVG.

Layouts show throughput and accuracy together, and categorical color is stable
across figures. See docs/performance/benchmarking.md.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Colorblind-safe categorical ramp (Okabe-Ito), most-distinct entries first.
_CATEGORICAL = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#7F7F7F",  # grey
]

_RC = {
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


class Palette:
    """Stable name -> color mapping, so a series keeps its color across figures."""

    def __init__(self, colors: list[str] | None = None) -> None:
        self._colors = colors or _CATEGORICAL
        self._assigned: dict[str, str] = {}

    def color(self, key: str) -> str:
        if key not in self._assigned:
            self._assigned[key] = self._colors[len(self._assigned) % len(self._colors)]
        return self._assigned[key]


def new_figure(nrows: int = 1, ncols: int = 1, figsize=None):
    """A styled figure + a flat list of axes (always a list, even for 1x1)."""
    plt.rcParams.update(_RC)
    figsize = figsize or (6.5 * ncols, 4.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    return fig, [ax for row in axes for ax in row]


def _si(value, _pos=None) -> str:
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= scale:
            trimmed = f"{value / scale:.1f}".rstrip("0").rstrip(".")
            return f"{trimmed}{unit}"
    if abs(value) < 1 and value != 0:
        return f"{value:g}"
    return f"{value:.0f}"


def finish_axis(
    ax,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    xticks=None,
    logx: bool = False,
    logy: bool = False,
    si_x: bool = True,
    legend: bool = False,
) -> None:
    """Apply labels / scales / ticks consistently."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    if xticks is not None:
        ax.set_xticks(list(xticks))
        if si_x:
            ax.xaxis.set_major_formatter(FuncFormatter(_si))
        ax.minorticks_off()
    if legend:
        ax.legend()


def bar_labels(ax, bars, fmt="{:.0f}") -> None:
    """Print each bar's value above it."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            fontsize=8,
        )


def save_figure(fig, path: str) -> None:
    """Write PNG next to an SVG of the same name, then close the figure."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(os.path.splitext(path)[0] + ".svg")
    plt.close(fig)
