"""Cheap training-health statistics computed from tensors the step already has.

See docs/internals/training-health.md for what each one detects and its alarm
threshold.
"""

import collections

import torch

from kohakuwullm.data.packing import IGNORE_INDEX


class SpikeScore:
    """Share of recent gradient norms that sit far from their rolling mean.

    Args:
        window: values retained.
        sigma: how many standard deviations counts as a spike.
    """

    def __init__(self, window: int = 1000, sigma: float = 7.0) -> None:
        self.values: collections.deque = collections.deque(maxlen=window)
        self.sigma = sigma

    def update(self, value: float) -> float:
        """Add one gradient norm; return the current spike score.

        Centre and spread are median and scaled MAD: a mean and standard
        deviation are dragged by the spikes being detected, so a run of them
        raises the threshold past itself and scores zero.
        """
        if value == value and value not in (float("inf"), float("-inf")):
            self.values.append(float(value))
        if len(self.values) < 32:
            return 0.0
        data = torch.tensor(self.values)
        centre = data.median()
        spread = (data - centre).abs().median() * 1.4826
        if spread <= 0:
            return 0.0
        return float(((data - centre).abs() > self.sigma * spread).float().mean())


def icl_score(
    labels: torch.Tensor,
    token_loss: torch.Tensor,
    cu_seqlens: torch.Tensor,
    early: int = 50,
    late: int = 500,
) -> float | None:
    """``loss`` at in-document position ``late`` minus at ``early``.

    Negative means later tokens are cheaper, which is context being used. Needs
    documents at least ``late`` tokens long; returns ``None`` when none are.
    """
    bounds = cu_seqlens.tolist()
    position = torch.empty_like(labels, dtype=torch.long)
    for start, stop in zip(bounds, bounds[1:]):
        position[start:stop] = torch.arange(stop - start, device=labels.device)

    real = labels != IGNORE_INDEX
    lo = real & (position >= early) & (position < early * 2)
    hi = real & (position >= late) & (position < late * 2)
    if int(lo.sum()) == 0 or int(hi.sum()) == 0:
        return None
    return float(token_loss[hi].mean() - token_loss[lo].mean())


def router_entropy(probabilities: torch.Tensor) -> float:
    """Mean per-token entropy of a router distribution, in nats."""
    p = probabilities.clamp_min(1e-9)
    return float((-(p * p.log()).sum(-1)).mean())


def router_similarity(weight: torch.Tensor) -> float:
    """How far a router's expert rows have collapsed onto their mean, in [0, 1].

    ``(n * ||mean of normalised rows||^2 - 1) / (n - 1)``; 1 means the router
    has stopped discriminating between experts.
    """
    n = weight.shape[0]
    if n < 2:
        return 0.0
    rows = weight.float()
    rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    mean = rows.mean(0)
    return float((n * mean.dot(mean) - 1.0) / (n - 1))
