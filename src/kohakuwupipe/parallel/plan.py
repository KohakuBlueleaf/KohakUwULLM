"""Cost-balanced contiguous split of a layer stack across pipeline stages.

Model-agnostic: costs and parameter counts arrive as plain numbers, so nothing
here knows what a layer is. See docs/kohakuwupipe/plan.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StagePlan:
    """Which layers belong to one stage, and where it sits in the pipeline.

    Architecture-free: a subclass names what its own ends carry.
    See docs/kohakuwupipe/plan.md.
    """

    index: int
    num_stages: int
    start_layer: int
    end_layer: int  # exclusive
    cost: float

    @property
    def num_layers(self) -> int:
        return self.end_layer - self.start_layer

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.index == self.num_stages - 1


def _prefix(values, depth: int) -> list[float]:
    """Running sum of a per-layer vector, or of a scalar broadcast over ``depth``."""
    if isinstance(values, (int, float)):
        values = [float(values)] * depth
    values = [float(v) for v in values]
    if len(values) != depth:
        raise ValueError(f"expected {depth} per-layer values, got {len(values)}")
    total, out = 0.0, [0.0]
    for value in values:
        total += value
        out.append(total)
    return out


def partition(
    depth: int,
    num_stages: int,
    layer_cost,
    head_cost: float,
    layer_params=0.0,
    head_params: float = 0.0,
    embed_params: float = 0.0,
    allow_empty_last: bool = False,
) -> list[int]:
    """Contiguous cut points minimizing the slowest stage; ties break on memory.

    ``layer_cost`` and ``layer_params`` are each one number for every layer or a
    per-layer sequence. ``head_cost`` and ``head_params`` are charged to the last
    stage, ``embed_params`` to the first; ``allow_empty_last`` lets the last
    stage hold no layers. See docs/kohakuwupipe/plan.md.
    """
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")
    if num_stages > depth:
        raise ValueError(f"cannot split {depth} layers across {num_stages} stages")

    cost = _prefix(layer_cost, depth)
    mem = _prefix(layer_params, depth)
    inf = (float("inf"), float("inf"))
    best = [[inf] * (depth + 1) for _ in range(num_stages + 1)]
    cut = [[0] * (depth + 1) for _ in range(num_stages + 1)]
    for i in range(depth + 1):
        last = cost[depth] - cost[i] + head_cost
        held = mem[depth] - mem[i] + head_params
        best[1][i] = (last, held * held)
        cut[1][i] = depth
    for s in range(2, num_stages + 1):
        for i in range(depth + 1):
            upper = depth + 1 if (s == 2 and allow_empty_last) else depth - s + 2
            for j in range(i + 1, upper):
                mine = cost[j] - cost[i]
                held = mem[j] - mem[i] + (embed_params if i == 0 else 0.0)
                tail_max, tail_sq = best[s - 1][j]
                key = (max(mine, tail_max), held * held + tail_sq)
                if key < best[s][i]:
                    best[s][i] = key
                    cut[s][i] = j
    bounds, at = [0], 0
    for s in range(num_stages, 1, -1):
        at = cut[s][at]
        bounds.append(at)
    bounds.append(depth)
    return bounds


def partition_v(
    depth: int,
    num_ranks: int,
    layer_cost,
    head_cost: float,
    embed_cost: float = 0.0,
    allow_empty: bool = True,
) -> list[int]:
    """Cut points for ``2 * num_ranks`` chunks placed in a V, balanced per rank.

    A V pairs chunk ``i`` with chunk ``2 * num_ranks - 1 - i`` on one rank, so the
    quantity to equalize is the pair's cost, not each chunk's. Rank 0 carries both
    the embedding and the head and therefore holds the fewest layers.
    See docs/kohakuwupipe/plan.md.
    """
    chunks = 2 * num_ranks
    if depth < chunks and not allow_empty:
        raise ValueError(f"cannot split {depth} layers across {chunks} chunks")
    cost = _prefix(layer_cost, depth)
    extra = [0.0] * num_ranks
    extra[0] = head_cost + embed_cost

    def counts_for(target: float) -> list[int] | None:
        """Layers per rank under a per-rank ceiling, or ``None`` if it cannot hold."""
        taken, at = [], 0
        for rank in range(num_ranks):
            budget = target - extra[rank]
            if budget < 0:
                return None
            floor = 0 if allow_empty else 2
            n = floor
            while at + n < depth and cost[at + n + 1] - cost[at] <= budget:
                n += 1
            taken.append(n)
            at += n
        if at < depth:
            return None
        taken[-1] += depth - at
        return taken

    low, high = 0.0, cost[depth] + head_cost + embed_cost
    for _ in range(64):
        mid = (low + high) / 2
        if counts_for(mid) is None:
            low = mid
        else:
            high = mid
    counts = counts_for(high)
    if counts is None:
        raise ValueError("no V split satisfies the per-rank ceiling")

    # Each rank's layers split between its front chunk and its back chunk; the
    # back one carries the head on rank 0, so the front takes the remainder.
    # Rank 0's back chunk carries the head, so its layers go to the front chunk.
    front, back = [], []
    for rank, n in enumerate(counts):
        half = n if rank == 0 else n // 2
        front.append(half)
        back.append(n - half)
    bounds, at = [0], 0
    for n in front:
        at += n
        bounds.append(at)
    for n in reversed(back):
        at += n
        bounds.append(at)
    return bounds


def plan_stages(
    depth: int,
    num_stages: int,
    layer_cost,
    head_cost: float,
    layer_params=0.0,
    head_params: float = 0.0,
    embed_params: float = 0.0,
    allow_empty_last: bool = False,
    plan_cls: type = StagePlan,
) -> list[StagePlan]:
    """One ``plan_cls`` per stage, in pipeline order."""
    bounds = partition(
        depth,
        num_stages,
        layer_cost,
        head_cost,
        layer_params=layer_params,
        head_params=head_params,
        embed_params=embed_params,
        allow_empty_last=allow_empty_last,
    )
    cost = _prefix(layer_cost, depth)
    plans = []
    for stage in range(num_stages):
        start, end = bounds[stage], bounds[stage + 1]
        has_head = stage == num_stages - 1
        plans.append(
            plan_cls(
                index=stage,
                num_stages=num_stages,
                start_layer=start,
                end_layer=end,
                cost=cost[end] - cost[start] + (head_cost if has_head else 0.0),
            )
        )
    return plans


def plan_from_layers(
    layers, layer_cost=1.0, head_cost: float = 0.0, plan_cls: type = StagePlan
) -> list[StagePlan]:
    """Stage plans from an explicit per-stage layer count, bypassing the cost model.

    ``layers`` has one entry per stage; the costs are carried for reporting only.
    See docs/kohakuwupipe/plan.md.
    """
    counts = [int(n) for n in layers]
    if not counts or any(n < 0 for n in counts):
        raise ValueError(f"expected non-negative layer counts, got {counts}")
    cost = _prefix(layer_cost, sum(counts))
    plans, start = [], 0
    for index, n in enumerate(counts):
        has_head = index == len(counts) - 1
        plans.append(
            plan_cls(
                index=index,
                num_stages=len(counts),
                start_layer=start,
                end_layer=start + n,
                cost=cost[start + n] - cost[start] + (head_cost if has_head else 0.0),
            )
        )
        start += n
    return plans


def describe(plans: list[StagePlan]) -> str:
    """A table of the split: layers, endpoints and cost share per stage."""
    total = sum(p.cost for p in plans) or 1.0
    depth = plans[-1].end_layer
    lines = [
        f"pipeline split: {len(plans)} stages over {depth} layers",
        f"  {'stage':>5}  {'layers':>10}  {'n':>3}  {'ends':>11}  cost share",
    ]
    for plan in plans:
        ends = "+".join(
            name
            for name, held in (("first", plan.is_first), ("last", plan.is_last))
            if held
        )
        span = f"{plan.start_layer}..{plan.end_layer - 1}"
        lines.append(
            f"  {plan.index:>5}  {span:>10}  {plan.num_layers:>3}  {ends or '-':>11}"
            f"  {100 * plan.cost / total:>9.1f}%"
        )
    return "\n".join(lines)
