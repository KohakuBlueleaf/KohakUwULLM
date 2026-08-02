"""Hardware description a GEMM plan is computed from.

Fields split into three groups: queried from the driver, measured with a
microbenchmark, and calibrated from a kernel fit. Only the queried group has a
safe default; the other two must come from the card you are on.

See docs/internals/kernel-dev.md and docs/performance/gemm.md.
"""

import dataclasses
import json

import torch


@dataclasses.dataclass(frozen=True)
class Device:
    """Everything `plan()` needs to know about a card.

    ``mma_peak`` is TFLOP/s for the accumulate dtype a trainer uses (fp32), from
    back-to-back ``mma.sync`` with register-resident operands. ``dram_bw`` and
    ``l2_bw`` are GB/s. ``bar_tax`` maps CTAs-per-SM to the fraction of the MMA
    issue rate a barrier-synchronised main loop sustains.
    """

    name: str
    sms: int
    regs_per_sm: int
    smem_per_cta: int
    smem_per_sm: int
    max_threads_per_sm: int
    l2_bytes: int

    mma_peak: float
    dram_bw: float
    l2_bw: float

    bar_tax: tuple[float, ...] = (0.935, 0.958)
    reg_overhead: int = 64
    ldsm_ref: float = 0.375
    ldsm_slope: float = 0.90
    sk_balance: float = 14.0
    uneven_k: float = 0.62

    @classmethod
    def query(cls, index: int = 0, **overrides) -> "Device":
        """Driver-reported fields plus caller-supplied measured ones."""
        p = torch.cuda.get_device_properties(index)
        base = dict(
            name=p.name,
            sms=p.multi_processor_count,
            regs_per_sm=p.regs_per_multiprocessor,
            smem_per_cta=p.shared_memory_per_block_optin,
            smem_per_sm=p.shared_memory_per_multiprocessor,
            max_threads_per_sm=p.max_threads_per_multi_processor,
            l2_bytes=p.l2_cache_size,
            mma_peak=float("nan"),
            dram_bw=float("nan"),
            l2_bw=float("nan"),
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_json(cls, path: str) -> "Device":
        """Load a card description written by `to_json`."""
        with open(path) as fh:
            raw = json.load(fh)
        raw["bar_tax"] = tuple(raw["bar_tax"])
        return cls(**raw)

    def to_json(self, path: str) -> None:
        """Write this description so a plan can be reproduced off the card."""
        with open(path, "w") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

    def validate(self) -> None:
        """Raise if a measured field was never filled in."""
        missing = [
            f
            for f in ("mma_peak", "dram_bw", "l2_bw")
            if not (getattr(self, f) == getattr(self, f))
        ]
        if missing:
            raise ValueError(
                f"{self.name}: measure {', '.join(missing)} on this card and pass "
                "them to Device.query(...); see docs/internals/kernel-dev.md section 1"
            )


RTX_5090 = Device(
    name="NVIDIA GeForce RTX 5090",
    sms=170,
    regs_per_sm=65536,
    smem_per_cta=101376,
    smem_per_sm=101376,
    max_threads_per_sm=1536,
    l2_bytes=100663296,
    mma_peak=276.8,
    dram_bw=1830.0,
    l2_bw=13300.0,
)
