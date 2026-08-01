"""Figures for the attention, MLP/MoE and block benchmark suites."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    with open(p) as fh:
        return json.load(fh)


def main() -> None:
    out = "out/bench"
    os.makedirs(f"{out}/plots", exist_ok=True)

    a = load(f"{out}/attn/attn_suite.json")
    bf = [r for r in a if r["dtype"] == "bf16"]
    lbl = [f"{r['T']//1024}k\nh{r['head']}" for r in bf]
    x = range(len(bf))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for key, off, name in (
        ("sdpa", -0.27, "torch SDPA"),
        ("varlen", 0.0, "our varlen bf16"),
        ("mxfp8", 0.27, "our MXFP8"),
    ):
        ax[0].bar([i + off for i in x], [r.get(key) or 0 for r in bf], 0.26, label=name)
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(lbl)
    ax[0].set_ylabel("ms")
    ax[0].set_title("attention forward, causal")
    ax[0].legend(fontsize=8)

    ax[1].bar([i - 0.2 for i in x], [r["bf16_rms"] for r in bf], 0.4, label="bf16")
    ax[1].bar([i + 0.2 for i in x], [r["mxfp8_rms"] for r in bf], 0.4, label="MXFP8")
    ax[1].set_yscale("log")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels(lbl)
    ax[1].set_ylabel("relative RMS vs fp64")
    ax[1].set_title("forward error")
    ax[1].legend(fontsize=8)

    ws = [r for r in bf if r.get("varlen_w1024")]
    wl = [f"{r['T']//1024}k\nh{r['head']}" for r in ws]
    xw = range(len(ws))
    for key, off, name in (
        ("varlen", -0.27, "full"),
        ("varlen_w4096", 0.0, "window 4096"),
        ("varlen_w1024", 0.27, "window 1024"),
    ):
        ax[2].bar(
            [i + off for i in xw], [r.get(key) or 0 for r in ws], 0.26, label=name
        )
    ax[2].set_xticks(list(xw))
    ax[2].set_xticklabels(wl)
    ax[2].set_ylabel("ms")
    ax[2].set_title("our varlen bf16, sliding window")
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/attention.png", dpi=130)
    plt.close()

    m = load(f"{out}/mlp_moe/mlp_moe.json")
    mb = [r for r in m if r["dtype"] == "bf16"]
    keys = [
        "torch_16bit",
        "torch_16bit_compiled",
        "ours_16bit",
        "mxfp8_unfused",
        "mxfp8_fused",
        "mxfp8_vendor",
        "mxfp8_vendor_compiled",
    ]
    lbl2 = [f"{r['T']//1024}k\n{r['D']}x{r['FFN']}" for r in mb]
    x2 = range(len(mb))
    plt.figure(figsize=(10, 4.5))
    w = 0.12
    for j, k in enumerate(keys):
        plt.bar(
            [i + (j - 3) * w for i in x2],
            [r.get(k) or 0 for r in mb],
            w,
            label=k.replace("_", " "),
        )
    plt.xticks(list(x2), lbl2)
    plt.ylabel("ms")
    plt.title("MLP: 16-bit vs MXFP8 paths")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/mlp.png", dpi=130)
    plt.close()

    b = load(f"{out}/block/block.json")
    lbl3 = [f"T{r['T']//1024}k D{r['D']}" for r in b]
    x3 = range(len(b))
    plt.figure(figsize=(9, 4.5))
    for j, k in enumerate(["torch", "torch_compiled", "ours"]):
        plt.bar(
            [i + (j - 1) * 0.27 for i in x3],
            [r.get(k) or 0 for r in b],
            0.26,
            label=k.replace("_", " "),
        )
    plt.xticks(list(x3), lbl3)
    plt.ylabel("ms")
    plt.title("transformer block, forward")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}/plots/block.png", dpi=130)
    plt.close()
    print("wrote", f"{out}/plots/")


if __name__ == "__main__":
    main()
