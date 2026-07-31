# KohakUwULLM

An extensible **decoder-only LLM training framework**. Not a "trainer" -- the
framework provides slots; you select or drop in what you need.

Reproducing a Llama-shaped dense model, a Gemma-shaped sliding-window model, or a
DeepSeekMoE sparse model is a **config**. Adding a new norm / MLP / attention /
position encoding / router / data source is a registry entry or a dotted import
path -- no core edits.

Sibling to [KohakuLatentMaid](https://github.com/KohakuBlueleaf/KohakuLatentMaid)
(diffusion) and [KohakuTerrarium](https://github.com/KohakuBlueleaf/KohakuTerrarium)
(agents): same house conventions, same stack.

## What's in it

- **Own architecture implementation.** `transformers` is used for the tokenizer
  and nothing else. GQA, QK-norm, RoPE with linear/NTK/YaRN scaling, sliding
  windows with global-layer interleave and a per-layer choice of attention
  kernel, attention sinks, DeepSeekMoE with aux-loss-free load balancing
  (Sinkhorn, expert-choice and ReLU routers registered beside it), z-loss.
  Embeddings are untied by default.
- **Packed variable-length training.** No padding is ever computed on, and
  attention cannot cross a document boundary. Worth up to 7x on this corpus.
- **Native MXFP8 training**, verified against bf16 rather than assumed: e4m3
  values with UE8M0 per-32 block scales, on the attention projections, the
  feed-forward pair and the routed experts. Measured 1.08-1.22x end to end on
  dense and **1.53-2.31x on MoE**, at a loss gap inside run-to-run noise.
- **Triton kernels where they win**, benchmarked and precision-tested: fused
  RMSNorm, fused SwiGLU, a grouped GEMM for MoE experts, fully fused MXFP8
  expert kernels, and a chunked z-loss that never materializes logits.
- **Lightning trainer** with token-exact gradient accumulation, DDP, and
  cost-balanced pipeline-parallel splitting for models that do not fit one card.
- **Benchmarks as a deliverable.** `scripts/bench/` produces figures that always
  show accuracy next to speed.

## Quickstart

```bash
uv pip install -e ".[dev,bench]"

# tokenizer: DeepSeek-V4 pruned to 64k + 1536 special slots = 65536
.venv/bin/python scripts/tokenizer/build_tokenizer.py --out models/tokenizer

# smoke run: 25M model, 1 GPU, 50 steps
kogine run scripts/train/lm.py --config configs/lm/debug.py

# production
kogine run scripts/train/lm.py --config configs/lm/tipo_500m.py    # dense Kohaku-500M
kogine run scripts/train/lm.py --config configs/lm/tipo_moe_1b.py  # Kohaku-MoE-1B
```

## The preset ladder

Nine rungs on a fixed 65536 vocabulary, untied embeddings, and sparsity pinned at
`top_k/experts = 0.125` so hyper-parameters transfer between rungs. Counts are measured
by building each rung on the meta device, not solved -- a closed form omits the norms,
the router matrix and the second embedding matrix.

| dense | total | | sparse | total / active |
|---|---|---|---|---|
| Kohaku-200M | 204M | | Kohaku-MoE-1B | 991M / 248M |
| Kohaku-500M | 546M | | Kohaku-MoE-2B | 1953M / 411M |
| Kohaku-1B | 982M | | Kohaku-MoE-3B | 2907M / 617M |
| Kohaku-1.5B | 1514M | | Kohaku-MoE-5B | 4934M / 943M |
| | | | Kohaku-MoE-8B | 7713M / 1371M |

The rungs interleave: their effective capacities `sqrt(active x total)` form one smooth
sequence, 204 / 381 / 546 / 756 / 982 / 1182 / 1514 / 1953 / 2902 M. Note that the
`active` in *that* expression excludes the embedding, the head and the router -- the
quantity the ladder's targets were solved in. The `active` in the table is the
repo-wide one, which includes all three and is 20-30% higher on every sparse rung.
`bench/ladder.py` reports both, under different names, for exactly that reason.

## Documentation

| Doc | What it covers |
|---|---|
| [docs/concepts/architecture.md](docs/concepts/architecture.md) | Backbone, components, presets, MoE, packing |
| [docs/guides/writing-configs.md](docs/guides/writing-configs.md) | The full knob catalogue |
| [docs/guides/extending.md](docs/guides/extending.md) | Adding a component / router / renderer / kernel |
| [docs/performance/performance.md](docs/performance/performance.md) | Measured results on RTX 5090 |
| [docs/internals/pipeline.md](docs/internals/pipeline.md) | Pipeline parallelism |
| [docs/internals/data.md](docs/internals/data.md) | Corpus, prompt rendering, packing, tokenizer |
| [docs/internals/mxfp8.md](docs/internals/mxfp8.md) | MXFP8: what is swapped, what is verified, what is not |
| [CLAUDE.md](CLAUDE.md) | Coding conventions |

## Selected measurements (4x RTX 5090, sm_120)

Every figure below is measured on this box, both arms of every comparison on the same
card, with the losing configuration reported as readily as the winning one.

- **MXFP8 is worth more the larger the model**, at micro 8192 under 4-card
  pipelining: dense 1.081x / 1.139x / 1.221x at 200M / 500M / 1B, and sparse
  1.533x / 1.661x / 1.895x / **2.308x** at MoE-1B / 2B / 5B / 8B. The loss gap over
  400 steps is 1.2x the bf16-against-bf16 noise floor -- inside run-to-run scatter.
- **Where that came from, separated rather than claimed.** Measured as the ladder
  actually built: eager experts -> grouped bf16 is **5.7x**, grouped bf16 -> grouped
  fp8 is **1.49x**, grouped fp8 -> fully fused is **1.65x**. The shipped bf16
  baseline is already grouped, so end-to-end ratios measure only the last two.
- **On MoE, fp8 also saves 40-44% of peak memory** -- the fused epilogues never
  materialize the `(tokens*top_k, hidden)` intermediates. On dense it costs ~2 bytes
  per swapped parameter instead. The two families are opposite and neither number
  transfers to the other.
- **FlashAttention-4 does not run on consumer Blackwell.** It needs TMEM, an
  sm_100 feature. FA2-class is the ceiling; PyTorch 2.13's
  `varlen_attn` is exactly that, natively, with a trainable backward.
- **fp16 accumulation is 1.4-1.55x faster** than fp32 accumulation on GeForce
  tensor cores (325 vs 232 TFLOP/s). Error grows with reduction depth; split-K
  buys it back. Used only in the MoE grouped GEMM, never in a backward.
- **The LM head is a memory problem.** At vocab 65536 / 16k tokens: naive 16.5 GiB,
  chunked 0.83 GiB. And `F.linear_cross_entropy(options=None)` is the
  *materializing* path -- the chunked one needs an explicit options object.
- **Packing is worth 7x** at the length spread this corpus actually has.
- **Pipelining beats DDP on this box** and the reason is the fabric: no NVLink, every
  pair over the PCIe host bridge, so a ring all-reduce of the whole gradient is the
  access pattern it handles worst. Pipeline boundary sends are 12-29 MB point-to-point
  and overlap with compute; DDP rows show 4-16% step spread where pipeline rows sit
  at 0.1-0.6%.

Full details and the figures: [docs/performance/performance.md](docs/performance/performance.md).

## Requirements

Python 3.10+, **torch >= 2.13** (for `F.linear_cross_entropy` and
`torch.nn.attention.varlen.varlen_attn`), triton >= 3.7, an NVIDIA GPU.

## License

Apache-2.0. See [LICENSE.md](LICENSE.md).
