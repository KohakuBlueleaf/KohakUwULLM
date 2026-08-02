# scripts/bench/_archive

Benchmarks that answered their question and were superseded. Kept rather than
deleted because the *method* in several of them is still the reference for how a
sweep here is constructed, and because a result nobody can re-derive is worth less
than a script nobody runs.

None of these is wired into `run_all.sh`. All of them predate the Kohaku ladder and
write into `out/bench/train/matrix/`, a directory that no longer exists — so running one
as-is produces a result that cannot be compared against anything current.

| file | what it measured | superseded by |
|---|---|---|
| `e2e.py` | four parallelism strategies (single / ddp / pipeline / pp+ckpt) in one rank group | `e2e/step_throughput.py`, which fixes the *step* rather than the microbatch |
| `e2e_driver.py` | one isolated process group per (preset, strategy), then aggregate | `e2e/e2e_kohaku.sh` — same isolation-per-row rule, Kohaku presets |
| `e2e_real.sh` | dense + MoE, pipeline + DDP, at 262144 tok/step | `e2e/e2e_kohaku.sh` |
| `e2e_full.sh` | microbatch size first, then dtype and optimizer at the winner | `e2e/e2e_kohaku.sh` (bf16 + Muon settled) and `e2e/dtype_speed.sh` |
| `e2e_micro.sh` | microbatch sweep by *count*, off the powers of two | folded into `e2e/step_throughput.py --micro-counts` |
| `e2e_8b.sh` | does MoE-8B-A1B fit under Muon + checkpointing | `e2e/lowbit_8b.sh` (memory) and the `Kohaku-MoE-8B` rows of `e2e/e2e_kohaku.sh` |
| `moe_matrix.sh` | the MoE half of the preset matrix, pre-Kohaku presets | `e2e/e2e_kohaku.sh`, whose ladder is the sequence the design targets were solved in |
| `run_all_bench.sh` | every module benchmark, then the e2e matrix | `run_all.sh` — which absorbed its four extra module stages |

`run_all_bench.sh` is archived for a second reason worth recording: it selected
benchmarks with `[ -f "scripts/bench/$b.py" ] || continue`, so once the suite moved
into subdirectories it skipped every module stage and exited green. A guard that
fails open is worse than no guard, and a driver that enumerates scripts by path is
how you build one.
