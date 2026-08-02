#!/usr/bin/env bash
# Per-module benchmarks, then the e2e matrix on the best head kernel.

set -u
cd /xg7/KohakUwULLM
LOG=${LOG:-/tmp/all_bench.log}
: >"$LOG"

echo "### MODULE BENCHMARKS ###" >>"$LOG"
for b in kernels attention head_options global_ratio bandwidth hgemm_acc chunked_ce; do
  [ -f "scripts/bench/$b.py" ] || continue
  echo "=== module: $b ===" >>"$LOG"
  CUDA_VISIBLE_DEVICES=0 timeout 3600 .venv/bin/python "scripts/bench/$b.py" \
    --out "out/bench/$b" >>"$LOG" 2>&1
done
echo "### MODULES DONE ###" >>"$LOG"

echo "### E2E MATRIX ###" >>"$LOG"
run() {
  echo "=== $1 $5 ($3) ===" >>"$LOG"
  timeout 2400 .venv/bin/python scripts/bench/e2e/step_throughput.py --gpus 4 \
    --preset "$1" --mode "$2" --budget 262144 \
    --micro-tokens $3 --warmup 2 --iters 3 --out out/bench/train/matrix --tag "$5" $4 \
    >>"$LOG" 2>&1
  sleep 5
}
for preset in MoE-1B-A120M MoE-1B-A280M MoE-2B-A370M MoE-3B-A500M \
              MoE-3B-A500M-wide MoE-3B-A500M-deep MoE-8B-A1B \
              Nano-1B Nano-1B-wide Nano-1B-deep Nano-500M Nano-200M; do
  run "$preset" pipeline "4096 8192" "" "pp4"
  run "$preset" ddp "4096 8192" "--grad-ckpt" "ddp4_ckpt"
done
echo "### ALL DONE ###" >>"$LOG"
