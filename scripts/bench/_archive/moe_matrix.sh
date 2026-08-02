#!/usr/bin/env bash
# Per-module head figure, then the MoE half of the preset matrix.
#
# Dense scales are already measured; MoE and 8B-A1B are what remain. The head
# benchmark runs first because it needs one GPU and takes minutes, and its
# result feeds the stage-cost calibration the MoE runs depend on.

set -u
cd /xg7/KohakUwULLM

OUT=${OUT:-out/bench/train/matrix}
BUDGET=${BUDGET:-262144}
LOG=${LOG:-/tmp/moe_matrix.log}
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 timeout 1800 .venv/bin/python scripts/bench/kernel/head_options.py \
  --out out/bench/kernel/head_options >>"$LOG" 2>&1
echo "HEAD FIGURE DONE" >>"$LOG"

run() {
  local preset=$1 mode=$2 micro=$3 extra=$4 tag=$5
  echo "=== $preset $tag ($micro) ===" >>"$LOG"
  timeout 2400 .venv/bin/python scripts/bench/e2e/step_throughput.py --gpus 4 \
    --preset "$preset" --mode "$mode" --budget "$BUDGET" \
    --micro-tokens $micro --warmup 2 --iters 3 \
    --out "$OUT" --tag "$tag" $extra >>"$LOG" 2>&1
  sleep 5
}

# MoE first: these decide the architecture questions. 8B-A1B is last because it
# is the most likely to OOM and the least useful to have blocking the rest.
MOE=${MOE:-"MoE-1B-A120M MoE-1B-A280M MoE-2B-A370M MoE-3B-A500M MoE-3B-A500M-wide MoE-3B-A500M-deep MoE-8B-A1B"}
DENSE=${DENSE:-"Nano-1B Nano-1B-wide Nano-1B-deep"}

for preset in $MOE $DENSE; do
  run "$preset" pipeline "4096 8192" "" "pp4"
  run "$preset" pipeline "4096 8192" "--grad-ckpt" "pp4_ckpt"
  run "$preset" ddp "4096 8192" "--grad-ckpt" "ddp4_ckpt"
done

echo "MOE MATRIX DONE" >>"$LOG"
