#!/usr/bin/env bash
# End-to-end throughput across dtype, optimizer, microbatch size and strategy.
#
# Phase 1 finds the best microbatch size per preset -- it is the variable with
# the largest effect and every later comparison depends on it. Phase 2 sweeps
# dtype and optimizer at that size rather than re-sweeping everything, which
# would be a 4x longer run for information the microbatch sweep already has.

set -u
cd /xg7/KohakUwULLM
LOG=${LOG:-out/bench/train/matrix/e2e_full.log}
BUDGET=${BUDGET:-262144}
mkdir -p out/bench/train/matrix
: >"$LOG"

run() {
  local preset=$1 mode=$2 micro=$3 dtype=$4 optim=$5 extra=$6 tag=$7
  echo "=== $preset $tag micro=$micro ===" | tee -a "$LOG"
  timeout 2400 .venv/bin/python scripts/bench/e2e/step_throughput.py --gpus 4 \
    --preset "$preset" --mode "$mode" \
    --budget "$BUDGET" --micro-tokens $micro --param-dtype "$dtype" \
    --optimizer "$optim" --warmup 2 --iters 3 \
    --out out/bench/train/matrix --tag "$tag" $extra 2>&1 \
    | grep -aE "pp4|ddp4|budget|OOM|Error" | tee -a "$LOG"
  sleep 4
}

# Phase 1: microbatch size. bf16 makes 16384/32768 reachable at all.
for preset in Nano-1B MoE-3B-A500M MoE-8B-A1B; do
  run "$preset" pipeline "4096 8192 16384 32768" bf16 adamw "" "pp4_micro"
done

# Phase 2: dtype x optimizer, both strategies.
for preset in Nano-1B MoE-3B-A500M MoE-8B-A1B; do
  for dtype in fp32 bf16 fp16; do
    for optim in adamw muon; do
      run "$preset" pipeline "8192" "$dtype" "$optim" "" "pp4_${dtype}_${optim}"
      run "$preset" ddp "8192" "$dtype" "$optim" "--grad-ckpt" "ddp4_${dtype}_${optim}"
    done
  done
done

echo "E2E FULL DONE" | tee -a "$LOG"
