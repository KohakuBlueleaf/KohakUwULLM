#!/usr/bin/env bash
# Microbatch sweep by count, so sizes are not restricted to powers of two.
#
# Varlen packing fills any size and the memory ceiling does not land on a power
# of two, so the useful range between 8192 and OOM was never explored.

set -u
cd /xg7/KohakUwULLM
LOG=${LOG:-out/bench/train/matrix/e2e_micro.log}
mkdir -p out/bench/train/matrix
: >"$LOG"

run() {
  echo "=== $1 counts=$2 optim=$3 ckpt=$4 ===" | tee -a "$LOG"
  timeout 3000 .venv/bin/python scripts/bench/e2e/step_throughput.py --gpus 4 \
    --preset "$1" --mode pipeline --budget 262144 \
    --micro-counts $2 --param-dtype bf16 --optimizer "$3" $4 \
    --warmup 2 --iters 3 --out out/bench/train/matrix --tag "pp4_micro_$3$5" 2>&1 \
    | grep -aE "pp4 +[0-9]|budget|OOM$|Error" | tee -a "$LOG"
  sleep 4
}

run MoE-8B-A1B   "32 26 24 22 20 18 16" muon --grad-ckpt _ckpt
run MoE-3B-A500M "32 26 22 18 16 12"    muon --grad-ckpt _ckpt
run Nano-1B      "32 24 18 16 12 8"     muon ""            ""
echo "E2E MICRO DONE" | tee -a "$LOG"
