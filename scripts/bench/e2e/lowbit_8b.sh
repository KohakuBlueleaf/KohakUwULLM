#!/usr/bin/env bash
# Does MoE-8B-A1B fit, and under which precision/optimizer combination?
#
# fp32+AdamW is the baseline that OOMs. The other three separate the two levers:
# bf16 halves params and grads, Muon halves optimizer state (one momentum buffer
# instead of two moments).

set -u
cd /xg7/KohakUwULLM
LOG=${LOG:-out/bench/train/stage_memory/lowbit_8b.log}
GPU=${GPU:-0}
PRESET=${PRESET:-MoE-8B-A1B}
mkdir -p out/bench/train/stage_memory
: >"$LOG"

for combo in "bf16 muon" "bf16 adamw"; do
  set -- $combo
  echo "=== $PRESET params=$1 optimizer=$2 ===" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 .venv/bin/python \
    scripts/bench/e2e/stage_memory.py --preset "$PRESET" \
    --param-dtype "$1" --optimizer "$2" --micro-tokens 4096 2>&1 \
    | grep -avE "^\s*$" | tee -a "$LOG"
done

echo "LOWBIT 8B DONE" | tee -a "$LOG"
