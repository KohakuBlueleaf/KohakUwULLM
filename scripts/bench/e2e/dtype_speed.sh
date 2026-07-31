#!/usr/bin/env bash
# Does native 16-bit make the trunk faster, not just smaller?
#
# fp32 params under autocast pay a cast on every weight read and hold 2x the
# bytes; native bf16/fp16 pay neither. Dense first: no routing to confound it.

set -u
cd /xg7/KohakUwULLM
LOG=${LOG:-out/bench/train/stage_balance/dtype_speed.log}
GPU=${GPU:-0}
mkdir -p out/bench/train/stage_balance
: >"$LOG"

for preset in Nano-200M Nano-500M Nano-1B; do
  for dt in fp32 bf16 fp16; do
    echo "=== $preset $dt ===" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU timeout 1800 .venv/bin/python \
      scripts/bench/e2e/stage_balance.py --preset "$preset" --param-dtype "$dt" \
      --micro-tokens 8192 --out out/bench/train/stage_balance 2>&1 \
      | grep -avE "^\s*$" | tee -a "$LOG"
  done
done
echo "DTYPE SPEED DONE" | tee -a "$LOG"
