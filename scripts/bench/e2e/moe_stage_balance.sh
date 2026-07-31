#!/usr/bin/env bash
# Per-stage runtime and memory for the MoE presets, on one GPU.
#
# Runs before the 4-GPU e2e sweep: the split it measures is the one that sweep
# will use, and at 8B the binding stage decides whether the preset fits at all.

set -u
cd /xg7/KohakUwULLM
LOG=${LOG:-/tmp/moe_balance.log}
GPU=${GPU:-0}
: >"$LOG"

for preset in MoE-1B-A120M MoE-1B-A280M MoE-2B-A370M MoE-3B-A500M \
              MoE-3B-A500M-wide MoE-3B-A500M-deep MoE-8B-A1B; do
  for micro in 4096 8192; do
    echo "=== $preset micro=$micro ===" >>"$LOG"
    CUDA_VISIBLE_DEVICES=$GPU timeout 1800 .venv/bin/python \
      scripts/bench/e2e/stage_balance.py --preset "$preset" --micro-tokens "$micro" \
      --out "out/bench/train/stage_balance" >>"$LOG" 2>&1
  done
done
echo "MOE BALANCE DONE" >>"$LOG"
