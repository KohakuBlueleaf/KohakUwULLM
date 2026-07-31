#!/usr/bin/env bash
# Global-attention ratio across scales and contexts, dense and MoE.
#
# The single-scale sweep answered the question only for Nano-500M at 2048
# context. Attention is a larger share of a small model's cost and of a long
# context, so the ratio that is free at one point on that grid need not be free
# at another; this runs the same sweep across the matrix, then draws the one
# cross-scale figure.
#
#   GPU=3 bash scripts/bench/e2e/global_ratio_matrix.sh
#
# Token budgets differ per case because a 1B model at 16k tokens does not fit in
# 32 GB. Only within-case comparisons are made, so a per-case budget costs
# nothing; a budget large enough for every case would starve the small ones of
# occupancy and make their curves noise.

set -u
cd /xg7/KohakUwULLM

OUT=${OUT:-out/bench/train/global_ratio}
LOG=${LOG:-out/bench/train/global_ratio.log}
GPU=${GPU:-3}
mkdir -p "$OUT"
: >"$LOG"

# preset:seq_len:window:tokens
CASES=${CASES:-"
Nano-200M:2048:1024:16384
Nano-200M:4096:1024:16384
Nano-500M:2048:1024:16384
Nano-500M:4096:1024:16384
Nano-1B:2048:1024:8192
Nano-1B:4096:1024:8192
MoE-1B-A120M:2048:1024:16384
MoE-1B-A120M:4096:1024:16384
"}

for case in $CASES; do
  IFS=: read -r preset seq window tokens <<<"$case"
  echo "=== $preset seq=$seq window=$window tokens=$tokens ===" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 .venv/bin/python scripts/bench/e2e/global_ratio.py \
    --preset "$preset" --seq-len "$seq" --window "$window" --tokens "$tokens" \
    --out "$OUT/${preset}_s${seq}" 2>&1 | tee -a "$LOG"
done

CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python scripts/bench/e2e/global_ratio.py \
  --summarize "$OUT" 2>&1 | tee -a "$LOG"

echo "GLOBAL RATIO MATRIX DONE" | tee -a "$LOG"
