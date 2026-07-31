#!/usr/bin/env bash
# MoE-8B-A1B: full 16-bit, Muon, gradient checkpointing, microbatch sweep.
#
# The earlier attempt OOMed because it defaulted to AdamW (23.35 GiB/stage) and
# ran without checkpointing. Muon drops the second moment (18.95) and 1F1B holds
# several microbatches in flight on stage 0, which is what checkpointing bounds.

set -u
cd /xg7/KohakUwULLM
LOG=${LOG:-out/bench/train/matrix/e2e_8b.log}
mkdir -p out/bench/train/matrix
: >"$LOG"

run() {
  echo "=== $1 dtype=$2 optim=$3 micro=$4 ckpt=$5 ===" | tee -a "$LOG"
  timeout 2400 .venv/bin/torchrun --standalone --nproc_per_node=4 \
    scripts/bench/e2e/step_throughput.py --preset "$1" --mode pipeline --budget 262144 \
    --micro-tokens $4 --param-dtype "$2" --optimizer "$3" $5 \
    --warmup 2 --iters 3 --out out/bench/train/matrix --tag "pp4_$2_$3$6" 2>&1 \
    | grep -aE "pp4 +[0-9]|budget|OOM$|Error" | tee -a "$LOG"
  sleep 4
}

run MoE-8B-A1B bf16 muon  "2048 4096 8192" --grad-ckpt _ckpt
run MoE-8B-A1B bf16 muon  "2048 4096"      ""          ""
run MoE-8B-A1B bf16 adamw "2048 4096"      --grad-ckpt _ckpt
run MoE-3B-A500M bf16 muon "4096 8192 16384" ""        ""
echo "E2E 8B DONE" | tee -a "$LOG"
