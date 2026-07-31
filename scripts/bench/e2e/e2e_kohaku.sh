#!/usr/bin/env bash
# Four-card token throughput for the nine Kohaku rungs, at 262144 tokens/step.
#
# bf16 parameters and Muon, because that is the configuration the runs will actually
# use: bf16 measured ~3% ahead of fp32-under-autocast on the dense stages, and Muon
# holds one momentum buffer against AdamW's two fp32 moments, which is what decides
# whether the larger sparse rungs fit at all. No AdamW arm: it is not a candidate
# optimizer here, so a row measuring it would only be a number nobody acts on.
#
# fp8 appears nowhere here on purpose: this is the baseline arm. The MXFP8 arm is
# `e2e_kohaku_mxfp8.sh`, which writes into the same directory and waits for this script
# to exit rather than racing it -- two sweeps on the same four cards would make both
# sets of numbers worthless.
#
# Gradient checkpointing: DDP needs it everywhere at this budget since each rank holds
# the whole model. Under pipelining only the top two sparse rungs need it -- the stage
# split already cuts activation memory fourfold.

set -u
cd /xg7/KohakUwULLM
OUT=${OUT:-out/bench/train/kohaku_e2e}
LOG=${LOG:-$OUT/e2e_kohaku.log}
mkdir -p "$OUT"
: >"$LOG"

echo "cards: $(nvidia-smi --query-gpu=index,name --format=csv,noheader | tr '\n' ';')" | tee -a "$LOG"
echo "start: $(date -Is)" | tee -a "$LOG"

run() {
  local preset=$1 mode=$2 extra=$3 tag=$4 opt=${5:-muon}
  echo "=== $preset $tag ===" | tee -a "$LOG"
  # Fresh torchrun per row: a previous strategy's allocator state makes the next row's
  # peak-memory number meaningless, and that has produced a wrong answer here before.
  timeout 2700 .venv/bin/torchrun --standalone --nproc_per_node=4 \
    scripts/bench/e2e/step_throughput.py --preset "$preset" --mode "$mode" \
    --budget 262144 --micro-tokens 4096 8192 --param-dtype bf16 \
    --optimizer "$opt" --steps 32 --measure-last 16 \
    --out "$OUT" --tag "$tag" $extra 2>&1 \
    | grep -aE "pp4|ddp4|budget|tokens_per_s|OOM|out of memory|Error|Traceback" \
    | tee -a "$LOG"
  sleep 5
}

for preset in Kohaku-200M Kohaku-500M Kohaku-1B Kohaku-1.5B; do
  run "$preset" pipeline "" pp4_bf16_muon
  run "$preset" ddp "--grad-ckpt" ddp4_bf16_muon
done

for preset in Kohaku-MoE-1B Kohaku-MoE-2B Kohaku-MoE-3B; do
  run "$preset" pipeline "" pp4_bf16_muon
  run "$preset" ddp "--grad-ckpt" ddp4_bf16_muon
done

for preset in Kohaku-MoE-5B Kohaku-MoE-8B; do
  run "$preset" pipeline "--grad-ckpt" pp4_bf16_muon_ckpt
  run "$preset" ddp "--grad-ckpt" ddp4_bf16_muon
done

# Drift control: the first rung again, an hour and change later, on cards that have
# been at 100% the whole time. Without it a rung-to-rung difference cannot be told
# apart from the clocks walking down, and this box has already turned a 1.055x into a
# 1.074x that way. Compare against the pp4_bf16_muon row of the same preset.
run Kohaku-200M pipeline "" pp4_bf16_muon_drift

echo "done: $(date -Is)" | tee -a "$LOG"
echo "KOHAKU E2E DONE" | tee -a "$LOG"
