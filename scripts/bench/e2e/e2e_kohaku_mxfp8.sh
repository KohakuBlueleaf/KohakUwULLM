#!/usr/bin/env bash
# The MXFP8 arm of the Kohaku 4-card sweep, against the bf16 rows in the same directory.
#
# Full swap, experts included. The earlier attention-only scope existed because the MoE
# fp8 path was broken; it is not any more. Two defects were found and fixed: a
# `@triton.autotune` keyed on the varlen token count that re-benchmarked every step and
# cost 365 ms/step, and an fp16 multiply in the WGRAD kernel that flushed 1.26e-7
# gradients to zero and was the entire 0.21-nat loss gap. Post-fix, MoE-1B-A280M
# measures 1.078x bf16 in a real 400-step run at 1.2x the noise floor.
#
# Dense rungs first: they finish in ~20 minutes and answer "does this help end to end"
# before the sparse rungs take an hour.
#
# Waits for the baseline sweep rather than racing it. Two runs on the same four cards
# would make both sets of numbers worthless, and the whole point of the arm is that its
# only difference from the bf16 rows is the dtype.

set -u
cd /xg7/KohakUwULLM
OUT=${OUT:-out/bench/train/kohaku_e2e}
LOG=${LOG:-$OUT/e2e_kohaku_mxfp8.log}
mkdir -p "$OUT"
: >"$LOG"

while pgrep -f "e2e_kohak[u]\.sh" >/dev/null; do
  echo "waiting for the bf16 sweep to finish: $(date -Is)" | tee -a "$LOG"
  sleep 60
done
echo "cards free, starting: $(date -Is)" | tee -a "$LOG"

run() {
  local preset=$1 mode=$2 extra=$3 tag=$4
  echo "=== $preset $tag ===" | tee -a "$LOG"
  timeout 2700 .venv/bin/torchrun --standalone --nproc_per_node=4 \
    scripts/bench/e2e/step_throughput.py --preset "$preset" --mode "$mode" \
    --budget 262144 --micro-tokens 4096 8192 --param-dtype bf16 \
    --optimizer muon --mxfp8 --steps 32 --measure-last 16 \
    --out "$OUT" --tag "$tag" $extra 2>&1 \
    | grep -aE "pp4|ddp4|budget|tokens_per_s|OOM|out of memory|Error|Traceback|ALLOCATED" \
    | tee -a "$LOG"
  sleep 5
}

for preset in Kohaku-200M Kohaku-500M Kohaku-1B Kohaku-1.5B; do
  run "$preset" pipeline "" pp4_bf16_mxfp8_muon
  run "$preset" ddp "--grad-ckpt" ddp4_bf16_mxfp8_muon
done

echo "DENSE MXFP8 DONE: $(date -Is)" | tee -a "$LOG"

for preset in Kohaku-MoE-1B Kohaku-MoE-2B Kohaku-MoE-3B; do
  run "$preset" pipeline "" pp4_bf16_mxfp8_muon
  run "$preset" ddp "--grad-ckpt" ddp4_bf16_mxfp8_muon
done

for preset in Kohaku-MoE-5B Kohaku-MoE-8B; do
  run "$preset" pipeline "--grad-ckpt" pp4_bf16_mxfp8_muon_ckpt
  run "$preset" ddp "--grad-ckpt" ddp4_bf16_mxfp8_muon
done

echo "KOHAKU MXFP8 DONE: $(date -Is)" | tee -a "$LOG"
