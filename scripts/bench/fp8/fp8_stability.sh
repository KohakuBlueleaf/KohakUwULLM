#!/bin/bash
# LR stability sweep: bf16 vs MXFP8 round-up at multiples of the base rate.
#
# Attacks the main A/B's one real limitation. 651.8M tokens cannot reach the 300B
# where the published MX divergence appeared, and no longer run on this hardware
# will. But the *mechanism* behind that divergence -- quantization noise eating
# stability margin -- is measurable in 2000 steps: put the recipe at its edge and
# see whether the two dtypes break in the same place. A too-high lr fails in
# hundreds of steps, not hundreds of thousands.
#
# Three properties this script exists to guarantee, each learned from a failure:
#
#   * ONE CARD, SEQUENTIALLY. Every bf16/fp8 pair must be same-card to be a
#     speedup at all -- GPU 0 and GPU 1 differ by 2.7% in sustained clock against
#     a ~7% effect. Running two arms in parallel on two cards would be faster and
#     would reintroduce exactly that error. These are the only admissible speedup
#     pairs in the fp8 work, so the serialization is the point, not an oversight.
#   * A FINISHED RUN IS A .json. A killed arm leaves a partial CSV that looks
#     complete to anything reading the directory, and a half-trained run reduces
#     to "converged to a worse loss" -- once worth a spurious +0.77 loss
#     regression. So completion is tested by the summary, never by the CSV.
#   * NEVER REDO A FINISHED RUN. Reruns are free to invoke and expensive to wait
#     for; an interrupted sweep resumes where it stopped.
#
# Usage:  CARD=3 bash scripts/bench/fp8/fp8_stability.sh
set -u
cd "$(dirname "$0")/../.."

CARD=${CARD:?set CARD to the GPU index you own -- there is deliberately no default}
OUT=${OUT:-out/bench/train/fp8_stability}
LOG=${LOG:-$OUT/logs}
STEPS=${STEPS:-2000}
BASE_LR=3e-4
# 1x-8x maps the shipped regime; 16x-64x looks for the baseline's edge, because a
# sweep in which *neither* dtype breaks bounds fp8's margin cost without measuring
# it. Note what four octaves of this actually showed: degradation flattens while
# grad-clip fires on only 2.7-3.6% of steps throughout, i.e. clipping is barely
# engaged and AdamW's per-parameter normalisation is what bounds the run. Under that
# mechanism higher lr buys a worse optimum rather than a blow-up, so there may be no
# edge to find by this route -- which is a result, not a reason to keep climbing.
MULTIPLIERS=${MULTIPLIERS:-"1 2 4 8 16 32 64"}

mkdir -p "$LOG"
for mult in $MULTIPLIERS; do
    lr=$(.venv/bin/python -c "print($BASE_LR * $mult)")
    for arm in bf16 fp8_up; do
        tag="${arm}_lr${mult}x"
        [ -f "$OUT/$tag.json" ] && { echo "$(date -Is) $tag already done"; continue; }
        echo "$(date -Is) $tag lr=$lr on GPU$CARD"
        CUDA_VISIBLE_DEVICES=$CARD .venv/bin/python scripts/bench/fp8/fp8_training.py \
            --stage train --arm "$arm" --steps "$STEPS" --lr "$lr" --tag "$tag" \
            --out "$OUT" > "$LOG/$tag.log" 2>&1
        if [ -f "$OUT/$tag.json" ]; then
            echo "$(date -Is)   $tag OK"
        else
            echo "$(date -Is)   $tag DIED -- see $LOG/$tag.log"
        fi
    done
done
echo "$(date -Is) sweep done; plot with scripts/bench/fp8/fp8_stability_plot.py"
