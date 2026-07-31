#!/usr/bin/env bash
# Run the whole benchmark suite and regenerate every figure under out/bench/.
#
# Single-GPU stages pin GPU 0 and must own it: a contended run does not fail, it
# reads low, and nothing in the numbers says so. The end-to-end stage uses all four
# via torchrun. Each stage writes its own JSON next to its PNG, so a stage that
# fails leaves the others' figures intact.
#
#   bash scripts/bench/run_all.sh              # everything
#   bash scripts/bench/run_all.sh e2e          # one stage
set -u

PY=${PY:-.venv/bin/python}
# Persist Triton's compile/autotune cache across the isolated benchmark
# processes; otherwise each one re-tunes every kernel from scratch.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$PWD/.triton_cache}
mkdir -p "$TRITON_CACHE_DIR"
OUT=${OUT:-out/bench}
STAGES=${*:-hgemm bandwidth attention kernels head head_options chunked_ce global_ratio data e2e}
mkdir -p "$OUT"

run() {
  local name=$1; shift
  echo ""
  echo "==================== $name ===================="
  local start=$SECONDS
  if "$@" > "$OUT/$name.log" 2>&1; then
    echo "  OK  ($((SECONDS - start))s)  -> $OUT/$name.log"
  else
    echo "  FAILED ($((SECONDS - start))s) -- last lines of $OUT/$name.log:"
    tail -15 "$OUT/$name.log" | sed 's/^/    /'
  fi
}

for stage in $STAGES; do
  case $stage in
    hgemm)
      # --out is `hgemm`, not `hgemm_acc`: the script's own default, kept so a
      # re-run lands on the existing directory rather than forking a second one.
      CUDA_VISIBLE_DEVICES=0 run hgemm $PY scripts/bench/kernel/hgemm_acc.py --out "$OUT/hgemm" ;;
    bandwidth)
      # First, and not only for ordering: every "% of peak" in the stages below
      # divides by a bandwidth this stage is the reference for.
      CUDA_VISIBLE_DEVICES=0 run bandwidth $PY scripts/bench/kernel/bandwidth.py --out "$OUT/bandwidth" ;;
    attention)
      CUDA_VISIBLE_DEVICES=0 run attention $PY scripts/bench/kernel/attention.py --out "$OUT/attention" ;;
    kernels)
      CUDA_VISIBLE_DEVICES=0 run kernels $PY scripts/bench/kernel/kernels.py --out "$OUT/kernels"
      # The one measurement here that draws nothing: its three figures live in a
      # separate plotter, so a rejected layout cannot cost the measurement again.
      run kernels_plot $PY scripts/bench/kernel/kernels_plot.py --dir "$OUT/kernels" ;;
    head)
      CUDA_VISIBLE_DEVICES=0 run head $PY scripts/bench/kernel/head.py --out "$OUT/head" ;;
    head_options)
      CUDA_VISIBLE_DEVICES=0 run head_options $PY scripts/bench/kernel/head_options.py \
        --out "$OUT/head_options" ;;
    chunked_ce)
      CUDA_VISIBLE_DEVICES=0 run chunked_ce $PY scripts/bench/kernel/chunked_ce.py \
        --out "$OUT/chunked_ce" ;;
    global_ratio)
      # The matrix script, not the single sweep: one preset at one context answers
      # the ratio question only for that point on the grid. Its LOG is redirected
      # into the result directory because its default collides with the $OUT/$name.log
      # this driver already holds open, and the inner truncate corrupts both.
      GPU=0 LOG="$OUT/global_ratio/matrix.log" \
        run global_ratio bash scripts/bench/e2e/global_ratio_matrix.sh ;;
    data)
      run data $PY scripts/bench/data/data.py --out "$OUT/data" \
        --roots /xg7/caption-datasets /Yamiyoru/caption-datasets ;;
    e2e)
      # The Kohaku sweep drives step_throughput itself, one torchrun per row, so it
      # takes no --presets: the ladder is the point and a subset is not a ladder.
      run e2e bash scripts/bench/e2e/e2e_kohaku_mxfp8.sh ;;
    *)
      echo "unknown stage: $stage" ;;
  esac
done

echo ""
echo "==================== figures ===================="
ls -la "$OUT"/*/*.png 2>/dev/null
