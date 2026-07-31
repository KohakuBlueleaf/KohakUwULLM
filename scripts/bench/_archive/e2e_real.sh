#!/usr/bin/env bash
# The real end-to-end sweep: dense and MoE, pipeline and DDP, at 262144 tok/step.
#
# bf16 parameters throughout -- 8B-A1B does not fit otherwise, and the dense
# stage measurements put native bf16 ~3% ahead of fp32-under-autocast.

set -u
cd /xg7/KohakUwULLM
LOG=${LOG:-out/bench/train/matrix/e2e_real.log}
mkdir -p out/bench/train/matrix
: >"$LOG"

run() {
  echo "=== $1 $5 micro=$3 ===" | tee -a "$LOG"
  timeout 2400 .venv/bin/torchrun --standalone --nproc_per_node=4 \
    scripts/bench/e2e/step_throughput.py --preset "$1" --mode "$2" --budget 262144 \
    --micro-tokens $3 --param-dtype bf16 --warmup 2 --iters 3 \
    --out out/bench/train/matrix --tag "$5" $4 2>&1 \
    | grep -aE "pp4|ddp4|budget|OOM|Error" | tee -a "$LOG"
  sleep 4
}

for preset in Nano-1B MoE-1B-A120M MoE-2B-A370M MoE-3B-A500M MoE-3B-A500M-deep MoE-8B-A1B; do
  run "$preset" pipeline "4096 8192" "" "pp4_bf16"
  run "$preset" ddp "4096 8192" "--grad-ckpt" "ddp4_bf16"
done
echo "E2E REAL DONE" | tee -a "$LOG"
