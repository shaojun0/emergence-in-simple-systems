#!/bin/bash
# Main 3-arm comparison at each arm's own best learning rate (from the sweep).
# 2 threads per arm, three arms in parallel.
cd /home/linaro/dsh/spectral_mlm_cpu || exit 1
mkdir -p runs/main logs
SEED=${1:-0}
STEPS=${2:-6000}
run() {
  OMP_NUM_THREADS=2 python3 -u spectral_bert.py \
    --mix "$1" --lr "$2" --steps "$STEPS" --seed "$SEED" \
    --eval_every 250 --eval_batches 12 --eval_batch 8 \
    --extrapolate 96 192 384 768 \
    --outdir runs/main > "logs/main_${1}_s${SEED}.log" 2>&1
}
run spectral 1.2 &
run fnet     1.2 &
run attn     0.8 &
wait
echo "ALL DONE seed=$SEED steps=$STEPS"
