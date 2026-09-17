#!/bin/bash
# Robustness controls for the two threats to the main claim:
#   (1) is attention's failure just a bad initial basin?  -> --wo_zero (block
#       starts as exact identity, GPT-2 style) at two learning rates
#   (2) is the result seed noise?                          -> seed 1 for the
#       two arms that matter (spectral vs attn)
cd /home/linaro/dsh/spectral_mlm_cpu || exit 1
mkdir -p runs/followup logs
COMMON="--steps 3000 --eval_every 250 --eval_batches 6 --eval_batch 8 --outdir runs/followup"
OMP_NUM_THREADS=1 python3 -u spectral_bert.py --mix attn --wo_zero --lr 0.8 $COMMON \
    --seed 0 > logs/fu_attn_wozero_lr08_s0.log 2>&1 &
OMP_NUM_THREADS=1 python3 -u spectral_bert.py --mix attn --wo_zero --lr 1.2 $COMMON \
    --seed 0 > logs/fu_attn_wozero_lr12_s0.log 2>&1 &
OMP_NUM_THREADS=1 python3 -u spectral_bert.py --mix attn --lr 0.8 $COMMON \
    --seed 1 > logs/fu_attn_s1.log 2>&1 &
OMP_NUM_THREADS=1 python3 -u spectral_bert.py --mix spectral --lr 1.2 $COMMON \
    --seed 1 > logs/fu_spectral_s1.log 2>&1 &
wait
echo "ALL FOLLOWUPS DONE"
