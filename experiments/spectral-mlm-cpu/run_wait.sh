#!/bin/bash
# Wait until every run except the 6000-step attention arm has finished, then
# stop that arm too.  Rationale: the attention arm is ~2x slower per step in
# NumPy, so 6000 steps would run until ~22:30; the comparison is made at equal
# step counts, and by the time the others finish it will have ~2500-3000 steps,
# which is enough to compare against the same prefix of spectral/fnet.
cd /home/linaro/dsh/spectral_mlm_cpu || exit 1
wait_for() {
  while pgrep -f "$1" >/dev/null; do sleep 20; done
  echo "$(date +%H:%M) finished: $1"
}
wait_for "mix spectral --lr 1.2 --steps 6000"
wait_for "mix fnet --lr 1.2 --steps 6000"
wait_for "outdir runs/followup"
echo "$(date +%H:%M) stopping the long attention arm"
pkill -f "mix attn --lr 0.8 --steps 6000"
sleep 5
echo "DONE: remaining=$(pgrep -fc '[s]pectral_bert.py --mix')"
