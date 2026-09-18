#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage D2: harder/longer follow-ups to Stage D.

  * p64 with attention at its other learning rate and 2x the steps -- is
    attention's failure on the long-range copy an lr/budget artefact?
  * an anchored variable-lag copy task ("s \\n s \\n") where content-based
    routing is genuinely required; does attention beat the fixed spectral
    filter there?
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
SCRIPT = os.path.join(HERE, "spectral_lm_torch.py")
DATA = r"D:\dsh\work\data"
RUNS = os.path.join(ROOT, "runs", "synth")
LOGS = os.path.join(ROOT, "logs")
STEPS = 6000


def job(tag, data, mix, lr, seed=0):
    cmd = [PY, "-u", SCRIPT, "--mix", mix, "--data", data, "--encoding", "utf-8",
           "--outdir", RUNS, "--tag", tag, "--steps", str(STEPS),
           "--warmup", "100", "--lr", str(lr), "--seed", str(seed),
           "--eval_every", "150", "--eval_batches", "8", "--eval_batch", "8",
           "--device", "cuda", "--d", "96", "--L", "3", "--F", "384",
           "--T", "128", "--batch", "32", "--heads", "4", "--t_max", "512",
           "--extrapolate", "128", "256", "512", "--save_weights"]
    name = "%s_%s_lr%g_s%d" % (tag, mix, lr, seed)
    return dict(name=name, cmd=cmd, log=os.path.join(LOGS, "D2_%s.log" % name),
                final=os.path.join(RUNS, "%s_%s_T*_lr%g_s%d.final.json"
                                   % (tag, mix, lr, seed)))


def main():
    ind = os.path.join(DATA, "synth_induct.txt")
    if not os.path.exists(ind):
        subprocess.check_call([PY, os.path.join(HERE, "make_synthetic.py"),
                               "--task", "induct", "--chars", "4000000", "--out", ind])
    jobs = []
    for mix, lr in (("attn", 0.8), ("attn", 0.4), ("spectral", 0.8), ("stencil", 0.8)):
        jobs.append(job("p64b", os.path.join(DATA, "synth_period64.txt"), mix, lr))
    for mix, lr in (("attn", 0.2), ("attn", 0.8), ("spectral", 0.8),
                    ("stencil", 0.8), ("fnet", 0.8)):
        jobs.append(job("induct", ind, mix, lr))
    print("stage D2: %d jobs" % len(jobs), flush=True)
    for j in jobs:
        if glob.glob(j["final"]):
            print("  skip %s" % j["name"], flush=True)
            continue
        t0 = time.time()
        with open(j["log"], "w") as fh:
            rc = subprocess.call(j["cmd"], stdout=fh, stderr=subprocess.STDOUT)
        print("  %-30s rc=%d %.0fs" % (j["name"], rc, time.time() - t0), flush=True)
    print("stage D2 done", flush=True)


if __name__ == "__main__":
    main()
