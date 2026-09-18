#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage A: faithful reproduction of experiment 1 at its ORIGINAL scale.

Runs the upstream `spectral_bert.py` (pure NumPy, hand-derived gradients,
plain SGD) unmodified, on this machine, to establish a ground truth before
any scaling.  Equivalent to the upstream run_main.sh / run_followup.sh plus a
longer learning-rate sweep (upstream Q7 asks for one).

Usage:  python run_stageA.py [group ...]      # default: main followup
        python run_stageA.py list
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

UPSTREAM = r"D:\dsh\work\eiss\experiments\spectral-mlm-cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # D:\dsh\work\scaleup
DATA = os.path.join(UPSTREAM, "tinyshakespeare.txt")
RUNS = os.path.join(ROOT, "runs")
LOGS = os.path.join(ROOT, "logs")

EXTRAP = ["--extrapolate", "96", "192", "384", "768"]


def job(name, mix, lr, steps, group, seed=0, extra=(), eval_every=250,
        eval_batches=12, eval_batch=8, warmup=200, save_weights=False):
    outdir = os.path.join(RUNS, group)
    cmd = [sys.executable, "-u", os.path.join(UPSTREAM, "spectral_bert.py"),
           "--mix", mix, "--lr", str(lr), "--steps", str(steps),
           "--seed", str(seed), "--data", DATA, "--outdir", outdir,
           "--eval_every", str(eval_every), "--eval_batches", str(eval_batches),
           "--eval_batch", str(eval_batch), "--warmup", str(warmup)]
    cmd += list(extra)
    if save_weights:
        cmd.append("--save_weights")
    return dict(name=name, group=group, cmd=cmd,
                log=os.path.join(LOGS, "A_%s_%s.log" % (group, name)))


def build(groups):
    jobs = []
    if "main" in groups:
        # upstream run_main.sh, seed 0, 6000 steps, each arm at its own best lr
        jobs.append(job("spectral_lr1.2_s0", "spectral", 1.2, 6000, "main",
                        extra=EXTRAP, save_weights=True))
        jobs.append(job("fnet_lr1.2_s0", "fnet", 1.2, 6000, "main", extra=EXTRAP))
        jobs.append(job("attn_lr0.8_s0", "attn", 0.8, 6000, "main", extra=EXTRAP))
        jobs.append(job("attn_lr0.2_s0", "attn", 0.2, 6000, "main", extra=EXTRAP))
    if "followup" in groups:
        # upstream run_followup.sh, 3000 steps
        c = dict(steps=3000, eval_batches=6)
        jobs.append(job("attn_wozero_lr0.8_s0", "attn", 0.8, group="followup",
                        extra=("--wo_zero",), **c))
        jobs.append(job("attn_wozero_lr1.2_s0", "attn", 1.2, group="followup",
                        extra=("--wo_zero",), **c))
        jobs.append(job("attn_lr0.8_s1", "attn", 0.8, group="followup", seed=1, **c))
        jobs.append(job("spectral_lr1.2_s1", "spectral", 1.2, group="followup",
                        seed=1, **c))
    if "lrsweep" in groups:
        for lr in (0.05, 0.1, 0.2, 0.4, 0.8, 1.2):
            for mix in ("spectral", "fnet", "attn"):
                jobs.append(job("%s_lr%s_s0" % (mix, lr), mix, lr, 3000,
                                group="lrsweep", eval_every=250, eval_batches=6,
                                eval_batch=8))
    if "diag" in groups:
        # short diagnostic run with weights saved, matching upstream runs/diag
        jobs.append(job("spectral_lr1.2_s2", "spectral", 1.2, 1500, "diag",
                        seed=2, save_weights=True, eval_every=250,
                        eval_batches=6, eval_batch=8))
    return jobs


def main():
    args = [a for a in sys.argv[1:]]
    if args and args[0] == "list":
        for j in build(["main", "followup", "lrsweep", "diag"]):
            print(j["group"], j["name"], " ".join(j["cmd"][2:]))
        return
    groups = args or ["main", "followup"]
    if "all" in groups:
        groups = ["main", "followup", "lrsweep", "diag"]
    jobs = build(groups)
    os.makedirs(LOGS, exist_ok=True)
    for g in set(j["group"] for j in jobs):
        os.makedirs(os.path.join(RUNS, g), exist_ok=True)

    threads = max(1, 14 // max(1, len(jobs)))
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    print("stage A: %d jobs, %d threads each" % (len(jobs), threads), flush=True)
    t0 = time.time()

    def run(j):
        t = time.time()
        with open(j["log"], "w") as fh:
            rc = subprocess.call(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env)
        print("  %-28s rc=%d  %.0fs" % (j["name"], rc, time.time() - t), flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
        rcs = list(ex.map(run, jobs))
    print("stage A done in %.0fs; failures=%d"
          % (time.time() - t0, sum(1 for r in rcs if r)), flush=True)


if __name__ == "__main__":
    main()
