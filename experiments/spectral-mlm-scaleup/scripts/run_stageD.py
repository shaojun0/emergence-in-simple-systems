#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage D: the upstream repo's own "most important" open question (Q1).

Finding 1 of the upstream report is that the learned spectral kernel collapses
to a local template (lag 0 = 93%, >8 = 0.6%) and that the model never uses the
global mixing it is parameterised for.  Upstream itself says this is a property
of the *task* (criterion 6), not of the operator.  These synthetic corpora make
the task's required range an explicit experimental variable:

  p4      x_t = b[t mod 4]      local  (a 3-layer |lag|<=2 stencil reaches +-6)
  p64     x_t = b[t mod 64]     long-range: the only route is a lag-64 copy
  double  blocks "s + s", block length random in [8,56]
                                content-dependent routing (induction head);
                                a data-INDEPENDENT convolution cannot do it

Expected, if the upstream argument is right:
  p4     spectral kernel stays local, stencil matches spectral
  p64    spectral kernel develops a spike at lag 64 and beats stencil decisively
  double spectral cannot solve it; attention can
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
SCRIPT = os.path.join(HERE, "spectral_lm_torch.py")
DATA = r"D:\dsh\work\data"
RUNS = os.path.join(ROOT, "runs")
LOGS = os.path.join(ROOT, "logs")

TASKS = {
    "p4": "synth_period4.txt",
    "p64": "synth_period64.txt",
    "double": "synth_double.txt",
}
ARMS = ("spectral", "stencil", "attn", "fnet")
STEPS = 3000


def gen_corpora():
    for name, L in (("synth_period4.txt", 4), ("synth_period32.txt", 32),
                    ("synth_period64.txt", 64)):
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            continue
        subprocess.check_call([PY, os.path.join(HERE, "make_synthetic.py"),
                               "--task", "periodic", "--L", str(L),
                               "--chars", "4000000", "--out", path])
    dbl = os.path.join(DATA, "synth_double.txt")
    if not os.path.exists(dbl):
        subprocess.check_call([PY, os.path.join(HERE, "make_synthetic.py"),
                               "--task", "double", "--chars", "4000000", "--out", dbl])


def job(task, mix, lr):
    data = os.path.join(DATA, TASKS[task])
    cmd = [PY, "-u", SCRIPT, "--mix", mix, "--data", data, "--encoding", "utf-8",
           "--outdir", os.path.join(RUNS, "synth"), "--tag", task,
           "--steps", str(STEPS), "--warmup", "100", "--lr", str(lr),
           "--eval_every", "100", "--eval_batches", "8", "--eval_batch", "8",
           "--device", "cuda", "--d", "96", "--L", "3", "--F", "384",
           "--T", "128", "--batch", "32", "--heads", "4", "--t_max", "512",
           "--extrapolate", "128", "256", "512", "--save_weights"]
    name = "%s_%s_lr%g" % (task, mix, lr)
    return dict(name=name, cmd=cmd, log=os.path.join(LOGS, "D_%s.log" % name),
                final=os.path.join(RUNS, "synth", "%s_%s_*.final.json" % (task, mix)))


def main():
    gen_corpora()
    import glob
    lrs = {"spectral": 0.8, "stencil": 0.8, "attn": 0.2, "fnet": 0.8}
    jobs = [job(t, m, lrs[m]) for t in ("p4", "p64", "double") for m in ARMS]
    os.makedirs(os.path.join(RUNS, "synth"), exist_ok=True)
    print("stage D: %d jobs" % len(jobs), flush=True)
    t_all = time.time()
    for j in jobs:
        if glob.glob(j["final"]):
            print("  skip %s" % j["name"], flush=True)
            continue
        t0 = time.time()
        with open(j["log"], "w") as fh:
            rc = subprocess.call(j["cmd"], stdout=fh, stderr=subprocess.STDOUT)
        print("  %-28s rc=%d  %.0fs" % (j["name"], rc, time.time() - t0), flush=True)
    print("stage D done in %.0fs" % (time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
