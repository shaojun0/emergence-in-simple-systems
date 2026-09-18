#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage C: the scale-up.  Same objective, same optimizer, same skeleton as the
upstream CPU experiment -- but on enwik8 (100 MB, byte-level) instead of Tiny
Shakespeare (1.1 MB), at d/L/T up to 4x/2x/4x the upstream size, on GPU.

Runs sequentially (one GPU).  A job is skipped if its .final.json exists, so the
driver can be re-invoked after an interruption.

  python run_stageC.py lr        # learning-rate sweep at the smallest rung
  python run_stageC.py ladder    # the main scale ladder
  python run_stageC.py extra     # fnet/gated at the top rung + the LN-bug control
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
SCRIPT = os.path.join(HERE, "spectral_lm_torch.py")
DATA = r"D:\dsh\work\data\enwik8"
RUNS = os.path.join(ROOT, "runs")
LOGS = os.path.join(ROOT, "logs")

# name -> (d, L, F, T, batch, heads)
SCALES = {
    "s1": dict(d=96, L=3, F=384, T=128, batch=32, heads=4),
    "s2": dict(d=192, L=4, F=768, T=256, batch=16, heads=6),
    "s3": dict(d=384, L=6, F=1536, T=512, batch=8, heads=6),
}
LADDER_STEPS = 10000
LR_STEPS = 2500
EXTRA = ["--extrapolate", "128", "256", "512", "1024"]


def job(group, mix, scale, lr, steps, seed=0, extra=(), save=False, tag=""):
    sc = SCALES[scale]
    cmd = [PY, "-u", SCRIPT, "--mix", mix, "--data", DATA, "--encoding", "latin-1",
           "--outdir", os.path.join(RUNS, group), "--tag", tag or scale,
           "--steps", str(steps), "--warmup", str(min(200, max(20, steps // 20))),
           "--lr", str(lr), "--seed", str(seed),
           "--eval_every", str(max(50, steps // 40)), "--eval_batches", "8",
           "--eval_batch", "8", "--device", "cuda",
           "--d", str(sc["d"]), "--L", str(sc["L"]), "--F", str(sc["F"]),
           "--T", str(sc["T"]), "--batch", str(sc["batch"]),
           "--heads", str(sc["heads"]), "--t_max", "1024"]
    cmd += list(extra)
    if save:
        cmd.append("--save_weights")
    name = "%s_%s_%s_lr%g_s%d" % (group, scale, mix, lr, seed)
    return dict(name=name, group=group, cmd=cmd,
                log=os.path.join(LOGS, "C_%s.log" % name),
                final=os.path.join(RUNS, group,
                                   "%s_%s_T*_s%d_lr%g.final.json"
                                   % (tag or scale, mix, seed, lr)))


def build(which):
    jobs = []
    if which == "lr":
        for lr in (0.05, 0.2, 0.8):
            for mix in ("spectral", "attn", "stencil", "fnet"):
                jobs.append(job("lr", mix, "s1", lr, LR_STEPS, tag="lrsweep"))
    elif which == "ladder":
        # From the lr sweep at s1 (2500 steps, enwik8): lr 0.8 is best or
        # near-best for every arm, so the ladder uses ONE matched setting for
        # all arms and scales -- the cleanest controlled comparison.
        best = {"s1": dict(spectral=0.8, attn=0.8, stencil=0.8, fnet=0.8),
                "s2": dict(spectral=0.8, attn=0.8, stencil=0.8, fnet=0.8),
                "s3": dict(spectral=0.8, attn=0.8, stencil=0.8, fnet=0.8)}
        for scale in ("s1", "s2", "s3"):
            for mix, lr in best[scale].items():
                save = mix in ("spectral", "stencil")
                jobs.append(job("ladder", mix, scale, lr, LADDER_STEPS,
                                extra=EXTRA if mix in ("spectral", "attn", "stencil") else (),
                                save=save))
    elif which == "fair":
        # upstream Q7: is attention's plain-SGD collapse an lr artefact?
        # Give it the two documented fixes at full ladder length.
        for scale in ("s1", "s2", "s3"):
            j = job("fair", "attn", scale, 0.2, LADDER_STEPS, tag="%s_lr02" % scale,
                    extra=EXTRA)
            jobs.append(j)
            j = job("fair", "attn", scale, 0.8, LADDER_STEPS, tag="%s_wozero" % scale,
                    extra=EXTRA)
            j["cmd"].append("--wo_zero")
            jobs.append(j)
    elif which == "extra":
        for mix in ("fnet", "gated"):
            jobs.append(job("extra", mix, "s3", 0.8, LADDER_STEPS,
                            extra=EXTRA, save=(mix == "gated")))
        # does upstream's LayerNorm/W1 gradient bug change the outcome at scale?
        for bug in (False, True):
            j = job("extra", "spectral", "s2", 0.8, LADDER_STEPS,
                    tag="lnbug%s" % ("_on" if bug else "_off"),
                    extra=EXTRA, save=True)
            if bug:
                j["cmd"].append("--upstream_bug")
            j["name"] += "_bug" if bug else "_fixed"
            jobs.append(j)
    return jobs


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ladder"
    jobs = build(which)
    os.makedirs(LOGS, exist_ok=True)
    for g in set(j["group"] for j in jobs):
        os.makedirs(os.path.join(RUNS, g), exist_ok=True)
    print("stage C [%s]: %d jobs" % (which, len(jobs)), flush=True)
    t_all = time.time()
    for j in jobs:
        done = [f for f in _glob(j["final"]) if os.path.exists(f)]
        if done:
            print("  skip %s (already done)" % j["name"], flush=True)
            continue
        t0 = time.time()
        with open(j["log"], "w") as fh:
            rc = subprocess.call(j["cmd"], stdout=fh, stderr=subprocess.STDOUT)
        print("  %-40s rc=%d  %.0fs" % (j["name"], rc, time.time() - t0), flush=True)
    print("stage C [%s] done in %.0fs" % (which, time.time() - t_all), flush=True)


def _glob(pat):
    import glob
    return glob.glob(pat)


if __name__ == "__main__":
    main()
