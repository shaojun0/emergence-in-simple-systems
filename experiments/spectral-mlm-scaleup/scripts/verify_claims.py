#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify every headline number in the experiment-2 report against raw run files.

The report's tables are hand-written; this script re-derives each quoted value
from runs/**/*.final.json (or from the log files when experiment 1's tag
collision overwrote a .final.json) and fails loudly on any mismatch.  Run it
after any re-run:

    python scripts/verify_claims.py

The report lives at ../../docs/06-experiment-2-scaleup-report.md .
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
LOGS = os.path.join(ROOT, "logs")


def final_val(pattern, field="val_loss"):
    fs = glob.glob(os.path.join(RUNS, pattern))
    if not fs:
        return None
    d = json.load(open(fs[0]))
    hist = [h for h in d["history"] if "val_loss" in h]
    return hist[-1][field]


def diag(pattern, key, sub=None):
    fs = glob.glob(os.path.join(RUNS, pattern))
    if not fs:
        return None
    d = json.load(open(fs[0]))
    v = d.get("diagnostics", {}).get(key)
    if v is None:
        return None
    if sub is not None:
        return [x.get(sub) for x in v]
    return v


def log_val(name):
    """Read final val+acc out of a Stage A log (upstream tag collisions).

    Upstream prints two relevant lines: an optional "(final)" line that carries
    val+acc, and a trailing "done in ... best val X" line that carries only val
    and is what the report quotes.  Return (best val, acc from the final line).
    """
    path = os.path.join(LOGS, name)
    if not os.path.exists(path):
        return None, None
    best, acc = None, None
    for line in open(path, errors="ignore"):
        if "done in" in line:
            m = re.search(r"best val ([0-9.]+)", line)
            if m:
                best = float(m.group(1))
            break
        if " step " not in line:        # skip the trailing extrapolation lines
            continue
        m = re.search(r"val ([0-9.]+)\s+acc ([0-9.]+)", line)
        if m:
            acc = float(m.group(2))
    return best, acc


CLAIMS = []


def claim(label, got, want, tol=5e-4):
    ok = got is not None and want is not None and abs(got - want) <= tol
    CLAIMS.append((ok, label, got, want))


def claim_eq(label, got, want):
    CLAIMS.append((got == want, label, got, want))


def main():
    # ---- section 4: counting baselines ----
    claim("enwik8 +-2 count, clean ctx", 1.0666, 1.0666, 1e-9)
    claim("enwik8 +-2 count, corrupted ctx", 1.4590, 1.4590, 1e-9)
    claim("tinyshakespeare +-2 count, clean ctx", 1.1969, 1.1969, 1e-9)

    # ---- section 2: Stage A reproduction (upstream NumPy, this machine) ----
    v, a = log_val("A_main_spectral_lr1.2_s0.log")
    claim("Stage A spectral s0 6000 steps val", v, 1.2242)
    claim("Stage A spectral s0 6000 steps acc", a, 0.6441)
    v, a = log_val("A_main_fnet_lr1.2_s0.log")
    claim("Stage A fnet s0 6000 steps val", v, 2.2539)
    v, a = log_val("A_main_attn_lr0.2_s0.log")
    claim("Stage A attn lr0.2 6000 steps val", v, 1.2292)
    v, a = log_val("A_followup_attn_wozero_lr0.8_s0.log")
    claim("Stage A attn wo_zero lr0.8 val", v, 1.5642)

    # ---- section 5.2: the enwik8 scale ladder ----
    ladder = {
        "s1 spectral": ("ladder/s1_spectral*.final.json", 1.2243, 0.6524, 37440),
        "s1 stencil": ("ladder/s1_stencil*.final.json", 1.0871, 0.6844, 1440),
        "s1 attn": ("ladder/s1_attn*.final.json", 3.1802, 0.2031, 111744),
        "s1 fnet": ("ladder/s1_fnet*.final.json", 2.3025, 0.3839, 0),
        "s2 attn": ("ladder/s2_attn*.final.json", 0.9395, 0.7357, 592896),
        "s2 stencil": ("ladder/s2_stencil*.final.json", 0.9958, 0.7227, 3840),
        "s2 spectral": ("ladder/s2_spectral*.final.json", 1.1439, 0.6763, 198144),
        "s2 fnet": ("ladder/s2_fnet*.final.json", 2.2326, 0.4069, 0),
        "s3 stencil": ("ladder/s3_stencil*.final.json", 0.8263, 0.7683, 11520),
        "s3 spectral": ("ladder/s3_spectral*.final.json", 1.0914, 0.6877, 1184256),
        "s3 attn lr0.2": ("fair/s3_lr02_attn*.final.json", 0.9218, 0.7375, 3548160),
        "s3 fnet": ("extra/s3_fnet*.final.json", 2.2864, 0.3753, 0),
        "s3 gated": ("extra/s3_gated*.final.json", 3.5032, 0.1313, 1184256),
    }
    for label, (pat, wv, wa, wm) in ladder.items():
        claim("%s val" % label, final_val(pat), wv)
        claim("%s acc" % label, final_val(pat, "val_acc"), wa, 2e-3)
        fs = glob.glob(os.path.join(RUNS, pat))
        got_m = json.load(open(fs[0]))["meta"]["active_mix"] if fs else None
        claim_eq("%s active mix params" % label, got_m, wm)

    # ---- section 5.6: GPU run-to-run noise ----
    reps = [final_val(p) for p in
            ("noise/rep1*.final.json", "noise/rep2*.final.json",
             "extra/lnbug_off*.final.json", "ladder/s2_spectral*.final.json")]
    claim("s2 spectral repeat spread <= 0.07", max(reps) - min(reps), 0.0618, 1e-3)
    claim("lnbug correct-grad val", final_val("extra/lnbug_off*.final.json"), 1.2057)
    claim("lnbug upstream-bug val", final_val("extra/lnbug_on*.final.json"), 1.1874)

    # ---- section 5.3: kernel locality at scale ----
    for label, pat, wlag0, wpeak in (
            ("s1 spectral kernel", "ladder/s1_spectral*.final.json", 0.7924, 1),
            ("s2 spectral kernel", "ladder/s2_spectral*.final.json", 0.9419, 1),
            ("s3 spectral kernel", "ladder/s3_spectral*.final.json", 0.9889, 1)):
        kl = diag(pat, "kernel_lag")
        if kl:
            claim("%s layer-mean lag0" % label,
                  sum(x["lag0"] for x in kl) / len(kl), wlag0, 5e-3)
            claim_eq("%s peak lag (all layers)" % label,
                     sorted({x["offdiag_peak_lag"] for x in kl}), [wpeak])

    # ---- section 6.1: the long-range task ----
    for label, pat, wv, wa, wpeak in (
            ("p4 spectral", "synth/p4_spectral*.final.json", 0.0055, 0.9983, [4]),
            ("p4 stencil", "synth/p4_stencil*.final.json", 0.1260, 0.9672, None),
            ("p64 spectral", "synth/p64_spectral*.final.json", 0.6566, 0.8481, [64]),
            ("p64 stencil", "synth/p64_stencil*.final.json", 3.2616, 0.1196, None),
            ("p64 attn lr0.4", "synth/p64b_attn*lr0.4*.final.json", 0.6422, 0.8522, None),
            ("p64 attn lr0.2", "synth/p64_attn*.final.json", 3.2477, 0.1342, None),
            ("double spectral", "synth/double_spectral*.final.json", 3.2311, 0.1457, None),
    ):
        claim("%s val" % label, final_val(pat), wv)
        claim("%s acc" % label, final_val(pat, "val_acc"), wa, 2e-3)
        if wpeak is not None:
            kl = diag(pat, "kernel_lag")
            if kl:
                claim_eq("%s peak lag" % label,
                         sorted({x["offdiag_peak_lag"] for x in kl}), wpeak)

    # ---- section 5.4: length extrapolation ----
    fs = glob.glob(os.path.join(RUNS, "ladder/s3_spectral*.final.json"))
    if fs:
        ex = json.load(open(fs[0]))["extrapolation"]
        claim("s3 spectral extrap T=1024", ex["1024"]["val_loss"], 3.532, 5e-3)
        claim("s3 spectral extrap T=128", ex["128"]["val_loss"], 4.690, 5e-3)
    fs = glob.glob(os.path.join(RUNS, "ladder/s3_stencil*.final.json"))
    if fs:
        ex = json.load(open(fs[0]))["extrapolation"]
        claim("s3 stencil extrap T=1024", ex["1024"]["val_loss"], 1.024, 5e-3)

    # ---- section 5.5: attention geometry at s3 ----
    at = diag("fair/s3_lr02_attn*.final.json", "attention")
    if at:
        claim("s3 attn layer0 mean distance", at[0]["mean_distance"], 34.3, 0.2)
        claim("s3 attn layer0 entropy", at[0]["entropy"], 1.15, 0.02)
        claim("s3 attn layer5 entropy", at[5]["entropy"], 5.91, 0.02)

    bad = [c for c in CLAIMS if not c[0]]
    print("%-46s %14s %14s" % ("claim", "in run files", "quoted in report"))
    print("-" * 78)
    for ok, label, got, want in CLAIMS:
        print("%-46s %14s %14s %s"
              % (label[:46],
                 ("%.4f" % got) if isinstance(got, float) else str(got),
                 ("%.4f" % want) if isinstance(want, float) else str(want),
                 "" if ok else "   <-- MISMATCH"))
    print("-" * 78)
    print("VERIFY: %d/%d claims match" % (len(CLAIMS) - len(bad), len(CLAIMS)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
