#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the three-arm comparison from runs/main/*.jsonl and *.final.json."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = {"unigram (zero context)": (3.3477, 0.1460),
        "trigram Markov": (2.4450, 0.3840)}

COLORS = {"spectral": "#1f77b4", "fnet": "#2ca02c", "attn": "#d62728"}
LABEL = {"spectral": "spectral (learnable FFT filter, 28k mix params)",
         "fnet": "fnet (unparameterized FFT, 0 mix params)",
         "attn": "attn (4-head self-attention, 112k mix params)"}


def load(pattern="runs/main/*.jsonl"):
    runs = {}
    for f in sorted(glob.glob(pattern)):
        mix = os.path.basename(f).split("_")[0]
        hist = []
        for l in open(f):
            if not l.strip():
                continue
            try:
                hist.append(json.loads(l))
            except ValueError:
                pass  # truncated last line from a killed process
        runs[mix] = hist
    return runs


def finals(pattern="runs/main/*.final.json"):
    out = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        out[d["meta"]["mix"]] = d
    return out


def main():
    runs, fin = load(), finals()
    if not runs:
        print("no runs/main/*.jsonl yet")
        return
    # robustness controls: per-arm learning rate / init / seed variations
    fu = {}
    for f in sorted(glob.glob("runs/followup/*.jsonl")):
        hist = []
        for l in open(f):
            if not l.strip():
                continue
            try:
                hist.append(json.loads(l))
            except ValueError:
                pass
        if hist and "wozero" not in f:      # the wozero jsonl is corrupted by
            fu[os.path.basename(f)[:-6]] = hist   # a tag collision; see REPORT.md
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0][0]
    for mix, hist in runs.items():
        ax.plot([h["step"] for h in hist], [h["val_loss"] for h in hist],
                color=COLORS.get(mix), label=LABEL.get(mix, mix), lw=2)
    for name, hist in fu.items():
        ax.plot([h["step"] for h in hist], [h["val_loss"] for h in hist],
                color=COLORS.get(name.split("_")[0]), lw=1.1, ls="--", alpha=.75,
                label="control: " + name.replace("_T96_d96_L3", ""))
    for name, (h, _) in BASE.items():
        ax.axhline(h, ls="--", lw=1, color="gray")
        ax.text(ax.get_xlim()[1], h, " " + name, va="center", fontsize=8, color="gray")
    ax.set_xlabel("step"); ax.set_ylabel("validation masked-LM cross-entropy")
    ax.set_title("BERT-style masked LM, CPU, NumPy only, plain SGD")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[0][1]
    for mix, hist in runs.items():
        ax.plot([h["step"] for h in hist], [h["val_acc"] for h in hist],
                color=COLORS.get(mix), label=mix, lw=2)
    for name, hist in fu.items():
        ax.plot([h["step"] for h in hist], [h["val_acc"] for h in hist],
                color=COLORS.get(name.split("_")[0]), lw=1.1, ls="--", alpha=.75,
                label="control: " + name.replace("_T96_d96_L3", ""))
    for name, (_, a) in BASE.items():
        ax.axhline(a, ls="--", lw=1, color="gray")
        ax.text(ax.get_xlim()[1], a, " " + name, va="center", fontsize=8, color="gray")
    ax.set_xlabel("step"); ax.set_ylabel("validation masked-token accuracy")
    ax.set_title("masked-token accuracy"); ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = axes[1][0]
    for mix, d in fin.items():
        ex = d.get("extrapolation") or {}
        if not ex:
            continue
        ts = sorted(int(k) for k in ex)
        ax.plot(ts, [ex[str(t)]["val_loss"] for t in ts], "o-",
                color=COLORS.get(mix), label=mix, lw=2)
    ax.axvline(d.get("meta", {}).get("T", 96), ls=":", color="k", lw=1)
    ax.text(d.get("meta", {}).get("T", 96), ax.get_ylim()[1], " trained T ",
            fontsize=8, va="top")
    ax.set_xscale("log", base=2); ax.set_xticks([96, 192, 384, 768])
    ax.set_xticklabels(["96", "192", "384", "768"])
    ax.set_xlabel("evaluation sequence length (never trained on)")
    ax.set_ylabel("validation loss")
    ax.set_title("zero-shot length extrapolation"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1][1]
    for mix, d in fin.items():
        spec = d.get("spectrum")
        if not spec:
            continue
        for l, prof in enumerate(spec):
            ax.plot(prof, color=COLORS.get(mix), alpha=.35 + .25 * l,
                    label="%s layer %d" % (mix, l) if l == 0 else None)
    ax.set_xlabel("frequency index k")
    ax.set_ylabel("mean |H(k)|  (1.0 = identity filter)")
    ax.set_title("learned spectral filter magnitude (spectral arm)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig("runs/main/comparison.png", dpi=130)
    print("wrote runs/main/comparison.png")


if __name__ == "__main__":
    main()
