#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the scale-up report."""
import glob
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
LN2 = math.log(2)


def read_final(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def read_jsonl(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        rows = []
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        if rows:
            out.append((os.path.basename(f)[:-6], rows))
    return out


def fig_ladder():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    colors = dict(spectral="tab:red", attn="tab:blue", stencil="tab:green",
                  fnet="tab:gray", gated="tab:purple")
    base = 1.0666
    fair = 1.4590
    for ax, scale in zip(axes, ("s1", "s2", "s3")):
        for name, rows in read_jsonl(os.path.join(ROOT, "runs", "ladder",
                                                  "%s_*.jsonl" % scale)):
            mix = name.split("_")[1]
            tok = [(r["step"] + 1) * 4096 / 1e6 for r in rows]
            ax.plot(tok, [r["val_loss"] for r in rows], label=mix,
                    color=colors.get(mix, "k"), lw=1.6)
        for name, rows in read_jsonl(os.path.join(ROOT, "runs", "fair",
                                                  "%s_*.jsonl" % scale)):
            mix = name.split("_")[2]
            tok = [(r["step"] + 1) * 4096 / 1e6 for r in rows]
            ax.plot(tok, [r["val_loss"] for r in rows], label="%s (lr0.2)" % mix,
                    color=colors.get(mix, "k"), lw=1.6, ls="--")
        ax.axhline(base, ls="--", color="k", lw=1,
                   label="bidir +-2 count (clean ctx)" if scale == "s1" else None)
        ax.axhline(fair, ls=":", color="k", lw=1,
                   label="bidir +-2 count (corrupted ctx)" if scale == "s1" else None)
        ax.set_title("enwik8 %s" % scale)
        ax.set_xlabel("training tokens (M)")
        ax.set_yscale("log")
    axes[0].set_ylabel("masked-LM val loss (nats)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Scale ladder on enwik8 (100 MB): same skeleton, same SGD, only the mixer changes")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_ladder.png"), dpi=130)


def fig_lr():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = dict(spectral="tab:red", attn="tab:blue", stencil="tab:green",
                  fnet="tab:gray")
    data = {}
    for f in glob.glob(os.path.join(ROOT, "runs", "lr", "*.final.json")):
        with open(f) as fh:
            d = json.load(fh)
        m = d["meta"]
        mix, lr = m["mix"], m["args"]["lr"]
        last = [h for h in d["history"] if "val_loss" in h][-1]
        data.setdefault(mix, []).append((lr, last["val_loss"], last["val_acc"]))
    for mix, pts in data.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=mix,
                color=colors.get(mix, "k"))
    ax.axhline(1.0666, ls="--", color="k", lw=1, label="bidir +-2 count model")
    ax.axhline(3.5431, ls=":", color="k", lw=1, label="unigram")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (plain SGD + momentum)")
    ax.set_ylabel("val loss @ 2500 steps (nats)")
    ax.set_title("enwik8 s1 (d=96,L=3,T=128): learning-rate sensitivity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lr.png"), dpi=130)


def fig_kernel():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    styles = {"p4": "-", "p64": "-", "double": "--"}
    cols = {"p4": "tab:green", "p64": "tab:red", "double": "tab:blue"}
    for task in ("p4", "p64", "double"):
        fs = glob.glob(os.path.join(ROOT, "runs", "synth",
                                    "%s_spectral_*.final.json" % task))
        if not fs:
            continue
        d = json.load(open(fs[0]))
        kl = d.get("diagnostics", {}).get("kernel_lag")
        if not kl:
            continue
        T = d["meta"]["T"]
        L = len(kl)
        xs = list(range(0, min(T // 2, 80) + 1))
        ys = []
        for k in xs:
            if k == 0:
                ys.append(sum(lay.get("lag0", 0) for lay in kl) / L)
            else:
                ys.append(sum(lay.get("lag%d" % k, 0) for lay in kl) / L)
        ax.plot(xs, ys, styles[task], color=cols[task],
                label="%s (spectral)" % task)
    ax.set_yscale("log")
    ax.set_xlabel("|lag| of the learned circular mixing kernel")
    ax.set_ylabel("normalised kernel energy per lag (layer mean)")
    ax.set_title("Does the kernel track what the task requires?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_kernel.png"), dpi=130)


def parse_stage_a_log(path):
    """Per-step curve out of a Stage A stdout log.

    Stage A ran through 实验一's *unmodified* NumPy script, whose run `tag`
    does not contain the learning rate.  `attn lr0.2` and `attn lr0.8` therefore
    share `runs/main/attn_T96_d96_L3_s0.jsonl` and overwrite each other, as do
    the two `wo_zero` runs.  The stdout log is consequently the authoritative
    per-step record for Stage A, and this figure reads it instead of the
    collided jsonl files.
    """
    pat = re.compile(r"step\s+(\d+)(?:\s+\(final\))?\s+"
                     r"train\s+([-\d.eE+]+)\s+val\s+([-\d.eE+]+)\s+"
                     r"acc\s+([-\d.eE+]+)")
    rows = []
    for line in open(path, errors="ignore"):
        if "zero-shot" in line or "done in" in line:
            break
        m = pat.search(line)
        if m:
            rows.append(dict(step=int(m.group(1)),
                             train_loss=float(m.group(2)),
                             val_loss=float(m.group(3)),
                             val_acc=float(m.group(4))))
    return rows


def fig_repro():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    want = [("A_main_spectral_lr1.2_s0.log", "spectral lr1.2", "tab:red"),
            ("A_main_attn_lr0.2_s0.log", "attn lr0.2 (stable, ties spectral)", "tab:blue"),
            ("A_main_attn_lr0.8_s0.log", "attn lr0.8 (collapses)", "tab:cyan"),
            ("A_main_fnet_lr1.2_s0.log", "fnet lr1.2", "tab:gray")]
    for name, lab, col in want:
        path = os.path.join(ROOT, "logs", name)
        if not os.path.exists(path):
            print("  missing log:", path)
            continue
        rows = parse_stage_a_log(path)
        if not rows:
            print("  no rows parsed from", path)
            continue
        ax.plot([r["step"] for r in rows], [r["val_loss"] for r in rows],
                label=lab, color=col, lw=1.7)
    ax.axhline(1.1969, ls="--", color="k", lw=1, label="bidir +-2 count model")
    ax.axhline(3.3684, ls=":", color="k", lw=1, label="unigram")
    ax.set_xlabel("step")
    ax.set_ylabel("Tiny Shakespeare val loss (nats)")
    ax.set_title("Stage A: experiment 1 NumPy code (unmodified), reproduced here")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_repro.png"), dpi=130)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_repro, fig_lr, fig_ladder, fig_kernel):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:                # noqa: BLE001
            print("FAIL", fn.__name__, e)
