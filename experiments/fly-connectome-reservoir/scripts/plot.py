#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for experiment 3.

  fig_reservoir.png  val loss / accuracy of the fly connectome vs its controls,
                     against the experiment-1 counting references
  fig_wiring.png     what the controls actually preserve: degree distributions,
                     and the eigenvalue spectrum of the (radius-matched) wiring

Usage: python scripts/plot.py --results results/results.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")

COL = {"linear": "tab:blue", "mlp": "tab:orange",
       "fly": "tab:red", "degree_swap": "tab:blue", "row_weight_shuffle": "tab:green",
       "shuffle_weights": "tab:orange", "er": "tab:gray", "er_synth": "tab:gray"}


def _rows(res):
    """Accept either a raw run JSON ('results') or a summarize.py output ('summary')."""
    if "results" in res:
        return res["results"]
    out = []
    for key, s in res.get("summary", {}).items():
        g, fam = key.split("|")
        out.append(dict(graph=g, readout=fam, val_loss=s["loss_mean"],
                        val_acc=s["acc_mean"], loss_sd=s.get("loss_sd", 0.0),
                        acc_sd=s.get("acc_sd", 0.0)))
    return out


def fig_reservoir(res, out):
    r = _rows(res)
    fams = []
    for x in r:
        f = x.get("readout", "linear")
        if f not in fams:
            fams.append(f)
    names = []
    for x in r:
        if x["graph"] not in names:
            names.append(x["graph"])
    lut = {(x["graph"], x.get("readout", "linear")): x for x in r}
    ref = res["reference"]
    nf = max(1, len(fams))
    width = 0.8 / nf

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, key, ylab, title in ((axes[0], "val_loss", "val loss (nats)", "loss"),
                                 (axes[1], "val_acc", "val accuracy", "accuracy")):
        for fi, fam in enumerate(fams):
            xs, ys, es = [], [], []
            for gi, g in enumerate(names):
                d = lut.get((g, fam))
                if d is None:
                    continue
                xs.append(gi + fi * width - 0.4 + width / 2)
                ys.append(d[key])
                es.append(d.get("loss_sd" if key == "val_loss" else "acc_sd", 0.0))
            ax.bar(xs, ys, width=width * 0.92, color=COL.get(fam, "k"), label=fam,
                   yerr=es if any(es) else None, capsize=2,
                   error_kw=dict(lw=0.8, ecolor="k"))
        rkey = "loss" if key == "val_loss" else "acc"
        for lbl, ls in (("unigram", ":"), ("3-gram markov", "--"),
                        ("4-gram markov", "-"), ("bidir +-2 (clean ctx)", "-.")):
            if lbl in ref:
                ax.axhline(ref[lbl][rkey], ls=ls, color="k", lw=1, label=lbl)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title("%s: does the fly wiring help? (%s)" % (title, res.get("mode", "causal")))
        ax.legend(fontsize=6.5)

    fig.suptitle("Frozen connectome reservoir, only the readout is trained "
                 "(TinyShakespeare char-level %s)" % res.get("mode", "causal"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


def fig_wiring(res, out, subgraph=None, connectome=None):
    import scipy.sparse as sp
    sys.path.insert(0, HERE)
    import graphs as G

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    W = None
    if subgraph and os.path.exists(subgraph):
        W, _st = G.load_subgraph(subgraph)
    elif (connectome or res.get("connectome")) and os.path.exists(
            connectome or res.get("connectome")):
        W, _ids, _st = G.load_flywire(connectome or res["connectome"],
                                      n_keep=res["N"], signed=res.get("signed", True))
    if W is not None:
        graphs = [("fly", W),
                  ("degree_swap", G.degree_preserving_swap(W, seed=res.get("seed", 0))[0]),
                  ("er", G.er_graph(W.shape[0], W.nnz, seed=res.get("seed", 0) + 12345))]
        for nm, M in graphs:
            din = np.asarray((M != 0).sum(1)).ravel()
            dout = np.asarray((M != 0).sum(0)).ravel()
            for d, ls in ((din, "-"), (dout, "--")):
                v = np.bincount(d[d > 0])
                x = np.nonzero(v)[0]
                axes[0].plot(x, v[x] / v[x].sum(), ls, color=COL.get(nm, "k"),
                             label="%s %s" % (nm, "in" if ls == "-" else "out"))
            Mn, _ = G.rescale_to_radius(G.normalise(M, mode="row"), res.get("radius", 0.95))
            ev = np.linalg.eigvals(Mn.toarray() if sp.issparse(Mn) else Mn)
            axes[1].scatter(ev.real, ev.imag, s=3, alpha=0.4, color=COL.get(nm, "k"), label=nm)
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
        axes[0].set_xlabel("degree"); axes[0].set_ylabel("frequency")
        axes[0].set_title("degree distributions")
        axes[0].legend(fontsize=6, ncol=2)
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("Re"); axes[1].set_ylabel("Im")
        axes[1].set_title("eigenvalues at rho=%.2f" % res.get("radius", 0.95))
        axes[1].legend(fontsize=6)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "no subgraph available\n(pass --subgraph)", ha="center",
                    va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


def fig_memory(probe_paths, out):
    """Decoding accuracy for x_{t-k} from the reservoir state, one line per wiring."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    uni = None
    for p in probe_paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        uni = d.get("unigram_acc", uni)
        lags = d["lags"]
        for key, prof in d["profiles"].items():
            mem = prof["memory"]
            label = "%s (leak=%s, iscale=%s)" % (d["graph"], prof.get("leak"),
                                                 prof.get("input_scale"))
            ax.plot(lags, [mem[str(k)] for k in lags], "o-",
                    color=COL.get(d["graph"], "k"), label=label, lw=1.6, ms=4)
    if uni is not None:
        ax.axhline(uni, ls=":", color="k", lw=1,
                   label="trivial unigram (%.3f)" % uni)
    ax.set_xlabel("lag k  (the readout must recover x_{t-k} from h_t)")
    ax.set_ylabel("decoding accuracy")
    ax.set_title("What the reservoir still remembers\n"
                 "a 4-gram needs lags 1-3 to survive")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--subgraph", default=None)
    ap.add_argument("--connectome", default=None)
    ap.add_argument("--probes", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)
    with open(a.results, encoding="utf-8") as f:
        res = json.load(f)
    fig_reservoir(res, os.path.join(FIGDIR, "fig_reservoir.png"))
    fig_wiring(res, os.path.join(FIGDIR, "fig_wiring.png"),
               subgraph=a.subgraph, connectome=a.connectome)
    probes = a.probes or [os.path.join(os.path.dirname(HERE), "results", n)
                          for n in ("probe_fly.json", "probe_er.json")]
    fig_memory(probes, os.path.join(FIGDIR, "fig_memory.png"))
