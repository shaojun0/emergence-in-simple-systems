#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge several run_experiment JSONs (one per seed) into a mean +/- sd table.

    python scripts/summarize.py --inputs results/causal_seed*.json --out results/causal_summary.json

Also prints the fly-vs-control differences, which is what experiment 3 is
actually about: the absolute loss only says whether the reservoir learned a
4-gram, the differences say whether the FLY WIRING mattered.
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default="fly")
    a = ap.parse_args()

    paths = []
    for pat in a.inputs:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print("no inputs matched")
        return 1

    rows, ref, meta = {}, {}, {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        ref = d.get("reference", ref)
        meta = dict(mode=d.get("mode"), seed=d.get("seed"), N=d.get("N"),
                    radius=d.get("radius"), leaks=d.get("leaks"),
                    input_scales=d.get("input_scales"))
        for r in d["results"]:
            rows.setdefault((r["graph"], r.get("readout", "linear")), []).append(r)

    print("mode=%s  N=%s  radius=%s  seeds=%d  files=%d"
          % (meta.get("mode"), meta.get("N"), meta.get("radius"), len(paths), len(paths)))
    print()
    hdr = "%-20s %-7s %4s %18s %18s" % ("graph", "readout", "n", "val loss", "val acc")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    graphs = []
    for (g, fam) in sorted(rows, key=lambda k: (k[1], k[0])):
        if g not in graphs:
            graphs.append(g)
        ls = [x["val_loss"] for x in rows[(g, fam)]]
        ac = [x["val_acc"] for x in rows[(g, fam)]]
        summary["%s|%s" % (g, fam)] = dict(
            n=len(ls), loss_mean=st.mean(ls), loss_sd=st.stdev(ls) if len(ls) > 1 else 0.0,
            acc_mean=st.mean(ac), acc_sd=st.stdev(ac) if len(ac) > 1 else 0.0)
        print("%-20s %-7s %4d   %.4f +/- %.4f   %.4f +/- %.4f"
              % (g, fam, len(ls), st.mean(ls), summary["%s|%s" % (g, fam)]["loss_sd"],
                 st.mean(ac), summary["%s|%s" % (g, fam)]["acc_sd"]))

    print()
    print("=== reference lines (TinyShakespeare, from the experiment-1/2 tables) ===")
    for k in ("unigram", "2-gram markov", "3-gram markov", "4-gram markov",
              "bidir +-2 (clean ctx)"):
        if k in ref:
            print("  %-24s loss %.4f  acc %.4f" % (k, ref[k]["loss"], ref[k]["acc"]))

    print()
    print("=== does the fly wiring beat its controls? (delta loss, + = fly worse) ===")
    for fam in sorted({f for (_g, f) in rows}):
        base = summary.get("%s|%s" % (a.baseline, fam))
        if base is None:
            continue
        print("  [%s]" % fam)
        for g in graphs:
            if g == a.baseline:
                continue
            o = summary.get("%s|%s" % (g, fam))
            if o is None:
                continue
            d = base["loss_mean"] - o["loss_mean"]
            pooled = (base["loss_sd"] ** 2 + o["loss_sd"] ** 2) ** 0.5
            verdict = "fly WORSE" if d > 0 else "fly better"
            within = "  (within noise)" if abs(d) < pooled else ""
            print("    fly - %-20s = %+0.4f   %s%s" % (g, d, verdict, within))

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(dict(meta=meta, reference=ref, summary=summary,
                           inputs=paths), f, ensure_ascii=False, indent=1)
        print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
