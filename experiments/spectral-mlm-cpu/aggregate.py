#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate runs/*/*.final.json into a comparison table + markdown report."""
import glob
import json
import os
import sys

SPARK = "▁▂▃▄▅▆▇█"


def spark(vals):
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return SPARK[0] * len(vals)
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def active_mix_params(meta):
    """Mixing parameters actually *used* at the training length.

    The spectral arm pre-allocates filter rows for every frequency up to
    T_max//2+1 so that it can be evaluated on longer sequences; only the rows
    up to T_train//2+1 receive gradients.  Reporting the allocation instead of
    the active count would make the spectral arm look 8x more expensive than
    it is.
    """
    mix, T, d, L = meta.get("mix"), meta.get("T"), meta.get("d"), meta.get("L")
    if not all([mix, T, d, L]):
        return meta.get("mix_params")
    if mix == "spectral":
        return 2 * (T // 2 + 1) * d * L
    if mix == "fnet":
        return 0
    if mix == "attn":
        return 4 * d * d * L + 4 * d * L
    return meta.get("mix_params")


def jsonl_only_row(path):
    """A run that was stopped before it could write its .final.json."""
    hist = []
    for l in open(path):
        if not l.strip():
            continue
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if isinstance(d, dict) and "val_loss" in d:
            hist.append(d)
    if not hist:
        return None
    base = os.path.basename(path)[:-6]
    return dict(tag=base + " [stopped]", mix=base.split("_")[0], params=None,
                mix_params=None, active_mix=None, lr=None, seconds=None,
                best_val=min(h["val_loss"] for h in hist),
                final_val=hist[-1]["val_loss"], final_acc=hist[-1]["val_acc"],
                curve=[h["val_loss"] for h in hist], extrap={}, spectrum=None,
                cloze=[], path=path)


def main(pattern="runs/*/*.final.json"):
    rows = []
    have_final = set()
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            d = json.load(fh)
        meta = d.get("meta", {})
        have_final.add(os.path.join(os.path.dirname(f),
                                    meta.get("tag", "") + ".jsonl"))
        hist = [h for h in d.get("history", []) if isinstance(h, dict) and "val_loss" in h]
        if not hist:
            continue
        best = min(h["val_loss"] for h in hist)
        last = hist[-1]
        rows.append(dict(
            tag=meta.get("tag", os.path.basename(f)), mix=meta.get("mix"),
            params=meta.get("params"), mix_params=meta.get("mix_params"),
            active_mix=active_mix_params(meta),
            steps=meta.get("steps"), lr=meta.get("args", {}).get("lr"),
            seconds=meta.get("seconds"), best_val=best,
            final_val=last["val_loss"], final_acc=last["val_acc"],
            curve=[h["val_loss"] for h in hist],
            extrap=d.get("extrapolation", {}), spectrum=d.get("spectrum"),
            cloze=d.get("cloze", []), path=f))
    for j in sorted(glob.glob(os.path.join(os.path.dirname(pattern), "*.jsonl"))):
        if j in have_final:
            continue
        r = jsonl_only_row(j)
        if r:
            rows.append(r)
    if not rows:
        print("no runs found for", pattern)
        return
    rows.sort(key=lambda r: (r["mix"], r["best_val"]))
    print("%-38s %9s %11s %6s %7s %7s  %s" %
          ("run", "params", "mix(active)", "lr", "bestval", "valacc", "val-loss curve"))
    print("-" * 112)
    for r in rows:
        print("%-38s %9s %11s %6s %7.4f %7.4f  %s" %
              (r["tag"], r["params"] if r["params"] else "-",
               r["active_mix"] if r["active_mix"] is not None else "-",
               ("%.2f" % r["lr"]) if r["lr"] else "-", r["best_val"],
               r["final_acc"], spark(r["curve"])))
    print()
    for r in rows:
        if r["extrap"]:
            ext = "  ".join("T=%s:%.3f/%.3f" % (k, v["val_loss"], v["val_acc"])
                            for k, v in sorted(r["extrap"].items(), key=lambda kv: int(kv[0])))
            print("extrapolation %-30s %s" % (r["tag"], ext))
    if "--md" in sys.argv:
        with open("REPORT.md", "w") as fh:
            fh.write("# spectral vs attention vs FNet -- masked-LM on CPU (NumPy only)\n\n")
            fh.write("| run | params | mixing params (active) | lr | best val | final val acc |\n")
            fh.write("|---|---|---|---|---|---|\n")
            for r in rows:
                fh.write("| %s | %s | %s | %s | %.4f | %.4f |\n" %
                         (r["tag"], r["params"] if r["params"] else "-",
                          r["active_mix"] if r["active_mix"] is not None else "-",
                          ("%.2f" % r["lr"]) if r["lr"] else "-",
                          r["best_val"], r["final_acc"]))
            fh.write("\n## zero-shot length extrapolation (val loss / acc)\n\n")
            for r in rows:
                if r["extrap"]:
                    fh.write("- **%s**: " % r["tag"])
                    fh.write(", ".join("T=%s %.3f/%.3f" % (k, v["val_loss"], v["val_acc"])
                                       for k, v in sorted(r["extrap"].items(),
                                                          key=lambda kv: int(kv[0]))))
                    fh.write("\n")
        print("wrote REPORT.md")


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("--")] or ["runs/*/*.final.json"])
