#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate every runs/**/*.final.json into comparison tables + a markdown
report fragment.  BPC = val_loss / ln 2 (bits per character) so the enwik8
numbers can be compared with the char-level literature, with the caveat that
this is a *masked* LM objective, not a causal one.
"""
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LN2 = math.log(2.0)


def active_mix(m):
    """Mixing parameters actually used at T_train (upstream's definition)."""
    if m.get("active_mix"):
        return m["active_mix"]
    mix, T, d, L = m.get("mix"), m.get("T"), m.get("d"), m.get("L")
    if not all([mix, T, d, L]):
        return m.get("mix_params")
    if mix == "spectral":
        return 2 * (T // 2 + 1) * d * L
    if mix == "fnet":
        return 0
    if mix == "attn":
        return 4 * d * d * L + 4 * d * L
    if mix == "stencil":
        return (2 * m.get("args", {}).get("stencil_k", 2) + 1) * d * L
    return m.get("mix_params")


def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        with open(f) as fh:
            d = json.load(fh)
        m = d.get("meta", {})
        hist = [h for h in d.get("history", []) if "val_loss" in h]
        if not hist:
            continue
        last = hist[-1]
        best = min(hist, key=lambda h: h["val_loss"])
        rows.append(dict(
            tag=m.get("tag", os.path.basename(f)), mix=m.get("mix"),
            T=m.get("T"), d=m.get("d"), L=m.get("L"), F=m.get("F"),
            params=m.get("params"), active_mix=active_mix(m),
            steps=m.get("steps"), lr=m.get("args", {}).get("lr"),
            seed=m.get("args", {}).get("seed"),
            tokens=m.get("tokens_per_step", 0) * m.get("steps", 0),
            corpus_mb=(m.get("corpus_bytes") or 0) / 1e6,
            seconds=m.get("seconds"),
            bug=m.get("args", {}).get("upstream_bug"),
            best=best["val_loss"], best_step=best["step"],
            final_val=last["val_loss"], final_acc=last["val_acc"],
            bpc=last["val_loss"] / LN2,
            curve=[h["val_loss"] for h in hist],
            extrap=d.get("extrapolation", {}),
            diag=d.get("diagnostics", {}),
            path=f))
    return rows


def lag_summary(diag):
    kl = diag.get("kernel_lag")
    if not kl:
        return "-"
    out = []
    for lay in kl:
        out.append("L%d lag0=%.3f ±1=%.3f >8=%.4f r90=%s"
                   % (lay["layer"], lay.get("lag0", 0),
                      lay.get("lag1", 0),
                      sum(v for k, v in lay.items()
                          if k.startswith("lag") and int(k[3:]) > 8),
                      lay.get("offdiag_r90")))
    return "; ".join(out)


def main(pattern=None, md=False):
    pats = sys.argv[1:] or [os.path.join(ROOT, "runs", "*", "*.final.json")]
    pats = [p for p in pats if not p.startswith("--")]
    rows = []
    for p in pats:
        rows.extend(load(p))
    if not rows:
        print("no runs found")
        return
    rows.sort(key=lambda r: (r["corpus_mb"], r["T"], r["mix"], r["lr"] or 0))
    hdr = ("%-44s %6s %4s %4s %4s %9s %8s %5s %8s %7s %7s %8s"
           % ("run", "T", "d", "L", "mix", "params", "act.mix", "lr",
              "val", "acc", "bpc", "minutes"))
    print(hdr)
    print("-" * len(hdr))
    lines = []
    for r in rows:
        print("%-44s %6d %4d %4d %4s %9s %8s %5s %8.4f %7.4f %8.4f %8.1f"
              % (r["tag"][:44], r["T"], r["d"], r["L"], str(r["mix"])[:4],
                 r["params"], r["active_mix"], ("%g" % r["lr"]) if r["lr"] else "-",
                 r["final_val"], r["final_acc"], r["bpc"],
                 (r["seconds"] or 0) / 60))
        lines.append(r)
    print()
    print("== kernel lag profile (spectral / gated) ==")
    for r in lines:
        s = lag_summary(r["diag"])
        if s != "-":
            print("  %-46s %s" % (r["tag"][:46], s))
    print()
    print("== attention geometry (mean attended distance / entropy / max weight) ==")
    for r in lines:
        a = r["diag"].get("attention")
        if a:
            print("  %-46s %s" % (r["tag"][:46], " | ".join(
                "L%d dist=%.1f H=%.2f maxw=%.3f" % (i, x["mean_distance"],
                                                    x["entropy"], x["max_weight"])
                for i, x in enumerate(a))))
    print()
    print("== zero-shot length extrapolation (val loss / acc) ==")
    for r in lines:
        if r["extrap"]:
            print("  %-46s %s" % (r["tag"][:46], "  ".join(
                "T=%s %.3f/%.3f" % (k, v["val_loss"], v["val_acc"])
                for k, v in sorted(r["extrap"].items(), key=lambda kv: int(kv[0])))))
    if md:
        out = os.path.join(ROOT, "RESULTS.md")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("# 自动汇总（aggregate.py）\n\n")
            fh.write("| run | corpus(MB) | T | d | L | mix | params | active mix | lr | "
                     "val | acc | BPC | min |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for r in lines:
                fh.write("| %s | %.1f | %d | %d | %d | %s | %s | %s | %s | %.4f | %.4f | "
                         "%.4f | %.1f |\n"
                         % (r["tag"], r["corpus_mb"], r["T"], r["d"], r["L"], r["mix"],
                            r["params"], r["active_mix"],
                            ("%g" % r["lr"]) if r["lr"] else "-",
                            r["final_val"], r["final_acc"], r["bpc"],
                            (r["seconds"] or 0) / 60))
            fh.write("\n## kernel lag profile\n\n")
            for r in lines:
                s = lag_summary(r["diag"])
                if s != "-":
                    fh.write("- **%s**: %s\n" % (r["tag"], s))
            fh.write("\n## length extrapolation\n\n")
            for r in lines:
                if r["extrap"]:
                    fh.write("- **%s**: %s\n" % (r["tag"], ", ".join(
                        "T=%s %.3f/%.3f" % (k, v["val_loss"], v["val_acc"])
                        for k, v in sorted(r["extrap"].items(), key=lambda kv: int(kv[0])))))
        print("\nwrote", out)


if __name__ == "__main__":
    main(md="--md" in sys.argv)
