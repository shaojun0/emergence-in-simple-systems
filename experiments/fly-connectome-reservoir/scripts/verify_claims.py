#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-derive every number quoted in the experiment-3 report from the raw results.

Same contract as experiment 2's verify_claims.py: the report's tables are
hand-written, this script recomputes each quoted value from
results/causal_summary.json (and from the subgraph sidecar) and fails loudly on
any mismatch.

    python scripts/verify_claims.py
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RESULTS = os.path.join(EXP, "results")
SUBGRAPH = os.path.join(RESULTS, "fly_subgraph_N1000.npz")

CLAIMS = []


def claim(label, got, want, tol=5e-4):
    ok = got is not None and want is not None and abs(got - want) <= tol
    CLAIMS.append((ok, label, got, want))


def claim_eq(label, got, want):
    CLAIMS.append((got == want, label, got, want))


# ---- quoted in the report: reference lines (verified in experiments 1 and 2) ----
REF = {
    "unigram": (3.3684, 0.1437),
    "2-gram markov": (2.4829, 0.2709),
    "3-gram markov": (2.0740, 0.3842),
    "4-gram markov": (1.8285, 0.4648),
    "bidir +-2 (clean ctx)": (1.1969, 0.7261),
}

# ---- quoted in the report: mean +/- sd over 3 seeds, N=1000, leak=0.97, iscale=10 ----
MEANS = {
    ("mlp", "fly"): (2.0720, 0.4254, 0.0122),
    ("mlp", "er"): (2.0279, 0.4358, 0.0263),
    ("mlp", "row_weight_shuffle"): (2.0271, 0.4406, 0.0144),
    ("mlp", "degree_swap"): (2.0748, 0.4217, 0.0132),
    ("mlp", "shuffle_weights"): (2.0795, 0.4126, 0.0173),
    ("linear", "fly"): (2.2965, 0.3496, 0.0115),
    ("linear", "er"): (2.2774, 0.3581, 0.0157),
    ("linear", "row_weight_shuffle"): (2.2809, 0.3546, 0.0251),
    ("linear", "degree_swap"): (2.3214, 0.3412, 0.0173),
    ("linear", "shuffle_weights"): (2.3455, 0.3347, 0.0264),
}

# ---- quoted in the report: the FlyWire top-1000 subgraph ----
SUBGRAPH_STATS = dict(n_nodes=1000, n_edges=53362, n_full=16847997,
                      synapses_full=54492922, mean_syn_per_edge=17.9733,
                      max_syn_per_edge=2405, frac_neg=0.54363)


def main():
    # ---------- reference lines ----------
    files = sorted(glob.glob(os.path.join(RESULTS, "causal_seed*.json")))
    if not files:
        print("MISSING results/causal_seed*.json -- run the experiment first")
        return 1
    with open(files[0], encoding="utf-8") as f:
        ref = json.load(f)["reference"]
    for k, (lo, ac) in REF.items():
        claim("reference %-22s loss" % k, ref.get(k, {}).get("loss"), lo, 1e-4)
        claim("reference %-22s acc" % k, ref.get(k, {}).get("acc"), ac, 1e-4)

    # ---------- per-graph means over seeds ----------
    summary = os.path.join(RESULTS, "causal_summary.json")
    if not os.path.exists(summary):
        print("MISSING results/causal_summary.json -- run scripts/summarize.py")
        return 1
    with open(summary, encoding="utf-8") as f:
        S = json.load(f)["summary"]
    for (fam, g), (lo, ac, sd) in MEANS.items():
        key = "%s|%s" % (g, fam)
        if key not in S:
            claim("summary row %s" % key, None, 1.0)
            continue
        s = S[key]
        claim("%-7s %-19s mean loss" % (fam, g), s["loss_mean"], lo, 1e-4)
        claim("%-7s %-19s mean acc " % (fam, g), s["acc_mean"], ac, 1e-4)
        claim("%-7s %-19s loss sd  " % (fam, g), s["loss_sd"], sd, 1e-4)
        claim_eq("%-7s %-19s n seeds" % (fam, g), s["n"], 3)

    # ---------- the headline comparison ----------
    for fam in ("mlp", "linear"):
        fly = S["fly|%s" % fam]["loss_mean"]
        er = S["er|%s" % fam]["loss_mean"]
        rws = S["row_weight_shuffle|%s" % fam]["loss_mean"]
        claim("%s: er beats fly by 0.0440" % fam, fly - er,
              0.0440 if fam == "mlp" else 0.0192, 1e-3)
        claim("%s: fly is worse than BOTH er and row_weight_shuffle" % fam,
              1.0 if (fly > er and fly > rws) else 0.0, 1.0, 0.0)

    # ---------- subgraph provenance ----------
    side = SUBGRAPH + ".json"
    if os.path.exists(side):
        with open(side, encoding="utf-8") as f:
            st = json.load(f)["stats"]
        for k, want in SUBGRAPH_STATS.items():
            claim("subgraph %-18s" % k, st.get(k), want, 1e-3)
    else:
        claim("subgraph sidecar exists", None, 1.0)

    # ---------- delegate the two independent checkers ----------
    for script, expect in (("verify_graphs.py", "VERIFY_GRAPHS: 23/23"),
                           ("test_load_flywire.py", "TEST_LOAD_FLYWIRE: 10/10")):
        try:
            out = subprocess.run([sys.executable, os.path.join(HERE, script),
                                  "--N", "400"] if script == "verify_graphs.py"
                                 else [sys.executable, os.path.join(HERE, script)],
                                 capture_output=True, text=True, timeout=1800)
            claim_eq("%s -> %s" % (script, expect), expect in out.stdout, True)
        except Exception as e:                                     # noqa: BLE001
            claim("%s ran" % script, str(e), "ok")

    bad = [c for c in CLAIMS if not c[0]]
    print("%-58s %16s %16s" % ("claim", "in result files", "quoted in report"))
    print("-" * 94)
    for ok, label, got, want in CLAIMS:
        fmt = lambda v: ("%.4f" % v) if isinstance(v, float) else str(v)
        print("%-58s %16s %16s %s" % (label[:58], fmt(got), fmt(want),
                                      "" if ok else "   <-- MISMATCH"))
    print("-" * 94)
    print("VERIFY: %d/%d claims match" % (len(CLAIMS) - len(bad), len(CLAIMS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
