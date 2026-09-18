#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline test of `load_flywire` against a synthetic feather with the real schema.

`scripts/verify_graphs.py --connectome ...` can only run once the 812 MB Zenodo
file has landed, so this test builds a tiny table with exactly the columns that
`proofread_connections_783.feather` has and checks the two things that are easy
to get silently wrong:

  1. ORIENTATION -- a synapse from neuron A (pre) to neuron B (post) must end up
     at W[B, A], not W[A, B].
  2. SIGN -- a GABA- or glutamate-dominant connection must become negative.

Run:  python scripts/test_load_flywire.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graphs as G  # noqa: E402


def build_fake(path):
    """Neurons 100..104; a known directed wiring with known transmitter calls."""
    rows = [
        # pre,  post, syn, gaba, ach,  glut
        (100, 200, 5, 0.01, 0.95, 0.02),   # 100 -> 200, cholinergic (excitatory)
        (200, 100, 3, 0.90, 0.05, 0.04),   # 200 -> 100, GABAergic   (inhibitory)
        (101, 200, 7, 0.02, 0.03, 0.90),   # 101 -> 200, glutamatergic(inhibitory)
        (102, 201, 9, 0.01, 0.97, 0.01),   # 102 -> 201, excitatory
        (103, 201, 2, 0.01, 0.96, 0.01),   # 103 -> 201, excitatory
        (104, 202, 4, 0.01, 0.94, 0.01),   # 104 -> 202, excitatory
        (100, 203, 6, 0.01, 0.93, 0.01),   # 100 -> 203, excitatory
    ]
    df = pd.DataFrame(rows, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count",
                                     "gaba_avg", "ach_avg", "glut_avg"])
    df["neuropil"] = "FAKE"
    df.to_feather(path)
    return df


def main():
    tmp = os.path.join(tempfile.gettempdir(), "fake_flywire.feather")
    build_fake(tmp)
    W, keep, st = G.load_flywire(tmp, n_keep=10, signed=True, weight="log")
    print("stats:", st)
    print("keep_ids:", list(keep))

    idx = {int(v): i for i, v in enumerate(keep)}
    checks = []

    def chk(label, ok, detail=""):
        checks.append(ok)
        print("  %-52s %s %s" % (label, "PASS" if ok else "FAIL", detail))

    Wd = W.toarray()
    a, b, c, d, e = (idx[100], idx[200], idx[101], idx[102], idx[104])

    chk("orientation: synapse 100->200 lands at W[post=200, pre=100]",
        np.isclose(Wd[b, a], np.log1p(5)),
        "W[200,100]=%.4f want %.4f" % (Wd[b, a], np.log1p(5)))
    chk("orientation: synapse 200->100 lands at W[post=100, pre=200]",
        np.isclose(Wd[a, b], -np.log1p(3)),
        "W[100,200]=%.4f want %.4f" % (Wd[a, b], -np.log1p(3)))
    chk("sign: GABA-dominant 200->100 is negative", Wd[a, b] < 0)
    chk("sign: glutamate-dominant 101->200 is negative", Wd[b, c] < 0)
    chk("sign: cholinergic 102->201 is positive", Wd[idx[201], d] > 0)
    chk("sign: cholinergic 104->202 is positive", Wd[idx[202], e] > 0)
    chk("no synapse where none exists (100->102 == 0)", Wd[d, a] == 0.0)
    chk("node count capped at n_keep", W.shape[0] <= 10, "N=%d" % W.shape[0])
    chk("edge count matches the table", W.nnz == 7, "nnz=%d" % W.nnz)

    # unsigned variant must equal |signed|
    Wu, _k, _s = G.load_flywire(tmp, n_keep=10, signed=False, weight="log")
    chk("unsigned == |signed|", np.allclose(np.abs(W.toarray()), Wu.toarray()))

    bad = checks.count(False)
    print("\nTEST_LOAD_FLYWIRE: %d/%d checks pass" % (len(checks) - bad, len(checks)))
    os.remove(tmp)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
