#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check that every control graph preserves exactly what it claims to preserve.

This is the "gradcheck" of experiment 3: if a control silently fails to preserve
the property it is supposed to hold fixed, the fly-vs-control comparison is
meaningless (it would be confounded with a change in dynamics). Run it before
trusting any result:

    python scripts/verify_graphs.py

Run against the real connectome as well:

    python scripts/verify_graphs.py --connectome <feather> --N 300
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graphs as G  # noqa: E402

CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append((bool(ok), label, detail))
    print("  %-58s %s %s" % (label, "PASS" if ok else "FAIL", detail))


def unweighted_deg(W):
    B = (W != 0)
    return np.asarray(B.sum(1)).ravel(), np.asarray(B.sum(0)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", default=None)
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.connectome:
        W, _ids, st = G.load_flywire(a.connectome, n_keep=a.N, signed=True)
        print("fly subgraph:", st)
    else:
        W = G.er_graph(a.N, 6 * a.N, seed=a.seed, signed=True)
        print("synthetic ER: n=%d nnz=%d" % (W.shape[0], W.nnz))

    n, m = W.shape[0], W.nnz
    ind_o, ind_i = unweighted_deg(W)
    print("\nbase: n=%d nnz=%d  unweighted out-deg max=%d  in-deg max=%d"
          % (n, m, ind_o.max(), ind_i.max()))

    # ---- degree_swap: unweighted degree sequences + weight multiset ----
    print("\ndegree_swap (double-edge swap)")
    Wd, done, want = G.degree_preserving_swap(W, seed=a.seed)
    do, di = unweighted_deg(Wd)
    check("edge count preserved", Wd.nnz == W.nnz, "%d vs %d" % (Wd.nnz, W.nnz))
    check("unweighted out-degree sequence preserved", np.array_equal(ind_o, do))
    check("unweighted in-degree sequence preserved", np.array_equal(ind_i, di))
    check("weight multiset preserved",
          np.allclose(np.sort(W.data), np.sort(Wd.data)))
    check("topology actually changed",
          not np.array_equal((W != 0).toarray(), (Wd != 0).toarray()))
    check("swap acceptance reasonable", done >= 0.5 * want, "%d/%d" % (done, want))

    # ---- row_weight_shuffle: exact row sums, topology untouched ----
    print("\nrow_weight_shuffle (permute incoming weights within each neuron)")
    Wr = G.row_weight_shuffle(W, seed=a.seed)
    check("edge count preserved", Wr.nnz == W.nnz, "%d vs %d" % (Wr.nnz, W.nnz))
    check("topology preserved EXACTLY",
          np.array_equal((W != 0).toarray(), (Wr != 0).toarray()))
    check("weighted in-degree (row sums) preserved EXACTLY",
          np.allclose(np.asarray(W.sum(1)).ravel(), np.asarray(Wr.sum(1)).ravel()))
    check("weight multiset preserved", np.allclose(np.sort(W.data), np.sort(Wr.data)))
    check("within-neuron weight assignment changed",
          not np.array_equal(W.data, Wr.data))

    # ---- shuffle_weights: topology kept, weights permuted ----
    print("\nshuffle_weights (permute weights on fixed topology)")
    Ww = G.shuffle_weights(W, seed=a.seed)
    check("topology preserved EXACTLY",
          np.array_equal((W != 0).toarray(), (Ww != 0).toarray()))
    check("weight multiset preserved", np.allclose(np.sort(W.data), np.sort(Ww.data)))
    check("weight-to-edge assignment changed",
          not np.allclose(W.data, Ww.data))
    check("row sums NOT preserved (unlike row_weight_shuffle)",
          not np.allclose(np.asarray(W.sum(1)).ravel(), np.asarray(Ww.sum(1)).ravel()))

    # ---- er: matched n and m only ----
    print("\ner (Erdos-Renyi, matched n and m)")
    We = G.er_graph(n, m, seed=a.seed + 12345, signed=True)
    check("node count preserved", We.shape[0] == n)
    check("edge count preserved", We.nnz == m, "%d vs %d" % (We.nnz, m))
    check("degree sequence NOT preserved (as intended)",
          not np.array_equal(ind_o, unweighted_deg(We)[0]))

    # ---- spectral radius normalisation ----
    print("\nspectral radius normalisation")
    for nm, M in (("fly", W), ("degree_swap", Wd), ("row_weight_shuffle", Wr),
                  ("shuffle_weights", Ww), ("er", We)):
        Mn = G.normalise(M, mode="row")
        Ms, r0 = G.rescale_to_radius(Mn, 0.95)
        r1 = G.spectral_radius(Ms)
        check("%-15s rho == 0.95 after rescale" % nm, abs(r1 - 0.95) < 1e-6,
              "rho_raw=%.4f rho=%.6f" % (r0, r1))

    bad = [c for c in CHECKS if not c[0]]
    print("\n" + "-" * 78)
    print("VERIFY_GRAPHS: %d/%d checks pass" % (len(CHECKS) - len(bad), len(CHECKS)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
