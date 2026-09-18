#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the derived FlyWire subgraph so the experiment reproduces without 812 MB.

    python scripts/export_subgraph.py --connectome <feather> --N 1000

Writes results/fly_subgraph_N<N>.npz (+ a .json sidecar with provenance). The
sparse matrix is ~0.5 MB, so it IS committed, while the 812 MB feather is not.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graphs as G  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", required=True)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--signed", action="store_true", default=True)
    ap.add_argument("--unsigned", dest="signed", action="store_false")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    W, keep, st = G.load_flywire(a.connectome, n_keep=a.N, signed=a.signed)
    out = a.out or os.path.join(REPO, "experiments", "fly-connectome-reservoir",
                                "results", "fly_subgraph_N%d%s.npz"
                                % (a.N, "" if a.signed else "_unsigned"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    G.save_subgraph(out, W, keep, st)
    print("wrote %s  (%.1f KB)" % (out, os.path.getsize(out) / 1024))
    for k, v in st.items():
        print("   %-20s %s" % (k, v))
    print("   %-20s %.4f" % ("density", W.nnz / W.shape[0] ** 2))
    print("   %-20s %.3f / %.3f" % ("weight min/max", W.data.min(), W.data.max()))


if __name__ == "__main__":
    main()
