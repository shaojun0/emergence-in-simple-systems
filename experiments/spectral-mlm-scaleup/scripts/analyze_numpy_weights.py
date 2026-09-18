#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independently reproduce upstream's section 4.5 ("what did the mixer learn")
from the weights saved by the upstream NumPy code on this machine.

  python analyze_numpy_weights.py <weights.npz> [T]
"""
import os
import sys

import numpy as np

UPSTREAM = r"D:\dsh\work\eiss\experiments\spectral-mlm-cpu"
sys.path.insert(0, UPSTREAM)
import spectral_bert as sb        # noqa: E402


def main(path, T=96):
    z = np.load(path)
    layers = [k[: -len(".Gr")] for k in z.files if k.endswith(".Gr")]
    layers.sort(key=lambda s: int(s.split(".")[0]))
    print("weights: %s   (%d layers, T=%d)" % (os.path.basename(path), len(layers), T))
    print("%-6s %10s %10s %10s %10s %10s %10s %10s %10s"
          % ("layer", "mean|H|", "max|H|", "self_min", "self_med", "self_max",
             "lag0", "lag±1", ">8"))
    tot = {}
    for l in layers:
        Gr, Gi = z[l + ".Gr"], z[l + ".Gi"]
        H = sb.herm_filter(Gr, Gi, T)
        h = np.fft.ifft(H, axis=0).real                      # (T,d) circular kernel
        e = (h ** 2).sum(axis=1)
        e = e / e.sum()
        self_frac = (h[0] ** 2) / ((h ** 2).sum(axis=0) + 1e-30)
        lag1 = e[1] + e[T - 1]
        far = sum(e[k] + e[T - k] for k in range(9, T // 2 + 1))
        for name, val in (("lag0", e[0]), ("lag1", lag1), ("far", far)):
            tot[name] = tot.get(name, 0.0) + val
        print("%-6s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.6f"
              % (l, np.abs(H).mean(), np.abs(H).max(), self_frac.min(),
                 np.median(self_frac), self_frac.max(), e[0], lag1, far))
    n = len(layers)
    print("layer-mean: lag0=%.4f  lag±1=%.4f  lag±2=%.4f  >8=%.6f"
          % (tot["lag0"] / n, tot["lag1"] / n,
             sum(0 for _ in []) or _lag2(z, layers, T) / n, tot["far"] / n))


def _lag2(z, layers, T):
    s = 0.0
    for l in layers:
        H = sb.herm_filter(z[l + ".Gr"], z[l + ".Gi"], T)
        h = np.fft.ifft(H, axis=0).real
        e = (h ** 2).sum(axis=1)
        e = e / e.sum()
        s += e[2] + e[T - 2]
    return s


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 96)
