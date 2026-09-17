#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-parameter-group gradient diagnosis for spectral_bert.Model."""
import sys
import numpy as np

sys.path.insert(0, ".")
from spectral_bert import Model


def check(mix, L, d=4, F=6, V=7, T=6, B=2, seed=0, eps=1e-6, n_probe=4):
    cfg = dict(V=V, d=d, F=F, L=L, T_train=T, T_max=64, mix=mix, heads=2,
               pos="sin", seed=seed)
    m = Model(cfg)
    rng = np.random.RandomState(3)
    idx = rng.randint(0, V, size=(B, T))
    mask = np.zeros((B, T), dtype=bool)
    mask[:, 1] = True
    mask[0, 3] = True
    labels = rng.randint(0, V, size=(B, T))

    def L_():
        return m.forward(idx, mask, labels)[0]

    loss, acc, c = m.forward(idx, mask, labels)
    m.backward(c)

    rows = []
    for name in sorted(m.P):
        arr = m.P[name]
        flat = arr.reshape(-1)
        picks = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        worst = 0.0
        for j in picks:
            orig = flat[j]
            flat[j] = orig + eps
            lp = L_()
            flat[j] = orig - eps
            lm = L_()
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            ana = m.G[name].reshape(-1)[j]
            den = max(1e-9, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / den)
        rows.append((worst, name, float(arr.size)))
    print("=== mix=%s L=%d  loss=%.6f ===" % (mix, L, loss))
    for w, name, sz in sorted(rows, reverse=True):
        flag = "OK  " if w < 1e-5 else ("warn" if w < 1e-3 else "FAIL")
        print("  %s %-12s size=%-5d rel_err=%.3e" % (flag, name, sz, w))


if __name__ == "__main__":
    for mix in ["spectral", "attn", "fnet"]:
        check(mix, L=0)
    for mix in ["spectral", "attn", "fnet"]:
        check(mix, L=1)
