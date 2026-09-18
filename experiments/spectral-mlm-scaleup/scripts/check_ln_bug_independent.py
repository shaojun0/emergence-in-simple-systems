#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent, self-contained verification of experiment 1's W1 gradient bug.

Claim under test
----------------
`spectral_bert.py::_block_backward` computes the MLP first-layer weight gradient
from the LayerNorm activation *before* the gain/bias are applied:

    bb, ln2_c = ln_forward(h1, ln2g, ln2b)     # forward feeds bb = xh*g + b into W1
    f = bb @ W1 + b1
    ...
    G["%d.W1" % l] += cache[0].T @ df          # cache[0] is xh, i.e. PRE-gain

`ln_forward` returns `bb = xh * g + b` but caches `(xh, inv, g)`.  The correct
gradient is `bb.T @ df`; experiment 1 uses `xh.T @ df`.  At initialisation
`g = 1, b = 0`, so `bb == xh` and the two agree exactly -- which is why
`--gradcheck`, run only at the initial point, does not catch it.

Method (deliberately different from scripts/check_ln_bug.py)
-----------------------------------------------------------
Central finite differences on the *real* model, in two states:

  A. at initialisation (g=1, b=0)  -- what experiment 1's --gradcheck tests
  B. with the second LayerNorm perturbed away from (1, 0) -- i.e. after training

For each state we compare the numerical derivative against both the gradient
experiment 1 actually computes and the corrected formula.  In state B the
numeric/experiment-1 mismatch should be orders of magnitude above the
difference noise, while the corrected formula should sit at the noise floor.

Run:  python scripts/check_ln_bug_independent.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP1 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "spectral-mlm-cpu")
sys.path.insert(0, EXP1)
import spectral_bert as sb  # noqa: E402


def build(T=10, d=8, F=12, L=2, V=11, B=4, seed=0):
    np.random.seed(seed)
    cfg = dict(V=V, d=d, F=F, L=L, T_train=T, T_max=64, mix="spectral",
               heads=2, pos="sin", seed=seed)
    m = sb.Model(cfg)
    rng = np.random.RandomState(1)
    idx = rng.randint(0, V, size=(B, T))
    mask = rng.rand(B, T) < 0.4
    for b in range(B):
        mask[b, 0] = True
    labels = rng.randint(0, V, size=(B, T))
    return m, idx, mask, labels


def perturb_ln2(m, L, d, scale):
    """Push every block's second LayerNorm away from (g=1, b=0)."""
    for l in range(L):
        m.P["%d.ln2g" % l] = 1.0 + scale * np.random.RandomState(1000 + l).normal(0.0, 0.5, size=d)
        m.P["%d.ln2b" % l] = scale * np.random.RandomState(2000 + l).normal(0.0, 0.5, size=d)


def run(m, idx, mask, labels):
    """Forward+backward, capturing each block's ln2 cache and the ReLU adjoint df."""
    captured, dfs = [], []
    orig_block, orig_relu_b = m._block_backward, sb.relu_backward

    def wrapped_block(dh, l, c):
        captured.append((l, c["%d.ln2" % l]))
        return orig_block(dh, l, c)

    def wrapped_relu_b(dy, cache):
        r = orig_relu_b(dy, cache)
        dfs.append(r)
        return r

    m._block_backward = wrapped_block
    sb.relu_backward = wrapped_relu_b
    try:
        loss, _acc, c = m.forward(idx, mask, labels)
        m.backward(c)
    finally:
        m._block_backward = orig_block
        sb.relu_backward = orig_relu_b

    info = {}
    for i, (l, cache) in enumerate(captured):
        xh, _inv, g = cache
        bb = xh * g + m.P["%d.ln2b" % l]       # exactly what the forward pass fed to W1
        df = dfs[i]
        d = xh.shape[-1]
        buggy = m.G["%d.W1" % l].copy()
        correction = (bb - xh).reshape(-1, d).T @ df.reshape(-1, df.shape[-1])
        info[l] = dict(xh=xh, bb=bb, buggy=buggy, correct=buggy + correction)
    return loss, info


def finite_difference(m, idx, mask, labels, l, probes, eps=1e-5):
    out = []
    flat = m.P["%d.W1" % l].reshape(-1)
    for j in probes:
        orig = flat[j]
        flat[j] = orig + eps
        lp = m.forward(idx, mask, labels)[0]
        flat[j] = orig - eps
        lm = m.forward(idx, mask, labels)[0]
        flat[j] = orig
        out.append((j, (lp - lm) / (2 * eps)))
    return out


def report(title, scale, L=2, d=8, F=12):
    m, idx, mask, labels = build(L=L, d=d, F=F)
    if scale:
        perturb_ln2(m, L, d, scale)
    loss, info = run(m, idx, mask, labels)

    print("=" * 78)
    print("%s" % title)
    print("  ln2 perturbation scale = %.3g   masked-LM loss = %.6f" % (scale, loss))
    print("=" * 78)

    for l in range(L):
        xh, bb = info[l]["xh"], info[l]["bb"]
        print("\n  block %d:  max|bb - xh| = %.4e%s"
              % (l, float(np.abs(bb - xh).max()),
                 "   <- 0 exactly => bug invisible" if scale == 0 else ""))
        probes = np.random.RandomState(7 + l).choice(m.P["%d.W1" % l].size,
                                                     size=5, replace=False)
        fd = finite_difference(m, idx, mask, labels, l, probes)
        cbug = info[l]["buggy"].reshape(-1)
        ccor = info[l]["correct"].reshape(-1)

        print("    %-8s %-16s %-16s %-16s %-11s %-11s" %
              ("elem", "numerical", "analytic(exp1)", "analytic(fixed)",
               "|num-exp1|", "|num-fixed|"))
        eb = ec = 0.0
        for j, num in fd:
            eb += abs(num - cbug[j])
            ec += abs(num - ccor[j])
            print("    [%3d]    % -16.9e % -16.9e % -16.9e % -11.3e % -11.3e"
                  % (j, num, cbug[j], ccor[j], abs(num - cbug[j]), abs(num - ccor[j])))
        n = len(fd)
        print("    mean abs err:   experiment 1 = %.3e     fixed = %.3e"
              % (eb / n, ec / n))
    print()


def main():
    print("experiment 1 code under test:", EXP1)
    print("numpy", np.__version__)
    print()
    report("A. AT INITIALISATION (g=1, b=0) -- what --gradcheck tests", 0.0)
    report("B. AWAY FROM INITIALISATION (g!=1, b!=0) -- after training moves LN", 0.7)

    # Machine-checkable verdict on state B.
    m, idx, mask, labels = build()
    perturb_ln2(m, 2, 8, 0.7)
    _loss, info = run(m, idx, mask, labels)
    fd = finite_difference(m, idx, mask, labels, 0,
                           np.random.RandomState(7).choice(m.P["0.W1"].size, 5, False))
    eb = np.mean([abs(num - info[0]["buggy"].reshape(-1)[j]) for j, num in fd])
    ec = np.mean([abs(num - info[0]["correct"].reshape(-1)[j]) for j, num in fd])
    print("VERDICT: |num - experiment1| = %.3e   vs   |num - corrected| = %.3e"
          % (eb, ec))
    if eb > 100 * max(ec, 1e-14):
        print("BUG CONFIRMED: experiment 1's W1 gradient disagrees with finite "
              "differences; the corrected formula matches to the noise floor.")
        return 0
    print("NOT CONFIRMED: could not separate experiment 1's gradient from the "
          "corrected one.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
