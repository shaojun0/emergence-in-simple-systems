#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic corpora for the long-range tests (upstream repo's open question Q1).

The upstream report's central empirical finding is that on Tiny Shakespeare the
learned spectral kernel collapses to a local template (lag 0: 93%, >8: 0.6%)
-- and it explicitly says this is a property of the *task*, not of the
operator ("判据 6").  These corpora make the task require long range instead,
so the same kernel readout becomes a real test.

  periodic :  x_t = b[t mod L], b re-randomised every SPAN characters.
              The only way to predict a masked token is to copy it from a
              fixed long lag (L).  A |lag|<=2 stencil cannot; a global
              circular filter can, exactly, with a spike at lag L.
              -> does the learned kernel extend to lag L?  (Q1)

  double   :  blocks "s + s" with a random block length n in [NMIN,NMAX].
              Predicting the second copy needs *content-based* matching
              (an induction head), i.e. a data-DEPENDENT route.
              A fixed convolution cannot do this at all.
              -> the fundamental limit of a pure spectral mixer (Q3/Q1)

Both are written as plain character files so the existing char-level pipeline
(train/eval/masking/kernel readout) is used unchanged.
"""
import argparse
import numpy as np

SYMS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"          # 32 payload symbols


def periodic(n_chars, L, span, seed):
    rng = np.random.RandomState(seed)
    out = []
    for start in range(0, n_chars, span):
        b = rng.randint(0, len(SYMS), size=L)
        m = min(span, n_chars - start)
        idx = (np.arange(start, start + m) % L)
        out.append("".join(SYMS[i] for i in b[idx]))
    return "".join(out)


def double(n_chars, nmin, nmax, seed):
    rng = np.random.RandomState(seed)
    out = []
    tot = 0
    while tot < n_chars:
        n = rng.randint(nmin, nmax + 1)
        s = [SYMS[i] for i in rng.randint(0, len(SYMS), size=n)]
        blk = "".join(s + s)
        out.append(blk)
        tot += len(blk)
    return "".join(out)[:n_chars]


def induct(n_chars, nmin, nmax, seed):
    """Variable-lag copy WITH an anchor: blocks are "s \\n s \\n" with a random
    block length n.  To predict a masked token in the second copy the model must
    locate the matching delimiter by content and read off the token at the same
    relative offset.  A data-independent convolution cannot do this (n varies);
    an induction head can."""
    rng = np.random.RandomState(seed)
    out, tot = [], 0
    while tot < n_chars:
        n = rng.randint(nmin, nmax + 1)
        s = "".join(SYMS[i] for i in rng.randint(0, len(SYMS), size=n))
        out.append(s + "\n" + s + "\n")
        tot += 2 * n + 2
    return "".join(out)[:n_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["periodic", "double", "induct"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chars", type=int, default=4_000_000)
    ap.add_argument("--L", type=int, default=64, help="period for --task periodic")
    ap.add_argument("--span", type=int, default=4096)
    ap.add_argument("--nmin", type=int, default=8)
    ap.add_argument("--nmax", type=int, default=56)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.task == "periodic":
        txt = periodic(a.chars, a.L, a.span, a.seed)
    elif a.task == "double":
        txt = double(a.chars, a.nmin, a.nmax, a.seed)
    else:
        txt = induct(a.chars, a.nmin, a.nmax, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print("wrote %s (%d chars, vocab %d, task=%s)"
          % (a.out, len(txt), len(set(txt)), a.task))


if __name__ == "__main__":
    main()
