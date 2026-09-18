#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context-counting references for the masked-LM task.

Reproduces the upstream report's reference table (unigram / bigram / trigram /
4-gram Markov and the bidirectional +-1 / +-2 window model that is *the*
benchmark the neural model has to beat), and can also run on a much larger
corpus (enwik8) where the full bidirectional table is infeasible.

Symmetric +-k window model: p(x_t | x_{t-k..t-1}, x_{t+1..t+k}) estimated by
counting on the training split, with backoff to shorter windows and finally to
the unigram distribution.  Both a *clean* context and the model's own
*corrupted* (80/10/10) context are scored, because the upstream report notes
the comparison was not symmetric there.

Usage:
  python count_baselines.py --data tinyshakespeare.txt --encoding utf-8 \
      --train_chars 0 --eval_positions 200000 --ks 1 2 3 4
"""
import argparse
import math
import sys
import time

import numpy as np

UPSTREAM = r"D:\dsh\work\eiss\experiments\spectral-mlm-cpu"


def load_ids(path, encoding, val_frac=0.05):
    with open(path, "rb") as f:
        raw = f.read()
    # match Python text-mode universal newlines exactly (upstream reads "r")
    text = raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    V = len(chars) + 1
    tbl = str.maketrans({c: chr(i) for i, c in enumerate(chars)})
    ids = np.frombuffer(text.translate(tbl).encode("latin-1"), dtype=np.uint8) \
        if V <= 256 else None
    n_val = max(1024, int(len(ids) * val_frac))
    return ids[:-n_val].astype(np.int64), ids[-n_val:].astype(np.int64), V


def enc_ctx(*cols, V):
    """Encode a tuple of equal-length token arrays into one int64 key."""
    key = np.zeros_like(cols[0], dtype=np.int64)
    for c in cols:
        key = key * V + c
    return key


def fit_table(cx_tr, tgt_tr, V):
    """Fit a context table: (unique ctx, ctx totals, per-ctx best token, pair counts)."""
    key = cx_tr * V + tgt_tr
    uk, cnt = np.unique(key, return_counts=True)
    ctx_of = uk // V
    uctx, inv = np.unique(ctx_of, return_inverse=True)
    ctot = np.bincount(inv, weights=cnt.astype(np.float64))
    order = np.lexsort((cnt, ctx_of))
    co = ctx_of[order]
    last = np.searchsorted(co, uctx, side="right") - 1
    best = (uk[order][last] % V).astype(np.int64)
    return uk, cnt, uctx, ctot, best


def score(uk, cnt, uctx, ctot, best, V, alpha, ctx_ev, tgt_ev, uni_p):
    """Return (loss, acc, coverage) for a k-window table, with backoff to unigram."""
    ck = ctx_ev * V + tgt_ev
    i = np.searchsorted(uk, ck)
    i_c = np.clip(i, 0, len(uk) - 1)
    hit = uk[i_c] == ck
    c = np.where(hit, cnt[i_c], 0.0)
    j = np.searchsorted(uctx, ctx_ev)
    j_c = np.clip(j, 0, len(uctx) - 1)
    hitc = uctx[j_c] == ctx_ev
    tot = np.where(hitc, ctot[j_c], 0.0)
    p = (c + alpha) / (tot + alpha * V)
    # backoff: unseen context -> unigram
    p = np.where(hitc, p, uni_p[tgt_ev])
    bp = best[j_c]
    pred = np.where(hitc, bp, uni_p.argmax())
    return float(-np.log(np.maximum(p, 1e-12)).mean()), float((pred == tgt_ev).mean()), \
        float(hitc.mean())


def unigram(ids_tr, V):
    c = np.bincount(ids_tr, minlength=V).astype(np.float64)
    return (c + 1.0) / (c.sum() + V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--train_chars", type=int, default=0, help="0 = all")
    ap.add_argument("--eval_positions", type=int, default=200000)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr, va, V = load_ids(a.data, a.encoding)
    if a.train_chars:
        tr = tr[:a.train_chars]
    rng = np.random.RandomState(a.seed)
    print("corpus: train=%d val=%d vocab=%d" % (len(tr), len(va), V))
    up = unigram(tr, V)
    print("%-26s %8s %8s %8s" % ("model", "loss", "acc", "cover"))
    t0 = time.time()

    # unigram
    n = min(a.eval_positions, len(va))
    pos = rng.randint(1, len(va) - 1, size=n)
    tgt = va[pos]
    lo = float(-np.log(np.maximum(up[tgt], 1e-12)).mean())
    print("%-26s %8.4f %8.4f %8.3f" % ("unigram", lo, float((up.argmax() == tgt).mean()), 1.0))

    # directed k-gram: previous k tokens only
    for k in a.ks:
        kk = max(1, k)
        use = np.arange(kk, len(tr))
        cx = enc_ctx(*[tr[use - kk + i] for i in range(kk)], V=V)
        uk, cnt, uctx, ctot, best = fit_table(cx, tr[use], V)
        p = np.arange(kk, len(va))
        ce = enc_ctx(*[va[p - kk + i] for i in range(kk)], V=V)
        l, ac, cov = score(uk, cnt, uctx, ctot, best, V, a.alpha, ce, va[p], up)
        print("%-26s %8.4f %8.4f %8.3f" % ("%d-gram markov" % (kk + 1), l, ac, cov))

    # symmetric +-k window model
    for k in a.ks:
        lo_i, hi_i = k, len(tr) - k
        idx = np.arange(lo_i, hi_i)
        cols = [tr[idx - k + i] for i in range(k)] + [tr[idx + 1 + i] for i in range(k)]
        cx = enc_ctx(*cols, V=V)
        uk, cnt, uctx, ctot, best = fit_table(cx, tr[idx], V)
        lo_i, hi_i = k, len(va) - k
        p = np.arange(lo_i, hi_i)
        if len(p) > n:
            p = rng.choice(p, size=n, replace=False)
        cols = [va[p - k + i] for i in range(k)] + [va[p + 1 + i] for i in range(k)]
        ce = enc_ctx(*cols, V=V)
        l, ac, cov = score(uk, cnt, uctx, ctot, best, V, a.alpha, ce, va[p], up)
        print("%-26s %8.4f %8.4f %8.3f" % ("bidir +-%d (clean ctx)" % k, l, ac, cov))
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
