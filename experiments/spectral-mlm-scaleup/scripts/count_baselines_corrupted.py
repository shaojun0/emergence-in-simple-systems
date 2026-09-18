#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The +-2 counting reference, evaluated on the SAME corrupted context the
neural models see.

Upstream's report notes the asymmetry itself: its counting baselines read the
*clean* context while the MLM sees 80/10/10-corrupted input.  Since the +-2
window model is the yardstick for "did the network learn anything beyond a
4-token local template?", the comparison has to be made on equal information.

Procedure: generate masked batches exactly as the model's data pipeline does,
then for every masked position use the *largest* fully-observed subset of its
four neighbours {t-2,t-1,t+1,t+2} and look the answer up in a table fitted on
that same context shape.  If no neighbour is clean, fall back to the unigram
distribution.  The result is an upper bound on what a pure local counter can do
with the model's information.
"""
import argparse
import itertools
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from count_baselines import enc_ctx, fit_table, load_ids, score, unigram  # noqa: E402


def make_batch(ids, rng, B, T, V, mask_id, mask_prob=0.15):
    n = len(ids)
    starts = rng.randint(0, n - T - 1, size=B)
    x = np.stack([ids[s:s + T] for s in starts])
    labels = x.copy()
    mask = rng.rand(B, T) < mask_prob
    for b in range(B):
        if not mask[b].any():
            mask[b, rng.randint(T)] = True
    r = rng.rand(B, T)
    x = x.copy()
    x[mask & (r < 0.8)] = mask_id
    rnd = mask & (r >= 0.8) & (r < 0.9)
    x[rnd] = rng.randint(0, V, size=int(rnd.sum()))
    return x, mask, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--train_chars", type=int, default=20_000_000)
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr, va, V = load_ids(a.data, a.encoding)
    mask_id = V - 1
    if a.train_chars:
        tr = tr[:a.train_chars]
    k = a.k
    offs = list(range(-k, 0)) + list(range(1, k + 1))
    up = unigram(tr, V)

    use = np.arange(k, len(tr) - k)
    tables = {}
    for r in range(len(offs), 0, -1):
        for sub in itertools.combinations(range(len(offs)), r):
            cols = [tr[use + offs[j]] for j in sub]
            tables[sub] = fit_table(enc_ctx(*cols, V=V), tr[use], V)
    print("fitted %d context tables over %d training positions" % (len(tables), len(use)))

    rng = np.random.RandomState(a.seed)
    n_all, l_sum, acc_sum = 0, 0.0, 0.0
    l_sub = {}
    for _ in range(a.batches):
        x, mask, labels = make_batch(va, rng, a.B, a.T, V, mask_id)
        bs, ts = np.where(mask)
        clean = np.zeros((len(bs), len(offs)), dtype=bool)
        for j, o in enumerate(offs):
            idx = ts + o
            ok = (idx >= 0) & (idx < a.T)
            v = x[bs, np.clip(idx, 0, a.T - 1)]
            clean[:, j] = ok & (v != mask_id)
        # group positions by their maximal clean subset
        key = np.array([tuple(np.nonzero(clean[i])[0]) for i in range(len(bs))],
                       dtype=object)
        tgt = labels[bs, ts]
        pos_ctx = x[bs[:, None],
                    np.clip(ts[:, None] + np.array(offs)[None, :], 0, a.T - 1)]
        for sub in list(tables):
            sel = np.array([tuple(s) == sub for s in key])
            if not sel.any():
                continue
            sub = tuple(sub)
            cols = [pos_ctx[sel, j] for j in sub]
            if sub:
                uk, cnt, uctx, ctot, best = tables[sub]
                l, ac, _ = score(uk, cnt, uctx, ctot, best, V, a.alpha,
                                 enc_ctx(*cols, V=V), tgt[sel], up)
            else:
                p = up[tgt[sel]]
                l = float(-np.log(np.maximum(p, 1e-12)).mean())
                ac = float((up.argmax() == tgt[sel]).mean())
            l_sum += l * sel.sum()
            acc_sum += ac * sel.sum()
            l_sub[sub] = l_sub.get(sub, [0, 0.0])
            l_sub[sub][0] += int(sel.sum())
            l_sub[sub][1] += l * int(sel.sum())
        n_all += len(bs)
    print("== +-%d count model on the model's OWN corrupted context ==" % k)
    print("   masked positions scored: %d" % n_all)
    print("   loss = %.4f   acc = %.4f" % (l_sum / n_all, acc_sum / n_all))
    print("   (clean-context +-%d reference is the number to beat)" % k)
    print("   breakdown by number of clean context tokens:")
    by_n = {}
    for sub, (n, ls) in l_sub.items():
        by_n.setdefault(len(sub), [0, 0.0])
        by_n[len(sub)][0] += n
        by_n[len(sub)][1] += ls
    for r in sorted(by_n, reverse=True):
        n, ls = by_n[r]
        print("     %d clean ctx: %7d positions (%.1f%%)  loss=%.4f"
              % (r, n, 100.0 * n / n_all, ls / max(1, n)))


if __name__ == "__main__":
    main()
