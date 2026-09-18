#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corpus, corruption and reference baselines for experiment 3.

Tokenisation, the train/val split and the 80/10/10 corruption deliberately
match experiment 1 (`experiments/spectral-mlm-cpu/spectral_bert.py`) so that
every number here is comparable to the experiment-1/2 tables.

The count references are imported from experiment 2's verified
`count_baselines.py` rather than re-derived, so there is exactly one
implementation of the counting model in the repository.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # .../fly-connectome-reservoir/scripts
EXPERIMENTS = os.path.dirname(os.path.dirname(HERE))       # .../experiments
REPO = os.path.dirname(EXPERIMENTS)                        # repo root
_EXP2 = os.path.join(EXPERIMENTS, "spectral-mlm-scaleup", "scripts")
if _EXP2 not in sys.path:
    sys.path.insert(0, _EXP2)
import count_baselines as cb  # noqa: E402


def load_corpus(path, val_frac=0.05):
    """Exactly experiment 1's Corpus: sorted unique chars, ids 0..V-2, mask id V-1."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n_val = max(1024, int(len(ids) * val_frac))
    return dict(ids=ids, train=ids[:-n_val], val=ids[-n_val:], vocab=chars,
                V=len(chars) + 1, mask_id=len(chars), stoi=stoi)


def apply_corruption(ids, rng, mask_id, V, mask_prob=0.15):
    """Experiment 1's 80/10/10 corruption, applied to a whole sequence.

    15% of positions are selected; of those, 80% become [MASK], 10% become a
    uniform random token in [0, V) and 10% are left alone. At least one position
    per sequence is guaranteed to be masked. Returns (corrupted ids, mask).
    """
    mask = rng.rand(len(ids)) < mask_prob
    if not mask.any():
        mask[rng.randint(len(ids))] = True
    out = ids.copy()
    r = rng.rand(len(ids))
    out[mask & (r < 0.8)] = mask_id
    rnd = mask & (r >= 0.8) & (r < 0.9)
    out[rnd] = rng.randint(0, V, size=int(rnd.sum()))
    return out, mask


def reference_baselines(tr, va, V, ks=(1, 2, 3, 4), eval_positions=200000,
                        alpha=0.1, seed=0):
    """The experiment-1/2 reference table, recomputed on this machine.

    Returns dict:  name -> (loss, acc, coverage)
    * "<k+1>-gram markov"  = DIRECTED, context = k preceding tokens (this is the
      "4-gram markov" row of the experiment-1 tables when k=3).
    * "bidir +-k (clean ctx)" = symmetric window, clean context (upper oracle).
    """
    rng = np.random.RandomState(seed)
    up = cb.unigram(tr, V)
    n = min(eval_positions, len(va))
    out = {}

    pos = rng.randint(1, len(va) - 1, size=n)
    tgt = va[pos]
    out["unigram"] = (float(-np.log(np.maximum(up[tgt], 1e-12)).mean()),
                      float((up.argmax() == tgt).mean()), 1.0)

    for k in ks:
        kk = max(1, k)
        use = np.arange(kk, len(tr))
        cx = cb.enc_ctx(*[tr[use - kk + i] for i in range(kk)], V=V)
        tab = cb.fit_table(cx, tr[use], V)
        p = np.arange(kk, len(va))
        ce = cb.enc_ctx(*[va[p - kk + i] for i in range(kk)], V=V)
        out["%d-gram markov" % (kk + 1)] = cb.score(*tab, V, alpha, ce, va[p], up)

    for k in ks:
        lo, hi = k, len(tr) - k
        idx = np.arange(lo, hi)
        cols = [tr[idx - k + i] for i in range(k)] + [tr[idx + 1 + i] for i in range(k)]
        tab = cb.fit_table(cb.enc_ctx(*cols, V=V), tr[idx], V)
        lo, hi = k, len(va) - k
        p = np.arange(lo, hi)
        if len(p) > n:
            p = rng.choice(p, size=n, replace=False)
        cols = [va[p - k + i] for i in range(k)] + [va[p + 1 + i] for i in range(k)]
        out["bidir +-%d (clean ctx)" % k] = cb.score(
            *tab, V, alpha, cb.enc_ctx(*cols, V=V), va[p], up)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 2, 3, 4])
    a = ap.parse_args()
    c = load_corpus(a.data)
    print("corpus: train=%d val=%d vocab=%d V=%d mask_id=%d"
          % (len(c["train"]), len(c["val"]), len(c["vocab"]), c["V"], c["mask_id"]))
    ref = reference_baselines(c["train"], c["val"], c["V"], ks=a.ks)
    print("%-28s %8s %8s %8s" % ("model", "loss", "acc", "cover"))
    for k, (l, ac, cov) in ref.items():
        print("%-28s %8.4f %8.4f %8.3f" % (k, l, ac, cov))
