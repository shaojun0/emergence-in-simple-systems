#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump a (params, batch, loss, gradients) reference from the UPSTREAM NumPy
implementation, so the torch port can be checked against it exactly.

Usage:
  python dump_numpy_reference.py --mix spectral --T 24 --d 16 --L 2 --F 32 \
      --out D:/dsh/work/scaleup/ref/spectral_T24.npz
"""
import argparse
import json
import os
import sys

import numpy as np

UPSTREAM = r"D:\dsh\work\eiss\experiments\spectral-mlm-cpu"
sys.path.insert(0, UPSTREAM)
import spectral_bert as sb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--t_max", type=int, default=64)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--F", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--B", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos", default="sin")
    ap.add_argument("--wo_zero", action="store_true")
    ap.add_argument("--data", default=os.path.join(UPSTREAM, "tinyshakespeare.txt"))
    ap.add_argument("--updates", type=int, default=0,
                    help="run this many upstream SGD steps first; at 0 updates "
                         "upstream's buggy W1 gradient is invisible")
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    corp = sb.Corpus(a.data)
    sb.Globals.mask_id = corp.mask_id
    sb.Globals.V = corp.V
    cfg = dict(V=corp.V, d=a.d, F=a.F, L=a.L, T_train=a.T, T_max=a.t_max,
               mix=a.mix, heads=a.heads, pos=a.pos, seed=a.seed, wo_zero=a.wo_zero)
    m = sb.Model(cfg)
    rng = np.random.RandomState(4242)
    steps_total = a.updates + 1
    batches = [sb.make_batch(corp.train, rng, a.B, a.T) for _ in range(steps_total)]
    for s in range(a.updates):
        xs, mks, ys = batches[s]
        _, _, cs = m.forward(xs, mks, ys)
        m.backward(cs)
        m.step(sb.lr_at(s, steps_total, a.lr, a.warmup), 0.9, 1.0)
    x, mask, labels = batches[-1]
    loss, acc, c = m.forward(x, mask, labels)
    m.backward(c)

    meta = dict(V=corp.V, d=a.d, F=a.F, L=a.L, T=a.T, T_max=a.t_max, mix=a.mix,
                heads=a.heads, pos=a.pos, seed=a.seed, wo_zero=a.wo_zero,
                updates=a.updates, upstream_bug=True)
    out = dict(meta=json.dumps(meta), x=x, mask=mask, labels=labels,
               loss=np.float64(loss), acc=np.float64(acc))
    for k, v in m.P.items():
        out[k] = v
    for k, v in m.G.items():
        out["G_" + k] = v
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, **out)
    print("wrote %s | loss=%.12f acc=%.6f | params=%d" % (a.out, loss, acc, m.n_params()))


if __name__ == "__main__":
    main()
