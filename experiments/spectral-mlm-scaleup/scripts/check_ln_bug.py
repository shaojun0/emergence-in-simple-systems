#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finite-difference proof of the upstream LayerNorm/MLP gradient bug.

At parameters taken after ONE upstream SGD step, perturb entries of W1 and
compare the numerical derivative of the loss against
  (a) upstream's hand-derived gradient  (uses the pre-gain LN activation)
  (b) the true gradient                 (uses the post-gain LN activation)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = r"D:\dsh\work\eiss\experiments\spectral-mlm-cpu"
sys.path.insert(0, UPSTREAM)
import spectral_bert as sb        # noqa: E402
import torch                      # noqa: E402
import spectral_lm_torch as st    # noqa: E402


def main(T=24, d=16, L=2, F=32, heads=4, B=6, nupd=1, lr=0.4, warmup=20, seed=0,
         eps=1e-5, n_probe=6):
    corp = sb.Corpus(os.path.join(UPSTREAM, "tinyshakespeare.txt"))
    sb.Globals.mask_id, sb.Globals.V = corp.mask_id, corp.V
    cfg = dict(V=corp.V, d=d, F=F, L=L, T_train=T, T_max=64, mix="spectral",
               heads=heads, pos="sin", seed=seed)
    rng = np.random.RandomState(999)
    batches = [sb.make_batch(corp.train, rng, B, T) for _ in range(nupd + 1)]
    m = sb.Model(cfg)
    for s in range(nupd):
        x, mk, y = batches[s]
        _, _, c = m.forward(x, mk, y)
        m.backward(c)
        m.step(sb.lr_at(s, nupd + 1, lr, warmup), 0.9, 1.0)
    x, mk, y = batches[nupd]
    loss, _, c = m.forward(x, mk, y)
    m.backward(c)

    tcfg = dict(cfg)
    tcfg["upstream_bug"] = True
    bug = st.Model(tcfg).to("cuda").to(torch.float64)
    bug.load_numpy_params(m.P)
    good = st.Model(dict(tcfg, upstream_bug=False)).to("cuda").to(torch.float64)
    good.load_numpy_params(m.P)
    xt = torch.as_tensor(x, device="cuda")
    mt = torch.as_tensor(mk, device="cuda")
    yt = torch.as_tensor(y, device="cuda")
    for mod in (bug, good):
        mod.zero_grad()
        l, _, _ = mod(xt, mt, yt)
        l.backward()

    print("reference loss=%.10f ; upstream G[0.ln2g]=%s"
          % (loss, np.array2string(np.round(m.P["0.ln2g"][:3], 6))))
    print("Upstream's W1 gradient equals xh^T@df (pre-gain), the truth is bb^T@df.\n")
    print("%-14s %14s %14s %14s %10s %10s"
          % ("entry", "numeric", "upstream", "correct", "err(up)", "err(cor)"))
    P = m.P["0.W1"]
    rng2 = np.random.RandomState(0)
    picks = [(int(i), int(j)) for i, j in
             zip(rng2.randint(0, d, n_probe), rng2.randint(0, F, n_probe))]
    eu, ec = [], []
    for i, j in picks:
        o = P[i, j]
        P[i, j] = o + eps
        lp = m.forward(x, mk, y)[0]
        P[i, j] = o - eps
        lm = m.forward(x, mk, y)[0]
        P[i, j] = o
        num = (lp - lm) / (2 * eps)
        up = m.G["0.W1"][i, j]
        co = good.blocks[0].W1.grad.cpu().numpy()[i, j]
        eu.append(abs(num - up))
        ec.append(abs(num - co))
        print("[%2d,%2d] %14.6e %14.6e %14.6e %10.2e %10.2e"
              % (i, j, num, up, co, abs(num - up), abs(num - co)))
    print("\nmean |numeric - upstream| = %.3e     mean |numeric - correct| = %.3e"
          % (np.mean(eu), np.mean(ec)))
    print("BUG CONFIRMED" if np.mean(eu) > 10 * np.mean(ec) else "no evidence of bug")
    # also verify the port's buggy gradient matches upstream's exactly
    print("port(upstream_bug).W1grad vs numpy G[0.W1]: max|diff| = %.3e"
          % np.abs(bug.blocks[0].W1.grad.cpu().numpy() - m.G["0.W1"]).max())
    print("port(correct).W1grad     vs numpy G[0.W1]: max|diff| = %.3e"
          % np.abs(good.blocks[0].W1.grad.cpu().numpy() - m.G["0.W1"]).max())


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
