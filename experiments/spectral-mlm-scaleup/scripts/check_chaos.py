#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control for check_trajectory: is numpy-vs-torch drift a porting bug, or just
float64 round-off amplified by the training dynamics?

Compares the divergence of
    (a) upstream NumPy vs the SAME NumPy code with one 1-ULP perturbation
    (b) upstream NumPy vs the torch port
over an identical batch sequence.  If (a) and (b) grow at the same rate, the
port is exact and the drift is intrinsic Lyapunov amplification.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = r"D:\dsh\work\eiss\experiments\spectral-mlm-cpu"
sys.path.insert(0, UPSTREAM)
import spectral_bert as sb        # noqa: E402


def train_np(cfg, batches, steps, lr, warmup, jitter=0.0):
    m = sb.Model(cfg)
    if jitter:
        m.P["WE"][0, 0] = np.nextafter(m.P["WE"][0, 0], np.inf)
    snap = []
    for s in range(steps):
        x, mk, y = batches[s]
        loss, acc, c = m.forward(x, mk, y)
        m.backward(c)
        m.step(sb.lr_at(s, steps, lr, warmup), 0.9, 1.0)
        if s % 5 == 0 or s == steps - 1:
            snap.append((s, {k: v.copy() for k, v in m.P.items()}))
    return m, snap


def main(mix="spectral", steps=150, T=24, d=16, L=2, F=32, heads=4, B=6, lr=0.4,
         warmup=20, seed=0):
    corp = sb.Corpus(os.path.join(UPSTREAM, "tinyshakespeare.txt"))
    sb.Globals.mask_id, sb.Globals.V = corp.mask_id, corp.V
    cfg = dict(V=corp.V, d=d, F=F, L=L, T_train=T, T_max=64, mix=mix,
               heads=heads, pos="sin", seed=seed)
    rng = np.random.RandomState(999)
    batches = [sb.make_batch(corp.train, rng, B, T) for _ in range(steps)]

    m0, snap0 = train_np(cfg, batches, steps, lr, warmup, jitter=0.0)
    m1, snap1 = train_np(cfg, batches, steps, lr, warmup, jitter=1.0)

    # torch port
    import torch
    import spectral_lm_torch as st
    model = st.Model(dict(cfg)).to("cuda").to(torch.float64)
    model.load_numpy_params(sb.Model(cfg).P)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.0)
    snap2 = []

    def tparam(k):
        return st_param(model, k)

    def maxdiff(a, get):
        return max(float(np.abs(v - get(k).detach().cpu().numpy()).max())
                   for k, v in a.items())

    with torch.no_grad():
        snap2.append((0, None))
    for s in range(steps):
        x, mk, y = batches[s]
        xt = torch.as_tensor(x, device="cuda")
        mt = torch.as_tensor(mk, device="cuda")
        yt = torch.as_tensor(y, device="cuda")
        opt.zero_grad(set_to_none=True)
        loss, _, _ = model(xt, mt, yt)
        loss.backward()
        st.clip_like_upstream(model.parameters(), 1.0)
        opt.param_groups[0]["lr"] = sb.lr_at(s, steps, lr, warmup)
        opt.step()
        if s % 5 == 0 or s == steps - 1:
            snap2.append((s, None))

    print("mix=%s  steps=%d lr=%s" % (mix, steps, lr))
    print(" step   numpy-vs-numpy(1ulp)   numpy-vs-torch")
    d1 = {s: p for s, p in snap1}
    d2 = {s: p for s, p in snap2}
    for s, ps in snap0:
        a = max(float(np.abs(ps[k] - d1[s][k]).max()) for k in ps)
        b = max(float(np.abs(ps[k] - tparam(k).detach().cpu().numpy()).max())
                for k in ps)
        print("%5d   %20.3e   %14.3e" % (s + 1, a, b))


def st_param(model, k):
    if k == "WE":
        return model.WE
    if k == "lnfg":
        return model.lnf.weight
    if k == "lnfb":
        return model.lnf.bias
    li, rest = k.split(".")
    blk = model.blocks[int(li)]
    if rest == "Gr":
        return blk.mix.Gr
    if rest == "Gi":
        return blk.mix.Gi
    if rest.startswith("W") and rest in ("Wq", "Wk", "Wv", "Wo"):
        return getattr(blk.mix, rest)
    if rest in ("bq", "bk", "bv", "bo"):
        return getattr(blk.mix, rest)
    return {"ln1g": blk.ln1.weight, "ln1b": blk.ln1.bias,
            "ln2g": blk.ln2.weight, "ln2b": blk.ln2.bias,
            "W1": blk.W1, "b1": blk.b1, "W2": blk.W2, "b2": blk.b2,
            "Wq": None}[rest]


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main(mix=sys.argv[1] if len(sys.argv) > 1 else "spectral")
