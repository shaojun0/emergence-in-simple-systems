#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Localise the numpy-vs-torch mismatch: take the parameters AFTER one upstream
NumPy update, load them into the port, and compare forward + backward there.
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


def torch_param(model, k):
    if k == "WE":
        return model.WE
    if k == "lnfg":
        return model.lnf.weight
    if k == "lnfb":
        return model.lnf.bias
    li, rest = k.split(".")
    blk = model.blocks[int(li)]
    return {"ln1g": blk.ln1.weight, "ln1b": blk.ln1.bias,
            "ln2g": blk.ln2.weight, "ln2b": blk.ln2.bias,
            "W1": blk.W1, "b1": blk.b1, "W2": blk.W2, "b2": blk.b2,
            "Gr": blk.mix.Gr, "Gi": blk.mix.Gi,
            "Wq": getattr(blk.mix, "Wq", None), "Wk": getattr(blk.mix, "Wk", None),
            "Wv": getattr(blk.mix, "Wv", None), "Wo": getattr(blk.mix, "Wo", None),
            "bq": getattr(blk.mix, "bq", None), "bk": getattr(blk.mix, "bk", None),
            "bv": getattr(blk.mix, "bv", None), "bo": getattr(blk.mix, "bo", None)}[rest]


def main(mix="spectral", steps_to_take=1, T=24, d=16, L=2, F=32, heads=4, B=6,
         lr=0.4, warmup=20, seed=0):
    corp = sb.Corpus(os.path.join(UPSTREAM, "tinyshakespeare.txt"))
    sb.Globals.mask_id, sb.Globals.V = corp.mask_id, corp.V
    cfg = dict(V=corp.V, d=d, F=F, L=L, T_train=T, T_max=64, mix=mix,
               heads=heads, pos="sin", seed=seed)
    rng = np.random.RandomState(999)
    batches = [sb.make_batch(corp.train, rng, B, T) for _ in range(steps_to_take + 1)]
    m = sb.Model(cfg)
    for s in range(steps_to_take):
        x, mk, y = batches[s]
        _, _, c = m.forward(x, mk, y)
        m.backward(c)
        m.step(sb.lr_at(s, steps_to_take + 1, lr, warmup), 0.9, 1.0)

    x, mk, y = batches[steps_to_take]
    loss_n, _, c = m.forward(x, mk, y)
    m.backward(c)

    model = st.Model(dict(cfg)).to("cuda").to(torch.float64)
    model.load_numpy_params(m.P)
    xt = torch.as_tensor(x, device="cuda")
    mt = torch.as_tensor(mk, device="cuda")
    yt = torch.as_tensor(y, device="cuda")
    loss_t, _, logits = model(xt, mt, yt)
    with torch.no_grad():
        p = torch.softmax(logits[mt].double(), -1)
        pl = p.gather(1, yt[mt].unsqueeze(1)).squeeze(1)
        loss_eps = float(-torch.log(pl + 1e-9).mean())
    print("after %d numpy update(s): loss numpy=%.12f torch(-log(p+1e-9))=%.12f diff=%.3e"
          % (steps_to_take, loss_n, loss_eps, abs(loss_n - loss_eps)))
    model.zero_grad()
    loss_t.backward()
    rows = []
    for k in sorted(m.G):
        t = torch_param(model, k)
        if t is None:
            continue
        ana = m.G[k]
        got = t.grad.detach().cpu().numpy()
        dmax = float(np.abs(ana - got).max())
        scale = float(np.abs(ana).max()) + 1e-300
        rows.append((dmax, dmax / scale, k, scale))
    rows.sort(reverse=True)
    print("%-8s %12s %12s %12s" % ("param", "max|dg|", "rel", "max|g|"))
    for dmax, rel, k, scale in rows[:12]:
        print("%-8s %12.3e %12.3e %12.3e" % (k, dmax, rel, scale))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main(mix=sys.argv[1] if len(sys.argv) > 1 else "spectral",
         steps_to_take=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
