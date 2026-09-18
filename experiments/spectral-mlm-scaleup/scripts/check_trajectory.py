#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage B2: end-to-end trajectory equivalence.

Gradient equality is necessary but not sufficient -- it says nothing about the
optimizer plumbing (momentum buffers, global-norm clipping, the warmup/decay
schedule, the data order).  This script runs the upstream NumPy trainer and the
torch port from the SAME initial weights over the SAME batch sequence in
float64 and compares the per-step training loss and the validation curve.

Any divergence means the port is not the same experiment.
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
UPSTREAM = r"D:\dsh\work\eiss\experiments\spectral-mlm-cpu"
sys.path.insert(0, UPSTREAM)
import spectral_bert as sb                                        # noqa: E402
import spectral_lm_torch as st                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(mix="spectral", steps=150, T=24, d=16, L=2, F=32, heads=4, B=6,
         lr=0.4, warmup=20, seed=0):
    corp = sb.Corpus(os.path.join(UPSTREAM, "tinyshakespeare.txt"))
    sb.Globals.mask_id, sb.Globals.V = corp.mask_id, corp.V
    cfg = dict(V=corp.V, d=d, F=F, L=L, T_train=T, T_max=64, mix=mix,
               heads=heads, pos="sin", seed=seed)
    m = sb.Model(cfg)
    rng = np.random.RandomState(999)
    batches = [sb.make_batch(corp.train, rng, B, T) for _ in range(steps)]
    val_batches = [sb.make_batch(corp.val, np.random.RandomState(5), B, T)
                   for _ in range(6)]

    # ---- upstream NumPy ----
    np_losses = []
    for s in range(steps):
        x, mk, y = batches[s]
        loss, acc, c = m.forward(x, mk, y)
        m.backward(c)
        m.step(sb.lr_at(s, steps, lr, warmup), 0.9, 1.0)
        np_losses.append(float(loss))
    np_val = []
    for x, mk, y in val_batches:
        l, a, _ = m.forward(x, mk, y)
        np_val.append((float(l), float(a)))

    # ---- torch port, same init, same batches ----
    tcfg = dict(cfg)
    tcfg["T_max"] = 64
    tcfg["upstream_bug"] = True          # replicate upstream's W1 gradient bug
    tcfg["loss_eps"] = 1e-9              # and its -log(p+1e-9) forward
    model = st.Model(tcfg).to("cuda").to(torch.float64)
    # upstream's Model(cfg) is deterministic given the seed, so a fresh model is
    # bit-identical to the initial state of `m`
    m0 = sb.Model(cfg)
    model.load_numpy_params(m0.P)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.0)
    th_losses = []
    for s in range(steps):
        x, mk, y = batches[s]
        xt = torch.as_tensor(x, device="cuda")
        mt = torch.as_tensor(mk, device="cuda")
        yt = torch.as_tensor(y, device="cuda")
        opt.zero_grad(set_to_none=True)
        loss, acc, _ = model(xt, mt, yt)
        loss.backward()
        st.clip_like_upstream(model.parameters(), 1.0)
        opt.param_groups[0]["lr"] = sb.lr_at(s, steps, lr, warmup)
        opt.step()
        th_losses.append(float(loss.detach()))
    th_val = []
    with torch.no_grad():
        for x, mk, y in val_batches:
            l, a, _ = model(torch.as_tensor(x, device="cuda"),
                            torch.as_tensor(mk, device="cuda"),
                            torch.as_tensor(y, device="cuda"))
            th_val.append((float(l), a))

    dl = np.abs(np.array(np_losses) - np.array(th_losses))
    dv = np.abs(np.array([v[0] for v in np_val]) - np.array([v[0] for v in th_val]))
    print("[%s] %d steps  max|d train_loss|=%.3e (final %.3e)  "
          "max|d val_loss|=%.3e  rel_val=%.3e"
          % (mix, steps, dl.max(), dl[-1], dv.max(), dv.max() / abs(np_val[0][0])))
    print("     numpy  train[0,1,last] = %.10f %.10f %.10f"
          % (np_losses[0], np_losses[1], np_losses[-1]))
    print("     torch  train[0,1,last] = %.10f %.10f %.10f"
          % (th_losses[0], th_losses[1], th_losses[-1]))
    print("     numpy  val = %s" % ["%.8f" % v[0] for v in np_val])
    print("     torch  val = %s" % ["%.8f" % v[0] for v in th_val])
    tol = 1e-9
    ok = dl.max() < tol and dv.max() < tol
    print("TRAJECTORY:", "PASS" if ok else "FAIL", "(tol %g)" % tol)
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    rc = 0
    for mix in ("spectral", "fnet", "attn"):
        rc |= main(mix=mix)
    raise SystemExit(rc)
