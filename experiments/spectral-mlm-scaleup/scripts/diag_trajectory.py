#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose how the tiny (1e-14) per-step numerical difference between the
NumPy and torch implementations grows over a trajectory: linear accumulation
(would indicate a systematic bug) or exponential (float64 round-off chaos)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = r"D:\dsh\work\eiss\experiments\spectral-mlm-cpu"
sys.path.insert(0, UPSTREAM)
import spectral_bert as sb        # noqa: E402
import spectral_lm_torch as st    # noqa: E402


def main(mix="spectral", steps=80, T=24, d=16, L=2, F=32, heads=4, B=6, lr=0.4,
         warmup=20, seed=0):
    corp = sb.Corpus(os.path.join(UPSTREAM, "tinyshakespeare.txt"))
    sb.Globals.mask_id, sb.Globals.V = corp.mask_id, corp.V
    cfg = dict(V=corp.V, d=d, F=F, L=L, T_train=T, T_max=64, mix=mix,
               heads=heads, pos="sin", seed=seed)
    m = sb.Model(cfg)
    rng = np.random.RandomState(999)
    batches = [sb.make_batch(corp.train, rng, B, T) for _ in range(steps)]

    model = st.Model(dict(cfg)).to("cuda").to(torch.float64)
    model.load_numpy_params(sb.Model(cfg).P)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.0)

    def pdiff():
        with torch.no_grad():
            tot = 0.0
            for k, v in m.P.items():
                if k == "WE":
                    t = model.WE
                elif k.endswith(".Gr"):
                    t = model.blocks[int(k.split(".")[0])].mix.Gr
                elif k.endswith(".Gi"):
                    t = model.blocks[int(k.split(".")[0])].mix.Gi
                elif k == "lnfg":
                    t = model.lnf.weight
                elif k == "lnfb":
                    t = model.lnf.bias
                else:
                    li = int(k.split(".")[0]); rest = k.split(".")[1]
                    blk = model.blocks[li]
                    t = {"ln1g": blk.ln1.weight, "ln1b": blk.ln1.bias,
                         "ln2g": blk.ln2.weight, "ln2b": blk.ln2.bias,
                         "W1": blk.W1, "b1": blk.b1, "W2": blk.W2, "b2": blk.b2}[rest]
                tot = max(tot, float(np.abs(v - t.detach().cpu().numpy()).max()))
            return tot

    print("step  param_maxdiff   lossdiff     grad_maxdiff  gnorm_np   gnorm_th")
    for s in range(steps):
        x, mk, y = batches[s]
        loss_n, _, c = m.forward(x, mk, y)
        m.backward(c)

        xt = torch.as_tensor(x, device="cuda")
        mt = torch.as_tensor(mk, device="cuda")
        yt = torch.as_tensor(y, device="cuda")
        opt.zero_grad(set_to_none=True)
        loss_t, _, _ = model(xt, mt, yt)
        loss_t.backward()

        gdiff, gn_np, gn_th = 0.0, 0.0, 0.0
        with torch.no_grad():
            for k, v in m.G.items():
                pass
        # compare gradients through the parameter-name bridge
        th = {n: (model.WE if n == "WE" else None) for n in m.G}
        for n in m.G:
            t = _torch_param(model, n)
            d = float(np.abs(m.G[n] - t.grad.detach().cpu().numpy()).max())
            gdiff = max(gdiff, d)
            gn_np += float((m.G[n] ** 2).sum())
            gn_th += float(t.grad.detach().pow(2).sum())
        gn_np, gn_th = np.sqrt(gn_np), np.sqrt(gn_th)

        m.step(sb.lr_at(s, steps, lr, warmup), 0.9, 1.0)
        st.clip_like_upstream(model.parameters(), 1.0)
        opt.param_groups[0]["lr"] = sb.lr_at(s, steps, lr, warmup)
        opt.step()
        if s % 5 == 0 or s < 3:
            print("%4d  %.3e  %.3e  %.3e  %.10f  %.10f"
                  % (s, pdiff(), abs(float(loss_n) - float(loss_t.detach())),
                     gdiff, gn_np, gn_th))


def _torch_param(model, k):
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
    return {"ln1g": blk.ln1.weight, "ln1b": blk.ln1.bias,
            "ln2g": blk.ln2.weight, "ln2b": blk.ln2.bias,
            "W1": blk.W1, "b1": blk.b1, "W2": blk.W2, "b2": blk.b2}[rest]


if __name__ == "__main__":
    main(mix=sys.argv[1] if len(sys.argv) > 1 else "spectral")
