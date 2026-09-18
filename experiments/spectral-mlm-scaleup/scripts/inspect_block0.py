#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare block-0 intermediates (emb+PE, LN1, mixer output, LN2 output) and the
W1 gradient between NumPy and torch at parameters taken after N upstream updates.
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


def main(mix="spectral", nupd=1, T=24, d=16, L=2, F=32, heads=4, B=6, lr=0.4,
         warmup=20, seed=0):
    corp = sb.Corpus(os.path.join(UPSTREAM, "tinyshakespeare.txt"))
    sb.Globals.mask_id, sb.Globals.V = corp.mask_id, corp.V
    cfg = dict(V=corp.V, d=d, F=F, L=L, T_train=T, T_max=64, mix=mix,
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
    loss_n, _, c = m.forward(x, mk, y)
    m.backward(c)

    model = st.Model(dict(cfg)).to("cuda").to(torch.float64)
    model.load_numpy_params(m.P)
    cap = {}
    hs = [model.blocks[0].ln1.register_forward_hook(
              lambda mod, i, o: cap.__setitem__("ln1", o.detach())),
          model.blocks[0].ln2.register_forward_hook(
              lambda mod, i, o: cap.__setitem__("ln2", o.detach()))]
    xt = torch.as_tensor(x, device="cuda")
    mt = torch.as_tensor(mk, device="cuda")
    yt = torch.as_tensor(y, device="cuda")
    loss_t, _, _ = model(xt, mt, yt)
    for h in hs:
        h.remove()
    blk = model.blocks[0]
    with torch.no_grad():
        z_th = blk.mix(cap["ln1"]).cpu().numpy()
        h0_th = (model.WE[xt] + model.pos(T).unsqueeze(0)).cpu().numpy()
    model.zero_grad()
    loss_t.backward()

    h0_np = m.P["WE"][x] + sb.sincos_table(T, d)
    a_np = c["0.ln1"][0]
    a_th = cap["ln1"].cpu().numpy()
    xh_np = c["0.ln2"][0]
    xh_th = cap["ln2"].cpu().numpy()

    def rep(nm, a, b):
        print("%-12s max|diff| = %.3e   max|.| = %.3e" % (nm, np.abs(a - b).max(),
                                                          np.abs(a).max()))

    print("after %d numpy update(s), T=%d d=%d L=%d mix=%s" % (nupd, T, d, L, mix))
    print("loss diff    = %.3e" % abs(float(loss_n) - float(loss_t.detach())))
    with torch.no_grad():
        print("ln1g  numpy max|.|=%.6e torch max|.|=%.6e  max|diff|=%.3e"
              % (np.abs(m.P["0.ln1g"]).max(), float(blk.ln1.weight.abs().max()),
                 np.abs(m.P["0.ln1g"] - blk.ln1.weight.cpu().numpy()).max()))
        print("ln1b  numpy max|.|=%.6e torch max|.|=%.6e  max|diff|=%.3e"
              % (np.abs(m.P["0.ln1b"]).max(), float(blk.ln1.bias.abs().max()),
                 np.abs(m.P["0.ln1b"] - blk.ln1.bias.cpu().numpy()).max()))
        print("ln2g  numpy max|.|=%.6e torch max|.|=%.6e  max|diff|=%.3e"
              % (np.abs(m.P["0.ln2g"]).max(), float(blk.ln2.weight.abs().max()),
                 np.abs(m.P["0.ln2g"] - blk.ln2.weight.cpu().numpy()).max()))
    a_manual = ((h0_th - h0_th.mean(-1, keepdims=True))
                / np.sqrt(h0_th.var(-1, keepdims=True) + 1e-5)
                * m.P["0.ln1g"] + m.P["0.ln1b"])
    rep("ln1 manual", a_manual, a_th)
    rep("h0 emb+PE", h0_np, h0_th)
    rep("ln1(a)", a_np, a_th)
    rep("mixer z", c["0.spec"][2] * 0 + _np_mix(c, m, T, x, mk, y), z_th)
    rep("ln2(xh)", xh_np, xh_th)
    dd = np.abs(xh_np - xh_th)
    order = np.argsort(dd.ravel())[::-1][:6]
    print("  worst xh entries:")
    for flat in order:
        b, t, ch = np.unravel_index(flat, dd.shape)
        print("    (b=%d,t=%d,c=%d) numpy=%.10f torch=%.10f diff=%.3e"
              % (b, t, ch, xh_np[b, t, ch], xh_th[b, t, ch], dd[b, t, ch]))
    print("  per-channel max diff:", ["%.1e" % v for v in dd.max(axis=(0, 1))])
    print("  per-position max diff:", ["%.1e" % v for v in dd.max(axis=2).max(axis=0)])


def _np_mix(c, m, T, x, mk, y):
    """Recompute the numpy block-0 mixer output from its own cache."""
    a = c["0.ln1"][0]
    X = np.fft.fft(a, axis=1)
    Gf = sb.herm_filter(m.P["0.Gr"], m.P["0.Gi"], T)
    return np.fft.ifft(X * Gf, axis=1).real


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main(mix=sys.argv[1] if len(sys.argv) > 1 else "spectral",
         nupd=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
