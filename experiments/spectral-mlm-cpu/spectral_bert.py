#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_bert.py -- a BERT-style masked-language model (MLM) trained on CPU
using nothing but NumPy.

Design constraints ("不使用现代深度学习那一套"):
  * no PyTorch / JAX / TensorFlow / autograd -- every gradient is hand-derived
  * no GPU, no CUDA, no mixed precision
  * no Adam/AdamW -- plain SGD + momentum (+ global-norm clipping)
  * no dropout, no weight decay, no EMA
  * the only imports are argparse/json/math/os/time/numpy

The token-mixing sublayer is the experimental variable (--mix):
  fnet     : unparameterized real part of the FFT   (Lee-Thorp et al., NAACL 2022)
  spectral : learnable per-frequency complex filter H(k), i.e. a global
             convolution implemented via FFT (FNO / GFNet / AFNO family)
  attn     : hand-written multi-head self-attention (control)

All three arms share the same skeleton: sinusoidal positions, pre-LN,
residuals, GELU-free ReLU MLP, tied input/output embeddings, identical
optimizer, identical data order.  The only thing that differs is how tokens
talk to each other.

Use --gradcheck to verify every hand-derived backward pass against central
finite differences.
"""

import argparse
import json
import math
import os
import time

import numpy as np

# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def ln_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    return xh * g + b, (xh, inv, g)


def ln_backward(dy, cache):
    xh, inv, g = cache
    dyg = dy * g
    dx = inv * (dyg - dyg.mean(-1, keepdims=True)
                - xh * (dyg * xh).mean(-1, keepdims=True))
    axes = tuple(range(dy.ndim - 1))
    return dx, (dy * xh).sum(axis=axes), dy.sum(axis=axes)


def relu_forward(x):
    return np.maximum(x, 0.0), (x > 0.0)


def relu_backward(dy, cache):
    return dy * cache


def sincos_table(T, d):
    """Parameter-free sinusoidal positions (Vaswani et al. 2017).

    Kept parameter-free on purpose: it lets us evaluate the same weights on
    sequence lengths never seen in training (the FNO 'resolution independence'
    claim, tested on language).
    """
    pos = np.arange(T)[:, None].astype(np.float64)
    i = np.arange(0, d, 2).astype(np.float64)
    div = np.exp(-math.log(10000.0) * i / d)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


# --------------------------------------------------------------------------
# spectral filtering helpers (Hermitian-symmetric filter for real signals)
# --------------------------------------------------------------------------


def herm_filter(Gr, Gi, T):
    """Build the full length-T Hermitian filter from the m=T//2+1 free taps."""
    m = T // 2 + 1
    Gf = np.empty((T, Gr.shape[1]), dtype=np.complex128)
    Gf[:m] = Gr[:m] + 1j * Gi[:m]
    for k in range(m, T):
        Gf[k] = np.conj(Gf[T - k])
    if T % 2 == 0:
        Gf[m - 1] = Gf[m - 1].real      # Nyquist must be real
    return Gf


def herm_filter_backward(dGf, T, m_max):
    """Adjoint of herm_filter: full (T,d) cotangent -> (m_max,d) cotangent."""
    m = T // 2 + 1
    dGr = dGf.real[:m].copy()
    dGi = dGf.imag[:m].copy()
    for k in range(m, T):
        j = T - k
        dGr[j] += dGf.real[k]
        dGi[j] -= dGf.imag[k]
    if T % 2 == 0:
        dGi[m - 1] = 0.0
    if m < m_max:                        # pad the unused high frequencies
        dGr = np.concatenate([dGr, np.zeros((m_max - m, dGr.shape[1]))], 0)
        dGi = np.concatenate([dGi, np.zeros((m_max - m, dGi.shape[1]))], 0)
    return dGr, dGi


def fnet_forward(a):
    """FNet token mixing: y = Re(FFT(x)). Fixed, parameter-free, all-to-all."""
    return np.fft.fft(a, axis=1).real


def fnet_backward(dz, T):
    """Adjoint of a -> Re(FFT(a)).

    X = FFT(a) has no 1/T factor while IFFT does, so the adjoint picks the
    factor T back up:  d/da = T * Re(IFFT(dz)).
    """
    return T * np.fft.ifft(dz.astype(np.complex128), axis=1).real


def spectral_forward(a, Gr, Gi, T):
    """Learnable global convolution: y = Re(IFFT(H . FFT(x))).

    H is a per-frequency, per-channel complex gain (diagonal in frequency
    domain == a bank of circulant/global convolution kernels in time domain).
    """
    X = np.fft.fft(a, axis=1)
    Gf = herm_filter(Gr, Gi, T)
    Y = X * Gf
    z = np.fft.ifft(Y, axis=1).real
    return z, (X, Gf, T)


def spectral_backward(dz, cache, m_max):
    """Hand-derived adjoint of spectral_forward (see derivation below).

    With L real and z = Re(IFFT(Y)):
        dY = FFT(dz) / T                     (real/imag parts are the cotangents)
        dX = dY * conj(H)
        dx = T * Re(IFFT(dX))                (IFFT carries a 1/T that FFT lacks)
        dH = sum_batch dY * conj(X)
    """
    X, Gf, T = cache
    dY = np.fft.fft(dz, axis=1) / T
    da = T * np.fft.ifft(dY * np.conj(Gf), axis=1).real
    dGf = (dY * np.conj(X)).sum(axis=0)
    dGr, dGi = herm_filter_backward(dGf, T, m_max)
    return da, dGr, dGi


# --------------------------------------------------------------------------
# hand-written multi-head self-attention (control arm)
# --------------------------------------------------------------------------


def attn_forward(a, Wq, bq, Wk, bk, Wv, bv, Wo, bo, heads):
    B, T, d = a.shape
    dh = d // heads
    Q = a @ Wq + bq
    K = a @ Wk + bk
    V = a @ Wv + bv
    sh = lambda x: x.reshape(B, T, heads, dh).transpose(0, 2, 1, 3)
    Qh, Kh, Vh = sh(Q), sh(K), sh(V)
    scale = 1.0 / math.sqrt(dh)
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) * scale
    A = softmax(scores, -1)
    ctx = A @ Vh
    ctx_m = ctx.transpose(0, 2, 1, 3).reshape(B, T, d)
    out = ctx_m @ Wo + bo
    return out, (a, Qh, Kh, Vh, A, ctx_m, scale,
                 (B, T, d, heads, dh), Wq, Wk, Wv, Wo)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


class Model(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg["seed"])
        self.P = {}
        self.G = {}
        self.M = {}          # momentum buffers
        self._init_params()
        self._pe_cache = {}

    # ---------------- parameter bookkeeping ----------------
    def _glorot(self, shape):
        fan_in, fan_out = shape[0], shape[1]
        lim = math.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-lim, lim, size=shape)

    def _init_params(self):
        cfg = self.cfg
        V, d, L, F, T = cfg["V"], cfg["d"], cfg["L"], cfg["F"], cfg["T_train"]
        m_max = cfg["T_max"] // 2 + 1
        P = self.P
        P["WE"] = self.rng.normal(0.0, 0.02, size=(V, d))
        if cfg["pos"] == "learned":
            P["P"] = self.rng.normal(0.0, 0.02, size=(cfg["T_max"], d))
        for l in range(L):
            P["%d.ln1g" % l] = np.ones(d)
            P["%d.ln1b" % l] = np.zeros(d)
            P["%d.ln2g" % l] = np.ones(d)
            P["%d.ln2b" % l] = np.zeros(d)
            if cfg["mix"] == "attn":
                for nm in ("Wq", "Wk", "Wv", "Wo"):
                    P["%d.%s" % (l, nm)] = self._glorot((d, d))
                    P["%d.b%s" % (l, nm[1])] = np.zeros(d)
                if cfg.get("wo_zero"):
                    # Classic zero-init of the residual branch output (GPT-2
                    # style): the block starts as a pure identity, so the
                    # question "is attention's plain-SGD failure just a bad
                    # initial basin?" can be tested directly.
                    P["%d.Wo" % l] = np.zeros((d, d))
                    P["%d.bo" % l] = np.zeros(d)
            elif cfg["mix"] == "spectral":
                # start as the identity filter: the spectral arm begins life
                # exactly as a "do-nothing" mixer and must learn to deviate.
                P["%d.Gr" % l] = np.ones((m_max, d))
                P["%d.Gi" % l] = np.zeros((m_max, d))
            P["%d.W1" % l] = self._glorot((d, F))
            P["%d.b1" % l] = np.zeros(F)
            P["%d.W2" % l] = self._glorot((F, d))
            P["%d.b2" % l] = np.zeros(d)
        P["lnfg"] = np.ones(d)
        P["lnfb"] = np.zeros(d)
        for k in P:
            self.G[k] = np.zeros_like(P[k])
            self.M[k] = np.zeros_like(P[k])

    def n_params(self):
        return int(sum(v.size for v in self.P.values()))

    def n_mix_params(self):
        return int(sum(v.size for k, v in self.P.items()
                       if (".Gr" in k or ".Gi" in k or ".Wq" in k or ".Wk" in k
                           or ".Wv" in k or ".Wo" in k
                           or k.endswith(".bq") or k.endswith(".bk")
                           or k.endswith(".bv") or k.endswith(".bo"))))

    def pos_encoding(self, T):
        if "P" in self.P:
            return self.P["P"][:T]
        key = T
        if key not in self._pe_cache:
            self._pe_cache[key] = sincos_table(T, self.cfg["d"])
        return self._pe_cache[key]

    # ---------------- forward ----------------
    def forward(self, idx, mask, labels):
        cfg = self.cfg
        B, T = idx.shape
        L, d = cfg["L"], cfg["d"]
        c = {"idx": idx, "mask": mask, "T": T}
        h = self.P["WE"][idx] + self.pos_encoding(T)[None, :, :]
        for l in range(L):
            h = self._block_forward(h, l, T, c)
        hf, lnf_c = ln_forward(h, self.P["lnfg"], self.P["lnfb"])
        c["lnf"] = lnf_c
        c["hf"] = hf
        logits = hf @ self.P["WE"].T                       # tied embeddings
        lg = logits[mask]
        lab = labels[mask]
        p = softmax(lg, -1)
        loss = -np.log(p[np.arange(lab.size), lab] + 1e-9).mean()
        acc = float((p.argmax(-1) == lab).mean())
        c["probs"], c["lab"] = p, lab
        c["mask_bs"], c["mask_ts"] = np.where(mask)
        return loss, acc, c

    def _block_forward(self, h, l, T, c):
        mix = self.cfg["mix"]
        pre = "%d." % l
        a, ln1_c = ln_forward(h, self.P[pre + "ln1g"], self.P[pre + "ln1b"])
        c[pre + "ln1"] = ln1_c
        if mix == "attn":
            z, ac = attn_forward(
                a, self.P[pre + "Wq"], self.P[pre + "bq"],
                self.P[pre + "Wk"], self.P[pre + "bk"],
                self.P[pre + "Wv"], self.P[pre + "bv"],
                self.P[pre + "Wo"], self.P[pre + "bo"], self.cfg["heads"])
            c[pre + "attn"] = ac
        elif mix == "fnet":
            z = fnet_forward(a)
        else:
            z, mc = spectral_forward(a, self.P[pre + "Gr"], self.P[pre + "Gi"], T)
            c[pre + "spec"] = mc
        h1 = h + z
        c[pre + "h1"] = h1
        bb, ln2_c = ln_forward(h1, self.P[pre + "ln2g"], self.P[pre + "ln2b"])
        c[pre + "ln2"] = ln2_c
        f = bb @ self.P[pre + "W1"] + self.P[pre + "b1"]
        rel, rel_c = relu_forward(f)
        c[pre + "rel"] = rel          # activations, needed for dW2
        c[pre + "relmask"] = rel_c    # 0/1 mask, needed for the ReLU adjoint
        o = rel @ self.P[pre + "W2"] + self.P[pre + "b2"]
        return h1 + o

    # ---------------- backward ----------------
    def backward(self, c):
        cfg = self.cfg
        B, T = c["idx"].shape
        L, d, V = cfg["L"], cfg["d"], cfg["V"]
        G, P = self.G, self.P
        for k in G:
            G[k].fill(0.0)

        p, lab = c["probs"], c["lab"]
        M = lab.size
        dlg = p.copy()
        dlg[np.arange(M), lab] -= 1.0
        dlg /= M
        dlogits = np.zeros((B, T, V))
        dlogits[c["mask"]] = dlg
        G["WE"] += dlogits.reshape(-1, V).T @ c["hf"].reshape(-1, d)
        dhf = dlogits @ P["WE"]
        dh, dg, db = ln_backward(dhf, c["lnf"])
        G["lnfg"] += dg
        G["lnfb"] += db

        for l in reversed(range(L)):
            dh = self._block_backward(dh, l, c)

        np.add.at(G["WE"], c["idx"], dh)
        if "P" in P:
            G["P"][:T] += dh.sum(axis=0)

    def _block_backward(self, dh, l, c):
        pre = "%d." % l
        P, G = self.P, self.G
        do = dh
        G[pre + "W2"] += c[pre + "rel"].reshape(-1, c[pre + "rel"].shape[-1]).T \
            @ do.reshape(-1, do.shape[-1])
        G[pre + "b2"] += do.sum(axis=(0, 1))
        drel = do @ P[pre + "W2"].T
        df = relu_backward(drel, c[pre + "relmask"])
        G[pre + "W1"] += c[pre + "ln2"][0].reshape(-1, c[pre + "ln2"][0].shape[-1]).T \
            @ df.reshape(-1, df.shape[-1])
        G[pre + "b1"] += df.sum(axis=(0, 1))
        dbb = df @ P[pre + "W1"].T
        dh1, dg2, db2 = ln_backward(dbb, c[pre + "ln2"])
        G[pre + "ln2g"] += dg2
        G[pre + "ln2b"] += db2
        dh1 = dh1 + do                                  # residual

        dz = dh1
        mix = self.cfg["mix"]
        if mix == "attn":
            da, dWq, dbq, dWk, dbk, dWv, dbv, dWo, dbo = attn_backward_full(
                dz, c[pre + "attn"])
            G[pre + "Wq"] += dWq; G[pre + "bq"] += dbq
            G[pre + "Wk"] += dWk; G[pre + "bk"] += dbk
            G[pre + "Wv"] += dWv; G[pre + "bv"] += dbv
            G[pre + "Wo"] += dWo; G[pre + "bo"] += dbo
        elif mix == "fnet":
            da = fnet_backward(dz, c["T"])
        else:
            m_max = P[pre + "Gr"].shape[0]
            da, dGr, dGi = spectral_backward(dz, c[pre + "spec"], m_max)
            G[pre + "Gr"] += dGr
            G[pre + "Gi"] += dGi

        dln1, dg1, db1 = ln_backward(da, c[pre + "ln1"])
        G[pre + "ln1g"] += dg1
        G[pre + "ln1b"] += db1
        # d/dh = d/dh1 (already holds both the residual and the MLP branch)
        #        + d/dh through LN1
        return dh1 + dln1

    # ---------------- SGD with momentum ----------------
    def step(self, lr, momentum, clip):
        tot = 0.0
        for k in self.G:
            tot += float((self.G[k] ** 2).sum())
        tot = math.sqrt(tot)
        scale = 1.0 if (clip <= 0 or tot <= clip) else clip / (tot + 1e-12)
        for k in self.P:
            g = self.G[k] * scale
            self.M[k] = momentum * self.M[k] + g
            self.P[k] -= lr * self.M[k]
        return tot


# the multi-head attention backward, written out in full (the stub above is
# replaced by this complete implementation)
def attn_backward_full(dout, cache):
    (a, Qh, Kh, Vh, A, ctx_m, scale, (B, T, d, heads, dh), Wq, Wk, Wv, Wo) = cache
    do2 = dout.reshape(B * T, d)
    dWo = ctx_m.reshape(B * T, d).T @ do2
    dbo = dout.sum(axis=(0, 1))
    dctx_m = (dout @ Wo.T).reshape(B, T, heads, dh).transpose(0, 2, 1, 3)
    dA = dctx_m @ Vh.transpose(0, 1, 3, 2)
    dVh = A.transpose(0, 1, 3, 2) @ dctx_m
    dscores = A * (dA - (dA * A).sum(-1, keepdims=True))
    dQh = (dscores @ Kh) * scale
    dKh = (dscores.transpose(0, 1, 3, 2) @ Qh) * scale
    un = lambda x: x.transpose(0, 2, 1, 3).reshape(B, T, d)
    dQ, dK, dV = un(dQh), un(dKh), un(dVh)
    ar = a.reshape(B * T, d)
    dWq = ar.T @ dQ.reshape(B * T, d)
    dWk = ar.T @ dK.reshape(B * T, d)
    dWv = ar.T @ dV.reshape(B * T, d)
    dbq = dQ.sum(axis=(0, 1)); dbk = dK.sum(axis=(0, 1)); dbv = dV.sum(axis=(0, 1))
    da = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
    return da, dWq, dbq, dWk, dbk, dWv, dbv, dWo, dbo


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


class Corpus(object):
    def __init__(self, path, val_frac=0.05):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.vocab = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.mask_id = len(chars)                      # [MASK]
        self.V = len(chars) + 1
        ids = np.array([self.stoi[ch] for ch in text], dtype=np.int64)
        n_val = max(1024, int(len(ids) * val_frac))
        self.train = ids[:-n_val]
        self.val = ids[-n_val:]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "?") for i in ids)


def make_batch(ids, rng, B, T, mask_prob=0.15):
    starts = rng.randint(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[s:s + T] for s in starts])
    labels = x.copy()
    mask = rng.rand(B, T) < mask_prob
    # guarantee at least one masked position per row
    for b in range(B):
        if not mask[b].any():
            mask[b, rng.randint(T)] = True
    r = rng.rand(B, T)
    x = x.copy()
    sel = mask & (r < 0.8)
    x[sel] = Globals.mask_id
    rnd = mask & (r >= 0.8) & (r < 0.9)
    x[rnd] = rng.randint(0, Globals.V, size=int(rnd.sum()))
    return x, mask, labels


class Globals(object):
    mask_id = 0
    V = 0


# --------------------------------------------------------------------------
# gradient check
# --------------------------------------------------------------------------


def gradcheck(mix, seed=0, eps=1e-5, n_probe=8):
    """Central-difference check of every hand-derived backward pass.

    Criterion: the largest ABSOLUTE discrepancy, normalised by the RMS of all
    analytic gradients (a per-entry relative error is meaningless when an
    individual gradient is ~1e-5, where float64 difference noise dominates).
    """
    np.random.seed(seed)
    T, d, F, L, V, B = 10, 8, 12, 2, 11, 4
    cfg = dict(V=V, d=d, F=F, L=L, T_train=T, T_max=64, mix=mix, heads=2,
               pos="sin", seed=seed)
    m = Model(cfg)
    rng = np.random.RandomState(1)
    idx = rng.randint(0, V, size=(B, T))
    mask = rng.rand(B, T) < 0.4
    for b in range(B):
        mask[b, 0] = True
    labels = rng.randint(0, V, size=(B, T))

    def loss_at():
        return m.forward(idx, mask, labels)[0]

    loss, acc, c = m.forward(idx, mask, labels)
    m.backward(c)

    gscale = math.sqrt(sum(float((m.G[k] ** 2).sum()) for k in m.G)
                       / max(1, sum(m.G[k].size for k in m.G)))
    worst_abs, worst_txt = 0.0, ""
    n_checked = 0
    for name in sorted(m.P):
        flat = m.P[name].reshape(-1)
        gall = m.G[name].reshape(-1)
        picks = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
        for j in picks:
            orig = flat[j]
            flat[j] = orig + eps
            lp = loss_at()
            flat[j] = orig - eps
            lm = loss_at()
            flat[j] = orig
            num = (lp - lm) / (2 * eps)
            ana = gall[j]
            n_checked += 1
            if abs(num - ana) > worst_abs:
                worst_abs = abs(num - ana)
                worst_txt = "%s[%d] num=%.6e ana=%.6e" % (name, j, num, ana)
    rel = worst_abs / max(gscale, 1e-12)
    print("[gradcheck:%s] checked=%d  |grad|rms=%.3e  max_abs_err=%.3e  "
          "global_rel=%.2e  (%s)" % (mix, n_checked, gscale, worst_abs, rel, worst_txt))
    return rel


# --------------------------------------------------------------------------
# training / evaluation
# --------------------------------------------------------------------------


def evaluate(m, ids, B, T, batches, seed=1234):
    rng = np.random.RandomState(seed)
    tot, acc = 0.0, 0.0
    for _ in range(batches):
        x, mk, y = make_batch(ids, rng, B, T)
        l, a, _ = m.forward(x, mk, y)
        tot += l
        acc += a
    return tot / batches, acc / batches


def lr_at(step, steps, base, warmup, final_frac=0.05):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return base * (1 - t * (1 - final_frac))


def train(args):
    corp = Corpus(args.data)
    Globals.mask_id = corp.mask_id
    Globals.V = corp.V
    T, d, L, F = args.T, args.d, args.L, args.F
    cfg = dict(V=corp.V, d=d, F=F, L=L, T_train=T, T_max=args.t_max, mix=args.mix,
               heads=args.heads, pos=args.pos, seed=args.seed,
               wo_zero=args.wo_zero)
    m = Model(cfg)
    if args.wo_zero:
        args.tag_suffix = "_wozero"
    tag = "%s_T%d_d%d_L%d_s%d%s" % (args.mix, T, d, L, args.seed,
                                    "_wozero" if args.wo_zero else "")
    print("== run %s | vocab=%d | params=%d (mix=%d, %.1f%%) | threads=%s"
          % (tag, corp.V, m.n_params(), m.n_mix_params(),
             100.0 * m.n_mix_params() / m.n_params(), os.environ.get("OMP_NUM_THREADS", "?")))
    rng = np.random.RandomState(args.seed * 7919 + 13)
    log_path = os.path.join(args.outdir, tag + ".jsonl")
    t0 = time.time()
    running, seen = 0.0, 0
    hist = []
    with open(log_path, "w") as log:
        for step in range(args.steps):
            if step % args.eval_every == 0:
                vl, va = evaluate(m, corp.val, args.eval_batch, T, args.eval_batches)
                rec = dict(step=step, val_loss=vl, val_acc=va,
                           train_loss=(running / max(1, seen)), lr=lr_at(step, args.steps, args.lr, args.warmup),
                           elapsed=time.time() - t0)
                hist.append(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print("[%s] step %5d  train %.4f  val %.4f  acc %.4f  (%.0fs)"
                      % (tag, step, rec["train_loss"], vl, va, rec["elapsed"]))
                running, seen = 0.0, 0
            x, mk, y = make_batch(corp.train, rng, args.batch, T, args.mask_prob)
            loss, acc, c = m.forward(x, mk, y)
            m.backward(c)
            gnorm = m.step(lr_at(step, args.steps, args.lr, args.warmup),
                           args.momentum, args.clip)
            running += loss
            seen += 1

        # always record a final evaluation so short runs are comparable
        if (args.steps - 1) % args.eval_every != 0:
            vl, va = evaluate(m, corp.val, args.eval_batch, T, args.eval_batches)
            rec = dict(step=args.steps - 1, val_loss=vl, val_acc=va,
                       train_loss=(running / max(1, seen)),
                       lr=lr_at(args.steps - 1, args.steps, args.lr, args.warmup),
                       elapsed=time.time() - t0, final=True)
            hist.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("[%s] step %5d (final) train %.4f  val %.4f  acc %.4f  (%.0fs)"
                  % (tag, rec["step"], rec["train_loss"], vl, va, rec["elapsed"]))

    # ---- extras: length extrapolation + learned spectrum ----
    extras = {}
    if args.extrapolate:
        extras["extrapolation"] = {}
        for Lt in args.extrapolate:
            vl, va = evaluate(m, corp.val, args.extrap_batch, Lt, args.eval_batches)
            extras["extrapolation"][str(Lt)] = dict(val_loss=vl, val_acc=va)
            print("[%s] zero-shot T=%d  val %.4f  acc %.4f" % (tag, Lt, vl, va))

    if args.mix == "spectral":
        prof = []
        for l in range(L):
            Gr, Gi = m.P["%d.Gr" % l], m.P["%d.Gi" % l]
            prof.append(np.sqrt(Gr ** 2 + Gi ** 2).mean(axis=1).tolist())
        extras["spectrum"] = prof

        # What did the mixer actually learn: magnitude attenuation, or phase?
        # y = IFFT(H . FFT(x)) is circular convolution with kernel h = IFFT(H),
        # so h is the (circulant) mixing kernel and h[0]/||h|| says how much of
        # the transform is identity-like versus genuine cross-position mixing.
        ana = []
        for l in range(L):
            H = herm_filter(m.P["%d.Gr" % l], m.P["%d.Gi" % l], T)   # (T,d)
            h = np.fft.ifft(H, axis=0).real                           # (T,d)
            energy = (h ** 2).sum(axis=0) + 1e-12
            self_frac = (h[0] ** 2) / energy
            ph = np.angle(H)
            ana.append(dict(
                layer=l,
                mean_abs_H=float(np.abs(H).mean()),
                max_abs_H=float(np.abs(H).max()),
                mean_self_fraction=float(self_frac.mean()),      # 1.0 = identity
                mean_offdiag_fraction=float((1.0 - self_frac).mean()),
                phase_std=float(ph.std()),
                phase_absmean=float(np.abs(ph).mean())))
        extras["mixer_analysis"] = ana
        if args.save_weights:
            np.savez(os.path.join(args.outdir, tag + ".weights.npz"), **m.P)

    # ---- qualitative cloze samples ----
    samples = []
    rng2 = np.random.RandomState(7)
    x, mk, y = make_batch(corp.val, rng2, 8, T)
    _, _, c = m.forward(x, mk, y)
    pred = c["probs"].argmax(-1)
    bs, ts = c["mask_bs"], c["mask_ts"]
    shown = {}
    for i in range(len(bs)):
        b, p = int(bs[i]), int(ts[i])
        shown[b] = shown.get(b, 0)
        if shown[b] >= 4:
            continue
        shown[b] += 1
        lo = max(0, p - 24)
        samples.append(dict(
            context=corp.decode(x[b, lo:p + 25]),
            pos_in_window=int(p - lo),
            target=corp.decode([y[b, p]]),
            pred=corp.decode([pred[i]])))
    extras["cloze"] = samples[:24]

    total = time.time() - t0
    extras["meta"] = dict(tag=tag, mix=args.mix, T=T, d=d, L=L, F=F, steps=args.steps,
                          params=m.n_params(), mix_params=m.n_mix_params(),
                          vocab=corp.V, seconds=total, args=vars(args))
    with open(os.path.join(args.outdir, tag + ".final.json"), "w") as f:
        json.dump(dict(history=hist, **extras), f, indent=1)
    print("[%s] done in %.0fs; best val %.4f" % (tag, total, min(r["val_loss"] for r in hist)))
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="spectral",
                    choices=["spectral", "fnet", "attn"])
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--t_max", type=int, default=768)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--F", type=int, default=384)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--extrap_batch", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--pos", default="sin", choices=["sin", "learned"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wo_zero", action="store_true",
                    help="zero-init the attention output projection (attn arm only)")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--extrapolate", type=int, nargs="*", default=[])
    ap.add_argument("--save_weights", action="store_true")
    args = ap.parse_args()
    if args.gradcheck:
        bad = []
        for mix in ["spectral", "fnet", "attn"]:
            w = gradcheck(mix)
            if w > 1e-4:
                bad.append(mix)
        print("GRADCHECK:", "FAIL %s" % bad if bad else
              "ALL PASS (max global-relative gradient error < 1e-4)")
        return
    os.makedirs(args.outdir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
