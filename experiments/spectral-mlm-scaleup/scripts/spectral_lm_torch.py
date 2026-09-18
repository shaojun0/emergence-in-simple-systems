#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""spectral_lm_torch.py -- GPU/scale port of the upstream NumPy experiment
`experiments/spectral-mlm-cpu/spectral_bert.py` (repo: emergence-in-simple-systems).

What is kept IDENTICAL to the upstream NumPy implementation
----------------------------------------------------------
  * the skeleton:  h = Emb(x) + sin/cos PE ; per block
        a = LN(h); z = Mix(a); h = h + z; h = h + MLP(LN(h))
    final LN, tied embeddings, logits = LN(h) @ Emb^T
  * the objective: BERT-style MLM, independent 15% mask, 80/10/10 replacement,
    >=1 masked position per row, cross-entropy on masked positions only
  * the three upstream mixing arms and their exact parameterisations
        fnet     : z = Re(FFT(a))                                   (0 params)
        spectral : z = Re(IFFT(H . FFT(a))), H Hermitian, H=1 init  (FNO/GFNet)
        attn     : plain multi-head self-attention, Glorot init, optional Wo=0
  * the optimizer: plain SGD + momentum 0.9, global-norm clip 1.0,
    warmup-200 + linear decay to 5%   (NO Adam, NO weight decay, NO dropout)
  * the data pipeline, the random-crop batching, the eval protocol, and the
    learned-kernel (lag distribution) analysis

What is new (the point of the port)
-----------------------------------
  * runs on GPU, so corpus N x d x L x T x steps can be scaled up orders of
    magnitude beyond the upstream 1.1 MB / 0.45 M-param CPU run
  * two extra arms that upstream lists as open questions:
        stencil : depthwise *circular* conv, support |lag| <= k  (upstream Q2)
                  -- exactly the local restriction of the spectral arm, so
                     "local kernel vs global kernel" is a controlled comparison
        gated   : content-dependent channel gating of the spectral filter (Q3)
  * gradient accumulation, checkpoint/resume, richer diagnostics

Verification: `--equiv_npz` compares this implementation against a dump made by
the upstream NumPy code, parameter-by-parameter and gradient-by-gradient.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# LR schedule (bit-identical to upstream lr_at)
# --------------------------------------------------------------------------

def lr_at(step, steps, base, warmup, final_frac=0.05):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, steps - warmup)
    return base * (1 - t * (1 - final_frac))


def sincos_table(T, d, device, dtype):
    pos = torch.arange(T, device=device, dtype=torch.float64)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float64)
    div = torch.exp(-math.log(10000.0) * i / d)
    pe = torch.zeros(T, d, device=device, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


# --------------------------------------------------------------------------
# mixing layers
# --------------------------------------------------------------------------

class UpstreamCE(torch.autograd.Function):
    """Upstream's masked-LM loss: forward is -log(p[label] + eps).mean(), while
    the hand-derived backward is the exact cross-entropy gradient (upstream adds
    the epsilon only in the forward pass).  Both halves are replicated so the
    port's reported loss *and* its gradients match upstream exactly."""

    @staticmethod
    def forward(ctx, logits, labels, eps):
        p = torch.softmax(logits, dim=-1)
        pl = p.gather(1, labels.unsqueeze(1)).squeeze(1)
        ctx.save_for_backward(p, labels)
        return -torch.log(pl + eps).mean()

    @staticmethod
    def backward(ctx, g):
        p, labels = ctx.saved_tensors
        M = labels.numel()
        d = p.clone()
        d[torch.arange(M, device=d.device), labels] -= 1.0
        d /= M
        return d * g, None, None


class FNetMix(nn.Module):
    """y = Re(FFT(x)); zero parameters (Lee-Thorp et al., NAACL 2022)."""

    def forward(self, a):
        return torch.fft.fft(a, dim=1).real

    def n_params(self):
        return 0

    def active_params(self, T):
        return 0


class SpectralMix(nn.Module):
    """Global circular convolution y = Re(IFFT(H . FFT(x))), H Hermitian.

    H is stored as its T_max//2+1 free real/imag rows, exactly like upstream's
    herm_filter().  Rows past T//2+1 are unused at length T but exist so the
    same weights can be evaluated at longer T (upstream's extrapolation test).
    """

    def __init__(self, d, t_max, identity=True, seed=0):
        super().__init__()
        self.m_max = t_max // 2 + 1
        if identity:
            gr = torch.ones(self.m_max, d)
            gi = torch.zeros(self.m_max, d)
        else:
            g = torch.Generator().manual_seed(seed)
            gr = 1.0 + 0.02 * torch.randn(self.m_max, d, generator=g)
            gi = 0.02 * torch.randn(self.m_max, d, generator=g)
        self.Gr = nn.Parameter(gr)
        self.Gi = nn.Parameter(gi)

    def _H(self, T, dtype):
        M = T // 2 + 1
        mv = torch.ones(M, 1, dtype=dtype, device=self.Gr.device)
        mv[0] = 0.0                       # DC imaginary part cannot matter
        if T % 2 == 0 and M > 1:
            mv[M - 1] = 0.0               # Nyquist component must be real
        return torch.complex(self.Gr[:M], self.Gi[:M] * mv)

    def forward(self, a):
        T = a.shape[1]
        X = torch.fft.rfft(a, dim=1)
        return torch.fft.irfft(X * self._H(T, a.dtype), n=T, dim=1)

    def n_params(self):
        return 2 * self.m_max * self.Gr.shape[1]

    def active_params(self, T):
        return 2 * (T // 2 + 1) * self.Gr.shape[1]

    def kernel(self, T):
        """(real circular kernel h (T,d), half spectrum H (T//2+1,d))."""
        H = self._H(T, self.Gr.dtype).detach()
        return torch.fft.irfft(H, n=T, dim=0), H


class GatedSpectralMix(SpectralMix):
    """AFNO-flavoured content-dependent channel gate on the spectral filter:

        g = W2 . ReLU(W1 . mean_freq|X| + b1) + b2      (B, d)
        y = IFFT( H . X . softplus(g) )

    Initialised so softplus(g) == 1 exactly (W2 = 0, b2 = softplus^-1(1)),
    i.e. the arm starts as the plain identity spectral mixer.
    """

    def __init__(self, d, t_max, seed=0):
        super().__init__(d, t_max, identity=True, seed=seed)
        self.W1 = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W1)
        self.b1 = nn.Parameter(torch.zeros(d))
        self.W2 = nn.Parameter(torch.zeros(d, d))
        self.b2 = nn.Parameter(torch.full((d,), math.log(math.e - 1.0)))

    def forward(self, a):
        T = a.shape[1]
        X = torch.fft.rfft(a, dim=1)
        s = X.abs().mean(dim=1)                          # (B,d)
        g = F.softplus(s @ self.W1.T + self.b1) @ self.W2.T + self.b2
        g = F.softplus(g)
        return torch.fft.irfft(X * self._H(T, a.dtype) * g.unsqueeze(1), n=T, dim=1)

    def n_params(self):
        d = self.Gr.shape[1]
        return 2 * self.m_max * d + 2 * d * d + 2 * d


class StencilMix(nn.Module):
    """Depthwise *circular* convolution with support |lag| <= k (upstream Q2).

    A spectral filter whose time-domain kernel has support <= k is exactly this
    family, so this arm is the strictly-local subset of the spectral arm.
    Starts at the identity (centre tap = 1), like spectral.
    """

    def __init__(self, d, k=2):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.zeros(d, 1, 2 * k + 1))
        with torch.no_grad():
            self.w[:, 0, k] = 1.0

    def forward(self, a):
        x = a.transpose(1, 2)
        x = F.pad(x, (self.k, self.k), mode="circular")
        return F.conv1d(x, self.w, groups=self.w.shape[0]).transpose(1, 2)

    def n_params(self):
        return self.w.numel()

    def active_params(self, T):
        return self.w.numel()


class AttnMix(nn.Module):
    """Plain multi-head self-attention -- the upstream control arm."""

    def __init__(self, d, heads, wo_zero=False):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        for nm in ("Wq", "Wk", "Wv", "Wo"):
            p = nn.Parameter(torch.empty(d, d))
            nn.init.xavier_uniform_(p)
            setattr(self, nm, p)
            setattr(self, "b" + nm[1], nn.Parameter(torch.zeros(d)))
        if wo_zero:
            with torch.no_grad():
                self.Wo.zero_()
                self.bo.zero_()

    def forward(self, a):
        B, T, d = a.shape
        sh = lambda x: x.reshape(B, T, self.h, self.dh).transpose(1, 2)
        Q = sh(a @ self.Wq + self.bq)
        K = sh(a @ self.Wk + self.bk)
        V = sh(a @ self.Wv + self.bv)
        A = torch.softmax((Q @ K.transpose(-1, -2)) / math.sqrt(self.dh), dim=-1)
        ctx = (A @ V).transpose(1, 2).reshape(B, T, d)
        return ctx @ self.Wo + self.bo

    def n_params(self):
        return 4 * self.Wq.numel() + 4 * self.Wq.shape[0]

    def active_params(self, T):
        return self.n_params()


def make_mix(cfg, layer_seed):
    mix = cfg["mix"]
    if mix == "fnet":
        return FNetMix()
    if mix == "spectral":
        return SpectralMix(cfg["d"], cfg["T_max"], seed=layer_seed)
    if mix == "gated":
        return GatedSpectralMix(cfg["d"], cfg["T_max"], seed=layer_seed)
    if mix == "stencil":
        return StencilMix(cfg["d"], cfg.get("stencil_k", 2))
    if mix == "attn":
        return AttnMix(cfg["d"], cfg["heads"], cfg.get("wo_zero", False))
    raise ValueError(mix)


class UpstreamMLP(torch.autograd.Function):
    """Reproduces a bug in the upstream hand-derived backward pass.

    Upstream's `_block_backward` computes

        G[W1] += c[pre+"ln2"][0]^T @ df

    where `c[pre+"ln2"][0]` is the *pre-gain* normalised activation xh, while
    the forward pass actually uses `bb = xh*g + b`.  The true gradient is
    `bb^T @ df`.  At initialisation g=1, b=0, so xh == bb and the error is
    invisible -- which is precisely where upstream runs its finite-difference
    gradcheck.  After any update that moves the LN2 gain/bias, the W1 gradient
    is wrong (we measure a 6e-5 relative error after a single step).

    This Function implements upstream's forward exactly and its (buggy)
    backward exactly, so the port can be verified against upstream bit-for-bit.
    """

    @staticmethod
    def forward(ctx, bb, xh, W1, b1):
        ctx.save_for_backward(xh, W1)
        return bb @ W1 + b1

    @staticmethod
    def backward(ctx, df):
        xh, W1 = ctx.saved_tensors
        dW1 = xh.reshape(-1, xh.shape[-1]).T @ df.reshape(-1, df.shape[-1])
        db1 = df.sum(dim=(0, 1))
        dbb = df @ W1.T
        return dbb, None, dW1, db1


def ln_forward_ex(x, w, b, eps=1e-5):
    """LayerNorm returning both the output and the pre-gain normalised value."""
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    xh = (x - mu) / torch.sqrt(var + eps)
    return xh * w + b, xh


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, cfg, layer_seed):
        super().__init__()
        d, Fd = cfg["d"], cfg["F"]
        self.bug = bool(cfg.get("upstream_bug", False))
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.mix = make_mix(cfg, layer_seed)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.W1 = nn.Parameter(torch.empty(d, Fd))
        self.b1 = nn.Parameter(torch.zeros(Fd))
        self.W2 = nn.Parameter(torch.empty(Fd, d))
        self.b2 = nn.Parameter(torch.zeros(d))
        for p, fan_in, fan_out in ((self.W1, d, Fd), (self.W2, Fd, d)):
            lim = math.sqrt(6.0 / (fan_in + fan_out))
            with torch.no_grad():
                p.uniform_(-lim, lim)

    def forward(self, h):
        h = h + self.mix(self.ln1(h))
        if self.bug:
            b, xh = ln_forward_ex(h, self.ln2.weight, self.ln2.bias, self.ln2.eps)
            f = UpstreamMLP.apply(b, xh, self.W1, self.b1)
        else:
            b = self.ln2(h)
            f = b @ self.W1 + self.b1
        return h + F.relu(f) @ self.W2 + self.b2


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, L, V = cfg["d"], cfg["L"], cfg["V"]
        g = torch.Generator().manual_seed(cfg["seed"])
        self.WE = nn.Parameter(torch.randn(V, d, generator=g) * 0.02)
        if cfg["pos"] == "learned":
            self.P = nn.Parameter(torch.randn(cfg["T_max"], d, generator=g) * 0.02)
        else:
            self.register_parameter("P", None)
        self.blocks = nn.ModuleList([Block(cfg, cfg["seed"] * 131 + l) for l in range(L)])
        self.lnf = nn.LayerNorm(d, eps=1e-5)
        self._pe_cache = {}

    def pos(self, T):
        if self.P is not None:
            return self.P[:T]
        key = (T, self.WE.device)
        if key not in self._pe_cache:
            self._pe_cache[key] = sincos_table(T, self.cfg["d"], self.WE.device,
                                               self.WE.dtype)
        return self._pe_cache[key]

    def forward(self, idx, mask=None, labels=None):
        B, T = idx.shape
        h = self.WE[idx] + self.pos(T).unsqueeze(0)
        for blk in self.blocks:
            h = blk(h)
        hf = self.lnf(h)
        logits = hf @ self.WE.T
        if mask is None:
            return logits
        lg = logits[mask]
        lab = labels[mask]
        eps = self.cfg.get("loss_eps", 0.0)
        if eps:
            loss = UpstreamCE.apply(lg, lab, eps)
        else:
            loss = F.cross_entropy(lg, lab)
        acc = (lg.argmax(-1) == lab).to(torch.float64).mean().item()
        return loss, acc, logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_mix_params(self):
        return sum(b.mix.n_params() for b in self.blocks)

    def n_active_mix_params(self, T):
        return sum(b.mix.active_params(T) for b in self.blocks)

    # ---- bridge to the upstream NumPy parameter names ----
    def load_numpy_params(self, npz, strict=True):
        if hasattr(npz, "files"):
            names = set(npz.files)
            get = lambda k: npz[k]                       # noqa: E731
        else:                                            # plain dict
            names = set(npz.keys())
            get = lambda k: npz[k]                       # noqa: E731
        with torch.no_grad():
            self.WE.copy_(torch.as_tensor(get("WE"), dtype=self.WE.dtype))
            if self.P is not None and "P" in names:
                self.P.copy_(torch.as_tensor(get("P"), dtype=self.P.dtype))
            for l, blk in enumerate(self.blocks):
                blk.ln1.weight.copy_(torch.as_tensor(get("%d.ln1g" % l), dtype=blk.ln1.weight.dtype))
                blk.ln1.bias.copy_(torch.as_tensor(get("%d.ln1b" % l), dtype=blk.ln1.bias.dtype))
                blk.ln2.weight.copy_(torch.as_tensor(get("%d.ln2g" % l), dtype=blk.ln2.weight.dtype))
                blk.ln2.bias.copy_(torch.as_tensor(get("%d.ln2b" % l), dtype=blk.ln2.bias.dtype))
                blk.W1.copy_(torch.as_tensor(get("%d.W1" % l), dtype=blk.W1.dtype))
                blk.b1.copy_(torch.as_tensor(get("%d.b1" % l), dtype=blk.b1.dtype))
                blk.W2.copy_(torch.as_tensor(get("%d.W2" % l), dtype=blk.W2.dtype))
                blk.b2.copy_(torch.as_tensor(get("%d.b2" % l), dtype=blk.b2.dtype))
                mx = blk.mix
                if self.cfg["mix"] == "spectral":
                    mx.Gr.copy_(torch.as_tensor(get("%d.Gr" % l), dtype=mx.Gr.dtype))
                    mx.Gi.copy_(torch.as_tensor(get("%d.Gi" % l), dtype=mx.Gi.dtype))
                elif self.cfg["mix"] == "attn":
                    for nm in ("Wq", "Wk", "Wv", "Wo"):
                        getattr(mx, nm).copy_(torch.as_tensor(get("%d.%s" % (l, nm)),
                                                              dtype=getattr(mx, nm).dtype))
                        bnm = "b" + nm[1]
                        getattr(mx, bnm).copy_(torch.as_tensor(get("%d.%s" % (l, bnm)),
                                                               dtype=getattr(mx, bnm).dtype))
            self.lnf.weight.copy_(torch.as_tensor(get("lnfg"), dtype=self.lnf.weight.dtype))
            self.lnf.bias.copy_(torch.as_tensor(get("lnfb"), dtype=self.lnf.bias.dtype))

    def numpy_named_grads(self):
        out = {"WE": self.WE.grad, "lnfg": self.lnf.weight.grad, "lnfb": self.lnf.bias.grad}
        if self.P is not None:
            out["P"] = self.P.grad
        for l, blk in enumerate(self.blocks):
            out["%d.ln1g" % l] = blk.ln1.weight.grad
            out["%d.ln1b" % l] = blk.ln1.bias.grad
            out["%d.ln2g" % l] = blk.ln2.weight.grad
            out["%d.ln2b" % l] = blk.ln2.bias.grad
            out["%d.W1" % l] = blk.W1.grad
            out["%d.b1" % l] = blk.b1.grad
            out["%d.W2" % l] = blk.W2.grad
            out["%d.b2" % l] = blk.b2.grad
            mx = blk.mix
            if self.cfg["mix"] == "spectral":
                out["%d.Gr" % l] = mx.Gr.grad
                out["%d.Gi" % l] = mx.Gi.grad
            elif self.cfg["mix"] == "attn":
                for nm in ("Wq", "Wk", "Wv", "Wo"):
                    out["%d.%s" % (l, nm)] = getattr(mx, nm).grad
                    bnm = "b" + nm[1]
                    out["%d.%s" % (l, bnm)] = getattr(mx, bnm).grad
        return out


def clip_like_upstream(params, clip):
    """Exact replica of upstream Model.step()'s global-norm clipping.

    torch.nn.utils.clip_grad_norm_ uses clip/(total+1e-6), which perturbs every
    update by ~1e-6 relative -- enough to make a long trajectory drift away from
    the NumPy reference.  This version uses clip/total, like upstream.
    """
    if clip <= 0:
        return 0.0
    params = list(params)                       # model.parameters() is a generator
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    tot = math.sqrt(tot)
    scale = 1.0 if tot <= clip else clip / (tot + 1e-12)
    if scale != 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
    return tot


# --------------------------------------------------------------------------
# data (identical semantics to upstream Corpus / make_batch)
# --------------------------------------------------------------------------

class Corpus(object):
    def __init__(self, path, encoding="utf-8", val_frac=0.05):
        with open(path, "rb") as f:
            raw = f.read()
        # match Python text-mode universal newlines (upstream reads with "r")
        text = raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        chars = sorted(set(text))
        self.vocab = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.mask_id = len(chars)
        self.V = len(chars) + 1
        if self.V <= 256:
            tbl = str.maketrans({ch: chr(i) for i, ch in enumerate(chars)})
            ids = np.frombuffer(text.translate(tbl).encode("latin-1"), dtype=np.uint8)
        else:                                            # rare: >256 distinct symbols
            arr = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
            lut = np.zeros(int(arr.max()) + 1, dtype=np.int32)
            for ch, i in self.stoi.items():
                lut[ord(ch)] = i
            ids = lut[arr]
        self.raw_ids = ids
        n_val = max(1024, int(len(ids) * val_frac))
        self.train = ids[:-n_val]
        self.val = ids[-n_val:]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "?") for i in ids)


class DeviceCorpus(object):
    """Whole corpus resident on the device, sampled on-device (uint8/uint16)."""

    def __init__(self, corpus, device):
        self.mask_id = corpus.mask_id
        self.V = corpus.V
        self.train = torch.as_tensor(np.ascontiguousarray(corpus.train), device=device)
        self.val = torch.as_tensor(np.ascontiguousarray(corpus.val), device=device)


def make_batch_gpu(ids, gen, B, T, mask_prob, V, mask_id):
    n = ids.shape[0]
    starts = torch.randint(0, n - T - 1, (B,), generator=gen, device=ids.device)
    x = ids[starts.unsqueeze(1) + torch.arange(T, device=ids.device).unsqueeze(0)].long()
    labels = x.clone()
    mask = torch.rand(B, T, generator=gen, device=ids.device) < mask_prob
    empty = ~mask.any(dim=1)
    if bool(empty.any()):                                # >=1 mask per row
        idx = torch.nonzero(empty).flatten()
        cols = torch.randint(0, T, (idx.numel(),), generator=gen, device=ids.device)
        mask[idx, cols] = True
    r = torch.rand(B, T, generator=gen, device=ids.device)
    x = x.clone()
    x[mask & (r < 0.8)] = mask_id
    rnd = mask & (r >= 0.8) & (r < 0.9)
    x[rnd] = torch.randint(0, V, (int(rnd.sum()),), generator=gen, device=ids.device)
    return x, mask, labels


def evaluate(model, ids, B, T, batches, V, mask_id, seed=1234):
    gen = torch.Generator(device=ids.device).manual_seed(seed)
    tot, acc = 0.0, 0.0
    with torch.no_grad():
        for _ in range(batches):
            x, mk, y = make_batch_gpu(ids, gen, B, T, 0.15, V, mask_id)
            l, a, _ = model(x, mk, y)
            tot += float(l)
            acc += a
    return tot / batches, acc / batches


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def kernel_lag_profile(h, n_lags=129, r_frac=0.90):
    """Channel-summed energy of the circular kernel h (T,d) per lag + r90."""
    T = h.shape[0]
    e = (h ** 2).sum(dim=1)
    tot = float(e.sum()) + 1e-30
    prof = {"lag0": float(e[0]) / tot}
    for k in range(1, min(n_lags, T // 2 + 1)):
        v = float(e[k]) if k == T - k else float(e[k] + e[T - k])
        prof["lag%d" % k] = v / tot
    off = e.clone()
    off[0] = 0.0
    o = float(off.sum())
    if o > 0:
        dist = torch.minimum(torch.arange(T, device=h.device),
                             (T - torch.arange(T, device=h.device)) % T)
        # +-k folded into one bin so +lag and -lag are not counted twice
        folded = torch.zeros(T // 2 + 1, device=h.device, dtype=e.dtype)
        folded.index_add_(0, dist, off)
        peak = int(folded[1:].argmax()) + 1 if T > 2 else 0
        order = torch.argsort(dist)
        cs = torch.cumsum(off[order], 0) / o
        hit = (cs >= r_frac).nonzero()
        r90 = int(dist[order][int(hit[0])]) if len(hit) else int(dist.max())
    else:
        peak, r90 = 0, 0
    prof["offdiag_peak_lag"] = peak
    prof["offdiag_r90"] = r90
    prof["self_fraction"] = prof["lag0"]
    prof["offdiag_frac"] = 1.0 - prof["lag0"]
    return prof


def attn_stats(model, ids, V, mask_id, T, B=8):
    """Mean attention entropy / attended distance / max weight per layer."""
    gen = torch.Generator(device=ids.device).manual_seed(99)
    x, mk, y = make_batch_gpu(ids, gen, B, T, 0.15, V, mask_id)
    store, handles = {}, []

    def mk_hook(l):
        def hook(mod, inp, out):
            a = inp[0]
            Bx, Tx, _ = a.shape
            sh = lambda z: z.reshape(Bx, Tx, mod.h, mod.dh).transpose(1, 2)
            with torch.no_grad():
                Q = sh(a @ mod.Wq + mod.bq)
                K = sh(a @ mod.Wk + mod.bk)
                A = torch.softmax((Q @ K.transpose(-1, -2)) / math.sqrt(mod.dh), -1)
                idx = torch.arange(Tx, device=A.device, dtype=A.dtype)
                diff = (idx[None, None, :, None] - idx[None, None, None, :]).abs()
                store[l] = dict(
                    entropy=float(-(A * torch.log(A + 1e-12)).sum(-1).mean()),
                    mean_distance=float((A * diff).sum(-1).mean()),
                    max_weight=float(A.max(-1).values.mean()),
                    uniform_distance=float(Tx / 3.0))
        return hook

    for l, blk in enumerate(model.blocks):
        handles.append(blk.mix.register_forward_hook(mk_hook(l)))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return [store[l] for l in sorted(store)]


def mixer_diagnostics(model, T, corp_ids=None, V=None, mask_id=None):
    out = {}
    mix = model.cfg["mix"]
    if mix in ("spectral", "gated"):
        layers = []
        for l, blk in enumerate(model.blocks):
            h, H = blk.mix.kernel(T)
            p = kernel_lag_profile(h)
            energy = (h ** 2).sum(0) + 1e-12
            self_frac = (h[0] ** 2) / energy
            layers.append(dict(layer=l, mean_abs_H=float(H.abs().mean()),
                               max_abs_H=float(H.abs().max()),
                               mean_self_fraction=float(self_frac.mean()),
                               min_self_fraction=float(self_frac.min()),
                               frac_channels_mostly_global=float((self_frac < 0.5)
                                                                 .to(torch.float64).mean()),
                               **p))
        out["kernel_lag"] = layers
    if mix == "stencil":
        out["stencil_abs_sum"] = [float(b.mix.w.detach().abs().sum()) for b in model.blocks]
    if mix == "attn" and corp_ids is not None:
        out["attention"] = attn_stats(model, corp_ids, V, mask_id, T)
    return out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(args):
    dev = torch.device(args.device)
    dt = torch.float64 if args.dtype == "float64" else torch.float32
    if args.tf32 and dt == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    corp = Corpus(args.data, encoding=args.encoding, val_frac=args.val_frac)
    dc = DeviceCorpus(corp, dev)
    vocab, mask_id = corp.V, corp.mask_id
    T, d, L, Fd = args.T, args.d, args.L, args.F
    cfg = dict(V=vocab, d=d, F=Fd, L=L, T_max=max(args.t_max, T), mix=args.mix,
               heads=args.heads, pos=args.pos, seed=args.seed,
               wo_zero=args.wo_zero, stencil_k=args.stencil_k,
               upstream_bug=args.upstream_bug,
               loss_eps=(1e-9 if args.upstream_bug else 0.0))
    model = Model(cfg).to(dev).to(dt)
    n_par, n_mix = model.n_params(), model.n_mix_params()
    tag = "%s_T%d_d%d_L%d_s%d" % (args.mix, T, d, L, args.seed)
    if args.wo_zero:
        tag += "_wozero"
    tag += "_lr%g" % args.lr
    if args.tag:
        tag = args.tag + "_" + tag
    print("== run %s | corpus %.1fMB vocab=%d | params=%d (mix=%d active=%d) | %s %s"
          % (tag, os.path.getsize(args.data) / 1e6, vocab, n_par, n_mix,
             model.n_active_mix_params(T), dev, dt), flush=True)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          dampening=0.0, weight_decay=0.0, nesterov=False)
    gen = torch.Generator(device=dev).manual_seed(args.seed * 7919 + 13)
    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, tag + ".jsonl")
    ckpt_path = os.path.join(args.outdir, tag + ".ckpt.pt")
    t0, hist, start_step = time.time(), [], 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        start_step, hist = ck["step"], ck["hist"]
        t0 = time.time() - ck["elapsed"]
        print("   resumed from step %d" % start_step, flush=True)

    running, seen = 0.0, 0
    log = open(log_path, "a" if start_step else "w")
    for step in range(start_step, args.steps):
        if step % args.eval_every == 0:
            vl, va = evaluate(model, dc.val, args.eval_batch, T, args.eval_batches,
                              vocab, mask_id)
            rec = dict(step=step, val_loss=vl, val_acc=va,
                       train_loss=(running / max(1, seen)),
                       lr=lr_at(step, args.steps, args.lr, args.warmup),
                       elapsed=time.time() - t0)
            hist.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("[%s] step %6d  train %.4f  val %.4f  acc %.4f  (%.0fs)"
                  % (tag, step, rec["train_loss"], vl, va, rec["elapsed"]), flush=True)
            running, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, mk, y = make_batch_gpu(dc.train, gen, args.batch, T, args.mask_prob,
                                      vocab, mask_id)
            loss, acc, _ = model(x, mk, y)
            (loss / args.grad_accum).backward()
            running += float(loss.detach())
            seen += 1
        clip_like_upstream(model.parameters(), args.clip)
        opt.param_groups[0]["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        opt.step()
        if args.save_every and step % args.save_every == 0 and step > 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            gen=gen.get_state(), step=step, hist=hist,
                            elapsed=time.time() - t0), ckpt_path)

    if (args.steps - 1) % args.eval_every != 0:
        vl, va = evaluate(model, dc.val, args.eval_batch, T, args.eval_batches,
                          vocab, mask_id)
        rec = dict(step=args.steps - 1, val_loss=vl, val_acc=va,
                   train_loss=(running / max(1, seen)),
                   lr=lr_at(args.steps - 1, args.steps, args.lr, args.warmup),
                   elapsed=time.time() - t0, final=True)
        hist.append(rec)
        log.write(json.dumps(rec) + "\n")
    log.close()

    extras = {}
    if args.extrapolate:
        extras["extrapolation"] = {}
        for Lt in args.extrapolate:
            vl, va = evaluate(model, dc.val, args.extrap_batch, Lt, args.eval_batches,
                              vocab, mask_id)
            extras["extrapolation"][str(Lt)] = dict(val_loss=vl, val_acc=va)
            print("[%s] zero-shot T=%d  val %.4f  acc %.4f" % (tag, Lt, vl, va), flush=True)

    extras["diagnostics"] = mixer_diagnostics(model, T, dc.val, vocab, mask_id)
    if args.save_weights:
        np.savez(os.path.join(args.outdir, tag + ".weights.npz"),
                 **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
                    if v.dtype.is_floating_point})

    samples = []
    gen2 = torch.Generator(device=dev).manual_seed(7)
    x, mk, y = make_batch_gpu(dc.val, gen2, 8, T, 0.15, vocab, mask_id)
    with torch.no_grad():
        _, _, logits = model(x, mk, y)
    pred = logits.argmax(-1)
    bs, ts = torch.nonzero(mk, as_tuple=True)
    shown = {}
    for i in range(len(bs)):
        b, p = int(bs[i]), int(ts[i])
        shown[b] = shown.get(b, 0)
        if shown[b] >= 4:
            continue
        shown[b] += 1
        lo = max(0, p - 24)
        samples.append(dict(context=corp.decode(x[b, lo:p + 25].cpu().numpy()),
                            pos_in_window=int(p - lo),
                            target=corp.decode([int(y[b, p])]),
                            pred=corp.decode([int(pred[b, p])])))
    extras["cloze"] = samples[:24]
    total = time.time() - t0
    extras["meta"] = dict(tag=tag, mix=args.mix, T=T, d=d, L=L, F=Fd, steps=args.steps,
                          params=n_par, mix_params=n_mix,
                          active_mix=model.n_active_mix_params(T), vocab=vocab,
                          corpus_bytes=os.path.getsize(args.data),
                          tokens_per_step=args.batch * args.grad_accum * T,
                          seconds=total, device=str(dev), dtype=args.dtype,
                          args=vars(args))
    with open(os.path.join(args.outdir, tag + ".final.json"), "w") as f:
        json.dump(dict(history=hist, **extras), f, indent=1)
    print("[%s] done in %.0fs; best val %.4f  (%.2f steps/s)"
          % (tag, total, min(r["val_loss"] for r in hist),
             args.steps / max(1e-9, total)), flush=True)
    return hist


# --------------------------------------------------------------------------
# numerical equivalence check against the upstream NumPy implementation
# --------------------------------------------------------------------------

def run_equivalence(args):
    z = np.load(args.equiv_npz, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    dev = torch.device(args.device)
    dt = torch.float64
    cfg = dict(V=int(meta["V"]), d=int(meta["d"]), F=int(meta["F"]), L=int(meta["L"]),
               T_max=int(meta["T_max"]), mix=meta["mix"], heads=int(meta["heads"]),
               pos=meta["pos"], seed=int(meta["seed"]), wo_zero=bool(meta.get("wo_zero", False)),
               loss_eps=1e-9,
               upstream_bug=(args.equiv_bug == "upstream"))
    model = Model(cfg).to(dev).to(dt)
    model.load_numpy_params(z)
    x = torch.as_tensor(z["x"], device=dev)
    mask = torch.as_tensor(z["mask"], device=dev)
    labels = torch.as_tensor(z["labels"], device=dev)
    loss, acc, _ = model(x, mask, labels)
    ref_loss = float(z["loss"])
    loss.backward()
    grads = model.numpy_named_grads()
    rows = []
    worst = 0.0
    gnames = sorted(k[2:] for k in z.files if k.startswith("G_"))
    gscale = math.sqrt(sum(float(np.sum(np.asarray(z["G_" + k]) ** 2)) for k in gnames)
                       / max(1, sum(np.asarray(z["G_" + k]).size for k in gnames)))
    for name in gnames:
        ana = np.asarray(z["G_" + name], dtype=np.float64)
        got = grads[name].detach().cpu().numpy().astype(np.float64)
        dmax = float(np.abs(ana - got).max())
        denom = float(np.abs(ana).max()) + 1e-30
        rows.append((name, ana.shape, dmax, dmax / denom))
        worst = max(worst, dmax)
    print("== equivalence vs upstream NumPy (%s, T=%d d=%d L=%d F=%d heads=%d)"
          % (meta["mix"], meta["T"], meta["d"], meta["L"], meta["F"], meta["heads"]))
    print("   loss  numpy=%.12f  torch=%.12f  |diff|=%.3e"
          % (ref_loss, float(loss), abs(ref_loss - float(loss))))
    print("   acc   numpy=%.12f  torch=%.12f" % (float(z["acc"]), acc))
    print("   %-10s %-14s %12s %12s" % ("param", "shape", "max|dg|", "rel"))
    for name, shape, dmax, rel in rows:
        print("   %-10s %-14s %12.3e %12.3e" % (name, str(tuple(shape)), dmax, rel))
    print("   |grad|rms(numpy)=%.4e   worst max|dg|=%.3e   global_rel=%.3e"
          % (gscale, worst, worst / max(gscale, 1e-30)))
    ok = (abs(ref_loss - float(loss)) < 1e-12
          and worst / max(gscale, 1e-30) < 1e-12)
    print("EQUIVALENCE:", "PASS" if ok else "FAIL",
          "(criteria: |dloss| < 1e-12 and gradient global_rel < 1e-12, float64)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="spectral",
                    choices=["spectral", "fnet", "attn", "stencil", "gated"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--t_max", type=int, default=768)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--F", type=int, default=384)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--stencil_k", type=int, default=2)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--grad_accum", type=int, default=1)
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
    ap.add_argument("--wo_zero", action="store_true")
    ap.add_argument("--extrapolate", type=int, nargs="*", default=[])
    ap.add_argument("--save_weights", action="store_true")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--equiv_npz", default=None,
                    help="run the upstream-equivalence check against this dump")
    ap.add_argument("--equiv_bug", default="upstream", choices=["upstream", "correct"],
                    help="with --equiv_npz: replicate upstream's buggy W1 gradient, "
                         "or use the mathematically correct one")
    ap.add_argument("--upstream_bug", action="store_true",
                    help="replicate upstream's buggy W1 gradient (pre-gain LN "
                         "activation instead of post-gain)")
    args = ap.parse_args()
    if args.equiv_npz:
        raise SystemExit(run_equivalence(args))
    assert args.data, "--data is required"
    train(args)


if __name__ == "__main__":
    main()
