#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen-connectome reservoir: scan, trained linear readout, metrics.

The reservoir is a fixed nonlinear dynamical system driven by token identity:

    h_t = tanh( W @ (leak * h_{t-1} + (1 - leak) * u_t) ),   u_t = U[:, tok_t]

with `W` the (frozen) wiring matrix and `U` a (frozen) random input projection.
Only the readout `logits_t = h_t @ Wout + b` is trained, by cross-entropy with
Adam. This is textbook reservoir computing: the wiring is never touched, so any
difference between graphs is a property of the graph, not of an optimiser.

Because the recurrence is sequential, the corpus is processed as many
independent chunks run in parallel (a batch dimension). Each chunk starts from
h=0, so the first `warmup` steps of every chunk are discarded -- with leak*l_d
< 1 the transient decays geometrically, so a few tens of steps is ample.
"""
import numpy as np
import torch


# ---------------------------------------------------------------- construction

def make_input(N, V, scale=1.0, seed=0, device="cpu", dtype=torch.float32):
    """Frozen random input projection U of shape (N, V)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = torch.randn(N, V, generator=g) * scale
    return U.to(device=device, dtype=dtype)


def to_tensor_W(W, device="cpu", dtype=torch.float32):
    """Dense (N,N) torch tensor from a sparse or dense matrix."""
    import scipy.sparse as sp
    if sp.issparse(W):
        W = W.toarray()
    return torch.as_tensor(np.asarray(W), device=device, dtype=dtype)


# ---------------------------------------------------------------- the scan

@torch.no_grad()
def scan(W, U, x, leak=0.6, warmup=0, reverse=False):
    """Run the reservoir over a batch of sequences.

    W : (N,N) tensor with W[post, pre] (the convention used throughout this
        experiment). The update is
            h_post <- tanh( sum_pre W[post, pre] * drive_pre ),
        i.e. `drive @ W.T`, NOT `drive @ W`.
    U : (N,V) input projection
    x : (B,L) long tensor of token ids
    Returns H : (B, L - warmup, N) tensor of states (positions < warmup dropped).
    """
    B, L = x.shape
    N = W.shape[0]
    dev, dt = W.device, W.dtype
    WT = W.t().contiguous()
    h = torch.zeros(B, N, device=dev, dtype=dt)
    idx = torch.arange(L, device=dev)
    if reverse:
        idx = torch.flip(idx, [0])
    outs = []
    for t in range(L):
        u = U.index_select(1, x[:, idx[t]]).t()          # (B,N)
        h = torch.tanh((leak * h + (1.0 - leak) * u) @ WT)
        outs.append(h)
    H = torch.stack(outs, dim=1)                          # (B,L,N)
    if reverse:
        H = torch.flip(H, [1])
    return H[:, warmup:] if warmup else H


def make_chunks(ids, L):
    """Split a 1-D id array into non-overlapping chunks of length L: (n_chunks, L)."""
    n = (len(ids) // L) * L
    return ids[:n].reshape(-1, L).copy()


# ---------------------------------------------------------------- readout

def fit_readout(X, Y, V, epochs=60, lr=0.05, wd=1e-5, batch=4096,
                seed=0, device="cpu", verbose=False, Xval=None, Yval=None):
    """Train a linear softmax readout by cross-entropy (Adam). Returns (W,b)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt = torch.as_tensor(X, device=device, dtype=torch.float32)
    Yt = torch.as_tensor(Y, device=device, dtype=torch.long)
    F = Xt.shape[1]
    W = torch.zeros(F, V, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(V, device=device, dtype=torch.float32, requires_grad=True)
    torch.nn.init.normal_(W, std=0.01)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    n = Xt.shape[0]
    best, best_state, no_improve = float("inf"), None, 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            logits = Xt[j] @ W + b
            loss = torch.nn.functional.cross_entropy(logits, Yt[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        if Xval is not None:
            vl = loss_of(W, b, Xval, Yval, device)
            if vl < best - 1e-5:
                best, no_improve = vl, 0
                best_state = (W.detach().clone(), b.detach().clone())
            else:
                no_improve += 1
                if no_improve >= 8:
                    break
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print("    epoch %3d train %.4f%s"
                  % (ep, tot / n, "" if Xval is None else "  val %.4f" % best))
    if best_state is not None:
        W, b = best_state
    return W.detach(), b.detach()


@torch.no_grad()
def loss_of(W, b, X, Y, device="cpu", batch=16384):
    Xt = torch.as_tensor(X, device=device, dtype=torch.float32)
    Yt = torch.as_tensor(Y, device=device, dtype=torch.long)
    tot = 0.0
    for i in range(0, Xt.shape[0], batch):
        logits = Xt[i:i + batch] @ W + b
        tot += float(torch.nn.functional.cross_entropy(
            logits, Yt[i:i + batch], reduction="sum"))
    return tot / Xt.shape[0]


@torch.no_grad()
def metrics(W, b, X, Y, device="cpu", batch=16384):
    """Returns (loss_nats, accuracy) over the supplied positions."""
    Xt = torch.as_tensor(X, device=device, dtype=torch.float32)
    Yt = torch.as_tensor(Y, device=device, dtype=torch.long)
    tot, correct = 0.0, 0
    for i in range(0, Xt.shape[0], batch):
        logits = Xt[i:i + batch] @ W + b
        tot += float(torch.nn.functional.cross_entropy(
            logits, Yt[i:i + batch], reduction="sum"))
        correct += int((logits.argmax(-1) == Yt[i:i + batch]).sum())
    n = Xt.shape[0]
    return tot / n, correct / n


def pick_device(prefer="cuda"):
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------- MLP readout
# A one-hidden-layer readout is standard reservoir-computing practice and is used
# here purely as a CAPACITY CONTROL: if the MLP readout also cannot approach the
# 4-gram reference, the limit is the reservoir's features rather than the linear
# readout. The wiring stays frozen either way.

class MLPReadout(torch.nn.Module):
    def __init__(self, F, V, hidden=512, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.fc1 = torch.nn.Linear(F, hidden)
        self.fc2 = torch.nn.Linear(hidden, V)
        with torch.no_grad():
            self.fc1.weight.normal_(0, 1.0 / max(1.0, F ** 0.5), generator=g)
            self.fc1.bias.zero_()
            self.fc2.weight.normal_(0, 0.01, generator=g)
            self.fc2.bias.zero_()

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def fit_readout_mlp(X, Y, V, hidden=512, epochs=60, lr=0.01, wd=1e-5,
                    batch=4096, seed=0, device="cpu", Xval=None, Yval=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt = torch.as_tensor(X, device=device, dtype=torch.float32)
    Yt = torch.as_tensor(Y, device=device, dtype=torch.long)
    mod = MLPReadout(Xt.shape[1], V, hidden=hidden, seed=seed).to(device)
    opt = torch.optim.Adam(mod.parameters(), lr=lr, weight_decay=wd)
    n = Xt.shape[0]
    best, best_state, no_improve = float("inf"), None, 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            loss = torch.nn.functional.cross_entropy(mod(Xt[j]), Yt[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if Xval is not None:
            vl, _a = metrics_module(mod, Xval, Yval, device)
            if vl < best - 1e-5:
                best, no_improve = vl, 0
                best_state = {k: v.detach().clone() for k, v in mod.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= 8:
                    break
    if best_state is not None:
        mod.load_state_dict(best_state)
    mod.eval()
    return mod


@torch.no_grad()
def metrics_module(mod, X, Y, device="cpu", batch=16384):
    Xt = torch.as_tensor(X, device=device, dtype=torch.float32)
    Yt = torch.as_tensor(Y, device=device, dtype=torch.long)
    tot, correct = 0.0, 0
    for i in range(0, Xt.shape[0], batch):
        logits = mod(Xt[i:i + batch])
        tot += float(torch.nn.functional.cross_entropy(
            logits, Yt[i:i + batch], reduction="sum"))
        correct += int((logits.argmax(-1) == Yt[i:i + batch]).sum())
    n = Xt.shape[0]
    return tot / n, correct / n
