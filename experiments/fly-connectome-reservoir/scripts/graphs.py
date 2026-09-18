#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wiring matrices for experiment 3: the FlyWire connectome and its controls.

Convention (following the FLM precedent): `W[post, pre]` is the weight of the
connection from `pre` to `post`, so the reservoir update is
`h_post <- tanh(sum_pre W[post, pre] * drive_pre)`.

Controls, all rescaled to the same spectral radius so that the *dynamics* are
comparable and only the *wiring* differs:

  fly            the FlyWire subgraph
  fly_shuffled   same topology, connection weights permuted
  degree_swap    same topology and same degree sequence, edges rewired by
                 double-edge swaps (destroys everything except the degrees)
  er             Erdos-Renyi with the same node and edge count
"""
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------- spectrum

def spectral_radius(W, absolute=False, iters=300, seed=0, tol=1e-10):
    """Largest |eigenvalue| of W.

    absolute=False (default) returns rho(W), the quantity that actually governs
    the reservoir dynamics `h <- tanh(leak*W h + ...)`, and the correct target
    for rescaling.

    absolute=True returns rho(|W|) by power iteration. For a SIGNED matrix this
    is much larger than rho(W) -- row-normalisation makes the row sums 1 while
    the absolute values do not cancel -- so rescaling by it silently crushes the
    dynamics into the linear regime. It is kept only as a diagnostic.
    """
    if absolute:
        n = W.shape[0]
        v = np.random.RandomState(seed).rand(n).astype(np.float64)
        v /= np.linalg.norm(v) or 1.0
        Wabs = abs(W)
        nrm, last = 0.0, 0.0
        for _ in range(iters):
            u = Wabs @ v
            nrm = float(np.linalg.norm(u))
            if nrm == 0.0:
                return 0.0
            v = u / nrm
            if abs(nrm - last) <= tol * max(1.0, nrm):
                break
            last = nrm
        return nrm

    Wc = W if sp.issparse(W) else np.asarray(W)
    n = Wc.shape[0]
    if n <= 1500:
        ev = np.linalg.eigvals(Wc.toarray() if sp.issparse(Wc) else Wc)
        return float(np.abs(ev).max())
    import scipy.sparse.linalg as spla
    ev = spla.eigs(sp.csr_matrix(Wc), k=1, which="LM", return_eigenvectors=False,
                   maxiter=5000)
    return float(np.abs(ev).max())


def rescale_to_radius(W, target, seed=0):
    """Scale W so that rho(W) == target. Returns (W_scaled, rho_before)."""
    rho = spectral_radius(W, absolute=False, seed=seed)
    if rho <= 0:
        return W, 0.0
    return W * (target / rho), rho


# ---------------------------------------------------------------- normalisation

def normalise(W, mode="row"):
    """row: each postsynaptic neuron's incoming weights sum to 1 (FLM's choice).
    global: keep raw weights (caller rescales by spectral radius).
    """
    if mode == "global":
        return sp.csr_matrix(W, dtype=np.float64) if sp.issparse(W) \
            else np.asarray(W, dtype=np.float64)
    Wc = sp.csr_matrix(W, dtype=np.float64) if not sp.issparse(W) else W.astype(np.float64)
    rs = np.asarray(Wc.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ Wc


# ---------------------------------------------------------------- controls

def shuffle_weights(W, seed=0):
    """Same topology, non-zero weights permuted among the existing edges."""
    Wc = sp.csr_matrix(W)
    rng = np.random.RandomState(seed)
    data = Wc.data.copy()
    rng.shuffle(data)
    return sp.csr_matrix((data, Wc.indices, Wc.indptr), shape=Wc.shape)


def degree_preserving_swap(W, seed=0, n_swap_factor=10, max_attempts_factor=60):
    """Randomise a directed graph by double-edge swaps, preserving BOTH degree sequences.

    Take two edges (b -> a) and (d -> c) and replace them with (b -> c) and
    (d -> a). Out-degrees of b, d and in-degrees of a, c are unchanged, so both
    the in- and the out-degree sequences survive while all topology beyond the
    degrees is destroyed. Weights travel with their edge index.

    Returns (W_randomised, swaps_done, swaps_requested).
    """
    Wc = sp.csr_matrix(W)
    coo = Wc.tocoo()
    ei = coo.row.astype(np.int64)          # post  (target of the edge)
    ej = coo.col.astype(np.int64)          # pre   (source of the edge)
    ew = coo.data.astype(np.float64)
    m = len(ei)
    n = Wc.shape[0]
    rng = np.random.RandomState(seed)
    existing = set(zip(ei.tolist(), ej.tolist()))     # (post, pre)
    n_swap = int(n_swap_factor * m)
    done = attempts = 0
    max_attempts = max_attempts_factor * n_swap + 1000
    while done < n_swap and attempts < max_attempts:
        attempts += 1
        i, j = rng.randint(0, m, size=2)
        if i == j:
            continue
        a, b = ei[i], ej[i]                # edge i:  b -> a
        c, d = ei[j], ej[j]                # edge j:  d -> c
        # new edges: b -> c  and  d -> a
        if b == c or d == a:               # would create self-loops
            continue
        if (c, b) in existing or (a, d) in existing:
            continue
        existing.discard((a, b)); existing.discard((c, d))
        existing.add((c, b)); existing.add((a, d))
        wi, wj = ew[i], ew[j]
        ei[i], ej[i], ew[i] = c, b, wi
        ei[j], ej[j], ew[j] = a, d, wj
        done += 1
    return sp.csr_matrix((ew, (ei, ej)), shape=(n, n)), done, n_swap


def er_graph(n, m, seed=0, signed=False):
    """Erdos-Renyi directed graph with n nodes and m edges (no self-loops)."""
    rng = np.random.RandomState(seed)
    total = n * (n - 1)
    m = int(min(m, total))
    idx = rng.choice(total, size=m, replace=False)
    post = idx // (n - 1)
    pre = idx % (n - 1)
    pre = np.where(pre >= post, pre + 1, pre)
    data = rng.rand(m) + 0.5
    if signed:
        data = data * rng.choice([-1.0, 1.0], size=m)
    return sp.csr_matrix((data, (post, pre)), shape=(n, n))


def row_weight_shuffle(W, seed=0):
    """Keep the topology EXACTLY; permute each neuron's incoming weights.

    Within every postsynaptic neuron the weights sitting on its incoming edges
    are shuffled. Topology, edge count and every row sum (the neuron's total
    incoming weight, i.e. its input gain) are preserved exactly; only the
    assignment of "how strong is the connection from this particular source"
    is destroyed.

    This is deliberately distinct from `shuffle_weights`, which permutes weights
    globally and therefore does NOT preserve row sums.
    """
    Wc = sp.csr_matrix(W).tocsr()
    rng = np.random.RandomState(seed)
    data = Wc.data.copy()
    for i in range(Wc.shape[0]):
        s, e = Wc.indptr[i], Wc.indptr[i + 1]
        if e - s > 1:
            data[s:e] = rng.permutation(data[s:e])
    return sp.csr_matrix((data, Wc.indices.copy(), Wc.indptr.copy()), shape=Wc.shape)


def save_subgraph(path, W, keep_ids, stats):
    """Persist a derived subgraph (+ its neuron ids and provenance) as .npz/.json.

    The whole experiment reduces the 812 MB FlyWire feather to a ~0.5 MB sparse
    matrix, so committing the derived subgraph makes experiment 3 reproducible
    without re-downloading the connectome.
    """
    import json
    sp.save_npz(path, sp.csr_matrix(W))
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(dict(stats=stats, n_kept=int(len(keep_ids)),
                       keep_ids_head=[int(v) for v in keep_ids[:10]]),
                  f, ensure_ascii=False, indent=1)
    return path


def load_subgraph(path):
    """Inverse of save_subgraph. Returns (W, stats)."""
    import json
    W = sp.load_npz(path)
    with open(path + ".json", encoding="utf-8") as f:
        meta = json.load(f)
    return W, meta.get("stats", {})


# ---------------------------------------------------------------- FlyWire

_COLS = ["pre_pt_root_id", "post_pt_root_id", "syn_count",
         "gaba_avg", "ach_avg", "glut_avg"]


def load_flywire(path, n_keep=1000, signed=True, weight="log", min_syn=1):
    """Reduce `proofread_connections_783.feather` to an n_keep subgraph.

    Neurons are ranked by total synapse volume (in + out) and the top `n_keep`
    are kept, which yields the strongest-connected core of the brain rather than
    an arbitrary slice. Rows are W[post, pre] (see module docstring).

    IMPORTANT: the table has one row per (pre, post, NEUROPIL) triple, so the
    same neuron pair can appear many times. Synapse counts must therefore be
    SUMMED PER PAIR BEFORE the log transform -- summing log1p values afterwards
    would be meaningless (it produces log-scale weights of ~40). Transmitter
    probabilities are aggregated as a synapse-count-weighted mean before the sign
    is derived, for the same reason.

    signed=True marks a connection inhibitory (-1) when GABA or glutamate
    dominates acetylcholine, following the usual Drosophila convention. The FLM
    precedent explicitly does NOT use signs, so this is an ablation here.
    """
    import pandas as pd

    cols = ["pre_pt_root_id", "post_pt_root_id", "syn_count",
            "gaba_avg", "ach_avg", "glut_avg"]
    df = pd.read_feather(path, columns=cols)
    n_full, syn_full = len(df), int(df["syn_count"].sum())
    if min_syn > 1:
        df = df[df["syn_count"] >= min_syn]

    pre = df["pre_pt_root_id"].to_numpy()
    post = df["post_pt_root_id"].to_numpy()
    syn = df["syn_count"].to_numpy().astype(np.float64)

    ids, inv = np.unique(np.concatenate([pre, post]), return_inverse=True)
    p = inv[:len(pre)]      # presynaptic index
    q = inv[len(pre):]      # postsynaptic index
    vol = np.bincount(p, weights=syn, minlength=len(ids)) + \
        np.bincount(q, weights=syn, minlength=len(ids))
    order = np.argsort(-vol)[:n_keep]
    n = len(order)

    keep = np.zeros(len(ids), dtype=bool)
    keep[order] = True
    sel = keep[p] & keep[q]

    keep_ids = ids[order]
    remap = np.full(len(ids), -1, dtype=np.int64)
    remap[order] = np.arange(n)

    sub = pd.DataFrame({
        # W[post, pre] -> row = postsynaptic, col = presynaptic
        "r": remap[q[sel]],
        "c": remap[p[sel]],
        "syn": syn[sel],
    })
    for nm in ("gaba", "ach", "glut"):
        sub[nm] = df["%s_avg" % nm].to_numpy()[sel] * syn[sel]
    g = sub.groupby(["r", "c"], sort=False).sum()
    tot = g["syn"].to_numpy()
    w = np.log1p(tot) if weight == "log" else tot
    if signed:
        gaba = g["gaba"].to_numpy() / tot
        ach = g["ach"].to_numpy() / tot
        glut = g["glut"].to_numpy() / tot
        w = np.where((gaba > ach) | (glut > ach), -w, w)

    W = sp.csr_matrix((w, (g.index.get_level_values(0).to_numpy(),
                           g.index.get_level_values(1).to_numpy())), shape=(n, n))
    stats = dict(n_nodes=n, n_edges=int(W.nnz), n_full=n_full,
                 synapses_full=syn_full, n_synapses_kept=int(tot.sum()),
                 mean_syn_per_edge=float(tot.mean()), max_syn_per_edge=int(tot.max()),
                 frac_neg=float((w < 0).mean()) if signed else 0.0)
    return W, keep_ids, stats
