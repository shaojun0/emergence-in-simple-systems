#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What does the reservoir actually remember?

A reservoir can only beat a 4-gram if its state still carries information about
tokens 1..3 steps back. This probe trains the *same* linear readout on the same
states but with the label shifted by k, i.e. it asks "can a linear function of
h_t recover x_{t-k}?" for k = 1..K. Accuracy above the trivial unigram predictor
means the lag-k token is still decodable from the state.

This is the diagnostic that explains the reservoir's language-model ceiling, and
it is a clean fly-vs-control comparison in its own right: a wiring that forgets
faster simply cannot represent a 4-gram.

Feature collection is streamed group-by-group (and subsampled within each group)
so memory stays bounded on a 2 GB GPU.

    python scripts/probe_memory.py --synthetic --N 500
    python scripts/probe_memory.py --connectome <feather> --N 1000
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flydata  # noqa: E402
import graphs as G  # noqa: E402
import reservoir as R  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DEFAULT_CORPUS = os.path.join(REPO, "experiments", "spectral-mlm-cpu",
                              "tinyshakespeare.txt")


def collect_lagged(Wt, U, ids, args, dev, leak, lags):
    """Stream the corpus, returning per-lag (features, labels) subsamples."""
    L = args.chunk
    chunks = R.make_chunks(ids, L)
    nc = chunks.shape[0]
    maxlag = max(lags)
    keep_lo, keep_hi = args.warmup, L - maxlag
    per = max(1, keep_hi - keep_lo)
    total = max(1, nc * per)
    Hacc = {k: [] for k in lags}
    Yacc = {k: [] for k in lags}
    stat_sample = []
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)          # explicit: keeps the probe reproducible
    for g in range(0, nc, args.batch):
        grp = torch.as_tensor(chunks[g:g + args.batch], device=dev, dtype=torch.long)
        H = R.scan(Wt, U, grp, leak=leak, warmup=args.warmup)     # (b, L-w, N)
        sl = slice(keep_lo - args.warmup, keep_hi - args.warmup)
        Hs = H[:, sl].reshape(-1, H.shape[-1])                    # (b*per, N)
        base = torch.arange(keep_lo, keep_hi, device=dev)
        if len(stat_sample) < 4:
            stat_sample.append(Hs[:4096].detach().clone())
        want_each = max(1, int(round((args.probe_positions / len(lags)) * len(base) * grp.shape[0] / total)))
        for k in lags:
            Yf = grp[:, base - k].reshape(-1)
            sel = torch.randperm(Hs.shape[0], device=dev, generator=gen)[
                :min(want_each, Hs.shape[0])]
            Hacc[k].append(Hs[sel].cpu())
            Yacc[k].append(Yf[sel].cpu())
        del H, Hs
    lags_data = {k: (torch.cat(Hacc[k]).numpy(), torch.cat(Yacc[k]).numpy())
                 for k in lags}
    stats = state_stats(torch.cat(stat_sample)) if stat_sample else {}
    return lags_data, stats


def state_stats(x):
    x = x.float()
    return dict(mean=float(x.mean()), std=float(x.std()),
                frac_sat=float((x.abs() > 0.99).float().mean()),
                frac_dead=float((x.abs() < 0.01).float().mean()),
                per_unit_std_mean=float(x.std(0).mean()))


def memory_profile(Wt, U, ids, args, dev, leak, lags):
    data, stats = collect_lagged(Wt, U, ids, args, dev, leak, lags)
    prof = {}
    for k in lags:
        X, Y = data[k]
        n_tr = int(len(Y) * 0.8)
        Wo, b = R.fit_readout(X[:n_tr], Y[:n_tr], args.V, epochs=args.probe_epochs,
                              lr=0.05, batch=4096, seed=0, device=dev)
        _l, acc = R.metrics(Wo, b, X[n_tr:], Y[n_tr:], device=dev)
        prof[k] = float(acc)
    return prof, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--connectome", default=None)
    ap.add_argument("--subgraph", default=None,
                    help="derived .npz from export_subgraph.py (no 812 MB download needed)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--leaks", type=float, nargs="*", default=[0.6, 0.9])
    ap.add_argument("--radius", type=float, default=0.95)
    ap.add_argument("--input_scales", type=float, nargs="*", default=[1.0])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--probe_positions", type=int, default=60000)
    ap.add_argument("--probe_epochs", type=int, default=40)
    ap.add_argument("--lags", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = R.pick_device(a.device)
    corp = flydata.load_corpus(a.corpus)
    a.V = corp["V"]
    tr = corp["train"]
    uni = float(np.bincount(tr, minlength=a.V).max() / len(tr))

    if a.synthetic:
        W = G.er_graph(a.N, 6 * a.N, seed=a.seed, signed=True)
        title = "er_synth"
    elif a.subgraph:
        W, _st = G.load_subgraph(a.subgraph)
        print("derived subgraph:", _st)
        title = "fly"
    else:
        W, _ids, st = G.load_flywire(a.connectome, n_keep=a.N, signed=True)
        print("fly subgraph:", st)
        title = "fly"

    Ws, rho0 = G.rescale_to_radius(G.normalise(W, mode="row"), a.radius)
    Wt = R.to_tensor_W(Ws, device=dev)
    print("%s: N=%d nnz=%d rho_raw=%.4f -> %.2f   trivial-unigram acc=%.4f"
          % (title, W.shape[0], W.nnz, rho0, a.radius, uni))

    results = {}
    for leak in a.leaks:
        for iscale in a.input_scales:
            U = R.make_input(W.shape[0], a.V, scale=iscale, seed=a.seed, device=dev)
            prof, st = memory_profile(Wt, U, tr, a, dev, leak, a.lags)
            results["leak%.3f_iscale%.3g" % (leak, iscale)] = dict(
                leak=leak, input_scale=iscale, memory=prof, state=st)
            print("\nleak=%.3f iscale=%-6.3g state std=%.3f sat=%.3f dead=%.3f"
                  % (leak, iscale, st["std"], st["frac_sat"], st["frac_dead"]))
            print("   %-8s %s" % ("lag", " ".join("%6d" % k for k in a.lags)))
            print("   %-8s %s" % ("acc", " ".join("%6.3f" % prof[k] for k in a.lags)))
            print("   %-8s %s" % ("vs uni", " ".join("%+6.3f" % (prof[k] - uni)
                                                     for k in a.lags)))

    out = a.out or os.path.join(REPO, "experiments", "fly-connectome-reservoir",
                                "results", "probe_%s.json" % title)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(graph=title, N=a.N, mode="memory_probe", unigram_acc=uni,
                       lags=a.lags, radius=a.radius, input_scales=a.input_scales,
                       profiles=results), f, ensure_ascii=False, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
