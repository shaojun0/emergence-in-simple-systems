#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment 3: does a frozen fly connectome, used as a reservoir, learn a 4-gram?

Two task framings, both with the SAME frozen wiring and the SAME trained linear
readout, so the only difference between rows of the results table is the graph:

  causal  next-token prediction; context = the whole past.
          Directly comparable to the "4-gram markov" reference, which is also
          directed with 3 tokens of context.
  mlm     experiment 1's masked LM: 15% of tokens corrupted 80/10/10, features
          are the concatenated forward and backward reservoir states, loss is
          taken on masked positions only. Comparable to the experiment-1/2
          neural numbers.

Usage
  python run_experiment.py --synthetic --quick
  python run_experiment.py --connectome <feather> --N 1000 --mode both
"""
import argparse
import json
import os
import sys
import time

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


# ---------------------------------------------------------------- state collection

def _subsample(f, y, want, dev, gen=None):
    """Random subsample of positions, drawn from an EXPLICIT generator.

    The generator is seeded once per run (not per call) and is created
    identically for every graph, so all graphs see the SAME training positions.
    That makes the fly-vs-control comparison paired and the whole run
    reproducible; an unseeded torch.randperm would make both false.
    """
    if want is None or f.shape[0] <= want:
        return f, y
    sel = torch.randperm(f.shape[0], device=dev, generator=gen)[:want]
    return f[sel], y[sel]


def collect_causal(Wt, U, ids, args, dev, rng, leak, keep_n=None, gen=None):
    """Features h_t (after consuming x_t), labels x_{t+1}.

    The corpus is cut into non-overlapping chunks of `chunk` tokens and scanned
    `batch` chunks at a time. Each chunk restarts from h=0, so the first
    `warmup` steps are dropped. Features are subsampled *within each group* so
    that memory stays bounded on large corpora.
    """
    L = args.chunk
    chunks = R.make_chunks(ids, L)                       # (n_chunks, L)
    nc = chunks.shape[0]
    per = L - args.warmup - 1
    total = max(1, nc * per)
    fs, ys = [], []
    for g in range(0, nc, args.batch):
        grp = torch.as_tensor(chunks[g:g + args.batch], device=dev, dtype=torch.long)
        H = R.scan(Wt, U, grp, leak=leak, warmup=args.warmup)        # (b, L-w, N)
        f = H[:, :-1].reshape(-1, H.shape[-1])
        y = grp[:, args.warmup + 1:].reshape(-1)
        want = None if keep_n is None else max(1, int(round(keep_n * len(y) / total)))
        f, y = _subsample(f, y, want, dev, gen)
        fs.append(f.cpu())
        ys.append(y.cpu())
    return torch.cat(fs).numpy(), torch.cat(ys).numpy()


def collect_mlm(Wt, U, ids, args, dev, rng, mask_id, leak, keep_n=None, gen=None):
    """Bidirectional features at masked positions; labels are the clean tokens."""
    V = mask_id + 1                  # corpus layout: mask id == len(vocab), V == +1
    L = args.chunk
    chunks = R.make_chunks(ids, L)
    nc = chunks.shape[0]
    per = L - args.warmup
    total_masked = max(1.0, nc * per * args.mask_prob)
    fs, ys = [], []
    for g in range(0, nc, args.batch):
        grp = chunks[g:g + args.batch]
        xc = grp.copy()
        mask = np.zeros_like(grp, dtype=bool)
        for b in range(xc.shape[0]):
            xc[b], mask[b] = flydata.apply_corruption(
                grp[b], np.random.RandomState(rng.randint(1 << 30)), mask_id, V)
        gt = torch.as_tensor(xc, device=dev, dtype=torch.long)
        Hf = R.scan(Wt, U, gt, leak=leak, warmup=0)
        Hb = R.scan(Wt, U, gt, leak=leak, warmup=0, reverse=True)
        sl = slice(args.warmup, L)
        F = torch.cat([Hf[:, sl], Hb[:, sl]], dim=-1)            # (b, L-w, 2N)
        mk = torch.as_tensor(mask[:, sl], device=dev)
        f = F[mk]
        y = gt[:, sl][mk]
        want = None if keep_n is None else max(1, int(round(keep_n * len(y) / total_masked)))
        f, y = _subsample(f, y, want, dev, gen)
        fs.append(f.cpu())
        ys.append(y.cpu())
    return torch.cat(fs).numpy(), torch.cat(ys).numpy()


# ---------------------------------------------------------------- one reservoir

def _readout_fns(fam, args, dev):
    """Return (fit, measure) for one readout family.

    'linear' is the headline configuration (only a linear map is trained).
    'mlp' is a capacity control: if a one-hidden-layer readout also cannot
    approach the 4-gram reference, the limit is the reservoir, not the readout.
    The wiring stays frozen in both cases.
    """
    if fam == "mlp":
        def fit(Xf, Yf, Xv, Yv, V):
            return R.fit_readout_mlp(Xf, Yf, V, hidden=args.readout_hidden,
                                     epochs=args.epochs, lr=args.lr_mlp,
                                     batch=args.readout_batch, seed=args.seed,
                                     device=dev, Xval=Xv, Yval=Yv)

        def measure(mod, X, Y):
            return R.metrics_module(mod, X, Y, device=dev)
        return fit, measure

    def fit(Xf, Yf, Xv, Yv, V):
        return R.fit_readout(Xf, Yf, V, epochs=args.epochs, lr=args.lr,
                             batch=args.readout_batch, seed=args.seed,
                             device=dev, Xval=Xv, Yval=Yv)

    def measure(model, X, Y):
        return R.metrics(model[0], model[1], X, Y, device=dev)
    return fit, measure


def run_one(name, W, stats, corpus, args, dev, ref):
    """Normalise -> rescale -> scan -> fit readout(s) -> report val metrics.

    Returns a LIST of result dicts, one per readout family. The reservoir is
    scanned once per (leak, input_scale) and shared by every family, because the
    readout choice does not change the states.
    """
    V, mask_id = corpus["V"], corpus["mask_id"]
    tr, va = corpus["train"], corpus["val"]
    n_fit = int(len(tr) * (1.0 - args.readout_val_frac))
    tr_fit, tr_val = tr[:n_fit], tr[n_fit:]

    Wn = G.normalise(W, mode=args.norm)
    Ws, rho0 = G.rescale_to_radius(Wn, args.radius)
    Wt = R.to_tensor_W(Ws, device=dev)

    base = dict(graph=name, N=int(W.shape[0]), nnz=int(stats.get("n_edges", W.nnz)),
                rho_raw=float(rho0), radius=args.radius, norm=args.norm)
    fams = ["linear", "mlp"] if args.readout == "both" else [args.readout]
    fns = {f: _readout_fns(f, args, dev) for f in fams}
    bests = {f: None for f in fams}

    rng = np.random.RandomState(args.seed)
    # Seeded identically for every graph, so all graphs see the same subsampled
    # training positions -> a paired comparison and a reproducible run.
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    for leak in args.leaks:
        for iscale in args.input_scales:
            U = R.make_input(W.shape[0], V, scale=iscale, seed=args.seed, device=dev)
            t0 = time.time()
            if args.mode == "causal":
                Xf, Yf = collect_causal(Wt, U, tr_fit, args, dev, rng, leak,
                                        args.fit_positions, gen)
                Xv, Yv = collect_causal(Wt, U, tr_val, args, dev, rng, leak, None, gen)
            else:
                Xf, Yf = collect_mlm(Wt, U, tr_fit, args, dev, rng, mask_id, leak,
                                     args.fit_positions, gen)
                Xv, Yv = collect_mlm(Wt, U, tr_val, args, dev, rng, mask_id, leak,
                                     None, gen)
            for f in fams:
                model = fns[f][0](Xf, Yf, Xv, Yv, V)
                vl, vacc = fns[f][1](model, Xv, Yv)
                print("    [%s] leak=%.3f iscale=%-5.3g readout-val loss %.4f acc %.4f"
                      % (f, leak, iscale, vl, vacc), flush=True)
                if bests[f] is None or vl < bests[f]["readout_val_loss"]:
                    bests[f] = dict(readout=f, leak=leak, input_scale=iscale,
                                    readout_val_loss=vl, readout_val_acc=vacc,
                                    model=model, U=U)
            print("    (scan+fit %.0fs)" % (time.time() - t0), flush=True)

    out = []
    for f in fams:
        b = bests[f]
        U, model, leak = b.pop("U"), b.pop("model"), b["leak"]
        rng2 = np.random.RandomState(args.seed + 777)
        if args.mode == "causal":
            Xe, Ye = collect_causal(Wt, U, va, args, dev, rng2, leak, None)
        else:
            Xe, Ye = collect_mlm(Wt, U, va, args, dev, rng2, mask_id, leak, None)
        loss, acc = fns[f][1](model, Xe, Ye)
        r = dict(base)
        r.update(b)
        r.update(val_loss=loss, val_acc=acc, n_eval=int(len(Ye)))
        print("  %-18s [%-6s] N=%-5d nnz=%-7d leak=%.3f iscale=%-5.3g  "
              "VAL loss %.4f acc %.4f  (4-gram ref %.4f / %.4f)"
              % (name, f, r["N"], r["nnz"], leak, r["input_scale"], loss, acc,
                 ref["4-gram markov"][0], ref["4-gram markov"][1]), flush=True)
        out.append(r)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--connectome", default=None, help="path to proofread_connections_*.feather")
    ap.add_argument("--subgraph", default=None,
                    help="path to a derived .npz from export_subgraph.py (no 812 MB download needed)")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic graphs only")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--mode", choices=["causal", "mlm"], default="causal")
    ap.add_argument("--leaks", type=float, nargs="*", default=[0.6],
                    help="leak rates to try; effective per-step decay is leak*radius")
    ap.add_argument("--radius", type=float, default=0.95)
    ap.add_argument("--norm", choices=["row", "global"], default="row")
    ap.add_argument("--input_scales", type=float, nargs="*", default=[1.0])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=128)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--fit_positions", type=int, default=150000)
    ap.add_argument("--readout_val_frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--readout", choices=["linear", "mlp", "both"], default="linear")
    ap.add_argument("--readout_hidden", type=int, default=512)
    ap.add_argument("--lr_mlp", type=float, default=0.01)
    ap.add_argument("--readout_batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--signed", action="store_true", default=True)
    ap.add_argument("--unsigned", dest="signed", action="store_false")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.quick:
        a.chunk, a.batch, a.warmup = 512, 8, 64
        a.fit_positions, a.epochs = 20000, 15
        a.N = min(a.N, 300)

    dev = R.pick_device(a.device)
    print("device:", dev, "| torch", R.torch.__version__)
    t_start = time.time()

    corpus = flydata.load_corpus(a.corpus)
    print("corpus: train=%d val=%d V=%d vocab=%d"
          % (len(corpus["train"]), len(corpus["val"]), corpus["V"], len(corpus["vocab"])))
    ref = flydata.reference_baselines(corpus["train"], corpus["val"], corpus["V"])
    print("%-28s %8s %8s" % ("reference", "loss", "acc"))
    for k, (l, ac, _c) in ref.items():
        print("%-28s %8.4f %8.4f" % (k, l, ac))

    todo = []
    if a.synthetic:
        for n in (a.N,):
            m = int(6 * n)                     # ~6 edges per node
            Ws = G.er_graph(n, m, seed=a.seed, signed=a.signed)
            todo.append(("er_synth", Ws, dict(n_edges=m)))
    else:
        if a.subgraph:
            W, st = G.load_subgraph(a.subgraph)
            print("derived subgraph:", st)
        else:
            assert a.connectome, "need --connectome, --subgraph or --synthetic"
            W, ids, st = G.load_flywire(a.connectome, n_keep=a.N, signed=a.signed)
            print("fly subgraph:", st)
        todo.append(("fly", W, st))
        todo.append(("shuffle_weights", G.shuffle_weights(W, seed=a.seed),
                     dict(n_edges=W.nnz)))
        todo.append(("row_weight_shuffle", G.row_weight_shuffle(W, seed=a.seed),
                     dict(n_edges=W.nnz)))
        Wd, done, want = G.degree_preserving_swap(W, seed=a.seed)
        print("degree swap: %d/%d swaps" % (done, want))
        todo.append(("degree_swap", Wd, dict(n_edges=Wd.nnz)))
        todo.append(("er", G.er_graph(W.shape[0], W.nnz, seed=a.seed + 12345,
                                      signed=a.signed), dict(n_edges=W.nnz)))

    results = []
    for nm, W, st in todo:
        results.extend(run_one(nm, W, st, corpus, a, dev, ref))

    summary = dict(mode=a.mode, seed=a.seed, leaks=a.leaks, radius=a.radius,
                   norm=a.norm, N=a.N, signed=a.signed, readout=a.readout,
                   reference={k: dict(loss=v[0], acc=v[1]) for k, v in ref.items()},
                   results=results, seconds=time.time() - t_start)
    out = a.out or os.path.join(REPO, "experiments", "fly-connectome-reservoir",
                                "results", "smoke.json" if a.quick else "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("wrote", out, "(%.0fs)" % (time.time() - t_start))


if __name__ == "__main__":
    sys.exit(main())
