#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage B: prove the torch port reproduces the upstream NumPy implementation
exactly -- same parameters, same batch, same loss, same gradients.

Two regimes:
  * updates=0 : the reference is taken at initialisation, where upstream's
    gradcheck also runs.
  * updates=1 : the reference is taken AFTER one real SGD step.  Here
    upstream's hand-derived W1 gradient (pre-gain LN activation) is wrong, so
    the port must replicate the bug to match; running with --equiv_bug correct
    is expected to FAIL on W1 only, which is the demonstration of the bug.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.path.join(ROOT, "ref")
PY = sys.executable

CASES = [
    # mix,       T,  d,  L, F,  heads, extra
    ("spectral", 24, 16, 2, 32, 4, []),
    ("spectral", 25, 16, 2, 32, 4, []),           # odd T: no Nyquist bin
    ("fnet", 24, 16, 2, 32, 4, []),
    ("attn", 24, 16, 2, 32, 4, []),
    ("attn", 24, 16, 2, 32, 4, ["--wo_zero"]),
]


def dump(mix, T, d, L, F, h, extra, updates, npz):
    cmd = [PY, os.path.join(HERE, "dump_numpy_reference.py"), "--mix", mix,
           "--T", str(T), "--d", str(d), "--L", str(L), "--F", str(F),
           "--heads", str(h), "--updates", str(updates), "--out", npz] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
    return r.stdout.strip()


def check(npz, bug):
    r = subprocess.run([PY, os.path.join(HERE, "spectral_lm_torch.py"),
                        "--equiv_npz", npz, "--device", "cuda",
                        "--equiv_bug", bug], capture_output=True, text=True)
    return r.stdout


def main():
    os.makedirs(REF, exist_ok=True)
    fails, total = 0, 0
    for updates in (0, 1):
        print("=" * 100)
        print("### reference taken after %d upstream SGD update(s)" % updates)
        for mix, T, d, L, F, h, extra in CASES:
            name = "%s_T%d%s_u%d" % (mix, T, "_wozero" if extra else "", updates)
            npz = os.path.join(REF, name + ".npz")
            print(dump(mix, T, d, L, F, h, extra, updates, npz), flush=True)
            out = check(npz, "upstream")
            print("\n".join(l for l in out.splitlines()
                            if "loss  numpy" in l or "EQUIVALENCE" in l
                            or ("0.W1" in l) or "global_rel" in l), flush=True)
            total += 1
            if "PASS" not in out:
                fails += 1
            if updates == 1:
                out2 = check(npz, "correct")
                bad = [l.strip() for l in out2.splitlines()
                       if "EQUIVALENCE" in l or (l.strip().startswith("0.W1"))
                       or "global_rel" in l]
                print("   [--equiv_bug correct] ->", " | ".join(bad), flush=True)
            print("-" * 100, flush=True)
    print("STAGE B: %d/%d upstream-replication cases PASS" % (total - fails, total))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
