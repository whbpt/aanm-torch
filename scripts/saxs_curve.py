#!/usr/bin/env python3
"""Compute a differentiable Debye SAXS curve from a PDB (CA nodes by default).

Usage: python scripts/saxs_curve.py in.pdb --qmin 0.01 --qmax 0.5 --nq 100 \
          [--out curve.dat] [--element-weighted]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aanm_torch.pdbio import parse_atoms  # noqa: E402
from aanm_torch.saxs import FORM_FACTORS, saxs_curve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdb")
    ap.add_argument("--qmin", type=float, default=0.01)
    ap.add_argument("--qmax", type=float, default=0.5)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    atoms, _ = parse_atoms(args.pdb)
    coords = torch.tensor([v[0] for v in atoms.values()], dtype=torch.float64)
    if args.element_weighted:
        # crude: residue-name -> dominant element class is not tracked by the
        # parser; use CA-node carbon weights unless atom names are present
        w = torch.tensor([FORM_FACTORS.get("C", 6.0)] * coords.shape[0],
                         dtype=torch.float64)
    else:
        w = None
    q = torch.linspace(args.qmin, args.qmax, args.nq, dtype=torch.float64)
    I = saxs_curve(coords, q, w)
    lines = [f"{float(qq):.5f} {float(ii):.6f}" for qq, ii in zip(q, I)]
    out = args.out or os.path.splitext(args.pdb)[0] + "_saxs.dat"
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out} ({args.nq} points, q {args.qmin}-{args.qmax} 1/A)")


if __name__ == "__main__":
    main()
