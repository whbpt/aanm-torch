#!/usr/bin/env python3
"""NMFF-style flexible fitting of a PDB into an MRC density map.

Optimises ANM mode coefficients against the real-space cross-correlation by
autograd (see aanm_torch.cryofit). Requires a mode-2 (float32) MRC map.

Usage: python scripts/cryofit.py model.pdb map.mrc [--n-modes 20] \
          [--sigma 2.0] [--iters 300] [--lr 0.2] [--out fitted.pdb]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aanm_torch.cryofit import flexible_fit, read_mrc  # noqa: E402
from aanm_torch.geom import rmsd  # noqa: E402
from aanm_torch.pdbio import nodes_of, parse_atoms  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdb")
    ap.add_argument("mrc")
    ap.add_argument("--n-modes", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--cutoff", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    atoms, order = parse_atoms(args.pdb, bb_only=True)
    nodes = nodes_of(atoms)
    keys = sorted(nodes)
    coords = torch.tensor([nodes[k] for k in keys], dtype=torch.float64)
    grid, voxel, origin = read_mrc(args.mrc)
    print(f"map: {tuple(grid.shape)} voxel {voxel:.2f} A origin {origin}")

    fitted, info = flexible_fit(coords, grid, origin, voxel, sigma=args.sigma,
                                n_modes=args.n_modes, iters=args.iters,
                                lr=args.lr, cutoff=args.cutoff)
    print(f"ccc: {info['ccc_start']:.4f} -> {info['ccc_final']:.4f} | "
          f"node rmsd moved {float(rmsd(coords, fitted)):.2f} A")

    out = args.out or os.path.splitext(args.pdb)[0] + "_fitted.pdb"
    pos = {k: v for k, v in zip(keys, fitted.numpy())}
    with open(out, "w") as f:
        serial = 0
        for k in order:
            serial += 1
            c, r, ins, an = k
            p = pos.get(k, atoms[k][0])
            rn = atoms[k][1]
            f.write("ATOM  " + f"{serial:5d}" + " " + f"{an:^4s}" + " " + f"{rn:>3s}" + " " + c
                    + f"{r:4d}" + ins + "   "
                    + f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
