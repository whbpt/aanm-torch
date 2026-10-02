#!/usr/bin/env python3
"""AANM endpoint-to-endpoint path with atom reconstruction + clash metrics.

Torch rewrite of designer/machine/scripts/aanm_path.py (ProDy backend replaced
by aanm_torch; runs on CUDA/MPS/CPU via --device).

Usage: python scripts/aanm_path.py <stateA.pdb> <stateB.pdb> <n_steps> [out_prefix]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aanm_torch import pick_device
from aanm_torch.adaptive import calc_adaptive_anm_one_way
from aanm_torch.backbone import reconstruct_backbone
from aanm_torch.clash import clash_count
from aanm_torch.geom import rmsd
from aanm_torch.pdbio import nodes_of, parse_atoms, suggest_cutoff, write_models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stateA")
    ap.add_argument("stateB")
    ap.add_argument("n_steps", type=int)
    ap.add_argument("out_prefix", nargs="?", default="aanm")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu (default auto)")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                    help="float64 matches ProDy numerically; float32 is faster on GPU")
    ap.add_argument("--n-modes", type=int, default=10)
    ap.add_argument("--f", type=float, default=0.3, help="AANM step size")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="ANM cutoff in A (default: auto — 4x median node spacing,"
                         " ~15 for protein CA, ~25 for nucleic P)")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--solver", default="auto", choices=["auto", "eigh", "lobpcg", "eigsh"])
    ap.add_argument("--fmin-max", type=float, default=0.6)
    ap.add_argument("--target-rmsd", type=float, default=1.0)
    ap.add_argument("--min-rmsd-diff", type=float, default=0.05)
    ap.add_argument("--aligned", action="store_true",
                    help="inputs are already optimally superposed; skip the initial fit")
    ap.add_argument("--sidechains", action="store_true",
                    help="parse all atoms and reconstruct whole residues (not just N/CA/C/O)")
    ap.add_argument("--blend", type=float, default=0.2,
                    help="A->B source blend window as path fraction (0 = hard switch)")
    ap.add_argument("--quiet", action="store_true", help="no per-step progress lines")
    ap.add_argument("--allow-empty", action="store_true",
                    help="write the output even if reconstruction produced no atoms")
    args = ap.parse_args()

    device = pick_device(args.device, args.dtype)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"device: {device}, dtype: {args.dtype}")

    atomsA, _ = parse_atoms(args.stateA, bb_only=not args.sidechains)
    atomsB, _ = parse_atoms(args.stateB, bb_only=not args.sidechains)
    nodesA = nodes_of(atomsA)
    nodesB = nodes_of(atomsB)
    common = sorted(set(nodesA) & set(nodesB))
    cutoff = args.cutoff if args.cutoff is not None else suggest_cutoff(
        {k: nodesA[k] for k in common})
    print(f"cutoff: {cutoff:.1f} A (auto)" if args.cutoff is None else f"cutoff: {cutoff:.1f} A")
    coordsA = torch.tensor(np.array([nodesA[k] for k in common]), dtype=dtype, device=device)
    coordsB = torch.tensor(np.array([nodesB[k] for k in common]), dtype=dtype, device=device)
    print(f"common nodes (CA/P): {len(common)}, total rmsd A->B: "
          f"{float(rmsd(coordsA, coordsB)):.2f} A")

    frames, info = calc_adaptive_anm_one_way(
        coordsA, coordsB, args.n_steps, n_modes=args.n_modes, f=args.f,
        cutoff=cutoff, gamma=args.gamma, Fmin_max=args.fmin_max,
        target_rmsd=args.target_rmsd, min_rmsd_diff=args.min_rmsd_diff,
        aligned=args.aligned, solver=args.solver, verbose=not args.quiet)
    if info["converged"]:
        print(f"AANM converged early after {len(frames)} steps "
              f"(rmsd {info['rmsds'][-1]:.2f} A)")
    frames.append(coordsB.clone())
    print(f"AANM frames: {len(frames)}")

    # node-level clash metric per frame (<3.5 A between non-neighbor nodes)
    print("frame | rmsd_to_B | node_clashes(<3.5A) | worst")
    for i, fr in enumerate(frames):
        r = float(rmsd(fr, coordsB))
        c, w = clash_count(fr, cutoff=3.5, skip=2)
        print(f"  {i:3d} | {r:9.2f} | {c:15d} | {w:.2f}")

    # baseline: linear interpolation CA clashes at midpoint
    mid = (coordsA + coordsB) / 2
    c, w = clash_count(mid, cutoff=3.5, skip=2)
    print(f"  linear midpoint: node clashes {c} worst {w:.2f}")

    # atom reconstruction + clash metric; covalent neighbours are excluded via
    # residue groups (sidechain mode) or a 4-atom list offset (backbone mode)
    min_atoms = 1 if args.sidechains else 4
    bbframes = reconstruct_backbone(frames, atomsA, atomsB, common, device, dtype,
                                    blend=args.blend, min_atoms=min_atoms)
    print("clash scan (<1.6A, covalent neighbours skipped) per frame:")
    order = [k for k in atomsA if k in bbframes[-1]]
    if not order:
        order = list(bbframes[-1].keys())
    res_ids = {}
    groups = [res_ids.setdefault(k[:3], len(res_ids)) for k in order]
    for i, bb in enumerate(bbframes):
        if not bb:
            print(f"  frame {i}: no atoms reconstructed")
            continue
        pts = torch.tensor(np.array([bb[k][0] for k in order if k in bb]),
                           dtype=dtype, device=device)
        g = torch.tensor([groups[j] for j, k in enumerate(order) if k in bb],
                         device=device) if args.sidechains else None
        c, w = clash_count(pts, cutoff=1.6, skip=4 if g is None else 1, groups=g)
        print(f"  frame {i}: {c} clashes, worst {w:.2f} A")

    if not bbframes[-1] and not args.allow_empty:
        sys.exit("error: reconstruction produced no atoms; use --allow-empty to write anyway")
    out = f"{args.out_prefix}_frames.pdb"
    write_models(out, bbframes, order)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
