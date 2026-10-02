#!/usr/bin/env python3
"""AANM-engine morph trajectory builder (torch rewrite).

Two elastic-network scopes (config `aanm.network`):
  "anchor"   AANM path on the anchor chain only (default; config
             'aanm.anchor_chain', default first chain); its reconstructed
             backbone is copied to every chain via per-chain site transforms —
             assumes homo-oligomer chains share the same sequence.
  "assembly" one AANM network over the CAs of all chains shared by the two
             states, so relative inter-chain motion (rotations, breathing)
             comes out of the soft modes; each chain is reconstructed
             individually from the shared path.

Stepping modes (config `aanm.mode`): "oneway" (default) or "alternating"
(deform both endpoints toward each other; path includes both exact endpoints).
Whole residues are reconstructed when config `sidechains` is true (atom
positions ride rigidly on their residue's neighbour-CA local frame; the
elastic network still runs on CAs). Caps (chains absent in some states) slide
in/out along z if configured.

Transition paths are computed in lockstep with a single batched eig call when
possible (config `batch`: "auto" batches on CUDA; dense solver regime only).

Accepts the same config JSON as the original script, e.g.
    python scripts/make_aanm_trajectory.py --config machines/serca/config_aanm.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aanm_torch import pick_device
from aanm_torch.adaptive import calc_adaptive_anm_alternating, calc_adaptive_anm_batched, calc_adaptive_anm_one_way
from aanm_torch.backbone import reconstruct_backbone
from aanm_torch.clash import clash_count
from aanm_torch.geom import kabsch, rot_interp
from aanm_torch.pdbio import nodes_of, parse_atoms, suggest_cutoff, write_models


def resample(path, n):
    if len(path) >= n:
        idx = np.linspace(0, len(path) - 1, n).astype(int)
        return [path[i] for i in idx]
    out = []
    for i in range(n):
        x = i / (n - 1) * (len(path) - 1)
        j = int(x)
        w = x - j
        j2 = min(j + 1, len(path) - 1)
        out.append((1 - w) * path[j] + w * path[j2])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu (default auto)")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--quiet", action="store_true", help="no per-step progress lines")
    ap.add_argument("--allow-empty", action="store_true",
                    help="write the output even if some morph frames are empty")
    args = ap.parse_args()

    device = pick_device(args.device, args.dtype)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    verbose = not args.quiet
    print(f"device: {device}, dtype: {args.dtype}")

    cfg = json.load(open(args.config))
    d = cfg["dir"]
    states = cfg["states"]
    dwell_n = cfg.get("dwell", 18)
    morph_n = cfg.get("morph", 20)
    loop = cfg.get("loop", True)
    aanm = cfg.get("aanm", {})
    n_steps = aanm.get("n_steps", 30)
    f_step = aanm.get("f", 0.3)
    n_modes = aanm.get("n_modes", 8)
    anchor_chain = aanm.get("anchor_chain", None)
    network = aanm.get("network", "anchor")
    step_mode = aanm.get("mode", "oneway")
    blend = aanm.get("blend", 0.2)
    solver = aanm.get("solver", "auto")
    cutoff = aanm.get("cutoff", None)  # None -> auto from node spacing
    gamma = aanm.get("gamma", 1.0)
    fmin_max = aanm.get("fmin_max", 0.6)
    target_rmsd = aanm.get("target_rmsd", 1.0)
    min_rmsd_diff = aanm.get("min_rmsd_diff", 0.05)
    sidechains = cfg.get("sidechains", False)
    allow_empty = args.allow_empty or cfg.get("allow_empty", False)
    slide = cfg.get("slides", {})
    batch_cfg = cfg.get("batch", "auto")

    min_atoms = 1 if sidechains else 4
    data = {}
    nodesets = {}
    for st in states:
        fn = f"{d}/{st['file']}"
        atoms, _ = parse_atoms(fn, bb_only=not sidechains)
        data[st["name"]] = atoms
        nodesets[st["name"]] = nodes_of(atoms)
        print(f"{st['name']}: {len(atoms)} {'atoms' if sidechains else 'backbone atoms'}, "
              f"{len(nodesets[st['name']])} network nodes")

    names = [s["name"] for s in states]
    chains = {nm: sorted({k[0] for k in data[nm].keys()}) for nm in names}
    if anchor_chain is None:
        anchor_chain = sorted(chains[names[0]])[0]
    print(f"network: {network}, stepping: {step_mode}"
          + (f", anchor chain: {anchor_chain}" if network == "anchor" else ""))

    if cutoff is None:
        cutoff = suggest_cutoff(nodesets[names[0]])
        print(f"cutoff: {cutoff:.1f} A (auto)")

    # --- per-transition AANM paths ---
    transitions = [(names[i], names[(i + 1) % len(names)]) for i in range(len(names) - 1)]
    if loop:
        transitions.append((names[-1], names[0]))

    commons = {}
    for a, b in transitions:
        na = {k: v for k, v in nodesets[a].items()
              if network == "assembly" or k[0] == anchor_chain}
        nb = {k: v for k, v in nodesets[b].items()
              if network == "assembly" or k[0] == anchor_chain}
        commons[(a, b)] = sorted(set(na) & set(nb))

    def coords_of(nm, common):
        ns = nodesets[nm]
        return torch.tensor(np.array([ns[k] for k in common]), dtype=dtype, device=device)

    def step_kw():
        return dict(n_modes=n_modes, f=f_step, cutoff=cutoff, gamma=gamma,
                    Fmin_max=fmin_max, target_rmsd=target_rmsd,
                    min_rmsd_diff=min_rmsd_diff, solver=solver, verbose=verbose)

    paths = {}
    do_batch = (step_mode == "oneway" and
                (batch_cfg is True or (batch_cfg == "auto" and device.type == "cuda")))
    if do_batch:
        by_n = {}
        for tr in transitions:
            by_n.setdefault(len(commons[tr]), []).append(tr)
        for n, trs in by_n.items():
            pairs = [(coords_of(a, commons[(a, b)]), coords_of(b, commons[(a, b)]))
                     for a, b in trs]
            labels = [f"{a}->{b}" for a, b in trs]
            results = calc_adaptive_anm_batched(pairs, n_steps, labels=labels, **step_kw())
            for (a, b), (A, B), (frames, info) in zip(trs, pairs, results):
                frames.append(coords_of(b, commons[(a, b)]).clone())
                paths[(a, b)] = (commons[(a, b)], resample(frames, morph_n + 1))
                r0 = float(torch.sqrt(((A - B) ** 2).sum() / len(commons[(a, b)])))
                print(f"{a}->{b}: rmsd {r0:.2f} A -> {len(paths[(a, b)][1])} resampled frames")
    else:
        for a, b in transitions:
            common = commons[(a, b)]
            A = coords_of(a, common)
            B = coords_of(b, common)
            r0 = float(torch.sqrt(((A - B) ** 2).sum() / len(common)))
            if step_mode == "alternating":
                frames, info = calc_adaptive_anm_alternating(A, B, n_steps, **step_kw())
            else:
                frames, info = calc_adaptive_anm_one_way(A, B, n_steps, **step_kw())
                frames.append(B.clone())
            paths[(a, b)] = (common, resample(frames, morph_n + 1))
            print(f"{a}->{b}: rmsd {r0:.2f} A -> {len(paths[(a, b)][1])} resampled frames")

    # --- backbone frames along each path ---
    bbframes = {}
    for a, b in transitions:
        common, path = paths[(a, b)]
        bbframes[(a, b)] = reconstruct_backbone(path, data[a], data[b],
                                                 common, device, dtype,
                                                 blend=blend, min_atoms=min_atoms)

    frames = []
    scan = []  # clash-scan flags: skip duplicate dwell frames

    def dwell(nm):
        frames.append(dict(data[nm]))

    def morph(a, b):
        common, path = paths[(a, b)]
        bbf = bbframes[(a, b)]
        cha, chb = set(chains[a]), set(chains[b])
        # site transforms per chain: S_0 ~ c_A @ R.T + t ; S_n ~ c_B @ R.T + t
        # anchor network keys residues without the chain id so homo-oligomer
        # chains match the anchor's residue numbering; assembly keeps the chain
        if network == "assembly":
            def reskey(k):
                return k[:3]
        else:
            def reskey(k):
                return k[1:3]
        site = {}
        S0r = {reskey(k): v for k, v in zip(common, path[0])}
        Snr = {reskey(k): v for k, v in zip(common, path[-1])}
        for c in cha | chb:
            fitA = fitB = None
            if c in cha:
                ca_r = {reskey(k): v[0] for k, v in data[a].items()
                        if k[0] == c and k in nodesets[a]}
                keys = sorted(set(ca_r) & set(S0r))
                if len(keys) >= 30:
                    T = torch.stack([S0r[k] for k in keys])
                    M = torch.tensor(np.array([ca_r[k] for k in keys]), dtype=dtype, device=device)
                    fitA = kabsch(T, M)
            if c in chb:
                ca_r = {reskey(k): v[0] for k, v in data[b].items()
                        if k[0] == c and k in nodesets[b]}
                keys = sorted(set(ca_r) & set(Snr))
                if len(keys) >= 30:
                    T = torch.stack([Snr[k] for k in keys])
                    M = torch.tensor(np.array([ca_r[k] for k in keys]), dtype=dtype, device=device)
                    fitB = kabsch(T, M)
            site[c] = (fitA, fitB)
        for i in range(1, morph_n + 1):
            f = i / morph_n
            fr = {}
            bbframe = bbf[i]
            # chains present in both endpoints: anchor backbone + site transform
            for c in cha & chb:
                fitA, fitB = site[c]
                if fitA is None or fitB is None:
                    continue
                RA, tA = fitA
                RB, tB = fitB
                Rc = rot_interp(RA, RB, f)
                tc = (1 - f) * tA + f * tB
                if network == "anchor":
                    bb_keys = list(bbframe.keys())
                else:
                    bb_keys = [k for k in bbframe if k[0] == c]
                if not bb_keys:
                    continue
                P = torch.tensor(np.array([bbframe[k][0] for k in bb_keys]),
                                 dtype=dtype, device=device)
                moved = ((P - tc) @ Rc).cpu().numpy()
                for j, k in enumerate(bb_keys):
                    if network == "anchor":
                        _, r, ins, an = k
                        fr[(c, r, ins, an)] = (moved[j], bbframe[k][1])
                    else:
                        fr[k] = (moved[j], bbframe[k][1])
            # chains only in b (appearing): slide or pop
            for c in chb - cha:
                st = slide.get("bind", {}).get(c)
                for k, (p, rn) in data[b].items():
                    if k[0] == c:
                        if st is not None:
                            z = p[2] + (st - p[2]) * (1 - f)
                            fr[k] = (np.array([p[0], p[1], z]), rn)
                        elif f >= 0.5:
                            fr[k] = (p.copy(), rn)
            # chains only in a (disappearing): slide or pop
            for c in cha - chb:
                st = slide.get("release", {}).get(c)
                for k, (p, rn) in data[a].items():
                    if k[0] == c:
                        if st is not None:
                            z = p[2] + (st - p[2]) * f
                            fr[k] = (np.array([p[0], p[1], z]), rn)
                        elif f < 0.5:
                            fr[k] = (p.copy(), rn)
            frames.append(fr)
            scan.append(True)

    for i, nm in enumerate(names):
        for j in range(dwell_n):
            dwell(nm)
            scan.append(j == 0)  # scan the first copy of each dwell block only
        nxt = names[(i + 1) % len(names)]
        if i < len(names) - 1 or loop:
            morph(nm, nxt)

    empty = [i for i, fr in enumerate(frames) if not fr]
    if empty and not allow_empty:
        sys.exit(f"error: {len(empty)} empty frames (e.g. {empty[:5]}); reconstruction "
                 f"produced no atoms — check inputs or pass --allow-empty")

    # --- write multi-model PDB ---
    ref_order = list(data[names[-1]].keys())
    out_path = f"{d}/{cfg.get('out', cfg['name'] + '_aanm_trajectory.pdb')}"
    write_models(out_path, frames, ref_order)
    size = os.path.getsize(out_path)
    print(f"trajectory: {out_path}, {len(frames)} frames, {size / 1e6:.1f} MB")

    # --- clash report (duplicate dwell frames skipped; covalent neighbours
    # excluded via residue groups) ---
    print("clash scan (backbone <1.6A, <2.0A):")
    res_ids = {}
    groups = [res_ids.setdefault(k[:3], len(res_ids)) for k in ref_order]
    g_t = torch.tensor(groups, device=device)
    worst = 0
    for i, fr in enumerate(frames):
        if not scan[i]:
            continue
        pts = torch.tensor(np.array([fr[k][0] for k in ref_order if k in fr]),
                           dtype=dtype, device=device)
        g = g_t[[j for j, k in enumerate(ref_order) if k in fr]]
        c16, _ = clash_count(pts, 1.6, 1, groups=g)
        c20, _ = clash_count(pts, 2.0, 1, groups=g)
        if c16 or c20:
            print(f"  frame {i}: <1.6A:{c16} <2.0A:{c20}")
            worst = max(worst, c16)
    print("max clashes in any frame:", worst)


if __name__ == "__main__":
    main()
