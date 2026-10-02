"""Atom reconstruction for CA-only AANM frames.

Places every atom of a residue using a Kabsch frame fitted on the CA of the
same-chain previous/current/next residue, taking atom positions from the
nearer endpoint. Backbone dictionaries yield N/CA/C/O; full-atom dictionaries
yield whole residues (sidechains ride rigidly on their residue's local frame).

Two upgrades over the original script's logic:
  - neighbour slots are same-chain only (the original used sorted-position
    neighbours, so the last residue of chain A borrowed a CA from chain B)
  - the A->B source switch at the path midpoint is blended over `blend`
    fraction of the path instead of hard-switching (blend=0 keeps the switch)
"""
import numpy as np
import torch

from .geom import kabsch_batch


def _nbr_topology(common, res_list):
    """Per-residue (prev, self, next) node indices into `common`, same chain
    only. Node atoms can be protein CA or nucleic P — the node key of each
    residue is simply its own entry in `common`.

    Chain termini have -1 slots and are invalidated by the caller.
    """
    R = len(res_list)
    common_index = {k: i for i, k in enumerate(common)}
    node_of_res = {k[:3]: k for k in common}
    prev_pos = [-1] * R
    last_of_chain = {}
    for p, res in enumerate(res_list):
        ch = res[0]
        if ch in last_of_chain:
            prev_pos[p] = last_of_chain[ch]
        last_of_chain[ch] = p
    next_pos = [-1] * R
    next_of_chain = {}
    for p in range(R - 1, -1, -1):
        ch = res_list[p][0]
        next_pos[p] = next_of_chain.get(ch, -1)
        next_of_chain[ch] = p
    nbr_idx = torch.full((R, 3), -1, dtype=torch.long)
    for p, res in enumerate(res_list):
        nbr_idx[p, 1] = common_index[node_of_res[res]]
        if prev_pos[p] >= 0:
            nbr_idx[p, 0] = common_index[node_of_res[res_list[prev_pos[p]]]]
        if next_pos[p] >= 0:
            nbr_idx[p, 2] = common_index[node_of_res[res_list[next_pos[p]]]]
    return nbr_idx, common_index


def reconstruct_backbone(frames_ca, atomsA, atomsB, common, device=None,
                         dtype=torch.float64, blend=0.2, min_atoms=4):
    """frames_ca: list of (R, 3) CA tensors in `common` order; atomsA/atomsB are
    {(chain, resnum, icode, atom): (pos, resname)} dictionaries.

    Returns a list of {key: (np pos, resname)} dicts, one per frame.
    """
    if device is None:
        device = frames_ca[0].device if torch.is_tensor(frames_ca[0]) else torch.device("cpu")
    res_list = sorted({k[:3] for k in common})
    R = len(res_list)
    nbr_idx, common_index = _nbr_topology(common, res_list)

    def prep(src):
        has_nbrs = (nbr_idx >= 0).all(dim=1)
        by_res = {}
        for k, v in src.items():
            by_res.setdefault(k[:3], []).append((k, v[0], v[1]))
        valid = torch.zeros(R, dtype=torch.bool)
        keys_flat = []
        coords_flat = []
        names_flat = []
        res_pos = []
        for p, res in enumerate(res_list):
            ra = by_res.get(res)
            if ra and len(ra) >= min_atoms and bool(has_nbrs[p]):
                valid[p] = True
                for k, pos, rn in ra:
                    keys_flat.append(k)
                    coords_flat.append(pos)
                    names_flat.append(rn)
                    res_pos.append(p)
        coords = (torch.tensor(np.array(coords_flat), dtype=dtype, device=device)
                  if coords_flat else torch.zeros(0, 3, dtype=dtype, device=device))
        rp = (torch.tensor(res_pos, dtype=torch.long, device=device) if res_pos
              else torch.zeros(0, dtype=torch.long, device=device))
        src_ca = np.zeros((R, 3, 3))
        for p in range(R):
            for s in range(3):
                q = int(nbr_idx[p, s])
                if q >= 0:
                    src_ca[p, s] = src[common[q]][0]
        sel = valid.nonzero().flatten()
        slot = torch.full((R,), -1, dtype=torch.long)
        slot[sel] = torch.arange(sel.numel(), device=device)
        return {"keys": keys_flat, "coords": coords, "names": names_flat,
                "res_pos": rp, "src_ca": torch.tensor(src_ca, dtype=dtype, device=device),
                "sel": sel, "slot": slot}

    prepA = prep(atomsA)
    prepB = prep(atomsB)

    def place(prepP, ca):
        n = prepP["sel"].numel()
        if n == 0:
            return None
        T = ca[nbr_idx[prepP["sel"]]]
        M = prepP["src_ca"][prepP["sel"]]
        Rb, tb = kabsch_batch(T, M)
        cM = M.mean(1)
        cT = T.mean(1)
        a_slot = prepP["slot"][prepP["res_pos"]]
        X = prepP["coords"] - cM[a_slot]
        RbT = Rb[a_slot].transpose(1, 2)
        Pl = torch.einsum("ai,aij->aj", X, RbT) + cT[a_slot]
        return Pl.cpu().numpy()

    out_frames = []
    for f, ca in enumerate(frames_ca):
        frac = f / max(len(frames_ca) - 1, 1)
        if blend > 0:
            h = blend / 2.0
            alpha = min(max((frac - (0.5 - h)) / (2.0 * h), 0.0), 1.0)
        else:
            alpha = 0.0 if frac < 0.5 else 1.0

        PlA = place(prepA, ca) if alpha < 1.0 else None
        PlB = place(prepB, ca) if alpha > 0.0 else None

        bb = {}
        if PlA is not None and PlB is not None:
            idxB = {k: a for a, k in enumerate(prepB["keys"])}
            for a, k in enumerate(prepA["keys"]):
                pA = PlA[a]
                b = idxB.get(k)
                p = (1 - alpha) * pA + alpha * PlB[b] if b is not None else pA
                bb[k] = (p, prepA["names"][a])
            for a, k in enumerate(prepB["keys"]):
                if k not in bb:
                    bb[k] = (PlB[a], prepB["names"][a])
        elif PlA is not None:
            for a, k in enumerate(prepA["keys"]):
                bb[k] = (PlA[a], prepA["names"][a])
        elif PlB is not None:
            for a, k in enumerate(prepB["keys"]):
                bb[k] = (PlB[a], prepB["names"][a])
        out_frames.append(bb)
    return out_frames
