"""Backbone PDB parsing/writing, same fixed-column format as the original scripts."""
import numpy as np

BB = ("N", "CA", "C", "O")
# phosphodiester backbone; P is the network node for nucleotides (the Cα analog)
NUC_BB = ("P", "O5'", "C5'", "C4'", "C3'", "O3'")
BB_ANY = BB + NUC_BB


def parse_atoms(fn, bb_only=False):
    """Parse ATOM records.

    Returns ({(chain, resnum, icode, atom): (pos, resname)}, insertion_order).
    Later duplicates overwrite earlier ones but keep their original position.
    With bb_only=True keep the protein backbone (N/CA/C/O) plus the nucleic
    backbone (P/O5'/C5'/C4'/C3'/O3') — mixed protein–nucleic complexes parse
    in one pass; pure inputs are unaffected by the other set.
    """
    atoms = {}
    order = []
    for ln in open(fn):
        if not ln.startswith("ATOM"):
            continue
        an = ln[12:16].strip()
        if bb_only and an not in BB_ANY:
            continue
        c = ln[21]
        r = int(ln[22:26])
        ins = ln[26]
        resn = ln[17:20]
        key = (c, r, ins, an)
        x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
        if key not in atoms:
            order.append(key)
        atoms[key] = (np.array([x, y, z]), resn)
    return atoms, order


def parse_bb(fn):
    return parse_atoms(fn, bb_only=True)


def ca_of(atoms):
    return {k: v[0] for k, v in atoms.items() if k[3] == "CA"}


def nodes_of(atoms):
    """One elastic-network node per residue: CA for amino acids, P for
    nucleotides (the Cα analog). Returns {node_key: pos}."""
    out = {}
    for k, v in atoms.items():
        if k[3] == "CA" or (k[3] == "P" and (k[:3] + ("CA",)) not in atoms):
            out[k] = v[0]
    return out


def suggest_cutoff(nodes, lo=12.0, hi=30.0, factor=4.0):
    """ANM cutoff heuristic: 4x the median same-chain sequential node spacing
    (≈15 A for protein Cα, ≈25 A for nucleic P), clamped to [lo, hi].
    `nodes` is a {node_key: pos} dict from nodes_of()."""
    by_chain = {}
    for k, p in nodes.items():
        by_chain.setdefault(k[0], []).append((k[1], k[2], np.asarray(p)))
    ds = []
    for ch, lst in by_chain.items():
        lst.sort(key=lambda t: (t[0], t[1]))
        for a, b in zip(lst, lst[1:]):
            d = float(np.linalg.norm(b[2] - a[2]))
            if 1.0 < d < 15.0:  # ignore numbering jumps
                ds.append(d)
    med = float(np.median(ds)) if ds else 3.8
    return min(max(factor * med, lo), hi)


def atom_line(serial, an, resn, c, r, ins, pos):
    return ("ATOM  " + f"{serial:5d}" + " " + f"{an:^4s}" + " " + f"{resn:>3s}" + " " + c
            + f"{r:4d}" + ins + "   "
            + f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}")


def write_models(path, frames, order):
    """Write frames ({key: (pos, resname)},...) as a multi-model PDB using `order` keys."""
    with open(path, "w") as out:
        for i, fr in enumerate(frames, 1):
            out.write(f"MODEL     {i:4d}\n")
            serial = 0
            for k in order:
                if k in fr:
                    serial += 1
                    pos, resn = fr[k]
                    c, r, ins, an = k
                    out.write(atom_line(serial, an, resn, c, r, ins, pos) + "\n")
            out.write("ENDMDL\n")
