"""Synthetic structures for tests: helical CA chains with fake residues.

Helices keep neighbour-CA triangles non-degenerate (a straight CA line would
make the local Kabsch frames ill-defined).
"""
import numpy as np

AA_NAMES = ["ALA", "GLY", "SER", "LEU", "VAL", "THR"]

ATOM_OFFSETS = {
    "N": (-0.5, 0.2, -1.2),
    "CA": (0.0, 0.0, 0.0),
    "C": (1.2, -0.3, 0.1),
    "O": (2.2, 0.3, -0.4),
    "CB": (-0.4, 1.5, 0.2),
}

NUC_OFFSETS = {
    "P": (0.0, 0.0, 0.0),
    "O5'": (1.6, 0.0, 0.0),
    "C5'": (2.9, 0.9, 0.0),
    "C4'": (2.4, 2.3, 0.2),
    "C3'": (0.9, 2.6, 0.5),
    "O3'": (0.4, 3.9, 0.8),
    "N1": (3.6, 3.1, 0.4),
    "C1'": (3.2, 1.8, 1.1),
}

NUC_NAMES = ["RA", "RU", "RG", "RC"]


def helix_p(n, radius=13.0, rise=3.0, step=0.42, phase=0.0, center=(0.0, 0.0, 0.0)):
    """Phosphate 'backbone' along a helical arc, ~6.2 A per step (A-form-ish)."""
    t = np.arange(n) * step
    p = np.stack([radius * np.cos(phase + t),
                  radius * np.sin(phase + t),
                  rise * t], axis=1)
    return p + np.asarray(center)


def nucleotide_atoms(p, chain, num, with_base=True):
    """Fake nucleotide: P at the node position, phosphodiester + base atoms offset."""
    atoms = {}
    rn = NUC_NAMES[num % len(NUC_NAMES)]
    for an, off in NUC_OFFSETS.items():
        if an in ("N1", "C1'") and not with_base:
            continue
        atoms[(chain, num, " ", an)] = (p + np.asarray(off), rn)
    return atoms


def nuc_chain(pdb_ca, chain, resnum0=1):
    """Nucleic chain atoms from an array of P positions."""
    atoms = {}
    for i, p in enumerate(pdb_ca):
        atoms.update(nucleotide_atoms(p, chain, resnum0 + i))
    return atoms


def helix_ca(n, radius=7.0, rise=1.1, phase=0.0, center=(0.0, 0.0, 0.0)):
    t = np.arange(n) * 0.9
    ca = np.stack([radius * np.cos(phase + t),
                   radius * np.sin(phase + t),
                   rise * t], axis=1)
    return ca + np.asarray(center)


def chain_atoms(ca, chain, resnum0=1, with_sidechains=False):
    """Atoms dict {(chain, resnum, ' ', atom): (pos, resname)} for a CA array."""
    atoms = {}
    for i, c in enumerate(ca):
        rn = AA_NAMES[i % len(AA_NAMES)]
        for an, off in ATOM_OFFSETS.items():
            if an == "CB" and not with_sidechains:
                continue
            atoms[(chain, resnum0 + i, " ", an)] = (c + np.asarray(off), rn)
    return atoms


def rigid(p, seed=0):
    """Deterministic small rigid transform for building a second state."""
    rng = np.random.RandomState(seed)
    A = rng.randn(3, 3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    t = rng.randn(3) * 5.0
    return p @ Q.T + t


def state_pair(n=60, radius_a=7.0, radius_b=10.0, with_sidechains=False):
    """Single-chain two-state fixture: same helix, different curvature + pose."""
    caA = helix_ca(n, radius_a)
    caB = rigid(helix_ca(n, radius_b, phase=0.7), seed=1)
    atomsA = chain_atoms(caA, "A", with_sidechains=with_sidechains)
    atomsB = chain_atoms(caB, "A", with_sidechains=with_sidechains)
    ca_keys = sorted(set(atomsA) & set(atomsB))
    ca_keys = [k for k in ca_keys if k[3] == "CA"]
    return atomsA, atomsB, caA, caB, ca_keys


def write_pdb(path, atoms):
    with open(path, "w") as out:
        for s, (k, (p, rn)) in enumerate(atoms.items(), 1):
            c, r, ins, an = k
            out.write("ATOM  " + f"{s:5d}" + " " + f"{an:^4s}" + " " + f"{rn:>3s}" + " " + c
                      + f"{r:4d}" + ins + "   "
                      + f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n")
