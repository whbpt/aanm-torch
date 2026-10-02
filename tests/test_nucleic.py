import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import (chain_atoms, helix_ca, helix_p, nuc_chain, rigid,  # noqa: E402
                   state_pair, write_pdb)
from aanm_torch.adaptive import calc_adaptive_anm_one_way  # noqa: E402
from aanm_torch.backbone import reconstruct_backbone  # noqa: E402
from aanm_torch.pdbio import nodes_of, parse_atoms, suggest_cutoff  # noqa: E402


def nuc_state_pair(n=60, r_a=12.0, r_b=16.0):
    pA = helix_p(n, radius=r_a)
    pB = rigid(helix_p(n, radius=r_b, phase=1.2), seed=9)
    atomsA = nuc_chain(pA, "A")
    atomsB = nuc_chain(pB, "A")
    nodesA = nodes_of(atomsA)
    nodesB = nodes_of(atomsB)
    common = sorted(set(nodesA) & set(nodesB))
    return atomsA, atomsB, pA, pB, common


class TestPdbioNucleic(unittest.TestCase):
    def test_parse_and_nodes(self):
        atomsA, _, _, _, _ = nuc_state_pair(n=20)
        parsed, _ = parse_atoms(self._write(atomsA), bb_only=True)
        names = {k[3] for k in parsed}
        self.assertIn("P", names)
        self.assertIn("C4'", names)
        self.assertNotIn("N1", names)  # base atoms are not backbone
        nodes = nodes_of(parsed)
        self.assertEqual(len(nodes), 20)
        self.assertTrue(all(k[3] == "P" for k in nodes))

    def test_mixed_complex_nodes(self):
        atoms = {}
        atoms.update(chain_atoms(helix_ca(15), "A"))
        atoms.update(nuc_chain(helix_p(15, center=(50, 0, 0)), "B"))
        parsed, _ = parse_atoms(self._write(atoms), bb_only=True)
        nodes = nodes_of(parsed)
        kinds = {(k[0], k[3]) for k in nodes}
        self.assertEqual(kinds, {("A", "CA"), ("B", "P")})
        self.assertEqual(len(nodes), 30)

    def test_suggest_cutoff(self):
        atomsA, _, _, _, _ = nuc_state_pair(n=40)
        c_nuc = suggest_cutoff(nodes_of(atomsA))
        self.assertGreater(c_nuc, 20.0)
        # protein case with realistic 3.8 A Cα spacing (the default synth helix
        # is spaced for geometry tests, not realism)
        t = np.arange(40) * 0.6
        prot = np.stack([4.0 * np.cos(t), 4.0 * np.sin(t), 4.95 * t], axis=1)
        c_prot = suggest_cutoff(nodes_of(chain_atoms(prot, "A")))
        self.assertLess(abs(c_prot - 15.0), 2.5)

    @staticmethod
    def _write(atoms):
        import tempfile
        import os
        fn = os.path.join(tempfile.mkdtemp(), "x.pdb")
        write_pdb(fn, atoms)
        return fn


class TestNucleicBackbone(unittest.TestCase):
    def test_reconstruct(self):
        atomsA, atomsB, pA, pB, common = nuc_state_pair(n=40)
        frames = [torch.tensor((1 - t) * pA + t * pB, dtype=torch.float64)
                  for t in np.linspace(0, 1, 5)]
        out = reconstruct_backbone(frames, atomsA, atomsB, common)
        res = {k[:3] for k in out[0]}
        self.assertEqual(len(res), 38)  # termini skipped, as for protein
        n_p = sum(1 for k in out[0] if k[3] == "P")
        self.assertEqual(n_p, 38)
        pos = {k[:3]: v[0] for k, v in out[0].items() if k[3] == "P"}
        dev = max(np.linalg.norm(pos[k[:3]] - pA[i])
                  for i, k in enumerate(common) if k[:3] in pos)
        self.assertLess(dev, 1e-8)  # frame 0 == state A exactly


class TestNucleicAdaptive(unittest.TestCase):
    def test_aanm_on_nucleic_nodes(self):
        # bending target (overlaps the softest modes); radial expansion stalls
        # in one-way AANM for helices regardless of molecule type
        atomsA, atomsB, pA, pB, common = nuc_state_pair(n=80)
        n = len(pA)
        pB = rigid(pA + np.stack([np.zeros(n),
                                  9.0 * np.sin(np.linspace(0, np.pi, n)),
                                  np.zeros(n)], axis=1), seed=11)
        A = torch.tensor(pA, dtype=torch.float64)
        B = torch.tensor(pB, dtype=torch.float64)
        _, info = calc_adaptive_anm_one_way(A, B, 8, n_modes=8, f=0.3,
                                            cutoff=suggest_cutoff(nodes_of(atomsA)))
        self.assertLess(info["rmsds"][-1], 0.75 * info["rmsds"][0])


if __name__ == "__main__":
    unittest.main()
