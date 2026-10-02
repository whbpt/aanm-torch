import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import chain_atoms, helix_ca, state_pair  # noqa: E402
from aanm_torch.backbone import reconstruct_backbone  # noqa: E402


def linear_frames(caA, caB, n=9):
    ts = np.linspace(0.0, 1.0, n)
    return [torch.tensor((1 - t) * caA + t * caB, dtype=torch.float64) for t in ts]


class TestReconstructBackbone(unittest.TestCase):
    def setUp(self):
        self.atomsA, self.atomsB, self.caA, self.caB, self.ca_keys = state_pair(n=40)

    def test_frames_cover_all_but_termini(self):
        frames = linear_frames(self.caA, self.caB, n=9)
        out = reconstruct_backbone(frames, self.atomsA, self.atomsB, self.ca_keys)
        self.assertEqual(len(out), 9)
        res = {k[:3] for k in out[0]}
        self.assertEqual(len(res), 38)  # first/last residue skipped (2 CA neighbours)
        for bb in out:
            for k, (p, rn) in bb.items():
                self.assertEqual(len(p), 3)

    def test_frame0_ca_matches_path(self):
        frames = linear_frames(self.caA, self.caB, n=5)
        out = reconstruct_backbone(frames, self.atomsA, self.atomsB, self.ca_keys)
        pos = {k[:3]: v[0] for k, v in out[0].items() if k[3] == "CA"}
        dev = max(np.linalg.norm(pos[k[:3]] - self.caA[i])
                  for i, k in enumerate(self.ca_keys) if k[:3] in pos)
        self.assertLess(dev, 1e-8)

    def test_midpoint_blend_is_continuity_upgrade(self):
        frames = linear_frames(self.caA, self.caB, n=11)

        def max_step(out):
            worst = 0.0
            for a, b in zip(out[:-1], out[1:]):
                for k in a:
                    if k in b:
                        worst = max(worst, np.linalg.norm(a[k][0] - b[k][0]))
            return worst

        hard = reconstruct_backbone(frames, self.atomsA, self.atomsB, self.ca_keys, blend=0.0)
        soft = reconstruct_backbone(frames, self.atomsA, self.atomsB, self.ca_keys, blend=0.3)
        self.assertLessEqual(max_step(soft), max_step(hard) + 1e-12)

    def test_chain_boundary_neighbours_are_same_chain(self):
        caA = np.concatenate([helix_ca(12, center=(0, 0, 0)),
                              helix_ca(12, center=(40, 0, 0))])
        caB = caA + 1.0
        atomsA = {}
        atomsA.update(chain_atoms(caA[:12], "A"))
        atomsA.update(chain_atoms(caA[12:], "B"))
        atomsB = {}
        atomsB.update(chain_atoms(caB[:12], "A"))
        atomsB.update(chain_atoms(caB[12:], "B"))
        ca_keys = sorted(k for k in atomsA if k[3] == "CA")
        frames = [torch.tensor(caA, dtype=torch.float64), torch.tensor(caB, dtype=torch.float64)]
        out = reconstruct_backbone(frames, atomsA, atomsB, ca_keys)
        res = {k[:3] for k in out[0]}
        # terminus of every chain is skipped, interiors are placed
        self.assertNotIn(("A", 12, " "), res)
        self.assertNotIn(("B", 1, " "), res)
        self.assertIn(("A", 2, " "), res)
        self.assertIn(("B", 11, " "), res)

    def test_sidechains_ride_along(self):
        atomsA, atomsB, caA, caB, ca_keys = state_pair(n=30, with_sidechains=True)
        frames = linear_frames(caA, caB, n=5)
        out = reconstruct_backbone(frames, atomsA, atomsB, ca_keys, min_atoms=1)
        n_cb = sum(1 for k in out[0] if k[3] == "CB")
        self.assertEqual(n_cb, 28)
        pos = {k: v[0] for k, v in out[0].items()}
        for k in list(pos)[:5]:
            if k[3] == "CB":
                ca = pos[(k[0], k[1], k[2], "CA")]
                self.assertLess(np.linalg.norm(pos[k] - ca), 3.0)


if __name__ == "__main__":
    unittest.main()
