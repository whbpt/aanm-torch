import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import helix_ca, rigid  # noqa: E402
from aanm_torch.geom import kabsch_fit, rmsd  # noqa: E402
from aanm_torch.pathway import anm_pathway, stiff_fraction  # noqa: E402


def bent(ca, amp=9.0):
    out = ca.copy()
    out[:, 1] += amp * np.sin(np.linspace(0, np.pi, len(out)))
    return out


class TestPathway(unittest.TestCase):
    def setUp(self):
        self.A = torch.tensor(helix_ca(70), dtype=torch.float64)
        self.B = kabsch_fit(torch.tensor(rigid(bent(helix_ca(70)), seed=5),
                                         dtype=torch.float64), self.A)

    def test_exact_endpoints(self):
        frames, info = anm_pathway(self.A, self.B, n_cycles=20)
        self.assertLess(float((frames[0] - self.A).abs().max()), 1e-12)
        self.assertLess(float((frames[-1] - self.B).abs().max()), 1e-12)
        self.assertGreater(len(frames), 2)

    def test_fronts_meet(self):
        frames, info = anm_pathway(self.A, self.B, n_cycles=40, target_rmsd=1.5)
        self.assertLess(info["meeting_rmsd"], 3.0)

    def test_steps_are_purely_soft(self):
        frames, _ = anm_pathway(self.A, self.B, n_cycles=12, n_relax=6)
        worst = 0.0
        for a, b in zip(frames[:-2], frames[1:-1]):
            worst = max(worst, stiff_fraction(a, b, n_soft=6))
        self.assertLess(worst, 1e-8)
        # the closing step into the fixed B endpoint is exempt from the
        # soft-purity criterion; it only has to stay continuous
        interior = [float((b - a).norm()) for a, b in zip(frames[:-2], frames[1:-1])]
        closing = float((frames[-1] - frames[-2]).norm())
        self.assertLess(closing, 3.0 * max(interior))

    def test_path_progresses(self):
        frames, _ = anm_pathway(self.A, self.B, n_cycles=25)
        mid = frames[len(frames) // 2]
        r_A = float(rmsd(mid, self.A))
        r_B = float(rmsd(mid, self.B))
        r_AB = float(rmsd(self.A, self.B))
        # the midpoint frame should sit between the endpoints
        self.assertLess(r_A, r_AB)
        self.assertLess(r_B, r_AB)


if __name__ == "__main__":
    unittest.main()
