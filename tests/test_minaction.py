import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import helix_ca, rigid  # noqa: E402
from aanm_torch.minaction import min_action_path, path_action, elastic_potential  # noqa: E402
from aanm_torch.anm import anm_hessian  # noqa: E402
from aanm_torch.geom import kabsch_fit  # noqa: E402


def bent(ca, amp=9.0):
    out = ca.copy()
    out[:, 1] += amp * np.sin(np.linspace(0, np.pi, len(out)))
    return out


class TestMinAction(unittest.TestCase):
    def setUp(self):
        self.A = torch.tensor(helix_ca(60), dtype=torch.float64)
        # compare in the aligned frame: min_action_path aligns B internally
        self.B = kabsch_fit(torch.tensor(rigid(bent(helix_ca(60)), seed=2),
                                         dtype=torch.float64), self.A)

    def test_endpoints_fixed(self):
        frames, _ = min_action_path(self.A, self.B, n_images=10, n_iters=150)
        self.assertEqual(len(frames), 10)
        self.assertLess(float((frames[0] - self.A).abs().max()), 1e-12)
        self.assertLess(float((frames[-1] - self.B).abs().max()), 1e-12)

    def test_action_beats_linear(self):
        # dt=10 puts the action in the potential-dominant regime where the
        # optimal path measurably curves away from the straight interpolation
        frames, info = min_action_path(self.A, self.B, n_images=12, dt=10.0)
        self.assertLess(info["final_action"], 0.95 * info["linear_action"])
        # and the returned frames' action matches the reported final one
        xA, xB = self.A, self.B
        HA = anm_hessian(xA)
        HB = anm_hessian(xB)
        vs = info["vscale"]
        self.assertAlmostEqual(
            float(path_action(frames, xA, HA, xB, HB, dt=10.0, vscale=vs)),
            info["final_action"], places=6)

    def test_smoothness_not_worse(self):
        # least-action paths avoid abrupt jumps: max step should not exceed
        # the linear path's max step by much
        frames, _ = min_action_path(self.A, self.B, n_images=12, n_iters=300)
        lin = [(1 - t) * self.A + t * self.B
               for t in torch.linspace(0, 1, 12, dtype=torch.float64)]
        max_opt = max(float((b - a).norm()) for a, b in zip(frames[:-1], frames[1:]))
        max_lin = max(float((b - a).norm()) for a, b in zip(lin[:-1], lin[1:]))
        self.assertLess(max_opt, 1.2 * max_lin)


if __name__ == "__main__":
    unittest.main()
