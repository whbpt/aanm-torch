import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from synth import helix_ca, rigid, state_pair  # noqa: E402
from aanm_torch.adaptive import (calc_adaptive_anm_alternating,  # noqa: E402
                                 calc_adaptive_anm_batched, calc_adaptive_anm_one_way)
from aanm_torch.geom import kabsch_fit  # noqa: E402


def bent_chain(ca, amp=8.0):
    """Smooth transverse bending — overlaps strongly with a helix's softest mode."""
    out = ca.copy()
    n = len(out)
    out[:, 1] += amp * np.sin(np.linspace(0.0, np.pi, n))
    return out


class TestOneWay(unittest.TestCase):
    def setUp(self):
        # radial expansion stalls in one-way AANM (ProDy does the same — softest
        # helix modes are bending); use a bending target instead
        self.A = torch.tensor(helix_ca(100, radius=7.0), dtype=torch.float64)
        self.B = torch.tensor(rigid(bent_chain(helix_ca(100, radius=7.0)), seed=4),
                              dtype=torch.float64)

    def test_rmsd_decreases(self):
        frames, info = calc_adaptive_anm_one_way(self.A, self.B, 10, n_modes=10, f=0.3)
        self.assertLessEqual(len(frames), 10)
        self.assertLess(info["rmsds"][-1], 0.7 * info["rmsds"][0])
        mono = all(b < a + 0.5 for a, b in zip(info["rmsds"], info["rmsds"][1:]))
        self.assertTrue(mono)

    def test_callback_matches_info(self):
        seen = []
        calc_adaptive_anm_one_way(self.A, self.B, 4, n_modes=6,
                                  callback_func=lambda **kw: seen.append(kw["rmsd"]))
        frames, info = calc_adaptive_anm_one_way(self.A, self.B, 4, n_modes=6)
        self.assertEqual(seen, info["rmsds"][1:])


class TestAlternating(unittest.TestCase):
    def test_exact_endpoints(self):
        A = torch.tensor(helix_ca(90, radius=7.0), dtype=torch.float64)
        B = torch.tensor(rigid(helix_ca(90, radius=10.5, phase=0.4), seed=5),
                         dtype=torch.float64)
        frames, info = calc_adaptive_anm_alternating(A, B, 6, n_modes=8)
        A0 = kabsch_fit(A, B)  # the loop starts from the superposed A, as ProDy does
        self.assertLess(float((frames[0] - A0).abs().max()), 1e-9)
        self.assertLess(float((frames[-1] - B).abs().max()), 1e-9)
        # continuity along the whole path
        worst = max(float((a - b).abs().max()) for a, b in zip(frames[:-1], frames[1:]))
        self.assertLess(worst, 6.0)


class TestBatchedParity(unittest.TestCase):
    def test_batched_equals_sequential(self):
        pairs = []
        for seed in range(3):
            A = torch.tensor(helix_ca(90, radius=7.0 + seed), dtype=torch.float64)
            B = torch.tensor(rigid(helix_ca(90, radius=10.0 + seed, phase=0.5),
                                   seed=seed), dtype=torch.float64)
            pairs.append((A, B))
        kw = dict(n_modes=8, f=0.3)
        seq = [calc_adaptive_anm_one_way(a, b, 7, solver="eigh", **kw) for a, b in pairs]
        bat = calc_adaptive_anm_batched(pairs, 7, solver="eigh", **kw)
        for (fs, info), (fb, ib) in zip(seq, bat):
            self.assertEqual(len(fs), len(fb))
            worst = max(float((x - y).abs().max()) for x, y in zip(fs, fb))
            self.assertLess(worst, 1e-9)
            self.assertEqual(info["converged"], ib["converged"])

    def test_mismatched_sizes_rejected(self):
        a1 = torch.randn(50, 3, dtype=torch.float64)
        b1 = torch.randn(50, 3, dtype=torch.float64)
        a2 = torch.randn(60, 3, dtype=torch.float64)
        b2 = torch.randn(60, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            calc_adaptive_anm_batched([(a1, b1), (a2, b2)], 3, solver="eigh")


if __name__ == "__main__":
    unittest.main()
