import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import helix_ca  # noqa: E402
from aanm_torch.cryofit import ccc, flexible_fit, read_mrc, simulate_density, write_mrc  # noqa: E402
from aanm_torch.geom import rmsd  # noqa: E402


def bent(ca, amp=8.0):
    out = ca.copy()
    out[:, 1] += amp * np.sin(np.linspace(0, np.pi, len(out)))
    return out


def make_box(coords, voxel=1.5, margin=6.0):
    lo = coords.min(0).values - margin
    hi = coords.max(0).values + margin
    shape = tuple(int(np.ceil((hi - lo)[k] / voxel)) + 1 for k in range(3))
    return tuple(float(v) for v in lo), shape


class TestMRC(unittest.TestCase):
    def test_roundtrip(self):
        g = torch.randn(5, 6, 7)
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, "t.mrc")
            write_mrc(fn, g, voxel=1.25, origin=(3.0, -2.0, 0.5))
            g2, vo, org = read_mrc(fn)
        self.assertTrue(torch.allclose(g, g2, atol=1e-6))
        self.assertAlmostEqual(vo, 1.25)
        self.assertEqual(org, (3.0, -2.0, 0.5))


class TestDensity(unittest.TestCase):
    def test_mass_conservation_and_peak(self):
        coords = torch.tensor(helix_ca(20), dtype=torch.float64)
        origin, shape = make_box(coords)
        voxel, sigma = 1.5, 2.0
        sim = simulate_density(coords, origin, shape, voxel=voxel, sigma=sigma)
        # unnormalised grid Gaussians integrate to (2 pi sigma^2)^{3/2} per atom
        per_atom = (2 * np.pi * sigma ** 2) ** 1.5 / voxel ** 3
        self.assertAlmostEqual(float(sim.sum()), 20.0 * per_atom, delta=0.1 * 20 * per_atom)
        # peak sits near some atom
        idx = sim.argmax()
        i, j, k = np.unravel_index(int(idx), shape)
        p = torch.tensor([origin[0] + 1.5 * i, origin[1] + 1.5 * j, origin[2] + 1.5 * k])
        dmin = float(torch.cdist(p.unsqueeze(0), coords).min())
        self.assertLess(dmin, 2.0)

    def test_differentiable(self):
        coords = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
        sim = simulate_density(coords, (0., 0., 0.), (8, 8, 8), 1.0, 1.5)
        sim.sum().backward()
        self.assertGreater(float(coords.grad.abs().sum()), 0.0)


class TestFlexibleFit(unittest.TestCase):
    def test_recovers_soft_mode_target(self):
        # NMFF's working regime: the conformational change lives in the soft
        # modes, and the fit recovers it essentially exactly
        A = torch.tensor(helix_ca(40), dtype=torch.float64)
        from aanm_torch.anm import anm_hessian, anm_modes_dense
        H = anm_hessian(A)
        _, E6 = anm_modes_dense(H, 6)
        g = torch.Generator().manual_seed(2)
        scale = A.shape[0] ** 0.5
        c_true = torch.randn(6, generator=g, dtype=torch.float64) * 1.5
        B = A + (E6 @ (c_true * scale)).view_as(A)
        origin, shape = make_box(B)
        target = simulate_density(B, origin, shape, voxel=1.5, sigma=2.0)
        fitted, info = flexible_fit(A, target, origin, 1.5, sigma=2.0,
                                    n_modes=10, iters=250, lr=0.2)
        r0 = float(rmsd(A, B))
        self.assertGreater(info["ccc_final"], 0.99)
        self.assertLess(float(rmsd(fitted, B)), 0.1 * r0)

    def test_arbitrary_deformation_is_a_documented_limit(self):
        # a deformation outside the soft-mode subspace (this synthetic
        # transverse bend projects only ~0.4 onto the softest 50 modes) can
        # raise the correlation through unrealistic motion — NMFF cannot
        # recover what its mode basis cannot express
        A = torch.tensor(helix_ca(40), dtype=torch.float64)
        B = torch.tensor(bent(helix_ca(40)), dtype=torch.float64)
        origin, shape = make_box(B)
        target = simulate_density(B, origin, shape, voxel=1.5, sigma=2.0)
        c_start = ccc(simulate_density(A, origin, shape, 1.5, 2.0), target)
        fitted, info = flexible_fit(A, target, origin, 1.5, sigma=2.0,
                                    n_modes=30, iters=400, lr=0.2)
        self.assertGreater(info["ccc_final"], c_start + 0.1)


if __name__ == "__main__":
    unittest.main()
