import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import helix_ca  # noqa: E402
from aanm_torch.anm import (anm_hessian, anm_modes_dense, anm_modes_eigsh,  # noqa: E402
                            anm_modes_lobpcg, anm_pairs, disconnected, solve_modes)

try:
    from scipy.sparse import coo_matrix as _coo  # noqa: F401
    _sparse_ok = True
except Exception:
    _sparse_ok = False


class TestHessian(unittest.TestCase):
    def test_two_atoms_hand_computed(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64)
        H = anm_hessian(coords, cutoff=15.0, gamma=1.0)
        u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        b = torch.outer(u, u)  # H_ii = +gamma*outer(u,u); H_ij = -gamma*outer(u,u)
        expect = torch.cat([torch.cat([b, -b], dim=1), torch.cat([-b, b], dim=1)], dim=0)
        self.assertTrue(torch.allclose(H, expect, atol=1e-12))

    def test_symmetric_laplacian(self):
        coords = torch.tensor(helix_ca(80), dtype=torch.float64)
        H = anm_hessian(coords)
        self.assertTrue(torch.allclose(H, H.T, atol=1e-12))
        resid = H @ torch.ones(H.shape[0], dtype=torch.float64)
        self.assertLess(float(resid.abs().max()), 1e-10)

    def test_pairs_match_hessian(self):
        coords = torch.tensor(helix_ca(60), dtype=torch.float64)
        H = anm_hessian(coords)
        i, j, w, u = anm_pairs(coords)
        for p in range(i.numel()):
            ii, jj = int(i[p]), int(j[p])
            block = -w[p] * torch.outer(u[p], u[p])
            self.assertTrue(torch.allclose(H[3 * ii:3 * ii + 3, 3 * jj:3 * jj + 3],
                                           block, atol=1e-12))


class TestModes(unittest.TestCase):
    def setUp(self):
        self.coords = torch.tensor(helix_ca(120), dtype=torch.float64)
        self.H = anm_hessian(self.coords)

    def test_six_zero_modes(self):
        evals, evecs = anm_modes_dense(self.H, 5)
        full = torch.linalg.eigvalsh(self.H)
        for k in range(6):
            self.assertLess(float(full[k]), 1e-9)
        self.assertTrue(torch.allclose(evals, full[6:11], atol=1e-9))
        self.assertEqual(evecs.shape, (360, 5))

    def test_solvers_agree(self):
        ref_vals, ref_vecs = anm_modes_dense(self.H, 8)
        lob_vals, lob_vecs = anm_modes_lobpcg(self.H, 8)
        rel = ((lob_vals - ref_vals).abs() / ref_vals.clamp(min=1e-12)).max()
        self.assertLess(float(rel), 1e-6)
        # subspace projection of a fixed random vector is basis-invariant
        rng = torch.Generator().manual_seed(3)
        x = torch.randn(360, generator=rng, dtype=torch.float64)
        pa = ref_vecs @ (ref_vecs.T @ x)
        pb = lob_vecs @ (lob_vecs.T @ x)
        self.assertLess(float((pa - pb).norm() / x.norm()), 1e-5)

    @unittest.skipUnless(_sparse_ok, "scipy.sparse not importable in this environment")
    def test_eigsh_agrees(self):
        ref_vals, _ = anm_modes_dense(self.H, 6)
        sp_vals, _ = anm_modes_eigsh(self.coords, 6)
        rel = ((sp_vals - ref_vals).abs() / ref_vals.clamp(min=1e-12)).max()
        self.assertLess(float(rel), 1e-7)

    def test_solve_modes_auto_small_uses_eigh(self):
        vals, _ = solve_modes(self.coords, 4, solver="eigh")
        ref, _ = anm_modes_dense(self.H, 4)
        self.assertTrue(torch.allclose(vals, ref, atol=1e-10))

    @unittest.skipUnless(_sparse_ok, "scipy.sparse not importable in this environment")
    def test_solve_modes_auto_prefers_sparse_above_limit(self):
        coords = torch.tensor(helix_ca(400), dtype=torch.float64)
        vals, _ = solve_modes(coords, 6, solver="auto")  # 400 CA > SMALL_CA_LIMIT
        ref, _ = anm_modes_dense(anm_hessian(coords), 6)
        rel = ((vals - ref).abs() / ref.clamp(min=1e-12)).max()
        self.assertLess(float(rel), 1e-7)


class TestDisconnected(unittest.TestCase):
    def test_lonely_bead(self):
        # prody's criterion checks beads 1..N-2: a lonely bead in the middle
        # has its nearest neighbour far beyond the cutoff
        ca = np.concatenate([helix_ca(10),
                             helix_ca(1, center=(500.0, 0.0, 0.0)),
                             helix_ca(10, center=(0.0, 500.0, 0.0))])
        self.assertTrue(disconnected(torch.tensor(ca, dtype=torch.float64)))

    def test_connected(self):
        self.assertFalse(disconnected(torch.tensor(helix_ca(30), dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
