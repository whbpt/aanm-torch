import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aanm_torch.geom import kabsch, kabsch_batch, rot_interp, rmsd  # noqa: E402


class TestKabsch(unittest.TestCase):
    def test_recovers_rotation_translation(self):
        rng = np.random.RandomState(0)
        M = rng.randn(50, 3)
        Q, _ = np.linalg.qr(rng.randn(3, 3))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = rng.randn(3)
        T = M @ Q.T + t
        R, tt = kabsch(torch.tensor(T), torch.tensor(M))
        # convention: T = M @ R.T + t, so the recovered rotation is Q itself
        self.assertTrue(torch.allclose(R, torch.tensor(Q), atol=1e-10))
        self.assertTrue(torch.allclose(R @ torch.tensor(M).T + tt.unsqueeze(1),
                                       torch.tensor(T).T, atol=1e-9))

    def test_proper_rotation_for_mirrored_target(self):
        rng = np.random.RandomState(1)
        M = torch.tensor(rng.randn(30, 3))
        T = M.clone()
        T[:, 0] *= -1.0  # reflection
        R, _ = kabsch(T, M)
        self.assertAlmostEqual(float(torch.det(R)), 1.0, places=10)

    def test_batch_matches_single(self):
        rng = np.random.RandomState(2)
        Ts, Ms = [], []
        for _ in range(5):
            M = rng.randn(20, 3)
            Q, _ = np.linalg.qr(rng.randn(3, 3))
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            Ts.append(M @ Q.T + rng.randn(3))
            Ms.append(M)
        T = torch.tensor(np.stack(Ts))
        M = torch.tensor(np.stack(Ms))
        Rb, tb = kabsch_batch(T, M)
        for i in range(5):
            R, t = kabsch(T[i], M[i])
            self.assertTrue(torch.allclose(Rb[i], R, atol=1e-10))
            self.assertTrue(torch.allclose(tb[i], t, atol=1e-10))


class TestRotInterp(unittest.TestCase):
    def test_endpoints(self):
        R0 = torch.eye(3, dtype=torch.float64)
        th = 0.7
        K = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float64)
        R1 = torch.matrix_exp(th * K)
        self.assertTrue(torch.allclose(rot_interp(R0, R1, 0.0), R0, atol=1e-12))
        self.assertTrue(torch.allclose(rot_interp(R0, R1, 1.0), R1, atol=1e-10))
        Rh = rot_interp(R0, R1, 0.5)
        self.assertAlmostEqual(float(torch.det(Rh)), 1.0, places=12)


class TestRmsd(unittest.TestCase):
    def test_known(self):
        a = torch.zeros(2, 3, dtype=torch.float64)
        b = torch.zeros(2, 3, dtype=torch.float64)
        b[0, 0] = 3.0
        b[1, 1] = 4.0
        self.assertAlmostEqual(float(rmsd(a, b)), 3.5355339059327378, places=10)


if __name__ == "__main__":
    unittest.main()
