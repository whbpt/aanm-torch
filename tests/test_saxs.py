import sys
import unittest
from pathlib import Path

import math

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aanm_torch.saxs import saxs_curve  # noqa: E402


class TestSaxs(unittest.TestCase):
    def test_two_atoms_analytic(self):
        # I(q) = 2 f^2 (1 + sinc(q r))
        r = 5.0
        coords = torch.tensor([[0.0, 0, 0], [r, 0, 0]], dtype=torch.float64)
        q = torch.linspace(0.01, 0.5, 17, dtype=torch.float64)
        f = 6.0
        I = saxs_curve(coords, q, torch.full((2,), f, dtype=torch.float64))
        expect = 2 * f * f * (1 + torch.sin(q * r) / (q * r))
        self.assertTrue(torch.allclose(I, expect, atol=1e-10))

    def test_q_zero_limit(self):
        # I(0) = (sum f)^2; checked at small q via the sinc expansion
        coords = torch.tensor([[0.0, 0, 0], [5.0, 1, 1]], dtype=torch.float64)
        q = torch.tensor([1e-4], dtype=torch.float64)
        w = torch.tensor([6.0, 7.0], dtype=torch.float64)
        I = saxs_curve(coords, q, w)
        self.assertAlmostEqual(float(I[0]), 13.0 ** 2, delta=1e-4)

    def test_matches_naive_loop(self):
        g = torch.Generator().manual_seed(0)
        coords = torch.randn(4, 30, 3, dtype=torch.float64, generator=g)
        w = (torch.randn(4, 30, dtype=torch.float64, generator=g)).abs() + 1
        q = torch.linspace(0.02, 0.4, 9, dtype=torch.float64)
        I = saxs_curve(coords, q, w)
        wf = w.tolist()
        cf = coords.tolist()
        for b in range(4):
            for qi, qq in enumerate(q.tolist()):
                ref = 0.0
                for i in range(30):
                    for j in range(30):
                        r = math.dist(cf[b][i], cf[b][j])
                        ref += wf[b][i] * wf[b][j] * (1.0 if r == 0 else math.sin(qq * r) / (qq * r))
                self.assertAlmostEqual(float(I[b, qi]), ref, delta=1e-8)

    def test_differentiable(self):
        coords = torch.randn(20, 3, dtype=torch.float64, requires_grad=True)
        q = torch.linspace(0.02, 0.4, 5, dtype=torch.float64)
        I = saxs_curve(coords, q)
        I.sum().backward()
        self.assertIsNotNone(coords.grad)
        with torch.no_grad():
            eps = 1e-6
            c2 = coords.detach().clone()
            c2[3, 1] += eps
            num = (saxs_curve(c2, q).sum() - saxs_curve(coords.detach(), q).sum()) / eps
        self.assertAlmostEqual(float(coords.grad[3, 1]), float(num), delta=1e-3)


if __name__ == "__main__":
    unittest.main()
