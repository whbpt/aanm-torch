import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aanm_torch.clash import clash_count  # noqa: E402


class TestClash(unittest.TestCase):
    def test_basic_count_and_worst(self):
        pts = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]], dtype=torch.float64)
        c, w = clash_count(pts, cutoff=1.6, skip=0)
        self.assertEqual(c, 1)  # only pair (0,1) is within 1.6 A
        self.assertAlmostEqual(w, 0.6, places=12)

    def test_list_skip(self):
        pts = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
        c, _ = clash_count(pts, cutoff=1.6, skip=1)
        self.assertEqual(c, 0)

    def test_groups_mode(self):
        pts = torch.tensor([[0.0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [1.5, 0, 0]],
                           dtype=torch.float64)
        g = torch.tensor([0, 0, 1, 1])
        # skip=0 excludes same-group pairs, so only (0/1)x(2/3) cross pairs count
        c, _ = clash_count(pts, cutoff=1.6, skip=0, groups=g)
        self.assertEqual(c, 4)  # (0,2) (0,3) (1,2) (1,3)
        # skip=1 additionally excludes adjacent groups
        c2, _ = clash_count(pts, cutoff=1.6, skip=1, groups=g)
        self.assertEqual(c2, 0)

    def test_empty(self):
        c, w = clash_count(torch.zeros(1, 3), cutoff=1.6)
        self.assertEqual((c, w), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
