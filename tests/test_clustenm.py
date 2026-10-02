import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import helix_ca  # noqa: E402
from aanm_torch.clash import clash_count  # noqa: E402
from aanm_torch.clustenm import clustenm  # noqa: E402
from aanm_torch.geom import rmsd  # noqa: E402


class TestClustENM(unittest.TestCase):
    def setUp(self):
        self.coords = torch.tensor(helix_ca(50), dtype=torch.float64)

    def test_ensemble_diversity_and_cleanliness(self):
        reps, ens = clustenm(self.coords, n_iters=2, n_gen=16, amp=2.0, seed=0)
        self.assertGreater(ens.shape[0], 10)
        spreads = [float(rmsd(c, self.coords)) for c in ens]
        self.assertGreater(max(spreads), 0.5)   # real diversity
        self.assertLess(max(spreads), 15.0)     # but still the same molecule
        worst = 0
        for c in ens:
            n_cl, _ = clash_count(c, cutoff=4.0, skip=2)
            worst = max(worst, n_cl)
        self.assertLessEqual(worst, 4)          # clash filter held

    def test_representatives_recover_two_basins(self):
        # unbiased ClustENM fluctuates around its seeds; seeded with both
        # basins, clustering keeps one representative in each
        A = self.coords
        B = A.clone()
        B[:, 1] += 8.0 * torch.sin(torch.linspace(0, 3.14159, B.shape[0],
                                                  dtype=torch.float64))
        reps, _ = clustenm([A, B], n_iters=1, n_gen=24, amp=1.5,
                           n_clusters=2, seed=1)
        dA = [float(rmsd(r, A)) for r in reps]
        dB = [float(rmsd(r, B)) for r in reps]
        self.assertEqual(len(reps), 2)
        self.assertLess(min(dA), 2.0)
        self.assertLess(min(dB), 2.0)

    def test_deterministic_with_seed(self):
        r1, e1 = clustenm(self.coords, n_iters=1, n_gen=8, seed=3)
        r2, e2 = clustenm(self.coords, n_iters=1, n_gen=8, seed=3)
        self.assertEqual(e1.shape, e2.shape)
        self.assertLess(float((e1 - e2).abs().max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
