#!/usr/bin/env python3
"""Sparse-solver scaling benchmark for aanm_torch.

Builds compact protein-like CA chains (protein-density random walks), then for
each size measures: pair-list/Hessian-sparse assembly, eigsh for 10 non-rigid
modes, and one full adaptive step (Hessian + solve + update). Dense eigh is
timed where still feasible to show the crossover.

Run with a scipy-healthy python, e.g.:
    .venv-bioemu/bin/python experiments/04_bench_sparse.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aanm_torch.adaptive import calc_adaptive_anm_one_way  # noqa: E402
from aanm_torch.anm import anm_hessian, anm_modes_dense, anm_modes_eigsh, anm_pairs  # noqa: E402


def globular_chain(n, seed=0):
    """Random walk at protein-like density (~3.8 A steps) confined to a sphere
    of radius ~2.6 * n^(1/3) A, so the contact density is realistic."""
    rng = np.random.RandomState(seed)
    r = 2.6 * n ** (1 / 3.0)
    pts = np.zeros((n, 3))
    for k in range(1, n):
        step = rng.randn(3)
        step *= 3.8 / np.linalg.norm(step)
        p = pts[k - 1] + step
        if np.linalg.norm(p) > r:
            p *= r / np.linalg.norm(p)
        pts[k] = p
    return pts


def bench(n, n_modes=10, dense_limit=2600):
    coords = torch.tensor(globular_chain(n), dtype=torch.float64)
    B = torch.tensor(globular_chain(n, seed=1), dtype=torch.float64)

    t0 = time.time()
    i, j, w, u = anm_pairs(coords)
    t_pairs = time.time() - t0

    t0 = time.time()
    ev, evec = anm_modes_eigsh(coords, n_modes)
    t_eigsh = time.time() - t0

    t_dense = None
    if n <= dense_limit:
        t0 = time.time()
        anm_modes_dense(anm_hessian(coords), n_modes)
        t_dense = time.time() - t0

    t0 = time.time()
    frames, info = calc_adaptive_anm_one_way(coords, B, 1, n_modes=n_modes)
    t_step = time.time() - t0

    print(f"n={n:6d} (dof {3 * n:6d}) | pairs {int(i.numel()):8d} "
          f"| pair-list {t_pairs:6.2f}s | eigsh {t_eigsh:7.2f}s"
          + (f" | dense eigh {t_dense:7.2f}s" if t_dense else " | dense: n/a")
          + f" | full AANM step {t_step:7.2f}s | lam1 {float(ev[0]):.3e}")


if __name__ == "__main__":
    torch.set_num_threads(max(torch.get_num_threads() // 2, 1))
    print(f"threads: {torch.get_num_threads()}")
    for n in (500, 1000, 2000, 5000, 10000, 20000):
        bench(n)
