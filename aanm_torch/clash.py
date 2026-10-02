"""Backbone clash counting (torch).

Exact pairwise counter: a clash is an atom pair closer than `cutoff` that is
not covalently adjacent. Two adjacency modes:
  - default: skip pairs with |i - j| <= skip in list order
  - groups:  pass an integer residue/group id per atom; skip pairs whose group
    ids differ by <= skip (works for sidechain atoms where list order no
    longer tracks bonding)

This replaces the original grid-bucketed counter, which only compared atoms
landing in the *same* 1 A cell and therefore missed pairs across cell
boundaries.
"""
import torch


def clash_count(pts, cutoff=1.6, skip=3, groups=None, chunk=8192):
    """pts: (N, 3) tensor (or list/array convertible) -> (count, worst overlap A)."""
    P = torch.as_tensor(pts)
    if P.device.type == "mps" and P.dtype == torch.float64:
        P = P.float()  # MPS cdist lacks float64
    N = P.shape[0]
    if N < 2:
        return 0, 0.0
    if groups is not None:
        G = torch.as_tensor(groups, device=P.device)
    count = 0
    worst = 0.0
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        D = torch.cdist(P[i0:i1], P)
        rows = torch.arange(i0, i1, device=P.device).unsqueeze(1)
        cols = torch.arange(N, device=P.device).unsqueeze(0)
        if groups is None:
            near = (cols - rows).abs() <= skip
        else:
            near = (G[cols] - G[rows]).abs() <= skip
        m = (cols > rows) & ~near & (D < cutoff)
        if bool(m.any()):
            count += int(m.sum())
            worst = max(worst, float((cutoff - D[m]).max()))
    return count, worst
