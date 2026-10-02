"""Differentiable SAXS curve calculation (Debye formula).

I(q) = sum_ij f_i f_j sinc(q r_ij), computed with batched pairwise distances —
O(N^2) per q, fully differentiable w.r.t. coordinates, so experimental curves
can be used directly inside gradient-based refinement.

Approximation level (documented, deliberate): constant per-atom form factors
(electron counts) instead of q-dependent Cromer-Mann with solvent exclusion —
fine for relative curves, differentiable fitting and ensembles; not for
absolute-intensity matching against experiment.
"""
import torch

# simple electron-count form factors; CA-only networks can use "C"
FORM_FACTORS = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "P": 15.0, "S": 16.0}


def saxs_curve(coords: torch.Tensor, q: torch.Tensor, weights: torch.Tensor = None,
               chunk: int = 4096) -> torch.Tensor:
    """Debye scattering intensity.

    coords: (..., n, 3) in A; q: (nq,) in 1/A; weights: per-atom form factors
    (..., n) — defaults to carbon (CA networks). Returns (..., nq).
    """
    if weights is None:
        weights = coords.new_full(coords.shape[:-1], FORM_FACTORS["C"])
    d = torch.cdist(coords, coords)                                  # (..., n, n)
    ww = (weights.unsqueeze(-1) * weights.unsqueeze(-2)).unsqueeze(0)  # (1, ..., n, n)
    mask = (d > 0).unsqueeze(0)                                       # (1, ..., n, n)
    out_shape = coords.shape[:-2] + q.shape
    I = torch.empty(out_shape, dtype=coords.dtype, device=coords.device)
    flat_q = q.reshape(-1)
    for k0 in range(0, flat_q.numel(), chunk):
        qc = flat_q[k0:k0 + chunk].view(-1, *([1] * d.dim()))        # (k, 1..., 1)
        qr = qc * d.unsqueeze(0)                                      # (k, ..., n, n)
        sinc = torch.where(mask, torch.sin(qr) / qr, torch.ones_like(qr))
        I[..., k0:k0 + qc.numel()] = (ww * sinc).sum((-2, -1)).movedim(0, -1)
    return I
