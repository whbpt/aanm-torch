"""Rigid-body geometry helpers (torch): Kabsch fit, rotation interpolation, RMSD."""
import torch


def kabsch(T, M):
    """R, t such that T ~ M @ R.T + t (same convention as the original kabsch)."""
    Mp = torch.as_tensor(M)
    Tp = torch.as_tensor(T, dtype=Mp.dtype, device=Mp.device)
    cT = Tp.mean(0)
    cM = Mp.mean(0)
    H = (Mp - cM).T @ (Tp - cT)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=H.dtype, device=H.device)) @ U.T
    t = cT - R @ cM
    return R, t


def kabsch_batch(T, M):
    """Batched variant: T, M (B, k, 3) -> R (B, 3, 3), t (B, 3) with T ~ M @ R.T + t."""
    cT = T.mean(1)
    cM = M.mean(1)
    H = (M - cM.unsqueeze(1)).transpose(1, 2) @ (T - cT.unsqueeze(1))
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2))
    one = torch.ones_like(d)
    Dm = torch.diag_embed(torch.stack([one, one, d], dim=1))
    R = Vt.transpose(1, 2) @ Dm @ U.transpose(1, 2)
    t = cT - torch.bmm(R, cM.unsqueeze(2)).squeeze(2)
    return R, t


def kabsch_fit(A, B):
    """Superpose A onto B (optimal rigid fit), like ProDy's superpose(A, B)."""
    R, t = kabsch(B, A)
    return A @ R.T + t


def rot_interp(R0, R1, f):
    """Interpolate rotation R0->R1 at fraction f via the axis-angle log map."""
    Rrel = R0.T @ R1
    c = torch.clamp((torch.diagonal(Rrel).sum() - 1) / 2, -1.0, 1.0)
    th = torch.acos(c)
    if float(th) < 1e-6:
        return R0.clone()
    v = torch.stack([Rrel[2, 1] - Rrel[1, 2],
                     Rrel[0, 2] - Rrel[2, 0],
                     Rrel[1, 0] - Rrel[0, 1]]) / (2 * torch.sin(th))
    K = torch.stack([
        torch.stack([torch.zeros_like(v[0]), -v[2], v[1]]),
        torch.stack([v[2], torch.zeros_like(v[0]), -v[0]]),
        torch.stack([-v[1], v[0], torch.zeros_like(v[0])]),
    ])
    eye = torch.eye(3, dtype=K.dtype, device=K.device)
    return R0 @ (eye + torch.sin(f * th) * K + (1 - torch.cos(f * th)) * K @ K)


def rmsd(a, b):
    return torch.sqrt(((a - b) ** 2).sum(dim=-1).mean())
