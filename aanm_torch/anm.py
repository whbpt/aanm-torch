"""ANM Hessian construction and normal-mode extraction on torch tensors.

Dense Hessian: for each atom pair within `cutoff`, the 3x3 block is
    H_ij = -gamma * outer(r_ij, r_ij) / |r_ij|^2
with H_ii = -sum_j H_ij; modes are the eigenvectors of ascending eigenvalues
skipping the 6 rigid-body (zero) modes — numerically equivalent to ProDy's
ANM.buildHessian + calcModes.

Three solvers:
  "eigh"   torch.linalg.eigh on the dense Hessian (default up to ~1300 CA)
  "lobpcg" torch.lobpcg on the dense Hessian (GPU-friendly, no scipy needed)
  "eigsh"  scipy.sparse eigsh with a small negative shift (matrix assembled
           sparsely from the pair list; the only viable path for whole-
           assembly networks of big oligomers, runs on CPU)
"""
import warnings

import torch

_eigh_warned = False

# auto policy: dense eigh is only faster on tiny systems; the sparse shift-
# invert path (scipy eigsh, O(bonds) memory) is the default above this size
SMALL_CA_LIMIT = 250

# batched dense eigh across transitions (a GPU optimisation in adaptive.py)
# is only worthwhile below roughly this many CA
DENSE_CA_LIMIT = 1300


def _eigh(H):
    global _eigh_warned
    try:
        return torch.linalg.eigh(H)
    except RuntimeError:
        # e.g. MPS lacks eigh; diagonalise on CPU and move back
        if not _eigh_warned:
            warnings.warn(f"eigh not available on {H.device}, falling back to CPU for every step")
            _eigh_warned = True
        evals, evecs = torch.linalg.eigh(H.to("cpu"))
        return evals.to(H.device), evecs.to(H.device)


def anm_hessian(coords, cutoff=15.0, gamma=1.0, chunk=256):
    """coords: (N, 3) tensor -> (3N, 3N) Hessian on the same device/dtype.

    Built in row chunks so the (N, N, 3, 3) pair-block tensor is never
    materialised in full (float64 N=2500 would need ~1 GB).
    """
    N = coords.shape[0]
    cutoff2 = cutoff * cutoff
    H = coords.new_zeros(3 * N, 3 * N)
    diag_blocks = coords.new_zeros(N, 3, 3)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        diff = coords[i0:i1].unsqueeze(1) - coords.unsqueeze(0)  # (C, N, 3)
        d2 = (diff * diff).sum(-1)
        bonded = (d2 <= cutoff2) & (d2 > 0)  # coincident atoms are not bonded (ProDy would divide by zero)
        w = torch.where(bonded, -gamma / d2.clamp(min=1e-12), torch.zeros_like(d2))
        blocks = diff.unsqueeze(-1) * diff.unsqueeze(-2) * w.unsqueeze(-1).unsqueeze(-2)  # (C, N, 3, 3)
        H[i0 * 3:i1 * 3, :] = blocks.permute(0, 2, 1, 3).reshape((i1 - i0) * 3, 3 * N)
        diag_blocks[i0:i1] = -blocks.sum(dim=1)
    idx3 = torch.arange(3, device=coords.device)
    r = torch.arange(N, device=coords.device).unsqueeze(1) * 3 + idx3  # 3i+a
    ri = r.unsqueeze(2).expand(N, 3, 3)
    rj = r.unsqueeze(1).expand(N, 3, 3)
    H[ri, rj] = diag_blocks  # H[3i+a, 3i+b] = diag_blocks[i, a, b]
    return H


def anm_pairs(coords, cutoff=15.0, gamma=1.0, chunk=2048):
    """Bonded pair lists of the ANM network.

    Returns (i, j, w, u): node indices (each unordered pair once, i < j),
    spring weight w = gamma and unit vector u = i->j. The Hessian block of a
    pair is -w * outer(u, u) (note: with unit vectors the weight carries no
    1/r^2 factor — that matches ProDy's super_element = -gamma*outer(r,r)/r^2).
    """
    N = coords.shape[0]
    cutoff2 = cutoff * cutoff
    is_, js, us = [], [], []
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        diff = coords[i0:i1].unsqueeze(1) - coords.unsqueeze(0)  # (C, N, 3)
        d2 = (diff * diff).sum(-1)
        m = (d2 <= cutoff2) & (d2 > 0)
        rows = torch.arange(i1 - i0, device=coords.device)
        m[rows, torch.arange(i0, i1, device=coords.device)] = False
        m[rows.unsqueeze(1), torch.arange(N, device=coords.device).unsqueeze(0)] &= \
            (torch.arange(N, device=coords.device).unsqueeze(0) >
             torch.arange(i0, i1, device=coords.device).unsqueeze(1))  # keep j > i
        ii, jj = m.nonzero(as_tuple=True)
        if ii.numel() == 0:
            continue
        v = diff[ii, jj]
        r2 = d2[ii, jj]
        is_.append(ii + i0)
        js.append(jj)
        us.append(v / r2.sqrt().unsqueeze(-1))
    if not is_:
        z = torch.zeros(0, dtype=torch.long, device=coords.device)
        return z, z, coords.new_full((0,), gamma), coords.new_zeros(0, 3)
    n = sum(x.numel() for x in is_)
    return (torch.cat(is_), torch.cat(js),
            coords.new_full((n,), gamma), torch.cat(us))


def anm_modes_dense(H, n_modes, n_zero=6):
    """Lowest `n_modes` non-rigid modes via dense eigh: (evals, evecs (3N, m))."""
    evals, evecs = _eigh(H)
    n_modes = min(n_modes, evals.numel() - n_zero)
    return evals[n_zero:n_zero + n_modes], evecs[:, n_zero:n_zero + n_modes]


def anm_modes_lobpcg(H, n_modes, n_zero=6, tol=1e-9, niter=1000, seed=0):
    """Lowest non-rigid modes via torch.lobpcg on the dense Hessian."""
    n = H.shape[0]
    k = min(n_zero + n_modes, n - 2)
    g = torch.Generator(device=H.device if H.device.type != "mps" else "cpu")
    g.manual_seed(seed)
    X = torch.randn(n, k, generator=g, dtype=H.dtype, device=H.device)
    evals, evecs = torch.lobpcg(H, k=k, X=X, largest=False, tol=tol, niter=niter)
    order = evals.argsort()
    evals, evecs = evals[order], evecs[:, order]
    if not torch.isfinite(evals).all():
        raise RuntimeError("lobpcg failed to converge; try solver 'eigh'/'eigsh' "
                           "or raise niter")
    return evals[n_zero:n_zero + n_modes], evecs[:, n_zero:n_zero + n_modes]


def anm_modes_eigsh(coords, n_modes, cutoff=15.0, gamma=1.0, n_zero=6,
                    sigma=-1e-3, seed=0):
    """Lowest non-rigid modes via scipy.sparse eigsh (shift-invert).

    The Hessian is assembled as sparse COO from the pair list, so memory stays
    O(bonds); runs on CPU (scipy), modes are moved back to the input device.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh

    N = coords.shape[0]
    n3 = 3 * N
    k = min(n_zero + n_modes, n3 - 2)
    i, j, w, u = anm_pairs(coords, cutoff, gamma)
    i = i.cpu().numpy()
    j = j.cpu().numpy()
    w = w.cpu().numpy().astype(np.float64)
    u = u.cpu().numpy().astype(np.float64)
    # block B_k = -w * outer(u, u) per pair; off-diagonal slots get B_k
    # (both mirrors), each endpoint's diagonal block gets -B_k (full 3x3)
    B = -w[:, None, None] * u[:, :, None] * u[:, None, :]        # (P, 3, 3)
    R = (3 * i)[:, None] + np.arange(3)[None, :]                 # (P, 3) 3i+a
    C = (3 * j)[:, None] + np.arange(3)[None, :]                 # (P, 3) 3j+a
    Ra = np.broadcast_to(R[:, :, None], B.shape).ravel()         # row index by a
    Rb = np.broadcast_to(R[:, None, :], B.shape).ravel()         # col index by b
    Ca = np.broadcast_to(C[:, :, None], B.shape).ravel()
    Cb = np.broadcast_to(C[:, None, :], B.shape).ravel()
    bk = B.ravel()
    r = np.concatenate([Ra, Ca, Ra, Ca])
    c = np.concatenate([Cb, Rb, Rb, Cb])
    v = np.concatenate([bk, bk, -bk, -bk])
    H = coo_matrix((v, (r, c)), shape=(n3, n3)).tocsr()
    evals, evecs = eigsh(H.astype(np.float64), k=k, sigma=sigma, which="LM")
    order = np.argsort(evals)
    evals = torch.tensor(np.asarray(evals[order]), dtype=torch.float64,
                         device=coords.device)
    evecs = torch.tensor(np.asarray(evecs[:, order]), dtype=torch.float64,
                         device=coords.device)
    return evals[n_zero:n_zero + n_modes], evecs[:, n_zero:n_zero + n_modes]


def _scipy_sparse_available():
    try:
        from scipy.sparse import coo_matrix  # noqa: F401
        return True
    except Exception:
        return False


def solve_modes(coords, n_modes, cutoff=15.0, gamma=1.0, solver="auto", n_zero=6):
    """Dispatch to a solver. 'auto' (sparse-first): dense eigh up to
    SMALL_CA_LIMIT CA, scipy eigsh above when scipy.sparse is importable,
    else torch lobpcg."""
    N = coords.shape[0]
    if solver == "auto":
        if N <= SMALL_CA_LIMIT:
            solver = "eigh"
        elif _scipy_sparse_available():
            solver = "eigsh"
        else:
            solver = "lobpcg"
    if solver == "eigh":
        return anm_modes_dense(anm_hessian(coords, cutoff, gamma), n_modes, n_zero)
    if solver == "lobpcg":
        return anm_modes_lobpcg(anm_hessian(coords, cutoff, gamma), n_modes, n_zero)
    if solver == "eigsh":
        return anm_modes_eigsh(coords, n_modes, cutoff, gamma, n_zero)
    raise ValueError(f"unknown solver {solver!r}")


def disconnected(coords, cutoff=15.0, chunk=1024):
    """True if any bead's nearest neighbour is farther than `cutoff` (ProDy
    checkDisconnection stopping criterion), chunked to bound memory."""
    N = coords.shape[0]
    if N < 3:
        return False
    worst = 0.0
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        dm = torch.cdist(coords[i0:i1], coords)
        diag = torch.arange(i1 - i0, device=coords.device)
        dm[diag, diag] = float("inf")
        nn = dm.min(dim=1).values  # nearest neighbour of each row in this chunk
        lo = max(i0, 1)            # prody checks beads 1..N-2
        hi = min(i1, N - 1)
        if hi > lo:
            worst = max(worst, float(nn[lo - i0:hi - i0].max()))
    return worst > cutoff
