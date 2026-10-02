"""cryo-EM flexible fitting with ANM modes and autograd (NMFF/DireX-style).

Classic normal-mode flexible fitting (NMFF, Tama et al. 2004; DireX,
Schröder et al. 2007) drives a structure into a density map by optimising
normal-mode coordinates against the real-space correlation. Here the ANM
mode coefficients are torch parameters and the whole chain — coordinates ->
simulated Gaussian density -> cross-correlation — is differentiable, so the
fit is a plain gradient descent restricted (by construction) to soft
collective motions.

MRC support is deliberately minimal: mode-2 (float32) maps, no compression,
headers written/read for grid/voxel/origin only.
"""
import struct

import numpy as np
import torch

from .anm import anm_hessian, anm_modes_dense


def write_mrc(path, grid, voxel, origin):
    """grid: (nx, ny, nz) tensor/array; voxel float; origin (3,) in A."""
    g = np.ascontiguousarray(np.asarray(grid, dtype=np.float32))
    nx, ny, nz = g.shape
    vo = float(voxel)
    h = bytearray(1024)
    struct.pack_into("<3i", h, 0, nx, ny, nz)
    struct.pack_into("<i", h, 12, 2)                       # mode = float32
    struct.pack_into("<3i", h, 16, 0, 0, 0)                # nxstart...
    struct.pack_into("<3i", h, 28, nx, ny, nz)             # mx,my,mz
    struct.pack_into("<3f", h, 40, nx * vo, ny * vo, nz * vo)
    struct.pack_into("<i", h, 92, 0)                       # nsymbt
    struct.pack_into("<3f", h, 176, float(g.min()), float(g.max()), float(g.mean()))
    struct.pack_into("<3f", h, 196, *[float(o) for o in origin])
    h[208:212] = b"MAP "
    with open(path, "wb") as f:
        f.write(h)
        f.write(g.tobytes())


def read_mrc(path):
    """Returns (grid (nx,ny,nz) float32 tensor, voxel, origin (3,) tuple)."""
    with open(path, "rb") as f:
        h = f.read(1024)
        nx, ny, nz = struct.unpack_from("<3i", h, 0)
        mode, = struct.unpack_from("<i", h, 12)
        nsymbt, = struct.unpack_from("<i", h, 92)
        xl, yl, zl = struct.unpack_from("<3f", h, 40)
        ox, oy, oz = struct.unpack_from("<3f", h, 196)
        if mode != 2:
            raise ValueError(f"only mode-2 (float32) MRC supported, got mode {mode}")
        f.seek(1024 + nsymbt)
        data = np.frombuffer(f.read(nx * ny * nz * 4), dtype="<f4")
    grid = torch.from_numpy(data.reshape(nx, ny, nz).copy())
    voxel = float(xl / max(nx, 1))
    return grid, voxel, (float(ox), float(oy), float(oz))


def simulate_density(coords, origin, shape, voxel, sigma=2.0, weights=None,
                     grid_chunk=200000):
    """Gaussian-blurred density of `coords` (n,3 in A) on a regular grid.

    Differentiable w.r.t. coords. shape = (nx, ny, nz); origin (3,) in A."""
    device, dtype = coords.device, coords.dtype
    nx, ny, nz = shape
    if weights is None:
        weights = coords.new_ones(coords.shape[0])
    xs = origin[0] + voxel * torch.arange(nx, device=device, dtype=dtype)
    ys = origin[1] + voxel * torch.arange(ny, device=device, dtype=dtype)
    zs = origin[2] + voxel * torch.arange(nz, device=device, dtype=dtype)
    grid = coords.new_zeros(nx * ny * nz)
    cutoff2 = (3.0 * sigma) ** 2
    # chunk along the slowest axis so each contiguous block of the flat grid
    # (row-major (nx, ny, nz)) is filled in one go
    di = max(1, grid_chunk // (ny * nz))
    for i0 in range(0, nx, di):
        i1 = min(i0 + di, nx)
        gx, gy, gz = torch.meshgrid(xs[i0:i1], ys, zs, indexing="ij")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)
        d2 = torch.cdist(pts, coords) ** 2                 # (P, n)
        contrib = torch.exp(-d2 / (2 * sigma * sigma)) * weights
        contrib = torch.where(d2 < cutoff2, contrib, torch.zeros_like(contrib))
        grid[i0 * ny * nz:i1 * ny * nz] = contrib.sum(1)
    return grid.reshape(nx, ny, nz)


def ccc(sim, exp):
    """Real-space cross correlation of two grids."""
    a = sim.reshape(-1)
    b = exp.reshape(-1).to(sim.dtype)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def flexible_fit(coords0, target_grid, origin, voxel, sigma=2.0, n_modes=20,
                 iters=300, lr=0.05, cutoff=15.0, verbose=False):
    """Fit `coords0` (n,3) into `target_grid` by optimising ANM mode
    coefficients with Adam on (1 - CCC).

    Returns (coords, info): fitted coordinates and the CCC trace, initial
    CCC, final CCC and the mode coefficients.
    """
    x0 = coords0.detach().clone().double()
    H = anm_hessian(x0, cutoff=cutoff)
    _, E = anm_modes_dense(H, n_modes)
    # coefficients carry RMSD-scale units: |E @ (c*sqrt(n))| = sqrt(n)|c|,
    # i.e. |c| is the RMSD of the induced deformation in A
    scale = float(x0.shape[0]) ** 0.5
    c = torch.zeros(E.shape[1], dtype=torch.float64, device=x0.device,
                    requires_grad=True)

    def coords_of(c):
        return x0 + (E @ (c * scale)).view_as(x0)

    opt = torch.optim.Adam([c], lr=lr)
    c0 = ccc(simulate_density(x0, origin, target_grid.shape, voxel, sigma),
             target_grid)
    hist = [c0]
    best = (c0, x0.clone(), c.detach().clone())
    for it in range(iters):
        opt.zero_grad()
        sim = simulate_density(coords_of(c), origin, target_grid.shape, voxel, sigma)
        loss = 1.0 - (sim.reshape(-1) @ target_grid.reshape(-1).to(sim.dtype)) \
            / (sim.reshape(-1).norm() * target_grid.reshape(-1).to(sim.dtype).norm() + 1e-12)
        loss.backward()
        opt.step()
        with torch.no_grad():
            cc = ccc(simulate_density(coords_of(c), origin,
                                      target_grid.shape, voxel, sigma), target_grid)
            hist.append(cc)
            if cc > best[0]:
                best = (cc, coords_of(c).clone(), c.detach().clone())
        if verbose and it % 25 == 0:
            print(f"  iter {it}: ccc {cc:.4f} (start {c0:.4f})")
    return best[1], {"ccc_start": c0, "ccc_final": best[0], "ccc_trace": hist,
                     "coefficients": best[2]}
