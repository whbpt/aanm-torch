"""ClustENM-style ensemble generation (torch).

Iterates the classic ClustENM loop (Kurkcuoglu & Doruker, 2016; ProDy has a
reference implementation): around each representative conformer, sample
thermal excitations of the softest ANM modes, filter unphysical samples
(clashes), cluster, and keep cluster representatives as the next iteration's
seeds — the network is rebuilt at each representative, so the ensemble
diffuses through conformational space.

The original relaxes samples by all-atom energy minimisation with a force
field; here relaxation is implicit in the sampling (only soft modes are
excited, so no local strain is introduced) plus the clash filter. All steps
batch over conformers.
"""
import torch

from .anm import anm_hessian, anm_modes_dense
from .clash import clash_count


def _sample_around(rep, n_gen, n_modes, amp, cutoff, generator, device):
    """Thermal soft-mode excitations, scale-free: amplitudes c_k ~
    sqrt(temp/lambda_k) with temp = amp^2 * mean(lambda_soft), so `amp`
    directly sets the expected RMSD spread in A regardless of system size."""
    H = anm_hessian(rep.double(), cutoff=cutoff)
    evals, evecs = anm_modes_dense(H, n_modes)
    temp = amp * amp * float(evals.mean())
    c = torch.randn(n_gen, evals.numel(), generator=generator,
                    dtype=torch.float64, device=device) \
        * (temp / evals.clamp(min=1e-12)).sqrt()
    dx = (evecs @ c.T).T                          # (n_gen, 3n)
    return rep.unsqueeze(0) + dx.view(n_gen, -1, 3)


def _kmeans(x, k, iters=12, generator=None):
    """Plain Lloyd's k-means on flattened coordinates; returns centers (k,n,3)."""
    flat = x.reshape(x.shape[0], -1)
    idx = torch.randperm(x.shape[0], generator=generator)[:k]
    centers = flat[idx].clone()
    for _ in range(iters):
        d2 = torch.cdist(flat, centers)
        assign = d2.argmin(dim=1)
        for j in range(k):
            sel = flat[assign == j]
            if sel.numel():
                centers[j] = sel.mean(0)
    return centers.reshape(k, *x.shape[1:])


def clustenm(coords, n_iters=3, n_gen=32, n_modes=10, amp=1.5,
             cutoff=15.0, n_clusters=4, max_clashes=4, clash_cutoff=4.0,
             seed=0, verbose=False):
    """Generate a coarse equilibrium-like ensemble around one or more seed
    conformations (unbiased local fluctuations, ClustENM-style; transitions
    between distant basins need steering, e.g. the adaptive/pathway modules).

    coords: (n,3) tensor or list of (n,3) seed conformers.
    Returns (representatives, ensemble): representatives is a list of
    (n,3) tensors, ensemble all accepted conformers.
    `amp` sets the RMSD scale of each sampling round in A (softest modes get
    the largest kicks, stiffer soft modes proportionally less).
    """
    device = coords[0].device if isinstance(coords, list) else coords.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    reps = ([c.detach().clone() for c in coords] if isinstance(coords, list)
            else [coords.detach().clone()])
    all_confs = []
    for it in range(n_iters):
        batch = []
        for rep in reps:
            confs = _sample_around(rep, n_gen, n_modes, amp, cutoff, g, device)
            # clash filter (keep the seed itself always)
            keep = [rep]
            for c in confs:
                n_cl, _ = clash_count(c, cutoff=clash_cutoff, skip=2)
                if n_cl <= max_clashes:
                    keep.append(c)
            batch.extend(keep)
        x = torch.stack(batch)
        centers = _kmeans(x, min(n_clusters, x.shape[0]), generator=g)
        reps = [centers[j] for j in range(centers.shape[0])]
        all_confs.append(x)
        if verbose:
            print(f"  iter {it}: {x.shape[0]} conformers -> {len(reps)} reps")
    return reps, torch.cat(all_confs)
