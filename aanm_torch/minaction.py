"""Least-action path on a two-endpoint elastic potential (MinActionPath-style).

The classic MinActionPath (Franklin, Koehl, Doniach & Delarue, JCP 2007)
minimises the overdamped-Langevin action of a path between two structures on
an elastic-network potential. This torch version discretises M images,
minimises the Onsager-Machlup action

    S = sum_k [ |x_{k+1}-x_k|^2 / (4 D dt) + (dt / 2) (V(x_k) + V(x_{k+1})) ]

with V(x) = 1/2 (x-xA)^T H_A (x-xA) + 1/2 (x-xB)^T H_B (x-xB) built from the
two endpoint ANMs, and optimises the interior images by autograd (Adam).
Endpoints stay fixed. The potential term is normalised by the endpoint
energies so the kinetic/potential balance is system-size independent; the
absolute action scale is therefore arbitrary — compare paths relatively.
"""
import torch

from .anm import anm_hessian


def elastic_potential(x, xA, HA, xB, HB):
    """Two-endpoint ANM potential, x/xA/xB (n,3) flattened H's (3n,3n)."""
    dA = (x - xA).reshape(-1)
    dB = (x - xB).reshape(-1)
    return 0.5 * (dA @ (HA @ dA) + dB @ (HB @ dB))


def path_action(path, xA, HA, xB, HB, dt=1.0, diff=1.0, vscale=1.0):
    """Onsager-Machlup action of a list of (n,3) images (endpoints included)."""
    kin = 0.0
    pot = 0.0
    for k in range(len(path) - 1):
        step = (path[k + 1] - path[k]).reshape(-1)
        kin = kin + step @ step / (4.0 * diff * dt)
        pot = pot + (dt / 2.0) * (elastic_potential(path[k], xA, HA, xB, HB)
                                  + elastic_potential(path[k + 1], xA, HA, xB, HB))
    return kin + pot / vscale


def min_action_path(coordsA, coordsB, n_images=12, n_iters=1500, lr=0.1,
                    dt=1.0, diff=1.0, cutoff=15.0, aligned=True, verbose=False):
    """Least-action path between two endpoint structures.

    coordsA/coordsB: (n,3) tensors (A or nm — units only set the action scale).
    With aligned=True (default) B is Kabsch-fitted onto A first: the path is
    defined up to rigid motion anyway, and an unaligned pair buries the
    elastic deformations under a meaningless global transform.
    Returns (frames, info): frames is a list of n_images (n,3) tensors from A
    to B; info carries the action history and the linear-path action.
    """
    xA = coordsA.detach().clone()
    xB = coordsB.detach().clone()
    if aligned:
        from .geom import kabsch_fit
        xB = kabsch_fit(xB, xA)
    HA = anm_hessian(xA.double(), cutoff=cutoff)
    HB = anm_hessian(xB.double(), cutoff=cutoff)
    # normalise the potential term by endpoint energies (size-independent balance)
    vscale = max(float(elastic_potential(xA, xA, HA, xB, HB)
                       + elastic_potential(xB, xA, HA, xB, HB)), 1e-12)

    M = n_images
    ts = torch.linspace(0.0, 1.0, M, dtype=torch.float64)
    interior = torch.stack([(1 - t) * xA + t * xB for t in ts[1:-1]]).clone()
    interior.requires_grad_(True)
    hist = []

    def total_action():
        return path_action([xA] + list(interior) + [xB], xA, HA, xB, HB, dt, diff, vscale)

    with torch.no_grad():
        lin_action = float(total_action())

    # Adam with best-so-far bookkeeping: converges to the same minimum as
    # LBFGS on this smooth objective (verified) and keeps one gradient per
    # step, which scales and composes better
    opt = torch.optim.Adam([interior], lr=lr)
    best = (lin_action, [f.detach().clone() for f in interior])
    for it in range(n_iters):
        opt.zero_grad()
        S = total_action()
        S.backward()
        opt.step()
        with torch.no_grad():
            s = float(total_action())
            hist.append(s)
            if s < best[0]:
                best = (s, [f.detach().clone() for f in interior])
        if verbose and it % 100 == 0:
            print(f"  iter {it}: action {s:.4f} (linear {lin_action:.4f})")
    frames = [xA] + best[1] + [xB]
    return frames, {"action": hist, "linear_action": lin_action,
                    "final_action": best[0], "vscale": vscale}
