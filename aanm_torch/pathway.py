"""ANMPathway-style two-sided pathway (torch).

Inspired by ANMPathway.o (Das/Atilgan-group, 2015): deform both endpoints
toward each other with adaptive ANM stepping, and relax every recorded frame
so that its displacement since the previous frame lies entirely in the
softest-mode subspace of its own network — i.e. no inter-frame motion excites
stiff (locally strained) directions, the physical criterion behind
"minimize-then-step" pathway schemes.

Simplified relative to the published server (no explicit transition-state
search on a switching potential); the relaxation criterion here is the
soft-mode purity of each step, which is checkable exactly.
"""
import torch

from .anm import anm_hessian, anm_modes_dense
from .geom import kabsch_fit, rmsd


def _soft_step(front, other, n_modes, f, cutoff):
    """One combined-mode adaptive step of `front` toward `other`."""
    d = (other - front).reshape(-1)
    H = anm_hessian(front.double(), cutoff=cutoff)
    _, E = anm_modes_dense(H, n_modes)
    v = E @ (E.T @ d)
    vv = float(v @ v)
    if vv < 1e-12:
        return front, H
    s = f * float(v @ d) / vv
    return front + (s * v).view_as(front), H


def _relax(prev, cur, H, n_relax):
    """Project (cur - prev) onto the softest n_relax modes of H (plus the 6
    rigid modes, which carry no strain). Returns the relaxed structure."""
    delta = (cur - prev).reshape(-1)
    evals, evecs = torch.linalg.eigh(H)
    n_keep = 6 + n_relax
    E = evecs[:, :n_keep]
    return prev + (E @ (E.T @ delta)).view_as(prev)


def settle_softness(frames, n_relax=6, cutoff=15.0, passes=2):
    """Re-relax interior frames in path order so every step (frames[i-1] ->
    frames[i]) is a pure soft-mode deformation of H(frames[i-1]). Needed
    because the two halves are generated in opposite directions and the
    meeting junction is never projected. Endpoints stay fixed."""
    frames = [f.clone() for f in frames]
    for _ in range(passes):
        for i in range(1, len(frames) - 1):
            H = anm_hessian(frames[i - 1].double(), cutoff=cutoff)
            frames[i] = _relax(frames[i - 1], frames[i], H, n_relax)
    return frames


def stiff_fraction(prev, cur, cutoff=15.0, n_soft=6):
    """Fraction of (cur-prev)^T H(prev) (cur-prev) carried by modes stiffer
    than the n_soft softest — 0 for a purely soft step."""
    delta = (cur - prev).reshape(-1)
    H = anm_hessian(prev.double(), cutoff=cutoff)
    evals, evecs = torch.linalg.eigh(H)
    E_stiff = evecs[:, 6 + n_soft:]
    c = E_stiff.T @ delta
    num = float((evals[6 + n_soft:] * c * c).sum())
    den = float((evals * (evecs.T @ delta) ** 2).sum())
    return num / max(den, 1e-30)


def anm_pathway(coordsA, coordsB, n_cycles=40, n_modes=10, f=0.3,
                cutoff=15.0, n_relax=6, target_rmsd=1.0, aligned=True):
    """Two-sided adaptive pathway. Returns (frames, info): frames run from the
    exact A endpoint to the exact B endpoint (aligned frame if aligned=True),
    meeting in the middle; info has the meeting RMSD and cycle count.

    Every interior step is a pure soft-mode deformation (see settle_softness);
    the closing step into the fixed B endpoint is exempt — an anchored
    endpoint cannot be reprojected."""
    A = coordsA.detach().clone()
    B = coordsB.detach().clone()
    if aligned:
        B = kabsch_fit(B, A)
    front_a, front_b = A.clone(), B.clone()
    rec_a, rec_b = A.clone(), B.clone()          # last recorded frames
    frames_a, frames_b = [A.clone()], [B.clone()]

    for cycle in range(n_cycles):
        new_a, H_a = _soft_step(front_a, front_b, n_modes, f, cutoff)
        new_b, H_b = _soft_step(front_b, front_a, n_modes, f, cutoff)
        # relax: each recorded step must be a pure soft-mode deformation
        front_a = _relax(rec_a, new_a, H_a, n_relax)
        front_b = _relax(rec_b, new_b, H_b, n_relax)
        rec_a, rec_b = front_a.clone(), front_b.clone()
        frames_a.append(front_a.clone())
        frames_b.append(front_b.clone())
        gap = float(rmsd(front_a, front_b))
        if gap < target_rmsd:
            break

    full = frames_a + frames_b[::-1]
    full = settle_softness(full, n_relax=n_relax, cutoff=cutoff)
    return full, {"meeting_rmsd": gap, "cycles": cycle + 1}
