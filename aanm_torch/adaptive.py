"""Adaptive ANM stepping, a faithful torch port of ProDy calcAdaptiveANM
(AANM_ONEWAY / AANM_ALTERNATING) including the on-the-fly Fmin mode selection,
the adaptive n_modes doubling, and the three convergence criteria.

The per-step math lives in _step_core so the sequential, alternating and
batched drivers are guaranteed to apply identical updates; only the
eigendecomposition differs (per-matrix, or one stacked batched eigh for all
still-running transitions).
"""
import sys

import torch

from .anm import DENSE_CA_LIMIT, _eigh, anm_hessian, disconnected, solve_modes
from .geom import kabsch_fit, rmsd


class _State:
    __slots__ = ("n_modes", "d_first_norm", "rmsds", "reset_fmin", "converged")

    def __init__(self, n_modes, rmsd0):
        self.n_modes = n_modes
        self.d_first_norm = None
        self.rmsds = [rmsd0]
        self.reset_fmin = True
        self.converged = False


def _step_core(A, B, E, st, f, cutoff, min_rmsd_diff, target_rmsd, Fmin, Fmin_max):
    """One adaptive-ANM deformation of A toward B using precomputed modes E
    (3N, m). Mutates A and st; returns a dict with the post-step frame, rmsd,
    selected-mode count, defvec and critical cumulative overlap."""
    d = (B - A).reshape(-1)
    d_norm = float(d.norm())
    if st.d_first_norm is None:
        st.d_first_norm = d_norm

    overlaps = E.T @ d
    norm_ov = overlaps / d_norm
    c_sq = torch.cumsum(norm_ov * norm_ov, dim=0)

    fmin = Fmin
    if fmin is None:
        fmin = 0.0 if st.reset_fmin else 1.0 - (d_norm / st.d_first_norm) ** 0.5
    if fmin > Fmin_max:
        fmin = Fmin_max

    if fmin == 0.0 and st.reset_fmin:
        sel = torch.argmax(norm_ov.abs()).reshape(1)
    else:
        torf = c_sq <= fmin
        if bool(torf.any()) and not bool(torf.all()):
            last = torf.nonzero()[-1, 0]
            torf[last + 1] = True
        elif not bool(torf.any()):
            torf[0] = True
        sel = torf.nonzero().flatten()
    n_sel = sel.numel()

    v = E[:, sel] @ overlaps[sel]
    s = f * (v @ d) / (v @ v)
    A += s * v.view_as(A)

    r = float(rmsd(A, B))
    st.rmsds.append(r)

    if n_sel > st.n_modes - 5:
        st.n_modes = min(st.n_modes * 2, A.shape[0] - 6)
    if len(st.rmsds) > 4:
        dr = [abs(st.rmsds[i + 1] - st.rmsds[i]) for i in range(len(st.rmsds) - 1)]
        if all(x < min_rmsd_diff for x in dr[-4:]):
            st.converged = True
    if st.rmsds[-1] < target_rmsd or disconnected(A, cutoff):
        st.converged = True
    st.reset_fmin = False

    return {"frame": A.clone(), "rmsd": r, "n_sel": n_sel, "defvec": d,
            "c_sq": float(c_sq[sel].max())}


def _emit(callback_func, info, A, B):
    if callback_func is not None:
        callback_func(init=A, tar=B, defvec=info["defvec"],
                      c_sq=info["c_sq"], rmsd=info["rmsd"])


def _info(st):
    return {"rmsds": st.rmsds, "converged": st.converged, "n_modes_final": st.n_modes}


def _vprint(verbose, msg):
    if verbose:
        print(msg, file=sys.stderr)


def calc_adaptive_anm_one_way(coordsA, coordsB, n_steps, n_modes=20, f=0.2,
                              cutoff=15.0, gamma=1.0, Fmin=None, Fmin_max=0.6,
                              min_rmsd_diff=0.05, target_rmsd=1.0,
                              aligned=False, solver="auto", verbose=False,
                              callback_func=None):
    """Deform coordsA toward coordsB with at most `n_steps` adaptive ANM cycles.

    Returns (frames, info): frames holds the (N, 3) tensor after each completed
    step (the starting structure is not included; the caller appends the target
    endpoint if wanted, matching the original scripts).
    """
    A = coordsA.clone()
    B = coordsB
    if not aligned:
        A = kabsch_fit(A, B)
    st = _State(min(n_modes, A.shape[0] - 6), float(rmsd(A, B)))
    frames = []
    for step in range(n_steps):
        _, E = solve_modes(A, st.n_modes, cutoff, gamma, solver)
        info = _step_core(A, B, E, st, f, cutoff, min_rmsd_diff, target_rmsd,
                          Fmin, Fmin_max)
        frames.append(info["frame"])
        _vprint(verbose, f"  step {step + 1}/{n_steps} rmsd {info['rmsd']:.3f} "
                         f"sel {info['n_sel']} n_modes {st.n_modes}")
        _emit(callback_func, info, A, B)
        if st.converged:
            break
    return frames, _info(st)


def calc_adaptive_anm_alternating(coordsA, coordsB, n_steps, n_modes=20, f=0.2,
                                  cutoff=15.0, gamma=1.0, Fmin=None, Fmin_max=0.6,
                                  min_rmsd_diff=0.05, target_rmsd=1.0,
                                  aligned=False, solver="auto", verbose=False,
                                  callback_func=None):
    """Alternate deforming A toward B and B toward A (ProDy AANM_ALTERNATING).

    Returns (frames, info) where frames runs continuously from the exact A
    endpoint to the exact B endpoint (A, A-side steps, B-side steps reversed, B).
    """
    A = coordsA.clone()
    B = coordsB.clone()
    if not aligned:
        A = kabsch_fit(A, B)
    st = _State(min(n_modes, A.shape[0] - 6), float(rmsd(A, B)))
    framesA = [A.clone()]
    framesB = [B.clone()]
    for n in range(n_steps):
        _, E = solve_modes(A, st.n_modes, cutoff, gamma, solver)
        info = _step_core(A, B, E, st, f, cutoff, min_rmsd_diff, target_rmsd,
                          Fmin, Fmin_max)
        framesA.append(info["frame"])
        _vprint(verbose, f"  cycle {n + 1}/{n_steps} A->B rmsd {info['rmsd']:.3f}")
        _emit(callback_func, info, A, B)
        if st.converged:
            break

        _, E = solve_modes(B, st.n_modes, cutoff, gamma, solver)
        info = _step_core(B, A, E, st, f, cutoff, min_rmsd_diff, target_rmsd,
                          Fmin, Fmin_max)
        framesB.append(info["frame"])
        _vprint(verbose, f"  cycle {n + 1}/{n_steps} B->A rmsd {info['rmsd']:.3f}")
        _emit(callback_func, info, B, A)
        if st.converged:
            break
    return framesA + framesB[::-1], _info(st)


def calc_adaptive_anm_batched(pairs, n_steps, n_modes=20, f=0.2,
                              cutoff=15.0, gamma=1.0, Fmin=None, Fmin_max=0.6,
                              min_rmsd_diff=0.05, target_rmsd=1.0,
                              aligned=False, solver="auto", mem_budget=2 ** 31,
                              verbose=False, labels=None, callback_funcs=None):
    """One-way adaptive ANM for several (A, B) pairs in lockstep.

    All still-running transitions get their Hessians stacked and diagonalised
    in a single batched eigh call (dense solver regime), which saturates the
    GPU much better than sequential per-matrix calls. Non-dense solvers fall
    back to sequential runs. Pairs must share the same N; frames/info per pair
    are returned exactly as from calc_adaptive_anm_one_way.
    """
    T = len(pairs)
    if T == 0:
        return []
    N = pairs[0][0].shape[0]
    if any(p[0].shape[0] != N for p in pairs):
        raise ValueError("batched AANM requires all pairs to have the same number of atoms")

    if solver == "auto":
        batched = N <= DENSE_CA_LIMIT
    else:
        batched = solver == "eigh"
    if not batched:
        out = []
        for t, (a, b) in enumerate(pairs):
            cb = callback_funcs[t] if callback_funcs else None
            if verbose and labels:
                _vprint(True, f"[{labels[t]}]")
            out.append(calc_adaptive_anm_one_way(
                a, b, n_steps, n_modes=n_modes, f=f, cutoff=cutoff, gamma=gamma,
                Fmin=Fmin, Fmin_max=Fmin_max, min_rmsd_diff=min_rmsd_diff,
                target_rmsd=target_rmsd, aligned=aligned, solver=solver,
                verbose=verbose, callback_func=cb))
        return out

    As = [p[0].clone() for p in pairs]
    Bs = [p[1] for p in pairs]
    for t in range(T):
        if not aligned:
            As[t] = kabsch_fit(As[t], Bs[t])
    sts = [_State(min(n_modes, N - 6), float(rmsd(As[t], Bs[t]))) for t in range(T)]
    frames = [[] for _ in range(T)]
    done = [st.converged for st in sts]
    n3 = 3 * N
    per = max(1, int(mem_budget // (2 * n3 * n3 * As[0].element_size())))

    for step in range(n_steps):
        act = [t for t in range(T) if not done[t]]
        if not act:
            break
        for b0 in range(0, len(act), per):
            idxs = act[b0:b0 + per]
            Hs = torch.stack([anm_hessian(As[t], cutoff, gamma) for t in idxs])
            _, evecs = _eigh(Hs)
            for s, t in enumerate(idxs):
                E = evecs[s][:, 6:6 + sts[t].n_modes]
                info = _step_core(As[t], Bs[t], E, sts[t], f, cutoff,
                                  min_rmsd_diff, target_rmsd, Fmin, Fmin_max)
                frames[t].append(info["frame"])
                lbl = f"[{labels[t]}] " if labels else ""
                _vprint(verbose, f"  {lbl}step {step + 1}/{n_steps} "
                                 f"rmsd {info['rmsd']:.3f} sel {info['n_sel']}")
                if callback_funcs and callback_funcs[t]:
                    _emit(callback_funcs[t], info, As[t], Bs[t])
                if sts[t].converged:
                    done[t] = True
    return [(frames[t], _info(sts[t])) for t in range(T)]
