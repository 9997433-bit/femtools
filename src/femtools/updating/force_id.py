"""Inverse force identification from measured FRFs and operating responses.

For a linear system the operating response spectrum is
:math:`X(\\omega) = H(\\omega) F(\\omega)`.  Recovering :math:`F` requires
inverting a frequency-dependent, typically ill-conditioned transfer matrix, so
this module offers

* plain (pseudo-)inversion / least squares,
* truncated SVD (TSVD),
* Tikhonov regularisation with a fixed, GCV-selected or L-curve-selected
  parameter,

evaluated independently on every frequency line.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["ForceIdResult", "identify_harmonic_forces", "tikhonov_solve"]


@dataclass
class ForceIdResult:
    """Identified harmonic forces.

    Attributes
    ----------
    forces:
        ``(n_in, n_freq)`` complex force spectra (or ``(n_in,)`` for a single line).
    freq_hz:
        Frequency axis.
    reconstruction:
        ``H @ forces`` — the response explained by the identified forces.
    residual:
        ``response - reconstruction``.
    relative_residual:
        Per-line ``||residual|| / ||response||``.
    condition_number:
        Per-line 2-norm condition number of ``H``.
    alpha:
        Per-line regularisation parameter actually used.
    rank:
        Per-line numerical/effective rank used in the inversion.
    """

    forces: np.ndarray
    freq_hz: np.ndarray
    reconstruction: np.ndarray
    residual: np.ndarray
    relative_residual: np.ndarray
    condition_number: np.ndarray
    alpha: np.ndarray
    rank: np.ndarray
    method: str = "tikhonov"
    singular_values: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # -- convenience ----------------------------------------------------
    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self.forces if dtype is None else self.forces.astype(dtype)

    def __getitem__(self, key: Any) -> Any:
        return self.forces[key]

    @property
    def F(self) -> np.ndarray:
        return self.forces

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.forces)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.forces))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.forces.shape

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ForceIdResult(shape={self.forces.shape}, method={self.method!r}, "
            f"mean_rel_residual={float(np.mean(self.relative_residual)):.3e})"
        )


# ----------------------------------------------------------------------
def _as_frf_array(frf: Any) -> np.ndarray:
    """Coerce an FRF container into a complex ``(n_out, n_in, n_freq)`` array."""
    for attr in ("H", "frf", "data", "values"):
        if hasattr(frf, attr) and not isinstance(frf, np.ndarray):
            frf = getattr(frf, attr)
            break
    H = np.asarray(frf)
    if H.dtype.kind != "c":
        H = H.astype(complex)
    return H


def _gcv_alpha(
    s: np.ndarray, beta: np.ndarray, m: int, residual_floor: float = 0.0
) -> float:
    """Generalised cross-validation choice of the Tikhonov parameter.

    ``residual_floor`` is ``||b||^2 - ||U^H b||^2``, the part of the measurement
    that lies outside the range of ``A``.  With a thin SVD it must be added
    back, otherwise GCV degenerates to zero regularisation whenever the system
    is barely over-determined.
    """
    s = np.asarray(s, dtype=float)
    beta = np.abs(np.asarray(beta).ravel()[: s.size])
    if s.size == 0 or s[0] <= 0:
        return 0.0
    lo, hi = max(s[-1], s[0] * 1e-10), s[0]
    grid = np.geomspace(lo * 1e-3, hi * 1.0e1, 120)
    best_a, best_v = grid[0], math.inf
    for a in grid:
        f = a**2 / (s**2 + a**2)  # filter factors on the residual
        num = float(np.sum((f * beta) ** 2)) + max(residual_floor, 0.0)
        den = (m - s.size + float(np.sum(f))) ** 2
        v = num / den if den > 0 else math.inf
        if v < best_v:
            best_v, best_a = v, a
    return float(best_a)


def _lcurve_alpha(
    s: np.ndarray, beta: np.ndarray, residual_floor: float = 0.0
) -> float:
    """Maximum-curvature (L-curve) choice of the Tikhonov parameter."""
    s = np.asarray(s, dtype=float)
    beta = np.abs(np.asarray(beta).ravel()[: s.size])
    if s.size == 0 or s[0] <= 0:
        return 0.0
    grid = np.geomspace(max(s[-1], s[0] * 1e-12) * 1e-3, s[0] * 10.0, 120)
    rho, eta = [], []
    for a in grid:
        f = s**2 / (s**2 + a**2)
        res2 = float(np.sum(((1 - f) * beta) ** 2)) + max(residual_floor, 0.0)
        rho.append(0.5 * np.log(res2 + 1e-300))
        eta.append(np.log(np.sqrt(np.sum((f * beta / np.maximum(s, 1e-300)) ** 2)) + 1e-300))
    rho_a, eta_a = np.asarray(rho), np.asarray(eta)
    if rho_a.size < 5:
        return float(grid[0])
    d1r, d1e = np.gradient(rho_a), np.gradient(eta_a)
    d2r, d2e = np.gradient(d1r), np.gradient(d1e)
    curv = np.abs(d1r * d2e - d1e * d2r) / np.power(d1r**2 + d1e**2, 1.5) + 1e-300
    return float(grid[int(np.argmax(curv))])


def tikhonov_solve(
    A: np.ndarray, b: np.ndarray, alpha: float
) -> tuple[np.ndarray, float, int]:
    """Solve ``min ||A x - b||^2 + alpha^2 ||x||^2`` via SVD.

    Returns ``(x, alpha, effective_rank)``.
    """
    U, s, Vh = np.linalg.svd(np.asarray(A), full_matrices=False)
    beta = U.conj().T @ np.asarray(b)
    if alpha <= 0:
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        keep = s > tol
        x = Vh.conj().T @ np.where(keep, beta / np.where(keep, s, 1.0), 0.0)
        return x, 0.0, int(np.count_nonzero(keep))
    f = s / (s**2 + alpha**2)
    x = Vh.conj().T @ (f * beta)
    rank = float(np.sum(s**2 / (s**2 + alpha**2)))
    return x, float(alpha), int(round(rank))


def identify_harmonic_forces(
    frf: Any,
    response: Any,
    freq_hz: ArrayLike | None = None,
    *,
    method: str = "tikhonov",
    alpha: Any = "gcv",
    rank: int | None = None,
    max_condition: float | None = None,
    rcond: float = 1.0e-12,
    input_dofs: Sequence[Any] | None = None,
    output_dofs: Sequence[Any] | None = None,
    return_reconstruction: bool = True,
) -> ForceIdResult:
    """Identify the harmonic force spectra that produced ``response``.

    Parameters
    ----------
    frf:
        Complex transfer matrix, shaped ``(n_out, n_in, n_freq)`` (the
        :func:`femtools.dynamics.frf.modal_frf` convention), ``(n_out, n_freq)``
        for a single input, or ``(n_out, n_in)`` / ``(n_out,)`` for a single
        frequency line.  Objects exposing ``.H`` or ``.frf`` are unwrapped.
    response:
        Measured operating response spectra, ``(n_out, n_freq)`` or ``(n_out,)``.
    freq_hz:
        Frequency axis; taken from ``frf.freq_hz`` when available.
    method:
        ``"tikhonov"`` (default), ``"tsvd"``, ``"pinv"`` / ``"lstsq"``.
    alpha:
        Tikhonov parameter: a float, a per-line array, ``"gcv"`` (default),
        ``"lcurve"``, or ``None``/``0`` for no regularisation.
    rank:
        Number of retained singular values for ``method="tsvd"``.
    max_condition:
        Optional hard cap on the effective condition number: the Tikhonov
        parameter is raised to at least ``s_max / max_condition``.  A cheap and
        very effective safety net when the excitation set is nearly rank
        deficient and the automatic rules under-regularise.

    Returns
    -------
    ForceIdResult

    Notes
    -----
    Regularisation is essential near anti-resonances and whenever the number of
    measured responses barely exceeds the number of unknown forces.  GCV (the
    default) is close to optimal for well-conditioned transfer matrices and
    correctly backs off to almost no regularisation on clean data; ``"lcurve"``
    is the more robust choice for strongly ill-conditioned excitation sets but
    over-smooths noise-free data.  ``max_condition`` is a blunt but reliable
    safety net for both.
    """
    if freq_hz is None:
        freq_hz = getattr(frf, "freq_hz", None)
        if freq_hz is None:
            freq_hz = getattr(frf, "frequencies", None)

    H = _as_frf_array(frf)
    X = np.asarray(response)
    if X.dtype.kind != "c":
        X = X.astype(complex)

    single_line = False
    if H.ndim == 1:
        H = H[:, None, None]
        single_line = True
    elif H.ndim == 2:
        if X.ndim == 1 and H.shape[0] == X.shape[0]:
            # (n_out, n_in) at one frequency
            H = H[:, :, None]
            single_line = True
        else:
            # (n_out, n_freq) single input
            H = H[:, None, :]
    elif H.ndim != 3:
        raise ValueError(f"FRF must have 1-3 dimensions, got shape {H.shape}")

    n_out, n_in, n_freq = H.shape

    if X.ndim == 1:
        X = X[:, None] if single_line or n_freq == 1 else X[None, :]
    if X.shape[0] != n_out:
        if X.shape[-1] == n_out and X.ndim == 2:
            X = X.T
        else:
            raise ValueError(
                f"response has {X.shape[0]} rows but FRF has {n_out} outputs"
            )
    if X.shape[1] != n_freq:
        raise ValueError(
            f"response has {X.shape[1]} frequency lines but FRF has {n_freq}"
        )

    if output_dofs is not None:
        sel = np.asarray(output_dofs, dtype=int)
        H, X = H[sel, :, :], X[sel, :]
        n_out = H.shape[0]
    if input_dofs is not None:
        sel = np.asarray(input_dofs, dtype=int)
        H = H[:, sel, :]
        n_in = H.shape[1]

    f_axis = (
        np.arange(n_freq, dtype=float)
        if freq_hz is None
        else np.asarray(freq_hz, dtype=float).ravel()[:n_freq]
    )

    meth = str(method).lower()
    if meth in ("lsq", "lstsq", "ls"):
        meth = "pinv"
    if meth not in ("tikhonov", "tsvd", "pinv"):
        raise ValueError(f"unknown method {method!r}")

    alpha_spec = alpha
    alpha_arr = None
    if isinstance(alpha, (int, float, np.floating)) and not isinstance(alpha, bool):
        alpha_arr = np.full(n_freq, float(alpha))
    elif alpha is None:
        alpha_arr = np.zeros(n_freq)
    elif not isinstance(alpha, str):
        alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float).ravel(), (n_freq,)).copy()

    F = np.zeros((n_in, n_freq), dtype=complex)
    cond = np.zeros(n_freq)
    used_alpha = np.zeros(n_freq)
    used_rank = np.zeros(n_freq, dtype=int)
    svals = np.zeros((min(n_out, n_in), n_freq))

    for k in range(n_freq):
        A = H[:, :, k]
        b = X[:, k]
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        svals[: s.size, k] = s
        cond[k] = float(s[0] / s[-1]) if s.size and s[-1] > 0 else math.inf
        beta = U.conj().T @ b

        if meth == "pinv":
            tol = rcond * (s[0] if s.size else 0.0)
            keep = s > tol
            xk = Vh.conj().T @ np.where(keep, beta / np.where(keep, s, 1.0), 0.0)
            used_alpha[k] = 0.0
            used_rank[k] = int(np.count_nonzero(keep))
        elif meth == "tsvd":
            r = int(rank) if rank is not None else int(np.count_nonzero(s > rcond * s[0]))
            r = max(1, min(r, s.size))
            xk = Vh[:r].conj().T @ (beta[:r] / s[:r])
            used_alpha[k] = 0.0
            used_rank[k] = r
        else:  # tikhonov
            floor = max(
                float(np.vdot(b, b).real) - float(np.vdot(beta, beta).real), 0.0
            )
            if alpha_arr is not None:
                a = float(alpha_arr[k])
            elif str(alpha_spec).lower() == "lcurve":
                a = _lcurve_alpha(s, beta, floor)
            else:
                a = _gcv_alpha(s, beta, n_out, floor)
            if max_condition is not None and s.size:
                a = max(a, float(s[0]) / float(max_condition))
            if a > 0:
                filt = s / (s**2 + a**2)
                xk = Vh.conj().T @ (filt * beta)
                used_rank[k] = int(round(float(np.sum(s**2 / (s**2 + a**2)))))
            else:
                keep = s > rcond * (s[0] if s.size else 0.0)
                xk = Vh.conj().T @ np.where(keep, beta / np.where(keep, s, 1.0), 0.0)
                used_rank[k] = int(np.count_nonzero(keep))
            used_alpha[k] = a
        F[:, k] = xk

    recon = np.einsum("oik,ik->ok", H, F) if return_reconstruction else np.zeros_like(X)
    resid = X - recon
    denom = np.linalg.norm(X, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, np.linalg.norm(resid, axis=0) / denom, 0.0)

    if single_line:
        return ForceIdResult(
            forces=F[:, 0],
            freq_hz=f_axis[:1],
            reconstruction=recon[:, 0],
            residual=resid[:, 0],
            relative_residual=rel[:1],
            condition_number=cond[:1],
            alpha=used_alpha[:1],
            rank=used_rank[:1],
            method=meth,
            singular_values=svals[:, 0],
        )
    return ForceIdResult(
        forces=F,
        freq_hz=f_axis,
        reconstruction=recon,
        residual=resid,
        relative_residual=rel,
        condition_number=cond,
        alpha=used_alpha,
        rank=used_rank,
        method=meth,
        singular_values=svals,
    )
