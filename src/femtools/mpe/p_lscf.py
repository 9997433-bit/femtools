"""Poly-reference Least-Squares Complex Frequency-domain estimator (p-LSCF).

The p-LSCF / "PolyMAX"-class estimator fits a right matrix-fraction model in the
:math:`z`-domain to the measured FRF matrix,

.. math::
    \\hat H_o(\\omega_k) = \\left(\\sum_{r=0}^{n} z_k^{\\,r} \\beta_{o,r}\\right)
        \\left(\\sum_{r=0}^{n} z_k^{\\,r} \\alpha_r\\right)^{-1},
    \\qquad z_k = e^{\\,j\\omega_k \\Delta t},

with a *common denominator* matrix polynomial.  Eliminating the numerator
coefficients analytically leaves compact reduced normal equations for
:math:`\\alpha`; the system poles are the eigenvalues of the companion matrix of
the denominator.

The method's appeal is that the resulting stabilisation diagrams are extremely
clean: spurious ("computational") poles are strongly damped and therefore easy
to reject, while physical poles stabilise at very low model order.

Reference: B. Peeters et al., *The PolyMAX frequency-domain method: a new
standard for modal parameter estimation*, Shock and Vibration 11 (2004).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .common import (
    ModalParameterResult,
    as_frf_array,
    lsfd,
    poles_from_roots,
    select_physical_poles,
    select_stable_modes,
    stabilization_diagram,
)

__all__ = ["poly_lscf", "polymax", "denominator_coefficients"]


def denominator_coefficients(
    H: np.ndarray,
    freq_hz: np.ndarray,
    order: int,
    dt: float,
    *,
    weighting: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the p-LSCF reduced normal equations for the denominator polynomial.

    Returns ``alpha`` of shape ``(order + 1, n_in, n_in)`` with
    ``alpha[order] == I`` (the standard normalisation).
    """
    n_out, n_in, n_freq = H.shape
    n = int(order)
    w = 2.0 * math.pi * np.asarray(freq_hz, dtype=float)
    z = np.exp(1j * w * float(dt))
    # Omega[k, r] = z_k^r  ->  (n_freq, n+1)
    Omega = np.vander(z, n + 1, increasing=True)
    if weighting is not None:
        # Weighting scales the whole residual of each spectral line, i.e. both
        # the numerator basis Omega and the product Omega (x) H that follows.
        Omega = Omega * np.asarray(weighting, dtype=float).ravel()[:, None]

    m = (n + 1) * n_in
    M = np.zeros((m, m))
    R = np.real(Omega.conj().T @ Omega)  # (n+1, n+1)
    R_reg = R + 1.0e-14 * (np.trace(R) / R.shape[0]) * np.eye(R.shape[0])

    for o in range(n_out):
        Ho = H[o]  # (n_in, n_freq)
        # X[k, r*n_in + i] = z_k^r * H_{o,i}(w_k)  ==  kron(Omega_k, H_o(w_k))
        X = (Omega[:, :, None] * Ho.T[:, None, :]).reshape(n_freq, m)
        S = np.real(Omega.conj().T @ X)  # (n+1, m)
        T = np.real(X.conj().T @ X)  # (m, m)
        M += 2.0 * (T - S.T @ np.linalg.solve(R_reg, S))

    # Enforce alpha_n = I and solve for the remaining blocks.
    p = n * n_in
    Maa = M[:p, :p]
    Mab = M[:p, p:]
    reg = 1e-12 * (np.trace(Maa) / max(p, 1)) * np.eye(p)
    alpha_low = -np.linalg.solve(Maa + reg, Mab)  # (n*n_in, n_in)
    alpha = np.zeros((n + 1, n_in, n_in))
    alpha[:n] = alpha_low.reshape(n, n_in, n_in)
    alpha[n] = np.eye(n_in)
    return alpha


def _companion_poles(alpha: np.ndarray, dt: float) -> np.ndarray:
    """Continuous-time poles from the denominator matrix polynomial."""
    n1, n_in, _ = alpha.shape
    n = n1 - 1
    if n_in == 1:
        coeffs = alpha[:, 0, 0][::-1]  # highest power first
        z = np.roots(coeffs)
    else:
        A = np.zeros((n * n_in, n * n_in))
        A[: (n - 1) * n_in, n_in:] = np.eye((n - 1) * n_in)
        for r in range(n):
            A[(n - 1) * n_in :, r * n_in : (r + 1) * n_in] = -alpha[r].T
        z = np.linalg.eigvals(A)
    return poles_from_roots(z, dt)


def poly_lscf(
    frf: Any,
    freq_hz: ArrayLike | None = None,
    order: int = 20,
    *,
    dt: float | None = None,
    fs: float | None = None,
    orders: Sequence[int] | None = None,
    order_min: int | None = None,
    order_step: int = 2,
    f_range: tuple[float, float] | None = None,
    n_modes: int | None = None,
    max_damping: float = 0.25,
    min_damping: float = 0.0,
    weighting: np.ndarray | str | None = None,
    mode_shapes: bool = True,
    stabilization: bool = True,
    stabilization_level: str = "d",
    tol_freq: float = 0.01,
    tol_damp: float = 0.05,
    tol_mac: float = 0.02,
    min_count: int = 2,
    cluster_tol: float = 0.01,
    lower_residual: bool = True,
    upper_residual: bool = True,
) -> ModalParameterResult:
    """Estimate modal parameters from FRFs with the p-LSCF / PolyMAX method.

    Parameters
    ----------
    frf:
        Complex FRF data shaped ``(n_out, n_in, n_freq)`` (also accepts
        ``(n_out, n_freq)`` and objects exposing ``.H``/``.frf``).
    freq_hz:
        Frequency axis in Hz; read from ``frf.freq_hz`` when omitted.
    order:
        Maximum polynomial order.  With ``stabilization=True`` (default) the
        estimator sweeps orders and builds a stabilisation diagram; the physical
        poles are the clusters that repeat across orders.
    orders / order_min / order_step:
        Explicit list of model orders, or the sweep range
        ``range(order_min, order + 1, order_step)`` (default ``order_min=4``).
    f_range:
        ``(f_lo, f_hi)`` band of interest.  Data outside is discarded, which
        both speeds up and stabilises the fit.
    n_modes:
        Keep only the ``n_modes`` best-supported (most frequently stabilising)
        modes.
    max_damping:
        Reject poles with a damping ratio above this value (spurious poles are
        typically heavily damped).
    mode_shapes:
        Run :func:`femtools.mpe.common.lsfd` afterwards to obtain residues,
        mode shapes and residual terms.

    Returns
    -------
    ModalParameterResult

    Notes
    -----
    The discrete-time step defaults to ``dt = 1 / (2 f_max)`` over the selected
    band, i.e. the band is mapped onto the full unit circle, which is the
    conditioning-optimal choice for a frequency-domain fit.
    """
    if freq_hz is None:
        freq_hz = getattr(frf, "freq_hz", None)
        if freq_hz is None:
            freq_hz = getattr(frf, "frequencies", None)
        if freq_hz is None:
            raise ValueError("freq_hz must be given (or carried by the FRF object)")
    H = as_frf_array(frf)
    f = np.asarray(freq_hz, dtype=float).ravel()
    if f.size != H.shape[2]:
        raise ValueError(f"freq_hz has {f.size} lines but FRF has {H.shape[2]}")

    if f_range is not None:
        sel = (f >= f_range[0]) & (f <= f_range[1])
        if sel.sum() < 8:
            raise ValueError("f_range selects fewer than 8 spectral lines")
        f, H = f[sel], H[:, :, sel]

    if dt is None:
        dt = 1.0 / (2.0 * float(fs)) if fs else 1.0 / (2.0 * float(np.max(f)))

    wt = None
    if isinstance(weighting, str):
        kind = weighting.lower()
        if kind in ("unity", "none", "uniform"):
            wt = None
        elif kind in ("inverse", "1/h", "amplitude"):
            mag = np.sqrt(np.mean(np.abs(H) ** 2, axis=(0, 1)))
            wt = 1.0 / np.maximum(mag, 1e-30)
            wt = wt / np.mean(wt)
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
    elif weighting is not None:
        wt = np.asarray(weighting, dtype=float).ravel()

    band = (float(f.min()), float(f.max())) if f_range is None else f_range

    if orders is not None:
        order_list = sorted(int(o) for o in orders)
    elif stabilization:
        omin = int(order_min) if order_min is not None else max(2, min(4, order))
        order_list = list(range(omin, int(order) + 1, max(1, int(order_step))))
        if order_list[-1] != int(order):
            order_list.append(int(order))
    else:
        order_list = [int(order)]

    pole_sets: dict[int, np.ndarray] = {}
    for o in order_list:
        try:
            alpha = denominator_coefficients(H, f, o, dt, weighting=wt)
            s = _companion_poles(alpha, dt)
        except np.linalg.LinAlgError:  # pragma: no cover - singular normal equations
            continue
        pole_sets[o] = select_physical_poles(
            s, f_range=band, max_damping=max_damping, min_damping=min_damping
        )

    if not pole_sets:
        raise RuntimeError("p-LSCF failed to produce any physical poles")

    diagram = None
    if len(pole_sets) > 1:
        diagram = stabilization_diagram(
            pole_sets, tol_freq=tol_freq, tol_damp=tol_damp, tol_mac=tol_mac
        )
        reps, counts = select_stable_modes(
            diagram,
            level=stabilization_level,
            cluster_tol=cluster_tol,
            min_count=min_count,
            n_modes=n_modes,
        )
        if reps.size == 0:
            reps = pole_sets[order_list[-1]]
            counts = np.ones(reps.size, dtype=int)
    else:
        reps = pole_sets[order_list[-1]]
        counts = np.ones(reps.size, dtype=int)

    if n_modes is not None and reps.size > n_modes:
        keep = np.argsort(-counts)[: int(n_modes)]
        reps, counts = reps[np.sort(keep)], counts[np.sort(keep)]

    idx = np.argsort(np.abs(reps))
    lam = reps[idx]
    wn = np.abs(lam)
    result = ModalParameterResult(
        freq_hz=wn / (2.0 * math.pi),
        damping=np.where(wn > 0, -lam.real / wn, 0.0),
        poles=lam,
        order=order_list[-1],
        method="p-LSCF",
        stabilization=diagram,
        extras={"dt": dt, "orders": order_list, "cluster_counts": counts[idx]},
    )

    if mode_shapes and lam.size:
        fit = lsfd(
            H, f, lam, lower_residual=lower_residual, upper_residual=upper_residual
        )
        result.residues = fit["residues"]
        result.mode_shapes = fit["mode_shapes"]
        result.participation = fit["participation"]
        result.lower_residual = fit["lower_residual"]
        result.upper_residual = fit["upper_residual"]
        result.fit_error = fit["fit_error"]
    return result


def polymax(*args: Any, **kwargs: Any) -> ModalParameterResult:
    """Alias for :func:`poly_lscf` (the method's common commercial name)."""
    return poly_lscf(*args, **kwargs)
