"""Least Squares Complex Exponential (LSCE) time-domain modal estimator.

LSCE is the multi-response generalisation of Prony's method.  Every impulse
response function of a linear time-invariant system is a sum of complex
exponentials,

.. math::
    h(k\\Delta t) = \\sum_{r=1}^{2N} A_r\\, z_r^{\\,k}, \\qquad z_r = e^{\\lambda_r \\Delta t},

so all the ``z_r`` are roots of one common (real) autoregressive polynomial

.. math::
    \\sum_{i=0}^{2N} \\beta_i \\, h(k+i) = 0 \\quad \\forall k ,

whose coefficients are shared by every measured IRF.  Stacking that relation for
all responses and time shifts gives a heavily over-determined linear system;
solving it in the least-squares sense and rooting the polynomial yields the
system poles.  Residues / mode shapes then follow from a linear fit
(:func:`femtools.mpe.common.lsfd` in the frequency domain, or a direct
time-domain least squares).

The classic literature name for the multi-reference variant is *PRCE*
(Poly-Reference Complex Exponential); it is obtained here automatically when
several reference (input) columns are supplied.
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

__all__ = ["lsce", "irf_from_frf", "prony"]


def irf_from_frf(
    frf: Any,
    freq_hz: ArrayLike,
    *,
    window: str | None = "exponential",
    window_factor: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Impulse response functions from a single-sided FRF via inverse FFT.

    The FRF must be uniformly sampled.  When its first line is not an integer
    multiple of the line spacing the spectrum is linearly interpolated onto the
    DC-aligned grid, because padding a misaligned band biases *every* identified
    frequency by the offset.  Supplying a DC-aligned axis (``f[0] = k*df``)
    avoids the interpolation and is therefore preferable for lightly damped
    modes that span only a few spectral lines.

    Returns
    -------
    (irf, dt)
        ``irf`` shaped ``(n_out, n_in, n_time)``, ``dt`` the resulting time step.
    """
    H = as_frf_array(frf)
    f = np.asarray(freq_hz, dtype=float).ravel()
    if f.size < 2:
        raise ValueError("need at least two spectral lines")
    df = float(np.mean(np.diff(f)))

    # The inverse FFT requires the spectrum to sit on a DC-aligned grid
    # (f_k = k*df).  A measurement band such as 1.0 ... 250 Hz generally is not,
    # and padding it naively biases every identified frequency by the offset, so
    # resample onto the aligned grid first.
    k0 = f[0] / df
    if abs(k0 - round(k0)) > 1.0e-9:
        k_lo = int(math.ceil(k0 - 1.0e-9))
        k_hi = int(math.floor(f[-1] / df + 1.0e-9))
        grid = np.arange(k_lo, k_hi + 1, dtype=float) * df
        H = np.apply_along_axis(
            lambda col: np.interp(grid, f, col.real) + 1j * np.interp(grid, f, col.imag),
            2,
            H,
        )
        f = grid
        n_dc = k_lo
    else:
        n_dc = int(round(k0))

    n_lines = f.size
    if n_dc > 0:
        pad = np.zeros(H.shape[:2] + (n_dc,), dtype=complex)
        H = np.concatenate([pad, H], axis=2)
        n_lines = H.shape[2]
    n_fft = 2 * (n_lines - 1)
    h = np.fft.irfft(H, n=n_fft, axis=2) * (n_fft * df)
    dt = 1.0 / (n_fft * df)
    if window:
        n_t = h.shape[2]
        if window.lower().startswith("exp"):
            tau = -(n_t - 1) / math.log(max(window_factor, 1e-12))
            w = np.exp(-np.arange(n_t) / tau)
            h = h * w[None, None, :]
        elif window.lower() in ("none", "boxcar", "rect"):
            pass
        else:
            raise ValueError(f"unknown window {window!r}")
    return h, dt


def prony(
    h: np.ndarray, dt: float, order: int, *, rcond: float | None = None
) -> np.ndarray:
    """Continuous-time poles of ``order`` complex-exponential pairs.

    ``h`` is ``(n_signals, n_time)``.  Returns ``2*order`` continuous poles.
    """
    h = np.atleast_2d(np.asarray(h, dtype=float))
    n_sig, n_t = h.shape
    p = 2 * int(order)
    n_rows_per_sig = n_t - p
    if n_rows_per_sig < 1:
        raise ValueError(
            f"time histories too short: need > {p} samples for order {order}"
        )
    # Hankel system:  sum_{i=0}^{p-1} beta_i h[k+i] = -h[k+p]
    A = np.empty((n_sig * n_rows_per_sig, p))
    b = np.empty(n_sig * n_rows_per_sig)
    for s in range(n_sig):
        idx = np.arange(n_rows_per_sig)
        A[s * n_rows_per_sig : (s + 1) * n_rows_per_sig, :] = np.lib.stride_tricks.sliding_window_view(  # noqa: E501
            h[s], p
        )[:n_rows_per_sig]
        b[s * n_rows_per_sig : (s + 1) * n_rows_per_sig] = -h[s][idx + p]
    beta, *_ = np.linalg.lstsq(A, b, rcond=rcond)
    coeffs = np.concatenate([[1.0], beta[::-1]])  # highest power first
    z = np.roots(coeffs)
    return poles_from_roots(z, dt)


def lsce(
    data: Any,
    dt: float | None = None,
    *,
    freq_hz: ArrayLike | None = None,
    fs: float | None = None,
    order: int = 20,
    orders: Sequence[int] | None = None,
    order_min: int | None = None,
    order_step: int = 2,
    n_modes: int | None = None,
    f_range: tuple[float, float] | None = None,
    max_damping: float = 0.25,
    min_damping: float = 0.0,
    stabilization: bool = True,
    stabilization_level: str = "d",
    tol_freq: float = 0.01,
    tol_damp: float = 0.05,
    min_count: int = 2,
    cluster_tol: float = 0.01,
    mode_shapes: bool = True,
    window: str | None = "exponential",
    window_factor: float = 0.01,
    n_samples: int | None = None,
) -> ModalParameterResult:
    """Least-Squares Complex Exponential modal parameter estimation.

    Parameters
    ----------
    data:
        Either impulse response functions (real, ``(n_out, n_in, n_t)`` or
        ``(n_signals, n_t)``) together with ``dt``/``fs``, or a **complex** FRF
        matrix together with ``freq_hz`` — in that case the IRFs are obtained by
        inverse FFT first.
    order:
        Number of assumed mode pairs (the AR polynomial has degree ``2*order``).
        With ``stabilization=True`` a sweep of orders is run and the physical
        poles are those that stabilise.
    f_range, max_damping:
        Physical-pole acceptance window.
    mode_shapes:
        Fit residues / mode shapes with LSFD.  Requires FRF input (or a
        ``freq_hz`` axis) so the fit can be done in the frequency domain.

    Returns
    -------
    ModalParameterResult

    Notes
    -----
    An exponential window (default) is applied to the IRFs to suppress leakage
    at the end of the record; its effect on the identified damping is removed
    analytically afterwards.
    """
    arr = np.asarray(data)
    H_freq = None
    if arr.dtype.kind == "c":
        if freq_hz is None:
            freq_hz = getattr(data, "freq_hz", None)
        if freq_hz is None:
            raise ValueError("complex input is treated as an FRF and needs freq_hz")
        H_freq = as_frf_array(arr)
        f_axis = np.asarray(freq_hz, dtype=float).ravel()
        if f_range is not None:
            sel = (f_axis >= f_range[0]) & (f_axis <= f_range[1])
            if sel.sum() >= 8:
                H_freq, f_axis = H_freq[:, :, sel], f_axis[sel]
        h3, dt_calc = irf_from_frf(
            H_freq, f_axis, window=window, window_factor=window_factor
        )
        dt = dt_calc if dt is None else float(dt)
        h = h3.reshape(-1, h3.shape[2])
        beta_window = (
            -math.log(max(window_factor, 1e-12)) / ((h3.shape[2] - 1) * dt)
            if window and window.lower().startswith("exp")
            else 0.0
        )
    else:
        if dt is None:
            if fs is None:
                raise ValueError("dt or fs must be given for time-domain input")
            dt = 1.0 / float(fs)
        h3 = arr.astype(float)
        if h3.ndim == 3:
            h = h3.reshape(-1, h3.shape[2])
        elif h3.ndim == 2:
            h = h3
        elif h3.ndim == 1:
            h = h3[None, :]
        else:
            raise ValueError(f"cannot interpret data of shape {arr.shape}")
        f_axis = None if freq_hz is None else np.asarray(freq_hz, dtype=float)  # type: ignore[assignment]
        beta_window = 0.0

    if n_samples is not None:
        h = h[:, : int(n_samples)]

    if orders is not None:
        order_list = sorted(int(o) for o in orders)
    elif stabilization:
        omin = int(order_min) if order_min is not None else max(2, min(4, order))
        order_list = list(range(omin, int(order) + 1, max(1, int(order_step))))
        if order_list[-1] != int(order):
            order_list.append(int(order))
    else:
        order_list = [int(order)]

    band = f_range
    if band is None and f_axis is not None:
        band = (float(f_axis.min()), float(f_axis.max()))
    elif band is None:
        band = (0.0, 0.5 / float(dt))

    pole_sets: dict[int, np.ndarray] = {}
    for o in order_list:
        try:
            s = prony(h, float(dt), o)
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover
            continue
        # Undo the exponential window's artificial damping.
        s = s + beta_window
        pole_sets[o] = select_physical_poles(
            s, f_range=band, max_damping=max_damping, min_damping=min_damping
        )

    if not pole_sets:
        raise RuntimeError("LSCE failed to produce any physical poles")

    diagram = None
    if len(pole_sets) > 1:
        diagram = stabilization_diagram(pole_sets, tol_freq=tol_freq, tol_damp=tol_damp)
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
        keep = np.sort(np.argsort(-counts)[: int(n_modes)])
        reps, counts = reps[keep], counts[keep]

    idx = np.argsort(np.abs(reps))
    lam = reps[idx]
    wn = np.abs(lam)
    result = ModalParameterResult(
        freq_hz=wn / (2.0 * math.pi),
        damping=np.where(wn > 0, -lam.real / wn, 0.0),
        poles=lam,
        order=order_list[-1],
        method="LSCE",
        stabilization=diagram,
        extras={"dt": float(dt), "orders": order_list, "cluster_counts": counts[idx]},
    )

    if mode_shapes and lam.size and H_freq is not None and f_axis is not None:
        fit = lsfd(H_freq, f_axis, lam)
        result.residues = fit["residues"]
        result.mode_shapes = fit["mode_shapes"]
        result.participation = fit["participation"]
        result.lower_residual = fit["lower_residual"]
        result.upper_residual = fit["upper_residual"]
        result.fit_error = fit["fit_error"]
    elif mode_shapes and lam.size:
        result.mode_shapes = _time_domain_shapes(h3, float(dt), lam)
    return result


def _time_domain_shapes(h3: np.ndarray, dt: float, lam: np.ndarray) -> np.ndarray | None:
    """Least-squares residues directly from the IRFs (time domain)."""
    h3 = np.asarray(h3, dtype=float)
    if h3.ndim == 2:
        h3 = h3[:, None, :]
    if h3.ndim != 3:
        return None
    n_out, n_in, n_t = h3.shape
    t = np.arange(n_t) * dt
    cols = []
    for r in range(lam.size):
        e = np.exp(lam[r] * t)
        cols.append(2.0 * np.real(e))
        cols.append(-2.0 * np.imag(e))
    A = np.column_stack(cols)
    shapes = np.zeros((n_out, lam.size), dtype=complex)
    for o in range(n_out):
        x, *_ = np.linalg.lstsq(A, h3[o, 0, :], rcond=None)
        shapes[o] = x[0::2] + 1j * x[1::2]
    for j in range(shapes.shape[1]):
        k = int(np.argmax(np.abs(shapes[:, j])))
        if abs(shapes[k, j]) > 0:
            shapes[:, j] /= shapes[k, j]
    return shapes
