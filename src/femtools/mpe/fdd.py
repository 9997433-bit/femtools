"""Frequency Domain Decomposition (FDD) and Enhanced FDD (EFDD).

Output-only (operational) modal analysis.  The estimator takes the spectral
density matrix of the measured responses and decomposes it line by line,

.. math::
    G_{yy}(\\omega_k) = U_k \\, S_k \\, U_k^H .

Near an isolated resonance the response is dominated by a single mode, so the
first singular value peaks and the corresponding left singular vector is an
unscaled estimate of the mode shape (Brincker, Zhang & Andersen, 2000).

``efdd`` additionally isolates the "SDOF bell" around each peak using a MAC
criterion, transforms it back to an auto-correlation function and estimates the
damping ratio from the logarithmic decrement, together with a refined natural
frequency from the zero crossings.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .common import ModalParameterResult, mac, modal_poles_from_fz

__all__ = ["fdd", "efdd", "cross_spectral_density", "singular_value_decomposition"]


def cross_spectral_density(
    data: np.ndarray,
    fs: float,
    *,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
    scaling: str = "density",
) -> tuple[np.ndarray, np.ndarray]:
    """Full cross-spectral-density matrix of multi-channel time data.

    Parameters
    ----------
    data:
        ``(n_channels, n_samples)`` (a ``(n_samples, n_channels)`` array is
        transposed automatically when it is clearly "tall").
    fs:
        Sampling frequency [Hz].

    Returns
    -------
    (freq_hz, G)
        ``G`` has shape ``(n_channels, n_channels, n_freq)`` and is Hermitian on
        every line.
    """
    from scipy.signal import csd

    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] > x.shape[1]:
        x = x.T
    n_ch = x.shape[0]
    nperseg = int(min(nperseg, x.shape[1]))
    if noverlap is None:
        noverlap = nperseg // 2

    f, g00 = csd(
        x[0], x[0], fs=fs, nperseg=nperseg, noverlap=noverlap, window=window,
        detrend=detrend, scaling=scaling,
    )
    G = np.zeros((n_ch, n_ch, f.size), dtype=complex)
    G[0, 0] = g00
    for i in range(n_ch):
        for j in range(i, n_ch):
            if i == 0 and j == 0:
                continue
            _, gij = csd(
                x[i], x[j], fs=fs, nperseg=nperseg, noverlap=noverlap, window=window,
                detrend=detrend, scaling=scaling,
            )
            G[i, j] = gij
            if i != j:
                G[j, i] = np.conj(gij)
    return np.asarray(f, dtype=float), G


def singular_value_decomposition(
    G: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Line-by-line SVD of a spectral matrix.

    Returns ``(sv, U)`` with ``sv`` shaped ``(n_ch, n_freq)`` (descending) and
    ``U`` shaped ``(n_ch, n_ch, n_freq)``.
    """
    G = np.asarray(G)
    n_ch, _, n_freq = G.shape
    sv = np.zeros((n_ch, n_freq))
    U = np.zeros((n_ch, n_ch, n_freq), dtype=complex)
    for k in range(n_freq):
        u, s, _ = np.linalg.svd(G[:, :, k])
        sv[:, k] = s
        U[:, :, k] = u
    return sv, U


def _prepare(
    data: Any,
    freq_hz: ArrayLike | None,
    fs: float | None,
    nperseg: int,
    noverlap: int | None,
    window: str,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return ``(freq_hz, G, fs)`` from either a spectral matrix or time data."""
    if hasattr(data, "psd") or hasattr(data, "G"):
        G = np.asarray(getattr(data, "psd", None) if hasattr(data, "psd") else data.G)
        f = np.asarray(getattr(data, "freq_hz", freq_hz), dtype=float)
        return f, G, fs
    arr = np.asarray(data)
    if arr.ndim == 3:
        if freq_hz is None:
            raise ValueError("freq_hz is required when passing a spectral matrix")
        return np.asarray(freq_hz, dtype=float), arr.astype(complex), fs
    if arr.ndim in (1, 2):
        if freq_hz is not None and arr.dtype.kind == "c":
            # (n_ch, n_freq) spectra -> outer-product spectral matrix
            X = arr if arr.ndim == 2 else arr[None, :]
            G = np.einsum("ik,jk->ijk", X, X.conj())
            return np.asarray(freq_hz, dtype=float), G, fs
        if fs is None:
            raise ValueError("fs is required when passing time data")
        f, G = cross_spectral_density(
            arr, fs, nperseg=nperseg, noverlap=noverlap, window=window
        )
        return f, G, fs
    raise ValueError(f"cannot interpret data of shape {arr.shape}")


def _pick_peaks(
    sv1: np.ndarray,
    f: np.ndarray,
    *,
    n_modes: int | None,
    f_range: tuple[float, float] | None,
    min_distance_hz: float,
    prominence: float,
) -> np.ndarray:
    """Peak-pick the first singular value curve."""
    from scipy.signal import find_peaks

    mask = np.ones(f.size, dtype=bool)
    if f_range is not None:
        mask = (f >= f_range[0]) & (f <= f_range[1])
    y = np.where(mask, sv1, 0.0)
    ylog = 10.0 * np.log10(np.maximum(y, np.max(y) * 1e-12))
    df = float(np.mean(np.diff(f))) if f.size > 1 else 1.0
    dist = max(1, int(round(min_distance_hz / df))) if min_distance_hz > 0 else 1
    span = float(np.max(ylog) - np.min(ylog))
    peaks, props = find_peaks(ylog, distance=dist, prominence=prominence * span)
    if peaks.size == 0:
        peaks, props = find_peaks(ylog, distance=dist)
    if peaks.size == 0:
        return np.zeros(0, dtype=int)
    if n_modes is not None and peaks.size > n_modes:
        best = np.argsort(-props.get("prominences", sv1[peaks]))[:n_modes]
        peaks = np.sort(peaks[best])
    return peaks


def _interpolate_peak(f: np.ndarray, y: np.ndarray, k: int) -> float:
    """Sub-bin peak location by a parabolic fit through the log spectrum.

    Fitting the logarithm (rather than the linear magnitude) is markedly less
    biased for the sharply peaked spectra of lightly damped modes.
    """
    if k <= 0 or k >= f.size - 1:
        return float(f[k])
    floor = max(float(np.max(y)) * 1e-30, 1e-300)
    y0, y1, y2 = (
        math.log(max(float(y[k - 1]), floor)),
        math.log(max(float(y[k]), floor)),
        math.log(max(float(y[k + 1]), floor)),
    )
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(f[k])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(f[k] + delta * (f[k + 1] - f[k]))


def fdd(
    data: Any,
    freq_hz: ArrayLike | None = None,
    *,
    fs: float | None = None,
    n_modes: int | None = None,
    f_range: tuple[float, float] | None = None,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    prominence: float = 0.05,
    min_distance_hz: float = 0.0,
    peaks: ArrayLike | None = None,
    interpolate: bool = True,
    damping: float = 0.0,
) -> ModalParameterResult:
    """Frequency Domain Decomposition — peak picking on the first singular value.

    Parameters
    ----------
    data:
        ``(n_ch, n_samples)`` time histories (with ``fs``), a
        ``(n_ch, n_ch, n_freq)`` spectral density matrix (with ``freq_hz``), or
        ``(n_ch, n_freq)`` complex spectra.
    n_modes:
        Keep only the ``n_modes`` most prominent peaks.
    f_range:
        Restrict peak picking to this band.
    peaks:
        Explicit peak frequencies [Hz] to use instead of automatic picking.
    damping:
        Damping ratio assigned to every mode (plain FDD does not estimate it);
        use :func:`efdd` for identified damping.

    Returns
    -------
    ModalParameterResult
        ``extras`` carries ``singular_values``, ``spectral_freq`` and
        ``singular_vectors`` for plotting the FDD diagram.

    Notes
    -----
    FDD assumes broadband (approximately white) excitation, light damping and
    well-separated modes; closely spaced modes appear in the *second* singular
    value, which is returned in ``extras["singular_values"]``.
    """
    f, G, fs = _prepare(data, freq_hz, fs, nperseg, noverlap, window)
    sv, U = singular_value_decomposition(G)
    sv1 = sv[0]

    if peaks is not None:
        wanted = np.atleast_1d(np.asarray(peaks, dtype=float)).ravel()
        idx = np.asarray(
            [int(np.argmin(np.abs(f - pk))) for pk in wanted], dtype=int
        )
    else:
        idx = _pick_peaks(
            sv1,
            f,
            n_modes=n_modes,
            f_range=f_range,
            min_distance_hz=min_distance_hz,
            prominence=prominence,
        )
    if idx.size == 0:
        raise RuntimeError("FDD found no spectral peaks; check f_range / prominence")

    fn = np.array(
        [_interpolate_peak(f, sv1, int(k)) if interpolate else f[k] for k in idx], dtype=float
    )
    shapes = np.column_stack([U[:, 0, int(k)] for k in idx])
    for j in range(shapes.shape[1]):
        m = int(np.argmax(np.abs(shapes[:, j])))
        if abs(shapes[m, j]) > 0:
            shapes[:, j] = shapes[:, j] / shapes[m, j]

    zeta = np.full(fn.size, float(damping))
    return ModalParameterResult(
        freq_hz=fn,
        damping=zeta,
        poles=modal_poles_from_fz(fn, zeta),
        mode_shapes=shapes,
        method="FDD",
        extras={
            "singular_values": sv,
            "singular_vectors": U,
            "spectral_freq": f,
            "peak_indices": idx,
            "fs": fs,
        },
    )


def efdd(
    data: Any,
    freq_hz: ArrayLike | None = None,
    *,
    fs: float | None = None,
    n_modes: int | None = None,
    f_range: tuple[float, float] | None = None,
    nperseg: int = 1024,
    noverlap: int | None = None,
    window: str = "hann",
    prominence: float = 0.05,
    min_distance_hz: float = 0.0,
    peaks: ArrayLike | None = None,
    mac_threshold: float = 0.85,
    max_bell_width: float = 0.5,
    corr_range: tuple[float, float] = (0.15, 0.90),
    min_points: int = 5,
) -> ModalParameterResult:
    """Enhanced FDD — SDOF bell extraction, log-decrement damping.

    For every FDD peak the neighbouring spectral lines whose first singular
    vector still correlates with the peak vector (``MAC >= mac_threshold``) form
    the SDOF auto-spectrum of that mode.  Its inverse FFT is the free-decay
    auto-correlation; a straight-line fit of ``ln|r_k|`` over the extrema gives
    the logarithmic decrement (hence the damping ratio) and the zero crossings
    give a refined damped natural frequency.

    Parameters
    ----------
    mac_threshold:
        Correlation limit that delimits the SDOF bell (0.85 is a common choice).
    corr_range:
        Fraction of the normalised correlation function used for the fit,
        skipping the noisy start and tail.

    Returns
    -------
    ModalParameterResult
        With identified ``damping``; ``extras["bells"]`` holds the per-mode
        spectral bands and correlation fits.
    """
    base = fdd(
        data,
        freq_hz,
        fs=fs,
        n_modes=n_modes,
        f_range=f_range,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        prominence=prominence,
        min_distance_hz=min_distance_hz,
        peaks=peaks,
    )
    sv = base.extras["singular_values"]
    U = base.extras["singular_vectors"]
    f = base.extras["spectral_freq"]
    idx = base.extras["peak_indices"]
    df = float(np.mean(np.diff(f))) if f.size > 1 else 1.0

    fn_out = np.zeros(idx.size)
    zeta_out = np.zeros(idx.size)
    bells: list[dict[str, Any]] = []

    for j, k in enumerate(idx):
        k = int(k)
        ref = U[:, 0, k]
        lo = k
        while lo - 1 >= 0 and mac(U[:, 0, lo - 1], ref) >= mac_threshold:
            lo -= 1
            if (f[k] - f[lo]) > max_bell_width * f[k]:
                break
        hi = k
        while hi + 1 < f.size and mac(U[:, 0, hi + 1], ref) >= mac_threshold:
            hi += 1
            if (f[hi] - f[k]) > max_bell_width * f[k]:
                break

        bell = np.zeros(f.size)
        bell[lo : hi + 1] = sv[0, lo : hi + 1]
        n_bell = hi - lo + 1

        fd, zeta, fit = _bell_to_damping(bell, df, corr_range, min_points)
        if not math.isfinite(zeta) or zeta <= 0.0 or zeta > 0.3 or n_bell < min_points:
            # Fall back on the half-power bandwidth of the bell.
            zeta = _half_power_damping(f, sv[0], k)
            fd = base.freq_hz[j]
            fit = {"method": "half-power"}
        fn = fd / math.sqrt(max(1.0 - zeta**2, 1e-12)) if fd > 0 else base.freq_hz[j]
        fn_out[j] = fn if math.isfinite(fn) and fn > 0 else base.freq_hz[j]
        zeta_out[j] = max(zeta, 0.0)
        bells.append(
            {
                "mode": j,
                "band": (float(f[lo]), float(f[hi])),
                "n_lines": int(n_bell),
                "damped_freq_hz": float(fd),
                **fit,
            }
        )

    return ModalParameterResult(
        freq_hz=fn_out,
        damping=zeta_out,
        poles=modal_poles_from_fz(fn_out, zeta_out),
        mode_shapes=base.mode_shapes,
        method="EFDD",
        extras={**base.extras, "bells": bells},
    )


def _bell_to_damping(
    bell: np.ndarray,
    df: float,
    corr_range: tuple[float, float],
    min_points: int,
) -> tuple[float, float, dict[str, Any]]:
    """Log-decrement damping from an SDOF spectral bell."""
    n_fft = 2 * (bell.size - 1)
    if n_fft < 8:
        return math.nan, math.nan, {}
    corr = np.fft.irfft(bell, n=n_fft)
    corr = corr[: n_fft // 2]
    if corr.size < 8 or corr[0] <= 0:
        return math.nan, math.nan, {}
    corr = corr / corr[0]
    dt = 1.0 / (n_fft * df)

    # Extrema of the free decay
    peaks_idx: list[int] = []
    for i in range(1, corr.size - 1):
        if (corr[i] - corr[i - 1]) * (corr[i + 1] - corr[i]) < 0:
            peaks_idx.append(i)
    if len(peaks_idx) < min_points:
        return math.nan, math.nan, {}
    pk = np.asarray(peaks_idx)
    amp = np.abs(corr[pk])
    lo_a, hi_a = corr_range
    sel = (amp <= hi_a * amp.max()) & (amp >= lo_a * amp.max())
    if np.count_nonzero(sel) < min_points:
        sel = np.ones(pk.size, dtype=bool)
    pk_s, amp_s = pk[sel], amp[sel]
    amp_s = np.maximum(amp_s, 1e-300)

    # log-decrement: ln|r_k| = ln|r_0| - zeta*wn*t_k
    k_axis = np.arange(pk_s.size, dtype=float)
    slope, intercept = np.polyfit(k_axis, np.log(amp_s), 1)
    delta = -2.0 * slope  # per full cycle (two extrema per cycle)
    zeta = delta / math.sqrt(4.0 * math.pi**2 + delta**2)

    # Refined damped frequency from the extrema spacing (half period each).
    if pk_s.size >= 2:
        t = pk_s * dt
        slope_t, _ = np.polyfit(np.arange(pk_s.size, dtype=float), t, 1)
        fd = 1.0 / (2.0 * slope_t) if slope_t > 0 else math.nan
    else:  # pragma: no cover
        fd = math.nan
    r2 = float(
        1.0
        - np.sum((np.log(amp_s) - (slope * k_axis + intercept)) ** 2)
        / max(np.sum((np.log(amp_s) - np.mean(np.log(amp_s))) ** 2), 1e-300)
    )
    return fd, zeta, {"log_decrement": float(delta), "r2": r2, "n_extrema": int(pk_s.size)}


def _half_power_damping(f: np.ndarray, sv1: np.ndarray, k: int) -> float:
    """Half-power bandwidth damping estimate around index ``k``."""
    target = sv1[k] / 2.0  # singular values of a PSD scale with amplitude^2
    lo = k
    while lo > 0 and sv1[lo] > target:
        lo -= 1
    hi = k
    while hi < f.size - 1 and sv1[hi] > target:
        hi += 1
    if hi <= lo or f[k] <= 0:
        return math.nan
    return float((f[hi] - f[lo]) / (2.0 * f[k]))
