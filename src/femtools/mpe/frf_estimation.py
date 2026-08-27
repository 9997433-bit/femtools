"""Non-parametric FRF estimation from measured time data (Welch averaging).

Given the excitation ``x`` and the response ``y`` of a linear system, the
frequency response is estimated from *averaged* auto- and cross-spectra.  The
three classical estimators differ only in which noise source they assume
dominates (Bendat & Piersol, *Random Data*, ch. 6--7):

``H1`` — noise on the **output**

.. math::
    H_1(\\omega) = G_{yx}(\\omega)\\, G_{xx}(\\omega)^{-1}

``H2`` — noise on the **input**

.. math::
    H_2(\\omega) = G_{yy}(\\omega)\\, G_{xy}(\\omega)^{-1}

``\\gamma^2`` — the ordinary coherence, the fraction of the output power at
each line that is linearly explained by the input,

.. math::
    \\gamma^2_{oi}(\\omega) =
        \\frac{|G_{yx,oi}(\\omega)|^2}{G_{xx,ii}(\\omega)\\, G_{yy,oo}(\\omega)}
        \\; = \\; \\frac{H_1}{H_2} \\quad\\text{(SISO)} .

``H1`` under-estimates the FRF at resonance (where the response noise is
relatively large) and ``H2`` over-estimates it at anti-resonance, so the two
bracket the truth; their ratio is exactly the SISO coherence, which makes the
pair a built-in quality check.  All spectra are Welch estimates: the record is
split into overlapping windowed segments and the segment cross-products are
averaged, which trades frequency resolution for variance
(:math:`\\mathrm{var} \\propto 1/n_{avg}`).

MIMO is supported: with ``n_in`` simultaneous, *linearly independent* inputs
:math:`G_{xx}` is an ``(n_in, n_in)`` matrix on every line and the estimator
solves the matrix equation instead of dividing.  Then the meaningful quality
measure is the **multiple** coherence, which accounts for all inputs at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "FRFEstimate",
    "SpectralMatrices",
    "welch_spectra",
    "estimate_h1",
    "estimate_h2",
    "estimate_frf",
    "coherence",
    "multiple_coherence",
]


# ----------------------------------------------------------------------
def _as_channels(data: ArrayLike, name: str) -> np.ndarray:
    """Coerce time data into ``(n_channels, n_samples)``."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 1-D or 2-D, got shape {arr.shape}")
    # Records are far longer than they are wide; a "tall" array is (n_samples, n_ch).
    if arr.shape[0] > arr.shape[1]:
        return np.ascontiguousarray(arr.T)
    return arr


def _detrend(seg: np.ndarray, kind: Any) -> np.ndarray:
    """Remove the mean (``"constant"``) or a straight line (``"linear"``)."""
    if kind in (None, False, "none"):
        return seg
    key = str(kind).lower()
    if key in ("constant", "mean", "true"):
        return seg - np.mean(seg, axis=-1, keepdims=True)
    if key in ("linear", "detrend"):
        n = seg.shape[-1]
        t = np.arange(n, dtype=float)
        t = t - t.mean()
        denom = float(t @ t)
        slope = (seg @ t) / denom if denom > 0 else np.zeros(seg.shape[:-1])
        return seg - np.mean(seg, axis=-1, keepdims=True) - slope[..., None] * t
    raise ValueError(f"unknown detrend {kind!r}")


@dataclass
class SpectralMatrices:
    """Welch-averaged auto- and cross-spectral matrices.

    Attributes
    ----------
    freq_hz:
        One-sided frequency axis [Hz].
    Gxx:
        ``(n_in, n_in, n_freq)`` input spectral matrix (Hermitian per line).
    Gyy:
        ``(n_out, n_out, n_freq)`` output spectral matrix.
    Gyx:
        ``(n_out, n_in, n_freq)`` cross-spectral matrix ``E[Y X^H]``.
    n_averages:
        Number of averaged segments.
    """

    freq_hz: np.ndarray
    Gxx: np.ndarray
    Gyy: np.ndarray
    Gyx: np.ndarray
    n_averages: int = 1
    nperseg: int = 0
    noverlap: int = 0
    window: str = "hann"
    scaling: str = "density"
    fs: float = math.nan

    @property
    def Gxy(self) -> np.ndarray:
        """``(n_in, n_out, n_freq)`` cross spectrum ``E[X Y^H] = G_yx^H``."""
        return np.conj(np.swapaxes(self.Gyx, 0, 1))

    @property
    def n_in(self) -> int:
        return int(self.Gxx.shape[0])

    @property
    def n_out(self) -> int:
        return int(self.Gyy.shape[0])

    @property
    def n_freq(self) -> int:
        return int(self.freq_hz.size)


def welch_spectra(
    x: ArrayLike,
    y: ArrayLike | None = None,
    fs: float = 1.0,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: Any = "constant",
    scaling: str = "density",
    n_averages: int | None = None,
) -> SpectralMatrices:
    """Welch-averaged spectral matrices of ``x`` (inputs) and ``y`` (outputs).

    Every cross product is formed from the **same** set of segment FFTs, so the
    estimators built on top see a consistent set of averages.

    Parameters
    ----------
    x, y:
        ``(n_in, n_samples)`` excitation and ``(n_out, n_samples)`` response.
        A 1-D array is one channel; a "tall" 2-D array is transposed.
        ``y=None`` computes the input spectra only (``Gyy = Gxx``).
    fs:
        Sampling frequency [Hz].
    nperseg, noverlap:
        Segment length and overlap in samples (default: 8 segments with 50 %
        overlap, i.e. ``nperseg = n_samples // 5``, at least 8 samples).
    window:
        Any :func:`scipy.signal.get_window` name (``"hann"`` by default,
        ``"boxcar"``/``"rect"`` for impact testing, ``"flattop"`` for amplitude
        accuracy).
    scaling:
        ``"density"`` (per Hz, default) or ``"spectrum"``.  The FRF estimators
        are insensitive to this choice — it cancels in the ratio.
    n_averages:
        Convenience alternative to ``nperseg``: split the record into this many
        50 %-overlapping segments.

    Returns
    -------
    SpectralMatrices
    """
    from scipy.signal import get_window

    X = _as_channels(x, "x")
    Y = X if y is None else _as_channels(y, "y")
    if Y.shape[1] != X.shape[1]:
        raise ValueError(
            f"x has {X.shape[1]} samples but y has {Y.shape[1]}; they must be simultaneous"
        )
    n_samples = X.shape[1]
    if n_samples < 4:
        raise ValueError("need at least 4 samples")

    if nperseg is None:
        if n_averages is not None and int(n_averages) > 0:
            k = int(n_averages)
            # k segments with 50 % overlap span (k + 1) * nperseg / 2 samples
            nperseg = max(8, int(2 * n_samples // (k + 1)))
        else:
            nperseg = max(8, n_samples // 5)
    nperseg = int(min(int(nperseg), n_samples))
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(min(max(int(noverlap), 0), nperseg - 1))
    step = nperseg - noverlap
    n_seg = 1 + (n_samples - nperseg) // step
    if n_seg < 1:  # pragma: no cover - guarded by the clamps above
        raise ValueError("segment length exceeds the record")

    w = np.asarray(get_window(window, nperseg), dtype=float)
    if str(scaling).lower() in ("density", "psd"):
        scale = 2.0 / (float(fs) * float(w @ w))
    elif str(scaling).lower() in ("spectrum", "power"):
        scale = 2.0 / float(np.sum(w) ** 2)
    else:
        raise ValueError(f"unknown scaling {scaling!r}")

    starts = np.arange(n_seg) * step
    n_freq = nperseg // 2 + 1

    def _segment_fft(data: np.ndarray) -> np.ndarray:
        seg = np.stack([data[:, s : s + nperseg] for s in starts], axis=0)
        seg = _detrend(seg, detrend)
        return np.fft.rfft(seg * w[None, None, :], axis=-1)  # (n_seg, n_ch, n_freq)

    Xf = _segment_fft(X)
    Yf = Xf if y is None else _segment_fft(Y)

    Gxx = scale * np.einsum("sif,sjf->ijf", Xf, Xf.conj(), optimize=True) / n_seg
    Gyy = (
        Gxx
        if y is None
        else scale * np.einsum("sof,spf->opf", Yf, Yf.conj(), optimize=True) / n_seg
    )
    Gyx = (
        Gxx
        if y is None
        else scale * np.einsum("sof,sif->oif", Yf, Xf.conj(), optimize=True) / n_seg
    )

    # One-sided spectra: DC and (for even nperseg) Nyquist are not folded.
    fold = np.ones(n_freq)
    fold[0] = 0.5
    if nperseg % 2 == 0:
        fold[-1] = 0.5
    Gxx = Gxx * fold
    Gyy = Gyy * fold
    Gyx = Gyx * fold

    return SpectralMatrices(
        freq_hz=np.fft.rfftfreq(nperseg, d=1.0 / float(fs)),
        Gxx=Gxx,
        Gyy=Gyy,
        Gyx=Gyx,
        n_averages=int(n_seg),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        window=str(window),
        scaling=str(scaling).lower(),
        fs=float(fs),
    )


# ----------------------------------------------------------------------
@dataclass
class FRFEstimate:
    """Estimated FRF matrix plus the spectra and quality metrics behind it.

    Attributes
    ----------
    H:
        ``(n_out, n_in, n_freq)`` complex FRF, in the same units as ``y/x``.
    freq_hz:
        One-sided frequency axis [Hz].
    coherence:
        ``(n_out, n_in, n_freq)`` ordinary coherence.
    multiple_coherence:
        ``(n_out, n_freq)`` multiple coherence (all inputs at once); equals the
        ordinary coherence for a single input.
    method:
        ``"H1"`` or ``"H2"``.
    n_averages:
        Number of averaged Welch segments — coherence estimated from a single
        segment is identically 1 and therefore meaningless.
    """

    H: np.ndarray
    freq_hz: np.ndarray
    coherence: np.ndarray | None = None
    multiple_coherence: np.ndarray | None = None
    method: str = "H1"
    n_averages: int = 1
    spectra: SpectralMatrices | None = None
    condition_number: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.array(self.H, dtype=dtype, copy=copy)

    @property
    def frf(self) -> np.ndarray:
        return self.H

    @property
    def shape(self) -> tuple[int, ...]:
        return self.H.shape

    @property
    def n_out(self) -> int:
        return int(self.H.shape[0])

    @property
    def n_in(self) -> int:
        return int(self.H.shape[1])

    def band(self, f_range: tuple[float, float]) -> FRFEstimate:
        """Return a copy restricted to ``f_range`` (inclusive)."""
        sel = (self.freq_hz >= f_range[0]) & (self.freq_hz <= f_range[1])
        return FRFEstimate(
            H=self.H[:, :, sel],
            freq_hz=self.freq_hz[sel],
            coherence=None if self.coherence is None else self.coherence[:, :, sel],
            multiple_coherence=(
                None if self.multiple_coherence is None else self.multiple_coherence[:, sel]
            ),
            method=self.method,
            n_averages=self.n_averages,
            spectra=self.spectra,
            condition_number=(
                None if self.condition_number is None else self.condition_number[sel]
            ),
            extras=dict(self.extras),
        )

    def mean_coherence(self, f_range: tuple[float, float] | None = None) -> float:
        """Mean ordinary coherence, optionally over a band (a quick data check)."""
        if self.coherence is None:  # pragma: no cover - always set by the estimators
            return math.nan
        g = self.coherence
        if f_range is not None:
            sel = (self.freq_hz >= f_range[0]) & (self.freq_hz <= f_range[1])
            g = g[:, :, sel]
        return float(np.mean(g)) if g.size else math.nan

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FRFEstimate(method={self.method!r}, shape={self.H.shape}, "
            f"n_averages={self.n_averages})"
        )


def _solve_right(A: np.ndarray, B: np.ndarray, rcond: float, ridge: float) -> np.ndarray:
    """Solve ``X B = A`` per frequency line, i.e. ``X = A B^{-1}``.

    ``A`` is ``(m, k, n_freq)`` and ``B`` is ``(k, k, n_freq)``.  A Tikhonov
    ``ridge`` (relative to the mean diagonal of ``B``) and a pseudo-inverse
    fallback keep rank-deficient input spectra — the usual symptom of
    correlated shakers — from producing infinities.
    """
    m, k, n_freq = A.shape
    out = np.zeros((m, k, n_freq), dtype=complex)
    for f in range(n_freq):
        Bf = B[:, :, f]
        if ridge:
            trace = float(np.real(np.trace(Bf))) / max(k, 1)
            Bf = Bf + ridge * max(trace, 1e-300) * np.eye(k)
        try:
            out[:, :, f] = np.linalg.solve(Bf.T, A[:, :, f].T).T
        except np.linalg.LinAlgError:
            out[:, :, f] = A[:, :, f] @ np.linalg.pinv(Bf, rcond=rcond)
    return out


def _ordinary_coherence(sp: SpectralMatrices) -> np.ndarray:
    """``(n_out, n_in, n_freq)`` ordinary coherence from a spectral matrix set."""
    gxx = np.real(np.einsum("iif->if", sp.Gxx))
    gyy = np.real(np.einsum("oof->of", sp.Gyy))
    den = gyy[:, None, :] * gxx[None, :, :]
    num = np.abs(sp.Gyx) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return np.clip(np.asarray(g2, dtype=float), 0.0, 1.0)


def _multiple_coherence(sp: SpectralMatrices, rcond: float, ridge: float) -> np.ndarray:
    """``(n_out, n_freq)`` multiple coherence (all inputs considered together)."""
    H1 = _solve_right(sp.Gyx, sp.Gxx, rcond, ridge)
    gyy = np.real(np.einsum("oof->of", sp.Gyy))
    num = np.real(np.einsum("oif,oif->of", H1, np.conj(sp.Gyx)))
    with np.errstate(divide="ignore", invalid="ignore"):
        g2 = np.where(gyy > 0, num / np.where(gyy > 0, gyy, 1.0), 0.0)
    return np.clip(np.asarray(g2, dtype=float), 0.0, 1.0)


def _prepare(
    x: Any,
    y: Any,
    fs: float | None,
    kwargs: dict[str, Any],
) -> SpectralMatrices:
    """Accept either raw time data or an already computed spectral matrix set."""
    if isinstance(x, SpectralMatrices):
        return x
    if fs is None:
        raise ValueError("fs is required when passing time data")
    return welch_spectra(x, y, fs, **kwargs)


def estimate_h1(
    x: Any,
    y: Any = None,
    fs: float | None = None,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: Any = "constant",
    scaling: str = "density",
    n_averages: int | None = None,
    rcond: float = 1.0e-12,
    ridge: float = 0.0,
    f_range: tuple[float, float] | None = None,
) -> FRFEstimate:
    """``H1`` FRF estimate — least squares assuming noise on the **output**.

    .. math:: H_1 = G_{yx} G_{xx}^{-1}

    Parameters
    ----------
    x, y:
        ``(n_in, n_samples)`` excitation and ``(n_out, n_samples)`` response
        time histories, or a pre-computed :class:`SpectralMatrices` as ``x``.
    fs:
        Sampling frequency [Hz].
    nperseg, noverlap, window, detrend, n_averages:
        Welch settings, see :func:`welch_spectra`.
    ridge:
        Relative Tikhonov regularisation added to the diagonal of ``G_xx``
        before the solve.  Only useful for MIMO data with partially correlated
        shakers; 0 (default) does a plain solve.
    f_range:
        Optionally restrict the returned lines to this band.

    Returns
    -------
    FRFEstimate
        ``H`` shaped ``(n_out, n_in, n_freq)`` plus ordinary and multiple
        coherence.

    Notes
    -----
    ``H1`` is biased **low** at resonance, because uncorrelated output noise
    inflates ``G_yy`` but not ``G_yx``.  With ``n_avg`` averages the normalised
    random error of ``|H1|`` is
    :math:`\\sqrt{(1-\\gamma^2)/(2\\gamma^2 n_{avg})}` — which is why coherence
    is reported alongside the FRF rather than as an afterthought.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.mpe.frf_estimation import estimate_h1
    >>> rng = np.random.default_rng(0)
    >>> u = rng.standard_normal(8192)          # white excitation
    >>> v = np.convolve(u, [0.5, 0.25], mode="same")   # known 2-tap system
    >>> est = estimate_h1(u, v, fs=1024.0, nperseg=512)
    >>> est.H.shape[0], est.H.shape[1]
    (1, 1)
    >>> bool(est.mean_coherence() > 0.99)
    True
    """
    sp = _prepare(
        x,
        y,
        fs,
        dict(
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            detrend=detrend,
            scaling=scaling,
            n_averages=n_averages,
        ),
    )
    H = _solve_right(sp.Gyx, sp.Gxx, rcond, ridge)
    est = FRFEstimate(
        H=H,
        freq_hz=sp.freq_hz,
        coherence=_ordinary_coherence(sp),
        multiple_coherence=_multiple_coherence(sp, rcond, ridge),
        method="H1",
        n_averages=sp.n_averages,
        spectra=sp,
        condition_number=_line_condition(sp.Gxx),
    )
    return est if f_range is None else est.band(f_range)


def estimate_h2(
    x: Any,
    y: Any = None,
    fs: float | None = None,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: Any = "constant",
    scaling: str = "density",
    n_averages: int | None = None,
    rcond: float = 1.0e-12,
    ridge: float = 0.0,
    f_range: tuple[float, float] | None = None,
) -> FRFEstimate:
    """``H2`` FRF estimate — least squares assuming noise on the **input**.

    .. math:: H_2 = G_{yy} G_{xy}^{-1}

    With a **single input** each response is handled on its own,
    ``H2_o = G_yy,oo / G_xy,o``, which satisfies ``H1 / H2 = gamma^2`` exactly
    for every response.  With several inputs ``G_xy`` is the ``(n_in, n_out)``
    cross spectrum and is inverted as a matrix, which requires a square problem
    (``n_out == n_in``) — the estimator has no meaningful over-determined form,
    because mixing responses that carry independent noise is precisely what it
    is trying to avoid.

    Notes
    -----
    ``H2`` is biased **high** at resonance and is the better estimator near
    anti-resonances, where the response — and hence the measured input, through
    the impedance of the shaker — is dominated by input noise.  Quoting both
    ``H1`` and ``H2`` brackets the true FRF.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.mpe.frf_estimation import estimate_h1, estimate_h2
    >>> rng = np.random.default_rng(1)
    >>> u = rng.standard_normal(8192)
    >>> v = np.convolve(u, [1.0, -0.4], mode="same") + 0.01 * rng.standard_normal(8192)
    >>> h1 = estimate_h1(u, v, fs=512.0, nperseg=512)
    >>> h2 = estimate_h2(u, v, fs=512.0, nperseg=512)
    >>> ratio = np.abs(h1.H / h2.H)[0, 0]
    >>> bool(np.allclose(ratio[1:], h1.coherence[0, 0][1:], atol=1e-9))
    True
    """
    sp = _prepare(
        x,
        y,
        fs,
        dict(
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            detrend=detrend,
            scaling=scaling,
            n_averages=n_averages,
        ),
    )
    n_out, n_in = sp.n_out, sp.n_in
    Gxy = sp.Gxy  # (n_in, n_out, n_freq)
    H = np.zeros((n_out, n_in, sp.n_freq), dtype=complex)
    if n_in == 1:
        # Single input: every response is treated on its own, using its own
        # autopower.  This is the form for which H1 / H2 == gamma^2 holds.
        gyy = np.einsum("oof->of", sp.Gyy)
        den = Gxy[0]  # (n_out, n_freq)
        with np.errstate(divide="ignore", invalid="ignore"):
            H[:, 0, :] = np.where(np.abs(den) > 0, gyy / np.where(den == 0, 1.0, den), 0.0)
    else:
        if n_out != n_in:
            raise ValueError(
                f"the MIMO H2 estimator inverts the (n_in, n_out) cross spectrum and "
                f"therefore needs a square problem, got {n_out} outputs for {n_in} "
                "inputs; estimate one output at a time, or use estimate_h1"
            )
        for f in range(sp.n_freq):
            A = Gxy[:, :, f]  # (n_in, n_out)
            Gyy = sp.Gyy[:, :, f]
            if ridge:
                # Tikhonov on the normal equations of  X A = Gyy.
                trace = float(np.real(np.trace(sp.Gxx[:, :, f]))) / max(n_in, 1)
                lhs = A @ A.conj().T + ridge * max(trace, 1e-300) * np.eye(n_in)
                rhs = Gyy @ A.conj().T
                H[:, :, f] = np.linalg.lstsq(lhs.T, rhs.T, rcond=rcond)[0].T
            else:
                # X A = Gyy  ->  A^T X^T = Gyy^T
                H[:, :, f] = np.linalg.lstsq(A.T, Gyy.T, rcond=rcond)[0].T
    est = FRFEstimate(
        H=H,
        freq_hz=sp.freq_hz,
        coherence=_ordinary_coherence(sp),
        multiple_coherence=_multiple_coherence(sp, rcond, ridge),
        method="H2",
        n_averages=sp.n_averages,
        spectra=sp,
        condition_number=_line_condition(sp.Gxx),
    )
    return est if f_range is None else est.band(f_range)


def estimate_frf(
    x: Any,
    y: Any = None,
    fs: float | None = None,
    *,
    method: str = "h1",
    **kwargs: Any,
) -> FRFEstimate:
    """Dispatch to :func:`estimate_h1` or :func:`estimate_h2` by name."""
    key = str(method).strip().lower()
    if key in ("h1", "1"):
        return estimate_h1(x, y, fs, **kwargs)
    if key in ("h2", "2"):
        return estimate_h2(x, y, fs, **kwargs)
    raise ValueError(f"unknown FRF estimator {method!r}; expected 'h1' or 'h2'")


def coherence(
    x: Any,
    y: Any = None,
    fs: float | None = None,
    *,
    kind: str = "ordinary",
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: Any = "constant",
    n_averages: int | None = None,
    rcond: float = 1.0e-12,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Coherence between excitation ``x`` and response ``y``.

    Parameters
    ----------
    kind:
        ``"ordinary"`` (default) returns the pairwise
        :math:`\\gamma^2_{oi}` shaped ``(n_out, n_in, n_freq)``;
        ``"multiple"`` returns the multiple coherence ``(n_out, n_freq)``,
        which is the right measure when several inputs act at once.

    Returns
    -------
    (freq_hz, gamma2)

    Notes
    -----
    Coherence is only meaningful when several segments are averaged: a single
    segment gives :math:`\\gamma^2 \\equiv 1` for *any* pair of signals.  Values
    below 1 come from output noise, leakage, non-linearity, or unmeasured
    inputs — the last of which is exactly what multiple coherence separates out.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.mpe.frf_estimation import coherence
    >>> rng = np.random.default_rng(2)
    >>> u = rng.standard_normal(16384)
    >>> v = np.convolve(u, [1.0, 0.5], mode="same")
    >>> f, g2 = coherence(u, v, fs=1024.0, nperseg=1024)
    >>> bool(np.mean(g2) > 0.99)
    True
    """
    sp = _prepare(
        x,
        y,
        fs,
        dict(
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            detrend=detrend,
            n_averages=n_averages,
        ),
    )
    key = str(kind).strip().lower()
    if key in ("ordinary", "pairwise", "simple", ""):
        return sp.freq_hz, _ordinary_coherence(sp)
    if key in ("multiple", "multi"):
        return sp.freq_hz, _multiple_coherence(sp, rcond, ridge)
    raise ValueError(f"unknown coherence kind {kind!r}")


def multiple_coherence(
    x: Any, y: Any = None, fs: float | None = None, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Multiple coherence — shorthand for ``coherence(..., kind="multiple")``."""
    return coherence(x, y, fs, kind="multiple", **kwargs)


def _line_condition(G: np.ndarray) -> np.ndarray:
    """Condition number of the input spectral matrix on every line."""
    n = G.shape[0]
    if n == 1:
        return np.ones(G.shape[2])
    out = np.empty(G.shape[2])
    for f in range(G.shape[2]):
        s = np.linalg.svd(G[:, :, f], compute_uv=False)
        out[f] = float(s[0] / s[-1]) if s[-1] > 0 else math.inf
    return out
