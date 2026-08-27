"""Stationary random response: force PSD in, response PSD and RMS out.

For a linear system excited by a stationary Gaussian process the response is fully
described by the cross-spectral density matrix

    S_uu(f) = H(f) S_ff(f) H(f)^H

with ``H`` the receptance (or mobility / accelerance) block from
:func:`~femtools.dynamics.frf.modal_frf`. Everything here is *one-sided*: ``S_ff`` is
given in ``unit^2/Hz`` over ``f >= 0``, so the variance of response DOF ``o`` is the plain
integral of its auto-spectrum,

    sigma_o^2 = integral_0^inf S_oo(f) df

and, the process being zero mean, the RMS *is* the 1-sigma level. The higher spectral
moments ``m_k = integral f^k S(f) df`` give the mean upcrossing rate ``sqrt(m2/m0)``
(Rice) and, with it, a Davenport peak factor for a stated exposure duration — the
quantities a random-vibration load case is actually assessed on.

Accuracy is bounded by two things the caller controls: the frequency grid must resolve
every resonance in the band (a half-power bandwidth of ``2 zeta f_r`` spanned by a
handful of lines at least, otherwise the trapezoidal integral misses most of the peak),
and the band must be wide enough that the tails carry no variance. :func:`miles_rms` is
the closed-form SDOF answer to check a single-mode case against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._utils import TWO_PI
from .frf import FRFResult, modal_frf

__all__ = ["PSDResult", "miles_rms", "psd_response"]


def _integrate(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Trapezoidal integral of ``y`` over the last axis with abscissa ``f``."""
    if f.size < 2:
        return np.zeros(y.shape[:-1])
    # np.trapezoid is the numpy >= 2 spelling of np.trapz; the project supports both.
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # noqa: B009
    return np.asarray(trapezoid(y, x=f, axis=-1))


@dataclass
class PSDResult:
    """Stationary response spectra and the statistics derived from them.

    Attributes
    ----------
    psd:
        One-sided auto-spectral density of every response DOF, shape
        ``(n_out, n_freq)``, real and non-negative, in ``unit^2/Hz``.
    cross_psd:
        Full complex cross-spectral matrix ``(n_out, n_out, n_freq)``, or ``None`` when
        it was not requested. Hermitian at every line; its diagonal is :attr:`psd`.
    freq_hz:
        Frequency axis in Hz, shape ``(n_freq,)``.
    rms:
        Root-mean-square response per output DOF, shape ``(n_out,)``. Equivalently the
        1-sigma level, since the process is zero mean.
    outputs, inputs:
        DOF indices behind the rows of :attr:`psd` and the excitation set.
    response:
        ``"receptance"``, ``"mobility"`` or ``"accelerance"`` — the units of :attr:`psd`
        follow it (displacement^2/Hz, velocity^2/Hz, acceleration^2/Hz).
    """

    psd: np.ndarray
    freq_hz: np.ndarray
    rms: np.ndarray
    cross_psd: np.ndarray | None = None
    outputs: np.ndarray | None = None
    inputs: np.ndarray | None = None
    response: str = "receptance"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        psd = np.atleast_2d(np.asarray(self.psd, dtype=float))
        self.psd = np.ascontiguousarray(psd)
        self.freq_hz = np.atleast_1d(np.asarray(self.freq_hz, dtype=float)).reshape(-1)
        if self.freq_hz.size != self.psd.shape[1]:
            raise ValueError(
                f"freq_hz has {self.freq_hz.size} points but psd has {self.psd.shape[1]}"
            )
        self.rms = np.atleast_1d(np.asarray(self.rms, dtype=float)).reshape(-1)
        if self.rms.size != self.psd.shape[0]:
            raise ValueError("rms must have one entry per response DOF")
        if self.outputs is not None:
            self.outputs = np.asarray(self.outputs, dtype=int).reshape(-1)
        if self.inputs is not None:
            self.inputs = np.asarray(self.inputs, dtype=int).reshape(-1)

    # -- shape ------------------------------------------------------------
    @property
    def n_out(self) -> int:
        """Number of response DOFs."""
        return int(self.psd.shape[0])

    @property
    def n_freq(self) -> int:
        """Number of frequency lines."""
        return int(self.psd.shape[1])

    @property
    def omega(self) -> np.ndarray:
        """Frequency axis in rad/s."""
        return TWO_PI * self.freq_hz

    # -- statistics -------------------------------------------------------
    @property
    def sigma(self) -> np.ndarray:
        """1-sigma level per response DOF; identical to :attr:`rms` (zero-mean process)."""
        return self.rms

    @property
    def three_sigma(self) -> np.ndarray:
        """3-sigma level per response DOF, the usual random-vibration design number."""
        return 3.0 * self.rms

    @property
    def variance(self) -> np.ndarray:
        """Response variance ``sigma^2`` per DOF, i.e. the zeroth spectral moment."""
        return self.rms**2

    def moment(self, order: int = 0) -> np.ndarray:
        """Spectral moment ``m_k = integral f^k S(f) df`` per response DOF, ``f`` in Hz."""
        k = int(order)
        weight = self.freq_hz**k if k else np.ones(self.n_freq)
        return _integrate(self.psd * weight[None, :], self.freq_hz)

    def zero_crossing_rate_hz(self) -> np.ndarray:
        """Mean rate of upward zero crossings ``sqrt(m2/m0)``, in Hz (Rice).

        For a narrow-band response this is the apparent frequency of the signal; it is
        what turns an RMS level into a cycle count for fatigue.
        """
        m0, m2 = self.moment(0), self.moment(2)
        return np.sqrt(np.divide(m2, m0, out=np.zeros_like(m2), where=m0 > 0.0))

    def peak_factor(self, duration_s: float) -> np.ndarray:
        """Davenport peak factor for an exposure of ``duration_s`` seconds.

        ``g = sqrt(2 ln(nu T)) + 0.5772 / sqrt(2 ln(nu T))`` with ``nu`` the upcrossing
        rate. Falls back to ``0`` where the response has no variance.
        """
        t = float(duration_s)
        if t <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s}")
        n = self.zero_crossing_rate_hz() * t
        arg = 2.0 * np.log(np.where(n > 1.0, n, np.e**0.5))
        g = np.sqrt(arg) + 0.5772 / np.sqrt(arg)
        return np.where(self.rms > 0.0, g, 0.0)

    def peak(self, duration_s: float) -> np.ndarray:
        """Expected largest excursion over ``duration_s``: ``peak_factor * rms``."""
        return self.peak_factor(duration_s) * self.rms

    def cumulative_rms(self) -> np.ndarray:
        """Running RMS accumulated up to each frequency line, shape ``(n_out, n_freq)``.

        The curve rises in steps at the resonances, which is the quickest way to see
        which modes actually drive the response.
        """
        if self.n_freq < 2:
            return np.zeros_like(self.psd)
        df = np.diff(self.freq_hz)
        panels = 0.5 * (self.psd[:, 1:] + self.psd[:, :-1]) * df[None, :]
        running = np.concatenate(
            [np.zeros((self.n_out, 1)), np.cumsum(panels, axis=1)], axis=1
        )
        return np.sqrt(np.clip(running, 0.0, None))

    def index_at(self, freq_hz: float) -> int:
        """Index of the frequency line closest to ``freq_hz``."""
        return int(np.argmin(np.abs(self.freq_hz - float(freq_hz))))


def _as_force_psd(spec: Any, n_in: int, n_freq: int) -> np.ndarray:
    """Broadcast a force-PSD specification to ``(n_in, n_in, n_freq)`` complex.

    Accepted forms, in the order they are tried:

    * scalar — one constant level, uncorrelated across the inputs;
    * ``(n_freq,)`` — one spectrum shared by every input, uncorrelated;
    * ``(n_in,)`` — a constant level per input, uncorrelated;
    * ``(n_in, n_freq)`` — an auto-spectrum per input, uncorrelated;
    * ``(n_in, n_in)`` — a constant cross-spectral matrix;
    * ``(n_in, n_in, n_freq)`` — the general correlated case.

    With ``n_in == n_freq`` the shapes collide; the per-frequency reading wins, so pass
    the 3-D form when that is not what you meant.
    """
    arr = np.asarray(spec)
    if arr.dtype.kind not in "fiuc":
        raise TypeError(f"force_psd must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(complex, copy=False)
    eye = np.eye(n_in)

    if arr.ndim == 0:
        S = eye[:, :, None] * np.ones(n_freq) * arr
    elif arr.ndim == 1:
        if arr.size == n_freq:
            S = eye[:, :, None] * arr[None, None, :]
        elif arr.size == n_in:
            S = (np.diag(arr))[:, :, None] * np.ones(n_freq)
        else:
            raise ValueError(
                f"1-D force_psd must have {n_freq} entries (one per frequency line) or "
                f"{n_in} (one per input), got {arr.size}"
            )
    elif arr.ndim == 2:
        if arr.shape == (n_in, n_freq):
            S = np.zeros((n_in, n_in, n_freq), dtype=complex)
            idx = np.arange(n_in)
            S[idx, idx, :] = arr
        elif arr.shape == (n_in, n_in):
            S = arr[:, :, None] * np.ones(n_freq)
        else:
            raise ValueError(
                f"2-D force_psd must be {(n_in, n_freq)} (auto-spectra) or "
                f"{(n_in, n_in)} (constant cross-spectra), got {arr.shape}"
            )
    elif arr.ndim == 3:
        if arr.shape[:2] != (n_in, n_in) or arr.shape[2] not in (1, n_freq):
            raise ValueError(
                f"3-D force_psd must be {(n_in, n_in, n_freq)}, got {arr.shape}"
            )
        S = arr * np.ones(n_freq) if arr.shape[2] == 1 else arr
    else:
        raise ValueError(f"force_psd must be at most 3-D, got shape {arr.shape}")

    S = np.ascontiguousarray(S, dtype=complex)
    diag = np.einsum("iif->if", S)
    if np.any(np.abs(diag.imag) > 1e-9 * (np.abs(diag).max() or 1.0)):
        raise ValueError("the auto-spectra on the diagonal of force_psd must be real")
    if np.any(diag.real < 0.0):
        raise ValueError("force_psd has a negative auto-spectrum; a PSD cannot be negative")
    if n_in > 1:
        asym = np.abs(S - np.conj(np.swapaxes(S, 0, 1))).max()
        if asym > 1e-8 * (np.abs(S).max() or 1.0):
            raise ValueError(
                "force_psd must be Hermitian at every line: S_jk = conj(S_kj)"
            )
    return S


def psd_response(
    modal: Any,
    force_psd: Any,
    freq_hz: Any = None,
    damping: Any = None,
    *,
    inputs: Any = None,
    outputs: Any = None,
    response: str = "receptance",
    cross: bool = False,
) -> PSDResult:
    """Stationary response PSD and RMS of a modal model under a force PSD.

    ``S_uu(f) = H(f) S_ff(f) H(f)^H`` with ``H`` synthesised by
    :func:`~femtools.dynamics.frf.modal_frf`, followed by
    ``rms_o = sqrt(integral S_oo df)`` over the supplied band.

    Parameters
    ----------
    modal:
        Modal model — anything :func:`~femtools.dynamics.modal.as_modal` accepts — or an
        already computed :class:`~femtools.dynamics.frf.FRFResult`, in which case
        ``freq_hz``, ``damping``, ``inputs``, ``outputs`` and ``response`` must be left
        alone because the FRF already fixes them.
    force_psd:
        One-sided force spectral density in ``force^2/Hz``: a scalar, a ``(n_freq,)``
        spectrum shared by every input, a ``(n_in,)`` level per input, ``(n_in, n_freq)``
        auto-spectra — all four uncorrelated — a constant ``(n_in, n_in)`` cross-spectral
        matrix, or the general ``(n_in, n_in, n_freq)`` correlated case. Where the shapes
        collide (``n_in == n_freq``) the per-frequency reading wins.
    freq_hz:
        Frequency lines in Hz. They set both the synthesis grid and the integration
        band, so they must resolve the resonances (see the module docstring).
    damping:
        Anything :func:`~femtools.dynamics.damping.as_damping` understands. A random
        analysis without damping is meaningless — the variance diverges at the poles.
    inputs, outputs:
        Excitation / response DOF selection, as in
        :func:`~femtools.dynamics.frf.modal_frf`.
    response:
        ``"receptance"`` (default), ``"mobility"`` or ``"accelerance"``; the PSD comes
        out in the square of that unit per Hz.
    cross:
        Also return the full complex cross-spectral matrix between the response DOFs.
        It costs ``n_out^2 n_freq`` complex numbers, so it is off by default.

    Returns
    -------
    PSDResult
    """
    if isinstance(modal, FRFResult):
        clashes = [
            name
            for name, value in (
                ("freq_hz", freq_hz),
                ("damping", damping),
                ("inputs", inputs),
                ("outputs", outputs),
            )
            if value is not None
        ]
        if response not in ("receptance", modal.response):
            clashes.append("response")
        if clashes:
            raise ValueError(
                "an FRFResult already fixes the frequency lines, damping, DOFs and "
                f"response type; drop {', '.join(clashes)}"
            )
        frf = modal
    else:
        if freq_hz is None:
            raise ValueError("freq_hz is required to synthesise the FRF")
        frf = modal_frf(modal, inputs, outputs, freq_hz, damping, response=response)

    H = frf.H
    n_out, n_in, n_freq = H.shape
    f = frf.freq_hz
    S_ff = _as_force_psd(force_psd, n_in, n_freq)

    # H S_ff, then close it with H^H — once for the full matrix, contracted for the
    # diagonal so that the common case never allocates (n_out, n_out, n_freq).
    HS = np.einsum("ojf,jkf->okf", H, S_ff, optimize=True)
    if cross:
        S_uu = np.einsum("okf,pkf->opf", HS, np.conj(H), optimize=True)
        auto = np.real(np.einsum("oof->of", S_uu))
    else:
        S_uu = None
        auto = np.real(np.einsum("okf,okf->of", HS, np.conj(H), optimize=True))
    auto = np.clip(auto, 0.0, None)  # round-off can push a near-zero line negative

    rms = np.sqrt(np.clip(_integrate(auto, f), 0.0, None))

    meta = {
        "n_in": n_in,
        "n_freq": n_freq,
        "band_hz": (float(f[0]), float(f[-1])) if f.size else (0.0, 0.0),
        "frf_method": frf.method,
        "frf_meta": dict(frf.meta),
    }
    if f.size < 2:
        meta["warning"] = "a single frequency line carries no bandwidth, so rms is zero"
    return PSDResult(
        psd=auto,
        freq_hz=f.copy(),
        rms=rms,
        cross_psd=S_uu,
        outputs=None if frf.outputs is None else frf.outputs.copy(),
        inputs=None if frf.inputs is None else frf.inputs.copy(),
        response=frf.response,
        meta=meta,
    )


def miles_rms(
    freq_hz: float, zeta: float, psd_level: float, *, modal_mass: float = 1.0
) -> float:
    """Closed-form RMS response of an SDOF to a white input (Miles' equation).

    ``sigma = sqrt(pi/2 * f_n * S_0 / (2 zeta)) / (omega_n^2 m)`` — the exact variance of
    an SDOF of unit-normalised mass ``m`` driven by a flat one-sided force PSD ``S_0``,
    integrated over an infinite band. Use it to sanity-check a :func:`psd_response` run
    on a single mode, or as a quick estimate when the input really is broadband relative
    to the half-power bandwidth ``2 zeta f_n``.

    Parameters
    ----------
    freq_hz:
        Natural frequency in Hz.
    zeta:
        Viscous damping ratio.
    psd_level:
        Flat one-sided force PSD in ``force^2/Hz``.
    modal_mass:
        Generalised mass of the mode; ``1.0`` for a mass-normalised mode shape.

    Returns
    -------
    float
        RMS displacement (1-sigma).
    """
    fn, z = float(freq_hz), float(zeta)
    if fn <= 0.0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz}")
    if z <= 0.0:
        raise ValueError(f"zeta must be positive, got {zeta}")
    if psd_level < 0.0:
        raise ValueError(f"psd_level must be non-negative, got {psd_level}")
    omega_n = TWO_PI * fn
    force_variance = (np.pi / 2.0) * fn * float(psd_level) / (2.0 * z)
    return float(np.sqrt(force_variance) / (omega_n**2 * float(modal_mass)))
