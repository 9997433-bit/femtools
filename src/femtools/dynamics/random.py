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

Accuracy is bounded by two things the caller controls: the band must be wide enough that
the tails carry no variance, and the grid must resolve the *shape* of the spectrum, not
merely hit the peaks. The second is the one that bites. A grid fine enough at a resonance
is usually still coarse on its shoulders, where the curvature is highest, and the
trapezoidal integral converges as ``df^2``: on a 5 %-damped SDOF, 8 k uniform lines over
100 ``f_n`` leave 6e-4 relative error in the RMS while 40 k geometrically spaced lines
reach 1e-8. Log spacing is the cheap fix. :func:`miles_rms` is the closed-form SDOF answer
to check a single-mode case against.

Support (base) acceleration
---------------------------
A random-vibration test specification is almost never a force PSD: it is the acceleration
of the shaker head, and the structure is bolted to it. Writing the motion of the structure
as ``u_abs = r x_g + u`` — the rigid ride along with the support plus the deformation
relative to it — turns the support motion into an equivalent force on the *relative*
coordinates,

    M u'' + C u' + K u = -M r a_g(t),

so nothing new is needed: the excitation is the ordinary force PSD of ``-L a_g`` with
``L = (M r)`` restricted to the excited DOFs, and ``psd_response(..., base_accel=S_aa)``
assembles exactly that. What comes out is then the **relative** response (deformation, and
so stress) unless ``base_absolute=True`` adds the support motion back, which is what an
accelerometer on the structure would read.

The SDOF closed forms both sides of that switch, for a flat one-sided ``S_a`` and a
mass-normalised mode, are

    sigma_rel = sqrt(S_a / (8 zeta omega_n^3))      (relative displacement)
    sigma_abs = sqrt(pi f_n S_a / (4 zeta)) * sqrt(1 + 4 zeta^2)   (absolute acceleration)

The first is :func:`miles_rms` evaluated on the acceleration level; the second is Miles'
number ``sqrt(pi/2 f_n Q S_a)`` times the ``sqrt(1 + 4 zeta^2)`` that the damper's own
force contributes and that the usual quotation of the equation drops (0.08 % at 2 %
damping, 2 % at 10 %).

Storage
-------
:func:`dump_psd` and :func:`load_psd` put a :class:`PSDResult` on disk as a plain ``.npz``
archive, the same way :func:`~femtools.dynamics.frf.dump_frf` stores an FRF and
:func:`~femtools.dynamics.superelement.dump_cms` a reduced component. The spectra and the
frequency axis are written as raw ``float64`` / ``complex128``, so they come back
**bit-identical** — a random-vibration load case is assessed on the RMS and the 3-sigma
level it produces, and those numbers are only comparable between runs if the spectrum
behind them did not move in its last bits on the way through the file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._utils import TWO_PI, dumps_meta, get_field, json_meta, npz_path, npz_text
from .frf import FRFResult, modal_frf

__all__ = ["PSDResult", "dump_psd", "load_psd", "miles_rms", "psd_response"]


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


def _as_force_psd(spec: Any, n_in: int, n_freq: int, name: str = "force_psd") -> np.ndarray:
    """Broadcast a PSD specification to ``(n_in, n_in, n_freq)`` complex.

    Accepted forms, in the order they are tried:

    * scalar — one constant level, uncorrelated across the inputs;
    * ``(n_freq,)`` — one spectrum shared by every input, uncorrelated;
    * ``(n_in,)`` — a constant level per input, uncorrelated;
    * ``(n_in, n_freq)`` — an auto-spectrum per input, uncorrelated;
    * ``(n_in, n_in)`` — a constant cross-spectral matrix;
    * ``(n_in, n_in, n_freq)`` — the general correlated case.

    With ``n_in == n_freq`` the shapes collide; the per-frequency reading wins, so pass
    the 3-D form when that is not what you meant. ``name`` only labels the diagnostics;
    :func:`psd_response` reuses this for ``base_accel``, whose columns are support
    directions rather than input DOFs.
    """
    arr = np.asarray(spec)
    if arr.dtype.kind not in "fiuc":
        raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
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
                f"1-D {name} must have {n_freq} entries (one per frequency line) or "
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
                f"2-D {name} must be {(n_in, n_freq)} (auto-spectra) or "
                f"{(n_in, n_in)} (constant cross-spectra), got {arr.shape}"
            )
    elif arr.ndim == 3:
        if arr.shape[:2] != (n_in, n_in) or arr.shape[2] not in (1, n_freq):
            raise ValueError(f"3-D {name} must be {(n_in, n_in, n_freq)}, got {arr.shape}")
        S = arr * np.ones(n_freq) if arr.shape[2] == 1 else arr
    else:
        raise ValueError(f"{name} must be at most 3-D, got shape {arr.shape}")

    S = np.ascontiguousarray(S, dtype=complex)
    diag = np.einsum("iif->if", S)
    if np.any(np.abs(diag.imag) > 1e-9 * (np.abs(diag).max() or 1.0)):
        raise ValueError(f"the auto-spectra on the diagonal of {name} must be real")
    if np.any(diag.real < 0.0):
        raise ValueError(f"{name} has a negative auto-spectrum; a PSD cannot be negative")
    if n_in > 1:
        asym = np.abs(S - np.conj(np.swapaxes(S, 0, 1))).max()
        if asym > 1e-8 * (np.abs(S).max() or 1.0):
            raise ValueError(f"{name} must be Hermitian at every line: S_jk = conj(S_kj)")
    return S


def _as_base_matrix(
    value: Any, n_rows: int, name: str, n_dir: int | None = None
) -> np.ndarray:
    """Broadcast a participation / influence specification to ``(n_rows, n_dir)`` real.

    A scalar fills the column, a ``(n_rows,)`` vector *is* the single column, and a 2-D
    array is taken as-is with one column per support direction.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((n_rows, 1 if n_dir is None else n_dir), float(arr))
    if arr.ndim == 1:
        if arr.size != n_rows:
            raise ValueError(
                f"a 1-D {name} describes a single support direction, so it needs one "
                f"entry per DOF ({n_rows} here), got {arr.size}"
            )
        return arr.reshape(n_rows, 1)
    if arr.ndim == 2:
        if arr.shape[0] != n_rows:
            raise ValueError(f"{name} must have {n_rows} rows, got shape {arr.shape}")
        if n_dir is not None and arr.shape[1] != n_dir:
            raise ValueError(
                f"{name} has {arr.shape[1]} support directions but the excitation has "
                f"{n_dir}"
            )
        return arr
    raise ValueError(f"{name} must be at most 2-D, got shape {arr.shape}")


def _base_ride_scale(response: str, omega: np.ndarray) -> np.ndarray:
    """Multiplier turning the support *acceleration* into the requested response kind.

    The support rides at ``a_g``; its velocity is ``a_g / (i w)`` and its displacement
    ``-a_g / w^2``. Both are singular at ``f = 0``, where a stationary acceleration
    spectrum carries no information about the displacement of the support at all, so the
    DC line is refused rather than returned as an infinity.
    """
    if response == "accelerance":
        return np.ones(omega.shape, dtype=complex)
    if np.any(omega <= 0.0):
        raise ValueError(
            f"the absolute {response} response needs the support displacement or "
            "velocity, which is the acceleration divided by w^2 or i w and is unbounded "
            "at f = 0; drop the DC line, ask for response='accelerance', or leave "
            "base_absolute off and read the relative response"
        )
    if response == "mobility":
        return 1.0 / (1j * omega)
    return -1.0 / omega**2 + 0j


def _base_transfer(
    H: np.ndarray,  # noqa: N803
    response: str,
    omega: np.ndarray,
    base_accel: Any,
    base_participation: Any,
    base_influence: Any,
    base_absolute: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Support-acceleration transfer ``G``, its input spectrum and the provenance meta.

    The support motion enters the relative coordinates as the equivalent force
    ``-L a_g``, so ``G_rel = -H L``; the absolute response adds the rigid ride ``r a_g``,
    scaled from acceleration to whatever ``response`` asks for. On an SDOF with
    ``L = r = 1`` and a mass-normalised mode this reproduces the textbook
    transmissibility ``(w_n^2 + 2 i zeta w_n w) / (w_n^2 - w^2 + 2 i zeta w_n w)``
    exactly, which is the check the closed forms in the module docstring rest on.
    """
    n_out, n_in, n_freq = H.shape
    L = _as_base_matrix(
        1.0 if base_participation is None else base_participation,
        n_in,
        "base_participation",
    )
    n_dir = int(L.shape[1])
    S_aa = _as_force_psd(base_accel, n_dir, n_freq, "base_accel")

    G = -np.einsum("oif,id->odf", H, L.astype(complex), optimize=True)
    if base_absolute:
        R = _as_base_matrix(  # noqa: N806
            1.0 if base_influence is None else base_influence,
            n_out,
            "base_influence",
            n_dir,
        )
        ride = _base_ride_scale(response, omega)
        G = G + R.astype(complex)[:, :, None] * ride[None, None, :]

    return (
        np.ascontiguousarray(G, dtype=complex),
        S_aa,
        {
            "excitation": "base_accel",
            "base_frame": "absolute" if base_absolute else "relative",
            "n_base_dir": n_dir,
            "base_participation": L.copy(),
        },
    )


def psd_response(
    modal: Any,
    force_psd: Any = None,
    freq_hz: Any = None,
    damping: Any = None,
    *,
    base_accel: Any = None,
    base_participation: Any = None,
    base_influence: Any = None,
    base_absolute: bool = False,
    inputs: Any = None,
    outputs: Any = None,
    response: str = "receptance",
    cross: bool = False,
) -> PSDResult:
    """Stationary response PSD and RMS of a modal model under a force or base-motion PSD.

    ``S_uu(f) = G(f) S(f) G(f)^H`` followed by ``rms_o = sqrt(integral S_oo df)`` over the
    supplied band. For a force PSD ``G`` is the FRF ``H`` synthesised by
    :func:`~femtools.dynamics.frf.modal_frf`; for a support acceleration it is the
    base-to-response transfer built from the same ``H`` (see below and the module
    docstring). Exactly one of ``force_psd`` and ``base_accel`` is given.

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
    base_accel:
        One-sided acceleration spectral density of the *support*, in ``accel^2/Hz``, in
        place of ``force_psd``. The same six shapes are accepted, but their leading
        dimension counts **support directions** (``n_dir``, the number of columns of
        ``base_participation``, one by default), not input DOFs — a single rigid support
        drives every excited DOF perfectly coherently, which the rank-one force spectrum
        ``L S_aa L^H`` expresses and a per-input reading would not.
    base_participation:
        ``L``, the load-participation block ``(M r)`` restricted to the ``inputs`` DOFs,
        shape ``(n_in,)`` or ``(n_in, n_dir)``; a scalar fills it. It converts a support
        acceleration into the equivalent force ``-L a_g``, so for a lumped-mass model it
        is simply the mass at each excited DOF in each support direction. Defaults to
        ones, which is the SDOF / unit-modal-mass case and is what the closed forms in the
        module docstring assume.
    base_influence:
        ``r`` restricted to the response DOFs, shape ``(n_out,)`` or ``(n_out, n_dir)``:
        how far each output moves when the support moves one unit and the structure does
        not deform. Only used when ``base_absolute`` is set; defaults to ones, i.e. every
        output rides along with the support one for one.
    base_absolute:
        Return the *absolute* response (the support motion plus the deformation, which is
        what an accelerometer on the structure reads) instead of the relative one. Off by
        default, because the relative motion is what carries the stress. With
        ``response="receptance"`` or ``"mobility"`` this needs the support displacement or
        velocity and therefore refuses a line at ``f = 0``.
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
        ``meta["excitation"]`` says which path was taken and, for a support acceleration,
        ``meta["base_frame"]`` whether the spectra are relative or absolute.
    """
    if (force_psd is None) == (base_accel is None):
        raise ValueError(
            "exactly one of force_psd and base_accel is required; they are two "
            "descriptions of the excitation, not two excitations"
        )
    if base_accel is None:
        unused = [
            name
            for name, value in (
                ("base_participation", base_participation),
                ("base_influence", base_influence),
                ("base_absolute", base_absolute or None),
            )
            if value is not None
        ]
        if unused:
            raise ValueError(
                f"{', '.join(unused)} only applies to a base_accel excitation; a force "
                "PSD has no support motion to ride on"
            )

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

    if base_accel is None:
        G = H
        S_ee = _as_force_psd(force_psd, n_in, n_freq)
        meta_excitation: dict[str, Any] = {"excitation": "force"}
    else:
        G, S_ee, meta_excitation = _base_transfer(
            H,
            frf.response,
            TWO_PI * f,
            base_accel,
            base_participation,
            base_influence,
            base_absolute,
        )

    # G S, then close it with G^H — once for the full matrix, contracted for the
    # diagonal so that the common case never allocates (n_out, n_out, n_freq).
    GS = np.einsum("ojf,jkf->okf", G, S_ee, optimize=True)
    if cross:
        S_uu = np.einsum("okf,pkf->opf", GS, np.conj(G), optimize=True)
        auto = np.real(np.einsum("oof->of", S_uu))
    else:
        S_uu = None
        auto = np.real(np.einsum("okf,okf->of", GS, np.conj(G), optimize=True))
    auto = np.clip(auto, 0.0, None)  # round-off can push a near-zero line negative

    rms = np.sqrt(np.clip(_integrate(auto, f), 0.0, None))

    meta = {
        "n_in": n_in,
        "n_freq": n_freq,
        "band_hz": (float(f[0]), float(f[-1])) if f.size else (0.0, 0.0),
        "frf_method": frf.method,
        "frf_meta": dict(frf.meta),
        **meta_excitation,
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


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

#: Tag written into every archive; :func:`load_psd` refuses anything else.
PSD_FORMAT = "femtools.dynamics.random/1"

#: Fields an archive cannot be missing — everything else has a defined default.
_REQUIRED = ("psd", "freq_hz", "rms")
#: Written only when the source carries them; absent in the archive means ``None``.
_OPTIONAL_DOFS = ("outputs", "inputs")


def _as_psd(result: Any) -> PSDResult:
    """Coerce a duck-typed spectrum carrier to a :class:`PSDResult` for storage."""
    if isinstance(result, PSDResult):
        return result
    psd = get_field(result, "psd")
    freq_hz = get_field(result, "freq_hz")
    if psd is None or freq_hz is None:
        absent = "psd" if psd is None else "freq_hz"
        raise TypeError(
            f"{type(result).__name__} has no {absent!r}; a block of response spectra "
            "must carry the auto-spectra psd and the frequency axis freq_hz to be worth "
            "storing"
        )
    rms = get_field(result, "rms")
    if rms is None:
        # The RMS is the integral of the auto-spectra over the band, so a source that
        # carries the spectra but not the statistic is not missing anything: derive it
        # with the same trapezoidal rule psd_response uses rather than refuse the dump.
        axis = np.atleast_1d(np.asarray(freq_hz, dtype=float)).reshape(-1)
        auto = np.atleast_2d(np.asarray(psd, dtype=float))
        rms = np.sqrt(np.clip(_integrate(auto, axis), 0.0, None))
    response = get_field(result, "response")
    return PSDResult(
        psd=psd,
        freq_hz=freq_hz,
        rms=rms,
        cross_psd=get_field(result, "cross_psd"),
        outputs=get_field(result, "outputs"),
        inputs=get_field(result, "inputs"),
        response=str(response) if response else "receptance",
        meta=dict(get_field(result, "meta") or {}),
    )


def dump_psd(result: Any, path: Any, *, compress: bool = False, meta: Any = None) -> Any:
    """Write a block of response spectra to an ``.npz`` archive.

    A random-vibration run is cheap to *state* and expensive to *reproduce*: the spectra
    depend on the modal basis, the damping model, the excitation shapes and — most easily
    lost of all — the frequency grid, because the RMS is a trapezoidal integral over
    exactly the lines that were synthesised and a regenerated grid gives a slightly
    different number. Storing the result is how the 3-sigma level in a report stays tied
    to the spectrum it was read off. This is the :func:`~femtools.dynamics.frf.dump_frf`
    of the random side and writes the same kind of archive: one array per field, a
    ``format`` tag, and ``meta`` as JSON.

    :attr:`PSDResult.psd`, :attr:`PSDResult.freq_hz` and :attr:`PSDResult.rms` are stored
    as raw ``float64`` and the cross-spectral matrix as raw ``complex128``, so all four
    come back **bit-identical**.

        dump_psd(psd_response(modes, 1.0, f, 0.02), "liftoff.npz")
        S = load_psd("liftoff.npz")          # a PSDResult again
        S.three_sigma, S.peak(duration_s=60.0)

    Parameters
    ----------
    result:
        A :class:`PSDResult`, or any object or mapping exposing at least ``psd`` and
        ``freq_hz`` — ``rms``, ``cross_psd``, ``outputs``, ``inputs``, ``response`` and
        ``meta`` are stored when present. A source that carries no ``rms`` gets it
        integrated from its own spectra. A duck-typed source goes through
        :class:`PSDResult`'s own validation first, so a 1-D ``psd`` is read as the
        single-output block it is.
    path:
        Destination. A ``str`` or path-like without an ``.npz`` suffix gets one, and the
        resolved :class:`~pathlib.Path` is returned; an open binary file object is written
        to as-is and returned unchanged.
    compress:
        Use ``np.savez_compressed``. Response spectra are dense and span orders of
        magnitude across a resonance, so they rarely compress well; this is off by
        default and the bits that come back are identical either way.
    meta:
        Extra entries merged into the stored ``meta`` mapping, overriding the source's own.
        Anything JSON cannot represent is stored as its ``str``, and tuples come back as
        lists — ``meta`` is provenance, not data.

    Returns
    -------
    pathlib.Path or file object
        Where the archive was written.

    Raises
    ------
    TypeError
        If ``result`` carries no ``psd`` and ``freq_hz``.
    ValueError
        If ``cross_psd`` is present but is not ``(n_out, n_out, n_freq)``.
    """
    spectra = _as_psd(result)

    payload: dict[str, Any] = {
        "psd": spectra.psd,
        "freq_hz": spectra.freq_hz,
        "rms": spectra.rms,
        "format": np.array(PSD_FORMAT),
        "response": np.array(spectra.response),
        "source_class": np.array(type(result).__name__),
        "meta_json": np.array(dumps_meta(json_meta(spectra, meta))),
    }
    if spectra.cross_psd is not None:
        cross = np.ascontiguousarray(spectra.cross_psd, dtype=complex)
        expected = (spectra.n_out, spectra.n_out, spectra.n_freq)
        if cross.shape != expected:
            raise ValueError(
                f"cross_psd must be {expected} to match the auto-spectra, got "
                f"{cross.shape}"
            )
        payload["cross_psd"] = cross
    for name in _OPTIONAL_DOFS:
        value = getattr(spectra, name)
        # Absent from the archive is how "the block was never restricted to a DOF
        # selection" is written down; an empty array would say the opposite.
        if value is not None:
            payload[name] = np.asarray(value, dtype=np.int64).reshape(-1)

    target = npz_path(path)
    save = np.savez_compressed if compress else np.savez
    save(target if target is not None else path, **payload)
    return target if target is not None else path


def load_psd(path: Any) -> PSDResult:
    """Read response spectra back from an ``.npz`` archive written by :func:`dump_psd`.

    ``psd``, ``freq_hz``, ``rms`` and ``cross_psd`` are bit-identical to what was written,
    so every statistic derived from them — :attr:`~PSDResult.three_sigma`,
    :meth:`~PSDResult.moment`, :meth:`~PSDResult.zero_crossing_rate_hz`,
    :meth:`~PSDResult.peak` — reproduces exactly. ``outputs`` / ``inputs`` come back as
    the DOF selections they were, or as ``None`` when the block was never restricted to
    one, and ``cross_psd`` is ``None`` when the run did not request it. ``response`` is
    restored, so a stored acceleration spectrum still reads in ``accel^2/Hz``. ``meta``
    round-trips through JSON, so its tuples arrive as lists, and gains a ``loaded_from``
    entry.

    Parameters
    ----------
    path:
        Source archive: a path, a path without its ``.npz`` suffix, or an open binary
        file object.

    Returns
    -------
    PSDResult

    Raises
    ------
    ValueError
        If the archive was not written by :func:`dump_psd`, or has lost ``psd``,
        ``freq_hz`` or ``rms``.
    """
    target = npz_path(path)
    with np.load(target if target is not None else path, allow_pickle=False) as data:
        tag = npz_text(data, "format")
        if tag != PSD_FORMAT:
            raise ValueError(
                f"{path!r} is not a femtools PSD archive (format tag "
                f"{tag or 'absent'!r}, expected {PSD_FORMAT!r})"
            )
        missing = [name for name in _REQUIRED if name not in data.files]
        if missing:
            raise ValueError(
                f"{path!r} claims to be a PSD archive but is missing {', '.join(missing)}"
            )
        arrays = {name: np.array(data[name]) for name in _REQUIRED}
        cross_psd = np.array(data["cross_psd"]) if "cross_psd" in data.files else None
        dofs = {
            name: (np.array(data[name]) if name in data.files else None)
            for name in _OPTIONAL_DOFS
        }
        response = npz_text(data, "response", "receptance")
        meta_text = npz_text(data, "meta_json", "{}")

    meta = dict(json.loads(meta_text or "{}"))
    meta["loaded_from"] = str(target) if target is not None else repr(path)
    return PSDResult(
        psd=arrays["psd"],
        freq_hz=arrays["freq_hz"],
        rms=arrays["rms"],
        cross_psd=cross_psd,
        outputs=dofs["outputs"],
        inputs=dofs["inputs"],
        response=response,
        meta=meta,
    )
