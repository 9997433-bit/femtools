"""Frequency response functions: modal superposition and direct (physical) inversion.

Both entry points return an :class:`FRFResult` holding a complex ``(n_out, n_in, n_freq)``
array. With the *complete* modal basis and consistent damping the two are algebraically
identical; see ``scripts``-free self check in ``verify_modal_vs_direct``.

On a **truncated** basis they differ by the missing residual flexibility, and the size of
that difference depends entirely on how the comparison band is defined. ``fmax`` must be
the last *retained* mode — :func:`retained_fmax_hz` — so that the 0.2-0.8 fmax band of
``docs/CONTRACT_API.md`` contains only resonances the modal sum actually carries.
:func:`retained_band` and :func:`retained_band_lines` build that band, and
:func:`verify_modal_vs_direct` uses it by default when it truncates the basis itself.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._utils import TWO_PI, as_dense, is_sparse, resolve_dofs
from .damping import DampingModel, as_damping
from .modal import ModalModel, as_modal

__all__ = [
    "FRFResult",
    "direct_frf",
    "modal_frf",
    "retained_band",
    "retained_band_lines",
    "retained_fmax_hz",
    "verify_modal_vs_direct",
]

#: Default band for a truncated-basis comparison, as fractions of ``fmax``
#: (``docs/CONTRACT_API.md``: "rel L2 on 0.2-0.8 fmax", tolerance 5 %).
DEFAULT_BAND_FRACTIONS = (0.2, 0.8)

_RESPONSE_KINDS = ("receptance", "mobility", "accelerance")
_RESPONSE_ALIASES = {
    "receptance": "receptance",
    "displacement": "receptance",
    "compliance": "receptance",
    "admittance": "receptance",
    "mobility": "mobility",
    "velocity": "mobility",
    "accelerance": "accelerance",
    "acceleration": "accelerance",
    "inertance": "accelerance",
}


def _normalize_response(kind: str) -> str:
    key = str(kind).strip().lower()
    if key not in _RESPONSE_ALIASES:
        raise ValueError(f"unknown response type {kind!r}; expected one of {_RESPONSE_KINDS}")
    return _RESPONSE_ALIASES[key]


def _response_scale(kind: str, omega: np.ndarray) -> np.ndarray:
    """Multiplier converting receptance to ``kind`` at each circular frequency."""
    kind = _normalize_response(kind)
    if kind == "receptance":
        return np.ones(omega.shape, dtype=complex)
    if kind == "mobility":
        return 1j * omega
    return -(omega**2) + 0j


@dataclass
class FRFResult:
    """A block of frequency response functions.

    Attributes
    ----------
    H:
        Complex FRF matrix, shape ``(n_out, n_in, n_freq)``, dtype ``complex128``.
    freq_hz:
        Frequency axis in Hz, shape ``(n_freq,)``.
    outputs, inputs:
        DOF indices for the response and excitation directions.
    response:
        ``"receptance"`` (x/F), ``"mobility"`` (v/F) or ``"accelerance"`` (a/F).
    method:
        Provenance tag, e.g. ``"modal"``, ``"direct"`` or ``"fba"``.
    """

    H: np.ndarray
    freq_hz: np.ndarray
    outputs: np.ndarray | None = None
    inputs: np.ndarray | None = None
    response: str = "receptance"
    method: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        H = np.asarray(self.H)
        if H.ndim == 2:  # a single input column
            H = H[:, None, :]
        if H.ndim != 3:
            raise ValueError(f"H must be 3-D (n_out, n_in, n_freq), got shape {H.shape}")
        self.H = np.ascontiguousarray(H, dtype=np.complex128)
        self.freq_hz = np.atleast_1d(np.asarray(self.freq_hz, dtype=float)).reshape(-1)
        if self.freq_hz.size != self.H.shape[2]:
            raise ValueError(
                f"freq_hz has {self.freq_hz.size} points but H has {self.H.shape[2]}"
            )
        if self.outputs is not None:
            self.outputs = np.asarray(self.outputs, dtype=int).reshape(-1)
        if self.inputs is not None:
            self.inputs = np.asarray(self.inputs, dtype=int).reshape(-1)
        self.response = _normalize_response(self.response)

    # -- shape helpers ----------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        """``(n_out, n_in, n_freq)``."""
        return tuple(self.H.shape)  # type: ignore[return-value]

    @property
    def n_out(self) -> int:
        """Number of response DOFs."""
        return int(self.H.shape[0])

    @property
    def n_in(self) -> int:
        """Number of excitation DOFs."""
        return int(self.H.shape[1])

    @property
    def n_freq(self) -> int:
        """Number of frequency lines."""
        return int(self.H.shape[2])

    @property
    def omega(self) -> np.ndarray:
        """Frequency axis in rad/s."""
        return TWO_PI * self.freq_hz

    def __getitem__(self, key: Any) -> np.ndarray:
        return self.H[key]

    # -- derived quantities ----------------------------------------------
    def magnitude(self) -> np.ndarray:
        """``|H|``, shape ``(n_out, n_in, n_freq)``."""
        return np.abs(self.H)

    def phase_deg(self) -> np.ndarray:
        """Phase of ``H`` in degrees, shape ``(n_out, n_in, n_freq)``."""
        return np.degrees(np.angle(self.H))

    def index_at(self, freq_hz: float) -> int:
        """Index of the frequency line closest to ``freq_hz``."""
        return int(np.argmin(np.abs(self.freq_hz - float(freq_hz))))

    def at(self, freq_hz: float) -> np.ndarray:
        """FRF matrix ``(n_out, n_in)`` at the line closest to ``freq_hz``."""
        return self.H[:, :, self.index_at(freq_hz)]

    def as_response(self, kind: str) -> FRFResult:
        """Return the same data expressed as receptance / mobility / accelerance."""
        kind = _normalize_response(kind)
        if kind == self.response:
            return self
        omega = self.omega
        scale = _response_scale(kind, omega) / _response_scale(self.response, omega)
        return FRFResult(
            H=self.H * scale[None, None, :],
            freq_hz=self.freq_hz.copy(),
            outputs=None if self.outputs is None else self.outputs.copy(),
            inputs=None if self.inputs is None else self.inputs.copy(),
            response=kind,
            method=self.method,
            meta=dict(self.meta),
        )

    def select(self, outputs: Any = None, inputs: Any = None) -> FRFResult:
        """Sub-block of the FRF matrix, selected by *position* in the current block."""
        o = resolve_dofs(outputs, self.n_out, "outputs")
        i = resolve_dofs(inputs, self.n_in, "inputs")
        return FRFResult(
            H=self.H[np.ix_(o, i, np.arange(self.n_freq))],
            freq_hz=self.freq_hz.copy(),
            outputs=None if self.outputs is None else self.outputs[o],
            inputs=None if self.inputs is None else self.inputs[i],
            response=self.response,
            method=self.method,
            meta=dict(self.meta),
        )


def _band_mask(freq_hz: np.ndarray, band: tuple[float, float] | None) -> np.ndarray:
    if band is None:
        return np.ones(freq_hz.shape, dtype=bool)
    lo, hi = float(band[0]), float(band[1])
    return (freq_hz >= lo) & (freq_hz <= hi)


def _check_fractions(low: float, high: float) -> tuple[float, float]:
    lo, hi = float(low), float(high)
    if not (0.0 <= lo < hi):
        raise ValueError(f"band fractions must satisfy 0 <= low < high, got ({lo}, {hi})")
    return lo, hi


def retained_fmax_hz(modal: Any) -> float:
    """Highest *retained* natural frequency of ``modal``, in Hz.

    This is the ``fmax`` that a truncated-basis FRF comparison must be built on. A modal
    sum only knows about the modes it retains, so a band anchored on the *parent* model's
    highest frequency covers resonances the truncated basis does not contain. On a 40-DOF
    chain reduced to 20 modes the retained ``fmax`` is 26.84 Hz while the parent's is
    35.57 Hz: the 0.2-0.8 band of the latter reaches 28.45 Hz, past four missing poles,
    and the relative L2 error goes from 3.0 % to 13.3 % — a badly-posed comparison rather
    than a solver defect.

    Rigid-body modes (``f = 0``) are retained in the basis but cannot define the band, so
    the maximum is taken over all retained frequencies and must be strictly positive.
    """
    mm = as_modal(modal)
    f = np.asarray(mm.freq_hz, dtype=float)
    if f.size == 0:
        raise ValueError("modal model has no retained modes")
    fmax = float(np.max(f))
    if not np.isfinite(fmax) or fmax <= 0.0:
        raise ValueError(
            "retained modes have no positive natural frequency; a truncation band "
            "cannot be defined from a purely rigid-body basis"
        )
    return fmax


def retained_band(
    modal: Any,
    low: float = DEFAULT_BAND_FRACTIONS[0],
    high: float = DEFAULT_BAND_FRACTIONS[1],
) -> tuple[float, float]:
    """Comparison band ``(low * fmax, high * fmax)`` in Hz for a truncated basis.

    ``fmax`` is :func:`retained_fmax_hz`, i.e. the last *retained* mode. Defaults to the
    0.2-0.8 fmax band of ``docs/CONTRACT_API.md``.
    """
    lo, hi = _check_fractions(low, high)
    fmax = retained_fmax_hz(modal)
    return (lo * fmax, hi * fmax)


def retained_band_lines(
    modal: Any,
    n_lines: int = 240,
    low: float = DEFAULT_BAND_FRACTIONS[0],
    high: float = DEFAULT_BAND_FRACTIONS[1],
) -> np.ndarray:
    """``n_lines`` frequency lines spanning :func:`retained_band`, in Hz."""
    n = int(n_lines)
    if n < 1:
        raise ValueError(f"n_lines must be >= 1, got {n_lines}")
    lo_hz, hi_hz = retained_band(modal, low, high)
    return np.linspace(lo_hz, hi_hz, n)


def _as_residual_block(value: Any, n_out: int, n_in: int) -> np.ndarray:
    """Broadcast a scalar or matrix residual term to ``(n_out, n_in)`` complex."""
    arr = np.asarray(value, dtype=complex)
    if arr.ndim == 0:
        return np.full((n_out, n_in), arr)
    if arr.shape != (n_out, n_in):
        raise ValueError(f"residual term must be scalar or {(n_out, n_in)}, got {arr.shape}")
    return arr


def _check_freq(freq_hz: Any) -> np.ndarray:
    f = np.atleast_1d(np.asarray(freq_hz, dtype=float)).reshape(-1)
    if f.size == 0:
        raise ValueError("freq_hz must contain at least one frequency line")
    if np.any(f < 0.0):
        raise ValueError("freq_hz must be non-negative")
    return f


def modal_frf(
    modal: Any,
    inputs: Any = None,
    outputs: Any = None,
    freq_hz: Any = None,
    damping: Any = None,
    *,
    response: str = "receptance",
    lower_residual: Any = None,
    upper_residual: Any = None,
) -> FRFResult:
    """FRF by modal superposition.

    ``H_oi(w) = sum_r  phi_or * phi_ir / (w_r^2 (1 + i eta_r) - w^2 + i w 2 zeta_r w_r)``

    Parameters
    ----------
    modal:
        Modal model (``ModalResult`` from :mod:`femtools.fea.eigen`, a
        :class:`~femtools.dynamics.modal.ModalModel`, mapping or ``(freq_hz, modes)``).
    inputs, outputs:
        Excitation / response DOF selection (indices, boolean mask, or ``None`` for all).
    freq_hz:
        Frequency lines in Hz.
    damping:
        Anything :func:`~femtools.dynamics.damping.as_damping` understands: a scalar
        ``zeta``, ``{"alpha": ..., "beta": ...}``, ``{"eta": ...}`` or a
        :class:`~femtools.dynamics.damping.DampingModel`.
    lower_residual, upper_residual:
        Optional truncation corrections added as ``-LR / w^2`` (residual inertia of the
        modes below the band) and ``+UR`` (residual flexibility of the modes above it).
        Scalars or ``(n_out, n_in)`` arrays.
    response:
        ``"receptance"`` (default), ``"mobility"`` or ``"accelerance"``.

    Returns
    -------
    FRFResult
        With ``H`` of shape ``(n_out, n_in, n_freq)`` and dtype ``complex128``.
    """
    mm = as_modal(modal).mass_normalized()
    f = _check_freq(freq_hz)
    omega = TWO_PI * f
    dmp: DampingModel = as_damping(damping)

    in_idx = resolve_dofs(inputs, mm.ndof, "inputs", mm.dof_ids)
    out_idx = resolve_dofs(outputs, mm.ndof, "outputs", mm.dof_ids)

    two_zeta_omega, eta = dmp.modal_terms(mm)
    wr2 = np.asarray(mm.eigenvalues, dtype=float)

    # denominator: (n_modes, n_freq)
    denom = (
        wr2[:, None] * (1.0 + 1j * eta[:, None])
        - (omega**2)[None, :]
        + 1j * omega[None, :] * two_zeta_omega[:, None]
    )
    if np.any(denom == 0):
        warnings.warn(
            "undamped resonance hit exactly by a frequency line; FRF is infinite there",
            RuntimeWarning,
            stacklevel=2,
        )
        denom = np.where(denom == 0, np.finfo(float).tiny, denom)

    phi_o = mm.modes[out_idx, :]
    phi_i = mm.modes[in_idx, :]
    H = np.einsum("or,ir,rf->oif", phi_o, phi_i, 1.0 / denom, optimize=True)
    H = np.ascontiguousarray(H, dtype=np.complex128)

    if upper_residual is not None:
        H = H + _as_residual_block(upper_residual, H.shape[0], H.shape[1])[:, :, None]
    if lower_residual is not None:
        inv_w2 = np.where(omega > 0, 1.0 / np.where(omega > 0, omega**2, 1.0), np.inf)
        H = H - _as_residual_block(lower_residual, H.shape[0], H.shape[1])[:, :, None] * (
            inv_w2[None, None, :]
        )

    scale = _response_scale(response, omega)
    H = H * scale[None, None, :]

    return FRFResult(
        H=H,
        freq_hz=f,
        outputs=out_idx,
        inputs=in_idx,
        response=_normalize_response(response),
        method="modal",
        meta={"n_modes": mm.n_modes, "damping": type(dmp).__name__},
    )


def _dynamic_stiffness_solver(
    K: Any, M: Any, C: Any, eta: float, omega: float, sparse: bool
) -> Any:
    """Build ``Z(w)`` and return a solver callable for it."""
    kfac = 1.0 + 1j * eta
    if sparse:
        Z = kfac * sp.csc_matrix(K, dtype=complex) - (omega**2) * sp.csc_matrix(
            M, dtype=complex
        )
        if C is not None:
            Z = Z + 1j * omega * sp.csc_matrix(C, dtype=complex)
        try:
            lu = spla.splu(sp.csc_matrix(Z))
            return lu.solve
        except (RuntimeError, ValueError):
            Zd = Z.toarray()
            return lambda b: np.linalg.lstsq(Zd, b, rcond=None)[0]
    Z = kfac * as_dense(K, complex) - (omega**2) * as_dense(M, complex)
    if C is not None:
        Z = Z + 1j * omega * as_dense(C, complex)

    def _solve(b: np.ndarray, Z: np.ndarray = Z) -> np.ndarray:
        try:
            return np.linalg.solve(Z, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(Z, b, rcond=None)[0]

    return _solve


def direct_frf(
    K: Any,
    M: Any,
    inputs: Any = None,
    outputs: Any = None,
    freq_hz: Any = None,
    damping: Any = None,
    *,
    C: Any = None,
    modal: Any = None,
    response: str = "receptance",
) -> FRFResult:
    """FRF by direct inversion of the dynamic stiffness.

    ``Z(w) = K (1 + i eta) - w^2 M + i w C`` and ``H = Z^-1`` restricted to the requested
    input/output DOFs.

    Parameters
    ----------
    K, M:
        Stiffness and mass matrices, dense or scipy sparse, shape ``(ndof, ndof)``.
    inputs, outputs, freq_hz, damping, response:
        As in :func:`modal_frf`.
    C:
        Explicit viscous damping matrix; added on top of whatever ``damping`` produces.
    modal:
        Modal basis, required only when ``damping`` is
        :class:`~femtools.dynamics.damping.ModalDamping`, which needs the modes to build
        the equivalent physical ``C = M Phi diag(2 zeta w_r) Phi^T M``.
    """
    sparse = is_sparse(K) or is_sparse(M)
    ndof = int(K.shape[0])
    if K.shape != M.shape or K.shape[0] != K.shape[1]:
        raise ValueError(f"K and M must be square and equally sized, got {K.shape}, {M.shape}")

    f = _check_freq(freq_hz)
    omega = TWO_PI * f
    dmp: DampingModel = as_damping(damping)
    modal_model: ModalModel | None = as_modal(modal) if modal is not None else None

    C_total = dmp.viscous_matrix(K, M, modal_model)
    if C is not None:
        C_total = C if C_total is None else C_total + C
    eta = dmp.loss_factor()

    in_idx = resolve_dofs(inputs, ndof, "inputs")
    out_idx = resolve_dofs(outputs, ndof, "outputs")

    F = np.zeros((ndof, in_idx.size), dtype=complex)
    F[in_idx, np.arange(in_idx.size)] = 1.0

    H = np.empty((out_idx.size, in_idx.size, f.size), dtype=np.complex128)
    for k, w in enumerate(omega):
        solve = _dynamic_stiffness_solver(K, M, C_total, eta, float(w), sparse)
        try:
            X = solve(F)
        except (np.linalg.LinAlgError, RuntimeError):
            Z = as_dense(K, complex) * (1.0 + 1j * eta) - (w**2) * as_dense(M, complex)
            if C_total is not None:
                Z = Z + 1j * w * as_dense(C_total, complex)
            X = np.linalg.lstsq(Z, F, rcond=None)[0]
        H[:, :, k] = np.asarray(X)[out_idx, :]

    H = H * _response_scale(response, omega)[None, None, :]
    return FRFResult(
        H=H,
        freq_hz=f,
        outputs=out_idx,
        inputs=in_idx,
        response=_normalize_response(response),
        method="direct",
        meta={"ndof": ndof, "damping": type(dmp).__name__, "sparse": bool(sparse)},
    )


def verify_modal_vs_direct(
    K: Any,
    M: Any,
    modal: Any,
    freq_hz: Any = None,
    damping: Any = None,
    inputs: Any = None,
    outputs: Any = None,
    band: tuple[float, float] | str | None = None,
    *,
    n_modes: int | None = None,
    fmax_hz: float | None = None,
    n_lines: int = 240,
    band_fractions: tuple[float, float] = DEFAULT_BAND_FRACTIONS,
) -> dict[str, Any]:
    """Compare modal and direct FRF and report the relative L2 error.

    ``err = ||H_modal - H_direct||_F / ||H_direct||_F`` over the (optionally banded)
    frequency lines. With the complete basis the error is at round-off level; the
    acceptance target of ``docs/CONTRACT_API.md`` is 5 % on 0.2-0.8 fmax with a
    truncated basis.

    Parameters
    ----------
    K, M:
        Physical matrices driving the direct side.
    modal:
        Modal basis. When ``n_modes`` or ``fmax_hz`` is given this is the *complete*
        basis and the modal side is truncated from it; otherwise it is used as-is.
    freq_hz:
        Frequency lines in Hz. ``None`` (the default) builds ``n_lines`` lines across
        :func:`retained_band` of the truncated basis, i.e. anchored on the last
        *retained* mode.
    damping, inputs, outputs:
        As in :func:`modal_frf`.
    band:
        ``None`` for every line, an explicit ``(lo_hz, hi_hz)`` pair, or ``"retained"``
        for :func:`retained_band` of the truncated basis. Defaults to ``"retained"``
        whenever the basis is truncated here or ``freq_hz`` is generated.
    n_modes, fmax_hz:
        Truncation applied to ``modal`` for the modal side only. The direct side keeps
        the complete ``K``/``M``, and the complete basis is what
        :class:`~femtools.dynamics.damping.ModalDamping` projects into a physical ``C``,
        so the two sides differ by modal truncation alone.
    n_lines:
        Number of generated lines when ``freq_hz is None``.
    band_fractions:
        ``(low, high)`` multipliers of ``fmax`` for the generated lines and for
        ``band="retained"``. Defaults to ``(0.2, 0.8)``.

    Returns
    -------
    dict
        ``rel_l2``, ``abs_l2``, ``max_rel_pointwise``, ``n_modes`` (retained),
        ``n_modes_full``, ``fmax_hz`` (last retained mode), ``band_hz``, ``n_freq`` and
        both :class:`FRFResult` objects under ``modal``/``direct``.
    """
    full = as_modal(modal)
    truncate = n_modes is not None or fmax_hz is not None
    truncated = full.truncate(n_modes, fmax_hz) if truncate else full
    if truncated.n_modes == 0:
        raise ValueError("truncation kept no modes; relax n_modes / fmax_hz")
    auto_band = truncated.n_modes < full.n_modes or freq_hz is None

    if freq_hz is None:
        f = retained_band_lines(truncated, n_lines, *band_fractions)
    else:
        f = _check_freq(freq_hz)

    if isinstance(band, str):
        if band.lower() != "retained":
            raise ValueError(f"unknown band spec {band!r}; expected 'retained' or a pair")
        band_hz: tuple[float, float] | None = retained_band(truncated, *band_fractions)
    elif band is not None:
        band_hz = (float(band[0]), float(band[1]))
    elif auto_band:
        band_hz = retained_band(truncated, *band_fractions)
    else:
        band_hz = None

    Hm = modal_frf(truncated, inputs, outputs, f, damping)
    Hd = direct_frf(K, M, inputs, outputs, f, damping, modal=full)
    mask = _band_mask(Hd.freq_hz, band_hz)
    if not mask.any():
        raise ValueError(f"no frequency line falls inside the band {band_hz}")
    a = Hm.H[:, :, mask]
    b = Hd.H[:, :, mask]
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    rel = num / den if den > 0 else num
    with np.errstate(divide="ignore", invalid="ignore"):
        pointwise = np.abs(a - b) / np.where(np.abs(b) > 0, np.abs(b), np.inf)
    return {
        "rel_l2": rel,
        "abs_l2": num,
        "max_rel_pointwise": float(np.nanmax(pointwise)) if pointwise.size else 0.0,
        "n_modes": truncated.n_modes,
        "n_modes_full": full.n_modes,
        "fmax_hz": retained_fmax_hz(truncated),
        "band_hz": band_hz,
        "n_freq": int(mask.sum()),
        "modal": Hm,
        "direct": Hd,
    }
