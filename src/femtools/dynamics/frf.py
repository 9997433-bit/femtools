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

:func:`direct_frf` is defined on ``(K, M)`` and keeps that signature, but it also accepts a
single argument saying where the matrices come from — an assembly or a model, which is
assembled and reduced to its free partition by :func:`~femtools.dynamics.system.as_system`.
The mesh-backed forms additionally address DOFs as ``(node_id, component)``.

:func:`dump_frf` and :func:`load_frf` put a computed block of FRFs on disk as a single
``.npz`` archive, the way :func:`~femtools.dynamics.superelement.dump_cms` does for a
reduced component. ``H`` and ``freq_hz`` come back **bit-identical**.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._utils import (
    TWO_PI,
    as_dense,
    dumps_meta,
    get_field,
    json_meta,
    npz_path,
    npz_text,
    resolve_dofs,
)
from .damping import DampingModel, as_damping
from .modal import ModalModel, as_modal
from .system import SystemMatrices, as_system, resolve_selection

__all__ = [
    "FRFResult",
    "direct_frf",
    "dump_frf",
    "load_frf",
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
        current = _response_scale(self.response, omega)
        # Mobility and accelerance are identically zero at f = 0 whatever the structure
        # does statically, so dividing them back by i*w / -w^2 there is 0/0. It used to
        # come out as a nan line announced by nothing but a numpy RuntimeWarning.
        if np.any(current == 0.0):
            raise ValueError(
                f"converting {self.response} to {kind} divides by omega, and this FRF "
                "has a line at f = 0 where the source is identically zero and carries "
                "nothing about the static response; drop the DC line before converting"
            )
        scale = _response_scale(kind, omega) / current
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
        block = _as_residual_block(lower_residual, H.shape[0], H.shape[1])
        above_dc = omega > 0
        # -LR / w^2 is unbounded at DC. Saying so beats the 0 * inf = nan that silently
        # poisoned the line before; a zero residual is still the no-op it should be.
        if not above_dc.all() and np.any(block != 0):
            raise ValueError(
                "lower_residual is a residual inertia term -LR/w^2, which is singular at "
                "f = 0; drop the DC line from freq_hz or pass lower_residual=0 there"
            )
        inv_w2 = np.zeros(omega.shape)
        inv_w2[above_dc] = 1.0 / omega[above_dc] ** 2
        H = H - block[:, :, None] * inv_w2[None, None, :]

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


def _damping_matrix(value: Any, ndof: int, name: str) -> Any:
    """Check that a viscous damping term really is an ``(ndof, ndof)`` matrix.

    ``Z(w)`` is built as ``... + 1j w C``, and numpy broadcasts a length-``ndof`` vector
    across the *rows* of that sum rather than onto its diagonal. A caller passing the
    diagonal of ``C`` therefore used to get a fully populated, plausible-looking and
    quietly wrong dynamic stiffness.
    """
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        value = np.asarray(value, dtype=float)
        shape = value.shape
    if tuple(shape) != (ndof, ndof):
        hint = (
            "; a 1-D array is not read as a diagonal, it would broadcast across every "
            "row of Z(w)"
            if len(shape) == 1
            else ""
        )
        raise ValueError(
            f"{name} must be a square ({ndof}, {ndof}) viscous damping matrix, got "
            f"shape {tuple(shape)}{hint}"
        )
    return value


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
    M: Any = None,
    inputs: Any = None,
    outputs: Any = None,
    freq_hz: Any = None,
    damping: Any = None,
    *,
    C: Any = None,
    modal: Any = None,
    response: str = "receptance",
    assemble: Mapping[str, Any] | None = None,
) -> FRFResult:
    """FRF by direct inversion of the dynamic stiffness.

    ``Z(w) = K (1 + i eta) - w^2 M + i w C`` and ``H = Z^-1`` restricted to the requested
    input/output DOFs.

    Parameters
    ----------
    K, M:
        Stiffness and mass matrices, dense or scipy sparse, shape ``(ndof, ndof)``. ``K``
        may instead be a *single* argument describing where those matrices come from — an
        ``AssemblyResult``, a ``ModalResult`` carrying one, or a model database, which is
        assembled here (see :func:`~femtools.dynamics.system.as_system`). In that case the
        system solved is the free-free partition ``Kff``/``Mff``, any damping assembled
        with the model is included, and ``inputs``/``outputs`` additionally accept
        ``(node_id, component)`` pairs.
    inputs, outputs, freq_hz, damping, response:
        As in :func:`modal_frf`.
    C:
        Explicit viscous damping matrix; added on top of whatever ``damping`` produces.
    modal:
        Modal basis, required only when ``damping`` is
        :class:`~femtools.dynamics.damping.ModalDamping`, which needs the modes to build
        the equivalent physical ``C = M Phi diag(2 zeta w_r) Phi^T M``. With a model or an
        assembly this may be the string ``"auto"``, which solves the complete basis here.
    assemble:
        Keyword arguments for :func:`femtools.fea.assemble.assemble_km`, used only when a
        model is assembled here.

    Examples
    --------
    The two calls below solve the same system; the second needs no knowledge of how the
    free partition is numbered::

        direct_frf(assembly.Kff, assembly.Mff, inputs=[37], freq_hz=f, damping=0.01)
        direct_frf(model, inputs=[(17, "uz")], freq_hz=f, damping=0.01, modal="auto")
    """
    system: SystemMatrices = as_system(K, M, assemble=assemble)
    K, M = system.K, system.M
    sparse = system.sparse
    ndof = system.ndof

    f = _check_freq(freq_hz)
    omega = TWO_PI * f
    dmp: DampingModel = as_damping(damping)
    modal_model: ModalModel | None = system.modal_basis(modal)

    try:
        C_total = dmp.viscous_matrix(K, M, modal_model)
    except ValueError as exc:
        if modal_model is None and system.can_solve_modal:
            raise ValueError(
                f"{exc}. This system came from a model, so modal='auto' would solve the "
                "complete basis here"
            ) from exc
        raise
    C_total = _damping_matrix(C_total, ndof, f"the C matrix of {type(dmp).__name__}")
    if system.C is not None:
        C_total = system.C if C_total is None else C_total + system.C
    if C is not None:
        C = _damping_matrix(C, ndof, "C")
        C_total = C if C_total is None else C_total + C
    eta = dmp.loss_factor()

    in_idx = system.resolve(inputs, "inputs")
    out_idx = system.resolve(outputs, "outputs")

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
    meta = system.meta()
    meta["damping"] = type(dmp).__name__
    labels_in = system.labels(in_idx)
    if labels_in is not None:
        meta["input_dofs"] = system.global_dofs(in_idx)
        meta["output_dofs"] = system.global_dofs(out_idx)
        meta["input_labels"] = labels_in
        meta["output_labels"] = system.labels(out_idx)
    return FRFResult(
        H=H,
        freq_hz=f,
        outputs=out_idx,
        inputs=in_idx,
        response=_normalize_response(response),
        method="direct",
        meta=meta,
    )


def verify_modal_vs_direct(
    K: Any,
    M: Any = None,
    modal: Any = None,
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
    full_modes: int | None = None,
    assemble: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare modal and direct FRF and report the relative L2 error.

    ``err = ||H_modal - H_direct||_F / ||H_direct||_F`` over the (optionally banded)
    frequency lines. With the complete basis the error is at round-off level; the
    acceptance target of ``docs/CONTRACT_API.md`` is 5 % on 0.2-0.8 fmax with a
    truncated basis.

    Parameters
    ----------
    K, M:
        Physical matrices driving the direct side, or a single assembly / model argument
        as in :func:`direct_frf`.
    modal:
        Modal basis. When ``n_modes`` or ``fmax_hz`` is given this is the *complete*
        basis and the modal side is truncated from it; otherwise it is used as-is. With
        an assembly or a model it may be left out entirely, in which case the complete
        basis is solved here — ``verify_modal_vs_direct(model, n_modes=20)`` is then the
        whole acceptance case.
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
    full_modes:
        Size of the basis solved here when ``modal`` is omitted. The default is the
        complete basis, because that is what makes a
        :class:`~femtools.dynamics.damping.ModalDamping` ``C`` — and hence the comparison
        itself — exact. Damping that has a physical form of its own (Rayleigh, structural,
        explicit ``C``) does not need it, so on a large model pass something just above
        ``n_modes`` instead of paying for the full spectrum.
    assemble:
        Keyword arguments for :func:`femtools.fea.assemble.assemble_km`, used only when a
        model is assembled here.

    Returns
    -------
    dict
        ``rel_l2``, ``abs_l2``, ``max_rel_pointwise``, ``n_modes`` (retained),
        ``n_modes_full``, ``fmax_hz`` (last retained mode), ``band_hz``, ``n_freq``,
        ``system`` and both :class:`FRFResult` objects under ``modal``/``direct``.
    """
    system = as_system(K, M, assemble=assemble)
    if modal is None:
        if not system.can_solve_modal:
            raise ValueError(
                "a modal basis is required; it can only be solved here when the first "
                "argument is an assembly or a model"
            )
        full = system.solve_modal(full_modes)
    else:
        if full_modes is not None:
            raise ValueError("full_modes only applies when the basis is solved here")
        full = system.align_modal(modal)
    inputs = resolve_selection(system, inputs, "inputs")
    outputs = resolve_selection(system, outputs, "outputs")
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
    Hd = direct_frf(system, None, inputs, outputs, f, damping, modal=full)
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
        "system": system,
        "modal": Hm,
        "direct": Hd,
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

#: Tag written into every archive; :func:`load_frf` refuses anything else.
FRF_FORMAT = "femtools.dynamics.frf/1"

#: Fields an archive cannot be missing — everything else has a defined default.
_REQUIRED = ("H", "freq_hz")
#: Written only when the source carries them; absent in the archive means ``None``.
_OPTIONAL_DOFS = ("outputs", "inputs")


def _as_frf(result: Any) -> FRFResult:
    """Coerce a duck-typed FRF carrier to an :class:`FRFResult` for storage."""
    if isinstance(result, FRFResult):
        return result
    H = get_field(result, "H")
    freq_hz = get_field(result, "freq_hz")
    if H is None or freq_hz is None:
        absent = "H" if H is None else "freq_hz"
        raise TypeError(
            f"{type(result).__name__} has no {absent!r}; a block of frequency response "
            "functions must carry the complex matrix H and the frequency axis freq_hz "
            "to be worth storing"
        )
    response = get_field(result, "response")
    return FRFResult(
        H=H,
        freq_hz=freq_hz,
        outputs=get_field(result, "outputs"),
        inputs=get_field(result, "inputs"),
        response=str(response) if response else "receptance",
        method=str(get_field(result, "method") or ""),
        meta=dict(get_field(result, "meta") or {}),
    )


def dump_frf(result: Any, path: Any, *, compress: bool = False, meta: Any = None) -> Any:
    """Write a block of frequency response functions to an ``.npz`` archive.

    An FRF is expensive in a way that is easy to forget: a direct solve factorises the
    dynamic stiffness once per frequency line, so a few hundred lines of a mesh-backed
    model cost more than the eigen solve that a modal FRF is built on. Whatever comes
    next — a curve fit, an FRF-based assembly, a plot in a report, a comparison against a
    measurement taken a month later — should not have to re-solve it, and that is all this
    function is for. It is the :func:`~femtools.dynamics.superelement.dump_cms` of the
    forced-response side and writes the same kind of archive: one array per field, a
    ``format`` tag, and ``meta`` as JSON.

    ``H`` is stored as raw ``complex128`` and ``freq_hz`` as raw ``float64``, so both come
    back **bit-identical**. That matters more here than the size of the file: an FRF that
    moved in its last bits between the run that computed it and the run that consumes it
    is an FRF whose resonances, damping estimates and mode shapes cannot be compared with
    anyone else's.

        dump_frf(modal_frf(modes, [7], [7], f, 0.02), "drive_point.npz")
        H = load_frf("drive_point.npz")      # an FRFResult again
        H.magnitude()[0, 0].argmax()

    Parameters
    ----------
    result:
        An :class:`FRFResult`, or any object or mapping exposing at least ``H`` and
        ``freq_hz`` — ``outputs``, ``inputs``, ``response``, ``method`` and ``meta`` are
        stored when present. A duck-typed source goes through :class:`FRFResult`'s own
        validation first, so a 2-D ``H`` is read as the single-input block it is.
    path:
        Destination. A ``str`` or path-like without an ``.npz`` suffix gets one, and the
        resolved :class:`~pathlib.Path` is returned; an open binary file object is written
        to as-is and returned unchanged.
    compress:
        Use ``np.savez_compressed``. An FRF is dense complex data that rarely compresses
        as well as a reduction basis does, so this is off by default; the bits that come
        back are identical either way.
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
        If ``result`` carries no ``H`` and ``freq_hz``.
    """
    frf = _as_frf(result)

    payload: dict[str, Any] = {
        "H": frf.H,
        "freq_hz": frf.freq_hz,
        "format": np.array(FRF_FORMAT),
        "response": np.array(frf.response),
        "method": np.array(frf.method),
        "source_class": np.array(type(result).__name__),
        "meta_json": np.array(dumps_meta(json_meta(frf, meta))),
    }
    for name in _OPTIONAL_DOFS:
        value = getattr(frf, name)
        # Absent from the archive is how "the block was never restricted to a DOF
        # selection" is written down; an empty array would say the opposite.
        if value is not None:
            payload[name] = np.asarray(value, dtype=np.int64).reshape(-1)

    target = npz_path(path)
    save = np.savez_compressed if compress else np.savez
    save(target if target is not None else path, **payload)
    return target if target is not None else path


def load_frf(path: Any) -> FRFResult:
    """Read a block of FRFs back from an ``.npz`` archive written by :func:`dump_frf`.

    ``H`` and ``freq_hz`` are bit-identical to what was written, and ``outputs`` /
    ``inputs`` come back as the DOF selections they were, or as ``None`` when the block
    was never restricted to one. ``response`` is restored, so a stored accelerance is
    still an accelerance and :meth:`FRFResult.as_response` converts from the right place.
    ``meta`` round-trips through JSON, so its tuples arrive as lists, and gains a
    ``loaded_from`` entry.

    Parameters
    ----------
    path:
        Source archive: a path, a path without its ``.npz`` suffix, or an open binary
        file object.

    Returns
    -------
    FRFResult

    Raises
    ------
    ValueError
        If the archive was not written by :func:`dump_frf`, or has lost ``H`` or
        ``freq_hz``.
    """
    target = npz_path(path)
    with np.load(target if target is not None else path, allow_pickle=False) as data:
        tag = npz_text(data, "format")
        if tag != FRF_FORMAT:
            raise ValueError(
                f"{path!r} is not a femtools FRF archive (format tag "
                f"{tag or 'absent'!r}, expected {FRF_FORMAT!r})"
            )
        missing = [name for name in _REQUIRED if name not in data.files]
        if missing:
            raise ValueError(
                f"{path!r} claims to be an FRF archive but is missing {', '.join(missing)}"
            )
        H = np.array(data["H"])
        freq_hz = np.array(data["freq_hz"])
        dofs = {
            name: (np.array(data[name]) if name in data.files else None)
            for name in _OPTIONAL_DOFS
        }
        response = npz_text(data, "response", "receptance")
        method = npz_text(data, "method")
        meta_text = npz_text(data, "meta_json", "{}")

    meta = dict(json.loads(meta_text or "{}"))
    meta["loaded_from"] = str(target) if target is not None else repr(path)
    return FRFResult(
        H=H,
        freq_hz=freq_hz,
        outputs=dofs["outputs"],
        inputs=dofs["inputs"],
        response=response,
        method=method,
        meta=meta,
    )
