"""Harmonic (steady-state) forced response and operating deflection shapes.

Given a harmonic load ``f(t) = Re{F e^{i w t}}`` the steady-state response is
``x(t) = Re{X(w) e^{i w t}}``; the complex vector ``X(w)`` *is* the operating deflection
shape (ODS) at that frequency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._utils import TWO_PI, is_sparse
from .damping import DampingModel, as_damping
from .frf import _check_freq, _dynamic_stiffness_solver, _normalize_response
from .modal import ModalModel, as_modal

__all__ = ["HarmonicResult", "harmonic_response"]


@dataclass
class HarmonicResult:
    """Steady-state harmonic response over a frequency axis.

    Attributes
    ----------
    freq_hz:
        Frequency lines, shape ``(n_freq,)``.
    displacement:
        Complex displacement amplitudes, shape ``(ndof, n_freq)``.
    load:
        Complex load amplitudes actually applied, shape ``(ndof, n_freq)``.
    method:
        ``"modal"`` or ``"direct"``.
    """

    freq_hz: np.ndarray
    displacement: np.ndarray
    load: np.ndarray
    method: str = ""
    modal_coordinates: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.freq_hz = np.atleast_1d(np.asarray(self.freq_hz, dtype=float)).reshape(-1)
        self.displacement = np.ascontiguousarray(self.displacement, dtype=np.complex128)
        self.load = np.ascontiguousarray(self.load, dtype=np.complex128)
        if self.displacement.shape[1] != self.freq_hz.size:
            raise ValueError("displacement must have one column per frequency line")

    @property
    def ndof(self) -> int:
        """Number of physical DOFs."""
        return int(self.displacement.shape[0])

    @property
    def n_freq(self) -> int:
        """Number of frequency lines."""
        return int(self.freq_hz.size)

    @property
    def omega(self) -> np.ndarray:
        """Frequency axis in rad/s."""
        return TWO_PI * self.freq_hz

    @property
    def velocity(self) -> np.ndarray:
        """``i w X``, shape ``(ndof, n_freq)``."""
        return 1j * self.omega[None, :] * self.displacement

    @property
    def acceleration(self) -> np.ndarray:
        """``-w^2 X``, shape ``(ndof, n_freq)``."""
        return -(self.omega**2)[None, :] * self.displacement

    def index_at(self, freq_hz: float) -> int:
        """Index of the frequency line closest to ``freq_hz``."""
        return int(np.argmin(np.abs(self.freq_hz - float(freq_hz))))

    def ods(
        self, freq_hz: float, normalize: bool = False, real: bool = False
    ) -> np.ndarray:
        """Operating deflection shape at the line closest to ``freq_hz``.

        With ``normalize`` the shape is scaled to unit largest magnitude and rotated so
        that its dominant component is real-positive; with ``real`` only the real part of
        that rotated shape is returned (the classical "in-phase" ODS animation frame).
        """
        x = self.displacement[:, self.index_at(freq_hz)].copy()
        if normalize or real:
            k = int(np.argmax(np.abs(x)))
            if np.abs(x[k]) > 0:
                x = x * np.exp(-1j * np.angle(x[k]))
                if normalize:
                    x = x / np.abs(x[k])
        return x.real.copy() if real else x

    def amplitude(self) -> np.ndarray:
        """``|X|``, shape ``(ndof, n_freq)``."""
        return np.abs(self.displacement)

    def phase_deg(self) -> np.ndarray:
        """Phase of ``X`` in degrees, shape ``(ndof, n_freq)``."""
        return np.degrees(np.angle(self.displacement))

    def rms_curve(self, dofs: Any = None) -> np.ndarray:
        """Euclidean norm of the ODS at every frequency line, shape ``(n_freq,)``."""
        x = self.displacement if dofs is None else self.displacement[np.asarray(dofs), :]
        return np.linalg.norm(x, axis=0)

    def peak_frequencies(self, dofs: Any = None) -> np.ndarray:
        """Frequencies of the local maxima of :meth:`rms_curve`."""
        y = self.rms_curve(dofs)
        if y.size < 3:
            return np.empty(0)
        interior = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])) + 1
        return self.freq_hz[interior]


def _build_load(load: Any, ndof: int, n_freq: int, omega: np.ndarray) -> np.ndarray:
    """Normalise a load specification to a complex ``(ndof, n_freq)`` array."""
    if load is None:
        raise ValueError("a load specification is required")
    if callable(load) and not isinstance(load, np.ndarray):
        cols = [np.asarray(load(w), dtype=complex).reshape(-1) for w in omega]
        F = np.stack(cols, axis=1)
    elif isinstance(load, Mapping):
        F = np.zeros((ndof, n_freq), dtype=complex)
        for dof, value in load.items():
            v = np.asarray(value, dtype=complex).reshape(-1)
            if v.size == 1:
                F[int(dof), :] += v[0]
            elif v.size == n_freq:
                F[int(dof), :] += v
            else:
                raise ValueError(
                    f"load for DOF {dof} must be scalar or length {n_freq}, got {v.size}"
                )
        return F
    else:
        arr = np.asarray(load, dtype=complex)
        if arr.ndim == 1:
            if arr.size != ndof:
                raise ValueError(f"load vector must have length {ndof}, got {arr.size}")
            F = np.repeat(arr[:, None], n_freq, axis=1)
        elif arr.ndim == 2:
            F = arr
        else:
            raise ValueError(f"load must be 1-D or 2-D, got shape {arr.shape}")

    if F.shape[0] != ndof:
        raise ValueError(f"load has {F.shape[0]} rows but the model has {ndof} DOFs")
    if F.shape[1] == 1 and n_freq != 1:
        F = np.repeat(F, n_freq, axis=1)
    if F.shape[1] != n_freq:
        raise ValueError(f"load has {F.shape[1]} columns but there are {n_freq} lines")
    return np.ascontiguousarray(F, dtype=np.complex128)


def harmonic_response(
    K: Any = None,
    M: Any = None,
    load: Any = None,
    freq_hz: Any = None,
    damping: Any = None,
    *,
    C: Any = None,
    modal: Any = None,
    method: str = "auto",
    response: str = "receptance",
) -> HarmonicResult:
    """Steady-state response to a harmonic load; the result carries the ODS.

    Parameters
    ----------
    K, M:
        Stiffness / mass matrices for the direct method (optional if ``modal`` is given).
    load:
        Complex load amplitudes. One of: a ``(ndof,)`` vector (same at every line), an
        ``(ndof, n_freq)`` array, a ``{dof: amplitude_or_series}`` mapping, or a callable
        ``omega -> vector``.
    freq_hz:
        Frequency lines in Hz.
    damping:
        Anything :func:`~femtools.dynamics.damping.as_damping` accepts.
    C:
        Extra explicit viscous damping matrix (direct method only).
    modal:
        Modal model; enables (and is required by) ``method="modal"``.
    method:
        ``"modal"``, ``"direct"``, or ``"auto"`` (direct when ``K``/``M`` are available).
    response:
        Kinematic quantity stored in :attr:`HarmonicResult.displacement`:
        ``"receptance"``/``"displacement"`` (default), ``"mobility"``/``"velocity"`` or
        ``"accelerance"``/``"acceleration"``.

    Returns
    -------
    HarmonicResult
    """
    kind = _normalize_response(response)
    f = _check_freq(freq_hz)
    omega = TWO_PI * f
    dmp: DampingModel = as_damping(damping)
    modal_model: ModalModel | None = as_modal(modal) if modal is not None else None

    if method == "auto":
        method = "direct" if (K is not None and M is not None) else "modal"
    if method not in ("modal", "direct"):
        raise ValueError(f"unknown method {method!r}; expected 'modal', 'direct' or 'auto'")

    if method == "modal":
        if modal_model is None:
            raise ValueError("method='modal' requires a modal model")
        mm = modal_model.mass_normalized()
        ndof = mm.ndof
        F = _build_load(load, ndof, f.size, omega)
        two_zeta_omega, eta = dmp.modal_terms(mm)
        wr2 = np.asarray(mm.eigenvalues, dtype=float)
        denom = (
            wr2[:, None] * (1.0 + 1j * eta[:, None])
            - (omega**2)[None, :]
            + 1j * omega[None, :] * two_zeta_omega[:, None]
        )
        denom = np.where(denom == 0, np.finfo(float).tiny, denom)
        q = (mm.modes.T @ F) / denom
        X = mm.modes @ q
        meta: dict[str, Any] = {"n_modes": mm.n_modes}
        modal_coords: np.ndarray | None = q
    else:
        if K is None or M is None:
            raise ValueError("method='direct' requires K and M")
        ndof = int(K.shape[0])
        F = _build_load(load, ndof, f.size, omega)
        sparse = is_sparse(K) or is_sparse(M)
        C_total = dmp.viscous_matrix(K, M, modal_model)
        if C is not None:
            C_total = C if C_total is None else C_total + C
        eta_phys = dmp.loss_factor()
        X = np.empty((ndof, f.size), dtype=np.complex128)
        for k, w in enumerate(omega):
            solve = _dynamic_stiffness_solver(K, M, C_total, eta_phys, float(w), sparse)
            X[:, k] = np.asarray(solve(F[:, k])).reshape(-1)
        meta = {"ndof": ndof, "sparse": bool(sparse)}
        modal_coords = None

    if kind == "mobility":
        X = X * (1j * omega)[None, :]
    elif kind == "accelerance":
        X = X * (-(omega**2))[None, :]

    meta["response"] = kind
    meta["damping"] = type(dmp).__name__
    return HarmonicResult(
        freq_hz=f,
        displacement=X,
        load=F,
        method=method,
        modal_coordinates=modal_coords,
        meta=meta,
    )