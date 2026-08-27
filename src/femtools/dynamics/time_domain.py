"""Transient response by modal superposition.

Each retained mode is an uncoupled SDOF oscillator

    q_r'' + 2 zeta_r w_r q_r' + w_r^2 q_r = phi_r^T f(t)

integrated with the exact recurrence for a load that varies linearly inside a step
(Nigam-Jennings). That recurrence is unconditionally stable and exact for piecewise-linear
excitation, so ``dt`` only has to resolve the load and the highest retained mode. A
Newmark average-acceleration variant is available for comparison.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._utils import as_dense, resolve_dofs
from .damping import DampingModel, as_damping
from .modal import ModalModel, as_modal

__all__ = ["TimeHistoryResult", "time_history"]


@dataclass
class TimeHistoryResult:
    """Transient response of a modal model.

    Attributes
    ----------
    t:
        Time axis, shape ``(n_steps,)``.
    displacement, velocity, acceleration:
        Physical response at the requested output DOFs, shape ``(n_out, n_steps)``.
    modal_displacement, modal_velocity, modal_acceleration:
        Modal coordinates, shape ``(n_modes, n_steps)``.
    """

    t: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    modal_displacement: np.ndarray
    modal_velocity: np.ndarray
    modal_acceleration: np.ndarray
    outputs: np.ndarray | None = None
    method: str = "exact"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> float:
        """Time step."""
        return float(self.t[1] - self.t[0]) if self.t.size > 1 else 0.0

    @property
    def n_steps(self) -> int:
        """Number of time samples."""
        return int(self.t.size)

    def index_at(self, time: float) -> int:
        """Index of the sample closest to ``time``."""
        return int(np.argmin(np.abs(self.t - float(time))))

    def at(self, time: float) -> np.ndarray:
        """Displacement vector at the sample closest to ``time``."""
        return self.displacement[:, self.index_at(time)]

    def peak(self) -> np.ndarray:
        """Peak absolute displacement per output DOF, shape ``(n_out,)``."""
        return np.max(np.abs(self.displacement), axis=1)

    def rms(self) -> np.ndarray:
        """RMS displacement per output DOF, shape ``(n_out,)``."""
        return np.sqrt(np.mean(self.displacement**2, axis=1))


def _build_force(
    force: Any,
    ndof: int,
    t: np.ndarray | None,
    dt: float | None,
    n_steps: int | None,
    force_dofs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(F, t)`` with ``F`` shaped ``(ndof, n_steps)``."""
    if t is not None:
        t = np.asarray(t, dtype=float).reshape(-1)
        if t.size < 2:
            raise ValueError("t must contain at least two samples")
        steps = np.diff(t)
        if not np.allclose(steps, steps[0], rtol=1e-9, atol=0.0):
            raise ValueError("time axis must be uniformly spaced")
        dt = float(steps[0])
        n_steps = int(t.size)

    if callable(force) and not isinstance(force, np.ndarray):
        if t is None:
            if dt is None or n_steps is None:
                raise ValueError("a callable force needs either t, or dt and n_steps")
            t = dt * np.arange(int(n_steps))
        F = np.stack([np.asarray(force(float(ti)), dtype=float).reshape(-1) for ti in t], axis=1)
    elif isinstance(force, Mapping):
        series = {int(k): np.asarray(v, dtype=float).reshape(-1) for k, v in force.items()}
        lengths = {v.size for v in series.values()}
        if len(lengths) != 1:
            raise ValueError("all force series must have the same length")
        n_steps = lengths.pop()
        F = np.zeros((ndof, n_steps))
        for dof, v in series.items():
            F[dof, :] += v
    else:
        arr = np.asarray(force, dtype=float)
        if arr.ndim == 1:
            if force_dofs is not None or ndof == 1:
                arr = arr.reshape(1, -1)
            else:
                raise ValueError(
                    "a 1-D force series is ambiguous for a multi-DOF model; pass "
                    "force_dofs, a (ndof, n_steps) array, or a {dof: series} mapping"
                )
        if force_dofs is not None:
            idx = resolve_dofs(force_dofs, ndof, "force_dofs")
            if arr.shape[0] != idx.size:
                raise ValueError(
                    f"force has {arr.shape[0]} rows but {idx.size} force_dofs were given"
                )
            F = np.zeros((ndof, arr.shape[1]))
            F[idx, :] = arr
        else:
            if arr.shape[0] != ndof:
                raise ValueError(f"force must have {ndof} rows, got {arr.shape[0]}")
            F = arr
        n_steps = F.shape[1]

    if t is None:
        if dt is None:
            raise ValueError("dt is required when no time axis is given")
        t = float(dt) * np.arange(F.shape[1])
    if F.shape[1] != t.size:
        raise ValueError(f"force has {F.shape[1]} samples but the time axis has {t.size}")
    return np.ascontiguousarray(F, dtype=float), t


def _nigam_jennings(
    omega: np.ndarray, zeta: np.ndarray, dt: float
) -> tuple[np.ndarray, ...]:
    """Exact piecewise-linear recurrence coefficients ``(A, B, C, D, Ap, Bp, Cp, Dp)``.

    Evaluated in complex arithmetic so that over-damped modes (``zeta > 1``) work too; the
    imaginary parts cancel and only the real part is kept.
    """
    w = np.asarray(omega, dtype=float)
    z = np.asarray(zeta, dtype=float)
    # Avoid the removable singularity at exactly critical damping.
    z = np.where(np.abs(1.0 - z**2) < 1e-12, z * (1.0 - 1e-7) - 1e-9, z)

    zc = z.astype(complex)
    root = np.sqrt(1.0 - zc**2)
    wd = w * root
    e = np.exp(-zc * w * dt)
    s = np.sin(wd * dt)
    c = np.cos(wd * dt)
    w2 = (w**2).astype(complex)

    A = e * (zc / root * s + c)
    B = e * (s / wd)
    C = (1.0 / w2) * (
        2.0 * zc / (w * dt)
        + e * (((1.0 - 2.0 * zc**2) / (wd * dt) - zc / root) * s - (1.0 + 2.0 * zc / (w * dt)) * c)
    )
    D = (1.0 / w2) * (
        1.0 - 2.0 * zc / (w * dt)
        + e * ((2.0 * zc**2 - 1.0) / (wd * dt) * s + 2.0 * zc / (w * dt) * c)
    )
    Ap = -e * (w / root * s)
    Bp = e * (c - zc / root * s)
    Cp = (1.0 / w2) * (
        -1.0 / dt + e * ((w / root + zc / (root * dt)) * s + c / dt)
    )
    Dp = (1.0 / (w2 * dt)) * (1.0 - e * (zc / root * s + c))
    return tuple(np.real(x) for x in (A, B, C, D, Ap, Bp, Cp, Dp))


def _rigid_coefficients(c: np.ndarray, dt: float) -> tuple[np.ndarray, ...]:
    """Exact coefficients for a zero-frequency mode ``q'' + c q' = p``.

    Returns ``(E, phi1, phi2, phi3)`` such that

    ``v1 = v0 E + dt (p0 phi1 + dp phi2)`` and
    ``q1 = q0 + dt v0 phi1 + dt^2 (p0 phi2 + dp phi3)``

    with ``dp = p1 - p0``. Small ``c dt`` uses the series expansion to avoid cancellation.
    """
    x = np.asarray(c, dtype=float) * dt
    E = np.exp(-x)
    small = x < 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = np.where(small, 1.0, x)
        phi1 = np.where(small, 1.0 - x / 2 + x**2 / 6 - x**3 / 24, (1.0 - E) / xs)
        phi2 = np.where(small, 0.5 - x / 6 + x**2 / 24, (1.0 - phi1) / xs)
        phi3 = np.where(small, 1.0 / 6 - x / 24 + x**2 / 120, (0.5 - phi2) / xs)
    return E, phi1, phi2, phi3


def _integrate_exact(
    omega: np.ndarray, c: np.ndarray, P: np.ndarray, dt: float, q0: np.ndarray,
    qd0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact piecewise-linear recurrence for all modes at once. ``P`` is ``(n_modes, n_t)``."""
    n_modes, n_steps = P.shape
    q = np.zeros((n_modes, n_steps))
    qd = np.zeros((n_modes, n_steps))
    q[:, 0] = q0
    qd[:, 0] = qd0

    rigid = omega <= 0.0
    flex = ~rigid
    has_flex = bool(np.any(flex))
    has_rigid = bool(np.any(rigid))
    if has_flex:
        zeta_f = c[flex] / (2.0 * omega[flex])
        A, B, C, D, Ap, Bp, Cp, Dp = _nigam_jennings(omega[flex], zeta_f, dt)
    if has_rigid:
        E, p1c, p2c, p3c = _rigid_coefficients(c[rigid], dt)

    for k in range(n_steps - 1):
        if has_flex:
            qk, qdk = q[flex, k], qd[flex, k]
            pk, pk1 = P[flex, k], P[flex, k + 1]
            q[flex, k + 1] = A * qk + B * qdk + C * pk + D * pk1
            qd[flex, k + 1] = Ap * qk + Bp * qdk + Cp * pk + Dp * pk1
        if has_rigid:
            qk, qdk = q[rigid, k], qd[rigid, k]
            pk = P[rigid, k]
            dp = P[rigid, k + 1] - pk
            q[rigid, k + 1] = qk + dt * qdk * p1c + dt**2 * (pk * p2c + dp * p3c)
            qd[rigid, k + 1] = qdk * E + dt * (pk * p1c + dp * p2c)
    return q, qd


def _integrate_newmark(
    omega: np.ndarray, c: np.ndarray, P: np.ndarray, dt: float, q0: np.ndarray,
    qd0: np.ndarray, beta: float = 0.25, gamma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Newmark integration of the uncoupled modal equations (default: average acceleration)."""
    n_modes, n_steps = P.shape
    k = omega**2
    q = np.zeros((n_modes, n_steps))
    qd = np.zeros((n_modes, n_steps))
    qdd = np.zeros((n_modes, n_steps))
    q[:, 0] = q0
    qd[:, 0] = qd0
    qdd[:, 0] = P[:, 0] - c * qd0 - k * q0

    keff = 1.0 / (beta * dt**2) + gamma * c / (beta * dt) + k
    a1 = 1.0 / (beta * dt**2)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    b1 = gamma / (beta * dt)
    b2 = gamma / beta - 1.0
    b3 = dt * (gamma / (2.0 * beta) - 1.0)
    for n in range(n_steps - 1):
        rhs = (
            P[:, n + 1]
            + a1 * q[:, n] + a2 * qd[:, n] + a3 * qdd[:, n]
            + c * (b1 * q[:, n] + b2 * qd[:, n] + b3 * qdd[:, n])
        )
        q[:, n + 1] = rhs / keff
        dq = q[:, n + 1] - q[:, n]
        qdd[:, n + 1] = a1 * dq - a2 * qd[:, n] - a3 * qdd[:, n]
        qd[:, n + 1] = qd[:, n] + dt * ((1.0 - gamma) * qdd[:, n] + gamma * qdd[:, n + 1])
    return q, qd


def time_history(
    modal: Any,
    force: Any = None,
    dt: float | None = None,
    damping: Any = None,
    *,
    t: Any = None,
    n_steps: int | None = None,
    force_dofs: Any = None,
    outputs: Any = None,
    q0: Any = None,
    qd0: Any = None,
    x0: Any = None,
    v0: Any = None,
    M: Any = None,
    n_modes: int | None = None,
    method: str = "exact",
) -> TimeHistoryResult:
    """Transient response by modal superposition.

    Parameters
    ----------
    modal:
        Modal model (mass-normalised internally).
    force:
        Physical load history: an ``(ndof, n_steps)`` array, an ``(n_force, n_steps)`` array
        together with ``force_dofs``, a ``{dof: series}`` mapping, or a callable
        ``t -> vector``.
    dt:
        Time step (ignored when ``t`` is given).
    damping:
        Anything :func:`~femtools.dynamics.damping.as_damping` accepts. Structural damping
        has no causal time-domain form and is converted to ``zeta = eta / 2`` with a warning.
    t:
        Explicit uniform time axis.
    n_steps:
        Number of samples, needed only for a callable force without ``t``.
    outputs:
        Physical DOFs to report; defaults to all.
    q0, qd0:
        Initial modal displacement / velocity.
    x0, v0:
        Initial physical displacement / velocity, projected onto the modal basis using
        ``M`` when supplied (``q0 = Phi^T M x0``) and by least squares otherwise.
    n_modes:
        Truncate the basis to the lowest ``n_modes`` modes.
    method:
        ``"exact"`` (Nigam-Jennings piecewise-linear recurrence) or ``"newmark"``.

    Returns
    -------
    TimeHistoryResult
    """
    mm: ModalModel = as_modal(modal).mass_normalized()
    if n_modes is not None:
        mm = mm.truncate(n_modes=n_modes)
    dmp: DampingModel = as_damping(damping)

    F, taxis = _build_force(force, mm.ndof, t, dt, n_steps, force_dofs)
    step = float(taxis[1] - taxis[0]) if taxis.size > 1 else 0.0
    if step <= 0.0:
        raise ValueError("the time step must be positive")

    two_zeta_omega, eta = dmp.modal_terms(mm)
    if np.any(eta != 0.0):
        warnings.warn(
            "structural (hysteretic) damping is not causal in the time domain; "
            "using the equivalent viscous ratio zeta = eta / 2",
            RuntimeWarning,
            stacklevel=2,
        )
    omega = mm.omega
    # q'' + c q' + w^2 q = p, with hysteretic eta folded in as an equivalent viscous term.
    c_modal = two_zeta_omega + eta * omega

    P = mm.modes.T @ F

    qi = np.zeros(mm.n_modes)
    qdi = np.zeros(mm.n_modes)
    if q0 is not None:
        qi = np.asarray(q0, dtype=float).reshape(-1)
    elif x0 is not None:
        qi = _project_state(mm, np.asarray(x0, dtype=float).reshape(-1), M)
    if qd0 is not None:
        qdi = np.asarray(qd0, dtype=float).reshape(-1)
    elif v0 is not None:
        qdi = _project_state(mm, np.asarray(v0, dtype=float).reshape(-1), M)

    if method == "exact":
        q, qd = _integrate_exact(omega, c_modal, P, step, qi, qdi)
    elif method == "newmark":
        q, qd = _integrate_newmark(omega, c_modal, P, step, qi, qdi)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'exact' or 'newmark'")

    qdd = P - c_modal[:, None] * qd - (omega**2)[:, None] * q

    out_idx = resolve_dofs(outputs, mm.ndof, "outputs", mm.dof_ids)
    phi_o = mm.modes[out_idx, :]
    return TimeHistoryResult(
        t=taxis,
        displacement=phi_o @ q,
        velocity=phi_o @ qd,
        acceleration=phi_o @ qdd,
        modal_displacement=q,
        modal_velocity=qd,
        modal_acceleration=qdd,
        outputs=out_idx,
        method=method,
        meta={"n_modes": mm.n_modes, "dt": step, "damping": type(dmp).__name__},
    )


def _project_state(mm: ModalModel, x: np.ndarray, M: Any) -> np.ndarray:
    """Project a physical state vector onto mass-normalised modal coordinates."""
    if x.size != mm.ndof:
        raise ValueError(f"initial state must have {mm.ndof} entries, got {x.size}")
    if M is not None:
        return mm.modes.T @ (as_dense(M) @ x)
    return np.linalg.lstsq(mm.modes, x, rcond=None)[0]
