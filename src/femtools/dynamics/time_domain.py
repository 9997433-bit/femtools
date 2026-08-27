"""Transient response by modal superposition.

Each retained mode is an uncoupled SDOF oscillator

    q_r'' + 2 zeta_r w_r q_r' + w_r^2 q_r = phi_r^T f(t)

integrated with the exact recurrence for a load that varies linearly inside a step
(the ramp-invariant / Nigam-Jennings recurrence). That recurrence is unconditionally
stable and exact for piecewise-linear excitation, so ``dt`` only has to resolve the load
and the highest retained mode. A Newmark average-acceleration variant is available for
comparison.

The recurrence coefficients are *evaluated* through the block matrix exponential of
:func:`_ramp_coefficients` rather than through the textbook closed form in
``sin``/``cos``/``exp``. The closed form is correct but not computable in floating point
across the range of modes a real basis contains: its load coefficients divide by
``omega^2`` a bracket that cancels to nothing as ``omega -> 0``, so a rigid-body mode that
an eigensolver reports at 1e-7 Hz instead of exactly 0 — the normal outcome, since
``eigh`` returns ``-1.8e-11`` for such an eigenvalue — loses every significant digit. See
the docstring of :func:`_ramp_coefficients` for the measured breakdown.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla

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


def _ramp_coefficients(
    omega: np.ndarray, c: np.ndarray, dt: float
) -> tuple[np.ndarray, ...]:
    """Exact piecewise-linear recurrence coefficients ``(A, B, C, D, Ap, Bp, Cp, Dp)``.

    For ``q'' + c q' + w^2 q = p`` with ``p`` linear across the step::

        q_{n+1} = A q_n + B q'_n + C p_n + D p_{n+1}
        q'_{n+1} = Ap q_n + Bp q'_n + Cp p_n + Dp p_{n+1}

    The coefficients are the first-order-hold discretisation of the companion system
    ``x' = Ac x + Bc p``, ``Ac = [[0, 1], [-w^2, -c]]``, ``Bc = [0, 1]^T``, and are read
    off one block matrix exponential per mode (Van Loan, *Computing integrals involving
    the matrix exponential*, IEEE TAC 23(3), 1978)::

        expm(dt * [[Ac, Bc, 0 ],      [[ Ad, G1, G2 ],
                   [0,  0,  1 ],  =    [ 0,  1,  dt ],
                   [0,  0,  0 ]])      [ 0,  0,  1  ]]

    with ``Ad`` the transition matrix, ``G1`` the step (zero-order-hold) load column and
    ``G2 / dt`` the ramp one, so ``[C, Cp] = G1 - G2/dt`` and ``[D, Dp] = G2/dt``.

    This is *not* how the recurrence is usually written. The textbook closed form in
    ``exp(-zeta w dt)``, ``sin(w_d dt)`` and ``cos(w_d dt)`` is algebraically identical but
    is not computable in floating point over the range of modes a real basis holds,
    because its load coefficients divide a bracket that cancels to ``O((w dt)^2)`` by
    ``w^2``. Measured against a 60-digit reference, the closed form loses (worst
    coefficient, ``zeta = 0.02``) 5 digits at ``w dt = 1e-4``, 11 at ``1e-5`` and all of
    them at ``1e-6``; with the heavy damping that Rayleigh ``alpha`` gives a near-rigid
    mode it also overflows to ``nan``. The scaling-and-squaring Padé evaluation behind
    ``expm`` stays at round-off across that whole range, and reproduces the ``w = 0``
    limit (``C = dt^2/3``, ``D = dt^2/6``, ``Cp = Dp = dt/2``) and critical damping
    exactly, so rigid-body and over-damped modes need no special case. The price is
    ``1e-15`` instead of ``1e-16`` at moderate ``w dt`` and one small exponential per mode
    at set-up, outside the time loop.
    """
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    cc = np.atleast_1d(np.asarray(c, dtype=float))
    n = w.size
    if n == 0:
        empty = np.zeros(0)
        return tuple(empty.copy() for _ in range(8))

    blocks = np.zeros((n, 4, 4))
    blocks[:, 0, 1] = 1.0
    blocks[:, 1, 0] = -(w**2)
    blocks[:, 1, 1] = -cc
    blocks[:, 1, 2] = 1.0
    blocks[:, 2, 3] = 1.0
    E = np.asarray(sla.expm(blocks * float(dt)))

    Ad = E[:, :2, :2]
    step = E[:, :2, 2]  # zero-order-hold load column
    ramp = E[:, :2, 3] / float(dt)  # ramp load column
    return (
        Ad[:, 0, 0],
        Ad[:, 0, 1],
        step[:, 0] - ramp[:, 0],
        ramp[:, 0],
        Ad[:, 1, 0],
        Ad[:, 1, 1],
        step[:, 1] - ramp[:, 1],
        ramp[:, 1],
    )


def _integrate_exact(
    omega: np.ndarray, c: np.ndarray, P: np.ndarray, dt: float, q0: np.ndarray,
    qd0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact piecewise-linear recurrence for all modes at once. ``P`` is ``(n_modes, n_t)``.

    One code path covers rigid-body, under-, critically and over-damped modes; see
    :func:`_ramp_coefficients` for why the zero-frequency limit needs no branch of its own.
    """
    n_modes, n_steps = P.shape
    q = np.zeros((n_modes, n_steps))
    qd = np.zeros((n_modes, n_steps))
    q[:, 0] = q0
    qd[:, 0] = qd0

    A, B, C, D, Ap, Bp, Cp, Dp = _ramp_coefficients(omega, c, dt)
    for k in range(n_steps - 1):
        qk, qdk = q[:, k], qd[:, k]
        pk, pk1 = P[:, k], P[:, k + 1]
        q[:, k + 1] = A * qk + B * qdk + C * pk + D * pk1
        qd[:, k + 1] = Ap * qk + Bp * qdk + Cp * pk + Dp * pk1
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
        qi = _modal_state(mm, q0, "q0")
    elif x0 is not None:
        qi = _project_state(mm, np.asarray(x0, dtype=float).reshape(-1), M)
    if qd0 is not None:
        qdi = _modal_state(mm, qd0, "qd0")
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


def _modal_state(mm: ModalModel, value: Any, name: str) -> np.ndarray:
    """Validate an initial modal state vector against the size of the basis.

    A silent size mismatch here is worse than useless: a length-1 ``q0`` broadcasts over
    every mode, so asking for "mode 0 displaced by one" starts *all* of them at one.
    """
    arr = np.atleast_1d(np.asarray(value, dtype=float)).reshape(-1)
    if arr.size != mm.n_modes:
        raise ValueError(
            f"{name} must have one entry per retained mode ({mm.n_modes}), got {arr.size}"
        )
    return arr


def _project_state(mm: ModalModel, x: np.ndarray, M: Any) -> np.ndarray:
    """Project a physical state vector onto mass-normalised modal coordinates."""
    if x.size != mm.ndof:
        raise ValueError(f"initial state must have {mm.ndof} entries, got {x.size}")
    if M is not None:
        return mm.modes.T @ (as_dense(M) @ x)
    return np.linalg.lstsq(mm.modes, x, rcond=None)[0]
