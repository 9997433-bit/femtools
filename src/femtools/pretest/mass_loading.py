"""Accelerometer mass-loading estimate.

Transducers add mass to the structure, which lowers the measured frequencies
and, when the added mass is not small, also distorts the mode shapes.  Both
effects are estimated here from the *unloaded* modal model alone, so a
pretest can check the sensor budget before any hardware is mounted.

First order (Rayleigh quotient, ``dM`` small)::

    df_j / f_j = -0.5 * sum_s m_s |phi_sj|^2 / m_j

Modal projection (default, valid for larger masses as long as the target
modes span the response)::

    diag(m_j w_j^2) x = lam (diag(m_j) + Phi_s^T diag(m) Phi_s) x

Only translational point masses are modelled; rotary inertia and the
stiffness of the mounting are ignored, so the estimate is a lower bound on
the true frequency shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..correlation._linalg import as_mode_matrix, mode_frequencies, row_index
from ..correlation.mac import mac_matrix

__all__ = ["MassLoadingResult", "mass_loading", "sensor_mass_limit"]


@dataclass
class MassLoadingResult:
    """Estimated effect of the transducer masses on the modal model."""

    freq_hz: NDArray[np.float64]
    freq_hz_loaded: NDArray[np.float64]
    contribution: NDArray[np.float64]
    method: str = "modal"
    modes_loaded: NDArray[Any] | None = None
    mac: NDArray[np.float64] | None = None
    added_mass: NDArray[np.float64] | None = None

    @property
    def delta_hz(self) -> NDArray[np.float64]:
        """Signed frequency change (negative: the mass lowers the frequency)."""
        return self.freq_hz_loaded - self.freq_hz

    @property
    def relative_shift(self) -> NDArray[np.float64]:
        """``(f_loaded - f) / f`` per mode."""
        out = np.zeros_like(self.freq_hz)
        np.divide(self.delta_hz, self.freq_hz, out=out, where=self.freq_hz > 0.0)
        return out

    @property
    def max_relative_shift(self) -> float:
        return float(np.abs(self.relative_shift).max()) if self.freq_hz.size else 0.0

    @property
    def worst_mode(self) -> int:
        return int(np.argmax(np.abs(self.relative_shift))) if self.freq_hz.size else -1

    def table(self) -> str:
        head = f"{'mode':>5} {'f [Hz]':>12} {'f_load [Hz]':>12} {'df [%]':>9}"
        lines = [head, "-" * len(head)]
        rel = 100.0 * self.relative_shift
        for j in range(self.freq_hz.size):
            lines.append(
                f"{j:>5} {self.freq_hz[j]:>12.4f} {self.freq_hz_loaded[j]:>12.4f} {rel[j]:>9.3f}"
            )
        return "\n".join(lines)


def _sensor_partition(
    phi: NDArray[Any], added_mass: ArrayLike, dofs: ArrayLike | None
) -> tuple[NDArray[Any], NDArray[np.float64]]:
    n_dof = phi.shape[0]
    m = np.asarray(added_mass, dtype=float).reshape(-1)
    if dofs is None:
        if m.size == 1:
            rows = np.arange(n_dof, dtype=np.intp)
            m = np.full(n_dof, float(m[0]))
        elif m.size == n_dof:
            rows = np.arange(n_dof, dtype=np.intp)
        else:
            raise ValueError(f"added_mass has {m.size} entries; pass `dofs` to say where they act")
    else:
        rows = row_index(dofs, n_dof)
        if m.size == 1:
            m = np.full(rows.size, float(m[0]))
        if m.size != rows.size:
            raise ValueError(f"added_mass has {m.size} entries but {rows.size} DOFs are loaded")
    if np.any(m < 0.0):
        raise ValueError("added_mass must be non-negative")
    if rows.size and (rows.min() < 0 or rows.max() >= n_dof):
        raise ValueError("dofs index outside the mode shape rows")
    return phi[rows, :], m


def mass_loading(
    phi: ArrayLike,
    freq_hz: ArrayLike | None = None,
    added_mass: ArrayLike = 0.0,
    dofs: ArrayLike | None = None,
    *,
    method: str = "modal",
    generalized_mass: ArrayLike | None = None,
) -> MassLoadingResult:
    """Estimate the frequency (and shape) shift caused by sensor masses.

    Parameters
    ----------
    phi:
        Mass-normalized mode shapes ``(n_dof, n_mode)`` of the unloaded
        structure, in consistent units (SI: m/sqrt(kg)).
    freq_hz:
        Corresponding natural frequencies [Hz]; taken from ``phi`` when it is
        a modal result carrying them.
    added_mass:
        Mass per sensor [kg]: a scalar for identical transducers, or one
        value per entry of ``dofs``.
    dofs:
        Row indices (or a boolean mask) of ``phi`` where the masses act,
        i.e. the sensor DOFs.  ``None`` loads every DOF.
    method:
        ``"modal"`` (default) solves the reduced eigenproblem in the space of
        the given modes; ``"first_order"`` uses the Rayleigh-quotient
        approximation and returns no shapes.
    generalized_mass:
        Modal masses ``m_j`` when ``phi`` is not mass-normalized.

    Returns
    -------
    MassLoadingResult
        Loaded frequencies, per-sensor contribution to the shift and — for
        the modal method — the loaded shapes and their MAC against the
        unloaded ones (values below 1 signal shape distortion).
    """
    p = as_mode_matrix(phi, "phi")
    if freq_hz is None:
        freq_hz = mode_frequencies(phi)
        if freq_hz is None:
            raise ValueError("freq_hz is required unless phi carries its own frequencies")
    f = np.asarray(freq_hz, dtype=float).reshape(-1)
    if f.size != p.shape[1]:
        raise ValueError(f"freq_hz has {f.size} entries but phi has {p.shape[1]} modes")
    if np.any(f < 0.0):
        raise ValueError("freq_hz must be non-negative")

    if generalized_mass is None:
        gen = np.ones(f.size)
    else:
        gen = np.asarray(generalized_mass, dtype=float).reshape(-1)
        if gen.size != f.size:
            raise ValueError(f"generalized_mass has {gen.size} entries, expected {f.size}")
        if np.any(gen <= 0.0):
            raise ValueError("generalized_mass must be positive")

    phi_s, m = _sensor_partition(p, added_mass, dofs)
    contribution = (m[:, None] * np.abs(phi_s) ** 2) / gen[None, :]

    key = str(method).lower().replace("-", "_")
    if key in ("first_order", "rayleigh", "perturbation"):
        rel = -0.5 * contribution.sum(axis=0)
        return MassLoadingResult(
            freq_hz=f,
            freq_hz_loaded=f * (1.0 + rel),
            contribution=contribution,
            method="first_order",
            added_mass=m,
        )
    if key != "modal":
        raise ValueError(f"unknown method {method!r}; use 'modal' or 'first_order'")

    omega = 2.0 * np.pi * f
    stiff = np.diag(gen * omega**2)
    m_mod = np.diag(gen) + np.real(phi_s.conj().T @ (m[:, None] * phi_s))
    m_mod = 0.5 * (m_mod + m_mod.T)

    chol = np.linalg.cholesky(m_mod)
    inv_l = np.linalg.inv(chol)
    reduced = inv_l @ stiff @ inv_l.T
    reduced = 0.5 * (reduced + reduced.T)
    lam, vec = np.linalg.eigh(reduced)
    lam = np.clip(lam, 0.0, None)
    coeff = inv_l.T @ vec

    order = np.argsort(lam, kind="stable")
    lam = lam[order]
    coeff = coeff[:, order]
    modes_loaded = p @ coeff
    f_loaded = np.sqrt(lam) / (2.0 * np.pi)

    # Keep the loaded modes in correspondence with the unloaded ones.
    mac = mac_matrix(p, modes_loaded)
    return MassLoadingResult(
        freq_hz=f,
        freq_hz_loaded=f_loaded,
        contribution=contribution,
        method="modal",
        modes_loaded=modes_loaded,
        mac=mac,
        added_mass=m,
    )


def sensor_mass_limit(
    phi: ArrayLike,
    max_relative_shift: float = 0.01,
    dofs: ArrayLike | None = None,
    *,
    generalized_mass: ArrayLike | None = None,
    n_sensors: int | None = None,
) -> NDArray[np.float64]:
    """Largest transducer mass per sensor DOF meeting a frequency-shift budget.

    Inverts the first-order estimate: ``m_max = 2 * shift / max_j |phi_sj|^2``,
    divided by ``n_sensors`` when several equally-loaded sensors share the
    budget.  Returns one limit [kg] per sensor DOF.

    ``dofs`` takes row indices or a boolean mask, as in :func:`mass_loading`.
    """
    p = as_mode_matrix(phi, "phi")
    rows = np.arange(p.shape[0], dtype=np.intp) if dofs is None else row_index(dofs, p.shape[0])
    if rows.size and (rows.min() < -p.shape[0] or rows.max() >= p.shape[0]):
        raise ValueError("dofs index outside the mode shape rows")
    if max_relative_shift <= 0.0:
        raise ValueError("max_relative_shift must be positive")
    if generalized_mass is None:
        gen = np.ones(p.shape[1])
    else:
        gen = np.asarray(generalized_mass, dtype=float).reshape(-1)
        if gen.size != p.shape[1]:
            raise ValueError(f"generalized_mass has {gen.size} entries, expected {p.shape[1]}")
        if np.any(gen <= 0.0):
            raise ValueError("generalized_mass must be positive")
    worst = (np.abs(p[rows, :]) ** 2 / gen[None, :]).max(axis=1)
    share = float(n_sensors) if n_sensors else 1.0
    limit = np.full(rows.size, np.inf)
    np.divide(2.0 * max_relative_shift, worst * share, out=limit, where=worst > 0.0)
    return limit
