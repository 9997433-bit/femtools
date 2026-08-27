"""Damping models: modal (viscous per-mode zeta), Rayleigh (proportional) and structural.

Every model exposes two views of the same physics:

* a **modal** view, :meth:`DampingModel.modal_terms`, returning ``(2*zeta_r*omega_r, eta_r)``
  so that the modal receptance denominator is

  ``D_r(omega) = omega_r**2 * (1 + 1j*eta_r) - omega**2 + 1j*omega*(2*zeta_r*omega_r)``

* a **physical** view, :meth:`DampingModel.viscous_matrix` and
  :meth:`DampingModel.loss_factor`, giving the viscous matrix ``C`` and the hysteretic
  loss factor used to build the dynamic stiffness

  ``Z(omega) = K*(1 + 1j*eta) - omega**2 * M + 1j*omega*C``

Keeping both views consistent is what makes ``modal_frf`` and ``direct_frf`` agree to
machine precision when the full modal basis is retained.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from ._utils import TWO_PI, as_dense, broadcast_scalar, is_sparse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .modal import ModalModel

__all__ = [
    "CombinedDamping",
    "DampingModel",
    "ModalDamping",
    "NoDamping",
    "RayleighDamping",
    "StructuralDamping",
    "ViscousDamping",
    "as_damping",
    "rayleigh_coefficients",
]


class DampingModel(ABC):
    """Base class for damping descriptions."""

    @abstractmethod
    def modal_terms(self, modal: ModalModel) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(two_zeta_omega, eta)`` arrays of length ``modal.n_modes``."""

    def modal_zeta(self, modal: ModalModel) -> np.ndarray:
        """Equivalent viscous damping ratio per mode (0 where ``omega_r == 0``)."""
        two_zeta_omega, eta = self.modal_terms(modal)
        omega = modal.omega
        zeta = np.zeros_like(omega)
        nz = omega > 0.0
        zeta[nz] = two_zeta_omega[nz] / (2.0 * omega[nz])
        # A hysteretic loss factor eta corresponds to an equivalent zeta = eta / 2.
        return zeta + 0.5 * eta

    def viscous_matrix(self, K: Any, M: Any, modal: ModalModel | None = None) -> Any | None:
        """Physical viscous damping matrix ``C``, or ``None`` when the model has none."""
        return None

    def loss_factor(self) -> float:
        """Hysteretic loss factor multiplying ``K`` in the dynamic stiffness."""
        return 0.0

    def __add__(self, other: DampingModel) -> CombinedDamping:
        if not isinstance(other, DampingModel):
            return NotImplemented
        return CombinedDamping([self, other])


@dataclass(frozen=True)
class NoDamping(DampingModel):
    """Undamped model. Receptance is purely real away from the poles."""

    def modal_terms(self, modal: ModalModel) -> tuple[np.ndarray, np.ndarray]:
        zeros = np.zeros(modal.n_modes)
        return zeros, zeros.copy()


@dataclass(frozen=True)
class ModalDamping(DampingModel):
    """Per-mode viscous damping ratio ``zeta``.

    ``zeta`` is either a scalar applied to every mode or one value per retained mode.
    The physical equivalent (used by :func:`~femtools.dynamics.frf.direct_frf`) is the
    Caughey/Rayleigh-family matrix ``C = M Phi diag(2 zeta_r omega_r) Phi^T M``, which
    requires the modal basis and is exact only when the basis is complete.
    """

    zeta: float | Sequence[float] | np.ndarray = 0.0

    def modal_terms(self, modal: ModalModel) -> tuple[np.ndarray, np.ndarray]:
        zeta = broadcast_scalar(self.zeta, modal.n_modes, "zeta")
        return 2.0 * zeta * modal.omega, np.zeros(modal.n_modes)

    def viscous_matrix(self, K: Any, M: Any, modal: ModalModel | None = None) -> Any | None:
        if modal is None:
            raise ValueError(
                "ModalDamping needs the modal basis to build a physical C matrix; "
                "pass modal=... to direct_frf or supply C explicitly"
            )
        mm = modal.mass_normalized()
        two_zeta_omega, _ = self.modal_terms(mm)
        phi = mm.modes
        Md = M if not is_sparse(M) else sp.csr_matrix(M)
        mphi = Md @ phi
        mphi = as_dense(mphi)
        return (mphi * two_zeta_omega[None, :]) @ mphi.T


@dataclass(frozen=True)
class RayleighDamping(DampingModel):
    """Proportional damping ``C = alpha * M + beta * K``.

    The induced modal damping ratio is ``zeta_r = (alpha / omega_r + beta * omega_r) / 2``.
    """

    alpha: float = 0.0
    beta: float = 0.0

    def modal_terms(self, modal: ModalModel) -> tuple[np.ndarray, np.ndarray]:
        omega = modal.omega
        # 2 zeta_r omega_r = alpha + beta * omega_r**2 (finite even for rigid-body modes).
        return self.alpha + self.beta * omega**2, np.zeros(modal.n_modes)

    def viscous_matrix(self, K: Any, M: Any, modal: ModalModel | None = None) -> Any | None:
        if self.alpha == 0.0 and self.beta == 0.0:
            return None
        if is_sparse(K) or is_sparse(M):
            return self.alpha * sp.csr_matrix(M) + self.beta * sp.csr_matrix(K)
        return self.alpha * as_dense(M) + self.beta * as_dense(K)


@dataclass(frozen=True)
class StructuralDamping(DampingModel):
    """Hysteretic (structural) damping: complex stiffness ``K (1 + 1j*eta)``.

    ``eta`` may be a scalar (usable in both the modal and the physical view) or one
    value per mode (modal view only).
    """

    eta: float | Sequence[float] | np.ndarray = 0.0

    def modal_terms(self, modal: ModalModel) -> tuple[np.ndarray, np.ndarray]:
        eta = broadcast_scalar(self.eta, modal.n_modes, "eta")
        return np.zeros(modal.n_modes), eta

    def loss_factor(self) -> float:
        eta = np.atleast_1d(np.asarray(self.eta, dtype=float))
        if eta.size != 1:
            raise ValueError(
                "per-mode structural damping cannot be mapped to a single physical "
                "loss factor; use a scalar eta for direct_frf"
            )
        return float(eta.reshape(-1)[0])


@dataclass(frozen=True)
class ViscousDamping(DampingModel):
    """Explicit physical viscous damping matrix ``C`` (``ndof x ndof``).

    In the modal view the matrix is projected and its diagonal retained
    (``2 zeta_r omega_r = phi_r^T C phi_r``), i.e. the classical modal-damping
    assumption. Off-diagonal modal coupling is reported by :meth:`coupling_ratio`.
    """

    C: Any = None

    def modal_terms(self, modal: ModalModel) -> tuple[np.ndarray, np.ndarray]:
        mm = modal.mass_normalized()
        Cm = as_dense(mm.modes.T @ (self.C @ mm.modes))
        return np.diag(Cm).copy(), np.zeros(mm.n_modes)

    def coupling_ratio(self, modal: ModalModel) -> float:
        """Max off-diagonal / diagonal magnitude of the projected ``C`` (0 = classical)."""
        mm = modal.mass_normalized()
        Cm = np.abs(as_dense(mm.modes.T @ (self.C @ mm.modes)))
        diag = np.diag(Cm).copy()
        off = Cm - np.diag(diag)
        scale = diag.max() if diag.size and diag.max() > 0 else 1.0
        return float(off.max() / scale) if off.size else 0.0

    def viscous_matrix(self, K: Any, M: Any, modal: ModalModel | None = None) -> Any | None:
        return self.C


@dataclass(frozen=True)
class CombinedDamping(DampingModel):
    """Superposition of several damping models."""

    models: Sequence[DampingModel] = ()

    def modal_terms(self, modal: ModalModel) -> tuple[np.ndarray, np.ndarray]:
        two_zeta_omega = np.zeros(modal.n_modes)
        eta = np.zeros(modal.n_modes)
        for m in self.models:
            a, b = m.modal_terms(modal)
            two_zeta_omega = two_zeta_omega + a
            eta = eta + b
        return two_zeta_omega, eta

    def loss_factor(self) -> float:
        return float(sum(m.loss_factor() for m in self.models))

    def viscous_matrix(self, K: Any, M: Any, modal: ModalModel | None = None) -> Any | None:
        total = None
        for m in self.models:
            Ci = m.viscous_matrix(K, M, modal)
            if Ci is None:
                continue
            total = Ci if total is None else total + Ci
        return total


def rayleigh_coefficients(
    f1_hz: float, f2_hz: float, zeta1: float, zeta2: float | None = None
) -> tuple[float, float]:
    """Solve for ``(alpha, beta)`` matching ``zeta`` at two frequencies.

    With ``zeta2 is None`` the same ratio is imposed at both frequencies.
    """
    if zeta2 is None:
        zeta2 = zeta1
    w1, w2 = TWO_PI * float(f1_hz), TWO_PI * float(f2_hz)
    if w1 <= 0.0 or w2 <= 0.0 or np.isclose(w1, w2):
        raise ValueError("f1_hz and f2_hz must be positive and distinct")
    A = 0.5 * np.array([[1.0 / w1, w1], [1.0 / w2, w2]])
    alpha, beta = np.linalg.solve(A, np.array([float(zeta1), float(zeta2)]))
    return float(alpha), float(beta)


def as_damping(spec: Any) -> DampingModel:
    """Coerce ``spec`` into a :class:`DampingModel`.

    Accepted forms:

    * ``None`` / ``0`` -> :class:`NoDamping`
    * scalar or array  -> :class:`ModalDamping` (viscous zeta)
    * mapping with ``zeta`` / ``alpha``+``beta`` / ``eta`` / ``C`` keys
    * a square matrix   -> :class:`ViscousDamping`
    * an existing :class:`DampingModel` (returned unchanged)
    """
    if spec is None:
        return NoDamping()
    if isinstance(spec, DampingModel):
        return spec
    if isinstance(spec, Mapping):
        parts: list[DampingModel] = []
        keys = set(spec)
        if "zeta" in keys:
            parts.append(ModalDamping(spec["zeta"]))
        if "alpha" in keys or "beta" in keys:
            parts.append(RayleighDamping(float(spec.get("alpha", 0.0)),
                                         float(spec.get("beta", 0.0))))
        if "eta" in keys:
            parts.append(StructuralDamping(spec["eta"]))
        if "C" in keys:
            parts.append(ViscousDamping(spec["C"]))
        unknown = keys - {"zeta", "alpha", "beta", "eta", "C"}
        if unknown:
            raise ValueError(f"unknown damping keys: {sorted(unknown)}")
        if not parts:
            return NoDamping()
        return parts[0] if len(parts) == 1 else CombinedDamping(parts)
    if is_sparse(spec):
        return ViscousDamping(spec)
    arr = np.asarray(spec, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 1:
        return ViscousDamping(arr)
    if arr.ndim == 0 and float(arr) == 0.0:
        return NoDamping()
    return ModalDamping(arr if arr.ndim else float(arr))
