"""Modal model container and adapters.

The dynamics package consumes modal data produced by :func:`femtools.fea.eigen.solve_modes`
(a ``ModalResult`` with ``freq_hz``, ``eigenvalues``, ``modes`` and ``generalized_mass``).
To keep the dynamics package importable and testable on its own, everything here is
duck-typed: :func:`as_modal` accepts that object, a :class:`ModalModel`, a mapping, or a
plain ``(freq_hz, modes)`` tuple.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._utils import TWO_PI, as_dense

__all__ = ["ModalModel", "as_modal"]


@dataclass
class ModalModel:
    """A truncated modal model.

    Attributes
    ----------
    freq_hz:
        Undamped natural frequencies in Hz, shape ``(n_modes,)``, ascending.
    modes:
        Mode shape matrix ``Phi``, shape ``(ndof, n_modes)``. Mass-normalised by
        convention (``Phi.T @ M @ Phi == I``); if not, provide ``generalized_mass``.
    generalized_mass:
        Modal masses ``m_r = phi_r.T M phi_r``, shape ``(n_modes,)``. Defaults to ones.
    eigenvalues:
        ``omega_r**2`` in (rad/s)^2, shape ``(n_modes,)``. Derived from ``freq_hz`` when
        not supplied.
    dof_ids:
        Optional global DOF labels for the rows of ``modes``, shape ``(ndof,)``. When
        given, DOF selections may be expressed with these labels instead of row indices.
    """

    freq_hz: np.ndarray
    modes: np.ndarray
    generalized_mass: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    dof_ids: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.freq_hz = np.atleast_1d(np.asarray(self.freq_hz, dtype=float)).reshape(-1)
        modes = as_dense(self.modes)
        if modes.ndim == 1:
            modes = modes.reshape(-1, 1)
        if modes.ndim != 2:
            raise ValueError(f"modes must be 2-D (ndof, n_modes), got shape {modes.shape}")
        if modes.shape[1] != self.freq_hz.size:
            raise ValueError(
                f"modes has {modes.shape[1]} columns but freq_hz has "
                f"{self.freq_hz.size} entries"
            )
        self.modes = modes

        if self.generalized_mass is None:
            self.generalized_mass = np.ones(self.n_modes)
        else:
            gm = np.atleast_1d(np.asarray(self.generalized_mass, dtype=float)).reshape(-1)
            if gm.size == 1:
                gm = np.full(self.n_modes, float(gm[0]))
            if gm.size != self.n_modes:
                raise ValueError("generalized_mass length must match the number of modes")
            if np.any(gm <= 0.0):
                raise ValueError("generalized_mass entries must be strictly positive")
            self.generalized_mass = gm

        if self.eigenvalues is None:
            self.eigenvalues = (TWO_PI * self.freq_hz) ** 2
        else:
            ev = np.atleast_1d(np.asarray(self.eigenvalues, dtype=float)).reshape(-1)
            if ev.size != self.n_modes:
                raise ValueError("eigenvalues length must match the number of modes")
            self.eigenvalues = ev

        if self.dof_ids is not None:
            ids = np.atleast_1d(np.asarray(self.dof_ids, dtype=int)).reshape(-1)
            if ids.size != self.ndof:
                raise ValueError("dof_ids length must match the number of rows in modes")
            self.dof_ids = ids

    # -- basic properties -------------------------------------------------
    @property
    def ndof(self) -> int:
        """Number of physical DOFs (rows of ``modes``)."""
        return int(self.modes.shape[0])

    @property
    def n_modes(self) -> int:
        """Number of retained modes."""
        return int(self.freq_hz.size)

    @property
    def omega(self) -> np.ndarray:
        """Undamped circular natural frequencies in rad/s, shape ``(n_modes,)``."""
        return TWO_PI * self.freq_hz

    def mass_normalized(self) -> ModalModel:
        """Return an equivalent model whose modes satisfy ``Phi.T M Phi = I``."""
        gm = np.asarray(self.generalized_mass, dtype=float)
        if np.allclose(gm, 1.0):
            return self
        return ModalModel(
            freq_hz=self.freq_hz.copy(),
            modes=self.modes / np.sqrt(gm)[None, :],
            generalized_mass=np.ones(self.n_modes),
            eigenvalues=np.asarray(self.eigenvalues).copy(),
            dof_ids=None if self.dof_ids is None else self.dof_ids.copy(),
            meta=dict(self.meta),
        )

    def select(self, indices: Sequence[int] | np.ndarray) -> ModalModel:
        """Return a modal model restricted to the given mode indices."""
        idx = np.asarray(indices, dtype=int).reshape(-1)
        return ModalModel(
            freq_hz=self.freq_hz[idx],
            modes=self.modes[:, idx],
            generalized_mass=np.asarray(self.generalized_mass)[idx],
            eigenvalues=np.asarray(self.eigenvalues)[idx],
            dof_ids=None if self.dof_ids is None else self.dof_ids.copy(),
            meta=dict(self.meta),
        )

    def truncate(self, n_modes: int | None = None, fmax_hz: float | None = None) -> ModalModel:
        """Keep the first ``n_modes`` modes and/or those below ``fmax_hz``."""
        keep = np.ones(self.n_modes, dtype=bool)
        if fmax_hz is not None:
            keep &= self.freq_hz <= float(fmax_hz)
        idx = np.flatnonzero(keep)
        if n_modes is not None:
            idx = idx[: int(n_modes)]
        return self.select(idx)


def as_modal(obj: Any) -> ModalModel:
    """Coerce ``obj`` into a :class:`ModalModel`.

    Accepts a :class:`ModalModel`, any object exposing ``freq_hz``/``modes`` (such as
    ``femtools.fea.eigen.ModalResult``), a mapping with those keys, or a
    ``(freq_hz, modes)`` / ``(freq_hz, modes, generalized_mass)`` tuple.
    """
    if isinstance(obj, ModalModel):
        return obj
    if isinstance(obj, Mapping):
        data = dict(obj)
        return ModalModel(
            freq_hz=data["freq_hz"],
            modes=data["modes"],
            generalized_mass=data.get("generalized_mass"),
            eigenvalues=data.get("eigenvalues"),
            dof_ids=data.get("dof_ids"),
            meta=dict(data.get("meta", {})),
        )
    if isinstance(obj, tuple | list) and 2 <= len(obj) <= 3:
        freq, modes = obj[0], obj[1]
        gm = obj[2] if len(obj) == 3 else None
        return ModalModel(freq_hz=freq, modes=modes, generalized_mass=gm)
    if hasattr(obj, "freq_hz") and hasattr(obj, "modes"):
        return ModalModel(
            freq_hz=obj.freq_hz,
            modes=obj.modes,
            generalized_mass=getattr(obj, "generalized_mass", None),
            eigenvalues=getattr(obj, "eigenvalues", None),
            dof_ids=getattr(obj, "dof_ids", None),
        )
    raise TypeError(
        "cannot interpret object as a modal model; expected ModalModel, ModalResult, "
        f"mapping or (freq_hz, modes) tuple, got {type(obj).__name__}"
    )
