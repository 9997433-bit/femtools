"""Self-contained reference models used when ``femtools.fea`` is unavailable.

The updating / sensitivity machinery in this package is solver independent: it
only ever needs a *response function* ``p -> r(p)``.  To keep the package
testable in isolation (and to provide the classic Friswell--Mottershead
demonstration cases) this module implements a handful of tiny analytical
models with numpy only:

* :class:`TwoDOFModel`      -- the 2-DOF spring/mass fallback,
* :class:`AxialBarModel`    -- ``n``-element axial rod, consistent mass,
* :class:`BeamModel`        -- ``n``-element Euler--Bernoulli beam.

All of them expose :meth:`assemble` ``-> (K, M)`` and :meth:`frequencies`, and
are parameterised by *relative* stiffness multipliers (one per region), which is
exactly the form used by sensitivity based model updating.

When ``femtools.fea`` **is** importable, prefer
:func:`femtools.updating.responses.modal_response_function`, which drives the
real solver.  These classes remain useful as fast, dependency-free golden cases.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh

__all__ = [
    "ReferenceModel",
    "TwoDOFModel",
    "AxialBarModel",
    "BeamModel",
    "analytical_axial_frequencies",
    "analytical_cantilever_frequencies",
    "make_updating_testcase",
]


class ReferenceModel:
    """Base class: turns an ``assemble(values) -> (K, M)`` into modal data."""

    n_parameters: int = 0
    parameter_names: tuple[str, ...] = ()

    def assemble(self, values: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def stiffness_derivatives(self) -> list[np.ndarray]:
        """Exact ``dK/dp_j`` — reference models are linear in their multipliers."""
        raise NotImplementedError

    def mass_derivatives(self) -> list[np.ndarray]:
        """Exact ``dM/dp_j`` (zero for stiffness-only parameters)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _values(self, values: ArrayLike | None) -> np.ndarray:
        if values is None:
            return np.ones(self.n_parameters)
        v = np.atleast_1d(np.asarray(values, dtype=float))
        if v.size != self.n_parameters:
            raise ValueError(f"expected {self.n_parameters} values, got {v.size}")
        return v

    def eig(
        self, values: ArrayLike | None = None, n_modes: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(freq_hz, mass-normalised modes)`` in ascending order."""
        K, M = self.assemble(values)
        lam, phi = eigh(K, M)
        lam = np.clip(lam, 0.0, None)
        freq = np.sqrt(lam) / (2.0 * math.pi)
        order = np.argsort(freq)
        freq, phi = freq[order], phi[:, order]
        if n_modes is not None:
            freq, phi = freq[:n_modes], phi[:, :n_modes]
        # eigh already M-normalises; fix signs for reproducibility
        for j in range(phi.shape[1]):
            k = int(np.argmax(np.abs(phi[:, j])))
            if phi[k, j] < 0:
                phi[:, j] *= -1.0
        return freq, phi

    def frequencies(
        self, values: ArrayLike | None = None, n_modes: int | None = None
    ) -> np.ndarray:
        return self.eig(values, n_modes)[0]

    def modes(
        self, values: ArrayLike | None = None, n_modes: int | None = None
    ) -> np.ndarray:
        return self.eig(values, n_modes)[1]

    def response_function(self, n_modes: int | None = None):
        """Return ``f(p) -> freq_hz[:n_modes]`` suitable for the updater."""

        def _f(p):
            return self.frequencies(p, n_modes)

        return _f


# ----------------------------------------------------------------------
# 2-DOF fallback
# ----------------------------------------------------------------------
@dataclass
class TwoDOFModel(ReferenceModel):
    """Classic 2-DOF chain: ground -k1- m1 -k2- m2.

    Parameters are *multipliers* on ``k1`` and ``k2`` (nominal value 1.0), so a
    10 % stiffness error is simply ``[1.1, 1.0]``.
    """

    masses: tuple[float, float] = (1.0, 1.0)
    stiffnesses: tuple[float, float] = (1.0e4, 1.0e4)
    parameter_names: tuple[str, ...] = ("k1", "k2")

    @property
    def n_parameters(self) -> int:  # type: ignore[override]
        return 2

    def assemble(self, values: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        a = self._values(values)
        k1 = self.stiffnesses[0] * a[0]
        k2 = self.stiffnesses[1] * a[1]
        K = np.array([[k1 + k2, -k2], [-k2, k2]], dtype=float)
        M = np.diag(np.asarray(self.masses, dtype=float))
        return K, M

    def stiffness_derivatives(self) -> list[np.ndarray]:
        """Exact ``dK/dp`` (the model is linear in both parameters)."""
        k1, k2 = self.stiffnesses
        dK1 = np.array([[k1, 0.0], [0.0, 0.0]])
        dK2 = np.array([[k2, -k2], [-k2, k2]])
        return [dK1, dK2]

    def mass_derivatives(self) -> list[np.ndarray]:
        return [np.zeros((2, 2)), np.zeros((2, 2))]

    def analytical_frequencies(
        self, values: ArrayLike | None = None
    ) -> np.ndarray:  # pragma: no cover - identical to eig for 2x2
        a = self._values(values)
        k1 = self.stiffnesses[0] * a[0]
        k2 = self.stiffnesses[1] * a[1]
        m1, m2 = self.masses
        b = (k1 + k2) / m1 + k2 / m2
        c = (k1 * k2) / (m1 * m2)
        disc = math.sqrt(max(b * b - 4.0 * c, 0.0))
        w2 = np.array([(b - disc) / 2.0, (b + disc) / 2.0])
        return np.sqrt(np.clip(w2, 0.0, None)) / (2.0 * math.pi)


# ----------------------------------------------------------------------
# Axial rod
# ----------------------------------------------------------------------
@dataclass
class AxialBarModel(ReferenceModel):
    """Uniform axial rod discretised with 2-node bar elements (consistent mass).

    The rod is split into ``n_regions`` equal groups of elements; each group's
    Young's modulus is scaled by one parameter.
    """

    n_elem: int = 10
    length: float = 1.0
    area: float = 1.0e-4
    E: float = 210.0e9
    rho: float = 7850.0
    n_regions: int = 2
    bc: str = "fixed-free"
    parameter_names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.parameter_names:
            self.parameter_names = tuple(f"E{i + 1}" for i in range(self.n_regions))

    @property
    def n_parameters(self) -> int:  # type: ignore[override]
        return self.n_regions

    def _region_of(self, e: int) -> int:
        return min(int(e * self.n_regions / self.n_elem), self.n_regions - 1)

    def assemble(self, values: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        a = self._values(values)
        n = self.n_elem
        le = self.length / n
        ndof = n + 1
        K = np.zeros((ndof, ndof))
        M = np.zeros((ndof, ndof))
        for e in range(n):
            scale = a[self._region_of(e)]
            ke = self.E * scale * self.area / le * np.array([[1.0, -1.0], [-1.0, 1.0]])
            me = self.rho * self.area * le / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
            idx = [e, e + 1]
            K[np.ix_(idx, idx)] += ke
            M[np.ix_(idx, idx)] += me
        return self._apply_bc(K, M)

    def _apply_bc(self, K: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bc = self.bc.lower().replace("_", "-")
        if bc in ("free", "free-free"):
            return K, M
        if bc in ("fixed-free", "clamped-free", "cantilever"):
            keep = np.arange(1, K.shape[0])
        elif bc in ("fixed-fixed", "clamped-clamped"):
            keep = np.arange(1, K.shape[0] - 1)
        else:
            raise ValueError(f"unknown bc {self.bc!r}")
        return K[np.ix_(keep, keep)], M[np.ix_(keep, keep)]

    def stiffness_derivatives(self) -> list[np.ndarray]:
        """Exact ``dK/dp_j`` obtained by assembling only region ``j``."""
        out = []
        for j in range(self.n_regions):
            sel = np.zeros(self.n_regions)
            sel[j] = 1.0
            n = self.n_elem
            le = self.length / n
            ndof = n + 1
            K = np.zeros((ndof, ndof))
            for e in range(n):
                if self._region_of(e) != j:
                    continue
                ke = self.E * self.area / le * np.array([[1.0, -1.0], [-1.0, 1.0]])
                idx = [e, e + 1]
                K[np.ix_(idx, idx)] += ke
            Kb, _ = self._apply_bc(K, np.zeros_like(K))
            out.append(Kb)
        return out

    def mass_derivatives(self) -> list[np.ndarray]:
        K, M = self.assemble()
        return [np.zeros_like(M) for _ in range(self.n_regions)]


# ----------------------------------------------------------------------
# Euler-Bernoulli beam
# ----------------------------------------------------------------------
@dataclass
class BeamModel(ReferenceModel):
    """Planar Euler--Bernoulli beam (2 DOF/node: w, theta), consistent mass.

    Split into ``n_regions`` equal element groups; each group's ``E*I`` is
    scaled by one relative parameter.  This is the 2-parameter beam used by the
    acceptance case *"updating E on 2-param beam (10 % error) -> recover within
    2 %"*.
    """

    n_elem: int = 10
    length: float = 1.0
    E: float = 210.0e9
    I: float = 8.333333333333333e-10  # noqa: E741 - matches engineering notation
    rho: float = 7850.0
    area: float = 1.0e-4
    n_regions: int = 2
    bc: str = "cantilever"
    parameter_names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.parameter_names:
            self.parameter_names = tuple(f"E{i + 1}" for i in range(self.n_regions))

    @property
    def n_parameters(self) -> int:  # type: ignore[override]
        return self.n_regions

    def _region_of(self, e: int) -> int:
        return min(int(e * self.n_regions / self.n_elem), self.n_regions - 1)

    @staticmethod
    def _ke(EI: float, le: float) -> np.ndarray:
        c = EI / le**3
        return c * np.array(
            [
                [12.0, 6.0 * le, -12.0, 6.0 * le],
                [6.0 * le, 4.0 * le**2, -6.0 * le, 2.0 * le**2],
                [-12.0, -6.0 * le, 12.0, -6.0 * le],
                [6.0 * le, 2.0 * le**2, -6.0 * le, 4.0 * le**2],
            ]
        )

    @staticmethod
    def _me(rhoA: float, le: float) -> np.ndarray:
        c = rhoA * le / 420.0
        return c * np.array(
            [
                [156.0, 22.0 * le, 54.0, -13.0 * le],
                [22.0 * le, 4.0 * le**2, 13.0 * le, -3.0 * le**2],
                [54.0, 13.0 * le, 156.0, -22.0 * le],
                [-13.0 * le, -3.0 * le**2, -22.0 * le, 4.0 * le**2],
            ]
        )

    def _assemble_raw(self, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_elem
        le = self.length / n
        ndof = 2 * (n + 1)
        K = np.zeros((ndof, ndof))
        M = np.zeros((ndof, ndof))
        rhoA = self.rho * self.area
        for e in range(n):
            s = scales[self._region_of(e)]
            idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            K[np.ix_(idx, idx)] += self._ke(self.E * s * self.I, le)
            M[np.ix_(idx, idx)] += self._me(rhoA, le)
        return K, M

    def assemble(self, values: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        a = self._values(values)
        K, M = self._assemble_raw(a)
        return self._apply_bc(K, M)

    def _apply_bc(self, K: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bc = self.bc.lower().replace("_", "-")
        ndof = K.shape[0]
        if bc in ("free", "free-free"):
            keep = np.arange(ndof)
        elif bc in ("cantilever", "fixed-free", "clamped-free"):
            keep = np.arange(2, ndof)
        elif bc in ("simply-supported", "pinned-pinned", "ss"):
            keep = np.array([i for i in range(ndof) if i not in (0, ndof - 2)])
        elif bc in ("fixed-fixed", "clamped-clamped"):
            keep = np.arange(2, ndof - 2)
        else:
            raise ValueError(f"unknown bc {self.bc!r}")
        return K[np.ix_(keep, keep)], M[np.ix_(keep, keep)]

    def stiffness_derivatives(self) -> list[np.ndarray]:
        out = []
        for j in range(self.n_regions):
            sel = np.zeros(self.n_regions)
            sel[j] = 1.0
            K, M = self._assemble_raw(sel)
            Kb, _ = self._apply_bc(K, M)
            out.append(Kb)
        return out

    def mass_derivatives(self) -> list[np.ndarray]:
        K, M = self.assemble()
        return [np.zeros_like(M) for _ in range(self.n_regions)]


# ----------------------------------------------------------------------
# Analytical golden values
# ----------------------------------------------------------------------
def analytical_axial_frequencies(
    E: float, rho: float, length: float, n_modes: int = 3, bc: str = "fixed-free"
) -> np.ndarray:
    """Continuum rod natural frequencies [Hz]."""
    c = math.sqrt(E / rho)
    k = np.arange(1, n_modes + 1)
    bc = bc.lower().replace("_", "-")
    if bc in ("fixed-free", "clamped-free", "cantilever"):
        return (2 * k - 1) * c / (4.0 * length)
    if bc in ("fixed-fixed", "clamped-clamped", "free-free"):
        return k * c / (2.0 * length)
    raise ValueError(f"unknown bc {bc!r}")


_EB_BETA_L = {
    "cantilever": (1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839),
    "free-free": (4.73004074, 7.85320462, 10.99560784, 14.13716549, 17.27875966),
    "simply-supported": (math.pi, 2 * math.pi, 3 * math.pi, 4 * math.pi, 5 * math.pi),
    "fixed-fixed": (4.73004074, 7.85320462, 10.99560784, 14.13716549, 17.27875966),
}


def analytical_cantilever_frequencies(
    E: float,
    I: float,  # noqa: E741
    rho: float,
    area: float,
    length: float,
    n_modes: int = 3,
    bc: str = "cantilever",
) -> np.ndarray:
    """Euler--Bernoulli beam natural frequencies [Hz]."""
    key = bc.lower().replace("_", "-")
    if key in ("fixed-free", "clamped-free"):
        key = "cantilever"
    if key in ("pinned-pinned", "ss"):
        key = "simply-supported"
    betas = np.asarray(_EB_BETA_L[key][:n_modes], dtype=float)
    return (betas**2) / (2.0 * math.pi * length**2) * math.sqrt(E * I / (rho * area))


# ----------------------------------------------------------------------
def make_updating_testcase(
    kind: str = "beam",
    *,
    error: float | Sequence[float] = 0.10,
    n_modes: int = 4,
    noise: float = 0.0,
    seed: int = 0,
):
    """Build a canonical "wrong-stiffness" updating problem.

    Returns ``(response_fn, p_true, p_start, targets, model)`` where
    ``response_fn(p) -> freq_hz[:n_modes]``.  ``p_true`` are the multipliers of
    the *truth* model, ``p_start`` the (erroneous) starting values, all 1.0.

    Example: ``error=0.10`` makes the truth model 10 % stiffer in region 1, i.e.
    the analyst's model is 10 % wrong in ``E``.
    """
    kind = kind.lower()
    if kind in ("beam", "eb", "bending"):
        model: ReferenceModel = BeamModel(n_elem=10, n_regions=2)
    elif kind in ("bar", "rod", "axial"):
        model = AxialBarModel(n_elem=10, n_regions=2)
    elif kind in ("2dof", "two_dof", "twodof"):
        model = TwoDOFModel()
        n_modes = min(n_modes, 2)
    else:
        raise ValueError(f"unknown testcase {kind!r}")

    npar = model.n_parameters
    if np.isscalar(error):
        p_true = np.ones(npar)
        p_true[0] = 1.0 + float(error)  # type: ignore[arg-type]
    else:
        p_true = 1.0 + np.asarray(error, dtype=float)
    p_start = np.ones(npar)

    targets = model.frequencies(p_true, n_modes)
    if noise:
        rng = np.random.default_rng(seed)
        targets = targets * (1.0 + noise * rng.standard_normal(targets.shape))

    return model.response_function(n_modes), p_true, p_start, targets, model
