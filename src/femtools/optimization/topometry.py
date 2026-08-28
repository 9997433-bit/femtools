"""Topometry optimization: element-wise sizing on an existing mesh.

Topometry is the element-by-element sizing problem: every element of a mesh the
user already has keeps its geometry and its connectivity, and gets its **own**
design variable -- the thickness of a shell element, or a density (material)
scale factor for a solid.  It is the classical distinction drawn in the
topology-optimization literature (Bendsoe & Sigmund, *Topology Optimization*,
2nd ed., ch. 1; Sigmund & Maute, *Topology optimization approaches*, SMO 48,
2013): *topology* decides where material exists on a design domain the optimizer
meshes itself, *topometry* (also "free sizing" / "element-wise sizing") keeps the
structure and tunes how much material each existing element carries.

This module is therefore deliberately different from
:func:`femtools.optimization.topology.topology_simp`, which owns its structured
plane-stress Q4 grid and its own mini solver.  Here the analysis is the real
femtools FEA kernel (:func:`femtools.fea.static.solve_static`) applied to a real
:class:`~femtools.core.model.FEModel`, so shells, solids, beams, supports and
loads all behave exactly as they do in any other analysis.

.. math::
    \\min_{x}\\; c(x) = f^{T} u(x) \\quad\\text{s.t.}\\quad K(x)\\,u = f,\\;\\;
    \\sum_e m_e x_e \\le V^{*},\\;\\; x^{\\min} \\le x \\le x^{\\max}

with :math:`m_e` the element area (thickness design) or volume (density design).

Sensitivities
-------------
The compliance gradient is the standard self-adjoint result
:math:`\\partial c/\\partial x_e = -u_e^{T}\\,(\\partial K_e/\\partial x_e)\\,u_e`,
and both element derivatives are available in closed form:

* **thickness** -- a flat shell element is exactly cubic in the thickness,
  :math:`K_e(t) = t\\,A_e + t^{3} B_e` (membrane, transverse shear and the
  drilling penalty scale with :math:`t`, bending with :math:`t^3`), so
  :math:`A_e` and :math:`B_e` are recovered once from two element builds and
  :math:`\\partial K_e/\\partial t = A_e + 3 t^{2} B_e` is exact for the whole
  run.  The decomposition is verified against a third thickness and any element
  that fails the check falls back to a central difference of its own matrix.
* **density** -- with the modified SIMP interpolation
  :math:`E(x) = E_{\\min} + x^{p}\\,(E_0 - E_{\\min})` the element matrix is
  exactly proportional to the modulus, so
  :math:`\\partial K_e/\\partial x = p\\,x^{p-1}(E_0-E_{\\min})/E_0 \\cdot K_e^0`.

Because no node ever moves, no element can distort or invert; the result carries
:attr:`TopometryResult.min_size_ratio` (computed with
:func:`femtools.optimization.shape.element_size_ratios`) as the measured proof.

Documented subset
-----------------
Implemented: minimum compliance for one static load case, per-element thickness
(``QUAD4`` / ``TRIA3``) or density scale (any isotropic element), a linear
volume / mean-thickness constraint, the optimality-criteria update and SLSQP,
Sigmund's sensitivity or density filter generalised to an unstructured mesh via
element-centroid distances, and per-element bounds.

Not implemented here: stress or displacement constraints, multiple load cases,
frequency objectives, discrete (black/white) penalisation of thicknesses, and
composite ply-thickness ("free-size") design variables.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .shape import element_size_ratios
from .size import Constraint, size_optimize

__all__ = ["TopometryResult", "topometry_optimize"]

#: Element types whose design variable can be a thickness.
SHELL_ELEMENTS: frozenset[str] = frozenset(
    {"QUAD4", "CQUAD4", "QUAD", "Q4", "SHELL4", "TRIA3", "CTRIA3", "TRI3", "TRIA", "SHELL3"}
)

#: Element types whose design variable can only be a density scale.
SOLID_ELEMENTS: frozenset[str] = frozenset({"HEX8", "CHEXA", "HEXA", "TET4", "CTETRA", "TETRA"})

_SIGNS_HEX8 = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
    ]
)

_GAUSS2 = (-1.0 / math.sqrt(3.0), +1.0 / math.sqrt(3.0))


# ----------------------------------------------------------------------
# result
# ----------------------------------------------------------------------
@dataclass
class TopometryResult:
    """Outcome of :func:`topometry_optimize`.

    Attributes
    ----------
    x:
        The final design field the returned model carries, one value per
        designed element, ordered like :attr:`element_ids`: thicknesses for
        ``design="thickness"``, density scales in ``[x_min, 1]`` for
        ``design="density"``.  Under a density filter this is the *filtered*
        field; the raw design variables are kept in
        ``extras["design_variables"]``.
    compliance / initial_compliance:
        ``f^T u`` at the optimum and at the starting design.
    volume / initial_volume / volume_limit:
        Material volume :math:`\\sum_e m_e x_e` and the constraint value it was
        held against (an *upper* bound; the optimum sits on it).
    model:
        Deep copy of the input model carrying the optimal design -- one property
        (and, for the density design, one material) per designed element, so it
        can be solved, written out or post-processed like any other model.
    min_size_ratio:
        Smallest element size measure relative to the input mesh.  Topometry
        never moves a node, so this is 1 and certifies that no element has been
        distorted or inverted.
    history:
        Per-iteration ``{iteration, compliance, volume, change, ...}`` records.
    """

    x: np.ndarray
    design: str
    compliance: float
    initial_compliance: float
    volume: float
    initial_volume: float
    volume_limit: float
    element_ids: list[Any]
    measures: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    iterations: int = 0
    change: float = math.nan
    converged: bool = False
    method: str = "OC"
    message: str = ""
    model: Any = None
    displacement: np.ndarray | None = None
    strain_energy: np.ndarray | None = None
    min_size_ratio: float = 1.0
    n_fev: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.atleast_1d(np.asarray(self.x, dtype=float))
        self.measures = np.atleast_1d(np.asarray(self.measures, dtype=float))

    # -- array-ish conveniences -----------------------------------------
    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.array(self.x, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        return int(self.x.size)

    def __getitem__(self, key: Any) -> Any:
        """Positional access; use :meth:`to_dict` to look a design value up by id."""
        return self.x[key]

    @property
    def values(self) -> np.ndarray:
        return self.x

    @property
    def thickness(self) -> np.ndarray:
        """Element thicknesses (only meaningful for ``design="thickness"``)."""
        if self.design != "thickness":
            raise AttributeError("this result carries density scales, not thicknesses")
        return self.x

    @property
    def density(self) -> np.ndarray:
        """Element density scales (only for ``design="density"``)."""
        if self.design != "density":
            raise AttributeError("this result carries thicknesses, not density scales")
        return self.x

    @property
    def fun(self) -> float:
        return self.compliance

    @property
    def mean_thickness(self) -> float:
        """Area-weighted mean of the design field (``volume / total measure``)."""
        total = float(np.sum(self.measures))
        return float(self.volume / total) if total > 0.0 else math.nan

    @property
    def improvement(self) -> float:
        """Fractional compliance drop relative to the starting design."""
        if not math.isfinite(self.initial_compliance) or self.initial_compliance == 0.0:
            return math.nan
        return float((self.initial_compliance - self.compliance) / abs(self.initial_compliance))

    @property
    def feasible(self) -> bool:
        return bool(self.volume <= self.volume_limit * (1.0 + 1.0e-6))

    @property
    def compliance_history(self) -> np.ndarray:
        return np.array([rec["compliance"] for rec in self.history], dtype=float)

    def to_dict(self) -> dict[Any, float]:
        """``{element_id: design value}``."""
        return {eid: float(v) for eid, v in zip(self.element_ids, self.x, strict=True)}

    def summary(self) -> str:
        return "\n".join(
            [
                f"topometry_optimize({self.design}, {self.method}) -> "
                f"{'converged' if self.converged else 'stopped'} in "
                f"{self.iterations} iterations ({self.message})",
                f"  compliance {self.initial_compliance:.6g} -> {self.compliance:.6g} "
                f"({100.0 * self.improvement:+.3f} %)",
                f"  volume {self.initial_volume:.6g} -> {self.volume:.6g} "
                f"(limit {self.volume_limit:.6g})",
                f"  design range [{float(np.min(self.x)):.6g}, {float(np.max(self.x)):.6g}], "
                f"min element size ratio {self.min_size_ratio:.6g}",
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TopometryResult(design={self.design!r}, n={self.x.size}, "
            f"compliance={self.compliance:.6g}, volume={self.volume:.6g}, "
            f"iterations={self.iterations})"
        )


# ----------------------------------------------------------------------
# model access helpers (duck typed, like the rest of the optimization layer)
# ----------------------------------------------------------------------
def _records(container: Any) -> list[tuple[Any, Any]]:
    from femtools.fea.protocols import iter_records

    return list(iter_records(container))


def _get(record: Any, names: Sequence[str], default: Any = None) -> Any:
    from femtools.fea.protocols import get_any

    found = get_any(record, names, None)
    return default if found is None else found


def _set(record: Any, names: Sequence[str], value: Any) -> None:
    if isinstance(record, dict):
        for name in names:
            if name in record:
                record[name] = value
                return
        record[names[0]] = value
        return
    for name in names:
        if hasattr(record, name):
            setattr(record, name, value)
            return
    raise TypeError(f"cannot write {names[0]!r} onto record {record!r}")


def _next_id(container: Mapping[Any, Any]) -> int:
    ids = [k for k in container if isinstance(k, (int, np.integer))]
    return int(max(ids)) + 1 if ids else 1


def _element_type(record: Any) -> str:
    return str(_get(record, ("type", "etype", "element_type", "kind"), "")).strip().upper()


def _element_nodes(record: Any) -> tuple[Any, ...]:
    return tuple(_get(record, ("nodes", "node_ids", "connectivity", "conn", "grids"), ()) or ())


def _normalize_loads(model: Any, loads: Any) -> Any:
    """Turn ``model.loads`` records into the ``{(node, dof): value}`` form.

    :class:`femtools.core.model.Load` stores force and moment *vectors*, which
    the load builder does not read back; expanding them here keeps
    ``loads=None`` meaning "use the loads already on the model".  Every load set
    (``sid``) contributes, as it would in a solver run without a case control.
    """
    if loads is not None:
        return loads
    records = getattr(model, "loads", None)
    if not records:
        return None
    pairs: dict[tuple[Any, int], float] = {}
    for record in records:
        node = getattr(record, "node_id", None)
        expand = getattr(record, "as_dof_values", None)
        if node is None or expand is None:
            return None  # a foreign load record: let the FEA kernel interpret it
        for comp, value in expand():
            key = (node, int(comp))
            pairs[key] = pairs.get(key, 0.0) + float(value)
    return pairs or None


def _polygon_area(points: np.ndarray) -> float:
    """Area of a planar (or mildly warped) triangle / quadrilateral."""
    vec = 0.5 * np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    return float(np.linalg.norm(vec))


def _tet4_volume(points: np.ndarray) -> float:
    return abs(
        float(
            np.linalg.det(
                np.stack([points[1] - points[0], points[2] - points[0], points[3] - points[0]])
            )
        )
        / 6.0
    )


def _hex8_volume(points: np.ndarray) -> float:
    """Volume of a trilinear brick (2x2x2 Gauss, exact for the trilinear map)."""
    total = 0.0
    s = _SIGNS_HEX8
    for xi in _GAUSS2:
        for eta in _GAUSS2:
            for zeta in _GAUSS2:
                dn = np.empty((8, 3))
                dn[:, 0] = 0.125 * s[:, 0] * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
                dn[:, 1] = 0.125 * s[:, 1] * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 2] * zeta)
                dn[:, 2] = 0.125 * s[:, 2] * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta)
                total += abs(float(np.linalg.det(dn.T @ points)))
    return total


def _measure(etype: str, points: np.ndarray) -> float:
    """Area of a shell element, volume of a solid one."""
    if etype in SHELL_ELEMENTS:
        return _polygon_area(points)
    if etype in ("TET4", "CTETRA", "TETRA"):
        return _tet4_volume(points)
    if etype in ("HEX8", "CHEXA", "HEXA"):
        return _hex8_volume(points)
    raise ValueError(f"no size measure for element type {etype!r}")


# ----------------------------------------------------------------------
# design elements
# ----------------------------------------------------------------------
@dataclass
class _Design:
    """Everything the optimizer needs about one designed element."""

    eid: Any
    element: Any
    etype: str
    prop: Any
    material: Any
    measure: float
    centroid: np.ndarray
    t0: float = 0.0
    e0: float = 0.0
    g0: float = 0.0
    rho0: float = 0.0
    dof_pairs: list[tuple[Any, int]] = field(default_factory=list)
    dofs: np.ndarray | None = None
    k_lin: np.ndarray | None = None  # thickness: the part proportional to t
    k_cub: np.ndarray | None = None  # thickness: the part proportional to t^3
    k_unit: np.ndarray | None = None  # density: stiffness at the unscaled material
    exact: bool = True


def _filter_weights(centroids: np.ndarray, radius: float) -> tuple[sp.csr_matrix, np.ndarray]:
    """Sigmund's cone weights on an unstructured mesh (element-centroid distance)."""
    n = centroids.shape[0]
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for i in range(n):
        d = np.linalg.norm(centroids - centroids[i], axis=1)
        near = np.flatnonzero(d < radius)
        rows.append(np.full(near.size, i))
        cols.append(near)
        vals.append(radius - d[near])
    H = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    return H, np.asarray(H.sum(axis=1)).ravel()


# ----------------------------------------------------------------------
class _Problem:
    """The compliance / volume model behind both optimizers."""

    def __init__(
        self,
        model: Any,
        design: str,
        elements: Sequence[Any] | None,
        loads: Any,
        penal: float,
        emin_ratio: float,
        solver_kwargs: dict[str, Any],
    ) -> None:
        from femtools.fea.elements import ModelIndex, element_matrices

        self.reference = model
        self.work = copy.deepcopy(model)
        self.loads = _normalize_loads(self.work, loads)
        self.penal = float(penal)
        self.emin = float(emin_ratio)
        self.solver_kwargs = dict(solver_kwargs)
        self.n_fev = 0

        self.design_kind = _resolve_design_kind(self.work, design, elements)
        self.design = _collect_design_elements(self.work, self.design_kind, elements)
        if not self.design:
            raise ValueError(
                f"no designable elements found for design={self.design_kind!r}; "
                "the model must contain shell elements (thickness) or elements with an "
                "isotropic material (density)"
            )

        _split_entities(self.work, self.design, self.design_kind)

        # The property / material split added records, so the element index has
        # to be built after it.
        index = ModelIndex.build(self.work)
        build_kwargs = {
            key: self.solver_kwargs[key]
            for key in ("drill_factor", "options", "lumped_mass")
            if key in self.solver_kwargs
        }

        def stiffness(de: _Design) -> np.ndarray:
            em = element_matrices(self.work, de.eid, de.element, index=index, **build_kwargs)
            if em.k is None:  # pragma: no cover - defensive
                raise ValueError(f"element {de.eid} ({de.etype}) has no stiffness matrix")
            if not de.dof_pairs:
                de.dof_pairs = list(em.dofs)
            return np.asarray(em.k, dtype=float)

        self._stiffness = stiffness
        for de in self.design:
            if self.design_kind == "thickness":
                _decompose_thickness(de, stiffness)
            else:
                de.k_unit = stiffness(de)

        self.measures = np.array([de.measure for de in self.design], dtype=float)
        self.centroids = np.array([de.centroid for de in self.design], dtype=float)
        self.filter: sp.csr_matrix | None = None
        self.filter_sums: np.ndarray | None = None
        self.filter_kind = "none"
        self._cache_key: bytes | None = None
        self._cache: dict[str, Any] = {}

    # -- filtering -------------------------------------------------------
    def set_filter(self, radius: float, kind: str) -> None:
        if radius <= 0.0 or kind.lower().startswith("none"):
            return
        name = kind.lower()
        if name.startswith("sens"):
            self.filter_kind = "sensitivity"
        elif name.startswith("dens"):
            self.filter_kind = "density"
        else:
            raise ValueError(f"unknown filter {kind!r}; expected 'sensitivity' or 'density'")
        self.filter, self.filter_sums = _filter_weights(self.centroids, float(radius))

    def physical(self, x: np.ndarray) -> np.ndarray:
        """Design variables -> the field the analysis actually sees."""
        if self.filter is None or self.filter_kind != "density":
            return np.asarray(x, dtype=float)
        assert self.filter_sums is not None
        return np.asarray(self.filter @ np.asarray(x, dtype=float)).ravel() / self.filter_sums

    # -- the model ---------------------------------------------------------
    def write(self, xphys: np.ndarray) -> None:
        for de, value in zip(self.design, xphys, strict=True):
            if self.design_kind == "thickness":
                _set(de.prop, ("t", "thickness", "T", "h"), float(value))
            else:
                scale = self.emin + float(value) ** self.penal * (1.0 - self.emin)
                _set(de.material, ("E", "e"), de.e0 * scale)
                if de.g0:
                    _set(de.material, ("G", "g"), de.g0 * scale)
                if de.rho0:
                    _set(de.material, ("rho", "density"), de.rho0 * float(value))

    # -- analysis -----------------------------------------------------------
    def analyse(self, x: np.ndarray) -> dict[str, Any]:
        from femtools.fea.static import solve_static

        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        if self._cache_key == key:
            return self._cache

        xphys = self.physical(x)
        self.write(xphys)
        result: Any = solve_static(
            self.work, self.loads, full_result=True, **self.solver_kwargs
        )
        self.n_fev += 1
        u = np.asarray(result.u, dtype=float)
        if u.ndim != 1:
            raise ValueError(
                "topometry_optimize solves a single static load case; "
                "pass one load vector, not a matrix of load cases"
            )
        load = np.asarray(result.load, dtype=float).ravel()
        compliance = float(u @ load)
        assembly = result.assembly
        if self.design[0].dofs is None:
            for de in self.design:
                de.dofs = np.array(
                    [assembly.dof_map.index(nid, comp) for nid, comp in de.dof_pairs], dtype=int
                )
        u_basic = np.asarray(assembly.to_basic(u), dtype=float)

        n = len(self.design)
        dc = np.empty(n)
        energy = np.empty(n)
        for i, de in enumerate(self.design):
            ue = u_basic[de.dofs]
            if self.design_kind == "thickness":
                t = float(xphys[i])
                dk = _thickness_derivative(de, t, self._stiffness)
                ke = _thickness_stiffness(de, t, self._stiffness)
                dc[i] = -float(ue @ dk @ ue)
                energy[i] = 0.5 * float(ue @ ke @ ue)
            else:
                assert de.k_unit is not None
                base = float(ue @ de.k_unit @ ue)
                xe = max(float(xphys[i]), 1.0e-300)
                scale = self.emin + xe**self.penal * (1.0 - self.emin)
                dc[i] = -self.penal * xe ** (self.penal - 1.0) * (1.0 - self.emin) * base
                energy[i] = 0.5 * scale * base

        dv = self.measures.copy()
        if self.filter is not None and self.filter_kind == "density":
            assert self.filter_sums is not None
            dc = np.asarray(self.filter @ (dc / self.filter_sums)).ravel()
            dv = np.asarray(self.filter @ (dv / self.filter_sums)).ravel()
        elif self.filter is not None and self.filter_kind == "sensitivity":
            assert self.filter_sums is not None
            eps = 1.0e-9 * float(np.max(np.abs(x))) + 1.0e-300
            dc = np.asarray(self.filter @ (x * dc)).ravel() / (
                self.filter_sums * np.maximum(x, eps)
            )

        self._cache_key = key
        self._cache = {
            "x": x,
            "xphys": xphys,
            "compliance": compliance,
            "dc": dc,
            "dv": dv,
            "u": u,
            "volume": float(self.measures @ xphys),
            "strain_energy": energy,
        }
        return self._cache

    # -- objective / constraint faces used by the SLSQP driver ------------
    def compliance(self, x: np.ndarray) -> float:
        return float(self.analyse(x)["compliance"])

    def compliance_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.analyse(x)["dc"], dtype=float)

    def volume(self, x: np.ndarray) -> float:
        return float(self.measures @ self.physical(x))

    def volume_gradient(self, x: np.ndarray) -> np.ndarray:
        if self.filter is None or self.filter_kind != "density":
            return self.measures.copy()
        assert self.filter_sums is not None
        return np.asarray(self.filter @ (self.measures / self.filter_sums)).ravel()


def _resolve_design_kind(model: Any, design: str, elements: Sequence[Any] | None) -> str:
    name = str(design).strip().lower()
    if name in ("thickness", "t", "shell", "size", "sizing"):
        return "thickness"
    if name in ("density", "rho", "simp", "material", "solid"):
        return "density"
    if name not in ("auto", ""):
        raise ValueError(f"unknown design {design!r}; expected 'thickness', 'density' or 'auto'")
    wanted = None if elements is None else set(elements)
    for eid, record in _records(_get(model, ("elements", "elems", "element"))):
        if wanted is not None and eid not in wanted:
            continue
        if _element_type(record) in SHELL_ELEMENTS:
            return "thickness"
    return "density"


def _collect_design_elements(
    model: Any, kind: str, elements: Sequence[Any] | None
) -> list[_Design]:
    nodes = dict(_records(_get(model, ("nodes", "grids", "points"))))
    properties = dict(_records(_get(model, ("properties", "props", "property"))))
    materials = dict(_records(_get(model, ("materials", "mats", "material"))))
    all_elements = _records(_get(model, ("elements", "elems", "element")))
    selected = None
    if elements is not None:
        known = {eid for eid, _ in all_elements}
        missing = [e for e in elements if e not in known]
        if missing:
            raise KeyError(f"element(s) {missing} are not in the model")
        selected = set(elements)

    from femtools.fea.protocols import node_xyz

    out: list[_Design] = []
    for eid, record in all_elements:
        if selected is not None and eid not in selected:
            continue
        etype = _element_type(record)
        if kind == "thickness" and etype not in SHELL_ELEMENTS:
            continue
        if kind == "density" and etype not in (SHELL_ELEMENTS | SOLID_ELEMENTS):
            continue
        conn = _element_nodes(record)
        if len(conn) < 3:
            continue
        points = np.array([node_xyz(nodes[nid]) for nid in conn], dtype=float)
        pid = _get(record, ("property_id", "pid", "property"))
        prop = properties.get(pid)
        if prop is None and len(properties) == 1:
            prop = next(iter(properties.values()))
        if prop is None:
            raise ValueError(f"element {eid}: no property to size (property_id={pid!r})")
        mid = _get(prop, ("material_id", "mid", "material", "mat_id"))
        material = materials.get(mid)
        if material is None and len(materials) == 1:
            material = next(iter(materials.values()))
        if material is None:
            raise ValueError(f"element {eid}: no material found through property {pid!r}")
        de = _Design(
            eid=eid,
            element=record,
            etype=etype,
            prop=prop,
            material=material,
            measure=_measure(etype, points),
            centroid=points.mean(axis=0),
        )
        if kind == "thickness":
            t0 = _get(prop, ("t", "thickness", "T", "h"))
            if t0 is None or float(t0) <= 0.0:
                raise ValueError(
                    f"element {eid} ({etype}): a thickness design variable needs a positive "
                    f"thickness on property {pid!r}, found {t0!r}"
                )
            de.t0 = float(t0)
        else:
            e0 = _get(material, ("E", "e", "E1"))
            if e0 is None or float(e0) <= 0.0:
                raise ValueError(
                    f"element {eid}: a density design variable needs a positive Young's "
                    f"modulus on material {mid!r}, found {e0!r}"
                )
            de.e0 = float(e0)
            de.g0 = float(_get(material, ("G", "g"), 0.0) or 0.0)
            de.rho0 = float(_get(material, ("rho", "density"), 0.0) or 0.0)
        out.append(de)
    return out


def _split_entities(model: Any, design: list[_Design], kind: str) -> None:
    """Give every designed element a private property (and material).

    A property is normally shared by a whole panel; element-wise sizing needs one
    record per element, so the shared ones are cloned.  Elements outside the
    design set keep the originals untouched.
    """
    properties = _get(model, ("properties", "props", "property"))
    materials = _get(model, ("materials", "mats", "material"))
    if not isinstance(properties, dict):
        raise TypeError("topometry_optimize needs a model whose `properties` is a mapping")
    if kind == "density" and not isinstance(materials, dict):
        raise TypeError("topometry_optimize needs a model whose `materials` is a mapping")

    next_pid = _next_id(properties)
    next_mid = _next_id(materials) if isinstance(materials, dict) else 1
    for de in design:
        if kind == "density":
            material = copy.deepcopy(de.material)
            mid, next_mid = next_mid, next_mid + 1
            _set(material, ("id", "mid"), mid)
            materials[mid] = material
            de.material = material
        prop = copy.deepcopy(de.prop)
        pid, next_pid = next_pid, next_pid + 1
        _set(prop, ("id", "pid"), pid)
        if kind == "density":
            _set(prop, ("material_id", "mid", "material", "mat_id"), _get(de.material, ("id",)))
        properties[pid] = prop
        de.prop = prop
        _set(de.element, ("property_id", "pid", "property"), pid)


def _decompose_thickness(de: _Design, stiffness: Callable[[_Design], np.ndarray]) -> None:
    """Recover ``K_e(t) = t A + t^3 B`` from two element builds, then verify it."""
    t_ref = de.t0
    t1, t2, t3 = t_ref, 2.0 * t_ref, 1.5 * t_ref
    _set(de.prop, ("t", "thickness", "T", "h"), t1)
    k1 = stiffness(de)
    _set(de.prop, ("t", "thickness", "T", "h"), t2)
    k2 = stiffness(de)
    b = (k2 / t2 - k1 / t1) / (t2**2 - t1**2)
    a = k1 / t1 - t1**2 * b
    _set(de.prop, ("t", "thickness", "T", "h"), t3)
    k3 = stiffness(de)
    scale = float(np.max(np.abs(k3))) or 1.0
    de.exact = bool(np.max(np.abs(t3 * a + t3**3 * b - k3)) <= 1.0e-9 * scale)
    de.k_lin, de.k_cub = a, b
    _set(de.prop, ("t", "thickness", "T", "h"), t_ref)


def _thickness_stiffness(
    de: _Design, t: float, stiffness: Callable[[_Design], np.ndarray]
) -> np.ndarray:
    if de.exact:
        assert de.k_lin is not None and de.k_cub is not None
        return t * de.k_lin + t**3 * de.k_cub
    _set(de.prop, ("t", "thickness", "T", "h"), t)
    return stiffness(de)


def _thickness_derivative(
    de: _Design, t: float, stiffness: Callable[[_Design], np.ndarray]
) -> np.ndarray:
    if de.exact:
        assert de.k_lin is not None and de.k_cub is not None
        return de.k_lin + 3.0 * t * t * de.k_cub
    h = 1.0e-6 * t
    plus = _thickness_stiffness(de, t + h, stiffness)
    minus = _thickness_stiffness(de, t - h, stiffness)
    _set(de.prop, ("t", "thickness", "T", "h"), t)
    return (plus - minus) / (2.0 * h)


# ----------------------------------------------------------------------
# bounds, starting point and the volume constraint
# ----------------------------------------------------------------------
def _resolve_bounds(
    problem: _Problem, bounds: Any, x_min: float | None, x_max: float | None
) -> tuple[np.ndarray, np.ndarray]:
    n = len(problem.design)
    if bounds is None:
        if problem.design_kind == "thickness":
            base = np.array([de.t0 for de in problem.design], dtype=float)
            lo = (0.2 if x_min is None else float(x_min)) * base
            hi = (5.0 if x_max is None else float(x_max)) * base
        else:
            lo = np.full(n, 1.0e-3 if x_min is None else float(x_min))
            hi = np.full(n, 1.0 if x_max is None else float(x_max))
        return lo, hi

    if isinstance(bounds, Mapping):
        lo = np.empty(n)
        hi = np.empty(n)
        for i, de in enumerate(problem.design):
            pair = bounds.get(de.eid)
            if pair is None:
                raise KeyError(f"bounds mapping has no entry for element {de.eid}")
            lo[i], hi[i] = float(pair[0]), float(pair[1])
        return lo, hi

    seq = list(bounds)
    if len(seq) == 2 and all(np.isscalar(b) for b in seq):
        return np.full(n, float(seq[0])), np.full(n, float(seq[1]))
    if len(seq) != n:
        raise ValueError(f"bounds has {len(seq)} entries but {n} elements are designed")
    lo = np.array([float(b[0]) for b in seq], dtype=float)
    hi = np.array([float(b[1]) for b in seq], dtype=float)
    return lo, hi


def _resolve_x0(
    problem: _Problem,
    x0: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    volume_fraction: float | None,
) -> np.ndarray:
    n = len(problem.design)
    if x0 is None:
        if problem.design_kind == "thickness":
            start = np.array([de.t0 for de in problem.design], dtype=float)
        else:
            # The classic SIMP start: uniform material at the allowed fraction,
            # which is feasible, so the reported improvement compares two designs
            # holding the same amount of material.
            start = np.full(n, 1.0 if volume_fraction is None else float(volume_fraction))
    elif np.isscalar(x0):
        start = np.full(n, float(x0))  # type: ignore[arg-type]
    else:
        start = np.asarray(x0, dtype=float).ravel()
        if start.size != n:
            raise ValueError(f"x0 has {start.size} entries but {n} elements are designed")
    return np.clip(start, lo, hi)


def _resolve_limit(
    problem: _Problem,
    x0: np.ndarray,
    hi: np.ndarray,
    volume_fraction: float | None,
    max_volume: float | None,
    mean_thickness: float | None,
) -> tuple[float, str]:
    given = [
        name
        for name, value in (
            ("volume_fraction", volume_fraction),
            ("max_volume", max_volume),
            ("mean_thickness", mean_thickness),
        )
        if value is not None
    ]
    if len(given) > 1:
        raise ValueError(f"pass at most one volume constraint, got {given}")
    measures = problem.measures
    if max_volume is not None:
        return float(max_volume), "max_volume"
    if volume_fraction is not None:
        return float(volume_fraction) * float(measures @ hi), "volume_fraction"
    if mean_thickness is not None:
        if problem.design_kind != "thickness":
            raise ValueError("`mean_thickness` only applies to the thickness design")
        return float(mean_thickness) * float(np.sum(measures)), "mean_thickness"
    return float(measures @ problem.physical(x0)), "initial_volume"


# ----------------------------------------------------------------------
# optimality criteria
# ----------------------------------------------------------------------
def _oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    limit: float,
    measures: np.ndarray,
    physical: Callable[[np.ndarray], np.ndarray],
    *,
    move: float,
    damping: float,
) -> np.ndarray:
    """One optimality-criteria step, with bisection on the Lagrange multiplier.

    The update is the classical :math:`x \\leftarrow x\\,B_e^{\\eta}` with
    :math:`B_e = -\\partial c/\\partial x_e \\big/ \\lambda\\,\\partial v/\\partial x_e`
    (Bendsoe & Sigmund), clipped to the move limit and the bounds.  The volume
    is a monotone decreasing function of :math:`\\lambda`, so the multiplier is
    bracketed by geometric expansion and then bisected -- which keeps the update
    scale free, i.e. usable for thicknesses in metres as well as for densities
    in ``[0, 1]``.
    """
    step = move * (hi - lo)
    safe_dv = np.where(np.abs(dv) > 0.0, dv, 1.0e-30)

    def design(lam: float) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            be = np.clip(-dc / (lam * safe_dv), 0.0, 1.0e12)
        return np.clip(x * be**damping, np.maximum(lo, x - step), np.minimum(hi, x + step))

    def volume(lam: float) -> float:
        return float(measures @ physical(design(lam)))

    ratios = -dc / safe_dv
    positive = ratios[ratios > 0.0]
    lam = float(np.median(positive)) if positive.size else 1.0
    l1, l2 = lam, lam
    for _ in range(200):
        if volume(l1) > limit:
            break
        l1 *= 0.5
        if l1 < 1.0e-300:
            break
    for _ in range(200):
        if volume(l2) < limit:
            break
        l2 *= 2.0
        if l2 > 1.0e300:
            break
    for _ in range(200):
        if (l2 - l1) <= 1.0e-12 * (l1 + l2):
            break
        mid = 0.5 * (l1 + l2)
        if volume(mid) > limit:
            l1 = mid
        else:
            l2 = mid
    return design(0.5 * (l1 + l2))


# ----------------------------------------------------------------------
def topometry_optimize(
    model: Any,
    *,
    design: str = "auto",
    elements: Sequence[Any] | None = None,
    loads: Any = None,
    objective: str = "compliance",
    volume_fraction: float | None = None,
    max_volume: float | None = None,
    mean_thickness: float | None = None,
    bounds: Any = None,
    x_min: float | None = None,
    x_max: float | None = None,
    x0: Any = None,
    penal: float = 3.0,
    emin_ratio: float = 1.0e-9,
    method: str = "oc",
    move: float = 0.2,
    damping: float = 0.5,
    max_iter: int = 100,
    tol: float = 1.0e-2,
    objective_tol: float = 1.0e-6,
    filter_radius: float = 0.0,
    filter: str = "sensitivity",  # noqa: A002 - the literature's name for it
    solver_kwargs: dict[str, Any] | None = None,
    callback: Callable[[int, np.ndarray, float], None] | None = None,
    verbose: bool = False,
) -> TopometryResult:
    """Element-wise sizing of an existing mesh for minimum compliance.

    Every designed element gets one design variable -- its own thickness
    (``design="thickness"``, shells) or its own density scale
    (``design="density"``, SIMP-interpolated modulus) -- and the material is
    redistributed over the mesh until the compliance ``f^T u`` is minimal for a
    given amount of material.  Unlike
    :func:`femtools.optimization.topology.topology_simp`, which builds and
    solves its own structured Q4 grid, this works on the model the caller
    already has and analyses it with the standard FEA kernel.

    Parameters
    ----------
    model:
        Any model the FEA kernel accepts (:class:`femtools.core.model.FEModel`
        or a duck-typed equivalent).  It is never modified: the optimizer works
        on a deep copy, which is returned in :attr:`TopometryResult.model`.
    design:
        ``"thickness"``, ``"density"``, or ``"auto"`` (default: thickness when
        the selection contains a shell element, density otherwise).
    elements:
        Ids of the elements to design.  ``None`` designs every element the
        chosen design kind applies to; everything else stays frozen.
    loads:
        Load in any form :func:`femtools.fea.loads.build_load_vector` accepts.
        ``None`` uses the model's own loads.  One load case.
    volume_fraction, max_volume, mean_thickness:
        The (single) material constraint, as a fraction of the volume at the
        upper bound, an absolute volume, or an area-weighted mean thickness.
        With none of them given the *starting* volume is kept, i.e. the material
        already in the model is redistributed.
    bounds:
        ``(lo, hi)`` for every element, a per-element sequence of pairs, or a
        ``{element_id: (lo, hi)}`` mapping.  The default keeps a thickness in
        ``[0.2 t_0, 5 t_0]`` and a density in ``[1e-3, 1]``; ``x_min`` / ``x_max``
        rescale those defaults (as factors of ``t_0`` for the thickness design).
    penal:
        SIMP exponent for the density design.  Ignored by the thickness design,
        which is genuine sizing and needs no penalisation.
    method:
        ``"oc"`` (default) -- the optimality-criteria fixed point, which handles
        thousands of variables under one linear constraint -- or ``"slsqp"``,
        which runs :func:`femtools.optimization.size_optimize` with the same
        analytic gradients.
    move, damping:
        OC move limit (as a fraction of each variable's range) and the exponent
        :math:`\\eta` of the update.
    max_iter, tol, objective_tol:
        The OC iteration stops when the largest design change falls below
        ``tol`` (as a fraction of each variable's range -- the same measure and
        the same default as :func:`topology_simp`), or when a feasible design
        has changed the compliance by less than ``objective_tol`` relatively
        for two iterations in a row.  A minimum-compliance sizing problem with
        several near-equivalent optima can keep shuffling a few elements long
        after the compliance has settled, so running out of iterations is a
        normal, and reported, outcome.
    filter_radius, filter:
        Radius (in model length units) and kind of Sigmund's mesh-independence
        filter, applied over element-centroid distances: ``"sensitivity"``
        (default) or ``"density"``.  ``filter_radius=0`` (default) disables it;
        element-wise sizing does not checkerboard the way a density topology
        does, so it is off unless asked for.
    callback:
        ``callback(iteration, x, compliance)`` after every OC iteration.

    Returns
    -------
    TopometryResult

    Examples
    --------
    Redistribute the material of a uniform cantilever plate::

        res = topometry_optimize(plate, loads={(tip, "uz"): -1.0e3})
        assert res.compliance < res.initial_compliance
        assert res.min_size_ratio == 1.0     # no element moved, none inverted
    """
    if str(objective).strip().lower() not in ("compliance", "strain-energy", "strain_energy"):
        raise ValueError(
            f"unknown objective {objective!r}; topometry_optimize minimises 'compliance'"
        )
    method_name = str(method).strip().lower()
    if method_name not in ("oc", "optimality-criteria", "slsqp"):
        raise ValueError(f"unknown method {method!r}; expected 'oc' or 'slsqp'")

    problem = _Problem(
        model,
        design=design,
        elements=elements,
        loads=loads,
        penal=penal,
        emin_ratio=emin_ratio,
        solver_kwargs=dict(solver_kwargs or {}),
    )
    problem.set_filter(filter_radius, filter)

    lo, hi = _resolve_bounds(problem, bounds, x_min, x_max)
    if np.any(lo <= 0.0) or np.any(hi <= lo):
        raise ValueError("design bounds must satisfy 0 < lower < upper for every element")
    x = _resolve_x0(problem, x0, lo, hi, volume_fraction)
    limit, limit_kind = _resolve_limit(
        problem, x, hi, volume_fraction, max_volume, mean_thickness
    )

    start = problem.analyse(x)
    if abs(start["compliance"]) <= 0.0:
        raise ValueError(
            "the compliance of the starting design is zero: the model carries no load, "
            "so there is nothing to optimize"
        )
    c0 = float(start["compliance"])
    v0 = float(start["volume"])

    history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "compliance": c0,
            "volume": v0,
            "change": math.nan,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }
    ]
    converged = False
    message = "maximum iterations reached"
    change = math.nan
    it = 0

    if method_name == "slsqp":
        con = Constraint(
            fun=lambda z: np.array([limit - problem.volume(z)]),
            jac=lambda z: -problem.volume_gradient(z)[None, :],
            type="ineq",
            name="volume",
        )
        res = size_optimize(
            problem.compliance,
            x,
            bounds=list(zip(lo, hi, strict=True)),
            constraints=[con],
            jac=problem.compliance_gradient,
            method="SLSQP",
            max_iter=int(max_iter),
            tol=float(tol) * 1.0e-3,
            scaling="auto",
            keep_history=False,
        )
        x = np.clip(np.asarray(res.x, dtype=float), lo, hi)
        converged = bool(res.success)
        message = res.message
        it = int(res.n_iter)
        state = problem.analyse(x)
        history.append(
            {
                "iteration": it,
                "compliance": float(state["compliance"]),
                "volume": float(state["volume"]),
                "change": math.nan,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }
        )
    else:
        stagnant = 0
        for it in range(1, int(max_iter) + 1):
            state = problem.analyse(x)
            previous = float(state["compliance"])
            x_new = _oc_update(
                x,
                np.asarray(state["dc"], dtype=float),
                np.asarray(state["dv"], dtype=float),
                lo,
                hi,
                limit,
                problem.measures,
                problem.physical,
                move=move,
                damping=damping,
            )
            change = float(np.max(np.abs(x_new - x) / (hi - lo)))
            x = x_new
            updated = problem.analyse(x)
            record = {
                "iteration": it,
                "compliance": float(updated["compliance"]),
                "volume": float(updated["volume"]),
                "change": change,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
            }
            history.append(record)
            if verbose:
                print(
                    f"[topometry] it={it:3d} c={record['compliance']:.6e} "
                    f"v={record['volume']:.6e} change={change:.3e}"
                )
            if callback is not None:
                callback(it, x.copy(), record["compliance"])
            if change < tol:
                converged = True
                message = f"design change below tol ({tol:g})"
                break
            # The OC fixed point can spend many iterations shuffling a few
            # elements around a design whose compliance no longer moves; a
            # feasible design that has stopped paying for the shuffling is
            # converged for every practical purpose.
            drop = abs(record["compliance"] - previous) / max(abs(previous), 1.0e-300)
            stagnant = stagnant + 1 if drop < objective_tol else 0
            if stagnant >= 2 and record["volume"] <= limit * (1.0 + 1.0e-6):
                converged = True
                message = f"compliance stagnated (< {objective_tol:g} per iteration)"
                break

    final = problem.analyse(x)
    ratios = element_size_ratios(problem.work, model)
    return TopometryResult(
        # The physical field is what the returned model carries; without a
        # density filter it *is* the design variable vector.
        x=np.asarray(final["xphys"], dtype=float),
        design=problem.design_kind,
        compliance=float(final["compliance"]),
        initial_compliance=c0,
        volume=float(final["volume"]),
        initial_volume=v0,
        volume_limit=limit,
        element_ids=[de.eid for de in problem.design],
        measures=problem.measures,
        bounds=(lo, hi),
        iterations=it,
        change=change,
        converged=converged,
        method="OC" if method_name != "slsqp" else "SLSQP",
        message=message,
        model=problem.work,
        displacement=np.asarray(final["u"], dtype=float),
        strain_energy=np.asarray(final["strain_energy"], dtype=float),
        min_size_ratio=float(np.min(ratios)) if ratios.size else 1.0,
        n_fev=problem.n_fev,
        history=history,
        extras={
            "design_variables": x,
            "constraint": limit_kind,
            "penal": float(penal),
            "filter": problem.filter_kind,
            "filter_radius": float(filter_radius),
            "exact_sensitivities": all(de.exact for de in problem.design),
        },
    )
