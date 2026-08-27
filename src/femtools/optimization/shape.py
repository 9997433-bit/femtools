"""Shape optimization: nodal coordinates as design variables.

Classical FE shape optimization (Haftka & Grandhi, *Structural shape
optimization -- a survey*, CMAME 57, 1986; Haftka & Gürdal, *Elements of
Structural Optimization*, ch. 7) moves selected grid points of an existing mesh
and re-analyses the structure at every design point:

.. math::
    \\min_{\\delta}\\; f\\big(x_0 + \\delta\\big) \\quad\\text{s.t.}\\quad
    |\\delta| \\le \\delta_{max},\\;\\; q(x_0 + \\delta) \\ge q_{min} ,

with :math:`\\delta` the coordinate offsets of the moved nodes.  The two
objectives implemented here are the two that shape optimization is classically
used for: **maximise a natural frequency** (drive a mode out of an excitation
band) and **minimise compliance** (stiffest structure for a given load).

The characteristic difficulty of shape optimization is not the optimiser but
the mesh: a search that is free to move nodes will happily collapse or invert
elements, and the analysis then returns a meaningless -- often *better* --
objective.  Two barriers guard against it, either or both of which can be
active:

``min_quality``
    An inequality constraint on the smallest element size ratio
    :math:`\\min_e\\, m_e(x)/m_e(x_0)`, where :math:`m_e` is the element length
    (bars/beams), the signed area (shells, taken about the initial normal so a
    fold shows up as a sign change) or the signed volume / corner Jacobian
    (solids).  A collapsing element drives the ratio to zero and an inverted one
    makes it negative, so a single scalar constraint covers both failures.

``smoothing``
    A Laplacian regulariser :math:`\\sum_i \\|x_i - \\bar{x}_{N(i)}\\|^2 / h^2`
    added to the objective, i.e. the classic mesh-smoothing energy used as a
    shape regulariser.  It keeps a moved node near the centroid of its
    neighbours and thereby suppresses the saw-tooth boundaries that unregularised
    node-based shape optimization is known for.

Documented subset
-----------------
Implemented: nodal coordinates (any subset of components) as design variables,
frequency / compliance / user-callable objectives, SLSQP and trust-constr, move
limits, the mesh-quality constraint and the Laplacian regulariser, plus
arbitrary extra user constraints.

Not implemented here: CAD-parameter or basis-vector (morphing) shape
parameterisations, analytic shape sensitivities (the material-derivative
approach), and automatic remeshing.  Gradients are finite differences of the
re-analysed model, which is why the move limits and the quality barrier matter.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .size import Constraint, size_optimize

__all__ = ["ShapeResult", "shape_optimize", "element_size_ratios"]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
_AXIS_NAME = ("x", "y", "z")

#: Node triples whose edge vectors form the corner Jacobian of a HEX8 corner
#: (the standard "verdict" corner ordering: for every node, the three edges
#: leaving it, ordered so that a right-handed brick has positive determinants).
_HEX8_CORNERS = (
    (0, 1, 3, 4),
    (1, 2, 0, 5),
    (2, 3, 1, 6),
    (3, 0, 2, 7),
    (4, 7, 5, 0),
    (5, 4, 6, 1),
    (6, 5, 7, 2),
    (7, 6, 4, 3),
)

_LINE_ELEMENTS = frozenset({"BAR2", "BEAM2", "TRUSS2D", "ROD", "CBAR", "CBEAM"})
_SHELL_ELEMENTS = frozenset({"TRIA3", "QUAD4", "TRI3", "CTRIA3", "CQUAD4"})
_IGNORED_ELEMENTS = frozenset({"MASS", "SPRING", "DAMPER", "CONM2", "CELAS", "CDAMP"})


# ----------------------------------------------------------------------
# model access (duck typed, exactly like the FEA kernel)
# ----------------------------------------------------------------------
def _node_container(model: Any) -> Any:
    nodes = getattr(model, "nodes", None)
    if nodes is None and isinstance(model, Mapping):
        nodes = model.get("nodes")
    if nodes is None:
        raise TypeError("model has no `nodes` container")
    return nodes


def _element_records(model: Any) -> list[tuple[Any, Any]]:
    from femtools.fea.protocols import iter_records

    elements = getattr(model, "elements", None)
    if elements is None and isinstance(model, Mapping):
        elements = model.get("elements")
    return list(iter_records(elements))


def _read_xyz(model: Any, node_id: Any) -> np.ndarray:
    from femtools.fea.protocols import node_xyz

    return node_xyz(_node_container(model)[node_id])


def _write_xyz(model: Any, node_id: Any, xyz: np.ndarray) -> None:
    container = _node_container(model)
    record = container[node_id]
    value = np.asarray(xyz, dtype=float).reshape(3)
    if isinstance(record, (list, tuple, np.ndarray)):
        container[node_id] = value
        return
    if isinstance(record, dict):
        for key in ("xyz", "coords", "coordinates", "position"):
            if key in record:
                record[key] = value
                return
        if all(k in record for k in ("x", "y", "z")):
            record["x"], record["y"], record["z"] = (float(v) for v in value)
            return
        record["xyz"] = value
        return
    for key in ("xyz", "coords", "coordinates", "position"):
        if hasattr(record, key):
            setattr(record, key, value)
            return
    raise TypeError(f"cannot write coordinates onto node record {node_id!r}")


def _element_nodes(record: Any) -> tuple[Any, ...]:
    from femtools.fea.protocols import get_any

    return tuple(get_any(record, ("nodes", "connectivity", "conn", "grids"), ()))


def _element_type(record: Any) -> str:
    from femtools.fea.protocols import get_any

    return str(get_any(record, ("type", "etype", "element_type"), "")).upper()


# ----------------------------------------------------------------------
# mesh quality
# ----------------------------------------------------------------------
def _area_vector(points: np.ndarray) -> np.ndarray:
    """Newell area vector of a planar polygon (magnitude = area)."""
    return 0.5 * np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)


def _corner_areas(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Signed corner areas of a polygon about ``normal`` (folds go negative)."""
    n = points.shape[0]
    out = np.empty(n)
    for i in range(n):
        e_next = points[(i + 1) % n] - points[i]
        e_prev = points[(i - 1) % n] - points[i]
        out[i] = 0.5 * float(np.cross(e_next, e_prev) @ normal)
    return out


def _hex_corner_jacobians(points: np.ndarray) -> np.ndarray:
    out = np.empty(len(_HEX8_CORNERS))
    for k, (i, a, b, c) in enumerate(_HEX8_CORNERS):
        out[k] = np.linalg.det(
            np.stack([points[a] - points[i], points[b] - points[i], points[c] - points[i]])
        )
    return out


def _element_measure(etype: str, points: np.ndarray, normal: np.ndarray | None) -> float | None:
    """Smallest signed size measure of one element (``None`` = not geometric)."""
    if etype in _IGNORED_ELEMENTS or points.shape[0] < 2:
        return None
    if etype in _LINE_ELEMENTS:
        return float(np.linalg.norm(points[1] - points[0]))
    if normal is not None and (etype in _SHELL_ELEMENTS or points.shape[0] in (3, 4)):
        return float(min(_corner_areas(points, normal).min(), _area_vector(points) @ normal))
    if etype in ("TET4", "CTETRA"):
        return float(
            np.linalg.det(
                np.stack([points[1] - points[0], points[2] - points[0], points[3] - points[0]])
            )
            / 6.0
        )
    if etype in ("HEX8", "CHEXA"):
        return float(_hex_corner_jacobians(points).min())
    return None


class _QualityMonitor:
    """Per-element size measures, normalised by the initial mesh."""

    def __init__(self, model: Any) -> None:
        self.records: list[tuple[Any, str, tuple[Any, ...], np.ndarray | None]] = []
        base: list[float] = []
        for eid, record in _element_records(model):
            etype = _element_type(record)
            nodes = _element_nodes(record)
            if etype in _IGNORED_ELEMENTS or len(nodes) < 2:
                continue
            points = np.array([_read_xyz(model, n) for n in nodes], dtype=float)
            normal = None
            if len(nodes) in (3, 4) and etype not in ("TET4", "CTETRA"):
                vec = _area_vector(points)
                norm = float(np.linalg.norm(vec))
                if norm <= 0.0:  # pragma: no cover - degenerate input mesh
                    continue
                normal = vec / norm
            m0 = _element_measure(etype, points, normal)
            if m0 is None or m0 == 0.0:
                continue
            self.records.append((eid, etype, nodes, normal))
            base.append(m0)
        self.reference = np.asarray(base, dtype=float)

    def __len__(self) -> int:
        return len(self.records)

    def ratios(self, model: Any) -> np.ndarray:
        """Current size measures divided by their initial values."""
        if not self.records:
            return np.ones(0)
        out = np.empty(len(self.records))
        for k, (_eid, etype, nodes, normal) in enumerate(self.records):
            points = np.array([_read_xyz(model, n) for n in nodes], dtype=float)
            m = _element_measure(etype, points, normal)
            out[k] = math.nan if m is None else m
        return out / self.reference

    def worst(self, model: Any) -> float:
        r = self.ratios(model)
        return float(np.min(r)) if r.size else 1.0


def element_size_ratios(model: Any, reference: Any) -> np.ndarray:
    """Element size measures of ``model`` divided by those of ``reference``.

    A ratio of 1 means the element is unchanged, 0 that it has collapsed and a
    negative value that it has inverted (folded shell, tangled solid).  Element
    lengths are used for bars and beams, signed areas for shells (about the
    normal of the *reference* mesh) and signed volumes / corner Jacobians for
    solids; point elements are skipped.
    """
    return _QualityMonitor(reference).ratios(model)


def _neighbour_map(model: Any) -> dict[Any, list[Any]]:
    """Mesh connectivity: node -> the nodes it shares an element with."""
    neighbours: dict[Any, set[Any]] = {}
    for _eid, record in _element_records(model):
        nodes = _element_nodes(record)
        if len(nodes) < 2:
            continue
        for a in nodes:
            bucket = neighbours.setdefault(a, set())
            for b in nodes:
                if b != a:
                    bucket.add(b)
    return {k: sorted(v, key=str) for k, v in neighbours.items()}


def _mean_edge_length(model: Any) -> float:
    lengths: list[float] = []
    for _eid, record in _element_records(model):
        nodes = _element_nodes(record)
        if len(nodes) < 2:
            continue
        pts = np.array([_read_xyz(model, n) for n in nodes], dtype=float)
        lengths.extend(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:], strict=False))
    finite = [ln for ln in lengths if ln > 0]
    return float(np.mean(finite)) if finite else 1.0


# ----------------------------------------------------------------------
@dataclass
class ShapeResult:
    """Outcome of :func:`shape_optimize`.

    Attributes
    ----------
    x:
        Optimal coordinate **offsets**, one entry per design variable in the
        order of :attr:`variables`.
    value / initial_value:
        Physical objective (frequency in Hz, or compliance) at the optimum and
        at the starting shape.
    coordinates / initial_coordinates:
        ``(n_moved, 3)`` positions of the moved nodes.
    model:
        Deep copy of the input model with the optimal coordinates written back.
    min_size_ratio:
        Smallest element size measure relative to the initial mesh; ``> 0``
        certifies that no element has inverted.
    """

    x: np.ndarray
    fun: float
    value: float
    initial_value: float
    objective: str
    nodes: list[Any]
    variables: list[tuple[Any, str]]
    coordinates: np.ndarray
    initial_coordinates: np.ndarray
    model: Any = None
    success: bool = False
    message: str = ""
    n_iter: int = 0
    n_fev: int = 0
    constraint_violation: float = 0.0
    min_size_ratio: float = 1.0
    initial_min_size_ratio: float = 1.0
    method: str = "SLSQP"
    history: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.atleast_1d(np.asarray(self.x, dtype=float))
        self.coordinates = np.atleast_2d(np.asarray(self.coordinates, dtype=float))
        self.initial_coordinates = np.atleast_2d(
            np.asarray(self.initial_coordinates, dtype=float)
        )

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.array(self.x, dtype=dtype, copy=copy)

    @property
    def displacement(self) -> np.ndarray:
        """``(n_moved, 3)`` movement of every moved node."""
        return self.coordinates - self.initial_coordinates

    @property
    def max_movement(self) -> float:
        d = self.displacement
        return float(np.max(np.linalg.norm(d, axis=1))) if d.size else 0.0

    @property
    def feasible(self) -> bool:
        return self.constraint_violation <= 1.0e-6 and self.min_size_ratio > 0.0

    @property
    def improvement(self) -> float:
        """Relative objective gain: frequency *rise*, compliance *drop*."""
        if self.initial_value == 0.0 or not math.isfinite(self.initial_value):
            return math.nan
        gain = (self.value - self.initial_value) / abs(self.initial_value)
        return float(gain if self.objective == "frequency" else -gain)

    def to_dict(self) -> dict[str, float]:
        return {
            f"{node}.{axis}": float(v)
            for (node, axis), v in zip(self.variables, self.x, strict=True)
        }

    def summary(self) -> str:
        lines = [
            f"shape_optimize({self.objective}, {self.method}) -> "
            f"{'converged' if self.success else 'stopped'} in {self.n_iter} iterations",
            f"  objective {self.initial_value:.6g} -> {self.value:.6g} "
            f"({100.0 * self.improvement:+.3f} %)",
            f"  min element size ratio {self.min_size_ratio:.4f} "
            f"(max node movement {self.max_movement:.4g})",
        ]
        for (node, axis), v in zip(self.variables, self.x, strict=True):
            lines.append(f"  node {node} d{axis} = {v:+.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ShapeResult(objective={self.objective!r}, value={self.value:.6g}, "
            f"success={self.success}, min_size_ratio={self.min_size_ratio:.4g})"
        )


# ----------------------------------------------------------------------
def _resolve_variables(
    nodes: Sequence[Any], directions: Any
) -> list[tuple[Any, str]]:
    """Expand ``(nodes, directions)`` into ``[(node_id, "x"|"y"|"z"), ...]``."""

    def _axes(spec: Any) -> list[str]:
        if spec is None:
            return list(_AXIS_NAME)
        if isinstance(spec, str):
            tokens = [t for t in spec.lower().replace(",", " ").split() if t] or [spec.lower()]
            if len(tokens) == 1 and len(tokens[0]) > 1 and all(c in "xyz012" for c in tokens[0]):
                tokens = list(tokens[0])
            out = []
            for t in tokens:
                if t not in _AXIS_INDEX:
                    raise ValueError(f"unknown direction {t!r}; expected x, y or z")
                out.append(_AXIS_NAME[_AXIS_INDEX[t]])
            return out
        if isinstance(spec, (int, np.integer)):
            return [_AXIS_NAME[int(spec)]]
        return [a for item in spec for a in _axes(item)]

    node_list = list(nodes)
    if isinstance(directions, Mapping):
        per_node = [_axes(directions.get(n, None)) for n in node_list]
    elif (
        directions is not None
        and not isinstance(directions, (str, int, np.integer))
        and len(list(directions)) == len(node_list)
        and any(not isinstance(d, (str, int, np.integer)) for d in directions)
    ):
        per_node = [_axes(d) for d in directions]
    else:
        common = _axes(directions)
        per_node = [list(common) for _ in node_list]

    variables: list[tuple[Any, str]] = []
    for node, axes in zip(node_list, per_node, strict=True):
        seen: set[str] = set()
        for axis in axes:
            if axis not in seen:
                seen.add(axis)
                variables.append((node, axis))
    if not variables:
        raise ValueError("no design variables: `nodes` / `directions` select nothing")
    return variables


def _objective_evaluator(
    objective: Any,
    mode: int,
    n_modes: int | None,
    skip_rigid: bool,
    loads: Any,
    solver_kwargs: dict[str, Any],
) -> tuple[Callable[[Any], float], str, int]:
    """``(evaluate(model) -> value, name, sign)``; ``sign=-1`` means maximise."""
    if callable(objective):
        return (lambda m: float(objective(m))), "custom", +1

    name = str(objective).strip().lower()
    if name in ("frequency", "freq", "f1", "eigenvalue", "modal"):
        from femtools.fea.eigen import solve_modes

        n_req = int(n_modes) if n_modes else int(mode) + 6

        def _frequency(m: Any) -> float:
            res = solve_modes(m, n_modes=n_req, **solver_kwargs)
            f = np.asarray(res.freq_hz, dtype=float)
            f = f[np.isfinite(f)]
            if skip_rigid and f.size:
                f = f[f > 1.0e-6 * max(float(f.max()), 1.0)]
            if f.size <= mode:
                raise RuntimeError(
                    f"the model has {f.size} elastic modes, mode index {mode} was requested"
                )
            return float(np.sort(f)[mode])

        return _frequency, "frequency", -1

    if name in ("compliance", "stiffness", "strain-energy", "strain_energy"):
        from femtools.fea.static import solve_static

        def _compliance(m: Any) -> float:
            res = solve_static(m, loads, full_result=True, **solver_kwargs)
            return float(np.asarray(res.u).ravel() @ np.asarray(res.load).ravel())

        return _compliance, "compliance", +1

    raise ValueError(
        f"unknown objective {objective!r}; expected 'frequency', 'compliance' or a callable"
    )


def shape_optimize(
    model: Any,
    nodes: Sequence[Any],
    *,
    directions: Any = "xyz",
    objective: Any = "frequency",
    mode: int = 0,
    n_modes: int | None = None,
    skip_rigid: bool = True,
    loads: Any = None,
    move_limit: Any = "auto",
    bounds: Any = None,
    min_quality: float | None = 0.2,
    smoothing: float = 0.0,
    constraints: Any = None,
    method: str = "SLSQP",
    max_iter: int = 50,
    tol: float = 1.0e-8,
    step: float = 1.0e-4,
    fd_method: str = "central",
    solver_kwargs: dict[str, Any] | None = None,
    keep_history: bool = True,
    verbose: bool = False,
) -> ShapeResult:
    """Move selected nodes to maximise a frequency or minimise compliance.

    Parameters
    ----------
    model:
        Any model the FEA kernel accepts (:class:`femtools.core.model.FEModel`
        or a duck-typed equivalent).  It is never modified: the optimiser works
        on a deep copy.
    nodes:
        Ids of the nodes that may move.
    directions:
        Which coordinate components are free: ``"xyz"`` (default), ``"x"``,
        ``"xy"``, a per-node sequence, or a ``{node_id: "xy"}`` mapping.
    objective:
        ``"frequency"`` -- maximise the ``mode``-th elastic natural frequency --
        ``"compliance"`` -- minimise ``u^T f`` -- or a callable
        ``f(model) -> float`` that is minimised.
    loads:
        Loads for the compliance objective, in any form
        :func:`femtools.fea.build_load_vector` accepts (e.g.
        ``{(node_id, dof): value}``).  ``None`` uses the model's own loads.
    move_limit:
        Bound on each coordinate offset, in model length units.  ``"auto"``
        (default) uses a quarter of the mean edge length; a scalar or a
        per-variable sequence sets it explicitly.  ``bounds`` overrides it with
        explicit ``(lo, hi)`` pairs on the offsets.
    min_quality:
        Lower bound on the smallest element size ratio
        ``min_e m_e(x)/m_e(x_0)`` (see :func:`element_size_ratios`).  The
        default ``0.2`` forbids both collapse and inversion; ``None`` disables
        the constraint.
    smoothing:
        Weight of the Laplacian mesh regulariser added to the (normalised)
        objective.  ``0`` (default) leaves the objective pure.
    constraints:
        Extra constraints on the offset vector, in any form
        :func:`femtools.optimization.size_optimize` accepts.
    method:
        ``"SLSQP"`` (default) or ``"trust-constr"``; any other SciPy
        ``minimize`` method that supports bounds also works.

    Returns
    -------
    ShapeResult

    Examples
    --------
    Raise the first frequency of a fixed-fixed bar chain by moving one node::

        res = shape_optimize(model, nodes=[3], directions="x",
                             objective="frequency", move_limit=0.15)
        assert res.value > res.initial_value and res.min_size_ratio > 0
    """
    variables = _resolve_variables(list(nodes), directions)
    node_ids = list(dict.fromkeys(n for n, _ in variables))
    n_var = len(variables)

    work = copy.deepcopy(model)
    x_ref = {nid: _read_xyz(work, nid) for nid in node_ids}
    quality = _QualityMonitor(work)
    href = _mean_edge_length(work)

    if bounds is None:
        if isinstance(move_limit, str):
            if move_limit.strip().lower() != "auto":
                raise ValueError(f"unknown move_limit {move_limit!r}")
            limit = np.full(n_var, 0.25 * href)
        else:
            limit = np.asarray(move_limit, dtype=float)
            limit = np.full(n_var, float(limit)) if limit.ndim == 0 else limit
        if limit.size != n_var:
            raise ValueError(f"move_limit has {limit.size} entries but {n_var} variables")
        if np.any(limit <= 0):
            raise ValueError("move_limit must be positive")
        bnds: Any = list(zip(-limit, limit, strict=True))
        scaling: Any = limit
    else:
        bnds = bounds
        limit = np.full(n_var, 0.25 * href)
        scaling = None

    solver_kw = dict(solver_kwargs or {})
    evaluate, obj_name, sign = _objective_evaluator(
        objective, int(mode), n_modes, bool(skip_rigid), loads, solver_kw
    )

    neighbours = _neighbour_map(work) if smoothing else {}

    def _apply(offsets: np.ndarray) -> None:
        delta: dict[Any, np.ndarray] = {nid: np.zeros(3) for nid in node_ids}
        for (nid, axis), value in zip(variables, offsets, strict=True):
            delta[nid][_AXIS_INDEX[axis]] = float(value)
        for nid in node_ids:
            _write_xyz(work, nid, x_ref[nid] + delta[nid])

    def _laplacian() -> float:
        if not smoothing:
            return 0.0
        total = 0.0
        for nid in node_ids:
            neigh = neighbours.get(nid, [])
            if not neigh:
                continue
            centre = np.mean([_read_xyz(work, n) for n in neigh], axis=0)
            total += float(np.sum((_read_xyz(work, nid) - centre) ** 2))
        return total / (href**2 * max(len(node_ids), 1))

    _apply(np.zeros(n_var))
    value0 = evaluate(work)
    if not math.isfinite(value0) or value0 == 0.0:
        raise ValueError(f"the objective is {value0} at the initial shape; cannot normalise")
    ratio0 = quality.worst(work)
    fref = abs(value0)
    f_start = sign * value0 / fref + smoothing * _laplacian()

    def _normalised(offsets: np.ndarray) -> float:
        _apply(np.asarray(offsets, dtype=float))
        try:
            value = evaluate(work)
        except Exception:
            # A tangled mesh can make the analysis fail outright; report it as a
            # very poor design instead of aborting the whole optimisation.
            return 1.0e6
        return sign * value / fref + smoothing * _laplacian()

    all_constraints: list[Any] = []
    if min_quality is not None:

        def _quality_margin(offsets: np.ndarray) -> float:
            _apply(np.asarray(offsets, dtype=float))
            return quality.worst(work) - float(min_quality)

        all_constraints.append(Constraint(fun=_quality_margin, type="ineq", name="min_quality"))
    if constraints is not None:
        extra = constraints if isinstance(constraints, (list, tuple)) else [constraints]
        all_constraints.extend(extra)

    opt = size_optimize(
        _normalised,
        np.zeros(n_var),
        bounds=bnds,
        constraints=all_constraints or None,
        method=method,
        max_iter=int(max_iter),
        tol=float(tol),
        step=float(step),
        fd_method=fd_method,
        scaling=scaling,
        objective_scaling=None,
        keep_history=keep_history,
        names=[f"{nid}.{axis}" for nid, axis in variables],
        verbose=verbose,
    )

    x_opt = np.asarray(opt.x, dtype=float)
    f_opt = _normalised(x_opt)
    ratio = quality.worst(work)
    infeasible = min_quality is not None and ratio < float(min_quality) - 1.0e-9
    if f_opt >= f_start or infeasible:
        # SLSQP can end on its last trial point rather than on the best one;
        # never report a shape that is worse -- or more tangled -- than the mesh
        # the caller handed in.
        x_opt = np.zeros(n_var)
        f_opt = _normalised(x_opt)
        ratio = ratio0
    value = evaluate(work)

    return ShapeResult(
        x=x_opt,
        fun=float(f_opt),
        value=float(value),
        initial_value=float(value0),
        objective=obj_name,
        nodes=node_ids,
        variables=variables,
        coordinates=np.array([_read_xyz(work, nid) for nid in node_ids]),
        initial_coordinates=np.array([x_ref[nid] for nid in node_ids]),
        model=copy.deepcopy(work),
        success=bool(opt.success) and ratio > 0.0,
        message=str(opt.message),
        n_iter=int(opt.n_iter),
        n_fev=int(opt.n_fev),
        constraint_violation=float(opt.constraint_violation),
        min_size_ratio=float(ratio),
        initial_min_size_ratio=float(ratio0),
        method=method,
        history=list(opt.history),
        extras={
            "move_limit": limit,
            "mean_edge_length": href,
            "n_elements_monitored": len(quality),
            "size_ratios": quality.ratios(work),
            "smoothing": float(smoothing),
            "laplacian": _laplacian(),
            "objective_start": f_start,
            "optimizer": opt,
        },
    )
