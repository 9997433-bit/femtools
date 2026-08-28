"""Multipoint constraints: ``RBE2`` rigid bodies as a degree-of-freedom transform.

An ``RBE2`` welds a set of *dependent* grid points to one *independent* grid
point: every listed component of a dependent node is driven by the six DOFs of
the independent node through the rigid-body kinematics

.. math::

    u_m = u_n + \\theta_n \\times r, \\qquad \\theta_m = \\theta_n,
    \\qquad r = x_m - x_n .

That is the public Nastran ``RBE2`` card layout (independent grid ``GN``,
component list ``CM``, dependent grids ``GM``), and it is nothing more than a
statement about kinematics; the constraint is imposed here the classical way
(Cook, *Concepts and Applications of Finite Element Analysis*, §13.5;
Zienkiewicz & Taylor, *The Finite Element Method*, master-slave elimination),
by building the transformation ``G`` with

* ``G[i, i] = 1`` for every retained (independent) DOF ``i``,
* ``G[d, :]`` holding the rigid-body coefficients for every eliminated
  (dependent) DOF ``d``, and a **zero column** at ``d``,

and taking the congruence ``G^T A G`` of every system matrix.  Loads follow the
same map by virtual work, ``f -> G^T f``, which is what carries a force applied
on a rigid offset back to the independent node *as a force and a moment*.

Two properties of ``G`` are worth stating because the rest of the kernel relies
on them.  It is **idempotent** (``G @ G == G``): a dependent row references
independent columns only, chains having been resolved when the transform was
built.  Filling the dependent entries of a vector is therefore safe to repeat,
and a quadratic form ``v^T (G^T A G) v`` is unchanged by it -- which is why
mode shapes can be reported with the dependent motion filled in while staying
mass-orthonormal against the transformed mass matrix.  And it is **exact**: no
penalty stiffness, no Lagrange multiplier, no extra equations, so the
constrained model keeps exactly the six rigid-body modes of the free-free
structure it describes.

The transform is consumed by :func:`femtools.fea.assemble.assemble_km`, which
honours ``model.rbe2`` by default; :func:`apply_rbe2` builds one explicitly.
The ``RBE2`` records themselves live on
:class:`femtools.core.model.FEModel` (``add_rbe2``) and are read duck-typed,
like everything else the kernel consumes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .dofmap import DofMap
from .elements import ModelIndex
from .protocols import get_any, iter_records

__all__ = [
    "MAX_CHAIN_DEPTH",
    "ConstraintTransform",
    "apply_rbe2",
    "rbe2_records",
    "resolve_mpc",
]

#: How deep an ``RBE2`` chain (a dependent node acting as the independent node
#: of a further rigid body) is resolved before the model is declared pathological.
MAX_CHAIN_DEPTH = 8

#: Names an ``RBE2``-like record may use for its independent grid point.
_INDEPENDENT_NAMES = ("independent", "independent_node", "gn", "master", "master_node")

#: ... and for its dependent grid points.
_DEPENDENT_NAMES = ("dependents", "dependent", "dependent_nodes", "gm", "slaves", "slave_nodes")

#: ... and for the constrained component list (Nastran ``CM``).
_COMPONENT_NAMES = ("components", "cm", "component", "dofs", "dof")

#: Containers on a model that may hold rigid body records.
_CONTAINER_NAMES = ("rbe2", "rbe2s", "rigid_elements", "rigids", "mpc", "mpcs")


def _skew(r: np.ndarray) -> np.ndarray:
    """``skew(r) @ v == cross(r, v)``."""
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    return np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])


def _components(spec: Any, *, what: str) -> list[int]:
    """Nastran ``CM`` digits (1..6) -> zero based component indices.

    Integers are **one based** here, unlike
    :func:`femtools.fea.protocols.normalize_dof`: an ``RBE2`` component field
    is a Nastran component list, and :class:`femtools.core.model.RBE2` already
    validates it as ``1..6``.  Both the packed form (``123456``, ``"1236"``)
    and a sequence (``(1, 2, 3)``) are accepted.
    """
    if spec is None:
        return [0, 1, 2, 3, 4, 5]
    if isinstance(spec, (bool, np.bool_)):
        raise ValueError(f"{what}: a boolean is not a component list")
    if isinstance(spec, (int, np.integer)):
        spec = str(int(spec))
    if isinstance(spec, (float, np.floating)):
        if float(spec) != int(spec):
            raise ValueError(f"{what}: non integer component {spec!r}")
        spec = str(int(spec))
    if isinstance(spec, str):
        digits = spec.replace(" ", "").replace(",", "")
        if not digits.isdigit():
            raise ValueError(f"{what}: unrecognised component list {spec!r}")
        raw: Iterable[Any] = list(digits)
    elif isinstance(spec, Iterable):
        raw = list(spec)
    else:
        raise ValueError(f"{what}: unrecognised component list {spec!r}")

    out: list[int] = []
    for item in raw:
        value = int(str(item).strip())
        if not 1 <= value <= 6:
            raise ValueError(
                f"{what}: component {item!r} is out of range; RBE2 components are "
                "Nastran 1..6 (UX..RZ)"
            )
        if value - 1 not in out:
            out.append(value - 1)
    if not out:
        raise ValueError(f"{what}: empty component list")
    return sorted(out)


def rbe2_records(model: Any, rbe2: Any = None) -> list[tuple[Any, Any]]:
    """Collect ``(id, record)`` pairs of the rigid bodies to apply.

    ``rbe2`` overrides the model's own table; a single record is accepted as
    well as a list or a mapping.
    """
    container = rbe2
    if container is None:
        for name in _CONTAINER_NAMES:
            container = get_any(model, name, None)
            if container is not None:
                break
    if container is None:
        return []
    if get_any(container, _INDEPENDENT_NAMES, None) is not None:
        # A single record rather than a container of them.
        return [(get_any(container, ("id", "eid"), 1), container)]
    return [(rid, record) for rid, record in iter_records(container) if record is not None]


# ---------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------


@dataclass
class ConstraintTransform:
    """The multipoint constraint transformation of one model.

    Attributes
    ----------
    dof_map:
        The DOF numbering the transform is written against.
    G:
        ``(n_dof, n_dof)`` operator.  Identity on the independent DOFs, the
        rigid-body coefficients on the dependent rows, zero on the dependent
        columns.  Idempotent, so applying it twice changes nothing.
    dependent:
        Eliminated global DOF indices, ascending.
    sources:
        ``dependent dof -> id of the rigid body that eliminated it``, kept so
        a conflict can be reported against the card the user wrote.

    The rectangular form the master-slave literature calls ``T`` -- the
    ``(n_dof, n_independent)`` matrix with ``u = T @ u_independent`` -- is
    :attr:`T`; ``G`` is the same map written as a square projector so that DOF
    numbering, and therefore every index the assembler hands out, is preserved.
    """

    dof_map: DofMap
    G: sp.csr_matrix
    dependent: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    sources: dict[int, Any] = field(default_factory=dict)

    # -- construction ---------------------------------------------------
    @classmethod
    def identity(cls, dof_map: DofMap) -> ConstraintTransform:
        """The empty transform: nothing is eliminated."""
        n = dof_map.n_dof
        return cls(dof_map=dof_map, G=sp.identity(n, format="csr"))

    @classmethod
    def from_rows(
        cls,
        dof_map: DofMap,
        rows: dict[int, dict[int, float]],
        *,
        sources: dict[int, Any] | None = None,
        max_depth: int = MAX_CHAIN_DEPTH,
    ) -> ConstraintTransform:
        """Assemble ``G`` from ``dependent dof -> {source dof: coefficient}``.

        Chains are resolved here (a source that is itself dependent is
        substituted), so every row of the result references independent DOFs
        only.  A cycle, or a chain deeper than *max_depth*, is an error.
        """
        if not rows:
            return cls.identity(dof_map)
        resolved = _resolve_chains(rows, sources or {}, max_depth)
        n = dof_map.n_dof
        dependent = np.array(sorted(resolved), dtype=int)

        keep = np.ones(n, dtype=bool)
        keep[dependent] = False
        independent = np.flatnonzero(keep)
        data = [np.ones(independent.size)]
        rr = [independent]
        cc = [independent]
        for dof, row in resolved.items():
            if not row:
                continue
            cols = np.fromiter(row.keys(), dtype=int, count=len(row))
            vals = np.fromiter(row.values(), dtype=float, count=len(row))
            rr.append(np.full(cols.size, dof, dtype=int))
            cc.append(cols)
            data.append(vals)
        G = sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rr), np.concatenate(cc))), shape=(n, n)
        ).tocsr()
        G.sum_duplicates()
        return cls(dof_map=dof_map, G=G, dependent=dependent, sources=dict(sources or {}))

    # -- basics ---------------------------------------------------------
    @property
    def n_dof(self) -> int:
        return int(self.G.shape[0])

    @property
    def is_identity(self) -> bool:
        return int(self.dependent.size) == 0

    @property
    def n_dependent(self) -> int:
        return int(self.dependent.size)

    @property
    def independent(self) -> np.ndarray:
        """Retained global DOF indices, ascending."""
        mask = np.ones(self.n_dof, dtype=bool)
        mask[self.dependent] = False
        return np.flatnonzero(mask)

    @property
    def T(self) -> sp.csr_matrix:  # noqa: N802 - the name used in the literature
        """``(n_dof, n_independent)`` form with ``u = T @ u_independent``."""
        return sp.csr_matrix(self.G[:, self.independent])

    def matrix(self) -> sp.csr_matrix:
        """The square operator ``G`` (see :attr:`G`)."""
        return self.G

    def dependent_nodes(self) -> list[Any]:
        return _unique_nodes(self.dof_map, self.dependent)

    def independent_nodes(self) -> list[Any]:
        """Nodes actually referenced by a dependent row."""
        if self.is_identity:
            return []
        cols = np.unique(self.G[self.dependent, :].tocoo().col)
        return _unique_nodes(self.dof_map, cols)

    def nodes(self) -> list[Any]:
        """Every node the constraint touches, dependent or independent."""
        seen = dict.fromkeys(self.dependent_nodes())
        seen.update(dict.fromkeys(self.independent_nodes()))
        return list(seen)

    # -- use --------------------------------------------------------------
    def congruence(self, matrix: sp.spmatrix) -> sp.csr_matrix:
        """``G^T @ matrix @ G``, the constrained form of a system matrix."""
        if self.is_identity:
            return matrix.tocsr()
        return (self.G.T @ matrix.tocsr() @ self.G).tocsr()

    def to_full(self, vector: Any) -> np.ndarray:
        """Fill the dependent entries of a displacement vector (or columns).

        ``G`` is idempotent, so a vector that already carries its dependent
        motion is returned unchanged.
        """
        v = np.asarray(vector)
        return v if self.is_identity else np.asarray(self.G @ v)

    def to_independent(self, vector: Any) -> np.ndarray:
        """Move a load vector onto the independent DOFs: ``G^T @ f``.

        This is the virtual-work image of :meth:`to_full`; a force applied to a
        dependent node arrives at the independent node as a force *and* the
        moment of its offset.
        """
        v = np.asarray(vector)
        return v if self.is_identity else np.asarray(self.G.T @ v)

    def reduce(self, matrix: sp.spmatrix) -> sp.csr_matrix:
        """The constrained matrix restricted to the independent DOFs."""
        reduced = self.congruence(matrix)
        idx = self.independent
        return sp.csr_matrix(reduced[idx][:, idx])

    def summary(self) -> str:  # pragma: no cover - reporting helper
        return (
            f"ConstraintTransform(n_dof={self.n_dof}, dependent={self.n_dependent}, "
            f"nodes={len(self.dependent_nodes())})"
        )


def _unique_nodes(dof_map: DofMap, dofs: Iterable[int]) -> list[Any]:
    seen: dict[Any, None] = {}
    for dof in dofs:
        seen.setdefault(dof_map.dof_node(int(dof)), None)
    return list(seen)


def _resolve_chains(
    rows: dict[int, dict[int, float]],
    sources: dict[int, Any],
    max_depth: int,
) -> dict[int, dict[int, float]]:
    """Substitute dependent sources until every row is purely independent."""
    resolved: dict[int, dict[int, float]] = {}
    visiting: set[int] = set()

    def expand(dof: int, depth: int) -> dict[int, float]:
        done = resolved.get(dof)
        if done is not None:
            return done
        if dof in visiting:
            raise ValueError(
                f"circular multipoint constraint: DOF {dof} depends on itself "
                f"(rigid body {sources.get(dof)!r})"
            )
        if depth > max_depth:
            raise ValueError(
                f"multipoint constraint chain deeper than {max_depth} at DOF {dof} "
                f"(rigid body {sources.get(dof)!r}); check for a rigid body whose "
                "independent node is dependent on another one"
            )
        visiting.add(dof)
        out: dict[int, float] = {}
        for src, coeff in rows[dof].items():
            if coeff == 0.0:
                continue
            if src in rows:
                for deep, deep_coeff in expand(src, depth + 1).items():
                    out[deep] = out.get(deep, 0.0) + coeff * deep_coeff
            else:
                out[src] = out.get(src, 0.0) + coeff
        visiting.discard(dof)
        resolved[dof] = {k: v for k, v in out.items() if v != 0.0}
        return resolved[dof]

    for dof in rows:
        expand(dof, 1)
    return resolved


# ---------------------------------------------------------------------------
# RBE2
# ---------------------------------------------------------------------------


def apply_rbe2(
    model: Any,
    rbe2: Any = None,
    *,
    dof_map: DofMap | None = None,
    dofs_per_node: int = 6,
    index: ModelIndex | None = None,
) -> ConstraintTransform:
    """Build the :class:`ConstraintTransform` of a model's rigid bodies.

    Parameters
    ----------
    model
        Anything satisfying :class:`~femtools.fea.protocols.ModelLike`; only
        the node coordinates are read.
    rbe2
        The rigid bodies to apply.  ``None`` (default) reads ``model.rbe2``
        (:meth:`femtools.core.model.FEModel.add_rbe2`).  A list, a mapping or a
        single record is accepted, so an explicit rigid body can be applied to
        a model that does not carry one.
    dof_map
        DOF numbering to write the transform against.  Built from the model's
        nodes -- exactly as :func:`~femtools.fea.assemble.assemble_km` builds
        it -- when omitted.
    dofs_per_node
        Six for the standard 3D structural model.  A rigid body with an offset
        needs the rotations of its independent node, so a smaller value is only
        accepted for coincident nodes.

    Returns
    -------
    ConstraintTransform
        Empty (``is_identity``) when the model has no rigid bodies.

    Raises
    ------
    ValueError
        A component outside ``1..6``, a DOF made dependent twice, a rigid body
        chain that loops or is deeper than :data:`MAX_CHAIN_DEPTH`, or an
        offset that cannot be expressed with the available DOFs.
    KeyError
        A referenced node is not in the DOF map.
    """
    index = ModelIndex.build(model) if index is None else index
    if dof_map is None:
        dof_map = DofMap.from_nodes(index.nodes, dofs_per_node)
    dofs_per_node = dof_map.dofs_per_node

    records = rbe2_records(model, rbe2)
    if not records:
        return ConstraintTransform.identity(dof_map)

    rows: dict[int, dict[int, float]] = {}
    sources: dict[int, Any] = {}
    for rid, record in records:
        rid = get_any(record, ("id", "eid"), rid)
        _add_rbe2_rows(record, rid, dof_map, index, dofs_per_node, rows, sources)
    return ConstraintTransform.from_rows(dof_map, rows, sources=sources)


def _add_rbe2_rows(
    record: Any,
    rid: Any,
    dof_map: DofMap,
    index: ModelIndex,
    dofs_per_node: int,
    rows: dict[int, dict[int, float]],
    sources: dict[int, Any],
) -> None:
    what = f"RBE2 {rid}"
    independent = get_any(record, _INDEPENDENT_NAMES, None)
    if independent is None:
        raise ValueError(f"{what}: no independent node (looked for {_INDEPENDENT_NAMES})")
    dependents = get_any(record, _DEPENDENT_NAMES, None)
    if dependents is None:
        raise ValueError(f"{what}: no dependent nodes (looked for {_DEPENDENT_NAMES})")
    if isinstance(dependents, (int, np.integer, str)):
        dependents = (dependents,)
    dependents = list(dependents)
    if not dependents:
        raise ValueError(f"{what}: at least one dependent node is required")
    comps = _components(get_any(record, _COMPONENT_NAMES, None), what=what)
    if max(comps) >= dofs_per_node:
        raise ValueError(
            f"{what}: component {max(comps) + 1} needs {max(comps) + 1} DOFs per node, "
            f"the model has {dofs_per_node}"
        )

    try:
        origin = np.asarray(index.xyz(independent), dtype=float)
        base = np.fromiter(
            (dof_map.index(independent, c) for c in range(dofs_per_node)),
            dtype=int,
            count=dofs_per_node,
        )
    except KeyError as exc:
        raise KeyError(f"{what}: independent node {independent!r} is not in the model") from exc

    for dependent in dependents:
        if dependent == independent:
            raise ValueError(f"{what}: independent node {independent!r} cannot also be dependent")
        try:
            offset = np.asarray(index.xyz(dependent), dtype=float) - origin
            slave = np.fromiter(
                (dof_map.index(dependent, c) for c in range(dofs_per_node)),
                dtype=int,
                count=dofs_per_node,
            )
        except KeyError as exc:
            raise KeyError(f"{what}: dependent node {dependent!r} is not in the model") from exc

        # u_m = u_n + theta_n x r  =  u_n - skew(r) theta_n,  theta_m = theta_n.
        lever = -_skew(offset)
        rigid = bool(np.any(offset))
        if rigid and dofs_per_node < 6:
            raise ValueError(
                f"{what}: node {dependent!r} is offset from {independent!r} but the model "
                f"has only {dofs_per_node} DOFs per node, so the rigid rotation term "
                "cannot be represented"
            )
        for comp in comps:
            dof = int(slave[comp])
            if dof in rows:
                raise ValueError(
                    f"{what}: DOF {comp + 1} of node {dependent!r} is already dependent "
                    f"(rigid body {sources[dof]!r}); a DOF may be eliminated only once"
                )
            row = {int(base[comp]): 1.0}
            if comp < 3 and rigid:
                for j in range(3):
                    coeff = float(lever[comp, j])
                    if coeff != 0.0:
                        row[int(base[3 + j])] = coeff
            rows[dof] = row
            sources[dof] = rid


def resolve_mpc(
    model: Any,
    mpc: Any = None,
    *,
    dof_map: DofMap,
    index: ModelIndex | None = None,
) -> ConstraintTransform:
    """Interpret the ``mpc`` argument of :func:`~femtools.fea.assemble.assemble_km`.

    ``None`` builds the transform from ``model.rbe2``, ``False`` disables
    multipoint constraints entirely, a :class:`ConstraintTransform` is used as
    given and anything else is read as an explicit set of rigid body records
    (which then *replaces* the model's own table).
    """
    if mpc is False:
        return ConstraintTransform.identity(dof_map)
    if isinstance(mpc, ConstraintTransform):
        if mpc.n_dof != dof_map.n_dof:
            raise ValueError(
                f"constraint transform is written for {mpc.n_dof} DOFs but this model "
                f"has {dof_map.n_dof}"
            )
        return mpc
    if sp.issparse(mpc) or isinstance(mpc, np.ndarray):
        raise TypeError(
            "mpc= takes a ConstraintTransform (femtools.fea.mpc.apply_rbe2) or RBE2 "
            "records, not a bare matrix"
        )
    return apply_rbe2(model, mpc, dof_map=dof_map, index=index)
