"""Candidate sensor DOFs sliced out of a solved FE model.

A pretest never runs on the raw global mode matrix: an accelerometer measures
one translation of one node, so the candidate set is the translational
partition of the free DOFs, restricted to the nodes the test can physically
reach.  :func:`candidate_dofs` performs that slice against the real DOF map of
a :class:`~femtools.fea.eigen.ModalResult` and returns everything the
selection routines need::

    modal = solve_modes(model, n_modes=20)
    xyz = node_coordinates(model, modal)
    em = effective_mass(modal, modal.M, dof_map=modal, coords=xyz)
    targets = select_target_modes(em, f_max=200.0)

    cand = translational_dofs(modal, mode_index=targets)
    sensors = effective_independence(cand.phi, 12, candidate_dofs=cand.dofs)
    print(cand.take(sensors.index).labels)           # ['12Z', '31Z', ...]

The slice is a fancy index over the mode matrix, so the cost is one copy of
the retained rows; no Python loop touches a DOF.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..correlation._linalg import as_mode_matrix, mode_frequencies, mode_source, row_index
from ..correlation.dofmap import DOFMap
from ._result import IdSequenceMixin

__all__ = [
    "CandidateSet",
    "candidate_dofs",
    "translational_dofs",
    "node_coordinates",
]

#: Component ids an accelerometer can observe.
TRANSLATIONS: tuple[int, int, int] = (1, 2, 3)


@dataclass
class CandidateSet(IdSequenceMixin):
    """Candidate sensor DOFs and the target mode partition at those DOFs.

    Behaves like :attr:`dofs`, the global DOF numbers of the retained rows,
    so it can be handed straight to a selection routine as ``candidate_dofs``.
    """

    phi: NDArray[Any]
    dofs: NDArray[np.intp]
    dof_map: DOFMap
    mode_index: NDArray[np.intp]
    freq_hz: NDArray[np.float64] | None = None
    coords: NDArray[np.float64] | None = None
    dropped: NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, dtype=np.intp))

    _id_field = "dofs"

    #: Alias of :attr:`dofs` — the rows of the *global* mode matrix retained.
    @property
    def index(self) -> NDArray[np.intp]:
        return self.dofs

    @property
    def labels(self) -> list[str]:
        """Human readable DOF labels, e.g. ``['12X', '12Y', '31Z']``."""
        return self.dof_map.labels

    @property
    def nodes(self) -> NDArray[np.int64]:
        return self.dof_map.nodes

    @property
    def components(self) -> NDArray[np.int8]:
        return self.dof_map.components

    @property
    def n_candidates(self) -> int:
        return int(self.phi.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.phi.shape[1])

    def take(self, index: ArrayLike) -> CandidateSet:
        """Sub-set for the given candidate rows (e.g. ``EFIResult.index``).

        Accepts positions or a boolean mask over the candidates.
        """
        rows = row_index(index, self.n_candidates)
        return CandidateSet(
            phi=self.phi[rows],
            dofs=self.dofs[rows],
            dof_map=self.dof_map.take(rows),
            mode_index=self.mode_index,
            freq_hz=self.freq_hz,
            coords=None if self.coords is None else self.coords[rows],
        )

    def __repr__(self) -> str:  # pragma: no cover - reporting helper
        preview = ", ".join(self.labels[:6])
        more = ", ..." if self.n_candidates > 6 else ""
        return f"CandidateSet({self.n_candidates} dof x {self.n_modes} modes: {preview}{more})"


def candidate_dofs(
    source: Any,
    *,
    modes: ArrayLike | None = None,
    dof_map: Any = None,
    components: Iterable[Any] | None = TRANSLATIONS,
    nodes: ArrayLike | None = None,
    exclude: ArrayLike | None = None,
    mode_index: ArrayLike | None = None,
    active_only: bool = True,
    coords: Any = None,
) -> CandidateSet:
    """Slice the measurable DOFs of a solved model into a candidate set.

    Parameters
    ----------
    source:
        A :class:`~femtools.fea.eigen.ModalResult` (mode shapes, DOF map,
        free-DOF partition and frequencies are all taken from it), or any
        object :meth:`DOFMap.from_mapping` understands when ``modes`` is
        given separately.
    modes:
        Mode shapes ``(n_dof, n_mode)`` over the **full** DOF space; defaults
        to the modes carried by ``source``.
    dof_map:
        Overrides the DOF map of ``source``.
    components:
        Measurable components, default the three translations.  Labels
        (``"z"``, ``"-Z"``) and one based ids are accepted; ``None`` keeps
        every component.
    nodes:
        Node ids the test can reach; default every node in the map.
    exclude:
        Node ids (or ``(node, component)`` keys) to drop, e.g. the fixed
        interface or a location without access.
    mode_index:
        Target mode columns, typically a
        :class:`~femtools.pretest.target_modes.TargetModeSelection`.
    active_only:
        Drop candidates that carry no motion: DOFs outside the assembly's
        free set (single point constrained, massless, fictitious drilling)
        and rows that are exactly zero in every target mode.  Such rows have
        zero effective independence and would only dilute the ranking.
    coords:
        Nodal coordinates for :attr:`CandidateSet.coords` — a model, a
        ``{node: xyz}`` mapping or an ``(n_node, 3)`` array (see
        :func:`node_coordinates`).  Defaults to ``source`` when it can
        provide them.

    Returns
    -------
    CandidateSet
        ``result.phi`` is the ``(n_candidate, n_target)`` partition,
        ``result.dofs`` the global DOF numbers of its rows and
        ``result.dof_map`` their ``(node, component)`` labels.
    """
    dmap = DOFMap.from_mapping(dof_map if dof_map is not None else source)
    phi_full = as_mode_matrix(modes if modes is not None else mode_source(source), "modes")
    if phi_full.shape[0] != len(dmap):
        raise ValueError(
            f"the mode shapes have {phi_full.shape[0]} rows but the DOF map "
            f"describes {len(dmap)} DOF"
        )

    cols = _mode_columns(mode_index, phi_full.shape[1])
    rows = dmap.select(components=components, nodes=nodes)
    if exclude is not None:
        rows = rows[~np.isin(rows, _excluded_rows(dmap, exclude))]

    phi = phi_full[np.ix_(rows, cols)]
    dropped = np.zeros(0, dtype=np.intp)
    if active_only:
        keep = np.any(phi != 0.0, axis=1)
        free = getattr(source, "free_dof", None)
        if free is not None and np.size(free):
            keep &= np.isin(rows, np.asarray(free, dtype=np.intp).reshape(-1))
        dropped = rows[~keep]
        rows, phi = rows[keep], phi[keep]

    freq = mode_frequencies(source)
    freqs = None if freq is None or freq.size != phi_full.shape[1] else freq[cols]
    kept = dmap.take(rows)
    return CandidateSet(
        phi=phi,
        dofs=rows.astype(np.intp),
        dof_map=kept,
        mode_index=cols,
        freq_hz=freqs,
        coords=_candidate_coords(coords if coords is not None else source, dmap, kept, rows),
        dropped=dropped.astype(np.intp),
    )


def _candidate_coords(
    source: Any, full: DOFMap, kept: DOFMap, rows: NDArray[np.intp]
) -> NDArray[np.float64] | None:
    """Coordinates of the retained candidate rows.

    Resolved against the *whole* DOF map first and sliced afterwards: a bare
    coordinate array carries no node ids, so its rows can only be read
    positionally against the complete node list — matching it to the retained
    subset would drop it silently as soon as one candidate is excluded.
    """
    whole = node_coordinates(source, full, per_dof=True, missing="none")
    if whole is not None:
        return whole[rows]
    return node_coordinates(source, kept, per_dof=True, missing="none")


def translational_dofs(source: Any, **kwargs: Any) -> CandidateSet:
    """Candidate set restricted to the translational DOFs (X, Y, Z).

    Shorthand for :func:`candidate_dofs` with ``components=(1, 2, 3)``; see
    it for the remaining options.
    """
    kwargs.setdefault("components", TRANSLATIONS)
    return candidate_dofs(source, **kwargs)


def node_coordinates(
    source: Any,
    dof_map: Any = None,
    *,
    per_dof: bool = False,
    missing: str = "raise",
) -> NDArray[np.float64] | None:
    """Nodal coordinates of a model, ordered to match a DOF map.

    Parameters
    ----------
    source:
        A model exposing ``nodes`` (``{id: node}`` with ``xyz``/``coords``),
        a ``{node: xyz}`` mapping, an ``(ids, xyz)`` pair, an ``(n_node, 3)``
        array or an object carrying any of those (a ``ModalResult`` reaches
        its model through ``assembly``).
    dof_map:
        Target ordering.  Without it the coordinates come back in the
        model's own node order.  A bare array carries no node ids, so its
        rows are then read in the order of ``np.unique(dof_map.nodes)``.
    per_dof:
        Return one row per DOF of ``dof_map`` instead of one per node, ready
        for :func:`~femtools.pretest.target_modes.rigid_body_modes`.
    missing:
        ``"raise"`` (default) or ``"none"``, which returns ``None`` when the
        coordinates cannot be recovered.
    """
    table = _coordinate_table(source)
    if table is None:
        if missing == "none":
            return None
        raise ValueError("no nodal coordinates found on the given source")
    ids, xyz = table
    if dof_map is None:
        return xyz

    dmap = DOFMap.from_mapping(dof_map)
    wanted = dmap.nodes if per_dof else np.unique(dmap.nodes)
    if ids is None:
        # A bare array has no ids of its own: its rows are the map's nodes in
        # ascending order, the convention `rigid_body_modes` already uses.
        ids = np.unique(dmap.nodes)
        if ids.size != xyz.shape[0]:
            if missing == "none":
                return None
            raise ValueError(
                f"coordinate array has {xyz.shape[0]} rows but the DOF map covers "
                f"{ids.size} nodes; pass a {{node: xyz}} mapping or an (ids, xyz) pair"
            )
    order = np.argsort(ids, kind="stable")
    pos = np.clip(np.searchsorted(ids[order], wanted), 0, max(ids.size - 1, 0))
    found = ids[order][pos] == wanted if ids.size else np.zeros(wanted.size, dtype=bool)
    if not found.all():
        if missing == "none":
            return None
        unknown = np.unique(wanted[~found])[:5].tolist()
        raise KeyError(f"no coordinates for node(s) {unknown}")
    return xyz[order][pos]


def _coordinate_table(source: Any) -> tuple[NDArray[np.int64] | None, NDArray[np.float64]] | None:
    """``(node_ids, xyz)`` recovered from a model-like object, if possible.

    The ids are ``None`` for a bare coordinate array, which carries none.
    """
    if source is None:
        return None
    if isinstance(source, dict):
        if not source:
            return None
        ids = np.fromiter((int(k) for k in source), dtype=np.int64, count=len(source))
        xyz = np.array([np.asarray(v, dtype=float).reshape(-1)[:3] for v in source.values()])
        return ids, xyz
    if isinstance(source, tuple) and len(source) == 2 and np.ndim(source[1]) == 2:
        pair_ids = np.asarray(source[0], dtype=np.int64).reshape(-1)
        pair_xyz = _xyz_array(source[1])
        if pair_xyz is None:
            return None
        if pair_ids.size != pair_xyz.shape[0]:
            raise ValueError(f"{pair_ids.size} node ids for {pair_xyz.shape[0]} coordinate rows")
        return pair_ids, pair_xyz
    if isinstance(source, (np.ndarray, list)):
        bare = _xyz_array(source)
        return None if bare is None else (None, bare)
    nodes = getattr(source, "nodes", None)
    if isinstance(nodes, dict) and nodes:
        ids = np.fromiter((int(k) for k in nodes), dtype=np.int64, count=len(nodes))
        xyz = np.array([_node_xyz(v) for v in nodes.values()])
        return ids, xyz
    for name in ("model", "assembly"):
        nested = getattr(source, name, None)
        if nested is not None and nested is not source:
            table = _coordinate_table(nested)
            if table is not None:
                return table
    return None


def _xyz_array(source: Any) -> NDArray[np.float64] | None:
    """``(n_node, 3)`` coordinates from an array-like, or ``None`` if it is not one."""
    try:
        arr = np.asarray(source, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        return None
    if arr.shape[1] == 2:  # a planar test geometry
        arr = np.column_stack((arr, np.zeros(arr.shape[0])))
    return arr


def _node_xyz(node: Any) -> NDArray[np.float64]:
    for name in ("xyz", "coords", "coordinates", "position"):
        value = getattr(node, name, None)
        if value is not None:
            return np.asarray(value, dtype=float).reshape(-1)[:3]
    return np.asarray(node, dtype=float).reshape(-1)[:3]


def _mode_columns(mode_index: ArrayLike | None, n_mode: int) -> NDArray[np.intp]:
    if mode_index is None:
        return np.arange(n_mode, dtype=np.intp)
    cols = np.asarray(mode_index).reshape(-1)
    if cols.dtype == bool:
        if cols.size != n_mode:
            raise ValueError(f"a boolean mode_index needs {n_mode} entries, got {cols.size}")
        return np.flatnonzero(cols).astype(np.intp)
    cols = cols.astype(np.intp)
    if cols.size and (cols.min() < 0 or cols.max() >= n_mode):
        raise ValueError(f"mode_index out of range 0..{n_mode - 1}")
    return cols


def _excluded_rows(dmap: DOFMap, exclude: ArrayLike) -> NDArray[np.intp]:
    """Rows addressed by ``exclude``: whole nodes, DOF keys or labels."""
    items = np.atleast_1d(np.asarray(exclude, dtype=object)).tolist()
    is_node = [isinstance(i, int) and not isinstance(i, bool) for i in items]
    plain = [i for i, flag in zip(items, is_node, strict=True) if flag]
    keyed = [i for i, flag in zip(items, is_node, strict=True) if not flag]
    rows = dmap.select(nodes=plain) if plain else np.zeros(0, dtype=np.intp)
    if keyed:
        rows = np.union1d(rows, dmap.indices(keyed, missing="skip"))
    return rows.astype(np.intp)
