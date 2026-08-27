"""Target mode selection: modal effective mass and frequency-band filtering.

A pretest starts by deciding *which* modes the test must capture.  The usual
criterion is the modal effective mass: the share of the rigid-body mass that
each mode mobilizes in a given direction.  Modes with a negligible effective
mass are hard to excite and hard to measure, and are normally dropped from
the target set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..correlation._linalg import as_mode_matrix, safe_divide, weighted
from ..correlation.dofmap import DOFMap
from ._result import IdSequenceMixin

__all__ = [
    "DIRECTIONS",
    "EffectiveMassResult",
    "TargetModeSelection",
    "rigid_body_modes",
    "effective_mass",
    "select_target_modes",
]

#: Rigid-body direction labels, in the order of the returned columns.
DIRECTIONS: tuple[str, ...] = ("TX", "TY", "TZ", "RX", "RY", "RZ")


def rigid_body_modes(
    coords: ArrayLike | None = None,
    *,
    ref_point: ArrayLike | None = None,
    dofs_per_node: int = 6,
    dof_map: Any = None,
    n_dof: int | None = None,
) -> NDArray[np.float64]:
    """Unit rigid-body vectors ``R`` of shape ``(n_dof, 6)``.

    Column ``d`` is the displacement field of a unit rigid translation
    (``d < 3``) or a unit rotation about the reference point (``d >= 3``),
    so that ``R^T M R`` is the 6x6 rigid-body mass matrix about that point.

    Parameters
    ----------
    coords:
        ``(n_node, 3)`` nodal coordinates.  Required for the rotational
        columns; without them only the three translations are built (the
        rotational columns are zero).
    ref_point:
        Rotation centre; defaults to the origin (use the centre of gravity
        for the classical effective-mass table).
    dofs_per_node:
        3 or 6, used when ``dof_map`` is not given.  DOFs are assumed to be
        node-major (all DOFs of node 0, then node 1, ...).
    dof_map:
        Explicit :class:`~femtools.correlation.dofmap.DOFMap` (or anything it
        can be built from) describing the DOF order; ``coords`` must then be
        given per *node id* through ``coords`` indexed the same way as
        ``np.unique(dof_map.nodes)``, or as a mapping ``{node: xyz}``.
    n_dof:
        Total DOF count, needed only when neither ``coords`` nor ``dof_map``
        determine it.
    """
    if dof_map is not None:
        dmap = DOFMap.from_mapping(dof_map)
        nodes = dmap.nodes
        comps = dmap.components
        xyz = _node_coords(coords, nodes)
        ndof = len(dmap)
    else:
        if coords is None:
            if n_dof is None:
                raise ValueError("provide coords, dof_map or n_dof")
            if dofs_per_node not in (3, 6):
                raise ValueError("dofs_per_node must be 3 or 6")
            n_node = n_dof // dofs_per_node
            xyz_nodes = np.zeros((n_node, 3))
        else:
            xyz_nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
            n_node = xyz_nodes.shape[0]
        if dofs_per_node not in (3, 6):
            raise ValueError("dofs_per_node must be 3 or 6")
        ndof = n_node * dofs_per_node
        comps = np.tile(np.arange(1, dofs_per_node + 1), n_node).astype(np.int8)
        xyz = np.repeat(xyz_nodes, dofs_per_node, axis=0)

    origin = np.zeros(3) if ref_point is None else np.asarray(ref_point, dtype=float).reshape(3)
    rel = xyz - origin

    r = np.zeros((ndof, 6))
    x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
    for d in range(3):
        r[comps == d + 1, d] = 1.0
    # Translation induced by a unit rotation: u = e_d x r
    tx, ty, tz = comps == 1, comps == 2, comps == 3
    r[ty, 3] = -z[ty]
    r[tz, 3] = y[tz]
    r[tx, 4] = z[tx]
    r[tz, 4] = -x[tz]
    r[tx, 5] = -y[tx]
    r[ty, 5] = x[ty]
    for d in range(3, 6):
        r[comps == d + 1, d] = 1.0
    return r


def _node_coords(coords: Any, nodes: NDArray[np.int64]) -> NDArray[np.float64]:
    if coords is None:
        return np.zeros((nodes.size, 3))
    if isinstance(coords, dict):
        return np.array([np.asarray(coords[int(n)], dtype=float).reshape(3) for n in nodes])
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must be (n_node, 3), got shape {arr.shape}")
    if arr.shape[0] == nodes.size:
        return arr
    unique = np.unique(nodes)
    if arr.shape[0] != unique.size:
        raise ValueError(
            f"coords has {arr.shape[0]} rows but the DOF map covers {unique.size} nodes"
        )
    lookup = {int(n): i for i, n in enumerate(unique.tolist())}
    return arr[[lookup[int(n)] for n in nodes]]


@dataclass
class EffectiveMassResult:
    """Modal participation and effective mass, per mode and direction."""

    effective_mass: NDArray[np.float64]
    participation: NDArray[np.float64]
    total_mass: NDArray[np.float64]
    freq_hz: NDArray[np.float64] | None = None
    directions: tuple[str, ...] = DIRECTIONS

    @property
    def fraction(self) -> NDArray[np.float64]:
        """Effective mass as a fraction of the rigid-body mass per direction."""
        return safe_divide(self.effective_mass, self.total_mass[None, :])

    @property
    def cumulative_fraction(self) -> NDArray[np.float64]:
        """Cumulative :attr:`fraction` in mode order."""
        return np.cumsum(self.fraction, axis=0)

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        arr = self.effective_mass
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.array(arr, copy=True) if copy else arr

    def __getitem__(self, item: Any) -> Any:
        return self.effective_mass[item]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.effective_mass.shape

    def table(self, directions: ArrayLike | None = None) -> str:
        """Plain-text effective mass table (fractions in percent)."""
        cols = (
            list(range(len(self.directions)))
            if directions is None
            else list(np.atleast_1d(directions))
        )
        head = f"{'mode':>5} {'f [Hz]':>10} " + " ".join(f"{self.directions[c]:>8}" for c in cols)
        lines = [head, "-" * len(head)]
        frac = 100.0 * self.fraction
        for j in range(self.effective_mass.shape[0]):
            f = np.nan if self.freq_hz is None else float(self.freq_hz[j])
            lines.append(f"{j:>5} {f:>10.4f} " + " ".join(f"{frac[j, c]:>8.3f}" for c in cols))
        total = 100.0 * self.fraction.sum(axis=0)
        lines.append(f"{'sum':>5} {'':>10} " + " ".join(f"{total[c]:>8.3f}" for c in cols))
        return "\n".join(lines)


def effective_mass(
    phi: ArrayLike,
    mass: Any = None,
    rb_modes: ArrayLike | None = None,
    *,
    coords: ArrayLike | None = None,
    dof_map: Any = None,
    ref_point: ArrayLike | None = None,
    dofs_per_node: int = 6,
    freq_hz: ArrayLike | None = None,
    generalized_mass: ArrayLike | None = None,
) -> EffectiveMassResult:
    """Modal participation factors and effective masses.

    For mass-normalized modes the participation factor of mode ``j`` in
    rigid-body direction ``d`` is ``L[j, d] = phi_j^T M r_d`` and the
    effective mass is ``L[j, d]^2 / m_j`` with ``m_j = phi_j^T M phi_j``.
    Summed over a complete mode set the effective masses recover the
    rigid-body mass ``diag(R^T M R)``.

    Parameters
    ----------
    phi:
        Mode shapes ``(n_dof, n_mode)``.
    mass:
        Mass matrix (dense, sparse, 1-D lumped diagonal, or ``None`` for the
        identity).
    rb_modes:
        Rigid-body vectors ``(n_dof, n_dir)``.  Built from ``coords`` /
        ``dof_map`` / ``dofs_per_node`` by :func:`rigid_body_modes` when
        omitted.
    ref_point:
        Reference point for the rotational directions.
    freq_hz:
        Optional frequencies, carried through to the result for reporting.
    generalized_mass:
        Modal masses ``m_j``; computed from ``phi`` and ``mass`` when omitted.

    Returns
    -------
    EffectiveMassResult
        ``result.effective_mass`` is ``(n_mode, n_dir)``;
        ``result.fraction`` normalizes it by the rigid-body mass.
    """
    p = as_mode_matrix(phi, "phi")
    n_dof, n_mode = p.shape

    if rb_modes is None:
        r = rigid_body_modes(
            coords,
            ref_point=ref_point,
            dofs_per_node=dofs_per_node,
            dof_map=dof_map,
            n_dof=n_dof,
        )
    else:
        r = np.asarray(rb_modes, dtype=float)
        if r.ndim == 1:
            r = r.reshape(-1, 1)
    if r.shape[0] != n_dof:
        raise ValueError(f"rigid-body vectors have {r.shape[0]} rows, expected {n_dof}")

    mr = weighted(mass, r)
    participation = np.real(p.conj().T @ mr)

    if generalized_mass is None:
        mp = weighted(mass, p)
        gen = np.real(np.einsum("ij,ij->j", p.conj(), mp))
    else:
        gen = np.asarray(generalized_mass, dtype=float).reshape(-1)
        if gen.size != n_mode:
            raise ValueError(f"generalized_mass has {gen.size} entries, expected {n_mode}")

    eff = safe_divide(participation**2, gen[:, None])
    total = np.real(np.einsum("ij,ij->j", r, mr))

    labels = (
        DIRECTIONS[: r.shape[1]] if r.shape[1] <= 6 else tuple(f"D{i}" for i in range(r.shape[1]))
    )
    freqs = None if freq_hz is None else np.asarray(freq_hz, dtype=float).reshape(-1)
    if freqs is not None and freqs.size != n_mode:
        raise ValueError(f"freq_hz has {freqs.size} entries, expected {n_mode}")
    return EffectiveMassResult(
        effective_mass=eff,
        participation=participation,
        total_mass=total,
        freq_hz=freqs,
        directions=labels,
    )


@dataclass
class TargetModeSelection(IdSequenceMixin):
    """Selected target modes (behaves like the array of mode indices)."""

    indices: NDArray[np.intp]
    freq_hz: NDArray[np.float64] | None = None
    fraction: NDArray[np.float64] | None = None
    achieved_fraction: NDArray[np.float64] | None = None
    directions: tuple[str, ...] = ()
    reason: str = ""
    rejected: NDArray[np.intp] = field(default_factory=lambda: np.empty(0, dtype=np.intp))

    _id_field = "indices"

    @property
    def n_modes(self) -> int:
        return int(self.indices.size)

    @property
    def frequencies(self) -> NDArray[np.float64]:
        if self.freq_hz is None:
            return np.empty(0)
        return self.freq_hz[self.indices]


def select_target_modes(
    freq_hz: ArrayLike | EffectiveMassResult | None = None,
    effective_mass: ArrayLike | EffectiveMassResult | None = None,
    *,
    f_min: float | None = None,
    f_max: float | None = None,
    n_modes: int | None = None,
    mass_fraction: float | None = 0.9,
    min_fraction: float = 0.0,
    directions: ArrayLike | None = None,
    include: ArrayLike | None = None,
    exclude: ArrayLike | None = None,
) -> TargetModeSelection:
    """Choose the modes a modal test has to capture.

    Selection proceeds in three stages: a frequency-band filter, then an
    effective-mass criterion (every mode above ``min_fraction`` in any
    requested direction, plus, greedily, the largest contributors until
    ``mass_fraction`` of the rigid-body mass is reached in every direction),
    and finally an optional hard cap of ``n_modes``.

    Parameters
    ----------
    freq_hz:
        Natural frequencies [Hz].  An :class:`EffectiveMassResult` may be
        passed here directly, in which case its frequencies are used.
    effective_mass:
        ``(n_mode, n_dir)`` effective mass array or an
        :class:`EffectiveMassResult`.  Without it the selection is purely
        frequency based.
    f_min, f_max:
        Frequency band; modes outside it are rejected.  A typical pretest
        uses ``f_max`` = upper limit of the excitation band, and ``f_min``
        just above 0 to drop rigid-body modes.
    n_modes:
        Maximum number of target modes (best contributors first, then lowest
        frequency).
    mass_fraction:
        Cumulative effective mass fraction to reach per direction, e.g.
        ``0.9``.  ``None`` disables the criterion.
    min_fraction:
        Modes contributing at least this fraction in any direction are
        always kept.
    directions:
        Column indices (or labels) of the directions to consider; default
        all directions present.
    include, exclude:
        Mode indices forced into / out of the target set.

    Returns
    -------
    TargetModeSelection
        Indices sorted ascending (i.e. in frequency order); the object can be
        used directly as an index array.
    """
    em_result: EffectiveMassResult | None = None
    if isinstance(freq_hz, EffectiveMassResult):
        em_result = freq_hz
        freq_hz = em_result.freq_hz
    if isinstance(effective_mass, EffectiveMassResult):
        em_result = effective_mass
        if freq_hz is None:
            freq_hz = em_result.freq_hz

    if em_result is not None:
        em = em_result.fraction
        labels = em_result.directions
    elif effective_mass is not None:
        em = np.asarray(effective_mass, dtype=float)
        if em.ndim == 1:
            em = em.reshape(-1, 1)
        totals = em.sum(axis=0)
        em = safe_divide(em, totals[None, :])
        labels = DIRECTIONS[: em.shape[1]] if em.shape[1] <= 6 else ()
    else:
        em = None
        labels = ()

    freqs = None if freq_hz is None else np.asarray(freq_hz, dtype=float).reshape(-1)
    n = em.shape[0] if em is not None else (freqs.size if freqs is not None else 0)
    if n == 0:
        raise ValueError("provide freq_hz and/or effective_mass")
    if em is not None and freqs is not None and freqs.size != n:
        raise ValueError(f"freq_hz has {freqs.size} entries but effective_mass has {n} modes")

    band = np.ones(n, dtype=bool)
    if freqs is not None:
        if f_min is not None:
            band &= freqs >= f_min
        if f_max is not None:
            band &= freqs <= f_max
    if exclude is not None:
        band[np.asarray(exclude, dtype=int).reshape(-1)] = False

    keep = np.zeros(n, dtype=bool)
    reason = "frequency band"
    achieved: NDArray[np.float64] | None = None

    if em is None:
        keep |= band
    else:
        cols = _direction_columns(directions, labels, em.shape[1])
        sub = em[:, cols]
        if min_fraction > 0.0:
            keep |= band & (sub.max(axis=1) >= min_fraction)
        if mass_fraction is not None:
            keep |= _greedy_mass(sub, band, float(mass_fraction), keep)
            reason = f"effective mass >= {mass_fraction:.3g} per direction"
        elif min_fraction > 0.0:
            reason = f"effective mass >= {min_fraction:.3g}"
        else:
            keep |= band
        achieved = sub[keep].sum(axis=0)

    if include is not None:
        keep[np.asarray(include, dtype=int).reshape(-1)] = True

    idx = np.flatnonzero(keep).astype(np.intp)
    if n_modes is not None and idx.size > n_modes:
        if em is not None:
            cols = _direction_columns(directions, labels, em.shape[1])
            rank = np.argsort(-em[idx][:, cols].max(axis=1), kind="stable")
        elif freqs is not None:
            rank = np.argsort(freqs[idx], kind="stable")
        else:  # pragma: no cover - unreachable, n>0 requires one of the two
            rank = np.arange(idx.size)
        idx = np.sort(idx[rank[:n_modes]])
        reason += f", capped to {n_modes} modes"
        if em is not None:
            cols = _direction_columns(directions, labels, em.shape[1])
            achieved = em[idx][:, cols].sum(axis=0)

    rejected = np.setdiff1d(np.arange(n, dtype=np.intp), idx)
    return TargetModeSelection(
        indices=idx,
        freq_hz=freqs,
        fraction=em,
        achieved_fraction=achieved,
        directions=tuple(labels),
        reason=reason,
        rejected=rejected,
    )


def _direction_columns(
    directions: ArrayLike | None, labels: tuple[str, ...], n_dir: int
) -> NDArray[np.intp]:
    if directions is None:
        return np.arange(n_dir, dtype=np.intp)
    cols: list[int] = []
    for d in np.atleast_1d(np.asarray(directions, dtype=object)):
        if isinstance(d, str):
            key = d.strip().upper()
            if key not in labels:
                raise ValueError(f"unknown direction {d!r}; available: {labels}")
            cols.append(labels.index(key))
        else:
            cols.append(int(d))
    if any(c < 0 or c >= n_dir for c in cols):
        raise ValueError(f"direction index out of range 0..{n_dir - 1}")
    return np.asarray(cols, dtype=np.intp)


def _greedy_mass(
    fraction: NDArray[np.float64],
    band: NDArray[np.bool_],
    target: float,
    already: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Add candidate modes until every direction reaches ``target``."""
    keep = already.copy()
    reachable = fraction[band].sum(axis=0)
    goal = np.minimum(target, reachable)
    while True:
        got = fraction[keep].sum(axis=0)
        short = got < goal - 1e-12
        if not short.any():
            return keep
        available = band & ~keep
        if not available.any():
            return keep
        score = np.where(available, fraction[:, short].max(axis=1), -np.inf)
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            return keep
        keep[best] = True
