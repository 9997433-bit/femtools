"""Sensor-set reduction: MAC-based elimination and kinetic energy ranking.

Two complements to EFI:

* :func:`eliminate_by_mac` drops, one at a time, the candidate DOF whose
  removal leaves the best auto-MAC of the target modes.  It optimizes
  *distinguishability* of the modes directly, which is what the correlation
  step later needs, while EFI optimizes the response covariance.
* :func:`nodal_kinetic_energy` ranks locations by the modal kinetic energy
  they carry — a cheap measure of signal-to-noise which is often used to
  pre-screen candidates before running EFI.

Both work on the candidate partition produced by
:func:`~femtools.pretest.candidates.translational_dofs`, and
:func:`nodal_kinetic_energy` aggregates per node when given the DOF map of a
solved model (``nodal_kinetic_energy(modal.modes, modal.M, modal)``).
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
    "SensorSelection",
    "eliminate_by_mac",
    "nodal_kinetic_energy",
    "aggregate_by_node",
    "select_by_kinetic_energy",
]

_MAX_BLOCK_ELEMENTS = 4_000_000


@dataclass
class SensorSelection(IdSequenceMixin):
    """Selected sensor DOFs (behaves like the array of selected ids)."""

    dofs: NDArray[Any]
    index: NDArray[np.intp]
    mac: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    score: float = float("nan")
    removed: NDArray[Any] = field(default_factory=lambda: np.empty(0))
    history: list[tuple[int, float]] = field(default_factory=list)
    criterion: str = "max"

    _id_field = "dofs"

    @property
    def selected(self) -> NDArray[Any]:
        """Alias of :attr:`dofs`."""
        return self.dofs

    @property
    def n_sensors(self) -> int:
        return int(np.size(self.dofs))

    @property
    def max_off_diagonal(self) -> float:
        """Largest off-diagonal auto-MAC of the target modes at the sensors."""
        if self.mac.size == 0:
            return float("nan")
        mask = ~np.eye(self.mac.shape[0], dtype=bool)
        return float(self.mac[mask].max()) if mask.any() else 0.0


def _gram(p: NDArray[Any]) -> NDArray[Any]:
    return p.conj().T @ p


def _mac_from_gram(gram: NDArray[Any]) -> NDArray[np.float64]:
    d = np.real(np.einsum("...ii->...i", gram))
    num = np.real(gram * gram.conj())
    den = d[..., :, None] * d[..., None, :]
    return safe_divide(num, den)


def _removal_scores(
    p: NDArray[Any], gram: NDArray[Any], criterion: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Score and worst off-diagonal MAC of the set left by removing each row.

    Uses the rank-one downdate ``G - phi_q phi_q^H`` so all candidates are
    evaluated without recomputing any Gram matrix.  The worst off-diagonal is
    returned alongside the criterion score because ``mac_limit`` constrains it
    whatever the criterion ranks by; for ``"max"`` the two coincide.
    """
    k, m = p.shape
    off = ~np.eye(m, dtype=bool)
    scores = np.empty(k)
    worst = np.empty(k)
    block = max(1, min(k, _MAX_BLOCK_ELEMENTS // max(m * m, 1)))
    for start in range(0, k, block):
        chunk = p[start : start + block]
        outer = chunk.conj()[:, :, None] * chunk[:, None, :]
        g_q = gram[None, :, :] - outer
        d = np.real(np.einsum("kii->ki", g_q))
        mac = _mac_from_gram(g_q)
        vals = mac[:, off].reshape(chunk.shape[0], -1)
        top = vals.max(axis=1) if vals.size else np.zeros(chunk.shape[0])
        if criterion == "max":
            s = top
        elif criterion in ("sum", "mean"):
            s = vals.sum(axis=1)
            if criterion == "mean" and vals.shape[1]:
                s = s / vals.shape[1]
        else:
            raise ValueError(f"unknown criterion {criterion!r}; use 'max', 'sum' or 'mean'")
        # A removal that annihilates a mode must never be chosen.
        alive = (d > 0.0).all(axis=1)
        scores[start : start + chunk.shape[0]] = np.where(alive, s, np.inf)
        worst[start : start + chunk.shape[0]] = np.where(alive, top, np.inf)
    return scores, worst


def eliminate_by_mac(
    phi: ArrayLike,
    n_sensors: int | None = None,
    *,
    candidate_dofs: ArrayLike | None = None,
    criterion: str = "max",
    keep: ArrayLike | None = None,
    mac_limit: float | None = None,
) -> SensorSelection:
    """Reduce a candidate DOF set by greedy MAC-based elimination.

    At every step the candidate whose removal yields the lowest off-diagonal
    auto-MAC of the target modes is discarded, so the retained sensors keep
    the target modes as distinguishable as possible.

    Parameters
    ----------
    phi:
        Target mode shapes at the candidate DOFs, ``(n_candidate, n_mode)``.
    n_sensors:
        Number of sensors to retain (default: the number of target modes).
    candidate_dofs:
        Ids for the rows of ``phi``; defaults to ``0 .. n_candidate-1``.
    criterion:
        ``"max"`` (default) minimizes the largest off-diagonal MAC,
        ``"sum"``/``"mean"`` minimize the total off-diagonal MAC, which is
        less sensitive to a single stubborn mode pair.
    keep:
        Candidate ids that must be retained.
    mac_limit:
        Stop early if removing one more sensor would push the largest
        off-diagonal MAC above this value.  Applies to every criterion: the
        limit is a property of the retained set, not of the search order.

    Returns
    -------
    SensorSelection
        Selected ids, their rows in ``phi``, the resulting auto-MAC matrix
        and the elimination history.
    """
    p = as_mode_matrix(phi, "phi")
    n_cand, n_mode = p.shape
    if n_cand == 0 or n_mode == 0:
        raise ValueError("phi must contain at least one candidate DOF and one mode")

    ids = np.arange(n_cand) if candidate_dofs is None else np.asarray(candidate_dofs).reshape(-1)
    if ids.size != n_cand:
        raise ValueError(f"candidate_dofs has {ids.size} entries but phi has {n_cand} rows")

    target = n_mode if n_sensors is None else int(n_sensors)
    if target < 1:
        raise ValueError("n_sensors must be >= 1")
    if target > n_cand:
        raise ValueError(f"n_sensors={target} exceeds the {n_cand} candidates")

    locked = np.zeros(n_cand, dtype=bool)
    if keep is not None:
        wanted = np.unique(np.asarray(keep).reshape(-1))
        pos = np.flatnonzero(np.isin(ids, wanted))
        if pos.size != wanted.size:
            missing = set(wanted.tolist()) - set(ids[pos].tolist())
            raise ValueError(f"keep contains unknown candidate ids: {sorted(missing)}")
        locked[pos] = True
        if locked.sum() > target:
            raise ValueError(f"keep requests {int(locked.sum())} sensors but n_sensors={target}")

    alive = np.ones(n_cand, dtype=bool)
    gram = _gram(p)
    off = ~np.eye(n_mode, dtype=bool)
    removed: list[int] = []
    history: list[tuple[int, float]] = []

    while int(alive.sum()) > target:
        rows = np.flatnonzero(alive)
        scores, worst = _removal_scores(p[rows], gram, criterion)
        scores = np.where(locked[rows], np.inf, scores)
        if mac_limit is not None:
            # The limit constrains the largest off-diagonal MAC of the set
            # that would be left, not the score the criterion ranks by, so a
            # candidate that breaches it is skipped rather than ranked.  The
            # loop then ends only when no removal at all stays inside.
            scores = np.where(worst > mac_limit, np.inf, scores)
        best = int(np.argmin(scores))
        if not np.isfinite(scores[best]):
            break
        q = int(rows[best])
        gram = gram - np.outer(p[q].conj(), p[q])
        alive[q] = False
        removed.append(q)
        history.append((int(alive.sum()), float(scores[best])))

    rows = np.flatnonzero(alive)
    mac = _mac_from_gram(_gram(p[rows]))
    score = float(mac[off].max()) if off.any() else 0.0
    return SensorSelection(
        dofs=ids[rows],
        index=rows.astype(np.intp),
        mac=mac,
        score=score,
        removed=ids[np.asarray(removed, dtype=np.intp)]
        if removed
        else np.empty(0, dtype=ids.dtype),
        history=history,
        criterion=criterion,
    )


def nodal_kinetic_energy(
    phi: ArrayLike,
    mass: Any = None,
    dof_map: Any = None,
    *,
    normalize: bool = True,
    absolute: bool = False,
    aggregate: str | None = None,
    return_ids: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[Any], NDArray[np.float64]]:
    """Modal kinetic energy distribution ``phi * (M phi)``.

    Parameters
    ----------
    phi:
        Mode shapes ``(n_dof, n_mode)``.
    mass:
        Mass matrix, 1-D lumped mass, or ``None`` for the identity (which
        reduces the measure to the squared modal amplitude).
    dof_map:
        DOF map used to sum the DOF energies per node.  When given, the
        result is per node unless ``aggregate="dof"``.
    normalize:
        Scale each mode so its energies sum to 1 (already the case for
        mass-normalized modes and a consistent mass matrix).
    absolute:
        Take magnitudes first.  A consistent mass matrix can produce small
        negative DOF energies; use this when ranking locations.
    aggregate:
        ``"node"`` (default when ``dof_map`` is given) or ``"dof"``.
    return_ids:
        Also return the node ids (or DOF positions) of the rows.

    Returns
    -------
    ndarray or (ids, ndarray)
        ``(n_dof, n_mode)`` — or ``(n_node, n_mode)`` when aggregated.
    """
    p = as_mode_matrix(phi, "phi")
    mp = weighted(mass, p)
    energy = np.real(p.conj() * mp)
    if absolute:
        energy = np.abs(energy)

    if aggregate is None:
        aggregate = "node" if dof_map is not None else "dof"
    if aggregate not in ("node", "dof"):
        raise ValueError(f"unknown aggregate {aggregate!r}; use 'node' or 'dof'")

    ids: NDArray[Any] = np.arange(energy.shape[0])
    if aggregate == "node":
        if dof_map is None:
            raise ValueError("aggregate='node' requires dof_map")
        ids, energy = aggregate_by_node(energy, dof_map)

    if normalize:
        total = energy.sum(axis=0)
        energy = safe_divide(energy, total[None, :])

    return (ids, energy) if return_ids else energy


def aggregate_by_node(
    values: ArrayLike, dof_map: Any
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Sum per-DOF quantities over the DOFs of each node.

    Returns ``(node_ids, summed_values)`` with the node ids sorted ascending.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    dmap = DOFMap.from_mapping(dof_map)
    if len(dmap) != arr.shape[0]:
        raise ValueError(f"dof_map has {len(dmap)} DOF but values have {arr.shape[0]} rows")
    node_ids, inverse = np.unique(dmap.nodes, return_inverse=True)
    out = np.zeros((node_ids.size, arr.shape[1]))
    np.add.at(out, inverse, arr)
    return node_ids, out


def select_by_kinetic_energy(
    phi: ArrayLike,
    n_sensors: int,
    mass: Any = None,
    *,
    candidate_dofs: ArrayLike | None = None,
    mode_weights: ArrayLike | None = None,
) -> SensorSelection:
    """Pick the ``n_sensors`` DOFs carrying the most average kinetic energy.

    A fast pre-screening step: it maximizes signal level but, unlike
    :func:`eliminate_by_mac` or EFI, ignores whether the retained DOFs keep
    the target modes independent.
    """
    p = as_mode_matrix(phi, "phi")
    n_cand, n_mode = p.shape
    ids = np.arange(n_cand) if candidate_dofs is None else np.asarray(candidate_dofs).reshape(-1)
    if ids.size != n_cand:
        raise ValueError(f"candidate_dofs has {ids.size} entries but phi has {n_cand} rows")
    if not 1 <= int(n_sensors) <= n_cand:
        raise ValueError(f"n_sensors must be within 1..{n_cand}")

    energy = np.asarray(
        nodal_kinetic_energy(p, mass, normalize=True, absolute=True, aggregate="dof")
    )
    if mode_weights is None:
        w = np.ones(n_mode)
    else:
        w = np.asarray(mode_weights, dtype=float).reshape(-1)
        if w.size != n_mode:
            raise ValueError(f"mode_weights has {w.size} entries, expected {n_mode}")
    ranked = np.argsort(-(energy @ w), kind="stable")[: int(n_sensors)]
    rows = np.sort(ranked).astype(np.intp)
    mac = _mac_from_gram(_gram(p[rows]))
    off = ~np.eye(n_mode, dtype=bool)
    return SensorSelection(
        dofs=ids[rows],
        index=rows,
        mac=mac,
        score=float(mac[off].max()) if off.any() else 0.0,
        criterion="kinetic-energy",
    )
