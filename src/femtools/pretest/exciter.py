"""Exciter placement: driving-point residues and drive point selection.

Sensor placement decides what the test can *see*; exciter placement decides
what it can *drive*.  A shaker (or hammer) sitting on a node line of a target
mode leaves that mode out of the measured FRF column no matter how good the
sensor set is, so a pretest picks the drive points that give every target mode
a strong response.

The measure is the *driving point residue*.  For mass-normalized modes the
receptance at DOF ``i`` is

``H_ii(w) = sum_j  phi[i,j]^2 / (w_j^2 - w^2 + 2 i zeta_j w_j w)``

whose residue at the pole of mode ``j`` has magnitude

``DPR[i, j] = |phi[i, j]|^2 / (2 m_j w_dj)``,   ``w_dj = w_j sqrt(1 - zeta_j^2)``

with ``m_j`` the generalized mass (1 for mass-normalized modes).  Up to the
constant factor 2 this is the ``phi^2 / w`` weighting of Kammer's EI-DPR
(:func:`~femtools.pretest.efi.effective_independence` with
``method="efi-dpr"``), which uses the same quantity to bias *sensor* placement
towards responsive locations.

:func:`select_exciters` ranks drive points by that residue.  Its default
criterion is deliberately not the total response: summing over modes is
maximized by a location that shouts in one mode and is silent in another,
which is exactly the failure a pretest must avoid.  Instead it maximizes the
*weakest* target mode's normalized residue over the chosen set (a max-min
cover), so adding an exciter is only worthwhile when it improves the mode that
is currently driven worst.

References: D. C. Kammer, "Sensor placement for on-orbit modal identification
and correlation of large space structures", J. Guidance, Control and Dynamics
14(2), 1991; D. J. Ewins, *Modal Testing: Theory, Practice and Application*,
2nd ed., ch. 3 (residues of the driving point FRF).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..correlation._linalg import as_mode_matrix, mode_frequencies, safe_divide
from ._result import IdSequenceMixin

__all__ = [
    "DrivingPointResidues",
    "ExciterSelection",
    "driving_point_residues",
    "select_exciters",
]


def _per_mode(
    value: ArrayLike, n_mode: int, name: str, *, allow_scalar: bool = True
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1 and allow_scalar and n_mode != 1:
        return np.full(n_mode, float(arr[0]))
    if arr.size != n_mode:
        raise ValueError(f"{name} has {arr.size} entries but there are {n_mode} modes")
    return arr


@dataclass
class DrivingPointResidues:
    """Driving point residues of a candidate DOF set (:func:`driving_point_residues`).

    ``np.asarray(result)`` gives the ``(n_candidate, n_mode)`` residue matrix.
    """

    residues: NDArray[np.float64]
    dofs: NDArray[Any]
    freq_hz: NDArray[np.float64]
    damping: NDArray[np.float64]
    generalized_mass: NDArray[np.float64]

    @property
    def dpr(self) -> NDArray[np.float64]:
        """Residue summed over the modes, one value per candidate DOF."""
        return self.residues.sum(axis=1)

    @property
    def normalized(self) -> NDArray[np.float64]:
        """Residues scaled per mode by the best candidate for that mode.

        Modes differ in residue by orders of magnitude — a low-frequency
        bending mode dwarfs a high-frequency local one — so the raw matrix
        cannot be compared across columns.  Here every column peaks at 1 and
        an entry is the fraction of the best achievable response for that
        mode; a zero marks a node point.
        """
        peak = self.residues.max(axis=0) if self.residues.size else np.zeros(0)
        return safe_divide(self.residues, peak[None, :])

    @property
    def best_index(self) -> NDArray[np.intp]:
        """Row of the best drive point for each mode."""
        if self.residues.size == 0:
            return np.zeros(0, dtype=np.intp)
        return np.argmax(self.residues, axis=0).astype(np.intp)

    @property
    def best_dofs(self) -> NDArray[Any]:
        """Id of the best drive point for each mode."""
        return self.dofs[self.best_index]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.residues.shape

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        arr = self.residues
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.array(arr, copy=True) if copy else arr

    def __getitem__(self, item: Any) -> Any:
        return self.residues[item]

    def table(self, limit: int = 20) -> str:
        """Plain-text ranking of the candidates by total residue."""
        order = np.argsort(-self.dpr, kind="stable")[: max(int(limit), 0)]
        norm = self.normalized
        head = f"{'rank':>5} {'dof':>10} {'DPR':>12} {'min/mode':>10} {'weakest':>8}"
        lines = [head, "-" * len(head)]
        for rank, row in enumerate(order):
            weak = int(np.argmin(norm[row])) if norm.shape[1] else -1
            lines.append(
                f"{rank:>5} {self.dofs[row]!s:>10} {self.dpr[row]:>12.5g} "
                f"{norm[row].min() if norm.shape[1] else 0.0:>10.4f} {weak:>8}"
            )
        if order.size < self.residues.shape[0]:
            lines.append(f"... {self.residues.shape[0] - order.size} more candidates")
        return "\n".join(lines)


def driving_point_residues(
    phi: ArrayLike,
    freq_hz: ArrayLike | None = None,
    *,
    dofs: ArrayLike | None = None,
    damping: ArrayLike | None = None,
    generalized_mass: ArrayLike | None = None,
) -> DrivingPointResidues:
    """Driving point residue of every candidate DOF, per target mode.

    ``DPR[i, j] = |phi[i, j]|^2 / (2 m_j w_dj)`` — the magnitude of the
    residue of the driving point receptance ``H_ii`` at the pole of mode
    ``j``, i.e. how strongly a force applied at DOF ``i`` drives that mode
    (and, reciprocally, how strongly the mode answers at that DOF).

    Parameters
    ----------
    phi:
        Mode shapes at the candidate drive points, ``(n_candidate, n_mode)``.
        A :class:`~femtools.pretest.candidates.CandidateSet` or a
        :class:`~femtools.fea.eigen.ModalResult` is accepted and supplies its
        frequencies on its own.
    freq_hz:
        Target mode frequencies [Hz].  Inherited from ``phi`` when it carries
        them; ``phi`` must be mass-normalized or ``generalized_mass`` given.
    dofs:
        Ids for the rows of ``phi``; defaults to ``0 .. n_candidate-1``.  Pass
        ``CandidateSet.dofs`` to get global DOF numbers back.
    damping:
        Modal damping ratio, scalar or per mode.  Only shifts the pole to the
        damped frequency ``w_d = w sqrt(1 - zeta^2)``; below a few percent
        this is a sub-percent correction.
    generalized_mass:
        Modal masses ``m_j = phi_j^T M phi_j``; defaults to 1, i.e.
        mass-normalized modes.

    Returns
    -------
    DrivingPointResidues
        ``result.residues`` is the ``(n_candidate, n_mode)`` matrix,
        ``result.normalized`` its per-mode scaled version (1 at the best
        candidate for each mode) and ``result.dpr`` the sum over modes.
    """
    p = as_mode_matrix(phi, "phi")
    n_cand, n_mode = p.shape
    if n_cand == 0 or n_mode == 0:
        raise ValueError("phi must contain at least one candidate DOF and one mode")

    ids = np.arange(n_cand) if dofs is None else np.asarray(dofs).reshape(-1)
    if ids.size != n_cand:
        raise ValueError(f"dofs has {ids.size} entries but phi has {n_cand} rows")

    if freq_hz is None:
        freq_hz = mode_frequencies(phi)
    if freq_hz is None:
        raise ValueError("freq_hz is required (or pass a modal result carrying it)")
    freqs = _per_mode(freq_hz, n_mode, "freq_hz", allow_scalar=False)
    if np.any(freqs <= 0.0):
        raise ValueError("driving point residues need strictly positive frequencies")

    zeta = np.zeros(n_mode) if damping is None else _per_mode(damping, n_mode, "damping")
    if np.any(zeta < 0.0) or np.any(zeta >= 1.0):
        raise ValueError("damping ratios must lie in [0, 1)")
    gen = (
        np.ones(n_mode)
        if generalized_mass is None
        else _per_mode(generalized_mass, n_mode, "generalized_mass")
    )
    if np.any(gen <= 0.0):
        raise ValueError("generalized_mass must be strictly positive")

    omega_d = 2.0 * np.pi * freqs * np.sqrt(1.0 - zeta**2)
    residues = np.real(p * p.conj()) / (2.0 * gen * omega_d)[None, :]
    return DrivingPointResidues(
        residues=residues,
        dofs=ids,
        freq_hz=freqs,
        damping=zeta,
        generalized_mass=gen,
    )


@dataclass
class ExciterSelection(IdSequenceMixin):
    """Selected exciter DOFs (behaves like the array of selected ids)."""

    dofs: NDArray[Any]
    index: NDArray[np.intp]
    coverage: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    residues: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    dpr: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    score: float = float("nan")
    history: list[float] = field(default_factory=list)
    threshold: float | None = None
    criterion: str = "min"

    _id_field = "dofs"

    @property
    def selected(self) -> NDArray[Any]:
        """Alias of :attr:`dofs`."""
        return self.dofs

    @property
    def n_exciters(self) -> int:
        return int(np.size(self.dofs))

    @property
    def weakest_mode(self) -> int:
        """Target mode driven worst by the selected set (-1 when unknown)."""
        if self.coverage.size == 0:
            return -1
        return int(np.argmin(self.coverage))

    @property
    def covered(self) -> NDArray[np.bool_]:
        """Per-mode flag ``coverage >= threshold`` (all True without one)."""
        if self.threshold is None:
            return np.ones(self.coverage.shape, dtype=bool)
        return self.coverage >= self.threshold

    def table(self) -> str:
        """Plain-text report of the drive points and the mode coverage."""
        head = f"{'#':>3} {'dof':>10} {'DPR':>12}"
        lines = [head, "-" * len(head)]
        for k, (d, s) in enumerate(
            zip(np.atleast_1d(self.dofs), np.atleast_1d(self.dpr), strict=True)
        ):
            lines.append(f"{k:>3} {d!s:>10} {float(s):>12.5g}")
        lines.append("")
        head2 = f"{'mode':>5} {'coverage':>10}"
        lines += [head2, "-" * len(head2)]
        for j, c in enumerate(np.atleast_1d(self.coverage)):
            flag = "" if self.threshold is None or c >= self.threshold else "  <- weak"
            lines.append(f"{j:>5} {float(c):>10.4f}{flag}")
        lines.append(f"worst mode coverage = {self.score:.4f} (criterion: {self.criterion})")
        return "\n".join(lines)


def _lexi_argmax(values: NDArray[np.float64]) -> int:
    """Row of ``values`` that is lexicographically largest, lowest index first.

    ``values`` holds each candidate's coverage vector sorted ascending, so the
    comparison is "best worst mode, then best second-worst mode, ...".  A plain
    ``argmax`` of the minimum would leave the frequent ties (identical worst
    mode, e.g. a mode nobody drives) to be broken arbitrarily.
    """
    n_row, n_col = values.shape
    keys = [-np.arange(n_row, dtype=float)] + [values[:, j] for j in range(n_col - 1, -1, -1)]
    return int(np.lexsort(keys)[-1])


def _set_score(coverage: NDArray[np.float64], key: str) -> tuple[float, ...]:
    """Comparable score of a selected set from the coverage it achieves."""
    if key == "sum":
        return (float(coverage.sum()),)
    return tuple(np.sort(coverage).tolist())


def _best_candidate(
    norm: NDArray[np.float64],
    base: NDArray[np.float64],
    allowed: NDArray[np.bool_],
    key: str,
) -> tuple[int, tuple[float, ...]]:
    """Candidate that best complements the coverage ``base`` already achieved."""
    gain = np.maximum(norm, base[None, :])
    if key == "sum":
        score = np.where(allowed, gain.sum(axis=1), -np.inf)
        row = int(np.argmax(score))
        return row, (float(score[row]),)
    blocked = np.full((norm.shape[0], 1), -np.inf)
    ranked = np.where(allowed[:, None], np.sort(gain, axis=1), blocked)
    row = _lexi_argmax(ranked)
    return row, tuple(ranked[row].tolist())


def select_exciters(
    phi: ArrayLike,
    n_exciters: int = 1,
    freq_hz: ArrayLike | None = None,
    *,
    dofs: ArrayLike | None = None,
    damping: ArrayLike | None = None,
    generalized_mass: ArrayLike | None = None,
    residues: Any = None,
    criterion: str = "min",
    mode_weights: ArrayLike | None = None,
    threshold: float | None = None,
    keep: ArrayLike | None = None,
    exclude: ArrayLike | None = None,
    coords: ArrayLike | None = None,
    min_separation: float | None = None,
    refine: bool = True,
) -> ExciterSelection:
    """Choose the drive points of a modal test by driving point residue.

    The set is built greedily, then improved by exchanges.  With the default
    ``criterion="min"`` each step
    adds the candidate that maximizes the *worst* target mode's normalized
    residue over the whole selected set — a mode counts as driven as soon as
    one exciter drives it, so the criterion is the max-min cover

    ``score(S) = min_j  w_j max_{i in S} DPR_norm[i, j]``

    and a second shaker is only worth its channel when it lifts the mode that
    the first one drives worst.  Greedy alone is a poor fit for that criterion
    — the first pick is made before the others are known — so the set is then
    refined by exchanges: each drive point in turn is replaced by the best
    alternative, until no single swap improves the score.  The outcome is a
    local optimum — the exact problem is a max-min cover, whose exhaustive
    search is exponential in the number of drive points — but the exchange
    pass removes most of the gap that greedy alone leaves.

    Both criteria compare the *normalized* residues (:attr:`\
DrivingPointResidues.normalized`).  Raw residues fall off as ``1/w``, so
    summing them over modes would rank a drive point by how well it excites
    the lowest target mode and by almost nothing else.

    Parameters
    ----------
    phi:
        Mode shapes at the candidate drive points, ``(n_candidate, n_mode)``;
        a :class:`~femtools.pretest.candidates.CandidateSet` works directly.
    n_exciters:
        Number of drive points to select.
    freq_hz, damping, generalized_mass:
        Passed to :func:`driving_point_residues`.
    dofs:
        Ids for the rows of ``phi``; defaults to ``0 .. n_candidate-1``.
    residues:
        Pre-computed :class:`DrivingPointResidues` (or a raw
        ``(n_candidate, n_mode)`` matrix), which skips the recomputation.
    criterion:
        ``"min"`` (default) maximizes the weakest mode as above; ``"sum"``
        maximizes the total normalized residue, which gives the strongest
        overall response but may ignore a mode entirely.
    mode_weights:
        Per-mode importance ``w_j`` applied to the normalized residues, e.g.
        to insist on the modes an update campaign depends on.
    threshold:
        Normalized residue below which a mode counts as *not* driven; only
        reported (:attr:`ExciterSelection.covered`), never enforced, so a
        mode no candidate can drive does not fail the selection.
    keep:
        Candidate ids that must be selected (a fixed shaker attachment).
        They are taken first, in the order given.
    exclude:
        Candidate ids that must not be selected (no access, a bearing, ...).
    coords:
        ``(n_candidate, 3)`` coordinates of the candidate DOFs, required by
        ``min_separation``.  ``CandidateSet.coords`` provides them.
    min_separation:
        Minimum distance between two drive points.  Exciters crowded on the
        same panel measure the same thing; this enforces a spatial spread.
        The selection stops early if no candidate is far enough away.
    refine:
        Run the exchange pass after the greedy build (default).  Turn it off
        to see the plain greedy sequence, e.g. to compare successive set sizes.

    Returns
    -------
    ExciterSelection
        ``result.dofs`` are the drive point ids in selection order,
        ``result.coverage`` the per-mode normalized residue achieved by the
        set and ``result.score`` its worst entry.
    """
    key = str(criterion).lower()
    if key not in ("min", "sum"):
        raise ValueError(f"unknown criterion {criterion!r}; use 'min' or 'sum'")

    dpr = _as_residues(
        phi,
        residues,
        freq_hz=freq_hz,
        dofs=dofs,
        damping=damping,
        generalized_mass=generalized_mass,
    )
    ids = dpr.dofs
    n_cand, n_mode = dpr.residues.shape

    target = int(n_exciters)
    if not 1 <= target <= n_cand:
        raise ValueError(f"n_exciters must be within 1..{n_cand}")

    norm = dpr.normalized
    if mode_weights is not None:
        w = np.asarray(mode_weights, dtype=float).reshape(-1)
        if w.size != n_mode:
            raise ValueError(f"mode_weights has {w.size} entries, expected {n_mode}")
        if np.any(w < 0.0):
            raise ValueError("mode_weights must be non-negative")
        norm = norm * w[None, :]

    available = np.ones(n_cand, dtype=bool)
    if exclude is not None:
        available &= ~np.isin(ids, np.asarray(exclude).reshape(-1))
    forced = _lookup_ids(ids, keep, "keep") if keep is not None else np.zeros(0, dtype=np.intp)
    if forced.size > target:
        raise ValueError(f"keep requests {forced.size} exciters but n_exciters={target}")
    if forced.size and not available[forced].all():
        clash = ids[forced[~available[forced]]].tolist()
        raise ValueError(f"candidate ids {clash} are both kept and excluded")

    xyz = None
    spacing = 0.0
    if min_separation is not None:
        if coords is None:
            raise ValueError("min_separation requires coords")
        spacing = float(min_separation)
        if spacing < 0.0:
            raise ValueError("min_separation must be non-negative")
        xyz = np.asarray(coords, dtype=float).reshape(n_cand, -1)

    def feasible(others: list[int]) -> NDArray[np.bool_]:
        """Candidates still selectable next to the ones in ``others``."""
        ok = available.copy()
        ok[others] = False
        if xyz is not None and others:
            far = np.linalg.norm(xyz[:, None, :] - xyz[None, others, :], axis=2)
            ok &= (far >= spacing).all(axis=1)
        return ok

    chosen: list[int] = []
    history: list[float] = []
    coverage = np.zeros(n_mode)

    while len(chosen) < target:
        if len(chosen) < forced.size:
            pick = int(forced[len(chosen)])
        else:
            allowed = feasible(chosen)
            if not allowed.any():
                break
            pick, _ = _best_candidate(norm, coverage, allowed, key)
        chosen.append(pick)
        coverage = np.maximum(coverage, norm[pick])
        history.append(float(coverage.min()) if n_mode else 0.0)

    if refine and len(chosen) > forced.size + 1:
        chosen, coverage = _exchange(chosen, int(forced.size), norm, feasible, key)
        if n_mode and history and float(coverage.min()) != history[-1]:
            history.append(float(coverage.min()))

    rows = np.asarray(chosen, dtype=np.intp)
    return ExciterSelection(
        dofs=ids[rows],
        index=rows,
        coverage=coverage,
        residues=dpr.residues[rows],
        dpr=dpr.dpr[rows],
        score=float(coverage.min()) if n_mode else 0.0,
        history=history,
        threshold=None if threshold is None else float(threshold),
        criterion=key,
    )


def _exchange(
    chosen: list[int],
    n_forced: int,
    norm: NDArray[np.float64],
    feasible: Callable[[list[int]], NDArray[np.bool_]],
    key: str,
    max_sweeps: int = 20,
) -> tuple[list[int], NDArray[np.float64]]:
    """Improve a selection by replacing one drive point at a time.

    Greedy commits to its first pick before it knows the rest of the set,
    which the max-min criterion punishes.  Sweeping over the slots and
    replacing each by the best complement of the *other* slots repairs that at
    the cost of one greedy step per slot and sweep.  Only strict improvements
    are accepted, so the loop terminates.
    """
    current = list(chosen)
    best = _set_score(norm[current].max(axis=0), key)
    for _ in range(max_sweeps):
        improved = False
        for slot in range(n_forced, len(current)):
            others = [c for i, c in enumerate(current) if i != slot]
            base = norm[others].max(axis=0) if others else np.zeros(norm.shape[1])
            allowed = feasible(others)
            if not allowed.any():
                continue
            row, score = _best_candidate(norm, base, allowed, key)
            if score > best:
                current[slot] = row
                best = score
                improved = True
        if not improved:
            break
    return current, norm[current].max(axis=0)


def _as_residues(
    phi: ArrayLike,
    residues: Any,
    *,
    freq_hz: ArrayLike | None,
    dofs: ArrayLike | None,
    damping: ArrayLike | None,
    generalized_mass: ArrayLike | None,
) -> DrivingPointResidues:
    """Reuse a pre-computed residue set, or compute one from ``phi``."""
    if residues is None:
        return driving_point_residues(
            phi,
            freq_hz,
            dofs=dofs,
            damping=damping,
            generalized_mass=generalized_mass,
        )
    if isinstance(residues, DrivingPointResidues):
        return residues
    matrix = np.asarray(residues, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"residues must be 2-D, got shape {matrix.shape}")
    if np.any(matrix < 0.0):
        raise ValueError("residues must be non-negative")
    n_cand, n_mode = matrix.shape
    ids = np.arange(n_cand) if dofs is None else np.asarray(dofs).reshape(-1)
    if ids.size != n_cand:
        raise ValueError(f"dofs has {ids.size} entries but residues has {n_cand} rows")
    return DrivingPointResidues(
        residues=matrix,
        dofs=ids,
        freq_hz=np.full(n_mode, np.nan),
        damping=np.zeros(n_mode),
        generalized_mass=np.ones(n_mode),
    )


def _lookup_ids(ids: NDArray[Any], wanted: ArrayLike, name: str) -> NDArray[np.intp]:
    """Rows of ``ids`` addressed by ``wanted``, in the order given."""
    items = np.atleast_1d(np.asarray(wanted)).reshape(-1)
    rows: list[int] = []
    for item in items.tolist():
        hit = np.flatnonzero(ids == item)
        if hit.size == 0:
            raise ValueError(f"{name} contains unknown candidate id: {item!r}")
        rows.append(int(hit[0]))
    out = np.asarray(rows, dtype=np.intp)
    if np.unique(out).size != out.size:
        raise ValueError(f"{name} contains duplicate candidate ids")
    return out
