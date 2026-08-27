"""Kammer Effective Independence (EFI/EI) sensor placement.

Given the target mode shapes evaluated at a set of candidate DOFs, EFI ranks
each candidate by its contribution to the linear independence of the target
mode partitions, and iteratively discards the least informative one.

The effective independence distribution is the diagonal of the orthogonal
projector onto the column space of ``Phi``::

    E = Phi (Phi^T Phi)^-1 Phi^T,      ED_i = E[i, i]

``ED_i`` lies in ``[0, 1]``, sums to the number of target modes, and equals
the fractional contribution of candidate ``i`` to the determinant of the
Fisher information matrix — dropping the smallest ``ED`` therefore costs the
least determinant, which is Kammer's criterion.  It is computed here from a
thin SVD (``ED_i = ||U[i, :]||^2``) rather than by forming ``(Phi^T Phi)^-1``,
which squares the condition number.

Reference: D. C. Kammer, "Sensor placement for on-orbit modal identification
and correlation of large space structures", J. Guidance, Control and
Dynamics 14(2), 1991.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..correlation._linalg import as_mode_matrix
from ._result import IdSequenceMixin

__all__ = ["EFIResult", "effective_independence", "efi_distribution"]


def efi_distribution(phi: ArrayLike, *, rcond: float = 1e-12) -> NDArray[np.float64]:
    """Effective independence distribution ``ED`` of the rows of ``phi``.

    ``ED[i]`` is the leverage of candidate DOF ``i``; ``sum(ED)`` equals the
    numerical rank of ``phi`` (the number of independent target modes).
    """
    p = as_mode_matrix(phi, "phi")
    if p.size == 0:
        return np.zeros(p.shape[0])
    u, s, _ = np.linalg.svd(p, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros(p.shape[0])
    rank = int(np.count_nonzero(s > rcond * s[0]))
    u = u[:, :rank]
    return np.real(u * u.conj()).sum(axis=1)


@dataclass
class EFIResult(IdSequenceMixin):
    """Outcome of :func:`effective_independence`.

    The object behaves like ``dofs``, the ranked list of selected candidate
    ids (``np.asarray(result)``, ``len``, iteration and indexing all use it).
    """

    dofs: NDArray[Any]
    efi: NDArray[np.float64]
    index: NDArray[np.intp]
    ranking: NDArray[Any] = field(default_factory=lambda: np.empty(0))
    removed: NDArray[Any] = field(default_factory=lambda: np.empty(0))
    ed_initial: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    history: list[tuple[int, float]] = field(default_factory=list)
    n_target_modes: int = 0
    method: str = "efi"

    _id_field = "dofs"

    #: Alias of :attr:`dofs`.
    @property
    def selected(self) -> NDArray[Any]:
        return self.dofs

    @property
    def values(self) -> NDArray[np.float64]:
        """Alias of :attr:`efi`."""
        return self.efi

    @property
    def n_sensors(self) -> int:
        return int(np.size(self.dofs))

    def table(self) -> str:
        head = f"{'rank':>5} {'dof':>10} {'EFI':>10}"
        lines = [head, "-" * len(head)]
        for k, (d, e) in enumerate(
            zip(np.atleast_1d(self.dofs), np.atleast_1d(self.efi), strict=True)
        ):
            lines.append(f"{k:>5} {d!s:>10} {float(e):>10.5f}")
        lines.append(
            f"sum(EFI) = {float(np.sum(self.efi)):.5f} (target modes: {self.n_target_modes})"
        )
        return "\n".join(lines)


def effective_independence(
    phi: ArrayLike,
    n_sensors: int | None = None,
    *,
    candidate_dofs: ArrayLike | None = None,
    mass: ArrayLike | None = None,
    freq_hz: ArrayLike | None = None,
    method: str = "efi",
    batch: int | float = 1,
    keep: ArrayLike | None = None,
    rcond: float = 1e-12,
) -> EFIResult:
    """Select sensor DOFs by Kammer's Effective Independence method.

    Parameters
    ----------
    phi:
        Target mode shapes at the candidate DOFs, ``(n_candidate, n_mode)``.
        Rows are candidate sensor DOFs (translations of the measurable
        directions), columns the target modes from
        :func:`~femtools.pretest.target_modes.select_target_modes`.
    n_sensors:
        Number of sensors to retain.  Defaults to the number of target modes,
        which is the minimum for an invertible Fisher information matrix.
    candidate_dofs:
        Ids reported for the rows of ``phi`` (DOF numbers, labels, ...).
        Defaults to ``0 .. n_candidate-1``.
    mass:
        Optional lumped mass (or any positive weight) per candidate DOF.  The
        rows are scaled by ``sqrt(mass)``, giving the kinetic-energy weighted
        variant that favours high-energy locations.
    freq_hz:
        Target mode frequencies, required for ``method="efi-dpr"``.
    method:
        ``"efi"`` (default) or ``"efi-dpr"``, which multiplies ED by the
        driving-point residue ``sum_j phi_ij^2 / omega_j`` so that sensors
        also give a strong response (Kammer's EI-DPR).
    batch:
        Candidates removed per iteration: an integer count, or a fraction in
        ``(0, 1)`` of the remaining candidates.  ``1`` (default) is the exact
        sequential algorithm; larger values trade accuracy for speed on very
        large candidate sets.
    keep:
        Candidate ids that must be retained (e.g. drive-point locations).
    rcond:
        Relative singular value cutoff used when the mode partition is rank
        deficient.

    Returns
    -------
    EFIResult
        ``result.dofs`` are the selected candidate ids ranked by their final
        EFI value (largest first) and ``result.efi`` the matching values;
        ``result.ranking`` ranks *all* candidates by elimination order and
        ``result.removed`` lists the discarded ones, worst first.
    """
    p = as_mode_matrix(phi, "phi")
    n_cand, n_mode = p.shape
    if n_cand == 0 or n_mode == 0:
        raise ValueError("phi must contain at least one candidate DOF and one mode")

    ids = np.arange(n_cand) if candidate_dofs is None else np.asarray(candidate_dofs).reshape(-1)
    if ids.size != n_cand:
        raise ValueError(f"candidate_dofs has {ids.size} entries but phi has {n_cand} rows")

    work = p
    if mass is not None:
        m = np.asarray(mass, dtype=float).reshape(-1)
        if m.size == 1:
            m = np.full(n_cand, float(m[0]))
        if m.size != n_cand:
            raise ValueError(f"mass has {m.size} entries but phi has {n_cand} rows")
        if np.any(m < 0.0):
            raise ValueError("mass weights must be non-negative")
        work = np.sqrt(m)[:, None] * p

    key = str(method).lower().replace("_", "-")
    if key not in ("efi", "ei", "efi-dpr", "ei-dpr"):
        raise ValueError(f"unknown method {method!r}; use 'efi' or 'efi-dpr'")
    dpr = None
    if key.endswith("dpr"):
        if freq_hz is None:
            raise ValueError("method='efi-dpr' requires freq_hz")
        omega = 2.0 * np.pi * np.asarray(freq_hz, dtype=float).reshape(-1)
        if omega.size != n_mode:
            raise ValueError(f"freq_hz has {omega.size} entries but phi has {n_mode} modes")
        if np.any(omega <= 0.0):
            raise ValueError("efi-dpr requires strictly positive frequencies")
        dpr = (np.abs(work) ** 2 / omega[None, :]).sum(axis=1)

    target = n_mode if n_sensors is None else int(n_sensors)
    if target < n_mode:
        raise ValueError(
            f"n_sensors={target} is below the {n_mode} target modes; the mode "
            "partition would be rank deficient"
        )
    if target > n_cand:
        raise ValueError(f"n_sensors={target} exceeds the {n_cand} candidates")

    locked = np.zeros(n_cand, dtype=bool)
    if keep is not None:
        wanted = np.asarray(keep).reshape(-1)
        pos = np.flatnonzero(np.isin(ids, wanted))
        if pos.size != np.unique(wanted).size:
            missing = set(np.unique(wanted).tolist()) - set(ids[pos].tolist())
            raise ValueError(f"keep contains unknown candidate ids: {sorted(missing)}")
        locked[pos] = True
        if locked.sum() > target:
            raise ValueError(f"keep requests {int(locked.sum())} sensors but n_sensors={target}")

    ed_initial = efi_distribution(work, rcond=rcond)
    alive = np.ones(n_cand, dtype=bool)
    removed: list[int] = []
    history: list[tuple[int, float]] = []
    ed = ed_initial

    while int(alive.sum()) > target:
        rows = np.flatnonzero(alive)
        ed = efi_distribution(work[rows], rcond=rcond)
        score = ed if dpr is None else ed * dpr[rows]
        history.append((int(rows.size), float(score.min())))

        movable = ~locked[rows]
        if not movable.any():  # pragma: no cover - guarded by the keep check
            break
        n_remove = _batch_size(batch, int(rows.size), target, int(movable.sum()))
        order = np.argsort(np.where(movable, score, np.inf), kind="stable")
        drop = rows[order[:n_remove]]
        alive[drop] = False
        removed.extend(drop.tolist())

    rows = np.flatnonzero(alive)
    ed = efi_distribution(work[rows], rcond=rcond)
    final_score = ed if dpr is None else ed * dpr[rows]
    history.append((int(rows.size), float(final_score.min()) if rows.size else 0.0))

    order = np.argsort(-final_score, kind="stable")
    sel_rows = rows[order]
    ranking = (
        np.concatenate([ids[sel_rows], ids[np.asarray(removed[::-1], dtype=np.intp)]])
        if removed
        else ids[sel_rows]
    )

    return EFIResult(
        dofs=ids[sel_rows],
        efi=final_score[order],
        index=sel_rows.astype(np.intp),
        ranking=ranking,
        removed=ids[np.asarray(removed, dtype=np.intp)]
        if removed
        else np.empty(0, dtype=ids.dtype),
        ed_initial=ed_initial,
        history=history,
        n_target_modes=n_mode,
        method=key,
    )


def _batch_size(batch: int | float, n_alive: int, target: int, n_movable: int) -> int:
    """Number of candidates to drop in one iteration."""
    surplus = n_alive - target
    if isinstance(batch, float) and not float(batch).is_integer():
        if not 0.0 < batch < 1.0:
            raise ValueError("a fractional batch must lie in (0, 1)")
        size = max(1, int(np.floor(batch * n_alive)))
    else:
        size = int(batch)
        if size < 1:
            raise ValueError("batch must be >= 1")
    return max(1, min(size, surplus, n_movable))
