"""Updating-parameter selection (effective sensitivity / column subset selection).

Sensitivity-based updating breaks down long before it runs out of parameters:
two parameters whose response columns point in nearly the same direction cannot
be separated by any amount of data, and the normal equations
:math:`(S^T W S)\\Delta p = S^T W \\Delta r` amplify the measurement noise by the
condition number of :math:`A = S^T W S`.  The cure is not more regularisation,
it is *choosing fewer parameters*.

This module ranks and prunes the candidate set from the sensitivity matrix
alone, before any updating run.

The default criterion is the parameter-space analogue of Kammer's Effective
Independence: since

.. math::
    \\det\\!\\left(A_{-j}\\right) = \\det(A)\\;\\left(A^{-1}\\right)_{jj},

dropping the parameter with the largest diagonal entry of :math:`A^{-1}` costs
the least information, and

.. math::
    \\left(A^{-1}\\right)_{jj} = \\frac{1}{A_{jj}\\,(1 - R_j^2)}

is large exactly when a parameter is either insensitive (small :math:`A_{jj}`)
or collinear with the others (:math:`R_j^2 \\to 1`).  One criterion therefore
covers both failure modes, and backward elimination on it is a D-optimal
design.  ``1/(A_{jj} (A^{-1})_{jj}) = 1 - R_j^2`` is reported as the
*independence* of each parameter — the reciprocal of its variance inflation
factor.

Alternatives: ``"qr"`` (one-shot rank-revealing pivoted QR, the classic column
subset selection of Businger & Golub), ``"forward"`` (greedy D-optimal growth
from the empty set), ``"sensitivity"`` (plain column-norm ranking, which sees
magnitude but is blind to collinearity) and ``"correlation"`` (drop the less
sensitive member of every over-correlated pair).

Sensitivities are compared **after normalisation**: an ``E`` in pascals and a
thickness in millimetres are otherwise incomparable.  The default
``normalize="relative"`` uses :math:`\\partial\\ln r/\\partial\\ln p`, which is
what makes the ranking dimensionless.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .parameters import ParameterSet, as_parameters
from .sensitivity import SensitivityResult, sensitivity_matrix

__all__ = ["ParameterSelection", "select_parameters", "parameter_correlation"]


@dataclass
class ParameterSelection:
    """Outcome of :func:`select_parameters`.

    Attributes
    ----------
    indices:
        Column indices of the selected parameters, ascending.
    names:
        Their names.
    scores:
        Per-parameter selection score in ``[0, 1]``, higher = keep.  For the
        ranking methods (``"efi"``, ``"qr"``, ``"forward"``) it is the
        *retention rank*: 1.0 for the parameter the criterion values most,
        decreasing linearly down the elimination order, so it stays meaningful
        even when the raw criterion degenerates on a rank-deficient set.  For
        ``"sensitivity"``/``"correlation"`` it is the column norm relative to
        the largest one.
    independence:
        ``1 - R_j^2`` per parameter for the *full* candidate set — 1.0 means
        the column is orthogonal to all others, 0 means fully collinear.  Note
        that it is identically 0 for every parameter whenever there are more
        candidates than responses, since the columns are then necessarily
        linearly dependent; :attr:`subset_independence` is the value that
        matters once the set has been pruned.
    subset_independence:
        ``1 - R_j^2`` of the selected parameters among *themselves*.
    condition_number, condition_full:
        Condition number of ``S`` restricted to the selection, and of the full
        candidate matrix.
    correlation:
        Parameter cross-correlation matrix of the full candidate set.
    order:
        Elimination / insertion order actually followed, most valuable first.
    """

    indices: np.ndarray
    names: list[str]
    scores: np.ndarray
    independence: np.ndarray
    condition_number: float
    condition_full: float
    correlation: np.ndarray
    rejected: np.ndarray
    subset_independence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    order: list[int] = field(default_factory=list)
    method: str = "efi"
    normalize: str | None = "relative"
    history: list[dict[str, Any]] = field(default_factory=list)
    parameter_names: list[str] = field(default_factory=list)

    # -- container behaviour --------------------------------------------
    def __len__(self) -> int:
        return int(self.indices.size)

    def __iter__(self):
        return iter(self.indices.tolist())

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            return item in self.names
        return int(item) in self.indices.tolist()

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.array(self.indices, dtype=dtype, copy=copy)

    @property
    def n_selected(self) -> int:
        return int(self.indices.size)

    @property
    def mask(self) -> np.ndarray:
        """Boolean mask over the full candidate set."""
        m = np.zeros(len(self.parameter_names) or self.scores.size, dtype=bool)
        m[self.indices] = True
        return m

    @property
    def rejected_names(self) -> list[str]:
        return [self.parameter_names[j] for j in self.rejected.tolist()]

    def subset(self, parameters: Any) -> ParameterSet:
        """Return the selected parameters as a :class:`ParameterSet`."""
        pset = as_parameters(parameters)
        return ParameterSet([pset[int(j)] for j in self.indices])

    def summary(self) -> str:
        lines = [
            f"select_parameters({self.method}) kept {self.n_selected} of "
            f"{len(self.parameter_names)} parameters",
            f"  condition number {self.condition_full:.4g} -> {self.condition_number:.4g}",
        ]
        sel = {int(j): i for i, j in enumerate(self.indices.tolist())}
        for j, name in enumerate(self.parameter_names):
            if j in sel and self.subset_independence.size:
                indep = f"independence={self.subset_independence[sel[j]]:.4f}"
            else:
                indep = "independence=      -"
            flag = "keep" if j in sel else "drop"
            lines.append(f"  {name:<14s} {flag}  score={self.scores[j]:.4g}  {indep}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ParameterSelection(method={self.method!r}, names={self.names!r}, "
            f"cond={self.condition_number:.3g})"
        )


# ----------------------------------------------------------------------
def _weighted(S: np.ndarray, weights: Any) -> np.ndarray:
    """Apply response weights as ``W^{1/2} S`` so that ``S^T S = S^T W S``."""
    if weights is None:
        return S
    w = np.asarray(weights, dtype=float)
    if w.ndim == 0:
        return math.sqrt(float(w)) * S
    if w.ndim == 1:
        if w.size != S.shape[0]:
            raise ValueError(f"weights length {w.size} != {S.shape[0]} responses")
        return np.sqrt(np.maximum(w, 0.0))[:, None] * S
    if w.shape != (S.shape[0], S.shape[0]):
        raise ValueError(f"weight matrix must be {(S.shape[0], S.shape[0])}, got {w.shape}")
    vals, vecs = np.linalg.eigh(0.5 * (w + w.T))
    root = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
    return root @ S


def parameter_correlation(S: ArrayLike) -> np.ndarray:
    """Cross-correlation matrix of the sensitivity **columns**.

    ``|corr| close to 1`` marks a pair of parameters that the chosen responses
    cannot tell apart, regardless of how good the data is.
    """
    A = np.atleast_2d(np.asarray(S, dtype=float))
    norms = np.linalg.norm(A, axis=0)
    safe = np.where(norms > 0, norms, 1.0)
    Q = A / safe[None, :]
    C = Q.T @ Q
    C[norms == 0, :] = 0.0
    C[:, norms == 0] = 0.0
    np.fill_diagonal(C, 1.0)
    return C


def _condition(S: np.ndarray) -> float:
    if S.size == 0 or S.shape[1] == 0:
        return math.nan
    # Condition of the *estimation* problem, i.e. over all n_par directions:
    # a matrix with more columns than rows leaves a null direction that no
    # amount of data resolves, so it is reported as infinite rather than as
    # the (finite) condition number of its row space.
    sv = np.linalg.svd(S, compute_uv=False)
    if S.shape[1] > sv.size or np.any(sv <= 0):
        return math.inf
    return float(sv[0] / sv[-1])


def _independence(S: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """``(1 - R_j^2, diag(A^-1))`` for the columns of ``S``."""
    n = S.shape[1]
    A = S.T @ S
    diag = np.diag(A).copy()
    scale = float(np.mean(diag[diag > 0])) if np.any(diag > 0) else 1.0
    Areg = A + ridge * scale * np.eye(n)
    try:
        Ainv = np.linalg.inv(Areg)
    except np.linalg.LinAlgError:  # pragma: no cover
        Ainv = np.linalg.pinv(Areg)
    inv_diag = np.diag(Ainv).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        indep = np.where(
            (diag > 0) & (inv_diag > 0), 1.0 / np.maximum(diag * inv_diag, 1e-300), 0.0
        )
    return np.clip(indep, 0.0, 1.0), inv_diag


def _backward_efi(S: np.ndarray, ridge: float) -> list[dict[str, Any]]:
    """Backward D-optimal elimination, all the way down to one parameter.

    Returns one history record per subset size (largest first), each holding
    the retained columns, their condition number, the smallest independence in
    the subset and the column dropped next.
    """
    keep = list(range(S.shape[1]))
    history: list[dict[str, Any]] = []
    while True:
        sub = S[:, keep]
        indep, inv_diag = _independence(sub, ridge)
        record: dict[str, Any] = {
            "n": len(keep),
            "kept": list(keep),
            "condition": _condition(sub),
            "min_independence": float(np.min(indep)) if indep.size else 1.0,
            "dropped": None,
        }
        if len(keep) <= 1:
            history.append(record)
            return history
        worst = int(np.argmax(inv_diag))
        record["dropped"] = keep[worst]
        history.append(record)
        keep.pop(worst)


def _forward_dopt(S: np.ndarray, n_select: int, ridge: float) -> list[int]:
    """Greedy forward selection maximising ``det(S_sub^T S_sub)``."""
    n = S.shape[1]
    chosen: list[int] = []
    remaining = set(range(n))
    while len(chosen) < n_select and remaining:
        best_j, best_v = None, -math.inf
        for j in sorted(remaining):
            trial = S[:, [*chosen, j]]
            A = trial.T @ trial
            A = A + ridge * max(float(np.mean(np.diag(A))), 1e-300) * np.eye(A.shape[0])
            sign, logdet = np.linalg.slogdet(A)
            v = logdet if sign > 0 else -math.inf
            if v > best_v:
                best_j, best_v = j, v
        if best_j is None:  # pragma: no cover
            break
        chosen.append(best_j)
        remaining.discard(best_j)
    return chosen


def _rank_scores(order: Sequence[int], n_par: int) -> np.ndarray:
    """Map a most-valuable-first ordering to scores decreasing from 1 to 0."""
    scores = np.zeros(n_par)
    if n_par == 1:
        return np.ones(1)
    for rank, j in enumerate(order):
        scores[int(j)] = 1.0 - rank / (n_par - 1)
    return scores


def _qr_pivot(S: np.ndarray, n_select: int) -> list[int]:
    """Rank-revealing pivoted QR column subset selection."""
    from scipy.linalg import qr

    _, _, piv = qr(S, mode="economic", pivoting=True)
    return [int(j) for j in piv[:n_select]]


def _correlation_prune(
    S: np.ndarray, max_correlation: float, n_select: int | None
) -> list[int]:
    """Drop the weaker member of every pair above ``max_correlation``."""
    C = np.abs(parameter_correlation(S))
    strength = np.linalg.norm(S, axis=0)
    keep = list(np.argsort(-strength))
    out: list[int] = []
    for j in keep:
        if any(C[j, k] > max_correlation for k in out):
            continue
        out.append(int(j))
        if n_select is not None and len(out) >= n_select:
            break
    return sorted(out)


def select_parameters(
    sensitivity: Any,
    p0: Any = None,
    *,
    parameters: Any = None,
    n_select: int | None = None,
    method: str = "efi",
    normalize: str | None = "relative",
    weights: Any = None,
    max_condition: float | None = None,
    min_independence: float | None = None,
    max_correlation: float = 0.95,
    ridge: float = 1.0e-12,
    names: Sequence[str] | None = None,
    **sensitivity_kwargs: Any,
) -> ParameterSelection:
    """Choose which parameters to update, from their sensitivity matrix.

    Parameters
    ----------
    sensitivity:
        A ``(n_response, n_parameter)`` matrix, a
        :class:`femtools.updating.sensitivity.SensitivityResult`, or anything
        :func:`femtools.updating.sensitivity_matrix` accepts (a model, a
        reference model, or a response callable) — in the last case ``p0`` and
        ``parameters`` are forwarded to compute it.
    n_select:
        How many parameters to keep.  When omitted the size is decided by
        ``max_condition`` and/or ``min_independence``; if neither is given,
        every parameter with an independence above 1 % survives.
    method:
        ``"efi"`` (default, backward D-optimal elimination — see the module
        docstring), ``"qr"``, ``"forward"``, ``"sensitivity"`` or
        ``"correlation"``.
    normalize:
        ``"relative"`` (default, ``dln r/dln p``), ``"parameter"``,
        ``"response"``, ``"column"`` (unit column norms — a pure independence
        ranking that ignores magnitude) or ``None``.  Only ``"relative"``,
        ``"parameter"`` and ``"response"`` need a
        :class:`SensitivityResult`; a bare matrix is left as it is.
    weights:
        Response weights (scalar, vector or matrix), applied as ``W^{1/2} S``
        so the criterion sees the same information matrix the updater will.
    max_condition:
        Keep eliminating until the retained columns have a condition number
        below this value.
    min_independence:
        Reject parameters whose ``1 - R^2`` falls below this value.
    max_correlation:
        Pair correlation limit for ``method="correlation"``.

    Returns
    -------
    ParameterSelection

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.updating.selection import select_parameters
    >>> S = np.array([[1.0, 0.5, 0.0],       # p2 is a weaker copy of p1
    ...               [2.0, 1.0, 1.0],
    ...               [3.0, 1.5 + 1e-6, 2.0]])
    >>> sel = select_parameters(S, n_select=2, normalize=None)
    >>> sel.names
    ['p1', 'p3']
    >>> bool(sel.condition_number < sel.condition_full)
    True
    >>> float(round(sel.independence[1], 9))     # p2 carries no new information
    0.0
    """
    # ---- resolve the sensitivity matrix --------------------------------
    res: SensitivityResult | None = None
    if isinstance(sensitivity, SensitivityResult):
        res = sensitivity
    elif isinstance(sensitivity, np.ndarray) or (
        isinstance(sensitivity, (list, tuple)) and not callable(sensitivity)
    ):
        pass
    else:
        res = sensitivity_matrix(
            sensitivity, p0, parameters=parameters, **sensitivity_kwargs
        )

    norm_key = None if normalize is None else str(normalize).strip().lower()
    if res is not None:
        if norm_key in ("relative", "parameter", "response", "log", "loglog", "p", "r"):
            S = np.asarray(res.normalized(norm_key), dtype=float)
        else:
            S = np.asarray(res.matrix, dtype=float)
        par_names = list(res.parameter_names)
    else:
        S = np.atleast_2d(np.asarray(sensitivity, dtype=float))
        par_names = [f"p{j + 1}" for j in range(S.shape[1])]
    if names is not None:
        par_names = [str(n) for n in names]
    elif parameters is not None:
        try:
            par_names = list(as_parameters(parameters).names)
        except (TypeError, ValueError):  # pragma: no cover - naming is cosmetic
            pass
    if len(par_names) != S.shape[1]:
        raise ValueError(
            f"{len(par_names)} parameter names for {S.shape[1]} sensitivity columns"
        )

    S = _weighted(S, weights)
    if norm_key in ("column", "unit", "columns"):
        norms = np.linalg.norm(S, axis=0)
        S = S / np.where(norms > 0, norms, 1.0)[None, :]

    n_par = S.shape[1]
    if n_par == 0:
        raise ValueError("the sensitivity matrix has no columns")
    cond_full = _condition(S)
    corr = parameter_correlation(S)
    indep_full, _ = _independence(S, ridge)
    col_norm = np.linalg.norm(S, axis=0)

    # ---- how many to keep ----------------------------------------------
    target = n_par if n_select is None else int(np.clip(int(n_select), 1, n_par))

    key = str(method).strip().lower().replace("_", "-")
    if key in ("efi", "efs", "effective-independence", "effective-sensitivity", "d-optimal"):
        key = "efi"
        history = _backward_efi(S, ridge)
        if n_select is not None:
            keep = next(h["kept"] for h in history if h["n"] == target)
        else:
            keep = _largest_acceptable(history, max_condition, min_independence)
        dropped = [h["dropped"] for h in reversed(history) if h["dropped"] is not None]
        order = [int(j) for j in history[-1]["kept"]] + [int(j) for j in dropped]
        scores = _rank_scores(order, n_par)
    elif key == "qr":
        order = _qr_pivot(S, n_par)
        keep = sorted(order[:target])
        history = []
        scores = _rank_scores(order, n_par)
    elif key in ("forward", "greedy"):
        order = _forward_dopt(S, n_par, ridge)
        keep = sorted(order[:target])
        history = []
        scores = _rank_scores(order, n_par)
    elif key in ("sensitivity", "magnitude", "norm"):
        order = [int(j) for j in np.argsort(-col_norm)]
        keep = sorted(int(j) for j in order[:target])
        history = []
        scores = col_norm / max(float(np.max(col_norm)), 1e-300)
    elif key in ("correlation", "collinearity"):
        keep = _correlation_prune(S, max_correlation, n_select)
        history = []
        scores = col_norm / max(float(np.max(col_norm)), 1e-300)
        order = [int(j) for j in np.argsort(-col_norm)]
    else:
        raise ValueError(
            f"unknown selection method {method!r}; expected one of "
            "'efi', 'qr', 'forward', 'sensitivity', 'correlation'"
        )

    if min_independence is not None and key != "efi":
        keep = [j for j in keep if indep_full[j] >= min_independence] or keep[:1]

    keep_arr = np.asarray(sorted(int(j) for j in keep), dtype=int)
    rejected = np.asarray([j for j in range(n_par) if j not in set(keep_arr.tolist())], dtype=int)
    subset_indep, _ = _independence(S[:, keep_arr], ridge)

    return ParameterSelection(
        indices=keep_arr,
        names=[par_names[j] for j in keep_arr.tolist()],
        scores=scores,
        independence=indep_full,
        subset_independence=subset_indep,
        condition_number=_condition(S[:, keep_arr]),
        condition_full=cond_full,
        correlation=corr,
        rejected=rejected,
        order=[int(j) for j in order],
        method=key,
        normalize=norm_key,
        history=history,
        parameter_names=par_names,
    )


def _largest_acceptable(
    history: list[dict[str, Any]],
    max_condition: float | None,
    min_independence: float | None,
) -> list[int]:
    """Largest subset in the elimination history that meets the thresholds."""
    indep_limit = 0.01 if min_independence is None else float(min_independence)
    for record in history:  # largest first
        ok = record["min_independence"] >= indep_limit
        if max_condition is not None:
            ok = ok and record["condition"] <= float(max_condition)
        if ok:
            return list(record["kept"])
    return list(history[-1]["kept"])
