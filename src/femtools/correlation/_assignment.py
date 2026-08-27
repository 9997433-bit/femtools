"""Rectangular linear sum assignment (Hungarian / Jonker-Volgenant).

``scipy.optimize.linear_sum_assignment`` is used when SciPy is importable;
otherwise the pure-NumPy shortest-augmenting-path solver below is used so the
correlation package keeps working in a minimal environment.  Both return an
assignment of minimum total cost.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["linear_sum_assignment"]


def linear_sum_assignment(
    cost: ArrayLike, maximize: bool = False
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Solve ``min sum(cost[row_ind, col_ind])`` over one-to-one assignments."""
    c = np.asarray(cost, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {c.shape}")
    if not np.isfinite(c).all():
        raise ValueError("cost matrix must be finite")
    try:
        from scipy.optimize import linear_sum_assignment as _scipy_lsa
    except ImportError:
        pass
    else:
        r, col = _scipy_lsa(c, maximize=maximize)
        return np.asarray(r, dtype=np.intp), np.asarray(col, dtype=np.intp)
    return _jv(-c if maximize else c)


def _jv(cost: NDArray[np.float64]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    n, m = cost.shape
    if n == 0 or m == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    transposed = n > m
    if transposed:
        cost = cost.T.copy()
        n, m = m, n

    u = np.zeros(n)
    v = np.zeros(m)
    path = np.full(m, -1, dtype=np.intp)
    col4row = np.full(n, -1, dtype=np.intp)
    row4col = np.full(m, -1, dtype=np.intp)

    for cur_row in range(n):
        sink, min_val, sr, sc, shortest = _augment(cost, u, v, path, row4col, cur_row)
        if sink < 0:  # pragma: no cover - cannot happen for a finite cost matrix
            raise ValueError("cost matrix is infeasible")

        u[cur_row] += min_val
        others = sr.copy()
        others[cur_row] = False
        if others.any():
            u[others] += min_val - shortest[col4row[others]]
        v[sc] -= min_val - shortest[sc]

        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            col4row[i], j = j, col4row[i]
            if i == cur_row:
                break

    rows = np.arange(n, dtype=np.intp)
    cols = col4row.astype(np.intp)
    if transposed:
        order = np.argsort(cols, kind="stable")
        return cols[order], rows[order]
    return rows, cols


def _augment(
    cost: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    path: NDArray[np.intp],
    row4col: NDArray[np.intp],
    start_row: int,
) -> tuple[int, float, NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    n, m = cost.shape
    min_val = 0.0
    i = start_row
    remaining = np.arange(m - 1, -1, -1, dtype=np.intp)
    num_remaining = m
    sr = np.zeros(n, dtype=bool)
    sc = np.zeros(m, dtype=bool)
    shortest = np.full(m, np.inf)
    sink = -1

    while sink == -1:
        sr[i] = True
        rem = remaining[:num_remaining]
        r = min_val + cost[i, rem] - u[i] - v[rem]
        better = r < shortest[rem]
        if better.any():
            improved = rem[better]
            path[improved] = i
            shortest[improved] = r[better]

        sp = shortest[rem]
        lowest = sp.min()
        if not np.isfinite(lowest):
            return -1, lowest, sr, sc, shortest
        tied = np.flatnonzero(sp == lowest)
        free = tied[row4col[rem[tied]] == -1]
        index = int(free[0] if free.size else tied[0])

        min_val = float(lowest)
        j = int(rem[index])
        if row4col[j] == -1:
            sink = j
        else:
            i = int(row4col[j])
        sc[j] = True
        num_remaining -= 1
        remaining[index] = remaining[num_remaining]

    return sink, min_val, sr, sc, shortest
