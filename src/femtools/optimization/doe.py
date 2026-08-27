"""Design of experiments: space-filling and factorial sampling plans.

All generators return a plain ``(n_samples, n_dim)`` float array in the order
of the supplied ``bounds`` (or in the unit hypercube when no bounds are given),
and are fully deterministic for a fixed ``seed``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "latin_hypercube",
    "full_factorial",
    "random_sampling",
    "sobol",
    "central_composite",
    "scale_to_bounds",
    "maximin_distance",
    "discrepancy",
]


def _resolve_bounds(
    bounds: Any, n_dim: int | None
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    if bounds is None:
        if n_dim is None:
            raise ValueError("either `bounds` or `n_dim` must be given")
        return None, None, int(n_dim)
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("a 1-D `bounds` must be a single (lo, hi) pair")
        if n_dim is None:
            raise ValueError("`n_dim` is required when `bounds` is a single (lo, hi) pair")
        lo = np.full(int(n_dim), arr[0])
        hi = np.full(int(n_dim), arr[1])
        return lo, hi, int(n_dim)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"`bounds` must have shape (n_dim, 2), got {arr.shape}")
    if n_dim is not None and int(n_dim) != arr.shape[0]:
        raise ValueError(f"`n_dim`={n_dim} conflicts with bounds of shape {arr.shape}")
    if np.any(arr[:, 1] < arr[:, 0]):
        raise ValueError("`bounds` has an upper limit below its lower limit")
    return arr[:, 0].copy(), arr[:, 1].copy(), arr.shape[0]


def scale_to_bounds(unit: np.ndarray, bounds: Any) -> np.ndarray:
    """Map samples from the unit hypercube onto ``bounds``."""
    unit = np.asarray(unit, dtype=float)
    lo, hi, _ = _resolve_bounds(bounds, unit.shape[1])
    if lo is None:
        return unit
    return lo[None, :] + unit * (hi - lo)[None, :]


def maximin_distance(samples: np.ndarray) -> float:
    """Smallest pairwise Euclidean distance (larger = better space filling)."""
    x = np.asarray(samples, dtype=float)
    if x.shape[0] < 2:
        return math.inf
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


def discrepancy(samples: np.ndarray) -> float:
    """Centred L2 discrepancy (lower = more uniform); unit-hypercube inputs."""
    x = np.asarray(samples, dtype=float)
    n, d = x.shape
    t1 = (13.0 / 12.0) ** d
    a = 1.0 + 0.5 * np.abs(x - 0.5) - 0.5 * (x - 0.5) ** 2
    t2 = float(np.sum(np.prod(a, axis=1))) * 2.0 / n
    b = (
        1.0
        + 0.5 * np.abs(x[:, None, :] - 0.5)
        + 0.5 * np.abs(x[None, :, :] - 0.5)
        - 0.5 * np.abs(x[:, None, :] - x[None, :, :])
    )
    t3 = float(np.sum(np.prod(b, axis=2))) / n**2
    return float(t1 - t2 + t3)


# ----------------------------------------------------------------------
def _split_count_and_bounds(
    args: tuple[Any, ...], n_samples: Any, bounds: Any
) -> tuple[Any, Any]:
    """Resolve the two accepted positional orders into ``(n_samples, bounds)``."""
    for arg in args:
        is_count = isinstance(arg, (int, np.integer)) and not isinstance(arg, bool)
        if is_count:
            if n_samples is not None:
                raise TypeError("n_samples given twice")
            n_samples = arg
        else:
            if bounds is not None:
                raise TypeError("bounds given twice")
            bounds = arg
    return n_samples, bounds


def latin_hypercube(
    *args: Any,
    n_samples: int | None = None,
    bounds: Any = None,
    n_dim: int | None = None,
    seed: int | np.random.Generator | None = None,
    criterion: str = "maximin",
    iterations: int = 25,
    centered: bool = False,
) -> np.ndarray:
    """Latin hypercube sample of ``n_samples`` points.

    Every one-dimensional projection is stratified into ``n_samples`` equal
    bins with exactly one point in each.

    Both positional orders are accepted, since either reads naturally::

        latin_hypercube(24, bounds)
        latin_hypercube(bounds, n_samples=24)

    Parameters
    ----------
    n_samples:
        Number of design points.
    bounds:
        ``(n_dim, 2)`` array of ``(lo, hi)`` limits, a single ``(lo, hi)`` pair
        combined with ``n_dim``, or ``None`` for the unit hypercube.
    n_dim:
        Dimensionality (required when ``bounds`` does not imply it).
    seed:
        Seed or :class:`numpy.random.Generator` — results are reproducible.
    criterion:
        ``"maximin"`` (default) retries ``iterations`` plans and keeps the one
        with the largest minimum pairwise distance; ``"center"`` places points
        at bin centres; ``"random"``/``"none"`` does a single random plan;
        ``"correlation"`` minimises the maximum absolute column correlation.
    centered:
        Force bin-centre placement regardless of ``criterion``.

    Returns
    -------
    ndarray
        ``(n_samples, n_dim)`` samples.

    Examples
    --------
    >>> from femtools.optimization.doe import latin_hypercube
    >>> x = latin_hypercube(10, [(0.0, 1.0), (100.0, 200.0)], seed=0)
    >>> x.shape
    (10, 2)
    """
    if len(args) > 2:
        raise TypeError(f"latin_hypercube takes at most 2 positional arguments, got {len(args)}")
    n_samples, bounds = _split_count_and_bounds(args, n_samples, bounds)
    if n_samples is None:
        raise TypeError("n_samples is required")
    n = int(n_samples)
    if n <= 0:
        raise ValueError("n_samples must be positive")
    lo, hi, d = _resolve_bounds(bounds, n_dim)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    crit = str(criterion).lower()
    use_center = centered or crit in ("center", "centre", "centered", "c")

    def _one_plan() -> np.ndarray:
        out = np.empty((n, d))
        for j in range(d):
            bins = rng.permutation(n).astype(float)
            offset = np.full(n, 0.5) if use_center else rng.random(n)
            out[:, j] = (bins + offset) / n
        return out

    if crit in ("maximin", "m") and not use_center:
        best, best_score = None, -math.inf
        for _ in range(max(1, int(iterations))):
            cand = _one_plan()
            score = maximin_distance(cand)
            if score > best_score:
                best, best_score = cand, score
        unit = best
    elif crit in ("correlation", "corr"):
        best, best_score = None, math.inf
        for _ in range(max(1, int(iterations))):
            cand = _one_plan()
            if d > 1:
                cm = np.corrcoef(cand, rowvar=False)
                score = float(np.max(np.abs(cm - np.eye(d))))
            else:
                score = 0.0
            if score < best_score:
                best, best_score = cand, score
        unit = best
    else:
        unit = _one_plan()

    assert unit is not None
    if lo is None:
        return unit
    return lo[None, :] + unit * (hi - lo)[None, :]


def _explicit_axes(levels: Any) -> list[np.ndarray] | None:
    """Return per-dimension level values when ``levels`` lists them explicitly."""
    if isinstance(levels, (str, bytes)) or np.isscalar(levels):
        return None
    try:
        entries = list(levels)
    except TypeError:
        return None
    if not entries or not all(np.ndim(entry) == 1 for entry in entries):
        return None
    return [np.asarray(entry, dtype=float).ravel() for entry in entries]


def full_factorial(
    levels: int | Sequence[int] | Sequence[Sequence[float]],
    bounds: Any = None,
    *,
    n_dim: int | None = None,
    order: str = "c",
) -> np.ndarray:
    """Full factorial grid.

    Parameters
    ----------
    levels:
        Either level *counts* — a single integer (combined with ``n_dim`` or
        with the dimensionality implied by ``bounds``) or one count per
        dimension — or the level *values* themselves, given as one sequence per
        dimension (in which case ``bounds`` is not used).
    bounds:
        Limits per dimension; ``None`` gives levels spread over ``[0, 1]``.
    order:
        ``"c"`` (default) varies the last factor fastest, ``"f"`` the first.

    Returns
    -------
    ndarray
        ``(prod(levels), n_dim)`` grid points.

    Examples
    --------
    >>> from femtools.optimization.doe import full_factorial
    >>> full_factorial(2, [(0.0, 1.0), (10.0, 20.0)])
    array([[ 0., 10.],
           [ 0., 20.],
           [ 1., 10.],
           [ 1., 20.]])
    >>> full_factorial([[-1.0, 1.0], [5.0]])
    array([[-1.,  5.],
           [ 1.,  5.]])
    """
    axes = _explicit_axes(levels)
    if axes is not None:
        if bounds is not None:
            raise ValueError("`bounds` is not accepted when `levels` lists level values")
        if any(axis.size < 1 for axis in axes):
            raise ValueError("every dimension needs at least one level")
        d = len(axes)
    else:
        if np.isscalar(levels):
            lo, hi, d = _resolve_bounds(bounds, n_dim)
            lv = [int(levels)] * d  # type: ignore[arg-type]
        else:
            lv = [int(x) for x in np.asarray(levels).ravel()]
            lo, hi, d = _resolve_bounds(bounds, len(lv))
            if d != len(lv):
                raise ValueError(f"levels has {len(lv)} entries but bounds imply {d} dimensions")
        if any(x < 1 for x in lv):
            raise ValueError("every dimension needs at least one level")

        axes = []
        for j, m in enumerate(lv):
            axis = np.array([0.5]) if m == 1 else np.linspace(0.0, 1.0, m)
            if lo is not None and hi is not None:
                axis = lo[j] + axis * (hi[j] - lo[j])
            axes.append(axis)

    combos = list(itertools.product(*axes))
    grid = np.asarray(combos, dtype=float).reshape(-1, d)
    if str(order).lower() in ("f", "fortran"):
        idx = np.lexsort(tuple(grid[:, j] for j in range(d)))
        grid = grid[idx]
    return grid


def random_sampling(
    n_samples: int,
    bounds: Any = None,
    *,
    n_dim: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Plain uniform Monte-Carlo sampling (baseline for DOE comparisons)."""
    lo, hi, d = _resolve_bounds(bounds, n_dim)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    unit = rng.random((int(n_samples), d))
    return unit if lo is None else lo[None, :] + unit * (hi - lo)[None, :]


def sobol(
    n_samples: int,
    bounds: Any = None,
    *,
    n_dim: int | None = None,
    seed: int | None = None,
    scramble: bool = True,
) -> np.ndarray:
    """Scrambled Sobol' low-discrepancy sequence (via :mod:`scipy.stats.qmc`)."""
    from scipy.stats import qmc

    lo, hi, d = _resolve_bounds(bounds, n_dim)
    eng = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    unit = eng.random(int(n_samples))
    return unit if lo is None else lo[None, :] + unit * (hi - lo)[None, :]


def central_composite(
    bounds: Any = None,
    *,
    n_dim: int | None = None,
    alpha: str | float = "orthogonal",
    center_points: int = 1,
    face: str = "circumscribed",
) -> np.ndarray:
    """Central composite design: factorial corners + axial (star) + centre points.

    ``alpha`` may be a number, ``"orthogonal"``/``"rotatable"`` (``2**(d/4)``) or
    ``"faced"`` (``alpha = 1``, i.e. a face-centred design).
    """
    lo, hi, d = _resolve_bounds(bounds, n_dim)
    corners = np.asarray(list(itertools.product([-1.0, 1.0], repeat=d)), dtype=float)
    if isinstance(alpha, str):
        a = 1.0 if alpha.lower() in ("faced", "face", "ccf") or face.lower() in (
            "faced",
            "face",
        ) else 2.0 ** (d / 4.0)
    else:
        a = float(alpha)
    star = np.zeros((2 * d, d))
    for j in range(d):
        star[2 * j, j] = -a
        star[2 * j + 1, j] = a
    centre = np.zeros((max(int(center_points), 0), d))
    coded = np.vstack([corners, star, centre])
    unit = (coded / (2.0 * a)) + 0.5
    return unit if lo is None else lo[None, :] + unit * (hi - lo)[None, :]
