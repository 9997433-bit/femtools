"""Multi-objective design: Pareto fronts by scalarisation, with an NSGA fallback.

Structural design is almost never single-objective — mass against stiffness,
stiffness against a frequency margin, cost against damping.  There is no single
best design, only the *Pareto set*: those designs that cannot be improved in one
objective without giving up another.

Two families of methods are provided, and the choice between them matters.

**Scalarisation** (:func:`pareto_weighted`) turns the vector problem into a
sequence of ordinary constrained minimisations, one per weight vector, each
solved with the gradient-based :func:`femtools.optimization.size_optimize`.  It
is fast and accurate, and it inherits SLSQP's ability to honour nonlinear
constraints exactly.  Two scalarisations are available:

``"weighted-sum"``
    :math:`\\min \\sum_i w_i \\bar f_i`.  Cheap and smooth, but it can only ever
    return points on the *convex hull* of the front: a concave stretch of the
    true front is invisible to every possible weight vector.  This is not a
    numerical shortcoming, it is a property of supporting hyperplanes.

``"chebyshev"``
    :math:`\\min \\max_i w_i (\\bar f_i - z_i^\\ast)` with an augmentation term,
    solved through the smooth epigraph form (an auxiliary variable ``t`` with
    one inequality per objective) so SLSQP still sees a differentiable problem.
    Every Pareto point — convex or not — is the solution of some weighted
    Chebyshev problem, so this recovers concave fronts that the weighted sum
    cannot reach.

**Population search** (``refine="nsga"``) adds a compact NSGA-II-style pass —
non-dominated sorting, crowding distance, binary tournament, SBX crossover and
polynomial mutation — seeded with the scalarisation results.  It needs no
gradients and no convexity, and is the tool of choice when the objectives are
noisy or discontinuous; it is slower and only approximately converged.

Objectives are normalised on the *payoff table*: the single-objective anchor
runs (the unit weight vectors, which are always solved first) supply the ideal
and nadir points, so mass in kilograms and frequency in hertz enter the
scalarisation on equal terms without the user inventing scale factors.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .size import Constraint, size_optimize

__all__ = [
    "ParetoResult",
    "pareto_weighted",
    "pareto_front",
    "non_dominated_sort",
    "crowding_distance",
    "simplex_lattice",
    "hypervolume",
]


# ----------------------------------------------------------------------
def pareto_front(f: ArrayLike, *, maximize: bool = False) -> np.ndarray:
    """Boolean mask of the non-dominated rows of ``f``.

    A point dominates another when it is no worse in every objective and
    strictly better in at least one.  Duplicated points are all kept.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.optimization.multi import pareto_front
    >>> f = np.array([[1.0, 4.0], [2.0, 2.0], [3.0, 5.0], [4.0, 1.0]])
    >>> pareto_front(f)
    array([ True,  True, False,  True])
    """
    F = np.atleast_2d(np.asarray(f, dtype=float))
    if maximize:
        F = -F
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    order = np.argsort(F[:, 0], kind="stable")
    for i in order:
        if not keep[i]:
            continue
        # anything weakly worse in all objectives and strictly worse in one
        le = np.all(F[i][None, :] <= F, axis=1)
        lt = np.any(F[i][None, :] < F, axis=1)
        dominated = le & lt
        dominated[i] = False
        keep &= ~dominated
    return keep


def non_dominated_sort(f: ArrayLike) -> np.ndarray:
    """Pareto rank of every row (0 = first front), as in NSGA-II."""
    F = np.atleast_2d(np.asarray(f, dtype=float))
    n = F.shape[0]
    rank = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)
    level = 0
    while remaining.any():
        idx = np.nonzero(remaining)[0]
        front = pareto_front(F[idx])
        if not front.any():  # pragma: no cover - only for NaN objectives
            rank[idx] = level
            break
        rank[idx[front]] = level
        remaining[idx[front]] = False
        level += 1
    return rank


def crowding_distance(f: ArrayLike) -> np.ndarray:
    """NSGA-II crowding distance: the objective-space cuboid around each point."""
    F = np.atleast_2d(np.asarray(f, dtype=float))
    n, m = F.shape
    if n <= 2:
        return np.full(n, math.inf)
    d = np.zeros(n)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        spread = F[order[-1], j] - F[order[0], j]
        d[order[0]] = d[order[-1]] = math.inf
        if spread <= 0:
            continue
        d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / spread
    return d


def simplex_lattice(n_obj: int, n_div: int) -> np.ndarray:
    """Das & Dennis weight vectors: all ``w >= 0`` with ``sum(w) == 1``.

    ``n_div`` is the number of subdivisions per axis, giving
    ``C(n_div + n_obj - 1, n_obj - 1)`` vectors.

    Examples
    --------
    >>> from femtools.optimization.multi import simplex_lattice
    >>> simplex_lattice(2, 4).T
    array([[0.  , 0.25, 0.5 , 0.75, 1.  ],
           [1.  , 0.75, 0.5 , 0.25, 0.  ]])
    """
    m, k = int(n_obj), int(n_div)
    if m < 2:
        raise ValueError("n_obj must be at least 2")
    if k < 1:
        raise ValueError("n_div must be at least 1")
    rows = []
    # every way of placing m-1 dividers among k+m-1 slots
    for cuts in combinations(range(k + m - 1), m - 1):
        prev, parts = -1, []
        for c in cuts:
            parts.append(c - prev - 1)
            prev = c
        parts.append(k + m - 2 - prev)
        rows.append(parts)
    return np.asarray(rows, dtype=float) / k


def hypervolume(f: ArrayLike, reference: ArrayLike) -> float:
    """Dominated hypervolume of a minimisation front below ``reference``.

    The single most informative scalar summary of a front: it rewards both
    convergence and spread, and needs no knowledge of the true front.

    Examples
    --------
    >>> from femtools.optimization.multi import hypervolume
    >>> hypervolume([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], [4.0, 4.0])
    6.0
    """
    F = np.atleast_2d(np.asarray(f, dtype=float))
    ref = np.asarray(reference, dtype=float).ravel()
    if F.shape[1] != ref.size:
        raise ValueError(f"{F.shape[1]}-objective front against a {ref.size}-D reference")
    F = F[np.all(F <= ref[None, :], axis=1)]
    if F.size == 0:
        return 0.0
    F = F[pareto_front(F)]
    return float(_hv(F, ref))


def _hv(F: np.ndarray, ref: np.ndarray) -> float:
    """Recursive slicing hypervolume (Zitzler's HSO), exact for small fronts."""
    if F.shape[0] == 0:
        return 0.0
    if ref.size == 1:
        return float(max(0.0, ref[0] - F[:, 0].min()))
    order = np.argsort(F[:, -1], kind="stable")
    S = F[order]
    total = 0.0
    for i in range(S.shape[0]):
        upper = S[i + 1, -1] if i + 1 < S.shape[0] else ref[-1]
        depth = float(upper - S[i, -1])
        if depth <= 0.0:
            continue
        slab = S[: i + 1, :-1]
        slab = slab[pareto_front(slab)] if slab.shape[0] > 1 else slab
        total += depth * _hv(slab, ref[:-1])
    return total


# ----------------------------------------------------------------------
@dataclass
class ParetoResult:
    """Outcome of :func:`pareto_weighted`.

    Attributes
    ----------
    x, f:
        The non-dominated designs and their objective vectors, sorted by the
        first objective.
    weights:
        The weight vector that produced each retained design (``nan`` for
        points contributed by the NSGA refinement).
    all_x, all_f:
        Every candidate found, dominated ones included.
    ideal, nadir:
        Payoff-table corners used for normalisation.
    success, feasible:
        Whether the inner solver converged, and whether the returned design
        satisfies the constraints.  These can disagree on Chebyshev runs:
        SLSQP legitimately reports "positive directional derivative" when it
        stops on the min-max kink, with a perfectly good design in hand, so
        judge such points by :attr:`feasible` and by dominance.
    hypervolume:
        Dominated hypervolume relative to :attr:`nadir`.
    """

    x: np.ndarray
    f: np.ndarray
    weights: np.ndarray
    all_x: np.ndarray
    all_f: np.ndarray
    ideal: np.ndarray
    nadir: np.ndarray
    success: np.ndarray
    feasible: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    n_fev: int = 0
    method: str = "weighted-sum"
    hypervolume: float = math.nan
    names: list[str] = field(default_factory=list)
    objective_names: list[str] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.f.shape[0])

    def __iter__(self):
        return iter(zip(self.x, self.f, strict=True))

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.x[i], self.f[i]

    @property
    def n_objectives(self) -> int:
        return int(self.f.shape[1])

    def normalized(self) -> np.ndarray:
        """Front mapped onto ``[0, 1]`` by the payoff table."""
        span = np.where(self.nadir - self.ideal > 0, self.nadir - self.ideal, 1.0)
        return (self.f - self.ideal[None, :]) / span[None, :]

    def knee(self) -> int:
        """Index of the knee: the point closest to the (normalised) ideal.

        With no stated preference between objectives this is the usual
        compromise choice.
        """
        return int(np.argmin(np.linalg.norm(self.normalized(), axis=1)))

    def best(self, objective: int) -> int:
        """Index of the front point that minimises one objective."""
        return int(np.argmin(self.f[:, int(objective)]))

    def summary(self) -> str:
        obj = self.objective_names or [f"f{i + 1}" for i in range(self.n_objectives)]
        lines = [
            f"pareto_weighted({self.method}): {len(self)} non-dominated of "
            f"{self.all_f.shape[0]} designs, {self.n_fev} objective evaluations",
            "  ideal " + "  ".join(f"{n}={v:.6g}" for n, v in zip(obj, self.ideal, strict=True)),
            "  nadir " + "  ".join(f"{n}={v:.6g}" for n, v in zip(obj, self.nadir, strict=True)),
            f"  hypervolume(nadir) {self.hypervolume:.6g}",
        ]
        knee = self.knee()
        for i, fv in enumerate(self.f):
            tag = " <- knee" if i == knee else ""
            lines.append("  " + "  ".join(f"{v:12.6g}" for v in fv) + tag)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ParetoResult(method={self.method!r}, n_points={len(self)}, n_fev={self.n_fev})"


# ----------------------------------------------------------------------
class _ObjectiveCache:
    """Evaluate the objective vector once per design point.

    Scalarisation calls the objectives from both the merit function and the
    epigraph constraints, and the finite-difference gradient calls them again
    for every variable; without memoisation an FE-backed objective is run
    several times at exactly the same design.
    """

    def __init__(self, funs: list[Callable[[np.ndarray], Any]], vector: bool) -> None:
        self._funs = funs
        self._vector = vector
        self._cache: dict[bytes, np.ndarray] = {}
        self.n_fev = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        key = x.tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if self._vector:
            val = np.atleast_1d(np.asarray(self._funs[0](x), dtype=float)).ravel()
        else:
            val = np.asarray([float(f(x)) for f in self._funs], dtype=float)
        self.n_fev += 1
        if len(self._cache) > 100_000:  # pragma: no cover - long runs only
            self._cache.clear()
        self._cache[key] = val
        return val


def _normalize_objectives(objectives: Any, x0: np.ndarray) -> tuple[_ObjectiveCache, int]:
    if callable(objectives):
        probe = np.atleast_1d(np.asarray(objectives(x0), dtype=float)).ravel()
        if probe.size < 2:
            raise ValueError(
                "pareto_weighted needs at least 2 objectives; pass a list of "
                "callables or one callable returning a vector"
            )
        return _ObjectiveCache([objectives], vector=True), int(probe.size)
    funs = list(objectives)
    if len(funs) < 2:
        raise ValueError("pareto_weighted needs at least 2 objectives")
    return _ObjectiveCache(funs, vector=False), len(funs)


def _weight_vectors(weights: Any, n_obj: int, n_weights: int) -> np.ndarray:
    if weights is not None:
        W = np.atleast_2d(np.asarray(weights, dtype=float))
        if W.shape[1] != n_obj:
            raise ValueError(f"weights have {W.shape[1]} columns for {n_obj} objectives")
        if np.any(W < 0):
            raise ValueError("weights must be non-negative")
        row = W.sum(axis=1)
        return W / np.where(row > 0, row, 1.0)[:, None]
    if n_obj == 2:
        t = np.linspace(0.0, 1.0, max(int(n_weights), 2))
        W = np.column_stack([1.0 - t, t])
    else:
        n_div = 1
        while len(simplex_lattice(n_obj, n_div + 1)) <= max(int(n_weights), n_obj):
            n_div += 1
        W = simplex_lattice(n_obj, n_div)
    # anchors first: they build the payoff table the rest are normalised on
    eye = np.eye(n_obj)
    is_anchor = np.array([bool(np.any(np.all(np.isclose(w, eye), axis=1))) for w in W])
    return np.vstack([W[is_anchor], W[~is_anchor]])


def pareto_weighted(
    objectives: Any,
    x0: ArrayLike,
    *,
    bounds: Any = None,
    constraints: Any = None,
    weights: ArrayLike | None = None,
    n_weights: int = 11,
    scalarization: str = "weighted-sum",
    normalize: bool = True,
    rho: float = 1.0e-3,
    warm_start: bool = True,
    refine: str | None = None,
    pop_size: int = 40,
    n_gen: int = 30,
    seed: int | None = None,
    names: Sequence[str] | None = None,
    objective_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> ParetoResult:
    """Trace a Pareto front by repeated scalarised minimisation.

    Parameters
    ----------
    objectives:
        A sequence of scalar callables ``f_i(x)``, or one callable returning
        the whole objective vector.
    x0:
        Starting design.
    bounds:
        As for :func:`femtools.optimization.size_optimize`; finite bounds are
        required for ``refine="nsga"``.
    constraints:
        Passed straight to the inner solver, so nonlinear constraints are
        honoured at every point of the front.
    weights:
        Explicit ``(n_runs, n_obj)`` weight vectors.  When omitted a uniform
        set of ``n_weights`` (2 objectives) or a Das & Dennis simplex lattice
        (3+ objectives) is used, always including the single-objective
        anchors.
    scalarization:
        ``"weighted-sum"`` (default) or ``"chebyshev"``.  Use the latter when
        the front may be non-convex — the weighted sum provably cannot return
        points on a concave stretch, no matter how the weights are chosen.
    normalize:
        Normalise objectives by the payoff table built from the anchor runs.
        Turn this off only if the objectives are already commensurate.
    rho:
        Augmentation weight for the Chebyshev scalarisation, which rules out
        weakly-efficient solutions.
    warm_start:
        Start each run from the design of the nearest solved weight vector
        (continuation).  Much faster, and it keeps the front connected.
    refine:
        ``"nsga"`` runs an NSGA-II-style population pass seeded with the
        scalarisation results; ``None`` (default) skips it.
    pop_size, n_gen, seed:
        NSGA settings.
    kwargs:
        Forwarded to :func:`size_optimize` (``max_iter``, ``tol``, ...).

    Returns
    -------
    ParetoResult

    Examples
    --------
    A convex two-objective problem — the two objectives pull the single design
    variable in opposite directions:

    >>> import numpy as np
    >>> from femtools.optimization.multi import pareto_weighted
    >>> res = pareto_weighted(
    ...     [lambda x: float(x[0] ** 2), lambda x: float((x[0] - 2.0) ** 2)],
    ...     x0=[0.5], bounds=[(-1.0, 3.0)], n_weights=5,
    ... )
    >>> len(res) >= 4
    True
    >>> bool(np.all(np.diff(res.f[:, 1]) <= 1e-9))   # a proper trade-off curve
    True
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float)).astype(float)
    n_var = x0.size
    cache, n_obj = _normalize_objectives(objectives, x0)
    W = _weight_vectors(weights, n_obj, n_weights)
    kind = str(scalarization).strip().lower().replace("_", "-")
    if kind in ("weighted-sum", "sum", "linear", "ws"):
        kind = "weighted-sum"
    elif kind in ("chebyshev", "tchebycheff", "tchebychev", "minmax", "min-max"):
        kind = "chebyshev"
    else:
        raise ValueError(f"unknown scalarization {scalarization!r}")

    f0 = cache(x0)
    ideal = f0.copy()
    nadir = f0.copy()
    # The payoff table is built from the anchor runs, which come first, and
    # then frozen: if the normalisation kept drifting, every interior weight
    # would be solving a slightly different problem and the front would come
    # out unevenly spaced.
    n_anchor = int(np.sum([bool(np.count_nonzero(w) == 1) for w in W]))
    scale_ref: list[np.ndarray] = [ideal.copy(), nadir.copy()]
    solved: list[tuple[np.ndarray, np.ndarray]] = []  # (weight, design)
    all_x: list[np.ndarray] = [x0.copy()]
    all_f: list[np.ndarray] = [f0.copy()]
    used_w: list[np.ndarray] = [np.full(n_obj, math.nan)]
    success: list[bool] = [True]
    feasible: list[bool] = [True]
    results: list[Any] = []

    def norm(fv: np.ndarray) -> np.ndarray:
        if not normalize:
            return fv
        lo, hi = scale_ref
        span = hi - lo
        span = np.where(span > 0, span, np.where(np.abs(lo) > 0, np.abs(lo), 1.0))
        return (fv - lo) / span

    def merit_value(x: np.ndarray, w: np.ndarray) -> float:
        fn = norm(cache(x))
        if kind == "weighted-sum":
            return float(np.dot(w, fn))
        return float(np.max(w[w > 0] * fn[w > 0])) + rho * float(np.sum(fn))

    def one_run(w: np.ndarray, start: np.ndarray) -> tuple[Any, np.ndarray]:
        if kind == "weighted-sum":

            def merit(x: np.ndarray, _w: np.ndarray = w) -> float:
                return float(np.dot(_w, norm(cache(x))))

            res = size_optimize(
                merit,
                start,
                bounds=bounds,
                constraints=constraints,
                names=list(names) if names is not None else None,
                **kwargs,
            )
            return res, np.asarray(res.x, dtype=float)
        res = _chebyshev_run(cache, norm, w, start, n_var, bounds, constraints, rho, kwargs)
        return res, np.asarray(res.x, dtype=float)[:n_var]

    for run, w in enumerate(W):
        start = x0
        # The anchors always start from x0: warm-starting one anchor from
        # another lands on the previous objective's optimum, which is often a
        # stationary point of the objective now being minimised, and the run
        # stops there having moved nothing.
        if warm_start and solved and run >= n_anchor:
            j = int(np.argmin([float(np.sum((w - wv) ** 2)) for wv, _ in solved]))
            start = solved[j][1]
        res, x_opt = one_run(w, start)
        # A warm start that is already a stationary point of the scalarised
        # merit stalls a first-order solver on the spot -- the classic case is
        # an anchor design handed to a neighbouring weight.  Detect the
        # no-progress run and pay for one restart from x0.
        if not np.array_equal(start, x0) and merit_value(x_opt, w) >= merit_value(start, w):
            res_b, x_b = one_run(w, x0)
            if merit_value(x_b, w) < merit_value(x_opt, w):
                res, x_opt = res_b, x_b
        fv = cache(x_opt)
        ideal = np.minimum(ideal, fv)
        nadir = np.maximum(nadir, fv)
        if run < n_anchor:
            scale_ref = [ideal.copy(), nadir.copy()]
        solved.append((w, x_opt))
        all_x.append(x_opt)
        all_f.append(fv)
        used_w.append(w)
        success.append(bool(res.success))
        feasible.append(bool(res.constraint_violation <= 1.0e-6))
        results.append(res)

    if refine is not None:
        key = str(refine).strip().lower()
        if key not in ("nsga", "nsga2", "nsga-ii", "genetic", "ga"):
            raise ValueError(f"unknown refinement {refine!r}; expected 'nsga' or None")
        seeds = np.asarray(all_x, dtype=float)
        gx, gf = _nsga_lite(
            cache,
            bounds,
            seeds,
            n_var,
            constraints=constraints,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
        )
        for xv, fv in zip(gx, gf, strict=True):
            all_x.append(xv)
            all_f.append(fv)
            used_w.append(np.full(n_obj, math.nan))
            success.append(True)
            feasible.append(True)
        if gf.size:
            ideal = np.minimum(ideal, gf.min(axis=0))
            nadir = np.maximum(nadir, gf.max(axis=0))

    AX = np.asarray(all_x, dtype=float)
    AF = np.asarray(all_f, dtype=float)
    AW = np.asarray(used_w, dtype=float)
    mask = pareto_front(AF)
    idx = np.nonzero(mask)[0]
    idx = idx[np.argsort(AF[idx, 0], kind="stable")]

    front = AF[idx]
    ref = nadir
    hv = hypervolume(front, ref) if np.all(nadir > ideal) else math.nan

    return ParetoResult(
        x=AX[idx],
        f=front,
        weights=AW[idx],
        all_x=AX,
        all_f=AF,
        ideal=ideal,
        nadir=nadir,
        success=np.asarray([success[i] for i in idx], dtype=bool),
        feasible=np.asarray([feasible[i] for i in idx], dtype=bool),
        n_fev=cache.n_fev,
        method=kind if refine is None else f"{kind}+nsga",
        hypervolume=hv,
        names=[str(s) for s in names] if names is not None else [],
        objective_names=(
            [str(s) for s in objective_names]
            if objective_names is not None
            else [f"f{i + 1}" for i in range(n_obj)]
        ),
        results=results,
    )


def _chebyshev_run(
    cache: _ObjectiveCache,
    norm: Callable[[np.ndarray], np.ndarray],
    w: np.ndarray,
    start: np.ndarray,
    n_var: int,
    bounds: Any,
    constraints: Any,
    rho: float,
    kwargs: dict[str, Any],
) -> Any:
    """Weighted Chebyshev step in smooth epigraph form.

    ``min_x max_i w_i (f_i - z_i)`` is non-differentiable exactly where it
    matters (at the maximiser), so it is solved as ``min_{x,t} t`` subject to
    ``t >= w_i (f_i(x) - z_i)``, which SLSQP handles with ordinary gradients.
    """
    # An objective with zero weight contributes the constraint ``t >= 0``,
    # which is not a statement about the design at all: it merely pins the
    # epigraph variable and stops the anchor runs from reaching their true
    # single-objective optimum.  Only the active objectives get a row.
    active = np.nonzero(w > 0.0)[0]
    if active.size == 0:  # pragma: no cover - guarded by _weight_vectors
        raise ValueError("a weight vector must have at least one positive entry")
    w_act = w[active]

    fw = w_act * norm(np.asarray(cache(start)))[active]
    t0 = float(np.max(fw))
    z0 = np.concatenate([start, [t0]])

    def obj(z: np.ndarray) -> float:
        # Augmented Chebyshev: the sum term rules out weakly-efficient points
        # (and, with a zero weight, gives the anchor run something to descend).
        return float(z[-1]) + rho * float(np.sum(norm(cache(z[:n_var]))))

    def epigraph(z: np.ndarray) -> np.ndarray:
        return z[-1] - w_act * norm(cache(z[:n_var]))[active]

    def on_design(fn: Callable[[np.ndarray], Any]) -> Callable[[np.ndarray], Any]:
        """Lift a constraint on ``x`` to the augmented variable ``[x, t]``."""

        def wrapped(z: np.ndarray) -> Any:
            return np.asarray(fn(z[:n_var]))

        return wrapped

    cons: list[Any] = [Constraint(fun=epigraph, type="ineq", name="chebyshev")]
    if constraints is not None:
        extra = constraints if isinstance(constraints, (list, tuple)) else [constraints]
        for c in extra:
            if isinstance(c, Constraint):
                cons.append(Constraint(fun=on_design(c.fun), type=c.type, name=c.name))
            elif callable(c):
                cons.append(Constraint(fun=on_design(c)))
            else:
                raise TypeError(f"cannot interpret constraint {c!r} for chebyshev")

    zb = None
    if bounds is not None:
        lo, hi = _expand_bounds(bounds, n_var)
        zb = [*zip(lo, hi, strict=True), (None, None)]
    opts = dict(kwargs)
    opts.setdefault("objective_scaling", None)
    res = size_optimize(obj, z0, bounds=zb, constraints=cons, **opts)

    # ``t`` is bookkeeping, not design: snap it onto the exact epigraph value
    # and report the violation of the *design* constraints only, so a solver
    # that stops a hair short of the min-max kink is not mistaken for one that
    # returned an infeasible structure.
    x_opt = np.asarray(res.x, dtype=float)[:n_var]
    res.x = np.concatenate([x_opt, [float(np.max(w_act * norm(cache(x_opt))[active]))]])
    res.constraint_violation = max((c.violation(x_opt) for c in cons[1:]), default=0.0)
    return res


def _expand_bounds(bounds: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    arr = list(bounds)
    if len(arr) == 2 and all(np.isscalar(b) or b is None for b in arr):
        lo = -np.inf if arr[0] is None else float(arr[0])
        hi = np.inf if arr[1] is None else float(arr[1])
        return np.full(n, lo), np.full(n, hi)
    if len(arr) != n:
        raise ValueError(f"bounds length {len(arr)} != {n} variables")
    lo_v = np.array([-np.inf if b is None or b[0] is None else float(b[0]) for b in arr])
    hi_v = np.array([np.inf if b is None or b[1] is None else float(b[1]) for b in arr])
    return lo_v, hi_v


# ----------------------------------------------------------------------
def _violation_function(constraints: Any) -> Callable[[np.ndarray], float] | None:
    """Total constraint violation of a design, or ``None`` when unconstrained."""
    if constraints is None:
        return None
    items = constraints if isinstance(constraints, (list, tuple)) else [constraints]
    cons = [c if isinstance(c, Constraint) else Constraint(fun=c) for c in items]
    if not cons:
        return None

    def violation(x: np.ndarray) -> float:
        return float(sum(c.violation(x) for c in cons))

    return violation


def _constrained_rank(F: np.ndarray, viol: np.ndarray) -> np.ndarray:
    """Deb's constrained non-dominated sort.

    A feasible design always beats an infeasible one, and among infeasible
    designs the less violating one wins — so the population is pulled into the
    feasible region before it starts trading objectives off against each other.
    """
    rank = np.empty(F.shape[0], dtype=int)
    ok = viol <= 0.0
    if np.any(ok):
        rank[ok] = non_dominated_sort(F[ok])
        offset = int(rank[ok].max()) + 1
    else:
        offset = 0
    if np.any(~ok):
        order = np.argsort(viol[~ok], kind="stable")
        ranks = np.empty(order.size, dtype=int)
        ranks[order] = np.arange(order.size)
        rank[~ok] = offset + ranks
    return rank


def _nsga_lite(
    fun: Callable[[np.ndarray], np.ndarray],
    bounds: Any,
    seeds: np.ndarray,
    n_var: int,
    *,
    constraints: Any = None,
    pop_size: int = 40,
    n_gen: int = 30,
    seed: int | None = None,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compact NSGA-II: SBX + polynomial mutation, elitist (mu+lambda) survival."""
    if bounds is None:
        raise ValueError("refine='nsga' requires finite bounds")
    lo, hi = _expand_bounds(bounds, n_var)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("refine='nsga' requires finite bounds on every variable")
    rng = np.random.default_rng(seed)
    vfun = _violation_function(constraints)

    def violations(P: np.ndarray) -> np.ndarray:
        if vfun is None:
            return np.zeros(P.shape[0])
        return np.asarray([vfun(x) for x in P], dtype=float)

    n_pop = max(int(pop_size), 4)
    pop = np.empty((n_pop, n_var))
    n_seed = min(seeds.shape[0], n_pop)
    pop[:n_seed] = np.clip(seeds[:n_seed], lo, hi)
    pop[n_seed:] = lo + rng.random((n_pop - n_seed, n_var)) * (hi - lo)
    F = np.asarray([fun(x) for x in pop], dtype=float)
    V = violations(pop)

    def tournament(rank: np.ndarray, crowd: np.ndarray, k: int) -> np.ndarray:
        a = rng.integers(0, rank.size, size=k)
        b = rng.integers(0, rank.size, size=k)
        better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
        return np.where(better, a, b)

    for _ in range(max(int(n_gen), 0)):
        rank = _constrained_rank(F, V)
        crowd = np.zeros(rank.size)
        for r in np.unique(rank):
            m = rank == r
            crowd[m] = crowding_distance(F[m])
        p1 = tournament(rank, crowd, n_pop)
        p2 = tournament(rank, crowd, n_pop)

        # simulated binary crossover
        u = rng.random((n_pop, n_var))
        beta = np.where(
            u <= 0.5, (2.0 * u) ** (1.0 / (eta_c + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)),
        )
        a, b = pop[p1], pop[p2]
        child = np.where(
            rng.random((n_pop, n_var)) < 0.5,
            0.5 * ((1.0 + beta) * a + (1.0 - beta) * b),
            0.5 * ((1.0 - beta) * a + (1.0 + beta) * b),
        )

        # polynomial mutation
        span = hi - lo
        mut = rng.random((n_pop, n_var)) < (1.0 / n_var)
        r = rng.random((n_pop, n_var))
        delta = np.where(
            r < 0.5,
            (2.0 * r) ** (1.0 / (eta_m + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - r)) ** (1.0 / (eta_m + 1.0)),
        )
        child = np.where(mut, child + delta * span[None, :], child)
        child = np.clip(child, lo, hi)

        Fc = np.asarray([fun(x) for x in child], dtype=float)
        pool = np.vstack([pop, child])
        Fpool = np.vstack([F, Fc])
        Vpool = np.concatenate([V, violations(child)])

        rank = _constrained_rank(Fpool, Vpool)
        keep: list[int] = []
        for r_level in np.unique(rank):
            idx = np.nonzero(rank == r_level)[0]
            if len(keep) + idx.size <= n_pop:
                keep.extend(idx.tolist())
                continue
            cd = crowding_distance(Fpool[idx])
            room = n_pop - len(keep)
            keep.extend(idx[np.argsort(-cd)[:room]].tolist())
            break
        sel = np.asarray(keep, dtype=int)
        pop, F, V = pool[sel], Fpool[sel], Vpool[sel]

    ok = V <= 0.0
    if not np.any(ok):
        return np.zeros((0, n_var)), np.zeros((0, F.shape[1]))
    pop, F = pop[ok], F[ok]
    mask = pareto_front(F)
    return pop[mask], F[mask]
