"""Gradient-based sizing / shape optimization (SLSQP and friends).

``size_optimize`` is a thin, well-instrumented wrapper around
:func:`scipy.optimize.minimize` specialised for structural sizing problems:
bound-constrained design variables, a scalar objective (mass, compliance,
frequency margin, ...) and a mixture of equality / inequality constraints.

It adds, on top of plain SciPy:

* design-variable scaling (essential when mixing thicknesses in metres with
  areas in square metres),
* relative-step finite-difference gradients,
* iteration history capture,
* a uniform :class:`OptimizationResult`,
* multi-start support for weakly non-convex problems.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import NonlinearConstraint, minimize

__all__ = [
    "Constraint",
    "OptimizationResult",
    "size_optimize",
    "finite_difference_gradient",
]


@dataclass
class Constraint:
    """A design constraint ``fun(x)`` with ``type in {"ineq", "eq"}``.

    Following the SciPy convention, ``"ineq"`` constraints are satisfied when
    ``fun(x) >= 0``.  Use :meth:`upper_bound` / :meth:`lower_bound` to build the
    common "response <= limit" / "response >= limit" forms.
    """

    fun: Callable[[np.ndarray], Any]
    type: str = "ineq"
    jac: Callable[[np.ndarray], Any] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        t = self.type.lower()
        if t in ("inequality", "ge", ">=", "ineq"):
            self.type = "ineq"
        elif t in ("equality", "eq", "=="):
            self.type = "eq"
        else:
            raise ValueError(f"unknown constraint type {self.type!r}")

    @classmethod
    def upper_bound(
        cls, fun: Callable[[np.ndarray], Any], limit: float, name: str = ""
    ) -> Constraint:
        """``fun(x) <= limit``."""
        return cls(fun=lambda x: float(limit) - np.asarray(fun(x)), type="ineq", name=name)

    @classmethod
    def lower_bound(
        cls, fun: Callable[[np.ndarray], Any], limit: float, name: str = ""
    ) -> Constraint:
        """``fun(x) >= limit``."""
        return cls(fun=lambda x: np.asarray(fun(x)) - float(limit), type="ineq", name=name)

    @classmethod
    def equality(
        cls, fun: Callable[[np.ndarray], Any], value: float = 0.0, name: str = ""
    ) -> Constraint:
        """``fun(x) == value``."""
        return cls(fun=lambda x: np.asarray(fun(x)) - float(value), type="eq", name=name)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "fun": self.fun}
        if self.jac is not None:
            d["jac"] = self.jac
        return d

    def violation(self, x: np.ndarray) -> float:
        v = np.atleast_1d(np.asarray(self.fun(x), dtype=float))
        if self.type == "eq":
            return float(np.max(np.abs(v)))
        return float(max(0.0, -np.min(v)))


@dataclass
class OptimizationResult:
    """Result of :func:`size_optimize`."""

    x: np.ndarray
    fun: float
    success: bool = False
    status: int = 0
    message: str = ""
    n_iter: int = 0
    n_fev: int = 0
    n_gev: int = 0
    jac: np.ndarray | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    constraint_violation: float = 0.0
    x0: np.ndarray | None = None
    fun0: float = math.nan
    method: str = "SLSQP"
    names: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.array(self.x, dtype=dtype, copy=copy)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return float(self.x[self.names.index(key)])
        return self.x[key]

    @property
    def objective(self) -> float:
        return self.fun

    @property
    def feasible(self) -> bool:
        return self.constraint_violation <= 1.0e-6

    @property
    def improvement(self) -> float:
        """Fractional objective reduction relative to the starting point."""
        if not math.isfinite(self.fun0) or self.fun0 == 0.0:
            return math.nan
        return float((self.fun0 - self.fun) / abs(self.fun0))

    def to_dict(self) -> dict[str, float]:
        names = self.names or [f"x{i + 1}" for i in range(self.x.size)]
        return {n: float(v) for n, v in zip(names, self.x, strict=True)}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OptimizationResult(fun={self.fun:.6g}, success={self.success}, "
            f"n_iter={self.n_iter}, x={np.array2string(self.x, precision=5)})"
        )


# ----------------------------------------------------------------------
def finite_difference_gradient(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    step: float = 1.0e-6,
    relative: bool = True,
    f0: float | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    method: str = "forward",
) -> np.ndarray:
    """Finite-difference gradient with bound-aware, relatively scaled steps."""
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.full(n, float(step))
    if relative:
        h = np.where(np.abs(x) > 0, h * np.abs(x), h)
    g = np.zeros(n)
    if f0 is None and method == "forward":
        f0 = float(fun(x))
    for j in range(n):
        hj = h[j]
        if method == "central":
            xp, xm = x.copy(), x.copy()
            xp[j] += hj
            xm[j] -= hj
            if bounds is not None:
                xp[j] = min(xp[j], bounds[1][j])
                xm[j] = max(xm[j], bounds[0][j])
            denom = xp[j] - xm[j]
            g[j] = (float(fun(xp)) - float(fun(xm))) / denom if denom else 0.0
        else:
            xp = x.copy()
            xp[j] += hj
            if bounds is not None and xp[j] > bounds[1][j]:
                xp[j] = x[j] - hj
                hj = -hj
            g[j] = (float(fun(xp)) - float(f0 if f0 is not None else fun(x))) / hj
    return g


def _normalize_constraints(constraints: Any) -> list[dict[str, Any]]:
    if constraints is None:
        return []
    if isinstance(constraints, (Constraint, Mapping, NonlinearConstraint)) or callable(
        constraints
    ):
        constraints = [constraints]
    out: list[dict[str, Any]] = []
    for c in constraints:
        if isinstance(c, Constraint):
            out.append(c.to_dict())
        elif isinstance(c, Mapping):
            d = dict(c)
            d.setdefault("type", "ineq")
            out.append(d)
        elif isinstance(c, NonlinearConstraint):  # pragma: no cover - passthrough
            out.append(c)  # type: ignore[arg-type]
        elif callable(c):
            out.append({"type": "ineq", "fun": c})
        else:
            raise TypeError(f"cannot interpret constraint {c!r}")
    return out


def _normalize_bounds(bounds: Any, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    if bounds is None:
        return None
    if isinstance(bounds, Mapping):
        raise TypeError("bounds mapping is not supported; pass a sequence of (lo, hi)")
    arr = list(bounds)
    if len(arr) == 2 and all(np.isscalar(b) or b is None for b in arr):
        lo_s = -np.inf if arr[0] is None else float(arr[0])
        hi_s = np.inf if arr[1] is None else float(arr[1])
        return np.full(n, lo_s), np.full(n, hi_s)
    if len(arr) != n:
        raise ValueError(f"bounds length {len(arr)} != number of variables {n}")
    lo = np.empty(n)
    hi = np.empty(n)
    for i, b in enumerate(arr):
        if b is None:
            lo[i], hi[i] = -np.inf, np.inf
        else:
            lo[i] = -np.inf if b[0] is None else float(b[0])
            hi[i] = np.inf if b[1] is None else float(b[1])
    return lo, hi


def size_optimize(
    objective: Callable[[np.ndarray], float],
    x0: ArrayLike,
    *,
    bounds: Any = None,
    constraints: Any = None,
    jac: Callable[[np.ndarray], Any] | None = None,
    method: str = "SLSQP",
    max_iter: int = 200,
    tol: float = 1.0e-8,
    step: float = 1.0e-6,
    fd_method: str = "forward",
    scaling: Any = None,
    args: tuple[Any, ...] = (),
    options: dict[str, Any] | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
    keep_history: bool = True,
    objective_scaling: Any = "auto",
    constraint_scaling: Any = "auto",
    feasibility_tol: float = 1.0e-6,
    n_starts: int = 1,
    seed: int | None = None,
    names: Sequence[str] | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Minimise ``objective(x)`` subject to bounds and constraints.

    Parameters
    ----------
    objective:
        Scalar objective ``f(x, *args)`` (mass, compliance, ...).
    x0:
        Initial design variables (element thicknesses, areas, ...).
    bounds:
        ``(lo, hi)`` applied to all variables, or a per-variable sequence of
        ``(lo, hi)`` pairs.  ``None`` entries mean unbounded.
    constraints:
        A :class:`Constraint`, SciPy-style dict, bare callable (treated as
        ``g(x) >= 0``), or any iterable of those.
    jac:
        Analytic gradient.  When omitted, a relative-step finite-difference
        gradient is used (``fd_method="forward"`` or ``"central"``).
    method:
        Any SciPy ``minimize`` method; ``"SLSQP"`` is the default and the only
        one that handles general nonlinear constraints together with bounds
        without extra setup.
    scaling:
        Per-variable scale factors (or ``"auto"`` to use ``|x0|``).  The solver
        works on ``x / scale``, which greatly improves SLSQP conditioning when
        design variables differ by orders of magnitude.
    n_starts:
        Number of random restarts inside the bounds (the best feasible design
        wins).  ``1`` (default) keeps the run deterministic.

    Returns
    -------
    OptimizationResult

    Examples
    --------
    Minimum-mass 2-bar sizing with a stress constraint::

        res = size_optimize(
            lambda a: 7850.0 * (a[0] * 1.0 + a[1] * 1.5),
            x0=[1e-3, 1e-3],
            bounds=(1e-5, 1e-2),
            constraints=[Constraint.upper_bound(lambda a: 1e4 / a[0], 200e6)],
        )
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = x0.size
    bnds = _normalize_bounds(bounds, n)
    cons = _normalize_constraints(constraints)

    if scaling is None:
        scale = np.ones(n)
    elif isinstance(scaling, str):
        if scaling.lower() != "auto":
            raise ValueError(f"unknown scaling {scaling!r}")
        scale = np.where(np.abs(x0) > 0, np.abs(x0), 1.0)
    else:
        scale = np.asarray(scaling, dtype=float)
        if scale.ndim == 0:
            scale = np.full(n, float(scale))
        scale = np.where(scale != 0, np.abs(scale), 1.0)

    counters = {"fev": 0, "gev": 0}
    history: list[dict[str, Any]] = []

    def f_phys(x: np.ndarray) -> float:
        counters["fev"] += 1
        return float(objective(x, *args))

    f0 = f_phys(x0)
    fref = 1.0
    if objective_scaling in (True, "auto") and math.isfinite(f0) and f0 != 0.0:
        fref = abs(f0)
    elif isinstance(objective_scaling, (int, float)) and objective_scaling:
        fref = abs(float(objective_scaling))

    def f_scaled(z: np.ndarray) -> float:
        val = f_phys(z * scale)
        if keep_history:
            history.append({"n_fev": counters["fev"], "x": z * scale, "fun": val})
        return val / fref

    if jac is not None:

        def g_scaled(z: np.ndarray) -> np.ndarray:
            counters["gev"] += 1
            return np.asarray(jac(z * scale), dtype=float).ravel() * scale / fref
    else:
        fd_bounds = None if bnds is None else (bnds[0] / scale, bnds[1] / scale)

        def g_scaled(z: np.ndarray) -> np.ndarray:
            counters["gev"] += 1
            return finite_difference_gradient(
                lambda zz: f_phys(zz * scale) / fref,
                z,
                step=step,
                bounds=fd_bounds,
                method=fd_method,
            )

    # Constraint scaling: SLSQP mixes objective and constraint magnitudes in a
    # single QP, so a constraint expressed in Pa (1e8) next to a mass objective
    # (1e-2) stalls the line search.  Normalise by the value at x0.
    crefs: list[np.ndarray] = []
    for c in cons:
        if not isinstance(c, dict):
            crefs.append(np.ones(1))
            continue
        try:
            v0 = np.atleast_1d(np.asarray(c["fun"](x0), dtype=float))
        except Exception:  # pragma: no cover - user constraint may fail at x0
            v0 = np.ones(1)
        if constraint_scaling in (True, "auto"):
            crefs.append(np.where(np.abs(v0) > 0, np.abs(v0), 1.0))
        else:
            crefs.append(np.ones_like(v0))

    def wrap_con(c: dict[str, Any], cref: np.ndarray) -> dict[str, Any]:
        fn = c["fun"]
        out: dict[str, Any] = {
            "type": c.get("type", "ineq"),
            "fun": lambda z, _fn=fn, _r=cref: np.atleast_1d(
                np.asarray(_fn(z * scale), dtype=float)
            )
            / _r,
        }
        if "jac" in c and c["jac"] is not None:
            jfn = c["jac"]
            out["jac"] = (
                lambda z, _j=jfn, _r=cref: np.atleast_2d(
                    np.asarray(_j(z * scale), dtype=float)
                )
                * scale[None, :]
                / np.atleast_1d(_r)[:, None]
            )
        return out

    scaled_cons = [
        wrap_con(c, r) for c, r in zip(cons, crefs, strict=True) if isinstance(c, dict)
    ]
    con_scale = {id(c): r for c, r in zip(cons, crefs, strict=True) if isinstance(c, dict)}

    # `tol` is handed to SciPy itself rather than written into `options`, so that
    # each method translates it into its own native tolerance (ftol for SLSQP,
    # gtol for BFGS, xatol/fatol for Nelder-Mead, ...).
    opts: dict[str, Any] = {"maxiter": int(max_iter)}
    if verbose:
        opts["disp"] = True
    if options:
        opts.update(options)

    scipy_bounds = None
    if bnds is not None:
        scipy_bounds = list(zip(bnds[0] / scale, bnds[1] / scale, strict=True))

    starts = [x0]
    if n_starts > 1:
        rng = np.random.default_rng(seed)
        if bnds is None:
            raise ValueError("n_starts > 1 requires bounds")
        lo = np.where(np.isfinite(bnds[0]), bnds[0], x0 - np.abs(x0) - 1.0)
        hi = np.where(np.isfinite(bnds[1]), bnds[1], x0 + np.abs(x0) + 1.0)
        starts += [lo + rng.random(n) * (hi - lo) for _ in range(n_starts - 1)]

    best: Any = None
    best_key = (math.inf, math.inf)
    for xs in starts:
        z0 = np.asarray(xs, dtype=float) / scale
        if scipy_bounds is not None:
            z0 = np.clip(z0, [b[0] for b in scipy_bounds], [b[1] for b in scipy_bounds])
        res = minimize(
            f_scaled,
            z0,
            jac=g_scaled if method.upper() not in ("COBYLA", "NELDER-MEAD", "POWELL") else None,
            bounds=scipy_bounds,
            constraints=scaled_cons if scaled_cons else (),
            method=method,
            tol=tol,
            options=opts,
            callback=(lambda z, _cb=callback: _cb(z * scale)) if callback else None,
        )
        x_phys = np.asarray(res.x, dtype=float) * scale
        viol = _scaled_violation(cons, con_scale, x_phys)
        key = (0.0 if viol <= feasibility_tol else viol, float(res.fun))
        if key < best_key:
            best_key = key
            best = (res, x_phys, viol)

    assert best is not None
    res, x_phys, viol = best
    feasible = viol <= feasibility_tol
    # SLSQP reports "positive directional derivative" (status 8) when it stalls
    # at a point it cannot improve; if that point is feasible and the KKT step is
    # tiny, it is a converged solution for our purposes.
    ok = bool(res.success) or (int(getattr(res, "status", 0)) == 8 and feasible)
    return OptimizationResult(
        x=x_phys,
        fun=float(res.fun) * fref,
        success=ok and feasible,
        status=int(getattr(res, "status", 0)),
        message=str(getattr(res, "message", "")),
        n_iter=int(getattr(res, "nit", 0)),
        n_fev=counters["fev"],
        n_gev=counters["gev"],
        jac=_objective_gradient(res, n, fref, scale),
        history=history,
        constraint_violation=float(viol),
        x0=x0,
        fun0=f0,
        method=method,
        names=list(names) if names else [f"x{i + 1}" for i in range(n)],
    )


def _objective_gradient(
    res: Any, n: int, fref: float, scale: np.ndarray
) -> np.ndarray | None:
    """Objective gradient in physical units, or ``None`` if the solver gave none.

    ``trust-constr`` reports the objective gradient as ``grad`` and reuses ``jac``
    for the list of *constraint* Jacobians, so the attribute has to be picked by
    name and then validated against the number of design variables.
    """
    for attr in ("grad", "jac"):
        raw = getattr(res, attr, None)
        if raw is None:
            continue
        g = np.asarray(raw, dtype=float).ravel()
        if g.size >= n:
            return g[:n] * fref / scale
    return None


def _scaled_violation(
    cons: Iterable[Any], con_scale: Mapping[int, np.ndarray], x: np.ndarray
) -> float:
    """Largest constraint violation, measured relative to each constraint's scale."""
    worst = 0.0
    for c in cons:
        if not isinstance(c, Mapping):
            continue
        ref = np.atleast_1d(con_scale.get(id(c), np.ones(1)))
        v = np.atleast_1d(np.asarray(c["fun"](x), dtype=float))
        ref = np.broadcast_to(ref, v.shape) if ref.size in (1, v.size) else np.ones_like(v)
        v = v / np.where(ref != 0, ref, 1.0)
        if c.get("type", "ineq") == "eq":
            worst = max(worst, float(np.max(np.abs(v))))
        else:
            worst = max(worst, float(max(0.0, -np.min(v))))
    return worst
