"""Sensitivity-based model updating (Friswell & Mottershead).

Two estimators are implemented on top of the same damped Gauss--Newton loop:

**Weighted least squares** (``method="wls"``)

.. math::
    \\Delta p = \\left(S^T W_\\epsilon S + \\lambda D\\right)^{-1} S^T W_\\epsilon\\,
                \\left(r_m - r(p)\\right)

**Bayesian / minimum-variance** (``method="bayesian"``)

.. math::
    \\Delta p = \\left(S^T C_\\epsilon^{-1} S + C_p^{-1}\\right)^{-1}
        \\left[S^T C_\\epsilon^{-1}(r_m - r(p)) + C_p^{-1}(p_0 - p)\\right]

Both run inside a bound-constrained, line-searched, Levenberg--Marquardt damped
iteration so that large initial errors (the classic "10 % wrong Young's
modulus") converge robustly rather than overshooting.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .parameters import (
    Parameter,
    ParameterSet,
    apply_parameters,
    as_parameters,
    parameter_bounds,
    snapshot_baseline,
)
from .responses import ResponseSpec, modal_response_function
from .sensitivity import SensitivityResult, sensitivity_matrix

__all__ = ["UpdateResult", "update_model", "UpdateOptions"]


@dataclass
class UpdateOptions:
    """Solver settings for :func:`update_model`."""

    method: str = "wls"
    max_iter: int = 30
    tol: float = 1.0e-6
    residual_tol: float = 1.0e-12
    step: float = 1.0e-4
    sensitivity_method: str = "central"
    lm_damping: float = 1.0e-3
    lm_increase: float = 10.0
    lm_decrease: float = 0.3
    line_search: bool = True
    max_line_search: int = 12
    max_relative_step: float = 0.5
    verbose: bool = False


@dataclass
class UpdateResult:
    """Outcome of :func:`update_model`.

    Attributes
    ----------
    parameters:
        ``{name: value}`` of the updated parameters.
    x / values:
        Final parameter vector (same order as the input parameter set).
    initial:
        Starting parameter vector.
    response / residual:
        Final predicted response and ``targets - response``.
    converged, n_iter, message:
        Convergence diagnostics.
    history:
        Per-iteration records (``iteration``, ``x``, ``cost``, ``rms``, ``lambda``).
    covariance:
        Posterior parameter covariance estimate (``None`` if not computable).
    model:
        Updated deep copy of the input model (``None`` for callback problems).
    """

    parameters: dict[str, float]
    x: np.ndarray
    initial: np.ndarray
    targets: np.ndarray
    response: np.ndarray
    residual: np.ndarray
    initial_response: np.ndarray
    initial_residual: np.ndarray
    converged: bool = False
    n_iter: int = 0
    message: str = ""
    cost: float = math.nan
    initial_cost: float = math.nan
    history: list[dict[str, Any]] = field(default_factory=list)
    sensitivity: SensitivityResult | None = None
    covariance: np.ndarray | None = None
    parameter_std: np.ndarray | None = None
    model: Any = None
    parameter_names: list[str] = field(default_factory=list)
    method: str = "wls"
    n_evaluations: int = 0

    # -- convenience ----------------------------------------------------
    @property
    def values(self) -> np.ndarray:
        return self.x

    @property
    def p(self) -> np.ndarray:
        return self.x

    @property
    def updated_model(self) -> Any:
        return self.model

    @property
    def success(self) -> bool:
        return self.converged

    @property
    def rms_error(self) -> float:
        """RMS of the *relative* response error (dimensionless)."""
        return float(_relative_rms(self.targets, self.response))

    @property
    def initial_rms_error(self) -> float:
        return float(_relative_rms(self.targets, self.initial_response))

    @property
    def relative_error(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.targets != 0, self.residual / self.targets, 0.0)

    @property
    def residual_norm(self) -> float:
        return float(np.linalg.norm(self.residual))

    @property
    def improvement(self) -> float:
        """Fractional reduction of the relative RMS error (1.0 = perfect)."""
        i0 = self.initial_rms_error
        return float(1.0 - self.rms_error / i0) if i0 > 0 else 0.0

    def __getitem__(self, key: str) -> float:
        return self.parameters[key]

    def summary(self) -> str:
        lines = [
            f"update_model({self.method}) -> {'converged' if self.converged else 'stopped'} "
            f"in {self.n_iter} iterations ({self.message})",
            f"  relative RMS response error: {self.initial_rms_error:.4e} "
            f"-> {self.rms_error:.4e}",
        ]
        for i, name in enumerate(self.parameter_names):
            lines.append(
                f"  {name:<12s} {self.initial[i]:.6g} -> {self.x[i]:.6g} "
                f"({100.0 * (self.x[i] / self.initial[i] - 1.0):+.3f} %)"
                if self.initial[i]
                else f"  {name:<12s} {self.initial[i]:.6g} -> {self.x[i]:.6g}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"UpdateResult(converged={self.converged}, n_iter={self.n_iter}, "
            f"rms_error={self.rms_error:.3e}, parameters={self.parameters})"
        )


def _relative_rms(targets: np.ndarray, response: np.ndarray) -> float:
    t = np.asarray(targets, dtype=float)
    r = np.asarray(response, dtype=float)
    scale = np.where(np.abs(t) > 0, np.abs(t), 1.0)
    e = (r - t) / scale
    return float(np.sqrt(np.mean(e**2))) if e.size else 0.0


# ----------------------------------------------------------------------
def _build_weight_matrix(
    weights: Any, targets: np.ndarray
) -> tuple[np.ndarray, str]:
    """Return an ``(n, n)`` weight matrix ``W_eps`` plus a description."""
    n = targets.size
    if weights is None or (isinstance(weights, str) and weights.lower() == "relative"):
        scale = np.where(np.abs(targets) > 0, np.abs(targets), 1.0)
        return np.diag(1.0 / scale**2), "relative"
    if isinstance(weights, str):
        w = weights.lower()
        if w in ("unit", "identity", "none", "ones"):
            return np.eye(n), "unit"
        if w in ("inverse", "1/r", "reciprocal"):
            scale = np.where(np.abs(targets) > 0, np.abs(targets), 1.0)
            return np.diag(1.0 / scale), "inverse"
        raise ValueError(f"unknown weights {weights!r}")
    arr = np.asarray(weights, dtype=float)
    if arr.ndim == 0:
        return float(arr) * np.eye(n), "scalar"
    if arr.ndim == 1:
        if arr.size != n:
            raise ValueError(f"weights length {arr.size} != number of responses {n}")
        return np.diag(arr), "diagonal"
    if arr.shape != (n, n):
        raise ValueError(f"weight matrix must be {(n, n)}, got {arr.shape}")
    return arr, "full"


def _inverse_spd(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.linalg.pinv(A)


def update_model(
    model: Any,
    parameters: Any = None,
    targets: Any = None,
    *,
    response: Callable[[np.ndarray], np.ndarray] | None = None,
    weights: Any = None,
    method: str = "wls",
    bounds: Any = None,
    p0: Any = None,
    prior: Any = None,
    prior_cov: Any = None,
    prior_std: Any = None,
    measurement_cov: Any = None,
    noise_std: Any = 0.01,
    max_iter: int = 30,
    tol: float = 1.0e-6,
    step: float = 1.0e-4,
    sensitivity_method: str = "central",
    regularization: float = 0.0,
    lm_damping: float = 1.0e-3,
    line_search: bool = True,
    max_relative_step: float = 0.5,
    n_modes: int = 10,
    spec: ResponseSpec | None = None,
    solver: Callable[..., Any] | None = None,
    options: UpdateOptions | None = None,
    verbose: bool = False,
    callback: Callable[[int, np.ndarray, float], None] | None = None,
) -> UpdateResult:
    """Update ``model`` so its response matches ``targets``.

    Parameters
    ----------
    model:
        An :class:`femtools.core.model.FEModel`, a
        :class:`femtools.updating.reference.ReferenceModel`, or a callable
        ``f(p) -> response`` (solver-free / callback mode).
    parameters:
        Parameter specification — see :func:`femtools.updating.as_parameters`.
        Accepts :class:`Parameter` objects, dicts, name lists, or an integer
        count when using a callable response.
    targets:
        Measured response vector (e.g. experimental frequencies in Hz).  May be a
        mapping ``{name: value}``.
    response:
        Explicit response callback; overrides the model-driven one.
    weights:
        ``None``/``"relative"`` (default, ``W = diag(1/r_m^2)``), ``"unit"``,
        a per-response weight vector, or a full weight matrix.
    method:
        ``"wls"`` (default) or ``"bayesian"``.
    bounds:
        ``(lo, hi)`` pair, per-parameter sequence of pairs, or ``{name: (lo, hi)}``.
        Falls back to the bounds carried by the parameters themselves.
    prior, prior_cov, prior_std:
        Bayesian prior mean / covariance (or per-parameter standard deviation).
        Default prior mean is ``p0``; default std is 25 % of ``|p0|``.
    measurement_cov, noise_std:
        Bayesian measurement covariance, or (default) the measurement standard
        deviation expressed in the scale set by ``weights`` — with the default
        relative weighting, ``noise_std=0.01`` means 1 % measurement error.
    regularization:
        Extra Tikhonov weight added to the normal equations (on top of the
        adaptive Levenberg--Marquardt damping).

    Returns
    -------
    UpdateResult

    Examples
    --------
    Recover a 10 % Young's modulus error on the 2-parameter beam::

        from femtools.updating import update_model
        from femtools.updating.reference import make_updating_testcase

        f, p_true, p0, targets, beam = make_updating_testcase("beam", error=0.10)
        res = update_model(f, ["E1", "E2"], targets, p0=p0, bounds=(0.5, 2.0))
        assert abs(res.x[0] - p_true[0]) / p_true[0] < 0.02
    """
    opts = options or UpdateOptions(
        method=method,
        max_iter=max_iter,
        tol=tol,
        step=step,
        sensitivity_method=sensitivity_method,
        lm_damping=lm_damping,
        line_search=line_search,
        max_relative_step=max_relative_step,
        verbose=verbose,
    )
    meth = opts.method.lower()
    if meth in ("bayes", "bayesian", "map", "minimum-variance"):
        meth = "bayesian"
    elif meth in ("wls", "ls", "lsq", "least-squares", "gauss-newton", "sensitivity"):
        meth = "wls"
    else:
        raise ValueError(f"unknown updating method {method!r}")

    from .reference import ReferenceModel

    # ---- parameters ---------------------------------------------------
    if parameters is None:
        if isinstance(model, ReferenceModel):
            parameters = list(model.parameter_names)
        else:
            raise ValueError("`parameters` must be provided")
    pset: ParameterSet = as_parameters(parameters)

    # ---- response function --------------------------------------------
    fea_model: Any = None
    if response is not None:
        fun = response
    elif isinstance(model, ReferenceModel):
        fun = model.response_function(n_modes if spec is None else spec.n_modes)
    elif callable(model):
        fun = model
    else:
        fea_model = model
        fun = modal_response_function(model, pset, spec, solver=solver, n_modes=n_modes)

    # ---- start point ---------------------------------------------------
    if p0 is not None:
        x0 = np.atleast_1d(np.asarray(p0, dtype=float))
    else:
        x0 = pset.values
        if isinstance(model, ReferenceModel) and np.all(x0 == 1.0):
            x0 = np.ones(len(pset))
    if x0.size != len(pset):
        raise ValueError(f"p0 has {x0.size} entries but {len(pset)} parameters were given")

    lo, hi = parameter_bounds(pset, bounds)
    x0 = np.clip(x0, lo, hi)

    # ---- targets -------------------------------------------------------
    if targets is None:
        raise ValueError("`targets` must be provided")
    if isinstance(targets, Mapping):
        target_names = list(targets.keys())
        t = np.asarray([float(v) for v in targets.values()], dtype=float)
    else:
        target_names = []
        t = np.atleast_1d(np.asarray(targets, dtype=float)).ravel()

    n_eval = 0

    def evaluate(x: np.ndarray) -> np.ndarray:
        nonlocal n_eval
        n_eval += 1
        r = np.atleast_1d(np.asarray(fun(x), dtype=float)).ravel()
        if r.size < t.size:
            raise ValueError(
                f"response has {r.size} entries but {t.size} targets were given"
            )
        return r[: t.size]

    W, weight_kind = _build_weight_matrix(weights, t)

    # ---- Bayesian priors ------------------------------------------------
    Cp_inv = None
    x_prior = x0.copy()
    if meth == "bayesian":
        if prior is not None:
            x_prior = np.atleast_1d(np.asarray(prior, dtype=float))
        if prior_cov is not None:
            Cp = np.asarray(prior_cov, dtype=float)
            if Cp.ndim == 1:
                Cp = np.diag(Cp)
            elif Cp.ndim == 0:
                Cp = float(Cp) * np.eye(len(pset))
        else:
            if prior_std is not None:
                sd = np.asarray(prior_std, dtype=float)
                if sd.ndim == 0:
                    sd = np.full(len(pset), float(sd))
            else:
                sd = np.where(np.abs(x_prior) > 0, 0.25 * np.abs(x_prior), 0.25)
            Cp = np.diag(sd**2)
        Cp_inv = _inverse_spd(Cp)
        if measurement_cov is not None:
            Ce = np.asarray(measurement_cov, dtype=float)
            if Ce.ndim == 1:
                Ce = np.diag(Ce)
            elif Ce.ndim == 0:
                Ce = float(Ce) * np.eye(t.size)
            W = _inverse_spd(Ce)
        else:
            # `weights` only fixes the *relative* importance of the responses; the
            # Bayesian estimator additionally needs their absolute scale.  Interpret
            # `noise_std` as the measurement standard deviation in the units implied
            # by the weight matrix (default: 1 % relative error).
            if np.isscalar(noise_std):
                sig = float(np.asarray(noise_std, dtype=float))
                if sig > 0:
                    W = W / (sig**2)
            else:
                ns = np.asarray(noise_std, dtype=float)
                W = _inverse_spd(np.diag(ns**2))

    # ---- cost ------------------------------------------------------------
    def cost_of(x: np.ndarray, r: np.ndarray) -> float:
        d = t - r
        c = float(d @ W @ d)
        if Cp_inv is not None:
            dp = x - x_prior
            c += float(dp @ Cp_inv @ dp)
        return c

    x = x0.copy()
    r = evaluate(x)
    r0 = r.copy()
    cost = cost_of(x, r)
    cost0 = cost
    lam = float(opts.lm_damping)
    history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "x": x.copy(),
            "cost": cost,
            "rms": _relative_rms(t, r),
            "lambda": lam,
            "response": r.copy(),
        }
    ]
    if opts.verbose:
        print(f"[update] it=0 cost={cost:.6e} rms={_relative_rms(t, r):.6e} x={x}")

    converged = False
    message = "maximum iterations reached"
    S: SensitivityResult | None = None
    it = 0

    for it in range(1, opts.max_iter + 1):
        S = sensitivity_matrix(
            fun,
            x,
            parameters=pset,
            method=opts.sensitivity_method,
            step=opts.step,
            r0=r,
            bounds=(lo, hi),
        )
        Smat = np.asarray(S)[: t.size, :]
        n_eval += S.n_evaluations

        d = t - r
        A = Smat.T @ W @ Smat
        b = Smat.T @ W @ d
        if Cp_inv is not None:
            A = A + Cp_inv
            b = b + Cp_inv @ (x_prior - x)
        if regularization:
            A = A + regularization * np.eye(A.shape[0])

        diagA = np.diag(A).copy()
        diagA[diagA <= 0] = 1.0

        accepted = False
        for _ls in range(opts.max_line_search if opts.line_search else 1):
            try:
                dx = np.linalg.solve(A + lam * np.diag(diagA), b)
            except np.linalg.LinAlgError:  # pragma: no cover
                dx = np.linalg.lstsq(A + lam * np.diag(diagA), b, rcond=None)[0]
            # trust region: cap the relative parameter change per iteration
            scale = np.where(np.abs(x) > 0, np.abs(x), 1.0)
            rel = np.max(np.abs(dx) / scale) if dx.size else 0.0
            if opts.max_relative_step > 0 and rel > opts.max_relative_step:
                dx = dx * (opts.max_relative_step / rel)
            x_try = np.clip(x + dx, lo, hi)
            r_try = evaluate(x_try)
            c_try = cost_of(x_try, r_try)
            if np.isfinite(c_try) and c_try < cost:
                accepted = True
                break
            lam = min(lam * opts.lm_increase, 1.0e12)
            if not opts.line_search:
                break

        if not accepted:
            message = "no further cost reduction (line search exhausted)"
            converged = True
            history.append(
                {
                    "iteration": it,
                    "x": x.copy(),
                    "cost": cost,
                    "rms": _relative_rms(t, r),
                    "lambda": lam,
                    "response": r.copy(),
                    "rejected": True,
                }
            )
            break

        dx_eff = x_try - x
        rel_change = float(
            np.max(np.abs(dx_eff) / np.where(np.abs(x) > 0, np.abs(x), 1.0))
        )
        cost_drop = (cost - c_try) / cost if cost > 0 else 0.0
        x, r, cost = x_try, r_try, c_try
        lam = max(lam * opts.lm_decrease, 1.0e-12)

        history.append(
            {
                "iteration": it,
                "x": x.copy(),
                "cost": cost,
                "rms": _relative_rms(t, r),
                "lambda": lam,
                "response": r.copy(),
                "step_norm": rel_change,
            }
        )
        if opts.verbose:
            print(
                f"[update] it={it} cost={cost:.6e} rms={_relative_rms(t, r):.6e} "
                f"lam={lam:.2e} x={x}"
            )
        if callback is not None:
            callback(it, x.copy(), cost)

        if rel_change < opts.tol:
            converged = True
            message = f"parameter change below tol ({opts.tol:g})"
            break
        if cost <= opts.residual_tol:
            converged = True
            message = "residual below tolerance"
            break
        if cost_drop < opts.tol * 1.0e-2 and it > 1:
            converged = True
            message = "cost stagnated"
            break

    # ---- posterior covariance -------------------------------------------
    covariance = None
    par_std = None
    if S is not None:
        Smat = np.asarray(S)[: t.size, :]
        A = Smat.T @ W @ Smat
        if Cp_inv is not None:
            A = A + Cp_inv
        try:
            covariance = np.linalg.inv(A)
            if meth == "wls":
                dof = max(t.size - len(pset), 1)
                dres = t - r
                s2 = float(dres @ W @ dres) / dof
                covariance = covariance * s2
            par_std = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        except np.linalg.LinAlgError:  # pragma: no cover
            covariance = None

    # ---- write parameters back -------------------------------------------
    updated_model = None
    if fea_model is not None:
        base = snapshot_baseline(fea_model, pset)
        updated_model = apply_parameters(fea_model, pset, x, copy_model=True, baseline=base)
    for p_obj, v in zip(pset, x, strict=True):
        p_obj.value = float(v)

    result = UpdateResult(
        parameters=pset.to_dict(x),
        x=x,
        initial=x0,
        targets=t,
        response=r,
        residual=t - r,
        initial_response=r0,
        initial_residual=t - r0,
        converged=converged,
        n_iter=it,
        message=message,
        cost=cost,
        initial_cost=cost0,
        history=history,
        sensitivity=S,
        covariance=covariance,
        parameter_std=par_std,
        model=updated_model,
        parameter_names=list(pset.names),
        method=meth,
        n_evaluations=n_eval,
    )
    if target_names:
        result.history[-1].setdefault("target_names", target_names)
    if weight_kind:
        result.history[0].setdefault("weights", weight_kind)
    return result


def make_parameter(  # pragma: no cover - thin convenience wrapper
    name: str, kind: str = "E", target: Any = None, **kwargs: Any
) -> Parameter:
    """Shorthand for ``Parameter(name=..., kind=..., target=...)``."""
    return Parameter(name=name, kind=kind, target=target, **kwargs)
