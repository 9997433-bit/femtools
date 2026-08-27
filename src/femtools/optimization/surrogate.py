"""Polynomial response-surface models (RSM) for surrogate-based design.

A response surface replaces an expensive analysis by a low-order polynomial
fitted to a design of experiments, so an optimiser, a Monte-Carlo run or a
sensitivity study can be driven at zero marginal cost.  The workhorse is the
full quadratic

.. math::
    \\hat y(z) = \\beta_0 + \\sum_j \\beta_j z_j
              + \\sum_{j<k} \\beta_{jk} z_j z_k
              + \\sum_j \\beta_{jj} z_j^2,

which is the lowest order able to represent curvature and therefore an
interior optimum — the reason Box & Wilson built the whole RSM methodology on
it.  Its :math:`(d+1)(d+2)/2` coefficients are what a central-composite or
Box-Behnken plan is sized to support.

Two details separate a usable implementation from a numerically embarrassing
one:

*Coding.*  Fitting in physical units (a modulus in pascals next to a thickness
in millimetres) squares the spread of the design matrix and destroys the
normal equations.  Inputs are therefore mapped to :math:`z \\in [-1, 1]` over
the sampled range before any term is formed, and mapped back transparently on
prediction.

*Honest error.*  A quadratic through :math:`p` points with :math:`n \\approx p`
has an :math:`R^2` near one no matter how wrong it is.  Every fit therefore
also reports the leave-one-out :math:`Q^2 = 1 - \\mathrm{PRESS}/\\mathrm{TSS}`,
obtained exactly from the hat-matrix diagonal — a cheap, unbiased answer to
"does this surface predict, or has it merely interpolated?".

Because the fitted model is a quadratic, its stationary point and canonical
form are available in closed form (:meth:`RSMFit.stationary_point`), which is
the classical way to read an optimum, a ridge or a saddle off a response
surface.

Examples
--------
>>> import numpy as np
>>> from femtools.optimization import central_composite, fit_rsm, predict_rsm
>>> x = central_composite(bounds=[(1.0, 3.0), (0.0, 4.0)])
>>> y = 5.0 + (x[:, 0] - 2.0) ** 2 + 0.5 * x[:, 1]
>>> rsm = fit_rsm(x, y)
>>> bool(rsm.r2 > 0.999)
True
>>> float(np.round(predict_rsm(rsm, [[2.0, 0.0]])[0], 6))
5.0
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "RSMFit",
    "fit_rsm",
    "predict_rsm",
    "rsm_terms",
    "design_matrix",
]


# ----------------------------------------------------------------------
def rsm_terms(
    n_dim: int, order: int = 2, *, interactions: bool = True, pure_quadratic: bool = True
) -> tuple[np.ndarray, list[str]]:
    """Exponent table and names of the polynomial terms.

    Parameters
    ----------
    n_dim:
        Number of input variables.
    order:
        1 for a linear model, 2 for a quadratic one.
    interactions:
        Include the cross terms ``z_j z_k`` (order 2 only).
    pure_quadratic:
        Include the squares ``z_j^2`` (order 2 only).  Setting this to False
        with ``interactions=True`` gives the "linear plus interaction" model
        that a two-level factorial plan can support.

    Returns
    -------
    (powers, names):
        ``powers`` has shape ``(n_terms, n_dim)``; ``names`` labels them.

    Examples
    --------
    >>> from femtools.optimization.surrogate import rsm_terms
    >>> _, names = rsm_terms(2, 2)
    >>> names
    ['1', 'x1', 'x2', 'x1*x2', 'x1^2', 'x2^2']
    """
    d = int(n_dim)
    if d < 1:
        raise ValueError("n_dim must be at least 1")
    o = int(order)
    if o not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    powers = [np.zeros(d, dtype=int)]
    names = ["1"]
    for j in range(d):
        e = np.zeros(d, dtype=int)
        e[j] = 1
        powers.append(e)
        names.append(f"x{j + 1}")
    if o >= 2 and interactions:
        for j in range(d):
            for k in range(j + 1, d):
                e = np.zeros(d, dtype=int)
                e[j] = e[k] = 1
                powers.append(e)
                names.append(f"x{j + 1}*x{k + 1}")
    if o >= 2 and pure_quadratic:
        for j in range(d):
            e = np.zeros(d, dtype=int)
            e[j] = 2
            powers.append(e)
            names.append(f"x{j + 1}^2")
    return np.asarray(powers, dtype=int), names


def design_matrix(z: ArrayLike, powers: ArrayLike) -> np.ndarray:
    """Evaluate the polynomial terms of ``powers`` at the (coded) points ``z``."""
    Z = np.atleast_2d(np.asarray(z, dtype=float))
    P = np.atleast_2d(np.asarray(powers, dtype=int))
    if Z.shape[1] != P.shape[1]:
        raise ValueError(f"{Z.shape[1]} inputs do not match {P.shape[1]}-variable terms")
    X = np.ones((Z.shape[0], P.shape[0]))
    for t, p in enumerate(P):
        for j in np.nonzero(p)[0]:
            X[:, t] *= Z[:, j] ** int(p[j])
    return X


# ----------------------------------------------------------------------
@dataclass
class RSMFit:
    """A fitted polynomial response surface (result of :func:`fit_rsm`).

    Prediction goes through :meth:`predict` (or simply calling the object);
    the coefficients are stored **in coded space**, so read them through
    :meth:`coefficient_table` rather than assuming physical units.

    Attributes
    ----------
    coefficients:
        ``(n_terms,)`` for a single response, ``(n_terms, n_response)``
        otherwise, in the order of :attr:`term_names`.
    center, scale:
        Coding transform, ``z = (x - center) / scale``.
    r2, r2_adj, q2:
        Coefficient of determination, its degrees-of-freedom-adjusted form,
        and the leave-one-out predictive :math:`Q^2`.  A large gap between
        ``r2`` and ``q2`` is the signature of an over-fitted surface.
    rmse, press:
        Residual root-mean-square error and the PRESS statistic.
    std_error, t_values:
        Coefficient standard errors and ``beta/se``; ``None`` when the fit has
        no residual degrees of freedom.  ``|t| < 2`` marks a term the data
        does not actually support.
    condition_number:
        Condition number of the coded design matrix.  Above ~1e3 the plan is
        too degenerate for the requested term set.
    """

    coefficients: np.ndarray
    term_names: list[str]
    powers: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    r2: Any
    r2_adj: Any
    q2: Any
    rmse: Any
    press: Any
    residuals: np.ndarray
    fitted: np.ndarray
    n_samples: int
    condition_number: float
    order: int = 2
    ridge: float = 0.0
    std_error: np.ndarray | None = None
    t_values: np.ndarray | None = None
    input_names: list[str] = field(default_factory=list)
    response_names: list[str] = field(default_factory=list)
    leverage: np.ndarray = field(default_factory=lambda: np.zeros(0))
    single: bool = True

    # -- shape helpers ---------------------------------------------------
    @property
    def n_dim(self) -> int:
        return int(self.powers.shape[1])

    @property
    def n_terms(self) -> int:
        return int(self.powers.shape[0])

    @property
    def n_response(self) -> int:
        return 1 if self.single else int(np.asarray(self.coefficients).shape[1])

    @property
    def dof(self) -> int:
        """Residual degrees of freedom."""
        return int(self.n_samples - self.n_terms)

    def encode(self, x: ArrayLike) -> np.ndarray:
        """Physical -> coded coordinates."""
        X = np.atleast_2d(np.asarray(x, dtype=float))
        if X.shape[1] != self.n_dim:
            raise ValueError(f"expected {self.n_dim} inputs, got {X.shape[1]}")
        return (X - self.center[None, :]) / self.scale[None, :]

    def decode(self, z: ArrayLike) -> np.ndarray:
        """Coded -> physical coordinates."""
        Z = np.atleast_2d(np.asarray(z, dtype=float))
        return self.center[None, :] + Z * self.scale[None, :]

    # -- evaluation ------------------------------------------------------
    def _as_points(self, x: ArrayLike) -> tuple[np.ndarray, bool]:
        """``(points, was_a_single_point)`` — 1-D input is one point, unless
        the surface is univariate, where it is a list of points."""
        X = np.asarray(x, dtype=float)
        if X.ndim == 0:
            return X.reshape(1, 1), True
        if X.ndim == 1:
            if self.n_dim == 1:
                return X[:, None], X.size == 1
            return X[None, :], True
        return X, False

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Evaluate the surface at one or many points."""
        X, one = self._as_points(x)
        y = design_matrix(self.encode(X), self.powers) @ np.asarray(self.coefficients)
        return y[0] if one else y

    def __call__(self, x: ArrayLike) -> np.ndarray:
        return self.predict(x)

    def gradient(self, x: ArrayLike) -> np.ndarray:
        """Gradient in *physical* units at ``x``, shape ``(n_dim,)``."""
        z = self.encode(self._as_points(x)[0])[0]
        beta = np.atleast_2d(np.asarray(self.coefficients).T)  # (n_resp, n_terms)
        g = np.zeros((beta.shape[0], self.n_dim))
        for t, p in enumerate(self.powers):
            for j in np.nonzero(p)[0]:
                other = p.copy()
                other[j] -= 1
                term = float(p[j]) * np.prod(z ** other.astype(float))
                g[:, j] += beta[:, t] * term
        g = g / self.scale[None, :]
        return g[0] if self.single else g

    def hessian(self) -> np.ndarray:
        """Constant Hessian in physical units (quadratic models only)."""
        if self.order < 2:
            return np.zeros((self.n_dim, self.n_dim))
        beta = np.atleast_2d(np.asarray(self.coefficients).T)
        H = np.zeros((beta.shape[0], self.n_dim, self.n_dim))
        for t, p in enumerate(self.powers):
            nz = np.nonzero(p)[0]
            if int(p.sum()) != 2:
                continue
            if nz.size == 1:
                H[:, nz[0], nz[0]] += 2.0 * beta[:, t]
            else:
                H[:, nz[0], nz[1]] += beta[:, t]
                H[:, nz[1], nz[0]] += beta[:, t]
        H = H / (self.scale[None, :, None] * self.scale[None, None, :])
        return H[0] if self.single else H

    def stationary_point(self, response: int = 0) -> dict[str, Any]:
        """Canonical analysis: the stationary point of the fitted quadratic.

        Returns a dict with the physical location ``x``, the predicted value
        ``y``, the Hessian eigenvalues ``eigenvalues`` and a ``kind`` of
        ``"minimum"``, ``"maximum"``, ``"saddle"`` or ``"ridge"`` (a
        near-singular Hessian, i.e. a direction along which the response is
        flat and the "optimum" is not identified).
        """
        if self.order < 2:
            raise ValueError("a linear surface has no stationary point")
        beta = np.atleast_2d(np.asarray(self.coefficients).T)[int(response)]
        Hm = self.hessian() if self.single else np.asarray(self.hessian())[int(response)]
        # gradient of the coded model at the coded origin, then in physical units
        g0 = np.zeros(self.n_dim)
        for t, p in enumerate(self.powers):
            if int(p.sum()) == 1:
                g0[int(np.nonzero(p)[0][0])] = beta[t]
        g_phys = g0 / self.scale
        # shift: grad(x) = g(center) + H (x - center) = 0
        vals, vecs = np.linalg.eigh(0.5 * (Hm + Hm.T))
        scale_v = float(np.max(np.abs(vals))) if vals.size else 0.0
        tol = 1.0e-8 * max(scale_v, 1.0e-300)
        if np.any(np.abs(vals) <= tol):
            kind = "ridge"
            inv = np.where(np.abs(vals) > tol, 1.0 / np.where(vals == 0, 1.0, vals), 0.0)
            step = -(vecs @ (inv * (vecs.T @ g_phys)))
        else:
            kind = "minimum" if np.all(vals > 0) else "maximum" if np.all(vals < 0) else "saddle"
            step = -np.linalg.solve(Hm, g_phys)
        x_star = self.center + step
        y_star = self.predict(x_star[None, :])
        y_val = float(np.asarray(y_star).ravel()[int(response) if not self.single else 0])
        return {
            "x": x_star,
            "y": y_val,
            "kind": kind,
            "eigenvalues": vals,
            "eigenvectors": vecs,
        }

    # -- reporting -------------------------------------------------------
    def coefficient_table(self, response: int = 0) -> dict[str, float]:
        """Coded coefficients keyed by term name."""
        beta = np.atleast_2d(np.asarray(self.coefficients).T)[int(response)]
        return {n: float(v) for n, v in zip(self.term_names, beta, strict=True)}

    def summary(self, response: int = 0) -> str:
        r2 = float(np.atleast_1d(self.r2)[int(response)])
        q2 = float(np.atleast_1d(self.q2)[int(response)])
        rmse = float(np.atleast_1d(self.rmse)[int(response)])
        adj = float(np.atleast_1d(self.r2_adj)[int(response)])
        name = (
            self.response_names[int(response)]
            if len(self.response_names) > int(response)
            else f"y{int(response) + 1}"
        )
        lines = [
            f"RSM order {self.order}: {self.n_terms} terms from {self.n_samples} "
            f"samples in {self.n_dim} variables  [{name}]",
            f"  R2 {r2:.6f}   R2_adj {adj:.6f}   Q2(LOO) {q2:.6f}   RMSE {rmse:.6g}",
            f"  design condition number {self.condition_number:.4g}",
        ]
        beta = np.atleast_2d(np.asarray(self.coefficients).T)[int(response)]
        tvals = None
        if self.t_values is not None:
            tv_arr = np.asarray(self.t_values)
            tvals = tv_arr if tv_arr.ndim == 1 else tv_arr[:, int(response)]
        for t, term in enumerate(self.term_names):
            tv = "" if tvals is None else f"   t={float(tvals[t]):8.2f}"
            lines.append(f"    {term:<12s} {beta[t]:14.6g}{tv}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RSMFit(order={self.order}, n_terms={self.n_terms}, "
            f"n_samples={self.n_samples}, r2={np.atleast_1d(self.r2)[0]:.5f}, "
            f"q2={np.atleast_1d(self.q2)[0]:.5f})"
        )


# ----------------------------------------------------------------------
def _resolve_order(order: Any, interactions: Any, pure_quadratic: Any) -> tuple[int, bool, bool]:
    inter = True if interactions is None else bool(interactions)
    pure = True if pure_quadratic is None else bool(pure_quadratic)
    if isinstance(order, str):
        key = order.strip().lower()
        if key in ("linear", "first", "1"):
            return 1, False, False
        if key in ("interaction", "interactions", "2fi", "linear+interaction"):
            return 2, True, False
        if key in ("quadratic", "second", "full", "2"):
            return 2, inter, pure
        if key in ("pure-quadratic", "pure_quadratic", "purequadratic"):
            return 2, False, True
        raise ValueError(f"unknown RSM order {order!r}")
    o = int(order)
    if o == 1:
        return 1, False, False
    return o, inter, pure


def fit_rsm(
    x: ArrayLike,
    y: Any = None,
    *,
    order: Any = 2,
    interactions: bool | None = None,
    pure_quadratic: bool | None = None,
    weights: ArrayLike | None = None,
    ridge: float = 0.0,
    code: bool | str = True,
    names: Sequence[str] | None = None,
    response_names: Sequence[str] | None = None,
) -> RSMFit:
    """Fit a polynomial response surface to sampled data.

    Parameters
    ----------
    x:
        ``(n_samples, n_dim)`` design points, in physical units.  A 1-D array
        is read as ``n_samples`` points of a single variable.
    y:
        ``(n_samples,)`` or ``(n_samples, n_response)`` responses, or a
        callable evaluated at each row of ``x`` (handy straight after a DOE).
    order:
        ``2`` (default, full quadratic), ``1``/``"linear"``,
        ``"interaction"`` (linear + cross terms, no squares) or
        ``"pure-quadratic"`` (linear + squares, no cross terms).
    interactions, pure_quadratic:
        Fine-grained control over which second-order terms are included.
    weights:
        Per-sample weights, applied as a weighted least-squares fit — use
        replicate counts, or ``1/sigma^2`` when the samples carry noise.
    ridge:
        Tikhonov weight on the non-constant coded coefficients, relative to
        the mean squared column norm.  Only needed for plans too small for
        the term set; it biases the fit, so prefer dropping terms.
    code:
        ``True``/``"range"`` (default) maps each variable to ``[-1, 1]`` over
        its sampled range, ``"std"`` standardises to zero mean and unit
        variance, ``False`` fits in physical units (not recommended).
    names, response_names:
        Optional labels used by :meth:`RSMFit.summary`.

    Returns
    -------
    RSMFit

    Raises
    ------
    ValueError
        If there are fewer samples than terms (an under-determined fit),
        unless ``ridge > 0`` explicitly accepts the regularised solution.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.optimization.surrogate import fit_rsm
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1.0, 1.0, size=(40, 2))
    >>> y = 3.0 + 2.0 * x[:, 0] - x[:, 1] + 4.0 * x[:, 0] * x[:, 1] + x[:, 1] ** 2
    >>> rsm = fit_rsm(x, y)
    >>> bool(rsm.r2 > 1 - 1e-12 and rsm.q2 > 1 - 1e-12)
    True
    >>> sorted(k for k, v in rsm.coefficient_table().items() if abs(v) > 1e-9)
    ['1', 'x1', 'x1*x2', 'x2', 'x2^2']
    """
    X_raw = np.asarray(x, dtype=float)
    if X_raw.ndim == 1:
        X_raw = X_raw[:, None]
    if X_raw.ndim != 2:
        raise ValueError(f"x must be 2-D (n_samples, n_dim), got shape {X_raw.shape}")
    n, d = X_raw.shape

    if callable(y):
        fun: Callable[[np.ndarray], Any] = y
        y = np.asarray([np.asarray(fun(row), dtype=float) for row in X_raw], dtype=float)
    if y is None:
        raise TypeError("fit_rsm requires responses `y`")
    Y = np.asarray(y, dtype=float)
    single = Y.ndim == 1
    Y2 = Y[:, None] if single else Y
    if Y2.shape[0] != n:
        raise ValueError(f"{Y2.shape[0]} responses for {n} design points")

    # ---- coding ---------------------------------------------------------
    code_key = "range" if code is True else ("none" if code is False else str(code).lower())
    if code_key in ("range", "coded", "minmax"):
        lo, hi = X_raw.min(axis=0), X_raw.max(axis=0)
        center = 0.5 * (lo + hi)
        scale = 0.5 * (hi - lo)
    elif code_key in ("std", "standard", "zscore"):
        center = X_raw.mean(axis=0)
        scale = X_raw.std(axis=0)
    elif code_key in ("none", "off", "physical"):
        center = np.zeros(d)
        scale = np.ones(d)
    else:
        raise ValueError(f"unknown coding {code!r}")
    scale = np.where(np.abs(scale) > 0, scale, 1.0)
    Z = (X_raw - center[None, :]) / scale[None, :]

    o, inter, pure = _resolve_order(order, interactions, pure_quadratic)
    powers, term_names = rsm_terms(d, o, interactions=inter, pure_quadratic=pure)
    if names is not None:
        if len(names) != d:
            raise ValueError(f"{len(names)} names for {d} variables")
        lookup = {f"x{j + 1}": str(names[j]) for j in range(d)}
        term_names = [
            "1"
            if t == "1"
            else "*".join(
                lookup.get(part.split("^")[0], part.split("^")[0])
                + ("^2" if part.endswith("^2") else "")
                for part in t.split("*")
            )
            for t in term_names
        ]
    p = powers.shape[0]
    if n < p and ridge <= 0.0:
        raise ValueError(
            f"{n} samples cannot determine {p} terms; sample more points, lower "
            f"`order`, or pass `ridge` to accept a regularised fit"
        )

    A = design_matrix(Z, powers)
    cond = float(np.linalg.cond(A)) if min(A.shape) else math.nan

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n:
            raise ValueError(f"{w.size} weights for {n} samples")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        sw = np.sqrt(w)[:, None]
        Aw, Yw = A * sw, Y2 * sw
    else:
        Aw, Yw = A, Y2

    # ---- solve ----------------------------------------------------------
    G = Aw.T @ Aw
    if ridge > 0.0:
        reg = float(ridge) * max(float(np.mean(np.diag(G))), 1e-300)
        R = reg * np.eye(p)
        R[0, 0] = 0.0  # never shrink the mean
        G = G + R
        beta = np.linalg.solve(G, Aw.T @ Yw)
        Ginv = np.linalg.inv(G)
    else:
        beta, *_ = np.linalg.lstsq(Aw, Yw, rcond=None)
        Ginv = np.linalg.pinv(G)

    fitted = A @ beta
    resid = Y2 - fitted

    # ---- statistics ------------------------------------------------------
    wv = np.ones(n) if w is None else w
    wsum = float(np.sum(wv))
    ybar = np.sum(wv[:, None] * Y2, axis=0) / max(wsum, 1e-300)
    sse = np.sum(wv[:, None] * resid**2, axis=0)
    sst = np.sum(wv[:, None] * (Y2 - ybar[None, :]) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(sst > 0, 1.0 - sse / np.where(sst > 0, sst, 1.0), 1.0)
    dof = n - p
    if dof > 0 and n > 1:
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / dof
    else:
        r2_adj = np.full(r2.shape, math.nan)
    rmse = np.sqrt(sse / max(wsum, 1e-300))

    # leave-one-out via the hat-matrix diagonal (Allen's PRESS).  With no
    # residual degrees of freedom the surface interpolates, every leverage is
    # one and PRESS says nothing, so it is reported as NaN rather than as a
    # spurious Q2 of 1.
    lev = np.clip(np.einsum("ij,jk,ik->i", Aw, Ginv, Aw), 0.0, 1.0)
    if dof > 0:
        loo = resid / np.maximum(1.0 - lev, 1e-12)[:, None]
        press = np.sum(wv[:, None] * loo**2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            q2 = np.where(sst > 0, 1.0 - press / np.where(sst > 0, sst, 1.0), 1.0)
    else:
        press = np.full(Y2.shape[1], math.nan)
        q2 = np.full(Y2.shape[1], math.nan)

    std_err: np.ndarray | None = None
    tvals: np.ndarray | None = None
    if dof > 0:
        sigma2 = sse / dof
        var = np.clip(np.diag(Ginv), 0.0, None)
        std_err = np.sqrt(np.outer(var, sigma2))  # (p, n_resp)
        with np.errstate(divide="ignore", invalid="ignore"):
            tvals = np.where(std_err > 0, beta / np.where(std_err > 0, std_err, 1.0), 0.0)
        if single:
            std_err = std_err[:, 0]
            tvals = tvals[:, 0]

    return RSMFit(
        coefficients=beta[:, 0] if single else beta,
        term_names=term_names,
        powers=powers,
        center=center,
        scale=scale,
        r2=float(r2[0]) if single else r2,
        r2_adj=float(r2_adj[0]) if single else r2_adj,
        q2=float(q2[0]) if single else q2,
        rmse=float(rmse[0]) if single else rmse,
        press=float(press[0]) if single else press,
        residuals=resid[:, 0] if single else resid,
        fitted=fitted[:, 0] if single else fitted,
        n_samples=n,
        condition_number=cond,
        order=o,
        ridge=float(ridge),
        std_error=std_err,
        t_values=tvals,
        input_names=(
            [str(s) for s in names] if names is not None else [f"x{j + 1}" for j in range(d)]
        ),
        response_names=(
            [str(s) for s in response_names]
            if response_names is not None
            else [f"y{i + 1}" for i in range(Y2.shape[1])]
        ),
        leverage=lev,
        single=single,
    )


def predict_rsm(model: Any, x: Any = None) -> np.ndarray:
    """Evaluate a fitted response surface.

    Accepts either argument order — ``predict_rsm(rsm, x)`` and
    ``predict_rsm(x, rsm)`` both work — so it reads naturally whichever way
    round the call site puts them.

    Parameters
    ----------
    model:
        An :class:`RSMFit` from :func:`fit_rsm`.
    x:
        A single point (1-D) or ``(n_points, n_dim)`` array, in physical
        units.  A single point returns a scalar for a single-response fit.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.optimization.surrogate import fit_rsm, predict_rsm
    >>> xs = np.linspace(-1.0, 1.0, 9)[:, None]
    >>> rsm = fit_rsm(xs, 1.0 + xs[:, 0] ** 2)
    >>> float(np.round(predict_rsm(rsm, [0.5]), 9))
    1.25
    """
    if not isinstance(model, RSMFit) and isinstance(x, RSMFit):
        model, x = x, model
    if not isinstance(model, RSMFit):
        raise TypeError("predict_rsm expects an RSMFit from fit_rsm()")
    if x is None:
        raise TypeError("predict_rsm requires the points to evaluate")
    return model.predict(x)
