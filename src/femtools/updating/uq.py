"""Uncertainty quantification for sensitivity-based model updating.

Two complementary estimators of the parameter uncertainty left after an
updating run (Friswell & Mottershead, *Finite Element Model Updating in
Structural Dynamics*, ch. 8; Mottershead, Link & Friswell, MSSP 2011):

**First order (linearised) covariance** -- :func:`parameter_covariance`

Around the converged parameters the residual is linear in ``theta`` through the
updating Jacobian ``J = dr/dtheta``, so the weighted least squares estimator
``theta_hat = (J^T W J)^{-1} J^T W r_m`` propagates the measurement covariance
``C_r`` as the sandwich

.. math::
    \\mathrm{Cov}(\\theta) = (J^T W J)^{-1}\\,\\left(J^T W C_r W J\\right)\\,
                             (J^T W J)^{-1} ,

which collapses to the familiar minimum-variance form
:math:`(J^T C_r^{-1} J)^{-1}` for the statistically optimal weighting
``W = C_r^{-1}``.  With no covariance information at all the residual itself
provides the scale, :math:`\\hat\\sigma^2 = r^T W r / (n_r - n_\\theta)`, and the
covariance is :math:`\\hat\\sigma^2 (J^T W J)^{-1}`.

**Monte Carlo** -- :func:`monte_carlo_update`

The linearisation is only as good as the local model.  Re-running the whole
updating loop on perturbed data ("Monte Carlo model updating") makes no
linearity assumption and additionally exposes non-uniqueness: the empirical
covariance of the re-identified parameters is the answer, and the sample cloud
shows skew or multi-modality that a covariance matrix cannot.  The random
stream is always explicit -- ``seed`` is a required argument, so every reported
number is reproducible.

Documented subset
-----------------
Implemented: first-order covariance (plain, weighted, sandwich and
prior-informed forms), correlation / standard deviation / confidence intervals,
and Monte Carlo re-updating with three resampling schemes (measurement noise on
the targets, residual bootstrap, perturbed start points).

Not implemented here: Markov-chain Monte Carlo posteriors, Bayesian model
selection and interval/fuzzy (non-probabilistic) model updating.  The Bayesian
*point* estimator (maximum a posteriori) lives in
:func:`femtools.updating.update_model` with ``method="bayesian"``, and its
posterior covariance is the ``prior_cov`` branch of
:func:`parameter_covariance`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

__all__ = ["UQResult", "parameter_covariance", "monte_carlo_update"]


@dataclass
class UQResult:
    """Parameter uncertainty: a mean, a covariance and (optionally) samples.

    Attributes
    ----------
    mean:
        Parameter vector the uncertainty refers to -- the converged values for
        :func:`parameter_covariance`, the sample mean for
        :func:`monte_carlo_update`.
    covariance:
        ``(n_parameter, n_parameter)`` covariance matrix.
    samples:
        ``(n_samples, n_parameter)`` Monte Carlo draws, or ``None`` for the
        analytic first-order estimate.
    parameter_names:
        Names in the same order as ``mean``.
    method:
        How the covariance was obtained, e.g. ``"first-order"``,
        ``"monte-carlo(targets)"``.
    """

    mean: np.ndarray
    covariance: np.ndarray
    samples: np.ndarray | None = None
    parameter_names: list[str] = field(default_factory=list)
    method: str = "first-order"
    n_samples: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = np.atleast_1d(np.asarray(self.mean, dtype=float)).ravel()
        self.covariance = np.atleast_2d(np.asarray(self.covariance, dtype=float))
        if self.samples is not None:
            self.samples = np.atleast_2d(np.asarray(self.samples, dtype=float))
            if not self.n_samples:
                self.n_samples = int(self.samples.shape[0])
        n = self.mean.size
        if self.covariance.shape != (n, n):
            raise ValueError(
                f"covariance must be {(n, n)}, got {self.covariance.shape}"
            )
        if not self.parameter_names:
            self.parameter_names = [f"p{j + 1}" for j in range(n)]

    # -- basic statistics ------------------------------------------------
    @property
    def variance(self) -> np.ndarray:
        return np.clip(np.diag(self.covariance), 0.0, None)

    @property
    def std(self) -> np.ndarray:
        """Parameter standard deviations (square root of the diagonal)."""
        return np.sqrt(self.variance)

    @property
    def cov(self) -> np.ndarray:
        return self.covariance

    @property
    def n_parameters(self) -> int:
        return int(self.mean.size)

    @property
    def coefficient_of_variation(self) -> np.ndarray:
        """``std / |mean|`` -- the dimensionless uncertainty of each parameter."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(self.mean) > 0, self.std / np.abs(self.mean), np.inf)

    @property
    def correlation(self) -> np.ndarray:
        """Parameter correlation matrix; near-unit off-diagonals flag redundancy."""
        s = self.std
        safe = np.where(s > 0, s, 1.0)
        R = self.covariance / np.outer(safe, safe)
        R[s <= 0, :] = 0.0
        R[:, s <= 0] = 0.0
        np.fill_diagonal(R, 1.0)
        return np.clip(R, -1.0, 1.0)

    def interval(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Two-sided ``level`` confidence interval ``(lower, upper)``.

        Empirical percentiles when samples are available, otherwise the
        Gaussian ``mean +- z * std`` of the linearised estimate.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        if self.samples is not None and self.samples.size:
            q = 100.0 * (1.0 - level) / 2.0
            lo = np.percentile(self.samples, q, axis=0)
            hi = np.percentile(self.samples, 100.0 - q, axis=0)
            return lo, hi
        z = float(ndtri(0.5 + 0.5 * level))
        return self.mean - z * self.std, self.mean + z * self.std

    def percentile(self, q: ArrayLike) -> np.ndarray:
        """Sample percentiles (Monte Carlo results only)."""
        if self.samples is None:
            raise ValueError("percentiles need Monte Carlo samples")
        return np.percentile(self.samples, np.asarray(q, dtype=float), axis=0)

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {"mean": float(self.mean[j]), "std": float(self.std[j])}
            for j, name in enumerate(self.parameter_names)
        }

    def __getitem__(self, key: str | int) -> float:
        if isinstance(key, str):
            return float(self.mean[self.parameter_names.index(key)])
        return float(self.mean[key])

    def summary(self) -> str:
        lo, hi = self.interval(0.95)
        lines = [f"UQResult({self.method}, n_samples={self.n_samples})"]
        for j, name in enumerate(self.parameter_names):
            lines.append(
                f"  {name:<12s} {self.mean[j]:12.6g} +- {self.std[j]:.4g}"
                f"   [{lo[j]:.6g}, {hi[j]:.6g}]"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"UQResult(method={self.method!r}, n_parameters={self.n_parameters}, "
            f"n_samples={self.n_samples})"
        )


# ----------------------------------------------------------------------
def _as_square(value: Any, n: int, what: str) -> np.ndarray:
    """Coerce a scalar / vector / matrix into an ``(n, n)`` matrix."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr) * np.eye(n)
    if arr.ndim == 1:
        if arr.size != n:
            raise ValueError(f"{what} has {arr.size} entries but {n} were expected")
        return np.diag(arr)
    if arr.shape != (n, n):
        raise ValueError(f"{what} must be {(n, n)}, got {arr.shape}")
    return arr


def _inverse(A: np.ndarray, rcond: float) -> np.ndarray:
    """Inverse of a (possibly ill-conditioned) symmetric matrix."""
    A = 0.5 * (A + A.T)
    try:
        cond = np.linalg.cond(A)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        cond = math.inf
    if not math.isfinite(cond) or cond > 1.0 / max(rcond, 1e-300):
        return np.linalg.pinv(A, rcond=rcond)
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:  # pragma: no cover - cond already screened
        return np.linalg.pinv(A, rcond=rcond)


def _jacobian_of(source: Any) -> tuple[np.ndarray, np.ndarray | None, Any, list[str]]:
    """``(J, residual, mean, names)`` from a matrix, sensitivity or update result."""
    residual = getattr(source, "residual", None)
    mean = getattr(source, "x", None)
    names = list(getattr(source, "parameter_names", []) or [])

    sens = getattr(source, "sensitivity", None)
    if sens is not None:  # an UpdateResult
        if not names:
            names = list(getattr(sens, "parameter_names", []) or [])
        source = sens
    if mean is None:
        mean = getattr(source, "p0", None)
    if not names:
        names = list(getattr(source, "parameter_names", []) or [])
    J = np.atleast_2d(np.asarray(source, dtype=float))
    res = None if residual is None else np.asarray(residual, dtype=float).ravel()
    return J, res, mean, names


def parameter_covariance(
    jacobian: Any,
    residual_cov: Any = None,
    *,
    weights: Any = None,
    residual: ArrayLike | None = None,
    mean: ArrayLike | None = None,
    prior_cov: Any = None,
    regularization: float = 0.0,
    rcond: float = 1.0e-12,
    parameter_names: Sequence[str] | None = None,
) -> UQResult:
    """First-order parameter covariance of an updating problem.

    Parameters
    ----------
    jacobian:
        ``(n_response, n_parameter)`` updating Jacobian ``J = dr/dtheta``.  A
        :class:`femtools.updating.SensitivityResult` or a complete
        :class:`femtools.updating.UpdateResult` may be passed instead, in which
        case the residual, the converged parameters and the parameter names are
        taken from it.
    residual_cov:
        Measurement (residual) covariance ``C_r``: a full matrix, a vector of
        per-response variances, or a scalar variance.  ``None`` estimates the
        scale from the residual instead.
    weights:
        Weight matrix ``W`` used by the estimator (matrix, per-response vector
        or scalar).  Defaults to ``C_r^{-1}`` when ``residual_cov`` is given --
        the minimum-variance choice, for which the sandwich collapses to
        ``(J^T C_r^{-1} J)^{-1}`` -- and to the identity otherwise.
    residual:
        Final residual ``r_m - r(theta)``; only used to estimate the noise
        variance when ``residual_cov`` is ``None``.
    prior_cov:
        Prior parameter covariance ``C_theta``.  Supplying it adds ``C_theta^{-1}``
        to the information matrix, i.e. returns the *posterior* covariance of the
        Bayesian estimator rather than the least-squares one.
    regularization:
        Tikhonov term added to the information matrix (use the same value as the
        updating run when the problem was regularised).

    Returns
    -------
    UQResult
        ``covariance`` plus ``extras`` with the information matrix, the residual
        variance estimate ``sigma2``, the degrees of freedom and the condition
        number of ``J^T W J``.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.updating.uq import parameter_covariance
    >>> J = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    >>> uq = parameter_covariance(J, residual_cov=0.01)
    >>> np.round(uq.std, 4)
    array([0.0745, 0.0471])
    """
    J, res_from_source, mean_from_source, names = _jacobian_of(jacobian)
    if J.ndim != 2:  # pragma: no cover - atleast_2d guarantees 2-D
        raise ValueError(f"jacobian must be 2-D, got shape {J.shape}")
    n_r, n_p = J.shape

    r = None
    if residual is not None:
        r = np.atleast_1d(np.asarray(residual, dtype=float)).ravel()
    elif res_from_source is not None:
        r = res_from_source
    if r is not None and r.size != n_r:
        raise ValueError(
            f"residual has {r.size} entries but the Jacobian has {n_r} rows"
        )

    Cr = None if residual_cov is None else _as_square(residual_cov, n_r, "residual_cov")
    if weights is not None:
        W = _as_square(weights, n_r, "weights")
        sandwich = Cr is not None
    elif Cr is not None:
        W = _inverse(Cr, rcond)
        sandwich = False  # minimum-variance weighting: the sandwich collapses
    else:
        W = np.eye(n_r)
        sandwich = False

    # Residual scale.  With a measurement covariance the absolute scale is
    # already set; otherwise the weighted residual provides the classic
    # unbiased estimate sigma^2 = r^T W r / (n_r - n_theta).
    dof = max(n_r - n_p, 0)
    sigma2 = 1.0
    if Cr is None:
        if r is not None and dof > 0:
            sigma2 = float(r @ W @ r) / dof
        elif r is not None:  # pragma: no cover - saturated fit, no spare data
            sigma2 = 0.0

    info = J.T @ W @ J
    if prior_cov is not None:
        info = info + _inverse(_as_square(prior_cov, n_p, "prior_cov"), rcond)
    if regularization:
        info = info + float(regularization) * np.eye(n_p)

    info_inv = _inverse(info, rcond)
    if sandwich:
        cov = info_inv @ (J.T @ W @ Cr @ W @ J) @ info_inv
    else:
        cov = sigma2 * info_inv
    cov = 0.5 * (cov + cov.T)

    if mean is not None:
        theta = np.atleast_1d(np.asarray(mean, dtype=float)).ravel()
    elif mean_from_source is not None:
        theta = np.atleast_1d(np.asarray(mean_from_source, dtype=float)).ravel()
    else:
        theta = np.zeros(n_p)
    if theta.size != n_p:
        raise ValueError(f"mean has {theta.size} entries but {n_p} parameters were given")

    if parameter_names is not None:
        names = [str(s) for s in parameter_names]
    if names and len(names) != n_p:
        names = []

    s = np.linalg.svd(J, compute_uv=False)
    return UQResult(
        mean=theta,
        covariance=cov,
        samples=None,
        parameter_names=names,
        method="first-order (sandwich)" if sandwich else "first-order",
        extras={
            "information": info,
            "sigma2": float(sigma2),
            "dof": int(dof),
            "n_responses": int(n_r),
            "jacobian_condition": float(s[0] / s[-1]) if s.size and s[-1] > 0 else math.inf,
        },
    )


# ----------------------------------------------------------------------
_PERTURB_TARGETS = ("targets", "target", "measurement", "noise", "gaussian")
_PERTURB_RESIDUAL = ("residual", "residuals", "bootstrap")
_PERTURB_START = ("parameters", "parameter", "start", "start-point", "initial")


def _noise_sampler(
    rng: np.random.Generator,
    n_r: int,
    residual_cov: Any,
    noise_std: Any,
    targets: np.ndarray,
    relative: bool,
    residual: np.ndarray | None,
):
    """Return ``(draw() -> ndarray, description)`` for the measurement noise."""
    if residual_cov is not None:
        C = _as_square(residual_cov, n_r, "residual_cov")
        # A symmetric square root works for singular (rank deficient) C too,
        # where a Cholesky factorisation would fail.
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        L = V * np.sqrt(np.clip(w, 0.0, None))[None, :]
        return (lambda: L @ rng.standard_normal(n_r)), "residual_cov"

    if noise_std is None:
        if residual is not None and residual.size:
            sd = float(np.sqrt(np.mean(residual**2)))
        else:  # pragma: no cover - update_model always returns a residual
            sd = 0.0
        if sd <= 0.0:
            # A perfect fit carries no information about the measurement noise;
            # fall back to a 1 % relative error rather than sampling nothing.
            std = 0.01 * np.abs(targets)
            return (lambda: std * rng.standard_normal(n_r)), "1% relative (default)"
        return (lambda: sd * rng.standard_normal(n_r)), f"residual rms {sd:.4g}"

    sd_arr = np.asarray(noise_std, dtype=float)
    if sd_arr.ndim == 0:
        sd_arr = np.full(n_r, float(sd_arr))
    elif sd_arr.size != n_r:
        raise ValueError(f"noise_std has {sd_arr.size} entries but {n_r} targets were given")
    if relative:
        sd_arr = sd_arr * np.abs(targets)
    return (lambda: sd_arr * rng.standard_normal(n_r)), "noise_std"


def monte_carlo_update(
    model: Any,
    parameters: Any = None,
    targets: Any = None,
    n: int = 100,
    *,
    seed: int,
    noise_std: Any = None,
    residual_cov: Any = None,
    relative: bool = False,
    perturb: str = "targets",
    start_scatter: float = 0.1,
    warm_start: bool = True,
    keep_samples: bool = True,
    on_failure: str = "skip",
    **update_kwargs: Any,
) -> UQResult:
    """Monte Carlo model updating: re-identify ``n`` times on perturbed data.

    A nominal :func:`femtools.updating.update_model` run fixes the reference
    solution; ``n`` further runs are then driven by resampled data and their
    empirical mean and covariance are returned.  This is the non-linear
    counterpart of :func:`parameter_covariance`: it needs no linearity
    assumption and it exposes an ill-posed problem as a wide, strongly
    correlated sample cloud.

    Parameters
    ----------
    model, parameters, targets:
        Exactly as for :func:`femtools.updating.update_model`; extra keyword
        arguments are forwarded unchanged, so weights, bounds, method and
        tolerances behave identically in every sample.
    n:
        Number of Monte Carlo samples.
    seed:
        **Required** seed of the ``numpy`` generator, so a reported covariance
        can always be reproduced.
    noise_std, residual_cov, relative:
        Measurement noise model.  ``residual_cov`` (matrix / vector / scalar
        variance) wins over ``noise_std`` (standard deviation, made relative to
        each target with ``relative=True``).  With neither, the RMS of the
        nominal residual is used, which treats the misfit that updating could
        not remove as measurement noise.
    perturb:
        ``"targets"`` (default) adds Gaussian measurement noise to the measured
        responses; ``"residual"`` runs a residual bootstrap (the nominal
        residual entries are resampled with replacement and added to the fitted
        response); ``"parameters"`` keeps the data and instead scatters the
        starting point by ``start_scatter`` (relative), which probes uniqueness
        rather than noise propagation.
    warm_start:
        Start every sample from the nominal solution (default).  ``False``
        restarts from the original ``p0``, which is slower but independent.
    on_failure:
        ``"skip"`` (default) drops samples whose update raised, ``"raise"``
        propagates the exception.

    Returns
    -------
    UQResult
        ``mean`` / ``covariance`` are the sample statistics, ``samples`` is the
        ``(n_ok, n_parameter)`` cloud, and ``extras`` carries the nominal
        :class:`~femtools.updating.UpdateResult`, the number of failures and
        the noise description.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.updating.uq import monte_carlo_update
    >>> A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    >>> uq = monte_carlo_update(lambda p: A @ p, 2, A @ np.array([1.0, 2.0]),
    ...                         40, seed=7, noise_std=0.01, weights="unit",
    ...                         p0=[0.5, 0.5])
    >>> bool(np.all(np.abs(uq.mean - [1.0, 2.0]) < 0.02))
    True
    """
    from .updater import update_model

    if seed is None:
        raise ValueError("`seed` is required so that Monte Carlo results are reproducible")
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    mode = str(perturb).strip().lower().replace("_", "-")
    if mode in _PERTURB_TARGETS:
        mode = "targets"
    elif mode in _PERTURB_RESIDUAL:
        mode = "residual"
    elif mode in _PERTURB_START:
        mode = "parameters"
    else:
        raise ValueError(
            f"unknown perturb={perturb!r}; expected 'targets', 'residual' or 'parameters'"
        )
    fail = str(on_failure).strip().lower()
    if fail not in ("skip", "raise"):
        raise ValueError(f"unknown on_failure={on_failure!r}; expected 'skip' or 'raise'")

    p0 = update_kwargs.pop("p0", None)
    base = update_model(model, parameters, targets, p0=p0, **update_kwargs)
    t = np.asarray(base.targets, dtype=float).ravel()
    n_r = t.size
    x_nom = np.asarray(base.x, dtype=float).ravel()
    n_p = x_nom.size
    start = x_nom.copy() if warm_start else (None if p0 is None else np.asarray(p0, float))

    rng = np.random.default_rng(int(seed))
    draw, noise_kind = _noise_sampler(
        rng, n_r, residual_cov, noise_std, t, relative, np.asarray(base.residual, dtype=float)
    )

    r_fit = np.asarray(base.response, dtype=float).ravel()
    r_res = np.asarray(base.residual, dtype=float).ravel()
    scatter = float(start_scatter)

    samples = np.empty((n, n_p))
    ok = np.zeros(n, dtype=bool)
    n_iter = np.zeros(n, dtype=int)
    for k in range(n):
        if mode == "targets":
            t_k, s_k = t + draw(), start
        elif mode == "residual":
            t_k = r_fit + r_res[rng.integers(0, n_r, n_r)]
            s_k = start
        else:
            t_k = t
            base_start = x_nom if start is None else np.asarray(start, dtype=float)
            step = np.where(np.abs(base_start) > 0, np.abs(base_start), 1.0)
            s_k = base_start + scatter * step * rng.standard_normal(n_p)
        try:
            res = update_model(model, parameters, t_k, p0=s_k, **update_kwargs)
        except Exception:
            if fail == "raise":
                raise
            continue
        samples[k] = np.asarray(res.x, dtype=float).ravel()
        n_iter[k] = int(res.n_iter)
        ok[k] = True

    good = samples[ok]
    if good.shape[0] < 2:
        raise RuntimeError(
            f"only {good.shape[0]} of {n} Monte Carlo updates succeeded; "
            "the covariance needs at least two samples"
        )
    mean = good.mean(axis=0)
    cov = np.atleast_2d(np.cov(good, rowvar=False, ddof=1))

    return UQResult(
        mean=mean,
        covariance=cov,
        samples=good if keep_samples else None,
        parameter_names=list(base.parameter_names),
        method=f"monte-carlo({mode})",
        n_samples=int(good.shape[0]),
        extras={
            "nominal": base,
            "nominal_x": x_nom,
            "n_requested": n,
            "n_failed": int(n - good.shape[0]),
            "seed": int(seed),
            "noise": noise_kind,
            "mean_iterations": float(np.mean(n_iter[ok])),
        },
    )
