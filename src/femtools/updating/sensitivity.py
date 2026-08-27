"""Response sensitivity matrices (Friswell & Mottershead, ch. 3).

The sensitivity matrix collects :math:`S_{ij} = \\partial r_i / \\partial p_j`
for responses :math:`r` (frequencies, ``1-MAC``, FRF samples, ...) and updating
parameters :math:`p` (E, rho, thickness, spring stiffness, ...).

Three routes are provided:

``method="forward" | "central" | "complex"``
    Finite (or complex-step) differences of an arbitrary response callback.
    Works with *any* solver, including callbacks, and is the default.

``method="analytic"``
    Closed form eigenvalue derivatives
    :math:`\\partial\\lambda_i/\\partial p_j = \\phi_i^T (K_{,j} - \\lambda_i M_{,j}) \\phi_i`
    for mass-normalised modes, converted to Hz.  Requires the matrix
    derivatives (supplied explicitly, or provided by
    :mod:`femtools.updating.reference` models).

``method="semi-analytic"``
    Same eigenvalue formula but with :math:`K_{,j}, M_{,j}` obtained by
    differencing the *assembled matrices*.  For parameters that enter linearly
    (Young's modulus, density, spring stiffness) this is exact to round-off
    while costing only one extra assembly per parameter — no extra eigensolve.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .parameters import ParameterSet, apply_parameters, as_parameters, snapshot_baseline
from .responses import ResponseSpec, modal_response_function

__all__ = [
    "SensitivityResult",
    "sensitivity_matrix",
    "finite_difference_jacobian",
    "analytic_frequency_sensitivity",
    "eigenvector_sensitivity",
    "relative_sensitivity",
]

_FD_METHODS = {"forward", "backward", "central", "complex", "fd", "cs"}


@dataclass
class SensitivityResult:
    """``(n_response, n_parameter)`` sensitivity matrix plus provenance.

    The object is array-like: ``np.asarray(S)``, ``S.shape``, ``S[i, j]``,
    ``S @ x`` and ``S.T`` all behave like the underlying ndarray, so it can be
    used interchangeably with a plain matrix.
    """

    matrix: np.ndarray
    p0: np.ndarray
    r0: np.ndarray
    parameter_names: list[str] = field(default_factory=list)
    response_names: list[str] = field(default_factory=list)
    method: str = "central"
    step: np.ndarray | float = 1.0e-4
    normalization: str | None = None
    n_evaluations: int = 0

    def __post_init__(self) -> None:
        self.matrix = np.atleast_2d(np.asarray(self.matrix, dtype=float))
        self.p0 = np.atleast_1d(np.asarray(self.p0, dtype=float))
        self.r0 = np.atleast_1d(np.asarray(self.r0, dtype=float))
        if not self.parameter_names:
            self.parameter_names = [f"p{j + 1}" for j in range(self.matrix.shape[1])]
        if not self.response_names:
            self.response_names = [f"r{i + 1}" for i in range(self.matrix.shape[0])]

    # -- array protocol -------------------------------------------------
    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = self.matrix if dtype is None else self.matrix.astype(dtype)
        return arr

    def __getitem__(self, key: Any) -> Any:
        return self.matrix[key]

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def __matmul__(self, other: Any) -> Any:
        return self.matrix @ other

    def __rmatmul__(self, other: Any) -> Any:
        return other @ self.matrix

    @property
    def shape(self) -> tuple[int, ...]:
        return self.matrix.shape

    @property
    def ndim(self) -> int:
        return self.matrix.ndim

    @property
    def T(self) -> np.ndarray:
        return self.matrix.T

    # -- aliases --------------------------------------------------------
    @property
    def S(self) -> np.ndarray:
        return self.matrix

    @property
    def values(self) -> np.ndarray:
        return self.matrix

    @property
    def jacobian(self) -> np.ndarray:
        return self.matrix

    # -- analysis -------------------------------------------------------
    def normalized(self, kind: str = "relative") -> np.ndarray:
        """Return a scaled copy: ``relative`` (dlnr/dlnp), ``parameter`` or ``response``."""
        S = self.matrix
        kind = kind.lower()
        if kind in ("none", ""):
            return S.copy()
        if kind in ("parameter", "p", "semi"):
            return S * self.p0[None, :]
        if kind in ("response", "r"):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(self.r0[:, None] != 0, S / self.r0[:, None], 0.0)
        if kind in ("relative", "log", "loglog"):
            with np.errstate(divide="ignore", invalid="ignore"):
                out = S * self.p0[None, :]
                return np.where(self.r0[:, None] != 0, out / self.r0[:, None], 0.0)
        raise ValueError(f"unknown normalization {kind!r}")

    def condition_number(self) -> float:
        s = np.linalg.svd(self.matrix, compute_uv=False)
        s = s[s > 0]
        return float(s[0] / s[-1]) if s.size else math.inf

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            rn: {pn: float(self.matrix[i, j]) for j, pn in enumerate(self.parameter_names)}
            for i, rn in enumerate(self.response_names)
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SensitivityResult(shape={self.shape}, method={self.method!r}, "
            f"params={self.parameter_names!r})"
        )


# ----------------------------------------------------------------------
def finite_difference_jacobian(
    fun: Callable[[np.ndarray], np.ndarray],
    x0: ArrayLike,
    *,
    method: str = "central",
    step: float | Sequence[float] = 1.0e-4,
    relative: bool = True,
    f0: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Finite-difference Jacobian of ``fun`` at ``x0``.

    Returns ``(J, f0, h, n_eval)`` with ``J[i, j] = df_i/dx_j``.
    ``relative=True`` scales the step with ``|x_j|`` (falling back to the
    absolute ``step`` when ``x_j == 0``).
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = x0.size
    method = method.lower()
    h = np.asarray(step, dtype=float)
    if h.ndim == 0:
        h = np.full(n, float(h))
    if relative:
        h = np.where(np.abs(x0) > 0, h * np.abs(x0), h)
    h = np.where(h == 0.0, 1.0e-8, h)

    n_eval = 0
    if method in ("complex", "cs"):
        cols = []
        for j in range(n):
            xp = x0.astype(complex)
            xp[j] += 1j * h[j]
            fp = np.atleast_1d(np.asarray(fun(xp)))
            cols.append(np.imag(fp) / h[j])
            n_eval += 1
        if f0 is None:
            f0 = np.atleast_1d(np.asarray(fun(x0), dtype=float))
            n_eval += 1
        return np.column_stack(cols), np.asarray(f0, dtype=float), h, n_eval

    if f0 is None:
        f0 = np.atleast_1d(np.asarray(fun(x0), dtype=float))
        n_eval += 1
    f0 = np.asarray(f0, dtype=float).ravel()

    lo = hi = None
    if bounds is not None:
        lo, hi = bounds

    cols = []
    for j in range(n):
        hj = h[j]
        if method in ("central",):
            xp, xm = x0.copy(), x0.copy()
            xp[j] += hj
            xm[j] -= hj
            if lo is not None and hi is not None:
                # shrink to stay feasible, degrade to one-sided if needed
                if xp[j] > hi[j] or xm[j] < lo[j]:
                    room = min(hi[j] - x0[j], x0[j] - lo[j])
                    if room > 0:
                        hj = min(hj, room)
                        xp[j], xm[j] = x0[j] + hj, x0[j] - hj
                    else:
                        hj = min(hj, max(hi[j] - x0[j], 1e-300))
                        xp[j], xm[j] = x0[j] + hj, x0[j]
                        fp = np.atleast_1d(np.asarray(fun(xp), dtype=float)).ravel()
                        n_eval += 1
                        cols.append((fp - f0) / hj)
                        continue
            fp = np.atleast_1d(np.asarray(fun(xp), dtype=float)).ravel()
            fm = np.atleast_1d(np.asarray(fun(xm), dtype=float)).ravel()
            n_eval += 2
            cols.append((fp - fm) / (2.0 * hj))
        elif method in ("backward",):
            xm = x0.copy()
            xm[j] -= hj
            fm = np.atleast_1d(np.asarray(fun(xm), dtype=float)).ravel()
            n_eval += 1
            cols.append((f0 - fm) / hj)
        else:  # forward
            xp = x0.copy()
            xp[j] += hj
            if hi is not None and xp[j] > hi[j]:
                hj = -hj
                xp[j] = x0[j] + hj
            fp = np.atleast_1d(np.asarray(fun(xp), dtype=float)).ravel()
            n_eval += 1
            cols.append((fp - f0) / hj)
    return np.column_stack(cols), f0, h, n_eval


# ----------------------------------------------------------------------
def analytic_frequency_sensitivity(
    freq_hz: ArrayLike,
    modes: np.ndarray,
    dK: Sequence[Any],
    dM: Sequence[Any] | None = None,
    *,
    output: str = "hz",
) -> np.ndarray:
    """Exact modal sensitivities for mass-normalised modes.

    .. math::
        \\frac{\\partial\\lambda_i}{\\partial p_j}
          = \\phi_i^T\\left(\\frac{\\partial K}{\\partial p_j}
            - \\lambda_i \\frac{\\partial M}{\\partial p_j}\\right)\\phi_i

    Parameters
    ----------
    freq_hz:
        Natural frequencies [Hz] of the modes in ``modes``.
    modes:
        ``(ndof, n_modes)`` mass-normalised mode shapes (``Phi^T M Phi = I``).
    dK, dM:
        Sequences (one entry per parameter) of stiffness/mass derivative
        matrices; dense or sparse.  ``dM=None`` means mass-independent.
    output:
        ``"hz"`` -> df/dp,  ``"lambda"`` -> dlambda/dp,  ``"rad"`` -> domega/dp.

    Notes
    -----
    For a global Young's modulus multiplier ``p`` (``K = p K_0``) this reduces to
    ``dlambda/dp = lambda/p`` and therefore ``df/dp = f/(2p)`` exactly.
    """
    freq = np.asarray(freq_hz, dtype=float).ravel()
    phi = np.asarray(modes)
    n_modes = min(freq.size, phi.shape[1])
    lam = (2.0 * math.pi * freq[:n_modes]) ** 2
    npar = len(dK)
    S = np.zeros((n_modes, npar))
    for j in range(npar):
        Kj = dK[j]
        Mj = None if dM is None else dM[j]
        KP = np.asarray(Kj @ phi[:, :n_modes])
        MP = None if Mj is None else np.asarray(Mj @ phi[:, :n_modes])
        for i in range(n_modes):
            num = float(np.real(phi[:, i].conj() @ KP[:, i]))
            if MP is not None:
                num -= lam[i] * float(np.real(phi[:, i].conj() @ MP[:, i]))
            S[i, j] = num
    if output.lower() in ("lambda", "eigenvalue", "lam"):
        return S
    if output.lower() in ("rad", "omega", "rad/s"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(lam[:, None] > 0, S / (2.0 * np.sqrt(lam)[:, None]), 0.0)
    # Hz:  f = sqrt(lam)/(2 pi)  ->  df/dp = dlam/dp / (8 pi^2 f)
    denom = 8.0 * math.pi**2 * freq[:n_modes]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom[:, None] > 0, S / denom[:, None], 0.0)


def eigenvector_sensitivity(
    freq_hz: ArrayLike,
    modes: np.ndarray,
    dK: Sequence[Any],
    dM: Sequence[Any] | None = None,
    *,
    mode_index: int = 0,
) -> np.ndarray:
    """Fox--Kapoor modal-superposition eigenvector derivative.

    Returns ``(ndof, n_parameter)`` containing ``d(phi_i)/dp_j`` for the mode
    ``mode_index``.  Accuracy is limited by modal truncation: pass as many modes
    as available.
    """
    freq = np.asarray(freq_hz, dtype=float).ravel()
    phi = np.asarray(modes)
    lam = (2.0 * math.pi * freq) ** 2
    i = int(mode_index)
    n_modes = phi.shape[1]
    out = np.zeros((phi.shape[0], len(dK)), dtype=phi.dtype)
    for j in range(len(dK)):
        Kj, Mj = dK[j], (None if dM is None else dM[j])
        A_phi_i = np.asarray(Kj @ phi[:, i])
        if Mj is not None:
            A_phi_i = A_phi_i - lam[i] * np.asarray(Mj @ phi[:, i])
        coeffs = np.zeros(n_modes, dtype=complex)
        for k in range(n_modes):
            if k == i:
                if Mj is None:
                    coeffs[k] = 0.0
                else:
                    coeffs[k] = -0.5 * (phi[:, i].conj() @ np.asarray(Mj @ phi[:, i]))
                continue
            denom = lam[i] - lam[k]
            if abs(denom) < 1e-30:
                continue
            coeffs[k] = (phi[:, k].conj() @ A_phi_i) / denom
        out[:, j] = (phi @ coeffs).real if np.isrealobj(phi) else phi @ coeffs
    return out


def relative_sensitivity(S: Any, p0: ArrayLike, r0: ArrayLike) -> np.ndarray:
    """``dln r / dln p`` from an absolute sensitivity matrix."""
    S = np.asarray(S, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    r0 = np.asarray(r0, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = S * p0[None, :]
        return np.where(r0[:, None] != 0, out / r0[:, None], 0.0)


# ----------------------------------------------------------------------
def _matrix_derivatives(
    model: Any,
    pset: ParameterSet,
    p0: np.ndarray,
    *,
    step: float,
    assembler: Callable[..., Any] | None,
) -> tuple[list[Any], list[Any], Any, Any]:
    """Central-difference derivatives of the assembled K, M w.r.t. each parameter."""
    from .reference import ReferenceModel
    from .responses import _default_assembler

    if isinstance(model, ReferenceModel):
        # Reference models are linear in their multipliers -> exact derivatives.
        K0, M0 = model.assemble(p0)
        return list(model.stiffness_derivatives()), list(model.mass_derivatives()), K0, M0

    asm = assembler or _default_assembler()
    base = snapshot_baseline(model, pset)

    def _km(p: np.ndarray) -> tuple[Any, Any]:
        m = apply_parameters(model, pset, p, copy_model=True, baseline=base)
        res = asm(m)
        K = getattr(res, "K", None)
        M = getattr(res, "M", None)
        if K is None and isinstance(res, (tuple, list)):
            K, M = res[0], res[1]
        return K, M

    K0, M0 = _km(p0)
    dK: list[Any] = []
    dM: list[Any] = []
    for j in range(len(pset)):
        h = step * (abs(p0[j]) if p0[j] != 0 else 1.0)
        pp, pm = p0.copy(), p0.copy()
        pp[j] += h
        pm[j] -= h
        Kp, Mp = _km(pp)
        Km, Mm = _km(pm)
        dK.append((Kp - Km) * (1.0 / (2.0 * h)))
        dM.append((Mp - Mm) * (1.0 / (2.0 * h)))
    return dK, dM, K0, M0


def sensitivity_matrix(
    response: Any,
    p0: Any = None,
    *,
    parameters: Any = None,
    method: str = "central",
    step: float | Sequence[float] = 1.0e-4,
    relative_step: bool = True,
    r0: ArrayLike | None = None,
    normalize: str | None = None,
    bounds: tuple[Any, Any] | None = None,
    n_modes: int = 10,
    spec: ResponseSpec | None = None,
    solver: Callable[..., Any] | None = None,
    assembler: Callable[..., Any] | None = None,
    dK: Sequence[Any] | None = None,
    dM: Sequence[Any] | None = None,
    modes: np.ndarray | None = None,
    freq_hz: ArrayLike | None = None,
    response_names: Sequence[str] | None = None,
) -> SensitivityResult:
    """Compute :math:`\\partial r_i / \\partial p_j`.

    Parameters
    ----------
    response:
        Either a callable ``f(p) -> ndarray``, an :class:`FEModel` (in which
        case ``parameters`` is required and a modal response is built), or a
        :class:`femtools.updating.reference.ReferenceModel`.
    p0:
        Parameter values at which to linearise.  Defaults to the parameters'
        own ``value`` fields, or ``ones`` for reference models.
    parameters:
        Parameter specification (see :func:`femtools.updating.as_parameters`).
        Required for the model-based route, optional for callables (used only
        for naming / bounds).
    method:
        ``"forward"``, ``"central"`` (default), ``"backward"``, ``"complex"``,
        ``"analytic"`` or ``"semi-analytic"``.
    step:
        Perturbation size.  With ``relative_step=True`` (default) it is a
        fraction of ``|p_j|``.
    normalize:
        Optional post-scaling: ``"relative"``, ``"parameter"`` or ``"response"``.

    Returns
    -------
    SensitivityResult
        Array-like ``(n_response, n_parameter)`` result.

    Examples
    --------
    >>> from femtools.updating.reference import BeamModel
    >>> from femtools.updating import sensitivity_matrix
    >>> beam = BeamModel(n_elem=10, n_regions=2)
    >>> S = sensitivity_matrix(beam, [1.0, 1.0], n_modes=3)
    >>> S.shape
    (3, 2)
    """
    from .reference import ReferenceModel

    method_l = str(method).lower().replace("_", "-")
    pset: ParameterSet | None = None
    if parameters is not None:
        pset = as_parameters(parameters)

    # ---- resolve p0 ---------------------------------------------------
    if p0 is None:
        if pset is not None:
            p0_arr = pset.values
        elif isinstance(response, ReferenceModel):
            p0_arr = np.ones(response.n_parameters)
        else:
            raise ValueError("p0 (or `parameters`) must be given")
    else:
        p0_arr = np.atleast_1d(np.asarray(p0, dtype=float))

    par_names = (
        list(pset.names)
        if pset is not None
        else (
            list(response.parameter_names)
            if isinstance(response, ReferenceModel)
            else [f"p{j + 1}" for j in range(p0_arr.size)]
        )
    )

    # ---- analytic / semi-analytic route -------------------------------
    if method_l in ("analytic", "semi-analytic", "semianalytic", "modal"):
        if dK is None:
            if isinstance(response, ReferenceModel):
                dK_l, dM_l, _, _ = _matrix_derivatives(
                    response, pset or as_parameters(par_names), p0_arr,
                    step=float(np.asarray(step).ravel()[0]), assembler=assembler,
                )
            else:
                if pset is None:
                    raise ValueError("analytic sensitivities need `parameters` for a model")
                dK_l, dM_l, _, _ = _matrix_derivatives(
                    response, pset, p0_arr,
                    step=float(np.asarray(step).ravel()[0]), assembler=assembler,
                )
        else:
            dK_l, dM_l = list(dK), (None if dM is None else list(dM))  # type: ignore[assignment]

        if modes is None or freq_hz is None:
            phi: np.ndarray | None
            if isinstance(response, ReferenceModel):
                f_calc, phi = response.eig(p0_arr, n_modes)
            else:
                from .responses import solve_modal

                m = apply_parameters(response, pset, p0_arr, copy_model=True)  # type: ignore[arg-type]
                f_calc, phi = solve_modal(m, n_modes, solver=solver)
            freq_used = f_calc if freq_hz is None else np.asarray(freq_hz, dtype=float)
            if modes is None and phi is None:
                raise ValueError("analytic sensitivities require mode shapes")
            modes_used = np.asarray(phi if modes is None else modes)
        else:
            freq_used = np.asarray(freq_hz, dtype=float)
            modes_used = np.asarray(modes)
        S = analytic_frequency_sensitivity(freq_used, modes_used, dK_l, dM_l)
        r0_arr = np.asarray(freq_used, dtype=float)[: S.shape[0]]
        res = SensitivityResult(
            matrix=S,
            p0=p0_arr,
            r0=r0_arr,
            parameter_names=par_names,
            response_names=(
                list(response_names)
                if response_names
                else [f"f{i + 1}" for i in range(S.shape[0])]
            ),
            method=method_l,
            step=np.asarray(step, dtype=float),
        )
        if normalize:
            res.matrix = res.normalized(normalize)
            res.normalization = normalize
        return res

    if method_l not in _FD_METHODS:
        raise ValueError(f"unknown sensitivity method {method!r}")

    # ---- build response callable --------------------------------------
    if callable(response) and not isinstance(response, ReferenceModel):
        fun = response
    elif isinstance(response, ReferenceModel):
        fun = response.response_function(n_modes)
    else:
        if pset is None:
            raise ValueError("`parameters` is required when passing an FEModel")
        fun = modal_response_function(
            response, pset, spec, solver=solver, n_modes=n_modes
        )

    bnds = None
    if bounds is not None:
        lo, hi = bounds
        bnds = (np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    elif pset is not None:
        lo, hi = pset.lower, pset.upper
        if np.any(np.isfinite(lo)) or np.any(np.isfinite(hi)):
            bnds = (lo, hi)

    J, f0, h, n_eval = finite_difference_jacobian(
        fun,
        p0_arr,
        method=method_l,
        step=step,
        relative=relative_step,
        f0=None if r0 is None else np.asarray(r0, dtype=float),
        bounds=bnds,
    )
    res = SensitivityResult(
        matrix=J,
        p0=p0_arr,
        r0=f0,
        parameter_names=par_names,
        response_names=list(response_names) if response_names else [],
        method=method_l,
        step=h,
        n_evaluations=n_eval,
    )
    if normalize:
        res.matrix = res.normalized(normalize)
        res.normalization = normalize
    return res
