"""FRF-based model updating (weighted least squares on frequency response samples).

Modal updating condenses a measurement into a handful of natural frequencies and
mode shapes, which throws away the damping, the anti-resonances and everything
between the peaks.  FRF-based updating (Lin & Ewins, 1990; Friswell &
Mottershead, ch. 7) skips the modal identification step and fits the *measured
response itself*, sample by sample:

.. math::
    \\Delta p = \\left(S^T W S + \\lambda D\\right)^{-1} S^T W
                \\left(r_m - r(p)\\right), \\qquad
    r = \\mathcal{R}\\big(H(\\omega_k; p)\\big),

where :math:`\\mathcal{R}` turns the complex FRF at the selected lines into a
real residual vector — ``log10|H|`` by default, because it is dimensionless and
gives resonances and anti-resonances comparable influence, or the stacked real
and imaginary parts when the phase must be matched too.

The price is a residual surface with one local minimum per resonance shift: a
parameter error that moves a peak by more than its own half-power bandwidth
sees the *neighbouring* peak as the closer target.  Three defences are built in
and all are on by default: the log-magnitude residual (which is far less peaky
than ``|H|``), the trust region and line search inherited from
:func:`femtools.updating.update_model`, and line selection that samples the
band uniformly rather than clustering on the peaks.

The measured FRF may be a synthesised one, an :class:`FRFResult` from
:mod:`femtools.dynamics.frf`, or a measurement from
:func:`femtools.mpe.frf_estimation.estimate_h1` — in the last case pass the
measured ``coherence`` as well and the samples are weighted by their own
estimated variance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .parameters import (
    ParameterSet,
    apply_parameters,
    as_parameters,
    snapshot_baseline,
    unwrap_model,
)
from .responses import solve_modal
from .updater import UpdateResult, update_model

__all__ = [
    "update_from_frf",
    "modal_frf_samples",
    "frf_sample_function",
    "frf_residual",
]

_RESIDUALS = {
    "log_magnitude": "log_magnitude",
    "log-magnitude": "log_magnitude",
    "logmag": "log_magnitude",
    "log": "log_magnitude",
    "log10": "log_magnitude",
    "magnitude": "magnitude",
    "mag": "magnitude",
    "abs": "magnitude",
    "real_imag": "real_imag",
    "real-imag": "real_imag",
    "complex": "real_imag",
    "reim": "real_imag",
    "real": "real",
    "imag": "imag",
    "log_magnitude_phase": "log_magnitude_phase",
    "logmag_phase": "log_magnitude_phase",
}


def _as_frf(H: Any) -> np.ndarray:
    """Coerce an FRF container into a complex ``(n_out, n_in, n_freq)`` array."""
    if not isinstance(H, np.ndarray):
        for attr in ("H", "frf", "data", "values"):
            if hasattr(H, attr):
                H = getattr(H, attr)
                break
    arr = np.asarray(H)
    if arr.dtype.kind != "c":
        arr = arr.astype(complex)
    if arr.ndim == 1:
        return arr[None, None, :]
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim != 3:
        raise ValueError(f"FRF must be (n_out, n_in, n_freq), got shape {arr.shape}")
    return arr


def frf_residual(H: Any, kind: str = "log_magnitude", *, floor: float = 1e-300) -> np.ndarray:
    """Turn a complex FRF block into the real residual vector of ``kind``.

    ``"log_magnitude"`` (default), ``"magnitude"``, ``"real_imag"``, ``"real"``,
    ``"imag"`` or ``"log_magnitude_phase"`` (unwrapped phase in radians appended
    to the log magnitude).
    """
    key = _RESIDUALS.get(str(kind).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown residual {kind!r}; expected one of {sorted(set(_RESIDUALS.values()))}"
        )
    arr = _as_frf(H)
    if key == "log_magnitude":
        return np.log10(np.maximum(np.abs(arr), floor)).ravel()
    if key == "magnitude":
        return np.abs(arr).ravel()
    if key == "real":
        return arr.real.ravel()
    if key == "imag":
        return arr.imag.ravel()
    if key == "real_imag":
        return np.concatenate([arr.real.ravel(), arr.imag.ravel()])
    # log magnitude + unwrapped phase
    phase = np.unwrap(np.angle(arr), axis=-1)
    return np.concatenate([np.log10(np.maximum(np.abs(arr), floor)).ravel(), phase.ravel()])


def _residual_multiplicity(kind: str) -> int:
    """How many residual entries each complex FRF sample produces."""
    key = _RESIDUALS[str(kind).strip().lower()]
    return 2 if key in ("real_imag", "log_magnitude_phase") else 1


# ----------------------------------------------------------------------
def _modal_damping(freq_hz: np.ndarray, damping: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-mode ``(zeta, eta)`` from a scalar, a vector, or a damping mapping."""
    f = np.asarray(freq_hz, dtype=float).ravel()
    zeros = np.zeros(f.size)
    if damping is None:
        return zeros, zeros
    if isinstance(damping, dict):
        keys = {str(k).lower(): v for k, v in damping.items()}
        if "eta" in keys:
            return zeros, np.broadcast_to(
                np.atleast_1d(np.asarray(keys["eta"], dtype=float)), f.shape
            ).astype(float)
        if "zeta" in keys:
            return (
                np.broadcast_to(
                    np.atleast_1d(np.asarray(keys["zeta"], dtype=float)), f.shape
                ).astype(float),
                zeros,
            )
        alpha = float(keys.get("alpha", 0.0))
        beta = float(keys.get("beta", 0.0))
        w = 2.0 * math.pi * f
        with np.errstate(divide="ignore", invalid="ignore"):
            zeta = np.where(w > 0, 0.5 * (alpha / np.where(w > 0, w, 1.0) + beta * w), 0.0)
        return zeta, zeros
    arr = np.atleast_1d(np.asarray(damping, dtype=float))
    if arr.size == 1:
        return np.full(f.size, float(arr[0])), zeros
    if arr.size < f.size:
        out = np.zeros(f.size)
        out[: arr.size] = arr
        return out, zeros
    return arr[: f.size].astype(float), zeros


def modal_frf_samples(
    freq_modal: ArrayLike,
    modes: np.ndarray,
    freq_hz: ArrayLike,
    outputs: Sequence[int],
    inputs: Sequence[int],
    *,
    damping: Any = 0.02,
    response: str = "receptance",
) -> np.ndarray:
    """Modal-superposition FRF block ``(n_out, n_in, n_freq)``.

    .. math::
        H_{oi}(\\omega) = \\sum_r \\frac{\\phi_{or}\\phi_{ir}}
            {\\omega_r^2(1 + i\\eta_r) - \\omega^2 + 2 i \\zeta_r \\omega_r \\omega}

    ``modes`` must be mass normalised (``Phi^T M Phi = I``), which is what
    :func:`femtools.fea.eigen.solve_modes` and the reference models return.
    ``outputs`` / ``inputs`` index the *rows* of ``modes``, i.e. the free-DOF
    ordering of the solved model.
    """
    fr = np.asarray(freq_modal, dtype=float).ravel()
    phi = np.asarray(modes)
    f = np.asarray(freq_hz, dtype=float).ravel()
    out_idx = np.asarray(outputs, dtype=int)
    in_idx = np.asarray(inputs, dtype=int)
    zeta, eta = _modal_damping(fr, damping)

    w = 2.0 * math.pi * f
    wr = 2.0 * math.pi * fr
    denom = (
        (wr**2)[:, None] * (1.0 + 1j * eta[:, None])
        - (w**2)[None, :]
        + 2j * zeta[:, None] * wr[:, None] * w[None, :]
    )
    denom = np.where(denom == 0, np.finfo(float).tiny, denom)
    H = np.einsum(
        "or,ir,rf->oif", phi[out_idx, :], phi[in_idx, :], 1.0 / denom, optimize=True
    )
    kind = str(response).strip().lower()
    if kind in ("mobility", "velocity"):
        H = H * (1j * w)[None, None, :]
    elif kind in ("accelerance", "inertance", "acceleration"):
        H = H * (-(w**2))[None, None, :]
    elif kind not in ("receptance", "compliance", "displacement"):
        raise ValueError(f"unknown response type {response!r}")
    return np.ascontiguousarray(H, dtype=complex)


def frf_sample_function(
    model: Any,
    parameters: Any,
    freq_hz: ArrayLike,
    outputs: Sequence[int],
    inputs: Sequence[int],
    *,
    damping: Any = 0.02,
    n_modes: int = 20,
    response: str = "receptance",
    solver: Callable[..., Any] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build ``p -> H(p)`` returning a complex ``(n_out, n_in, n_freq)`` block.

    Works with an :class:`femtools.core.model.FEModel` (through
    :func:`femtools.fea.eigen.solve_modes` or an explicit ``solver``) and with
    the analytical :mod:`femtools.updating.reference` models.
    """
    from .reference import ReferenceModel

    f = np.asarray(freq_hz, dtype=float).ravel()

    if isinstance(model, ReferenceModel):

        def _f(p: np.ndarray) -> np.ndarray:
            fr, phi = model.eig(p, n_modes)
            return modal_frf_samples(
                fr, phi, f, outputs, inputs, damping=damping, response=response
            )

        return _f

    fea_model = unwrap_model(model)
    pset = as_parameters(parameters)
    base = snapshot_baseline(fea_model, pset)

    def _g(p: np.ndarray) -> np.ndarray:
        m = apply_parameters(fea_model, pset, p, copy_model=True, baseline=base)
        fr, phi = solve_modal(m, n_modes, solver=solver)
        if phi is None:
            raise RuntimeError("the modal solver returned no mode shapes")
        return modal_frf_samples(
            fr, phi, f, outputs, inputs, damping=damping, response=response
        )

    return _g


# ----------------------------------------------------------------------
def _select_lines(
    f: np.ndarray,
    band: tuple[float, float] | None,
    lines: ArrayLike | None,
    n_lines: int | None,
) -> np.ndarray:
    """Indices of the spectral lines that enter the residual."""
    if lines is not None:
        idx = np.atleast_1d(np.asarray(lines, dtype=int)).ravel()
        if np.any((idx < 0) | (idx >= f.size)):
            raise ValueError("`lines` contains an index outside the frequency axis")
        return idx
    mask = np.ones(f.size, dtype=bool)
    if band is not None:
        mask &= (f >= band[0]) & (f <= band[1])
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise ValueError(f"band {band} selects no spectral line")
    if n_lines is not None and 0 < int(n_lines) < idx.size:
        pick = np.linspace(0, idx.size - 1, int(n_lines))
        idx = idx[np.unique(np.round(pick).astype(int))]
    return idx


def _coherence_weights(
    coh: Any, shape: tuple[int, int, int], lines: np.ndarray, multiplicity: int
) -> np.ndarray:
    """Inverse-variance sample weights from a measured coherence array.

    The random error of an ``H1`` estimate scales as
    :math:`(1-\\gamma^2)/(2\\gamma^2 n_{avg})`, so the inverse-variance weight of
    a sample is proportional to :math:`\\gamma^2/(1-\\gamma^2)`.  The result is
    normalised to a mean of one, which keeps the overall residual scale (and
    therefore the Levenberg--Marquardt damping) comparable to the unweighted
    problem.
    """
    g2 = np.asarray(coh, dtype=float)
    n_out, n_in, n_freq = shape
    if g2.ndim == 1:
        g2 = np.broadcast_to(g2[None, None, :], (n_out, n_in, g2.size))
    elif g2.ndim == 2:
        g2 = np.broadcast_to(g2[:, None, :], (g2.shape[0], n_in, g2.shape[1]))
    if g2.shape[2] < int(np.max(lines)) + 1:
        raise ValueError("the coherence array is shorter than the frequency axis")
    sub = np.clip(g2[:, :, lines], 0.0, 1.0)
    w = sub / np.maximum(1.0 - sub, 1.0e-6)
    w = np.broadcast_to(w, (n_out, n_in, lines.size)).ravel()
    if multiplicity > 1:
        w = np.tile(w, multiplicity)
    mean = float(np.mean(w))
    return w / mean if mean > 0 else np.ones_like(w)


def _default_weights(kind: str, H_sel: np.ndarray) -> Any:
    """Weighting used when the caller gives neither ``weights`` nor ``coherence``.

    The modal default (``1/r^2``) is wrong for every residual that can pass
    through zero: the real and imaginary parts do so at every resonance, and
    weighting them by ``1/r^2`` hands the fit to whichever sample happens to
    sit closest to a zero crossing.  They are instead scaled by the *magnitude*
    of the measured sample, which makes the complex residual relative — the FRF
    of a lightly damped structure spans several decades, and unit weights would
    reduce the fit to the two tallest resonances.  ``log10|H|`` is already
    relative and dimensionless, so it is weighted uniformly.
    """
    if kind in ("log_magnitude", "log_magnitude_phase"):
        return "unit"
    if kind == "magnitude":
        return None  # update_model's "relative" default, i.e. 1/|H|^2
    mag = np.abs(H_sel).ravel()
    w = 1.0 / np.maximum(mag, np.max(mag) * 1e-12 if mag.size else 1.0) ** 2
    if kind == "real_imag":
        w = np.tile(w, 2)
    mean = float(np.mean(w))
    return w / mean if mean > 0 else "unit"


def update_from_frf(
    model: Any,
    parameters: Any = None,
    measured: Any = None,
    freq_hz: ArrayLike | None = None,
    *,
    outputs: Sequence[int] | None = None,
    inputs: Sequence[int] | None = None,
    frf: Callable[[np.ndarray], Any] | None = None,
    damping: Any = 0.02,
    n_modes: int = 20,
    response: str = "receptance",
    residual: str = "log_magnitude",
    band: tuple[float, float] | None = None,
    lines: ArrayLike | None = None,
    n_lines: int | None = None,
    coherence: Any = None,
    weights: Any = None,
    solver: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> UpdateResult:
    """Update ``model`` so that its FRF matches a measured one.

    Parameters
    ----------
    model:
        :class:`femtools.core.model.FEModel`, a
        :class:`femtools.updating.reference.ReferenceModel`, or ``None`` when
        an explicit ``frf`` callback is supplied.
    parameters:
        Parameter specification, see :func:`femtools.updating.as_parameters`.
    measured:
        Measured FRF: a complex ``(n_out, n_in, n_freq)`` array (1-D and 2-D
        are broadcast), or any object exposing ``.H`` / ``.frf`` — an
        :class:`femtools.dynamics.frf.FRFResult` or an
        :class:`femtools.mpe.frf_estimation.FRFEstimate`, for instance.
    freq_hz:
        Frequency axis of ``measured``; read from the object when omitted.
    outputs, inputs:
        Response / excitation DOF indices into the mode-shape rows.  Defaults
        are ``range(n_out)`` and ``range(n_in)`` of the measured block, i.e.
        the measurement is assumed to sit on the first DOF — pass them
        explicitly for anything else.
    frf:
        Explicit ``p -> H`` callback; overrides the model-driven route (use it
        for a solver this package cannot drive itself).
    damping:
        Modal damping used to *predict* the FRF: a scalar ``zeta``, one value
        per mode, ``{"alpha": ..., "beta": ...}`` (Rayleigh) or ``{"eta": ...}``
        (structural).  Damping is not identified here; a wrong damping value
        mostly biases the peak amplitudes, which is exactly what the
        log-magnitude residual de-emphasises.
    n_modes:
        Modes retained in the FRF synthesis.  Use clearly more than the band
        contains: the residual flexibility of the truncated modes shifts
        anti-resonances, and an anti-resonance mismatch is indistinguishable
        from a parameter error.
    residual:
        ``"log_magnitude"`` (default), ``"magnitude"``, ``"real_imag"``,
        ``"real"``, ``"imag"`` or ``"log_magnitude_phase"``.
    band, lines, n_lines:
        Line selection: a frequency band, explicit line indices, or a uniform
        subsample of ``n_lines`` lines.  Fewer, well spread lines usually beat
        every line of a narrow band.
    coherence:
        Measured coherence, shaped like ``measured`` or ``(n_freq,)``.  Turns
        into inverse-variance sample weights ``gamma^2/(1-gamma^2)``.
    weights:
        Explicit weights, overriding ``coherence``; anything
        :func:`femtools.updating.update_model` accepts.
    **kwargs:
        Passed through to :func:`femtools.updating.update_model`
        (``method``, ``bounds``, ``p0``, ``max_iter``, ``tol``, ``step``,
        ``regularization``, ``prior``, ``verbose``, ...).

    Returns
    -------
    UpdateResult
        With ``history[0]["frf"]`` carrying the line indices, the residual kind
        and the predicted FRF before and after updating.

    Notes
    -----
    Two failure modes are worth knowing about, because both look like a
    converged solve:

    * ``residual="real_imag"`` needs the resonances to be *resolved*.  Where a
      lightly damped peak spans fewer than a couple of lines, a small parameter
      change moves the complex FRF by more than its own value and the first
      trust-region step is rejected.  ``"log_magnitude"`` (the default) is
      insensitive to this and is the residual to reach for first.
    * With **absolute** measurement noise (a noise floor rather than a constant
      percentage), the lines around the anti-resonances carry no information at
      all, yet an unweighted log-magnitude fit gives them the same say as the
      resonances.  Passing the measured ``coherence`` fixes this outright.

    Examples
    --------
    Recover a 10 % stiffness error from FRF samples alone::

        import numpy as np
        from femtools.updating.reference import BeamModel
        from femtools.updating.frf_updating import frf_sample_function, update_from_frf

        beam = BeamModel(n_elem=10, n_regions=2)
        f = np.linspace(5.0, 400.0, 400)
        truth = frf_sample_function(beam, None, f, [8], [8])(np.array([1.10, 1.0]))
        res = update_from_frf(beam, ["E1", "E2"], truth, f, outputs=[8], inputs=[8],
                              p0=[1.0, 1.0], bounds=(0.5, 1.5))
        assert abs(res.x[0] - 1.10) < 0.002
    """
    if measured is None:
        raise ValueError("`measured` (the measured FRF) must be given")
    if freq_hz is None:
        freq_hz = getattr(measured, "freq_hz", None)
        if freq_hz is None:
            freq_hz = getattr(measured, "frequencies", None)
    if freq_hz is None:
        raise ValueError("freq_hz must be given (or carried by the measured FRF object)")

    H_meas = _as_frf(measured)
    f_axis = np.asarray(freq_hz, dtype=float).ravel()
    if f_axis.size != H_meas.shape[2]:
        raise ValueError(
            f"freq_hz has {f_axis.size} lines but the measured FRF has {H_meas.shape[2]}"
        )
    n_out, n_in, _ = H_meas.shape

    idx = _select_lines(f_axis, band, lines, n_lines)
    f_sel = f_axis[idx]
    H_sel = H_meas[:, :, idx]

    out_idx = list(range(n_out)) if outputs is None else list(np.asarray(outputs, dtype=int))
    in_idx = list(range(n_in)) if inputs is None else list(np.asarray(inputs, dtype=int))
    if len(out_idx) != n_out or len(in_idx) != n_in:
        raise ValueError(
            f"the measured FRF is ({n_out}, {n_in}) but {len(out_idx)} outputs and "
            f"{len(in_idx)} inputs were given"
        )

    kind = _RESIDUALS.get(str(residual).strip().lower())
    if kind is None:
        raise ValueError(
            f"unknown residual {residual!r}; "
            f"expected one of {sorted(set(_RESIDUALS.values()))}"
        )
    targets = frf_residual(H_sel, kind)

    # ---- prediction ----------------------------------------------------
    if frf is not None:
        predict = frf
    elif model is None:
        raise ValueError("either `model` or an explicit `frf` callback must be given")
    else:
        predict = frf_sample_function(
            model,
            parameters,
            f_sel,
            out_idx,
            in_idx,
            damping=damping,
            n_modes=n_modes,
            response=response,
            solver=solver,
        )

    bad_method = str(kwargs.get("method", "wls")).lower().replace("_", "-")
    if bad_method in ("analytic", "semi-analytic", "semianalytic", "modal"):
        raise ValueError(
            f"method={kwargs['method']!r} differentiates the *eigenvalues* and cannot "
            "produce FRF sensitivities; use the default finite differences "
            '(method="wls" or "bayesian")'
        )

    n_expected = targets.size
    cache: dict[str, np.ndarray] = {}

    def _response(p: np.ndarray) -> np.ndarray:
        H = _as_frf(predict(p))
        if H.shape[2] != f_sel.size:
            # An explicit callback may return the full axis; take the lines back.
            if H.shape[2] == f_axis.size:
                H = H[:, :, idx]
            else:
                raise ValueError(
                    f"the FRF callback returned {H.shape[2]} lines, expected "
                    f"{f_sel.size}"
                )
        r = frf_residual(H, kind)
        if r.size != n_expected:
            raise ValueError(
                f"predicted residual has {r.size} entries but the measurement has "
                f"{n_expected}; check `outputs` / `inputs`"
            )
        cache.setdefault("initial", H)
        return r

    if weights is None and coherence is not None:
        weights = _coherence_weights(
            coherence, H_meas.shape, idx, _residual_multiplicity(kind)
        )
    if weights is None:
        weights = _default_weights(kind, H_sel)

    from .reference import ReferenceModel

    result = update_model(
        model if model is not None else _response,
        parameters,
        targets,
        response=_response,
        weights=weights,
        **kwargs,
    )
    result.history[0]["frf"] = {
        "lines": idx,
        "freq_hz": f_sel,
        "residual": kind,
        "n_samples": int(targets.size),
        "measured": H_sel,
        "initial": cache.get("initial"),
        "updated": _as_frf(predict(result.x)),
    }

    # `update_model` only writes the parameters back into the model it drives
    # itself; here it is driven by our response callback, so do it explicitly.
    if result.model is None and model is not None and not isinstance(model, ReferenceModel):
        if not callable(model):
            fea_model = unwrap_model(model)
            pset: ParameterSet = as_parameters(parameters)
            base = snapshot_baseline(fea_model, pset)
            result.model = apply_parameters(
                fea_model, pset, result.x, copy_model=True, baseline=base
            )
    return result
