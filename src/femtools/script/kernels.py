"""Adaptive adapters over the lazily bound Round-10 analysis kernels.

The CLI (``femtools.cli``), the FSL script engine
(``femtools.script.engine``) and the GUI (``femtools.gui.state``) all
expose the same Round-10 kernels -- ZZ superconvergent patch recovery,
the Eigensystem Realization Algorithm and the SEREP-expanded MAC.  The
kernels themselves land independently, so each adapter here

* imports its kernel lazily and lets :class:`ImportError` escape --
  every surface turns that into its own "not in this installation"
  failure (CLI exit code 3, ``ScriptError``, HTTP 400) instead of a
  traceback;
* calls the kernel adaptively, probing the signature where the exact
  argument order is a contract of the kernel's own round rather than of
  this package (the same convention as the CLI's spectral-estimator
  shim);
* normalizes the returned object into the small duck-typed shape the
  surfaces render.

Everything here is shared plumbing in the spirit of
:mod:`femtools.script.loading`: no command parsing, no printing.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

__all__ = ["recover_spr_nodal", "era_from_data", "expanded_mac_matrix"]


# ---------------------------------------------------------------------------
# ZZ-SPR
# ---------------------------------------------------------------------------
def recover_spr_nodal(model: Any, static: Any, stress: Any = None) -> Any:
    """Superconvergent-patch-recovered nodal stresses of a solved static case.

    Runs the centroid recovery first (SPR fits the patch polynomial
    through the superconvergent centroid samples) and hands the result
    to :func:`femtools.fea.recover.recover_spr`.  The SPR kernel is
    developed independently, so the call is signature-adaptive: it
    accepts either the ``(stress, model)`` order of ``average_nodal``
    or a ``(model, u)`` order like ``recover_stress``.

    Parameters
    ----------
    model:
        The model the static case was solved on.
    static:
        The ``StaticResult`` (or bare displacement field) of the solve.
    stress:
        An already recovered centroid ``StressResult``; computed with
        :func:`femtools.fea.recover.recover_stress` when omitted.

    Returns
    -------
    A nodal stress result (``node_ids`` + per-node tensors), whatever
    concrete type the kernel returns.

    Raises
    ------
    ImportError
        When ``recover_spr`` (or the centroid recovery it feeds on) is
        not available in this installation.
    """
    from femtools.fea.recover import recover_spr, recover_stress

    if stress is None:
        stress = recover_stress(model, static)

    try:
        params = list(inspect.signature(recover_spr).parameters)
    except (TypeError, ValueError):
        params = []
    first = params[0].lower() if params else ""
    second = params[1].lower() if len(params) > 1 else ""

    if first in ("model", "mesh", "fe_model"):
        # (model, u)-style kernel: prefer the displacement field when the
        # second argument is named like one, else hand it the centroid result
        other = static if second in ("u", "static", "displacement", "field") else stress
        attempts = [(model, other), (model, stress), (stress, model)]
    else:
        # average_nodal-style kernel: (stress, model)
        attempts = [(stress, model), (model, stress), (model, static)]

    last_exc: Exception | None = None
    for args in attempts:
        try:
            return recover_spr(*args)
        except (TypeError, AttributeError) as exc:
            last_exc = exc
    raise TypeError(
        f"could not call recover_spr with any supported argument order: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# ERA
# ---------------------------------------------------------------------------
def era_from_data(
    h: Any = None,
    dt: float | None = None,
    *,
    frf: Any = None,
    freq_hz: Any = None,
    order: int = 10,
    n_modes: int | None = None,
    f_range: tuple[float, float] | None = None,
) -> Any:
    """Modal parameters from impulse responses (or an FRF block) via ERA.

    Give either the impulse-response block ``h`` (time along the last
    axis) with its sample step ``dt``, or an FRF block ``frf`` with its
    ``freq_hz`` axis -- the FRF is inverse-transformed first with
    :func:`femtools.mpe.lsce.irf_from_frf`.  Tuning keywords the kernel
    does not accept are dropped, so the adapter keeps working across
    signature variants of the independently developed ERA kernel.

    Returns the kernel's ``ModalParameterResult``.

    Raises
    ------
    ImportError
        When ``femtools.mpe.era`` is not available in this installation.
    ValueError
        When neither ``h``+``dt`` nor ``frf``+``freq_hz`` is given.
    """
    from femtools.mpe.era import era

    if h is None and (frf is None or freq_hz is None):
        raise ValueError(
            "era_from_data needs impulse responses (h= with dt=) or an FRF "
            "block (frf= with freq_hz=)"
        )
    if h is not None and dt is None:
        raise ValueError("impulse-response input needs the sample step dt=")

    tuning: dict[str, Any] = {"order": int(order)}
    if n_modes is not None:
        tuning["n_modes"] = int(n_modes)
    if f_range is not None:
        tuning["f_range"] = (float(f_range[0]), float(f_range[1]))

    try:
        params = inspect.signature(era).parameters
    except (TypeError, ValueError):
        params = {}
    has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if params and not has_varkw:
        tuning = {k: v for k, v in tuning.items() if k in params}

    if h is None:
        if "freq_hz" in params or has_varkw:
            # the kernel takes the spectrum natively and inverse-transforms it
            return era(frf, freq_hz=np.asarray(freq_hz, dtype=float), **tuning)
        from femtools.mpe.lsce import irf_from_frf

        h, dt = irf_from_frf(frf, freq_hz)

    if params and "dt" in params:
        return era(h, dt=float(dt), **tuning)
    return era(h, float(dt), **tuning)


# ---------------------------------------------------------------------------
# expanded MAC
# ---------------------------------------------------------------------------
def expanded_mac_matrix(phi: Any, master: Any) -> tuple[np.ndarray, Any]:
    """SEREP-expanded MAC of a mode set against itself through a master subset.

    Restricts ``phi`` to the ``master`` rows, expands the restriction
    back onto the full DOF set through the same basis and correlates the
    expansion with the original modes -- the Round-10 identity check
    (diagonal 1, off-diagonal ~0 for the retained modes).

    Parameters
    ----------
    phi:
        Full mode matrix ``(n_full, n_modes)`` (a modal result works too).
    master:
        Row indices of the measured (master) DOFs.

    Returns
    -------
    (mac, expansion)
        The MAC matrix as a plain ndarray and the kernel's expansion /
        result object (``None`` when the kernel returned only a matrix).

    Raises
    ------
    ImportError
        When ``expanded_mac`` is not available in this installation.
    """
    from femtools.correlation.expansion import expanded_mac

    matrix = np.asarray(getattr(phi, "modes", phi))
    rows = np.asarray(master, dtype=int).reshape(-1)
    phi_test = matrix[rows, :]

    try:
        result = expanded_mac(phi_test, phi, rows)
    except TypeError:
        # convenience signature: the kernel restricts the modes itself
        result = expanded_mac(phi, rows)

    expansion: Any = None
    if isinstance(result, tuple):
        mac = result[0]
        expansion = result[1] if len(result) > 1 else None
    else:
        mac = None
        for attr in ("mac", "mac_matrix", "matrix", "values"):
            mac = getattr(result, attr, None)
            if mac is not None:
                break
        if mac is None:
            mac = result  # a bare matrix (ndarray-like result types included)
        else:
            expansion = getattr(result, "expansion", result)
    return np.asarray(mac, dtype=float), expansion
