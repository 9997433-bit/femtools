"""Response builders for model updating.

The updater only needs a callable ``p -> r(p)`` returning a real response
vector.  This module builds such callables either

* from an :class:`femtools.core.model.FEModel` plus the FEA solver
  (:func:`modal_response_function`, :func:`frf_response_function`), or
* from any user supplied callback (pass it straight to the updater).

``femtools.fea`` is imported lazily so that this package stays importable — and
fully usable through callbacks and the analytical reference models — even while
the FEA layer is not yet available.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .parameters import ParameterSet, apply_parameters, as_parameters, snapshot_baseline

__all__ = [
    "ResponseSpec",
    "have_fea",
    "solve_modal",
    "modal_response_function",
    "frf_response_function",
    "mac_vector",
    "pair_by_mac",
]


# ----------------------------------------------------------------------
# Optional FEA bridge
# ----------------------------------------------------------------------
def have_fea() -> bool:
    """``True`` when ``femtools.fea`` can be imported."""
    try:  # pragma: no cover - depends on sibling package availability
        import femtools.fea.eigen  # noqa: F401
    except Exception:
        return False
    return True


def _default_modal_solver() -> Callable[..., Any]:
    try:
        from femtools.fea.eigen import solve_modes
    except Exception as exc:  # pragma: no cover - exercised when fea is absent
        raise RuntimeError(
            "femtools.fea is not available; pass an explicit `solver=` callback "
            "or use a response callable / femtools.updating.reference model."
        ) from exc
    return solve_modes


def _default_assembler() -> Callable[..., Any]:
    try:
        from femtools.fea.assemble import assemble_km
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "femtools.fea.assemble is not available; analytic sensitivities "
            "require either the FEA layer or explicit dK/dM matrices."
        ) from exc
    return assemble_km


def _unpack_modal(result: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract ``(freq_hz, modes)`` from a solver result, duck-typed."""
    if result is None:
        raise ValueError("modal solver returned None")
    freq = getattr(result, "freq_hz", None)
    if freq is None:
        freq = getattr(result, "frequencies", None)
    modes = getattr(result, "modes", None)
    if modes is None:
        modes = getattr(result, "phi", None)
    if freq is None:
        if isinstance(result, (tuple, list)) and len(result) >= 1:
            freq = result[0]
            if modes is None and len(result) >= 2:
                modes = result[1]
        else:
            freq = result
    return np.asarray(freq, dtype=float).ravel(), (
        None if modes is None else np.asarray(modes)
    )


def solve_modal(
    model: Any,
    n_modes: int = 10,
    *,
    solver: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Solve the eigenproblem, returning ``(freq_hz, modes)``.

    Works with :func:`femtools.fea.eigen.solve_modes`, with any callable of the
    same shape, and with :class:`femtools.updating.reference.ReferenceModel`.
    """
    from .reference import ReferenceModel

    if isinstance(model, ReferenceModel):
        return model.eig(None, n_modes)
    fn = solver or _default_modal_solver()
    return _unpack_modal(fn(model, n_modes=n_modes, **kwargs))


# ----------------------------------------------------------------------
# MAC helpers (local copy so this package does not depend on femtools.correlation)
# ----------------------------------------------------------------------
def _mac_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(np.asarray(a))
    b = np.atleast_2d(np.asarray(b))
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    num = np.abs(a.conj().T @ b) ** 2
    na = np.real(np.sum(a.conj() * a, axis=0))
    nb = np.real(np.sum(b.conj() * b, axis=0))
    den = np.outer(na, nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, 0.0)
    return np.asarray(out, dtype=float)


def mac_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Modal Assurance Criterion matrix (local implementation)."""
    return _mac_matrix(a, b)


def mac_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Diagonal MAC values between paired mode-shape columns of ``a`` and ``b``."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    n = min(a.shape[1], b.shape[1])
    return np.array([_mac_matrix(a[:, [i]], b[:, [i]])[0, 0] for i in range(n)])


def pair_by_mac(
    modes: np.ndarray, reference: np.ndarray, *, threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one pairing of ``modes`` columns against ``reference``.

    Returns ``(index_into_modes, mac_value)`` with one entry per reference mode
    (``-1`` when no partner exceeds ``threshold``).
    """
    mac = _mac_matrix(modes, reference)  # (n_test, n_ref)
    n_test, n_ref = mac.shape
    pairs = np.full(n_ref, -1, dtype=int)
    vals = np.zeros(n_ref)
    used: set[int] = set()
    order = np.argsort(-mac.max(axis=0))
    for j in order:
        col = mac[:, j].copy()
        for u in used:
            col[u] = -1.0
        i = int(np.argmax(col))
        if col[i] >= threshold and col[i] > 0:
            pairs[j] = i
            vals[j] = mac[i, j]
            used.add(i)
    return pairs, vals


# ----------------------------------------------------------------------
@dataclass
class ResponseSpec:
    """Declares which quantities enter the residual vector.

    Attributes
    ----------
    n_modes:
        Number of modes solved for.
    mode_indices:
        Which (0-based, sorted-ascending) modes contribute frequencies.
        ``None`` -> all of them.
    reference_modes:
        Reference mode-shape matrix.  When given, mode pairing is done by MAC and
        (optionally) ``1 - MAC`` values are appended to the residual.
    include_mac:
        Append ``1 - MAC`` per paired mode to the response vector.
    dof_indices:
        Row selection applied to computed mode shapes before MAC (measured DOF).
    frf:
        Optional ``(freq_hz, inputs, outputs)`` request; the magnitude (or
        log-magnitude) of the FRF at those lines is appended.
    log_frf:
        Use ``log10|H|`` instead of ``|H|`` (recommended: makes the residual
        scale-free).
    """

    n_modes: int = 10
    mode_indices: Sequence[int] | None = None
    reference_modes: np.ndarray | None = None
    include_mac: bool = False
    dof_indices: Sequence[int] | None = None
    frf: dict[str, Any] | None = None
    log_frf: bool = True
    skip_rigid_body: bool = False
    rigid_body_tol: float = 1.0e-3
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        idx = list(self.mode_indices) if self.mode_indices is not None else list(
            range(self.n_modes)
        )
        out = [f"f{i + 1}" for i in idx]
        if self.include_mac:
            out += [f"1-MAC{i + 1}" for i in idx]
        return out


def modal_response_function(
    model: Any,
    parameters: Any,
    spec: ResponseSpec | None = None,
    *,
    solver: Callable[..., Any] | None = None,
    n_modes: int | None = None,
    reference_modes: np.ndarray | None = None,
    include_mac: bool = False,
    mode_indices: Sequence[int] | None = None,
    dof_indices: Sequence[int] | None = None,
    solver_kwargs: dict[str, Any] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build ``p -> [frequencies, (1-MAC)]`` for an :class:`FEModel`.

    The model is deep-copied for every evaluation, so the caller's database is
    never mutated.
    """
    pset: ParameterSet = as_parameters(parameters)
    if spec is None:
        spec = ResponseSpec(
            n_modes=n_modes if n_modes is not None else 10,
            mode_indices=mode_indices,
            reference_modes=reference_modes,
            include_mac=include_mac,
            dof_indices=dof_indices,
        )
    base = snapshot_baseline(model, pset)
    kwargs = dict(solver_kwargs or {})

    def _f(p: np.ndarray) -> np.ndarray:
        m = apply_parameters(model, pset, p, copy_model=True, baseline=base)
        freq, modes = solve_modal(m, spec.n_modes, solver=solver, **kwargs)
        return assemble_response(freq, modes, spec)

    return _f


def assemble_response(
    freq: np.ndarray, modes: np.ndarray | None, spec: ResponseSpec
) -> np.ndarray:
    """Combine frequencies / MAC into a single real response vector."""
    freq = np.asarray(freq, dtype=float).ravel()
    if spec.skip_rigid_body:
        keep = freq > spec.rigid_body_tol
        freq = freq[keep]
        if modes is not None:
            modes = np.asarray(modes)[:, keep]
    ref = spec.reference_modes
    if ref is not None and modes is not None:
        phi = np.asarray(modes)
        if spec.dof_indices is not None:
            phi = phi[np.asarray(spec.dof_indices, dtype=int), :]
        pairs, macs = pair_by_mac(phi, np.asarray(ref))
        valid = pairs >= 0
        f_sel = np.where(valid, freq[np.clip(pairs, 0, len(freq) - 1)], np.nan)
        parts = [f_sel]
        if spec.include_mac:
            parts.append(1.0 - macs)
        return np.concatenate(parts)
    idx = (
        np.asarray(spec.mode_indices, dtype=int)
        if spec.mode_indices is not None
        else np.arange(min(spec.n_modes, freq.size))
    )
    idx = idx[idx < freq.size]
    return freq[idx]


def frf_response_function(
    model: Any,
    parameters: Any,
    freq_hz: ArrayLike,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    damping: Any = 0.02,
    n_modes: int = 20,
    solver: Callable[..., Any] | None = None,
    log_magnitude: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build ``p -> log10|H(omega)|`` sampled at ``freq_hz``.

    Requires ``femtools.dynamics.frf`` (or an equivalent ``solver`` callback that
    returns a complex array shaped ``(n_out, n_in, n_freq)``).
    """
    pset = as_parameters(parameters)
    base = snapshot_baseline(model, pset)
    freq_hz = np.asarray(freq_hz, dtype=float)

    def _frf(m: Any) -> np.ndarray:
        if solver is not None:
            return np.asarray(solver(m, freq_hz, inputs, outputs))
        from femtools.dynamics.frf import modal_frf  # lazy
        from femtools.fea.eigen import solve_modes

        modal = solve_modes(m, n_modes=n_modes)
        res = modal_frf(modal, inputs, outputs, freq_hz, damping)
        H = getattr(res, "H", None)
        if H is None:
            H = getattr(res, "frf", res)
        return np.asarray(H)

    def _f(p: np.ndarray) -> np.ndarray:
        m = apply_parameters(model, pset, p, copy_model=True, baseline=base)
        H = np.asarray(_frf(m))
        mag = np.abs(H).ravel()
        if log_magnitude:
            return np.log10(np.maximum(mag, 1e-300))
        return mag

    return _f
