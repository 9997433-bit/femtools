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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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

__all__ = [
    "ResponseSpec",
    "have_fea",
    "solve_modal",
    "modal_response_function",
    "frf_response_function",
    "static_displacement_response",
    "static_stress_response",
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
def _as_mode_columns(x: np.ndarray) -> np.ndarray:
    """Return ``x`` as an ``(n_dof, n_modes)`` matrix.

    A 1-D input is a *single* mode shape, so it becomes one column;
    ``np.atleast_2d`` would instead make it a row of n single-DOF modes, whose
    MAC matrix is all ones.
    """
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr[:, None]
    return np.atleast_2d(arr)


def _mac_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _as_mode_columns(a)
    b = _as_mode_columns(b)
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
    a = _as_mode_columns(a)
    b = _as_mode_columns(b)
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
    model = unwrap_model(model)
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


#: Local DOF spellings accepted by :func:`static_displacement_response`, mapped
#: onto the 0-based component convention of :mod:`femtools.core.model`.
_DOF_ALIASES: dict[str, int] = {
    "ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5,
    "x": 0, "y": 1, "z": 2, "tx": 0, "ty": 1, "tz": 2,
    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
}


def _normalize_component(comp: Any) -> int:
    """``"uz"`` / ``"3"`` / ``2`` -> the 0-based local DOF index."""
    if isinstance(comp, (int, np.integer)) and not isinstance(comp, bool):
        value = int(comp)
        if not 0 <= value < 6:
            raise ValueError(f"local dof {comp!r} is outside 0..5")
        return value
    key = str(comp).strip().lower()
    try:
        return _DOF_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown displacement component {comp!r}; expected 0..5 or one of "
            f"{sorted(set(_DOF_ALIASES))}"
        ) from None


def _loaded_dofs(model: Any, loads: Any) -> list[tuple[Any, int]]:
    """The ``(node, dof)`` pairs a load acts on -- the default measurement set."""
    pairs: list[tuple[Any, int]] = []
    if isinstance(loads, Mapping):
        for key in loads:
            node, comp = key if isinstance(key, tuple) else (key, 0)
            pairs.append((node, _normalize_component(comp)))
        return pairs
    if loads is None:
        for load in getattr(model, "loads", None) or ():
            node = getattr(load, "node_id", None)
            if node is None:
                continue
            as_dofs = getattr(load, "as_dof_values", None)
            if as_dofs is None:  # pragma: no cover - foreign load record
                continue
            pairs.extend((node, int(comp)) for comp, _value in as_dofs())
    return pairs


def _normalize_static_loads(model: Any, loads: Any) -> Any:
    """Expand ``model.loads`` force/moment vectors into ``{(node, dof): value}``.

    The load builder reads component keys, not the vectors
    :class:`femtools.core.model.Load` stores, so ``loads=None`` is resolved here
    to keep it meaning "use the loads already on the model".
    """
    if loads is not None:
        return loads
    pairs: dict[tuple[Any, int], float] = {}
    for record in getattr(model, "loads", None) or ():
        node = getattr(record, "node_id", None)
        expand = getattr(record, "as_dof_values", None)
        if node is None or expand is None:  # pragma: no cover - foreign load record
            return None
        for comp, value in expand():
            key = (node, int(comp))
            pairs[key] = pairs.get(key, 0.0) + float(value)
    return pairs or None


def _resolve_response_dofs(model: Any, dofs: Any, loads: Any) -> list[tuple[Any, int] | int]:
    """Normalise the requested measurement DOFs.

    Accepts ``(node_id, component)`` pairs (component as ``0..5`` or ``"uz"``),
    a ``{node_id: components}`` mapping, a
    :class:`~femtools.core.model.DOFSet`, plain global equation numbers, or
    ``None`` -- which measures wherever the load is applied, i.e. the drive
    points of the test.
    """
    if dofs is None:
        pairs = _loaded_dofs(model, loads)
        if not pairs:
            raise ValueError(
                "no measurement DOFs given and the model carries no nodal load to take "
                "them from; pass `dofs=[(node_id, 'uz'), ...]`"
            )
        return list(pairs)
    if isinstance(dofs, Mapping):
        out: list[tuple[Any, int] | int] = []
        for node, comps in dofs.items():
            if isinstance(comps, (str, int, np.integer)):
                comps = [comps]
            out.extend((node, _normalize_component(c)) for c in comps)
        return out
    resolved: list[tuple[Any, int] | int] = []
    for item in dofs:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            resolved.append(int(item))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            resolved.append((item[0], _normalize_component(item[1])))
        else:
            raise TypeError(
                f"cannot interpret measurement dof {item!r}; expected (node_id, component) "
                "or a global equation number"
            )
    return resolved


def _static_indices(
    model: Any, pairs: Sequence[tuple[Any, int] | int], assembly: Any = None
) -> np.ndarray:
    """Global equation numbers of the requested DOFs."""
    if assembly is not None:
        dof_map = assembly.dof_map
        return np.array(
            [p if isinstance(p, int) else dof_map.index(p[0], p[1]) for p in pairs], dtype=int
        )
    table = model.dof_map() if hasattr(model, "dof_map") else None
    if table is None:  # pragma: no cover - foreign model without a DOF map
        raise TypeError(
            "the static solver returned a bare vector and the model has no dof_map(); "
            "pass global equation numbers as `dofs`"
        )
    return np.array(
        [p if isinstance(p, int) else table[(p[0], p[1])] for p in pairs], dtype=int
    )


def _default_static_solver() -> Callable[..., Any]:
    try:
        from femtools.fea.static import solve_static
    except Exception as exc:  # pragma: no cover - exercised when fea is absent
        raise RuntimeError(
            "femtools.fea is not available; pass an explicit `solver=` callback to "
            "static_displacement_response."
        ) from exc
    return solve_static


def static_displacement_response(
    model: Any,
    parameters: Any,
    dofs: Any = None,
    *,
    loads: Any = None,
    solver: Callable[..., Any] | None = None,
    solver_kwargs: dict[str, Any] | None = None,
    scale: ArrayLike | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build ``p -> u(p)`` at selected DOFs from a linear static solution.

    This is the static counterpart of :func:`modal_response_function`: a
    first-class response for :func:`femtools.updating.update_model`, so a model
    can be updated against measured *deflections* (a static test, a proof load,
    a dial-gauge or DIC field) instead of, or alongside, measured frequencies.

    Parameters
    ----------
    model:
        :class:`femtools.core.model.FEModel` (or a project wrapper around one).
        It is deep-copied for every evaluation, so the caller's database is
        never touched.
    parameters:
        Parameter specification -- see :func:`femtools.updating.as_parameters`.
    dofs:
        Measurement DOFs: ``(node_id, component)`` pairs with the component
        given as ``0..5`` or as a label (``"uz"``, ``"z"``, ``"rx"``, ...), a
        ``{node_id: "xyz"}``-style mapping, a
        :class:`~femtools.core.model.DOFSet`, or plain global equation numbers.
        ``None`` (default) measures every loaded DOF.
    loads:
        Anything :func:`femtools.fea.loads.build_load_vector` accepts.  ``None``
        uses the model's own loads.
    solver:
        Optional ``solver(model, loads) -> u`` callback replacing
        :func:`femtools.fea.static.solve_static`.  It may return a full
        displacement vector or a :class:`~femtools.fea.static.StaticResult`.
    scale:
        Optional multiplier applied to the response (a scalar, or one factor per
        DOF).  Handy to bring millimetre-scale test data and metre-scale FE
        results onto one residual.

    Returns
    -------
    Callable ``p -> ndarray`` of displacements in the basic (global) frame.

    Examples
    --------
    Recover a wrong Young's modulus from one measured tip deflection::

        f = static_displacement_response(model, [{"type": "material", "id": 1,
                                                  "name": "E"}], [(tip, "uz")])
        res = update_model(model, parameters, targets=[u_measured], response=f)
    """
    model = unwrap_model(model)
    pset: ParameterSet = as_parameters(parameters)
    base = snapshot_baseline(model, pset)
    pairs = _resolve_response_dofs(model, dofs, loads)
    kwargs = dict(solver_kwargs or {})
    factor = None if scale is None else np.asarray(scale, dtype=float)
    cached_index: list[np.ndarray | None] = [None]

    def _solve(m: Any) -> np.ndarray:
        applied = _normalize_static_loads(m, loads)
        if solver is not None:
            out = solver(m, applied)
        else:
            out = _default_static_solver()(m, applied, full_result=True, **kwargs)
        assembly = getattr(out, "assembly", None)
        u = np.asarray(getattr(out, "u", out), dtype=float)
        if u.ndim != 1:
            raise ValueError(
                "static_displacement_response expects one load case; the solver returned "
                f"a {u.shape} displacement matrix"
            )
        if assembly is not None:
            u = np.asarray(assembly.to_basic(u), dtype=float)
        if cached_index[0] is None:
            cached_index[0] = _static_indices(m, pairs, assembly)
        return u[cached_index[0]]

    def _f(p: np.ndarray) -> np.ndarray:
        m = apply_parameters(model, pset, p, copy_model=True, baseline=base)
        values = _solve(m)
        return values if factor is None else values * factor

    return _f


#: Voigt component names accepted by :func:`static_stress_response`, mapped onto
#: the column of ``StressResult.stress`` / ``StressResult.strain``.
_STRESS_COMPONENTS: dict[str, int] = {
    "xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "zx": 5,
    "11": 0, "22": 1, "33": 2, "12": 3, "23": 4, "13": 5,
    "sxx": 0, "syy": 1, "szz": 2, "sxy": 3, "syz": 4, "szx": 5,
    "exx": 0, "eyy": 1, "ezz": 2, "exy": 3, "eyz": 4, "ezx": 5,
    "axial": 0,
}

#: Spellings of the frame-independent equivalent stress.
_VON_MISES_NAMES = frozenset({"von_mises", "vonmises", "mises", "vm", "equivalent", "seqv"})


def _recover_stress() -> Callable[..., Any]:
    try:
        from femtools.fea.recover import recover_stress
    except Exception as exc:  # pragma: no cover - exercised when fea is absent
        raise RuntimeError(
            "femtools.fea.recover is not available; static_stress_response needs the "
            "stress recovery kernel."
        ) from exc
    return recover_stress


def static_stress_response(
    model: Any,
    parameters: Any,
    elements: Any = None,
    *,
    component: str = "von_mises",
    quantity: str = "stress",
    frame: str = "element",
    layer: Any = "mid",
    loads: Any = None,
    enforced: Mapping[Any, float] | None = None,
    solver: Callable[..., Any] | None = None,
    solver_kwargs: dict[str, Any] | None = None,
    scale: ArrayLike | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build ``p -> sigma(p)`` at selected elements from a linear static solution.

    The stress counterpart of :func:`static_displacement_response`: it feeds
    :func:`femtools.fea.recover.recover_stress` into
    :func:`femtools.updating.update_model`, so a model can be updated against
    measured *stresses or strains* — strain gauges, a photoelastic or DIC field,
    a proof-load survey — instead of, or alongside, deflections and frequencies.

    Parameters
    ----------
    model:
        :class:`femtools.core.model.FEModel` (or a project wrapper around one),
        deep-copied for every evaluation.
    parameters:
        Parameter specification -- see :func:`femtools.updating.as_parameters`.
    elements:
        Element ids to report, in the given order.  ``None`` (default) reports
        every element the recovery covers, in model order.
    component:
        ``"von_mises"`` (default) or a Voigt component (``"xx"``, ``"xy"``, ...).
    quantity:
        ``"stress"`` (default) or ``"strain"`` — the latter is what a gauge
        rosette actually measures.
    frame:
        ``"element"`` (default, the frame ``StressResult.stress`` is written in)
        or ``"basic"`` to rotate into global axes first.  Ignored for
        ``component="von_mises"``, which is frame independent.
    layer:
        Through-thickness station for shells, as in :func:`recover_stress`.
    loads:
        Anything :func:`femtools.fea.loads.build_load_vector` accepts. ``None``
        uses the model's own loads.
    enforced:
        ``{(node_id, dof): value}`` prescribed displacements, handed to
        :func:`femtools.fea.static.solve_static` — the same thing as
        ``solver_kwargs={"enforced": ...}``, spelled out because it is what
        makes a stress residual informative at all.  Under **dead load** a
        statically determinate structure carries ``sigma = F / A`` whatever its
        modulus, so a stress residual there is blind to ``E``; driving the same
        model by prescribed displacement gives ``sigma = E * delta / L``, which
        is exactly linear in it.  (Measured *strain* is the other way round:
        informative under dead load, blind under enforced displacement.)
    solver:
        Optional ``solver(model, loads) -> StaticResult`` callback replacing
        :func:`femtools.fea.static.solve_static`.  It must return a result the
        recovery can read the assembly from, not a bare vector.
    scale:
        Optional multiplier applied to the response (scalar or one per element).

    Returns
    -------
    Callable ``p -> ndarray`` with one entry per requested element.

    Examples
    --------
    Recover a wrong Young's modulus from a constant-stress patch driven by
    prescribed end displacements::

        f = static_stress_response(model, parameters, component="xx",
                                   enforced={(tip, 0): 1.0e-4})
        res = update_model(model, parameters, targets=measured, response=f)
    """
    model = unwrap_model(model)
    pset: ParameterSet = as_parameters(parameters)
    base = snapshot_baseline(model, pset)
    kwargs = dict(solver_kwargs or {})
    if enforced is not None:
        if "enforced" in kwargs:
            raise ValueError(
                "pass the prescribed displacements either as `enforced` or inside "
                "`solver_kwargs`, not both"
            )
        if solver is not None:
            raise ValueError(
                "`enforced` is handed to femtools.fea.static.solve_static, which a "
                "custom `solver` replaces; apply the prescribed displacements inside "
                "the solver callback instead"
            )
        kwargs["enforced"] = dict(enforced)
    factor = None if scale is None else np.asarray(scale, dtype=float)

    key = str(component).strip().lower()
    if key in _VON_MISES_NAMES:
        column = None
    elif key in _STRESS_COMPONENTS:
        column = _STRESS_COMPONENTS[key]
    else:
        raise ValueError(
            f"unknown stress component {component!r}; expected 'von_mises' or one of "
            f"{sorted(set(_STRESS_COMPONENTS))}"
        )

    what = str(quantity).strip().lower()
    if what not in ("stress", "strain"):
        raise ValueError(f"quantity must be 'stress' or 'strain', got {quantity!r}")
    basic = str(frame).strip().lower()
    if basic not in ("element", "basic", "global"):
        raise ValueError(f"frame must be 'element' or 'basic', got {frame!r}")
    if column is None and what == "strain":
        raise ValueError("von Mises is a stress measure; pass an explicit strain component")

    wanted = None if elements is None else list(elements)
    cached_rows: list[np.ndarray | None] = [None]

    def _values(result: Any) -> np.ndarray:
        if column is None:
            return np.asarray(result.von_mises, dtype=float)
        if what == "strain":
            table = result.strain_basic if basic != "element" else result.strain
        else:
            table = result.stress_basic if basic != "element" else result.stress
        return np.asarray(table, dtype=float)[:, column]

    def _f(p: np.ndarray) -> np.ndarray:
        m = apply_parameters(model, pset, p, copy_model=True, baseline=base)
        applied = _normalize_static_loads(m, loads)
        if solver is not None:
            out = solver(m, applied)
        else:
            out = _default_static_solver()(m, applied, full_result=True, **kwargs)
        recovered = _recover_stress()(m, out, layer=layer)
        if cached_rows[0] is None:
            cached_rows[0] = np.array(
                range(len(recovered)) if wanted is None
                else [recovered.index_of(eid) for eid in wanted],
                dtype=int,
            )
        values = _values(recovered)[cached_rows[0]]
        return values if factor is None else values * factor

    return _f


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
    model = unwrap_model(model)
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
