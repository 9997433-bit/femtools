"""Reproducible verification cases for the element library.

These builders exist so that the acceptance numbers quoted in the docs and in
the test suite come from one place instead of being re-derived (and slightly
re-invented) in every test module.  Everything is expressed with plain
dictionaries, which the duck-typed kernel accepts directly
(:mod:`femtools.fea.protocols`); the same cases run unchanged against
:class:`femtools.core.model.FEModel`.

The HEX8 cases below are the classical anti-locking checks:

``hex8_bending_ratio``
    Slender cantilever with a single element through the thickness, compared
    with the Timoshenko tip deflection.  A trilinear brick integrated 2x2x2
    shear-locks to about 0.64 of the reference; the default incompatible-mode
    formulation reaches 0.99.
``hex8_patch_test_error``
    Constant-stress patch test on a distorted mesh with one fully enclosed
    interior node -- the check the Taylor correction of the internal modes
    exists for.
``hex8_rigid_body_frequencies``
    Free-free block: exactly six zero frequencies and no seventh, i.e. the
    softened element has not gained an hourglass mechanism.

Mesh distortion caveat
----------------------

The accuracy the incompatible-mode element buys back is not distortion
invariant, and the two distortions that look equally bad to the eye behave
completely differently.  ``hex_cantilever`` can build either of them through
its ``distortion`` argument, so the claim is reproducible rather than folklore:

``distortion="parallelogram"``
    An affine shear of the whole mesh.  The Jacobian is still constant inside
    each element, the Taylor scaling ``det J0 / det J`` is exactly one and the
    internal modes stay exact.  The tip ratio of the six-element cantilever is
    flat at 0.975 out to ``skew=0.8``, while the plain 2x2x2 element decays
    from 0.445 to 0.152 over the same range.  Skew angle on its own is
    therefore *not* the quantity to write a mesh-quality rule against.
``distortion="trapezoid"``
    Alternating trapezoids, the MacNeal distortion.  Now the Jacobian varies
    within the element, the internal gradients are only correct at the centre
    and the advantage disappears quickly: 0.975 at ``skew=0`` falls to 0.518 at
    ``skew=0.2`` and to 0.215 at ``skew=0.4``, where the element is no better
    than the locking one it replaced.

:func:`hex8_jacobian_spread` reports the quantity that actually predicts this,
``max|det J| / min|det J|`` over the Gauss points of an element.  It stays at
1.0 for any parallelepiped however skewed; the trapezoidal cases above score
1.26 and 1.60.  As a rule of thumb the incompatible modes are worth their cost
up to a spread of roughly 1.1 and have lost most of their advantage past 1.5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg as sla

from .assemble import assemble_km
from .eigen import mass_normalize, solve_modes
from .elements import ModelIndex, element_spec
from .elements.solid import _hex_shape
from .protocols import get_any, iter_records
from .quadrature import gauss_3d
from .reduction import guyan, irs, serep
from .static import solve_static

__all__ = [
    "DISTORTIONS",
    "beam_cantilever",
    "complete_spectrum_quality",
    "guyan_condensation_error",
    "hex8_bending_ratio",
    "hex8_jacobian_spread",
    "hex8_patch_test_error",
    "hex8_rigid_body_frequencies",
    "hex_cantilever",
    "reduction_frequency_errors",
    "serep_slave_recovery",
    "timoshenko_tip_deflection",
]

#: Distortion patterns understood by :func:`hex_cantilever`; see the module
#: docstring for what each one does to the incompatible-mode element.
DISTORTIONS: tuple[str, ...] = ("none", "parallelogram", "trapezoid")


def _model(nodes: dict[int, Any], elements: dict[int, Any], E: float, nu: float, rho: float):
    return {
        "nodes": nodes,
        "elements": elements,
        "materials": {1: {"E": E, "nu": nu, "rho": rho}},
        "properties": {1: {"type": "solid", "material_id": 1}},
        "spcs": [],
    }


def hex_cantilever(
    nx: int = 10,
    ny: int = 1,
    nz: int = 1,
    *,
    length: float = 10.0,
    width: float = 1.0,
    height: float = 1.0,
    E: float = 1.0e7,
    nu: float = 0.3,
    rho: float = 1.0,
    tip_force: float = 1.0,
    clamped: bool = True,
    distortion: str = "none",
    skew: float = 0.0,
) -> tuple[dict[str, Any], list[int], dict[tuple[int, int], float]]:
    """Structured HEX8 mesh of a beam, clamped at ``x = 0``.

    ``distortion`` (one of :data:`DISTORTIONS`) warps the mesh in the ``x``-``z``
    plane by ``skew`` without moving the end faces off their planes, so the
    Timoshenko reference stays applicable:

    ``"parallelogram"``
        Affine shear, ``x += skew * (z - height/2)``.  Every element keeps a
        constant Jacobian.
    ``"trapezoid"``
        Alternating trapezoids, ``x += skew * dx * (-1)**i * (z - height/2)``
        on node plane ``i``, which makes the Jacobian vary inside the element.

    Returns the model, the node ids on the free end face and a load dictionary
    spreading ``tip_force`` over that face in the ``z`` direction.
    """
    if distortion not in DISTORTIONS:
        raise ValueError(f"unknown distortion {distortion!r}; expected one of {DISTORTIONS}")

    dx = length / nx
    nodes: dict[int, Any] = {}
    ids: dict[tuple[int, int, int], int] = {}
    counter = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                x = length * i / nx
                z = height * k / nz
                if distortion == "parallelogram":
                    x += skew * (z - 0.5 * height)
                elif distortion == "trapezoid":
                    x += skew * dx * (1.0 if i % 2 else -1.0) * (z - 0.5 * height)
                ids[(i, j, k)] = counter
                nodes[counter] = {"xyz": (x, width * j / ny, z)}
                counter += 1

    elements: dict[int, Any] = {}
    eid = 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                elements[eid] = {
                    "type": "HEX8",
                    "property_id": 1,
                    "nodes": (
                        ids[(i, j, k)],
                        ids[(i + 1, j, k)],
                        ids[(i + 1, j + 1, k)],
                        ids[(i, j + 1, k)],
                        ids[(i, j, k + 1)],
                        ids[(i + 1, j, k + 1)],
                        ids[(i + 1, j + 1, k + 1)],
                        ids[(i, j + 1, k + 1)],
                    ),
                }
                eid += 1

    model = _model(nodes, elements, E, nu, rho)
    if clamped:
        model["spcs"] = [
            {"node_id": ids[(0, j, k)], "dofs": (0, 1, 2)}
            for j in range(ny + 1)
            for k in range(nz + 1)
        ]
    tip_nodes = [ids[(nx, j, k)] for j in range(ny + 1) for k in range(nz + 1)]
    loads = {(nid, 2): tip_force / len(tip_nodes) for nid in tip_nodes}
    return model, tip_nodes, loads


def timoshenko_tip_deflection(
    force: float = 1.0,
    length: float = 10.0,
    width: float = 1.0,
    height: float = 1.0,
    E: float = 1.0e7,
    nu: float = 0.3,
    kappa: float = 5.0 / 6.0,
) -> float:
    """``P L^3 / (3 E I) + P L / (kappa G A)`` for a rectangular section."""
    inertia = width * height**3 / 12.0
    shear_modulus = E / (2.0 * (1.0 + nu))
    return force * length**3 / (3.0 * E * inertia) + force * length / (
        kappa * shear_modulus * width * height
    )


def hex8_bending_ratio(
    formulation: str | None = None,
    *,
    nx: int = 10,
    ny: int = 1,
    nz: int = 1,
    **case: Any,
) -> float:
    """Mean tip deflection of :func:`hex_cantilever` over the Timoshenko value.

    Extra keywords go to :func:`hex_cantilever`, so ``distortion`` and ``skew``
    turn this into the mesh-distortion sweep described in the module docstring.
    """
    model, tip_nodes, loads = hex_cantilever(nx, ny, nz, **case)
    options = {"hex8": formulation} if formulation else None
    asm = assemble_km(model, options=options)
    u = solve_static(model, loads, assembly=asm)
    tip = float(np.mean([u[asm.dof_map.index(nid, 2)] for nid in tip_nodes]))
    reference = timoshenko_tip_deflection(
        force=float(case.get("tip_force", 1.0)),
        length=float(case.get("length", 10.0)),
        width=float(case.get("width", 1.0)),
        height=float(case.get("height", 1.0)),
        E=float(case.get("E", 1.0e7)),
        nu=float(case.get("nu", 0.3)),
    )
    return tip / reference


def hex8_patch_test_error(
    formulation: str | None = None,
    *,
    distortion: float = 0.3,
    seed: int = 7,
) -> float:
    """Relative error of the enclosed node in a constant-stress patch test.

    A 2x2x2 element block is distorted on every interior plane, the 26 outer
    nodes are driven with an exact linear displacement field and the single
    enclosed node must land on the same field.
    """
    rng = np.random.default_rng(seed)
    coords: dict[tuple[int, int, int], np.ndarray] = {}
    ids: dict[tuple[int, int, int], int] = {}
    nodes: dict[int, Any] = {}
    counter = 1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                point = np.array([float(i), float(j), float(k)])
                # Only interior planes move: the outer surfaces stay planar so
                # the enforced field remains a pure constant-stress state.
                free = np.array([0 < i < 2, 0 < j < 2, 0 < k < 2], dtype=float)
                point = point + distortion * free * rng.uniform(-1.0, 1.0, 3)
                coords[(i, j, k)] = point
                ids[(i, j, k)] = counter
                nodes[counter] = {"xyz": tuple(point)}
                counter += 1

    elements: dict[int, Any] = {}
    eid = 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                elements[eid] = {
                    "type": "HEX8",
                    "property_id": 1,
                    "nodes": (
                        ids[(i, j, k)],
                        ids[(i + 1, j, k)],
                        ids[(i + 1, j + 1, k)],
                        ids[(i, j + 1, k)],
                        ids[(i, j, k + 1)],
                        ids[(i + 1, j, k + 1)],
                        ids[(i + 1, j + 1, k + 1)],
                        ids[(i, j + 1, k + 1)],
                    ),
                }
                eid += 1

    model = _model(nodes, elements, 1.0e7, 0.3, 1.0)
    gradient = np.array(
        [[1.0e-4, 2.0e-5, 3.0e-5], [5.0e-6, -2.0e-4, 1.0e-5], [1.0e-5, 3.0e-5, 1.5e-4]]
    )
    interior = (1, 1, 1)
    enforced: dict[tuple[int, int], float] = {}
    spcs = []
    for key, point in coords.items():
        if key == interior:
            continue
        exact = gradient @ point
        spcs.append({"node_id": ids[key], "dofs": (0, 1, 2)})
        for comp in range(3):
            enforced[(ids[key], comp)] = float(exact[comp])
    model["spcs"] = spcs

    options = {"hex8": formulation} if formulation else None
    asm = assemble_km(model, options=options)
    u = solve_static(model, {}, assembly=asm, enforced=enforced)
    exact = gradient @ coords[interior]
    got = np.array([u[asm.dof_map.index(ids[interior], comp)] for comp in range(3)])
    return float(np.max(np.abs(got - exact)) / np.max(np.abs(exact)))


def hex8_rigid_body_frequencies(
    formulation: str | None = None,
    *,
    nx: int = 3,
    ny: int = 1,
    nz: int = 1,
    n_modes: int = 10,
    **case: Any,
) -> np.ndarray:
    """Frequencies (Hz) of an unconstrained HEX8 block, ascending.

    The first six must be zero and the seventh strictly positive: a formulation
    that has traded shear locking for an hourglass mechanism shows up here as a
    seventh (near) zero frequency.
    """
    model, _, _ = hex_cantilever(nx, ny, nz, clamped=False, **case)
    options = {"hex8": formulation} if formulation else None
    result = solve_modes(model, n_modes=n_modes, options=options)
    return np.asarray(result.freq_hz, dtype=float)


#: Section and material of :func:`beam_cantilever`; the same numbers the
#: BEAM2 golden case in the test suite uses, so frequencies are comparable.
BEAM_CANTILEVER: dict[str, float] = {
    "E": 70.0e9,
    "rho": 2700.0,
    "L": 2.0,
    "A": 8.0e-4,
    "Iy": 3.0e-8,
    "Iz": 6.0e-8,
    "J": 9.0e-8,
}


def beam_cantilever(n_elements: int = 16) -> dict[str, Any]:
    """BEAM2 cantilever with the section of :data:`BEAM_CANTILEVER`.

    The reduction and complex-mode cases below all run on this model: it is
    small enough for a dense reference solve, its spectrum spans eleven decades
    (which is what makes the eigensolver accuracy question interesting) and its
    six lowest frequencies are the ones the analytical golden case already
    pins down.
    """
    d = BEAM_CANTILEVER
    nodes = {
        i + 1: {"xyz": (d["L"] * i / n_elements, 0.0, 0.0)} for i in range(int(n_elements) + 1)
    }
    elements = {
        i + 1: {"type": "BEAM2", "property_id": 1, "nodes": (i + 1, i + 2)}
        for i in range(int(n_elements))
    }
    return {
        "nodes": nodes,
        "elements": elements,
        "materials": {1: {"E": d["E"], "nu": 0.3, "rho": d["rho"]}},
        "properties": {
            1: {
                "type": "beam",
                "material_id": 1,
                "A": d["A"],
                "Iy": d["Iy"],
                "Iz": d["Iz"],
                "J": d["J"],
            }
        },
        "spcs": [{"node_id": 1, "dofs": (0, 1, 2, 3, 4, 5)}],
    }


def _cantilever_system(n_elements: int = 16, *, every: int = 2):
    """``(Kff, Mff, master_positions)`` for the reduction cases.

    The masters are the two lateral translations of every ``every``-th node,
    i.e. the DOFs a shaker test would actually instrument.
    """
    model = beam_cantilever(n_elements)
    asm = assemble_km(model)
    position = {int(g): i for i, g in enumerate(asm.free_dof)}
    master = [
        position[asm.dof_map.index(nid, comp)]
        for nid in range(2, int(n_elements) + 2, int(every))
        for comp in (1, 2)
        if asm.dof_map.index(nid, comp) in position
    ]
    return asm, asm.Kff.toarray(), asm.Mff.toarray(), np.array(master, dtype=int)


def guyan_condensation_error(n_elements: int = 16, *, every: int = 2) -> float:
    """``max|T^T K T - (K_mm - K_ms K_ss^-1 K_sm)| / max|Schur|``.

    Zero to round-off by construction: this is the identity that says the Guyan
    basis really is static condensation and not merely something close to it.
    """
    _asm, K, _M, master = _cantilever_system(n_elements, every=every)
    result = guyan(K, master)
    m, s = result.master, result.slave
    schur = K[np.ix_(m, m)] - K[np.ix_(m, s)] @ np.linalg.solve(
        K[np.ix_(s, s)], K[np.ix_(s, m)]
    )
    return float(np.max(np.abs(result.K_red - schur)) / np.max(np.abs(schur)))


def reduction_frequency_errors(
    n_elements: int = 16, *, every: int = 2, n_modes: int = 6
) -> dict[str, float]:
    """Worst relative frequency error of each reduction over the first modes.

    Returns ``{"guyan": ..., "irs": ..., "serep": ...}``.  The ordering is the
    point: Guyan over-predicts (it throws the slave inertia away), IRS recovers
    most of that, and SEREP is exact because its basis is built from the modes
    being compared against.
    """
    _asm, K, M, master = _cantilever_system(n_elements, every=every)
    # One eigensolve for both the reference frequencies and the SEREP basis:
    # two calls would differ in the last bits and put a 1e-10 floor under the
    # SEREP row, which is exact to round-off.
    lam, phi = sla.eigh(K, M)
    exact = np.sqrt(np.clip(lam, 0.0, None)) / (2.0 * np.pi)
    modes = phi[:, : int(n_modes)]

    out: dict[str, float] = {}
    for name, reduced in (
        ("guyan", guyan(K, master, M)),
        ("irs", irs(K, M, master)),
        ("serep", serep(modes, master, K, M)),
    ):
        got, _ = reduced.reduced_modes()
        k = min(int(n_modes), got.size)
        out[name] = float(np.max(np.abs(got[:k] - exact[:k]) / exact[:k]))
    return out


def serep_slave_recovery(
    n_elements: int = 16, *, every: int = 4, n_modes: int = 6
) -> dict[str, float]:
    """How well SEREP rebuilds the *unmeasured* DOFs from the measured ones.

    Returns the relative error on the slave partition and the worst per-mode
    MAC there.  With at least ``n_modes`` independent sensors the recovery is
    exact, which is the property that separates SEREP from the static bases:
    the same table reports what Guyan and IRS manage on the same sensor set.
    """
    _asm, K, M, master = _cantilever_system(n_elements, every=every)
    modes = sla.eigh(K, M)[1][:, : int(n_modes)]
    reduced = serep(modes, master, K, M)
    slave = reduced.slave

    recovered = reduced.T @ modes[reduced.master, :]
    a, b = recovered[slave], modes[slave]
    mac = np.array(
        [
            (a[:, j] @ b[:, j]) ** 2 / ((a[:, j] @ a[:, j]) * (b[:, j] @ b[:, j]))
            for j in range(modes.shape[1])
        ]
    )
    static = {
        "guyan": guyan(K, master, M),
        "irs": irs(K, M, master),
    }
    return {
        "n_master": float(master.size),
        "n_slave": float(slave.size),
        "serep_slave_error": float(
            np.linalg.norm(a - b) / np.linalg.norm(b)
        ),
        "serep_worst_mac": float(mac.min()),
        "guyan_slave_error": float(
            np.linalg.norm(static["guyan"].T @ modes[master] - modes) / np.linalg.norm(modes)
        ),
        "irs_slave_error": float(
            np.linalg.norm(static["irs"].T @ modes[master] - modes) / np.linalg.norm(modes)
        ),
    }


def complete_spectrum_quality(n_elements: int = 16) -> dict[str, float]:
    """Backward error of a *complete* modal basis of the beam cantilever.

    ``residual`` is ``max ||K phi - lambda M phi|| / ||K||``, ``orthogonality``
    is ``max|Phi^T M Phi - I|`` and ``k_diagonality`` the largest off-diagonal
    of ``Phi^T K Phi`` relative to its diagonal.  All three are at round-off
    once the complete spectrum is taken through ``scipy.linalg.eigh`` instead
    of shift-invert; the third is the one that decides whether a full modal
    superposition reproduces a direct solve.
    """
    model = beam_cantilever(n_elements)
    asm = assemble_km(model)
    K = asm.Kff.toarray()
    M = asm.Mff.toarray()
    result = solve_modes(model, n_modes=K.shape[0], assembly=asm)
    phi = mass_normalize(result.modes[asm.free_dof], asm.Mff)
    lam = result.eigenvalues

    kphi = phi.T @ K @ phi
    diagonal = np.abs(np.diag(kphi)).max()
    return {
        "n_free": float(K.shape[0]),
        "residual": float(
            np.max(np.linalg.norm(K @ phi - (M @ phi) * lam, axis=0)) / np.linalg.norm(K, 2)
        ),
        "orthogonality": float(np.max(np.abs(phi.T @ M @ phi - np.eye(phi.shape[1])))),
        "k_diagonality": float(
            np.max(np.abs(kphi - np.diag(np.diag(kphi)))) / diagonal
        ),
        "condition": float(lam.max() / lam.min()),
    }


def hex8_jacobian_spread(model: Any) -> float:
    """Worst ``max|det J| / min|det J|`` over the Gauss points of any HEX8.

    This is the mesh-quality number that predicts how much of the
    incompatible-mode advantage survives (see the module docstring).  It is
    1.0 (to round-off) for any parallelepiped, however skewed or stretched,
    because the Taylor scaling is then exact; it grows as the element becomes
    trapezoidal and the internal gradients stop being correct away from the
    centre.  Returns 1.0 for a model without HEX8 elements.
    """
    index = ModelIndex.build(model)
    derivatives = [_hex_shape(*point)[1] for point in gauss_3d(2)[0]]
    worst = 1.0
    for _eid, element in iter_records(get_any(model, ("elements", "elems", "element"), None)):
        etype = str(get_any(element, ("type", "etype", "element_type", "kind"), "")).strip()
        try:
            spec = element_spec(etype)
        except KeyError:
            continue
        if spec.name != "HEX8":
            continue
        conn = get_any(element, ("nodes", "node_ids", "connectivity", "conn", "grids"), ())
        xyz = np.array([index.xyz(nid) for nid in tuple(conn)[:8]], dtype=float)
        dets = np.abs([np.linalg.det(dn.T @ xyz) for dn in derivatives])
        if dets.min() > 0.0:
            worst = max(worst, float(dets.max() / dets.min()))
    return worst
