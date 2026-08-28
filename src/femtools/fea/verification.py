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

Enhanced assumed strain is not the way out of that: ``hex8_eas_equivalence``
builds the Simo-Rifai EAS-9 brick independently and shows it is the shipped
incompatible-modes element to round-off on any hexahedron, trapezoids included.

Shell orientation
-----------------

``shell_drilling_orientation_gap`` runs the same flat plate twice, once in a
global plane and once tilted onto an oblique normal, and reports that the two
now agree in every respect -- six rigid body modes, the same solved set and the
same elastic frequencies.  Holding that for the tilted plate is what the
per-node rotational frames of :mod:`femtools.fea.nodal_frames` are for.

Stress recovery and rigid bodies
--------------------------------

``stress_patch_error`` is the constant-strain patch test of
:mod:`femtools.fea.recover`, run for each of the structural element types of
:data:`PATCH_TYPES` on a deliberately irregular mesh: the boundary is driven
with an exact linear field, the enclosed node is left free and the recovered
centroid stress of every element is compared with the analytic constant state.
The same figure is reported after ``average_nodal`` and after ``recover_spr``
have put the element values on the nodes, since neither smoothing is allowed
to damage a state that is already constant.

``tet_patch_model`` builds the tetrahedral half of that: four tetrahedra
filling one irregular outer tetrahedron, as ``TET4`` or as ``TET10``.
``tet10_rigid_body_frequencies`` runs the same block free-free, where the
four-point quadrature of the quadratic tetrahedron has to leave exactly six
zero frequencies -- it is the minimum rule that can.

``rbe2_rigid_pair`` and ``rbe2_offset_moment`` are the two statements a rigid
body element has to satisfy (:mod:`femtools.fea.mpc`): welding two nodes leaves
a free-free structure with exactly six rigid body modes and the analytic rigid
body mass matrix, and a load on a rigid offset arrives at the independent node
as a force *and* the moment of the offset.

``rbe3_spider`` and ``rbe3_load_path`` are the corresponding pair for the
*interpolation* constraint, which is a different statement and not a rigid weld:
a mass hung on the reference grid of a free-free spider leaves exactly six
rigid body modes and arrives in full at the weighted centroid of the
independents, and a force on that grid is shared out in proportion to the
weights -- equally, for equal weights, whatever the geometry.

``mpc_mixed_chain`` runs the two card types in one model, hanging a rigid arm
off an interpolated reference grid and (with ``direction="rbe3_on_rbe2"``) the
other way round.  :func:`femtools.fea.mpc.apply_mpc` composes them into one
transform, so the case is what says the composition is still exact: six rigid
body modes and not a seventh, no stiffness anywhere on the constraint, and the
hung mass delivered in full to the point the two kinematics carry it to.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.linalg as sla

from .assemble import assemble_km
from .eigen import mass_normalize, solve_modes
from .elements import ModelIndex, element_matrices, element_spec
from .elements.solid import TET10_EDGES, _hex_shape, _strain_matrix
from .materials import MaterialData, plane_stress_D, solid_D
from .protocols import get_any, iter_records
from .quadrature import gauss_3d
from .reduction import guyan, irs, serep
from .static import solve_static

__all__ = [
    "DISTORTIONS",
    "MPC_CHAIN_DIRECTIONS",
    "PATCH_GRADIENT",
    "PATCH_TYPES",
    "beam_cantilever",
    "complete_spectrum_quality",
    "guyan_condensation_error",
    "hex8_bending_ratio",
    "hex8_eas_equivalence",
    "hex8_jacobian_spread",
    "hex8_patch_test_error",
    "hex8_rigid_body_frequencies",
    "hex_cantilever",
    "mpc_mixed_chain",
    "rbe2_offset_moment",
    "rbe2_rigid_pair",
    "rbe3_load_path",
    "rbe3_spider",
    "reduction_frequency_errors",
    "serep_slave_recovery",
    "shell_plate",
    "shell_drilling_orientation_gap",
    "stress_patch_error",
    "tet10_rigid_body_frequencies",
    "tet_bending_ratio",
    "tet_cantilever",
    "tet_patch_model",
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


def _distorted_hex_patch(
    distortion: float, seed: int
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[tuple[int, int, int], int], dict, dict]:
    """The 2x2x2 HEX8 patch block: ``(coords, ids, nodes, elements)``.

    Every interior plane is displaced at random; the outer surfaces stay planar
    so that an enforced linear field on them remains a pure constant-stress
    state.  Shared by :func:`hex8_patch_test_error` and
    :func:`stress_patch_error`.
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
    return coords, ids, nodes, elements


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
    coords, ids, nodes, elements = _distorted_hex_patch(distortion, seed)
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


def shell_plate(
    nx: int = 3,
    ny: int = 3,
    *,
    etype: str = "QUAD4",
    side: float = 1.0,
    thickness: float = 0.01,
    E: float = 70.0e9,
    nu: float = 0.3,
    rho: float = 2700.0,
    rotation: np.ndarray | None = None,
    clamped_edge: bool = False,
) -> dict[str, Any]:
    """Flat square shell mesh, in the global x-y plane unless *rotation* says otherwise.

    ``rotation`` is a ``(3, 3)`` orthogonal matrix applied to every node, which
    turns the same structure into a mesh whose normal is no longer a global
    axis -- the configuration :func:`shell_drilling_orientation_gap` measures.
    ``etype`` is ``"QUAD4"`` or ``"TRIA3"`` (each cell split into two triangles).
    """
    ids: dict[tuple[int, int], int] = {}
    nodes: dict[int, Any] = {}
    counter = 1
    R = None if rotation is None else np.asarray(rotation, dtype=float)
    for i in range(nx + 1):
        for j in range(ny + 1):
            point = np.array([side * i / nx, side * j / ny, 0.0])
            if R is not None:
                point = R @ point
            ids[(i, j)] = counter
            nodes[counter] = {"xyz": tuple(point)}
            counter += 1

    elements: dict[int, Any] = {}
    eid = 1
    for i in range(nx):
        for j in range(ny):
            n1, n2, n3, n4 = ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)], ids[(i, j + 1)]
            if str(etype).upper() == "TRIA3":
                for conn in ((n1, n2, n3), (n1, n3, n4)):
                    elements[eid] = {"type": "TRIA3", "property_id": 1, "nodes": conn}
                    eid += 1
            else:
                elements[eid] = {"type": "QUAD4", "property_id": 1, "nodes": (n1, n2, n3, n4)}
                eid += 1

    spcs: list[Any] = []
    if clamped_edge:
        spcs = [
            {"node_id": ids[(0, j)], "dofs": (0, 1, 2, 3, 4, 5)} for j in range(ny + 1)
        ]
    return {
        "nodes": nodes,
        "elements": elements,
        "materials": {1: {"E": E, "nu": nu, "rho": rho}},
        "properties": {1: {"type": "shell", "material_id": 1, "t": thickness}},
        "spcs": spcs,
    }


#: Rotation used by :func:`shell_drilling_orientation_gap`: a plate normal that
#: is deliberately not parallel to any global axis.
_OBLIQUE = np.linalg.qr(
    np.array([[0.8, 0.3, -0.5], [-0.2, 0.9, 0.4], [0.6, -0.1, 0.7]])
)[0]


def shell_drilling_orientation_gap(
    etype: str = "QUAD4",
    *,
    nx: int = 3,
    ny: int = 3,
    n_modes: int = 9,
    nodal_frames: bool = True,
) -> dict[str, float]:
    """Does the free-free spectrum of one flat plate depend on its orientation?

    It must not, and the interesting part is that making it not depend on the
    orientation takes work.  A flat shell has no genuine stiffness about its own
    normal, so ``TRIA3`` and ``QUAD4`` add a rank deficient drilling penalty and
    the assembler drops the drilling DOFs that receive nothing else.  Dropping
    one *global* rotation is only possible while the normal is a global axis;
    for any other orientation the drilling direction is a mix of ``rx``, ``ry``
    and ``rz``, and a mesh assembled in the basic frame keeps a zero-energy
    drilling mechanism that reads as a seventh rigid body mode.  The assembler
    avoids that by solving the rotations of each shell node in a triad built on
    its averaged normal (:mod:`femtools.fea.nodal_frames`), so the drilling
    rotation is one degree of freedom again.

    Returns, for the axis-aligned and the rotated copy of the same plate, the
    number of (near) zero frequencies, the first elastic frequency, the size of
    the solved set, how many drilling DOFs were removed, how many nodes needed
    a local triad and whether ``assemble_km`` warned.  Everything except
    ``*_frame_nodes`` now agrees between the two, which is the acceptance
    statement: same plate, same spectrum, six rigid body modes either way.

    Pass ``nodal_frames=False`` to reproduce the pre-frame behaviour -- seven
    zero frequencies and a warning on the oblique plate.
    """
    out: dict[str, float] = {}
    for label, rotation in (("aligned", None), ("oblique", _OBLIQUE)):
        model = shell_plate(nx, ny, etype=etype, rotation=rotation)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            asm = assemble_km(model, nodal_frames=nodal_frames)
        result = solve_modes(model, n_modes=n_modes, assembly=asm)
        freq = np.asarray(result.freq_hz, dtype=float)
        elastic = freq[freq > 1.0e-6 * max(freq.max(), 1.0)]
        out[f"{label}_zero_modes"] = float(freq.size - elastic.size)
        out[f"{label}_first_elastic_hz"] = float(elastic[0]) if elastic.size else float("nan")
        out[f"{label}_free_dof"] = float(asm.n_free)
        out[f"{label}_drilling_dof"] = float(asm.drilling_dof.size)
        out[f"{label}_frame_nodes"] = float(len(asm.framed_nodes))
        out[f"{label}_warned"] = float(len(caught))
    return out


#: Natural-strain interpolation of the nine parameter Simo-Rifai enhancement,
#: rows in the Voigt order ``(11, 22, 33, 12, 23, 31)``: one linear term per
#: normal strain and the two "missing" linear terms per shear strain.  This is
#: the smallest set that satisfies ``\int_\square M d\xi = 0`` (the patch test
#: condition) while spanning the bending modes the trilinear field lacks.
_EAS9_TERMS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 0),
    (3, 1),
    (4, 1),
    (4, 2),
    (5, 0),
    (5, 2),
)


def _eas9_interpolation(natural: np.ndarray) -> np.ndarray:
    M = np.zeros((6, len(_EAS9_TERMS)))
    for column, (row, coordinate) in enumerate(_EAS9_TERMS):
        M[row, column] = natural[coordinate]
    return M


def _natural_strain_transform(J0: np.ndarray) -> np.ndarray:  # noqa: N803
    """The 6x6 ``T0`` with ``E_natural = T0 @ eps_cartesian`` for the Voigt order above.

    ``E_ab = dx_i/dxi_a * eps_ij * dx_j/dxi_b`` written out with engineering
    shear strains, i.e. ``T0`` is built from products of the Jacobian entries at
    the element centre.  Simo and Rifai map the enhancement the other way, so
    the element uses its inverse.
    """
    j = np.asarray(J0, dtype=float)

    def row(a: int, b: int) -> list[float]:
        shear = 1.0 if a == b else 2.0
        return [
            shear * j[a, 0] * j[b, 0],
            shear * j[a, 1] * j[b, 1],
            shear * j[a, 2] * j[b, 2],
            j[a, 0] * j[b, 1] + j[a, 1] * j[b, 0] if a != b else j[a, 0] * j[a, 1],
            j[a, 1] * j[b, 2] + j[a, 2] * j[b, 1] if a != b else j[a, 1] * j[a, 2],
            j[a, 0] * j[b, 2] + j[a, 2] * j[b, 0] if a != b else j[a, 0] * j[a, 2],
        ]

    return np.array([row(0, 0), row(1, 1), row(2, 2), row(0, 1), row(1, 2), row(0, 2)])


def hex8_eas9_stiffness(xyz: np.ndarray, E: float = 1.0e7, nu: float = 0.3) -> np.ndarray:
    """Reference Simo-Rifai enhanced assumed strain (EAS-9) brick stiffness.

    An independent 24x24 implementation of

    .. math:: \\tilde\\varepsilon(\\xi) = \\frac{\\det J_0}{\\det J(\\xi)}\\,
              T_0^{-1} M(\\xi)\\, \\alpha

    (Simo & Rifai 1990, *A class of mixed assumed strain methods*, IJNME 29;
    the hexahedral extension of Simo & Armero 1992 / Andelfinger & Ramm 1993),
    with the nine amplitudes ``alpha`` statically condensed out.  It exists to
    be *compared* with the shipped element rather than to be used: see
    :func:`hex8_eas_equivalence`.
    """
    xyz = np.asarray(xyz, dtype=float)[:8]
    D = solid_D(MaterialData(E=float(E), nu=float(nu)))
    points, weights = gauss_3d(2)

    _, dn0 = _hex_shape(0.0, 0.0, 0.0)
    J0 = dn0.T @ xyz
    det0 = float(np.linalg.det(J0))
    T0_inv = np.linalg.inv(_natural_strain_transform(J0))

    n_alpha = len(_EAS9_TERMS)
    k = np.zeros((24, 24))
    k_ua = np.zeros((24, n_alpha))
    k_aa = np.zeros((n_alpha, n_alpha))
    for point, weight in zip(points, weights, strict=True):
        _, dn = _hex_shape(*point)
        J = dn.T @ xyz
        det = float(np.linalg.det(J))
        B = _strain_matrix(np.linalg.solve(J, dn.T).T)
        G = (det0 / det) * (T0_inv @ _eas9_interpolation(np.asarray(point, dtype=float)))
        scale = weight * abs(det)
        DB = D @ B
        DG = D @ G
        k += scale * (B.T @ DB)
        k_ua += scale * (B.T @ DG)
        k_aa += scale * (G.T @ DG)
    return k - k_ua @ np.linalg.solve(k_aa, k_ua.T)


def hex8_eas_equivalence(*, seed: int = 3, trials: int = 4, distortion: float = 0.25) -> dict:
    """Is the shipped incompatible-modes brick the Simo-Rifai EAS-9 element?

    It is, exactly -- which is the reason femtools does not carry an EAS option
    alongside the default.  The two derivations look nothing alike (one adds
    nine internal *displacement* amplitudes and corrects their gradients by
    ``det J0 / det J``, the other adds nine *strain* amplitudes mapped through
    the natural-strain transform of the element centre), but they condense to
    the same 24x24 matrix on any hexahedron.  The equivalence is also the
    reason EAS is not a cure for the trapezoidal distortion loss documented in
    the module docstring: the distorted element it would replace is itself.

    Returns the worst relative difference over ``trials`` randomly distorted
    bricks plus the three named shapes, and -- as the control that keeps the
    comparison honest -- the same figure for the *transposed* natural-strain
    convention, which is a genuinely different element.
    """
    rng = np.random.default_rng(int(seed))
    unit = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    shapes = [unit.copy()]
    skewed = unit.copy()
    skewed[:, 0] += 0.4 * skewed[:, 2]
    shapes.append(skewed)
    trapezoid = unit.copy()
    trapezoid[:, 0] += 0.3 * (trapezoid[:, 2] - 0.5) * (2.0 * trapezoid[:, 0] - 1.0)
    shapes.append(trapezoid)
    shapes += [unit + distortion * rng.uniform(-1.0, 1.0, (8, 3)) for _ in range(int(trials))]

    worst = 0.0
    worst_control = 0.0
    zero_modes = 0
    for xyz in shapes:
        model = _model(
            {i + 1: {"xyz": tuple(xyz[i])} for i in range(8)},
            {1: {"type": "HEX8", "property_id": 1, "nodes": tuple(range(1, 9))}},
            1.0e7,
            0.3,
            1.0,
        )
        shipped = np.asarray(element_matrices(model, 1, model["elements"][1]).k, dtype=float)
        reference = hex8_eas9_stiffness(xyz)
        scale = float(np.max(np.abs(shipped)))
        worst = max(worst, float(np.max(np.abs(reference - shipped))) / scale)

        eigenvalues = np.linalg.eigvalsh(0.5 * (reference + reference.T))
        zero_modes = max(
            zero_modes, int(np.count_nonzero(np.abs(eigenvalues) < 1.0e-8 * eigenvalues.max()))
        )

        control = _hex8_eas9_transposed(xyz)
        worst_control = max(worst_control, float(np.max(np.abs(control - shipped))) / scale)

    return {
        "n_shapes": float(len(shapes)),
        "worst_relative_difference": worst,
        "worst_relative_difference_transposed": worst_control,
        "eas9_zero_modes": float(zero_modes),
    }


def _hex8_eas9_transposed(xyz: np.ndarray) -> np.ndarray:
    """EAS-9 built with ``T0^-T`` instead of ``T0^-1``: a different element.

    Only here so :func:`hex8_eas_equivalence` can show that its agreement is an
    algebraic identity and not an insensitivity of the comparison.
    """
    xyz = np.asarray(xyz, dtype=float)[:8]
    D = solid_D(MaterialData(E=1.0e7, nu=0.3))
    points, weights = gauss_3d(2)
    _, dn0 = _hex_shape(0.0, 0.0, 0.0)
    J0 = dn0.T @ xyz
    det0 = float(np.linalg.det(J0))
    T0_inv_t = np.linalg.inv(_natural_strain_transform(J0)).T

    n_alpha = len(_EAS9_TERMS)
    k = np.zeros((24, 24))
    k_ua = np.zeros((24, n_alpha))
    k_aa = np.zeros((n_alpha, n_alpha))
    for point, weight in zip(points, weights, strict=True):
        _, dn = _hex_shape(*point)
        J = dn.T @ xyz
        det = float(np.linalg.det(J))
        B = _strain_matrix(np.linalg.solve(J, dn.T).T)
        G = (det0 / det) * (T0_inv_t @ _eas9_interpolation(np.asarray(point, dtype=float)))
        scale = weight * abs(det)
        k += scale * (B.T @ D @ B)
        k_ua += scale * (B.T @ D @ G)
        k_aa += scale * (G.T @ D @ G)
    return k - k_ua @ np.linalg.solve(k_aa, k_ua.T)


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


# ---------------------------------------------------------------------------
# stress recovery
# ---------------------------------------------------------------------------

#: Element types :func:`stress_patch_error` covers, in the order documented.
PATCH_TYPES: tuple[str, ...] = ("BAR2", "BEAM2", "TRIA3", "QUAD4", "TET4", "TET10", "HEX8")

#: Displacement gradient of the solid and shell patch tests.  Deliberately
#: unsymmetric, so a recovery that dropped the rotational part of the gradient
#: would show up rather than cancel.
PATCH_GRADIENT = np.array(
    [[1.0e-4, 2.0e-5, 3.0e-5], [5.0e-6, -2.0e-4, 1.0e-5], [1.0e-5, 3.0e-5, 1.5e-4]]
)

#: Section and material of the line-element patch cases.
_PATCH_LINE = {"E": 2.1e11, "nu": 0.3, "rho": 7800.0, "A": 3.0e-4, "I": 2.0e-8, "J": 4.0e-8}


def _line_patch_model(etype: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[int]]:
    """Three unequal ``BAR2``/``BEAM2`` elements on one oblique line.

    Returns the model, the unit axis, the node positions along it and the two
    end node ids.  Unequal spacing is the point: a recovery that assumed a
    uniform mesh would be caught by it.
    """
    axis = np.array([2.0, -1.0, 0.5])
    axis = axis / np.linalg.norm(axis)
    stations = np.array([0.0, 0.37, 0.71, 1.0]) * 1.5
    nodes = {i + 1: {"xyz": tuple(axis * s)} for i, s in enumerate(stations)}
    elements = {
        i + 1: {"type": etype, "property_id": 1, "nodes": (i + 1, i + 2)} for i in range(3)
    }
    d = _PATCH_LINE
    prop: dict[str, Any] = {"type": "bar", "material_id": 1, "A": d["A"]}
    if etype == "BEAM2":
        prop.update({"type": "beam", "Iy": d["I"], "Iz": d["I"], "J": d["J"]})
    model = {
        "nodes": nodes,
        "elements": elements,
        "materials": {1: {"E": d["E"], "nu": d["nu"], "rho": d["rho"]}},
        "properties": {1: prop},
        "spcs": [],
    }
    return model, axis, stations, [1, len(stations)]


def _shell_patch_model(etype: str, distortion: float) -> tuple[dict[str, Any], dict, list[int]]:
    """3x3 node membrane patch in the global x-y plane with a moved interior node."""
    ids: dict[tuple[int, int], int] = {}
    coords: dict[int, np.ndarray] = {}
    nodes: dict[int, Any] = {}
    counter = 1
    for i in range(3):
        for j in range(3):
            point = np.array([float(i), float(j), 0.0])
            if i == 1 and j == 1:
                point = point + distortion * np.array([0.31, -0.24, 0.0])
            ids[(i, j)] = counter
            coords[counter] = point
            nodes[counter] = {"xyz": tuple(point)}
            counter += 1

    elements: dict[int, Any] = {}
    eid = 1
    for i in range(2):
        for j in range(2):
            n1, n2, n3, n4 = ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)], ids[(i, j + 1)]
            if etype == "TRIA3":
                for conn in ((n1, n2, n3), (n1, n3, n4)):
                    elements[eid] = {"type": "TRIA3", "property_id": 1, "nodes": conn}
                    eid += 1
            else:
                elements[eid] = {"type": "QUAD4", "property_id": 1, "nodes": (n1, n2, n3, n4)}
                eid += 1

    model = {
        "nodes": nodes,
        "elements": elements,
        "materials": {1: {"E": 70.0e9, "nu": 0.3, "rho": 2700.0}},
        "properties": {1: {"type": "shell", "material_id": 1, "t": 0.02}},
        "spcs": [],
    }
    boundary = [nid for key, nid in ids.items() if key != (1, 1)]
    return model, coords, boundary


def tet_patch_model(
    etype: str = "TET4", *, distortion: float = 0.3
) -> tuple[dict[str, Any], dict, list[int]]:
    """Four tetrahedra filling one irregular outer tetrahedron around a free node.

    Returns ``(model, coords, boundary)``: the four corners of the outer
    tetrahedron are its whole surface for ``TET4``, so they are the boundary
    and the enclosed fifth node is the one the patch test solves for.

    ``etype="TET10"`` adds the midside node of every edge, which makes the
    patch a much harder case than a count of elements suggests: the boundary
    now carries the midsides of the six outer edges as well, and **five** nodes
    are left free -- the enclosed corner and the midsides of the four interior
    edges.  The midside nodes sit exactly at the middle of their edge, so every
    element keeps a constant Jacobian; a ``TET10`` with genuinely curved edges
    only passes the patch test to the accuracy of its quadrature, which is a
    property of the element and not of this mesh (see :func:`tet10`).
    """
    name = str(etype).strip().upper()
    if name not in ("TET4", "TET10"):
        raise ValueError(f"unknown tetrahedron type {etype!r}; expected TET4 or TET10")

    outer = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.2, 1.1, 0.0], [0.4, 0.3, 1.2]])
    interior = outer.mean(axis=0) + distortion * np.array([0.11, -0.09, 0.07])
    coords = {i + 1: p for i, p in enumerate(np.vstack([outer, interior]))}
    faces = ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
    corner_sets = [(*face, 5) for face in faces]
    boundary = [1, 2, 3, 4]

    if name == "TET4":
        elements = {
            eid + 1: {"type": "TET4", "property_id": 1, "nodes": conn}
            for eid, conn in enumerate(corner_sets)
        }
    else:
        outer_edges = {frozenset(pair) for face in faces for pair in
                       ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))}
        midside: dict[frozenset[int], int] = {}
        elements = {}
        for eid, conn in enumerate(corner_sets, start=1):
            nodes_of = list(conn)
            for a, b in TET10_EDGES:
                key = frozenset((conn[a], conn[b]))
                nid = midside.get(key)
                if nid is None:
                    nid = midside[key] = len(coords) + 1
                    coords[nid] = 0.5 * (coords[conn[a]] + coords[conn[b]])
                    if key in outer_edges:
                        boundary.append(nid)
                nodes_of.append(nid)
            elements[eid] = {"type": "TET10", "property_id": 1, "nodes": tuple(nodes_of)}

    nodes = {nid: {"xyz": tuple(float(v) for v in point)} for nid, point in coords.items()}
    return _model(nodes, elements, 1.0e7, 0.3, 1.0), coords, boundary


#: Six tetrahedra filling one hexahedron, all sharing its ``0-6`` diagonal
#: (Kuhn's decomposition).  Applied with the same diagonal in every cell it is
#: conforming: neighbouring cells split their shared face the same way.
_HEX_TO_TETS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
)


def tet_cantilever(
    nx: int = 10,
    ny: int = 1,
    nz: int = 1,
    *,
    etype: str = "TET4",
    **case: Any,
) -> tuple[dict[str, Any], list[int], dict[tuple[int, int], float]]:
    """:func:`hex_cantilever` with every brick split into six tetrahedra.

    Same geometry, same clamp, same tip load, so the tip ratio is directly
    comparable with the hexahedral one -- which is the point: a tetrahedral
    mesh is what an automatic mesher produces, and the two tet types answer
    very differently on it.  ``etype`` is ``"TET4"`` or ``"TET10"``; the
    quadratic mesh adds one node at the middle of every edge, shared between
    the elements that meet on it.
    """
    name = str(etype).strip().upper()
    if name not in ("TET4", "TET10"):
        raise ValueError(f"unknown tetrahedron type {etype!r}; expected TET4 or TET10")

    bricks, _brick_tip, _brick_loads = hex_cantilever(nx, ny, nz, **case)
    coords = {nid: np.asarray(node["xyz"], dtype=float) for nid, node in bricks["nodes"].items()}
    midside: dict[frozenset[int], int] = {}
    elements: dict[int, Any] = {}
    eid = 1
    for brick in bricks["elements"].values():
        corners = tuple(brick["nodes"])
        for tet in _HEX_TO_TETS:
            conn = [corners[m] for m in tet]
            if name == "TET10":
                for a, b in TET10_EDGES:
                    key = frozenset((conn[a], conn[b]))
                    nid = midside.get(key)
                    if nid is None:
                        nid = midside[key] = len(coords) + 1
                        coords[nid] = 0.5 * (coords[conn[a]] + coords[conn[b]])
                    conn.append(nid)
            elements[eid] = {"type": name, "property_id": 1, "nodes": tuple(conn)}
            eid += 1

    nodes = {nid: {"xyz": tuple(float(v) for v in point)} for nid, point in coords.items()}
    model = {**bricks, "nodes": nodes, "elements": elements}
    # The clamp and the tip load follow the geometry, so the midside nodes on
    # the two end faces have to join them or the case is not the same case.
    length = float(case.get("length", 10.0))
    if bricks["spcs"]:
        model["spcs"] = [
            {"node_id": nid, "dofs": (0, 1, 2)}
            for nid, point in coords.items()
            if abs(point[0]) < 1.0e-12
        ]
    tip = [nid for nid, point in coords.items() if abs(point[0] - length) < 1.0e-12]
    force = float(case.get("tip_force", 1.0))
    return model, tip, {(nid, 2): force / len(tip) for nid in tip}


def tet_bending_ratio(
    etype: str = "TET10", *, nx: int = 10, ny: int = 1, nz: int = 1, **case: Any
) -> float:
    """Mean tip deflection of :func:`tet_cantilever` over the Timoshenko value.

    The measurement the quadratic tetrahedron exists for.  On the ten-by-one
    mesh the constant-strain tetrahedron reaches 0.219 of the reference -- a
    tetrahedral mesh of ``TET4`` is not a bending model at all -- while the
    same mesh read as ``TET10`` reaches 0.976, within a hair of the 0.985 the
    incompatible-mode brick manages with an eighth of the elements.
    """
    model, tip_nodes, loads = tet_cantilever(nx, ny, nz, etype=etype, **case)
    asm = assemble_km(model)
    u = solve_static(model, loads, assembly=asm)
    tip = float(np.mean([u[asm.dof_map.index(nid, 2)] for nid in tip_nodes]))
    return tip / timoshenko_tip_deflection(
        force=float(case.get("tip_force", 1.0)),
        length=float(case.get("length", 10.0)),
        width=float(case.get("width", 1.0)),
        height=float(case.get("height", 1.0)),
        E=float(case.get("E", 1.0e7)),
        nu=float(case.get("nu", 0.3)),
    )


def tet10_rigid_body_frequencies(
    *, distortion: float = 0.3, n_modes: int = 10
) -> np.ndarray:
    """Frequencies (Hz) of the unconstrained ``TET10`` patch block, ascending.

    The four-point rule is the *minimum* one that leaves a ten-node tetrahedron
    with rank 24 out of 30, so this is where a quadrature that had been reduced
    one point too far would announce itself: the first six frequencies must be
    zero and the seventh strictly positive.
    """
    model, _coords, _boundary = tet_patch_model("TET10", distortion=distortion)
    return np.asarray(solve_modes(model, n_modes=n_modes).freq_hz, dtype=float)


def stress_patch_error(etype: str = "HEX8", **case: Any) -> dict[str, float]:
    """Constant-strain patch test of :func:`femtools.fea.recover.recover_stress`.

    A small distorted mesh of *etype* is driven on its boundary with an exact
    linear displacement field, the enclosed node (or nodes) are left free, and
    the recovered centroid stress of **every** element is compared with the
    analytic constant state:

    * solids get the full linear field ``u = PATCH_GRADIENT @ x``;
    * shells get its in-plane part, which is a plane-stress membrane state;
    * line elements get uniform axial extension along their own axis, the only
      constant-strain state a rod or a beam has.

    Comparison is made in the basic frame, so it also checks the element frames
    the recovery reports.  Returns the worst relative error over all elements
    of ``stress`` and ``strain`` plus ``displacement``, the classical patch
    test measure on the free node, and ``nodal``, the same stress error after
    :func:`femtools.fea.recover.average_nodal` has smoothed it onto the nodes
    -- a constant state is the one field an average cannot damage, so it has to
    survive to the same precision.  ``spr`` is that last figure once more for
    :func:`femtools.fea.recover.recover_spr`, which has to reproduce a constant
    exactly for a different reason: the polynomial it fits over each patch is
    then the constant itself.
    """
    from .recover import average_nodal, recover_spr, recover_stress  # loaded on demand

    name = str(etype).strip().upper()
    if name not in PATCH_TYPES:
        raise ValueError(f"unknown patch type {etype!r}; expected one of {PATCH_TYPES}")
    distortion = float(case.get("distortion", 0.3))
    seed = int(case.get("seed", 7))

    if name in ("BAR2", "BEAM2"):
        model, axis, stations, ends = _line_patch_model(name)
        strain = 1.5e-4
        enforced = {
            (nid, comp): float(strain * stations[nid - 1] * axis[comp])
            for nid in ends
            for comp in range(3)
        }
        spcs = [
            {"node_id": nid, "dofs": tuple(range(6 if name == "BEAM2" else 3))} for nid in ends
        ]
        if name == "BEAM2":
            enforced.update({(nid, comp): 0.0 for nid in ends for comp in (3, 4, 5)})
        else:
            # A chain of pin-jointed rods is a mechanism transverse to its own
            # axis, so the interior nodes are held there (at the exact value)
            # and only their axial motion is solved for -- the 1D patch test.
            for nid in (2, 3):
                spcs.append({"node_id": nid, "dofs": (1, 2)})
                exact = strain * stations[nid - 1] * axis
                enforced.update({(nid, comp): float(exact[comp]) for comp in (1, 2)})
        model["spcs"] = spcs
        exact_stress = _PATCH_LINE["E"] * strain * np.outer(axis, axis)
        exact_strain = strain * np.outer(axis, axis) - _PATCH_LINE["nu"] * strain * (
            np.eye(3) - np.outer(axis, axis)
        )
        free_check: dict[int, np.ndarray] = {
            nid: strain * stations[nid - 1] * axis for nid in (2, 3)
        }
    elif name in ("TRIA3", "QUAD4"):
        model, coords, boundary = _shell_patch_model(name, distortion)
        gradient = PATCH_GRADIENT.copy()
        gradient[2, :] = 0.0
        gradient[:, 2] = 0.0
        enforced = {}
        for nid in boundary:
            exact = gradient @ coords[nid]
            enforced.update({(nid, comp): float(exact[comp]) for comp in range(3)})
            enforced.update({(nid, comp): 0.0 for comp in (3, 4, 5)})
        model["spcs"] = [{"node_id": nid, "dofs": (0, 1, 2, 3, 4, 5)} for nid in boundary]
        strain_tensor = 0.5 * (gradient + gradient.T)
        Dm = plane_stress_D(MaterialData(E=70.0e9, nu=0.3, G=70.0e9 / 2.6))
        plane = Dm @ np.array(
            [strain_tensor[0, 0], strain_tensor[1, 1], 2.0 * strain_tensor[0, 1]]
        )
        exact_stress = np.array(
            [[plane[0], plane[2], 0.0], [plane[2], plane[1], 0.0], [0.0, 0.0, 0.0]]
        )
        exact_strain = strain_tensor.copy()
        exact_strain[2, 2] = -0.3 * (plane[0] + plane[1]) / 70.0e9
        free = [nid for nid in coords if nid not in boundary]
        free_check = {nid: gradient @ coords[nid] for nid in free}
    else:
        if name in ("TET4", "TET10"):
            model, coords, boundary = tet_patch_model(name, distortion=distortion)
        else:
            hex_coords, ids, nodes, elements = _distorted_hex_patch(distortion, seed)
            model = _model(nodes, elements, 1.0e7, 0.3, 1.0)
            coords = {ids[key]: point for key, point in hex_coords.items()}
            boundary = [nid for key, nid in ids.items() if key != (1, 1, 1)]
        gradient = PATCH_GRADIENT
        enforced = {}
        for nid in boundary:
            exact = gradient @ coords[nid]
            enforced.update({(nid, comp): float(exact[comp]) for comp in range(3)})
        model["spcs"] = [{"node_id": nid, "dofs": (0, 1, 2)} for nid in boundary]
        strain_tensor = 0.5 * (gradient + gradient.T)
        voigt = np.array(
            [
                strain_tensor[0, 0],
                strain_tensor[1, 1],
                strain_tensor[2, 2],
                2.0 * strain_tensor[0, 1],
                2.0 * strain_tensor[1, 2],
                2.0 * strain_tensor[0, 2],
            ]
        )
        s = solid_D(MaterialData(E=1.0e7, nu=0.3, G=1.0e7 / 2.6)) @ voigt
        exact_stress = np.array(
            [[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]]
        )
        exact_strain = strain_tensor
        free_check = {nid: gradient @ coords[nid] for nid in coords if nid not in boundary}

    asm = assemble_km(model)
    u = np.asarray(solve_static(model, {}, assembly=asm, enforced=enforced))
    result = recover_stress(model, u, assembly=asm)

    def _voigt_of(tensor: np.ndarray, *, engineering: bool) -> np.ndarray:
        factor = 2.0 if engineering else 1.0
        return np.array(
            [
                tensor[0, 0],
                tensor[1, 1],
                tensor[2, 2],
                factor * tensor[0, 1],
                factor * tensor[1, 2],
                factor * tensor[0, 2],
            ]
        )

    want_stress = _voigt_of(exact_stress, engineering=False)
    want_strain = _voigt_of(exact_strain, engineering=True)
    scale_stress = float(np.max(np.abs(want_stress)))
    scale_strain = float(np.max(np.abs(want_strain)))
    stress_error = float(np.max(np.abs(result.stress_basic - want_stress))) / scale_stress
    strain_error = float(np.max(np.abs(result.strain_basic - want_strain))) / scale_strain

    nodal = average_nodal(result, model)
    nodal_error = max(
        float(np.max(np.abs(nodal.stress - want_stress))) / scale_stress,
        float(np.max(np.abs(nodal.strain - want_strain))) / scale_strain,
    )
    patch = recover_spr(result, model)
    spr_error = max(
        float(np.max(np.abs(patch.stress - want_stress))) / scale_stress,
        float(np.max(np.abs(patch.strain - want_strain))) / scale_strain,
    )

    displacement_error = 0.0
    for nid, exact in free_check.items():
        got = u[asm.dof_map.node_dofs(nid)][:3]
        displacement_error = max(
            displacement_error,
            float(np.max(np.abs(got - exact)) / max(np.max(np.abs(exact)), 1.0e-300)),
        )
    return {
        "stress": stress_error,
        "strain": strain_error,
        "nodal": nodal_error,
        "spr": spr_error,
        "displacement": displacement_error,
        "elements": float(len(result)),
        "nodes": float(len(nodal)),
        "spr_terms": float(np.min(patch.patch_terms)) if len(patch) else 0.0,
    }


# ---------------------------------------------------------------------------
# RBE2 rigid bodies
# ---------------------------------------------------------------------------


def rbe2_rigid_pair(
    offset: Any = (1.0, 0.5, -0.3),
    *,
    masses: tuple[float, float] = (2.0, 3.0),
    inertias: tuple[float, float] = (0.5, 0.2),
) -> dict[str, float]:
    """Two concentrated masses welded by one ``RBE2``, free-free.

    The classical check that a rigid body element is a *kinematic* statement
    and not a stiff spring: eliminating the dependent node leaves the six DOFs
    of the independent one, they carry no stiffness, and the reduced mass is
    exactly the rigid body mass matrix of the pair about the independent node,

    ``[[m I, -m2 skew(r)], [m2 skew(r)^T, I1 + I2 - m2 skew(r) skew(r)]]``.

    Reports the number of zero frequencies, the size of the solved set and the
    worst deviation of the reduced mass from that analytic matrix.
    """
    r = np.asarray(offset, dtype=float).reshape(3)
    m1, m2 = float(masses[0]), float(masses[1])
    i1, i2 = float(inertias[0]), float(inertias[1])
    model = {
        "nodes": {1: {"xyz": (0.0, 0.0, 0.0)}, 2: {"xyz": tuple(r)}},
        "elements": {
            1: {"type": "MASS", "nodes": (1,), "m": m1, "I11": i1, "I22": i1, "I33": i1},
            2: {"type": "MASS", "nodes": (2,), "m": m2, "I11": i2, "I22": i2, "I33": i2},
        },
        "materials": {},
        "properties": {},
        "spcs": [],
        "rbe2": [{"id": 1, "independent": 1, "dependents": (2,), "components": (1, 2, 3, 4, 5, 6)}],
    }
    asm = assemble_km(model)
    frequencies = solve_modes(model, n_modes=6, assembly=asm).freq_hz

    skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    expected = np.zeros((6, 6))
    expected[:3, :3] = (m1 + m2) * np.eye(3)
    expected[:3, 3:] = -m2 * skew
    expected[3:, :3] = -m2 * skew.T
    expected[3:, 3:] = (i1 + i2) * np.eye(3) - m2 * (skew @ skew)
    got = asm.Mff.toarray()
    return {
        "zero_modes": float(np.count_nonzero(np.asarray(frequencies) < 1.0e-6)),
        "free_dof": float(asm.n_free),
        "dependent_dof": float(asm.mpc_dof.size),
        "stiffness_norm": float(abs(asm.Kff).max()) if asm.Kff.nnz else 0.0,
        "mass_error": float(np.max(np.abs(got - expected)) / np.max(np.abs(expected))),
    }


def rbe2_offset_moment(
    n_elements: int = 8, *, arm: float = 0.25, force: float = 120.0
) -> dict[str, float]:
    """A rigid arm on the tip of a cantilever must deliver a moment.

    A ``BEAM2`` cantilever gets one extra node a distance ``arm`` above its tip
    and an ``RBE2`` welding it to the tip.  An axial force on that node is the
    textbook offset load: at the tip it becomes the same force *plus* the
    moment ``arm * force``.  The case is measured three ways -- against a model
    where that force and moment are applied directly, against the analytic
    Euler-Bernoulli cantilever, and against the rigid kinematics themselves
    (``u_arm == u_tip + theta_tip x r``).
    """
    d = BEAM_CANTILEVER
    length, arm = float(d["L"]), float(arm)
    base = beam_cantilever(n_elements)
    tip = n_elements + 1

    direct = {**base, "nodes": dict(base["nodes"])}
    rigid = {
        **base,
        "nodes": {**base["nodes"], tip + 1: {"xyz": (length, 0.0, arm)}},
        "rbe2": [{"id": 1, "independent": tip, "dependents": (tip + 1,)}],
    }

    asm_rigid = assemble_km(rigid)
    u_rigid = np.asarray(solve_static(rigid, {(tip + 1, 0): force}, assembly=asm_rigid))
    u_direct = np.asarray(
        solve_static(direct, {(tip, 0): force, (tip, 4): arm * force})
    )

    dofs = asm_rigid.dof_map
    beam_dofs = np.concatenate([dofs.node_dofs(nid) for nid in range(1, tip + 1)])
    gap = float(np.max(np.abs(u_rigid[beam_dofs] - u_direct[: beam_dofs.size])))
    scale = float(np.max(np.abs(u_direct)))

    tip_u = u_rigid[dofs.node_dofs(tip)]
    arm_u = u_rigid[dofs.node_dofs(tip + 1)]
    lever = np.array([0.0, 0.0, arm])
    kinematics = float(
        np.max(np.abs(arm_u[:3] - (tip_u[:3] + np.cross(tip_u[3:], lever))))
        + np.max(np.abs(arm_u[3:] - tip_u[3:]))
    )

    ei = d["E"] * d["Iy"]
    moment = arm * force
    return {
        "tip_axial": float(tip_u[0]),
        "analytic_axial": force * length / (d["E"] * d["A"]),
        "tip_deflection": float(tip_u[2]),
        "analytic_deflection": -moment * length**2 / (2.0 * ei),
        # The kernel's beam convention is ry = -dw/dx, so a tip moment that
        # deflects the beam down rotates the tip section positively.
        "tip_rotation": float(tip_u[4]),
        "analytic_rotation": moment * length / ei,
        "direct_gap": gap / scale,
        "rigid_kinematics": kinematics / max(float(np.max(np.abs(arm_u))), 1.0e-300),
        "moment": moment,
    }


# ---------------------------------------------------------------------------
# RBE3 interpolation constraints
# ---------------------------------------------------------------------------


def _triangle(radius: float) -> dict[int, tuple[float, float, float]]:
    """Three points on a circle about the origin, so the centroid is the origin."""
    angles = np.deg2rad([90.0, 210.0, 330.0])
    return {
        i + 1: (float(radius * np.cos(a)), float(radius * np.sin(a)), 0.0)
        for i, a in enumerate(angles)
    }


def _rigid_body_vectors(dof_map: Any, coords: dict[int, Any]) -> np.ndarray:
    """``(n_dof, 6)`` unit rigid body motions about the origin."""
    R = np.zeros((dof_map.n_dof, 6))
    for nid, xyz in coords.items():
        dofs = dof_map.node_dofs(nid)
        x = np.asarray(xyz, dtype=float)
        for k in range(3):
            R[dofs[k], k] = 1.0
            axis = np.zeros(3)
            axis[k] = 1.0
            R[dofs[:3], 3 + k] = np.cross(axis, x)
            R[dofs[3 + k], 3 + k] = 1.0
    return R


def _point_mass_rigid_body(mass: float, position: np.ndarray) -> np.ndarray:
    """``6x6`` rigid body mass matrix of a point mass, about the origin."""
    c = np.asarray(position, dtype=float).reshape(3)
    skew = np.array([[0.0, -c[2], c[1]], [c[2], 0.0, -c[0]], [-c[1], c[0], 0.0]])
    out = np.zeros((6, 6))
    out[:3, :3] = mass * np.eye(3)
    out[:3, 3:] = -mass * skew
    out[3:, :3] = -mass * skew.T
    out[3:, 3:] = -mass * (skew @ skew)
    return out


def rbe3_spider(
    weights: Any = None,
    *,
    mass: float = 2.5,
    radius: float = 0.6,
    dependent_xyz: Any = None,
    area: float = 4.0e-4,
) -> dict[str, float]:
    """A concentrated mass hung on the reference grid of an ``RBE3`` spider, free-free.

    Three pin-jointed ``BAR2`` rods form a triangle -- an exactly determinate
    rigid body in space, six rigid body modes and no mechanism -- and a
    ``MASS`` sits on a fourth node tied to the three vertices by one ``RBE3``
    (:mod:`femtools.fea.mpc`).  The statements checked are the two an
    interpolation constraint has to satisfy:

    * the structure is still free-free, with **exactly six** zero frequencies
      and no stiffness anywhere on the constraint -- unlike a penalty spring,
      and unlike an ``RBE2``, the spider does not weld the triangle solid;
    * the mass arrives in full.  Because the dependent motion is the weighted
      average ``u_d = sum_i w_i u_i / sum_j w_j``, a rigid body motion of the
      triangle moves the dependent node to the *weighted centroid* of the
      vertices, so the reduced rigid body mass matrix must be that of the bare
      triangle plus a point mass sitting at that centroid -- exactly, for any
      weights and wherever the reference grid itself is placed.

    ``dependent_xyz`` moves the reference grid off the centroid, which is the
    case where the second statement is worth reading twice: the mass is still
    delivered in full and the six modes are still there, but it is delivered to
    the weighted centroid rather than to where the node was drawn.
    """
    coords = _triangle(radius)
    centre = (
        np.zeros(3) if dependent_xyz is None else np.asarray(dependent_xyz, dtype=float)
    )
    w = (
        np.full(3, 1.0 / 3.0)
        if weights is None
        else np.asarray(weights, dtype=float) / float(np.sum(weights))
    )
    centroid = sum(w[i] * np.asarray(coords[i + 1]) for i in range(3))

    nodes = {nid: {"xyz": xyz} for nid, xyz in coords.items()}
    nodes[4] = {"xyz": tuple(float(v) for v in centre)}
    bars = {
        1: {"type": "BAR2", "property_id": 1, "nodes": (1, 2)},
        2: {"type": "BAR2", "property_id": 1, "nodes": (2, 3)},
        3: {"type": "BAR2", "property_id": 1, "nodes": (3, 1)},
    }
    common = {
        "nodes": nodes,
        "materials": {1: {"E": 2.1e11, "nu": 0.3, "rho": 7800.0}},
        "properties": {1: {"type": "bar", "material_id": 1, "A": area}},
        "spcs": [],
    }
    bare = {**common, "elements": dict(bars)}
    spider = {
        **common,
        "elements": {**bars, 4: {"type": "MASS", "nodes": (4,), "m": mass}},
        "rbe3": [
            {
                "id": 1,
                "dependent": 4,
                "independents": (1, 2, 3),
                "components": (1, 2, 3),
                **({} if weights is None else {"weights": tuple(weights)}),
            }
        ],
    }

    asm = assemble_km(spider)
    frequencies = solve_modes(spider, n_modes=8, assembly=asm).freq_hz

    R = _rigid_body_vectors(asm.dof_map, {**coords, 4: nodes[4]["xyz"]})
    got = R.T @ (asm.M @ R)
    reference = assemble_km(bare)
    expected = R.T @ (reference.M @ R) + _point_mass_rigid_body(mass, centroid)

    return {
        "zero_modes": float(np.count_nonzero(np.asarray(frequencies) < 1.0e-6)),
        "first_elastic_hz": float(frequencies[6]),
        "free_dof": float(asm.n_free),
        "dependent_dof": float(asm.mpc_dof.size),
        "constraint_stiffness": float(
            abs(asm.K - reference.K).max() if (asm.K - reference.K).nnz else 0.0
        ),
        "rigid_mass_error": float(
            np.max(np.abs(got - expected)) / np.max(np.abs(expected))
        ),
    }


def rbe3_load_path(
    weights: Any = None,
    *,
    force: float = 900.0,
    length: float = 1.2,
    area: float = 5.0e-4,
    E: float = 2.1e11,
    radius: float = 0.4,
) -> dict[str, float]:
    """A force on the reference grid of an ``RBE3`` is shared out by weight.

    Three parallel ``BAR2`` legs stand on a fixed base; their top nodes are the
    independents of one ``RBE3`` whose reference grid carries the load.  The
    legs are the whole load path, so what each one ends up carrying *is* the
    share the constraint handed it, and the answer is analytic: by virtual work
    the transpose of ``u_d = sum_i w_i u_i / sum_j w_j`` sends
    ``f_i = w_i / sum_j w_j`` of the force down leg ``i``, which stretches by
    ``f_i L / (E A)``.  Equal weights therefore give three equal shares whatever
    the geometry -- the property that separates an interpolation constraint
    from a rigid one, whose shares would follow the stiffnesses instead.

    The dependent displacement is then the *weighted average of unequal leg
    extensions*, ``sum_i w_i^2 F L / (E A)``, which is the cheapest way to see
    that the spider is not a rigid plate: a rigid cap would have made the three
    legs move together.
    """
    coords = _triangle(radius)
    w = (
        np.full(3, 1.0 / 3.0)
        if weights is None
        else np.asarray(weights, dtype=float) / float(np.sum(weights))
    )
    nodes: dict[int, Any] = {}
    elements: dict[int, Any] = {}
    spcs: list[dict[str, Any]] = []
    for nid, (x, y, _z) in coords.items():
        nodes[nid] = {"xyz": (x, y, length)}
        nodes[nid + 10] = {"xyz": (x, y, 0.0)}
        elements[nid] = {"type": "BAR2", "property_id": 1, "nodes": (nid + 10, nid)}
        spcs.append({"node_id": nid + 10, "dofs": (0, 1, 2)})
        # The legs are pin-jointed, so their transverse motion is a mechanism;
        # holding it leaves the axial load path the case is about.
        spcs.append({"node_id": nid, "dofs": (0, 1)})
    nodes[4] = {"xyz": (0.0, 0.0, length)}

    model = {
        "nodes": nodes,
        "elements": elements,
        "materials": {1: {"E": E, "nu": 0.3, "rho": 7800.0}},
        "properties": {1: {"type": "bar", "material_id": 1, "A": area}},
        "spcs": spcs,
        "rbe3": [
            {
                "id": 1,
                "dependent": 4,
                "independents": (1, 2, 3),
                "components": (1, 2, 3),
                **({} if weights is None else {"weights": tuple(weights)}),
            }
        ],
    }

    from .recover import recover_stress  # local: verification is imported on demand

    asm = assemble_km(model)
    u = np.asarray(solve_static(model, {(4, 2): force}, assembly=asm))
    stress = recover_stress(model, u, assembly=asm)

    axial = np.array([stress.extras[nid]["axial_force"] for nid in (1, 2, 3)])
    extension = np.array([u[asm.dof_map.index(nid, 2)] for nid in (1, 2, 3)])
    exact = w * force * length / (E * area)
    dependent = float(u[asm.dof_map.index(4, 2)])
    return {
        "share_error": float(np.max(np.abs(axial / force - w))),
        "min_share": float(np.min(axial / force)),
        "max_share": float(np.max(axial / force)),
        "extension_error": float(np.max(np.abs(extension - exact)) / np.max(np.abs(exact))),
        "dependent": dependent,
        "analytic_dependent": float(np.sum(w * exact)),
        "average_gap": abs(dependent - float(w @ extension)) / abs(dependent),
    }


# ---------------------------------------------------------------------------
# the two card types in one model
# ---------------------------------------------------------------------------

#: Which card hangs off which in :func:`mpc_mixed_chain`.
MPC_CHAIN_DIRECTIONS: tuple[str, ...] = ("rbe2_on_rbe3", "rbe3_on_rbe2")

#: Section and material of the free-free ``BEAM2`` triangle the chain hangs on.
_CHAIN_MATERIAL = {"E": 7.0e10, "nu": 0.3, "rho": 2700.0}
_CHAIN_SECTION = {
    "type": "beam",
    "material_id": 1,
    "A": 8.0e-4,
    "Iy": 3.0e-8,
    "Iz": 6.0e-8,
    "J": 9.0e-8,
    "orientation": (0.0, 0.0, 1.0),
}


def mpc_mixed_chain(
    direction: str = "rbe2_on_rbe3",
    *,
    weights: Any = None,
    mass: float = 2.5,
    radius: float = 0.6,
    arm: Any = (0.15, -0.1, 0.4),
) -> dict[str, float]:
    """An ``RBE2`` and an ``RBE3`` chained together, free-free.

    A closed ``BEAM2`` triangle is the structure -- a frame, so all six
    components of all three nodes carry stiffness and inertia -- and two extra
    nodes hang off it through one card of each type, composed by
    :func:`femtools.fea.mpc.apply_mpc`:

    ``direction="rbe2_on_rbe3"``
        node 4 is the reference grid of an ``RBE3`` interpolating the three
        vertices, and node 5 is welded to node 4 by an ``RBE2``.  The rigid arm
        therefore hangs off a node that is itself only an average.
    ``direction="rbe3_on_rbe2"``
        the reverse: node 4 is welded to vertex 1 by an ``RBE2``, and node 5 is
        the reference grid of an ``RBE3`` whose independents include that rigid
        node.

    The mass sits on node 5, at the far end of the chain, so it can only be
    delivered by both kinematics working in series.  Where it is delivered *to*
    is analytic and is the interesting part.  An ``RBE2`` carries a rigid body
    motion exactly and moves the mass by its own lever, while an ``RBE3``
    carries it to the *weighted centroid* of the independents (see
    :func:`rbe3_spider`), so the point the two hand it to is

    * ``sum_i w_i x_i + (x_5 - x_4)`` in the first direction, and
    * ``sum_i w_i x_i`` over the independents of the ``RBE3`` -- one of which
      is the rigid node -- in the second.

    Reports the zero-frequency count, the first elastic frequency, the size of
    the solved set, the largest entry of ``G^T K G`` minus the unconstrained
    ``K`` (the constraint may not add stiffness), the idempotency residual
    ``max|G G - G|`` (which is what says the chain was resolved rather than
    left nested), and the relative error of the reduced rigid body mass against
    the bare triangle plus a point mass at that delivered position.
    """
    if direction not in MPC_CHAIN_DIRECTIONS:
        raise ValueError(
            f"unknown chain direction {direction!r}; expected one of {MPC_CHAIN_DIRECTIONS}"
        )
    from .mpc import apply_mpc  # local: verification is imported on demand

    coords = _triangle(radius)
    lever = np.asarray(arm, dtype=float).reshape(3)
    w = (
        np.full(3, 1.0 / 3.0)
        if weights is None
        else np.asarray(weights, dtype=float) / float(np.sum(weights))
    )
    spelled = {} if weights is None else {"weights": tuple(weights)}
    nodes: dict[int, Any] = {nid: {"xyz": xyz} for nid, xyz in coords.items()}

    if direction == "rbe2_on_rbe3":
        nodes[4] = {"xyz": (0.0, 0.0, 0.0)}
        nodes[5] = {"xyz": tuple(float(v) for v in lever)}
        rbe3 = [
            {
                "id": 1,
                "dependent": 4,
                "independents": (1, 2, 3),
                "components": (1, 2, 3, 4, 5, 6),
                **spelled,
            }
        ]
        rbe2 = [{"id": 1, "independent": 4, "dependents": (5,)}]
        centroid = sum(w[i] * np.asarray(coords[i + 1]) for i in range(3))
        delivered = centroid + (np.asarray(nodes[5]["xyz"]) - np.asarray(nodes[4]["xyz"]))
    else:
        nodes[4] = {"xyz": tuple(float(v) for v in np.asarray(coords[1]) + lever)}
        nodes[5] = {"xyz": (0.0, 0.0, 0.0)}
        rbe2 = [{"id": 1, "independent": 1, "dependents": (4,)}]
        rbe3 = [
            {
                "id": 1,
                "dependent": 5,
                "independents": (4, 2, 3),
                "components": (1, 2, 3, 4, 5, 6),
                **spelled,
            }
        ]
        driving = (np.asarray(nodes[4]["xyz"]), np.asarray(coords[2]), np.asarray(coords[3]))
        delivered = sum(w[i] * driving[i] for i in range(3))

    beams = {
        eid: {"type": "BEAM2", "property_id": 1, "nodes": conn}
        for eid, conn in enumerate(((1, 2), (2, 3), (3, 1)), start=1)
    }
    common = {
        "nodes": nodes,
        "materials": {1: dict(_CHAIN_MATERIAL)},
        "properties": {1: dict(_CHAIN_SECTION)},
        "spcs": [],
    }
    bare = {**common, "elements": dict(beams)}
    model = {
        **common,
        "elements": {**beams, 4: {"type": "MASS", "nodes": (5,), "m": mass}},
        "rbe2": rbe2,
        "rbe3": rbe3,
    }

    asm = assemble_km(model)
    reference = assemble_km(bare, mpc=False)
    frequencies = np.asarray(solve_modes(model, n_modes=10, assembly=asm).freq_hz)

    transform = apply_mpc(model)
    G = transform.G.toarray()
    R = _rigid_body_vectors(asm.dof_map, {nid: node["xyz"] for nid, node in nodes.items()})
    got = R.T @ (asm.M @ R)
    expected = R.T @ (reference.M @ R) + _point_mass_rigid_body(mass, delivered)
    gap = asm.K - reference.K

    return {
        "zero_modes": float(np.count_nonzero(frequencies < 1.0e-6)),
        "first_elastic_hz": float(frequencies[6]),
        "free_dof": float(asm.n_free),
        "dependent_dof": float(transform.n_dependent),
        "constraint_stiffness": float(abs(gap).max()) if gap.nnz else 0.0,
        "idempotency": float(np.max(np.abs(G @ G - G))),
        "rigid_mass_error": float(np.max(np.abs(got - expected)) / np.max(np.abs(expected))),
        "delivered_x": float(delivered[0]),
        "delivered_y": float(delivered[1]),
        "delivered_z": float(delivered[2]),
    }
