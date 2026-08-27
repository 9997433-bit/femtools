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
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .assemble import assemble_km
from .eigen import solve_modes
from .static import solve_static

__all__ = [
    "hex8_bending_ratio",
    "hex8_patch_test_error",
    "hex8_rigid_body_frequencies",
    "hex_cantilever",
    "timoshenko_tip_deflection",
]


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
) -> tuple[dict[str, Any], list[int], dict[tuple[int, int], float]]:
    """Structured HEX8 mesh of a beam, clamped at ``x = 0``.

    Returns the model, the node ids on the free end face and a load dictionary
    spreading ``tip_force`` over that face in the ``z`` direction.
    """
    nodes: dict[int, Any] = {}
    ids: dict[tuple[int, int, int], int] = {}
    counter = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                ids[(i, j, k)] = counter
                nodes[counter] = {
                    "xyz": (length * i / nx, width * j / ny, height * k / nz)
                }
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
    """Mean tip deflection of :func:`hex_cantilever` over the Timoshenko value."""
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
