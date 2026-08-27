"""Scalar / concentrated elements: ``MASS``, ``SPRING`` and ``DAMPER``.

``SPRING`` and ``DAMPER`` share one kinematic description:

* an explicit component (``c1``/``c2``, ``dof``, ``components``) makes them
  behave like ``CELAS2``/``CDAMP2`` scalar elements,
* two distinct nodes without a component give an axial spring along the
  connecting line,
* a single node (or ``None``/absent second node) grounds the element.

A vector coefficient (3 or 6 entries) creates one uncoupled scalar element per
component, which is the usual way of writing bushings.
"""

from __future__ import annotations

import numpy as np

from .base import ElementContext, ElementMatrices, register

__all__ = ["mass_element", "spring", "damper"]

_PAIR = np.array([[1.0, -1.0], [-1.0, 1.0]])


@register(
    "MASS",
    n_nodes=(1, 2),
    dofs_per_node=(0, 1, 2, 3, 4, 5),
    family="scalar",
    description="Concentrated mass and rotary inertia at a grid point",
    aliases=("CONM2", "CONM", "POINTMASS", "LUMPEDMASS", "CMASS"),
)
def mass_element(ctx: ElementContext) -> ElementMatrices:
    nid = ctx.node_ids[0]
    raw = ctx.value(("m", "M", "mass", "Mass"), None)
    if raw is None:
        raw = 0.0
    arr = np.atleast_1d(np.asarray(raw, dtype=float)).ravel()
    if arr.size == 1:
        trans = np.full(3, float(arr[0]))
    elif arr.size >= 3:
        trans = arr[:3].astype(float)
    else:
        raise ValueError(f"element {ctx.element_id} (MASS): cannot interpret mass {raw!r}")

    m = np.zeros((6, 6))
    m[0, 0], m[1, 1], m[2, 2] = trans

    inertia = ctx.value(("I", "inertia", "I_matrix"), None)
    if inertia is not None:
        block = np.asarray(inertia, dtype=float)
        if block.shape == (3, 3):
            m[3:6, 3:6] = 0.5 * (block + block.T)
        elif block.size == 3:
            m[3, 3], m[4, 4], m[5, 5] = block.ravel()
        elif block.size == 6:
            i11, i21, i22, i31, i32, i33 = block.ravel()
            m[3:6, 3:6] = np.array(
                [[i11, i21, i31], [i21, i22, i32], [i31, i32, i33]], dtype=float
            )
        else:
            raise ValueError(f"element {ctx.element_id} (MASS): bad inertia {inertia!r}")
    else:
        i11 = ctx.number(("I11", "Ixx", "ixx"), 0.0) or 0.0
        i22 = ctx.number(("I22", "Iyy", "iyy"), 0.0) or 0.0
        i33 = ctx.number(("I33", "Izz", "izz"), 0.0) or 0.0
        i21 = ctx.number(("I21", "Ixy", "ixy"), 0.0) or 0.0
        i31 = ctx.number(("I31", "Ixz", "ixz"), 0.0) or 0.0
        i32 = ctx.number(("I32", "Iyz", "iyz"), 0.0) or 0.0
        m[3:6, 3:6] = np.array([[i11, i21, i31], [i21, i22, i32], [i31, i32, i33]], dtype=float)

    return ElementMatrices(dofs=[(nid, comp) for comp in range(6)], m=m)


def _coefficient(ctx: ElementContext, names: tuple[str, ...], what: str) -> np.ndarray:
    raw = ctx.value(names, None)
    if raw is None:
        raise ValueError(
            f"element {ctx.element_id} ({ctx.etype}): missing {what} "
            f"(looked for {names})"
        )
    return np.atleast_1d(np.asarray(raw, dtype=float)).ravel()


def _scalar_connection(ctx: ElementContext, coeff: np.ndarray):
    """Return ``(dofs, matrix)`` for a spring/damper style connection."""
    node_ids = list(ctx.node_ids)
    comps1 = ctx.dof_spec(("c1", "dof1", "dofs", "dof", "components", "component"), None)
    comps2 = ctx.dof_spec(("c2", "dof2"), None) or comps1

    grounded = len(node_ids) < 2
    if not grounded and np.linalg.norm(ctx.coords[1] - ctx.coords[0]) == 0.0 and not comps1:
        grounded = True  # coincident nodes cannot define an axis

    if not comps1:
        if coeff.size == 6:
            comps1 = comps2 = list(range(6))
        elif coeff.size == 3:
            comps1 = comps2 = [0, 1, 2]
        elif grounded:
            comps1 = comps2 = [0, 1, 2]
        else:
            # Axial spring along the element line.
            e1 = ctx.coords[1] - ctx.coords[0]
            e1 = e1 / np.linalg.norm(e1)
            T = np.zeros((2, 6))
            T[0, 0:3] = e1
            T[1, 3:6] = e1
            dofs = [(node_ids[0], c) for c in (0, 1, 2)] + [
                (node_ids[1], c) for c in (0, 1, 2)
            ]
            return dofs, T.T @ (float(coeff[0]) * _PAIR) @ T

    if len(comps2) != len(comps1):
        comps2 = list(comps1)
    values = coeff if coeff.size == len(comps1) else np.full(len(comps1), float(coeff[0]))
    if coeff.size in (3, 6) and coeff.size != len(comps1):
        values = np.array([coeff[c] for c in comps1], dtype=float)

    if grounded:
        dofs = [(node_ids[0], c) for c in comps1]
        return dofs, np.diag(values.astype(float))

    dofs = [(node_ids[0], c) for c in comps1] + [(node_ids[1], c) for c in comps2]
    n = len(comps1)
    mat = np.zeros((2 * n, 2 * n))
    for i in range(n):
        mat[i, i] += values[i]
        mat[n + i, n + i] += values[i]
        mat[i, n + i] -= values[i]
        mat[n + i, i] -= values[i]
    return dofs, mat


@register(
    "SPRING",
    n_nodes=(1, 2),
    dofs_per_node=(0, 1, 2, 3, 4, 5),
    family="scalar",
    description="Scalar / axial / grounded spring (CELAS-like)",
    aliases=("CELAS", "CELAS2", "CBUSH", "SPRING2", "ELAS"),
)
def spring(ctx: ElementContext) -> ElementMatrices:
    coeff = _coefficient(ctx, ("k", "K", "stiffness", "spring_k", "ke"), "spring stiffness 'k'")
    dofs, mat = _scalar_connection(ctx, coeff)
    return ElementMatrices(dofs=dofs, k=mat)


@register(
    "DAMPER",
    n_nodes=(1, 2),
    dofs_per_node=(0, 1, 2, 3, 4, 5),
    family="scalar",
    description="Scalar / axial / grounded viscous damper (CDAMP-like)",
    aliases=("CDAMP", "CDAMP2", "DASHPOT", "VISC"),
)
def damper(ctx: ElementContext) -> ElementMatrices:
    coeff = _coefficient(ctx, ("c", "C", "damping", "damper_c", "ce"), "damping coefficient 'c'")
    dofs, mat = _scalar_connection(ctx, coeff)
    return ElementMatrices(dofs=dofs, c=mat)
