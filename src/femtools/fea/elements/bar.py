"""Axial (rod / truss) elements: ``BAR2`` and ``TRUSS2D``.

``BAR2`` is a pin-jointed 3D rod: axial stiffness only, translational mass in
all three directions.  Bending of a two-node line element is provided by
``BEAM2``.
"""

from __future__ import annotations

import numpy as np

from .base import ElementContext, ElementMatrices, register

__all__ = ["bar2", "truss2d"]

_AXIAL_K = np.array([[1.0, -1.0], [-1.0, 1.0]])
_AXIAL_M = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0


def _axial_data(ctx: ElementContext) -> tuple[float, np.ndarray, float, float]:
    p1, p2 = ctx.coords[0], ctx.coords[1]
    d = p2 - p1
    length = float(np.linalg.norm(d))
    if length <= 0.0:
        raise ValueError(f"element {ctx.element_id}: zero length rod")
    area = ctx.number(("A", "a", "area", "Area"), None)
    if area is None:
        raise ValueError(f"element {ctx.element_id} ({ctx.etype}): missing cross section area 'A'")
    ea = ctx.number(("EA",), None)
    axial = float(ea) if ea is not None else ctx.mat.E * float(area)
    mass_per_len = ctx.mat.rho * float(area)
    nsm = ctx.number(("nsm", "NSM", "non_structural_mass"), 0.0) or 0.0
    return length, d / length, axial, mass_per_len + float(nsm)


@register(
    "BAR2",
    n_nodes=2,
    dofs_per_node=(0, 1, 2),
    family="line",
    description="Two-node 3D axial rod (pin jointed), consistent mass",
    aliases=("ROD", "CROD", "BAR", "CONROD", "TRUSS", "TRUSS3D"),
)
def bar2(ctx: ElementContext) -> ElementMatrices:
    length, e1, axial, mass_per_len = _axial_data(ctx)
    dofs = [(ctx.node_ids[0], c) for c in (0, 1, 2)] + [(ctx.node_ids[1], c) for c in (0, 1, 2)]

    # Direction cosine row vector maps the 6 global translations onto the
    # 2 axial dofs.
    T = np.zeros((2, 6))
    T[0, 0:3] = e1
    T[1, 3:6] = e1
    k = T.T @ (axial / length * _AXIAL_K) @ T

    m_line = mass_per_len * length * _AXIAL_M
    if ctx.lumped_mass:
        m_line = np.diag(m_line.sum(axis=1))
    m = np.zeros((6, 6))
    for a in range(2):
        for b in range(2):
            m[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = m_line[a, b] * np.eye(3)
    return ElementMatrices(dofs=dofs, k=k, m=m)


@register(
    "TRUSS2D",
    n_nodes=2,
    dofs_per_node=(0, 1),
    family="line",
    description="Two-node planar (global XY) axial truss, consistent mass",
    aliases=("ROD2D", "TRUSS2"),
)
def truss2d(ctx: ElementContext) -> ElementMatrices:
    length, e1, axial, mass_per_len = _axial_data(ctx)
    if abs(e1[2]) > 1.0e-9:
        raise ValueError(
            f"element {ctx.element_id}: TRUSS2D must lie in the global XY plane "
            f"(direction cosine z = {e1[2]:.3e})"
        )
    dofs = [(ctx.node_ids[0], c) for c in (0, 1)] + [(ctx.node_ids[1], c) for c in (0, 1)]
    T = np.zeros((2, 4))
    T[0, 0:2] = e1[:2]
    T[1, 2:4] = e1[:2]
    k = T.T @ (axial / length * _AXIAL_K) @ T

    m_line = mass_per_len * length * _AXIAL_M
    if ctx.lumped_mass:
        m_line = np.diag(m_line.sum(axis=1))
    m = np.zeros((4, 4))
    for a in range(2):
        for b in range(2):
            m[2 * a : 2 * a + 2, 2 * b : 2 * b + 2] = m_line[a, b] * np.eye(2)
    return ElementMatrices(dofs=dofs, k=k, m=m)
