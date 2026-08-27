"""Solid elements: ``TET4`` (constant strain) and ``HEX8`` (2x2x2 Gauss).

Solid nodes only carry the three translational DOFs; any rotational DOF left
untouched at such a node is removed by the assembler.
"""

from __future__ import annotations

import numpy as np

from ..materials import solid_D
from ..quadrature import gauss_3d
from .base import ElementContext, ElementMatrices, register

__all__ = ["tet4", "hex8"]

_TET_MASS = np.array(
    [[2.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 2.0]]
)


def _strain_matrix(grad: np.ndarray) -> np.ndarray:
    """Assemble the ``(6, 3n)`` strain-displacement matrix from nodal gradients."""
    n = grad.shape[0]
    B = np.zeros((6, 3 * n))
    for i in range(n):
        gx, gy, gz = grad[i]
        B[0, 3 * i] = gx
        B[1, 3 * i + 1] = gy
        B[2, 3 * i + 2] = gz
        B[3, 3 * i] = gy
        B[3, 3 * i + 1] = gx
        B[4, 3 * i + 1] = gz
        B[4, 3 * i + 2] = gy
        B[5, 3 * i] = gz
        B[5, 3 * i + 2] = gx
    return B


def _expand_mass(block: np.ndarray) -> np.ndarray:
    n = block.shape[0]
    out = np.zeros((3 * n, 3 * n))
    for a in range(n):
        for b in range(n):
            out[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] = block[a, b] * np.eye(3)
    return out


@register(
    "TET4",
    n_nodes=4,
    dofs_per_node=(0, 1, 2),
    family="solid",
    description="Four-node constant strain tetrahedron, 12 DOF",
    aliases=("CTETRA", "TETRA4", "TET"),
)
def tet4(ctx: ElementContext) -> ElementMatrices:
    xyz = ctx.coords[:4]
    M = np.column_stack([np.ones(4), xyz])
    det = float(np.linalg.det(M))
    volume = abs(det) / 6.0
    if volume <= 0.0:
        raise ValueError(f"element {ctx.element_id}: degenerate TET4 (zero volume)")
    C = np.linalg.inv(M)
    grad = C[1:4, :].T  # (4, 3): rows are dN_i/d{x,y,z}

    B = _strain_matrix(grad)
    D = solid_D(ctx.mat)
    k = volume * (B.T @ D @ B)

    block = (ctx.mat.rho * volume / 20.0) * _TET_MASS
    if ctx.lumped_mass:
        block = np.diag(block.sum(axis=1))
    m = _expand_mass(block)

    dofs = [(nid, comp) for nid in ctx.node_ids[:4] for comp in range(3)]
    return ElementMatrices(dofs=dofs, k=k, m=m)


_HEX_NODES = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
)


def _hex_shape(xi: float, eta: float, zeta: float):
    s = _HEX_NODES
    n = 0.125 * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta)
    dn = np.empty((8, 3))
    dn[:, 0] = 0.125 * s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * zeta)
    dn[:, 1] = 0.125 * s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * zeta)
    dn[:, 2] = 0.125 * s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta)
    return n, dn


@register(
    "HEX8",
    n_nodes=8,
    dofs_per_node=(0, 1, 2),
    family="solid",
    description="Eight-node trilinear hexahedron, 2x2x2 Gauss, 24 DOF",
    aliases=("CHEXA", "HEXA8", "HEX", "BRICK8"),
)
def hex8(ctx: ElementContext) -> ElementMatrices:
    xyz = ctx.coords[:8]
    D = solid_D(ctx.mat)
    pts, wts = gauss_3d(2)

    k = np.zeros((24, 24))
    m = np.zeros((24, 24))
    for (xi, eta, zeta), w in zip(pts, wts, strict=True):
        n, dn = _hex_shape(xi, eta, zeta)
        J = dn.T @ xyz
        det = float(np.linalg.det(J))
        if det == 0.0:
            raise ValueError(f"element {ctx.element_id}: degenerate HEX8 (zero Jacobian)")
        grad = np.linalg.solve(J, dn.T).T
        scale = w * abs(det)
        B = _strain_matrix(grad)
        k += scale * (B.T @ D @ B)
        m += _expand_mass(np.outer(n, n) * (scale * ctx.mat.rho))

    if ctx.lumped_mass:
        m = np.diag(m.sum(axis=1))

    dofs = [(nid, comp) for nid in ctx.node_ids[:8] for comp in range(3)]
    return ElementMatrices(dofs=dofs, k=k, m=m)
