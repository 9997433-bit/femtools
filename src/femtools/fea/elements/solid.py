"""Solid elements: ``TET4`` (constant strain) and ``HEX8`` (trilinear brick).

Solid nodes only carry the three translational DOFs; any rotational DOF left
untouched at such a node is removed by the assembler.

The plain trilinear brick integrated with 2x2x2 Gauss shear-locks badly in
bending: a single element through the thickness recovers roughly two thirds of
the reference tip deflection of a cantilever.  :func:`hex8` therefore defaults
to the **incompatible modes** formulation of Wilson et al. with the Taylor,
Beresford and Wilson correction, which passes the patch test on distorted
meshes and removes the parasitic shear stiffness.  See
:data:`HEX8_FORMULATIONS` for the alternatives and how to select them.

Both solids reject geometry they cannot integrate meaningfully: a hexahedron
whose Jacobian changes sign inside the element is folded through itself, and a
solid of either type whose volume has collapsed relative to its own size would
otherwise contribute stiffness far outside the dynamic range of the rest of the
model.  Consistently reversed node ordering is *not* an error; it integrates
correctly and stays accepted.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..materials import solid_D
from ..protocols import get_any
from ..quadrature import gauss_3d
from .base import ElementContext, ElementMatrices, register

__all__ = ["DEGENERACY_TOLERANCE", "HEX8_FORMULATIONS", "hex8", "hex8_formulation", "tet4"]

#: A solid whose volume is below this fraction of the cube of its bounding-box
#: diagonal is rejected as degenerate.
#:
#: The element stiffness of a flattened solid grows like the inverse of its
#: thinnest dimension, so an element at this aspect ratio contributes entries
#: some twelve orders of magnitude above the rest of the model and destroys the
#: conditioning of the whole assembly rather than just its own answer.  The
#: threshold is deliberately far below any mesh a user would defend: a
#: plate-like brick with a 1000:1 aspect ratio still scores about 3.5e-4.
DEGENERACY_TOLERANCE = 1.0e-12

_TET_MASS = np.array(
    [[2.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 2.0]]
)


def _size_scale(xyz: np.ndarray) -> float:
    """Bounding-box diagonal, the length the element volume is measured against."""
    return float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))


def _check_volume(element_id: Any, etype: str, xyz: np.ndarray, volume: float) -> None:
    scale = _size_scale(xyz)
    if volume <= DEGENERACY_TOLERANCE * scale**3:
        raise ValueError(
            f"element {element_id}: degenerate {etype} (volume {volume:.3e} is below "
            f"{DEGENERACY_TOLERANCE:g} x the cube of its {scale:.3e} bounding-box diagonal)"
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


def _dilatational_matrix(grad: np.ndarray) -> np.ndarray:
    """``(6, 3n)`` matrix holding only the volumetric part of ``_strain_matrix``.

    The first three rows are ``div(N_i)/3`` so that adding
    ``_dilatational_matrix(gbar - grad)`` to ``_strain_matrix(grad)`` swaps the
    point-wise dilatation for a volume averaged one (the classic B-bar trick).
    """
    n = grad.shape[0]
    out = np.zeros((6, 3 * n))
    out[:3, :] = grad.ravel() / 3.0
    return out


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
    _check_volume(ctx.element_id, "TET4", xyz, volume)
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


def _hex_enhanced_dn(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Natural derivatives of the three bubbles ``1-xi^2``, ``1-eta^2``, ``1-zeta^2``.

    Row ``a`` is ``dNa/d(xi, eta, zeta)``; the bubbles are separable so the
    array is diagonal.  Each bubble carries three displacement amplitudes, so
    the element gains nine internal DOFs.
    """
    return np.diag((-2.0 * xi, -2.0 * eta, -2.0 * zeta))


#: HEX8 stiffness formulations, in the order they are documented.
#:
#: ``"incompatible"``
#:     Wilson/Taylor incompatible modes (default).  Nine internal DOFs carrying
#:     the quadratic bubbles are added and statically condensed out, which
#:     restores the linear bending strain the trilinear field cannot represent.
#:     The internal gradients use the Jacobian at the element centre scaled by
#:     ``det J0 / det J`` (Taylor, Beresford and Wilson 1976) so the element
#:     passes the constant-stress patch test on distorted meshes.  Passing the
#:     patch test is not the same as staying accurate: the bending advantage
#:     survives any parallelepiped shape untouched but decays quickly once the
#:     Jacobian varies *within* the element.  See
#:     :mod:`femtools.fea.verification` for the sweep and for
#:     ``hex8_jacobian_spread``, the mesh measure that predicts it.
#: ``"bbar"``
#:     Mean dilatation, identical to selectively reduced integration (volumetric
#:     term at the centroid, deviatoric term 2x2x2).  Cures volumetric locking as
#:     ``nu -> 0.5`` on meshes that resolve the bending direction, but it does
#:     *not* cure shear locking and it over-softens thin bending when there is
#:     only one element through the thickness -- use it for bulky,
#:     nearly-incompressible parts, not for plate-like meshes.
#: ``"full"``
#:     Plain 2x2x2 Gauss displacement element, kept for reference and patch
#:     testing.  Shear-locks in bending.
#:
#: Combining ``"bbar"`` with the internal modes is deliberately *not* offered:
#: projecting the bubbles onto the mean dilatation leaves the condensed matrix
#: rank deficient (three spurious zero-energy modes), and leaving them
#: unprojected makes the element grossly over-soft.
HEX8_FORMULATIONS: tuple[str, ...] = ("incompatible", "bbar", "full")

_HEX8_ALIASES: dict[str, str] = {
    "incompatible": "incompatible",
    "incompatiblemodes": "incompatible",
    "im": "incompatible",
    "q6": "incompatible",
    "qm6": "incompatible",
    "wilson": "incompatible",
    "taylor": "incompatible",
    "default": "incompatible",
    "auto": "incompatible",
    "bbar": "bbar",
    "meandilatation": "bbar",
    "selective": "bbar",
    "selectivereduced": "bbar",
    "sri": "bbar",
    "reduced": "bbar",
    "full": "full",
    "fullintegration": "full",
    "standard": "full",
    "displacement": "full",
    "2x2x2": "full",
    "gauss2": "full",
}

#: Keys searched on the element / property record and in the assembly options.
#: Deliberately specific: a bare ``integration`` field is exactly the kind of
#: name a future ``PSOLID`` reader would use for something else.
_HEX8_OPTION_KEYS = ("hex8", "hex8_formulation", "solid_formulation", "formulation")


def hex8_formulation(spec: Any, *, default: str = "incompatible") -> str:
    """Normalise a HEX8 formulation designator (see :data:`HEX8_FORMULATIONS`)."""
    if spec is None:
        return default
    key = str(spec).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    resolved = _HEX8_ALIASES.get(key)
    if resolved is None:
        raise ValueError(
            f"unknown HEX8 formulation {spec!r}; expected one of {HEX8_FORMULATIONS}"
        )
    return resolved


def _resolve_formulation(ctx: ElementContext) -> str:
    """Element/property setting wins over the assembly-wide ``options`` entry."""
    raw = ctx.value(_HEX8_OPTION_KEYS, None)
    if raw is None and ctx.options:
        raw = get_any(ctx.options, _HEX8_OPTION_KEYS, None)
    try:
        return hex8_formulation(raw)
    except ValueError as exc:
        raise ValueError(f"element {ctx.element_id} (HEX8): {exc}") from None


@register(
    "HEX8",
    n_nodes=8,
    dofs_per_node=(0, 1, 2),
    family="solid",
    description="Eight-node hexahedron, incompatible modes (Wilson/Taylor), 24 DOF",
    aliases=("CHEXA", "HEXA8", "HEX", "BRICK8"),
)
def hex8(ctx: ElementContext) -> ElementMatrices:
    xyz = ctx.coords[:8]
    D = solid_D(ctx.mat)
    pts, wts = gauss_3d(2)
    formulation = _resolve_formulation(ctx)
    enhanced = formulation == "incompatible"
    mean_dilatation = formulation == "bbar"

    n_gp = wts.size
    shapes = np.empty((n_gp, 8))
    jacobians = np.empty((n_gp, 3, 3))
    naturals = np.empty((n_gp, 8, 3))
    dets = np.empty(n_gp)
    for g, (xi, eta, zeta) in enumerate(pts):
        shapes[g], naturals[g] = _hex_shape(xi, eta, zeta)
        jacobians[g] = naturals[g].T @ xyz
        dets[g] = np.linalg.det(jacobians[g])

    scales = wts * np.abs(dets)
    _check_volume(ctx.element_id, "HEX8", xyz, float(scales.sum()))
    if not (np.all(dets > 0.0) or np.all(dets < 0.0)):
        # Taking ``abs(det)`` above keeps a mesh written with the node ordering
        # reversed integrating correctly.  A sign change *within* one element is
        # a different animal: the hexahedron is folded through itself, part of
        # it is integrated inside out, and the resulting matrices are wrong
        # while still looking perfectly healthy (positive definite, six rigid
        # body modes).  Refuse rather than return a plausible wrong answer.
        raise ValueError(
            f"element {ctx.element_id}: folded HEX8 (the Jacobian determinant changes "
            f"sign over the element, from {dets.min():.3e} to {dets.max():.3e}); "
            "check the node ordering"
        )

    grads = np.empty((n_gp, 8, 3))
    grads_e = np.empty((n_gp, 3, 3)) if enhanced else None

    if enhanced:
        # Taylor/Beresford/Wilson: the internal modes are mapped with the
        # Jacobian frozen at the element centre so that their integral over the
        # element vanishes -- the condition for passing the patch test.
        _, dn0 = _hex_shape(0.0, 0.0, 0.0)
        J0 = dn0.T @ xyz
        det0 = float(np.linalg.det(J0))
        if det0 == 0.0:
            raise ValueError(f"element {ctx.element_id}: degenerate HEX8 (zero Jacobian)")

    m = np.zeros((24, 24))
    for g in range(n_gp):
        grads[g] = np.linalg.solve(jacobians[g], naturals[g].T).T
        if enhanced:
            enh = _hex_enhanced_dn(*pts[g])
            grads_e[g] = np.linalg.solve(J0, enh.T).T * (det0 / dets[g])
        m += _expand_mass(np.outer(shapes[g], shapes[g]) * (scales[g] * ctx.mat.rho))

    if mean_dilatation:
        gbar = np.einsum("g,gij->ij", scales, grads) / float(scales.sum())

    k = np.zeros((24, 24))
    kua = np.zeros((24, 9)) if enhanced else None
    kaa = np.zeros((9, 9)) if enhanced else None
    for g in range(n_gp):
        B = _strain_matrix(grads[g])
        if mean_dilatation:
            B = B + _dilatational_matrix(gbar - grads[g])
        DB = D @ B
        k += scales[g] * (B.T @ DB)
        if enhanced:
            G = _strain_matrix(grads_e[g])
            DG = D @ G
            kua += scales[g] * (B.T @ DG)
            kaa += scales[g] * (G.T @ DG)

    if enhanced and kaa.any():
        # Static condensation of the nine internal amplitudes.  The result is
        # the Schur complement of a positive semi-definite matrix whose only
        # null space is the six rigid body modes, so the condensation cannot
        # introduce a zero-energy (hourglass) mechanism.
        kaa = 0.5 * (kaa + kaa.T)
        try:
            coupling = np.linalg.solve(kaa, kua.T)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate geometry
            warnings.warn(
                f"element {ctx.element_id}: HEX8 internal modes are singular; "
                "using a pseudo-inverse for the static condensation",
                RuntimeWarning,
                stacklevel=2,
            )
            coupling = np.linalg.pinv(kaa) @ kua.T
        k = k - kua @ coupling
        k = 0.5 * (k + k.T)

    if ctx.lumped_mass:
        m = np.diag(m.sum(axis=1))

    dofs = [(nid, comp) for nid in ctx.node_ids[:8] for comp in range(3)]
    return ElementMatrices(dofs=dofs, k=k, m=m)
