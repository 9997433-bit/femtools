"""``BEAM2``: two-node 3D Euler-Bernoulli beam with 12 degrees of freedom.

Local DOF order is ``[u1 v1 w1 rx1 ry1 rz1 u2 v2 w2 rx2 ry2 rz2]``.
Bending in the local x-y plane uses ``Iz`` (Nastran ``I1``) and bending in the
local x-z plane uses ``Iy`` (Nastran ``I2``).  When shear areas are supplied the
stiffness switches to the Timoshenko form via the usual ``phi`` correction while
the consistent mass stays Euler-Bernoulli.
"""

from __future__ import annotations

import numpy as np

from .base import ElementContext, ElementMatrices, register
from .frames import line_frame

__all__ = ["beam2", "beam_local_matrices"]


def _section(ctx: ElementContext) -> dict[str, float]:
    area = ctx.number(("A", "a", "area", "Area"), None)
    if area is None:
        raise ValueError(f"element {ctx.element_id} (BEAM2): missing cross section area 'A'")
    iz = ctx.number(("I1", "Iz", "Izz", "I_z", "I"), None)
    iy = ctx.number(("I2", "Iy", "Iyy", "I_y"), None)
    if iz is None and iy is None:
        raise ValueError(
            f"element {ctx.element_id} (BEAM2): missing bending inertia "
            "(expected I1/Iz and/or I2/Iy)"
        )
    iz_val = float(iz if iz is not None else iy)  # type: ignore[arg-type]
    iy_val = float(iy if iy is not None else iz_val)
    iz, iy = iz_val, iy_val
    j = ctx.number(("J", "Jx", "It", "torsion_constant", "Ix"), None)
    if j is None:
        j = iy + iz
    nsm = ctx.number(("nsm", "NSM", "non_structural_mass"), 0.0) or 0.0
    # Effective shear areas (0 or None -> no shear flexibility, i.e. Euler-Bernoulli).
    asy = ctx.number(("As_y", "Asy", "A_sy", "shear_area_y"), None)
    asz = ctx.number(("As_z", "Asz", "A_sz", "shear_area_z"), None)
    k1 = ctx.number(("K1", "k1", "ky"), None)
    k2 = ctx.number(("K2", "k2", "kz"), None)
    if asy is None and k1:
        asy = float(k1) * float(area)
    if asz is None and k2:
        asz = float(k2) * float(area)
    return {
        "A": float(area),
        "Iy": iy,
        "Iz": iz,
        "J": float(j),
        "nsm": float(nsm),
        "Asy": float(asy) if asy else 0.0,
        "Asz": float(asz) if asz else 0.0,
    }


def _bending_block(ei: float, length: float, phi: float, sign: float) -> np.ndarray:
    """4x4 bending stiffness for dofs ``[w, theta, w, theta]``.

    ``sign = +1`` for the x-y plane (``theta = rz``) and ``sign = -1`` for the
    x-z plane where ``ry = -dw/dx``.
    """
    ll = length
    f = ei / (ll**3 * (1.0 + phi))
    s = sign
    return f * np.array(
        [
            [12.0, s * 6.0 * ll, -12.0, s * 6.0 * ll],
            [s * 6.0 * ll, (4.0 + phi) * ll * ll, -s * 6.0 * ll, (2.0 - phi) * ll * ll],
            [-12.0, -s * 6.0 * ll, 12.0, -s * 6.0 * ll],
            [s * 6.0 * ll, (2.0 - phi) * ll * ll, -s * 6.0 * ll, (4.0 + phi) * ll * ll],
        ]
    )


def _bending_mass(mu: float, length: float, sign: float) -> np.ndarray:
    """Euler-Bernoulli consistent translational mass for a bending plane."""
    ll = length
    s = sign
    return (mu * ll / 420.0) * np.array(
        [
            [156.0, s * 22.0 * ll, 54.0, -s * 13.0 * ll],
            [s * 22.0 * ll, 4.0 * ll * ll, s * 13.0 * ll, -3.0 * ll * ll],
            [54.0, s * 13.0 * ll, 156.0, -s * 22.0 * ll],
            [-s * 13.0 * ll, -3.0 * ll * ll, -s * 22.0 * ll, 4.0 * ll * ll],
        ]
    )


def beam_local_matrices(
    length: float,
    E: float,
    G: float,
    rho: float,
    sec: dict[str, float],
    *,
    lumped: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(12, 12)`` local stiffness and mass matrices."""
    A, Iy, Iz, J = sec["A"], sec["Iy"], sec["Iz"], sec["J"]
    mu = rho * A + sec["nsm"]

    phi_y = 0.0  # bending in x-y plane (shear along y)
    phi_z = 0.0  # bending in x-z plane (shear along z)
    if sec["Asy"] > 0.0 and G > 0.0:
        phi_y = 12.0 * E * Iz / (G * sec["Asy"] * length**2)
    if sec["Asz"] > 0.0 and G > 0.0:
        phi_z = 12.0 * E * Iy / (G * sec["Asz"] * length**2)

    k = np.zeros((12, 12))
    axial = E * A / length
    k[np.ix_([0, 6], [0, 6])] = axial * np.array([[1.0, -1.0], [-1.0, 1.0]])
    torsion = G * J / length
    k[np.ix_([3, 9], [3, 9])] = torsion * np.array([[1.0, -1.0], [-1.0, 1.0]])

    xy = [1, 5, 7, 11]   # v1, rz1, v2, rz2  -> uses Iz
    xz = [2, 4, 8, 10]   # w1, ry1, w2, ry2  -> uses Iy
    k[np.ix_(xy, xy)] = _bending_block(E * Iz, length, phi_y, +1.0)
    k[np.ix_(xz, xz)] = _bending_block(E * Iy, length, phi_z, -1.0)

    m = np.zeros((12, 12))
    m[np.ix_([0, 6], [0, 6])] = (mu * length / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    polar = rho * J
    m[np.ix_([3, 9], [3, 9])] = (polar * length / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    m[np.ix_(xy, xy)] = _bending_mass(mu, length, +1.0)
    m[np.ix_(xz, xz)] = _bending_mass(mu, length, -1.0)

    if lumped:
        total = mu * length
        m_l = np.zeros((12, 12))
        for i in (0, 1, 2, 6, 7, 8):
            m_l[i, i] = 0.5 * total
        rot = polar * length / 2.0
        for i in (3, 9):
            m_l[i, i] = rot
        inertia = 0.5 * total * length**2 / 78.0
        for i in (4, 5, 10, 11):
            m_l[i, i] = inertia
        m = m_l
    return k, m


@register(
    "BEAM2",
    n_nodes=2,
    dofs_per_node=(0, 1, 2, 3, 4, 5),
    family="line",
    description="Two-node 3D Euler-Bernoulli beam, 12 DOF, consistent mass",
    aliases=("BEAM", "CBEAM", "CBAR", "BEAM3D"),
)
def beam2(ctx: ElementContext) -> ElementMatrices:
    orientation = ctx.value(("orientation", "v", "vector", "x3", "g0_vector", "orient"), None)
    if orientation is not None:
        orientation = np.asarray(orientation, dtype=float).ravel()
        if orientation.size != 3:
            orientation = None
    length, R = line_frame(ctx.coords[0], ctx.coords[1], orientation)
    sec = _section(ctx)
    k_loc, m_loc = beam_local_matrices(
        length, ctx.mat.E, ctx.mat.G, ctx.mat.rho, sec, lumped=ctx.lumped_mass
    )

    T = np.zeros((12, 12))
    for blk in range(4):
        T[3 * blk : 3 * blk + 3, 3 * blk : 3 * blk + 3] = R
    k = T.T @ k_loc @ T
    m = T.T @ m_loc @ T

    dofs = [(nid, c) for nid in ctx.node_ids[:2] for c in range(6)]
    return ElementMatrices(dofs=dofs, k=k, m=m)
