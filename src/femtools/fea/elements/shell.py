"""Flat shell elements: ``TRIA3`` (CST + DKT) and ``QUAD4`` (2x2 Mindlin).

Both elements carry six degrees of freedom per node in the global frame:

* membrane   -- constant strain triangle / bilinear quad (2x2 Gauss),
* bending    -- DKT (triangle) / Reissner-Mindlin with MITC4 assumed
  transverse shear (quad, 2x2 Gauss),
* drilling   -- a rank deficient fictitious stiffness that leaves rigid body
  rotations energy free.  The assembler removes drilling DOFs that receive no
  genuine stiffness, so a flat plate produces no spurious mechanism.

Orientation caveat
------------------

That last step needs the drilling direction to *be* a global DOF.  A flat mesh
lying in a plane spanned by two global axes has its drilling rotation on a
single component (``rz`` for an x-y plate, ``ry`` for an x-z one) and the
assembler drops it exactly.  Tilt the same mesh so its normal is oblique and
the drilling direction becomes a combination of ``rx``, ``ry`` and ``rz``: no
whole DOF can be removed without taking a genuine bending rotation with it, so
the per-element null spaces survive as one global zero-energy mechanism and the
free-free spectrum shows a *seventh* zero frequency.  The elastic frequencies
are unaffected -- :func:`femtools.fea.verification.shell_drilling_orientation_gap`
reproduces both spectra and they agree to twelve digits -- and
:func:`femtools.fea.assemble.assemble_km` warns when it detects the retained
mechanism.  Removing it for good needs a per-node rotational frame rather than
an index set of eliminated DOFs, which is a change to the assembly contract
rather than to this module.
"""

from __future__ import annotations

import numpy as np

from ..materials import plane_stress_D
from ..protocols import as_bool
from ..quadrature import gauss_2d, tri_rule
from .base import ElementContext, ElementMatrices, register
from .frames import shell_frame

__all__ = ["tria3", "quad4", "dkt_bending_stiffness"]

_MEMBRANE_ONLY = {"membrane", "pmembrane", "plane_stress", "planestress", "cst", "pmemb"}
_BENDING_ONLY = {"plate", "bending", "pplate", "kirchhoff", "mindlin"}


def _shell_props(ctx: ElementContext, area: float) -> dict:
    t = ctx.number(("t", "T", "thickness", "h", "T1"), None)
    if t is None or t <= 0.0:
        raise ValueError(f"element {ctx.element_id} ({ctx.etype}): missing thickness 't'")
    t = float(t)
    # Read the *property* type only: the element already owns a ``type`` field.
    ptype = str(ctx.prop_value(("type", "ptype", "prop_type"), "") or "").strip().lower()
    membrane = ptype not in _BENDING_ONLY
    bending = ptype not in _MEMBRANE_ONLY
    if as_bool(ctx.value(("membrane_only",), False)):
        bending = False
    if as_bool(ctx.value(("bending_only",), False)):
        membrane = False
    i_ratio = ctx.number(("I_ratio", "bending_ratio", "twelveI_t3", "MID2_ratio"), 1.0)
    ts_t = ctx.number(("ts_t", "TS_T", "shear_ratio"), 5.0 / 6.0)
    nsm = ctx.number(("nsm", "NSM", "non_structural_mass"), 0.0) or 0.0
    return {
        "t": t,
        "membrane": membrane,
        "bending": bending,
        "i_ratio": float(i_ratio if i_ratio else 1.0),
        "kappa": float(ts_t if ts_t else 5.0 / 6.0),
        "mass_per_area": ctx.mat.rho * t + float(nsm),
        "rot_inertia_per_area": ctx.mat.rho * t**3 / 12.0,
        "area": area,
    }


def _mass_components(sp: dict) -> tuple[int, ...]:
    """Local translational DOFs that carry mass for this property.

    A membrane-only property leaves the out-of-plane translation massless (and
    stiffness free) so the assembler removes it, which is what makes a plain
    2D plane-stress model solvable without extra constraints.
    """
    comps: list[int] = []
    if sp["membrane"]:
        comps += [0, 1]
    if sp["bending"]:
        comps.append(2)
    return tuple(comps)


def _rotary_components(sp: dict) -> tuple[int, ...]:
    comps: list[int] = []
    if sp["bending"]:
        comps += [3, 4]
        if sp["membrane"]:
            comps.append(5)
    return tuple(comps)


def _drilling(n: int, c: float) -> np.ndarray:
    """Rank deficient drilling penalty: zero energy for a uniform rotation."""
    return c * (np.eye(n) - np.ones((n, n)) / float(n))


def _ix(rows: list[int]) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(rows, dtype=int)
    return idx[:, None], idx[None, :]


def _scatter(local: np.ndarray, block: np.ndarray, rows) -> None:
    ix = rows if isinstance(rows, tuple) else _ix(rows)
    local[ix] += block


_T3_MEM_IX = _ix([0, 1, 6, 7, 12, 13])
_T3_BEN_IX = _ix([2, 3, 4, 8, 9, 10, 14, 15, 16])
_T3_DRILL_IX = _ix([5, 11, 17])
_Q4_MEM_IX = _ix([0, 1, 6, 7, 12, 13, 18, 19])
_Q4_BEN_IX = _ix([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22])
_Q4_DRILL_IX = _ix([5, 11, 17, 23])


def _to_global(mat_local: np.ndarray, R: np.ndarray, n_nodes: int) -> np.ndarray:
    T = np.zeros_like(mat_local)
    for blk in range(2 * n_nodes):
        T[3 * blk : 3 * blk + 3, 3 * blk : 3 * blk + 3] = R
    return T.T @ mat_local @ T


# --------------------------------------------------------------------------
# TRIA3
# --------------------------------------------------------------------------


def _cst_membrane(xy: np.ndarray, Dm: np.ndarray, t: float):
    x, y = xy[:, 0], xy[:, 1]
    b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]])
    c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]])
    area2 = x[0] * b[0] + x[1] * b[1] + x[2] * b[2]
    area = 0.5 * area2
    if abs(area) < 1.0e-300:
        raise ValueError("degenerate TRIA3 (zero area)")
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2 * i] = b[i]
        B[1, 2 * i + 1] = c[i]
        B[2, 2 * i] = c[i]
        B[2, 2 * i + 1] = b[i]
    B /= area2
    return area, t * area * (B.T @ Dm @ B)


def dkt_bending_stiffness(xy: np.ndarray, Db: np.ndarray) -> np.ndarray:
    """Batoz DKT plate bending stiffness.

    DOF order ``[w1, rx1, ry1, w2, rx2, ry2, w3, rx3, ry3]`` with the
    kinematics ``u = z*ry``, ``v = -z*rx`` (rotation vector convention), i.e.
    the same convention as the Mindlin quad below.
    """
    x, y = xy[:, 0], xy[:, 1]
    x23, y23 = x[1] - x[2], y[1] - y[2]
    x31, y31 = x[2] - x[0], y[2] - y[0]
    x12, y12 = x[0] - x[1], y[0] - y[1]
    area2 = x31 * y12 - x12 * y31
    if abs(area2) < 1.0e-300:
        raise ValueError("degenerate TRIA3 (zero area)")

    xs = (x23, x31, x12)
    ys = (y23, y31, y12)
    ll = [xs[i] ** 2 + ys[i] ** 2 for i in range(3)]
    P = [-6.0 * xs[i] / ll[i] for i in range(3)]      # index 0,1,2 <-> sides 4,5,6
    q = [3.0 * xs[i] * ys[i] / ll[i] for i in range(3)]
    tt = [-6.0 * ys[i] / ll[i] for i in range(3)]
    r = [3.0 * ys[i] ** 2 / ll[i] for i in range(3)]
    P4, P5, P6 = P
    q4, q5, q6 = q
    t4, t5, t6 = tt
    r4, r5, r6 = r

    pts, wts = tri_rule(2)
    K = np.zeros((9, 9))
    for (xi, eta), w in zip(pts, wts, strict=True):
        hx_xi = np.array([
            P6 * (1.0 - 2.0 * xi) + (P5 - P6) * eta,
            q6 * (1.0 - 2.0 * xi) - (q5 + q6) * eta,
            -4.0 + 6.0 * (xi + eta) + r6 * (1.0 - 2.0 * xi) - eta * (r5 + r6),
            -P6 * (1.0 - 2.0 * xi) + eta * (P4 + P6),
            q6 * (1.0 - 2.0 * xi) - eta * (q6 - q4),
            -2.0 + 6.0 * xi + r6 * (1.0 - 2.0 * xi) + eta * (r4 - r6),
            -eta * (P5 + P4),
            eta * (q4 - q5),
            -eta * (r5 - r4),
        ])
        hy_xi = np.array([
            t6 * (1.0 - 2.0 * xi) + (t5 - t6) * eta,
            1.0 + r6 * (1.0 - 2.0 * xi) - (r5 + r6) * eta,
            -q6 * (1.0 - 2.0 * xi) + eta * (q5 + q6),
            -t6 * (1.0 - 2.0 * xi) + eta * (t4 + t6),
            -1.0 + r6 * (1.0 - 2.0 * xi) + eta * (r4 - r6),
            -q6 * (1.0 - 2.0 * xi) - eta * (q4 - q6),
            -eta * (t4 + t5),
            eta * (r4 - r5),
            -eta * (q4 - q5),
        ])
        hx_eta = np.array([
            -P5 * (1.0 - 2.0 * eta) - (P6 - P5) * xi,
            q5 * (1.0 - 2.0 * eta) - (q5 + q6) * xi,
            -4.0 + 6.0 * (xi + eta) + r5 * (1.0 - 2.0 * eta) - xi * (r5 + r6),
            xi * (P4 + P6),
            xi * (q4 - q6),
            -xi * (r6 - r4),
            P5 * (1.0 - 2.0 * eta) - xi * (P4 + P5),
            q5 * (1.0 - 2.0 * eta) + xi * (q4 - q5),
            -2.0 + 6.0 * eta + r5 * (1.0 - 2.0 * eta) + xi * (r4 - r5),
        ])
        hy_eta = np.array([
            -t5 * (1.0 - 2.0 * eta) - (t6 - t5) * xi,
            1.0 + r5 * (1.0 - 2.0 * eta) - (r5 + r6) * xi,
            -q5 * (1.0 - 2.0 * eta) + xi * (q5 + q6),
            xi * (t4 + t6),
            xi * (r4 - r6),
            -xi * (q4 - q6),
            t5 * (1.0 - 2.0 * eta) - xi * (t4 + t5),
            -1.0 + r5 * (1.0 - 2.0 * eta) + xi * (r4 - r5),
            -q5 * (1.0 - 2.0 * eta) - xi * (q4 - q5),
        ])
        B = np.vstack([
            y31 * hx_xi + y12 * hx_eta,
            -x31 * hy_xi - x12 * hy_eta,
            -x31 * hx_xi - x12 * hx_eta + y31 * hy_xi + y12 * hy_eta,
        ]) / area2
        K += (w * abs(area2)) * (B.T @ Db @ B)
    return K


@register(
    "TRIA3",
    n_nodes=3,
    dofs_per_node=(0, 1, 2, 3, 4, 5),
    family="shell",
    description="Three-node flat shell: CST membrane + DKT bending, 18 DOF",
    aliases=("CTRIA3", "TRI3", "TRIA", "SHELL3"),
)
def tria3(ctx: ElementContext) -> ElementMatrices:
    R, xy, _ = shell_frame(ctx.coords[:3])
    Dm = plane_stress_D(ctx.mat)
    area_tmp = 0.5 * abs(
        (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
        - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1])
    )
    sp = _shell_props(ctx, area_tmp)
    t = sp["t"]

    k_loc = np.zeros((18, 18))
    kd_loc = np.zeros((18, 18))
    area, km = _cst_membrane(xy, Dm, t)
    area = abs(area)
    if sp["membrane"]:
        _scatter(k_loc, km, _T3_MEM_IX)
    if sp["bending"]:
        Db = Dm * (t**3 / 12.0) * sp["i_ratio"]
        kb = dkt_bending_stiffness(xy, Db)
        _scatter(k_loc, kb, _T3_BEN_IX)

    if sp["membrane"] and sp["bending"]:
        c = ctx.drill_factor * ctx.mat.E * t * area
        _scatter(kd_loc, _drilling(3, c), _T3_DRILL_IX)

    mpa, rpa = sp["mass_per_area"], sp["rot_inertia_per_area"]
    m_loc = np.zeros((18, 18))
    tri_mass = (mpa * area / 12.0) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    if ctx.lumped_mass:
        tri_mass = np.diag(tri_mass.sum(axis=1))
    for a in range(3):
        for b in range(3):
            for comp in _mass_components(sp):
                m_loc[6 * a + comp, 6 * b + comp] += tri_mass[a, b]
    rot = rpa * area / 3.0
    for a in range(3):
        for comp in _rotary_components(sp):
            m_loc[6 * a + comp, 6 * a + comp] += rot

    return ElementMatrices(
        dofs=[(nid, comp) for nid in ctx.node_ids[:3] for comp in range(6)],
        k=_to_global(k_loc + kd_loc, R, 3),
        m=_to_global(m_loc, R, 3),
        k_drill=_to_global(kd_loc, R, 3),
    )


# --------------------------------------------------------------------------
# QUAD4
# --------------------------------------------------------------------------


def _quad_shape(xi: float, eta: float):
    n = 0.25 * np.array(
        [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
    )
    dn = 0.25 * np.array(
        [
            [-(1 - eta), -(1 - xi)],
            [(1 - eta), -(1 + xi)],
            [(1 + eta), (1 + xi)],
            [-(1 + eta), (1 - xi)],
        ]
    )
    return n, dn


#: 2x2 Gauss rule with the bilinear shape functions evaluated once.
_Q4_GAUSS2 = tuple(
    (tuple(pt), float(w), *_quad_shape(float(pt[0]), float(pt[1])))
    for pt, w in zip(*gauss_2d(2), strict=True)
)


def _mitc4_tying_rows(xy: np.ndarray) -> tuple[np.ndarray, ...]:
    """Covariant transverse shear rows at the four MITC4 tying points.

    Returns ``(B_rz@A, B_sz@B, B_rz@C, B_sz@D)`` with the tying points
    ``A(0,-1)``, ``B(1,0)``, ``C(0,1)``, ``D(-1,0)`` and the plate DOF ordering
    ``[w, rx, ry]`` per node.
    """
    out = []
    for (xi, eta), direction in (
        ((0.0, -1.0), 0),
        ((1.0, 0.0), 1),
        ((0.0, 1.0), 0),
        ((-1.0, 0.0), 1),
    ):
        n, dn = _quad_shape(xi, eta)
        J = dn.T @ xy  # rows: d(x,y)/dxi and d(x,y)/deta
        dxd, dyd = J[direction, 0], J[direction, 1]
        row = np.zeros(12)
        for i in range(4):
            row[3 * i] = dn[i, direction]
            row[3 * i + 1] = -n[i] * dyd
            row[3 * i + 2] = n[i] * dxd
        out.append(row)
    return tuple(out)


def _quad_jacobian(xy: np.ndarray, dn: np.ndarray):
    J = dn.T @ xy
    det = float(np.linalg.det(J))
    if det <= 0.0:
        if det == 0.0:
            raise ValueError("degenerate QUAD4 (zero Jacobian)")
        # Reversed node ordering: |det| keeps the integration positive.
        det = abs(det)
    dnxy = np.linalg.solve(J, dn.T).T
    return det, dnxy


@register(
    "QUAD4",
    n_nodes=4,
    dofs_per_node=(0, 1, 2, 3, 4, 5),
    family="shell",
    description=(
        "Four-node flat shell: bilinear membrane (2x2 Gauss) + Reissner-Mindlin "
        "plate with MITC4 assumed shear, 24 DOF"
    ),
    aliases=("CQUAD4", "QUAD", "SHELL4", "Q4"),
)
def quad4(ctx: ElementContext) -> ElementMatrices:
    R, xy, _ = shell_frame(ctx.coords[:4])
    Dm = plane_stress_D(ctx.mat)

    area = 0.0
    for _pt, w, _n, dn in _Q4_GAUSS2:
        det, _ = _quad_jacobian(xy, dn)
        area += w * det

    sp = _shell_props(ctx, area)
    t = sp["t"]
    Db = Dm * (t**3 / 12.0) * sp["i_ratio"]
    Ds = np.eye(2) * (sp["kappa"] * ctx.mat.G * t)

    k_loc = np.zeros((24, 24))
    kd_loc = np.zeros((24, 24))
    m_loc = np.zeros((24, 24))

    mem_rows = _Q4_MEM_IX
    ben_rows = _Q4_BEN_IX

    tie_a, tie_b, tie_c, tie_d = _mitc4_tying_rows(xy) if sp["bending"] else (None,) * 4

    mpa = sp["mass_per_area"]
    for (xi, eta), w, n, dn in _Q4_GAUSS2:
        det, g = _quad_jacobian(xy, dn)
        scale = w * det

        if sp["membrane"]:
            Bm = np.zeros((3, 8))
            for i in range(4):
                Bm[0, 2 * i] = g[i, 0]
                Bm[1, 2 * i + 1] = g[i, 1]
                Bm[2, 2 * i] = g[i, 1]
                Bm[2, 2 * i + 1] = g[i, 0]
            _scatter(k_loc, (scale * t) * (Bm.T @ Dm @ Bm), mem_rows)

        if sp["bending"]:
            Bb = np.zeros((3, 12))
            for i in range(4):
                Bb[0, 3 * i + 2] = g[i, 0]
                Bb[1, 3 * i + 1] = -g[i, 1]
                Bb[2, 3 * i + 1] = -g[i, 0]
                Bb[2, 3 * i + 2] = g[i, 1]
            _scatter(k_loc, scale * (Bb.T @ Db @ Bb), ben_rows)

            # MITC4 assumed transverse shear: no locking and, unlike reduced
            # integration, no spurious zero energy mode.
            b_rz = 0.5 * (1.0 - eta) * tie_a + 0.5 * (1.0 + eta) * tie_c
            b_sz = 0.5 * (1.0 - xi) * tie_d + 0.5 * (1.0 + xi) * tie_b
            Bs = np.linalg.solve(dn.T @ xy, np.vstack([b_rz, b_sz]))
            _scatter(k_loc, scale * (Bs.T @ Ds @ Bs), ben_rows)

        # Consistent translational mass (u, v, w share the bilinear basis).
        nn = np.outer(n, n) * (scale * mpa)
        for a in range(4):
            for b in range(4):
                for comp in _mass_components(sp):
                    m_loc[6 * a + comp, 6 * b + comp] += nn[a, b]

    if ctx.lumped_mass:
        m_loc = np.diag(m_loc.sum(axis=1))

    rot = sp["rot_inertia_per_area"] * area / 4.0
    for a in range(4):
        for comp in _rotary_components(sp):
            m_loc[6 * a + comp, 6 * a + comp] += rot

    if sp["membrane"] and sp["bending"]:
        c = ctx.drill_factor * ctx.mat.E * t * area
        _scatter(kd_loc, _drilling(4, c), _Q4_DRILL_IX)

    return ElementMatrices(
        dofs=[(nid, comp) for nid in ctx.node_ids[:4] for comp in range(6)],
        k=_to_global(k_loc + kd_loc, R, 4),
        m=_to_global(m_loc, R, 4),
        k_drill=_to_global(kd_loc, R, 4),
    )
