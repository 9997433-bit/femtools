#!/usr/bin/env python3
"""TET10 quadratic tetrahedron: constant-strain patch, rigid modes, ZZ-SPR.

The Round-10 element is the 10-node quadratic tetrahedron (4 corners + 6
midside nodes; Zienkiewicz & Taylor, *The Finite Element Method*, Vol. 1;
Bathe, *Finite Element Procedures* §5.3). Its quadratic field contains the
complete linear one, so a mesh carrying an exactly linear displacement field
u = a + A x must report the exact constant strain sym(A) and the exact
constant stress D sym(A) at every recovery point — the constant-strain patch
test, to 1e-12 even on a distorted patch (ACCEPTANCE Round-10 row 29). A
single free-free TET10 (solid: 3 translational DOFs per node) must keep
exactly 6 rigid-body modes.

The last section exercises the second Round-10 recovery kernel when it has
landed: `recover_spr`, Zienkiewicz-Zhu superconvergent patch recovery (IJNME
33(7), 1992). A linear polynomial is fitted over the centroid samples of the
elements incident on each node and evaluated at the node; a constant stress
state lies inside the fitted polynomial space, so it must survive at every
node exactly — same sharp gate as `average_nodal`, different estimator
(ACCEPTANCE Round-10 row 30).

Both sections skip with a message instead of crashing while the Round-10
kernels are not on this tree. See docs/algorithms/fea.md (sections 14-15)
and docs/ACCEPTANCE.md (Round-10 status).
"""

from __future__ import annotations

import numpy as np

try:
    from femtools.core.model import FEModel
    from femtools.fea import assemble_km
    from femtools.fea.eigen import solve_modes
    from femtools.fea.elements import tet10  # noqa: F401  (registers etype "TET10")
    from femtools.fea.recover import recover_stress
except ImportError as exc:  # kernels not on this tree yet
    print(f"SKIP: TET10 kernels not importable on this tree ({exc})")
    raise SystemExit(0) from None

E, NU, RHO = 210.0e9, 0.3, 7850.0

# Parent tetrahedron (deliberately irregular, no right angles), metres.
CORNERS = 0.1 * np.array(
    [
        [0.00, 0.00, 0.00],
        [1.10, 0.05, -0.02],
        [0.40, 0.95, 0.06],
        [0.35, 0.30, 1.05],
    ]
)
# A strictly interior point splits it into 4 sub-tets sharing one buried node.
INTERIOR = CORNERS.T @ np.array([0.30, 0.24, 0.21, 0.25])

# Nastran CTETRA / textbook midside order: G5..G10 sit on edges
# (G1,G2), (G2,G3), (G3,G1), (G1,G4), (G2,G4), (G3,G4).
EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))

# Imposed linear field u = A0 + GRAD x (GRAD deliberately non-symmetric: the
# recovery must report sym(GRAD), the rotation must drop out).
A0 = np.array([2.0e-4, -1.0e-4, 3.0e-4])
GRAD = 1.0e-3 * np.array(
    [
        [1.00, 0.40, -0.30],
        [-0.20, 0.80, 0.50],
        [0.10, -0.60, 1.20],
    ]
)


def isotropic_d(e: float, nu: float) -> np.ndarray:
    """6x6 isotropic D for Voigt order (xx, yy, zz, xy, yz, zx), eng. shear."""
    lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = e / (2.0 * (1.0 + nu))
    d = np.zeros((6, 6))
    d[:3, :3] = lam
    d[np.arange(3), np.arange(3)] += 2.0 * mu
    d[np.arange(3, 6), np.arange(3, 6)] = mu
    return d


def exact_state() -> tuple[np.ndarray, np.ndarray]:
    """Exact Voigt strain (engineering shear) and stress of the imposed field."""
    eps_t = 0.5 * (GRAD + GRAD.T)
    eps = np.array(
        [
            eps_t[0, 0], eps_t[1, 1], eps_t[2, 2],
            2.0 * eps_t[0, 1], 2.0 * eps_t[1, 2], 2.0 * eps_t[0, 2],
        ]
    )
    return eps, isotropic_d(E, NU) @ eps


def oriented(corners: np.ndarray, tet: tuple[int, int, int, int]) -> tuple[int, ...]:
    """Return the 4 corner indices reordered for a positive Jacobian."""
    a, b, c, d = (corners[i] for i in tet)
    if np.linalg.det(np.column_stack([b - a, c - a, d - a])) < 0.0:
        return (tet[0], tet[1], tet[3], tet[2])
    return tet


def build_mesh(corners: np.ndarray, tets: list[tuple[int, int, int, int]],
               etype: str, name: str) -> FEModel:
    """TET10 (or TET4) mesh over ``tets``; midside nodes deduplicated."""
    model = FEModel(name=name)
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="solid", material_id=1)

    node_id: dict[tuple[float, ...], int] = {}

    def nid_of(xyz: np.ndarray) -> int:
        key = tuple(np.round(xyz, 12))
        if key not in node_id:
            node_id[key] = len(node_id) + 1
            model.add_node(id=node_id[key], xyz=tuple(xyz))
        return node_id[key]

    for eid, tet in enumerate(tets, start=1):
        cix = oriented(corners, tet)
        conn = [nid_of(corners[i]) for i in cix]
        if etype == "TET10":
            for i, j in EDGES:
                conn.append(nid_of(0.5 * (corners[cix[i]] + corners[cix[j]])))
        model.add_element(id=eid, type=etype, nodes=tuple(conn), property_id=1)
    return model


def imposed_field(model: FEModel, asm: object) -> np.ndarray:
    """Global displacement vector of the field u = A0 + GRAD x at every node."""
    u = np.zeros(asm.ndof)
    for nid, node in model.nodes.items():
        disp = A0 + GRAD @ np.asarray(node.xyz)
        for k, label in enumerate(("ux", "uy", "uz")):
            u[asm.dof_map[(nid, label)]] = disp[k]
    return u


def check(label: str, value: float, tol: float) -> bool:
    ok = value < tol
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {value:.2e} (tol {tol:.0e})")
    return ok


def main() -> int:
    checks: list[bool] = []
    eps_exact, sig_exact = exact_state()
    sig_scale = float(np.max(np.abs(sig_exact)))
    eps_scale = float(np.max(np.abs(eps_exact)))
    patch_tets = [(4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)]
    corners5 = np.vstack([CORNERS, INTERIOR])

    # -- 1. constant-strain patch: 4 distorted TET10 around a buried node -----
    print("1. TET10 constant-strain patch (4 distorted elements, 1 interior node):")
    model = build_mesh(corners5, patch_tets, "TET10", "tet10-patch")
    asm = assemble_km(model)
    print(f"   {len(model.elements)} TET10, {len(model.nodes)} nodes, "
          f"{asm.ndof} DOFs (free-free, displacement-imposed)")
    result = recover_stress(model, imposed_field(model, asm), assembly=asm)
    if result.skipped or result.n_elements != len(model.elements):
        print(f"  [FAIL] recovery skipped elements: {result.skipped}")
        checks.append(False)
    else:
        dev_e = float(np.max(np.abs(result.strain_basic - eps_exact))) / eps_scale
        dev_s = float(np.max(np.abs(result.stress_basic - sig_exact))) / sig_scale
        checks.append(check("strain = sym(GRAD) at every centroid (rel)", dev_e, 1e-12))
        checks.append(check("stress = D sym(GRAD)               (rel)", dev_s, 1e-12))

    # rigid-body mass: unit-x translation r gives r^T M r = rho * V(parent tet)
    v_parent = abs(np.linalg.det((CORNERS[1:] - CORNERS[0]).T)) / 6.0
    r = np.zeros(asm.ndof)
    for nid in model.nodes:
        r[asm.dof_map[(nid, "ux")]] = 1.0
    m_dev = abs(float(r @ (asm.M @ r)) - RHO * v_parent) / (RHO * v_parent)
    checks.append(check("consistent mass: r^T M r = rho V   (rel)", m_dev, 1e-10))

    # -- 2. one free-free TET10: exactly 6 rigid-body modes -------------------
    print("2. single free-free TET10 (30 DOFs):")
    single = build_mesh(CORNERS, [(0, 1, 2, 3)], "TET10", "tet10-single")
    modal = solve_modes(single, n_modes=12)
    freqs = np.asarray(modal.freq_hz, dtype=float)
    n_zero = int(np.sum(freqs < 1.0e-4 * freqs[-1]))
    ok = n_zero == 6
    elastic = f"; first elastic {freqs[6]:.1f} Hz" if freqs.size > 6 else ""
    print(f"  [{'ok' if ok else 'FAIL'}] rigid-body modes: {n_zero} "
          f"(want exactly 6){elastic}")
    checks.append(ok)

    # -- 3. ZZ superconvergent patch recovery (kernel-gated) ------------------
    checks.extend(spr_section(corners5, patch_tets, result, model, sig_exact, sig_scale))

    print("PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


def spr_section(corners5: np.ndarray, patch_tets: list, tet10_stress: object,
                tet10_model: FEModel, sig_exact: np.ndarray,
                sig_scale: float) -> list[bool]:
    """`recover_spr` keeps a constant stress state exact at every node.

    Runs on the TET4 twin of the patch (centroids are the Barlow points of
    linear elements — the superconvergent samples the ZZ fit uses); the TET10
    mesh is offered too, tolerating a documented skip. Section skipped whole
    while the kernel has not landed.
    """
    try:
        from femtools.fea.recover import recover_spr
    except ImportError:
        print("3. recover_spr: kernel not on this tree yet -- section skipped")
        return []

    checks: list[bool] = []
    print("3. recover_spr (Zienkiewicz-Zhu 1992): constant patch exact at nodes:")
    tet4 = build_mesh(corners5, patch_tets, "TET4", "tet4-spr-patch")
    asm4 = assemble_km(tet4)
    stress4 = recover_stress(tet4, imposed_field(tet4, asm4), assembly=asm4)
    nodal = recover_spr(stress4, tet4)
    values = np.asarray(nodal.stress)
    dev = float(np.max(np.abs(values - sig_exact))) / sig_scale
    checks.append(check("TET4 patch: SPR stress at every node   (rel)", dev, 1e-10))
    ok_all = len(nodal.node_ids) == len(tet4.nodes)
    print(f"  [{'ok' if ok_all else 'FAIL'}] every node carries a value "
          f"({len(nodal.node_ids)}/{len(tet4.nodes)})")
    checks.append(ok_all)

    try:
        nodal10 = recover_spr(tet10_stress, tet10_model)
    except NotImplementedError as exc:
        print(f"   TET10 in SPR skipped by the kernel (documented): {exc}")
    else:
        dev10 = float(np.max(np.abs(np.asarray(nodal10.stress) - sig_exact))) / sig_scale
        checks.append(check("TET10 patch: SPR stress at every node  (rel)", dev10, 1e-10))
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
