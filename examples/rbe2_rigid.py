#!/usr/bin/env python3
"""RBE2 rigid links: exact master-slave elimination, no penalty springs.

An RBE2 welds dependent grid points to one independent grid point through the
small-rotation rigid-body kinematics

    u_d = u_i + theta_i x r,    theta_d = theta_i,    r = x_d - x_i,

imposed the classical master-slave way (Cook ch. 13; Zienkiewicz & Taylor):
the dependent DOFs are eliminated through a transformation G, system matrices
follow by congruence G^T A G and loads by virtual work f -> G^T f. Because no
penalty stiffness enters, the constraint is exact: a free-free structure with
a rigid arm keeps exactly 6 rigid-body modes, and a force on a rigid offset
arrives at the independent node as the equivalent force *and* moment.

This example demonstrates all three properties on a BEAM2 cantilever with a
rigid arm welded to its tip, and — when the Round-8 interpolation kernel has
landed — the corresponding RBE3 gates (dependent = weighted average of the
independents, still 6 rigid-body modes, equal force shares).

See docs/algorithms/fea.md (sections 11-12) and docs/ACCEPTANCE.md
(Round-7/Round-8 status blocks).
"""

from __future__ import annotations

import numpy as np

try:
    from femtools.core.model import FEModel
    from femtools.fea import apply_rbe2, assemble_km, solve_modes, solve_static
except ImportError as exc:  # kernels not on this tree yet
    print(f"SKIP: RBE2 kernels not importable on this tree ({exc})")
    raise SystemExit(0) from None

E, NU, RHO = 2.1e11, 0.3, 7800.0
ARM = np.array([0.4, 0.0, 0.3])          # rigid offset r from beam tip to node 3
FORCE = np.array([0.0, 200.0, -500.0])   # load applied at the offset node


def beam(n_elem: int = 6, clamped: bool = True, with_arm: bool = True) -> FEModel:
    """BEAM2 cantilever along x; node ``n_elem + 2`` is the rigid-arm tip."""
    model = FEModel(name="rbe2-demo")
    for i in range(n_elem + 1):
        model.add_node(id=i + 1, xyz=(i / n_elem, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="beam", material_id=1,
                       A=6.0e-4, Iy=5.0e-8, Iz=5.0e-8, J=8.0e-8)
    for i in range(n_elem):
        model.add_element(id=i + 1, type="BEAM2", nodes=(i + 1, i + 2), property_id=1)
    if clamped:
        model.add_spc(node_id=1, mask=(True,) * 6)
    if with_arm:
        tip = n_elem + 1
        model.add_node(id=tip + 1, xyz=tuple(np.array([1.0, 0.0, 0.0]) + ARM))
        model.add_rbe2(id=1, independent=tip, dependents=(tip + 1,))
    return model


def check(label: str, value: float, tol: float) -> bool:
    ok = value < tol
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {value:.2e} (tol {tol:.0e})")
    return ok


def main() -> int:
    n_elem = 6
    tip, arm_node = n_elem + 1, n_elem + 2
    checks: list[bool] = []

    # -- 1. rigid-arm kinematics after a static solve -----------------------
    model = beam(n_elem)
    asm = assemble_km(model)                      # honors model.rbe2 by default
    loads = {(arm_node, c): float(f) for c, f in enumerate(FORCE) if f != 0.0}
    u = solve_static(model, loads, assembly=asm)

    u_tip = u[asm.dof_map.node_dofs(tip)]
    u_arm = u[asm.dof_map.node_dofs(arm_node)]
    print("1. load on the rigid offset, kinematics of the welded node:")
    dev_rot = np.max(np.abs(u_arm[3:] - u_tip[3:])) / np.max(np.abs(u_tip[3:]))
    expect = u_tip[:3] + np.cross(u_tip[3:], ARM)
    dev_tra = np.max(np.abs(u_arm[:3] - expect)) / np.max(np.abs(expect))
    checks.append(check("theta_dependent = theta_independent (rel)", dev_rot, 1e-12))
    checks.append(check("u_d = u_i + theta_i x r          (rel)", dev_tra, 1e-12))

    # -- 2. virtual work: G^T f carries force + moment of the offset --------
    print("2. load transfer G^T f vs the analytic force + moment r x F:")
    transform = apply_rbe2(model)                 # same table, explicit transform
    f_full = np.zeros(transform.n_dof)
    dm = transform.dof_map
    for c, val in enumerate(FORCE):
        f_full[dm.index(arm_node, c)] = val
    f_ind = transform.to_independent(f_full)
    got = np.array([f_ind[dm.index(tip, c)] for c in range(6)])
    want = np.concatenate([FORCE, np.cross(ARM, FORCE)])
    dev = np.max(np.abs(got - want)) / np.max(np.abs(want))
    checks.append(check("force+moment at the independent node (rel)", dev, 1e-12))

    # ... and the whole solution matches loading the tip with F and r x F.
    plain = beam(n_elem, with_arm=False)
    loads_eq = {(tip, c): float(v) for c, v in enumerate(want) if v != 0.0}
    u_eq = solve_static(plain, loads_eq)
    asm_eq = assemble_km(plain)
    dev_u = np.max(np.abs(u_eq[asm_eq.dof_map.node_dofs(tip)] - u_tip)) / np.max(np.abs(u_tip))
    checks.append(check("tip motion = equivalent force+moment case (rel)", dev_u, 1e-10))

    # -- 3. a free-free structure with a rigid arm keeps 6 rigid-body modes -
    print("3. free-free beam with the rigid arm:")
    free = beam(n_elem, clamped=False)
    freqs = solve_modes(free, n_modes=9).freq_hz
    n_rbm = int(np.count_nonzero(freqs < 1e-6))
    f_ref = solve_modes(beam(n_elem, clamped=False, with_arm=False), n_modes=9).freq_hz
    dev_el = abs(freqs[6] - f_ref[6]) / f_ref[6]
    ok_rbm = n_rbm == 6
    print(f"  [{'ok' if ok_rbm else 'FAIL'}] rigid-body modes: {n_rbm} (want exactly 6)")
    checks.append(ok_rbm)
    checks.append(check("first elastic mode untouched by the arm (rel)", dev_el, 1e-9))

    # -- 4. RBE3 interpolation (Round 8; runs once the kernel has landed) ---
    checks.extend(rbe3_section())

    ok = all(checks)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def rbe3_section() -> list[bool]:
    """RBE3: dependent node = weighted average of the independents.

    Round-8 gates (docs/ACCEPTANCE.md): a mass RBE3-tied to a triangle of
    independent nodes keeps exactly 6 rigid-body modes, and with equal
    weights a force on the dependent node splits into equal translational
    shares (G^T f). Skipped while ``apply_rbe3`` has not landed.
    """
    try:
        from femtools.fea.mpc import apply_rbe3
    except ImportError:
        apply_rbe3 = None
    if apply_rbe3 is None:
        print("4. RBE3 interpolation: kernel not on this tree yet -- section skipped")
        return []

    print("4. RBE3 interpolation (Round 8): mass averaged onto a triangle:")
    model = FEModel(name="rbe3-demo")
    corners = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    for i, xyz in enumerate(corners):
        model.add_node(id=i + 1, xyz=xyz)
    model.add_node(id=4, xyz=(1.0 / 3.0, 1.0 / 3.0, 0.0))   # triangle centroid
    model.add_material(id=1, type="isotropic", E=7.0e10, nu=0.33, rho=2700.0)
    model.add_property(id=1, type="shell", material_id=1, t=0.005)
    model.add_property(id=2, type="lumped", m=1.5)
    model.add_element(id=1, type="TRIA3", nodes=(1, 2, 3), property_id=1)
    model.add_element(id=2, type="MASS", nodes=(4,), property_id=2)
    model.add_rbe3(id=1, dependent=4, independents=(1, 2, 3))  # translations, equal weights

    checks: list[bool] = []
    freqs = solve_modes(model, n_modes=8).freq_hz
    n_rbm = int(np.count_nonzero(freqs < 1e-6))
    ok_rbm = n_rbm == 6
    print(f"  [{'ok' if ok_rbm else 'FAIL'}] rigid-body modes: {n_rbm} (want exactly 6)")
    checks.append(ok_rbm)

    transform = apply_rbe3(model)
    dm = transform.dof_map
    f_full = np.zeros(transform.n_dof)
    f_full[dm.index(4, 2)] = -9.0                 # 9 N down on the dependent node
    f_ind = transform.to_independent(f_full)
    shares = np.array([f_ind[dm.index(n, 2)] for n in (1, 2, 3)])
    dev = np.max(np.abs(shares - (-3.0))) / 3.0
    checks.append(check("equal weights -> equal force shares (rel)", dev, 1e-12))
    return checks


if __name__ == "__main__":
    raise SystemExit(main())
