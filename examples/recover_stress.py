#!/usr/bin/env python3
"""Stress recovery: element-centroid stresses vs closed-form references.

Recovery is *differentiate, then constitute*: gather the element displacements,
evaluate the strain-displacement matrix B at the centroid — the bilinear /
trilinear element's superconvergent (Barlow) point — and back-substitute the
same constitutive matrix D the stiffness was built from. Two consequences are
demonstrated here:

1. a constant-stress state is recovered *exactly* (the patch-test property):
   a HEX8 bar under uniform tension reports sxx = P/A at every centroid, all
   other components zero, von Mises = P/A;
2. beam elements report their local end resultants (extras): a BEAM2
   cantilever with a transverse tip load carries the analytic linear bending
   moment M(x) = P (L - x), exactly, because the Hermite interpolation is
   exact for end loads.

When the Round-8 kernel has landed, the nodal-average section runs as well:
average_nodal spreads centroid values onto incident nodes (1/n_adj), and on a
constant-stress patch every nodal value stays exact.

See docs/algorithms/fea.md (sections 10, 13) and docs/ACCEPTANCE.md.
"""

from __future__ import annotations

import numpy as np

try:
    from femtools.core.model import FEModel
    from femtools.fea import assemble_km, recover_stress, solve_static
except ImportError as exc:  # kernels not on this tree yet
    print(f"SKIP: stress recovery kernels not importable on this tree ({exc})")
    raise SystemExit(0) from None

E, NU, RHO = 210.0e9, 0.3, 7850.0

# -- part 1: HEX8 bar under uniform tension ---------------------------------
BAR_L, BAR_W = 0.4, 0.1          # 4 cubes of side BAR_W
N_SOLID = 4
TENSION = 2.0e5                  # N, total axial force -> sxx = P/A = 20 MPa

# -- part 2: BEAM2 cantilever with a transverse tip load ---------------------
BEAM_L, N_BEAM = 1.0, 4
TIP = -1.0e3                     # N, applied along -z at the tip


def solid_bar() -> FEModel:
    """HEX8 bar along x; restraints are compatible with exact uniform tension.

    The root face is *not* clamped: ux = 0 on the whole face, uy = 0 only on
    the y = 0 edge, uz = 0 only on the z = 0 edge, so the Poisson contraction
    u = (P/AE) (x, -nu y, -nu z) is unobstructed and the constant state is
    exact everywhere, not just away from the boundary.
    """
    model = FEModel(name="tension-bar")
    ids: dict[tuple[int, int, int], int] = {}
    counter = 1
    for i in range(N_SOLID + 1):
        for j in range(2):
            for k in range(2):
                ids[(i, j, k)] = counter
                model.add_node(id=counter, xyz=(BAR_L * i / N_SOLID, BAR_W * j, BAR_W * k))
                counter += 1
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="solid", material_id=1)
    for i in range(N_SOLID):
        model.add_element(
            id=i + 1,
            type="HEX8",
            nodes=(
                ids[(i, 0, 0)], ids[(i + 1, 0, 0)], ids[(i + 1, 1, 0)], ids[(i, 1, 0)],
                ids[(i, 0, 1)], ids[(i + 1, 0, 1)], ids[(i + 1, 1, 1)], ids[(i, 1, 1)],
            ),
            property_id=1,
        )
    for j in range(2):
        for k in range(2):
            model.add_spc(
                node_id=ids[(0, j, k)],
                mask=(True, j == 0, k == 0, False, False, False),
            )
    for j in range(2):  # uniform traction: equal corner shares on the end face
        for k in range(2):
            model.add_load(node_id=ids[(N_SOLID, j, k)], force=(TENSION / 4.0, 0.0, 0.0))
    return model


def beam_cantilever() -> FEModel:
    model = FEModel(name="tip-loaded-beam")
    for i in range(N_BEAM + 1):
        model.add_node(id=i + 1, xyz=(BEAM_L * i / N_BEAM, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="beam", material_id=1,
                       A=6.0e-4, Iy=5.0e-8, Iz=5.0e-8, J=8.0e-8)
    for i in range(N_BEAM):
        model.add_element(id=i + 1, type="BEAM2", nodes=(i + 1, i + 2), property_id=1)
    model.add_spc(node_id=1, mask=(True,) * 6)
    model.add_load(node_id=N_BEAM + 1, force=(0.0, 0.0, TIP))
    return model


def check(label: str, value: float, tol: float) -> bool:
    ok = value < tol
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {value:.2e} (tol {tol:.0e})")
    return ok


def main() -> int:
    checks: list[bool] = []

    # -- 1. constant-stress patch: HEX8 bar in tension -----------------------
    exact = TENSION / BAR_W**2
    print(f"1. HEX8 bar under uniform tension (exact sxx = P/A = {exact / 1e6:.1f} MPa):")
    model = solid_bar()
    asm = assemble_km(model)
    u = solve_static(model, assembly=asm)
    result = recover_stress(model, u, assembly=asm)

    sxx = result.stress[:, 0]
    others = result.stress[:, 1:]
    dev_s = float(np.max(np.abs(sxx - exact)) / exact)
    dev_o = float(np.max(np.abs(others)) / exact)
    dev_vm = float(np.max(np.abs(result.von_mises - exact)) / exact)
    checks.append(check("sxx = P/A at every centroid   (rel)", dev_s, 1e-9))
    checks.append(check("all other components zero     (rel)", dev_o, 1e-9))
    checks.append(check("von Mises = P/A               (rel)", dev_vm, 1e-9))

    # -- 2. BEAM2 resultants: linear bending moment ---------------------------
    print("2. BEAM2 cantilever, tip load: end moments vs M(x) = P (L - x):")
    beam = beam_cantilever()
    asm_b = assemble_km(beam)
    u_b = solve_static(beam, assembly=asm_b)
    res_b = recover_stress(beam, u_b, assembly=asm_b)

    dev_m = 0.0
    for i in range(N_BEAM):
        x_start = BEAM_L * i / N_BEAM
        want = TIP * (BEAM_L - x_start)          # bending about local y
        got = float(res_b.extras[i + 1]["moments"][0])
        dev_m = max(dev_m, abs(got - want) / abs(TIP * BEAM_L))
    dev_n = max(
        abs(float(res_b.extras[i + 1]["axial_force"])) for i in range(N_BEAM)
    ) / abs(TIP)
    checks.append(check("end moment My1 = P (L - x)    (rel)", dev_m, 1e-12))
    checks.append(check("no axial force under Fz       (rel)", dev_n, 1e-12))

    # -- 3. nodal averaging (Round 8; runs once the kernel has landed) --------
    try:
        from femtools.fea.recover import average_nodal
    except ImportError:
        average_nodal = None
    if average_nodal is None:
        print("3. average_nodal: kernel not on this tree yet -- section skipped")
    else:
        print("3. average_nodal: constant patch stays exact at every node:")
        nodal = average_nodal(result, model)
        values = np.asarray(nodal.stress)
        dev = float(np.max(np.abs(values[:, 0] - exact)) / exact)
        checks.append(check("averaged nodal sxx = P/A      (rel)", dev, 1e-9))

    ok = all(checks)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
