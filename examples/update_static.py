#!/usr/bin/env python3
"""Static model updating: recover a Young's modulus error from a proof load.

The static counterpart of examples/update_youngs.py (Friswell & Mottershead,
*Finite Element Model Updating in Structural Dynamics*, ch. 3): instead of a
modal survey, the "test" is a proof load with dial gauges. The truth cantilever
is 10% stiffer than the nominal FE model, so its measured tip deflection under
a fixed tip force is 1/1.1 times the nominal prediction; because the static
response is exactly inversely proportional to a global stiffness multiplier,
`update_from_static` must recover E to round-off — the 2% acceptance tolerance
(case 6) only allows for FD sensitivities and multi-parameter runs.

The Hermite BEAM2 element is nodally exact for end loads, so the measured tip
deflection is also pinned against delta = F L^3 / (3 E I) (ACCEPTANCE case 11)
before any updating happens.

See docs/algorithms/updating.md and docs/ACCEPTANCE.md (Round-8/Round-9 status).
"""

from __future__ import annotations

import numpy as np

try:
    from femtools.core.model import FEModel
    from femtools.fea.static import solve_static
    from femtools.updating import update_from_static
except ImportError as exc:  # kernels not on this tree yet
    print(f"SKIP: static-updating kernels not importable on this tree ({exc})")
    raise SystemExit(0) from None

L = 2.0
E0, NU, RHO = 210e9, 0.3, 7850.0
E_TRUE = 1.10 * E0            # the as-built modulus the gauges actually see
A, IY, IZ, J = 8.0e-4, 3.0e-8, 6.0e-8, 9.0e-8
N_ELEM = 8
TIP_FORCE = -1.0e3            # N, transverse (z) at the free end


def build_model(young: float) -> FEModel:
    """BEAM2 cantilever along x, clamped at node 1, tip load at the last node."""
    model = FEModel(name="cantilever-static-update")
    model.add_material(id=1, type="isotropic", E=young, nu=NU, rho=RHO)
    model.add_property(id=1, type="beam", material_id=1, A=A, Iy=IY, Iz=IZ, J=J)
    for i in range(N_ELEM + 1):
        model.add_node(id=i + 1, xyz=(L * i / N_ELEM, 0.0, 0.0))
    for i in range(N_ELEM):
        model.add_element(id=i + 1, type="BEAM2", nodes=(i + 1, i + 2), property_id=1)
    model.add_spc(node_id=1, mask=(True,) * 6)
    model.add_load(node_id=N_ELEM + 1, force=(0.0, 0.0, TIP_FORCE))
    return model


def gauge_readings(model: FEModel, dofs: list[tuple[int, int]]) -> np.ndarray:
    """The truth model's deflections at ``dofs`` — the synthetic test data."""
    u = solve_static(model, None)
    table = model.dof_map()
    return np.array([float(u[table[key]]) for key in dofs])


def main() -> int:
    tip = N_ELEM + 1
    checks: list[bool] = []

    # -- 1. the "test": a proof load on the truth structure ------------------
    truth = build_model(E_TRUE)
    u_tip = gauge_readings(truth, [(tip, 2)])[0]
    delta_exact = TIP_FORCE * L**3 / (3.0 * E_TRUE * IY)   # case 11 closed form
    dev = abs(u_tip - delta_exact) / abs(delta_exact)
    print(f"measured tip deflection (E = 1.10 E0): {u_tip:.6e} m")
    ok = dev < 1e-9
    print(f"  [{'ok' if ok else 'FAIL'}] matches F L^3 / 3 E I: rel dev {dev:.2e} "
          "(Hermite nodal exactness, tol 1e-9)")
    checks.append(ok)

    # -- 2. one dial gauge updates the nominal model -------------------------
    nominal = build_model(E0)
    u_nom = gauge_readings(nominal, [(tip, 2)])[0]
    print(f"nominal FE prediction (E = E0):        {u_nom:.6e} m "
          f"({abs(u_nom - u_tip) / abs(u_tip):.2%} off)")

    result = update_from_static(nominal, {(tip, "uz"): u_tip})
    e_updated = result["E"] * E0
    rel_err = abs(e_updated - E_TRUE) / E_TRUE
    print(f"\nconverged: {result.converged} in {result.n_iter} iterations")
    print(f"recovered E = {e_updated:.6e} Pa (true {E_TRUE:.6e} Pa)")
    ok = bool(result.converged) and rel_err < 0.02
    print(f"  [{'ok' if ok else 'FAIL'}] relative E error {rel_err:.2e} "
          "(acceptance case 6: < 2e-2; static path reaches round-off)")
    checks.append(ok)

    ok = nominal.materials[1].E == E0
    print(f"  [{'ok' if ok else 'FAIL'}] input model not mutated "
          f"(E still {nominal.materials[1].E:.3e} Pa)")
    checks.append(ok)

    # -- 3. a gauge sweep: tip + midspan + tip rotation, same answer ---------
    dofs = [(tip, 2), (N_ELEM // 2 + 1, 2), (tip, 4)]
    targets = gauge_readings(truth, dofs)
    swept = update_from_static(
        nominal, targets, dofs=[(tip, "uz"), (N_ELEM // 2 + 1, "uz"), (tip, "ry")]
    )
    rel_err3 = abs(swept["E"] * E0 - E_TRUE) / E_TRUE
    ok = bool(swept.converged) and rel_err3 < 0.02
    print(f"  [{'ok' if ok else 'FAIL'}] 3-gauge sweep recovers the same E: "
          f"rel err {rel_err3:.2e}, rms residual {swept.rms_error:.2e}")
    checks.append(ok)

    print("PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
