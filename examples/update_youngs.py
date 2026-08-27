#!/usr/bin/env python3
"""Model updating: recover a seeded Young's modulus error from frequencies.

Golden updating scenario (ACCEPTANCE case 6): synthetic "test" frequencies are
produced by a cantilever whose E is 10% higher than the nominal model. Because
the whole stiffness scales with E (and M does not depend on it), lambda_r is
exactly linear in E and sensitivity-based updating must recover E within 2%.

See docs/algorithms/updating.md.
"""

from __future__ import annotations

import numpy as np
from femtools.core.model import FEModel
from femtools.fea.eigen import solve_modes
from femtools.updating.updater import update_model

L = 1.0
B, H = 0.02, 0.03
E0, NU, RHO = 210e9, 0.3, 7850.0
E_TRUE = 1.10 * E0          # the "as-built" modulus the test data comes from
N_ELEM = 5
N_FREQS = 3                 # measured frequencies used as residuals


def build_model(young: float) -> FEModel:
    a = B * H
    iy, iz = H * B**3 / 12.0, B * H**3 / 12.0
    j = H * B**3 * (1.0 / 3.0 - 0.21 * (B / H) * (1.0 - B**4 / (12.0 * H**4)))
    model = FEModel(name="cantilever-update")
    for i in range(N_ELEM + 1):
        model.add_node(id=i + 1, xyz=(L * i / N_ELEM, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=young, nu=NU, rho=RHO)
    model.add_property(id=1, type="beam", material_id=1, A=a, Iy=iy, Iz=iz, J=j)
    for i in range(N_ELEM):
        model.add_element(id=i + 1, type="BEAM2", nodes=(i + 1, i + 2), property_id=1)
    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
    return model


def main() -> int:
    # Synthetic test data from the perturbed structure
    f_test = solve_modes(build_model(E_TRUE), n_modes=N_FREQS).freq_hz
    print(f"'measured' frequencies (E = 1.10 E0): {np.round(f_test, 4)} Hz")

    nominal = build_model(E0)
    f_start = solve_modes(nominal, n_modes=N_FREQS).freq_hz
    print(f"nominal FE frequencies (E = E0):      {np.round(f_start, 4)} Hz")
    print(f"initial max rel freq error: {np.max(np.abs(f_start - f_test) / f_test):.3%}")

    result = update_model(
        nominal,
        parameters=[{"type": "material", "id": 1, "name": "E", "lower": 0.5, "upper": 2.0}],
        measured={"freq_hz": f_test},
        method="analytic",
        max_iter=20,
        tol=1e-8,
    )

    e_updated = result.p[0] * E0
    rel_err = abs(e_updated - E_TRUE) / E_TRUE
    print(f"\nconverged: {result.converged} in {len(result.history)} iterations")
    print(f"recovered E = {e_updated:.6e} Pa (true {E_TRUE:.6e} Pa)")
    print(f"relative parameter error: {rel_err:.3e}  (acceptance: < 2e-2)")

    f_final = solve_modes(result.model, n_modes=N_FREQS).freq_hz
    print(f"updated FE frequencies: {np.round(f_final, 4)} Hz")

    ok = bool(result.converged and rel_err < 0.02)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
