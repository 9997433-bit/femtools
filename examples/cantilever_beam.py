#!/usr/bin/env python3
"""Cantilever beam modal analysis vs Euler-Bernoulli theory.

Builds a steel cantilever from 10 BEAM2 elements (rectangular 20x30 mm
section, so the two bending planes are distinct), solves the first modes and
compares them against the analytical Euler-Bernoulli frequencies

    f_n = (beta_n L)^2 / (2 pi L^2) * sqrt(E I / (rho A)),

with cos(bL)cosh(bL) = -1 roots beta_n L = 1.875104, 4.694091, 7.854757.
Also verifies mass normalization Phi^T M Phi = I.

See docs/algorithms/fea.md and docs/ACCEPTANCE.md (cases 2, 3a).
"""

from __future__ import annotations

import numpy as np

from femtools.core.model import FEModel
from femtools.fea.assemble import assemble_km
from femtools.fea.eigen import solve_modes

# --- structure definition -------------------------------------------------
L = 1.0          # m
B, H = 0.02, 0.03  # section width (local y) and depth (local z), m
E, NU, RHO = 210e9, 0.3, 7850.0
N_ELEM = 10

A = B * H
IY = H * B**3 / 12.0   # weak-plane bending
IZ = B * H**3 / 12.0   # strong-plane bending
# St-Venant torsion constant, rectangular section (h/b = 1.5)
J = H * B**3 * (1.0 / 3.0 - 0.21 * (B / H) * (1.0 - B**4 / (12.0 * H**4)))

BETA_L = np.array([1.8751040687, 4.6940911330, 7.8547574382])


def build_model() -> FEModel:
    model = FEModel(name="cantilever")
    for i in range(N_ELEM + 1):
        model.add_node(id=i + 1, xyz=(L * i / N_ELEM, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="beam", material_id=1, A=A, Iy=IY, Iz=IZ, J=J)
    for i in range(N_ELEM):
        model.add_element(id=i + 1, type="BEAM2", nodes=(i + 1, i + 2), property_id=1)
    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
    return model


def analytical_bending_freqs() -> np.ndarray:
    """First three EB frequencies for each bending plane, merged ascending."""
    freqs = [
        (bl**2 / (2.0 * np.pi * L**2)) * np.sqrt(E * inertia / (RHO * A))
        for inertia in (IY, IZ)
        for bl in BETA_L
    ]
    return np.sort(np.array(freqs))


def main() -> int:
    model = build_model()
    modal = solve_modes(model, n_modes=8)

    f_ref = analytical_bending_freqs()          # 6 lowest modes are all bending
    f_fem = modal.freq_hz[: len(f_ref)]
    rel_err = np.abs(f_fem - f_ref) / f_ref

    print("Cantilever beam: FE (10 x BEAM2) vs Euler-Bernoulli theory")
    print(f"{'mode':>4} {'f_FE [Hz]':>12} {'f_EB [Hz]':>12} {'rel err':>10}")
    for i, (ff, fr, err) in enumerate(zip(f_fem, f_ref, rel_err, strict=True), start=1):
        print(f"{i:>4} {ff:>12.4f} {fr:>12.4f} {err:>10.2e}")

    # Mass normalization check: Phi^T M Phi = I (ACCEPTANCE case 3a)
    asm = assemble_km(model)
    gram = modal.modes.T @ (asm.M @ modal.modes)
    norm_err = np.max(np.abs(gram - np.eye(gram.shape[0])))
    print(f"\nmax |Phi^T M Phi - I| = {norm_err:.2e}")

    ok = bool(np.all(rel_err < 0.02) and norm_err < 1e-8)
    print("PASS" if ok else "FAIL", "(tol: 2% on frequencies, 1e-8 on normalization)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
