#!/usr/bin/env python3
"""Free-interface CMS (Rubin / MacNeal) vs Craig-Bampton on a split beam.

A fixed-fixed steel beam (20 BEAM2 elements) is cut at midspan into two
10-element components: A is clamped at x = 0 and free at the interface, B is
clamped at x = L and free at the interface. Each component is reduced with
free-interface normal modes plus residual-flexibility attachment modes
(Rubin keeps the residual inertia, MacNeal drops it), the two superelements
are coupled on the 6 shared interface DOFs, and the coupled frequencies are
compared against the unsplit 20-element reference and the analytic
fixed-fixed roots cosh(bL)cos(bL) = 1 (bL = 4.7300, 7.8532, ...). The
existing fixed-interface Craig-Bampton reduction runs as a baseline.

Uses the Round-4 kernel `femtools.dynamics.cms_free` (owner R4-O2, merged).
A `FreeCMSResult` orders its generalized coordinates [kept modes...,
residual modes...] -- none of them is a physical DOF -- so the coupling goes
through `free_interface_assembly`, which ties the *physical* interface DOFs
of the two components rigidly (null-space elimination) and condenses
whatever carries no mass (MacNeal's residual block). `CraigBamptonResult`
does expose its boundary DOFs as the leading generalized coordinates, so the
CB baseline is assembled primally by the local `couple` harness. See
docs/algorithms/dynamics.md section 9 and docs/ACCEPTANCE.md (case 18).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from femtools.core.model import FEModel
from femtools.dynamics.cms_free import free_interface_assembly, macneal, rubin
from femtools.dynamics.craig_bampton import craig_bampton
from femtools.fea.assemble import assemble_km
from femtools.fea.eigen import solve_modes

L = 1.0
B, H = 0.02, 0.03
E, NU, RHO = 210e9, 0.3, 7850.0
N_ELEM = 20                      # full model; each component gets half
IFACE_COMPS = ("ux", "uy", "uz", "rx", "ry", "rz")
N_COMP_MODES = 8                 # kept free-interface modes per component
N_CMP = 4                        # coupled modes compared against the reference
BETA_L = np.array([4.7300407449, 7.8532046241, 10.9956078381])


def build_beam(node_ids: range, x0: float, x1: float, spc_node: int,
               name: str) -> FEModel:
    a = B * H
    iy, iz = H * B**3 / 12.0, B * H**3 / 12.0
    j = H * B**3 * (1.0 / 3.0 - 0.21 * (B / H) * (1.0 - B**4 / (12.0 * H**4)))
    model = FEModel(name=name)
    ids = list(node_ids)
    n_el = len(ids) - 1
    for k, nid in enumerate(ids):
        model.add_node(id=nid, xyz=(x0 + (x1 - x0) * k / n_el, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="beam", material_id=1, A=a, Iy=iy, Iz=iz, J=j)
    for k in range(n_el):
        model.add_element(id=k + 1, type="BEAM2", nodes=(ids[k], ids[k + 1]),
                          property_id=1)
    model.add_spc(node_id=spc_node, mask=(True,) * 6)
    return model


def free_index(asm, node_id: int, component: str) -> int:
    pos = np.flatnonzero(asm.free_dof == asm.dof_map.index(node_id, component))
    if pos.size != 1:
        raise ValueError(f"DOF ({node_id}, {component}) is not a free DOF")
    return int(pos[0])


def couple(res_a: object, res_b: object, n_b: int) -> np.ndarray:
    """Primal assembly of two superelements sharing their n_b boundary DOFs.

    Assumes generalized coordinate order [boundary..., modal...] in both
    reduced models (the CraigBamptonResult convention -- FreeCMSResult does
    NOT follow it, see the module docstring). Returns the coupled
    frequencies in Hz. QZ is used instead of eigh so a singular mass block
    would surface as infinite eigenvalues to discard, not as an error.
    """
    ka, ma = res_a.K, res_a.M
    kb, mb = res_b.K, res_b.M
    na, nb_modal = ka.shape[0] - n_b, kb.shape[0] - n_b
    n = n_b + na + nb_modal
    ia = np.r_[np.arange(n_b), n_b + np.arange(na)]
    ib = np.r_[np.arange(n_b), n_b + na + np.arange(nb_modal)]
    kg = np.zeros((n, n))
    mg = np.zeros((n, n))
    kg[np.ix_(ia, ia)] += ka
    kg[np.ix_(ib, ib)] += kb
    mg[np.ix_(ia, ia)] += ma
    mg[np.ix_(ib, ib)] += mb
    lam = sla.eig(0.5 * (kg + kg.T), 0.5 * (mg + mg.T), right=False)
    lam = lam[np.isfinite(lam)]
    lam = np.real(lam[np.abs(lam.imag) <= 1e-6 * np.maximum(1.0, np.abs(lam.real))])
    return np.sqrt(np.clip(np.sort(lam), 0.0, None)) / (2.0 * np.pi)


def main() -> int:
    mid = N_ELEM // 2 + 1        # shared interface node id (x = L/2)
    full = build_beam(range(1, N_ELEM + 2), 0.0, L, spc_node=1, name="full")
    full.add_spc(node_id=N_ELEM + 1, mask=(True,) * 6)
    f_ref = solve_modes(full, n_modes=N_CMP + 2).freq_hz[:N_CMP]

    a_iy = np.array([(bl**2 / (2.0 * np.pi * L**2)) * np.sqrt(E * inertia / (RHO * B * H))
                     for inertia in (H * B**3 / 12.0, B * H**3 / 12.0)
                     for bl in BETA_L])
    print(f"fixed-fixed beam, {N_ELEM} BEAM2: reference {np.round(f_ref, 2)} Hz")
    print(f"analytic fixed-fixed roots:       {np.round(np.sort(a_iy)[:N_CMP], 2)} Hz\n")

    comp_a = build_beam(range(1, mid + 1), 0.0, L / 2, spc_node=1, name="A")
    comp_b = build_beam(range(mid, N_ELEM + 2), L / 2, L, spc_node=N_ELEM + 1,
                        name="B")
    asms = [assemble_km(comp) for comp in (comp_a, comp_b)]
    bnds = [np.array([free_index(asm, mid, c) for c in IFACE_COMPS])
            for asm in asms]
    ties = [("A", int(ia), "B", int(ib)) for ia, ib in zip(*bnds, strict=True)]

    results: dict[str, np.ndarray] = {}
    for label, reduce_fn in (("rubin", rubin), ("macneal", macneal)):
        red_a, red_b = (reduce_fn(asm.Kff, asm.Mff, bnd, n_modes=N_COMP_MODES)
                        for asm, bnd in zip(asms, bnds, strict=True))
        coupled = free_interface_assembly([("A", red_a), ("B", red_b)], ties)
        results[label] = coupled.freq_hz[:N_CMP]

    cb_a, cb_b = (craig_bampton(asm.Kff, asm.Mff, bnd, n_modes=N_COMP_MODES)
                  for asm, bnd in zip(asms, bnds, strict=True))
    results["craig_bampton"] = couple(cb_a, cb_b, len(IFACE_COMPS))[:N_CMP]

    print(f"{'mode':>4} {'full [Hz]':>10} "
          + " ".join(f"{lbl:>12} {'err':>9}" for lbl in results))
    errs = {lbl: np.abs(f - f_ref) / f_ref for lbl, f in results.items()}
    for r in range(N_CMP):
        row = f"{r + 1:>4} {f_ref[r]:>10.3f}"
        for lbl in results:
            row += f" {results[lbl][r]:>12.3f} {errs[lbl][r]:>9.2e}"
        print(row)

    print(f"\n(each component: {N_COMP_MODES} kept modes + "
          f"{len(IFACE_COMPS)} interface DOFs)")
    checks = [
        bool(np.max(errs["rubin"]) < 0.01),           # Rubin: < 1% on modes 1-4
        bool(np.max(errs["macneal"]) < 0.03),         # MacNeal: < 3% (no residual inertia)
        bool(np.max(errs["craig_bampton"]) < 0.01),   # CB baseline: < 1%
    ]
    for lbl, tol in (("rubin", 0.01), ("macneal", 0.03), ("craig_bampton", 0.01)):
        print(f"  {lbl:>14}: max rel err = {np.max(errs[lbl]):.2e} (tol {tol:g})")

    ok = all(checks)
    print("\nPASS" if ok else "\nFAIL", f"({sum(checks)}/{len(checks)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
