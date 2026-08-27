#!/usr/bin/env python3
"""Model reduction: Guyan static condensation, IRS, and SEREP on a cantilever.

Reduces the 10-element BEAM2 cantilever (same structure as
examples/cantilever_beam.py) to 6 master DOFs -- the uy/uz translations at
x = 0.3 L, 0.6 L and the tip -- and checks the classic invariants:

  1. Guyan is *exact* for statics: a load applied at master DOFs gives the
     same master displacements in the reduced and in the full model.
  2. Guyan eigenvalues are upper bounds (Rayleigh-Ritz) and degrade with
     mode number; the first mode per plane is accurate to a fraction of 1%.
  3. IRS (O'Callahan's Improved Reduced System) adds the first-order inertia
     correction and beats Guyan on the average frequency error.
  4. SEREP with as many masters as retained modes reproduces the retained
     modal frequencies and shapes essentially exactly.

Uses the Round-4 kernel `femtools.fea.reduction` (owner R4-O1, frozen in
.agent_workspace/REMAINING.md); this example fails with ImportError until it
lands. See docs/algorithms/fea.md section 9 and docs/ACCEPTANCE.md (case 17).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from femtools.core.model import FEModel
from femtools.fea.assemble import assemble_km
from femtools.fea.eigen import solve_modes
from femtools.fea.reduction import guyan, irs, serep

L = 1.0
B, H = 0.02, 0.03
E, NU, RHO = 210e9, 0.3, 7850.0
N_ELEM = 10
MASTER_NODES = (4, 7, 11)      # x = 0.3 L, 0.6 L, tip
MASTER_COMPS = ("uy", "uz")    # both bending planes
N_KEPT = 6                     # compared modes = the 6 lowest (all bending)


def build_model() -> FEModel:
    a = B * H
    iy, iz = H * B**3 / 12.0, B * H**3 / 12.0
    j = H * B**3 * (1.0 / 3.0 - 0.21 * (B / H) * (1.0 - B**4 / (12.0 * H**4)))
    model = FEModel(name="reduction-demo")
    for i in range(N_ELEM + 1):
        model.add_node(id=i + 1, xyz=(L * i / N_ELEM, 0.0, 0.0))
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="beam", material_id=1, A=a, Iy=iy, Iz=iz, J=j)
    for i in range(N_ELEM):
        model.add_element(id=i + 1, type="BEAM2", nodes=(i + 1, i + 2), property_id=1)
    model.add_spc(node_id=1, mask=(True, True, True, True, True, True))
    return model


def free_index(asm, node_id: int, component: str) -> int:
    """Position of (node, component) inside the free-DOF partition."""
    pos = np.flatnonzero(asm.free_dof == asm.dof_map.index(node_id, component))
    if pos.size != 1:
        raise ValueError(f"DOF ({node_id}, {component}) is not a free DOF")
    return int(pos[0])


def reduction_basis(res: object, n_full: int, n_master: int) -> np.ndarray:
    """(n_full, n_master) basis T from a ReductionResult, tuple, or bare array.

    The frozen contract says `guyan(K, master) -> T, Krr` and
    `serep(phi, master_rows) -> T`; this helper accepts a ReductionResult
    with a `.T` field, a plain `(T, K_red, ...)` tuple, or a bare ndarray.
    """
    cand: object
    if isinstance(res, np.ndarray):
        cand = res
    else:
        t = getattr(res, "T", None)
        if isinstance(t, np.ndarray):
            cand = t
        elif isinstance(res, (tuple, list)) and len(res) > 0:
            cand = res[0]
        else:
            raise TypeError(f"cannot find a reduction basis in {type(res).__name__}")
    T = np.asarray(cand, dtype=float)
    if T.shape == (n_master, n_full) and n_master != n_full:
        T = T.T
    if T.shape != (n_full, n_master):
        raise ValueError(f"reduction basis has shape {T.shape}, "
                         f"expected {(n_full, n_master)}")
    return T


def reduced_freqs(T: np.ndarray, K, M) -> np.ndarray:
    """Frequencies [Hz] of the reduced model K_r = T'KT, M_r = T'MT."""
    kr = T.T @ (K @ T)
    mr = T.T @ (M @ T)
    lam = sla.eigh(0.5 * (kr + kr.T), 0.5 * (mr + mr.T), eigvals_only=True)
    return np.sqrt(np.clip(lam, 0.0, None)) / (2.0 * np.pi)


def main() -> int:
    model = build_model()
    asm = assemble_km(model)
    modal = solve_modes(model, n_modes=N_KEPT + 2, assembly=asm)
    K, M = asm.Kff, asm.Mff
    n_free = K.shape[0]

    master_dofs = [(n, c) for n in MASTER_NODES for c in MASTER_COMPS]
    master = np.array([free_index(asm, n, c) for n, c in master_dofs])
    n_m = master.size
    f_full = modal.freq_hz[:N_KEPT]
    print(f"cantilever, {n_free} free DOFs -> {n_m} masters "
          f"{[f'{n}:{c}' for n, c in master_dofs]}")
    print(f"full model, first {N_KEPT} modes: {np.round(f_full, 3)} Hz\n")
    checks: list[bool] = []

    # --- 1. Guyan: static exactness at the masters --------------------------
    t_guyan = reduction_basis(guyan(K, master), n_free, n_m)
    f = np.zeros(n_free)
    f[free_index(asm, 11, "uz")] = 1.0        # unit tip load, a master DOF
    u_full = spla.spsolve(K.tocsc(), f)
    k_red = t_guyan.T @ (K @ t_guyan)
    u_red = np.linalg.solve(k_red, f[master])
    static_err = float(np.max(np.abs(u_red - u_full[master]))
                       / np.max(np.abs(u_full[master])))
    print(f"Guyan static exactness at masters: rel err = {static_err:.2e} "
          "(exact up to round-off)")
    checks.append(static_err < 1e-8)

    # --- 2/3. Guyan vs IRS eigenvalue accuracy ------------------------------
    f_guyan = reduced_freqs(t_guyan, K, M)[:N_KEPT]
    t_irs = reduction_basis(irs(K, M, master), n_free, n_m)
    f_irs = reduced_freqs(t_irs, K, M)[:N_KEPT]

    # --- 4. SEREP on the free partition of the mode shapes ------------------
    phi = modal.modes[asm.free_dof, :N_KEPT]           # (n_free, 6)
    t_serep = reduction_basis(serep(phi, master), n_free, n_m)
    recon = float(np.max(np.abs(phi - t_serep @ phi[master, :]))
                  / np.max(np.abs(phi)))
    f_serep = reduced_freqs(t_serep, K, M)[:N_KEPT]

    err_g = np.abs(f_guyan - f_full) / f_full
    err_i = np.abs(f_irs - f_full) / f_full
    err_s = np.abs(f_serep - f_full) / f_full
    print(f"\n{'mode':>4} {'full [Hz]':>10} {'Guyan':>10} {'err':>9} "
          f"{'IRS':>10} {'err':>9} {'SEREP':>10} {'err':>9}")
    for r in range(N_KEPT):
        print(f"{r + 1:>4} {f_full[r]:>10.3f} {f_guyan[r]:>10.3f} {err_g[r]:>9.2e} "
              f"{f_irs[r]:>10.3f} {err_i[r]:>9.2e} {f_serep[r]:>10.3f} {err_s[r]:>9.2e}")

    print(f"\nGuyan upper-bound property: min(f_red - f_full) = "
          f"{np.min(f_guyan - f_full):+.3e} Hz (must be >= 0)")
    checks.append(bool(np.all(f_guyan >= f_full * (1.0 - 1e-9))))
    checks.append(err_g[0] < 0.01)                      # first mode < 1%
    print(f"mean rel err: Guyan {np.mean(err_g):.2e}, IRS {np.mean(err_i):.2e} "
          "(IRS must improve on Guyan)")
    checks.append(float(np.mean(err_i)) <= float(np.mean(err_g)))
    print(f"SEREP: shape reconstruction max dev = {recon:.2e}, "
          f"max freq err = {np.max(err_s):.2e} (both ~ round-off)")
    checks.append(recon < 1e-8)
    checks.append(float(np.max(err_s)) < 1e-6)

    ok = all(checks)
    print("\nPASS" if ok else "\nFAIL", f"({sum(checks)}/{len(checks)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
