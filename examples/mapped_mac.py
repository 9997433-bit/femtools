#!/usr/bin/env python3
"""Test-grid mapping: MAC between mode sets living on different node orders.

A modal test measures on digitized sensor points in the lab frame; the FE
model numbers its own nodes. Before any MAC can be formed, each measurement
point has to be matched to the mesh node it sits on, and the FE shapes have to
be gathered on the test points in the test order. This example composes the
Round-7/Round-8 kernels that do exactly that:

    ids, dist = map_nearest_nodes(xyz_test_in_fe_frame, model)
    phi_fe    = mapped_mode_matrix(modal, ids, dofs=("uz",))
    mac       = mac_matrix(phi_test, phi_fe)

The "test campaign" here is a translated copy of a QUAD4 cantilever plate with
deliberately shuffled node ids: translation changes no element geometry, so
its modes are the reference modes up to sign, and after undoing the offset
and matching nearest nodes the mapped MAC diagonal must be exactly 1
(ACCEPTANCE Round-8 mapped-shape row: measured 4.4e-16 on this tree). No new
formula is involved — the MAC is the classical one, only the bookkeeping is
exercised (docs/algorithms/correlation.md). When the Round-9 one-call wrapper
`femtools.correlation.mapped_mac` has landed, the last section checks that it
reproduces the composed matrix.
"""

from __future__ import annotations

import numpy as np

try:
    from femtools.core.model import FEModel
    from femtools.correlation import mac_matrix, map_nearest_nodes
    from femtools.correlation.dofmap import mapped_mode_matrix
    from femtools.fea.eigen import solve_modes
except ImportError as exc:  # kernels not on this tree yet
    print(f"SKIP: mapped-correlation kernels not importable on this tree ({exc})")
    raise SystemExit(0) from None

LX, LY, T = 0.6, 0.3, 0.004
NX, NY = 6, 3
E, NU, RHO = 7.0e10, 0.33, 2700.0
OFFSET = np.array([0.35, -0.20, 0.15])   # lab frame = FE frame + OFFSET
N_MODES = 6
SEED = 42


def plate(offset: np.ndarray, id_map: dict[int, int]) -> FEModel:
    """Clamped-edge QUAD4 plate; node ids and coordinates per the caller."""
    model = FEModel(name="mapped-mac-plate")
    model.add_material(id=1, type="isotropic", E=E, nu=NU, rho=RHO)
    model.add_property(id=1, type="shell", material_id=1, t=T)
    grid: dict[tuple[int, int], int] = {}
    seq = 1
    for i in range(NX + 1):
        for j in range(NY + 1):
            grid[(i, j)] = id_map[seq]
            xyz = np.array([LX * i / NX, LY * j / NY, 0.0]) + offset
            model.add_node(id=id_map[seq], xyz=tuple(xyz))
            seq += 1
    eid = 1
    for i in range(NX):
        for j in range(NY):
            model.add_element(
                id=eid, type="QUAD4", property_id=1,
                nodes=(grid[(i, j)], grid[(i + 1, j)],
                       grid[(i + 1, j + 1)], grid[(i, j + 1)]),
            )
            eid += 1
    for j in range(NY + 1):
        model.add_spc(node_id=grid[(0, j)], mask=(True,) * 6)
    return model


def check(label: str, value: float, tol: float) -> bool:
    ok = value < tol
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {value:.2e} (tol {tol:.0e})")
    return ok


def main() -> int:
    n_nodes = (NX + 1) * (NY + 1)
    checks: list[bool] = []

    # -- the FE model, and the "test article": translated + renumbered -------
    fe_model = plate(np.zeros(3), {i: i for i in range(1, n_nodes + 1)})
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_nodes) + 101      # test node ids: shuffled, offset
    test_model = plate(OFFSET, {i + 1: int(perm[i]) for i in range(n_nodes)})

    modal_fe = solve_modes(fe_model, n_modes=N_MODES)
    modal_test = solve_modes(test_model, n_modes=N_MODES)
    df = np.max(np.abs(np.asarray(modal_test.freq_hz) - np.asarray(modal_fe.freq_hz)))
    print(f"plate modes {np.round(np.asarray(modal_fe.freq_hz), 2)} Hz")
    checks.append(check("translation leaves the spectrum untouched (Hz)", df, 1e-6))

    # -- "digitize" the sensors: every test node, in a shuffled sweep order --
    order = rng.permutation(n_nodes)
    test_ids = [int(perm[k]) for k in order]
    xyz_lab = np.array([test_model.nodes[nid].xyz for nid in test_ids])
    phi_test = mapped_mode_matrix(modal_test, test_ids, dofs=("uz",))

    # -- map the lab points onto the FE mesh and gather the FE shapes --------
    ids_fe, dist = map_nearest_nodes(xyz_lab - OFFSET, fe_model, unique=True)
    print(f"matched {len(ids_fe)} sensors onto FE nodes "
          f"(one-to-one: {len(set(ids_fe)) == n_nodes})")
    checks.append(check("worst sensor-to-node distance (m)", float(np.max(dist)), 1e-9))

    phi_fe = mapped_mode_matrix(modal_fe, ids_fe, dofs=("uz",))
    mac = mac_matrix(phi_test, phi_fe)
    dev = float(np.max(np.abs(np.diag(mac) - 1.0)))
    off = float(np.max(np.abs(mac - np.diag(np.diag(mac)))))
    checks.append(check("mapped MAC diagonal: max |diag - 1|", dev, 1e-10))
    print(f"  (largest off-diagonal MAC: {off:.3f} — distinct plate modes)")

    # -- a mis-digitized sensor is flagged, not silently attached ------------
    rogue = np.vstack([xyz_lab - OFFSET, [10.0, 10.0, 10.0]])
    ids_r, dist_r = map_nearest_nodes(rogue, fe_model, tol=1e-3)
    ok = ids_r[-1] == -1 and np.all(np.asarray(ids_r[:-1]) != -1)
    print(f"  [{'ok' if ok else 'FAIL'}] rogue sensor beyond tol maps to -1 "
          f"(true distance kept: {dist_r[-1]:.1f} m)")
    checks.append(ok)

    # -- Round 9: the one-call wrapper reproduces the composed matrix --------
    checks.extend(wrapper_section(phi_test, xyz_lab, modal_fe, fe_model, mac))

    print("PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


def wrapper_section(phi_test: np.ndarray, xyz_lab: np.ndarray, modal_fe: object,
                    fe_model: FEModel, mac_composed: np.ndarray) -> list[bool]:
    """`mapped_mac` = map + gather + MAC in one call (Round 9).

    Nothing new is computed — the wrapper is contractually the bookkeeping of
    the three calls above done once. Skipped while the kernel has not landed.
    """
    try:
        from femtools.correlation import mapped_mac
    except ImportError:
        print("mapped_mac wrapper: kernel not on this tree yet -- section skipped")
        return []

    result = mapped_mac(phi_test, xyz_lab - OFFSET, modal_fe, fe_model,
                        unique=True, dofs=("uz",))
    dev = float(np.max(np.abs(np.asarray(result) - mac_composed)))
    ok_id = dev == 0.0
    print(f"  [{'ok' if ok_id else 'FAIL'}] one-call mapped_mac == composed matrix "
          f"(max |dev| = {dev:.1e}, want bit-identical)")
    ok_diag = float(result.min_diagonal) > 1.0 - 1e-10
    print(f"  [{'ok' if ok_diag else 'FAIL'}] result.min_diagonal = "
          f"{result.min_diagonal:.15f} (want 1 to 1e-10)")
    return [ok_id, ok_diag]


if __name__ == "__main__":
    raise SystemExit(main())
