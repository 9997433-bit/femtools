#!/usr/bin/env python3
"""Topometry optimization: per-element thickness field on a cantilever plate.

Topometry sizing gives every element of an *existing* mesh its own design
variable — here the shell thickness — and redistributes a fixed amount of
material for minimum compliance f^T u (Bendsoe & Sigmund, *Topology
Optimization*; the element-wise sizing variant). Unlike topology_simp, which
builds its own structured grid, topometry_optimize works on the model the
caller already has and analyses it with the standard FEA kernel; thickness
sensitivities are analytic (shell stiffness is cubic in t for bending).

The demonstrator is the classic clamped plate with a transverse free-edge
load: bending dominates, so material is worth far more at the root than at
the tip and the uniform start is far from optimal. Expected outcome: the
optimality-criteria loop at least halves the compliance while conserving the
material volume exactly, thickens the root and thins the tip.

See docs/algorithms/optimization.md and docs/ACCEPTANCE.md (Round-7 status).
"""

from __future__ import annotations

import numpy as np

try:
    from femtools.core.model import FEModel
    from femtools.fea.static import solve_static
    from femtools.optimization import topometry_optimize
except ImportError as exc:  # kernels not on this tree yet
    print(f"SKIP: topometry kernels not importable on this tree ({exc})")
    raise SystemExit(0) from None

NX, NY = 6, 3                # element grid (elements numbered column by column)
LX, LY = 0.9, 0.45           # m
T0 = 5.0e-3                  # uniform starting thickness, m
TIP_LOAD = -1.0e3            # N, transverse at the free-edge midside node


def cantilever_plate() -> tuple[FEModel, int]:
    """Clamped-free QUAD4 plate, transverse point load at the free edge."""
    model = FEModel(name="topometry-plate")
    model.add_material(id=1, type="isotropic", E=70.0e9, nu=0.33, rho=2700.0)
    model.add_property(id=1, type="shell", material_id=1, t=T0)
    ids: dict[tuple[int, int], int] = {}
    counter = 1
    for i in range(NX + 1):
        for j in range(NY + 1):
            ids[(i, j)] = counter
            model.add_node(id=counter, xyz=(LX * i / NX, LY * j / NY, 0.0))
            counter += 1
    eid = 1
    for i in range(NX):
        for j in range(NY):
            model.add_element(
                id=eid,
                type="QUAD4",
                nodes=(ids[(i, j)], ids[(i + 1, j)], ids[(i + 1, j + 1)], ids[(i, j + 1)]),
                property_id=1,
            )
            eid += 1
    for j in range(NY + 1):
        model.add_spc(node_id=ids[(0, j)], mask=(True,) * 6)
    tip = ids[(NX, NY // 2)]
    model.add_load(node_id=tip, force=(0.0, 0.0, TIP_LOAD))
    return model, tip


def main() -> int:
    model, tip = cantilever_plate()
    result = topometry_optimize(model, max_iter=200)
    checks: list[bool] = []

    def check(label: str, ok: bool, detail: str) -> None:
        checks.append(bool(ok))
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {detail}")

    print(f"topometry ({result.method}, design={result.design!r}): "
          f"{result.iterations} iterations, converged={result.converged}")
    ratio = result.compliance / result.initial_compliance
    check("compliance at least halved", ratio < 0.5,
          f"{result.initial_compliance:.4e} -> {result.compliance:.4e} J "
          f"(ratio {ratio:.3f}, want < 0.5)")

    vol_dev = abs(result.volume - result.initial_volume) / result.initial_volume
    check("material volume conserved", vol_dev < 1e-9,
          f"rel dev {vol_dev:.2e} (tol 1e-9, constraint "
          f"{result.extras['constraint']!r})")

    lo, hi = result.bounds
    in_bounds = bool(np.all(result.x >= lo - 1e-15) and np.all(result.x <= hi + 1e-15))
    check("thickness bounds respected", in_bounds,
          f"t in [{result.x.min() * 1e3:.2f}, {result.x.max() * 1e3:.2f}] mm "
          f"(bounds [{np.min(lo) * 1e3:.2f}, {np.max(hi) * 1e3:.2f}] mm)")

    # Elements are numbered column by column: ids 1..NY root, last NY the tip.
    t_of = result.to_dict()
    root = float(np.mean([t_of[e] for e in range(1, NY + 1)]))
    tip_col = float(np.mean([t_of[e] for e in range(NX * NY - NY + 1, NX * NY + 1)]))
    check("root thickened, tip thinned", root > tip_col,
          f"mean t root {root * 1e3:.2f} mm > tip column {tip_col * 1e3:.2f} mm")

    check("input model untouched", model.properties[1].t == T0,
          f"property t still {model.properties[1].t * 1e3:.2f} mm")

    # The returned model must reproduce the reported compliance independently.
    u = solve_static(result.model, {(tip, "uz"): TIP_LOAD})
    dof = result.model.dof_map()[(tip, 2)]
    dev = abs(float(u[dof] * TIP_LOAD) - result.compliance) / result.compliance
    check("returned model reproduces the compliance", dev < 1e-9,
          f"rel dev {dev:.2e} (tol 1e-9)")

    print("\nthickness field (mm), root column at the left:")
    for j in reversed(range(NY)):
        row = [t_of[i * NY + j + 1] * 1e3 for i in range(NX)]
        print("   " + "  ".join(f"{t:5.2f}" for t in row))

    ok = all(checks)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
