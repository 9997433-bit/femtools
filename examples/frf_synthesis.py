#!/usr/bin/env python3
"""FRF synthesis: modal superposition vs direct dynamic-stiffness inversion.

Acceptance case 7b: on a 10-element BEAM2 cantilever with 20 retained modes
and light Rayleigh damping (~1% at the band anchors), the modal FRF must match
the direct FRF within 5% relative L2 norm on the 0.2-0.8 f_max band. Rayleigh
damping is used on both sides so the two solutions model identical physics
(modal zeta_r = alpha/(2 w_r) + beta w_r / 2 vs assembled C = alpha M + beta K).

Saves a magnitude/phase overlay to frf_synthesis.png.

See docs/algorithms/dynamics.md and docs/ACCEPTANCE.md (case 7b).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from femtools.core.model import FEModel
from femtools.dynamics.frf import direct_frf, modal_frf
from femtools.fea.assemble import assemble_km
from femtools.fea.eigen import solve_modes

L = 1.0
B, H = 0.02, 0.03
E, NU, RHO = 210e9, 0.3, 7850.0
N_ELEM = 10
N_MODES = 20
ZETA_TARGET = 0.01  # ~1% damping at both Rayleigh anchor frequencies


def build_model() -> FEModel:
    a = B * H
    iy, iz = H * B**3 / 12.0, B * H**3 / 12.0
    j = H * B**3 * (1.0 / 3.0 - 0.21 * (B / H) * (1.0 - B**4 / (12.0 * H**4)))
    model = FEModel(name="frf-demo")
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


def main() -> int:
    model = build_model()
    asm = assemble_km(model)
    modal = solve_modes(model, n_modes=N_MODES, assembly=asm)
    f_max = modal.freq_hz[-1]

    # Rayleigh anchors at first and last retained mode, equal zeta at both:
    # alpha = 2 z w1 w2 / (w1 + w2), beta = 2 z / (w1 + w2)
    w1, w2 = 2.0 * np.pi * modal.freq_hz[0], 2.0 * np.pi * f_max
    damping = {
        "alpha": 2.0 * ZETA_TARGET * w1 * w2 / (w1 + w2),
        "beta": 2.0 * ZETA_TARGET / (w1 + w2),
    }

    tip, mid = N_ELEM + 1, N_ELEM // 2 + 1
    dofs = [(tip, "uz"), (mid, "uz")]    # tip driving point + midspan transfer
    # modal_frf indexes the full DOF space of the mode shapes; direct_frf works
    # on the SPC-reduced free partition of K and M -- two views of the same DOFs.
    in_full = [asm.dof_map.index(tip, "uz")]
    out_full = [asm.dof_map.index(n, c) for n, c in dofs]
    in_free = [free_index(asm, tip, "uz")]
    out_free = [free_index(asm, n, c) for n, c in dofs]
    freq_hz = np.linspace(0.2 * f_max, 0.8 * f_max, 400)

    h_modal = modal_frf(modal, in_full, out_full, freq_hz, damping)
    h_direct = direct_frf(asm.Kff, asm.Mff, in_free, out_free, freq_hz, damping)

    print(f"band: {freq_hz[0]:.1f} - {freq_hz[-1]:.1f} Hz "
          f"({N_MODES} modes, f_max = {f_max:.1f} Hz)")
    errs = []
    for i_out, out_dof in enumerate(dofs):
        hm, hd = h_modal.H[i_out, 0, :], h_direct.H[i_out, 0, :]
        err = np.linalg.norm(hm - hd) / np.linalg.norm(hd)
        errs.append(err)
        print(f"  output {out_dof}: rel L2(modal - direct) = {err:.3%}")

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for i_out, out_dof in enumerate(dofs):
        ax_mag.semilogy(freq_hz, np.abs(h_modal.H[i_out, 0, :]),
                        label=f"modal, out={out_dof}")
        ax_mag.semilogy(freq_hz, np.abs(h_direct.H[i_out, 0, :]), "--",
                        label=f"direct, out={out_dof}")
        ax_ph.plot(freq_hz, np.degrees(np.angle(h_modal.H[i_out, 0, :])))
        ax_ph.plot(freq_hz, np.degrees(np.angle(h_direct.H[i_out, 0, :])), "--")
    ax_mag.set_ylabel("|H| [m/N]")
    ax_mag.legend(fontsize=8)
    ax_ph.set_xlabel("frequency [Hz]")
    ax_ph.set_ylabel("phase [deg]")
    fig.suptitle("Cantilever receptance: modal superposition vs direct solve")
    fig.tight_layout()
    fig.savefig("frf_synthesis.png", dpi=150)
    print("plot written to frf_synthesis.png")

    ok = bool(max(errs) < 0.05)
    print("PASS" if ok else "FAIL", "(tol: 5% rel L2 on the 0.2-0.8 f_max band)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
