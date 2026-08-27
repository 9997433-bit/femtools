#!/usr/bin/env python3
"""Sensor selection with Effective Independence (Kammer EFI).

Toy pretest case from the acceptance table (case 8): a 10-DOF fixed-free
spring-mass chain (unit masses and stiffnesses, modes from a dense eigen
oracle), 2 target modes, 4 sensors to place. A good selection keeps the
retained-row AutoMAC close to identity: max off-diagonal < 0.15.

Also cross-checks against MAC-based backward elimination.

See docs/algorithms/pretest.md and docs/ACCEPTANCE.md (case 8).
"""

from __future__ import annotations

import numpy as np
from femtools.correlation.mac import mac_matrix
from femtools.pretest.efi import effective_independence
from femtools.pretest.sensor import eliminate_by_mac

N_DOF = 10
N_TARGET_MODES = 2
N_SENSORS = 4


def chain_modes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mass-normalized modes of a fixed-free unit spring-mass chain (oracle)."""
    k = 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    k[-1, -1] = 1.0  # free end
    lam, phi = np.linalg.eigh(k)  # M = I -> eigh output is mass-normalized
    return np.sqrt(lam) / (2.0 * np.pi), phi


def main() -> int:
    np.set_printoptions(precision=3, suppress=True)
    freq_hz, phi = chain_modes(N_DOF)
    targets = phi[:, :N_TARGET_MODES]  # candidate rows = all 10 DOFs
    print(f"target modes at {freq_hz[0]:.4f} and {freq_hz[1]:.4f} Hz "
          f"(unit chain, {N_DOF} DOF)")

    # --- EFI backward elimination ------------------------------------------
    efi = effective_independence(targets, n_sensors=N_SENSORS)
    print(f"\nEFI selected sensor DOFs (0-based): {sorted(efi.selected)}")
    print(f"final E_D values: {efi.ed_final}")
    print(f"log det(Q) history: {efi.logdet_history}")

    mac_sel = mac_matrix(targets[sorted(efi.selected), :])
    max_offdiag = np.max(np.abs(mac_sel - np.diag(np.diag(mac_sel))))
    print(f"AutoMAC of selected rows:\n{mac_sel}")
    print(f"max off-diagonal MAC = {max_offdiag:.4f}  (acceptance: < 0.15)")

    # --- cross-check: MAC-based elimination --------------------------------
    mac_elim = eliminate_by_mac(targets, n_sensors=N_SENSORS)
    print(f"\neliminate_by_mac kept DOFs: {sorted(mac_elim.selected)} "
          f"(final max off-diag = {mac_elim.max_offdiag:.4f})")

    # --- reference: full candidate set (for contrast) ----------------------
    mac_full = mac_matrix(targets)
    full_offdiag = np.max(np.abs(mac_full - np.diag(np.diag(mac_full))))
    print(f"for reference, all-candidate off-diag MAC = {full_offdiag:.4f}")

    ok = bool(
        max_offdiag < 0.15
        and mac_elim.max_offdiag < 0.15
        and len(efi.selected) == N_SENSORS
    )
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
