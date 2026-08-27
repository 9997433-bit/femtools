#!/usr/bin/env python3
"""MAC identities, mode pairing, and the double-mode trap.

Demonstrates with seeded synthetic mode sets:
  1. MAC(self) of an orthonormal basis is the identity matrix.
  2. MAC is invariant to (complex) scaling of either shape.
  3. pair_modes matches a shuffled, perturbed copy back to the original.
  4. Repeated ("double") modes: two exact models can show near-zero pairwise
     MAC inside a degenerate subspace -- only subspace-aware scoring survives.

See docs/algorithms/correlation.md and docs/ACCEPTANCE.md (cases 4a-4c).
"""

from __future__ import annotations

import numpy as np
from femtools.correlation.mac import mac_matrix
from femtools.correlation.pairing import pair_modes

RNG = np.random.default_rng(42)
N_DOF, N_MODES = 12, 5


def main() -> int:
    np.set_printoptions(precision=3, suppress=True)
    checks: list[bool] = []

    # 1. Orthonormal basis => MAC = I
    q, _ = np.linalg.qr(RNG.standard_normal((N_DOF, N_MODES)))
    mac_self = mac_matrix(q)
    off = mac_self - np.diag(np.diag(mac_self))
    print("AutoMAC of an orthonormal basis:\n", mac_self)
    checks.append(np.max(np.abs(np.diag(mac_self) - 1.0)) < 1e-12)
    checks.append(np.max(np.abs(off)) < 1e-10)

    # 2. Scale/phase invariance: MAC(a x, b y) == MAC(x, y)
    scales = RNG.standard_normal(N_MODES) * np.exp(1j * RNG.uniform(0, 2 * np.pi, N_MODES))
    mac_scaled = mac_matrix(q * scales, q * 3.7)
    checks.append(np.max(np.abs(mac_scaled - mac_self)) < 1e-12)
    print(f"\nscale invariance: max |MAC(aX, bX) - MAC(X, X)| = "
          f"{np.max(np.abs(mac_scaled - mac_self)):.2e}")

    # 3. Pairing a shuffled + perturbed copy
    freq_a = np.array([10.0, 25.0, 42.0, 61.0, 88.0])
    perm = RNG.permutation(N_MODES)
    phi_b = q[:, perm] + 0.05 * RNG.standard_normal((N_DOF, N_MODES))
    freq_b = freq_a[perm] * (1.0 + 0.02 * RNG.standard_normal(N_MODES))
    pairs = pair_modes(q, phi_b, freq_a, freq_b, mac_threshold=0.5)
    print("\npairing (index_a -> index_b, MAC):")
    recovered = {}
    for p in pairs:
        print(f"  {p.index_a} -> {p.index_b}   MAC = {p.mac:.3f}")
        recovered[p.index_a] = p.index_b
    checks.append(all(perm[recovered[i]] == i for i in recovered) and len(pairs) == N_MODES)

    # 4. Double modes: same 2-D subspace, rotated 45 degrees
    sub = q[:, :2]
    rot = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)],
                    [np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    sub_rot = sub @ rot
    pairwise = mac_matrix(sub, sub_rot)
    # Subspace MAC (S2MAC): projection of each rotated shape onto span(sub)
    proj = sub @ np.linalg.solve(sub.T @ sub, sub.T @ sub_rot)
    s2mac = np.sum(np.abs(proj) ** 2, axis=0) / np.sum(np.abs(sub_rot) ** 2, axis=0)
    print("\ndouble-mode demo (identical subspace, 45 deg rotation):")
    print("  pairwise MAC:\n", pairwise)
    print("  subspace MAC (S2MAC):", s2mac)
    checks.append(np.max(pairwise) < 0.6)          # pairwise MAC collapses...
    checks.append(np.min(s2mac) > 1.0 - 1e-10)     # ...but the subspace agrees exactly

    ok = all(checks)
    print("\nPASS" if ok else "\nFAIL", f"({sum(checks)}/{len(checks)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
