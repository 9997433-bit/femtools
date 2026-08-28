#!/usr/bin/env python3
"""ERA on a synthetic 2-DOF chain: poles and shapes from impulse responses.

The Eigensystem Realization Algorithm (Juang, J.N., Pappa, R.S., "An
Eigensystem Realization Algorithm for Modal Parameter Identification and
Model Reduction", J. Guidance, Control, and Dynamics 8(5), 1985, pp. 620-627)
builds a minimal discrete state-space realization (A, B, C) from Markov
parameters — sampled impulse-response functions — via the SVD of their block
Hankel matrix; poles come from eig(A), shapes from C psi.

The test bench is a 2-DOF spring-mass chain whose modal truth is closed form.
Part 1 feeds ERA the exactly sampled analytic IRF: the sampled response of a
linear system *is* a discrete LTI system with poles z = exp(lambda dt), so a
correct single-order realization (`stabilization=False`, the documented path
for clean pulse responses) recovers frequencies, damping and shapes to
round-off — the acceptance gate leaves one spectral line df = 1/T of headroom
and requires MAC > 0.99 (ACCEPTANCE Round-10 row 31). Part 2 reaches ERA the
way a test lab would: a synthesized FRF handed over directly, so the kernel
runs `irf_from_frf` internally and removes the exponential-window bias from
the realization analytically (the damping stays checkable), with the
stabilization sweep left on.

Skips whole with a message while the Round-10 kernel is not on this tree.
See docs/algorithms/mpe_rbpe.md (section 8) and docs/ACCEPTANCE.md
(Round-10 status).
"""

from __future__ import annotations

import numpy as np

try:
    from femtools.correlation.mac import mac_matrix
    from femtools.mpe.era import era
    from femtools.mpe.synthetic import synthetic_frf
except ImportError as exc:  # ERA kernel not on this tree yet
    print(f"SKIP: ERA kernels not importable on this tree ({exc})")
    raise SystemExit(0) from None

K_SPRING = 1000.0        # N/m, ground-m1 and m1-m2
MASS = 1.0               # kg each
ZETA = 0.02              # modal damping, both modes
FS = 64.0                # IRF sampling rate [Hz]
DURATION = 8.0           # IRF length [s] -> df = 1/8 = 0.125 Hz
ORDER = 8                # state-order headroom over the 4 physical states
IN_DOF = 0               # impulse at mass 1
DF_FRF = 0.0625          # FRF line spacing [Hz] for part 2


def chain_modes() -> tuple[np.ndarray, np.ndarray]:
    """Exact frequencies [Hz] and mass-normalized shapes of the 2-DOF chain."""
    k = K_SPRING * np.array([[2.0, -1.0], [-1.0, 1.0]])
    lam, phi = np.linalg.eigh(k / MASS)
    return np.sqrt(lam) / (2.0 * np.pi), phi / np.sqrt(MASS)


def analytic_irf(f_n: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """(2, 1, n_t) receptance IRF for a unit impulse at IN_DOF (modal sum)."""
    t = np.arange(int(DURATION * FS)) / FS
    wr = 2.0 * np.pi * f_n
    wd = wr * np.sqrt(1.0 - ZETA**2)
    h = np.zeros((2, 1, t.size))
    for r in range(2):
        env = np.exp(-ZETA * wr[r] * t) * np.sin(wd[r] * t) / wd[r]
        h[:, 0, :] += np.outer(phi[:, r] * phi[IN_DOF, r], env)
    return h


def gate(ident: object, f_n: np.ndarray, phi: np.ndarray, df: float,
         label: str) -> list[bool]:
    """Match identified poles to the truth and apply the Round-10 gates."""
    fid = np.asarray(ident.freq_hz, dtype=float)
    zid = np.asarray(ident.damping, dtype=float)
    shapes = getattr(ident, "mode_shapes", None)
    checks: list[bool] = []
    print(f"{label}: identified {fid.size} poles at {np.round(np.sort(fid), 4)} Hz")
    for r, fr in enumerate(f_n):
        i = int(np.argmin(np.abs(fid - fr)))
        ferr = abs(fid[i] - fr)
        zerr = abs(zid[i] - ZETA) / ZETA
        ok_f = ferr <= df
        line = (f"  [{'ok' if ok_f else 'FAIL'}] mode {r + 1}: "
                f"f = {fid[i]:.4f} Hz (truth {fr:.4f}, |df| = {ferr:.2e} "
                f"<= one line {df:.4f} Hz), zeta = {zid[i]:.5f} (err {zerr:.2%})")
        checks.append(ok_f)
        checks.append(zerr < 0.5)
        if shapes is not None:
            m = float(mac_matrix(phi[:, r:r + 1],
                                 np.asarray(shapes)[:, i:i + 1])[0, 0])
            line += f", MAC = {m:.6f}"
            checks.append(m > 0.99)
        print(line)
    if shapes is None:
        print("  [FAIL] no mode shapes returned -- the Round-10 gate needs MAC > 0.99")
        checks.append(False)
    return checks


def main() -> int:
    f_n, phi = chain_modes()
    print(f"2-DOF chain truth: f = {np.round(f_n, 4)} Hz, zeta = {ZETA:.0%} each")
    checks: list[bool] = []

    # -- part 1: single-order ERA on the exactly sampled analytic IRF ---------
    h = analytic_irf(f_n, phi)
    ident = era(h, fs=FS, order=ORDER, n_modes=2, stabilization=False)
    checks.extend(gate(ident, f_n, phi, 1.0 / DURATION,
                       "\nERA on the analytic IRF (single order)"))

    # -- part 2: the lab route -- FRF in, IRF + window handled by the kernel --
    freq = np.arange(257) * DF_FRF                 # DC-aligned grid to 16 Hz
    syn = synthetic_frf(freq, f_n, ZETA, mode_shapes=phi, n_out=2, n_in=1,
                        input_dofs=[IN_DOF], kind="receptance", noise=0.0)
    ident2 = era(syn.frf, freq_hz=freq, order=ORDER, n_modes=2)
    checks.extend(gate(ident2, f_n, phi, DF_FRF,
                       "\nERA on a synthetic FRF (stabilization sweep)"))

    print("PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
