#!/usr/bin/env python3
"""FRF estimation (H1/H2/coherence) and covariance-driven SSI on a 3-DOF chain.

Part 1 (input-output, EMA): a 3-DOF spring-mass chain driven by seeded white
noise at DOF 1 is simulated exactly (zero-order-hold modal state space), 10%
measurement noise is added to the outputs, and the H1/H2 Welch estimators are
compared against the analytic receptance. Two identities are pinned:
|H1| <= |H2| line by line, and coherence^2 = |H1| / |H2| (all three estimates
share one segmentation) -- with output-only noise H1 is the unbiased choice.

Part 2 (output-only, OMA): ambient responses from
`femtools.mpe.synthetic.synthetic_response` (unknown broadband excitation)
are fed to covariance-driven stochastic subspace identification, which must
recover the three modal frequencies, damping ratios, and shapes.

Uses the Round-4 kernels `femtools.mpe.frf_estimation` and `femtools.mpe.ssi`
(owner R4-O4, merged). `ssi_cov` sweeps model orders up to `order` and keeps
the poles that stabilise across orders, so it is called with headroom
(order=20 for 6 physical states) plus `n_modes`/`f_range` to select the three
physical poles. See docs/algorithms/mpe_rbpe.md sections 5-6 and
docs/ACCEPTANCE.md (cases 19-20).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.signal as sig

from femtools.correlation.mac import mac_matrix
from femtools.mpe.frf_estimation import coherence, estimate_h1, estimate_h2
from femtools.mpe.ssi import ssi_cov
from femtools.mpe.synthetic import synthetic_response

FS = 64.0                 # sampling rate [Hz]
DURATION = 1200.0         # H1/H2 record length [s]
NPERSEG = 8192            # Welch segment (df = FS/NPERSEG = 7.8 mHz)
NOISE = 0.10              # output noise, fraction of channel std
ZETA = 0.02
BAND = (1.0, 12.0)        # analysis band [Hz]
K_SPRING, MASS = 1000.0, 1.0
IN_DOF = 0


def chain_modes() -> tuple[np.ndarray, np.ndarray]:
    """Mass-normalized modes/frequencies of the fixed-free 3-DOF unit chain."""
    k = K_SPRING * np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    m = MASS * np.eye(3)
    lam, phi = sla.eigh(k, m)          # phi' M phi = I
    return np.sqrt(lam) / (2.0 * np.pi), phi


def simulate_forced(
    f_n: np.ndarray, phi: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Exact (ZOH) response of the proportionally damped chain to white noise
    at IN_DOF. Returns (force record, displacements (3, n), modal ZOH TFs)."""
    n = int(DURATION * FS)
    u = rng.standard_normal(n)
    q = np.empty((3, n))
    tfs = []
    for r in range(3):
        wn = 2.0 * np.pi * f_n[r]
        a = np.array([[0.0, 1.0], [-wn**2, -2.0 * ZETA * wn]])
        b = np.array([[0.0], [1.0]])
        sysd = sig.StateSpace(a, b, [[1.0, 0.0]], [[0.0]]).to_discrete(1.0 / FS)
        num, den = sig.ss2tf(sysd.A, sysd.B, sysd.C, sysd.D)
        tfs.append((num[0], den))
        q[r] = sig.lfilter(num[0], den, phi[IN_DOF, r] * u)
    return u, phi @ q, tfs


def discrete_receptance(freq_hz: np.ndarray, tfs: list, phi: np.ndarray) -> np.ndarray:
    """(3, n_f) exact FRF of the ZOH-sampled system -- what a perfect
    estimator would recover from the sampled records (the continuous
    receptance times a half-sample delay and sinc rolloff)."""
    h = np.zeros((3, freq_hz.size), dtype=complex)
    for r, (num, den) in enumerate(tfs):
        _, hr = sig.freqz(num, den, worN=freq_hz, fs=FS)
        h += np.outer(phi[:, r] * phi[IN_DOF, r], hr)
    return h


def analytic_receptance(freq_hz: np.ndarray, f_n: np.ndarray,
                        phi: np.ndarray) -> np.ndarray:
    """(3, n_f) receptance rows for a unit force at IN_DOF (modal sum)."""
    w = 2.0 * np.pi * freq_hz
    wr = 2.0 * np.pi * f_n
    h = np.zeros((3, freq_hz.size), dtype=complex)
    for r in range(3):
        h += np.outer(phi[:, r] * phi[IN_DOF, r],
                      1.0 / (wr[r]**2 - w**2 + 2j * ZETA * wr[r] * w))
    return h


def freq_and_frf(res: object) -> tuple[np.ndarray, np.ndarray]:
    """(freq_hz, H (n_out, n_in, n_f)) from an FrfEstimate-like object or a
    plain (freq_hz, H) tuple -- tolerant until the R4-O4 kernel lands."""
    if isinstance(res, tuple) and len(res) == 2:
        f, h = res
    else:
        f = getattr(res, "freq_hz", None)
        h = getattr(res, "H", None)
        if h is None:
            h = getattr(res, "frf", None)
        if f is None or h is None:
            raise TypeError(f"cannot interpret FRF estimate {type(res).__name__}")
    h = np.asarray(h)
    if h.ndim == 1:
        h = h[None, None, :]
    elif h.ndim == 2:
        h = h[:, None, :]
    return np.asarray(f, dtype=float), h


def main() -> int:
    rng = np.random.default_rng(2024)
    f_n, phi = chain_modes()
    print(f"3-DOF chain truth: f = {np.round(f_n, 4)} Hz, zeta = {ZETA:.0%} each")
    checks: list[bool] = []

    # --- part 1: H1 / H2 / coherence ----------------------------------------
    u, y, tfs = simulate_forced(f_n, phi, rng)
    y_meas = y + NOISE * np.std(y, axis=1, keepdims=True) * rng.standard_normal(y.shape)

    f_ax, h1 = freq_and_frf(estimate_h1(u, y_meas, FS, nperseg=NPERSEG))
    _, h2 = freq_and_frf(estimate_h2(u, y_meas, FS, nperseg=NPERSEG))
    _, g2 = freq_and_frf(coherence(u, y_meas, FS, nperseg=NPERSEG))
    band = (f_ax >= BAND[0]) & (f_ax <= BAND[1])
    df = float(f_ax[1] - f_ax[0])

    ratio = np.abs(h1[:, 0, band]) / np.abs(h2[:, 0, band])
    print(f"\nWelch: nperseg={NPERSEG}, df={df * 1e3:.1f} mHz, "
          f"{int(2 * DURATION * FS / NPERSEG) - 1} averaged segments")
    print(f"|H1| <= |H2| identity: max |H1|/|H2| = {np.max(ratio):.6f}")
    checks.append(bool(np.max(ratio) <= 1.0 + 1e-9))
    coh_dev = float(np.max(np.abs(np.real(g2[:, 0, band]) - ratio)))
    print(f"coherence identity gamma^2 = |H1|/|H2|: max dev = {coh_dev:.2e}")
    checks.append(coh_dev < 1e-6)
    med_coh = float(np.median(np.real(g2[:, 0, band])))
    print(f"median in-band coherence = {med_coh:.3f} "
          f"({NOISE:.0%} output noise -> dips only at antiresonances)")
    checks.append(med_coh > 0.9)

    h_zoh = discrete_receptance(f_ax, tfs, phi)
    rel = np.abs(h1[:, 0, band] - h_zoh[:, band]) / np.abs(h_zoh[:, band])
    med_err = float(np.median(rel))
    print(f"H1 vs ZOH-exact FRF of the sampled system: "
          f"median in-band rel err = {med_err:.2%}")
    checks.append(med_err < 0.10)
    h_ct = analytic_receptance(f_ax, f_n, phi)
    mag_dev = np.abs(np.abs(h1[:, 0, band]) - np.abs(h_ct[:, band])) / np.abs(h_ct[:, band])
    print(f"|H1| vs continuous-time modal receptance: median dev = "
          f"{float(np.median(mag_dev)):.2%} (ZOH sampling adds a half-sample "
          "delay and sinc rolloff, so compare magnitudes only)")
    for r, fr in enumerate(f_n):
        win = (f_ax >= 0.9 * fr) & (f_ax <= 1.1 * fr)
        f_pk = float(f_ax[win][np.argmax(np.abs(h1[2, 0, win]))])
        ok_pk = abs(f_pk - fr) <= max(2.0 * df, 0.015 * fr)
        print(f"  mode {r + 1}: |H1| peak at {f_pk:.4f} Hz (truth {fr:.4f})")
        checks.append(ok_pk)

    # --- part 2: output-only SSI-cov ----------------------------------------
    sim = synthetic_response(f_n, ZETA, mode_shapes=phi, fs=FS,
                             duration=600.0, seed=11, noise=0.02)
    ident = ssi_cov(sim.data, fs=FS, order=20, n_modes=3, f_range=BAND)
    fid = np.asarray(ident.freq_hz, dtype=float)
    zid = np.asarray(ident.damping, dtype=float)
    shapes = getattr(ident, "mode_shapes", None)

    print(f"\nSSI-cov identified {fid.size} poles: {np.round(np.sort(fid), 4)} Hz")
    for r, fr in enumerate(f_n):
        i = int(np.argmin(np.abs(fid - fr)))
        ferr = abs(fid[i] - fr) / fr
        zerr = abs(zid[i] - ZETA) / ZETA
        line = (f"  mode {r + 1}: f = {fid[i]:.4f} Hz (err {ferr:.2%}), "
                f"zeta = {zid[i]:.4f} (err {zerr:.0%})")
        checks.append(ferr < 0.02)
        checks.append(zerr < 0.5)
        if shapes is not None:
            m = float(mac_matrix(phi[:, r:r + 1], np.asarray(shapes)[:, i:i + 1])[0, 0])
            line += f", MAC = {m:.4f}"
            checks.append(m > 0.9)
        print(line)
    if shapes is None:
        print("  (no mode shapes returned -- MAC check skipped)")

    ok = all(checks)
    print("\nPASS" if ok else "\nFAIL", f"({sum(checks)}/{len(checks)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
