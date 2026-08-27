"""Shared modal-parameter-estimation infrastructure.

Poles, results containers, stabilisation diagrams and the least-squares
frequency-domain (LSFD) residue/mode-shape estimator used by every estimator in
this package.

Conventions
-----------
* A pole is stored as a **continuous-time** complex value
  :math:`\\lambda = -\\zeta\\omega_n + j\\omega_n\\sqrt{1-\\zeta^2}`.
* Natural frequency ``f_n = |lambda| / (2 pi)`` [Hz], damping ratio
  ``zeta = -Re(lambda) / |lambda|``.
* FRF arrays follow the framework convention ``(n_out, n_in, n_freq)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Pole",
    "ModalParameterResult",
    "StabilizationDiagram",
    "poles_from_roots",
    "modal_poles_from_fz",
    "select_physical_poles",
    "stabilization_diagram",
    "select_stable_modes",
    "lsfd",
    "synthesize_frf",
    "mac",
    "as_frf_array",
]


def as_frf_array(frf: Any) -> np.ndarray:
    """Coerce an FRF container into a complex ``(n_out, n_in, n_freq)`` array."""
    if not isinstance(frf, np.ndarray):
        for attr in ("H", "frf", "data", "values"):
            if hasattr(frf, attr):
                frf = getattr(frf, attr)
                break
    H = np.asarray(frf)
    if H.dtype.kind != "c":
        H = H.astype(complex)
    if H.ndim == 1:
        H = H[None, None, :]
    elif H.ndim == 2:
        H = H[:, None, :]
    elif H.ndim != 3:
        raise ValueError(f"FRF must have 1-3 dimensions, got shape {H.shape}")
    return H


def mac(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar MAC between two (complex) mode-shape vectors."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    na = float(np.real(a.conj() @ a))
    nb = float(np.real(b.conj() @ b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(abs(a.conj() @ b) ** 2 / (na * nb))


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Pole:
    """A single continuous-time modal pole."""

    value: complex
    order: int | None = None
    stable: bool = True

    @property
    def s(self) -> complex:
        return self.value

    @property
    def omega_n(self) -> float:
        return float(abs(self.value))

    @property
    def freq_hz(self) -> float:
        return float(abs(self.value) / (2.0 * math.pi))

    @property
    def damped_freq_hz(self) -> float:
        return float(abs(self.value.imag) / (2.0 * math.pi))

    @property
    def damping(self) -> float:
        m = abs(self.value)
        return float(-self.value.real / m) if m > 0 else 0.0

    @property
    def zeta(self) -> float:
        return self.damping

    def __repr__(self) -> str:  # pragma: no cover
        return f"Pole(f={self.freq_hz:.4f} Hz, zeta={100 * self.damping:.3f} %)"


@dataclass
class ModalParameterResult:
    """Identified modal model.

    Attributes
    ----------
    freq_hz, damping:
        Natural frequencies [Hz] and damping ratios [-], ascending in frequency.
    poles:
        Continuous-time complex poles (one per mode, positive imaginary part).
    mode_shapes:
        ``(n_out, n_modes)`` complex mode shapes, or ``None`` when the estimator
        was run in pole-only mode.
    residues:
        ``(n_out, n_in, n_modes)`` complex residue matrices when available.
    stabilization:
        The stabilisation diagram used to pick the physical poles.
    """

    freq_hz: np.ndarray
    damping: np.ndarray
    poles: np.ndarray
    mode_shapes: np.ndarray | None = None
    residues: np.ndarray | None = None
    participation: np.ndarray | None = None
    order: int = 0
    method: str = ""
    stabilization: StabilizationDiagram | None = None
    lower_residual: np.ndarray | None = None
    upper_residual: np.ndarray | None = None
    fit_error: float = math.nan
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.freq_hz = np.atleast_1d(np.asarray(self.freq_hz, dtype=float))
        self.damping = np.atleast_1d(np.asarray(self.damping, dtype=float))
        self.poles = np.atleast_1d(np.asarray(self.poles, dtype=complex))

    def __len__(self) -> int:
        return int(self.freq_hz.size)

    @property
    def n_modes(self) -> int:
        return int(self.freq_hz.size)

    @property
    def damping_percent(self) -> np.ndarray:
        return 100.0 * self.damping

    @property
    def modes(self) -> np.ndarray | None:
        return self.mode_shapes

    @property
    def phi(self) -> np.ndarray | None:
        return self.mode_shapes

    def pole_objects(self) -> list[Pole]:
        return [Pole(value=complex(p), order=self.order) for p in self.poles]

    def sort(self) -> ModalParameterResult:
        """Return a copy sorted by ascending frequency."""
        idx = np.argsort(self.freq_hz)
        return ModalParameterResult(
            freq_hz=self.freq_hz[idx],
            damping=self.damping[idx],
            poles=self.poles[idx],
            mode_shapes=None if self.mode_shapes is None else self.mode_shapes[:, idx],
            residues=None if self.residues is None else self.residues[:, :, idx],
            participation=None
            if self.participation is None
            else np.asarray(self.participation)[..., idx],
            order=self.order,
            method=self.method,
            stabilization=self.stabilization,
            lower_residual=self.lower_residual,
            upper_residual=self.upper_residual,
            fit_error=self.fit_error,
            extras=dict(self.extras),
        )

    def table(self) -> str:
        lines = [f"{'mode':>4s} {'f [Hz]':>12s} {'zeta [%]':>10s}"]
        for i, (f, z) in enumerate(zip(self.freq_hz, self.damping, strict=True), start=1):
            lines.append(f"{i:>4d} {f:12.5f} {100.0 * z:10.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ModalParameterResult(method={self.method!r}, n_modes={self.n_modes}, "
            f"f={np.array2string(self.freq_hz, precision=4)})"
        )


# ----------------------------------------------------------------------
def modal_poles_from_fz(
    freq_hz: ArrayLike, damping: ArrayLike
) -> np.ndarray:
    """Continuous-time poles from natural frequencies [Hz] and damping ratios."""
    f = np.atleast_1d(np.asarray(freq_hz, dtype=float))
    z = np.broadcast_to(np.atleast_1d(np.asarray(damping, dtype=float)), f.shape)
    wn = 2.0 * math.pi * f
    return -z * wn + 1j * wn * np.sqrt(np.maximum(1.0 - z**2, 0.0))


def poles_from_roots(z: ArrayLike, dt: float) -> np.ndarray:
    """Discrete-time roots ``z`` -> continuous-time poles ``ln(z)/dt``."""
    z = np.asarray(z, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log(z) / float(dt)
    return s


def select_physical_poles(
    s: ArrayLike,
    *,
    f_range: tuple[float, float] | None = None,
    max_damping: float = 0.25,
    min_damping: float = 0.0,
    min_freq: float = 1.0e-9,
) -> np.ndarray:
    """Keep stable, under-damped poles with positive imaginary part.

    Complex-conjugate duplicates are collapsed to the ``Im > 0`` representative.
    """
    s = np.asarray(s, dtype=complex)
    keep = (s.imag > 0) & (s.real < 0)
    wn = np.abs(s)
    keep &= wn > (2.0 * math.pi * min_freq)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(wn > 0, -s.real / wn, 1.0)
    keep &= (zeta <= max_damping) & (zeta >= min_damping)
    if f_range is not None:
        f = wn / (2.0 * math.pi)
        keep &= (f >= f_range[0]) & (f <= f_range[1])
    out = s[keep]
    return out[np.argsort(np.abs(out))]


# ----------------------------------------------------------------------
@dataclass
class StabilizationDiagram:
    """Stabilisation ("stability") diagram across model orders.

    Attributes
    ----------
    orders:
        Model order of every candidate pole (parallel to ``poles``).
    poles:
        Candidate continuous-time poles.
    status:
        Per-pole label: ``"n"`` new, ``"f"`` stable frequency, ``"d"`` stable
        frequency+damping, ``"v"`` stable frequency+damping+vector.
    """

    orders: np.ndarray
    poles: np.ndarray
    status: np.ndarray
    vectors: np.ndarray | None = None
    tol_freq: float = 0.01
    tol_damp: float = 0.05
    tol_mac: float = 0.02

    @property
    def freq_hz(self) -> np.ndarray:
        return np.abs(self.poles) / (2.0 * math.pi)

    @property
    def damping(self) -> np.ndarray:
        wn = np.abs(self.poles)
        return np.where(wn > 0, -self.poles.real / wn, 0.0)

    def stable_mask(self, level: str = "d") -> np.ndarray:
        levels = {"n": ("n", "f", "d", "v"), "f": ("f", "d", "v"), "d": ("d", "v"), "v": ("v",)}
        allowed = levels[level]
        return np.array([s in allowed for s in self.status], dtype=bool)

    def cluster(
        self, level: str = "d", *, tol: float = 0.01, min_count: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """Group stable poles into modes.

        Returns ``(representative_poles, counts)`` sorted by frequency; each
        representative is the median-order member of a frequency cluster.
        """
        m = self.stable_mask(level)
        p = self.poles[m]
        if p.size == 0:
            return np.zeros(0, dtype=complex), np.zeros(0, dtype=int)
        f = np.abs(p) / (2.0 * math.pi)
        order = np.argsort(f)
        p, f = p[order], f[order]
        groups: list[list[int]] = [[0]]
        for i in range(1, f.size):
            ref = f[groups[-1][-1]]
            if abs(f[i] - ref) <= tol * max(ref, 1e-12):
                groups[-1].append(i)
            else:
                groups.append([i])
        reps, counts = [], []
        for g in groups:
            if len(g) < min_count:
                continue
            sub = p[g]
            # median (in frequency) member is the most representative
            k = int(np.argsort(np.abs(sub))[len(sub) // 2])
            reps.append(sub[k])
            counts.append(len(g))
        return np.asarray(reps, dtype=complex), np.asarray(counts, dtype=int)

    def as_text(self, n_bins: int = 60) -> str:  # pragma: no cover - display aid
        f = self.freq_hz
        if f.size == 0:
            return "(empty)"
        fmin, fmax = f.min(), f.max()
        lines = []
        for o in np.unique(self.orders):
            row = ["."] * n_bins
            for fi, st in zip(f[self.orders == o], self.status[self.orders == o], strict=True):
                b = int((fi - fmin) / max(fmax - fmin, 1e-30) * (n_bins - 1))
                row[b] = st
            lines.append(f"{int(o):4d} |" + "".join(row))
        return "\n".join(lines)


def select_stable_modes(
    diagram: StabilizationDiagram,
    *,
    level: str = "d",
    cluster_tol: float = 0.01,
    min_count: int = 2,
    n_modes: int | None = None,
    relax: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the physical poles out of a stabilisation diagram.

    Clustering starts at the requested stability ``level`` and, when ``relax``
    is set, progressively falls back to weaker criteria (frequency-only, then
    "new") until at least one mode — or ``n_modes`` of them — is found.
    """
    levels = [level] + ([lv for lv in ("d", "f", "n") if lv != level] if relax else [])
    best: tuple[np.ndarray, np.ndarray] = (
        np.zeros(0, dtype=complex),
        np.zeros(0, dtype=int),
    )
    for lv in levels:
        reps, counts = diagram.cluster(lv, tol=cluster_tol, min_count=min_count)
        if reps.size > best[0].size:
            best = (reps, counts)
        if n_modes is None:
            if reps.size:
                return reps, counts
        elif reps.size >= n_modes:
            return reps, counts
    return best


def stabilization_diagram(
    pole_sets: dict[int, np.ndarray],
    *,
    vector_sets: dict[int, np.ndarray] | None = None,
    tol_freq: float = 0.01,
    tol_damp: float = 0.05,
    tol_mac: float = 0.02,
) -> StabilizationDiagram:
    """Label poles of consecutive model orders as new / stable.

    A pole of order ``n`` is compared against every pole of the previous order
    present in ``pole_sets``; the best match decides its label.
    """
    orders = sorted(pole_sets)
    all_orders: list[int] = []
    all_poles: list[complex] = []
    all_status: list[str] = []
    for k, o in enumerate(orders):
        cur = np.asarray(pole_sets[o], dtype=complex)
        prev = np.asarray(pole_sets[orders[k - 1]], dtype=complex) if k > 0 else np.zeros(
            0, dtype=complex
        )
        for i, p in enumerate(cur):
            status = "n"
            if prev.size:
                f = abs(p) / (2 * math.pi)
                fp = np.abs(prev) / (2 * math.pi)
                dfr = np.abs(fp - f) / max(f, 1e-30)
                j = int(np.argmin(dfr))
                if dfr[j] <= tol_freq:
                    status = "f"
                    zc = -p.real / abs(p) if abs(p) > 0 else 0.0
                    zp = -prev[j].real / abs(prev[j]) if abs(prev[j]) > 0 else 0.0
                    if abs(zp - zc) <= tol_damp * max(abs(zc), 1e-6):
                        status = "d"
                        if vector_sets is not None and o in vector_sets:
                            vc = np.asarray(vector_sets[o])
                            vp = np.asarray(vector_sets[orders[k - 1]])
                            if i < vc.shape[1] and j < vp.shape[1]:
                                if 1.0 - mac(vc[:, i], vp[:, j]) <= tol_mac:
                                    status = "v"
            all_orders.append(o)
            all_poles.append(p)
            all_status.append(status)
    return StabilizationDiagram(
        orders=np.asarray(all_orders, dtype=int),
        poles=np.asarray(all_poles, dtype=complex),
        status=np.asarray(all_status, dtype="<U1"),
        tol_freq=tol_freq,
        tol_damp=tol_damp,
        tol_mac=tol_mac,
    )


# ----------------------------------------------------------------------
def lsfd(
    frf: Any,
    freq_hz: ArrayLike,
    poles: ArrayLike,
    *,
    lower_residual: bool = True,
    upper_residual: bool = True,
    rcond: float | None = None,
) -> dict[str, Any]:
    """Least-squares frequency-domain estimation of residues and residual terms.

    Fits

    .. math::
        H_{oi}(\\omega) = \\sum_r \\left[\\frac{A_{oi,r}}{j\\omega - \\lambda_r}
          + \\frac{A^*_{oi,r}}{j\\omega - \\lambda_r^*}\\right]
          - \\frac{LR_{oi}}{\\omega^2} + UR_{oi}

    for known poles ``lambda_r``, linear in the unknown residues.

    Returns a dict with ``residues`` ``(n_out, n_in, n_modes)``, ``mode_shapes``
    ``(n_out, n_modes)``, ``participation`` ``(n_in, n_modes)``,
    ``lower_residual``, ``upper_residual``, ``synthesis`` and ``fit_error``.
    """
    H = as_frf_array(frf)
    n_out, n_in, n_freq = H.shape
    f = np.asarray(freq_hz, dtype=float).ravel()[:n_freq]
    lam = np.asarray(poles, dtype=complex).ravel()
    n_modes = lam.size
    w = 2.0 * math.pi * f
    jw = 1j * w

    cols: list[np.ndarray] = []
    for r in range(n_modes):
        u = 1.0 / (jw - lam[r])
        v = 1.0 / (jw - np.conj(lam[r]))
        cols.append(u + v)  # coefficient of Re(A_r)
        cols.append(1j * (u - v))  # coefficient of Im(A_r)
    n_res = 0
    if lower_residual:
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(w > 0, -1.0 / np.maximum(w**2, 1e-300), 0.0)
        cols.append(lr.astype(complex))
        n_res += 1
    if upper_residual:
        cols.append(np.ones(n_freq, dtype=complex))
        n_res += 1

    A = np.column_stack(cols)  # (n_freq, 2*n_modes + n_res)
    Areal = np.vstack([A.real, A.imag])  # real unknowns

    residues = np.zeros((n_out, n_in, n_modes), dtype=complex)
    LR = np.zeros((n_out, n_in))
    UR = np.zeros((n_out, n_in))
    synth = np.zeros_like(H)

    for o in range(n_out):
        for i in range(n_in):
            b = H[o, i, :]
            breal = np.concatenate([b.real, b.imag])
            x, *_ = np.linalg.lstsq(Areal, breal, rcond=rcond)
            for r in range(n_modes):
                residues[o, i, r] = x[2 * r] + 1j * x[2 * r + 1]
            k = 2 * n_modes
            if lower_residual:
                LR[o, i] = x[k]
                k += 1
            if upper_residual:
                UR[o, i] = x[k]
            synth[o, i, :] = A @ x.astype(complex)

    err = float(np.linalg.norm(synth - H)) / max(float(np.linalg.norm(H)), 1e-300)

    # Rank-1 factorisation of every residue matrix -> mode shape x participation
    shapes = np.zeros((n_out, n_modes), dtype=complex)
    part = np.zeros((n_in, n_modes), dtype=complex)
    for r in range(n_modes):
        Ar = residues[:, :, r]
        if n_in == 1:
            shapes[:, r] = Ar[:, 0]
            part[0, r] = 1.0
        else:
            U, S, Vh = np.linalg.svd(Ar, full_matrices=False)
            # A_r = phi_r L_r^T, and the leading SVD term is (S_0 u_0) Vh[0, :],
            # so the participation row is Vh[0, :] itself: conjugating it here
            # would return the complex conjugate of the participation factors.
            shapes[:, r] = U[:, 0] * S[0]
            part[:, r] = Vh[0, :]
        nrm = np.max(np.abs(shapes[:, r]))
        if nrm > 0:
            k = int(np.argmax(np.abs(shapes[:, r])))
            # Scaling the shape to a unit driving component moves that factor
            # into the participation, keeping the pair a factorisation of the
            # residue: A_r == outer(mode_shapes[:, r], participation[:, r]).
            scale = shapes[k, r]
            shapes[:, r] = shapes[:, r] / scale
            part[:, r] = part[:, r] * scale

    return {
        "residues": residues,
        "mode_shapes": shapes,
        "participation": part,
        "lower_residual": LR,
        "upper_residual": UR,
        "synthesis": synth,
        "fit_error": float(err),
    }


def synthesize_frf(
    poles: ArrayLike,
    residues: np.ndarray,
    freq_hz: ArrayLike,
    *,
    lower_residual: np.ndarray | None = None,
    upper_residual: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild an FRF matrix from a modal model (inverse of :func:`lsfd`)."""
    lam = np.asarray(poles, dtype=complex).ravel()
    R = np.asarray(residues, dtype=complex)
    if R.ndim == 2:
        R = R[:, None, :]
    n_out, n_in, n_modes = R.shape
    f = np.asarray(freq_hz, dtype=float).ravel()
    jw = 1j * 2.0 * math.pi * f
    H = np.zeros((n_out, n_in, f.size), dtype=complex)
    for r in range(min(n_modes, lam.size)):
        u = 1.0 / (jw - lam[r])
        v = 1.0 / (jw - np.conj(lam[r]))
        H += R[:, :, r, None] * u[None, None, :] + np.conj(R[:, :, r, None]) * v[None, None, :]
    if lower_residual is not None:
        w = 2.0 * math.pi * f
        with np.errstate(divide="ignore"):
            H += np.asarray(lower_residual)[:, :, None] * (-1.0 / np.maximum(w**2, 1e-300))
    if upper_residual is not None:
        H += np.asarray(upper_residual)[:, :, None]
    return H
