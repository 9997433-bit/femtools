"""Automatic mode pairing between two mode sets (test vs analysis).

Two strategies are provided:

``greedy``
    Repeatedly accept the highest remaining MAC value.  This is the classical
    correlation-table behaviour: fast, order independent, and it never forces
    a poor pair when a mode has no counterpart.
``hungarian``
    Globally optimal one-to-one assignment maximizing the total MAC (linear
    sum assignment).  Preferable when several modes are close in shape, e.g.
    for repeated or nearly repeated roots, where greedy can lock in a
    locally-best but globally-inferior pair.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._assignment import linear_sum_assignment
from .mac import mac_matrix

__all__ = ["ModePair", "PairingResult", "pair_modes"]

_METHODS = {
    "greedy": "greedy",
    "hungarian": "hungarian",
    "munkres": "hungarian",
    "optimal": "hungarian",
    "lsa": "hungarian",
    "auto": "auto",
}


@dataclass(frozen=True)
class ModePair:
    """One matched mode pair."""

    index_a: int
    index_b: int
    mac: float
    freq_a: float = float("nan")
    freq_b: float = float("nan")

    @property
    def freq_error(self) -> float:
        """Relative frequency error ``(f_a - f_b) / f_b`` (NaN if unknown)."""
        if not np.isfinite(self.freq_a) or not np.isfinite(self.freq_b) or self.freq_b == 0.0:
            return float("nan")
        return (self.freq_a - self.freq_b) / self.freq_b

    @property
    def freq_error_pct(self) -> float:
        """Relative frequency error in percent."""
        return 100.0 * self.freq_error

    def __iter__(self) -> Iterator[int]:
        yield self.index_a
        yield self.index_b


@dataclass
class PairingResult:
    """Result of :func:`pair_modes`.

    Iterating or indexing the result yields :class:`ModePair` objects; the
    array views (``index_a``, ``mac_values``, ...) are convenient for
    vectorized post-processing.
    """

    pairs: list[ModePair]
    mac: NDArray[np.float64]
    method: str = "greedy"
    unpaired_a: list[int] = field(default_factory=list)
    unpaired_b: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[ModePair]:
        return iter(self.pairs)

    def __getitem__(self, item: int) -> ModePair:
        return self.pairs[item]

    @property
    def index_a(self) -> NDArray[np.intp]:
        return np.array([p.index_a for p in self.pairs], dtype=np.intp)

    @property
    def index_b(self) -> NDArray[np.intp]:
        return np.array([p.index_b for p in self.pairs], dtype=np.intp)

    @property
    def mac_values(self) -> NDArray[np.float64]:
        return np.array([p.mac for p in self.pairs], dtype=float)

    @property
    def freq_error(self) -> NDArray[np.float64]:
        return np.array([p.freq_error for p in self.pairs], dtype=float)

    def as_pairs(self) -> NDArray[np.intp]:
        """``(n_pair, 2)`` index array, accepted by :func:`~femtools.correlation.mac.comac`."""
        if not self.pairs:
            return np.empty((0, 2), dtype=np.intp)
        return np.column_stack((self.index_a, self.index_b))

    def table(self) -> str:
        """Plain-text correlation table."""
        head = (
            f"{'#':>3} {'A':>4} {'B':>4} {'MAC':>7} {'f_A [Hz]':>11} {'f_B [Hz]':>11} {'df [%]':>8}"
        )
        lines = [head, "-" * len(head)]
        for k, p in enumerate(self.pairs):
            lines.append(
                f"{k:>3} {p.index_a:>4} {p.index_b:>4} {p.mac:>7.4f} "
                f"{p.freq_a:>11.4f} {p.freq_b:>11.4f} {p.freq_error_pct:>8.2f}"
            )
        if self.unpaired_a:
            lines.append(f"unpaired A: {self.unpaired_a}")
        if self.unpaired_b:
            lines.append(f"unpaired B: {self.unpaired_b}")
        return "\n".join(lines)


def _freqs(freq: ArrayLike | None, n: int, name: str) -> NDArray[np.float64]:
    if freq is None:
        return np.full(n, np.nan)
    arr = np.asarray(freq, dtype=float).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} has {arr.size} entries but there are {n} modes")
    return arr


def _greedy(
    mac: NDArray[np.float64], allowed: NDArray[np.bool_], threshold: float
) -> list[tuple[int, int]]:
    work = np.where(allowed, mac, -np.inf)
    out: list[tuple[int, int]] = []
    for _ in range(min(work.shape)):
        flat = int(np.argmax(work))
        i, j = np.unravel_index(flat, work.shape)
        best = work[i, j]
        if not np.isfinite(best) or best < threshold:
            break
        out.append((int(i), int(j)))
        work[i, :] = -np.inf
        work[:, j] = -np.inf
    return out


def _hungarian(
    mac: NDArray[np.float64], allowed: NDArray[np.bool_], threshold: float
) -> list[tuple[int, int]]:
    if mac.size == 0:
        return []
    # Forbidden pairs get a finite penalty (rather than inf) so the assignment
    # stays feasible; they are dropped again after the solve.
    penalty = mac.max(initial=1.0) + 1.0
    cost = np.where(allowed, -mac, penalty)
    rows, cols = linear_sum_assignment(cost)
    keep = allowed[rows, cols] & (mac[rows, cols] >= threshold)
    return [(int(i), int(j)) for i, j in zip(rows[keep], cols[keep], strict=True)]


def pair_modes(
    phi_a: ArrayLike,
    phi_b: ArrayLike,
    freq_a: ArrayLike | None = None,
    freq_b: ArrayLike | None = None,
    *,
    method: str = "greedy",
    mac_threshold: float = 0.0,
    freq_tol: float | None = None,
    weights: Any = None,
    mac: ArrayLike | None = None,
) -> PairingResult:
    """Pair the modes of ``phi_a`` with those of ``phi_b`` by MAC.

    Parameters
    ----------
    phi_a, phi_b:
        Mode shape matrices ``(n_dof, n_mode)`` on a common DOF set.
    freq_a, freq_b:
        Optional natural frequencies [Hz], used for the reported frequency
        error and for the ``freq_tol`` gate.
    method:
        ``"greedy"`` (default), ``"hungarian"`` (aliases ``"optimal"``,
        ``"munkres"``, ``"lsa"``) or ``"auto"``, which uses the optimal
        assignment for small problems and greedy above 512 modes.
    mac_threshold:
        Pairs with a MAC below this value are rejected and both modes are
        reported as unpaired.
    freq_tol:
        Optional relative frequency window: candidates with
        ``|f_a - f_b| / f_b > freq_tol`` are not allowed to pair.
    weights:
        Optional weighting (e.g. mass matrix) forwarded to
        :func:`~femtools.correlation.mac.mac_matrix`.
    mac:
        Pre-computed MAC matrix ``(n_a, n_b)``; skips the recomputation.

    Returns
    -------
    PairingResult
        Pairs sorted by ``index_a``, plus the MAC matrix and the unpaired
        mode indices of both sets.
    """
    key = _METHODS.get(str(method).lower())
    if key is None:
        raise ValueError(f"unknown pairing method {method!r}; use {sorted(set(_METHODS))}")

    if mac is None:
        mac_mat = mac_matrix(phi_a, phi_b, weights=weights)
    else:
        mac_mat = np.asarray(mac, dtype=float)
        if mac_mat.ndim != 2:
            raise ValueError(f"mac must be 2-D, got shape {mac_mat.shape}")

    n_a, n_b = mac_mat.shape
    fa = _freqs(freq_a, n_a, "freq_a")
    fb = _freqs(freq_b, n_b, "freq_b")

    allowed = np.ones_like(mac_mat, dtype=bool)
    if freq_tol is not None:
        if freq_tol < 0.0:
            raise ValueError("freq_tol must be non-negative")
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(fa[:, None] - fb[None, :]) / np.where(fb == 0.0, np.nan, fb)[None, :]
        allowed &= ~(rel > freq_tol)  # NaN (unknown frequency) stays allowed

    if key == "auto":
        key = "hungarian" if max(n_a, n_b) <= 512 else "greedy"
    solver = _greedy if key == "greedy" else _hungarian
    matches = sorted(solver(mac_mat, allowed, float(mac_threshold)))

    pairs = [
        ModePair(
            index_a=i,
            index_b=j,
            mac=float(mac_mat[i, j]),
            freq_a=float(fa[i]),
            freq_b=float(fb[j]),
        )
        for i, j in matches
    ]
    used_a = {p.index_a for p in pairs}
    used_b = {p.index_b for p in pairs}
    return PairingResult(
        pairs=pairs,
        mac=mac_mat,
        method=key,
        unpaired_a=[i for i in range(n_a) if i not in used_a],
        unpaired_b=[j for j in range(n_b) if j not in used_b],
    )
