"""Frequency Based Assembly (FBA) — coupling components through measured/computed FRFs.

Simplified Lagrange-multiplier frequency based substructuring (LM-FBS). With the
uncoupled admittances stacked block-diagonally, ``Y = blkdiag(Y_A, Y_B)``, compatibility
``B u = 0`` at the interface and equilibrium ``g = -B^T lambda`` give

    Y_coupled = Y - Y B^T (B Y B^T)^-1 B Y

evaluated line by line. The interface DOFs appear twice in the coupled result (once per
component) and carry identical responses, which is a convenient consistency check.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._utils import resolve_dofs
from .frf import FRFResult, _check_freq

__all__ = ["frf_based_assembly"]


def _as_admittance(H: Any, name: str) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Extract a square ``(n, n, n_freq)`` admittance block and its frequency axis."""
    freq = None
    response = "receptance"
    if isinstance(H, FRFResult):
        freq = H.freq_hz
        response = H.response
        arr = H.H
    else:
        arr = np.asarray(H)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3-D (n, n, n_freq) admittance, got {arr.shape}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be square in its DOF dimensions (the full admittance of the "
            f"component's DOF set), got {arr.shape[0]}x{arr.shape[1]}"
        )
    return np.ascontiguousarray(arr, dtype=np.complex128), freq, response


def frf_based_assembly(
    H_a: Any,
    H_b: Any,
    coupling: Sequence[tuple[int, int]],
    freq_hz: Any = None,
    *,
    outputs: Any = None,
    inputs: Any = None,
    rcond: float = 1e-12,
) -> FRFResult:
    """Couple two components given their admittance matrices (LM-FBS).

    Parameters
    ----------
    H_a, H_b:
        Square admittance blocks of the two components, either
        :class:`~femtools.dynamics.frf.FRFResult` objects or ``(n, n, n_freq)`` arrays.
        Both must be given for the same frequency lines and the same response type
        (receptance is the natural choice).
    coupling:
        Interface DOF pairs ``(dof_in_a, dof_in_b)``; the coupled model enforces
        ``u_a[dof_a] == u_b[dof_b]`` and equal-and-opposite interface forces.
    freq_hz:
        Frequency axis; taken from the inputs when they are ``FRFResult`` objects.
    outputs, inputs:
        Selection in the *stacked* DOF space: component A occupies ``0 .. n_a-1`` and
        component B ``n_a .. n_a+n_b-1``. Defaults to the full coupled matrix.
    rcond:
        Relative cutoff for the pseudo-inverse of the interface flexibility
        ``B Y B^T``, used when the interface problem is rank deficient.

    Returns
    -------
    FRFResult
        Coupled FRFs, ``method="fba"``, with ``meta`` reporting the DOF offsets.
    """
    Ya, fa, ra = _as_admittance(H_a, "H_a")
    Yb, fb, rb = _as_admittance(H_b, "H_b")

    if Ya.shape[2] != Yb.shape[2]:
        raise ValueError(
            f"H_a has {Ya.shape[2]} frequency lines but H_b has {Yb.shape[2]}"
        )
    if ra != rb:
        raise ValueError(f"H_a is {ra} but H_b is {rb}; convert them to the same type first")

    if freq_hz is None:
        freq_hz = fa if fa is not None else fb
    if freq_hz is None:
        freq_hz = np.arange(Ya.shape[2], dtype=float)
    f = _check_freq(freq_hz)
    if f.size != Ya.shape[2]:
        raise ValueError(f"freq_hz has {f.size} lines but the admittances have {Ya.shape[2]}")
    if fa is not None and fb is not None and not np.allclose(fa, fb):
        raise ValueError("H_a and H_b are defined on different frequency axes")

    na, nb, nf = Ya.shape[0], Yb.shape[0], f.size
    n = na + nb

    pairs = [tuple(p) for p in coupling]
    if not pairs:
        raise ValueError("at least one coupling DOF pair is required")
    B = np.zeros((len(pairs), n))
    for r, (da, db) in enumerate(pairs):
        ia = int(da) + (na if int(da) < 0 else 0)
        ib = int(db) + (nb if int(db) < 0 else 0)
        if not 0 <= ia < na:
            raise IndexError(f"coupling DOF {da} out of range for component A ({na} DOFs)")
        if not 0 <= ib < nb:
            raise IndexError(f"coupling DOF {db} out of range for component B ({nb} DOFs)")
        B[r, ia] += 1.0
        B[r, na + ib] -= 1.0

    Yc = np.empty((n, n, nf), dtype=np.complex128)
    for k in range(nf):
        Y = np.zeros((n, n), dtype=complex)
        Y[:na, :na] = Ya[:, :, k]
        Y[na:, na:] = Yb[:, :, k]
        YBt = Y @ B.T
        S = B @ YBt
        try:
            Sinv_BY = np.linalg.solve(S, B @ Y)
        except np.linalg.LinAlgError:
            Sinv_BY = np.linalg.pinv(S, rcond=rcond) @ (B @ Y)
        Yc[:, :, k] = Y - YBt @ Sinv_BY

    out = resolve_dofs(outputs, n, "outputs")
    inp = resolve_dofs(inputs, n, "inputs")
    H = Yc[np.ix_(out, inp, np.arange(nf))]

    return FRFResult(
        H=H,
        freq_hz=f,
        outputs=out,
        inputs=inp,
        response=ra,
        method="fba",
        meta={
            "n_a": na,
            "n_b": nb,
            "dof_offset_b": na,
            "n_coupling": len(pairs),
            "coupling": pairs,
        },
    )
