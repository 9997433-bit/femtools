"""Result containers shared across the stack (fea, dynamics, correlation, io).

All containers are plain dataclasses over numpy arrays, with shape validation
at construction time.  Rows of shape/response matrices are identified by a
``dof_index``: an ordered sequence of ``(node_id, dof)`` pairs where ``dof``
is the 0-based local DOF (0..5 == UX, UY, UZ, RX, RY, RZ).  ``dof_index[i]``
labels row ``i``.

Frequency is always Hz.  Eigenvalues are ``(2*pi*f)**2`` in (rad/s)^2.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DofPair",
    "ModalResult",
    "StaticResult",
    "FRFResult",
    "ODSResult",
    "normalize_dof_index",
]

DofPair = tuple[int, int]
"""A ``(node_id, dof)`` pair; ``dof`` is 0-based (0..5 -> UX..RZ)."""


def normalize_dof_index(dof_index: Iterable[Sequence[int]] | None) -> tuple[DofPair, ...] | None:
    """Normalize any iterable of (node, dof) pairs to a tuple of int tuples.

    Accepts lists/tuples of pairs, ``(n, 2)`` integer arrays, or a
    ``dict[(node, dof), row]`` (rows must then form a permutation 0..n-1).
    """
    if dof_index is None:
        return None
    if isinstance(dof_index, dict):
        n = len(dof_index)
        out: list[DofPair | None] = [None] * n
        for pair, row in dof_index.items():
            r = int(row)
            if not 0 <= r < n or out[r] is not None:
                raise ValueError("dof_index dict values must be a permutation of 0..n-1")
            out[r] = (int(pair[0]), int(pair[1]))
        return tuple(out)  # type: ignore[arg-type]
    pairs = []
    for p in dof_index:
        node, dof = int(p[0]), int(p[1])
        if not 0 <= dof <= 5:
            raise ValueError(f"dof must be 0..5 (UX..RZ), got {dof} for node {node}")
        pairs.append((node, dof))
    return tuple(pairs)


def _check_dof_index(dof_index: tuple[DofPair, ...] | None, n_rows: int, what: str) -> None:
    if dof_index is not None and len(dof_index) != n_rows:
        raise ValueError(
            f"{what}: dof_index has {len(dof_index)} entries but data has {n_rows} rows"
        )


@dataclass
class ModalResult:
    """Real or complex normal modes.

    Attributes
    ----------
    freq_hz:
        ``(n_modes,)`` natural frequencies in Hz, ascending.
    eigenvalues:
        ``(n_modes,)`` eigenvalues ``omega^2`` in (rad/s)^2 (may be complex
        for damped/complex modes).
    modes:
        ``(n_dof, n_modes)`` mode shape matrix; column ``j`` is mode ``j``.
    generalized_mass:
        ``(n_modes,)`` modal masses (1.0 for mass-normalized modes).
    dof_index:
        Optional row labels, ``dof_index[i] == (node_id, dof)``.
    damping:
        Optional ``(n_modes,)`` modal damping ratios (zeta).
    """

    freq_hz: NDArray[np.float64]
    eigenvalues: NDArray
    modes: NDArray
    generalized_mass: NDArray[np.float64]
    dof_index: tuple[DofPair, ...] | None = None
    damping: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        self.freq_hz = np.atleast_1d(np.asarray(self.freq_hz, dtype=float))
        self.eigenvalues = np.atleast_1d(np.asarray(self.eigenvalues))
        self.modes = np.atleast_2d(np.asarray(self.modes))
        self.generalized_mass = np.atleast_1d(np.asarray(self.generalized_mass, dtype=float))
        self.dof_index = normalize_dof_index(self.dof_index)
        if self.damping is not None:
            self.damping = np.atleast_1d(np.asarray(self.damping, dtype=float))
        n = self.n_modes
        if self.eigenvalues.shape != (n,):
            raise ValueError(f"eigenvalues shape {self.eigenvalues.shape} != ({n},)")
        if self.modes.shape[1] != n:
            raise ValueError(f"modes has {self.modes.shape[1]} columns, expected {n} (one per mode)")
        if self.generalized_mass.shape != (n,):
            raise ValueError(f"generalized_mass shape {self.generalized_mass.shape} != ({n},)")
        if self.damping is not None and self.damping.shape != (n,):
            raise ValueError(f"damping shape {self.damping.shape} != ({n},)")
        _check_dof_index(self.dof_index, self.modes.shape[0], "ModalResult")

    @property
    def n_modes(self) -> int:
        return int(self.freq_hz.shape[0])

    @property
    def n_dof(self) -> int:
        return int(self.modes.shape[0])

    def mode(self, j: int) -> NDArray:
        """Shape vector of mode ``j`` (0-based)."""
        return self.modes[:, j]

    def dof_rows(self, pairs: Iterable[Sequence[int]]) -> NDArray[np.intp]:
        """Row indices of the given (node, dof) pairs (requires ``dof_index``)."""
        if self.dof_index is None:
            raise ValueError("ModalResult has no dof_index; cannot look up (node, dof) rows")
        lookup = {p: i for i, p in enumerate(self.dof_index)}
        try:
            return np.asarray([lookup[(int(p[0]), int(p[1]))] for p in pairs], dtype=np.intp)
        except KeyError as exc:
            raise KeyError(f"(node, dof) pair {exc.args[0]} not present in dof_index") from None

    def select_dofs(self, pairs: Iterable[Sequence[int]]) -> ModalResult:
        """Reduced result containing only the rows of the given (node, dof) pairs."""
        pairs_t = normalize_dof_index(pairs)
        assert pairs_t is not None
        rows = self.dof_rows(pairs_t)
        return ModalResult(
            freq_hz=self.freq_hz.copy(),
            eigenvalues=self.eigenvalues.copy(),
            modes=self.modes[rows, :].copy(),
            generalized_mass=self.generalized_mass.copy(),
            dof_index=pairs_t,
            damping=None if self.damping is None else self.damping.copy(),
        )


@dataclass
class StaticResult:
    """Static solution.

    Attributes
    ----------
    u:
        ``(n_dof,)`` displacement vector (or ``(n_dof, n_cases)``).
    dof_index:
        Optional row labels.
    reactions:
        Optional reaction force vector, same shape as ``u``.
    load_case:
        Identifier of the load case (or list of ids for multi-case ``u``).
    """

    u: NDArray[np.float64]
    dof_index: tuple[DofPair, ...] | None = None
    reactions: NDArray[np.float64] | None = None
    load_case: int | str | Sequence[int | str] = 1

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=float)
        if self.u.ndim not in (1, 2):
            raise ValueError(f"u must be 1-D or 2-D, got shape {self.u.shape}")
        self.dof_index = normalize_dof_index(self.dof_index)
        if self.reactions is not None:
            self.reactions = np.asarray(self.reactions, dtype=float)
            if self.reactions.shape != self.u.shape:
                raise ValueError(
                    f"reactions shape {self.reactions.shape} != u shape {self.u.shape}"
                )
        _check_dof_index(self.dof_index, self.u.shape[0], "StaticResult")

    @property
    def n_dof(self) -> int:
        return int(self.u.shape[0])


@dataclass
class FRFResult:
    """Frequency response functions.

    Attributes
    ----------
    freq_hz:
        ``(n_freq,)`` frequency axis in Hz.
    h_complex:
        ``(n_out, n_in, n_freq)`` complex FRF matrix.
    inputs:
        ``n_in`` reference/excitation DOFs as ``(node_id, dof)`` pairs.
    outputs:
        ``n_out`` response DOFs as ``(node_id, dof)`` pairs.
    kind:
        FRF type: ``"receptance"`` (displacement/force), ``"mobility"``
        (velocity/force) or ``"accelerance"`` (acceleration/force).
    """

    freq_hz: NDArray[np.float64]
    h_complex: NDArray[np.complex128]
    inputs: tuple[DofPair, ...]
    outputs: tuple[DofPair, ...]
    kind: str = "receptance"

    def __post_init__(self) -> None:
        self.freq_hz = np.atleast_1d(np.asarray(self.freq_hz, dtype=float))
        self.h_complex = np.asarray(self.h_complex, dtype=complex)
        inputs = normalize_dof_index(self.inputs)
        outputs = normalize_dof_index(self.outputs)
        if inputs is None or outputs is None:
            raise ValueError("FRFResult requires explicit inputs and outputs")
        self.inputs, self.outputs = inputs, outputs
        expected = (len(self.outputs), len(self.inputs), self.freq_hz.shape[0])
        if self.h_complex.shape != expected:
            raise ValueError(
                f"h_complex shape {self.h_complex.shape} != (n_out, n_in, n_freq) = {expected}"
            )

    @property
    def n_freq(self) -> int:
        return int(self.freq_hz.shape[0])

    @property
    def n_in(self) -> int:
        return len(self.inputs)

    @property
    def n_out(self) -> int:
        return len(self.outputs)

    def h(self, output: Sequence[int], input: Sequence[int]) -> NDArray[np.complex128]:
        """FRF curve ``(n_freq,)`` for one (output, input) DOF pair."""
        out_pair = (int(output[0]), int(output[1]))
        in_pair = (int(input[0]), int(input[1]))
        try:
            i_out = self.outputs.index(out_pair)
        except ValueError:
            raise KeyError(f"output DOF {out_pair} not in FRFResult.outputs") from None
        try:
            i_in = self.inputs.index(in_pair)
        except ValueError:
            raise KeyError(f"input DOF {in_pair} not in FRFResult.inputs") from None
        return self.h_complex[i_out, i_in, :]


@dataclass
class ODSResult:
    """Operating deflection shapes: complex shapes over a frequency (or order) axis.

    Attributes
    ----------
    freq_hz:
        ``(n_freq,)`` frequency lines in Hz.
    shapes:
        ``(n_dof, n_freq)`` complex deflection shape per frequency line.
    dof_index:
        Optional row labels.
    """

    freq_hz: NDArray[np.float64]
    shapes: NDArray[np.complex128]
    dof_index: tuple[DofPair, ...] | None = None
    name: str = "ODS"

    def __post_init__(self) -> None:
        self.freq_hz = np.atleast_1d(np.asarray(self.freq_hz, dtype=float))
        self.shapes = np.atleast_2d(np.asarray(self.shapes, dtype=complex))
        self.dof_index = normalize_dof_index(self.dof_index)
        if self.shapes.shape[1] != self.freq_hz.shape[0]:
            raise ValueError(
                f"shapes has {self.shapes.shape[1]} columns, expected {self.freq_hz.shape[0]}"
            )
        _check_dof_index(self.dof_index, self.shapes.shape[0], "ODSResult")

    @property
    def n_dof(self) -> int:
        return int(self.shapes.shape[0])

    @property
    def n_freq(self) -> int:
        return int(self.freq_hz.shape[0])

    def shape_at(self, freq: float) -> NDArray[np.complex128]:
        """Deflection shape at the frequency line closest to ``freq``."""
        i = int(np.argmin(np.abs(self.freq_hz - float(freq))))
        return self.shapes[:, i]
