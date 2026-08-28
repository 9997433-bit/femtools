"""Mode shape expansion from the measured DOFs to the full FE DOF set.

A modal test delivers a few dozen translations; the FE model carries tens of
thousands of DOFs.  Expansion fills the gap the other way round from a
reduction: it maps the measured shape onto every DOF of the model, which is
what a full-space cross-orthogonality check, a shape-difference plot or a
force-appropriation study needs.

Two classical families are implemented, both driven by the analytical model
and therefore only as good as it is:

``SEREP`` (:func:`expand_serep`, O'Callahan/Avitabile/Riemer 1989)
    Least-squares fit of the measured shape in the FE modal basis,
    ``psi_full = Phi (Phi_m)^+ psi_m``.  It reproduces the analytical modes
    exactly and filters measurement noise onto the retained subspace, but it
    can only produce shapes that live in that subspace: nothing the mode set
    cannot represent survives.
``Guyan`` (:func:`expand_guyan`, static / dynamic expansion)
    The measured DOFs are treated as the master set of a static condensation
    and the slave DOFs follow the stiffness, ``psi_s = -K_ss^-1 K_sm psi_m``.
    Independent of any mode set — so an unexpected shape does survive — but
    it neglects the slave inertia and degrades as the frequency rises.  With
    ``mass`` and ``freq_hz`` the exact dynamic expansion
    ``-(K_ss - w^2 M_ss)^-1 (K_sm - w^2 M_sm)`` is used instead, which
    removes that error at the price of one factorization per mode.

Both return the same :class:`ExpansionResult`, whose ``modes`` are ordered on
the full DOF set of the model, so the expanded test shapes can go straight
into :func:`~femtools.correlation.mac.mac_matrix` or
:func:`~femtools.correlation.orthogonality.cross_orthogonality` against the FE
modes.  :func:`expanded_mac` is exactly that pairing in one call — expand,
then correlate on the full DOF set — with the SEREP self-consistency check
(expanding the analytical modes themselves must give them back, so the MAC
diagonal is 1) as its built-in sanity criterion.

The master DOFs may be given as row indices, as a boolean mask, as DOF keys
(``"12Z"``, ``(12, 3)``), or implicitly by handing over the DOF map of the
test: the common DOFs of the two maps are then matched automatically and the
sensor scale factors are applied like in
:func:`~femtools.correlation.dofmap.align_modes`.

The transforms themselves — the reduction side of the same operators, used to
build a test-analysis model rather than to expand a shape — belong to
``femtools.fea.reduction``; both functions here can return theirs with
``return_transform=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import ArrayLike, NDArray

from ._linalg import as_mode_matrix, mode_frequencies, row_index
from .dofmap import DOFMap, restrict
from .mac import mac_matrix

__all__ = [
    "ExpandedMACResult",
    "ExpansionResult",
    "expand_serep",
    "expand_guyan",
    "expanded_mac",
]


@dataclass
class ExpansionResult:
    """Expanded mode shapes on the full DOF set.

    ``np.asarray(result)`` gives the ``(n_full, n_mode)`` expanded matrix, so
    the result can be handed straight to the correlation routines.
    """

    modes: NDArray[Any]
    master: NDArray[np.intp]
    slave: NDArray[np.intp]
    method: str = "serep"
    residual: NDArray[np.float64] | None = None
    coefficients: NDArray[Any] | None = None
    transform: NDArray[Any] | None = None
    rank: int = 0
    freq_hz: NDArray[np.float64] | None = None
    dof_map: DOFMap | None = None

    #: Alias of :attr:`modes`.
    @property
    def phi(self) -> NDArray[Any]:
        return self.modes

    @property
    def n_master(self) -> int:
        return int(self.master.size)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.modes.shape

    @property
    def measured(self) -> NDArray[Any]:
        """The expanded shapes restricted back to the measured DOFs."""
        return self.modes[self.master]

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[Any]:
        arr = self.modes
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.array(arr, copy=True) if copy else arr

    def __getitem__(self, item: Any) -> Any:
        return self.modes[item]

    def table(self) -> str:
        """Per-mode report of the fit at the measured DOFs."""
        head = f"{'mode':>5} {'f [Hz]':>10} {'residual':>10}"
        lines = [head, "-" * len(head)]
        res = np.zeros(self.modes.shape[1]) if self.residual is None else self.residual
        for j in range(self.modes.shape[1]):
            f = np.nan if self.freq_hz is None else float(self.freq_hz[j])
            lines.append(f"{j:>5} {f:>10.4f} {float(res[j]):>10.3e}")
        lines.append(f"method: {self.method}, {self.n_master} master DOF, rank {self.rank}")
        return "\n".join(lines)


def _maybe_dofmap(source: Any, n_full: int) -> DOFMap | None:
    """DOF map carried by a modal result, or ``None`` for a bare array."""
    if source is None or isinstance(source, (np.ndarray, list, tuple)):
        return None
    try:
        dmap = DOFMap.from_mapping(source)
    except (ValueError, TypeError, KeyError):
        return None
    return dmap if len(dmap) == n_full else None


def _model_dofmap(dof_map: Any, n_full: int) -> DOFMap | None:
    """The DOF map of the model as a :class:`DOFMap`, checked against its size."""
    dmap = None if dof_map is None else DOFMap.from_mapping(dof_map)
    if dmap is not None and len(dmap) != n_full:
        raise ValueError(f"dof_map describes {len(dmap)} DOF but the model has {n_full}")
    return dmap


def _master_index(master: Any, dmap: DOFMap | None, n_full: int) -> NDArray[np.intp]:
    """Rows of the full model selected by ``master`` (indices, mask or DOF keys)."""
    arr = np.asarray(master)
    if arr.dtype == bool or np.issubdtype(arr.dtype, np.integer):
        rows_full = row_index(arr, n_full)
    else:
        if dmap is None:
            raise ValueError("resolving master DOF keys needs a dof_map")
        keys = master.keys if isinstance(master, DOFMap) else master
        rows_full = dmap.indices(keys)
    if rows_full.size == 0:
        raise ValueError("no master DOF selected")
    if np.unique(rows_full).size != rows_full.size:
        raise ValueError("the master DOFs contain duplicates")
    if rows_full.min() < 0 or rows_full.max() >= n_full:
        raise ValueError(f"master DOF out of range 0..{n_full - 1}")
    return rows_full


def _master_rows(
    master: Any,
    dof_map: Any,
    test_map: Any,
    n_full: int,
    n_test: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64] | None, DOFMap | None]:
    """Resolve the master DOFs to ``(rows_full, rows_test, scale, full_map)``.

    ``rows_full[k]`` is the row of the full model that carries the measured
    row ``rows_test[k]``; ``scale`` is the sensor factor of the test map, or
    ``None`` when no map was involved.
    """
    dmap = _model_dofmap(dof_map, n_full)

    if master is None:
        if test_map is None:
            raise ValueError("give the master DOFs, or a test_map to match against dof_map")
        if dmap is None:
            raise ValueError("matching a test_map needs the dof_map of the model")
        tmap = DOFMap.from_mapping(test_map)
        if len(tmap) != n_test:
            raise ValueError(f"test_map has {len(tmap)} DOF but phi_test has {n_test} rows")
        rows_full, rows_test = dmap.common(tmap)
        if rows_full.size == 0:
            raise ValueError("the test and model DOF maps have no DOF in common")
        return rows_full, rows_test, tmap.scale[rows_test], dmap

    rows_full = _master_index(master, dmap, n_full)
    if rows_full.size != n_test:
        raise ValueError(f"{rows_full.size} master DOF but phi_test has {n_test} rows")
    return rows_full, np.arange(n_test, dtype=np.intp), None, dmap


def _measured(
    phi_test: ArrayLike, rows_test: NDArray[np.intp], scale: NDArray[np.float64] | None
) -> NDArray[Any]:
    psi = as_mode_matrix(phi_test, "phi_test")[rows_test]
    return psi if scale is None else psi * scale[:, None]


def _residual(fit: NDArray[Any], measured: NDArray[Any]) -> NDArray[np.float64]:
    """Relative norm of the unrepresented part of each measured shape."""
    num = np.linalg.norm(fit - measured, axis=0)
    den = np.linalg.norm(measured, axis=0)
    out = np.zeros(num.shape)
    np.divide(num, den, out=out, where=den > 0.0)
    return out


def expand_serep(
    phi_test: ArrayLike,
    modes: ArrayLike,
    master: Any = None,
    *,
    dof_map: Any = None,
    test_map: Any = None,
    n_modes: int | None = None,
    freq_hz: ArrayLike | None = None,
    rcond: float = 1e-12,
    return_transform: bool = False,
) -> ExpansionResult:
    """Expand measured mode shapes with the FE modal basis (SEREP).

    The measured shape is fitted in the least-squares sense by the analytical
    modes restricted to the measured DOFs and then evaluated everywhere::

        q = pinv(Phi[master]) psi_m        (modal coordinates of the test shape)
        psi_full = Phi q                   (expanded shape)

    The fit is exact when ``psi_m`` lies in the span of ``Phi[master]`` — in
    particular SEREP reproduces the analytical modes themselves — and the
    residual reported per mode measures how much of the measured shape the
    truncated basis cannot represent.  Keep the number of basis modes well
    below the number of sensors: with as many modes as sensors the fit is
    square, the residual is zero by construction and the measurement noise is
    expanded along with the shape.

    Parameters
    ----------
    phi_test:
        Measured mode shapes, ``(n_master, n_mode)`` in the order of the
        master DOFs (or of ``test_map``).
    modes:
        Analytical basis ``(n_full, n_basis)`` over the full DOF set; a
        :class:`~femtools.fea.eigen.ModalResult` is accepted and also
        supplies its DOF map and frequencies.
    master:
        Measured DOFs as rows of ``modes``: indices, a boolean mask, DOF keys
        (``"12Z"``) resolved through ``dof_map``, or ``None`` to match
        ``test_map`` against ``dof_map``.
    dof_map:
        DOF map of the model; defaults to the one carried by ``modes``.
    test_map:
        DOF map of the measured channels, used when ``master`` is ``None``.
    n_modes:
        Keep only the first ``n_modes`` columns of the basis.
    freq_hz:
        Frequencies of the *measured* modes, kept for reporting.
    rcond:
        Relative singular value cutoff of the pseudo-inverse.
    return_transform:
        Also return the ``(n_full, n_master)`` expansion matrix ``T`` with
        ``psi_full = T psi_m``, e.g. to expand a whole FRF matrix or to build
        a test-analysis model ``T^T M T``.

    Returns
    -------
    ExpansionResult
        ``result.modes`` are the expanded shapes on the full DOF set,
        ``result.coefficients`` their modal coordinates and
        ``result.residual`` the relative misfit at the measured DOFs.
    """
    phi = as_mode_matrix(modes, "modes")
    n_full = phi.shape[0]
    if n_modes is not None:
        if not 1 <= int(n_modes) <= phi.shape[1]:
            raise ValueError(f"n_modes must be within 1..{phi.shape[1]}")
        phi = phi[:, : int(n_modes)]

    psi_raw = as_mode_matrix(phi_test, "phi_test")
    source_map = dof_map if dof_map is not None else _maybe_dofmap(modes, n_full)
    rows_full, rows_test, scale, dmap = _master_rows(
        master, source_map, test_map, n_full, psi_raw.shape[0]
    )
    psi = _measured(psi_raw, rows_test, scale)

    phi_m = phi[rows_full]
    if phi_m.shape[0] < phi_m.shape[1]:
        raise ValueError(
            f"{phi_m.shape[0]} master DOF cannot determine {phi_m.shape[1]} basis modes; "
            "measure more DOFs or lower n_modes"
        )
    pinv, rank = _pseudo_inverse(phi_m, rcond)
    coeff = pinv @ psi
    expanded = phi @ coeff

    if freq_hz is None:
        freq_hz = mode_frequencies(phi_test)
    freqs = None if freq_hz is None else np.asarray(freq_hz, dtype=float).reshape(-1)
    if freqs is not None and freqs.size != psi.shape[1]:
        freqs = None

    return ExpansionResult(
        modes=expanded,
        master=rows_full,
        slave=np.setdiff1d(np.arange(n_full, dtype=np.intp), rows_full),
        method="serep",
        residual=_residual(expanded[rows_full], psi),
        coefficients=coeff,
        transform=phi @ pinv if return_transform else None,
        rank=rank,
        freq_hz=freqs,
        dof_map=dmap,
    )


def _pseudo_inverse(a: NDArray[Any], rcond: float) -> tuple[NDArray[Any], int]:
    """Moore-Penrose inverse of ``a`` and its numerical rank, from one SVD."""
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    cut = rcond * (s[0] if s.size else 0.0)
    rank = int(np.count_nonzero(s > cut))
    inv = np.zeros_like(s)
    inv[:rank] = 1.0 / s[:rank]
    return (vt.conj().T * inv) @ u.conj().T, rank


def _as_operator(matrix: Any, n_full: int, name: str, *, like_sparse: bool) -> Any:
    """Validate a full-size operator, expanding a 1-D lumped diagonal.

    ``cross_orthogonality``, ``effective_mass`` and the rest of the package
    read a 1-D array as a lumped diagonal, so ``expand_guyan`` does too — but
    a diagonal cannot be partitioned into row/column blocks as it stands, and
    restricting it as if it could yields a silently wrong expansion.  The
    diagonal is therefore materialized here, sparse or dense to match the
    stiffness it will be combined with.
    """
    if sp.issparse(matrix):
        shape = matrix.shape
    else:
        matrix = np.asarray(matrix)
        shape = matrix.shape
        if matrix.ndim == 1:
            if matrix.size != n_full:
                raise ValueError(f"{name} has {matrix.size} entries but the model has {n_full} DOF")
            return sp.diags(matrix) if like_sparse else np.diag(matrix)
    if shape != (n_full, n_full):
        raise ValueError(f"{name} has shape {shape}, expected ({n_full}, {n_full})")
    return matrix


def _solve(a: Any, b: NDArray[Any]) -> NDArray[Any]:
    """Solve ``a x = b`` for a dense or sparse ``a``, with a lstsq fallback."""
    rhs = np.asarray(b)
    if not sp.issparse(a):
        dense = np.asarray(a)
        try:
            return np.linalg.solve(dense, rhs)
        except np.linalg.LinAlgError:  # singular slave partition
            return np.linalg.lstsq(dense, rhs, rcond=None)[0]

    matrix = sp.csc_matrix(a)
    try:
        factorized = spla.splu(matrix)
    except RuntimeError:  # singular slave partition

        def solve(x: NDArray[Any]) -> NDArray[Any]:
            return np.column_stack([spla.lsqr(matrix, x[:, j])[0] for j in range(x.shape[1])])
    else:
        solve = factorized.solve

    # A real factorization cannot take a complex right-hand side, but it can
    # take its two halves: damped mode shapes are complex, stiffness is not.
    if np.iscomplexobj(rhs) and not np.issubdtype(matrix.dtype, np.complexfloating):
        return np.asarray(solve(np.ascontiguousarray(rhs.real))) + 1j * np.asarray(
            solve(np.ascontiguousarray(rhs.imag))
        )
    return np.asarray(solve(rhs))


def expand_guyan(
    phi_test: ArrayLike,
    stiffness: Any,
    master: Any = None,
    *,
    mass: Any = None,
    freq_hz: ArrayLike | None = None,
    dof_map: Any = None,
    test_map: Any = None,
    return_transform: bool = False,
) -> ExpansionResult:
    """Expand measured mode shapes by static (Guyan) or dynamic condensation.

    The measured DOFs are the master set of a condensation of the analytical
    stiffness; the unmeasured (slave) DOFs follow from equilibrium with no
    load applied to them::

        psi_s = -K_ss^-1 K_sm psi_m                       (static, Guyan)
        psi_s = -(K_ss - w^2 M_ss)^-1 (K_sm - w^2 M_sm) psi_m   (dynamic)

    The static form is exact only at zero frequency: it discards the inertia
    of the slave DOFs, so its error grows with the ratio of the mode
    frequency to the first frequency of the structure held at the master
    DOFs.  Supplying ``mass`` and ``freq_hz`` switches to the dynamic form,
    which is exact for a mode of the analytical model at that frequency, at
    the cost of one factorization per distinct frequency.

    Unlike :func:`expand_serep` this uses no mode set, so a measured shape
    that the analytical modes cannot represent is not filtered out — at the
    price of amplifying measurement noise, which the stiffness happily
    propagates into the slave DOFs.

    Parameters
    ----------
    phi_test:
        Measured mode shapes, ``(n_master, n_mode)``.
    stiffness:
        Full stiffness matrix ``(n_full, n_full)``, dense or sparse.
    master:
        Measured DOFs: indices, a boolean mask, DOF keys, or ``None`` to
        match ``test_map`` against ``dof_map``.
    mass:
        Full mass matrix, required for the dynamic expansion.  Dense, sparse,
        or a 1-D lumped diagonal.
    freq_hz:
        Frequency of each measured mode [Hz].  With ``mass`` this selects the
        dynamic expansion; on its own it is only kept for reporting.
    dof_map, test_map:
        As in :func:`expand_serep`.
    return_transform:
        Also return the ``(n_full, n_master)`` static expansion matrix.  Only
        available for the static form, since the dynamic transform differs
        from mode to mode.

    Returns
    -------
    ExpansionResult
        ``result.modes`` reproduce the measurement exactly at the master DOFs
        and carry the condensation result at the slave DOFs.
    """
    shape = np.shape(stiffness)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"stiffness must be square, got shape {shape}")
    n_full = int(shape[0])
    if mass is not None:
        mass = _as_operator(mass, n_full, "mass", like_sparse=sp.issparse(stiffness))

    psi_raw = as_mode_matrix(phi_test, "phi_test")
    rows_full, rows_test, scale, dmap = _master_rows(
        master, dof_map, test_map, n_full, psi_raw.shape[0]
    )
    psi = _measured(psi_raw, rows_test, scale)
    slave = np.setdiff1d(np.arange(n_full, dtype=np.intp), rows_full)

    freqs = None if freq_hz is None else np.asarray(freq_hz, dtype=float).reshape(-1)
    if freqs is not None and freqs.size != psi.shape[1]:
        raise ValueError(f"freq_hz has {freqs.size} entries but there are {psi.shape[1]} modes")
    dynamic = mass is not None and freqs is not None
    if mass is not None and freqs is None:
        raise ValueError("the dynamic expansion needs freq_hz as well as mass")

    expanded = np.zeros((n_full, psi.shape[1]), dtype=psi.dtype)
    expanded[rows_full] = psi
    transform = None
    if slave.size:
        k_ss = restrict(stiffness, slave)
        k_sm = restrict(stiffness, slave, rows_full)
        if dynamic:
            m_ss = restrict(mass, slave)
            m_sm = restrict(mass, slave, rows_full)
            omega_sq = (2.0 * np.pi * np.asarray(freqs, dtype=float)) ** 2
            for j, w2 in enumerate(omega_sq.tolist()):
                rhs = -(np.asarray(k_sm @ psi[:, j]) - w2 * np.asarray(m_sm @ psi[:, j]))
                expanded[slave, j] = _solve(k_ss - w2 * m_ss, rhs.reshape(-1, 1)).reshape(-1)
        elif return_transform:
            t_gs = _solve(k_ss, -(k_sm.toarray() if sp.issparse(k_sm) else np.asarray(k_sm)))
            expanded[slave] = t_gs @ psi
            transform = np.zeros((n_full, rows_full.size))
            transform[rows_full] = np.eye(rows_full.size)
            transform[slave] = t_gs
        else:
            expanded[slave] = _solve(k_ss, -np.asarray(k_sm @ psi))

    return ExpansionResult(
        modes=expanded,
        master=rows_full,
        slave=slave,
        method="dynamic" if dynamic else "guyan",
        residual=np.zeros(psi.shape[1]),
        transform=transform,
        rank=int(rows_full.size),
        freq_hz=freqs,
        dof_map=dmap,
    )


@dataclass
class ExpandedMACResult:
    """MAC of expanded shapes against the full-DOF reference modes.

    ``np.asarray(result)`` gives the ``(n_test, n_reference)`` MAC matrix, and
    the result also unpacks as the documented pair ``mac, expansion =
    expanded_mac(...)`` for callers that only want the two arrays.
    """

    mac: NDArray[np.float64]
    expansion: ExpansionResult
    reference: NDArray[Any]
    freq_test: NDArray[np.float64] | None = None
    freq_reference: NDArray[np.float64] | None = None

    #: Alias of :attr:`mac`.
    @property
    def values(self) -> NDArray[np.float64]:
        return self.mac

    @property
    def modes(self) -> NDArray[Any]:
        """The expanded shapes on the full DOF set."""
        return self.expansion.modes

    @property
    def master(self) -> NDArray[np.intp]:
        return self.expansion.master

    @property
    def residual(self) -> NDArray[np.float64] | None:
        """Relative misfit of the SEREP fit at the master DOFs, per mode."""
        return self.expansion.residual

    @property
    def method(self) -> str:
        return self.expansion.method

    @property
    def n_master(self) -> int:
        return self.expansion.n_master

    @property
    def shape(self) -> tuple[int, ...]:
        return self.mac.shape

    @property
    def diagonal(self) -> NDArray[np.float64]:
        """``mac[k, k]``, the correlation of each expanded mode with its own."""
        n = min(self.mac.shape) if self.mac.size else 0
        idx = np.arange(n)
        return np.asarray(self.mac[idx, idx], dtype=float)

    @property
    def min_diagonal(self) -> float:
        diag = self.diagonal
        return float(diag.min()) if diag.size else float("nan")

    @property
    def diagonal_error(self) -> float:
        """``max |1 - mac[k, k]|``: the reference-independent SEREP self-check.

        Unlike :attr:`identity_error` this ignores the off-diagonal, which is
        the auto-MAC of the reference set and therefore not zero unless that
        set is orthogonal under the MAC weighting.
        """
        diag = self.diagonal
        return float(np.abs(diag - 1.0).max()) if diag.size else 0.0

    @property
    def max_off_diagonal(self) -> float:
        """Largest MAC between two *different* modes (0 for a single mode)."""
        if self.mac.size == 0:
            return 0.0
        mask = np.ones(self.mac.shape, dtype=bool)
        n = min(self.mac.shape)
        idx = np.arange(n)
        mask[idx, idx] = False
        return float(self.mac[mask].max()) if mask.any() else 0.0

    @property
    def identity_error(self) -> float:
        """``max |mac - I|``, the worst of :attr:`diagonal_error` and the off-diagonal.

        0 means the expanded shapes reproduce the reference modes *and* those
        modes are mutually uncorrelated under the criterion used — true for an
        orthonormal basis, and for mass-normalized FE modes with
        ``weights=mass``, but not for the plain MAC of a general FE mode set,
        whose auto-MAC has non-zero off-diagonal terms of its own.
        """
        if self.mac.size == 0:
            return 0.0
        eye = np.zeros(self.mac.shape)
        n = min(self.mac.shape)
        idx = np.arange(n)
        eye[idx, idx] = 1.0
        return float(np.abs(self.mac - eye).max())

    def is_identity(self, tol: float = 1e-10) -> bool:
        """Whether the whole table is the identity to ``tol``.

        Use :attr:`diagonal_error` instead when the reference modes are not
        orthogonal under the criterion, which is the usual case for an
        unweighted MAC of FE modes.
        """
        return self.identity_error <= float(tol)

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        arr = self.mac
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return np.array(arr, copy=True) if copy else arr

    def __iter__(self) -> Any:
        """Unpack as ``(mac, expansion)``."""
        return iter((self.mac, self.expansion))

    def __getitem__(self, item: Any) -> Any:
        return self.mac[item]

    def table(self) -> str:
        """Per-mode report: paired MAC, worst off-diagonal and SEREP residual."""
        head = f"{'mode':>5} {'f [Hz]':>10} {'MAC':>8} {'off-diag':>9} {'residual':>10}"
        lines = [head, "-" * len(head)]
        diag = self.diagonal
        res = self.residual
        freqs = self.freq_test
        for k in range(self.mac.shape[0]):
            f = np.nan if freqs is None or k >= len(freqs) else float(freqs[k])
            paired = float(diag[k]) if k < diag.size else np.nan
            row = np.asarray(self.mac[k], dtype=float)
            others = np.delete(row, k) if k < row.size else row
            off = float(others.max()) if others.size else 0.0
            r = np.nan if res is None or k >= len(res) else float(res[k])
            lines.append(f"{k:>5} {f:>10.4f} {paired:>8.4f} {off:>9.2e} {r:>10.3e}")
        lines.append(
            f"method: {self.method}, {self.n_master} master DOF, "
            f"max |1 - MAC| = {self.diagonal_error:.3e}, "
            f"max |MAC - I| = {self.identity_error:.3e}"
        )
        return "\n".join(lines)


def expanded_mac(
    phi_test: ArrayLike,
    modes: ArrayLike,
    master: Any = None,
    *,
    reference: ArrayLike | None = None,
    dof_map: Any = None,
    test_map: Any = None,
    n_modes: int | None = None,
    freq_hz: ArrayLike | None = None,
    rcond: float = 1e-12,
    weights: Any = None,
    return_transform: bool = False,
) -> ExpandedMACResult:
    """SEREP-expand the measured shapes and MAC them against the FE modes.

    A MAC computed on the measured DOFs alone answers a smaller question than
    it appears to: with a few dozen sensors two genuinely different shapes can
    look alike (spatial aliasing), and the auto-MAC of the FE modes restricted
    to those sensors is the usual warning for it.  Expanding the test shapes
    onto the full DOF set with :func:`expand_serep` first and correlating
    there — O'Callahan, Avitabile & Riemer, *System Equivalent Reduction
    Expansion Process*, Proc. 7th IMAC, 1989, with Allemang's MAC — moves the
    comparison back onto the complete model::

        psi_full = Phi pinv(Phi[master]) psi_m       (SEREP expansion)
        mac[i, j] = MAC(psi_full[:, i], Phi_ref[:, j])

    The check that makes the composition trustworthy is its own fixed point:
    feed the analytical modes *restricted to the master DOFs* back in and
    SEREP must return them unchanged, so every paired MAC comes back as 1 to
    round-off (:attr:`ExpandedMACResult.diagonal_error`) and the whole table
    collapses onto the auto-MAC of the reference set — the identity for an
    orthonormal basis, and for mass-normalized FE modes with ``weights=mass``,
    where :attr:`ExpandedMACResult.identity_error` is the number to read.  A
    departure from it is not a correlation result but a defect of the master
    set: too few sensors, a rank-deficient ``Phi[master]``, or a basis
    truncated below the modes being fitted.  Note what the fixed point does
    *not* prove: SEREP can only produce shapes inside the span of the basis,
    so the expanded test shapes are filtered onto it and their MAC is
    optimistic by construction; read :attr:`ExpandedMACResult.residual`, the
    part of the measurement the basis could not represent, alongside the
    table.

    Parameters
    ----------
    phi_test:
        Measured mode shapes, ``(n_master, n_mode)`` in the order of the
        master DOFs (or of ``test_map``).  A full ``(n_full, n_mode)`` set is
        accepted as well and is restricted to ``master`` first, which is what
        the self-check above needs: ``expanded_mac(fe_modes, fe_modes,
        master)``.
    modes:
        Analytical basis ``(n_full, n_basis)`` over the full DOF set, as in
        :func:`expand_serep`; a
        :class:`~femtools.fea.eigen.ModalResult` also supplies its DOF map
        and frequencies.
    master, dof_map, test_map, n_modes, rcond, return_transform:
        As in :func:`expand_serep`.
    reference:
        Full-DOF mode set the expanded shapes are correlated against.
        Defaults to the expansion basis itself (truncated by ``n_modes``),
        which is the SEREP self-consistency check.
    freq_hz:
        Frequencies of the *measured* modes, kept for the report.
    weights:
        MAC weighting as in :func:`~femtools.correlation.mac.mac_matrix`; the
        full mass matrix gives the (squared) normalized cross-orthogonality
        on the expanded DOF set instead of the plain MAC.

    Returns
    -------
    ExpandedMACResult
        ``result.mac`` is the ``(n_mode, n_reference)`` table,
        ``result.expansion`` the underlying
        :class:`ExpansionResult`; the object also unpacks as
        ``mac, expansion = expanded_mac(...)``.

    See Also
    --------
    expand_serep : the expansion on its own.
    femtools.correlation.mac.mac_matrix : the criterion on its own.
    femtools.correlation.orthogonality.cross_orthogonality : mass-weighted
        alternative, unsquared and signed.
    """
    basis = as_mode_matrix(modes, "modes")
    n_full = basis.shape[0]
    if n_modes is not None:
        if not 1 <= int(n_modes) <= basis.shape[1]:
            raise ValueError(f"n_modes must be within 1..{basis.shape[1]}")
        basis = basis[:, : int(n_modes)]

    if freq_hz is None:
        freq_hz = mode_frequencies(phi_test)
    psi = as_mode_matrix(phi_test, "phi_test")

    # A full-size test matrix is the self-check: restrict it to the masters
    # so the caller does not have to spell out `fe_modes[master]`.
    if master is not None and test_map is None and psi.shape[0] == n_full:
        source_map = dof_map if dof_map is not None else _maybe_dofmap(modes, n_full)
        rows = _master_index(master, _model_dofmap(source_map, n_full), n_full)
        if rows.size != n_full:
            psi = psi[rows]

    expansion = expand_serep(
        psi,
        modes,
        master,
        dof_map=dof_map,
        test_map=test_map,
        n_modes=n_modes,
        freq_hz=freq_hz,
        rcond=rcond,
        return_transform=return_transform,
    )

    if reference is None:
        ref = basis
        ref_freq = mode_frequencies(modes)
    else:
        ref = as_mode_matrix(reference, "reference")
        ref_freq = mode_frequencies(reference)
    if ref.shape[0] != n_full:
        raise ValueError(f"reference has {ref.shape[0]} DOF but the model has {n_full}")
    if ref_freq is not None:
        ref_freq = np.asarray(ref_freq, dtype=float).reshape(-1)[: ref.shape[1]]
        if ref_freq.size != ref.shape[1]:
            ref_freq = None

    return ExpandedMACResult(
        mac=mac_matrix(expansion.modes, ref, weights=weights),
        expansion=expansion,
        reference=ref,
        freq_test=expansion.freq_hz,
        freq_reference=ref_freq,
    )
