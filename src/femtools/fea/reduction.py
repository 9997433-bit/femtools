"""Model order reduction onto a retained (master) set of degrees of freedom.

Three classical reduction bases are provided, all of them producing an
``n_dof x n_master`` transformation ``T`` with ``u = T @ u_master``:

``guyan``
    Static (Guyan-Irons) condensation.  Exact at zero frequency, and the
    reduced stiffness ``T^T K T`` is *identically* the Schur complement
    ``K_mm - K_ms K_ss^-1 K_sm``; the mass is only approximated, which is what
    makes the reduced frequencies too high.
``irs``
    O'Callahan's Improved Reduced System.  One inertia correction term is added
    to the static basis, which removes most of the Guyan frequency bias without
    needing an eigensolution of the parent model.
``serep``
    System Equivalent Reduction Expansion Process.  Built from a set of parent
    mode shapes instead of from ``K``: within the span of those modes the
    reduction is *exact*, so the retained frequencies are reproduced to machine
    precision and the slave motion is recovered from the master partition
    alone.

The three share :class:`ReductionResult`, which also unpacks as ``T, K_red``
for the terse call style ``T, Krr = guyan(K, master)``.

References
----------
Guyan, R. J. (1965), "Reduction of stiffness and mass matrices", AIAA J. 3(2).
O'Callahan, J. C. (1989), "A procedure for an improved reduced system (IRS)
model", 7th IMAC.
O'Callahan, Avitabile & Riemer (1989), "System equivalent reduction expansion
process (SEREP)", 7th IMAC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["ReductionResult", "guyan", "irs", "serep"]


@dataclass
class ReductionResult:
    """A reduction basis and the matrices it produces.

    Attributes
    ----------
    T:
        ``(n_dof, n_master)`` transformation with ``u = T @ u_master``.  Its
        master rows are the identity, so ``T[master] == I`` for the physical
        bases (``guyan``, ``irs``) and to round-off for ``serep``.
    master:
        Retained DOF indices into the parent numbering, in the order the caller
        supplied them; column ``j`` of ``T`` belongs to ``master[j]``.
    slave:
        The condensed complement of ``master``, ascending.
    K_red, M_red:
        ``T^T K T`` and ``T^T M T`` when the corresponding parent matrix was
        available, otherwise ``None``.
    method:
        ``"guyan"``, ``"irs"`` or ``"serep"``.
    """

    T: np.ndarray
    master: np.ndarray
    slave: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    K_red: np.ndarray | None = None
    M_red: np.ndarray | None = None
    method: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- shape ---------------------------------------------------------
    @property
    def n_dof(self) -> int:
        """Size of the parent model."""
        return int(self.T.shape[0])

    @property
    def n_master(self) -> int:
        """Size of the reduced model."""
        return int(self.T.shape[1])

    @property
    def n_slave(self) -> int:
        return int(self.slave.size)

    # -- aliases used across the rest of the package --------------------
    @property
    def K(self) -> np.ndarray | None:
        return self.K_red

    @property
    def M(self) -> np.ndarray | None:
        return self.M_red

    @property
    def master_dofs(self) -> np.ndarray:
        return self.master

    @property
    def slave_dofs(self) -> np.ndarray:
        return self.slave

    def __iter__(self):
        """Allow ``T, Krr = guyan(K, master)``."""
        return iter((self.T, self.K_red))

    def __getitem__(self, index: int):
        return (self.T, self.K_red)[index]

    # -- use -----------------------------------------------------------
    def expand(self, u_master: np.ndarray) -> np.ndarray:
        """Recover the full DOF vector (or set of columns) from master data."""
        arr = np.asarray(u_master)
        if arr.shape[0] != self.n_master:
            raise ValueError(
                f"expected {self.n_master} master rows, got {arr.shape[0]}"
            )
        return self.T @ arr

    def restrict(self, u_full: np.ndarray) -> np.ndarray:
        """Pick the master partition out of a full DOF vector or matrix."""
        arr = np.asarray(u_full)
        return arr[self.master] if arr.ndim == 1 else arr[self.master, :]

    def reduced_modes(self, n_modes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Normal modes of the reduced pencil, expanded to the full DOF space.

        Returns ``(freq_hz, phi)`` with ``phi`` of shape ``(n_dof, n_kept)``
        and mass-normalised against the *reduced* mass.

        The reduced mass can be singular: a SEREP basis built from fewer modes
        than there are sensors spans only an ``n_modes``-dimensional subspace,
        so ``M_red`` has exactly ``n_master - n_modes`` zero eigenvalues and a
        plain ``eigh(K_red, M_red)`` fails outright.  The pencil is therefore
        solved inside the subspace ``M_red`` actually spans, which drops those
        null directions instead of turning them into infinite frequencies.
        """
        if self.K_red is None or self.M_red is None:
            raise ValueError("reduced_modes needs both K_red and M_red")
        w, V = np.linalg.eigh(_symmetrize(self.M_red))
        keep = w > max(float(w.max()), 0.0) * self.n_master * 1.0e-14
        if not keep.any():
            return np.zeros(0), np.zeros((self.n_dof, 0))
        B = V[:, keep] / np.sqrt(w[keep])
        lam, Q = np.linalg.eigh(_symmetrize(B.T @ self.K_red @ B))
        if n_modes is not None:
            lam, Q = lam[: int(n_modes)], Q[:, : int(n_modes)]
        return np.sqrt(np.clip(lam, 0.0, None)) / (2.0 * np.pi), self.T @ (B @ Q)

    def expansion_error(self, u_full: np.ndarray) -> float:
        """Relative error of ``T @ u_full[master]`` against ``u_full``.

        The single number that says whether the retained set is rich enough:
        it is zero for SEREP whenever ``u_full`` lies in the span of the modes
        the basis was built from, and grows with the inertia content the static
        bases cannot represent.
        """
        arr = np.asarray(u_full, dtype=float)
        recovered = self.expand(self.restrict(arr))
        denominator = np.linalg.norm(arr)
        if denominator == 0.0:
            return 0.0
        return float(np.linalg.norm(recovered - arr) / denominator)

    def __repr__(self) -> str:  # pragma: no cover - reporting helper
        return (
            f"ReductionResult(method={self.method!r}, n_dof={self.n_dof}, "
            f"n_master={self.n_master})"
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _as_dense(A: Any) -> np.ndarray:  # noqa: N803
    """Dense float view of a dense or sparse matrix."""
    if sp.issparse(A):
        return np.asarray(A.toarray(), dtype=float)
    return np.asarray(A, dtype=float)


def _resolve_master(master: Any, n_dof: int) -> np.ndarray:
    """Master indices as an int array, order preserved, duplicates dropped.

    A boolean mask is accepted as well.  The order matters: it fixes the
    meaning of the rows and columns of ``K_red``, so it is the caller's sensor
    ordering that survives rather than an internally sorted one.
    """
    idx = np.asarray(master)
    if idx.dtype == bool:
        if idx.size != n_dof:
            raise ValueError(f"boolean master mask has size {idx.size}, expected {n_dof}")
        idx = np.flatnonzero(idx)
    idx = np.asarray(idx, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError("at least one master DOF is required")
    negative = idx < 0
    if negative.any():
        idx = np.where(negative, idx + n_dof, idx)
    if idx.min() < 0 or idx.max() >= n_dof:
        raise ValueError(f"master DOF index out of range for a model with {n_dof} DOFs")
    _, first = np.unique(idx, return_index=True)
    return idx[np.sort(first)]


def _slave_of(master: np.ndarray, n_dof: int) -> np.ndarray:
    mask = np.ones(n_dof, dtype=bool)
    mask[master] = False
    return np.flatnonzero(mask)


def _block(A: Any, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:  # noqa: N803
    """Dense ``A[rows, cols]`` sub-block for dense or sparse ``A``."""
    if sp.issparse(A):
        return np.asarray(A.tocsr()[rows, :][:, cols].toarray(), dtype=float)
    return np.asarray(A, dtype=float)[np.ix_(rows, cols)]


def _solver(Kss: Any):  # noqa: N803
    """Return ``x -> Kss^-1 x`` for the (square, non-singular) slave block.

    A sparse block is factorised with SuperLU, a dense one with a Cholesky
    factorisation falling back to LU: the slave stiffness of a well-posed model
    is symmetric positive definite, but a model with rigid body freedom in the
    slave set is not, and that case should still reduce rather than raise.
    """
    if sp.issparse(Kss):
        lu = spla.splu(sp.csc_matrix(Kss))
        return lambda rhs: np.asarray(lu.solve(np.asarray(rhs, dtype=float)))
    dense = np.asarray(Kss, dtype=float)
    symmetric = 0.5 * (dense + dense.T)
    try:
        cho = sla.cho_factor(symmetric, lower=True)
    except (np.linalg.LinAlgError, ValueError):
        lu_piv = sla.lu_factor(dense)
        return lambda rhs: sla.lu_solve(lu_piv, rhs)
    return lambda rhs: sla.cho_solve(cho, rhs)


def _symmetrize(A: np.ndarray) -> np.ndarray:  # noqa: N803
    return 0.5 * (A + A.T)


def _solve_spd(A: np.ndarray, B: np.ndarray) -> np.ndarray:  # noqa: N803
    """``A^-1 B`` for a symmetric ``A``, tolerating a singular reduced mass."""
    try:
        return np.asarray(sla.solve(A, B, assume_a="pos"))
    except (np.linalg.LinAlgError, ValueError):
        return np.asarray(np.linalg.lstsq(A, B, rcond=None)[0])


def _check_square_pair(K: Any, M: Any) -> int:  # noqa: N803
    n = int(K.shape[0])
    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be square")
    if M is not None and tuple(M.shape) != (n, n):
        raise ValueError("K and M must have the same shape")
    return n


def _static_basis(K: Any, master: np.ndarray, slave: np.ndarray) -> np.ndarray:  # noqa: N803
    """``T`` of static condensation: identity on masters, ``-Kss^-1 Ksm`` below."""
    n_dof = int(K.shape[0])
    T = np.zeros((n_dof, master.size))
    T[master, np.arange(master.size)] = 1.0
    if slave.size:
        solve = _solver(_block(K, slave, slave))
        T[slave, :] = -solve(_block(K, slave, master))
    return T


# ----------------------------------------------------------------------
# reduction bases
# ----------------------------------------------------------------------


def guyan(
    K: Any,  # noqa: N803
    master: Any,
    M: Any = None,  # noqa: N803
) -> ReductionResult:
    """Static (Guyan-Irons) condensation onto ``master``.

    The slave DOFs are assumed to carry no inertia, so they follow the masters
    through the static relation ``u_s = -K_ss^-1 K_sm u_m``.  The resulting
    ``T^T K T`` is the exact Schur complement of the parent stiffness: Guyan
    costs nothing in static accuracy and everything in dynamic accuracy, and
    the reduced frequencies are always **upper** bounds because discarding the
    slave inertia can only stiffen the model.

    Parameters
    ----------
    K:
        Parent stiffness, dense or sparse, ``(n, n)``.
    master:
        Retained DOF indices (or a boolean mask of length ``n``).  The order is
        preserved and fixes the row/column order of ``K_red``.
    M:
        Optional parent mass; when given, ``M_red = T^T M T`` is filled in.

    Returns
    -------
    ReductionResult
        Also unpacks as ``T, K_red``.
    """
    n_dof = _check_square_pair(K, M)
    m = _resolve_master(master, n_dof)
    s = _slave_of(m, n_dof)

    T = _static_basis(K, m, s)
    K_red = _symmetrize(T.T @ _as_dense(K) @ T)
    M_red = None if M is None else _symmetrize(T.T @ _as_dense(M) @ T)

    return ReductionResult(
        T=T,
        master=m,
        slave=s,
        K_red=K_red,
        M_red=M_red,
        method="guyan",
        meta={"n_dof": n_dof, "sparse": bool(sp.issparse(K))},
    )


def irs(
    K: Any,  # noqa: N803
    M: Any,  # noqa: N803
    master: Any,
) -> ReductionResult:
    """O'Callahan's Improved Reduced System (IRS) reduction.

    The static basis is corrected with one term of the inertia the Guyan
    assumption throws away::

        T_irs = T_g + S M T_g M_g^-1 K_g

    where ``T_g`` is the Guyan basis, ``M_g = T_g^T M T_g``,
    ``K_g = T_g^T K T_g`` and ``S`` is zero except for ``K_ss^-1`` on the slave
    block.  The correction is a *pseudo-static* response to the mass forces the
    retained motion generates, so it needs no eigensolution of the parent
    model, yet it typically removes an order of magnitude of the Guyan
    frequency bias.  Unlike Guyan the result is no longer a bound: IRS
    frequencies may sit on either side of the exact ones.

    Parameters
    ----------
    K, M:
        Parent stiffness and mass, dense or sparse, ``(n, n)``.
    master:
        Retained DOF indices, or a boolean mask of length ``n``.

    Returns
    -------
    ReductionResult
        With ``K_red`` and ``M_red`` both filled in.
    """
    n_dof = _check_square_pair(K, M)
    if M is None:
        raise ValueError("IRS needs the parent mass matrix")
    m = _resolve_master(master, n_dof)
    s = _slave_of(m, n_dof)

    T_g = _static_basis(K, m, s)
    K_g = _symmetrize(T_g.T @ _as_dense(K) @ T_g)
    M_g = _symmetrize(T_g.T @ _as_dense(M) @ T_g)

    T = T_g.copy()
    if s.size:
        # S M T_g M_g^-1 K_g, with S non-zero on the slave block only, so the
        # master rows of the correction vanish and T keeps its identity block.
        forces = _as_dense(M) @ T_g @ _solve_spd(M_g, K_g)
        T[s, :] += _solver(_block(K, s, s))(forces[s, :])

    K_red = _symmetrize(T.T @ _as_dense(K) @ T)
    M_red = _symmetrize(T.T @ _as_dense(M) @ T)

    return ReductionResult(
        T=T,
        master=m,
        slave=s,
        K_red=K_red,
        M_red=M_red,
        method="irs",
        meta={
            "n_dof": n_dof,
            "sparse": bool(sp.issparse(K)),
            "guyan_T": T_g,
            "guyan_K": K_g,
            "guyan_M": M_g,
        },
    )


def serep(
    phi: np.ndarray,
    master_rows: Any,
    K: Any = None,  # noqa: N803
    M: Any = None,  # noqa: N803
    *,
    rcond: float | None = None,
) -> ReductionResult:
    """System Equivalent Reduction Expansion Process (SEREP).

    Given parent mode shapes ``phi`` of shape ``(n, n_modes)`` and the rows a
    test would actually measure, the basis is::

        T = phi @ pinv(phi[master_rows])

    ``T @ phi[master_rows] == phi`` exactly when ``n_master >= n_modes`` and the
    measured partition has full column rank, so *every* retained mode -- shape
    and frequency alike -- survives the reduction to machine precision.  That
    is the property the static bases cannot offer, and it is what makes SEREP
    the natural expansion operator for test data: the slave (unmeasured) DOFs
    are reconstructed from the master ones through the analytical modes.

    The price is that the modes must exist first, and that the result is only
    as good as the mode set: motion outside ``span(phi)`` is filtered out, and
    ``T[master]`` is the identity only in the square, full-rank case (otherwise
    it is the orthogonal projector onto the measured mode partition).

    Parameters
    ----------
    phi:
        Parent mode shapes, ``(n_dof, n_modes)``.
    master_rows:
        Measured DOF indices, or a boolean mask of length ``n_dof``.
    K, M:
        Optional parent matrices; when given the reduced pair is filled in.
    rcond:
        Relative cut-off for the pseudo-inverse, passed to
        :func:`numpy.linalg.pinv`.  The default lets numpy choose.

    Returns
    -------
    ReductionResult
    """
    modes = np.asarray(phi, dtype=float)
    if modes.ndim == 1:
        modes = modes[:, None]
    if modes.ndim != 2:
        raise ValueError("phi must be a (n_dof, n_modes) array of mode shapes")
    n_dof, n_modes = modes.shape
    if n_modes == 0:
        raise ValueError("phi has no modes to build a SEREP basis from")

    m = _resolve_master(master_rows, n_dof)
    s = _slave_of(m, n_dof)
    phi_m = modes[m, :]
    pinv = np.linalg.pinv(phi_m) if rcond is None else np.linalg.pinv(phi_m, rcond=rcond)
    T = modes @ pinv

    K_red = None if K is None else _symmetrize(T.T @ _as_dense(K) @ T)
    M_red = None if M is None else _symmetrize(T.T @ _as_dense(M) @ T)

    singular = np.linalg.svd(phi_m, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular.max() * max(phi_m.shape) * 1.0e-15))
    denominator = np.linalg.norm(modes)
    residual = float(np.linalg.norm(T @ phi_m - modes) / denominator) if denominator else 0.0

    return ReductionResult(
        T=T,
        master=m,
        slave=s,
        K_red=K_red,
        M_red=M_red,
        method="serep",
        meta={
            "n_dof": int(n_dof),
            "n_modes": int(n_modes),
            "rank": rank,
            "condition": float(singular.max() / singular.min()) if singular.min() > 0 else np.inf,
            "mode_reconstruction_error": residual,
            "exact": bool(rank >= n_modes),
        },
    )
