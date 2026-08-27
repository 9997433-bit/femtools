"""Craig-Bampton fixed-interface component mode synthesis.

The physical DOFs are split into boundary (retained) and interior (condensed) sets. The
reduction basis combines static constraint modes with fixed-interface normal modes::

    u_i = Psi_c u_b + Phi_i q,      Psi_c = -K_ii^-1 K_ib
    T   = [[ I , 0     ],
           [ Psi_c, Phi_i ]]

so that ``K_r = T^T K T`` and ``M_r = T^T M T``. Taking ``n_modes = 0`` degenerates to
Guyan (static) condensation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._utils import TWO_PI, as_dense, factorized_solver, is_sparse, resolve_dofs, symmetrize
from .modal import ModalModel

__all__ = ["CraigBamptonResult", "craig_bampton"]


@dataclass
class CraigBamptonResult:
    """Reduced Craig-Bampton component.

    Attributes
    ----------
    K, M:
        Reduced stiffness and mass, shape ``(n_b + n_modal, n_b + n_modal)``. The
        generalised DOF order is ``[boundary_dofs..., modal coordinates...]``.
    T:
        Reduction basis in the *original* DOF ordering, shape ``(ndof, n_b + n_modal)``.
    boundary_dofs, interior_dofs:
        Index arrays into the original DOF numbering.
    fixed_freq_hz:
        Fixed-interface natural frequencies of the retained modes, in Hz.
    constraint_modes:
        ``Psi_c``, shape ``(n_i, n_b)``.
    fixed_modes:
        ``Phi_i``, shape ``(n_i, n_modal)``, mass-normalised w.r.t. ``M_ii``.
    """

    K: np.ndarray
    M: np.ndarray
    T: np.ndarray
    boundary_dofs: np.ndarray
    interior_dofs: np.ndarray
    fixed_freq_hz: np.ndarray
    constraint_modes: np.ndarray
    fixed_modes: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_boundary(self) -> int:
        """Number of retained boundary DOFs."""
        return int(self.boundary_dofs.size)

    @property
    def n_modal(self) -> int:
        """Number of retained fixed-interface modes."""
        return int(self.fixed_freq_hz.size)

    @property
    def n_reduced(self) -> int:
        """Size of the reduced model."""
        return self.n_boundary + self.n_modal

    @property
    def ndof(self) -> int:
        """Size of the parent (full) model."""
        return int(self.T.shape[0])

    def expand(self, y: np.ndarray) -> np.ndarray:
        """Expand reduced coordinates back to full physical DOFs (``u = T y``)."""
        arr = np.asarray(y)
        if arr.ndim == 1:
            return self.T @ arr
        return self.T @ arr

    def solve_modes(self, n_modes: int | None = None) -> ModalModel:
        """Eigen-solve the reduced model and expand the modes to the full DOF space."""
        lam, Q = sla.eigh(symmetrize(self.K), symmetrize(self.M))
        lam = np.clip(lam, 0.0, None)
        if n_modes is not None:
            lam, Q = lam[: int(n_modes)], Q[:, : int(n_modes)]
        return ModalModel(
            freq_hz=np.sqrt(lam) / TWO_PI,
            modes=self.T @ Q,
            generalized_mass=np.ones(lam.size),
            eigenvalues=lam,
            meta={"source": "craig_bampton"},
        )


def _submatrix(A: Any, rows: np.ndarray, cols: np.ndarray, sparse: bool) -> Any:  # noqa: N803
    """Row/column sub-block that works for both dense and sparse matrices."""
    if sparse:
        return A[rows, :][:, cols]
    return A[np.ix_(rows, cols)]


def _fixed_interface_modes(
    Kii: Any, Mii: Any, n_modes: int, sparse: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``n_modes`` fixed-interface modes, mass-normalised. Returns ``(w2, Phi)``."""
    n_i = int(Kii.shape[0])
    if n_modes <= 0 or n_i == 0:
        return np.zeros(0), np.zeros((n_i, 0))
    n_modes = min(int(n_modes), n_i)
    if sparse and n_modes < n_i - 1:
        try:
            vals, vecs = spla.eigsh(
                sp.csr_matrix(Kii), k=n_modes, M=sp.csr_matrix(Mii), sigma=0.0, which="LM"
            )
        except (RuntimeError, ValueError, spla.ArpackNoConvergence):
            vals, vecs = sla.eigh(as_dense(Kii), as_dense(Mii))
            vals, vecs = vals[:n_modes], vecs[:, :n_modes]
    else:
        vals, vecs = sla.eigh(symmetrize(as_dense(Kii)), symmetrize(as_dense(Mii)))
        vals, vecs = vals[:n_modes], vecs[:, :n_modes]
    order = np.argsort(vals)
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    # Enforce exact mass normalisation (eigsh can drift).
    gm = np.einsum("ir,ir->r", vecs, np.asarray(as_dense(Mii) @ vecs))
    vecs = vecs / np.sqrt(np.where(gm > 0, gm, 1.0))[None, :]
    return vals, vecs


def craig_bampton(
    K: Any,
    M: Any,
    boundary_dofs: Any,
    n_modes: int = 0,
    *,
    interior_dofs: Any = None,
) -> CraigBamptonResult:
    """Reduce ``(K, M)`` to boundary DOFs plus fixed-interface modes.

    Parameters
    ----------
    K, M:
        Full stiffness and mass matrices, dense or sparse, shape ``(ndof, ndof)``.
    boundary_dofs:
        DOFs retained physically (interface / loaded / measured DOFs).
    n_modes:
        Number of fixed-interface normal modes to keep. ``0`` gives Guyan condensation.
    interior_dofs:
        Interior set; defaults to the complement of ``boundary_dofs``.

    Returns
    -------
    CraigBamptonResult
    """
    sparse = is_sparse(K) or is_sparse(M)
    ndof = int(K.shape[0])
    if K.shape != M.shape or K.shape[0] != K.shape[1]:
        raise ValueError("K and M must be square and equally sized")

    b = np.unique(resolve_dofs(boundary_dofs, ndof, "boundary_dofs"))
    if interior_dofs is None:
        i = np.setdiff1d(np.arange(ndof, dtype=int), b)
    else:
        i = np.unique(resolve_dofs(interior_dofs, ndof, "interior_dofs"))
        if np.intersect1d(b, i).size:
            raise ValueError("boundary_dofs and interior_dofs must be disjoint")
    if b.size == 0:
        raise ValueError("at least one boundary DOF is required")

    Kmat = sp.csr_matrix(K) if sparse else as_dense(K)
    Mmat = sp.csr_matrix(M) if sparse else as_dense(M)

    Kii = _submatrix(Kmat, i, i, sparse)
    Kib = _submatrix(Kmat, i, b, sparse)
    Mii = _submatrix(Mmat, i, i, sparse)

    if i.size:
        solve = factorized_solver(Kii)
        Psi_c = -np.asarray(solve(as_dense(Kib))).reshape(i.size, b.size)
    else:
        Psi_c = np.zeros((0, b.size))

    w2, Phi_i = _fixed_interface_modes(Kii, Mii, n_modes, sparse)

    n_q = Phi_i.shape[1]
    T = np.zeros((ndof, b.size + n_q))
    T[b, np.arange(b.size)] = 1.0
    if i.size:
        T[np.ix_(i, np.arange(b.size))] = Psi_c
        if n_q:
            T[np.ix_(i, b.size + np.arange(n_q))] = Phi_i

    Kr = symmetrize(as_dense(T.T @ (Kmat @ T)))
    Mr = symmetrize(as_dense(T.T @ (Mmat @ T)))
    # The constraint modes are K-orthogonal to the fixed-interface modes by construction;
    # zero the round-off so the reduced model has the textbook block structure.
    if n_q:
        Kr[: b.size, b.size :] = 0.0
        Kr[b.size :, : b.size] = 0.0
        Kr[b.size :, b.size :] = np.diag(w2)

    return CraigBamptonResult(
        K=Kr,
        M=Mr,
        T=T,
        boundary_dofs=b,
        interior_dofs=i,
        fixed_freq_hz=np.sqrt(np.clip(w2, 0.0, None)) / TWO_PI,
        constraint_modes=Psi_c,
        fixed_modes=Phi_i,
        meta={"ndof": ndof, "sparse": bool(sparse), "n_requested_modes": int(n_modes)},
    )
