"""Residual (attachment) vectors compensating modal truncation.

Classical MacNeal residual flexibility: for a set of load directions ``F`` the static
response ``K^-1 F`` is stripped of the content already carried by the retained modes,
leaving the *residual flexibility*. Mass-orthonormalising what remains and solving the
small eigenproblem on that subspace yields residual vectors that can simply be appended
to the modal basis (they behave like extra high-frequency modes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._utils import TWO_PI, as_dense, factorized_solver, resolve_dofs, symmetrize
from .modal import ModalModel, as_modal

__all__ = ["ResidualVectorResult", "residual_vectors"]


@dataclass
class ResidualVectorResult:
    """Residual vectors and the residual flexibility they were built from.

    Attributes
    ----------
    vectors:
        Residual vectors ``Psi``, shape ``(ndof, n_res)``, mass-orthonormal and
        K-orthogonal (``Psi.T M Psi = I``, ``Psi.T K Psi = diag(omega^2)``).
    freq_hz:
        Pseudo-frequencies of the residual vectors in Hz, shape ``(n_res,)``.
    residual_flexibility:
        ``K^-1 F`` minus the retained-mode content, shape ``(ndof, n_force)``. This is the
        static ("upper") residual usable directly by :func:`~femtools.dynamics.frf.modal_frf`.
    force_dofs:
        DOF indices of the unit load directions, or ``None`` when explicit forces were given.
    """

    vectors: np.ndarray
    freq_hz: np.ndarray
    eigenvalues: np.ndarray
    residual_flexibility: np.ndarray
    force_dofs: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_res(self) -> int:
        """Number of retained residual vectors."""
        return int(self.vectors.shape[1])

    def upper_residual(self, outputs: Any = None, inputs: Any = None) -> np.ndarray:
        """Static residual flexibility block for the given output/input DOFs.

        ``inputs`` are matched against :attr:`force_dofs`; the block can be passed as
        ``upper_residual`` to :func:`~femtools.dynamics.frf.modal_frf`.
        """
        ndof = self.residual_flexibility.shape[0]
        out = resolve_dofs(outputs, ndof, "outputs")
        if inputs is None:
            cols = np.arange(self.residual_flexibility.shape[1])
        else:
            if self.force_dofs is None:
                raise ValueError("input selection needs force_dofs to be known")
            want = resolve_dofs(inputs, ndof, "inputs")
            lookup = {int(d): i for i, d in enumerate(self.force_dofs)}
            missing = [int(d) for d in want if int(d) not in lookup]
            if missing:
                raise ValueError(f"no residual flexibility computed for DOFs {missing}")
            cols = np.array([lookup[int(d)] for d in want], dtype=int)
        return self.residual_flexibility[np.ix_(out, cols)]

    def augment(self, modal: Any) -> ModalModel:
        """Return ``modal`` with the residual vectors appended as extra modes."""
        mm = as_modal(modal).mass_normalized()
        return ModalModel(
            freq_hz=np.concatenate([mm.freq_hz, self.freq_hz]),
            modes=np.hstack([mm.modes, self.vectors]),
            generalized_mass=np.ones(mm.n_modes + self.n_res),
            eigenvalues=np.concatenate([np.asarray(mm.eigenvalues), self.eigenvalues]),
            dof_ids=None if mm.dof_ids is None else mm.dof_ids.copy(),
            meta={**mm.meta, "n_residual": self.n_res},
        )


def residual_vectors(
    K: Any,
    M: Any,
    modal: Any,
    force_dofs: Any = None,
    *,
    forces: Any = None,
    tol: float = 1e-10,
    rigid_tol_hz: float = 1e-4,
) -> ResidualVectorResult:
    """Compute MacNeal residual vectors for the given load directions.

    Parameters
    ----------
    K, M:
        Stiffness and mass matrices, shape ``(ndof, ndof)``.
    modal:
        Retained modal basis whose contribution is removed from the static response.
    force_dofs:
        DOFs carrying unit loads. Ignored when ``forces`` is given; defaults to all DOFs
        only if neither is supplied (rarely what you want, so prefer being explicit).
    forces:
        Explicit load matrix ``(ndof, n_force)`` overriding ``force_dofs``.
    tol:
        Relative threshold on the mass-orthogonality spectrum below which a candidate
        residual direction is discarded as linearly dependent.
    rigid_tol_hz:
        Modes below this frequency are treated as rigid-body modes and handled by
        inertia relief rather than by static flexibility.

    Returns
    -------
    ResidualVectorResult
    """
    mm = as_modal(modal).mass_normalized()
    Kd = as_dense(K)
    Md = as_dense(M)
    ndof = Kd.shape[0]
    if Kd.shape != Md.shape or Kd.shape[0] != Kd.shape[1]:
        raise ValueError("K and M must be square and equally sized")
    if mm.ndof != ndof:
        raise ValueError(f"modal basis has {mm.ndof} DOFs but K/M have {ndof}")

    if forces is not None:
        F = np.asarray(forces, dtype=float)
        if F.ndim == 1:
            F = F.reshape(-1, 1)
        if F.shape[0] != ndof:
            raise ValueError(f"forces must have {ndof} rows, got {F.shape[0]}")
        fdofs: np.ndarray | None = None
    else:
        fdofs = resolve_dofs(force_dofs, ndof, "force_dofs", mm.dof_ids)
        F = np.zeros((ndof, fdofs.size))
        F[fdofs, np.arange(fdofs.size)] = 1.0

    rigid = mm.freq_hz <= rigid_tol_hz
    phi_rb = mm.modes[:, rigid]
    phi_el = mm.modes[:, ~rigid]
    wr2_el = np.asarray(mm.eigenvalues, dtype=float)[~rigid]

    # Inertia relief: remove the resultant that the rigid-body modes cannot resist.
    Fe = F.copy()
    if phi_rb.size:
        Fe = Fe - (Md @ phi_rb) @ (phi_rb.T @ F)

    solve = factorized_solver(Kd if not phi_rb.size else _shifted(Kd, Md, phi_rb))
    Xs = np.asarray(solve(Fe)).reshape(ndof, -1)
    if phi_rb.size:  # purge any rigid-body content re-introduced by the shift
        Xs = Xs - phi_rb @ (phi_rb.T @ (Md @ Xs))

    # Strip the flexibility already represented by the retained elastic modes.
    if phi_el.size:
        Xr = Xs - phi_el @ ((phi_el.T @ Fe) / wr2_el[:, None])
    else:
        Xr = Xs

    # Mass-orthonormalise the remainder and drop dependent directions.
    G = symmetrize(Xr.T @ (Md @ Xr))
    evals, evecs = np.linalg.eigh(G)
    scale = evals.max() if evals.size and evals.max() > 0 else 0.0
    keep = evals > max(tol * scale, 0.0)
    if not np.any(keep):
        empty = np.zeros((ndof, 0))
        return ResidualVectorResult(
            vectors=empty,
            freq_hz=np.zeros(0),
            eigenvalues=np.zeros(0),
            residual_flexibility=Xr,
            force_dofs=fdofs,
            meta={"n_candidates": int(Xr.shape[1]), "dropped": int(Xr.shape[1])},
        )
    R = Xr @ (evecs[:, keep] / np.sqrt(evals[keep])[None, :])

    Kr = symmetrize(R.T @ (Kd @ R))
    lam, Q = np.linalg.eigh(Kr)
    lam = np.clip(lam, 0.0, None)
    order = np.argsort(lam)
    lam = lam[order]
    Psi = R @ Q[:, order]

    return ResidualVectorResult(
        vectors=Psi,
        freq_hz=np.sqrt(lam) / TWO_PI,
        eigenvalues=lam,
        residual_flexibility=Xr,
        force_dofs=fdofs,
        meta={
            "n_candidates": int(Xr.shape[1]),
            "dropped": int(Xr.shape[1] - Psi.shape[1]),
            "n_rigid": int(rigid.sum()),
        },
    )


def _shifted(K: np.ndarray, M: np.ndarray, phi_rb: np.ndarray) -> np.ndarray:
    """Make a singular free-free ``K`` invertible by pinning the rigid-body subspace.

    Adding ``M phi_rb phi_rb^T M`` shifts the rigid-body eigenvalues to 1 without touching
    the elastic subspace, so the static solve stays exact for inertia-relieved loads.
    """
    if not phi_rb.size:
        return K
    Mp = M @ phi_rb
    return K + Mp @ Mp.T
