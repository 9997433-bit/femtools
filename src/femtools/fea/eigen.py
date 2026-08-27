"""Real normal modes analysis (undamped generalized eigenvalue problem)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .assemble import AssemblyResult, assemble_km
from .dofmap import DofMap

__all__ = ["ModalResult", "solve_modes", "mass_normalize"]

_TWO_PI = 2.0 * np.pi


@dataclass
class ModalResult:
    """Normal modes of a structure.

    ``modes`` is ``(n_dof, n_modes)`` over the **full** DOF space, with zeros on
    constrained DOFs, so ``modes.T @ M @ modes`` is the identity for the global
    mass matrix as well as for its free-free partition.
    """

    freq_hz: np.ndarray
    eigenvalues: np.ndarray
    modes: np.ndarray
    generalized_mass: np.ndarray
    generalized_stiffness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dof_map: DofMap | None = None
    free_dof: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    assembly: AssemblyResult | None = None
    method: str = ""
    residual: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # -- aliases used across the rest of the package ------------------
    @property
    def phi(self) -> np.ndarray:
        return self.modes

    @property
    def omega(self) -> np.ndarray:
        """Circular frequencies in rad/s."""
        return _TWO_PI * self.freq_hz

    @property
    def n_modes(self) -> int:
        return int(self.freq_hz.size)

    @property
    def n_dof(self) -> int:
        return int(self.modes.shape[0])

    @property
    def K(self) -> sp.csr_matrix | None:
        return None if self.assembly is None else self.assembly.K

    @property
    def M(self) -> sp.csr_matrix | None:
        return None if self.assembly is None else self.assembly.M

    @property
    def modes_free(self) -> np.ndarray:
        return self.modes[self.free_dof, :] if self.free_dof.size else self.modes

    def mode(self, index: int) -> np.ndarray:
        return self.modes[:, int(index)]

    def node_mode(self, node_id: Any, index: int) -> np.ndarray:
        if self.dof_map is None:
            raise ValueError("ModalResult has no DOF map")
        return self.modes[self.dof_map.node_dofs(node_id), int(index)]

    def orthogonality_error(self) -> float:
        """``max|Phi.T M Phi - I|`` for the returned modes."""
        if self.assembly is None:
            return float("nan")
        g = self.modes.T @ (self.assembly.M @ self.modes)
        return float(np.max(np.abs(g - np.eye(g.shape[0]))))

    def __repr__(self) -> str:  # pragma: no cover - reporting helper
        freqs = ", ".join(f"{f:.6g}" for f in self.freq_hz[:6])
        tail = ", ..." if self.n_modes > 6 else ""
        return f"ModalResult(n_modes={self.n_modes}, freq_hz=[{freqs}{tail}])"


def mass_normalize(phi: np.ndarray, M: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Return modes satisfying ``phi.T @ M @ phi = I`` to machine precision.

    A Cholesky factor of the (already nearly diagonal) generalized mass matrix
    is used, which also cleans up the arbitrary mixing that ARPACK leaves
    within clusters of repeated eigenvalues.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.size == 0:
        return phi
    g = phi.T @ (M @ phi)
    g = 0.5 * (g + g.T)
    try:
        L = np.linalg.cholesky(g)
        phi_n = sla.solve_triangular(L, phi.T, lower=True).T
    except np.linalg.LinAlgError:
        # Fall back to per-mode scaling when the Gram matrix is not PD
        # (massless mechanism in the retained set).
        scale = np.sqrt(np.abs(np.diag(g)))
        scale[scale == 0.0] = 1.0
        phi_n = phi / scale
    # Deterministic sign: largest magnitude entry of each mode is positive.
    for j in range(phi_n.shape[1]):
        col = phi_n[:, j]
        k = int(np.argmax(np.abs(col)))
        if col[k] < 0.0:
            phi_n[:, j] = -col
    return phi_n


def _auto_sigma(K: sp.csr_matrix, M: sp.csr_matrix) -> float:
    """A small negative shift for a possibly singular stiffness matrix.

    Shift-invert maps ``lambda`` to ``1 / (lambda - sigma)``, so a rigid body
    mode is separated from the first elastic mode by ``1 + lambda_1/|sigma|``:
    the shift must be *small* compared with the elastic spectrum, not merely
    non-zero.  It is therefore placed nine decades below the spectral radius
    estimate ``max(K_ii / M_ii)``, which keeps the separation large while
    leaving ``K - sigma*M`` comfortably factorisable in double precision.
    """
    return -1.0e-9 * spectral_scale(K, M)


def spectral_scale(K: sp.csr_matrix, M: sp.csr_matrix) -> float:
    """Upper bound estimate of the largest eigenvalue, ``max(K_ii / M_ii)``."""
    kd = np.abs(K.diagonal())
    md = np.abs(M.diagonal())
    good = (kd > 0.0) & (md > 0.0)
    if good.any():
        hi = float((kd[good] / md[good]).max())
    elif md.sum() > 0.0:
        hi = float(kd.sum() / md.sum())
    else:
        hi = 1.0
    return hi if np.isfinite(hi) and hi > 0.0 else 1.0


def _m_orthonormalize(X: np.ndarray, M: sp.spmatrix) -> np.ndarray:
    """Return an ``M``-orthonormal basis of the column space of ``X``."""
    G = X.T @ (M @ X)
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    keep = w > max(w.max(), 0.0) * 1.0e-12
    if not keep.any():
        return X[:, :0]
    return X @ (V[:, keep] / np.sqrt(w[keep]))


def _augment(X: np.ndarray, Y: np.ndarray, M: sp.spmatrix, tol: float = 1.0e-8) -> np.ndarray:
    """Extend the ``M``-orthonormal basis ``X`` with the new content of ``Y``.

    A column of ``Y`` is only admitted when the part of it that is ``M``-
    orthogonal to ``X`` is a non-negligible fraction of the column itself.
    Without that test the round-off left over from an already converged
    direction would be renormalised into a unit vector of pure noise.
    """
    if Y.size == 0:
        return X
    before = np.sqrt(np.abs(np.einsum("ij,ij->j", Y, M @ Y)))
    # Two Gram-Schmidt passes: the shift-invert image is dominated by the
    # lowest modes by nine decades, so one pass does not leave Y numerically
    # M-orthogonal to X.
    for _ in range(2):
        Y = Y - X @ (X.T @ (M @ Y))
    after = np.sqrt(np.abs(np.einsum("ij,ij->j", Y, M @ Y)))
    keep = after > tol * np.maximum(before, np.finfo(float).tiny)
    if not keep.any():
        return X
    new = _m_orthonormalize(Y[:, keep], M)
    return np.hstack([X, new]) if new.shape[1] else X


def _rayleigh_ritz_refine(
    K: sp.csr_matrix,
    M: sp.csr_matrix,
    solve_shifted,
    vecs: np.ndarray,
    n_req: int,
    *,
    n_pad: int,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift-invert subspace iteration seeded with the ARPACK vectors.

    Rayleigh-Ritz over a subspace can only over-estimate eigenvalues, so
    enriching the ARPACK basis with a few inverse-iterated random vectors is a
    monotone improvement.  In practice this is what recovers the members of a
    repeated cluster (rigid body modes) that ARPACK occasionally drops.

    Each sweep spans the *union* of the current basis and its shift-inverted
    image rather than replacing one with the other.  Replacing would be fatal
    for a free-free model: shift-invert maps the rigid body modes to
    ``1 / |sigma|``, nine decades above the elastic spectrum, so after two or
    three sweeps every column has converged onto the six rigid body modes and
    the elastic modes are lost to the rank tolerance.
    """
    n = K.shape[0]
    if n_pad > 0:
        rng = np.random.default_rng(0xF3A1)
        pad = solve_shifted(M @ rng.standard_normal((n, n_pad)))
        X = np.hstack([vecs, pad])
    else:
        X = vecs.copy()
    Z = _m_orthonormalize(X, M)
    budget = int(min(n, max(2 * Z.shape[1], n_req + 8)))
    for _ in range(max(1, iterations)):
        if Z.shape[1] == 0 or Z.shape[1] >= budget:
            break
        Z = _augment(Z, solve_shifted(M @ Z), M)
    # The Ritz projection below assumes ``Z.T M Z == I``; re-orthonormalise so
    # that a loss of orthogonality during the sweeps cannot manufacture Ritz
    # values outside the spectrum of the pencil.
    Z = _m_orthonormalize(Z, M)
    if Z.shape[1] == 0:
        return np.zeros(0), Z
    Kr = Z.T @ (K @ Z)
    w, S = np.linalg.eigh(0.5 * (Kr + Kr.T))
    take = min(n_req, Z.shape[1])
    return w[:take], (Z @ S)[:, :take]


def _dense_shift_invert(
    K: np.ndarray, M: np.ndarray, n_modes: int, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dense generalized solve tolerant of a singular (semi-definite) ``M``."""
    A = K - sigma * M
    try:
        L = np.linalg.cholesky(0.5 * (A + A.T))
    except np.linalg.LinAlgError:
        vals, vecs = sla.eigh(0.5 * (K + K.T), 0.5 * (M + M.T))
        order = np.argsort(vals)[:n_modes]
        return vals[order], vecs[:, order]
    S = sla.solve_triangular(L, M, lower=True)
    S = sla.solve_triangular(L, S.T, lower=True).T
    S = 0.5 * (S + S.T)
    mu, y = np.linalg.eigh(S)
    order = np.argsort(mu)[::-1]
    mu = mu[order]
    y = y[:, order]
    keep = mu > (np.max(np.abs(mu)) * 1.0e-14 if mu.size else 0.0)
    mu, y = mu[keep], y[:, keep]
    lam = sigma + 1.0 / mu
    x = sla.solve_triangular(L, y, lower=True, trans="T")
    order = np.argsort(lam)[: min(n_modes, lam.size)]
    return lam[order], x[:, order]


def solve_modes(
    model: Any,
    n_modes: int = 10,
    shift: float = 0.0,
    *,
    assembly: AssemblyResult | None = None,
    method: str = "auto",
    sigma: float | None = None,
    freq_shift_hz: float | None = None,
    tol: float = 0.0,
    maxiter: int | None = None,
    dense_threshold: int = 30,
    refine: bool = True,
    **assemble_kwargs: Any,
) -> ModalResult:
    """Compute the lowest ``n_modes`` real normal modes.

    Parameters
    ----------
    model
        Model database (duck typed).
    n_modes
        Number of modes requested; silently clipped to the number of free DOFs.
    shift
        Eigenvalue shift in ``omega**2`` (rad/s)**2.  ``0.0`` (the default)
        selects an automatic small negative shift so that rigid body modes and
        singular stiffness matrices are handled by the shift-invert
        factorisation.
    freq_shift_hz
        Convenience alternative to ``shift`` expressed in Hz.
    method
        ``"auto"`` (ARPACK shift-invert, dense for small systems), ``"eigsh"``
        or ``"dense"``.
    refine
        Run a Rayleigh-Ritz refinement on the ARPACK basis enriched with a few
        inverse-iterated vectors.  Cheap (the shift-invert factorisation is
        reused) and makes clusters of repeated eigenvalues reliable.

    Returns
    -------
    ModalResult
        Frequencies in ascending Hz, eigenvalues in (rad/s)**2 and
        mass-normalized mode shapes over the full DOF space.
    """
    asm = assembly if assembly is not None else assemble_km(model, **assemble_kwargs)
    Kff = asm.Kff
    Mff = asm.Mff
    n_free = Kff.shape[0]
    if n_free == 0:
        raise ValueError("no free degrees of freedom left after constraints")

    n_req = int(max(1, min(int(n_modes), n_free)))

    if freq_shift_hz is not None:
        shift = (_TWO_PI * float(freq_shift_hz)) ** 2
    if sigma is None:
        sigma = float(shift) if shift else _auto_sigma(Kff, Mff)

    use_dense = method == "dense" or (
        method == "auto" and (n_free <= dense_threshold or n_free < n_req + 2)
    )
    if method == "eigsh" and n_free < n_req + 2:
        use_dense = True

    lam: np.ndarray
    vecs: np.ndarray
    used = "dense-shift-invert"
    if use_dense:
        lam, vecs = _dense_shift_invert(Kff.toarray(), Mff.toarray(), n_req, sigma)
    else:
        used = "eigsh-shift-invert"
        # Spare vectors and a generous Krylov subspace keep ARPACK from
        # dropping a member of a cluster of repeated eigenvalues (rigid body
        # modes, symmetric structures).  The start vector is seeded so results
        # are reproducible run to run.
        k_solve = int(min(n_free - 1, n_req + max(4, n_req // 2)))
        ncv = int(min(n_free, max(3 * k_solve + 1, k_solve + 30)))
        v0 = np.random.default_rng(20240501).standard_normal(n_free)
        try:
            lu = splu((Kff - sigma * Mff).tocsc())
            opinv = LinearOperator((n_free, n_free), matvec=lu.solve, dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lam, vecs = eigsh(
                    Kff,
                    k=k_solve,
                    M=Mff,
                    sigma=sigma,
                    which="LM",
                    ncv=ncv,
                    v0=v0,
                    OPinv=opinv,
                    tol=tol,
                    maxiter=maxiter,
                )
            if refine:
                used = "eigsh-shift-invert + Rayleigh-Ritz refinement"
                lam, vecs = _rayleigh_ritz_refine(
                    Kff,
                    Mff,
                    lu.solve,
                    vecs,
                    n_req,
                    n_pad=int(min(max(4, n_req // 4), max(0, n_free - k_solve))),
                )
        except Exception:  # pragma: no cover - ARPACK/factorisation failure
            used = "dense-shift-invert (eigsh fallback)"
            lam, vecs = _dense_shift_invert(Kff.toarray(), Mff.toarray(), n_req, sigma)

    order = np.argsort(lam)[:n_req]
    lam = np.asarray(lam, dtype=float)[order]
    vecs = np.asarray(vecs, dtype=float)[:, order]
    vecs = mass_normalize(vecs, Mff)

    # A mode counts as rigid when its eigenvalue is negligible against the
    # spectral radius of the system, not merely against the modes returned.
    zero_tol = 1.0e-11 * spectral_scale(Kff, Mff)
    lam_clean = np.where(np.abs(lam) < zero_tol, 0.0, lam)
    freq = np.sqrt(np.clip(lam_clean, 0.0, None)) / _TWO_PI

    modes_full = asm.expand(vecs)
    gen_mass = np.einsum("ij,ij->j", vecs, Mff @ vecs)
    gen_stiff = np.einsum("ij,ij->j", vecs, Kff @ vecs)
    residual = np.linalg.norm(Kff @ vecs - (Mff @ vecs) * lam, axis=0)

    return ModalResult(
        freq_hz=freq,
        eigenvalues=lam,
        modes=modes_full,
        generalized_mass=gen_mass,
        generalized_stiffness=gen_stiff,
        dof_map=asm.dof_map,
        free_dof=asm.free_dof,
        assembly=asm,
        method=used,
        residual=residual,
    )
