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

__all__ = [
    "ComplexModalResult",
    "ModalResult",
    "mass_normalize",
    "solve_complex_modes",
    "solve_modes",
]

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


def _dense_full_spectrum(
    K: np.ndarray, M: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Complete spectrum of an SPD pencil through ``scipy.linalg.eigh``.

    ``eigh`` reduces ``K x = lambda M x`` with a Cholesky factor of ``M`` and
    then runs a symmetric tridiagonal solver, so the backward error stays at
    the level of ``eps * ||K||``.  The shift-invert path used for a partial
    spectrum cannot match that: it computes ``lambda`` as ``sigma + 1/mu``,
    which for the high end of the spectrum means recovering a large number
    from the reciprocal of a tiny one, and the residual of the top modes
    degrades to about ``1e-10`` relative.  That is invisible when only the
    first few modes are wanted, but a *complete* modal basis is normally
    wanted precisely because it is going to be summed back into something
    exact -- an FRF or a full modal expansion -- where the truncation error is
    zero and 1e-10 is the entire error budget.

    Returns ``None`` when the pencil is not symmetric positive definite, so the
    caller can fall back to the shift-invert path that tolerates rigid body
    modes and semi-definite mass.
    """
    Ks = 0.5 * (K + K.T)
    Ms = 0.5 * (M + M.T)
    try:
        # An explicit Cholesky is the SPD test: eigh itself would happily
        # return negative eigenvalues for an indefinite K.
        np.linalg.cholesky(Ks)
        vals, vecs = sla.eigh(Ks, Ms)
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


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


def _finish(
    asm: AssemblyResult, lam: np.ndarray, vecs: np.ndarray, used: str
) -> ModalResult:
    """Normalise, expand and package a converged eigenpair set."""
    Kff = asm.Kff
    Mff = asm.Mff
    lam = np.asarray(lam, dtype=float)
    vecs = mass_normalize(np.asarray(vecs, dtype=float), Mff)

    # A mode counts as rigid when its eigenvalue is negligible against the
    # spectral radius of the system, not merely against the modes returned.
    zero_tol = 1.0e-11 * spectral_scale(Kff, Mff)
    lam_clean = np.where(np.abs(lam) < zero_tol, 0.0, lam)
    freq = np.sqrt(np.clip(lam_clean, 0.0, None)) / _TWO_PI

    return ModalResult(
        freq_hz=freq,
        eigenvalues=lam,
        modes=asm.expand(vecs),
        generalized_mass=np.einsum("ij,ij->j", vecs, Mff @ vecs),
        generalized_stiffness=np.einsum("ij,ij->j", vecs, Kff @ vecs),
        dof_map=asm.dof_map,
        free_dof=asm.free_dof,
        assembly=asm,
        method=used,
        residual=np.linalg.norm(Kff @ vecs - (Mff @ vecs) * lam, axis=0),
    )


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
        ``"auto"`` (ARPACK shift-invert, dense for small systems, and the
        direct symmetric solver when the *complete* spectrum of an SPD pencil
        is requested), ``"eigsh"``, ``"dense"`` or ``"eigh"``.  ``"eigh"``
        forces the direct solver and raises if the pencil is not positive
        definite.
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

    # A request for every mode of an SPD pencil is the one case where the
    # direct symmetric solver is both affordable and strictly better than
    # shift-invert; see _dense_full_spectrum.
    full_spectrum = n_req >= n_free
    if method == "eigh" or (method in ("auto", "dense") and full_spectrum):
        direct = _dense_full_spectrum(Kff.toarray(), Mff.toarray())
        if direct is not None:
            lam, vecs = direct[0][:n_req], direct[1][:, :n_req]
            return _finish(asm, lam, vecs, "dense-eigh (SPD pencil, complete spectrum)")
        if method == "eigh":
            raise ValueError(
                "method='eigh' requires a symmetric positive definite Kff and Mff; "
                "use method='auto' for a model with rigid body modes"
            )

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
    return _finish(asm, lam, vecs, used)


# ----------------------------------------------------------------------
# damped (complex) modes
# ----------------------------------------------------------------------


@dataclass
class ComplexModalResult:
    """Complex modes of the damped pencil ``(lambda^2 M + lambda C + K) phi = 0``.

    ``modes`` is the *displacement* partition of the state vectors, shape
    ``(n_dof, n_modes)`` and complex: unlike a normal mode it carries a phase
    per DOF, so the structure no longer passes through its undeformed shape all
    at once.  Only one member of each conjugate pair is returned.

    Attributes
    ----------
    freq_hz:
        Undamped natural frequency ``|lambda| / 2 pi`` -- the quantity that
        matches the real normal mode when the damping goes to zero.
    zeta:
        Modal damping ratio ``-Re(lambda) / |lambda|``.  Positive for a stable
        mode; exactly ``1`` marks an overdamped (real, non-oscillatory) root.

    Overdamped roots deserve a warning.  ``(omega_n, zeta)`` describes a
    *conjugate pair*, so it is simply not defined for the two unpaired real
    roots an overdamped mode decays through: for those, ``zeta`` is 1 by the
    formula above and ``freq_hz`` is ``|lambda| / 2 pi``, the decay rate
    expressed as a frequency rather than an oscillation frequency.  Test them
    with :attr:`is_underdamped` before reading them as modal parameters.
    damped_freq_hz:
        ``Im(lambda) / 2 pi``, the frequency actually observed in a ring-down.
    eigenvalues:
        The roots ``lambda`` themselves.
    """

    freq_hz: np.ndarray
    zeta: np.ndarray
    modes: np.ndarray
    eigenvalues: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=complex))
    damped_freq_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dof_map: DofMap | None = None
    free_dof: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    assembly: AssemblyResult | None = None
    method: str = ""
    residual: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def phi(self) -> np.ndarray:
        return self.modes

    @property
    def n_modes(self) -> int:
        return int(self.freq_hz.size)

    @property
    def n_dof(self) -> int:
        return int(self.modes.shape[0])

    @property
    def omega(self) -> np.ndarray:
        """Undamped circular frequencies in rad/s."""
        return _TWO_PI * self.freq_hz

    @property
    def is_underdamped(self) -> np.ndarray:
        return self.zeta < 1.0

    @property
    def modal_phase_deg(self) -> np.ndarray:
        """Phase of every DOF of every mode, in degrees.

        A complex mode whose phases cluster around two values 180 degrees apart
        is effectively a real (normal) mode; a spread in between is the
        signature of non-proportional damping.
        """
        return np.degrees(np.angle(self.modes))

    def mpc(self) -> np.ndarray:
        """Modal phase collinearity of every mode, in ``[0, 1]``.

        One for a mode whose DOFs all move in phase (or exactly out of phase)
        -- a real normal mode wearing a complex scaling -- and it falls towards
        zero as the phases scatter.  This is the number that tells whether the
        damping is proportional in practice: proportional damping leaves the
        real modes untouched and scores 1 to round-off, while a localised
        damper drags the modes it acts on well below it.

        The index is the eccentricity of the scatter of ``(Re, Im)`` pairs
        about their mean, i.e. ``((s1 - s2) / (s1 + s2))**2`` for the two
        eigenvalues of their covariance (Pappa, Elliott & Schenk, 1993).
        """
        out = np.zeros(self.modes.shape[1])
        for j in range(self.modes.shape[1]):
            col = self.modes[:, j]
            points = np.column_stack([col.real, col.imag])
            points = points - points.mean(axis=0)
            if not np.any(points):
                out[j] = 1.0
                continue
            s = np.linalg.eigvalsh(points.T @ points)
            total = float(s.sum())
            out[j] = 1.0 if total <= 0.0 else float(((s[1] - s[0]) / total) ** 2)
        return out

    def mode(self, index: int) -> np.ndarray:
        return self.modes[:, int(index)]

    def real_modes(self) -> np.ndarray:
        """Best real approximation of each complex mode.

        Each column is rotated so its dominant entry is real and then the real
        part is taken -- the usual way of feeding complex test modes into a
        real-mode correlation.
        """
        out = np.zeros(self.modes.shape)
        for j in range(self.modes.shape[1]):
            col = self.modes[:, j]
            k = int(np.argmax(np.abs(col)))
            if col[k] != 0.0:
                col = col * np.exp(-1j * np.angle(col[k]))
            out[:, j] = col.real
        return out

    def __repr__(self) -> str:  # pragma: no cover - reporting helper
        pairs = ", ".join(
            f"{f:.6g}Hz/{z:.4g}" for f, z in zip(self.freq_hz[:4], self.zeta[:4], strict=False)
        )
        tail = ", ..." if self.n_modes > 4 else ""
        return f"ComplexModalResult(n_modes={self.n_modes}, [{pairs}{tail}])"


def _scale_quadratic(
    K: np.ndarray, M: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fan-Lin-Van Dooren scaling of the quadratic pencil.

    A structural pencil in SI units is scaled catastrophically badly: the
    cantilever below has ``||K|| ~ 1e9`` and ``||M|| ~ 1e-3``, so the two
    diagonal blocks of the linearisation differ by twelve decades and QZ
    balances one of them into the round-off of the other.  Substituting
    ``lambda = gamma * mu`` with ``gamma = sqrt(||K|| / ||M||)`` and rescaling
    the whole pencil by ``delta`` puts all three coefficient matrices at unit
    norm, which is worth six to seven digits of eigenvalue accuracy on a
    typical model.  Returns ``(Kt, Mt, Ct, gamma)`` with ``lambda = gamma *
    mu``.

    Fan, Lin & Van Dooren (2004), "Normwise scaling of second order polynomial
    matrices", SIAM J. Matrix Anal. Appl. 26(1).
    """
    nk = float(np.linalg.norm(K))
    nm = float(np.linalg.norm(M))
    nc = float(np.linalg.norm(C))
    if not (nk > 0.0 and nm > 0.0 and np.isfinite(nk) and np.isfinite(nm)):
        return K, M, C, 1.0
    gamma = np.sqrt(nk / nm)
    delta = 2.0 / (nk + gamma * nc)
    return delta * K, (delta * gamma * gamma) * M, (delta * gamma) * C, float(gamma)


def _state_space_pencil(
    K: np.ndarray, M: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric linearisation of the quadratic pencil.

    With ``A z' + B z = 0`` and ``z = [u; u']``::

        A = [[C, M], [M, 0]],    B = [[K, 0], [0, -M]]

    the first block row reproduces ``M u'' + C u' + K u = 0`` and the second is
    the identity ``M u' = M u'``.  Both blocks stay symmetric, which keeps the
    conjugate pairs of a real system exactly paired, and a *singular* ``M`` is
    handled by the QZ algorithm as infinite eigenvalues rather than by an
    explicit inverse.
    """
    n = K.shape[0]
    zero = np.zeros((n, n))
    A = np.block([[C, M], [M, zero]])
    B = np.block([[K, zero], [zero, -M]])
    return A, B


def solve_complex_modes(
    K: Any,  # noqa: N803
    M: Any = None,  # noqa: N803
    C: Any = None,  # noqa: N803
    *,
    n_modes: int | None = None,
    method: str = "auto",
    sort: str = "frequency",
    include_overdamped: bool = True,
    zero_tol: float = 1.0e-11,
) -> ComplexModalResult:
    """Solve the damped eigenproblem ``(lambda^2 M + lambda C + K) phi = 0``.

    Proportional (Rayleigh) damping leaves the real normal modes untouched and
    only shifts the eigenvalues off the imaginary axis; non-proportional
    damping -- a discrete damper, a viscoelastic patch, a joint -- couples them,
    and then the mode shapes themselves become complex.  This routine makes no
    proportionality assumption, so it is the reference the modal damping
    shortcut has to be checked against.

    Parameters
    ----------
    K, M, C:
        Stiffness, mass and damping, dense or sparse, all ``(n, n)``.  A single
        :class:`~femtools.fea.assemble.AssemblyResult` may be passed instead,
        in which case its free-free partitions are used and the result is
        expanded to the full DOF space.  ``C=None`` means undamped, which
        reproduces the real normal modes with ``zeta == 0``.
    n_modes:
        Keep only the lowest this many; ``None`` keeps all ``n`` of them.
    method:
        ``"auto"``/``"state_space"`` for the symmetric linearisation above, or
        ``"companion"`` for the explicit first-order form
        ``[[0, I], [-M^-1 K, -M^-1 C]]``, which is cheaper but needs a
        non-singular ``M``.
    sort:
        ``"frequency"`` (ascending ``|lambda|``, the default) or ``"damping"``
        (ascending ``zeta``).
    include_overdamped:
        Keep the real negative roots of overdamped modes.  These have no
        conjugate partner and ``zeta >= 1``; dropping them gives the
        oscillatory spectrum only.
    zero_tol:
        Relative threshold below which an eigenvalue counts as a rigid body
        mode (zero frequency).

    Returns
    -------
    ComplexModalResult
    """
    asm: AssemblyResult | None = None
    if isinstance(K, AssemblyResult):
        asm = K
        K, M, C = asm.Kff, asm.Mff, asm.Cff
    if M is None:
        raise ValueError("solve_complex_modes needs a mass matrix")

    Kd = _dense(K)
    Md = _dense(M)
    Cd = np.zeros_like(Kd) if C is None else _dense(C)
    n = Kd.shape[0]
    if Kd.shape != Md.shape or Kd.shape != Cd.shape or n != Kd.shape[1]:
        raise ValueError("K, M and C must be square and equally sized")
    if n == 0:
        raise ValueError("no degrees of freedom to solve")

    Kt, Mt, Ct, gamma = _scale_quadratic(Kd, Md, Cd)
    if method == "companion":
        lam, states = _companion_modes(Kt, Mt, Ct)
        used = "companion first-order"
    elif method in ("auto", "state_space", "state-space"):
        lam, states = _linearized_modes(Kt, Mt, Ct)
        used = "symmetric state-space linearisation (QZ)"
    else:
        raise ValueError(f"unknown method {method!r}")
    lam = lam * gamma

    # A real system has conjugate pairs; report one member of each, taking the
    # one with a non-negative imaginary part so the frequency is positive.
    keep = lam.imag >= 0.0
    if not include_overdamped:
        keep &= np.abs(lam.imag) > zero_tol * np.max(np.abs(lam), initial=1.0)
    lam = lam[keep]
    states = states[:, keep]

    magnitude = np.abs(lam)
    scale = float(magnitude.max()) if magnitude.size else 1.0
    magnitude = np.where(magnitude < zero_tol * scale, 0.0, magnitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = np.where(magnitude > 0.0, -lam.real / np.where(magnitude > 0.0, magnitude, 1.0), 0.0)
    # An undamped mode comes back with |zeta| of order 1e-13 and an arbitrary
    # sign; left alone, half of those look like negative damping and trip every
    # downstream stability check.  Snap them to zero -- no real structure has a
    # damping ratio that small anyway.
    zeta = np.where(np.abs(zeta) < zero_tol, 0.0, zeta)

    if sort == "damping":
        order = np.lexsort((magnitude, zeta))
    elif sort == "frequency":
        order = np.lexsort((zeta, magnitude))
    else:
        raise ValueError(f"unknown sort {sort!r}")
    if n_modes is not None:
        order = order[: int(max(0, n_modes))]
    lam = lam[order]
    states = states[:, order]
    magnitude = magnitude[order]
    zeta = zeta[order]

    modes = states[:n, :]
    # Scale each mode so its largest entry is 1: the state vectors come out of
    # QZ with an arbitrary complex scaling, which would otherwise leak into the
    # phases the caller reads off modal_phase_deg.
    for j in range(modes.shape[1]):
        col = modes[:, j]
        k = int(np.argmax(np.abs(col)))
        if col[k] != 0.0:
            modes[:, j] = col / col[k]

    absolute = np.linalg.norm((Md @ modes) * lam**2 + (Cd @ modes) * lam + Kd @ modes, axis=0)
    reference = (
        np.abs(lam) ** 2 * np.linalg.norm(Md)
        + np.abs(lam) * np.linalg.norm(Cd)
        + np.linalg.norm(Kd)
    ) * np.linalg.norm(modes, axis=0)
    residual = absolute / np.maximum(reference, np.finfo(float).tiny)

    return ComplexModalResult(
        freq_hz=magnitude / _TWO_PI,
        zeta=zeta,
        modes=asm.expand(modes) if asm is not None else modes,
        eigenvalues=lam,
        damped_freq_hz=lam.imag / _TWO_PI,
        dof_map=None if asm is None else asm.dof_map,
        free_dof=np.zeros(0, dtype=int) if asm is None else asm.free_dof,
        assembly=asm,
        method=used,
        residual=residual,
    )


def _dense(A: Any) -> np.ndarray:  # noqa: N803
    if sp.issparse(A):
        return np.asarray(A.toarray(), dtype=float)
    return np.asarray(A, dtype=float)


def _linearized_modes(
    K: np.ndarray, M: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of the symmetric linearisation, infinite roots removed."""
    A, B = _state_space_pencil(K, M, C)
    # The homogeneous form keeps (alpha, beta) apart instead of forming
    # alpha/beta = inf, so an eigenvalue at infinity can be recognised and
    # dropped rather than poisoning the sort with a NaN.
    w, vecs = sla.eig(-B, A, homogeneous_eigvals=True)
    alpha, beta = np.asarray(w[0]).ravel(), np.asarray(w[1]).ravel()
    # beta == 0 is an eigenvalue at infinity: the deficiency of a singular mass
    # matrix, i.e. a DOF with stiffness but no inertia. It is not a mode.
    finite = np.abs(beta) > np.max(np.abs(beta), initial=1.0) * 1.0e-12
    return (alpha[finite] / beta[finite]).astype(complex), vecs[:, finite]


def _companion_modes(
    K: np.ndarray, M: np.ndarray, C: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenpairs of ``[[0, I], [-M^-1 K, -M^-1 C]]``."""
    n = K.shape[0]
    try:
        lu_piv = sla.lu_factor(M)
    except (np.linalg.LinAlgError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("method='companion' needs a non-singular mass matrix") from exc
    A = np.block(
        [
            [np.zeros((n, n)), np.eye(n)],
            [-sla.lu_solve(lu_piv, K), -sla.lu_solve(lu_piv, C)],
        ]
    )
    lam, vecs = sla.eig(A)
    return np.asarray(lam, dtype=complex), vecs
