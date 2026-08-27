"""Free-interface component mode synthesis: MacNeal and Rubin.

A free-interface component is described by the modes it has with *nothing* attached to
its interface. That is what makes the family the natural companion of measured modal
data — a test article can be hung on soft springs, it cannot be given a fixed interface —
and it is also its weakness: the interface flexibility carried by the deleted modes is
missing, so a component coupled on its truncated free modes alone comes out far too
stiff. Both methods implemented here restore that flexibility explicitly.

The *residual flexibility* of the deleted modes, evaluated for unit loads at the
interface DOFs ``b``, is

    G_d = K^-1 - sum_{r kept} phi_r phi_r^T / omega_r^2,      Psi_a = G_d[:, b]

(inertia-relieved through :func:`~femtools.dynamics.residuals.residual_vectors` when the
component is free-free, so that a singular ``K`` is not a problem). Mass-orthonormalising
those attachment columns turns them into residual pseudo-modes ``Psi`` that extend the
kept basis, ``T = [Phi_k | Psi]``, and because the residual flexibility is M- and
K-orthogonal to the kept modes by construction the reduced model is diagonal::

    T^T K T = diag(Lambda_k, Lambda_res),     T^T M T = I

The two classical methods differ only in what they do with the *inertia* of the residual
coordinates:

* :func:`macneal` neglects it (``M_res = 0``). The residual coordinates then respond
  statically and contribute exactly the first-order residual flexibility — MacNeal's
  hybrid method. The reduced mass matrix is singular by design, which is not a defect:
  massless coordinates are condensed when the component is coupled or solved.
* :func:`rubin` keeps it. The deleted-mode receptance is then reproduced to second order,
  ``G_d + omega^2 G_d M G_d + O(omega^4)``, which is Rubin's improvement over MacNeal.

:func:`free_interface_assembly` couples the reduced components through their *physical*
interface DOFs (rigid ties by null-space elimination, elastic ties as rank-one stiffness
terms) and solves the coupled eigenproblem, condensing whatever carries no mass. A
Rubin component whose mass matrix is regular can equally well be handed to
:func:`~femtools.dynamics.mba.modal_based_assembly` as an ordinary modal model through
:meth:`FreeCMSResult.solve_modes`.

References
----------
MacNeal, R.H., *A Hybrid Method of Component Mode Synthesis*, Computers & Structures,
1(4), 1971, pp. 581-601. Rubin, S., *Improved Component-Mode Representation for
Structural Dynamic Analysis*, AIAA Journal, 13(8), 1975, pp. 995-1006.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._utils import TWO_PI, as_dense, is_sparse, resolve_dofs, symmetrize
from .mba import _nullspace
from .modal import ModalModel, as_modal
from .residuals import residual_vectors

__all__ = [
    "FreeCMSComponent",
    "FreeCMSResult",
    "FreeInterfaceAssembly",
    "free_interface_assembly",
    "macneal",
    "rubin",
]

#: Relative threshold below which a reduced coordinate counts as massless.
MASS_TOL = 1e-10


@dataclass
class FreeCMSResult:
    """A component reduced on free-interface modes plus residual flexibility.

    Attributes
    ----------
    K, M:
        Reduced stiffness and mass, shape ``(n_kept + n_residual,)`` square. The
        generalised DOF order is ``[kept free-interface modes..., residual modes...]``.
        ``K`` is diagonal and ``M`` is the identity for :func:`rubin`; :func:`macneal`
        zeroes the residual rows and columns of ``M``, which makes it singular.
    T:
        Reduction basis ``[Phi_k | Psi]`` in the original DOF ordering, shape
        ``(ndof, n_kept + n_residual)``.
    boundary_dofs, interior_dofs:
        Index arrays into the original DOF numbering. The boundary set is what the
        residual attachment modes are computed for.
    free_freq_hz:
        Frequencies of the retained free-interface normal modes, in Hz (rigid-body modes
        included, at 0 Hz).
    residual_freq_hz:
        Pseudo-frequencies of the residual modes, in Hz. They are above the retained band
        by construction and are not physical resonances.
    residual_flexibility:
        ``G_d[:, b]``, the raw residual attachment modes before orthonormalisation, shape
        ``(ndof, n_boundary)``.
    method:
        ``"rubin"`` or ``"macneal"``.
    """

    K: np.ndarray
    M: np.ndarray
    T: np.ndarray
    boundary_dofs: np.ndarray
    interior_dofs: np.ndarray
    free_freq_hz: np.ndarray
    residual_freq_hz: np.ndarray
    residual_flexibility: np.ndarray
    method: str = "rubin"
    meta: dict[str, Any] = field(default_factory=dict)

    # -- sizes ------------------------------------------------------------
    @property
    def n_kept(self) -> int:
        """Number of retained free-interface normal modes."""
        return int(self.free_freq_hz.size)

    @property
    def n_residual(self) -> int:
        """Number of residual modes actually retained (rank of the residual flexibility)."""
        return int(self.residual_freq_hz.size)

    @property
    def n_reduced(self) -> int:
        """Size of the reduced model."""
        return self.n_kept + self.n_residual

    @property
    def n_boundary(self) -> int:
        """Number of interface DOFs."""
        return int(self.boundary_dofs.size)

    @property
    def ndof(self) -> int:
        """Size of the parent (full) model."""
        return int(self.T.shape[0])

    # -- basis blocks -----------------------------------------------------
    @property
    def normal_modes(self) -> np.ndarray:
        """Retained free-interface normal modes, shape ``(ndof, n_kept)``."""
        return self.T[:, : self.n_kept]

    @property
    def residual_modes(self) -> np.ndarray:
        """Mass-orthonormalised residual modes, shape ``(ndof, n_residual)``."""
        return self.T[:, self.n_kept :]

    def interface_flexibility(self) -> np.ndarray:
        """Residual flexibility seen at the interface, shape ``(n_boundary, n_boundary)``.

        This is the ``G_d[b, b]`` block that MacNeal's method adds to the interface
        compliance of the truncated modal model.
        """
        return self.residual_flexibility[self.boundary_dofs, :]

    def expand(self, y: np.ndarray) -> np.ndarray:
        """Expand reduced coordinates back to full physical DOFs (``u = T y``)."""
        return self.T @ np.asarray(y)

    def solve_modes(self, n_modes: int | None = None) -> ModalModel:
        """Eigen-solve the reduced component and expand the modes to full physical DOFs.

        Massless coordinates (MacNeal's residual set) are condensed statically first, so
        the result is the free component's own modes — for a *free* component that is the
        retained normal modes again, since nothing loads the interface. The residual
        coordinates earn their keep only once the component is coupled; see
        :func:`free_interface_assembly`.
        """
        lam, Q = _semidefinite_modes(self.K, self.M)
        if n_modes is not None:
            lam, Q = lam[: int(n_modes)], Q[:, : int(n_modes)]
        return ModalModel(
            freq_hz=np.sqrt(lam) / TWO_PI,
            modes=self.T @ Q,
            generalized_mass=np.ones(lam.size),
            eigenvalues=lam,
            meta={"source": f"cms_free.{self.method}"},
        )


def _semidefinite_modes(
    K: Any, M: Any, tol: float = MASS_TOL
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``K q = lam M q`` allowing a singular (semi-definite) ``M``.

    Coordinates carrying no mass cannot oscillate; they follow the massive ones
    statically. They are therefore rotated out (eigen-decomposition of ``M``), condensed
    into the massive block by a Schur complement, and recovered afterwards. With a
    regular ``M`` this is a plain :func:`scipy.linalg.eigh`.

    Returns ``(lam, Q)`` with ascending ``lam`` and ``Q^T M Q = I``.
    """
    Ks = symmetrize(as_dense(K))
    Ms = symmetrize(as_dense(M))
    w, V = np.linalg.eigh(Ms)
    scale = float(w.max()) if w.size else 0.0
    massive = w > max(tol * scale, 0.0)
    if bool(massive.all()):
        lam, Q = sla.eigh(Ks, Ms)
        return np.clip(lam, 0.0, None), Q
    if not bool(massive.any()):
        raise ValueError("the reduced model carries no mass; nothing to solve")

    Vm, V0 = V[:, massive], V[:, ~massive]
    Kmm = symmetrize(Vm.T @ Ks @ Vm)
    K0m = V0.T @ Ks @ Vm
    K00 = symmetrize(V0.T @ Ks @ V0)
    try:
        X = -np.linalg.solve(K00, K0m)
    except np.linalg.LinAlgError as exc:  # a massless *and* stiffness-free direction
        raise ValueError(
            "the massless coordinates of the reduced model are also unrestrained by "
            "stiffness, so they form a mechanism and cannot be condensed"
        ) from exc
    Keff = symmetrize(Kmm + K0m.T @ X)
    lam, Qm = sla.eigh(Keff, np.diag(w[massive]))
    return np.clip(lam, 0.0, None), Vm @ Qm + V0 @ (X @ Qm)


def _free_interface_modes(
    K: Any, M: Any, n_modes: int, sparse: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``n_modes`` free-interface modes, mass-normalised. Returns ``(w2, Phi)``.

    ``K`` may be singular (a free-free component), so the shift-invert spectral transform
    is taken slightly *below* zero instead of at zero.
    """
    ndof = int(K.shape[0])
    n_modes = min(int(n_modes), ndof)
    if n_modes <= 0:
        return np.zeros(0), np.zeros((ndof, 0))
    if sparse and n_modes < ndof - 1:
        Kc, Mc = sp.csr_matrix(K), sp.csr_matrix(M)
        mass = float(Mc.diagonal().sum())
        stiff = float(Kc.diagonal().sum())
        sigma = -1e-6 * stiff / mass if mass > 0.0 and stiff > 0.0 else -1.0
        try:
            vals, vecs = spla.eigsh(Kc, k=n_modes, M=Mc, sigma=sigma, which="LM")
        except (RuntimeError, ValueError, spla.ArpackNoConvergence):
            vals, vecs = sla.eigh(symmetrize(as_dense(K)), symmetrize(as_dense(M)))
            vals, vecs = vals[:n_modes], vecs[:, :n_modes]
    else:
        vals, vecs = sla.eigh(symmetrize(as_dense(K)), symmetrize(as_dense(M)))
        vals, vecs = vals[:n_modes], vecs[:, :n_modes]
    order = np.argsort(vals)
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    gm = np.einsum("ir,ir->r", vecs, np.asarray(as_dense(M) @ vecs))
    vecs = vecs / np.sqrt(np.where(gm > 0, gm, 1.0))[None, :]
    return vals, vecs


def _kept_basis(
    K: Any,
    M: Any,
    n_modes: int | None,
    modal: Any,
    sparse: bool,
    rigid_tol_hz: float,
) -> tuple[ModalModel, int]:
    """Resolve the retained free-interface basis and count its rigid-body modes."""
    if modal is not None:
        mm = as_modal(modal).mass_normalized()
        if n_modes is not None:
            mm = mm.truncate(int(n_modes))
        if mm.n_modes == 0:
            raise ValueError("the supplied modal basis retains no modes")
        return mm, int(np.count_nonzero(mm.freq_hz <= rigid_tol_hz))

    if n_modes is None:
        raise ValueError(
            "n_modes is required when no free-interface basis is supplied; it counts "
            "the lowest free normal modes to retain, rigid-body modes included"
        )
    n_modes = int(n_modes)
    if n_modes <= 0:
        raise ValueError(f"n_modes must be >= 1, got {n_modes}")
    ndof = int(K.shape[0])
    # One mode beyond the retained set: it is the cheapest way to prove that no
    # rigid-body mode was left out, and inertia relief is silently wrong if one was.
    w2, phi = _free_interface_modes(K, M, min(n_modes + 1, ndof), sparse)
    freq = np.sqrt(w2) / TWO_PI
    n_rigid = int(np.count_nonzero(freq[:n_modes] <= rigid_tol_hz))
    if freq.size > n_modes and freq[n_modes] <= rigid_tol_hz:
        raise ValueError(
            f"n_modes={n_modes} truncates the rigid-body set of this component "
            f"(mode {n_modes + 1} is still at {freq[n_modes]:.3e} Hz); residual "
            "flexibility needs every rigid-body mode to inertia-relieve the loads"
        )
    kept = ModalModel(
        freq_hz=freq[:n_modes],
        modes=phi[:, :n_modes],
        generalized_mass=np.ones(n_modes),
        eigenvalues=w2[:n_modes],
        meta={"source": "cms_free.free_interface_modes"},
    )
    return kept, n_rigid


def _significant_residuals(
    kept: ModalModel,
    res: Any,
    b: np.ndarray,
    rigid_tol_hz: float,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the residual set when it carries no interface flexibility.

    :func:`~femtools.dynamics.residuals.residual_vectors` mass-normalises whatever is
    left of the static response, and that normalisation destroys the evidence of how much
    was left: with a complete modal basis it returns a perfectly respectable unit-mass
    vector assembled from round-off. The decision therefore has to be taken on the *raw*
    residual flexibility, by comparing the interface compliance it adds with the
    compliance the retained modes already provide. Rank deficiency *within* a non-trivial
    residual set is a different question, and ``residual_vectors`` has already answered it.
    """
    psi = res.vectors
    lam = np.asarray(res.eigenvalues, dtype=float)
    if psi.shape[1] == 0:
        return psi, lam

    elastic = np.flatnonzero(kept.freq_hz > rigid_tol_hz)
    w2 = np.asarray(kept.eigenvalues, dtype=float)[elastic]
    phi_b = kept.modes[np.ix_(b, elastic)]
    modal_c = float(np.linalg.norm((phi_b / w2[None, :]) @ phi_b.T)) if w2.size else 0.0
    residual_c = float(np.linalg.norm(res.residual_flexibility[b, :]))
    total = modal_c + residual_c
    if total <= 0.0 or residual_c <= tol * total:
        return psi[:, :0], lam[:0]
    return psi, lam


def _free_cms(
    K: Any,
    M: Any,
    boundary_dofs: Any,
    n_modes: int | None,
    *,
    method: str,
    modal: Any,
    interior_dofs: Any,
    tol: float,
    rigid_tol_hz: float,
) -> FreeCMSResult:
    """Shared MacNeal/Rubin kernel; ``method`` decides whether residual inertia is kept."""
    if K.shape != M.shape or K.shape[0] != K.shape[1]:
        raise ValueError("K and M must be square and equally sized")
    sparse = is_sparse(K) or is_sparse(M)
    ndof = int(K.shape[0])

    b = np.unique(resolve_dofs(boundary_dofs, ndof, "boundary_dofs"))
    if b.size == 0:
        raise ValueError("at least one interface DOF is required")
    if interior_dofs is None:
        i = np.setdiff1d(np.arange(ndof, dtype=int), b)
    else:
        i = np.unique(resolve_dofs(interior_dofs, ndof, "interior_dofs"))
        if np.intersect1d(b, i).size:
            raise ValueError("boundary_dofs and interior_dofs must be disjoint")

    kept, n_rigid = _kept_basis(K, M, n_modes, modal, sparse, rigid_tol_hz)
    res = residual_vectors(K, M, kept, b, tol=tol, rigid_tol_hz=rigid_tol_hz)
    psi, lam_res = _significant_residuals(kept, res, b, rigid_tol_hz, tol)

    T = np.hstack([kept.modes, psi])
    n_k, n_r = kept.n_modes, int(psi.shape[1])

    Kr = symmetrize(as_dense(T.T @ (K @ T)))
    Mr = symmetrize(as_dense(T.T @ (M @ T)))
    # The residual modes are M- and K-orthogonal to the kept modes analytically; report
    # how well that survived the numerics instead of assuming it.
    coupling = 0.0
    if n_k and n_r:
        scale = max(float(np.abs(np.diag(Mr)).max()), np.finfo(float).tiny)
        coupling = float(np.abs(Mr[:n_k, n_k:]).max() / scale)
    if method == "macneal":
        Mr[n_k:, :] = 0.0
        Mr[:, n_k:] = 0.0

    return FreeCMSResult(
        K=Kr,
        M=Mr,
        T=T,
        boundary_dofs=b,
        interior_dofs=i,
        free_freq_hz=kept.freq_hz.copy(),
        residual_freq_hz=np.sqrt(np.clip(lam_res, 0.0, None)) / TWO_PI,
        residual_flexibility=res.residual_flexibility,
        method=method,
        meta={
            "ndof": ndof,
            "sparse": bool(sparse),
            "n_rigid": n_rigid,
            "n_kept": n_k,
            "n_residual": n_r,
            "n_residual_dropped": int(res.meta.get("dropped", 0)) + res.n_res - n_r,
            "residual_inertia": method == "rubin",
            "mass_coupling": coupling,
        },
    )


def rubin(
    K: Any,
    M: Any,
    boundary_dofs: Any,
    n_modes: int | None = None,
    *,
    modal: Any = None,
    interior_dofs: Any = None,
    tol: float = 1e-10,
    rigid_tol_hz: float = 1e-4,
) -> FreeCMSResult:
    """Reduce ``(K, M)`` on free-interface modes plus residual flexibility *with* inertia.

    Rubin's method keeps the mass of the residual coordinates, so the deleted-mode
    receptance is matched to second order in ``omega`` rather than statically. The
    reduced model is ``K = diag(Lambda_k, Lambda_res)``, ``M = I``.

    Parameters
    ----------
    K, M:
        Full stiffness and mass matrices, dense or sparse, shape ``(ndof, ndof)``. A
        free-free component (singular ``K``) is expected and handled.
    boundary_dofs:
        Interface DOFs, i.e. the load directions the residual flexibility is computed
        for. Include any DOF the component will be excited at, not only the ties.
    n_modes:
        Number of lowest free-interface normal modes to retain, rigid-body modes
        included. Required unless ``modal`` is given, in which case it truncates it.
    modal:
        Pre-computed free-interface basis (anything
        :func:`~femtools.dynamics.modal.as_modal` accepts). Saves the eigen-solve and
        lets measured modes be used, but it is then the caller's job to include every
        rigid-body mode.
    interior_dofs:
        Interior set; defaults to the complement of ``boundary_dofs``.
    tol:
        Relative threshold below which a residual direction is dropped as dependent.
    rigid_tol_hz:
        Modes below this frequency count as rigid-body modes.

    Returns
    -------
    FreeCMSResult
    """
    return _free_cms(
        K,
        M,
        boundary_dofs,
        n_modes,
        method="rubin",
        modal=modal,
        interior_dofs=interior_dofs,
        tol=tol,
        rigid_tol_hz=rigid_tol_hz,
    )


def macneal(
    K: Any,
    M: Any,
    boundary_dofs: Any,
    n_modes: int | None = None,
    *,
    modal: Any = None,
    interior_dofs: Any = None,
    tol: float = 1e-10,
    rigid_tol_hz: float = 1e-4,
) -> FreeCMSResult:
    """Reduce ``(K, M)`` on free-interface modes plus *massless* residual flexibility.

    MacNeal's hybrid method neglects the inertia of the residual coordinates, so they
    respond statically and add exactly the first-order residual flexibility to the
    truncated modal model. The reduced mass matrix is therefore singular in the residual
    block; :meth:`FreeCMSResult.solve_modes` and :func:`free_interface_assembly` condense
    those coordinates. Arguments are as in :func:`rubin`.

    Returns
    -------
    FreeCMSResult
    """
    return _free_cms(
        K,
        M,
        boundary_dofs,
        n_modes,
        method="macneal",
        modal=modal,
        interior_dofs=interior_dofs,
        tol=tol,
        rigid_tol_hz=rigid_tol_hz,
    )


# ---------------------------------------------------------------------------
# assembly of reduced free-interface components
# ---------------------------------------------------------------------------
@dataclass
class FreeCMSComponent:
    """One reduced component in an assembly, with its slice of the global spaces."""

    component: FreeCMSResult
    name: str = ""
    dof_offset: int = 0
    coord_offset: int = 0

    @property
    def ndof(self) -> int:
        """Number of physical DOFs of this component."""
        return self.component.ndof

    @property
    def n_reduced(self) -> int:
        """Number of generalised coordinates contributed by this component."""
        return self.component.n_reduced

    def global_dof(self, local_dof: int) -> int:
        """Map a local physical DOF index to the assembled global physical index."""
        d = int(local_dof)
        if d < 0:
            d += self.ndof
        if not 0 <= d < self.ndof:
            raise IndexError(f"DOF {local_dof} out of range for component {self.name!r}")
        return self.dof_offset + d


@dataclass
class FreeInterfaceAssembly(ModalModel):
    """Modes of components coupled through their physical interface DOFs.

    :attr:`modes` lives in the *stacked physical* DOF space: component ``c`` occupies
    ``dof_offset[c] : dof_offset[c] + ndof[c]``, exactly as in
    :class:`~femtools.dynamics.mba.AssemblyResult`. Interface DOFs consequently appear
    once per component and carry the same motion, which is a free consistency check.
    """

    components: tuple[FreeCMSComponent, ...] = ()
    constraint_matrix: np.ndarray | None = None
    generalized_modes: np.ndarray | None = None

    def component_names(self) -> tuple[str, ...]:
        """Names of the assembled components, in DOF order."""
        return tuple(c.name for c in self.components)

    def component_modes(self, which: int | str) -> np.ndarray:
        """Rows of :attr:`modes` belonging to one component, shape ``(ndof_c, n_modes)``."""
        c = self._find(which)
        return self.modes[c.dof_offset : c.dof_offset + c.ndof, :]

    def global_dof(self, which: int | str, local_dof: int) -> int:
        """Global (stacked) physical DOF index of ``local_dof`` inside component ``which``."""
        return self._find(which).global_dof(local_dof)

    def _find(self, which: int | str) -> FreeCMSComponent:
        if isinstance(which, int):
            return self.components[which]
        for c in self.components:
            if c.name == which:
                return c
        raise KeyError(f"no component named {which!r}")


def _prepare_components(components: Sequence[Any]) -> list[FreeCMSComponent]:
    comps: list[FreeCMSComponent] = []
    dof_offset = 0
    coord_offset = 0
    for idx, item in enumerate(components):
        if isinstance(item, FreeCMSComponent):
            comp = FreeCMSComponent(component=item.component, name=item.name or f"C{idx}")
        elif isinstance(item, tuple | list) and len(item) == 2 and isinstance(item[0], str):
            comp = FreeCMSComponent(component=item[1], name=item[0])
        else:
            comp = FreeCMSComponent(component=item, name=f"C{idx}")
        if not isinstance(comp.component, FreeCMSResult):
            raise TypeError(
                "components must be FreeCMSResult objects (from rubin/macneal), got "
                f"{type(comp.component).__name__}"
            )
        comp.dof_offset = dof_offset
        comp.coord_offset = coord_offset
        dof_offset += comp.ndof
        coord_offset += comp.n_reduced
        comps.append(comp)
    if not comps:
        raise ValueError("at least one component is required")
    return comps


def _resolve_endpoint(comps: list[FreeCMSComponent], spec: Any) -> int:
    """Resolve ``(component, local_dof)`` (or a plain stacked index) to a physical DOF."""
    if isinstance(spec, tuple | list) and len(spec) == 2:
        which, dof = spec
        if isinstance(which, str):
            match = [c for c in comps if c.name == which]
            if not match:
                raise KeyError(f"no component named {which!r}")
            return match[0].global_dof(int(dof))
        return comps[int(which)].global_dof(int(dof))
    return int(spec)


def _parse_connections(
    comps: list[FreeCMSComponent], connections: Sequence[Any]
) -> tuple[list[tuple[int, int]], list[tuple[int, int, float]]]:
    """Split connections into rigid physical DOF pairs and elastic ties."""
    rigid: list[tuple[int, int]] = []
    springs: list[tuple[int, int, float]] = []
    for spec in connections:
        if isinstance(spec, Mapping):
            d = dict(spec)
            a = _resolve_endpoint(comps, d.pop("a"))
            b = _resolve_endpoint(comps, d.pop("b"))
            k = d.pop("k", None)
            if d:
                raise ValueError(f"unknown connection keys: {sorted(d)}")
        else:
            seq = tuple(spec)
            if len(seq) in (4, 5):
                a = _resolve_endpoint(comps, (seq[0], seq[1]))
                b = _resolve_endpoint(comps, (seq[2], seq[3]))
                k = seq[4] if len(seq) == 5 else None
            elif len(seq) == 2:
                a = _resolve_endpoint(comps, seq[0])
                b = _resolve_endpoint(comps, seq[1])
                k = None
            else:
                raise ValueError(f"cannot interpret connection spec {spec!r}")
        if k is None:
            rigid.append((a, b))
        else:
            springs.append((a, b, float(k)))
    return rigid, springs


def free_interface_assembly(
    components: Sequence[Any],
    connections: Sequence[Any] = (),
    *,
    n_modes: int | None = None,
    mass_tol: float = MASS_TOL,
) -> FreeInterfaceAssembly:
    """Couple reduced free-interface components into one modal model.

    Each component contributes its generalised coordinates ``[q_k, q_res]``; the coupling
    is written on the *physical* DOFs those coordinates expand to (``T``). Rigid ties are
    eliminated exactly through the null space of the compatibility matrix, elastic ties
    enter as rank-one stiffness terms, and coordinates left massless by
    :func:`macneal` are condensed statically before the eigen-solve.

    Parameters
    ----------
    components:
        :class:`FreeCMSResult` objects, ``(name, result)`` pairs or
        :class:`FreeCMSComponent` instances.
    connections:
        Interface ties. ``(ci, dof_i, cj, dof_j)`` is a rigid link between physical DOF
        ``dof_i`` of component ``ci`` and ``dof_j`` of ``cj``;
        ``(ci, dof_i, cj, dof_j, k)`` is an elastic link of stiffness ``k``. Mappings
        ``{"a": (ci, dof), "b": (cj, dof), "k": ...}`` and plain stacked-DOF pairs also
        work.
    n_modes:
        Keep only the lowest ``n_modes`` assembled modes.
    mass_tol:
        Relative threshold below which an assembled coordinate counts as massless.

    Returns
    -------
    FreeInterfaceAssembly
        A modal model over the stacked physical DOFs of the components.
    """
    comps = _prepare_components(components)
    n_phys = sum(c.ndof for c in comps)
    n_gen = sum(c.n_reduced for c in comps)

    Kg = np.zeros((n_gen, n_gen))
    Mg = np.zeros((n_gen, n_gen))
    Tg = np.zeros((n_phys, n_gen))
    for c in comps:
        g = slice(c.coord_offset, c.coord_offset + c.n_reduced)
        Kg[g, g] = c.component.K
        Mg[g, g] = c.component.M
        Tg[c.dof_offset : c.dof_offset + c.ndof, g] = c.component.T

    rigid, springs = _parse_connections(comps, connections)
    for a, b, k in springs:
        w = Tg[a, :] - Tg[b, :]
        Kg = Kg + k * np.outer(w, w)

    B: np.ndarray | None = None
    Z = np.eye(n_gen)
    if rigid:
        B = np.zeros((len(rigid), n_phys))
        for row, (a, b) in enumerate(rigid):
            B[row, a] += 1.0
            B[row, b] -= 1.0
        Z = _nullspace(B @ Tg)
        if Z.shape[1] == 0:
            raise ValueError(
                "the rigid connections leave no free coordinates; the component bases "
                "are too coarse to satisfy compatibility"
            )

    Kz = symmetrize(Z.T @ Kg @ Z)
    Mz = symmetrize(Z.T @ Mg @ Z)
    lam, Y = _semidefinite_modes(Kz, Mz, mass_tol)
    if n_modes is not None:
        lam, Y = lam[: int(n_modes)], Y[:, : int(n_modes)]

    Q = Z @ Y
    return FreeInterfaceAssembly(
        freq_hz=np.sqrt(lam) / TWO_PI,
        modes=Tg @ Q,
        generalized_mass=np.ones(lam.size),
        eigenvalues=lam,
        meta={
            "method": "free_interface_cms",
            "methods": tuple(c.component.method for c in comps),
            "n_components": len(comps),
            "n_rigid_links": 0 if B is None else int(B.shape[0]),
            "n_springs": len(springs),
            "n_generalized": n_gen,
            "n_free": int(Z.shape[1]),
        },
        components=tuple(comps),
        constraint_matrix=B,
        generalized_modes=Q,
    )
