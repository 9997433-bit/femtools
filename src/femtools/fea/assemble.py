"""Global assembly of the stiffness, mass and damping matrices."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .dofmap import DofMap
from .elements import ModelIndex, element_matrices, element_spec
from .nodal_frames import NodalFrames, shell_nodal_frames
from .protocols import get_any, iter_records, spc_entries

__all__ = ["AssemblyResult", "assemble_km"]


@dataclass
class AssemblyResult:
    """Assembled system matrices and the DOF partition used to solve them.

    ``free_dof`` are the equations actually solved: everything that is not
    single-point constrained, not empty (no stiffness *and* no mass) and not a
    purely fictitious drilling rotation.  ``unconstrained_dof`` keeps the plain
    "not SPC'd" set for callers that need it.

    Analysis frame
    --------------

    Matrices and full-length vectors are expressed in the *analysis* frame: the
    basic (global) frame everywhere except at the rotations of the shell nodes
    listed in :attr:`frames`, which live in a per-node triad whose third axis is
    the averaged shell normal (:mod:`femtools.fea.nodal_frames`).  That is what
    lets the drilling rotation of an arbitrarily oriented plate be removed as a
    single DOF.  Translations are never rotated, and the frame is the identity
    unless a shell normal is oblique, so for every axis-aligned model the two
    frames are the same object.  :meth:`to_basic` and :meth:`from_basic` convert
    displacement *or* force vectors between the two.
    """

    K: sp.csr_matrix
    M: sp.csr_matrix
    C: sp.csr_matrix
    dof_map: DofMap
    free_dof: np.ndarray
    spc_dof: np.ndarray
    null_dof: np.ndarray
    drilling_dof: np.ndarray
    spc_values: np.ndarray
    n_dof: int
    dofs_per_node: int = 6
    node_ids: list[Any] = field(default_factory=list)
    element_ids: list[Any] = field(default_factory=list)
    skipped_elements: dict[Any, str] = field(default_factory=dict)
    frames: NodalFrames | None = None

    # -- convenience --------------------------------------------------
    @property
    def ndof(self) -> int:
        return self.n_dof

    @property
    def n_free(self) -> int:
        return int(self.free_dof.size)

    @property
    def fixed_dof(self) -> np.ndarray:
        """All DOFs removed from the solution set (SPC + empty + drilling)."""
        mask = np.ones(self.n_dof, dtype=bool)
        mask[self.free_dof] = False
        return np.flatnonzero(mask)

    @property
    def unconstrained_dof(self) -> np.ndarray:
        mask = np.ones(self.n_dof, dtype=bool)
        mask[self.spc_dof] = False
        return np.flatnonzero(mask)

    def reduce(self, matrix: sp.spmatrix) -> sp.csr_matrix:
        """Extract the free-free partition of a global matrix."""
        f = self.free_dof
        return sp.csr_matrix(matrix.tocsr()[f][:, f])

    @property
    def Kff(self) -> sp.csr_matrix:
        return self.reduce(self.K)

    @property
    def Mff(self) -> sp.csr_matrix:
        return self.reduce(self.M)

    @property
    def Cff(self) -> sp.csr_matrix:
        return self.reduce(self.C)

    def expand(self, u_free: np.ndarray, *, include_spc: bool = True) -> np.ndarray:
        """Scatter a free-set vector (or set of columns) into full DOF space."""
        u_free = np.asarray(u_free)
        if u_free.ndim == 1:
            out = np.zeros(self.n_dof, dtype=u_free.dtype)
            out[self.free_dof] = u_free
            if include_spc and self.spc_dof.size:
                out[self.spc_dof] = self.spc_values[self.spc_dof].astype(out.dtype, copy=False)
            return out
        out = np.zeros((self.n_dof, u_free.shape[1]), dtype=u_free.dtype)
        out[self.free_dof, :] = u_free
        return out

    def restrict(self, u_full: np.ndarray) -> np.ndarray:
        u_full = np.asarray(u_full)
        return u_full[self.free_dof] if u_full.ndim == 1 else u_full[self.free_dof, :]

    # -- analysis frame <-> basic frame --------------------------------
    def to_basic(self, vector: np.ndarray) -> np.ndarray:
        """Rotate a full-length vector out of the analysis frame.

        Works for displacements and for forces alike: the transformation is
        orthogonal, so both share it.  A no-op for a model without oblique
        shell nodes.
        """
        return vector if self.frames is None else self.frames.to_basic(vector)

    def from_basic(self, vector: np.ndarray) -> np.ndarray:
        """Rotate a full-length basic-frame vector into the analysis frame."""
        return vector if self.frames is None else self.frames.from_basic(vector)

    @property
    def framed_nodes(self) -> list[Any]:
        """Nodes whose rotations are solved in a local triad."""
        return [] if self.frames is None else self.frames.framed_nodes

    def __iter__(self):
        """Allow ``K, M, C = assemble_km(model)``."""
        return iter((self.K, self.M, self.C))

    def __getitem__(self, index: int):
        return (self.K, self.M, self.C)[index]

    def summary(self) -> str:  # pragma: no cover - reporting helper
        framed = 0 if self.frames is None else self.frames.n_framed
        return (
            f"AssemblyResult(n_dof={self.n_dof}, free={self.n_free}, "
            f"spc={self.spc_dof.size}, empty={self.null_dof.size}, "
            f"drilling={self.drilling_dof.size}, framed_nodes={framed}, "
            f"elements={len(self.element_ids)})"
        )


class _Triplets:
    __slots__ = ("rows", "cols", "vals")

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []

    def add(self, gdof: np.ndarray, block: np.ndarray) -> None:
        n = gdof.size
        rr = np.repeat(gdof, n)
        cc = np.tile(gdof, n)
        vv = np.asarray(block, dtype=float).ravel()
        nz = vv != 0.0
        if not nz.any():
            return
        self.rows.append(rr[nz])
        self.cols.append(cc[nz])
        self.vals.append(vv[nz])

    def build(self, n: int) -> sp.csr_matrix:
        if not self.vals:
            return sp.csr_matrix((n, n))
        mat = sp.coo_matrix(
            (
                np.concatenate(self.vals),
                (np.concatenate(self.rows), np.concatenate(self.cols)),
            ),
            shape=(n, n),
        ).tocsr()
        mat.sum_duplicates()
        return mat


def _row_norms(matrix: sp.csr_matrix) -> np.ndarray:
    return np.asarray(abs(matrix).sum(axis=1)).ravel()


def _retained_drilling_mechanism(
    K: sp.csr_matrix,  # noqa: N803
    K_drill: sp.csr_matrix,  # noqa: N803
    free_mask: np.ndarray,
    dofs_per_node: int,
    k_scale: float,
) -> bool:
    """Is a fictitious drilling rotation still a zero-energy mode of the free set?

    The drilling penalty is rank deficient on purpose: rotating a whole element
    about its own normal has to stay free, or the rigid body modes would be
    lost.  On a *flat* mesh those per-element null spaces line up into one
    global mechanism, which is what the elimination above exists to remove.
    Since the assembly is expressed in the per-node rotational frames of
    :mod:`femtools.fea.nodal_frames`, the drilling direction of a flat patch is
    local component 5 whatever the orientation of the plate, so the elimination
    reaches every case it used to miss.

    What is left for this check are the meshes where no single DOF *is* the
    mechanism: a node whose rotations had to stay in the basic frame because
    they are single point constrained there, or a user assembling with
    ``nodal_frames=False``.  Rebuild the candidate mechanism -- every drilling
    node rotating about the common normal -- and measure its strain energy, so
    such a case is reported instead of being returned as a zero frequency.  A
    folded or curved shell has no common normal, the candidate is not a null
    vector and the test stays quiet.
    """
    if dofs_per_node < 6:
        return False
    rotations = np.arange(K.shape[0]).reshape(-1, dofs_per_node)[:, 3:6]
    weight = np.abs(K_drill.diagonal())[rotations].sum(axis=1)
    if not weight.any():
        return False

    # Every nodal block of the penalty is a multiple of ``n n^T``, so the
    # dominant eigenvector of the busiest one is the drilling direction.
    lead = rotations[int(np.argmax(weight))]
    block = K_drill[lead, :][:, lead].toarray()
    normal = np.linalg.eigh(0.5 * (block + block.T))[1][:, -1]

    v = np.zeros(K.shape[0])
    active = rotations[weight > 0.0]
    v[active.ravel()] = np.tile(normal, active.shape[0])
    v[~free_mask] = 0.0
    norm = float(v @ v)
    if norm == 0.0:
        return False
    return abs(float(v @ (K @ v))) <= 1.0e-12 * k_scale * norm


def assemble_km(
    model: Any,
    *,
    dofs_per_node: int = 6,
    apply_spc: bool = True,
    remove_null_dofs: bool = True,
    suppress_drilling: bool = True,
    nodal_frames: bool = True,
    lumped_mass: bool = False,
    drill_factor: float = 1.0e-3,
    rayleigh: tuple[float, float] | None = None,
    element_filter: Callable[[Any, Any], bool] | Iterable[Any] | None = None,
    on_unknown: str = "raise",
    options: dict[str, Any] | None = None,
) -> AssemblyResult:
    """Assemble ``K``, ``M`` and ``C`` for *model*.

    Parameters
    ----------
    model
        Anything satisfying :class:`~femtools.fea.protocols.ModelLike`.
    dofs_per_node
        Six for the standard 3D structural model.
    apply_spc
        Eliminate DOFs listed in ``model.spcs`` from the free set.
    remove_null_dofs
        Drop DOFs that receive neither stiffness nor mass (rotations of a truss
        or solid mesh, unreferenced grid points).  Keeps ``Kff`` non-singular.
    suppress_drilling
        Drop shell drilling rotations whose *only* stiffness is the fictitious
        drilling penalty; this removes the spurious mechanism of a flat mesh.
    nodal_frames
        Solve the rotations of a shell node in a local triad whose third axis
        is the averaged shell normal (:mod:`femtools.fea.nodal_frames`), which
        is what makes the drilling elimination above work for a plate at any
        orientation rather than only for one lying in a global plane.  The
        triad is the identity wherever the normal already is a global axis, so
        turning this off only changes an oblique model.
    lumped_mass
        Use diagonal element mass matrices instead of consistent ones.
    rayleigh
        ``(alpha, beta)`` for ``C += alpha*M + beta*K`` on top of the assembled
        ``DAMPER`` elements.
    element_filter
        Callable ``(element_id, element) -> bool`` or an explicit collection of
        element ids to include.
    on_unknown
        ``"raise"`` (default), ``"skip"`` or ``"warn"`` for unregistered types.
    options
        Solver-wide element options, passed through to every element builder.
        Currently read by ``HEX8``, which accepts ``{"hex8": name}`` with
        *name* one of :data:`femtools.fea.HEX8_FORMULATIONS` (default
        ``"incompatible"``).  A ``formulation`` field on the element or on its
        property overrides the assembly-wide setting.

    Returns
    -------
    AssemblyResult
    """
    index = ModelIndex.build(model)
    if not index.nodes:
        raise ValueError("model has no nodes")
    dof_map = DofMap.from_nodes(index.nodes, dofs_per_node)
    n = dof_map.n_dof

    keep: Callable[[Any, Any], bool]
    if element_filter is None:
        keep = lambda _eid, _el: True  # noqa: E731
    elif callable(element_filter):
        keep = element_filter
    else:
        wanted = set(element_filter)
        keep = lambda eid, _el: eid in wanted  # noqa: E731

    tk, tm, tc, td = _Triplets(), _Triplets(), _Triplets(), _Triplets()
    element_ids: list[Any] = []
    skipped: dict[Any, str] = {}

    for eid, element in iter_records(get_any(model, ("elements", "elems", "element"), None)):
        if element is None or not keep(eid, element):
            continue
        etype = str(get_any(element, ("type", "etype", "element_type", "kind"), "")).upper()
        try:
            element_spec(etype)
        except KeyError as exc:
            if on_unknown == "skip":
                skipped[eid] = str(exc)
                continue
            if on_unknown == "warn":
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
                skipped[eid] = str(exc)
                continue
            raise
        em = element_matrices(
            model,
            eid,
            element,
            lumped_mass=lumped_mass,
            drill_factor=drill_factor,
            options=options,
            index=index,
        )
        gdof = np.fromiter(
            (dof_map.index(nid, comp) for nid, comp in em.dofs), dtype=int, count=len(em.dofs)
        )
        if em.k is not None:
            tk.add(gdof, em.k)
        if em.m is not None:
            tm.add(gdof, em.m)
        if em.c is not None:
            tc.add(gdof, em.c)
        if em.k_drill is not None:
            td.add(gdof, em.k_drill)
        element_ids.append(eid)

    K = tk.build(n)
    M = tm.build(n)
    C = tc.build(n)
    K_drill = td.build(n)

    # Symmetrise: element matrices are symmetric, this only removes round-off.
    K = ((K + K.T) * 0.5).tocsr()
    M = ((M + M.T) * 0.5).tocsr()
    C = ((C + C.T) * 0.5).tocsr()

    if rayleigh is not None:
        alpha, beta = (float(rayleigh[0]), float(rayleigh[1]))
        C = (C + alpha * M + beta * K).tocsr()

    # -- DOF partition -------------------------------------------------
    spc_values = np.zeros(n, dtype=float)
    spc_mask = np.zeros(n, dtype=bool)
    if apply_spc:
        for node_id, comp, value in spc_entries(model, dofs_per_node=dofs_per_node):
            try:
                spc_index = dof_map.index(node_id, comp)
            except KeyError:
                continue
            spc_mask[spc_index] = True
            spc_values[spc_index] = value

    # -- per-node rotational frames --------------------------------------
    # A rotational SPC is written in the basic frame and only remains a single
    # DOF constraint there, so those nodes keep the basic triad; everything
    # else follows its averaged shell normal.
    frames = NodalFrames(dof_map=dof_map)
    if nodal_frames and dofs_per_node >= 6:
        constrained_rotations = {
            dof_map.dof_node(int(d))
            for d in np.flatnonzero(spc_mask)
            if int(d) % dofs_per_node >= 3
        }
        frames = shell_nodal_frames(model, dof_map, index=index, skip=constrained_rotations)
    if not frames.is_identity:
        K = frames.congruence(K)
        M = frames.congruence(M)
        C = frames.congruence(C)
        K_drill = frames.congruence(K_drill)

    k_rows = _row_norms(K)
    m_rows = _row_norms(M)
    c_rows = _row_norms(C)
    k_scale = k_rows.max() if k_rows.size else 0.0
    m_scale = m_rows.max() if m_rows.size else 0.0
    c_scale = c_rows.max() if c_rows.size else 0.0

    null_mask = np.zeros(n, dtype=bool)
    if remove_null_dofs:
        null_mask = (
            (k_rows <= 1.0e-12 * k_scale)
            & (m_rows <= 1.0e-12 * m_scale)
            & (c_rows <= 1.0e-12 * c_scale if c_scale > 0 else np.ones(n, dtype=bool))
        )
        null_mask &= ~spc_mask

    drill_mask = np.zeros(n, dtype=bool)
    if suppress_drilling and K_drill.nnz:
        drill_diag = np.abs(K_drill.diagonal())
        real_rows = _row_norms((K - K_drill).tocsr())
        drill_mask = (drill_diag > 0.0) & (real_rows <= 1.0e-10 * k_scale)
        drill_mask &= ~spc_mask & ~null_mask

    free_mask = ~(spc_mask | null_mask | drill_mask)
    if (
        suppress_drilling
        and K_drill.nnz
        and _retained_drilling_mechanism(K, K_drill, free_mask, dofs_per_node, k_scale)
    ):
        warnings.warn(
            "this flat shell mesh keeps a fictitious drilling mechanism: its drilling "
            "rotations could not be expressed as one degree of freedom each, so the "
            "assembly carries a spurious zero-energy mode (it surfaces as an extra zero "
            "frequency alongside the six rigid body modes). Assemble with "
            "nodal_frames=True, or constrain one drilling rotation.",
            RuntimeWarning,
            stacklevel=2,
        )
    return AssemblyResult(
        K=K,
        M=M,
        C=C,
        dof_map=dof_map,
        free_dof=np.flatnonzero(free_mask),
        spc_dof=np.flatnonzero(spc_mask),
        null_dof=np.flatnonzero(null_mask),
        drilling_dof=np.flatnonzero(drill_mask),
        spc_values=spc_values,
        n_dof=n,
        dofs_per_node=dofs_per_node,
        node_ids=dof_map.node_ids,
        element_ids=element_ids,
        skipped_elements=skipped,
        frames=frames,
    )
