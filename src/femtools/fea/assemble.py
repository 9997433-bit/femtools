"""Global assembly of the stiffness, mass and damping matrices."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .dofmap import DofMap
from .elements import ModelIndex, element_matrices, element_spec
from .protocols import get_any, iter_records, spc_entries

__all__ = ["AssemblyResult", "assemble_km"]


@dataclass
class AssemblyResult:
    """Assembled system matrices and the DOF partition used to solve them.

    ``free_dof`` are the equations actually solved: everything that is not
    single-point constrained, not empty (no stiffness *and* no mass) and not a
    purely fictitious drilling rotation.  ``unconstrained_dof`` keeps the plain
    "not SPC'd" set for callers that need it.
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

    def __iter__(self):
        """Allow ``K, M, C = assemble_km(model)``."""
        return iter((self.K, self.M, self.C))

    def __getitem__(self, index: int):
        return (self.K, self.M, self.C)[index]

    def summary(self) -> str:  # pragma: no cover - reporting helper
        return (
            f"AssemblyResult(n_dof={self.n_dof}, free={self.n_free}, "
            f"spc={self.spc_dof.size}, empty={self.null_dof.size}, "
            f"drilling={self.drilling_dof.size}, elements={len(self.element_ids)})"
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


def assemble_km(
    model: Any,
    *,
    dofs_per_node: int = 6,
    apply_spc: bool = True,
    remove_null_dofs: bool = True,
    suppress_drilling: bool = True,
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
                import warnings

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
    )
