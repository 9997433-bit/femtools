"""Resolve *what* a direct (physical) solve runs on.

The direct entry points of this package — :func:`~femtools.dynamics.frf.direct_frf`,
:func:`~femtools.dynamics.harmonic.harmonic_response` and
:func:`~femtools.dynamics.frf.verify_modal_vs_direct` — are defined on a pair of matrices
``(K, M)``. That is the right primitive: it keeps the package usable on hand-built systems,
on reduced models and on matrices that never came from a mesh. It is also inconvenient when
the matrices *do* come from a mesh, because the caller then has to assemble, pick the free
partition and translate ``(node, component)`` into a row of that partition before it can ask
for the FRF between two grid points.

:func:`as_system` closes that gap without changing the primitive. It accepts

* ``(K, M)`` — two matrices, exactly as before and with no import of :mod:`femtools.fea`;
* an ``AssemblyResult`` (or anything exposing ``Kff``/``Mff``) as the sole argument;
* a ``ModalResult`` carrying the assembly it was solved from;
* a model database (anything :func:`femtools.fea.assemble.assemble_km` accepts).

and returns a :class:`SystemMatrices` describing one square, solvable system. For the mesh
cases that system is the **free-free partition** ``Kff``/``Mff``: the global matrices have
empty rows wherever a DOF is single-point constrained or carries neither stiffness nor mass,
so ``Z(w)`` built from them is singular by construction and the FRF would be meaningless.
Row *i* of the returned system is therefore free DOF *i*, and :meth:`SystemMatrices.resolve`
is what turns ``(node, component)`` into that number.

:mod:`femtools.fea` is imported lazily, inside the branches that need it, so
:mod:`femtools.dynamics` still imports and runs on its own with nothing but numpy/scipy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from ._utils import is_sparse, resolve_dofs
from .modal import ModalModel, as_modal

__all__ = ["SystemMatrices", "as_system", "is_matrix_like", "resolve_selection"]


def is_matrix_like(obj: Any) -> bool:
    """True for something that is already a matrix rather than a model or assembly."""
    if obj is None:
        return False
    if sp.issparse(obj) or isinstance(obj, np.ndarray):
        return True
    if isinstance(obj, (list, tuple)):
        return True
    shape = getattr(obj, "shape", None)
    return isinstance(shape, tuple) and len(shape) == 2


def _as_matrix(obj: Any) -> Any:
    """Pass sparse matrices through, coerce anything else to a 2-D ndarray."""
    if sp.issparse(obj):
        return obj
    arr = obj if isinstance(obj, np.ndarray) else np.asarray(obj)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {arr.shape}")
    return arr


def _is_assembly_like(obj: Any) -> bool:
    return hasattr(obj, "Kff") and hasattr(obj, "Mff") and hasattr(obj, "free_dof")


def _is_model_like(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        return "nodes" in obj
    return hasattr(obj, "nodes") and hasattr(obj, "elements")


def _node_component_pairs(spec: Any) -> list[tuple[Any, Any]] | None:
    """Interpret ``spec`` as ``(node_id, component)`` pairs, or return ``None``.

    Two forms are recognised, both unambiguous against a plain list of row indices:

    * a sequence whose every entry is itself a length-2 sequence —
      ``[(7, "uz"), (4, 2)]``. A *flat* ``(7, 2)`` still means "rows 7 and 2", so a
      single pair has to be written as ``[(7, 2)]``;
    * a mapping ``{node_id: components}`` — ``{7: "uz", 4: [0, 2]}``.
    """
    if spec is None or isinstance(spec, (slice, str, np.ndarray)):
        return None
    if isinstance(spec, Mapping):
        from femtools.fea.protocols import normalize_dof_list

        pairs: list[tuple[Any, Any]] = []
        for node_id, comps in spec.items():
            for comp in normalize_dof_list(comps):
                pairs.append((node_id, comp))
        return pairs
    if isinstance(spec, (list, tuple)):
        items = list(spec)
        if items and all(
            isinstance(item, (list, tuple)) and len(item) == 2 for item in items
        ):
            return [(item[0], item[1]) for item in items]
    return None


@dataclass
class SystemMatrices:
    """One square physical system plus the bookkeeping needed to address its rows.

    Attributes
    ----------
    K, M:
        The matrices actually solved. For a mesh source these are the free-free
        partition, not the global matrices.
    C:
        Viscous damping that came *with* the source (``DAMPER`` elements, an assembly-time
        Rayleigh term), or ``None``. This is added on top of whatever the ``damping``
        argument produces; it is not the caller's explicit ``C``.
    dof_map, free_dof, assembly, model:
        Present only for a mesh source. ``free_dof`` holds the global DOF number of each
        row of ``K``.
    source:
        ``"matrices"``, ``"assembly"`` or ``"model"``.
    """

    K: Any
    M: Any
    C: Any = None
    dof_map: Any = None
    free_dof: np.ndarray | None = None
    assembly: Any = None
    model: Any = None
    source: str = "matrices"
    _inverse: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.K.shape != self.M.shape or self.K.shape[0] != self.K.shape[1]:
            raise ValueError(
                f"K and M must be square and equally sized, got {self.K.shape}, "
                f"{self.M.shape}"
            )
        if self.free_dof is not None:
            self.free_dof = np.asarray(self.free_dof, dtype=int).reshape(-1)
            if self.free_dof.size != self.ndof:
                raise ValueError(
                    f"free_dof has {self.free_dof.size} entries but the system has "
                    f"{self.ndof} rows"
                )

    # -- basics -----------------------------------------------------------
    @property
    def ndof(self) -> int:
        """Number of equations actually solved (rows of ``K``)."""
        return int(self.K.shape[0])

    @property
    def sparse(self) -> bool:
        """True when either matrix is scipy sparse."""
        return is_sparse(self.K) or is_sparse(self.M)

    @property
    def n_dof_global(self) -> int | None:
        """Size of the *unreduced* DOF space, or ``None`` without a DOF map."""
        return None if self.dof_map is None else int(self.dof_map.n_dof)

    @property
    def can_solve_modal(self) -> bool:
        """True when :meth:`solve_modal` can produce a basis from the source."""
        return self.assembly is not None

    # -- DOF addressing ---------------------------------------------------
    def _global_to_row(self) -> np.ndarray:
        """Lookup of global DOF number -> row of this system, ``-1`` where absent."""
        if self._inverse is None:
            n_global = self.n_dof_global
            if n_global is None or self.free_dof is None:
                raise ValueError("this system has no DOF map")
            inverse = np.full(n_global, -1, dtype=int)
            inverse[self.free_dof] = np.arange(self.free_dof.size)
            self._inverse = inverse
        return self._inverse

    def resolve(self, spec: Any, name: str = "dofs") -> np.ndarray:
        """Normalise a DOF selection to row indices of this system.

        Everything :func:`femtools.dynamics._utils.resolve_dofs` accepts still works and
        still means *rows of the solved system*. In addition, a mesh-backed system accepts
        ``(node_id, component)`` pairs and ``{node_id: components}`` mappings; components
        may be spelled as an index or as a label (``"uz"``, ``"rx"``, Nastran ``"3"``).
        """
        pairs = _node_component_pairs(spec)
        if pairs is None:
            return resolve_dofs(spec, self.ndof, name)
        if self.dof_map is None:
            raise ValueError(
                f"{name} was given as (node, component) pairs, but this system was built "
                "from plain matrices and has no DOF map; pass row indices instead"
            )
        inverse = self._global_to_row()
        rows = np.empty(len(pairs), dtype=int)
        for position, (node_id, component) in enumerate(pairs):
            try:
                global_dof = self.dof_map.index(node_id, component)
            except KeyError as exc:
                raise ValueError(f"{name}: node {node_id!r} is not in the model") from exc
            row = int(inverse[global_dof])
            if row < 0:
                raise ValueError(
                    f"{name}: DOF ({node_id!r}, {component!r}) is constrained or carries "
                    "no stiffness and mass, so it is not part of the solved system"
                )
            rows[position] = row
        return rows

    def global_dofs(self, rows: np.ndarray) -> np.ndarray | None:
        """Global DOF numbers of the given rows, or ``None`` without a DOF map."""
        if self.free_dof is None:
            return None
        return self.free_dof[np.asarray(rows, dtype=int)]

    def labels(self, rows: np.ndarray) -> list[tuple[Any, int]] | None:
        """``(node_id, component)`` of the given rows, or ``None`` without a DOF map."""
        global_dofs = self.global_dofs(rows)
        if global_dofs is None or self.dof_map is None:
            return None
        return [
            (self.dof_map.dof_node(dof), self.dof_map.dof_component(dof))
            for dof in global_dofs
        ]

    # -- modal basis ------------------------------------------------------
    def align_modal(self, modal: Any) -> ModalModel:
        """Coerce ``modal`` to a basis whose rows match the rows of this system.

        ``femtools.fea.eigen.solve_modes`` returns mode shapes over the *global* DOF space
        with zeros on the constrained DOFs, while a mesh-backed system solves the free
        partition. Such a basis is restricted here instead of failing on a shape mismatch.
        """
        mm = as_modal(modal)
        if mm.ndof == self.ndof:
            return mm
        if self.free_dof is not None and mm.ndof == self.n_dof_global:
            return ModalModel(
                freq_hz=mm.freq_hz.copy(),
                modes=mm.modes[self.free_dof, :],
                generalized_mass=np.asarray(mm.generalized_mass).copy(),
                eigenvalues=np.asarray(mm.eigenvalues).copy(),
                dof_ids=None if mm.dof_ids is None else mm.dof_ids[self.free_dof],
                meta=dict(mm.meta),
            )
        raise ValueError(
            f"modal basis has {mm.ndof} rows but the system has {self.ndof} equations"
            + (
                f" ({self.n_dof_global} before constraints)"
                if self.n_dof_global is not None
                else ""
            )
        )

    def solve_modal(self, n_modes: int | None = None) -> ModalModel:
        """Solve normal modes from the source assembly, aligned to this system.

        ``n_modes`` defaults to *every* free DOF, i.e. the complete basis, which is what
        makes a ``ModalDamping`` ``C`` and hence a modal-vs-direct comparison exact. That
        is an ``ndof``-sized eigenproblem; pass ``n_modes`` on a large model.
        """
        if self.assembly is None:
            raise ValueError(
                "this system was built from plain matrices; a modal basis cannot be "
                "solved from it, pass modal=... explicitly"
            )
        from femtools.fea.eigen import solve_modes

        count = self.ndof if n_modes is None else int(n_modes)
        solved = solve_modes(self.model, n_modes=count, assembly=self.assembly)
        return self.align_modal(solved)

    def modal_basis(self, modal: Any) -> ModalModel | None:
        """Resolve the ``modal`` argument of a direct solver.

        ``None`` stays ``None``; ``"auto"`` solves the complete basis from the source;
        anything else is coerced and aligned.
        """
        if modal is None:
            return None
        if isinstance(modal, str):
            if modal.lower() not in ("auto", "full"):
                raise ValueError(f"unknown modal spec {modal!r}; expected 'auto' or a basis")
            return self.solve_modal()
        return self.align_modal(modal)

    # -- reporting --------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        """Provenance to attach to a result object."""
        info: dict[str, Any] = {"ndof": self.ndof, "sparse": self.sparse, "source": self.source}
        if self.assembly is not None:
            info["assembly"] = self.assembly
            info["dof_map"] = self.dof_map
            info["free_dof"] = self.free_dof
        return info


def _from_assembly(assembly: Any, model: Any, source: str) -> SystemMatrices:
    C = getattr(assembly, "Cff", None)
    if C is not None and getattr(C, "nnz", 1) == 0:
        C = None
    return SystemMatrices(
        K=assembly.Kff,
        M=assembly.Mff,
        C=C,
        dof_map=getattr(assembly, "dof_map", None),
        free_dof=np.asarray(assembly.free_dof, dtype=int),
        assembly=assembly,
        model=model,
        source=source,
    )


def as_system(
    K: Any,
    M: Any = None,
    *,
    assemble: Mapping[str, Any] | None = None,
) -> SystemMatrices:
    """Coerce a ``(K, M)`` pair, an assembly or a model into a :class:`SystemMatrices`.

    Parameters
    ----------
    K:
        Stiffness matrix, an ``AssemblyResult``, a ``ModalResult`` that carries one, a
        model database, or an existing :class:`SystemMatrices`.
    M:
        Mass matrix. Required when — and only when — ``K`` is a matrix.
    assemble:
        Keyword arguments forwarded to :func:`femtools.fea.assemble.assemble_km` when a
        model has to be assembled here (``lumped_mass``, ``rayleigh``, ``apply_spc`` ...).

    Notes
    -----
    Every mesh-backed branch returns the free-free partition; see the module docstring.
    """
    if isinstance(K, SystemMatrices):
        if M is not None:
            raise ValueError("M must be omitted when the first argument is a SystemMatrices")
        return K

    if M is not None:
        if not is_matrix_like(K):
            raise TypeError(
                f"M was supplied, so K must be a matrix, got {type(K).__name__}"
            )
        return SystemMatrices(K=_as_matrix(K), M=_as_matrix(M), source="matrices")

    if K is None:
        raise TypeError("expected matrices, an assembly or a model, got None")

    if _is_assembly_like(K):
        return _from_assembly(K, getattr(K, "model", None), "assembly")

    if hasattr(K, "assembly"):
        assembly = K.assembly
        if assembly is None:
            raise ValueError(
                f"{type(K).__name__} carries no assembly, so the physical matrices are "
                "not available from it; pass K and M, an AssemblyResult or the model"
            )
        if _is_assembly_like(assembly):
            return _from_assembly(assembly, getattr(K, "model", None), "assembly")

    if _is_model_like(K):
        from femtools.fea.assemble import assemble_km

        built = assemble_km(K, **dict(assemble or {}))
        return _from_assembly(built, K, "model")

    if is_matrix_like(K):
        raise ValueError(
            "K is a matrix, so M is required; pass direct_frf(K, M, ...) or hand in an "
            "assembly / model as the single positional argument"
        )

    raise TypeError(
        "cannot interpret the first argument as a system; expected a (K, M) matrix pair, "
        "an AssemblyResult, a ModalResult carrying one, or a model database, got "
        f"{type(K).__name__}"
    )


def resolve_selection(system: SystemMatrices, spec: Any, name: str) -> Any:
    """Resolve a DOF selection when the system can address nodes, else pass it through.

    A plain-matrix system leaves the selection alone so that a modal basis keeps its own
    ``dof_ids`` label handling; a mesh-backed system resolves once, up front, which is what
    guarantees the modal and direct sides of a comparison look at the same DOFs.
    """
    if system.dof_map is None:
        return spec
    return system.resolve(spec, name)
