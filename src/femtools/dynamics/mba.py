"""Modal Based Assembly (MBA) and Structural Dynamic Modification (SDM).

SDM predicts how the modes of a structure change when discrete springs and masses are
added, working entirely in the modal space of the unmodified structure::

    (Lambda + Phi^T dK Phi) q = lambda (I + Phi^T dM Phi) q,   Phi_new = Phi q

MBA is the same machinery applied to several modal components at once: the components'
modal models are stacked block-diagonally in a global DOF space and then tied together,
either exactly (rigid links, eliminated through the null space of the compatibility
constraints) or elastically (connection springs, handled as an SDM modification).

Because the results subclass :class:`~femtools.dynamics.modal.ModalModel` they can be fed
straight into :func:`~femtools.dynamics.frf.modal_frf`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from ._utils import TWO_PI, as_dense, symmetrize
from .modal import ModalModel, as_modal

__all__ = [
    "AssemblyResult",
    "MassModification",
    "ModalComponent",
    "ModificationResult",
    "SpringModification",
    "modal_based_assembly",
    "structural_dynamic_modification",
]


@dataclass(frozen=True)
class SpringModification:
    """A discrete spring of stiffness ``k``.

    ``dof_j is None`` means the spring is grounded at ``dof_i``; otherwise it connects
    ``dof_i`` and ``dof_j`` and contributes ``k (e_i - e_j)(e_i - e_j)^T`` to ``dK``.
    """

    dof_i: int
    k: float
    dof_j: int | None = None


@dataclass(frozen=True)
class MassModification:
    """A lumped mass ``mass`` added at ``dof``, contributing ``mass e_d e_d^T`` to ``dM``."""

    dof: int
    mass: float


@dataclass
class ModalComponent:
    """One modal component in an assembly, with its slice of the global DOF space."""

    modal: ModalModel
    name: str = ""
    dof_offset: int = 0
    mode_offset: int = 0

    @property
    def ndof(self) -> int:
        """Number of physical DOFs of this component."""
        return self.modal.ndof

    @property
    def n_modes(self) -> int:
        """Number of modes contributed by this component."""
        return self.modal.n_modes

    def global_dof(self, local_dof: int) -> int:
        """Map a local DOF index to the assembled global DOF index."""
        d = int(local_dof)
        if d < 0:
            d += self.ndof
        if not 0 <= d < self.ndof:
            raise IndexError(f"DOF {local_dof} out of range for component {self.name!r}")
        return self.dof_offset + d


@dataclass
class ModificationResult(ModalModel):
    """Modes of a structure after an SDM spring/mass modification.

    Adds the baseline frequencies and the (sparse) modification matrices to the
    :class:`~femtools.dynamics.modal.ModalModel` interface.
    """

    baseline_freq_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dK: Any = None
    dM: Any = None

    @property
    def freq_shift_hz(self) -> np.ndarray:
        """``freq_hz - baseline_freq_hz`` over the overlapping modes."""
        n = min(self.freq_hz.size, self.baseline_freq_hz.size)
        return self.freq_hz[:n] - self.baseline_freq_hz[:n]


@dataclass
class AssemblyResult(ModalModel):
    """Modes of an assembled (coupled) modal model."""

    components: tuple[ModalComponent, ...] = ()
    constraint_matrix: np.ndarray | None = None

    def component_names(self) -> tuple[str, ...]:
        """Names of the assembled components, in DOF order."""
        return tuple(c.name for c in self.components)

    def component_modes(self, which: int | str) -> np.ndarray:
        """Rows of :attr:`modes` belonging to one component, shape ``(ndof_c, n_modes)``."""
        c = self._find(which)
        return self.modes[c.dof_offset : c.dof_offset + c.ndof, :]

    def global_dof(self, which: int | str, local_dof: int) -> int:
        """Global DOF index of ``local_dof`` inside component ``which``."""
        return self._find(which).global_dof(local_dof)

    def _find(self, which: int | str) -> ModalComponent:
        if isinstance(which, int):
            return self.components[which]
        for c in self.components:
            if c.name == which:
                return c
        raise KeyError(f"no component named {which!r}")


# ---------------------------------------------------------------------------
# specification parsing
# ---------------------------------------------------------------------------
def _as_spring(spec: Any) -> SpringModification:
    if isinstance(spec, SpringModification):
        return spec
    if isinstance(spec, Mapping):
        d = dict(spec)
        if "dofs" in d:
            i, j = d.pop("dofs")
        else:
            i, j = d.pop("dof_i", d.pop("dof", None)), d.pop("dof_j", None)
        k = d.pop("k", d.pop("stiffness", None))
        if i is None or k is None:
            raise ValueError(f"spring spec needs a DOF and a stiffness: {spec!r}")
        if d:
            raise ValueError(f"unknown spring keys: {sorted(d)}")
        return SpringModification(int(i), float(k), None if j is None else int(j))
    seq = tuple(spec)
    if len(seq) == 2:
        return SpringModification(int(seq[0]), float(seq[1]), None)
    if len(seq) == 3:
        return SpringModification(int(seq[0]), float(seq[2]), int(seq[1]))
    raise ValueError(f"cannot interpret spring spec {spec!r}")


def _as_mass(spec: Any) -> MassModification:
    if isinstance(spec, MassModification):
        return spec
    if isinstance(spec, Mapping):
        d = dict(spec)
        dof = d.pop("dof", None)
        m = d.pop("mass", d.pop("m", None))
        if dof is None or m is None:
            raise ValueError(f"mass spec needs a DOF and a mass: {spec!r}")
        if d:
            raise ValueError(f"unknown mass keys: {sorted(d)}")
        return MassModification(int(dof), float(m))
    seq = tuple(spec)
    if len(seq) != 2:
        raise ValueError(f"cannot interpret mass spec {spec!r}")
    return MassModification(int(seq[0]), float(seq[1]))


def _modal_projection(
    phi: np.ndarray,
    springs: Sequence[SpringModification],
    masses: Sequence[MassModification],
) -> tuple[np.ndarray, np.ndarray]:
    """Project discrete springs/masses onto the modal basis without forming ndof matrices."""
    n = phi.shape[1]
    dKm = np.zeros((n, n))
    dMm = np.zeros((n, n))
    for s in springs:
        v = phi[s.dof_i, :].copy()
        if s.dof_j is not None:
            v = v - phi[s.dof_j, :]
        dKm += s.k * np.outer(v, v)
    for m in masses:
        u = phi[m.dof, :]
        dMm += m.mass * np.outer(u, u)
    return dKm, dMm


def _physical_matrices(
    ndof: int,
    springs: Sequence[SpringModification],
    masses: Sequence[MassModification],
) -> tuple[Any, Any]:
    """Assemble the sparse physical ``dK`` and ``dM`` from discrete modifications."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for s in springs:
        if s.dof_j is None:
            rows.append(s.dof_i)
            cols.append(s.dof_i)
            vals.append(s.k)
        else:
            i, j = s.dof_i, s.dof_j
            rows += [i, j, i, j]
            cols += [i, j, j, i]
            vals += [s.k, s.k, -s.k, -s.k]
    dK = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    dM = sp.coo_matrix(
        ([m.mass for m in masses], ([m.dof for m in masses], [m.dof for m in masses])),
        shape=(ndof, ndof),
    ).tocsr()
    return dK, dM


def _solve_reduced(
    phi: np.ndarray,
    wr2: np.ndarray,
    dKm: np.ndarray,
    dMm: np.ndarray,
    Z: np.ndarray | None,
    n_modes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the (optionally constrained) modal eigenproblem; returns ``(lam, Phi_new)``."""
    A = symmetrize(np.diag(wr2) + dKm)
    B = symmetrize(np.eye(wr2.size) + dMm)
    basis = phi
    if Z is not None:
        A = symmetrize(Z.T @ A @ Z)
        B = symmetrize(Z.T @ B @ Z)
        basis = phi @ Z
    if A.shape[0] == 0:
        return np.zeros(0), np.zeros((phi.shape[0], 0))
    lam, Q = sla.eigh(A, B)
    lam = np.clip(lam, 0.0, None)
    order = np.argsort(lam)
    lam, Q = lam[order], Q[:, order]
    if n_modes is not None:
        lam, Q = lam[: int(n_modes)], Q[:, : int(n_modes)]
    return lam, basis @ Q


def structural_dynamic_modification(
    modal: Any,
    springs: Sequence[Any] = (),
    masses: Sequence[Any] = (),
    *,
    dK: Any = None,
    dM: Any = None,
    n_modes: int | None = None,
    return_matrices: bool = True,
) -> ModificationResult:
    """Predict modes after adding discrete springs and masses (SDM).

    Parameters
    ----------
    modal:
        Baseline modal model. Its truncation bounds the accuracy of the prediction: only
        the retained subspace can represent the modified shapes.
    springs:
        Spring modifications: :class:`SpringModification`, ``(dof, k)`` (grounded),
        ``(dof_i, dof_j, k)``, or ``{"dofs": (i, j), "k": ...}``.
    masses:
        Mass modifications: :class:`MassModification`, ``(dof, mass)`` or
        ``{"dof": ..., "mass": ...}``.
    dK, dM:
        Optional full ``(ndof, ndof)`` modification matrices added on top of the discrete
        ones (dense or sparse).
    n_modes:
        Keep only the lowest ``n_modes`` modified modes.
    return_matrices:
        Attach the assembled sparse ``dK``/``dM`` to the result.

    Returns
    -------
    ModificationResult
        A modal model of the modified structure, mass-normalised w.r.t. ``M + dM``.
    """
    mm = as_modal(modal).mass_normalized()
    spr = [_as_spring(s) for s in springs]
    mas = [_as_mass(m) for m in masses]
    for s in spr:
        for d in (s.dof_i, s.dof_j):
            if d is not None and not 0 <= d < mm.ndof:
                raise IndexError(f"spring DOF {d} out of range for {mm.ndof} DOFs")
    for m in mas:
        if not 0 <= m.dof < mm.ndof:
            raise IndexError(f"mass DOF {m.dof} out of range for {mm.ndof} DOFs")

    dKm, dMm = _modal_projection(mm.modes, spr, mas)
    if dK is not None:
        dKm = dKm + as_dense(mm.modes.T @ (dK @ mm.modes))
    if dM is not None:
        dMm = dMm + as_dense(mm.modes.T @ (dM @ mm.modes))

    lam, Phi_new = _solve_reduced(
        mm.modes, np.asarray(mm.eigenvalues, dtype=float), dKm, dMm, None, n_modes
    )

    dK_out = dM_out = None
    if return_matrices:
        dK_out, dM_out = _physical_matrices(mm.ndof, spr, mas)
        if dK is not None:
            dK_out = dK_out + dK
        if dM is not None:
            dM_out = dM_out + dM

    return ModificationResult(
        freq_hz=np.sqrt(lam) / TWO_PI,
        modes=Phi_new,
        generalized_mass=np.ones(lam.size),
        eigenvalues=lam,
        dof_ids=None if mm.dof_ids is None else mm.dof_ids.copy(),
        meta={"method": "sdm", "n_baseline_modes": mm.n_modes},
        baseline_freq_hz=mm.freq_hz.copy(),
        dK=dK_out,
        dM=dM_out,
    )


def _prepare_components(components: Sequence[Any]) -> list[ModalComponent]:
    comps: list[ModalComponent] = []
    dof_offset = 0
    mode_offset = 0
    for idx, item in enumerate(components):
        if isinstance(item, ModalComponent):
            comp = ModalComponent(
                modal=as_modal(item.modal).mass_normalized(),
                name=item.name or f"C{idx}",
            )
        elif isinstance(item, tuple | list) and len(item) == 2 and isinstance(item[0], str):
            comp = ModalComponent(modal=as_modal(item[1]).mass_normalized(), name=item[0])
        else:
            comp = ModalComponent(modal=as_modal(item).mass_normalized(), name=f"C{idx}")
        comp.dof_offset = dof_offset
        comp.mode_offset = mode_offset
        dof_offset += comp.ndof
        mode_offset += comp.n_modes
        comps.append(comp)
    if not comps:
        raise ValueError("at least one component is required")
    return comps


def _resolve_endpoint(comps: list[ModalComponent], spec: Any) -> int:
    """Resolve ``(component, local_dof)`` (or a plain global index) to a global DOF."""
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
    comps: list[ModalComponent], connections: Sequence[Any]
) -> tuple[list[tuple[int, int]], list[SpringModification]]:
    """Split connections into rigid DOF pairs and connection springs."""
    rigid: list[tuple[int, int]] = []
    springs: list[SpringModification] = []
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
            if len(seq) == 4:
                a = _resolve_endpoint(comps, (seq[0], seq[1]))
                b = _resolve_endpoint(comps, (seq[2], seq[3]))
                k = None
            elif len(seq) == 5:
                a = _resolve_endpoint(comps, (seq[0], seq[1]))
                b = _resolve_endpoint(comps, (seq[2], seq[3]))
                k = seq[4]
            elif len(seq) == 2:
                a = _resolve_endpoint(comps, seq[0])
                b = _resolve_endpoint(comps, seq[1])
                k = None
            else:
                raise ValueError(f"cannot interpret connection spec {spec!r}")
        if k is None:
            rigid.append((a, b))
        else:
            springs.append(SpringModification(a, float(k), b))
    return rigid, springs


def _nullspace(A: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis of the null space of ``A`` (columns)."""
    if A.size == 0:
        return np.eye(A.shape[1])
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    tol = (s.max() if s.size else 0.0) * rtol
    rank = int((s > tol).sum())
    return Vt[rank:, :].T


def modal_based_assembly(
    components: Sequence[Any],
    connections: Sequence[Any] = (),
    *,
    springs: Sequence[Any] = (),
    masses: Sequence[Any] = (),
    n_modes: int | None = None,
    rigid_penalty: float | None = None,
) -> AssemblyResult:
    """Couple modal components into one assembled modal model.

    The components' modal bases are stacked block-diagonally, giving a global DOF space
    that is the concatenation of the component DOF spaces (component ``c`` occupies
    ``dof_offset[c] : dof_offset[c] + ndof[c]``).

    Parameters
    ----------
    components:
        Modal models, ``(name, modal)`` pairs, or :class:`ModalComponent` instances.
    connections:
        Interface ties. ``(ci, dof_i, cj, dof_j)`` is a *rigid* link (exact compatibility,
        eliminated through the constraint null space); ``(ci, dof_i, cj, dof_j, k)`` is an
        elastic link of stiffness ``k``. Mappings ``{"a": (ci, dof), "b": (cj, dof),
        "k": ...}`` are also accepted, as are plain global DOF pairs.
    springs, masses:
        Extra modifications expressed in *global* DOF indices (see
        :func:`structural_dynamic_modification`).
    n_modes:
        Keep only the lowest ``n_modes`` assembled modes.
    rigid_penalty:
        If given, rigid links are enforced with a penalty spring of this stiffness instead
        of by null-space elimination. Useful when the exact constraints would make the
        reduced basis rank-deficient.

    Returns
    -------
    AssemblyResult
    """
    comps = _prepare_components(components)
    ndof = sum(c.ndof for c in comps)
    nmod = sum(c.n_modes for c in comps)

    phi = np.zeros((ndof, nmod))
    wr2 = np.zeros(nmod)
    for c in comps:
        r = slice(c.dof_offset, c.dof_offset + c.ndof)
        m = slice(c.mode_offset, c.mode_offset + c.n_modes)
        phi[r, m] = c.modal.modes
        wr2[m] = np.asarray(c.modal.eigenvalues, dtype=float)

    rigid, conn_springs = _parse_connections(comps, connections)
    extra_springs = [_as_spring(s) for s in springs]
    all_springs = conn_springs + extra_springs
    all_masses = [_as_mass(m) for m in masses]

    if rigid and rigid_penalty is not None:
        all_springs = all_springs + [
            SpringModification(a, float(rigid_penalty), b) for a, b in rigid
        ]
        rigid = []

    B: np.ndarray | None = None
    Z: np.ndarray | None = None
    if rigid:
        B = np.zeros((len(rigid), ndof))
        for row, (a, b) in enumerate(rigid):
            B[row, a] += 1.0
            B[row, b] -= 1.0
        Z = _nullspace(B @ phi)
        if Z.shape[1] == 0:
            raise ValueError(
                "the rigid connections leave no free modal coordinates; the component "
                "bases are too coarse to satisfy compatibility"
            )

    dKm, dMm = _modal_projection(phi, all_springs, all_masses)
    lam, Phi_new = _solve_reduced(phi, wr2, dKm, dMm, Z, n_modes)

    return AssemblyResult(
        freq_hz=np.sqrt(lam) / TWO_PI,
        modes=Phi_new,
        generalized_mass=np.ones(lam.size),
        eigenvalues=lam,
        meta={
            "method": "mba",
            "n_components": len(comps),
            "n_rigid_links": 0 if B is None else int(B.shape[0]),
            "n_springs": len(all_springs),
        },
        components=tuple(comps),
        constraint_matrix=B,
    )
