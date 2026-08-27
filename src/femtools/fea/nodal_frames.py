"""Per-node rotational frames for shell meshes.

A flat shell element has no stiffness about its own normal, so ``TRIA3`` and
``QUAD4`` carry a rank deficient drilling penalty and the assembler removes the
drilling rotations that receive nothing else.  Removing them *by index* only
works while the drilling direction happens to be a global DOF: tilt the same
plate and the drilling rotation becomes a mixture of ``rx``, ``ry`` and ``rz``,
no whole DOF can be dropped, and the mesh keeps a zero-energy mechanism that
reads as a seventh rigid body mode.

The cure is the one every production solver uses -- give each node its own
rotational frame (Nastran's ``GRID`` displacement coordinate system, the
"nodal/local triad" of the shell literature: Bathe, *Finite Element
Procedures*, §5.5.2 "skew boundary conditions"; MacNeal, *Finite Elements:
Their Design and Performance*, §7 on shell normals).  At a shell node the local
3-axis is the area-weighted average of the normals of the attached shell
elements, and the two remaining axes span the tangent plane, so the drilling
rotation *is* local component 5 again and the existing one-DOF elimination
applies at any orientation.

Conventions
-----------

:class:`NodalFrames` stores one ``(3, 3)`` matrix ``R`` per framed node whose
**columns** are the local basis vectors expressed in the basic (global) frame,
so a nodal rotation vector transforms as ``theta_basic = R @ theta_analysis``.
The whole assembly is then a congruence with the block diagonal

``Lambda = diag(I3, R)`` per node (identity for every unframed node), which is
orthogonal; displacements *and* forces therefore share one pair of maps,
:meth:`NodalFrames.to_basic` and :meth:`NodalFrames.from_basic`.

Nothing is done where nothing is needed: when the averaged normal is already
parallel to a global axis the node keeps the identity frame, so an axis-aligned
model assembles to bit-for-bit the same matrices as before this module existed.
Nodes carrying a rotational single point constraint are left alone as well --
an SPC is written in the basic frame and only stays a single-DOF constraint
there.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .dofmap import DofMap
from .elements import ModelIndex, element_spec
from .protocols import get_any, iter_records

__all__ = [
    "NodalFrames",
    "averaged_shell_normals",
    "rotation_triad",
    "shell_nodal_frames",
]

#: An averaged normal closer than this to a global axis is treated as *being*
#: that axis, which keeps the frame (and the assembled matrices) identical.
AXIS_TOLERANCE = 1.0e-12


def _element_normal_area(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit normal and area of a flat three or four node surface element.

    Uses the cross product of the diagonals for a quadrilateral, which is the
    exact area of any planar quad and degrades gracefully for a mildly warped
    one; the same quantity :func:`femtools.fea.elements.frames.shell_frame`
    builds its local 3-axis from.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 3:
        cross = np.cross(coords[1] - coords[0], coords[2] - coords[0])
    else:
        cross = np.cross(coords[2] - coords[0], coords[3] - coords[1])
    norm = float(np.linalg.norm(cross))
    if norm <= 0.0:
        raise ValueError("degenerate surface element (zero normal)")
    return cross / norm, 0.5 * norm


def averaged_shell_normals(
    model: Any, *, index: ModelIndex | None = None
) -> dict[Any, np.ndarray]:
    """Area-weighted average shell normal at every node touched by a shell.

    Element normals are sign-aligned with the first one seen at each node
    before they are averaged, so a mesh whose element numbering flips halfway
    through still produces one consistent normal field.  A node whose attached
    normals cancel (two panels folded back on each other) is left out: it has
    no meaningful triad, and the drilling elimination must not fire there.
    """
    index = ModelIndex.build(model) if index is None else index
    total: dict[Any, np.ndarray] = {}
    reference: dict[Any, np.ndarray] = {}

    for _eid, element in iter_records(get_any(model, ("elements", "elems", "element"), None)):
        if element is None:
            continue
        etype = str(get_any(element, ("type", "etype", "element_type", "kind"), "")).strip()
        try:
            spec = element_spec(etype)
        except KeyError:
            continue
        if spec.family != "shell":
            continue
        conn = tuple(
            get_any(element, ("nodes", "node_ids", "connectivity", "conn", "grids"), ()) or ()
        )
        conn = conn[:4] if len(conn) >= 4 else conn[:3]
        if len(conn) < 3:
            continue
        try:
            coords = np.array([index.xyz(nid) for nid in conn], dtype=float)
            normal, area = _element_normal_area(coords)
        except (KeyError, TypeError, ValueError):
            continue
        for nid in conn:
            ref = reference.setdefault(nid, normal)
            sign = -1.0 if float(ref @ normal) < 0.0 else 1.0
            total[nid] = total.get(nid, np.zeros(3)) + (sign * area) * normal

    out: dict[Any, np.ndarray] = {}
    for nid, vector in total.items():
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            out[nid] = vector / norm
    return out


def rotation_triad(normal: Any) -> np.ndarray:
    """Right-handed ``(3, 3)`` triad whose third **column** is *normal*.

    The in-plane pair is anchored on the global axis the normal is *least*
    aligned with, which makes the construction stable (never a near-zero
    projection) and reproducible: the same normal always gives the same triad,
    so two assemblies of the same model agree bit for bit.
    """
    n = np.asarray(normal, dtype=float).ravel()
    if n.size != 3:
        raise ValueError("a nodal normal must have three components")
    length = float(np.linalg.norm(n))
    if length <= 0.0:
        raise ValueError("cannot build a nodal frame from a zero normal")
    n = n / length

    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(n)))] = 1.0
    e1 = reference - float(reference @ n) * n
    e1 = e1 / float(np.linalg.norm(e1))
    e2 = np.cross(n, e1)
    return np.column_stack([e1, e2, n])


def _is_global_axis(normal: np.ndarray, tol: float) -> bool:
    return bool(1.0 - float(np.max(np.abs(normal))) <= tol)


@dataclass(frozen=True)
class NodalFrames:
    """The rotational triads of one assembly, and the maps they induce.

    ``frames`` maps a node id to the ``(3, 3)`` matrix whose columns are the
    local axes in basic coordinates; every node absent from it keeps the basic
    frame.  ``normals`` records the averaged shell normal of *every* shell node
    including the ones left unframed, which is what makes the decision
    reportable rather than merely internal.
    """

    dof_map: DofMap
    frames: dict[Any, np.ndarray] = field(default_factory=dict)
    normals: dict[Any, np.ndarray] = field(default_factory=dict)

    # -- basics -------------------------------------------------------
    @property
    def is_identity(self) -> bool:
        return not self.frames

    @property
    def n_framed(self) -> int:
        return len(self.frames)

    @property
    def framed_nodes(self) -> list[Any]:
        return list(self.frames)

    def frame(self, node_id: Any) -> np.ndarray:
        """Triad of *node_id*; the identity for an unframed node."""
        found = self.frames.get(node_id)
        return np.eye(3) if found is None else found

    def rotation_dofs(self, node_id: Any) -> np.ndarray:
        base = self.dof_map.position(node_id) * self.dof_map.dofs_per_node
        return np.arange(base + 3, base + 6, dtype=int)

    def is_framed(self, node_id: Any) -> bool:
        return node_id in self.frames

    # -- the transformation -------------------------------------------
    def matrix(self) -> sp.csr_matrix:
        """The orthogonal ``Lambda`` with ``u_basic = Lambda @ u_analysis``."""
        n = self.dof_map.n_dof
        if self.is_identity:
            return sp.identity(n, format="csr")
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        touched = np.zeros(n, dtype=bool)
        for nid, R in self.frames.items():
            idx = self.rotation_dofs(nid)
            touched[idx] = True
            rows.append(np.repeat(idx, 3))
            cols.append(np.tile(idx, 3))
            vals.append(np.asarray(R, dtype=float).ravel())
        rest = np.flatnonzero(~touched)
        rows.append(rest)
        cols.append(rest)
        vals.append(np.ones(rest.size))
        return sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tocsr()

    def to_basic(self, vector: Any) -> np.ndarray:
        """Analysis frame -> basic frame (displacements *and* forces)."""
        v = np.asarray(vector)
        return v if self.is_identity else np.asarray(self.matrix() @ v)

    def from_basic(self, vector: Any) -> np.ndarray:
        """Basic frame -> analysis frame."""
        v = np.asarray(vector)
        return v if self.is_identity else np.asarray(self.matrix().T @ v)

    def congruence(self, matrix: sp.spmatrix) -> sp.csr_matrix:
        """``Lambda.T @ matrix @ Lambda``, the assembled form of the change of frame."""
        if self.is_identity:
            return matrix.tocsr()
        lam = self.matrix()
        return (lam.T @ matrix.tocsr() @ lam).tocsr()

    def summary(self) -> str:  # pragma: no cover - reporting helper
        return f"NodalFrames(framed={self.n_framed}, shell_nodes={len(self.normals)})"


def shell_nodal_frames(
    model: Any,
    dof_map: DofMap,
    *,
    index: ModelIndex | None = None,
    skip: Iterable[Any] = (),
    axis_tolerance: float = AXIS_TOLERANCE,
) -> NodalFrames:
    """Build the rotational triads of a shell model.

    A node is framed when it carries an averaged shell normal that is *not*
    parallel to a global axis and it is not listed in ``skip`` (the caller
    passes the nodes whose rotations are single point constrained, because an
    SPC is written in the basic frame).  Everything else keeps the identity, so
    the transformation is a no-op for a model without oblique shells.
    """
    if dof_map.dofs_per_node < 6:
        return NodalFrames(dof_map=dof_map)
    normals = averaged_shell_normals(model, index=index)
    skipped = set(skip)
    frames: dict[Any, np.ndarray] = {}
    for nid, normal in normals.items():
        if nid in skipped or _is_global_axis(normal, axis_tolerance):
            continue
        try:
            dof_map.position(nid)
        except KeyError:  # pragma: no cover - defensive
            continue
        frames[nid] = rotation_triad(normal)
    return NodalFrames(dof_map=dof_map, frames=frames, normals=normals)
