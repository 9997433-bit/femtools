"""Rigid alignment of the test geometry with the analysis geometry.

Correlation assumes both models describe the same points in the same frame.
A test geometry rarely does: it is digitized in the frame of the laser
tracker or the test rig, so it sits rotated, shifted and — when the sensor
positions were measured in different units or from a scaled drawing — scaled
with respect to the FE mesh.  Correlating before fixing that compares
different directions with each other and produces a diffuse, uninformative
MAC.

:func:`align_geometry` solves the orthogonal Procrustes problem for the two
point clouds (Kabsch, extended by Umeyama for the scale factor): it returns
the rotation ``R``, translation ``t`` and optional uniform scale ``s``
minimizing the weighted sum of squares ``sum_i w_i ||s R p_i + t - q_i||^2``,
with the reflection excluded by construction — a mirrored "fit" would map a
right-handed test frame onto a left-handed one and silently flip a sign in
every mode shape.

The transform then moves both the geometry and the measured directions::

    fit = align_geometry(test_xyz, fe_xyz)      # test frame -> FE frame
    xyz_in_fe = fit.apply(test_xyz)
    phi_in_fe = fit.rotate_modes(phi_test, test_map)

Rotating the mode shapes is the part that is easy to forget: an
accelerometer measures a component along a *test* axis, so once the geometry
is rotated, each measured translation triad has to be rotated with it before
its X component means the same thing as the model's.

References: W. Kabsch, Acta Cryst. A32, 1976; S. Umeyama, IEEE PAMI 13(4),
1991 (least-squares estimation of similarity transformations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._linalg import as_mode_matrix
from .dofmap import DOFMap

__all__ = ["AlignmentResult", "align_geometry", "rotate_modes"]


@dataclass
class AlignmentResult:
    """Rigid (or similarity) transform mapping the source points onto the target."""

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    scale: float = 1.0
    rms: float = 0.0
    source: NDArray[np.float64] | None = None
    target: NDArray[np.float64] | None = None
    ids: NDArray[Any] | None = None
    reflection: bool = False
    rank: int = 0
    underdetermined: bool = False

    @property
    def n_points(self) -> int:
        return 0 if self.source is None else int(self.source.shape[0])

    @property
    def dim(self) -> int:
        return int(self.rotation.shape[0])

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Homogeneous ``(d+1, d+1)`` transform ``[[sR, t], [0, 1]]``."""
        d = self.dim
        out = np.eye(d + 1)
        out[:d, :d] = self.scale * self.rotation
        out[:d, d] = self.translation
        return out

    @property
    def angle_deg(self) -> float:
        """Rotation angle [deg] (about the axis :attr:`axis` in 3-D)."""
        cos = (np.trace(self.rotation) - (self.dim - 2)) / 2.0
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    @property
    def axis(self) -> NDArray[np.float64]:
        """Unit rotation axis (3-D only; arbitrary for a null rotation)."""
        if self.dim != 3:
            raise ValueError("a rotation axis is only defined in 3-D")
        vals, vecs = np.linalg.eig(self.rotation)
        col = int(np.argmin(np.abs(vals - 1.0)))
        axis = np.real(vecs[:, col])
        norm = float(np.linalg.norm(axis))
        return axis / norm if norm > 0.0 else axis

    @property
    def residuals(self) -> NDArray[np.float64]:
        """Distance between each transformed source point and its target."""
        if self.source is None or self.target is None:
            return np.zeros(0)
        return np.linalg.norm(self.apply(self.source) - self.target, axis=1)

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        """Transform coordinates: ``s R p + t``."""
        p = np.asarray(points, dtype=float)
        flat = p.reshape(-1, self.dim)
        out = self.scale * (flat @ self.rotation.T) + self.translation
        return out.reshape(p.shape)

    def apply_vectors(self, vectors: ArrayLike) -> NDArray[np.float64]:
        """Rotate direction vectors: ``R v``, with no translation or scale.

        Displacements, velocities and accelerations are directions, not
        positions: they must not pick up the translation, and the scale is a
        unit/geometry factor that does not belong on a measured amplitude.
        """
        v = np.asarray(vectors, dtype=float)
        return (v.reshape(-1, self.dim) @ self.rotation.T).reshape(v.shape)

    def rotate_modes(self, phi: ArrayLike, dof_map: Any) -> NDArray[Any]:
        """Rotate the DOF triads of a mode set into the target frame.

        See :func:`rotate_modes`.
        """
        return rotate_modes(phi, dof_map, self.rotation)

    def inverse(self) -> AlignmentResult:
        """The transform mapping the target points back onto the source."""
        inv_scale = 1.0 / self.scale
        rot = self.rotation.T
        return AlignmentResult(
            rotation=rot,
            translation=-inv_scale * (rot @ self.translation),
            scale=inv_scale,
            rms=self.rms * inv_scale,
            source=self.target,
            target=self.source,
            ids=self.ids,
            reflection=self.reflection,
        )

    def table(self) -> str:
        """Plain-text summary of the fitted transform."""
        lines = [
            f"points        {self.n_points}",
            f"scale         {self.scale:.6f}",
            f"rotation      {self.angle_deg:.4f} deg",
            "translation   " + " ".join(f"{t:.6g}" for t in self.translation),
            f"rms residual  {self.rms:.6g}",
        ]
        if self.reflection:
            lines.append("reflection    yes (right-handedness not preserved)")
        if self.underdetermined:
            lines.append(f"WARNING       points span only {self.rank}-D: rotation not unique")
        return "\n".join(lines)


def _points(source: Any, name: str) -> tuple[NDArray[Any] | None, NDArray[np.float64]]:
    """``(ids, xyz)`` from an array, a ``{node: xyz}`` mapping or an ``(ids, xyz)`` pair."""
    ids: NDArray[Any] | None = None
    if isinstance(source, dict):
        if not source:
            raise ValueError(f"{name} is empty")
        ids = np.array(list(source.keys()))
        xyz = np.array([np.asarray(v, dtype=float).reshape(-1) for v in source.values()])
    elif isinstance(source, tuple) and len(source) == 2 and np.ndim(source[1]) == 2:
        ids = np.asarray(source[0]).reshape(-1)
        xyz = np.asarray(source[1], dtype=float)
        if ids.size != xyz.shape[0]:
            raise ValueError(f"{name}: {ids.size} ids for {xyz.shape[0]} points")
    else:
        xyz = np.asarray(source, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] not in (2, 3):
        raise ValueError(f"{name} must be (n_point, 2) or (n_point, 3), got shape {xyz.shape}")
    return ids, xyz


def _correspondence(
    ids_a: NDArray[Any] | None,
    ids_b: NDArray[Any] | None,
    pairs: Any,
    n_a: int,
    n_b: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[Any] | None]:
    """Matched row positions of the two clouds, plus the shared ids if any."""
    if pairs is not None:
        arr = np.asarray(pairs)
        if arr.ndim == 2 and arr.shape[1] == 2:
            ia, ib = arr[:, 0].astype(np.intp), arr[:, 1].astype(np.intp)
        elif arr.ndim == 1 and arr.size == n_a:
            ia, ib = np.arange(n_a, dtype=np.intp), arr.astype(np.intp)
        else:
            raise ValueError("correspondence must be (n_pair, 2) pairs or one target per source")
        if ia.size and (ia.max() >= n_a or ib.max() >= n_b or ia.min() < 0 or ib.min() < 0):
            raise ValueError("correspondence index out of range")
        return ia, ib, None if ids_a is None else ids_a[ia]

    if ids_a is not None and ids_b is not None:
        common, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
        if common.size == 0:
            raise ValueError("the two point sets have no id in common")
        order = np.argsort(ia, kind="stable")
        return ia[order].astype(np.intp), ib[order].astype(np.intp), common[order]

    if n_a != n_b:
        raise ValueError(
            f"{n_a} source points but {n_b} target points; give matching ids or a correspondence"
        )
    return np.arange(n_a, dtype=np.intp), np.arange(n_b, dtype=np.intp), ids_a


def align_geometry(
    source: Any,
    target: Any,
    *,
    correspondence: Any = None,
    weights: ArrayLike | None = None,
    scale: bool = False,
    reflection: bool = False,
) -> AlignmentResult:
    """Fit the rigid transform that maps the ``source`` points onto ``target``.

    Solves ``min_{R, t, s} sum_i w_i ||s R p_i + t - q_i||^2`` over rotations
    ``R`` (``det R = +1``), translations ``t`` and, when ``scale`` is set, one
    uniform factor ``s`` — the classical Procrustes / Kabsch–Umeyama fit, in
    closed form from the SVD of the weighted cross-covariance.  The usual use
    is ``align_geometry(test_xyz, fe_xyz)``, giving the transform that brings
    the test geometry into the model frame.

    Parameters
    ----------
    source, target:
        Point clouds: ``(n_point, 3)`` (or ``(n_point, 2)``) arrays, a
        ``{node: xyz}`` mapping, or an ``(ids, xyz)`` pair.  When both carry
        ids, the common ids are matched automatically — which is what makes
        this usable with the node numbering of the two models.  Otherwise the
        rows must already correspond one to one.
        :func:`~femtools.pretest.candidates.node_coordinates` extracts either
        form from a model.
    correspondence:
        Explicit matching, overriding the ids: ``(n_pair, 2)`` index pairs, or
        one target row index per source row.
    weights:
        Per-point weight, e.g. the inverse variance of a digitized position,
        or 0 to ignore a suspect point.
    scale:
        Also fit a uniform scale factor (unit mismatch, scaled drawing).
        Off by default: a free scale absorbs a genuine geometric error.
    reflection:
        Allow ``det R = -1``.  Off by default; a reflection is almost always
        a sign convention error in the test data rather than a real transform.

    Returns
    -------
    AlignmentResult
        The transform, its RMS residual and the matched point sets.  Use
        :meth:`AlignmentResult.apply` for coordinates and
        :meth:`AlignmentResult.rotate_modes` for the measured shapes.

    Notes
    -----
    The points must span enough space to pin the rotation down: sensors in a
    straight line (a beam) leave the spin about that line free, and every
    member of that family fits with the same zero residual.  Such a fit is
    reported with ``result.underdetermined`` set and resolved by returning the
    *smallest* rotation of the family, which leaves the unconstrained spin at
    zero instead of picking an arbitrary value.  ``result.rank`` gives the
    dimension the matched points actually span.
    """
    ids_a, p_all = _points(source, "source")
    ids_b, q_all = _points(target, "target")
    if p_all.shape[1] != q_all.shape[1]:
        raise ValueError(f"{p_all.shape[1]}-D source but {q_all.shape[1]}-D target")

    ia, ib, ids = _correspondence(ids_a, ids_b, correspondence, p_all.shape[0], q_all.shape[0])
    p = p_all[ia]
    q = q_all[ib]
    n, d = p.shape

    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size == 1:
            w = np.full(n, float(w[0]))
        if w.size != n:
            raise ValueError(f"weights has {w.size} entries but {n} points are matched")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("the weights are all zero")
    if n < d:
        raise ValueError(f"{n} matched point(s) cannot determine a {d}-D rotation")

    mean_p = (w @ p) / total
    mean_q = (w @ q) / total
    pc = p - mean_p
    qc = q - mean_q

    cov = (qc * w[:, None]).T @ pc / total
    scatter = (pc * w[:, None]).T @ pc / total
    rank = int(np.linalg.matrix_rank(scatter))
    underdetermined = rank < (d if reflection else d - 1)
    if underdetermined:
        # A cloud that spans too little (collinear sensors, a single point)
        # leaves a whole family of rotations fitting equally well.  Biasing
        # the objective towards `trace(R)` picks the smallest rotation of that
        # family instead of an arbitrary member of it.
        reference = float(np.linalg.norm(cov, 2))
        cov = cov + (1e-9 * reference if reference > 0.0 else 1.0) * np.eye(d)
    u, s_val, vt = np.linalg.svd(cov)
    correction = np.ones(d)
    if not reflection and np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[-1] = -1.0
        s_val = s_val * correction
    rot = (u * correction) @ vt

    var_p = float((w @ np.einsum("ij,ij->i", pc, pc)) / total)
    factor = float(s_val.sum() / var_p) if (scale and var_p > 0.0) else 1.0
    shift = mean_q - factor * (rot @ mean_p)

    residual = factor * (p @ rot.T) + shift - q
    rms = float(np.sqrt((w @ np.einsum("ij,ij->i", residual, residual)) / total))

    return AlignmentResult(
        rotation=rot,
        translation=shift,
        scale=factor,
        rms=rms,
        source=p,
        target=q,
        ids=ids,
        reflection=bool(np.linalg.det(rot) < 0.0),
        rank=rank,
        underdetermined=underdetermined,
    )


def _triads(dmap: DOFMap, components: tuple[int, int, int]) -> NDArray[np.intp]:
    """Rows of the complete ``components`` triads, one row of 3 positions per node."""
    nodes_per: list[NDArray[np.int64]] = []
    pos_per: list[NDArray[np.intp]] = []
    for c in components:
        sel = np.flatnonzero(dmap.components == c).astype(np.intp)
        nodes_per.append(dmap.nodes[sel])
        pos_per.append(sel)
    common = nodes_per[0]
    for nodes in nodes_per[1:]:
        common = np.intersect1d(common, nodes)
    rows = np.empty((common.size, 3), dtype=np.intp)
    for k, (nodes, pos) in enumerate(zip(nodes_per, pos_per, strict=True)):
        order = np.argsort(nodes, kind="stable")
        rows[:, k] = pos[order][np.searchsorted(nodes[order], common)]
    return rows


def rotate_modes(phi: ArrayLike, dof_map: Any, rotation: ArrayLike) -> NDArray[Any]:
    """Rotate the translation (and rotation) triads of a mode set.

    Each node contributes a vector — its measured translation — expressed in
    the component directions of ``dof_map``; rotating the geometry means
    replacing that vector by ``R v``.  Nodes whose ``X, Y, Z`` triad is
    incomplete keep their rows unchanged: a single-axis accelerometer cannot
    be re-oriented, only discarded, and silently zeroing it would be worse
    than leaving it out of the alignment.

    Rotational DOFs (``RX, RY, RZ``) are rotated the same way when all three
    are present, which is what a pseudo-rotation measurement or an expanded
    shape provides.
    """
    p = as_mode_matrix(phi, "phi")
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got shape {r.shape}")
    dmap = DOFMap.from_mapping(dof_map)
    if len(dmap) != p.shape[0]:
        raise ValueError(f"dof_map has {len(dmap)} DOF but phi has {p.shape[0]} rows")

    out = p.astype(complex) if np.iscomplexobj(p) else p.astype(float)
    for comps in ((1, 2, 3), (4, 5, 6)):
        rows = _triads(dmap, comps)
        if rows.size:
            out[rows] = np.einsum("ij,njm->nim", r, out[rows])
    return out
