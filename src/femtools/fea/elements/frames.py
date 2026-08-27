"""Local coordinate frames for line and surface elements."""

from __future__ import annotations

import numpy as np

__all__ = ["line_frame", "shell_frame", "default_orientation"]

_EY = np.array([0.0, 1.0, 0.0])
_EZ = np.array([0.0, 0.0, 1.0])


def default_orientation(axis: np.ndarray) -> np.ndarray:
    """Pick a sensible orientation vector for a beam whose axis is *axis*.

    Global ``+Y`` is preferred so that a beam running along global ``X`` gets a
    local frame identical to the global one; ``+Z`` is used when the axis is
    nearly parallel to ``+Y``.
    """
    if abs(float(np.dot(axis, _EY))) < 0.9:
        return _EY.copy()
    return _EZ.copy()


def line_frame(p1: np.ndarray, p2: np.ndarray, orientation: np.ndarray | None = None):
    """Return ``(length, R)`` for a two-node line element.

    ``R`` has the local basis vectors as *rows*, so ``v_local = R @ v_global``.
    Local ``x`` runs from node 1 to node 2; local ``y`` lies in the plane
    spanned by the axis and the orientation vector (Nastran "plane 1").
    """
    d = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(d))
    if length <= 0.0:
        raise ValueError("line element has zero length")
    e1 = d / length

    v = None if orientation is None else np.asarray(orientation, dtype=float).ravel()
    if v is None or v.size != 3 or not np.isfinite(v).all() or np.linalg.norm(v) == 0.0:
        v = default_orientation(e1)
    e2 = v - np.dot(v, e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1.0e-12:
        v = default_orientation(e1)
        e2 = v - np.dot(v, e1) * e1
        n2 = np.linalg.norm(e2)
        if n2 < 1.0e-12:
            v = _EZ if abs(e1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e2 = v - np.dot(v, e1) * e1
            n2 = np.linalg.norm(e2)
    e2 = e2 / n2
    e3 = np.cross(e1, e2)
    return length, np.vstack([e1, e2, e3])


def shell_frame(coords: np.ndarray):
    """Return ``(R, xy, centroid)`` for a flat (or mildly warped) surface.

    ``R`` has the local basis as rows, ``xy`` are the node coordinates
    projected onto the local mid-plane.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    centroid = coords.mean(axis=0)
    if n == 3:
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        normal = np.cross(v1, v2)
        e1 = v1
    else:
        d1 = coords[2] - coords[0]
        d2 = coords[3] - coords[1]
        normal = np.cross(d1, d2)
        e1 = (coords[1] + coords[2]) - (coords[0] + coords[3])
    nn = np.linalg.norm(normal)
    if nn < 1.0e-300:
        raise ValueError("degenerate surface element (zero normal)")
    e3 = normal / nn
    e1 = e1 - np.dot(e1, e3) * e3
    n1 = np.linalg.norm(e1)
    if n1 < 1.0e-300:
        raise ValueError("degenerate surface element (cannot build local x)")
    e1 = e1 / n1
    e2 = np.cross(e3, e1)
    R = np.vstack([e1, e2, e3])
    local = (coords - centroid) @ R.T
    return R, local[:, :2].copy(), centroid
