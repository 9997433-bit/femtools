"""Gauss and simplex quadrature rules used by the element library."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["gauss_1d", "gauss_2d", "gauss_3d", "tri_rule", "tet_rule"]


@lru_cache(maxsize=8)
def gauss_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre points and weights on ``[-1, 1]``."""
    pts, wts = np.polynomial.legendre.leggauss(int(n))
    return pts.astype(float), wts.astype(float)


def gauss_2d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor product rule on the bi-unit square; returns ``(pts(m,2), w(m,))``."""
    p, w = gauss_1d(n)
    xi, eta = np.meshgrid(p, p, indexing="ij")
    wi, wj = np.meshgrid(w, w, indexing="ij")
    return np.column_stack([xi.ravel(), eta.ravel()]), (wi * wj).ravel()


def gauss_3d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor product rule on the bi-unit cube; returns ``(pts(m,3), w(m,))``."""
    p, w = gauss_1d(n)
    xi, eta, zeta = np.meshgrid(p, p, p, indexing="ij")
    wi, wj, wk = np.meshgrid(w, w, w, indexing="ij")
    return (
        np.column_stack([xi.ravel(), eta.ravel(), zeta.ravel()]),
        (wi * wj * wk).ravel(),
    )


def tri_rule(order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Area-coordinate rules on the unit triangle (weights sum to 1/2)."""
    if order <= 1:
        return np.array([[1.0 / 3.0, 1.0 / 3.0]]), np.array([0.5])
    if order == 2:
        # Three mid-side points: exact for quadratics, used by DKT.
        pts = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        return pts, np.full(3, 1.0 / 6.0)
    a, b = 0.6, 0.2
    pts = np.array([[1.0 / 3.0, 1.0 / 3.0], [a, b], [b, a], [b, b]])
    wts = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
    return pts, wts


def tet_rule(order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Volume-coordinate rules on the unit tetrahedron (weights sum to 1/6)."""
    if order <= 1:
        return np.array([[0.25, 0.25, 0.25]]), np.array([1.0 / 6.0])
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    pts = np.array([[a, a, a], [b, a, a], [a, b, a], [a, a, b]])
    return pts, np.full(4, 1.0 / 24.0)
