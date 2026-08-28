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


#: Barycentric orbit of the 15-point degree-5 tetrahedron rule of Keast (1986),
#: *Moderate-degree tetrahedral quadrature formulas*, CMAME 55, pp. 339-348:
#: ``(a, b, weight)`` for the permutations of ``(a, a, b, b)``.  Split out so
#: the constants can be read against the paper.
_KEAST5_AABB = (0.066550153573664, 0.433449846426336, 0.010949141561386)


def tet_rule(order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Volume-coordinate rules on the unit tetrahedron (weights sum to 1/6).

    The returned points are ``(L2, L3, L4)``; ``L1 = 1 - L2 - L3 - L4``.  The
    *order* asked for selects the smallest rule that carries it:

    ``<= 1``
        The centroid, exact for linear integrands.
    ``2`` to ``3``
        The four symmetric points of the classical degree-2 rule, which is
        what a straight-edged ``TET10`` stiffness needs (constant Jacobian, so
        the integrand is quadratic).
    ``>= 4``
        Keast's 15-point degree-5 rule, all weights positive.  Wanted for the
        consistent ``TET10`` mass, whose integrand ``N_i N_j`` is quartic and
        which the four-point rule would get wrong by several percent.
    """
    if order <= 1:
        return np.array([[0.25, 0.25, 0.25]]), np.array([1.0 / 6.0])
    if order <= 3:
        a = (5.0 - np.sqrt(5.0)) / 20.0
        b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
        pts = np.array([[a, a, a], [b, a, a], [a, b, a], [a, a, b]])
        return pts, np.full(4, 1.0 / 24.0)

    points: list[list[float]] = [[0.25, 0.25, 0.25]]
    weights: list[float] = [0.030283678097089]
    for value, weight in ((0.0, 0.006026785714286), (8.0 / 11.0, 0.011645249086029)):
        rest = (1.0 - value) / 3.0
        bary = [value, rest, rest, rest]
        for k in range(4):
            rotated = bary[k:] + bary[:k]
            points.append(rotated[1:])
            weights.append(weight)
    a, b, weight = _KEAST5_AABB
    for i in range(4):
        for j in range(i + 1, 4):
            bary = [b, b, b, b]
            bary[i] = bary[j] = a
            points.append(bary[1:])
            weights.append(weight)
    w = np.array(weights)
    # The published weights are decimals and miss 1/6 in the last bits; a
    # rule that does not integrate a constant exactly would show up as a
    # missing microgram of total mass.
    return np.array(points), w * ((1.0 / 6.0) / w.sum())
