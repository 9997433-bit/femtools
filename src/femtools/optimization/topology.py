"""2-D SIMP topology optimization (minimum compliance).

Solid Isotropic Material with Penalisation on a structured mesh of bilinear
plane-stress Q4 elements, solved with the optimality-criteria (OC) update and
either sensitivity or density filtering (Sigmund's mesh-independence filters).

.. math::
    \\min_{\\rho} \\; c(\\rho) = U^T K(\\rho) U
    \\quad\\text{s.t.}\\quad K(\\rho)U = F, \\;
    \\frac{V(\\rho)}{V_0} = f, \\; 0 < \\rho_{\\min} \\le \\rho \\le 1

with the modified SIMP interpolation
:math:`E(\\rho) = E_{\\min} + \\rho^p (E_0 - E_{\\min})`.

Everything (element stiffness, assembly, solve) is self-contained here so that
the optimizer does not depend on the general FEA layer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

__all__ = ["TopologyResult", "topology_simp", "element_stiffness_q4"]


@dataclass
class TopologyResult:
    """Outcome of :func:`topology_simp`.

    Attributes
    ----------
    density:
        ``(nely, nelx)`` final (filtered) design field, row 0 = top of the mesh.
    compliance:
        Final compliance ``U^T K U``.
    history:
        Per-iteration ``{iteration, compliance, volume, change}`` records.
    displacement:
        Final nodal displacement vector.
    """

    density: np.ndarray
    compliance: float
    iterations: int
    change: float
    converged: bool
    volume_fraction: float
    history: list[dict[str, Any]] = field(default_factory=list)
    displacement: np.ndarray | None = None
    nelx: int = 0
    nely: int = 0
    penal: float = 3.0
    rmin: float = 1.5
    extras: dict[str, Any] = field(default_factory=dict)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self.density if dtype is None else self.density.astype(dtype)

    def __getitem__(self, key: Any) -> Any:
        return self.density[key]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.density.shape

    @property
    def x(self) -> np.ndarray:
        return self.density

    @property
    def rho(self) -> np.ndarray:
        return self.density

    @property
    def densities(self) -> np.ndarray:
        return self.density

    @property
    def design(self) -> np.ndarray:
        return self.density

    @property
    def compliance_history(self) -> np.ndarray:
        """Compliance at every iteration, as a plain array."""
        return np.array([rec["compliance"] for rec in self.history], dtype=float)

    @property
    def discreteness(self) -> float:
        """Sigmund's measure of "greyness": 0 = fully black/white, 1 = all grey."""
        r = self.density.ravel()
        return float(np.mean(4.0 * r * (1.0 - r)))

    def as_text(self, threshold: float = 0.5) -> str:  # pragma: no cover - display aid
        return "\n".join(
            "".join("#" if v > threshold else ("+" if v > 0.2 else ".") for v in row)
            for row in self.density
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TopologyResult({self.nely}x{self.nelx}, compliance={self.compliance:.6g}, "
            f"vol={self.volume_fraction:.3f}, iterations={self.iterations})"
        )


# ----------------------------------------------------------------------
def element_stiffness_q4(nu: float = 0.3, E: float = 1.0, thickness: float = 1.0) -> np.ndarray:
    """8x8 plane-stress stiffness of a unit square bilinear Q4 element."""
    k = np.array(
        [
            1.0 / 2.0 - nu / 6.0,
            1.0 / 8.0 + nu / 8.0,
            -1.0 / 4.0 - nu / 12.0,
            -1.0 / 8.0 + 3.0 * nu / 8.0,
            -1.0 / 4.0 + nu / 12.0,
            -1.0 / 8.0 - nu / 8.0,
            nu / 6.0,
            1.0 / 8.0 - 3.0 * nu / 8.0,
        ]
    )
    KE = np.array(
        [
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
        ]
    )
    return (E * thickness / (1.0 - nu**2)) * KE


def _node_id(nelx: int, nely: int, ix: int, iy: int) -> int:
    """Column-major node numbering (matches the classic 99-line code)."""
    return ix * (nely + 1) + iy


def _edof(nelx: int, nely: int) -> np.ndarray:
    """``(n_elem, 8)`` DOF connectivity, elements ordered column-major."""
    edof = np.zeros((nelx * nely, 8), dtype=int)
    e = 0
    for ix in range(nelx):
        for iy in range(nely):
            n1 = _node_id(nelx, nely, ix, iy)
            n2 = _node_id(nelx, nely, ix + 1, iy)
            n3 = _node_id(nelx, nely, ix + 1, iy + 1)
            n4 = _node_id(nelx, nely, ix, iy + 1)
            edof[e] = [
                2 * n1, 2 * n1 + 1,
                2 * n2, 2 * n2 + 1,
                2 * n3, 2 * n3 + 1,
                2 * n4, 2 * n4 + 1,
            ]
            e += 1
    return edof


def _element_index(nelx: int, nely: int, ix: int, iy: int) -> int:
    return ix * nely + iy


def _filter_matrix(nelx: int, nely: int, rmin: float) -> Any:
    """Cone-shaped neighbourhood weights ``H`` (sparse) and its row sums."""
    if rmin <= 1.0:
        n = nelx * nely
        H = coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
        return H, np.ones(n)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    r = int(math.ceil(rmin) - 1)
    for ix in range(nelx):
        for iy in range(nely):
            e1 = _element_index(nelx, nely, ix, iy)
            for jx in range(max(ix - r, 0), min(ix + r + 1, nelx)):
                for jy in range(max(iy - r, 0), min(iy + r + 1, nely)):
                    e2 = _element_index(nelx, nely, jx, jy)
                    w = rmin - math.hypot(ix - jx, iy - jy)
                    if w > 0.0:
                        rows.append(e1)
                        cols.append(e2)
                        vals.append(w)
    n = nelx * nely
    H = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def _default_bc(
    case: str, nelx: int, nely: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(force_vector, fixed_dofs)`` for a named benchmark."""
    ndof = 2 * (nelx + 1) * (nely + 1)
    F = np.zeros(ndof)
    case = case.lower().replace("_", "-")
    if case in ("mbb", "half-mbb", "mbb-beam"):
        # Half MBB beam: unit downward load at the top-left corner, symmetry on
        # the left edge, roller under the bottom-right corner.
        F[2 * _node_id(nelx, nely, 0, 0) + 1] = -1.0
        fixed = np.union1d(
            np.array([2 * _node_id(nelx, nely, 0, iy) for iy in range(nely + 1)]),
            np.array([2 * _node_id(nelx, nely, nelx, nely) + 1]),
        )
    elif case in ("cantilever", "cantilever-tip"):
        # Left edge clamped, unit downward load at the mid-height of the right edge.
        F[2 * _node_id(nelx, nely, nelx, nely // 2) + 1] = -1.0
        fixed = np.array(
            [2 * _node_id(nelx, nely, 0, iy) + k for iy in range(nely + 1) for k in (0, 1)]
        )
    elif case in ("cantilever-corner", "l-load"):
        F[2 * _node_id(nelx, nely, nelx, nely) + 1] = -1.0
        fixed = np.array(
            [2 * _node_id(nelx, nely, 0, iy) + k for iy in range(nely + 1) for k in (0, 1)]
        )
    elif case in ("bridge", "simply-supported"):
        F[2 * _node_id(nelx, nely, nelx // 2, 0) + 1] = -1.0
        fixed = np.array(
            [
                2 * _node_id(nelx, nely, 0, nely),
                2 * _node_id(nelx, nely, 0, nely) + 1,
                2 * _node_id(nelx, nely, nelx, nely) + 1,
            ]
        )
    else:
        raise ValueError(f"unknown boundary-condition case {case!r}")
    return F, np.unique(np.asarray(fixed, dtype=int))


def _oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volfrac: float,
    move: float = 0.2,
    xmin: float = 1.0e-3,
) -> np.ndarray:
    """Optimality-criteria update with bisection on the Lagrange multiplier."""
    l1, l2 = 0.0, 1.0e9
    n = x.size
    target = volfrac * n
    xnew = x.copy()
    while (l2 - l1) / max(l2 + l1, 1e-30) > 1.0e-9:
        lmid = 0.5 * (l1 + l2)
        with np.errstate(divide="ignore", invalid="ignore"):
            be = np.maximum(-dc / (lmid * np.maximum(dv, 1e-30)), 0.0)
        xnew = np.clip(x * np.sqrt(be), np.maximum(xmin, x - move), np.minimum(1.0, x + move))
        if xnew.sum() > target:
            l1 = lmid
        else:
            l2 = lmid
    return xnew


def topology_simp(
    nelx: int = 60,
    nely: int = 20,
    volfrac: float = 0.4,
    penal: float = 3.0,
    rmin: float = 1.5,
    *,
    bc: str = "mbb",
    forces: Any = None,
    fixed_dofs: Sequence[int] | None = None,
    passive: np.ndarray | None = None,
    x0: np.ndarray | float | None = None,
    max_iter: int = 100,
    tol: float = 0.01,
    filter: str = "sensitivity",  # noqa: A002 - matches the literature's naming
    move: float = 0.2,
    E0: float = 1.0,
    Emin: float = 1.0e-9,
    nu: float = 0.3,
    thickness: float = 1.0,
    continuation: bool = False,
    callback: Any = None,
    verbose: bool = False,
    loads: Any = None,
    spcs: Sequence[int] | None = None,
    seed: int | None = None,
) -> TopologyResult:
    """Minimum-compliance SIMP topology optimization on a ``nelx x nely`` grid.

    Parameters
    ----------
    nelx, nely:
        Number of elements along x (horizontal) and y (vertical).
    volfrac:
        Prescribed volume fraction.
    penal:
        SIMP penalisation exponent (3 is the standard choice).
    rmin:
        Filter radius in element units; ``rmin <= 1`` disables filtering (and
        invites checkerboarding).
    bc:
        Named benchmark: ``"mbb"`` (half MBB beam, default), ``"cantilever"``,
        ``"cantilever-corner"`` or ``"bridge"``.  Ignored when both ``forces``
        and ``fixed_dofs`` are supplied.
    forces:
        Custom load: a full ``2*(nelx+1)*(nely+1)`` vector, a matrix of load
        cases (one column each), or a list of ``(node, dof, value)`` triples with
        ``dof in {0, 1}``.
    fixed_dofs:
        Custom constrained DOF indices.
    passive:
        ``(nely, nelx)`` integer mask: ``1`` forces void, ``2`` forces solid.
    filter:
        ``"sensitivity"`` (default, Sigmund 1997) or ``"density"``.
    continuation:
        Ramp ``penal`` from 1 to the requested value over the first iterations,
        which reduces the risk of poor local minima.
    loads, spcs:
        Spelling aliases for ``forces`` and ``fixed_dofs``.
    seed:
        Accepted for signature compatibility; the OC iteration is deterministic
        and does not use it.

    Returns
    -------
    TopologyResult

    Notes
    -----
    The mesh is numbered column-major (node ``(ix, iy)`` -> ``ix*(nely+1)+iy``)
    with ``iy=0`` at the top, matching the classic 99-line reference
    implementation, so published benchmark compliances are directly comparable.
    """
    nelx, nely = int(nelx), int(nely)
    n_elem = nelx * nely
    ndof = 2 * (nelx + 1) * (nely + 1)

    if loads is not None:
        if forces is not None:
            raise TypeError("pass either `forces` or `loads`, not both")
        forces = loads
    if spcs is not None:
        if fixed_dofs is not None:
            raise TypeError("pass either `fixed_dofs` or `spcs`, not both")
        fixed_dofs = spcs

    # -- loads / supports ------------------------------------------------
    if forces is None or fixed_dofs is None:
        F_def, fixed_def = _default_bc(bc, nelx, nely)
        F = F_def if forces is None else None
        fixed = fixed_def if fixed_dofs is None else np.asarray(fixed_dofs, dtype=int)
    else:
        F, fixed = None, np.asarray(fixed_dofs, dtype=int)

    if forces is not None:
        arr = np.asarray(forces, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] != ndof:
            F = np.zeros(ndof)
            for node, dof, val in arr:
                F[2 * int(node) + int(dof)] += float(val)
        else:
            F = arr
    assert F is not None
    F = np.atleast_2d(F.T).T if F.ndim == 1 else F
    if F.ndim == 1:
        F = F[:, None]
    if F.shape[0] != ndof:
        raise ValueError(f"force vector has {F.shape[0]} entries, expected {ndof}")
    n_cases = F.shape[1]

    free = np.setdiff1d(np.arange(ndof), fixed)
    if free.size == 0:
        raise ValueError("all DOFs are constrained")

    edof = _edof(nelx, nely)
    KE = element_stiffness_q4(nu=nu, E=1.0, thickness=thickness)
    iK = np.kron(edof, np.ones((8, 1), dtype=int)).ravel()
    jK = np.kron(edof, np.ones((1, 8), dtype=int)).ravel()
    KE_flat = KE.ravel()

    H, Hs = _filter_matrix(nelx, nely, rmin)

    # -- design field (column-major internally, (nely, nelx) externally) --
    if x0 is None:
        x = np.full(n_elem, float(volfrac))
    elif np.isscalar(x0):
        x = np.full(n_elem, float(x0))  # type: ignore[arg-type]
    else:
        x = _to_internal(np.asarray(x0, dtype=float), nelx, nely)

    passive_v = None
    if passive is not None:
        passive_v = _to_internal(np.asarray(passive), nelx, nely)
        x = np.where(passive_v == 1, 1.0e-3, np.where(passive_v == 2, 1.0, x))

    xphys = x.copy()
    history: list[dict[str, Any]] = []
    change = 1.0
    c = math.nan
    U = np.zeros((ndof, n_cases))
    it = 0
    converged = False

    for it in range(1, max_iter + 1):
        p = penal
        if continuation:
            p = min(penal, 1.0 + (penal - 1.0) * it / max(1, 0.5 * max_iter))

        # ---- FE analysis ------------------------------------------------
        Ee = Emin + xphys**p * (E0 - Emin)
        sK = (KE_flat[None, :] * Ee[:, None]).ravel()
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        Kff = K[free, :][:, free]
        U = np.zeros((ndof, n_cases))
        sol = spsolve(Kff, F[free, :] if n_cases > 1 else F[free, 0])
        U[free, :] = np.asarray(sol).reshape(free.size, n_cases)

        # ---- compliance and sensitivities --------------------------------
        ce = np.zeros(n_elem)
        for lc in range(n_cases):
            Ue = U[edof, lc]
            ce += np.einsum("ij,jk,ik->i", Ue, KE, Ue)
        c = float(np.sum(Ee * ce))
        dc = -p * xphys ** (p - 1.0) * (E0 - Emin) * ce
        dv = np.ones(n_elem)

        # ---- filtering ----------------------------------------------------
        if rmin > 1.0:
            if filter.lower().startswith("sens"):
                dc = np.asarray(H @ (x * dc)).ravel() / (Hs * np.maximum(x, 1.0e-3))
            else:
                dc = np.asarray(H @ (dc / Hs)).ravel()
                dv = np.asarray(H @ (dv / Hs)).ravel()

        # ---- OC update -----------------------------------------------------
        xnew = _oc_update(x, dc, dv, volfrac, move=move)
        if passive_v is not None:
            xnew = np.where(passive_v == 1, 1.0e-3, np.where(passive_v == 2, 1.0, xnew))
        change = float(np.max(np.abs(xnew - x)))
        x = xnew
        xphys = np.asarray(H @ x).ravel() / Hs if (
            rmin > 1.0 and not filter.lower().startswith("sens")
        ) else x.copy()

        rec = {
            "iteration": it,
            "compliance": c,
            "volume": float(np.mean(xphys)),
            "change": change,
            "penal": p,
        }
        history.append(rec)
        if verbose:
            print(
                f"[simp] it={it:3d} c={c:11.4f} vol={rec['volume']:.3f} ch={change:.4f}"
            )
        if callback is not None:
            callback(it, _to_external(xphys, nelx, nely), c)
        if change < tol and (not continuation or p >= penal):
            converged = True
            break

    return TopologyResult(
        density=_to_external(xphys, nelx, nely),
        compliance=c,
        iterations=it,
        change=change,
        converged=converged,
        volume_fraction=float(np.mean(xphys)),
        history=history,
        displacement=U[:, 0] if n_cases == 1 else U,
        nelx=nelx,
        nely=nely,
        penal=penal,
        rmin=rmin,
    )


def _to_external(v: np.ndarray, nelx: int, nely: int) -> np.ndarray:
    """Column-major element vector -> ``(nely, nelx)`` image."""
    return np.asarray(v, dtype=float).reshape(nelx, nely).T


def _to_internal(a: np.ndarray, nelx: int, nely: int) -> np.ndarray:
    """``(nely, nelx)`` image (or flat vector) -> column-major element vector."""
    a = np.asarray(a)
    if a.ndim == 1:
        if a.size != nelx * nely:
            raise ValueError(f"expected {nelx * nely} elements, got {a.size}")
        return a.astype(float)
    if a.shape != (nely, nelx):
        raise ValueError(f"expected shape {(nely, nelx)}, got {a.shape}")
    return a.T.ravel().astype(float)
