"""Rigid Body Property Extraction (RBPE) from measured FRFs.

Below the first flexible resonance a structure suspended in free-free conditions
responds as a rigid body, so its accelerance matrix flattens out onto the
so-called **mass line**:

.. math::
    A(\\omega) \\;=\\; \\frac{\\ddot X(\\omega)}{F(\\omega)}
      \\;\\longrightarrow\\; T_{out}\\, M_{rb}^{-1}\\, T_{in}^{T},

where :math:`T` maps the 6 rigid-body DOF of a reference point onto the measured
(or excited) sensor directions,

.. math::
    T_i = \\begin{bmatrix} n_i^T & (r_i \\times n_i)^T \\end{bmatrix},
    \\qquad r_i = p_i - p_{ref}.

Fitting the symmetric :math:`6\\times 6` inverse mass matrix to the measured mass
line and inverting it gives the total mass, the centre of gravity and the full
inertia tensor:

.. math::
    M_{rb} = \\begin{bmatrix} m I_3 & -m\\,\\tilde c \\\\
                              m\\,\\tilde c & I_{ref}\\end{bmatrix},
    \\qquad I_{cg} = I_{ref} - m\\left(\\|c\\|^2 I_3 - c c^T\\right).

Two options exist for the departures from that ideal that every real test has.

**Inertia restraint** (``restraint="inertia"``).  The default estimator fits
:math:`M^{-1}` and inverts it, which is exactly where measurement noise does
the most damage: the smallest singular value of the fitted inverse becomes the
largest of the mass matrix.  The inertia-restraint estimator instead *restrains
the measured acceleration field to the rigid-body subspace*,

.. math::
    \\ddot q = T_{out}^{+}\\, \\ddot x,

and imposes Newton-Euler equilibrium on the result, :math:`M \\ddot q = T_{in}^T
F`, which is linear in :math:`M` itself.  No inversion of a fitted quantity is
involved, and the projection discards whatever part of the measured motion is
not rigid-body — the flexible contamination the mass line is most exposed to.
It needs enough sensors to determine all six rigid-body DOF (``rank(T_out) =
6``); the excitation set must still span all six, exactly as for the mass line,
because a rank-deficient set of rigid-body accelerations leaves part of
:math:`M` unobserved however the equations are arranged.  What the projection
buys is immunity to the *non-rigid* part of the measured motion: on data
contaminated by a nearby flexible mode it roughly halves the inertia error.
Against purely random FRF noise it is marginally worse than the mass line,
because the noise then sits in the coefficient matrix rather than the
right-hand side.

**Mounting stiffness** (``mount_k=...``).  A test article hangs on bungees or
sits on air springs, so the support contributes a rigid-body stiffness
:math:`K` and the response is no longer a flat mass line:

.. math::
    A(\\omega) = T_{out}\\left[M - K/\\omega^2\\right]^{-1} T_{in}^T .

The suspension therefore acts as an *apparent negative mass* :math:`K/\\omega^2`
which diverges as the frequency falls — the reason a mass line taken too close
to the suspension modes reads high.  Given ``mount_k`` the fit is done line by
line and the term is removed analytically, which extends the usable band down
towards the suspension frequencies instead of away from them.  With
``mount_k="fit"`` both :math:`M` and :math:`K` are identified together, since
the equilibrium equations stay linear in the pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "RigidBodyProperties",
    "rigid_body_properties",
    "rigid_body_transform",
    "rigid_body_mass_matrix",
    "mount_stiffness_matrix",
    "mass_line",
    "skew",
]


def skew(v: ArrayLike) -> np.ndarray:
    """Skew-symmetric matrix such that ``skew(a) @ b == cross(a, b)``."""
    x, y, z = np.asarray(v, dtype=float).ravel()
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# ----------------------------------------------------------------------
@dataclass
class RigidBodyProperties:
    """Identified rigid-body inertia properties.

    Attributes
    ----------
    mass:
        Total mass.
    cog:
        Centre of gravity relative to ``reference_point`` (add the reference to
        get global coordinates; see :attr:`cog_global`).
    inertia:
        ``3x3`` inertia tensor **about the centre of gravity**.
    inertia_reference:
        ``3x3`` inertia tensor about the reference point.
    mass_matrix:
        Full ``6x6`` rigid-body mass matrix about the reference point.
    principal_moments, principal_axes:
        Eigen-decomposition of :attr:`inertia` (ascending moments; the axes are
        the columns of ``principal_axes``).
    residual:
        Relative Frobenius error of the mass-line fit.
    """

    mass: float
    cog: np.ndarray
    inertia: np.ndarray
    inertia_reference: np.ndarray
    mass_matrix: np.ndarray
    reference_point: np.ndarray
    principal_moments: np.ndarray
    principal_axes: np.ndarray
    residual: float = math.nan
    condition_number: float = math.nan
    band: tuple[float, float] | None = None
    method: str = "massline"
    extras: dict[str, Any] = field(default_factory=dict)

    # -- convenience ----------------------------------------------------
    @property
    def m(self) -> float:
        return self.mass

    @property
    def cg(self) -> np.ndarray:
        return self.cog

    @property
    def center_of_gravity(self) -> np.ndarray:
        return self.cog

    @property
    def cog_global(self) -> np.ndarray:
        return self.cog + self.reference_point

    @property
    def Ixx(self) -> float:
        return float(self.inertia[0, 0])

    @property
    def Iyy(self) -> float:
        return float(self.inertia[1, 1])

    @property
    def Izz(self) -> float:
        return float(self.inertia[2, 2])

    @property
    def products_of_inertia(self) -> tuple[float, float, float]:
        """``(Ixy, Ixz, Iyz)`` of the inertia tensor about the CG."""
        return (
            float(self.inertia[0, 1]),
            float(self.inertia[0, 2]),
            float(self.inertia[1, 2]),
        )

    @property
    def radii_of_gyration(self) -> np.ndarray:
        if self.mass <= 0:
            return np.full(3, math.nan)
        return np.sqrt(np.clip(self.principal_moments / self.mass, 0.0, None))

    def is_physical(self, tol: float = 1.0e-9) -> bool:
        """``True`` when mass > 0 and the CG inertia tensor is positive definite."""
        if not (self.mass > 0):
            return False
        w = np.linalg.eigvalsh(self.inertia)
        if np.min(w) <= tol * max(np.max(np.abs(w)), 1.0):
            return False
        # triangle inequalities of the principal moments
        a, b, c = np.sort(w)
        return bool(a + b >= c * (1.0 - 1e-9))

    def summary(self) -> str:
        ix, iy, iz = self.products_of_inertia
        return (
            f"mass            = {self.mass:.6g}\n"
            f"cog (ref frame) = [{self.cog[0]:.6g}, {self.cog[1]:.6g}, {self.cog[2]:.6g}]\n"
            f"Ixx, Iyy, Izz   = {self.Ixx:.6g}, {self.Iyy:.6g}, {self.Izz:.6g}\n"
            f"Ixy, Ixz, Iyz   = {ix:.6g}, {iy:.6g}, {iz:.6g}\n"
            f"principal       = {np.array2string(self.principal_moments, precision=6)}\n"
            f"fit residual    = {self.residual:.4e}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RigidBodyProperties(mass={self.mass:.6g}, "
            f"cog={np.array2string(self.cog, precision=5)}, "
            f"residual={self.residual:.3e})"
        )


# ----------------------------------------------------------------------
def rigid_body_transform(
    positions: np.ndarray,
    directions: np.ndarray,
    reference_point: ArrayLike = (0.0, 0.0, 0.0),
    *,
    normalize: bool = True,
) -> np.ndarray:
    """``(n, 6)`` matrix mapping rigid-body DOF at a reference point to sensors.

    Row ``i`` is ``[n_i^T, (r_i x n_i)^T]`` where ``r_i = p_i - p_ref``.
    """
    p = np.atleast_2d(np.asarray(positions, dtype=float))
    d = np.atleast_2d(np.asarray(directions, dtype=float))
    if p.shape[1] != 3 or d.shape[1] != 3:
        raise ValueError("positions and directions must have 3 columns")
    if p.shape[0] != d.shape[0]:
        raise ValueError("positions and directions must have the same length")
    ref = np.asarray(reference_point, dtype=float).ravel()
    if ref.size != 3:
        raise ValueError("reference_point must have 3 components")
    if normalize:
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.where(norms > 0, norms, 1.0)
    r = p - ref[None, :]
    return np.hstack([d, np.cross(r, d)])


def rigid_body_mass_matrix(
    mass: float,
    cog: ArrayLike = (0.0, 0.0, 0.0),
    inertia: Any = None,
    *,
    about: str = "cog",
) -> np.ndarray:
    """Assemble the ``6x6`` rigid-body mass matrix about the origin.

    Parameters
    ----------
    mass, cog:
        Total mass and centre of gravity relative to the reference point.
    inertia:
        ``3x3`` tensor, a 3-vector of diagonal moments, or ``(Ixx, Iyy, Izz,
        Ixy, Ixz, Iyz)``.
    about:
        Whether ``inertia`` is given about the ``"cog"`` (default, the parallel
        axis theorem is applied) or about the ``"reference"`` point.
    """
    c = np.asarray(cog, dtype=float).ravel()
    if inertia is None:
        I3 = np.zeros((3, 3))
    else:
        arr = np.asarray(inertia, dtype=float)
        if arr.shape == (3, 3):
            I3 = arr
        elif arr.size == 3:
            I3 = np.diag(arr.ravel())
        elif arr.size == 6:
            ixx, iyy, izz, ixy, ixz, iyz = arr.ravel()
            I3 = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        else:
            raise ValueError("inertia must be 3x3, 3 or 6 values")
    if about.lower() in ("cog", "cg", "center", "centre"):
        I_ref = I3 + mass * (float(c @ c) * np.eye(3) - np.outer(c, c))
    else:
        I_ref = I3
    S = mass * skew(c)
    M = np.zeros((6, 6))
    M[:3, :3] = mass * np.eye(3)
    M[:3, 3:] = -S
    M[3:, :3] = S
    M[3:, 3:] = I_ref
    return M


def mount_stiffness_matrix(
    mounts: Any, reference_point: ArrayLike = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """Rigid-body ``6x6`` stiffness of a suspension, about ``reference_point``.

    Each mount contributes :math:`k_j t_j t_j^T` with :math:`t_j = [n_j^T,
    (r_j \\times n_j)^T]`, i.e. exactly the same rigid-body transform that maps
    the body DOF onto a sensor direction.

    Parameters
    ----------
    mounts:
        One of

        * a scalar — three translational springs of that rate at the reference
          point, ``diag(k, k, k, 0, 0, 0)``;
        * a 3- or 6-vector — the diagonal of the rigid-body stiffness;
        * a ``6x6`` matrix — used as it is (symmetrised);
        * a sequence of ``{"position": ..., "direction": ..., "stiffness": ...}``
          dicts, or a sequence of ``(position, direction, stiffness)`` tuples —
          the physical description of a real suspension.
    reference_point:
        Point the matrix is expressed about.

    Examples
    --------
    Two vertical springs one metre either side of the origin: they carry the
    vertical translation and, through their offset, the roll about ``x``.

    >>> import numpy as np
    >>> from femtools.rbpe import mount_stiffness_matrix
    >>> K = mount_stiffness_matrix(
    ...     [((0.0, 1.0, 0.0), (0, 0, 1), 500.0), ((0.0, -1.0, 0.0), (0, 0, 1), 500.0)]
    ... )
    >>> np.round(np.diag(K), 6)
    array([   0.,    0., 1000., 1000.,    0.,    0.])
    """
    ref = np.asarray(reference_point, dtype=float).ravel()
    if isinstance(mounts, (int, float, np.floating, np.integer)):
        return np.diag([float(mounts)] * 3 + [0.0] * 3)
    # A physical mount list is ragged (3-vector, 3-vector, scalar) or holds
    # dicts, so it cannot be coerced to a float array; anything that can be is
    # one of the numeric spellings of the matrix itself.
    try:
        num = np.asarray(mounts, dtype=float)
    except (TypeError, ValueError):
        num = None
    if num is not None:
        if num.size == 0:
            return np.zeros((6, 6))
        if num.size == 1:
            return np.diag([float(num.ravel()[0])] * 3 + [0.0] * 3)
        if num.shape == (6, 6):
            return 0.5 * (num + num.T)
        if num.ndim == 1 and num.size == 3:
            return np.diag([*num.tolist(), 0.0, 0.0, 0.0])
        if num.ndim == 1 and num.size == 6:
            return np.diag(num)
        raise ValueError(
            f"cannot interpret a numeric mount specification of shape {num.shape}; "
            "expected a scalar, a 3- or 6-vector diagonal, or a 6x6 matrix"
        )

    K = np.zeros((6, 6))
    for item in mounts:
        if isinstance(item, dict):
            pos = item.get("position", item.get("xyz"))
            direction = item.get("direction", item.get("dir"))
            k = item.get("stiffness", item.get("k"))
        else:
            seq = list(item)
            if len(seq) != 3:
                raise ValueError(f"cannot interpret mount {item!r}")
            pos, direction, k = seq
        if pos is None or direction is None or k is None:
            raise ValueError(f"mount {item!r} needs a position, direction and stiffness")
        t = rigid_body_transform(
            np.asarray(pos, dtype=float).reshape(1, 3),
            np.asarray(direction, dtype=float).reshape(1, 3),
            ref,
        )[0]
        K += float(k) * np.outer(t, t)
    return K


def _length_scale(T: np.ndarray) -> float:
    """Characteristic moment arm of a rigid-body transform.

    The translation and rotation halves of a rigid-body DOF vector carry
    different units, so a least-squares fit that mixes them silently weights
    metres against radians.  Rescaling the rotations by this length makes the
    two halves comparable, which is the difference between a well-conditioned
    normal matrix and one whose condition number is the square of the model's
    size in millimetres.
    """
    trans = float(np.linalg.norm(T[:, :3]))
    rot = float(np.linalg.norm(T[:, 3:]))
    if trans <= 0 or rot <= 0:
        return 1.0
    return rot / trans


def _sym_basis() -> list[np.ndarray]:
    """The 21 symmetric ``6x6`` basis matrices, upper-triangle order."""
    basis = []
    for a, b in zip(*np.triu_indices(6), strict=True):
        E = np.zeros((6, 6))
        E[a, b] = 1.0
        E[b, a] = 1.0
        basis.append(E)
    return basis


def _from_upper(x: np.ndarray) -> np.ndarray:
    M = np.zeros((6, 6))
    iu = np.triu_indices(6)
    M[iu] = x
    return M + M.T - np.diag(np.diag(M))


def _fit_mass_restrained(
    T_out: np.ndarray,
    T_in: np.ndarray,
    A_lines: np.ndarray,
    omega: np.ndarray,
    *,
    stiffness: np.ndarray | None = None,
    fit_stiffness: bool = False,
    rcond: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray | None, float, float]:
    """Inertia-restraint fit of ``M`` (and optionally ``K``).

    ``A_lines`` is ``(n_out, n_in, n_line)`` real accelerance.  The measured
    accelerations are restrained to the rigid-body subspace, ``qdd =
    pinv(T_out) @ A``, and Newton-Euler equilibrium ``(M - K/w^2) qdd = T_in^T``
    is imposed in the least-squares sense over the 21 (or 42) unknowns.
    """
    tol = 1e-10 * max(float(np.linalg.norm(T_out)), 1.0)
    if np.linalg.matrix_rank(T_out, tol=tol) < 6:
        raise ValueError(
            "restraint='inertia' needs responses spanning all six rigid-body DOF; "
            "the sensor set gives rank(T_out) < 6"
        )
    basis = _sym_basis()
    n_in = T_in.shape[0]
    n_line = A_lines.shape[2]
    n_unk = 21 * (2 if fit_stiffness else 1)
    rows = 6 * n_in * n_line
    if rows < n_unk:
        raise ValueError(
            f"{n_in} inputs over {n_line} lines give {rows} equations for {n_unk} "
            "unknowns; add inputs, widen the band, or drop mount_k='fit'"
        )

    # Work in scaled DOF (rotations multiplied by a characteristic length) so
    # the six equilibrium equations carry comparable magnitudes.
    L = _length_scale(T_out)
    Sinv = np.diag([1.0, 1.0, 1.0, 1.0 / L, 1.0 / L, 1.0 / L])
    S = np.diag([1.0, 1.0, 1.0, L, L, L])
    T_out = T_out @ Sinv
    T_in = T_in @ Sinv
    if stiffness is not None:
        stiffness = Sinv @ stiffness @ Sinv

    Tp = np.linalg.pinv(T_out)
    B = np.zeros((rows, n_unk))
    rhs = np.zeros(rows)
    Tin_T = T_in.T  # (6, n_in)
    r = 0
    for line in range(n_line):
        Q = Tp @ A_lines[:, :, line]  # (6, n_in) rigid-body accelerations
        block = slice(r, r + 6 * n_in)
        for u, E in enumerate(basis):
            B[block, u] = (E @ Q).ravel()
        target = Tin_T.copy()
        if fit_stiffness:
            inv_w2 = 1.0 / max(omega[line] ** 2, 1e-300)
            for u, E in enumerate(basis):
                B[block, 21 + u] = (-inv_w2 * (E @ Q)).ravel()
        elif stiffness is not None:
            target = target + (stiffness @ Q) / max(omega[line] ** 2, 1e-300)
        rhs[block] = target.ravel()
        r += 6 * n_in

    sv = np.linalg.svd(B, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    x, *_ = np.linalg.lstsq(B, rhs, rcond=rcond)
    resid = float(np.linalg.norm(B @ x - rhs)) / max(float(np.linalg.norm(rhs)), 1e-300)
    M = S @ _from_upper(x[:21]) @ S
    K = S @ _from_upper(x[21:]) @ S if fit_stiffness else None
    return M, K, resid, cond


def _decompose(M6: np.ndarray, reference_point: np.ndarray) -> dict[str, Any]:
    """Split a ``6x6`` rigid-body mass matrix into mass / CG / inertia."""
    M = 0.5 * (np.asarray(M6, dtype=float) + np.asarray(M6, dtype=float).T)
    m = float(np.trace(M[:3, :3]) / 3.0)
    if m <= 0:
        raise ValueError(f"identified mass is not positive ({m:g})")
    # S = m * skew(c) using the antisymmetric part of the coupling blocks
    S = 0.5 * (M[3:, :3] - M[:3, 3:])
    S = 0.5 * (S - S.T)
    c = np.array([S[2, 1], S[0, 2], S[1, 0]]) / m
    I_ref = M[3:, 3:]
    I_cg = I_ref - m * (float(c @ c) * np.eye(3) - np.outer(c, c))
    I_cg = 0.5 * (I_cg + I_cg.T)
    w, V = np.linalg.eigh(I_cg)
    return {
        "mass": m,
        "cog": c,
        "inertia": I_cg,
        "inertia_reference": I_ref,
        "mass_matrix": M,
        "reference_point": reference_point,
        "principal_moments": w,
        "principal_axes": V,
    }


# ----------------------------------------------------------------------
def _as_frf(frf: Any) -> np.ndarray:
    if not isinstance(frf, np.ndarray):
        for attr in ("H", "frf", "data", "values"):
            if hasattr(frf, attr):
                frf = getattr(frf, attr)
                break
    H = np.asarray(frf)
    if H.dtype.kind != "c":
        H = H.astype(complex)
    if H.ndim == 2:
        H = H[:, None, :]
    if H.ndim != 3:
        raise ValueError(f"FRF must be (n_out, n_in, n_freq), got {H.shape}")
    return H


def _to_accelerance(H: np.ndarray, f: np.ndarray, frf_type: str) -> np.ndarray:
    w = 2.0 * math.pi * f
    kind = frf_type.lower()
    if kind in ("accelerance", "inertance", "acceleration", "a"):
        return H
    if kind in ("mobility", "velocity", "v"):
        return H * (1j * w)[None, None, :]
    if kind in ("receptance", "compliance", "displacement", "d"):
        return H * (-(w**2))[None, None, :]
    if kind in ("apparent_mass", "dynamic_mass"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / H
    raise ValueError(f"unknown frf_type {frf_type!r}")


def _auto_band(
    A: np.ndarray, f: np.ndarray, *, flatness_tol: float = 0.02, min_lines: int = 6
) -> tuple[float, float]:
    """Pick the flattest low-frequency band, i.e. the rigid-body mass line.

    Flexible modes contaminate the accelerance plateau roughly as
    :math:`(\\omega/\\omega_1)^2`, so the band is extended upwards only while the
    real part stays within ``flatness_tol`` of its low-frequency value.
    """
    pos = np.nonzero(f > 0)[0]
    if pos.size == 0:  # pragma: no cover
        return float(f[0]), float(f[-1])
    from scipy.ndimage import uniform_filter1d

    fp = f[pos]
    Ar = np.real(A[:, :, pos])
    # Smooth along frequency first: random measurement noise must not truncate
    # the band, whereas the flexible contamination we want to avoid is a smooth
    # upward trend that survives averaging.
    win = max(3, min(11, fp.size // 20) | 1)
    Ars = uniform_filter1d(Ar, size=win, axis=2, mode="nearest") if fp.size > win else Ar
    n_ref = max(3, min(fp.size // 50, 10))
    ref = np.mean(Ars[:, :, :n_ref], axis=2)
    scale = max(float(np.linalg.norm(ref)), 1e-300)
    dev = np.linalg.norm(Ars - ref[:, :, None], axis=(0, 1)) / scale
    # Adaptive floor: never cut the band tighter than the residual noise scatter.
    if fp.size > 4:
        scatter = float(np.median(np.abs(np.diff(dev[: max(4, fp.size // 10)]))))
    else:  # pragma: no cover
        scatter = 0.0
    threshold = max(flatness_tol, 5.0 * scatter)
    ok = dev < threshold
    k = int(np.argmax(~ok)) if np.any(~ok) else fp.size
    k = max(k, min(min_lines, fp.size))
    return float(fp[0]), float(fp[k - 1])


def mass_line(
    frf: Any,
    freq_hz: ArrayLike,
    *,
    band: tuple[float, float] | None = None,
    frf_type: str = "accelerance",
    statistic: str = "mean",
    flatness_tol: float = 0.02,
) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """Extract the (real) rigid-body mass line from an FRF matrix.

    Returns ``(A_massline, band, line_indices)`` where ``A_massline`` has shape
    ``(n_out, n_in)``.
    """
    H = _as_frf(frf)
    f = np.asarray(freq_hz, dtype=float).ravel()
    if f.size != H.shape[2]:
        raise ValueError(f"freq_hz has {f.size} lines but FRF has {H.shape[2]}")
    A = _to_accelerance(H, f, frf_type)
    if band is None:
        band = _auto_band(A, f, flatness_tol=flatness_tol)
    sel = np.nonzero((f >= band[0]) & (f <= band[1]) & np.isfinite(f))[0]
    if sel.size == 0:
        raise ValueError(f"band {band} selects no spectral lines")
    sub = np.real(A[:, :, sel])
    if statistic == "median":
        vals = np.median(sub, axis=2)
    elif statistic == "mean":
        vals = np.mean(sub, axis=2)
    elif statistic in ("lowest", "first"):
        vals = sub[:, :, 0]
    else:
        raise ValueError(f"unknown statistic {statistic!r}")
    return vals, (float(band[0]), float(band[1])), sel


def _sensor_arrays(
    spec: Any, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a sensor specification into ``(positions, directions)``."""
    if spec is None:
        raise ValueError(f"`{name}` must be given (positions + directions)")
    if isinstance(spec, tuple) and len(spec) == 2:
        pos, dirs = spec
        return (
            np.atleast_2d(np.asarray(pos, dtype=float)),
            np.atleast_2d(np.asarray(dirs, dtype=float)),
        )
    if isinstance(spec, np.ndarray) and spec.ndim == 2 and spec.shape[1] == 6:
        return spec[:, :3].astype(float), spec[:, 3:].astype(float)
    pos_l: list[Any] = []
    dir_l: list[Any] = []
    for item in spec:
        if isinstance(item, dict):
            pos_l.append(item.get("position", item.get("xyz")))
            d = item.get("direction", item.get("dir"))
            if d is None and "dof" in item:
                d = np.eye(3)[int(item["dof"]) % 3]
            dir_l.append(d)
        else:
            arr = np.asarray(item, dtype=float).ravel()
            if arr.size == 6:
                pos_l.append(arr[:3])
                dir_l.append(arr[3:])
            elif len(item) == 2:
                pos_l.append(item[0])
                dir_l.append(item[1])
            else:
                raise ValueError(f"cannot interpret {name} entry {item!r}")
    return (
        np.atleast_2d(np.asarray(pos_l, dtype=float)),
        np.atleast_2d(np.asarray(dir_l, dtype=float)),
    )


def _fit_inverse_mass(
    T_out: np.ndarray, T_in: np.ndarray, A: np.ndarray, *, rcond: float = 1e-12
) -> tuple[np.ndarray, float, float]:
    """Least-squares symmetric ``N = M^-1`` such that ``T_out N T_in^T ~= A``.

    The 21 independent entries of the symmetric ``6x6`` are the unknowns.
    """
    n_out, n_in = A.shape
    iu = np.triu_indices(6)
    n_unk = iu[0].size  # 21
    B = np.zeros((n_out * n_in, n_unk))
    rhs = np.zeros(n_out * n_in)
    row = 0
    for o in range(n_out):
        to = T_out[o]
        for i in range(n_in):
            ti = T_in[i]
            outer = np.outer(to, ti)
            sym = outer + outer.T
            coef = sym[iu]
            # diagonal terms are counted once, not twice
            coef[iu[0] == iu[1]] = np.diag(outer)
            B[row] = coef
            rhs[row] = A[o, i]
            row += 1
    sv = np.linalg.svd(B, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    x, *_ = np.linalg.lstsq(B, rhs, rcond=rcond)
    N = np.zeros((6, 6))
    N[iu] = x
    N = N + N.T - np.diag(np.diag(N))
    resid = float(np.linalg.norm(B @ x - rhs)) / max(float(np.linalg.norm(rhs)), 1e-300)
    return N, resid, cond


def rigid_body_properties(
    frf: Any = None,
    freq_hz: ArrayLike | None = None,
    *,
    sensors: Any = None,
    inputs: Any = None,
    positions: np.ndarray | None = None,
    directions: np.ndarray | None = None,
    input_positions: np.ndarray | None = None,
    input_directions: np.ndarray | None = None,
    reference_point: Any = (0.0, 0.0, 0.0),
    band: tuple[float, float] | None = None,
    frf_type: str = "accelerance",
    method: str = "massline",
    restraint: str | None = None,
    mount_k: Any = None,
    statistic: str = "mean",
    flatness_tol: float = 0.02,
    mass_matrix: np.ndarray | None = None,
    inverse_mass_matrix: np.ndarray | None = None,
    rcond: float = 1.0e-12,
) -> RigidBodyProperties:
    """Identify mass, centre of gravity and inertia tensor from FRF mass lines.

    Parameters
    ----------
    frf:
        FRF matrix ``(n_out, n_in, n_freq)`` measured in free-free conditions.
        Objects exposing ``.H``/``.frf`` are unwrapped.
    freq_hz:
        Frequency axis; read from the FRF object when omitted.
    sensors, inputs:
        Response and excitation DOF definitions.  Each accepts
        ``(positions, directions)`` as a tuple of ``(n, 3)`` arrays, an
        ``(n, 6)`` array of ``[x, y, z, nx, ny, nz]`` rows, or a list of
        ``{"position": ..., "direction": ...}`` dicts.  ``inputs`` defaults to
        ``sensors`` restricted to the first ``n_in`` entries (driving points).
    reference_point:
        Point about which the ``6x6`` mass matrix is expressed: a 3-vector,
        ``"origin"``, or ``"centroid"`` (mean sensor position).
    band:
        Frequency band of the rigid-body mass line.  When omitted it is detected
        automatically as the flattest low-frequency plateau (see
        ``flatness_tol``).
    flatness_tol:
        Relative deviation of the real part still accepted inside the
        auto-detected mass line band (default 2 %).
    frf_type:
        ``"accelerance"`` (default), ``"mobility"``, ``"receptance"`` or
        ``"apparent_mass"``.
    method:
        ``"massline"`` averages the real part over the band and fits once;
        ``"band"`` fits all selected spectral lines simultaneously.  A
        ``mount_k`` correction always implies ``"band"``, since the apparent
        mass then depends on frequency.
    restraint:
        ``None``/``"none"`` (default) fits the inverse mass matrix and inverts
        it; ``"inertia"`` uses the inertia-restraint estimator, which projects
        the measured accelerations onto the rigid-body subspace and imposes
        Newton-Euler equilibrium, giving ``M`` directly (see the module
        docstring).  Prefer it when the data is noisy or slightly flexible, and
        when there are more sensors than the six DOF strictly need.
    mount_k:
        Rigid-body stiffness of the suspension, in any form accepted by
        :func:`mount_stiffness_matrix`, or the string ``"fit"`` to identify it
        together with the inertia (``restraint="inertia"`` only).  The
        identified stiffness and the resulting suspension frequencies are
        returned in ``extras``.
    mass_matrix / inverse_mass_matrix:
        Skip the FRF stage and decompose a known ``6x6`` matrix instead (useful
        for verification against an FE model).

    Returns
    -------
    RigidBodyProperties

    Notes
    -----
    The inverse mass matrix has 21 independent entries, so the excitation set
    must span all six rigid-body DOF: at least 6 independent input directions
    (and enough responses) are required, otherwise the fit is rank deficient and
    the reported ``condition_number`` explodes.  The inertia-restraint estimator
    needs that same span on the input side *and* ``rank(T_out) = 6`` on the
    response side, since it solves ``M Q = T_in^T`` from ``6 * n_in``
    equilibrium equations per spectral line and a rank-deficient ``Q`` leaves
    part of ``M`` unobserved.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.rbpe import rigid_body_properties, rigid_body_mass_matrix
    >>> M = rigid_body_mass_matrix(12.5, (0.1, -0.05, 0.2), (0.8, 1.1, 1.5))
    >>> p = rigid_body_properties(mass_matrix=M)
    >>> round(p.mass, 9), np.round(p.cog, 9)
    (12.5, array([ 0.1 , -0.05,  0.2 ]))

    The same body hung on soft mounts and measured from 6 to 15 Hz, close
    enough to the suspension modes that the mass line reads high.  Declaring
    the mounts removes the apparent negative mass exactly:

    >>> from femtools.rbpe import mount_stiffness_matrix, rigid_body_transform
    >>> pos = np.repeat([[sx, sy, sz] for sx in (-0.5, 0.5)
    ...                  for sy in (-0.4, 0.4) for sz in (-0.3, 0.3)], 3, axis=0)
    >>> dirs = np.tile(np.eye(3), (8, 1))
    >>> mounts = [((x, y, -0.3), (0, 0, 1), 900.0)
    ...           for x in (-0.5, 0.5) for y in (-0.4, 0.4)]
    >>> K = mount_stiffness_matrix(mounts)
    >>> To = rigid_body_transform(pos, dirs, (0, 0, 0))
    >>> Ti = To[[0, 4, 8, 13, 17, 22]]
    >>> f = np.linspace(6.0, 15.0, 21)
    >>> w = 2 * np.pi * f
    >>> H = np.stack([-(wk**2) * (To @ np.linalg.solve(K - wk**2 * M, Ti.T))
    ...               for wk in w], axis=2)
    >>> args = dict(sensors=(pos, dirs), inputs=(pos[[0, 4, 8, 13, 17, 22]],
    ...             dirs[[0, 4, 8, 13, 17, 22]]), band=(6.0, 15.0))
    >>> round(rigid_body_properties(H, f, **args).mass, 3)          # mounts ignored
    12.14
    >>> round(rigid_body_properties(H, f, mount_k=mounts, **args).mass, 9)
    12.5
    """
    if isinstance(reference_point, str):
        key = reference_point.lower()
        if key in ("origin", "global"):
            ref = np.zeros(3)
        elif key in ("centroid", "mean"):
            if sensors is not None:
                pos_tmp = _sensor_arrays(sensors, "sensors")[0]
            elif positions is not None:
                pos_tmp = np.atleast_2d(np.asarray(positions, dtype=float))
            else:
                raise ValueError('reference_point="centroid" needs sensor positions')
            ref = np.mean(np.asarray(pos_tmp, dtype=float), axis=0)
        else:
            raise ValueError(f"unknown reference_point {reference_point!r}")
    else:
        ref = np.asarray(reference_point, dtype=float).ravel()
        if ref.size != 3:
            raise ValueError("reference_point must have 3 components")

    # ---- direct matrix decomposition ---------------------------------
    if mass_matrix is not None or inverse_mass_matrix is not None:
        M6 = (
            np.asarray(mass_matrix, dtype=float)
            if mass_matrix is not None
            else np.linalg.inv(np.asarray(inverse_mass_matrix, dtype=float))
        )
        if M6.shape != (6, 6):
            raise ValueError(f"mass matrix must be 6x6, got {M6.shape}")
        parts = _decompose(M6, ref)
        return RigidBodyProperties(**parts, residual=0.0, condition_number=1.0, method="matrix")

    # ---- FRF route -----------------------------------------------------
    if frf is None:
        raise ValueError("either `frf` or `mass_matrix` must be given")
    if freq_hz is None:
        for attr in ("freq_hz", "frequencies"):
            carried = getattr(frf, attr, None)
            if carried is not None:
                freq_hz = carried
                break
    if freq_hz is None:
        raise ValueError("freq_hz must be given (or carried by the FRF object)")

    if sensors is not None:
        pos, dirs = _sensor_arrays(sensors, "sensors")
    elif positions is not None and directions is not None:
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        dirs = np.atleast_2d(np.asarray(directions, dtype=float))
    else:
        raise ValueError("sensor positions and directions are required")

    H = _as_frf(frf)
    n_out, n_in, _ = H.shape
    if pos.shape[0] != n_out:
        raise ValueError(f"{pos.shape[0]} sensors given but FRF has {n_out} outputs")

    if inputs is not None:
        ipos, idirs = _sensor_arrays(inputs, "inputs")
    elif input_positions is not None and input_directions is not None:
        ipos = np.atleast_2d(np.asarray(input_positions, dtype=float))
        idirs = np.atleast_2d(np.asarray(input_directions, dtype=float))
    else:
        ipos, idirs = pos[:n_in], dirs[:n_in]
    if ipos.shape[0] != n_in:
        raise ValueError(f"{ipos.shape[0]} inputs given but FRF has {n_in} inputs")

    T_out = rigid_body_transform(pos, dirs, ref)
    T_in = rigid_body_transform(ipos, idirs, ref)

    # ---- support condition ---------------------------------------------
    fit_stiffness = isinstance(mount_k, str) and mount_k.strip().lower() == "fit"
    K6: np.ndarray | None = None
    if mount_k is not None and not fit_stiffness:
        K6 = mount_stiffness_matrix(mount_k, ref)
        if K6.shape != (6, 6):  # pragma: no cover - guarded in the helper
            raise ValueError(f"mount stiffness must be 6x6, got {K6.shape}")

    rkey = "none" if restraint is None else str(restraint).strip().lower()
    if rkey in ("none", "massline", "inverse", "inverse-mass"):
        rkey = "none"
    elif rkey in ("inertia", "inertia-restraint", "inertia_restraint", "irm", "restrained"):
        rkey = "inertia"
    else:
        raise ValueError(
            f"unknown restraint {restraint!r}; expected None/'none' or 'inertia'"
        )
    if fit_stiffness and rkey != "inertia":
        raise ValueError("mount_k='fit' requires restraint='inertia'")

    A_line, band_used, sel = mass_line(
        H, freq_hz, band=band, frf_type=frf_type, statistic=statistic,
        flatness_tol=flatness_tol,
    )

    mkey = method.lower()
    if mkey in ("massline", "mean", "average"):
        mkey = "massline"
    elif mkey in ("band", "lines", "all"):
        mkey = "band"
    else:
        raise ValueError(f"unknown method {method!r}")
    # The apparent mass M - K/w^2 is frequency dependent, so a suspension
    # correction can only be applied line by line.
    per_line = mkey == "band" or K6 is not None or fit_stiffness

    f = np.asarray(freq_hz, dtype=float).ravel()
    A_all = np.real(_to_accelerance(H, f, frf_type)[:, :, sel])
    w_sel = 2.0 * math.pi * f[sel]
    if per_line and np.any(w_sel <= 0):
        keep = w_sel > 0
        A_all, w_sel, sel = A_all[:, :, keep], w_sel[keep], sel[keep]
        if w_sel.size == 0:
            raise ValueError("a mounting-stiffness correction needs non-zero frequencies")

    extras: dict[str, Any] = {"T_out": T_out, "T_in": T_in}
    K_fitted: np.ndarray | None = None

    if rkey == "inertia":
        lines = A_all if per_line else A_line[:, :, None]
        omegas = w_sel if per_line else np.ones(1)
        M6, K_fitted, resid, cond = _fit_mass_restrained(
            T_out, T_in, lines, omegas,
            stiffness=K6, fit_stiffness=fit_stiffness, rcond=rcond,
        )
        N = np.linalg.pinv(M6)
    else:
        if per_line:
            Ms = []
            for k in range(A_all.shape[2]):
                Nk, _, _ = _fit_inverse_mass(T_out, T_in, A_all[:, :, k], rcond=rcond)
                M_eff = np.linalg.pinv(Nk)
                if K6 is not None:
                    M_eff = M_eff + K6 / w_sel[k] ** 2
                Ms.append(M_eff)
            M6 = np.mean(Ms, axis=0)
            M6 = 0.5 * (M6 + M6.T)
            N = np.linalg.pinv(M6)
            pred = np.stack(
                [
                    T_out @ np.linalg.pinv(M6 - (0.0 if K6 is None else K6 / wk**2)) @ T_in.T
                    for wk in w_sel
                ],
                axis=2,
            )
            resid = float(np.linalg.norm(pred - A_all)) / max(
                float(np.linalg.norm(A_all)), 1e-300
            )
            _, _, cond = _fit_inverse_mass(T_out, T_in, A_line, rcond=rcond)
        else:
            N, resid, cond = _fit_inverse_mass(T_out, T_in, A_line, rcond=rcond)
            try:
                M6 = np.linalg.inv(N)
            except np.linalg.LinAlgError as exc:  # pragma: no cover
                raise RuntimeError(
                    "the identified inverse mass matrix is singular; the excitation "
                    "set probably does not span all six rigid-body DOF"
                ) from exc

    K_used = K_fitted if K_fitted is not None else K6
    if K_used is not None:
        extras["mount_stiffness"] = K_used
        extras["suspension_hz"] = _suspension_frequencies(K_used, M6)
    extras["inverse_mass_matrix"] = N

    parts = _decompose(M6, ref)
    tag = mkey if rkey == "none" else f"{mkey}+inertia"
    if fit_stiffness:
        tag += "+fitk"
    elif K6 is not None:
        tag += "+mount"
    return RigidBodyProperties(
        **parts,
        residual=resid,
        condition_number=cond,
        band=band_used,
        method=tag,
        extras=extras,
    )


def _suspension_frequencies(K: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Rigid-body-on-mounts frequencies in Hz, ascending (zeros for free DOF)."""
    from scipy.linalg import eigh

    try:
        w = eigh(0.5 * (K + K.T), 0.5 * (M + M.T), eigvals_only=True)
    except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - defensive
        return np.full(6, math.nan)
    return np.sqrt(np.clip(np.asarray(w, dtype=float), 0.0, None)) / (2.0 * math.pi)
