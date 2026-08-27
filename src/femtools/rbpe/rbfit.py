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
        ``"band"`` fits all selected spectral lines simultaneously.
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
    the reported ``condition_number`` explodes.

    Examples
    --------
    >>> import numpy as np
    >>> from femtools.rbpe import rigid_body_properties, rigid_body_mass_matrix
    >>> M = rigid_body_mass_matrix(12.5, (0.1, -0.05, 0.2), (0.8, 1.1, 1.5))
    >>> p = rigid_body_properties(mass_matrix=M)
    >>> round(p.mass, 9), np.round(p.cog, 9)
    (12.5, array([ 0.1 , -0.05,  0.2 ]))
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
        freq_hz = getattr(frf, "freq_hz", None) or getattr(frf, "frequencies", None)
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

    A_line, band_used, sel = mass_line(
        H, freq_hz, band=band, frf_type=frf_type, statistic=statistic,
        flatness_tol=flatness_tol,
    )

    if method.lower() in ("massline", "mean", "average"):
        N, resid, cond = _fit_inverse_mass(T_out, T_in, A_line, rcond=rcond)
    elif method.lower() in ("band", "lines", "all"):
        f = np.asarray(freq_hz, dtype=float).ravel()
        A_all = np.real(_to_accelerance(H, f, frf_type)[:, :, sel])
        Ns = []
        for k in range(A_all.shape[2]):
            Nk, _, _ = _fit_inverse_mass(T_out, T_in, A_all[:, :, k], rcond=rcond)
            Ns.append(Nk)
        N = np.mean(Ns, axis=0)
        pred = T_out @ N @ T_in.T
        resid = float(np.linalg.norm(pred[:, :, None] - A_all)) / max(
            float(np.linalg.norm(A_all)), 1e-300
        )
        _, _, cond = _fit_inverse_mass(T_out, T_in, A_line, rcond=rcond)
    else:
        raise ValueError(f"unknown method {method!r}")

    try:
        M6 = np.linalg.inv(N)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise RuntimeError(
            "the identified inverse mass matrix is singular; the excitation set "
            "probably does not span all six rigid-body DOF"
        ) from exc

    parts = _decompose(M6, ref)
    return RigidBodyProperties(
        **parts,
        residual=resid,
        condition_number=cond,
        band=band_used,
        method=method.lower(),
        extras={"inverse_mass_matrix": N, "T_out": T_out, "T_in": T_in},
    )
