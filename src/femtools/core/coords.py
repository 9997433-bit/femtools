"""Local coordinate systems (cartesian, cylindrical, spherical).

Conventions (matching common FE practice, e.g. Nastran CORD2R/C/S):

* A :class:`CoordSys` is defined by an ``origin`` and a ``rotation`` matrix
  whose *columns* are the local x/y/z axes expressed in the global frame.
* Cylindrical coordinates are ``(r, theta, z)`` with ``theta`` in **degrees**
  measured from the local x-axis about the local z-axis.
* Spherical coordinates are ``(r, theta, phi)`` with ``theta`` the polar
  angle in degrees measured from the local z-axis and ``phi`` the azimuth in
  degrees from the local x-axis in the local xy-plane.
* Vector transforms are evaluated *at a point*: for curvilinear systems the
  physical basis (e_r, e_theta, ...) depends on where the vector is attached.
  On the axis (r == 0) the basis degenerates; the local cartesian axes are
  used there as a documented fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["CoordSys", "CoordSysType"]

CoordSysType = Literal["cartesian", "cylindrical", "spherical"]

_VALID_TYPES: tuple[str, ...] = ("cartesian", "cylindrical", "spherical")


def _as_unit(v: ArrayLike, name: str) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-300:
        raise ValueError(f"{name} vector must be non-zero")
    return a / n


@dataclass
class CoordSys:
    """A local coordinate system.

    Attributes
    ----------
    id:
        Integer label (0 is reserved for the global system).
    type:
        ``"cartesian"``, ``"cylindrical"`` or ``"spherical"``.
    origin:
        Origin in global cartesian coordinates, shape ``(3,)``.
    rotation:
        Orthonormal ``(3, 3)`` matrix; columns are the local x, y, z axes in
        the global frame.
    """

    id: int
    type: CoordSysType = "cartesian"
    origin: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    rotation: NDArray[np.float64] = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        if self.type not in _VALID_TYPES:
            raise ValueError(
                f"invalid coordinate system type {self.type!r}; expected {_VALID_TYPES}"
            )
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        err = float(np.abs(self.rotation.T @ self.rotation - np.eye(3)).max())
        if err > 1e-8:
            raise ValueError(
                f"rotation matrix of CoordSys {self.id} is not orthonormal (err={err:.2e})"
            )

    # -- constructors ----------------------------------------------------
    @classmethod
    def global_system(cls) -> CoordSys:
        return cls(id=0)

    @classmethod
    def from_axes(
        cls,
        id: int,
        origin: ArrayLike = (0.0, 0.0, 0.0),
        zaxis: ArrayLike = (0.0, 0.0, 1.0),
        xz_vector: ArrayLike = (1.0, 0.0, 0.0),
        type: CoordSysType = "cartesian",
    ) -> CoordSys:
        """Build from an origin, a z-axis direction, and a vector in the +xz half-plane."""
        ez = _as_unit(zaxis, "zaxis")
        xz = np.asarray(xz_vector, dtype=float).reshape(3)
        ex = xz - (xz @ ez) * ez
        n = float(np.linalg.norm(ex))
        if n < 1e-12:
            raise ValueError("xz_vector is parallel to the z-axis; cannot orient the x-axis")
        ex /= n
        ey = np.cross(ez, ex)
        rot = np.column_stack([ex, ey, ez])
        return cls(
            id=id, type=type, origin=np.asarray(origin, dtype=float).reshape(3), rotation=rot
        )

    @classmethod
    def from_points(
        cls,
        id: int,
        a: ArrayLike,
        b: ArrayLike,
        c: ArrayLike,
        type: CoordSysType = "cartesian",
    ) -> CoordSys:
        """Nastran CORD2R/C/S style: A = origin, B = point on the +z axis,
        C = point in the +xz half-plane (all in global coordinates)."""
        a = np.asarray(a, dtype=float).reshape(3)
        b = np.asarray(b, dtype=float).reshape(3)
        c = np.asarray(c, dtype=float).reshape(3)
        return cls.from_axes(id=id, origin=a, zaxis=b - a, xz_vector=c - a, type=type)

    # -- point transforms --------------------------------------------------
    def to_global(self, coords: ArrayLike) -> NDArray[np.float64]:
        """Local coordinates -> global cartesian.

        ``coords`` has shape ``(3,)`` or ``(n, 3)`` and is interpreted per
        ``self.type`` (angles in degrees for curvilinear systems).
        """
        arr = np.asarray(coords, dtype=float)
        single = arr.ndim == 1
        pts = np.atleast_2d(arr).astype(float)
        if pts.shape[1] != 3:
            raise ValueError(f"expected (..., 3) coordinates, got shape {arr.shape}")
        if self.type == "cartesian":
            local_cart = pts
        elif self.type == "cylindrical":
            r, th, z = pts[:, 0], np.deg2rad(pts[:, 1]), pts[:, 2]
            local_cart = np.column_stack([r * np.cos(th), r * np.sin(th), z])
        else:  # spherical
            r, th, ph = pts[:, 0], np.deg2rad(pts[:, 1]), np.deg2rad(pts[:, 2])
            st = np.sin(th)
            local_cart = np.column_stack([r * st * np.cos(ph), r * st * np.sin(ph), r * np.cos(th)])
        out = local_cart @ self.rotation.T + self.origin
        return out[0] if single else out

    def to_local(self, xyz: ArrayLike) -> NDArray[np.float64]:
        """Global cartesian -> local coordinates (angles in degrees)."""
        arr = np.asarray(xyz, dtype=float)
        single = arr.ndim == 1
        pts = np.atleast_2d(arr).astype(float)
        if pts.shape[1] != 3:
            raise ValueError(f"expected (..., 3) coordinates, got shape {arr.shape}")
        local_cart = (pts - self.origin) @ self.rotation
        if self.type == "cartesian":
            out = local_cart
        elif self.type == "cylindrical":
            x, y, z = local_cart[:, 0], local_cart[:, 1], local_cart[:, 2]
            out = np.column_stack([np.hypot(x, y), np.rad2deg(np.arctan2(y, x)), z])
        else:  # spherical
            x, y, z = local_cart[:, 0], local_cart[:, 1], local_cart[:, 2]
            r = np.sqrt(x * x + y * y + z * z)
            with np.errstate(invalid="ignore", divide="ignore"):
                cos_t = np.clip(np.divide(z, np.where(r == 0.0, 1.0, r)), -1.0, 1.0)
                theta = np.rad2deg(np.arccos(cos_t))
            theta = np.where(r == 0.0, 0.0, theta)
            phi = np.rad2deg(np.arctan2(y, x))
            out = np.column_stack([r, theta, phi])
        return out[0] if single else out

    # -- physical vector basis ---------------------------------------------
    def basis_at(self, xyz_global: ArrayLike) -> NDArray[np.float64]:
        """Orthonormal physical basis at a global point; columns are the local
        direction unit vectors expressed in global coordinates.

        cartesian:   (e_x, e_y, e_z)          -- position independent
        cylindrical: (e_r, e_theta, e_z)
        spherical:   (e_r, e_theta, e_phi)

        Degenerate locations (on the z-axis for cylindrical, at the origin or
        poles for spherical) fall back to the local cartesian axes.
        """
        if self.type == "cartesian":
            return self.rotation.copy()
        p = np.asarray(xyz_global, dtype=float).reshape(3)
        local = (p - self.origin) @ self.rotation
        x, y, z = local
        if self.type == "cylindrical":
            rr = float(np.hypot(x, y))
            if rr < 1e-12:
                return self.rotation.copy()
            th = np.arctan2(y, x)
            e_r = np.array([np.cos(th), np.sin(th), 0.0])
            e_t = np.array([-np.sin(th), np.cos(th), 0.0])
            e_z = np.array([0.0, 0.0, 1.0])
            local_basis = np.column_stack([e_r, e_t, e_z])
        else:  # spherical
            r = float(np.sqrt(x * x + y * y + z * z))
            rr = float(np.hypot(x, y))
            if r < 1e-12 or rr < 1e-12:
                return self.rotation.copy()
            th = np.arccos(np.clip(z / r, -1.0, 1.0))
            ph = np.arctan2(y, x)
            st, ct, sp, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
            e_r = np.array([st * cp, st * sp, ct])
            e_t = np.array([ct * cp, ct * sp, -st])
            e_p = np.array([-sp, cp, 0.0])
            local_basis = np.column_stack([e_r, e_t, e_p])
        return self.rotation @ local_basis

    def transform_vector_to_global(
        self, v_local: ArrayLike, at: ArrayLike = (0.0, 0.0, 0.0)
    ) -> NDArray[np.float64]:
        """Vector components in this system's physical basis -> global components.

        ``at`` is the global point where the vector is attached (relevant for
        curvilinear systems only).
        """
        v = np.asarray(v_local, dtype=float).reshape(3)
        return self.basis_at(at) @ v

    def transform_vector_to_local(
        self, v_global: ArrayLike, at: ArrayLike = (0.0, 0.0, 0.0)
    ) -> NDArray[np.float64]:
        """Global vector components -> components in this system's physical basis at ``at``."""
        v = np.asarray(v_global, dtype=float).reshape(3)
        return self.basis_at(at).T @ v

    # -- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "origin": self.origin.tolist(),
            "rotation": self.rotation.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CoordSys:
        return cls(
            id=int(d["id"]),
            type=d.get("type", "cartesian"),
            origin=np.asarray(d.get("origin", (0.0, 0.0, 0.0)), dtype=float),
            rotation=np.asarray(d.get("rotation", np.eye(3)), dtype=float),
        )
