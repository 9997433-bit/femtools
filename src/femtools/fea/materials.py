"""Material data extraction and constitutive matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocols import as_float, get_any

__all__ = [
    "MaterialData",
    "material_from_record",
    "plane_stress_D",
    "solid_D",
]


@dataclass(frozen=True)
class MaterialData:
    """Isotropic / orthotropic elastic constants in a solver friendly form."""

    E: float = 0.0
    nu: float = 0.0
    rho: float = 0.0
    G: float = 0.0
    E2: float | None = None
    G12: float | None = None
    G1z: float | None = None
    G2z: float | None = None
    alpha: float = 0.0
    damping: float = 0.0
    name: str = ""
    kind: str = "isotropic"

    @property
    def is_orthotropic(self) -> bool:
        return self.E2 is not None and not np.isclose(self.E2, self.E)


def material_from_record(record: object | None) -> MaterialData:
    """Build :class:`MaterialData` from any material-like record.

    Missing shear modulus is derived from ``E`` and ``nu``; missing ``nu`` is
    derived from ``E`` and ``G`` when both are available.
    """
    if record is None:
        return MaterialData()
    E = as_float(get_any(record, ("E", "e", "E1", "Ex", "youngs", "young", "modulus"), None))
    nu = as_float(get_any(record, ("nu", "NU", "poisson", "nu12", "PoissonRatio", "v"), None))
    rho = as_float(get_any(record, ("rho", "RHO", "density", "dens"), None), 0.0) or 0.0
    G = as_float(get_any(record, ("G", "g", "shear_modulus"), None))
    E2 = as_float(get_any(record, ("E2", "Ey"), None))
    G12 = as_float(get_any(record, ("G12",), None))
    G1z = as_float(get_any(record, ("G1z", "G13"), None))
    G2z = as_float(get_any(record, ("G2z", "G23"), None))
    alpha = as_float(get_any(record, ("alpha", "A", "cte", "thermal_expansion"), None), 0.0) or 0.0
    damping = as_float(get_any(record, ("ge", "GE", "eta", "structural_damping"), None), 0.0) or 0.0
    name = str(get_any(record, ("name", "label"), "") or "")
    kind = str(get_any(record, ("type", "kind", "mat_type"), "isotropic") or "isotropic")

    if E is None and G is not None and nu is not None:
        E = 2.0 * G * (1.0 + nu)
    if E is None:
        E = 0.0
    if nu is None:
        nu = 0.0 if G is None or G == 0.0 else max(-0.999, min(0.499, E / (2.0 * G) - 1.0))
    if G is None or G == 0.0:
        G = E / (2.0 * (1.0 + nu)) if E else 0.0

    return MaterialData(
        E=float(E),
        nu=float(nu),
        rho=float(rho),
        G=float(G),
        E2=E2,
        G12=G12,
        G1z=G1z,
        G2z=G2z,
        alpha=float(alpha),
        damping=float(damping),
        name=name,
        kind=str(kind).lower(),
    )


def plane_stress_D(mat: MaterialData) -> np.ndarray:
    """Plane stress constitutive matrix for ``[sxx, syy, sxy]``.

    Orthotropic input (``E2``/``G12`` present) is honoured, otherwise the
    isotropic form is used.
    """
    if mat.is_orthotropic:
        e1 = mat.E
        e2 = float(mat.E2)  # type: ignore[arg-type]
        nu12 = mat.nu
        nu21 = nu12 * e2 / e1
        denom = 1.0 - nu12 * nu21
        g12 = mat.G12 if mat.G12 else mat.G
        return np.array(
            [
                [e1 / denom, nu12 * e2 / denom, 0.0],
                [nu12 * e2 / denom, e2 / denom, 0.0],
                [0.0, 0.0, g12],
            ],
            dtype=float,
        )
    e, nu = mat.E, mat.nu
    factor = e / (1.0 - nu * nu) if abs(1.0 - nu * nu) > 0.0 else 0.0
    return factor * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]], dtype=float
    )


def solid_D(mat: MaterialData) -> np.ndarray:
    """3D isotropic constitutive matrix for ``[xx, yy, zz, xy, yz, zx]``."""
    e, nu = mat.E, mat.nu
    denom = (1.0 + nu) * (1.0 - 2.0 * nu)
    if abs(denom) == 0.0:
        return np.zeros((6, 6))
    lam = e * nu / denom
    mu = e / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6), dtype=float)
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D
