"""Element stress and strain recovery from a solved displacement field.

The solver returns displacements; what an analyst reports is stress.  This
module closes that gap for the six structural element types of the kernel --
``BAR2``, ``BEAM2``, ``TRIA3``, ``QUAD4``, ``TET4`` and ``HEX8`` -- by
evaluating each element's own strain-displacement operator at its **centroid**
and pushing the result through the same constitutive matrix the stiffness was
built with.  This is the classical displacement-method recovery of any FE text
(Cook, *Concepts and Applications of Finite Element Analysis*, §6.12 and §15.6;
Bathe, *Finite Element Procedures*, §4.2.1; Zienkiewicz & Taylor, §6): no
extrapolation, no nodal averaging, no smoothing.

Because the recovery uses the element's exact ``B`` matrix, a mesh carrying an
exactly linear displacement field returns the exact constant stress state --
the constant-strain patch test -- to machine precision, on distorted meshes as
well as on regular ones.  :func:`femtools.fea.verification.stress_patch_error`
runs that check for every type.

Conventions
-----------

``StressResult`` stores one six-component Voigt tensor per element, ordered
``(xx, yy, zz, xy, yz, zx)`` -- the ordering of
:func:`femtools.fea.materials.solid_D` -- with **engineering** shear strains,
so ``sigma = D @ eps`` holds componentwise.

* Solids report in the basic (global) frame.
* Shells and line elements report in their own element frame, whose axes are in
  :attr:`StressResult.frame`; :attr:`StressResult.stress_basic` rotates them
  out.  A shell tensor is the state at the through-thickness position selected
  by ``layer`` (``"mid"`` by default, also ``"top"``, ``"bottom"`` or an
  explicit offset), i.e. membrane plus ``z`` times curvature.
* Components a formulation does not carry are completed from its own stress
  assumption rather than left at zero: ``ezz`` of a shell follows the plane
  stress condition ``szz = 0``, and the transverse contraction of a rod or beam
  follows its uniaxial state.  Stress and strain therefore always describe one
  consistent state.

Nodal averaging
---------------

:func:`average_nodal` is the one smoothing step the module does offer, and it
is the simplest one there is: the value reported at a node is the unweighted
mean of the centroid tensors of the elements incident on it,

.. math::

    \\sigma_a = \\frac{1}{n_a} \\sum_{e \\ni a} \\sigma_e ,

the classical nodal average of Cook §6.13 / Zienkiewicz & Taylor §14.  It is
averaged in the **basic** frame, since tensors written in different element
frames cannot be added.  Two properties follow directly and are what the
function is for: a constant stress state survives it exactly at every node
(every contribution is the same number), and no element is preferred over
another, so the result carries no information the recovery did not have.

It is explicitly **not** superconvergent patch recovery: Zienkiewicz-Zhu fits a
polynomial through the superconvergent sampling points of a patch and is the
basis of an error estimator; this is an arithmetic mean and makes no claim
beyond continuity.  Where the two differ -- a coarse mesh with a steep gradient
-- SPR is the better answer, and the honest reading of a large jump between the
averaged nodal value and the element value is that the mesh is too coarse.

Superconvergent patch recovery
------------------------------

:func:`recover_spr` is that better answer, and it is a different function
rather than a mode of :func:`average_nodal`, whose ``1 / n_a`` weighting is
frozen.  Following Zienkiewicz and Zhu (*The superconvergent patch recovery and
a posteriori error estimates. Part 1: The recovery technique*, IJNME **33**\\ (7),
1992, pp. 1331-1364), the value at a node is read off a polynomial fitted by
least squares over the *patch* of elements meeting there,

.. math::

    \\sigma^*(x) = P(x)\\, a, \\qquad
    a = \\arg\\min_a \\sum_{e \\ni a} \\lVert P(x_e) a - \\sigma_e \\rVert^2 ,

with ``P`` the same polynomial order as the element shape functions -- linear
here -- and the sampling points ``x_e`` the points at which the finite element
stress is superconvergent.  For linear elements those are the element centroids
(Barlow, *Optimal stress locations in finite element models*, IJNME 10, 1976),
which is exactly what :func:`recover_stress` already reports, so the recovery
needs no second pass over the elements.

Two properties matter and both are checked in the test suite.  A constant
stress state is reproduced exactly at every node, interior and boundary alike,
because the fitted polynomial is then the constant itself.  And, unlike the
average, a *linear* stress field is reproduced exactly at every node whose
patch can carry the full polynomial: SPR puts the nodal value on the trend of
the patch instead of on the mean of its neighbours, and the mean of a linear
field over a lopsided patch is not the value at the node.

Which nodes those are is reported rather than assumed.  A node on the surface
of a solid mesh sees only sampling points on one side of itself, and one that
sits on an edge or a corner of the mesh sees very few; asking such a patch for
a full linear extrapolation is how SPR is known to misbehave, and it is the
case Zienkiewicz and Zhu handle by evaluating an *interior* patch at the
boundary node instead.  femtools takes the other documented way out and lowers
the order of the polynomial until the fit is no longer an extrapolation it
cannot support (see :data:`SPR_MAX_AMPLIFICATION`), which in the limit is the
arithmetic mean again.  :attr:`NodalStressResult.patch_terms` says how many
terms each node actually got.

The one deviation from the paper worth naming: femtools fits the same linear
polynomial for every element type, ``TET10`` included, and samples it at the
element centroid.  The superconvergent points of a quadratic tetrahedron are
its four Gauss points, not its centroid, so for ``TET10`` the result is an
honest least-squares patch fit -- constant-stress exact, continuous, better
than the average -- but not a formally superconvergent one.

What is deliberately *not* here: no extrapolation of Gauss point values to the
nodes, no nonlinear or plastic material.  Those are separate decisions with
their own error, and the centroid value is the one a linear element actually
represents best.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .assemble import AssemblyResult
from .dofmap import DofMap
from .elements import ModelIndex, build_context, element_spec
from .elements.bar import _axial_data
from .elements.base import ElementContext
from .elements.beam import _section, beam_local_matrices
from .elements.frames import line_frame, shell_frame
from .elements.shell import (
    _mitc4_tying_rows,
    _quad_jacobian,
    _quad_shape,
    _shell_props,
    cst_strain_matrix,
    dkt_curvature_matrix,
)
from .elements.solid import (
    TET10_CENTROID,
    _hex_shape,
    _strain_matrix,
    _tet10_shape,
    tet10_gradient,
)
from .materials import plane_stress_D, solid_D
from .protocols import get_any, iter_records
from .quadrature import gauss_2d

__all__ = [
    "COMPONENTS",
    "NodalStressResult",
    "StressResult",
    "average_nodal",
    "recover_spr",
    "recover_strain",
    "recover_stress",
    "von_mises",
]

#: Voigt ordering of every stress and strain vector in this module.
COMPONENTS: tuple[str, ...] = ("xx", "yy", "zz", "xy", "yz", "zx")

#: Element families with no stress state of their own (springs, dampers,
#: concentrated masses).  They are reported in ``StressResult.skipped``.
_SKIPPED_FAMILIES = frozenset({"scalar"})

_LAYERS: dict[str, float] = {"mid": 0.0, "middle": 0.0, "neutral": 0.0, "top": 0.5, "bottom": -0.5}


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Von Mises equivalent stress of one Voigt tensor or of a ``(n, 6)`` set."""
    s = np.atleast_2d(np.asarray(stress, dtype=float))
    dev = (
        (s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2
    )
    shear = s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2
    out = np.sqrt(0.5 * dev + 3.0 * shear)
    return out if np.ndim(stress) > 1 else out[0]


def _tensor(voigt: np.ndarray, *, shear_scale: float = 1.0) -> np.ndarray:
    """``(3, 3)`` matrix of a Voigt vector; ``shear_scale=0.5`` for strains."""
    xx, yy, zz, xy, yz, zx = (float(v) for v in voigt)
    xy, yz, zx = xy * shear_scale, yz * shear_scale, zx * shear_scale
    return np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])


def _voigt(tensor: np.ndarray, *, shear_scale: float = 1.0) -> np.ndarray:
    t = np.asarray(tensor, dtype=float)
    return np.array(
        [
            t[0, 0],
            t[1, 1],
            t[2, 2],
            t[0, 1] / shear_scale,
            t[1, 2] / shear_scale,
            t[0, 2] / shear_scale,
        ]
    )


def _rotate(values: np.ndarray, frames: np.ndarray, *, shear_scale: float) -> np.ndarray:
    """Rotate ``(n, 6)`` element-frame tensors into the basic frame."""
    out = np.empty_like(values)
    for i, (voigt, R) in enumerate(zip(values, frames, strict=True)):
        local = _tensor(voigt, shear_scale=shear_scale)
        out[i] = _voigt(R.T @ local @ R, shear_scale=shear_scale)
    return out


@dataclass
class StressResult:
    """Centroid stress and strain of every recovered element.

    Attributes
    ----------
    element_ids:
        Element identifiers, in model order; row ``i`` of every array belongs
        to ``element_ids[i]``.
    etypes:
        Canonical element type of each row.
    stress, strain:
        ``(n_elements, 6)`` Voigt tensors in the element frame, ordered
        :data:`COMPONENTS`, with engineering shear strains.
    frame:
        ``(n_elements, 3, 3)`` matrices whose **rows** are the element axes in
        basic coordinates, so ``v_element = frame[i] @ v_basic``.  The identity
        for solids.
    centroid:
        ``(n_elements, 3)`` element centroids in basic coordinates.
    layer:
        Through-thickness position the shell rows were evaluated at, as a
        fraction of the thickness (``0`` mid-surface, ``+0.5`` top).
    extras:
        Per-element quantities that are not a stress tensor: the local end
        forces and moments of ``BAR2``/``BEAM2``, the shell stress resultants.
    skipped:
        ``element id -> reason`` for every element left out (a concentrated
        mass has no stress state, an unregistered type cannot be recovered).
    """

    element_ids: list[Any] = field(default_factory=list)
    etypes: list[str] = field(default_factory=list)
    stress: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    strain: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    frame: np.ndarray = field(default_factory=lambda: np.zeros((0, 3, 3)))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    location: str = "centroid"
    layer: float = 0.0
    extras: dict[Any, dict[str, Any]] = field(default_factory=dict)
    skipped: dict[Any, str] = field(default_factory=dict)

    # -- basics ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.element_ids)

    @property
    def n_elements(self) -> int:
        return len(self.element_ids)

    @property
    def components(self) -> tuple[str, ...]:
        return COMPONENTS

    def index_of(self, element_id: Any) -> int:
        """Row of *element_id*."""
        try:
            return self.element_ids.index(element_id)
        except ValueError:
            pass
        try:
            return self.element_ids.index(int(element_id))
        except (ValueError, TypeError):
            reason = self.skipped.get(element_id)
            raise KeyError(
                f"element {element_id!r} is not in this result"
                + (f" ({reason})" if reason else "")
            ) from None

    # -- derived quantities ------------------------------------------------
    @property
    def von_mises(self) -> np.ndarray:
        """``(n_elements,)`` von Mises equivalent stress (frame independent)."""
        return von_mises(self.stress) if len(self) else np.zeros(0)

    @property
    def principal(self) -> np.ndarray:
        """``(n_elements, 3)`` principal stresses, descending."""
        out = np.empty((len(self), 3))
        for i, voigt in enumerate(self.stress):
            out[i] = np.linalg.eigvalsh(_tensor(voigt))[::-1]
        return out

    @property
    def max_shear(self) -> np.ndarray:
        """Tresca half-difference ``(s1 - s3) / 2``."""
        p = self.principal
        return 0.5 * (p[:, 0] - p[:, 2]) if len(self) else np.zeros(0)

    @property
    def stress_basic(self) -> np.ndarray:
        """The stress tensors rotated into the basic (global) frame."""
        return _rotate(self.stress, self.frame, shear_scale=1.0)

    @property
    def strain_basic(self) -> np.ndarray:
        """The strain tensors rotated into the basic (global) frame."""
        return _rotate(self.strain, self.frame, shear_scale=0.5)

    def tensor(self, element_id: Any) -> np.ndarray:
        """``(3, 3)`` stress tensor of one element, in its own frame."""
        return _tensor(self.stress[self.index_of(element_id)])

    def element(self, element_id: Any) -> dict[str, Any]:
        """Everything recovered for one element, as a plain dictionary."""
        i = self.index_of(element_id)
        out: dict[str, Any] = {
            "element_id": self.element_ids[i],
            "type": self.etypes[i],
            "stress": self.stress[i],
            "strain": self.strain[i],
            "von_mises": float(self.von_mises[i]),
            "frame": self.frame[i],
            "centroid": self.centroid[i],
        }
        out.update(self.extras.get(self.element_ids[i], {}))
        return out

    def summary(self) -> str:  # pragma: no cover - reporting helper
        peak = float(self.von_mises.max()) if len(self) else 0.0
        return (
            f"StressResult(elements={len(self)}, skipped={len(self.skipped)}, "
            f"location={self.location}, layer={self.layer:+.2f}t, "
            f"max_von_mises={peak:.6g})"
        )


# ---------------------------------------------------------------------------
# per-element recovery
# ---------------------------------------------------------------------------


@dataclass
class _Recovered:
    stress: np.ndarray
    strain: np.ndarray
    frame: np.ndarray
    centroid: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)


def _uniaxial(exx: float, sxx: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """``(strain, stress)`` of a rod/beam fibre in uniaxial stress."""
    lateral = -nu * exx
    return (
        np.array([exx, lateral, lateral, 0.0, 0.0, 0.0]),
        np.array([sxx, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


def _plane_stress_ezz(sxx: float, syy: float, E: float, nu: float) -> float:
    """Through-thickness strain implied by ``szz = 0``."""
    return -nu * (sxx + syy) / E if E > 0.0 else 0.0


def _bar_recovery(ctx: ElementContext, disp: np.ndarray, _z: float) -> _Recovered:
    length, e1, axial_stiffness, _ = _axial_data(ctx)
    area = float(ctx.require(("A", "a", "area", "Area"), "cross section area 'A'"))
    modulus = axial_stiffness / area if area else 0.0
    exx = float((disp[1, :3] - disp[0, :3]) @ e1) / length
    sxx = modulus * exx
    strain, stress = _uniaxial(exx, sxx, ctx.mat.nu)
    _, R = line_frame(ctx.coords[0], ctx.coords[1], None)
    return _Recovered(
        stress=stress,
        strain=strain,
        frame=R,
        centroid=0.5 * (ctx.coords[0] + ctx.coords[1]),
        extras={"axial_force": sxx * area, "length": length},
    )


def _beam_recovery(ctx: ElementContext, disp: np.ndarray, _z: float) -> _Recovered:
    orientation = ctx.value(("orientation", "v", "vector", "x3", "g0_vector", "orient"), None)
    if orientation is not None:
        orientation = np.asarray(orientation, dtype=float).ravel()
        if orientation.size != 3:
            orientation = None
    length, R = line_frame(ctx.coords[0], ctx.coords[1], orientation)
    sec = _section(ctx)
    k_local, _ = beam_local_matrices(length, ctx.mat.E, ctx.mat.G, ctx.mat.rho, sec)

    local = np.concatenate([R @ disp[0, :3], R @ disp[0, 3:], R @ disp[1, :3], R @ disp[1, 3:]])
    forces = k_local @ local

    exx = float(local[6] - local[0]) / length
    sxx = ctx.mat.E * exx
    strain, stress = _uniaxial(exx, sxx, ctx.mat.nu)
    return _Recovered(
        stress=stress,
        strain=strain,
        frame=R,
        centroid=0.5 * (ctx.coords[0] + ctx.coords[1]),
        extras={
            # Local end forces, the quantity a beam is actually checked with:
            # the centroid of a beam sits on its neutral axis, so bending shows
            # up in the moments rather than in the axial stress above.
            "end_forces": forces,
            "axial_force": float(forces[6]),
            "torque": float(forces[9]),
            "moments": np.array([forces[4], forces[5], forces[10], forces[11]]),
            "length": length,
        },
    )


def _shell_kinematics(ctx: ElementContext, disp: np.ndarray, n: int):
    """``(R, xy, centroid, membrane dofs, plate dofs)`` of a flat shell element."""
    R, xy, centroid = shell_frame(ctx.coords[:n])
    local = np.empty((n, 6))
    for i in range(n):
        local[i, :3] = R @ disp[i, :3]
        local[i, 3:] = R @ disp[i, 3:]
    membrane = local[:, :2].ravel()
    plate = np.concatenate([[local[i, 2], local[i, 3], local[i, 4]] for i in range(n)])
    return R, xy, centroid, membrane, plate


def _shell_result(
    ctx: ElementContext,
    sp: dict,
    R: np.ndarray,
    centroid: np.ndarray,
    eps_m: np.ndarray,
    kappa: np.ndarray,
    gamma: np.ndarray,
    z_fraction: float,
) -> _Recovered:
    Dm = plane_stress_D(ctx.mat)
    t = sp["t"]
    z = z_fraction * t
    in_plane = eps_m + z * kappa
    sigma = Dm @ in_plane
    Db = Dm * (t**3 / 12.0) * sp["i_ratio"]
    moments = Db @ kappa
    shear_stress = sp["kappa"] * ctx.mat.G * gamma

    ezz = (
        0.0
        if ctx.mat.is_orthotropic
        else _plane_stress_ezz(float(sigma[0]), float(sigma[1]), ctx.mat.E, ctx.mat.nu)
    )
    strain = np.array([in_plane[0], in_plane[1], ezz, in_plane[2], gamma[1], gamma[0]])
    stress = np.array([sigma[0], sigma[1], 0.0, sigma[2], shear_stress[1], shear_stress[0]])
    return _Recovered(
        stress=stress,
        strain=strain,
        frame=R,
        centroid=centroid,
        extras={
            "membrane_strain": eps_m,
            "curvature": kappa,
            "membrane_force": (Dm @ eps_m) * t,
            "moment": moments,
            "transverse_shear": sp["kappa"] * ctx.mat.G * t * gamma,
            "thickness": t,
        },
    )


def _tria3_recovery(ctx: ElementContext, disp: np.ndarray, z_fraction: float) -> _Recovered:
    R, xy, centroid, membrane, plate = _shell_kinematics(ctx, disp, 3)
    area, Bm = cst_strain_matrix(xy)
    sp = _shell_props(ctx, abs(area))

    eps_m = Bm @ membrane if sp["membrane"] else np.zeros(3)
    # DKT is a discrete-Kirchhoff element: the curvature is the whole bending
    # story and the transverse shear strain is zero by construction.
    kappa = dkt_curvature_matrix(xy, 1.0 / 3.0, 1.0 / 3.0) @ plate if sp["bending"] else np.zeros(3)
    return _shell_result(ctx, sp, R, centroid, eps_m, kappa, np.zeros(2), z_fraction)


def _quad4_recovery(ctx: ElementContext, disp: np.ndarray, z_fraction: float) -> _Recovered:
    R, xy, centroid, membrane, plate = _shell_kinematics(ctx, disp, 4)

    area = 0.0
    for pt, w in zip(*gauss_2d(2), strict=True):
        det, _ = _quad_jacobian(xy, _quad_shape(*pt)[1])
        area += w * det
    sp = _shell_props(ctx, area)

    _n, dn = _quad_shape(0.0, 0.0)
    _det, g = _quad_jacobian(xy, dn)

    eps_m = np.zeros(3)
    if sp["membrane"]:
        Bm = np.zeros((3, 8))
        for i in range(4):
            Bm[0, 2 * i] = g[i, 0]
            Bm[1, 2 * i + 1] = g[i, 1]
            Bm[2, 2 * i] = g[i, 1]
            Bm[2, 2 * i + 1] = g[i, 0]
        eps_m = Bm @ membrane

    kappa = np.zeros(3)
    gamma = np.zeros(2)
    if sp["bending"]:
        Bb = np.zeros((3, 12))
        for i in range(4):
            Bb[0, 3 * i + 2] = g[i, 0]
            Bb[1, 3 * i + 1] = -g[i, 1]
            Bb[2, 3 * i + 1] = -g[i, 0]
            Bb[2, 3 * i + 2] = g[i, 1]
        kappa = Bb @ plate
        tie_a, tie_b, tie_c, tie_d = _mitc4_tying_rows(xy)
        b_rz = 0.5 * tie_a + 0.5 * tie_c
        b_sz = 0.5 * tie_d + 0.5 * tie_b
        gamma = np.linalg.solve(dn.T @ xy, np.vstack([b_rz, b_sz])) @ plate

    return _shell_result(ctx, sp, R, centroid, eps_m, kappa, gamma, z_fraction)


def _tet4_recovery(ctx: ElementContext, disp: np.ndarray, _z: float) -> _Recovered:
    xyz = ctx.coords[:4]
    matrix = np.column_stack([np.ones(4), xyz])
    grad = np.linalg.inv(matrix)[1:4, :].T
    strain = _strain_matrix(grad) @ disp[:4, :3].ravel()
    stress = solid_D(ctx.mat) @ strain
    return _Recovered(
        stress=stress, strain=strain, frame=np.eye(3), centroid=xyz.mean(axis=0)
    )


def _tet10_recovery(ctx: ElementContext, disp: np.ndarray, _z: float) -> _Recovered:
    """Quadratic tetrahedron, evaluated at the element centroid.

    A ``TET10`` carries a *linear* strain field, so unlike the four-node tet
    it has a stress gradient of its own and the single value reported is the
    centroid one.  With straight edges that is exactly the mean of the four
    Gauss point values the stiffness was integrated with, which is why the
    centroid is the sample the recovery reports and the sample
    :func:`recover_spr` fits its patch polynomial through.
    """
    xyz = ctx.coords[:10]
    try:
        grad, _det = tet10_gradient(xyz, TET10_CENTROID)
    except ValueError as exc:
        raise ValueError(f"element {ctx.element_id}: {exc}") from None
    strain = _strain_matrix(grad) @ disp[:10, :3].ravel()
    stress = solid_D(ctx.mat) @ strain
    n, _dn = _tet10_shape(*TET10_CENTROID)
    return _Recovered(stress=stress, strain=strain, frame=np.eye(3), centroid=n @ xyz)


def _hex8_recovery(ctx: ElementContext, disp: np.ndarray, _z: float) -> _Recovered:
    xyz = ctx.coords[:8]
    _n, dn0 = _hex_shape(0.0, 0.0, 0.0)
    J0 = dn0.T @ xyz
    if abs(float(np.linalg.det(J0))) == 0.0:
        raise ValueError(f"element {ctx.element_id}: degenerate HEX8 (zero Jacobian)")
    grad = np.linalg.solve(J0, dn0.T).T
    # The nine incompatible modes of the shipped element carry the derivatives
    # of ``1 - xi^2`` and friends, all of which vanish at the element centre --
    # so the centroid strain is the compatible one whatever the formulation.
    strain = _strain_matrix(grad) @ disp[:8, :3].ravel()
    stress = solid_D(ctx.mat) @ strain
    return _Recovered(
        stress=stress, strain=strain, frame=np.eye(3), centroid=xyz.mean(axis=0)
    )


#: Recovery rule per canonical element type.
_RECOVERY: dict[str, Callable[[ElementContext, np.ndarray, float], _Recovered]] = {
    "BAR2": _bar_recovery,
    "TRUSS2D": _bar_recovery,
    "BEAM2": _beam_recovery,
    "TRIA3": _tria3_recovery,
    "QUAD4": _quad4_recovery,
    "TET4": _tet4_recovery,
    "TET10": _tet10_recovery,
    "HEX8": _hex8_recovery,
}


# ---------------------------------------------------------------------------
# the public entry points
# ---------------------------------------------------------------------------


def _layer_fraction(layer: Any) -> float:
    if layer is None:
        return 0.0
    if isinstance(layer, str):
        key = layer.strip().lower()
        if key not in _LAYERS:
            raise ValueError(
                f"unknown layer {layer!r}; expected one of {sorted(_LAYERS)} or a "
                "fraction of the thickness"
            )
        return _LAYERS[key]
    value = float(layer)
    if not -0.5 <= value <= 0.5:
        raise ValueError(
            f"layer {value} is outside the section: give a fraction of the thickness "
            "in [-0.5, 0.5]"
        )
    return value


def _displacement_field(
    model: Any,
    u: Any,
    assembly: AssemblyResult | None,
    dof_map: DofMap | None,
    index: ModelIndex,
) -> tuple[np.ndarray, DofMap]:
    """Return the basic-frame displacement vector and the DOF map to read it with."""
    if u is None:
        raise TypeError("recover_stress needs a displacement field")
    # A StaticResult carries both the field and the assembly it belongs to.
    inner = get_any(u, ("u", "displacements"), None)
    if inner is not None and not isinstance(u, (np.ndarray, list, tuple)):
        if assembly is None:
            assembly = get_any(u, ("assembly",), None)
        u = inner

    vector = np.asarray(u, dtype=float)
    if vector.ndim != 1:
        raise ValueError(
            f"recover_stress takes one displacement field at a time, got shape "
            f"{vector.shape}; loop over the columns"
        )

    if assembly is not None:
        dof_map = assembly.dof_map if dof_map is None else dof_map
        # Rigid bodies first (their dependent entries may be zero if the field
        # came from an eigensolution), then out of the analysis frame.
        vector = assembly.to_basic(assembly.recover_dependent(vector))
    if dof_map is None:
        dof_map = DofMap.from_nodes(index.nodes, 6)
    if vector.size != dof_map.n_dof:
        raise ValueError(
            f"displacement vector has {vector.size} entries but the model has "
            f"{dof_map.n_dof} DOFs"
        )
    return vector, dof_map


def recover_stress(
    model: Any,
    u: Any = None,
    *,
    assembly: AssemblyResult | None = None,
    dof_map: DofMap | None = None,
    elements: Callable[[Any, Any], bool] | Iterable[Any] | None = None,
    layer: Any = "mid",
    on_unknown: str = "raise",
    index: ModelIndex | None = None,
) -> StressResult:
    """Recover the centroid stress and strain of every element of *model*.

    Parameters
    ----------
    model
        Anything satisfying :class:`~femtools.fea.protocols.ModelLike`.
    u
        The displacement field: a full-length vector, or the
        :class:`~femtools.fea.static.StaticResult` of
        :func:`~femtools.fea.static.solve_static`, which also supplies the
        assembly.  Read in the basic frame; pass ``assembly`` (or a
        ``StaticResult``) when the field is in an analysis frame that differs
        from it -- an obliquely oriented shell -- or when the model has rigid
        bodies whose dependent motion still has to be filled in.
    assembly
        The assembly the field belongs to.  Supplies the DOF numbering, the
        nodal frames and the multipoint constraints.
    dof_map
        Explicit DOF numbering, when there is no assembly to take it from.
        Defaults to the one :func:`~femtools.fea.assemble.assemble_km` builds.
    elements
        Restrict the recovery: a callable ``(element_id, element) -> bool`` or
        an explicit collection of element ids.
    layer
        Through-thickness position for shells: ``"mid"`` (default), ``"top"``,
        ``"bottom"`` or a fraction of the thickness in ``[-0.5, 0.5]``.  Solids
        and line elements ignore it.
    on_unknown
        ``"raise"`` (default), ``"skip"`` or ``"warn"`` for element types that
        are not registered at all.  Registered types without a stress state
        (``MASS``, ``SPRING``, ``DAMPER``) are always skipped and reported in
        :attr:`StressResult.skipped`.

    Returns
    -------
    StressResult
    """
    index = ModelIndex.build(model) if index is None else index
    z_fraction = _layer_fraction(layer)
    field_vector, dofs = _displacement_field(model, u, assembly, dof_map, index)

    keep: Callable[[Any, Any], bool]
    if elements is None:
        keep = lambda _eid, _el: True  # noqa: E731
    elif callable(elements):
        keep = elements
    else:
        wanted = set(elements)
        keep = lambda eid, _el: eid in wanted  # noqa: E731

    result = StressResult(layer=z_fraction)
    stresses: list[np.ndarray] = []
    strains: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    centroids: list[np.ndarray] = []

    for eid, element in iter_records(get_any(model, ("elements", "elems", "element"), None)):
        if element is None or not keep(eid, element):
            continue
        etype = str(get_any(element, ("type", "etype", "element_type", "kind"), "")).upper()
        try:
            spec = element_spec(etype)
        except KeyError as exc:
            if on_unknown == "raise":
                raise
            if on_unknown == "warn":
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
            result.skipped[eid] = str(exc)
            continue
        rule = _RECOVERY.get(spec.name)
        if rule is None:
            result.skipped[eid] = (
                f"{spec.name} ({spec.family}) has no stress state to recover"
                if spec.family in _SKIPPED_FAMILIES
                else f"no stress recovery is implemented for {spec.name}"
            )
            continue

        ctx = build_context(model, eid, element, index=index)
        if not spec.accepts(len(ctx.node_ids)):
            raise ValueError(
                f"element {eid} ({spec.name}): got {len(ctx.node_ids)} nodes, "
                f"expected one of {spec.n_nodes}"
            )
        disp = np.zeros((len(ctx.node_ids), 6))
        for i, nid in enumerate(ctx.node_ids):
            nodal = field_vector[dofs.node_dofs(nid)]
            disp[i, : nodal.size] = nodal[:6]
        recovered = rule(ctx, disp, z_fraction)

        result.element_ids.append(eid)
        result.etypes.append(spec.name)
        stresses.append(recovered.stress)
        strains.append(recovered.strain)
        frames.append(recovered.frame)
        centroids.append(recovered.centroid)
        if recovered.extras:
            result.extras[eid] = recovered.extras

    result.stress = np.array(stresses, dtype=float) if stresses else np.zeros((0, 6))
    result.strain = np.array(strains, dtype=float) if strains else np.zeros((0, 6))
    result.frame = np.array(frames, dtype=float) if frames else np.zeros((0, 3, 3))
    result.centroid = np.array(centroids, dtype=float) if centroids else np.zeros((0, 3))
    return result


def recover_strain(model: Any, u: Any = None, **kwargs: Any) -> StressResult:
    """Recover the centroid strain of every element of *model*.

    Same arguments and same object as :func:`recover_stress` -- the two are one
    computation, since the stress of a linear element *is* its strain pushed
    through the constitutive matrix.  The separate name exists so that code
    reading :attr:`StressResult.strain` says what it means.
    """
    return recover_stress(model, u, **kwargs)


# ---------------------------------------------------------------------------
# nodal averaging
# ---------------------------------------------------------------------------


@dataclass
class NodalStressResult:
    """Element stress and strain averaged onto the nodes.

    The counterpart of :class:`StressResult` for the smoothed field: one Voigt
    tensor per node instead of one per element, always in the **basic**
    (global) frame, because that is the only frame the contributions of
    differently oriented elements share.

    Attributes
    ----------
    node_ids:
        Nodes carrying a value, in the order they were first met; row ``i`` of
        every array belongs to ``node_ids[i]``.  Nodes with no recovered
        element on them are not listed at all rather than being reported as
        zero.
    stress, strain:
        ``(n_nodes, 6)`` Voigt tensors, ordered :data:`COMPONENTS`, with
        engineering shear strains.
    count:
        ``(n_nodes,)`` number of elements averaged into each node, the ``n_a``
        of the ``1 / n_a`` weighting -- and, for :func:`recover_spr`, the
        number of sampling points in the patch.
    xyz:
        ``(n_nodes, 3)`` node coordinates, when the model supplied them.
    element_ids, etypes, layer:
        Provenance: the elements that contributed, their types and the
        through-thickness position the shell rows were evaluated at.
    location:
        ``"node"`` for the average, ``"spr"`` for the patch recovery.
    patch_terms:
        SPR only: how many terms of the linear polynomial the patch could
        actually support at each node (4 on a three-dimensional patch, fewer
        where the sampling points are coplanar, collinear or too few).  Empty
        for :func:`average_nodal`, which fits nothing.
    """

    node_ids: list[Any] = field(default_factory=list)
    stress: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    strain: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    element_ids: list[Any] = field(default_factory=list)
    etypes: list[str] = field(default_factory=list)
    location: str = "node"
    layer: float = 0.0
    patch_terms: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def __len__(self) -> int:
        return len(self.node_ids)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def components(self) -> tuple[str, ...]:
        return COMPONENTS

    def index_of(self, node_id: Any) -> int:
        """Row of *node_id*."""
        try:
            return self.node_ids.index(node_id)
        except ValueError:
            pass
        try:
            return self.node_ids.index(int(node_id))
        except (ValueError, TypeError):
            raise KeyError(f"node {node_id!r} carries no averaged stress") from None

    @property
    def von_mises(self) -> np.ndarray:
        """``(n_nodes,)`` von Mises stress *of the averaged tensor*.

        Not the average of the element von Mises values: the equivalent stress
        is a nonlinear function of the tensor, and averaging the tensor first
        is what keeps a constant state exact.
        """
        return von_mises(self.stress) if len(self) else np.zeros(0)

    @property
    def principal(self) -> np.ndarray:
        """``(n_nodes, 3)`` principal stresses, descending."""
        out = np.empty((len(self), 3))
        for i, voigt in enumerate(self.stress):
            out[i] = np.linalg.eigvalsh(_tensor(voigt))[::-1]
        return out

    def tensor(self, node_id: Any) -> np.ndarray:
        """``(3, 3)`` averaged stress tensor at one node."""
        return _tensor(self.stress[self.index_of(node_id)])

    def node(self, node_id: Any) -> dict[str, Any]:
        """Everything averaged onto one node, as a plain dictionary."""
        i = self.index_of(node_id)
        return {
            "node_id": self.node_ids[i],
            "stress": self.stress[i],
            "strain": self.strain[i],
            "von_mises": float(self.von_mises[i]),
            "count": int(self.count[i]),
            "xyz": self.xyz[i],
        }

    def summary(self) -> str:  # pragma: no cover - reporting helper
        peak = float(self.von_mises.max()) if len(self) else 0.0
        return (
            f"NodalStressResult(nodes={len(self)}, elements={len(self.element_ids)}, "
            f"location={self.location}, layer={self.layer:+.2f}t, "
            f"max_von_mises={peak:.6g})"
        )


def average_nodal(
    stress: StressResult,
    model: Any,
    *,
    nodes: Callable[[Any], bool] | Iterable[Any] | None = None,
    index: ModelIndex | None = None,
) -> NodalStressResult:
    """Average centroid element stresses onto the nodes they touch.

    Each node gets the unweighted mean of the basic-frame tensors of the
    ``n_a`` elements incident on it (see the module docstring).  This is the
    classical nodal average, **not** Zienkiewicz-Zhu superconvergent patch
    recovery: no polynomial is fitted, no patch is assembled and no error
    estimate is produced.

    Because every contribution to a node is an equal share, a field that is
    already constant over the mesh -- the constant-stress patch test of
    :func:`femtools.fea.verification.stress_patch_error` -- comes through
    exactly at every node, interior and boundary alike.

    Parameters
    ----------
    stress
        The :class:`StressResult` of :func:`recover_stress` (or
        :func:`recover_strain`, which returns the same object).
    model
        The model the result was recovered from; only the element connectivity
        is read, plus the node coordinates when they are available.
    nodes
        Restrict the output: a callable ``(node_id) -> bool`` or an explicit
        collection of node ids.  Elements are still averaged in full, so a
        restricted node keeps the same value it would have had.
    index
        A prebuilt :class:`~femtools.fea.elements.ModelIndex`, to skip
        re-reading the node table.

    Returns
    -------
    NodalStressResult
        Empty when the recovery produced no elements.
    """
    if not isinstance(stress, StressResult):
        raise TypeError(
            "average_nodal takes the StressResult of recover_stress, got "
            f"{type(stress).__name__}"
        )
    index = ModelIndex.build(model) if index is None else index

    keep: Callable[[Any], bool]
    if nodes is None:
        keep = lambda _nid: True  # noqa: E731
    elif callable(nodes):
        keep = nodes
    else:
        wanted = set(nodes)
        keep = lambda nid: nid in wanted  # noqa: E731

    connectivity = _connectivity(model, stress.element_ids)
    stress_basic = stress.stress_basic
    strain_basic = stress.strain_basic

    order: list[Any] = []
    rows: dict[Any, int] = {}
    totals: list[np.ndarray] = []
    counts: list[int] = []
    used: list[Any] = []
    used_types: list[str] = []
    for i, eid in enumerate(stress.element_ids):
        node_ids = connectivity.get(eid)
        if not node_ids:
            continue
        used.append(eid)
        used_types.append(stress.etypes[i])
        contribution = np.concatenate([stress_basic[i], strain_basic[i]])
        for nid in node_ids:
            row = rows.get(nid)
            if row is None:
                row = rows[nid] = len(order)
                order.append(nid)
                totals.append(np.zeros(12))
                counts.append(0)
            totals[row] += contribution
            counts[row] += 1

    selected = [i for i, nid in enumerate(order) if keep(nid)]
    result = NodalStressResult(
        node_ids=[order[i] for i in selected],
        element_ids=used,
        etypes=used_types,
        layer=stress.layer,
    )
    if not selected:
        return result

    summed = np.array([totals[i] for i in selected], dtype=float)
    result.count = np.array([counts[i] for i in selected], dtype=int)
    averaged = summed / result.count[:, None]
    result.stress = averaged[:, :6]
    result.strain = averaged[:, 6:]
    result.xyz = np.array(
        [_node_xyz_or_nan(index, nid) for nid in result.node_ids], dtype=float
    )
    return result


def _connectivity(model: Any, element_ids: Iterable[Any]) -> dict[Any, tuple[Any, ...]]:
    """``element id -> node ids`` for the elements a recovery reported."""
    wanted = set(element_ids)
    out: dict[Any, tuple[Any, ...]] = {}
    for eid, element in iter_records(get_any(model, ("elements", "elems", "element"), None)):
        if element is None or eid not in wanted:
            continue
        raw = get_any(element, ("nodes", "node_ids", "connectivity", "conn", "grids"), None)
        if raw is None:
            continue
        if isinstance(raw, (int, np.integer, str)):
            raw = (raw,)
        # A blank connection slot is the "grounded" marker of a scalar element.
        out[eid] = tuple(nid for nid in raw if nid is not None)
    return out


def _node_xyz_or_nan(index: ModelIndex, node_id: Any) -> np.ndarray:
    try:
        return np.asarray(index.xyz(node_id), dtype=float)
    except (KeyError, TypeError, ValueError):
        return np.full(3, np.nan)


# ---------------------------------------------------------------------------
# superconvergent patch recovery
# ---------------------------------------------------------------------------

#: A column of the SPR basis is dropped when this little of it survives the
#: projection onto the columns already accepted.  The offsets are scaled to the
#: patch radius first, so the test is on a dimensionless quantity.
SPR_RANK_TOLERANCE = 1.0e-8

#: How much a patch fit may amplify the scatter of its own samples before a
#: polynomial term is given up.
#:
#: Evaluating the fit at the node is a linear functional ``w`` of the sampled
#: stresses, and ``||w|| sqrt(n)`` is by how much it can magnify a perturbation
#: of them; the plain average scores exactly 1.  A patch that is short of
#: sampling points, or whose points all sit on one side of the node -- the
#: boundary patches Zienkiewicz and Zhu handle by borrowing an interior fit --
#: is *extrapolating*, and an exactly determined extrapolation over a short
#: baseline can multiply round-off by several thousand.  Dropping the last
#: polynomial term until the score is back under this bound trades a little of
#: the recovery's reach for an answer that cannot be worse than the average by
#: more than a fixed factor.
SPR_MAX_AMPLIFICATION = 20.0


def _spr_basis(offsets: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """The usable columns of ``[1, dx, dy, dz]`` over one patch, and which they are.

    A patch whose sampling points are coplanar (a flat shell mesh), collinear
    (a chain of rods) or simply too few cannot support the full linear
    polynomial, and asking a least-squares solver for it anyway returns the
    minimum-norm answer, which does *not* reproduce a constant.  Accepting the
    columns one at a time by modified Gram-Schmidt and dropping the ones that
    add nothing leaves a basis of full column rank that still starts with the
    constant, so the fit is unique and a constant field survives exactly.
    """
    P = np.column_stack([np.ones(len(offsets)), np.asarray(offsets, dtype=float)])
    kept: list[int] = []
    basis = np.zeros((len(offsets), 0))
    for j in range(P.shape[1]):
        column = P[:, j]
        residual = column - basis @ (basis.T @ column) if basis.shape[1] else column.copy()
        norm = float(np.linalg.norm(residual))
        if norm > SPR_RANK_TOLERANCE * max(float(np.linalg.norm(column)), 1.0):
            kept.append(j)
            basis = np.column_stack([basis, residual / norm])
    return P[:, kept], kept


def _spr_weights(P: np.ndarray) -> np.ndarray:
    """``w`` with ``w @ values`` the least-squares fit evaluated at the node.

    The offsets the basis was built from are measured from the node, so the
    polynomial there is its constant term: ``w = P (P^T P)^-1 e_0``.
    """
    e0 = np.zeros(P.shape[1])
    e0[0] = 1.0
    return P @ np.linalg.solve(P.T @ P, e0)


def _spr_fit(offsets: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, int]:
    """Fit one patch and evaluate it at the node; returns ``(values, n_terms)``."""
    mean = data.mean(axis=0)
    P, _kept = _spr_basis(offsets)
    limit = SPR_MAX_AMPLIFICATION / np.sqrt(len(data))
    while P.shape[1] > 1:
        w = _spr_weights(P)
        if float(np.linalg.norm(w)) <= limit:
            # Fitting the deviation from the patch mean rather than the values
            # themselves is the same fit -- the constant column is always in
            # the basis, so shifting the data shifts the constant coefficient
            # by the same amount -- but it makes a constant field come through
            # at the round-off of a mean rather than of a linear solve.
            return mean + w @ (data - mean), P.shape[1]
        P = P[:, :-1]
    return mean, 1


def recover_spr(
    stress: StressResult,
    model: Any,
    *,
    nodes: Callable[[Any], bool] | Iterable[Any] | None = None,
    index: ModelIndex | None = None,
) -> NodalStressResult:
    """Zienkiewicz-Zhu superconvergent patch recovery of the nodal stresses.

    Zienkiewicz, O.C. and Zhu, J.Z., *The superconvergent patch recovery and a
    posteriori error estimates. Part 1: The recovery technique*, IJNME
    **33**\\ (7), 1992, pp. 1331-1364.  For each node, a linear polynomial is
    fitted by least squares through the recovered stresses of the elements
    meeting at it -- sampled at their centroids, the superconvergent points of
    a linear element (Barlow 1976) -- and evaluated at the node.  See the
    module docstring for the statement of the method and for what femtools
    does differently.

    This is *not* :func:`average_nodal`, which stays the plain ``1 / n_a``
    arithmetic mean and is unchanged by this function existing.  The two agree
    on a constant field (both are exact) and part company as soon as the
    stress has a gradient: SPR reproduces a linear field exactly at every node,
    including the boundary nodes where the average is biased towards the
    interior.

    Parameters
    ----------
    stress
        The :class:`StressResult` of :func:`recover_stress` -- the element
        centroid values *are* the sampling points, so no second pass over the
        elements is needed.
    model
        The model it was recovered from; the element connectivity and the node
        coordinates are read.
    nodes
        Restrict the output: a callable ``(node_id) -> bool`` or an explicit
        collection of node ids.  Patches are still assembled in full, so a
        restricted node keeps the value it would have had.
    index
        A prebuilt :class:`~femtools.fea.elements.ModelIndex`.

    Returns
    -------
    NodalStressResult
        With :attr:`~NodalStressResult.location` ``"spr"`` and
        :attr:`~NodalStressResult.patch_terms` recording how many polynomial
        terms each patch could carry.  A node whose coordinates the model does
        not give, or whose patch supports nothing but the constant, falls back
        to the arithmetic mean of its patch -- which is what a one-term fit
        *is*, so the fallback is the same formula rather than a different one.
    """
    if not isinstance(stress, StressResult):
        raise TypeError(
            "recover_spr takes the StressResult of recover_stress, got "
            f"{type(stress).__name__}"
        )
    index = ModelIndex.build(model) if index is None else index

    keep: Callable[[Any], bool]
    if nodes is None:
        keep = lambda _nid: True  # noqa: E731
    elif callable(nodes):
        keep = nodes
    else:
        wanted = set(nodes)
        keep = lambda nid: nid in wanted  # noqa: E731

    connectivity = _connectivity(model, stress.element_ids)
    samples = np.hstack([stress.stress_basic, stress.strain_basic])

    order: list[Any] = []
    patches: dict[Any, list[int]] = {}
    used: list[Any] = []
    used_types: list[str] = []
    for i, eid in enumerate(stress.element_ids):
        node_ids = connectivity.get(eid)
        if not node_ids:
            continue
        used.append(eid)
        used_types.append(stress.etypes[i])
        for nid in node_ids:
            rows = patches.get(nid)
            if rows is None:
                rows = patches[nid] = []
                order.append(nid)
            rows.append(i)

    selected = [nid for nid in order if keep(nid)]
    result = NodalStressResult(
        node_ids=selected,
        element_ids=used,
        etypes=used_types,
        location="spr",
        layer=stress.layer,
    )
    if not selected:
        return result

    values = np.empty((len(selected), 12))
    counts = np.empty(len(selected), dtype=int)
    terms = np.empty(len(selected), dtype=int)
    xyz = np.empty((len(selected), 3))
    for row, nid in enumerate(selected):
        patch = patches[nid]
        counts[row] = len(patch)
        xyz[row] = _node_xyz_or_nan(index, nid)
        data = samples[patch]
        if not np.all(np.isfinite(xyz[row])):
            terms[row] = 1
            values[row] = data.mean(axis=0)
            continue
        offsets = stress.centroid[patch] - xyz[row]
        radius = float(np.max(np.abs(offsets))) if offsets.size else 0.0
        if radius > 0.0:
            offsets = offsets / radius
        values[row], terms[row] = _spr_fit(offsets, data)

    result.count = counts
    result.patch_terms = terms
    result.xyz = xyz
    result.stress = values[:, :6]
    result.strain = values[:, 6:]
    return result
