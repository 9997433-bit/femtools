"""Mesh / database integrity checks.

:func:`validate_model` runs a battery of structural checks over an
:class:`~femtools.core.model.FEModel` and returns a
:class:`ValidationReport`.  Errors indicate a model that solvers must
reject; warnings indicate suspicious but solvable data.

Checks
------
=========  ========  ====================================================
code       severity  meaning
=========  ========  ====================================================
E_NODE     error     element references an undefined node
E_PROP     error     element requires a property that is undefined/missing
E_MAT      error     property references an undefined material
E_PTYPE    error     property type incompatible with element type
E_DEGEN    error     element with repeated nodes (degenerate)
E_INVERT   error     inverted solid element (non-positive volume/Jacobian)
E_ZERO     error     zero-length line element / zero-area shell
W_ORPHAN   warning   node not referenced by any element
W_SPC      warning   SPC/load references an undefined node
W_WARP     warning   badly warped QUAD4 (non-planar beyond heuristic)
W_NOMAT    warning   material has no density (mass matrix will be empty)
=========  ========  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .model import _ELEMENT_NEEDS_PROPERTY, Element, FEModel

__all__ = ["ValidationIssue", "ValidationReport", "validate_model"]

_PROPERTY_TYPE_FOR_ELEMENT: dict[str, tuple[str, ...]] = {
    "BAR2": ("bar", "beam"),
    "TRUSS2D": ("bar", "beam"),
    "BEAM2": ("beam",),
    "QUAD4": ("shell",),
    "TRIA3": ("shell",),
    "HEX8": ("solid",),
    "TET4": ("solid",),
    "MASS": ("lumped",),
    "SPRING": ("lumped",),
    "DAMPER": ("lumped",),
}


@dataclass(frozen=True)
class ValidationIssue:
    """One finding of :func:`validate_model`."""

    severity: str  # "error" | "warning"
    code: str
    message: str
    entity: str = ""  # "node" | "element" | "property" | "material" | "spc" | "load"
    entity_id: int | None = None

    def __str__(self) -> str:
        loc = f" [{self.entity} {self.entity_id}]" if self.entity else ""
        return f"{self.severity.upper():7s} {self.code}: {self.message}{loc}"


@dataclass
class ValidationReport:
    """Collected issues; ``ok`` is True when there are no errors."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def __len__(self) -> int:
        return len(self.issues)

    def __str__(self) -> str:
        if not self.issues:
            return "ValidationReport: OK (no issues)"
        head = f"ValidationReport: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([head, *(str(i) for i in self.issues)])

    def _add(self, severity: str, code: str, message: str, entity: str = "", entity_id: int | None = None) -> None:
        self.issues.append(ValidationIssue(severity, code, message, entity, entity_id))


# -- geometric helpers -------------------------------------------------------


def tet4_volume(xyz: NDArray[np.float64]) -> float:
    """Signed volume of a 4-node tetrahedron (positive for right-handed order)."""
    a, b, c, d = xyz
    return float(np.dot(np.cross(b - a, c - a), d - a) / 6.0)


def hex8_jacobian_center(xyz: NDArray[np.float64]) -> float:
    """Determinant of the trilinear Jacobian at the centroid of a HEX8.

    Node ordering: bottom face 1-2-3-4 (counter-clockwise viewed from +z),
    top face 5-6-7-8 above them.  Negative -> inverted element.
    """
    # dN/dxi, dN/deta, dN/dzeta at (0,0,0) for the standard trilinear brick
    signs = np.array(
        [
            (-1, -1, -1),
            (+1, -1, -1),
            (+1, +1, -1),
            (-1, +1, -1),
            (-1, -1, +1),
            (+1, -1, +1),
            (+1, +1, +1),
            (-1, +1, +1),
        ],
        dtype=float,
    )
    dn = signs / 8.0  # dNi/dxi_j at the center = sign_ij / 8
    jac = dn.T @ xyz  # (3, 3)
    return float(np.linalg.det(jac))


def tria3_area(xyz: NDArray[np.float64]) -> float:
    a, b, c = xyz
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))


def quad4_area(xyz: NDArray[np.float64]) -> float:
    """Area of a (possibly warped) quad via the two-triangle split 1-2-3 + 1-3-4."""
    return tria3_area(xyz[[0, 1, 2]]) + tria3_area(xyz[[0, 2, 3]])


def quad4_warp(xyz: NDArray[np.float64]) -> float:
    """Warp measure: distance of nodes from the mean plane / mean edge length."""
    centroid = xyz.mean(axis=0)
    d1 = xyz[2] - xyz[0]
    d2 = xyz[3] - xyz[1]
    n = np.cross(d1, d2)
    nn = float(np.linalg.norm(n))
    if nn < 1e-300:
        return np.inf
    n = n / nn
    dist = float(np.abs((xyz - centroid) @ n).max())
    edges = np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1)
    mean_edge = float(edges.mean())
    return dist / mean_edge if mean_edge > 0.0 else np.inf


# -- main entry ---------------------------------------------------------------


def _check_element_geometry(model: FEModel, el: Element, rep: ValidationReport, length_tol: float) -> None:
    xyz = np.array([model.nodes[n].xyz for n in el.nodes], dtype=float)
    if el.type in ("BAR2", "BEAM2", "TRUSS2D", "SPRING", "DAMPER") and el.n_nodes == 2:
        if float(np.linalg.norm(xyz[1] - xyz[0])) <= length_tol and el.type in (
            "BAR2",
            "BEAM2",
            "TRUSS2D",
        ):
            rep._add("error", "E_ZERO", f"{el.type} element has (near) zero length", "element", el.id)
    elif el.type == "TRIA3":
        if tria3_area(xyz) <= length_tol**2:
            rep._add("error", "E_ZERO", "TRIA3 element has (near) zero area", "element", el.id)
    elif el.type == "QUAD4":
        if quad4_area(xyz) <= length_tol**2:
            rep._add("error", "E_ZERO", "QUAD4 element has (near) zero area", "element", el.id)
        else:
            warp = quad4_warp(xyz)
            if warp > 0.1:
                rep._add(
                    "warning",
                    "W_WARP",
                    f"QUAD4 element is badly warped (warp={warp:.3f} > 0.1)",
                    "element",
                    el.id,
                )
    elif el.type == "TET4":
        vol = tet4_volume(xyz)
        if vol <= 0.0:
            rep._add(
                "error",
                "E_INVERT",
                f"TET4 element has non-positive volume ({vol:.3e}); node order is inverted",
                "element",
                el.id,
            )
    elif el.type == "HEX8":
        det = hex8_jacobian_center(xyz)
        if det <= 0.0:
            rep._add(
                "error",
                "E_INVERT",
                f"HEX8 element has non-positive Jacobian at center ({det:.3e}); inverted or collapsed",
                "element",
                el.id,
            )


def validate_model(model: FEModel, length_tol: float = 1e-12) -> ValidationReport:
    """Run all integrity checks on ``model`` (see module docstring)."""
    rep = ValidationReport()
    used_nodes: set[int] = set()

    for el in model.elements.values():
        missing_nodes = [n for n in el.nodes if n not in model.nodes]
        if missing_nodes:
            rep._add(
                "error",
                "E_NODE",
                f"element references undefined node(s) {missing_nodes}",
                "element",
                el.id,
            )
        used_nodes.update(el.nodes)

        # degenerate connectivity: repeated node ids (grounded SPRING/DAMPER/MASS excluded)
        if el.type not in ("MASS", "SPRING", "DAMPER") and len(set(el.nodes)) != len(el.nodes):
            rep._add("error", "E_DEGEN", "element has repeated node ids", "element", el.id)

        # property linkage
        if el.property_id is None:
            if el.type in _ELEMENT_NEEDS_PROPERTY:
                rep._add("error", "E_PROP", f"{el.type} element has no property", "element", el.id)
        else:
            prop = model.properties.get(el.property_id)
            if prop is None:
                rep._add(
                    "error",
                    "E_PROP",
                    f"element references undefined property {el.property_id}",
                    "element",
                    el.id,
                )
            else:
                allowed = _PROPERTY_TYPE_FOR_ELEMENT.get(el.type, ())
                if allowed and prop.type not in allowed:
                    rep._add(
                        "error",
                        "E_PTYPE",
                        f"{el.type} element uses property {prop.id} of type {prop.type!r}; "
                        f"expected {allowed}",
                        "element",
                        el.id,
                    )

        # geometry checks only when all nodes resolve
        if not missing_nodes and len(set(el.nodes)) == len(el.nodes):
            _check_element_geometry(model, el, rep, length_tol)

    for prop in model.properties.values():
        if prop.material_id is not None and prop.material_id not in model.materials:
            rep._add(
                "error",
                "E_MAT",
                f"property references undefined material {prop.material_id}",
                "property",
                prop.id,
            )

    for mat in model.materials.values():
        if mat.type == "isotropic" and (mat.rho is None or mat.rho == 0.0):
            rep._add(
                "warning",
                "W_NOMAT",
                "material has no (or zero) density; it contributes no mass",
                "material",
                mat.id,
            )

    for nid in model.nodes:
        if nid not in used_nodes:
            rep._add("warning", "W_ORPHAN", "node is not referenced by any element", "node", nid)

    for spc in model.spcs:
        if spc.node_id not in model.nodes:
            rep._add("warning", "W_SPC", f"SPC references undefined node {spc.node_id}", "spc", spc.node_id)
    for load in model.loads:
        if load.node_id not in model.nodes:
            rep._add("warning", "W_SPC", f"load references undefined node {load.node_id}", "load", load.node_id)

    return rep
