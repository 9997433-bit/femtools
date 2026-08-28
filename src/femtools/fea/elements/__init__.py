"""femtools element library.

Importing this package registers every element type.  The public entry points
are :func:`available_elements` (contract API), :func:`element_spec` and
:func:`element_matrices`.
"""

from __future__ import annotations

from typing import Any

from .bar import bar2, truss2d
from .base import (
    REGISTRY,
    ElementContext,
    ElementMatrices,
    ElementSpec,
    ModelIndex,
    available_elements,
    build_context,
    element_spec,
    register,
)
from .beam import beam2
from .frames import line_frame, shell_frame
from .scalar import damper, mass_element, spring
from .shell import quad4, tria3
from .solid import hex8, tet4, tet10

__all__ = [
    "ElementContext",
    "ElementMatrices",
    "ElementSpec",
    "ModelIndex",
    "REGISTRY",
    "available_elements",
    "bar2",
    "beam2",
    "build_context",
    "damper",
    "element_info",
    "element_matrices",
    "element_spec",
    "hex8",
    "line_frame",
    "mass_element",
    "quad4",
    "register",
    "shell_frame",
    "spring",
    "tet4",
    "tet10",
    "tria3",
    "truss2d",
]


def element_info() -> dict[str, dict[str, Any]]:
    """Describe every registered element type (name -> metadata)."""
    out: dict[str, dict[str, Any]] = {}
    for spec in sorted({id(s): s for s in REGISTRY.values()}.values(), key=lambda s: s.name):
        aliases = sorted(k for k, v in REGISTRY.items() if v is spec and k != spec.name)
        out[spec.name] = {
            "family": spec.family,
            "n_nodes": spec.n_nodes,
            "dofs_per_node": spec.dofs_per_node,
            "description": spec.description,
            "aliases": aliases,
        }
    return out


def element_matrices(
    model: Any,
    element_id: Any,
    element: Any,
    *,
    lumped_mass: bool = False,
    drill_factor: float = 1.0e-3,
    options: dict[str, Any] | None = None,
    index: ModelIndex | None = None,
) -> ElementMatrices:
    """Build the stiffness/mass/damping contribution of a single element."""
    ctx = build_context(
        model,
        element_id,
        element,
        lumped_mass=lumped_mass,
        drill_factor=drill_factor,
        options=options,
        index=index,
    )
    spec = element_spec(ctx.etype)
    if not spec.accepts(len(ctx.node_ids)):
        raise ValueError(
            f"element {element_id} ({ctx.etype}): got {len(ctx.node_ids)} nodes, "
            f"expected one of {spec.n_nodes}"
        )
    return spec.builder(ctx)
