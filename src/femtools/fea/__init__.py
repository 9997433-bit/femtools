"""Finite element kernel: element library, assembly, statics and normal modes.

The kernel is deliberately decoupled from the model database: everything is
consumed through the structural types in :mod:`femtools.fea.protocols`, so it
works with :mod:`femtools.core.model` as well as with plain dataclasses or
dictionaries that expose the same field names.

:mod:`femtools.fea.verification` (imported on demand) holds the reproducible
patch, locking and rigid-body cases quoted in the documentation.
"""

from __future__ import annotations

from .assemble import AssemblyResult, assemble_km
from .dofmap import DofMap
from .eigen import ModalResult, mass_normalize, solve_modes
from .elements import available_elements, element_info, element_matrices, element_spec
from .elements.solid import HEX8_FORMULATIONS, hex8_formulation
from .loads import build_load_vector
from .materials import MaterialData, material_from_record, plane_stress_D, solid_D
from .protocols import DOF_LABELS, ElementLike, ModelLike, NodeLike
from .static import StaticResult, solve_static

__all__ = [
    "DOF_LABELS",
    "HEX8_FORMULATIONS",
    "AssemblyResult",
    "DofMap",
    "ElementLike",
    "MaterialData",
    "ModalResult",
    "ModelLike",
    "NodeLike",
    "StaticResult",
    "assemble_km",
    "available_elements",
    "build_load_vector",
    "element_info",
    "element_matrices",
    "element_spec",
    "hex8_formulation",
    "mass_normalize",
    "material_from_record",
    "plane_stress_D",
    "solid_D",
    "solve_modes",
    "solve_static",
]
