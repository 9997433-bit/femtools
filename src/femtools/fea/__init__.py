"""Finite element kernel: element library, assembly, statics and normal modes.

The kernel is deliberately decoupled from the model database: everything is
consumed through the structural types in :mod:`femtools.fea.protocols`, so it
works with :mod:`femtools.core.model` as well as with plain dataclasses or
dictionaries that expose the same field names.

:mod:`femtools.fea.verification` (imported on demand) holds the reproducible
patch, locking and rigid-body cases quoted in the documentation, and
:mod:`femtools.fea.reduction` the Guyan / IRS / SEREP condensation bases.
"""

from __future__ import annotations

from .assemble import AssemblyResult, assemble_km
from .dofmap import DofMap
from .eigen import (
    ComplexModalResult,
    ModalResult,
    mass_normalize,
    solve_complex_modes,
    solve_modes,
)
from .elements import available_elements, element_info, element_matrices, element_spec
from .elements.solid import HEX8_FORMULATIONS, hex8_formulation
from .loads import build_load_vector
from .materials import MaterialData, material_from_record, plane_stress_D, solid_D
from .nodal_frames import NodalFrames, averaged_shell_normals, shell_nodal_frames
from .protocols import DOF_LABELS, ElementLike, ModelLike, NodeLike
from .reduction import ReductionResult, guyan, irs, serep
from .static import StaticResult, solve_static

__all__ = [
    "DOF_LABELS",
    "HEX8_FORMULATIONS",
    "AssemblyResult",
    "ComplexModalResult",
    "DofMap",
    "ElementLike",
    "MaterialData",
    "ModalResult",
    "ModelLike",
    "NodalFrames",
    "NodeLike",
    "ReductionResult",
    "StaticResult",
    "assemble_km",
    "available_elements",
    "averaged_shell_normals",
    "build_load_vector",
    "element_info",
    "element_matrices",
    "element_spec",
    "guyan",
    "hex8_formulation",
    "irs",
    "mass_normalize",
    "material_from_record",
    "plane_stress_D",
    "serep",
    "shell_nodal_frames",
    "solid_D",
    "solve_complex_modes",
    "solve_modes",
    "solve_static",
]
