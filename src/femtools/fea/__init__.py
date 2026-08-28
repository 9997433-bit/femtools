"""Finite element kernel: element library, assembly, statics and normal modes.

The kernel is deliberately decoupled from the model database: everything is
consumed through the structural types in :mod:`femtools.fea.protocols`, so it
works with :mod:`femtools.core.model` as well as with plain dataclasses or
dictionaries that expose the same field names.

:mod:`femtools.fea.verification` (imported on demand) holds the reproducible
patch, locking and rigid-body cases quoted in the documentation,
:mod:`femtools.fea.reduction` the Guyan / IRS / SEREP condensation bases,
:mod:`femtools.fea.recover` the element stress and strain recovery and
:mod:`femtools.fea.mpc` the ``RBE2`` rigid bodies, which
:func:`~femtools.fea.assemble.assemble_km` applies on its own.
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
from .mpc import ConstraintTransform, apply_rbe2
from .nodal_frames import NodalFrames, averaged_shell_normals, shell_nodal_frames
from .protocols import DOF_LABELS, ElementLike, ModelLike, NodeLike
from .recover import StressResult, recover_strain, recover_stress
from .reduction import ReductionResult, guyan, irs, serep
from .static import StaticResult, solve_static

__all__ = [
    "DOF_LABELS",
    "HEX8_FORMULATIONS",
    "AssemblyResult",
    "ComplexModalResult",
    "ConstraintTransform",
    "DofMap",
    "ElementLike",
    "MaterialData",
    "ModalResult",
    "ModelLike",
    "NodalFrames",
    "NodeLike",
    "ReductionResult",
    "StaticResult",
    "StressResult",
    "apply_rbe2",
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
    "recover_strain",
    "recover_stress",
    "serep",
    "shell_nodal_frames",
    "solid_D",
    "solve_complex_modes",
    "solve_modes",
    "solve_static",
]
