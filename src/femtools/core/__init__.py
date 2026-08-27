"""femtools.core — in-memory relational FE/test database.

Re-exports the full public core API::

    from femtools.core import FEModel, Node, Element, Material, Property
    from femtools.core import NodeSet, ElementSet, UnitSystem, CoordSys
    from femtools.core import ModalResult, StaticResult, FRFResult, ODSResult
"""

from __future__ import annotations

from .coords import CoordSys
from .model import (
    DOF_LABELS,
    ELEMENT_NODE_COUNTS,
    ELEMENT_TYPES,
    MATERIAL_TYPES,
    NDOF_PER_NODE,
    PROPERTY_TYPES,
    SPC,
    DOFSet,
    Element,
    FEModel,
    Load,
    Material,
    ModelError,
    Node,
    Property,
    comps_to_mask,
    mask_to_comps,
)
from .results import (
    DofPair,
    FRFResult,
    ModalResult,
    ODSResult,
    StaticResult,
    normalize_dof_index,
)
from .sets import ElementSet, NodeSet
from .units import (
    UnitError,
    UnitSystem,
    convert,
    convert_force,
    convert_frequency,
    convert_length,
    convert_mass,
    convert_time,
)
from .validation import ValidationIssue, ValidationReport, validate_model

__all__ = [
    # model
    "FEModel",
    "Node",
    "Element",
    "Material",
    "Property",
    "SPC",
    "Load",
    "DOFSet",
    "ModelError",
    "NDOF_PER_NODE",
    "DOF_LABELS",
    "ELEMENT_TYPES",
    "ELEMENT_NODE_COUNTS",
    "PROPERTY_TYPES",
    "MATERIAL_TYPES",
    "comps_to_mask",
    "mask_to_comps",
    # sets
    "NodeSet",
    "ElementSet",
    # units
    "UnitSystem",
    "UnitError",
    "convert",
    "convert_length",
    "convert_force",
    "convert_mass",
    "convert_time",
    "convert_frequency",
    # coords
    "CoordSys",
    # results
    "ModalResult",
    "StaticResult",
    "FRFResult",
    "ODSResult",
    "DofPair",
    "normalize_dof_index",
    # validation
    "validate_model",
    "ValidationReport",
    "ValidationIssue",
]
