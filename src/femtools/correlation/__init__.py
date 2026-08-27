"""Test/analysis correlation: MAC family, mode pairing, FRF criteria, orthogonality.

Typical workflow::

    from femtools.correlation import align_modes, mac_matrix, pair_modes

    phi_t, phi_a, dofs = align_modes(phi_test, map_test, phi_fe, map_fe)
    result = pair_modes(phi_t, phi_a, f_test, f_fe, method="hungarian")
    print(result.table())

A solved model can be used wherever mode shapes are expected: a
:class:`~femtools.fea.eigen.ModalResult` supplies its ``modes``, its real
``dof_map`` and its frequencies on its own, so the analysis side of the same
workflow reads ``align_modes(phi_test, map_test, modal)`` followed by
``pair_modes(phi_t, phi_a, f_test)``.

The scalar single-vector MAC lives at :func:`femtools.correlation.mac.mac`
(exported here as :func:`mac_value`); the name ``femtools.correlation.mac``
itself remains the module.
"""

from __future__ import annotations

from .alignment import AlignmentResult, align_geometry, rotate_modes
from .dofmap import (
    COMPONENT_NAMES,
    DOFMap,
    align_modes,
    as_dofmap,
    match_dofs,
    parse_component,
    parse_components,
    parse_dof_label,
    restrict,
)
from .expansion import ExpansionResult, expand_guyan, expand_serep
from .frf_corr import csac, csf, fdac, frac, frf_difference
from .mac import (
    FMACResult,
    comac,
    ecomac,
    fmac,
    mac_matrix,
    mac_pairs,
    mac_value,
    modal_scale_factor,
    poc,
)
from .orthogonality import (
    auto_orthogonality,
    cross_orthogonality,
    off_diagonal_max,
    orthogonality_error,
)
from .pairing import ModePair, PairingResult, pair_modes

__all__ = [
    "COMPONENT_NAMES",
    "AlignmentResult",
    "DOFMap",
    "ExpansionResult",
    "FMACResult",
    "ModePair",
    "PairingResult",
    "align_geometry",
    "align_modes",
    "as_dofmap",
    "auto_orthogonality",
    "comac",
    "cross_orthogonality",
    "csac",
    "csf",
    "ecomac",
    "expand_guyan",
    "expand_serep",
    "fdac",
    "fmac",
    "frac",
    "frf_difference",
    "mac_matrix",
    "mac_pairs",
    "mac_value",
    "match_dofs",
    "modal_scale_factor",
    "off_diagonal_max",
    "orthogonality_error",
    "pair_modes",
    "parse_component",
    "parse_components",
    "parse_dof_label",
    "poc",
    "restrict",
    "rotate_modes",
]
