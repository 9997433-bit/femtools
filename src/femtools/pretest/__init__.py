"""Pretest planning: target modes, sensor placement and mass loading.

Typical workflow, starting from a solved model::

    from femtools.fea.eigen import solve_modes
    from femtools.pretest import (
        effective_mass, select_target_modes, translational_dofs,
        effective_independence, mass_loading, node_coordinates,
    )

    modal = solve_modes(model, n_modes=30)
    em = effective_mass(modal, modal.M, dof_map=modal,
                        coords=node_coordinates(model, modal), freq_hz=modal.freq_hz)
    targets = select_target_modes(em, f_max=200.0, mass_fraction=0.9)

    cand = translational_dofs(modal, mode_index=targets)     # measurable rows
    sensors = effective_independence(cand.phi, 12, candidate_dofs=cand.dofs)
    check = mass_loading(cand.phi, cand.freq_hz, added_mass=0.005, dofs=sensors.index)

Every entry point also accepts plain arrays, so a test data set that never
went through the FEA kernel works exactly the same way.
"""

from __future__ import annotations

from .candidates import (
    CandidateSet,
    candidate_dofs,
    node_coordinates,
    translational_dofs,
)
from .efi import EFIResult, effective_independence, efi_distribution
from .mass_loading import MassLoadingResult, mass_loading, sensor_mass_limit
from .sensor import (
    SensorSelection,
    aggregate_by_node,
    eliminate_by_mac,
    nodal_kinetic_energy,
    select_by_kinetic_energy,
)
from .target_modes import (
    DIRECTIONS,
    EffectiveMassResult,
    TargetModeSelection,
    effective_mass,
    rigid_body_modes,
    select_target_modes,
)

__all__ = [
    "DIRECTIONS",
    "CandidateSet",
    "EFIResult",
    "EffectiveMassResult",
    "MassLoadingResult",
    "SensorSelection",
    "TargetModeSelection",
    "aggregate_by_node",
    "candidate_dofs",
    "effective_independence",
    "effective_mass",
    "efi_distribution",
    "eliminate_by_mac",
    "mass_loading",
    "nodal_kinetic_energy",
    "node_coordinates",
    "rigid_body_modes",
    "select_by_kinetic_energy",
    "select_target_modes",
    "sensor_mass_limit",
    "translational_dofs",
]
