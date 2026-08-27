"""Pretest planning: target modes, sensor placement and mass loading.

Typical workflow::

    from femtools.pretest import (
        effective_mass, select_target_modes, effective_independence,
        eliminate_by_mac, mass_loading,
    )

    em = effective_mass(phi, M, coords=xyz, freq_hz=f)
    targets = select_target_modes(em, f_max=200.0, mass_fraction=0.9)
    sensors = effective_independence(phi[cand_rows][:, targets.indices], n_sensors=12,
                                     candidate_dofs=cand_ids)
    check = mass_loading(phi, f, added_mass=0.005, dofs=sensors.index)
"""

from __future__ import annotations

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
    "EFIResult",
    "EffectiveMassResult",
    "MassLoadingResult",
    "SensorSelection",
    "TargetModeSelection",
    "aggregate_by_node",
    "effective_independence",
    "effective_mass",
    "efi_distribution",
    "eliminate_by_mac",
    "mass_loading",
    "nodal_kinetic_energy",
    "rigid_body_modes",
    "select_by_kinetic_energy",
    "select_target_modes",
    "sensor_mass_limit",
]
