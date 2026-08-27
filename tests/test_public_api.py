"""Smoke tests for the frozen top-level public API."""

from __future__ import annotations

import femtools


PUBLIC_API = {
    "AssemblyResult",
    "CoordSys",
    "DOFSet",
    "Element",
    "ElementSet",
    "FEModel",
    "FRFResult",
    "Material",
    "ModalResult",
    "Node",
    "NodeSet",
    "Property",
    "SPC",
    "ScriptEngine",
    "StaticResult",
    "UnitSystem",
    "UpdateResult",
    "assemble_km",
    "available_elements",
    "comac",
    "craig_bampton",
    "cross_orthogonality",
    "csac",
    "csf",
    "direct_frf",
    "effective_independence",
    "effective_mass",
    "efdd",
    "eliminate_by_mac",
    "fdd",
    "frac",
    "full_factorial",
    "harmonic_response",
    "identify_harmonic_forces",
    "latin_hypercube",
    "load_project",
    "lsce",
    "mac_matrix",
    "modal_based_assembly",
    "modal_frf",
    "nodal_kinetic_energy",
    "pair_modes",
    "poc",
    "poly_lscf",
    "read_bdf",
    "read_unv",
    "residual_vectors",
    "rigid_body_properties",
    "save_project",
    "select_target_modes",
    "sensitivity_matrix",
    "size_optimize",
    "solve_modes",
    "solve_static",
    "time_history",
    "topology_simp",
    "update_model",
    "validate_model",
    "write_bdf",
    "write_unv",
}


def test_frozen_top_level_public_api_imports() -> None:
    """Every documented top-level symbol resolves through the lazy importer."""
    assert PUBLIC_API <= set(femtools.__all__)
    assert all(getattr(femtools, name) is not None for name in PUBLIC_API)
