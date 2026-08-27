"""femtools — solver-independent structural dynamics CAE framework.

Functional equivalent (original implementation) of the FEMtools product family:
Framework, Dynamics, Pretest & Correlation, Model Updating, Optimization, MPE, RBPE.

The frozen public API of ``docs/CONTRACT_API.md`` is re-exported at the top level::

    import femtools

    model = femtools.FEModel(name="beam")
    modal = femtools.solve_modes(model, n_modes=10)
    mac = femtools.mac_matrix(modal.modes, modal.modes)

Symbols are resolved lazily (PEP 562): ``import femtools`` stays cheap and free of
circular imports (``femtools.cli`` / ``femtools.gui`` import ``femtools`` themselves),
while ``femtools.<name>`` triggers the import of only the subpackage that defines it.
The exception hierarchy of ``femtools.core.errors`` (``FemtoolsError`` and subclasses,
``docs/ARCHITECTURE.md`` §9) is re-exported the same way.

Name-collision policy: where a class name exists in more than one subpackage, the
top level exports the type produced by the contract entry points — ``ModalResult``
and ``StaticResult`` from ``femtools.fea``, ``FRFResult`` from ``femtools.dynamics``,
``AssemblyResult`` from ``femtools.fea.assemble``. The containers of
``femtools.core.results`` remain available under their subpackage path.

Round-4 API (``docs/PRODUCT_MAP.md`` tag *R4*): the names frozen in
``.agent_workspace/REMAINING.md`` — ``femtools.guyan``, ``femtools.rubin``,
``femtools.estimate_h1``, ``femtools.read_pch``, ``femtools.SolverDriver``, … — were
carried through Round 4 as a *provisional* lazy tier and are **stable** exports since
Round 5: every module is merged, so they live in ``_EXPORTS`` (with ``TYPE_CHECKING``
re-exports) like the rest of the contract, and the provisional machinery is retired.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Top-level name -> defining module (contract API of docs/CONTRACT_API.md, the frozen
# Round-4 API of .agent_workspace/REMAINING.md — docs/PRODUCT_MAP.md tag R4 — plus the
# exception hierarchy promised at package root by docs/ARCHITECTURE.md §9).
_EXPORTS: dict[str, str] = {
    # error hierarchy
    "FemtoolsError": "femtools.core.errors",
    "ModelError": "femtools.core.errors",
    "MeshError": "femtools.core.errors",
    "UnitError": "femtools.core.errors",
    "FileFormatError": "femtools.core.errors",
    "AssemblyError": "femtools.core.errors",
    "SolverError": "femtools.core.errors",
    "ConvergenceError": "femtools.core.errors",
    "CompatibilityError": "femtools.core.errors",
    # core database
    "FEModel": "femtools.core.model",
    "Node": "femtools.core.model",
    "Element": "femtools.core.model",
    "Material": "femtools.core.model",
    "Property": "femtools.core.model",
    "SPC": "femtools.core.model",
    "DOFSet": "femtools.core.model",
    "NodeSet": "femtools.core.sets",
    "ElementSet": "femtools.core.sets",
    "UnitSystem": "femtools.core.units",
    "CoordSys": "femtools.core.coords",
    "validate_model": "femtools.core.validation",
    # I/O
    "read_unv": "femtools.io.unv",
    "write_unv": "femtools.io.unv",
    "read_bdf": "femtools.io.bdf",
    "write_bdf": "femtools.io.bdf",
    "read_pch": "femtools.io.pch",
    "write_pch": "femtools.io.pch",
    "read_cdb": "femtools.io.cdb",
    "save_project": "femtools.io.project",
    "load_project": "femtools.io.project",
    # solver drivers
    "SolverDriver": "femtools.drivers.base",
    # FEA
    "assemble_km": "femtools.fea.assemble",
    "AssemblyResult": "femtools.fea.assemble",
    "solve_static": "femtools.fea.static",
    "StaticResult": "femtools.fea.static",
    "solve_modes": "femtools.fea.eigen",
    "ModalResult": "femtools.fea.eigen",
    "solve_complex_modes": "femtools.fea.eigen",
    "ComplexModalResult": "femtools.fea.eigen",
    "available_elements": "femtools.fea.elements",
    "guyan": "femtools.fea.reduction",
    "irs": "femtools.fea.reduction",
    "serep": "femtools.fea.reduction",
    "ReductionResult": "femtools.fea.reduction",
    # dynamics
    "modal_frf": "femtools.dynamics.frf",
    "direct_frf": "femtools.dynamics.frf",
    "FRFResult": "femtools.dynamics.frf",
    "harmonic_response": "femtools.dynamics.harmonic",
    "modal_based_assembly": "femtools.dynamics.mba",
    "craig_bampton": "femtools.dynamics.craig_bampton",
    "rubin": "femtools.dynamics.cms_free",
    "macneal": "femtools.dynamics.cms_free",
    "FreeCMSResult": "femtools.dynamics.cms_free",
    "psd_response": "femtools.dynamics.random",
    "PSDResult": "femtools.dynamics.random",
    "time_history": "femtools.dynamics.time_domain",
    "residual_vectors": "femtools.dynamics.residuals",
    # correlation
    "mac_matrix": "femtools.correlation.mac",
    "comac": "femtools.correlation.mac",
    "poc": "femtools.correlation.mac",
    "fmac": "femtools.correlation.mac",
    "pair_modes": "femtools.correlation.pairing",
    "frac": "femtools.correlation.frf_corr",
    "csac": "femtools.correlation.frf_corr",
    "csf": "femtools.correlation.frf_corr",
    "cross_orthogonality": "femtools.correlation.orthogonality",
    "expand_guyan": "femtools.correlation.expansion",
    "expand_serep": "femtools.correlation.expansion",
    "align_geometry": "femtools.correlation.alignment",
    # pretest
    "effective_mass": "femtools.pretest.target_modes",
    "select_target_modes": "femtools.pretest.target_modes",
    "effective_independence": "femtools.pretest.efi",
    "eliminate_by_mac": "femtools.pretest.sensor",
    "nodal_kinetic_energy": "femtools.pretest.sensor",
    "driving_point_residues": "femtools.pretest.exciter",
    "select_exciters": "femtools.pretest.exciter",
    # updating
    "sensitivity_matrix": "femtools.updating.sensitivity",
    "update_model": "femtools.updating.updater",
    "UpdateResult": "femtools.updating.updater",
    "update_from_frf": "femtools.updating.frf_updating",
    "select_parameters": "femtools.updating.selection",
    "identify_harmonic_forces": "femtools.updating.force_id",
    # optimization
    "size_optimize": "femtools.optimization.size",
    "topology_simp": "femtools.optimization.topology",
    "latin_hypercube": "femtools.optimization.doe",
    "full_factorial": "femtools.optimization.doe",
    "fit_rsm": "femtools.optimization.surrogate",
    "predict_rsm": "femtools.optimization.surrogate",
    "pareto_weighted": "femtools.optimization.multi",
    # MPE / RBPE
    "poly_lscf": "femtools.mpe.p_lscf",
    "fdd": "femtools.mpe.fdd",
    "efdd": "femtools.mpe.fdd",
    "lsce": "femtools.mpe.lsce",
    "estimate_h1": "femtools.mpe.frf_estimation",
    "estimate_h2": "femtools.mpe.frf_estimation",
    "coherence": "femtools.mpe.frf_estimation",
    "ssi_cov": "femtools.mpe.ssi",
    "rigid_body_properties": "femtools.rbpe.rbfit",
    # scripting
    "ScriptEngine": "femtools.script.engine",
}

# Subpackages reachable as attributes (femtools.fea, femtools.cli, ...).
_SUBMODULES = frozenset(
    {
        "cli",
        "core",
        "correlation",
        "drivers",
        "dynamics",
        "fea",
        "gui",
        "io",
        "mpe",
        "optimization",
        "pretest",
        "rbpe",
        "script",
        "updating",
        "viz",
    }
)

__all__ = [
    "__version__",
    *sorted(_EXPORTS),
    *sorted(_SUBMODULES),
]


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value  # cache: next access skips __getattr__
        return value
    if name in _SUBMODULES:
        return import_module(f"femtools.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_EXPORTS) | _SUBMODULES)


# Static-analysis view of the lazy exports (PEP 484 re-exports).
if TYPE_CHECKING:
    from femtools.core.coords import CoordSys as CoordSys
    from femtools.core.errors import (
        AssemblyError as AssemblyError,
    )
    from femtools.core.errors import (
        CompatibilityError as CompatibilityError,
    )
    from femtools.core.errors import (
        ConvergenceError as ConvergenceError,
    )
    from femtools.core.errors import (
        FemtoolsError as FemtoolsError,
    )
    from femtools.core.errors import (
        FileFormatError as FileFormatError,
    )
    from femtools.core.errors import (
        MeshError as MeshError,
    )
    from femtools.core.errors import (
        ModelError as ModelError,
    )
    from femtools.core.errors import (
        SolverError as SolverError,
    )
    from femtools.core.errors import (
        UnitError as UnitError,
    )
    from femtools.core.model import (
        SPC as SPC,
    )
    from femtools.core.model import (
        DOFSet as DOFSet,
    )
    from femtools.core.model import (
        Element as Element,
    )
    from femtools.core.model import (
        FEModel as FEModel,
    )
    from femtools.core.model import (
        Material as Material,
    )
    from femtools.core.model import (
        Node as Node,
    )
    from femtools.core.model import (
        Property as Property,
    )
    from femtools.core.sets import ElementSet as ElementSet
    from femtools.core.sets import NodeSet as NodeSet
    from femtools.core.units import UnitSystem as UnitSystem
    from femtools.core.validation import validate_model as validate_model
    from femtools.correlation.alignment import align_geometry as align_geometry
    from femtools.correlation.expansion import (
        expand_guyan as expand_guyan,
    )
    from femtools.correlation.expansion import (
        expand_serep as expand_serep,
    )
    from femtools.correlation.frf_corr import csac as csac
    from femtools.correlation.frf_corr import csf as csf
    from femtools.correlation.frf_corr import frac as frac
    from femtools.correlation.mac import comac as comac
    from femtools.correlation.mac import fmac as fmac
    from femtools.correlation.mac import mac_matrix as mac_matrix
    from femtools.correlation.mac import poc as poc
    from femtools.correlation.orthogonality import cross_orthogonality as cross_orthogonality
    from femtools.correlation.pairing import pair_modes as pair_modes
    from femtools.drivers.base import SolverDriver as SolverDriver
    from femtools.dynamics.cms_free import (
        FreeCMSResult as FreeCMSResult,
    )
    from femtools.dynamics.cms_free import (
        macneal as macneal,
    )
    from femtools.dynamics.cms_free import (
        rubin as rubin,
    )
    from femtools.dynamics.craig_bampton import craig_bampton as craig_bampton
    from femtools.dynamics.frf import (
        FRFResult as FRFResult,
    )
    from femtools.dynamics.frf import (
        direct_frf as direct_frf,
    )
    from femtools.dynamics.frf import (
        modal_frf as modal_frf,
    )
    from femtools.dynamics.harmonic import harmonic_response as harmonic_response
    from femtools.dynamics.mba import modal_based_assembly as modal_based_assembly
    from femtools.dynamics.random import (
        PSDResult as PSDResult,
    )
    from femtools.dynamics.random import (
        psd_response as psd_response,
    )
    from femtools.dynamics.residuals import residual_vectors as residual_vectors
    from femtools.dynamics.time_domain import time_history as time_history
    from femtools.fea.assemble import AssemblyResult as AssemblyResult
    from femtools.fea.assemble import assemble_km as assemble_km
    from femtools.fea.eigen import (
        ComplexModalResult as ComplexModalResult,
    )
    from femtools.fea.eigen import (
        ModalResult as ModalResult,
    )
    from femtools.fea.eigen import (
        solve_complex_modes as solve_complex_modes,
    )
    from femtools.fea.eigen import (
        solve_modes as solve_modes,
    )
    from femtools.fea.elements import available_elements as available_elements
    from femtools.fea.reduction import (
        ReductionResult as ReductionResult,
    )
    from femtools.fea.reduction import (
        guyan as guyan,
    )
    from femtools.fea.reduction import (
        irs as irs,
    )
    from femtools.fea.reduction import (
        serep as serep,
    )
    from femtools.fea.static import StaticResult as StaticResult
    from femtools.fea.static import solve_static as solve_static
    from femtools.io.bdf import read_bdf as read_bdf
    from femtools.io.bdf import write_bdf as write_bdf
    from femtools.io.cdb import read_cdb as read_cdb
    from femtools.io.pch import read_pch as read_pch
    from femtools.io.pch import write_pch as write_pch
    from femtools.io.project import load_project as load_project
    from femtools.io.project import save_project as save_project
    from femtools.io.unv import read_unv as read_unv
    from femtools.io.unv import write_unv as write_unv
    from femtools.mpe.fdd import efdd as efdd
    from femtools.mpe.fdd import fdd as fdd
    from femtools.mpe.frf_estimation import (
        coherence as coherence,
    )
    from femtools.mpe.frf_estimation import (
        estimate_h1 as estimate_h1,
    )
    from femtools.mpe.frf_estimation import (
        estimate_h2 as estimate_h2,
    )
    from femtools.mpe.lsce import lsce as lsce
    from femtools.mpe.p_lscf import poly_lscf as poly_lscf
    from femtools.mpe.ssi import ssi_cov as ssi_cov
    from femtools.optimization.doe import (
        full_factorial as full_factorial,
    )
    from femtools.optimization.doe import (
        latin_hypercube as latin_hypercube,
    )
    from femtools.optimization.multi import pareto_weighted as pareto_weighted
    from femtools.optimization.size import size_optimize as size_optimize
    from femtools.optimization.surrogate import (
        fit_rsm as fit_rsm,
    )
    from femtools.optimization.surrogate import (
        predict_rsm as predict_rsm,
    )
    from femtools.optimization.topology import topology_simp as topology_simp
    from femtools.pretest.efi import effective_independence as effective_independence
    from femtools.pretest.exciter import (
        driving_point_residues as driving_point_residues,
    )
    from femtools.pretest.exciter import (
        select_exciters as select_exciters,
    )
    from femtools.pretest.sensor import (
        eliminate_by_mac as eliminate_by_mac,
    )
    from femtools.pretest.sensor import (
        nodal_kinetic_energy as nodal_kinetic_energy,
    )
    from femtools.pretest.target_modes import (
        effective_mass as effective_mass,
    )
    from femtools.pretest.target_modes import (
        select_target_modes as select_target_modes,
    )
    from femtools.rbpe.rbfit import rigid_body_properties as rigid_body_properties
    from femtools.script.engine import ScriptEngine as ScriptEngine
    from femtools.updating.force_id import identify_harmonic_forces as identify_harmonic_forces
    from femtools.updating.frf_updating import update_from_frf as update_from_frf
    from femtools.updating.selection import select_parameters as select_parameters
    from femtools.updating.sensitivity import sensitivity_matrix as sensitivity_matrix
    from femtools.updating.updater import UpdateResult as UpdateResult
    from femtools.updating.updater import update_model as update_model
