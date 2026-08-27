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

Round-4 transition tier: the frozen remaining API (``.agent_workspace/REMAINING.md``,
``docs/PRODUCT_MAP.md`` tag *R4-wip*) is pre-wired below as **provisional** lazy
exports — ``femtools.guyan``, ``femtools.estimate_h1``, ``femtools.read_pch``, … —
that resolve as soon as their defining module is merged into the tree. Until then,
attribute access raises a descriptive ``AttributeError`` (so ``hasattr`` probing and
``from femtools import <name>`` fail cleanly) and the name is kept out of
``__all__``; a Round-4 module that is absent can never break ``import femtools`` or
``from femtools import *``. Once merged and verified, names move from
``_PROVISIONAL_EXPORTS`` to ``_EXPORTS`` (and gain ``TYPE_CHECKING`` re-exports).
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Top-level name -> defining module (contract API of docs/CONTRACT_API.md,
# plus the exception hierarchy promised at package root by docs/ARCHITECTURE.md §9).
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
    "save_project": "femtools.io.project",
    "load_project": "femtools.io.project",
    # FEA
    "assemble_km": "femtools.fea.assemble",
    "AssemblyResult": "femtools.fea.assemble",
    "solve_static": "femtools.fea.static",
    "StaticResult": "femtools.fea.static",
    "solve_modes": "femtools.fea.eigen",
    "ModalResult": "femtools.fea.eigen",
    "available_elements": "femtools.fea.elements",
    # dynamics
    "modal_frf": "femtools.dynamics.frf",
    "direct_frf": "femtools.dynamics.frf",
    "FRFResult": "femtools.dynamics.frf",
    "harmonic_response": "femtools.dynamics.harmonic",
    "modal_based_assembly": "femtools.dynamics.mba",
    "craig_bampton": "femtools.dynamics.craig_bampton",
    "time_history": "femtools.dynamics.time_domain",
    "residual_vectors": "femtools.dynamics.residuals",
    # correlation
    "mac_matrix": "femtools.correlation.mac",
    "comac": "femtools.correlation.mac",
    "poc": "femtools.correlation.mac",
    "pair_modes": "femtools.correlation.pairing",
    "frac": "femtools.correlation.frf_corr",
    "csac": "femtools.correlation.frf_corr",
    "csf": "femtools.correlation.frf_corr",
    "cross_orthogonality": "femtools.correlation.orthogonality",
    # pretest
    "effective_mass": "femtools.pretest.target_modes",
    "select_target_modes": "femtools.pretest.target_modes",
    "effective_independence": "femtools.pretest.efi",
    "eliminate_by_mac": "femtools.pretest.sensor",
    "nodal_kinetic_energy": "femtools.pretest.sensor",
    # updating
    "sensitivity_matrix": "femtools.updating.sensitivity",
    "update_model": "femtools.updating.updater",
    "UpdateResult": "femtools.updating.updater",
    "identify_harmonic_forces": "femtools.updating.force_id",
    # optimization
    "size_optimize": "femtools.optimization.size",
    "topology_simp": "femtools.optimization.topology",
    "latin_hypercube": "femtools.optimization.doe",
    "full_factorial": "femtools.optimization.doe",
    # MPE / RBPE
    "poly_lscf": "femtools.mpe.p_lscf",
    "fdd": "femtools.mpe.fdd",
    "efdd": "femtools.mpe.fdd",
    "lsce": "femtools.mpe.lsce",
    "rigid_body_properties": "femtools.rbpe.rbfit",
    # scripting
    "ScriptEngine": "femtools.script.engine",
}

# Provisional top-level name -> defining module: the Round-4 API frozen in
# .agent_workspace/REMAINING.md (docs/PRODUCT_MAP.md tag R4-wip). These modules are
# merged by other Round-4 agents; each name resolves lazily once its module exists
# and raises a descriptive AttributeError until then (see _resolve_provisional).
_PROVISIONAL_EXPORTS: dict[str, str] = {
    # fea — condensation/expansion bases and complex modes (R4-O1)
    "guyan": "femtools.fea.reduction",
    "irs": "femtools.fea.reduction",
    "serep": "femtools.fea.reduction",
    "ReductionResult": "femtools.fea.reduction",
    "solve_complex_modes": "femtools.fea.eigen",
    "ComplexModalResult": "femtools.fea.eigen",
    # dynamics — free-interface CMS and random/PSD response (R4-O2)
    "rubin": "femtools.dynamics.cms_free",
    "macneal": "femtools.dynamics.cms_free",
    "FreeCMSResult": "femtools.dynamics.cms_free",
    "psd_response": "femtools.dynamics.random",
    "PSDResult": "femtools.dynamics.random",
    # pretest / correlation (R4-O3)
    "driving_point_residues": "femtools.pretest.exciter",
    "select_exciters": "femtools.pretest.exciter",
    "expand_guyan": "femtools.correlation.expansion",
    "expand_serep": "femtools.correlation.expansion",
    "fmac": "femtools.correlation.mac",
    "align_geometry": "femtools.correlation.alignment",
    # updating / optimization / mpe (R4-O4)
    "update_from_frf": "femtools.updating.frf_updating",
    "select_parameters": "femtools.updating.selection",
    "fit_rsm": "femtools.optimization.surrogate",
    "predict_rsm": "femtools.optimization.surrogate",
    "pareto_weighted": "femtools.optimization.multi",
    "estimate_h1": "femtools.mpe.frf_estimation",
    "estimate_h2": "femtools.mpe.frf_estimation",
    "coherence": "femtools.mpe.frf_estimation",
    "ssi_cov": "femtools.mpe.ssi",
    # io / drivers (R4-F2)
    "read_pch": "femtools.io.pch",
    "write_pch": "femtools.io.pch",
    "read_cdb": "femtools.io.cdb",
    "SolverDriver": "femtools.drivers.base",
}

# Subpackages reachable as attributes (femtools.fea, femtools.cli, ...).
_SUBMODULES = frozenset(
    {
        "cli",
        "core",
        "correlation",
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

# Subpackages scheduled for Round 4 that may not exist in the tree yet.
_PROVISIONAL_SUBMODULES = frozenset({"drivers"})

assert not set(_PROVISIONAL_EXPORTS) & set(_EXPORTS), "provisional name shadows a stable export"
assert not _PROVISIONAL_SUBMODULES & _SUBMODULES, "provisional subpackage shadows a stable one"

_PKG_DIR = os.path.dirname(__file__)


def _module_source(module_name: str) -> str | None:
    """Path of the module's source file if it is in the tree, else ``None``.

    Deliberately a filesystem probe and not ``importlib.util.find_spec``: ``find_spec``
    executes parent-package ``__init__`` modules (``femtools.fea`` and friends import
    numpy/scipy eagerly), which would defeat the cheap lazy import.
    """
    base = os.path.join(_PKG_DIR, *module_name.removeprefix("femtools.").split("."))
    for candidate in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(candidate):
            return candidate
    return None


def _defines_name(source_path: str, name: str) -> bool:
    """Heuristic: does the module source define ``name`` at top level?

    A source-text scan, again to avoid importing numpy-heavy modules at package-import
    time. It recognizes column-0 ``def``/``class``/assignment statements; exotic
    definitions (re-exports, ``__getattr__`` factories) are missed until the name is
    promoted into ``_EXPORTS`` after the Round-4 merge — a benign, transient miss.
    """
    prefixes = (
        f"def {name}(",
        f"async def {name}(",
        f"class {name}(",
        f"class {name}:",
        f"{name} = ",
        f"{name}: ",
    )
    try:
        with open(source_path, encoding="utf-8") as stream:
            return any(line.startswith(prefixes) for line in stream)
    except OSError:
        return False


def _available_provisional_names() -> list[str]:
    """Provisional names whose definitions are already present in the tree.

    Only these enter ``__all__`` (evaluated once, at import time): every ``__all__``
    entry must resolve or ``from femtools import *`` breaks, and the whole point of
    the provisional tier is that a missing Round-4 module never breaks an import.
    """
    names = [
        name
        for name, module_name in _PROVISIONAL_EXPORTS.items()
        if (path := _module_source(module_name)) is not None and _defines_name(path, name)
    ]
    names += [sub for sub in _PROVISIONAL_SUBMODULES if _module_source(f"femtools.{sub}")]
    return names


__all__ = [
    "__version__",
    *sorted(_EXPORTS),
    *sorted(_SUBMODULES),
    *sorted(_available_provisional_names()),
]


def _pending(name: str, module_name: str) -> AttributeError:
    return AttributeError(
        f"module {__name__!r} has no attribute {name!r} yet: it belongs to the frozen "
        f"Round-4 API expected in {module_name!r} (docs/PRODUCT_MAP.md tag R4-wip), "
        "which has not been merged into this tree."
    )


def _resolve_provisional(name: str) -> object:
    module_name = _PROVISIONAL_EXPORTS[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == module_name or module_name.startswith(missing + "."):
            # The scheduled module (or its scheduled parent package) is not in the
            # tree yet: report "pending", keep `hasattr`/star-import semantics clean.
            raise _pending(name, module_name) from None
        raise  # a merged module is missing a real dependency: surface it unchanged
    try:
        return getattr(module, name)
    except AttributeError:
        # Module merged but the symbol not added yet (e.g. femtools.fea.eigen exists
        # today without solve_complex_modes): same user-facing story as above.
        raise _pending(name, module_name) from None


def __getattr__(name: str) -> object:
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value  # cache: next access skips __getattr__
        return value
    if name in _PROVISIONAL_EXPORTS:
        value = _resolve_provisional(name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return import_module(f"femtools.{name}")
    if name in _PROVISIONAL_SUBMODULES:
        try:
            return import_module(f"femtools.{name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"femtools.{name}":
                raise _pending(name, f"femtools.{name}") from None
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_EXPORTS) | _SUBMODULES)


# Static-analysis view of the *stable* lazy exports (PEP 484 re-exports). Provisional
# Round-4 names intentionally have no entries here until their modules are merged:
# static imports of not-yet-existing modules would turn every type-check run red.
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
    from femtools.correlation.frf_corr import csac as csac
    from femtools.correlation.frf_corr import csf as csf
    from femtools.correlation.frf_corr import frac as frac
    from femtools.correlation.mac import comac as comac
    from femtools.correlation.mac import mac_matrix as mac_matrix
    from femtools.correlation.mac import poc as poc
    from femtools.correlation.orthogonality import cross_orthogonality as cross_orthogonality
    from femtools.correlation.pairing import pair_modes as pair_modes
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
    from femtools.dynamics.residuals import residual_vectors as residual_vectors
    from femtools.dynamics.time_domain import time_history as time_history
    from femtools.fea.assemble import AssemblyResult as AssemblyResult
    from femtools.fea.assemble import assemble_km as assemble_km
    from femtools.fea.eigen import ModalResult as ModalResult
    from femtools.fea.eigen import solve_modes as solve_modes
    from femtools.fea.elements import available_elements as available_elements
    from femtools.fea.static import StaticResult as StaticResult
    from femtools.fea.static import solve_static as solve_static
    from femtools.io.bdf import read_bdf as read_bdf
    from femtools.io.bdf import write_bdf as write_bdf
    from femtools.io.project import load_project as load_project
    from femtools.io.project import save_project as save_project
    from femtools.io.unv import read_unv as read_unv
    from femtools.io.unv import write_unv as write_unv
    from femtools.mpe.fdd import efdd as efdd
    from femtools.mpe.fdd import fdd as fdd
    from femtools.mpe.lsce import lsce as lsce
    from femtools.mpe.p_lscf import poly_lscf as poly_lscf
    from femtools.optimization.doe import (
        full_factorial as full_factorial,
    )
    from femtools.optimization.doe import (
        latin_hypercube as latin_hypercube,
    )
    from femtools.optimization.size import size_optimize as size_optimize
    from femtools.optimization.topology import topology_simp as topology_simp
    from femtools.pretest.efi import effective_independence as effective_independence
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
    from femtools.updating.sensitivity import sensitivity_matrix as sensitivity_matrix
    from femtools.updating.updater import UpdateResult as UpdateResult
    from femtools.updating.updater import update_model as update_model
