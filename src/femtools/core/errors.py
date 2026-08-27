"""femtools exception hierarchy (see ``docs/ARCHITECTURE.md`` section 9).

::

    FemtoolsError(Exception)
    ├── ModelError(ValueError)     # integrity: duplicate id, dangling reference,
    │   │                          #   bad element arity
    │   └── MeshError              # mesh/geometry integrity (degenerate elements,
    │                              #   orphan/duplicate nodes)
    ├── UnitError(ValueError)      # unknown unit names / inconsistent unit systems
    ├── FileFormatError(ValueError)# unparseable UNV/BDF/project content
    ├── AssemblyError              # degenerate geometry at assembly time, asymmetry
    ├── SolverError                # singular factorization, ARPACK breakdown
    │   └── ConvergenceError       # iterative process exceeded max_iter / diverged
    └── CompatibilityError         # mismatched DOF sets / frequency grids between
                                   #   result objects

Compatibility notes
-------------------
* ``ModelError``, ``UnitError`` and ``FileFormatError`` (and therefore the io
  subclasses ``BdfError`` / ``ProjectError``) still subclass ``ValueError``,
  so pre-existing ``except ValueError`` call sites keep working.
* Every class accepts a plain message (``raise SolverError("singular K")``);
  the optional structured context below is attached as attributes when given.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FemtoolsError",
    "ModelError",
    "MeshError",
    "UnitError",
    "FileFormatError",
    "AssemblyError",
    "SolverError",
    "ConvergenceError",
    "CompatibilityError",
]


class FemtoolsError(Exception):
    """Base class of every femtools-specific error."""


class ModelError(FemtoolsError, ValueError):
    """Invalid model construction: duplicate ids, dangling references,
    bad element arity, malformed DOF masks, ..."""


class MeshError(ModelError):
    """Mesh/geometry integrity error: degenerate or inverted elements,
    duplicate/orphan nodes, invalid connectivity."""


class UnitError(FemtoolsError, ValueError):
    """Unknown unit names or inconsistent unit systems."""


class FileFormatError(FemtoolsError, ValueError):
    """Unparseable external file content (UNV dataset, BDF card, project
    archive).  Optional context: ``file``, ``line``, ``dataset``, ``card``."""

    def __init__(
        self,
        *args: Any,
        file: str | None = None,
        line: int | None = None,
        dataset: int | None = None,
        card: str | None = None,
    ) -> None:
        super().__init__(*args)
        self.file = file
        self.line = line
        self.dataset = dataset
        self.card = card


class AssemblyError(FemtoolsError):
    """Element/global matrix assembly failed: degenerate geometry
    (zero-length bar, negative Jacobian), asymmetric result, ..."""


class SolverError(FemtoolsError):
    """Numerical solve failed: singular factorization, ARPACK breakdown.
    Optional context: ``dofs`` (suspect global DOF numbers or (node, dof)
    pairs)."""

    def __init__(self, *args: Any, dofs: Any | None = None) -> None:
        super().__init__(*args)
        self.dofs = dofs


class ConvergenceError(SolverError):
    """Iterative process exceeded ``max_iter`` or diverged.  Optional
    context: ``history`` (per-iteration residuals/objectives)."""

    def __init__(self, *args: Any, dofs: Any | None = None, history: Any | None = None) -> None:
        super().__init__(*args, dofs=dofs)
        self.history = history


class CompatibilityError(FemtoolsError):
    """Two result/model objects cannot be combined: mismatched DOF sets,
    different frequency grids, incompatible shapes."""
