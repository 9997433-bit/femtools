"""femtools.drivers — adapter contracts for external FE solvers.

::

    from femtools.drivers import SolverDriver   # PEP 544 Protocol

femtools ships the contract only (see :mod:`femtools.drivers.base`):
vendor adapters live with the user, built on the text translators of
:mod:`femtools.io` (BDF/punch for Nastran, CDB for ANSYS, UNV for
everything else).  No proprietary binary result parsers, by design.
"""

from __future__ import annotations

from .base import SolverDriver

__all__ = ["SolverDriver"]
