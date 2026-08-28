"""femtools.drivers — adapter contracts for external FE solvers.

::

    from femtools.drivers import SolverDriver        # PEP 544 Protocol
    from femtools.drivers import NastranPunchDriver  # SOL 103 via text files

femtools ships the contract (see :mod:`femtools.drivers.base`) plus one
concrete text-only adapter: :class:`NastranPunchDriver`, which drives a
locally installed Nastran through the BDF/punch translators of
:mod:`femtools.io` (and degrades gracefully when no Nastran is
installed).  Everything crosses the boundary as text -- no proprietary
binary result parsers (OP2/RST/ODB), by design.
"""

from __future__ import annotations

from .base import SolverDriver
from .nastran import NastranPunchDriver

__all__ = ["SolverDriver", "NastranPunchDriver"]
