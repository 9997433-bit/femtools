"""femtools.drivers — adapter contracts for external FE solvers.

::

    from femtools.drivers import SolverDriver        # PEP 544 Protocol
    from femtools.drivers import NastranPunchDriver  # SOL 103 via text files
    from femtools.drivers import AnsysCdbDriver      # MAPDL via CDB + text results
    from femtools.drivers import AbaqusInpDriver     # Abaqus via INP + text results

femtools ships the contract (see :mod:`femtools.drivers.base`) plus
three concrete text-only adapters: :class:`NastranPunchDriver`, which
drives a locally installed Nastran through the BDF/punch translators of
:mod:`femtools.io`, and the Round-8 :class:`AnsysCdbDriver` /
:class:`AbaqusInpDriver`, which write their decks with
:func:`~femtools.io.write_cdb` / :func:`~femtools.io.write_inp` and read
modal results back from punch/UNV text files.  All three degrade
gracefully when no solver is installed.  Everything crosses the boundary
as text -- no proprietary binary result parsers (OP2/RST/ODB), by
design.
"""

from __future__ import annotations

from .abaqus import AbaqusInpDriver
from .ansys import AnsysCdbDriver
from .base import SolverDriver
from .nastran import NastranPunchDriver

__all__ = ["SolverDriver", "NastranPunchDriver", "AnsysCdbDriver", "AbaqusInpDriver"]
