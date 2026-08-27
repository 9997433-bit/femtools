"""Solver driver contract: run an external FE solver through text files.

femtools is solver-independent: models cross the boundary as text decks
(BDF, CDB, UNV) and results come back as text (punch, UNV).  A *driver* is
the adapter that owns one such boundary -- it knows how to write the input
deck for its solver, how to launch the executable, and which result file
to read back.  :class:`SolverDriver` is the :class:`typing.Protocol` those
adapters implement; femtools itself ships **no** vendor drivers and **no**
proprietary binary parsers (OP2/RST/ODB are explicitly out of scope --
punch/UNV text output covers the modal exchange loop).

The contract is structural (PEP 544): any object with the right methods
qualifies, no registration or subclassing required.  A minimal Nastran
adapter built entirely from femtools translators looks like::

    import subprocess
    from pathlib import Path

    from femtools.io import read_pch, write_bdf

    class NastranPunchDriver:
        '''SOL 103 via a local Nastran installation, results via .pch.'''

        name = "nastran-punch"

        def __init__(self, executable: str = "nastran") -> None:
            self.executable = executable

        def is_available(self) -> bool:
            return shutil.which(self.executable) is not None

        def write_input(self, model, workdir) -> Path:
            deck = Path(workdir) / f"{model.name}.bdf"
            write_bdf(deck, model)   # caller appends SOL/case control
            return deck

        def run(self, input_file, timeout=None) -> Path:
            subprocess.run([self.executable, str(input_file)],
                           cwd=Path(input_file).parent,
                           check=True, timeout=timeout)
            return Path(input_file).with_suffix(".pch")

        def read_modal(self, result_file):
            return read_pch(result_file)

    driver: SolverDriver = NastranPunchDriver()   # structurally typed
    assert isinstance(driver, SolverDriver)       # runtime-checkable

Error convention: drivers should raise
:class:`~femtools.core.errors.SolverError` when the external run fails and
let the io-layer :class:`~femtools.core.errors.FileFormatError` subclasses
propagate from the translators.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..core.model import FEModel
    from ..core.results import ModalResult

__all__ = ["SolverDriver"]


@runtime_checkable
class SolverDriver(Protocol):
    """Adapter for one external FE solver (structural typing, PEP 544).

    Implementations provide the four-step exchange loop
    ``write_input -> run -> read_modal`` plus an availability probe;
    ``isinstance(obj, SolverDriver)`` checks conformance at runtime
    (member presence only, as usual for runtime-checkable protocols).
    """

    name: str
    """Short identifier of the solver/driver (e.g. ``"nastran-punch"``)."""

    def is_available(self) -> bool:
        """Whether the external solver can be launched on this machine
        (executable found, license reachable, ...).  Must not raise."""
        ...

    def write_input(self, model: FEModel, workdir: str | Path) -> Path:
        """Write the solver input deck for ``model`` into ``workdir``.

        Returns the path of the main input file (the one handed to
        :meth:`run`).  Translation losses follow the femtools io
        convention: aggregated ``UserWarning`` s, never silent.
        """
        ...

    def run(self, input_file: str | Path, timeout: float | None = None) -> Path:
        """Execute the solver on ``input_file`` and block until it finishes.

        Returns the path of the result file to hand to :meth:`read_modal`.
        Raises :class:`~femtools.core.errors.SolverError` when the run
        fails or ``timeout`` (seconds) expires.
        """
        ...

    def read_modal(self, result_file: str | Path) -> ModalResult:
        """Read frequencies and mode shapes from a solver result file
        (e.g. via :func:`femtools.io.read_pch` /
        :func:`femtools.io.read_unv`)."""
        ...
