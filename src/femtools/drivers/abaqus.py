"""Abaqus driver over text files: INP in, punch/UNV text out.

:class:`AbaqusInpDriver` is the concrete
:class:`~femtools.drivers.base.SolverDriver` for a locally installed
Abaqus (standard command line: ``abaqus job=<name> input=<deck>
interactive``).  Like every femtools driver it is built entirely from
the femtools translators and the standard library -- **no** proprietary
binary parsers (no ODB, by design; text punch/UNV output covers the
modal exchange loop):

* :meth:`~AbaqusInpDriver.write_input` writes the model with
  :func:`femtools.io.write_inp` -- the deck is exactly the input-file
  text subset that translator produces (mesh, sections, materials; the
  ``*STEP`` block belongs to the caller's analysis setup).
* :meth:`~AbaqusInpDriver.run` launches the executable found via
  :func:`shutil.which` (default alias ``abaqus``) and raises
  :class:`~femtools.core.errors.SolverError` when it is missing, exits
  non-zero, times out or leaves no ``.pch``/``.unv`` text result next to
  the deck.
* :meth:`~AbaqusInpDriver.read_modal` reads text results back with
  :func:`femtools.io.read_pch` / :func:`femtools.io.read_unv`; handing
  it an ``.odb`` path raises :class:`~femtools.core.errors.SolverError`
  naming ODB as N/A.

Nothing here requires Abaqus to be installed:
:meth:`~AbaqusInpDriver.is_available` probes the PATH and the test suite
exercises the loop with a stub executable (Round-7 convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..core.errors import SolverError
from ..io.inp import write_inp
from ._textrun import read_text_modal, resolve_executable, run_text_solver, sanitized_stem

if TYPE_CHECKING:
    from ..core.model import FEModel
    from ..core.results import ModalResult

__all__ = ["AbaqusInpDriver"]

#: Launcher aliases probed (in order) when no explicit executable is given.
_DEFAULT_EXECUTABLES = ("abaqus",)


class AbaqusInpDriver:
    """Abaqus through a local installation, model as INP, results as text.

    Implements the :class:`~femtools.drivers.base.SolverDriver` protocol
    (structurally; ``isinstance`` checks pass at runtime).

    Parameters
    ----------
    executable:
        Name (resolved on the PATH) or path of the Abaqus launcher.
        Defaults to the ``abaqus`` alias.
    """

    name = "abaqus-inp"

    def __init__(self, executable: str | None = None) -> None:
        self.executable = None if executable is None else str(executable)

    def __repr__(self) -> str:
        return f"AbaqusInpDriver(executable={self.executable!r})"

    # -- SolverDriver protocol -------------------------------------------------
    def is_available(self) -> bool:
        """Whether an Abaqus launcher resolves on the PATH.  Never raises."""
        return resolve_executable(self.executable, _DEFAULT_EXECUTABLES) is not None

    def write_input(self, model: FEModel, workdir: str | Path) -> Path:
        """Write ``<model.name>.inp`` into ``workdir`` via :func:`write_inp`.

        The deck carries the same keyword subset (and the same aggregated
        translation warnings) as a direct :func:`femtools.io.write_inp`
        call; the ``*STEP`` block is the caller's analysis setup.
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        deck = workdir / f"{sanitized_stem(model.name)}.inp"
        write_inp(deck, model)
        return deck

    def run(self, input_file: str | Path, timeout: float | None = None) -> Path:
        """Execute Abaqus interactively and return the text result path.

        Launches ``<exe> job=<stem> input=<deck> interactive`` next to the
        deck and returns the ``.pch`` or ``.unv`` file the run produced.
        Raises :class:`~femtools.core.errors.SolverError` when the
        executable is not available, exits non-zero, exceeds ``timeout``
        (seconds) or produces no text modal result (a lone ``.odb`` is
        called out: ODB is N/A in femtools).
        """
        input_file = Path(input_file)
        exe = resolve_executable(self.executable, _DEFAULT_EXECUTABLES)
        if exe is None:
            tried = self.executable or " / ".join(_DEFAULT_EXECUTABLES)
            raise SolverError(
                f"Abaqus executable {tried!r} not found on the PATH "
                "(AbaqusInpDriver.is_available() is False)"
            )
        cmd = [exe, f"job={input_file.stem}", f"input={input_file.name}", "interactive"]
        return run_text_solver("Abaqus", cmd, input_file, timeout, ".odb", "ODB")

    def read_modal(self, result_file: str | Path) -> ModalResult:
        """Read frequencies and mode shapes from a ``.pch``/``.unv`` text file.

        An ``.odb`` path raises :class:`~femtools.core.errors.SolverError`:
        the ODB binary format is N/A in femtools by design.
        """
        return read_text_modal("AbaqusInpDriver", result_file, ".odb", "ODB")
