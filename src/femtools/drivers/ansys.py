"""ANSYS MAPDL driver over text files: CDB in, punch/UNV text out.

:class:`AnsysCdbDriver` is the concrete
:class:`~femtools.drivers.base.SolverDriver` for a locally installed
ANSYS Mechanical APDL (batch command line: ``mapdl -b -i job.cdb -o
job.out``).  Like every femtools driver it is built entirely from the
femtools translators and the standard library -- **no** proprietary
binary parsers (no RST, by design; text punch/UNV output covers the
modal exchange loop):

* :meth:`~AnsysCdbDriver.write_input` writes the model with
  :func:`femtools.io.write_cdb` -- the deck is exactly the coded-database
  archive that translator produces (``CDREAD``-able model data; the
  analysis commands around it belong to the caller's APDL setup).
* :meth:`~AnsysCdbDriver.run` launches the executable found via
  :func:`shutil.which` (default aliases ``ansys`` and ``mapdl``) and
  raises :class:`~femtools.core.errors.SolverError` when it is missing,
  exits non-zero, times out or leaves no ``.pch``/``.unv`` text result
  next to the deck.
* :meth:`~AnsysCdbDriver.read_modal` reads text results back with
  :func:`femtools.io.read_pch` / :func:`femtools.io.read_unv`; handing
  it a ``.rst`` path raises :class:`~femtools.core.errors.SolverError`
  naming RST as N/A.

Nothing here requires ANSYS to be installed:
:meth:`~AnsysCdbDriver.is_available` probes the PATH and the test suite
exercises the loop with a stub executable (Round-7 convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..core.errors import SolverError
from ..io.cdb import write_cdb
from ._textrun import read_text_modal, resolve_executable, run_text_solver, sanitized_stem

if TYPE_CHECKING:
    from ..core.model import FEModel
    from ..core.results import ModalResult

__all__ = ["AnsysCdbDriver"]

#: Launcher aliases probed (in order) when no explicit executable is given.
_DEFAULT_EXECUTABLES = ("ansys", "mapdl")


class AnsysCdbDriver:
    """ANSYS MAPDL through a local installation, model as CDB, results as text.

    Implements the :class:`~femtools.drivers.base.SolverDriver` protocol
    (structurally; ``isinstance`` checks pass at runtime).

    Parameters
    ----------
    executable:
        Name (resolved on the PATH) or path of the MAPDL launcher.  When
        omitted, the ``ansys`` and ``mapdl`` aliases are probed in order.
    """

    name = "ansys-cdb"

    def __init__(self, executable: str | None = None) -> None:
        self.executable = None if executable is None else str(executable)

    def __repr__(self) -> str:
        return f"AnsysCdbDriver(executable={self.executable!r})"

    # -- SolverDriver protocol -------------------------------------------------
    def is_available(self) -> bool:
        """Whether an MAPDL launcher resolves on the PATH.  Never raises."""
        return resolve_executable(self.executable, _DEFAULT_EXECUTABLES) is not None

    def write_input(self, model: FEModel, workdir: str | Path) -> Path:
        """Write ``<model.name>.cdb`` into ``workdir`` via :func:`write_cdb`.

        The deck carries the same record subset (and the same aggregated
        translation warnings) as a direct :func:`femtools.io.write_cdb`
        call; solution commands are the caller's APDL setup.
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        deck = workdir / f"{sanitized_stem(model.name)}.cdb"
        write_cdb(deck, model)
        return deck

    def run(self, input_file: str | Path, timeout: float | None = None) -> Path:
        """Execute MAPDL in batch mode and return the text result path.

        Launches ``<exe> -b -i <deck> -o <deck>.out`` next to the deck and
        returns the ``.pch`` or ``.unv`` file the run produced.  Raises
        :class:`~femtools.core.errors.SolverError` when the executable is
        not available, exits non-zero, exceeds ``timeout`` (seconds) or
        produces no text modal result (a lone ``.rst`` is called out: RST
        is N/A in femtools).
        """
        input_file = Path(input_file)
        exe = resolve_executable(self.executable, _DEFAULT_EXECUTABLES)
        if exe is None:
            tried = self.executable or " / ".join(_DEFAULT_EXECUTABLES)
            raise SolverError(
                f"ANSYS executable {tried!r} not found on the PATH "
                "(AnsysCdbDriver.is_available() is False)"
            )
        cmd = [exe, "-b", "-i", str(input_file), "-o", str(input_file.with_suffix(".out"))]
        return run_text_solver("ANSYS", cmd, input_file, timeout, ".rst", "RST")

    def read_modal(self, result_file: str | Path) -> ModalResult:
        """Read frequencies and mode shapes from a ``.pch``/``.unv`` text file.

        A ``.rst`` path raises :class:`~femtools.core.errors.SolverError`:
        the RST binary format is N/A in femtools by design.
        """
        return read_text_modal("AnsysCdbDriver", result_file, ".rst", "RST")
