"""femtools.io — translators between external formats and the core database.

::

    from femtools.io import read_unv, write_unv      # Universal Files
    from femtools.io import read_bdf, write_bdf      # Nastran bulk data
    from femtools.io import read_pch, write_pch      # Nastran punch (modes)
    from femtools.io import read_pch_static          # Nastran punch (SOL 101 statics)
    from femtools.io import read_pch_stress          # Nastran punch ($STRESSES text)
    from femtools.io import read_cdb, write_cdb      # ANSYS coded database
    from femtools.io import read_inp, write_inp      # Abaqus input file (text)
    from femtools.io import read_k, write_k          # LS-DYNA keyword (text)
    from femtools.io import save_project, load_project  # .ftproj
"""

from __future__ import annotations

from .bdf import BdfError, read_bdf, write_bdf
from .cdb import CdbError, read_cdb, write_cdb
from .inp import InpError, read_inp, write_inp
from .kfile import KFileError, read_k, write_k
from .pch import (
    PchError,
    PchStressResult,
    read_pch,
    read_pch_static,
    read_pch_stress,
    write_pch,
)
from .project import Project, ProjectError, load_project, save_project
from .unv import Traceline, UnvData, UnvFunction, read_unv, write_unv

__all__ = [
    "read_unv",
    "write_unv",
    "UnvData",
    "UnvFunction",
    "Traceline",
    "read_bdf",
    "write_bdf",
    "BdfError",
    "read_pch",
    "read_pch_static",
    "read_pch_stress",
    "write_pch",
    "PchStressResult",
    "PchError",
    "read_cdb",
    "write_cdb",
    "CdbError",
    "read_inp",
    "write_inp",
    "InpError",
    "read_k",
    "write_k",
    "KFileError",
    "save_project",
    "load_project",
    "Project",
    "ProjectError",
]
