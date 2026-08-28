"""femtools.viz — plotting for models and results.

All functions work headless (Agg backend is selected automatically when
no display is present), accept an optional ``ax=`` to draw into an
existing figure, and return the :class:`matplotlib.figure.Figure`.
Pass ``outfile=`` to save directly to disk.

matplotlib is the default and only required plotting backend.  When the
optional ``plotly`` package is installed, every plot function also
accepts ``backend="plotly"`` (see :func:`set_default_backend` /
:func:`plotly_available`); plotly is never imported unless a plotly
plot is requested, so importing :mod:`femtools.viz` never needs it.

:func:`plot_mesh3d` renders a shaded 3-D view of a model through the
optional ``pyvista`` package when it is importable (see
:func:`pyvista_available`) and falls back to the matplotlib 3-D
wireframe otherwise; like plotly, pyvista is never imported by
``import femtools.viz`` itself.  :func:`plot_stress` colors the mesh by
a recovered stress field (von Mises by default) with matplotlib, or
through the same optional pyvista path with ``backend="pyvista"``.

:mod:`femtools.viz.report` additionally builds self-contained HTML or
plain-text MAC correlation reports (matplotlib is optional there: the
embedded heatmap is skipped when it is unavailable).
"""

from __future__ import annotations

from femtools.viz.plots import (
    get_default_backend,
    plot_frf,
    plot_mac,
    plot_mesh,
    plot_mesh3d,
    plot_mode,
    plot_psd,
    plot_stress,
    plotly_available,
    pyvista_available,
    set_default_backend,
)
from femtools.viz.report import mac_report_html, mac_report_text, save_mac_report

__all__ = [
    "plot_mesh",
    "plot_mesh3d",
    "plot_stress",
    "plot_mac",
    "plot_frf",
    "plot_psd",
    "plot_mode",
    "plotly_available",
    "pyvista_available",
    "get_default_backend",
    "set_default_backend",
    "mac_report_html",
    "mac_report_text",
    "save_mac_report",
]
