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
    plot_mode,
    plotly_available,
    set_default_backend,
)
from femtools.viz.report import mac_report_html, mac_report_text, save_mac_report

__all__ = [
    "plot_mesh",
    "plot_mac",
    "plot_frf",
    "plot_mode",
    "plotly_available",
    "get_default_backend",
    "set_default_backend",
    "mac_report_html",
    "mac_report_text",
    "save_mac_report",
]
