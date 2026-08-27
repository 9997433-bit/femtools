"""femtools.viz — matplotlib plotting for models and results.

All functions work headless (Agg backend is selected automatically when
no display is present), accept an optional ``ax=`` to draw into an
existing figure, and return the :class:`matplotlib.figure.Figure`.
Pass ``outfile=`` to save directly to disk.

:mod:`femtools.viz.report` additionally builds self-contained HTML or
plain-text MAC correlation reports (matplotlib is optional there: the
embedded heatmap is skipped when it is unavailable).
"""

from __future__ import annotations

from femtools.viz.plots import plot_frf, plot_mac, plot_mesh, plot_mode
from femtools.viz.report import mac_report_html, mac_report_text, save_mac_report

__all__ = [
    "plot_mesh",
    "plot_mac",
    "plot_frf",
    "plot_mode",
    "mac_report_html",
    "mac_report_text",
    "save_mac_report",
]
