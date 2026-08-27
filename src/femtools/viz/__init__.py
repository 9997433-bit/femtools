"""femtools.viz — matplotlib plotting for models and results.

All functions work headless (Agg backend is selected automatically when
no display is present), accept an optional ``ax=`` to draw into an
existing figure, and return the :class:`matplotlib.figure.Figure`.
Pass ``outfile=`` to save directly to disk.
"""

from __future__ import annotations

from femtools.viz.plots import plot_frf, plot_mac, plot_mesh, plot_mode

__all__ = ["plot_mesh", "plot_mac", "plot_frf", "plot_mode"]
