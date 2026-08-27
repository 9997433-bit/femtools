"""femtools.gui — headless-friendly local web GUI.

Two interchangeable backends serve the same JSON API and single-page
front end:

- **stdlib** (:mod:`femtools.gui.server`): zero-dependency
  ``http.server`` implementation, always available.
- **fastapi** (:mod:`femtools.gui.webapp`): optional, used when
  ``fastapi`` and ``uvicorn`` are installed (``pip install
  femtools[web]``).

Entry point::

    from femtools.gui import run_gui
    run_gui(host="127.0.0.1", port=8765, backend="auto")
"""

from __future__ import annotations

from femtools.gui.server import run_gui
from femtools.gui.state import GuiState

__all__ = ["run_gui", "GuiState", "create_fastapi_app"]


def create_fastapi_app(state: GuiState | None = None):
    """Build the FastAPI application (raises ImportError without fastapi)."""
    from femtools.gui.webapp import create_app

    return create_app(state or GuiState())
