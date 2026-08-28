"""Optional FastAPI GUI backend (``pip install femtools[web]``).

Mirrors the stdlib server's API exactly; both are adapters over
:class:`femtools.gui.state.GuiState`.  Importing this module requires
``fastapi``.

Note: no ``from __future__ import annotations`` here — FastAPI must
evaluate the endpoint annotations eagerly because the request model is a
function-local class.
"""

from typing import Any

from femtools.gui.page import INDEX_HTML
from femtools.gui.state import GuiApiError, GuiState

__all__ = ["create_app"]


def create_app(state: GuiState | None = None) -> Any:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from pydantic import BaseModel

    import femtools

    gui = state or GuiState()
    app = FastAPI(
        title="femtools GUI",
        version=femtools.__version__,
        description="Headless-friendly web front end for the femtools framework.",
    )

    class ScriptRequest(BaseModel):
        source: str

    class LoadRequest(BaseModel):
        path: str

    @app.exception_handler(GuiApiError)
    async def _api_error(_request: Any, exc: GuiApiError) -> JSONResponse:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/api/status")
    async def status() -> dict:
        return gui.status()

    @app.get("/api/model")
    async def model() -> dict:
        return gui.model_summary()

    @app.get("/api/results")
    async def results() -> dict:
        return gui.results_summary()

    @app.get("/api/stress")
    async def stress(name: str = "", max_rows: int = 20) -> dict:
        return gui.stress_table(name=name or None, max_rows=max_rows)

    @app.post("/api/script")
    async def run_script(request: ScriptRequest) -> dict:
        return gui.run_script(request.source)

    @app.post("/api/load")
    async def load_model(request: LoadRequest) -> dict:
        return gui.load_model(request.path)

    @app.get("/api/load")
    async def load_model_by_query(path: str = "") -> dict:
        return gui.load_model(path)

    @app.get("/api/plot/{kind}")
    async def plot(kind: str, name: str = "mac", index: int = 0) -> Response:
        png = gui.render_png(kind, name=name, index=index)
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    return app
