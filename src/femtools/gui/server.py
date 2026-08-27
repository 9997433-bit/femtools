"""Zero-dependency GUI backend on stdlib ``http.server``.

Also hosts :func:`run_gui`, the single entry point that picks the
backend: ``auto`` prefers FastAPI + uvicorn when installed and silently
falls back to this stdlib server otherwise.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from femtools.gui.page import INDEX_HTML
from femtools.gui.state import GuiApiError, GuiState

__all__ = ["run_gui", "make_stdlib_server", "FemtoolsRequestHandler"]


class FemtoolsRequestHandler(BaseHTTPRequestHandler):
    """Routes the shared GUI API; the state lives on the server object."""

    server_version = "femtools-gui"
    protocol_version = "HTTP/1.1"

    # -- helpers --------------------------------------------------------
    @property
    def state(self) -> GuiState:
        return self.server.gui_state  # type: ignore[attr-defined]

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def _send_error_json(self, message: str, code: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, code=code)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # keep the console quiet; errors surface via HTTP responses

    # -- routing --------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/status":
                self._send_json(self.state.status())
            elif url.path == "/api/model":
                self._send_json(self.state.model_summary())
            elif url.path == "/api/results":
                self._send_json(self.state.results_summary())
            elif url.path.startswith("/api/plot/"):
                kind = url.path.rsplit("/", 1)[-1]
                png = self.state.render_png(
                    kind,
                    name=query.get("name", "mac"),
                    index=int(query.get("index", 0)),
                )
                self._send(200, png, "image/png")
            else:
                self._send_error_json(f"not found: {url.path}", code=404)
        except GuiApiError as exc:
            self._send_error_json(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self._send_error_json(f"{type(exc).__name__}: {exc}", code=500)

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        url = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                self._send_error_json("request body must be JSON")
                return
            if url.path == "/api/script":
                self._send_json(self.state.run_script(payload.get("source", "")))
            else:
                self._send_error_json(f"not found: {url.path}", code=404)
        except GuiApiError as exc:
            self._send_error_json(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self._send_error_json(f"{type(exc).__name__}: {exc}", code=500)


def make_stdlib_server(host: str, port: int, state: GuiState) -> ThreadingHTTPServer:
    """Create (but do not start) the stdlib GUI server."""
    server = ThreadingHTTPServer((host, port), FemtoolsRequestHandler)
    server.gui_state = state  # type: ignore[attr-defined]
    return server


def _maybe_open_browser(url: str, enabled: bool) -> None:
    if not enabled:
        return
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()


def run_gui(
    host: str = "127.0.0.1",
    port: int = 8765,
    backend: str = "auto",
    model: Any = None,
    open_browser: bool = False,
    state: GuiState | None = None,
) -> None:
    """Run the femtools GUI server (blocks until Ctrl-C).

    ``backend``: ``"stdlib"`` forces the zero-dependency server,
    ``"fastapi"`` requires fastapi + uvicorn, ``"auto"`` prefers FastAPI
    and falls back to stdlib.  Runs fully headless; set
    ``open_browser=True`` to launch a local browser.
    """
    if backend not in ("auto", "stdlib", "fastapi"):
        raise ValueError(f"backend must be 'auto', 'stdlib' or 'fastapi', got {backend!r}")
    state = state or GuiState(model=model)
    url = f"http://{host}:{port}/"

    if backend in ("auto", "fastapi"):
        try:
            import uvicorn

            from femtools.gui.webapp import create_app
        except ImportError as exc:
            if backend == "fastapi":
                raise ImportError(
                    "the fastapi GUI backend needs 'fastapi' and 'uvicorn' "
                    "(pip install femtools[web])"
                ) from exc
        else:
            print(f"femtools GUI (fastapi) on {url}  — Ctrl-C to stop")
            _maybe_open_browser(url, open_browser)
            uvicorn.run(create_app(state), host=host, port=port, log_level="warning")
            return

    server = make_stdlib_server(host, port, state)
    print(f"femtools GUI (stdlib) on {url}  — Ctrl-C to stop")
    _maybe_open_browser(url, open_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
