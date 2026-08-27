"""Backend-independent GUI application state.

Both GUI backends (stdlib ``http.server`` and FastAPI) are thin HTTP
adapters over :class:`GuiState`, which owns a single
:class:`~femtools.script.engine.ScriptEngine` and renders plots to PNG
bytes with the Agg backend.  All mutating operations are serialized with
a lock so the threaded stdlib server is safe.
"""

from __future__ import annotations

import importlib.util
import io
import threading
from pathlib import Path
from typing import Any

import femtools
from femtools.script.engine import ScriptEngine, ScriptError

_SIBLINGS = ("core", "io", "fea", "dynamics", "correlation", "pretest",
             "updating", "optimization", "mpe", "rbpe", "script", "viz", "gui")


class GuiApiError(Exception):
    """A user-facing API failure (maps to HTTP 400)."""


class GuiState:
    def __init__(self, model: Any = None):
        self._lock = threading.RLock()
        self.engine = ScriptEngine()
        if model is not None:
            self.engine.model = model

    # ------------------------------------------------------------------
    @property
    def model(self) -> Any:
        return self.engine.model

    def status(self) -> dict:
        modules = {
            name: importlib.util.find_spec(f"femtools.{name}") is not None
            for name in _SIBLINGS
        }
        return {
            "version": femtools.__version__,
            "modules": modules,
            "has_model": self.model is not None,
        }

    def model_summary(self) -> dict:
        model = self.model
        if model is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "name": getattr(model, "name", None),
            "n_nodes": len(getattr(model, "nodes", {})),
            "n_elements": len(getattr(model, "elements", {})),
            "n_materials": len(getattr(model, "materials", {})),
            "n_properties": len(getattr(model, "properties", {})),
            "n_spcs": len(getattr(model, "spcs", [])),
        }

    def results_summary(self) -> dict:
        out = []
        for name, result in self.engine.results.items():
            entry: dict[str, Any] = {"name": name, "type": type(result).__name__}
            freqs = getattr(result, "freq_hz", None)
            if freqs is not None:
                entry["freq_hz"] = [float(f) for f in list(freqs)]
            shape = getattr(result, "shape", None)
            if shape is not None:
                entry["shape"] = list(shape)
            out.append(entry)
        return {"results": out}

    def load_model(self, path: str) -> dict:
        """Load a model file (.ftproj/.json/.unv/.bdf/.inp/.k) into the session.

        The file lives on the machine running the server (the GUI is a
        local tool), so a plain path is enough — no upload needed.
        Results stored in ``.ftproj`` / ``.unv`` files are imported into
        the session's result store.
        """
        text = (path or "").strip()
        if not text:
            raise GuiApiError("empty path: pass the path of a model file on this machine")
        fs_path = Path(text).expanduser()
        if not fs_path.is_file():
            raise GuiApiError(f"no such file: {fs_path}")
        try:
            from femtools.script.loading import load_model_file
        except ImportError as exc:  # pragma: no cover - loading.py ships with script
            raise GuiApiError(f"model loading is unavailable: {exc}") from exc
        try:
            loaded = load_model_file(fs_path)
        except ImportError as exc:
            raise GuiApiError(
                f"loading {fs_path.suffix!r} files needs femtools module "
                f"{exc.name or exc}, which is not installed"
            ) from exc
        except Exception as exc:  # unreadable/malformed file: user-facing 400
            raise GuiApiError(f"could not load {fs_path.name}: {exc}") from exc
        with self._lock:
            self.engine.model = loaded.model
            for name, result in loaded.results.items():
                self.engine.results[name] = result
            self.engine.log.append(f"(gui) load {fs_path}")
            return {
                "ok": True,
                "path": str(fs_path),
                "format": loaded.format,
                "model": self.model_summary(),
                "results": self.results_summary()["results"],
            }

    def run_script(self, source: str) -> dict:
        if not isinstance(source, str) or not source.strip():
            raise GuiApiError("empty script")
        with self._lock:
            n_before = len(self.engine.log)
            try:
                self.engine.run(source)
            except ScriptError as exc:
                return {
                    "ok": False,
                    "error": str(exc),
                    "executed": self.engine.log[n_before:],
                    "model": self.model_summary(),
                }
            return {
                "ok": True,
                "executed": self.engine.log[n_before:],
                "model": self.model_summary(),
                "results": self.results_summary()["results"],
            }

    # ------------------------------------------------------------------
    def render_png(self, kind: str, *, name: str = "mac", index: int = 0) -> bytes:
        """Render a plot ('mesh', 'mac' or 'mode') to PNG bytes (Agg)."""
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        from femtools import viz

        with self._lock:
            # pin matplotlib: PNG bytes are produced via fig.savefig below,
            # regardless of any process-wide viz.set_default_backend switch
            if kind == "mesh":
                if self.model is None:
                    raise GuiApiError("no model loaded")
                fig = viz.plot_mesh(self.model, backend="matplotlib")
            elif kind == "mac":
                result = self.engine.results.get(name)
                if result is None:
                    raise GuiApiError(
                        f"no result named {name!r}; run the MAC command first"
                    )
                fig = viz.plot_mac(result, title=f"MAC ({name})", backend="matplotlib")
            elif kind == "mode":
                if self.model is None:
                    raise GuiApiError("no model loaded")
                modal = None
                for result in self.engine.results.values():
                    if getattr(result, "modes", None) is not None:
                        modal = result
                if modal is None:
                    raise GuiApiError("no modal result; run SOLVE MODES first")
                fig = viz.plot_mode(self.model, modal, index=index,
                                    backend="matplotlib")
            else:
                raise GuiApiError(f"unknown plot kind {kind!r}")
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            finally:
                plt.close(fig)
            return buf.getvalue()
