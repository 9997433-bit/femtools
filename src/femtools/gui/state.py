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

    def _latest_static(self) -> tuple[Any, str | None]:
        """The most recent stored result carrying a displacement field.

        Call with the lock held.  Returns ``(None, None)`` when no
        static result is stored.
        """
        static, static_name = None, None
        for rname, result in self.engine.results.items():
            if (getattr(result, "u", None) is not None
                    and getattr(result, "modes", None) is None):
                static, static_name = result, rname
        return static, static_name

    @staticmethod
    def _recover_stress(model: Any, static: Any) -> Any:
        """``recover_stress`` with GUI-facing (HTTP 400) failure modes."""
        try:
            from femtools.fea.recover import recover_stress
        except ImportError as exc:
            raise GuiApiError(
                "stress recovery is unavailable: femtools module "
                f"{exc.name or exc} is not installed"
            ) from exc
        try:
            return recover_stress(model, static)
        except (ValueError, KeyError) as exc:
            raise GuiApiError(f"stress recovery failed: {exc}") from exc

    def stress_table(self, name: str | None = None, max_rows: Any = 20) -> dict:
        """Recover element stresses from a stored static result, as JSON.

        ``name`` selects the static result (default: the most recent
        stored result carrying a displacement field).  Raises
        :class:`GuiApiError` (HTTP 400) when no model is loaded, no
        static result exists, or the stress-recovery kernel is not
        installed.
        """
        try:
            max_rows = int(max_rows)
        except (TypeError, ValueError):
            raise GuiApiError(f"max_rows must be an integer, got {max_rows!r}") from None
        if max_rows < 1:
            raise GuiApiError("max_rows must be at least 1")

        with self._lock:
            model = self.model
            if model is None:
                raise GuiApiError("no model loaded")

            if name:
                static = self.engine.results.get(name)
                if static is None:
                    raise GuiApiError(f"no result named {name!r} "
                                      f"(available: {sorted(self.engine.results)})")
                if getattr(static, "u", None) is None:
                    raise GuiApiError(f"result {name!r} is not a static result "
                                      "(it carries no displacement field)")
                static_name = name
            else:
                static, static_name = self._latest_static()
            if static is None:
                raise GuiApiError(
                    "no static result: run SOLVE STATIC (script tab) first")

            stress = self._recover_stress(model, static)

            ids = list(getattr(stress, "element_ids", []))
            if not ids:
                raise GuiApiError("the static result covers no recoverable elements")
            etypes = list(getattr(stress, "etypes", []) or ["?"] * len(ids))
            vm = [float(v) for v in stress.von_mises]
            rows = []
            for i, eid in enumerate(ids[:max_rows]):
                rows.append({
                    "element": eid,
                    "type": etypes[i] if i < len(etypes) else "?",
                    "von_mises": vm[i],
                    "stress": [float(c) for c in stress.stress[i]],
                })
            return {
                "ok": True,
                "result": static_name,
                "components": list(getattr(stress, "components",
                                           ("xx", "yy", "zz", "xy", "yz", "zx"))),
                "n_elements": len(ids),
                "truncated": len(ids) > max_rows,
                "max_von_mises": max(vm),
                "skipped": {str(k): v for k, v in
                            getattr(stress, "skipped", {}).items()},
                "elements": rows,
            }

    # ------------------------------------------------------------------
    def render_png(self, kind: str, *, name: str = "mac", index: int = 0) -> bytes:
        """Render a plot ('mesh', 'mac', 'mode' or 'stress') to PNG bytes (Agg)."""
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
            elif kind == "stress":
                if self.model is None:
                    raise GuiApiError("no model loaded")
                static, static_name = self._latest_static()
                if static is None:
                    raise GuiApiError("no static result; run SOLVE STATIC first")
                stress = self._recover_stress(self.model, static)
                fig = viz.plot_stress(self.model, stress, backend="matplotlib",
                                      title=f"von Mises stress ({static_name})")
            else:
                raise GuiApiError(f"unknown plot kind {kind!r}")
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            finally:
                plt.close(fig)
            return buf.getvalue()
