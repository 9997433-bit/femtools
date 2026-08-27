"""Smoke tests for the semicolon-delimited scripting language."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("femtools.core.model")
script_module = pytest.importorskip("femtools.script.engine")


def _model_from(engine: Any, result: Any) -> Any:
    candidates = (result, getattr(result, "model", None), getattr(engine, "model", None))
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "nodes"):
            return candidate
    project = getattr(engine, "project", None)
    if project is not None and hasattr(project, "model"):
        return project.model
    raise AssertionError("ScriptEngine must retain or return the active FEModel")


def test_script_engine_creates_project_and_node() -> None:
    engine = script_module.ScriptEngine()
    result = engine.run("NEW PROJECT; ADD NODE 1 0 0 0")
    model = _model_from(engine, result)

    assert 1 in model.nodes
    node = model.nodes[1]
    assert tuple(node.xyz) == pytest.approx((0.0, 0.0, 0.0))
