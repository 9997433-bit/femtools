"""femtools.script — the FSL (Femtools Scripting Language) engine.

FSL is an original, line-oriented command language for driving femtools
models and analyses.  It is *not* a reimplementation of any proprietary
scripting product; the grammar is documented in ``README.md`` next to
this file.

Usage::

    from femtools.script.engine import ScriptEngine
    engine = ScriptEngine()
    engine.run("NEW PROJECT beam; ADD NODE 1 0 0 0")
"""

from __future__ import annotations

from femtools.script.engine import ScriptEngine, ScriptError

__all__ = ["ScriptEngine", "ScriptError"]
