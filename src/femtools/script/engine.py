"""FSL (Femtools Scripting Language) interpreter.

FSL is an original, line-oriented command language.  A script is a
sequence of statements separated by newlines and/or semicolons; ``#``
starts a comment.  Keywords are case-insensitive, options use
``KEY=value`` syntax.  See ``README.md`` in this package for the full
grammar.

The engine binds lazily to the rest of femtools: ``femtools.core`` is
only imported when ``NEW PROJECT`` runs, ``femtools.fea`` when
``SOLVE MODES`` runs, and so on.  Missing siblings produce a clear
:class:`ScriptError` instead of an import-time crash.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import Any

__all__ = ["ScriptEngine", "ScriptError"]


class ScriptError(Exception):
    """Raised on any FSL parse or execution failure.

    Carries the offending statement and its 1-based statement index when
    raised from :meth:`ScriptEngine.run`.
    """

    def __init__(self, message: str, statement: str | None = None, index: int | None = None):
        self.statement = statement
        self.index = index
        if statement is not None:
            location = f"statement {index}: " if index is not None else ""
            message = f"{message}\n  in {location}{statement!r}"
        super().__init__(message)


def _strip_comment(line: str) -> str:
    """Remove a ``#`` comment, honouring double-quoted strings."""
    out = []
    in_quote = False
    for ch in line:
        if ch == '"':
            in_quote = not in_quote
        elif ch == "#" and not in_quote:
            break
        out.append(ch)
    return "".join(out)


def _split_statements(text: str) -> list[str]:
    """Split script text into statements on newlines and semicolons."""
    statements: list[str] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        # split on ';' outside quotes
        buf = []
        in_quote = False
        for ch in line:
            if ch == '"':
                in_quote = not in_quote
            if ch == ";" and not in_quote:
                statements.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        statements.append("".join(buf))
    return [s.strip() for s in statements if s.strip()]


def _parse_scalar(token: str) -> Any:
    """Parse a token as int, then float, else return the string itself."""
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _parse_value(token: str) -> Any:
    """Parse an option value: comma lists become tuples of scalars."""
    if "," in token:
        return tuple(_parse_scalar(t) for t in token.split(",") if t != "")
    return _parse_scalar(token)


def _split_args_options(tokens: list[str]) -> tuple[list[Any], dict[str, Any]]:
    """Separate positional arguments from ``KEY=value`` options."""
    args: list[Any] = []
    opts: dict[str, Any] = {}
    for tok in tokens:
        eq = tok.find("=")
        if eq > 0 and tok[:eq].replace("_", "").isalpha():
            opts[tok[:eq].upper()] = _parse_value(tok[eq + 1 :])
        else:
            args.append(_parse_scalar(tok))
    return args, opts


def _parse_spc_mask(spec: Any) -> tuple[bool, ...]:
    """Parse an SPC constraint spec into a 6-tuple of booleans.

    Accepts ``ALL``, ``FREE``, a 6-character 0/1 string (also matched when
    the tokenizer already turned e.g. ``111111`` into an int), or a comma
    list / int sequence of 1-based DOF numbers.
    """
    if isinstance(spec, tuple):  # DOF list, e.g. DOF=1,2,3
        mask = [False] * 6
        for d in spec:
            if not isinstance(d, int) or not 1 <= d <= 6:
                raise ScriptError(f"SPC DOF numbers must be integers 1..6, got {d!r}")
            mask[d - 1] = True
        return tuple(mask)
    if isinstance(spec, int):
        spec = str(spec)
        if len(spec) < 6:
            # a single DOF number such as `SPC 1 DOF=3`
            return _parse_spc_mask((int(spec),))
    if isinstance(spec, str):
        word = spec.upper()
        if word == "ALL":
            return (True,) * 6
        if word == "FREE":
            return (False,) * 6
        if len(word) == 6 and set(word) <= {"0", "1"}:
            return tuple(c == "1" for c in word)
    raise ScriptError(
        f"cannot parse SPC mask {spec!r}: expected ALL, FREE, a 6-char 0/1 string, "
        "or DOF=<comma list of 1..6>"
    )


class ScriptEngine:
    """Interpreter for FSL scripts.

    Attributes
    ----------
    model:
        The active :class:`femtools.core.model.FEModel`, or ``None``
        before ``NEW PROJECT`` has run.
    results:
        Named analysis results (``SOLVE MODES``/``MAC`` outputs).
    log:
        Human-readable trace of executed statements.
    """

    def __init__(self, model_factory: Callable[..., Any] | None = None, echo: bool = False):
        self.model: Any = None
        self.results: dict[str, Any] = {}
        self.log: list[str] = []
        self.echo = echo
        self._model_factory = model_factory
        self._last_modes_name: str | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, text: str) -> ScriptEngine:
        """Execute a script (one or more statements). Returns ``self``."""
        for index, statement in enumerate(_split_statements(text), start=1):
            try:
                self.execute(statement)
            except ScriptError as exc:
                if exc.statement is None:
                    raise ScriptError(str(exc), statement=statement, index=index) from exc
                raise
        return self

    def run_file(self, path: str) -> ScriptEngine:
        """Execute an FSL script file. Returns ``self``."""
        with open(path, encoding="utf-8") as fh:
            return self.run(fh.read())

    def execute(self, statement: str) -> None:
        """Execute a single statement."""
        try:
            tokens = shlex.split(statement, posix=True)
        except ValueError as exc:
            raise ScriptError(f"tokenization failed: {exc}", statement=statement) from exc
        if not tokens:
            return
        keyword = tokens[0].upper()
        handler = self._DISPATCH.get(keyword)
        if handler is None:
            raise ScriptError(
                f"unknown command {keyword!r} (known: {', '.join(sorted(self._DISPATCH))})",
                statement=statement,
            )
        handler(self, tokens[1:], statement)
        self._trace(statement)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _trace(self, message: str) -> None:
        self.log.append(message)
        if self.echo:
            print(f"fsl> {message}")

    def _require_model(self, statement: str) -> Any:
        if self.model is None:
            raise ScriptError("no active project: run NEW PROJECT first", statement=statement)
        return self.model

    def _make_model(self, name: str) -> Any:
        if self._model_factory is not None:
            return self._model_factory(name=name)
        try:
            from femtools.core.model import FEModel
        except ImportError as exc:
            raise ScriptError(
                "femtools.core.model is not available; the core model database "
                "is required for NEW PROJECT"
            ) from exc
        return FEModel(name=name)

    # ------------------------------------------------------------------
    # command handlers
    # ------------------------------------------------------------------
    def _cmd_new(self, rest: list[str], statement: str) -> None:
        args, opts = _split_args_options(rest)
        if not args or str(args[0]).upper() != "PROJECT":
            raise ScriptError("expected NEW PROJECT [name]", statement=statement)
        name = str(args[1]) if len(args) > 1 else str(opts.pop("NAME", "untitled"))
        if opts:
            raise ScriptError(f"unexpected options {sorted(opts)}", statement=statement)
        self.model = self._make_model(name)
        self.results.clear()
        self._last_modes_name = None

    def _cmd_add(self, rest: list[str], statement: str) -> None:
        if not rest:
            raise ScriptError("expected ADD NODE|MAT|PROP|ELEM ...", statement=statement)
        kind = rest[0].upper()
        sub = {
            "NODE": self._add_node,
            "MAT": self._add_mat,
            "MATERIAL": self._add_mat,
            "PROP": self._add_prop,
            "PROPERTY": self._add_prop,
            "ELEM": self._add_elem,
            "ELEMENT": self._add_elem,
        }.get(kind)
        if sub is None:
            raise ScriptError(f"unknown ADD target {kind!r}", statement=statement)
        args, opts = _split_args_options(rest[1:])
        sub(args, opts, statement)

    def _add_node(self, args: list[Any], opts: dict[str, Any], statement: str) -> None:
        model = self._require_model(statement)
        if len(args) != 4:
            raise ScriptError("expected ADD NODE <id> <x> <y> <z>", statement=statement)
        nid, x, y, z = args
        if not isinstance(nid, int):
            raise ScriptError(f"node id must be an integer, got {nid!r}", statement=statement)
        try:
            xyz = (float(x), float(y), float(z))
        except (TypeError, ValueError) as exc:
            raise ScriptError(f"bad node coordinates {args[1:]!r}", statement=statement) from exc
        model.add_node(id=nid, xyz=xyz, **{k.lower(): v for k, v in opts.items()})

    def _add_mat(self, args: list[Any], opts: dict[str, Any], statement: str) -> None:
        model = self._require_model(statement)
        if len(args) != 1 or not isinstance(args[0], int):
            raise ScriptError("expected ADD MAT <id> [TYPE=..] [E=..] ...", statement=statement)
        kwargs = {k.lower(): v for k, v in opts.items()}
        kwargs.setdefault("type", "isotropic")
        # material symbols are conventionally uppercase in femtools.core
        for key in ("e", "g"):
            if key in kwargs:
                kwargs[key.upper()] = kwargs.pop(key)
        model.add_material(id=args[0], **kwargs)

    # FSL option keys are case-insensitive; the FE kernel uses these
    # canonical mixed-case section names.
    _PROP_KEY_ALIASES = {
        "IY": "Iy", "IZ": "Iz", "IYY": "Iyy", "IZZ": "Izz", "IXX": "Ixx",
        "IXY": "Ixy", "IXZ": "Ixz", "IYZ": "Iyz",
        "T": "t", "THICKNESS": "t", "H": "t",
        "K": "k", "STIFFNESS": "k", "M": "m", "MASS": "m", "C": "c",
        "NSM": "nsm", "ASY": "Asy", "ASZ": "Asz",
    }

    def _add_prop(self, args: list[Any], opts: dict[str, Any], statement: str) -> None:
        model = self._require_model(statement)
        if len(args) != 1 or not isinstance(args[0], int):
            raise ScriptError("expected ADD PROP <id> TYPE=.. MAT=.. ...", statement=statement)
        if "TYPE" not in opts:
            raise ScriptError("ADD PROP requires TYPE=<bar|beam|shell|...>", statement=statement)
        kwargs: dict[str, Any] = {"type": str(opts.pop("TYPE")).lower()}
        if "MAT" in opts:
            kwargs["material_id"] = opts.pop("MAT")
        elif "MATERIAL_ID" in opts:
            kwargs["material_id"] = opts.pop("MATERIAL_ID")
        # canonical aliases first; otherwise short symbolic names (A, I, J,
        # I1, EA, ...) stay uppercase and longer names go lowercase.
        for key, value in opts.items():
            canon = self._PROP_KEY_ALIASES.get(key, key if len(key) <= 2 else key.lower())
            kwargs[canon] = value
        model.add_property(id=args[0], **kwargs)

    def _add_elem(self, args: list[Any], opts: dict[str, Any], statement: str) -> None:
        model = self._require_model(statement)
        if len(args) != 1 or not isinstance(args[0], int):
            raise ScriptError(
                "expected ADD ELEM <id> TYPE=<type> NODES=<n1,n2,...> [PROP=<id>]",
                statement=statement,
            )
        etype = opts.pop("TYPE", None)
        nodes = opts.pop("NODES", None)
        if etype is None or nodes is None:
            raise ScriptError("ADD ELEM requires TYPE= and NODES=", statement=statement)
        if isinstance(nodes, int):
            nodes = (nodes,)
        if not isinstance(nodes, tuple) or not all(isinstance(n, int) for n in nodes):
            raise ScriptError(f"NODES must be a comma list of node ids, got {nodes!r}",
                              statement=statement)
        kwargs: dict[str, Any] = {}
        prop = opts.pop("PROP", None)
        if prop is not None:
            kwargs["property_id"] = prop
        kwargs.update({k.lower(): v for k, v in opts.items()})
        model.add_element(id=args[0], type=str(etype).upper(), nodes=nodes, **kwargs)

    def _cmd_spc(self, rest: list[str], statement: str) -> None:
        model = self._require_model(statement)
        args, opts = _split_args_options(rest)
        if not args or not isinstance(args[0], int):
            raise ScriptError("expected SPC <node_id> <mask>|ALL|DOF=1,2,...",
                              statement=statement)
        node_id = args[0]
        if "DOF" in opts:
            spec: Any = opts.pop("DOF")
            if isinstance(spec, int):
                spec = (spec,)
        elif len(args) >= 2:
            spec = args[1]
        else:
            spec = "ALL"
        if opts:
            raise ScriptError(f"unexpected options {sorted(opts)}", statement=statement)
        mask = _parse_spc_mask(spec)
        model.add_spc(node_id=node_id, mask=mask)

    def _cmd_solve(self, rest: list[str], statement: str) -> None:
        model = self._require_model(statement)
        args, opts = _split_args_options(rest)
        if not args or str(args[0]).upper() != "MODES":
            raise ScriptError("expected SOLVE MODES [N=..] [SHIFT=..] [NAME=..]",
                              statement=statement)
        n_modes = opts.pop("N", opts.pop("MODES", 10))
        shift = opts.pop("SHIFT", 0.0)
        name = str(opts.pop("NAME", "modes"))
        if opts:
            raise ScriptError(f"unexpected options {sorted(opts)}", statement=statement)
        try:
            from femtools.fea.eigen import solve_modes
        except ImportError as exc:
            raise ScriptError(
                "femtools.fea is not available; SOLVE MODES needs the FEA solver"
            ) from exc
        result = solve_modes(model, n_modes=int(n_modes), shift=float(shift))
        self.results[name] = result
        self._last_modes_name = name

    def _cmd_mac(self, rest: list[str], statement: str) -> None:
        args, opts = _split_args_options(rest)
        if args:
            raise ScriptError("MAC takes only options: [A=name] [B=name] [NAME=name]",
                              statement=statement)
        a_name = opts.pop("A", self._last_modes_name)
        b_name = opts.pop("B", a_name)
        name = str(opts.pop("NAME", "mac"))
        if opts:
            raise ScriptError(f"unexpected options {sorted(opts)}", statement=statement)
        if a_name is None or b_name is None:
            raise ScriptError("MAC needs modal results: run SOLVE MODES first or pass A=/B=",
                              statement=statement)
        a_name, b_name = str(a_name), str(b_name)
        for label in (a_name, b_name):
            if label not in self.results:
                raise ScriptError(f"no result named {label!r} "
                                  f"(available: {sorted(self.results)})", statement=statement)
        try:
            from femtools.correlation.mac import mac_matrix
        except ImportError as exc:
            raise ScriptError(
                "femtools.correlation is not available; MAC needs the correlation module"
            ) from exc
        phi_a = getattr(self.results[a_name], "modes", self.results[a_name])
        phi_b = getattr(self.results[b_name], "modes", self.results[b_name])
        self.results[name] = mac_matrix(phi_a, phi_b)

    def _cmd_save(self, rest: list[str], statement: str) -> None:
        model = self._require_model(statement)
        args, opts = _split_args_options(rest)
        if opts or len(args) != 1:
            raise ScriptError("expected SAVE <path>", statement=statement)
        path = str(args[0])
        try:
            from femtools.io.project import save_project
        except ImportError as exc:
            raise ScriptError(
                "femtools.io is not available; SAVE needs the project I/O module"
            ) from exc
        save_project(model, path)

    def _cmd_print(self, rest: list[str], statement: str) -> None:
        model = self.model
        args, _ = _split_args_options(rest)
        what = str(args[0]).upper() if args else "SUMMARY"
        if what == "SUMMARY":
            if model is None:
                print("no active project")
            else:
                print(
                    f"project {getattr(model, 'name', '?')!r}: "
                    f"{len(getattr(model, 'nodes', {}))} nodes, "
                    f"{len(getattr(model, 'elements', {}))} elements, "
                    f"{len(getattr(model, 'materials', {}))} materials, "
                    f"{len(getattr(model, 'properties', {}))} properties, "
                    f"{len(getattr(model, 'spcs', []))} SPCs"
                )
        elif what == "RESULTS":
            for key, value in self.results.items():
                print(f"{key}: {type(value).__name__}")
        else:
            raise ScriptError(f"PRINT expects SUMMARY or RESULTS, got {what!r}",
                              statement=statement)

    _DISPATCH: dict[str, Callable[[ScriptEngine, list[str], str], None]] = {
        "NEW": _cmd_new,
        "ADD": _cmd_add,
        "SPC": _cmd_spc,
        "SOLVE": _cmd_solve,
        "MAC": _cmd_mac,
        "SAVE": _cmd_save,
        "PRINT": _cmd_print,
    }
