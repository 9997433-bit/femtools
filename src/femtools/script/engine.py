"""FSL (Femtools Scripting Language) interpreter.

FSL is an original, line-oriented command language.  A script is a
sequence of statements separated by newlines and/or semicolons; ``#``
starts a comment.  Keywords are case-insensitive, options use
``KEY=value`` syntax.  See ``README.md`` in this package for the full
grammar.

The engine binds lazily to the rest of femtools: ``femtools.core`` is
only imported when ``NEW PROJECT`` runs, ``femtools.fea`` when
``SOLVE MODES`` / ``SOLVE STATIC`` / ``RECOVER STRESS`` runs, and so
on.  Missing siblings produce a clear :class:`ScriptError` instead of
an import-time crash.

``SET name=value`` assigns a script variable; later statements can
reference it as ``$name`` anywhere a token is expected (``$$`` writes a
literal dollar sign).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from typing import Any

__all__ = ["ScriptEngine", "ScriptError"]

# `$name` variable references (set with SET); `$$` is a literal dollar sign
_VAR_PATTERN = re.compile(r"\$(\$|[A-Za-z_]\w*)")
_SET_NAME_PATTERN = re.compile(r"^[A-Za-z_]\w*$")


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


def _load_mapping(model: Any) -> dict[tuple[int, int], float] | None:
    """Flatten ``model.loads`` records into ``{(node_id, dof): value}``.

    ``solve_static`` reads that mapping directly, whereas the shape of a
    stored :class:`~femtools.core.model.Load` record (``force``/``moment``
    vectors) is not among the record layouts its load builder probes for.
    Returns ``None`` when the model carries no loads (or non-``Load``
    records the kernel should interpret itself).
    """
    records = getattr(model, "loads", None)
    if not records:
        return None
    mapping: dict[tuple[int, int], float] = {}
    try:
        for record in records:
            for dof, value in record.as_dof_values():
                key = (record.node_id, dof)
                mapping[key] = mapping.get(key, 0.0) + value
    except (AttributeError, TypeError):
        return None
    return mapping or None


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


def _parse_node_list(spec: Any, what: str) -> tuple[int, ...]:
    """Parse an RBE node list: one node id or a comma list of ids."""
    if isinstance(spec, int):
        return (spec,)
    if isinstance(spec, tuple) and spec and all(isinstance(n, int) for n in spec):
        return tuple(spec)
    raise ScriptError(f"{what} must be a node id or a comma list of node ids, got {spec!r}")


def _parse_rbe_components(spec: Any, what: str) -> tuple[int, ...]:
    """Parse RBE DOF components: a comma list (1,2,3) or compact digits (123)."""
    if isinstance(spec, tuple):
        comps = spec
    elif isinstance(spec, int) and spec > 0:
        comps = tuple(int(c) for c in str(spec))
    else:
        comps = None
    if not comps or any(not isinstance(c, int) or not 1 <= c <= 6 for c in comps):
        raise ScriptError(
            f"{what} components must be DOF numbers 1..6 "
            f"(comma list like 1,2,3 or compact digits like 123), got {spec!r}"
        )
    return tuple(int(c) for c in comps)


def _parse_weights(spec: Any) -> tuple[float, ...]:
    """Parse RBE3 weights: one number or a comma list of numbers."""
    values = spec if isinstance(spec, tuple) else (spec,)
    try:
        return tuple(float(v) for v in values)
    except (TypeError, ValueError):
        raise ScriptError(f"WEIGHTS must be a comma list of numbers, got {spec!r}") from None


class ScriptEngine:
    """Interpreter for FSL scripts.

    Attributes
    ----------
    model:
        The active :class:`femtools.core.model.FEModel`, or ``None``
        before ``NEW PROJECT`` has run.
    results:
        Named analysis results (``SOLVE MODES``/``SOLVE STATIC``/``MAC``
        outputs).
    variables:
        Script variables assigned with ``SET name=value`` (values are
        kept as the raw text after ``=`` and substituted wherever a
        later statement writes ``$name``).
    log:
        Human-readable trace of executed statements.
    """

    def __init__(self, model_factory: Callable[..., Any] | None = None, echo: bool = False):
        self.model: Any = None
        self.results: dict[str, Any] = {}
        self.variables: dict[str, str] = {}
        self.log: list[str] = []
        self.echo = echo
        self._model_factory = model_factory
        self._last_modes_name: str | None = None
        self._last_static_name: str | None = None

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
        tokens = [self._expand_variables(tok, statement) for tok in tokens]
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

    def _expand_variables(self, token: str, statement: str) -> str:
        """Replace ``$name`` references with SET values (``$$`` -> ``$``)."""
        if "$" not in token:
            return token

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name == "$":
                return "$"
            value = self.variables.get(name.upper())
            if value is None:
                raise ScriptError(
                    f"undefined variable ${name} (assign it first: SET {name.upper()}=value)",
                    statement=statement,
                )
            return value

        return _VAR_PATTERN.sub(replace, token)

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
        self._last_static_name = None

    def _cmd_add(self, rest: list[str], statement: str) -> None:
        if not rest:
            raise ScriptError("expected ADD NODE|MAT|PROP|ELEM|LOAD|RBE2|RBE3 ...",
                              statement=statement)
        kind = rest[0].upper()
        sub = {
            "NODE": self._add_node,
            "MAT": self._add_mat,
            "MATERIAL": self._add_mat,
            "PROP": self._add_prop,
            "PROPERTY": self._add_prop,
            "ELEM": self._add_elem,
            "ELEMENT": self._add_elem,
            "LOAD": self._add_load,
            "RBE2": self._add_rbe2,
            "RBE3": self._add_rbe3,
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

    _LOAD_COMPONENTS = {"FX": 0, "FY": 1, "FZ": 2, "MX": 0, "MY": 1, "MZ": 2}

    def _add_load(self, args: list[Any], opts: dict[str, Any], statement: str) -> None:
        model = self._require_model(statement)
        if len(args) != 1 or not isinstance(args[0], int):
            raise ScriptError(
                "expected ADD LOAD <node_id> [FX=..] [FY=..] [FZ=..] [MX=..] [MY=..] [MZ=..]",
                statement=statement,
            )
        force = [0.0, 0.0, 0.0]
        moment = [0.0, 0.0, 0.0]
        has_force = has_moment = False
        for key, value in opts.items():
            index = self._LOAD_COMPONENTS.get(key)
            if index is None:
                raise ScriptError(
                    f"unknown load component {key!r} (use FX FY FZ MX MY MZ)",
                    statement=statement,
                )
            try:
                magnitude = float(value)
            except (TypeError, ValueError) as exc:
                raise ScriptError(f"bad load value {key}={value!r}",
                                  statement=statement) from exc
            if key.startswith("F"):
                force[index] = magnitude
                has_force = True
            else:
                moment[index] = magnitude
                has_moment = True
        if not (has_force or has_moment):
            raise ScriptError("ADD LOAD needs at least one component (FX..MZ)",
                              statement=statement)
        model.add_load(node_id=args[0],
                       force=force if has_force else None,
                       moment=moment if has_moment else None)

    def _add_rbe2(self, args: list[Any], opts: dict[str, Any], statement: str) -> None:
        model = self._require_model(statement)
        if len(args) != 1 or not isinstance(args[0], int):
            raise ScriptError(
                "expected ADD RBE2 <id> INDEP=<node> DEP=<n1,n2,...> [DOF=<comps>]",
                statement=statement,
            )
        indep = opts.pop("INDEP", opts.pop("INDEPENDENT", None))
        dep = opts.pop("DEP", opts.pop("DEPS", opts.pop("DEPENDENTS", None)))
        if indep is None or dep is None:
            raise ScriptError("ADD RBE2 requires INDEP=<node> and DEP=<node list>",
                              statement=statement)
        if not isinstance(indep, int):
            raise ScriptError(f"INDEP must be a single node id, got {indep!r}",
                              statement=statement)
        kwargs: dict[str, Any] = {}
        dof = opts.pop("DOF", opts.pop("COMPONENTS", None))
        if dof is not None:
            kwargs["components"] = _parse_rbe_components(dof, "RBE2")
        if opts:
            raise ScriptError(f"unexpected options {sorted(opts)}", statement=statement)
        model.add_rbe2(id=args[0], independent=indep,
                       dependents=_parse_node_list(dep, "DEP"), **kwargs)

    def _add_rbe3(self, args: list[Any], opts: dict[str, Any], statement: str) -> None:
        model = self._require_model(statement)
        if len(args) != 1 or not isinstance(args[0], int):
            raise ScriptError(
                "expected ADD RBE3 <id> DEP=<node> INDEP=<n1,n2,...> "
                "[DOF=<comps>] [IDOF=<comps>] [WEIGHTS=<w1,w2,...>]",
                statement=statement,
            )
        dep = opts.pop("DEP", opts.pop("DEPENDENT", None))
        indep = opts.pop("INDEP", opts.pop("INDEPENDENTS", None))
        if dep is None or indep is None:
            raise ScriptError("ADD RBE3 requires DEP=<node> and INDEP=<node list>",
                              statement=statement)
        if not isinstance(dep, int):
            raise ScriptError(f"DEP must be a single node id, got {dep!r}",
                              statement=statement)
        kwargs: dict[str, Any] = {}
        dof = opts.pop("DOF", opts.pop("COMPONENTS", None))
        if dof is not None:
            kwargs["components"] = _parse_rbe_components(dof, "RBE3")
        idof = opts.pop("IDOF", opts.pop("INDEPENDENT_COMPONENTS", None))
        if idof is not None:
            kwargs["independent_components"] = _parse_rbe_components(idof, "RBE3")
        weights = opts.pop("WEIGHTS", opts.pop("WEIGHT", None))
        if weights is not None:
            kwargs["weights"] = _parse_weights(weights)
        if opts:
            raise ScriptError(f"unexpected options {sorted(opts)}", statement=statement)
        model.add_rbe3(id=args[0], dependent=dep,
                       independents=_parse_node_list(indep, "INDEP"), **kwargs)

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
        kind = str(args[0]).upper() if args else ""
        if kind == "MODES":
            self._solve_modes(model, args, opts, statement)
        elif kind == "STATIC":
            self._solve_static(model, args, opts, statement)
        else:
            raise ScriptError(
                "expected SOLVE MODES [N=..] [SHIFT=..] [NAME=..] or SOLVE STATIC [NAME=..]",
                statement=statement,
            )

    def _solve_modes(self, model: Any, args: list[Any], opts: dict[str, Any],
                     statement: str) -> None:
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

    def _solve_static(self, model: Any, args: list[Any], opts: dict[str, Any],
                      statement: str) -> None:
        if len(args) > 1:
            raise ScriptError("expected SOLVE STATIC [NAME=..]", statement=statement)
        name = str(opts.pop("NAME", "static"))
        if opts:
            raise ScriptError(f"unexpected options {sorted(opts)}", statement=statement)
        try:
            from femtools.fea.static import solve_static
        except ImportError as exc:
            raise ScriptError(
                "femtools.fea is not available; SOLVE STATIC needs the FEA solver"
            ) from exc
        try:
            result = solve_static(model, _load_mapping(model), full_result=True)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            # e.g. a singular stiffness from an under-constrained model
            raise ScriptError(f"SOLVE STATIC failed: {exc}", statement=statement) from exc
        self.results[name] = result
        self._last_static_name = name

    def _cmd_recover(self, rest: list[str], statement: str) -> None:
        args, opts = _split_args_options(rest)
        if len(args) != 1 or str(args[0]).upper() != "STRESS":
            raise ScriptError("expected RECOVER STRESS [NAME=..] [RESULT=..]",
                              statement=statement)
        model = self._require_model(statement)
        name = str(opts.pop("NAME", "stress"))
        result_name = opts.pop("RESULT", self._last_static_name)
        if opts:
            raise ScriptError(f"unexpected options {sorted(opts)}", statement=statement)
        if result_name is None:
            raise ScriptError(
                "no static result: run SOLVE STATIC first (or pass RESULT=name)",
                statement=statement,
            )
        result_name = str(result_name)
        static = self.results.get(result_name)
        if static is None:
            raise ScriptError(f"no result named {result_name!r} "
                              f"(available: {sorted(self.results)})", statement=statement)
        if getattr(static, "u", None) is None:
            raise ScriptError(
                f"result {result_name!r} is not a static result "
                "(it carries no displacement field)",
                statement=statement,
            )
        try:
            from femtools.fea.recover import recover_stress
        except ImportError as exc:
            raise ScriptError(
                "femtools.fea is not available; RECOVER STRESS needs the "
                "stress-recovery kernel"
            ) from exc
        try:
            self.results[name] = recover_stress(model, static)
        except (ValueError, KeyError) as exc:
            # e.g. a displacement field of the wrong length, or an element
            # type without a registered recovery rule
            raise ScriptError(f"RECOVER STRESS failed: {exc}", statement=statement) from exc

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

    def _cmd_set(self, rest: list[str], statement: str) -> None:
        if not rest:
            raise ScriptError("expected SET NAME=value [NAME=value ...]",
                              statement=statement)
        for token in rest:
            name, sep, value = token.partition("=")
            if not sep or not _SET_NAME_PATTERN.match(name):
                raise ScriptError(
                    f"expected NAME=value assignments, got {token!r} "
                    "(names start with a letter or underscore)",
                    statement=statement,
                )
            self.variables[name.upper()] = value

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
        "SET": _cmd_set,
        "SOLVE": _cmd_solve,
        "RECOVER": _cmd_recover,
        "MAC": _cmd_mac,
        "SAVE": _cmd_save,
        "PRINT": _cmd_print,
    }
