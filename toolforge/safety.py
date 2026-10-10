"""Static policy check for generated code — the first of two security layers.

Generated tools must be *pure computation*: no filesystem, no processes, no network,
no reflection tricks. The check runs on the AST before any code executes. It is
deliberately an allow-list (modules) plus deny-lists (builtins, dunder access), so
an unknown construct fails closed. The sandbox (``sandbox.py``) is the second layer
for anything the static check cannot see.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

ALLOWED_MODULES = frozenset({
    "math", "cmath", "statistics", "decimal", "fractions", "random", "numbers",
    "re", "string", "textwrap", "unicodedata", "difflib",
    "json", "csv", "base64", "binascii", "hashlib", "hmac", "zlib", "uuid",
    "datetime", "calendar", "time", "zoneinfo",
    "collections", "itertools", "functools", "operator", "heapq", "bisect",
    "dataclasses", "enum", "typing", "copy", "array", "ipaddress",
    "urllib.parse", "html", "io",
})

BANNED_NAMES = frozenset({
    "eval", "exec", "compile", "__import__", "open", "input", "breakpoint",
    "globals", "locals", "vars", "help", "exit", "quit", "memoryview",
    "setattr", "delattr", "getattr", "__builtins__", "__loader__", "__spec__",
})

# attributes that are harmless and common enough to allow
SAFE_DUNDERS = frozenset({"__name__", "__doc__", "__init__", "__eq__", "__lt__", "__repr__",
                          "__hash__", "__len__", "__iter__", "__next__", "__post_init__"})

# time.sleep could stall the executor; io is allowed only for in-memory buffers
BANNED_ATTRIBUTES = frozenset({"sleep", "open", "FileIO", "open_code", "mro"})


@dataclass(frozen=True)
class Violation:
    rule: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"line {self.line}: [{self.rule}] {self.detail}"


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[Violation] = []

    def _flag(self, node: ast.AST, rule: str, detail: str) -> None:
        self.violations.append(Violation(rule, getattr(node, "lineno", 0), detail))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(node, alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self._flag(node, "import", "relative imports are not allowed")
        else:
            self._check_module(node, node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    self._flag(node, "import", "star imports are not allowed")

    def _check_module(self, node: ast.AST, module: str) -> None:
        if module not in ALLOWED_MODULES and module.split(".")[0] not in ALLOWED_MODULES:
            self._flag(node, "import", f"module {module!r} is not on the allow-list")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            self._flag(node, "builtin", f"use of {node.id!r} is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr.startswith("__") and attr not in SAFE_DUNDERS:
            self._flag(node, "dunder", f"access to {attr!r} is not allowed")
        elif attr.startswith("_") and not attr.startswith("__"):
            self._flag(node, "private", f"access to private attribute {attr!r} is not allowed")
        elif attr in BANNED_ATTRIBUTES:
            self._flag(node, "attribute", f"use of {attr!r} is not allowed")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._flag(node, "async", "async functions are not allowed")

    def visit_Global(self, node: ast.Global) -> None:
        self._flag(node, "global", "global statements are not allowed")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and "__" in node.value and any(
            d in node.value for d in ("__class__", "__subclasses__", "__globals__", "__builtins__", "__import__")
        ):
            self._flag(node, "dunder", "string literal references a dunder escape path")


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare) or len(node.test.comparators) != 1:
        return False
    sides = [node.test.left, node.test.comparators[0]]
    return (isinstance(node.test.ops[0], ast.Eq)
            and any(isinstance(x, ast.Name) and x.id == "__name__" for x in sides)
            and any(isinstance(x, ast.Constant) and x.value == "__main__" for x in sides))


def strip_demo_code(code: str) -> tuple[str, int]:
    """Remove demo code a model appends after the tool: an ``if __name__ == "__main__":`` block and
    bare top-level expressions such as ``print(solve(...))``. Neither is part of a tool, and neither
    would run as one; removing code can only shrink what runs. Returns (code, lines removed). Any
    other top-level statement is left for ``check_code`` to reject."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, 0
    drop: set[int] = set()
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if _is_main_guard(node) or (isinstance(node, ast.Expr) and not is_docstring):
            drop.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    if not drop:
        return code, 0
    lines = code.splitlines(keepends=True)
    kept = "".join(line for i, line in enumerate(lines, start=1) if i not in drop)
    try:
        ast.parse(kept)
    except SyntaxError:  # e.g. two statements shared a line; leave it to the policy check
        return code, 0
    return kept, len(drop)


def check_code(code: str, func_name: str) -> list[Violation]:
    """Return every policy violation; an empty list means the code may run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Violation("syntax", e.lineno or 0, e.msg)]

    checker = _Checker()
    checker.visit(tree)
    violations = checker.violations

    top_level_funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if func_name not in top_level_funcs:
        violations.append(Violation("contract", 0, f"no top-level function named {func_name!r}"))

    allowed_top = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not isinstance(node, allowed_top) and not is_docstring:
            violations.append(Violation("contract", node.lineno,
                                        f"top-level {type(node).__name__} statements are not allowed"))
    return violations
