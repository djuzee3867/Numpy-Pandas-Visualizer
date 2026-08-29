"""Shared helpers for every explainer.

The safety rule
---------------
Explainers run *before* the line actually executes, and must be **read-only**.
Touching user state would make side effects happen twice.

So each one checks the receiver's type first and only then evaluates the full
expression: subscript evaluates `a[key]` only once it knows `a` is an ndarray
(`ndarray.__getitem__` has no side effects). In-place lines such as `b += 10`
have no explainer at all, because they cannot be replayed.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

import snapshot

MISSING = object()
EXPR_FILE = "<explain>"
IDX = "__viz_idx__"      # temporary name for the flat-index probe array
MAX_CELLS = 400          # bigger than this and the picture is unreadable anyway


# Only calls that are deterministic and side-effect free.
# Explainers re-evaluate user expressions; allowing arbitrary calls would
# either run side effects twice (a function that appends to a list) or draw
# a picture from different values than the ones that actually ran.
SAFE_METHODS = {
    # numpy: construct / reshape / reduce
    "arange", "array", "asarray", "zeros", "ones", "full", "eye", "linspace",
    "reshape", "transpose", "ravel", "flatten", "squeeze", "swapaxes", "astype", "copy",
    "sum", "mean", "min", "max", "prod", "std", "var", "any", "all", "argmin", "argmax",
    # pandas
    "DataFrame", "Series", "groupby", "merge", "join", "pivot", "pivot_table", "melt",
    "count", "size", "median", "first", "last", "nunique",
    "head", "tail", "reset_index", "set_index", "sort_values", "sort_index",
    "abs", "round", "dropna", "fillna", "unique", "value_counts",
}


def is_safe(node) -> bool:
    """Is this expression safe to evaluate again (deterministic, no side effects)?"""
    if node is None:
        return True
    if isinstance(node, (ast.Name, ast.Constant)):
        return True
    if isinstance(node, ast.Attribute):
        return is_safe(node.value)
    if isinstance(node, ast.Subscript):
        return is_safe(node.value) and is_safe(node.slice)
    if isinstance(node, ast.Slice):
        return all(is_safe(part) for part in (node.lower, node.upper, node.step))
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(is_safe(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(is_safe(k) for k in node.keys) and all(is_safe(v) for v in node.values)
    if isinstance(node, ast.BinOp):
        return is_safe(node.left) and is_safe(node.right)
    if isinstance(node, ast.UnaryOp):
        return is_safe(node.operand)
    if isinstance(node, ast.Compare):
        return is_safe(node.left) and all(is_safe(c) for c in node.comparators)
    if isinstance(node, ast.Call):
        # Only whitelisted attribute calls; never a bare function call
        if not isinstance(node.func, ast.Attribute) or node.func.attr not in SAFE_METHODS:
            return False
        return (is_safe(node.func.value)
                and all(is_safe(arg) for arg in node.args)
                and all(is_safe(kw.value) for kw in node.keywords))
    return False


def statement_value(stmt):
    """The right-hand expression, or None if the statement is not explainable."""
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        return stmt.value
    if isinstance(stmt, ast.Expr):
        return stmt.value
    return None


def safe_eval(node, env):
    """Evaluate a sub-expression in the user's namespace; MISSING on failure."""
    if node is None:
        return MISSING
    try:
        expr = ast.Expression(body=node)
        ast.fix_missing_locations(expr)
        return eval(compile(expr, EXPR_FILE, "eval"), env)  # noqa: S307
    except Exception:
        return MISSING


def keyword_value(node, name, env, default=None):
    """Value of a keyword argument, or default when absent or unevaluable."""
    for kw in getattr(node, "keywords", []):
        if kw.arg == name:
            value = safe_eval(kw.value, env)
            return default if value is MISSING else value
    return default


def unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "…"


def too_big(value) -> bool:
    if isinstance(value, np.ndarray):
        return value.size > MAX_CELLS
    if isinstance(value, pd.DataFrame):
        return value.shape[0] * value.shape[1] > MAX_CELLS
    if isinstance(value, pd.Series):
        return value.shape[0] > MAX_CELLS
    return False


def fmt_value(value) -> str:
    """A single value, formatted for a sub-step title."""
    try:
        if isinstance(value, (np.generic, np.ndarray)):
            value = value.item()
        if isinstance(value, float):
            return f"{round(value, 4):g}"
        return str(value)
    except Exception:
        return "…"


def shape_text(shape) -> str:
    return " × ".join(str(int(n)) for n in shape) if len(shape) else "0-D"


# ------------------------------------------------------------------ payload

def box(label, value, highlight=None, note=None) -> dict:
    item = {"label": label, "snap": snapshot.snap(value, highlight)}
    if note:
        item["note"] = note
    return item


def substep(op, title, boxes, note=None) -> dict:
    payload = {"op": op, "title": title, "boxes": boxes}
    if note:
        payload["note"] = note
    return payload


# ------------------------------------------------------------------ mask

def index_array(arr: np.ndarray) -> np.ndarray:
    """An array whose every cell holds its own flat position (C order).

    The trick behind most explainers: push this through the same operation as
    the real data and see where the numbers land. That reveals which cells were
    selected, or where each element moved to.
    """
    return np.arange(arr.size).reshape(arr.shape)


def mask_from_picked(arr: np.ndarray, picked) -> np.ndarray:
    """Indexing result over index_array -> mask in the original shape."""
    mask = np.zeros(arr.size, dtype=bool)
    flat = np.asarray(picked).ravel()
    if flat.size:
        mask[flat.astype(int)] = True
    return mask.reshape(arr.shape)


def row_highlight(obj, positions) -> np.ndarray:
    """Highlight whole rows by position (2-D mask for a DataFrame, 1-D for a Series)."""
    positions = np.asarray(list(positions), dtype=int)
    if isinstance(obj, pd.DataFrame):
        mask = np.zeros(obj.shape, dtype=bool)
        if positions.size:
            mask[positions, :] = True
        return mask
    mask = np.zeros(obj.shape[0], dtype=bool)
    if positions.size:
        mask[positions] = True
    return mask


def column_highlight(frame: pd.DataFrame, names) -> np.ndarray:
    """Highlight whole columns by name."""
    wanted = set(names)
    mask = np.zeros(frame.shape, dtype=bool)
    for i, column in enumerate(frame.columns):
        if column in wanted:
            mask[:, i] = True
    return mask
