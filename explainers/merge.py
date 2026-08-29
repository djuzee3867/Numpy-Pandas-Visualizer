"""merge / join: which rows find a partner and which do not."""

from __future__ import annotations

import ast

import pandas as pd

from .base import (MISSING, box, column_highlight, keyword_value, row_highlight,
                   safe_eval, statement_value, substep, too_big)


def _operands(node, env):
    """Handles both left.merge(right) and pd.merge(left, right)."""
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "merge":
        return None, None
    receiver = safe_eval(node.func.value, env)
    if isinstance(receiver, pd.DataFrame):
        right = safe_eval(node.args[0], env) if node.args else MISSING
        return receiver, right
    if len(node.args) >= 2:
        return safe_eval(node.args[0], env), safe_eval(node.args[1], env)
    return None, None


def _keys(node, env, left, right):
    """Join keys; with no `on`, fall back to shared column names like pandas."""
    on = keyword_value(node, "on", env)
    if isinstance(on, str):
        return [on]
    if isinstance(on, (list, tuple)):
        return list(on)
    return [c for c in left.columns if c in set(right.columns)]


def _matched(frame, keys, other):
    """Which rows of `frame` have a key that also appears in `other`."""
    try:
        mine = list(map(tuple, frame[keys].to_numpy()))
        theirs = set(map(tuple, other[keys].to_numpy()))
        return [i for i, value in enumerate(mine) if value in theirs]
    except Exception:
        return []


def try_explain(stmt, env):
    node = statement_value(stmt)
    if not isinstance(node, ast.Call):
        return None

    left, right = _operands(node, env)
    if not isinstance(left, pd.DataFrame) or not isinstance(right, pd.DataFrame):
        return None
    if too_big(left) or too_big(right):
        return None

    result = safe_eval(node, env)
    if not isinstance(result, pd.DataFrame) or too_big(result):
        return None

    keys = _keys(node, env, left, right)
    if not keys or any(k not in left.columns or k not in right.columns for k in keys):
        return None
    how = keyword_value(node, "how", env, "inner")

    lhit = _matched(left, keys, right)
    rhit = _matched(right, keys, left)
    lmiss = left.shape[0] - len(lhit)
    rmiss = right.shape[0] - len(rhit)

    return [
        substep(
            "merge",
            f"join key: {', '.join(map(str, keys))}  ·  how={how}",
            [box("left", left, column_highlight(left, keys)),
             box("right", right, column_highlight(right, keys))],
            note="rows are paired up by the highlighted column(s)",
        ),
        substep(
            "merge",
            f"{len(lhit)} of {left.shape[0]} left rows and "
            f"{len(rhit)} of {right.shape[0]} right rows find a partner",
            [box("left", left, row_highlight(left, lhit)),
             box("right", right, row_highlight(right, rhit))],
            note=_miss_note(how, lmiss, rmiss),
        ),
        substep("merge", f"result: {result.shape[0]} rows × {result.shape[1]} cols",
                [box("result", result)]),
    ]


def _miss_note(how, lmiss, rmiss):
    if how == "inner":
        return f"how=inner drops the {lmiss + rmiss} unmatched rows"
    if how == "left":
        return f"how=left keeps all left rows — {lmiss} of them get NaN on the right"
    if how == "right":
        return f"how=right keeps all right rows — {rmiss} of them get NaN on the left"
    return f"how={how} keeps every row — unmatched ones get NaN on the missing side"
