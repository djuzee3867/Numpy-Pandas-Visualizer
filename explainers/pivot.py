"""pivot / pivot_table / melt: swapping between wide and long tables."""

from __future__ import annotations

import ast

import pandas as pd

from .base import (MISSING, box, column_highlight, keyword_value, safe_eval,
                   statement_value, substep, too_big)

OPS = {"pivot": "wide", "pivot_table": "wide", "melt": "long"}
ROLE_KEYWORDS = ("index", "columns", "values", "id_vars", "value_vars")


def _named_columns(node, env, frame):
    """Column names named in the arguments, so we can highlight their role."""
    found = []
    for name in ROLE_KEYWORDS:
        value = keyword_value(node, name, env)
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, (list, tuple)):
            found.extend(v for v in value if isinstance(v, str))
    return [c for c in found if c in frame.columns]


def try_explain(stmt, env):
    node = statement_value(stmt)
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in OPS):
        return None

    source = safe_eval(node.func.value, env)
    if not isinstance(source, pd.DataFrame) or too_big(source):
        return None
    result = safe_eval(node, env)
    if not isinstance(result, pd.DataFrame) or too_big(result):
        return None

    op = node.func.attr
    used = _named_columns(node, env, source)
    highlight = column_highlight(source, used) if used else None
    direction = OPS[op]

    role_note = ", ".join(f"{name}={keyword_value(node, name, env)}"
                          for name in ROLE_KEYWORDS
                          if keyword_value(node, name, env) is not None)

    return [
        substep(
            op,
            f"{op}() reads the highlighted columns" if used else f"{op}() reads the whole frame",
            [box("source", source, highlight)],
            note=role_note or None,
        ),
        substep(
            op,
            f"reshaped into a {direction} table: "
            f"{result.shape[0]} rows × {result.shape[1]} cols",
            [box("result", result)],
            note=f"the values are the same, only their address in the table changed "
                 f"({source.shape[0]} × {source.shape[1]} before)",
        ),
    ]
