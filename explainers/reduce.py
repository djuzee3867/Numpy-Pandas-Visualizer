"""Axis reductions: which group of cells folds into which output value."""

from __future__ import annotations

import ast

import numpy as np

from .base import (MISSING, box, fmt_value, safe_eval, shape_text,
                   statement_value, substep, too_big)

AGGS = {"sum", "mean", "min", "max", "prod", "std", "var", "any", "all", "argmin", "argmax"}
MAX_GROUPS = 6


def _axis_of(node, env):
    for kw in node.keywords:
        if kw.arg == "axis":
            value = safe_eval(kw.value, env)
            return None if value is MISSING else value
    if node.args:
        value = safe_eval(node.args[0], env)
        return None if value is MISSING else value
    return None


def try_explain(stmt, env):
    node = statement_value(stmt)
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in AGGS):
        return None

    src = safe_eval(node.func.value, env)
    if not isinstance(src, np.ndarray) or src.ndim == 0 or too_big(src):
        return None

    axis = _axis_of(node, env)
    if axis is not None and not isinstance(axis, (int, np.integer)):
        return None                      # tuple axis, not drawable yet
    if isinstance(axis, (int, np.integer)):
        axis = int(axis) % src.ndim

    result = safe_eval(node, env)
    if result is MISSING:
        return None
    op = node.func.attr

    if axis is None:
        return [
            substep("reduce", f"{op}() with no axis folds every cell into one value",
                    [box("source", src, np.ones(src.shape, dtype=bool))]),
            substep("reduce", f"result: {fmt_value(result)}", [box("result", np.asarray(result))]),
        ]

    out = np.asarray(result)
    out_shape = tuple(n for i, n in enumerate(src.shape) if i != axis)
    total = int(np.prod(out_shape)) if out_shape else 1

    steps = [substep(
        "reduce",
        f"axis={axis} disappears — cells along that axis fold together",
        [box("source", src)],
        note=f"{shape_text(src.shape)} becomes {shape_text(out_shape)}, so there are {total} groups",
    )]

    for k in range(min(total, MAX_GROUPS)):
        coords = list(np.unravel_index(k, out_shape)) if out_shape else []
        key = coords[:axis] + [slice(None)] + coords[axis:]
        mask = np.zeros(src.shape, dtype=bool)
        mask[tuple(key)] = True
        value = out[tuple(coords)] if coords else out
        steps.append(substep(
            "reduce",
            f"group {k}: {op} of the highlighted cells = {fmt_value(value)}",
            [box("source", src, mask)],
        ))

    if total > MAX_GROUPS:
        steps.append(substep("reduce", f"… {total - MAX_GROUPS} more groups, same idea", []))

    steps.append(substep("reduce", f"result: shape {shape_text(out.shape)}", [box("result", out)]))
    return steps
