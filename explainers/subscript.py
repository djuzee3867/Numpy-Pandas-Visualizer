"""numpy indexing / slicing / boolean masks.

Covers both `x = a[key]` and `a[key] = value`.
"""

from __future__ import annotations

import ast

import numpy as np

from .base import (IDX, MISSING, box, index_array, mask_from_picked, safe_eval,
                   shape_text, statement_value, substep, too_big, unparse)


def _selected_mask(subscript_node, src, env):
    """Index index_array with the original AST to learn which cells were picked."""
    probe = ast.Subscript(value=ast.Name(id=IDX, ctx=ast.Load()),
                          slice=subscript_node.slice, ctx=ast.Load())
    ast.copy_location(probe, subscript_node)
    picked = safe_eval(probe, {**env, IDX: index_array(src)})
    if picked is MISSING:
        return None
    return mask_from_picked(src, picked)


def _explain_assign(stmt, env):
    """a[a < 0] = 0 — show which cells are about to be overwritten."""
    target = stmt.targets[0]
    src = safe_eval(target.value, env)
    if not isinstance(src, np.ndarray) or src.ndim == 0 or too_big(src):
        return None
    mask = _selected_mask(target, src, env)
    if mask is None:
        return None

    name = unparse(target.value)
    return [substep(
        "subscript",
        f"{unparse(target)} picks {int(mask.sum())} cells — they are about to be overwritten",
        [box(name, src, mask)],
        note=f"the assignment happens in place, so {name} itself changes",
    )]


def try_explain(stmt, env):
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Subscript)):
        return _explain_assign(stmt, env)

    node = statement_value(stmt)
    if not isinstance(node, ast.Subscript):
        return None

    src = safe_eval(node.value, env)
    if not isinstance(src, np.ndarray) or src.ndim == 0 or too_big(src):
        return None
    mask = _selected_mask(node, src, env)
    if mask is None:
        return None

    # Safe now: the receiver is an ndarray, and numpy's __getitem__ is pure
    result = safe_eval(node, env)
    if result is MISSING:
        return None

    name = unparse(node.value)
    steps = [substep(
        "subscript",
        f"{unparse(node)} selects the highlighted cells",
        [box(name, src, mask)],
        note=f"{int(mask.sum())} of {src.size} cells",
    )]

    if isinstance(result, np.ndarray):
        shares = bool(np.shares_memory(result, src))
        note = ("the result is a view — it shares memory with the source, "
                "so writing to it writes through") if shares else \
               "the result is a copy — it owns its own buffer"
        steps.append(substep("subscript", f"result: shape {shape_text(result.shape)}",
                             [box("result", result)], note=note))
    else:
        steps.append(substep("subscript", "result: a single element",
                             [box("result", np.asarray(result))]))
    return steps
