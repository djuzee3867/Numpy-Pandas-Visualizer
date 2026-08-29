"""Broadcasting: a binary operator whose two sides have different shapes."""

from __future__ import annotations

import ast

import numpy as np

from .base import (MISSING, box, safe_eval, shape_text, statement_value,
                   substep, too_big, unparse)


def _aligned(shape, ndim):
    """Pad the shape with leading 1s — broadcasting rule number one."""
    return (1,) * (ndim - len(shape)) + tuple(shape)


def try_explain(stmt, env):
    node = statement_value(stmt)
    if not isinstance(node, ast.BinOp):
        return None

    left = safe_eval(node.left, env)
    right = safe_eval(node.right, env)
    if left is MISSING or right is MISSING:
        return None
    if not isinstance(left, np.ndarray) and not isinstance(right, np.ndarray):
        return None
    if too_big(left) or too_big(right):
        return None

    lshape, rshape = np.shape(left), np.shape(right)
    if lshape == rshape:
        return None      # nothing to explain, shapes already match

    result = safe_eval(node, env)
    if not isinstance(result, np.ndarray) or too_big(result):
        return None

    ndim = result.ndim
    lpad, rpad = _aligned(lshape, ndim), _aligned(rshape, ndim)
    lname, rname = unparse(node.left), unparse(node.right)

    steps = [substep(
        "broadcast",
        f"shapes line up from the right: ({shape_text(lpad)}) and ({shape_text(rpad)})",
        [box(lname, np.asarray(left)), box(rname, np.asarray(right))],
        note=f"missing axes are padded with 1, then every 1 is stretched to match — "
             f"result will be {shape_text(result.shape)}",
    )]

    try:
        lb = np.broadcast_to(left, result.shape)
        rb = np.broadcast_to(right, result.shape)
    except Exception:
        return None

    steps.append(substep(
        "broadcast",
        f"both operands stretched to {shape_text(result.shape)}",
        [box(f"{lname} stretched", lb), box(f"{rname} stretched", rb)],
        note="no data is copied — numpy reuses the same values by walking with stride 0",
    ))
    steps.append(substep("broadcast", "then the operator runs cell by cell",
                         [box("result", result)]))
    return steps
