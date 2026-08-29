"""reshape / transpose / ravel: tracking where each element ends up."""

from __future__ import annotations

import ast

import numpy as np

from .base import (IDX, MISSING, box, index_array, safe_eval, shape_text,
                   statement_value, substep, too_big, unparse)

RESHAPERS = {"reshape", "transpose", "ravel", "flatten", "swapaxes", "squeeze"}


def _receiver_and_rebuilder(node):
    """Returns (receiver expression, rebuilder that swaps the receiver)."""
    if isinstance(node, ast.Attribute) and node.attr == "T":
        def rebuild(name):
            return ast.Attribute(value=name, attr="T", ctx=ast.Load())
        return node.value, rebuild, "transpose"

    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in RESHAPERS):
        attr = node.func.attr

        def rebuild(name):
            return ast.Call(func=ast.Attribute(value=name, attr=attr, ctx=ast.Load()),
                            args=node.args, keywords=node.keywords)
        return node.func.value, rebuild, attr

    return None, None, None


def try_explain(stmt, env):
    node = statement_value(stmt)
    receiver, rebuild, op = _receiver_and_rebuilder(node)
    if receiver is None:
        return None

    src = safe_eval(receiver, env)
    if not isinstance(src, np.ndarray) or src.ndim == 0 or too_big(src):
        return None
    result = safe_eval(node, env)
    if not isinstance(result, np.ndarray):
        return None

    # Push index_array through the same operation to learn each cell's origin
    probe = rebuild(ast.Name(id=IDX, ctx=ast.Load()))
    ast.copy_location(probe, node)
    moved = safe_eval(probe, {**env, IDX: index_array(src)})
    if moved is MISSING or not isinstance(moved, np.ndarray):
        return None

    name = unparse(receiver)
    return [
        substep(
            "reshape",
            "numpy numbers every element in C order — last axis moves fastest",
            [box(name, src), box("flat order", index_array(src))],
            note="these numbers are positions in memory, not values",
        ),
        substep(
            "reshape",
            f"{op} lays the same elements out as {shape_text(result.shape)}",
            [box("came from", moved), box("result", result)],
            note="same buffer, different way of walking it — "
                 f"{'this is a view' if np.shares_memory(result, src) else 'this one had to copy'}",
        ),
    ]
