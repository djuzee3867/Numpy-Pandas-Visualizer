"""groupby, shown as split / apply / combine.

Rather than take apart the whole `df.groupby("k")["v"].sum()` chain, evaluate
the receiver of `.sum()` and check whether it is a GroupBy object. If it is,
ask it directly which rows belong to which group — shorter, and it survives
the many ways people write the same thing.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from .base import (MISSING, box, fmt_value, row_highlight, safe_eval,
                   statement_value, substep, too_big)

AGGS = {"sum", "mean", "min", "max", "count", "size", "median", "std", "var",
        "first", "last", "nunique", "prod"}
MAX_GROUPS = 6


def _is_groupby(obj) -> bool:
    return type(obj).__module__.startswith("pandas.core.groupby")


def _key_label(key) -> str:
    return " / ".join(map(str, key)) if isinstance(key, tuple) else str(key)


def _key_series(grouped, source):
    """A "key of each row" column, rebuilt from grouped.indices.

    Needed when one column was selected after grouping
    (`df.groupby("team")["score"]`): a SeriesGroupBy only keeps the selected
    Series, so the grouping column is gone and nothing on screen explains why
    those rows belong together.
    """
    labels = np.empty(len(source), dtype=object)
    for key, rows in grouped.indices.items():
        labels[np.asarray(rows, dtype=int)] = _key_label(key)
    name = getattr(grouped, "keys", None)
    if not isinstance(name, str):
        name = "group key"
    return pd.Series(labels, index=source.index, name=name)


def try_explain(stmt, env):
    node = statement_value(stmt)
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in AGGS):
        return None

    grouped = safe_eval(node.func.value, env)
    if grouped is MISSING or not _is_groupby(grouped):
        return None

    source = grouped.obj          # the DataFrame or Series being grouped
    if too_big(source):
        return None
    result = safe_eval(node, env)
    if result is MISSING:
        return None

    op = node.func.attr
    positions = grouped.indices   # {group name -> row positions}
    keys = list(positions)

    # When only a Series is left, the grouping column is off screen —
    # rebuild it, or nothing shows why these rows are together
    key_column = _key_series(grouped, source) if isinstance(source, pd.Series) else None

    def frame_boxes(rows=None):
        highlight = None if rows is None else row_highlight(source, rows)
        boxes = []
        if key_column is not None:
            key_high = None if rows is None else row_highlight(key_column, rows)
            boxes.append(box(str(key_column.name), key_column, key_high))
        boxes.append(box("source", source, highlight))
        return boxes

    steps = [substep(
        "groupby",
        f"split — the rows fall into {len(keys)} groups",
        frame_boxes(),
        note=f"pandas splits the rows, runs {op}() on each group, then combines the answers",
    )]

    for key in keys[:MAX_GROUPS]:
        rows = positions[key]
        piece = source.iloc[rows]
        value = safe_group_value(piece, op)
        boxes = frame_boxes(rows) + [box(f'group "{_key_label(key)}"', piece)]
        count = f"{len(rows)} row" + ("" if len(rows) == 1 else "s")
        if isinstance(value, (pd.Series, pd.DataFrame)):
            boxes.append(box(f"{op}() of the group", value))
            title = f'group "{key}" — {count}'
        else:
            title = f'group "{key}" — {count}, {op}() = {fmt_value(value)}'
        steps.append(substep("groupby", title, boxes))

    if len(keys) > MAX_GROUPS:
        steps.append(substep("groupby", f"… {len(keys) - MAX_GROUPS} more groups, same idea", []))

    steps.append(substep("groupby", "combine — one entry per group",
                         [box("result", result)]))
    return steps


def safe_group_value(piece, op):
    try:
        return getattr(piece, op)()
    except Exception:
        return MISSING
