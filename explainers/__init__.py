"""Break one line of code into sub-steps.

`sys.settrace` only sees *between* lines, so `df.groupby("k").sum()` is a
single jump. These explainers read the AST of the line about to run and, when
it matches a registered shape, recompute the intermediate stages themselves
(read-only — see the rules in base.py).

Registration order runs specific to general, because shapes overlap: `.sum()`
could be a groupby or a numpy axis reduction. The first to accept it wins.
"""

from __future__ import annotations

import ast

from . import binop, groupby, merge, pivot, reduce, reshape, subscript
from .base import is_safe, statement_value

EXPLAINERS = (
    groupby.try_explain,     # .sum() on a GroupBy
    merge.try_explain,
    pivot.try_explain,
    reduce.try_explain,      # .sum() on an ndarray
    reshape.try_explain,
    subscript.try_explain,
    binop.try_explain,
)

MAX_SUBSTEPS_PER_LINE = 14


def statement_index(tree: ast.AST) -> dict:
    """line number -> statement, so the trace callback can look up an AST fast.

    Only statements explainers care about, and only the first one per line
    (`x = 1; y = 2` explains just the first).
    """
    index = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            index.setdefault(node.lineno, node)
    return index


def _replayable(stmt) -> bool:
    """Can an explainer safely evaluate this line's expressions a second time?

    Every explainer re-evaluates user expressions. Lines that call the user's
    own functions, or produce different values each time (np.random.rand), are
    skipped: otherwise side effects happen twice, or the picture drawn does not
    match what actually ran.
    """
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Subscript)):
        return is_safe(stmt.targets[0])
    return is_safe(statement_value(stmt))


def explain(stmt, env) -> list:
    """Sub-steps for this statement; empty when no explainer accepts it.

    A failing explainer is skipped silently — explaining is a bonus and must
    never take the whole trace down with it.
    """
    if not _replayable(stmt):
        return []

    for try_explain in EXPLAINERS:
        try:
            result = try_explain(stmt, env)
        except Exception:
            result = None
        if result:
            return result[:MAX_SUBSTEPS_PER_LINE]
    return []
