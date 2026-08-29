"""Turn a single Python value into a JSON-safe dict the frontend can draw.

Every dict carries a ``kind`` key; the JS side picks a renderer from it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MAX_ROWS = 20          # rows kept (second-to-last axis)
MAX_COLS = 20          # columns kept (last axis)
MAX_LEADING = 4        # slices kept per leading axis, for 3-D and up
MAX_ITEMS = 50         # list / tuple / dict
MAX_REPR = 200         # repr length for values we do not understand


# ---------------------------------------------------------------- single values

def _short_repr(value) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        return f"<repr failed: {type(exc).__name__}>"
    if len(text) > MAX_REPR:
        text = text[:MAX_REPR] + "…"
    return text


def _cell(value):
    """One cell -> a JSON-safe value.

    Every flavour of NA (None, NaN, NaT, pd.NA) collapses to ``None``, so the
    frontend cannot tell them apart. Accepted limitation.
    """
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return _short_repr(value)


def _walk(obj):
    """Make a whole ndarray.tolist() result JSON-safe."""
    if isinstance(obj, list):
        return [_walk(x) for x in obj]
    return _cell(obj)


def _walk_bool(obj):
    """Like _walk but forces bool — used for explainer highlight masks."""
    if isinstance(obj, list):
        return [_walk_bool(x) for x in obj]
    return bool(obj)


def _label(value) -> str:
    """Row/column label; a MultiIndex entry becomes "a / b"."""
    if isinstance(value, tuple):
        return " / ".join(_label(v) for v in value)
    cell = _cell(value)
    return "NaN" if cell is None else str(cell)


# ---------------------------------------------------------------- numpy

def _pointer(arr: np.ndarray):
    """Buffer address, used to spot views. None when it cannot be read."""
    try:
        return int(arr.__array_interface__["data"][0])
    except Exception:
        return None


def _snap_ndarray(arr: np.ndarray, highlight=None) -> dict:
    """highlight: bool array shaped like arr; True cells are emphasised."""
    info = {
        "kind": "ndarray",
        "dtype": str(arr.dtype),
        "shape": [int(n) for n in arr.shape],
        "ndim": int(arr.ndim),
        "size": int(arr.size),
        # is_view just means "has a base" — that base is often an unnamed
        # temporary (reshape returns one). The badge in the UI uses
        # shares_memory_with instead, which only looks at named variables.
        "is_view": arr.base is not None,
        "owns_data": bool(arr.flags.owndata),
        "ptr": _pointer(arr),
    }
    if arr.ndim == 0:
        info.update(data=_cell(arr[()]), shown=[], truncated=False)
        return info

    limits = []
    for axis in range(arr.ndim):
        if axis == arr.ndim - 1:
            limits.append(MAX_COLS)
        elif axis == arr.ndim - 2:
            limits.append(MAX_ROWS)
        else:
            limits.append(MAX_LEADING)

    cut = tuple(slice(0, min(n, lim)) for n, lim in zip(arr.shape, limits))
    window = arr[cut]
    info["shown"] = [int(n) for n in window.shape]
    info["truncated"] = info["shown"] != info["shape"]
    info["data"] = _walk(window.tolist())
    if highlight is not None:
        info["highlight"] = _walk_bool(np.asarray(highlight)[cut].tolist())
    return info


# ---------------------------------------------------------------- pandas

def _snap_dataframe(df: pd.DataFrame, highlight=None) -> dict:
    """highlight: bool array shaped (rows, cols)."""
    window = df.iloc[:MAX_ROWS, :MAX_COLS]
    rows, cols = window.shape
    extra = {}
    if highlight is not None:
        extra["highlight"] = _walk_bool(np.asarray(highlight)[:MAX_ROWS, :MAX_COLS].tolist())
    return {
        **extra,
        "kind": "DataFrame",
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "shown": [int(rows), int(cols)],
        "truncated": window.shape != df.shape,
        "columns": [_label(c) for c in window.columns],
        "index": [_label(i) for i in window.index],
        "index_name": None if df.index.name is None else _label(df.index.name),
        "dtypes": [str(t) for t in window.dtypes],
        "data": [[_cell(window.iat[r, c]) for c in range(cols)] for r in range(rows)],
    }


def _snap_series(s: pd.Series, highlight=None) -> dict:
    """highlight: 1-D bool array as long as the Series."""
    window = s.iloc[:MAX_ROWS]
    extra = {}
    if highlight is not None:
        # A Series grid is one column wide, so the mask has to be nested
        flags = _walk_bool(np.asarray(highlight)[:MAX_ROWS].tolist())
        extra["highlight"] = [[flag] for flag in flags]
    return {
        **extra,
        "kind": "Series",
        "dtype": str(s.dtype),
        "shape": [int(s.shape[0])],
        "shown": [int(window.shape[0])],
        "truncated": window.shape[0] != s.shape[0],
        "name": None if s.name is None else _label(s.name),
        "index": [_label(i) for i in window.index],
        "index_name": None if s.index.name is None else _label(s.index.name),
        "data": [_cell(v) for v in window.tolist()],
    }


# ---------------------------------------------------------------- entry point

def snap(value, highlight=None) -> dict:
    """One value -> a dict for the frontend.

    highlight only applies to ndarray / DataFrame / Series, and only explainers
    pass it.
    """
    if isinstance(value, np.ndarray):
        return _snap_ndarray(value, highlight)
    if isinstance(value, pd.DataFrame):
        return _snap_dataframe(value, highlight)
    if isinstance(value, pd.Series):
        return _snap_series(value, highlight)
    if isinstance(value, np.generic):
        return {"kind": "scalar", "py_type": type(value).__name__,
                "dtype": str(value.dtype), "data": _cell(value)}
    if value is None or isinstance(value, (bool, int, float, str, complex)):
        return {"kind": "scalar", "py_type": type(value).__name__,
                "dtype": type(value).__name__, "data": _cell(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)[:MAX_ITEMS]
        return {"kind": "sequence", "py_type": type(value).__name__,
                "len": len(value), "truncated": len(value) > MAX_ITEMS,
                "data": [_cell(v) for v in items]}
    if isinstance(value, dict):
        keys = list(value)[:MAX_ITEMS]
        return {"kind": "mapping", "py_type": "dict",
                "len": len(value), "truncated": len(value) > MAX_ITEMS,
                "data": [[_label(k), _cell(value[k])] for k in keys]}
    return {"kind": "opaque", "py_type": type(value).__name__, "repr": _short_repr(value)}

