"""Turn a chain of Python frames into "frames + heap", Python Tutor style.

Why not just snapshot.py
------------------------
`snapshot.snap()` returns a self-contained blob, which suits ndarray/DataFrame
(they share at the buffer level). Plain Python is a *graph of objects* instead:
`a = b` has to read as two names pointing at one box, not two identical boxes.

So every object goes into one heap and is referred to by ref. Frames and
container cells hold only refs, and the frontend draws an arrow per ref.
Cycles (linked lists, graphs) terminate because ids are remembered on the way in.
"""

from __future__ import annotations

import inspect
import types

import numpy as np
import pandas as pd

import snapshot

MAX_OBJECTS = 80          # stop registering past this, so loops cannot blow up
MAX_ITEMS = 50            # cells per container
MAX_ATTRS = 30            # attributes per instance / members per class

# These are drawn inside the frame itself and never get their own box
INLINE_TYPES = (bool, int, float, complex, str, bytes, type(None))


def _signature(func) -> str:
    name = getattr(func, "__name__", "?")
    try:
        return f"{name}{inspect.signature(func)}"
    except (TypeError, ValueError):
        return f"{name}(…)"


class Heap:
    """One step's object table. The same ref always means the same object."""

    MAX_ALIAS_ARRAYS = 12     # past this, skip the O(n^2) shares_memory scan

    def __init__(self):
        self.objects: dict[str, dict] = {}
        self._refs: dict[int, str] = {}   # id(obj) -> ref
        self._keep: list = []             # hold refs so gc cannot recycle an id
        self._arrays: list = []           # (ref, ndarray) for the buffer-sharing scan
        self._counter = 0
        self.truncated = False

    # -------------------------------------------------------------- values

    def value(self, obj) -> dict:
        """One value in a frame slot or a container cell.

        Returns {"inline": ...} for primitives, {"ref": "o3"} for objects.
        """
        if isinstance(obj, INLINE_TYPES) or isinstance(obj, np.generic):
            return {"inline": snapshot.snap(obj)}
        ref = self.register(obj)
        if ref is None:
            return {"inline": {"kind": "scalar", "py_type": type(obj).__name__,
                               "dtype": type(obj).__name__, "data": "…"}}
        return {"ref": ref}

    # -------------------------------------------------------------- object

    def register(self, obj) -> str | None:
        """Register an object and return its ref. None once MAX_OBJECTS is hit."""
        key = id(obj)
        if key in self._refs:
            return self._refs[key]
        if len(self.objects) >= MAX_OBJECTS:
            self.truncated = True
            return None

        ref = f"o{self._counter}"
        self._counter += 1
        self._refs[key] = ref
        self._keep.append(obj)
        # Reserve the slot before filling it, so self-referencing objects terminate
        self.objects[ref] = {"ref": ref, "kind": "opaque", "label": type(obj).__name__,
                             "repr": "…"}
        self.objects[ref] = self._describe(obj, ref)
        if isinstance(obj, np.ndarray):
            self._arrays.append((ref, obj))
        return ref

    def mark_shared_memory(self) -> None:
        """Fill in shares_memory_with (as refs) for ndarrays over one buffer.

        Different from two names pointing at one object: arrays that share a
        buffer really are separate objects, so they get separate boxes joined
        by a dashed line.
        """
        if len(self._arrays) > self.MAX_ALIAS_ARRAYS:
            return
        for ref, arr in self._arrays:
            shares = []
            for other_ref, other in self._arrays:
                if other_ref == ref:
                    continue
                try:
                    if np.shares_memory(arr, other):
                        shares.append(other_ref)
                except Exception:
                    pass
            self.objects[ref]["shares_memory_with"] = shares

    def _describe(self, obj, ref) -> dict:
        for builder in (self._numeric, self._sequence, self._mapping,
                        self._function, self._klass, self._instance):
            payload = builder(obj)
            if payload is not None:
                payload["ref"] = ref
                return payload
        return {"ref": ref, "kind": "opaque", "label": type(obj).__name__,
                "repr": snapshot._short_repr(obj)}

    # -------------------------------------------------------------- builders

    def _numeric(self, obj):
        """ndarray / DataFrame / Series reuse snapshot.py's payload as-is."""
        if not isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
            return None
        payload = snapshot.snap(obj)
        payload["label"] = payload["kind"]
        return payload

    def _sequence(self, obj):
        if not isinstance(obj, (list, tuple, set, frozenset)):
            return None
        items = list(obj)
        kind = type(obj).__name__
        return {
            "kind": "sequence",
            "label": kind,
            "py_type": kind,
            "len": len(items),
            "indexed": isinstance(obj, (list, tuple)),
            "truncated": len(items) > MAX_ITEMS,
            "items": [self.value(item) for item in items[:MAX_ITEMS]],
        }

    def _mapping(self, obj):
        if not isinstance(obj, dict):
            return None
        keys = list(obj)
        return {
            "kind": "mapping",
            "label": "dict",
            "len": len(keys),
            "truncated": len(keys) > MAX_ITEMS,
            "entries": [[snapshot._label(k), self.value(obj[k])] for k in keys[:MAX_ITEMS]],
        }

    def _function(self, obj):
        if not isinstance(obj, (types.FunctionType, types.MethodType,
                                types.BuiltinFunctionType)):
            return None
        return {"kind": "function", "label": "function", "signature": _signature(obj)}

    def _klass(self, obj):
        if not isinstance(obj, type):
            return None
        members = []
        for name, value in list(vars(obj).items())[:MAX_ATTRS]:
            if name.startswith("__") and name.endswith("__") and name != "__init__":
                continue
            members.append([name, self.value(value)])
        return {"kind": "class", "label": f"class {obj.__name__}", "members": members}

    def _instance(self, obj):
        """Instance of a user-defined class, read from __dict__ or __slots__."""
        attrs = {}
        if hasattr(obj, "__dict__") and isinstance(vars(obj), dict):
            attrs = dict(vars(obj))
        elif hasattr(type(obj), "__slots__"):
            attrs = {slot: getattr(obj, slot) for slot in type(obj).__slots__
                     if hasattr(obj, slot)}
        else:
            return None
        rows = [[name, self.value(value)]
                for name, value in list(attrs.items())[:MAX_ATTRS]]
        return {"kind": "instance", "label": f"{type(obj).__name__} instance",
                "attrs": rows, "truncated": len(attrs) > MAX_ATTRS}


def _hidden(name: str, value, skip: set) -> bool:
    if name in skip:
        return True
    if name.startswith("__") and name.endswith("__"):
        return True
    return isinstance(value, types.ModuleType)


def snap_frames(frames, skip=()) -> tuple[list, dict, bool]:
    """frames = [(title, scope, depth, namespace, extra), ...], outermost first.

    extra is an optional trailing slot such as ("Return value", value).
    Returns (frame payload, heap, whether a cap was hit).
    """
    skip = set(skip)
    heap = Heap()
    payload = []

    for title, scope, depth, namespace, extra in frames:
        slots = []
        for name, value in list(namespace.items()):
            if _hidden(name, value, skip):
                continue
            try:
                slots.append({"name": name, "value": heap.value(value)})
            except Exception as exc:
                slots.append({"name": name, "value": {"inline": {
                    "kind": "scalar", "py_type": "error", "dtype": "error",
                    "data": f"<snapshot failed: {type(exc).__name__}>"}}})
        if extra is not None:
            label, value = extra
            slots.append({"name": label, "value": heap.value(value), "special": True})
        payload.append({"title": title, "scope": scope, "depth": depth, "slots": slots})

    heap.mark_shared_memory()
    return payload, heap.objects, heap.truncated
