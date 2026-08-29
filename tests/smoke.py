"""Fast checks over snapshot / heap / tracer / explainers, no pytest needed.

    python tests/smoke.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The Windows console defaults to cp1252 and chokes on non-ASCII output
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd

import snapshot
import tracer


def check(label: str, condition: bool) -> None:
    print(f"{'ok  ' if condition else 'FAIL'} {label}")
    if not condition:
        raise SystemExit(1)


def json_safe(obj) -> bool:
    """Everything leaving snapshot must serialize, or /api/trace fails outright."""
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (TypeError, ValueError) as exc:
        print("   ", exc)
        return False


# ------------------------------------------------------------------ snapshot

a = np.arange(6).reshape(2, 3)
snap_a = snapshot.snap(a)
check("2-D ndarray keeps its shape", snap_a["shape"] == [2, 3])
check("2-D ndarray keeps its data", snap_a["data"] == [[0, 1, 2], [3, 4, 5]])
# reshape returns a view of the unnamed arange result, so is_view is True
check("reshape counts as a view (it has a base)", snap_a["is_view"] is True)
check("a freshly built array has no base", snapshot.snap(np.zeros(3))["is_view"] is False)
check("ndarray snapshot serializes", json_safe(snap_a))

check("a slice is detected as a view", snapshot.snap(a[:, 1:])["is_view"] is True)
check("a copy is not a view", snapshot.snap(a[:, 1:].copy())["is_view"] is False)

big = np.zeros((50, 50))
snap_big = snapshot.snap(big)
check("a big array is cut down to 20x20", snap_big["shown"] == [20, 20] and snap_big["truncated"])

check("0-D arrays do not break", snapshot.snap(np.array(7))["data"] == 7)
check("3-D arrays are captured", snapshot.snap(np.zeros((2, 3, 4)))["shown"] == [2, 3, 4])

df = pd.DataFrame({"k": ["x", "y"], "v": [1.0, float("nan")]})
snap_df = snapshot.snap(df)
check("DataFrame columns are right", snap_df["columns"] == ["k", "v"])
check("NaN becomes null", snap_df["data"][1][1] is None)
check("DataFrame snapshot serializes", json_safe(snap_df))

snap_s = snapshot.snap(df["v"])
check("Series name is right", snap_s["name"] == "v")

check("inf does not break JSON", json_safe(snapshot.snap(np.array([np.inf, -np.inf]))))
check("scalars work", snapshot.snap(42)["data"] == 42)
check("lists work", snapshot.snap([1, 2, 3])["data"] == [1, 2, 3])
check("dicts work", snapshot.snap({"a": 1})["data"] == [["a", 1]])


# ------------------------------------------------------------------ heap

import heap as heap_mod  # noqa: E402


def slots(step, frame=0):
    """{name -> value} for one frame; value is {"inline":...} or {"ref":...}."""
    return {s["name"]: s["value"] for s in step["frames"][frame]["slots"]}


def deref(step, name, frame=0):
    """Follow a ref into the heap; inline values are returned as-is."""
    value = slots(step, frame)[name]
    return step["heap"][value["ref"]] if "ref" in value else value["inline"]


frames, objects, _ = heap_mod.snap_frames(
    [("Global frame", "<module>", 0,
      {"a": a, "b": a[:, 1:], "c": a.copy(), "np": np, "text": "hi", "nothing": None},
      None)],
    skip={"np"})
names = {s["name"]: s["value"] for s in frames[0]["slots"]}
check("snap_frames skips names in skip", "np" not in names)
check("primitives live in the frame, not a box", "inline" in names["text"] and "inline" in names["nothing"])
check("an ndarray gets its own box", "ref" in names["a"])
check("b is seen to share a buffer with a",
      objects[names["b"]["ref"]]["shares_memory_with"] == [names["a"]["ref"]])
check("c shares with nobody", objects[names["c"]["ref"]]["shares_memory_with"] == [])


# -------------------------------------------------------------------- tracer

# explain is off here: these check trace structure, sub-steps would skew counts
result = tracer.trace("import numpy as np\nx = np.arange(3)\ny = x * 2\nprint(y)\n",
                      explain=False)
check("trace has no error", result["error"] is None)
check("one step per line plus the closing step", len(result["steps"]) == 5)
check("last step is done", result["steps"][-1]["event"] == "done")
check("last step sees the value of y", deref(result["steps"][-1], "y")["data"] == [0, 2, 4])
check("stdout is captured", result["stdout"].strip() == "[0 2 4]")
check("stdout_len accumulates correctly", result["steps"][-1]["stdout_len"] == len(result["stdout"]))
check("preloaded np is not shown as a variable", "np" not in slots(result["steps"][0]))
check("trace serializes", json_safe(result))

err = tracer.trace("x = 1\ny = x / 0\n")
check("division by zero is reported, not raised", err["error"]["type"] == "ZeroDivisionError")
check("the error points at the right line", err["error"]["line"] == 2)
check("last step is error", err["steps"][-1]["event"] == "error")

asked = tracer.trace('name = input("who? ")\nprint(name)\n')
check("input() stops to ask instead of hanging the server",
      asked["error"] is None and asked["awaiting_input"]["prompt"] == "who? ")
check("last step says it is waiting for input", asked["steps"][-1]["event"] == "input")

answered = tracer.trace('name = input("who? ")\nprint(name)\n', inputs=["ann"])
check("with answers supplied it runs to the end",
      answered["awaiting_input"] is None and answered["error"] is None)
check("input is echoed into stdout like a terminal",
      answered["stdout"] == "who? ann\nann\n")

partial = tracer.trace('a = input("1? ")\nb = input("2? ")\n', inputs=["x"])
check("a partial answer set stops at the next prompt", partial["awaiting_input"]["prompt"] == "2? ")
check("the replaced input is not shown as a variable", "input" not in slots(answered["steps"][-1]))

bad = tracer.trace("x = (1\n")
check("SyntaxError is reported", bad["error"]["type"] == "SyntaxError")
check("SyntaxError produces no steps", bad["steps"] == [])

loop = tracer.trace("i = 0\nwhile True:\n    i += 1\n", max_steps=50)
check("an infinite loop trips the guard", loop["stopped"] == "max_steps")
check("the guard keeps the step count bounded", len(loop["steps"]) <= 52)

scoped = tracer.trace("def f(n):\n    m = n + 1\n    return m\n\nr = f(1)\n", explain=False)
check("the trace follows calls into functions", any(s["scope"] == "f" for s in scoped["steps"]))
check("call depth is counted", any(s["depth"] == 1 for s in scoped["steps"]))


# --------------------------------------------------------------- explainers

def substeps(code, **kwargs):
    result = tracer.trace(code, **kwargs)
    assert result["error"] is None, result["error"]
    return [s for s in result["steps"] if s["event"] == "explain"]


def ops(code, **kwargs):
    return {s["op"] for s in substeps(code, **kwargs)}


PRELUDE = "import numpy as np\nimport pandas as pd\n"

subs = substeps(PRELUDE + "a = np.arange(12).reshape(3, 4)\nblock = a[0:2, 1:3]\n")
picked = [s for s in subs if s["op"] == "subscript"]
check("subscript produces sub-steps", len(picked) == 2)
mask = picked[0]["boxes"][0]["snap"]["highlight"]
check("the highlight matches the cells actually sliced",
      [row[1:3] for row in mask[0:2]] == [[True, True], [True, True]]
      and sum(sum(row) for row in mask) == 4)
check("it says whether the result is a view or a copy", "view" in picked[1]["note"])

check("a boolean mask also lands in subscript",
      "subscript" in ops(PRELUDE + "a = np.array([3, -1, 4])\npicked = a[a > 0]\n"))
check("writing through a mask is explained too",
      "subscript" in ops(PRELUDE + "a = np.array([3, -1, 4])\na[a < 0] = 0\n"))

check("broadcasting produces 3 sub-steps",
      len(substeps(PRELUDE + "a = np.arange(3)\nb = a.reshape(3, 1)\nc = a + b\n"
                   )) == 3 + 2)  # broadcast 3 + reshape 2
check("equal shapes need no broadcasting explanation",
      "broadcast" not in ops(PRELUDE + "a = np.arange(3)\nb = np.arange(3)\nc = a + b\n"))

check("reshape produces sub-steps", "reshape" in ops(PRELUDE + "a = np.arange(6)\nb = a.reshape(2, 3)\n"))
check("transpose also lands in reshape", "reshape" in ops(PRELUDE + "a = np.arange(6).reshape(2, 3)\nb = a.T\n"))

reduced = substeps(PRELUDE + "a = np.arange(6).reshape(2, 3)\nt = a.sum(axis=0)\n")
check("reduce breaks into one sub-step per group", len([s for s in reduced if s["op"] == "reduce"]) == 3 + 2)

grouped = substeps(PRELUDE + 'df = pd.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})\n'
                             'g = df.groupby("k")["v"].sum()\n')
check("groupby gives split + groups + combine", len(grouped) == 4)
check("groupby highlights the rows of each group",
      grouped[1]["boxes"][0]["snap"]["highlight"] == [[True], [False], [True]])

merged = substeps(PRELUDE + 'l = pd.DataFrame({"id": [1, 2], "a": [1, 2]})\n'
                            'r = pd.DataFrame({"id": [2, 3], "b": [3, 4]})\n'
                            'out = l.merge(r, on="id")\n')
check("merge produces 3 sub-steps", len(merged) == 3)

check("melt lands in the pivot explainer",
      "melt" in ops(PRELUDE + 'df = pd.DataFrame({"k": ["a"], "x": [1], "y": [2]})\n'
                              'long = df.melt(id_vars="k")\n'))

# The replay guard, the load-bearing part of the explainers
side_effect = tracer.trace(PRELUDE + "calls = []\ndef make():\n    calls.append(1)\n"
                                     "    return np.arange(6)\nb = make().reshape(2, 3)\n")
check("a line calling the user's own function is not explained",
      not [s for s in side_effect["steps"] if s["event"] == "explain"])
check("the user's function is called exactly once",
      deref(side_effect["steps"][-1], "calls")["len"] == 1)
check("values that differ every run are not explained",
      not ops(PRELUDE + "b = np.random.rand(6).reshape(2, 3)\n"))

check("every sub-step points at a real parent step",
      all(0 <= s["parent"] < s["step"] for s in grouped))
check("explain can be turned off", not ops(PRELUDE + "a = np.arange(6)\nb = a.reshape(2, 3)\n", explain=False))
check("the sub-step cap works",
      len(substeps(PRELUDE + "a = np.arange(6)\nfor i in range(20):\n    b = a.reshape(2, 3)\n",
                   max_explain=4)) <= 4)
check("a trace with sub-steps serializes", json_safe(tracer.trace(
    PRELUDE + 'df = pd.DataFrame({"k": ["a", "b"], "v": [1.0, float("nan")]})\n'
              'g = df.groupby("k")["v"].sum()\n')))



# ------------------------------------------------- plain Python: frames + heap

plain = tracer.trace("nums = [3, 1, 2]\nalias = nums\ngrid = [[1, 2], [3, 4]]\n", explain=False)
last = plain["steps"][-1]
check("two names for one object share a ref",
      slots(last)["nums"]["ref"] == slots(last)["alias"]["ref"])
check("nested lists: inner cells are refs, not text",
      all("ref" in item for item in deref(last, "grid")["items"]))
check("a plain-Python trace serializes", json_safe(plain))

klass = tracer.trace("class Point:\n    def __init__(self, x):\n        self.x = x\n\np = Point(5)\n",
                     explain=False)
last = klass["steps"][-1]
check("an instance gets a box with its attributes",
      deref(last, "p")["kind"] == "instance" and deref(last, "p")["attrs"][0][0] == "x")
check("a class gets its own box", deref(last, "Point")["kind"] == "class")
check("a function shows its signature, not a memory address",
      "__init__(self, x)" in json.dumps(last["heap"], ensure_ascii=False))

stack = tracer.trace("def outer(n):\n    return inner(n)\n\ndef inner(n):\n    return n + 1\n\nr = outer(1)\n",
                     explain=False)
deepest = max(stack["steps"], key=lambda s: len(s["frames"]))
check("the whole call stack is visible, not one frame", len(deepest["frames"]) == 3)
check("frames run outermost to innermost", deepest["frames"][0]["title"] == "Global frame")

returns = [s for s in stack["steps"] if s["event"] == "return"]
check("there is a step when a function returns", len(returns) == 2)
check("the return step shows the returned value",
      any(slot.get("special") and slot["name"] == "Return value"
          for step in returns for frame in step["frames"] for slot in frame["slots"]))

cycle = tracer.trace("a = []\na.append(a)\n", explain=False)
check("a self-referencing object does not loop forever", cycle["error"] is None)
last = cycle["steps"][-1]
check("the self-pointing cell reuses the same ref",
      deref(last, "a")["items"][0]["ref"] == slots(last)["a"]["ref"])

# Two caps: cells per container (hit first) and objects per heap
wide = tracer.trace("items = [" + ", ".join(f"[{i}]" for i in range(90)) + "]\n", explain=False)
outer = deref(wide["steps"][-1], "items")
check("a long container is cut at the cell cap",
      len(outer["items"]) == heap_mod.MAX_ITEMS and outer["truncated"] is True)
check("but the real length is still reported", outer["len"] == 90)

deep = tracer.trace("deep = " + "[" * 90 + "1" + "]" * 90 + "\n", explain=False)
check("deep nesting does not blow up the heap",
      len(deep["steps"][-1]["heap"]) == heap_mod.MAX_OBJECTS)
check("it reports that some objects were left out", deep["steps"][-1]["heap_truncated"] is True)
# ---------------------------------------------------------------------- API

import server  # noqa: E402

client = server.app.test_client()

res = client.post("/api/trace", json={"code": "zeta = 1\nalpha = 2\nmid = 3\n"})
check("API keeps variables in creation order",
      list(slots(res.get_json()["steps"][-1])) == ["zeta", "alpha", "mid"])

res = client.post("/api/trace", json={"code": "1/0"})
check("broken code still returns 200 with error", res.status_code == 200 and bool(res.get_json()["error"]))

res = client.post("/api/trace", json={})
check("missing code returns 400 in English",
      res.status_code == 400 and res.get_json()["error"]["message"] == "No code was sent")

print("\nall good")
