"""Run user code line by line with sys.settrace, snapshotting state per step.

A step means: **about to run** that line, same as Python Tutor. The effect of
the last line therefore shows up in the trailing event="done" step.

Security
--------
This file execs user-supplied code with no sandbox. Local use only, behind a
server bound to 127.0.0.1. Do not deploy it.

The guards (step cap, timeout) are only checked *between lines*, so a heavy
loop inside numpy's C layer will not be interrupted.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd

import explainers
import heap

# The Windows console defaults to cp1252 and chokes on non-ASCII output
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

USER_FILE = "<visualizer>"
DEFAULT_MAX_STEPS = 500
DEFAULT_TIMEOUT = 5.0
DEFAULT_MAX_EXPLAIN = 200   # sub-steps allowed across the whole trace
MAX_EXPLAIN_PER_LINE = 3    # a line in a loop is explained a few times only


class _Stop(BaseException):
    """Guard tripped. BaseException so user `except Exception` cannot swallow it."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _NeedInput(BaseException):
    """input() was called but the supplied answers ran out.

    The server keeps no session. It stops here, the frontend collects the
    answer into a list, and the whole program is **re-run** with every answer
    so far (same trick as Python Tutor). Replays stay deterministic.
    """

    def __init__(self, prompt: str):
        super().__init__(prompt)
        self.prompt = prompt


def _user_frames(frame) -> list:
    """The user's frames, outermost (global) to innermost (running)."""
    chain = []
    current = frame
    while current is not None:
        if current.f_code.co_filename == USER_FILE:
            chain.append(current)
        current = current.f_back
    chain.reverse()
    return chain


def _user_depth(frame) -> int:
    """Frame depth, counting only frames from user code."""
    return len(_user_frames(frame)) - 1


def _frame_inputs(frame, extra=None) -> list:
    """Input for heap.snap_frames; the extra slot goes on the innermost frame."""
    chain = _user_frames(frame)
    inputs = []
    for depth, item in enumerate(chain):
        scope = item.f_code.co_name
        title = "Global frame" if scope == "<module>" else scope
        inputs.append((title, scope, depth, item.f_locals,
                       extra if item is frame else None))
    return inputs


def _format_error(exc: BaseException) -> dict:
    """Pull just the user-code line out of a traceback."""
    line = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == USER_FILE:
            line = tb.tb_lineno
        tb = tb.tb_next
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "line": line,
        "text": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


def trace(code: str,
          max_steps: int = DEFAULT_MAX_STEPS,
          timeout: float = DEFAULT_TIMEOUT,
          explain: bool = True,
          max_explain: int = DEFAULT_MAX_EXPLAIN,
          inputs=None) -> dict:
    """Run the code and return the whole trace. Never raises: failures are in it."""
    source_lines = code.splitlines()
    result = {
        "steps": [],
        "stdout": "",
        "error": None,
        "stopped": None,          # None | "max_steps" | "timeout"
        "awaiting_input": None,   # {"prompt": ...} when input() ran out of answers
        "source": source_lines,
        "limits": {"max_steps": max_steps, "timeout": timeout},
    }

    try:
        tree = ast.parse(code, USER_FILE)
        compiled = compile(tree, USER_FILE, "exec")
    except SyntaxError as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": exc.msg,
            "line": exc.lineno,
            "text": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }
        return result

    # np / pd come preloaded; baseline is the set of names never shown as
    # variables. input is replaced by one that reads the supplied answers.
    env = {"__name__": "__main__", "np": np, "pd": pd, "input": None}
    baseline = set(env)

    queue = list(inputs or [])
    taken = {"n": 0}

    def fake_input(prompt=""):
        text = "" if prompt is None else str(prompt)
        if taken["n"] < len(queue):
            value = queue[taken["n"]]
            taken["n"] += 1
            print(text + value)     # echo it, the way a real terminal would
            return value
        raise _NeedInput(text)

    env["input"] = fake_input

    out = io.StringIO()
    steps = result["steps"]
    counter = {"n": 0, "explained": 0}
    started = time.perf_counter()

    statements = explainers.statement_index(tree) if explain else {}
    tried = {}          # line number -> how many times we tried to explain it
    hopeless = set()    # lines no explainer accepts; do not retry them

    def record(frame, event: str, extra=None) -> None:
        frames, objects, truncated = heap.snap_frames(_frame_inputs(frame, extra),
                                                      skip=baseline)
        steps.append({
            "step": len(steps),
            "line": frame.f_lineno,
            "event": event,
            "scope": frame.f_code.co_name,
            "depth": _user_depth(frame),
            "frames": frames,
            "heap": objects,
            "heap_truncated": truncated,
            "stdout_len": out.tell(),
        })

    def add_explanation(frame) -> None:
        """Append sub-steps after this line's step, if an explainer accepts it."""
        line = frame.f_lineno
        if counter["explained"] >= max_explain or line in hopeless:
            return
        if tried.get(line, 0) >= MAX_EXPLAIN_PER_LINE:
            return
        stmt = statements.get(line)
        if stmt is None:
            hopeless.add(line)
            return

        tried[line] = tried.get(line, 0) + 1
        env = {**frame.f_globals, **frame.f_locals}
        subs = explainers.explain(stmt, env)
        if not subs:
            hopeless.add(line)
            return

        parent = len(steps) - 1
        for sub in subs:
            steps.append({
                "step": len(steps),
                "line": line,
                "event": "explain",
                "parent": parent,
                "scope": frame.f_code.co_name,
                "depth": _user_depth(frame),
                "op": sub["op"],
                "title": sub["title"],
                "note": sub.get("note"),
                "boxes": sub["boxes"],
                "stdout_len": out.tell(),
            })
            counter["explained"] += 1

    def _budget() -> None:
        counter["n"] += 1
        if counter["n"] > max_steps:
            raise _Stop("max_steps")
        if time.perf_counter() - started > timeout:
            raise _Stop("timeout")

    def tracefunc(frame, event, arg):
        if frame.f_code.co_filename != USER_FILE:
            return None

        if event == "line":
            _budget()
            record(frame, "line")
            if explain:
                add_explanation(frame)
            return tracefunc

        # Show the return value as an extra slot before the frame disappears.
        # The module frame is skipped: the trailing done step already covers it.
        if event == "return" and frame.f_code.co_name != "<module>":
            _budget()
            record(frame, "return", extra=("Return value", arg))

        return tracefunc

    last_line = len(source_lines)
    with contextlib.redirect_stdout(out):
        # The server's stdin is still the terminal. Without this swap, code
        # reading stdin would hang the server waiting for a keypress.
        real_stdin, sys.stdin = sys.stdin, io.StringIO()
        sys.settrace(tracefunc)
        try:
            exec(compiled, env)
        except _Stop as stop:
            result["stopped"] = stop.reason
        except _NeedInput as need:
            result["awaiting_input"] = {"prompt": need.prompt, "answered": taken["n"]}
        except BaseException as exc:  # noqa: BLE001 — user code raising is normal
            result["error"] = _format_error(exc)
            last_line = result["error"]["line"] or last_line
        finally:
            sys.settrace(None)
            sys.stdin = real_stdin

    # Closing step: state after the last line ran. Without it the effect of
    # that line would never be visible.
    frames, objects, truncated = heap.snap_frames(
        [("Global frame", "<module>", 0, env, None)], skip=baseline)
    steps.append({
        "step": len(steps),
        "line": last_line,
        "event": ("error" if result["error"]
                  else "input" if result["awaiting_input"] else "done"),
        "scope": "<module>",
        "depth": 0,
        "frames": frames,
        "heap": objects,
        "heap_truncated": truncated,
        "stdout_len": out.tell(),
    })

    result["stdout"] = out.getvalue()
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <file.py>", file=sys.stderr)
        raise SystemExit(2)
    with open(sys.argv[1], encoding="utf-8") as fh:
        payload = trace(fh.read())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
