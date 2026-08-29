"""numpy / pandas visualizer — run this file.

    python server.py

then open http://127.0.0.1:5000

This server execs whatever code is posted to it, with no sandbox, so it binds
to 127.0.0.1 only. Do not change that to 0.0.0.0 and do not deploy it.
"""

from __future__ import annotations

import sys

from flask import Flask, jsonify, render_template, request

import tracer

# The Windows console defaults to cp1252 and chokes on non-ASCII output
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MAX_CODE_CHARS = 20_000
MAX_INPUTS = 200        # guard against code that loops on input()

app = Flask(__name__)

# Flask sorts JSON keys by default, which would list variables
# alphabetically instead of in creation order — the order that reads well
app.json.sort_keys = False


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/trace")
def api_trace():
    payload = request.get_json(silent=True) or {}
    code = payload.get("code")

    if not isinstance(code, str) or not code.strip():
        return jsonify(error={"type": "BadRequest", "message": "No code was sent",
                              "line": None, "text": "No code was sent"}), 400
    if len(code) > MAX_CODE_CHARS:
        message = f"Code is longer than the {MAX_CODE_CHARS} character limit"
        return jsonify(error={"type": "BadRequest", "message": message,
                              "line": None, "text": message}), 400

    # Answers to input() pile up on the client; every run replays them all
    raw = payload.get("inputs") or []
    inputs = [str(value) for value in raw][:MAX_INPUTS] if isinstance(raw, list) else []

    # User code raising is normal: report it inside a 200, not as a 500
    return jsonify(tracer.trace(code, inputs=inputs))


if __name__ == "__main__":
    print("numpy / pandas visualizer -> http://127.0.0.1:8000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=8000, debug=True)
