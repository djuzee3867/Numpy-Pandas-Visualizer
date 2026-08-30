# Numpy Pandas Visualizer

A step-by-step visualizer for Python code, built for teaching **numpy** and **pandas**.

It started from using [pythontutor.com](https://pythontutor.com/visualize.html). That tool
is excellent for teaching plain Python, but it does not cover numpy and pandas — and those are
exactly the parts students find hardest to picture, because the interesting work happens inside a
single line. So this was built to teach them: same idea, extended to arrays and dataframes, with
the option to open one line up and walk through what it does. 

# Website<br>
[Numpy Pandas Visualizer](https://numpy-pandas-visualizer.djuzee.site/) 

[DEMO](https://project.djuzee.site/python)

## Quick Start

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

```bash
python server.py
```

Open <http://127.0.0.1:8000>, write some Python or pick from **Examples**, then press
**Visualize Execution**.

Step with `Next >` / `< Prev` or the arrow keys; `Home` / `End` jump to the ends; `Ctrl`+`Enter`
runs; `Reset` clears everything. The editor stays editable at all times.

> The server `exec`s whatever is in the editor with **no sandbox** only. Don't deploy it.

Tests: `python tests/smoke.py`

## Features

- **Frames and objects**, Python Tutor style — one box per object, so `b = a` draws two arrows into
  one box instead of two look-alike boxes.
- **Call stack, classes, instances, linked lists**, and a `Return value` slot when a function
  returns.
- **Real grids** for `ndarray` (1-D to 3-D), `DataFrame` (with dtypes) and `Series` — not `repr()`.
  `NaN` looks different from `0`.
- **Views vs copies**: arrays sharing a buffer are joined by a dashed line and labelled.
- **Changed cells highlighted** against the previous step.
- **Operation explainers** that expand one line into sub-steps: slicing and boolean masks,
  broadcasting, reshape/transpose, axis reductions, `groupby`, `merge`, `pivot`/`melt`.
- **`input()` works**, `print()` output is shown per step, and errors are reported inline.
- Light/dark themes, adjustable font size, draggable panes.

Explainers are read-only: a line is only expanded when its expressions can be safely evaluated
again, so calls to your own functions or to `np.random` are left alone rather than run twice.


Most of this code, were written with [Claude](https://claude.com/claude-code).
**It can be wrong** — please read the code before relying on it
