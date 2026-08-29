/* ============================================================
   numpy / pandas visualizer — frontend
   The whole trace arrives from /api/trace in one response; stepping happens
   locally, so the buttons never hit the server again.

   Layout follows Python Tutor: a Frames column of variable names, an Objects
   column of the values, and SVG wires between them.
   ============================================================ */


const EXAMPLES = [
  {
    group: "Python",
    name: "Variables and swap",
    code: `a = 1
b = 2
a, b = b, a
print(a, b)
`,
  },
  {
    group: "Python",
    name: "Loop that builds a list",
    code: `squares = []
for n in range(1, 6):
    squares.append(n * n)

print(squares)
`,
  },
  {
    group: "Python",
    name: "Two names, one list",
    code: `first = [1, 2, 3]
second = first          # the same list, not a copy
third = first[:]        # a real copy

second.append(4)        # first changes too
third.append(99)        # first is untouched
print(first, third)
`,
  },
  {
    group: "Python",
    name: "Nested list (grid)",
    code: `grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
row = grid[1]           # points into grid, no copy
row[0] = 99             # writes through
center = grid[1][1]
print(grid, center)
`,
  },
  {
    group: "Python",
    name: "Counting with a dict",
    code: `words = ["a", "b", "a", "c", "a"]
counts = {}

for word in words:
    counts[word] = counts.get(word, 0) + 1

print(counts)
`,
  },
  {
    group: "Python",
    name: "Bubble sort",
    code: `nums = [5, 2, 4, 1]

for i in range(len(nums)):
    for j in range(len(nums) - 1 - i):
        if nums[j] > nums[j + 1]:
            nums[j], nums[j + 1] = nums[j + 1], nums[j]

print(nums)
`,
  },
  {
    group: "Python",
    name: "Recursion and the call stack",
    code: `def total(items):
    if not items:
        return 0
    return items[0] + total(items[1:])

answer = total([1, 2, 3, 4])
print(answer)
`,
  },
  {
    group: "Python",
    name: "Class with two objects",
    code: `class Account:
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

a = Account("ann", 100)
b = Account("bob", 0)
a.deposit(50)
b.deposit(20)
print(a.balance, b.balance)
`,
  },
  {
    group: "Python",
    name: "Linked list of nodes",
    code: `class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

head = Node("a")
head.next = Node("b")
head.next.next = Node("c")

node = head
while node is not None:
    print(node.value)
    node = node.next
`,
  },
  {
    group: "Python",
    name: "Reading input()",
    code: `name = input("What is your name? ")
age = input("How old are you? ")

print("hello", name)
print("next year you turn", int(age) + 1)
`,
  },
  {
    group: "numpy",
    name: "2-D slicing",
    code: `a = np.arange(12).reshape(3, 4)
row = a[1]
col = a[:, 2]
block = a[0:2, 1:3]
strided = a[::2, ::2]
`,
  },
  {
    group: "numpy",
    name: "view vs copy",
    code: `a = np.arange(6).reshape(2, 3)
view = a[:, 1:]          # shares memory with a
copy = a[:, 1:].copy()   # its own buffer

view += 100              # a changes too
copy += 1000             # a is untouched
`,
  },
  {
    group: "numpy",
    name: "broadcasting",
    code: `a = np.arange(3)                 # shape (3,)
b = np.arange(3).reshape(3, 1)  # shape (3, 1)
total = a + b                   # -> shape (3, 3)
scaled = a * 10
`,
  },
  {
    group: "numpy",
    name: "boolean mask",
    code: `a = np.array([3, -1, 4, -1, 5, -9])
mask = a > 0
picked = a[mask]
a[a < 0] = 0
`,
  },
  {
    group: "pandas",
    name: "groupby",
    code: `df = pd.DataFrame({
    "team": ["a", "b", "a", "b", "a"],
    "score": [10, 7, 13, 9, 5],
})
total = df.groupby("team")["score"].sum()
mean = df.groupby("team")["score"].mean()
`,
  },
  {
    group: "pandas",
    name: "merge",
    code: `left = pd.DataFrame({"id": [1, 2, 3], "name": ["ann", "bob", "cy"]})
right = pd.DataFrame({"id": [2, 3, 4], "score": [88.0, 92.0, 70.0]})

inner = left.merge(right, on="id")
outer = left.merge(right, on="id", how="outer")
`,
  },
  {
    group: "pandas",
    name: "melt / pivot",
    code: `wide = pd.DataFrame({
    "city": ["bkk", "cnx"],
    "jan": [32, 28],
    "feb": [33, 29],
})
long = wide.melt(id_vars="city", var_name="month", value_name="temp")
back = long.pivot(index="city", columns="month", values="temp")
`,
  },
];

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const OBJECT_GAP = 28;         // gap between a parent box's right edge and its child
const MAX_INDENT = 420;        // deeper than this, stop stepping right

const state = {
  trace: null,
  index: 0,
  inputs: [],     // answers to input(), replayed on every run
  memLinks: [],   // dashed links between buffer-sharing arrays, redrawn on scroll
};

/* ------------------------------------------------------------ DOM helpers */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function chip(text, cls) {
  return el("span", cls ? `chip ${cls}` : "chip", text);
}

/* ------------------------------------------------------------ formatting */

function fmt(value) {
  if (value === null) return "NaN";
  if (value === true) return "True";
  if (value === false) return "False";
  if (value === "Infinity") return "∞";
  if (value === "-Infinity") return "-∞";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 1e-4 || abs >= 1e6)) return value.toExponential(3);
    return String(Math.round(value * 1e4) / 1e4);
  }
  return String(value);
}

function cellClass(value) {
  if (value === null) return "na";
  if (typeof value === "number" || typeof value === "boolean") return "";
  return "text";
}

/* ------------------------------------------------------------ diff */

// Drop keys that move without the value changing (numpy reallocates, ptr shifts)
function stableKey(snap) {
  if (!snap) return null;
  const clone = { ...snap };
  delete clone.ptr;
  delete clone.shares_memory_with;
  return JSON.stringify(clone);
}

function sameLayout(now, prev) {
  if (!prev || now.kind !== prev.kind) return false;
  if (JSON.stringify(now.shown) !== JSON.stringify(prev.shown)) return false;
  if (now.kind === "DataFrame") {
    return JSON.stringify(now.columns) === JSON.stringify(prev.columns)
        && JSON.stringify(now.index) === JSON.stringify(prev.index);
  }
  if (now.kind === "Series") {
    return JSON.stringify(now.index) === JSON.stringify(prev.index);
  }
  return true;
}

/* ------------------------------------------------------------ grids */

function grid2d(rows, prevRows, opts = {}) {
  const { rowHeads = null, colHeads = null, dtypes = null, cornerText = "",
          highlight = null } = opts;
  const table = el("table", "grid");
  const width = rows.length ? rows[0].length : 0;

  if (colHeads) {
    const head = el("tr");
    if (rowHeads) head.appendChild(el("th", "corner", cornerText));
    colHeads.forEach((label) => head.appendChild(el("th", "colhead", String(label))));
    table.appendChild(head);

    if (dtypes) {
      const row = el("tr", "dtype-row");
      if (rowHeads) row.appendChild(el("th", "corner", ""));
      dtypes.forEach((t) => row.appendChild(el("th", "", t)));
      table.appendChild(row);
    }
  }

  rows.forEach((cells, r) => {
    const tr = el("tr");
    if (rowHeads) tr.appendChild(el("th", "rowhead", String(rowHeads[r])));
    for (let c = 0; c < width; c += 1) {
      const value = cells[c];
      const td = el("td", cellClass(value), fmt(value));
      const before = prevRows && prevRows[r] ? prevRows[r][c] : undefined;
      if (prevRows && before !== undefined && before !== value) td.classList.add("chg");
      if (highlight && highlight[r] && highlight[r][c]) td.classList.add("sel");
      tr.appendChild(td);
    }
    table.appendChild(tr);
  });

  return table;
}

const range = (n) => Array.from({ length: n }, (_, i) => i);

function truncNote(snap) {
  if (!snap.truncated) return null;
  const full = snap.shape.join(" × ");
  const shown = snap.shown.join(" × ");
  return el("div", "trunc-note", `showing ${shown} of ${full} — the rest is omitted`);
}

/* ------------------------------------------------------------ renderers */

function renderNdarray(snap, prev) {
  const body = el("div");
  const usePrev = sameLayout(snap, prev) ? prev : null;

  if (snap.ndim === 0) {
    body.appendChild(el("div", `scalar-value ${snap.data === null ? "na" : ""}`, fmt(snap.data)));
    return body;
  }

  if (snap.ndim === 1) {
    body.appendChild(grid2d([snap.data], usePrev ? [usePrev.data] : null, {
      colHeads: range(snap.shown[0]),
      highlight: snap.highlight ? [snap.highlight] : null,
    }));
  } else if (snap.ndim === 2) {
    body.appendChild(grid2d(snap.data, usePrev ? usePrev.data : null, {
      rowHeads: range(snap.shown[0]),
      colHeads: range(snap.shown[1]),
      highlight: snap.highlight,
    }));
  } else if (snap.ndim === 3) {
    snap.data.forEach((plane, i) => {
      body.appendChild(el("div", "slice-label", `[${i}, :, :]`));
      body.appendChild(grid2d(plane, usePrev ? usePrev.data[i] : null, {
        rowHeads: range(snap.shown[1]),
        colHeads: range(snap.shown[2]),
        highlight: snap.highlight ? snap.highlight[i] : null,
      }));
    });
  } else {
    body.appendChild(el("div", "opaque-repr",
      `${snap.ndim}-D array — not drawn in this version (shape ${snap.shape.join(" × ")})`));
  }

  const note = truncNote(snap);
  if (note) body.appendChild(note);
  return body;
}

function renderDataFrame(snap, prev) {
  const body = el("div");
  const usePrev = sameLayout(snap, prev) ? prev : null;
  body.appendChild(grid2d(snap.data, usePrev ? usePrev.data : null, {
    rowHeads: snap.index,
    colHeads: snap.columns,
    dtypes: snap.dtypes,
    cornerText: snap.index_name || "",
    highlight: snap.highlight,
  }));
  const note = truncNote(snap);
  if (note) body.appendChild(note);
  return body;
}

function renderSeries(snap, prev) {
  const body = el("div");
  const usePrev = sameLayout(snap, prev) ? prev : null;
  body.appendChild(grid2d(snap.data.map((v) => [v]), usePrev ? usePrev.data.map((v) => [v]) : null, {
    rowHeads: snap.index,
    colHeads: [snap.name === null ? "value" : snap.name],
    dtypes: [snap.dtype],
    cornerText: snap.index_name || "",
    highlight: snap.highlight,
  }));
  const note = truncNote(snap);
  if (note) body.appendChild(note);
  return body;
}

function renderSequence(snap) {
  const body = el("div");
  const wrap = el("div", "inline-items");
  snap.data.forEach((v) => wrap.appendChild(el("span", "item", fmt(v))));
  body.appendChild(wrap);
  if (snap.truncated) {
    body.appendChild(el("div", "trunc-note", `showing ${snap.data.length} of ${snap.len} items`));
  }
  return body;
}

function renderMapping(snap) {
  const body = el("div");
  body.appendChild(grid2d(snap.data.map(([, v]) => [v]), null, {
    rowHeads: snap.data.map(([k]) => k),
    colHeads: ["value"],
  }));
  if (snap.truncated) {
    body.appendChild(el("div", "trunc-note", `showing ${snap.data.length} of ${snap.len} keys`));
  }
  return body;
}

function renderOpaque(snap) {
  return el("div", "opaque-repr", snap.repr);
}

function renderScalar(snap) {
  return el("div", `scalar-value ${snap.data === null ? "na" : ""}`, fmt(snap.data));
}

const RENDERERS = {
  ndarray: renderNdarray,
  DataFrame: renderDataFrame,
  Series: renderSeries,
  sequence: renderSequence,
  mapping: renderMapping,
  scalar: renderScalar,
  opaque: renderOpaque,
};

/* ------------------------------------------------------------ labels */

function shapeText(snap) {
  if (snap.kind === "ndarray") return snap.ndim === 0 ? "0-D" : snap.shape.join(" × ");
  if (snap.kind === "DataFrame") {
    const [r, c] = snap.shape;
    return `${r} ${r === 1 ? "row" : "rows"} × ${c} ${c === 1 ? "col" : "cols"}`;
  }
  if (snap.kind === "Series") return `${snap.shape[0]} values`;
  if (snap.kind === "sequence") return `${snap.len} ${snap.len === 1 ? "item" : "items"}`;
  if (snap.kind === "mapping") return `${snap.len} ${snap.len === 1 ? "key" : "keys"}`;
  if (snap.kind === "instance" || snap.kind === "class") return null;
  return null;
}

function typeLabel(snap) {
  if (snap.kind === "scalar" || snap.kind === "opaque") return snap.py_type;
  if (snap.kind === "sequence") return snap.py_type;
  if (snap.kind === "mapping") return "dict";
  return snap.kind;
}

/* --------------------------------------------------- values in frames and cells

   A value from the backend has one of two shapes:
     {"inline": <snap of a primitive>}  -> draw the text right there
     {"ref": "o3"}                      -> draw a dot, wire it to box heap-o3
   ------------------------------------------------------------------------- */

function inlineText(snap) {
  if (!snap) return "?";
  if (snap.py_type === "NoneType") return "None";
  if (snap.py_type === "str") return JSON.stringify(snap.data);
  return fmt(snap.data);
}

function valueCell(value, cls) {
  if (value && value.ref) {
    const slot = el("span", `${cls} ptr`);
    slot.dataset.target = `heap-${value.ref}`;
    slot.appendChild(el("span", "ptr-dot"));
    return slot;
  }
  const snap = value ? value.inline : null;
  const cell = el("span", cls, inlineText(snap));
  if (snap && snap.py_type === "NoneType") cell.classList.add("none");
  else if (snap && snap.py_type === "str") cell.classList.add("str");
  return cell;
}

const stableValue = (value) => JSON.stringify(value);

/* ------------------------------------------------------------ heap boxes */

function sequenceBody(entry, prev) {
  if (!entry.items.length) return el("div", "empty-note", "empty");
  const table = el("table", "grid seq");

  if (entry.indexed) {
    const head = el("tr");
    entry.items.forEach((_, i) => head.appendChild(el("th", "colhead", String(i))));
    table.appendChild(head);
  }

  const row = el("tr");
  entry.items.forEach((value, i) => {
    const td = el("td");
    const before = prev && prev.items ? prev.items[i] : undefined;
    if (before !== undefined && stableValue(before) !== stableValue(value)) {
      td.classList.add("chg");
    }
    td.appendChild(valueCell(value, "val"));
    row.appendChild(td);
  });
  table.appendChild(row);

  const wrap = el("div");
  wrap.appendChild(table);
  if (entry.truncated) {
    wrap.appendChild(el("div", "trunc-note", `showing ${entry.items.length} of ${entry.len} items`));
  }
  return wrap;
}

function rowsBody(pairs, prevPairs, keyClass) {
  const wrap = el("div", "obj-rows");
  if (!pairs.length) return el("div", "empty-note", "empty");

  const before = new Map((prevPairs || []).map(([name, value]) => [name, stableValue(value)]));
  pairs.forEach(([name, value]) => {
    const row = el("div", "obj-row");
    row.appendChild(el("span", `obj-key ${keyClass}`, name));
    row.appendChild(valueCell(value, "obj-val"));
    if (before.has(name) && before.get(name) !== stableValue(value)) row.classList.add("chg-row");
    wrap.appendChild(row);
  });
  return wrap;
}

function heapBody(entry, prev) {
  switch (entry.kind) {
    case "sequence":
      return sequenceBody(entry, prev);
    case "mapping":
      return rowsBody(entry.entries, prev ? prev.entries : null, "is-key");
    case "instance":
      return rowsBody(entry.attrs, prev ? prev.attrs : null, "is-attr");
    case "class":
      return rowsBody(entry.members, prev ? prev.members : null, "is-attr");
    case "function":
      return el("div", "signature", entry.signature);
    default:
      return (RENDERERS[entry.kind] || renderOpaque)(entry, prev);
  }
}

function buildHeapObject(entry, prev, memGroup, nameOf) {
  const box = el("div", "obj");
  box.id = `heap-${entry.ref}`;

  const head = el("div", "obj-head");
  head.appendChild(el("span", "obj-type", entry.label || entry.kind));

  const shape = shapeText(entry);
  if (shape) head.appendChild(chip(shape));
  if (entry.dtype && entry.kind !== "DataFrame") head.appendChild(chip(entry.dtype));

  // Arrays over one buffer are separate objects, so separate boxes, dashed link
  const shares = entry.shares_memory_with;
  if (shares && shares.length) {
    const badge = chip("", "mem");
    badge.append("⚭ shares memory with ");
    shares.forEach((other, i) => {
      if (i) badge.append(", ");
      badge.appendChild(el("code", null, nameOf.get(other) || other));
    });
    badge.style.setProperty("--mem-color", `var(--mem-${memGroup % 5})`);
    head.appendChild(badge);
    box.style.borderColor = `var(--mem-${memGroup % 5})`;
  }

  const isNew = prev === undefined;
  if (isNew) {
    box.classList.add("is-new");
    head.appendChild(chip("new", "new"));
  } else if (stableKey(entry) !== stableKey(prev)) {
    head.appendChild(chip("changed", "changed"));
  }

  box.appendChild(head);
  const body = el("div", "obj-body");
  body.appendChild(heapBody(entry, isNew ? undefined : prev));
  box.appendChild(body);
  return box;
}

function childRefs(entry) {
  const refs = [];
  const take = (value) => {
    if (value && value.ref) refs.push(value.ref);
  };
  (entry.items || []).forEach(take);
  (entry.entries || []).forEach(([, value]) => take(value));
  (entry.attrs || []).forEach(([, value]) => take(value));
  (entry.members || []).forEach(([, value]) => take(value));
  return refs;
}

/* Walk the reference tree depth-first from the frame variables, placing each
   object as it is met. Children then follow their parent in cell order, which
   is what keeps the wires from crossing. (Ordering purely by depth pulls any
   directly-named box ahead of its siblings and tangles them.) */

function heapLayout(frames, heap) {
  const seen = new Set();
  const parent = new Map();
  const order = [];

  const visit = (ref, from) => {
    if (!heap[ref] || seen.has(ref)) return;
    seen.add(ref);
    if (from) parent.set(ref, from);
    order.push(ref);
    childRefs(heap[ref]).forEach((child) => visit(child, ref));
  };

  frames.forEach((frame) => frame.slots.forEach((slot) => {
    const ref = slot.value && slot.value.ref;
    if (ref) visit(ref, null);
  }));
  // Anything nothing points at any more still gets shown, at the end
  Object.keys(heap).forEach((ref) => visit(ref, null));

  return { order, parent };
}

/* Start each child past its parent's *measured* right edge, not at a fixed step.

   A fixed step leaves the child sitting under the parent (boxes are wider than
   the step), so the wire has to dive through the parent box before turning
   back. Clearing the right edge turns every wire into a short curve down and
   to the right, through empty space. */

function positionObjects(order, parent) {
  const offset = new Map();
  order.forEach((ref) => {
    const box = document.getElementById(`heap-${ref}`);
    if (!box) return;
    const from = parent.get(ref);
    let left = 0;
    if (from && offset.has(from)) {
      const parentBox = document.getElementById(`heap-${from}`);
      left = offset.get(from) + (parentBox ? parentBox.offsetWidth : 0) + OBJECT_GAP;
    }
    left = Math.min(left, MAX_INDENT);
    offset.set(ref, left);
    box.style.marginLeft = `${left}px`;
  });
}

// Group buffer-sharing arrays so each group gets one colour
function memoryGroups(heap) {
  const groups = new Map();
  let next = 0;
  Object.keys(heap).forEach((ref) => {
    if (groups.has(ref)) return;
    const shares = heap[ref].shares_memory_with;
    if (!shares || !shares.length) return;
    const stack = [ref];
    while (stack.length) {
      const current = stack.pop();
      if (groups.has(current)) continue;
      groups.set(current, next);
      ((heap[current] || {}).shares_memory_with || []).forEach((other) => {
        if (!groups.has(other) && heap[other]) stack.push(other);
      });
    }
    next += 1;
  });
  return groups;
}

/* ------------------------------------------------------------ frames + objects */

function renderHeap(step, prevStep) {
  const framesCol = $("frames");
  const objectsCol = $("objects");
  framesCol.textContent = "";
  objectsCol.textContent = "";
  state.memLinks = [];

  const frames = step.frames || [];
  const heap = step.heap || {};
  // The previous step's heap compares directly: refs come from the same ids
  const prevHeap = prevStep ? (prevStep.heap || null) : null;

  // Which refs have a name in a frame, for the "shares memory with a" badge
  const nameOf = new Map();
  frames.forEach((frame) => frame.slots.forEach((slot) => {
    if (slot.value && slot.value.ref && !nameOf.has(slot.value.ref)) {
      nameOf.set(slot.value.ref, slot.name);
    }
  }));

  frames.forEach((frame, i) => {
    const active = i === frames.length - 1;
    const card = el("div", `frame${active ? " is-active" : ""}`);
    card.appendChild(el("div", "frame-title", frame.title));

    if (!frame.slots.length) {
      card.appendChild(el("div", "frame-empty", "no variables yet"));
    } else {
      const rows = el("div", "frame-rows");
      frame.slots.forEach((slot) => {
        const row = el("div", `frame-row${slot.special ? " is-special" : ""}`);
        row.appendChild(el("span", "fname", slot.name));
        row.appendChild(valueCell(slot.value, "fval"));
        rows.appendChild(row);
      });
      card.appendChild(rows);
    }
    framesCol.appendChild(card);
  });

  const groups = memoryGroups(heap);
  const { order, parent } = heapLayout(frames, heap);

  order.forEach((ref) => {
    // No previous step to diff against: compare with itself, highlight nothing
    const prev = prevHeap ? prevHeap[ref] : heap[ref];
    objectsCol.appendChild(buildHeapObject(heap[ref], prev, groups.get(ref) ?? 0, nameOf));
  });
  positionObjects(order, parent);   // after insertion: it measures real widths

  if (!Object.keys(heap).length) {
    objectsCol.appendChild(el("div", "empty-note", "no objects yet"));
  }
  if (step.heap_truncated) {
    objectsCol.appendChild(el("div", "trunc-note", "too many objects — the rest are not drawn"));
  }

  const seen = new Set();
  Object.keys(heap).forEach((ref) => {
    (heap[ref].shares_memory_with || []).forEach((other) => {
      const key = [ref, other].sort().join(" ");
      if (seen.has(key) || !heap[other]) return;
      seen.add(key);
      state.memLinks.push({
        from: `heap-${ref}`,
        to: `heap-${other}`,
        color: `var(--mem-${(groups.get(ref) ?? 0) % 5})`,
      });
    });
  });
}

/* ------------------------------------------------------------ SVG wires */

function svgPath(d, cls, color) {
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", d);
  path.setAttribute("class", cls);
  if (cls === "wire") path.setAttribute("marker-end", "url(#arrowhead)");
  if (color) path.style.stroke = color;
  return path;
}

function drawWires() {
  const heap = $("heap");
  const svg = $("wires");
  svg.querySelectorAll("path.wire, path.memwire").forEach((p) => p.remove());

  const base = heap.getBoundingClientRect();
  const ox = heap.scrollLeft - base.left;
  const oy = heap.scrollTop - base.top;
  svg.setAttribute("width", heap.scrollWidth);
  svg.setAttribute("height", heap.scrollHeight);

  heap.querySelectorAll("[data-target]").forEach((slot) => {
    const target = document.getElementById(slot.dataset.target);
    if (!target) return;

    const a = slot.getBoundingClientRect();
    const b = target.getBoundingClientRect();

    // Wires land on the target's left edge, level with its header
    const x2 = b.left + ox - 6;
    const y2 = b.top + Math.min(18, b.height / 2) + oy;

    // A dot inside a box (list cell, instance row) has to leave downward;
    // leaving rightward would cut across its own box every time
    const inside = slot.closest(".obj") !== null;
    const x1 = inside ? a.left + a.width / 2 + ox : a.right + ox;
    const y1 = inside ? a.bottom + oy : a.top + a.height / 2 + oy;

    const reach = Math.max(28, Math.abs(x2 - x1) * 0.45, Math.abs(y2 - y1) * 0.22);
    const c1x = inside ? x1 : x1 + reach;
    const c1y = inside ? y1 + reach : y1;

    svg.appendChild(svgPath(
      `M ${x1} ${y1} C ${c1x} ${c1y}, ${x2 - reach} ${y2}, ${x2} ${y2}`, "wire"));
  });

  state.memLinks.forEach(({ from, to, color }) => {
    const a = document.getElementById(from);
    const b = document.getElementById(to);
    if (!a || !b) return;
    const ra = a.getBoundingClientRect();
    const rb = b.getBoundingClientRect();
    const x1 = ra.right + ox;
    const y1 = ra.top + ra.height / 2 + oy;
    const x2 = rb.right + ox;
    const y2 = rb.top + rb.height / 2 + oy;
    const bulge = Math.max(x1, x2) + 20;
    svg.appendChild(svgPath(`M ${x1} ${y1} C ${bulge} ${y1}, ${bulge} ${y2}, ${x2} ${y2}`, "memwire", color));
  });
}

/* ------------------------------------------------------------ code + line arrows */

/* The editor stays editable; there is no separate view mode. Line highlights
   are painted behind the textarea by hl-layer, which holds the same text in
   transparent ink so it wraps identically and the bands always line up. */

function paintCode(lines, marks = {}) {
  const gutter = $("edit-gutter");
  const layer = $("hl-layer");
  gutter.textContent = "";
  layer.textContent = "";

  lines.forEach((text, i) => {
    const lineNo = i + 1;
    const isJust = lineNo === marks.justLine;
    const isNext = lineNo === marks.nextLine;
    const isErr = lineNo === marks.errorLine;

    const row = el("div", "gutter-line");
    if (isJust) row.classList.add("is-just");
    if (isNext) row.classList.add("is-next");
    if (isErr) row.classList.add("is-err");

    // One arrow slot, same x on every line. When a line is both (a one-line
    // loop), pink wins: that is the line about to run
    const mark = el("span", "mark", isJust || isNext ? "➡" : "");
    if (isNext) mark.classList.add("is-next");
    else if (isJust) mark.classList.add("is-just");
    row.appendChild(mark);
    row.appendChild(el("span", "ln", String(lineNo)));
    gutter.appendChild(row);

    // The band needs the real text, or its height will not match when wrapping
    const band = el("div", "hl-line", text.length ? text : " ");
    if (lineNo === marks.justLine) band.classList.add("is-just");
    if (lineNo === marks.nextLine) band.classList.add("is-next");
    if (lineNo === marks.errorLine) band.classList.add("is-err");
    layer.appendChild(band);
  });

  syncScroll();
}

function syncScroll() {
  const code = $("code");
  $("hl-layer").scrollTop = code.scrollTop;
  $("hl-layer").scrollLeft = code.scrollLeft;
  $("edit-gutter").scrollTop = code.scrollTop;
}

function scrollToLine(lineNo) {
  if ($("wrap").checked) return;   // with wrap on, line height is not constant
  const code = $("code");
  const styles = getComputedStyle(code);
  const height = parseFloat(styles.lineHeight);
  if (!height) return;
  const top = (lineNo - 1) * height;
  if (top < code.scrollTop || top > code.scrollTop + code.clientHeight - height) {
    code.scrollTop = Math.max(0, top - code.clientHeight / 2);
    syncScroll();
  }
}

function renderCode(step, prevStep, trace) {
  const finished = step.event === "done" || step.event === "error";
  const marks = {
    errorLine: trace.error ? trace.error.line : null,
    nextLine: finished ? null : step.line,
    justLine: prevStep ? prevStep.line : (finished ? step.line : null),
  };
  paintCode(trace.source, marks);
  scrollToLine(marks.nextLine || marks.justLine || 1);
}

/* ------------------------------------------------------------ page chrome */

function stepDescription(step) {
  if (step.event === "explain") return step.title;
  if (step.event === "input") return "waiting for input";
  if (step.event === "done") return "Program terminated";
  if (step.event === "error") return `Error on line ${step.line}`;
  return `about to run line ${step.line}`;
}

/* --- Sub-steps have no state of their own; they borrow the parent's picture --- */

function frameStepAt(index) {
  const step = state.trace.steps[index];
  return step.event === "explain" ? state.trace.steps[step.parent] : step;
}

function previousFrameStep(index) {
  for (let i = index - 1; i >= 0; i -= 1) {
    if (state.trace.steps[i].event !== "explain") return state.trace.steps[i];
  }
  return null;
}

function renderExplain(step) {
  const panel = $("explain");
  panel.textContent = "";
  if (step.event !== "explain") {
    panel.classList.add("hidden");
    return;
  }
  panel.classList.remove("hidden");

  const head = el("div", "explain-head");
  head.appendChild(el("span", "explain-op", step.op));
  head.appendChild(el("span", "explain-title", step.title));
  panel.appendChild(head);

  if (step.note) panel.appendChild(el("div", "explain-note", step.note));
  if (!step.boxes.length) return;

  const boxes = el("div", "explain-boxes");
  step.boxes.forEach((item) => {
    const wrap = el("div", "ebox");
    wrap.appendChild(el("div", "ebox-label", item.label));
    const body = el("div", "ebox-body");
    body.appendChild((RENDERERS[item.snap.kind] || renderOpaque)(item.snap, undefined));
    wrap.appendChild(body);
    boxes.appendChild(wrap);
  });
  panel.appendChild(boxes);
}

function renderBanner(trace) {
  const banner = $("banner");
  banner.className = "banner";
  banner.textContent = "";

  if (trace.error) {
    banner.classList.add("error");
    const where = trace.error.line ? ` on line ${trace.error.line}` : "";
    banner.append(`Error${where}: `);
    banner.appendChild(el("b", null, trace.error.text));
    return;
  }
  if (trace.stopped === "max_steps") {
    banner.classList.add("warn");
    banner.textContent =
      `Stopped recording at the ${trace.limits.max_steps}-step limit — what you see is only the start of the run.`;
    return;
  }
  if (trace.stopped === "timeout") {
    banner.classList.add("warn");
    banner.textContent = `Stopped recording after the ${trace.limits.timeout}s time limit.`;
    return;
  }
  banner.classList.add("hidden");
}

function renderStdout(step, trace) {
  const out = $("stdout");
  const text = trace.stdout.slice(0, step.stdout_len);
  if (text) {
    out.classList.remove("empty");
    out.textContent = text;
  } else {
    out.classList.add("empty");
    out.textContent = "— no output yet —";
  }
}

function render() {
  const { trace, index } = state;
  if (!trace || !trace.steps.length) return;

  const step = trace.steps[index];
  const base = frameStepAt(index);          // the step that owns the picture
  const prevStep = previousFrameStep(base.step);

  renderCode(base, prevStep, trace);
  renderHeap(base, prevStep);
  renderStdout(step, trace);
  renderExplain(step);
  requestAnimationFrame(drawWires);

  $("step-now").textContent = index + 1;
  $("step-total").textContent = trace.steps.length;
  $("step-desc").textContent = stepDescription(step);
  $("slider").value = index;

  $("first").disabled = index === 0;
  $("prev").disabled = index === 0;
  $("next").disabled = index === trace.steps.length - 1;
  $("last").disabled = index === trace.steps.length - 1;
}

function goto(index) {
  if (!state.trace) return;
  const max = state.trace.steps.length - 1;
  state.index = Math.max(0, Math.min(max, index));
  render();
}

/* ------------------------------------------------------------ editor */

function syncGutter() {
  paintCode($("code").value.split("\n"));
}

// Once the user types, the highlight no longer matches the code: clear it
function invalidateHighlight() {
  syncGutter();
  state.inputs = [];   // different program, old answers no longer line up
  hideAsk();
  if (state.trace) $("status").textContent = "code changed — press Visualize Execution";
}

function applyEditorPrefs() {
  const size = $("font-size").value;
  const wrap = $("wrap").checked;
  document.documentElement.style.setProperty("--code-size", `${size}px`);
  document.documentElement.style.setProperty("--code-wrap", wrap ? "pre-wrap" : "pre");
  // With wrap on, source lines and visual lines diverge, so hide the numbers
  $("edit-gutter").classList.toggle("hidden", wrap);
  syncGutter();
  try {
    localStorage.setItem("npviz-editor", JSON.stringify({ size, wrap }));
  } catch (err) {
    /* storage disabled, ignore */
  }
}

function loadEditorPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem("npviz-editor") || "{}");
    if (saved.size) $("font-size").value = saved.size;
    if (saved.wrap) $("wrap").checked = true;
  } catch (err) {
    /* corrupt value, use the default */
  }
  applyEditorPrefs();
}

/* ------------------------------------------------------------ reset */

function startRun() {
  state.inputs = [];   // pressing Visualize starts over; past answers do not count
  visualize();
}

function resetTrace() {
  state.trace = null;
  state.index = 0;
  state.memLinks = [];
  state.inputs = [];
  hideAsk();

  $("code").value = "";
  syncGutter();
  resetLayout();

  syncGutter();
  $("banner").classList.add("hidden");
  $("explain").classList.add("hidden");
  $("heap-cols").classList.add("hidden");
  $("placeholder").classList.remove("hidden");
  $("frames").textContent = "";
  $("objects").textContent = "";
  drawWires();

  $("stdout").classList.add("empty");
  $("stdout").textContent = "— no output yet —";
  $("status").textContent = "";
  $("step-now").textContent = "–";
  $("step-total").textContent = "–";
  $("step-desc").textContent = "";
  $("slider").value = 0;
  $("slider").disabled = true;
  $("reset").disabled = true;
  ["first", "prev", "next", "last"].forEach((id) => { $(id).disabled = true; });
}

/* ------------------------------------------------------------ input() prompts

   The server keeps no session. When input() runs out of answers it stops and
   says so; we collect the answer into state.inputs and *re-run the whole
   program* with everything gathered so far.
   ------------------------------------------------------------------------- */

function showAsk(info) {
  $("ask-prompt").textContent = info.prompt || "input()";
  $("ask").classList.remove("hidden");
  $("ask-value").focus();
}

function hideAsk() {
  $("ask").classList.add("hidden");
  $("ask-value").value = "";
}

function sendInput() {
  const field = $("ask-value");
  state.inputs.push(field.value);
  field.value = "";
  visualize({ jumpToEnd: true });
}

/* ------------------------------------------------------------ running code */

async function visualize({ jumpToEnd = false } = {}) {
  const code = $("code").value;
  const status = $("status");
  if (!code.trim()) {
    status.textContent = "write some code first, or pick an example";
    return;
  }
  status.textContent = "running…";
  $("run").disabled = true;

  try {
    const res = await fetch("/api/trace", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, inputs: state.inputs }),
    });
    const data = await res.json();

    if (!res.ok) {
      status.textContent = data.error ? data.error.message : `HTTP ${res.status}`;
      return;
    }

    state.trace = data;
    // Waiting on input, or just answered: jump to the end to show where it stopped
    state.index = (data.awaiting_input || jumpToEnd)
      ? Math.max(0, data.steps.length - 1)
      : 0;

    const slider = $("slider");
    slider.max = Math.max(0, data.steps.length - 1);
    slider.value = state.index;
    slider.disabled = data.steps.length < 2;

    renderBanner(data);
    if (data.awaiting_input) showAsk(data.awaiting_input);
    else hideAsk();
    $("placeholder").classList.add("hidden");
    $("heap-cols").classList.remove("hidden");

    if (!data.steps.length) {
      $("explain").classList.add("hidden");
      renderCode({ line: data.error ? data.error.line : 0, event: "error" }, null, data);
      $("frames").textContent = "";
      $("objects").textContent = "";
      $("step-now").textContent = "–";
      $("step-total").textContent = "–";
      $("step-desc").textContent = "";
      ["first", "prev", "next", "last"].forEach((id) => { $(id).disabled = true; });
      status.textContent = "nothing to step through";
      return;
    }

    render();
    $("reset").disabled = false;
    status.textContent = data.awaiting_input
      ? "waiting for input"
      : `${data.steps.length} steps`;
  } catch (err) {
    status.textContent = `cannot reach the server: ${err}`;
  } finally {
    $("run").disabled = false;
  }
}

/* ------------------------------------------------------------ resizing panes

   Two splitters: the vertical one is the code column's right edge, the
   horizontal one is the boundary between the code pane and stdout.
   ------------------------------------------------------------------------- */

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function saveLayout() {
  try {
    localStorage.setItem("npviz-layout", JSON.stringify({
      left: document.documentElement.style.getPropertyValue("--left-w"),
      out: document.documentElement.style.getPropertyValue("--out-h"),
    }));
  } catch (err) {
    /* storage disabled, ignore */
  }
}

function resetLayout() {
  document.documentElement.style.removeProperty("--left-w");
  document.documentElement.style.removeProperty("--out-h");
  try {
    localStorage.removeItem("npviz-layout");
  } catch (err) {
    /* storage disabled, ignore */
  }
  drawWires();
}

function loadLayout() {
  try {
    const saved = JSON.parse(localStorage.getItem("npviz-layout") || "{}");
    if (saved.left) document.documentElement.style.setProperty("--left-w", saved.left);
    if (saved.out) document.documentElement.style.setProperty("--out-h", saved.out);
  } catch (err) {
    /* corrupt value, use the default */
  }
}

function initSplitters() {
  const layout = document.querySelector(".layout");
  const leftCol = document.querySelector(".col-left");

  function draggable(handle, move) {
    handle.addEventListener("pointerdown", (down) => {
      down.preventDefault();
      handle.setPointerCapture(down.pointerId);
      handle.classList.add("is-dragging");

      const onMove = (event) => {
        move(event);
        drawWires();   // the heap column changed width, wires must follow
      };
      const onUp = () => {
        handle.classList.remove("is-dragging");
        handle.removeEventListener("pointermove", onMove);
        handle.removeEventListener("pointerup", onUp);
        saveLayout();
        drawWires();
      };

      handle.addEventListener("pointermove", onMove);
      handle.addEventListener("pointerup", onUp);
    });
  }

  draggable($("split-v"), (event) => {
    const box = layout.getBoundingClientRect();
    const width = clamp(event.clientX - box.left, 280, box.width - 340);
    document.documentElement.style.setProperty("--left-w", `${Math.round(width)}px`);
  });

  draggable($("split-h"), (event) => {
    const box = leftCol.getBoundingClientRect();
    const height = clamp(box.bottom - event.clientY, 60, box.height - 220);
    document.documentElement.style.setProperty("--out-h", `${Math.round(height)}px`);
  });
}

/* ------------------------------------------------------------ theme

   Two states only, one click to flip. (A three-state cycle through "follow
   system" needed several clicks before anything looked different.) */

function currentTheme() {
  const chosen = document.documentElement.getAttribute("data-theme");
  if (chosen === "light" || chosen === "dark") return chosen;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  $("theme").textContent = theme === "dark" ? "☾" : "☀";
  $("theme").title = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
  try {
    localStorage.setItem("npviz-theme", theme);
  } catch (err) {
    /* private mode or storage disabled, ignore */
  }
}

function initTheme() {
  let saved = "";
  try {
    saved = localStorage.getItem("npviz-theme") || "";
  } catch (err) {
    saved = "";
  }
  // An older build may have stored "" (follow system): fall back to the system theme
  applyTheme(saved === "light" || saved === "dark" ? saved : currentTheme());

  $("theme").addEventListener("click", () => {
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
  });
}

/* ------------------------------------------------------------ startup */

function init() {

  const picker = $("examples");
  const optgroups = new Map();
  EXAMPLES.forEach((ex, i) => {
    if (!optgroups.has(ex.group)) {
      const group = document.createElement("optgroup");
      group.label = ex.group;
      picker.appendChild(group);
      optgroups.set(ex.group, group);
    }
    optgroups.get(ex.group).appendChild(new Option(ex.name, String(i)));
  });
  picker.addEventListener("change", () => {
    const ex = EXAMPLES[Number(picker.value)];
    if (!ex) return;
    $("code").value = ex.code;
    picker.value = "";
    syncGutter();
    startRun();
  });

  $("run").addEventListener("click", startRun);
  // A form so Enter in the text field submits the way browsers already do
  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    sendInput();
  });
  // Fallback for paths where implicit submission does not fire
  $("ask-value").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendInput();
    }
  });
  $("reset").addEventListener("click", resetTrace);
  $("code").addEventListener("input", invalidateHighlight);
  $("code").addEventListener("scroll", syncScroll);
  $("wrap").addEventListener("change", applyEditorPrefs);
  $("font-size").addEventListener("input", applyEditorPrefs);

  $("first").addEventListener("click", () => goto(0));
  $("prev").addEventListener("click", () => goto(state.index - 1));
  $("next").addEventListener("click", () => goto(state.index + 1));
  $("last").addEventListener("click", () => goto(state.trace.steps.length - 1));
  $("slider").addEventListener("input", (e) => goto(Number(e.target.value)));

  $("code").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      startRun();
    }
  });

  document.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "textarea" || tag === "input" || tag === "select") return;
    if (!state.trace) return;
    const moves = {
      ArrowLeft: () => goto(state.index - 1),
      ArrowRight: () => goto(state.index + 1),
      " ": () => goto(state.index + 1),
      Home: () => goto(0),
      End: () => goto(state.trace.steps.length - 1),
    };
    if (moves[e.key]) {
      e.preventDefault();
      moves[e.key]();
    }
  });

  // Wires are computed from live positions, so redraw whenever anything moves
  $("heap").addEventListener("scroll", drawWires);
  window.addEventListener("resize", drawWires);

  initTheme();
  loadEditorPrefs();
  loadLayout();
  initSplitters();
}

init();
