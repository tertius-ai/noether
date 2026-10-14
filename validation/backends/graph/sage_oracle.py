# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Sage oracle for the graph-core parity harness (topic U1).

`sage.graphs` is the canonical topic-U1 reference. Sage
is driven as a subprocess via `validation.backends.sage._run.run_sage`
— never `import sage.*` — so a Sage installed only in a conda env is
still found.

`available()` reports whether the `sage` binary is locatable;
`evaluate()` builds a small Sage script, runs it, and parses the JSON
it prints. The script uses Sage's `connected_components`,
`shortest_path_lengths`, and `distance` — for a digraph it builds the
underlying undirected graph for weak components, matching graph-core's
`connected_components`.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from validation.backends.graph.invariants import canonical
from validation.backends.sage._run import (
    SageError,
    sage_available,
    sage_binary,
)

# A large sparse graph's spec JSON easily exceeds the OS `ARG_MAX`, so
# the graph oracle cannot pass the script via `sage -c`. It writes a
# `.sage` script to a temp file (spec embedded as a file-read, not an
# argv string) and runs `sage <file>` instead.
_SAGE_TIMEOUT = 300


def available() -> bool:
    """True iff the `sage` binary can be located on this host."""
    return sage_available()


_SCRIPT = """
import json
with open("{spec_path}") as _f:
    spec = json.load(_f)
n = spec["n"]
directed = spec["directed"]
edges = spec["edges"]

# Build the graph. Sage Graph/DiGraph with multiedges + loops to span
# the graph-core input matrix.
if directed:
    G = DiGraph(n, multiedges=True, loops=True)
else:
    G = Graph(n, multiedges=True, loops=True)
G.add_vertices(range(n))
for s, t, w in edges:
    G.add_edge(s, t, w)

# Weak components: for a digraph, components of the underlying graph.
base = G.to_undirected() if directed else G
labels = [0] * n
for label, comp in enumerate(base.connected_components(sort=True)):
    for v in comp:
        labels[v] = label

# BFS depth — unweighted distance (by_weight=False).
bfs_depth = {{}}
for src in spec.get("sources", []):
    dl = G.shortest_path_lengths(src, by_weight=False)
    bfs_depth[str(src)] = [dl.get(v, None) for v in range(n)]

# Shortest-path distance — weighted (by_weight=True).
path_distance = {{}}
for s, t in spec.get("paths", []):
    try:
        d = G.distance(s, t, by_weight=True)
    except Exception:
        d = None
    if d == Infinity or d == +Infinity:
        d = None
    path_distance["%d-%d" % (s, t)] = (None if d is None else int(d))

# § 1 `subgraph` cell — built on the kernel's distinct, first-appearance
# vmap so Sage's induced graph is renumbered to the same indices.
subset = spec.get("subgraph_subset")
vlabels = spec.get("vertex_labels")
elabels = spec.get("edge_labels")

out = {{
    "order": n,
    "component_labels": labels,
    "bfs_depth": bfs_depth,
    "path_distance": path_distance,
}}

if subset is not None:
    seen = set()
    vmap = []
    for v in subset:
        if v not in seen:
            seen.add(v)
            vmap.append(int(v))
    keep = set(vmap)
    new_id = {{old: i for i, old in enumerate(vmap)}}
    sub_edges = []
    for idx, (s, t, w) in enumerate(edges):
        if s in keep and t in keep:
            sub_edges.append((new_id[s], new_id[t], int(w), idx))
    if directed:
        H = DiGraph(len(vmap), multiedges=True, loops=True)
    else:
        H = Graph(len(vmap), multiedges=True, loops=True)
    H.add_vertices(range(len(vmap)))
    for ns, nt, w, _ in sub_edges:
        H.add_edge(ns, nt, w)
    H_base = H.to_undirected() if directed else H
    sub_labels = [0] * len(vmap)
    for lbl, comp in enumerate(H_base.connected_components(sort=True)):
        for v in comp:
            sub_labels[v] = lbl
    sub_out = {{
        "order": len(vmap),
        "edge_count": len(sub_edges),
        "component_labels": sub_labels,
    }}
    if vlabels is not None:
        sub_out["vertex_labels"] = [int(vlabels[old]) for old in vmap]
    if elabels is not None:
        sub_out["edge_labels"] = [int(elabels[idx]) for _, _, _, idx in sub_edges]
    out["subgraph"] = sub_out

if directed:
    # Sage's directed reverse.
    H = DiGraph(n, multiedges=True, loops=True)
    H.add_vertices(range(n))
    for s, t, w in edges:
        H.add_edge(t, s, w)
    H_base = H.to_undirected()
    tr_labels = [0] * n
    for lbl, comp in enumerate(H_base.connected_components(sort=True)):
        for v in comp:
            tr_labels[v] = lbl
    tr_out = {{
        "order": n,
        "edge_count": len(edges),
        "component_labels": tr_labels,
    }}
    if vlabels is not None:
        tr_out["vertex_labels"] = [int(v) for v in vlabels]
    if elabels is not None:
        tr_out["edge_labels"] = [int(e) for e in elabels]
    out["transpose"] = tr_out

print(json.dumps(out))
"""


def evaluate(spec: dict[str, Any]) -> dict[str, Any]:
    """Compute the canonical comparable invariants of `spec` via Sage."""
    binary = sage_binary()
    if binary is None:
        raise SageError("sage not found (checked PATH, NOETHER_SAGE_BIN, conda envs)")
    # Embed the spec as a temp JSON file and the script as a temp
    # `.sage` file — `sage <file>`, not `sage -c`, so a large graph's
    # JSON never hits the OS argv length limit.
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = Path(tmp) / "spec.json"
        script_path = Path(tmp) / "oracle.sage"
        spec_path.write_text(json.dumps(spec))
        script_path.write_text(_SCRIPT.format(spec_path=spec_path))
        try:
            result = subprocess.run(
                [binary, str(script_path)],
                capture_output=True,
                text=True,
                timeout=_SAGE_TIMEOUT,
            )
        except subprocess.TimeoutExpired as e:
            raise SageError(f"sage timed out after {_SAGE_TIMEOUT}s") from e
    if result.returncode != 0:
        raise SageError(
            f"sage exited {result.returncode}:\n"
            f"--- stderr ---\n{result.stderr}\n"
            f"--- stdout ---\n{result.stdout}"
        )
    # Sage may print banners; the JSON is the last non-empty line.
    line = [ln for ln in result.stdout.splitlines() if ln.strip()][-1]
    raw = json.loads(line)
    return canonical(raw)
