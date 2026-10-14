# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""igraph oracle for the graph-core parity harness (topic U1).

igraph is the C-backed speed reference and an independent correctness oracle
alongside NetworkX and Sage.

`available()` reports whether `python-igraph` is importable;
`evaluate()` computes the comparable invariants for one graph spec
using igraph's `connected_components` / `distances`.
"""

from __future__ import annotations

from typing import Any

from validation.backends.graph.invariants import canonical


def available() -> bool:
    """True iff python-igraph is importable on this host."""
    try:
        import igraph  # noqa: F401
    except ImportError:
        return False
    return True


def _build(spec: dict[str, Any]):
    import igraph as ig

    g = ig.Graph(n=spec["n"], directed=spec["directed"])
    edges = [(s, t) for s, t, _ in spec["edges"]]
    weights = [w for _, _, w in spec["edges"]]
    g.add_edges(edges)
    if edges:
        g.es["weight"] = weights
    return g


def evaluate(spec: dict[str, Any]) -> dict[str, Any]:
    """Compute the canonical comparable invariants of `spec` via igraph."""
    g = _build(spec)
    n = spec["n"]

    # Component membership — weak connectivity for digraphs.
    mode = "weak" if spec["directed"] else "strong"
    membership = g.connected_components(mode=mode).membership
    labels = list(membership)

    # BFS depth — unweighted distances (all weights treated as 1).
    bfs_depth: dict[int, list] = {}
    for src in spec.get("sources", []):
        # `distances` with no weights = hop count; inf = unreachable.
        row = g.distances(source=src)[0]
        bfs_depth[src] = [(None if d == float("inf") else int(d)) for d in row]

    # Shortest-path distance — weighted distances.
    path_distance: dict[str, Any] = {}
    weights = g.es["weight"] if g.ecount() else None
    for s, t in spec.get("paths", []):
        key = f"{s}-{t}"
        d = g.distances(source=s, target=t, weights=weights)[0][0]
        path_distance[key] = None if d == float("inf") else int(d)

    raw: dict[str, Any] = {
        "order": n,
        "component_labels": labels,
        "bfs_depth": bfs_depth,
        "path_distance": path_distance,
    }

    subset = spec.get("subgraph_subset")
    vlabels = spec.get("vertex_labels")
    elabels = spec.get("edge_labels")
    if subset is not None:
        raw["subgraph"] = _igraph_subgraph(spec, subset, vlabels, elabels)
    if spec["directed"]:
        raw["transpose"] = _igraph_transpose(spec, vlabels, elabels)

    return canonical(raw)


def _igraph_subgraph(spec, subset, vlabels, elabels):
    """igraph's induced subgraph + label-carry under the kernel's
    semantics (distinct, first-appearance vmap)."""
    import igraph as ig

    seen = set()
    vmap: list[int] = []
    for v in subset:
        if v not in seen:
            seen.add(v)
            vmap.append(v)
    keep = set(vmap)
    new_id = {old: i for i, old in enumerate(vmap)}

    sub_edges = []
    for idx, (s, t, w) in enumerate(spec["edges"]):
        if s in keep and t in keep:
            sub_edges.append((new_id[s], new_id[t], w, idx))

    sub = ig.Graph(n=len(vmap), directed=spec["directed"])
    sub.add_edges([(s, t) for s, t, _, _ in sub_edges])

    mode = "weak" if spec["directed"] else "strong"
    sub_labels = list(sub.connected_components(mode=mode).membership) if len(vmap) else []

    out: dict[str, Any] = {
        "order": len(vmap),
        "edge_count": len(sub_edges),
        "component_labels": sub_labels,
    }
    if vlabels is not None:
        out["vertex_labels"] = [int(vlabels[old]) for old in vmap]
    if elabels is not None:
        out["edge_labels"] = [int(elabels[idx]) for _, _, _, idx in sub_edges]
    return out


def _igraph_transpose(spec, vlabels, elabels):
    """igraph's directed reverse + carried labels."""
    import igraph as ig

    n = spec["n"]
    h = ig.Graph(n=n, directed=True)
    h.add_edges([(t, s) for s, t, _ in spec["edges"]])
    labels = list(h.connected_components(mode="weak").membership) if n else []
    out: dict[str, Any] = {
        "order": n,
        "edge_count": len(spec["edges"]),
        "component_labels": labels,
    }
    if vlabels is not None:
        out["vertex_labels"] = [int(v) for v in vlabels]
    if elabels is not None:
        out["edge_labels"] = [int(e) for e in elabels]
    return out
