# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""NetworkX oracle for the graph-core parity harness (topic U1).

NetworkX is a full oracle here — graph theory is its home turf and it
is the de-facto reference graph library (Sage's graphs partly wrap
it). It is also the Python speed baseline.

`available()` reports whether NetworkX can be imported; `evaluate()`
computes the comparable invariants for one graph spec. Only standard
NetworkX algorithms are used: `connected_components` /
`weakly_connected_components`, `single_source_shortest_path_length`
(unweighted BFS distance), and `dijkstra_path_length`.
"""

from __future__ import annotations

from typing import Any

from validation.backends.graph.invariants import canonical


def available() -> bool:
    """True iff NetworkX is importable on this host."""
    try:
        import networkx  # noqa: F401
    except ImportError:
        return False
    return True


def _build(spec: dict[str, Any]):
    import networkx as nx

    g = nx.MultiDiGraph() if spec["directed"] else nx.MultiGraph()
    g.add_nodes_from(range(spec["n"]))
    for s, t, w in spec["edges"]:
        g.add_edge(s, t, weight=w)
    return g


def evaluate(spec: dict[str, Any]) -> dict[str, Any]:
    """Compute the canonical comparable invariants of `spec` via NetworkX."""
    import networkx as nx

    g = _build(spec)
    n = spec["n"]

    # Component labels — weakly connected for digraphs, matching
    # graph-core's `connected_components`.
    if spec["directed"]:
        comps = nx.weakly_connected_components(g)
    else:
        comps = nx.connected_components(g)
    labels = [0] * n
    for label, comp in enumerate(comps):
        for v in comp:
            labels[v] = label

    # BFS depth — unweighted single-source shortest path length.
    bfs_depth: dict[int, list] = {}
    for src in spec.get("sources", []):
        dist = nx.single_source_shortest_path_length(g, src)
        bfs_depth[src] = [dist.get(v) for v in range(n)]

    # Shortest-path distance — Dijkstra over the `weight` attribute.
    path_distance: dict[str, Any] = {}
    for s, t in spec.get("paths", []):
        key = f"{s}-{t}"
        try:
            d = nx.dijkstra_path_length(g, s, t, weight="weight")
            path_distance[key] = int(d)
        except nx.NetworkXNoPath:
            path_distance[key] = None

    raw: dict[str, Any] = {
        "order": n,
        "component_labels": labels,
        "bfs_depth": bfs_depth,
        "path_distance": path_distance,
    }

    # § 1 `subgraph` cell.
    subset = spec.get("subgraph_subset")
    vlabels = spec.get("vertex_labels")
    elabels = spec.get("edge_labels")
    if subset is not None:
        raw["subgraph"] = _networkx_subgraph(g, spec, subset, vlabels, elabels)

    # § 1 `transpose` cell — directed only.
    if spec["directed"]:
        raw["transpose"] = _networkx_transpose(g, spec, vlabels, elabels)

    return canonical(raw)


def _networkx_subgraph(g, spec, subset, vlabels, elabels):
    """NetworkX's induced subgraph + carried labels, in the harness's
    output shape."""
    import networkx as nx

    # Mirror the kernel's `subgraph` semantics: distinct vertices in
    # first-appearance order, renumbered 0..k. NetworkX's
    # `g.subgraph(nodes)` is a *view* on the original ids; we still
    # have to recompute structural invariants on the induced graph in
    # the kernel's renumbering for the cross-oracle compare to land.
    seen = set()
    vmap = []
    for v in subset:
        if v not in seen:
            seen.add(v)
            vmap.append(v)
    keep = set(vmap)
    new_id = {old: i for i, old in enumerate(vmap)}

    # Walk the original edges in order so we preserve the harness's
    # edge-index semantics (a multi-edge between the same endpoints
    # keeps its insertion-order rank).
    sub_edges = []  # (new_s, new_t, weight, orig_edge_idx)
    for idx, (s, t, w) in enumerate(spec["edges"]):
        if s in keep and t in keep:
            sub_edges.append((new_id[s], new_id[t], w, idx))

    if spec["directed"]:
        sub = nx.MultiDiGraph()
    else:
        sub = nx.MultiGraph()
    sub.add_nodes_from(range(len(vmap)))
    for ns, nt, w, _ in sub_edges:
        sub.add_edge(ns, nt, weight=w)

    if spec["directed"]:
        sub_comps_iter = nx.weakly_connected_components(sub)
    else:
        sub_comps_iter = nx.connected_components(sub)
    sub_labels = [0] * len(vmap)
    for label, comp in enumerate(sub_comps_iter):
        for v in comp:
            sub_labels[v] = label

    out: dict[str, Any] = {
        "order": len(vmap),
        "edge_count": len(sub_edges),
        "component_labels": sub_labels,
    }
    if vlabels is not None:
        # NetworkX preserves node attributes through a subgraph view —
        # so the carried sequence here is exactly the public contract's
        # "labels preserved under subgraph" guarantee.
        out["vertex_labels"] = [int(vlabels[old]) for old in vmap]
    if elabels is not None:
        out["edge_labels"] = [int(elabels[idx]) for _, _, _, idx in sub_edges]
    return out


def _networkx_transpose(g, spec, vlabels, elabels):
    """NetworkX's directed reverse + carried labels."""
    import networkx as nx

    n = spec["n"]
    h = nx.MultiDiGraph()
    h.add_nodes_from(range(n))
    for s, t, w in spec["edges"]:
        h.add_edge(t, s, weight=w)
    comps = nx.weakly_connected_components(h)
    labels = [0] * n
    for lbl, comp in enumerate(comps):
        for v in comp:
            labels[v] = lbl
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
