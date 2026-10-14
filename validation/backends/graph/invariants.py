# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Canonical comparable invariants for the graph-core parity harness.

The parity harness scores by cross-CAS *consensus*, which needs every
oracle to return a value that is `==`-comparable and equal exactly
when the mathematics is equal. BFS/DFS *visit order* is only
well-defined under a fixed neighbour-ordering convention (graph-core
ascending target id) — and NetworkX / igraph / Sage do not
all honour that convention. So this module canonicalises every
oracle's raw output to **order-independent** invariants that all
oracles must agree on:

* `component_count` — number of (weakly) connected components.
* `component_partition` — the partition of the vertex set into
  components, as a frozenset of frozensets (label-independent, so it
  does not matter how each oracle numbers its components).
* `bfs_depth` — for each requested source, the multiset of
  vertex→distance pairs (BFS layer = unweighted shortest-path
  distance). Visit *order* is dropped; the depth function is canonical.
* `reachable` — for each source, the frozenset of reached vertices.
* `path_distance` — for each `(s, t)` query, the shortest-path
  distance, or `None` when unreachable.
* `subgraph_*` — order / edge_count / component_count /
  component_partition of the induced subgraph on the requested subset.
  The full oracle row for `subgraph`;
  the subset, vmap, and label-carry rules are *the kernel's*, so each
  oracle takes the subset / vmap as given and reports its own induced
  subgraph + its own label-carry behaviour.
* `subgraph_carried_vertex_labels` /
  `subgraph_carried_edge_labels` — when integer labels are attached,
  the vertex-/edge-label sequence after the subgraph carry. Encoded
  as a frozenset of `(new_index, label)` for vertex labels (the contract's
  preservation guarantee: new vertex `i` has the label of old vertex
  `vmap[i]`) and a frozenset of `(survivor_position, label)` for edge
  labels (the surviving edges in original-edge-order).
* `transpose_*` — order / edge_count / component_count /
  component_partition of the transpose. Reported only for directed
  graphs (undirected transpose is the identity). Plus
  `transpose_carried_vertex_labels` / `transpose_carried_edge_labels`
  when labels are attached — these equal the originals (the contract's
  "transpose preserves labels" guarantee made comparable across
  oracles).

These order-independent parity targets include `subgraph`, `transpose`, and
`Labeling` as well as traversal and connectivity.
"""

from __future__ import annotations

from typing import Any


def canonical(raw: dict[str, Any]) -> frozenset:
    """Canonicalise one oracle's raw result to the comparable value.

    `raw` carries (all optional except `order`):

    * `order` — vertex count.
    * `component_labels` — list, label per vertex.
    * `bfs_depth` — `{source: [depth_or_None, ...]}`.
    * `path_distance` — `{"s-t": distance_or_None}`.

    Returns a **`frozenset` of `(name, value)` pairs** — a single
    hashable, `==`-comparable object so the parity harness can take a
    `Counter` consensus over it (the harness rejects non-hashable
    case values). Two oracles' results compare equal exactly when
    every order-independent invariant agrees.
    """
    items: list[tuple[str, Any]] = []
    n = int(raw["order"])
    items.append(("order", n))

    labels = raw.get("component_labels")
    if labels is not None:
        items.append(("component_count", len(set(labels))))
        # Partition as a frozenset of frozensets — independent of how
        # any particular oracle numbers its components.
        groups: dict[Any, list[int]] = {}
        for v, lab in enumerate(labels):
            groups.setdefault(lab, []).append(v)
        items.append(
            (
                "component_partition",
                frozenset(frozenset(g) for g in groups.values()),
            )
        )

    bfs_depth = raw.get("bfs_depth")
    if bfs_depth is not None:
        depth_inv = []
        reachable = []
        for src, depths in bfs_depth.items():
            src_i = int(src)
            depth_inv.append(
                (
                    src_i,
                    frozenset((v, d) for v, d in enumerate(depths) if d is not None),
                )
            )
            reachable.append(
                (
                    src_i,
                    frozenset(v for v, d in enumerate(depths) if d is not None),
                )
            )
        items.append(("bfs_depth", frozenset(depth_inv)))
        items.append(("reachable", frozenset(reachable)))

    path_distance = raw.get("path_distance")
    if path_distance is not None:
        items.append(("path_distance", frozenset(path_distance.items())))

    # ── `subgraph` cell ─────────────────────────────────────────────
    sub = raw.get("subgraph")
    if sub is not None:
        items.append(("subgraph_order", int(sub["order"])))
        items.append(("subgraph_edge_count", int(sub["edge_count"])))
        sub_labels = sub.get("component_labels")
        if sub_labels is not None:
            items.append(("subgraph_component_count", len(set(sub_labels))))
            sub_groups: dict[Any, list[int]] = {}
            for v, lab in enumerate(sub_labels):
                sub_groups.setdefault(lab, []).append(v)
            items.append(
                (
                    "subgraph_component_partition",
                    frozenset(frozenset(g) for g in sub_groups.values()),
                )
            )
        # Label preservation under subgraph. The vertex-label invariant
        # is `frozenset((new_id, label))`; the edge-label invariant is
        # `frozenset((survivor_position, label))` — i.e. the carried
        # sequence indexed by 0..edge_count(sub). Both are order-
        # independent within the cell (just the multiset), so they
        # match across oracles even when components are numbered
        # differently.
        sub_vlabels = sub.get("vertex_labels")
        if sub_vlabels is not None:
            items.append(
                (
                    "subgraph_carried_vertex_labels",
                    frozenset((i, int(lab)) for i, lab in enumerate(sub_vlabels)),
                )
            )
        sub_elabels = sub.get("edge_labels")
        if sub_elabels is not None:
            items.append(
                (
                    "subgraph_carried_edge_labels",
                    frozenset((i, int(lab)) for i, lab in enumerate(sub_elabels)),
                )
            )

    # ── `transpose` cell ────────────────────────────────────────────
    tr = raw.get("transpose")
    if tr is not None:
        items.append(("transpose_order", int(tr["order"])))
        items.append(("transpose_edge_count", int(tr["edge_count"])))
        tr_labels = tr.get("component_labels")
        if tr_labels is not None:
            items.append(("transpose_component_count", len(set(tr_labels))))
            tr_groups: dict[Any, list[int]] = {}
            for v, lab in enumerate(tr_labels):
                tr_groups.setdefault(lab, []).append(v)
            items.append(
                (
                    "transpose_component_partition",
                    frozenset(frozenset(g) for g in tr_groups.values()),
                )
            )
        tr_vlabels = tr.get("vertex_labels")
        if tr_vlabels is not None:
            items.append(
                (
                    "transpose_carried_vertex_labels",
                    frozenset((i, int(lab)) for i, lab in enumerate(tr_vlabels)),
                )
            )
        tr_elabels = tr.get("edge_labels")
        if tr_elabels is not None:
            items.append(
                (
                    "transpose_carried_edge_labels",
                    frozenset((i, int(lab)) for i, lab in enumerate(tr_elabels)),
                )
            )

    return frozenset(items)
