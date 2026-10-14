# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Parity cases for topic U1 — graph storage and traversal (`graph-core`).

Scores the kernel's storage, traversal, connectivity, and shortest-path
surface against established graph oracles by consensus.

Oracles
-------
* **NetworkX** — a full oracle here and the Python speed baseline.
  Participates iff `networkx` imports.
* **igraph** — the C-backed correctness + speed oracle. Participates
  iff `python-igraph` imports.
* **Sage** — `sage.graphs`, the canonical U1 reference; driven as a
  subprocess. Participates iff the `sage` binary is locatable.
* **noether** — the system under test: the `graph-core` crate, reached
  through its `parity_oracle` example binary (see `_noether_graph`).

Magma also has native graphs and slots in the same way once a Magma
graph adapter exists on the licensed machine; the consensus model
tolerates its absence.

Per-function coverage
---------------------
Every accepted graph-core function appears in at least one oracle-compared
cell:

* `graph_from_edges` / `graph_from_unweighted` — exercised by every
  case (every corpus row builds the graph and reports its invariants).
* `csr_adjacency` — exercised structurally: `component_partition`
  and BFS depth are computed from the CSR, so a wrong CSR taints
  both. A direct row would be redundant.
* `subgraph` — every case carries a `subgraph_subset` and the
  scorecard scores the induced subgraph's order / edge_count /
  component_count / component_partition under the kernel's first-
  appearance vmap.
* `transpose` — every directed case scores the transpose's
  order / edge_count / component_count / component_partition. The
  comparison cell. (Undirected transpose is the identity, so the cell is
  vacuous there; not reported.)
* `bfs` / `dfs` — scored via BFS depth + reachable set (canonical,
  order-independent — visit order is not portable across oracles).
* `connected_components` — scored via component count + partition.
* `shortest_path` — scored via the per-(s,t) distance map.
* `edge_vertex_labels` (`Labeling`) — every case carries integer
  vertex labels and integer edge labels. The scorecard checks the
  `Labeling.subgraph` and `Labeling.transpose` *preservation
  guarantees* by computing the carried sequences on each oracle and
  comparing them: every oracle must agree that new vertex `i` of the
  subgraph has the label of old vertex `vmap[i]` and that transpose carries labels
  verbatim. This is the cross-oracle preservation check; in-process
  per-label round-trip is covered by `tests/labels.rs`.

Comparable form
---------------
BFS/DFS *visit order* is only well-defined under a fixed neighbour-
ordering convention (graph-core uses ascending target id), which
NetworkX / igraph / Sage do not all honour. So every oracle's output
is canonicalised (`backends.graph.invariants.canonical`) to the
**order-independent** invariants — component count and
partition, BFS depth (= unweighted distance), reachable set,
shortest-path distance, induced-subgraph + transpose structural
invariants, label-preservation invariants — and the scorecard compares
those.

Golden corpus
-------------
graph kind: undirected, directed, multigraph, edge-weighted.
order: 8, 256, 4096.
structure: path, cycle, complete, complete-bipartite, sparse random
(m ≈ 2n), disconnected (≥ 3 components), with self-loops.

Running
-------
    python validation/run_parity.py --topic U1

The `noether` backend builds `graph-core`'s `parity_oracle` example on
first use (or honours `GRAPH_CORE_ORACLE_BIN`). No FLINT is needed —
graph-core is pure Rust.
"""

from __future__ import annotations

from typing import Any

from validation.harness.parity import ParityCase
from validation.parity._noether_graph import query as noether_query

# ── corpus generators ───────────────────────────────────────────────
#
# Each generator returns the `parity_oracle` request schema: a dict
# with `n`, `directed`, `edges` ([s, t, w] triples), `sources` (BFS/DFS
# roots) and `paths` ((s, t) shortest-path queries).


def _spec(n: int, directed: bool, edges: list[tuple[int, int, int]]) -> dict[str, Any]:
    """Assemble a request spec; pick a spread of sources / path queries.

    Every case carries:

    * `sources` — BFS/DFS roots, the same canonical spread on every
      case so the harness exercises the start / middle / end of the
      vertex set.
    * `paths` — three shortest-path queries on the same span.
    * `subgraph_subset` — a deterministic ~half-the-vertex-set sample
      for the `subgraph` cell. Spans the ends of the index range so
      it cuts edges of every generated structure.
    * `vertex_labels` / `edge_labels` — synthetic integer labels (10v+1
      / 100e+1). The `Labeling` cell scores label preservation
      under `subgraph` and `transpose` against the public guarantee.
    """
    sources = sorted({0, n // 2, n - 1})
    paths = [(0, n - 1), (0, n // 2), (n // 2, n - 1)]
    # A deterministic, distinct, span-of-the-vertex-set subset. Keep
    # roughly half the vertices, taking every other id so the subset
    # touches both endpoints of the vertex range.
    subset = list(range(0, n, 2))
    # Synthetic integer labels — distinct per vertex/edge so a wrong
    # carry shows up as a different multiset.
    vertex_labels = [10 * v + 1 for v in range(n)]
    edge_labels = [100 * e + 1 for e in range(len(edges))]
    return {
        "n": n,
        "directed": directed,
        "edges": [list(e) for e in edges],
        "sources": sources,
        "paths": [list(p) for p in paths],
        "subgraph_subset": subset,
        "vertex_labels": vertex_labels,
        "edge_labels": edge_labels,
    }


def _path(n: int, directed: bool, weight: int = 1) -> dict[str, Any]:
    edges = [(i, i + 1, weight) for i in range(n - 1)]
    return _spec(n, directed, edges)


def _cycle(n: int, directed: bool, weight: int = 1) -> dict[str, Any]:
    edges = [(i, (i + 1) % n, weight) for i in range(n)]
    return _spec(n, directed, edges)


def _complete(n: int, directed: bool) -> dict[str, Any]:
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if directed or i < j:
                edges.append((i, j, 1))
    return _spec(n, directed, edges)


def _bipartite(a: int, b: int, directed: bool) -> dict[str, Any]:
    edges = [(i, a + j, 1) for i in range(a) for j in range(b)]
    return _spec(a + b, directed, edges)


def _disconnected(comps: int, each: int, directed: bool) -> dict[str, Any]:
    edges = []
    for c in range(comps):
        base = c * each
        for i in range(each - 1):
            edges.append((base + i, base + i + 1, 1))
    return _spec(comps * each, directed, edges)


def _self_loops(directed: bool) -> dict[str, Any]:
    # A triangle plus a self-loop on every vertex.
    edges = [(0, 1, 1), (1, 2, 1), (2, 0, 1), (0, 0, 1), (1, 1, 1), (2, 2, 1)]
    return _spec(3, directed, edges)


def _multigraph(directed: bool) -> dict[str, Any]:
    # A triangle with the 0-1 edge tripled.
    edges = [(0, 1, 1), (0, 1, 1), (0, 1, 1), (1, 2, 1), (2, 0, 1)]
    return _spec(3, directed, edges)


def _sparse_random(n: int, m: int, directed: bool, seed: int) -> dict[str, Any]:
    # SplitMix64 — deterministic, matches the Rust test generator's
    # spirit (the harness only needs a fixed reproducible graph).
    state = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

    def nxt() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    edges = []
    for _ in range(m):
        u = nxt() % n
        v = nxt() % n
        edges.append((u, v, 1))
    return _spec(n, directed, edges)


def _weighted(directed: bool) -> dict[str, Any]:
    # A weighted ladder: two routes of differing total weight, so the
    # Dijkstra path is non-trivial.
    n = 6
    edges = [
        (0, 1, 1),
        (1, 2, 1),
        (2, 5, 1),
        (0, 3, 4),
        (3, 4, 4),
        (4, 5, 4),
        (1, 4, 2),
    ]
    return _spec(n, directed, edges)


def _corpus() -> list[tuple[str, dict[str, Any]]]:
    """The full input matrix as (label, request-spec) pairs."""
    corpus: list[tuple[str, dict[str, Any]]] = []
    for directed in (False, True):
        tag = "directed" if directed else "undirected"
        # Structures across orders 8 / 256 / 4096.
        for n in (8, 256, 4096):
            corpus.append((f"path n={n} {tag}", _path(n, directed)))
            corpus.append((f"cycle n={n} {tag}", _cycle(n, directed)))
            corpus.append(
                (f"sparse-random n={n} m={2 * n} {tag}", _sparse_random(n, 2 * n, directed, seed=n))
            )
        # Complete + bipartite — keep the order modest (dense).
        for n in (8, 64):
            corpus.append((f"complete K_{n} {tag}", _complete(n, directed)))
        corpus.append((f"bipartite K_4,5 {tag}", _bipartite(4, 5, directed)))
        corpus.append((f"bipartite K_16,16 {tag}", _bipartite(16, 16, directed)))
        # Disconnected — at least 3 components, two sizes.
        corpus.append((f"disconnected 3x8 {tag}", _disconnected(3, 8, directed)))
        corpus.append((f"disconnected 5x32 {tag}", _disconnected(5, 32, directed)))
        # Multigraph + self-loops.
        corpus.append((f"multigraph triangle {tag}", _multigraph(directed)))
        corpus.append((f"self-loops triangle {tag}", _self_loops(directed)))
        # Edge-weighted.
        corpus.append((f"weighted ladder {tag}", _weighted(directed)))
    return corpus


# ── oracle runners ──────────────────────────────────────────────────


def _networkx_runner(spec: dict[str, Any]):
    from validation.backends.graph import networkx_oracle

    return networkx_oracle.evaluate(spec)


def _igraph_runner(spec: dict[str, Any]):
    from validation.backends.graph import igraph_oracle

    return igraph_oracle.evaluate(spec)


def _sage_runner(spec: dict[str, Any]):
    from validation.backends.graph import sage_oracle

    return sage_oracle.evaluate(spec)


def _noether_runner(spec: dict[str, Any]):
    """Run graph-core's `parity_oracle` and canonicalise its output."""
    from validation.backends.graph.invariants import canonical

    raw = noether_query(spec)
    # The oracle reports BFS depth inside `bfs[src]["depth"]`; lift it
    # into the flat shape `canonical` consumes.
    bfs_depth = {src: payload["depth"] for src, payload in raw.get("bfs", {}).items()}
    path_distance = {
        key: (None if val is None else val["distance"]) for key, val in raw.get("paths", {}).items()
    }
    flat: dict[str, Any] = {
        "order": raw["order"],
        "component_labels": raw["component_labels"],
        "bfs_depth": bfs_depth,
        "path_distance": path_distance,
    }
    # Forward the optional cells (`subgraph`, `transpose`) verbatim;
    # `canonical` extracts the comparable invariants from each.
    if "subgraph" in raw:
        flat["subgraph"] = raw["subgraph"]
    if "transpose" in raw:
        flat["transpose"] = raw["transpose"]
    return canonical(flat)


# ── case assembly ───────────────────────────────────────────────────
#
# One `ParityCase` per graph: the canonicalised invariant dict bundles
# component count + partition, BFS depth / reachability, shortest-path
# distance, induced-subgraph structural invariants, transpose
# structural invariants, and label-preservation invariants. A single
# consensus over it scores the whole accepted surface — `graph_from_edges` /
# `graph_from_unweighted` / `csr_adjacency` / `subgraph` / `transpose` /
# `bfs` / `dfs` / `connected_components` / `shortest_path` /
# `edge_vertex_labels` — for that input. The `function` field names
# every accepted function the scorecard covers.

# Every accepted `graph-core` function is listed verbatim in the
# `function` field so the harness's per-(topic, function) header
# records every covered op.
FUNCTIONS_COVERED = (
    "graph_from_edges/graph_from_unweighted/csr_adjacency/"
    "subgraph/transpose/bfs/dfs/connected_components/"
    "shortest_path/edge_vertex_labels"
)


def cases() -> list[ParityCase]:
    out: list[ParityCase] = []
    for label, spec in _corpus():
        out.append(
            ParityCase(
                topic="U1",
                function=FUNCTIONS_COVERED,
                label=label,
                runners={
                    "sage": lambda s=spec: _sage_runner(s),
                    "networkx": lambda s=spec: _networkx_runner(s),
                    "igraph": lambda s=spec: _igraph_runner(s),
                    "noether": lambda s=spec: _noether_runner(s),
                },
            )
        )
    return out
