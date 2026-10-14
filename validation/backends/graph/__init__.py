# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Graph oracle backends for the `graph-core` parity harness (topic U1).

Three independent graph oracles:

* **NetworkX** — a full oracle here; the de-facto reference graph
  library (Sage's graphs partly wrap it) and the "what users run
  today" speed baseline. Python-importable; participates iff
  `networkx` imports.
* **igraph** — the C-backed speed gate; Python-importable via the
  `python-igraph` package; participates iff `igraph` imports.
* **Sage** — `sage.graphs`, the canonical U1 reference; driven as a
  subprocess via `validation.harness.cas`.

Every oracle returns the same canonical comparable form as the
`noether` backend's `parity_oracle` (see
`backends.graph.invariants.canonical`): component count, BFS/DFS
visit orders under the ascending-target-id neighbour convention, and
shortest-path distances. An oracle simply does not appear in the
scorecard on a host where its library is absent — the consensus model
tolerates that.
"""
