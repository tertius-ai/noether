# graph-core

`graph-core` is Noether's safe-Rust graph kernel. It owns canonical CSR graph
storage, typed labels, deterministic traversal, connected components, shortest
paths, triangle counting, and validated rooted-tree algorithms.

`noether::combin::Graph` and `Tree` are curated re-exports of these identities.
Graph automorphisms compose with `perm-core` in the facade; spectral and
numeric linear algebra compose with `Matrix<R>` rather than entering this
storage crate.

Graphs may be directed or undirected, weighted or unweighted, and may contain
parallel edges and loops. CSR rows are sorted by target id, so serial BFS and
DFS orders are deterministic. Invalid vertices and negative Dijkstra weights
return `GraphError`; no path is `Ok(None)`.

```rust
use graph_core::{bfs, connected_components, graph_from_unweighted};

let graph = graph_from_unweighted(
    5,
    &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    false,
).unwrap();
assert_eq!(bfs(&graph, 0).unwrap().order.len(), 5);
assert_eq!(connected_components(&graph).count, 1);
```

```bash
cargo test -p graph-core --release
cargo test -p graph-core --release --features parallel
scripts/bench-pin.sh cargo bench -p noether --bench facade_kernels graph-core
```

The optional `parallel` feature supplies Rayon traversal/SSSP paths over the
same scalar correctness floor. The crate contains no unsafe code.
