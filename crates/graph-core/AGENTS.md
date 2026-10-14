# graph-core guidance

Follow the repository and `crates/AGENTS.md` rules. This crate owns graph CSR
storage, labels, traversal, shortest paths, triangle counts, and tree
algorithms. `noether::combin` curates the same `Graph` and `Tree` identities.

## Invariants

- CSR rows are canonical and sorted by target id.
- Serial traversal order is deterministic.
- Invalid vertices and negative Dijkstra weights return typed errors; missing
  paths are `Ok(None)`.
- Parallel algorithms retain the serial mathematical result even when a
  predecessor witness is permitted to differ.
- Algorithms must remain deep-safe on long paths and trees.

The crate currently contains no unsafe code. New SIMD follows the parent
runtime-dispatch, scalar-fallback, differential-test, and benchmark rules.

## Commands

```bash
cargo test -p graph-core --release
cargo test -p graph-core --release --features parallel
cargo clippy -p graph-core --all-targets --features parallel -- -D warnings
scripts/bench-pin.sh cargo bench -p noether --bench facade_kernels graph-core
```
