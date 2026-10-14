# geom-core guidance

Follow the repository and `crates/AGENTS.md` rules. This crate owns Euclidean
point clouds, metrics, balls, exact predicate signs, and Delaunay/regular
triangulations. `noether::geometry` is its facade consumer; abstract topology
does not belong here.

## Correctness

- Orientation and in-sphere branches must be exact. Preserve the floating
  filter plus exact FLINT-rational fallback.
- Hand-check degenerate planar and spatial cases before randomized sweeps.
- Triangulations must reject rank-deficient input rather than guess a cell
  structure.
- Keep parity against `robust` and `spade`; they are test-only oracles.

The crate currently contains no unsafe code. If a SIMD predicate path needs
unsafe, isolate it in a named module, add runtime dispatch and scalar parity,
and update this file.

## Commands

```bash
cargo test -p geom-core --release
cargo clippy -p geom-core --all-targets -- -D warnings
scripts/bench-pin.sh cargo bench -p geom-core --bench kernel
scripts/bench-pin.sh cargo run --release -p geom-core --example delaunay_rate
```
