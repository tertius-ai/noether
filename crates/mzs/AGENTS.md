# mzs guidance

Follow the repository and `crates/AGENTS.md` rules. `mzs` owns indexed-row
sparse `F_2` storage and exact sparse multiplication, elimination, solve, and
nullspace. Noether calls it for sparse binary products above the audited
threshold.

## Invariants

- Indexed rows are sorted and duplicate-free.
- Conversions between indexed rows and packed-dense forms are lossless.
- Sparse/dense routing is justified by pinned crossover measurements and
  always checked against scalar multiplication.
- Elimination identities (`A K = 0`, rank-nullity, solved `A X = B`) accompany
  randomized cases.

The crate currently contains no unsafe code. New SIMD must be isolated in a
named module with runtime detection and scalar differential tests.

## Commands

```bash
cargo test -p mzs --release
cargo clippy -p mzs --all-targets -- -D warnings
scripts/bench-pin.sh cargo bench -p mzs --bench matmul
TERTIUS_BENCH_CPUS=group scripts/bench-pin.sh \
  cargo run --release -p mzs --features parallel --example bench_vs_m4rst
cargo test -p mzs --features parallel --example bench_vs_m4rst
```

The comparison example uses m4rst only as a development dependency. Its
streaming checks compare complete packed rows without quadratic byte buffers.
Use `N_VALS`, `DENSITIES`, `WARMUP`, and `RUNS` for a bounded sweep; the README
documents the historical grid and measured limits.
