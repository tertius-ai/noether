# groebner-kernel guidance

Follow the repository and `crates/AGENTS.md` rules. This crate owns fixed
Gröbner implementation kernels. Noether owns public ideal semantics, strategy
selection, verification evidence, and backend-independent normalization.

All optimized answers are differentially checked against small generic cases.
Signature pruning must fall back or continue unpruned when final exact
verification fails.

`ahash` tables are membership sets and lookup indices, never ordering
authorities. Canonical bases, transformations, module outputs and evidence are
stable across process hash seeds and admitted Rayon thread counts; keep
`tests/determinism.rs` green when adding a hash-backed work structure.

## Commands

```bash
cargo test -p groebner-kernel --release
cargo clippy -p groebner-kernel --all-targets -- -D warnings
scripts/bench-pin.sh cargo bench -p groebner-kernel --bench groebner
```
