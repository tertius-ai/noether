# m4rste guidance

Follow the repository and `crates/AGENTS.md` rules. `m4rste` owns dense
`GF(2^e)` matrices for `1 <= e <= 16`; Noether calls it for binary extension
fields and `m4rst` owns its bit-plane multiplication substrate.

## Kernel invariants

- `Gf2e` validates the irreducible-polynomial presentation.
- Packed entries use exactly `e` bits and padding remains canonical.
- Slice/cling, Newton-John elimination, PLUQ, solve, and SIMD paths are checked
  against scalar field arithmetic.
- Architecture-specific code must not execute without runtime feature
  detection.

Unsafe code is permitted only in `src/simd.rs`. Keep target-feature functions
small, preserve the scalar fallback, and extend the bit-identical differential
tests and pinned benchmark for every vector path.

## Commands

```bash
cargo test -p m4rste --release
cargo test -p m4rste --release --features validate-libm4rie
cargo clippy -p m4rste --all-targets -- -D warnings
scripts/bench-pin.sh cargo bench -p m4rste --bench m4rste
```

libm4rie/libm4ri are validation oracles only and must never enter a conveyed
dependency closure.
