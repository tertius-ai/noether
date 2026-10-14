# sparla guidance

Follow the repository and `crates/AGENTS.md` rules. `sparla` owns canonical
sparse exact matrices over `Z` and word-sized prime fields. Noether calls it
for matching `SparseMatrix<R>` storage and arithmetic.

## Invariants

- CSR rows are sorted, duplicate-free, and omit exact zeros.
- Integer entries use `flint-core::Integer`; algorithms do not narrow them.
- Prime-field constructors validate the modulus and reduce every entry.
- Normal forms and solve/kernel/image results carry reconstruction identities
  in tests; minimal cases are hand-checked before random sweeps.
- The `flint` feature is a validation/benchmark lane, not a production
  implementation switch.

The crate currently contains no unsafe code. FLINT unsafe ownership remains in
`flint-core`; raw `flint3-sys` use is development-only here.

## Commands

```bash
cargo test -p sparla --release
cargo test -p sparla --release --features flint --test parity_flint
cargo clippy -p sparla --all-targets -- -D warnings
scripts/bench-pin.sh cargo bench -p sparla --bench matmul
```
