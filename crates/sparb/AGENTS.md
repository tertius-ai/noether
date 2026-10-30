# sparb guidance

Follow the repository and `crates/AGENTS.md` rules. `sparb` owns canonical CSR
over rigorous real and complex balls. Noether calls it for sparse matrices over
`RealBallField` and `ComplexBallField`.

## Invariants

- Exact zero entries may be dropped; an inexact ball containing zero must stay.
- Every result encloses the true value at the recorded precision.
- COO duplicates are summed before canonical CSR storage.
- Real and complex sparse operations are checked against reused-buffer FLINT
  dense references and hand-checkable enclosures.
- Parallel SpMV has the same enclosure contract as serial SpMV.

The crate itself contains no unsafe block. FFI ownership and primitive ball
arithmetic live in the named `flint-core` boundary modules. If sparse SIMD is
added, isolate and name its module here.

## Commands

```bash
cargo test -p sparb --release --features parallel
cargo clippy -p sparb --all-targets --features parallel -- -D warnings
scripts/bench-pin.sh cargo bench -p sparb --bench spmv --features parallel
scripts/bench-pin.sh cargo run --release -p sparb --example bench_compare
crates/sparb/scripts/bench.sh
```

The runner retains source, native-library, and CPU/cache provenance under
`bench-runs/sparb`. The README distinguishes its current real/complex-ball
comparison from the restored historical f64/MPFR/Arb measurements.
