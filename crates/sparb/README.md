# sparb

`sparb` owns canonical CSR matrices over rigorous real and complex FLINT balls.
It provides COO ingestion, structural operations, add/subtract/scale,
matrix-vector and matrix-matrix products, rigorous norms, and spectral
enclosures.

Noether selects it behind `SparseMatrix<RealBallField>` and
`SparseMatrix<ComplexBallField>`. A ball containing zero remains stored because
dropping it would lose uncertainty. Every arithmetic result encloses the true
value at the matrix's fixed working precision.

```rust
use sparb::{Arb, CsrArb};

let a = CsrArb::from_coo_f64(
    2,
    2,
    80,
    &[(0, 0, 2.0, 0.01), (1, 1, 3.0, 0.01)],
);
let result = a.spmv(&[Arb::from_f64(1.0), Arb::from_f64(2.0)]);
assert!(result[0].radius(80) > 0.0);
```

Safe ownership and scalar operations come from `flint-core`; this crate owns
the sparse structure and algorithms. The optional `parallel` feature adds
Rayon SpMV over the same serial enclosure contract.

```bash
cargo test -p sparb --release --features parallel
scripts/bench-pin.sh cargo bench -p sparb --bench spmv --features parallel
scripts/bench-pin.sh cargo run --release -p sparb --example bench_compare
```

The shipped crate contains no unsafe block. Raw FLINT matrices are used only
as development parity and benchmark references.

## Historical ball-arithmetic comparisons

These tables preserve the 2026-05-16T10:55:09Z Tertius
`crates/sparb/docs/BENCHMARKS.md` report, retained at Tertius commit
`1c8f3868`. The recorded host was a Ryzen 9 7950X3D with 32 hardware
threads, Linux 6.8.0-100-generic, rustc 1.95.0, FLINT 3.5.0,
GMP 6.3.0, and MPFR 4.2.1. The fixture seed was `0xB0FF`.
Per-run CPU affinity, explicit parallel thread counts, and the exact
benchmarked revision were not retained. These numbers are historical
measurements, not current performance guarantees.

The SpMV rows report mean microseconds over 50 repetitions after one
warmup, with fewer repetitions for the dense reference. The old tracks
were a plain f64 sparse floor, a hand-written MPFR sparse floor without
ball radii, serial and parallel sparb, and dense FLINT ball arithmetic.
The f64/MPFR tracks do not provide the same enclosure contract as sparb.

| track | n | density | nnz | µs/SpMV |
|-------|---|---------|-----|--------:|
| f64 sparse | 1024 | 0.01 | 10485 | 6.3 |
| MPFR @ 53 | 1024 | 0.01 | 10485 | 365.3 |
| sparb @ 53 | 1024 | 0.01 | 10485 | 182.0 |
| MPFR @ 128 | 1024 | 0.01 | 10485 | 407.7 |
| sparb @ 128 | 1024 | 0.01 | 10485 | 174.0 |
| sparb @ 53 (parallel) | 1024 | 0.01 | 10485 | 129.6 |
| sparb @ 128 (parallel) | 1024 | 0.01 | 10485 | 85.1 |
| FLINT arb_mat @ 53 dense | 1024 | 0.01 | 10485 | 3593.1 |
| FLINT arb_mat @ 128 dense | 1024 | 0.01 | 10485 | 3531.5 |
| f64 sparse | 1024 | 0.1 | 104857 | 38.9 |
| MPFR @ 53 | 1024 | 0.1 | 104857 | 2506.0 |
| sparb @ 53 | 1024 | 0.1 | 104857 | 1376.0 |
| MPFR @ 128 | 1024 | 0.1 | 104857 | 2826.9 |
| sparb @ 128 | 1024 | 0.1 | 104857 | 1425.3 |
| sparb @ 53 (parallel) | 1024 | 0.1 | 104857 | 317.9 |
| sparb @ 128 (parallel) | 1024 | 0.1 | 104857 | 302.6 |
| FLINT arb_mat @ 53 dense | 1024 | 0.1 | 104857 | 3968.9 |
| FLINT arb_mat @ 128 dense | 1024 | 0.1 | 104857 | 4060.1 |
| f64 sparse | 4096 | 0.005 | 83886 | 46.9 |
| MPFR @ 53 | 4096 | 0.005 | 83886 | 2967.5 |
| sparb @ 53 | 4096 | 0.005 | 83886 | 1820.4 |
| MPFR @ 128 | 4096 | 0.005 | 83886 | 3304.8 |
| sparb @ 128 | 4096 | 0.005 | 83886 | 1880.6 |
| sparb @ 53 (parallel) | 4096 | 0.005 | 83886 | 367.8 |
| sparb @ 128 (parallel) | 4096 | 0.005 | 83886 | 311.7 |

The multiplication rows below report microseconds per operation. Precision
is in bits. The column `nnz(A)` records the original fixture count.

| track | n | density | nnz(A) | µs/matmul |
|-------|---|---------|-------:|----------:|
| sparb matmul @ 53 | 128 | 0.05 | 819 | 125.4 |
| sparb matmul @ 128 | 128 | 0.05 | 819 | 115.2 |
| FLINT arb_mat_mul @ 53 dense | 128 | 0.05 | 819 | 1835.5 |
| FLINT arb_mat_mul @ 128 dense | 128 | 0.05 | 819 | 1885.2 |
| sparb matmul @ 53 | 256 | 0.05 | 3276 | 1140.6 |
| sparb matmul @ 128 | 256 | 0.05 | 3276 | 1059.6 |
| FLINT arb_mat_mul @ 53 dense | 256 | 0.05 | 3276 | 9785.5 |
| FLINT arb_mat_mul @ 128 dense | 256 | 0.05 | 3276 | 9638.8 |
| sparb matmul @ 53 | 256 | 0.1 | 6553 | 4214.1 |
| sparb matmul @ 128 | 256 | 0.1 | 6553 | 3976.1 |
| FLINT arb_mat_mul @ 53 dense | 256 | 0.1 | 6553 | 10887.7 |
| FLINT arb_mat_mul @ 128 dense | 256 | 0.1 | 6553 | 10215.0 |

### Running the current comparison

The restored runner builds the existing `bench_compare` example and
captures its output together with host, source, FLINT, and affinity
provenance under `bench-runs/sparb/comparison.log`. It uses
`scripts/bench-pin.sh`; `LOG` overrides the destination.

```bash
crates/sparb/scripts/bench.sh
scripts/bench-pin.sh cargo bench -p sparb --bench spmv --features parallel
```

Set `FLINT_PREFIX` for a custom installation, `FLINT_REVISION` to
record its source/build identity, and the affinity wrapper's
`TERTIUS_BENCH_*` variables to choose the CPU/cache placement.
The runner otherwise uses the repository-managed FLINT prefix.

The current example compares serial real and complex ball SpMV and
multiplication at 53/128-bit precision, with the matching FLINT dense
operation on the same entries. It checks enclosure overlap before timing.
Its fixture grid and arithmetic differ from the old report: it does not
regenerate the f64, MPFR, or parallel SpMV rows. Parallel SpMV remains
covered by the Criterion suite. New results must be labeled with the
current harness and must not replace historical rows without their
measurement context.
