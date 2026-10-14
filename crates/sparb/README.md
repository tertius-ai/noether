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
