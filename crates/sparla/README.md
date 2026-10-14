# sparla

`sparla` is Noether's safe-Rust sparse exact linear-algebra kernel over
integers and word-sized prime fields. `CsrZ` and `CsrFp` provide canonical COO
ingestion, transpose, multiplication, rank, determinant, kernel, image, and
solve; integer matrices also provide Hermite and Smith forms.

Noether selects these storage types behind `SparseMatrix<IntegerRing>` and
sparse prime-field matrices. Generic canonical CSR arithmetic remains the
floor for operations or contexts outside the specialized regime.
Simplicial Betti numbers and integral `homology_invariants` keep their
canonical boundary operators in `CsrZ`: rational rank is integral rank, while
exact integer torsion comes from the sparse Smith factors. The lean integral
result does not materialize or retain a dense boundary matrix.

```rust
use sparla::{CsrFp, CsrZ, Integer};

let integer = CsrZ::from_coo(
    2,
    2,
    &[(0, 0, Integer::from(2)), (1, 1, Integer::from(3))],
);
assert_eq!(integer.det(), Integer::from(6));

let field = CsrFp::from_coo(2, 2, &[(0, 0, 1), (1, 1, 1)], 7);
assert_eq!(field.rref().rank, 2);
```

Integer cells use `flint-core` ownership; sparse storage and algorithms remain
safe Rust. The `flint` feature gates slower dense parity and comparison tools,
not the production scalar dependency.

```bash
cargo test -p sparla --release
cargo test -p sparla --release --features flint --test parity_flint
scripts/bench-pin.sh cargo bench -p sparla --bench matmul
scripts/bench-pin.sh .venv/bin/python validation/bench/sage_homology.py \
  --sage /path/to/sage
```
