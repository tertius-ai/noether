# mzs

`mzs` is a pure-Rust sparse linear-algebra kernel over `F_2`. Its `Mzs` matrix
stores each row as sorted column indices. It provides multiplication, dense
interop, transpose, RREF, rank, solve, and nullspace.

Noether selects `mzs` behind sparse `Matrix<F_2>` multiplication once the
inputs cross the measured nonzero threshold. Small or unsupported cases stay
on the facade's canonical CSR implementation.

```rust
use mzs::Mzs;

let a = Mzs::from_triples(2, 3, &[(0, 0), (0, 2), (1, 1)]);
let kernel = a.nullspace();
assert_eq!(a.matmul(&kernel).nnz(), 0);
```

```bash
cargo test -p mzs --release
scripts/bench-pin.sh cargo bench -p mzs --bench matmul
```

The crate contains no unsafe code. Its property suite checks algebraic laws
and parity between sparse operations and dense/scalar references.
