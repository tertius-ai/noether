# pwvla

`pwvla` is a dense packed linear-algebra kernel for small prime-power residue
rings. `Mat<T: Z2n>` uses native wrapping arithmetic for `Z/2^n`; `MatZpn`
packs small `Z/p^n` elements into words with Barrett reduction.

Noether selects `Mat<T>` for supported word-sized power-of-two residue rings.
Odd prime powers remain on FLINT unless a fresh pinned benchmark establishes a
winning `MatZpn` regime. The public identity is always `Matrix<ResidueRing>`.

```rust
use pwvla::zpn::MatZpn;

let a = MatZpn::from_fn(5, 2, 3, 3, |row, col| ((row + col) as u64) % 25);
let identity = MatZpn::identity(5, 2, 3);
assert_eq!(a.matmul(&identity), a);
```

The kernel provides multiplication and BLAS-like operations plus PLUQ, rank,
determinant, RREF, solve, inverse, and nullspace in its supported packed
regime. `matmul` runtime-dispatches to AVX2 or NEON where supported and keeps
`matmul_scalar` as its differential-test floor. SIMD code is present in
portable builds and cannot execute before a CPU feature check; Rayon is an
optional feature.

```bash
cargo test -p pwvla --release --features parallel
scripts/bench-pin.sh cargo bench -p pwvla --bench matmul --features parallel
scripts/bench-pin.sh cargo run --release -p pwvla \
    --features flint --example bench_z2n_flint
```

Unsafe code is confined to the architecture-specific modules named in
`AGENTS.md`.
