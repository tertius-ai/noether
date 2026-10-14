# m4rste

`m4rste` is a pure-Rust dense linear-algebra kernel over `GF(2^e)` for
`1 <= e <= 16`. `Gf2e` records the irreducible polynomial presentation and
`Mzed` stores elements bit-sliced over an `m4rst::Mzd` backing matrix.

Noether selects this kernel for dense arithmetic over binary extension fields
of degree 2 through 16. Field invariants outside the kernel's tuned surface
convert once to the facade's FLINT representation; generic field elimination
remains the differential floor.

```rust
use m4rste::{Gf2e, Mzed};

let field = Gf2e::new(0x11b); // GF(256), AES polynomial
let a = Mzed::from_u16_buffer(field, 2, 2, &[1, 2, 3, 4]);
let identity = Mzed::identity(field, 2);
assert_eq!(Mzed::matmul(&a, &identity).to_u16_buffer(), a.to_u16_buffer());
```

Bit-sliced multiplication reduces to `m4rst` products. Elimination uses
Newton-John blocking; the SIMD module runtime-dispatches GFNI/AVX2/NEON paths
over a scalar implementation and tests them differentially.

```bash
cargo test -p m4rste --release
scripts/bench-pin.sh cargo bench -p m4rste --bench m4rste
scripts/bench-pin.sh cargo run -p m4rste --release \
  --features validate-libm4rie --example bench_vs_m4rie
```

The optional libm4rie parity feature is validation-only and never enters a
conveyed build. Its comparison retains opaque preloaded libm4rie inputs,
excludes buffer conversion, and checks every result through a separate
flat-buffer oracle. On the repository benchmark host's pinned 96 MB-cache
core, multiplication over GF(4) wins from order 128, GF(16) is about 1.4--1.7×
faster throughout orders 64--256, and GF(256) is about 2× faster throughout
that sweep. GF(256) rank and RREF are also faster throughout orders 64--256;
small-field elimination is still 1.5--3× slower because it expands rows into
scalar elements while libm4rie eliminates in its packed representation. Unsafe
code is confined to `src/simd.rs`.
