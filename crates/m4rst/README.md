# m4rst

`m4rst` is a pure-Rust dense linear-algebra kernel over `F_2`. It owns the
bit-packed `Mzd` matrix and multiplication, transpose, XOR, PLUQ, echelon/RREF,
rank, solve, nullspace, and inverse operations.

Noether selects it behind dense `Matrix<R>` arithmetic for `Z/2Z` and prime
fields of characteristic two. `m4rste` also uses it as the bit-plane
multiplication substrate for binary extension fields. The facade retains its
generic scalar implementation as a differential correctness floor.

```rust
use m4rst::Mzd;

let a = Mzd::random(256, 256, 0xA);
let identity = Mzd::identity(256);
assert_eq!(Mzd::matmul(&a, &identity).as_words(), a.as_words());

let kernel = a.nullspace();
assert_eq!(Mzd::matmul(&a, &kernel).rank(), 0);
```

The multiplication dispatcher selects naive, M4RM, blocked, Strassen, and
parallel leaves by shape and thread count. Allocation-sensitive paths expose
`*_into` forms for reusable destinations. Large branch and determinism tests
exercise the production thresholds in normal/stress runs. The fused M4RM/PLUQ
row combine runtime-dispatches to explicit AVX2 or NEON over a scalar fallback;
portable builds retain every path.

```bash
cargo test -p m4rst --release --features parallel
scripts/bench-pin.sh cargo bench -p m4rst --bench matmul --features parallel
scripts/bench-pin.sh cargo run -p m4rst --release \
  --features validate-libm4ri --example bench_fair_dispatch
scripts/bench-pin.sh cargo run -p m4rst --release \
  --features validate-libm4ri --example bench_elimination
```

The optional libm4ri validation feature is an oracle only and never enters a
conveyed build. Its comparison keeps opaque libm4ri matrices preloaded, excludes
buffer marshalling, and bit-checks every result through an independent copy-out
path. In the current pinned exact-output sweep on the repository benchmark
host's 96 MB-cache core, square multiplication is faster than libm4ri at orders
256, 2,048, and 8,192; the advantage narrows from about 1.8× at order 256 to
about 1.3× at order 8,192. At order 192, the PLE rank path is about 5--8%
slower than libm4ri while the packed Gauss–Jordan RREF path is faster on that
same pinned core. Unsafe code is confined to the view, naive-matmul, and SIMD
modules named in `AGENTS.md`.
