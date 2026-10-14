# poly-core

`poly-core` is Noether's low-level polynomial and fixed-truncation series
substrate. It supplies packed monomials, deterministic
monomial interning through an open-addressed exact table, structure-of-arrays
arena polynomials, reusable heap/hash accumulators, finite truncation regions,
and generic sparse arithmetic. Additive cached monomial fingerprints make
product lookup allocation-free when the exact product is already interned;
exact exponent comparison remains authoritative. Dense bounded exponent boxes
dispatch privately to a Kronecker-indexed coefficient workspace; sparse shapes
retain hash accumulation. Both lanes work over every coefficient adapter and
reject truncated products before coefficient arithmetic. Public mathematical
contexts and result semantics remain in `noether::commalg`.

The `f2` module specializes this substrate to coefficient-free sparse
polynomials and one-bit-per-monomial fixed-region series over `F_2`. Sparse
addition is support symmetric difference, packed-series addition is XOR, and
Frobenius squaring doubles exponents without constructing cross terms.

The default `parallel` feature adds an adaptive dense-product lane. It splits
the left terms into deterministic worker chunks, accumulates into private
Kronecker arrays, and merges those arrays in chunk order. Dispatch requires at
least 65,536 coefficient products and caps all worker-private arrays together
at 32 MiB. Sparse output, large coefficient slots, a one-worker pool, and
larger expected workspaces retain the serial correctness floor. Noether calls
this lane only after its sticky native FLINT multiplication has declined the
operation.

Run the exact/property suite and the current pinned regime benchmarks with:

```bash
cargo test -p poly-core --release
TERTIUS_BENCH_CPUS=group scripts/bench-pin.sh \
  cargo bench -p poly-core --bench polynomial
```

The narrow `CoefficientRing` contract is for validated hot-loop adapters. Its
elements are thread-safe values so a kernel may move deterministic private
work units across workers without changing the mathematical adapter. It does
not duplicate Noether's public algebraic hierarchy.
