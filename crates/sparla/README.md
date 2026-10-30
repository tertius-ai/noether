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

## Historical FLINT comparison

The table preserves the Tertius `crates/sparla/docs/BENCHMARKS.md`
report, retained at Tertius commit `1c8f3868`. It measures square sparse
multiplication over the prime field with p=65537 against dense
`nmod_mat_mul`, in release mode, with one timed call per cell and a
single thread. Times are milliseconds. The report did not record a
complete host/toolchain fingerprint, CPU affinity, or exact benchmarked
revision, so the values do not establish current dispatch thresholds.

| N | density | sparla ms | FLINT ms | ratio |
|---|---|---:|---:|---:|
| 1000 | 0.1% | 0.038 | 73.091 | **1937× sparla** |
| 1000 | 1% | 0.896 | 74.257 | **83× sparla** |
| 1000 | 5% | 16.320 | 73.066 | **4.5× sparla** |
| 500 | 0.1% | 0.007 | 10.283 | **1579× sparla** |
| 500 | 1% | 0.117 | 10.294 | **88× sparla** |
| 500 | 5% | 2.344 | 10.770 | **4.6× sparla** |
| 500 | 25% | 10.054 | 10.758 | **1.07× sparla** (tie) |
| 200 | 1% | 0.012 | 0.730 | **59× sparla** |
| 200 | 25% | 1.082 | 0.716 | 0.66× (FLINT wins) |
| 100 | 25% | 0.173 | 0.111 | 0.64× (FLINT wins) |
| 50 | 100% | 0.075 | 0.015 | 0.21× (FLINT wins) |

The ratio column preserves the old report's values, which used timing
precision not retained in its rounded millisecond columns. LCG fixtures
used seeds `0xC0FF_EE42` and `0xBAAD_F00D`. Density is the requested
number of sampled entries divided by N², not the final nonzero fraction:
coordinates are sampled with replacement and duplicates are summed.
In particular, a requested 100% density produced about 63% distinct
coordinates. Every measured sparse product was compared exactly with
FLINT.

Run the retained harness from the repository root:

```bash
scripts/bench-pin.sh cargo run --locked --release -p sparla \
  --features flint --example matmul_bench_flint
scripts/bench-pin.sh cargo bench -p sparla --bench matmul
```

The paired example reports both the dense kernel time and total dense
conversion/multiplication/copy-out time, and aborts on a differing product.
Use the emitted actual nonzero counts when interpreting density.
`FLINT_PREFIX` selects an installation; its shared library must be on
the loader path. Criterion's density and size sweeps provide statistical
regression measurements separately from this single-run comparison.
