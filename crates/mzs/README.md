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

## Historical sparse/dense comparison

These tables preserve the Tertius `crates/mzs/docs/MZS_VS_M4RST.md`
report, retained at Tertius commit `1c8f3868`. Each cell compared the same
square binary matrices using serial mzs, parallel mzs, and the public m4rst
dispatcher with its parallel feature enabled. The original harness used
ChaCha8 fixtures with seeds `0xA` and `0xB`, a single timed run without
warmup, and a streaming comparison of sparse and packed-dense products.

The report records a Ryzen 9 7950X3D (16 cores / 32 threads, AVX2) and
128 GiB RAM on a non-quiesced development host. It does not retain per-run
affinity, an exact benchmarked revision, or explicit Rayon thread counts.
These results illustrate the measured crossover; they do not establish
current dispatch thresholds or repeatable speedup bounds.

### Parallel sparse versus parallel dense

The winner and speedup in each cell are as recorded in the original report.
A dash denotes an unmeasured cell.

| Density × N | N=256 | N=1024 | N=4096 | N=16384 | N=65536 | N=131072 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **0.1%** | m4rst 13× | **mzs-par 2.7×** | **mzs-par 29×** | **mzs-par 36×** | **mzs-par 85×** | **mzs-par 117×** |
| **0.5%** | — | **mzs-par 1.4×** | **mzs-par 7.8×** | **mzs-par 2.2×** | **mzs-par 2.6×** | **mzs-par 4.3×** |
| **1%** | m4rst 3.2× | **mzs-par 1.7×** | **mzs-par 2.2×** | m4rst 1.9× | — | — |
| **5%** | m4rst 3.8× | m4rst 5.0× | m4rst 5.9× | — | — | — |
| **10%** | m4rst 9.2× | m4rst 13× | — | — | — | — |
| **50%** | m4rst 52× | — | — | — | — | — |

### Recorded wall times

`mzs-serial` is `Mzs::matmul`, `mzs-par` is `Mzs::matmul_parallel`,
and `m4rst-par` is `Mzd::matmul` with parallel dispatch available.
The serial order-131,072, density-0.5% cell was omitted because of runtime.

| N | density | mzs-serial | mzs-par | m4rst-par |
|---:|---:|---:|---:|---:|
| 256 | 0.1% | 0.00 ms | 1.34 ms | 0.10 ms |
| 256 | 1% | 0.02 ms | 0.16 ms | 0.05 ms |
| 256 | 5% | 0.39 ms | 0.23 ms | 0.06 ms |
| 256 | 10% | 1.61 ms | 0.46 ms | 0.05 ms |
| 256 | 50% | 22.22 ms | 3.13 ms | 0.06 ms |
| 1024 | 0.1% | 0.05 ms | 0.17 ms | 0.46 ms |
| 1024 | 0.5% | 0.31 ms | 0.32 ms | 0.46 ms |
| 1024 | 1% | 1.00 ms | 0.30 ms | 0.51 ms |
| 1024 | 5% | 33.22 ms | 2.57 ms | 0.51 ms |
| 1024 | 10% | 96.47 ms | 6.90 ms | 0.55 ms |
| 4096 | 0.1% | 1.03 ms | 0.51 ms | 14.94 ms |
| 4096 | 0.5% | 15.35 ms | 1.96 ms | 15.35 ms |
| 4096 | 1% | 69.82 ms | 6.43 ms | 14.21 ms |
| 4096 | 5% | 1.68 s | 95.05 ms | 16.01 ms |
| 16384 | 0.1% | 42.49 ms | 4.23 ms | 153.19 ms |
| 16384 | 0.5% | 1.40 s | 69.60 ms | 149.76 ms |
| 16384 | 1% | 6.20 s | 294.07 ms | 154.16 ms |
| 65536 | 0.1% | 3.65 s | 165.85 ms | 14.18 s |
| 65536 | 0.5% | 104.05 s | 5.54 s | 14.52 s |
| 131072 | 0.1% | 30.34 s | 1.55 s | 181.74 s |
| 131072 | 0.5% | — | 42.88 s | 184.01 s |

### Recorded parallel speedup within mzs

These ratios were recorded with more timing precision than the rounded
wall-time table; they are retained as historical values.

| N | density | mzs-par speedup |
|---:|---:|---:|
| 1024 | 5% | 12.9× |
| 1024 | 10% | 14.0× |
| 4096 | 0.5% | 7.8× |
| 4096 | 1% | 10.9× |
| 4096 | 5% | 17.7× |
| 16384 | 0.5% | 20.1× |
| 16384 | 1% | 21.1× |
| 65536 | 0.1% | 22.0× |
| 65536 | 0.5% | 18.8× |
| 131072 | 0.1% | 19.6× |

### Running the comparison

The comparison is an example in this crate; m4rst is a development
dependency, so the production dependency graph stays unchanged. Build with
`parallel` to enable both parallel paths. Generate and check matrices
directly in sparse and packed storage, avoiding quadratic byte buffers.
Every timed sparse result is checked against a complete dense product
outside the timing region, including padding bits.

```bash
# Full historical size/density grid, up to order 131072.
TERTIUS_BENCH_CPUS=group scripts/bench-pin.sh \
  cargo run --locked --release -p mzs --features parallel \
  --example bench_vs_m4rst

# Bounded grid, including empty, thin, and non-word-aligned matrices.
TERTIUS_BENCH_CPUS=group scripts/bench-pin.sh \
  env N_VALS=0,1,65,256 DENSITIES=0,0.001,0.1,1 WARMUP=1 RUNS=3 \
  cargo run --locked --release -p mzs --features parallel \
  --example bench_vs_m4rst
```

Capture both output streams to retain the affinity wrapper's CPU/cache
provenance. The default is the original single-run grid; `RUNS` and
`WARMUP` set sampling, with medians reported. `N_VALS` and `DENSITIES`
are comma-separated lists; an override forms their Cartesian product,
defaulting the other axis to orders 256/1024/4096 or densities
0.001/0.005/0.01. `SKIP_SERIAL=1` skips serial timing throughout;
by default it is skipped only at order at least 131072 and density at
least 0.005. Seeds remain explicit in the output; the current workspace
RNG version is not a promise of identical historical random streams.
