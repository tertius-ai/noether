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

## Historical libm4rie comparison

The raw cells below preserve the 2026-05-19 Tertius
`crates/m4rste/docs/BENCHMARKS.md` report, retained at Tertius commit
`1c8f3868`. Each cell is **m4rste ms / libm4rie ms**, with the minimum
of five timed runs after two warmups. The report records bit-equal results
for each measured operation.

The recorded host was a Ryzen 9 7950X3D, 16 cores / 32 threads, 125 GB RAM,
Linux 6.8.0-100-generic, rustc 1.95.0, and the `powersave` governor.
References were the Debian/Ubuntu libm4rie and libm4ri 20200125-1 packages,
running serially. Per-run CPU affinity and the exact benchmarked revision
were not retained, and the report's cache description is inconsistent.
These are historical measurements, not current dispatcher evidence.

### GF(4), presentation x² + x + 1 (`0x7`)

| op     | N=64 m4rste / m4rie | N=128 m4rste / m4rie | N=256 m4rste / m4rie |
|--------|--------------------:|---------------------:|---------------------:|
| matmul |   0.017 / 0.032     |    0.036 / 0.109     |   0.111 / 0.425      |
| rank   |   0.030 / 0.025     |    0.104 / 0.081     |   0.383 / 0.300      |
| rref   |   0.042 / 0.031     |    0.119 / 0.098     |   0.406 / 0.366      |

### GF(16), presentation x⁴ + x + 1 (`0x13`)

| op     | N=64 m4rste / m4rie | N=128 m4rste / m4rie | N=256 m4rste / m4rie |
|--------|--------------------:|---------------------:|---------------------:|
| matmul |   0.053 / 0.043     |    0.112 / 0.151     |   0.356 / 0.594      |
| rank   |   0.042 / 0.040     |    0.166 / 0.130     |   0.767 / 0.491      |
| rref   |   0.057 / 0.046     |    0.240 / 0.155     |   1.164 / 0.600      |

### GF(256), AES presentation (`0x11B`)

| op     | N=64 m4rste / m4rie | N=128 m4rste / m4rie | N=256 m4rste / m4rie |
|--------|--------------------:|---------------------:|---------------------:|
| matmul |   0.158 / 0.331     |    0.338 / 0.640     |   1.066 / 1.767      |
| rank   |   0.043 / 0.311     |    0.182 / 0.504     |   0.838 / 1.241      |
| rref   |   0.050 / 0.314     |    0.225 / 0.542     |   1.126 / 1.405      |

The raw GF(16) order-64 multiplication cell favors libm4rie; it contradicts
the old report's blanket claim that m4rste won every multiplication cell.
The values above preserve the raw timings.

Run the current paired comparison from the repository root:

```bash
scripts/bench-pin.sh env OMP_NUM_THREADS=1 RUNS=5 WARMUP=2 \
  cargo run --locked --release -p m4rste \
  --features validate-libm4rie --example bench_vs_m4rie
M4RSTE_VALIDATE_LIBM4RIE=1 cargo test --release -p m4rste \
  --features validate-libm4rie --test parity_libm4rie -- --test-threads=1
```

The current harness reports medians and excludes input marshalling, unlike
the historical harness whose libm4rie path included element conversion.
It covers addition as well as multiplication, rank, and RREF over orders
64, 128, and 256, and fails on any parity disagreement. Set `M4RIE_PREFIX`
for a custom reference installation; its `lib/` must be on the loader path.
The oracle test runs with one test thread because the system reference's
field initialization uses shared state.
