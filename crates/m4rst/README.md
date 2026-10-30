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
conveyed build. The dispatcher and elimination comparisons keep opaque libm4ri
matrices preloaded, exclude buffer marshalling, and bit-check results through
an independent copy-out path. In the pinned exact-output sweep on the repository
benchmark host's 96 MB-cache core, square multiplication is faster than libm4ri
at orders 256, 2,048, and 8,192; the advantage narrows from about 1.8× at order 256 to
about 1.3× at order 8,192. At order 192, the PLE rank path is about 5--8%
slower than libm4ri while the packed Gauss–Jordan RREF path is faster on that
same pinned core. Unsafe code is confined to the view, naive-matmul, and SIMD
modules named in `AGENTS.md`.

## Historical comparison results

These tables preserve the measurements from the Tertius
`crates/m4rst/docs/BENCHMARKS.md` report, present at commit `1c8f3868`.
They describe that implementation and host, rather than a fresh run of this
checkout. The report did not retain CPU affinity or an exact benchmarked
revision, so these numbers are not evidence for changing current dispatch
thresholds. New measurements must use the pinned runners below. Ratios are
recomputed from the displayed times where the original report disagreed.

The recorded host was an AMD Ryzen 9 7950X3D (16 cores / 32 threads,
125 GB RAM), Linux 6.8.0-100-generic, rustc 1.95.0, and the `powersave`
governor. The reference was libm4ri 20260122 built with GCC 13.3.0,
`--enable-thread-safe --enable-openmp`, and `-fopenmp -g -O2`. Rust used
the release profile with ThinLTO and one codegen unit. The report's L3
description is internally inconsistent; no per-run cache placement survives.

The report describes medians of five runs after one warmup, but the retained
dispatcher runner timed a single call and the raw thread runner used fewer
samples at large orders. The exact sampling for each reported cell cannot be
reconstructed. The serial tuning runner did use five runs after one warmup.
Inputs used fixed seeds `0xA` and `0xB`; result allocation was timed and input
conversion was excluded. The old dispatcher runner had no exact-output
preflight. The restored dispatcher runner uses the current checked harness.

### Public dispatcher with equal thread counts

Times are milliseconds for square matrices of order N. T is the thread count
requested from both Rayon and OpenMP. The ratio is m4rst time / libm4ri time;
values below one favor m4rst.

| N | T | m4rst ms | libm4ri ms | m4rst / libm4ri |
|---:|---:|---:|---:|---:|
| 32,768 | 1 | 6,435 | 7,323 | 0.88× |
| 32,768 | 4 | 2,975 | 2,922 | 1.02× |
| 32,768 | 8 | 3,134 | 3,977 | 0.79× |
| 32,768 | 16 | 2,375 | 5,829 | 0.41× |
| 65,536 | 1 | 44,316 | 50,298 | 0.88× |
| 65,536 | 4 | 20,416 | 21,100 | 0.97× |
| 65,536 | 8 | 18,342 | 27,160 | 0.68× |
| 65,536 | 16 | 16,611 | 36,958 | 0.45× |

### Separate sweep with independently selected thread counts

This is a separate measurement series; its times differ from the equal-thread
table. The report selected T=16 for m4rst and T=4 for libm4ri throughout.
All times below are milliseconds, including the order-131,072 row originally
reported in seconds.

| N | m4rst ms (T=16) | libm4ri ms (T=4) | m4rst / libm4ri |
|---:|---:|---:|---:|
| 4,096 | 16.5 | 12.9 | 1.28× |
| 8,192 | 24.5 | 57.6 | 0.43× |
| 16,384 | 319 | 428 | 0.75× |
| 32,768 | 2,366 | 3,046 | 0.78× |
| 65,536 | 21,679 | 21,891 | 0.99× |
| 131,072 | 133,700 | 147,100 | 0.91× |

### Serial kernel tuning sweep

The m4rst column selects the best measured kernel and tuning parameters at
each N, rather than timing the public dispatcher. Both libm4ri algorithms ran
with `OMP_NUM_THREADS=1`. Times are milliseconds; ratios below are recomputed
from the displayed times against libm4ri's `mzd_mul` column.

| N | m4rst best ms | libm4ri M4RM ms | libm4ri mul ms | m4rst / libm4ri mul |
|---:|---:|---:|---:|---:|
| 256 | 0.03 | 0.03 | 0.03 | 1.00× |
| 512 | 0.10 | 0.11 | 0.12 | 0.83× |
| 1,024 | 0.42 | 0.57 | 0.57 | 0.74× |
| 2,048 | 2.30 | 2.72 | 2.71 | 0.85× |
| 4,096 | 15.74 | 19.22 | 19.21 | 0.82× |
| 8,192 | 117.51 | 134.41 | 150.13 | 0.78× |
| 16,384 | 884.81 | 1,012.35 | 1,053.98 | 0.84× |

The report also retains these paired kernel measurements. Times are
milliseconds. These are separate tuning runs, so their absolute times should
not be mixed with the preceding table.

| N | single-table M4RM | eight-table M4RM | speedup from eight tables |
|---:|---:|---:|---:|
| 1,024 | 0.64 | 0.37 | 1.73× |
| 2,048 | 3.49 | 2.44 | 1.43× |
| 4,096 | 24.93 | 14.76 | 1.69× |
| 8,192 | 189.27 | 114.20 | 1.66× |
| 16,384 | 3,048.36 | 1,148.39 | 2.65× |

| N | eight-table M4RM | with M-block size 2,048 |
|---:|---:|---:|
| 4,096 | 14.76 | 17.13 |
| 8,192 | 114.20 | 130.72 |
| 16,384 | 1,148.39 | 975.09 |

Another order-16,384 pair in the report measured Strassen with eight-table
leaves at 1,161 ms and M-blocked eight-table M4RM at 885 ms. This does not
establish a universal Strassen threshold; the current dispatcher also depends
on shape, thread count, and cache regime.

## Reproducing the comparisons

Run these commands from the repository root. Install libm4ri headers and the
shared library, or set `M4RI_PREFIX` to an installation with `include/` and
`lib/`. Thread comparisons require a thread-safe, OpenMP-enabled libm4ri build;
setting `OMP_NUM_THREADS` cannot add parallelism to a serial library. The
runners add a supplied prefix's `lib/` to `LD_LIBRARY_PATH` on Linux.

```bash
# Public Mzd::matmul dispatcher: orders 32768 and 65536, T=1,4,8,16.
crates/m4rst/scripts/bench_fair_dispatch.sh

# Raw serial/parallel eight-table kernels: orders 4096 through 131072,
# T=1,2,4,8,16. This measures different paths from the public dispatcher.
crates/m4rst/scripts/bench_threads.sh
.venv/bin/python crates/m4rst/scripts/summarize_threads.py \
  bench-runs/m4rst/threads.csv

# Small dispatcher sweep, with explicit sampling controls.
NS="256 2048" THREADS="1 2" WARMUP=1 RUNS=3 \
  crates/m4rst/scripts/bench_fair_dispatch.sh

# Serial k, blocking, and Strassen tuning, with single-core affinity.
scripts/bench-pin.sh env OMP_NUM_THREADS=1 \
  cargo run --locked -p m4rst --release \
  --features validate-libm4ri --example bench_systematic
```

The sweep scripts build once, then use a fresh process for each thread count.
Every measurement goes through `scripts/bench-pin.sh`. They default to
`TERTIUS_BENCH_CPUS=group`, keeping the sweep within one last-level-cache group;
that group may contain fewer physical cores than the historical host-wide
sweep requested. The wrapper records the selected CPUs, cache size, and
frequency policy. Its `TERTIUS_BENCH_CACHE`, `TERTIUS_BENCH_CPUS`, and
`TERTIUS_BENCH_CPU` overrides remain available. Requested thread counts can
exceed the affinity budget; interpret such cells as oversubscribed.

Dispatcher output and provenance go to `bench-runs/m4rst/fair-dispatch.log`
(`LOG` overrides it). The raw thread sweep writes `bench-runs/m4rst/threads.csv`
(`OUT_CSV`) and a companion `.log` (`LOG`) with provenance. Paths are relative
to the repository root unless absolute. Logs include the source revision,
dirty-tree status, toolchain, host, seeds, and library prefix. Set
`M4RI_REVISION` to record the reference library's exact source/build identity;
the scripts cannot infer its build flags from a shared object.

`NS` and `THREADS` are whitespace-separated dispatcher lists; `N_VALS` is a
comma-separated raw-kernel size list and `THREAD_LIST` its whitespace-separated
thread list. Dispatcher defaults are one warmup and five measured runs, with
`WARMUP` and `RUNS` overrides. The raw-kernel runner uses one warmup and three
runs through order 16,384, one warmup and two runs through 65,536, then one
timed run without warmup at larger orders. Both sweep runners compare complete
products against libm4ri before timing; the raw-kernel runner also checks its
serial path when that baseline is measured. Its CSV reports milliseconds; the
dispatcher log reports microseconds. The summarizer derives tables from the
CSV without assuming a particular host or available thread counts.
