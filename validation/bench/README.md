# Python boundary benchmarks

Mathematical and dispatch microbenchmarks live with their Rust owner under
`crates/*/benches` or `crates/noether/benches`. This directory contains paired
end-to-end measurements through the Python boundary and a few deliberately
large, non-CI workload drivers.

## Measurement contract

Run every benchmark through the CPU-affinity wrapper:

```bash
scripts/bench-pin.sh <benchmark command>
```

The wrapper chooses a deterministic last-level-cache group and prints the
placement. Python drivers also print `_bench_cpu.cpu_provenance()`, so an
unpinned result identifies itself. A performance result without that
provenance is not evidence.

Paired drivers construct equivalent inputs, time the same mathematical
operation, normalize the outputs, and reject differing mathematical answers.
Construction is included when it is part of the public workflow; shared
fixture generation and post-run admission checks are excluded and described
by the driver. Warm operation time and cold process time are distinct metrics.
Timeouts and unavailable backends are structured results, not wins.

The drivers emit JSON Lines containing revision, runtime, input hash, timing,
memory and semantic-admission data. Generated measurements are local artifacts
and are not committed.

## Native Rust benchmarks

The principal facade and kernel regression suites are:

```bash
scripts/bench-pin.sh cargo bench -p poly-core --bench polynomial
scripts/bench-pin.sh cargo bench -p groebner-kernel --bench groebner
scripts/bench-pin.sh cargo bench -p sparla --bench rref_fp
scripts/bench-pin.sh cargo bench -p noether --bench facade_kernels
scripts/bench-pin.sh cargo bench -p noether --bench arithmetic_boundaries
scripts/bench-pin.sh cargo bench -p noether --bench polynomial_arithmetic
scripts/bench-pin.sh cargo bench -p noether --bench power_series
scripts/bench-pin.sh cargo bench -p noether --bench quotient_arithmetic
scripts/bench-pin.sh cargo bench -p noether --bench homology_playbooks
scripts/bench-pin.sh cargo bench -p noether --bench padic_sparse_smith
scripts/bench-pin.sh cargo bench -p noether --bench finite_group_cohomology
```

`facade_kernels` includes integral-line symmetry and exact minimum-realization
workloads alongside the retained geometry, graph, lattice, and permutation
kernels. `homology_playbooks` owns the measured crossover between direct Smith,
algebraic Morse, and simplicial Morse routes. `padic_sparse_smith` owns the
sparse/dense p-adic cutover. `finite_group_cohomology` covers minimal
resolutions through the order-32 case `C_2^5`, plus integral `S_3` and product
tables; that scale case belongs here rather than in the ordinary unit-test
lane.

Criterion suites compare complete public operations as well as isolated
kernels where needed. Materialization and conversion costs must be included in
at least one cell for every specialized storage route.

The native-library comparisons retain both sides' inputs and time their
operation-native result allocation. Their separate copy-out paths establish
exact parity without charging either side for the other's representation:

```bash
scripts/bench-pin.sh cargo run -p m4rst --release \
  --features validate-libm4ri --example bench_fair_dispatch
scripts/bench-pin.sh cargo run -p m4rste --release \
  --features validate-libm4rie --example bench_vs_m4rie
scripts/bench-pin.sh cargo run -p pwvla --release \
  --features flint --example bench_z2n_flint
scripts/bench-pin.sh cargo run -p sparla --release \
  --features flint --example matmul_bench_flint
scripts/bench-pin.sh cargo run -p sparb --release \
  --example bench_compare
```

## Macaulay2

All Macaulay2 drivers use one explicit runtime contract. `NOETHER_M2_BIN`
selects the executable. Set either `NOETHER_M2_REVISION` for a binary package
or `NOETHER_M2_SOURCE` for the matching source checkout:

```bash
export NOETHER_M2_BIN=/path/to/M2
export NOETHER_M2_REVISION=<exact-revision>
```

Each record includes the resolved executable, launcher/native-binary
fingerprints, reported version, source revision and package versions. Never
combine records from differing fingerprints in one comparison.
All external-CAS subprocesses remove inherited `LD_*` and `DYLD_*` loader
overrides; a developer's FLINT build environment must not change which GMP or
FLINT an oracle loads.

| Driver | Timed mathematical surface | Independent admission |
|---|---|---|
| `groebner_compare.py` | single-thread Gröbner systems over prime fields or `Q` | exact basis-size/result checks; detailed Noether phase counters |
| `macaulay2_core.py` | ideals, modules, syzygies, resolutions, Betti/Hilbert data, polynomial and sparse polynomial-matrix arithmetic | exact matrix identities, complex identities, hand-known fixtures, and Noether witness import |
| `macaulay2_quotient.py` | quotient construction, normal forms and repeated arithmetic | every representative is reprojected through Noether's exact quotient |
| `macaulay2_presented_modules.py` | kernel, image, cokernel, exactness and homology of presented maps | maps are admitted in fixed presentations and reduced to canonical submodules |
| `macaulay2_derived.py` | tensor, Hom, Tor, Ext, annihilators, Fitting ideals and module decompositions | full presented results are admitted through Noether |
| `macaulay2_decomposition.py` | radicals, minimal primes, equidimensional and primary decomposition | typed primality/primary evidence and exact intersection reconstruction |
| `macaulay2_normalization.py` | Noether normalization | exact dimension, invertible coordinate change and finite-module witness |
| `macaulay2_affine_normalization.py` | affine normalization and conductors | source-conductor preimage, dimension/components and normal-target checks |

Inspect a driver's current frozen corpus and selectors with `--help` (and
`--list` where available). Representative runs are:

```bash
scripts/bench-pin.sh .venv/bin/python \
  validation/bench/groebner_compare.py \
  cyclic-3 katsura-7 cyclic-6 semiregular-4 \
  --prime 65521 --repeats 7

scripts/bench-pin.sh .venv/bin/python \
  validation/bench/groebner_compare.py \
  cyclic-4 katsura-4 --rational --repeats 7

scripts/bench-pin.sh .venv/bin/python \
  validation/bench/macaulay2_core.py \
  cyclic-5-q twisted-cubic-q module-chain-q --repeats 7

scripts/bench-pin.sh .venv/bin/python \
  validation/bench/macaulay2_decomposition.py \
  --warm-repeats 51 --cold-repeats 3

scripts/bench-pin.sh .venv/bin/python \
  validation/bench/macaulay2_normalization.py \
  --warm-repeats 51 --cold-repeats 3
```

External output that is not canonical is compared by its admitted
mathematical claim, not by presentation-dependent coordinates or generators.
Macaulay2 candidates that require Noether verification are reported as
candidate-plus-verification time, distinct from Noether's native verified
construction.

## Magma release gate

`magma_release.py` is the strict public-release contract. Its tracked manifest
defines 71 deterministic workloads spanning scalar and polynomial arithmetic,
dense linear algebra over five coefficient families, grevlex and lex ideal
Groebner bases, module Groebner bases and syzygies, several complete- and
non-complete-intersection resolutions, number fields and orders, elliptic
curves, multiple permutation-group families and fast/fallback subgroup
operations, graph
automorphisms, degree-zero and higher homology, and Kahler differentials. Every
case is both an exact cross-system check and a paired warm-operation benchmark.
One long-lived process per backend materializes fixtures before timing; the
report keeps startup/import/setup separate from samples and refuses to compare
speed until all normalized answers agree. The oracle runs with a fixed random
seed and no per-user startup file. Run it through the complete one-command gate
documented in `validation/README.md`, which also requires the paper-scale
surface oracle.

Consolidate compatible local JSONL records with:

```bash
.venv/bin/python validation/bench/macaulay2_scorecard.py \
  --require-version <version> results/*.jsonl
```

The scorecard rejects mixed executables, revisions and package versions. It
separates canonical agreement, admitted noncanonical witnesses, unsupported
lanes and unpaired results.

## Sage

`sage_workflows.py` is the cross-domain warm-throughput scorecard. Each backend
runs in a long-lived worker, imports outside the timer, warms once, and
calibrates the timed loop. `sage_homology.py` applies the same rules to growing
torus triangulations and repeated subdivisions of the real projective plane;
it compares rational Betti numbers and complete integral finitely generated
abelian-group invariants. Shared facet generation is untimed.
`sage_exact_systems.py` pins an integer affine solve and integer-submodule
canonicalization; matrices, ambient modules and elements are all constructed
before calibration and timing.

```bash
scripts/bench-pin.sh .venv/bin/python \
  validation/bench/sage_workflows.py --sage /path/to/sage

scripts/bench-pin.sh .venv/bin/python \
  validation/bench/sage_homology.py --sage /path/to/sage

scripts/bench-pin.sh .venv/bin/python \
  validation/bench/sage_exact_systems.py --sage /path/to/sage
```

`sage_power_series.py` compares Noether's hard fixed truncation with a Sage
series capped immediately after Frobenius. Its separate late-trim Sage lane is
a representation diagnostic, not a like-for-like speed comparison. The
driver records time, RSS, stored coefficient slots and terms above the cap.

`sage_exterior_algebra.py` compares canonical sparse wedge arithmetic and the
full degree-three matrix induced by a dense linear map. Both long-lived workers
construct fixtures before timing and reject any disagreement in canonical
coordinates.

```bash
scripts/bench-pin.sh .venv/bin/python \
  validation/bench/sage_power_series.py --sage /path/to/sage

scripts/bench-pin.sh .venv/bin/python \
  validation/bench/sage_exterior_algebra.py --sage /path/to/sage
```

## Hyperelliptic Frobenius

The Rust hyperelliptic harness separates input-to-model construction from
integral Frobenius reduction. Cases include odd and even degree, small
characteristic, genus eight, and a genuine unramified extension base.

```bash
scripts/bench-pin.sh cargo bench -p noether --bench hyperelliptic
```

The paired driver defaults to the same complete H1 Frobenius polynomial on
both routes. Direct enumeration computes `N1,...,Ng` and uses the Rust Newton
reconstruction; Kedlaya computes the polynomial p-adically. Coefficients must
agree exactly. Fixture construction is excluded; public evidence and result
allocation are included. The default base fields have orders 3 and 9. Each
method has one warmup and five timed samples; method order alternates, and
individual-call timeouts and unsupported cases remain structured results.
JSONL retains each input, the largest required direct extension `q^g`, count
certificates' route metadata, precision budgets, raw samples, binary fingerprint
and CPU provenance, with a companion CSV.

```bash
cargo build --release -p noether --example bench_hyperelliptic_counts
scripts/bench-pin.sh .venv/bin/python validation/bench/hyperelliptic_counts.py \
  --output target/bench/hyperelliptic-counts.jsonl
```

`--gp /path/to/gp` additionally checks every returned count and Frobenius polynomial using
PARI/GP after both timed routes, retaining the exact defining-field
presentation and oracle script. It does not enter the production dependency graph.

`--cases prime:residue-degree:genus ...` selects individual curve cases rather
than a field/genus product. The following corpus bounds `q^g` by 531441 while
covering genera one through six. Both methods return the whole polynomial.

```bash
scripts/bench-pin.sh .venv/bin/python validation/bench/hyperelliptic_counts.py \
  --goal frobenius --cases 3:12:1 3:6:2 3:4:3 3:3:4 13:1:5 3:2:6 \
  --timeout 60 --repeats 3 --output target/bench/hyperelliptic-full-frobenius.jsonl
```

`--goal point_count` explicitly selects the single-base-field-count workflow:
that goal compares direct `N1` with deriving `N1` from Kedlaya's full polynomial.
It measures that public workflow rather than equivalent algorithm outputs.
Large-field enumeration streams coordinates within the ten-million-candidate
bound. `--methods direct` can calibrate a chosen workload independently.

`--methods kedlaya --baseline-binary /path/to/preserved/executable` pairs the
current implementation with an earlier release build on each identical curve.
Implementation order alternates, and every returned coefficient is checked
across both versions. JSONL and CSV identify each implementation and executable
fingerprint; `--baseline-source-sha256` records the preserved source snapshot.

`--corpus validation/bench/hyperelliptic_models.json` selects sixty seeded
equations through genus twenty, with dense, sparse and split branches, square
and nonsquare leading coefficients, general equations `y^2+h*y=f`, and an
alternate finite-field presentation. Fixture mathematics lives in the Rust
worker. Every seed and exact equation is retained. Each row selects either
both full-polynomial methods or Kedlaya alone; `--methods` filters those choices.
The corpus selector cannot be combined with `--baseline-binary` or `--cases`.

```bash
scripts/bench-pin.sh .venv/bin/python validation/bench/hyperelliptic_counts.py \
  --corpus validation/bench/hyperelliptic_models.json --gp /path/to/gp \
  --warmups 1 --repeats 3 --timeout 60 \
  --output target/bench/hyperelliptic-expanded.jsonl
```

The independent oracle completes the square from the original `f,h`. A
required oracle failure exits unsuccessfully; a model with no returned
Noether polynomial is not counted as verified. The fixture records the
constructor's squarefree-branch verification separately from the stronger
weighted cone-Jacobian proof. A resource-limited weighted proof is retained as
unavailable. Timed failures, timeouts and completed measurements remain distinct.

The independent validation-only PARI/GP scorecard is the ignored
`pari_arbitrary_genus_and_even_degree_parity` integration test (through genus twenty):

```bash
cargo test -p noether --release --test hyperelliptic -- --include-ignored
```

## Scale-only workloads

`surface_frobenius.py` runs the complete characteristic-31 surface calculation
from arXiv:1906.02261: extension point counts through degree 11, Frobenius
reconstruction and the Tate divisor-rank upper bound. Smaller degrees are
smoke measurements only.

```bash
scripts/bench-pin.sh .venv/bin/python \
  validation/bench/surface_frobenius.py
```

`tritangent_stress.py` measures the complete ignored arithmetic-ideal stress
tests from the same paper. It records corpus hash, wall/CPU time, peak RSS,
thread count and exact input sizes. Run the full cases on one pinned cache
group:

```bash
TERTIUS_BENCH_CPUS=group scripts/bench-pin.sh \
  python3 validation/bench/tritangent_stress.py
```

The default per-case timeout is 30 minutes and produces a structured timeout
record. The cited 82-variable M24 input is not distributed with Noether; an
authorized copy would be external benchmark data, not unit validation or a
public API capability.
