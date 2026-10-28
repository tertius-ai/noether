# Architecture

This document describes the Rust facade, Python bindings, storage, concurrency,
and build profiles. For installation and calculations, see
[Using Noether](USING_NOETHER.md).

## Language boundaries

The system has four layers:

```text
Python user
  │  noether.<domain>, typed stubs, capabilities(), describe()
  ▼
python/noether
  │  imports, protocols, bounded boundary conversions
  ▼
crates/noether-python          private PyO3 module: noether._engine
  │  translation only
  ▼
crates/noether                 public Rust facade and accepted registry
  ├─ generic mathematical floor
  ├─ private routine dispatch
  ├─ typed mathematical playbooks
  └─ retained Rust kernels, flint-core/FLINT, optional subprocesses
```

The public Rust facade is the authority. Its accepted registry names every
public domain, type, operation, and root value. The domains are `category`,
`combin`, `commalg`, `geometry`, `groups`, `homalg`, `linalg`, `numthy`, and
`topology`. Python exposes the same organization and the same mathematical
values. PyO3 adapters translate inputs, errors, and owned results; they do not
contain mathematical algorithms.

The distinction prevents a common failure mode in mixed-language scientific
software: an ergonomic wrapper slowly becomes a second implementation with
different edge cases. A surface change is complete only when the Rust item,
accepted registry, PyO3 adapter, Python stub, discovery record, tests, and
documentation agree. The generated Python reference checks its declarations
against the Rust-owned registry. The Rust facade has a separate local
reference built with `make docs-rust`.

Runtime discovery is part of the interface. `capabilities()` reports what the
installed artifact can execute, while `describe()` reports accepted signatures,
contexts, evidence, failures, backing implementations, and a tested usage. The
machine-readable indexes are generated artifacts; they never decide dispatch.
The smaller engine schema registry covers only values admitted to generic
cross-process hydration. A class-local JSON method does not silently enlarge
that registry.

## Coefficient contexts

Noether models a matrix as `Matrix<R>`, not as unrelated integer, rational,
finite-field, and backend-specific matrix families. The context `R` supplies
the scalar laws and distinguishes, for example, integers from residues, a
prime field from an extension field, or one fixed p-adic precision from
another. Polynomial contexts additionally retain variables, monomial order,
and grading. Maps between contexts are explicit values rather than implicit
coercions.

This gives generic algorithms a precise domain while allowing private storage
to specialize. A matrix over the integers can remain a FLINT integer matrix;
one over `F_2` can remain packed `m4rst` storage. Both still implement the same
public matrix operations when their algebra permits them. Context mismatches
fail before computation instead of guessing a coercion.

The [Smith-normal-form example](../examples/04_smith_normal_form.py) retains
its source matrix for verification:

```python example=examples/04_smith_normal_form.py
matrix = Matrix.from_rows(Z, [[2, 4, 4], [6, 6, 12], [10, 4, 16]])
smith = matrix.smith_normal_form()
smith.verify()
assert all(
    right % left == 0
    for left, right in zip(smith.invariant_factors, smith.invariant_factors[1:])
)
```

The [Rust facade example](../crates/noether/examples/readme.rs) uses the same
integer context:

```rust example=crates/noether/examples/readme.rs
let matrix = Matrix::from_rows(
    Z,
    [
        [Integer::from(2), Integer::from(4)],
        [Integer::from(6), Integer::from(8)],
    ],
)
.expect("the rows are rectangular");
let result = matrix.smith_normal_form();
assert_eq!(
    result.invariant_factors(),
    &[Integer::from(2), Integer::from(4)]
);
result.verify().expect("the Smith form verifies");
```

## Verification and assumptions

Many mathematical outputs carry enough source data and evidence to support a
`verify()` operation. Deserialization reconstructs or rechecks canonical data;
changing a claimed invariant, validity class, source matrix, or witness must
be rejected. This matters at process boundaries, where a syntactically valid
JSON record is not automatically a valid mathematical result.

Noether distinguishes three cases:

1. An unconditional result follows from retained inputs and replayable exact
   evidence.
2. A conditional result records typed assumptions, such as exact differential
   ranks at fixed p-adic precision.
3. An unsupported or inconclusive regime returns a typed error or diagnostic,
   not a plausible substitute.

Small scalar answers need not grow artificial certificate objects. The rule is
instead proportional: a result whose correctness depends on a nontrivial
route, assumption, or witness exposes that dependency. Tests construct minimal
hand-checkable cases first, exercise algebraic laws with property testing,
compare specialized and generic routes, tamper with serialized evidence, and
use independent CAS or library oracles where available. A required oracle that
does not run is a failed validation, not a skipped success.

## Algorithm selection

Routine dispatch and a mathematical playbook answer different questions.

Routine dispatch selects an implementation of one operation from its context,
layout, and measured regime. The choice is private and preserves the public
result. Dense exact arithmetic may use FLINT; local binary matrices use
`m4rst`; supported binary extensions use `m4rste`; power-of-two residue
matrices use `pwvla`; sparse exact matrices use `mzs`, `sparla`, or `sparb`.
Every specialized route has a generic or independently checked floor and a
differential test. Dispatch metadata is audited in
`crates/noether/src/kernel_routes.rs`.

A playbook selects among mathematically different strategies whose
preconditions or evidence differ. Smith computations, integral-line
realization, exact cutting planes, rational reconstruction, and p-adic
calculations use typed outcomes to distinguish proved, conditional,
inconclusive, and unsupported cases. Python never chooses these routes on the
facade's behalf.

The executable
[twisted-cubic example](../examples/15_groebner_twisted_cubic.py) illustrates
the boundary. The ideal owns a canonical exact meaning; its private Gröbner
route can change without changing the returned ideal or its invariants.

```python example=examples/15_groebner_twisted_cubic.py
ring = PolynomialRing(Q, ["x", "y", "z", "w"])
x, y, z, w = ring.generators()
ideal = ring.ideal([x * z - y**2, x * w - y * z, y * w - z**2])
ideal.verify()
assert ideal.dimension() == 2
assert ideal.degree() == 3
assert ideal.hilbert_numerator() == [1, 0, -3, 2]
```

`poly-core` owns packed monomials and coefficient-independent sparse
accumulation. `groebner-kernel` supplies field, modular-rational, and integral
engines and reuses matrix kernels for its coefficient domains.
`module-core` shares module terms and Schreyer orders, while
`resolution-kernel` owns degree scheduling and standard-graded Betti
calculations. The facade alone owns public ideals, modules, resolutions,
completeness claims, and evidence.

## Storage and kernels

Each immutable facade value has one authoritative physical representation.
`Matrix<R>` and its one-column `Vector<R>` view share their payload through an
`Arc`, so a clone or GIL-detached borrow does not copy matrix coefficients.
FLINT-backed polynomials and fixed-precision series keep native whole-object
storage. Sparse values retain native CSR storage in their selected kernel.

Representation changes are explicit and bounded:

- construction enters the selected storage directly;
- an operation may make one authoritative transition to another retained
  kernel, such as a Howell-form boundary;
- a terminal invariant may materialize into the implementation that owns it;
  and
- observation or transport may copy data through `to_rows`, serialization,
  sparse-to-dense conversion, MPI distribution, or bounded collection.

Algorithm-local panels and scratch buffers do not become a second facade
payload. Conversion counters, shared-clone tests, differential tests, and
resident operation chains guard against accidental host/native or
kernel/kernel round trips.

Higher constructions reuse this substrate rather than opening parallel object
hierarchies. For example, the Kähler differential module of a quotient retains
the conormal presentation and checks the Jacobian map. The executable
[cusp example](../examples/39_kahler_differentials.py) differentiates in the
quotient and verifies that the defining relation maps to zero.

```python example=examples/39_kahler_differentials.py
relation = y**2 - x**3
cusp = ring.quotient(ring.ideal([relation]))
omega = cusp.kahler_differentials()

assert isinstance(omega, KahlerDifferentials)
dx = omega.differentiate(cusp(x))
dy = omega.differentiate(cusp(y))
assert omega.differentiate(cusp(x * y)) == dx * cusp(y) + dy * cusp(x)
assert omega.differentiate(cusp(relation)).is_zero()
```

## Concurrency and Python execution

The facade owns one process-local CPU budget and one affinity-aware Rayon
pool. The default comes from `std::thread::available_parallelism()`, so CPU
affinity and container limits are respected. `NOETHER_NUM_THREADS` may lower
that budget and is read once. A computational phase has one thread-team owner;
nested facade phases reuse the pool, and native libraries remain serial unless
an explicit measured route transfers ownership.

Substantial PyO3 adapters parse dynamic inputs into owned Rust values, release
the GIL, run the synchronous Rust calculation, and reacquire Python only to
construct the result. Immutable facade values already held by frozen Python
classes are borrowed across that boundary without coefficient copies. The
generated operation index records direct detach sites as `releases_gil`, and
overlap tests check that the claim is real.

Rayon and MPI occupy different levels. Rayon uses shared memory inside one
process. `DistributedMatrix<FiniteField>` is an explicit multi-process value
whose identity includes its process grid and tile distribution. It never
appears through silent dispatch from a local `Matrix`. Each MPI rank stores
only packed `m4rst` tiles; multiplication uses a two-dimensional SUMMA
schedule, and factorization, solve, nullspace, inverse, and manifest persistence
avoid implicit full-matrix gathers. Bounded `distribute` and `collect` calls are the
visible transition points. Detailed communication and memory invariants live
with the kernel in
[`m4rst-distributed`](../crates/m4rst-distributed/README.md).

## Deployment

The public distribution is the versioned source tree. It builds the typed
Python package, PyO3 extension, and generated discovery indexes against the
pinned FLINT/GMP/MPFR stack. The repository also retains audited wheel and
CPU/MPI image recipes for local qualification, but does not currently publish
those artifacts. Cluster-native MPI installations compile against the site's
MPI.

Noether does not provision credentials, choose a cloud provider, answer an
authentication challenge, or turn an ordinary local operation into a remote
one. A human or an LLM writes one Noether program and chooses where to execute
it. That executor may be a workstation, a Docker host, a Kubernetes job, a
Modal function, or a Slurm allocation. The repository supplies reproducible
CPU/MPI images and a Slurm handoff helper for immutable source, explicit
resources, job dependencies, and result collection. Provider policy, budgets,
and credentials remain outside the mathematical core.

This boundary keeps the local API small and makes a future managed compute
service another executor of the same program, not a privileged source of
mathematical semantics. Remote workers run the Rust facade and packaged
indexes; Markdown is excluded from their image and cannot influence runtime
behavior.

## Native dependencies and licensing

Noether's source is MIT-licensed. A locally built wheel or image adds
replaceable LGPL FLINT, GMP, and MPFR libraries beyond system components. Such
an image does not claim one aggregate OCI license for that mixed closure.
Candidate checks inspect libraries, symbols, RPATHs, CPU requirements, license
texts, and software bills of materials rather than inferring conveyance from a
source dependency graph.

Optional Macaulay2, Singular, and msolve capabilities are arm's-length
subprocess adapters isolated in `noether-bridges`. The conveyed profile enables
none of them and links no GPL library. Capability discovery reports such a
backend only when its feature is compiled and its executable is available.
Native mathematics remains the always-present floor and GPL-backed capability
is a backfill target rather than a permanent architectural dependency.

## Reproducibility and limits

Observable results are deterministic. Randomized operations require an
explicit seed; kernel parallelism must agree across thread counts. Benchmarks
run with recorded CPU placement, separate construction and conversion from
the timed kernel, warm external processes when comparing CAS systems, and
verify equality before comparing speed. Candidate builds test source and, when
requested locally, wheels, supported Python versions, native dependencies, and
clean container contexts.

Noether's accepted registry is also its boundary. It is not a compatibility
clone of a general-purpose CAS. Unsupported coefficient laws, unavailable
proof hypotheses, unbounded collection, and unavailable optional executors
fail explicitly. The portable wheel profile contains no MPI, GPU, provider SDK,
or linked GPL backend. These absences keep the core composable: new accelerators
must implement an existing mathematical contract, while a genuinely new
distributed object or validity claim must be named and tested as such.

## Native build profiles

A locally built portable wheel is not a scalar-only build. It contains the
explicit SIMD paths used by the local exact-linear-algebra kernels:

| Kernel | Portable-build vector paths | Selection |
| --- | --- | --- |
| `m4rst`, dense matrices over `F_2` | AVX2 on x86-64; NEON on arm64 | Runtime feature check on x86-64; arm64 architecture baseline |
| `m4rste`, supported dense matrices over `F_{2^e}` | AVX2/GFNI on supported x86-64 cases; NEON on supported arm64 cases | Runtime feature and field-regime checks, with scalar fallback |
| `pwvla`, supported small residue rings | AVX2 on x86-64; NEON on arm64 | Runtime feature, coefficient-width, modulus, and size checks, with scalar fallback |

SIMD is internal dispatch, not a Python-package extra or Cargo opt-in. The
portable build deliberately does **not** apply `target-cpu=native` to the whole
engine: Linux x86-64 uses the documented `x86-64-v3` floor, macOS wheels use
their release architecture floors, and wider explicit kernels execute only
when the running CPU supports them. A source checkout can use
`make engine-native` for machine-specific compiler code generation.

This is separate from optional capabilities. The portable wheel profile excludes the
system-MPI `distributed` feature and validation-only libm4ri/libm4rie linkage;
those omissions do not disable the local SIMD kernels. MPI requires
`make engine-mpi`, while libm4ri and libm4rie remain GPL test and benchmark
oracles and never enter a conveyed wheel.

## Distributed F2 linear algebra

MPI support is an explicit build profile. A workstation or cluster login node
uses its active MPI installation:

```bash
make engine-mpi
mpirun -n 4 .venv/bin/python examples/distributed/f2_linear_algebra.py
```

On Slurm, load the site's compiler, Python, FLINT, and MPI modules, build on a
shared filesystem, and launch the same program with the site's supported
`srun` or `mpirun` path. The minimal `examples/distributed/slurm.sbatch`
template assigns one MPI process per task and one bounded Noether thread budget
per process; account, partition, module, launcher, and filesystem choices
remain site configuration.

For scaling and aggregate-memory evidence, first build the Rust benchmark
against that same MPI and submit the separate parameterized template:

```bash
cargo build --locked --release -p m4rst-distributed --features parallel --example benchmark
NOETHER_MPI_BENCH_OP=pluq \
NOETHER_MPI_BENCH_ROWS=8192 \
NOETHER_MPI_BENCH_COLS=8192 \
sbatch examples/distributed/slurm-benchmark.sbatch
```

The workload supports `matmul`, `rank`, `echelon`, `rref`, `pluq`, and `solve`;
it emits JSON with process-grid geometry, packed input/output sizes, elapsed
time, per-rank application-level communication calls/payload bytes, and per-rank
resident bytes and process peak RSS after the operation and verification,
exact-verification status, and a logical hash.
Store each MPI step in a separate JSON-lines file and pass the complete set
to `scripts/check-mpi-scale.py`. The checker proves node/rank participation,
affinity, packed-byte conservation, cross-grid hashes, exact solve evidence,
source/binary identity, requested scaling wins, and aggregate memory against
an explicit physical-RAM byte count. The exact release invocation is shown in
`crates/m4rst-distributed/README.md`.
The tracked template never chooses the site's account, partition, modules, or
storage paths.

For a clean acceptance handoff, the repository helper can bundle the exact
current commit, place an immutable checkout on any already-reachable SSH site,
and retrieve a machine-readable capability report:

```bash
ssh researcher@cluster.example true
scripts/slurm-handoff.py stage \
  --host researcher@cluster.example \
  --remote-root /shared/project/noether-handoff
```

The source tree must be clean and committed. The report is written under the
gitignored `target/hpc/` tree and records source identity, CPU/filesystem facts,
visible tools, MPI, Slurm partitions/associations, and relevant module queries.
Git verifies the bundle, detached checkout, and clean tree on the login node;
staging then writes an adjacent revision marker so compute nodes need no Git
installation to preserve the source-identity check.
The tiny remote probe supports the older system Python commonly retained on
HPC login nodes; this does not lower Noether's Python 3.10 runtime requirement.
The helper does not establish a VPN, answer an authentication challenge, choose
credentials, or modify SSH configuration; making the ordinary `ssh ... true`
preflight work is entirely caller-owned. Account, partition, modules, budgets,
and build choices likewise remain caller-owned.

Once those choices are known, the same helper can submit only the fixed
build-plus-benchmark acceptance graph. Every resource is explicit and the
aggregate requested node-hours must fit under a caller-supplied ceiling:

```bash
scripts/slurm-handoff.py submit \
  --site target/hpc/researcher_cluster.example/REVISION/site.json \
  --partition compute \
  --module gcc/12.2 --module openmpi/4.1 --module rust/1.89 \
  --node-count 1 --node-count 2 \
  --operation matmul --operation rank --operation echelon --operation rref \
  --operation pluq --operation solve \
  --mpi-launcher srun \
  --ntasks-per-node 2 --cpus-per-task 8 --memory 64G \
  --build-cpus-per-task 8 --build-memory 16G \
  --build-walltime 00:30:00 --run-walltime 00:30:00 \
  --max-node-hours 10
```

For rectangular matmul, `--tile-rows` and `--tile-cols` describe the left
input, while optional `--right-tile-rows` and `--right-tile-cols` describe the
right input. The output inherits the left tile rows and right tile columns, so
the handoff's balance gate evaluates the actual output-owner layout rather than
assuming both operands share one geometry.

Use `--no-modules` instead of `--module` only when the ambient site toolchain
is intentional. Select `srun` when the MPI module supports Slurm's direct PMI
launch, or `mpirun` when that is the site's supported allocation launcher. The
`mpirun` path uses the requested task count and binds each rank to its requested
CPUs. After loading the explicit modules, a conventional rustup installation
is recovered from `~/.cargo/env` when present; no shell profile or ambient
submitted environment is otherwise relied upon. `--dry-run` validates and
prints the complete job graph without contacting the site. The build runs
inside a one-node allocation, and each job set has an isolated Cargo target so
concurrent module/toolchain profiles cannot overwrite one another. Each
benchmark depends on its successful build. The result names and Slurm job IDs
are stored in a gitignored job-set JSON file printed by the command.

Collection is deliberately one-shot rather than a resident monitor:

```bash
scripts/slurm-handoff.py collect --job-set target/hpc/.../job-sets/JOB_SET.json
```

Exit status 3 means jobs are still incomplete and the JSON result reports
their current accounting states; rerun later. A failed job's available stdout
and stderr are copied into the local run directory and their bounded tails are
included directly in the JSON response, so failure diagnosis needs no separate
SSH command. Completed files are retrieved
atomically and passed through the scale gate, which checks exact source and
binary identity as well as topology, hashes, memory, and any requested speedup
or aggregate-memory claims.

The reference images are built and audited with:

```bash
make image-cpu
make check-image-cpu
make image-mpi
make check-image-mpi
```

`scripts/build-images.sh` derives each image tag from the package version in
`pyproject.toml`. The MPI image contains an MPI-enabled package and one matched
Open MPI runtime. It is a reproducible reference for ordinary Docker/OCI hosts,
not a promise of ABI compatibility with an HPC site's MPI, PMIx, or
interconnect. Cluster-native builds are therefore the normal HPC path.

`DistributedMatrix` is deliberately distinct from local `Matrix`. Every rank
executes the same collective sequence and retains only its packed local tiles.
`distribute(..., max_bytes=...)` is an explicit packed upload;
`collect(..., max_bytes=...)` is the only whole-matrix gather. Arithmetic,
factorization, solve, nullspace, inverse, and persistence do not gather an
operand implicitly.

## Documentation builds

`make docs` generates the Python reference, user guide, and worked examples
under `target/noether-docs/`. Example explanations live in the scripts'
module docstrings. The site displays their imports and calculations without
license headers or the script entry point. The examples remain runnable
through `make test-examples`.

Missing authored API descriptions stay empty in the generated index and are
listed in the site's build manifest. The generator does not infer mathematical
guarantees from names. `make docs-rust` builds the Rust facade reference locally;
`make docs-mpi` includes the system-MPI API. Neither is bundled into the user site.

The Python API index and transport schemas are separate downloads in the
developer section. Source links record the commit used for the documentation
build. Generated pages, schemas, and navigation are untracked artifacts.

The GitHub Pages workflow checks documentation on pull requests and `main`.
Publication selects the highest stable `vMAJOR.MINOR.PATCH` tag by numeric
version and checks out its exact commit. A manual run can select that tag
explicitly. Prereleases and older maintenance tags cannot update the public
site. Publication checks the generated source identity and rechecks the
latest stable tag immediately before deployment. The `github-pages`
environment must permit deployments from stable tags.
