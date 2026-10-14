# Noether: An Evidence-Carrying Architecture for Scientific Mathematics

Benjamin Antieau and Deven Manam

## Abstract

Noether is a deterministic computation layer for scientific mathematics. Its
public Rust crate owns mathematical objects, algorithms, implementation
selection, and verification evidence; a typed Python package projects that
same surface without reimplementing the mathematics. Explicit coefficient
contexts give generic objects one mathematical identity, while persistent
native storage and specialized Rust kernels supply performance. Conditional
claims retain their assumptions, substantial Python calls release the GIL, and
unsupported hypotheses fail closed. The same facade runs in a local process,
an ordinary CPU container, or an explicit MPI program. This paper explains the
architectural choices that make those properties coexist.

## 1. Design question

A useful computer-algebra layer for humans and language models must answer more
than “can this calculation be expressed?” It must also answer:

- which ring, field, precision, grading, or presentation gives the object its
  identity;
- which hypotheses make the result valid and how those hypotheses can be
  replayed;
- whether a faster implementation changes representation or semantics;
- where expensive work executes and which thread team owns it; and
- what native and licensed code enters the artifact that is actually shipped.

Noether addresses these questions with five constraints. Mathematical truth
has one owner. Context is part of type identity. Accelerators remain private
unless they introduce a genuinely different mathematical object. Material
results are verifiable or explicitly conditional. Deployment runs the same
program rather than introducing a second remote mathematics API.

## 2. One authority, two language surfaces

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
documentation agree. The generated unified reference checks this projection
and places Python signatures beside their Rust declarations.

Runtime discovery is part of the interface. `capabilities()` reports what the
installed artifact can execute, while `describe()` reports accepted signatures,
contexts, evidence, failures, backing implementations, and a tested usage. The
machine-readable indexes are generated artifacts; they never decide dispatch.
The smaller engine schema registry covers only values admitted to generic
cross-process hydration. A class-local JSON method does not silently enlarge
that registry.

## 3. Coefficient contexts are mathematical identity

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

The following excerpt is taken verbatim from the executable
[Smith-normal-form example](../examples/04_smith_normal_form.py). The result
keeps its source and can replay its invariant-factor claim.

```python example=examples/04_smith_normal_form.py
matrix = Matrix.from_rows(Z, [[2, 4, 4], [6, 6, 12], [10, 4, 16]])
smith = matrix.smith_normal_form()
smith.verify()
assert all(
    right % left == 0
    for left, right in zip(smith.invariant_factors, smith.invariant_factors[1:])
)
```

The Rust surface expresses the same ownership directly. This excerpt comes
from the checked [Rust facade example](../crates/noether/examples/readme.rs).

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

## 4. Evidence is a result, not a log message

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

## 5. Routine dispatch and mathematical playbooks

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

## 6. Persistent representation and kernel composition

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

## 7. Concurrency and Python execution

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

## 8. Deployment is execution, not orchestration policy

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

## 9. Licensing follows the artifact boundary

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

## 10. Reproducibility and limits

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

## 11. Publication form

This document is both the living architecture reference and the source of the
short architectural paper rendered by `make docs`. Keeping it under `docs/`
gives the implementation one current architectural account and keeps every
link and example excerpt in the ordinary freshness gate. No duplicate
manuscript source lives under a permanent `paper/` directory. A fixed scholarly
snapshot with publication-specific metadata can live in a separate repository
without becoming a second authority for the evolving implementation.

The result is deliberately modest: Rust owns mathematical truth, Python makes
it usable, contexts make algebra explicit, evidence makes claims inspectable,
kernels make common paths fast, and deployment changes where the same program
runs rather than what it means.
