# Noether — agent guidance

Noether is the deterministic computation layer for scientific mathematics: a
public Rust facade, a typed Python projection, and the focused kernel crates
that the facade composes. This file is the repository contract. More specific
rules live in `crates/AGENTS.md` and a component's own `AGENTS.md`.

## Architecture and ownership

```text
Python domains (`python/noether`)
        ↓
private PyO3 transport (`crates/noether-python`)
        ↓
public Rust facade (`crates/noether`)
        ↓
Rust mathematical playbook and routine dispatch
        ↓
kernel crate, FLINT, or accepted subprocess bridge
```

Rust is the mathematical source of truth. The public surface has the same nine
owners in both languages: `category`, `combin`, `commalg`, `geometry`,
`groups`, `homalg`, `linalg`, `numthy`, and `topology`. Shared scalar,
context, error, validity, and discovery vocabulary belongs at the package root.
Natural operations are methods and standard protocols, not duplicate module
functions.

`crates/noether` owns public types, traits, algorithms, evidence, semantic
identities, and the accepted release index. `crates/noether-python` contains
transport only. `python/noether` supplies typed imports and small boundary
adapters; it never becomes a second CAS. `noether._engine` is private.

A public-surface change is atomic across its Rust owner, accepted index,
PyO3 adapter, Python import and stub, discovery/schema entry, tests, and useful
documentation. Do not add a Python-only mathematical object or operation.

## Mathematical model

A context and an element are distinct. Contexts own mathematical identity:
characteristic, modulus or presentation, variables, term order, and precision.
Operations reject incompatible contexts unless the caller supplies an explicit
checked map. Fixed p-adic, q-adic, ball, and series precision stays part of the
typed context; no layer silently guesses or refines it.

Generic Rust algorithms are the correctness floor. Specialized FLINT or kernel
storage is private, sticky across supported operations, and differentially
checked against that floor. Conversion occurs at an explicit boundary, never
inside an arithmetic loop.

Routine implementation dispatch chooses an implementation of one operation
from context, layout, and measured regime. A mathematical playbook chooses
between genuinely different strategies and exposes its hypotheses and
evidence. Both live in Rust; there is no public planning, costing,
`plan()`, or `execute()` framework.

## Correctness contract

A wrong confident answer is worse than a typed failure.

- Start with the smallest hand-verifiable case and test empty, thin, malformed,
  and context-mismatched inputs.
- State hypotheses and unsupported regimes in the API; fail closed when a
  precondition or proof is unavailable.
- Property-test algebraic laws, representation conversions, determinism, and
  specialized-versus-generic agreement over the stated domain.
- Use an independent CAS or library oracle where available. A skipped required
  oracle is unverified, never passed.
- Retain replayable evidence for material claims and test `verify()` and
  deserialization independently of construction.
- Public randomness takes and records an explicit seed.
- Every correctness or performance fix keeps the triggering regression and
  adds a nearby case that cannot depend on the same shortcut. Extend the paired
  CAS scorecard when one exists.
- Release candidates receive independent reviews of mathematics, performance,
  documentation, licensing, and packaging.

## Python and concurrency

Python performs validation, conversion, and bounded display. Substantial work
uses `Python::detach`; no `Bound`, `PyRef`, callback, or borrowed Python buffer
crosses that boundary. Representations and summaries must not trigger remote
work, materialize large storage, or perform unbounded computation.

The facade owns one affinity-aware Rayon pool. `NOETHER_NUM_THREADS` may only
lower the schedulable CPU count. Do not create another global pool, mutate
native-library thread counts, or nest uncontrolled parallelism. MPI code is
feature-gated and follows `crates/m4rst-distributed/AGENTS.md`.

## Performance and unsafe code

Persistent native storage and no-copy operations are the default where a
backend has a useful whole-object representation. Benchmark the conversion
boundary as well as the kernel. Run every measurement through:

```bash
scripts/bench-pin.sh <benchmark command>
```

Record the wrapper's CPU/cache provenance and compare paired implementations
in one machine state. On heterogeneous-cache machines, an unpinned result is
not evidence.

Kernel SIMD is runtime-dispatched over an always-present scalar fallback and
has a bit-identical differential test plus a pinned benchmark. Conveyed
x86-64 artifacts use the `x86-64-v3` floor; wider ISA paths remain runtime
choices. `target-cpu=native` is an explicit local opt-in. Never set environment
`RUSTFLAGS`, because it replaces configured flags rather than extending them.

Unsafe code stays in the small FFI, SIMD, or view modules named by the owning
component's `AGENTS.md`. Each block states its bounds, aliasing, initialization,
and thread-safety invariant. Adding unsafe code without updating that inventory
is a defect.

## Dependencies, licensing, and artifacts

Every crate and dependency must have a current facade caller or build/release
role. Native Rust is the correctness floor. LGPL FLINT/GMP/MPFR and similar
libraries may be dynamically linked with complete notices and relinking terms.

Production GPL capability is isolated in `crates/noether-bridges` behind an
operation-specific subprocess feature. It is absent from the default conveyed
profile; missing tools produce typed unavailable errors. Validation-only GPL
libraries never enter a release dependency closure. Audit the final wheel or
image, not only Cargo metadata.

Remote engine images exclude all Markdown. Runtime code must not read a `.md`
file. Conveyed CPU/MPI images are non-root and contain no tests, caches,
planning records, build tools, or linked GPL libraries.

## Documentation and repository discipline

Living documentation describes the current code without chronology,
codenames, planning, or strategy. The permanent design set is:

- `docs/ARCHITECTURE.md`;
- `docs/CONTEXTS_AND_PRECISION.md`;
- `docs/USING_NOETHER.md`.

Planning and measurement records belong in issues and pull requests;
nontechnical organizational strategy belongs outside this repository. Routine
feature detail belongs in rustdoc, Python docstrings, an existing design
document, or the owning component's `AGENTS.md`.

Generated API references, schemas, and the unified site are disposable build
artifacts. Regenerate them with `make api-docs` or `make docs`; do not commit
them or make runtime behavior depend on them. Keep README/AGENTS commands,
counts, paths, and names source-verified.

## Working gates

From the repository root:

```bash
make sync          # locked development environment
make engine        # wheel-compatible Python extension
make test          # acceptance smoke tests
make test-all      # required local pre-PR gate
make test-stress   # expanded properties and bounded slow/ignored tests
make docs          # unified local Python/Rust documentation
make candidate-linux # unpublished manylinux wheel candidate and interpreter checks
make candidate-macos # unpublished native macOS wheel candidate and interpreter checks
make validate      # explicitly enabled external-CAS lanes
make lint          # formatting, types, docs, shell, versions, Clippy
```

`make validate` is env-gated; claim oracle coverage only when the required
oracle ran. Use `make audit`, `make test-msrv`, `make test-sanitizer`, wheel and
image audits when the change reaches those boundaries. The ordinary
`make test-all` gate is MPI-free; run `make test-mpi` separately on a host with
system MPI.

Public push and pull-request CI runs only on GitHub-hosted runners. Routine and
pre-release evidence is also produced on the Linux development host; native
macOS wheel, interpreter, CAS, and release checks run on a maintainer's arm64
Mac. An x86_64 macOS claim requires a complete x86_64 toolchain under Rosetta
or a real Intel runner. Never execute public-repository code on a self-hosted
runner. Wheel and image workflows are unpublished qualification tools unless a
future release policy explicitly promotes those artifact classes.

Preserve unrelated user changes. Use `rg` for discovery and `apply_patch` for
edits. Never use destructive Git or broad recursive deletion to clean a dirty
tree.
