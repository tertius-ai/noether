# Kernel crates — agent guidance

These crates are focused components beneath the Noether facade. The root
`AGENTS.md` contract applies; this file adds the common kernel standard, and a
crate's own `AGENTS.md` names its exact invariants and commands.

## Ownership

Every retained crate names:

1. the operation and coefficient contexts it owns;
2. the Noether caller and dispatch condition that use it;
3. its generic reference or scalar fallback;
4. its independent evidence or differential oracle;
5. its measured regime and pinned benchmark.

A crate or optional feature without a current caller is deleted. Kernels may
have directly useful APIs, but `noether` owns the public mathematical identity,
strategy, evidence, and cross-domain traits. Re-export a kernel type only when
it is itself that identity; otherwise adapt it behind private facade storage.

Pure-Rust or Rust-owned kernels are `m4rst`, `m4rst-distributed`, `m4rste`,
`mzs`, `sparla`, `pwvla`, `sparb`, `perm-core`, `graph-core`, `geom-core`,
`convex-hull`, `lattice-enum`, `resolution-kernel`, and `groebner-kernel`.
Shared support belongs to `spar-core`, `flint-core`, `poly-core`,
`module-core`, `tertius-build`, upstream `flint3-sys`, and the isolated
`noether-bridges` subprocess boundary. All workspace crates are unpublished.

## API and storage

Use concrete mathematical types in kernels. Broad algebraic traits and generic
dispatch belong in `noether`. Hot elements normally borrow their parent
context; values that escape into facade storage, PyO3, detached work, or
distribution may use a thin owned seam such as `Arc<Parent>`.

Allocation-sensitive arithmetic supplies an `_into` form when reuse matters:

```rust
fn mul(&self, rhs: &Self) -> Self;
fn mul_into(&self, rhs: &Self, out: &mut Self);
```

The `_into` form owns the hot implementation and fully overwrites its output;
fallible calls leave the output unchanged on error. Value-returning forms
allocate once and delegate. FFI wrappers benchmark reusable forms against a
raw same-process library loop with equivalent allocation.

## Correctness and determinism

A mature kernel has focused unit cases, public integration tests, exact
properties, an independent differential route, and adversarial empty/thin/
boundary/malformed inputs. Minimal hand calculations precede randomized
sweeps. Assertions use exact values and reconstruction identities. Public
random algorithms accept explicit seeds.

Hash tables are internal membership and lookup structures, never observable
ordering authorities. Canonical results remain stable across process hash
seeds and admitted Rayon thread counts. An unavailable oracle is skipped or
reported unavailable, never counted as parity.

## Performance, SIMD, and unsafe code

Run benchmarks in release mode through the repository wrapper:

```bash
scripts/bench-pin.sh cargo bench -p <crate>
```

Record CPU/cache placement, absolute median time, input, precision, and thread
count. Dispatcher thresholds change only with pinned paired measurements and
differential cases on both sides of the boundary.

SIMD is runtime-dispatched over an always-present scalar fallback. Each vector
path has bit-identical differential tests and a benchmark. Unsafe code is
confined to the modules listed in the crate's `AGENTS.md`; each block documents
alignment, bounds, aliasing, initialization, and dispatch. A crate-level
`forbid(unsafe_code)` records current implementation, not a permanent ban on a
properly isolated vector path.

Rayon stays optional unless the facade has one measured owner for that phase.
Serial results remain available and parallel results are mathematically
identical and deterministic.

## Dependencies and layout

Kernel source is MIT and never links GPL. Oracle-only features use the
`validate-<oracle>` convention and stay outside default and conveyed graphs.
Dynamic LGPL dependencies are acceptable at an explicit safe wrapper boundary.
Re-audit the final artifact whenever a dependency or feature changes.

A substantive kernel normally has:

```text
src/          implementation and focused unit tests
tests/        integration, property, parity, and adversarial tests
benches/      reproducible Criterion harnesses
examples/     a small current example when useful
README.md     purpose, Noether caller, quick start, and measured status
AGENTS.md     local invariants, unsafe inventory, and commands
Cargo.toml    workspace-inherited metadata; no member release profile
```

Shared-contract, build-helper, and `*-sys` crates may be README-only when this
file covers them. Every non-generated Rust source has the repository SPDX and
copyright header. Public items have useful rustdoc.

## Gates

Use the crate's local commands while developing. Before a PR, run from the
repository root:

```bash
cargo fmt --all -- --check
cargo check --workspace --exclude m4rst-distributed
cargo test --workspace --release --exclude m4rst-distributed
make test-all
```

Also run `make test-mpi` on a host with system MPI when the distributed member
or facade changes, plus the relevant feature-specific oracle and pinned
benchmark. A link-closure change rebuilds and audits the final wheel or image.
