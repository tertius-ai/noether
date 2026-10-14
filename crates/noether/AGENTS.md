# Noether Rust facade — agent guidance

This file refines `../AGENTS.md` for the public `noether` crate. The facade is
the mathematical source of truth. Python projects it; kernels accelerate it.

## Ownership

- Public mathematical objects and operations live in their owning domain
  module. The crate root contains only cross-domain traits, errors, scalar
  contexts, discovery, and release metadata.
- A kernel type is re-exported only when it is itself the accepted mathematical
  identity. Otherwise the facade owns the public type and calls the kernel
  through private storage or an adapter.
- PyO3, Python naming, and wheel transport do not enter this crate. They belong
  in `noether-python` and `python/noether`.
- `public_surface.rs` owns the accepted type, operation, and root-value index.
  `discovery.rs` owns the accepted schema registry. Change them with the Rust
  API and its Python projection, never as an independent list-editing exercise.

## Algebra and dispatch

`Matrix<R>`, `Vector<R>`, polynomials, series, and related objects are
parameterized by coefficient contexts. New contexts implement the smallest
honest algebraic traits; do not introduce Sage-style parent/element machinery
or a backend-specific public matrix family.

Generic implementations are the correctness floor. Specialized storage and
kernels are private dispatch selected for a proved supported regime. The
specialized path must preserve the generic result type and have differential
tests against that floor. A backend choice is routine dispatch, not a public
mathematical playbook. Mathematical playbooks expose their hypotheses,
verification, and conditional validity in typed results.

Fail closed when a trait law or backend precondition is unavailable. Test the
smallest hand-checkable case, empty and thin shapes, context mismatches, and the
boundary at which dispatch changes.

## Unsafe and native storage

The crate root denies unsafe code. Raw FLINT ownership and calls are confined
to these explicitly named modules:

- `src/scalar.rs` and `src/ball.rs`;
- `src/commalg/residue.rs` and `src/commalg/polynomial_flint.rs`;
- `src/linalg/flint.rs`;
- `src/numthy/{integer,finite_field,number_field,number_field_order,padic,prime_ideal,qadic}.rs`.

Each owned C value is initialized exactly once, cleared exactly once, and
deep-cloned. Contexts outlive every value that borrows or stores their raw
pointer. `Send`/`Sync` implementations state the corresponding FLINT
immutability and ownership argument. Adding unsafe code anywhere else requires
updating this list and explaining why it cannot live in an existing boundary.

Prefer persistent FLINT whole-object storage for supported exact contexts.
Conversions occur at dispatch boundaries, not inside arithmetic loops. Generic
objects must still exist for every context that satisfies the public trait.

## Surface and verification

Public items have rustdoc and stable mathematical names. A surface change is
complete only when its semantic description, accepted index, PyO3 adapter,
typed Python module/stub, discovery schema, and projection tests agree. Run
`make api-docs` after changing public docstrings or the projected surface.

Useful inner-loop commands from the repository root are:

```bash
cargo test -p noether
cargo test -p noether --release
cargo test -p noether --test flint3_abi
```

Before a PR, run `make test-all`. Optimized paths also need the relevant pinned
benchmark and their generic differential tests.

## Local concurrency

The facade owns one affinity-aware Rayon pool through `src/concurrency.rs`.
`NOETHER_NUM_THREADS` is the only process override and may only lower the
schedulable CPU count. Parallel facade routes install their work in that pool;
they do not initialize another global pool or mutate native-library thread
counts. Enabling another kernel's `parallel` feature requires a facade caller,
a pinned benchmark, deterministic thread-count parity, and an explicit
one-owner integration here.
