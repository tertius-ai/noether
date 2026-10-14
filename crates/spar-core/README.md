# spar-core

Shared **contracts** for the Tertius sparse linear-algebra kernels.

`sparla` (exact ℤ / F_p) and `sparb` (Arb/Acb ball) are sibling crates that
cannot depend on one another. This
crate is the surface they all depend **up** into and share. It has zero
runtime dependencies and is `#![forbid(unsafe_code)]`.

Today that surface is one trait — `SparseRingStorage`, the
ring-independent structural view of a sparse matrix (shape, sparsity,
and the cheap structural predicates). Value-touching operations
(`get`, row iteration, SpMV, arithmetic) are scalar-specific and stay on
each kernel's own types.

The trait set grows only when a new member is genuinely ring-independent
and has more than one implementor. Algorithm-specific abstractions stay in
their mathematical owner rather than here.

```bash
cargo test -p spar-core --release
```

## Licensing

Licensed under the MIT License — see the repo-root `LICENSE`.
Copyright (c) 2026 Benjamin Antieau and Deven Manam

Every source file carries the SPDX header,
no `LICENSE` file. No third-party linkage (no deps at all).
