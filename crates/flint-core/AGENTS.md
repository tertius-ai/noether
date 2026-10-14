# flint-core — agent guidance

This file refines `../AGENTS.md` for the safe owned scalar layer over
`flint3-sys`.

`flint-core` owns RAII values and primitive arithmetic for `fmpz`, `fmpq`,
`arb`, and `acb`. It does not own mathematical parent contexts, matrices,
polynomials, number fields, or algorithm dispatch; those belong to consuming
kernels or the Noether facade.

The crate root denies unsafe code. The complete allowed unsafe set is:

- `src/integer.rs` — `fmpz` lifecycle, arithmetic, strings, and raw seams;
- `src/rational.rs` — `fmpq` lifecycle, canonical arithmetic, and raw seams;
- `src/ball.rs` — `arb` lifecycle, enclosure arithmetic, and encoding;
- `src/acb.rs` — `acb` lifecycle and rigorous complex arithmetic.

Every constructor fully initializes its C value, every `Drop` clears it once,
and every `Clone` is deep. Raw-pointer constructors document validity and
aliasing preconditions. Public mutable pointers borrow `&mut self`; callers may
not retain them beyond that borrow. `Send` and `Sync` implementations require a
written ownership and FLINT-read-only argument beside the implementation.

Integer and rational equality, hashing, serialization, and display operate on
canonical mathematical values, including FLINT's inline/heap `fmpz`
representation. Ball operations preserve containment; a midpoint-radius export
must remain conservative, and exact serialization must round-trip the entire
FLINT value rather than a display approximation.

Run from the repository root:

```bash
cargo test -p flint-core
cargo test -p flint-core --features serde
```

Changes to raw FLINT calls also require the facade ABI smoke tests:

```bash
cargo test -p noether --test flint3_abi
```
