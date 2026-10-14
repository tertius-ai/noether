# poly-core guidance

Follow the repository and `crates/AGENTS.md` rules. `poly-core` owns
coefficient-independent packed monomials, round-local monomial interning,
contiguous term arenas, finite truncation regions, and generic sparse
polynomial arithmetic. Its `f2` module owns coefficient-free sparse support
arithmetic and bit-packed fixed-region series. It does not own Noether
contexts, ideals, Gröbner
strategy selection, serialization, or Python projection.

## Invariants

- Monomials have one fixed arity and use the narrowest exact `u16`/`u32`
  representation.
- Monomial IDs are stable for the lifetime of an arena.
- Sparse polynomial terms are unique, nonzero, and strictly descending in the
  declared order.
- Truncation bounds are strict and are checked before coefficient
  multiplication.
- F2 sparse supports are unique and strictly descending in their ring order.
- F2 dense-series padding bits are zero and addition is packed XOR.
- Hash tables are internal accumulators and indices only. Every observable
  support is sorted by the declared monomial order before it leaves the crate.
- The crate contains no unsafe code and no native-library dependency.

## Commands

```bash
cargo test -p poly-core --release
cargo clippy -p poly-core --all-targets -- -D warnings
scripts/bench-pin.sh cargo bench -p poly-core --bench polynomial
```
