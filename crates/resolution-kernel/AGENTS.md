# resolution-kernel guidance

Follow the repository and `crates/AGENTS.md` rules. This crate owns packed
standard-graded Betti computation, deterministic degree-cell scheduling, and
Schreyer iteration over the verified module operations in `groebner-kernel`.
Noether owns public resolution semantics, completeness, minimality and
evidence.

## Invariants

- Every returned Betti multiplicity is an exact residue-space dimension after
  incoming and outgoing ranks are removed.
- Presentation degrees and module frames are verified before iteration.
- Schreyer degree cells are ordered by degree, then by stable pair position.
- Specialized F2, binary-extension and prime-field lanes agree with the
  generic exact-field floor.
- Resource and projective-dimension bounds fail closed without returning a
  partial result.
- The crate contains no unsafe code.

## Commands

```bash
cargo test -p resolution-kernel --release
cargo clippy -p resolution-kernel --all-targets -- -D warnings
scripts/bench-pin.sh cargo bench -p resolution-kernel --bench betti
```
