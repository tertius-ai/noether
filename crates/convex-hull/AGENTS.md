# convex-hull

Agent instructions for Noether's exact-rational bounded-polytope kernel. Read
`crates/AGENTS.md` first; its kernel rules apply here.

## Scope and boundary

`convex-hull` is a standalone safe-Rust algorithm kernel over the shared
`flint_core::Rational` scalar. It owns exact V/H conversion, incidence, face
lattices, and membership predicates for nonempty bounded rational polytopes.
Unbounded polyhedra and cones are outside this crate's surface.

The canonical Noether value is `noether::geometry::Polytope`. That thin facade
and this kernel use one FLINT `Rational` identity without string or packing
conversion. `noether-python::py::convex_hull` projects the facade as
`noether.geometry.Polytope`, using `fractions.Fraction` at the Python boundary.
Do not add loose kernel functions to the private extension namespace.

## Architecture

```text
src/
  lib.rs            crate root, Q alias, re-exports, q/q_point helpers
  error.rs          typed construction failures
  linalg.rs         exact rational linear algebra
  cone.rs           pointed-cone extreme-ray engine
  polytope.rs       Polytope and V/H enumeration
  beneath_beyond.rs incremental V-to-H hull
  face_lattice.rs   meet-closure face lattice and f-vector
  tests.rs          crate unit tests
tests/              correctness, adversarial, algebraic and proptest layers
```

The double-description engine computes H-to-V through homogenization and
V-to-H through a polar cone. Beneath-beyond is a distinct incremental V-to-H
algorithm and must remain differential-tested against double-description.

All facet arithmetic happens in the affine hull. Vertices remain in ambient
coordinates. For a lower-dimensional embedding, kernel facet inequalities use
reduced affine coordinates; the Noether facade therefore fails closed rather
than reporting them as ambient inequalities.

## Invariants

- All geometric decisions use exact rational arithmetic; no `f64` predicate
  decides incidence.
- The crate has no GPL link and forbids unsafe code. Exact scalar operations
  reach dynamically linked LGPL FLINT through `flint-core`.
- Extreme rays are reduced to primitive-integer form at every step to control
  coefficient growth.
- Beneath-beyond and double-description agree on their exact facet sets.
- Empty, infeasible, unbounded and malformed inputs return `HullError`.
- Lower-dimensional hulls, duplicate points and interior points are valid.
- Randomized tests use explicit deterministic seeds.

## Build and validation

```bash
cargo test -p convex-hull --release
cargo clippy -p convex-hull --lib -- -D warnings
cargo fmt -p convex-hull -- --check
```

The public boundary is covered by
`validation/unit/geometry/test_polytope_facade.py`. Sage parity lives
in `validation/parity/polyhedral/test_convex_hull_parity.py`; when
claiming oracle verification, confirm Sage actually ran rather than skipped.
Run performance comparisons through `scripts/bench-pin.sh` and record CPU
placement. The crate-owned command is:

```bash
scripts/bench-pin.sh cargo bench -p convex-hull --bench kernel
```

## Unsafe inventory

None. The crate root forbids unsafe code.
