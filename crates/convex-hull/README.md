# convex-hull

`convex-hull` is Noether's safe-Rust algorithm kernel for nonempty bounded convex
polytopes over the rational numbers. It stores both exact representations:
vertices and facet inequalities `a · x <= b`, together with vertex/facet
incidence and the face lattice.

The crate and Noether's public `noether::geometry::Polytope` adapter share the
FLINT-backed `noether::Rational` scalar without a representation conversion.
Python sees
the same semantic type as `noether.geometry.Polytope` with
`fractions.Fraction` coordinates.

## Surface

- `Polytope` and `convex_hull` construct a polytope from exact vertices.
- `vertex_enumeration` constructs vertices from a bounded H-representation.
- `facet_enumeration` and `beneath_beyond` compute facets from vertices by two
  independently tested algorithms.
- `face_lattice`, `f_vector`, `is_vertex`, and `is_facet` expose exact derived
  combinatorics and membership predicates.

Unbounded polyhedra and cones are intentionally not represented by this
bounded type. Lower-dimensional hulls are valid, but their kernel facet
coordinates live in a reduced affine frame; Noether does not present those as
ambient inequalities.

## Example

```rust
use convex_hull::{convex_hull, q_point};

let square = vec![
    q_point(&[0, 0]),
    q_point(&[1, 0]),
    q_point(&[0, 1]),
    q_point(&[1, 1]),
];
let polytope = convex_hull(&square).unwrap();
assert_eq!(polytope.f_vector(), vec![1, 4, 4, 1]);
assert_eq!(polytope.n_facets(), 4);
```

## Build

```bash
cargo test -p convex-hull --release
cargo clippy -p convex-hull --lib -- -D warnings
cargo fmt -p convex-hull -- --check
scripts/bench-pin.sh cargo bench -p convex-hull --bench kernel
make validate-sage
```

The crate differential-tests its two exact V-to-H algorithms on standard and
generated families. The external parity lane compares all accepted V/H
construction directions, exact rational vertices, face vectors, incidence
counts, and predicates with SageMath's `Polyhedron`; its current corpus has 46
Sage-compared cases in dimensions 2 through 8. On the repository benchmark
host's pinned 96 MB-cache core, the reference medians are 30.8 ms for the
256-point parabola hull, 38.4 ms for the independent beneath-beyond path,
about 0.19 ms for H-to-V on the 6-cube, and 0.55 ms for the 5-cube face
lattice.

The crate is exact, GPL-clean, and contains no unsafe code. Primitive rational
arithmetic reaches dynamically linked LGPL FLINT through `flint-core`.

Licensed under the MIT License; see the repository-root `LICENSE`.
