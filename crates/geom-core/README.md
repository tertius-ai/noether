# geom-core

`geom-core` is Noether's safe-Rust Euclidean geometry kernel. It owns point
clouds and finite metrics, filtered exact orientation and in-sphere signs,
circumspheres and minimum enclosing balls, and Delaunay/regular
triangulations.

Noether curates these values in `noether::geometry`; topology constructors
consume the resulting point clouds, metrics, and triangulations. Abstract
simplicial objects and persistence remain in `noether::topology`.

Predicate decisions use a fast floating filter and an exact FLINT-rational
fallback. Construction coordinates and filtration scales remain `f64` where a
rounded value does not control combinatorial branching. Malformed inputs
return errors; constructions that do not exist return `None`.

```rust
use geom_core::{delaunay, in_sphere, PointCloud, Sign};

let cloud = PointCloud::new(&[
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
    vec![0.0, 1.0],
]).unwrap();
assert_eq!(delaunay(&cloud).unwrap().n_cells(), 2);

let triangle: [&[f64]; 3] = [&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]];
assert_eq!(in_sphere(&triangle, &[1.0, 1.0]), Sign::Zero);
```

```bash
cargo test -p geom-core --release
scripts/bench-pin.sh cargo bench -p geom-core --bench kernel
```

The oracle surface matches the implementation surface: orientation and
in-sphere signs in dimensions two and three are compared sign-for-sign with
Shewchuk's independent `robust` predicates, including exactly degenerate
coarse grids; planar Delaunay cells and forced cell counts are compared with
`spade` on generic and cocircular inputs. Higher dimensions are checked by the
exact empty-sphere definition, completeness, orientation, and fast/general
path differentials. `robust` and `spade` are development-only and never ship.

On the repository benchmark host's pinned 96 MB-cache core, filtered planar
orientation and in-circle calls have reference medians of 16.0 ns and 45.4
ns. The 300-site generic planar triangulation is 0.69 ms, the 400-site
perturbed grid is 2.22 ms, and a 1,000-site three-dimensional minimum
enclosing ball is 36.3 µs. The pinned comparison program reports the generic
planar triangulator at roughly 12× `spade`, the cost of general-dimensional
exact branching rather than a specialised two-dimensional data structure.

The crate contains no unsafe code.
