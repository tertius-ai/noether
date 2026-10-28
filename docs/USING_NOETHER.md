# Using Noether

## Build from source

Noether is distributed as tagged source. There is no official PyPI package
or prebuilt wheel. To use the Python interface, install Rust 1.89+,
Python 3.10+, [uv](https://docs.astral.sh/uv/), and a C/C++ build toolchain.
On Linux the native build also needs Autoconf, Automake, Libtool, pkg-config,
Clang, and libclang; on macOS install the Xcode command-line tools and the
corresponding build utilities.

```bash
git clone --branch v0.3.2 --depth 1 https://github.com/tertius-ai/noether.git
cd noether
make sync
scripts/install-flint.sh
make engine
PYTHONPATH=python .venv/bin/python
```

The native installer builds GMP, MPFR, and FLINT under `.deps/flint/`.
To use an existing compatible FLINT 3.6 installation, set `FLINT_PREFIX`
to its prefix and omit the installer. `make engine` builds the Python
extension and generates its API indexes.

Run scripts from the checkout with `PYTHONPATH=python .venv/bin/python`.
This selects both the checkout's Python package and its development environment.
The package is imported as `noether`.

## First calculation

Compute the Smith normal form of an integer matrix:

```python
from noether import Z
from noether.linalg import Matrix

A = Matrix.from_rows(Z, [[2, 4], [6, 8]])
result = A.smith_normal_form()
assert result.invariant_factors == [2, 4]
result.verify()
```

The nonzero diagonal entries of the Smith form are 2 and 4. They describe
the cokernel of A: ℤ² → ℤ² as ℤ/2 ⊕ ℤ/4. The result retains the input;
`verify()` rechecks the claimed invariant factors.

Save the calculation as `first.py` and run it from the repository root:

```bash
PYTHONPATH=python .venv/bin/python first.py
```

## Coefficient contexts

The first argument of `Matrix.from_rows()` specifies its coefficients.
`Z` means integers and `Q` means rationals. Other contexts include residue
rings, finite fields, polynomial rings, and fixed-precision local fields.

```python
from noether import Q
from noether.commalg import PolynomialRing, ResidueRing
from noether.linalg import Matrix

rational = Matrix.from_rows(Q, [[1, 2], [3, 5]])
assert rational.det() == -1

mod12 = ResidueRing(12)
assert mod12(8) * mod12(2) == mod12(4)

polynomials = PolynomialRing(Q, ["x", "y"])
x, y = polynomials.generators()
assert (x + y)**2 == x**2 + 2*x*y + y**2
```

A context also records presentation data such as polynomial variables,
monomial order, or precision. Two matrices can be multiplied only when
their coefficient contexts agree and their dimensions fit. Changes of
context require an explicit conversion or map. See
[Contexts and precision](CONTEXTS_AND_PRECISION.md) for examples.

## Matrices

### Construction and arithmetic

```python
from noether import Z
from noether.linalg import Matrix, Vector

A = Matrix.from_rows(Z, [[1, 2], [3, 4]])
I = Matrix.identity(Z, 2)
zero = Matrix.zeros(Z, (2, 2))
v = Vector.from_values(Z, [1, -1])

assert A.shape == (2, 2)
assert A[0, 1] == 2
assert A @ I == A
assert A - A == zero
assert (A * 2)[1, 0] == 6
assert A.apply(v).to_values() == [-1, -1]
```

Indices start at zero; negative indices count from the end. `@` multiplies
matrices, while `*` multiplies by a scalar. Matrices are immutable.
`to_rows()` exports all entries to Python lists. Iteration also materializes
the rows, so use `A[i, j]` when only one entry is needed.

`from_rows()` rejects ragged rows. An empty list produces shape `(0, 0)`;
`Matrix.zeros(Z, (0, 3))` preserves a specified number of columns.
Use `SparseMatrix.from_triplets()` for sparse input; see
[Applying a sparse matrix](../examples/05_sparse_exact_operators.py).

### Systems over fields

`solve()` takes a right-hand-side vector and returns an affine solution
space over the same field:

```python
from noether import Q
from noether.linalg import Matrix, Vector

A = Matrix.from_rows(Q, [[1, 2, 3], [2, 4, 6]])
b = Vector.from_values(Q, [1, 2])
solutions = A.solve(b)
assert A @ solutions.particular == b
assert solutions.dimension == 2
solutions.verify()
```

The second equation is twice the first. There are two free directions,
and every solution is `solutions.particular` plus a linear combination
of `solutions.basis`. An inconsistent system raises `ValueError`.

`kernel()` returns the right kernel and `image()` the column image.
`rank()`, `rref()`, and `pluq()` also require supported field coefficients.
These field operations are available over rationals, finite fields,
number fields, and supported rational-function fields. Integer equations
and composite-modulus congruences use `solve_exact()`.

### Integer equations and congruences

```python
from noether import Z
from noether.commalg import ResidueRing
from noether.linalg import Matrix, Vector

integer = Matrix.from_rows(Z, [[6]]).solve_exact(Vector.from_values(Z, [9]))
assert not integer.is_consistent
assert integer.obstruction is not None
integer.obstruction.verify()

mod12 = ResidueRing(12)
congruence = Matrix.from_rows(mod12, [[8]]).solve_exact(Vector.from_values(mod12, [4]))
assert congruence.is_consistent
assert congruence.solutions is not None
congruence.verify()
```

The integer equation has no solution because 6 does not divide 9.
The congruence has four solutions modulo 12: 2, 5, 8, and 11.
Its solution module records one particular solution and all homogeneous
directions. An inconsistent `solve_exact()` result carries a divisibility
obstruction. See [Integer systems, congruences, and submodules](../examples/38_exact_linear_systems.py).

### Integer normal forms

Choose the normal form according to the information needed:

| Method | Result | Coefficients |
| --- | --- | --- |
| `smith_normal_form()` | Positive invariant factors, each dividing the next | Integers |
| `smith_normal_form_with_transforms()` | Smith form with unimodular change-of-basis matrices | Integers |
| `hermite_normal_form()` | Canonical row Hermite form | Integers |
| `howell_normal_form()` | Canonical row Howell form | Supported word-sized residue rings |
| `padic_smith_invariants()` | Invariant valuations visible below the precision cap | Fixed-precision p-adic integers |

For an m-by-n integer matrix of rank r, the Smith invariant factors
d₁, …, dᵣ describe the cokernel as ℤ^(m−r) ⊕ ⨁ᵢ ℤ/dᵢ.
Factors equal to 1 contribute trivial groups.

The default Smith algorithm is unconditional. The optional `algorithm="smooth"`
requires `smooth_bound` and can return a conditional result; inspect its
validity and recorded assumption. See [Smith normal form](../examples/04_smith_normal_form.py).

## Further calculations

The examples introduce other parts of the library:

| Subject | Start with |
| --- | --- |
| Finite fields | [Trace and norm](../examples/08_finite_field_extensions.py), [changing presentations](../examples/09_finite_field_isomorphisms.py) |
| Local fields | [Hensel lifting](../examples/10_padic_hensel_lifting.py), [ramification and precision](../examples/31_local_fields_and_precision.py) |
| Commutative algebra | [Ideals and elimination](../examples/15_groebner_twisted_cubic.py), [normalizing a cusp](../examples/36_affine_normalization.py) |
| Differential modules | [Differentials and cotangent dimensions](../examples/39_kahler_differentials.py) |
| Homological algebra | [Module reduction and resolutions](../examples/30_modules_and_resolutions.py), [exterior powers](../examples/40_exterior_algebra.py) |
| Combinatorics | [Graph symmetry](../examples/21_graph_symmetry.py), [Tutte polynomials](../examples/22_matroids_and_tutte.py) |
| Groups | [Permutation groups](../examples/19_permutation_groups.py), [conjugacy classes](../examples/20_conjugacy_classes.py) |
| Geometry | [Rational convex hulls](../examples/25_exact_geometry.py), [shortest lattice vectors](../examples/06_shortest_lattice_vectors.py) |
| Topology | [Persistence](../examples/24_homology_and_persistence.py), [cohomology operations](../examples/33_simplicial_cohomology_operations.py) |
| Higher categories | [The 2-oriental](../examples/28_steiner_categories.py) |

## Errors and verification

`ContextMismatchError` indicates incompatible coefficient contexts.
`UnsupportedOperationError` indicates an operation outside its supported
coefficient domain or hypotheses. Invalid dimensions and malformed input
commonly raise `ValueError`; indexing outside a matrix raises `IndexError`.

Some results expose `verify()`. Its arguments and scope depend on the
result: a Bézout identity checks supplied integers, a normal-form result
retains its matrix, and a bounded cohomology presentation verifies only
its stated range. Conditional results retain their assumptions.
Verification does not discharge an assumption supplied by the caller.

For example, `ProjectiveCompleteIntersection.assuming_smooth()` records
smoothness as a hypothesis. A bounded Eilenberg–Mac Lane cohomology
presentation states the degree through which it agrees with the full ring.
These restrictions are part of interpreting those results.

## Rust

Depend on the public facade at the required source tag:

```toml
[dependencies]
noether = { git = "https://github.com/tertius-ai/noether", tag = "v0.3.2" }
```

Rust builds need FLINT 3.6 available through `pkg-config`. For a non-system
installation, set `FLINT_PREFIX` and add its `lib/pkgconfig` directory to
`PKG_CONFIG_PATH`.

```rust
use noether::linalg::Matrix;
use noether::{Integer, Z};

let matrix = Matrix::from_rows(
    Z,
    [[Integer::from(2), Integer::from(4)],
     [Integer::from(6), Integer::from(8)]],
).unwrap();
let result = matrix.smith_normal_form();
assert_eq!(result.invariant_factors(), &[Integer::from(2), Integer::from(4)]);
result.verify().unwrap();
```

From the repository checkout, `make docs-rust` builds the public facade's
API reference under `target/doc/noether/`. `make docs-mpi` includes the
system-MPI API. This local reference is separate from the published user site.

## Inspecting the installed API

```python
import noether

caps = noether.capabilities()
assert caps.supports("linalg.Matrix.smith_normal_form")
description = noether.describe("linalg.Matrix.smith_normal_form")
print(description.signatures)
print(description.supported_rings)
```

`capabilities()` reports available operations in the installed build.
`describe()` returns signatures and documented restrictions, including
optional operations that may be unavailable locally. For Python signatures,
use the site's API reference or `help()` on the relevant type.

`make docs` builds the local user site at `target/noether-docs/index.html`.
[Architecture](ARCHITECTURE.md) describes native storage, concurrency,
MPI execution, and packaging for developers.
