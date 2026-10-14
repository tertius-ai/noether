# Using Noether

Choose a mathematical domain, construct an explicit coefficient context, then
use methods and standard operators on typed values. Backend names are normally
irrelevant: the same call uses a generic correctness floor or a specialized
FLINT/kernel representation according to the value's context and layout.

Noether currently publishes tagged source code, not a PyPI package or official
wheel. Build the Python package from the `v0.3.2` source tag and import its
`noether` package:

```bash
git clone --branch v0.3.2 --depth 1 https://github.com/tertius-ai/noether.git
cd noether
make sync
scripts/install-flint.sh
make engine
PYTHONPATH=python .venv/bin/python
```

The local build uses the pinned GMP, MPFR, and FLINT stack under
`.deps/flint/`. Advanced packagers can build audited wheel candidates for
CPython 3.10+ on manylinux2014 x86-64-v3, macOS x86-64 10.13+, and macOS arm64
11+, but Noether does not currently publish those candidates.

### Acceleration in portable builds

A locally built portable wheel is not a scalar-only build. It contains the
explicit SIMD paths used by the local exact-linear-algebra kernels:

| Kernel | Portable-build vector paths | Selection |
| --- | --- | --- |
| `m4rst`, dense matrices over `F_2` | AVX2 on x86-64; NEON on arm64 | Runtime feature check on x86-64; arm64 architecture baseline |
| `m4rste`, supported dense matrices over `F_{2^e}` | AVX2/GFNI on supported x86-64 cases; NEON on supported arm64 cases | Runtime feature and field-regime checks, with scalar fallback |
| `pwvla`, supported small residue rings | AVX2 on x86-64; NEON on arm64 | Runtime feature, coefficient-width, modulus, and size checks, with scalar fallback |

SIMD is internal dispatch, not a Python-package extra or Cargo opt-in. The
portable build deliberately does **not** apply `target-cpu=native` to the whole
engine: Linux x86-64 uses the documented `x86-64-v3` floor, macOS wheels use
their release architecture floors, and wider explicit kernels execute only
when the running CPU supports them. A source checkout can use
`make engine-native` for machine-specific compiler code generation.

This is separate from optional capabilities. The portable wheel profile excludes the
system-MPI `distributed` feature and validation-only libm4ri/libm4rie linkage;
those omissions do not disable the local SIMD kernels. MPI requires
`make engine-mpi`, while libm4ri and libm4rie remain GPL test and benchmark
oracles and never enter a conveyed wheel.

Rust users consume the facade from the versioned repository:

```toml
[dependencies]
noether = { git = "https://github.com/tertius-ai/noether", tag = "v0.3.2" }
```

Rust source builds need FLINT 3.6 available to `pkg-config`. For a non-system
installation, set `FLINT_PREFIX` and prepend
`$FLINT_PREFIX/lib/pkgconfig` to `PKG_CONFIG_PATH`; the repository's
`scripts/install-flint.sh` builds the pinned native stack.

## Python

Import from a public domain, never from `noether._engine`:

```python
from noether import Q, Z
from noether.commalg import PolynomialRing, PowerSeriesRing, SeriesTruncation
from noether.homalg import ChainComplex
from noether.linalg import Matrix

matrix = Matrix.from_rows(Q, [[1, 2], [3, 5]])
assert matrix.det() == -1

polynomials = PolynomialRing(Z, ["x", "y"])
x, y = polynomials.generators()
assert (x + y) ** 2 == x**2 + 2 * x * y + y**2

series = PowerSeriesRing(
    Z,
    ["x", "y"],
    SeriesTruncation.weighted_degree([1, 2], 6),
)
sx, sy = series.generators()
assert (sx**6).is_zero()
assert not (sx * sy).is_zero()

# Z --6--> Z has H_0 = Z/6. This path retains only the group invariants.
chains = ChainComplex.from_boundaries(Z, {1: Matrix.from_rows(Z, [[6]])})
group = chains.homology_invariants(0)
assert group.free_rank == 0
assert group.torsion_coefficients == [6]
```

The package ships annotations and `py.typed`; type checkers see contexts and
result types even though runtime classes cross PyO3.

### Exact systems and submodules

`Matrix.solve_exact` over `Z` or `Z/nZ` returns a typed result rather than
assuming field division.  A consistent result contains one particular
solution and canonical generators for the entire homogeneous solution module;
an inconsistent result contains a replayable divisibility obstruction.  This
is complete for composite moduli as well as prime moduli:

```python
from noether import Z
from noether.commalg import ResidueRing
from noether.linalg import Matrix, Vector

integer = Matrix.from_rows(Z, [[2, 4]]).solve_exact(Vector.from_values(Z, [6]))
assert integer.is_consistent
assert integer.solutions is not None
integer.verify()

z12 = ResidueRing(12)
modular = Matrix.from_rows(z12, [[8]]).solve_exact(Vector.from_values(z12, [4]))
assert modular.is_consistent
assert modular.solutions is not None
modular.verify()
```

Free-module maps and their kernel and image use the same exact scalar
contexts.  Their `KernelSubmodule` and `ImageSubmodule` results retain the map
and independently replay completeness; `.submodule` is the canonical module
of generators. `Submodule` canonicalizes arbitrary generating lists and provides
membership witnesses, sums, intersections, quotients and invariant data.  An
integer submodule additionally has a saturation; `finite_index()` fails
closed when the quotient has a free part.

```python
from noether.homalg import FreeModule, ModuleElement, Submodule

ambient = FreeModule(Z, 2)
twice_thrice = Submodule.from_generators(
    ambient,
    [
        ModuleElement(ambient, Vector.from_values(Z, [2, 0])),
        ModuleElement(ambient, Vector.from_values(Z, [0, 3])),
    ],
)
assert twice_thrice.finite_index() == 6
witness = twice_thrice.membership_witness(
    ModuleElement(ambient, Vector.from_values(Z, [6, 9]))
)
assert witness is not None
witness.verify()
assert twice_thrice.saturation().finite_index() == 1
```

In Rust the corresponding result types are `IntegerLinearSystemResult` and
`CongruenceSystemResult`, while the common subobject remains
`Submodule<IntegerRing>` or `Submodule<ResidueRing>`.  Proof-bearing values
validate when deserialized, and `verify()` independently replays their stated
equations and normal-form claims.

Kähler differentials are exact presented modules over affine algebras. The
universal derivation reduces into a canonical module normal form, while the
conormal evidence retains the chosen equations, their exact `I/I²`
presentation, and the Jacobian map:

```python
from noether import Q
from noether.commalg import PolynomialRing

ring = PolynomialRing(Q, ["x", "y"])
x, y = ring.generators()
cusp = ring.quotient(ring.ideal([y**2 - x**3]))
omega = cusp.kahler_differentials()

assert omega.differentiate(cusp(y**2 - x**3)).is_zero()
singular = cusp.cotangent_space_at([0, 0])
smooth = cusp.cotangent_space_at([1, 1])
assert (singular.embedding_dimension, smooth.embedding_dimension) == (2, 1)

complex = cusp.cotangent_complex()
assert (complex.amplitude.lowest_degree, complex.amplitude.highest_degree) == (-1, 0)
complex.verify()
```

`CotangentSpace.presentation_matrix` is `J(p)^T`, so its cokernel is the
cotangent fiber. The Python boundary exposes that field-valued presentation,
its exact rank, and the embedding dimension; the fully generic
`PresentedModule<C>` accessor remains available in Rust. A cotangent-complex
result is returned only when exact dimension and equation-count calculations
prove the retained equations form a regular sequence. Redundant equations
remain valid conormal evidence but fail that complete-intersection lane.

Finite strong Steiner categories live in their own domain. The directed atomic
basis is the stored value; cells, mapping categories, Gray products, and both
polygraphic homology and the Street source model are exact derived operations:

```python
from noether.category import AtomId, SteinerCategory

triangle = SteinerCategory.oriental(2)
first = triangle.atom(AtomId(1, 0))
second = triangle.atom(AtomId(1, 2))
assert first.compose_at(second, 0).top_atoms == [0, 2]
assert triangle.polygraphic_homology_invariants()[0].free_rank == 1
categorical = triangle.categorical_homology()
assert categorical.skeleton(3).verify()
assert categorical.component_poset_slice(2).poset.order == 6
```

`categorical_homology()` deliberately returns the lazy free stratified
simplicial commutative-monoid presentation on the Street nerve. It models
`Omega^infinity(H(N) smash Sigma^infinity_cat C_+)`, the free strict symmetric
monoidal infinity-category on `C`. Its zeroth homotopy Poset is usually
infinite but is canonically graded by particle number;
`component_poset_slice(r)` materializes the exact finite weight-`r` Poset.
`fundamental_poset(source_word, target_word)` takes the finite coproduct over
permutation sectors, while `particle_homotopy_poset(factors)` computes a fixed
higher sector as a cartesian product of source homotopy Posets. A requested
Street skeleton remains an independently bounded exact source presentation.

Eilenberg--Mac Lane spaces are symbolic rather than falsely finite simplicial
sets. The native exact homology playbook currently covers every `K(A, 1)` for
finitely generated abelian `A`, every `K(Z^r, 2)`, and `K(0, n)`:

```python
from noether.homalg import FinitelyGeneratedAbelianGroup
from noether.numthy import FiniteField
from noether.topology import EilenbergMacLaneSpace

c2_squared = FinitelyGeneratedAbelianGroup(0, [2, 2])
space = EilenbergMacLaneSpace(c2_squared, 1)
homology = space.integral_homology(3)
assert homology.groups[2].torsion_coefficients == [2]
assert homology.groups[3].torsion_coefficients == [2, 2, 2]
homology.verify()

# This is additive cohomology by universal coefficients, with the homology
# result retained as evidence; it is not yet a cup-product presentation.
cohomology = space.integral_cohomology(3)
assert cohomology.groups[2].torsion_coefficients == [2, 2]
cohomology.verify()

# In the same closed-form lanes, prime-field cohomology is a complete
# graded-commutative presentation with named generators and primary operations.
modular = space.modular_cohomology(FiniteField.prime(2))
assert [generator.degree for generator in modular.generators] == [1, 1]
assert modular.presentation.hilbert_function(3) == [1, 2, 3, 4]
modular.verify()

# Higher deloopings generally have infinitely generated cohomology.  The
# bounded Cartan--Serre lane therefore returns a different result type whose
# claim is explicitly limited to the requested degree.
k_z_3 = EilenbergMacLaneSpace(FinitelyGeneratedAbelianGroup(1), 3)
bounded = k_z_3.modular_cohomology_through_degree(FiniteField.prime(2), 16)
assert bounded.exact_through_degree == 16
assert [generator.degree for generator in bounded.generators] == [3, 5, 9]
assert [generator.display_name for generator in bounded.generators] == [
    "iota_3",
    "Sq^2(iota_3)",
    "Sq^4 Sq^2(iota_3)",
]
bounded.verify()

# Operation words are typed, validated values rather than display strings.
from noether.topology import AdmissibleCohomologyOperation

sq4_sq2 = AdmissibleCohomologyOperation.mod_two([4, 2])
assert sq4_sq2.display_name == "Sq^4 Sq^2"
assert bounded.generators[2].origin["operation"] == sq4_sq2
```

The bounded Cartan--Serre playbook covers `n >= 2` over every prime field for
every finitely generated abelian coefficient group; prime-to-`p` summands are
invisible. At `p = 2` it enumerates admissible Steenrod-square generators. At
odd `p` it enumerates admissible words in the Bockstein and reduced powers. A
free seed excludes a terminal primary Bockstein. A cyclic summand with
`p`-exponent greater than one contributes the two Cartan seeds: its fundamental
class and its higher Bockstein; both are modules over the quotient by the
terminal primary Bockstein. Extension fields still fail with a typed
unsupported-operation error. The finite presentation agrees with the full
ring through the recorded bound and makes no claim above it.

For a concrete finite abelian permutation group, the family completion route
first reconstructs its invariant factors and then retains the corresponding
`K(A,1)` result as exact evidence:

```python
from noether.groups import Permutation, PermutationGroup
from noether.numthy import FiniteField

c4 = PermutationGroup.from_generators(
    4, [Permutation.from_images([1, 2, 3, 0])]
)
complete = c4.complete_abelian_modular_cohomology(FiniteField.prime(2))
assert complete.invariant_factors == [4]
assert complete.completion_strategy == "finite_abelian_classifying_space"
complete.verify()
```

The same finite-group playbook also supplies exact integral truncations. Every
finite group has a scalable theorem-backed route through degree two, using
`H^2(BG; Z) = Ext(G_ab, Z)`. Higher requests use the normalized integral bar
resolution when it fits, retain Smith transformation witnesses, and expose
canonical additive groups and cup products:

```python
integral = c4.integral_cohomology(2)
assert integral.resolution_strategy == "universal_low_degree"
assert integral.additive_group(2).torsion_coefficients == [4]
assert integral.basis_cup_product(0, 0, 2, 0) == [1]
integral.verify()
```

Modular truncations accept the same typed Steenrod words. Instability and top
operations are evaluated in every resolution. Lower operations use the
normalized nerve directly, commute through the stable-elements lane, and use
closed cyclic-resolution formulas when applicable. Algebra-generator records
retain an operation, scalar, and decomposable correction whenever a primary
operation names the corresponding indecomposable class.

For a nonabelian group, first compute an exact resolution range and extract its
algebra approximation. `certify_symonds_parameter_criterion()` promotes it
only after proving parameterhood on every maximal elementary abelian subgroup,
proving finite module generation, and checking Symonds's numerical bound:

```python
s3 = PermutationGroup.from_generators(
    3,
    [
        Permutation.from_images([1, 0, 2]),
        Permutation.from_images([1, 2, 0]),
    ],
)
algebra = s3.modular_cohomology(3, 5).algebra_truncation()
complete = algebra.certify_symonds_parameter_criterion()
assert complete.completion_strategy == "symonds_parameter_criterion"
certificate = complete.symonds_parameter_certificate
assert certificate is not None
assert certificate.parameter_generator_indices == [1]
assert certificate.completion_degree_bound == 4
complete.verify()
```

An insufficient exact range or a resolution model without replayable
restriction evidence remains a typed failure; observing no new generators is
never treated as a completion proof.

Other deloopings fail with `UnsupportedOperationError`; an external CAS may
serve as a parity oracle, but is not a hidden fallback or correctness floor.

Simple-homotopy and anodyne equivalences are replayable zigzags, not Boolean
claims. A move may be a collapse or an expansion carrying the collapse that
reverses it; construction replays every move and JSON hydration verifies the
same source-to-target path.

## Rust

The corresponding Rust surface is the ordinary `noether` library:

```rust
use noether::linalg::Matrix;
use noether::homalg::ChainComplex;
use noether::{Integer, Q, Rational, Z};
use std::collections::BTreeMap;

let a = Matrix::from_rows(
    Q,
    [[Rational::from(1), Rational::from(2)],
     [Rational::from(3), Rational::from(5)]],
).unwrap();
assert_eq!(a.det().unwrap(), Rational::from(-1));

let boundary = Matrix::from_rows(Z, [[Integer::from(6)]]).unwrap();
let chains = ChainComplex::from_boundaries(Z, BTreeMap::from([(1, boundary)])).unwrap();
let group = chains.homology_invariants(0).unwrap();
assert_eq!(group.torsion_coefficients(), &[Integer::from(6)]);
```

Generic types are parameterized by context (`Matrix<R>`, `Vector<R>`,
`Polynomial<R>`). Context traits provide algebraic laws; specialized storage
is an internal enum and does not create a second public matrix family.

For integral homology, `homology_invariants()` returns only the canonical
`FinitelyGeneratedAbelianGroup`; its memory is proportional to the returned
invariant factors. `homology()` retains the adjacent differentials so that a
later `verify()` can independently recompute the claim. Over fields,
`betti_number()` is the corresponding invariants-only path, while `homology()`
also returns cycles, boundaries, and quotient representatives.

## Reusing exact finite actions

Permutation groups act exactly on points, subsets, and unordered pairs. Orbit
records include every member in stable order and recheck both closure and the
orbit-stabilizer identity:

```python
from noether.groups import Permutation, PermutationGroup

cyclic_four = PermutationGroup.from_generators(
    4, [Permutation.from_images([1, 2, 3, 0])]
)
pair_orbits = cyclic_four.orbits_on_pairs()
assert [orbit.size for orbit in pair_orbits] == [4, 2]
for orbit in pair_orbits:
    orbit.verify(cyclic_four)
```

When classifying many integral-line configurations against the same target,
compute its automorphism group once and pass it back explicitly:

```python
target_automorphisms = target.automorphism_group()
equivalence = source.equivalence_to(
    target, target_automorphisms=target_automorphisms
)
```

The faithful `signed_action` of an `IntegralLineAutomorphismGroup` retains the
kernel lost by its unoriented `line_action`. `lift_signed_action()` recovers the
unique verified unimodular matrix for one of its elements; this is useful when
a finite-group orbit computation must be transported back to lattice
coordinates. For a hot loop that needs only one external primitive line, use
`apply_signed_action_to_external_line()`; it reuses the group's exact cached
basis factorization without materializing the external-line orbit.

Minimum realization distinguishes an exact requested shell from fixed-rank
completion. `configuration.realize_minimum()` requires that the supplied lines
are the complete shell. `configuration.complete_minimum()` may admit additional
minimum lines only when their rank-one forms remain in the seed span. Both
operations return verified positive, negative, or inconclusive evidence; an
iteration limit never becomes a mathematical negative.

## Complete-intersection characteristic numbers

Intersection theory over products of projective spaces uses the exact Chow
ring `Z[h1,...,hr]/(h1^(n1+1),...,hr^(nr+1))`. Complete intersections require
an explicit regular-sequence or smoothness hypothesis; smooth-only tangent,
Todd, and Riemann--Roch operations fail closed on the weaker lane.

```python
from fractions import Fraction
from noether.commalg import ChowRing, ProjectiveCompleteIntersection

chow = ChowRing([1, 2])
h1, h2 = chow.generators()
curve = ProjectiveCompleteIntersection.assuming_smooth(
    chow, [[1, 1], [1, 2]]
)
answer = curve.line_bundle_euler_characteristic([2, -1])
answer.verify()
assert answer.hrr_integral == Fraction(answer.koszul_value)
```

The returned Riemann--Roch result retains two independently computed exact
integers: integration of `ch(O(a)) td(T_X)` in the Chow ring and the Koszul
inclusion--exclusion formula. Construction rejects the result if they differ.
See `examples/37_complete_intersection_theory.py` for an executable calculation.

## Discovery

Use canonical names when an LLM or tool needs to inspect the release surface:

```python
import noether

caps = noether.capabilities()
assert caps.supports("linalg.Matrix.smith_normal_form")
print(caps.local_concurrency.effective_threads)

description = noether.describe("linalg.Matrix.smith_normal_form")
print(description.signatures)
print(description.supported_rings)
print(description.example)
assert description.releases_gil
```

`capabilities()` describes what the installed artifact can call:
`supports("linalg.DistributedMatrix.rank")` is false in the ordinary CPU
wheel and true in an MPI build. `describe()` can still inspect an accepted
optional operation when it is not compiled, so an LLM can discover the build
requirement without mistaking that operation for an available one.

Types are discoverable the same way. For example, both of these show how a
poset is constructed from strict generating relations:

```python
from noether.combin import Poset
import noether

help(Poset)
print(noether.describe(Poset).example)

diamond = Poset(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
assert diamond.leq(0, 3)
assert diamond.covers.edge_count == 4
```

The accepted registry originates in Rust. `describe()` reads a generated
source-derived index packaged in the wheel, and `capabilities()` combines that
index with installed backend status. Generated JSON is disposable and not a
hand-maintained API contract. `NOETHER_NUM_THREADS` may lower the
affinity-aware process budget when it is set before first use. Discovery marks
direct substantial PyO3 boundaries with `releases_gil`; ordinary getters need
not detach.

From a source checkout, `make engine` regenerates the flat Python reference at
`API_REFERENCE.md`. `make docs` builds the public Rust facade with rustdoc and
joins it to the typed Python signatures, guarantees, backing kernels, and
executable examples at `target/noether-docs/index.html`. The generated site is
an untracked, disposable build artifact. Its landing page routes common
mathematical tasks to a canonical operation and a tested workflow; the guide,
contexts, architecture, serialized-evidence schemas, and domain indexes are
separate browsable views of the same source-derived surface. Domain entries
keep the Python form beside the corresponding Rust declaration and link back
to executable usages. The site also provides `api.json`, with every accepted
Python call and overload schema, and `engine-schemas.json`, with the narrower
Rust-owned cross-process transport registry. Class-local `to_json` and
`from_json` methods do not by themselves make a value part of that generic
transport registry. `make docs` fails when those projections disagree.

The default build uses the same configured CPU floor as the wheel. Developers
may opt into machine-specific code generation with `make engine-native` or
`make engine-mpi-native`; release evidence and portable correctness gates use
the baseline targets.

## Distributed F2 linear algebra

MPI support is an explicit build profile. A workstation or cluster login node
uses its active MPI installation:

```bash
make engine-mpi
mpirun -n 4 .venv/bin/python examples/distributed/f2_linear_algebra.py
```

On Slurm, load the site's compiler, Python, FLINT, and MPI modules, build on a
shared filesystem, and launch the same program with the site's supported
`srun` or `mpirun` path. The minimal `examples/distributed/slurm.sbatch`
template assigns one MPI process per task and one bounded Noether thread budget
per process; account, partition, module, launcher, and filesystem choices
remain site configuration.

For scaling and aggregate-memory evidence, first build the Rust benchmark
against that same MPI and submit the separate parameterized template:

```bash
cargo build --locked --release -p m4rst-distributed --features parallel --example benchmark
NOETHER_MPI_BENCH_OP=pluq \
NOETHER_MPI_BENCH_ROWS=8192 \
NOETHER_MPI_BENCH_COLS=8192 \
sbatch examples/distributed/slurm-benchmark.sbatch
```

The workload supports `matmul`, `rank`, `echelon`, `rref`, `pluq`, and `solve`;
it emits JSON with process-grid geometry, packed input/output sizes, elapsed
time, per-rank application-level communication calls/payload bytes, and per-rank
resident bytes and process peak RSS after the operation and verification,
exact-verification status, and a logical hash.
Store each MPI step in a separate JSON-lines file and pass the complete set
to `scripts/check-mpi-scale.py`. The checker proves node/rank participation,
affinity, packed-byte conservation, cross-grid hashes, exact solve evidence,
source/binary identity, requested scaling wins, and aggregate memory against
an explicit physical-RAM byte count. The exact release invocation is shown in
`crates/m4rst-distributed/README.md`.
The tracked template never chooses the site's account, partition, modules, or
storage paths.

For a clean acceptance handoff, the repository helper can bundle the exact
current commit, place an immutable checkout on any already-reachable SSH site,
and retrieve a machine-readable capability report:

```bash
ssh researcher@cluster.example true
scripts/slurm-handoff.py stage \
  --host researcher@cluster.example \
  --remote-root /shared/project/noether-handoff
```

The source tree must be clean and committed. The report is written under the
gitignored `target/hpc/` tree and records source identity, CPU/filesystem facts,
visible tools, MPI, Slurm partitions/associations, and relevant module queries.
Git verifies the bundle, detached checkout, and clean tree on the login node;
staging then writes an adjacent revision marker so compute nodes need no Git
installation to preserve the source-identity check.
The tiny remote probe supports the older system Python commonly retained on
HPC login nodes; this does not lower Noether's Python 3.10 runtime requirement.
The helper does not establish a VPN, answer an authentication challenge, choose
credentials, or modify SSH configuration; making the ordinary `ssh ... true`
preflight work is entirely caller-owned. Account, partition, modules, budgets,
and build choices likewise remain caller-owned.

Once those choices are known, the same helper can submit only the fixed
build-plus-benchmark acceptance graph. Every resource is explicit and the
aggregate requested node-hours must fit under a caller-supplied ceiling:

```bash
scripts/slurm-handoff.py submit \
  --site target/hpc/researcher_cluster.example/REVISION/site.json \
  --partition compute \
  --module gcc/12.2 --module openmpi/4.1 --module rust/1.89 \
  --node-count 1 --node-count 2 \
  --operation matmul --operation rank --operation echelon --operation rref \
  --operation pluq --operation solve \
  --mpi-launcher srun \
  --ntasks-per-node 2 --cpus-per-task 8 --memory 64G \
  --build-cpus-per-task 8 --build-memory 16G \
  --build-walltime 00:30:00 --run-walltime 00:30:00 \
  --max-node-hours 10
```

For rectangular matmul, `--tile-rows` and `--tile-cols` describe the left
input, while optional `--right-tile-rows` and `--right-tile-cols` describe the
right input. The output inherits the left tile rows and right tile columns, so
the handoff's balance gate evaluates the actual output-owner layout rather than
assuming both operands share one geometry.

Use `--no-modules` instead of `--module` only when the ambient site toolchain
is intentional. Select `srun` when the MPI module supports Slurm's direct PMI
launch, or `mpirun` when that is the site's supported allocation launcher. The
`mpirun` path uses the requested task count and binds each rank to its requested
CPUs. After loading the explicit modules, a conventional rustup installation
is recovered from `~/.cargo/env` when present; no shell profile or ambient
submitted environment is otherwise relied upon. `--dry-run` validates and
prints the complete job graph without contacting the site. The build runs
inside a one-node allocation, and each job set has an isolated Cargo target so
concurrent module/toolchain profiles cannot overwrite one another. Each
benchmark depends on its successful build. The result names and Slurm job IDs
are stored in a gitignored job-set JSON file printed by the command.

Collection is deliberately one-shot rather than a resident monitor:

```bash
scripts/slurm-handoff.py collect --job-set target/hpc/.../job-sets/JOB_SET.json
```

Exit status 3 means jobs are still incomplete and the JSON result reports
their current accounting states; rerun later. A failed job's available stdout
and stderr are copied into the local run directory and their bounded tails are
included directly in the JSON response, so failure diagnosis needs no separate
SSH command. Completed files are retrieved
atomically and passed through the scale gate, which checks exact source and
binary identity as well as topology, hashes, memory, and any requested speedup
or aggregate-memory claims.

The reference images are built and audited with:

```bash
make image-cpu
make check-image-cpu
make image-mpi
make check-image-mpi
```

`scripts/build-images.sh` derives each image tag from the package version in
`pyproject.toml`. The MPI image contains an MPI-enabled package and one matched
Open MPI runtime. It is a reproducible reference for ordinary Docker/OCI hosts,
not a promise of ABI compatibility with an HPC site's MPI, PMIx, or
interconnect. Cluster-native builds are therefore the normal HPC path.

`DistributedMatrix` is deliberately distinct from local `Matrix`. Every rank
executes the same collective sequence and retains only its packed local tiles.
`distribute(..., max_bytes=...)` is an explicit packed upload;
`collect(..., max_bytes=...)` is the only whole-matrix gather. Arithmetic,
factorization, solve, nullspace, inverse, and persistence do not gather an
operand implicitly.

## Evidence and errors

Results that carry mathematical obligations expose `verify()` and typed
validity or assumption data. Verification is explicit; serialization
round-trips revalidate semantic values. Unsupported contexts raise
`UnsupportedOperationError`, mismatched contexts raise `ContextMismatchError`,
and failed evidence raises `VerificationError`.

External CAS programs are validation or optional capabilities, not hidden
fallbacks. A skipped oracle does not verify a result.
