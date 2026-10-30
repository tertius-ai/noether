# noether

`noether` is the public Rust library for deterministic scientific mathematics.
It owns the mathematical types, traits, operations, semantic identities, and
accepted release surface. It has no Python or PyO3 dependency.

The public API is organized into nine mathematical domains:

- `category` — strict higher categories, Steiner complexes, and nerves;
- `combin` — graphs, posets, matroids, tableaux, crystals, and enumeration;
- `commalg` — residue rings, polynomials, series, ideals, and schemes;
- `geometry` — exact point configurations, triangulations, and polytopes;
- `groups` — permutations and finite permutation groups;
- `homalg` — chain complexes and homology;
- `linalg` — generic matrices, vectors, sparse matrices, and exact forms;
- `numthy` — finite, local, and number-field arithmetic;
- `topology` — simplicial objects, filtrations, and persistence.

Coefficient-bearing objects are parameterized by mathematical contexts. For
example, `Matrix<R>` has one public identity for every `R: Ring`: generic Rust
algorithms provide the correctness floor, while whole-object dispatch selects
FLINT and the retained kernel crates when their supported regime applies.
Backend choice is private and does not change the result type.

```rust
use noether::linalg::Matrix;
use noether::{Integer, Z};

let matrix = Matrix::from_rows(
    Z,
    [[Integer::from(1), Integer::from(2)],
     [Integer::from(3), Integer::from(4)]],
).unwrap();
assert_eq!(matrix.matmul(&Matrix::identity(Z, 2)).unwrap(), matrix);
```

`accepted_public_types`, `accepted_operations`, and
`accepted_public_values` expose the release index. `SemanticType` and the
schema registry supply stable machine-readable identities for accepted values.
The private `noether-python` crate translates this Rust surface into the typed
Python package.

The facade is distributed from the versioned repository rather than
crates.io:

```toml
[dependencies]
noether = { git = "https://github.com/tertius-ai/noether", tag = "v0.3.2" }
```

Source consumers need FLINT 3.6 available to `pkg-config`. For a non-system
installation, set `FLINT_PREFIX` and prepend
`$FLINT_PREFIX/lib/pkgconfig` to `PKG_CONFIG_PATH`.

From the repository root:

```bash
cargo test -p noether --release
cargo doc -p noether --no-deps
scripts/bench-pin.sh cargo bench -p noether
```

`scripts/install-flint.sh` installs the pinned native stack for repository
builds.

`commalg::HyperellipticCurve` owns a smooth projective model of
`y²+h(x)y=f(x)` over a finite field of odd characteristic. Square completion
and squarefreeness are checked in Rust; the model retains its degree-two
weighted cover of P1 and the residue degrees of its points at infinity.
Integral Kedlaya--Harrison reduction computes H1 Frobenius for arbitrary
positive genus, both degree parities, nonmonic models, and extension fields,
within explicit allocation bounds. Exhaustive counting is a validation route.

```bash
cargo run -p noether --release --example kedlaya -- 3 0,1,0,0,0,0,0,0,0,1
cargo test -p noether --release --test hyperelliptic
# Requires the validation-only PARI/GP executable:
cargo test -p noether --release --test hyperelliptic -- --include-ignored
```

The comma-separated coefficients are ascending. An optional final argument
sets the requested p-adic precision; insufficient precision is rejected.
`HyperellipticFrobeniusEvidence` records both Frobenius matrices, the binomial
truncation and the complete precision budget. Verification and deserialization
replay the Rust computation. `FrobeniusPolynomial` and `ZetaFunction` retain the
curve or surface evidence appropriate to their cohomological degree.
