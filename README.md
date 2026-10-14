# Noether

Noether is a Rust-first library for deterministic scientific mathematics, with
a typed Python interface. Rust owns the mathematical objects, algorithms,
dispatch, and verification evidence; Python provides ergonomic domain imports
without reimplementing the mathematics. It is an academic project by Benjamin
Antieau and Deven Manam.

The same nine domains organize both interfaces: `category`, `combin`,
`commalg`, `geometry`, `groups`, `homalg`, `linalg`, `numthy`, and `topology`.
Coefficient-bearing objects carry an explicit context, so a `Matrix<R>` has one
public identity while private dispatch selects FLINT or a retained Rust kernel
when its representation and measured regime apply.

## Install from the tagged source

Noether currently publishes source code only. There is no official PyPI
package, wheel, container image, or hosted computation service. Build the
`v0.3.2` source tag with Rust 1.89+, Python 3.10+,
[uv](https://docs.astral.sh/uv/), and a C/C++ build toolchain:

```bash
git clone --branch v0.3.2 --depth 1 https://github.com/tertius-ai/noether.git
cd noether
make sync
scripts/install-flint.sh
make engine
```

The installer builds the pinned GMP, MPFR, and FLINT stack under the ignored
`.deps/flint/` directory. To reuse a compatible FLINT 3.6 installation, set
`FLINT_PREFIX` to its prefix and omit the installer. The built package is
available from a source checkout with `PYTHONPATH=python`.

The Rust facade is the workspace crate at `crates/noether`. Rust users can
build it directly from the checkout:

```bash
cargo build -p noether
cargo run -p noether --example readme
```

## First calculation

```python
from noether import Z
from noether.linalg import Matrix

matrix = Matrix.from_rows(Z, [[2, 4], [6, 8]])
result = matrix.smith_normal_form()
assert result.invariant_factors == [2, 4]
result.verify()
```

Save that as `first.py`, then run:

```bash
python first.py
```

Noether has no Python runtime dependencies. Its public modules include precise
stubs and `py.typed`; `noether._engine` is private PyO3 transport.

## Find the right operation

Python exposes the Rust-owned release index at runtime:

```python
import noether

caps = noether.capabilities()
assert "linalg.Matrix.smith_normal_form" in caps.operations

op = noether.describe("linalg.Matrix.smith_normal_form")
print(op.signatures, op.supported_rings, op.evidence, op.errors)
```

For a guided introduction, read
[Using Noether](https://github.com/tertius-ai/noether/blob/main/docs/USING_NOETHER.md).
[Contexts and precision](https://github.com/tertius-ai/noether/blob/main/docs/CONTEXTS_AND_PRECISION.md)
explains rings, fields, maps, and fixed precision; the
[architecture paper](https://github.com/tertius-ai/noether/blob/main/docs/ARCHITECTURE.md)
explains the Rust/Python/kernel boundary. The 40
[executable examples](https://github.com/tertius-ai/noether/tree/main/examples) are compact
end-to-end workflows.
Build the unified local Python and Rust reference with `make docs`, then open
`target/noether-docs/index.html`.

## Correctness contract

A public mathematical path starts with hand-verifiable cases, states its
hypotheses and unsupported boundary, and must fail closed when they are not met.
Property tests cover algebraic laws, conversions, and specialized-versus-
generic agreement. Independent CAS or library oracles are used where
available; a skipped required oracle is not a pass. Evidence-bearing results
replay through `verify()`, and release candidates receive independent review.
A correctness or performance fix keeps its triggering regression and
adds a nearby case that cannot rely on the same shortcut.

Noether is not a compatibility clone of Sage, Magma, GAP, or Macaulay2. The
accepted surface reported by `capabilities()` is the support boundary.
Unavailable mathematics returns a typed error rather than a guessed result.
Optional external CAS subprocesses are user-supplied capabilities; the native
implementation remains the correctness floor and conveyed artifacts link no
GPL library.

## Develop

```bash
make help          # complete command index
make test          # acceptance smoke tests
make test-all      # local pre-PR gate
make test-stress   # expanded properties and bounded slow/ignored tests
make validate      # explicitly enabled external-CAS parity lanes
make candidate-linux # build an unpublished manylinux wheel candidate
make candidate-macos # build an unpublished native wheel candidate
make coverage      # Python and Rust line-coverage reports
```

Run benchmarks through `scripts/bench-pin.sh`; measurements without recorded
CPU placement are not performance evidence. `make engine-native` is the
explicit workstation-native build. Ordinary Linux x86-64 builds use the
`x86-64-v3` floor. A locally built portable wheel contains the runtime-dispatched
AVX2/GFNI or NEON paths in `m4rst`, `m4rste`, and `pwvla`; it omits
whole-engine `target-cpu=native` specialization, not those local SIMD kernels.
MPI support is a separate system-MPI build (`make engine-mpi`) and reference
image (`make image-mpi`).

## Repository map

```text
crates/noether/          public Rust facade
crates/noether-python/   private PyO3 transport
crates/*                 retained kernels and support crates
python/noether/          typed Python package
examples/                executable mathematical workflows
validation/              tests, oracle parity, corpus, and benchmarks
docs/                    three living design and usage documents
docker/                  CPU/MPI reference image definition
scripts/                 build, audit, benchmark, and release tools
licenses/                conveyed native-library notices and texts
```

Contributor and agent rules begin in
[AGENTS.md](https://github.com/tertius-ai/noether/blob/main/AGENTS.md), and the
human contribution workflow is in
[CONTRIBUTING.md](https://github.com/tertius-ai/noether/blob/main/CONTRIBUTING.md).
Report suspected vulnerabilities privately as described in
[SECURITY.md](https://github.com/tertius-ai/noether/blob/main/SECURITY.md).
Noether source is MIT-licensed. Locally built wheels dynamically link
replaceable LGPL FLINT, GMP, and MPFR libraries; exact conveyance and relinking terms are in
[the native-library notice](https://github.com/tertius-ai/noether/blob/main/licenses/NATIVE_LIBRARIES.txt).
