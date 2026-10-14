# lattice-enum

Safe-Rust **short-vector / closest-vector enumeration** kernel over
positive-definite integral Gram matrices.

Given a symmetric positive-definite **integer Gram matrix** `G`, enumerate

* `svp_enumerate(G, bound)` — every `v ∈ Zⁿ` with `vᵀGv ≤ B`, one
  representative per ± pair, canonical sign (first nonzero coord positive);
* `cvp_enumerate(G, target)` — the integer vector(s) closest to `target`
  under the `G`-inner product;
* `representation_vectors_collect(G, k)` — every `v` with `vᵀGv = k`
  (theta-series support);
* `gaussian_heuristic_bound(G)` / `enumeration_radius_set(G, count)` —
  Gaussian-heuristic sizing utilities;
* `pruned_enumeration(G, bound, pruning)` — linear (Gama–Nguyen–Regev) /
  extreme (Aono–Wang) pruned enumeration.

`noether::linalg::Lattice` is the public facade caller. It owns the
mathematical presentation and typed results; this crate owns the bounded
enumeration algorithms and remains directly usable from Rust.

## Correctness and validation

The bounded Schnorr--Euchner walk uses an exact FLINT-rational
square-root-free Gram--Schmidt factorization. Its positive-definiteness
verdict, partial norms, centers, and integer ranges are exact. Candidate norms
are independently rechecked with arbitrary-precision FLINT integers. Floating
point is limited to the optional Babai seed and Gaussian work estimates; exact
bounded SVP and CVP still run when no finite floating projection exists.

Rust differential tests exhaust small integer boxes for SVP and rational CVP,
including ill-conditioned forms and targets beyond `f64` unit resolution. The
Sage validation corpus compares bounded short-vector sets exactly on A2, A3,
diagonal and dense forms, and E8. Sage does not promise a finite,
all-closest-vectors CVP result, so CVP parity uses the exhaustive exact Rust
oracle instead of overstating the contract of Sage's iterator.

## Performance

The diagonal path uses word arithmetic when the inputs fit and FLINT integers
otherwise; non-diagonal forms use the general exact rational recursion. On the
project benchmark machine, pinned to CPU 0 and its 96 MB LLC group, current
release medians are approximately 9.80 us for the rank-8 diagonal minimum
shell, 271 us for rank-8 half-integral diagonal CVP, 25.4 ms for E8 through
norm 6, and 4.23 us for the condition-`~10^18` rank-2 corpus. These figures
describe the checked-in Criterion workloads, not a cross-machine promise.

## Discipline

The algorithm crate uses `#![forbid(unsafe_code)]` and has **no GPL runtime
dependency**. Exact Gram--Schmidt and verification arithmetic uses
`flint_core::{Integer, Rational}` over dynamically linked LGPL FLINT. It ships
in every facade profile, including the conveyed no-GPL wheel.

```bash
cargo test -p lattice-enum --release
NOETHER_VALIDATE_SAGE=1 PYTHONPATH=python .venv/bin/python -m pytest \
  validation/parity/linalg/test_lattice_enumeration_parity.py
scripts/bench-pin.sh cargo bench -p lattice-enum --bench kernel
```
