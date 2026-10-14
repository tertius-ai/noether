# module-core

`module-core` owns the packed module-term and recursively nested Schreyer-order
primitives used by Noether's module Gröbner and resolution engines. It reuses
`poly-core` monomial arenas, keeps component positions explicit, and has no
public-facade or Python dependency.

Its direct consumers are `groebner-kernel`, `resolution-kernel`, and the
`noether::homalg` adapters. The exact generic comparison implementation is the
correctness floor for coefficient-specialized module kernels.

Module terms always reference a caller-owned `poly-core::MonomialArena`.
Position ties are stable, and a Schreyer order retains the exact ordered
leading frame used to construct it; neither the arena nor the frame is inferred
or reordered at an algorithm boundary. Exercise the public contract with:

```bash
cargo test -p module-core --release
cargo clippy -p module-core --all-targets -- -D warnings
```
