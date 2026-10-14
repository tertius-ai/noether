# m4rst guidance

Follow the repository and `crates/AGENTS.md` rules. `m4rst` owns bit-packed
dense `F_2` matrices. Noether calls it for dense characteristic-two matrix
arithmetic; `m4rste` calls it for bit-plane products.

## Kernel invariants

- Padding bits outside the declared column count are always zero.
- `matmul_into` and decomposition `_into` forms fully overwrite reusable
  outputs and are the allocation-sensitive implementation owners.
- Dispatcher thresholds are changed only with pinned paired benchmarks and
  differential correctness at the affected branch.
- Parallel output is bit-identical for every thread count.

Unsafe code is permitted only in `src/view.rs`, `src/matmul/naive.rs`, and
`src/simd.rs`. Each block documents alignment, bounds, and aliasing. SIMD is
runtime-dispatched over scalar, differentially tested, and benchmarked.

## Commands

```bash
cargo test -p m4rst --release --features parallel
cargo test -p m4rst --release --features validate-libm4ri
cargo clippy -p m4rst --all-targets --features parallel -- -D warnings
scripts/bench-pin.sh cargo bench -p m4rst --bench matmul --features parallel
```

libm4ri is a validation oracle only and must never enter a conveyed dependency
closure.
