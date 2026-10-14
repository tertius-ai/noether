# pwvla guidance

Follow the repository and `crates/AGENTS.md` rules. `pwvla` owns dense packed
arithmetic over native `Z/2^n` cells and small `Z/p^n` cells. Noether currently
calls the native power-of-two route; odd-prime packed routing needs a pinned
win over FLINT before promotion.

## Invariants

- `Mat<T>` arithmetic is wrapping arithmetic in the represented `Z/2^n`.
- `MatZpn` values remain reduced and packing guard bits remain valid.
- SIMD and Strassen paths are identical to scalar Barrett/reference products.
- Allocation-reuse forms overwrite their destinations.
- Rayon remains optional and serial behavior is always present.

Unsafe code is confined to `src/neon.rs`, `src/neon_wide.rs`, `src/avx2.rs`,
and `src/zpn.rs`. Noether's conveyed x86-64 artifacts use the workspace's v3
floor. Architecture paths still use runtime dispatch over an always-present
scalar fallback so the crate remains safe for standalone consumers with a
lower compilation target and wider future paths remain optional.

## Commands

```bash
cargo test -p pwvla --release --features parallel
cargo clippy -p pwvla --all-targets --features parallel -- -D warnings
scripts/bench-pin.sh cargo bench -p pwvla --bench matmul --features parallel
scripts/bench-pin.sh cargo run --release -p pwvla \
    --features flint --example bench_z2n_flint
```
