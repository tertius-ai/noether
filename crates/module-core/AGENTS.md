# module-core guidance

Follow the repository and `crates/AGENTS.md` rules. This crate owns sparse
module-term ordering and Schreyer-frame primitives. Noether owns public free
modules, module maps, presentations, mathematical order values, evidence, and
dispatch.

Module terms always refer to a caller-owned `poly-core::MonomialArena`.
Component zero precedes component one when a monomial tie requires a position
comparison. Schreyer comparisons must retain the exact ordered leading frame;
they may not infer or reorder it.

## Commands

```bash
cargo test -p module-core --release
cargo clippy -p module-core --all-targets -- -D warnings
```
