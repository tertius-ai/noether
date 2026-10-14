# perm-core guidance

Follow the repository and `crates/AGENTS.md` rules. `perm-core` owns checked
permutations, BSGS/stabilizer-chain algorithms, subgroup backtracking,
conjugacy, and deterministic seeded sampling. `noether::groups` curates these
same identities.

## Invariants

- Permutations act on the right over `0..degree`.
- A returned randomized BSGS has passed the deterministic verification sweep.
- Sampling is exactly uniform over transversal choices and records its seed.
- Group order uses the crate's unbounded unsigned integer.
- Malformed images, degree mismatches, and bounded enumeration overflow return
  typed errors.

The crate currently contains no unsafe code. GAP is a validation oracle and
`nTj` labels are external names, never internal object identity.

## Commands

```bash
cargo test -p perm-core --release
cargo clippy -p perm-core --all-targets -- -D warnings
NOETHER_VALIDATE_GAP=1 make validate
scripts/bench-pin.sh cargo bench -p perm-core --bench kernel
```
