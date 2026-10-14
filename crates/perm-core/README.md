# perm-core

`perm-core` is Noether's safe-Rust permutation-group kernel. It owns checked
permutations, deterministic and seeded-verified Schreier-Sims construction,
base/strong generating sets, stabilizer chains, orbits, membership,
enumeration, conjugacy classes, subgroup backtracking, and structural tests.

`noether::groups` curates these same identities. Graph, poset, and matroid
symmetry operations return its `PermutationGroup`; GAP is an external
validation oracle rather than a runtime naming authority.

Permutations act on `0..degree` on the right. Randomized construction and
uniform sampling require an explicit seed; randomized BSGS output is verified
before it is returned.

Group and permutation orders are unbounded. The small `BigUInt` implementation
owns only the operations this domain needs, keeping the crate pure Rust and
free of native dependencies. Transversal representatives and their cached
inverses share immutable storage across derived stabilizer groups; membership
and sampling reuse composition buffers rather than allocating at every chain
level. Sylow construction uses the normalizer quotient for a bounded general
group and an iterated-wreath construction for a proved full symmetric group.
Conjugacy classes retain their exact size and canonical representative;
materializing the complete member list is lazy.

```rust
use perm_core::{BigUInt, Perm, PermGroup};

let s4 = PermGroup::from_generators(
    4,
    &[
        Perm::from_images(&[1, 0, 2, 3]).unwrap(),
        Perm::from_images(&[1, 2, 3, 0]).unwrap(),
    ],
).unwrap();
assert_eq!(s4.order(), BigUInt::from(24_u64));
```

## Correctness and oracle coverage

The primary GAP corpus compares 13 groups from degree 8 through 512: symmetric,
alternating, cyclic, dihedral, PSL(2,p), affine, small-base, embedded, and
large-degree actions. It compares group order, membership, two point orbits,
point-stabilizer order, and seeded stabilizer-chain orbit sizes. The Noether
side additionally verifies transversal and Schreier-tree witnesses,
orbit--stabilizer, Schreier-generator closure, iteration cardinality, and
seeded samples.

A second GAP corpus covers S4, A4, D4, C4, and V4 and compares transitivity,
solvability, nilpotency, perfectness, conjugacy-class/centralizer sizes,
symmetric-group normalizers, subgroup normalizers, and set stabilizers.
Backtrack witnesses are sift-verified in the implementation and tested by
property-based conjugation/normalizer laws. Random groups exercise
composition, inversion, generator membership, sampling membership, and
orbit--stabilizer. An unavailable GAP process is a skip unless
`NOETHER_VALIDATE_GAP=1`, in which case validation fails closed.

## Performance

On the project benchmark machine, pinned to CPU 0 and its 96 MB LLC group,
current release medians are approximately 140 us for constructing S12, 1.70 ms
for D512, 69 ns for PSL(2,31) membership, 3.50 us for the three-point
stabilizer in S12, 0.79 us for a seeded S32 sample, 6.3 us for a Sylow
2-subgroup of S8, and 46 us for the compact exact conjugacy classes of S8.
These are the checked-in Criterion workloads, not cross-machine promises.

```bash
cargo test -p perm-core --release
NOETHER_VALIDATE_GAP=1 make validate
scripts/bench-pin.sh cargo bench -p perm-core --bench kernel
```

The crate contains no unsafe code and no native dependency.
