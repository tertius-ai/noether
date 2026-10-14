// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Property-based correctness tests for the Schreier-Sims BSGS kernel. These
//! exercise group-theory invariants over random permutation groups:
//!
//!   * the orbit-stabilizer theorem `|Orb(p)|·|Stab(p)| = |G|` at every point;
//!   * `sift` of a member returns the identity residue and reports membership;
//!   * a uniform random element is a member of the group;
//!   * the defining generators are members.
//!
//! Random permutations are generated without an RNG dependency: a vector of
//! random keys is argsorted into a permutation of `0..degree`. Pure Rust —
//! `cargo test -p perm-core --release --test proptest`.

use perm_core::{BigUInt, Perm, PermGroup, Point};
use proptest::prelude::*;

/// A uniformly-ish random permutation of `0..degree`, built by argsorting a
/// vector of random keys (ties broken by the stable sort — still a bijection).
fn arb_perm(degree: usize) -> impl Strategy<Value = Perm> {
    prop::collection::vec(any::<u32>(), degree..=degree).prop_map(move |keys| {
        let mut idx: Vec<Point> = (0..degree as Point).collect();
        idx.sort_by_key(|&i| keys[i as usize]);
        Perm::from_images(&idx).expect("argsort yields a valid permutation")
    })
}

/// A random permutation group on degree 2..=6 with 1..=3 generators.
fn arb_group() -> impl Strategy<Value = PermGroup> {
    (2usize..=6, 1usize..=3).prop_flat_map(|(degree, ngens)| {
        prop::collection::vec(arb_perm(degree), ngens..=ngens)
            .prop_map(move |gens| PermGroup::from_generators(degree, &gens).expect("valid group"))
    })
}

fn arb_pair() -> impl Strategy<Value = (Perm, Perm)> {
    (0_usize..=64).prop_flat_map(|degree| (arb_perm(degree), arb_perm(degree)))
}

fn arb_group_extension() -> impl Strategy<Value = (Vec<Perm>, Perm)> {
    (2usize..=6, 1usize..=3).prop_flat_map(|(degree, ngens)| {
        (
            prop::collection::vec(arb_perm(degree), ngens..=ngens),
            arb_perm(degree),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_iters: 64, .. ProptestConfig::default() })]

    #[test]
    fn composition_matches_right_action((left, right) in arb_pair()) {
        let product = left.compose(&right).unwrap();
        for point in 0..left.degree() as Point {
            prop_assert_eq!(product.apply(point), right.apply(left.apply(point)));
        }
    }

    #[test]
    fn inverse_is_two_sided((permutation, _) in arb_pair()) {
        let inverse = permutation.inverse();
        prop_assert!(permutation.compose(&inverse).unwrap().is_identity());
        prop_assert!(inverse.compose(&permutation).unwrap().is_identity());
    }

    #[test]
    fn big_uint_arithmetic_matches_u128(left in any::<u64>(), right in any::<u64>()) {
        let left_big = BigUInt::from(left);
        let right_big = BigUInt::from(right);
        prop_assert_eq!(left_big.add(&right_big).to_decimal(),
            (left as u128 + right as u128).to_string());
        prop_assert_eq!(left_big.mul(&right_big).to_decimal(),
            (left as u128 * right as u128).to_string());
    }

    #[test]
    fn big_uint_word_division_matches_u128(
        high in any::<u64>(), low in any::<u64>(), divisor in 1_u64..=u64::MAX
    ) {
        let mut value = BigUInt::from(high);
        value.mul_assign_u64(u64::MAX);
        value = value.add(&BigUInt::from(low));
        let reference = high as u128 * u64::MAX as u128 + low as u128;
        let (quotient, remainder) = value.divmod_u64(divisor);
        prop_assert_eq!(quotient.to_decimal(), (reference / divisor as u128).to_string());
        prop_assert_eq!(remainder, (reference % divisor as u128) as u64);
    }

    /// Orbit-stabilizer theorem: `|Orb(p)| · |Stab(p)| = |G|` for every point.
    #[test]
    fn orbit_stabilizer_product_is_order(g in arb_group()) {
        for p in 0..g.degree() as Point {
            let os = g.orbit_stabilizer(p).expect("orbit_stabilizer");
            let product = BigUInt::from(os.orbit.len() as u64).mul(&os.stabilizer.order());
            prop_assert_eq!(product, g.order(), "|Orb|·|Stab| = |G| at point {}", p);
        }
    }

    /// A uniform random element is a member, and `sift` confirms it with an
    /// identity residue.
    #[test]
    fn random_element_sifts_to_identity(g in arb_group(), seed in any::<u64>()) {
        let r = g.uniform_random_element(seed);
        prop_assert!(g.contains(&r.element).expect("contains"), "random element ∈ G");
        let (residue, is_member) = g.sift(&r.element).expect("sift");
        prop_assert!(is_member, "sift reports membership for a group element");
        prop_assert!(residue.is_identity(), "sift residue of a member is the identity");
    }

    /// The defining generators are members of the group they generate.
    #[test]
    fn generators_are_members(g in arb_group()) {
        for gen in g.generators() {
            prop_assert!(g.contains(gen).expect("contains"), "generator ∈ G");
        }
    }


    /// Incremental generator discovery produces the same verified group as a
    /// fresh construction from the accumulated list.
    #[test]
    fn adjoin_generator_matches_fresh_construction((generators, extra) in arb_group_extension()) {
        let degree = extra.degree();
        let initial = PermGroup::from_generators(degree, &generators).expect("initial group");
        let extended = initial.adjoin_generator(&extra).expect("incremental extension");
        let mut all_generators = generators;
        all_generators.push(extra);
        let rebuilt = PermGroup::from_generators(degree, &all_generators).expect("fresh group");

        prop_assert_eq!(extended.order(), rebuilt.order());
        for generator in all_generators {
            prop_assert!(extended.contains(&generator).expect("membership"));
        }
    }
}
