// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Every public operation exercised against
//! independently-known reference values for the standard small groups
//! (symmetric, alternating, dihedral, cyclic).

use perm_core::{
    base_and_strong_generating_set, element_membership_test, group_order, iterate_elements, orbit,
    orbit_stabilizer, pointwise_stabilizer, schreier_generators, schreier_tree, stabilizer_chain,
    transversal, uniform_random_element, BigUInt, Perm, PermGroup,
};

mod common;
use common::{alternating, cyclic, dihedral, symmetric};

#[test]
fn symmetric_order_matches_factorial() {
    for n in 1..=7 {
        let g = symmetric(n);
        let mut expected = BigUInt::one();
        for k in 1..=n as u64 {
            expected.mul_assign_u64(k);
        }
        assert_eq!(g.order(), expected, "|S_{n}|");
    }
}

#[test]
fn symmetric_certificate_is_invariant_under_relabeling() {
    // The cycle is (0 2 4 6 1 3 5), and the transposition swaps two
    // adjacent points in those cycle coordinates. This proves S_7 without
    // depending on the conventional labels or generator order.
    let cycle = Perm::from_images(&[2, 3, 4, 5, 6, 0, 1]).unwrap();
    let transposition = Perm::from_images(&[2, 1, 0, 3, 4, 5, 6]).unwrap();
    let group = PermGroup::from_generators(7, &[cycle, transposition]).unwrap();
    assert_eq!(group.order(), BigUInt::from(5_040_u64));
    assert!(group
        .contains(&Perm::from_images(&[6, 1, 2, 3, 4, 5, 0]).unwrap())
        .unwrap());
}

#[test]
fn disconnected_cycle_transposition_orbit_uses_general_schreier_sims() {
    // Translating the edge {0,2} around a 6-cycle gives two disconnected
    // triangles. The resulting imprimitive group is not S_6, so the exact
    // symmetric certificate must not fire.
    let cycle = Perm::from_images(&[1, 2, 3, 4, 5, 0]).unwrap();
    let transposition = Perm::from_images(&[2, 1, 0, 3, 4, 5]).unwrap();
    let group = PermGroup::from_generators(6, &[cycle, transposition]).unwrap();
    assert_eq!(group.order(), BigUInt::from(72_u64));
}

#[test]
fn alternating_order_is_half_factorial() {
    for n in 2..=7 {
        let g = alternating(n);
        let mut expected = BigUInt::one();
        for k in 1..=n as u64 {
            expected.mul_assign_u64(k);
        }
        let (q, _) = expected.divmod_u64_for_test(2);
        assert_eq!(g.order(), q, "|A_{n}|");
    }
}

#[test]
fn cyclic_order_is_n() {
    for n in 1..=20 {
        let g = cyclic(n);
        assert_eq!(g.order(), BigUInt::from(n as u64));
    }
}

#[test]
fn dihedral_order_is_2n() {
    for n in 3..=15 {
        let g = dihedral(n);
        assert_eq!(g.order(), BigUInt::from(2 * n as u64));
    }
}

#[test]
fn group_order_free_function_matches_method() {
    let s5 = symmetric(5);
    let n = group_order(5, s5.generators()).unwrap();
    assert_eq!(n, s5.order());
}

#[test]
fn base_and_strong_generating_set_round_trip() {
    let s5 = symmetric(5);
    let (base, sgs) = base_and_strong_generating_set(5, s5.generators()).unwrap();
    // Reconstruct from the SGS alone — base + SGS suffice to recover
    // the group, so the order must agree.
    let rebuilt = PermGroup::from_generators(5, &sgs).unwrap();
    assert_eq!(rebuilt.order(), s5.order());
    assert!(!base.is_empty());
}

#[test]
fn adjoining_generators_matches_fresh_verified_construction() {
    let cycle = Perm::from_images(&[1, 2, 3, 4, 5, 0]).unwrap();
    let first = Perm::from_images(&[1, 0, 2, 3, 4, 5]).unwrap();
    let second = Perm::from_images(&[0, 1, 3, 2, 4, 5]).unwrap();
    let cyclic = PermGroup::from_generators(6, core::slice::from_ref(&cycle)).unwrap();
    let extended = cyclic
        .adjoin_generator(&first)
        .unwrap()
        .adjoin_generator(&second)
        .unwrap();
    let rebuilt = PermGroup::from_generators(6, &[cycle, first, second.clone()]).unwrap();

    assert_eq!(extended.order(), rebuilt.order());
    for element in rebuilt.iter_elements() {
        assert!(extended.contains(&element).unwrap());
    }
    // Adjoining an existing member is an exact no-op.
    assert_eq!(
        extended.adjoin_generator(&second).unwrap().order(),
        rebuilt.order()
    );
}

#[test]
fn element_membership_test_accepts_generators_rejects_arbitrary() {
    let a5 = alternating(5);
    for g in a5.generators() {
        assert!(element_membership_test(&a5, g).unwrap());
    }
    // The transposition (0,1) is odd — should NOT be in A_5.
    let odd = Perm::from_images(&[1, 0, 2, 3, 4]).unwrap();
    assert!(!element_membership_test(&a5, &odd).unwrap());
}

#[test]
fn membership_for_full_symmetric_is_always_true() {
    let s5 = symmetric(5);
    // Generate a few hand-picked permutations and verify each is "in".
    let candidates: &[&[u32]] = &[
        &[0, 1, 2, 3, 4],
        &[4, 3, 2, 1, 0],
        &[1, 2, 3, 4, 0],
        &[2, 0, 4, 1, 3],
    ];
    for c in candidates {
        let p = Perm::from_images(c).unwrap();
        assert!(s5.contains(&p).unwrap());
    }
}

#[test]
fn orbit_of_full_symmetric_is_everything() {
    let s6 = symmetric(6);
    let o = orbit(&s6, 3).unwrap();
    let mut sorted: Vec<u32> = o;
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn transversal_reps_map_base_to_orbit() {
    let s4 = symmetric(4);
    let t = transversal(&s4, 0).unwrap();
    assert_eq!(t.base, 0);
    assert_eq!(t.orbit.len(), 4);
    for (i, p) in t.orbit.iter().enumerate() {
        assert_eq!(t.reps[i].apply(0), *p);
    }
}

#[test]
fn schreier_tree_predecessor_is_witness() {
    let s5 = symmetric(5);
    let tree = schreier_tree(&s5, 0).unwrap();
    assert_eq!(tree.orbit_len(), 5);
    for &p in &tree.orbit {
        if let Some((parent, gi)) = tree.pred[p as usize] {
            assert_eq!(tree.generators_used[gi as usize].apply(parent), p);
        }
    }
}

#[test]
fn schreier_generators_lie_in_point_stabilizer() {
    let s5 = symmetric(5);
    let gens = schreier_generators(&s5, 0).unwrap();
    assert!(!gens.is_empty());
    for sg in &gens {
        assert_eq!(sg.perm.apply(0), 0, "Schreier gen fixes the base point");
    }
}

#[test]
fn pointwise_stabilizer_of_s_n_is_s_n_minus_one() {
    let s5 = symmetric(5);
    let stab = pointwise_stabilizer(&s5, &[0]).unwrap();
    // Stab_{S_5}(0) ≅ S_4, order 24.
    assert_eq!(stab.order(), BigUInt::from(24u64));
}

#[test]
fn iterated_pointwise_stabilizer_drains_to_identity() {
    let s4 = symmetric(4);
    let stab = pointwise_stabilizer(&s4, &[0, 1, 2]).unwrap();
    assert_eq!(stab.order(), BigUInt::one());
}

#[test]
fn orbit_stabilizer_identity() {
    // |orbit| * |stab| = |G| (Burnside's elementary orbit-stabilizer)
    let s5 = symmetric(5);
    let os = orbit_stabilizer(&s5, 2).unwrap();
    let mut product = BigUInt::from(os.orbit.len() as u64);
    product = product.mul(&os.stabilizer.order());
    assert_eq!(product, s5.order());
}

#[test]
fn stabilizer_chain_product_is_group_order() {
    let groups: Vec<PermGroup> = vec![symmetric(5), alternating(5), dihedral(7), cyclic(12)];
    for g in &groups {
        let chain = stabilizer_chain(g);
        let mut p = BigUInt::one();
        for step in &chain.steps {
            p.mul_assign_u64(step.orbit.len() as u64);
        }
        assert_eq!(p, g.order(), "|G| = prod transversals");
    }
}

#[test]
fn iterate_elements_visits_each_element_once() {
    let s4 = symmetric(4);
    let mut seen: Vec<Perm> = Vec::new();
    for elt in iterate_elements(&s4) {
        // Each elt is a permutation of degree 4 lying in the group.
        assert_eq!(elt.degree(), 4);
        assert!(s4.contains(&elt).unwrap());
        // No duplicates.
        assert!(!seen.contains(&elt), "duplicate element");
        seen.push(elt);
    }
    // |S_4| = 24 — every element exactly once.
    assert_eq!(seen.len(), 24);
}

#[test]
fn uniform_random_element_is_seed_deterministic() {
    let s5 = symmetric(5);
    let a = uniform_random_element(&s5, 12345);
    let b = uniform_random_element(&s5, 12345);
    assert_eq!(a.element, b.element, "same seed → same draw");
    assert_eq!(a.seed, 12345);
    assert!(s5.contains(&a.element).unwrap());
}

#[test]
fn uniform_random_element_distinct_seeds_typically_differ() {
    let s7 = symmetric(7);
    let a = uniform_random_element(&s7, 1);
    let b = uniform_random_element(&s7, 2);
    // Not strictly required but with |S_7| = 5040 a tied draw on two
    // chosen seeds is overwhelmingly unlikely. Fail loudly if it
    // happens — the RNG construction is suspect.
    assert_ne!(a.element, b.element);
}

#[test]
fn permutation_order_is_unbounded() {
    let cycle_lengths = [
        2_usize, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
    ];
    let degree: usize = cycle_lengths.iter().sum();
    let mut images: Vec<u32> = (0..degree as u32).collect();
    let mut start = 0;
    let mut expected = BigUInt::one();
    for length in cycle_lengths {
        for offset in 0..length {
            images[start + offset] = (start + (offset + 1) % length) as u32;
        }
        start += length;
        expected.mul_assign_u64(length as u64);
    }
    let permutation = Perm::from_images(&images).unwrap();
    assert_eq!(permutation.order(), expected);
    assert!(permutation.order() > BigUInt::from(u64::MAX));
}

#[test]
fn randomized_variant_matches_deterministic_on_order() {
    // PSL(2,5) ≅ A_5 — randomized BSGS, verified, must produce the
    // same order as the deterministic construction.
    let a5 = alternating(5);
    let randomized = PermGroup::from_generators_randomized(5, a5.generators(), 42, 10).unwrap();
    assert_eq!(randomized.order(), a5.order());
}

#[test]
fn randomized_variant_on_small_group_completes() {
    let dn = dihedral(6);
    let randomized =
        PermGroup::from_generators_randomized(dn.degree(), dn.generators(), 7, 15).unwrap();
    assert_eq!(randomized.order(), dn.order());
}

#[test]
fn trivial_group_has_order_one() {
    let g = PermGroup::from_generators(5, &[Perm::identity(5)]).unwrap();
    assert_eq!(g.order(), BigUInt::one());
    assert!(g.base().is_empty());
    // The only element is the identity.
    let mut elts: Vec<_> = iterate_elements(&g).collect();
    assert_eq!(elts.len(), 1);
    assert!(elts.pop().unwrap().is_identity());
    // Iteration must remain well-defined on the trivial group.
    assert_eq!(g.order(), BigUInt::one());
}

#[test]
fn empty_generator_list_is_trivial_group() {
    let g = PermGroup::from_generators(4, &[]).unwrap();
    assert_eq!(g.order(), BigUInt::one());
    assert!(g.base().is_empty());
}

#[test]
fn iterate_matches_random_sample_membership() {
    let a5 = alternating(5);
    // Every random sample must be one of the iterated elements.
    let elts: Vec<Perm> = iterate_elements(&a5).collect();
    assert_eq!(elts.len(), 60);
    let unique: std::collections::HashSet<_> = elts.iter().cloned().collect();
    assert_eq!(
        unique.len(),
        60,
        "A5 iteration must contain every element once"
    );
    for seed in 0..32u64 {
        let r = uniform_random_element(&a5, seed);
        assert!(elts.contains(&r.element));
    }
}

#[test]
fn schreier_generators_generate_point_stabilizer() {
    // The Schreier generators of Stab_G(b) form a generating set —
    // running them through Schreier-Sims must yield the same order
    // as `pointwise_stabilizer`.
    let s6 = symmetric(6);
    let sg: Vec<Perm> = schreier_generators(&s6, 0)
        .unwrap()
        .into_iter()
        .map(|x| x.perm)
        .collect();
    let stab_via_sg = PermGroup::from_generators(6, &sg).unwrap();
    let stab_direct = pointwise_stabilizer(&s6, &[0]).unwrap();
    assert_eq!(stab_via_sg.order(), stab_direct.order());
}

// ── small helper on the bigint test surface ────────────────────────

trait DivModForTest {
    fn divmod_u64_for_test(&self, d: u64) -> (BigUInt, u64);
}

impl DivModForTest for BigUInt {
    fn divmod_u64_for_test(&self, d: u64) -> (BigUInt, u64) {
        self.divmod_u64(d)
    }
}
