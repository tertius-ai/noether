// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Algebraic-invariant tests — properties that *every* finite
//! permutation group must obey, exercised on a small zoo so that a
//! kernel regression that breaks a public function (Schreier-Sims,
//! sift, transversal product) is caught even when no individual
//! reference value is encoded.

use perm_core::{
    iterate_elements, orbit_stabilizer, stabilizer_chain, uniform_random_element, BigUInt,
    PermGroup,
};

mod common;
use common::{alternating, cyclic, dihedral, psl2p, symmetric};

fn zoo() -> Vec<(String, PermGroup)> {
    vec![
        ("S_3".to_string(), symmetric(3)),
        ("S_4".to_string(), symmetric(4)),
        ("S_5".to_string(), symmetric(5)),
        ("A_4".to_string(), alternating(4)),
        ("A_5".to_string(), alternating(5)),
        ("C_8".to_string(), cyclic(8)),
        ("D_6".to_string(), dihedral(6)),
        ("D_8".to_string(), dihedral(8)),
        ("PSL(2,5)".to_string(), psl2p(5)),
        ("PSL(2,7)".to_string(), psl2p(7)),
    ]
}

#[test]
fn order_equals_product_of_transversal_lengths() {
    for (name, g) in zoo() {
        let chain = stabilizer_chain(&g);
        let mut p = BigUInt::one();
        for step in &chain.steps {
            p.mul_assign_u64(step.orbit.len() as u64);
        }
        assert_eq!(p, g.order(), "{name}: |G| = prod transversals");
    }
}

#[test]
fn orbit_stabilizer_identity_holds_for_every_point() {
    for (name, g) in zoo() {
        for p in 0..g.degree() as u32 {
            let os = orbit_stabilizer(&g, p).unwrap();
            let mut product = BigUInt::from(os.orbit.len() as u64);
            product = product.mul(&os.stabilizer.order());
            assert_eq!(product, g.order(), "{name} @ {p}: |orb| * |stab| = |G|");
        }
    }
}

#[test]
fn every_iterated_element_is_a_member() {
    for (name, g) in zoo() {
        // Limit iteration for the larger groups.
        let mut counted = 0u64;
        for elt in iterate_elements(&g) {
            assert!(g.contains(&elt).unwrap(), "{name}: iterated elt is member");
            counted += 1;
            if counted > 1500 {
                break;
            }
        }
    }
}

#[test]
fn uniform_random_elements_are_members() {
    for (name, g) in zoo() {
        for seed in 0..32u64 {
            let r = uniform_random_element(&g, seed);
            assert!(g.contains(&r.element).unwrap(), "{name} @ {seed}");
        }
    }
}

#[test]
fn uniform_random_sample_distribution_is_flat() {
    // For S_4, 24 elements, 24*200 = 4800 samples; each element
    // should appear roughly 200 times. Allow generous slack.
    let s4 = symmetric(4);
    let n = 4800u64;
    let mut counts: std::collections::HashMap<Vec<u32>, u64> = std::collections::HashMap::new();
    for i in 0..n {
        let r = uniform_random_element(&s4, i + 1);
        let key: Vec<u32> = r.element.images().to_vec();
        *counts.entry(key).or_insert(0) += 1;
    }
    // All 24 distinct elements should appear.
    assert_eq!(counts.len(), 24, "every elt drawn at least once");
    // Each within roughly 5-95% of the uniform expectation 200.
    for (k, &c) in &counts {
        assert!(
            (80..=360).contains(&c),
            "element {:?} drawn {} times — outside the uniform band",
            k,
            c,
        );
    }
}

#[test]
fn psl_2_5_order_is_60() {
    let g = psl2p(5);
    assert_eq!(g.order(), BigUInt::from(60u64));
}

#[test]
fn psl_2_7_order_is_168() {
    let g = psl2p(7);
    assert_eq!(g.order(), BigUInt::from(168u64));
}

#[test]
fn random_construction_matches_deterministic_on_zoo() {
    for (name, g) in zoo() {
        let r = PermGroup::from_generators_randomized(g.degree(), g.generators(), 0xC0FFEE, 20)
            .unwrap();
        assert_eq!(r.order(), g.order(), "{name}: randomized == deterministic");
    }
}
