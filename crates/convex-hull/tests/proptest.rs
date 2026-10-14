// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Layer 3 — property-based tests (proptest). Random rational point
//! clouds: the hull is valid, every vertex is an input point, the
//! `f`-vector satisfies Euler–Poincaré, and beneath–beyond agrees with
//! the double-description facets.

use convex_hull::*;
use proptest::prelude::*;

fn property_config(default_cases: u32) -> ProptestConfig {
    let configured = ProptestConfig::default();
    if std::env::var_os("PROPTEST_CASES").is_some() {
        configured
    } else {
        ProptestConfig {
            cases: default_cases,
            ..configured
        }
    }
}

fn euler_ok(p: &Polytope) -> bool {
    p.f_vector()
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let dim = i as isize - 1;
            if dim.rem_euclid(2) == 0 {
                f as isize
            } else {
                -(f as isize)
            }
        })
        .sum::<isize>()
        == 0
}

fn facet_rows(facets: &[Facet]) -> Vec<Vec<Q>> {
    let mut rows = facets
        .iter()
        .map(|facet| {
            let mut row = facet.normal.clone();
            row.push(facet.rhs.clone());
            row
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

proptest! {
    #![proptest_config(property_config(80))]

    #[test]
    fn random_3d_hull(rows in prop::collection::vec(prop::collection::vec(-2i64..3, 3), 5..14)) {
        let pts: Vec<Vec<Q>> =
            rows.iter().map(|r| r.iter().map(|&x| q(x, 1)).collect()).collect();
        let p = convex_hull(&pts).unwrap();
        prop_assert!(euler_ok(&p));
        for v in p.vertices() {
            prop_assert!(pts.iter().any(|x| x == v));
        }
        if p.is_full_dimensional() {
            let (bb, _) = facet_enumeration(&pts).unwrap();
            prop_assert_eq!(facet_rows(&bb), facet_rows(p.facets()));
        }
    }

    #[test]
    fn random_4d_hull(rows in prop::collection::vec(prop::collection::vec(0i64..2, 4), 6..12)) {
        let pts: Vec<Vec<Q>> =
            rows.iter().map(|r| r.iter().map(|&x| q(x, 1)).collect()).collect();
        let p = convex_hull(&pts).unwrap();
        prop_assert!(euler_ok(&p));
    }
}
