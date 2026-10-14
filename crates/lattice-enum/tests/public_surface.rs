// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use flint_core::Integer;
use lattice_enum::{cvp_enumerate, representation_vectors_collect, svp_enumerate};

#[test]
fn diagonal_gram_public_surface_is_exact() {
    let gram = vec![
        vec![Integer::from(1), Integer::from(0)],
        vec![Integer::from(0), Integer::from(2)],
    ];

    let shortest = svp_enumerate(&gram, None, None).expect("positive-definite Gram");
    assert_eq!(shortest.count, 2);
    assert_eq!(shortest.max_norm, 1);
    assert_eq!(shortest.vectors, vec![vec![1, 0]]);

    let represented =
        representation_vectors_collect(&gram, Integer::from(2)).expect("finite exact shell");
    assert_eq!(represented, vec![vec![0, 1]]);

    let closest =
        cvp_enumerate(&gram, &[3, 0], 2, None).expect("half-integral target has closest vectors");
    assert_eq!(closest.distance_sq_num, 1);
    assert_eq!(closest.distance_sq_den, 4);
    assert_eq!(closest.vectors, vec![vec![1, 0], vec![2, 0]]);
}

#[test]
fn closest_vector_is_exact_beyond_f64_unit_resolution() {
    let gram = vec![vec![Integer::one()]];
    let target = 9_007_199_254_740_993_i64;
    let closest = cvp_enumerate(&gram, &[target], 1, None).unwrap();
    assert_eq!(closest.distance_sq_num, 0);
    assert_eq!(closest.vectors, vec![vec![target]]);
}

#[test]
fn bounded_search_does_not_require_a_floating_gram_projection() {
    let huge: Integer = format!("1{}", "0".repeat(400)).parse().unwrap();
    let gram = vec![vec![huge]];

    let short = svp_enumerate(&gram, Some(Integer::one()), Some(1)).unwrap();
    assert!(short.vectors.is_empty());

    let closest = cvp_enumerate(&gram, &[1], 2, None).unwrap();
    assert_eq!(closest.vectors, vec![vec![0], vec![1]]);
}
