// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Surface-geometry acceptance cases extracted from arXiv:1906.02261.
//!
//! The fixed sextic and selected tritangent belong to this research driver;
//! the library operations exercised here remain general cyclic-cover tools.

use noether::commalg::{
    CyclicHypersurfaceCover, GeometricDivisorRank, MonomialOrder, PointCountMethod, Polynomial,
    PolynomialRing, ProjectiveSpace, SplitCyclicPullback, WeightedProjectiveHypersurface,
    WeightedProjectiveSpace,
};
use noether::linalg::Matrix;
use noether::numthy::FiniteField;
use noether::{Integer, Rational, Ring, Q};

const PAPER_SEXTIC_SUPPORT: [[u64; 3]; 14] = [
    [6, 0, 0],
    [5, 0, 1],
    [4, 2, 0],
    [4, 0, 2],
    [3, 1, 2],
    [2, 2, 2],
    [1, 5, 0],
    [1, 4, 1],
    [1, 0, 5],
    [0, 6, 0],
    [0, 3, 3],
    [0, 2, 4],
    [0, 1, 5],
    [0, 0, 6],
];

fn paper_cover() -> CyclicHypersurfaceCover<FiniteField> {
    let field = FiniteField::prime(31_u8).unwrap();
    let plane = ProjectiveSpace::with_names(field.clone(), 2, ["x", "y", "z"]).unwrap();
    let branch = plane
        .coordinate_ring()
        .from_u64_terms(
            PAPER_SEXTIC_SUPPORT
                .iter()
                .map(|exponent| (exponent.to_vec(), field.element(-1_i8))),
        )
        .unwrap();
    CyclicHypersurfaceCover::<FiniteField>::new(plane, 2, branch).unwrap()
}

fn paper_weighted_hypersurface() -> WeightedProjectiveHypersurface<FiniteField> {
    let field = FiniteField::prime(31_u8).unwrap();
    let ambient =
        WeightedProjectiveSpace::with_names(field.clone(), vec![3, 1, 1, 1], ["w", "x", "y", "z"])
            .unwrap();
    let ring = ambient.coordinate_ring();
    let w = ring.generator(0).unwrap();
    let branch = ring
        .from_u64_terms(PAPER_SEXTIC_SUPPORT.iter().map(|exponent| {
            (
                vec![0, exponent[0], exponent[1], exponent[2]],
                field.element(-1_i8),
            )
        }))
        .unwrap();
    ambient
        .hypersurface(ring.sub(&w.pow(2).unwrap(), &branch).unwrap())
        .unwrap()
}

fn paper_split_tritangent(cover: &CyclicHypersurfaceCover<FiniteField>) -> SplitCyclicPullback {
    let field = cover.base().base().clone();
    let ring = PolynomialRing::new(field.clone(), ["t"], MonomialOrder::Lexicographic).unwrap();
    let t = ring.generator(0).unwrap();
    let z = ring
        .add(
            &ring.constant(field.element(24)).unwrap(),
            &ring
                .mul(&ring.constant(field.element(23)).unwrap(), &t)
                .unwrap(),
        )
        .unwrap();
    SplitCyclicPullback::line(cover, vec![ring.one(), t, z]).unwrap()
}

fn paper_tate_normalized_polynomial() -> Polynomial<noether::RationalField> {
    let ring = PolynomialRing::new(Q, ["t"], MonomialOrder::Lexicographic).unwrap();
    let numerators = [
        31_i64, -12, 15, -6, -3, -6, -5, 5, -13, 15, -22, 15, -13, 5, -5, -6, -3, -6, 15, -12, 31,
    ];
    let degree_twenty = ring
        .from_u64_terms(
            numerators
                .into_iter()
                .enumerate()
                .map(|(exponent, numerator)| {
                    (
                        vec![exponent as u64],
                        Rational::new(numerator.into(), 31.into()).unwrap(),
                    )
                }),
        )
        .unwrap();
    let t = ring.generator(0).unwrap();
    let t_minus_one = ring.sub(&t, &ring.one()).unwrap();
    ring.mul(&t_minus_one.pow(2).unwrap(), &degree_twenty)
        .unwrap()
}

#[test]
fn paper_surface_presentations_smoothness_and_split_tritangent_agree() {
    let cover = paper_cover();
    assert_eq!(cover.weighted_realization(), &paper_weighted_hypersurface());
    assert_eq!(
        cover.weighted_realization().ambient().weights(),
        &[3, 1, 1, 1]
    );

    let smoothness = cover.smoothness_evidence().unwrap();
    assert_eq!(smoothness.branch_is_smooth(), Some(true));
    assert!(smoothness.weighted_jacobian().is_quasismooth());
    assert!(smoothness.is_smooth());

    let split = paper_split_tritangent(&cover);
    assert_eq!(split.component_count(), 2);
    assert_eq!(
        split.restricted_branch(),
        &split.branch_root().pow(2).unwrap()
    );
    let expected = Matrix::from_rows(
        noether::Z,
        [
            [Integer::from(-2), Integer::from(3)],
            [Integer::from(3), Integer::from(-2)],
        ],
    )
    .unwrap();
    assert_eq!(split.intersection_matrix().unwrap(), expected);
    assert_eq!(split.divisor_rank_lower_bound().unwrap().lower_bound(), 2);
}

#[test]
#[ignore = "full characteristic-31 p-adic K3 Frobenius stress calculation"]
fn paper_characteristic_31_frobenius_and_divisor_rank() {
    let cover = paper_cover();
    let cohomology = cover.surface_cohomology().unwrap();
    assert_eq!(cohomology.second_betti_number(), 22);
    assert_eq!(cohomology.required_point_count_degree(), 11);

    let counts = cover.point_counts(1..=11).unwrap();
    assert!(counts
        .counts()
        .iter()
        .all(|record| record.evidence().method() == PointCountMethod::PadicDoubleCoverTrace));
    let frobenius = cohomology.frobenius_polynomial(&counts).unwrap();
    let upper = frobenius.geometric_divisor_rank_upper_bound().unwrap();
    assert_eq!(
        upper.tate_normalized_polynomial(),
        &paper_tate_normalized_polynomial()
    );
    assert_eq!(upper.upper_bound(), 2);

    let lower = paper_split_tritangent(&cover)
        .divisor_rank_lower_bound()
        .unwrap();
    let rank: GeometricDivisorRank = upper.certify_exact(&lower).unwrap();
    assert_eq!(rank.rank(), 2);
}
