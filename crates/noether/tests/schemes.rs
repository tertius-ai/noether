// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    AffineSpace, FiniteSchemeClosedPointSet, MonomialOrder, ProjectiveSpace,
    RationalPointEnumeration, Scheme,
};
use noether::numthy::{FiniteField, PointCountCertificate};
use noether::{Integer, Ring, SemanticType};

#[test]
fn affine_hyperbola_has_a_trusted_and_partitioned_count() {
    let field = FiniteField::prime(3_u64).unwrap();
    let space = AffineSpace::with_names(field.clone(), 2, ["x", "y"]).unwrap();
    let ring = space.coordinate_ring();
    let variables = ring.generators().unwrap();
    let equation = (&(&variables[0] * &variables[1]).unwrap() - &ring.one()).unwrap();
    let scheme = space
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();

    let trusted = scheme.rational_points().unwrap().collect::<Vec<_>>();
    let (count, certificate) = scheme.point_count_partitioned(4).unwrap();

    assert_eq!(trusted.len(), 2);
    assert_eq!(scheme.dimension().unwrap(), 1);
    assert_eq!(scheme.degree().unwrap().to_string(), "2");
    assert!(scheme.is_smooth().unwrap());
    assert_eq!(
        scheme.singular_subscheme().unwrap().dimension().unwrap(),
        -1
    );
    assert_eq!(count.to_string(), "2");
    assert_eq!(certificate.point_count(), count);
    assert_eq!(certificate.candidate_count(), 9);
    assert_eq!(certificate.shard_candidate_counts(), [3, 2, 2, 2]);
    certificate.verify(&scheme).unwrap();
    assert_eq!(
        PointCountCertificate::from_json(&certificate.to_json().unwrap()).unwrap(),
        certificate
    );
}

#[test]
fn projective_points_are_unique_normalized_and_homogeneous() {
    let field = FiniteField::prime(3_u64).unwrap();
    let space = ProjectiveSpace::with_names(field.clone(), 1, ["u", "v"]).unwrap();
    let ring = space.coordinate_ring();
    let whole = space
        .closed_subscheme(ring.ideal(Vec::new()).unwrap())
        .unwrap();

    let points = whole.rational_points().unwrap().collect::<Vec<_>>();
    assert_eq!(points.len(), 4);
    assert_eq!(whole.dimension().unwrap(), 1);
    assert_eq!(whole.degree().unwrap().to_string(), "1");
    let hilbert = whole.hilbert_polynomial().unwrap();
    assert_eq!(
        hilbert,
        (&hilbert.ring().generator(0).unwrap() + &hilbert.ring().one()).unwrap()
    );
    assert_eq!(whole.point_count().unwrap().0.to_string(), "4");

    let normalized = whole
        .point([field.element(2_u8), field.element(1_u8)])
        .unwrap();
    assert_eq!(normalized.coordinates()[0], field.one());
    assert_eq!(normalized.coordinates()[1], field.element(2_u8));
    normalized.verify().unwrap();

    let variables = ring.generators().unwrap();
    let nonhomogeneous = (&variables[0] + &ring.one()).unwrap();
    assert!(space
        .closed_subscheme(ring.ideal([nonhomogeneous]).unwrap())
        .is_err());
}

#[test]
fn extension_fields_use_the_same_compiled_enumerator() {
    let field = FiniteField::extension(3_u64, 2).unwrap();
    let space = AffineSpace::new(field, 1).unwrap();
    let ring = space.coordinate_ring();
    let scheme = space
        .closed_subscheme(ring.ideal(Vec::new()).unwrap())
        .unwrap();

    let slow = scheme.rational_points().unwrap().count();
    let first = scheme.point_count_partitioned(3).unwrap();
    let second = scheme.point_count_partitioned(3).unwrap();

    assert_eq!(slow, 9);
    assert_eq!(first.0.to_string(), "9");
    assert_eq!(first, second);
    first.1.verify(&scheme).unwrap();
}

#[test]
fn closed_point_solver_detects_a_quadratic_point_without_enumeration() {
    let field = FiniteField::prime(3_u64).unwrap();
    let space = AffineSpace::with_names(field.clone(), 1, ["x"]).unwrap();
    let ring = space.coordinate_ring();
    let x = ring.generator(0).unwrap();
    let equation = (&x.pow(2).unwrap() + &ring.one()).unwrap();
    let scheme = space
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();

    let solution = scheme.closed_points().unwrap();
    assert_eq!(solution.points().len(), 1);
    assert_eq!(solution.geometric_point_count(), Integer::from(2));
    let point = &solution.points()[0];
    assert_eq!(point.degree(), 2);
    assert_eq!(point.residue_field().order(), Integer::from(9));
    assert_eq!(point.frobenius_orbit().len(), 2);
    assert_ne!(point.frobenius_orbit()[0], point.frobenius_orbit()[1]);
    point.verify().unwrap();
    solution.verify().unwrap();
    assert_eq!(
        FiniteSchemeClosedPointSet::from_json(&solution.to_json().unwrap()).unwrap(),
        solution
    );
}

#[test]
fn closed_point_solver_embeds_an_extension_base_into_the_residue_field() {
    let field = FiniteField::with_modulus(2, vec![1, 1, 1]).unwrap();
    let space = AffineSpace::with_names(field.clone(), 1, ["x"]).unwrap();
    let ring = space.coordinate_ring();
    let x = ring.generator(0).unwrap();
    let equation =
        (&(&x.pow(2).unwrap() + &x).unwrap() + &ring.constant(field.generator()).unwrap()).unwrap();
    let scheme = space
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();

    let solution = scheme.closed_points().unwrap();
    assert_eq!(solution.points().len(), 1);
    let point = &solution.points()[0];
    assert_eq!(point.degree(), 2);
    assert_eq!(point.residue_field().degree(), 4);
    assert_eq!(point.base_embedding().source(), &field);
    point.verify().unwrap();
}

#[test]
fn closed_point_solver_handles_a_huge_prime_linear_system() {
    let prime = "170141183460469231731687303715884105727"
        .parse::<Integer>()
        .unwrap();
    let field = FiniteField::prime(prime).unwrap();
    let space = AffineSpace::with_names(field.clone(), 1, ["x"]).unwrap();
    let ring = space.coordinate_ring();
    let equation =
        (&ring.generator(0).unwrap() - &ring.constant(field.element(42)).unwrap()).unwrap();
    let scheme = space
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();

    let solution = scheme.closed_points().unwrap();
    assert_eq!(solution.points().len(), 1);
    assert_eq!(solution.points()[0].degree(), 1);
    assert_eq!(solution.points()[0].coordinates(), &[field.element(42)]);
}

#[test]
fn one_projective_model_matches_slow_and_compiled_counts_over_f3_and_f9() {
    for (field, expected) in [
        (FiniteField::prime(3_u64).unwrap(), 4_usize),
        (FiniteField::extension(3_u64, 2).unwrap(), 10_usize),
    ] {
        let plane = ProjectiveSpace::with_names(field, 2, ["x", "y", "z"]).unwrap();
        let ring = plane.coordinate_ring();
        let variables = ring.generators().unwrap();
        let equation =
            (&(&variables[0] * &variables[2]).unwrap() - &variables[1].pow(2).unwrap()).unwrap();
        let conic = plane
            .closed_subscheme(ring.ideal([equation]).unwrap())
            .unwrap();

        let slow_count = conic.rational_points().unwrap().count();
        let first = conic.point_count_partitioned(4).unwrap();
        let second = conic.point_count_partitioned(4).unwrap();

        assert_eq!(slow_count, expected);
        assert_eq!(first.0.to_string(), expected.to_string());
        assert_eq!(first, second);
        assert_eq!(
            first.1.shard_candidate_counts().iter().sum::<u64>(),
            first.1.candidate_count()
        );
        first.1.verify(&conic).unwrap();
    }
}

#[test]
fn point_enumeration_resumes_exactly_and_shards_cover_the_stream() {
    let field = FiniteField::prime(3_u64).unwrap();
    let space = AffineSpace::new(field, 2).unwrap();
    let ring = space.coordinate_ring();
    let scheme = space
        .closed_subscheme(ring.ideal(Vec::new()).unwrap())
        .unwrap();
    let full = scheme.rational_points().unwrap().collect::<Vec<_>>();

    let mut live = scheme.rational_points().unwrap();
    let first = live.next().unwrap();
    let restored = RationalPointEnumeration::from_json(&live.to_json().unwrap()).unwrap();
    let mut resumed = vec![first];
    resumed.extend(restored);
    assert_eq!(resumed, full);
    let mut completed = scheme.rational_points_shard(2, 4).unwrap();
    completed.by_ref().for_each(drop);
    assert_eq!(
        RationalPointEnumeration::from_json(&completed.to_json().unwrap())
            .unwrap()
            .count(),
        0
    );

    let mut sharded = Vec::new();
    for shard in 0..4 {
        sharded.extend(scheme.rational_points_shard(shard, 4).unwrap());
    }
    sharded.sort_by_key(|point| point.canonical_bytes().unwrap());
    let mut sorted_full = full;
    sorted_full.sort_by_key(|point| point.canonical_bytes().unwrap());
    assert_eq!(sharded, sorted_full);
}

#[test]
fn scheme_json_revalidates_ambient_and_equations() {
    let field = FiniteField::prime(5_u64).unwrap();
    let space = ProjectiveSpace::new(field, 1).unwrap();
    let ring = space.coordinate_ring();
    let variables = ring.generators().unwrap();
    let equation = (&variables[0] * &variables[1]).unwrap();
    let scheme = space
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();

    let encoded = scheme.to_json().unwrap();
    assert_eq!(Scheme::<FiniteField>::from_json(&encoded).unwrap(), scheme);

    let mut payload = serde_json::from_str::<serde_json::Value>(&encoded).unwrap();
    payload["ambient_kind"] = serde_json::Value::String("unknown".to_owned());
    assert!(Scheme::<FiniteField>::from_json(&payload.to_string()).is_err());

    let point = scheme
        .point([scheme.base().one(), scheme.base().zero()])
        .unwrap();
    assert_eq!(
        noether::commalg::SchemePoint::<FiniteField>::from_json(&point.to_json().unwrap()).unwrap(),
        point
    );
}

#[test]
fn rational_ambient_spaces_share_the_exact_scheme_shape() {
    let affine = AffineSpace::with_names(noether::Q, 2, ["x", "y"]).unwrap();
    let ring = affine.coordinate_ring();
    let scheme = affine
        .closed_subscheme(ring.ideal([ring.generator(0).unwrap()]).unwrap())
        .unwrap();

    assert!(scheme.is_affine());
    assert_eq!(scheme.base(), &noether::Q);
    assert_eq!(
        Scheme::<noether::RationalField>::from_json(&scheme.to_json().unwrap()).unwrap(),
        scheme
    );
    assert_eq!(
        AffineSpace::<noether::RationalField>::from_json(&affine.to_json().unwrap()).unwrap(),
        affine
    );

    let alternate_ring =
        noether::commalg::PolynomialRing::new(noether::Q, ["x", "y"], MonomialOrder::Lexicographic)
            .unwrap();
    assert!(affine
        .closed_subscheme(alternate_ring.ideal(Vec::new()).unwrap())
        .is_err());
}

#[test]
fn cusp_jacobian_recovers_the_hand_checked_singular_subscheme() {
    let space = AffineSpace::with_names(noether::Q, 2, ["x", "y"]).unwrap();
    let ring = space.coordinate_ring();
    let variables = ring.generators().unwrap();
    let (x, y) = (&variables[0], &variables[1]);
    let equation = (&y.pow(2).unwrap() - &x.pow(3).unwrap()).unwrap();
    let cusp = space
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();

    let singular = cusp.singular_subscheme().unwrap();
    assert_eq!(singular.dimension().unwrap(), 0);
    assert_eq!(
        singular.defining_ideal(),
        &ring.ideal([x.pow(2).unwrap(), y.clone()]).unwrap()
    );
    assert!(!cusp.is_smooth().unwrap());
    assert_eq!(cusp.jacobian_subscheme().unwrap(), singular);
}

#[test]
fn projective_conic_is_smooth_and_non_complete_intersections_fail_closed() {
    let projective = ProjectiveSpace::with_names(noether::Q, 2, ["x", "y", "z"]).unwrap();
    let ring = projective.coordinate_ring();
    let variables = ring.generators().unwrap();
    let equation = (&(&variables[0].pow(2).unwrap() + &variables[1].pow(2).unwrap()).unwrap()
        + &variables[2].pow(2).unwrap())
        .unwrap();
    let conic = projective
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();
    assert!(conic.is_smooth().unwrap());
    assert_eq!(conic.singular_subscheme().unwrap().dimension().unwrap(), -1);

    let affine = AffineSpace::with_names(noether::Q, 4, ["x", "y", "z", "w"]).unwrap();
    let ring = affine.coordinate_ring();
    let variables = ring.generators().unwrap();
    let (x, y, z, w) = (&variables[0], &variables[1], &variables[2], &variables[3]);
    let twisted_cubic_cone = affine
        .closed_subscheme(
            ring.ideal([
                (&(x * z).unwrap() - &y.pow(2).unwrap()).unwrap(),
                (&(x * w).unwrap() - &(y * z).unwrap()).unwrap(),
                (&(y * w).unwrap() - &z.pow(2).unwrap()).unwrap(),
            ])
            .unwrap(),
        )
        .unwrap();
    assert!(twisted_cubic_cone.jacobian_subscheme().is_ok());
    assert!(twisted_cubic_cone.singular_subscheme().is_err());
    assert!(twisted_cubic_cone.is_smooth().is_err());
}
