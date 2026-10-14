// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    Degree, FreeAbelianGrading, MonomialOrder, PolynomialRing, PolynomialRingHomomorphism,
    QuotientElement, QuotientError, QuotientRing, QuotientRingMap, QuotientRingMapError,
};
use noether::linalg::Matrix;
use noether::numthy::FiniteField;
use noether::{
    CommutativeRing, IdentityCoefficientMap, Integer, RationalField, Ring, SemanticType, Q,
};

fn artinian_quotient() -> (
    QuotientRing<PolynomialRing<RationalField>>,
    Vec<noether::commalg::Polynomial<RationalField>>,
) {
    let ring =
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let variables = ring.generators().unwrap();
    let ideal = ring
        .ideal([
            (&variables[0].pow(2).unwrap() - &variables[1]).unwrap(),
            variables[1].pow(2).unwrap(),
        ])
        .unwrap();
    (ring.quotient(&ideal).unwrap(), variables)
}

#[test]
fn canonical_lifts_make_quotient_equality_and_ring_laws_exact() {
    let (quotient, variables) = artinian_quotient();
    let x = quotient.element(&variables[0]).unwrap();
    let y = quotient.element(&variables[1]).unwrap();

    assert_eq!(x.pow(2), y);
    assert!(x.pow(4).is_zero());
    assert_eq!(quotient.element(&variables[0].pow(2).unwrap()).unwrap(), y);
    assert_eq!(quotient.element(&y.lift().clone()).unwrap(), y);
    assert_eq!((&x + &y).unwrap(), (&y + &x).unwrap());
    assert_eq!(
        &(&x + &quotient.one()).unwrap() * &y,
        Ok((&(&x * &y).unwrap() + &y).unwrap())
    );
    assert_eq!((&x - &x).unwrap(), quotient.zero());
    assert_eq!((-&x).unwrap().lift().nonzero_term_count(), 1);
    assert!(quotient.contains(&x));
    assert_eq!(
        QuotientRing::<PolynomialRing<RationalField>>::TYPE_ID,
        "noether/commalg/QuotientRing@1"
    );
    assert_eq!(
        QuotientElement::<PolynomialRing<RationalField>>::TYPE_ID,
        "noether/commalg/QuotientElement@1"
    );
}

#[test]
fn quotient_context_identity_is_presentation_exact_and_contexts_are_checked() {
    let (quotient, variables) = artinian_quotient();
    let rebuilt_ideal = quotient
        .base_ring()
        .ideal(quotient.defining_ideal().generators().iter().cloned())
        .unwrap();
    let rebuilt = quotient.base_ring().quotient(&rebuilt_ideal).unwrap();
    assert_eq!(quotient, rebuilt);
    assert_eq!(
        quotient.element(&variables[0]).unwrap(),
        rebuilt.element(&variables[0]).unwrap()
    );

    let other_ideal = quotient
        .base_ring()
        .ideal([variables[0].pow(2).unwrap()])
        .unwrap();
    let other = quotient.base_ring().quotient(&other_ideal).unwrap();
    let left = quotient.element(&variables[0]).unwrap();
    let right = other.element(&variables[0]).unwrap();
    assert!((&left + &right).is_err());

    let differently_named =
        PolynomialRing::new(Q, ["u", "v"], MonomialOrder::GradedReverseLexicographic).unwrap();
    assert!(differently_named.quotient(&rebuilt_ideal).is_err());
}

#[test]
fn nilpotent_units_have_exact_inverses_and_nonunits_fail_closed() {
    let (quotient, variables) = artinian_quotient();
    let x = quotient.element(&variables[0]).unwrap();
    let one_plus_x = (&quotient.one() + &x).unwrap();
    let inverse = one_plus_x.inverse().unwrap();

    assert_eq!((&one_plus_x * &inverse).unwrap(), quotient.one());
    assert!(one_plus_x.is_unit().unwrap());
    assert!(!x.is_unit().unwrap());
    assert_eq!(x.inverse().unwrap_err(), QuotientError::NotAUnit);

    let unit_ideal = quotient
        .base_ring()
        .ideal([quotient.base_ring().one()])
        .unwrap();
    let zero_ring = quotient.base_ring().quotient(&unit_ideal).unwrap();
    assert_eq!(zero_ring.zero(), zero_ring.one());
    assert!(zero_ring.zero().is_unit().unwrap());
    assert_eq!(zero_ring.zero().inverse().unwrap(), zero_ring.zero());
}

#[test]
fn quotient_gradings_and_hilbert_data_are_owned_by_the_context() {
    let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
    let [x, y] = ring.generators().unwrap().try_into().unwrap();
    let ideal = ring.ideal([x.pow(2).unwrap()]).unwrap();
    let quotient = ring.quotient(&ideal).unwrap();

    assert!(quotient.is_graded());
    assert_eq!(quotient.grading(), Some(ring.grading()));
    assert_eq!(
        quotient.hilbert_series().unwrap(),
        ideal.hilbert_series().unwrap()
    );
    assert_eq!(
        quotient.hilbert_numerator().unwrap(),
        vec![Integer::from(1), Integer::zero(), Integer::from(-1)]
    );
    assert_eq!(quotient.dimension().unwrap(), 1);
    assert_eq!(quotient.degree().unwrap(), Integer::from(2));
    assert_eq!(
        quotient
            .element(&(&x * &y).unwrap())
            .unwrap()
            .multihomogeneous_degree()
            .unwrap(),
        Some(Degree::new(vec![2.into()]))
    );

    let inhomogeneous = ring.ideal([(&x.pow(2).unwrap() - &x).unwrap()]).unwrap();
    let ungraded = ring.quotient(&inhomogeneous).unwrap();
    assert!(!ungraded.is_graded());
    assert!(ungraded.grading().is_none());
    assert_eq!(
        ungraded
            .element(&y)
            .unwrap()
            .multihomogeneous_degree()
            .unwrap_err(),
        QuotientError::NotGraded
    );

    let grading = FreeAbelianGrading::new(
        2,
        vec![
            Degree::new(vec![1.into(), 0.into()]),
            Degree::new(vec![0.into(), 1.into()]),
        ],
    )
    .unwrap();
    let bigraded = PolynomialRing::with_grading(
        Q,
        ["s", "t"],
        MonomialOrder::GradedReverseLexicographic,
        grading.clone(),
    )
    .unwrap();
    let [s, t] = bigraded.generators().unwrap().try_into().unwrap();
    let bigraded_ideal = bigraded.ideal([(&s * &t).unwrap()]).unwrap();
    let bigraded_quotient = bigraded.quotient(&bigraded_ideal).unwrap();
    assert_eq!(bigraded_quotient.grading(), Some(&grading));
}

#[test]
fn quotient_json_revalidates_canonical_lifts() {
    let (quotient, variables) = artinian_quotient();
    let value = quotient.element(&variables[0].pow(3).unwrap()).unwrap();

    assert_eq!(
        QuotientRing::<PolynomialRing<RationalField>>::from_json(&quotient.to_json().unwrap())
            .unwrap(),
        quotient
    );
    assert_eq!(
        QuotientElement::<PolynomialRing<RationalField>>::from_json(&value.to_json().unwrap())
            .unwrap(),
        value
    );

    let mut wire: serde_json::Value = serde_json::from_str(&value.to_json().unwrap()).unwrap();
    wire["lift"] = serde_json::to_value(variables[0].pow(3).unwrap()).unwrap();
    assert!(QuotientElement::<PolynomialRing<RationalField>>::from_json(
        &serde_json::to_string(&wire).unwrap()
    )
    .is_err());
}

#[test]
fn matrices_and_finite_algebras_reuse_the_quotient_ring_contract() {
    let field = FiniteField::prime(101_u64).unwrap();
    let ring = PolynomialRing::new(field, ["x"], MonomialOrder::default()).unwrap();
    let x = ring.generator(0).unwrap();
    let ideal = ring.ideal([x.pow(3).unwrap()]).unwrap();
    let quotient = ring.quotient(&ideal).unwrap();
    let x = quotient.element(&x).unwrap();
    let matrix = Matrix::from_rows(
        quotient.clone(),
        [[quotient.one(), x.clone()], [x.clone(), quotient.one()]],
    )
    .unwrap();
    let square = matrix.matmul(&matrix).unwrap();
    let expected_diagonal = (&quotient.one() + &x.pow(2)).unwrap();
    assert_eq!(square.get(0, 0).unwrap(), expected_diagonal);
    assert_eq!(square.get(0, 1).unwrap(), (&x + &x).unwrap());

    let finite = quotient.finite_algebra().unwrap();
    assert_eq!(finite.dimension(), 3);
    assert_eq!(finite.source(), quotient.defining_ideal());
    assert_eq!(finite.multiplication_operator_nonzeros(), vec![2]);
}

fn assert_commutative_ring<R: CommutativeRing>() {}
fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn quotient_context_and_elements_are_generic_thread_safe_ring_values() {
    assert_commutative_ring::<QuotientRing<PolynomialRing<RationalField>>>();
    assert_send_sync::<QuotientRing<PolynomialRing<RationalField>>>();
    assert_send_sync::<QuotientElement<PolynomialRing<RationalField>>>();
}

#[test]
fn quotient_maps_descend_only_when_relations_vanish() {
    type RationalPolynomialMap = PolynomialRingHomomorphism<
        RationalField,
        RationalField,
        IdentityCoefficientMap<RationalField>,
    >;
    type RationalQuotientMap = QuotientRingMap<
        PolynomialRing<RationalField>,
        PolynomialRing<RationalField>,
        RationalPolynomialMap,
    >;

    let (source, source_variables) = artinian_quotient();
    let target_ring = PolynomialRing::new(Q, ["t"], MonomialOrder::default()).unwrap();
    let t = target_ring.generator(0).unwrap();
    let target_ideal = target_ring.ideal([t.pow(4).unwrap()]).unwrap();
    let target = target_ring.quotient(&target_ideal).unwrap();
    let ambient_map = source
        .base_ring()
        .homomorphism(
            target.base_ring(),
            IdentityCoefficientMap::new(Q),
            vec![t.clone(), t.pow(2).unwrap()],
        )
        .unwrap();
    let descended = source.induced_map(&target, ambient_map).unwrap();

    let x_plus_y = source
        .element(&(&source_variables[0] + &source_variables[1]).unwrap())
        .unwrap();
    assert_eq!(
        descended.apply(&x_plus_y).unwrap(),
        target.element(&(&t + &t.pow(2).unwrap()).unwrap()).unwrap()
    );
    descended.verify().unwrap();
    assert_eq!(
        RationalQuotientMap::from_json(&descended.to_json().unwrap()).unwrap(),
        descended
    );

    let final_ring = PolynomialRing::new(Q, ["u"], MonomialOrder::default()).unwrap();
    let u = final_ring.generator(0).unwrap();
    let final_ideal = final_ring.ideal([u.pow(2).unwrap()]).unwrap();
    let final_quotient = final_ring.quotient(&final_ideal).unwrap();
    let outer_ambient_map = target
        .base_ring()
        .homomorphism(
            final_quotient.base_ring(),
            IdentityCoefficientMap::new(Q),
            vec![u.clone()],
        )
        .unwrap();
    let outer = target
        .induced_map(&final_quotient, outer_ambient_map)
        .unwrap();
    let composed = outer.compose(&descended).unwrap();
    assert_eq!(
        composed.apply(&x_plus_y).unwrap(),
        final_quotient.element(&u).unwrap()
    );

    let invalid_ambient_map = source
        .base_ring()
        .homomorphism(
            target.base_ring(),
            IdentityCoefficientMap::new(Q),
            vec![t, target.base_ring().zero()],
        )
        .unwrap();
    assert_eq!(
        source
            .induced_map(&target, invalid_ambient_map)
            .unwrap_err(),
        QuotientRingMapError::RelationDoesNotVanish { generator: 0 }
    );
}

#[test]
fn prepared_word_prime_quotient_reduction_matches_the_exact_ideal_floor() {
    let field = FiniteField::prime(65_521_u64).unwrap();
    let ring = PolynomialRing::new(
        field.clone(),
        ["x0", "x1", "x2", "x3"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let mut state = 0x4D59_5DF4_D0F3_3173_u64;
    let reducers = (0..4)
        .map(|_| {
            ring.from_u64_terms(
                (0..4)
                    .flat_map(|left| (left..4).map(move |right| (left, right)))
                    .map(|(left, right)| {
                        let mut exponent = vec![0; 4];
                        exponent[left] += 1;
                        exponent[right] += 1;
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        (exponent, field.element((state % 65_521).max(1)))
                    }),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let ideal = ring.ideal(reducers).unwrap();
    assert!(ideal.generators().len() >= 8);
    assert!(
        ideal
            .generators()
            .iter()
            .map(|value| value.nonzero_term_count())
            .sum::<usize>()
            >= 32
    );
    let quotient = ring.quotient(&ideal).unwrap();
    let variables = ring.generators().unwrap();
    let left = (&variables[0].pow(3).unwrap() + &variables[1].pow(3).unwrap()).unwrap();
    let right = (&variables[2].pow(3).unwrap() + &variables[3].pow(3).unwrap()).unwrap();
    let ambient_product = (&left * &right).unwrap();
    let exact = ideal.normal_form(&ambient_product).unwrap();
    let quotient_product =
        (&quotient.element(&left).unwrap() * &quotient.element(&right).unwrap()).unwrap();

    assert_eq!(quotient_product.lift(), &exact);
    assert_eq!(quotient.element(&ambient_product).unwrap().lift(), &exact);
}
