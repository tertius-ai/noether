// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Bounded properties for high-level exact objects and certified enclosures.
//!
//! Each randomized family follows a small hand-checkable seed.  The tests use
//! independent formulas, group laws, Euler--Poincare, or replay verification;
//! they do not merely repeat the implementation under another name.

use std::fmt::Debug;

use noether::category::SteinerCategory;
use noether::commalg::{MonomialOrder, PolynomialRing};
use noether::numthy::{
    EllipticCurve, EllipticCurvePoint, FiniteField, FractionalIdeal, NumberField,
    NumberFieldElement, NumberFieldOrder,
};
use noether::topology::{SimplicialComplex, StrongCoreReduction};
use noether::{
    CommutativeRing, ComplexBall, ComplexBallField, Integer, Rational, RealBall, RealBallField,
    Ring, Q,
};
use proptest::collection::vec;
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

fn rational(numerator: i16, denominator_minus_one: u8) -> Rational {
    Rational::new(
        Integer::from(numerator),
        Integer::from(u64::from(denominator_minus_one) + 1),
    )
    .unwrap()
}

fn quadratic_field(radicand: i64, generator: &str) -> NumberField {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let x = ring.generator(0).unwrap();
    let polynomial = (&x.pow(2).unwrap() + &ring.constant((-radicand).into()).unwrap()).unwrap();
    NumberField::new(polynomial, Some(generator.to_owned())).unwrap()
}

fn quadratic_element(field: &NumberField, scalar: Rational, root: Rational) -> NumberFieldElement {
    field.element(&[scalar, root]).unwrap()
}

fn assert_ring_laws<R>(ring: &R, left: &R::Element, middle: &R::Element, right: &R::Element)
where
    R: CommutativeRing,
    R::Element: Debug,
{
    assert_eq!(
        ring.add(&ring.add(left, middle).unwrap(), right).unwrap(),
        ring.add(left, &ring.add(middle, right).unwrap()).unwrap()
    );
    assert_eq!(
        ring.mul(&ring.mul(left, middle).unwrap(), right).unwrap(),
        ring.mul(left, &ring.mul(middle, right).unwrap()).unwrap()
    );
    assert_eq!(
        ring.mul(left, &ring.add(middle, right).unwrap()).unwrap(),
        ring.add(
            &ring.mul(left, middle).unwrap(),
            &ring.mul(left, right).unwrap()
        )
        .unwrap()
    );
    assert_eq!(ring.add(left, &ring.zero()).unwrap(), left.clone());
    assert_eq!(ring.mul(left, &ring.one()).unwrap(), left.clone());
    assert_eq!(
        ring.add(left, &ring.neg(left).unwrap()).unwrap(),
        ring.zero()
    );
}

fn elliptic_curve(prime: u64) -> EllipticCurve<FiniteField> {
    let field = FiniteField::prime(prime).unwrap();
    EllipticCurve::new(
        field.clone(),
        [
            field.zero(),
            field.zero(),
            field.zero(),
            field.one(),
            field.one(),
        ],
    )
    .unwrap()
}

fn elliptic_points(curve: &EllipticCurve<FiniteField>) -> Vec<EllipticCurvePoint<FiniteField>> {
    let prime = curve
        .base()
        .characteristic()
        .to_string()
        .parse::<u64>()
        .unwrap();
    let mut points = vec![curve.origin().unwrap()];
    for x in 0..prime {
        for y in 0..prime {
            if let Ok(point) = curve.point(curve.base().element(x), curve.base().element(y)) {
                points.push(point);
            }
        }
    }
    points
}

fn simplicial_complex(vertex_count: usize, masks: &[u8]) -> SimplicialComplex {
    let mut facets = masks
        .iter()
        .map(|mask| {
            (0..vertex_count)
                .filter(|vertex| mask & (1 << vertex) != 0)
                .map(|vertex| vertex as u32)
                .collect::<Vec<_>>()
        })
        .filter(|facet| !facet.is_empty())
        .collect::<Vec<_>>();
    if facets.is_empty() {
        facets.push(vec![0]);
    }
    SimplicialComplex::from_facets(&facets)
}

fn category_case(family: u8, dimension: usize) -> SteinerCategory {
    match family % 5 {
        0 => SteinerCategory::globe(dimension).unwrap(),
        1 => SteinerCategory::oriental(dimension).unwrap(),
        2 => SteinerCategory::cube(dimension.min(2)).unwrap(),
        3 => SteinerCategory::globe(dimension)
            .unwrap()
            .disjoint_union(&SteinerCategory::point())
            .unwrap(),
        _ => SteinerCategory::globe(1)
            .unwrap()
            .gray_tensor(&SteinerCategory::globe(dimension.min(2)).unwrap())
            .unwrap(),
    }
}

fn add_rationals(left: &Rational, right: &Rational) -> Rational {
    Q.add(left, right).unwrap()
}

fn sub_rationals(left: &Rational, right: &Rational) -> Rational {
    Q.sub(left, right).unwrap()
}

fn mul_rationals(left: &Rational, right: &Rational) -> Rational {
    Q.mul(left, right).unwrap()
}

#[test]
fn quadratic_order_and_ideal_hand_seed() {
    let field = quadratic_field(-1, "i");
    let value = quadratic_element(&field, 2.into(), 3.into());
    assert_eq!(value.trace(), Rational::from(4));
    assert_eq!(value.norm(), Rational::from(13));

    let order = field.equation_order().unwrap();
    assert_eq!(order.discriminant(), Integer::from(-4));
    let two = order.fractional_ideal(&[field.scalar(2.into())]).unwrap();
    let half = order
        .fractional_ideal(&[field.scalar(Rational::new(1.into(), 2.into()).unwrap())])
        .unwrap();
    assert_eq!(
        two.mul(&half).unwrap(),
        order.fractional_ideal(&[field.one()]).unwrap()
    );
}

#[test]
fn elliptic_group_law_hand_seed() {
    let curve = elliptic_curve(5);
    let points = elliptic_points(&curve);
    let point = points.iter().find(|point| !point.is_infinity()).unwrap();
    let origin = curve.origin().unwrap();
    assert_eq!(point.checked_add(&origin).unwrap(), point.clone());
    assert!(point
        .checked_add(&point.negated().unwrap())
        .unwrap()
        .is_infinity());
    point.verify().unwrap();
}

#[test]
fn category_and_topology_evidence_hand_seed() {
    let category = SteinerCategory::oriental(2).unwrap();
    let label = category.canonical_label().unwrap();
    assert!(label.verify(&category));
    assert!(category.street_nerve_skeleton(2).unwrap().verify());
    category.polygraphic_homology().unwrap().verify().unwrap();

    let disk = SimplicialComplex::from_facets(&[vec![0, 1, 2]]);
    let reduction = disk.strong_core_reduction().unwrap();
    reduction.verify().unwrap();
    assert_eq!(reduction.core().count(0), 1);
}

#[test]
fn real_and_complex_ball_hand_seed() {
    let field = RealBallField::new(96).unwrap();
    let third = Rational::new(1.into(), 3.into()).unwrap();
    let approximation = field
        .mul(&field.element(third.clone()), &field.element(3))
        .unwrap();
    assert!(approximation.overlaps(&field.one()).unwrap());

    let complex = ComplexBallField::new(96).unwrap();
    let imaginary_unit = complex.element_parts(0, 1);
    assert!(imaginary_unit
        .pow(2)
        .overlaps(&complex.element(-1))
        .unwrap());
}

#[test]
fn local_property_defaults_yield_to_an_explicit_stress_override() {
    let configured = property_config(48);
    match std::env::var("PROPTEST_CASES") {
        Ok(value) => {
            if let Ok(cases) = value.parse::<u32>() {
                assert_eq!(configured.cases, cases);
            }
        }
        Err(_) => assert_eq!(configured.cases, 48),
    }
}

proptest! {
    #![proptest_config(property_config(48))]

    #[test]
    fn quadratic_fields_orders_and_principal_ideals_obey_exact_laws(
        radicand in prop::sample::select(vec![-7_i64, -5, -3, -2, -1, 2, 3, 5, 6, 7]),
        coefficients in vec((-40_i16..=40, 0_u8..=7), 6),
        integral_coefficients in vec(-40_i16..=40, 6),
        first_generator in 1_i16..=8,
        second_generator in 1_i16..=8,
    ) {
        let field = quadratic_field(radicand, "a");
        let values = coefficients
            .iter()
            .map(|(numerator, denominator)| rational(*numerator, *denominator))
            .collect::<Vec<_>>();
        let left = quadratic_element(&field, values[0].clone(), values[1].clone());
        let middle = quadratic_element(&field, values[2].clone(), values[3].clone());
        let right = quadratic_element(&field, values[4].clone(), values[5].clone());
        assert_ring_laws(&field, &left, &middle, &right);

        let expected_trace = add_rationals(&values[0], &values[0]);
        let root_square = mul_rationals(&values[1], &values[1]);
        let expected_norm = sub_rationals(
            &mul_rationals(&values[0], &values[0]),
            &mul_rationals(&Rational::from(radicand), &root_square),
        );
        assert_eq!(left.trace(), expected_trace);
        assert_eq!(left.norm(), expected_norm);
        assert_eq!(field.add(&left, &middle).unwrap().trace(), add_rationals(&left.trace(), &middle.trace()));
        assert_eq!(field.mul(&left, &middle).unwrap().norm(), mul_rationals(&left.norm(), &middle.norm()));
        if left != field.zero() {
            assert_eq!(field.mul(&left, &left.inverse().unwrap()).unwrap(), field.one());
        }

        let field_json = field.to_json().unwrap();
        assert_eq!(field.to_json().unwrap(), field_json);
        assert_eq!(NumberField::from_json(&field_json).unwrap(), field);
        assert_eq!(NumberFieldElement::from_json(&left.to_json().unwrap()).unwrap(), left);

        let order = field.equation_order().unwrap();
        let integral = integral_coefficients
            .iter()
            .map(|coefficient| Rational::from(*coefficient))
            .collect::<Vec<_>>();
        let order_left = quadratic_element(&field, integral[0].clone(), integral[1].clone());
        let order_middle = quadratic_element(&field, integral[2].clone(), integral[3].clone());
        let order_right = quadratic_element(&field, integral[4].clone(), integral[5].clone());
        assert!(order.contains(&order_left));
        assert_ring_laws(&order, &order_left, &order_middle, &order_right);
        assert_eq!(NumberFieldOrder::from_json(&order.to_json().unwrap()).unwrap(), order);

        let first = Integer::from(first_generator);
        let second = Integer::from(second_generator);
        let first_ideal = order.fractional_ideal(&[field.scalar(Rational::from(first.clone()))]).unwrap();
        let second_ideal = order.fractional_ideal(&[field.scalar(Rational::from(second.clone()))]).unwrap();
        let expected = order
            .fractional_ideal(&[field.scalar(Rational::from(&first * &second))])
            .unwrap();
        assert_eq!(first_ideal.mul(&second_ideal).unwrap(), expected);
        assert_eq!(
            first_ideal.mul(&second_ideal).unwrap().norm(),
            mul_rationals(&first_ideal.norm(), &second_ideal.norm())
        );
        assert_eq!(FractionalIdeal::from_json(&first_ideal.to_json().unwrap()).unwrap(), first_ideal);
    }

    #[test]
    fn finite_field_elliptic_points_satisfy_the_full_group_laws(
        prime in prop::sample::select(vec![3_u64, 5, 7, 11, 13, 17, 19]),
        first_index in any::<usize>(),
        second_index in any::<usize>(),
        third_index in any::<usize>(),
        first_scalar in -12_i16..=12,
        second_scalar in -12_i16..=12,
    ) {
        let curve = elliptic_curve(prime);
        let points = elliptic_points(&curve);
        let left = &points[first_index % points.len()];
        let middle = &points[second_index % points.len()];
        let right = &points[third_index % points.len()];
        let origin = curve.origin().unwrap();

        assert_eq!(
            left.checked_add(middle).unwrap().checked_add(right).unwrap(),
            left.checked_add(&middle.checked_add(right).unwrap()).unwrap()
        );
        assert_eq!(left.checked_add(&origin).unwrap(), left.clone());
        assert!(left.checked_add(&left.negated().unwrap()).unwrap().is_infinity());

        let first_scalar = Integer::from(first_scalar);
        let second_scalar = Integer::from(second_scalar);
        let scalar_sum = &first_scalar + &second_scalar;
        assert_eq!(
            left.scalar_mul(&scalar_sum).unwrap(),
            left.scalar_mul(&first_scalar)
                .unwrap()
                .checked_add(&left.scalar_mul(&second_scalar).unwrap())
                .unwrap()
        );
        assert_eq!(
            left.checked_add(middle).unwrap().scalar_mul(&first_scalar).unwrap(),
            left.scalar_mul(&first_scalar)
                .unwrap()
                .checked_add(&middle.scalar_mul(&first_scalar).unwrap())
                .unwrap()
        );
        left.verify().unwrap();

        let curve_json = curve.to_json().unwrap();
        assert_eq!(curve.to_json().unwrap(), curve_json);
        assert_eq!(EllipticCurve::<FiniteField>::from_json(&curve_json).unwrap(), curve);
        assert_eq!(
            EllipticCurvePoint::<FiniteField>::from_json(&left.to_json().unwrap()).unwrap(),
            *left
        );
    }

    #[test]
    fn steiner_evidence_is_replayable_and_deterministic(
        family in any::<u8>(),
        dimension in 0_usize..=3,
    ) {
        let category = category_case(family, dimension);
        category.verify().unwrap();
        let json = serde_json::to_string(&category).unwrap();
        assert_eq!(serde_json::to_string(&category).unwrap(), json);
        assert_eq!(serde_json::from_str::<SteinerCategory>(&json).unwrap(), category);

        let first_label = category.canonical_label().unwrap();
        let second_label = category.canonical_label().unwrap();
        assert!(first_label.verify(&category));
        assert_eq!(first_label.encoding(), second_label.encoding());
        assert_eq!(first_label.category(), second_label.category());
        assert_eq!(category.canonical_form().unwrap().canonical_form().unwrap(), category.canonical_form().unwrap());

        let first_nerve = category.street_nerve_skeleton(2).unwrap();
        let second_nerve = category.street_nerve_skeleton(2).unwrap();
        assert!(first_nerve.verify());
        assert_eq!(first_nerve, second_nerve);

        let full = category.polygraphic_homology().unwrap();
        full.verify().unwrap();
        let lean = category.polygraphic_homology_invariants().unwrap();
        for group in full.groups() {
            assert_eq!(lean[&group.degree()].free_rank(), group.free_rank());
            assert_eq!(lean[&group.degree()].torsion_coefficients(), group.torsion_coefficients());
        }
    }

    #[test]
    fn simplicial_evidence_replays_and_euler_matches_field_homology(
        vertex_count in 1_usize..=6,
        masks in vec(any::<u8>(), 1..=8),
    ) {
        let complex = simplicial_complex(vertex_count, &masks);
        let mut reordered_masks = masks.clone();
        reordered_masks.reverse();
        reordered_masks.extend(masks.iter().copied());
        assert_eq!(complex, simplicial_complex(vertex_count, &reordered_masks));

        let json = complex.to_json().unwrap();
        assert_eq!(complex.to_json().unwrap(), json);
        assert_eq!(SimplicialComplex::from_json(&json).unwrap(), complex);
        complex.chain_complex(Q).unwrap().verify().unwrap();

        let homological_euler = (0..=complex.dimension() as usize)
            .map(|degree| {
                let rank = complex.homology(Q, degree).unwrap().dimension() as i64;
                if degree % 2 == 0 { rank } else { -rank }
            })
            .sum::<i64>();
        assert_eq!(homological_euler, complex.euler_characteristic());

        let reduction = complex.strong_core_reduction().unwrap();
        reduction.verify().unwrap();
        let restored = StrongCoreReduction::from_json(&reduction.to_json().unwrap()).unwrap();
        assert_eq!(restored, reduction);
        assert_eq!(reduction.core().strong_core().unwrap(), reduction.core().clone());
    }

    #[test]
    fn ball_arithmetic_encloses_exact_real_and_complex_formulas(
        precision in prop::sample::select(vec![32_u64, 53, 80, 96, 128, 192, 256]),
        numerators in vec(-1_000_i16..=1_000, 8),
        denominators in vec(0_u8..=15, 8),
    ) {
        let values = numerators
            .iter()
            .zip(&denominators)
            .map(|(numerator, denominator)| rational(*numerator, *denominator))
            .collect::<Vec<_>>();
        let real = RealBallField::new(precision).unwrap();
        let left = real.element(values[0].clone());
        let middle = real.element(values[1].clone());
        let right = real.element(values[2].clone());
        let sum = real.add(&left, &middle).unwrap();
        let product = real.mul(&left, &middle).unwrap();
        assert!(sum.overlaps(&real.element(add_rationals(&values[0], &values[1]))).unwrap());
        assert!(product.overlaps(&real.element(mul_rationals(&values[0], &values[1]))).unwrap());
        let distributed = real.mul(&left, &real.add(&middle, &right).unwrap()).unwrap();
        let expanded = real
            .add(&real.mul(&left, &middle).unwrap(), &real.mul(&left, &right).unwrap())
            .unwrap();
        assert!(distributed.overlaps(&expanded).unwrap());
        assert_eq!(RealBall::from_json(&product.to_json().unwrap()).unwrap(), product);

        let complex = ComplexBallField::new(precision).unwrap();
        let complex_left = complex.element_parts(values[0].clone(), values[1].clone());
        let complex_right = complex.element_parts(values[2].clone(), values[3].clone());
        let complex_product = complex.mul(&complex_left, &complex_right).unwrap();
        let expected_real = sub_rationals(
            &mul_rationals(&values[0], &values[2]),
            &mul_rationals(&values[1], &values[3]),
        );
        let expected_imaginary = add_rationals(
            &mul_rationals(&values[0], &values[3]),
            &mul_rationals(&values[1], &values[2]),
        );
        assert!(complex_product
            .overlaps(&complex.element_parts(expected_real, expected_imaginary))
            .unwrap());

        let second_left = complex.element_parts(values[4].clone(), values[5].clone());
        let second_right = complex.element_parts(values[6].clone(), values[7].clone());
        let distributed = complex
            .mul(&complex_left, &complex.add(&second_left, &second_right).unwrap())
            .unwrap();
        let expanded = complex
            .add(
                &complex.mul(&complex_left, &second_left).unwrap(),
                &complex.mul(&complex_left, &second_right).unwrap(),
            )
            .unwrap();
        assert!(distributed.overlaps(&expanded).unwrap());
        assert_eq!(ComplexBall::from_json(&complex_product.to_json().unwrap()).unwrap(), complex_product);
    }
}
