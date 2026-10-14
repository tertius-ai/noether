// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    PowerSeries, PowerSeriesError, PowerSeriesRing, ResidueRing, SeriesTruncation,
    SeriesTruncationMode, WeightedGrading,
};
use noether::linalg::Matrix;
use noether::numthy::{FiniteField, PadicRing, QadicRing};
use noether::{
    CommutativeRing, Integer, PrecisionContext, Ring, SemanticType, UnitError, UnitInversion, Q, Z,
};
use serde::de::DeserializeOwned;
use serde::Serialize;

fn integer_series_ring(
    variables: &[&str],
    truncation: SeriesTruncation,
) -> PowerSeriesRing<noether::IntegerRing> {
    PowerSeriesRing::new(Z, variables.iter().copied(), truncation).unwrap()
}

#[test]
fn truncation_policies_are_finite_typed_context_identity() {
    let univariate = SeriesTruncation::univariate(5).unwrap();
    let total = SeriesTruncation::total_degree(5).unwrap();
    let boxed = SeriesTruncation::box_bounds(vec![2, 3]).unwrap();
    let weighted = SeriesTruncation::weighted_degree(vec![1, 2], 5).unwrap();

    assert_eq!(univariate.mode(), SeriesTruncationMode::Univariate);
    assert_eq!(total.mode(), SeriesTruncationMode::TotalDegree);
    assert_eq!(boxed.mode(), SeriesTruncationMode::Box);
    assert_eq!(weighted.mode(), SeriesTruncationMode::WeightedDegree);
    assert!(univariate.contains_exponent(&[4]));
    assert!(!univariate.contains_exponent(&[5]));
    assert!(total.contains_exponent(&[2, 2]));
    assert!(!total.contains_exponent(&[3, 2]));
    assert!(boxed.contains_exponent(&[1, 2]));
    assert!(!boxed.contains_exponent(&[2, 0]));
    assert!(weighted.contains_exponent(&[0, 2]));
    assert!(!weighted.contains_exponent(&[1, 2]));
    assert!(SeriesTruncation::total_degree(7).unwrap().refines(&total));
    assert!(!total.refines(&weighted));

    let total_ring = integer_series_ring(&["x", "y"], total);
    let box_ring = integer_series_ring(&["x", "y"], boxed);
    let weighted_ring = integer_series_ring(&["u", "v"], weighted);
    assert_eq!(total_ring.coefficient_capacity_bound(), Integer::from(25));
    assert_eq!(box_ring.coefficient_capacity_bound(), Integer::from(6));
    assert_eq!(
        weighted_ring.coefficient_capacity_bound(),
        Integer::from(15)
    );
    assert_eq!(weighted_ring.precision(), weighted_ring.truncation());
}

#[test]
fn positive_gradings_define_exact_fixed_storage_regions() {
    let standard = WeightedGrading::standard(3).unwrap();
    let grading = WeightedGrading::new(vec![1, 2, 3]).unwrap();
    assert_eq!(standard.weights(), &[1, 1, 1]);
    assert!(standard.is_standard());
    assert_eq!(grading.degree(&[2, 1, 1]).unwrap(), 7);
    assert!(WeightedGrading::new(vec![1, 0, 3]).is_err());
    assert!(grading.degree(&[1, 2]).is_err());

    let truncation = SeriesTruncation::graded(grading.clone(), 6).unwrap();
    let retained = (0..6)
        .flat_map(|x| (0..3).flat_map(move |y| (0..2).map(move |z| vec![x, y, z])))
        .filter(|exponent| truncation.contains_exponent(exponent))
        .collect::<Vec<_>>();
    assert_eq!(
        retained,
        vec![
            vec![0, 0, 0],
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![0, 1, 1],
            vec![0, 2, 0],
            vec![1, 0, 0],
            vec![1, 0, 1],
            vec![1, 1, 0],
            vec![1, 2, 0],
            vec![2, 0, 0],
            vec![2, 0, 1],
            vec![2, 1, 0],
            vec![3, 0, 0],
            vec![3, 1, 0],
            vec![4, 0, 0],
            vec![5, 0, 0],
        ]
    );

    let ring = PowerSeriesRing::with_grading(Z, ["x", "y", "z"], grading.clone(), 6).unwrap();
    assert_eq!(ring.grading(), Some(grading));
    assert_eq!(ring.coefficient_capacity_bound(), Integer::from(36));
    assert_eq!(ring.generator(0).unwrap().ring(), &ring);

    let default_ring = PowerSeriesRing::with_precision(Z, ["x", "y", "z"], 6).unwrap();
    assert_eq!(
        default_ring.grading(),
        Some(WeightedGrading::standard(3).unwrap())
    );
}

#[test]
fn weighted_gradings_have_revalidated_typed_json() {
    let grading = WeightedGrading::new(vec![1, 2, 3]).unwrap();
    assert_eq!(
        WeightedGrading::from_json(&grading.to_json().unwrap()).unwrap(),
        grading
    );
    assert!(WeightedGrading::from_json(
        r#"{"type":"noether/commalg/WeightedGrading@1","weights":[1,0,3]}"#,
    )
    .is_err());
}

fn assert_series_json_round_trip<R>(ring: &PowerSeriesRing<R>, value: &PowerSeries<R>)
where
    R: CommutativeRing + Serialize + DeserializeOwned,
    R::Element: Serialize + DeserializeOwned,
{
    assert_eq!(
        PowerSeriesRing::<R>::from_json(&ring.to_json().unwrap()).unwrap(),
        *ring
    );
    assert_eq!(
        PowerSeries::<R>::from_json(&value.to_json().unwrap()).unwrap(),
        *value
    );
}

#[test]
fn truncations_and_generic_series_have_revalidated_typed_json() {
    for truncation in [
        SeriesTruncation::univariate(5).unwrap(),
        SeriesTruncation::total_degree(5).unwrap(),
        SeriesTruncation::box_bounds(vec![2, 3]).unwrap(),
        SeriesTruncation::weighted_degree(vec![1, 2], 5).unwrap(),
    ] {
        assert_eq!(
            SeriesTruncation::from_json(&truncation.to_json().unwrap()).unwrap(),
            truncation
        );
    }

    let integers = PowerSeriesRing::new(
        Z,
        ["x", "y"],
        SeriesTruncation::weighted_degree(vec![1, 2], 6).unwrap(),
    )
    .unwrap();
    let integer_value = integers
        .from_terms([
            (vec![0, 1], Integer::from(5_u8)),
            (vec![2, 0], Integer::from(3_u8)),
        ])
        .unwrap();
    assert_series_json_round_trip(&integers, &integer_value);

    let rationals =
        PowerSeriesRing::new(Q, ["t"], SeriesTruncation::univariate(4).unwrap()).unwrap();
    let rational_value = rationals
        .from_terms([(vec![1], noether::Rational::from(2_i8))])
        .unwrap();
    assert_series_json_round_trip(&rationals, &rational_value);

    let residue = ResidueRing::new(12_u8).unwrap();
    let residues = PowerSeriesRing::new(
        residue.clone(),
        ["t"],
        SeriesTruncation::univariate(4).unwrap(),
    )
    .unwrap();
    let residue_value = residues
        .from_terms([(vec![1], residue.element(5_u8))])
        .unwrap();
    assert_series_json_round_trip(&residues, &residue_value);

    let finite = FiniteField::extension(3, 2).unwrap();
    let finite_series = PowerSeriesRing::new(
        finite.clone(),
        ["t"],
        SeriesTruncation::univariate(4).unwrap(),
    )
    .unwrap();
    let finite_value = finite_series
        .from_terms([(vec![1], finite.generator())])
        .unwrap();
    assert_series_json_round_trip(&finite_series, &finite_value);

    let padic = PadicRing::new(3_u8, 4).unwrap();
    let padic_series = PowerSeriesRing::new(
        padic.clone(),
        ["t"],
        SeriesTruncation::univariate(4).unwrap(),
    )
    .unwrap();
    let padic_value = padic_series
        .from_terms([(vec![1], padic.element(-1_i8))])
        .unwrap();
    assert_series_json_round_trip(&padic_series, &padic_value);

    let qadic = QadicRing::new(finite, 4).unwrap();
    let qadic_series = PowerSeriesRing::new(
        qadic.clone(),
        ["t"],
        SeriesTruncation::univariate(4).unwrap(),
    )
    .unwrap();
    let qadic_value = qadic_series
        .from_terms([(vec![1], qadic.generator())])
        .unwrap();
    assert_series_json_round_trip(&qadic_series, &qadic_value);

    let mut unordered: serde_json::Value =
        serde_json::from_str(&integer_value.to_json().unwrap()).unwrap();
    unordered["terms"].as_array_mut().unwrap().reverse();
    assert!(PowerSeries::<noether::IntegerRing>::from_json(&unordered.to_string()).is_err());

    let mut zero_term: serde_json::Value =
        serde_json::from_str(&integer_value.to_json().unwrap()).unwrap();
    zero_term["terms"][0]["coefficient"] = serde_json::Value::String("0".to_owned());
    assert!(PowerSeries::<noether::IntegerRing>::from_json(&zero_term.to_string()).is_err());

    let malformed = r#"{"type":"noether/commalg/SeriesTruncation@1","mode":"box","bound":3}"#;
    assert!(SeriesTruncation::from_json(malformed).is_err());
}

#[test]
fn construction_arithmetic_and_derivatives_use_one_generic_ring_floor() {
    let ring = integer_series_ring(&["x", "y"], SeriesTruncation::total_degree(4).unwrap());
    let f = ring
        .from_terms([
            (vec![0, 0], Integer::from(1)),
            (vec![1, 0], Integer::from(2)),
            (vec![0, 2], Integer::from(3)),
        ])
        .unwrap();
    let g = ring
        .from_terms([
            (vec![1, 0], Integer::from(1)),
            (vec![0, 1], Integer::from(1)),
        ])
        .unwrap();
    let product = ring.mul(&f, &g).unwrap();
    let doubled = f.scalar_mul(&Integer::from(2)).unwrap();

    assert_eq!(product.coefficient(&[1, 0]).unwrap(), Integer::from(1));
    assert_eq!(product.coefficient(&[0, 1]).unwrap(), Integer::from(1));
    assert_eq!(product.coefficient(&[2, 0]).unwrap(), Integer::from(2));
    assert_eq!(product.coefficient(&[1, 1]).unwrap(), Integer::from(2));
    assert_eq!(product.coefficient(&[1, 2]).unwrap(), Integer::from(3));
    assert_eq!(product.coefficient(&[0, 3]).unwrap(), Integer::from(3));
    assert_eq!(doubled.coefficient(&[0, 2]).unwrap(), Integer::from(6));
    assert_eq!(
        ring.generators().unwrap(),
        [ring.generator(0).unwrap(), ring.generator(1).unwrap()]
    );
    let derivative_x = f.derivative(0).unwrap();
    assert_eq!(derivative_x.constant_coefficient(), Integer::from(2));
    assert_eq!(derivative_x.ring().truncation().bound(), Some(3));
    assert_eq!(
        f.derivative(1).unwrap().coefficient(&[0, 1]).unwrap(),
        Integer::from(6)
    );
    assert!(matches!(
        f.coefficient(&[4, 0]),
        Err(PowerSeriesError::ExponentOutsideTruncation { .. })
    ));
    assert!(ring.from_terms([(vec![4, 0], Integer::from(1))]).is_err());

    let precision_one = integer_series_ring(&["x"], SeriesTruncation::univariate(1).unwrap());
    assert!(matches!(
        precision_one.one().derivative(0),
        Err(PowerSeriesError::InsufficientDerivativePrecision { .. })
    ));
}

#[test]
fn weighted_padic_series_have_fixed_memory_and_precision_contexts() {
    let padics = PadicRing::new(3_u8, 3).unwrap();
    let ring = PowerSeriesRing::new(
        padics.clone(),
        ["u", "v"],
        SeriesTruncation::weighted_degree(vec![1, 2], 5).unwrap(),
    )
    .unwrap();
    let f = ring
        .from_terms([
            (vec![0, 0], padics.element(1_u8)),
            (vec![1, 0], padics.element(3_u8)),
            (vec![0, 1], padics.element(2_u8)),
        ])
        .unwrap();
    let square = ring.mul(&f, &f).unwrap();

    assert_eq!(
        square.coefficient(&[0, 0]).unwrap().representative(),
        Some(Integer::from(1))
    );
    assert_eq!(
        square.coefficient(&[1, 0]).unwrap().representative(),
        Some(Integer::from(6))
    );
    assert_eq!(
        square.coefficient(&[0, 1]).unwrap().representative(),
        Some(Integer::from(4))
    );
    assert_eq!(
        square.coefficient(&[1, 1]).unwrap().representative(),
        Some(Integer::from(12))
    );
    assert_eq!(
        square.coefficient(&[0, 2]).unwrap().representative(),
        Some(Integer::from(4))
    );
    assert_eq!(ring.coefficient_capacity_bound(), Integer::from(15));
    assert_eq!(ring.coefficient_ring().absolute_precision(), 3);
}

#[test]
fn weighted_qadic_series_bound_every_coefficient_and_monomial_slot() {
    let residue = FiniteField::extension(3, 2).unwrap();
    let witt = QadicRing::new(residue, 6).unwrap();
    let ring = PowerSeriesRing::new(
        witt.clone(),
        ["x", "y", "z"],
        SeriesTruncation::weighted_degree(vec![1, 2, 3], 6).unwrap(),
    )
    .unwrap();
    let x = ring.generator(0).unwrap();
    let y = ring.generator(1).unwrap();
    let z = ring.generator(2).unwrap();
    let value = ring
        .add(
            &ring.add(&x.pow(2).unwrap(), &y).unwrap(),
            &z.scalar_mul(&witt.generator()).unwrap(),
        )
        .unwrap();
    let square = ring.mul(&value, &value).unwrap();

    assert_eq!(ring.coefficient_capacity_bound(), Integer::from(36));
    assert_eq!(ring.coefficient_ring().absolute_precision(), 6);
    assert_eq!(square.coefficient(&[4, 0, 0]).unwrap(), witt.one());
    assert_eq!(square.coefficient(&[2, 1, 0]).unwrap(), witt.element(2_u8));
    assert!(square.coefficient(&[6, 0, 0]).is_err());
    assert!(ring.from_terms([(vec![0, 0, 2], witt.one())]).is_err());
}

#[test]
fn unit_inversion_works_over_fields_and_nonfield_coefficient_rings() {
    let padics = PadicRing::new(3_u8, 3).unwrap();
    let ring = PowerSeriesRing::new(
        padics.clone(),
        ["u"],
        SeriesTruncation::univariate(5).unwrap(),
    )
    .unwrap();
    let u = ring.generator(0).unwrap();
    let one_plus_u = ring.add(&ring.one(), &u).unwrap();
    let inverse = one_plus_u.inverse().unwrap();

    assert_eq!(ring.mul(&one_plus_u, &inverse).unwrap(), ring.one());
    assert_eq!(
        inverse.coefficient(&[0]).unwrap().representative(),
        Some(Integer::from(1))
    );
    assert_eq!(
        inverse.coefficient(&[1]).unwrap().representative(),
        Some(Integer::from(26))
    );
    assert_eq!(
        inverse.coefficient(&[4]).unwrap().representative(),
        Some(Integer::from(1))
    );
    assert_eq!(ring.inverse_unit(&one_plus_u).unwrap(), inverse);

    let nonunit = ring.constant(padics.element(3_u8)).unwrap();
    assert!(matches!(nonunit.inverse(), Err(UnitError::NotAUnit { .. })));
}

#[test]
fn exact_division_uses_a_canonical_checked_truncated_quotient() {
    let ring = integer_series_ring(&["x", "y"], SeriesTruncation::total_degree(6).unwrap());
    let x = ring.generator(0).unwrap();
    let y = ring.generator(1).unwrap();
    let unit = ring.add(&ring.one(), &y).unwrap();
    let divisor = ring.mul(&x, &unit).unwrap();
    let quotient = ring
        .from_terms([
            (vec![0, 0], Integer::from(1)),
            (vec![1, 0], Integer::from(2)),
            (vec![0, 2], Integer::from(-3)),
        ])
        .unwrap();
    let dividend = ring.mul(&divisor, &quotient).unwrap();

    let recovered = dividend.exact_div(&divisor).unwrap();
    assert_eq!(recovered, quotient);
    assert_eq!(ring.mul(&divisor, &recovered).unwrap(), dividend);
    assert_eq!(ring.zero().exact_div(&divisor).unwrap(), ring.zero());
    assert!(matches!(
        x.exact_div(&y),
        Err(PowerSeriesError::NotExactlyDivisible)
    ));
    assert!(matches!(
        ring.zero().exact_div(&ring.zero()),
        Err(PowerSeriesError::DivisionByZero)
    ));
    assert!(matches!(
        ring.zero().exact_div(&ring.add(&x, &y).unwrap()),
        Err(PowerSeriesError::DivisorNotMonomialTimesUnit)
    ));

    let other = integer_series_ring(&["x", "y"], SeriesTruncation::total_degree(5).unwrap());
    assert!(matches!(
        dividend.exact_div(&other.one()),
        Err(PowerSeriesError::ContextMismatch(_))
    ));

    let boundary_ring = integer_series_ring(&["t"], SeriesTruncation::univariate(4).unwrap());
    let t = boundary_ring.generator(0).unwrap();
    let boundary_divisor = boundary_ring
        .mul(&t, &boundary_ring.add(&boundary_ring.one(), &t).unwrap())
        .unwrap();
    let canonical = t.exact_div(&boundary_divisor).unwrap();
    assert_eq!(canonical.coefficient(&[0]).unwrap(), Integer::from(1));
    assert_eq!(canonical.coefficient(&[1]).unwrap(), Integer::from(-1));
    assert_eq!(canonical.coefficient(&[2]).unwrap(), Integer::from(1));
    assert_eq!(canonical.coefficient(&[3]).unwrap(), Integer::from(-1));
    assert_eq!(boundary_ring.mul(&boundary_divisor, &canonical).unwrap(), t);
}

#[test]
fn composition_checks_that_substitution_descends_to_the_quotient() {
    let ring = integer_series_ring(
        &["u", "v"],
        SeriesTruncation::weighted_degree(vec![1, 2], 5).unwrap(),
    );
    let u = ring.generator(0).unwrap();
    let v = ring.generator(1).unwrap();
    let u_squared = u.pow(2).unwrap();

    assert_eq!(
        v.compose(&[u.clone(), u_squared.clone()]).unwrap(),
        u_squared
    );
    assert!(matches!(
        v.compose(&[u.clone(), u]),
        Err(PowerSeriesError::InvalidComposition { variable: 1, .. })
    ));

    let box_ring = integer_series_ring(
        &["x", "y"],
        SeriesTruncation::box_bounds(vec![2, 3]).unwrap(),
    );
    let x = box_ring.generator(0).unwrap();
    let y = box_ring.generator(1).unwrap();
    assert!(x.compose(&[y.clone(), y]).is_err());
}

#[test]
fn truncation_restriction_discards_terms_only_through_an_explicit_map() {
    let ring = integer_series_ring(
        &["u", "v"],
        SeriesTruncation::weighted_degree(vec![1, 2], 5).unwrap(),
    );
    let value = ring
        .from_terms([
            (vec![0, 0], Integer::from(1)),
            (vec![2, 0], Integer::from(2)),
            (vec![0, 1], Integer::from(3)),
            (vec![4, 0], Integer::from(4)),
        ])
        .unwrap();
    let restricted = value
        .restrict_truncation(SeriesTruncation::weighted_degree(vec![1, 2], 3).unwrap())
        .unwrap();

    assert_eq!(restricted.coefficient(&[0, 0]).unwrap(), Integer::from(1));
    assert_eq!(restricted.coefficient(&[2, 0]).unwrap(), Integer::from(2));
    assert_eq!(restricted.coefficient(&[0, 1]).unwrap(), Integer::from(3));
    assert!(restricted.coefficient(&[4, 0]).is_err());
    assert!(value
        .restrict_truncation(SeriesTruncation::total_degree(3).unwrap())
        .is_err());
}

#[test]
fn context_identity_is_serialization_stable_and_matrices_are_automatic() {
    let ring = integer_series_ring(&["x"], SeriesTruncation::univariate(4).unwrap());
    let other_name = integer_series_ring(&["t"], SeriesTruncation::univariate(4).unwrap());
    let other_precision = integer_series_ring(&["x"], SeriesTruncation::univariate(5).unwrap());
    let x = ring.generator(0).unwrap();
    let matrix = Matrix::from_rows(
        ring.clone(),
        [[ring.one(), x.clone()], [x.clone(), ring.one()]],
    )
    .unwrap();

    assert_ne!(
        ring.canonical_bytes().unwrap(),
        other_name.canonical_bytes().unwrap()
    );
    assert_ne!(
        ring.canonical_bytes().unwrap(),
        other_precision.canonical_bytes().unwrap()
    );
    assert_eq!(matrix.matmul(&matrix).unwrap().shape(), (2, 2));
    assert_eq!(
        matrix
            .matmul(&matrix)
            .unwrap()
            .get(0, 1)
            .unwrap()
            .coefficient(&[1])
            .unwrap(),
        Integer::from(2)
    );

    let foreign = other_name.one();
    assert!((&ring.one() + &foreign).is_err());
    assert!(matches!(
        other_name.inverse_unit(&ring.one()),
        Err(UnitError::ContextMismatch(_))
    ));
    assert_send_sync::<PowerSeriesRing<noether::IntegerRing>>();
    assert_send_sync::<PowerSeries<noether::IntegerRing>>();
}

#[test]
fn invalid_contexts_fail_before_allocating_series() {
    assert!(
        PowerSeriesRing::new(Z, ["x", "x"], SeriesTruncation::total_degree(3).unwrap()).is_err()
    );
    assert!(PowerSeriesRing::new(Z, ["x", "y"], SeriesTruncation::univariate(3).unwrap()).is_err());
    assert!(SeriesTruncation::total_degree(0).is_err());
    assert!(SeriesTruncation::weighted_degree(vec![1, 0], 4).is_err());
}

fn assert_send_sync<T: Send + Sync>() {}
