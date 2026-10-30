// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{MonomialOrder, PolynomialError, PolynomialRing, ResidueRing};
use noether::numthy::{FiniteField, QadicRing};
use noether::{Integer, Rational, Ring, Q, Z};
use proptest::prelude::*;

#[test]
fn bezout_normalizes_and_handles_zero_inputs() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
    let x = ring.generator(0).unwrap();
    let f = (&x.pow(2).unwrap() - &ring.one()).unwrap();
    let g = (&x - &ring.one()).unwrap();
    let (d, s, t) = f.extended_gcd(&g).unwrap();
    assert_eq!(d, g);
    assert_eq!((&(&s * &f).unwrap() + &(&t * &g).unwrap()).unwrap(), d);
    assert_eq!(
        ring.zero().extended_gcd(&ring.zero()).unwrap(),
        (ring.zero(), ring.one(), ring.zero())
    );
    let two_f = f.scalar_mul(&Rational::from(2)).unwrap();
    assert_eq!(two_f.extended_gcd(&ring.zero()).unwrap().0, f);
    assert!(matches!(
        g.inverse_mod(&f),
        Err(PolynomialError::NotCoprime)
    ));
    let modulus = (&x.pow(2).unwrap() + &ring.one()).unwrap();
    let inverse = x.inverse_mod(&modulus).unwrap();
    assert_eq!(
        (&x * &inverse).unwrap().div_rem(&modulus).unwrap().1,
        ring.one()
    );
    let other_ring = PolynomialRing::new(Q, ["y"], MonomialOrder::default()).unwrap();
    assert!(f.extended_gcd(&other_ring.one()).is_err());
    let multivariate = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
    assert!(multivariate
        .one()
        .extended_gcd(&multivariate.one())
        .is_err());
}

fn monic_reconstruction<R: noether::CommutativeRing>(ring: PolynomialRing<R>) {
    let x = ring.generator(0).unwrap();
    let divisor = (&x.pow(2).unwrap() + &ring.one()).unwrap();
    let dividend = (&x.pow(7).unwrap() + &x).unwrap();
    let (q, r) = dividend.monic_div_rem(&divisor).unwrap();
    assert_eq!((&(&q * &divisor).unwrap() + &r).unwrap(), dividend);
    assert!(r.degree(None).unwrap().is_none_or(|d| d < 2));
    assert_eq!(
        ring.zero().monic_div_rem(&divisor).unwrap(),
        (ring.zero(), ring.zero())
    );
    assert!(matches!(
        dividend.monic_div_rem(&ring.zero()),
        Err(PolynomialError::DivisionByZero)
    ));
}

#[test]
fn monic_division_works_over_rings_with_zero_divisors() {
    monic_reconstruction(PolynomialRing::new(Z, ["x"], MonomialOrder::default()).unwrap());
    let residues = ResidueRing::new(9).unwrap();
    let ring = PolynomialRing::new(residues.clone(), ["x"], MonomialOrder::default()).unwrap();
    monic_reconstruction(ring.clone());
    let nonmonic = ring
        .generator(0)
        .unwrap()
        .scalar_mul(&residues.element(3))
        .unwrap();
    assert!(matches!(
        ring.one().monic_div_rem(&nonmonic),
        Err(PolynomialError::NonMonicDivisor)
    ));
    let witt = QadicRing::new(FiniteField::extension(3, 2).unwrap(), 7).unwrap();
    monic_reconstruction(PolynomialRing::new(witt, ["x"], MonomialOrder::default()).unwrap());
}

// The original value-returning long division is an independent arithmetic
// floor for the support-updating implementation used by Kedlaya reduction.
fn monic_reference<R: noether::CommutativeRing>(
    dividend: &noether::commalg::Polynomial<R>,
    divisor: &noether::commalg::Polynomial<R>,
) -> (
    noether::commalg::Polynomial<R>,
    noether::commalg::Polynomial<R>,
) {
    let ring = dividend.ring();
    let degree = divisor.degree(None).unwrap().unwrap();
    let mut quotient = ring.zero();
    let mut remainder = dividend.clone();
    while let Some(current) = remainder.degree(None).unwrap() {
        if current < degree {
            break;
        }
        let term = ring
            .from_terms([(
                vec![&current - &degree],
                remainder.leading_coefficient().unwrap(),
            )])
            .unwrap();
        quotient = (&quotient + &term).unwrap();
        remainder = (&remainder - &(&term * divisor).unwrap()).unwrap();
    }
    (quotient, remainder)
}

#[test]
fn monic_division_dense_regression_and_sparse_arbitrary_exponents() {
    let witt = QadicRing::new(FiniteField::extension(13, 2).unwrap(), 17).unwrap();
    let ring = PolynomialRing::new(witt.clone(), ["x"], MonomialOrder::default()).unwrap();
    let dividend = ring
        .from_terms((0..193).map(|i| {
            (
                vec![Integer::from(i)],
                witt.add(
                    &witt.element(17 * i + 3),
                    &witt.mul(&witt.generator(), &witt.element(i)).unwrap(),
                )
                .unwrap(),
            )
        }))
        .unwrap();
    let divisor = ring
        .from_terms((0..12).map(|i| {
            (
                vec![Integer::from(i)],
                witt.element(if i == 11 { 1 } else { i + 2 }),
            )
        }))
        .unwrap();
    assert_eq!(
        dividend.monic_div_rem(&divisor).unwrap(),
        monic_reference(&dividend, &divisor)
    );
    assert_eq!(
        dividend.monic_div_rem(&ring.one()).unwrap(),
        (dividend.clone(), ring.zero())
    );
    assert_eq!(
        ring.one().monic_div_rem(&divisor).unwrap(),
        (ring.zero(), ring.one())
    );
    let other = PolynomialRing::new(
        QadicRing::new(FiniteField::extension(13, 2).unwrap(), 18).unwrap(),
        ["x"],
        MonomialOrder::default(),
    )
    .unwrap();
    assert!(dividend.monic_div_rem(&other.one()).is_err());

    // This cannot rely on a dense-degree allocation or the q-adic regression.
    let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::default()).unwrap();
    let huge: Integer = "184467440737095516160000".parse().unwrap();
    let x_huge = ring.from_terms([(vec![huge], Integer::one())]).unwrap();
    let divisor = (&x_huge + &ring.one()).unwrap();
    let quotient = (&x_huge + &ring.constant(Integer::from(2)).unwrap()).unwrap();
    let dividend = (&divisor * &quotient).unwrap();
    assert_eq!(
        dividend.monic_div_rem(&divisor).unwrap(),
        (quotient, ring.zero())
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn monic_division_matches_reference_with_cancellation(
        a in proptest::collection::vec(0_u64..27, 0..55),
        b in proptest::collection::vec(0_u64..27, 0..8),
    ) {
        let coefficients = ResidueRing::new(27).unwrap();
        let ring = PolynomialRing::new(coefficients.clone(), ["x"], MonomialOrder::default()).unwrap();
        let make = |v: &[u64]| ring.from_terms(v.iter().enumerate().map(|(i, c)| (vec![Integer::from(i)], coefficients.element(*c)))).unwrap();
        let dividend = make(&a);
        let divisor = (&make(&b) + &ring.from_terms([(vec![Integer::from(b.len())], coefficients.one())]).unwrap()).unwrap();
        let (q, r) = dividend.monic_div_rem(&divisor).unwrap();
        prop_assert_eq!((q.clone(), r.clone()), monic_reference(&dividend, &divisor));
        prop_assert_eq!((&(&q * &divisor).unwrap() + &r).unwrap(), dividend);
        prop_assert!(r.degree(None).unwrap().is_none_or(|degree| degree < Integer::from(b.len())));
    }
}

proptest! {
    #[test]
    fn bezout_and_modular_inversion_reconstruct(a in proptest::collection::vec(0_u64..7, 0..8), b in proptest::collection::vec(0_u64..7, 0..8)) {
        let field = FiniteField::prime(7).unwrap();
        let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
        let make = |v: &[u64]| ring.from_terms(v.iter().enumerate().map(|(i, c)| (vec![Integer::from(i)], field.element(*c)))).unwrap();
        let (a, b) = (make(&a), make(&b));
        let (g, s, t) = a.extended_gcd(&b).unwrap();
        prop_assert_eq!((&(&s * &a).unwrap() + &(&t * &b).unwrap()).unwrap(), g.clone());
        prop_assert_eq!(g.clone(), a.gcd(&b).unwrap());
        if g == ring.one() && !b.is_zero() {
            let inverse = a.inverse_mod(&b).unwrap();
            prop_assert_eq!((&a * &inverse).unwrap().div_rem(&b).unwrap().1, ring.one().div_rem(&b).unwrap().1);
        }
    }
}
