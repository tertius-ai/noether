// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Exact univariate acceptance cases extracted from arXiv:1906.02261.

use std::str::FromStr;

use noether::commalg::{MonomialOrder, Polynomial, PolynomialRing};
use noether::numthy::{cyclotomic_polynomial, FiniteField};
use noether::{Integer, IntegerRing, Rational, Q, Z};

fn integer_polynomial(
    ring: &PolynomialRing<IntegerRing>,
    coefficients: &[&str],
) -> Polynomial<IntegerRing> {
    ring.from_u64_terms(
        coefficients
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| **coefficient != "0")
            .map(|(exponent, coefficient)| {
                (
                    vec![exponent as u64],
                    Integer::from_str(coefficient).unwrap(),
                )
            }),
    )
    .unwrap()
}

#[test]
fn k3_weil_polynomial_has_exactly_the_stated_cyclotomic_factor() {
    let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
    let degree_twenty_numerators = [
        31_i64, -12, 15, -6, -3, -6, -5, 5, -13, 15, -22, 15, -13, 5, -5, -6, -3, -6, 15, -12, 31,
    ];
    let degree_twenty = ring
        .from_u64_terms(degree_twenty_numerators.into_iter().enumerate().map(
            |(exponent, numerator)| {
                (
                    vec![exponent as u64],
                    Rational::new(numerator.into(), 31.into()).unwrap(),
                )
            },
        ))
        .unwrap();
    let phi_one = cyclotomic_polynomial(1)
        .unwrap()
        .base_change(&Z.rational_embedding())
        .unwrap();
    let normalized_weil = (&phi_one.pow(2).unwrap() * &degree_twenty).unwrap();

    let factorization = normalized_weil.factor().unwrap();
    factorization.verify().unwrap();
    assert_eq!(factorization.coefficient(), &Rational::one());
    assert_eq!(factorization.factors().len(), 2);
    assert!(factorization.factors().contains(&(phi_one.clone(), 2)));
    assert!(factorization
        .factors()
        .contains(&(degree_twenty.clone(), 1)));

    // These are precisely the indices n with Euler phi(n) = 20. The
    // irreducible degree-twenty factor is therefore not cyclotomic.
    for index in [25, 33, 44, 50, 66] {
        let cyclotomic = cyclotomic_polynomial(index)
            .unwrap()
            .base_change(&Z.rational_embedding())
            .unwrap();
        assert_ne!(degree_twenty, cyclotomic);
    }
}

#[test]
fn restricted_sextics_recover_roots_in_every_prime_field_lane() {
    let cases: [(FiniteField, &[i64]); 3] = [
        // f(1,t,4+t) over F_5 in the paper's split tritangent.
        (FiniteField::prime(5_u8).unwrap(), &[4, 2, 3, 1, 1]),
        // f(1,t,24+23t) over F_31 in the Picard-rank calculation.
        (
            FiniteField::prime(31_u8).unwrap(),
            &[20, 17, 2, 27, 22, 10, 4],
        ),
        // A synthetic restricted sextic exercises the arbitrary-size-prime
        // backend; the paper's two larger line coefficients are not printed.
        (
            FiniteField::prime(Integer::from_str("18446744073709551629").unwrap()).unwrap(),
            &[],
        ),
    ];

    for (field, paper_coefficients) in cases {
        let ring = PolynomialRing::new(field.clone(), ["t"], MonomialOrder::Lexicographic).unwrap();
        let source =
            if paper_coefficients.is_empty() {
                let cubic = ring
                    .from_u64_terms([12_i64, 2, 18, 2].into_iter().enumerate().map(
                        |(exponent, coefficient)| {
                            (vec![exponent as u64], field.element(coefficient))
                        },
                    ))
                    .unwrap();
                cubic.pow(2).unwrap()
            } else {
                ring.from_u64_terms(paper_coefficients.iter().copied().enumerate().map(
                    |(exponent, coefficient)| (vec![exponent as u64], field.element(coefficient)),
                ))
                .unwrap()
            };

        assert!(source.is_nth_power(2).unwrap());
        let root = source.nth_root(2).unwrap().unwrap();
        assert_eq!(root.pow(2).unwrap(), source);
        assert!(root.degree(None).unwrap().unwrap() <= 3);
    }
}

#[test]
fn k3_critical_value_polynomial_has_one_strictly_negative_real_root() {
    let ring = PolynomialRing::new(Z, ["t"], MonomialOrder::Lexicographic).unwrap();
    // This is the eliminant of f_x, f_y and t-f on the z=1 patch in
    // verify.magma. Coefficients are stored in increasing exponent order.
    let critical = integer_polynomial(
        &ring,
        &[
            "3360562185644360160973024557760060102333744225",
            "72583254065809636635306332337703824366188434270",
            "726654657703922408940557551155032090684638933826",
            "4491703719438691166049926993010365150003754711011",
            "19256353004968473541580950264495768027780300745039",
            "60962873077723427409680718610171857266200338167358",
            "148201393034895700713856853648997648220419859045614",
            "284110776451050852671039168867589647389115131321640",
            "437803897038398722059166553513985177579371161559422",
            "549949732424928206881207552937639747979608663294517",
            "568814322958527604958344109338564919774781998504501",
            "487510238853906640383319739836454123296419603393994",
            "347120875264540046734539521147909916728268867388475",
            "205001911678589276856746882042973886033642846174586",
            "99795207219245456392194890309892042934811552651358",
            "39637318624461089569909394131301819370491354575414",
            "12746752079295662883612929235826851436347673895036",
            "3411467117927526957743561248737057757569536641175",
            "897931764752726405097675671836017715602757282832",
            "313702111241376644531747319308739183459079705481",
            "134635992963754717236436477478853663889787847041",
            "51203458867071515178816535793913929841295545523",
            "14781904815184226503773321261056140343696882688",
            "3030216345151105843803927021876499585901617152",
            "402365467566860107202765507308898559483445248",
            "27003416146933923083721127498966008518934528",
        ],
    );

    let roots = critical.real_roots().unwrap();
    assert_eq!(roots.len(), 1);
    let largest = roots.last().unwrap();
    largest.verify().unwrap();
    assert_eq!(largest.sign().unwrap(), -1);
    let refined = largest.refine(32).unwrap();
    assert!(refined.upper() < &Rational::zero());
    assert!(!largest.ball(192).unwrap().contains_zero());
}
