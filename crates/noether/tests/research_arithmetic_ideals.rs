// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Paper-derived acceptance cases for arithmetic polynomial ideals.

use std::str::FromStr;

use noether::commalg::{AffineSpace, MonomialOrder, Polynomial, PolynomialRing};
use noether::numthy::FiniteField;
use noether::{EuclideanDomain, IdentityCoefficientMap, Integer, Ring, Z};

const UNIVERSAL_BLOCK_BASIS: &str = include_str!("data/universal_sextic_square_block_basis.txt");

const PAPER_RESTRICTED_SEXTIC_COEFFICIENTS: [&str; 7] = [
    "-a^6-a^5-a^2-a-1",
    "-6*b*a^5-5*b*a^4-a^5-2*b*a-a^2-b",
    "-15*b^2*a^4-10*b^2*a^3-5*b*a^4-a^4-b^2-2*b*a-a^2-1",
    "-20*b^3*a^3-10*b^3*a^2-10*b^2*a^3-4*b*a^3-a^3-b^2-2*b*a",
    "-15*b^4*a^2-5*b^4*a-10*b^3*a^2-6*b^2*a^2-3*b*a^2-b^2-a",
    "-6*b^5*a-b^5-5*b^4*a-4*b^3*a-3*b^2*a-b-1",
    "-b^6-b^5-b^4-b^3-1",
];

const PAPER_BAD_PRIMES: [&str; 5] = [
    "5",
    "31",
    "7517",
    "84716037398136110308799",
    concat!(
        "4424904772196959344085200612883251617292465803437757948",
        "5992572698404066491363246248977477562371729031497984350",
        "0902180031058767256453958545754450340721124283977338015",
        "3664612642260759001523868554216076825404419681"
    ),
];

fn parse_integer_polynomial(
    ring: &PolynomialRing<noether::IntegerRing>,
    expression: &str,
) -> Polynomial<noether::IntegerRing> {
    let expanded_signs = expression.replace('-', "+-");
    let mut terms = Vec::new();
    for signed_term in expanded_signs.split('+').filter(|term| !term.is_empty()) {
        let (sign, term) = signed_term
            .strip_prefix('-')
            .map_or((Integer::one(), signed_term), |term| {
                (Integer::from(-1), term)
            });
        let mut coefficient = sign;
        let mut exponent = vec![Integer::zero(); ring.variables().len()];
        for factor in term.split('*') {
            if factor.bytes().all(|byte| byte.is_ascii_digit()) {
                coefficient *= Integer::from_str(factor).unwrap();
                continue;
            }
            let (variable, power) = factor
                .split_once('^')
                .map_or((factor, 1_u64), |(variable, power)| {
                    (variable, power.parse::<u64>().unwrap())
                });
            let index = ring
                .variables()
                .iter()
                .position(|candidate| candidate == variable)
                .unwrap_or_else(|| panic!("unknown test-fixture variable {variable}"));
            exponent[index] += Integer::from(power);
        }
        terms.push((exponent, coefficient));
    }
    ring.from_terms(terms).unwrap()
}

fn universal_block_basis() -> (
    PolynomialRing<noether::IntegerRing>,
    Vec<Polynomial<noether::IntegerRing>>,
) {
    let ring = PolynomialRing::new(
        Z,
        ["d0", "d1", "d2", "d3", "d4", "d5", "d6"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let basis = UNIVERSAL_BLOCK_BASIS
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| parse_integer_polynomial(&ring, line))
        .collect::<Vec<_>>();
    assert_eq!(basis.len(), 53);
    (ring, basis)
}

fn paper_tritangent_generators() -> (
    PolynomialRing<noether::IntegerRing>,
    Vec<Polynomial<noether::IntegerRing>>,
    Vec<Polynomial<noether::IntegerRing>>,
) {
    let (universal_ring, universal_basis) = universal_block_basis();
    let ring =
        PolynomialRing::new(Z, ["b", "a"], MonomialOrder::GradedReverseLexicographic).unwrap();
    let coefficients = PAPER_RESTRICTED_SEXTIC_COEFFICIENTS
        .iter()
        .map(|expression| parse_integer_polynomial(&ring, expression))
        .collect::<Vec<_>>();
    let specialization = universal_ring
        .homomorphism(&ring, IdentityCoefficientMap::new(Z), coefficients.clone())
        .unwrap();
    let generators = universal_basis
        .iter()
        .map(|generator| specialization.apply(generator).unwrap())
        .collect();
    (ring, generators, coefficients)
}

fn paper_bad_primes() -> Vec<Integer> {
    PAPER_BAD_PRIMES
        .iter()
        .map(|prime| Integer::from_str(prime).unwrap())
        .collect()
}

fn paper_bad_prime_product() -> Integer {
    paper_bad_primes()
        .into_iter()
        .fold(Integer::one(), |product, prime| product * prime)
}

fn universal_sextic_square_equations() -> (
    PolynomialRing<noether::IntegerRing>,
    Vec<Polynomial<noether::IntegerRing>>,
) {
    let ring = PolynomialRing::new(
        Z,
        [
            "c0", "c1", "c2", "c3", "d0", "d1", "d2", "d3", "d4", "d5", "d6",
        ],
        MonomialOrder::Lexicographic,
    )
    .unwrap();
    let variables = ring.generators().unwrap();
    let c = &variables[..4];
    let d = &variables[4..];
    let equations = (0..=6)
        .map(|degree| {
            let square_coefficient = (0..=degree)
                .filter(|left| *left < 4 && degree - *left < 4)
                .map(|left| ring.mul(&c[left], &c[degree - left]).unwrap())
                .fold(ring.zero(), |sum, term| ring.add(&sum, &term).unwrap());
            ring.sub(&d[degree], &square_coefficient).unwrap()
        })
        .collect();
    (ring, equations)
}

#[test]
fn stored_universal_square_locus_basis_vanishes_on_a_generic_cubic_square() {
    let (universal_ring, basis) = universal_block_basis();
    let cubic_ring = PolynomialRing::new(
        Z,
        ["c0", "c1", "c2", "c3"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let cubic = cubic_ring.generators().unwrap();
    let coefficients = (0..=6)
        .map(|degree| {
            (0..=degree)
                .filter(|left| *left < 4 && degree - *left < 4)
                .map(|left| cubic_ring.mul(&cubic[left], &cubic[degree - left]).unwrap())
                .fold(cubic_ring.zero(), |sum, term| {
                    cubic_ring.add(&sum, &term).unwrap()
                })
        })
        .collect::<Vec<_>>();
    let square_parameterization = universal_ring
        .homomorphism(&cubic_ring, IdentityCoefficientMap::new(Z), coefficients)
        .unwrap();

    assert_eq!(basis.len(), 53);
    for relation in basis {
        assert!(square_parameterization.apply(&relation).unwrap().is_zero());
    }
}

#[test]
#[ignore = "paper-scale integral elimination stress case"]
fn paper_scale_universal_sextic_square_elimination_is_verified() {
    let (ring, equations) = universal_sextic_square_equations();
    let ideal = ring.ideal(equations).unwrap();
    let elimination = ideal.eliminate(&[0, 1, 2, 3]).unwrap();

    assert!(!elimination.generators().is_empty());
    assert!(elimination.generators().iter().all(|generator| {
        generator
            .terms()
            .all(|(exponent, _)| exponent[..4].iter().all(Integer::is_zero))
    }));
    elimination.verify().unwrap();
}

#[test]
#[ignore = "paper-scale rational certificate stress case"]
fn paper_scale_independent_paper_tritangent_witnesses_certify_the_bad_prime_support() {
    let (ring, generators, _) = paper_tritangent_generators();

    // These are the overlapping slices used by the ancillary Magma program:
    // basis entries 40..51 and 30..40 (one-based, inclusive). Neither family
    // contains the other. The witnesses are computed independently and
    // checked against the exact supplied families, not against a precomputed
    // integral basis.
    let first_family = generators[39..51].to_vec();
    let second_family = generators[29..40].to_vec();
    let first = ring.nullstellensatz_certificate(first_family).unwrap();
    let second = ring.nullstellensatz_certificate(second_family).unwrap();
    first.verify().unwrap();
    second.verify().unwrap();

    let common_support = first.principal_ideal().sum(second.principal_ideal());
    let common_generator = common_support.principal_generator();
    assert!(!common_generator.is_zero());
    for prime in paper_bad_primes() {
        let (_, remainder) = Z.div_rem(&common_generator, &prime).unwrap();
        assert!(
            remainder.is_zero(),
            "the independently certified support lost paper prime {prime}"
        );
    }
}

#[test]
#[ignore = "paper-scale integral strong-basis measurement; may reach its resource bound"]
fn paper_scale_final_paper_tritangent_ideal_has_linear_integral_basis_and_exact_support() {
    let (ring, generators, _) = paper_tritangent_generators();
    let ideal = ring.ideal(generators).unwrap();
    assert_eq!(ideal.generators().len(), 3);

    let mut constant = None;
    let mut pivots = [false; 2];
    for generator in ideal.generators() {
        let nonconstant = generator
            .terms()
            .filter(|(exponent, _)| exponent.iter().any(|value| !value.is_zero()))
            .collect::<Vec<_>>();
        if nonconstant.is_empty() {
            assert!(constant.is_none());
            constant = Some(
                generator
                    .coefficient(&[Integer::zero(), Integer::zero()])
                    .unwrap(),
            );
            continue;
        }
        assert_eq!(nonconstant.len(), 1);
        let (exponent, coefficient) = nonconstant[0];
        let pivot = if exponent == [Integer::one(), Integer::zero()] {
            0
        } else if exponent == [Integer::zero(), Integer::one()] {
            1
        } else {
            panic!("paper basis generator is not linear: {generator}");
        };
        assert!(coefficient == &Integer::one() || coefficient == &Integer::from(-1));
        assert!(!pivots[pivot]);
        pivots[pivot] = true;
    }
    assert_eq!(pivots, [true, true]);
    assert_eq!(constant.unwrap(), paper_bad_prime_product());
    ideal.verify().unwrap();

    let support = ideal.arithmetic_fiber_support(10_000, 1024).unwrap();
    assert!(support.is_exact());
    assert_eq!(
        support.support_ideal().principal_generator(),
        paper_bad_prime_product()
    );
    support.verify().unwrap();
}

#[test]
#[ignore = "paper-scale prime-field solving, including a 211-digit characteristic"]
fn paper_scale_paper_tritangent_reductions_have_one_split_rational_closed_point() {
    let (integral_ring, generators, coefficients) = paper_tritangent_generators();
    for prime in paper_bad_primes() {
        let field = FiniteField::prime(prime.clone()).unwrap();
        let coefficient_map = Z.finite_field_map(&field);
        let ring = integral_ring.base_change(&coefficient_map).unwrap();
        let reduced_generators = generators
            .iter()
            .map(|generator| generator.base_change(&coefficient_map).unwrap())
            .collect::<Vec<_>>();
        let space = AffineSpace::with_names(field.clone(), 2, ["b", "a"]).unwrap();
        assert_eq!(space.coordinate_ring(), &ring);
        let scheme = space
            .closed_subscheme(ring.ideal(reduced_generators).unwrap())
            .unwrap();

        let solution = scheme.closed_points().unwrap();
        assert_eq!(solution.points().len(), 1, "paper prime {prime}");
        let point = &solution.points()[0];
        assert_eq!(point.degree(), 1, "paper prime {prime}");
        point.verify().unwrap();
        solution.verify().unwrap();

        let restricted_coefficients = coefficients
            .iter()
            .map(|coefficient| {
                coefficient
                    .base_change(&coefficient_map)
                    .unwrap()
                    .evaluate(point.coordinates())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let univariate =
            PolynomialRing::new(field.clone(), ["t"], MonomialOrder::Lexicographic).unwrap();
        let restricted = univariate
            .from_u64_terms(
                restricted_coefficients
                    .into_iter()
                    .enumerate()
                    .map(|(degree, coefficient)| (vec![degree as u64], coefficient)),
            )
            .unwrap();
        let square_root = restricted
            .nth_root(2)
            .unwrap()
            .unwrap_or_else(|| panic!("paper tritangent does not split at {prime}"));
        assert_eq!(square_root.pow(2).unwrap(), restricted);
        assert!(square_root.degree(None).unwrap().unwrap() <= 3);
    }
}
