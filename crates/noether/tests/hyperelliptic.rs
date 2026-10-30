// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    FrobeniusPolynomial, HyperellipticCurve, HyperellipticFrobeniusEvidence, MonomialOrder,
    PolynomialRing, ZetaFunction,
};
use noether::numthy::FiniteField;
use noether::{Integer, Ring};
use proptest::prelude::*;

fn model(p: u64, coefficients: &[i64]) -> HyperellipticCurve {
    let field = FiniteField::prime(p).unwrap();
    let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
    HyperellipticCurve::new(
        ring.from_terms(
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| (vec![Integer::from(i)], field.element(*c))),
        )
        .unwrap(),
    )
    .unwrap()
}
fn compare_counts(curve: &HyperellipticCurve, degrees: u64) {
    let frobenius = curve.frobenius_polynomial().unwrap();
    let counts = curve.cover().point_counts(1..=degrees).unwrap();
    for record in counts.counts() {
        let expected = &(&curve.base().order().pow(record.degree()) + &Integer::one())
            - &frobenius.trace(record.degree()).unwrap();
        assert_eq!(&expected, record.count());
    }
    assert_eq!(frobenius.degree(), 2 * curve.genus());
    assert_eq!(frobenius.cohomological_degree(), 1);
    assert_eq!(frobenius.cohomological_weight(), 1);
}

#[test]
fn smallest_curves_and_even_degree_twists() {
    for curve in [
        model(3, &[0, 1, 0, 1]),
        model(3, &[0, 1, 0, 0, 0, 1]),
        model(3, &[0, 1, 0, 0, 0, 0, 1]),
        model(3, &[0, 2, 0, 0, 0, 0, 2]),
        model(3, &[1, 1, 0, 0, 0, 0, 1]),
        model(3, &[2, 2, 0, 0, 0, 0, 2]),
        model(7, &[2, 1, 0, 0, 0, 1]),
    ] {
        compare_counts(&curve, 2);
        assert!(curve.smoothness_evidence().unwrap().is_smooth());
        assert_eq!(
            curve.cover().weighted_realization().ambient().weights(),
            &[(curve.genus() + 1) as u64, 1, 1]
        );
    }
    assert_eq!(
        model(3, &[0, 1, 0, 0, 0, 0, 1]).infinity_residue_degrees(),
        vec![1, 1]
    );
    assert_eq!(
        model(3, &[0, 2, 0, 0, 0, 0, 2]).infinity_residue_degrees(),
        vec![2]
    );
    let twist = model(3, &[2, 2, 0, 0, 0, 0, 2])
        .frobenius_evidence(None)
        .unwrap();
    assert_eq!(twist.quadratic_twist_sign(), -1);
    assert_eq!(
        twist.normalized_branch().unwrap(),
        model(3, &[1, 1, 0, 0, 0, 0, 1]).branch_equation().clone()
    );
    assert!(!twist
        .frobenius_polynomial()
        .unwrap()
        .trace(1)
        .unwrap()
        .is_zero());
}

#[test]
fn genus_four_and_five_in_characteristic_three() {
    for degree in [9, 12] {
        let mut coefficients = vec![0; degree + 1];
        coefficients[1] = 1;
        coefficients[degree] = 1;
        let curve = model(3, &coefficients);
        compare_counts(&curve, curve.genus() as u64);
    }
}

#[test]
fn genuine_extension_coefficients_and_square_completion() {
    let field = FiniteField::extension(3, 2).unwrap();
    let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
    let x = ring.generator(0).unwrap();
    let f = (&(&x.pow(5).unwrap() + &x.scalar_mul(&field.generator()).unwrap()).unwrap()
        + &ring.one())
        .unwrap();
    let h = (&x + &ring.one()).unwrap();
    let curve = HyperellipticCurve::from_equation(f.clone(), h.clone()).unwrap();
    let completed = HyperellipticCurve::new(curve.branch_equation().clone()).unwrap();
    assert_ne!(curve, completed);
    assert_eq!(
        curve
            .completion_shift()
            .unwrap()
            .scalar_mul(&field.element(2))
            .unwrap(),
        h
    );
    compare_counts(&curve, 2);
    let direct = curve.cover().point_counts([1, 2]).unwrap();
    assert_eq!(
        FrobeniusPolynomial::from_hyperelliptic_counts(&direct)
            .unwrap()
            .polynomial(),
        curve.frobenius_polynomial().unwrap().polynomial(),
    );
    assert_eq!(
        curve.frobenius_polynomial().unwrap().polynomial(),
        completed.frobenius_polynomial().unwrap().polynomial()
    );
}

#[test]
fn evidence_replay_and_independent_deserialization_reject_forgery() {
    let curve = model(3, &[0, 1, 0, 0, 0, 1]);
    curve.verify().unwrap();
    assert_eq!(
        HyperellipticCurve::from_json(&curve.to_json().unwrap()).unwrap(),
        curve
    );
    let evidence = curve.frobenius_evidence(None).unwrap();
    evidence.verify().unwrap();
    let high = curve
        .frobenius_evidence(Some(evidence.precision() + 1))
        .unwrap();
    assert!(high.working_precision() > evidence.working_precision());
    let encoded = evidence.to_json().unwrap();
    assert_eq!(
        serde_json::to_string(&curve).unwrap(),
        curve.to_json().unwrap()
    );
    assert_eq!(serde_json::to_string(&evidence).unwrap(), encoded);
    assert_eq!(
        HyperellipticFrobeniusEvidence::from_json(&encoded).unwrap(),
        evidence
    );
    let mut forged: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    forged["binomial_terms"] = serde_json::json!(0);
    assert!(HyperellipticFrobeniusEvidence::from_json(&forged.to_string()).is_err());
    assert!(serde_json::from_value::<HyperellipticFrobeniusEvidence>(forged).is_err());
    let mut forged: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    let rank = evidence.frobenius().source_dimension();
    forged["linear_frobenius"] = serde_json::to_value(noether::linalg::Matrix::zeros(
        evidence.frobenius().context().clone(),
        rank,
        rank,
    ))
    .unwrap();
    assert!(HyperellipticFrobeniusEvidence::from_json(&forged.to_string()).is_err());
    let frobenius = curve.frobenius_polynomial().unwrap();
    frobenius.verify().unwrap();
    assert_eq!(
        FrobeniusPolynomial::from_json(&frobenius.to_json().unwrap()).unwrap(),
        frobenius
    );
    let zeta = frobenius.zeta_function().unwrap();
    zeta.verify().unwrap();
    assert_eq!(
        ZetaFunction::from_json(&zeta.to_json().unwrap()).unwrap(),
        zeta
    );
    assert_eq!(
        zeta.numerator().degree(None).unwrap(),
        Some(Integer::from(4))
    );
    assert_eq!(
        zeta.denominator().degree(None).unwrap(),
        Some(Integer::from(2))
    );
}

#[test]
fn complete_frobenius_from_counts_matches_kedlaya_and_replays_its_own_source() {
    for curve in [
        model(3, &[0, 1, 0, 0, 0, 1]),
        model(3, &[2, 2, 0, 0, 0, 0, 2]),
        model(3, &[0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
    ] {
        let counts = curve
            .cover()
            .point_counts(1..=curve.genus() as u64 + 1)
            .unwrap();
        let reconstructed = FrobeniusPolynomial::from_hyperelliptic_counts(&counts).unwrap();
        let padic = curve.frobenius_polynomial().unwrap();
        assert_eq!(reconstructed.polynomial(), padic.polynomial());
        assert_eq!(reconstructed.degree(), 2 * curve.genus());
        assert_eq!(reconstructed.cohomological_degree(), 1);
        assert_eq!(reconstructed.cohomological_weight(), 1);
        assert_eq!(reconstructed.point_counts(), Some(&counts));
        assert!(reconstructed.hyperelliptic_evidence().is_none());
        reconstructed.verify().unwrap();
        let encoded = reconstructed.to_json().unwrap();
        assert_eq!(
            FrobeniusPolynomial::from_json(&encoded).unwrap(),
            reconstructed
        );
        assert_eq!(
            serde_json::from_str::<FrobeniusPolynomial>(&encoded).unwrap(),
            reconstructed
        );
        let mut forged: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        forged["cohomological_weight"] = serde_json::json!(2);
        assert!(FrobeniusPolynomial::from_json(&forged.to_string()).is_err());
        assert!(serde_json::from_value::<FrobeniusPolynomial>(forged).is_err());
    }
}

#[test]
fn curve_count_reconstruction_rejects_insufficient_and_singular_targets() {
    use noether::commalg::{AffineSpace, CyclicHypersurfaceCover, FrobeniusError, ProjectiveSpace};
    let affine = AffineSpace::new(FiniteField::prime(3).unwrap(), 1).unwrap();
    let ordinary = affine
        .closed_subscheme(affine.coordinate_ring().ideal([]).unwrap())
        .unwrap();
    let ordinary_counts = ordinary.point_counts([1]).unwrap();
    assert!(matches!(
        FrobeniusPolynomial::from_hyperelliptic_counts(&ordinary_counts),
        Err(FrobeniusError::UnsupportedCohomology(_))
    ));
    let curve = model(3, &[0, 1, 0, 0, 0, 1]);
    let counts = curve.cover().point_counts([1]).unwrap();
    assert!(matches!(
        FrobeniusPolynomial::from_hyperelliptic_counts(&counts),
        Err(FrobeniusError::InsufficientPointCounts {
            required_through: 2
        })
    ));
    let counts = curve.cover().point_counts([1, 3]).unwrap();
    assert!(matches!(
        FrobeniusPolynomial::from_hyperelliptic_counts(&counts),
        Err(FrobeniusError::InsufficientPointCounts {
            required_through: 2
        })
    ));
    for p in [2, 3] {
        let field = FiniteField::prime(p).unwrap();
        let base = ProjectiveSpace::new(field, 1).unwrap();
        let ring = base.coordinate_ring();
        let [x, z]: [_; 2] = ring.generators().unwrap().try_into().unwrap();
        let branch = (&x.pow(2).unwrap() - &z.pow(2).unwrap())
            .unwrap()
            .pow(2)
            .unwrap();
        let cover = CyclicHypersurfaceCover::<FiniteField>::new(base, 2, branch).unwrap();
        let counts = cover.point_counts([1]).unwrap();
        assert!(matches!(
            FrobeniusPolynomial::from_hyperelliptic_counts(&counts),
            Err(FrobeniusError::UnsupportedCohomology(_))
        ));
    }
}

#[test]
fn malformed_models_and_precision_fail_closed() {
    let field = FiniteField::prime(3).unwrap();
    let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
    let x = ring.generator(0).unwrap();
    for f in [
        ring.zero(),
        ring.one(),
        x.clone(),
        x.pow(2).unwrap(),
        x.pow(3).unwrap(),
    ] {
        assert!(HyperellipticCurve::new(f).is_err());
    }
    let other = PolynomialRing::new(field, ["z"], MonomialOrder::default()).unwrap();
    assert!(HyperellipticCurve::from_equation(x.clone(), other.one()).is_err());
    assert!(model(3, &[0, 1, 0, 1]).frobenius_evidence(Some(1)).is_err());
    let two = FiniteField::prime(2).unwrap();
    let ring = PolynomialRing::new(two, ["x"], MonomialOrder::default()).unwrap();
    assert!(HyperellipticCurve::new(ring.generator(0).unwrap().pow(3).unwrap()).is_err());
}

#[test]
fn semilinear_frobenius_over_odd_and_even_residue_degrees() {
    for field in [
        FiniteField::extension(3, 2).unwrap(),
        FiniteField::extension(3, 3).unwrap(),
        FiniteField::extension(3, 4).unwrap(),
        FiniteField::with_modulus(3, vec![1, 0, 1]).unwrap(),
    ] {
        let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        let f = (&(&x.pow(3).unwrap() + &x.scalar_mul(&field.generator()).unwrap()).unwrap()
            + &ring.one())
            .unwrap();
        let curve = HyperellipticCurve::new(f).unwrap();
        compare_counts(&curve, 1);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]
    #[test]
    fn dense_genus_four_models_match_exhaustive_counts(coefficients in proptest::collection::vec(0_i64..3,9)) {
        let field=FiniteField::prime(3).unwrap();
        let ring=PolynomialRing::new(field.clone(),["x"],MonomialOrder::default()).unwrap();
        let f=ring.from_terms(coefficients.iter().copied().chain([1]).enumerate().map(|(i,c)|(vec![Integer::from(i)],field.element(c)))).unwrap();
        prop_assume!(f.gcd(&f.derivative(0).unwrap()).unwrap()==ring.one());
        let curve=HyperellipticCurve::new(f).unwrap();
        compare_counts(&curve,4);
    }
}

#[test]
#[ignore = "requires the validation-only PARI/GP oracle"]
fn pari_arbitrary_genus_and_even_degree_parity() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut fixtures = Vec::new();
    for degree in [5, 6, 9, 12, 17, 18, 41, 42] {
        let mut coefficients = vec![0; degree + 1];
        coefficients[1] = 1;
        coefficients[degree] = 1;
        fixtures.push((3, coefficients));
    }
    fixtures.push((3, vec![1, 2, 2, 0, 1, 2, 0, 1, 1, 1]));
    fixtures.push((3, vec![2, 1, 2, 1, 0, 2, 2, 1, 0, 1, 1]));
    // Dense genus-five benchmark regressions for polynomial division.
    fixtures.push((13, vec![6, 0, 3, 2, 10, 1, 1, 10, 2, 3, 0, 1]));
    fixtures.push((13, vec![6, 0, 3, 2, 10, 1, 1, 10, 2, 3, 0, 6, 1]));
    for (p, degree) in [(5, 9), (5, 10), (7, 17), (7, 18)] {
        let mut coefficients = vec![0; degree + 1];
        coefficients[1] = 1;
        coefficients[degree] = 1;
        fixtures.push((p, coefficients));
    }
    for (p, coefficients) in fixtures {
        let degree = coefficients.len() - 1;
        let curve = model(p, &coefficients);
        let frobenius = curve.frobenius_polynomial().unwrap();
        let mut child = Command::new("gp")
            .args(["-fq"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("PARI/GP is required");
        writeln!(
            child.stdin.take().unwrap(),
            "v=Vec(hyperellcharpoly(Mod(1,{p})*Polrev([{}])));for(i=1,#v,print(v[i]));quit",
            coefficients
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
        .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success() && output.stderr.is_empty(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let oracle = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(str::parse::<Integer>)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let actual = (0..=2 * curve.genus())
            .map(|i| {
                frobenius
                    .polynomial()
                    .coefficient(&[Integer::from(2 * curve.genus() - i)])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, oracle, "characteristic {p}, degree {degree}");
    }
}

#[test]
#[ignore = "requires the validation-only PARI/GP oracle"]
fn pari_extension_field_and_twist_parity() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    for field in [
        FiniteField::extension(3, 2).unwrap(),
        FiniteField::extension(3, 3).unwrap(),
        FiniteField::extension(3, 4).unwrap(),
        FiniteField::with_modulus(3, vec![1, 0, 1]).unwrap(),
    ] {
        let degree = field.degree();
        let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        // Genus six over genuine F9 coefficients exercises semilinear
        // reduction independently of the prime-field performance fixtures.
        for exponent in if degree == 2 {
            vec![9, 13, 14]
        } else {
            vec![9]
        } {
            let f = (&(&x.pow(exponent).unwrap() + &x.scalar_mul(&field.generator()).unwrap())
                .unwrap()
                + &ring.one())
                .unwrap();
            let modulus = field
                .defining_modulus()
                .unwrap()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            for twist in [false, true] {
                let branch = if twist {
                    f.scalar_mul(&field.generator()).unwrap()
                } else {
                    f.clone()
                };
                let curve = HyperellipticCurve::new(branch).unwrap();
                let frobenius = curve.frobenius_polynomial().unwrap();
                let mut child = Command::new("gp")
                    .args(["-fq"])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .expect("PARI/GP is required");
                let factor = if twist { "a" } else { "1" };
                writeln!(child.stdin.take().unwrap(),"a=ffgen(Mod(1,3)*Polrev([{modulus}],z),'a);v=Vec(hyperellcharpoly({factor}*(x^{exponent}+a*x+1)));for(i=1,#v,print(v[i]));quit").unwrap();
                let output = child.wait_with_output().unwrap();
                assert!(
                    output.status.success() && output.stderr.is_empty(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let oracle = String::from_utf8(output.stdout)
                    .unwrap()
                    .lines()
                    .map(str::parse::<Integer>)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                let actual = (0..=2 * curve.genus())
                    .map(|i| {
                        frobenius
                            .polynomial()
                            .coefficient(&[Integer::from(2 * curve.genus() - i)])
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    actual, oracle,
                    "residue degree {degree}, polynomial degree {exponent}, twisted {twist}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires the validation-only PARI/GP oracle"]
fn pari_streamed_large_field_cyclic_counts() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    for field in [
        FiniteField::prime(4099).unwrap(),
        FiniteField::extension(3, 8).unwrap(),
    ] {
        let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        let setup = match field.defining_modulus() {
            None => "a=Mod(1,4099);".to_owned(),
            Some(modulus) => format!(
                "a=ffgen(Mod(1,3)*Polrev([{}],z),'a);",
                modulus
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        };
        for degree in [3, 4] {
            let f = (&(&x.pow(degree).unwrap() + &x.scalar_mul(&field.generator()).unwrap())
                .unwrap()
                + &ring.one())
                .unwrap();
            let curve = HyperellipticCurve::new(f).unwrap();
            let counts = curve.cover().point_counts([1]).unwrap();
            let mut child = Command::new("gp")
                .args(["-fq", "-s", "128M"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("PARI/GP is required");
            writeln!(
                child.stdin.take().unwrap(),
                "{setup}P=hyperellcharpoly(x^{degree}+a*x+1);print({}+1+polcoef(P,1));quit",
                field.order()
            )
            .unwrap();
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success() && output.stderr.is_empty(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let expected = String::from_utf8(output.stdout)
                .unwrap()
                .trim()
                .parse::<Integer>()
                .unwrap();
            assert_eq!(
                counts.count(1),
                Some(&expected),
                "degree {degree} over {field}"
            );
        }
    }
}
