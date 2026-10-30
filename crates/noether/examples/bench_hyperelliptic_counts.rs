// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Timed public Rust point-count operations; paired orchestration is in
//! `validation/bench/hyperelliptic_counts.py`. Run through `scripts/bench-pin.sh`.

use noether::commalg::{
    FrobeniusPolynomial, GroebnerError, HyperellipticCurve, HyperellipticError, MonomialOrder,
    PointCountError, PointCountMethod, PointCountSequence, PolynomialRing, WeightedProjectiveError,
};
use noether::numthy::{FiniteField, FiniteFieldElement};
use noether::{Field, FiniteContext, Integer, Ring};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::error::Error;
use std::hint::black_box;
use std::io::Write;
use std::time::Instant;

#[allow(
    clippy::large_enum_variant,
    reason = "keep public results alive without adding timed heap allocations for benchmark transport"
)]
enum Output {
    Direct(PointCountSequence),
    Frobenius(FrobeniusPolynomial),
}

fn emit(value: Value) -> Result<(), Box<dyn Error>> {
    println!("{value}");
    std::io::stdout().flush()?;
    Ok(())
}

fn fixture(
    p: u64,
    n: usize,
    genus: usize,
    even: bool,
) -> Result<HyperellipticCurve, Box<dyn Error>> {
    let field = FiniteField::extension(p, n)?;
    let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default())?;
    let degree = 2 * genus + 1 + usize::from(even);
    let generator = field.generator();
    let mut terms = Vec::with_capacity(degree + 1);
    for i in 0..degree {
        let a = field.element((37 * (i + 1) * (i + 1) + 11 * (genus + 1)) as u64);
        let b = field.mul(&generator, &field.element((13 * i + 7) as u64))?;
        terms.push((vec![Integer::from(i)], field.add(&a, &b)?));
    }
    terms.push((vec![Integer::from(degree)], field.one()));
    let f = ring.from_terms(terms)?;
    // A small prime field can collapse the coefficient pattern to a p-th
    // power. Adding x then gives derivative 1 without changing the degree.
    let f = if f.derivative(0)?.is_zero() {
        (&f + &ring.generator(0)?)?
    } else {
        f
    };
    for shift in field.elements() {
        let candidate = (&f + &ring.constant(shift)?)?;
        match HyperellipticCurve::new(candidate) {
            Ok(curve) => return Ok(curve),
            Err(HyperellipticError::InvalidModel(_)) => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err("no smooth fixture exists in the chosen constant-shift family".into())
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Family {
    #[default]
    Legacy,
    Dense,
    Sparse,
    Split,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Leading {
    #[default]
    Monic,
    Square,
    Nonsquare,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
#[serde(default, deny_unknown_fields)]
struct ModelSpec {
    family: Family,
    seed: u64,
    leading: Leading,
    general: bool,
    modulus: Option<Vec<u64>>,
}

// Counter-mixed, explicitly seeded benchmark coefficients. Every power-basis
// coordinate is varied; these fixtures do not claim uniform random sampling.
struct Coefficients(u64);

impl Coefficients {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn element(&mut self, field: &FiniteField) -> Result<FiniteFieldElement, Box<dyn Error>> {
        let p = field
            .characteristic()
            .to_u64_checked()
            .ok_or("word characteristic required")?;
        Ok(field.element_from_coefficients(
            &(0..field.degree())
                .map(|_| self.next() % p)
                .collect::<Vec<_>>(),
        )?)
    }
}

fn expanded_fixture(
    p: u64,
    n: usize,
    genus: usize,
    even: bool,
    spec: &ModelSpec,
) -> Result<HyperellipticCurve, Box<dyn Error>> {
    if spec == &ModelSpec::default() {
        return fixture(p, n, genus, even);
    }
    if spec.family == Family::Legacy && (spec.seed != 0 || spec.modulus.is_some()) {
        return Err("legacy fixtures require seed zero and the Conway presentation".into());
    }
    let field = match &spec.modulus {
        Some(modulus) => FiniteField::with_modulus(p, modulus.clone())?,
        None => FiniteField::extension(p, n)?,
    };
    if field.degree() != n {
        return Err("explicit modulus degree disagrees with residue degree".into());
    }
    let order = field
        .order()
        .to_u64_checked()
        .ok_or("word field order required for fixtures")?;
    let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default())?;
    let x = ring.generator(0)?;
    let degree = 2 * genus + 1 + usize::from(even);
    let mut coefficients = Coefficients(spec.seed);
    let mut branch = None;
    for _ in 0..256 {
        let candidate = match spec.family {
            Family::Legacy => fixture(p, n, genus, even)?.branch_equation().clone(),
            Family::Dense => {
                let mut terms = (0..degree)
                    .map(|i| {
                        coefficients
                            .element(&field)
                            .map(|value| (vec![Integer::from(i)], value))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                terms.push((vec![Integer::from(degree)], field.one()));
                ring.from_terms(terms)?
            }
            Family::Sparse => ring.from_terms([
                (vec![Integer::from(degree)], field.one()),
                (vec![Integer::one()], coefficients.element(&field)?),
                (vec![Integer::zero()], coefficients.element(&field)?),
            ])?,
            Family::Split => {
                if order < degree as u64 {
                    return Err("split fixture requires at least degree many field elements".into());
                }
                let mut roots = Vec::with_capacity(degree);
                for _ in 0..64 * degree {
                    let root = coefficients.element(&field)?;
                    if !roots.contains(&root) {
                        roots.push(root);
                    }
                    if roots.len() == degree {
                        break;
                    }
                }
                if roots.len() != degree {
                    return Err("seeded distinct-root fixture exceeded its draw budget".into());
                }
                let mut polynomial = ring.one();
                for root in roots {
                    polynomial = (&polynomial * &(&x - &ring.constant(root)?)?)?;
                }
                polynomial
            }
        };
        if candidate.gcd(&candidate.derivative(0)?)? == ring.one() {
            branch = Some(candidate);
            break;
        }
    }
    let branch = branch.ok_or("seeded fixture exceeded its smoothness search budget")?;
    let leading = match spec.leading {
        Leading::Monic => field.one(),
        Leading::Square => field
            .elements()
            .map(|value| value.pow(2))
            .find(|value| !value.is_zero() && !value.is_one())
            .ok_or("the field has no nontrivial nonzero square")?,
        Leading::Nonsquare => field
            .elements()
            .find(|value| !value.is_zero() && value.pow((order - 1) / 2) != field.one())
            .ok_or("the odd field had no nonsquare")?,
    };
    let branch = branch.scalar_mul(&leading)?;
    if !spec.general {
        return Ok(HyperellipticCurve::new(branch)?);
    }
    let mut coefficients = Coefficients(spec.seed ^ 0x4f1bbcdc6762c72b);
    let mut terms = (0..=genus)
        .map(|i| {
            coefficients
                .element(&field)
                .map(|value| (vec![Integer::from(i)], value))
        })
        .collect::<Result<Vec<_>, _>>()?;
    terms.push((vec![Integer::from(genus + 1)], field.one()));
    let h = ring.from_terms(terms)?;
    let inverse_four = field.inverse(&field.element(4))?;
    let f = (&branch - &h.pow(2)?.scalar_mul(&inverse_four)?)?;
    let curve = HyperellipticCurve::from_equation(f, h)?;
    if curve.branch_equation() != &branch {
        return Err("the general equation changed its completed-square branch".into());
    }
    Ok(curve)
}

fn geometry_status(curve: &HyperellipticCurve) -> Result<Value, Box<dyn Error>> {
    // The curve constructor and verify() prove squarefreeness. The stronger
    // weighted cone-Jacobian computation has its own bounded Groebner budget.
    curve.verify()?;
    match curve.smoothness_evidence() {
        Ok(proof) if proof.is_smooth() && proof.branch_is_smooth() == Some(true) => {
            Ok(json!({"curve":"verified", "weighted_jacobian":"verified"}))
        }
        Ok(_) => Err("the weighted geometry contradicted the smooth curve model".into()),
        Err(
            error @ HyperellipticError::Geometry(WeightedProjectiveError::Groebner(
                GroebnerError::ResourceLimit { .. },
            )),
        ) => Ok(
            json!({"curve":"verified", "weighted_jacobian":"unavailable", "error":error.to_string()}),
        ),
        Err(error) => Err(error.into()),
    }
}

fn operation(
    curve: &HyperellipticCurve,
    method: &str,
    goal: &str,
) -> Result<(Integer, Output), Box<dyn Error>> {
    if goal == "frobenius" {
        let frobenius = match method {
            "direct" => {
                let counts = curve.cover().point_counts(1..=curve.genus() as u64)?;
                if counts.counts().iter().any(|record| {
                    record.evidence().method() != PointCountMethod::ExhaustiveCyclicFibers
                }) {
                    return Err(
                        "the cover selected a trace formula rather than direct enumeration".into(),
                    );
                }
                FrobeniusPolynomial::from_hyperelliptic_counts(&counts)?
            }
            "kedlaya" => curve.frobenius_polynomial()?,
            _ => return Err("method must be direct or kedlaya".into()),
        };
        return Ok((frobenius.point_count(1)?, Output::Frobenius(frobenius)));
    }
    match method {
        "direct" => {
            let counts = curve.cover().point_counts([1])?;
            if counts.counts()[0].evidence().method() != PointCountMethod::ExhaustiveCyclicFibers {
                return Err(
                    "the cover selected a trace formula rather than direct enumeration".into(),
                );
            }
            Ok((
                counts.count(1).ok_or("missing count")?.clone(),
                Output::Direct(counts),
            ))
        }
        "kedlaya" => {
            let frobenius = curve.frobenius_polynomial()?;
            Ok((frobenius.point_count(1)?, Output::Frobenius(frobenius)))
        }
        _ => Err("method must be direct or kedlaya".into()),
    }
}

fn metadata(output: &Output) -> Value {
    match output {
        Output::Direct(counts) => {
            let evidence = counts.counts()[0].evidence();
            json!({
                "method": format!("{:?}", evidence.method()),
                "independent_root_scan": evidence.independently_verified(),
            })
        }
        Output::Frobenius(frobenius) => {
            let mut result = json!({
                "frobenius_descending": (0..=frobenius.degree()).map(|i| {
                    frobenius.polynomial().coefficient(&[Integer::from(frobenius.degree() - i)])
                        .expect("univariate characteristic polynomial").to_string()
                }).collect::<Vec<_>>(),
            });
            if let Some(evidence) = frobenius.hyperelliptic_evidence() {
                result["padic"] = json!({
                    "precision":evidence.precision(), "working_precision":evidence.working_precision(),
                    "denominator_loss":evidence.denominator_loss(), "binomial_terms":evidence.binomial_terms(),
                    "module_rank":evidence.frobenius().source_dimension(),
                });
            }
            if let Some(counts) = frobenius.point_counts() {
                result["extension_counts"] = json!(counts
                    .counts()
                    .iter()
                    .map(|record| json!({
                        "degree":record.degree(), "count":record.count().to_string(),
                        "field_order":counts.base().order().pow(record.degree()).to_string(),
                        "method":format!("{:?}", record.evidence().method()),
                    }))
                    .collect::<Vec<_>>());
            }
            result
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(8..=9).contains(&args.len()) {
        return Err("usage: bench_hyperelliptic_counts PRIME RESIDUE_DEGREE GENUS odd|even direct|kedlaya WARMUPS REPEATS frobenius|point_count [MODEL_JSON]".into());
    }
    let p = args[0].parse::<u64>()?;
    let n = args[1].parse::<usize>()?;
    let genus = args[2].parse::<usize>()?;
    let even = match args[3].as_str() {
        "odd" => false,
        "even" => true,
        _ => return Err("degree parity must be odd or even".into()),
    };
    let method = &args[4];
    let warmups = args[5].parse::<usize>()?;
    let repeats = args[6].parse::<usize>()?;
    let goal = args[7].as_str();
    if !matches!(goal, "frobenius" | "point_count") {
        return Err("goal must be frobenius or point_count".into());
    }
    if genus == 0 || genus > 64 || repeats == 0 || repeats > 100 || warmups > 100 {
        return Err("require genus 1..64, repeats 1..100, and warmups 0..100".into());
    }
    let spec = args
        .get(8)
        .map(|value| serde_json::from_str::<ModelSpec>(value))
        .transpose()?
        .unwrap_or_default();
    let curve = expanded_fixture(p, n, genus, even, &spec)?;
    let geometry = geometry_status(&curve)?;
    let original_degree = curve
        .f()
        .degree(None)?
        .and_then(|degree| degree.to_u64_checked())
        .ok_or("finite original degree required")?;
    emit(json!({
        "event": "fixture",
        "curve_json": curve.to_json()?,
        "field_order": curve.base().order().to_string(),
        "goal":goal,
        "direct_required_degrees":if goal == "frobenius" {curve.genus()} else {1},
        "largest_direct_field":curve.base().order().pow(if goal == "frobenius" {curve.genus() as u64} else {1}).to_string(),
        "field_modulus": curve.base().defining_modulus(),
        "model_spec":spec,
        "geometry":geometry,
        "infinity_residue_degrees":curve.infinity_residue_degrees(),
        "branch_json":curve.branch_equation().to_json()?,
        "coefficients": (0..=original_degree)
            .map(|i| curve.f().coefficient(&[Integer::from(i)]).map(|c| c.to_string()))
            .collect::<Result<Vec<_>, _>>()?,
    }))?;
    let mut expected = None;
    let mut last_metadata = Value::Null;
    for index in 0..warmups + repeats {
        emit(json!({"event":"start", "index":index, "warmup":index < warmups}))?;
        let started = Instant::now();
        let result = black_box(operation(black_box(&curve), method, goal));
        let seconds = started.elapsed().as_secs_f64();
        let (count, output) = match result {
            Ok(result) => result,
            Err(error) => {
                let unavailable = matches!(
                    error.downcast_ref::<HyperellipticError>(),
                    Some(
                        HyperellipticError::ResourceLimit { .. }
                            | HyperellipticError::Unsupported(_)
                    )
                ) || matches!(
                    error.downcast_ref::<PointCountError>(),
                    Some(
                        PointCountError::ResourceLimit { .. }
                            | PointCountError::TraceFormulaUnavailable(_)
                    )
                );
                emit(
                    json!({"event":"error", "kind":if unavailable {"unavailable"} else {"error"}, "error":error.to_string(), "seconds":seconds}),
                )?;
                return Ok(());
            }
        };
        if expected.as_ref().is_some_and(|previous| previous != &count) {
            return Err("nondeterministic count".into());
        }
        expected = Some(count.clone());
        last_metadata = metadata(&output);
        emit(json!({
            "event":"sample", "warmup":index < warmups,
            "seconds":seconds, "count":count.to_string(), "metadata":last_metadata,
        }))?;
    }
    let peak_rss_kib = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmHWM:"))
                .map(str::to_owned)
        })
        .and_then(|line| line.split_whitespace().nth(1)?.parse::<u64>().ok());
    emit(json!({
        "event":"complete", "count":expected.map(|c| c.to_string()),
        "metadata":last_metadata, "peak_rss_kib":peak_rss_kib,
    }))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct Corpus {
        schema: u32,
        models: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        id: String,
        prime: u64,
        residue_degree: usize,
        genus: usize,
        parity: String,
        fixture: ModelSpec,
        methods: Vec<String>,
    }

    #[test]
    fn expanded_corpus_has_reproducible_smooth_models_and_declared_geometry() {
        let corpus: Corpus = serde_json::from_str(include_str!(
            "../../../validation/bench/hyperelliptic_models.json"
        ))
        .unwrap();
        assert_eq!(corpus.schema, 1);
        assert_eq!(corpus.models.len(), 60);
        let mut identifiers = std::collections::BTreeSet::new();
        for case in corpus.models {
            assert!(identifiers.insert(case.id.clone()));
            let even = case.parity == "even";
            let curve = expanded_fixture(
                case.prime,
                case.residue_degree,
                case.genus,
                even,
                &case.fixture,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.id));
            assert_eq!(curve.genus(), case.genus, "{}", case.id);
            assert_eq!(
                curve.branch_equation().degree(None).unwrap(),
                Some(Integer::from(2 * case.genus + 1 + usize::from(even)))
            );
            assert_eq!(!curve.h().is_zero(), case.fixture.general, "{}", case.id);
            let field = curve.base();
            let leading = curve.branch_equation().leading_coefficient().unwrap();
            let sign = leading.pow((field.order().to_u64_checked().unwrap() - 1) / 2);
            assert_eq!(
                sign == field.one(),
                case.fixture.leading != Leading::Nonsquare,
                "{}",
                case.id
            );
            if case.fixture.leading == Leading::Square {
                assert_ne!(leading, field.one(), "{}", case.id);
            }
            if even {
                assert_eq!(
                    curve.infinity_residue_degrees(),
                    if case.fixture.leading == Leading::Nonsquare {
                        vec![2]
                    } else {
                        vec![1, 1]
                    }
                );
            }
            let geometry =
                geometry_status(&curve).unwrap_or_else(|error| panic!("{}: {error}", case.id));
            assert_eq!(geometry["curve"], "verified");
            if geometry["weighted_jacobian"] == "unavailable" {
                assert!(
                    case.genus >= 16 && case.methods == ["kedlaya"],
                    "{}",
                    case.id
                );
            }
            if case.methods.iter().any(|method| method == "direct") {
                assert!(field.order().pow(case.genus as u64) <= Integer::from(1_100_000));
            }
            if case.fixture.family == Family::Dense && field.degree() > 1 {
                assert!(curve
                    .branch_equation()
                    .terms()
                    .any(|(_, coefficient)| coefficient
                        .coefficients()
                        .iter()
                        .skip(1)
                        .any(|coordinate| !coordinate.is_zero())));
            }
            let repeated = expanded_fixture(
                case.prime,
                case.residue_degree,
                case.genus,
                even,
                &case.fixture,
            )
            .unwrap();
            assert_eq!(curve.to_json().unwrap(), repeated.to_json().unwrap());
        }
    }

    #[test]
    fn completion_square_scaling_and_true_extension_twists_preserve_expected_frobenius() {
        let base_spec = ModelSpec {
            family: Family::Dense,
            seed: 71,
            modulus: Some(vec![1, 0, 1]),
            ..ModelSpec::default()
        };
        for even in [false, true] {
            let base = expanded_fixture(3, 2, 2, even, &base_spec).unwrap();
            let completed = expanded_fixture(
                3,
                2,
                2,
                even,
                &ModelSpec {
                    general: true,
                    ..base_spec.clone()
                },
            )
            .unwrap();
            assert_eq!(base.branch_equation(), completed.branch_equation());
            assert_ne!(base, completed);
            let square = expanded_fixture(
                3,
                2,
                2,
                even,
                &ModelSpec {
                    general: true,
                    leading: Leading::Square,
                    ..base_spec.clone()
                },
            )
            .unwrap();
            let twist = expanded_fixture(
                3,
                2,
                2,
                even,
                &ModelSpec {
                    general: true,
                    leading: Leading::Nonsquare,
                    ..base_spec.clone()
                },
            )
            .unwrap();
            let frobenius = base.frobenius_polynomial().unwrap();
            assert_eq!(
                completed.frobenius_polynomial().unwrap().polynomial(),
                frobenius.polynomial()
            );
            assert_eq!(
                square.frobenius_polynomial().unwrap().polynomial(),
                frobenius.polynomial()
            );
            let t = frobenius.polynomial().ring().generator(0).unwrap();
            let minus_t = (&frobenius.polynomial().ring().zero() - &t).unwrap();
            assert_eq!(
                twist.frobenius_polynomial().unwrap().polynomial(),
                &frobenius.polynomial().substitute(&[minus_t]).unwrap()
            );
            assert_eq!(
                HyperellipticCurve::from_json(&twist.to_json().unwrap()).unwrap(),
                twist
            );
        }
    }

    #[test]
    fn fixture_regimes_fail_closed() {
        assert!(expanded_fixture(
            3,
            1,
            2,
            false,
            &ModelSpec {
                family: Family::Split,
                seed: 1,
                ..ModelSpec::default()
            }
        )
        .is_err());
        assert!(expanded_fixture(
            3,
            1,
            2,
            false,
            &ModelSpec {
                family: Family::Dense,
                leading: Leading::Square,
                ..ModelSpec::default()
            }
        )
        .is_err());
        assert!(expanded_fixture(
            3,
            3,
            2,
            false,
            &ModelSpec {
                family: Family::Dense,
                modulus: Some(vec![1, 0, 1]),
                ..ModelSpec::default()
            }
        )
        .is_err());
        assert!(expanded_fixture(
            3,
            1,
            2,
            false,
            &ModelSpec {
                seed: 1,
                ..ModelSpec::default()
            }
        )
        .is_err());
        assert!(serde_json::from_str::<ModelSpec>(r#"{"family":"dense","unknown":true}"#).is_err());
    }

    #[test]
    fn small_characteristic_fixtures_do_not_collapse_to_inseparable_equations() {
        for genus in [2, 5] {
            let curve = fixture(3, 1, genus, true).unwrap();
            assert_eq!(curve.genus(), genus);
            assert!(!curve.branch_equation().derivative(0).unwrap().is_zero());
            assert!(curve.smoothness_evidence().unwrap().is_smooth());
        }
    }

    #[test]
    fn full_frobenius_corpus_has_checked_smooth_fixtures() {
        for (p, n, g) in [
            (3, 12, 1),
            (3, 6, 2),
            (3, 4, 3),
            (3, 3, 4),
            (13, 1, 5),
            (3, 2, 6),
        ] {
            for even in [false, true] {
                let curve = fixture(p, n, g, even).unwrap();
                assert_eq!(curve.genus(), g);
                let smoothness = curve.smoothness_evidence().unwrap();
                assert!(smoothness.is_smooth());
                assert_eq!(smoothness.branch_is_smooth(), Some(true));
            }
        }
    }
}
