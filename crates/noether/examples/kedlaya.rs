// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Public-facade hyperelliptic point counting: no mathematics lives in this CLI.
//!
//! `cargo run -p noether --release --example kedlaya -- 3 0,1,0,0,0,0,0,0,0,1`
//! supplies ascending coefficients for `y^2=f(x)`. An optional final argument
//! specifies the target p-adic precision. Odd and even degrees, nonmonic models,
//! and all positive genera are admitted within the facade's resource bounds.

use noether::commalg::{HyperellipticCurve, MonomialOrder, PolynomialRing};
use noether::numthy::FiniteField;
use noether::Integer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.iter().any(|value| value == "--help") {
        println!("kedlaya [ODD_PRIME COEFFICIENTS_ASCENDING [PRECISION]]\nDefault: 3 0,1,0,0,0,0,0,0,0,1\nUses Noether's checked hyperelliptic model and integral Frobenius computation.");
        return Ok(());
    }
    if !arguments.is_empty() && !(2..=3).contains(&arguments.len()) {
        return Err("usage: kedlaya [ODD_PRIME COEFFICIENTS_ASCENDING [PRECISION]]".into());
    }
    let prime = arguments
        .first()
        .map_or(Ok(3), |value| value.parse::<u64>())?;
    let coefficients = arguments
        .get(1)
        .map_or("0,1,0,0,0,0,0,0,0,1", String::as_str)
        .split(',')
        .map(str::parse::<Integer>)
        .collect::<Result<Vec<_>, _>>()?;
    let precision = arguments
        .get(2)
        .map(|value| value.parse::<u64>())
        .transpose()?;
    let field = FiniteField::prime(prime)?;
    let ring = PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default())?;
    let polynomial = ring.from_terms(
        coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| (vec![Integer::from(i)], field.element(c.clone()))),
    )?;
    let curve = HyperellipticCurve::new(polynomial)?;
    let evidence = curve.frobenius_evidence(precision)?;
    let frobenius = evidence.frobenius_polynomial()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "curve":serde_json::from_str::<serde_json::Value>(&curve.to_json()?)?,
            "genus":curve.genus(),
            "infinity_residue_degrees":curve.infinity_residue_degrees(),
            "precision":evidence.precision(),
            "working_precision":evidence.working_precision(),
            "denominator_loss":evidence.denominator_loss(),
            "binomial_terms":evidence.binomial_terms(),
            "frobenius_polynomial":frobenius.polynomial().to_string(),
            "point_counts_degrees_1_through_genus":(1..=curve.genus()).map(|i|frobenius.point_count(i as u64)).collect::<Result<Vec<_>,_>>()?,
        }))?
    );
    Ok(())
}
