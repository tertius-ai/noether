// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Strict operation adapters for optional external Groebner computations.
//!
//! Process isolation remains in `noether-bridges`. This module owns the
//! mathematical side of the boundary: context-specific input, a deliberately
//! small output grammar, reconstruction in the original polynomial context,
//! and exact verification of ideal or rational-parametrization claims.

#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess"
))]
use core::fmt::Write as _;
use std::any::Any;
#[cfg(any(feature = "msolve-subprocess", test))]
use std::collections::HashMap;

#[cfg(any(feature = "msolve-subprocess", test))]
use crate::linalg::Matrix;
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{Field, Integer, Rational, RationalField, Ring};

use super::groebner::{interreduce, normal_form_against, verify_reduced_basis, GroebnerError};
#[cfg(any(feature = "msolve-subprocess", test))]
use super::{Ideal, MonomialOrder, RationalUnivariateRepresentation};
use super::{Polynomial, PolynomialRing};

#[cfg(any(
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
const PROTOCOL_START: &str = "NOETHER_GROEBNER_V1";
#[cfg(any(
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
const PROTOCOL_END: &str = "END_NOETHER_GROEBNER_V1";
const MAX_EXTERNAL_BASIS_SIZE: usize = 100_000;
const MAX_EXTERNAL_CERTIFICATE_CELLS: usize = 10_000_000;
#[cfg(any(feature = "msolve-subprocess", test))]
const MAX_EXTERNAL_CERTIFICATE_DEGREE: usize = 32;
const MAX_EXTERNAL_REDUCTION_STEPS: usize = 1_000_000;
#[cfg(any(feature = "msolve-subprocess", test))]
const MAX_EXTERNAL_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
#[cfg(any(feature = "msolve-subprocess", test))]
const MAX_EXTERNAL_PARSE_NODES: usize = 2_000_000;
#[cfg(any(feature = "msolve-subprocess", test))]
const MAX_EXTERNAL_PARSE_DEPTH: usize = 32;
#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess"
))]
const EXTERNAL_TIMEOUT_SECS: f64 = 3_600.0;

#[derive(Clone)]
enum ExternalCoefficientContext {
    Rational,
    Prime(FiniteField),
}

impl ExternalCoefficientContext {
    fn detect<R: Field>(ring: &PolynomialRing<R>) -> Option<Self> {
        let coefficients = ring.coefficient_ring() as &dyn Any;
        if coefficients.is::<RationalField>() {
            return Some(Self::Rational);
        }
        let field = coefficients.downcast_ref::<FiniteField>()?;
        (field.degree() == 1).then(|| Self::Prime(field.clone()))
    }

    #[cfg(any(
        feature = "macaulay2-subprocess",
        feature = "singular-subprocess",
        test
    ))]
    fn label(&self) -> String {
        match self {
            Self::Rational => "Q".to_owned(),
            Self::Prime(field) => format!("Fp:{}", field.characteristic()),
        }
    }

    #[cfg(feature = "macaulay2-subprocess")]
    fn macaulay2_ring(&self) -> String {
        match self {
            Self::Rational => "QQ".to_owned(),
            Self::Prime(field) => format!("ZZ/{}", field.characteristic()),
        }
    }

    #[cfg(feature = "singular-subprocess")]
    fn singular_ring(&self) -> String {
        match self {
            Self::Rational => "0".to_owned(),
            Self::Prime(field) => field.characteristic().to_string(),
        }
    }

    #[cfg(any(feature = "msolve-subprocess", test))]
    fn msolve_characteristic(&self) -> Option<u64> {
        match self {
            Self::Rational => Some(0),
            Self::Prime(field) => field.characteristic().to_u64_checked(),
        }
    }

    fn format_element<R: Field>(&self, value: &R::Element) -> Result<String, GroebnerError> {
        match self {
            Self::Rational => (value as &dyn Any)
                .downcast_ref::<Rational>()
                .map(ToString::to_string)
                .ok_or_else(|| external_error("Noether", "rational coefficient type mismatch")),
            Self::Prime(_) => (value as &dyn Any)
                .downcast_ref::<FiniteFieldElement>()
                .map(|value| value.representative().to_string())
                .ok_or_else(|| external_error("Noether", "prime-field coefficient type mismatch")),
        }
    }

    fn parse_element<R: Field>(
        &self,
        ring: &PolynomialRing<R>,
        numerator: Integer,
        denominator: Integer,
        backend: &'static str,
    ) -> Result<R::Element, GroebnerError> {
        if denominator.is_zero() {
            return Err(external_error(
                backend,
                "a coefficient has zero denominator",
            ));
        }
        let value: Box<dyn Any> = match self {
            Self::Rational => Box::new(
                Rational::new(numerator, denominator)
                    .map_err(|_| external_error(backend, "a coefficient has zero denominator"))?,
            ),
            Self::Prime(field) => {
                let denominator = field.element(denominator);
                let inverse = field.inverse(&denominator).map_err(|_| {
                    external_error(backend, "a coefficient denominator vanishes in the field")
                })?;
                Box::new(
                    field
                        .mul(&field.element(numerator), &inverse)
                        .map_err(|_| {
                            external_error(backend, "finite-field coefficient mismatch")
                        })?,
                )
            }
        };
        let value = value.downcast::<R::Element>().map_err(|_| {
            external_error(
                backend,
                "parsed coefficient does not match the original context",
            )
        })?;
        let value = *value;
        if !ring.coefficient_ring().contains(&value) {
            return Err(external_error(
                backend,
                "parsed coefficient is outside the original context",
            ));
        }
        Ok(value)
    }
}

struct CertifiedBasis<R: Field> {
    backend: &'static str,
    basis: Vec<Polynomial<R>>,
    /// `certificates[i][j]` multiplies source generator `i` in basis column `j`.
    certificates: Vec<Vec<Polynomial<R>>>,
}

/// Try exact certificate-bearing subprocesses after the native path reaches a
/// declared resource limit. Unsupported or unavailable tools return `None`;
/// malformed output is an error and is never silently accepted.
#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess"
))]
pub(super) fn resource_fallback<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    try_msolve(ring, generators)
        .or_else(|| try_singular(ring, generators))
        .or_else(|| try_macaulay2(ring, generators))
}

#[cfg(feature = "msolve-subprocess")]
pub(super) fn rur_resource_fallback<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Option<Result<RationalUnivariateRepresentation<R>, GroebnerError>> {
    try_msolve_rur(ideal)
}

#[cfg(feature = "msolve-subprocess")]
fn try_msolve_rur<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Option<Result<RationalUnivariateRepresentation<R>, GroebnerError>> {
    use noether_bridges::BridgeError;

    let ring = ideal.ring();
    let context = ExternalCoefficientContext::detect(ring)?;
    let characteristic = context.msolve_characteristic()?;
    if characteristic >= (1_u64 << 31) || ring.term_order().standard_name() != Some("grevlex") {
        return None;
    }
    let input = match msolve_input(ring, ideal.generators(), &context) {
        Ok(input) => input,
        Err(error) => return Some(Err(error)),
    };
    let output = match noether_bridges::msolve::run_msolve(
        &input,
        &["-P", "2", "-c", "0"],
        Some(EXTERNAL_TIMEOUT_SECS),
    ) {
        Ok(output) => output,
        Err(BridgeError::Unavailable(_)) => return None,
        Err(error) => return Some(Err(external_error("msolve", error.to_string()))),
    };
    Some(parse_msolve_rur(ideal, &context, &output))
}

#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess"
))]
fn try_msolve<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    #[cfg(feature = "msolve-subprocess")]
    {
        use noether_bridges::BridgeError;

        let context = ExternalCoefficientContext::detect(ring)?;
        // msolve 0.x/1.x prints an exact reduced basis only for prime fields;
        // characteristic-zero `-g 2` is a randomly selected modular image.
        let ExternalCoefficientContext::Prime(field) = &context else {
            return None;
        };
        if field
            .characteristic()
            .to_u64_checked()
            .is_none_or(|characteristic| characteristic >= (1_u64 << 31))
            || ring.term_order().standard_name() != Some("grevlex")
        {
            return None;
        }
        let input = match msolve_input(ring, generators, &context) {
            Ok(input) => input,
            Err(error) => return Some(Err(error)),
        };
        let output = match noether_bridges::msolve::run_msolve(
            &input,
            &["-g", "2"],
            Some(EXTERNAL_TIMEOUT_SECS),
        ) {
            Ok(output) => output,
            Err(BridgeError::Unavailable(_)) => return None,
            Err(error) => return Some(Err(external_error("msolve", error.to_string()))),
        };
        Some(
            parse_msolve_basis(ring, &context, &output)
                .and_then(|basis| reconstruct_basis_certificates(ring, generators, basis, "msolve"))
                .and_then(|candidate| verify_certified_basis(ring, generators, candidate)),
        )
    }
    #[cfg(not(feature = "msolve-subprocess"))]
    {
        let _ = (ring, generators);
        None
    }
}

#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess"
))]
fn try_macaulay2<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    #[cfg(feature = "macaulay2-subprocess")]
    {
        use noether_bridges::BridgeError;

        let context = ExternalCoefficientContext::detect(ring)?;
        let script = match macaulay2_script(ring, generators, &context) {
            Ok(script) => script,
            Err(error) => return Some(Err(error)),
        };
        let output = match noether_bridges::macaulay2::m2_eval(&script, Some(EXTERNAL_TIMEOUT_SECS))
        {
            Ok(output) => output,
            Err(BridgeError::Unavailable(_)) => return None,
            Err(error) => {
                return Some(Err(external_error("Macaulay2", error.to_string())));
            }
        };
        Some(
            parse_certified_basis(ring, generators.len(), &context, "Macaulay2", &output)
                .and_then(|candidate| verify_certified_basis(ring, generators, candidate)),
        )
    }
    #[cfg(not(feature = "macaulay2-subprocess"))]
    {
        let _ = (ring, generators);
        None
    }
}

#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess"
))]
fn try_singular<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
) -> Option<Result<Vec<Polynomial<R>>, GroebnerError>> {
    #[cfg(feature = "singular-subprocess")]
    {
        use noether_bridges::BridgeError;

        let context = ExternalCoefficientContext::detect(ring)?;
        let script = match singular_script(ring, generators, &context) {
            Ok(script) => script,
            Err(error) => return Some(Err(error)),
        };
        let output =
            match noether_bridges::singular::singular_eval(&script, Some(EXTERNAL_TIMEOUT_SECS)) {
                Ok(output) => output,
                Err(BridgeError::Unavailable(_)) => return None,
                Err(error) => return Some(Err(external_error("Singular", error.to_string()))),
            };
        Some(
            parse_certified_basis(ring, generators.len(), &context, "Singular", &output)
                .and_then(|candidate| verify_certified_basis(ring, generators, candidate)),
        )
    }
    #[cfg(not(feature = "singular-subprocess"))]
    {
        let _ = (ring, generators);
        None
    }
}

#[cfg(feature = "macaulay2-subprocess")]
fn macaulay2_script<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    context: &ExternalCoefficientContext,
) -> Result<String, GroebnerError> {
    if generators.is_empty() || generators.iter().any(Polynomial::is_zero) {
        return Err(external_error(
            "Macaulay2",
            "the certificate adapter requires a nonempty, nonzero generator family",
        ));
    }
    let order = match ring.term_order().standard_name() {
        Some("lex") => "Lex",
        Some("grlex") => "GLex",
        Some("grevlex") => "GRevLex",
        _ => return Err(external_error("Macaulay2", "unsupported monomial order")),
    };
    let variables = backend_variables(ring.variables().len());
    let values = generators
        .iter()
        .map(|value| format_polynomial(value, context, &variables, "Macaulay2"))
        .collect::<Result<Vec<_>, _>>()?
        .join(",");
    let mut script = String::new();
    writeln!(
        script,
        "R = {}[{}, MonomialOrder => {}];",
        context.macaulay2_ring(),
        variables.join(","),
        order
    )
    .expect("writing to String is infallible");
    writeln!(script, "F = matrix{{{{{values}}}}};").expect("writing to String is infallible");
    writeln!(script, "I = ideal F;").expect("writing to String is infallible");
    if matches!(context, ExternalCoefficientContext::Prime(_)) {
        writeln!(script, "G = groebnerBasis(I, Strategy => \"F4\");")
            .expect("writing to String is infallible");
    } else {
        writeln!(script, "G = gens gb I;").expect("writing to String is infallible");
    }
    writeln!(script, "T = G // F;").expect("writing to String is infallible");
    write_protocol_header(
        &mut script,
        context,
        &variables,
        ring.term_order().name(),
        generators.len(),
        "print ",
    );
    writeln!(script, "print(\"basis=\" | toString numColumns G);")
        .expect("writing to String is infallible");
    writeln!(
        script,
        "scan(0..numColumns G-1, j -> print(\"poly=\" | toExternalString G_(0,j)));"
    )
    .expect("writing to String is infallible");
    writeln!(script, "scan(0..numRows T-1, i -> scan(0..numColumns T-1, j -> print(\"cert=\" | toString i | \",\" | toString j | \":\" | toExternalString T_(i,j))));")
        .expect("writing to String is infallible");
    writeln!(script, "print \"{PROTOCOL_END}\";").expect("writing to String is infallible");
    writeln!(script, "exit 0;").expect("writing to String is infallible");
    Ok(script)
}

#[cfg(feature = "singular-subprocess")]
fn singular_script<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    context: &ExternalCoefficientContext,
) -> Result<String, GroebnerError> {
    if generators.is_empty() || generators.iter().any(Polynomial::is_zero) {
        return Err(external_error(
            "Singular",
            "the certificate adapter requires a nonempty, nonzero generator family",
        ));
    }
    let order = match ring.term_order().standard_name() {
        Some("lex") => "lp",
        Some("grlex") => "Dp",
        Some("grevlex") => "dp",
        _ => return Err(external_error("Singular", "unsupported monomial order")),
    };
    let variables = backend_variables(ring.variables().len());
    let values = generators
        .iter()
        .map(|value| format_polynomial(value, context, &variables, "Singular"))
        .collect::<Result<Vec<_>, _>>()?
        .join(",");
    let mut script = String::new();
    writeln!(
        script,
        "ring r = {},({}),{};",
        context.singular_ring(),
        variables.join(","),
        order
    )
    .expect("writing to String is infallible");
    writeln!(script, "ideal F = {values};").expect("writing to String is infallible");
    writeln!(script, "matrix T; ideal G = liftstd(F,T);").expect("writing to String is infallible");
    writeln!(script, "print(\"{PROTOCOL_START}\");").expect("writing to String is infallible");
    writeln!(script, "print(\"context={}\");", context.label())
        .expect("writing to String is infallible");
    writeln!(script, "print(\"variables={}\");", variables.join(","))
        .expect("writing to String is infallible");
    writeln!(script, "print(\"order={}\");", ring.term_order().name())
        .expect("writing to String is infallible");
    writeln!(script, "print(\"generators={}\");", generators.len())
        .expect("writing to String is infallible");
    writeln!(script, "print(\"basis=\" + string(size(G)));")
        .expect("writing to String is infallible");
    writeln!(
        script,
        "int i,j; for(j=1;j<=size(G);j++){{print(\"poly=\"+string(G[j]));}}"
    )
    .expect("writing to String is infallible");
    writeln!(script, "for(i=1;i<=nrows(T);i++){{for(j=1;j<=ncols(T);j++){{print(\"cert=\"+string(i-1)+\",\"+string(j-1)+\":\"+string(T[i,j]));}}}}")
        .expect("writing to String is infallible");
    writeln!(script, "print(\"{PROTOCOL_END}\"); quit;").expect("writing to String is infallible");
    Ok(script)
}

#[cfg(feature = "macaulay2-subprocess")]
fn write_protocol_header(
    output: &mut String,
    context: &ExternalCoefficientContext,
    variables: &[String],
    order: &str,
    generators: usize,
    command: &str,
) {
    for value in [
        PROTOCOL_START.to_owned(),
        format!("context={}", context.label()),
        format!("variables={}", variables.join(",")),
        format!("order={order}"),
        format!("generators={generators}"),
    ] {
        writeln!(output, "{command}\"{value}\";").expect("writing to String is infallible");
    }
}

#[cfg(feature = "msolve-subprocess")]
fn msolve_input<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    context: &ExternalCoefficientContext,
) -> Result<String, GroebnerError> {
    if generators.is_empty() || generators.iter().any(Polynomial::is_zero) {
        return Err(external_error(
            "msolve",
            "the basis adapter requires a nonempty, nonzero generator family",
        ));
    }
    if ring.term_order().standard_name() != Some("grevlex") {
        return Err(external_error(
            "msolve",
            "the basis adapter requires graded reverse lexicographic order",
        ));
    }
    let characteristic = context
        .msolve_characteristic()
        .ok_or_else(|| external_error("msolve", "the characteristic does not fit its input"))?;
    if characteristic >= (1_u64 << 31) {
        return Err(external_error(
            "msolve",
            "the characteristic must be smaller than 2^31",
        ));
    }
    let variables = backend_variables(ring.variables().len());
    let values = generators
        .iter()
        .map(|value| format_msolve_polynomial(value, context, &variables))
        .collect::<Result<Vec<_>, _>>()?;
    let mut input = format!("{}\n{characteristic}\n", variables.join(","));
    input.push_str(&values.join(",\n"));
    input.push('\n');
    Ok(input)
}

#[cfg(feature = "msolve-subprocess")]
fn format_msolve_polynomial<R: Field>(
    polynomial: &Polynomial<R>,
    context: &ExternalCoefficientContext,
    variables: &[String],
) -> Result<String, GroebnerError> {
    if polynomial.is_zero() {
        return Ok("0".to_owned());
    }
    let mut output = String::new();
    for (index, (exponents, coefficient)) in polynomial.terms().enumerate() {
        let coefficient = context.format_element::<R>(coefficient)?;
        if index != 0 && !coefficient.starts_with('-') {
            output.push('+');
        }
        output.push_str(&coefficient);
        for (variable, exponent) in variables.iter().zip(exponents) {
            if exponent.is_zero() {
                continue;
            }
            write!(output, "*{variable}^{exponent}").expect("writing to String is infallible");
        }
    }
    Ok(output)
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn parse_msolve_basis<R: Field>(
    ring: &PolynomialRing<R>,
    context: &ExternalCoefficientContext,
    output: &str,
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    if !output.is_ascii() {
        return Err(external_error("msolve", "basis output is not ASCII"));
    }
    let lines = output.lines().map(str::trim).collect::<Vec<_>>();
    if lines.len() < 5 {
        return Err(external_error("msolve", "basis output is truncated"));
    }
    let characteristic = context
        .msolve_characteristic()
        .ok_or_else(|| external_error("msolve", "the characteristic does not fit its output"))?;
    let variables = backend_variables(ring.variables().len());
    let (count, body_start) = if lines[0] == "#Reduced Groebner basis data" {
        if lines.len() < 8 || lines[1] != "#---" || lines[6] != "#---" {
            return Err(external_error("msolve", "basis metadata is truncated"));
        }
        if lines[2] != format!("#field characteristic: {characteristic}") {
            return Err(external_error(
                "msolve",
                "wrong or missing characteristic header",
            ));
        }
        if lines[3].strip_prefix("#variable order:").map(str::trim)
            != Some(variables.join(", ").as_str())
        {
            return Err(external_error(
                "msolve",
                "wrong or missing variable-order header",
            ));
        }
        if lines[4] != "#monomial order:       graded reverse lexicographical" {
            return Err(external_error(
                "msolve",
                "wrong or missing monomial-order header",
            ));
        }
        let count = lines[5]
            .strip_prefix("#length of basis:")
            .map(str::trim)
            .and_then(|value| {
                value.strip_suffix(" elements sorted by increasing leading monomials")
            })
            .ok_or_else(|| external_error("msolve", "wrong or missing basis-count header"))?;
        (count, 7)
    } else {
        if lines[0]
            != format!("#Reduced Groebner basis for input in characteristic {characteristic}")
        {
            return Err(external_error(
                "msolve",
                "wrong or missing characteristic header",
            ));
        }
        if lines[1] != format!("#for variable order {}", variables.join(", ")) {
            return Err(external_error(
                "msolve",
                "wrong or missing variable-order header",
            ));
        }
        if lines[2] != "#w.r.t. grevlex monomial ordering" {
            return Err(external_error(
                "msolve",
                "wrong or missing monomial-order header",
            ));
        }
        let count = lines[3]
            .strip_prefix("#consisting of ")
            .and_then(|value| value.strip_suffix(" elements:"))
            .ok_or_else(|| external_error("msolve", "wrong or missing basis-count header"))?;
        (count, 4)
    };
    if count.is_empty() || !count.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(external_error("msolve", "invalid basis count"));
    }
    let count = count
        .parse::<usize>()
        .map_err(|_| external_error("msolve", "basis count does not fit usize"))?;
    if count == 0 || count > MAX_EXTERNAL_BASIS_SIZE {
        return Err(external_error(
            "msolve",
            "basis count is outside the accepted boundary",
        ));
    }
    let body = lines[body_start..].join("");
    let body = body
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix("]:"))
        .ok_or_else(|| external_error("msolve", "basis list is truncated or malformed"))?;
    let records = body.split(',').collect::<Vec<_>>();
    if records.len() != count || records.iter().any(|record| record.is_empty()) {
        return Err(external_error(
            "msolve",
            "basis list disagrees with its declared count",
        ));
    }
    records
        .into_iter()
        .map(|record| parse_polynomial(ring, context, &variables, "msolve", record))
        .collect()
}

#[cfg(any(feature = "msolve-subprocess", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
enum MsolveValue {
    Integer(Integer),
    Name(String),
    List(Vec<Self>),
}

#[cfg(any(feature = "msolve-subprocess", test))]
struct MsolveValueParser<'a> {
    input: &'a [u8],
    position: usize,
    nodes: usize,
}

#[cfg(any(feature = "msolve-subprocess", test))]
impl<'a> MsolveValueParser<'a> {
    fn new(input: &'a str) -> Result<Self, GroebnerError> {
        if !input.is_ascii() || input.len() > MAX_EXTERNAL_OUTPUT_BYTES {
            return Err(external_error(
                "msolve",
                "parametrization output is non-ASCII or exceeds the accepted byte boundary",
            ));
        }
        Ok(Self {
            input: input.as_bytes(),
            position: 0,
            nodes: 0,
        })
    }

    fn parse(mut self) -> Result<MsolveValue, GroebnerError> {
        let value = self.value(0)?;
        self.whitespace();
        if self.take() != Some(b':') {
            return Err(external_error(
                "msolve",
                "parametrization output has no final colon",
            ));
        }
        self.whitespace();
        if self.position != self.input.len() {
            return Err(external_error(
                "msolve",
                "nonempty output follows the parametrization",
            ));
        }
        Ok(value)
    }

    fn value(&mut self, depth: usize) -> Result<MsolveValue, GroebnerError> {
        if depth > MAX_EXTERNAL_PARSE_DEPTH || self.nodes == MAX_EXTERNAL_PARSE_NODES {
            return Err(external_error(
                "msolve",
                "parametrization syntax exceeds the accepted structural boundary",
            ));
        }
        self.nodes += 1;
        self.whitespace();
        match self.peek() {
            Some(b'[') => self.list(depth + 1),
            Some(b'\'') => self.name(),
            Some(b'-' | b'0'..=b'9') => self.integer(),
            _ => Err(external_error("msolve", "unexpected parametrization token")),
        }
    }

    fn list(&mut self, depth: usize) -> Result<MsolveValue, GroebnerError> {
        if self.take() != Some(b'[') {
            return Err(external_error(
                "msolve",
                "parametrization list has no opening bracket",
            ));
        }
        self.whitespace();
        let mut values = Vec::new();
        if self.peek() == Some(b']') {
            self.position += 1;
            return Ok(MsolveValue::List(values));
        }
        loop {
            values.push(self.value(depth)?);
            self.whitespace();
            match self.take() {
                Some(b',') => self.whitespace(),
                Some(b']') => break,
                _ => {
                    return Err(external_error("msolve", "malformed parametrization list"));
                }
            }
        }
        Ok(MsolveValue::List(values))
    }

    fn name(&mut self) -> Result<MsolveValue, GroebnerError> {
        if self.take() != Some(b'\'') {
            return Err(external_error(
                "msolve",
                "parametrization name has no opening quote",
            ));
        }
        let start = self.position;
        while self.peek().is_some_and(|byte| byte != b'\'') {
            let byte = self.take().expect("the peeked byte exists");
            if !(byte.is_ascii_alphanumeric() || byte == b'_') {
                return Err(external_error(
                    "msolve",
                    "a parametrization variable uses unsupported characters",
                ));
            }
        }
        if start == self.position || self.take() != Some(b'\'') {
            return Err(external_error(
                "msolve",
                "an output variable name is empty or unterminated",
            ));
        }
        Ok(MsolveValue::Name(
            core::str::from_utf8(&self.input[start..self.position - 1])
                .expect("ASCII is valid UTF-8")
                .to_owned(),
        ))
    }

    fn integer(&mut self) -> Result<MsolveValue, GroebnerError> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        let digits = self.position;
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.position += 1;
        }
        if digits == self.position {
            return Err(external_error("msolve", "invalid output integer"));
        }
        let value = core::str::from_utf8(&self.input[start..self.position])
            .expect("ASCII is valid UTF-8")
            .parse::<Integer>()
            .map_err(|_| external_error("msolve", "output integer does not fit its grammar"))?;
        Ok(MsolveValue::Integer(value))
    }

    fn whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.position += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.position).copied()
    }

    fn take(&mut self) -> Option<u8> {
        let value = self.peek()?;
        self.position += 1;
        Some(value)
    }
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn parse_msolve_rur<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    context: &ExternalCoefficientContext,
    output: &str,
) -> Result<RationalUnivariateRepresentation<R>, GroebnerError> {
    let value = MsolveValueParser::new(output)?.parse()?;
    let outer = msolve_list(&value, "top-level parametrization")?;
    if outer.len() != 2 || msolve_integer(&outer[0], "solution status")? != &Integer::zero() {
        return Err(external_error(
            "msolve",
            "output does not contain one finite nonempty solution parametrization",
        ));
    }
    let data = msolve_list(&outer[1], "parametrization data")?;
    if data.len() != 6 {
        return Err(external_error(
            "msolve",
            "parametrization data has the wrong arity",
        ));
    }
    let characteristic = context
        .msolve_characteristic()
        .ok_or_else(|| external_error("msolve", "the characteristic does not fit its output"))?;
    if msolve_integer(&data[0], "field characteristic")? != &Integer::from(characteristic) {
        return Err(external_error(
            "msolve",
            "parametrization has the wrong field characteristic",
        ));
    }
    let arity = ideal.ring().variables().len();
    if msolve_usize(&data[1], "variable count")? != arity {
        return Err(external_error(
            "msolve",
            "parametrization has the wrong variable count",
        ));
    }
    let quotient_dimension = msolve_usize(&data[2], "quotient dimension")?;
    if quotient_dimension == 0 || quotient_dimension > MAX_EXTERNAL_BASIS_SIZE {
        return Err(external_error(
            "msolve",
            "quotient dimension is outside the accepted boundary",
        ));
    }
    let expected_variables = backend_variables(arity);
    let variables = msolve_list(&data[3], "variable names")?;
    if variables.len() != arity
        || variables
            .iter()
            .zip(&expected_variables)
            .any(|(value, expected)| !matches!(value, MsolveValue::Name(name) if name == expected))
    {
        return Err(external_error(
            "msolve",
            "parametrization changed or reordered the controlled variables",
        ));
    }
    let form = msolve_list(&data[4], "linear form")?;
    if form.len() != arity
        || form.iter().enumerate().any(|(index, value)| {
            msolve_integer(value, "linear-form coefficient").map_or(true, |coefficient| {
                if index + 1 == arity {
                    coefficient != &Integer::one()
                } else {
                    coefficient != &Integer::zero()
                }
            })
        })
    {
        return Err(external_error(
            "msolve",
            "parametrization does not use the final controlled variable as parameter",
        ));
    }
    let wrapper = msolve_list(&data[5], "parametrization family")?;
    if wrapper.len() != 2
        || msolve_integer(&wrapper[0], "parametrization count")? != &Integer::one()
    {
        return Err(external_error(
            "msolve",
            "output does not contain exactly one parametrization",
        ));
    }
    let components = msolve_list(&wrapper[1], "parametrization components")?;
    if components.len() != 3 {
        return Err(external_error(
            "msolve",
            "parametrization has the wrong component count",
        ));
    }
    let parameter_ring = PolynomialRing::new(
        ideal.ring().coefficient_ring().clone(),
        ["t"],
        MonomialOrder::Lexicographic,
    )
    .map_err(|error| external_error("msolve", error.to_string()))?;
    let equation = msolve_univariate_polynomial(
        &parameter_ring,
        context,
        &components[0],
        "parameter equation",
    )?;
    if equation.degree(Some(0))? != Some(Integer::from(quotient_dimension)) {
        return Err(external_error(
            "msolve",
            "parameter equation degree disagrees with the quotient dimension",
        ));
    }
    let denominator = msolve_univariate_polynomial(
        &parameter_ring,
        context,
        &components[1],
        "coordinate denominator",
    )?;
    if denominator.is_zero() {
        return Err(external_error("msolve", "coordinate denominator is zero"));
    }
    let coordinates = msolve_list(&components[2], "coordinate numerators")?;
    if coordinates.len() + 1 != arity {
        return Err(external_error(
            "msolve",
            "parametrization has the wrong number of coordinate numerators",
        ));
    }
    let mut coordinate_numerators = Vec::with_capacity(arity);
    for coordinate in coordinates {
        let record = msolve_list(coordinate, "coordinate numerator")?;
        let (polynomial, scale) = match context {
            ExternalCoefficientContext::Rational => {
                if record.len() != 2 {
                    return Err(external_error(
                        "msolve",
                        "a rational coordinate record has the wrong arity",
                    ));
                }
                let polynomial = msolve_univariate_polynomial(
                    &parameter_ring,
                    context,
                    &record[0],
                    "coordinate numerator",
                )?;
                let scale = context.parse_element(
                    &parameter_ring,
                    msolve_integer(&record[1], "coordinate scale")?.clone(),
                    Integer::one(),
                    "msolve",
                )?;
                (polynomial, scale)
            }
            ExternalCoefficientContext::Prime(_) => {
                if record.len() != 1 {
                    return Err(external_error(
                        "msolve",
                        "a prime-field coordinate record has the wrong arity",
                    ));
                }
                (
                    msolve_univariate_polynomial(
                        &parameter_ring,
                        context,
                        &record[0],
                        "coordinate numerator",
                    )?,
                    parameter_ring.coefficient_ring().one(),
                )
            }
        };
        let inverse_scale = parameter_ring
            .coefficient_ring()
            .inverse(&scale)
            .map_err(|_| external_error("msolve", "a coordinate scale is zero"))?;
        let negative_inverse = parameter_ring.coefficient_ring().neg(&inverse_scale)?;
        coordinate_numerators.push(polynomial.scalar_mul(&negative_inverse)?);
    }
    let parameter = parameter_ring.generator(0)?;
    coordinate_numerators.push(parameter_ring.mul(&parameter, &denominator)?);
    let linear_form = (0..arity)
        .map(|index| {
            if index + 1 == arity {
                parameter_ring.coefficient_ring().one()
            } else {
                parameter_ring.coefficient_ring().zero()
            }
        })
        .collect();
    super::groebner::verified_rur_from_external(
        ideal,
        equation,
        coordinate_numerators,
        denominator,
        linear_form,
        quotient_dimension,
    )
    .map_err(|error| external_error("msolve", error.to_string()))
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn msolve_univariate_polynomial<R: Field>(
    ring: &PolynomialRing<R>,
    context: &ExternalCoefficientContext,
    value: &MsolveValue,
    label: &'static str,
) -> Result<Polynomial<R>, GroebnerError> {
    let record = msolve_list(value, label)?;
    if record.len() != 2 {
        return Err(external_error(
            "msolve",
            format!("{label} has the wrong record arity"),
        ));
    }
    let degree = msolve_usize(&record[0], label)?;
    if degree > MAX_EXTERNAL_BASIS_SIZE {
        return Err(external_error(
            "msolve",
            format!("{label} degree exceeds the accepted boundary"),
        ));
    }
    let coefficients = msolve_list(&record[1], label)?;
    if coefficients.len() != degree.saturating_add(1) {
        return Err(external_error(
            "msolve",
            format!("{label} coefficient count disagrees with its degree"),
        ));
    }
    let polynomial = ring.from_terms(
        coefficients
            .iter()
            .enumerate()
            .map(|(exponent, coefficient)| {
                context
                    .parse_element(
                        ring,
                        msolve_integer(coefficient, label)?.clone(),
                        Integer::one(),
                        "msolve",
                    )
                    .map(|coefficient| (vec![Integer::from(exponent)], coefficient))
            })
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    Ok(polynomial)
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn msolve_list<'a>(
    value: &'a MsolveValue,
    label: &'static str,
) -> Result<&'a [MsolveValue], GroebnerError> {
    match value {
        MsolveValue::List(values) => Ok(values),
        _ => Err(external_error("msolve", format!("{label} is not a list"))),
    }
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn msolve_integer<'a>(
    value: &'a MsolveValue,
    label: &'static str,
) -> Result<&'a Integer, GroebnerError> {
    match value {
        MsolveValue::Integer(value) => Ok(value),
        _ => Err(external_error(
            "msolve",
            format!("{label} is not an integer"),
        )),
    }
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn msolve_usize(value: &MsolveValue, label: &'static str) -> Result<usize, GroebnerError> {
    msolve_integer(value, label)?
        .to_u64_checked()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| external_error("msolve", format!("{label} does not fit usize")))
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn reconstruct_basis_certificates<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    basis: Vec<Polynomial<R>>,
    backend: &'static str,
) -> Result<CertifiedBasis<R>, GroebnerError> {
    if generators.is_empty()
        || generators.iter().any(Polynomial::is_zero)
        || basis.is_empty()
        || basis.iter().any(Polynomial::is_zero)
    {
        return Err(external_error(
            backend,
            "certificate reconstruction requires nonempty, nonzero polynomials",
        ));
    }
    let generator_degrees = generators
        .iter()
        .map(|polynomial| polynomial_total_degree(polynomial, backend))
        .collect::<Result<Vec<_>, _>>()?;
    let basis_degree = basis
        .iter()
        .map(|polynomial| polynomial_total_degree(polynomial, backend))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    let initial_degree = generator_degrees
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(basis_degree);
    if initial_degree > MAX_EXTERNAL_CERTIFICATE_DEGREE {
        return Err(GroebnerError::ResourceLimit {
            resource: "external certificate degree",
            limit: MAX_EXTERNAL_CERTIFICATE_DEGREE,
        });
    }
    let arity = ring.variables().len();
    for degree in initial_degree..=MAX_EXTERNAL_CERTIFICATE_DEGREE {
        let equations = macaulay_monomials(arity, degree, MAX_EXTERNAL_CERTIFICATE_CELLS)?;
        let equation_index = equations
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, exponents)| (exponents, index))
            .collect::<HashMap<_, _>>();
        let mut unknowns = Vec::<(usize, Vec<u32>)>::new();
        for (generator, generator_degree) in generator_degrees.iter().copied().enumerate() {
            unknowns.extend(
                macaulay_monomials(
                    arity,
                    degree - generator_degree,
                    MAX_EXTERNAL_CERTIFICATE_CELLS,
                )?
                .into_iter()
                .map(|monomial| (generator, monomial)),
            );
        }
        let left_cells = equations.len().checked_mul(unknowns.len());
        let right_cells = equations.len().checked_mul(basis.len());
        if left_cells
            .and_then(|left| right_cells.and_then(|right| left.checked_add(right)))
            .is_none_or(|cells| cells > MAX_EXTERNAL_CERTIFICATE_CELLS)
        {
            return Err(GroebnerError::ResourceLimit {
                resource: "external certificate Macaulay cells",
                limit: MAX_EXTERNAL_CERTIFICATE_CELLS,
            });
        }
        let mut left = vec![vec![ring.coefficient_ring().zero(); unknowns.len()]; equations.len()];
        for (column, (generator, multiplier)) in unknowns.iter().enumerate() {
            for (exponents, coefficient) in generators[*generator].terms() {
                let product = multiplier
                    .iter()
                    .zip(exponents)
                    .map(|(left, right)| {
                        right
                            .to_u64_checked()
                            .and_then(|right| u32::try_from(right).ok())
                            .and_then(|right| left.checked_add(right))
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| external_error(backend, "a certificate exponent overflowed"))?;
                let row = *equation_index.get(&product).ok_or_else(|| {
                    external_error(backend, "a certificate term exceeded its degree bound")
                })?;
                left[row][column] = coefficient.clone();
            }
        }
        let mut right = vec![vec![ring.coefficient_ring().zero(); basis.len()]; equations.len()];
        for (column, polynomial) in basis.iter().enumerate() {
            for (exponents, coefficient) in polynomial.terms() {
                let exponents = exponents
                    .iter()
                    .map(|exponent| {
                        exponent
                            .to_u64_checked()
                            .and_then(|value| u32::try_from(value).ok())
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| external_error(backend, "a basis exponent overflowed"))?;
                let row = *equation_index.get(&exponents).ok_or_else(|| {
                    external_error(backend, "a basis term exceeded its declared degree")
                })?;
                right[row][column] = coefficient.clone();
            }
        }
        let left = Matrix::from_rows(ring.coefficient_ring().clone(), left)
            .map_err(|error| external_error(backend, error.to_string()))?;
        let right = Matrix::from_rows(ring.coefficient_ring().clone(), right)
            .map_err(|error| external_error(backend, error.to_string()))?;
        let Some(solution) = left
            .particular_solution_columns(&right)
            .map_err(|error| external_error(backend, error.to_string()))?
        else {
            continue;
        };
        let mut certificates = vec![vec![ring.zero(); basis.len()]; generators.len()];
        for (row, (generator, monomial)) in unknowns.into_iter().enumerate() {
            for (basis_column, certificate) in certificates[generator].iter_mut().enumerate() {
                let coefficient = solution
                    .get(row, basis_column)
                    .expect("a certificate solution coefficient exists");
                if coefficient == ring.coefficient_ring().zero() {
                    continue;
                }
                let term = ring
                    .from_terms([(
                        monomial
                            .iter()
                            .copied()
                            .map(Integer::from)
                            .collect::<Vec<_>>(),
                        coefficient,
                    )])
                    .map_err(|error| external_error(backend, error.to_string()))?;
                *certificate = ring
                    .add(certificate, &term)
                    .map_err(|error| external_error(backend, error.to_string()))?;
            }
        }
        return Ok(CertifiedBasis {
            backend,
            basis,
            certificates,
        });
    }
    Err(external_error(
        backend,
        format!(
            "no generator certificate was found through degree {MAX_EXTERNAL_CERTIFICATE_DEGREE}"
        ),
    ))
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn polynomial_total_degree<R: Field>(
    polynomial: &Polynomial<R>,
    backend: &'static str,
) -> Result<usize, GroebnerError> {
    polynomial
        .terms()
        .map(|(exponents, _)| {
            exponents.iter().try_fold(0_usize, |degree, exponent| {
                exponent
                    .to_u64_checked()
                    .and_then(|value| usize::try_from(value).ok())
                    .and_then(|value| degree.checked_add(value))
                    .ok_or_else(|| external_error(backend, "a polynomial degree overflowed"))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|degrees| degrees.into_iter().max().unwrap_or(0))
}

#[cfg(any(feature = "msolve-subprocess", test))]
fn macaulay_monomials(
    arity: usize,
    degree: usize,
    maximum_count: usize,
) -> Result<Vec<Vec<u32>>, GroebnerError> {
    fn visit(
        variable: usize,
        remaining: usize,
        exponents: &mut [u32],
        output: &mut Vec<Vec<u32>>,
        maximum_count: usize,
    ) -> Result<(), GroebnerError> {
        if variable == exponents.len() {
            if output.len() == maximum_count {
                return Err(GroebnerError::ResourceLimit {
                    resource: "external certificate monomials",
                    limit: maximum_count,
                });
            }
            output.push(exponents.to_vec());
            return Ok(());
        }
        for exponent in 0..=remaining {
            exponents[variable] = u32::try_from(exponent)
                .map_err(|_| external_error("Noether", "a Macaulay degree overflowed"))?;
            visit(
                variable + 1,
                remaining - exponent,
                exponents,
                output,
                maximum_count,
            )?;
        }
        Ok(())
    }

    let mut output = Vec::new();
    visit(0, degree, &mut vec![0; arity], &mut output, maximum_count)?;
    Ok(output)
}

fn backend_variables(arity: usize) -> Vec<String> {
    (0..arity).map(|index| format!("x{index}")).collect()
}

#[cfg(any(
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
fn format_polynomial<R: Field>(
    polynomial: &Polynomial<R>,
    context: &ExternalCoefficientContext,
    variables: &[String],
    backend: &'static str,
) -> Result<String, GroebnerError> {
    if polynomial.is_zero() {
        return Ok("0".to_owned());
    }
    polynomial
        .terms()
        .map(|(exponents, coefficient)| {
            let coefficient = context.format_element::<R>(coefficient)?;
            let mut factors = vec![format!("({coefficient})")];
            for (variable, exponent) in variables.iter().zip(exponents) {
                if exponent.is_zero() {
                    continue;
                }
                if exponent == &Integer::one() {
                    factors.push(variable.clone());
                } else {
                    factors.push(format!("{variable}^{exponent}"));
                }
            }
            if factors.is_empty() {
                return Err(external_error(backend, "a polynomial term has no factors"));
            }
            Ok(factors.join("*"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|terms| terms.join("+"))
}

#[cfg(any(
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
fn parse_certified_basis<R: Field>(
    ring: &PolynomialRing<R>,
    generator_count: usize,
    context: &ExternalCoefficientContext,
    backend: &'static str,
    output: &str,
) -> Result<CertifiedBasis<R>, GroebnerError> {
    let lines = output.lines().map(str::trim).collect::<Vec<_>>();
    let starts = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| (*line == PROTOCOL_START).then_some(index))
        .collect::<Vec<_>>();
    if starts.len() != 1 {
        return Err(external_error(
            backend,
            "output must contain exactly one protocol start marker",
        ));
    }
    let start = starts[0];
    let end = lines
        .iter()
        .enumerate()
        .skip(start + 1)
        .find_map(|(index, line)| (*line == PROTOCOL_END).then_some(index))
        .ok_or_else(|| external_error(backend, "truncated output has no protocol end marker"))?;
    if lines[end + 1..].iter().any(|line| !line.is_empty()) {
        return Err(external_error(
            backend,
            "nonempty output follows the protocol end marker",
        ));
    }
    let body = &lines[start + 1..end];
    if body.len() < 5 {
        return Err(external_error(backend, "truncated protocol header"));
    }
    require_header(body[0], "context", &context.label(), backend)?;
    let variables = backend_variables(ring.variables().len());
    require_header(body[1], "variables", &variables.join(","), backend)?;
    require_header(body[2], "order", ring.term_order().name(), backend)?;
    require_header(body[3], "generators", &generator_count.to_string(), backend)?;
    let basis_count = parse_count(body[4], "basis", backend)?;
    if basis_count > MAX_EXTERNAL_BASIS_SIZE {
        return Err(external_error(
            backend,
            "basis count exceeds the accepted boundary",
        ));
    }
    let certificate_cells = generator_count.checked_mul(basis_count).ok_or_else(|| {
        external_error(
            backend,
            "certificate matrix dimensions overflow address space",
        )
    })?;
    if certificate_cells > MAX_EXTERNAL_CERTIFICATE_CELLS {
        return Err(external_error(
            backend,
            "certificate matrix exceeds the accepted boundary",
        ));
    }
    let expected_lines = 5_usize
        .checked_add(basis_count)
        .and_then(|value| value.checked_add(certificate_cells))
        .ok_or_else(|| external_error(backend, "protocol line count overflows address space"))?;
    if body.len() != expected_lines {
        return Err(external_error(
            backend,
            format!(
                "protocol has {} body lines, expected {expected_lines}",
                body.len()
            ),
        ));
    }
    let basis = body[5..5 + basis_count]
        .iter()
        .map(|line| {
            line.strip_prefix("poly=")
                .ok_or_else(|| external_error(backend, "expected a polynomial record"))
                .and_then(|value| parse_polynomial(ring, context, &variables, backend, value))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut certificates = (0..generator_count)
        .map(|_| Vec::with_capacity(basis_count))
        .collect::<Vec<_>>();
    for (cell, line) in body[5 + basis_count..].iter().enumerate() {
        let row = cell / basis_count;
        let column = cell % basis_count;
        let prefix = format!("cert={row},{column}:");
        let value = line
            .strip_prefix(&prefix)
            .ok_or_else(|| external_error(backend, "certificate cells are missing or reordered"))?;
        certificates[row].push(parse_polynomial(ring, context, &variables, backend, value)?);
    }
    Ok(CertifiedBasis {
        backend,
        basis,
        certificates,
    })
}

#[cfg(any(
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
fn require_header(
    line: &str,
    key: &str,
    expected: &str,
    backend: &'static str,
) -> Result<(), GroebnerError> {
    if line == format!("{key}={expected}") {
        Ok(())
    } else {
        Err(external_error(
            backend,
            format!("wrong or missing {key} header"),
        ))
    }
}

#[cfg(any(
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
fn parse_count(line: &str, key: &str, backend: &'static str) -> Result<usize, GroebnerError> {
    let value = line
        .strip_prefix(&format!("{key}="))
        .ok_or_else(|| external_error(backend, format!("missing {key} count")))?;
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(external_error(backend, format!("invalid {key} count")));
    }
    value
        .parse()
        .map_err(|_| external_error(backend, format!("{key} count does not fit usize")))
}

fn parse_polynomial<R: Field>(
    ring: &PolynomialRing<R>,
    context: &ExternalCoefficientContext,
    variables: &[String],
    backend: &'static str,
    value: &str,
) -> Result<Polynomial<R>, GroebnerError> {
    if !value.is_ascii() {
        return Err(external_error(backend, "polynomial output is not ASCII"));
    }
    let compact = value
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .map(char::from)
        .collect::<String>();
    if compact.is_empty() {
        return Err(external_error(backend, "empty polynomial expression"));
    }
    let terms = split_terms(&compact, backend)?;
    let mut parsed = Vec::with_capacity(terms.len());
    for (negative, term) in terms {
        let factors = split_factors(term, backend)?;
        let mut numerator = if negative {
            -Integer::one()
        } else {
            Integer::one()
        };
        let mut denominator = Integer::one();
        let mut exponents = vec![Integer::zero(); variables.len()];
        let mut saw_factor = false;
        for factor in factors {
            saw_factor = true;
            let factor = strip_balanced_parentheses(factor, backend)?;
            if let Some((variable, exponent)) = parse_variable_factor(factor, variables, backend)? {
                exponents[variable] = &exponents[variable] + &exponent;
                continue;
            }
            let (factor_numerator, factor_denominator) = parse_rational_factor(factor, backend)?;
            numerator = &numerator * &factor_numerator;
            denominator = &denominator * &factor_denominator;
        }
        if !saw_factor {
            return Err(external_error(backend, "empty polynomial term"));
        }
        let coefficient = context.parse_element(ring, numerator, denominator, backend)?;
        parsed.push((exponents, coefficient));
    }
    ring.from_terms(parsed)
        .map_err(|error| external_error(backend, error.to_string()))
}

fn split_terms<'a>(
    value: &'a str,
    backend: &'static str,
) -> Result<Vec<(bool, &'a str)>, GroebnerError> {
    let bytes = value.as_bytes();
    let mut output = Vec::new();
    let mut start = 0;
    let mut depth = 0_usize;
    let mut negative = false;
    if matches!(bytes.first(), Some(b'+') | Some(b'-')) {
        negative = bytes[0] == b'-';
        start = 1;
    }
    let mut index = start;
    while index < bytes.len() {
        match bytes[index] {
            b'(' => depth = depth.saturating_add(1),
            b')' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| external_error(backend, "unbalanced polynomial parentheses"))?;
            }
            b'+' | b'-' if depth == 0 => {
                if index == start {
                    return Err(external_error(backend, "empty polynomial term"));
                }
                output.push((negative, &value[start..index]));
                negative = bytes[index] == b'-';
                start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    if depth != 0 {
        return Err(external_error(backend, "unbalanced polynomial parentheses"));
    }
    if start == bytes.len() {
        return Err(external_error(backend, "empty final polynomial term"));
    }
    output.push((negative, &value[start..]));
    Ok(output)
}

fn split_factors<'a>(value: &'a str, backend: &'static str) -> Result<Vec<&'a str>, GroebnerError> {
    let bytes = value.as_bytes();
    let mut factors = Vec::new();
    let mut start = 0;
    let mut depth = 0_usize;
    for (index, byte) in bytes.iter().copied().enumerate() {
        match byte {
            b'(' => depth = depth.saturating_add(1),
            b')' => {
                depth = depth
                    .checked_sub(1)
                    .ok_or_else(|| external_error(backend, "unbalanced factor parentheses"))?;
            }
            b'*' if depth == 0 => {
                if index == start {
                    return Err(external_error(backend, "empty multiplicative factor"));
                }
                factors.push(&value[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 || start == bytes.len() {
        return Err(external_error(backend, "malformed multiplicative factors"));
    }
    factors.push(&value[start..]);
    Ok(factors)
}

fn strip_balanced_parentheses<'a>(
    mut value: &'a str,
    backend: &'static str,
) -> Result<&'a str, GroebnerError> {
    while value.starts_with('(') && value.ends_with(')') {
        let mut depth = 0_i64;
        let mut encloses_all = true;
        for (index, byte) in value.bytes().enumerate() {
            match byte {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            if depth < 0 {
                return Err(external_error(backend, "unbalanced factor parentheses"));
            }
            if depth == 0 && index + 1 != value.len() {
                encloses_all = false;
                break;
            }
        }
        if depth != 0 {
            return Err(external_error(backend, "unbalanced factor parentheses"));
        }
        if !encloses_all {
            break;
        }
        value = &value[1..value.len() - 1];
    }
    if value.is_empty() {
        Err(external_error(backend, "empty parenthesized factor"))
    } else {
        Ok(value)
    }
}

fn parse_variable_factor(
    factor: &str,
    variables: &[String],
    backend: &'static str,
) -> Result<Option<(usize, Integer)>, GroebnerError> {
    for (index, variable) in variables.iter().enumerate() {
        if factor == variable {
            return Ok(Some((index, Integer::one())));
        }
        if let Some(exponent) = factor.strip_prefix(&format!("{variable}^")) {
            if exponent.is_empty() || !exponent.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(external_error(backend, "invalid monomial exponent"));
            }
            let exponent = exponent
                .parse::<Integer>()
                .map_err(|_| external_error(backend, "monomial exponent is not an integer"))?;
            return Ok(Some((index, exponent)));
        }
    }
    if factor.bytes().any(|byte| byte.is_ascii_alphabetic()) {
        return Err(external_error(backend, "unknown polynomial variable"));
    }
    Ok(None)
}

fn parse_rational_factor(
    factor: &str,
    backend: &'static str,
) -> Result<(Integer, Integer), GroebnerError> {
    let mut pieces = factor.split('/');
    let numerator = pieces
        .next()
        .ok_or_else(|| external_error(backend, "missing coefficient numerator"))?;
    let denominator = pieces.next().unwrap_or("1");
    if pieces.next().is_some()
        || numerator.is_empty()
        || denominator.is_empty()
        || !integer_literal(numerator)
        || !unsigned_integer_literal(denominator)
    {
        return Err(external_error(backend, "invalid rational coefficient"));
    }
    let numerator = numerator
        .parse::<Integer>()
        .map_err(|_| external_error(backend, "coefficient numerator is not an integer"))?;
    let denominator = denominator
        .parse::<Integer>()
        .map_err(|_| external_error(backend, "coefficient denominator is not an integer"))?;
    Ok((numerator, denominator))
}

fn integer_literal(value: &str) -> bool {
    let value = value
        .strip_prefix('-')
        .or_else(|| value.strip_prefix('+'))
        .unwrap_or(value);
    unsigned_integer_literal(value)
}

fn unsigned_integer_literal(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
}

fn verify_certified_basis<R: Field>(
    ring: &PolynomialRing<R>,
    generators: &[Polynomial<R>],
    candidate: CertifiedBasis<R>,
) -> Result<Vec<Polynomial<R>>, GroebnerError> {
    if candidate.certificates.len() != generators.len()
        || candidate
            .certificates
            .iter()
            .any(|row| row.len() != candidate.basis.len())
    {
        return Err(external_error(
            candidate.backend,
            "certificate matrix has inconsistent dimensions",
        ));
    }
    for (column, basis_element) in candidate.basis.iter().enumerate() {
        let represented = generators.iter().zip(&candidate.certificates).try_fold(
            ring.zero(),
            |sum, (generator, row)| {
                let term = ring.mul(generator, &row[column])?;
                ring.add(&sum, &term)
            },
        )?;
        if &represented != basis_element {
            return Err(GroebnerError::ExternalCandidate {
                backend: candidate.backend,
                reason: format!(
                    "certificate column {column} does not reconstruct its basis element"
                ),
            });
        }
    }
    let mut reduction_steps = 0;
    let reduced = interreduce(ring, candidate.basis, &mut reduction_steps)?;
    verify_reduced_basis(ring, &reduced)?;
    for generator in generators {
        if !normal_form_against(ring, generator, &reduced, &mut reduction_steps)?.is_zero() {
            return Err(GroebnerError::ExternalCandidate {
                backend: candidate.backend,
                reason: "an input generator is outside the candidate ideal".to_owned(),
            });
        }
        if reduction_steps > MAX_EXTERNAL_REDUCTION_STEPS {
            return Err(GroebnerError::ResourceLimit {
                resource: "external candidate verification steps",
                limit: MAX_EXTERNAL_REDUCTION_STEPS,
            });
        }
    }
    Ok(reduced)
}

fn external_error(backend: &'static str, reason: impl Into<String>) -> GroebnerError {
    GroebnerError::ExternalCandidate {
        backend,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;

    fn rational_case() -> (
        PolynomialRing<RationalField>,
        Vec<Polynomial<RationalField>>,
    ) {
        let ring = PolynomialRing::new(
            RationalField,
            ["u", "v"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let first = ring.sub(&ring.mul(&x, &x).unwrap(), &y).unwrap();
        let second = ring.sub(&ring.mul(&x, &y).unwrap(), &ring.one()).unwrap();
        (ring, vec![first, second])
    }

    fn certified_output() -> &'static str {
        "diagnostic prelude\n\
         NOETHER_GROEBNER_V1\n\
         context=Q\n\
         variables=x0,x1\n\
         order=grevlex\n\
         generators=2\n\
         basis=3\n\
         poly=x1^2-x0\n\
         poly=x0*x1-1\n\
         poly=x0^2-x1\n\
         cert=0,0:-x1\n\
         cert=0,1:0\n\
         cert=0,2:1\n\
         cert=1,0:x0\n\
         cert=1,1:1\n\
         cert=1,2:0\n\
         END_NOETHER_GROEBNER_V1\n"
    }

    fn prime_case() -> (PolynomialRing<FiniteField>, Vec<Polynomial<FiniteField>>) {
        let ring = PolynomialRing::new(
            FiniteField::prime(101_u64).unwrap(),
            ["u", "v"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let first = ring.sub(&ring.mul(&x, &x).unwrap(), &y).unwrap();
        let second = ring.sub(&ring.mul(&x, &y).unwrap(), &ring.one()).unwrap();
        (ring, vec![first, second])
    }

    #[test]
    fn certified_protocol_proves_both_ideal_containments() {
        let (ring, generators) = rational_case();
        let context = ExternalCoefficientContext::detect(&ring).unwrap();
        let parsed = parse_certified_basis(
            &ring,
            generators.len(),
            &context,
            "test",
            certified_output(),
        )
        .unwrap();
        let imported = verify_certified_basis(&ring, &generators, parsed).unwrap();
        assert_eq!(imported, ring.ideal(generators).unwrap().generators());
    }

    #[test]
    fn reconstructed_macaulay_certificates_prove_both_containments() {
        let (ring, generators) = rational_case();
        let basis = ring
            .ideal(generators.clone())
            .unwrap()
            .generators()
            .to_vec();
        let candidate = reconstruct_basis_certificates(&ring, &generators, basis, "test").unwrap();
        assert_eq!(
            verify_certified_basis(&ring, &generators, candidate).unwrap(),
            ring.ideal(generators).unwrap().generators()
        );

        let (ring, generators) = prime_case();
        let basis = ring
            .ideal(generators.clone())
            .unwrap()
            .generators()
            .to_vec();
        let candidate = reconstruct_basis_certificates(&ring, &generators, basis, "test").unwrap();
        assert_eq!(
            verify_certified_basis(&ring, &generators, candidate).unwrap(),
            ring.ideal(generators).unwrap().generators()
        );
    }

    #[test]
    fn msolve_basis_parser_is_strict_about_context_and_shape() {
        let (ring, _) = prime_case();
        let context = ExternalCoefficientContext::detect(&ring).unwrap();
        let output = "#Reduced Groebner basis for input in characteristic 101\n\
                      #for variable order x0, x1\n\
                      #w.r.t. grevlex monomial ordering\n\
                      #consisting of 3 elements:\n\
                      [1*x1^2+100*x0^1,\n\
                      1*x0^1*x1^1+100,\n\
                      1*x0^2+100*x1^1]:\n";
        let parsed = parse_msolve_basis(&ring, &context, output).unwrap();
        assert_eq!(parsed.len(), 3);
        for malformed in [
            output.replace("characteristic 101", "characteristic 103"),
            output.replace("variable order x0, x1", "variable order x1, x0"),
            output.replace("3 elements", "2 elements"),
            output.replace("]:", "]"),
            format!("{output}trailing"),
        ] {
            assert!(parse_msolve_basis(&ring, &context, &malformed).is_err());
        }
    }

    #[test]
    fn msolve_rur_parser_normalizes_and_verifies_rational_coordinates() {
        let (ring, generators) = rational_case();
        let ideal = ring.ideal(generators).unwrap();
        let context = ExternalCoefficientContext::detect(&ring).unwrap();
        let output = "[0, [0,\n\
            2,\n\
            3,\n\
            ['x0', 'x1'],\n\
            [0, 1],\n\
            [1,\n\
            [[3, [-1, 0, 0, 1]],\n\
            [2, [0, 0, 3]],\n\
            [\n\
            [[2, [0, -3, 0]],\n\
            1]\n\
            ]]]]]:";
        let representation = parse_msolve_rur(&ideal, &context, output).unwrap();
        representation.verify().unwrap();
        assert_eq!(representation.quotient_dimension(), 3);
        assert_eq!(
            representation.denominator(),
            &representation.parameter_ring().one()
        );
        let t = representation.parameter_ring().generator(0).unwrap();
        assert_eq!(representation.coordinate_numerators()[0], t.pow(2).unwrap());
        assert_eq!(representation.coordinate_numerators()[1], t);

        for malformed in [
            output.replace("[0, 1]", "[1, 0]"),
            output.replace("['x0', 'x1']", "['x1', 'x0']"),
            output.replace("[2, [0, 0, 3]]", "[2, [0, 0, 0]]"),
            output.replace("]:", "]"),
            format!("{output}junk"),
        ] {
            assert!(parse_msolve_rur(&ideal, &context, &malformed).is_err());
        }
    }

    #[test]
    fn msolve_rur_parser_normalizes_and_verifies_prime_coordinates() {
        let (ring, generators) = prime_case();
        let ideal = ring.ideal(generators).unwrap();
        let context = ExternalCoefficientContext::detect(&ring).unwrap();
        let output = "[0, [101,\n\
            2,\n\
            3,\n\
            ['x0', 'x1'],\n\
            [0, 1],\n\
            [1,\n\
            [[3, [100, 0, 0, 1]],\n\
            [0, [1]],\n\
            [\n\
            [[2, [0, 0, 100]]]\n\
            ]]]]]:";
        let representation = parse_msolve_rur(&ideal, &context, output).unwrap();
        representation.verify().unwrap();
        assert_eq!(representation.quotient_dimension(), 3);
        let t = representation.parameter_ring().generator(0).unwrap();
        assert_eq!(representation.coordinate_numerators()[0], t.pow(2).unwrap());
        assert_eq!(representation.coordinate_numerators()[1], t);
    }

    #[test]
    fn protocol_rejects_context_truncation_reordering_and_false_certificates() {
        let (ring, generators) = rational_case();
        let context = ExternalCoefficientContext::detect(&ring).unwrap();
        for malformed in [
            certified_output().replace("context=Q", "context=Fp:7"),
            certified_output().replace("END_NOETHER_GROEBNER_V1", ""),
            certified_output().replace("cert=0,1:0", "cert=1,0:0"),
        ] {
            assert!(
                parse_certified_basis(&ring, generators.len(), &context, "test", &malformed,)
                    .is_err()
            );
        }
        let false_certificate = certified_output().replace("cert=0,0:-x1", "cert=0,0:0");
        let parsed = parse_certified_basis(
            &ring,
            generators.len(),
            &context,
            "test",
            &false_certificate,
        )
        .unwrap();
        assert!(verify_certified_basis(&ring, &generators, parsed).is_err());
    }

    #[test]
    fn polynomial_parser_accepts_prime_field_coefficients_and_rejects_grammar_escape() {
        let field = FiniteField::prime(101_u64).unwrap();
        let ring =
            PolynomialRing::new(field, ["u", "v"], MonomialOrder::GradedReverseLexicographic)
                .unwrap();
        let context = ExternalCoefficientContext::detect(&ring).unwrap();
        let variables = backend_variables(2);
        let parsed = parse_polynomial(&ring, &context, &variables, "test", "x1^2+100*x0").unwrap();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        assert_eq!(parsed, ring.sub(&ring.mul(&y, &y).unwrap(), &x).unwrap());
        for malformed in ["x2", "x0**x1", "x0;quit", "x0^(2)", "1/0*x0"] {
            assert!(parse_polynomial(&ring, &context, &variables, "test", malformed).is_err());
        }
    }

    #[test]
    fn rational_source_format_round_trips_fractional_coefficients() {
        let ring = PolynomialRing::new(
            RationalField,
            ["u"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let half = Rational::new(Integer::one(), Integer::from(2)).unwrap();
        let value = ring.from_terms([(vec![Integer::from(3)], half)]).unwrap();
        let context = ExternalCoefficientContext::detect(&ring).unwrap();
        let variables = backend_variables(1);
        let encoded = format_polynomial(&value, &context, &variables, "test").unwrap();
        assert_eq!(
            parse_polynomial(&ring, &context, &variables, "test", &encoded).unwrap(),
            value
        );
    }

    #[cfg(feature = "macaulay2-subprocess")]
    #[test]
    fn macaulay2_live_candidate_is_certified_when_available() {
        let (ring, generators) = rational_case();
        let Some(result) = try_macaulay2(&ring, &generators) else {
            eprintln!("skipping: Macaulay2 unavailable");
            return;
        };
        assert_eq!(
            result.unwrap(),
            ring.ideal(generators).unwrap().generators()
        );

        let (ring, generators) = prime_case();
        let Some(result) = try_macaulay2(&ring, &generators) else {
            eprintln!("skipping: Macaulay2 unavailable");
            return;
        };
        assert_eq!(
            result.unwrap(),
            ring.ideal(generators).unwrap().generators()
        );
    }

    #[cfg(feature = "singular-subprocess")]
    #[test]
    fn singular_live_candidate_is_certified_when_available() {
        let (ring, generators) = rational_case();
        let Some(result) = try_singular(&ring, &generators) else {
            eprintln!("skipping: Singular unavailable");
            return;
        };
        assert_eq!(
            result.unwrap(),
            ring.ideal(generators).unwrap().generators()
        );

        let (ring, generators) = prime_case();
        let Some(result) = try_singular(&ring, &generators) else {
            eprintln!("skipping: Singular unavailable");
            return;
        };
        assert_eq!(
            result.unwrap(),
            ring.ideal(generators).unwrap().generators()
        );
    }

    #[cfg(feature = "msolve-subprocess")]
    #[test]
    fn msolve_live_prime_candidate_is_certified_when_available() {
        let (ring, generators) = prime_case();
        let Some(result) = try_msolve(&ring, &generators) else {
            eprintln!("skipping: msolve unavailable");
            return;
        };
        assert_eq!(
            result.unwrap(),
            ring.ideal(generators).unwrap().generators()
        );
    }

    #[cfg(feature = "msolve-subprocess")]
    #[test]
    fn msolve_live_rur_candidates_are_verified_when_available() {
        let ring = PolynomialRing::new(
            RationalField,
            ["u", "v"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let two = ring.add(&ring.one(), &ring.one()).unwrap();
        let ideal = ring
            .ideal([
                ring.sub(&x, &ring.one()).unwrap(),
                ring.sub(&y, &two).unwrap(),
            ])
            .unwrap();
        let Some(result) = try_msolve_rur(&ideal) else {
            eprintln!("skipping: msolve unavailable");
            return;
        };
        let representation = result.unwrap();
        assert_eq!(representation.quotient_dimension(), 1);
        representation.verify().unwrap();

        let ring = PolynomialRing::new(
            FiniteField::prime(101_u64).unwrap(),
            ["u", "v"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let two = ring.add(&ring.one(), &ring.one()).unwrap();
        let ideal = ring
            .ideal([
                ring.sub(&x, &ring.one()).unwrap(),
                ring.sub(&y, &two).unwrap(),
            ])
            .unwrap();
        let Some(result) = try_msolve_rur(&ideal) else {
            eprintln!("skipping: msolve unavailable");
            return;
        };
        let representation = result.unwrap();
        assert_eq!(representation.quotient_dimension(), 1);
        representation.verify().unwrap();
    }
}
