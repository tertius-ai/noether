// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Bounded, natively verified decompositions of exact polynomial ideals.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::linalg::LinalgError;
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{
    ContextMismatchError, Field, FiniteContext, Integer, NoetherError, Ring, SemanticType,
    TypeDescription,
};

use super::{
    GroebnerError, Ideal, MonomialIdealError, Polynomial, PolynomialContextError, PolynomialError,
    PolynomialFactorizationError, PolynomialRing, RationalFunction, RationalFunctionError,
    RationalFunctionField,
};

const MAX_SPLITTING_FIELD_ORDER: usize = 4_096;
const MAX_BINARY_FORM_DEGREE: usize = 8;

static IDEAL_DECOMPOSITION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/IdealDecomposition@1",
    rust_path: "noether::commalg::IdealDecomposition",
    python_path: Some("noether.commalg.IdealDecomposition"),
};

/// Failure to construct or verify a bounded exact ideal decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionError {
    /// A value belonged to a different exact context.
    ContextMismatch(ContextMismatchError),
    /// Polynomial-ring construction failed.
    PolynomialContext(PolynomialContextError),
    /// Exact polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Rational-function arithmetic failed.
    RationalFunction(RationalFunctionError),
    /// Canonical ideal arithmetic failed.
    Groebner(GroebnerError),
    /// Canonical monomial-ideal arithmetic failed.
    MonomialIdeal(MonomialIdealError),
    /// Exact univariate factorization failed.
    Factorization(PolynomialFactorizationError),
    /// Exact multiplication-operator linear algebra failed.
    Linalg(LinalgError),
    /// The requested exact case is outside the deliberately small native floor.
    Unsupported(String),
    /// A fixed native work bound was exceeded.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed limit.
        limit: usize,
    },
    /// Supplied or computed decomposition evidence failed verification.
    Invalid(String),
}

impl fmt::Display for DecompositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::RationalFunction(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::MonomialIdeal(error) => error.fmt(formatter),
            Self::Factorization(error) => error.fmt(formatter),
            Self::Linalg(error) => error.fmt(formatter),
            Self::Unsupported(reason) => {
                write!(
                    formatter,
                    "unsupported native ideal decomposition: {reason}"
                )
            }
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "native ideal decomposition exceeded its {resource} limit of {limit}"
            ),
            Self::Invalid(reason) => write!(formatter, "invalid ideal decomposition: {reason}"),
        }
    }
}

impl std::error::Error for DecompositionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::RationalFunction(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::MonomialIdeal(error) => Some(error),
            Self::Factorization(error) => Some(error),
            Self::Linalg(error) => Some(error),
            Self::Unsupported(_) | Self::ResourceLimit { .. } | Self::Invalid(_) => None,
        }
    }
}

impl From<ContextMismatchError> for DecompositionError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

impl From<PolynomialContextError> for DecompositionError {
    fn from(error: PolynomialContextError) -> Self {
        Self::PolynomialContext(error)
    }
}

impl From<PolynomialError> for DecompositionError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<RationalFunctionError> for DecompositionError {
    fn from(error: RationalFunctionError) -> Self {
        Self::RationalFunction(error)
    }
}

impl From<GroebnerError> for DecompositionError {
    fn from(error: GroebnerError) -> Self {
        Self::Groebner(error)
    }
}

impl From<MonomialIdealError> for DecompositionError {
    fn from(error: MonomialIdealError) -> Self {
        Self::MonomialIdeal(error)
    }
}

impl From<PolynomialFactorizationError> for DecompositionError {
    fn from(error: PolynomialFactorizationError) -> Self {
        Self::Factorization(error)
    }
}

impl From<LinalgError> for DecompositionError {
    fn from(error: LinalgError) -> Self {
        Self::Linalg(error)
    }
}

/// A verified finite family of top-dimensional minimal components.
///
/// The source and every component retain one exact polynomial-ring context.
/// When present, `frobenius_permutation[i]` is the component obtained by
/// applying coefficient Frobenius to component `i`.
#[derive(Clone, PartialEq, Eq)]
pub struct IdealDecomposition<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    components: Vec<Ideal<PolynomialRing<R>>>,
    frobenius_permutation: Option<Vec<usize>>,
}

impl<R: Field> IdealDecomposition<R> {
    /// The radical ideal reconstructed by intersecting the components.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Canonically ordered top-dimensional minimal components.
    pub fn components(&self) -> &[Ideal<PolynomialRing<R>>] {
        &self.components
    }

    /// Verified coefficient-Frobenius action, when the base carries one.
    pub fn frobenius_permutation(&self) -> Option<&[usize]> {
        self.frobenius_permutation.as_deref()
    }

    /// Recheck component contexts, dimensions, intersection, and permutation shape.
    pub fn verify(&self) -> Result<(), DecompositionError> {
        if self.components.is_empty() {
            return Err(DecompositionError::Invalid(
                "a decomposition must contain at least one component".to_owned(),
            ));
        }
        self.source.verify()?;
        let source_dimension = self.source.dimension()?;
        for component in &self.components {
            if component.ring() != self.source.ring() {
                return Err(DecompositionError::Invalid(
                    "a component has a different polynomial-ring context".to_owned(),
                ));
            }
            component.verify()?;
            if component.dimension()? != source_dimension {
                return Err(DecompositionError::Invalid(
                    "a component is not top-dimensional".to_owned(),
                ));
            }
        }
        let mut reconstructed = self.components[0].clone();
        for component in &self.components[1..] {
            reconstructed = reconstructed.intersection(component)?;
        }
        if reconstructed != self.source {
            return Err(DecompositionError::Invalid(
                "the component intersection does not reconstruct the source ideal".to_owned(),
            ));
        }
        if let Some(permutation) = &self.frobenius_permutation {
            if permutation.len() != self.components.len() {
                return Err(DecompositionError::Invalid(
                    "the Frobenius permutation has the wrong size".to_owned(),
                ));
            }
            let mut seen = vec![false; permutation.len()];
            for &image in permutation {
                if image >= permutation.len() || seen[image] {
                    return Err(DecompositionError::Invalid(
                        "the Frobenius action is not a permutation".to_owned(),
                    ));
                }
                seen[image] = true;
            }
        }
        Ok(())
    }
}

impl<R: Field> fmt::Debug for IdealDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IdealDecomposition")
            .field("components", &self.components.len())
            .field("has_frobenius", &self.frobenius_permutation.is_some())
            .finish()
    }
}

impl<R: Field> SemanticType for IdealDecomposition<R> {
    const TYPE_ID: &'static str = "noether/commalg/IdealDecomposition@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.source.canonical_bytes()?)?;
        for component in &self.components {
            append_component(&mut output, &component.canonical_bytes()?)?;
        }
        if let Some(permutation) = &self.frobenius_permutation {
            for image in permutation {
                let image = u64::try_from(*image).map_err(|_| {
                    NoetherError::Serialization(
                        "component index does not fit canonical framing".to_owned(),
                    )
                })?;
                output.extend_from_slice(&image.to_be_bytes());
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &IDEAL_DECOMPOSITION_DESCRIPTION
    }
}

type FiniteFunctionField = RationalFunctionField<FiniteField>;
type FiniteFunctionIdeal = Ideal<PolynomialRing<FiniteFunctionField>>;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteFunctionDecompositionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: FiniteFunctionIdeal,
    components: Vec<FiniteFunctionIdeal>,
    frobenius_permutation: Vec<usize>,
}

impl IdealDecomposition<FiniteFunctionField> {
    /// Recompute the absolute coefficient-Frobenius image of every component.
    pub fn verify_frobenius(&self) -> Result<(), DecompositionError> {
        self.verify()?;
        let permutation = self.frobenius_permutation.as_ref().ok_or_else(|| {
            DecompositionError::Invalid("the decomposition has no Frobenius action".to_owned())
        })?;
        for (index, component) in self.components.iter().enumerate() {
            let image = frobenius_ideal(component)?;
            if self.components.get(permutation[index]) != Some(&image) {
                return Err(DecompositionError::Invalid(format!(
                    "Frobenius image of component {index} does not match the recorded permutation"
                )));
            }
        }
        Ok(())
    }

    /// Serialize the source, components, and verified Frobenius permutation.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&FiniteFunctionDecompositionWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            components: self.components.clone(),
            frobenius_permutation: self.frobenius_permutation.clone().unwrap_or_default(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reverify the component intersection and Frobenius action.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<FiniteFunctionDecompositionWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            source: wire.source,
            components: wire.components,
            frobenius_permutation: Some(wire.frobenius_permutation),
        };
        result
            .verify_frobenius()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl FiniteFunctionIdeal {
    /// Change only the finite constant field of `F_q(c)[x_1,...,x_n]`.
    ///
    /// The native floor accepts identity maps and the canonical prime-subfield
    /// embedding `F_p -> F_{p^e}`. General finite-field embeddings require an
    /// explicit future embedding object and fail closed here.
    pub fn base_change_constant_field(
        &self,
        target: FiniteField,
    ) -> Result<Self, DecompositionError> {
        let source_functions = self.ring().coefficient_ring();
        let source = source_functions.coefficient_ring();
        require_constant_embedding(source, &target)?;
        let target_functions =
            RationalFunctionField::new(target, source_functions.variable().to_owned())?;
        let target_ring = PolynomialRing::with_grading(
            target_functions.clone(),
            self.ring().variables().to_vec(),
            self.ring().term_order(),
            self.ring().grading().clone(),
        )?;
        let generators = self
            .generators()
            .iter()
            .map(|generator| map_function_polynomial(generator, &target_ring, &target_functions))
            .collect::<Result<Vec<_>, _>>()?;
        target_ring.ideal(generators).map_err(Into::into)
    }

    /// Split a square-free homogeneous binary principal ideal over its constants.
    ///
    /// This bounded native case is sufficient for small geometric-component
    /// checks such as a conjugate pair of lines over `F_9(c)`. Coefficients must
    /// be constant in `c`, the form must split completely into distinct linear
    /// factors, and the field/order and degree caps are fixed.
    pub fn minimal_primes(
        &self,
    ) -> Result<IdealDecomposition<FiniteFunctionField>, DecompositionError> {
        self.verify()?;
        if self.ring().variables().len() != 2 || self.generators().len() != 1 {
            return Err(DecompositionError::Unsupported(
                "the native floor requires a principal ideal in a binary polynomial ring"
                    .to_owned(),
            ));
        }
        let generator = &self.generators()[0];
        let degree = homogeneous_binary_coefficients(generator)?;
        if degree == 0 || degree > MAX_BINARY_FORM_DEGREE {
            return Err(DecompositionError::ResourceLimit {
                resource: "binary-form degree",
                limit: MAX_BINARY_FORM_DEGREE,
            });
        }
        let base = self.ring().coefficient_ring().coefficient_ring();
        let field_order = bounded_field_order(base)?;
        if field_order > MAX_SPLITTING_FIELD_ORDER {
            return Err(DecompositionError::ResourceLimit {
                resource: "splitting-field order",
                limit: MAX_SPLITTING_FIELD_ORDER,
            });
        }
        let coefficients = binary_form_coefficients(generator, degree)?;
        let mut roots = Vec::new();
        for value in base.elements() {
            if evaluate_univariate(base, &coefficients, &value)?.is_zero() {
                roots.push(value);
            }
        }
        let highest_x_degree = coefficients
            .iter()
            .rposition(|coefficient| !coefficient.is_zero())
            .ok_or_else(|| {
                DecompositionError::Invalid("the source generator is zero".to_owned())
            })?;
        let infinity_roots = degree.saturating_sub(highest_x_degree);
        if infinity_roots > 1 || roots.len() + infinity_roots != degree {
            return Err(DecompositionError::Unsupported(
                "the binary form must split completely into distinct linear factors over the current constant field"
                    .to_owned(),
            ));
        }
        let variables = self.ring().generators()?;
        let (x, y) = (&variables[0], &variables[1]);
        let functions = self.ring().coefficient_ring();
        let mut components = Vec::with_capacity(degree);
        for root in roots {
            let root = functions.constant(root)?;
            let equation = (x - &y.scalar_mul(&root)?)?;
            components.push(self.ring().ideal([equation])?);
        }
        if infinity_roots == 1 {
            components.push(self.ring().ideal([y.clone()])?);
        }
        let mut keyed_components = components
            .into_iter()
            .map(|component| {
                component
                    .canonical_bytes()
                    .map(|bytes| (bytes, component))
                    .map_err(|error| DecompositionError::Invalid(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        keyed_components.sort_by(|left, right| left.0.cmp(&right.0));
        let components = keyed_components
            .into_iter()
            .map(|(_, component)| component)
            .collect::<Vec<_>>();
        let permutation = components
            .iter()
            .enumerate()
            .map(|(index, component)| {
                let image = frobenius_ideal(component)?;
                components
                    .iter()
                    .position(|candidate| candidate == &image)
                    .ok_or_else(|| {
                        DecompositionError::Invalid(format!(
                            "Frobenius image of component {index} is absent"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result = IdealDecomposition {
            source: self.clone(),
            components,
            frobenius_permutation: Some(permutation),
        };
        result.verify_frobenius()?;
        Ok(result)
    }
}

fn require_constant_embedding(
    source: &FiniteField,
    target: &FiniteField,
) -> Result<(), DecompositionError> {
    if source == target {
        return Ok(());
    }
    if source.degree() == 1 && source.characteristic() == target.characteristic() {
        return Ok(());
    }
    Err(DecompositionError::Unsupported(
        "constant-field base change currently accepts identity maps or a prime-subfield embedding"
            .to_owned(),
    ))
}

fn map_finite_element(
    value: &FiniteFieldElement,
    target: &FiniteField,
) -> Result<FiniteFieldElement, DecompositionError> {
    require_constant_embedding(value.field(), target)?;
    if value.field() == target {
        Ok(value.clone())
    } else {
        Ok(target.element(value.representative().clone()))
    }
}

fn map_finite_polynomial(
    value: &Polynomial<FiniteField>,
    target: &PolynomialRing<FiniteField>,
) -> Result<Polynomial<FiniteField>, DecompositionError> {
    target
        .from_terms(
            value
                .terms()
                .map(|(exponent, coefficient)| {
                    Ok((
                        exponent.to_vec(),
                        map_finite_element(coefficient, target.coefficient_ring())?,
                    ))
                })
                .collect::<Result<Vec<_>, DecompositionError>>()?,
        )
        .map_err(Into::into)
}

fn map_rational_function(
    value: &RationalFunction<FiniteField>,
    target: &FiniteFunctionField,
) -> Result<RationalFunction<FiniteField>, DecompositionError> {
    let numerator = map_finite_polynomial(value.numerator(), target.polynomial_ring())?;
    let denominator = map_finite_polynomial(value.denominator(), target.polynomial_ring())?;
    target.fraction(numerator, denominator).map_err(Into::into)
}

fn map_function_polynomial(
    value: &Polynomial<FiniteFunctionField>,
    target_ring: &PolynomialRing<FiniteFunctionField>,
    target_functions: &FiniteFunctionField,
) -> Result<Polynomial<FiniteFunctionField>, DecompositionError> {
    let terms = value
        .terms()
        .map(|(exponent, coefficient)| {
            Ok((
                exponent.to_vec(),
                map_rational_function(coefficient, target_functions)?,
            ))
        })
        .collect::<Result<Vec<_>, DecompositionError>>()?;
    target_ring.from_terms(terms).map_err(Into::into)
}

fn constant_finite_value(
    value: &RationalFunction<FiniteField>,
) -> Result<FiniteFieldElement, DecompositionError> {
    let numerator = constant_polynomial_value(value.numerator())?;
    let denominator = constant_polynomial_value(value.denominator())?;
    let field = value.field().coefficient_ring();
    let inverse = field.inverse(&denominator).map_err(|_| {
        DecompositionError::Invalid("a rational-function denominator is zero".to_owned())
    })?;
    field.mul(&numerator, &inverse).map_err(Into::into)
}

fn constant_polynomial_value(
    value: &Polynomial<FiniteField>,
) -> Result<FiniteFieldElement, DecompositionError> {
    if value
        .terms()
        .any(|(exponent, _)| exponent.iter().any(|power| power != &Integer::zero()))
    {
        return Err(DecompositionError::Unsupported(
            "binary-form coefficients must be constant in the rational-function generator"
                .to_owned(),
        ));
    }
    Ok(value.constant_coefficient())
}

fn homogeneous_binary_coefficients(
    value: &Polynomial<FiniteFunctionField>,
) -> Result<usize, DecompositionError> {
    let mut degree = None;
    for (exponent, _) in value.terms() {
        if exponent.len() != 2 {
            return Err(DecompositionError::Unsupported(
                "the native floor requires a binary form".to_owned(),
            ));
        }
        let x = integer_to_usize(&exponent[0])?;
        let y = integer_to_usize(&exponent[1])?;
        let total = x.checked_add(y).ok_or(DecompositionError::ResourceLimit {
            resource: "binary-form degree",
            limit: MAX_BINARY_FORM_DEGREE,
        })?;
        match degree {
            None => degree = Some(total),
            Some(expected) if expected == total => {}
            Some(_) => {
                return Err(DecompositionError::Unsupported(
                    "the source generator must be homogeneous".to_owned(),
                ));
            }
        }
    }
    degree.ok_or_else(|| DecompositionError::Invalid("the source generator is zero".to_owned()))
}

fn binary_form_coefficients(
    value: &Polynomial<FiniteFunctionField>,
    degree: usize,
) -> Result<Vec<FiniteFieldElement>, DecompositionError> {
    let base = value.ring().coefficient_ring().coefficient_ring();
    let mut coefficients = vec![base.zero(); degree + 1];
    for (exponent, coefficient) in value.terms() {
        let x_degree = integer_to_usize(&exponent[0])?;
        coefficients[x_degree] = constant_finite_value(coefficient)?;
    }
    Ok(coefficients)
}

fn evaluate_univariate(
    field: &FiniteField,
    coefficients: &[FiniteFieldElement],
    value: &FiniteFieldElement,
) -> Result<FiniteFieldElement, DecompositionError> {
    let mut result = field.zero();
    for coefficient in coefficients.iter().rev() {
        result = field.mul(&result, value)?;
        result = field.add(&result, coefficient)?;
    }
    Ok(result)
}

fn frobenius_ideal(
    component: &FiniteFunctionIdeal,
) -> Result<FiniteFunctionIdeal, DecompositionError> {
    let ring = component.ring();
    let generators = component
        .generators()
        .iter()
        .map(|generator| frobenius_polynomial(generator, ring))
        .collect::<Result<Vec<_>, _>>()?;
    ring.ideal(generators).map_err(Into::into)
}

fn frobenius_polynomial(
    value: &Polynomial<FiniteFunctionField>,
    ring: &PolynomialRing<FiniteFunctionField>,
) -> Result<Polynomial<FiniteFunctionField>, DecompositionError> {
    let functions = ring.coefficient_ring();
    let terms = value
        .terms()
        .map(|(exponent, coefficient)| {
            Ok((
                exponent.to_vec(),
                frobenius_rational_function(coefficient, functions)?,
            ))
        })
        .collect::<Result<Vec<_>, DecompositionError>>()?;
    ring.from_terms(terms).map_err(Into::into)
}

fn frobenius_rational_function(
    value: &RationalFunction<FiniteField>,
    field: &FiniteFunctionField,
) -> Result<RationalFunction<FiniteField>, DecompositionError> {
    let numerator = frobenius_finite_polynomial(value.numerator(), field.polynomial_ring())?;
    let denominator = frobenius_finite_polynomial(value.denominator(), field.polynomial_ring())?;
    field.fraction(numerator, denominator).map_err(Into::into)
}

fn frobenius_finite_polynomial(
    value: &Polynomial<FiniteField>,
    ring: &PolynomialRing<FiniteField>,
) -> Result<Polynomial<FiniteField>, DecompositionError> {
    ring.from_terms(
        value
            .terms()
            .map(|(exponent, coefficient)| (exponent.to_vec(), coefficient.frobenius(1))),
    )
    .map_err(Into::into)
}

fn bounded_field_order(field: &FiniteField) -> Result<usize, DecompositionError> {
    field
        .order()
        .to_string()
        .parse()
        .map_err(|_| DecompositionError::ResourceLimit {
            resource: "splitting-field order",
            limit: MAX_SPLITTING_FIELD_ORDER,
        })
}

fn integer_to_usize(value: &Integer) -> Result<usize, DecompositionError> {
    value
        .to_string()
        .parse()
        .map_err(|_| DecompositionError::ResourceLimit {
            resource: "binary-form degree",
            limit: MAX_BINARY_FORM_DEGREE,
        })
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization(
            "ideal-decomposition component does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}
