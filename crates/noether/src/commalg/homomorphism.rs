// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked homomorphisms out of polynomial rings.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Polynomial, PolynomialError, PolynomialRing};
use crate::{
    CoefficientMap, CommutativeRing, CompositeCoefficientMap, ContextMismatchError,
    EuclideanDomain, Integer, NoetherError, Ring, SemanticType, TypeDescription, VerificationError,
    Z,
};

static POLYNOMIAL_HOMOMORPHISM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PolynomialHomomorphism@1",
    rust_path: "noether::commalg::PolynomialHomomorphism",
    python_path: Some("noether.commalg.PolynomialHomomorphism"),
};
const POLYNOMIAL_HOMOMORPHISM_TYPE_ID: &str = "noether/commalg/PolynomialHomomorphism@1";

static POLYNOMIAL_CONSTANT_EMBEDDING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PolynomialConstantEmbedding@1",
    rust_path: "noether::commalg::PolynomialConstantEmbedding",
    python_path: None,
};

/// Failure to construct, apply, compose, or verify a polynomial homomorphism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolynomialHomomorphismError {
    /// Two declared contexts were not exactly equal.
    ContextMismatch(ContextMismatchError),
    /// The number of variable images did not match the source arity.
    ImageArity {
        /// Required number of images.
        expected: usize,
        /// Supplied number of images.
        actual: usize,
    },
    /// One image did not belong to the declared target context.
    ImageOutsideTarget {
        /// Zero-based source-variable index.
        variable: usize,
    },
    /// The coefficient map rejected one source coefficient.
    Coefficient(String),
    /// Target-ring arithmetic failed after endpoint validation.
    TargetArithmetic(String),
    /// A polynomial value failed exact construction.
    Polynomial(PolynomialError),
}

impl fmt::Display for PolynomialHomomorphismError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::ImageArity { expected, actual } => write!(
                formatter,
                "polynomial homomorphism received {actual} variable images; expected {expected}"
            ),
            Self::ImageOutsideTarget { variable } => write!(
                formatter,
                "image of source variable {variable} lies outside the declared target context"
            ),
            Self::Coefficient(detail) => {
                write!(formatter, "polynomial coefficient map failed: {detail}")
            }
            Self::TargetArithmetic(detail) => {
                write!(
                    formatter,
                    "polynomial homomorphism target arithmetic failed: {detail}"
                )
            }
            Self::Polynomial(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PolynomialHomomorphismError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::ImageArity { .. }
            | Self::ImageOutsideTarget { .. }
            | Self::Coefficient(_)
            | Self::TargetArithmetic(_) => None,
        }
    }
}

impl From<ContextMismatchError> for PolynomialHomomorphismError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<PolynomialError> for PolynomialHomomorphismError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

/// The canonical inclusion of a coefficient ring as constant polynomials.
#[derive(Clone, PartialEq, Eq)]
pub struct PolynomialConstantEmbedding<R: CommutativeRing> {
    target: PolynomialRing<R>,
}

#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>")
)]
struct PolynomialConstantEmbeddingWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    target: PolynomialRing<R>,
}

impl<R: CommutativeRing> PolynomialConstantEmbedding<R> {
    /// Construct the canonical constant inclusion into `target`.
    pub fn new(target: PolynomialRing<R>) -> Self {
        Self { target }
    }

    /// Verify the source as the target's exact coefficient context.
    pub fn verify(&self) -> Result<(), VerificationError> {
        Ok(())
    }

    /// Serialize the complete target polynomial context.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a constant inclusion.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: CommutativeRing + fmt::Display> CoefficientMap for PolynomialConstantEmbedding<R> {
    type Source = R;
    type Target = PolynomialRing<R>;
    type Error = PolynomialError;

    fn source(&self) -> &Self::Source {
        self.target.coefficient_ring()
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(&self, value: &R::Element) -> Result<Polynomial<R>, Self::Error> {
        self.target.constant(value.clone())
    }
}

impl<R> Serialize for PolynomialConstantEmbedding<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PolynomialConstantEmbeddingWire {
            type_id: Self::TYPE_ID.to_owned(),
            target: self.target.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de, R> Deserialize<'de> for PolynomialConstantEmbedding<R>
where
    R: CommutativeRing + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PolynomialConstantEmbeddingWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(Self::new(wire.target))
    }
}

impl<R: CommutativeRing> SemanticType for PolynomialConstantEmbedding<R> {
    const TYPE_ID: &'static str = "noether/commalg/PolynomialConstantEmbedding@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        endpoint_bytes(self.target.coefficient_ring(), &self.target)
    }

    fn describe() -> &'static TypeDescription {
        &POLYNOMIAL_CONSTANT_EMBEDDING_DESCRIPTION
    }
}

impl<R: CommutativeRing + fmt::Display> fmt::Debug for PolynomialConstantEmbedding<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolynomialConstantEmbedding")
            .field("source", &self.source().to_string())
            .field("target", self.target())
            .finish()
    }
}

impl<R: CommutativeRing + fmt::Display> fmt::Display for PolynomialConstantEmbedding<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.source(), self.target())
    }
}

/// A checked unital homomorphism from a polynomial ring to a commutative ring.
///
/// The target may itself be a polynomial ring. The coefficient map lands in
/// the actual target context, and `images[i]` is the image of the `i`-th
/// source variable. Thus evaluation, partial evaluation, arbitrary
/// substitutions, changes of arity, and simultaneous base change share one
/// value type.
#[derive(Clone, PartialEq, Eq)]
pub struct PolynomialHomomorphism<S, T, M>
where
    S: CommutativeRing,
    T: CommutativeRing,
    M: CoefficientMap<Source = S, Target = T>,
{
    source: PolynomialRing<S>,
    target: T,
    coefficient_map: M,
    images: Vec<T::Element>,
}

/// A polynomial homomorphism whose target is itself a polynomial ring.
///
/// The supplied coefficient map lands in the target coefficient ring and is
/// composed with the canonical constant-polynomial embedding.
pub type PolynomialRingHomomorphism<S, T, M> = PolynomialHomomorphism<
    S,
    PolynomialRing<T>,
    CompositeCoefficientMap<M, PolynomialConstantEmbedding<T>>,
>;

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "S: CommutativeRing + Deserialize<'de>, T: CommutativeRing + Deserialize<'de>, M: CoefficientMap<Source = S, Target = T> + Deserialize<'de>, T::Element: Deserialize<'de>"
    )
)]
struct PolynomialHomomorphismWire<S, T, M>
where
    S: CommutativeRing,
    T: CommutativeRing,
    M: CoefficientMap<Source = S, Target = T>,
{
    #[serde(rename = "type")]
    type_id: String,
    source: PolynomialRing<S>,
    target: T,
    coefficient_map: M,
    images: Vec<T::Element>,
}

impl<S, T, M> PolynomialHomomorphism<S, T, M>
where
    S: CommutativeRing + fmt::Display,
    T: CommutativeRing + fmt::Display,
    M: CoefficientMap<Source = S, Target = T>,
{
    /// Construct after checking every endpoint and variable image.
    pub fn new(
        source: PolynomialRing<S>,
        target: T,
        coefficient_map: M,
        images: Vec<T::Element>,
    ) -> Result<Self, PolynomialHomomorphismError> {
        if source.coefficient_ring() != coefficient_map.source() {
            return Err(ContextMismatchError {
                expected: source.coefficient_ring().to_string(),
                actual: coefficient_map.source().to_string(),
            }
            .into());
        }
        if &target != coefficient_map.target() {
            return Err(ContextMismatchError {
                expected: target.to_string(),
                actual: coefficient_map.target().to_string(),
            }
            .into());
        }
        if images.len() != source.variables().len() {
            return Err(PolynomialHomomorphismError::ImageArity {
                expected: source.variables().len(),
                actual: images.len(),
            });
        }
        if let Some(variable) = images.iter().position(|image| !target.contains(image)) {
            return Err(PolynomialHomomorphismError::ImageOutsideTarget { variable });
        }
        Ok(Self {
            source,
            target,
            coefficient_map,
            images,
        })
    }

    /// Exact source polynomial context.
    pub fn source(&self) -> &PolynomialRing<S> {
        &self.source
    }

    /// Exact target commutative-ring context.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Checked map used on source coefficients.
    pub fn coefficient_map(&self) -> &M {
        &self.coefficient_map
    }

    /// Images of source variables in declared order.
    pub fn images(&self) -> &[T::Element] {
        &self.images
    }

    /// Apply the homomorphism by a deterministic sparse term stream.
    pub fn apply(&self, value: &Polynomial<S>) -> Result<T::Element, PolynomialHomomorphismError> {
        if !self.source.contains(value) {
            return Err(ContextMismatchError {
                expected: self.source.to_string(),
                actual: value.ring().to_string(),
            }
            .into());
        }
        let mut result = self.target.zero();
        for (exponent, coefficient) in value.terms() {
            let mut term = self
                .coefficient_map
                .apply(coefficient)
                .map_err(|error| PolynomialHomomorphismError::Coefficient(error.to_string()))?;
            for (power, image) in exponent.iter().zip(&self.images) {
                let factor = pow_in_ring(&self.target, image, power)?;
                term = self.target.mul(&term, &factor).map_err(|error| {
                    PolynomialHomomorphismError::TargetArithmetic(error.to_string())
                })?;
            }
            result = self.target.add(&result, &term).map_err(|error| {
                PolynomialHomomorphismError::TargetArithmetic(error.to_string())
            })?;
        }
        Ok(result)
    }

    /// Verify exact endpoints and image membership.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.source.coefficient_ring() != self.coefficient_map.source() {
            return Err(VerificationError::InvalidWitness {
                claim: "polynomial homomorphism coefficient source equals source coefficients",
            });
        }
        if &self.target != self.coefficient_map.target() {
            return Err(VerificationError::InvalidWitness {
                claim: "polynomial homomorphism coefficient target equals target context",
            });
        }
        if self.images.len() != self.source.variables().len() {
            return Err(VerificationError::InvalidWitness {
                claim: "polynomial homomorphism has one image per source variable",
            });
        }
        if self.images.iter().any(|image| !self.target.contains(image)) {
            return Err(VerificationError::InvalidWitness {
                claim: "polynomial homomorphism images belong to its target context",
            });
        }
        Ok(())
    }

    /// Serialize all endpoint, coefficient-map, and image data.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        S: Serialize,
        T: Serialize,
        M: Serialize,
        T::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a complete homomorphism.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        S: for<'de> Deserialize<'de>,
        T: for<'de> Deserialize<'de>,
        M: for<'de> Deserialize<'de>,
        T::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R, U, N> PolynomialHomomorphism<R, U, N>
where
    R: CommutativeRing + fmt::Display,
    U: CommutativeRing + fmt::Display,
    N: CoefficientMap<Source = R, Target = U> + Clone,
{
    /// Compose `self` after a homomorphism whose target is `self.source()`.
    pub fn compose<S, M>(
        &self,
        inner: &PolynomialHomomorphism<S, PolynomialRing<R>, M>,
    ) -> Result<
        PolynomialHomomorphism<S, U, CompositeCoefficientMap<M, Self>>,
        PolynomialHomomorphismError,
    >
    where
        S: CommutativeRing + fmt::Display,
        M: CoefficientMap<Source = S, Target = PolynomialRing<R>> + Clone,
    {
        if inner.target() != self.source() {
            return Err(ContextMismatchError {
                expected: self.source().to_string(),
                actual: inner.target().to_string(),
            }
            .into());
        }
        let coefficient_map =
            CompositeCoefficientMap::new(inner.coefficient_map().clone(), self.clone())?;
        let images = inner
            .images()
            .iter()
            .map(|image| self.apply(image))
            .collect::<Result<Vec<_>, _>>()?;
        PolynomialHomomorphism::new(
            inner.source().clone(),
            self.target().clone(),
            coefficient_map,
            images,
        )
    }
}

impl<S, T, M> CoefficientMap for PolynomialHomomorphism<S, T, M>
where
    S: CommutativeRing + fmt::Display,
    T: CommutativeRing + fmt::Display,
    M: CoefficientMap<Source = S, Target = T>,
{
    type Source = PolynomialRing<S>;
    type Target = T;
    type Error = PolynomialHomomorphismError;

    fn source(&self) -> &Self::Source {
        &self.source
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(&self, value: &Polynomial<S>) -> Result<T::Element, Self::Error> {
        PolynomialHomomorphism::apply(self, value)
    }
}

impl<S, T, M> Serialize for PolynomialHomomorphism<S, T, M>
where
    S: CommutativeRing + Serialize,
    T: CommutativeRing + Serialize,
    M: CoefficientMap<Source = S, Target = T> + Serialize,
    T::Element: Serialize,
{
    fn serialize<A>(&self, serializer: A) -> Result<A::Ok, A::Error>
    where
        A: Serializer,
    {
        let mut state = serializer.serialize_struct("PolynomialHomomorphism", 5)?;
        state.serialize_field("type", POLYNOMIAL_HOMOMORPHISM_TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("target", &self.target)?;
        state.serialize_field("coefficient_map", &self.coefficient_map)?;
        state.serialize_field("images", &self.images)?;
        state.end()
    }
}

impl<'de, S, T, M> Deserialize<'de> for PolynomialHomomorphism<S, T, M>
where
    S: CommutativeRing + fmt::Display + Deserialize<'de>,
    T: CommutativeRing + fmt::Display + Deserialize<'de>,
    M: CoefficientMap<Source = S, Target = T> + Deserialize<'de>,
    T::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PolynomialHomomorphismWire::<S, T, M>::deserialize(deserializer)?;
        if wire.type_id != POLYNOMIAL_HOMOMORPHISM_TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                POLYNOMIAL_HOMOMORPHISM_TYPE_ID, wire.type_id
            )));
        }
        Self::new(wire.source, wire.target, wire.coefficient_map, wire.images)
            .map_err(serde::de::Error::custom)
    }
}

impl<S, T, M> SemanticType for PolynomialHomomorphism<S, T, M>
where
    S: CommutativeRing + fmt::Display,
    T: CommutativeRing + fmt::Display,
    M: CoefficientMap<Source = S, Target = T> + SemanticType + PartialEq + Eq,
{
    const TYPE_ID: &'static str = "noether/commalg/PolynomialHomomorphism@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = endpoint_bytes(self.source(), self.target())?;
        append_component(&mut output, &self.coefficient_map.canonical_bytes()?)?;
        append_length(&mut output, self.images.len())?;
        for image in &self.images {
            append_component(&mut output, &image.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &POLYNOMIAL_HOMOMORPHISM_DESCRIPTION
    }
}

impl<S, T, M> fmt::Debug for PolynomialHomomorphism<S, T, M>
where
    S: CommutativeRing + fmt::Display,
    T: CommutativeRing + fmt::Display,
    M: CoefficientMap<Source = S, Target = T>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolynomialHomomorphism")
            .field("source", &self.source)
            .field("target", &self.target.to_string())
            .field("image_count", &self.images.len())
            .finish()
    }
}

impl<S, T, M> fmt::Display for PolynomialHomomorphism<S, T, M>
where
    S: CommutativeRing + fmt::Display,
    T: CommutativeRing + fmt::Display,
    M: CoefficientMap<Source = S, Target = T>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} -> {}", self.source, self.target)
    }
}

impl<S: CommutativeRing + fmt::Display> PolynomialRing<S> {
    /// Construct a homomorphism to another polynomial ring.
    ///
    /// `coefficient_map` lands in the target coefficient ring; Noether
    /// composes it with the canonical constant-polynomial inclusion.
    pub fn homomorphism<T, M>(
        &self,
        target: &PolynomialRing<T>,
        coefficient_map: M,
        images: Vec<Polynomial<T>>,
    ) -> Result<PolynomialRingHomomorphism<S, T, M>, PolynomialHomomorphismError>
    where
        T: CommutativeRing + fmt::Display,
        M: CoefficientMap<Source = S, Target = T>,
    {
        let constant_embedding = PolynomialConstantEmbedding::new(target.clone());
        let coefficient_map = CompositeCoefficientMap::new(coefficient_map, constant_embedding)?;
        PolynomialHomomorphism::new(self.clone(), target.clone(), coefficient_map, images)
    }
}

fn pow_in_ring<R: CommutativeRing>(
    ring: &R,
    value: &R::Element,
    exponent: &Integer,
) -> Result<R::Element, PolynomialHomomorphismError> {
    let mut exponent = exponent.clone();
    let mut result = ring.one();
    let mut base = value.clone();
    let two = Integer::from(2_u8);
    while !exponent.is_zero() {
        let (quotient, remainder) = Z
            .div_rem(&exponent, &two)
            .map_err(|error| PolynomialHomomorphismError::TargetArithmetic(error.to_string()))?;
        if !remainder.is_zero() {
            result = ring.mul(&result, &base).map_err(|error| {
                PolynomialHomomorphismError::TargetArithmetic(error.to_string())
            })?;
        }
        exponent = quotient;
        if !exponent.is_zero() {
            base = ring.mul(&base, &base).map_err(|error| {
                PolynomialHomomorphismError::TargetArithmetic(error.to_string())
            })?;
        }
    }
    Ok(result)
}

fn endpoint_bytes<S: SemanticType, T: SemanticType>(
    source: &S,
    target: &T,
) -> Result<Vec<u8>, NoetherError> {
    let mut output = Vec::new();
    append_component(&mut output, &source.canonical_bytes()?)?;
    append_component(&mut output, &target.canonical_bytes()?)?;
    Ok(output)
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    append_length(output, component.len())?;
    output.extend_from_slice(component);
    Ok(())
}

fn append_length(output: &mut Vec<u8>, length: usize) -> Result<(), NoetherError> {
    let length = u64::try_from(length).map_err(|_| {
        NoetherError::Serialization("polynomial homomorphism component is too large".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::MonomialOrder;
    use super::*;
    use crate::numthy::FiniteField;
    use crate::{IdentityCoefficientMap, IntegerRationalEmbedding, Q};

    #[test]
    fn arbitrary_images_and_coefficient_base_change_apply_in_one_pass() {
        let source = PolynomialRing::new(Z, ["x", "y", "z"], MonomialOrder::Lexicographic).unwrap();
        let field = FiniteField::prime(5_u8).unwrap();
        let target = PolynomialRing::new(
            field.clone(),
            ["a", "b", "t"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let [a, b, t] = target.generators().unwrap().try_into().unwrap();
        let one = target.one();
        let bt = target.mul(&b, &t).unwrap();
        let line = target.add(&a, &bt).unwrap();
        let hom = source
            .homomorphism(
                &target,
                Z.finite_field_map(&field),
                vec![one, t.clone(), line.clone()],
            )
            .unwrap();

        let polynomial = source
            .from_u64_terms([
                (vec![1, 1, 0], Integer::from(2)),
                (vec![0, 0, 2], Integer::from(3)),
            ])
            .unwrap();
        let line_square = target.mul(&line, &line).unwrap();
        let expected = target
            .add(
                &t.scalar_mul(&field.element(2)).unwrap(),
                &line_square.scalar_mul(&field.element(3)).unwrap(),
            )
            .unwrap();
        assert_eq!(hom.apply(&polynomial).unwrap(), expected);
        hom.verify().unwrap();
    }

    #[test]
    fn evaluation_and_composition_have_exact_endpoint_semantics() {
        let source = PolynomialRing::new(Z, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        let middle = PolynomialRing::new(Q, ["u", "v"], MonomialOrder::Lexicographic).unwrap();
        let [u, v] = middle.generators().unwrap().try_into().unwrap();
        let inner = source
            .homomorphism(
                &middle,
                IntegerRationalEmbedding::new(),
                vec![u.clone(), v.clone()],
            )
            .unwrap();

        let target = PolynomialRing::new(Q, ["t"], MonomialOrder::Lexicographic).unwrap();
        let t = target.generator(0).unwrap();
        let outer = middle
            .homomorphism(
                &target,
                IdentityCoefficientMap::new(Q),
                vec![
                    target.add(&t, &target.one()).unwrap(),
                    target.mul(&t, &t).unwrap(),
                ],
            )
            .unwrap();
        let composed = outer.compose(&inner).unwrap();

        let polynomial = source
            .from_u64_terms([(vec![2, 0], Integer::one()), (vec![0, 1], Integer::from(3))])
            .unwrap();
        assert_eq!(
            composed.apply(&polynomial).unwrap(),
            outer.apply(&inner.apply(&polynomial).unwrap()).unwrap()
        );

        let evaluation = PolynomialHomomorphism::new(
            source,
            Q,
            IntegerRationalEmbedding::new(),
            vec![Q.element(2), Q.element(3)],
        )
        .unwrap();
        assert_eq!(evaluation.apply(&polynomial).unwrap(), Q.element(13));
    }
}
