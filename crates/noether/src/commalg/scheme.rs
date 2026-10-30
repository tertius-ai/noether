// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed affine and projective schemes over the accepted exact fields.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::{
    FiniteField, FiniteFieldConstructionError, FiniteFieldElement, FiniteFieldEmbedding,
    FiniteFieldEmbeddingError, PointCountCertificate,
};
use crate::{
    CommutativeRing, ContextMismatchError, Field, FiniteContext, Integer, NoetherError,
    RationalField, Ring, SemanticType, TypeDescription,
};

use super::{
    jacobian::{jacobian_minors, JacobianError},
    GroebnerError, Ideal, MonomialOrder, Polynomial, PolynomialContextError, PolynomialError,
    PolynomialFactorizationError, PolynomialRing, RationalUnivariateRepresentation,
};

const MAX_CACHED_FIELD_ELEMENTS: usize = 4_096;
pub(super) const MAX_ENUMERATION_CANDIDATES: usize = 10_000_000;
static AFFINE_SPACE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/AffineSpace@1",
    rust_path: "noether::commalg::AffineSpace",
    python_path: Some("noether.commalg.AffineSpace"),
};

static PROJECTIVE_SPACE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ProjectiveSpace@1",
    rust_path: "noether::commalg::ProjectiveSpace",
    python_path: Some("noether.commalg.ProjectiveSpace"),
};

static SCHEME_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/Scheme@1",
    rust_path: "noether::commalg::Scheme",
    python_path: Some("noether.commalg.Scheme"),
};

static SCHEME_POINT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/SchemePoint@1",
    rust_path: "noether::commalg::SchemePoint",
    python_path: Some("noether.commalg.SchemePoint"),
};

static POINT_ENUMERATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RationalPointEnumeration@1",
    rust_path: "noether::commalg::RationalPointEnumeration",
    python_path: Some("noether.commalg.RationalPointEnumeration"),
};

static FINITE_SCHEME_CLOSED_POINT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/FiniteSchemeClosedPoint@1",
    rust_path: "noether::commalg::FiniteSchemeClosedPoint",
    python_path: Some("noether.commalg.FiniteSchemeClosedPoint"),
};

static FINITE_SCHEME_CLOSED_POINT_SET_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/FiniteSchemeClosedPointSet@1",
    rust_path: "noether::commalg::FiniteSchemeClosedPointSet",
    python_path: Some("noether.commalg.FiniteSchemeClosedPointSet"),
};

/// Failure to construct or enumerate an exact scheme.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemeError {
    /// A polynomial-ring context could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Exact polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// A canonical polynomial ideal could not be constructed or used.
    Groebner(GroebnerError),
    /// A univariate parameter equation could not be factored exactly.
    Factorization(PolynomialFactorizationError),
    /// A finite residue-field presentation could not be constructed.
    FiniteFieldConstruction(FiniteFieldConstructionError),
    /// A checked base-to-residue-field embedding failed.
    FiniteFieldEmbedding(FiniteFieldEmbeddingError),
    /// An ideal or point belonged to another context.
    ContextMismatch(ContextMismatchError),
    /// Coordinate names did not match the declared dimension.
    CoordinateNameCount {
        /// Required number of names.
        expected: usize,
        /// Supplied number of names.
        actual: usize,
    },
    /// A projective defining equation was not homogeneous.
    NonHomogeneousGenerator,
    /// A point had the wrong coordinate count.
    CoordinateCount {
        /// Required number of coordinates.
        expected: usize,
        /// Supplied number of coordinates.
        actual: usize,
    },
    /// The all-zero tuple does not define a projective point.
    ZeroProjectiveCoordinates,
    /// Coordinates did not satisfy the defining equations.
    PointOutsideScheme,
    /// A shard specification was empty or out of range.
    InvalidShard {
        /// Requested zero-based shard index.
        index: usize,
        /// Declared shard count.
        count: usize,
    },
    /// A deliberately bounded exact scheme computation reached a fixed limit.
    ResourceLimit {
        /// Bounded resource that was exhausted.
        resource: &'static str,
        /// Fixed limit for that resource.
        limit: usize,
    },
    /// A geometric conclusion requires a presentation property not established here.
    UnsupportedGeometry(&'static str),
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for SchemeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Factorization(error) => error.fmt(formatter),
            Self::FiniteFieldConstruction(error) => error.fmt(formatter),
            Self::FiniteFieldEmbedding(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::CoordinateNameCount { expected, actual } => write!(
                formatter,
                "ambient space needs {expected} coordinate names, received {actual}"
            ),
            Self::NonHomogeneousGenerator => formatter
                .write_str("a projective closed subscheme requires homogeneous defining equations"),
            Self::CoordinateCount { expected, actual } => write!(
                formatter,
                "scheme point needs {expected} coordinates, received {actual}"
            ),
            Self::ZeroProjectiveCoordinates => {
                formatter.write_str("projective coordinates cannot all be zero")
            }
            Self::PointOutsideScheme => {
                formatter.write_str("coordinates do not satisfy the scheme's defining equations")
            }
            Self::InvalidShard { index, count } => write!(
                formatter,
                "point-enumeration shard {index} is invalid for {count} shards"
            ),
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "exact scheme computation exceeded its {resource} limit of {limit}"
            ),
            Self::UnsupportedGeometry(reason) => {
                write!(formatter, "unsupported geometric conclusion: {reason}")
            }
            Self::InvalidEncoding(message) => {
                write!(formatter, "invalid scheme encoding: {message}")
            }
        }
    }
}

impl std::error::Error for SchemeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::Factorization(error) => Some(error),
            Self::FiniteFieldConstruction(error) => Some(error),
            Self::FiniteFieldEmbedding(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PolynomialContextError> for SchemeError {
    fn from(error: PolynomialContextError) -> Self {
        Self::PolynomialContext(error)
    }
}

impl From<PolynomialError> for SchemeError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<GroebnerError> for SchemeError {
    fn from(error: GroebnerError) -> Self {
        Self::Groebner(error)
    }
}

impl From<PolynomialFactorizationError> for SchemeError {
    fn from(error: PolynomialFactorizationError) -> Self {
        Self::Factorization(error)
    }
}

impl From<FiniteFieldConstructionError> for SchemeError {
    fn from(error: FiniteFieldConstructionError) -> Self {
        Self::FiniteFieldConstruction(error)
    }
}

impl From<FiniteFieldEmbeddingError> for SchemeError {
    fn from(error: FiniteFieldEmbeddingError) -> Self {
        Self::FiniteFieldEmbedding(error)
    }
}

impl From<ContextMismatchError> for SchemeError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

/// Affine space with one explicit base and coordinate-ring context.
#[derive(Clone, PartialEq, Eq)]
pub struct AffineSpace<R: CommutativeRing> {
    coordinate_ring: PolynomialRing<R>,
}

impl<R: CommutativeRing> AffineSpace<R> {
    /// Construct standard affine `dimension`-space with coordinates `x1, ...`.
    pub fn new(base: R, dimension: usize) -> Result<Self, SchemeError> {
        let names = (1..=dimension).map(|index| format!("x{index}"));
        Ok(Self {
            coordinate_ring: PolynomialRing::new(base, names, MonomialOrder::default())?,
        })
    }

    /// Construct affine space with explicit ordered coordinate names.
    pub fn with_names<I, S>(base: R, dimension: usize, names: I) -> Result<Self, SchemeError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let names = names.into_iter().map(Into::into).collect::<Vec<_>>();
        if names.len() != dimension {
            return Err(SchemeError::CoordinateNameCount {
                expected: dimension,
                actual: names.len(),
            });
        }
        Ok(Self {
            coordinate_ring: PolynomialRing::new(base, names, MonomialOrder::default())?,
        })
    }

    pub(crate) fn from_coordinate_ring(coordinate_ring: PolynomialRing<R>) -> Self {
        Self { coordinate_ring }
    }

    /// Base coefficient context.
    pub fn base(&self) -> &R {
        self.coordinate_ring.coefficient_ring()
    }

    /// Affine dimension.
    pub fn dimension(&self) -> usize {
        self.coordinate_ring.variables().len()
    }

    /// Ordered coordinate names.
    pub fn coordinates(&self) -> &[String] {
        self.coordinate_ring.variables()
    }

    /// Exact polynomial coordinate ring.
    pub fn coordinate_ring(&self) -> &PolynomialRing<R> {
        &self.coordinate_ring
    }
}

/// Projective space with one explicit base and homogeneous coordinate ring.
#[derive(Clone, PartialEq, Eq)]
pub struct ProjectiveSpace<R: CommutativeRing> {
    coordinate_ring: PolynomialRing<R>,
}

impl<R: CommutativeRing> ProjectiveSpace<R> {
    /// Construct standard projective `dimension`-space with coordinates `x0, ...`.
    pub fn new(base: R, dimension: usize) -> Result<Self, SchemeError> {
        let count = dimension.checked_add(1).ok_or(SchemeError::ResourceLimit {
            resource: "coordinate count",
            limit: usize::MAX,
        })?;
        let names = (0..count).map(|index| format!("x{index}"));
        Ok(Self {
            coordinate_ring: PolynomialRing::new(base, names, MonomialOrder::default())?,
        })
    }

    /// Construct projective space with explicit ordered homogeneous coordinates.
    pub fn with_names<I, S>(base: R, dimension: usize, names: I) -> Result<Self, SchemeError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let expected = dimension.checked_add(1).ok_or(SchemeError::ResourceLimit {
            resource: "coordinate count",
            limit: usize::MAX,
        })?;
        let names = names.into_iter().map(Into::into).collect::<Vec<_>>();
        if names.len() != expected {
            return Err(SchemeError::CoordinateNameCount {
                expected,
                actual: names.len(),
            });
        }
        Ok(Self {
            coordinate_ring: PolynomialRing::new(base, names, MonomialOrder::default())?,
        })
    }

    pub(crate) fn from_coordinate_ring(
        coordinate_ring: PolynomialRing<R>,
    ) -> Result<Self, SchemeError> {
        if coordinate_ring.variables().is_empty() {
            return Err(SchemeError::CoordinateNameCount {
                expected: 1,
                actual: 0,
            });
        }
        Ok(Self { coordinate_ring })
    }

    /// Base coefficient context.
    pub fn base(&self) -> &R {
        self.coordinate_ring.coefficient_ring()
    }

    /// Projective dimension.
    pub fn dimension(&self) -> usize {
        self.coordinate_ring.variables().len() - 1
    }

    /// Ordered homogeneous coordinate names.
    pub fn coordinates(&self) -> &[String] {
        self.coordinate_ring.variables()
    }

    /// Exact homogeneous coordinate ring.
    pub fn coordinate_ring(&self) -> &PolynomialRing<R> {
        &self.coordinate_ring
    }
}

#[derive(Clone, PartialEq, Eq)]
enum SchemeAmbient<R: CommutativeRing> {
    Affine(AffineSpace<R>),
    Projective(ProjectiveSpace<R>),
}

/// A closed affine or projective scheme with one canonical defining ideal.
#[derive(Clone, PartialEq, Eq)]
pub struct Scheme<R: CommutativeRing> {
    ambient: SchemeAmbient<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
}

impl<R: CommutativeRing> Scheme<R> {
    /// Base field context.
    pub fn base(&self) -> &R {
        self.coordinate_ring().coefficient_ring()
    }

    /// Exact coordinate ring of the chosen ambient presentation.
    pub fn coordinate_ring(&self) -> &PolynomialRing<R> {
        match &self.ambient {
            SchemeAmbient::Affine(space) => space.coordinate_ring(),
            SchemeAmbient::Projective(space) => space.coordinate_ring(),
        }
    }

    /// Canonical defining ideal.
    pub fn defining_ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.defining_ideal
    }

    /// Whether this presentation has an affine ambient space.
    pub fn is_affine(&self) -> bool {
        matches!(self.ambient, SchemeAmbient::Affine(_))
    }

    /// Whether this presentation has a projective ambient space.
    pub fn is_projective(&self) -> bool {
        matches!(self.ambient, SchemeAmbient::Projective(_))
    }

    /// Affine ambient space, when this is an affine presentation.
    pub fn affine_ambient(&self) -> Option<&AffineSpace<R>> {
        match &self.ambient {
            SchemeAmbient::Affine(space) => Some(space),
            SchemeAmbient::Projective(_) => None,
        }
    }

    /// Projective ambient space, when this is a projective presentation.
    pub fn projective_ambient(&self) -> Option<&ProjectiveSpace<R>> {
        match &self.ambient {
            SchemeAmbient::Affine(_) => None,
            SchemeAmbient::Projective(space) => Some(space),
        }
    }
}

pub(super) fn affine_closed_subscheme<R: CommutativeRing>(
    space: &AffineSpace<R>,
    ideal: Ideal<PolynomialRing<R>>,
) -> Result<Scheme<R>, SchemeError> {
    if ideal.ring() != space.coordinate_ring() {
        return Err(ContextMismatchError {
            expected: space.coordinate_ring().to_string(),
            actual: ideal.ring().to_string(),
        }
        .into());
    }
    Ok(Scheme {
        ambient: SchemeAmbient::Affine(space.clone()),
        defining_ideal: ideal,
    })
}

pub(super) fn projective_closed_subscheme<R: CommutativeRing>(
    space: &ProjectiveSpace<R>,
    ideal: Ideal<PolynomialRing<R>>,
) -> Result<Scheme<R>, SchemeError> {
    if ideal.ring() != space.coordinate_ring() {
        return Err(ContextMismatchError {
            expected: space.coordinate_ring().to_string(),
            actual: ideal.ring().to_string(),
        }
        .into());
    }
    for generator in ideal.generators() {
        if generator.homogeneous_degree()?.is_none() && !generator.is_zero() {
            return Err(SchemeError::NonHomogeneousGenerator);
        }
    }
    Ok(Scheme {
        ambient: SchemeAmbient::Projective(space.clone()),
        defining_ideal: ideal,
    })
}

macro_rules! impl_exact_scheme_context {
    ($field:ty) => {
        impl AffineSpace<$field> {
            /// Construct a closed subscheme from an ideal in this coordinate ring.
            pub fn closed_subscheme(
                &self,
                ideal: Ideal<PolynomialRing<$field>>,
            ) -> Result<Scheme<$field>, SchemeError> {
                affine_closed_subscheme(self, ideal)
            }

            /// Serialize this ambient context.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }

            /// Restore and validate this ambient context.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl ProjectiveSpace<$field> {
            /// Construct a homogeneous closed subscheme in this projective space.
            pub fn closed_subscheme(
                &self,
                ideal: Ideal<PolynomialRing<$field>>,
            ) -> Result<Scheme<$field>, SchemeError> {
                projective_closed_subscheme(self, ideal)
            }

            /// Serialize this ambient context.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }

            /// Restore and validate this ambient context.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl Scheme<$field> {
            /// Dimension of this affine or projective presentation.
            pub fn dimension(&self) -> Result<i64, SchemeError> {
                let quotient_dimension = self.defining_ideal.dimension()?;
                Ok(if self.is_projective() && quotient_dimension >= 0 {
                    quotient_dimension - 1
                } else {
                    quotient_dimension
                })
            }

            /// Standard-graded degree (multiplicity) of this presentation.
            pub fn degree(&self) -> Result<Integer, SchemeError> {
                self.defining_ideal.degree().map_err(Into::into)
            }

            /// Standard Hilbert polynomial of the homogeneous/filtered coordinate ring.
            pub fn hilbert_polynomial(&self) -> Result<Polynomial<RationalField>, SchemeError> {
                self.defining_ideal.hilbert_polynomial().map_err(Into::into)
            }

            /// Rank-defect subscheme cut out by the expected-codimension
            /// Jacobian minors of the canonical defining basis.
            pub fn jacobian_subscheme(&self) -> Result<Self, SchemeError> {
                let quotient_dimension = self.defining_ideal.dimension()?;
                if quotient_dimension < 0 {
                    return Ok(self.clone());
                }
                let generators = jacobian_subscheme_generators(self, quotient_dimension)?;
                let ideal = self.coordinate_ring().ideal(generators)?;
                scheme_with_defining_ideal(self, ideal)
            }

            /// Scheme-theoretic singular locus for an established complete-
            /// intersection presentation over this perfect field.
            pub fn singular_subscheme(&self) -> Result<Self, SchemeError> {
                let quotient_dimension = self.defining_ideal.dimension()?;
                if quotient_dimension < 0 {
                    return Ok(self.clone());
                }
                let codimension = quotient_codimension(self, quotient_dimension)?;
                if self.defining_ideal.generators().len() != codimension {
                    return Err(SchemeError::UnsupportedGeometry(
                        "singular_subscheme currently requires a complete-intersection presentation certified by generator count and codimension",
                    ));
                }
                self.jacobian_subscheme()
            }

            /// Decide smoothness when the exact complete-intersection Jacobian
            /// criterion applies.
            pub fn is_smooth(&self) -> Result<bool, SchemeError> {
                Ok(self.singular_subscheme()?.dimension()? < 0)
            }

            /// Serialize this scheme presentation and canonical defining ideal.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }

            /// Restore and revalidate this scheme presentation.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl SemanticType for AffineSpace<$field> {
            const TYPE_ID: &'static str = "noether/commalg/AffineSpace@1";

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &AFFINE_SPACE_DESCRIPTION
            }
        }

        impl SemanticType for ProjectiveSpace<$field> {
            const TYPE_ID: &'static str = "noether/commalg/ProjectiveSpace@1";

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &PROJECTIVE_SPACE_DESCRIPTION
            }
        }

        impl SemanticType for Scheme<$field> {
            const TYPE_ID: &'static str = "noether/commalg/Scheme@1";

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                &SCHEME_DESCRIPTION
            }
        }
    };
}

impl_exact_scheme_context!(RationalField);
impl_exact_scheme_context!(FiniteField);

fn scheme_with_defining_ideal<R: CommutativeRing>(
    scheme: &Scheme<R>,
    ideal: Ideal<PolynomialRing<R>>,
) -> Result<Scheme<R>, SchemeError> {
    match &scheme.ambient {
        SchemeAmbient::Affine(space) => affine_closed_subscheme(space, ideal),
        SchemeAmbient::Projective(space) => projective_closed_subscheme(space, ideal),
    }
}

fn quotient_codimension<R: CommutativeRing>(
    scheme: &Scheme<R>,
    quotient_dimension: i64,
) -> Result<usize, SchemeError> {
    let quotient_dimension = usize::try_from(quotient_dimension).map_err(|_| {
        SchemeError::UnsupportedGeometry("a nonempty scheme had negative quotient dimension")
    })?;
    scheme
        .coordinate_ring()
        .variables()
        .len()
        .checked_sub(quotient_dimension)
        .ok_or(SchemeError::UnsupportedGeometry(
            "quotient dimension exceeds ambient coordinate-ring dimension",
        ))
}

fn jacobian_subscheme_generators<R: Field>(
    scheme: &Scheme<R>,
    quotient_dimension: i64,
) -> Result<Vec<Polynomial<R>>, SchemeError> {
    let defining = scheme.defining_ideal.generators();
    let codimension = quotient_codimension(scheme, quotient_dimension)?;
    let mut generators = defining.to_vec();
    generators
        .extend(jacobian_minors(&scheme.defining_ideal, codimension).map_err(map_jacobian_error)?);
    Ok(generators)
}

fn map_jacobian_error(error: JacobianError) -> SchemeError {
    match error {
        JacobianError::Polynomial(error) => SchemeError::Polynomial(error),
        JacobianError::ContextMismatch(error) => SchemeError::ContextMismatch(error),
        JacobianError::ResourceLimit { resource, limit } => {
            SchemeError::ResourceLimit { resource, limit }
        }
        JacobianError::Invalid(reason) => SchemeError::UnsupportedGeometry(reason),
    }
}

/// A checked rational point on one exact scheme.
#[derive(Clone, PartialEq, Eq)]
pub struct SchemePoint<R: CommutativeRing> {
    scheme: Scheme<R>,
    coordinates: Vec<R::Element>,
}

impl<R: Field> SchemePoint<R> {
    /// Target scheme.
    pub fn scheme(&self) -> &Scheme<R> {
        &self.scheme
    }

    /// Canonical affine or normalized projective coordinates.
    pub fn coordinates(&self) -> &[R::Element] {
        &self.coordinates
    }

    /// Re-evaluate every defining equation and the projective normalization.
    pub fn verify(&self) -> Result<(), SchemeError> {
        let rebuilt = self.scheme.point(self.coordinates.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(SchemeError::PointOutsideScheme)
        }
    }
}

impl<R: Field> SchemePoint<R> {
    /// Serialize this checked point.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        Self: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and recheck this point.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        Self: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> SemanticType for SchemePoint<R>
where
    Scheme<R>: SemanticType,
{
    const TYPE_ID: &'static str = "noether/commalg/SchemePoint@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_semantic_component(&mut output, &self.scheme.canonical_bytes()?)?;
        for coordinate in &self.coordinates {
            append_semantic_component(&mut output, &coordinate.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SCHEME_POINT_DESCRIPTION
    }
}

/// One geometric closed point of an affine finite-field scheme.
///
/// Coordinates live in an explicit residue-field presentation. The point is
/// represented by the lexicographically least conjugate in its relative
/// Frobenius orbit, so conjugate geometric coordinates define one canonical
/// value.
#[derive(Clone, PartialEq, Eq)]
pub struct FiniteSchemeClosedPoint {
    scheme: Scheme<FiniteField>,
    embedding: FiniteFieldEmbedding,
    coordinates: Vec<FiniteFieldElement>,
    degree: usize,
    frobenius_orbit: Vec<Vec<FiniteFieldElement>>,
}

impl FiniteSchemeClosedPoint {
    /// Scheme containing this closed point.
    pub fn scheme(&self) -> &Scheme<FiniteField> {
        &self.scheme
    }

    /// Checked embedding of the base field into the residue field.
    pub fn base_embedding(&self) -> &FiniteFieldEmbedding {
        &self.embedding
    }

    /// Explicit finite residue-field presentation.
    pub fn residue_field(&self) -> &FiniteField {
        self.embedding.target()
    }

    /// Canonical geometric coordinates in the residue field.
    pub fn coordinates(&self) -> &[FiniteFieldElement] {
        &self.coordinates
    }

    /// Degree of the closed point over the scheme's base field.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Complete relative-Frobenius orbit, starting at the canonical conjugate.
    pub fn frobenius_orbit(&self) -> &[Vec<FiniteFieldElement>] {
        &self.frobenius_orbit
    }

    /// Recheck the residue presentation, orbit, minimal degree, and equations.
    pub fn verify(&self) -> Result<(), SchemeError> {
        if !self.scheme.is_affine() {
            return Err(SchemeError::UnsupportedGeometry(
                "finite closed-point coordinates currently require an affine presentation",
            ));
        }
        self.embedding.verify()?;
        if self.embedding.source() != self.scheme.base() {
            return Err(SchemeError::InvalidEncoding(
                "closed-point embedding does not start at the scheme's base field".to_owned(),
            ));
        }
        if self.degree == 0
            || self.scheme.base().degree().checked_mul(self.degree)
                != Some(self.residue_field().degree())
        {
            return Err(SchemeError::InvalidEncoding(
                "closed-point degree does not match the residue-field presentation".to_owned(),
            ));
        }
        let arity = self.scheme.coordinate_ring().variables().len();
        if self.coordinates.len() != arity
            || self.frobenius_orbit.len() != self.degree
            || self.frobenius_orbit.first() != Some(&self.coordinates)
            || self.frobenius_orbit.iter().any(|conjugate| {
                conjugate.len() != arity
                    || conjugate
                        .iter()
                        .any(|coordinate| !self.residue_field().contains(coordinate))
            })
        {
            return Err(SchemeError::InvalidEncoding(
                "closed-point coordinates or Frobenius orbit have the wrong shape".to_owned(),
            ));
        }
        let relative_exponent = u64::try_from(self.scheme.base().degree()).map_err(|_| {
            SchemeError::UnsupportedGeometry(
                "the base-field degree does not fit the Frobenius exponent",
            )
        })?;
        let mut expected = self.coordinates.clone();
        for (index, conjugate) in self.frobenius_orbit.iter().enumerate() {
            if conjugate != &expected {
                return Err(SchemeError::InvalidEncoding(
                    "closed-point orbit is not generated by relative Frobenius".to_owned(),
                ));
            }
            if index != 0 && conjugate == &self.coordinates {
                return Err(SchemeError::InvalidEncoding(
                    "closed-point degree is not minimal for its Frobenius orbit".to_owned(),
                ));
            }
            if !scheme_coordinates_vanish_over_embedding(&self.scheme, &self.embedding, conjugate)?
            {
                return Err(SchemeError::PointOutsideScheme);
            }
            expected = conjugate
                .iter()
                .map(|coordinate| coordinate.frobenius(relative_exponent))
                .collect();
        }
        if expected != self.coordinates {
            return Err(SchemeError::InvalidEncoding(
                "closed-point Frobenius orbit does not close at its declared degree".to_owned(),
            ));
        }
        let least = self
            .frobenius_orbit
            .iter()
            .min_by_key(|conjugate| closed_point_coordinate_key(conjugate))
            .ok_or_else(|| {
                SchemeError::InvalidEncoding("closed-point orbit is empty".to_owned())
            })?;
        if least != &self.coordinates {
            return Err(SchemeError::InvalidEncoding(
                "closed point does not use its canonical Frobenius conjugate".to_owned(),
            ));
        }
        Ok(())
    }

    /// Serialize the complete residue-field and orbit evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly reverify a closed point.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for FiniteSchemeClosedPoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteSchemeClosedPoint")
            .field("degree", &self.degree)
            .field("residue_field", self.residue_field())
            .field("coordinates", &self.coordinates)
            .finish()
    }
}

impl fmt::Display for FiniteSchemeClosedPoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FiniteSchemeClosedPoint(degree={}, coordinates={:?})",
            self.degree, self.coordinates
        )
    }
}

impl SemanticType for FiniteSchemeClosedPoint {
    const TYPE_ID: &'static str = "noether/commalg/FiniteSchemeClosedPoint@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_SCHEME_CLOSED_POINT_DESCRIPTION
    }
}

/// Canonical closed-point solution of a zero-dimensional affine scheme.
#[derive(Clone, PartialEq, Eq)]
pub struct FiniteSchemeClosedPointSet {
    scheme: Scheme<FiniteField>,
    representation: Option<RationalUnivariateRepresentation<FiniteField>>,
    points: Vec<FiniteSchemeClosedPoint>,
}

impl FiniteSchemeClosedPointSet {
    /// Solved affine finite-field scheme.
    pub fn scheme(&self) -> &Scheme<FiniteField> {
        &self.scheme
    }

    /// RUR used for a nonempty zero-dimensional quotient.
    pub fn representation(&self) -> Option<&RationalUnivariateRepresentation<FiniteField>> {
        self.representation.as_ref()
    }

    /// Closed points in deterministic residue-presentation/coordinate order.
    pub fn points(&self) -> &[FiniteSchemeClosedPoint] {
        &self.points
    }

    /// Sum of closed-point degrees, without scheme multiplicities.
    pub fn geometric_point_count(&self) -> Integer {
        self.points.iter().fold(Integer::zero(), |sum, point| {
            sum + Integer::from(point.degree as u64)
        })
    }

    /// Recompute the RUR solution and compare every canonical closed point.
    pub fn verify(&self) -> Result<(), SchemeError> {
        for point in &self.points {
            point.verify()?;
            if point.scheme() != &self.scheme {
                return Err(SchemeError::InvalidEncoding(
                    "closed-point set contains a point on another scheme".to_owned(),
                ));
            }
        }
        let rebuilt = solve_finite_scheme_closed_points(&self.scheme)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(SchemeError::InvalidEncoding(
                "recomputation changed the finite closed-point solution".to_owned(),
            ))
        }
    }

    /// Serialize the scheme, RUR, and all verified closed points.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and recompute the complete solution.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for FiniteSchemeClosedPointSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FiniteSchemeClosedPointSet")
            .field("points", &self.points.len())
            .field("geometric_point_count", &self.geometric_point_count())
            .finish()
    }
}

impl fmt::Display for FiniteSchemeClosedPointSet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "FiniteSchemeClosedPointSet(points={}, geometric_points={})",
            self.points.len(),
            self.geometric_point_count()
        )
    }
}

impl SemanticType for FiniteSchemeClosedPointSet {
    const TYPE_ID: &'static str = "noether/commalg/FiniteSchemeClosedPointSet@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &FINITE_SCHEME_CLOSED_POINT_SET_DESCRIPTION
    }
}

impl<R: Field> Scheme<R> {
    /// Validate one rational point and normalize projective coordinates.
    pub fn point<I>(&self, coordinates: I) -> Result<SchemePoint<R>, SchemeError>
    where
        I: IntoIterator<Item = R::Element>,
    {
        let mut coordinates = coordinates.into_iter().collect::<Vec<_>>();
        let expected = self.coordinate_ring().variables().len();
        if coordinates.len() != expected {
            return Err(SchemeError::CoordinateCount {
                expected,
                actual: coordinates.len(),
            });
        }
        if coordinates.iter().any(|value| !self.base().contains(value)) {
            return Err(ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "point coordinate from another exact field context".to_owned(),
            }
            .into());
        }
        if self.is_projective() {
            let zero = self.base().zero();
            let pivot = coordinates
                .iter()
                .position(|value| value != &zero)
                .ok_or(SchemeError::ZeroProjectiveCoordinates)?;
            let inverse = self
                .base()
                .inverse(&coordinates[pivot])
                .map_err(|_| SchemeError::ZeroProjectiveCoordinates)?;
            coordinates = coordinates
                .iter()
                .map(|value| self.base().mul(value, &inverse))
                .collect::<Result<Vec<_>, _>>()?;
        }
        if !self.is_point_coordinates(&coordinates)? {
            return Err(SchemeError::PointOutsideScheme);
        }
        Ok(SchemePoint {
            scheme: self.clone(),
            coordinates,
        })
    }

    pub(crate) fn is_point_coordinates(
        &self,
        coordinates: &[R::Element],
    ) -> Result<bool, SchemeError> {
        let zero = self.base().zero();
        for equation in self.defining_ideal.generators() {
            if equation.evaluate(coordinates)? != zero {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

impl Scheme<FiniteField> {
    /// Solve a zero-dimensional affine scheme into closed points using a
    /// rational univariate representation and exact finite-field factorization.
    ///
    /// This path never enumerates the base field. Proper residue extensions
    /// use deterministic Conway presentations when the current word-
    /// characteristic finite-field backend admits them.
    pub fn closed_points(&self) -> Result<FiniteSchemeClosedPointSet, SchemeError> {
        solve_finite_scheme_closed_points(self)
    }

    /// Stream all rational points in canonical order.
    ///
    /// At most ten million ambient candidates are admitted. Fields of order
    /// at most 4096 use a shared coordinate table; larger fields decode each
    /// coordinate on demand and do not materialize the field.
    pub fn rational_points(&self) -> Result<RationalPointEnumeration, SchemeError> {
        self.rational_points_shard(0, 1)
    }

    /// Stream one deterministic congruence shard of the canonical candidates.
    pub fn rational_points_shard(
        &self,
        shard_index: usize,
        shard_count: usize,
    ) -> Result<RationalPointEnumeration, SchemeError> {
        RationalPointEnumeration::new(self.clone(), shard_index, shard_count, None)
    }

    /// Count rational points with one deterministic compiled Rust pass.
    pub fn point_count(&self) -> Result<(Integer, PointCountCertificate), SchemeError> {
        self.point_count_partitioned(1)
    }

    /// Count rational points in deterministic checksum-bearing shards.
    pub fn point_count_partitioned(
        &self,
        shard_count: usize,
    ) -> Result<(Integer, PointCountCertificate), SchemeError> {
        if shard_count == 0 {
            return Err(SchemeError::InvalidShard { index: 0, count: 0 });
        }
        let data = EnumerationData::new(self)?;
        let mut shard_point_counts = Vec::with_capacity(shard_count);
        let mut shard_candidate_counts = Vec::with_capacity(shard_count);
        let mut shard_checksums = Vec::with_capacity(shard_count);
        let mut total = 0_u64;
        for shard in 0..shard_count {
            let mut count = 0_u64;
            let mut scanned = 0_u64;
            let mut checksum = crate::point_count::CHECKSUM_OFFSET;
            let mut index = shard;
            while index < data.candidate_count {
                let coordinates = data.coordinates(index);
                let is_point = self.is_point_coordinates(&coordinates)?;
                crate::point_count::checksum_candidate(&mut checksum, index as u64, is_point);
                scanned += 1;
                if is_point {
                    count += 1;
                }
                index = index.saturating_add(shard_count);
            }
            total = total.saturating_add(count);
            shard_point_counts.push(count);
            shard_candidate_counts.push(scanned);
            shard_checksums.push(checksum);
        }
        let model_checksum = crate::point_count::checksum_bytes(
            &self
                .canonical_bytes()
                .map_err(|error| SchemeError::InvalidEncoding(error.to_string()))?,
        );
        let certificate = PointCountCertificate::from_verified_parts(
            model_checksum,
            data.candidate_count as u64,
            shard_candidate_counts,
            shard_point_counts,
            shard_checksums,
        )?;
        Ok((Integer::from(total), certificate))
    }
}

fn solve_finite_scheme_closed_points(
    scheme: &Scheme<FiniteField>,
) -> Result<FiniteSchemeClosedPointSet, SchemeError> {
    if !scheme.is_affine() {
        return Err(SchemeError::UnsupportedGeometry(
            "finite closed-point solving currently requires an affine presentation",
        ));
    }
    match scheme.dimension()? {
        -1 => {
            return Ok(FiniteSchemeClosedPointSet {
                scheme: scheme.clone(),
                representation: None,
                points: Vec::new(),
            });
        }
        0 => {}
        _ => {
            return Err(SchemeError::UnsupportedGeometry(
                "closed_points requires a zero-dimensional scheme",
            ));
        }
    }

    let representation = scheme
        .defining_ideal()
        .rational_univariate_representation()?;
    let factorization = representation.equation().factor()?;
    let mut points = Vec::with_capacity(factorization.factors().len());
    for (factor, _) in factorization.factors() {
        let degree = factor
            .degree(Some(0))?
            .ok_or_else(|| {
                SchemeError::InvalidEncoding(
                    "an RUR parameter factor has no finite degree".to_owned(),
                )
            })?
            .to_string()
            .parse::<usize>()
            .map_err(|_| SchemeError::ResourceLimit {
                resource: "closed-point residue degree",
                limit: usize::MAX,
            })?;
        if degree == 0 {
            return Err(SchemeError::InvalidEncoding(
                "an RUR parameter factor is constant".to_owned(),
            ));
        }
        points.push(closed_point_from_parameter_factor(
            scheme,
            &representation,
            factor,
            degree,
        )?);
    }
    points.sort_by_key(|point| {
        (
            point.degree,
            point.residue_field().to_string(),
            closed_point_coordinate_key(&point.coordinates),
        )
    });
    Ok(FiniteSchemeClosedPointSet {
        scheme: scheme.clone(),
        representation: Some(representation),
        points,
    })
}

fn closed_point_from_parameter_factor(
    scheme: &Scheme<FiniteField>,
    representation: &RationalUnivariateRepresentation<FiniteField>,
    factor: &Polynomial<FiniteField>,
    degree: usize,
) -> Result<FiniteSchemeClosedPoint, SchemeError> {
    let base = scheme.base();
    let (residue_field, embedding) = if degree == 1 {
        let residue_field = base.clone();
        let embedding = FiniteFieldEmbedding::new(
            base.clone(),
            residue_field.clone(),
            residue_field.generator(),
        )?;
        (residue_field, embedding)
    } else {
        let characteristic = base
            .word_characteristic()
            .ok_or(SchemeError::UnsupportedGeometry(
                "proper residue extensions currently require word-sized characteristic",
            ))?;
        let absolute_degree =
            base.degree()
                .checked_mul(degree)
                .ok_or(SchemeError::ResourceLimit {
                    resource: "absolute residue-field degree",
                    limit: usize::MAX,
                })?;
        let residue_field = FiniteField::extension(characteristic, absolute_degree)?;
        let embedding = deterministic_finite_field_embedding(base, &residue_field)?;
        (residue_field, embedding)
    };

    let mapped_factor = factor
        .base_change(&embedding)
        .map_err(|error| SchemeError::InvalidEncoding(error.to_string()))?;
    let parameter = first_linear_root(&mapped_factor)?;
    let mapped_denominator = representation
        .denominator()
        .base_change(&embedding)
        .map_err(|error| SchemeError::InvalidEncoding(error.to_string()))?;
    let denominator = mapped_denominator.evaluate(std::slice::from_ref(&parameter))?;
    let denominator_inverse = residue_field.inverse(&denominator).map_err(|_| {
        SchemeError::InvalidEncoding("the RUR denominator vanishes at a parameter root".to_owned())
    })?;
    let coordinates = representation
        .coordinate_numerators()
        .iter()
        .map(|coordinate| {
            let coordinate = coordinate
                .base_change(&embedding)
                .map_err(|error| SchemeError::InvalidEncoding(error.to_string()))?;
            let numerator = coordinate.evaluate(std::slice::from_ref(&parameter))?;
            residue_field
                .mul(&numerator, &denominator_inverse)
                .map_err(SchemeError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let relative_exponent = u64::try_from(base.degree()).map_err(|_| {
        SchemeError::UnsupportedGeometry("the base-field degree does not fit Frobenius")
    })?;
    let mut orbit = Vec::with_capacity(degree);
    let mut conjugate = coordinates;
    for _ in 0..degree {
        orbit.push(conjugate.clone());
        conjugate = conjugate
            .iter()
            .map(|coordinate| coordinate.frobenius(relative_exponent))
            .collect();
    }
    let least_index = orbit
        .iter()
        .enumerate()
        .min_by_key(|(_, conjugate)| closed_point_coordinate_key(conjugate))
        .map(|(index, _)| index)
        .ok_or_else(|| SchemeError::InvalidEncoding("closed-point orbit is empty".to_owned()))?;
    orbit.rotate_left(least_index);
    let point = FiniteSchemeClosedPoint {
        scheme: scheme.clone(),
        embedding,
        coordinates: orbit[0].clone(),
        degree,
        frobenius_orbit: orbit,
    };
    point.verify()?;
    Ok(point)
}

pub(crate) fn deterministic_finite_field_embedding(
    source: &FiniteField,
    target: &FiniteField,
) -> Result<FiniteFieldEmbedding, SchemeError> {
    if source.degree() == 1 {
        return FiniteFieldEmbedding::new(source.clone(), target.clone(), target.one())
            .map_err(Into::into);
    }
    let parameter_ring = PolynomialRing::new(target.clone(), ["t"], MonomialOrder::Lexicographic)?;
    let modulus = source
        .defining_modulus()
        .ok_or(SchemeError::UnsupportedGeometry(
            "a proper finite field has no defining modulus",
        ))?;
    let polynomial =
        parameter_ring.from_terms(modulus.iter().enumerate().map(|(degree, coefficient)| {
            (
                vec![Integer::from(degree)],
                target.element(Integer::from(*coefficient)),
            )
        }))?;
    let generator_image = first_linear_root(&polynomial)?;
    FiniteFieldEmbedding::new(source.clone(), target.clone(), generator_image).map_err(Into::into)
}

fn first_linear_root(
    polynomial: &Polynomial<FiniteField>,
) -> Result<FiniteFieldElement, SchemeError> {
    let factorization = polynomial.factor()?;
    let linear = factorization
        .factors()
        .iter()
        .map(|(factor, _)| factor)
        .find(|factor| {
            factor
                .degree(Some(0))
                .is_ok_and(|degree| degree == Some(Integer::one()))
        })
        .ok_or(SchemeError::UnsupportedGeometry(
            "the selected residue field does not split the parameter factor",
        ))?;
    let constant = linear.coefficient(&[Integer::zero()])?;
    polynomial
        .ring()
        .coefficient_ring()
        .neg(&constant)
        .map_err(Into::into)
}

fn scheme_coordinates_vanish_over_embedding(
    scheme: &Scheme<FiniteField>,
    embedding: &FiniteFieldEmbedding,
    coordinates: &[FiniteFieldElement],
) -> Result<bool, SchemeError> {
    let zero = embedding.target().zero();
    for equation in scheme.defining_ideal().generators() {
        let equation = equation
            .base_change(embedding)
            .map_err(|error| SchemeError::InvalidEncoding(error.to_string()))?;
        if equation.evaluate(coordinates)? != zero {
            return Ok(false);
        }
    }
    Ok(true)
}

fn closed_point_coordinate_key(coordinates: &[FiniteFieldElement]) -> Vec<u8> {
    let mut key = Vec::new();
    for coordinate in coordinates {
        let bytes = coordinate.representative().to_string().into_bytes();
        key.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        key.extend_from_slice(&bytes);
    }
    key
}

struct EnumerationData {
    field: FiniteField,
    field_order: usize,
    elements: Option<Vec<FiniteFieldElement>>,
    coordinate_count: usize,
    projective: bool,
    candidate_count: usize,
}

impl EnumerationData {
    fn new(scheme: &Scheme<FiniteField>) -> Result<Self, SchemeError> {
        let field = scheme.base().clone();
        let field_order = field
            .order()
            .to_u64_checked()
            .and_then(|order| usize::try_from(order).ok())
            .ok_or(SchemeError::ResourceLimit {
                resource: "finite-field coordinate radix",
                limit: usize::MAX,
            })?;
        let coordinate_count = scheme.coordinate_ring().variables().len();
        let candidate_count = if scheme.is_projective() {
            projective_candidate_count(field_order, coordinate_count)?
        } else {
            checked_power(field_order, coordinate_count)?
        };
        let elements =
            (field_order <= MAX_CACHED_FIELD_ELEMENTS).then(|| field.elements().collect());
        Ok(Self {
            field,
            field_order,
            elements,
            coordinate_count,
            projective: scheme.is_projective(),
            candidate_count,
        })
    }

    fn coordinates(&self, index: usize) -> Vec<FiniteFieldElement> {
        if self.projective {
            self.projective_coordinates(index)
        } else {
            self.cartesian_coordinates(self.coordinate_count, index)
        }
    }

    fn element(&self, index: usize) -> FiniteFieldElement {
        match &self.elements {
            Some(elements) => elements[index].clone(),
            None => self
                .field
                .element_at_index(index as u64)
                .expect("coordinate index is below the checked field cardinality"),
        }
    }

    fn cartesian_coordinates(&self, count: usize, mut index: usize) -> Vec<FiniteFieldElement> {
        let mut coordinates = vec![self.field.zero(); count];
        for coordinate in coordinates.iter_mut().rev() {
            *coordinate = self.element(index % self.field_order);
            index /= self.field_order;
        }
        coordinates
    }

    fn projective_coordinates(&self, mut index: usize) -> Vec<FiniteFieldElement> {
        let zero = self.field.zero();
        let one = self.field.one();
        for pivot in 0..self.coordinate_count {
            let suffix = self.coordinate_count - pivot - 1;
            let block = self.field_order.pow(suffix as u32);
            if index < block {
                let mut coordinates = vec![zero; pivot];
                coordinates.push(one);
                coordinates.extend(self.cartesian_coordinates(suffix, index));
                return coordinates;
            }
            index -= block;
        }
        unreachable!("validated projective index belongs to one pivot block")
    }
}

fn checked_power(base: usize, exponent: usize) -> Result<usize, SchemeError> {
    let mut value = 1_usize;
    for _ in 0..exponent {
        value = value.checked_mul(base).ok_or(SchemeError::ResourceLimit {
            resource: "candidate count",
            limit: MAX_ENUMERATION_CANDIDATES,
        })?;
        if value > MAX_ENUMERATION_CANDIDATES {
            return Err(SchemeError::ResourceLimit {
                resource: "candidate count",
                limit: MAX_ENUMERATION_CANDIDATES,
            });
        }
    }
    Ok(value)
}

fn projective_candidate_count(base: usize, coordinates: usize) -> Result<usize, SchemeError> {
    let mut total = 0_usize;
    for suffix in 0..coordinates {
        total =
            total
                .checked_add(checked_power(base, suffix)?)
                .ok_or(SchemeError::ResourceLimit {
                    resource: "candidate count",
                    limit: MAX_ENUMERATION_CANDIDATES,
                })?;
        if total > MAX_ENUMERATION_CANDIDATES {
            return Err(SchemeError::ResourceLimit {
                resource: "candidate count",
                limit: MAX_ENUMERATION_CANDIDATES,
            });
        }
    }
    Ok(total)
}

#[cfg(test)]
mod indexed_coordinate_tests {
    use super::*;

    #[test]
    fn lazy_and_cached_coordinates_have_identical_order_in_every_pivot_block() {
        for field in [
            FiniteField::prime(3).unwrap(),
            FiniteField::extension(3, 2).unwrap(),
            FiniteField::with_modulus(3, vec![1, 0, 1]).unwrap(),
        ] {
            for dimension in 0..=2 {
                let affine = AffineSpace::new(field.clone(), dimension).unwrap();
                let projective = ProjectiveSpace::new(field.clone(), dimension).unwrap();
                let schemes = [
                    affine
                        .closed_subscheme(affine.coordinate_ring().ideal([]).unwrap())
                        .unwrap(),
                    projective
                        .closed_subscheme(projective.coordinate_ring().ideal([]).unwrap())
                        .unwrap(),
                ];
                for scheme in schemes {
                    let mut data = EnumerationData::new(&scheme).unwrap();
                    let cached = (0..data.candidate_count)
                        .map(|index| data.coordinates(index))
                        .collect::<Vec<_>>();
                    data.elements = None;
                    for (index, expected) in cached.into_iter().enumerate() {
                        assert_eq!(data.coordinates(index), expected);
                    }
                }
            }
        }
    }
}

/// Deterministic lazy enumeration of rational points on one finite-field scheme.
#[derive(Clone)]
pub struct RationalPointEnumeration {
    scheme: Scheme<FiniteField>,
    data: std::sync::Arc<EnumerationData>,
    shard_index: usize,
    shard_count: usize,
    next_index: usize,
}

impl RationalPointEnumeration {
    fn new(
        scheme: Scheme<FiniteField>,
        shard_index: usize,
        shard_count: usize,
        next_index: Option<usize>,
    ) -> Result<Self, SchemeError> {
        if shard_count == 0 || shard_index >= shard_count {
            return Err(SchemeError::InvalidShard {
                index: shard_index,
                count: shard_count,
            });
        }
        let data = std::sync::Arc::new(EnumerationData::new(&scheme)?);
        let next_index = next_index.unwrap_or(shard_index);
        if next_index > data.candidate_count
            || (next_index < data.candidate_count && next_index % shard_count != shard_index)
        {
            return Err(SchemeError::InvalidEncoding(
                "enumeration cursor is outside its declared shard".to_owned(),
            ));
        }
        Ok(Self {
            scheme,
            data,
            shard_index,
            shard_count,
            next_index,
        })
    }

    /// Enumerated scheme.
    pub fn scheme(&self) -> &Scheme<FiniteField> {
        &self.scheme
    }

    /// Zero-based shard index.
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    /// Total number of disjoint shards.
    pub fn shard_count(&self) -> usize {
        self.shard_count
    }

    /// Total canonical ambient candidates across all shards.
    pub fn candidate_count(&self) -> usize {
        self.data.candidate_count
    }

    /// Current canonical candidate cursor.
    pub fn next_candidate_index(&self) -> usize {
        self.next_index
    }

    /// Serialize a resumable deterministic cursor.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&RationalPointEnumerationWire {
            type_id: Self::TYPE_ID.to_owned(),
            scheme: self.scheme.clone(),
            shard_index: self.shard_index,
            shard_count: self.shard_count,
            next_index: self.next_index,
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore a resumable deterministic cursor.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<RationalPointEnumerationWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(
            wire.scheme,
            wire.shard_index,
            wire.shard_count,
            Some(wire.next_index),
        )
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Iterator for RationalPointEnumeration {
    type Item = SchemePoint<FiniteField>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next_index < self.data.candidate_count {
            let index = self.next_index;
            self.next_index = index
                .checked_add(self.shard_count)
                .unwrap_or(self.data.candidate_count)
                .min(self.data.candidate_count);
            let coordinates = self.data.coordinates(index);
            if self
                .scheme
                .is_point_coordinates(&coordinates)
                .expect("internally generated finite-field coordinates match the scheme context")
            {
                return Some(SchemePoint {
                    scheme: self.scheme.clone(),
                    coordinates,
                });
            }
        }
        None
    }
}

impl PartialEq for RationalPointEnumeration {
    fn eq(&self, other: &Self) -> bool {
        self.scheme == other.scheme
            && self.shard_index == other.shard_index
            && self.shard_count == other.shard_count
            && self.next_index == other.next_index
    }
}

impl Eq for RationalPointEnumeration {}

impl SemanticType for RationalPointEnumeration {
    const TYPE_ID: &'static str = "noether/commalg/RationalPointEnumeration@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &POINT_ENUMERATION_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RationalPointEnumerationWire {
    #[serde(rename = "type")]
    type_id: String,
    scheme: Scheme<FiniteField>,
    shard_index: usize,
    shard_count: usize,
    next_index: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "R: CommutativeRing + Serialize",
    deserialize = "R: CommutativeRing + Deserialize<'de>"
))]
struct SpaceWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    coordinate_ring: PolynomialRing<R>,
}

impl<R> Serialize for AffineSpace<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("AffineSpace", 2)?;
        state.serialize_field("type", "noether/commalg/AffineSpace@1")?;
        state.serialize_field("coordinate_ring", self.coordinate_ring())?;
        state.end()
    }
}

impl<R> Serialize for ProjectiveSpace<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ProjectiveSpace", 2)?;
        state.serialize_field("type", "noether/commalg/ProjectiveSpace@1")?;
        state.serialize_field("coordinate_ring", self.coordinate_ring())?;
        state.end()
    }
}

macro_rules! impl_space_deserialize {
    ($field:ty) => {
        impl<'de> Deserialize<'de> for AffineSpace<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let wire = SpaceWire::<$field>::deserialize(deserializer)?;
                if wire.type_id != "noether/commalg/AffineSpace@1" {
                    return Err(serde::de::Error::custom("wrong affine-space type identity"));
                }
                Ok(Self::from_coordinate_ring(wire.coordinate_ring))
            }
        }

        impl<'de> Deserialize<'de> for ProjectiveSpace<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let wire = SpaceWire::<$field>::deserialize(deserializer)?;
                if wire.type_id != "noether/commalg/ProjectiveSpace@1" {
                    return Err(serde::de::Error::custom(
                        "wrong projective-space type identity",
                    ));
                }
                Self::from_coordinate_ring(wire.coordinate_ring).map_err(serde::de::Error::custom)
            }
        }
    };
}

impl_space_deserialize!(RationalField);
impl_space_deserialize!(FiniteField);

impl<R> fmt::Debug for AffineSpace<R>
where
    R: CommutativeRing,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AffineSpace")
            .field("dimension", &self.dimension())
            .field("base", &R::TYPE_ID)
            .finish()
    }
}

impl<R> fmt::Display for AffineSpace<R>
where
    R: CommutativeRing,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "AffineSpace({}, {})",
            self.dimension(),
            R::TYPE_ID
        )
    }
}

impl<R> fmt::Debug for ProjectiveSpace<R>
where
    R: CommutativeRing,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProjectiveSpace")
            .field("dimension", &self.dimension())
            .field("base", &R::TYPE_ID)
            .finish()
    }
}

impl<R> fmt::Display for ProjectiveSpace<R>
where
    R: CommutativeRing,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ProjectiveSpace({}, {})",
            self.dimension(),
            R::TYPE_ID
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "R: Field + Serialize, R::Element: Serialize",
    deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>"
))]
struct SchemeWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ambient_kind: String,
    coordinate_ring: PolynomialRing<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
}

impl<R> Serialize for Scheme<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Scheme", 4)?;
        state.serialize_field("type", "noether/commalg/Scheme@1")?;
        state.serialize_field(
            "ambient_kind",
            if self.is_affine() {
                "affine"
            } else {
                "projective"
            },
        )?;
        state.serialize_field("coordinate_ring", self.coordinate_ring())?;
        state.serialize_field("defining_ideal", self.defining_ideal())?;
        state.end()
    }
}

macro_rules! impl_scheme_deserialize {
    ($field:ty) => {
        impl<'de> Deserialize<'de> for Scheme<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let wire = SchemeWire::<$field>::deserialize(deserializer)?;
                if wire.type_id != "noether/commalg/Scheme@1" {
                    return Err(serde::de::Error::custom("wrong scheme type identity"));
                }
                match wire.ambient_kind.as_str() {
                    "affine" => AffineSpace::from_coordinate_ring(wire.coordinate_ring)
                        .closed_subscheme(wire.defining_ideal)
                        .map_err(serde::de::Error::custom),
                    "projective" => ProjectiveSpace::from_coordinate_ring(wire.coordinate_ring)
                        .and_then(|space| space.closed_subscheme(wire.defining_ideal))
                        .map_err(serde::de::Error::custom),
                    _ => Err(serde::de::Error::custom(
                        "scheme ambient_kind must be affine or projective",
                    )),
                }
            }
        }
    };
}

impl_scheme_deserialize!(RationalField);
impl_scheme_deserialize!(FiniteField);

impl<R: CommutativeRing> fmt::Debug for Scheme<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Scheme")
            .field(
                "ambient",
                &if self.is_affine() {
                    "affine"
                } else {
                    "projective"
                },
            )
            .field("equations", &self.defining_ideal.generators().len())
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Display for Scheme<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Scheme(ambient={}, equations={})",
            if self.is_affine() {
                "affine"
            } else {
                "projective"
            },
            self.defining_ideal.generators().len()
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "Scheme<R>: Serialize, R::Element: Serialize",
        deserialize = "Scheme<R>: Deserialize<'de>, R::Element: Deserialize<'de>"
    )
)]
struct SchemePointWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    scheme: Scheme<R>,
    coordinates: Vec<R::Element>,
}

impl<R> Serialize for SchemePoint<R>
where
    R: Field,
    Scheme<R>: Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SchemePoint", 3)?;
        state.serialize_field("type", "noether/commalg/SchemePoint@1")?;
        state.serialize_field("scheme", self.scheme())?;
        state.serialize_field("coordinates", self.coordinates())?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for SchemePoint<R>
where
    R: Field,
    Scheme<R>: Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SchemePointWire::<R>::deserialize(deserializer)?;
        if wire.type_id != "noether/commalg/SchemePoint@1" {
            return Err(serde::de::Error::custom("wrong scheme-point type identity"));
        }
        wire.scheme
            .point(wire.coordinates)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteSchemeClosedPointWire {
    #[serde(rename = "type")]
    type_id: String,
    scheme: Scheme<FiniteField>,
    embedding: FiniteFieldEmbedding,
    coordinates: Vec<FiniteFieldElement>,
    degree: usize,
    frobenius_orbit: Vec<Vec<FiniteFieldElement>>,
}

impl Serialize for FiniteSchemeClosedPoint {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FiniteSchemeClosedPoint", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("scheme", &self.scheme)?;
        state.serialize_field("embedding", &self.embedding)?;
        state.serialize_field("coordinates", &self.coordinates)?;
        state.serialize_field("degree", &self.degree)?;
        state.serialize_field("frobenius_orbit", &self.frobenius_orbit)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for FiniteSchemeClosedPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FiniteSchemeClosedPointWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let point = Self {
            scheme: wire.scheme,
            embedding: wire.embedding,
            coordinates: wire.coordinates,
            degree: wire.degree,
            frobenius_orbit: wire.frobenius_orbit,
        };
        point.verify().map_err(serde::de::Error::custom)?;
        Ok(point)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteSchemeClosedPointSetWire {
    #[serde(rename = "type")]
    type_id: String,
    scheme: Scheme<FiniteField>,
    representation: Option<RationalUnivariateRepresentation<FiniteField>>,
    points: Vec<FiniteSchemeClosedPoint>,
}

impl Serialize for FiniteSchemeClosedPointSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("FiniteSchemeClosedPointSet", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("scheme", &self.scheme)?;
        state.serialize_field("representation", &self.representation)?;
        state.serialize_field("points", &self.points)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for FiniteSchemeClosedPointSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = FiniteSchemeClosedPointSetWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let supplied = Self {
            scheme: wire.scheme,
            representation: wire.representation,
            points: wire.points,
        };
        supplied.verify().map_err(serde::de::Error::custom)?;
        Ok(supplied)
    }
}

fn append_semantic_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization("semantic component exceeds canonical framing".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

impl<R> fmt::Debug for SchemePoint<R>
where
    R: Field,
    R::Element: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("SchemePoint")
            .field(&self.coordinates)
            .finish()
    }
}

impl<R: Field> fmt::Display for SchemePoint<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "SchemePoint({} coordinates)",
            self.coordinates.len()
        )
    }
}

impl fmt::Debug for RationalPointEnumeration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RationalPointEnumeration")
            .field("shard_index", &self.shard_index)
            .field("shard_count", &self.shard_count)
            .field("next_index", &self.next_index)
            .field("candidate_count", &self.data.candidate_count)
            .finish()
    }
}
