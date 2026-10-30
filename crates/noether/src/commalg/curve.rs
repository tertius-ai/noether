// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked curves and canonical algebraic function fields.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Sub};

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::linalg::{LinalgError, Matrix};
use crate::numthy::{FiniteField, NumberField};
use crate::{
    CommutativeRing, ContextMismatchError, DivisionByZeroError, Field, Integer, IntegralDomain,
    InversionError, NoetherError, RationalField, Ring, SemanticType, TypeDescription, UnitError,
    UnitInversion, VerificationError,
};

use super::scheme::projective_closed_subscheme;
use super::MonomialOrder;
use super::{
    GroebnerError, Polynomial, PolynomialContextError, PolynomialError, PolynomialFactorization,
    PolynomialFactorizationError, PolynomialRing, PrimeIdeal, ProjectiveSpace,
    RationalFunctionError, RationalFunctionField, Scheme, SchemeError, SchemePoint,
};

static CURVE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/Curve@1",
    rust_path: "noether::commalg::Curve",
    python_path: Some("noether.commalg.Curve"),
};

static CURVE_FUNCTION_FIELD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CurveFunctionField@1",
    rust_path: "noether::commalg::CurveFunctionField",
    python_path: Some("noether.commalg.CurveFunctionField"),
};

static CURVE_FUNCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CurveFunction@1",
    rust_path: "noether::commalg::CurveFunction",
    python_path: Some("noether.commalg.CurveFunction"),
};

static CURVE_POINT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CurvePoint@1",
    rust_path: "noether::commalg::CurvePoint",
    python_path: Some("noether.commalg.CurvePoint"),
};

static CURVE_MORPHISM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CurveMorphism@1",
    rust_path: "noether::commalg::CurveMorphism",
    python_path: Some("noether.commalg.CurveMorphism"),
};

static CURVE_CLOSED_POINT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CurveClosedPoint@1",
    rust_path: "noether::commalg::CurveClosedPoint",
    python_path: Some("noether.commalg.CurveClosedPoint"),
};

static WEIL_DIVISOR_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/WeilDivisor@1",
    rust_path: "noether::commalg::WeilDivisor",
    python_path: Some("noether.commalg.WeilDivisor"),
};

static RIEMANN_ROCH_SPACE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RiemannRochSpace@1",
    rust_path: "noether::commalg::RiemannRochSpace",
    python_path: Some("noether.commalg.RiemannRochSpace"),
};

/// Failure to construct or operate on a checked curve or its function field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurveError {
    /// The projective scheme layer rejected the presentation or a point.
    Scheme(SchemeError),
    /// A polynomial context could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Exact polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Exact ideal arithmetic failed.
    Groebner(GroebnerError),
    /// Exact linear algebra for a section space failed.
    LinearAlgebra(LinalgError),
    /// Rational-function arithmetic in the chosen transcendence coordinate failed.
    RationalFunction(RationalFunctionError),
    /// A univariate intersection polynomial could not be factored exactly.
    Factorization(PolynomialFactorizationError),
    /// Values belonged to unequal curve or function-field contexts.
    ContextMismatch(ContextMismatchError),
    /// A plane-curve equation was not a nonzero homogeneous polynomial in three variables.
    InvalidPlaneEquation,
    /// The supplied plane equation did not cut out a projective curve.
    NotOneDimensional,
    /// The exact projective Jacobian criterion found a singular geometric point.
    SingularCurve,
    /// A Weierstrass equation had zero discriminant.
    SingularWeierstrassEquation,
    /// No deterministic affine presentation admitted a finite extension of `K(x)`.
    NoFunctionFieldPresentation,
    /// Division by the zero curve function was attempted.
    DivisionByZero,
    /// A bounded local expansion did not expose the first nonzero coefficient.
    LocalExpansionLimit {
        /// Maximum order computed in the chosen local parameter.
        limit: usize,
    },
    /// An exact curve algorithm reached a declared finite resource bound.
    ResourceLimit {
        /// Resource whose bound was reached.
        resource: &'static str,
        /// Declared bound.
        limit: usize,
    },
    /// A serialized or recomputed witness did not verify.
    InvalidWitness(&'static str),
}

impl fmt::Display for CurveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheme(error) => error.fmt(formatter),
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::LinearAlgebra(error) => error.fmt(formatter),
            Self::RationalFunction(error) => error.fmt(formatter),
            Self::Factorization(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::InvalidPlaneEquation => formatter.write_str(
                "a smooth projective plane curve needs one nonzero standard-homogeneous equation in three variables",
            ),
            Self::NotOneDimensional => {
                formatter.write_str("the supplied projective equation does not have dimension one")
            }
            Self::SingularCurve => formatter
                .write_str("the projective plane curve fails the exact Jacobian smoothness test"),
            Self::SingularWeierstrassEquation => {
                formatter.write_str("the Weierstrass equation has zero discriminant")
            }
            Self::NoFunctionFieldPresentation => formatter.write_str(
                "no deterministic affine chart presents the curve as a finite extension of K(x)",
            ),
            Self::DivisionByZero => formatter.write_str("division by zero curve function"),
            Self::LocalExpansionLimit { limit } => write!(
                formatter,
                "the local curve expansion did not determine a value through order {limit}"
            ),
            Self::ResourceLimit { resource, limit } => {
                write!(formatter, "curve computation exceeded its {resource} limit of {limit}")
            }
            Self::InvalidWitness(claim) => write!(formatter, "invalid curve witness: {claim}"),
        }
    }
}

impl std::error::Error for CurveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scheme(error) => Some(error),
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::RationalFunction(error) => Some(error),
            Self::Factorization(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::InvalidPlaneEquation
            | Self::NotOneDimensional
            | Self::SingularCurve
            | Self::SingularWeierstrassEquation
            | Self::NoFunctionFieldPresentation
            | Self::DivisionByZero
            | Self::LocalExpansionLimit { .. }
            | Self::ResourceLimit { .. }
            | Self::InvalidWitness(_) => None,
        }
    }
}

impl From<SchemeError> for CurveError {
    fn from(value: SchemeError) -> Self {
        Self::Scheme(value)
    }
}

impl From<PolynomialContextError> for CurveError {
    fn from(value: PolynomialContextError) -> Self {
        Self::PolynomialContext(value)
    }
}

impl From<PolynomialError> for CurveError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<GroebnerError> for CurveError {
    fn from(value: GroebnerError) -> Self {
        Self::Groebner(value)
    }
}

impl From<LinalgError> for CurveError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<RationalFunctionError> for CurveError {
    fn from(value: RationalFunctionError) -> Self {
        Self::RationalFunction(value)
    }
}

impl From<PolynomialFactorizationError> for CurveError {
    fn from(value: PolynomialFactorizationError) -> Self {
        Self::Factorization(value)
    }
}

impl From<ContextMismatchError> for CurveError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// The admitted checked presentation retained by a curve value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurvePresentationKind {
    /// One smooth homogeneous equation in the projective plane.
    SmoothProjectivePlane,
    /// A nonsingular projective generalized Weierstrass equation.
    SmoothProjectiveWeierstrass,
}

/// Replayable numerical evidence for the plane-curve geometry claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurveSmoothnessEvidence {
    quotient_dimension: i64,
    singular_cone_dimension: i64,
}

impl CurveSmoothnessEvidence {
    /// Krull dimension of the homogeneous coordinate-ring quotient.
    pub fn quotient_dimension(&self) -> i64 {
        self.quotient_dimension
    }

    /// Dimension of the affine singular cone; at most zero means empty in `Proj`.
    pub fn singular_cone_dimension(&self) -> i64 {
        self.singular_cone_dimension
    }
}

#[derive(Clone, PartialEq, Eq)]
enum CurvePresentation<R: Field> {
    Plane,
    Weierstrass { coefficients: [R::Element; 5] },
}

/// A checked smooth projective integral curve in an admitted exact presentation.
///
/// The initial native presentation is a smooth projective plane equation. A
/// generalized Weierstrass presentation is retained as a distinguished plane
/// model so later elliptic-curve operations can recover its coefficients
/// without pattern matching a polynomial.
#[derive(Clone, PartialEq, Eq)]
pub struct Curve<R: Field> {
    scheme: Scheme<R>,
    equation: Polynomial<R>,
    presentation: CurvePresentation<R>,
    evidence: CurveSmoothnessEvidence,
}

impl<R: Field> Curve<R> {
    /// Check one homogeneous plane equation by exact dimension and Jacobian tests.
    pub fn smooth_projective_plane(equation: Polynomial<R>) -> Result<Self, CurveError> {
        Self::from_plane_equation(equation, CurvePresentation::Plane)
    }

    /// Construct and check a generalized Weierstrass curve with coordinates `(x,y,z)`.
    pub fn smooth_projective_weierstrass(
        base: R,
        coefficients: [R::Element; 5],
    ) -> Result<Self, CurveError> {
        if coefficients
            .iter()
            .any(|coefficient| !base.contains(coefficient))
        {
            return Err(ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "Weierstrass coefficient from another field context".to_owned(),
            }
            .into());
        }
        if weierstrass_discriminant(&base, &coefficients)? == base.zero() {
            return Err(CurveError::SingularWeierstrassEquation);
        }
        let ring = PolynomialRing::new(
            base.clone(),
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )?;
        let one = base.one();
        let minus_one = base.neg(&one)?;
        let equation = ring.from_terms([
            (small_exponents([0, 2, 1]), one),
            (small_exponents([1, 1, 1]), coefficients[0].clone()),
            (small_exponents([0, 1, 2]), coefficients[2].clone()),
            (small_exponents([3, 0, 0]), minus_one),
            (small_exponents([2, 0, 1]), base.neg(&coefficients[1])?),
            (small_exponents([1, 0, 2]), base.neg(&coefficients[3])?),
            (small_exponents([0, 0, 3]), base.neg(&coefficients[4])?),
        ])?;
        Self::from_plane_equation(equation, CurvePresentation::Weierstrass { coefficients })
    }

    fn from_plane_equation(
        equation: Polynomial<R>,
        presentation: CurvePresentation<R>,
    ) -> Result<Self, CurveError> {
        let ring = equation.ring().clone();
        if ring.variables().len() != 3
            || equation.is_zero()
            || equation.homogeneous_degree()?.is_none()
            || !ring.grading().is_standard()
        {
            return Err(CurveError::InvalidPlaneEquation);
        }
        let ideal = ring.ideal([equation])?;
        if ideal.dimension()? != 2 {
            return Err(CurveError::NotOneDimensional);
        }
        let equation = ideal
            .generators()
            .first()
            .cloned()
            .ok_or(CurveError::InvalidPlaneEquation)?;
        let mut singular_generators = Vec::with_capacity(4);
        singular_generators.push(equation.clone());
        for variable in 0..3 {
            singular_generators.push(equation.derivative(variable)?);
        }
        let singular_cone_dimension = ring.ideal(singular_generators)?.dimension()?;
        if singular_cone_dimension > 0 {
            return Err(CurveError::SingularCurve);
        }
        let ambient = ProjectiveSpace::from_coordinate_ring(ring.clone())?;
        let scheme = projective_closed_subscheme(&ambient, ideal)?;
        Ok(Self {
            scheme,
            equation,
            presentation,
            evidence: CurveSmoothnessEvidence {
                quotient_dimension: 2,
                singular_cone_dimension,
            },
        })
    }

    /// Base coefficient field.
    pub fn base(&self) -> &R {
        self.scheme.base()
    }

    /// Checked underlying projective scheme.
    pub fn scheme(&self) -> &Scheme<R> {
        &self.scheme
    }

    /// Canonical homogeneous plane equation.
    pub fn equation(&self) -> &Polynomial<R> {
        &self.equation
    }

    /// Admitted presentation class.
    pub fn presentation_kind(&self) -> CurvePresentationKind {
        match self.presentation {
            CurvePresentation::Plane => CurvePresentationKind::SmoothProjectivePlane,
            CurvePresentation::Weierstrass { .. } => {
                CurvePresentationKind::SmoothProjectiveWeierstrass
            }
        }
    }

    /// Exact dimension and Jacobian evidence retained at construction.
    pub fn smoothness_evidence(&self) -> &CurveSmoothnessEvidence {
        &self.evidence
    }

    /// Generalized Weierstrass coefficients `(a1,a2,a3,a4,a6)`, when distinguished.
    pub fn weierstrass_coefficients(&self) -> Option<&[R::Element; 5]> {
        match &self.presentation {
            CurvePresentation::Plane => None,
            CurvePresentation::Weierstrass { coefficients } => Some(coefficients),
        }
    }

    /// Discriminant of the distinguished generalized Weierstrass model.
    ///
    /// A plain smooth plane presentation has no distinguished Weierstrass
    /// coefficients and therefore returns `None`.
    pub fn weierstrass_discriminant(&self) -> Result<Option<R::Element>, CurveError> {
        self.weierstrass_coefficients()
            .map(|coefficients| weierstrass_discriminant(self.base(), coefficients))
            .transpose()
    }

    /// Construct the canonical algebraic function field in a deterministic affine chart.
    pub fn function_field(&self) -> Result<CurveFunctionField<R>, CurveError> {
        CurveFunctionField::new(self.clone())
    }

    /// Validate one rational point on this curve.
    pub fn point<I>(&self, coordinates: I) -> Result<CurvePoint<R>, CurveError>
    where
        I: IntoIterator<Item = R::Element>,
    {
        let point = self.scheme.point(coordinates)?;
        Ok(CurvePoint {
            curve: self.clone(),
            point,
        })
    }

    /// Recompute the checked presentation and compare its canonical equation and evidence.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let rebuilt = Self::from_plane_equation(self.equation.clone(), self.presentation.clone())
            .map_err(|_| VerificationError::InvalidWitness {
            claim: "curve presentation passes dimension and Jacobian checks",
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "curve presentation and smoothness evidence recompute canonically",
            })
        }
    }

    /// Serialize the checked presentation and its exact coefficient context.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        let coefficients = match &self.presentation {
            CurvePresentation::Plane => serde_json::Value::Null,
            CurvePresentation::Weierstrass { coefficients } => {
                serde_json::to_value(coefficients).map_err(serialization_error)?
            }
        };
        encode_typed_json(
            Self::TYPE_ID,
            [
                (
                    "equation",
                    serde_json::to_value(&self.equation).map_err(serialization_error)?,
                ),
                (
                    "presentation",
                    serde_json::Value::String(match self.presentation {
                        CurvePresentation::Plane => "smooth_projective_plane".to_owned(),
                        CurvePresentation::Weierstrass { .. } => {
                            "smooth_projective_weierstrass".to_owned()
                        }
                    }),
                ),
                ("weierstrass_coefficients", coefficients),
            ],
        )
    }

    /// Restore and recheck a serialized smooth projective curve.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: DeserializeOwned,
        R::Element: DeserializeOwned,
    {
        let mut object = decode_typed_json(
            value,
            Self::TYPE_ID,
            &["equation", "presentation", "weierstrass_coefficients"],
        )?;
        let equation: Polynomial<R> = decode_field(&mut object, "equation")?;
        let presentation: String = decode_field(&mut object, "presentation")?;
        let coefficients: Option<[R::Element; 5]> =
            decode_field(&mut object, "weierstrass_coefficients")?;
        let rebuilt = match (presentation.as_str(), coefficients) {
            ("smooth_projective_plane", None) => Self::smooth_projective_plane(equation.clone()),
            ("smooth_projective_weierstrass", Some(coefficients)) => {
                Self::smooth_projective_weierstrass(
                    equation.ring().coefficient_ring().clone(),
                    coefficients,
                )
            }
            _ => Err(CurveError::InvalidWitness(
                "serialized curve presentation and coefficients agree",
            )),
        }
        .map_err(curve_serialization_error)?;
        if rebuilt.equation() != &equation {
            return Err(NoetherError::Serialization(
                "serialized curve equation is not canonical for its presentation".to_owned(),
            ));
        }
        Ok(rebuilt)
    }
}

impl<R: Field> fmt::Debug for Curve<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Curve")
            .field("base", &R::TYPE_ID)
            .field("presentation", &self.presentation_kind())
            .field("degree", &self.equation.degree(None).ok().flatten())
            .finish()
    }
}

impl<R: Field> fmt::Display for Curve<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Curve<{}>({:?})",
            R::TYPE_ID,
            self.presentation_kind()
        )
    }
}

impl<R: Field> SemanticType for Curve<R> {
    const TYPE_ID: &'static str = "noether/commalg/Curve@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.equation.canonical_bytes()?;
        output.push(match self.presentation {
            CurvePresentation::Plane => 0,
            CurvePresentation::Weierstrass { .. } => 1,
        });
        if let CurvePresentation::Weierstrass { coefficients } = &self.presentation {
            for coefficient in coefficients {
                append_component(&mut output, &coefficient.canonical_bytes()?)?;
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CURVE_DESCRIPTION
    }
}

/// A checked morphism between projective plane curves.
///
/// Three equal-degree homogeneous source sections define the target
/// coordinates. Construction verifies absence of projective base points and
/// exact satisfaction of the target equation. A nonconstant map between the
/// admitted smooth projective curves is finite, so closed-point pullback is
/// computed scheme-theoretically where its inverse image is zero-dimensional.
#[derive(Clone, PartialEq, Eq)]
pub struct CurveMorphism<R: Field> {
    source: Curve<R>,
    target: Curve<R>,
    coordinate_images: [Polynomial<R>; 3],
    coordinate_degree: Integer,
}

impl<R: Field> CurveMorphism<R> {
    /// Construct from three equal-degree homogeneous coordinate images.
    pub fn new(
        source: Curve<R>,
        target: Curve<R>,
        mut coordinate_images: [Polynomial<R>; 3],
    ) -> Result<Self, CurveError> {
        if source.base() != target.base() {
            return Err(ContextMismatchError {
                expected: source.base().canonical_bytes().map_or_else(
                    |_| R::TYPE_ID.to_owned(),
                    |bytes| format!("{}:{} bytes", R::TYPE_ID, bytes.len()),
                ),
                actual: "curve morphism target has another base context".to_owned(),
            }
            .into());
        }
        let ring = source.equation.ring();
        if coordinate_images
            .iter()
            .any(|image| image.ring() != ring || image.is_zero())
        {
            return Err(CurveError::InvalidWitness(
                "curve-morphism coordinates are nonzero sections in the source ring",
            ));
        }
        let degrees = coordinate_images
            .iter()
            .map(Polynomial::homogeneous_degree)
            .collect::<Result<Vec<_>, _>>()?;
        let coordinate_degree = degrees[0]
            .clone()
            .filter(|degree| degree.is_positive())
            .ok_or(CurveError::InvalidWitness(
                "curve-morphism coordinates are positive-degree homogeneous sections",
            ))?;
        if degrees
            .iter()
            .any(|degree| degree.as_ref() != Some(&coordinate_degree))
        {
            return Err(CurveError::InvalidWitness(
                "curve-morphism coordinate sections have one degree",
            ));
        }
        let leading =
            coordinate_images[0]
                .leading_coefficient()
                .ok_or(CurveError::InvalidWitness(
                    "a curve-morphism coordinate has a leading coefficient",
                ))?;
        let normalization = source.base().inverse(&leading).map_err(|_| {
            CurveError::InvalidWitness("a nonzero coordinate leading coefficient is invertible")
        })?;
        coordinate_images = coordinate_images
            .map(|image| image.scalar_mul(&normalization))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| CurveError::InvalidWitness("a curve morphism has three coordinates"))?;
        let coordinate_ideal = ring.ideal(coordinate_images.iter().cloned())?;
        let base_locus = source.scheme.defining_ideal().sum(&coordinate_ideal)?;
        if base_locus.dimension()? > 0 {
            return Err(CurveError::InvalidWitness(
                "curve-morphism coordinates have no common projective zero on the source curve",
            ));
        }
        let pulled_equation =
            evaluate_polynomial_at_map(&target.equation, ring, &coordinate_images)?;
        if !source.scheme.defining_ideal().contains(&pulled_equation)? {
            return Err(CurveError::InvalidWitness(
                "curve-morphism coordinates satisfy the target equation on the source",
            ));
        }
        Ok(Self {
            source,
            target,
            coordinate_images,
            coordinate_degree,
        })
    }

    /// Identity morphism.
    pub fn identity(curve: Curve<R>) -> Result<Self, CurveError> {
        let ring = curve.equation.ring().clone();
        Self::new(
            curve.clone(),
            curve,
            [ring.generator(0)?, ring.generator(1)?, ring.generator(2)?],
        )
    }

    /// Source curve.
    pub fn source(&self) -> &Curve<R> {
        &self.source
    }

    /// Target curve.
    pub fn target(&self) -> &Curve<R> {
        &self.target
    }

    /// Target homogeneous coordinates as source sections.
    pub fn coordinate_images(&self) -> &[Polynomial<R>; 3] {
        &self.coordinate_images
    }

    /// Common degree of the coordinate sections.
    pub fn coordinate_degree(&self) -> &Integer {
        &self.coordinate_degree
    }

    /// Recheck base-point freeness and target incidence.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let rebuilt = Self::new(
            self.source.clone(),
            self.target.clone(),
            self.coordinate_images.clone(),
        )
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "curve morphism is base-point free and lands on its target",
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "curve-morphism coordinate presentation is canonical",
            })
        }
    }

    /// Serialize both endpoint curves and the checked homogeneous sections.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        encode_typed_json(
            Self::TYPE_ID,
            [
                ("source", nested_json(self.source.to_json()?)?),
                ("target", nested_json(self.target.to_json()?)?),
                (
                    "coordinate_images",
                    serde_json::to_value(&self.coordinate_images).map_err(serialization_error)?,
                ),
            ],
        )
    }

    /// Restore and recheck a serialized curve morphism.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: DeserializeOwned,
        R::Element: DeserializeOwned,
    {
        let mut object = decode_typed_json(
            value,
            Self::TYPE_ID,
            &["source", "target", "coordinate_images"],
        )?;
        let source = Curve::<R>::from_json(&encode_nested_field(&mut object, "source")?)?;
        let target = Curve::<R>::from_json(&encode_nested_field(&mut object, "target")?)?;
        let coordinate_images: [Polynomial<R>; 3] = decode_field(&mut object, "coordinate_images")?;
        Self::new(source, target, coordinate_images).map_err(curve_serialization_error)
    }
}

impl<R: Field> fmt::Debug for CurveMorphism<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CurveMorphism")
            .field("base", &R::TYPE_ID)
            .field("coordinate_degree", &self.coordinate_degree)
            .finish()
    }
}

impl<R: Field> fmt::Display for CurveMorphism<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CurveMorphism<{}>(coordinate_degree={})",
            R::TYPE_ID,
            self.coordinate_degree
        )
    }
}

impl<R: Field> SemanticType for CurveMorphism<R> {
    const TYPE_ID: &'static str = "noether/commalg/CurveMorphism@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.source.canonical_bytes()?;
        append_component(&mut output, &self.target.canonical_bytes()?)?;
        for image in &self.coordinate_images {
            append_component(&mut output, &image.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CURVE_MORPHISM_DESCRIPTION
    }
}

fn evaluate_polynomial_at_map<R: Field>(
    polynomial: &Polynomial<R>,
    target_ring: &PolynomialRing<R>,
    images: &[Polynomial<R>; 3],
) -> Result<Polynomial<R>, CurveError> {
    if polynomial.ring().coefficient_ring() != target_ring.coefficient_ring()
        || images.iter().any(|image| image.ring() != target_ring)
    {
        return Err(ContextMismatchError {
            expected: target_ring.to_string(),
            actual: "polynomial map has incompatible coefficient or image contexts".to_owned(),
        }
        .into());
    }
    let mut result = target_ring.zero();
    for (exponent, coefficient) in polynomial.terms() {
        let mut term = target_ring.constant(coefficient.clone())?;
        for (image, exponent) in images.iter().zip(exponent) {
            term = (&term * &image.pow_integer(exponent)?)?;
        }
        result = (&result + &term)?;
    }
    Ok(result)
}

/// A checked rational point on one exact curve.
#[derive(Clone, PartialEq, Eq)]
pub struct CurvePoint<R: Field> {
    curve: Curve<R>,
    point: SchemePoint<R>,
}

impl<R: Field> CurvePoint<R> {
    /// Owning curve.
    pub fn curve(&self) -> &Curve<R> {
        &self.curve
    }

    /// Canonically normalized projective coordinates.
    pub fn coordinates(&self) -> &[R::Element] {
        self.point.coordinates()
    }

    /// Checked underlying scheme point.
    pub fn scheme_point(&self) -> &SchemePoint<R> {
        &self.point
    }

    /// Embed this rational point as a degree-one closed point.
    pub fn as_closed_point(&self) -> Result<CurveClosedPoint<R>, CurveError> {
        let ideal = rational_point_prime_ideal(self)?;
        let prime = PrimeIdeal::from_verified_parts(ideal, b"rational projective point".to_vec());
        CurveClosedPoint::from_verified_prime(self.curve.clone(), prime, Some(self.clone()))
    }

    /// Recheck membership and projective normalization.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.curve.verify()?;
        self.point
            .verify()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "curve point satisfies its checked projective scheme",
            })?;
        if self.point.scheme() == self.curve.scheme() {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "curve point and curve share one scheme presentation",
            })
        }
    }

    /// Serialize the owning curve and normalized projective coordinates.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        encode_typed_json(
            Self::TYPE_ID,
            [
                ("curve", nested_json(self.curve.to_json()?)?),
                (
                    "coordinates",
                    serde_json::to_value(self.coordinates()).map_err(serialization_error)?,
                ),
            ],
        )
    }

    /// Restore and recheck a normalized rational point.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: DeserializeOwned,
        R::Element: DeserializeOwned,
    {
        let mut object = decode_typed_json(value, Self::TYPE_ID, &["curve", "coordinates"])?;
        let curve = Curve::<R>::from_json(&encode_nested_field(&mut object, "curve")?)?;
        let coordinates: Vec<R::Element> = decode_field(&mut object, "coordinates")?;
        curve.point(coordinates).map_err(curve_serialization_error)
    }
}

fn rational_point_prime_ideal<R: Field>(
    point: &CurvePoint<R>,
) -> Result<super::Ideal<PolynomialRing<R>>, CurveError> {
    let ring = point.curve.equation.ring();
    let zero = point.curve.base().zero();
    let pivot = point
        .coordinates()
        .iter()
        .position(|coordinate| coordinate != &zero)
        .ok_or(CurveError::InvalidWitness(
            "a checked projective point has a nonzero coordinate",
        ))?;
    let pivot_generator = ring.generator(pivot)?;
    let mut generators = Vec::with_capacity(2);
    for coordinate in 0..3 {
        if coordinate == pivot {
            continue;
        }
        let scaled = pivot_generator.scalar_mul(&point.coordinates()[coordinate])?;
        generators.push((&ring.generator(coordinate)? - &scaled)?);
    }
    ring.ideal(generators).map_err(Into::into)
}

impl<R: Field> fmt::Debug for CurvePoint<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CurvePoint")
            .field("base", &R::TYPE_ID)
            .field("coordinates", &self.coordinates().len())
            .finish()
    }
}

impl<R: Field> fmt::Display for CurvePoint<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "CurvePoint<{}>", R::TYPE_ID)
    }
}

impl<R: Field> SemanticType for CurvePoint<R> {
    const TYPE_ID: &'static str = "noether/commalg/CurvePoint@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.curve.canonical_bytes()?;
        for coordinate in self.coordinates() {
            append_component(&mut output, &coordinate.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CURVE_POINT_DESCRIPTION
    }
}

/// A closed point (place) on a checked projective curve.
///
/// The point is represented by its homogeneous prime ideal. Its degree is the
/// exact standard-graded degree of that ideal. Degree-one points optionally
/// retain their checked rational coordinates for local evaluation.
#[derive(Clone)]
pub struct CurveClosedPoint<R: Field> {
    curve: Curve<R>,
    prime_ideal: PrimeIdeal<PolynomialRing<R>>,
    degree: Integer,
    rational_point: Option<CurvePoint<R>>,
}

impl<R: Field> CurveClosedPoint<R> {
    /// Construct from a previously certified homogeneous prime ideal.
    pub fn from_prime_ideal(
        curve: Curve<R>,
        prime_ideal: PrimeIdeal<PolynomialRing<R>>,
    ) -> Result<Self, CurveError> {
        Self::from_verified_prime(curve, prime_ideal, None)
    }

    fn from_verified_prime(
        curve: Curve<R>,
        prime_ideal: PrimeIdeal<PolynomialRing<R>>,
        mut rational_point: Option<CurvePoint<R>>,
    ) -> Result<Self, CurveError> {
        let ideal = prime_ideal.ideal();
        if ideal.ring() != curve.equation.ring() {
            return Err(ContextMismatchError {
                expected: curve.equation.ring().to_string(),
                actual: ideal.ring().to_string(),
            }
            .into());
        }
        if ideal.generators().iter().any(|generator| {
            generator
                .homogeneous_degree()
                .map_or(true, |degree| degree.is_none() && !generator.is_zero())
        }) {
            return Err(CurveError::InvalidWitness(
                "a projective closed-point prime ideal is homogeneous",
            ));
        }
        if ideal.dimension()? != 1 || !ideal.contains(&curve.equation)? {
            return Err(CurveError::InvalidWitness(
                "the homogeneous prime ideal cuts out a closed point on the curve",
            ));
        }
        let degree = ideal.degree()?;
        if !degree.is_positive() {
            return Err(CurveError::InvalidWitness(
                "a curve closed point has positive degree",
            ));
        }
        if rational_point.is_none() && degree == Integer::one() {
            rational_point = Some(recover_degree_one_point(&curve, ideal)?);
        }
        if let Some(point) = &rational_point {
            point.verify().map_err(|_| {
                CurveError::InvalidWitness("retained rational coordinates verify on the curve")
            })?;
            if point.curve() != &curve || degree != Integer::one() {
                return Err(CurveError::InvalidWitness(
                    "retained rational coordinates describe this degree-one point",
                ));
            }
            if rational_point_prime_ideal(point)? != *ideal {
                return Err(CurveError::InvalidWitness(
                    "retained rational coordinates define the supplied degree-one prime",
                ));
            }
        }
        Ok(Self {
            curve,
            prime_ideal,
            degree,
            rational_point,
        })
    }

    /// Owning curve.
    pub fn curve(&self) -> &Curve<R> {
        &self.curve
    }

    /// Homogeneous prime ideal defining the point.
    pub fn prime_ideal(&self) -> &PrimeIdeal<PolynomialRing<R>> {
        &self.prime_ideal
    }

    /// Residue degree over the constant field.
    pub fn degree(&self) -> &Integer {
        &self.degree
    }

    /// Retained rational coordinates, when this point was embedded from them.
    pub fn rational_point(&self) -> Option<&CurvePoint<R>> {
        self.rational_point.as_ref()
    }

    /// Recheck incidence, dimension, degree, and any rational coordinates.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let rebuilt = Self::from_verified_prime(
            self.curve.clone(),
            self.prime_ideal.clone(),
            self.rational_point.clone(),
        )
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "closed point is a homogeneous prime of projective dimension zero on its curve",
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "closed-point degree and presentation recompute canonically",
            })
        }
    }
}

fn recover_degree_one_point<R: Field>(
    curve: &Curve<R>,
    ideal: &super::Ideal<PolynomialRing<R>>,
) -> Result<CurvePoint<R>, CurveError> {
    let projective_ring = curve.equation.ring();
    let chart = (0..3)
        .find_map(|coordinate| {
            projective_ring
                .generator(coordinate)
                .ok()
                .and_then(|generator| {
                    ideal
                        .contains(&generator)
                        .ok()
                        .map(|inside| (inside, coordinate))
                })
                .and_then(|(inside, coordinate)| (!inside).then_some(coordinate))
        })
        .ok_or(CurveError::InvalidWitness(
            "a projective degree-one prime meets an affine chart",
        ))?;
    let affine_ring = PolynomialRing::new(
        curve.base().clone(),
        projective_ring
            .variables()
            .iter()
            .enumerate()
            .filter(|(coordinate, _)| *coordinate != chart)
            .map(|(_, name)| name.clone()),
        MonomialOrder::GradedReverseLexicographic,
    )?;
    let affine_ideal = affine_ring.ideal(
        ideal
            .generators()
            .iter()
            .map(|generator| dehomogenize_to_chart(generator, &affine_ring, chart))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    if affine_ideal.dimension()? != 0 || affine_ideal.degree()? != Integer::one() {
        return Err(CurveError::InvalidWitness(
            "a degree-one projective prime has a degree-one affine chart",
        ));
    }
    let mut coordinates = vec![curve.base().zero(); 3];
    coordinates[chart] = curve.base().one();
    let mut affine_coordinate = 0;
    for (projective_coordinate, coordinate_value) in coordinates.iter_mut().enumerate() {
        if projective_coordinate == chart {
            continue;
        }
        let generator = affine_ring.generator(affine_coordinate)?;
        let normal = affine_ideal.normal_form(&generator)?;
        let constant = normal.constant_coefficient();
        if normal != affine_ring.constant(constant.clone())? {
            return Err(CurveError::InvalidWitness(
                "a degree-one affine residue class is constant",
            ));
        }
        *coordinate_value = constant;
        affine_coordinate += 1;
    }
    let point = curve.point(coordinates)?;
    if rational_point_prime_ideal(&point)? != *ideal {
        return Err(CurveError::InvalidWitness(
            "recovered degree-one coordinates define the supplied prime",
        ));
    }
    Ok(point)
}

impl<R: Field> PartialEq for CurveClosedPoint<R> {
    fn eq(&self, other: &Self) -> bool {
        self.curve == other.curve && self.prime_ideal == other.prime_ideal
    }
}

impl<R: Field> Eq for CurveClosedPoint<R> {}

impl<R: Field> fmt::Debug for CurveClosedPoint<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CurveClosedPoint")
            .field("base", &R::TYPE_ID)
            .field("degree", &self.degree)
            .field("rational", &self.rational_point.is_some())
            .finish()
    }
}

impl<R: Field> fmt::Display for CurveClosedPoint<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CurveClosedPoint<{}>(degree={})",
            R::TYPE_ID,
            self.degree
        )
    }
}

impl<R: Field> SemanticType for CurveClosedPoint<R> {
    const TYPE_ID: &'static str = "noether/commalg/CurveClosedPoint@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.curve.canonical_bytes()?;
        append_component(&mut output, &self.prime_ideal.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CURVE_CLOSED_POINT_DESCRIPTION
    }
}

fn closed_point_to_json<R>(point: &CurveClosedPoint<R>) -> Result<String, NoetherError>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    encode_typed_json(
        CurveClosedPoint::<R>::TYPE_ID,
        [
            ("curve", nested_json(point.curve.to_json()?)?),
            (
                "defining_ideal",
                serde_json::to_value(point.prime_ideal.ideal()).map_err(serialization_error)?,
            ),
        ],
    )
}

fn decode_closed_point_parts<R>(
    value: &str,
) -> Result<(Curve<R>, super::Ideal<PolynomialRing<R>>), NoetherError>
where
    R: Field + DeserializeOwned,
    R::Element: DeserializeOwned,
{
    let mut object = decode_typed_json(
        value,
        CurveClosedPoint::<R>::TYPE_ID,
        &["curve", "defining_ideal"],
    )?;
    let curve = Curve::<R>::from_json(&encode_nested_field(&mut object, "curve")?)?;
    let ideal = decode_field(&mut object, "defining_ideal")?;
    Ok((curve, ideal))
}

fn closed_point_from_degree_one_json<R>(value: &str) -> Result<CurveClosedPoint<R>, NoetherError>
where
    R: Field + DeserializeOwned,
    R::Element: DeserializeOwned,
{
    let (curve, ideal) = decode_closed_point_parts(value)?;
    if ideal
        .degree()
        .map_err(|error| curve_serialization_error(error.into()))?
        != Integer::one()
    {
        return Err(NoetherError::Serialization(
            "closed points over this coefficient context deserialize only in degree one".to_owned(),
        ));
    }
    CurveClosedPoint::from_verified_prime(
        curve,
        PrimeIdeal::from_verified_parts(ideal, b"recovered rational projective point".to_vec()),
        None,
    )
    .map_err(curve_serialization_error)
}

fn closed_point_from_factorized_json<R, F>(
    value: &str,
    factor: F,
) -> Result<CurveClosedPoint<R>, NoetherError>
where
    R: Field + DeserializeOwned,
    R::Element: DeserializeOwned,
    F: Fn(&Polynomial<R>) -> Result<PolynomialFactorization<R>, PolynomialFactorizationError>,
{
    let (curve, ideal) = decode_closed_point_parts(value)?;
    if ideal
        .degree()
        .map_err(|error| curve_serialization_error(error.into()))?
        == Integer::one()
    {
        return CurveClosedPoint::from_verified_prime(
            curve,
            PrimeIdeal::from_verified_parts(ideal, b"recovered rational projective point".to_vec()),
            None,
        )
        .map_err(curve_serialization_error);
    }
    certify_closed_point_prime(curve, ideal, factor).map_err(curve_serialization_error)
}

fn certify_closed_point_prime<R, F>(
    curve: Curve<R>,
    ideal: super::Ideal<PolynomialRing<R>>,
    factor: F,
) -> Result<CurveClosedPoint<R>, CurveError>
where
    R: Field,
    F: Fn(&Polynomial<R>) -> Result<PolynomialFactorization<R>, PolynomialFactorizationError>,
{
    let projective_ring = curve.equation.ring();
    let chart = (0..3)
        .find_map(|coordinate| {
            let generator = projective_ring.generator(coordinate).ok()?;
            (!ideal.contains(&generator).ok()?).then_some(coordinate)
        })
        .ok_or(CurveError::InvalidWitness(
            "a projective closed point meets an affine chart",
        ))?;
    let affine_ring = PolynomialRing::new(
        curve.base().clone(),
        projective_ring
            .variables()
            .iter()
            .enumerate()
            .filter(|(coordinate, _)| *coordinate != chart)
            .map(|(_, name)| name.clone()),
        MonomialOrder::GradedReverseLexicographic,
    )?;
    let affine_ideal = affine_ring.ideal(
        ideal
            .generators()
            .iter()
            .map(|generator| dehomogenize_to_chart(generator, &affine_ring, chart))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    if affine_ideal.dimension()? != 0 {
        return Err(CurveError::InvalidWitness(
            "a projective closed point has a zero-dimensional affine chart",
        ));
    }
    let representation = affine_ideal.rational_univariate_representation()?;
    representation
        .verify()
        .map_err(|_| CurveError::InvalidWitness("the affine point RUR verifies"))?;
    if !representation.is_reduced() {
        return Err(CurveError::InvalidWitness(
            "a closed-point residue algebra is reduced",
        ));
    }
    let factorization = factor(representation.equation())?;
    factorization.verify()?;
    let [(irreducible, multiplicity)] = factorization.factors() else {
        return Err(CurveError::InvalidWitness(
            "a closed-point residue algebra has one irreducible parameter factor",
        ));
    };
    if *multiplicity != 1 {
        return Err(CurveError::InvalidWitness(
            "a closed-point parameter factor has multiplicity one",
        ));
    }
    let factor_degree = irreducible
        .degree(Some(0))?
        .ok_or(CurveError::InvalidWitness(
            "a closed-point parameter factor is nonconstant",
        ))?;
    if integer_to_usize(&factor_degree)? != representation.quotient_dimension()
        || ideal.degree()? != factor_degree
    {
        return Err(CurveError::InvalidWitness(
            "the irreducible affine residue field accounts for the complete projective degree",
        ));
    }
    CurveClosedPoint::from_verified_prime(
        curve,
        PrimeIdeal::from_verified_parts(
            ideal,
            b"verified irreducible affine RUR residue field".to_vec(),
        ),
        None,
    )
}

macro_rules! impl_factorized_closed_point_json {
    ($field:ty) => {
        impl CurveClosedPoint<$field> {
            /// Serialize this place by its checked curve and homogeneous prime ideal.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                closed_point_to_json(self)
            }

            /// Restore and prove maximality in a deterministic affine chart.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                closed_point_from_factorized_json(value, |polynomial: &Polynomial<$field>| {
                    polynomial.factor()
                })
            }
        }
    };
}

impl_factorized_closed_point_json!(RationalField);
impl_factorized_closed_point_json!(FiniteField);

macro_rules! impl_degree_one_closed_point_json {
    ($field:ty) => {
        impl CurveClosedPoint<$field> {
            /// Serialize a degree-one place over this exact function field.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                if self.degree() != &Integer::one() {
                    return Err(NoetherError::Serialization(
                        "non-rational closed-point serialization requires coefficient-field factorization"
                            .to_owned(),
                    ));
                }
                closed_point_to_json(self)
            }

            /// Restore and prove a degree-one place from its rational coordinates.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                closed_point_from_degree_one_json(value)
            }
        }
    };
}

impl_degree_one_closed_point_json!(RationalFunctionField<RationalField>);
impl_degree_one_closed_point_json!(RationalFunctionField<FiniteField>);
impl_degree_one_closed_point_json!(NumberField);

/// A canonical Weil divisor with arbitrary-precision integral coefficients.
#[derive(Clone, PartialEq, Eq)]
pub struct WeilDivisor<R: Field> {
    curve: Curve<R>,
    terms: Vec<(CurveClosedPoint<R>, Integer)>,
}

impl<R: Field> WeilDivisor<R> {
    /// Construct and canonically combine a finite family of point coefficients.
    pub fn new<I>(curve: Curve<R>, terms: I) -> Result<Self, CurveError>
    where
        I: IntoIterator<Item = (CurveClosedPoint<R>, Integer)>,
    {
        let mut combined: Vec<(CurveClosedPoint<R>, Integer)> = Vec::new();
        for (point, coefficient) in terms {
            if point.curve() != &curve {
                return Err(ContextMismatchError {
                    expected: Curve::<R>::TYPE_ID.to_owned(),
                    actual: "divisor point lies on another curve".to_owned(),
                }
                .into());
            }
            if coefficient.is_zero() {
                continue;
            }
            if let Some((_, existing)) = combined.iter_mut().find(|(other, _)| other == &point) {
                *existing = &*existing + &coefficient;
            } else {
                combined.push((point, coefficient));
            }
        }
        combined.retain(|(_, coefficient)| !coefficient.is_zero());
        let mut keyed = combined
            .into_iter()
            .map(|term| -> Result<_, CurveError> {
                let key = term.0.canonical_bytes().map_err(|_| {
                    CurveError::InvalidWitness("closed-point identity is serializable")
                })?;
                Ok((key, term))
            })
            .collect::<Result<Vec<_>, _>>()?;
        keyed.sort_by(|(left, _), (right, _)| left.cmp(right));
        let combined = keyed.into_iter().map(|(_, term)| term).collect();
        Ok(Self {
            curve,
            terms: combined,
        })
    }

    /// Zero divisor on one curve.
    pub fn zero(curve: Curve<R>) -> Self {
        Self {
            curve,
            terms: Vec::new(),
        }
    }

    /// Owning curve.
    pub fn curve(&self) -> &Curve<R> {
        &self.curve
    }

    /// Deterministically ordered nonzero point coefficients.
    pub fn terms(&self) -> &[(CurveClosedPoint<R>, Integer)] {
        &self.terms
    }

    /// Closed support in deterministic order.
    pub fn support(&self) -> impl Iterator<Item = &CurveClosedPoint<R>> {
        self.terms.iter().map(|(point, _)| point)
    }

    /// Coefficient of one closed point.
    pub fn coefficient(&self, point: &CurveClosedPoint<R>) -> Result<Integer, CurveError> {
        if point.curve() != &self.curve {
            return Err(ContextMismatchError {
                expected: Curve::<R>::TYPE_ID.to_owned(),
                actual: "divisor coefficient requested at a point on another curve".to_owned(),
            }
            .into());
        }
        Ok(self
            .terms
            .iter()
            .find_map(|(candidate, coefficient)| (candidate == point).then(|| coefficient.clone()))
            .unwrap_or_else(Integer::zero))
    }

    /// Exact divisor degree `sum n_P [k(P):K]`.
    pub fn degree(&self) -> Integer {
        self.terms
            .iter()
            .fold(Integer::zero(), |degree, (point, coefficient)| {
                degree + coefficient * point.degree()
            })
    }

    /// Effective positive part.
    pub fn positive_part(&self) -> Self {
        Self {
            curve: self.curve.clone(),
            terms: self
                .terms
                .iter()
                .filter(|(_, coefficient)| coefficient.is_positive())
                .cloned()
                .collect(),
        }
    }

    /// Effective negative part, with positive coefficients, so `D=D+ - D-`.
    pub fn negative_part(&self) -> Self {
        Self {
            curve: self.curve.clone(),
            terms: self
                .terms
                .iter()
                .filter(|(_, coefficient)| coefficient.is_negative())
                .map(|(point, coefficient)| (point.clone(), -coefficient))
                .collect(),
        }
    }

    /// Scale every coefficient by an arbitrary-precision integer.
    pub fn scale(&self, scalar: &Integer) -> Self {
        Self {
            curve: self.curve.clone(),
            terms: self
                .terms
                .iter()
                .filter_map(|(point, coefficient)| {
                    let coefficient = coefficient * scalar;
                    (!coefficient.is_zero()).then(|| (point.clone(), coefficient))
                })
                .collect(),
        }
    }

    /// Add another divisor on the same curve.
    pub fn checked_add(&self, other: &Self) -> Result<Self, CurveError> {
        if self.curve != other.curve {
            return Err(ContextMismatchError {
                expected: Curve::<R>::TYPE_ID.to_owned(),
                actual: "divisor belongs to another curve".to_owned(),
            }
            .into());
        }
        Self::new(
            self.curve.clone(),
            self.terms
                .iter()
                .cloned()
                .chain(other.terms.iter().cloned()),
        )
    }

    /// Subtract another divisor on the same curve.
    pub fn checked_sub(&self, other: &Self) -> Result<Self, CurveError> {
        self.checked_add(&other.scale(&Integer::from(-1)))
    }

    /// Negate all coefficients.
    pub fn negated(&self) -> Self {
        self.scale(&Integer::from(-1))
    }

    /// Compute the exact Riemann--Roch space `L(D)` on this smooth plane curve.
    pub fn riemann_roch_space(&self) -> Result<RiemannRochSpace<R>, CurveError> {
        compute_riemann_roch_space(self)
    }

    /// Recheck every point, coefficient normalization, ordering, and degree data.
    pub fn verify(&self) -> Result<(), VerificationError> {
        for (point, coefficient) in &self.terms {
            point.verify()?;
            if point.curve() != &self.curve || coefficient.is_zero() {
                return Err(VerificationError::InvalidWitness {
                    claim: "divisor support belongs to one curve with nonzero coefficients",
                });
            }
        }
        let rebuilt = Self::new(self.curve.clone(), self.terms.clone()).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "divisor terms canonically combine",
            }
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "divisor support is deterministically ordered and combined",
            })
        }
    }
}

impl<R: Field> fmt::Debug for WeilDivisor<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeilDivisor")
            .field("base", &R::TYPE_ID)
            .field("support", &self.terms.len())
            .field("degree", &self.degree())
            .finish()
    }
}

impl<R: Field> fmt::Display for WeilDivisor<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "WeilDivisor<{}>(support={}, degree={})",
            R::TYPE_ID,
            self.terms.len(),
            self.degree()
        )
    }
}

impl<R: Field> SemanticType for WeilDivisor<R> {
    const TYPE_ID: &'static str = "noether/commalg/WeilDivisor@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.curve.canonical_bytes()?;
        for (point, coefficient) in &self.terms {
            append_component(&mut output, &point.canonical_bytes()?)?;
            append_component(&mut output, &coefficient.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &WEIL_DIVISOR_DESCRIPTION
    }
}

fn divisor_to_json<R>(divisor: &WeilDivisor<R>) -> Result<String, NoetherError>
where
    R: Field + Serialize,
    R::Element: Serialize,
    CurveClosedPoint<R>: ClosedPointJson,
{
    let terms = divisor
        .terms
        .iter()
        .map(|(point, coefficient)| {
            Ok(serde_json::Value::Array(vec![
                nested_json(point.closed_point_to_json()?)?,
                serde_json::to_value(coefficient).map_err(serialization_error)?,
            ]))
        })
        .collect::<Result<Vec<_>, NoetherError>>()?;
    encode_typed_json(
        WeilDivisor::<R>::TYPE_ID,
        [
            ("curve", nested_json(divisor.curve.to_json()?)?),
            ("terms", serde_json::Value::Array(terms)),
        ],
    )
}

fn divisor_from_json<R>(value: &str) -> Result<WeilDivisor<R>, NoetherError>
where
    R: Field + Serialize + DeserializeOwned,
    R::Element: Serialize + DeserializeOwned,
    CurveClosedPoint<R>: ClosedPointJson,
{
    let original: serde_json::Value = serde_json::from_str(value).map_err(serialization_error)?;
    let mut object = decode_typed_json(value, WeilDivisor::<R>::TYPE_ID, &["curve", "terms"])?;
    let curve = Curve::<R>::from_json(&encode_nested_field(&mut object, "curve")?)?;
    let terms = object
        .remove("terms")
        .and_then(|value| value.as_array().cloned())
        .ok_or_else(|| {
            NoetherError::Serialization("serialized divisor terms must be an array".to_owned())
        })?;
    let terms = terms
        .into_iter()
        .map(|term| {
            let [point, coefficient]: [serde_json::Value; 2] = term
                .as_array()
                .cloned()
                .and_then(|values| values.try_into().ok())
                .ok_or_else(|| {
                    NoetherError::Serialization(
                        "each serialized divisor term must be a point-coefficient pair".to_owned(),
                    )
                })?;
            let point = CurveClosedPoint::<R>::closed_point_from_json(
                &serde_json::to_string(&point).map_err(serialization_error)?,
            )?;
            let coefficient = serde_json::from_value(coefficient).map_err(serialization_error)?;
            Ok((point, coefficient))
        })
        .collect::<Result<Vec<_>, NoetherError>>()?;
    let rebuilt = WeilDivisor::new(curve, terms).map_err(curve_serialization_error)?;
    let canonical: serde_json::Value =
        serde_json::from_str(&divisor_to_json(&rebuilt)?).map_err(serialization_error)?;
    if canonical != original {
        return Err(NoetherError::Serialization(
            "serialized divisor support is not canonically ordered and combined".to_owned(),
        ));
    }
    Ok(rebuilt)
}

trait ClosedPointJson: Sized {
    fn closed_point_to_json(&self) -> Result<String, NoetherError>;
    fn closed_point_from_json(value: &str) -> Result<Self, NoetherError>;
}

macro_rules! impl_curve_divisor_json {
    ($field:ty) => {
        impl ClosedPointJson for CurveClosedPoint<$field> {
            fn closed_point_to_json(&self) -> Result<String, NoetherError> {
                Self::to_json(self)
            }

            fn closed_point_from_json(value: &str) -> Result<Self, NoetherError> {
                Self::from_json(value)
            }
        }

        impl WeilDivisor<$field> {
            /// Serialize this canonically ordered finite divisor.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                divisor_to_json(self)
            }

            /// Restore every place, combine support, and reject noncanonical input.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                divisor_from_json(value)
            }
        }
    };
}

impl_curve_divisor_json!(RationalField);
impl_curve_divisor_json!(FiniteField);
impl_curve_divisor_json!(RationalFunctionField<RationalField>);
impl_curve_divisor_json!(RationalFunctionField<FiniteField>);
impl_curve_divisor_json!(NumberField);

impl<R: Field> Add<&WeilDivisor<R>> for &WeilDivisor<R> {
    type Output = Result<WeilDivisor<R>, CurveError>;

    fn add(self, rhs: &WeilDivisor<R>) -> Self::Output {
        self.checked_add(rhs)
    }
}

impl<R: Field> Sub<&WeilDivisor<R>> for &WeilDivisor<R> {
    type Output = Result<WeilDivisor<R>, CurveError>;

    fn sub(self, rhs: &WeilDivisor<R>) -> Self::Output {
        self.checked_sub(rhs)
    }
}

impl<R: Field> Neg for &WeilDivisor<R> {
    type Output = WeilDivisor<R>;

    fn neg(self) -> Self::Output {
        self.negated()
    }
}

/// A verified finite-dimensional Riemann--Roch space over the constant field.
///
/// The deterministic basis consists directly of curve functions. Private
/// equal-degree numerator/denominator sections retain the exact ideal-theoretic
/// construction witness used by [`verify`](Self::verify).
#[derive(Clone, PartialEq, Eq)]
pub struct RiemannRochSpace<R: Field> {
    divisor: WeilDivisor<R>,
    basis: Vec<CurveFunction<R>>,
    denominator_section: Polynomial<R>,
    numerator_sections: Vec<Polynomial<R>>,
}

impl<R: Field> RiemannRochSpace<R> {
    /// Source divisor.
    pub fn divisor(&self) -> &WeilDivisor<R> {
        &self.divisor
    }

    /// Dimension over the constant field.
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// Deterministically ordered basis in the curve function field.
    pub fn basis(&self) -> &[CurveFunction<R>] {
        &self.basis
    }

    /// Recompute the complete ideal-theoretic section calculation.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.divisor.verify()?;
        for function in &self.basis {
            function.verify()?;
            if function.field().curve() != self.divisor.curve() {
                return Err(VerificationError::InvalidWitness {
                    claim: "Riemann--Roch basis functions belong to the divisor's curve",
                });
            }
        }
        let rebuilt = compute_riemann_roch_space(&self.divisor).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "Riemann--Roch ideal quotient and section-space calculation recomputes",
            }
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "Riemann--Roch basis and dimension are canonical",
            })
        }
    }
}

impl<R: Field> fmt::Debug for RiemannRochSpace<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RiemannRochSpace")
            .field("base", &R::TYPE_ID)
            .field("dimension", &self.dimension())
            .field("divisor_degree", &self.divisor.degree())
            .finish()
    }
}

impl<R: Field> fmt::Display for RiemannRochSpace<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RiemannRochSpace<{}>(dimension={}, divisor_degree={})",
            R::TYPE_ID,
            self.dimension(),
            self.divisor.degree()
        )
    }
}

impl<R: Field> SemanticType for RiemannRochSpace<R> {
    const TYPE_ID: &'static str = "noether/commalg/RiemannRochSpace@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.divisor.canonical_bytes()?;
        append_component(&mut output, &self.denominator_section.canonical_bytes()?)?;
        for numerator in &self.numerator_sections {
            append_component(&mut output, &numerator.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &RIEMANN_ROCH_SPACE_DESCRIPTION
    }
}

fn riemann_roch_to_json<R>(space: &RiemannRochSpace<R>) -> Result<String, NoetherError>
where
    R: Field + Serialize,
    R::Element: Serialize,
    WeilDivisor<R>: DivisorJson,
{
    encode_typed_json(
        RiemannRochSpace::<R>::TYPE_ID,
        [("divisor", nested_json(space.divisor.divisor_to_json()?)?)],
    )
}

fn riemann_roch_from_json<R>(value: &str) -> Result<RiemannRochSpace<R>, NoetherError>
where
    R: Field,
    WeilDivisor<R>: DivisorJson,
{
    let mut object = decode_typed_json(value, RiemannRochSpace::<R>::TYPE_ID, &["divisor"])?;
    let divisor =
        WeilDivisor::<R>::divisor_from_json(&encode_nested_field(&mut object, "divisor")?)?;
    divisor
        .riemann_roch_space()
        .map_err(curve_serialization_error)
}

trait DivisorJson: Sized {
    fn divisor_to_json(&self) -> Result<String, NoetherError>;
    fn divisor_from_json(value: &str) -> Result<Self, NoetherError>;
}

macro_rules! impl_riemann_roch_json {
    ($field:ty) => {
        impl DivisorJson for WeilDivisor<$field> {
            fn divisor_to_json(&self) -> Result<String, NoetherError> {
                Self::to_json(self)
            }

            fn divisor_from_json(value: &str) -> Result<Self, NoetherError> {
                Self::from_json(value)
            }
        }

        impl RiemannRochSpace<$field> {
            /// Serialize the source divisor; the canonical basis is derived evidence.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                riemann_roch_to_json(self)
            }

            /// Restore the divisor and recompute the deterministic section basis.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                riemann_roch_from_json(value)
            }
        }
    };
}

impl_riemann_roch_json!(RationalField);
impl_riemann_roch_json!(FiniteField);
impl_riemann_roch_json!(RationalFunctionField<RationalField>);
impl_riemann_roch_json!(RationalFunctionField<FiniteField>);
impl_riemann_roch_json!(NumberField);

fn compute_riemann_roch_space<R: Field>(
    divisor: &WeilDivisor<R>,
) -> Result<RiemannRochSpace<R>, CurveError> {
    if let Some(space) = try_weierstrass_riemann_roch_space(divisor)? {
        return Ok(space);
    }
    let curve = divisor.curve();
    let ring = curve.equation.ring();
    let curve_ideal = curve.scheme.defining_ideal();
    let irrelevant = ring.ideal(ring.generators()?)?;
    let positive = divisor.positive_part();
    let negative = divisor.negative_part();
    let positive_ideal = effective_divisor_ideal(&positive, &irrelevant)?;
    let negative_ideal = effective_divisor_ideal(&negative, &irrelevant)?;

    let denominator_section = if positive.terms.is_empty() {
        ring.one()
    } else {
        let candidates = positive_ideal
            .generators()
            .iter()
            .filter_map(|generator| {
                let degree = generator.homogeneous_degree().ok().flatten()?;
                if generator.is_zero() || curve_ideal.contains(generator).ok()? || degree.sign() < 0
                {
                    return None;
                }
                Some((degree, generator.clone()))
            })
            .collect::<Vec<_>>();
        candidates
            .into_iter()
            .min_by(|(left, _), (right, _)| left.cmp(right))
            .map(|(_, generator)| generator)
            .ok_or(CurveError::InvalidWitness(
                "an effective divisor ideal contains a nonzero section off the curve ideal",
            ))?
    };
    let section_degree =
        denominator_section
            .homogeneous_degree()?
            .ok_or(CurveError::InvalidWitness(
                "the selected denominator section is homogeneous",
            ))?;
    let section_degree_u64 = integer_to_u64(&section_degree)?;
    let principal_section = ring.ideal([denominator_section.clone()])?;
    let complete_intersection = curve_ideal.sum(&principal_section)?;
    let residual = complete_intersection
        .colon(&positive_ideal)?
        .saturate(&irrelevant)?;
    let required_zeros = residual
        .product(&negative_ideal)?
        .sum(curve_ideal)?
        .saturate(&irrelevant)?;

    let section_basis = homogeneous_ideal_component(&required_zeros, section_degree_u64)?;
    let numerator_sections =
        quotient_section_basis(curve_ideal, section_basis, section_degree_u64)?;
    let function_field = curve.function_field()?;
    let basis = numerator_sections
        .iter()
        .map(|numerator| function_field.homogeneous_fraction(numerator, &denominator_section))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RiemannRochSpace {
        divisor: divisor.clone(),
        basis,
        denominator_section,
        numerator_sections,
    })
}

fn try_weierstrass_riemann_roch_space<R: Field>(
    divisor: &WeilDivisor<R>,
) -> Result<Option<RiemannRochSpace<R>>, CurveError> {
    if divisor.curve.presentation_kind() != CurvePresentationKind::SmoothProjectiveWeierstrass {
        return Ok(None);
    }
    let field = divisor.curve.base();
    let zero = field.zero();
    let one = field.one();
    let positive = divisor.positive_part();
    let pole_bound = match positive.terms.as_slice() {
        [] => 0,
        [(point, coefficient)]
            if point.rational_point().is_some_and(|point| {
                point.coordinates() == [zero.clone(), one.clone(), zero.clone()]
            }) =>
        {
            coefficient
                .to_u64_checked()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(CurveError::ResourceLimit {
                    resource: "Weierstrass pole bound",
                    limit: usize::MAX,
                })?
        }
        _ => return Ok(None),
    };
    let negative = divisor.negative_part();
    if negative
        .terms
        .iter()
        .any(|(point, _)| point.rational_point().is_none())
    {
        return Ok(None);
    }

    let functions = divisor.curve.function_field()?;
    let x = functions.x()?;
    let y = functions.y()?;
    let ring = divisor.curve.equation.ring();
    let common_degree = u64::try_from(pole_bound).map_err(|_| CurveError::ResourceLimit {
        resource: "Weierstrass Riemann--Roch section degree",
        limit: usize::MAX,
    })?;
    let denominator_section =
        ring.from_terms([(small_exponents([0, 0, common_degree]), field.one())])?;

    let mut ambient = vec![(0_usize, 0_usize, false, functions.one())];
    for power in 1..=pole_bound / 2 {
        ambient.push((2 * power, power, false, x.pow(power as u64)?));
    }
    if pole_bound >= 3 {
        for power in 0..=(pole_bound - 3) / 2 {
            let function = functions.checked_mul(&x.pow(power as u64)?, &y)?;
            ambient.push((2 * power + 3, power, true, function));
        }
    }
    ambient.sort_by_key(|(pole_order, x_power, has_y, _)| (*pole_order, *has_y, *x_power));

    let ambient_numerators = ambient
        .iter()
        .map(|(_, x_power, has_y, _)| {
            let x_power = u64::try_from(*x_power).map_err(|_| CurveError::ResourceLimit {
                resource: "Weierstrass monomial exponent",
                limit: usize::MAX,
            })?;
            let y_power = u64::from(*has_y);
            let used_degree = x_power + y_power;
            let z_power =
                common_degree
                    .checked_sub(used_degree)
                    .ok_or(CurveError::InvalidWitness(
                        "a bounded-pole Weierstrass monomial homogenizes in the common degree",
                    ))?;
            ring.from_terms([(small_exponents([x_power, y_power, z_power]), field.one())])
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>, CurveError>>()?;

    let condition_count = negative.terms.iter().try_fold(0_usize, |sum, (_, order)| {
        order
            .to_u64_checked()
            .and_then(|order| usize::try_from(order).ok())
            .and_then(|order| sum.checked_add(order))
            .ok_or(CurveError::ResourceLimit {
                resource: "Weierstrass vanishing-condition count",
                limit: usize::MAX,
            })
    })?;
    let mut rows = Vec::with_capacity(condition_count);
    for (point, order) in &negative.terms {
        let point = point.rational_point().ok_or(CurveError::InvalidWitness(
            "the Weierstrass fast path uses rational support",
        ))?;
        let order = order
            .to_u64_checked()
            .and_then(|order| usize::try_from(order).ok())
            .ok_or(CurveError::ResourceLimit {
                resource: "Weierstrass vanishing order",
                limit: usize::MAX,
            })?;
        // This ambient basis contains only `x^i` and `x^i y`. A cubic chart
        // consumes at most six absolute terms when its patch coordinate is
        // inverted, so sixteen guard terms suffice without invoking the much
        // larger generic rational-function allowance.
        let precision = order
            .checked_add(16)
            .filter(|precision| *precision <= 1_000_000)
            .ok_or(CurveError::ResourceLimit {
                resource: "Weierstrass local expansion precision",
                limit: 1_000_000,
            })?;
        let coordinate_germs = projective_coordinate_germs(point, precision)?;
        let germs = ambient
            .iter()
            .map(|(_, _, _, function)| {
                curve_function_germ_from_coordinates(function, &coordinate_germs)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for local_order in 0..order {
            let exponent = i64::try_from(local_order).map_err(|_| CurveError::ResourceLimit {
                resource: "local vanishing exponent",
                limit: usize::MAX,
            })?;
            rows.push(
                germs
                    .iter()
                    .map(|germ| germ.coefficient(exponent))
                    .collect::<Vec<_>>(),
            );
        }
    }
    let operator = if rows.is_empty() {
        Matrix::zeros(field.clone(), 0, ambient.len())
    } else {
        Matrix::from_rows(field.clone(), rows)?
    };
    let kernel = operator.kernel()?;
    let numerator_sections = kernel
        .basis()
        .iter()
        .map(|vector| {
            let mut numerator = ring.zero();
            for (coefficient, monomial) in vector.values().zip(&ambient_numerators) {
                numerator = (&numerator + &monomial.scalar_mul(&coefficient)?)?;
            }
            Ok(numerator)
        })
        .collect::<Result<Vec<_>, CurveError>>()?;
    let basis = numerator_sections
        .iter()
        .map(|numerator| functions.homogeneous_fraction(numerator, &denominator_section))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(RiemannRochSpace {
        divisor: divisor.clone(),
        basis,
        denominator_section,
        numerator_sections,
    }))
}

fn effective_divisor_ideal<R: Field>(
    divisor: &WeilDivisor<R>,
    irrelevant: &super::Ideal<PolynomialRing<R>>,
) -> Result<super::Ideal<PolynomialRing<R>>, CurveError> {
    let curve_ideal = divisor.curve.scheme.defining_ideal();
    let ring = curve_ideal.ring();
    let mut product = ring.ideal([ring.one()])?;
    for (point, coefficient) in &divisor.terms {
        if coefficient.is_negative() {
            return Err(CurveError::InvalidWitness(
                "an effective divisor has nonnegative coefficients",
            ));
        }
        let exponent = coefficient
            .to_u64_checked()
            .ok_or(CurveError::ResourceLimit {
                resource: "divisor multiplicity exponent",
                limit: usize::MAX,
            })?;
        product = product.product(&point.prime_ideal.ideal().pow(exponent)?)?;
    }
    product
        .sum(curve_ideal)?
        .saturate(irrelevant)
        .map_err(Into::into)
}

fn homogeneous_ideal_component<R: Field>(
    ideal: &super::Ideal<PolynomialRing<R>>,
    degree: u64,
) -> Result<Vec<Polynomial<R>>, CurveError> {
    const MAX_SECTION_MONOMIALS: usize = 100_000;
    let ring = ideal.ring();
    let monomials = ring.homogeneous_monomials(degree, MAX_SECTION_MONOMIALS)?;
    let exponents = monomials
        .iter()
        .map(|monomial| {
            monomial
                .terms()
                .next()
                .map(|(exponent, _)| exponent.to_vec())
                .ok_or(CurveError::InvalidWitness(
                    "a homogeneous monomial has one nonzero term",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let normal_forms = monomials
        .iter()
        .map(|monomial| ideal.normal_form(monomial).map_err(Into::into))
        .collect::<Result<Vec<_>, CurveError>>()?;
    let rows = exponents
        .iter()
        .map(|exponent| {
            normal_forms
                .iter()
                .map(|normal_form| normal_form.coefficient(exponent).map_err(Into::into))
                .collect::<Result<Vec<_>, CurveError>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operator = Matrix::from_rows(ring.coefficient_ring().clone(), rows)?;
    let kernel = operator.kernel()?;
    kernel
        .basis()
        .iter()
        .map(|vector| {
            ring.from_terms(exponents.iter().cloned().zip(vector.values()))
                .map_err(Into::into)
        })
        .collect()
}

fn quotient_section_basis<R: Field>(
    curve_ideal: &super::Ideal<PolynomialRing<R>>,
    sections: Vec<Polynomial<R>>,
    degree: u64,
) -> Result<Vec<Polynomial<R>>, CurveError> {
    const MAX_SECTION_MONOMIALS: usize = 100_000;
    let ring = curve_ideal.ring();
    let monomials = ring.homogeneous_monomials(degree, MAX_SECTION_MONOMIALS)?;
    let exponents = monomials
        .iter()
        .map(|monomial| {
            monomial
                .terms()
                .next()
                .map(|(exponent, _)| exponent.to_vec())
                .ok_or(CurveError::InvalidWitness(
                    "a homogeneous monomial has one nonzero term",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut basis = Vec::new();
    let mut rows: Vec<Vec<R::Element>> = Vec::new();
    let mut rank = 0;
    for section in sections {
        let reduced = curve_ideal.normal_form(&section)?;
        if reduced.is_zero() {
            continue;
        }
        let row = exponents
            .iter()
            .map(|exponent| reduced.coefficient(exponent).map_err(Into::into))
            .collect::<Result<Vec<_>, CurveError>>()?;
        let mut candidate_rows = rows.clone();
        candidate_rows.push(row.clone());
        let candidate_rank =
            Matrix::from_rows(ring.coefficient_ring().clone(), candidate_rows)?.rank()?;
        if candidate_rank > rank {
            rows.push(row);
            basis.push(section);
            rank = candidate_rank;
        }
    }
    Ok(basis)
}

/// Exact order of a function in a discrete valuation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurveFunctionOrder {
    /// Finite signed order: positive for a zero and negative for a pole.
    Finite(Integer),
    /// The identically zero function has infinite order at every point.
    Infinite,
}

/// Typed evaluation of a curve function at a rational point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurveFunctionEvaluation<R: Field> {
    /// A nonzero regular value in the constant field.
    Regular(R::Element),
    /// A zero, with `None` only for the identically zero function.
    Zero {
        /// Positive finite vanishing order, or `None` for the zero function.
        order: Option<Integer>,
    },
    /// A pole of the stated positive order.
    Pole {
        /// Positive pole order.
        order: Integer,
    },
}

/// The canonical function field of one checked projective plane curve.
///
/// A deterministic affine chart chooses coordinates `x` and `y`, views the
/// equation as a monic polynomial in `y` over `K(x)`, and stores every element
/// as its unique representative of `y`-degree below the extension degree.
#[derive(Clone, PartialEq, Eq)]
pub struct CurveFunctionField<R: Field> {
    curve: Curve<R>,
    transcendental_coordinate: usize,
    algebraic_coordinate: usize,
    patch_coordinate: usize,
    rational_field: RationalFunctionField<R>,
    extension_ring: PolynomialRing<RationalFunctionField<R>>,
    modulus: Polynomial<RationalFunctionField<R>>,
    extension_degree: usize,
}

impl<R: Field> CurveFunctionField<R> {
    fn new(curve: Curve<R>) -> Result<Self, CurveError> {
        let preferred = match curve.presentation {
            CurvePresentation::Weierstrass { .. } => vec![(0, 1, 2)],
            CurvePresentation::Plane => coordinate_presentations(),
        };
        for (x_index, y_index, patch_index) in preferred {
            if let Some(field) =
                Self::with_coordinates(curve.clone(), x_index, y_index, patch_index)?
            {
                return Ok(field);
            }
        }
        Err(CurveError::NoFunctionFieldPresentation)
    }

    fn with_coordinates(
        curve: Curve<R>,
        x_index: usize,
        y_index: usize,
        patch_index: usize,
    ) -> Result<Option<Self>, CurveError> {
        let names = curve.equation.ring().variables();
        let rational_field =
            RationalFunctionField::new(curve.base().clone(), names[x_index].clone())?;
        let extension_ring = PolynomialRing::new(
            rational_field.clone(),
            [names[y_index].clone()],
            MonomialOrder::Lexicographic,
        )?;
        let coefficient_ring = rational_field.polynomial_ring();
        let mut by_y = std::collections::BTreeMap::new();
        for (exponent, coefficient) in curve.equation.terms() {
            let y_power = exponent[y_index].clone();
            by_y.entry(y_power)
                .or_insert_with(Vec::new)
                .push((vec![exponent[x_index].clone()], coefficient.clone()));
            let _ = patch_index;
        }
        let Some(y_degree) = by_y.keys().next_back().cloned() else {
            return Ok(None);
        };
        if y_degree.is_zero() {
            return Ok(None);
        }
        let extension_degree = integer_to_usize(&y_degree)?;
        let mut terms = Vec::with_capacity(by_y.len());
        for (y_power, coefficient_terms) in by_y {
            let coefficient = coefficient_ring.from_terms(coefficient_terms)?;
            let coefficient = rational_field.from_polynomial(coefficient)?;
            terms.push((vec![y_power], coefficient));
        }
        let modulus = extension_ring.from_terms(terms)?.monic()?;
        Ok(Some(Self {
            curve,
            transcendental_coordinate: x_index,
            algebraic_coordinate: y_index,
            patch_coordinate: patch_index,
            rational_field,
            extension_ring,
            modulus,
            extension_degree,
        }))
    }

    /// Source curve.
    pub fn curve(&self) -> &Curve<R> {
        &self.curve
    }

    /// Coordinate used as the transcendental generator.
    pub fn transcendental_coordinate(&self) -> usize {
        self.transcendental_coordinate
    }

    /// Coordinate algebraic over the rational subfield.
    pub fn algebraic_coordinate(&self) -> usize {
        self.algebraic_coordinate
    }

    /// Homogeneous coordinate set to one in the selected affine chart.
    pub fn patch_coordinate(&self) -> usize {
        self.patch_coordinate
    }

    /// Rational subfield `K(x)` retained by this presentation.
    pub fn rational_subfield(&self) -> &RationalFunctionField<R> {
        &self.rational_field
    }

    /// Monic defining polynomial in `K(x)[y]`.
    pub fn modulus(&self) -> &Polynomial<RationalFunctionField<R>> {
        &self.modulus
    }

    /// Degree over the retained rational subfield.
    pub fn extension_degree(&self) -> usize {
        self.extension_degree
    }

    /// The affine coordinate `x` in the chosen chart.
    pub fn x(&self) -> Result<CurveFunction<R>, CurveError> {
        self.from_representative(
            self.extension_ring
                .constant(self.rational_field.generator())?,
        )
    }

    /// The affine coordinate `y` in the chosen chart.
    pub fn y(&self) -> Result<CurveFunction<R>, CurveError> {
        self.from_representative(self.extension_ring.generator(0)?)
    }

    /// The ratio of a homogeneous coordinate to the selected patch coordinate.
    pub fn coordinate(&self, coordinate: usize) -> Result<CurveFunction<R>, CurveError> {
        if coordinate == self.transcendental_coordinate {
            self.x()
        } else if coordinate == self.algebraic_coordinate {
            self.y()
        } else if coordinate == self.patch_coordinate {
            Ok(self.one())
        } else {
            Err(CurveError::InvalidWitness(
                "homogeneous coordinate index is outside the plane presentation",
            ))
        }
    }

    /// Embed one constant-field element.
    pub fn constant(&self, value: R::Element) -> Result<CurveFunction<R>, CurveError> {
        let coefficient = self.rational_field.constant(value)?;
        self.from_representative(self.extension_ring.constant(coefficient)?)
    }

    /// Restrict one homogeneous-coordinate polynomial to the curve's function field.
    ///
    /// Homogeneous coordinates are interpreted as ratios to the selected patch
    /// coordinate. For a homogeneous polynomial this differs from the original
    /// section by the corresponding patch-coordinate power, exactly as needed
    /// when forming a ratio of equal-degree sections.
    pub fn from_homogeneous_polynomial(
        &self,
        polynomial: &Polynomial<R>,
    ) -> Result<CurveFunction<R>, CurveError> {
        if polynomial.ring() != self.curve.equation.ring() {
            return Err(ContextMismatchError {
                expected: self.curve.equation.ring().to_string(),
                actual: polynomial.ring().to_string(),
            }
            .into());
        }
        let x = self.x()?;
        let y = self.y()?;
        let mut result = self.zero();
        for (exponent, coefficient) in polynomial.terms() {
            let mut term = self.constant(coefficient.clone())?;
            let x_power = integer_to_u64(&exponent[self.transcendental_coordinate])?;
            let y_power = integer_to_u64(&exponent[self.algebraic_coordinate])?;
            term = self.checked_mul(&term, &x.pow(x_power)?)?;
            term = self.checked_mul(&term, &y.pow(y_power)?)?;
            result = self.checked_add(&result, &term)?;
        }
        Ok(result)
    }

    /// Ratio of two equal-degree homogeneous sections restricted to the curve.
    pub fn homogeneous_fraction(
        &self,
        numerator: &Polynomial<R>,
        denominator: &Polynomial<R>,
    ) -> Result<CurveFunction<R>, CurveError> {
        let numerator_degree = numerator.homogeneous_degree()?;
        let denominator_degree = denominator.homogeneous_degree()?;
        if numerator_degree.is_none() || numerator_degree != denominator_degree {
            return Err(CurveError::InvalidWitness(
                "a projective function ratio uses homogeneous sections of equal degree",
            ));
        }
        let numerator = self.from_homogeneous_polynomial(numerator)?;
        let denominator = self.from_homogeneous_polynomial(denominator)?;
        numerator.checked_div(&denominator)
    }

    /// Reduce one polynomial in the retained algebraic coordinate modulo the curve equation.
    pub fn from_representative(
        &self,
        value: Polynomial<RationalFunctionField<R>>,
    ) -> Result<CurveFunction<R>, CurveError> {
        if !self.extension_ring.contains(&value) {
            return Err(ContextMismatchError {
                expected: self.extension_ring.to_string(),
                actual: value.ring().to_string(),
            }
            .into());
        }
        let (_, representative) = value.div_rem(&self.modulus)?;
        Ok(CurveFunction {
            field: self.clone(),
            representative,
        })
    }

    fn require(&self, value: &CurveFunction<R>) -> Result<(), CurveError> {
        if self == &value.field {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: Self::TYPE_ID.to_owned(),
                actual: "curve function belongs to another function field".to_owned(),
            }
            .into())
        }
    }

    fn checked_add(
        &self,
        left: &CurveFunction<R>,
        right: &CurveFunction<R>,
    ) -> Result<CurveFunction<R>, CurveError> {
        self.require(left)?;
        self.require(right)?;
        self.from_representative((&left.representative + &right.representative)?)
    }

    fn checked_sub(
        &self,
        left: &CurveFunction<R>,
        right: &CurveFunction<R>,
    ) -> Result<CurveFunction<R>, CurveError> {
        self.require(left)?;
        self.require(right)?;
        self.from_representative((&left.representative - &right.representative)?)
    }

    fn checked_mul(
        &self,
        left: &CurveFunction<R>,
        right: &CurveFunction<R>,
    ) -> Result<CurveFunction<R>, CurveError> {
        self.require(left)?;
        self.require(right)?;
        self.from_representative((&left.representative * &right.representative)?)
    }

    fn checked_neg(&self, value: &CurveFunction<R>) -> Result<CurveFunction<R>, CurveError> {
        self.require(value)?;
        self.from_representative((-&value.representative)?)
    }

    fn checked_inverse(&self, value: &CurveFunction<R>) -> Result<CurveFunction<R>, CurveError> {
        self.require(value)?;
        if value.is_zero() {
            return Err(CurveError::DivisionByZero);
        }
        let (gcd, coefficient) = extended_gcd_coefficient(&self.modulus, &value.representative)?;
        let degree = gcd.degree(Some(0))?;
        if degree.as_ref().is_some_and(|degree| !degree.is_zero()) || gcd.is_zero() {
            return Err(CurveError::InvalidWitness(
                "the checked integral curve equation did not yield a field quotient",
            ));
        }
        let scalar = gcd.constant_coefficient();
        let scalar_inverse = self
            .rational_field
            .inverse(&scalar)
            .map_err(|_| CurveError::InvalidWitness("extended gcd returned zero"))?;
        self.from_representative(coefficient.scalar_mul(&scalar_inverse)?)
    }

    /// Recheck the source curve, monic modulus, degree, and coordinate conversion.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.curve.verify()?;
        let rebuilt = Self::with_coordinates(
            self.curve.clone(),
            self.transcendental_coordinate,
            self.algebraic_coordinate,
            self.patch_coordinate,
        )
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "curve function-field presentation reconstructs",
        })?
        .ok_or(VerificationError::InvalidWitness {
            claim: "curve function-field equation has positive algebraic degree",
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "curve function-field presentation is canonical",
            })
        }
    }

    /// Serialize the owning curve; the canonical field presentation is derived.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        encode_typed_json(
            Self::TYPE_ID,
            [("curve", nested_json(self.curve.to_json()?)?)],
        )
    }

    /// Restore the curve and deterministically rebuild its function field.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: DeserializeOwned,
        R::Element: DeserializeOwned,
    {
        let mut object = decode_typed_json(value, Self::TYPE_ID, &["curve"])?;
        let curve = Curve::<R>::from_json(&encode_nested_field(&mut object, "curve")?)?;
        curve.function_field().map_err(curve_serialization_error)
    }
}

impl<R: Field> fmt::Debug for CurveFunctionField<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CurveFunctionField")
            .field("base", &R::TYPE_ID)
            .field("x", &self.transcendental_coordinate)
            .field("y", &self.algebraic_coordinate)
            .field("patch", &self.patch_coordinate)
            .field("degree", &self.extension_degree)
            .finish()
    }
}

impl<R: Field> fmt::Display for CurveFunctionField<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CurveFunctionField<{}>(degree={})",
            R::TYPE_ID,
            self.extension_degree
        )
    }
}

impl<R: Field> SemanticType for CurveFunctionField<R> {
    const TYPE_ID: &'static str = "noether/commalg/CurveFunctionField@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.curve.canonical_bytes()?;
        for coordinate in [
            self.transcendental_coordinate,
            self.algebraic_coordinate,
            self.patch_coordinate,
        ] {
            output.extend_from_slice(
                &u64::try_from(coordinate)
                    .map_err(|_| NoetherError::Serialization("coordinate index overflow".into()))?
                    .to_be_bytes(),
            );
        }
        append_component(&mut output, &self.modulus.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CURVE_FUNCTION_FIELD_DESCRIPTION
    }
}

impl<R: Field> Ring for CurveFunctionField<R> {
    type Element = CurveFunction<R>;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.field
    }

    fn zero(&self) -> Self::Element {
        CurveFunction {
            field: self.clone(),
            representative: self.extension_ring.zero(),
        }
    }

    fn one(&self) -> Self::Element {
        CurveFunction {
            field: self.clone(),
            representative: self.extension_ring.one(),
        }
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.checked_add(a, b).map_err(curve_context_error::<R>)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.checked_sub(a, b).map_err(curve_context_error::<R>)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.checked_mul(a, b).map_err(curve_context_error::<R>)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.checked_neg(value).map_err(curve_context_error::<R>)
    }
}

impl<R: Field> CommutativeRing for CurveFunctionField<R> {}
impl<R: Field> IntegralDomain for CurveFunctionField<R> {}

impl<R: Field> UnitInversion for CurveFunctionField<R> {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.inverse(value).map_err(UnitError::from)
    }
}

impl<R: Field> Field for CurveFunctionField<R> {
    fn inverse(&self, value: &Self::Element) -> Result<Self::Element, InversionError> {
        self.checked_inverse(value).map_err(|error| match error {
            CurveError::DivisionByZero => DivisionByZeroError.into(),
            CurveError::ContextMismatch(error) => error.into(),
            error => InversionError::ContextMismatch(curve_context_error::<R>(error)),
        })
    }
}

/// One canonical element of a checked curve function field.
#[derive(Clone, PartialEq, Eq)]
pub struct CurveFunction<R: Field> {
    field: CurveFunctionField<R>,
    representative: Polynomial<RationalFunctionField<R>>,
}

impl<R: Field> CurveFunction<R> {
    /// Owning curve function field.
    pub fn field(&self) -> &CurveFunctionField<R> {
        &self.field
    }

    /// Unique representative of algebraic-coordinate degree below the field degree.
    pub fn representative(&self) -> &Polynomial<RationalFunctionField<R>> {
        &self.representative
    }

    /// Canonical equal-degree homogeneous numerator and denominator sections.
    ///
    /// The zero function is represented by `(0,1)`. Every nonzero function is
    /// first reduced in the admitted `K(x)[y]` presentation and then cleared
    /// of coefficient denominators deterministically.
    pub fn homogeneous_representation(&self) -> Result<(Polynomial<R>, Polynomial<R>), CurveError> {
        if self.is_zero() {
            let ring = self.field.curve.equation.ring();
            return Ok((ring.zero(), ring.one()));
        }
        canonical_projective_fraction(self)
    }

    /// Whether this function is zero.
    pub fn is_zero(&self) -> bool {
        self.representative.is_zero()
    }

    /// Whether this function is one.
    pub fn is_one(&self) -> bool {
        self.representative == self.field.extension_ring.one()
    }

    /// Recover the constant-field value when this function is constant.
    pub fn as_constant(&self) -> Result<Option<R::Element>, CurveError> {
        if self.is_zero() {
            return Ok(Some(self.field.curve.base().zero()));
        }
        if self
            .representative
            .terms()
            .any(|(exponent, _)| !exponent[0].is_zero())
        {
            return Ok(None);
        }
        let coefficient = self.representative.coefficient(&[Integer::zero()])?;
        let numerator_degree = coefficient.numerator().degree(Some(0))?;
        let denominator_degree = coefficient.denominator().degree(Some(0))?;
        if numerator_degree
            .as_ref()
            .is_some_and(|degree| !degree.is_zero())
            || denominator_degree
                .as_ref()
                .is_some_and(|degree| !degree.is_zero())
        {
            return Ok(None);
        }
        let numerator = coefficient.numerator().constant_coefficient();
        let denominator = coefficient.denominator().constant_coefficient();
        let inverse = self
            .field
            .curve
            .base()
            .inverse(&denominator)
            .map_err(|_| CurveError::InvalidWitness("a reduced denominator is nonzero"))?;
        self.field
            .curve
            .base()
            .mul(&numerator, &inverse)
            .map(Some)
            .map_err(Into::into)
    }

    /// Raise to a nonnegative machine-word power.
    pub fn pow(&self, mut exponent: u64) -> Result<Self, CurveError> {
        let mut result = self.field.one();
        let mut base = self.clone();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = self.field.checked_mul(&result, &base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = self.field.checked_mul(&base, &base)?;
            }
        }
        Ok(result)
    }

    /// Checked multiplicative inverse.
    pub fn checked_inverse(&self) -> Result<Self, CurveError> {
        self.field.checked_inverse(self)
    }

    /// Divide by another function from this exact field.
    pub fn checked_div(&self, divisor: &Self) -> Result<Self, CurveError> {
        let inverse = self.field.checked_inverse(divisor)?;
        self.field.checked_mul(self, &inverse)
    }

    /// Exact discrete-valuation order at a checked rational point.
    ///
    /// The local implicit-function expansion is deliberately bounded. Use
    /// [`order_at_with_limit`](Self::order_at_with_limit) when a larger order
    /// is expected.
    pub fn order_at(&self, point: &CurvePoint<R>) -> Result<CurveFunctionOrder, CurveError> {
        self.order_at_with_limit(point, 256)
    }

    /// Exact order at a rational point, searching through `limit` local terms.
    pub fn order_at_with_limit(
        &self,
        point: &CurvePoint<R>,
        limit: usize,
    ) -> Result<CurveFunctionOrder, CurveError> {
        if point.curve() != self.field.curve() {
            return Err(ContextMismatchError {
                expected: Curve::<R>::TYPE_ID.to_owned(),
                actual: "evaluation point lies on another curve".to_owned(),
            }
            .into());
        }
        if self.is_zero() {
            return Ok(CurveFunctionOrder::Infinite);
        }
        let (order, _) = self.finite_order_germ(point, limit)?;
        Ok(CurveFunctionOrder::Finite(Integer::from(order)))
    }

    /// Evaluate at a rational point with distinct regular, zero, and pole results.
    pub fn evaluate_at(
        &self,
        point: &CurvePoint<R>,
    ) -> Result<CurveFunctionEvaluation<R>, CurveError> {
        self.evaluate_at_with_limit(point, 256)
    }

    /// Typed evaluation, searching through the supplied local-order bound.
    pub fn evaluate_at_with_limit(
        &self,
        point: &CurvePoint<R>,
        limit: usize,
    ) -> Result<CurveFunctionEvaluation<R>, CurveError> {
        if point.curve() != self.field.curve() {
            return Err(ContextMismatchError {
                expected: Curve::<R>::TYPE_ID.to_owned(),
                actual: "evaluation point lies on another curve".to_owned(),
            }
            .into());
        }
        if self.is_zero() {
            return Ok(CurveFunctionEvaluation::Zero { order: None });
        }
        // A homogeneous equal-degree fraction can be evaluated directly when
        // neither section vanishes. This is both the canonical projective
        // evaluation and an important fast path over symbolic coefficient
        // fields, where constructing hundreds of local series coefficients
        // would be needlessly expensive for a regular nonzero value.
        let (numerator, denominator) = self.homogeneous_representation()?;
        let numerator_value = numerator.evaluate(point.coordinates())?;
        let denominator_value = denominator.evaluate(point.coordinates())?;
        let zero = self.field.curve.base().zero();
        if numerator_value != zero && denominator_value != zero {
            let inverse = self
                .field
                .curve
                .base()
                .inverse(&denominator_value)
                .map_err(|_| {
                    CurveError::InvalidWitness("a nonzero evaluated denominator is invertible")
                })?;
            let value = self.field.curve.base().mul(&numerator_value, &inverse)?;
            return Ok(CurveFunctionEvaluation::Regular(value));
        }
        let (order, germ) = self.finite_order_germ(point, limit)?;
        if order > 0 {
            Ok(CurveFunctionEvaluation::Zero {
                order: Some(Integer::from(order)),
            })
        } else if order < 0 {
            Ok(CurveFunctionEvaluation::Pole {
                order: Integer::from(-order),
            })
        } else {
            let value = germ.coefficient(0);
            if value == self.field.curve.base().zero() {
                return Err(CurveError::InvalidWitness(
                    "a valuation-zero germ has nonzero constant coefficient",
                ));
            }
            Ok(CurveFunctionEvaluation::Regular(value))
        }
    }

    fn finite_order_germ(
        &self,
        point: &CurvePoint<R>,
        limit: usize,
    ) -> Result<(i64, LaurentGerm<R>), CurveError> {
        let mut searched = limit.min(8);
        loop {
            let germ = curve_function_germ(self, point, local_work_precision(self, searched)?)?;
            if let Some(order) = germ.valuation() {
                if order.unsigned_abs() > limit as u64 {
                    return Err(CurveError::LocalExpansionLimit { limit });
                }
                return Ok((order, germ));
            }
            if searched == limit {
                return Err(CurveError::LocalExpansionLimit { limit });
            }
            searched = searched.saturating_mul(2).max(1).min(limit);
        }
    }

    /// Recompute reduction modulo the defining equation.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.field.verify()?;
        let rebuilt = self
            .field
            .from_representative(self.representative.clone())
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "curve function representative reduces in its owning field",
            })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "curve function representative has degree below the defining equation",
            })
        }
    }

    /// Serialize a canonical equal-degree homogeneous fraction on its curve.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        let (numerator, denominator) = self
            .homogeneous_representation()
            .map_err(curve_serialization_error)?;
        encode_typed_json(
            Self::TYPE_ID,
            [
                ("field", nested_json(self.field.to_json()?)?),
                (
                    "numerator",
                    serde_json::to_value(numerator).map_err(serialization_error)?,
                ),
                (
                    "denominator",
                    serde_json::to_value(denominator).map_err(serialization_error)?,
                ),
            ],
        )
    }

    /// Restore, reduce, and reject a noncanonical homogeneous presentation.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: DeserializeOwned,
        R::Element: DeserializeOwned,
    {
        let mut object =
            decode_typed_json(value, Self::TYPE_ID, &["field", "numerator", "denominator"])?;
        let field =
            CurveFunctionField::<R>::from_json(&encode_nested_field(&mut object, "field")?)?;
        let numerator: Polynomial<R> = decode_field(&mut object, "numerator")?;
        let denominator: Polynomial<R> = decode_field(&mut object, "denominator")?;
        let rebuilt = if numerator.is_zero() && denominator.is_one() {
            field.zero()
        } else {
            field
                .homogeneous_fraction(&numerator, &denominator)
                .map_err(curve_serialization_error)?
        };
        let canonical = rebuilt
            .homogeneous_representation()
            .map_err(curve_serialization_error)?;
        if canonical != (numerator, denominator) {
            return Err(NoetherError::Serialization(
                "serialized curve-function fraction is not canonical".to_owned(),
            ));
        }
        Ok(rebuilt)
    }
}

impl CurveFunction<RationalField> {
    /// Complete principal Weil divisor over `Q`, including non-rational closed points.
    pub fn principal_divisor(&self) -> Result<WeilDivisor<RationalField>, CurveError> {
        principal_divisor_with_factorization(self, |polynomial: &Polynomial<RationalField>| {
            polynomial.factor()
        })
    }

    /// Independently recompute and compare a proposed principal divisor.
    pub fn verify_principal_divisor(
        &self,
        divisor: &WeilDivisor<RationalField>,
    ) -> Result<(), VerificationError> {
        if &self
            .principal_divisor()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "principal divisor is computable over Q",
            })?
            == divisor
        {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "proposed divisor equals the function's complete zero-minus-pole divisor",
            })
        }
    }
}

impl CurveFunction<FiniteField> {
    /// Complete principal Weil divisor over a finite field, with residue-field places retained.
    pub fn principal_divisor(&self) -> Result<WeilDivisor<FiniteField>, CurveError> {
        principal_divisor_with_factorization(self, |polynomial: &Polynomial<FiniteField>| {
            polynomial.factor()
        })
    }

    /// Independently recompute and compare a proposed principal divisor.
    pub fn verify_principal_divisor(
        &self,
        divisor: &WeilDivisor<FiniteField>,
    ) -> Result<(), VerificationError> {
        if &self
            .principal_divisor()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "principal divisor is computable over a finite field",
            })?
            == divisor
        {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "proposed divisor equals the function's complete zero-minus-pole divisor",
            })
        }
    }
}

impl WeilDivisor<RationalField> {
    /// Scheme-theoretic pullback along a finite checked curve morphism over `Q`.
    pub fn pullback(&self, morphism: &CurveMorphism<RationalField>) -> Result<Self, CurveError> {
        pullback_divisor_with_factorization(self, morphism, |polynomial| polynomial.factor())
    }
}

impl WeilDivisor<FiniteField> {
    /// Scheme-theoretic pullback along a finite checked curve morphism over a finite field.
    pub fn pullback(&self, morphism: &CurveMorphism<FiniteField>) -> Result<Self, CurveError> {
        pullback_divisor_with_factorization(self, morphism, |polynomial| polynomial.factor())
    }
}

fn pullback_divisor_with_factorization<R, F>(
    divisor: &WeilDivisor<R>,
    morphism: &CurveMorphism<R>,
    factor: F,
) -> Result<WeilDivisor<R>, CurveError>
where
    R: Field,
    F: Fn(&Polynomial<R>) -> Result<PolynomialFactorization<R>, PolynomialFactorizationError>
        + Copy,
{
    if divisor.curve() != morphism.target() {
        return Err(ContextMismatchError {
            expected: Curve::<R>::TYPE_ID.to_owned(),
            actual: "divisor is not on the morphism target".to_owned(),
        }
        .into());
    }
    let source_ring = morphism.source.equation.ring();
    let mut result = WeilDivisor::zero(morphism.source.clone());
    for (point, coefficient) in &divisor.terms {
        let mut generators = morphism
            .source
            .scheme
            .defining_ideal()
            .generators()
            .to_vec();
        for generator in point.prime_ideal.ideal().generators() {
            generators.push(evaluate_polynomial_at_map(
                generator,
                source_ring,
                &morphism.coordinate_images,
            )?);
        }
        let inverse_image = source_ring.ideal(generators)?;
        let effective = match inverse_image.dimension()? {
            -1 => WeilDivisor::zero(morphism.source.clone()),
            1 => {
                let expected_degree = inverse_image.degree()?;
                projective_ideal_divisor(
                    morphism.source(),
                    &inverse_image,
                    &expected_degree,
                    factor,
                )?
            }
            _ => {
                return Err(CurveError::InvalidWitness(
                    "divisor pullback is defined only for zero-dimensional point inverse images",
                ));
            }
        };
        result = result.checked_add(&effective.scale(coefficient))?;
    }
    Ok(result)
}

fn principal_divisor_with_factorization<R, F>(
    function: &CurveFunction<R>,
    factor: F,
) -> Result<WeilDivisor<R>, CurveError>
where
    R: Field,
    F: Fn(&Polynomial<R>) -> Result<PolynomialFactorization<R>, PolynomialFactorizationError>
        + Copy,
{
    if function.is_zero() {
        return Err(CurveError::InvalidWitness(
            "the zero function has no principal divisor",
        ));
    }
    let (numerator, denominator) = canonical_projective_fraction(function)?;
    let zeros = projective_section_divisor(function.field.curve(), &numerator, factor)?;
    let poles = projective_section_divisor(function.field.curve(), &denominator, factor)?;
    let divisor = zeros.checked_sub(&poles)?;
    if !divisor.degree().is_zero() {
        return Err(CurveError::InvalidWitness(
            "a principal divisor has degree zero",
        ));
    }
    Ok(divisor)
}

fn canonical_projective_fraction<R: Field>(
    function: &CurveFunction<R>,
) -> Result<(Polynomial<R>, Polynomial<R>), CurveError> {
    let field = function.field();
    let univariate = field.rational_field.polynomial_ring();
    let mut common_denominator = univariate.one();
    for (_, coefficient) in function.representative.terms() {
        let gcd = common_denominator.gcd(coefficient.denominator())?;
        let new_factor = coefficient.denominator().exact_div(&gcd)?;
        common_denominator = (&common_denominator * &new_factor)?;
    }
    let mut numerator_terms = Vec::new();
    for (y_exponent, coefficient) in function.representative.terms() {
        let scale = common_denominator.exact_div(coefficient.denominator())?;
        let scaled_numerator = (coefficient.numerator() * &scale)?;
        for (x_exponent, scalar) in scaled_numerator.terms() {
            numerator_terms.push((x_exponent[0].clone(), y_exponent[0].clone(), scalar.clone()));
        }
    }
    let denominator_terms = common_denominator
        .terms()
        .map(|(x_exponent, scalar)| (x_exponent[0].clone(), Integer::zero(), scalar.clone()))
        .collect::<Vec<_>>();
    let degree = numerator_terms
        .iter()
        .chain(&denominator_terms)
        .map(|(x, y, _)| x + y)
        .max()
        .ok_or(CurveError::InvalidWitness(
            "a nonzero function has a nonzero projective fraction",
        ))?;
    let numerator = homogenize_xy_terms(field, numerator_terms, &degree)?;
    let denominator = homogenize_xy_terms(field, denominator_terms, &degree)?;
    if numerator.is_zero() || denominator.is_zero() {
        return Err(CurveError::InvalidWitness(
            "a nonzero curve function has nonzero numerator and denominator sections",
        ));
    }
    Ok((numerator, denominator))
}

fn homogenize_xy_terms<R: Field>(
    field: &CurveFunctionField<R>,
    terms: Vec<(Integer, Integer, R::Element)>,
    degree: &Integer,
) -> Result<Polynomial<R>, CurveError> {
    let ring = field.curve.equation.ring();
    ring.from_terms(terms.into_iter().map(|(x, y, coefficient)| {
        let used = &x + &y;
        let patch = degree - &used;
        let mut exponent = vec![Integer::zero(); 3];
        exponent[field.transcendental_coordinate] = x;
        exponent[field.algebraic_coordinate] = y;
        exponent[field.patch_coordinate] = patch;
        (exponent, coefficient)
    }))
    .map_err(Into::into)
}

fn projective_section_divisor<R, F>(
    curve: &Curve<R>,
    section: &Polynomial<R>,
    factor: F,
) -> Result<WeilDivisor<R>, CurveError>
where
    R: Field,
    F: Fn(&Polynomial<R>) -> Result<PolynomialFactorization<R>, PolynomialFactorizationError>
        + Copy,
{
    let degree = section
        .homogeneous_degree()?
        .ok_or(CurveError::InvalidWitness(
            "a projective section is homogeneous",
        ))?;
    if degree.is_zero() {
        return Ok(WeilDivisor::zero(curve.clone()));
    }
    let curve_degree = curve
        .equation
        .homogeneous_degree()?
        .ok_or(CurveError::InvalidWitness(
            "a plane curve equation is homogeneous",
        ))?;
    let expected_degree = &curve_degree * &degree;
    let projective_ring = curve.equation.ring();
    let intersection = curve
        .scheme
        .defining_ideal()
        .sum(&projective_ring.ideal([section.clone()])?)?;
    projective_ideal_divisor(curve, &intersection, &expected_degree, factor)
}

fn projective_ideal_divisor<R, F>(
    curve: &Curve<R>,
    ideal: &super::Ideal<PolynomialRing<R>>,
    expected_degree: &Integer,
    factor: F,
) -> Result<WeilDivisor<R>, CurveError>
where
    R: Field,
    F: Fn(&Polynomial<R>) -> Result<PolynomialFactorization<R>, PolynomialFactorizationError>
        + Copy,
{
    let projective_ring = curve.equation.ring();
    if ideal.ring() != projective_ring {
        return Err(ContextMismatchError {
            expected: projective_ring.to_string(),
            actual: ideal.ring().to_string(),
        }
        .into());
    }
    let mut terms = Vec::new();
    for chart in 0..3 {
        let affine_names = projective_ring
            .variables()
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != chart)
            .map(|(_, name)| name.clone());
        let affine_ring = PolynomialRing::new(
            curve.base().clone(),
            affine_names,
            MonomialOrder::GradedReverseLexicographic,
        )?;
        let generators = ideal
            .generators()
            .iter()
            .map(|generator| dehomogenize_to_chart(generator, &affine_ring, chart))
            .collect::<Result<Vec<_>, _>>()?;
        let intersection = affine_ring.ideal(generators)?;
        match intersection.dimension()? {
            -1 => continue,
            0 => {}
            _ => {
                return Err(CurveError::InvalidWitness(
                    "a nonzero section cuts a zero-dimensional divisor on an integral curve",
                ));
            }
        }
        let representation = intersection.rational_univariate_representation()?;
        let factorization = factor(representation.equation())?;
        for (parameter_factor, multiplicity) in factorization.factors() {
            let affine_prime = affine_prime_from_rur_factor(&representation, parameter_factor)?;
            let projective_prime =
                projective_closure_of_affine_prime(&affine_prime, projective_ring, chart)?;
            let belongs_to_stratum = (0..chart).try_fold(true, |belongs, earlier| {
                projective_prime
                    .contains(&projective_ring.generator(earlier)?)
                    .map(|contains| belongs && contains)
                    .map_err(CurveError::from)
            })?;
            if !belongs_to_stratum {
                continue;
            }
            let point = CurveClosedPoint::from_prime_ideal(
                curve.clone(),
                PrimeIdeal::from_verified_parts(
                    projective_prime,
                    b"irreducible RUR factor in a projective chart".to_vec(),
                ),
            )?;
            terms.push((point, Integer::from(*multiplicity)));
        }
    }
    let divisor = WeilDivisor::new(curve.clone(), terms)?;
    if &divisor.degree() != expected_degree {
        return Err(CurveError::InvalidWitness(
            "projective chart decomposition has the complete Bezout degree",
        ));
    }
    Ok(divisor)
}

fn affine_prime_from_rur_factor<R: Field>(
    representation: &super::RationalUnivariateRepresentation<R>,
    factor: &Polynomial<R>,
) -> Result<super::Ideal<PolynomialRing<R>>, CurveError> {
    let ring = representation.source().ring();
    let mut linear_form = ring.zero();
    for (variable, coefficient) in representation.linear_form().iter().enumerate() {
        linear_form = (&linear_form + &ring.generator(variable)?.scalar_mul(coefficient)?)?;
    }
    let factor_at_parameter = substitute_univariate(factor, &linear_form)?;
    let denominator = substitute_univariate(representation.denominator(), &linear_form)?;
    let mut generators = vec![factor_at_parameter];
    for (variable, numerator) in representation.coordinate_numerators().iter().enumerate() {
        let numerator = substitute_univariate(numerator, &linear_form)?;
        let left = (&denominator * &ring.generator(variable)?)?;
        generators.push((&left - &numerator)?);
    }
    let prime = ring.ideal(generators)?;
    let factor_degree = factor.degree(Some(0))?.ok_or(CurveError::InvalidWitness(
        "an irreducible RUR factor is nonzero",
    ))?;
    if prime.dimension()? != 0 || prime.degree()? != factor_degree {
        return Err(CurveError::InvalidWitness(
            "an irreducible RUR factor defines a residue field of the claimed degree",
        ));
    }
    for generator in representation.source().generators() {
        if !prime.contains(generator)? {
            return Err(CurveError::InvalidWitness(
                "an RUR-factor prime contains the source intersection ideal",
            ));
        }
    }
    Ok(prime)
}

fn substitute_univariate<R: Field>(
    polynomial: &Polynomial<R>,
    image: &Polynomial<R>,
) -> Result<Polynomial<R>, CurveError> {
    let ring = image.ring();
    let mut result = ring.zero();
    for (exponent, coefficient) in polynomial.terms() {
        let power = image.pow_integer(&exponent[0])?;
        result = (&result + &power.scalar_mul(coefficient)?)?;
    }
    Ok(result)
}

fn dehomogenize_to_chart<R: Field>(
    polynomial: &Polynomial<R>,
    target: &PolynomialRing<R>,
    chart: usize,
) -> Result<Polynomial<R>, CurveError> {
    target
        .from_terms(polynomial.terms().map(|(exponent, coefficient)| {
            let mut exponent = exponent.to_vec();
            exponent.remove(chart);
            (exponent, coefficient.clone())
        }))
        .map_err(Into::into)
}

fn projective_closure_of_affine_prime<R: Field>(
    affine_prime: &super::Ideal<PolynomialRing<R>>,
    projective_ring: &PolynomialRing<R>,
    chart: usize,
) -> Result<super::Ideal<PolynomialRing<R>>, CurveError> {
    let generators = affine_prime
        .generators()
        .iter()
        .map(|generator| homogenize_affine_polynomial(generator, projective_ring, chart))
        .collect::<Result<Vec<_>, _>>()?;
    let closure = projective_ring.ideal(generators)?;
    let patch_ideal = projective_ring.ideal([projective_ring.generator(chart)?])?;
    closure.saturate(&patch_ideal).map_err(Into::into)
}

fn homogenize_affine_polynomial<R: Field>(
    polynomial: &Polynomial<R>,
    target: &PolynomialRing<R>,
    chart: usize,
) -> Result<Polynomial<R>, CurveError> {
    let degree = polynomial.degree(None)?.unwrap_or_else(Integer::zero);
    target
        .from_terms(polynomial.terms().map(|(exponent, coefficient)| {
            let used = exponent
                .iter()
                .fold(Integer::zero(), |sum, value| sum + value);
            let mut homogeneous = exponent.to_vec();
            homogeneous.insert(chart, &degree - &used);
            (homogeneous, coefficient.clone())
        }))
        .map_err(Into::into)
}

#[derive(Clone)]
struct LaurentGerm<R: Field> {
    field: R,
    coefficients: std::collections::BTreeMap<i64, R::Element>,
    known_through: i64,
}

impl<R: Field> LaurentGerm<R> {
    fn zero(field: R, known_through: i64) -> Self {
        Self {
            field,
            coefficients: std::collections::BTreeMap::new(),
            known_through,
        }
    }

    fn constant(field: R, value: R::Element, known_through: i64) -> Self {
        let mut result = Self::zero(field, known_through);
        if value != result.field.zero() {
            result.coefficients.insert(0, value);
        }
        result
    }

    fn parameter(field: R, known_through: i64) -> Self {
        let mut result = Self::zero(field.clone(), known_through);
        if known_through >= 1 {
            result.coefficients.insert(1, field.one());
        }
        result
    }

    fn valuation(&self) -> Option<i64> {
        self.coefficients.keys().next().copied()
    }

    fn coefficient(&self, exponent: i64) -> R::Element {
        self.coefficients
            .get(&exponent)
            .cloned()
            .unwrap_or_else(|| self.field.zero())
    }

    fn set_coefficient(&mut self, exponent: i64, value: R::Element) {
        if exponent > self.known_through {
            return;
        }
        if value == self.field.zero() {
            self.coefficients.remove(&exponent);
        } else {
            self.coefficients.insert(exponent, value);
        }
    }

    fn add(&self, other: &Self) -> Result<Self, CurveError> {
        self.require(other)?;
        let known_through = self.known_through.min(other.known_through);
        let mut result = Self::zero(self.field.clone(), known_through);
        for exponent in self
            .coefficients
            .keys()
            .chain(other.coefficients.keys())
            .copied()
            .filter(|exponent| *exponent <= known_through)
            .collect::<std::collections::BTreeSet<_>>()
        {
            result.set_coefficient(
                exponent,
                self.field
                    .add(&self.coefficient(exponent), &other.coefficient(exponent))?,
            );
        }
        Ok(result)
    }

    fn mul(&self, other: &Self) -> Result<Self, CurveError> {
        self.require(other)?;
        let known_through = match (self.valuation(), other.valuation()) {
            (Some(left), Some(right)) => {
                left + right + (self.known_through - left).min(other.known_through - right)
            }
            (None, Some(right)) => self.known_through + right,
            (Some(left), None) => other.known_through + left,
            (None, None) => self.known_through + other.known_through + 1,
        };
        let mut result = Self::zero(self.field.clone(), known_through);
        for (left_exponent, left_coefficient) in &self.coefficients {
            for (right_exponent, right_coefficient) in &other.coefficients {
                let exponent = left_exponent + right_exponent;
                if exponent > known_through {
                    continue;
                }
                let product = self.field.mul(left_coefficient, right_coefficient)?;
                let coefficient = self.field.add(&result.coefficient(exponent), &product)?;
                result.set_coefficient(exponent, coefficient);
            }
        }
        Ok(result)
    }

    fn pow(&self, mut exponent: usize) -> Result<Self, CurveError> {
        let mut result = Self::constant(self.field.clone(), self.field.one(), self.known_through);
        let mut base = self.clone();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.mul(&base)?;
            }
            exponent >>= 1;
            if exponent > 0 {
                base = base.mul(&base)?;
            }
        }
        Ok(result)
    }

    fn inverse(&self) -> Result<Self, CurveError> {
        let valuation = self.valuation().ok_or(CurveError::LocalExpansionLimit {
            limit: usize::try_from(self.known_through.max(0)).unwrap_or(usize::MAX),
        })?;
        let leading = self.coefficient(valuation);
        let leading_inverse = self
            .field
            .inverse(&leading)
            .map_err(|_| CurveError::InvalidWitness("a germ leading coefficient is nonzero"))?;
        let relative_precision = self.known_through - valuation;
        let known_through = -valuation + relative_precision;
        let mut normalized_inverse = vec![self.field.zero(); (relative_precision + 1) as usize];
        normalized_inverse[0] = leading_inverse.clone();
        for degree in 1..=relative_precision as usize {
            let mut sum = self.field.zero();
            for index in 1..=degree {
                let coefficient = self.coefficient(
                    valuation
                        + i64::try_from(index)
                            .map_err(|_| CurveError::LocalExpansionLimit { limit: degree })?,
                );
                let product = self
                    .field
                    .mul(&coefficient, &normalized_inverse[degree - index])?;
                sum = self.field.add(&sum, &product)?;
            }
            normalized_inverse[degree] =
                self.field.neg(&self.field.mul(&leading_inverse, &sum)?)?;
        }
        let mut result = Self::zero(self.field.clone(), known_through);
        for (degree, coefficient) in normalized_inverse.into_iter().enumerate() {
            let exponent = -valuation
                + i64::try_from(degree)
                    .map_err(|_| CurveError::LocalExpansionLimit { limit: degree })?;
            result.set_coefficient(exponent, coefficient);
        }
        Ok(result)
    }

    fn checked_div(&self, other: &Self) -> Result<Self, CurveError> {
        self.mul(&other.inverse()?)
    }

    fn require(&self, other: &Self) -> Result<(), CurveError> {
        if self.field == other.field {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "local germs use unequal coefficient contexts".to_owned(),
            }
            .into())
        }
    }
}

fn local_work_precision<R: Field>(
    function: &CurveFunction<R>,
    limit: usize,
) -> Result<usize, CurveError> {
    let mut degree_allowance = function.field.extension_degree.saturating_mul(8);
    for (_, coefficient) in function.representative.terms() {
        for polynomial in [coefficient.numerator(), coefficient.denominator()] {
            if let Some(degree) = polynomial.degree(Some(0))? {
                degree_allowance = degree_allowance.saturating_add(integer_to_usize(&degree)?);
            }
        }
    }
    limit
        .checked_mul(4)
        .and_then(|value| value.checked_add(degree_allowance))
        .and_then(|value| value.checked_add(32))
        .filter(|value| *value <= 1_000_000)
        .ok_or(CurveError::LocalExpansionLimit { limit })
}

fn curve_function_germ<R: Field>(
    function: &CurveFunction<R>,
    point: &CurvePoint<R>,
    precision: usize,
) -> Result<LaurentGerm<R>, CurveError> {
    let coordinate_germs = projective_coordinate_germs(point, precision)?;
    curve_function_germ_from_coordinates(function, &coordinate_germs)
}

fn curve_function_germ_from_coordinates<R: Field>(
    function: &CurveFunction<R>,
    coordinate_germs: &[LaurentGerm<R>; 3],
) -> Result<LaurentGerm<R>, CurveError> {
    let field = function.field.curve.base().clone();
    let x = coordinate_germs[function.field.transcendental_coordinate]
        .checked_div(&coordinate_germs[function.field.patch_coordinate])?;
    let y = coordinate_germs[function.field.algebraic_coordinate]
        .checked_div(&coordinate_germs[function.field.patch_coordinate])?;
    let known_through = coordinate_germs
        .iter()
        .map(|coordinate| coordinate.known_through)
        .min()
        .ok_or(CurveError::InvalidWitness(
            "a plane curve has three projective coordinate germs",
        ))?;
    let mut result = LaurentGerm::zero(field.clone(), known_through);
    for (exponent, coefficient) in function.representative.terms() {
        let y_power = integer_to_usize(&exponent[0])?;
        let numerator = evaluate_univariate_germ(coefficient.numerator(), &x)?;
        let denominator = evaluate_univariate_germ(coefficient.denominator(), &x)?;
        let coefficient = numerator.checked_div(&denominator)?;
        let term = coefficient.mul(&y.pow(y_power)?)?;
        result = result.add(&term)?;
    }
    Ok(result)
}

fn projective_coordinate_germs<R: Field>(
    point: &CurvePoint<R>,
    precision: usize,
) -> Result<[LaurentGerm<R>; 3], CurveError> {
    let field = point.curve.base().clone();
    let high = i64::try_from(precision)
        .map_err(|_| CurveError::LocalExpansionLimit { limit: precision })?;
    let zero = field.zero();
    let chart = point
        .coordinates()
        .iter()
        .position(|coordinate| coordinate != &zero)
        .ok_or(CurveError::InvalidWitness(
            "a checked projective point has a nonzero chart coordinate",
        ))?;
    let chart_inverse = field
        .inverse(&point.coordinates()[chart])
        .map_err(|_| CurveError::InvalidWitness("the chart coordinate is nonzero"))?;
    let normalized = point
        .coordinates()
        .iter()
        .map(|coordinate| field.mul(coordinate, &chart_inverse).map_err(Into::into))
        .collect::<Result<Vec<_>, CurveError>>()?;
    let local = (0..3).filter(|index| *index != chart).collect::<Vec<_>>();
    let derivatives = (0..3)
        .map(|variable| {
            point
                .curve
                .equation
                .derivative(variable)?
                .evaluate(&normalized)
        })
        .collect::<Result<Vec<_>, PolynomialError>>()?;
    let (parameter, dependent) = if derivatives[local[1]] != zero {
        (local[0], local[1])
    } else if derivatives[local[0]] != zero {
        (local[1], local[0])
    } else {
        return Err(CurveError::SingularCurve);
    };
    let derivative_inverse = field
        .inverse(&derivatives[dependent])
        .map_err(|_| CurveError::SingularCurve)?;

    let constant = |value: R::Element| LaurentGerm::constant(field.clone(), value, high);
    let mut coordinates = [
        constant(normalized[0].clone()),
        constant(normalized[1].clone()),
        constant(normalized[2].clone()),
    ];
    coordinates[chart] = constant(field.one());
    coordinates[parameter] =
        coordinates[parameter].add(&LaurentGerm::parameter(field.clone(), high))?;
    for order in 1..=precision {
        let residual = evaluate_homogeneous_germ(&point.curve.equation, &coordinates)?;
        let exponent = i64::try_from(order)
            .map_err(|_| CurveError::LocalExpansionLimit { limit: precision })?;
        let correction =
            field.neg(&field.mul(&residual.coefficient(exponent), &derivative_inverse)?)?;
        coordinates[dependent].set_coefficient(exponent, correction);
    }
    let residual = evaluate_homogeneous_germ(&point.curve.equation, &coordinates)?;
    if residual
        .coefficients
        .range(..=high)
        .any(|(_, coefficient)| coefficient != &zero)
    {
        return Err(CurveError::InvalidWitness(
            "implicit local expansion satisfies the curve equation",
        ));
    }
    Ok(coordinates)
}

fn evaluate_homogeneous_germ<R: Field>(
    polynomial: &Polynomial<R>,
    coordinates: &[LaurentGerm<R>; 3],
) -> Result<LaurentGerm<R>, CurveError> {
    let field = polynomial.ring().coefficient_ring().clone();
    let high = coordinates
        .iter()
        .map(|coordinate| coordinate.known_through)
        .min()
        .unwrap_or(0);
    let mut result = LaurentGerm::zero(field.clone(), high);
    for (exponent, coefficient) in polynomial.terms() {
        let mut term = LaurentGerm::constant(field.clone(), coefficient.clone(), high);
        for (coordinate, exponent) in coordinates.iter().zip(exponent) {
            term = term.mul(&coordinate.pow(integer_to_usize(exponent)?)?)?;
        }
        result = result.add(&term)?;
    }
    Ok(result)
}

fn evaluate_univariate_germ<R: Field>(
    polynomial: &Polynomial<R>,
    value: &LaurentGerm<R>,
) -> Result<LaurentGerm<R>, CurveError> {
    let mut result = LaurentGerm::zero(value.field.clone(), value.known_through);
    for (exponent, coefficient) in polynomial.terms() {
        let term = LaurentGerm::constant(
            value.field.clone(),
            coefficient.clone(),
            value.known_through,
        )
        .mul(&value.pow(integer_to_usize(&exponent[0])?)?)?;
        result = result.add(&term)?;
    }
    Ok(result)
}

impl<R: Field> fmt::Debug for CurveFunction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CurveFunction")
            .field("base", &R::TYPE_ID)
            .field("terms", &self.representative.nonzero_term_count())
            .finish()
    }
}

impl<R: Field> fmt::Display for CurveFunction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CurveFunction<{}>(terms={})",
            R::TYPE_ID,
            self.representative.nonzero_term_count()
        )
    }
}

impl<R: Field> SemanticType for CurveFunction<R> {
    const TYPE_ID: &'static str = "noether/commalg/CurveFunction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.field.canonical_bytes()?;
        append_component(&mut output, &self.representative.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CURVE_FUNCTION_DESCRIPTION
    }
}

impl<R: Field> Add<&CurveFunction<R>> for &CurveFunction<R> {
    type Output = Result<CurveFunction<R>, CurveError>;

    fn add(self, rhs: &CurveFunction<R>) -> Self::Output {
        self.field.checked_add(self, rhs)
    }
}

impl<R: Field> Sub<&CurveFunction<R>> for &CurveFunction<R> {
    type Output = Result<CurveFunction<R>, CurveError>;

    fn sub(self, rhs: &CurveFunction<R>) -> Self::Output {
        self.field.checked_sub(self, rhs)
    }
}

impl<R: Field> Mul<&CurveFunction<R>> for &CurveFunction<R> {
    type Output = Result<CurveFunction<R>, CurveError>;

    fn mul(self, rhs: &CurveFunction<R>) -> Self::Output {
        self.field.checked_mul(self, rhs)
    }
}

impl<R: Field> Div<&CurveFunction<R>> for &CurveFunction<R> {
    type Output = Result<CurveFunction<R>, CurveError>;

    fn div(self, rhs: &CurveFunction<R>) -> Self::Output {
        self.checked_div(rhs)
    }
}

impl<R: Field> Neg for &CurveFunction<R> {
    type Output = Result<CurveFunction<R>, CurveError>;

    fn neg(self) -> Self::Output {
        self.field.checked_neg(self)
    }
}

type ExtendedGcdCoefficientResult<R> = Result<
    (
        Polynomial<RationalFunctionField<R>>,
        Polynomial<RationalFunctionField<R>>,
    ),
    CurveError,
>;

fn extended_gcd_coefficient<R: Field>(
    modulus: &Polynomial<RationalFunctionField<R>>,
    value: &Polynomial<RationalFunctionField<R>>,
) -> ExtendedGcdCoefficientResult<R> {
    let (gcd, _, coefficient) = modulus.extended_gcd(value)?;
    Ok((gcd, coefficient))
}

fn coordinate_presentations() -> Vec<(usize, usize, usize)> {
    let mut presentations = Vec::with_capacity(6);
    for patch in (0..3).rev() {
        let remaining = (0..3).filter(|index| *index != patch).collect::<Vec<_>>();
        presentations.push((remaining[0], remaining[1], patch));
        presentations.push((remaining[1], remaining[0], patch));
    }
    presentations
}

fn integer_to_usize(value: &crate::Integer) -> Result<usize, CurveError> {
    value
        .to_u64_checked()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(CurveError::NoFunctionFieldPresentation)
}

fn integer_to_u64(value: &crate::Integer) -> Result<u64, CurveError> {
    value.to_u64_checked().ok_or(CurveError::ResourceLimit {
        resource: "polynomial or section exponent",
        limit: usize::MAX,
    })
}

fn small_exponents(values: [u64; 3]) -> Vec<crate::Integer> {
    values.into_iter().map(crate::Integer::from).collect()
}

fn natural_multiple<R: Field>(
    field: &R,
    value: &R::Element,
    mut factor: u64,
) -> Result<R::Element, CurveError> {
    let mut result = field.zero();
    let mut summand = value.clone();
    while factor > 0 {
        if factor & 1 == 1 {
            result = field.add(&result, &summand)?;
        }
        factor >>= 1;
        if factor > 0 {
            summand = field.add(&summand, &summand)?;
        }
    }
    Ok(result)
}

fn weierstrass_discriminant<R: Field>(
    field: &R,
    coefficients: &[R::Element; 5],
) -> Result<R::Element, CurveError> {
    let [a1, a2, a3, a4, a6] = coefficients;
    let a1_squared = field.mul(a1, a1)?;
    let a3_squared = field.mul(a3, a3)?;
    let a4_squared = field.mul(a4, a4)?;
    let b2 = field.add(&a1_squared, &natural_multiple(field, a2, 4)?)?;
    let b4 = field.add(&field.mul(a1, a3)?, &natural_multiple(field, a4, 2)?)?;
    let b6 = field.add(&a3_squared, &natural_multiple(field, a6, 4)?)?;
    let b8 = {
        let first = field.mul(&a1_squared, a6)?;
        let second = natural_multiple(field, &field.mul(a2, a6)?, 4)?;
        let third = field.mul(&field.mul(a1, a3)?, a4)?;
        let fourth = field.mul(a2, &a3_squared)?;
        let value = field.add(&first, &second)?;
        let value = field.sub(&value, &third)?;
        let value = field.add(&value, &fourth)?;
        field.sub(&value, &a4_squared)?
    };
    let term1 = field.neg(&field.mul(&field.mul(&b2, &b2)?, &b8)?)?;
    let b4_cubed = field.mul(&field.mul(&b4, &b4)?, &b4)?;
    let term2 = field.neg(&natural_multiple(field, &b4_cubed, 8)?)?;
    let term3 = field.neg(&natural_multiple(field, &field.mul(&b6, &b6)?, 27)?)?;
    let term4 = natural_multiple(field, &field.mul(&field.mul(&b2, &b4)?, &b6)?, 9)?;
    let value = field.add(&term1, &term2)?;
    let value = field.add(&value, &term3)?;
    field.add(&value, &term4).map_err(Into::into)
}

fn curve_context_error<R: Field>(error: CurveError) -> ContextMismatchError {
    match error {
        CurveError::ContextMismatch(error) => error,
        error => ContextMismatchError {
            expected: CurveFunctionField::<R>::TYPE_ID.to_owned(),
            actual: format!("curve-function arithmetic invariant failed: {error}"),
        },
    }
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len())
        .map_err(|_| NoetherError::Serialization("curve component is too large".to_owned()))?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

fn serialization_error(error: impl fmt::Display) -> NoetherError {
    NoetherError::Serialization(error.to_string())
}

fn curve_serialization_error(error: CurveError) -> NoetherError {
    NoetherError::Serialization(error.to_string())
}

fn encode_typed_json<const N: usize>(
    type_id: &str,
    fields: [(&str, serde_json::Value); N],
) -> Result<String, NoetherError> {
    let mut object = serde_json::Map::new();
    object.insert(
        "type".to_owned(),
        serde_json::Value::String(type_id.to_owned()),
    );
    for (name, value) in fields {
        object.insert(name.to_owned(), value);
    }
    serde_json::to_string(&object).map_err(serialization_error)
}

fn decode_typed_json(
    value: &str,
    expected_type: &str,
    fields: &[&str],
) -> Result<serde_json::Map<String, serde_json::Value>, NoetherError> {
    let value: serde_json::Value = serde_json::from_str(value).map_err(serialization_error)?;
    let mut object = value.as_object().cloned().ok_or_else(|| {
        NoetherError::Serialization("a typed curve value must be a JSON object".to_owned())
    })?;
    let actual_type = object
        .remove("type")
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| {
            NoetherError::Serialization("a typed curve value has no string type field".to_owned())
        })?;
    if actual_type != expected_type {
        return Err(NoetherError::Serialization(format!(
            "expected type {expected_type}, received {actual_type}"
        )));
    }
    let mut actual = object.keys().map(String::as_str).collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = fields.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(NoetherError::Serialization(format!(
            "serialized {expected_type} fields do not match the canonical schema"
        )));
    }
    Ok(object)
}

fn decode_field<T: DeserializeOwned>(
    object: &mut serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<T, NoetherError> {
    let value = object.remove(name).ok_or_else(|| {
        NoetherError::Serialization(format!("serialized curve value has no {name} field"))
    })?;
    serde_json::from_value(value).map_err(serialization_error)
}

fn nested_json(value: String) -> Result<serde_json::Value, NoetherError> {
    serde_json::from_str(&value).map_err(serialization_error)
}

fn encode_nested_field(
    object: &mut serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<String, NoetherError> {
    serde_json::to_string(&object.remove(name).ok_or_else(|| {
        NoetherError::Serialization(format!("serialized curve value has no {name} field"))
    })?)
    .map_err(serialization_error)
}

#[cfg(test)]
mod tests {
    use crate::{
        commalg::{
            Curve, CurveFunctionEvaluation, CurveFunctionOrder, CurveMorphism,
            RationalFunctionField, WeilDivisor,
        },
        Integer, Rational, Ring, Q,
    };

    fn rational(value: i64) -> Rational {
        Rational::from(value)
    }

    #[test]
    fn weierstrass_function_field_reduces_and_inverts_canonically() {
        let curve = Curve::smooth_projective_weierstrass(
            Q,
            [
                rational(0),
                rational(0),
                rational(0),
                rational(-1),
                rational(0),
            ],
        )
        .unwrap();
        assert_eq!(curve.smoothness_evidence().quotient_dimension(), 2);
        assert!(curve.smoothness_evidence().singular_cone_dimension() <= 0);
        curve.verify().unwrap();

        let functions = curve.function_field().unwrap();
        assert_eq!(functions.extension_degree(), 2);
        let x = functions.x().unwrap();
        let y = functions.y().unwrap();
        let y_squared = y.pow(2).unwrap();
        let x_cubed_minus_x = (&x.pow(3).unwrap() - &x).unwrap();
        assert_eq!(y_squared, x_cubed_minus_x);

        let value = (&x + &y).unwrap();
        let inverse = value.checked_inverse().unwrap();
        assert_eq!((&value * &inverse).unwrap(), functions.one());
        value.verify().unwrap();
        functions.verify().unwrap();
    }

    #[test]
    fn generic_smooth_plane_curve_gets_a_finite_function_field_presentation() {
        let ring = crate::commalg::PolynomialRing::new(
            Q,
            ["u", "v", "w"],
            crate::commalg::MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let equation = ring
            .from_terms([
                (vec![3.into(), 0.into(), 0.into()], rational(1)),
                (vec![0.into(), 3.into(), 0.into()], rational(1)),
                (vec![0.into(), 0.into(), 3.into()], rational(1)),
            ])
            .unwrap();
        let curve = Curve::smooth_projective_plane(equation).unwrap();
        let functions = curve.function_field().unwrap();
        assert_eq!(functions.extension_degree(), 3);
        assert_eq!(
            functions.modulus().leading_coefficient(),
            Some(functions.rational_subfield().one())
        );

        let constants = WeilDivisor::zero(curve).riemann_roch_space().unwrap();
        assert_eq!(constants.dimension(), 1);
        assert!(constants.basis()[0].is_one());
        constants.verify().unwrap();
    }

    #[test]
    fn singular_plane_and_weierstrass_models_fail_closed() {
        assert!(Curve::smooth_projective_weierstrass(
            Q,
            [
                rational(0),
                rational(0),
                rational(0),
                rational(0),
                rational(0)
            ]
        )
        .is_err());

        let ring = crate::commalg::PolynomialRing::new(
            Q,
            ["x", "y", "z"],
            crate::commalg::MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let x = ring.generator(0).unwrap();
        let singular = x.pow(2).unwrap();
        assert!(Curve::smooth_projective_plane(singular).is_err());
    }

    #[test]
    fn local_evaluation_distinguishes_regular_values_zeros_and_poles() {
        let curve = Curve::smooth_projective_weierstrass(
            Q,
            [
                rational(0),
                rational(0),
                rational(0),
                rational(-1),
                rational(0),
            ],
        )
        .unwrap();
        let point = curve
            .point([rational(0), rational(0), rational(1)])
            .unwrap();
        let infinity = curve
            .point([rational(0), rational(1), rational(0)])
            .unwrap();
        let functions = curve.function_field().unwrap();
        let x = functions.x().unwrap();
        let y = functions.y().unwrap();
        let inverse_x = x.checked_inverse().unwrap();
        let x_plus_one = (&x + &functions.one()).unwrap();

        assert_eq!(
            x.order_at(&point).unwrap(),
            CurveFunctionOrder::Finite(Integer::from(2))
        );
        assert_eq!(
            x.evaluate_at(&point).unwrap(),
            CurveFunctionEvaluation::Zero {
                order: Some(Integer::from(2))
            }
        );
        assert_eq!(
            inverse_x.evaluate_at(&point).unwrap(),
            CurveFunctionEvaluation::Pole {
                order: Integer::from(2)
            }
        );
        assert_eq!(
            x_plus_one.evaluate_at(&point).unwrap(),
            CurveFunctionEvaluation::Regular(rational(1))
        );
        assert_eq!(
            x.order_at(&infinity).unwrap(),
            CurveFunctionOrder::Finite(Integer::from(-2))
        );
        assert_eq!(
            y.order_at(&infinity).unwrap(),
            CurveFunctionOrder::Finite(Integer::from(-3))
        );
        let principal_x = x.principal_divisor().unwrap();
        let point_place = point.as_closed_point().unwrap();
        let infinity_place = infinity.as_closed_point().unwrap();
        assert_eq!(
            principal_x.coefficient(&point_place).unwrap(),
            Integer::from(2)
        );
        assert_eq!(
            principal_x.coefficient(&infinity_place).unwrap(),
            Integer::from(-2)
        );
        x.verify_principal_divisor(&principal_x).unwrap();
        point.verify().unwrap();
        infinity.verify().unwrap();
    }

    #[test]
    fn rational_points_embed_as_degree_one_places_and_divisors_are_canonical() {
        let curve = Curve::smooth_projective_weierstrass(
            Q,
            [
                rational(0),
                rational(0),
                rational(0),
                rational(-1),
                rational(0),
            ],
        )
        .unwrap();
        let point = curve
            .point([rational(0), rational(0), rational(1)])
            .unwrap()
            .as_closed_point()
            .unwrap();
        let infinity = curve
            .point([rational(0), rational(1), rational(0)])
            .unwrap()
            .as_closed_point()
            .unwrap();
        assert_eq!(point.degree(), &Integer::one());
        assert!(point.rational_point().is_some());

        let divisor = WeilDivisor::new(
            curve.clone(),
            [
                (point.clone(), Integer::from(-2)),
                (infinity.clone(), Integer::from(5)),
                (point.clone(), Integer::from(-3)),
            ],
        )
        .unwrap();
        assert_eq!(divisor.terms().len(), 2);
        assert_eq!(divisor.coefficient(&point).unwrap(), Integer::from(-5));
        assert_eq!(divisor.coefficient(&infinity).unwrap(), Integer::from(5));
        assert_eq!(divisor.degree(), Integer::zero());
        assert_eq!(divisor.positive_part().degree(), Integer::from(5));
        assert_eq!(divisor.negative_part().degree(), Integer::from(5));
        assert_eq!((&divisor + &(-&divisor)).unwrap(), WeilDivisor::zero(curve));
        divisor.verify().unwrap();
        let identity = CurveMorphism::identity(divisor.curve().clone()).unwrap();
        assert_eq!(divisor.pullback(&identity).unwrap(), divisor);
        identity.verify().unwrap();
        let ring = divisor.curve().equation().ring();
        let scaled_identity = CurveMorphism::new(
            divisor.curve().clone(),
            divisor.curve().clone(),
            [
                ring.generator(0).unwrap().scalar_mul(&rational(2)).unwrap(),
                ring.generator(1).unwrap().scalar_mul(&rational(2)).unwrap(),
                ring.generator(2).unwrap().scalar_mul(&rational(2)).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(scaled_identity, identity);

        let pole_divisor =
            WeilDivisor::new(infinity.curve().clone(), [(infinity, Integer::from(2))]).unwrap();
        let space = pole_divisor.riemann_roch_space().unwrap();
        assert_eq!(space.dimension(), 2);
        space.verify().unwrap();
    }

    #[test]
    fn fisher_order_five_riemann_roch_space_has_the_expected_generator() {
        let coefficients = RationalFunctionField::new(Q, "lambda").unwrap();
        let lambda = coefficients.generator();
        let one = coefficients.one();
        let zero = coefficients.zero();
        let one_minus_lambda = (&one - &lambda).unwrap();
        let minus_lambda = (-&lambda).unwrap();
        let curve = Curve::smooth_projective_weierstrass(
            coefficients.clone(),
            [
                one_minus_lambda,
                minus_lambda.clone(),
                minus_lambda,
                zero.clone(),
                zero.clone(),
            ],
        )
        .unwrap();
        let point = curve
            .point([zero.clone(), zero.clone(), one.clone()])
            .unwrap()
            .as_closed_point()
            .unwrap();
        let infinity = curve
            .point([zero.clone(), one.clone(), zero])
            .unwrap()
            .as_closed_point()
            .unwrap();
        let divisor = WeilDivisor::new(
            curve.clone(),
            [(infinity, Integer::from(5)), (point, Integer::from(-5))],
        )
        .unwrap();
        let space = divisor.riemann_roch_space().unwrap();
        assert_eq!(space.dimension(), 1);

        let functions = curve.function_field().unwrap();
        let x = functions.x().unwrap();
        let y = functions.y().unwrap();
        let minus_x_minus_one = (-&(&x + &functions.one()).unwrap()).unwrap();
        let expected = (&(&minus_x_minus_one * &y).unwrap() + &x.pow(2).unwrap()).unwrap();
        let quotient = space.basis()[0].checked_div(&expected).unwrap();
        assert!(quotient.as_constant().unwrap().is_some());
    }

    #[test]
    fn principal_divisor_retains_a_quadratic_closed_point() {
        let curve = Curve::smooth_projective_weierstrass(
            Q,
            [
                rational(0),
                rational(0),
                rational(0),
                rational(-1),
                rational(0),
            ],
        )
        .unwrap();
        let functions = curve.function_field().unwrap();
        let x_minus_two =
            (&functions.x().unwrap() - &functions.constant(rational(2)).unwrap()).unwrap();
        let divisor = x_minus_two.principal_divisor().unwrap();
        assert_eq!(divisor.degree(), Integer::zero());
        let quadratic = divisor
            .terms()
            .iter()
            .find(|(point, coefficient)| {
                point.degree() == &Integer::from(2) && coefficient == &Integer::one()
            })
            .map(|(point, _)| point)
            .expect("the conjugate roots define one quadratic place");
        assert!(quadratic.rational_point().is_none());
        assert!(divisor
            .terms()
            .iter()
            .any(|(point, coefficient)| point.degree() == &Integer::one()
                && coefficient == &Integer::from(-2)));
    }

    #[test]
    fn curve_values_round_trip_through_revalidated_typed_json() {
        let curve = Curve::smooth_projective_weierstrass(
            Q,
            [
                rational(0),
                rational(0),
                rational(0),
                rational(-1),
                rational(0),
            ],
        )
        .unwrap();
        let functions = curve.function_field().unwrap();
        let point = curve
            .point([rational(0), rational(0), rational(1)])
            .unwrap();
        let infinity = curve
            .point([rational(0), rational(1), rational(0)])
            .unwrap();
        let function = (&functions.x().unwrap() + &functions.y().unwrap()).unwrap();
        let divisor = functions.x().unwrap().principal_divisor().unwrap();
        let poles = WeilDivisor::new(
            curve.clone(),
            [(infinity.as_closed_point().unwrap(), Integer::from(2))],
        )
        .unwrap();
        let space = poles.riemann_roch_space().unwrap();
        let identity = CurveMorphism::identity(curve.clone()).unwrap();

        assert_eq!(
            Curve::<crate::RationalField>::from_json(&curve.to_json().unwrap()).unwrap(),
            curve
        );
        assert_eq!(
            super::CurveFunctionField::<crate::RationalField>::from_json(
                &functions.to_json().unwrap()
            )
            .unwrap(),
            functions
        );
        assert_eq!(
            super::CurvePoint::<crate::RationalField>::from_json(&point.to_json().unwrap())
                .unwrap(),
            point
        );
        assert_eq!(
            super::CurveClosedPoint::<crate::RationalField>::from_json(
                &point.as_closed_point().unwrap().to_json().unwrap()
            )
            .unwrap(),
            point.as_closed_point().unwrap()
        );
        assert_eq!(
            super::CurveFunction::<crate::RationalField>::from_json(&function.to_json().unwrap())
                .unwrap(),
            function
        );
        let shared_denominator = (&(&functions.one() + &functions.y().unwrap()).unwrap()
            / &functions.x().unwrap())
            .unwrap();
        assert_eq!(
            super::CurveFunction::<crate::RationalField>::from_json(
                &shared_denominator.to_json().unwrap()
            )
            .unwrap(),
            shared_denominator
        );
        assert_eq!(
            super::WeilDivisor::<crate::RationalField>::from_json(&divisor.to_json().unwrap())
                .unwrap(),
            divisor
        );
        assert_eq!(
            super::RiemannRochSpace::<crate::RationalField>::from_json(&space.to_json().unwrap())
                .unwrap(),
            space
        );
        assert_eq!(
            super::CurveMorphism::<crate::RationalField>::from_json(&identity.to_json().unwrap())
                .unwrap(),
            identity
        );

        let x_minus_two =
            (&functions.x().unwrap() - &functions.constant(rational(2)).unwrap()).unwrap();
        let quadratic = x_minus_two
            .principal_divisor()
            .unwrap()
            .support()
            .find(|point| point.degree() == &Integer::from(2))
            .cloned()
            .unwrap();
        assert_eq!(
            super::CurveClosedPoint::<crate::RationalField>::from_json(
                &quadratic.to_json().unwrap()
            )
            .unwrap(),
            quadratic
        );

        let mut tampered: serde_json::Value =
            serde_json::from_str(&function.to_json().unwrap()).unwrap();
        let patch = curve.equation.ring().generator(2).unwrap();
        let (numerator, denominator) = function.homogeneous_representation().unwrap();
        tampered["numerator"] = serde_json::to_value((&numerator * &patch).unwrap()).unwrap();
        tampered["denominator"] = serde_json::to_value((&denominator * &patch).unwrap()).unwrap();
        assert!(super::CurveFunction::<crate::RationalField>::from_json(
            &serde_json::to_string(&tampered).unwrap()
        )
        .is_err());
    }

    #[test]
    fn finite_field_nonrational_places_revalidate_from_json() {
        let field = crate::numthy::FiniteField::prime(5).unwrap();
        let curve = Curve::smooth_projective_weierstrass(
            field.clone(),
            [
                field.zero(),
                field.zero(),
                field.zero(),
                field.one(),
                field.one(),
            ],
        )
        .unwrap();
        let functions = curve.function_field().unwrap();
        let x_minus_one = (&functions.x().unwrap() - &functions.one()).unwrap();
        let divisor = x_minus_one.principal_divisor().unwrap();
        let quadratic = divisor
            .support()
            .find(|point| point.degree() == &Integer::from(2))
            .cloned()
            .unwrap();

        assert_eq!(
            super::CurveClosedPoint::<crate::numthy::FiniteField>::from_json(
                &quadratic.to_json().unwrap()
            )
            .unwrap(),
            quadratic
        );
        assert_eq!(
            super::WeilDivisor::<crate::numthy::FiniteField>::from_json(
                &divisor.to_json().unwrap()
            )
            .unwrap(),
            divisor
        );
    }
}
