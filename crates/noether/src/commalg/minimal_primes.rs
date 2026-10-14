// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Verified minimal primes of polynomial ideals over perfect fields.

use core::fmt;
use std::collections::BTreeSet;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::linalg::Matrix;
use crate::numthy::FiniteField;
use crate::{Field, NoetherError, RationalField, Ring, SemanticType, TypeDescription};

use super::{
    DecompositionError, GroebnerError, Ideal, Polynomial, PolynomialFactorization, PolynomialRing,
    RadicalIdeal, RationalUnivariateRepresentation,
};

const MAX_ZERO_DIMENSIONAL_COMPONENTS: usize = 4_096;
const MAX_MONOMIAL_TRANSVERSALS: usize = 4_096;
const MAX_MONOMIAL_TRANSVERSAL_CANDIDATES: usize = 32_768;

static POLYNOMIAL_PRIME_CERTIFICATE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PolynomialPrimeCertificate@1",
    rust_path: "noether::commalg::PolynomialPrimeCertificate",
    python_path: Some("noether.commalg.PolynomialPrimeCertificate"),
};

static MINIMAL_PRIME_DECOMPOSITION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/MinimalPrimeDecomposition@1",
    rust_path: "noether::commalg::MinimalPrimeDecomposition",
    python_path: Some("noether.commalg.MinimalPrimeDecomposition"),
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "method",
    rename_all = "snake_case",
    bound(
        serialize = "R: Serialize, R::Element: Serialize",
        deserialize = "R: Deserialize<'de>, R::Element: Deserialize<'de>"
    )
)]
enum PolynomialPrimeProof<R: Field> {
    IrreducibleHypersurface {
        factor: Polynomial<R>,
    },
    CoordinatePrime {
        variables: Vec<usize>,
    },
    AffineLinear {
        pivot_variables: Vec<usize>,
    },
    PolynomialGraph {
        pivot_variables: Vec<usize>,
    },
    SmoothProjectiveCompleteIntersection {
        codimension: usize,
        jacobian_minors: Vec<Polynomial<R>>,
        singular_locus: Ideal<PolynomialRing<R>>,
        projective_singular_saturation: Ideal<PolynomialRing<R>>,
    },
    HomogeneousAffinePatch {
        variable: usize,
        saturation: Ideal<PolynomialRing<R>>,
        affine_patch: Ideal<PolynomialRing<R>>,
        affine_patch_proof: Box<PolynomialPrimeProof<R>>,
    },
    ZeroDimensionalField {
        representation: Box<RationalUnivariateRepresentation<R>>,
    },
}

/// Exact evidence that a polynomial ideal is prime.
///
/// The admitted proof routes are an irreducible hypersurface in a polynomial
/// UFD; a coordinate, affine-linear, or polynomial-graph ideal with
/// polynomial-ring quotient; a smooth positive-dimensional projective
/// complete intersection; a homogeneous ideal embedded in a prime affine
/// localization; or a zero-dimensional quotient presented as a field by an
/// irreducible rational univariate representation.
#[derive(Clone, PartialEq, Eq)]
pub struct PolynomialPrimeCertificate<R: Field> {
    ideal: Ideal<PolynomialRing<R>>,
    proof: PolynomialPrimeProof<R>,
}

impl<R: Field> PolynomialPrimeCertificate<R> {
    /// Canonical polynomial ideal certified as prime.
    pub fn ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.ideal
    }

    /// Canonical name of the exact primality proof route.
    pub const fn proof_kind(&self) -> &'static str {
        match &self.proof {
            PolynomialPrimeProof::IrreducibleHypersurface { .. } => "irreducible_hypersurface",
            PolynomialPrimeProof::CoordinatePrime { .. } => "coordinate_prime",
            PolynomialPrimeProof::AffineLinear { .. } => "affine_linear",
            PolynomialPrimeProof::PolynomialGraph { .. } => "polynomial_graph",
            PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. } => {
                "smooth_projective_complete_intersection"
            }
            PolynomialPrimeProof::HomogeneousAffinePatch { .. } => "homogeneous_affine_patch",
            PolynomialPrimeProof::ZeroDimensionalField { .. } => "zero_dimensional_field",
        }
    }

    /// Irreducible generator when the hypersurface route applies.
    pub fn irreducible_factor(&self) -> Option<&Polynomial<R>> {
        match &self.proof {
            PolynomialPrimeProof::IrreducibleHypersurface { factor } => Some(factor),
            PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Sorted coordinate indices generating a coordinate prime.
    pub fn coordinate_variables(&self) -> Option<&[usize]> {
        match &self.proof {
            PolynomialPrimeProof::CoordinatePrime { variables } => Some(variables),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Sorted leading pivot variables for an affine-linear prime.
    pub fn affine_linear_pivots(&self) -> Option<&[usize]> {
        match &self.proof {
            PolynomialPrimeProof::AffineLinear { pivot_variables } => Some(pivot_variables),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Sorted pivot variables eliminated by a polynomial-graph prime.
    pub fn polynomial_graph_pivots(&self) -> Option<&[usize]> {
        match &self.proof {
            PolynomialPrimeProof::PolynomialGraph { pivot_variables } => Some(pivot_variables),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Codimension of a smooth projective complete-intersection proof.
    pub fn complete_intersection_codimension(&self) -> Option<usize> {
        match &self.proof {
            PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { codimension, .. } => {
                Some(*codimension)
            }
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Jacobian minors in a smooth projective complete-intersection proof.
    pub fn jacobian_minors(&self) -> Option<&[Polynomial<R>]> {
        match &self.proof {
            PolynomialPrimeProof::SmoothProjectiveCompleteIntersection {
                jacobian_minors, ..
            } => Some(jacobian_minors),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Affine-cone singular-locus ideal in the projective proof.
    pub fn singular_locus_ideal(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            PolynomialPrimeProof::SmoothProjectiveCompleteIntersection {
                singular_locus, ..
            } => Some(singular_locus),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Saturation proving that the projective singular locus is empty.
    pub fn projective_singular_saturation(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            PolynomialPrimeProof::SmoothProjectiveCompleteIntersection {
                projective_singular_saturation,
                ..
            } => Some(projective_singular_saturation),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Homogeneous coordinate inverted by the affine-patch proof.
    pub const fn homogeneous_patch_variable(&self) -> Option<usize> {
        match &self.proof {
            PolynomialPrimeProof::HomogeneousAffinePatch { variable, .. } => Some(*variable),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Recomputed `I : x_i^infinity` proving injection into the patch localization.
    pub fn homogeneous_patch_saturation(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            PolynomialPrimeProof::HomogeneousAffinePatch { saturation, .. } => Some(saturation),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Prime certificate for the affine slice `I + (x_i - 1)`.
    pub fn homogeneous_patch_prime(&self) -> Option<PolynomialPrimeCertificate<R>> {
        match &self.proof {
            PolynomialPrimeProof::HomogeneousAffinePatch {
                affine_patch,
                affine_patch_proof,
                ..
            } => Some(PolynomialPrimeCertificate {
                ideal: affine_patch.clone(),
                proof: affine_patch_proof.as_ref().clone(),
            }),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. }
            | PolynomialPrimeProof::ZeroDimensionalField { .. } => None,
        }
    }

    /// Irreducible RUR when the zero-dimensional field route applies.
    pub fn rational_univariate_representation(
        &self,
    ) -> Option<&RationalUnivariateRepresentation<R>> {
        match &self.proof {
            PolynomialPrimeProof::ZeroDimensionalField { representation } => Some(representation),
            PolynomialPrimeProof::IrreducibleHypersurface { .. }
            | PolynomialPrimeProof::CoordinatePrime { .. }
            | PolynomialPrimeProof::AffineLinear { .. }
            | PolynomialPrimeProof::PolynomialGraph { .. }
            | PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. }
            | PolynomialPrimeProof::HomogeneousAffinePatch { .. } => None,
        }
    }

    /// Exact theorem turning the retained evidence into a primality claim.
    pub const fn theorem(&self) -> &'static str {
        match &self.proof {
            PolynomialPrimeProof::IrreducibleHypersurface { .. } => {
                "an irreducible element generates a prime ideal in a polynomial UFD"
            }
            PolynomialPrimeProof::CoordinatePrime { .. } => {
                "the quotient by a coordinate ideal is a polynomial ring over a field"
            }
            PolynomialPrimeProof::AffineLinear { .. } => {
                "the quotient by a proper affine-linear ideal is a polynomial ring over a field"
            }
            PolynomialPrimeProof::PolynomialGraph { .. } => {
                "the quotient by a polynomial graph ideal is a polynomial ring over its free variables"
            }
            PolynomialPrimeProof::SmoothProjectiveCompleteIntersection { .. } => {
                "a positive-dimensional smooth projective complete intersection is geometrically connected and therefore geometrically integral"
            }
            PolynomialPrimeProof::HomogeneousAffinePatch { .. } => {
                "a homogeneous quotient injecting into a prime affine-patch localization is a domain"
            }
            PolynomialPrimeProof::ZeroDimensionalField { .. } => {
                "a zero-dimensional ideal is prime iff its quotient is a field"
            }
        }
    }
}

impl<R: Field> fmt::Debug for PolynomialPrimeCertificate<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolynomialPrimeCertificate")
            .field("proof_kind", &self.proof_kind())
            .field("generators", &self.ideal.generators().len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for PolynomialPrimeCertificate<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PolynomialPrimeCertificate(proof={}, generators={})",
            self.proof_kind(),
            self.ideal.generators().len()
        )
    }
}

impl<R> SemanticType for PolynomialPrimeCertificate<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/PolynomialPrimeCertificate@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &POLYNOMIAL_PRIME_CERTIFICATE_DESCRIPTION
    }
}

/// Verified minimal primes of an admitted polynomial ideal.
///
/// `radical` proves the passage from the source to its reduced quotient.
/// Every component has typed exact primality evidence. The canonically ordered
/// component intersection reconstructs the radical, proving completeness.
#[derive(Clone, PartialEq, Eq)]
pub struct MinimalPrimeDecomposition<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    radical: RadicalIdeal<R>,
    components: Vec<Ideal<PolynomialRing<R>>>,
    prime_certificates: Vec<PolynomialPrimeCertificate<R>>,
}

impl<R: Field> MinimalPrimeDecomposition<R> {
    /// Original ideal whose minimal primes are represented.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Verified radical of the source ideal.
    pub fn radical(&self) -> &RadicalIdeal<R> {
        &self.radical
    }

    /// Canonically ordered minimal prime ideals.
    pub fn components(&self) -> &[Ideal<PolynomialRing<R>>] {
        &self.components
    }

    /// Exact primality certificate for each component, in component order.
    pub fn prime_certificates(&self) -> &[PolynomialPrimeCertificate<R>] {
        &self.prime_certificates
    }

    /// Exact completeness theorem for the certified component list.
    pub const fn theorem(&self) -> &'static str {
        "the minimal primes intersect to the radical of the source ideal"
    }
}

impl<R: Field> fmt::Debug for MinimalPrimeDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MinimalPrimeDecomposition")
            .field("components", &self.components.len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for MinimalPrimeDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MinimalPrimeDecomposition(components={})",
            self.components.len()
        )
    }
}

impl<R> SemanticType for MinimalPrimeDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/MinimalPrimeDecomposition@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &MINIMAL_PRIME_DECOMPOSITION_DESCRIPTION
    }
}

pub(crate) trait MinimalPrimeCoefficientField:
    Field + super::radical::RadicalCoefficientField
{
    fn factor_polynomial(
        value: &Polynomial<Self>,
    ) -> Result<PolynomialFactorization<Self>, DecompositionError>;
}

impl MinimalPrimeCoefficientField for RationalField {
    fn factor_polynomial(
        value: &Polynomial<Self>,
    ) -> Result<PolynomialFactorization<Self>, DecompositionError> {
        value.factor().map_err(Into::into)
    }
}

impl MinimalPrimeCoefficientField for FiniteField {
    fn factor_polynomial(
        value: &Polynomial<Self>,
    ) -> Result<PolynomialFactorization<Self>, DecompositionError> {
        value.factor().map_err(Into::into)
    }
}

fn evaluate_univariate<R: Field>(
    value: &Polynomial<R>,
    argument: &Polynomial<R>,
) -> Result<Polynomial<R>, DecompositionError> {
    let target = argument.ring();
    let mut result = target.zero();
    for (exponent, coefficient) in value.terms() {
        let exponent = exponent[0].to_u64_checked().ok_or_else(|| {
            DecompositionError::Unsupported(
                "a splitting-polynomial exponent does not fit u64".to_owned(),
            )
        })?;
        let power = argument.pow(exponent)?;
        let term = power.scalar_mul(coefficient)?;
        result = target.add(&result, &term)?;
    }
    Ok(result)
}

fn linear_form<R: Field>(
    ring: &PolynomialRing<R>,
    coefficients: &[R::Element],
) -> Result<Polynomial<R>, DecompositionError> {
    let variables = ring.generators()?;
    if variables.len() != coefficients.len() {
        return Err(DecompositionError::Invalid(
            "an RUR splitting form has the wrong arity".to_owned(),
        ));
    }
    variables
        .iter()
        .zip(coefficients)
        .try_fold(ring.zero(), |sum, (variable, coefficient)| {
            let term = variable.scalar_mul(coefficient)?;
            ring.add(&sum, &term).map_err(Into::into)
        })
}

fn multiplication_matrix<R: Field>(
    quotient: &super::FiniteQuotientAlgebra<R>,
    variable: usize,
) -> Result<Matrix<R>, DecompositionError> {
    let dimension = quotient.dimension();
    let field = quotient.source().ring().coefficient_ring();
    let mut rows = vec![vec![field.zero(); dimension]; dimension];
    for (row, column, coefficient) in quotient.multiplication_operator_entries(variable)? {
        rows[row][column] = coefficient;
    }
    if dimension == 0 {
        Ok(Matrix::zeros(field.clone(), 0, 0))
    } else {
        Matrix::from_rows(field.clone(), rows).map_err(Into::into)
    }
}

fn map_parameter_to_variable<R: Field>(
    value: &Polynomial<R>,
    target: &PolynomialRing<R>,
    variable: usize,
) -> Result<Polynomial<R>, DecompositionError> {
    let arity = target.variables().len();
    let terms = value
        .terms()
        .map(|(exponent, coefficient)| {
            let mut target_exponent = vec![0.into(); arity];
            target_exponent[variable] = exponent[0].clone();
            (target_exponent, coefficient.clone())
        })
        .collect::<Vec<_>>();
    target.from_terms(terms).map_err(Into::into)
}

fn extend_ideal<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
    equation: Polynomial<R>,
) -> Result<Ideal<PolynomialRing<R>>, DecompositionError> {
    source
        .ring()
        .ideal(
            source
                .generators()
                .iter()
                .cloned()
                .chain(core::iter::once(equation)),
        )
        .map_err(Into::into)
}

fn rur_components<R: MinimalPrimeCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    representation: &RationalUnivariateRepresentation<R>,
) -> Result<Vec<Ideal<PolynomialRing<R>>>, DecompositionError> {
    if !representation.is_reduced() {
        return Err(DecompositionError::Invalid(
            "a radical splitting RUR is not reduced".to_owned(),
        ));
    }
    let factorization = R::factor_polynomial(representation.equation())?;
    factorization.verify()?;
    if factorization.factors().len() > MAX_ZERO_DIMENSIONAL_COMPONENTS {
        return Err(DecompositionError::ResourceLimit {
            resource: "zero-dimensional minimal-prime count",
            limit: MAX_ZERO_DIMENSIONAL_COMPONENTS,
        });
    }
    let parameter = linear_form(source.ring(), representation.linear_form())?;
    factorization
        .factors()
        .iter()
        .map(|(factor, multiplicity)| {
            if *multiplicity != 1 {
                return Err(DecompositionError::Invalid(
                    "a reduced RUR equation has a repeated factor".to_owned(),
                ));
            }
            extend_ideal(source, evaluate_univariate(factor, &parameter)?)
        })
        .collect()
}

fn coordinate_components<R: MinimalPrimeCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<Vec<Ideal<PolynomialRing<R>>>>, DecompositionError> {
    let quotient = source.finite_quotient_algebra()?;
    for variable in 0..source.ring().variables().len() {
        let characteristic = multiplication_matrix(&quotient, variable)?.charpoly()?;
        let factorization = R::factor_polynomial(&characteristic)?;
        factorization.verify()?;
        if factorization.factors().len() <= 1 {
            continue;
        }
        if factorization.factors().len() > MAX_ZERO_DIMENSIONAL_COMPONENTS {
            return Err(DecompositionError::ResourceLimit {
                resource: "zero-dimensional coordinate split count",
                limit: MAX_ZERO_DIMENSIONAL_COMPONENTS,
            });
        }
        let components = factorization
            .factors()
            .iter()
            .map(|(factor, _)| {
                extend_ideal(
                    source,
                    map_parameter_to_variable(factor, source.ring(), variable)?,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Some(components));
    }
    Ok(None)
}

fn decompose_radical<R: MinimalPrimeCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    remaining_depth: usize,
) -> Result<Vec<Ideal<PolynomialRing<R>>>, DecompositionError> {
    if source.generators().len() == 1 && source.generators()[0].is_one() {
        return Ok(Vec::new());
    }
    match source.rational_univariate_representation() {
        Ok(representation) => return rur_components(source, &representation),
        Err(GroebnerError::ResourceLimit {
            resource: "RUR primitive linear forms",
            ..
        }) => {}
        Err(error) => return Err(error.into()),
    }
    if remaining_depth == 0 {
        return Err(DecompositionError::ResourceLimit {
            resource: "zero-dimensional splitting depth",
            limit: MAX_ZERO_DIMENSIONAL_COMPONENTS,
        });
    }
    let children = coordinate_components(source)?.ok_or_else(|| {
        DecompositionError::Unsupported(
            "the reduced quotient is non-monogenic and no coordinate separates its components"
                .to_owned(),
        )
    })?;
    let mut components = Vec::new();
    for child in children {
        if &child == source {
            return Err(DecompositionError::Invalid(
                "a coordinate split did not decrease the quotient".to_owned(),
            ));
        }
        components.extend(decompose_radical(&child, remaining_depth - 1)?);
        if components.len() > MAX_ZERO_DIMENSIONAL_COMPONENTS {
            return Err(DecompositionError::ResourceLimit {
                resource: "zero-dimensional minimal-prime count",
                limit: MAX_ZERO_DIMENSIONAL_COMPONENTS,
            });
        }
    }
    Ok(components)
}

fn irreducible_rur<R: MinimalPrimeCoefficientField>(
    component: &Ideal<PolynomialRing<R>>,
) -> Result<RationalUnivariateRepresentation<R>, DecompositionError> {
    let representation = component.rational_univariate_representation()?;
    representation.verify()?;
    let factorization = R::factor_polynomial(representation.equation())?;
    factorization.verify()?;
    if representation.quotient_dimension() == 0
        || factorization.factors().len() != 1
        || factorization.factors()[0].1 != 1
    {
        return Err(DecompositionError::Invalid(
            "a minimal-prime quotient does not have an irreducible RUR equation".to_owned(),
        ));
    }
    Ok(representation)
}

fn zero_dimensional_prime_certificate<R: MinimalPrimeCoefficientField>(
    component: &Ideal<PolynomialRing<R>>,
) -> Result<PolynomialPrimeCertificate<R>, DecompositionError> {
    Ok(PolynomialPrimeCertificate {
        ideal: component.clone(),
        proof: PolynomialPrimeProof::ZeroDimensionalField {
            representation: Box::new(irreducible_rur(component)?),
        },
    })
}

fn hypersurface_prime_certificate<R: MinimalPrimeCoefficientField>(
    factor: &Polynomial<R>,
) -> Result<PolynomialPrimeCertificate<R>, DecompositionError> {
    let ideal = factor.ring().ideal([factor.clone()])?;
    let certificate = PolynomialPrimeCertificate {
        ideal,
        proof: PolynomialPrimeProof::IrreducibleHypersurface {
            factor: factor.clone(),
        },
    };
    verify_polynomial_prime_certificate(&certificate)?;
    Ok(certificate)
}

fn affine_linear_prime_certificate<R: MinimalPrimeCoefficientField>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<PolynomialPrimeCertificate<R>, DecompositionError> {
    let pivot_variables = super::radical::affine_linear_pivots(ideal).ok_or_else(|| {
        DecompositionError::Invalid(
            "an affine-linear prime candidate is not a proper affine-linear ideal".to_owned(),
        )
    })?;
    let certificate = PolynomialPrimeCertificate {
        ideal: ideal.clone(),
        proof: PolynomialPrimeProof::AffineLinear { pivot_variables },
    };
    verify_polynomial_prime_certificate(&certificate)?;
    Ok(certificate)
}

fn polynomial_graph_prime_certificate<R: MinimalPrimeCoefficientField>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<PolynomialPrimeCertificate<R>, DecompositionError> {
    let pivot_variables = super::radical::polynomial_graph_pivots(ideal).ok_or_else(|| {
        DecompositionError::Invalid(
            "a polynomial-graph prime candidate is not a polynomial graph".to_owned(),
        )
    })?;
    let certificate = PolynomialPrimeCertificate {
        ideal: ideal.clone(),
        proof: PolynomialPrimeProof::PolynomialGraph { pivot_variables },
    };
    verify_polynomial_prime_certificate(&certificate)?;
    Ok(certificate)
}

struct SmoothProjectiveCompleteIntersectionData<R: Field> {
    codimension: usize,
    jacobian_minors: Vec<Polynomial<R>>,
    singular_locus: Ideal<PolynomialRing<R>>,
    projective_singular_saturation: Ideal<PolynomialRing<R>>,
}

fn is_unit_ideal<R: Field>(ideal: &Ideal<PolynomialRing<R>>) -> bool {
    ideal.generators().len() == 1 && ideal.generators()[0].is_one()
}

fn smooth_projective_complete_intersection_data<R: MinimalPrimeCoefficientField>(
    radical: &super::RadicalIdeal<R>,
) -> Result<Option<SmoothProjectiveCompleteIntersectionData<R>>, DecompositionError> {
    let Some(codimension) = radical.complete_intersection_codimension() else {
        return Ok(None);
    };
    if radical.ideal().dimension()? < 2 {
        return Ok(None);
    }
    for generator in radical.ideal().generators() {
        if generator.homogeneous_degree()?.is_none() {
            return Ok(None);
        }
    }
    let irrelevant = radical
        .ideal()
        .ring()
        .ideal(radical.ideal().ring().generators()?)?;
    let singular_locus = radical.singular_locus_ideal().ok_or_else(|| {
        DecompositionError::Invalid(
            "a complete-intersection radical omitted its singular locus".to_owned(),
        )
    })?;
    let projective_singular_saturation = singular_locus.saturate(&irrelevant)?;
    if !is_unit_ideal(&projective_singular_saturation) {
        return Ok(None);
    }
    Ok(Some(SmoothProjectiveCompleteIntersectionData {
        codimension,
        jacobian_minors: radical
            .jacobian_minors()
            .ok_or_else(|| {
                DecompositionError::Invalid(
                    "a complete-intersection radical omitted its Jacobian minors".to_owned(),
                )
            })?
            .to_vec(),
        singular_locus: singular_locus.clone(),
        projective_singular_saturation,
    }))
}

fn smooth_projective_complete_intersection_prime_certificate<R: MinimalPrimeCoefficientField>(
    radical: &super::RadicalIdeal<R>,
) -> Result<Option<PolynomialPrimeCertificate<R>>, DecompositionError> {
    let Some(data) = smooth_projective_complete_intersection_data(radical)? else {
        return Ok(None);
    };
    let certificate = PolynomialPrimeCertificate {
        ideal: radical.ideal().clone(),
        proof: PolynomialPrimeProof::SmoothProjectiveCompleteIntersection {
            codimension: data.codimension,
            jacobian_minors: data.jacobian_minors,
            singular_locus: data.singular_locus,
            projective_singular_saturation: data.projective_singular_saturation,
        },
    };
    verify_polynomial_prime_certificate(&certificate)?;
    Ok(Some(certificate))
}

fn homogeneous_affine_patch_prime_certificate<R>(
    radical: &super::RadicalIdeal<R>,
) -> Result<Option<PolynomialPrimeCertificate<R>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let ideal = radical.ideal();
    if ideal.dimension()? == 0
        || ideal
            .generators()
            .iter()
            .any(|generator| !matches!(generator.homogeneous_degree(), Ok(Some(_))))
    {
        return Ok(None);
    }
    for variable in 0..ideal.ring().variables().len() {
        let coordinate = ideal.ring().generator(variable)?;
        let coordinate_ideal = ideal.ring().ideal([coordinate.clone()])?;
        let saturation = ideal.saturate(&coordinate_ideal)?;
        if &saturation != ideal {
            continue;
        }
        let affine_equation = ideal.ring().sub(&coordinate, &ideal.ring().one())?;
        let affine_patch = ideal.ring().ideal(
            ideal
                .generators()
                .iter()
                .cloned()
                .chain(core::iter::once(affine_equation)),
        )?;
        let patch_decomposition = match construct_minimal_primes(&affine_patch) {
            Ok(decomposition) => decomposition,
            Err(DecompositionError::Unsupported(_)) => continue,
            Err(error) => return Err(error),
        };
        if patch_decomposition.radical().ideal() != &affine_patch
            || patch_decomposition.prime_certificates().len() != 1
            || patch_decomposition.prime_certificates()[0].ideal() != &affine_patch
        {
            continue;
        }
        let certificate = PolynomialPrimeCertificate {
            ideal: ideal.clone(),
            proof: PolynomialPrimeProof::HomogeneousAffinePatch {
                variable,
                saturation,
                affine_patch,
                affine_patch_proof: Box::new(
                    patch_decomposition.prime_certificates()[0].proof.clone(),
                ),
            },
        };
        verify_polynomial_prime_certificate(&certificate)?;
        return Ok(Some(certificate));
    }
    Ok(None)
}

pub(super) fn coordinate_prime_certificate<R: MinimalPrimeCoefficientField>(
    ring: &PolynomialRing<R>,
    variables: Vec<usize>,
) -> Result<PolynomialPrimeCertificate<R>, DecompositionError> {
    let ideal = ring.ideal(
        variables
            .iter()
            .map(|variable| ring.generator(*variable))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let certificate = PolynomialPrimeCertificate {
        ideal,
        proof: PolynomialPrimeProof::CoordinatePrime { variables },
    };
    verify_polynomial_prime_certificate(&certificate)?;
    Ok(certificate)
}

pub(super) fn verify_polynomial_prime_certificate<R: MinimalPrimeCoefficientField>(
    certificate: &PolynomialPrimeCertificate<R>,
) -> Result<(), DecompositionError> {
    certificate.ideal.verify()?;
    match &certificate.proof {
        PolynomialPrimeProof::IrreducibleHypersurface { factor } => {
            if factor.ring() != certificate.ideal.ring()
                || factor.is_zero()
                || certificate.ideal != factor.ring().ideal([factor.clone()])?
            {
                return Err(DecompositionError::Invalid(
                    "an irreducible-hypersurface certificate has inconsistent data".to_owned(),
                ));
            }
            let factorization = R::factor_polynomial(factor)?;
            factorization.verify()?;
            if factorization.factors() != [(factor.clone(), 1)] {
                return Err(DecompositionError::Invalid(
                    "a hypersurface prime certificate has a reducible generator".to_owned(),
                ));
            }
        }
        PolynomialPrimeProof::CoordinatePrime { variables } => {
            if variables.windows(2).any(|pair| pair[0] >= pair[1])
                || variables
                    .last()
                    .is_some_and(|variable| *variable >= certificate.ideal.ring().variables().len())
            {
                return Err(DecompositionError::Invalid(
                    "a coordinate-prime certificate has noncanonical variables".to_owned(),
                ));
            }
            let rebuilt = certificate.ideal.ring().ideal(
                variables
                    .iter()
                    .map(|variable| certificate.ideal.ring().generator(*variable))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            if rebuilt != certificate.ideal {
                return Err(DecompositionError::Invalid(
                    "a coordinate-prime certificate does not generate its recorded ideal"
                        .to_owned(),
                ));
            }
        }
        PolynomialPrimeProof::AffineLinear { pivot_variables } => {
            let rebuilt =
                super::radical::affine_linear_pivots(&certificate.ideal).ok_or_else(|| {
                    DecompositionError::Invalid(
                        "an affine-linear prime certificate does not define an affine-linear ideal"
                            .to_owned(),
                    )
                })?;
            if &rebuilt != pivot_variables {
                return Err(DecompositionError::Invalid(
                    "an affine-linear prime certificate has incorrect pivot variables".to_owned(),
                ));
            }
        }
        PolynomialPrimeProof::PolynomialGraph { pivot_variables } => {
            let rebuilt =
                super::radical::polynomial_graph_pivots(&certificate.ideal).ok_or_else(|| {
                    DecompositionError::Invalid(
                        "a polynomial-graph prime certificate does not define a polynomial graph"
                            .to_owned(),
                    )
                })?;
            if &rebuilt != pivot_variables {
                return Err(DecompositionError::Invalid(
                    "a polynomial-graph prime certificate has incorrect pivot variables".to_owned(),
                ));
            }
        }
        PolynomialPrimeProof::SmoothProjectiveCompleteIntersection {
            codimension,
            jacobian_minors,
            singular_locus,
            projective_singular_saturation,
        } => {
            let radical = super::radical::construct_radical(&certificate.ideal)?;
            if radical.ideal() != &certificate.ideal {
                return Err(DecompositionError::Invalid(
                    "a projective complete-intersection prime is not reduced".to_owned(),
                ));
            }
            let rebuilt =
                smooth_projective_complete_intersection_data(&radical)?.ok_or_else(|| {
                    DecompositionError::Invalid(
                        "a projective complete-intersection prime is not projectively smooth"
                            .to_owned(),
                    )
                })?;
            if rebuilt.codimension != *codimension
                || rebuilt.jacobian_minors != *jacobian_minors
                || rebuilt.singular_locus != *singular_locus
                || rebuilt.projective_singular_saturation != *projective_singular_saturation
            {
                return Err(DecompositionError::Invalid(
                    "a projective complete-intersection prime has inconsistent Jacobian evidence"
                        .to_owned(),
                ));
            }
        }
        PolynomialPrimeProof::HomogeneousAffinePatch {
            variable,
            saturation,
            affine_patch,
            affine_patch_proof,
        } => {
            if *variable >= certificate.ideal.ring().variables().len()
                || certificate
                    .ideal
                    .generators()
                    .iter()
                    .any(|generator| !matches!(generator.homogeneous_degree(), Ok(Some(_))))
            {
                return Err(DecompositionError::Invalid(
                    "a homogeneous affine-patch certificate has nonhomogeneous or invalid data"
                        .to_owned(),
                ));
            }
            let coordinate = certificate.ideal.ring().generator(*variable)?;
            let coordinate_ideal = certificate.ideal.ring().ideal([coordinate.clone()])?;
            let rebuilt_saturation = certificate.ideal.saturate(&coordinate_ideal)?;
            let affine_equation = certificate
                .ideal
                .ring()
                .sub(&coordinate, &certificate.ideal.ring().one())?;
            let rebuilt_patch = certificate.ideal.ring().ideal(
                certificate
                    .ideal
                    .generators()
                    .iter()
                    .cloned()
                    .chain(core::iter::once(affine_equation)),
            )?;
            let affine_patch_prime = PolynomialPrimeCertificate {
                ideal: affine_patch.clone(),
                proof: affine_patch_proof.as_ref().clone(),
            };
            verify_polynomial_prime_certificate(&affine_patch_prime)?;
            if rebuilt_saturation != *saturation
                || saturation != &certificate.ideal
                || rebuilt_patch != *affine_patch
            {
                return Err(DecompositionError::Invalid(
                    "a homogeneous affine-patch certificate has inconsistent localization evidence"
                        .to_owned(),
                ));
            }
        }
        PolynomialPrimeProof::ZeroDimensionalField { representation } => {
            if representation.source() != &certificate.ideal
                || &irreducible_rur(&certificate.ideal)? != representation.as_ref()
            {
                return Err(DecompositionError::Invalid(
                    "a zero-dimensional field certificate does not prove its recorded ideal"
                        .to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn is_subset(left: &[usize], right: &[usize]) -> bool {
    left.iter().all(|value| right.binary_search(value).is_ok())
}

fn minimalize_transversals(candidates: BTreeSet<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut candidates = candidates.into_iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
    let mut minimal = Vec::<Vec<usize>>::new();
    for candidate in candidates {
        if !minimal
            .iter()
            .any(|existing| is_subset(existing, &candidate))
        {
            minimal.push(candidate);
        }
    }
    minimal.sort();
    minimal
}

fn monomial_minimal_transversals<R: Field>(
    radical: &super::MonomialIdeal<R>,
) -> Result<Vec<Vec<usize>>, DecompositionError> {
    if radical.is_unit() {
        return Ok(Vec::new());
    }
    let mut transversals = vec![Vec::new()];
    for generator in radical.minimal_generators() {
        let support = generator
            .iter()
            .enumerate()
            .filter_map(|(variable, exponent)| (!exponent.is_zero()).then_some(variable))
            .collect::<Vec<_>>();
        if support.is_empty() {
            return Ok(Vec::new());
        }
        let mut candidates = BTreeSet::new();
        for transversal in transversals {
            if support
                .iter()
                .any(|variable| transversal.binary_search(variable).is_ok())
            {
                candidates.insert(transversal);
                if candidates.len() > MAX_MONOMIAL_TRANSVERSAL_CANDIDATES {
                    return Err(DecompositionError::ResourceLimit {
                        resource: "monomial minimal-prime transversal candidates",
                        limit: MAX_MONOMIAL_TRANSVERSAL_CANDIDATES,
                    });
                }
                continue;
            }
            for variable in &support {
                let mut extended = transversal.clone();
                match extended.binary_search(variable) {
                    Ok(_) => {}
                    Err(index) => extended.insert(index, *variable),
                }
                candidates.insert(extended);
                if candidates.len() > MAX_MONOMIAL_TRANSVERSAL_CANDIDATES {
                    return Err(DecompositionError::ResourceLimit {
                        resource: "monomial minimal-prime transversal candidates",
                        limit: MAX_MONOMIAL_TRANSVERSAL_CANDIDATES,
                    });
                }
            }
        }
        transversals = minimalize_transversals(candidates);
        if transversals.len() > MAX_MONOMIAL_TRANSVERSALS {
            return Err(DecompositionError::ResourceLimit {
                resource: "monomial minimal-prime transversal count",
                limit: MAX_MONOMIAL_TRANSVERSALS,
            });
        }
    }
    Ok(transversals)
}

pub(super) fn sorted_prime_certificates<R>(
    certificates: Vec<PolynomialPrimeCertificate<R>>,
) -> Result<Vec<PolynomialPrimeCertificate<R>>, DecompositionError>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let mut keyed = certificates
        .into_iter()
        .map(|certificate| {
            certificate
                .ideal()
                .canonical_bytes()
                .map(|key| (key, certificate))
                .map_err(|error| DecompositionError::Invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    if keyed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(DecompositionError::Invalid(
            "a minimal-prime component was produced more than once".to_owned(),
        ));
    }
    Ok(keyed
        .into_iter()
        .map(|(_, certificate)| certificate)
        .collect())
}

fn sorted_components<R>(
    components: Vec<Ideal<PolynomialRing<R>>>,
) -> Result<Vec<Ideal<PolynomialRing<R>>>, DecompositionError>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let mut keyed = components
        .into_iter()
        .map(|component| {
            component
                .canonical_bytes()
                .map(|key| (key, component))
                .map_err(|error| DecompositionError::Invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    if keyed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(DecompositionError::Invalid(
            "a minimal-prime component was produced more than once".to_owned(),
        ));
    }
    Ok(keyed.into_iter().map(|(_, component)| component).collect())
}

fn ideal_is_subset<R: Field>(
    left: &Ideal<PolynomialRing<R>>,
    right: &Ideal<PolynomialRing<R>>,
) -> Result<bool, DecompositionError> {
    if left.ring() != right.ring() {
        return Err(DecompositionError::Invalid(
            "minimal-prime inclusion compared different polynomial contexts".to_owned(),
        ));
    }
    for generator in left.generators() {
        if !right.contains(generator)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn minimalize_prime_certificates<R>(
    certificates: Vec<PolynomialPrimeCertificate<R>>,
) -> Result<Vec<PolynomialPrimeCertificate<R>>, DecompositionError>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    if certificates.len() > MAX_ZERO_DIMENSIONAL_COMPONENTS {
        return Err(DecompositionError::ResourceLimit {
            resource: "recursive minimal-prime candidates",
            limit: MAX_ZERO_DIMENSIONAL_COMPONENTS,
        });
    }
    let mut unique = Vec::<PolynomialPrimeCertificate<R>>::new();
    for certificate in certificates {
        if !unique
            .iter()
            .any(|existing| existing.ideal() == certificate.ideal())
        {
            unique.push(certificate);
        }
    }
    let mut minimal = Vec::new();
    for (index, certificate) in unique.iter().enumerate() {
        let contains_smaller = unique
            .iter()
            .enumerate()
            .filter(|(other_index, _)| *other_index != index)
            .try_fold(false, |found, (_, other)| {
                if found {
                    Ok(true)
                } else {
                    ideal_is_subset(other.ideal(), certificate.ideal())
                }
            })?;
        if !contains_smaller {
            minimal.push(certificate.clone());
        }
    }
    sorted_prime_certificates(minimal)
}

pub(super) fn construct_minimal_primes<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<MinimalPrimeDecomposition<R>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let radical = super::radical::construct_radical(source)?;
    let certificates = if let Some(factors) = radical.principal_factors() {
        factors
            .iter()
            .map(|(factor, _)| hypersurface_prime_certificate(factor))
            .collect::<Result<Vec<_>, _>>()?
    } else if let Some((_, monomial_radical)) = radical.monomial_proof() {
        monomial_minimal_transversals(monomial_radical)?
            .into_iter()
            .map(|variables| coordinate_prime_certificate(source.ring(), variables))
            .collect::<Result<Vec<_>, _>>()?
    } else if radical.affine_linear_pivots().is_some() {
        vec![affine_linear_prime_certificate(radical.ideal())?]
    } else if radical.polynomial_graph_pivots().is_some() {
        vec![polynomial_graph_prime_certificate(radical.ideal())?]
    } else if let Some(factors) = radical.common_factors() {
        let residual_source = radical.common_factor_residual_source().ok_or_else(|| {
            DecompositionError::Invalid(
                "common-factor radical proof omitted its residual certificate".into(),
            )
        })?;
        let mut certificates = factors
            .iter()
            .map(|(factor, _)| hypersurface_prime_certificate(factor))
            .collect::<Result<Vec<_>, _>>()?;
        certificates.extend(
            construct_minimal_primes(residual_source)?
                .prime_certificates()
                .iter()
                .cloned(),
        );
        minimalize_prime_certificates(certificates)?
    } else if let Some(branch_sources) = radical.factor_split_branch_sources() {
        let mut certificates = Vec::new();
        for branch_source in branch_sources {
            certificates.extend(
                construct_minimal_primes(branch_source)?
                    .prime_certificates()
                    .iter()
                    .cloned(),
            );
        }
        minimalize_prime_certificates(certificates)?
    } else if let (Some(closed_source), Some(open_source)) = (
        radical.saturation_split_closed_source(),
        radical.saturation_split_open_source(),
    ) {
        let mut certificates = construct_minimal_primes(closed_source)?
            .prime_certificates()
            .to_vec();
        certificates.extend(
            construct_minimal_primes(open_source)?
                .prime_certificates()
                .iter()
                .cloned(),
        );
        minimalize_prime_certificates(certificates)?
    } else if radical.complete_intersection_codimension().is_some() {
        match smooth_projective_complete_intersection_prime_certificate(&radical)? {
            Some(certificate) => vec![certificate],
            None => match homogeneous_affine_patch_prime_certificate(&radical)? {
                Some(certificate) => vec![certificate],
                None => {
                    return Err(DecompositionError::Unsupported(
                        "minimal primes of a reduced positive-dimensional complete intersection require a certified component split or primality proof".to_owned(),
                    ));
                }
            },
        }
    } else {
        sorted_components(decompose_radical(
            radical.ideal(),
            MAX_ZERO_DIMENSIONAL_COMPONENTS,
        )?)?
        .iter()
        .map(zero_dimensional_prime_certificate)
        .collect::<Result<Vec<_>, _>>()?
    };
    let prime_certificates = sorted_prime_certificates(certificates)?;
    let components = prime_certificates
        .iter()
        .map(|certificate| certificate.ideal().clone())
        .collect();
    Ok(MinimalPrimeDecomposition {
        source: source.clone(),
        radical,
        components,
        prime_certificates,
    })
}

pub(super) fn verify_minimal_primes<R>(
    value: &MinimalPrimeDecomposition<R>,
) -> Result<(), DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    value.source.verify()?;
    super::radical::verify_radical(&value.radical)?;
    if value.radical.source() != &value.source
        || value.components.len() != value.prime_certificates.len()
    {
        return Err(DecompositionError::Invalid(
            "minimal-prime evidence has inconsistent source or component counts".to_owned(),
        ));
    }
    if sorted_components(value.components.clone())? != value.components {
        return Err(DecompositionError::Invalid(
            "minimal-prime components are not in canonical order".to_owned(),
        ));
    }
    for (component, certificate) in value.components.iter().zip(&value.prime_certificates) {
        if component.ring() != value.source.ring() || certificate.ideal() != component {
            return Err(DecompositionError::Invalid(
                "a prime certificate does not certify its recorded component".to_owned(),
            ));
        }
        verify_polynomial_prime_certificate(certificate)?;
    }
    let reconstructed = if let Some(first) = value.components.first() {
        let mut intersection = first.clone();
        for component in &value.components[1..] {
            intersection = intersection.intersection(component)?;
        }
        intersection
    } else {
        value.source.ring().ideal([value.source.ring().one()])?
    };
    if &reconstructed != value.radical.ideal() {
        return Err(DecompositionError::Invalid(
            "minimal-prime intersection does not reconstruct the radical".to_owned(),
        ));
    }
    if &construct_minimal_primes(&value.source)? != value {
        return Err(DecompositionError::Invalid(
            "recomputation changed the minimal-prime decomposition".to_owned(),
        ));
    }
    Ok(())
}

impl Ideal<PolynomialRing<RationalField>> {
    /// Prove all minimal primes of an admitted rational ideal.
    pub fn minimal_primes(
        &self,
    ) -> Result<MinimalPrimeDecomposition<RationalField>, DecompositionError> {
        construct_minimal_primes(self)
    }
}

impl Ideal<PolynomialRing<FiniteField>> {
    /// Prove all minimal primes of an admitted finite-field ideal.
    pub fn minimal_primes(
        &self,
    ) -> Result<MinimalPrimeDecomposition<FiniteField>, DecompositionError> {
        construct_minimal_primes(self)
    }
}

impl MinimalPrimeDecomposition<RationalField> {
    /// Recheck radical equality, component intersection, and every prime certificate.
    pub fn verify(&self) -> Result<(), DecompositionError> {
        verify_minimal_primes(self)
    }

    /// Deserialize and exactly reverify rational minimal-prime evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl MinimalPrimeDecomposition<FiniteField> {
    /// Recheck radical equality, component intersection, and every prime certificate.
    pub fn verify(&self) -> Result<(), DecompositionError> {
        verify_minimal_primes(self)
    }

    /// Deserialize and exactly reverify finite-field minimal-prime evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl PolynomialPrimeCertificate<RationalField> {
    /// Independently recheck the rational polynomial primality evidence.
    pub fn verify(&self) -> Result<(), DecompositionError> {
        verify_polynomial_prime_certificate(self)
    }

    /// Deserialize and exactly reverify rational polynomial primality evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl PolynomialPrimeCertificate<FiniteField> {
    /// Independently recheck the finite-field polynomial primality evidence.
    pub fn verify(&self) -> Result<(), DecompositionError> {
        verify_polynomial_prime_certificate(self)
    }

    /// Deserialize and exactly reverify finite-field polynomial primality evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> PolynomialPrimeCertificate<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the prime ideal and complete typed proof evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for PolynomialPrimeCertificate<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PolynomialPrimeCertificate", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ideal", &self.ideal)?;
        state.serialize_field("proof", &self.proof)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct PolynomialPrimeCertificateWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ideal: Ideal<PolynomialRing<R>>,
    proof: PolynomialPrimeProof<R>,
}

fn deserialize_polynomial_prime_certificate<'de, R, D>(
    deserializer: D,
) -> Result<PolynomialPrimeCertificate<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    D: Deserializer<'de>,
{
    let wire = PolynomialPrimeCertificateWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/PolynomialPrimeCertificate@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/PolynomialPrimeCertificate@1", wire.type_id
        )));
    }
    let result = PolynomialPrimeCertificate {
        ideal: wire.ideal,
        proof: wire.proof,
    };
    verify_polynomial_prime_certificate(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for PolynomialPrimeCertificate<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_polynomial_prime_certificate(deserializer)
    }
}

impl<'de> Deserialize<'de> for PolynomialPrimeCertificate<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_polynomial_prime_certificate(deserializer)
    }
}

impl<R> MinimalPrimeDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the complete radical, component, and prime evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for MinimalPrimeDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("MinimalPrimeDecomposition", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("radical", &self.radical)?;
        state.serialize_field("components", &self.components)?;
        state.serialize_field("prime_certificates", &self.prime_certificates)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, RadicalIdeal<R>: Deserialize<'de>, PolynomialPrimeCertificate<R>: Deserialize<'de>"
    )
)]
struct MinimalPrimeDecompositionWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    radical: RadicalIdeal<R>,
    components: Vec<Ideal<PolynomialRing<R>>>,
    prime_certificates: Vec<PolynomialPrimeCertificate<R>>,
}

fn deserialize_minimal_primes<'de, R, D>(
    deserializer: D,
) -> Result<MinimalPrimeDecomposition<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    RadicalIdeal<R>: Deserialize<'de>,
    PolynomialPrimeCertificate<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = MinimalPrimeDecompositionWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/MinimalPrimeDecomposition@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/MinimalPrimeDecomposition@1", wire.type_id
        )));
    }
    let result = MinimalPrimeDecomposition {
        source: wire.source,
        radical: wire.radical,
        components: wire.components,
        prime_certificates: wire.prime_certificates,
    };
    verify_minimal_primes(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for MinimalPrimeDecomposition<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_minimal_primes(deserializer)
    }
}

impl<'de> Deserialize<'de> for MinimalPrimeDecomposition<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_minimal_primes(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::numthy::FiniteField;
    use crate::{SemanticType, Q};

    #[test]
    fn rational_fat_points_split_into_verified_maximal_ideals() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let x = &variables[0];
        let y = &variables[1];
        let x_minus_one = (x - &ring.constant(1.into()).unwrap()).unwrap();
        let x_plus_two = (x + &ring.constant(2.into()).unwrap()).unwrap();
        let source = ring
            .ideal([
                (&x_minus_one.pow(2).unwrap() * &x_plus_two.pow(3).unwrap()).unwrap(),
                y.pow(2).unwrap(),
            ])
            .unwrap();

        let decomposition = source.minimal_primes().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        assert_eq!(decomposition.prime_certificates().len(), 2);
        assert!(decomposition
            .prime_certificates()
            .iter()
            .all(|certificate| certificate
                .rational_univariate_representation()
                .is_some_and(RationalUnivariateRepresentation::is_reduced)));
        assert_eq!(
            MinimalPrimeDecomposition::<RationalField>::from_json(
                &decomposition.to_json().unwrap()
            )
            .unwrap(),
            decomposition
        );
        assert_eq!(
            MinimalPrimeDecomposition::<RationalField>::TYPE_ID,
            "noether/commalg/MinimalPrimeDecomposition@1"
        );
    }

    #[test]
    fn nonmonogenic_f2_product_uses_recursive_coordinate_splits() {
        let field = FiniteField::prime(2_u64).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&variables[0].pow(2).unwrap() + &variables[0]).unwrap(),
                (&variables[1].pow(2).unwrap() + &variables[1]).unwrap(),
            ])
            .unwrap();

        assert!(source.rational_univariate_representation().is_err());
        let decomposition = source.minimal_primes().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 4);
        assert!(decomposition
            .prime_certificates()
            .iter()
            .all(|certificate| certificate
                .rational_univariate_representation()
                .is_some_and(|representation| representation.quotient_dimension() == 1)));
    }

    #[test]
    fn unit_ideal_has_no_minimal_primes() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let source = ring.ideal([ring.one()]).unwrap();

        let decomposition = source.minimal_primes().unwrap();
        decomposition.verify().unwrap();
        assert!(decomposition.components().is_empty());
        assert!(decomposition.prime_certificates().is_empty());
    }

    #[test]
    fn principal_hypersurface_minimal_primes_use_irreducible_factors() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let first = (&variables[0] + &variables[1]).unwrap();
        let second = (&variables[0] - &variables[2]).unwrap();
        let source = ring
            .ideal([(&first.pow(3).unwrap() * &second.pow(2).unwrap()).unwrap()])
            .unwrap();

        let decomposition = source.minimal_primes().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        for certificate in decomposition.prime_certificates() {
            certificate.verify().unwrap();
            assert_eq!(certificate.proof_kind(), "irreducible_hypersurface");
            assert!(certificate.irreducible_factor().is_some());
            assert_eq!(certificate.ideal().dimension().unwrap(), 2);
            assert_eq!(
                PolynomialPrimeCertificate::<RationalField>::from_json(
                    &certificate.to_json().unwrap()
                )
                .unwrap(),
                *certificate
            );
        }
    }

    #[test]
    fn monomial_minimal_primes_are_minimal_coordinate_transversals() {
        let field = FiniteField::prime(5_u64).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&variables[0].pow(3).unwrap() * &variables[1]).unwrap(),
                (&variables[0] * &variables[2].pow(4).unwrap()).unwrap(),
            ])
            .unwrap();

        let decomposition = source.minimal_primes().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        let mut coordinate_sets = decomposition
            .prime_certificates()
            .iter()
            .map(|certificate| {
                certificate.verify().unwrap();
                assert_eq!(certificate.proof_kind(), "coordinate_prime");
                certificate.coordinate_variables().unwrap().to_vec()
            })
            .collect::<Vec<_>>();
        coordinate_sets.sort();
        assert_eq!(coordinate_sets, vec![vec![0], vec![1, 2]]);
        assert_eq!(
            decomposition.components()[0]
                .dimension()
                .unwrap()
                .max(decomposition.components()[1].dimension().unwrap()),
            2
        );
    }

    #[test]
    fn zero_ideal_has_the_empty_coordinate_prime() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let source = ring.ideal(Vec::<Polynomial<RationalField>>::new()).unwrap();

        let decomposition = source.minimal_primes().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components(), &[source]);
        assert_eq!(
            decomposition.prime_certificates()[0].coordinate_variables(),
            Some([].as_slice())
        );
    }

    #[test]
    fn common_factor_split_removes_point_primes_already_on_a_hypersurface() {
        let field = FiniteField::prime(2_u64).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let x = &variables[0];
        let y = &variables[1];
        let common = (&(x + y).unwrap() + &ring.one()).unwrap();
        let x_frobenius = (&x.pow(2).unwrap() + x).unwrap();
        let y_frobenius = (&y.pow(2).unwrap() + y).unwrap();
        let source = ring
            .ideal([
                (&common * &x_frobenius).unwrap(),
                (&common * &y_frobenius).unwrap(),
            ])
            .unwrap();

        let decomposition = source.minimal_primes().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 3);
        let mut dimensions = decomposition
            .components()
            .iter()
            .map(|component| component.dimension().unwrap())
            .collect::<Vec<_>>();
        dimensions.sort();
        assert_eq!(dimensions, vec![0, 0, 1]);
        let mut proof_kinds = decomposition
            .prime_certificates()
            .iter()
            .map(PolynomialPrimeCertificate::proof_kind)
            .collect::<Vec<_>>();
        proof_kinds.sort();
        assert_eq!(
            proof_kinds,
            vec![
                "irreducible_hypersurface",
                "zero_dimensional_field",
                "zero_dimensional_field",
            ]
        );

        let point_zero = ring.ideal([x.clone(), y.clone()]).unwrap();
        let point_one = ring
            .ideal([(x + &ring.one()).unwrap(), (y + &ring.one()).unwrap()])
            .unwrap();
        let expected = ring
            .ideal([common])
            .unwrap()
            .intersection(&point_zero)
            .unwrap()
            .intersection(&point_one)
            .unwrap();
        assert_eq!(decomposition.radical().ideal(), &expected);
    }

    #[test]
    fn generator_factor_split_produces_affine_linear_minimal_primes() {
        let field = FiniteField::prime(65_521_u64).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let left = (&variables[0] + &ring.one()).unwrap();
        let right = (&variables[1] + &ring.one()).unwrap();
        let product = (&left.pow(2).unwrap() * &right.pow(3).unwrap()).unwrap();
        let source = ring.ideal([variables[2].clone(), product]).unwrap();

        let decomposition = source.minimal_primes().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        assert!(decomposition
            .prime_certificates()
            .iter()
            .all(|certificate| certificate.proof_kind() == "affine_linear"));
        assert!(decomposition
            .prime_certificates()
            .iter()
            .all(|certificate| certificate.affine_linear_pivots().unwrap().len() == 2));
        assert_eq!(
            decomposition.radical().ideal(),
            &ring
                .ideal([variables[2].clone(), (&left * &right).unwrap(),])
                .unwrap()
        );
        assert_eq!(
            MinimalPrimeDecomposition::<FiniteField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );

        let mut payload = serde_json::to_value(&decomposition.prime_certificates()[0]).unwrap();
        payload["proof"]["pivot_variables"] = serde_json::json!([]);
        assert!(
            serde_json::from_value::<PolynomialPrimeCertificate<FiniteField>>(payload).is_err()
        );
    }

    #[test]
    fn polynomial_graph_has_one_nonlinear_prime_certificate() {
        let ring = PolynomialRing::new(
            FiniteField::prime(65_521_u64).unwrap(),
            ["y", "z", "x"],
            MonomialOrder::Lexicographic,
        )
        .unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&variables[0] - &variables[2].pow(2).unwrap()).unwrap(),
                (&variables[1] - &variables[2].pow(3).unwrap()).unwrap(),
            ])
            .unwrap();

        let decomposition = source.minimal_primes().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components(), &[source]);
        assert_eq!(decomposition.prime_certificates().len(), 1);
        let certificate = &decomposition.prime_certificates()[0];
        assert_eq!(certificate.proof_kind(), "polynomial_graph");
        assert_eq!(certificate.polynomial_graph_pivots(), Some(&[0, 1][..]));
        let mut payload = serde_json::to_value(certificate).unwrap();
        payload["proof"]["pivot_variables"] = serde_json::json!([0]);
        assert!(
            serde_json::from_value::<PolynomialPrimeCertificate<FiniteField>>(payload).is_err()
        );
    }

    #[test]
    fn smooth_projective_complete_intersection_has_one_prime_certificate() {
        let ring = PolynomialRing::new(
            FiniteField::prime(65_521_u64).unwrap(),
            ["x", "y", "z", "w"],
            MonomialOrder::default(),
        )
        .unwrap();
        let variables = ring.generators().unwrap();
        let squares = variables
            .iter()
            .map(|variable| variable.pow(2).unwrap())
            .collect::<Vec<_>>();
        let first = squares
            .iter()
            .try_fold(ring.zero(), |sum, square| &sum + square)
            .unwrap();
        let second = squares
            .iter()
            .enumerate()
            .try_fold(ring.zero(), |sum, (index, square)| {
                let coefficient = ring
                    .coefficient_ring()
                    .element(u64::try_from(index + 1).unwrap());
                &sum + &square.scalar_mul(&coefficient).unwrap()
            })
            .unwrap();
        let source = ring.ideal([first, second]).unwrap();

        let decomposition = source.minimal_primes().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components(), &[source]);
        let certificate = &decomposition.prime_certificates()[0];
        assert_eq!(
            certificate.proof_kind(),
            "smooth_projective_complete_intersection"
        );
        assert_eq!(certificate.complete_intersection_codimension(), Some(2));
        assert_eq!(certificate.jacobian_minors().unwrap().len(), 6);
        let saturation = certificate.projective_singular_saturation().unwrap();
        assert!(saturation.generators().len() == 1 && saturation.generators()[0].is_one());
        assert_eq!(
            MinimalPrimeDecomposition::<FiniteField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );

        let mut payload = serde_json::to_value(certificate).unwrap();
        payload["proof"]["codimension"] = serde_json::json!(1);
        assert!(
            serde_json::from_value::<PolynomialPrimeCertificate<FiniteField>>(payload).is_err()
        );
    }

    #[test]
    fn homogeneous_curve_uses_a_prime_affine_patch() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                variables[1].clone(),
                (&variables[0].pow(2).unwrap() + &variables[2].pow(2).unwrap()).unwrap(),
            ])
            .unwrap();

        let decomposition = source.minimal_primes().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components(), std::slice::from_ref(&source));
        let certificate = &decomposition.prime_certificates()[0];
        assert_eq!(certificate.proof_kind(), "homogeneous_affine_patch");
        assert_eq!(certificate.homogeneous_patch_variable(), Some(0));
        assert_eq!(certificate.homogeneous_patch_saturation(), Some(&source));
        let patch_prime = certificate.homogeneous_patch_prime().unwrap();
        assert_eq!(patch_prime.proof_kind(), "zero_dimensional_field");
        assert_eq!(
            patch_prime.ideal(),
            &ring
                .ideal([
                    source.generators()[0].clone(),
                    source.generators()[1].clone(),
                    (&variables[0] - &ring.one()).unwrap(),
                ])
                .unwrap()
        );
        patch_prime.verify().unwrap();
        assert_eq!(
            PolynomialPrimeCertificate::<RationalField>::from_json(&certificate.to_json().unwrap())
                .unwrap(),
            *certificate
        );

        let mut payload = serde_json::to_value(certificate).unwrap();
        payload["proof"]["variable"] = serde_json::json!(3);
        assert!(
            serde_json::from_value::<PolynomialPrimeCertificate<RationalField>>(payload).is_err()
        );
    }
}
