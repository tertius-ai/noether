// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Verified radicals of exact polynomial ideals over perfect fields.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::linalg::Matrix;
use crate::numthy::FiniteField;
use crate::{Field, NoetherError, RationalField, Ring, SemanticType, TypeDescription};

use super::{
    DecompositionError, FiniteQuotientAlgebra, Ideal, MonomialIdeal, MonomialOrder, Polynomial,
    PolynomialFactorization, PolynomialRing,
};

const MAX_RADICAL_SPLIT_DEPTH: usize = 256;
const MAX_RADICAL_SPLIT_BRANCHES: usize = 4_096;
const MAX_SATURATION_SPLIT_CANDIDATES: usize = 512;

static RADICAL_IDEAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RadicalIdeal@1",
    rust_path: "noether::commalg::RadicalIdeal",
    python_path: Some("noether.commalg.RadicalIdeal"),
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
enum RadicalProof<R: Field> {
    PrincipalFactorization {
        factors: Vec<(Polynomial<R>, u64)>,
    },
    MonomialSupport {
        source: MonomialIdeal<R>,
        radical: MonomialIdeal<R>,
    },
    ZeroDimensionalCoordinates {
        characteristic_polynomials: Vec<Polynomial<R>>,
        coordinate_certificates: Vec<Polynomial<R>>,
    },
    AffineLinear {
        pivot_variables: Vec<usize>,
    },
    PolynomialGraph {
        pivot_variables: Vec<usize>,
    },
    CompleteIntersectionJacobian {
        codimension: usize,
        jacobian_minors: Vec<Polynomial<R>>,
        singular_locus: Ideal<PolynomialRing<R>>,
        singular_dimension: i64,
    },
    CommonFactorSplit {
        common_factors: Vec<(Polynomial<R>, u64)>,
        residual_source: Ideal<PolynomialRing<R>>,
        residual_radical: Ideal<PolynomialRing<R>>,
        residual_proof: Box<RadicalProof<R>>,
    },
    GeneratorFactorSplit {
        splitting_generator: Polynomial<R>,
        factors: Vec<(Polynomial<R>, u64)>,
        branch_sources: Vec<Ideal<PolynomialRing<R>>>,
        branch_radicals: Vec<Ideal<PolynomialRing<R>>>,
        branch_proofs: Vec<RadicalProof<R>>,
    },
    SaturationSplit {
        witness: Polynomial<R>,
        closed_source: Ideal<PolynomialRing<R>>,
        closed_radical: Ideal<PolynomialRing<R>>,
        closed_proof: Box<RadicalProof<R>>,
        open_source: Ideal<PolynomialRing<R>>,
        open_radical: Ideal<PolynomialRing<R>>,
        open_proof: Box<RadicalProof<R>>,
    },
}

/// A proved radical of a polynomial ideal.
///
/// The proof is chosen from exact principal or generator factorization,
/// coefficient-generic monomial support, polynomial-graph elimination,
/// complete-intersection Jacobian evidence, recursive factor splitting, or
/// Seidenberg coordinate polynomials. The first native contexts are `Q` and
/// finite fields, both of which are perfect.
#[derive(Clone, PartialEq, Eq)]
pub struct RadicalIdeal<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    radical: Ideal<PolynomialRing<R>>,
    proof: RadicalProof<R>,
}

impl<R: Field> RadicalIdeal<R> {
    /// Source ideal before nilpotents are removed.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Canonical radical ideal in the same polynomial-ring context.
    pub fn ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.radical
    }

    /// Characteristic polynomial of multiplication by each source variable.
    ///
    /// This is empty when a principal or monomial proof applies.
    pub fn characteristic_polynomials(&self) -> &[Polynomial<R>] {
        match &self.proof {
            RadicalProof::ZeroDimensionalCoordinates {
                characteristic_polynomials,
                ..
            } => characteristic_polynomials,
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => &[],
        }
    }

    /// Square-free coordinate polynomials adjoined to the source ideal.
    ///
    /// This is empty when a principal or monomial proof applies.
    pub fn coordinate_certificates(&self) -> &[Polynomial<R>] {
        match &self.proof {
            RadicalProof::ZeroDimensionalCoordinates {
                coordinate_certificates,
                ..
            } => coordinate_certificates,
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => &[],
        }
    }

    /// Canonical name of the exact proof route.
    pub const fn proof_kind(&self) -> &'static str {
        match &self.proof {
            RadicalProof::PrincipalFactorization { .. } => "principal_factorization",
            RadicalProof::MonomialSupport { .. } => "monomial_support",
            RadicalProof::ZeroDimensionalCoordinates { .. } => "zero_dimensional_coordinates",
            RadicalProof::AffineLinear { .. } => "affine_linear",
            RadicalProof::PolynomialGraph { .. } => "polynomial_graph",
            RadicalProof::CompleteIntersectionJacobian { .. } => "complete_intersection_jacobian",
            RadicalProof::CommonFactorSplit { .. } => "common_factor_split",
            RadicalProof::GeneratorFactorSplit { .. } => "generator_factor_split",
            RadicalProof::SaturationSplit { .. } => "saturation_split",
        }
    }

    /// Canonical irreducible factors and source multiplicities for a principal ideal.
    pub fn principal_factors(&self) -> Option<&[(Polynomial<R>, u64)]> {
        match &self.proof {
            RadicalProof::PrincipalFactorization { factors } => Some(factors),
            RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Canonical monomial source and radical when the monomial route applies.
    pub fn monomial_proof(&self) -> Option<(&MonomialIdeal<R>, &MonomialIdeal<R>)> {
        match &self.proof {
            RadicalProof::MonomialSupport { source, radical } => Some((source, radical)),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Pivot variables eliminated by an affine-linear prime ideal.
    pub fn affine_linear_pivots(&self) -> Option<&[usize]> {
        match &self.proof {
            RadicalProof::AffineLinear { pivot_variables } => Some(pivot_variables),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Pivot variables eliminated by a polynomial-graph prime ideal.
    pub fn polynomial_graph_pivots(&self) -> Option<&[usize]> {
        match &self.proof {
            RadicalProof::PolynomialGraph { pivot_variables } => Some(pivot_variables),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Codimension certified by a complete-intersection Jacobian proof.
    pub fn complete_intersection_codimension(&self) -> Option<usize> {
        match &self.proof {
            RadicalProof::CompleteIntersectionJacobian { codimension, .. } => Some(*codimension),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Expected-codimension Jacobian minors retained by the radical proof.
    pub fn jacobian_minors(&self) -> Option<&[Polynomial<R>]> {
        match &self.proof {
            RadicalProof::CompleteIntersectionJacobian {
                jacobian_minors, ..
            } => Some(jacobian_minors),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Singular-locus ideal retained by a complete-intersection proof.
    pub fn singular_locus_ideal(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            RadicalProof::CompleteIntersectionJacobian { singular_locus, .. } => {
                Some(singular_locus)
            }
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Dimension of the retained singular-locus ideal.
    pub fn singular_locus_dimension(&self) -> Option<i64> {
        match &self.proof {
            RadicalProof::CompleteIntersectionJacobian {
                singular_dimension, ..
            } => Some(*singular_dimension),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Common irreducible factors when the common-factor route applies.
    ///
    /// This is present only when the source was proved to have the exact form
    /// `hJ`; the factors include their multiplicities in `h`.
    pub fn common_factors(&self) -> Option<&[(Polynomial<R>, u64)]> {
        match &self.proof {
            RadicalProof::CommonFactorSplit { common_factors, .. } => Some(common_factors),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Residual source for a common-factor split.
    pub fn common_factor_residual_source(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            RadicalProof::CommonFactorSplit {
                residual_source, ..
            } => Some(residual_source),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Recursively certified residual radical for a common-factor split.
    pub fn common_factor_residual_radical(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            RadicalProof::CommonFactorSplit {
                residual_radical, ..
            } => Some(residual_radical),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::GeneratorFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Canonical source generator used by a recursive factor split.
    pub fn factor_split_generator(&self) -> Option<&Polynomial<R>> {
        match &self.proof {
            RadicalProof::GeneratorFactorSplit {
                splitting_generator,
                ..
            } => Some(splitting_generator),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Irreducible factors defining the branches of a recursive factor split.
    pub fn factor_split_factors(&self) -> Option<&[(Polynomial<R>, u64)]> {
        match &self.proof {
            RadicalProof::GeneratorFactorSplit { factors, .. } => Some(factors),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Strictly larger branch ideals used by a recursive factor split.
    pub fn factor_split_branch_sources(&self) -> Option<&[Ideal<PolynomialRing<R>>]> {
        match &self.proof {
            RadicalProof::GeneratorFactorSplit { branch_sources, .. } => Some(branch_sources),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Recursively certified branch radicals used by a factor split.
    pub fn factor_split_branch_radicals(&self) -> Option<&[Ideal<PolynomialRing<R>>]> {
        match &self.proof {
            RadicalProof::GeneratorFactorSplit {
                branch_radicals, ..
            } => Some(branch_radicals),
            RadicalProof::PrincipalFactorization { .. }
            | RadicalProof::MonomialSupport { .. }
            | RadicalProof::ZeroDimensionalCoordinates { .. }
            | RadicalProof::AffineLinear { .. }
            | RadicalProof::PolynomialGraph { .. }
            | RadicalProof::CompleteIntersectionJacobian { .. }
            | RadicalProof::CommonFactorSplit { .. }
            | RadicalProof::SaturationSplit { .. } => None,
        }
    }

    /// Polynomial whose vanishing and principal open set define a saturation split.
    pub fn saturation_split_witness(&self) -> Option<&Polynomial<R>> {
        match &self.proof {
            RadicalProof::SaturationSplit { witness, .. } => Some(witness),
            _ => None,
        }
    }

    /// Closed branch `I + (g)` of a saturation split.
    pub fn saturation_split_closed_source(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            RadicalProof::SaturationSplit { closed_source, .. } => Some(closed_source),
            _ => None,
        }
    }

    /// Recursively certified radical of the closed branch `I + (g)`.
    pub fn saturation_split_closed_radical(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            RadicalProof::SaturationSplit { closed_radical, .. } => Some(closed_radical),
            _ => None,
        }
    }

    /// Principal-open branch `I : g^infinity` of a saturation split.
    pub fn saturation_split_open_source(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            RadicalProof::SaturationSplit { open_source, .. } => Some(open_source),
            _ => None,
        }
    }

    /// Recursively certified radical of the principal-open branch.
    pub fn saturation_split_open_radical(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            RadicalProof::SaturationSplit { open_radical, .. } => Some(open_radical),
            _ => None,
        }
    }

    /// Exact theorem used to turn the retained certificates into a radical claim.
    pub const fn theorem(&self) -> &'static str {
        match &self.proof {
            RadicalProof::PrincipalFactorization { .. } => {
                "a principal ideal in a polynomial UFD is radical exactly when its generator is square-free"
            }
            RadicalProof::MonomialSupport { .. } => {
                "the radical of a monomial ideal is generated by the square-free supports of its monomials"
            }
            RadicalProof::ZeroDimensionalCoordinates { .. } => {
                "Seidenberg coordinate-polynomial lemma over a perfect field"
            }
            RadicalProof::AffineLinear { .. } => {
                "the quotient by a proper affine-linear ideal is a polynomial ring over the coefficient field"
            }
            RadicalProof::PolynomialGraph { .. } => {
                "the quotient by a polynomial graph ideal is a polynomial ring over the free variables"
            }
            RadicalProof::CompleteIntersectionJacobian { .. } => {
                "an unmixed complete intersection that is generically smooth over a perfect field is reduced"
            }
            RadicalProof::CommonFactorSplit { .. } => {
                "the radical of hJ is the intersection of the radicals of (h) and J"
            }
            RadicalProof::GeneratorFactorSplit { .. } => {
                "if a product of powers lies in I, then sqrt(I) is the intersection of sqrt(I + (f)) over its irreducible factors f"
            }
            RadicalProof::SaturationSplit { .. } => {
                "sqrt(I) is the intersection of sqrt(I + (g)) and sqrt(I : g^infinity)"
            }
        }
    }
}

impl<R: Field> fmt::Debug for RadicalIdeal<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RadicalIdeal")
            .field("proof_kind", &self.proof_kind())
            .field("generators", &self.radical.generators().len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for RadicalIdeal<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RadicalIdeal(proof={}, generators={})",
            self.proof_kind(),
            self.radical.generators().len()
        )
    }
}

impl<R> SemanticType for RadicalIdeal<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/RadicalIdeal@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &RADICAL_IDEAL_DESCRIPTION
    }
}

pub(crate) trait RadicalCoefficientField: Field {
    fn factor(
        value: &Polynomial<Self>,
    ) -> Result<PolynomialFactorization<Self>, DecompositionError>;

    fn square_free_part(value: &Polynomial<Self>) -> Result<Polynomial<Self>, DecompositionError> {
        let factorization = Self::factor(value)?;
        let mut result = value.ring().one();
        for (factor, _) in factorization.factors() {
            result = (&result * factor)?;
        }
        Ok(result)
    }
}

impl RadicalCoefficientField for RationalField {
    fn factor(
        value: &Polynomial<Self>,
    ) -> Result<PolynomialFactorization<Self>, DecompositionError> {
        value.factor().map_err(Into::into)
    }
}

impl RadicalCoefficientField for FiniteField {
    fn factor(
        value: &Polynomial<Self>,
    ) -> Result<PolynomialFactorization<Self>, DecompositionError> {
        value.factor().map_err(Into::into)
    }
}

fn multiplication_matrix<R: Field>(
    quotient: &FiniteQuotientAlgebra<R>,
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
    if variable >= arity {
        return Err(DecompositionError::Invalid(
            "a radical certificate names a variable outside its source ring".to_owned(),
        ));
    }
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

fn monomial_generators<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
) -> Option<Vec<Vec<crate::Integer>>> {
    source
        .generators()
        .iter()
        .map(|polynomial| {
            let mut terms = polynomial.terms();
            let (exponent, _) = terms.next()?;
            terms.next().is_none().then(|| exponent.to_vec())
        })
        .collect()
}

/// Pivot variables when a proper ideal is generated by affine-linear equations.
///
/// A canonical reduced basis of such an ideal solves each pivot variable as an
/// affine-linear expression in the remaining variables. Its quotient is
/// therefore a polynomial ring over the coefficient field.
pub(super) fn affine_linear_pivots<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
) -> Option<Vec<usize>> {
    let mut pivots = Vec::with_capacity(source.generators().len());
    for generator in source.generators() {
        for (exponent, _) in generator.terms() {
            let mut support_size = 0_usize;
            for value in exponent {
                if value.is_zero() {
                    continue;
                }
                if !value.is_one() {
                    return None;
                }
                support_size += 1;
                if support_size > 1 {
                    return None;
                }
            }
        }
        let (leading_exponent, _) = generator.terms().next()?;
        let pivot = leading_exponent.iter().position(|value| value.is_one())?;
        pivots.push(pivot);
    }
    pivots.sort_unstable();
    if pivots.windows(2).any(|pair| pair[0] == pair[1]) {
        return None;
    }
    Some(pivots)
}

/// Pivot variables when a reduced basis presents a polynomial graph.
///
/// Each leading monomial must be one distinct pivot variable and every tail
/// must involve only the complementary free variables. Substitution therefore
/// identifies the quotient with the polynomial ring on those free variables.
pub(super) fn polynomial_graph_pivots<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
) -> Option<Vec<usize>> {
    let mut pivots = Vec::with_capacity(source.generators().len());
    for generator in source.generators() {
        let (leading_exponent, _) = generator.terms().next()?;
        let mut pivot = None;
        for (variable, value) in leading_exponent.iter().enumerate() {
            if value.is_zero() {
                continue;
            }
            if !value.is_one() || pivot.is_some() {
                return None;
            }
            pivot = Some(variable);
        }
        pivots.push(pivot?);
    }
    pivots.sort_unstable();
    if pivots.windows(2).any(|pair| pair[0] == pair[1]) {
        return None;
    }
    for generator in source.generators() {
        for (term_index, (exponent, _)) in generator.terms().enumerate() {
            if term_index == 0 {
                continue;
            }
            if pivots.iter().any(|pivot| !exponent[*pivot].is_zero()) {
                return None;
            }
        }
    }
    Some(pivots)
}

fn construct_zero_dimensional_monomial_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    exponents: Vec<Vec<crate::Integer>>,
) -> Result<RadicalIdeal<R>, DecompositionError> {
    let monomial = source.ring().monomial_ideal(exponents)?;
    let quotient_dimension = monomial.finite_standard_monomials()?.len();
    let parameter_ring = PolynomialRing::new(
        source.ring().coefficient_ring().clone(),
        ["t"],
        MonomialOrder::Lexicographic,
    )?;
    let parameter = parameter_ring.generator(0)?;
    let characteristic = if quotient_dimension == 0 {
        parameter_ring.one()
    } else {
        parameter.pow(quotient_dimension as u64)?
    };
    let square_free = if quotient_dimension == 0 {
        parameter_ring.one()
    } else {
        parameter
    };
    let variable_count = source.ring().variables().len();
    let characteristic_polynomials = vec![characteristic; variable_count];
    let coordinate_certificates = (0..variable_count)
        .map(|variable| map_parameter_to_variable(&square_free, source.ring(), variable))
        .collect::<Result<Vec<_>, _>>()?;
    let generators = source
        .generators()
        .iter()
        .cloned()
        .chain(coordinate_certificates.iter().cloned())
        .collect::<Vec<_>>();
    let radical = source.ring().ideal(generators)?;
    Ok(RadicalIdeal {
        source: source.clone(),
        radical,
        proof: RadicalProof::ZeroDimensionalCoordinates {
            characteristic_polynomials,
            coordinate_certificates,
        },
    })
}

fn construct_positive_dimensional_monomial_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    exponents: Vec<Vec<crate::Integer>>,
) -> Result<RadicalIdeal<R>, DecompositionError> {
    let monomial_source = source.ring().monomial_ideal(exponents)?;
    let monomial_radical = monomial_source.radical()?;
    let radical = monomial_radical.to_ideal()?;
    Ok(RadicalIdeal {
        source: source.clone(),
        radical,
        proof: RadicalProof::MonomialSupport {
            source: monomial_source,
            radical: monomial_radical,
        },
    })
}

fn construct_affine_linear_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    pivot_variables: Vec<usize>,
) -> RadicalIdeal<R> {
    RadicalIdeal {
        source: source.clone(),
        radical: source.clone(),
        proof: RadicalProof::AffineLinear { pivot_variables },
    }
}

fn construct_polynomial_graph_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    pivot_variables: Vec<usize>,
) -> RadicalIdeal<R> {
    RadicalIdeal {
        source: source.clone(),
        radical: source.clone(),
        proof: RadicalProof::PolynomialGraph { pivot_variables },
    }
}

fn map_jacobian_error(error: super::jacobian::JacobianError) -> DecompositionError {
    match error {
        super::jacobian::JacobianError::Polynomial(error) => DecompositionError::Polynomial(error),
        super::jacobian::JacobianError::ContextMismatch(error) => {
            DecompositionError::ContextMismatch(error)
        }
        super::jacobian::JacobianError::ResourceLimit { resource, limit } => {
            DecompositionError::ResourceLimit { resource, limit }
        }
        super::jacobian::JacobianError::Invalid(reason) => {
            DecompositionError::Invalid(reason.to_owned())
        }
    }
}

fn construct_complete_intersection_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    dimension: i64,
) -> Result<Option<RadicalIdeal<R>>, DecompositionError> {
    let ambient_dimension = i64::try_from(source.ring().variables().len()).map_err(|_| {
        DecompositionError::ResourceLimit {
            resource: "ambient polynomial variable count",
            limit: usize::MAX,
        }
    })?;
    let codimension = usize::try_from(ambient_dimension - dimension).map_err(|_| {
        DecompositionError::Invalid(
            "a positive-dimensional ideal has impossible codimension".to_owned(),
        )
    })?;
    if codimension == 0 || source.generators().len() != codimension {
        return Ok(None);
    }
    let jacobian_minors =
        super::jacobian::jacobian_minors(source, codimension).map_err(map_jacobian_error)?;
    let singular_locus = source.ring().ideal(
        source
            .generators()
            .iter()
            .cloned()
            .chain(jacobian_minors.iter().cloned()),
    )?;
    let singular_dimension = singular_locus.dimension()?;
    if singular_dimension >= dimension {
        return Ok(None);
    }
    Ok(Some(RadicalIdeal {
        source: source.clone(),
        radical: source.clone(),
        proof: RadicalProof::CompleteIntersectionJacobian {
            codimension,
            jacobian_minors,
            singular_locus,
            singular_dimension,
        },
    }))
}

fn construct_principal_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<RadicalIdeal<R>, DecompositionError> {
    let generator = source.generators().first().ok_or_else(|| {
        DecompositionError::Invalid("a principal radical proof has no generator".to_owned())
    })?;
    let factorization = R::factor(generator)?;
    factorization.verify()?;
    let mut square_free = source.ring().one();
    for (factor, _) in factorization.factors() {
        square_free = (&square_free * factor)?;
    }
    let radical = source.ring().ideal([square_free])?;
    Ok(RadicalIdeal {
        source: source.clone(),
        radical,
        proof: RadicalProof::PrincipalFactorization {
            factors: factorization.factors().to_vec(),
        },
    })
}

fn generator_factorizations<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Vec<PolynomialFactorization<R>>, DecompositionError> {
    source
        .generators()
        .iter()
        .map(|generator| {
            let factorization = R::factor(generator)?;
            factorization.verify()?;
            Ok(factorization)
        })
        .collect()
}

pub(super) fn common_irreducible_factors<R: RadicalCoefficientField>(
    factorizations: &[PolynomialFactorization<R>],
) -> Vec<(Polynomial<R>, u64)> {
    let Some(first) = factorizations.first() else {
        return Vec::new();
    };
    let mut common = first.factors().to_vec();
    for factorization in &factorizations[1..] {
        common.retain_mut(|(factor, multiplicity)| {
            let available = factorization
                .factors()
                .iter()
                .find_map(|(candidate, exponent)| (candidate == factor).then_some(*exponent))
                .unwrap_or(0);
            *multiplicity = (*multiplicity).min(available);
            *multiplicity != 0
        });
        if common.is_empty() {
            break;
        }
    }
    common
}

fn construct_common_factor_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    factorizations: &[PolynomialFactorization<R>],
    remaining_depth: usize,
    allow_saturation: bool,
) -> Result<Option<RadicalIdeal<R>>, DecompositionError> {
    let common_factors = common_irreducible_factors(factorizations);
    if common_factors.is_empty() {
        return Ok(None);
    }
    let mut common = source.ring().one();
    let mut square_free = source.ring().one();
    for (factor, multiplicity) in &common_factors {
        common = (&common * &factor.pow(*multiplicity)?)?;
        square_free = (&square_free * factor)?;
    }
    let residual_source = source.ring().ideal(
        source
            .generators()
            .iter()
            .map(|generator| generator.exact_div(&common))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    if &residual_source == source {
        return Err(DecompositionError::Invalid(
            "a common-factor split did not decrease its source ideal".to_owned(),
        ));
    }
    let residual =
        construct_radical_with_depth(&residual_source, remaining_depth - 1, allow_saturation)?;
    let common_radical = source.ring().ideal([square_free])?;
    let radical = common_radical.intersection(residual.ideal())?;
    Ok(Some(RadicalIdeal {
        source: source.clone(),
        radical,
        proof: RadicalProof::CommonFactorSplit {
            common_factors,
            residual_source,
            residual_radical: residual.radical,
            residual_proof: Box::new(residual.proof),
        },
    }))
}

fn construct_generator_factor_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    factorizations: &[PolynomialFactorization<R>],
    remaining_depth: usize,
) -> Result<Option<RadicalIdeal<R>>, DecompositionError> {
    for (splitting_generator, factorization) in source.generators().iter().zip(factorizations) {
        if factorization.factors().is_empty()
            || factorization.factors() == [(splitting_generator.clone(), 1)]
        {
            continue;
        }
        if factorization.factors().len() > MAX_RADICAL_SPLIT_BRANCHES {
            return Err(DecompositionError::ResourceLimit {
                resource: "generator-factor radical branches",
                limit: MAX_RADICAL_SPLIT_BRANCHES,
            });
        }

        let mut branch_sources = Vec::with_capacity(factorization.factors().len());
        let mut branch_radicals = Vec::with_capacity(factorization.factors().len());
        let mut branch_proofs = Vec::with_capacity(factorization.factors().len());
        let mut supported = true;
        for (factor, _) in factorization.factors() {
            let factor_ideal = source.ring().ideal([factor.clone()])?;
            let branch_source = source.sum(&factor_ideal)?;
            if &branch_source == source {
                supported = false;
                break;
            }
            // Generator-factor splitting is a cheap preflight.  Recursive
            // closed/open searches belong to the saturation route below;
            // launching them here duplicates the same expanding search once
            // for every factor branch.
            let branch =
                match construct_radical_with_depth(&branch_source, remaining_depth - 1, false) {
                    Ok(branch) => branch,
                    Err(DecompositionError::Unsupported(_)) => {
                        supported = false;
                        break;
                    }
                    Err(error) => return Err(error),
                };
            branch_sources.push(branch_source);
            branch_radicals.push(branch.radical);
            branch_proofs.push(branch.proof);
        }
        if !supported {
            continue;
        }

        let mut radical = branch_radicals.first().cloned().ok_or_else(|| {
            DecompositionError::Invalid(
                "a generator-factor split produced no radical branches".to_owned(),
            )
        })?;
        for branch_radical in &branch_radicals[1..] {
            radical = radical.intersection(branch_radical)?;
        }
        return Ok(Some(RadicalIdeal {
            source: source.clone(),
            radical,
            proof: RadicalProof::GeneratorFactorSplit {
                splitting_generator: splitting_generator.clone(),
                factors: factorization.factors().to_vec(),
                branch_sources,
                branch_radicals,
                branch_proofs,
            },
        }));
    }
    Ok(None)
}

fn saturation_split_candidates<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    factorizations: &[PolynomialFactorization<R>],
) -> Result<Vec<Polynomial<R>>, DecompositionError> {
    let mut candidates = Vec::new();

    fn retain_candidate<R: RadicalCoefficientField>(
        source: &Ideal<PolynomialRing<R>>,
        candidates: &mut Vec<Polynomial<R>>,
        candidate: Polynomial<R>,
    ) -> Result<(), DecompositionError> {
        if candidates.len() >= MAX_SATURATION_SPLIT_CANDIDATES || candidate.is_zero() {
            return Ok(());
        }
        let candidate = candidate.monic()?;
        if candidate.is_one()
            || source.contains(&candidate)?
            || candidates.iter().any(|existing| existing == &candidate)
        {
            return Ok(());
        }
        candidates.push(candidate);
        Ok(())
    }

    // Coordinate witnesses are cheap and commonly separate coordinate or
    // boundary components.
    for variable in source.ring().generators()? {
        retain_candidate(source, &mut candidates, variable)?;
    }

    // A canonical basis element can factor even when its factor branches do
    // not all terminate independently. Try richer factors first: they more
    // often define a whole component, while low-term factors tend to cut it.
    let mut factors = factorizations
        .iter()
        .flat_map(|factorization| {
            factorization
                .factors()
                .iter()
                .map(|(factor, _)| factor.clone())
        })
        .collect::<Vec<_>>();
    factors.sort_by_key(|factor| std::cmp::Reverse(factor.nonzero_term_count()));
    for factor in factors {
        retain_candidate(source, &mut candidates, factor)?;
    }

    for generator in source.generators() {
        if candidates.len() >= MAX_SATURATION_SPLIT_CANDIDATES {
            break;
        }
        // Sparse equations defining a component are often visible among two
        // monomials of a reduced-basis element even when the equation itself
        // is not a basis element. Enumerate both unit-coefficient binomials;
        // the global cap keeps this exact heuristic bounded.
        let monomials = generator
            .terms()
            .map(|(exponent, _)| {
                source
                    .ring()
                    .from_terms([(exponent.to_vec(), source.ring().coefficient_ring().one())])
            })
            .collect::<Result<Vec<_>, _>>()?;
        'pairs: for left in 0..monomials.len() {
            for right in (left + 1)..monomials.len() {
                retain_candidate(
                    source,
                    &mut candidates,
                    (&monomials[left] - &monomials[right])?,
                )?;
                retain_candidate(
                    source,
                    &mut candidates,
                    (&monomials[left] + &monomials[right])?,
                )?;
                if candidates.len() >= MAX_SATURATION_SPLIT_CANDIDATES {
                    break 'pairs;
                }
            }
        }
    }
    Ok(candidates)
}

fn construct_saturation_split_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    factorizations: &[PolynomialFactorization<R>],
    remaining_depth: usize,
) -> Result<Option<RadicalIdeal<R>>, DecompositionError> {
    for witness in saturation_split_candidates(source, factorizations)? {
        let witness_ideal = source.ring().ideal([witness.clone()])?;
        let closed_source = source.sum(&witness_ideal)?;
        if &closed_source == source {
            continue;
        }
        let open_source = source.saturate(&witness_ideal)?;
        if &open_source == source {
            continue;
        }

        let closed = match construct_radical_with_depth(&closed_source, remaining_depth - 1, true) {
            Ok(value) => value,
            Err(DecompositionError::Unsupported(_)) => continue,
            Err(error) => return Err(error),
        };
        let open = match construct_radical_with_depth(&open_source, remaining_depth - 1, true) {
            Ok(value) => value,
            Err(DecompositionError::Unsupported(_)) => continue,
            Err(error) => return Err(error),
        };
        let radical = closed.radical.intersection(&open.radical)?;
        return Ok(Some(RadicalIdeal {
            source: source.clone(),
            radical,
            proof: RadicalProof::SaturationSplit {
                witness,
                closed_source,
                closed_radical: closed.radical,
                closed_proof: Box::new(closed.proof),
                open_source,
                open_radical: open.radical,
                open_proof: Box::new(open.proof),
            },
        }));
    }
    Ok(None)
}

pub(super) fn construct_zero_dimensional_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<RadicalIdeal<R>, DecompositionError> {
    if let Some(exponents) = monomial_generators(source) {
        return construct_zero_dimensional_monomial_radical(source, exponents);
    }
    let quotient = source.finite_quotient_algebra()?;
    let variable_count = source.ring().variables().len();
    if quotient.multiplication_operator_nonzeros().len() != variable_count {
        return Err(DecompositionError::Invalid(
            "the finite quotient has the wrong number of multiplication operators".to_owned(),
        ));
    }
    let mut characteristic_polynomials = Vec::with_capacity(variable_count);
    let mut coordinate_certificates = Vec::with_capacity(variable_count);
    for variable in 0..variable_count {
        let characteristic = multiplication_matrix(&quotient, variable)?.charpoly()?;
        let square_free = R::square_free_part(&characteristic)?;
        let certificate = map_parameter_to_variable(&square_free, source.ring(), variable)?;
        characteristic_polynomials.push(characteristic);
        coordinate_certificates.push(certificate);
    }
    let generators = source
        .generators()
        .iter()
        .cloned()
        .chain(coordinate_certificates.iter().cloned())
        .collect::<Vec<_>>();
    let radical = source.ring().ideal(generators)?;
    Ok(RadicalIdeal {
        source: source.clone(),
        radical,
        proof: RadicalProof::ZeroDimensionalCoordinates {
            characteristic_polynomials,
            coordinate_certificates,
        },
    })
}

fn construct_radical_with_depth<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
    remaining_depth: usize,
    allow_saturation: bool,
) -> Result<RadicalIdeal<R>, DecompositionError> {
    if let Some(exponents) = monomial_generators(source) {
        let monomial = source.ring().monomial_ideal(exponents.clone())?;
        if monomial.dimension()? > 0 {
            return construct_positive_dimensional_monomial_radical(source, exponents);
        }
        return construct_zero_dimensional_monomial_radical(source, exponents);
    }
    if source.ring().variables().len() > 1 && source.generators().len() == 1 {
        return construct_principal_radical(source);
    }
    if let Some(pivot_variables) = affine_linear_pivots(source) {
        return Ok(construct_affine_linear_radical(source, pivot_variables));
    }
    if let Some(pivot_variables) = polynomial_graph_pivots(source) {
        return Ok(construct_polynomial_graph_radical(source, pivot_variables));
    }
    let dimension = source.dimension()?;
    if dimension > 0 {
        if remaining_depth == 0 {
            return Err(DecompositionError::ResourceLimit {
                resource: "positive-dimensional radical splitting depth",
                limit: MAX_RADICAL_SPLIT_DEPTH,
            });
        }
        let factorizations = generator_factorizations(source)?;
        match construct_common_factor_radical(
            source,
            &factorizations,
            remaining_depth,
            allow_saturation,
        ) {
            Ok(Some(radical)) => return Ok(radical),
            Ok(None) | Err(DecompositionError::Unsupported(_)) => {}
            Err(error) => return Err(error),
        }
        if let Some(radical) =
            construct_generator_factor_radical(source, &factorizations, remaining_depth)?
        {
            return Ok(radical);
        }
        if let Some(radical) = construct_complete_intersection_radical(source, dimension)? {
            return Ok(radical);
        }
        if allow_saturation {
            if let Some(radical) =
                construct_saturation_split_radical(source, &factorizations, remaining_depth)?
            {
                return Ok(radical);
            }
        }
        return Err(DecompositionError::Unsupported(
            "general positive-dimensional radical computation requires a certified split into components or a terminal reducedness certificate".to_owned(),
        ));
    }
    construct_zero_dimensional_radical(source)
}

pub(super) fn construct_radical<R: RadicalCoefficientField>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<RadicalIdeal<R>, DecompositionError> {
    construct_radical_with_depth(source, MAX_RADICAL_SPLIT_DEPTH, true)
}

pub(super) fn verify_radical<R: RadicalCoefficientField>(
    value: &RadicalIdeal<R>,
) -> Result<(), DecompositionError> {
    value.source.verify()?;
    let rebuilt = construct_radical(&value.source)?;
    if &rebuilt != value {
        return Err(DecompositionError::Invalid(
            "recomputation changed the radical evidence".to_owned(),
        ));
    }
    Ok(())
}

impl Ideal<PolynomialRing<RationalField>> {
    /// Prove the radical of an admitted rational polynomial ideal.
    pub fn radical(&self) -> Result<RadicalIdeal<RationalField>, DecompositionError> {
        construct_radical(self)
    }
}

impl Ideal<PolynomialRing<FiniteField>> {
    /// Prove the radical of an admitted finite-field polynomial ideal.
    pub fn radical(&self) -> Result<RadicalIdeal<FiniteField>, DecompositionError> {
        construct_radical(self)
    }
}

impl RadicalIdeal<RationalField> {
    /// Independently recompute the rational radical evidence.
    pub fn verify(&self) -> Result<(), DecompositionError> {
        verify_radical(self)
    }

    /// Deserialize and independently recompute rational radical evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl RadicalIdeal<FiniteField> {
    /// Independently recompute the finite-field radical evidence.
    pub fn verify(&self) -> Result<(), DecompositionError> {
        verify_radical(self)
    }

    /// Deserialize and independently recompute finite-field radical evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> RadicalIdeal<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the complete radical evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for RadicalIdeal<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("RadicalIdeal", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("radical", &self.radical)?;
        state.serialize_field("proof", &self.proof)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct RadicalIdealWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    radical: Ideal<PolynomialRing<R>>,
    proof: RadicalProof<R>,
}

fn deserialize_radical<'de, R, D>(deserializer: D) -> Result<RadicalIdeal<R>, D::Error>
where
    R: RadicalCoefficientField + Deserialize<'de>,
    R::Element: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = RadicalIdealWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/RadicalIdeal@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/RadicalIdeal@1", wire.type_id
        )));
    }
    let result = RadicalIdeal {
        source: wire.source,
        radical: wire.radical,
        proof: wire.proof,
    };
    verify_radical(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for RadicalIdeal<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_radical(deserializer)
    }
}

impl<'de> Deserialize<'de> for RadicalIdeal<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_radical(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::numthy::FiniteField;
    use crate::{SemanticType, Q};

    #[test]
    fn rational_coordinate_certificates_remove_all_nilpotents() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([variables[0].pow(2).unwrap(), variables[1].pow(3).unwrap()])
            .unwrap();

        let radical = source.radical().unwrap();
        radical.verify().unwrap();
        assert_eq!(
            radical.ideal(),
            &ring
                .ideal([variables[0].clone(), variables[1].clone()])
                .unwrap()
        );
        assert_eq!(radical.characteristic_polynomials().len(), 2);
        assert_eq!(radical.coordinate_certificates().len(), 2);
        let parameter = radical.characteristic_polynomials()[0]
            .ring()
            .generator(0)
            .unwrap();
        assert_eq!(
            radical.characteristic_polynomials(),
            &[parameter.pow(6).unwrap(), parameter.pow(6).unwrap()]
        );
        assert_eq!(
            RadicalIdeal::<RationalField>::from_json(&radical.to_json().unwrap()).unwrap(),
            radical
        );
        assert_eq!(
            RadicalIdeal::<RationalField>::TYPE_ID,
            "noether/commalg/RadicalIdeal@1"
        );
    }

    #[test]
    fn unit_monomial_ideal_has_zero_dimensional_zero_algebra_evidence() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let source = ring.ideal([ring.one()]).unwrap();

        let radical = source.radical().unwrap();
        radical.verify().unwrap();
        assert_eq!(radical.ideal(), &source);
        assert!(radical
            .characteristic_polynomials()
            .iter()
            .all(Polynomial::is_one));
        assert!(radical
            .coordinate_certificates()
            .iter()
            .all(Polynomial::is_one));
    }

    #[test]
    fn finite_characteristic_multiplicities_do_not_use_the_trace_form() {
        let field = FiniteField::prime(2_u64).unwrap();
        let ring = PolynomialRing::new(field, ["x"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        let source = ring
            .ideal([(&x.pow(4).unwrap() + &x.pow(2).unwrap()).unwrap()])
            .unwrap();

        let radical = source.radical().unwrap();
        radical.verify().unwrap();
        assert_eq!(
            radical.ideal(),
            &ring.ideal([(&x.pow(2).unwrap() + &x).unwrap()]).unwrap()
        );
        assert_eq!(
            RadicalIdeal::<FiniteField>::from_json(&radical.to_json().unwrap()).unwrap(),
            radical
        );
    }

    #[test]
    fn nonmonogenic_reduced_f2_algebra_is_unchanged() {
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

        let radical = source.radical().unwrap();
        assert_eq!(radical.ideal(), &source);
        radical.verify().unwrap();
    }

    #[test]
    fn positive_dimensional_principal_ideal_uses_multivariate_factorization() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let first = (&variables[0] + &variables[1]).unwrap();
        let second = (&variables[0] - &variables[2]).unwrap();
        let source = ring
            .ideal([(&first.pow(3).unwrap() * &second.pow(2).unwrap()).unwrap()])
            .unwrap();

        let radical = source.radical().unwrap();
        radical.verify().unwrap();
        assert_eq!(radical.proof_kind(), "principal_factorization");
        assert_eq!(radical.principal_factors().unwrap().len(), 2);
        assert_eq!(
            radical.ideal(),
            &ring.ideal([(&first * &second).unwrap()]).unwrap()
        );
        assert_eq!(
            RadicalIdeal::<RationalField>::from_json(&radical.to_json().unwrap()).unwrap(),
            radical
        );
    }

    #[test]
    fn positive_dimensional_monomial_ideal_uses_support_reduction() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&variables[0].pow(3).unwrap() * &variables[1]).unwrap(),
                (&variables[1].pow(2).unwrap() * &variables[2].pow(4).unwrap()).unwrap(),
            ])
            .unwrap();

        let radical = source.radical().unwrap();
        radical.verify().unwrap();
        assert_eq!(radical.proof_kind(), "monomial_support");
        let (monomial_source, monomial_radical) = radical.monomial_proof().unwrap();
        assert_eq!(monomial_source.dimension().unwrap(), 2);
        assert_eq!(monomial_radical, &monomial_source.radical().unwrap());
        assert_eq!(radical.ideal(), &monomial_radical.to_ideal().unwrap());
    }

    #[test]
    fn unsupported_positive_dimensional_ideal_still_fails_closed() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z", "w"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&(&variables[0] * &variables[2]).unwrap() - &variables[1].pow(2).unwrap())
                    .unwrap(),
                (&(&variables[0] * &variables[3]).unwrap()
                    - &(&variables[1] * &variables[2]).unwrap())
                    .unwrap(),
                (&(&variables[1] * &variables[3]).unwrap() - &variables[2].pow(2).unwrap())
                    .unwrap(),
            ])
            .unwrap();
        assert!(source.radical().is_err());
    }

    #[test]
    fn common_factor_split_handles_a_nonmonomial_mixed_dimensional_ideal() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let common = (&variables[2] - &ring.one()).unwrap();
        let common_power = common.pow(2).unwrap();
        let source = ring
            .ideal([
                (&common_power * &variables[0]).unwrap(),
                (&common_power * &variables[1]).unwrap(),
            ])
            .unwrap();

        let radical = source.radical().unwrap();

        radical.verify().unwrap();
        assert_eq!(radical.proof_kind(), "common_factor_split");
        assert_eq!(
            radical.ideal(),
            &ring
                .ideal([
                    (&common * &variables[0]).unwrap(),
                    (&common * &variables[1]).unwrap(),
                ])
                .unwrap()
        );
        let factors = radical.common_factors().unwrap();
        let residual_source = radical.common_factor_residual_source().unwrap();
        let residual_radical = radical.common_factor_residual_radical().unwrap();
        assert_eq!(factors, [(common, 2)]);
        assert_eq!(
            residual_source,
            &ring
                .ideal([variables[0].clone(), variables[1].clone()])
                .unwrap()
        );
        assert_eq!(residual_radical, residual_source);
        assert_eq!(
            RadicalIdeal::<RationalField>::from_json(&radical.to_json().unwrap()).unwrap(),
            radical
        );

        let mut payload = serde_json::to_value(&radical).unwrap();
        let source_payload = payload["source"].clone();
        payload["proof"]["residual_radical"] = source_payload;
        assert!(serde_json::from_value::<RadicalIdeal<RationalField>>(payload).is_err());
    }

    #[test]
    fn affine_linear_ideals_are_radical_with_polynomial_ring_quotients() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&(&variables[0] + &variables[1]).unwrap() + &ring.one()).unwrap(),
                (&variables[2] - &variables[1]).unwrap(),
            ])
            .unwrap();

        let radical = source.radical().unwrap();

        radical.verify().unwrap();
        assert_eq!(radical.proof_kind(), "affine_linear");
        assert_eq!(radical.ideal(), &source);
        assert_eq!(radical.affine_linear_pivots().unwrap().len(), 2);
    }

    #[test]
    fn generator_factor_split_recurses_to_affine_linear_branches() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let left = (&variables[0] + &ring.one()).unwrap();
        let right = (&variables[1] + &ring.one()).unwrap();
        let product = (&left.pow(2).unwrap() * &right.pow(3).unwrap()).unwrap();
        let source = ring.ideal([variables[2].clone(), product.clone()]).unwrap();

        let radical = source.radical().unwrap();

        radical.verify().unwrap();
        assert_eq!(radical.proof_kind(), "generator_factor_split");
        assert_eq!(radical.factor_split_generator(), Some(&product));
        let factors = radical.factor_split_factors().unwrap();
        assert_eq!(factors.len(), 2);
        assert!(factors.contains(&(left.clone(), 2)));
        assert!(factors.contains(&(right.clone(), 3)));
        assert_eq!(radical.factor_split_branch_sources().unwrap().len(), 2);
        assert_eq!(radical.factor_split_branch_radicals().unwrap().len(), 2);
        assert_eq!(
            radical.ideal(),
            &ring
                .ideal([variables[2].clone(), (&left * &right).unwrap(),])
                .unwrap()
        );
        assert_eq!(
            RadicalIdeal::<RationalField>::from_json(&radical.to_json().unwrap()).unwrap(),
            radical
        );

        let mut payload = serde_json::to_value(&radical).unwrap();
        payload["proof"]["branch_radicals"] = serde_json::json!([]);
        assert!(serde_json::from_value::<RadicalIdeal<RationalField>>(payload).is_err());
    }

    #[test]
    fn nonlinear_polynomial_graph_is_radical_and_prime_terminated() {
        let ring = PolynomialRing::new(Q, ["y", "z", "x"], MonomialOrder::Lexicographic).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&variables[0] - &variables[2].pow(2).unwrap()).unwrap(),
                (&variables[1] - &variables[2].pow(3).unwrap()).unwrap(),
            ])
            .unwrap();

        let radical = source.radical().unwrap();

        radical.verify().unwrap();
        assert_eq!(radical.proof_kind(), "polynomial_graph");
        assert_eq!(radical.polynomial_graph_pivots(), Some(&[0, 1][..]));
        assert_eq!(radical.ideal(), &source);
    }

    #[test]
    fn generically_smooth_complete_intersection_is_certified_reduced() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let sphere = (&(&variables[0].pow(2).unwrap() + &variables[1].pow(2).unwrap()).unwrap()
            - &ring.one())
            .unwrap();
        let cover = (&variables[2].pow(2).unwrap() - &variables[0]).unwrap();
        let source = ring.ideal([sphere, cover]).unwrap();

        let radical = source.radical().unwrap();

        radical.verify().unwrap();
        assert_eq!(radical.proof_kind(), "complete_intersection_jacobian");
        assert_eq!(radical.complete_intersection_codimension(), Some(2));
        assert_eq!(radical.jacobian_minors().unwrap().len(), 3);
        assert_eq!(radical.singular_locus_dimension(), Some(-1));
        assert_eq!(radical.ideal(), &source);
        assert_eq!(
            RadicalIdeal::<RationalField>::from_json(&radical.to_json().unwrap()).unwrap(),
            radical
        );

        let mut payload = serde_json::to_value(&radical).unwrap();
        payload["proof"]["singular_dimension"] = serde_json::json!(0);
        assert!(serde_json::from_value::<RadicalIdeal<RationalField>>(payload).is_err());
    }

    #[test]
    fn saturation_split_separates_complete_intersection_components() {
        fn first_saturation_node(
            value: &RadicalIdeal<FiniteField>,
        ) -> Option<RadicalIdeal<FiniteField>> {
            match &value.proof {
                RadicalProof::SaturationSplit { .. } => Some(value.clone()),
                RadicalProof::CommonFactorSplit {
                    residual_source,
                    residual_radical,
                    residual_proof,
                    ..
                } => first_saturation_node(&RadicalIdeal {
                    source: residual_source.clone(),
                    radical: residual_radical.clone(),
                    proof: residual_proof.as_ref().clone(),
                }),
                RadicalProof::GeneratorFactorSplit {
                    branch_sources,
                    branch_radicals,
                    branch_proofs,
                    ..
                } => branch_sources
                    .iter()
                    .zip(branch_radicals)
                    .zip(branch_proofs)
                    .find_map(|((source, radical), proof)| {
                        first_saturation_node(&RadicalIdeal {
                            source: source.clone(),
                            radical: radical.clone(),
                            proof: proof.clone(),
                        })
                    }),
                RadicalProof::PrincipalFactorization { .. }
                | RadicalProof::MonomialSupport { .. }
                | RadicalProof::ZeroDimensionalCoordinates { .. }
                | RadicalProof::AffineLinear { .. }
                | RadicalProof::PolynomialGraph { .. }
                | RadicalProof::CompleteIntersectionJacobian { .. } => None,
            }
        }

        let field = FiniteField::prime(101_u64).unwrap();
        let ring =
            PolynomialRing::new(field, ["x", "y", "z", "w"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let [x, y, z, w] = variables.as_slice() else {
            panic!("expected four variables");
        };
        let scalar = |value| {
            ring.constant(ring.coefficient_ring().element(value))
                .unwrap()
        };
        let first = ring
            .ideal([
                (&(&(&x.pow(2).unwrap() + &z.pow(2).unwrap()).unwrap()
                    + &(&(&scalar(2) * z).unwrap() * w).unwrap())
                    .unwrap()
                    + &(&scalar(3) * &w.pow(2).unwrap()).unwrap())
                    .unwrap(),
                (&(&(&y.pow(2).unwrap() + &(&scalar(4) * &z.pow(2).unwrap()).unwrap()).unwrap()
                    + &(&(&scalar(5) * z).unwrap() * w).unwrap())
                    .unwrap()
                    + &(&scalar(6) * &w.pow(2).unwrap()).unwrap())
                    .unwrap(),
            ])
            .unwrap();
        let second = ring
            .ideal([
                (&(&(&x.pow(2).unwrap() + &y.pow(2).unwrap()).unwrap()
                    + &(&(&scalar(2) * y).unwrap() * w).unwrap())
                    .unwrap()
                    + &(&scalar(3) * &w.pow(2).unwrap()).unwrap())
                    .unwrap(),
                (&(&(&z.pow(2).unwrap() + &(&scalar(4) * &y.pow(2).unwrap()).unwrap()).unwrap()
                    + &(&(&scalar(5) * y).unwrap() * w).unwrap())
                    .unwrap()
                    + &(&scalar(6) * &w.pow(2).unwrap()).unwrap())
                    .unwrap(),
            ])
            .unwrap();
        let source = first.intersection(&second).unwrap();

        let radical = source.radical().unwrap();

        radical.verify().unwrap();
        assert_eq!(radical.proof_kind(), "saturation_split");
        assert_eq!(radical.ideal(), &source);
        assert!(radical.to_json().unwrap().contains("saturation_split"));

        let split = first_saturation_node(&radical).unwrap();
        split.verify().unwrap();
        assert_eq!(split.proof_kind(), "saturation_split");
        assert!(split.saturation_split_witness().is_some());
        let closed_source = split.saturation_split_closed_source().unwrap();
        let open_source = split.saturation_split_open_source().unwrap();
        assert_ne!(closed_source, split.source());
        assert_ne!(open_source, split.source());
        let reconstructed = split
            .saturation_split_closed_radical()
            .unwrap()
            .intersection(split.saturation_split_open_radical().unwrap())
            .unwrap();
        assert_eq!(&reconstructed, split.ideal());
        assert_eq!(
            RadicalIdeal::<FiniteField>::from_json(&split.to_json().unwrap()).unwrap(),
            split
        );
        let mut payload = serde_json::to_value(&split).unwrap();
        payload["proof"]["closed_radical"] =
            serde_json::to_value(ring.ideal([ring.one()]).unwrap()).unwrap();
        assert!(serde_json::from_value::<RadicalIdeal<FiniteField>>(payload).is_err());
        assert_eq!(
            RadicalIdeal::<FiniteField>::from_json(&radical.to_json().unwrap()).unwrap(),
            radical
        );
    }
}
