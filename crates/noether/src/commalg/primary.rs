// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Verified primary decompositions of admitted polynomial ideals.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::FiniteField;
use crate::{Field, Integer, NoetherError, RationalField, Ring, SemanticType, TypeDescription};

use super::associated_primes::{construct_associated_primes, verify_associated_primes};
use super::equidimensional_hull::{construct_equidimensional_hull, verify_equidimensional_hull};
use super::minimal_primes::{
    construct_minimal_primes, coordinate_prime_certificate, verify_polynomial_prime_certificate,
    MinimalPrimeCoefficientField,
};
use super::{
    AssociatedPrimeError, AssociatedPrimeSet, DecompositionError, EquidimensionalHull, Ideal,
    MonomialIdeal, Polynomial, PolynomialPrimeCertificate, PolynomialRing,
};

const MAX_MONOMIAL_IRREDUCIBLE_CANDIDATES: usize = 65_536;
const MAX_GRAPH_COMPLETE_INTERSECTION_COMPONENTS: usize = 4_096;
const MAX_EHV_EMBEDDED_EXPONENT: u64 = 8;
const MAX_EHV_PRIMARY_COMBINATIONS: usize = 4_096;

type PolynomialGraphResidual<R> = (Ideal<PolynomialRing<R>>, Vec<usize>, Vec<Polynomial<R>>);

static POLYNOMIAL_PRIMARY_CERTIFICATE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PolynomialPrimaryCertificate@1",
    rust_path: "noether::commalg::PolynomialPrimaryCertificate",
    python_path: Some("noether.commalg.PolynomialPrimaryCertificate"),
};

static PRIMARY_DECOMPOSITION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PrimaryDecomposition@1",
    rust_path: "noether::commalg::PrimaryDecomposition",
    python_path: Some("noether.commalg.PrimaryDecomposition"),
};

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "method",
    rename_all = "snake_case",
    bound(
        serialize = "R: Serialize, R::Element: Serialize",
        deserialize = "R: Deserialize<'de>, R::Element: Deserialize<'de>, AssociatedPrimeSet<R>: Deserialize<'de>, EquidimensionalHull<R>: Deserialize<'de>"
    )
)]
enum PolynomialPrimaryProof<R: Field> {
    Prime,
    EhvSingleAssociatedPrime {
        associated_primes: Box<AssociatedPrimeSet<R>>,
    },
    EhvEquidimensionalHull {
        hull: Box<EquidimensionalHull<R>>,
    },
    PolynomialGraphMonomialPrimary {
        base: Ideal<PolynomialRing<R>>,
        pivot_variables: Vec<usize>,
        residual: MonomialIdeal<R>,
        variables: Vec<usize>,
    },
    PolynomialGraphCompleteIntersectionPowers {
        base: Ideal<PolynomialRing<R>>,
        pivot_variables: Vec<usize>,
        factors: Vec<(Polynomial<R>, u64)>,
    },
    PrincipalPower {
        factor: Polynomial<R>,
        exponent: u64,
    },
    MonomialPrimary {
        source: MonomialIdeal<R>,
        variables: Vec<usize>,
    },
    ZeroDimensionalLocal {
        quotient_dimension: usize,
    },
}

/// Exact evidence that one polynomial ideal is primary.
///
/// The native proof routes are a prime ideal; the inverse image of a monomial
/// primary ideal through a polynomial-graph quotient; powers of a prime
/// complete-intersection regular sequence over a polynomial-graph base; a
/// power of an irreducible hypersurface; the structural criterion for a
/// monomial ideal primary to a coordinate prime; a zero-dimensional local
/// quotient with an explicit nilpotence bound; and a complete EHV computation
/// proving that the quotient has exactly one associated prime.
#[derive(Clone, PartialEq, Eq)]
pub struct PolynomialPrimaryCertificate<R: Field> {
    ideal: Ideal<PolynomialRing<R>>,
    associated_prime: PolynomialPrimeCertificate<R>,
    proof: PolynomialPrimaryProof<R>,
}

impl<R: Field> PolynomialPrimaryCertificate<R> {
    /// Canonical polynomial ideal certified as primary.
    pub fn ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.ideal
    }

    /// Exact prime equal to the radical of this primary ideal.
    pub fn associated_prime(&self) -> &PolynomialPrimeCertificate<R> {
        &self.associated_prime
    }

    /// Canonical name of the exact primary proof route.
    pub const fn proof_kind(&self) -> &'static str {
        match &self.proof {
            PolynomialPrimaryProof::Prime => "prime",
            PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. } => {
                "ehv_single_associated_prime"
            }
            PolynomialPrimaryProof::EhvEquidimensionalHull { .. } => "ehv_equidimensional_hull",
            PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { .. } => {
                "polynomial_graph_monomial_primary"
            }
            PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers {
                factors, ..
            } => {
                if factors.len() == 1 {
                    "polynomial_graph_hypersurface_power"
                } else {
                    "polynomial_graph_complete_intersection_powers"
                }
            }
            PolynomialPrimaryProof::PrincipalPower { .. } => "principal_power",
            PolynomialPrimaryProof::MonomialPrimary { .. } => "monomial_primary",
            PolynomialPrimaryProof::ZeroDimensionalLocal { .. } => "zero_dimensional_local",
        }
    }

    /// Nilpotence bound supplied by the zero-dimensional local route.
    pub const fn nilpotence_bound(&self) -> Option<usize> {
        match &self.proof {
            PolynomialPrimaryProof::ZeroDimensionalLocal { quotient_dimension } => {
                Some(*quotient_dimension)
            }
            PolynomialPrimaryProof::Prime
            | PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. }
            | PolynomialPrimaryProof::EhvEquidimensionalHull { .. }
            | PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { .. }
            | PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers { .. }
            | PolynomialPrimaryProof::PrincipalPower { .. }
            | PolynomialPrimaryProof::MonomialPrimary { .. } => None,
        }
    }

    /// Polynomial-graph base below a graph-relative primary component.
    pub fn polynomial_graph_base(&self) -> Option<&Ideal<PolynomialRing<R>>> {
        match &self.proof {
            PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { base, .. }
            | PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers { base, .. } => {
                Some(base)
            }
            PolynomialPrimaryProof::Prime
            | PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. }
            | PolynomialPrimaryProof::EhvEquidimensionalHull { .. }
            | PolynomialPrimaryProof::PrincipalPower { .. }
            | PolynomialPrimaryProof::MonomialPrimary { .. }
            | PolynomialPrimaryProof::ZeroDimensionalLocal { .. } => None,
        }
    }

    /// Pivot variables eliminated by the polynomial-graph base.
    pub fn polynomial_graph_pivots(&self) -> Option<&[usize]> {
        match &self.proof {
            PolynomialPrimaryProof::PolynomialGraphMonomialPrimary {
                pivot_variables, ..
            }
            | PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers {
                pivot_variables,
                ..
            } => Some(pivot_variables),
            PolynomialPrimaryProof::Prime
            | PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. }
            | PolynomialPrimaryProof::EhvEquidimensionalHull { .. }
            | PolynomialPrimaryProof::PrincipalPower { .. }
            | PolynomialPrimaryProof::MonomialPrimary { .. }
            | PolynomialPrimaryProof::ZeroDimensionalLocal { .. } => None,
        }
    }

    /// Irreducible free-variable factor defining the component radical.
    pub fn polynomial_graph_factor(&self) -> Option<&Polynomial<R>> {
        match &self.proof {
            PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers {
                factors, ..
            } if factors.len() == 1 => Some(&factors[0].0),
            PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers { .. } => None,
            PolynomialPrimaryProof::Prime
            | PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. }
            | PolynomialPrimaryProof::EhvEquidimensionalHull { .. }
            | PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { .. }
            | PolynomialPrimaryProof::PrincipalPower { .. }
            | PolynomialPrimaryProof::MonomialPrimary { .. }
            | PolynomialPrimaryProof::ZeroDimensionalLocal { .. } => None,
        }
    }

    /// Positive multiplicity of the hypersurface factor over the graph base.
    pub fn polynomial_graph_factor_exponent(&self) -> Option<u64> {
        match &self.proof {
            PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers {
                factors, ..
            } if factors.len() == 1 => Some(factors[0].1),
            PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers { .. } => None,
            PolynomialPrimaryProof::Prime
            | PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. }
            | PolynomialPrimaryProof::EhvEquidimensionalHull { .. }
            | PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { .. }
            | PolynomialPrimaryProof::PrincipalPower { .. }
            | PolynomialPrimaryProof::MonomialPrimary { .. }
            | PolynomialPrimaryProof::ZeroDimensionalLocal { .. } => None,
        }
    }

    /// Irreducible regular-sequence factors and their positive multiplicities.
    pub fn polynomial_graph_factors(&self) -> Option<&[(Polynomial<R>, u64)]> {
        match &self.proof {
            PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers {
                factors, ..
            } => Some(factors),
            PolynomialPrimaryProof::Prime
            | PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. }
            | PolynomialPrimaryProof::EhvEquidimensionalHull { .. }
            | PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { .. }
            | PolynomialPrimaryProof::PrincipalPower { .. }
            | PolynomialPrimaryProof::MonomialPrimary { .. }
            | PolynomialPrimaryProof::ZeroDimensionalLocal { .. } => None,
        }
    }

    /// Monomial primary ideal pulled back through a polynomial-graph quotient.
    pub fn polynomial_graph_monomial_source(&self) -> Option<&MonomialIdeal<R>> {
        match &self.proof {
            PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { residual, .. } => {
                Some(residual)
            }
            PolynomialPrimaryProof::Prime
            | PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. }
            | PolynomialPrimaryProof::EhvEquidimensionalHull { .. }
            | PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers { .. }
            | PolynomialPrimaryProof::PrincipalPower { .. }
            | PolynomialPrimaryProof::MonomialPrimary { .. }
            | PolynomialPrimaryProof::ZeroDimensionalLocal { .. } => None,
        }
    }

    /// Exact theorem turning the retained evidence into a primary claim.
    pub const fn theorem(&self) -> &'static str {
        match &self.proof {
            PolynomialPrimaryProof::Prime => "every prime ideal is primary",
            PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. } => {
                "an ideal in a Noetherian ring is primary exactly when its quotient has one associated prime"
            }
            PolynomialPrimaryProof::EhvEquidimensionalHull { .. } => {
                "an equidimensional hull is the intersection of the top-dimensional primary components, so a hull with one minimal prime is primary"
            }
            PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { .. } => {
                "the inverse image of a primary ideal is primary, and a polynomial-graph quotient is the polynomial ring on its free variables"
            }
            PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers {
                factors,
                ..
            } => {
                if factors.len() == 1 {
                    "a power of an irreducible element is primary in the polynomial UFD presented by the graph quotient"
                } else {
                    "powers of a regular sequence generating a prime complete intersection form a primary ideal in the polynomial ring presented by the graph quotient"
                }
            }
            PolynomialPrimaryProof::PrincipalPower { .. } => {
                "a positive power of a prime element generates a primary ideal in a UFD"
            }
            PolynomialPrimaryProof::MonomialPrimary { .. } => {
                "a monomial ideal supported on its coordinate radical and containing a pure power of every radical variable is primary"
            }
            PolynomialPrimaryProof::ZeroDimensionalLocal { .. } => {
                "an Artinian ring with one maximal ideal is local, and its defining ideal is primary"
            }
        }
    }

    /// Complete EHV associated-prime evidence used by the general primary route.
    pub fn ehv_associated_primes(&self) -> Option<&AssociatedPrimeSet<R>> {
        match &self.proof {
            PolynomialPrimaryProof::EhvSingleAssociatedPrime { associated_primes } => {
                Some(associated_primes.as_ref())
            }
            _ => None,
        }
    }

    /// EHV equidimensional-hull evidence used by an isolated primary component.
    pub fn ehv_equidimensional_hull(&self) -> Option<&EquidimensionalHull<R>> {
        match &self.proof {
            PolynomialPrimaryProof::EhvEquidimensionalHull { hull } => Some(hull.as_ref()),
            _ => None,
        }
    }
}

impl<R: Field> fmt::Debug for PolynomialPrimaryCertificate<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PolynomialPrimaryCertificate")
            .field("proof_kind", &self.proof_kind())
            .field("generators", &self.ideal.generators().len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for PolynomialPrimaryCertificate<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PolynomialPrimaryCertificate(proof={}, generators={})",
            self.proof_kind(),
            self.ideal.generators().len()
        )
    }
}

impl<R> SemanticType for PolynomialPrimaryCertificate<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/PolynomialPrimaryCertificate@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &POLYNOMIAL_PRIMARY_CERTIFICATE_DESCRIPTION
    }
}

/// Complete verified primary decomposition of an admitted polynomial ideal.
///
/// Components have distinct associated primes, are in canonical prime order,
/// and intersect exactly to `source`. Embedded associated primes are retained.
#[derive(Clone, PartialEq, Eq)]
pub struct PrimaryDecomposition<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    components: Vec<Ideal<PolynomialRing<R>>>,
    certificates: Vec<PolynomialPrimaryCertificate<R>>,
}

impl<R: Field> PrimaryDecomposition<R> {
    /// Original ideal reconstructed by the primary components.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Canonically ordered primary component ideals.
    pub fn components(&self) -> &[Ideal<PolynomialRing<R>>] {
        &self.components
    }

    /// Complete primary evidence in component order.
    pub fn certificates(&self) -> &[PolynomialPrimaryCertificate<R>] {
        &self.certificates
    }

    /// Associated primes, including embedded primes, in component order.
    pub fn associated_primes(&self) -> Vec<&PolynomialPrimeCertificate<R>> {
        self.certificates
            .iter()
            .map(PolynomialPrimaryCertificate::associated_prime)
            .collect()
    }

    /// Exact completeness theorem for the retained component list.
    pub const fn theorem(&self) -> &'static str {
        "distinct-primary-radical components intersect exactly to the source ideal"
    }
}

impl<R: Field> fmt::Debug for PrimaryDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrimaryDecomposition")
            .field("components", &self.components.len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for PrimaryDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PrimaryDecomposition(components={})",
            self.components.len()
        )
    }
}

impl<R> SemanticType for PrimaryDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/PrimaryDecomposition@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &PRIMARY_DECOMPOSITION_DESCRIPTION
    }
}

fn monomial_generators<R: Field>(source: &Ideal<PolynomialRing<R>>) -> Option<Vec<Vec<Integer>>> {
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

fn single_associated_prime<R>(
    component: &Ideal<PolynomialRing<R>>,
) -> Result<PolynomialPrimeCertificate<R>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let decomposition = construct_minimal_primes(component)?;
    if decomposition.prime_certificates().len() != 1 {
        return Err(DecompositionError::Invalid(
            "a proposed primary component does not have one associated prime".to_owned(),
        ));
    }
    Ok(decomposition.prime_certificates()[0].clone())
}

fn principal_primary_certificates<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Vec<PolynomialPrimaryCertificate<R>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let generator = source.generators().first().ok_or_else(|| {
        DecompositionError::Invalid("a principal primary decomposition has no generator".to_owned())
    })?;
    let factorization = R::factor_polynomial(generator)?;
    factorization.verify()?;
    factorization
        .factors()
        .iter()
        .map(|(factor, exponent)| {
            let component = source.ring().ideal([factor.pow(*exponent)?])?;
            Ok(PolynomialPrimaryCertificate {
                associated_prime: single_associated_prime(&component)?,
                ideal: component,
                proof: PolynomialPrimaryProof::PrincipalPower {
                    factor: factor.clone(),
                    exponent: *exponent,
                },
            })
        })
        .collect()
}

type GeneralizedExponent = Vec<Option<Integer>>;

fn generalized_exponent_is_in_ideal(
    value: &GeneralizedExponent,
    generators: &[Vec<Integer>],
) -> bool {
    generators.iter().any(|generator| {
        generator
            .iter()
            .zip(value)
            .all(|(required, actual)| actual.as_ref().is_none_or(|actual| required <= actual))
    })
}

fn enumerate_generalized_exponents(
    domains: &[Vec<Option<Integer>>],
) -> Result<Vec<GeneralizedExponent>, DecompositionError> {
    let count = domains.iter().try_fold(1_usize, |count, domain| {
        count.checked_mul(domain.len()).ok_or(())
    });
    if count.is_err() || count.is_ok_and(|count| count > MAX_MONOMIAL_IRREDUCIBLE_CANDIDATES) {
        return Err(DecompositionError::ResourceLimit {
            resource: "monomial irreducible-decomposition candidates",
            limit: MAX_MONOMIAL_IRREDUCIBLE_CANDIDATES,
        });
    }
    fn visit(
        domains: &[Vec<Option<Integer>>],
        position: usize,
        current: &mut GeneralizedExponent,
        output: &mut Vec<GeneralizedExponent>,
    ) {
        if position == domains.len() {
            output.push(current.clone());
            return;
        }
        for value in &domains[position] {
            current.push(value.clone());
            visit(domains, position + 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::with_capacity(count.unwrap_or(0));
    visit(domains, 0, &mut Vec::new(), &mut output);
    Ok(output)
}

fn maximal_generalized_standard_monomials(
    generators: &[Vec<Integer>],
    variable_count: usize,
) -> Result<Vec<GeneralizedExponent>, DecompositionError> {
    let mut finite_domains = vec![BTreeSet::new(); variable_count];
    for generator in generators {
        for (variable, exponent) in generator.iter().enumerate() {
            if !exponent.is_zero() {
                finite_domains[variable].insert(exponent - &Integer::one());
            }
        }
    }
    let domains = finite_domains
        .into_iter()
        .map(|values| {
            values
                .into_iter()
                .map(Some)
                .chain(core::iter::once(None))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut maximal = enumerate_generalized_exponents(&domains)?
        .into_iter()
        .filter(|candidate| !generalized_exponent_is_in_ideal(candidate, generators))
        .filter(|candidate| {
            candidate.iter().enumerate().all(|(variable, value)| {
                let Some(value) = value else {
                    return true;
                };
                let mut increased = candidate.clone();
                increased[variable] = Some(value + &Integer::one());
                generalized_exponent_is_in_ideal(&increased, generators)
            })
        })
        .collect::<Vec<_>>();
    maximal.sort();
    Ok(maximal)
}

fn irreducible_monomial_component<R: Field>(
    ring: &PolynomialRing<R>,
    maximal_standard: &GeneralizedExponent,
) -> Result<Ideal<PolynomialRing<R>>, DecompositionError> {
    let mut generators = Vec::new();
    for (variable, bound) in maximal_standard.iter().enumerate() {
        let Some(bound) = bound else {
            continue;
        };
        let mut exponent = vec![Integer::zero(); maximal_standard.len()];
        exponent[variable] = bound + &Integer::one();
        generators.push(ring.from_terms([(exponent, ring.coefficient_ring().one())])?);
    }
    ring.ideal(generators).map_err(Into::into)
}

fn monomial_primary_certificates<R>(
    source: &Ideal<PolynomialRing<R>>,
    exponents: Vec<Vec<Integer>>,
) -> Result<Vec<PolynomialPrimaryCertificate<R>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let monomial = source.ring().monomial_ideal(exponents)?;
    let maximal = maximal_generalized_standard_monomials(
        monomial.minimal_generators(),
        source.ring().variables().len(),
    )?;
    let mut grouped = BTreeMap::<Vec<usize>, Vec<Ideal<PolynomialRing<R>>>>::new();
    for value in maximal {
        let variables = value
            .iter()
            .enumerate()
            .filter_map(|(variable, bound)| bound.is_some().then_some(variable))
            .collect::<Vec<_>>();
        grouped
            .entry(variables)
            .or_default()
            .push(irreducible_monomial_component(source.ring(), &value)?);
    }
    grouped
        .into_iter()
        .map(|(variables, components)| {
            let mut components = components.into_iter();
            let mut component = components.next().ok_or_else(|| {
                DecompositionError::Invalid(
                    "a monomial primary group has no irreducible component".to_owned(),
                )
            })?;
            for next in components {
                component = component.intersection(&next)?;
            }
            let component_exponents = monomial_generators(&component).ok_or_else(|| {
                DecompositionError::Invalid(
                    "an intersection of monomial components ceased to be monomial".to_owned(),
                )
            })?;
            let monomial_component = component.ring().monomial_ideal(component_exponents)?;
            Ok(PolynomialPrimaryCertificate {
                associated_prime: coordinate_prime_certificate(
                    component.ring(),
                    variables.clone(),
                )?,
                ideal: component,
                proof: PolynomialPrimaryProof::MonomialPrimary {
                    source: monomial_component,
                    variables,
                },
            })
        })
        .collect()
}

fn zero_dimensional_primary_certificates<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Vec<PolynomialPrimaryCertificate<R>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let minimal = construct_minimal_primes(source)?;
    let primes = minimal.prime_certificates();
    let mut certificates = Vec::with_capacity(primes.len());
    for (index, prime) in primes.iter().enumerate() {
        let component = if primes.len() == 1 {
            source.clone()
        } else {
            let mut separator = source.ring().one();
            for (other_index, other) in primes.iter().enumerate() {
                if other_index == index {
                    continue;
                }
                let witness = other
                    .ideal()
                    .generators()
                    .iter()
                    .find(|generator| prime.ideal().contains(generator).is_ok_and(|value| !value))
                    .ok_or_else(|| {
                        DecompositionError::Invalid(
                            "distinct zero-dimensional primes have no separator".to_owned(),
                        )
                    })?;
                separator = (&separator * witness)?;
            }
            source.saturate(&source.ring().ideal([separator])?)?
        };
        let quotient_dimension = component.finite_quotient_algebra()?.dimension();
        certificates.push(PolynomialPrimaryCertificate {
            ideal: component,
            associated_prime: prime.clone(),
            proof: PolynomialPrimaryProof::ZeroDimensionalLocal { quotient_dimension },
        });
    }
    Ok(certificates)
}

fn radical_primary_certificates<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<Vec<PolynomialPrimaryCertificate<R>>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let minimal = construct_minimal_primes(source)?;
    if minimal.radical().ideal() != source {
        return Ok(None);
    }
    Ok(Some(
        minimal
            .prime_certificates()
            .iter()
            .cloned()
            .map(|associated_prime| PolynomialPrimaryCertificate {
                ideal: associated_prime.ideal().clone(),
                associated_prime,
                proof: PolynomialPrimaryProof::Prime,
            })
            .collect(),
    ))
}

fn common_factor_primary_certificates<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<Vec<PolynomialPrimaryCertificate<R>>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let factorizations = source
        .generators()
        .iter()
        .map(|generator| {
            let factorization = R::factor_polynomial(generator)?;
            factorization.verify()?;
            Ok(factorization)
        })
        .collect::<Result<Vec<_>, DecompositionError>>()?;
    let common_factors = super::radical::common_irreducible_factors(&factorizations);
    if common_factors.is_empty() || common_factors.iter().all(|(_, exponent)| *exponent == 1) {
        return Ok(None);
    }
    for (factor, exponent) in common_factors.iter().filter(|(_, exponent)| *exponent > 1) {
        let common = factor.pow(*exponent)?;
        let residual_source = source.ring().ideal(
            source
                .generators()
                .iter()
                .map(|generator| generator.exact_div(&common))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        if residual_source == *source {
            continue;
        }
        let residual = match construct_primary_decomposition(&residual_source) {
            Ok(residual) => residual,
            Err(DecompositionError::Unsupported(_)) => continue,
            Err(error) => return Err(error),
        };
        let common_ideal = source.ring().ideal([common])?;
        let mut certificates = principal_primary_certificates(&common_ideal)?;
        certificates.extend(residual.certificates().iter().cloned());
        let mut associated = BTreeSet::new();
        let mut distinct = true;
        for certificate in &certificates {
            let key = certificate
                .associated_prime()
                .canonical_bytes()
                .map_err(|error| DecompositionError::Invalid(error.to_string()))?;
            if !associated.insert(key) {
                distinct = false;
                break;
            }
        }
        if !distinct {
            continue;
        }
        let mut components = certificates.iter().map(PolynomialPrimaryCertificate::ideal);
        let Some(first) = components.next() else {
            continue;
        };
        let reconstructed = components.try_fold(first.clone(), |intersection, component| {
            intersection
                .intersection(component)
                .map_err(DecompositionError::from)
        })?;
        if reconstructed == *source {
            return Ok(Some(certificates));
        }
    }
    Ok(None)
}

fn polynomial_graph_base_and_residual<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<PolynomialGraphResidual<R>>, DecompositionError>
where
    R: Field,
{
    let (base_generators, residual_generators): (Vec<_>, Vec<_>) =
        source.generators().iter().cloned().partition(|generator| {
            let Some((leading_exponent, _)) = generator.terms().next() else {
                return false;
            };
            let mut nonzero = leading_exponent.iter().filter(|value| !value.is_zero());
            nonzero.next().is_some_and(Integer::is_one) && nonzero.next().is_none()
        });
    if residual_generators.is_empty() {
        return Ok(None);
    }
    let base = source.ring().ideal(base_generators)?;
    let Some(pivot_variables) = super::radical::polynomial_graph_pivots(&base) else {
        return Ok(None);
    };
    let residual_generators = residual_generators
        .iter()
        .map(|generator| base.normal_form(generator))
        .collect::<Result<Vec<_>, _>>()?;
    let residual_ideal = source.ring().ideal(residual_generators.clone())?;
    if base.sum(&residual_ideal)? != *source {
        return Ok(None);
    }
    Ok(Some((base, pivot_variables, residual_generators)))
}

fn polynomial_graph_monomial_primary_certificates<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<Vec<PolynomialPrimaryCertificate<R>>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let Some((base, pivot_variables, residual_generators)) =
        polynomial_graph_base_and_residual(source)?
    else {
        return Ok(None);
    };
    let residual_ideal = source.ring().ideal(residual_generators)?;
    let Some(exponents) = monomial_generators(&residual_ideal) else {
        return Ok(None);
    };
    if exponents.iter().any(|exponent| {
        pivot_variables
            .iter()
            .any(|variable| !exponent[*variable].is_zero())
    }) {
        return Ok(None);
    }
    let residual_certificates = monomial_primary_certificates(&residual_ideal, exponents)?;
    let mut certificates = Vec::with_capacity(residual_certificates.len());
    for residual_certificate in residual_certificates {
        let residual_exponents =
            monomial_generators(residual_certificate.ideal()).ok_or_else(|| {
                DecompositionError::Invalid(
                    "a monomial residual primary component ceased to be monomial".to_owned(),
                )
            })?;
        let residual = source.ring().monomial_ideal(residual_exponents)?;
        let variables = residual_certificate
            .associated_prime()
            .coordinate_variables()
            .ok_or_else(|| {
                DecompositionError::Invalid(
                    "a monomial residual primary component has a noncoordinate radical".to_owned(),
                )
            })?
            .to_vec();
        if variables
            .iter()
            .any(|variable| pivot_variables.binary_search(variable).is_ok())
        {
            return Ok(None);
        }
        let component = base.sum(residual_certificate.ideal())?;
        let prime_ideal = base.sum(residual_certificate.associated_prime().ideal())?;
        let associated_prime = single_associated_prime(&prime_ideal)?;
        if associated_prime.ideal() != &prime_ideal {
            return Ok(None);
        }
        certificates.push(PolynomialPrimaryCertificate {
            ideal: component,
            associated_prime,
            proof: PolynomialPrimaryProof::PolynomialGraphMonomialPrimary {
                base: base.clone(),
                pivot_variables: pivot_variables.clone(),
                residual,
                variables,
            },
        });
    }
    let mut components = certificates.iter().map(PolynomialPrimaryCertificate::ideal);
    let Some(first) = components.next() else {
        return Ok(None);
    };
    let reconstructed = components.try_fold(first.clone(), |intersection, component| {
        intersection
            .intersection(component)
            .map_err(DecompositionError::from)
    })?;
    if reconstructed == *source {
        Ok(Some(certificates))
    } else {
        Ok(None)
    }
}

fn polynomial_graph_complete_intersection_primary_certificates<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<Vec<PolynomialPrimaryCertificate<R>>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let Some((base, pivot_variables, residual_generators)) =
        polynomial_graph_base_and_residual(source)?
    else {
        return Ok(None);
    };

    let mut factor_choices = vec![Vec::<(Polynomial<R>, u64)>::new()];
    for residual in residual_generators {
        if residual.is_zero() || residual.is_one() {
            return Ok(None);
        }
        let factorization = R::factor_polynomial(&residual)?;
        factorization.verify()?;
        if factorization.factors().is_empty() {
            return Ok(None);
        }
        let component_count = factor_choices
            .len()
            .checked_mul(factorization.factors().len())
            .ok_or(DecompositionError::ResourceLimit {
                resource: "polynomial-graph complete-intersection primary components",
                limit: MAX_GRAPH_COMPLETE_INTERSECTION_COMPONENTS,
            })?;
        if component_count > MAX_GRAPH_COMPLETE_INTERSECTION_COMPONENTS {
            return Err(DecompositionError::ResourceLimit {
                resource: "polynomial-graph complete-intersection primary components",
                limit: MAX_GRAPH_COMPLETE_INTERSECTION_COMPONENTS,
            });
        }
        let mut extended = Vec::with_capacity(component_count);
        for choice in factor_choices {
            for (factor, exponent) in factorization.factors() {
                let mut next = choice.clone();
                next.push((factor.clone(), *exponent));
                extended.push(next);
            }
        }
        factor_choices = extended;
    }

    let factor_count = factor_choices.first().map_or(0, Vec::len);
    let Ok(factor_count) = i64::try_from(factor_count) else {
        return Ok(None);
    };
    let Some(expected_dimension) = base.dimension()?.checked_sub(factor_count) else {
        return Ok(None);
    };
    let mut certificates = Vec::with_capacity(factor_choices.len());
    for factors in factor_choices {
        let power_ideal = source.ring().ideal(
            factors
                .iter()
                .map(|(factor, exponent)| factor.pow(*exponent))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let component = base.sum(&power_ideal)?;
        let prime_ideal = base.sum(
            &source
                .ring()
                .ideal(factors.iter().map(|(factor, _)| factor.clone()))?,
        )?;
        let minimal = match construct_minimal_primes(&prime_ideal) {
            Ok(minimal) => minimal,
            Err(DecompositionError::Unsupported(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
        if minimal.radical().ideal() != &prime_ideal
            || minimal.prime_certificates().len() != 1
            || minimal.prime_certificates()[0].ideal() != &prime_ideal
            || prime_ideal.dimension()? != expected_dimension
        {
            return Ok(None);
        }
        certificates.push(PolynomialPrimaryCertificate {
            ideal: component,
            associated_prime: minimal.prime_certificates()[0].clone(),
            proof: PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers {
                base: base.clone(),
                pivot_variables: pivot_variables.clone(),
                factors,
            },
        });
    }
    let mut components = certificates.iter().map(PolynomialPrimaryCertificate::ideal);
    let Some(first) = components.next() else {
        return Ok(None);
    };
    let reconstructed = components.try_fold(first.clone(), |intersection, component| {
        intersection
            .intersection(component)
            .map_err(DecompositionError::from)
    })?;
    if reconstructed == *source {
        Ok(Some(certificates))
    } else {
        Ok(None)
    }
}

fn optional_associated_evidence<T>(
    result: Result<T, AssociatedPrimeError>,
) -> Result<Option<T>, DecompositionError> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(AssociatedPrimeError::Groebner(error)) => Err(DecompositionError::Groebner(error)),
        Err(AssociatedPrimeError::Linalg(error)) => Err(DecompositionError::Linalg(error)),
        Err(AssociatedPrimeError::Invalid(reason))
        | Err(AssociatedPrimeError::Decomposition(DecompositionError::Invalid(reason))) => {
            Err(DecompositionError::Invalid(reason))
        }
        // This route is the final bounded correctness floor. An explicitly
        // unsupported or resource-bounded nested decomposition leaves the
        // ideal for a later route rather than weakening the primary claim.
        Err(AssociatedPrimeError::Decomposition(
            DecompositionError::Unsupported(_) | DecompositionError::ResourceLimit { .. },
        )) => Ok(None),
        Err(AssociatedPrimeError::Decomposition(error)) => Err(error),
        Err(error) => Err(DecompositionError::Unsupported(format!(
            "EHV associated-prime construction is unavailable: {error}"
        ))),
    }
}

fn optional_ehv_associated_primes<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<AssociatedPrimeSet<R>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    optional_associated_evidence(construct_associated_primes(source))
}

fn optional_equidimensional_hull<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<EquidimensionalHull<R>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    optional_associated_evidence(construct_equidimensional_hull(source))
}

fn ehv_primary_certificate<R>(
    ideal: &Ideal<PolynomialRing<R>>,
    expected_prime: Option<&Ideal<PolynomialRing<R>>>,
) -> Result<Option<PolynomialPrimaryCertificate<R>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let Some(associated_primes) = optional_ehv_associated_primes(ideal)? else {
        return Ok(None);
    };
    let [associated_prime] = associated_primes.prime_certificates() else {
        return Ok(None);
    };
    if expected_prime.is_some_and(|expected| associated_prime.ideal() != expected) {
        return Ok(None);
    }
    Ok(Some(PolynomialPrimaryCertificate {
        ideal: ideal.clone(),
        associated_prime: associated_prime.clone(),
        proof: PolynomialPrimaryProof::EhvSingleAssociatedPrime {
            associated_primes: Box::new(associated_primes),
        },
    }))
}

fn ideal_is_subset<R>(
    left: &Ideal<PolynomialRing<R>>,
    right: &Ideal<PolynomialRing<R>>,
) -> Result<bool, DecompositionError>
where
    R: Field,
{
    if left.ring() != right.ring() {
        return Ok(false);
    }
    for generator in left.generators() {
        if !right.contains(generator)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Contract the source from its localization at one certified associated prime.
///
/// For every associated prime not contained in the target, choose one retained
/// generator outside the target and multiply those choices. The product is
/// outside the target because the target is prime, but belongs to every
/// component that localization must remove. Saturation by this separator
/// therefore retains exactly the primary layers whose associated primes are
/// contained in the target.
fn associated_prime_localization_contraction<R>(
    source: &Ideal<PolynomialRing<R>>,
    primes: &[PolynomialPrimeCertificate<R>],
    target_index: usize,
) -> Result<Ideal<PolynomialRing<R>>, DecompositionError>
where
    R: Field,
{
    let target = primes[target_index].ideal();
    let mut separator = source.ring().one();
    let mut has_factor = false;
    for (other_index, other) in primes.iter().enumerate() {
        if other_index == target_index || ideal_is_subset(other.ideal(), target)? {
            continue;
        }
        let mut factor = None;
        for generator in other.ideal().generators() {
            if !target.contains(generator)? {
                factor = Some(generator);
                break;
            }
        }
        let factor = factor.ok_or_else(|| {
            DecompositionError::Invalid(
                "associated-prime inclusion disagrees with localization separators".to_owned(),
            )
        })?;
        separator = (&separator * factor)?;
        has_factor = true;
    }
    if !has_factor {
        return Ok(source.clone());
    }
    source
        .saturate(&source.ring().ideal([separator])?)
        .map_err(Into::into)
}

fn intersect_primary_certificates<R>(
    certificates: &[PolynomialPrimaryCertificate<R>],
) -> Result<Option<Ideal<PolynomialRing<R>>>, DecompositionError>
where
    R: Field,
{
    let Some(first) = certificates.first() else {
        return Ok(None);
    };
    let mut intersection = first.ideal().clone();
    for certificate in &certificates[1..] {
        intersection = intersection.intersection(certificate.ideal())?;
    }
    Ok(Some(intersection))
}

fn search_embedded_primary_components<R>(
    source: &Ideal<PolynomialRing<R>>,
    choices: &[Vec<PolynomialPrimaryCertificate<R>>],
    index: usize,
    intersection: &Ideal<PolynomialRing<R>>,
    selected: &mut Vec<PolynomialPrimaryCertificate<R>>,
) -> Result<Option<Vec<PolynomialPrimaryCertificate<R>>>, DecompositionError>
where
    R: Field,
{
    if index == choices.len() {
        return Ok((intersection == source).then(|| selected.clone()));
    }
    for certificate in &choices[index] {
        let next_intersection = intersection.intersection(certificate.ideal())?;
        selected.push(certificate.clone());
        if let Some(result) = search_embedded_primary_components(
            source,
            choices,
            index + 1,
            &next_intersection,
            selected,
        )? {
            return Ok(Some(result));
        }
        selected.pop();
    }
    Ok(None)
}

/// General bounded primary discovery driven by complete EHV associated primes.
///
/// Minimal components are isolated by separators and saturation. For every
/// embedded prime `P`, saturation first contracts `I R_P` by removing all
/// associated-prime layers not contained in `P`; the deterministic family
/// `(I R_P intersect R) + P^m` is then searched up to a fixed exponent. No
/// candidate is accepted merely from its construction: its own complete
/// associated-prime computation must prove it primary, and the final
/// intersection must reproduce `I` exactly.
fn ehv_primary_certificates<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<Option<Vec<PolynomialPrimaryCertificate<R>>>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let Some(source_associated_primes) = optional_ehv_associated_primes(source)? else {
        return Ok(None);
    };
    let primes = source_associated_primes.prime_certificates();
    if primes.is_empty() {
        return Ok(None);
    }
    if primes.len() == 1 {
        return Ok(Some(vec![PolynomialPrimaryCertificate {
            ideal: source.clone(),
            associated_prime: primes[0].clone(),
            proof: PolynomialPrimaryProof::EhvSingleAssociatedPrime {
                associated_primes: Box::new(source_associated_primes),
            },
        }]));
    }

    let mut minimal_indices = Vec::new();
    for (index, prime) in primes.iter().enumerate() {
        let mut minimal = true;
        for (other_index, other) in primes.iter().enumerate() {
            if index != other_index
                && other.ideal() != prime.ideal()
                && ideal_is_subset(other.ideal(), prime.ideal())?
            {
                minimal = false;
                break;
            }
        }
        if minimal {
            minimal_indices.push(index);
        }
    }
    if minimal_indices.is_empty() {
        return Err(DecompositionError::Invalid(
            "a nonempty associated-prime set has no minimal member".to_owned(),
        ));
    }

    let mut isolated = Vec::with_capacity(minimal_indices.len());
    for &index in &minimal_indices {
        let prime = primes[index].ideal();
        let component = associated_prime_localization_contraction(source, primes, index)?;
        let Some(certificate) = ehv_primary_certificate(&component, Some(prime))? else {
            return Ok(None);
        };
        isolated.push(certificate);
    }

    let minimal_index_set = minimal_indices.into_iter().collect::<BTreeSet<_>>();
    let mut embedded_choices = Vec::new();
    let mut combination_count = 1usize;
    for (index, prime) in primes.iter().enumerate() {
        if minimal_index_set.contains(&index) {
            continue;
        }
        let localized_source = associated_prime_localization_contraction(source, primes, index)?;
        let mut choices = Vec::new();
        let mut previous = None;
        for exponent in 1..=MAX_EHV_EMBEDDED_EXPONENT {
            let candidate = localized_source.sum(&prime.ideal().pow(exponent)?)?;
            if previous.as_ref() == Some(&candidate) {
                break;
            }
            previous = Some(candidate.clone());
            let Some(hull) = optional_equidimensional_hull(&candidate)? else {
                continue;
            };
            let [associated_prime] = hull.prime_certificates() else {
                continue;
            };
            if associated_prime.ideal() != prime.ideal()
                || choices
                    .iter()
                    .any(|choice: &PolynomialPrimaryCertificate<R>| choice.ideal() == hull.ideal())
            {
                continue;
            }
            choices.push(PolynomialPrimaryCertificate {
                ideal: hull.ideal().clone(),
                associated_prime: associated_prime.clone(),
                proof: PolynomialPrimaryProof::EhvEquidimensionalHull {
                    hull: Box::new(hull),
                },
            });
        }
        if choices.is_empty() {
            return Ok(None);
        }
        combination_count = combination_count.checked_mul(choices.len()).ok_or(
            DecompositionError::ResourceLimit {
                resource: "EHV embedded-primary candidate combinations",
                limit: MAX_EHV_PRIMARY_COMBINATIONS,
            },
        )?;
        if combination_count > MAX_EHV_PRIMARY_COMBINATIONS {
            return Err(DecompositionError::ResourceLimit {
                resource: "EHV embedded-primary candidate combinations",
                limit: MAX_EHV_PRIMARY_COMBINATIONS,
            });
        }
        embedded_choices.push(choices);
    }

    let Some(isolated_intersection) = intersect_primary_certificates(&isolated)? else {
        return Ok(None);
    };
    if embedded_choices.is_empty() {
        return Ok((isolated_intersection == *source).then_some(isolated));
    }
    let mut selected = Vec::with_capacity(embedded_choices.len());
    let Some(mut embedded) = search_embedded_primary_components(
        source,
        &embedded_choices,
        0,
        &isolated_intersection,
        &mut selected,
    )?
    else {
        return Ok(None);
    };
    isolated.append(&mut embedded);
    Ok(Some(isolated))
}

fn sorted_primary_certificates<R>(
    certificates: Vec<PolynomialPrimaryCertificate<R>>,
) -> Result<Vec<PolynomialPrimaryCertificate<R>>, DecompositionError>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let mut keyed = certificates
        .into_iter()
        .map(|certificate| {
            certificate
                .associated_prime()
                .canonical_bytes()
                .map(|key| (key, certificate))
                .map_err(|error| DecompositionError::Invalid(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    if keyed.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(DecompositionError::Invalid(
            "a primary decomposition contains two components with the same radical".to_owned(),
        ));
    }
    Ok(keyed
        .into_iter()
        .map(|(_, certificate)| certificate)
        .collect())
}

fn certify_primary_ideal<R>(
    ideal: &Ideal<PolynomialRing<R>>,
) -> Result<PolynomialPrimaryCertificate<R>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    if let Some(certificate) = ehv_primary_certificate(ideal, None)? {
        return Ok(certificate);
    }
    let decomposition = construct_primary_decomposition(ideal)?;
    let [certificate] = decomposition.certificates() else {
        return Err(DecompositionError::Invalid(
            "the proposed ideal has more than one associated prime and is not primary".to_owned(),
        ));
    };
    if certificate.ideal() != ideal {
        return Err(DecompositionError::Invalid(
            "the proposed primary certificate does not certify the input ideal".to_owned(),
        ));
    }
    Ok(certificate.clone())
}

fn admit_primary_decomposition<R>(
    source: &Ideal<PolynomialRing<R>>,
    components: &[Ideal<PolynomialRing<R>>],
) -> Result<PrimaryDecomposition<R>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let mut certificates = Vec::with_capacity(components.len());
    for component in components {
        if component.ring() != source.ring() {
            return Err(DecompositionError::Invalid(
                "a proposed primary component belongs to a different polynomial ring".to_owned(),
            ));
        }
        certificates.push(certify_primary_ideal(component)?);
    }
    let certificates = sorted_primary_certificates(certificates)?;
    let components = certificates
        .iter()
        .map(|certificate| certificate.ideal().clone())
        .collect::<Vec<_>>();
    let result = PrimaryDecomposition {
        source: source.clone(),
        components,
        certificates,
    };
    verify_primary_decomposition(&result)?;
    Ok(result)
}

pub(super) fn construct_primary_decomposition<R>(
    source: &Ideal<PolynomialRing<R>>,
) -> Result<PrimaryDecomposition<R>, DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let certificates = if let Some(exponents) = monomial_generators(source) {
        monomial_primary_certificates(source, exponents)?
    } else if source.generators().len() == 1 {
        principal_primary_certificates(source)?
    } else if source.dimension()? == 0 {
        zero_dimensional_primary_certificates(source)?
    } else if let Some(certificates) = common_factor_primary_certificates(source)? {
        certificates
    } else if let Some(certificates) = radical_primary_certificates(source)? {
        certificates
    } else if let Some(certificates) = polynomial_graph_monomial_primary_certificates(source)? {
        certificates
    } else if let Some(certificates) =
        polynomial_graph_complete_intersection_primary_certificates(source)?
    {
        certificates
    } else if let Some(certificates) = ehv_primary_certificates(source)? {
        certificates
    } else {
        return Err(DecompositionError::Unsupported(
            "general positive-dimensional primary decomposition is not yet in the native floor"
                .to_owned(),
        ));
    };
    let certificates = sorted_primary_certificates(certificates)?;
    let components = certificates
        .iter()
        .map(|certificate| certificate.ideal().clone())
        .collect();
    Ok(PrimaryDecomposition {
        source: source.clone(),
        components,
        certificates,
    })
}

pub(super) fn verify_polynomial_primary_certificate<R>(
    certificate: &PolynomialPrimaryCertificate<R>,
) -> Result<(), DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    certificate.ideal.verify()?;
    verify_polynomial_prime_certificate(&certificate.associated_prime)?;
    let prime = certificate.associated_prime.ideal();
    if prime.ring() != certificate.ideal.ring() {
        return Err(DecompositionError::Invalid(
            "a primary certificate is not contained in its associated prime".to_owned(),
        ));
    }
    for generator in certificate.ideal.generators() {
        if !prime.contains(generator)? {
            return Err(DecompositionError::Invalid(
                "a primary certificate is not contained in its associated prime".to_owned(),
            ));
        }
    }
    match &certificate.proof {
        PolynomialPrimaryProof::Prime => {
            if certificate.ideal != *prime {
                return Err(DecompositionError::Invalid(
                    "a prime-primary certificate does not equal its associated prime".to_owned(),
                ));
            }
        }
        PolynomialPrimaryProof::EhvSingleAssociatedPrime { associated_primes } => {
            verify_associated_primes(associated_primes)
                .map_err(|error| DecompositionError::Invalid(error.to_string()))?;
            if associated_primes.source() != &certificate.ideal
                || associated_primes.prime_certificates().len() != 1
                || associated_primes.prime_certificates()[0] != certificate.associated_prime
            {
                return Err(DecompositionError::Invalid(
                    "an EHV primary certificate does not retain exactly its source and associated prime"
                        .to_owned(),
                ));
            }
        }
        PolynomialPrimaryProof::EhvEquidimensionalHull { hull } => {
            verify_equidimensional_hull(hull)
                .map_err(|error| DecompositionError::Invalid(error.to_string()))?;
            if hull.ideal() != &certificate.ideal
                || hull.prime_certificates().len() != 1
                || hull.prime_certificates()[0] != certificate.associated_prime
            {
                return Err(DecompositionError::Invalid(
                    "an EHV hull-primary certificate does not retain exactly one minimal prime"
                        .to_owned(),
                ));
            }
        }
        PolynomialPrimaryProof::PolynomialGraphMonomialPrimary {
            base,
            pivot_variables,
            residual,
            variables,
        } => {
            if base.ring() != certificate.ideal.ring()
                || residual.ring() != certificate.ideal.ring()
                || super::radical::polynomial_graph_pivots(base).as_ref() != Some(pivot_variables)
                || variables.windows(2).any(|pair| pair[0] >= pair[1])
                || variables
                    .iter()
                    .any(|variable| pivot_variables.binary_search(variable).is_ok())
                || residual.minimal_generators().iter().any(|exponent| {
                    pivot_variables
                        .iter()
                        .any(|variable| !exponent[*variable].is_zero())
                })
            {
                return Err(DecompositionError::Invalid(
                    "a polynomial-graph monomial-primary certificate has invalid base or residual data"
                        .to_owned(),
                ));
            }
            let residual_ideal = residual.to_ideal()?;
            let residual_certificates = monomial_primary_certificates(
                &residual_ideal,
                residual.minimal_generators().to_vec(),
            )?;
            if residual_certificates.len() != 1
                || residual_certificates[0].ideal() != &residual_ideal
                || residual_certificates[0]
                    .associated_prime()
                    .coordinate_variables()
                    != Some(variables)
                || base.sum(&residual_ideal)? != certificate.ideal
                || base.sum(residual_certificates[0].associated_prime().ideal())? != *prime
            {
                return Err(DecompositionError::Invalid(
                    "a polynomial-graph monomial-primary certificate is not the inverse image of its retained monomial primary ideal"
                        .to_owned(),
                ));
            }
        }
        PolynomialPrimaryProof::PolynomialGraphCompleteIntersectionPowers {
            base,
            pivot_variables,
            factors,
        } => {
            if base.ring() != certificate.ideal.ring()
                || factors.is_empty()
                || super::radical::polynomial_graph_pivots(base).as_ref() != Some(pivot_variables)
            {
                return Err(DecompositionError::Invalid(
                    "a polynomial-graph complete-intersection-primary certificate has invalid base data".to_owned(),
                ));
            }
            let mut power_equations = Vec::with_capacity(factors.len());
            let mut prime_equations = Vec::with_capacity(factors.len());
            for (factor, exponent) in factors {
                if factor.ring() != certificate.ideal.ring()
                    || *exponent == 0
                    || base.normal_form(factor)? != *factor
                {
                    return Err(DecompositionError::Invalid(
                        "a polynomial-graph complete-intersection-primary certificate has invalid factor data".to_owned(),
                    ));
                }
                let factorization = R::factor_polynomial(factor)?;
                factorization.verify()?;
                if factorization.factors() != [(factor.clone(), 1)] {
                    return Err(DecompositionError::Invalid(
                        "a polynomial-graph complete-intersection-primary certificate has a reducible retained factor".to_owned(),
                    ));
                }
                power_equations.push(factor.pow(*exponent)?);
                prime_equations.push(factor.clone());
            }
            let power_ideal = base.sum(&base.ring().ideal(power_equations)?)?;
            let prime_ideal = base.sum(&base.ring().ideal(prime_equations)?)?;
            let factor_count = i64::try_from(factors.len()).map_err(|_| {
                DecompositionError::Invalid(
                    "a polynomial-graph complete intersection has too many factors".to_owned(),
                )
            })?;
            if base.dimension()?.checked_sub(factor_count) != Some(prime.dimension()?)
                || power_ideal != certificate.ideal
                || prime_ideal != *prime
            {
                return Err(DecompositionError::Invalid(
                    "a polynomial-graph complete-intersection-primary certificate is not a prime-height regular-sequence power over its graph base".to_owned(),
                ));
            }
        }
        PolynomialPrimaryProof::PrincipalPower { factor, exponent } => {
            if *exponent == 0 || factor.ring() != certificate.ideal.ring() {
                return Err(DecompositionError::Invalid(
                    "a principal-primary certificate has invalid factor data".to_owned(),
                ));
            }
            let factorization = R::factor_polynomial(factor)?;
            factorization.verify()?;
            if factorization.factors() != [(factor.clone(), 1)]
                || certificate.ideal != factor.ring().ideal([factor.pow(*exponent)?])?
                || prime != &factor.ring().ideal([factor.clone()])?
            {
                return Err(DecompositionError::Invalid(
                    "a principal-primary certificate is not an irreducible power".to_owned(),
                ));
            }
        }
        PolynomialPrimaryProof::MonomialPrimary { source, variables } => {
            if source.to_ideal()? != certificate.ideal
                || certificate.associated_prime.coordinate_variables() != Some(variables)
                || variables.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(DecompositionError::Invalid(
                    "a monomial-primary certificate has inconsistent source or radical".to_owned(),
                ));
            }
            for generator in source.minimal_generators() {
                if generator.iter().enumerate().any(|(variable, exponent)| {
                    !exponent.is_zero() && variables.binary_search(&variable).is_err()
                }) {
                    return Err(DecompositionError::Invalid(
                        "a monomial primary generator uses a variable outside its radical"
                            .to_owned(),
                    ));
                }
            }
            for variable in variables {
                if !source.minimal_generators().iter().any(|generator| {
                    !generator[*variable].is_zero()
                        && generator
                            .iter()
                            .enumerate()
                            .all(|(position, exponent)| position == *variable || exponent.is_zero())
                }) {
                    return Err(DecompositionError::Invalid(
                        "a monomial primary ideal omits a pure power of a radical variable"
                            .to_owned(),
                    ));
                }
            }
        }
        PolynomialPrimaryProof::ZeroDimensionalLocal { quotient_dimension } => {
            if certificate.ideal.dimension()? != 0
                || *quotient_dimension == 0
                || certificate.ideal.finite_quotient_algebra()?.dimension() != *quotient_dimension
                || prime.dimension()? != 0
            {
                return Err(DecompositionError::Invalid(
                    "a zero-dimensional primary certificate has inconsistent dimension data"
                        .to_owned(),
                ));
            }
            let exponent = u64::try_from(*quotient_dimension).map_err(|_| {
                DecompositionError::Unsupported(
                    "a zero-dimensional nilpotence bound does not fit u64".to_owned(),
                )
            })?;
            let prime_power = prime.pow(exponent)?;
            for generator in prime_power.generators() {
                if !certificate.ideal.contains(generator)? {
                    return Err(DecompositionError::Invalid(
                        "the associated maximal ideal is not nilpotent modulo its primary component"
                            .to_owned(),
                    ));
                }
            }
        }
    }
    // The graph-monomial arm has already reconstructed both inverse images and
    // proved the residual coordinate radical. Re-running general minimal-prime
    // discovery here would replace that direct proof with an unrelated,
    // potentially exponential factor split.
    if !matches!(
        &certificate.proof,
        PolynomialPrimaryProof::PolynomialGraphMonomialPrimary { .. }
            | PolynomialPrimaryProof::EhvSingleAssociatedPrime { .. }
            | PolynomialPrimaryProof::EhvEquidimensionalHull { .. }
    ) {
        let minimal = construct_minimal_primes(&certificate.ideal)?;
        if minimal.prime_certificates().len() != 1
            || minimal.prime_certificates()[0].ideal() != certificate.associated_prime.ideal()
        {
            return Err(DecompositionError::Invalid(
                "a primary certificate does not have exactly its recorded associated prime"
                    .to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn verify_primary_decomposition<R>(
    value: &PrimaryDecomposition<R>,
) -> Result<(), DecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    value.source.verify()?;
    if value.components.len() != value.certificates.len()
        || sorted_primary_certificates(value.certificates.clone())? != value.certificates
    {
        return Err(DecompositionError::Invalid(
            "primary components and certificates are not in canonical correspondence".to_owned(),
        ));
    }
    for (component, certificate) in value.components.iter().zip(&value.certificates) {
        if component != certificate.ideal() || component.ring() != value.source.ring() {
            return Err(DecompositionError::Invalid(
                "a primary certificate does not certify its recorded component".to_owned(),
            ));
        }
        verify_polynomial_primary_certificate(certificate)?;
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
    if reconstructed != value.source {
        return Err(DecompositionError::Invalid(
            "primary components do not reconstruct the source ideal".to_owned(),
        ));
    }
    for omitted in 0..value.components.len() {
        let mut retained = value
            .components
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != omitted)
            .map(|(_, component)| component);
        let without_omitted = if let Some(first) = retained.next() {
            let mut value = first.clone();
            for component in retained {
                value = value.intersection(component)?;
            }
            value
        } else {
            value.source.ring().ideal([value.source.ring().one()])?
        };
        if without_omitted == value.source {
            return Err(DecompositionError::Invalid(
                "the proposed primary decomposition contains a redundant component".to_owned(),
            ));
        }
    }
    Ok(())
}

impl Ideal<PolynomialRing<RationalField>> {
    /// Independently prove that this rational polynomial ideal is primary.
    pub fn certify_primary(
        &self,
    ) -> Result<PolynomialPrimaryCertificate<RationalField>, DecompositionError> {
        certify_primary_ideal(self)
    }

    /// Admit independently discovered primary components after exact verification.
    pub fn admit_primary_decomposition(
        &self,
        components: &[Self],
    ) -> Result<PrimaryDecomposition<RationalField>, DecompositionError> {
        admit_primary_decomposition(self, components)
    }

    /// Compute a complete verified primary decomposition in the admitted rational regimes.
    pub fn primary_decomposition(
        &self,
    ) -> Result<PrimaryDecomposition<RationalField>, DecompositionError> {
        construct_primary_decomposition(self)
    }
}

impl Ideal<PolynomialRing<FiniteField>> {
    /// Independently prove that this finite-field polynomial ideal is primary.
    pub fn certify_primary(
        &self,
    ) -> Result<PolynomialPrimaryCertificate<FiniteField>, DecompositionError> {
        certify_primary_ideal(self)
    }

    /// Admit independently discovered primary components after exact verification.
    pub fn admit_primary_decomposition(
        &self,
        components: &[Self],
    ) -> Result<PrimaryDecomposition<FiniteField>, DecompositionError> {
        admit_primary_decomposition(self, components)
    }

    /// Compute a complete verified primary decomposition in the admitted finite-field regimes.
    pub fn primary_decomposition(
        &self,
    ) -> Result<PrimaryDecomposition<FiniteField>, DecompositionError> {
        construct_primary_decomposition(self)
    }
}

macro_rules! impl_primary_verification {
    ($coefficient:ty) => {
        impl PolynomialPrimaryCertificate<$coefficient> {
            /// Independently recheck the complete primary evidence.
            pub fn verify(&self) -> Result<(), DecompositionError> {
                verify_polynomial_primary_certificate(self)
            }

            /// Deserialize and exactly reverify primary evidence.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl PrimaryDecomposition<$coefficient> {
            /// Independently recheck every primary claim and the exact intersection.
            pub fn verify(&self) -> Result<(), DecompositionError> {
                verify_primary_decomposition(self)
            }

            /// Deserialize and exactly reverify a complete primary decomposition.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
    };
}

impl_primary_verification!(RationalField);
impl_primary_verification!(FiniteField);

impl<R> PolynomialPrimaryCertificate<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the primary ideal, associated prime, and complete proof evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for PolynomialPrimaryCertificate<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PolynomialPrimaryCertificate", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ideal", &self.ideal)?;
        state.serialize_field("associated_prime", &self.associated_prime)?;
        state.serialize_field("proof", &self.proof)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, PolynomialPrimeCertificate<R>: Deserialize<'de>, AssociatedPrimeSet<R>: Deserialize<'de>, EquidimensionalHull<R>: Deserialize<'de>"
    )
)]
struct PolynomialPrimaryCertificateWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ideal: Ideal<PolynomialRing<R>>,
    associated_prime: PolynomialPrimeCertificate<R>,
    proof: PolynomialPrimaryProof<R>,
}

fn deserialize_polynomial_primary_certificate<'de, R, D>(
    deserializer: D,
) -> Result<PolynomialPrimaryCertificate<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    PolynomialPrimeCertificate<R>: Deserialize<'de>,
    AssociatedPrimeSet<R>: Deserialize<'de>,
    EquidimensionalHull<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = PolynomialPrimaryCertificateWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/PolynomialPrimaryCertificate@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/PolynomialPrimaryCertificate@1", wire.type_id
        )));
    }
    let result = PolynomialPrimaryCertificate {
        ideal: wire.ideal,
        associated_prime: wire.associated_prime,
        proof: wire.proof,
    };
    verify_polynomial_primary_certificate(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for PolynomialPrimaryCertificate<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_polynomial_primary_certificate(deserializer)
    }
}

impl<'de> Deserialize<'de> for PolynomialPrimaryCertificate<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_polynomial_primary_certificate(deserializer)
    }
}

impl<R> PrimaryDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    /// Serialize the source and complete primary evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for PrimaryDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PrimaryDecomposition", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("components", &self.components)?;
        state.serialize_field("certificates", &self.certificates)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, PolynomialPrimaryCertificate<R>: Deserialize<'de>"
    )
)]
struct PrimaryDecompositionWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    components: Vec<Ideal<PolynomialRing<R>>>,
    certificates: Vec<PolynomialPrimaryCertificate<R>>,
}

fn deserialize_primary_decomposition<'de, R, D>(
    deserializer: D,
) -> Result<PrimaryDecomposition<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    PolynomialPrimaryCertificate<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = PrimaryDecompositionWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/commalg/PrimaryDecomposition@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            "noether/commalg/PrimaryDecomposition@1", wire.type_id
        )));
    }
    let result = PrimaryDecomposition {
        source: wire.source,
        components: wire.components,
        certificates: wire.certificates,
    };
    verify_primary_decomposition(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for PrimaryDecomposition<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_primary_decomposition(deserializer)
    }
}

impl<'de> Deserialize<'de> for PrimaryDecomposition<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_primary_decomposition(deserializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::{SemanticType, Q};

    #[test]
    fn principal_primary_components_retain_factor_multiplicities() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let first = (&variables[0] + &variables[1]).unwrap();
        let second = (&variables[0] - &variables[2]).unwrap();
        let source = ring
            .ideal([(&first.pow(3).unwrap() * &second.pow(2).unwrap()).unwrap()])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        assert!(decomposition
            .certificates()
            .iter()
            .all(|certificate| certificate.proof_kind() == "principal_power"));
        assert_eq!(
            PrimaryDecomposition::<RationalField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );
        assert_eq!(
            PrimaryDecomposition::<RationalField>::TYPE_ID,
            "noether/commalg/PrimaryDecomposition@1"
        );
    }

    #[test]
    fn monomial_primary_decomposition_includes_an_embedded_prime() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&variables[0].pow(3).unwrap() * &variables[1]).unwrap(),
                (&variables[0] * &variables[2].pow(4).unwrap()).unwrap(),
            ])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 3);
        let radicals = decomposition
            .associated_primes()
            .into_iter()
            .map(|prime| prime.coordinate_variables().unwrap().to_vec())
            .collect::<Vec<_>>();
        assert!(radicals.contains(&vec![0]));
        assert!(radicals.contains(&vec![0, 2]));
        assert!(radicals.contains(&vec![1, 2]));
        let minimal = source.minimal_primes().unwrap();
        assert_eq!(minimal.components().len(), 2);
    }

    #[test]
    fn zero_dimensional_monomial_primary_uses_its_coordinate_radical() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([variables[0].pow(4).unwrap(), variables[1].pow(3).unwrap()])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components(), [source]);
        assert_eq!(
            decomposition.associated_primes()[0].coordinate_variables(),
            Some([0, 1].as_slice())
        );
    }

    #[test]
    fn zero_dimensional_separators_isolate_local_primary_components() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let one = ring.constant(1.into()).unwrap();
        let two = ring.constant(2.into()).unwrap();
        let left = (&variables[0] - &one).unwrap();
        let right = (&variables[0] + &two).unwrap();
        let source = ring
            .ideal([
                (&left.pow(3).unwrap() * &right.pow(2).unwrap()).unwrap(),
                variables[1].pow(4).unwrap(),
            ])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        assert!(decomposition.certificates().iter().all(|certificate| {
            certificate.proof_kind() == "zero_dimensional_local"
                && certificate.nilpotence_bound().is_some()
        }));
    }

    #[test]
    fn finite_nonmonogenic_reduced_algebra_has_four_prime_components() {
        let field = FiniteField::prime(2_u64).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([
                (&variables[0].pow(2).unwrap() + &variables[0]).unwrap(),
                (&variables[1].pow(2).unwrap() + &variables[1]).unwrap(),
            ])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();
        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 4);
        assert!(decomposition
            .certificates()
            .iter()
            .all(|certificate| certificate.nilpotence_bound() == Some(1)));
    }

    #[test]
    fn positive_dimensional_radical_components_are_prime_primary() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let left = (&variables[0] + &ring.one()).unwrap();
        let right = (&variables[1] + &ring.one()).unwrap();
        let source = ring
            .ideal([variables[2].clone(), (&left * &right).unwrap()])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        assert!(decomposition.certificates().iter().all(|certificate| {
            certificate.proof_kind() == "prime"
                && certificate.ideal() == certificate.associated_prime().ideal()
                && certificate.associated_prime().proof_kind() == "affine_linear"
        }));
        assert_eq!(
            decomposition.components()[0]
                .intersection(&decomposition.components()[1])
                .unwrap(),
            source
        );
        assert_eq!(
            PrimaryDecomposition::<RationalField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );

        let mut payload = serde_json::to_value(&decomposition.certificates()[0]).unwrap();
        payload["proof"]["method"] = serde_json::json!("zero_dimensional_local");
        payload["proof"]["quotient_dimension"] = serde_json::json!(1);
        assert!(
            serde_json::from_value::<PolynomialPrimaryCertificate<RationalField>>(payload).is_err()
        );
    }

    #[test]
    fn polynomial_graph_hypersurface_retains_factor_multiplicities() {
        let ring = PolynomialRing::new(Q, ["z", "x", "y"], MonomialOrder::Lexicographic).unwrap();
        let variables = ring.generators().unwrap();
        let graph = (&variables[0] - &variables[1].pow(2).unwrap()).unwrap();
        let left = (&variables[1] + &ring.one()).unwrap();
        let right = (&variables[2] + &ring.one()).unwrap();
        let source = ring
            .ideal([
                graph.clone(),
                (&left.pow(2).unwrap() * &right.pow(3).unwrap()).unwrap(),
            ])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        assert!(decomposition.certificates().iter().all(|certificate| {
            certificate.proof_kind() == "polynomial_graph_hypersurface_power"
                && certificate.polynomial_graph_base()
                    == Some(&ring.ideal([graph.clone()]).unwrap())
                && certificate.polynomial_graph_pivots() == Some(&[0][..])
                && certificate.polynomial_graph_factor().is_some()
                && matches!(
                    certificate.associated_prime().proof_kind(),
                    "affine_linear" | "polynomial_graph"
                )
        }));
        let mut exponents = decomposition
            .certificates()
            .iter()
            .map(|certificate| certificate.polynomial_graph_factor_exponent().unwrap())
            .collect::<Vec<_>>();
        exponents.sort_unstable();
        assert_eq!(exponents, vec![2, 3]);
        assert_eq!(
            decomposition.components()[0]
                .intersection(&decomposition.components()[1])
                .unwrap(),
            source
        );
        assert_eq!(
            PrimaryDecomposition::<RationalField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );

        let mut payload = serde_json::to_value(&decomposition.certificates()[0]).unwrap();
        payload["proof"]["factors"][0][1] = serde_json::json!(1);
        assert!(
            serde_json::from_value::<PolynomialPrimaryCertificate<RationalField>>(payload).is_err()
        );
    }

    #[test]
    fn polynomial_graph_complete_intersection_splits_independent_factor_families() {
        let ring = PolynomialRing::new(
            Q,
            ["z", "x", "y", "u", "v", "t"],
            MonomialOrder::Lexicographic,
        )
        .unwrap();
        let variables = ring.generators().unwrap();
        let graph = (&variables[0] - &variables[5].pow(2).unwrap()).unwrap();
        let first_left = (&variables[1] + &ring.one()).unwrap();
        let first_right = (&variables[2] + &ring.one()).unwrap();
        let second_left = (&variables[3] + &ring.one()).unwrap();
        let second_right = (&variables[4] + &ring.one()).unwrap();
        let first = (&first_left.pow(2).unwrap() * &first_right.pow(3).unwrap()).unwrap();
        let second = (&second_left.pow(4).unwrap() * &second_right.pow(5).unwrap()).unwrap();
        let source = ring
            .ideal([graph.clone(), first.clone(), second.clone()])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 4);
        assert!(decomposition.certificates().iter().all(|certificate| {
            certificate.proof_kind() == "polynomial_graph_complete_intersection_powers"
                && certificate.polynomial_graph_base()
                    == Some(&ring.ideal([graph.clone()]).unwrap())
                && certificate.polynomial_graph_pivots() == Some(&[0][..])
                && certificate.polynomial_graph_factor().is_none()
                && certificate.polynomial_graph_factor_exponent().is_none()
                && certificate
                    .polynomial_graph_factors()
                    .is_some_and(|factors| factors.len() == 2)
                && certificate.associated_prime().proof_kind() == "polynomial_graph"
        }));
        let mut factor_sets = decomposition
            .certificates()
            .iter()
            .map(|certificate| {
                let factors = certificate.polynomial_graph_factors().unwrap();
                assert!(factors.iter().any(|(factor, exponent)| {
                    (factor == &first_left && *exponent == 2)
                        || (factor == &first_right && *exponent == 3)
                }));
                assert!(factors.iter().any(|(factor, exponent)| {
                    (factor == &second_left && *exponent == 4)
                        || (factor == &second_right && *exponent == 5)
                }));
                let mut exponents = factors
                    .iter()
                    .map(|(_, exponent)| *exponent)
                    .collect::<Vec<_>>();
                exponents.sort_unstable();
                exponents
            })
            .collect::<Vec<_>>();
        factor_sets.sort();
        assert_eq!(
            factor_sets,
            vec![vec![2, 4], vec![2, 5], vec![3, 4], vec![3, 5]]
        );
        let mut components = decomposition.components().iter();
        let reconstructed = components.next().unwrap().clone();
        let reconstructed = components.fold(reconstructed, |intersection, component| {
            intersection.intersection(component).unwrap()
        });
        assert_eq!(reconstructed, source);
        assert_eq!(
            PrimaryDecomposition::<RationalField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );

        let mut payload = serde_json::to_value(&decomposition.certificates()[0]).unwrap();
        payload["proof"]["factors"][0][1] = serde_json::json!(0);
        assert!(
            serde_json::from_value::<PolynomialPrimaryCertificate<RationalField>>(payload).is_err()
        );
    }

    #[test]
    fn polynomial_graph_lifts_monomial_embedded_primary_components() {
        let ring = PolynomialRing::new(Q, ["y", "z", "x", "u", "v"], MonomialOrder::Lexicographic)
            .unwrap();
        let variables = ring.generators().unwrap();
        let first_graph = (&variables[0] - &variables[2].pow(2).unwrap()).unwrap();
        let second_graph =
            (&variables[1] - &(&variables[3].pow(2).unwrap() + &variables[4]).unwrap()).unwrap();
        let first_monomial = (&variables[2].pow(3).unwrap() * &variables[3]).unwrap();
        let second_monomial = (&variables[2] * &variables[4].pow(4).unwrap()).unwrap();
        let base = ring
            .ideal([first_graph.clone(), second_graph.clone()])
            .unwrap();
        let source = ring
            .ideal([first_graph, second_graph, first_monomial, second_monomial])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 3);
        assert!(decomposition.certificates().iter().all(|certificate| {
            certificate.proof_kind() == "polynomial_graph_monomial_primary"
                && certificate.polynomial_graph_base() == Some(&base)
                && certificate.polynomial_graph_pivots() == Some(&[0, 1][..])
                && certificate.polynomial_graph_monomial_source().is_some()
                && certificate.polynomial_graph_factors().is_none()
                && certificate.associated_prime().proof_kind() == "polynomial_graph"
        }));
        let pivot_sets = decomposition
            .associated_primes()
            .iter()
            .map(|prime| prime.polynomial_graph_pivots().unwrap().to_vec())
            .collect::<Vec<_>>();
        assert!(pivot_sets.contains(&vec![0, 1, 2]));
        assert!(pivot_sets.contains(&vec![0, 1, 2, 4]));
        assert!(pivot_sets.contains(&vec![0, 1, 3, 4]));
        let mut dimensions = decomposition
            .components()
            .iter()
            .map(Ideal::dimension)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        dimensions.sort_unstable();
        assert_eq!(dimensions, [1, 1, 2]);
        let mut components = decomposition.components().iter();
        let reconstructed = components.next().unwrap().clone();
        let reconstructed = components.fold(reconstructed, |intersection, component| {
            intersection.intersection(component).unwrap()
        });
        assert_eq!(reconstructed, source);
        assert_eq!(
            PrimaryDecomposition::<RationalField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );

        let mut payload = serde_json::to_value(&decomposition.certificates()[0]).unwrap();
        payload["proof"]["pivot_variables"][0] = serde_json::json!(2);
        assert!(
            serde_json::from_value::<PolynomialPrimaryCertificate<RationalField>>(payload).is_err()
        );
    }

    #[test]
    fn common_factor_power_preserves_mixed_dimensional_multiplicity() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let common = (&variables[2] - &ring.one()).unwrap();
        let common_power = common.pow(3).unwrap();
        let source = ring
            .ideal([
                (&common_power * &variables[0]).unwrap(),
                (&common_power * &variables[1]).unwrap(),
            ])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        assert_eq!(
            decomposition
                .certificates()
                .iter()
                .filter(|certificate| certificate.proof_kind() == "principal_power")
                .count(),
            1
        );
        assert!(decomposition.certificates().iter().any(|certificate| {
            certificate.proof_kind() == "principal_power"
                && certificate.ideal() == &ring.ideal([common_power.clone()]).unwrap()
                && certificate.associated_prime().ideal() == &ring.ideal([common.clone()]).unwrap()
        }));
        assert!(decomposition.certificates().iter().any(|certificate| {
            certificate.proof_kind() == "monomial_primary"
                && certificate.ideal()
                    == &ring
                        .ideal([variables[0].clone(), variables[1].clone()])
                        .unwrap()
        }));
        assert_eq!(
            decomposition.components()[0]
                .intersection(&decomposition.components()[1])
                .unwrap(),
            source
        );
        assert_eq!(
            PrimaryDecomposition::<RationalField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );
    }

    #[test]
    fn common_factor_power_preserves_embedded_residual_components() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z", "w"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let common = (&variables[3] - &ring.one()).unwrap();
        let common_power = common.pow(2).unwrap();
        let first = (&variables[0].pow(3).unwrap() * &variables[1]).unwrap();
        let second = (&variables[0] * &variables[2].pow(4).unwrap()).unwrap();
        let source = ring
            .ideal([
                (&common_power * &first).unwrap(),
                (&common_power * &second).unwrap(),
            ])
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 4);
        assert_eq!(
            decomposition
                .certificates()
                .iter()
                .filter(|certificate| certificate.proof_kind() == "principal_power")
                .count(),
            1
        );
        assert_eq!(
            decomposition
                .certificates()
                .iter()
                .filter(|certificate| certificate.proof_kind() == "monomial_primary")
                .count(),
            3
        );
        let coordinate_radicals = decomposition
            .associated_primes()
            .iter()
            .filter_map(|prime| prime.coordinate_variables().map(<[usize]>::to_vec))
            .collect::<Vec<_>>();
        assert!(coordinate_radicals.contains(&vec![0]));
        assert!(coordinate_radicals.contains(&vec![0, 2]));
        assert!(coordinate_radicals.contains(&vec![1, 2]));
        let mut components = decomposition.components().iter();
        let reconstructed = components.next().unwrap().clone();
        let reconstructed = components.fold(reconstructed, |intersection, component| {
            intersection.intersection(component).unwrap()
        });
        assert_eq!(reconstructed, source);
    }

    #[test]
    fn ehv_separates_a_nonmonomial_embedded_primary_component() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let linear = (&variables[0] + &variables[1]).unwrap();
        let source = ring
            .ideal([linear.pow(2).unwrap(), (&linear * &variables[2]).unwrap()])
            .unwrap();
        let minimal_component = ring.ideal([linear.clone()]).unwrap();
        let embedded_prime = ring.ideal([linear.clone(), variables[2].clone()]).unwrap();
        let embedded_component = embedded_prime.pow(2).unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        assert!(decomposition.components().contains(&minimal_component));
        assert!(decomposition.components().contains(&embedded_component));
        assert!(decomposition.certificates().iter().any(|certificate| {
            certificate.ideal() == &minimal_component
                && certificate.proof_kind() == "ehv_single_associated_prime"
                && certificate
                    .ehv_associated_primes()
                    .is_some_and(|associated| associated.source() == certificate.ideal())
        }));
        assert!(decomposition.certificates().iter().any(|certificate| {
            certificate.ideal() == &embedded_component
                && certificate.proof_kind() == "ehv_equidimensional_hull"
                && certificate
                    .ehv_equidimensional_hull()
                    .is_some_and(|hull| hull.ideal() == certificate.ideal())
        }));
        assert_eq!(
            minimal_component.intersection(&embedded_component).unwrap(),
            source
        );
        assert_eq!(
            PrimaryDecomposition::<RationalField>::from_json(&decomposition.to_json().unwrap())
                .unwrap(),
            decomposition
        );

        let singleton = decomposition
            .certificates()
            .iter()
            .find(|certificate| certificate.ehv_associated_primes().is_some())
            .unwrap();
        let mut payload = serde_json::to_value(singleton).unwrap();
        payload["proof"]["associated_primes"]["source"] = serde_json::to_value(&source).unwrap();
        assert!(
            serde_json::from_value::<PolynomialPrimaryCertificate<RationalField>>(payload).is_err()
        );

        let hull = decomposition
            .certificates()
            .iter()
            .find(|certificate| certificate.ehv_equidimensional_hull().is_some())
            .unwrap();
        let mut payload = serde_json::to_value(hull).unwrap();
        payload["proof"]["hull"]["ideal"] = serde_json::to_value(&source).unwrap();
        assert!(
            serde_json::from_value::<PolynomialPrimaryCertificate<RationalField>>(payload).is_err()
        );
    }

    #[test]
    fn associated_prime_localization_separates_incomparable_embedded_components() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let linear = (&variables[0] + &variables[1]).unwrap();
        let minimal = ring.ideal([linear.clone()]).unwrap();
        let first_prime = ring.ideal([linear.clone(), variables[1].clone()]).unwrap();
        let second_prime = ring.ideal([linear.clone(), variables[2].clone()]).unwrap();
        let first = first_prime.pow(2).unwrap();
        let second = second_prime.pow(2).unwrap();
        let source = minimal
            .intersection(&first)
            .unwrap()
            .intersection(&second)
            .unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 3);
        assert!(decomposition.components().contains(&minimal));
        assert!(decomposition.components().contains(&first));
        assert!(decomposition.components().contains(&second));
        assert_eq!(
            decomposition
                .certificates()
                .iter()
                .filter(|certificate| certificate.proof_kind() == "ehv_equidimensional_hull")
                .count(),
            2
        );
        let reconstructed = decomposition.components()[0]
            .intersection(&decomposition.components()[1])
            .unwrap()
            .intersection(&decomposition.components()[2])
            .unwrap();
        assert_eq!(reconstructed, source);
    }

    #[test]
    fn proposed_primary_components_have_an_independent_admission_boundary() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let first = ring.ideal([variables[0].pow(2).unwrap()]).unwrap();
        let second = ring.ideal([variables[1].pow(3).unwrap()]).unwrap();
        let source = first.intersection(&second).unwrap();

        let first_certificate = first.certify_primary().unwrap();
        assert_eq!(first_certificate.ideal(), &first);
        first_certificate.verify().unwrap();

        let admitted = source
            .admit_primary_decomposition(&[second.clone(), first.clone()])
            .unwrap();
        admitted.verify().unwrap();
        assert_eq!(admitted.components().len(), 2);
        assert!(admitted.components().contains(&first));
        assert!(admitted.components().contains(&second));
        assert_eq!(
            PrimaryDecomposition::<RationalField>::from_json(&admitted.to_json().unwrap()).unwrap(),
            admitted
        );
    }

    #[test]
    fn primary_admission_rejects_nonprimary_and_redundant_proposals() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let first = ring.ideal([variables[0].clone()]).unwrap();
        let second = ring.ideal([variables[1].clone()]).unwrap();
        let source = first.intersection(&second).unwrap();
        let redundant = ring
            .ideal([variables[0].clone(), variables[1].clone()])
            .unwrap();

        assert!(source.certify_primary().is_err());
        assert!(source
            .admit_primary_decomposition(core::slice::from_ref(&first))
            .is_err());
        assert!(source
            .admit_primary_decomposition(&[first, second, redundant])
            .is_err());
    }
}
