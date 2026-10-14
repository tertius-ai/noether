// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical monomial ideals and their exact combinatorics.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CommutativeRing, ContextMismatchError, Field, Integer, NoetherError, RationalField,
    SemanticType, TypeDescription, Q,
};

use super::{
    hilbert, GroebnerError, HilbertSeries, Ideal, MonomialOrder, Polynomial, PolynomialRing,
};

const MAX_STANDARD_MONOMIALS: usize = 1_000_000;
const MAX_STANDARD_MONOMIAL_CANDIDATES: usize = 10_000_000;
const MAX_MONOMIAL_GENERATORS: usize = 100_000;

static MONOMIAL_IDEAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/MonomialIdeal@1",
    rust_path: "noether::commalg::MonomialIdeal",
    python_path: Some("noether.commalg.MonomialIdeal"),
};
const MONOMIAL_IDEAL_TYPE_ID: &str = "noether/commalg/MonomialIdeal@1";

/// Failure to construct or operate on a canonical monomial ideal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MonomialIdealError {
    /// An exponent vector had the wrong number of variables.
    ArityMismatch {
        /// Required number of variables.
        expected: usize,
        /// Supplied exponent count.
        actual: usize,
    },
    /// A monomial exponent was negative.
    NegativeExponent {
        /// Zero-based variable position.
        variable: usize,
    },
    /// An operation mixed different polynomial-ring contexts.
    ContextMismatch(ContextMismatchError),
    /// A finite standard-monomial basis was requested from a non-Artinian
    /// quotient.
    NotZeroDimensional,
    /// A bounded enumeration exceeded its explicit output cap.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Configured bound.
        limit: usize,
    },
    /// Deserialized or internally supplied generators were not the canonical
    /// minimal antichain promised by the type.
    NoncanonicalRepresentation,
}

impl fmt::Display for MonomialIdealError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArityMismatch { expected, actual } => write!(
                formatter,
                "monomial ideal expects {expected} exponents, received {actual}"
            ),
            Self::NegativeExponent { variable } => {
                write!(formatter, "monomial exponent {variable} is negative")
            }
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::NotZeroDimensional => formatter
                .write_str("finite standard monomials require a zero-dimensional monomial ideal"),
            Self::ResourceLimit { resource, limit } => {
                write!(
                    formatter,
                    "monomial ideal exceeded {resource} limit {limit}"
                )
            }
            Self::NoncanonicalRepresentation => {
                formatter.write_str("monomial ideal generators are not a canonical antichain")
            }
        }
    }
}

impl std::error::Error for MonomialIdealError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

/// A canonical monomial ideal in one explicit polynomial-ring context.
///
/// Generators are stored as a lexicographically ordered minimal divisibility
/// antichain. The empty antichain is the zero ideal; the singleton zero
/// exponent vector is the unit ideal.
#[derive(Clone)]
pub struct MonomialIdeal<R: CommutativeRing> {
    ring: PolynomialRing<R>,
    generators: Vec<Vec<Integer>>,
}

#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>")
)]
struct MonomialIdealWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    ring: PolynomialRing<R>,
    generators: Vec<Vec<Integer>>,
}

impl<R: CommutativeRing> PolynomialRing<R> {
    /// Construct a canonical monomial ideal from exponent vectors.
    pub fn monomial_ideal<I, E>(
        &self,
        generators: I,
    ) -> Result<MonomialIdeal<R>, MonomialIdealError>
    where
        I: IntoIterator<Item = E>,
        E: IntoIterator<Item = Integer>,
    {
        let arity = self.variables().len();
        let mut collected = Vec::new();
        for generator in generators {
            if collected.len() == MAX_MONOMIAL_GENERATORS {
                return Err(MonomialIdealError::ResourceLimit {
                    resource: "monomial generator count",
                    limit: MAX_MONOMIAL_GENERATORS,
                });
            }
            let mut exponent = Vec::with_capacity(arity);
            for value in generator {
                if exponent.len() == arity {
                    return Err(MonomialIdealError::ArityMismatch {
                        expected: arity,
                        actual: arity.saturating_add(1),
                    });
                }
                exponent.push(value);
            }
            if exponent.len() != arity {
                return Err(MonomialIdealError::ArityMismatch {
                    expected: arity,
                    actual: exponent.len(),
                });
            }
            collected.push(exponent);
        }
        MonomialIdeal::new(self.clone(), collected)
    }
}

impl<R: CommutativeRing> MonomialIdeal<R> {
    fn new(
        ring: PolynomialRing<R>,
        generators: Vec<Vec<Integer>>,
    ) -> Result<Self, MonomialIdealError> {
        if generators.len() > MAX_MONOMIAL_GENERATORS {
            return Err(MonomialIdealError::ResourceLimit {
                resource: "monomial generator count",
                limit: MAX_MONOMIAL_GENERATORS,
            });
        }
        let arity = ring.variables().len();
        for generator in &generators {
            if generator.len() != arity {
                return Err(MonomialIdealError::ArityMismatch {
                    expected: arity,
                    actual: generator.len(),
                });
            }
            if let Some(variable) = generator.iter().position(Integer::is_negative) {
                return Err(MonomialIdealError::NegativeExponent { variable });
            }
        }
        Ok(Self {
            ring,
            generators: minimalize(generators),
        })
    }

    /// Ambient polynomial-ring context.
    pub fn ring(&self) -> &PolynomialRing<R> {
        &self.ring
    }

    /// Minimal monomial generators as exact exponent vectors.
    pub fn minimal_generators(&self) -> &[Vec<Integer>] {
        &self.generators
    }

    /// Whether this is the zero ideal.
    pub fn is_zero(&self) -> bool {
        self.generators.is_empty()
    }

    /// Whether this is the unit ideal.
    pub fn is_unit(&self) -> bool {
        self.generators.len() == 1 && is_one(&self.generators[0])
    }

    /// Decide membership of one monomial exponent vector.
    pub fn contains_monomial(&self, exponent: &[Integer]) -> Result<bool, MonomialIdealError> {
        self.validate_exponent(exponent)?;
        Ok(self
            .generators
            .iter()
            .any(|generator| divides(generator, exponent)))
    }

    /// Sum of two monomial ideals.
    pub fn sum(&self, other: &Self) -> Result<Self, MonomialIdealError> {
        self.require_same_ring(other)?;
        self.generators
            .len()
            .checked_add(other.generators.len())
            .filter(|count| *count <= MAX_MONOMIAL_GENERATORS)
            .ok_or(MonomialIdealError::ResourceLimit {
                resource: "monomial generator count",
                limit: MAX_MONOMIAL_GENERATORS,
            })?;
        Self::new(
            self.ring.clone(),
            self.generators
                .iter()
                .chain(&other.generators)
                .cloned()
                .collect(),
        )
    }

    /// Product of two monomial ideals.
    pub fn product(&self, other: &Self) -> Result<Self, MonomialIdealError> {
        self.require_same_ring(other)?;
        self.check_pairwise_generator_count(other)?;
        let generators = self
            .generators
            .iter()
            .flat_map(|left| {
                other.generators.iter().map(|right| {
                    left.iter()
                        .zip(right)
                        .map(|(left, right)| left + right)
                        .collect()
                })
            })
            .collect();
        Self::new(self.ring.clone(), generators)
    }

    /// Intersection of two monomial ideals through pairwise least common
    /// multiples.
    pub fn intersection(&self, other: &Self) -> Result<Self, MonomialIdealError> {
        self.require_same_ring(other)?;
        self.check_pairwise_generator_count(other)?;
        let generators = self
            .generators
            .iter()
            .flat_map(|left| {
                other.generators.iter().map(|right| {
                    left.iter()
                        .zip(right)
                        .map(|(left, right)| left.max(right).clone())
                        .collect()
                })
            })
            .collect();
        Self::new(self.ring.clone(), generators)
    }

    /// Colon by one monomial.
    pub fn colon_monomial(&self, exponent: &[Integer]) -> Result<Self, MonomialIdealError> {
        self.validate_exponent(exponent)?;
        Self::new(
            self.ring.clone(),
            self.generators
                .iter()
                .map(|generator| {
                    generator
                        .iter()
                        .zip(exponent)
                        .map(|(generator, divisor)| {
                            if generator > divisor {
                                generator - divisor
                            } else {
                                Integer::zero()
                            }
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// Exact ideal quotient `self : other`.
    pub fn colon(&self, other: &Self) -> Result<Self, MonomialIdealError> {
        self.require_same_ring(other)?;
        let mut result = Self::new(
            self.ring.clone(),
            vec![vec![Integer::zero(); self.ring.variables().len()]],
        )?;
        for generator in &other.generators {
            result = result.intersection(&self.colon_monomial(generator)?)?;
        }
        Ok(result)
    }

    /// Radical, obtained by replacing every positive exponent by one.
    pub fn radical(&self) -> Result<Self, MonomialIdealError> {
        Self::new(
            self.ring.clone(),
            self.generators
                .iter()
                .map(|generator| {
                    generator
                        .iter()
                        .map(|exponent| {
                            if exponent.is_zero() {
                                Integer::zero()
                            } else {
                                Integer::one()
                            }
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// Nonnegative ideal power, with `I^0 = (1)`.
    pub fn pow(&self, mut exponent: u64) -> Result<Self, MonomialIdealError> {
        let mut result = Self::new(
            self.ring.clone(),
            vec![vec![Integer::zero(); self.ring.variables().len()]],
        )?;
        let mut power = self.clone();
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result.product(&power)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                power = power.product(&power)?;
            }
        }
        Ok(result)
    }

    /// Saturation by all powers of one monomial.
    pub fn saturate_monomial(&self, exponent: &[Integer]) -> Result<Self, MonomialIdealError> {
        self.validate_exponent(exponent)?;
        Self::new(
            self.ring.clone(),
            self.generators
                .iter()
                .map(|generator| {
                    generator
                        .iter()
                        .zip(exponent)
                        .map(|(power, saturating)| {
                            if saturating.is_zero() {
                                power.clone()
                            } else {
                                Integer::zero()
                            }
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// Saturation by all powers of another monomial ideal.
    pub fn saturate(&self, other: &Self) -> Result<Self, MonomialIdealError> {
        self.require_same_ring(other)?;
        let mut result = Self::new(
            self.ring.clone(),
            vec![vec![Integer::zero(); self.ring.variables().len()]],
        )?;
        for generator in &other.generators {
            result = result.intersection(&self.saturate_monomial(generator)?)?;
        }
        Ok(result)
    }

    /// Sorted variables occurring in a minimal generator.
    pub fn support(&self) -> Vec<usize> {
        (0..self.ring.variables().len())
            .filter(|&variable| {
                self.generators
                    .iter()
                    .any(|generator| !generator[variable].is_zero())
            })
            .collect()
    }

    /// Standard monomials of one exact total degree, with an explicit output
    /// cap.
    pub fn standard_monomials_of_degree(
        &self,
        degree: usize,
        maximum_count: usize,
    ) -> Result<Vec<Vec<Integer>>, MonomialIdealError> {
        let maximum_count = maximum_count.min(MAX_STANDARD_MONOMIALS);
        let mut output = Vec::new();
        let mut candidates = 0_usize;
        enumerate_degree(
            self.ring.variables().len(),
            degree,
            &mut vec![0; self.ring.variables().len()],
            0,
            &mut |candidate| {
                if candidates == MAX_STANDARD_MONOMIAL_CANDIDATES {
                    return Err(MonomialIdealError::ResourceLimit {
                        resource: "standard monomial candidate count",
                        limit: MAX_STANDARD_MONOMIAL_CANDIDATES,
                    });
                }
                candidates += 1;
                let candidate = candidate
                    .iter()
                    .copied()
                    .map(Integer::from)
                    .collect::<Vec<_>>();
                if !self
                    .generators
                    .iter()
                    .any(|generator| divides(generator, &candidate))
                {
                    if output.len() == maximum_count {
                        return Err(MonomialIdealError::ResourceLimit {
                            resource: "standard monomial count",
                            limit: maximum_count,
                        });
                    }
                    output.push(candidate);
                }
                Ok(())
            },
        )?;
        Ok(output)
    }

    /// Complete standard-monomial basis of a zero-dimensional quotient.
    pub fn finite_standard_monomials(&self) -> Result<Vec<Vec<Integer>>, MonomialIdealError> {
        if self.is_unit() {
            return Ok(Vec::new());
        }
        let mut bounds = Vec::with_capacity(self.ring.variables().len());
        for variable in 0..self.ring.variables().len() {
            let bound = self
                .generators
                .iter()
                .filter(|generator| {
                    generator
                        .iter()
                        .enumerate()
                        .all(|(index, exponent)| index == variable || exponent.is_zero())
                })
                .filter_map(|generator| generator[variable].to_u64_checked())
                .filter_map(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .min()
                .ok_or(MonomialIdealError::NotZeroDimensional)?;
            bounds.push(bound);
        }
        let box_size = bounds
            .iter()
            .try_fold(1_usize, |size, bound| size.checked_mul(*bound))
            .filter(|size| *size <= MAX_STANDARD_MONOMIALS)
            .ok_or(MonomialIdealError::ResourceLimit {
                resource: "finite standard monomial count",
                limit: MAX_STANDARD_MONOMIALS,
            })?;
        let mut output = Vec::with_capacity(box_size);
        enumerate_box(&bounds, &mut vec![0; bounds.len()], 0, &mut |candidate| {
            let candidate = candidate
                .iter()
                .copied()
                .map(Integer::from)
                .collect::<Vec<_>>();
            if !self
                .generators
                .iter()
                .any(|generator| divides(generator, &candidate))
            {
                output.push(candidate);
            }
        });
        Ok(output)
    }

    /// Numerator `K(t)` in the standard-graded identity
    /// `H(R/I,t)=K(t)/(1-t)^n`.
    pub fn hilbert_numerator(&self) -> Result<Vec<Integer>, GroebnerError> {
        self.require_standard_grading()?;
        hilbert::invariants(self.ring.variables().len(), &self.generators)
            .map(|data| data.numerator)
    }

    /// Hilbert numerator in the group algebra of the ring's full grading.
    ///
    /// This K-polynomial is defined without a positivity hypothesis. When the
    /// grading has finite pieces, adjoining denominator factors `1 - T^d`
    /// for the variable degrees `d` gives the exact multigraded Hilbert
    /// series of the quotient.
    pub fn multigraded_hilbert_numerator(
        &self,
    ) -> Result<Vec<(super::Degree, Integer)>, GroebnerError> {
        hilbert::multigraded_numerator(self.ring.grading(), &self.generators)
    }

    /// Exact Hilbert series in the group algebra of the ring grading.
    pub fn hilbert_series(&self) -> Result<HilbertSeries, GroebnerError> {
        HilbertSeries::new(
            self.ring.grading().clone(),
            self.multigraded_hilbert_numerator()?,
        )
        .map_err(|_| GroebnerError::InvalidBasis("the Hilbert series is not canonical"))
    }

    /// Krull dimension of `R/I`, with `-1` for the unit ideal.
    pub fn dimension(&self) -> Result<i64, GroebnerError> {
        hilbert::invariants(self.ring.variables().len(), &self.generators)
            .map(|data| data.dimension)
    }

    /// Standard-graded multiplicity of `R/I`.
    pub fn degree(&self) -> Result<Integer, GroebnerError> {
        self.require_standard_grading()?;
        hilbert::invariants(self.ring.variables().len(), &self.generators).map(|data| data.degree)
    }

    /// Standard Hilbert polynomial in the canonical variable `m`.
    pub fn hilbert_polynomial(&self) -> Result<Polynomial<RationalField>, GroebnerError> {
        self.require_standard_grading()?;
        let data = hilbert::invariants(self.ring.variables().len(), &self.generators)?;
        let coefficients = hilbert::hilbert_polynomial_coefficients(&data)?;
        let ring = PolynomialRing::new(Q, ["m"], MonomialOrder::GradedReverseLexicographic)
            .map_err(|_| {
                GroebnerError::InvalidBasis("the canonical Hilbert ring could not be built")
            })?;
        ring.from_terms(
            coefficients
                .into_iter()
                .enumerate()
                .filter(|(_, coefficient)| !coefficient.is_zero())
                .map(|(exponent, coefficient)| (vec![Integer::from(exponent)], coefficient)),
        )
        .map_err(Into::into)
    }

    /// Recheck the canonical minimal-antichain representation.
    pub fn verify(&self) -> Result<(), MonomialIdealError> {
        let rebuilt = Self::new(self.ring.clone(), self.generators.clone())?;
        if rebuilt.generators != self.generators {
            return Err(MonomialIdealError::NoncanonicalRepresentation);
        }
        Ok(())
    }

    /// Serialize the complete ring context and canonical exponent antichain.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a canonical monomial ideal.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn validate_exponent(&self, exponent: &[Integer]) -> Result<(), MonomialIdealError> {
        if exponent.len() != self.ring.variables().len() {
            return Err(MonomialIdealError::ArityMismatch {
                expected: self.ring.variables().len(),
                actual: exponent.len(),
            });
        }
        if let Some(variable) = exponent.iter().position(Integer::is_negative) {
            return Err(MonomialIdealError::NegativeExponent { variable });
        }
        Ok(())
    }

    fn require_same_ring(&self, other: &Self) -> Result<(), MonomialIdealError> {
        if self.ring == other.ring {
            Ok(())
        } else {
            Err(MonomialIdealError::ContextMismatch(ContextMismatchError {
                expected: self.ring.to_string(),
                actual: other.ring.to_string(),
            }))
        }
    }

    fn check_pairwise_generator_count(&self, other: &Self) -> Result<(), MonomialIdealError> {
        self.generators
            .len()
            .checked_mul(other.generators.len())
            .filter(|count| *count <= MAX_MONOMIAL_GENERATORS)
            .map(|_| ())
            .ok_or(MonomialIdealError::ResourceLimit {
                resource: "pairwise monomial generator count",
                limit: MAX_MONOMIAL_GENERATORS,
            })
    }

    fn require_standard_grading(&self) -> Result<(), GroebnerError> {
        if self.ring.grading().is_standard() {
            Ok(())
        } else {
            Err(GroebnerError::UnsupportedInvariant(
                "the univariate Hilbert numerator, polynomial, and degree require the standard grading",
            ))
        }
    }
}

impl<R: Field> MonomialIdeal<R> {
    /// Convert to the canonical polynomial ideal in the same ring.
    pub fn to_ideal(&self) -> Result<Ideal<PolynomialRing<R>>, GroebnerError> {
        let generators = self
            .generators
            .iter()
            .map(|exponent| {
                self.ring
                    .from_terms([(exponent.clone(), self.ring.coefficient_ring().one())])
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.ring.ideal(generators)
    }
}

impl<R> Serialize for MonomialIdeal<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        MonomialIdealWire {
            type_id: Self::TYPE_ID.to_owned(),
            ring: self.ring.clone(),
            generators: self.generators.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de, R> Deserialize<'de> for MonomialIdeal<R>
where
    R: CommutativeRing + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MonomialIdealWire::<R>::deserialize(deserializer)?;
        if wire.type_id != MONOMIAL_IDEAL_TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                MONOMIAL_IDEAL_TYPE_ID, wire.type_id
            )));
        }
        let generators = wire.generators;
        let result = Self::new(wire.ring, generators.clone()).map_err(serde::de::Error::custom)?;
        if result.generators != generators {
            return Err(serde::de::Error::custom(
                MonomialIdealError::NoncanonicalRepresentation,
            ));
        }
        Ok(result)
    }
}

impl<R: CommutativeRing> PartialEq for MonomialIdeal<R> {
    fn eq(&self, other: &Self) -> bool {
        self.ring == other.ring && self.generators == other.generators
    }
}

impl<R: CommutativeRing> Eq for MonomialIdeal<R> {}

impl<R: CommutativeRing> fmt::Debug for MonomialIdeal<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MonomialIdeal")
            .field("ring", &self.ring)
            .field("generator_count", &self.generators.len())
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Display for MonomialIdeal<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MonomialIdeal({}, generators={})",
            self.ring,
            self.generators.len()
        )
    }
}

impl<R> SemanticType for MonomialIdeal<R>
where
    R: CommutativeRing + Serialize,
{
    const TYPE_ID: &'static str = MONOMIAL_IDEAL_TYPE_ID;

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &MONOMIAL_IDEAL_DESCRIPTION
    }
}

fn minimalize(mut generators: Vec<Vec<Integer>>) -> Vec<Vec<Integer>> {
    generators.sort_by(|left, right| {
        total_degree(left)
            .cmp(&total_degree(right))
            .then_with(|| left.cmp(right))
    });
    generators.dedup();
    let mut antichain: Vec<Vec<Integer>> = Vec::with_capacity(generators.len());
    for generator in generators {
        if antichain.iter().any(|known| divides(known, &generator)) {
            continue;
        }
        antichain.retain(|known| !divides(&generator, known));
        antichain.push(generator);
    }
    antichain.sort();
    antichain
}

fn divides(left: &[Integer], right: &[Integer]) -> bool {
    left.iter().zip(right).all(|(left, right)| left <= right)
}

fn is_one(exponent: &[Integer]) -> bool {
    exponent.iter().all(Integer::is_zero)
}

fn total_degree(exponent: &[Integer]) -> Integer {
    exponent
        .iter()
        .fold(Integer::zero(), |degree, exponent| &degree + exponent)
}

fn enumerate_degree<F>(
    arity: usize,
    remaining: usize,
    candidate: &mut [usize],
    variable: usize,
    visit: &mut F,
) -> Result<(), MonomialIdealError>
where
    F: FnMut(&[usize]) -> Result<(), MonomialIdealError>,
{
    if variable + 1 == arity {
        candidate[variable] = remaining;
        visit(candidate)?;
        return Ok(());
    }
    if arity == 0 {
        if remaining == 0 {
            visit(candidate)?;
        }
        return Ok(());
    }
    for exponent in 0..=remaining {
        candidate[variable] = exponent;
        enumerate_degree(arity, remaining - exponent, candidate, variable + 1, visit)?;
    }
    Ok(())
}

fn enumerate_box<F>(bounds: &[usize], candidate: &mut [usize], variable: usize, visit: &mut F)
where
    F: FnMut(&[usize]),
{
    if variable == bounds.len() {
        visit(candidate);
        return;
    }
    for exponent in 0..bounds[variable] {
        candidate[variable] = exponent;
        enumerate_box(bounds, candidate, variable + 1, visit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{commalg::MonomialOrder, Q};

    fn exponents(values: &[&[u64]]) -> Vec<Vec<Integer>> {
        values
            .iter()
            .map(|value| value.iter().copied().map(Integer::from).collect())
            .collect()
    }

    fn ring() -> PolynomialRing<crate::RationalField> {
        PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap()
    }

    #[test]
    fn construction_canonicalizes_and_decides_membership() {
        let ring = ring();
        let ideal = ring
            .monomial_ideal(exponents(&[&[3, 1], &[2, 0], &[2, 0], &[1, 2]]))
            .unwrap();
        assert_eq!(ideal.minimal_generators(), exponents(&[&[1, 2], &[2, 0]]));
        assert!(ideal.contains_monomial(&exponents(&[&[4, 3]])[0]).unwrap());
        assert!(!ideal.contains_monomial(&exponents(&[&[1, 1]])[0]).unwrap());
        assert_eq!(ideal.support(), vec![0, 1]);
        assert!(ring
            .monomial_ideal(Vec::<Vec<Integer>>::new())
            .unwrap()
            .is_zero());
        assert!(ring
            .monomial_ideal(exponents(&[&[0, 0]]))
            .unwrap()
            .is_unit());
    }

    #[test]
    fn arithmetic_is_hand_checked() {
        let ring = ring();
        let left = ring.monomial_ideal(exponents(&[&[2, 0], &[1, 1]])).unwrap();
        let right = ring.monomial_ideal(exponents(&[&[0, 2]])).unwrap();

        assert_eq!(
            left.sum(&right).unwrap().minimal_generators(),
            exponents(&[&[0, 2], &[1, 1], &[2, 0]])
        );
        assert_eq!(
            left.product(&right).unwrap().minimal_generators(),
            exponents(&[&[1, 3], &[2, 2]])
        );
        assert_eq!(
            left.intersection(&right).unwrap().minimal_generators(),
            exponents(&[&[1, 2]])
        );
        assert_eq!(
            left.colon(&right).unwrap().minimal_generators(),
            exponents(&[&[1, 0]])
        );
        assert_eq!(
            left.radical().unwrap().minimal_generators(),
            exponents(&[&[1, 0]])
        );
        assert_eq!(
            left.saturate(&right).unwrap().minimal_generators(),
            exponents(&[&[1, 0]])
        );
        assert_eq!(
            left.pow(2).unwrap().minimal_generators(),
            exponents(&[&[2, 2], &[3, 1], &[4, 0]])
        );
    }

    #[test]
    fn standard_monomials_and_hilbert_data_agree() {
        let ring = ring();
        let ideal = ring.monomial_ideal(exponents(&[&[2, 0], &[0, 3]])).unwrap();
        assert_eq!(
            ideal.standard_monomials_of_degree(2, 10).unwrap(),
            exponents(&[&[0, 2], &[1, 1]])
        );
        assert_eq!(ideal.finite_standard_monomials().unwrap().len(), 6);
        assert_eq!(
            ideal.hilbert_numerator().unwrap(),
            [1, 0, -1, -1, 0, 1]
                .into_iter()
                .map(Integer::from)
                .collect::<Vec<_>>()
        );
        assert_eq!(ideal.dimension().unwrap(), 0);
        assert_eq!(ideal.degree().unwrap(), Integer::from(6));
        assert!(ideal.hilbert_polynomial().unwrap().is_zero());
    }

    #[test]
    fn multigraded_hilbert_numerator_is_exact() {
        let grading = crate::commalg::FreeAbelianGrading::new(
            2,
            vec![
                crate::commalg::Degree::new(vec![Integer::one(), Integer::zero()]),
                crate::commalg::Degree::new(vec![Integer::zero(), Integer::one()]),
            ],
        )
        .unwrap();
        let ring = PolynomialRing::with_grading(
            Q,
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
            grading,
        )
        .unwrap();
        let ideal = ring.monomial_ideal(exponents(&[&[2, 0], &[1, 1]])).unwrap();
        assert_eq!(
            ideal.multigraded_hilbert_numerator().unwrap(),
            vec![
                (
                    crate::commalg::Degree::new(vec![0.into(), 0.into()]),
                    1.into()
                ),
                (
                    crate::commalg::Degree::new(vec![1.into(), 1.into()]),
                    (-1).into()
                ),
                (
                    crate::commalg::Degree::new(vec![2.into(), 0.into()]),
                    (-1).into()
                ),
                (
                    crate::commalg::Degree::new(vec![2.into(), 1.into()]),
                    1.into()
                ),
            ]
        );
    }

    #[test]
    fn polynomial_ideal_round_trip_preserves_initial_ideal() {
        let ring = ring();
        let monomial = ring
            .monomial_ideal(exponents(&[&[2, 0], &[1, 1], &[0, 3]]))
            .unwrap();
        let polynomial = monomial.to_ideal().unwrap();
        assert_eq!(polynomial.initial_ideal(), monomial);
    }

    #[test]
    fn context_and_serialization_fail_closed() {
        let ring = ring();
        let ideal = ring.monomial_ideal(exponents(&[&[2, 0]])).unwrap();
        let other_ring =
            PolynomialRing::new(Q, ["u", "v"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let other = other_ring.monomial_ideal(exponents(&[&[2, 0]])).unwrap();
        assert!(matches!(
            ideal.sum(&other),
            Err(MonomialIdealError::ContextMismatch(_))
        ));

        let encoded = ideal.to_json().unwrap();
        assert_eq!(MonomialIdeal::from_json(&encoded).unwrap(), ideal);
        let mut wire: serde_json::Value = serde_json::from_str(&encoded).unwrap();
        let generators = wire["generators"].as_array_mut().unwrap();
        generators.push(generators[0].clone());
        let noncanonical = serde_json::to_string(&wire).unwrap();
        assert!(MonomialIdeal::<crate::RationalField>::from_json(&noncanonical).is_err());
    }
}
