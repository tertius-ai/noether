// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked monomial and semilinear substitution.

use core::fmt;

use serde::{Deserialize, Serialize};

use super::{Polynomial, PolynomialRing, PowerSeries, PowerSeriesRing, SeriesTruncation};
use crate::{
    CoefficientFrobeniusContext, CoefficientMap, CommutativeRing, ContextMismatchError, Integer,
    NoetherError, SemanticType, TypeDescription,
};

static MONOMIAL_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/MonomialMap@1",
    rust_path: "noether::commalg::MonomialMap",
    python_path: Some("noether.commalg.MonomialMap"),
};

/// Invalid defining data or input for a monomial endomorphism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MonomialMapError {
    /// One image exponent vector had the wrong arity.
    ImageArity {
        /// Source variable whose image was malformed.
        variable: usize,
        /// Required target arity.
        expected: usize,
        /// Supplied exponent-vector arity.
        actual: usize,
    },
    /// A monomial image contained a negative exponent.
    NegativeExponent {
        /// Source variable whose image was malformed.
        variable: usize,
        /// Target variable carrying the negative exponent.
        target: usize,
        /// Rejected exponent.
        exponent: Integer,
    },
    /// An input exponent vector had the wrong arity for the map.
    InputArity {
        /// Required source arity.
        expected: usize,
        /// Supplied input arity.
        actual: usize,
    },
    /// Composition was requested for maps of different arities.
    CompositionArity {
        /// Arity of the outer map.
        outer: usize,
        /// Arity of the inner map.
        inner: usize,
    },
}

impl fmt::Display for MonomialMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageArity {
                variable,
                expected,
                actual,
            } => write!(
                formatter,
                "monomial image {variable} has arity {actual}; expected {expected}"
            ),
            Self::NegativeExponent {
                variable,
                target,
                exponent,
            } => write!(
                formatter,
                "monomial image {variable} has negative exponent {exponent} at target variable {target}"
            ),
            Self::InputArity { expected, actual } => write!(
                formatter,
                "monomial exponent has arity {actual}; expected {expected}"
            ),
            Self::CompositionArity { outer, inner } => write!(
                formatter,
                "cannot compose monomial maps of arities {outer} and {inner}"
            ),
        }
    }
}

impl std::error::Error for MonomialMapError {}

/// Failure to perform checked monomial or semilinear substitution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemilinearSubstitutionError {
    /// The polynomial or series coefficient context was not the scalar map's source.
    ContextMismatch(ContextMismatchError),
    /// The monomial map was invalid for the source exponent.
    Monomial(MonomialMapError),
    /// The corresponding target ring context could not be constructed.
    TargetContext(String),
    /// Applying the coefficient map failed.
    Coefficient(String),
    /// Constructing the target polynomial or series failed.
    Value(String),
}

impl fmt::Display for SemilinearSubstitutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Monomial(error) => error.fmt(formatter),
            Self::TargetContext(detail) => {
                write!(
                    formatter,
                    "cannot construct substitution target context: {detail}"
                )
            }
            Self::Coefficient(detail) => {
                write!(formatter, "coefficient map application failed: {detail}")
            }
            Self::Value(detail) => {
                write!(formatter, "cannot construct substituted value: {detail}")
            }
        }
    }
}

impl std::error::Error for SemilinearSubstitutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Monomial(error) => Some(error),
            Self::TargetContext(_) | Self::Coefficient(_) | Self::Value(_) => None,
        }
    }
}

impl From<ContextMismatchError> for SemilinearSubstitutionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<MonomialMapError> for SemilinearSubstitutionError {
    fn from(value: MonomialMapError) -> Self {
        Self::Monomial(value)
    }
}

/// A monomial endomorphism on an ordered list of variables.
///
/// Row `i` of [`Self::images`] is the exponent vector of the image of source
/// variable `i`. The map acts on row exponent vectors by right multiplication.
#[derive(Clone, PartialEq, Eq)]
pub struct MonomialMap {
    images: Vec<Vec<Integer>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MonomialMapWire {
    #[serde(rename = "type")]
    type_id: String,
    images: Vec<Vec<Integer>>,
}

impl MonomialMap {
    /// Construct a square monomial map from one image exponent per variable.
    pub fn new(images: Vec<Vec<Integer>>) -> Result<Self, MonomialMapError> {
        let arity = images.len();
        for (variable, image) in images.iter().enumerate() {
            if image.len() != arity {
                return Err(MonomialMapError::ImageArity {
                    variable,
                    expected: arity,
                    actual: image.len(),
                });
            }
            if let Some((target, exponent)) = image
                .iter()
                .enumerate()
                .find(|(_, exponent)| exponent.is_negative())
            {
                return Err(MonomialMapError::NegativeExponent {
                    variable,
                    target,
                    exponent: exponent.clone(),
                });
            }
        }
        Ok(Self { images })
    }

    /// Identity monomial map on `arity` ordered variables.
    pub fn identity(arity: usize) -> Self {
        let images = (0..arity)
            .map(|row| {
                (0..arity)
                    .map(|column| Integer::from(u8::from(row == column)))
                    .collect()
            })
            .collect();
        Self { images }
    }

    /// Diagonal substitution `x_i -> x_i^power` on every variable.
    pub fn diagonal_power(arity: usize, power: Integer) -> Result<Self, MonomialMapError> {
        if power.is_negative() {
            return Err(MonomialMapError::NegativeExponent {
                variable: 0,
                target: 0,
                exponent: power,
            });
        }
        let images = (0..arity)
            .map(|row| {
                (0..arity)
                    .map(|column| {
                        if row == column {
                            power.clone()
                        } else {
                            Integer::zero()
                        }
                    })
                    .collect()
            })
            .collect();
        Ok(Self { images })
    }

    /// Number of ordered source and target variables.
    pub fn arity(&self) -> usize {
        self.images.len()
    }

    /// Exact exponent vectors of the variable images.
    pub fn images(&self) -> &[Vec<Integer>] {
        &self.images
    }

    /// Transform one arbitrary-precision nonnegative exponent vector.
    pub fn apply(&self, exponent: &[Integer]) -> Result<Vec<Integer>, MonomialMapError> {
        if exponent.len() != self.arity() {
            return Err(MonomialMapError::InputArity {
                expected: self.arity(),
                actual: exponent.len(),
            });
        }
        let mut output = vec![Integer::zero(); self.arity()];
        for (source, source_exponent) in exponent.iter().enumerate() {
            if source_exponent.is_negative() {
                return Err(MonomialMapError::NegativeExponent {
                    variable: source,
                    target: source,
                    exponent: source_exponent.clone(),
                });
            }
            for (target, image_exponent) in self.images[source].iter().enumerate() {
                output[target] = &output[target] + &(source_exponent * image_exponent);
            }
        }
        Ok(output)
    }

    /// Compose `self` after `inner` using exact exponent arithmetic.
    pub fn compose(&self, inner: &Self) -> Result<Self, MonomialMapError> {
        if self.arity() != inner.arity() {
            return Err(MonomialMapError::CompositionArity {
                outer: self.arity(),
                inner: inner.arity(),
            });
        }
        let images = inner
            .images
            .iter()
            .map(|exponent| self.apply(exponent))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(images)
    }

    /// Serialize the exact exponent matrix as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&MonomialMapWire {
            type_id: Self::TYPE_ID.to_owned(),
            images: self.images.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate every exponent vector.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<MonomialMapWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.images).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl SemanticType for MonomialMap {
    const TYPE_ID: &'static str = "noether/commalg/MonomialMap@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &MONOMIAL_MAP_DESCRIPTION
    }
}

impl fmt::Debug for MonomialMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MonomialMap")
            .field("images", &self.images)
            .finish()
    }
}

impl fmt::Display for MonomialMap {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "monomial map on {} variables", self.arity())
    }
}

fn polynomial_target<R: CommutativeRing, T: CommutativeRing>(
    source: &PolynomialRing<R>,
    coefficient_ring: T,
) -> Result<PolynomialRing<T>, SemilinearSubstitutionError> {
    PolynomialRing::with_grading(
        coefficient_ring,
        source.variables().iter().cloned(),
        source.term_order(),
        source.grading().clone(),
    )
    .map_err(|error| SemilinearSubstitutionError::TargetContext(error.to_string()))
}

fn series_target<R: CommutativeRing, T: CommutativeRing>(
    source: &PowerSeriesRing<R>,
    coefficient_ring: T,
) -> Result<PowerSeriesRing<T>, SemilinearSubstitutionError> {
    PowerSeriesRing::new(
        coefficient_ring,
        source.variables().iter().cloned(),
        source.truncation().clone(),
    )
    .map_err(|error| SemilinearSubstitutionError::TargetContext(error.to_string()))
}

fn retained_series_exponent(
    map: &MonomialMap,
    exponent: &[u64],
    truncation: &SeriesTruncation,
) -> Result<Option<Vec<u64>>, MonomialMapError> {
    let exact = exponent
        .iter()
        .copied()
        .map(Integer::from)
        .collect::<Vec<_>>();
    let transformed = map.apply(&exact)?;
    let Some(transformed) = transformed
        .iter()
        .map(Integer::to_u64_checked)
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    Ok(truncation
        .contains_exponent(&transformed)
        .then_some(transformed))
}

impl<R> Polynomial<R>
where
    R: CommutativeRing + fmt::Display,
{
    /// Substitute monomials while leaving coefficients unchanged.
    pub fn monomial_substitution(
        &self,
        monomial_map: &MonomialMap,
    ) -> Result<Self, SemilinearSubstitutionError> {
        if monomial_map.arity() != self.ring().variables().len() {
            return Err(MonomialMapError::InputArity {
                expected: monomial_map.arity(),
                actual: self.ring().variables().len(),
            }
            .into());
        }
        let terms = self
            .terms()
            .map(|(exponent, coefficient)| Ok((monomial_map.apply(exponent)?, coefficient.clone())))
            .collect::<Result<Vec<_>, SemilinearSubstitutionError>>()?;
        self.ring()
            .from_terms(terms)
            .map_err(|error| SemilinearSubstitutionError::Value(error.to_string()))
    }

    /// Fuse a checked coefficient map and monomial substitution in one term stream.
    pub fn semilinear_substitution<M>(
        &self,
        coefficient_map: &M,
        monomial_map: &MonomialMap,
    ) -> Result<Polynomial<M::Target>, SemilinearSubstitutionError>
    where
        M: CoefficientMap<Source = R>,
    {
        if self.ring().coefficient_ring() != coefficient_map.source() {
            return Err(ContextMismatchError {
                expected: coefficient_map.source().to_string(),
                actual: self.ring().coefficient_ring().to_string(),
            }
            .into());
        }
        if monomial_map.arity() != self.ring().variables().len() {
            return Err(MonomialMapError::InputArity {
                expected: monomial_map.arity(),
                actual: self.ring().variables().len(),
            }
            .into());
        }
        let target = polynomial_target(self.ring(), coefficient_map.target().clone())?;
        let mut terms = Vec::with_capacity(self.nonzero_term_count());
        for (exponent, coefficient) in self.terms() {
            let exponent = monomial_map.apply(exponent)?;
            let coefficient = coefficient_map
                .apply(coefficient)
                .map_err(|error| SemilinearSubstitutionError::Coefficient(error.to_string()))?;
            terms.push((exponent, coefficient));
        }
        target
            .from_terms(terms)
            .map_err(|error| SemilinearSubstitutionError::Value(error.to_string()))
    }

    /// Apply coefficient Frobenius and `x_i -> x_i^(p^exponent)` in one pass.
    pub fn frobenius_lift(&self, exponent: u64) -> Result<Self, SemilinearSubstitutionError>
    where
        R: CoefficientFrobeniusContext,
    {
        let coefficient_map = self
            .ring()
            .coefficient_ring()
            .coefficient_frobenius(exponent);
        let scale = self
            .ring()
            .coefficient_ring()
            .residue_characteristic()
            .pow(exponent);
        let monomial_map = MonomialMap::diagonal_power(self.ring().variables().len(), scale)?;
        self.semilinear_substitution(&coefficient_map, &monomial_map)
    }
}

impl<R> PowerSeries<R>
where
    R: CommutativeRing + fmt::Display,
{
    /// Substitute monomials and discard transformed terms outside the fixed cap.
    pub fn monomial_substitution(
        &self,
        monomial_map: &MonomialMap,
    ) -> Result<Self, SemilinearSubstitutionError> {
        if monomial_map.arity() != self.ring().variables().len() {
            return Err(MonomialMapError::InputArity {
                expected: monomial_map.arity(),
                actual: self.ring().variables().len(),
            }
            .into());
        }
        let mut terms = Vec::with_capacity(self.nonzero_term_count());
        for (exponent, coefficient) in self.terms() {
            if let Some(exponent) =
                retained_series_exponent(monomial_map, exponent, self.ring().truncation())?
            {
                terms.push((exponent, coefficient.clone()));
            }
        }
        self.ring()
            .from_terms(terms)
            .map_err(|error| SemilinearSubstitutionError::Value(error.to_string()))
    }

    /// Fuse monomial capping and coefficient mapping in one source-term stream.
    ///
    /// A transformed exponent is rejected before its coefficient map is applied.
    pub fn semilinear_substitution<M>(
        &self,
        coefficient_map: &M,
        monomial_map: &MonomialMap,
    ) -> Result<PowerSeries<M::Target>, SemilinearSubstitutionError>
    where
        M: CoefficientMap<Source = R>,
    {
        if self.ring().coefficient_ring() != coefficient_map.source() {
            return Err(ContextMismatchError {
                expected: coefficient_map.source().to_string(),
                actual: self.ring().coefficient_ring().to_string(),
            }
            .into());
        }
        if monomial_map.arity() != self.ring().variables().len() {
            return Err(MonomialMapError::InputArity {
                expected: monomial_map.arity(),
                actual: self.ring().variables().len(),
            }
            .into());
        }
        let target = series_target(self.ring(), coefficient_map.target().clone())?;
        let mut terms = Vec::with_capacity(self.nonzero_term_count());
        for (exponent, coefficient) in self.terms() {
            let Some(exponent) =
                retained_series_exponent(monomial_map, exponent, self.ring().truncation())?
            else {
                continue;
            };
            let coefficient = coefficient_map
                .apply(coefficient)
                .map_err(|error| SemilinearSubstitutionError::Coefficient(error.to_string()))?;
            terms.push((exponent, coefficient));
        }
        target
            .from_terms(terms)
            .map_err(|error| SemilinearSubstitutionError::Value(error.to_string()))
    }

    /// Apply coefficient Frobenius and `x_i -> x_i^(p^exponent)` with early capping.
    pub fn frobenius_lift(&self, exponent: u64) -> Result<Self, SemilinearSubstitutionError>
    where
        R: CoefficientFrobeniusContext,
    {
        let coefficient_map = self
            .ring()
            .coefficient_ring()
            .coefficient_frobenius(exponent);
        let scale = self
            .ring()
            .coefficient_ring()
            .residue_characteristic()
            .pow(exponent);
        let monomial_map = MonomialMap::diagonal_power(self.ring().variables().len(), scale)?;
        self.semilinear_substitution(&coefficient_map, &monomial_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numthy::{FiniteField, PadicRing, QadicRing};
    use crate::Ring;

    #[test]
    fn exact_monomial_maps_compose_and_serialize() {
        let map = MonomialMap::new(vec![
            vec![Integer::from(2), Integer::from(1)],
            vec![Integer::from(0), Integer::from(3)],
        ])
        .unwrap();
        assert_eq!(
            map.apply(&[Integer::from(4), Integer::from(5)]).unwrap(),
            vec![Integer::from(8), Integer::from(19)]
        );
        assert_eq!(
            MonomialMap::from_json(&map.to_json().unwrap()).unwrap(),
            map
        );
        let square = map.compose(&map).unwrap();
        assert_eq!(
            square.apply(&[Integer::from(1), Integer::from(0)]).unwrap(),
            vec![Integer::from(4), Integer::from(5)]
        );
    }

    #[test]
    fn polynomial_frobenius_uses_arbitrary_precision_exponents() {
        let ring = PadicRing::new(3_u8, 4).unwrap();
        let polynomials =
            PolynomialRing::new(ring, ["x"], super::super::MonomialOrder::Lexicographic).unwrap();
        let polynomial = polynomials.generator(0).unwrap();
        let mapped = polynomial.frobenius_lift(41).unwrap();
        assert_eq!(
            mapped.terms().next().unwrap().0,
            &[Integer::from(3_u8).pow(41)]
        );
    }

    #[test]
    fn qadic_series_frobenius_caps_before_retaining_coefficients() {
        let coefficient_ring =
            QadicRing::new(FiniteField::extension(3_u64, 2).unwrap(), 4).unwrap();
        let series_ring = PowerSeriesRing::with_grading(
            coefficient_ring.clone(),
            ["x", "y"],
            crate::commalg::WeightedGrading::new(vec![1, 2]).unwrap(),
            7,
        )
        .unwrap();
        let value = series_ring
            .from_terms([
                (vec![0, 0], coefficient_ring.generator()),
                (vec![1, 0], coefficient_ring.generator()),
                (vec![2, 0], coefficient_ring.one()),
                (vec![0, 2], coefficient_ring.one()),
            ])
            .unwrap();
        let mapped = value.frobenius_lift(1).unwrap();
        assert_eq!(mapped.nonzero_term_count(), 3);
        assert!(mapped
            .terms()
            .all(|(exponent, _)| { mapped.ring().truncation().contains_exponent(exponent) }));
        assert_eq!(
            mapped.coefficient(&[0, 0]).unwrap(),
            coefficient_ring.generator().frobenius(1)
        );
        assert_eq!(
            mapped.coefficient(&[3, 0]).unwrap(),
            coefficient_ring.generator().frobenius(1)
        );
        assert_eq!(mapped.coefficient(&[6, 0]).unwrap(), coefficient_ring.one());
    }

    #[test]
    fn transformed_exponents_beyond_u64_are_dropped_by_finite_series_caps() {
        let ring = PadicRing::new(3_u8, 3).unwrap();
        let series_ring = PowerSeriesRing::with_precision(ring, ["x"], 5).unwrap();
        let value = series_ring.generator(0).unwrap();
        let map = MonomialMap::diagonal_power(1, Integer::from(u64::MAX) + 1_i32).unwrap();
        assert!(value.monomial_substitution(&map).unwrap().is_zero());
    }
}
