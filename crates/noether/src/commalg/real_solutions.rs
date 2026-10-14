// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Certified real solution boxes obtained from rational univariate representations.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    BallError, NoetherError, Rational, RationalField, RealBall, RealBallField, Ring, SemanticType,
    TypeDescription,
};

use super::{
    GroebnerError, Polynomial, PolynomialError, RationalUnivariateRepresentation, RealRootError,
    RealRootIsolation,
};

const MAX_DENOMINATOR_REFINEMENTS: usize = 16;
const MAX_SOLUTION_POLYNOMIAL_DEGREE: usize = 4_096;

static REAL_SOLUTION_BOX_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/RealSolutionBox@1",
    rust_path: "noether::commalg::RealSolutionBox",
    python_path: Some("noether.commalg.RealSolutionBox"),
};

/// Failure to construct or verify certified real solutions from an RUR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RealSolutionError {
    /// The rational univariate representation failed exact verification.
    Groebner(GroebnerError),
    /// Exact real-root isolation failed.
    Root(RealRootError),
    /// Polynomial evaluation failed.
    Polynomial(PolynomialError),
    /// A real-ball context or enclosure was invalid.
    Ball(BallError),
    /// The common RUR denominator could not be proved nonzero on a root interval.
    DenominatorNotSeparated {
        /// Number of exact refinements attempted.
        refinements: usize,
    },
    /// Recomputed enclosure data did not match a supplied certificate.
    InvalidCertificate(&'static str),
    /// A univariate exponent exceeded the bounded native evaluation regime.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed limit.
        limit: usize,
    },
}

impl fmt::Display for RealSolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Groebner(error) => error.fmt(formatter),
            Self::Root(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Ball(error) => error.fmt(formatter),
            Self::DenominatorNotSeparated { refinements } => write!(
                formatter,
                "the RUR denominator still contains zero after {refinements} exact refinements"
            ),
            Self::InvalidCertificate(reason) => {
                write!(formatter, "invalid real-solution certificate: {reason}")
            }
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "real-solution construction exceeded its {resource} limit of {limit}"
            ),
        }
    }
}

impl std::error::Error for RealSolutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Groebner(error) => Some(error),
            Self::Root(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Ball(error) => Some(error),
            Self::DenominatorNotSeparated { .. }
            | Self::InvalidCertificate(_)
            | Self::ResourceLimit { .. } => None,
        }
    }
}

impl From<GroebnerError> for RealSolutionError {
    fn from(error: GroebnerError) -> Self {
        Self::Groebner(error)
    }
}

impl From<RealRootError> for RealSolutionError {
    fn from(error: RealRootError) -> Self {
        Self::Root(error)
    }
}

impl From<PolynomialError> for RealSolutionError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<BallError> for RealSolutionError {
    fn from(error: BallError) -> Self {
        Self::Ball(error)
    }
}

/// One certified real solution of a rational zero-dimensional system.
///
/// The exact parameter interval distinguishes the solution. `parameter_ball`
/// and every coordinate ball are rigorous Arb enclosures at `precision` bits.
/// Multiplicity is carried by the exact parameter-root certificate.
#[derive(Clone, PartialEq, Eq)]
pub struct RealSolutionBox {
    representation: RationalUnivariateRepresentation<RationalField>,
    parameter: RealRootIsolation,
    parameter_ball: RealBall,
    coordinates: Vec<RealBall>,
    precision: u64,
}

impl RealSolutionBox {
    /// Verified rational univariate representation used for this solution.
    pub fn representation(&self) -> &RationalUnivariateRepresentation<RationalField> {
        &self.representation
    }

    /// Exact isolating interval and multiplicity of the primitive parameter.
    pub fn parameter(&self) -> &RealRootIsolation {
        &self.parameter
    }

    /// Rigorous Arb enclosure of the primitive parameter.
    pub fn parameter_ball(&self) -> &RealBall {
        &self.parameter_ball
    }

    /// Coordinate enclosures in source-variable order.
    pub fn coordinates(&self) -> &[RealBall] {
        &self.coordinates
    }

    /// Arb working precision used for every returned ball.
    pub fn precision(&self) -> u64 {
        self.precision
    }

    /// Local multiplicity represented by this parameter root.
    pub fn multiplicity(&self) -> usize {
        self.parameter.multiplicity()
    }

    /// Recompute the certified solution family and check exact identity.
    pub fn verify(&self) -> Result<(), RealSolutionError> {
        self.representation.verify()?;
        let rebuilt = self.representation.real_solutions(self.precision)?;
        if rebuilt.iter().any(|candidate| candidate == self) {
            Ok(())
        } else {
            Err(RealSolutionError::InvalidCertificate(
                "recomputation did not reproduce this solution box",
            ))
        }
    }

    /// Serialize this certificate as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly reverify a typed certificate.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Debug for RealSolutionBox {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RealSolutionBox")
            .field("coordinates", &self.coordinates.len())
            .field("multiplicity", &self.multiplicity())
            .field("precision", &self.precision)
            .finish()
    }
}

impl fmt::Display for RealSolutionBox {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RealSolutionBox(coordinates={}, multiplicity={}, precision={})",
            self.coordinates.len(),
            self.multiplicity(),
            self.precision
        )
    }
}

impl SemanticType for RealSolutionBox {
    const TYPE_ID: &'static str = "noether/commalg/RealSolutionBox@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &REAL_SOLUTION_BOX_DESCRIPTION
    }
}

impl RationalUnivariateRepresentation<RationalField> {
    /// Enclose all distinct real solutions at one Arb working precision.
    ///
    /// The parameter roots are isolated and refined exactly before ball
    /// evaluation. If a future imported RUR has a nonconstant denominator,
    /// refinement continues until its ball excludes zero or fails closed at a
    /// fixed bound.
    pub fn real_solutions(
        &self,
        precision: u64,
    ) -> Result<Vec<RealSolutionBox>, RealSolutionError> {
        self.verify()?;
        let field = RealBallField::new(precision)?;
        self.equation()
            .real_roots()?
            .into_iter()
            .map(|root| build_solution(self, root, field))
            .collect()
    }
}

fn build_solution(
    representation: &RationalUnivariateRepresentation<RationalField>,
    root: RealRootIsolation,
    field: RealBallField,
) -> Result<RealSolutionBox, RealSolutionError> {
    let mut bits = field.precision_bits().saturating_add(16);
    for refinement in 0..MAX_DENOMINATOR_REFINEMENTS {
        let parameter = root.refine(bits)?;
        let parameter_ball =
            field.interval(parameter.lower().clone(), parameter.upper().clone())?;
        let denominator = evaluate_ball(representation.denominator(), &parameter_ball, field)?;
        if !denominator.contains_zero() {
            let coordinates = representation
                .coordinate_numerators()
                .iter()
                .map(|coordinate| {
                    let numerator = evaluate_ball(coordinate, &parameter_ball, field)?;
                    Ok(RealBall::from_raw(
                        field,
                        numerator
                            .raw()
                            .div(denominator.raw(), field.precision_bits() as i64),
                    ))
                })
                .collect::<Result<Vec<_>, RealSolutionError>>()?;
            return Ok(RealSolutionBox {
                representation: representation.clone(),
                parameter,
                parameter_ball,
                coordinates,
                precision: field.precision_bits(),
            });
        }
        bits = bits.saturating_mul(2);
        if refinement + 1 == MAX_DENOMINATOR_REFINEMENTS {
            return Err(RealSolutionError::DenominatorNotSeparated {
                refinements: MAX_DENOMINATOR_REFINEMENTS,
            });
        }
    }
    Err(RealSolutionError::DenominatorNotSeparated {
        refinements: MAX_DENOMINATOR_REFINEMENTS,
    })
}

fn evaluate_ball(
    polynomial: &Polynomial<RationalField>,
    value: &RealBall,
    field: RealBallField,
) -> Result<RealBall, RealSolutionError> {
    if polynomial.ring().variables().len() != 1 {
        return Err(RealSolutionError::InvalidCertificate(
            "an RUR component is not univariate",
        ));
    }
    let mut degree = 0_usize;
    let mut terms = Vec::new();
    for (exponent, coefficient) in polynomial.terms() {
        let exponent = exponent[0].to_string().parse::<usize>().map_err(|_| {
            RealSolutionError::ResourceLimit {
                resource: "univariate evaluation degree",
                limit: MAX_SOLUTION_POLYNOMIAL_DEGREE,
            }
        })?;
        if exponent > MAX_SOLUTION_POLYNOMIAL_DEGREE {
            return Err(RealSolutionError::ResourceLimit {
                resource: "univariate evaluation degree",
                limit: MAX_SOLUTION_POLYNOMIAL_DEGREE,
            });
        }
        degree = degree.max(exponent);
        terms.push((exponent, coefficient.clone()));
    }
    if terms.is_empty() {
        return Ok(field.zero());
    }
    let mut coefficients = vec![Rational::zero(); degree + 1];
    for (exponent, coefficient) in terms {
        coefficients[exponent] = coefficient;
    }
    coefficients
        .iter()
        .rev()
        .try_fold(field.zero(), |result, coefficient| {
            let product = field
                .mul(&result, value)
                .map_err(|_| RealSolutionError::InvalidCertificate("ball context mismatch"))?;
            field
                .add(&product, &field.element(coefficient.clone()))
                .map_err(|_| RealSolutionError::InvalidCertificate("ball context mismatch"))
        })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RealSolutionBoxWire {
    #[serde(rename = "type")]
    type_id: String,
    representation: RationalUnivariateRepresentation<RationalField>,
    parameter: RealRootIsolation,
    parameter_ball: RealBall,
    coordinates: Vec<RealBall>,
    precision: u64,
}

impl<'de> Deserialize<'de> for RealSolutionBox {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RealSolutionBoxWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let value = Self {
            representation: wire.representation,
            parameter: wire.parameter,
            parameter_ball: wire.parameter_ball,
            coordinates: wire.coordinates,
            precision: wire.precision,
        };
        value.verify().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl Serialize for RealSolutionBox {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("RealSolutionBox", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("representation", &self.representation)?;
        state.serialize_field("parameter", &self.parameter)?;
        state.serialize_field("parameter_ball", &self.parameter_ball)?;
        state.serialize_field("coordinates", &self.coordinates)?;
        state.serialize_field("precision", &self.precision)?;
        state.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::{MonomialOrder, PolynomialRing};
    use crate::Q;

    #[test]
    fn rur_projects_two_real_points_to_certified_boxes() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let x_squared_minus_one = ring.sub(&ring.mul(&x, &x).unwrap(), &ring.one()).unwrap();
        let y_minus_x = ring.sub(&y, &x).unwrap();
        let ideal = ring.ideal([x_squared_minus_one, y_minus_x]).unwrap();
        let representation = ideal.rational_univariate_representation().unwrap();
        let solutions = representation.real_solutions(128).unwrap();
        assert_eq!(solutions.len(), 2);
        for solution in solutions {
            assert_eq!(solution.coordinates().len(), 2);
            assert_eq!(solution.multiplicity(), 1);
            assert_eq!(solution.precision(), 128);
            solution.verify().unwrap();
            assert_eq!(
                RealSolutionBox::from_json(&solution.to_json().unwrap()).unwrap(),
                solution
            );
        }
    }
}
