// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Certified fractional lattices over p-adic local-field orders.

use core::fmt;
use core::ops::{Add, Mul};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::padic_local_order::{
    canonical_lattice, certified_index_valuation, element_from_flat_coordinates,
};
use super::{
    PadicLocalField, PadicLocalFieldElement, PadicLocalFieldOrder, PadicLocalOrderError,
    PadicValuation,
};
use crate::{
    ContextMismatchError, Integer, NoetherError, Ring, SemanticType, TypeDescription, UnitError,
};

static PADIC_LOCAL_FRACTIONAL_IDEAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicLocalFractionalIdeal@1",
    rust_path: "noether::numthy::PadicLocalFractionalIdeal",
    python_path: Some("noether.numthy.PadicLocalFractionalIdeal"),
};

/// Failure to construct or operate on a certified local fractional ideal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicLocalIdealError {
    /// Fractional ideals in this surface are nonzero.
    ZeroIdeal,
    /// A generator belonged to another local-field presentation.
    GeneratorContextMismatch {
        /// Zero-based rejected generator position.
        index: usize,
    },
    /// The available cap did not certify a full fractional lattice.
    InsufficientPrecision {
        /// Mathematical claim left unresolved.
        claim: &'static str,
        /// Uniform p-adic coefficient digits available.
        available: u64,
    },
    /// Fractional ideals from different order contexts were combined.
    ContextMismatch(ContextMismatchError),
    /// Inversion and a single prime exponent require the maximal DVR order.
    NonMaximalOrder,
    /// A maximal-order norm failed the expected residue-degree divisibility.
    InvalidNormValuation,
    /// Constructing or comparing the underlying certified order failed.
    Order(PadicLocalOrderError),
    /// A local-field unit operation failed.
    Unit(UnitError),
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for PadicLocalIdealError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroIdeal => {
                formatter.write_str("a local fractional ideal needs a nonzero generator")
            }
            Self::GeneratorContextMismatch { index } => write!(
                formatter,
                "local fractional-ideal generator {index} belongs to another field"
            ),
            Self::InsufficientPrecision { claim, available } => write!(
                formatter,
                "{claim} is unresolved at uniform p-adic precision {available}"
            ),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::NonMaximalOrder => formatter.write_str(
                "fractional-ideal inversion and prime-power factorization require the maximal local order",
            ),
            Self::InvalidNormValuation => formatter.write_str(
                "maximal-order fractional ideal has a norm valuation incompatible with the residue degree",
            ),
            Self::Order(error) => error.fmt(formatter),
            Self::Unit(error) => error.fmt(formatter),
            Self::InvalidEncoding(message) => {
                write!(formatter, "invalid local fractional-ideal encoding: {message}")
            }
        }
    }
}

impl std::error::Error for PadicLocalIdealError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Order(error) => Some(error),
            Self::Unit(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PadicLocalOrderError> for PadicLocalIdealError {
    fn from(value: PadicLocalOrderError) -> Self {
        Self::Order(value)
    }
}

impl From<ContextMismatchError> for PadicLocalIdealError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<UnitError> for PadicLocalIdealError {
    fn from(value: UnitError) -> Self {
        Self::Unit(value)
    }
}

/// A nonzero full-rank fractional lattice over a certified local order.
///
/// Negative field valuations are cleared by a recorded power of `p`; the
/// resulting integral lattice is canonicalized modulo the order's uniform
/// coefficient cap. This permits ideal arithmetic for non-maximal orders
/// without claiming invertibility or unique factorization there.
#[derive(Clone)]
pub struct PadicLocalFractionalIdeal {
    order: PadicLocalFieldOrder,
    basis: Vec<PadicLocalFieldElement>,
    scaled_rows: Vec<Vec<Integer>>,
    certified_precision: u64,
    denominator_power: u64,
    scaled_index_valuation: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicLocalFractionalIdealWire {
    #[serde(rename = "type")]
    type_id: String,
    order: PadicLocalFieldOrder,
    basis: Vec<PadicLocalFieldElement>,
}

impl PadicLocalFractionalIdeal {
    /// Construct the order-span of one or more nonzero ambient field elements.
    pub fn new(
        order: PadicLocalFieldOrder,
        generators: &[PadicLocalFieldElement],
    ) -> Result<Self, PadicLocalIdealError> {
        if generators.is_empty() || generators.iter().all(PadicLocalFieldElement::is_zero) {
            return Err(PadicLocalIdealError::ZeroIdeal);
        }
        for (index, generator) in generators.iter().enumerate() {
            if !order.field().contains(generator) {
                return Err(PadicLocalIdealError::GeneratorContextMismatch { index });
            }
        }

        let mut spanning = Vec::with_capacity(generators.len() * order.basis().len());
        for generator in generators {
            if generator.is_zero() {
                continue;
            }
            for basis_element in order.basis() {
                spanning.push(order.field().mul(generator, basis_element)?);
            }
        }
        Self::from_spanning_elements(order, &spanning)
    }

    fn from_spanning_elements(
        order: PadicLocalFieldOrder,
        spanning: &[PadicLocalFieldElement],
    ) -> Result<Self, PadicLocalIdealError> {
        let minimum_valuation = spanning
            .iter()
            .filter_map(|element| match element.valuation() {
                PadicValuation::Exact(value) => Some(value),
                PadicValuation::AtLeast(_) => None,
            })
            .min()
            .ok_or(PadicLocalIdealError::ZeroIdeal)?;
        let ramification_index =
            i64::try_from(order.field().ramification_index()).map_err(|_| {
                PadicLocalIdealError::InsufficientPrecision {
                    claim: "fractional-ideal denominator",
                    available: order.certified_precision(),
                }
            })?;
        let denominator_power = if minimum_valuation < 0 {
            let numerator = minimum_valuation.checked_neg().ok_or(
                PadicLocalIdealError::InsufficientPrecision {
                    claim: "fractional-ideal denominator",
                    available: order.certified_precision(),
                },
            )?;
            let quotient = numerator / ramification_index;
            let rounded = quotient + i64::from(numerator % ramification_index != 0);
            u64::try_from(rounded).map_err(|_| PadicLocalIdealError::InsufficientPrecision {
                claim: "fractional-ideal denominator",
                available: order.certified_precision(),
            })?
        } else {
            0
        };
        let clearing_scalar = p_power_element(order.field(), denominator_power as i64)?;
        let scaled = spanning
            .iter()
            .map(|element| {
                order
                    .field()
                    .mul(element, &clearing_scalar)
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>, PadicLocalIdealError>>()?;
        let certified_precision = scaled
            .iter()
            .filter_map(|element| {
                u64::try_from(element.known_absolute_precision())
                    .ok()
                    .map(|precision| precision / order.field().ramification_index() as u64)
            })
            .min()
            .unwrap_or(0)
            .min(order.certified_precision());
        if certified_precision == 0 {
            return Err(PadicLocalIdealError::InsufficientPrecision {
                claim: "fractional-ideal lattice coordinates",
                available: certified_precision,
            });
        }
        let rows = scaled
            .iter()
            .map(|element| {
                order
                    .field()
                    .integral_flat_coordinates(element, certified_precision)
                    .ok_or(PadicLocalIdealError::InsufficientPrecision {
                        claim: "fractional-ideal lattice coordinates",
                        available: certified_precision,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let scaled_rows = canonical_lattice(
            &rows,
            order.field().prime(),
            certified_precision,
            order.field().degree(),
        )?;
        let scaled_index_valuation =
            certified_index_valuation(&scaled_rows, order.field().prime(), certified_precision)?;
        let denominator = p_power_element(order.field(), -(denominator_power as i64))?;
        let basis = scaled_rows
            .iter()
            .map(|row| {
                let integral = element_from_flat_coordinates(order.field(), row)?;
                order
                    .field()
                    .mul(&integral, &denominator)
                    .map_err(PadicLocalIdealError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ideal = Self {
            order,
            basis,
            scaled_rows,
            certified_precision,
            denominator_power,
            scaled_index_valuation,
        };
        for basis_element in ideal.order.basis() {
            for ideal_element in &ideal.basis {
                let product = ideal.order.field().mul(basis_element, ideal_element)?;
                if !ideal.contains(&product) {
                    return Err(PadicLocalIdealError::InsufficientPrecision {
                        claim: "fractional-ideal order stability",
                        available: ideal.certified_precision,
                    });
                }
            }
        }
        Ok(ideal)
    }

    /// Coefficient order.
    pub fn order(&self) -> &PadicLocalFieldOrder {
        &self.order
    }

    /// Canonical certified `Z_p`-basis of this fractional lattice.
    pub fn basis(&self) -> &[PadicLocalFieldElement] {
        &self.basis
    }

    /// Uniform p-adic coefficient cap at which this fractional lattice is certified.
    pub const fn certified_precision(&self) -> u64 {
        self.certified_precision
    }

    /// Whether an ambient value lies in this fractional lattice.
    pub fn contains(&self, value: &PadicLocalFieldElement) -> bool {
        if !self.order.field().contains(value) {
            return false;
        }
        let Ok(clearing_scalar) =
            p_power_element(self.order.field(), self.denominator_power as i64)
        else {
            return false;
        };
        let Ok(scaled) = self.order.field().mul(value, &clearing_scalar) else {
            return false;
        };
        let Some(row) = self
            .order
            .field()
            .integral_flat_coordinates(&scaled, self.certified_precision)
        else {
            return false;
        };
        canonical_lattice(
            &[self.scaled_rows.clone(), vec![row]].concat(),
            self.order.field().prime(),
            self.certified_precision,
            self.order.field().degree(),
        )
        .is_ok_and(|rows| rows == self.scaled_rows)
    }

    /// Whether the fractional lattice is contained in its coefficient order.
    pub fn is_integral(&self) -> bool {
        self.basis
            .iter()
            .all(|element| self.order.contains(element))
    }

    /// Sum of two fractional ideals of the same certified order.
    pub fn add(&self, other: &Self) -> Result<Self, PadicLocalIdealError> {
        self.require_same_order(other)?;
        let mut generators = self.basis.clone();
        generators.extend(other.basis.clone());
        Self::new(self.order.clone(), &generators)
    }

    /// Product of two fractional ideals of the same certified order.
    pub fn mul(&self, other: &Self) -> Result<Self, PadicLocalIdealError> {
        self.require_same_order(other)?;
        let mut generators = Vec::with_capacity(self.basis.len() * other.basis.len());
        for left in &self.basis {
            for right in &other.basis {
                generators.push(self.order.field().mul(left, right)?);
            }
        }
        Self::new(self.order.clone(), &generators)
    }

    /// Scale by one nonzero ambient field element.
    pub fn scale(&self, scalar: &PadicLocalFieldElement) -> Result<Self, PadicLocalIdealError> {
        if !self.order.field().contains(scalar) {
            return Err(PadicLocalIdealError::ContextMismatch(
                ContextMismatchError {
                    expected: self.order.field().to_string(),
                    actual: "local-field element from another presentation".to_owned(),
                },
            ));
        }
        if scalar.is_zero() {
            return Err(PadicLocalIdealError::ZeroIdeal);
        }
        let generators = self
            .basis
            .iter()
            .map(|element| self.order.field().mul(element, scalar).map_err(Into::into))
            .collect::<Result<Vec<_>, PadicLocalIdealError>>()?;
        Self::new(self.order.clone(), &generators)
    }

    /// Whether this ideal is contained in another ideal of the same order.
    pub fn is_subset_of(&self, other: &Self) -> Result<bool, PadicLocalIdealError> {
        self.require_same_order(other)?;
        Ok(self.basis.iter().all(|element| other.contains(element)))
    }

    /// Signed `p`-valuation of the lattice norm relative to its coefficient order.
    pub fn norm_valuation(&self) -> Result<i64, PadicLocalIdealError> {
        let degree = i64::try_from(self.order.field().degree()).map_err(|_| {
            PadicLocalIdealError::InsufficientPrecision {
                claim: "fractional-ideal norm valuation",
                available: self.certified_precision,
            }
        })?;
        let scaled = i64::try_from(self.scaled_index_valuation).map_err(|_| {
            PadicLocalIdealError::InsufficientPrecision {
                claim: "fractional-ideal norm valuation",
                available: self.certified_precision,
            }
        })?;
        let order_index =
            i64::try_from(self.order.index_in_maximal_order_valuation()).map_err(|_| {
                PadicLocalIdealError::InsufficientPrecision {
                    claim: "fractional-ideal norm valuation",
                    available: self.certified_precision,
                }
            })?;
        let denominator = i64::try_from(self.denominator_power)
            .ok()
            .and_then(|power| power.checked_mul(degree))
            .ok_or(PadicLocalIdealError::InsufficientPrecision {
                claim: "fractional-ideal norm valuation",
                available: self.certified_precision,
            })?;
        scaled
            .checked_sub(denominator)
            .and_then(|value| value.checked_sub(order_index))
            .ok_or(PadicLocalIdealError::InsufficientPrecision {
                claim: "fractional-ideal norm valuation",
                available: self.certified_precision,
            })
    }

    /// Signed exponent of the unique maximal ideal, for the maximal order.
    pub fn maximal_ideal_exponent(&self) -> Result<i64, PadicLocalIdealError> {
        if !self.order.is_maximal() {
            return Err(PadicLocalIdealError::NonMaximalOrder);
        }
        let residue_degree = i64::try_from(self.order.field().residue_degree())
            .map_err(|_| PadicLocalIdealError::InvalidNormValuation)?;
        let norm = self.norm_valuation()?;
        if norm % residue_degree != 0 {
            return Err(PadicLocalIdealError::InvalidNormValuation);
        }
        Ok(norm / residue_degree)
    }

    /// Inverse fractional ideal, when the coefficient order is the maximal DVR.
    pub fn inverse(&self) -> Result<Self, PadicLocalIdealError> {
        let exponent = self.maximal_ideal_exponent()?;
        let generator = uniformizer_power(self.order.field(), -exponent)?;
        Self::new(self.order.clone(), &[generator])
    }

    /// Canonical typed JSON with full order identity and a revalidated basis.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and revalidate the fractional lattice and order action.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_same_order(&self, other: &Self) -> Result<(), PadicLocalIdealError> {
        if self.order == other.order {
            Ok(())
        } else {
            Err(PadicLocalIdealError::ContextMismatch(
                ContextMismatchError {
                    expected: self.order.to_string(),
                    actual: other.order.to_string(),
                },
            ))
        }
    }
}

impl PadicLocalFieldOrder {
    /// Construct the nonzero fractional ideal generated by ambient field values.
    pub fn fractional_ideal(
        &self,
        generators: &[PadicLocalFieldElement],
    ) -> Result<PadicLocalFractionalIdeal, PadicLocalIdealError> {
        PadicLocalFractionalIdeal::new(self.clone(), generators)
    }

    /// The unique maximal ideal `(pi)` of the certified maximal order.
    pub fn maximal_ideal(&self) -> Result<PadicLocalFractionalIdeal, PadicLocalIdealError> {
        if !self.is_maximal() {
            return Err(PadicLocalIdealError::NonMaximalOrder);
        }
        self.fractional_ideal(&[self.field().uniformizer()])
    }
}

impl PartialEq for PadicLocalFractionalIdeal {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
            && self.certified_precision == other.certified_precision
            && self.denominator_power == other.denominator_power
            && self.scaled_rows == other.scaled_rows
    }
}

impl Eq for PadicLocalFractionalIdeal {}

impl fmt::Debug for PadicLocalFractionalIdeal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PadicLocalFractionalIdeal")
            .field("degree", &self.order.field().degree())
            .field("denominator_power", &self.denominator_power)
            .field("certified_precision", &self.certified_precision)
            .field("norm_valuation", &self.norm_valuation().ok())
            .finish()
    }
}

impl fmt::Display for PadicLocalFractionalIdeal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "local fractional ideal of norm valuation {}",
            self.norm_valuation().unwrap_or_default()
        )
    }
}

impl SemanticType for PadicLocalFractionalIdeal {
    const TYPE_ID: &'static str = "noether/numthy/PadicLocalFractionalIdeal@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.order.canonical_bytes()?;
        append_component(&mut output, &self.certified_precision.to_be_bytes())?;
        append_component(&mut output, &self.denominator_power.to_be_bytes())?;
        for row in &self.scaled_rows {
            for entry in row {
                append_component(&mut output, &entry.canonical_bytes()?)?;
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_LOCAL_FRACTIONAL_IDEAL_DESCRIPTION
    }
}

impl Serialize for PadicLocalFractionalIdeal {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PadicLocalFractionalIdealWire {
            type_id: Self::TYPE_ID.to_owned(),
            order: self.order.clone(),
            basis: self.basis.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PadicLocalFractionalIdeal {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PadicLocalFractionalIdealWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.order, &wire.basis).map_err(serde::de::Error::custom)
    }
}

impl Add<&PadicLocalFractionalIdeal> for &PadicLocalFractionalIdeal {
    type Output = Result<PadicLocalFractionalIdeal, PadicLocalIdealError>;

    fn add(self, rhs: &PadicLocalFractionalIdeal) -> Self::Output {
        PadicLocalFractionalIdeal::add(self, rhs)
    }
}

impl Mul<&PadicLocalFractionalIdeal> for &PadicLocalFractionalIdeal {
    type Output = Result<PadicLocalFractionalIdeal, PadicLocalIdealError>;

    fn mul(self, rhs: &PadicLocalFractionalIdeal) -> Self::Output {
        PadicLocalFractionalIdeal::mul(self, rhs)
    }
}

fn p_power_element(
    field: &PadicLocalField,
    exponent: i64,
) -> Result<PadicLocalFieldElement, PadicLocalIdealError> {
    let base = field
        .coefficient_field()
        .coefficient_field()
        .element(field.prime().clone());
    let p = field
        .element_from_padic(&base)
        .map_err(PadicLocalOrderError::from)?;
    if exponent >= 0 {
        Ok(p.pow(exponent as u64))
    } else {
        Ok(p.inverse()?.pow(exponent.unsigned_abs()))
    }
}

fn uniformizer_power(
    field: &PadicLocalField,
    exponent: i64,
) -> Result<PadicLocalFieldElement, PadicLocalIdealError> {
    let uniformizer = field.uniformizer();
    if exponent >= 0 {
        Ok(uniformizer.pow(exponent as u64))
    } else {
        Ok(uniformizer.inverse()?.pow(exponent.unsigned_abs()))
    }
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization(
            "local fractional-ideal component does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}
