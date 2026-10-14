// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Certified `Z_p`-orders in Eisenstein-presented local fields.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{PadicLocalField, PadicLocalFieldElement, PadicLocalFieldError, PadicValuation};
use crate::linalg::LinalgError;
use crate::{
    CommutativeRing, ContextMismatchError, Integer, NoetherError, Ring, SemanticType,
    TypeDescription, UnitError, UnitInversion, Z,
};

static PADIC_LOCAL_FIELD_ORDER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PadicLocalFieldOrder@1",
    rust_path: "noether::numthy::PadicLocalFieldOrder",
    python_path: Some("noether.numthy.PadicLocalFieldOrder"),
};

/// Failure to construct or use a certified finite-precision local order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PadicLocalOrderError {
    /// The basis did not have the absolute local degree as its length.
    WrongBasisLength {
        /// Required `Z_p`-rank.
        degree: usize,
        /// Supplied basis length.
        actual: usize,
    },
    /// A basis element belonged to another presented local field.
    BasisContextMismatch {
        /// Zero-based rejected basis position.
        index: usize,
    },
    /// A basis element had certified negative valuation.
    NonIntegralBasis {
        /// Zero-based rejected basis position.
        index: usize,
        /// Certified negative valuation.
        valuation: i64,
    },
    /// The available cap could not certify a lattice or invariant claim.
    InsufficientPrecision {
        /// Mathematical claim that remained unresolved.
        claim: &'static str,
        /// Uniform coefficient digits available for the check.
        available: u64,
    },
    /// The basis failed to have full `Z_p`-rank at the certified cap.
    NotFullRank,
    /// The lattice did not contain one.
    MissingOne,
    /// The lattice was not closed under multiplication.
    NotClosed,
    /// The requested containment of local orders did not hold.
    NotContained,
    /// A single different exponent was requested for a non-maximal order.
    DifferentRequiresMaximalOrder,
    /// A local-field operation rejected an internal construction.
    LocalField(PadicLocalFieldError),
    /// Exact integer normal-form arithmetic failed.
    LinearAlgebra(LinalgError),
    /// A local-field ring operation rejected an internal context.
    ContextMismatch(ContextMismatchError),
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for PadicLocalOrderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongBasisLength { degree, actual } => write!(
                formatter,
                "p-adic local order needs {degree} basis elements; received {actual}"
            ),
            Self::BasisContextMismatch { index } => write!(
                formatter,
                "p-adic local order basis element {index} belongs to another field"
            ),
            Self::NonIntegralBasis { index, valuation } => write!(
                formatter,
                "p-adic local order basis element {index} has negative valuation {valuation}"
            ),
            Self::InsufficientPrecision { claim, available } => write!(
                formatter,
                "{claim} is unresolved at uniform p-adic precision {available}"
            ),
            Self::NotFullRank => formatter.write_str("p-adic local order basis is not full rank"),
            Self::MissingOne => formatter.write_str("p-adic local order does not contain one"),
            Self::NotClosed => {
                formatter.write_str("p-adic local order basis is not closed under multiplication")
            }
            Self::NotContained => formatter.write_str(
                "the first p-adic local order is not contained in the proposed overorder",
            ),
            Self::DifferentRequiresMaximalOrder => formatter.write_str(
                "a single uniformizer valuation for the different requires the maximal local order",
            ),
            Self::LocalField(error) => error.fmt(formatter),
            Self::LinearAlgebra(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::InvalidEncoding(message) => {
                write!(formatter, "invalid p-adic local order encoding: {message}")
            }
        }
    }
}

impl std::error::Error for PadicLocalOrderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LocalField(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PadicLocalFieldError> for PadicLocalOrderError {
    fn from(value: PadicLocalFieldError) -> Self {
        Self::LocalField(value)
    }
}

impl From<LinalgError> for PadicLocalOrderError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<ContextMismatchError> for PadicLocalOrderError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// A full-rank unital `Z_p`-order certified inside one local-field presentation.
///
/// The order stores the canonical row-Hermite lattice generated by the basis
/// together with `p^N O_K`, where `N` is the recorded uniform coefficient
/// precision. Equality therefore means equality of certified finite-precision
/// order contexts; it does not claim to distinguish lattices beyond `p^N`.
#[derive(Clone)]
pub struct PadicLocalFieldOrder {
    field: PadicLocalField,
    basis: Vec<PadicLocalFieldElement>,
    canonical_rows: Vec<Vec<Integer>>,
    certified_precision: u64,
    index_valuation: u64,
    maximal_different_valuation: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PadicLocalFieldOrderWire {
    #[serde(rename = "type")]
    type_id: String,
    field: PadicLocalField,
    basis: Vec<PadicLocalFieldElement>,
    certified_precision: u64,
}

impl PadicLocalFieldOrder {
    /// Validate an arbitrary full `Z_p`-basis at the field's largest uniform cap.
    pub fn new(
        field: PadicLocalField,
        basis: &[PadicLocalFieldElement],
    ) -> Result<Self, PadicLocalOrderError> {
        let precision = field.local_precision().absolute_uniformizer_precision()
            / field.ramification_index() as u64;
        Self::new_at_precision(field, basis, precision)
    }

    fn new_at_precision(
        field: PadicLocalField,
        basis: &[PadicLocalFieldElement],
        certified_precision: u64,
    ) -> Result<Self, PadicLocalOrderError> {
        let degree = field.degree();
        if basis.len() != degree {
            return Err(PadicLocalOrderError::WrongBasisLength {
                degree,
                actual: basis.len(),
            });
        }
        if certified_precision == 0
            || certified_precision
                > field.local_precision().absolute_uniformizer_precision()
                    / field.ramification_index() as u64
        {
            return Err(PadicLocalOrderError::InsufficientPrecision {
                claim: "local order construction",
                available: certified_precision,
            });
        }
        for (index, element) in basis.iter().enumerate() {
            if !field.contains(element) {
                return Err(PadicLocalOrderError::BasisContextMismatch { index });
            }
            if let PadicValuation::Exact(valuation) = element.valuation() {
                if valuation < 0 {
                    return Err(PadicLocalOrderError::NonIntegralBasis { index, valuation });
                }
            }
        }

        let coordinate_rows = basis
            .iter()
            .map(|element| {
                field
                    .integral_flat_coordinates(element, certified_precision)
                    .ok_or(PadicLocalOrderError::InsufficientPrecision {
                        claim: "local order basis coordinates",
                        available: certified_precision,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let canonical_rows =
            canonical_lattice(&coordinate_rows, field.prime(), certified_precision, degree)?;
        let index_valuation =
            certified_index_valuation(&canonical_rows, field.prime(), certified_precision)?;
        let basis = canonical_rows
            .iter()
            .map(|row| element_from_flat_coordinates(&field, row))
            .collect::<Result<Vec<_>, _>>()?;
        let maximal_different_valuation = maximal_different_valuation(&field)?;
        let order = Self {
            field,
            basis,
            canonical_rows,
            certified_precision,
            index_valuation,
            maximal_different_valuation,
        };
        if !order.contains(&order.field.one()) {
            return Err(PadicLocalOrderError::MissingOne);
        }
        for left in &order.basis {
            for right in &order.basis {
                let product = order.field.mul(left, right)?;
                if !order.contains(&product) {
                    return Err(PadicLocalOrderError::NotClosed);
                }
            }
        }
        Ok(order)
    }

    /// Ambient p-adic local field.
    pub fn field(&self) -> &PadicLocalField {
        &self.field
    }

    /// Canonical certified `Z_p`-basis in the ambient local field.
    pub fn basis(&self) -> &[PadicLocalFieldElement] {
        &self.basis
    }

    /// Uniform p-adic coefficient cap at which the lattice was certified.
    pub const fn certified_precision(&self) -> u64 {
        self.certified_precision
    }

    /// Whether an ambient field value belongs to this certified lattice.
    pub fn contains(&self, value: &PadicLocalFieldElement) -> bool {
        let Some(row) = self
            .field
            .integral_flat_coordinates(value, self.certified_precision)
        else {
            return false;
        };
        canonical_lattice(
            &[self.canonical_rows.clone(), vec![row]].concat(),
            self.field.prime(),
            self.certified_precision,
            self.field.degree(),
        )
        .is_ok_and(|rows| rows == self.canonical_rows)
    }

    /// Certified `p`-valuation of `[O_K : self]`.
    pub const fn index_in_maximal_order_valuation(&self) -> u64 {
        self.index_valuation
    }

    /// Certified `p`-valuation of `[super_order : self]`.
    pub fn index_valuation(&self, super_order: &Self) -> Result<u64, PadicLocalOrderError> {
        if self.field != super_order.field
            || self.certified_precision != super_order.certified_precision
            || self
                .basis
                .iter()
                .any(|element| !super_order.contains(element))
        {
            return Err(PadicLocalOrderError::NotContained);
        }
        self.index_valuation
            .checked_sub(super_order.index_valuation)
            .ok_or(PadicLocalOrderError::NotContained)
    }

    /// Whether this certified order equals the explicit ring of integers.
    pub const fn is_maximal(&self) -> bool {
        self.index_valuation == 0
    }

    /// The explicit maximal overorder `O_K` at the same certified cap.
    pub fn maximal_overorder(&self) -> Result<Self, PadicLocalOrderError> {
        let basis = standard_integral_basis(&self.field)?;
        Self::new_at_precision(self.field.clone(), &basis, self.certified_precision)
    }

    /// Certified `p`-valuation of the order discriminant.
    pub fn discriminant_valuation(&self) -> Result<u64, PadicLocalOrderError> {
        self.maximal_different_valuation
            .checked_mul(self.field.residue_degree() as u64)
            .and_then(|value| value.checked_add(self.index_valuation.saturating_mul(2)))
            .ok_or(PadicLocalOrderError::InsufficientPrecision {
                claim: "local order discriminant valuation",
                available: self.certified_precision,
            })
    }

    /// Uniformizer valuation of the different of the maximal local order.
    pub fn different_valuation(&self) -> Result<u64, PadicLocalOrderError> {
        if !self.is_maximal() {
            return Err(PadicLocalOrderError::DifferentRequiresMaximalOrder);
        }
        Ok(self.maximal_different_valuation)
    }

    /// Canonical typed JSON carrying the field, basis, and certification cap.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the order lattice and multiplication closure.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn mismatch(&self, value: &PadicLocalFieldElement) -> ContextMismatchError {
        ContextMismatchError {
            expected: self.to_string(),
            actual: if self.field.contains(value) {
                "local-field element outside the certified order".to_owned()
            } else {
                "local-field element from another field".to_owned()
            },
        }
    }
}

impl PadicLocalField {
    /// The maximal order `O_K = W(k)[pi]` in the admitted Eisenstein presentation.
    pub fn maximal_order(&self) -> Result<PadicLocalFieldOrder, PadicLocalOrderError> {
        PadicLocalFieldOrder::new(self.clone(), &standard_integral_basis(self)?)
    }

    /// The order generated by the declared unramified generator and uniformizer.
    ///
    /// Every currently admitted local field is Eisenstein-presented, so this
    /// equation order is the maximal order by construction.
    pub fn equation_order(&self) -> Result<PadicLocalFieldOrder, PadicLocalOrderError> {
        self.maximal_order()
    }

    /// Validate an arbitrary full `Z_p`-basis as a local order.
    pub fn order(
        &self,
        basis: &[PadicLocalFieldElement],
    ) -> Result<PadicLocalFieldOrder, PadicLocalOrderError> {
        PadicLocalFieldOrder::new(self.clone(), basis)
    }
}

impl PartialEq for PadicLocalFieldOrder {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.certified_precision == other.certified_precision
            && self.canonical_rows == other.canonical_rows
    }
}

impl Eq for PadicLocalFieldOrder {}

impl fmt::Debug for PadicLocalFieldOrder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PadicLocalFieldOrder")
            .field("degree", &self.field.degree())
            .field("certified_precision", &self.certified_precision)
            .field("index_valuation", &self.index_valuation)
            .finish()
    }
}

impl fmt::Display for PadicLocalFieldOrder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "p-adic local order of degree {} at coefficient precision {} with index valuation {}",
            self.field.degree(),
            self.certified_precision,
            self.index_valuation
        )
    }
}

impl SemanticType for PadicLocalFieldOrder {
    const TYPE_ID: &'static str = "noether/numthy/PadicLocalFieldOrder@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.field.canonical_bytes()?;
        append_component(&mut output, &self.certified_precision.to_be_bytes())?;
        for row in &self.canonical_rows {
            for entry in row {
                append_component(&mut output, &entry.canonical_bytes()?)?;
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PADIC_LOCAL_FIELD_ORDER_DESCRIPTION
    }
}

impl Serialize for PadicLocalFieldOrder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PadicLocalFieldOrderWire {
            type_id: Self::TYPE_ID.to_owned(),
            field: self.field.clone(),
            basis: self.basis.clone(),
            certified_precision: self.certified_precision,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PadicLocalFieldOrder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PadicLocalFieldOrderWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new_at_precision(wire.field, &wire.basis, wire.certified_precision)
            .map_err(serde::de::Error::custom)
    }
}

impl Ring for PadicLocalFieldOrder {
    type Element = PadicLocalFieldElement;

    fn contains(&self, value: &Self::Element) -> bool {
        PadicLocalFieldOrder::contains(self, value)
    }

    fn zero(&self) -> Self::Element {
        self.field.zero()
    }

    fn one(&self) -> Self::Element {
        self.field.one()
    }

    fn add(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        if !self.contains(left) {
            return Err(self.mismatch(left));
        }
        if !self.contains(right) {
            return Err(self.mismatch(right));
        }
        self.field.add(left, right)
    }

    fn sub(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        if !self.contains(left) {
            return Err(self.mismatch(left));
        }
        if !self.contains(right) {
            return Err(self.mismatch(right));
        }
        self.field.sub(left, right)
    }

    fn mul(
        &self,
        left: &Self::Element,
        right: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        if !self.contains(left) {
            return Err(self.mismatch(left));
        }
        if !self.contains(right) {
            return Err(self.mismatch(right));
        }
        self.field.mul(left, right)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        if !self.contains(value) {
            return Err(self.mismatch(value));
        }
        self.field.neg(value)
    }
}

impl CommutativeRing for PadicLocalFieldOrder {}

impl UnitInversion for PadicLocalFieldOrder {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        if !self.contains(value) {
            return Err(self.mismatch(value).into());
        }
        let inverse = self.field.inverse(value)?;
        if self.contains(&inverse) {
            Ok(inverse)
        } else {
            Err(UnitError::NotAUnit {
                context: Self::TYPE_ID,
            })
        }
    }
}

fn standard_integral_basis(
    field: &PadicLocalField,
) -> Result<Vec<PadicLocalFieldElement>, PadicLocalOrderError> {
    let degree = field.degree();
    let mut basis = Vec::with_capacity(degree);
    for uniformizer_power in 0..field.ramification_index() {
        for residue_power in 0..field.residue_degree() {
            let mut coordinates = vec![Integer::zero(); degree];
            coordinates[uniformizer_power * field.residue_degree() + residue_power] =
                Integer::one();
            basis.push(element_from_flat_coordinates(field, &coordinates)?);
        }
    }
    Ok(basis)
}

pub(super) fn element_from_flat_coordinates(
    field: &PadicLocalField,
    coordinates: &[Integer],
) -> Result<PadicLocalFieldElement, PadicLocalOrderError> {
    if coordinates.len() != field.degree() {
        return Err(PadicLocalOrderError::NotFullRank);
    }
    let coefficient_ring = field.coefficient_ring();
    let padic_ring = coefficient_ring.coefficient_ring();
    let qadic_coordinates = coordinates
        .chunks(field.residue_degree())
        .map(|chunk| {
            let coefficients = chunk
                .iter()
                .cloned()
                .map(|value| padic_ring.element(value))
                .collect::<Vec<_>>();
            coefficient_ring
                .element_from_coefficients(&coefficients)
                .map_err(|_| PadicLocalOrderError::NotFullRank)
        })
        .collect::<Result<Vec<_>, _>>()?;
    field
        .integral_element(&qadic_coordinates)
        .map_err(Into::into)
}

pub(super) fn canonical_lattice(
    rows: &[Vec<Integer>],
    prime: &Integer,
    precision: u64,
    degree: usize,
) -> Result<Vec<Vec<Integer>>, PadicLocalOrderError> {
    if rows.iter().any(|row| row.len() != degree) {
        return Err(PadicLocalOrderError::NotFullRank);
    }
    let modulus = prime.pow(precision);
    let mut generators = rows.to_vec();
    generators.extend((0..degree).map(|row| {
        (0..degree)
            .map(|column| {
                if row == column {
                    modulus.clone()
                } else {
                    Integer::zero()
                }
            })
            .collect::<Vec<_>>()
    }));
    let hnf = crate::linalg::Matrix::from_rows(Z, generators)?.hermite_normal_form();
    let mut result = Vec::with_capacity(degree);
    for row in 0..hnf.shape().0 {
        let entries = (0..degree)
            .map(|column| hnf.get(row, column))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| PadicLocalOrderError::NotFullRank)?;
        if entries.iter().any(|entry| !entry.is_zero()) {
            result.push(entries);
        }
    }
    if result.len() != degree {
        return Err(PadicLocalOrderError::NotFullRank);
    }
    Ok(result)
}

pub(super) fn certified_index_valuation(
    canonical_rows: &[Vec<Integer>],
    prime: &Integer,
    precision: u64,
) -> Result<u64, PadicLocalOrderError> {
    let matrix = crate::linalg::Matrix::from_rows(Z, canonical_rows.to_vec())?;
    let smith = matrix.smith_normal_form();
    if smith.rank() != canonical_rows.len() {
        return Err(PadicLocalOrderError::NotFullRank);
    }
    let mut total = 0_u64;
    for invariant in smith.invariant_factors() {
        let exponent = prime_valuation(invariant, prime);
        if exponent >= precision {
            return Err(PadicLocalOrderError::InsufficientPrecision {
                claim: "full rank of the local order basis",
                available: precision,
            });
        }
        total = total
            .checked_add(exponent)
            .ok_or(PadicLocalOrderError::InsufficientPrecision {
                claim: "local order index valuation",
                available: precision,
            })?;
    }
    Ok(total)
}

fn prime_valuation(value: &Integer, prime: &Integer) -> u64 {
    let mut quotient = value.abs();
    let mut valuation = 0_u64;
    while !quotient.is_zero() && quotient.is_divisible(prime) {
        quotient = &quotient / prime;
        valuation += 1;
    }
    valuation
}

fn maximal_different_valuation(field: &PadicLocalField) -> Result<u64, PadicLocalOrderError> {
    let coefficient_ring = field.coefficient_ring();
    let uniformizer = field.uniformizer();
    let mut derivative = field.zero();
    for power in 1..=field.ramification_index() {
        let coefficient = if power == field.ramification_index() {
            coefficient_ring.one()
        } else {
            field.defining_coefficients()[power].clone()
        };
        let scaled = coefficient_ring.mul(&coefficient, &coefficient_ring.element(power as u64))?;
        let scalar = field.integral_element(&[scaled])?;
        let term = field.mul(&scalar, &uniformizer.pow((power - 1) as u64))?;
        derivative = field.add(&derivative, &term)?;
    }
    match derivative.valuation() {
        PadicValuation::Exact(valuation) if valuation >= 0 => {
            u64::try_from(valuation).map_err(|_| PadicLocalOrderError::InsufficientPrecision {
                claim: "maximal-order different valuation",
                available: field.local_precision().absolute_uniformizer_precision(),
            })
        }
        PadicValuation::Exact(_) | PadicValuation::AtLeast(_) => {
            Err(PadicLocalOrderError::InsufficientPrecision {
                claim: "maximal-order different valuation",
                available: field.local_precision().absolute_uniformizer_precision(),
            })
        }
    }
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization(
            "p-adic local order component does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}
