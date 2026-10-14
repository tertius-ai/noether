// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Vectors over explicit coefficient contexts.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    CoefficientMap, CoefficientMappingError, NoetherError, Ring, SemanticType, TypeDescription,
};

use super::{IndexError, LinalgError, Matrix};

static VECTOR_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/Vector@1",
    rust_path: "noether::linalg::Vector",
    python_path: Some("noether.linalg.Vector"),
};

/// An immutable vector over one explicit coefficient context.
///
/// A vector reuses the corresponding one-column [`Matrix<R>`] storage. This
/// keeps generic and specialized representation selection identical for
/// vectors and matrices without making storage part of either public type.
///
/// # Examples
///
/// ```
/// use noether::linalg::Vector;
/// use noether::{Integer, Z};
///
/// let x = Vector::from_values(Z, [Integer::from(2), Integer::from(3)])?;
/// let y = Vector::from_values(Z, [Integer::from(5), Integer::from(7)])?;
/// assert_eq!(x.dot(&y)?, Integer::from(31));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Vector<R: Ring> {
    column: Matrix<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, E: Deserialize<'de>")
)]
struct VectorWire<R: Ring, E> {
    #[serde(rename = "type")]
    type_id: String,
    ring: R,
    values: Vec<E>,
}

impl<R: Ring> Vector<R> {
    /// Construct a vector from coefficients in deterministic order.
    pub fn from_values<I>(ring: R, values: I) -> Result<Self, LinalgError>
    where
        I: IntoIterator<Item = R::Element>,
    {
        let values: Vec<R::Element> = values.into_iter().collect();
        let column = if values.is_empty() {
            Matrix::zeros(ring, 0, 1)
        } else {
            Matrix::from_rows(ring, values.into_iter().map(|value| [value]))?
        };
        Ok(Self { column })
    }

    /// Construct the zero vector of the requested dimension.
    pub fn zeros(ring: R, dimension: usize) -> Self {
        Self {
            column: Matrix::zeros(ring, dimension, 1),
        }
    }

    /// The exact coefficient context.
    pub fn ring(&self) -> &R {
        self.column.ring()
    }

    /// Number of coefficients.
    pub fn dimension(&self) -> usize {
        self.column.shape().0
    }

    /// Materialize one coefficient, failing when the index is out of bounds.
    pub fn get(&self, index: usize) -> Result<R::Element, IndexError> {
        self.column.get(index, 0)
    }

    /// Iterate over materialized coefficients in deterministic order.
    pub fn values(&self) -> impl Iterator<Item = R::Element> + '_ {
        self.column.rows().filter_map(|row| row.into_iter().next())
    }

    pub(crate) fn into_values(self) -> Vec<R::Element> {
        self.column.into_entries()
    }

    /// Apply one checked scalar map coefficientwise while preserving dimension.
    pub fn map_coefficients<M>(&self, map: &M) -> Result<Vector<M::Target>, CoefficientMappingError>
    where
        M: CoefficientMap<Source = R>,
        R: fmt::Display,
    {
        Ok(Vector {
            column: self.column.map_coefficients(map)?,
        })
    }

    /// Materialize all coefficients in deterministic order.
    ///
    /// This is the eager vector counterpart of [`Matrix::to_rows`].
    pub fn to_values(&self) -> Vec<R::Element> {
        self.column.materialized_entries()
    }

    /// Serialize the complete coefficient context and ordered values.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a vector and its complete coefficient context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Convert explicitly to a one-column matrix.
    pub fn to_column_matrix(&self) -> Matrix<R> {
        self.column.clone()
    }

    /// Convert explicitly to a one-row matrix.
    pub fn to_row_matrix(&self) -> Matrix<R> {
        self.column.transpose()
    }

    /// Add vectors with equal contexts and dimensions.
    pub fn add(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.require_same_dimension(rhs, "vector addition")?;
        Ok(Self {
            column: self.column.add(&rhs.column)?,
        })
    }

    /// Subtract vectors with equal contexts and dimensions.
    pub fn sub(&self, rhs: &Self) -> Result<Self, LinalgError> {
        self.require_same_dimension(rhs, "vector subtraction")?;
        Ok(Self {
            column: self.column.sub(&rhs.column)?,
        })
    }

    /// Negate every coefficient.
    pub fn neg(&self) -> Result<Self, LinalgError> {
        Ok(Self {
            column: self.column.neg()?,
        })
    }

    /// Multiply every coefficient by a scalar in this context.
    pub fn scalar_mul(&self, scalar: &R::Element) -> Result<Self, LinalgError> {
        Ok(Self {
            column: self.column.scalar_mul(scalar)?,
        })
    }

    /// Compute the exact bilinear dot product.
    pub fn dot(&self, rhs: &Self) -> Result<R::Element, LinalgError> {
        self.require_same_dimension(rhs, "dot product")?;
        let product = self.column.transpose().matmul(&rhs.column)?;
        product.get(0, 0).map_err(|_| LinalgError::ShapeMismatch {
            operation: "dot product result",
            left: product.shape(),
            right: (1, 1),
        })
    }

    pub(super) fn from_column_matrix(column: Matrix<R>) -> Result<Self, LinalgError> {
        if column.shape().1 == 1 {
            Ok(Self { column })
        } else {
            Err(LinalgError::ShapeMismatch {
                operation: "column matrix to vector conversion",
                left: column.shape(),
                right: (column.shape().0, 1),
            })
        }
    }

    pub(super) fn column_matrix(&self) -> &Matrix<R> {
        &self.column
    }

    fn require_same_dimension(
        &self,
        rhs: &Self,
        operation: &'static str,
    ) -> Result<(), LinalgError> {
        if self.dimension() == rhs.dimension() {
            Ok(())
        } else {
            Err(LinalgError::DimensionMismatch {
                operation,
                left: self.dimension(),
                right: rhs.dimension(),
            })
        }
    }
}

impl<R> Serialize for Vector<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let values = self.to_values();
        let mut state = serializer.serialize_struct("Vector", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", self.ring())?;
        state.serialize_field("values", &values)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for Vector<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = VectorWire::<R, R::Element>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::from_values(wire.ring, wire.values).map_err(serde::de::Error::custom)
    }
}

impl<R: Ring> fmt::Debug for Vector<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vector")
            .field("ring_type", &R::TYPE_ID)
            .field("dimension", &self.dimension())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for Vector<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vector<{}>({})", R::TYPE_ID, self.dimension())
    }
}

impl<R: Ring> SemanticType for Vector<R> {
    const TYPE_ID: &'static str = "noether/linalg/Vector@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.column.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &VECTOR_DESCRIPTION
    }
}

impl<R: Ring> Add<&Vector<R>> for &Vector<R> {
    type Output = Result<Vector<R>, LinalgError>;

    fn add(self, rhs: &Vector<R>) -> Self::Output {
        Vector::add(self, rhs)
    }
}

impl<R: Ring> Sub<&Vector<R>> for &Vector<R> {
    type Output = Result<Vector<R>, LinalgError>;

    fn sub(self, rhs: &Vector<R>) -> Self::Output {
        Vector::sub(self, rhs)
    }
}

impl<R: Ring> Neg for &Vector<R> {
    type Output = Result<Vector<R>, LinalgError>;

    fn neg(self) -> Self::Output {
        Vector::neg(self)
    }
}

impl<R: Ring> Mul<&R::Element> for &Vector<R> {
    type Output = Result<Vector<R>, LinalgError>;

    fn mul(self, rhs: &R::Element) -> Self::Output {
        self.scalar_mul(rhs)
    }
}

impl<R: Ring> Mul<R::Element> for Vector<R> {
    type Output = Result<Vector<R>, LinalgError>;

    fn mul(self, rhs: R::Element) -> Self::Output {
        self.scalar_mul(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::ResidueRing;

    #[test]
    fn zero_dimensional_packed_vectors_have_zero_dot_product() {
        for modulus in [4_u8, 8_u8] {
            let ring = ResidueRing::new(modulus).unwrap();
            let vector = Vector::zeros(ring.clone(), 0);
            assert_eq!(vector.dot(&vector).unwrap(), ring.zero());
        }
    }
}
