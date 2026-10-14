// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Frobenius-semilinear maps over fixed-cap unramified p-adic contexts.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::numthy::{
    InvalidQadicCoordinatesError, PadicElement, PadicField, PadicRing, QadicElement, QadicField,
    QadicRing,
};
use crate::{
    CoefficientFrobeniusContext, CoefficientMap, CoefficientMappingError, CommutativeRing,
    NoetherError, Ring, SemanticType, TypeDescription, VerificationError,
};

use super::{LinalgError, LinearOperator, Matrix, Vector};

static FROBENIUS_SEMILINEAR_MAP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/FrobeniusSemilinearMap@1",
    rust_path: "noether::linalg::FrobeniusSemilinearMap",
    python_path: Some("noether.linalg.FrobeniusSemilinearMap"),
};

/// Failure to construct, apply, compose, or restrict a semilinear map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemilinearMapError {
    /// Ordinary matrix or vector algebra failed.
    LinearAlgebra(LinalgError),
    /// Applying coefficient Frobenius to a composite value failed.
    CoefficientMapping(CoefficientMappingError),
    /// The composed Frobenius exponent does not fit in `u64`.
    ExponentOverflow {
        /// Exponent of the outer map.
        outer: u64,
        /// Exponent of the inner map.
        inner: u64,
    },
    /// Restriction of scalars could not represent the expanded matrix shape.
    ExpandedShapeTooLarge {
        /// Shape over the q-adic context.
        shape: (usize, usize),
        /// Unramified residue degree used for expansion.
        residue_degree: usize,
    },
    /// A q-adic basis value could not be reconstructed from p-adic coordinates.
    PowerBasis(InvalidQadicCoordinatesError),
}

impl fmt::Display for SemilinearMapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LinearAlgebra(error) => error.fmt(formatter),
            Self::CoefficientMapping(error) => error.fmt(formatter),
            Self::ExponentOverflow { outer, inner } => write!(
                formatter,
                "cannot compose Frobenius-semilinear exponents {outer} and {inner}: their sum exceeds u64"
            ),
            Self::ExpandedShapeTooLarge {
                shape,
                residue_degree,
            } => write!(
                formatter,
                "cannot restrict a {}x{} matrix across residue degree {residue_degree}: the expanded shape is too large",
                shape.0, shape.1
            ),
            Self::PowerBasis(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SemilinearMapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LinearAlgebra(error) => Some(error),
            Self::CoefficientMapping(error) => Some(error),
            Self::PowerBasis(error) => Some(error),
            Self::ExponentOverflow { .. } | Self::ExpandedShapeTooLarge { .. } => None,
        }
    }
}

impl From<LinalgError> for SemilinearMapError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<CoefficientMappingError> for SemilinearMapError {
    fn from(value: CoefficientMappingError) -> Self {
        Self::CoefficientMapping(value)
    }
}

impl From<InvalidQadicCoordinatesError> for SemilinearMapError {
    fn from(value: InvalidQadicCoordinatesError) -> Self {
        Self::PowerBasis(value)
    }
}

/// A map `v -> A sigma^r(v)` between finite free modules over one context.
///
/// Vectors are columns, `A` has shape `target_dimension x source_dimension`,
/// and `sigma` is the context's checked coefficient Frobenius. Composition
/// records the unreduced sum of Frobenius exponents and twists the inner
/// matrix before multiplication.
#[derive(Clone, PartialEq, Eq)]
pub struct FrobeniusSemilinearMap<R: CoefficientFrobeniusContext> {
    matrix: Matrix<R>,
    frobenius_exponent: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "Matrix<R>: Serialize",
        deserialize = "Matrix<R>: Deserialize<'de>"
    )
)]
struct FrobeniusSemilinearMapWire<R: CoefficientFrobeniusContext> {
    #[serde(rename = "type")]
    type_id: String,
    matrix: Matrix<R>,
    frobenius_exponent: u64,
}

impl<R: CoefficientFrobeniusContext> FrobeniusSemilinearMap<R> {
    /// Construct the map `v -> A sigma^r(v)` represented by `matrix` and `r`.
    pub fn new(matrix: Matrix<R>, frobenius_exponent: u64) -> Self {
        Self {
            matrix,
            frobenius_exponent,
        }
    }

    /// Construct the identity map on a free module of the requested dimension.
    pub fn identity(context: R, dimension: usize) -> Self {
        Self::new(Matrix::identity(context, dimension), 0)
    }

    /// Exact coefficient context of the source and target modules.
    pub fn context(&self) -> &R {
        self.matrix.ring()
    }

    /// Matrix `A` in the convention `v -> A sigma^r(v)`.
    pub fn matrix(&self) -> &Matrix<R> {
        &self.matrix
    }

    /// Unreduced nonnegative Frobenius exponent `r`.
    pub fn frobenius_exponent(&self) -> u64 {
        self.frobenius_exponent
    }

    /// Dimension of the source free module.
    pub fn source_dimension(&self) -> usize {
        self.matrix.shape().1
    }

    /// Dimension of the target free module.
    pub fn target_dimension(&self) -> usize {
        self.matrix.shape().0
    }

    /// Recheck the matrix context and dimensions defining this map.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self
            .matrix
            .rows()
            .flatten()
            .all(|coefficient| self.context().contains(&coefficient))
        {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "a Frobenius-semilinear matrix belongs to its declared coefficient context",
            })
        }
    }

    /// Serialize the complete context, matrix, and Frobenius exponent.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        Matrix<R>: Serialize,
    {
        serde_json::to_string(&FrobeniusSemilinearMapWire {
            type_id: Self::TYPE_ID.to_owned(),
            matrix: self.matrix.clone(),
            frobenius_exponent: self.frobenius_exponent,
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a semilinear map and its exact context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        for<'de> Matrix<R>: Deserialize<'de>,
    {
        let wire = serde_json::from_str::<FrobeniusSemilinearMapWire<R>>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self::new(wire.matrix, wire.frobenius_exponent);
        result.verify()?;
        Ok(result)
    }
}

impl<R: CoefficientFrobeniusContext> FrobeniusSemilinearMap<R> {
    /// Apply `v -> A sigma^r(v)` after checking context and dimension.
    pub fn apply(&self, vector: &Vector<R>) -> Result<Vector<R>, SemilinearMapError> {
        let frobenius = self
            .context()
            .coefficient_frobenius(self.frobenius_exponent);
        let twisted = vector.map_coefficients(&frobenius)?;
        Ok(LinearOperator::apply(&self.matrix, &twisted)?)
    }

    /// Apply the semilinear map independently to every column of a dense block.
    pub fn apply_block(&self, block: &Matrix<R>) -> Result<Matrix<R>, SemilinearMapError> {
        let frobenius = self
            .context()
            .coefficient_frobenius(self.frobenius_exponent);
        let twisted = block.map_coefficients(&frobenius)?;
        Ok(self.matrix.matmul(&twisted)?)
    }

    /// Compose `self` after `inner`.
    ///
    /// If `self(v) = A sigma^r(v)` and `inner(v) = B sigma^s(v)`, the result is
    /// `A sigma^r(B) sigma^(r+s)(v)`.
    pub fn compose(&self, inner: &Self) -> Result<Self, SemilinearMapError> {
        let exponent = self
            .frobenius_exponent
            .checked_add(inner.frobenius_exponent)
            .ok_or(SemilinearMapError::ExponentOverflow {
                outer: self.frobenius_exponent,
                inner: inner.frobenius_exponent,
            })?;
        let frobenius = self
            .context()
            .coefficient_frobenius(self.frobenius_exponent);
        let twisted_inner = inner.matrix.map_coefficients(&frobenius)?;
        let matrix = self.matrix.matmul(&twisted_inner)?;
        Ok(Self::new(matrix, exponent))
    }
}

impl FrobeniusSemilinearMap<QadicRing> {
    /// Restrict this semilinear map to an ordinary `Z_p`-linear matrix.
    ///
    /// Module coordinates are flattened in module-major order, with each
    /// q-adic coordinate expanded in ascending power-basis order.
    pub fn restrict_scalars(&self) -> Result<Matrix<PadicRing>, SemilinearMapError> {
        restrict_scalars(self)
    }
}

impl FrobeniusSemilinearMap<QadicField> {
    /// Restrict this semilinear map to an ordinary `Q_p`-linear matrix.
    ///
    /// Module coordinates are flattened in module-major order, with each
    /// q-adic coordinate expanded in ascending power-basis order.
    pub fn restrict_scalars(&self) -> Result<Matrix<PadicField>, SemilinearMapError> {
        restrict_scalars(self)
    }
}

impl<R: CoefficientFrobeniusContext> SemanticType for FrobeniusSemilinearMap<R> {
    const TYPE_ID: &'static str = "noether/linalg/FrobeniusSemilinearMap@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let matrix = self.matrix.canonical_bytes()?;
        let length = u64::try_from(matrix.len()).map_err(|_| {
            NoetherError::Serialization("semilinear matrix is too large to frame".to_owned())
        })?;
        let mut output = Vec::with_capacity(8 + matrix.len() + 8);
        output.extend_from_slice(&length.to_be_bytes());
        output.extend_from_slice(&matrix);
        output.extend_from_slice(&self.frobenius_exponent.to_be_bytes());
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FROBENIUS_SEMILINEAR_MAP_DESCRIPTION
    }
}

impl<R: CoefficientFrobeniusContext> fmt::Debug for FrobeniusSemilinearMap<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrobeniusSemilinearMap")
            .field("matrix", &self.matrix)
            .field("frobenius_exponent", &self.frobenius_exponent)
            .finish()
    }
}

impl<R: CoefficientFrobeniusContext> fmt::Display for FrobeniusSemilinearMap<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Frobenius-semilinear map {} -> {} with exponent {} over {}",
            self.source_dimension(),
            self.target_dimension(),
            self.frobenius_exponent,
            self.context()
        )
    }
}

trait QadicScalarContext: CoefficientFrobeniusContext<Element = QadicElement> + fmt::Display {
    type Base: CommutativeRing<Element = PadicElement> + fmt::Display;

    fn residue_degree(&self) -> usize;
    fn base_context(&self) -> &Self::Base;
    fn element_from_base_coefficients(
        &self,
        coefficients: &[PadicElement],
    ) -> Result<QadicElement, InvalidQadicCoordinatesError>;
}

impl QadicScalarContext for QadicRing {
    type Base = PadicRing;

    fn residue_degree(&self) -> usize {
        self.residue_degree()
    }

    fn base_context(&self) -> &Self::Base {
        self.coefficient_ring()
    }

    fn element_from_base_coefficients(
        &self,
        coefficients: &[PadicElement],
    ) -> Result<QadicElement, InvalidQadicCoordinatesError> {
        self.element_from_coefficients(coefficients)
    }
}

impl QadicScalarContext for QadicField {
    type Base = PadicField;

    fn residue_degree(&self) -> usize {
        self.residue_degree()
    }

    fn base_context(&self) -> &Self::Base {
        self.coefficient_field()
    }

    fn element_from_base_coefficients(
        &self,
        coefficients: &[PadicElement],
    ) -> Result<QadicElement, InvalidQadicCoordinatesError> {
        self.element_from_coefficients(coefficients)
    }
}

fn restrict_scalars<R>(
    semilinear: &FrobeniusSemilinearMap<R>,
) -> Result<Matrix<R::Base>, SemilinearMapError>
where
    R: QadicScalarContext,
{
    let context = semilinear.context();
    let degree = context.residue_degree();
    let (rows, cols) = semilinear.matrix.shape();
    let expanded_rows =
        rows.checked_mul(degree)
            .ok_or(SemilinearMapError::ExpandedShapeTooLarge {
                shape: (rows, cols),
                residue_degree: degree,
            })?;
    let expanded_cols =
        cols.checked_mul(degree)
            .ok_or(SemilinearMapError::ExpandedShapeTooLarge {
                shape: (rows, cols),
                residue_degree: degree,
            })?;
    expanded_rows
        .checked_mul(expanded_cols)
        .ok_or(SemilinearMapError::ExpandedShapeTooLarge {
            shape: (rows, cols),
            residue_degree: degree,
        })?;
    if expanded_rows == 0 || expanded_cols == 0 {
        return Ok(Matrix::zeros(
            context.base_context().clone(),
            expanded_rows,
            expanded_cols,
        ));
    }

    let frobenius = context.coefficient_frobenius(semilinear.frobenius_exponent);
    let mut twisted_basis = Vec::with_capacity(degree);
    for basis_index in 0..degree {
        let mut coefficients = vec![context.base_context().zero(); degree];
        coefficients[basis_index] = context.base_context().one();
        let basis = context.element_from_base_coefficients(&coefficients)?;
        twisted_basis.push(frobenius.apply(&basis).map_err(|error| {
            SemilinearMapError::CoefficientMapping(CoefficientMappingError::Coefficient(
                error.to_string(),
            ))
        })?);
    }

    let mut expanded = vec![vec![context.base_context().zero(); expanded_cols]; expanded_rows];
    for (target_index, row) in semilinear.matrix.rows().enumerate() {
        for (source_index, coefficient) in row.into_iter().enumerate() {
            for (basis_index, basis) in twisted_basis.iter().enumerate() {
                let image = context.mul(&coefficient, basis).map_err(|error| {
                    SemilinearMapError::CoefficientMapping(
                        CoefficientMappingError::ContextMismatch(error),
                    )
                })?;
                for (coordinate_index, coordinate) in image.coefficients().into_iter().enumerate() {
                    expanded[target_index * degree + coordinate_index]
                        [source_index * degree + basis_index] = coordinate;
                }
            }
        }
    }
    Ok(Matrix::from_rows(context.base_context().clone(), expanded)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numthy::{FiniteField, PadicValuation};

    fn qadic_ring() -> QadicRing {
        QadicRing::new(FiniteField::extension(3, 2).unwrap(), 6).unwrap()
    }

    #[test]
    fn application_and_composition_use_the_declared_frobenius_convention() {
        let context = qadic_ring();
        let generator_plus_one = context.add(&context.generator(), &context.one()).unwrap();
        let a =
            Matrix::from_rows(context.clone(), [[generator_plus_one, context.element(2)]]).unwrap();
        let outer = FrobeniusSemilinearMap::new(a, 1);
        let b = Matrix::from_rows(
            context.clone(),
            [[context.generator()], [context.generator().frobenius(1)]],
        )
        .unwrap();
        let inner = FrobeniusSemilinearMap::new(b, 1);
        let vector = Vector::from_values(context.clone(), [context.generator()]).unwrap();

        let expected_inner = inner.apply(&vector).unwrap();
        let expected = outer.apply(&expected_inner).unwrap();
        let composed = outer.compose(&inner).unwrap();
        assert_eq!(composed.frobenius_exponent(), 2);
        assert_eq!(composed.apply(&vector).unwrap(), expected);
        assert_eq!(
            FrobeniusSemilinearMap::<QadicRing>::from_json(&composed.to_json().unwrap()).unwrap(),
            composed
        );
        composed.verify().unwrap();
    }

    #[test]
    fn composition_checks_shapes_contexts_and_exponent_overflow() {
        let context = qadic_ring();
        let identity = FrobeniusSemilinearMap::identity(context.clone(), 2);
        let vector =
            Vector::from_values(context.clone(), [context.generator(), context.element(2)])
                .unwrap();
        assert_eq!(identity.apply(&vector).unwrap(), vector);

        let huge = FrobeniusSemilinearMap::new(Matrix::identity(context.clone(), 2), u64::MAX);
        assert!(matches!(
            huge.compose(&FrobeniusSemilinearMap::new(
                Matrix::identity(context.clone(), 2),
                1
            )),
            Err(SemilinearMapError::ExponentOverflow { .. })
        ));

        let wrong_shape = FrobeniusSemilinearMap::new(Matrix::identity(context.clone(), 3), 0);
        assert!(matches!(
            identity.compose(&wrong_shape),
            Err(SemilinearMapError::LinearAlgebra(
                LinalgError::ShapeMismatch { .. }
            ))
        ));

        let other_context = QadicRing::new(FiniteField::extension(5, 2).unwrap(), 6).unwrap();
        let other = FrobeniusSemilinearMap::identity(other_context, 2);
        assert!(matches!(
            identity.compose(&other),
            Err(SemilinearMapError::CoefficientMapping(
                CoefficientMappingError::ContextMismatch(_)
            ))
        ));
    }

    #[test]
    fn scalar_restriction_agrees_on_every_power_basis_vector() {
        let context = qadic_ring();
        let matrix = Matrix::from_rows(
            context.clone(),
            [
                [context.generator(), context.one()],
                [context.element(2), context.generator().frobenius(1)],
            ],
        )
        .unwrap();
        let semilinear = FrobeniusSemilinearMap::new(matrix, 1);
        let restricted = semilinear.restrict_scalars().unwrap();
        let degree = context.residue_degree();
        assert_eq!(restricted.shape(), (4, 4));

        for module_index in 0..semilinear.source_dimension() {
            for basis_index in 0..degree {
                let mut qadic_coordinates = vec![context.zero(); semilinear.source_dimension()];
                let mut basis_coordinates = vec![context.coefficient_ring().zero(); degree];
                basis_coordinates[basis_index] = context.coefficient_ring().one();
                qadic_coordinates[module_index] = context
                    .element_from_coefficients(&basis_coordinates)
                    .unwrap();
                let qadic_vector = Vector::from_values(context.clone(), qadic_coordinates).unwrap();
                let expected = semilinear
                    .apply(&qadic_vector)
                    .unwrap()
                    .values()
                    .flat_map(|value| value.coefficients())
                    .collect::<Vec<_>>();

                let mut base_coordinates =
                    vec![context.coefficient_ring().zero(); semilinear.source_dimension() * degree];
                base_coordinates[module_index * degree + basis_index] =
                    context.coefficient_ring().one();
                let base_vector =
                    Vector::from_values(context.coefficient_ring().clone(), base_coordinates)
                        .unwrap();
                assert_eq!(
                    LinearOperator::apply(&restricted, &base_vector)
                        .unwrap()
                        .values()
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }

    #[test]
    fn field_scalar_restriction_preserves_negative_valuation_entries() {
        let field = QadicField::new(FiniteField::extension(3, 2).unwrap(), 6).unwrap();
        let inverse = field.inverse(&field.element(3)).unwrap();
        let matrix = Matrix::from_rows(field.clone(), [[inverse]]).unwrap();
        let semilinear = FrobeniusSemilinearMap::new(matrix, 1);
        let restricted = semilinear.restrict_scalars().unwrap();

        assert_eq!(restricted.shape(), (2, 2));
        assert!(restricted
            .rows()
            .flatten()
            .any(|coefficient| coefficient.valuation() == PadicValuation::Exact(-1)));
    }
}
