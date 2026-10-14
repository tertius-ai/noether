// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact affine linear systems over the integers and arbitrary residue rings.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::commalg::ResidueRing;
use crate::{
    Integer, IntegerRing, NoetherError, SemanticType, TypeDescription, VerificationError, Z,
};

use super::{LinalgError, Matrix, Vector};

static INTEGER_SOLUTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/IntegerSolutionModule@1",
    rust_path: "noether::linalg::IntegerSolutionModule",
    python_path: Some("noether.linalg.IntegerSolutionModule"),
};

static INTEGER_OBSTRUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/IntegerInconsistencyCertificate@1",
    rust_path: "noether::linalg::IntegerInconsistencyCertificate",
    python_path: Some("noether.linalg.IntegerInconsistencyCertificate"),
};

static INTEGER_RESULT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/IntegerLinearSystemResult@1",
    rust_path: "noether::linalg::IntegerLinearSystemResult",
    python_path: Some("noether.linalg.IntegerLinearSystemResult"),
};

static CONGRUENCE_SOLUTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/CongruenceSolutionModule@1",
    rust_path: "noether::linalg::CongruenceSolutionModule",
    python_path: Some("noether.linalg.CongruenceSolutionModule"),
};

static CONGRUENCE_OBSTRUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/CongruenceInconsistencyCertificate@1",
    rust_path: "noether::linalg::CongruenceInconsistencyCertificate",
    python_path: Some("noether.linalg.CongruenceInconsistencyCertificate"),
};

static CONGRUENCE_RESULT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/CongruenceSystemResult@1",
    rust_path: "noether::linalg::CongruenceSystemResult",
    python_path: Some("noether.linalg.CongruenceSystemResult"),
};

/// A nonempty affine solution set `x_0 + ker(A)` over the integers.
///
/// Columns of [`Self::homogeneous_generators`] are the unique column-Hermite
/// basis of the full integer kernel. The retained particular solution is one
/// exact lift; it is not advertised as a canonical representative modulo the
/// kernel.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct IntegerSolutionModule {
    operator: Matrix<IntegerRing>,
    rhs: Vector<IntegerRing>,
    particular_solution: Vector<IntegerRing>,
    homogeneous_generators: Matrix<IntegerRing>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerSolutionModuleUnchecked {
    operator: Matrix<IntegerRing>,
    rhs: Vector<IntegerRing>,
    particular_solution: Vector<IntegerRing>,
    homogeneous_generators: Matrix<IntegerRing>,
}

impl<'de> Deserialize<'de> for IntegerSolutionModule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let unchecked = IntegerSolutionModuleUnchecked::deserialize(deserializer)?;
        let result = Self {
            operator: unchecked.operator,
            rhs: unchecked.rhs,
            particular_solution: unchecked.particular_solution,
            homogeneous_generators: unchecked.homogeneous_generators,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl IntegerSolutionModule {
    /// Coefficient matrix `A` in `A*x=b`.
    pub fn operator(&self) -> &Matrix<IntegerRing> {
        &self.operator
    }

    /// Right-hand side `b`.
    pub fn rhs(&self) -> &Vector<IntegerRing> {
        &self.rhs
    }

    /// One exact integer solution.
    pub fn particular_solution(&self) -> &Vector<IntegerRing> {
        &self.particular_solution
    }

    /// Canonical columns generating every solution of `A*x=0`.
    pub fn homogeneous_generators(&self) -> &Matrix<IntegerRing> {
        &self.homogeneous_generators
    }

    /// Rank of the free homogeneous solution lattice.
    pub fn homogeneous_rank(&self) -> usize {
        self.homogeneous_generators.shape().1
    }

    /// Recheck the affine equation and replay the complete Smith calculation.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_integer_solution_equations(self)?;
        match compute_integer_system(&self.operator, &self.rhs).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "the integer linear system can be recomputed",
            }
        })? {
            IntegerLinearSystemResult::Solutions(recomputed)
                if recomputed.homogeneous_generators == self.homogeneous_generators =>
            {
                Ok(())
            }
            _ => Err(VerificationError::InvalidWitness {
                claim: "the retained affine lattice is the complete integer solution set",
            }),
        }
    }
}

/// A Farkas-style divisibility obstruction to an integer linear system.
///
/// If `w` is [`Self::row_combination`], `g` is the gcd of the entries of
/// `w*A`. The certificate proves inconsistency by `g` not dividing `w*b`
/// (with `g=0` interpreted as requiring `w*b=0`).
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct IntegerInconsistencyCertificate {
    operator: Matrix<IntegerRing>,
    rhs: Vector<IntegerRing>,
    row_combination: Vector<IntegerRing>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerInconsistencyCertificateUnchecked {
    operator: Matrix<IntegerRing>,
    rhs: Vector<IntegerRing>,
    row_combination: Vector<IntegerRing>,
}

impl<'de> Deserialize<'de> for IntegerInconsistencyCertificate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let unchecked = IntegerInconsistencyCertificateUnchecked::deserialize(deserializer)?;
        let result = Self {
            operator: unchecked.operator,
            rhs: unchecked.rhs,
            row_combination: unchecked.row_combination,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl IntegerInconsistencyCertificate {
    /// Coefficient matrix of the rejected system.
    pub fn operator(&self) -> &Matrix<IntegerRing> {
        &self.operator
    }

    /// Right-hand side of the rejected system.
    pub fn rhs(&self) -> &Vector<IntegerRing> {
        &self.rhs
    }

    /// Integer row combination exposing the failed divisibility condition.
    pub fn row_combination(&self) -> &Vector<IntegerRing> {
        &self.row_combination
    }

    /// Gcd of the coefficients in the combined left-hand side.
    pub fn left_gcd(&self) -> Integer {
        integer_obstruction_values(&self.operator, &self.rhs, &self.row_combination).0
    }

    /// Combined right-hand side `w*b`.
    pub fn combined_rhs(&self) -> Integer {
        integer_obstruction_values(&self.operator, &self.rhs, &self.row_combination).1
    }

    /// Recheck that the retained row combination proves nonexistence.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.row_combination.dimension() != self.operator.shape().0
            || self.rhs.dimension() != self.operator.shape().0
        {
            return Err(VerificationError::InvalidWitness {
                claim: "the integer obstruction has compatible dimensions",
            });
        }
        let (gcd, combined_rhs) =
            integer_obstruction_values(&self.operator, &self.rhs, &self.row_combination);
        if (gcd.is_zero() && combined_rhs.is_zero())
            || (!gcd.is_zero() && combined_rhs.is_divisible(&gcd))
        {
            return Err(VerificationError::InvalidWitness {
                claim: "the combined integer right-hand side violates the left gcd",
            });
        }
        Ok(())
    }
}

/// Exact outcome of solving one integer linear system.
#[derive(Clone, PartialEq, Eq)]
pub enum IntegerLinearSystemResult {
    /// The nonempty affine solution lattice.
    Solutions(IntegerSolutionModule),
    /// A replayable divisibility proof that no integer solution exists.
    Inconsistent(IntegerInconsistencyCertificate),
}

impl IntegerLinearSystemResult {
    /// Whether the system has an integer solution.
    pub fn is_consistent(&self) -> bool {
        matches!(self, Self::Solutions(_))
    }

    /// The affine solution module, when nonempty.
    pub fn solutions(&self) -> Option<&IntegerSolutionModule> {
        match self {
            Self::Solutions(solution) => Some(solution),
            Self::Inconsistent(_) => None,
        }
    }

    /// The inconsistency certificate, when the solution set is empty.
    pub fn obstruction(&self) -> Option<&IntegerInconsistencyCertificate> {
        match self {
            Self::Solutions(_) => None,
            Self::Inconsistent(certificate) => Some(certificate),
        }
    }

    /// Verify the complete solution module or inconsistency obstruction.
    pub fn verify(&self) -> Result<(), VerificationError> {
        match self {
            Self::Solutions(solution) => solution.verify(),
            Self::Inconsistent(certificate) => certificate.verify(),
        }
    }

    /// Serialize the typed result, including its complete replay witness.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        let wire = IntegerResultWire::from(self);
        serde_json::to_string(&wire).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a typed integer-system result.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: IntegerResultWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        let result = Self::try_from(wire)?;
        result.verify()?;
        Ok(result)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IntegerResultWire {
    #[serde(rename = "type")]
    type_id: String,
    outcome: IntegerOutcomeWire,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "status", rename_all = "snake_case")]
enum IntegerOutcomeWire {
    Solutions {
        solution: IntegerSolutionModule,
    },
    Inconsistent {
        certificate: IntegerInconsistencyCertificate,
    },
}

impl From<&IntegerLinearSystemResult> for IntegerResultWire {
    fn from(value: &IntegerLinearSystemResult) -> Self {
        let outcome = match value {
            IntegerLinearSystemResult::Solutions(solution) => IntegerOutcomeWire::Solutions {
                solution: solution.clone(),
            },
            IntegerLinearSystemResult::Inconsistent(certificate) => {
                IntegerOutcomeWire::Inconsistent {
                    certificate: certificate.clone(),
                }
            }
        };
        Self {
            type_id: IntegerLinearSystemResult::TYPE_ID.to_owned(),
            outcome,
        }
    }
}

impl TryFrom<IntegerResultWire> for IntegerLinearSystemResult {
    type Error = NoetherError;

    fn try_from(wire: IntegerResultWire) -> Result<Self, Self::Error> {
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(match wire.outcome {
            IntegerOutcomeWire::Solutions { solution } => Self::Solutions(solution),
            IntegerOutcomeWire::Inconsistent { certificate } => Self::Inconsistent(certificate),
        })
    }
}

/// A nonempty affine solution set to `A*x=b (mod n)`.
///
/// The homogeneous columns are in canonical Howell form (transposed to the
/// column convention) and generate the entire kernel, including torsion.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct CongruenceSolutionModule {
    operator: Matrix<ResidueRing>,
    rhs: Vector<ResidueRing>,
    particular_solution: Vector<ResidueRing>,
    homogeneous_generators: Matrix<ResidueRing>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CongruenceSolutionModuleUnchecked {
    operator: Matrix<ResidueRing>,
    rhs: Vector<ResidueRing>,
    particular_solution: Vector<ResidueRing>,
    homogeneous_generators: Matrix<ResidueRing>,
}

impl<'de> Deserialize<'de> for CongruenceSolutionModule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let unchecked = CongruenceSolutionModuleUnchecked::deserialize(deserializer)?;
        let result = Self {
            operator: unchecked.operator,
            rhs: unchecked.rhs,
            particular_solution: unchecked.particular_solution,
            homogeneous_generators: unchecked.homogeneous_generators,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl CongruenceSolutionModule {
    /// Coefficient matrix `A` in `A*x=b (mod n)`.
    pub fn operator(&self) -> &Matrix<ResidueRing> {
        &self.operator
    }

    /// Right-hand side `b`.
    pub fn rhs(&self) -> &Vector<ResidueRing> {
        &self.rhs
    }

    /// One exact residue-class solution.
    pub fn particular_solution(&self) -> &Vector<ResidueRing> {
        &self.particular_solution
    }

    /// Canonical columns generating the complete homogeneous solution module.
    pub fn homogeneous_generators(&self) -> &Matrix<ResidueRing> {
        &self.homogeneous_generators
    }

    /// Number of canonical Howell generators retained for the kernel.
    pub fn homogeneous_generator_count(&self) -> usize {
        self.homogeneous_generators.shape().1
    }

    /// Recheck the equations and replay the complete lifted integer solve.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_congruence_solution_equations(self)?;
        match compute_congruence_system(&self.operator, &self.rhs).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "the congruence system can be recomputed",
            }
        })? {
            CongruenceSystemResult::Solutions(recomputed)
                if recomputed.homogeneous_generators == self.homogeneous_generators =>
            {
                Ok(())
            }
            _ => Err(VerificationError::InvalidWitness {
                claim: "the retained affine module is the complete congruence solution set",
            }),
        }
    }
}

/// A divisibility obstruction proving a congruence system has no solution.
///
/// For the retained integer row `w`, let `g` be the gcd of `n` and all
/// coefficients of `w*A` (using canonical representatives). A solution would
/// force `g` to divide `w*b`; the certificate records a row for which it does
/// not.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct CongruenceInconsistencyCertificate {
    operator: Matrix<ResidueRing>,
    rhs: Vector<ResidueRing>,
    row_combination: Vector<IntegerRing>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CongruenceInconsistencyCertificateUnchecked {
    operator: Matrix<ResidueRing>,
    rhs: Vector<ResidueRing>,
    row_combination: Vector<IntegerRing>,
}

impl<'de> Deserialize<'de> for CongruenceInconsistencyCertificate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let unchecked = CongruenceInconsistencyCertificateUnchecked::deserialize(deserializer)?;
        let result = Self {
            operator: unchecked.operator,
            rhs: unchecked.rhs,
            row_combination: unchecked.row_combination,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl CongruenceInconsistencyCertificate {
    /// Coefficient matrix of the rejected congruence system.
    pub fn operator(&self) -> &Matrix<ResidueRing> {
        &self.operator
    }

    /// Right-hand side of the rejected congruence system.
    pub fn rhs(&self) -> &Vector<ResidueRing> {
        &self.rhs
    }

    /// Integer row combination exposing the failed congruence divisibility.
    pub fn row_combination(&self) -> &Vector<IntegerRing> {
        &self.row_combination
    }

    /// Gcd of the modulus and combined left-hand coefficients.
    pub fn left_gcd_with_modulus(&self) -> Integer {
        congruence_obstruction_values(&self.operator, &self.rhs, &self.row_combination).0
    }

    /// Combined right-hand side using canonical representatives.
    pub fn combined_rhs(&self) -> Integer {
        congruence_obstruction_values(&self.operator, &self.rhs, &self.row_combination).1
    }

    /// Recheck that the retained row proves nonexistence modulo the modulus.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.row_combination.dimension() != self.operator.shape().0
            || self.rhs.dimension() != self.operator.shape().0
            || self.operator.ring() != self.rhs.ring()
        {
            return Err(VerificationError::InvalidWitness {
                claim: "the congruence obstruction has compatible dimensions and contexts",
            });
        }
        let (gcd, combined_rhs) =
            congruence_obstruction_values(&self.operator, &self.rhs, &self.row_combination);
        if combined_rhs.is_divisible(&gcd) {
            return Err(VerificationError::InvalidWitness {
                claim: "the combined congruence right-hand side violates the left gcd",
            });
        }
        Ok(())
    }
}

/// Exact outcome of solving one system over an arbitrary `Z/nZ`.
#[derive(Clone, PartialEq, Eq)]
pub enum CongruenceSystemResult {
    /// The nonempty affine solution module.
    Solutions(CongruenceSolutionModule),
    /// A replayable divisibility proof that no solution exists.
    Inconsistent(CongruenceInconsistencyCertificate),
}

impl CongruenceSystemResult {
    /// Whether the system has a solution modulo its exact modulus.
    pub fn is_consistent(&self) -> bool {
        matches!(self, Self::Solutions(_))
    }

    /// The affine solution module, when nonempty.
    pub fn solutions(&self) -> Option<&CongruenceSolutionModule> {
        match self {
            Self::Solutions(solution) => Some(solution),
            Self::Inconsistent(_) => None,
        }
    }

    /// The inconsistency certificate, when the solution set is empty.
    pub fn obstruction(&self) -> Option<&CongruenceInconsistencyCertificate> {
        match self {
            Self::Solutions(_) => None,
            Self::Inconsistent(certificate) => Some(certificate),
        }
    }

    /// Verify the complete solution module or inconsistency obstruction.
    pub fn verify(&self) -> Result<(), VerificationError> {
        match self {
            Self::Solutions(solution) => solution.verify(),
            Self::Inconsistent(certificate) => certificate.verify(),
        }
    }

    /// Serialize the typed result, including its complete replay witness.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&CongruenceResultWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a typed congruence-system result.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: CongruenceResultWire = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        let result = Self::try_from(wire)?;
        result.verify()?;
        Ok(result)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CongruenceResultWire {
    #[serde(rename = "type")]
    type_id: String,
    outcome: CongruenceOutcomeWire,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields, tag = "status", rename_all = "snake_case")]
enum CongruenceOutcomeWire {
    Solutions {
        solution: CongruenceSolutionModule,
    },
    Inconsistent {
        certificate: CongruenceInconsistencyCertificate,
    },
}

impl From<&CongruenceSystemResult> for CongruenceResultWire {
    fn from(value: &CongruenceSystemResult) -> Self {
        let outcome = match value {
            CongruenceSystemResult::Solutions(solution) => CongruenceOutcomeWire::Solutions {
                solution: solution.clone(),
            },
            CongruenceSystemResult::Inconsistent(certificate) => {
                CongruenceOutcomeWire::Inconsistent {
                    certificate: certificate.clone(),
                }
            }
        };
        Self {
            type_id: CongruenceSystemResult::TYPE_ID.to_owned(),
            outcome,
        }
    }
}

impl TryFrom<CongruenceResultWire> for CongruenceSystemResult {
    type Error = NoetherError;

    fn try_from(wire: CongruenceResultWire) -> Result<Self, Self::Error> {
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Ok(match wire.outcome {
            CongruenceOutcomeWire::Solutions { solution } => Self::Solutions(solution),
            CongruenceOutcomeWire::Inconsistent { certificate } => Self::Inconsistent(certificate),
        })
    }
}

impl Matrix<IntegerRing> {
    /// Solve `self*x=rhs` over the integers, including a complete kernel lattice.
    pub fn solve_exact(
        &self,
        rhs: &Vector<IntegerRing>,
    ) -> Result<IntegerLinearSystemResult, LinalgError> {
        let result = compute_integer_system(self, rhs)?;
        match &result {
            IntegerLinearSystemResult::Solutions(solution) => {
                verify_integer_solution_equations(solution)?;
            }
            IntegerLinearSystemResult::Inconsistent(certificate) => certificate.verify()?,
        }
        Ok(result)
    }
}

impl Matrix<ResidueRing> {
    /// Solve `self*x=rhs` over this exact residue ring, including composite moduli.
    ///
    /// The implementation lifts `A*x=b (mod n)` to the integer system
    /// `[A nI] (x,z)=b`. This gives a correctness floor for every
    /// arbitrary-precision modulus without treating nonunits as field pivots.
    pub fn solve_exact(
        &self,
        rhs: &Vector<ResidueRing>,
    ) -> Result<CongruenceSystemResult, LinalgError> {
        let result = compute_congruence_system(self, rhs)?;
        match &result {
            CongruenceSystemResult::Solutions(solution) => {
                verify_congruence_solution_equations(solution)?;
            }
            CongruenceSystemResult::Inconsistent(certificate) => certificate.verify()?,
        }
        Ok(result)
    }
}

fn compute_integer_system(
    operator: &Matrix<IntegerRing>,
    rhs: &Vector<IntegerRing>,
) -> Result<IntegerLinearSystemResult, LinalgError> {
    let (rows, columns) = operator.shape();
    if rhs.dimension() != rows {
        return Err(LinalgError::DimensionMismatch {
            operation: "integer linear solve",
            left: rows,
            right: rhs.dimension(),
        });
    }

    let smith = operator.smith_normal_form_with_transforms()?;
    let transformed_rhs_matrix = smith.left().matmul(&rhs.to_column_matrix())?;
    let rank = smith.smith().rank();
    for row in 0..rows {
        let value = transformed_rhs_matrix
            .get(row, 0)
            .expect("transformed right-hand side has one column");
        let inconsistent = if row < rank {
            let diagonal = smith
                .smith()
                .form()
                .get(row, row)
                .expect("Smith rank lies on the diagonal");
            !value.is_divisible(&diagonal)
        } else {
            !value.is_zero()
        };
        if inconsistent {
            let row_combination = primitive_row(smith.left(), row)?;
            return Ok(IntegerLinearSystemResult::Inconsistent(
                IntegerInconsistencyCertificate {
                    operator: operator.clone(),
                    rhs: rhs.clone(),
                    row_combination,
                },
            ));
        }
    }

    let mut smith_coordinates = vec![Integer::zero(); columns];
    for (index, coordinate) in smith_coordinates.iter_mut().enumerate().take(rank) {
        let diagonal = smith
            .smith()
            .form()
            .get(index, index)
            .expect("Smith rank lies on the diagonal");
        let value = transformed_rhs_matrix
            .get(index, 0)
            .expect("transformed right-hand side has one column");
        *coordinate = &value / &diagonal;
    }
    let particular_matrix = smith
        .right()
        .matmul(&Vector::from_values(Z, smith_coordinates)?.to_column_matrix())?;
    let particular_solution = Vector::from_column_matrix(particular_matrix)?;

    let free_columns = (rank..columns)
        .map(|column| {
            (0..columns)
                .map(|row| {
                    smith
                        .right()
                        .get(row, column)
                        .expect("free Smith column is in bounds")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let raw_kernel = Matrix::from_validated_columns(Z, columns, free_columns);
    let homogeneous_generators = canonical_integer_columns(&raw_kernel);
    Ok(IntegerLinearSystemResult::Solutions(
        IntegerSolutionModule {
            operator: operator.clone(),
            rhs: rhs.clone(),
            particular_solution,
            homogeneous_generators,
        },
    ))
}

fn compute_congruence_system(
    operator: &Matrix<ResidueRing>,
    rhs: &Vector<ResidueRing>,
) -> Result<CongruenceSystemResult, LinalgError> {
    let (rows, columns) = operator.shape();
    if rhs.dimension() != rows {
        return Err(LinalgError::DimensionMismatch {
            operation: "congruence linear solve",
            left: rows,
            right: rhs.dimension(),
        });
    }
    if operator.ring() != rhs.ring() {
        return Err(LinalgError::ContextMismatch(crate::ContextMismatchError {
            expected: operator.ring().to_string(),
            actual: rhs.ring().to_string(),
        }));
    }

    let modulus = operator.ring().modulus().clone();
    let lifted_rows = (0..rows)
        .map(|row| {
            let mut values = (0..columns)
                .map(|column| {
                    operator
                        .get(row, column)
                        .expect("operator entry is in bounds")
                        .representative()
                        .clone()
                })
                .collect::<Vec<_>>();
            values.extend((0..rows).map(|column| {
                if row == column {
                    modulus.clone()
                } else {
                    Integer::zero()
                }
            }));
            values
        })
        .collect::<Vec<_>>();
    let lifted_operator = if rows == 0 {
        Matrix::zeros(Z, 0, columns)
    } else {
        Matrix::from_rows(Z, lifted_rows)?
    };
    let lifted_rhs = Vector::from_values(
        Z,
        rhs.values()
            .map(|value| value.representative().clone())
            .collect::<Vec<_>>(),
    )?;

    match compute_integer_system(&lifted_operator, &lifted_rhs)? {
        IntegerLinearSystemResult::Inconsistent(certificate) => Ok(
            CongruenceSystemResult::Inconsistent(CongruenceInconsistencyCertificate {
                operator: operator.clone(),
                rhs: rhs.clone(),
                row_combination: certificate.row_combination,
            }),
        ),
        IntegerLinearSystemResult::Solutions(solution) => {
            let ring = operator.ring().clone();
            let particular_solution = Vector::from_values(
                ring.clone(),
                solution
                    .particular_solution
                    .values()
                    .take(columns)
                    .map(|value| ring.element(value))
                    .collect::<Vec<_>>(),
            )?;
            let integer_kernel = solution.homogeneous_generators;
            let generator_count = integer_kernel.shape().1;
            let columns_values = (0..generator_count)
                .map(|column| {
                    (0..columns)
                        .map(|row| {
                            ring.element(
                                integer_kernel
                                    .get(row, column)
                                    .expect("lifted kernel projection is in bounds"),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let raw_kernel = Matrix::from_validated_columns(ring.clone(), columns, columns_values);
            let homogeneous_generators = canonical_congruence_columns(&raw_kernel)?;
            Ok(CongruenceSystemResult::Solutions(
                CongruenceSolutionModule {
                    operator: operator.clone(),
                    rhs: rhs.clone(),
                    particular_solution,
                    homogeneous_generators,
                },
            ))
        }
    }
}

pub(crate) fn canonical_integer_columns(matrix: &Matrix<IntegerRing>) -> Matrix<IntegerRing> {
    let (_, columns) = matrix.shape();
    if columns == 0 {
        return Matrix::zeros(Z, matrix.shape().0, 0);
    }
    let row_hnf = matrix.transpose().hermite_normal_form();
    let rows = row_hnf
        .rows()
        .filter(|row| row.iter().any(|entry| !entry.is_zero()))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        Matrix::zeros(Z, matrix.shape().0, 0)
    } else {
        Matrix::from_rows(Z, rows)
            .expect("Hermite rows have one common width")
            .transpose()
    }
}

pub(crate) fn canonical_congruence_columns(
    matrix: &Matrix<ResidueRing>,
) -> Result<Matrix<ResidueRing>, LinalgError> {
    let ring = matrix.ring().clone();
    if matrix.shape().1 == 0 {
        return Ok(Matrix::zeros(ring, matrix.shape().0, 0));
    }
    let row_howell = matrix.transpose().howell_normal_form()?;
    let rows = row_howell
        .rows()
        .filter(|row| row.iter().any(|entry| !entry.representative().is_zero()))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        Ok(Matrix::zeros(ring, matrix.shape().0, 0))
    } else {
        Ok(Matrix::from_rows(ring, rows)?.transpose())
    }
}

fn primitive_row(
    matrix: &Matrix<IntegerRing>,
    row: usize,
) -> Result<Vector<IntegerRing>, LinalgError> {
    let mut values = matrix
        .to_rows()
        .into_iter()
        .nth(row)
        .expect("requested Smith row is in bounds");
    let content = values
        .iter()
        .fold(Integer::zero(), |gcd, value| gcd.gcd(value));
    if !content.is_zero() && !content.is_one() {
        for value in &mut values {
            *value = &*value / &content;
        }
    }
    if values
        .iter()
        .find(|value| !value.is_zero())
        .is_some_and(Integer::is_negative)
    {
        for value in &mut values {
            *value = -&*value;
        }
    }
    Vector::from_values(Z, values)
}

fn verify_integer_solution_equations(
    solution: &IntegerSolutionModule,
) -> Result<(), VerificationError> {
    let (rows, columns) = solution.operator.shape();
    if solution.rhs.dimension() != rows
        || solution.particular_solution.dimension() != columns
        || solution.homogeneous_generators.shape().0 != columns
    {
        return Err(VerificationError::InvalidWitness {
            claim: "the integer affine solution module has compatible dimensions",
        });
    }
    let image = solution
        .operator
        .matmul(&solution.particular_solution.to_column_matrix())
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "the integer particular solution can be applied",
        })?;
    if image != solution.rhs.to_column_matrix() {
        return Err(VerificationError::InvalidWitness {
            claim: "the integer particular solution satisfies A*x=b",
        });
    }
    let kernel_image = solution
        .operator
        .matmul(&solution.homogeneous_generators)
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "the integer homogeneous generators can be applied",
        })?;
    if kernel_image != Matrix::zeros(Z, rows, solution.homogeneous_rank()) {
        return Err(VerificationError::InvalidWitness {
            claim: "every integer homogeneous generator lies in the kernel",
        });
    }
    if canonical_integer_columns(&solution.homogeneous_generators)
        != solution.homogeneous_generators
    {
        return Err(VerificationError::InvalidWitness {
            claim: "integer homogeneous generators are in canonical column-Hermite form",
        });
    }
    Ok(())
}

fn verify_congruence_solution_equations(
    solution: &CongruenceSolutionModule,
) -> Result<(), VerificationError> {
    let (rows, columns) = solution.operator.shape();
    if solution.rhs.dimension() != rows
        || solution.particular_solution.dimension() != columns
        || solution.homogeneous_generators.shape().0 != columns
        || solution.operator.ring() != solution.rhs.ring()
        || solution.operator.ring() != solution.particular_solution.ring()
        || solution.operator.ring() != solution.homogeneous_generators.ring()
    {
        return Err(VerificationError::InvalidWitness {
            claim: "the congruence affine solution module has compatible dimensions and contexts",
        });
    }
    let image = solution
        .operator
        .matmul(&solution.particular_solution.to_column_matrix())
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "the congruence particular solution can be applied",
        })?;
    if image != solution.rhs.to_column_matrix() {
        return Err(VerificationError::InvalidWitness {
            claim: "the congruence particular solution satisfies A*x=b",
        });
    }
    let kernel_image = solution
        .operator
        .matmul(&solution.homogeneous_generators)
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "the congruence homogeneous generators can be applied",
        })?;
    if kernel_image
        != Matrix::zeros(
            solution.operator.ring().clone(),
            rows,
            solution.homogeneous_generator_count(),
        )
    {
        return Err(VerificationError::InvalidWitness {
            claim: "every congruence homogeneous generator lies in the kernel",
        });
    }
    let canonical =
        canonical_congruence_columns(&solution.homogeneous_generators).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "congruence homogeneous generators admit Howell reduction",
            }
        })?;
    if canonical != solution.homogeneous_generators {
        return Err(VerificationError::InvalidWitness {
            claim: "congruence homogeneous generators are in canonical column-Howell form",
        });
    }
    Ok(())
}

fn integer_obstruction_values(
    operator: &Matrix<IntegerRing>,
    rhs: &Vector<IntegerRing>,
    weights: &Vector<IntegerRing>,
) -> (Integer, Integer) {
    let (_, columns) = operator.shape();
    let mut gcd = Integer::zero();
    for column in 0..columns {
        let mut coefficient = Integer::zero();
        for (row, weight) in weights.values().enumerate() {
            let entry = operator
                .get(row, column)
                .expect("certificate dimensions are checked");
            coefficient += weight * entry;
        }
        gcd = gcd.gcd(&coefficient);
    }
    let combined_rhs = weights
        .values()
        .zip(rhs.values())
        .fold(Integer::zero(), |sum, (weight, value)| sum + weight * value);
    (gcd.abs(), combined_rhs)
}

fn congruence_obstruction_values(
    operator: &Matrix<ResidueRing>,
    rhs: &Vector<ResidueRing>,
    weights: &Vector<IntegerRing>,
) -> (Integer, Integer) {
    let (_, columns) = operator.shape();
    let mut gcd = operator.ring().modulus().clone();
    for column in 0..columns {
        let mut coefficient = Integer::zero();
        for (row, weight) in weights.values().enumerate() {
            let entry = operator
                .get(row, column)
                .expect("certificate dimensions are checked");
            coefficient += weight * entry.representative();
        }
        gcd = gcd.gcd(&coefficient);
    }
    let combined_rhs = weights
        .values()
        .zip(rhs.values())
        .fold(Integer::zero(), |sum, (weight, value)| {
            sum + weight * value.representative()
        });
    (gcd.abs(), combined_rhs)
}

fn push_frame(output: &mut Vec<u8>, value: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(value.len()).map_err(|_| {
        NoetherError::Serialization(
            "exact linear-system canonical serialization exceeded u64".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}

macro_rules! impl_semantic {
    ($ty:ty, $id:literal, $description:ident, [$($field:ident),+ $(,)?]) => {
        impl SemanticType for $ty {
            const TYPE_ID: &'static str = $id;

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                let mut output = Vec::new();
                $(push_frame(&mut output, &self.$field.canonical_bytes()?)?;)+
                Ok(output)
            }

            fn describe() -> &'static TypeDescription {
                &$description
            }
        }
    };
}

impl_semantic!(
    IntegerSolutionModule,
    "noether/linalg/IntegerSolutionModule@1",
    INTEGER_SOLUTION_DESCRIPTION,
    [operator, rhs, particular_solution, homogeneous_generators]
);
impl_semantic!(
    IntegerInconsistencyCertificate,
    "noether/linalg/IntegerInconsistencyCertificate@1",
    INTEGER_OBSTRUCTION_DESCRIPTION,
    [operator, rhs, row_combination]
);
impl_semantic!(
    CongruenceSolutionModule,
    "noether/linalg/CongruenceSolutionModule@1",
    CONGRUENCE_SOLUTION_DESCRIPTION,
    [operator, rhs, particular_solution, homogeneous_generators]
);
impl_semantic!(
    CongruenceInconsistencyCertificate,
    "noether/linalg/CongruenceInconsistencyCertificate@1",
    CONGRUENCE_OBSTRUCTION_DESCRIPTION,
    [operator, rhs, row_combination]
);

impl SemanticType for IntegerLinearSystemResult {
    const TYPE_ID: &'static str = "noether/linalg/IntegerLinearSystemResult@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let (tag, bytes) = match self {
            Self::Solutions(solution) => (0, solution.canonical_bytes()?),
            Self::Inconsistent(certificate) => (1, certificate.canonical_bytes()?),
        };
        let mut output = vec![tag];
        push_frame(&mut output, &bytes)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &INTEGER_RESULT_DESCRIPTION
    }
}

impl SemanticType for CongruenceSystemResult {
    const TYPE_ID: &'static str = "noether/linalg/CongruenceSystemResult@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let (tag, bytes) = match self {
            Self::Solutions(solution) => (0, solution.canonical_bytes()?),
            Self::Inconsistent(certificate) => (1, certificate.canonical_bytes()?),
        };
        let mut output = vec![tag];
        push_frame(&mut output, &bytes)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CONGRUENCE_RESULT_DESCRIPTION
    }
}

impl fmt::Debug for IntegerSolutionModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegerSolutionModule")
            .field("shape", &self.operator.shape())
            .field("homogeneous_rank", &self.homogeneous_rank())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for IntegerInconsistencyCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegerInconsistencyCertificate")
            .field("shape", &self.operator.shape())
            .field("left_gcd", &self.left_gcd())
            .field("combined_rhs", &self.combined_rhs())
            .finish()
    }
}

impl fmt::Debug for IntegerLinearSystemResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solutions(solution) => solution.fmt(formatter),
            Self::Inconsistent(certificate) => certificate.fmt(formatter),
        }
    }
}

impl fmt::Debug for CongruenceSolutionModule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CongruenceSolutionModule")
            .field("shape", &self.operator.shape())
            .field("modulus", self.operator.ring().modulus())
            .field(
                "homogeneous_generators",
                &self.homogeneous_generator_count(),
            )
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for CongruenceInconsistencyCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CongruenceInconsistencyCertificate")
            .field("shape", &self.operator.shape())
            .field("modulus", self.operator.ring().modulus())
            .field("left_gcd", &self.left_gcd_with_modulus())
            .field("combined_rhs", &self.combined_rhs())
            .finish()
    }
}

impl fmt::Debug for CongruenceSystemResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solutions(solution) => solution.fmt(formatter),
            Self::Inconsistent(certificate) => certificate.fmt(formatter),
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::commalg::ResidueRing;
    use crate::{Integer, Ring, Z};

    use super::*;

    #[test]
    fn rank_one_integer_systems_have_hand_checkable_answers_and_obstructions() {
        let operator = Matrix::from_rows(Z, [[Integer::from(6)]]).unwrap();
        let soluble = operator
            .solve_exact(&Vector::from_values(Z, [Integer::from(12)]).unwrap())
            .unwrap();
        let solution = soluble.solutions().unwrap();
        assert_eq!(
            solution.particular_solution().to_values(),
            [Integer::from(2)]
        );
        assert_eq!(solution.homogeneous_rank(), 0);
        soluble.verify().unwrap();

        let impossible = operator
            .solve_exact(&Vector::from_values(Z, [Integer::from(9)]).unwrap())
            .unwrap();
        let certificate = impossible.obstruction().unwrap();
        assert_eq!(certificate.left_gcd(), Integer::from(6));
        assert_eq!(certificate.combined_rhs().abs(), Integer::from(9));
        impossible.verify().unwrap();
    }

    #[test]
    fn underdetermined_integer_system_returns_the_complete_kernel_lattice() {
        let operator =
            Matrix::from_rows(Z, [[Integer::from(2), Integer::from(4), Integer::from(0)]]).unwrap();
        let result = operator
            .solve_exact(&Vector::from_values(Z, [Integer::from(6)]).unwrap())
            .unwrap();
        let solution = result.solutions().unwrap();
        assert_eq!(solution.homogeneous_rank(), 2);
        assert_eq!(
            operator.matmul(solution.homogeneous_generators()).unwrap(),
            Matrix::zeros(Z, 1, 2)
        );
        result.verify().unwrap();
        let restored = IntegerLinearSystemResult::from_json(&result.to_json().unwrap()).unwrap();
        assert_eq!(restored, result);

        let alternate_particular = Vector::from_values(
            Z,
            (0..operator.shape().1).map(|row| {
                solution.particular_solution().get(row).unwrap()
                    + solution.homogeneous_generators().get(row, 0).unwrap()
            }),
        )
        .unwrap();
        let mut alternate: serde_json::Value =
            serde_json::from_str(&result.to_json().unwrap()).unwrap();
        alternate["outcome"]["solution"]["particular_solution"] =
            serde_json::to_value(alternate_particular).unwrap();
        IntegerLinearSystemResult::from_json(&alternate.to_string())
            .unwrap()
            .verify()
            .unwrap();

        let mut noncanonical: serde_json::Value =
            serde_json::from_str(&result.to_json().unwrap()).unwrap();
        noncanonical["outcome"]["solution"]
            .as_object_mut()
            .unwrap()
            .insert("unchecked_note".to_owned(), serde_json::json!(true));
        assert!(IntegerLinearSystemResult::from_json(&noncanonical.to_string()).is_err());
    }

    #[test]
    fn composite_modulus_solve_respects_nonunits() {
        let ring = ResidueRing::new(6).unwrap();
        let operator = Matrix::from_rows(ring.clone(), [[ring.element(4)]]).unwrap();
        let result = operator
            .solve_exact(&Vector::from_values(ring.clone(), [ring.element(2)]).unwrap())
            .unwrap();
        let solution = result.solutions().unwrap();
        assert_eq!(
            operator
                .matmul(&solution.particular_solution().to_column_matrix())
                .unwrap(),
            Matrix::from_rows(ring.clone(), [[ring.element(2)]]).unwrap()
        );
        assert_eq!(solution.homogeneous_generator_count(), 1);
        result.verify().unwrap();
        let alternate_particular = Vector::from_values(
            ring.clone(),
            [ring
                .add(
                    &solution.particular_solution().get(0).unwrap(),
                    &solution.homogeneous_generators().get(0, 0).unwrap(),
                )
                .unwrap()],
        )
        .unwrap();
        let mut alternate: serde_json::Value =
            serde_json::from_str(&result.to_json().unwrap()).unwrap();
        alternate["outcome"]["solution"]["particular_solution"] =
            serde_json::to_value(alternate_particular).unwrap();
        CongruenceSystemResult::from_json(&alternate.to_string())
            .unwrap()
            .verify()
            .unwrap();

        let impossible = Matrix::from_rows(ring.clone(), [[ring.element(2)]])
            .unwrap()
            .solve_exact(&Vector::from_values(ring.clone(), [ring.element(1)]).unwrap())
            .unwrap();
        let certificate = impossible.obstruction().unwrap();
        assert_eq!(certificate.left_gcd_with_modulus(), Integer::from(2));
        assert_eq!(certificate.combined_rhs().abs(), Integer::from(1));
        impossible.verify().unwrap();
        assert_eq!(
            CongruenceSystemResult::from_json(&impossible.to_json().unwrap()).unwrap(),
            impossible
        );
    }

    #[test]
    fn empty_and_thin_systems_fail_closed() {
        let empty = Matrix::zeros(Z, 0, 3)
            .solve_exact(&Vector::zeros(Z, 0))
            .unwrap();
        let solution = empty.solutions().unwrap();
        assert_eq!(solution.particular_solution().dimension(), 3);
        assert_eq!(solution.homogeneous_rank(), 3);

        let zero_row = Matrix::zeros(Z, 1, 0);
        assert!(zero_row
            .solve_exact(&Vector::from_values(Z, [Integer::one()]).unwrap())
            .unwrap()
            .obstruction()
            .is_some());
        assert!(matches!(
            zero_row.solve_exact(&Vector::zeros(Z, 0)),
            Err(LinalgError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn proof_bearing_leaf_deserialization_replays_every_claim() {
        let integer_operator = Matrix::from_rows(Z, [[Integer::from(2)]]).unwrap();
        let integer_solution = integer_operator
            .solve_exact(&Vector::from_values(Z, [Integer::from(4)]).unwrap())
            .unwrap();
        let mut forged_solution =
            serde_json::to_value(integer_solution.solutions().unwrap()).unwrap();
        forged_solution["particular_solution"] =
            serde_json::to_value(Vector::from_values(Z, [Integer::zero()]).unwrap()).unwrap();
        assert!(serde_json::from_value::<IntegerSolutionModule>(forged_solution).is_err());

        let integer_obstruction = integer_operator
            .solve_exact(&Vector::from_values(Z, [Integer::one()]).unwrap())
            .unwrap();
        let mut forged_obstruction =
            serde_json::to_value(integer_obstruction.obstruction().unwrap()).unwrap();
        forged_obstruction["row_combination"] =
            serde_json::to_value(Vector::from_values(Z, [Integer::zero()]).unwrap()).unwrap();
        assert!(
            serde_json::from_value::<IntegerInconsistencyCertificate>(forged_obstruction).is_err()
        );

        let ring = ResidueRing::new(6).unwrap();
        let congruence_operator = Matrix::from_rows(ring.clone(), [[ring.element(2)]]).unwrap();
        let congruence_solution = congruence_operator
            .solve_exact(&Vector::from_values(ring.clone(), [ring.element(2)]).unwrap())
            .unwrap();
        let mut forged_solution =
            serde_json::to_value(congruence_solution.solutions().unwrap()).unwrap();
        forged_solution["particular_solution"] =
            serde_json::to_value(Vector::from_values(ring.clone(), [ring.element(0)]).unwrap())
                .unwrap();
        assert!(serde_json::from_value::<CongruenceSolutionModule>(forged_solution).is_err());

        let congruence_obstruction = congruence_operator
            .solve_exact(&Vector::from_values(ring.clone(), [ring.element(1)]).unwrap())
            .unwrap();
        let mut forged_obstruction =
            serde_json::to_value(congruence_obstruction.obstruction().unwrap()).unwrap();
        forged_obstruction["row_combination"] =
            serde_json::to_value(Vector::from_values(Z, [Integer::zero()]).unwrap()).unwrap();
        assert!(
            serde_json::from_value::<CongruenceInconsistencyCertificate>(forged_obstruction)
                .is_err()
        );
    }

    proptest! {
        #[test]
        fn constructed_integer_solutions_are_never_rejected(
            rows in 0usize..4,
            columns in 0usize..4,
            entries in prop::collection::vec(-5i32..=5, 0..16),
            values in prop::collection::vec(-5i32..=5, 0..4),
        ) {
            let needed = rows * columns;
            let mut entries = entries;
            entries.resize(needed, 0);
            let operator = if rows == 0 {
                Matrix::zeros(Z, 0, columns)
            } else {
                Matrix::from_rows(
                    Z,
                    entries[..needed]
                        .chunks(columns.max(1))
                        .take(rows)
                        .map(|row| row.iter().copied().map(Integer::from).collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                ).unwrap()
            };
            let mut values = values;
            values.resize(columns, 0);
            let witness = Vector::from_values(
                Z,
                values.into_iter().take(columns).map(Integer::from).collect::<Vec<_>>(),
            ).unwrap();
            let rhs_matrix = operator.matmul(&witness.to_column_matrix()).unwrap();
            let rhs = Vector::from_column_matrix(rhs_matrix).unwrap();
            let result = operator.solve_exact(&rhs).unwrap();
            prop_assert!(result.is_consistent());
            prop_assert!(result.verify().is_ok());
        }

        #[test]
        fn rank_one_congruences_match_brute_force(
            modulus in 2u32..16,
            coefficient in 0u32..16,
            rhs_value in 0u32..16,
        ) {
            let ring = ResidueRing::new(modulus).unwrap();
            let operator = Matrix::from_rows(ring.clone(), [[ring.element(coefficient)]]).unwrap();
            let rhs = Vector::from_values(ring.clone(), [ring.element(rhs_value)]).unwrap();
            let result = operator.solve_exact(&rhs).unwrap();
            let brute_nonempty = (0..modulus).any(|value| {
                (coefficient * value) % modulus == rhs_value % modulus
            });
            prop_assert_eq!(result.is_consistent(), brute_nonempty);
            prop_assert!(result.verify().is_ok());
        }
    }
}
