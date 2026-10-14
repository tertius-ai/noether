// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact symmetric functions in the five classical bases.

use core::fmt;
use core::ops::{Add, Mul, Sub};
use core::str::FromStr;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::linalg::{Matrix, Vector};
use crate::{Integer, NoetherError, Rational, SemanticType, TypeDescription, Q};

use super::{CombinError, Partition};

/// Largest homogeneous component materialized by the 0.1 basis-conversion path.
///
/// The partition count is already 77 in degree 12. Exact transition matrices
/// grow quickly after this point, so an explicit error is safer than an
/// unexpectedly large allocation.
pub const MAX_SYMMETRIC_DEGREE: usize = 12;

/// One of the five classical bases of the rational symmetric-function ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymmetricBasis {
    /// Monomial symmetric functions `m_lambda`.
    Monomial,
    /// Power-sum symmetric functions `p_lambda`.
    PowerSum,
    /// Elementary symmetric functions `e_lambda`.
    Elementary,
    /// Complete-homogeneous symmetric functions `h_lambda`.
    CompleteHomogeneous,
    /// Schur symmetric functions `s_lambda`.
    Schur,
}

static SYMMETRIC_BASIS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/SymmetricBasis@1",
    rust_path: "noether::combin::SymmetricBasis",
    python_path: Some("noether.combin.SymmetricBasis"),
};

impl SymmetricBasis {
    /// Canonical stable label used by Python and JSON.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Monomial => "monomial",
            Self::PowerSum => "power_sum",
            Self::Elementary => "elementary",
            Self::CompleteHomogeneous => "complete_homogeneous",
            Self::Schur => "schur",
        }
    }
}

impl fmt::Display for SymmetricBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl FromStr for SymmetricBasis {
    type Err = CombinError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "monomial" => Ok(Self::Monomial),
            "power_sum" => Ok(Self::PowerSum),
            "elementary" => Ok(Self::Elementary),
            "complete_homogeneous" => Ok(Self::CompleteHomogeneous),
            "schur" => Ok(Self::Schur),
            basis => Err(CombinError::UnknownSymmetricBasis {
                basis: basis.to_owned(),
            }),
        }
    }
}

impl SemanticType for SymmetricBasis {
    const TYPE_ID: &'static str = "noether/combin/SymmetricBasis@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.label().as_bytes().to_vec())
    }

    fn describe() -> &'static TypeDescription {
        &SYMMETRIC_BASIS_DESCRIPTION
    }
}

/// A homogeneous symmetric function over `Q`, in one explicit classical basis.
///
/// Coefficients use Noether's FLINT-backed [`Rational`] values and are keyed by
/// typed [`Partition`] values. The zero function has no degree. Nonzero
/// constant functions have degree zero, represented by the empty partition.
#[derive(Clone, Debug)]
pub struct SymmetricFunction {
    basis: SymmetricBasis,
    terms: BTreeMap<Partition, Rational>,
    degree: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct SymmetricFunctionRepr {
    basis: SymmetricBasis,
    terms: Vec<SymmetricTermRepr>,
}

#[derive(Serialize, Deserialize)]
struct SymmetricTermRepr {
    partition: Partition,
    coefficient: String,
}

impl SymmetricFunction {
    /// Construct a homogeneous symmetric function from exact rational terms.
    ///
    /// Duplicate partitions are combined and zero coefficients are removed.
    pub fn new<I>(basis: SymmetricBasis, terms: I) -> Result<Self, CombinError>
    where
        I: IntoIterator<Item = (Partition, Rational)>,
    {
        let mut combined = BTreeMap::<Partition, Rational>::new();
        for (partition, coefficient) in terms {
            if coefficient.is_zero() {
                continue;
            }
            let slot = combined.entry(partition).or_insert_with(Rational::zero);
            *slot = &*slot + &coefficient;
        }
        combined.retain(|_, coefficient| !coefficient.is_zero());

        let mut degree = None;
        for partition in combined.keys() {
            let actual = partition.size();
            if let Some(expected) = degree {
                if actual != expected {
                    return Err(CombinError::MixedSymmetricDegrees { expected, actual });
                }
            } else {
                degree = Some(actual);
            }
        }
        if let Some(degree) = degree {
            require_degree(degree)?;
        }
        Ok(Self {
            basis,
            terms: combined,
            degree,
        })
    }

    /// The zero symmetric function in the requested display basis.
    #[must_use]
    pub fn zero(basis: SymmetricBasis) -> Self {
        Self {
            basis,
            terms: BTreeMap::new(),
            degree: None,
        }
    }

    /// The classical basis used by this representation.
    #[must_use]
    pub const fn basis(&self) -> SymmetricBasis {
        self.basis
    }

    /// Homogeneous degree, or `None` for zero.
    #[must_use]
    pub const fn degree(&self) -> Option<usize> {
        self.degree
    }

    /// Nonzero exact terms in deterministic partition order.
    pub fn terms(&self) -> impl ExactSizeIterator<Item = (&Partition, &Rational)> {
        self.terms.iter()
    }

    /// Re-express this value exactly in another classical basis.
    pub fn convert_basis(&self, basis: SymmetricBasis) -> Result<Self, CombinError> {
        if self.basis == basis {
            return Ok(self.clone());
        }
        let Some(degree) = self.degree else {
            return Ok(Self::zero(basis));
        };
        let partitions = partitions_of(degree)?;
        let source_matrix = basis_to_power_matrix(self.basis, degree, &partitions)?;
        let target_matrix = basis_to_power_matrix(basis, degree, &partitions)?;

        let mut power_coordinates = vec![Rational::zero(); partitions.len()];
        for (row, partition) in partitions.iter().enumerate() {
            let Some(coefficient) = self.terms.get(partition) else {
                continue;
            };
            for (column, coordinate) in power_coordinates.iter_mut().enumerate() {
                let entry = source_matrix
                    .get(row, column)
                    .map_err(|error| conversion_error(error.to_string()))?;
                *coordinate = &*coordinate + &(coefficient * &entry);
            }
        }

        let rhs = Vector::from_values(Q, power_coordinates)
            .map_err(|error| conversion_error(error.to_string()))?;
        let solution = target_matrix
            .transpose()
            .solve(&rhs)
            .map_err(|error| conversion_error(error.to_string()))?;
        let terms = partitions
            .into_iter()
            .zip(solution.particular().values())
            .filter(|(_, coefficient)| !coefficient.is_zero());
        Self::new(basis, terms)
    }

    /// Add homogeneous values, returning the result in `self`'s basis.
    pub fn add(&self, rhs: &Self) -> Result<Self, CombinError> {
        require_compatible_degrees(self.degree, rhs.degree)?;
        let rhs = rhs.convert_basis(self.basis)?;
        let mut terms = self.terms.clone();
        for (partition, coefficient) in rhs.terms {
            let slot = terms.entry(partition).or_insert_with(Rational::zero);
            *slot = &*slot + &coefficient;
        }
        Self::new(self.basis, terms)
    }

    /// Subtract homogeneous values, returning the result in `self`'s basis.
    pub fn sub(&self, rhs: &Self) -> Result<Self, CombinError> {
        require_compatible_degrees(self.degree, rhs.degree)?;
        let rhs = rhs.convert_basis(self.basis)?;
        let mut terms = self.terms.clone();
        for (partition, coefficient) in rhs.terms {
            let slot = terms.entry(partition).or_insert_with(Rational::zero);
            *slot = &*slot - &coefficient;
        }
        Self::new(self.basis, terms)
    }

    /// Multiply values in the symmetric-function ring, returning `self`'s basis.
    pub fn mul(&self, rhs: &Self) -> Result<Self, CombinError> {
        let (Some(left_degree), Some(right_degree)) = (self.degree, rhs.degree) else {
            return Ok(Self::zero(self.basis));
        };
        let degree =
            left_degree
                .checked_add(right_degree)
                .ok_or(CombinError::SymmetricDegreeTooLarge {
                    degree: usize::MAX,
                    max: MAX_SYMMETRIC_DEGREE,
                })?;
        require_degree(degree)?;

        let left = self.convert_basis(SymmetricBasis::PowerSum)?;
        let right = rhs.convert_basis(SymmetricBasis::PowerSum)?;
        let mut terms = BTreeMap::<Partition, Rational>::new();
        for (lambda, left_coefficient) in left.terms {
            for (mu, right_coefficient) in &right.terms {
                let partition = partition_union(&lambda, mu)?;
                let slot = terms.entry(partition).or_insert_with(Rational::zero);
                *slot = &*slot + &(&left_coefficient * right_coefficient);
            }
        }
        Self::new(SymmetricBasis::PowerSum, terms)?.convert_basis(self.basis)
    }
}

impl PartialEq for SymmetricFunction {
    fn eq(&self, other: &Self) -> bool {
        if self.degree != other.degree {
            return false;
        }
        if self.basis == other.basis {
            return self.terms == other.terms;
        }
        other
            .convert_basis(self.basis)
            .is_ok_and(|converted| self.terms == converted.terms)
    }
}

impl Eq for SymmetricFunction {}

impl fmt::Display for SymmetricFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SymmetricFunction(basis={}, degree={:?}, terms={})",
            self.basis,
            self.degree,
            self.terms.len()
        )
    }
}

impl Serialize for SymmetricFunction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SymmetricFunctionRepr {
            basis: self.basis,
            terms: self
                .terms
                .iter()
                .map(|(partition, coefficient)| SymmetricTermRepr {
                    partition: partition.clone(),
                    coefficient: coefficient.to_string(),
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SymmetricFunction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = SymmetricFunctionRepr::deserialize(deserializer)?;
        let terms = repr
            .terms
            .into_iter()
            .map(|term| {
                term.coefficient
                    .parse::<Rational>()
                    .map(|coefficient| (term.partition, coefficient))
                    .map_err(serde::de::Error::custom)
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        Self::new(repr.basis, terms).map_err(serde::de::Error::custom)
    }
}

static SYMMETRIC_FUNCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/SymmetricFunction@1",
    rust_path: "noether::combin::SymmetricFunction",
    python_path: Some("noether.combin.SymmetricFunction"),
};

impl SemanticType for SymmetricFunction {
    const TYPE_ID: &'static str = "noether/combin/SymmetricFunction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let canonical = self
            .convert_basis(SymmetricBasis::PowerSum)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        serde_json::to_vec(&canonical)
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &SYMMETRIC_FUNCTION_DESCRIPTION
    }
}

impl Add<&SymmetricFunction> for &SymmetricFunction {
    type Output = Result<SymmetricFunction, CombinError>;

    fn add(self, rhs: &SymmetricFunction) -> Self::Output {
        SymmetricFunction::add(self, rhs)
    }
}

impl Sub<&SymmetricFunction> for &SymmetricFunction {
    type Output = Result<SymmetricFunction, CombinError>;

    fn sub(self, rhs: &SymmetricFunction) -> Self::Output {
        SymmetricFunction::sub(self, rhs)
    }
}

impl Mul<&SymmetricFunction> for &SymmetricFunction {
    type Output = Result<SymmetricFunction, CombinError>;

    fn mul(self, rhs: &SymmetricFunction) -> Self::Output {
        SymmetricFunction::mul(self, rhs)
    }
}

fn require_degree(degree: usize) -> Result<(), CombinError> {
    if degree <= MAX_SYMMETRIC_DEGREE {
        Ok(())
    } else {
        Err(CombinError::SymmetricDegreeTooLarge {
            degree,
            max: MAX_SYMMETRIC_DEGREE,
        })
    }
}

fn require_compatible_degrees(
    left: Option<usize>,
    right: Option<usize>,
) -> Result<(), CombinError> {
    match (left, right) {
        (Some(expected), Some(actual)) if expected != actual => {
            Err(CombinError::MixedSymmetricDegrees { expected, actual })
        }
        _ => Ok(()),
    }
}

fn conversion_error(detail: String) -> CombinError {
    CombinError::SymmetricBasisConversion { detail }
}

fn partitions_of(degree: usize) -> Result<Vec<Partition>, CombinError> {
    require_degree(degree)?;
    let mut output = Vec::new();
    fn visit(
        remaining: usize,
        maximum: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Partition>,
    ) -> Result<(), CombinError> {
        if remaining == 0 {
            output.push(Partition::new(current.clone())?);
            return Ok(());
        }
        for part in (1..=remaining.min(maximum)).rev() {
            current.push(part);
            visit(remaining - part, part, current, output)?;
            current.pop();
        }
        Ok(())
    }
    if degree == 0 {
        output.push(Partition::new(Vec::new())?);
    } else {
        visit(degree, degree, &mut Vec::new(), &mut output)?;
    }
    Ok(output)
}

fn basis_to_power_matrix(
    basis: SymmetricBasis,
    degree: usize,
    partitions: &[Partition],
) -> Result<Matrix<crate::RationalField>, CombinError> {
    match basis {
        SymmetricBasis::PowerSum => Ok(Matrix::identity(Q, partitions.len())),
        SymmetricBasis::CompleteHomogeneous | SymmetricBasis::Elementary => {
            multiplicative_to_power_matrix(basis, partitions)
        }
        SymmetricBasis::Monomial => monomial_to_power_matrix(degree, partitions),
        SymmetricBasis::Schur => {
            let monomial = monomial_to_power_matrix(degree, partitions)?;
            let kostka = Matrix::from_rows(
                Q,
                partitions.iter().map(|shape| {
                    partitions
                        .iter()
                        .map(|content| Rational::from(kostka_number(shape, content)))
                        .collect::<Vec<_>>()
                }),
            )
            .map_err(|error| conversion_error(error.to_string()))?;
            kostka
                .matmul(&monomial)
                .map_err(|error| conversion_error(error.to_string()))
        }
    }
}

fn multiplicative_to_power_matrix(
    basis: SymmetricBasis,
    partitions: &[Partition],
) -> Result<Matrix<crate::RationalField>, CombinError> {
    let index = partitions
        .iter()
        .enumerate()
        .map(|(position, partition)| (partition.clone(), position))
        .collect::<BTreeMap<_, _>>();
    let rows = partitions
        .iter()
        .map(|lambda| {
            let mut expansion = BTreeMap::from([(Partition::new(Vec::new())?, Rational::one())]);
            for &part in lambda.parts() {
                expansion = multiply_expansions(&expansion, &generator_to_power(basis, part)?)?;
            }
            let mut row = vec![Rational::zero(); partitions.len()];
            for (partition, coefficient) in expansion {
                let position = index.get(&partition).copied().ok_or_else(|| {
                    conversion_error("basis expansion left its homogeneous component".to_owned())
                })?;
                row[position] = coefficient;
            }
            Ok(row)
        })
        .collect::<Result<Vec<_>, CombinError>>()?;
    Matrix::from_rows(Q, rows).map_err(|error| conversion_error(error.to_string()))
}

fn generator_to_power(
    basis: SymmetricBasis,
    degree: usize,
) -> Result<BTreeMap<Partition, Rational>, CombinError> {
    let mut output = BTreeMap::new();
    for partition in partitions_of(degree)? {
        let mut coefficient = Rational::new(Integer::one(), z_lambda(&partition))
            .map_err(|error| conversion_error(error.to_string()))?;
        if basis == SymmetricBasis::Elementary
            && (degree.saturating_sub(partition.parts().len())) % 2 == 1
        {
            coefficient = -coefficient;
        }
        output.insert(partition, coefficient);
    }
    Ok(output)
}

fn multiply_expansions(
    left: &BTreeMap<Partition, Rational>,
    right: &BTreeMap<Partition, Rational>,
) -> Result<BTreeMap<Partition, Rational>, CombinError> {
    let mut output = BTreeMap::<Partition, Rational>::new();
    for (lambda, left_coefficient) in left {
        for (mu, right_coefficient) in right {
            let partition = partition_union(lambda, mu)?;
            let slot = output.entry(partition).or_insert_with(Rational::zero);
            *slot = &*slot + &(left_coefficient * right_coefficient);
        }
    }
    output.retain(|_, coefficient| !coefficient.is_zero());
    Ok(output)
}

fn partition_union(left: &Partition, right: &Partition) -> Result<Partition, CombinError> {
    let mut parts = left
        .parts()
        .iter()
        .chain(right.parts())
        .copied()
        .collect::<Vec<_>>();
    parts.sort_unstable_by(|left, right| right.cmp(left));
    Partition::new(parts)
}

fn monomial_to_power_matrix(
    _degree: usize,
    partitions: &[Partition],
) -> Result<Matrix<crate::RationalField>, CombinError> {
    let homogeneous =
        multiplicative_to_power_matrix(SymmetricBasis::CompleteHomogeneous, partitions)?;
    let hall_dual_system = Matrix::from_rows(
        Q,
        homogeneous.rows().map(|row| {
            row.into_iter()
                .zip(partitions)
                .map(|(coefficient, partition)| &coefficient * &Rational::from(z_lambda(partition)))
                .collect::<Vec<_>>()
        }),
    )
    .map_err(|error| conversion_error(error.to_string()))?;

    let mut inverse_columns = Vec::with_capacity(partitions.len());
    for column in 0..partitions.len() {
        let rhs = Vector::from_values(
            Q,
            (0..partitions.len()).map(|row| Rational::from(u8::from(row == column))),
        )
        .map_err(|error| conversion_error(error.to_string()))?;
        inverse_columns.push(
            hall_dual_system
                .solve(&rhs)
                .map_err(|error| conversion_error(error.to_string()))?
                .particular()
                .values()
                .collect::<Vec<_>>(),
        );
    }
    Matrix::from_rows(
        Q,
        (0..partitions.len()).map(|row| {
            (0..partitions.len())
                .map(|column| inverse_columns[row][column].clone())
                .collect::<Vec<_>>()
        }),
    )
    .map_err(|error| conversion_error(error.to_string()))
}

fn z_lambda(partition: &Partition) -> Integer {
    let mut multiplicities = BTreeMap::<usize, usize>::new();
    for &part in partition.parts() {
        *multiplicities.entry(part).or_default() += 1;
    }
    let mut value = Integer::one();
    for (part, multiplicity) in multiplicities {
        for _ in 0..multiplicity {
            value = &value * &Integer::from(part);
        }
        for factor in 2..=multiplicity {
            value = &value * &Integer::from(factor);
        }
    }
    value
}

fn kostka_number(shape: &Partition, content: &Partition) -> u64 {
    let cells = shape
        .parts()
        .iter()
        .enumerate()
        .flat_map(|(row, &width)| (0..width).map(move |column| (row, column)))
        .collect::<Vec<_>>();
    let mut tableau = shape
        .parts()
        .iter()
        .map(|&width| vec![0_usize; width])
        .collect::<Vec<_>>();
    let mut remaining = content.parts().to_vec();

    fn count(
        position: usize,
        cells: &[(usize, usize)],
        tableau: &mut [Vec<usize>],
        remaining: &mut [usize],
    ) -> u64 {
        if position == cells.len() {
            return 1;
        }
        let (row, column) = cells[position];
        let row_minimum = if column == 0 {
            1
        } else {
            tableau[row][column - 1]
        };
        let column_minimum = if row == 0 {
            1
        } else {
            tableau[row - 1][column] + 1
        };
        let mut total = 0;
        for value in row_minimum.max(column_minimum)..=remaining.len() {
            if remaining[value - 1] == 0 {
                continue;
            }
            tableau[row][column] = value;
            remaining[value - 1] -= 1;
            total += count(position + 1, cells, tableau, remaining);
            remaining[value - 1] += 1;
        }
        tableau[row][column] = 0;
        total
    }

    count(0, &cells, &mut tableau, &mut remaining)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partition(parts: &[usize]) -> Partition {
        Partition::new(parts.to_vec()).unwrap()
    }

    fn function(basis: SymmetricBasis, terms: &[(&[usize], i64)]) -> SymmetricFunction {
        SymmetricFunction::new(
            basis,
            terms
                .iter()
                .map(|(parts, coefficient)| (partition(parts), Rational::from(*coefficient))),
        )
        .unwrap()
    }

    fn coefficient(function: &SymmetricFunction, parts: &[usize]) -> Rational {
        function
            .terms
            .get(&partition(parts))
            .cloned()
            .unwrap_or_else(Rational::zero)
    }

    #[test]
    fn partition_order_and_hall_duality_base_case_are_exact() {
        assert_eq!(
            partitions_of(4)
                .unwrap()
                .iter()
                .map(|partition| partition.parts().to_vec())
                .collect::<Vec<_>>(),
            vec![
                vec![4],
                vec![3, 1],
                vec![2, 2],
                vec![2, 1, 1],
                vec![1, 1, 1, 1]
            ]
        );
        assert_eq!(z_lambda(&partition(&[2, 1, 1])), Integer::from(4));

        let monomial_one = function(SymmetricBasis::Monomial, &[(&[1], 1)]);
        let power_one = monomial_one
            .convert_basis(SymmetricBasis::PowerSum)
            .unwrap();
        assert_eq!(power_one, function(SymmetricBasis::PowerSum, &[(&[1], 1)]));
    }

    #[test]
    fn schur_conversion_matches_hand_checked_degree_three_identity() {
        // s_(2,1) = (p_(1,1,1) - p_(3))/3.
        let schur = function(SymmetricBasis::Schur, &[(&[2, 1], 1)]);
        let power = schur.convert_basis(SymmetricBasis::PowerSum).unwrap();
        let third = Rational::new(Integer::one(), Integer::from(3)).unwrap();
        assert_eq!(coefficient(&power, &[1, 1, 1]), third);
        assert_eq!(
            coefficient(&power, &[3]),
            -Rational::new(Integer::one(), Integer::from(3)).unwrap()
        );
        assert_eq!(power.convert_basis(SymmetricBasis::Schur).unwrap(), schur);
    }

    #[test]
    fn schur_product_has_littlewood_richardson_coefficients() {
        let s_one = function(SymmetricBasis::Schur, &[(&[1], 1)]);
        let square = s_one.mul(&s_one).unwrap();
        assert_eq!(
            square,
            function(SymmetricBasis::Schur, &[(&[2], 1), (&[1, 1], 1)])
        );

        let s_twenty_one = function(SymmetricBasis::Schur, &[(&[2, 1], 1)]);
        let square = s_twenty_one.mul(&s_twenty_one).unwrap();
        assert_eq!(coefficient(&square, &[3, 2, 1]), Rational::from(2));
        assert_eq!(coefficient(&square, &[4, 2]), Rational::one());
        assert_eq!(coefficient(&square, &[3, 3]), Rational::one());
    }

    #[test]
    fn constructor_and_arithmetic_fail_closed() {
        assert_eq!(
            SymmetricFunction::new(
                SymmetricBasis::Schur,
                [
                    (partition(&[2]), Rational::one()),
                    (partition(&[1]), Rational::one()),
                ],
            ),
            Err(CombinError::MixedSymmetricDegrees {
                expected: 1,
                actual: 2,
            })
        );
        let too_large = partition(&[MAX_SYMMETRIC_DEGREE + 1]);
        assert_eq!(
            SymmetricFunction::new(SymmetricBasis::PowerSum, [(too_large, Rational::one())]),
            Err(CombinError::SymmetricDegreeTooLarge {
                degree: MAX_SYMMETRIC_DEGREE + 1,
                max: MAX_SYMMETRIC_DEGREE,
            })
        );
    }

    #[test]
    fn serialization_revalidates_and_canonical_identity_ignores_basis() {
        let schur = function(SymmetricBasis::Schur, &[(&[2, 1], 2)]);
        let json = serde_json::to_string(&schur).unwrap();
        let rebuilt: SymmetricFunction = serde_json::from_str(&json).unwrap();
        assert_eq!(rebuilt, schur);
        assert_eq!(
            schur.canonical_bytes().unwrap(),
            schur
                .convert_basis(SymmetricBasis::Monomial)
                .unwrap()
                .canonical_bytes()
                .unwrap()
        );
    }
}
