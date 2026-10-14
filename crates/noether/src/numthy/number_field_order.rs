// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact orders in absolute number fields.

#![allow(unsafe_code)]

use core::fmt;
use std::collections::BTreeMap;
use std::sync::Mutex;

use flint_sys::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{factor, NumberField, NumberFieldElement};
use crate::commalg::PrimeIdeal;
use crate::{
    CommutativeRing, ContextMismatchError, Field, Integer, IntegralDomain, NoetherError, Rational,
    Ring, SemanticType, TypeDescription, UnitError, UnitInversion,
};

static NUMBER_FIELD_ORDER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/NumberFieldOrder@1",
    rust_path: "noether::numthy::NumberFieldOrder",
    python_path: Some("noether.numthy.NumberFieldOrder"),
};

/// Failure to construct or compare an exact number-field order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumberFieldOrderError {
    /// The basis did not have the field degree as its length.
    WrongBasisLength {
        /// Required rank.
        degree: usize,
        /// Supplied basis length.
        actual: usize,
    },
    /// A basis element belonged to a different number field.
    BasisContextMismatch,
    /// The basis was linearly dependent over `Q`.
    NotFullRank,
    /// The lattice did not contain the multiplicative identity.
    MissingOne,
    /// The lattice was not closed under multiplication.
    NotClosed,
    /// The distinguished generator was not integral.
    GeneratorNotIntegral,
    /// Two orders were not in the required containment relation.
    NotContained,
    /// An order discriminant unexpectedly failed to be integral.
    NonIntegralDiscriminant,
    /// Round 2 currently needs a prime accepted by FLINT's word-modulus matrices.
    Round2PrimeTooLarge(Integer),
    /// A proved invariant of the Round-2 construction failed internally.
    Round2Invariant,
    /// Exact number-field arithmetic rejected an internal operation.
    ContextMismatch(ContextMismatchError),
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for NumberFieldOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongBasisLength { degree, actual } => {
                write!(f, "number-field order needs {degree} basis elements; received {actual}")
            }
            Self::BasisContextMismatch => {
                f.write_str("number-field order basis elements must belong to its field")
            }
            Self::NotFullRank => f.write_str("number-field order basis is not full rank"),
            Self::MissingOne => f.write_str("number-field order basis does not span one"),
            Self::NotClosed => {
                f.write_str("number-field order basis is not closed under multiplication")
            }
            Self::GeneratorNotIntegral => f.write_str(
                "the distinguished generator is not integral; construct an explicit order or ask for the maximal order",
            ),
            Self::NotContained => {
                f.write_str("the first number-field order is not contained in the second")
            }
            Self::NonIntegralDiscriminant => {
                f.write_str("validated number-field order had a nonintegral discriminant")
            }
            Self::Round2PrimeTooLarge(prime) => write!(
                f,
                "maximal-order Round 2 cannot represent the squarefull prime {prime} as a word-sized modulus"
            ),
            Self::Round2Invariant => {
                f.write_str("maximal-order Round 2 encountered an inconsistent exact lattice")
            }
            Self::ContextMismatch(error) => error.fmt(f),
            Self::InvalidEncoding(message) => {
                write!(f, "invalid number-field order encoding: {message}")
            }
        }
    }
}

impl std::error::Error for NumberFieldOrderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for NumberFieldOrderError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

struct IntegerMatrix {
    raw: fmpz_mat_t,
}

pub(super) struct ExactLattice {
    basis: IntegerMatrix,
    denominator: Integer,
}

impl ExactLattice {
    pub(super) fn from_spanning_elements(
        field: &NumberField,
        elements: &[NumberFieldElement],
    ) -> Option<Self> {
        let degree = field.degree();
        if elements.is_empty() || elements.iter().any(|element| !field.contains(element)) {
            return None;
        }
        let flat = elements
            .iter()
            .flat_map(NumberFieldElement::coefficients)
            .collect::<Vec<_>>();
        let (numerators, mut denominator) = to_common_denominator(&flat);
        let mut generators = IntegerMatrix::zero(elements.len(), degree);
        for row in 0..elements.len() {
            for column in 0..degree {
                generators.set_entry(row, column, &numerators[row * degree + column]);
            }
        }
        let hnf = generators.hnf();
        let nonzero_rows = (0..hnf.rows())
            .filter(|row| (0..degree).any(|column| !hnf.entry(*row, column).is_zero()))
            .collect::<Vec<_>>();
        if nonzero_rows.len() != degree {
            return None;
        }
        let mut basis = IntegerMatrix::zero(degree, degree);
        for (output_row, input_row) in nonzero_rows.into_iter().enumerate() {
            for column in 0..degree {
                basis.set_entry(output_row, column, &hnf.entry(input_row, column));
            }
        }

        let mut common = denominator.clone();
        for row in 0..degree {
            for column in 0..degree {
                common = integer_gcd(&common, &basis.entry(row, column).abs());
            }
        }
        if !common.is_one() {
            denominator = integer_exact_div(&denominator, &common);
            for row in 0..degree {
                for column in 0..degree {
                    let reduced = integer_exact_div(&basis.entry(row, column), &common);
                    basis.set_entry(row, column, &reduced);
                }
            }
        }
        Some(Self { basis, denominator })
    }

    pub(super) fn basis(&self, field: &NumberField) -> Vec<NumberFieldElement> {
        (0..field.degree())
            .map(|row| {
                let coefficients = (0..field.degree())
                    .map(|column| {
                        Rational::new(self.basis.entry(row, column), self.denominator.clone())
                            .expect("an exact lattice has positive denominator")
                    })
                    .collect::<Vec<_>>();
                field
                    .element(&coefficients)
                    .expect("an exact lattice row has field-degree-many coefficients")
            })
            .collect()
    }

    pub(super) fn contains(&self, element: &NumberFieldElement) -> bool {
        solve_in_basis(&self.basis, &self.denominator, &element.coefficients()).is_some()
    }

    pub(super) fn covolume_parts(&self, degree: usize) -> (Integer, Integer) {
        (
            self.basis.determinant().abs(),
            self.denominator.pow(degree as u64),
        )
    }
}

impl Clone for ExactLattice {
    fn clone(&self) -> Self {
        Self {
            basis: self.basis.clone(),
            denominator: self.denominator.clone(),
        }
    }
}

impl PartialEq for ExactLattice {
    fn eq(&self, other: &Self) -> bool {
        self.denominator == other.denominator
            && self.basis.rows() == other.basis.rows()
            && self.basis.columns() == other.basis.columns()
            && (0..self.basis.rows()).all(|row| {
                (0..self.basis.columns())
                    .all(|column| self.basis.entry(row, column) == other.basis.entry(row, column))
            })
    }
}

impl Eq for ExactLattice {}

impl IntegerMatrix {
    fn zero(rows: usize, columns: usize) -> Self {
        let mut raw: fmpz_mat_t = unsafe { std::mem::zeroed() };
        unsafe { fmpz_mat_init(raw.as_mut_ptr(), rows as slong, columns as slong) };
        Self { raw }
    }

    fn as_ptr(&self) -> *const fmpz_mat_struct {
        self.raw.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut fmpz_mat_struct {
        self.raw.as_mut_ptr()
    }

    fn entry(&self, row: usize, column: usize) -> Integer {
        unsafe {
            Integer::clone_from_ptr(fmpz_mat_entry(self.as_ptr(), row as slong, column as slong))
        }
    }

    fn set_entry(&mut self, row: usize, column: usize, value: &Integer) {
        unsafe {
            fmpz_set(
                fmpz_mat_entry(self.as_ptr(), row as slong, column as slong),
                value.as_ptr(),
            )
        };
    }

    fn determinant(&self) -> Integer {
        let mut output = Integer::zero();
        unsafe { fmpz_mat_det(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    fn transpose(&self) -> Self {
        let mut output = Self::zero(self.columns(), self.rows());
        unsafe { fmpz_mat_transpose(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    fn hnf(&self) -> Self {
        let mut output = Self::zero(self.rows(), self.columns());
        unsafe { fmpz_mat_hnf(output.as_mut_ptr(), self.as_ptr()) };
        output
    }

    fn rows(&self) -> usize {
        unsafe { fmpz_mat_nrows(self.as_ptr()) as usize }
    }

    fn columns(&self) -> usize {
        unsafe { fmpz_mat_ncols(self.as_ptr()) as usize }
    }
}

struct ModularMatrix {
    raw: nmod_mat_t,
}

impl ModularMatrix {
    fn zero(rows: usize, columns: usize, modulus: u64) -> Self {
        let mut raw: nmod_mat_t = unsafe { std::mem::zeroed() };
        unsafe { nmod_mat_init(raw.as_mut_ptr(), rows as slong, columns as slong, modulus) };
        Self { raw }
    }

    fn as_ptr(&self) -> *const nmod_mat_struct {
        self.raw.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut nmod_mat_struct {
        self.raw.as_mut_ptr()
    }

    fn set_entry(&mut self, row: usize, column: usize, value: u64) {
        unsafe { nmod_mat_set_entry(self.as_mut_ptr(), row as slong, column as slong, value) };
    }

    fn entry(&self, row: usize, column: usize) -> u64 {
        unsafe { nmod_mat_get_entry(self.as_ptr(), row as slong, column as slong) }
    }

    fn nullspace(&self) -> (Self, usize) {
        let columns = unsafe { (*self.as_ptr()).c as usize };
        let modulus = unsafe { (*self.as_ptr()).mod_.n };
        let mut output = Self::zero(columns, columns, modulus);
        let nullity = unsafe { nmod_mat_nullspace(output.as_mut_ptr(), self.as_ptr()) as usize };
        (output, nullity)
    }
}

impl Drop for ModularMatrix {
    fn drop(&mut self) {
        unsafe { nmod_mat_clear(self.as_mut_ptr()) };
    }
}

impl Clone for IntegerMatrix {
    fn clone(&self) -> Self {
        let mut output = Self::zero(self.rows(), self.columns());
        unsafe { fmpz_mat_set(output.as_mut_ptr(), self.as_ptr()) };
        output
    }
}

impl Drop for IntegerMatrix {
    fn drop(&mut self) {
        unsafe { fmpz_mat_clear(self.as_mut_ptr()) };
    }
}

// SAFETY: the owned FLINT matrix is immutable once stored on an order.
unsafe impl Send for IntegerMatrix {}
unsafe impl Sync for IntegerMatrix {}

pub(super) fn integer_lcm(left: &Integer, right: &Integer) -> Integer {
    let mut output = Integer::zero();
    unsafe { fmpz_lcm(output.as_mut_ptr(), left.as_ptr(), right.as_ptr()) };
    output
}

fn integer_gcd(left: &Integer, right: &Integer) -> Integer {
    let mut output = Integer::zero();
    unsafe { fmpz_gcd(output.as_mut_ptr(), left.as_ptr(), right.as_ptr()) };
    output
}

fn integer_exact_div(numerator: &Integer, denominator: &Integer) -> Integer {
    let mut output = Integer::zero();
    unsafe {
        fmpz_divexact(
            output.as_mut_ptr(),
            numerator.as_ptr(),
            denominator.as_ptr(),
        )
    };
    output
}

pub(super) fn integer_mod_u64(value: &Integer, modulus: u64) -> u64 {
    unsafe { fmpz_fdiv_ui(value.as_ptr(), modulus) }
}

fn to_common_denominator(coefficients: &[Rational]) -> (Vec<Integer>, Integer) {
    let denominator = coefficients.iter().fold(Integer::one(), |current, value| {
        integer_lcm(&current, &value.denominator())
    });
    let numerators = coefficients
        .iter()
        .map(|coefficient| {
            let scale = integer_exact_div(&denominator, &coefficient.denominator());
            &coefficient.numerator() * &scale
        })
        .collect();
    (numerators, denominator)
}

fn normalized_basis(elements: &[NumberFieldElement]) -> (IntegerMatrix, Integer) {
    let degree = elements.len();
    let flat = elements
        .iter()
        .flat_map(NumberFieldElement::coefficients)
        .collect::<Vec<_>>();
    let (numerators, mut denominator) = to_common_denominator(&flat);
    let mut matrix = IntegerMatrix::zero(degree, degree);
    for row in 0..degree {
        for column in 0..degree {
            matrix.set_entry(row, column, &numerators[row * degree + column]);
        }
    }
    matrix = matrix.hnf();

    let mut common = denominator.clone();
    for row in 0..degree {
        for column in 0..degree {
            common = integer_gcd(&common, &matrix.entry(row, column).abs());
        }
    }
    if !common.is_one() {
        denominator = integer_exact_div(&denominator, &common);
        for row in 0..degree {
            for column in 0..degree {
                let reduced = integer_exact_div(&matrix.entry(row, column), &common);
                matrix.set_entry(row, column, &reduced);
            }
        }
    }
    (matrix, denominator)
}

fn lattice_basis_from_generators(
    generators: &[Vec<Integer>],
    rank: usize,
) -> Result<Vec<Vec<Integer>>, NumberFieldOrderError> {
    if generators.iter().any(|row| row.len() != rank) {
        return Err(NumberFieldOrderError::Round2Invariant);
    }
    let mut matrix = IntegerMatrix::zero(generators.len(), rank);
    for (row, entries) in generators.iter().enumerate() {
        for (column, entry) in entries.iter().enumerate() {
            matrix.set_entry(row, column, entry);
        }
    }
    let hnf = matrix.hnf();
    let basis = (0..hnf.rows())
        .filter(|row| (0..rank).any(|column| !hnf.entry(*row, column).is_zero()))
        .map(|row| (0..rank).map(|column| hnf.entry(row, column)).collect())
        .collect::<Vec<Vec<Integer>>>();
    if basis.len() == rank {
        Ok(basis)
    } else {
        Err(NumberFieldOrderError::Round2Invariant)
    }
}

fn solve_in_basis(
    basis: &IntegerMatrix,
    denominator: &Integer,
    target: &[Rational],
) -> Option<Vec<Integer>> {
    solve_rational_in_basis(basis, denominator, target)?
        .into_iter()
        .map(|coordinate| {
            coordinate
                .denominator()
                .is_one()
                .then(|| coordinate.numerator())
        })
        .collect()
}

fn solve_rational_in_basis(
    basis: &IntegerMatrix,
    denominator: &Integer,
    target: &[Rational],
) -> Option<Vec<Rational>> {
    let degree = target.len();
    let (target_numerators, target_denominator) = to_common_denominator(target);
    let transpose = basis.transpose();
    let mut right_hand_side = IntegerMatrix::zero(degree, 1);
    for (row, numerator) in target_numerators.iter().enumerate() {
        right_hand_side.set_entry(row, 0, &(denominator * numerator));
    }
    let mut solution = IntegerMatrix::zero(degree, 1);
    let mut solution_denominator = Integer::zero();
    let solved = unsafe {
        fmpz_mat_solve(
            solution.as_mut_ptr(),
            solution_denominator.as_mut_ptr(),
            transpose.as_ptr(),
            right_hand_side.as_ptr(),
        )
    };
    if solved == 0 {
        return None;
    }

    let full_denominator = &solution_denominator * &target_denominator;
    let mut coordinates = Vec::with_capacity(degree);
    for row in 0..degree {
        let entry = solution.entry(row, 0);
        coordinates.push(Rational::new(entry, full_denominator.clone()).ok()?);
    }
    Some(coordinates)
}

fn rational_determinant(mut matrix: Vec<Vec<Rational>>) -> Rational {
    let dimension = matrix.len();
    let mut sign = 1_i64;
    for column in 0..dimension {
        let Some(pivot_row) = (column..dimension).find(|row| !matrix[*row][column].is_zero())
        else {
            return Rational::zero();
        };
        if pivot_row != column {
            matrix.swap(pivot_row, column);
            sign = -sign;
        }
        let pivot = matrix[column][column].clone();
        let pivot_row_values = matrix[column].clone();
        for row_values in matrix.iter_mut().take(dimension).skip(column + 1) {
            let factor = row_values[column]
                .checked_div(&pivot)
                .expect("a selected pivot is nonzero");
            for entry in column..dimension {
                let product = &factor * &pivot_row_values[entry];
                row_values[entry] = &row_values[entry] - &product;
            }
        }
    }
    matrix
        .iter()
        .enumerate()
        .fold(Rational::from(sign), |determinant, (index, row)| {
            &determinant * &row[index]
        })
}

fn trace_discriminant(basis: &[NumberFieldElement]) -> Result<Integer, NumberFieldOrderError> {
    let dimension = basis.len();
    let mut trace_matrix = vec![vec![Rational::zero(); dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let trace = basis[row].mul(&basis[column])?.trace();
            trace_matrix[row][column] = trace.clone();
            trace_matrix[column][row] = trace;
        }
    }
    let discriminant = rational_determinant(trace_matrix);
    if !discriminant.denominator().is_one() {
        return Err(NumberFieldOrderError::NonIntegralDiscriminant);
    }
    Ok(discriminant.numerator())
}

/// A full-rank unital `Z`-order in an exact number field.
///
/// The order stores a canonical row-Hermite basis in the distinguished power
/// basis of its field. Equality therefore compares lattices, not the supplied
/// ordering of basis vectors.
pub struct NumberFieldOrder {
    field: NumberField,
    basis: IntegerMatrix,
    denominator: Integer,
    discriminant: Integer,
    prime_decompositions: Mutex<BTreeMap<Integer, Vec<PrimeIdeal<NumberFieldOrder>>>>,
}

#[derive(Clone)]
pub(super) struct NumberFieldOrderCache {
    basis: IntegerMatrix,
    denominator: Integer,
    discriminant: Integer,
}

impl NumberFieldOrder {
    fn from_cache(field: NumberField, cache: NumberFieldOrderCache) -> Self {
        Self {
            field,
            basis: cache.basis,
            denominator: cache.denominator,
            discriminant: cache.discriminant,
            prime_decompositions: Mutex::new(BTreeMap::new()),
        }
    }

    fn cache(&self) -> NumberFieldOrderCache {
        NumberFieldOrderCache {
            basis: self.basis.clone(),
            denominator: self.denominator.clone(),
            discriminant: self.discriminant.clone(),
        }
    }

    /// Validate and construct the order spanned by an explicit `Z`-basis.
    pub fn new(
        field: NumberField,
        basis: &[NumberFieldElement],
    ) -> Result<Self, NumberFieldOrderError> {
        let degree = field.degree();
        if basis.len() != degree {
            return Err(NumberFieldOrderError::WrongBasisLength {
                degree,
                actual: basis.len(),
            });
        }
        if basis.iter().any(|element| !field.contains(element)) {
            return Err(NumberFieldOrderError::BasisContextMismatch);
        }
        let (basis_matrix, denominator) = normalized_basis(basis);
        if basis_matrix.determinant().is_zero() {
            return Err(NumberFieldOrderError::NotFullRank);
        }
        let mut order = Self {
            field,
            basis: basis_matrix,
            denominator,
            discriminant: Integer::zero(),
            prime_decompositions: Mutex::new(BTreeMap::new()),
        };
        if !order.contains(&order.field.one()) {
            return Err(NumberFieldOrderError::MissingOne);
        }
        let canonical_basis = order.basis();
        for left in &canonical_basis {
            for right in &canonical_basis {
                if !order.contains(&left.mul(right)?) {
                    return Err(NumberFieldOrderError::NotClosed);
                }
            }
        }
        order.discriminant = trace_discriminant(&canonical_basis)?;
        Ok(order)
    }

    /// The ambient number field.
    pub fn field(&self) -> &NumberField {
        &self.field
    }

    /// Canonical integral basis as exact field elements.
    pub fn basis(&self) -> Vec<NumberFieldElement> {
        (0..self.field.degree())
            .map(|row| {
                let coefficients = (0..self.field.degree())
                    .map(|column| {
                        Rational::new(self.basis.entry(row, column), self.denominator.clone())
                            .expect("an order basis has positive denominator")
                    })
                    .collect::<Vec<_>>();
                self.field
                    .element(&coefficients)
                    .expect("a canonical order row has field-degree-many coefficients")
            })
            .collect()
    }

    /// Exact discriminant of this order.
    pub fn discriminant(&self) -> Integer {
        self.discriminant.clone()
    }

    /// Whether an exact field element lies in this integral lattice.
    pub fn contains(&self, element: &NumberFieldElement) -> bool {
        self.field.contains(element)
            && solve_in_basis(&self.basis, &self.denominator, &element.coefficients()).is_some()
    }

    pub(super) fn integral_coordinates(
        &self,
        element: &NumberFieldElement,
    ) -> Option<Vec<Integer>> {
        if !self.field.contains(element) {
            return None;
        }
        solve_in_basis(&self.basis, &self.denominator, &element.coefficients())
    }

    pub(super) fn rational_coordinates(
        &self,
        element: &NumberFieldElement,
    ) -> Option<Vec<Rational>> {
        if !self.field.contains(element) {
            return None;
        }
        solve_rational_in_basis(&self.basis, &self.denominator, &element.coefficients())
    }

    fn elements_from_coordinates(
        &self,
        coordinates: &[Vec<Integer>],
    ) -> Result<Vec<NumberFieldElement>, NumberFieldOrderError> {
        let basis = self.basis();
        coordinates
            .iter()
            .map(|row| {
                if row.len() != basis.len() {
                    return Err(NumberFieldOrderError::Round2Invariant);
                }
                let mut coefficients = vec![Rational::zero(); self.field.degree()];
                for (coordinate, element) in row.iter().zip(&basis) {
                    let scalar = Rational::from(coordinate.clone());
                    for (output, coefficient) in coefficients.iter_mut().zip(element.coefficients())
                    {
                        *output = &*output + &(&scalar * &coefficient);
                    }
                }
                self.field
                    .element(&coefficients)
                    .map_err(|_| NumberFieldOrderError::Round2Invariant)
            })
            .collect()
    }

    pub(super) fn p_radical(
        &self,
        prime: u64,
    ) -> Result<Vec<NumberFieldElement>, NumberFieldOrderError> {
        let degree = self.field.degree();
        let degree_u64 =
            u64::try_from(degree).map_err(|_| NumberFieldOrderError::Round2Invariant)?;
        let mut frobenius_power = prime;
        while frobenius_power < degree_u64 {
            frobenius_power = frobenius_power
                .checked_mul(prime)
                .ok_or(NumberFieldOrderError::Round2Invariant)?;
        }

        let basis = self.basis();
        let mut frobenius = ModularMatrix::zero(degree, degree, prime);
        for (column, element) in basis.iter().enumerate() {
            let power = element.pow(frobenius_power)?;
            let coordinates = self
                .integral_coordinates(&power)
                .ok_or(NumberFieldOrderError::Round2Invariant)?;
            for (row, coordinate) in coordinates.iter().enumerate() {
                frobenius.set_entry(row, column, integer_mod_u64(coordinate, prime));
            }
        }

        let (kernel, nullity) = frobenius.nullspace();
        let mut generators = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|column| {
                        if row == column {
                            Integer::from(prime)
                        } else {
                            Integer::zero()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for column in 0..nullity {
            generators.push(
                (0..degree)
                    .map(|row| Integer::from(kernel.entry(row, column)))
                    .collect(),
            );
        }
        let coordinates = lattice_basis_from_generators(&generators, degree)?;
        self.elements_from_coordinates(&coordinates)
    }

    fn ring_of_multipliers(
        &self,
        prime: u64,
        radical: &[NumberFieldElement],
    ) -> Result<Self, NumberFieldOrderError> {
        let degree = self.field.degree();
        if radical.len() != degree {
            return Err(NumberFieldOrderError::Round2Invariant);
        }
        let (radical_basis, radical_denominator) = normalized_basis(radical);
        let order_basis = self.basis();
        let mut conditions = ModularMatrix::zero(degree * degree, degree, prime);
        for (column, basis_element) in order_basis.iter().enumerate() {
            for (radical_index, radical_element) in radical.iter().enumerate() {
                let product = basis_element.mul(radical_element)?;
                let coordinates = solve_in_basis(
                    &radical_basis,
                    &radical_denominator,
                    &product.coefficients(),
                )
                .ok_or(NumberFieldOrderError::Round2Invariant)?;
                for (coordinate_index, coordinate) in coordinates.iter().enumerate() {
                    conditions.set_entry(
                        radical_index * degree + coordinate_index,
                        column,
                        integer_mod_u64(coordinate, prime),
                    );
                }
            }
        }

        let (kernel, nullity) = conditions.nullspace();
        let mut generators = (0..degree)
            .map(|row| {
                (0..degree)
                    .map(|column| {
                        if row == column {
                            Integer::from(prime)
                        } else {
                            Integer::zero()
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for column in 0..nullity {
            generators.push(
                (0..degree)
                    .map(|row| Integer::from(kernel.entry(row, column)))
                    .collect(),
            );
        }
        let coordinates = lattice_basis_from_generators(&generators, degree)?;
        let numerators = self.elements_from_coordinates(&coordinates)?;
        let inverse_prime = Rational::new(Integer::one(), Integer::from(prime))
            .map_err(|_| NumberFieldOrderError::Round2Invariant)?;
        let basis = numerators
            .iter()
            .map(|element| element.mul(&self.field.scalar(inverse_prime.clone())))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(self.field.clone(), &basis)
    }

    fn maximal_at_prime(mut self, prime: u64) -> Result<Self, NumberFieldOrderError> {
        loop {
            let radical = self.p_radical(prime)?;
            let next = self.ring_of_multipliers(prime, &radical)?;
            if next == self {
                return Ok(next);
            }
            self = next;
        }
    }

    /// Whether this order is the maximal order of its ambient number field.
    pub fn is_maximal(&self) -> Result<bool, NumberFieldOrderError> {
        Ok(self.discriminant == self.field.maximal_order()?.discriminant)
    }

    /// Index `[super_order : self]` when `self` is contained in `super_order`.
    pub fn index(&self, super_order: &NumberFieldOrder) -> Result<Integer, NumberFieldOrderError> {
        if self.field != super_order.field
            || self
                .basis()
                .iter()
                .any(|element| !super_order.contains(element))
        {
            return Err(NumberFieldOrderError::NotContained);
        }
        let degree = self.field.degree() as u64;
        let numerator = &self.basis.determinant().abs() * &super_order.denominator.pow(degree);
        let denominator = &super_order.basis.determinant().abs() * &self.denominator.pow(degree);
        Ok(integer_exact_div(&numerator, &denominator))
    }

    pub(super) fn covolume_parts(&self) -> (Integer, Integer) {
        (
            self.basis.determinant().abs(),
            self.denominator.pow(self.field.degree() as u64),
        )
    }

    fn mismatch(&self, actual: &NumberFieldElement) -> ContextMismatchError {
        ContextMismatchError {
            expected: format!("NumberFieldOrder(discriminant={})", self.discriminant),
            actual: if self.field.contains(actual) {
                "number-field element outside order".to_owned()
            } else {
                "number-field element from another field".to_owned()
            },
        }
    }

    /// Serialize this exact order to canonical JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and validate an exact order from canonical JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    pub(super) fn cached_prime_decomposition(
        &self,
        prime: &Integer,
    ) -> Option<Vec<PrimeIdeal<NumberFieldOrder>>> {
        self.prime_decompositions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(prime)
            .cloned()
    }

    pub(super) fn cache_prime_decomposition(
        &self,
        prime: Integer,
        decomposition: &[PrimeIdeal<NumberFieldOrder>],
    ) {
        self.prime_decompositions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(prime, decomposition.to_vec());
    }
}

impl Clone for NumberFieldOrder {
    fn clone(&self) -> Self {
        Self {
            field: self.field.clone(),
            basis: self.basis.clone(),
            denominator: self.denominator.clone(),
            discriminant: self.discriminant.clone(),
            prime_decompositions: Mutex::new(BTreeMap::new()),
        }
    }
}

impl PartialEq for NumberFieldOrder {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.denominator == other.denominator
            && self.basis.rows() == other.basis.rows()
            && self.basis.columns() == other.basis.columns()
            && (0..self.basis.rows()).all(|row| {
                (0..self.basis.columns())
                    .all(|column| self.basis.entry(row, column) == other.basis.entry(row, column))
            })
    }
}

impl Eq for NumberFieldOrder {}

impl fmt::Debug for NumberFieldOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumberFieldOrder")
            .field("degree", &self.field.degree())
            .field("discriminant", &self.discriminant)
            .finish()
    }
}

impl fmt::Display for NumberFieldOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NumberFieldOrder(degree={}, discriminant={})",
            self.field.degree(),
            self.discriminant
        )
    }
}

impl SemanticType for NumberFieldOrder {
    const TYPE_ID: &'static str = "noether/numthy/NumberFieldOrder@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.field.canonical_bytes()?;
        append_component(&mut output, &self.denominator.canonical_bytes()?)?;
        for row in 0..self.basis.rows() {
            for column in 0..self.basis.columns() {
                append_component(
                    &mut output,
                    &self.basis.entry(row, column).canonical_bytes()?,
                )?;
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &NUMBER_FIELD_ORDER_DESCRIPTION
    }
}

impl Ring for NumberFieldOrder {
    type Element = NumberFieldElement;

    fn contains(&self, value: &Self::Element) -> bool {
        NumberFieldOrder::contains(self, value)
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

impl CommutativeRing for NumberFieldOrder {}
impl IntegralDomain for NumberFieldOrder {}

impl UnitInversion for NumberFieldOrder {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        if !self.contains(value) {
            return Err(self.mismatch(value).into());
        }
        let inverse = self.field.inverse(value).map_err(UnitError::from)?;
        if self.contains(&inverse) {
            Ok(inverse)
        } else {
            Err(UnitError::NotAUnit {
                context: Self::TYPE_ID,
            })
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumberFieldOrderRepr {
    #[serde(rename = "type")]
    type_id: String,
    field: NumberField,
    basis: Vec<Vec<String>>,
}

impl Serialize for NumberFieldOrder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        NumberFieldOrderRepr {
            type_id: Self::TYPE_ID.to_owned(),
            field: self.field.clone(),
            basis: self
                .basis()
                .into_iter()
                .map(|element| {
                    element
                        .coefficients()
                        .into_iter()
                        .map(|coefficient| coefficient.to_string())
                        .collect()
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for NumberFieldOrder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = NumberFieldOrderRepr::deserialize(deserializer)?;
        if repr.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                repr.type_id
            )));
        }
        let basis = repr
            .basis
            .iter()
            .map(|coefficients| {
                let coefficients = coefficients
                    .iter()
                    .map(|coefficient| {
                        coefficient
                            .parse::<Rational>()
                            .map_err(serde::de::Error::custom)
                    })
                    .collect::<Result<Vec<_>, D::Error>>()?;
                repr.field
                    .element(&coefficients)
                    .map_err(serde::de::Error::custom)
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        Self::new(repr.field, &basis).map_err(serde::de::Error::custom)
    }
}

impl NumberField {
    /// Validate the full-rank `Z`-order spanned by `basis`.
    pub fn order(
        &self,
        basis: &[NumberFieldElement],
    ) -> Result<NumberFieldOrder, NumberFieldOrderError> {
        NumberFieldOrder::new(self.clone(), basis)
    }

    /// The equation order `Z[a]` of the distinguished generator.
    ///
    /// This exists exactly when the defining monic polynomial has integer
    /// coefficients. A rational nonintegral presentation must use an explicit
    /// integral basis or the maximal-order computation.
    pub fn equation_order(&self) -> Result<NumberFieldOrder, NumberFieldOrderError> {
        if self
            .defining_coefficients()
            .iter()
            .any(|coefficient| !coefficient.denominator().is_one())
        {
            return Err(NumberFieldOrderError::GeneratorNotIntegral);
        }
        let generator = self.generator();
        let mut basis = Vec::with_capacity(self.degree());
        let mut power = self.one();
        for _ in 0..self.degree() {
            basis.push(power.clone());
            power = power.mul(&generator)?;
        }
        self.order(&basis)
    }

    fn integral_starting_order(&self) -> Result<NumberFieldOrder, NumberFieldOrderError> {
        match self.equation_order() {
            Ok(order) => return Ok(order),
            Err(NumberFieldOrderError::GeneratorNotIntegral) => {}
            Err(error) => return Err(error),
        }
        let denominator = self
            .defining_coefficients()
            .iter()
            .fold(Integer::one(), |current, coefficient| {
                integer_lcm(&current, &coefficient.denominator())
            });
        let integral_generator = self
            .generator()
            .mul(&self.scalar(Rational::from(denominator)))?;
        let mut basis = Vec::with_capacity(self.degree());
        let mut power = self.one();
        for _ in 0..self.degree() {
            basis.push(power.clone());
            power = power.mul(&integral_generator)?;
        }
        self.order(&basis)
    }

    /// Compute the maximal order (ring of integers) unconditionally.
    ///
    /// A rational presentation is first replaced by an integral generator.
    /// Round 2 then saturates at every prime whose square divides the starting
    /// order's discriminant. Complete FLINT factorization makes the result
    /// unconditional.
    pub fn maximal_order(&self) -> Result<NumberFieldOrder, NumberFieldOrderError> {
        if let Some(cached) = self.cached_maximal_order() {
            return cached.map(|cache| NumberFieldOrder::from_cache(self.clone(), cache));
        }
        let computed = self.compute_maximal_order();
        let cached = match &computed {
            Ok(order) => Ok(order.cache()),
            Err(error) => Err(error.clone()),
        };
        self.cache_maximal_order(cached);
        computed
    }

    fn compute_maximal_order(&self) -> Result<NumberFieldOrder, NumberFieldOrderError> {
        let mut order = self.integral_starting_order()?;
        let factorization =
            factor(&order.discriminant).map_err(|_| NumberFieldOrderError::Round2Invariant)?;
        for (prime, exponent) in factorization.factors() {
            if *exponent < 2 {
                continue;
            }
            let prime_word = prime
                .to_string()
                .parse::<u64>()
                .map_err(|_| NumberFieldOrderError::Round2PrimeTooLarge(prime.clone()))?;
            order = order.maximal_at_prime(prime_word)?;
        }
        Ok(order)
    }

    /// The field discriminant, namely the discriminant of the maximal order.
    pub fn discriminant(&self) -> Result<Integer, NumberFieldOrderError> {
        Ok(self.maximal_order()?.discriminant())
    }
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len()).map_err(|_| {
        NoetherError::Serialization(
            "number-field order component does not fit canonical framing".to_owned(),
        )
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}
