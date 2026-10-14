// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Prime ideals of maximal number-field orders.

#![allow(unsafe_code)]

use core::fmt;
use std::mem::MaybeUninit;

use flint_sys::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::number_field_order::integer_mod_u64;
use super::{FractionalIdeal, FractionalIdealError, NumberFieldElement, NumberFieldOrder};
use crate::commalg::{Ideal, PrimeIdeal};
use crate::{
    ContextMismatchError, Integer, NoetherError, Rational, Ring, SemanticType, VerificationError,
};

/// Failure to decompose a rational prime in a number-field order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimeDecompositionError {
    /// The input was not a positive proved rational prime.
    InvalidRationalPrime(Integer),
    /// The modular FLINT kernels require a word-sized rational prime.
    PrimeTooLarge(Integer),
    /// Unique prime-ideal factorization requires the maximal order.
    NonMaximalOrder,
    /// Exact order arithmetic failed.
    Order(super::NumberFieldOrderError),
    /// Exact ideal arithmetic failed.
    Ideal(FractionalIdealError),
    /// Values from incompatible number-field contexts were combined.
    ContextMismatch(ContextMismatchError),
    /// A proved invariant of the decomposition algorithm failed.
    Invariant(&'static str),
    /// Canonical serialized content was malformed.
    InvalidEncoding(String),
}

impl fmt::Display for PrimeDecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRationalPrime(prime) => {
                write!(
                    f,
                    "prime decomposition requires a positive proved prime, got {prime}"
                )
            }
            Self::PrimeTooLarge(prime) => write!(
                f,
                "prime decomposition cannot represent {prime} as a word-sized FLINT modulus"
            ),
            Self::NonMaximalOrder => {
                f.write_str("prime decomposition requires a maximal number-field order")
            }
            Self::Order(error) => error.fmt(f),
            Self::Ideal(error) => error.fmt(f),
            Self::ContextMismatch(error) => error.fmt(f),
            Self::Invariant(claim) => {
                write!(f, "prime-decomposition invariant failed: {claim}")
            }
            Self::InvalidEncoding(message) => {
                write!(f, "invalid number-field prime-ideal encoding: {message}")
            }
        }
    }
}

impl std::error::Error for PrimeDecompositionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Order(error) => Some(error),
            Self::Ideal(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<super::NumberFieldOrderError> for PrimeDecompositionError {
    fn from(value: super::NumberFieldOrderError) -> Self {
        Self::Order(value)
    }
}

impl From<FractionalIdealError> for PrimeDecompositionError {
    fn from(value: FractionalIdealError) -> Self {
        Self::Ideal(value)
    }
}

impl From<ContextMismatchError> for PrimeDecompositionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

struct ModMatrix {
    raw: nmod_mat_t,
    modulus: u64,
}

impl ModMatrix {
    fn zero(rows: usize, columns: usize, modulus: u64) -> Self {
        let mut raw: nmod_mat_t = unsafe { std::mem::zeroed() };
        unsafe { nmod_mat_init(raw.as_mut_ptr(), rows as slong, columns as slong, modulus) };
        Self { raw, modulus }
    }

    fn as_ptr(&self) -> *const nmod_mat_struct {
        self.raw.as_ptr()
    }

    fn as_mut_ptr(&mut self) -> *mut nmod_mat_struct {
        self.raw.as_mut_ptr()
    }

    fn set(&mut self, row: usize, column: usize, value: u64) {
        unsafe {
            nmod_mat_set_entry(
                self.as_mut_ptr(),
                row as slong,
                column as slong,
                value % self.modulus,
            )
        };
    }

    fn get(&self, row: usize, column: usize) -> u64 {
        unsafe { nmod_mat_get_entry(self.as_ptr(), row as slong, column as slong) }
    }

    fn rref(&mut self) -> usize {
        unsafe { nmod_mat_rref(self.as_mut_ptr()) as usize }
    }

    fn rank(&self) -> usize {
        unsafe { nmod_mat_rank(self.as_ptr()) as usize }
    }

    fn minimal_polynomial(&self) -> ModPolynomial {
        let mut polynomial = ModPolynomial::new(self.modulus);
        unsafe { nmod_mat_minpoly(polynomial.as_mut_ptr(), self.as_ptr()) };
        polynomial
    }
}

impl Drop for ModMatrix {
    fn drop(&mut self) {
        unsafe { nmod_mat_clear(self.as_mut_ptr()) };
    }
}

struct ModPolynomial {
    raw: nmod_poly_struct,
}

impl ModPolynomial {
    fn new(modulus: u64) -> Self {
        let mut raw = MaybeUninit::<nmod_poly_struct>::uninit();
        unsafe { nmod_poly_init(raw.as_mut_ptr(), modulus) };
        Self {
            raw: unsafe { raw.assume_init() },
        }
    }

    fn one(modulus: u64) -> Self {
        let mut output = Self::new(modulus);
        unsafe { nmod_poly_one(output.as_mut_ptr()) };
        output
    }

    fn from_coefficients(coefficients: &[u64], modulus: u64) -> Self {
        let mut output = Self::new(modulus);
        for (exponent, coefficient) in coefficients.iter().enumerate() {
            unsafe {
                nmod_poly_set_coeff_ui(
                    output.as_mut_ptr(),
                    exponent as slong,
                    coefficient % modulus,
                )
            };
        }
        output
    }

    fn modulus(&self) -> u64 {
        self.raw.mod_.n
    }

    fn as_ptr(&self) -> *const nmod_poly_struct {
        &self.raw
    }

    fn as_mut_ptr(&mut self) -> *mut nmod_poly_struct {
        &mut self.raw
    }

    fn degree(&self) -> Option<usize> {
        let degree = unsafe { nmod_poly_degree(self.as_ptr()) };
        (degree >= 0).then_some(degree as usize)
    }

    fn coefficient(&self, exponent: usize) -> u64 {
        unsafe { nmod_poly_get_coeff_ui(self.as_ptr(), exponent as slong) }
    }

    fn factor(&self) -> Vec<(Self, usize)> {
        let mut factorization = MaybeUninit::<nmod_poly_factor_struct>::uninit();
        unsafe {
            nmod_poly_factor_init(factorization.as_mut_ptr());
            let _ = nmod_poly_factor(factorization.as_mut_ptr(), self.as_ptr());
            let mut factorization = factorization.assume_init();
            let raw = &*(&factorization as *const nmod_poly_factor_struct);
            let mut factors = Vec::with_capacity(raw.num as usize);
            for index in 0..raw.num as usize {
                let mut factor = Self::new(self.modulus());
                nmod_poly_set(factor.as_mut_ptr(), raw.p.add(index));
                factors.push((factor, *raw.exp.add(index) as usize));
            }
            nmod_poly_factor_clear(&mut factorization);
            factors
        }
    }
}

impl Drop for ModPolynomial {
    fn drop(&mut self) {
        unsafe { nmod_poly_clear(self.as_mut_ptr()) };
    }
}

fn prime_as_word(prime: &Integer) -> Result<u64, PrimeDecompositionError> {
    prime
        .to_string()
        .parse::<u64>()
        .map_err(|_| PrimeDecompositionError::PrimeTooLarge(prime.clone()))
}

fn reduced_subspace(
    order: &NumberFieldOrder,
    ideal: &FractionalIdeal,
    prime: u64,
) -> Result<(Vec<usize>, Vec<Vec<u64>>), PrimeDecompositionError> {
    let degree = order.field().degree();
    let mut matrix = ModMatrix::zero(degree, degree, prime);
    for (row, element) in ideal.basis().iter().enumerate() {
        let coordinates =
            order
                .integral_coordinates(element)
                .ok_or(PrimeDecompositionError::Invariant(
                    "integral ideal basis left its coefficient order",
                ))?;
        for (column, coordinate) in coordinates.iter().enumerate() {
            matrix.set(row, column, integer_mod_u64(coordinate, prime));
        }
    }
    let rank = matrix.rref();
    let mut pivots = Vec::with_capacity(rank);
    let mut rows = Vec::with_capacity(rank);
    for row in 0..rank {
        let entries = (0..degree)
            .map(|column| matrix.get(row, column))
            .collect::<Vec<_>>();
        let Some(pivot) = entries.iter().position(|entry| *entry != 0) else {
            return Err(PrimeDecompositionError::Invariant(
                "modular RREF reported a zero pivot row",
            ));
        };
        pivots.push(pivot);
        rows.push(entries);
    }
    Ok((pivots, rows))
}

fn reduce_mod_subspace(vector: &mut [u64], pivots: &[usize], rows: &[Vec<u64>], prime: u64) {
    for (row, pivot) in rows.iter().zip(pivots) {
        let coefficient = vector[*pivot];
        if coefficient == 0 {
            continue;
        }
        for (entry, row_entry) in vector.iter_mut().zip(row) {
            let subtract =
                (u128::from(coefficient) * u128::from(*row_entry) % u128::from(prime)) as u64;
            *entry = (*entry + prime - subtract) % prime;
        }
    }
}

fn quotient_operator(
    order: &NumberFieldOrder,
    ideal: &FractionalIdeal,
    prime: u64,
    images: &[NumberFieldElement],
) -> Result<(ModMatrix, usize), PrimeDecompositionError> {
    let degree = order.field().degree();
    let (pivots, rows) = reduced_subspace(order, ideal, prime)?;
    let quotient_columns = (0..degree)
        .filter(|column| !pivots.contains(column))
        .collect::<Vec<_>>();
    let dimension = quotient_columns.len();
    if images.len() != degree {
        return Err(PrimeDecompositionError::Invariant(
            "quotient operator needs one image per order basis vector",
        ));
    }
    let mut operator = ModMatrix::zero(dimension, dimension, prime);
    for (column, basis_column) in quotient_columns.iter().enumerate() {
        let coordinates = order.integral_coordinates(&images[*basis_column]).ok_or(
            PrimeDecompositionError::Invariant("quotient operator image is not integral"),
        )?;
        let mut vector = coordinates
            .iter()
            .map(|coordinate| integer_mod_u64(coordinate, prime))
            .collect::<Vec<_>>();
        reduce_mod_subspace(&mut vector, &pivots, &rows, prime);
        for (row, basis_row) in quotient_columns.iter().enumerate() {
            operator.set(row, column, vector[*basis_row]);
        }
    }
    Ok((operator, dimension))
}

fn minimal_polynomial_mod_ideal(
    order: &NumberFieldOrder,
    ideal: &FractionalIdeal,
    element: &NumberFieldElement,
    prime: u64,
) -> Result<ModPolynomial, PrimeDecompositionError> {
    let images = order
        .basis()
        .iter()
        .map(|basis_element| element.mul(basis_element))
        .collect::<Result<Vec<_>, _>>()?;
    let (operator, dimension) = quotient_operator(order, ideal, prime, &images)?;
    if dimension == 0 {
        Ok(ModPolynomial::one(prime))
    } else {
        Ok(operator.minimal_polynomial())
    }
}

fn quotient_is_field(
    order: &NumberFieldOrder,
    ideal: &FractionalIdeal,
    prime: u64,
) -> Result<(bool, usize), PrimeDecompositionError> {
    let images = order
        .basis()
        .iter()
        .map(|basis_element| basis_element.pow(prime))
        .collect::<Result<Vec<_>, _>>()?;
    let (frobenius, dimension) = quotient_operator(order, ideal, prime, &images)?;
    if dimension == 0 || frobenius.rank() != dimension {
        return Ok((false, dimension));
    }

    let mut fixed = ModMatrix::zero(dimension, dimension, prime);
    for row in 0..dimension {
        for column in 0..dimension {
            let mut entry = frobenius.get(row, column);
            if row == column {
                entry = (entry + prime - 1) % prime;
            }
            fixed.set(row, column, entry);
        }
    }
    Ok((dimension - fixed.rank() == 1, dimension))
}

fn polynomial_at_element(
    order: &NumberFieldOrder,
    polynomial: &ModPolynomial,
    element: &NumberFieldElement,
) -> Result<NumberFieldElement, PrimeDecompositionError> {
    let Some(degree) = polynomial.degree() else {
        return Err(PrimeDecompositionError::Invariant(
            "an irreducible modular factor was zero",
        ));
    };
    let field = order.field();
    let mut value = field.zero();
    let mut power = field.one();
    for exponent in 0..=degree {
        let coefficient = polynomial.coefficient(exponent);
        if coefficient != 0 {
            value = value.add(&power.mul(&field.scalar(Rational::from(coefficient)))?)?;
        }
        if exponent != degree {
            power = power.mul(element)?;
        }
    }
    Ok(value)
}

fn kummer_primes(
    order: &NumberFieldOrder,
    prime: u64,
) -> Result<Option<Vec<FractionalIdeal>>, PrimeDecompositionError> {
    let equation_order = match order.field().equation_order() {
        Ok(equation_order) => equation_order,
        Err(super::NumberFieldOrderError::GeneratorNotIntegral) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let index = equation_order.index(order)?;
    if integer_mod_u64(&index, prime) == 0 {
        return Ok(None);
    }

    let coefficients = order
        .field()
        .defining_coefficients()
        .iter()
        .map(|coefficient| {
            if !coefficient.denominator().is_one() {
                return Err(PrimeDecompositionError::Invariant(
                    "equation-order polynomial unexpectedly had a denominator",
                ));
            }
            Ok(integer_mod_u64(&coefficient.numerator(), prime))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let polynomial = ModPolynomial::from_coefficients(&coefficients, prime);
    let theta = order.field().generator();
    let rational_prime = order.field().scalar(Rational::from(Integer::from(prime)));
    polynomial
        .factor()
        .into_iter()
        .map(|(factor, _)| {
            let value = polynomial_at_element(order, &factor, &theta)?;
            Ok(order.fractional_ideal(&[rational_prime.clone(), value])?)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn split_primes_over_radical(
    order: &NumberFieldOrder,
    prime: u64,
    radical: FractionalIdeal,
) -> Result<Vec<FractionalIdeal>, PrimeDecompositionError> {
    let mut pending = vec![radical];
    let mut primes = Vec::new();
    while let Some(ideal) = pending.pop() {
        let mut separated = false;
        for basis_element in order.basis() {
            let minimal = minimal_polynomial_mod_ideal(order, &ideal, &basis_element, prime)?;
            let factors = minimal.factor();
            let splits = factors.len() > 1
                || factors
                    .first()
                    .is_some_and(|(_, multiplicity)| *multiplicity > 1);
            if !splits {
                continue;
            }
            for (factor, _) in factors {
                let value = polynomial_at_element(order, &factor, &basis_element)?;
                let principal = order.fractional_ideal(&[value])?;
                let split = ideal.add(&principal)?;
                if split != ideal && !pending.iter().any(|candidate| candidate == &split) {
                    pending.push(split);
                    separated = true;
                }
            }
            if separated {
                break;
            }
        }
        if separated {
            continue;
        }
        let (is_field, _) = quotient_is_field(order, &ideal, prime)?;
        if !is_field {
            return Err(PrimeDecompositionError::Invariant(
                "residue algebra could not be separated into fields",
            ));
        }
        if !primes.iter().any(|candidate| candidate == &ideal) {
            primes.push(ideal);
        }
    }
    Ok(primes)
}

fn subset(left: &FractionalIdeal, right: &FractionalIdeal) -> bool {
    left.basis().iter().all(|element| right.contains(element))
}

fn ramification_index(
    order: &NumberFieldOrder,
    rational_prime_ideal: &FractionalIdeal,
    prime_ideal: &FractionalIdeal,
) -> Result<usize, PrimeDecompositionError> {
    let unit = order.fractional_ideal(&[order.one()])?;
    let mut power = unit;
    let mut valuation = 0;
    loop {
        let next = power.mul(prime_ideal)?;
        if !subset(rational_prime_ideal, &next) {
            break;
        }
        valuation += 1;
        if valuation > order.field().degree() {
            return Err(PrimeDecompositionError::Invariant(
                "ramification index exceeded the field degree",
            ));
        }
        power = next;
    }
    if valuation == 0 {
        return Err(PrimeDecompositionError::Invariant(
            "prime ideal did not contain its rational prime",
        ));
    }
    Ok(valuation)
}

fn encode_witness(prime: &Integer, ramification: usize, residue_degree: usize) -> Vec<u8> {
    format!("{prime}:{ramification}:{residue_degree}").into_bytes()
}

fn decode_witness(witness: &[u8]) -> Result<(Integer, usize, usize), PrimeDecompositionError> {
    let encoded = std::str::from_utf8(witness)
        .map_err(|error| PrimeDecompositionError::InvalidEncoding(error.to_string()))?;
    let mut parts = encoded.split(':');
    let prime = parts
        .next()
        .ok_or_else(|| PrimeDecompositionError::InvalidEncoding("missing prime".to_owned()))
        .and_then(|value| {
            Integer::from_str_radix(value, 10)
                .map_err(|error| PrimeDecompositionError::InvalidEncoding(error.to_string()))
        })?;
    let ramification = parts
        .next()
        .ok_or_else(|| {
            PrimeDecompositionError::InvalidEncoding("missing ramification index".to_owned())
        })?
        .parse::<usize>()
        .map_err(|error| PrimeDecompositionError::InvalidEncoding(error.to_string()))?;
    let residue_degree = parts
        .next()
        .ok_or_else(|| {
            PrimeDecompositionError::InvalidEncoding("missing residue degree".to_owned())
        })?
        .parse::<usize>()
        .map_err(|error| PrimeDecompositionError::InvalidEncoding(error.to_string()))?;
    if parts.next().is_some() || ramification == 0 || residue_degree == 0 {
        return Err(PrimeDecompositionError::InvalidEncoding(
            "prime-ideal witness has invalid fields".to_owned(),
        ));
    }
    Ok((prime, ramification, residue_degree))
}

fn fractional_from_ideal(
    ideal: &Ideal<NumberFieldOrder>,
) -> Result<FractionalIdeal, PrimeDecompositionError> {
    Ok(ideal.ring().fractional_ideal(ideal.generators())?)
}

fn verify_number_field_prime(
    order: &NumberFieldOrder,
    ideal: &Ideal<NumberFieldOrder>,
    witness: &[u8],
) -> Result<(), PrimeDecompositionError> {
    if ideal.ring() != order {
        return Err(PrimeDecompositionError::Invariant(
            "prime ideal carries a different coefficient order",
        ));
    }
    if !order.is_maximal()? {
        return Err(PrimeDecompositionError::NonMaximalOrder);
    }
    let (prime, claimed_ramification, claimed_residue_degree) = decode_witness(witness)?;
    if prime.sign() <= 0 || !prime.is_prime() {
        return Err(PrimeDecompositionError::InvalidRationalPrime(prime));
    }
    let prime_word = prime_as_word(&prime)?;
    let fractional = fractional_from_ideal(ideal)?;
    let (is_field, residue_degree) = quotient_is_field(order, &fractional, prime_word)?;
    if !is_field || residue_degree != claimed_residue_degree {
        return Err(PrimeDecompositionError::Invariant(
            "residue quotient is not the claimed finite field",
        ));
    }
    let expected_norm = prime.pow(residue_degree as u64);
    if fractional.norm() != Rational::from(expected_norm) {
        return Err(PrimeDecompositionError::Invariant(
            "prime-ideal norm is not p to the residue degree",
        ));
    }
    let rational = order.fractional_ideal(&[order.field().scalar(Rational::from(prime))])?;
    let ramification = ramification_index(order, &rational, &fractional)?;
    if ramification != claimed_ramification {
        return Err(PrimeDecompositionError::Invariant(
            "ramification index does not match the ideal valuation of p",
        ));
    }
    Ok(())
}

impl NumberFieldOrder {
    /// Decompose `(p)` into checked prime ideals of this maximal order.
    ///
    /// Kummer--Dedekind factorization is the fast path when the distinguished
    /// equation order is `p`-maximal. Native residue-algebra splitting handles
    /// index primes and rational presentations. Every result is verified as a
    /// field quotient, and the product is checked to reconstruct `(p)`.
    pub fn prime_ideals_above(
        &self,
        prime: &Integer,
    ) -> Result<Vec<PrimeIdeal<NumberFieldOrder>>, PrimeDecompositionError> {
        if prime.sign() <= 0 || !prime.is_prime() {
            return Err(PrimeDecompositionError::InvalidRationalPrime(prime.clone()));
        }
        let prime_word = prime_as_word(prime)?;
        if !self.is_maximal()? {
            return Err(PrimeDecompositionError::NonMaximalOrder);
        }
        if let Some(cached) = self.cached_prime_decomposition(prime) {
            return Ok(cached);
        }

        let mut raw_primes = match kummer_primes(self, prime_word)? {
            Some(primes) => primes,
            None => {
                let radical = self.fractional_ideal(&self.p_radical(prime_word)?)?;
                split_primes_over_radical(self, prime_word, radical)?
            }
        };
        raw_primes.sort_by(|left, right| {
            left.canonical_bytes()
                .expect("canonical exact ideals serialize")
                .cmp(
                    &right
                        .canonical_bytes()
                        .expect("canonical exact ideals serialize"),
                )
        });
        let rational =
            self.fractional_ideal(&[self.field().scalar(Rational::from(prime.clone()))])?;
        let mut decomposition = Vec::with_capacity(raw_primes.len());
        let mut sum_ef = 0usize;
        let mut product = self.fractional_ideal(&[self.one()])?;
        for raw_prime in raw_primes {
            let (is_field, residue_degree) = quotient_is_field(self, &raw_prime, prime_word)?;
            if !is_field {
                return Err(PrimeDecompositionError::Invariant(
                    "prime candidate had a nonfield residue quotient",
                ));
            }
            if raw_prime.norm() != Rational::from(prime.pow(residue_degree as u64)) {
                return Err(PrimeDecompositionError::Invariant(
                    "prime candidate norm was not p to the residue degree",
                ));
            }
            let ramification = ramification_index(self, &rational, &raw_prime)?;
            sum_ef = sum_ef.checked_add(ramification * residue_degree).ok_or(
                PrimeDecompositionError::Invariant("ramification-residue sum overflowed"),
            )?;
            for _ in 0..ramification {
                product = product.mul(&raw_prime)?;
            }
            let ideal = raw_prime.as_ideal()?;
            let witness = encode_witness(prime, ramification, residue_degree);
            decomposition.push(PrimeIdeal::from_verified_parts(ideal, witness));
        }
        if sum_ef != self.field().degree() {
            return Err(PrimeDecompositionError::Invariant(
                "sum of ramification times residue degrees did not equal the field degree",
            ));
        }
        if product != rational {
            return Err(PrimeDecompositionError::Invariant(
                "prime-ideal product did not reconstruct the rational prime ideal",
            ));
        }
        self.cache_prime_decomposition(prime.clone(), &decomposition);
        Ok(decomposition)
    }
}

impl PrimeIdeal<NumberFieldOrder> {
    /// The coefficient order of this prime ideal.
    pub fn order(&self) -> &NumberFieldOrder {
        self.ideal().ring()
    }

    /// Canonical integral basis of the prime ideal.
    pub fn basis(&self) -> &[NumberFieldElement] {
        self.ideal().generators()
    }

    /// The positive rational prime below this ideal.
    pub fn rational_prime(&self) -> Integer {
        decode_witness(self.witness())
            .expect("a constructed prime ideal has a validated witness")
            .0
    }

    /// Ramification index over the rational prime.
    pub fn ramification_index(&self) -> usize {
        decode_witness(self.witness())
            .expect("a constructed prime ideal has a validated witness")
            .1
    }

    /// Residue-field degree over the rational prime field.
    pub fn residue_degree(&self) -> usize {
        decode_witness(self.witness())
            .expect("a constructed prime ideal has a validated witness")
            .2
    }

    /// Absolute norm `p^f`.
    pub fn norm(&self) -> Integer {
        self.rational_prime().pow(self.residue_degree() as u64)
    }

    /// Whether an exact field element belongs to this prime ideal.
    pub fn contains(&self, element: &NumberFieldElement) -> bool {
        fractional_from_ideal(self.ideal()).is_ok_and(|ideal| ideal.contains(element))
    }

    /// Recheck primality and the ramification/residue witness.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_number_field_prime(self.order(), self.ideal(), self.witness()).map_err(|_| {
            VerificationError::InvalidWitness {
                claim:
                    "number-field ideal is prime with the claimed ramification and residue degree",
            }
        })
    }

    /// The same integral ideal viewed in fractional-ideal arithmetic.
    pub fn as_fractional_ideal(&self) -> FractionalIdeal {
        fractional_from_ideal(self.ideal())
            .expect("a checked number-field prime ideal is an integral fractional ideal")
    }

    /// Serialize this checked prime ideal to canonical JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and reverify a checked number-field prime ideal.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Ideal<NumberFieldOrder> {
    /// Whether an exact field element belongs to this integral ideal.
    pub fn contains(&self, element: &NumberFieldElement) -> bool {
        self.as_fractional_ideal().contains(element)
    }

    /// Absolute integral-ideal norm.
    pub fn norm(&self) -> Integer {
        let norm = self.as_fractional_ideal().norm();
        debug_assert!(norm.denominator().is_one());
        norm.numerator()
    }

    /// View this integral ideal in fractional-ideal arithmetic.
    pub fn as_fractional_ideal(&self) -> FractionalIdeal {
        fractional_from_ideal(self)
            .expect("a canonical number-field ideal reconstructs over its coefficient order")
    }

    /// Recompute and verify the canonical integral lattice basis.
    pub fn verify(&self) -> Result<(), FractionalIdealError> {
        let rebuilt = self
            .ring()
            .fractional_ideal(self.generators())?
            .as_ideal()?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(FractionalIdealError::InvalidLattice)
        }
    }

    /// Serialize this canonical integral ideal to JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and revalidate a canonical integral ideal from JSON.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumberFieldIdealRepr {
    order: NumberFieldOrder,
    basis: Vec<Vec<String>>,
}

impl Serialize for Ideal<NumberFieldOrder> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        NumberFieldIdealRepr {
            order: self.ring().clone(),
            basis: self
                .generators()
                .iter()
                .map(|element| {
                    element
                        .coefficients()
                        .iter()
                        .map(ToString::to_string)
                        .collect()
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Ideal<NumberFieldOrder> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = NumberFieldIdealRepr::deserialize(deserializer)?;
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
                repr.order
                    .field()
                    .element(&coefficients)
                    .map_err(serde::de::Error::custom)
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        repr.order
            .fractional_ideal(&basis)
            .and_then(|ideal| ideal.as_ideal())
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumberFieldPrimeIdealRepr {
    order: NumberFieldOrder,
    basis: Vec<Vec<String>>,
    rational_prime: String,
    ramification_index: usize,
    residue_degree: usize,
}

impl Serialize for PrimeIdeal<NumberFieldOrder> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        NumberFieldPrimeIdealRepr {
            order: self.order().clone(),
            basis: self
                .basis()
                .iter()
                .map(|element| {
                    element
                        .coefficients()
                        .iter()
                        .map(ToString::to_string)
                        .collect()
                })
                .collect(),
            rational_prime: self.rational_prime().to_string(),
            ramification_index: self.ramification_index(),
            residue_degree: self.residue_degree(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PrimeIdeal<NumberFieldOrder> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = NumberFieldPrimeIdealRepr::deserialize(deserializer)?;
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
                repr.order
                    .field()
                    .element(&coefficients)
                    .map_err(serde::de::Error::custom)
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        let fractional = repr
            .order
            .fractional_ideal(&basis)
            .map_err(serde::de::Error::custom)?;
        let ideal = fractional.as_ideal().map_err(serde::de::Error::custom)?;
        let prime =
            Integer::from_str_radix(&repr.rational_prime, 10).map_err(serde::de::Error::custom)?;
        let witness = encode_witness(&prime, repr.ramification_index, repr.residue_degree);
        let checked = Self::from_verified_parts(ideal, witness);
        checked.verify().map_err(serde::de::Error::custom)?;
        Ok(checked)
    }
}
