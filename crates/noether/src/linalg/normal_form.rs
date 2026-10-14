// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact integer matrix normal forms.

use core::fmt;

use crate::{
    Assumption, EuclideanDomain, Integer, IntegerRing, MathematicalValidity, NoetherError,
    SemanticType, TypeDescription, VerificationError, Z,
};

use super::{LinalgError, Matrix};

const MAX_SMOOTH_BOUND: u64 = 100_000;

static SMITH_NORMAL_FORM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/SmithNormalForm@1",
    rust_path: "noether::linalg::SmithNormalForm",
    python_path: Some("noether.linalg.SmithNormalForm"),
};

static SMITH_NORMAL_FORM_WITH_TRANSFORMS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/SmithNormalFormWithTransforms@1",
    rust_path: "noether::linalg::SmithNormalFormWithTransforms",
    python_path: Some("noether.linalg.SmithNormalFormWithTransforms"),
};

static SMITH_VERIFICATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/SmithVerification@1",
    rust_path: "noether::linalg::SmithVerification",
    python_path: Some("noether.linalg.SmithVerification"),
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum SmithStrategy {
    Exact,
    SmoothLocal,
}

/// Result-specific evidence for an integer Smith computation.
///
/// The exact strategy records no prime sweep. The smooth-local strategy records
/// every checked prime, the p-adic precision used, and the recovered valuations
/// of the nonzero invariant factors. Its smooth and deterministic bounds make
/// the distinction between conditional and unconditional validity explicit.
#[derive(Clone, PartialEq, Eq)]
pub struct SmithVerification {
    strategy: SmithStrategy,
    checked_primes: Vec<Integer>,
    local_precisions: Vec<u32>,
    local_valuations: Vec<Vec<u32>>,
    smooth_bound: Option<Integer>,
    deterministic_prime_bound: Option<Integer>,
}

impl SmithVerification {
    fn exact() -> Self {
        Self {
            strategy: SmithStrategy::Exact,
            checked_primes: Vec::new(),
            local_precisions: Vec::new(),
            local_valuations: Vec::new(),
            smooth_bound: None,
            deterministic_prime_bound: None,
        }
    }

    /// Stable mathematical strategy name.
    pub fn strategy(&self) -> &'static str {
        match self.strategy {
            SmithStrategy::Exact => "exact",
            SmithStrategy::SmoothLocal => "smooth_local",
        }
    }

    /// Inclusive prime bound checked by the smooth strategy.
    pub fn smooth_bound(&self) -> Option<&Integer> {
        self.smooth_bound.as_ref()
    }

    /// Deterministic upper bound on any torsion prime.
    pub fn deterministic_prime_bound(&self) -> Option<&Integer> {
        self.deterministic_prime_bound.as_ref()
    }

    /// Primes checked by the smooth strategy, in increasing order.
    pub fn checked_primes(&self) -> &[Integer] {
        &self.checked_primes
    }

    /// Iterate over `(prime, precision exponent, invariant valuations)` checks.
    pub fn local_checks(&self) -> impl Iterator<Item = (&Integer, u32, &[u32])> {
        self.checked_primes
            .iter()
            .zip(self.local_precisions.iter().copied())
            .zip(self.local_valuations.iter())
            .map(|((prime, precision), valuations)| (prime, precision, valuations.as_slice()))
    }
}

/// The Smith normal form of an integer matrix, without transformation matrices.
///
/// This is the fast default result. It records the diagonal form and invariant
/// factors, but deliberately does not compute left and right unimodular
/// transformations. A transform-producing calculation is a separate, explicit
/// operation because it can be much more expensive.
#[derive(Clone, PartialEq, Eq)]
pub struct SmithNormalForm {
    source: Matrix<IntegerRing>,
    form: Matrix<IntegerRing>,
    invariant_factors: Vec<Integer>,
    validity: MathematicalValidity,
    verification: SmithVerification,
}

impl SmithNormalForm {
    fn exact(source: Matrix<IntegerRing>, form: Matrix<IntegerRing>) -> Self {
        let invariant_factors = extract_invariant_factors(&form)
            .expect("FLINT returned a canonical integer Smith normal form");
        Self {
            source,
            form,
            invariant_factors,
            validity: MathematicalValidity::unconditional(),
            verification: SmithVerification::exact(),
        }
    }

    /// The input matrix retained as the result's verification witness.
    pub fn source(&self) -> &Matrix<IntegerRing> {
        &self.source
    }

    /// The diagonal Smith form, with the same shape as the input matrix.
    pub fn form(&self) -> &Matrix<IntegerRing> {
        &self.form
    }

    /// Nonzero diagonal entries in positive divisibility-chain order.
    pub fn invariant_factors(&self) -> &[Integer] {
        &self.invariant_factors
    }

    /// Rank of the input over the integers (equivalently over the rationals).
    pub fn rank(&self) -> usize {
        self.invariant_factors.len()
    }

    /// Mathematical assumptions under which this result is valid.
    pub fn validity(&self) -> &MathematicalValidity {
        &self.validity
    }

    /// Strategy-specific checks supporting the result.
    pub fn verification(&self) -> &SmithVerification {
        &self.verification
    }

    /// Recheck the diagonal conventions and recompute the form from the source.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let Some(extracted) = extract_invariant_factors(&self.form) else {
            return Err(VerificationError::InvalidWitness {
                claim: "Smith form is diagonal with positive divisibility-chain entries",
            });
        };
        if extracted != self.invariant_factors {
            return Err(VerificationError::InvalidWitness {
                claim: "Smith invariants match the diagonal form",
            });
        }
        match self.verification.strategy {
            SmithStrategy::Exact => {
                if !self.validity.is_unconditional()
                    || self.verification != SmithVerification::exact()
                    || self.source.smith_normal_form_backend() != self.form
                {
                    return Err(VerificationError::InvalidWitness {
                        claim: "the exact Smith strategy reproduces an unconditional form",
                    });
                }
            }
            SmithStrategy::SmoothLocal => {
                let Some(bound) = self
                    .verification
                    .smooth_bound
                    .as_ref()
                    .and_then(integer_to_u64)
                else {
                    return Err(VerificationError::InvalidWitness {
                        claim: "smooth Smith evidence has a word-sized bound",
                    });
                };
                let recomputed = compute_smooth_result(&self.source, bound).map_err(|_| {
                    VerificationError::InvalidWitness {
                        claim: "smooth Smith evidence can be deterministically recomputed",
                    }
                })?;
                if recomputed.form != self.form
                    || recomputed.invariant_factors != self.invariant_factors
                    || recomputed.validity != self.validity
                    || recomputed.verification != self.verification
                {
                    return Err(VerificationError::InvalidWitness {
                        claim: "smooth Smith evidence reproduces the conditional form",
                    });
                }
            }
        }
        Ok(())
    }
}

impl Matrix<IntegerRing> {
    /// Row Hermite form and its left unimodular basis change for homological
    /// playbooks that need a kernel splitting without a large Smith left
    /// transform.
    pub(crate) fn hermite_normal_form_with_transform(&self) -> (Self, Self) {
        self.hermite_normal_form_with_transform_backend()
    }

    /// Compute Smith normal form without transformation matrices.
    ///
    /// This dense exact route uses FLINT's arbitrary-precision integer matrix
    /// implementation. It is distinct from the future explicit
    /// transform-producing operation and from mathematical playbooks that can
    /// prove a smooth-case answer from modular evidence.
    pub fn smith_normal_form(&self) -> SmithNormalForm {
        let form = self.smith_normal_form_backend();
        SmithNormalForm::exact(self.clone(), form)
    }

    /// Compute Smith form from all local parts through `smooth_bound`.
    ///
    /// The result is conditional on the torsion order being
    /// `smooth_bound`-smooth unless the checked range covers the recorded
    /// deterministic prime bound. The bound is inclusive and capped to keep
    /// this direct public call predictably bounded.
    pub fn smith_normal_form_assuming_smooth(
        &self,
        smooth_bound: u64,
    ) -> Result<SmithNormalForm, LinalgError> {
        let result = compute_smooth_result(self, smooth_bound)?;
        result.verify()?;
        Ok(result)
    }

    /// Compute Smith normal form and both unimodular transformation matrices.
    ///
    /// This is an explicit, potentially much more expensive operation. The
    /// fast [`Self::smith_normal_form`] path never computes these witnesses.
    pub fn smith_normal_form_with_transforms(
        &self,
    ) -> Result<SmithNormalFormWithTransforms, super::LinalgError> {
        let (form, left, right) = self.smith_normal_form_with_transforms_backend();
        let result = SmithNormalFormWithTransforms {
            smith: SmithNormalForm::exact(self.clone(), form),
            left,
            right,
        };
        result.verify()?;
        Ok(result)
    }
}

/// Smith normal form together with explicit unimodular witnesses.
///
/// For a source matrix `A`, the result certifies `left * A * right = form`.
/// This richer result is produced only by the explicitly named transform
/// operation.
#[derive(Clone, PartialEq, Eq)]
pub struct SmithNormalFormWithTransforms {
    smith: SmithNormalForm,
    left: Matrix<IntegerRing>,
    right: Matrix<IntegerRing>,
}

impl SmithNormalFormWithTransforms {
    /// The Smith result shared with the fast no-transform endpoint.
    pub fn smith(&self) -> &SmithNormalForm {
        &self.smith
    }

    /// The left unimodular matrix.
    pub fn left(&self) -> &Matrix<IntegerRing> {
        &self.left
    }

    /// The right unimodular matrix.
    pub fn right(&self) -> &Matrix<IntegerRing> {
        &self.right
    }

    /// Verify canonical form, unimodularity, and the reconstruction identity.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.smith.verify()?;
        let (rows, columns) = self.smith.source().shape();
        if self.left.shape() != (rows, rows) || self.right.shape() != (columns, columns) {
            return Err(VerificationError::InvalidWitness {
                claim: "Smith transformation matrices have the required square shapes",
            });
        }
        if !self.left.determinant_backend().abs().is_one()
            || !self.right.determinant_backend().abs().is_one()
        {
            return Err(VerificationError::InvalidWitness {
                claim: "Smith transformation matrices are unimodular",
            });
        }
        let reconstructed = self
            .left
            .matmul(self.smith.source())
            .and_then(|product| product.matmul(&self.right))
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "Smith transformation matrices compose with the source",
            })?;
        if reconstructed != *self.smith.form() {
            return Err(VerificationError::InvalidWitness {
                claim: "left * source * right equals the Smith form",
            });
        }
        Ok(())
    }
}

impl fmt::Debug for SmithNormalForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmithNormalForm")
            .field("shape", &self.form.shape())
            .field("rank", &self.rank())
            .field("invariant_factors", &self.invariant_factors)
            .field("validity", &self.validity)
            .field("strategy", &self.verification.strategy())
            .finish()
    }
}

impl fmt::Display for SmithNormalForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SmithNormalForm(shape={}x{}, rank={}, invariants=[",
            self.form.shape().0,
            self.form.shape().1,
            self.rank()
        )?;
        for (index, invariant) in self.invariant_factors.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            invariant.fmt(f)?;
        }
        f.write_str("])")
    }
}

impl SemanticType for SmithNormalForm {
    const TYPE_ID: &'static str = "noether/linalg/SmithNormalForm@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        push_frame(&mut output, &self.source.canonical_bytes()?)?;
        push_frame(&mut output, &self.form.canonical_bytes()?)?;
        push_frame(&mut output, &self.validity.canonical_bytes()?)?;
        push_frame(&mut output, &self.verification.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SMITH_NORMAL_FORM_DESCRIPTION
    }
}

impl fmt::Debug for SmithVerification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmithVerification")
            .field("strategy", &self.strategy())
            .field("smooth_bound", &self.smooth_bound)
            .field("deterministic_prime_bound", &self.deterministic_prime_bound)
            .field("checked_primes", &self.checked_primes)
            .finish()
    }
}

impl fmt::Display for SmithVerification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SmithVerification(strategy={}", self.strategy())?;
        if let Some(bound) = &self.smooth_bound {
            write!(f, ", smooth_bound={bound}")?;
        }
        write!(f, ", checked_primes={})", self.checked_primes.len())
    }
}

impl SemanticType for SmithVerification {
    const TYPE_ID: &'static str = "noether/linalg/SmithVerification@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.strategy().as_bytes().to_vec();
        for bound in [&self.smooth_bound, &self.deterministic_prime_bound] {
            match bound {
                Some(value) => push_frame(&mut output, value.to_string().as_bytes())?,
                None => push_frame(&mut output, &[])?,
            }
        }
        for ((prime, precision), valuations) in self
            .checked_primes
            .iter()
            .zip(&self.local_precisions)
            .zip(&self.local_valuations)
        {
            push_frame(&mut output, prime.to_string().as_bytes())?;
            output.extend_from_slice(&precision.to_be_bytes());
            for valuation in valuations {
                output.extend_from_slice(&valuation.to_be_bytes());
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SMITH_VERIFICATION_DESCRIPTION
    }
}

impl fmt::Debug for SmithNormalFormWithTransforms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmithNormalFormWithTransforms")
            .field("smith", &self.smith)
            .field("left_shape", &self.left.shape())
            .field("right_shape", &self.right.shape())
            .finish()
    }
}

impl fmt::Display for SmithNormalFormWithTransforms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SmithNormalFormWithTransforms({})", self.smith)
    }
}

impl SemanticType for SmithNormalFormWithTransforms {
    const TYPE_ID: &'static str = "noether/linalg/SmithNormalFormWithTransforms@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        push_frame(&mut output, &self.smith.canonical_bytes()?)?;
        push_frame(&mut output, &self.left.canonical_bytes()?)?;
        push_frame(&mut output, &self.right.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SMITH_NORMAL_FORM_WITH_TRANSFORMS_DESCRIPTION
    }
}

fn compute_smooth_result(
    source: &Matrix<IntegerRing>,
    smooth_bound: u64,
) -> Result<SmithNormalForm, LinalgError> {
    if smooth_bound < 2 {
        return Err(LinalgError::InvalidSmoothBound {
            bound: smooth_bound,
        });
    }
    if smooth_bound > MAX_SMOOTH_BOUND {
        return Err(LinalgError::SmoothBoundTooLarge {
            bound: smooth_bound,
            maximum: MAX_SMOOTH_BOUND,
        });
    }

    let rank = source.rank_backend();
    let primes = primes_through(smooth_bound);
    let mut invariant_factors = vec![Integer::one(); rank];
    let mut checked_primes = Vec::with_capacity(primes.len());
    let mut local_precisions = Vec::with_capacity(primes.len());
    let mut local_valuations = Vec::with_capacity(primes.len());

    for prime in primes {
        let (valuations, precision) = local_smith_valuations(source, prime, rank)?;
        let prime_value = Integer::from(prime);
        for (factor, valuation) in invariant_factors.iter_mut().zip(&valuations) {
            if *valuation != 0 {
                *factor = &*factor * &prime_value.pow(u64::from(*valuation));
            }
        }
        checked_primes.push(prime_value);
        local_precisions.push(precision);
        local_valuations.push(valuations);
    }

    let deterministic_prime_bound = smith_torsion_bound(source, rank);
    let smooth_bound_integer = Integer::from(smooth_bound);
    let validity = if smooth_bound_integer >= deterministic_prime_bound {
        MathematicalValidity::unconditional()
    } else {
        MathematicalValidity::conditional([Assumption::TorsionOrderIsSmooth {
            bound: smooth_bound_integer.clone(),
        }])
    };
    let form = smith_form_from_factors(source.shape(), &invariant_factors)?;
    Ok(SmithNormalForm {
        source: source.clone(),
        form,
        invariant_factors,
        validity,
        verification: SmithVerification {
            strategy: SmithStrategy::SmoothLocal,
            checked_primes,
            local_precisions,
            local_valuations,
            smooth_bound: Some(smooth_bound_integer),
            deterministic_prime_bound: Some(deterministic_prime_bound),
        },
    })
}

fn smith_form_from_factors(
    shape: (usize, usize),
    invariant_factors: &[Integer],
) -> Result<Matrix<IntegerRing>, LinalgError> {
    let mut entries = vec![Integer::zero(); shape.0 * shape.1];
    for (index, factor) in invariant_factors.iter().enumerate() {
        entries[index * shape.1 + index] = factor.clone();
    }
    Matrix::from_entries(Z, shape.0, shape.1, entries)
}

fn smith_torsion_bound(source: &Matrix<IntegerRing>, rank: usize) -> Integer {
    if rank == 0 {
        return Integer::one();
    }
    let maximum = source
        .rows()
        .flatten()
        .map(|value| value.abs())
        .max()
        .expect("positive-rank matrix has an entry");
    // Hadamard: every rank minor is at most r^(r/2) M^r. The torsion
    // order divides every nonzero rank minor, so this integral ceiling also
    // bounds every possible torsion prime.
    &Integer::from(rank).pow(rank.div_ceil(2) as u64) * &maximum.pow(rank as u64)
}

fn primes_through(bound: u64) -> Vec<u64> {
    let mut primes = Vec::new();
    for candidate in 2..=bound {
        if primes
            .iter()
            .take_while(|prime| **prime <= candidate / **prime)
            .all(|prime| !candidate.is_multiple_of(*prime))
        {
            primes.push(candidate);
        }
    }
    primes
}

fn local_smith_valuations(
    source: &Matrix<IntegerRing>,
    prime: u64,
    rank: usize,
) -> Result<(Vec<u32>, u32), LinalgError> {
    let mut modulus = prime;
    let mut precision = 1_u32;
    while let Some(next) = modulus.checked_mul(prime) {
        modulus = next;
        precision += 1;
    }
    let (rows, columns) = source.shape();
    let mut active = (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| {
                    source
                        .get(row, column)
                        .expect("source coordinates are in bounds")
                        .modulo_u64(modulus)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut valuations = Vec::with_capacity(rank);
    let mut current_modulus = modulus;

    for valuation in 0..precision {
        let (unit_count, next) = local_unit_elimination(active, prime, current_modulus)?;
        valuations.extend(std::iter::repeat_n(valuation, unit_count));
        if valuations.len() == rank {
            return Ok((valuations, precision));
        }
        if valuations.len() > rank {
            return Err(LinalgError::Verification(
                VerificationError::InvalidWitness {
                    claim: "local Smith rank does not exceed exact rank",
                },
            ));
        }
        active = next;
        current_modulus /= prime;
    }

    Err(LinalgError::SmoothValuationTooLarge {
        prime,
        maximum_exponent: precision,
    })
}

fn local_unit_elimination(
    mut matrix: Vec<Vec<u64>>,
    prime: u64,
    modulus: u64,
) -> Result<(usize, Vec<Vec<u64>>), LinalgError> {
    let rows = matrix.len();
    let columns = matrix.first().map_or(0, Vec::len);
    let mut pivot_columns = vec![false; columns];
    let mut rank = 0;

    for column in 0..columns {
        let Some(pivot) = (rank..rows).find(|row| !matrix[*row][column].is_multiple_of(prime))
        else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse = modular_inverse(matrix[rank][column], modulus).ok_or(
            LinalgError::Verification(VerificationError::InvalidWitness {
                claim: "a local Smith unit pivot is invertible",
            }),
        )?;
        let pivot_row = matrix[rank].clone();
        for row in matrix.iter_mut().take(rows).skip(rank + 1) {
            let factor = multiply_mod(row[column], inverse, modulus);
            if factor == 0 {
                continue;
            }
            for (entry, pivot) in row.iter_mut().zip(&pivot_row) {
                let product = multiply_mod(factor, *pivot, modulus);
                *entry = subtract_mod(*entry, product, modulus);
            }
        }
        pivot_columns[column] = true;
        rank += 1;
        if rank == rows {
            break;
        }
    }

    let next_modulus = modulus / prime;
    let mut next = Vec::with_capacity(rows - rank);
    for row in matrix.into_iter().skip(rank) {
        let mut next_row = Vec::with_capacity(columns - rank);
        for (column, value) in row.into_iter().enumerate() {
            if pivot_columns[column] {
                continue;
            }
            if !value.is_multiple_of(prime) {
                return Err(LinalgError::Verification(
                    VerificationError::InvalidWitness {
                        claim: "the local Smith Schur complement is divisible by its prime",
                    },
                ));
            }
            next_row.push(if next_modulus == 1 {
                0
            } else {
                (value / prime) % next_modulus
            });
        }
        next.push(next_row);
    }
    Ok((rank, next))
}

fn multiply_mod(left: u64, right: u64, modulus: u64) -> u64 {
    (u128::from(left) * u128::from(right) % u128::from(modulus)) as u64
}

fn subtract_mod(left: u64, right: u64, modulus: u64) -> u64 {
    ((u128::from(left) + u128::from(modulus) - u128::from(right)) % u128::from(modulus)) as u64
}

fn modular_inverse(value: u64, modulus: u64) -> Option<u64> {
    let (mut old_r, mut r) = (i128::from(modulus), i128::from(value));
    let (mut old_t, mut t) = (0_i128, 1_i128);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_t, t) = (t, old_t - quotient * t);
    }
    (old_r == 1).then(|| old_t.rem_euclid(i128::from(modulus)) as u64)
}

fn integer_to_u64(value: &Integer) -> Option<u64> {
    value.to_string().parse().ok()
}

fn extract_invariant_factors(form: &Matrix<IntegerRing>) -> Option<Vec<Integer>> {
    let (rows, columns) = form.shape();
    let diagonal_length = rows.min(columns);
    let mut factors = Vec::new();
    let mut encountered_zero = false;
    for row in 0..rows {
        for column in 0..columns {
            let value = form.get(row, column).expect("in-bounds Smith entry");
            if row != column && !value.is_zero() {
                return None;
            }
        }
    }
    for index in 0..diagonal_length {
        let value = form.get(index, index).expect("in-bounds Smith diagonal");
        if value.is_zero() {
            encountered_zero = true;
            continue;
        }
        if encountered_zero || value.sign() < 0 {
            return None;
        }
        if let Some(previous) = factors.last() {
            let (_, remainder) = Z
                .div_rem(&value, previous)
                .expect("a recorded Smith invariant is nonzero");
            if !remainder.is_zero() {
                return None;
            }
        }
        factors.push(value);
    }
    Some(factors)
}

fn push_frame(output: &mut Vec<u8>, value: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(value.len())
        .map_err(|_| NoetherError::Serialization("Smith form component is too large".to_owned()))?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}
