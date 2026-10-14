// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Internal p-adic trace-formula acceleration for projective double covers.

use crate::linalg::Matrix;
use crate::numthy::{FiniteField, FiniteFieldElement, QadicElement, QadicRing};
use crate::{Integer, Rational, Ring};

use super::{
    CyclicCoverRamification, CyclicHypersurfaceCover, MonomialOrder, PointCountError, Polynomial,
    PolynomialRing, ResidueClass, ResidueRing,
};

const MAX_PADIC_PRECISION: u64 = 32;
const MAX_DENSE_SIMPLEX_COEFFICIENTS: usize = 10_000_000;
const MAX_TRACE_MATRIX_DIMENSION: usize = 4_096;
const MAX_TRACE_MATRIX_CELLS: usize = 20_000_000;
const CHECKSUM_MODULUS: u64 = 18_446_744_073_709_551_557;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PadicTraceCount {
    pub(super) degree: u64,
    pub(super) count: Integer,
    pub(super) precision: u64,
    pub(super) modulus: Integer,
    pub(super) count_residue: Integer,
    pub(super) trace_bound: Integer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PadicTraceTranscript {
    pub(super) counts: Vec<PadicTraceCount>,
    pub(super) stratum_count: usize,
    pub(super) largest_matrix_dimension: usize,
    pub(super) coefficient_checksum: u64,
}

pub(super) fn count_projective_double_cover(
    cover: &CyclicHypersurfaceCover<FiniteField>,
    degrees: &[u64],
) -> Result<PadicTraceTranscript, PointCountError> {
    let field = cover.base().base();
    if cover.base().dimension() != 2 || cover.cover_degree() != 2 {
        return Err(PointCountError::TraceFormulaUnavailable(
            "the current p-adic kernel admits double covers of P^2",
        ));
    }
    if cover.ramification() != CyclicCoverRamification::Tame {
        return Err(PointCountError::TraceFormulaUnavailable(
            "the p-adic double-cover trace formula requires odd characteristic",
        ));
    }
    let smoothness = cover.smoothness_evidence()?;
    if !smoothness.is_smooth() || smoothness.branch_is_smooth() != Some(true) {
        return Err(PointCountError::TraceFormulaUnavailable(
            "the Weil-bound reconstruction requires a proved smooth branch curve",
        ));
    }

    let prime =
        field
            .characteristic()
            .to_u64_checked()
            .ok_or(PointCountError::TraceFormulaUnavailable(
                "the p-adic trace kernel requires a word-sized prime",
            ))?;
    let field_order =
        field
            .order()
            .to_u64_checked()
            .ok_or(PointCountError::TraceFormulaUnavailable(
                "the p-adic trace kernel requires a word-sized finite base field",
            ))?;
    let branch_degree = cover
        .branch_equation()
        .homogeneous_degree()?
        .ok_or(PointCountError::TraceFormulaUnavailable(
            "the branch equation must be nonzero and homogeneous",
        ))?
        .to_u64_checked()
        .ok_or(PointCountError::TraceFormulaUnavailable(
            "the branch degree must fit a machine word",
        ))?;
    let second_betti = double_cover_second_betti(branch_degree)?;
    let precision_records = degrees
        .iter()
        .copied()
        .map(|degree| required_precision(prime, field_order, degree, second_betti))
        .collect::<Result<Vec<_>, _>>()?;
    let maximum_precision = precision_records
        .iter()
        .map(|record| record.precision)
        .max()
        .ok_or(PointCountError::InvalidExtensionDegrees)?;
    if maximum_precision > MAX_PADIC_PRECISION {
        return Err(PointCountError::ResourceLimit {
            resource: "p-adic trace precision",
            limit: MAX_PADIC_PRECISION,
        });
    }
    if field_order - 1 < 2 * maximum_precision {
        return Err(PointCountError::TraceFormulaUnavailable(
            "the current fast Eulerian coefficients require q-1 at least twice the precision",
        ));
    }

    if field.degree() != 1 {
        return count_unramified_projective_double_cover(
            cover,
            degrees,
            prime,
            field_order,
            branch_degree,
            precision_records,
            maximum_precision,
        );
    }

    let maximum_modulus = Integer::from(prime).pow(maximum_precision);
    let residues = ResidueRing::new(maximum_modulus.clone()).map_err(|error| {
        PointCountError::TraceFormulaFailure(format!("invalid p-adic residue ring: {error}"))
    })?;
    let coefficients = (1..=maximum_precision)
        .map(|precision| eulerian_coefficients(precision, &residues))
        .collect::<Result<Vec<_>, _>>()?;
    let mut total_residues = vec![residues.zero(); degrees.len()];
    let mut largest_matrix_dimension = 0_usize;
    let mut coefficient_checksum = FNV_OFFSET;
    let coordinate_count = cover.base().coordinates().len();
    let stratum_count = (1_usize << coordinate_count) - 1;

    for support_mask in 1_usize..(1_usize << coordinate_count) {
        let support = (0..coordinate_count)
            .filter(|index| support_mask & (1_usize << index) != 0)
            .collect::<Vec<_>>();
        let torus_dimension = support.len() - 1;
        if torus_dimension == 0 {
            let mut point = vec![field.zero(); coordinate_count];
            point[support[0]] = field.one();
            let value = cover.branch_equation().evaluate(&point)?;
            for (index, degree) in degrees.iter().copied().enumerate() {
                let roots = constant_double_cover_fiber(field, degree, &value)?;
                total_residues[index] =
                    residues.add(&total_residues[index], &residues.element(roots))?;
            }
            continue;
        }
        let polynomial = specialize_and_lift(cover.branch_equation(), &support, residues.clone())?;

        let traces = torus_character_sums(
            &polynomial,
            field_order,
            branch_degree,
            degrees,
            &precision_records,
            &coefficients,
            &mut largest_matrix_dimension,
            &mut coefficient_checksum,
        )?;
        for (index, (degree, character_sum)) in degrees.iter().copied().zip(traces).enumerate() {
            let torus_points = (&Integer::from(field_order).pow(degree) - &Integer::one())
                .pow(torus_dimension as u64);
            let fiber_count = residues.mul(
                &residues.element(torus_points),
                &residues.add(&residues.one(), &character_sum)?,
            )?;
            total_residues[index] = residues.add(&total_residues[index], &fiber_count)?;
        }
    }

    let counts = precision_records
        .into_iter()
        .zip(total_residues)
        .map(|(record, count_residue)| {
            recover_exact_surface_count(field_order, record, count_residue.representative())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PadicTraceTranscript {
        counts,
        stratum_count,
        largest_matrix_dimension,
        coefficient_checksum,
    })
}

#[derive(Debug, Clone)]
struct PrecisionRecord {
    degree: u64,
    precision: u64,
    modulus: Integer,
    trace_bound: Integer,
}

fn required_precision(
    prime: u64,
    field_order: u64,
    degree: u64,
    second_betti: u64,
) -> Result<PrecisionRecord, PointCountError> {
    let prime_integer = Integer::from(prime);
    let trace_bound = &Integer::from(second_betti) * &Integer::from(field_order).pow(degree);
    let required_modulus = &Integer::from(2) * &trace_bound;
    let mut precision = 1_u64;
    let mut modulus = prime_integer.clone();
    while modulus <= required_modulus {
        precision = precision
            .checked_add(1)
            .ok_or(PointCountError::ResourceLimit {
                resource: "p-adic trace precision",
                limit: u64::MAX,
            })?;
        modulus = &modulus * &prime_integer;
    }
    Ok(PrecisionRecord {
        degree,
        precision,
        modulus,
        trace_bound,
    })
}

fn double_cover_second_betti(branch_degree: u64) -> Result<u64, PointCountError> {
    let d = Integer::from(branch_degree);
    let genus = &(&(&d - &Integer::one()) * &(&d - &Integer::from(2))) / &Integer::from(2);
    let second_betti = &Integer::from(2) + &(&Integer::from(2) * &genus);
    second_betti
        .to_u64_checked()
        .ok_or(PointCountError::TraceFormulaUnavailable(
            "the surface Betti number must fit a machine word",
        ))
}

fn specialize_and_lift(
    branch: &Polynomial<FiniteField>,
    support: &[usize],
    residues: ResidueRing,
) -> Result<Polynomial<ResidueRing>, PointCountError> {
    let pivot = support[0];
    let variables = support
        .iter()
        .copied()
        .filter(|index| *index != pivot)
        .map(|index| branch.ring().variables()[index].clone())
        .collect::<Vec<_>>();
    let ring = PolynomialRing::new(
        residues.clone(),
        variables,
        MonomialOrder::GradedReverseLexicographic,
    )?;
    let coordinate_count = branch.ring().variables().len();
    ring.from_terms(branch.terms().filter_map(|(exponent, coefficient)| {
        let outside_support = (0..coordinate_count)
            .any(|index| !support.contains(&index) && !exponent[index].is_zero());
        (!outside_support).then(|| {
            let exponent = support
                .iter()
                .copied()
                .filter(|index| *index != pivot)
                .map(|index| exponent[index].clone())
                .collect::<Vec<_>>();
            (
                exponent,
                residues.element(coefficient.representative().clone()),
            )
        })
    }))
    .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn count_unramified_projective_double_cover(
    cover: &CyclicHypersurfaceCover<FiniteField>,
    degrees: &[u64],
    prime: u64,
    field_order: u64,
    branch_degree: u64,
    precision_records: Vec<PrecisionRecord>,
    maximum_precision: u64,
) -> Result<PadicTraceTranscript, PointCountError> {
    let field = cover.base().base();
    let coefficients = QadicRing::new(field.clone(), maximum_precision).map_err(|error| {
        PointCountError::TraceFormulaFailure(format!(
            "could not construct the unramified coefficient ring: {error}"
        ))
    })?;
    let eulerian = (1..=maximum_precision)
        .map(|precision| qadic_eulerian_coefficients(precision, &coefficients))
        .collect::<Result<Vec<_>, _>>()?;
    let mut total_residues = vec![coefficients.zero(); degrees.len()];
    let mut largest_matrix_dimension = 0_usize;
    let mut coefficient_checksum = FNV_OFFSET;
    let coordinate_count = cover.base().coordinates().len();
    let stratum_count = (1_usize << coordinate_count) - 1;

    for support_mask in 1_usize..(1_usize << coordinate_count) {
        let support = (0..coordinate_count)
            .filter(|index| support_mask & (1_usize << index) != 0)
            .collect::<Vec<_>>();
        let torus_dimension = support.len() - 1;
        if torus_dimension == 0 {
            let mut point = vec![field.zero(); coordinate_count];
            point[support[0]] = field.one();
            let value = cover.branch_equation().evaluate(&point)?;
            for (index, degree) in degrees.iter().copied().enumerate() {
                let roots = constant_double_cover_fiber(field, degree, &value)?;
                total_residues[index] =
                    coefficients.add(&total_residues[index], &coefficients.element(roots))?;
            }
            continue;
        }

        let polynomial = specialize_and_teichmuller_lift(
            cover.branch_equation(),
            &support,
            coefficients.clone(),
        )?;
        let traces = qadic_torus_character_sums(
            &polynomial,
            field_order,
            branch_degree,
            degrees,
            &precision_records,
            &eulerian,
            &mut largest_matrix_dimension,
            &mut coefficient_checksum,
        )?;
        for (index, (degree, character_sum)) in degrees.iter().copied().zip(traces).enumerate() {
            let torus_points = (&Integer::from(field_order).pow(degree) - &Integer::one())
                .pow(torus_dimension as u64);
            let fiber_count = coefficients.mul(
                &coefficients.element(torus_points),
                &coefficients.add(&coefficients.one(), &character_sum)?,
            )?;
            total_residues[index] = coefficients.add(&total_residues[index], &fiber_count)?;
        }
    }

    let counts = precision_records
        .into_iter()
        .zip(total_residues)
        .map(|(record, count_residue)| {
            let count_residue = qadic_base_residue(&count_residue)?;
            recover_exact_surface_count(field_order, record, &count_residue)
        })
        .collect::<Result<Vec<_>, _>>()?;
    debug_assert_eq!(coefficients.prime(), &Integer::from(prime));
    Ok(PadicTraceTranscript {
        counts,
        stratum_count,
        largest_matrix_dimension,
        coefficient_checksum,
    })
}

fn specialize_and_teichmuller_lift(
    branch: &Polynomial<FiniteField>,
    support: &[usize],
    coefficients: QadicRing,
) -> Result<Polynomial<QadicRing>, PointCountError> {
    let pivot = support[0];
    let variables = support
        .iter()
        .copied()
        .filter(|index| *index != pivot)
        .map(|index| branch.ring().variables()[index].clone())
        .collect::<Vec<_>>();
    let ring = PolynomialRing::new(
        coefficients.clone(),
        variables,
        MonomialOrder::GradedReverseLexicographic,
    )?;
    let coordinate_count = branch.ring().variables().len();
    let terms = branch
        .terms()
        .filter_map(|(exponent, coefficient)| {
            let outside_support = (0..coordinate_count)
                .any(|index| !support.contains(&index) && !exponent[index].is_zero());
            (!outside_support).then(|| {
                let exponent = support
                    .iter()
                    .copied()
                    .filter(|index| *index != pivot)
                    .map(|index| exponent[index].clone())
                    .collect::<Vec<_>>();
                coefficients
                    .teichmuller(coefficient)
                    .map(|coefficient| (exponent, coefficient))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    ring.from_terms(terms).map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn qadic_torus_character_sums(
    polynomial: &Polynomial<QadicRing>,
    field_order: u64,
    branch_degree: u64,
    degrees: &[u64],
    precision_records: &[PrecisionRecord],
    eulerian: &[Vec<QadicElement>],
    largest_matrix_dimension: &mut usize,
    coefficient_checksum: &mut u64,
) -> Result<Vec<QadicElement>, PointCountError> {
    let coefficients = polynomial.ring().coefficient_ring();
    let half = (field_order - 1) / 2;
    let mut power = polynomial.pow(half)?;
    let step = polynomial.pow(field_order - 1)?;
    let maximum_precision = precision_records
        .iter()
        .map(|record| record.precision)
        .max()
        .unwrap_or(0);
    let mut sums = vec![coefficients.zero(); degrees.len()];

    for coefficient_index in 0..maximum_precision {
        let odd_power = 2 * coefficient_index + 1;
        let polynomial_degree = branch_degree
            .checked_mul(odd_power)
            .and_then(|degree| degree.checked_mul(half))
            .ok_or(PointCountError::ResourceLimit {
                resource: "trace polynomial degree",
                limit: u64::MAX,
            })?;
        let operator_degree = polynomial_degree.div_ceil(field_order - 1);
        let matrix = qadic_trace_operator_matrix(
            &power,
            field_order,
            polynomial_degree,
            operator_degree,
            largest_matrix_dimension,
            coefficient_checksum,
        )?;
        let first_degree_index = precision_records
            .iter()
            .position(|record| record.precision > coefficient_index)
            .ok_or_else(|| {
                PointCountError::TraceFormulaFailure(
                    "an unused q-adic trace operator was constructed".to_owned(),
                )
            })?;
        let first_power = degrees[first_degree_index];
        let mut matrix_power = matrix_pow(&matrix, first_power)?;
        let mut current_power = first_power;
        for index in first_degree_index..degrees.len() {
            let requested_power = degrees[index];
            while current_power < requested_power {
                matrix_power = matrix_power.matmul(&matrix)?;
                current_power += 1;
            }
            let trace = matrix_power.trace()?;
            let coefficient = &eulerian[precision_records[index].precision as usize - 1]
                [coefficient_index as usize];
            sums[index] =
                coefficients.add(&sums[index], &coefficients.mul(coefficient, &trace)?)?;
        }
        if coefficient_index + 1 < maximum_precision {
            power = polynomial.ring().mul(&power, &step)?;
        }
    }
    Ok(sums)
}

fn qadic_trace_operator_matrix(
    power: &Polynomial<QadicRing>,
    field_order: u64,
    polynomial_degree: u64,
    operator_degree: u64,
    largest_matrix_dimension: &mut usize,
    coefficient_checksum: &mut u64,
) -> Result<Matrix<QadicRing>, PointCountError> {
    let polynomial_degree =
        usize::try_from(polynomial_degree).map_err(|_| PointCountError::ResourceLimit {
            resource: "dense simplex degree",
            limit: usize::MAX as u64,
        })?;
    let dense = QadicDenseSimplex::from_polynomial(power, polynomial_degree, coefficient_checksum)?;
    let operator_degree =
        usize::try_from(operator_degree).map_err(|_| PointCountError::ResourceLimit {
            resource: "trace operator degree",
            limit: usize::MAX as u64,
        })?;
    let basis = simplex_basis(power.ring().variables().len(), operator_degree)?;
    let dimension = basis.len();
    require_trace_matrix_size(dimension)?;
    *largest_matrix_dimension = (*largest_matrix_dimension).max(dimension);
    let rows = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| dense.operator_entry(&basis[row], &basis[column], field_order))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Matrix::from_rows(power.ring().coefficient_ring().clone(), rows).map_err(Into::into)
}

struct QadicDenseSimplex {
    arity: usize,
    maximum_degree: usize,
    coefficients: Vec<QadicElement>,
    zero: QadicElement,
}

impl QadicDenseSimplex {
    fn from_polynomial(
        polynomial: &Polynomial<QadicRing>,
        maximum_degree: usize,
        checksum: &mut u64,
    ) -> Result<Self, PointCountError> {
        let arity = polynomial.ring().variables().len();
        if arity > 2 {
            return Err(PointCountError::TraceFormulaUnavailable(
                "the current dense-simplex operator supports at most two torus variables",
            ));
        }
        let length = checked_simplex_size(arity, maximum_degree, "dense simplex coefficients")?;
        let coefficient_ring = polynomial.ring().coefficient_ring();
        let mut result = Self {
            arity,
            maximum_degree,
            coefficients: vec![coefficient_ring.zero(); length],
            zero: coefficient_ring.zero(),
        };
        for (exponent, coefficient) in polynomial.terms() {
            let exponent = exponent
                .iter()
                .map(|value| value.to_u64_checked())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    PointCountError::TraceFormulaFailure(
                        "a q-adic polynomial exponent did not fit a machine word".to_owned(),
                    )
                })?;
            let index = result.index(&exponent).ok_or_else(|| {
                PointCountError::TraceFormulaFailure(
                    "a q-adic polynomial term escaped its declared simplex".to_owned(),
                )
            })?;
            checksum_integer(checksum, index as u64);
            checksum_qadic(checksum, coefficient);
            result.coefficients[index] = coefficient.clone();
        }
        Ok(result)
    }

    fn index(&self, exponent: &[u64]) -> Option<usize> {
        simplex_index(self.arity, self.maximum_degree, exponent)
    }

    fn operator_entry(&self, output: &[u64], input: &[u64], field_order: u64) -> QadicElement {
        let mut exponent = [0_u64; 2];
        for variable in 0..self.arity {
            let Some(scaled) = output[variable].checked_mul(field_order) else {
                return self.zero.clone();
            };
            let Some(value) = scaled.checked_sub(input[variable]) else {
                return self.zero.clone();
            };
            exponent[variable] = value;
        }
        self.index(&exponent[..self.arity])
            .and_then(|index| self.coefficients.get(index))
            .cloned()
            .unwrap_or_else(|| self.zero.clone())
    }
}

fn qadic_eulerian_coefficients(
    precision: u64,
    coefficients: &QadicRing,
) -> Result<Vec<QadicElement>, PointCountError> {
    let twice_precision_factorial = factorial(2 * precision);
    let denominator = &(Integer::one() << (2 * precision - 1) as u32)
        * &(&factorial(precision) * &factorial(precision - 1));
    let normalization = Rational::new(twice_precision_factorial, denominator).map_err(|error| {
        PointCountError::TraceFormulaFailure(format!(
            "Eulerian normalization division failed: {error}"
        ))
    })?;
    (1..=precision)
        .map(|index| {
            let mut numerator = &normalization.numerator() * &binomial(precision - 1, index - 1);
            if index % 2 == 0 {
                numerator = -numerator;
            }
            let denominator = &normalization.denominator() * &Integer::from(2 * index - 1);
            let value = Rational::new(numerator, denominator).map_err(|error| {
                PointCountError::TraceFormulaFailure(format!(
                    "Eulerian coefficient division failed: {error}"
                ))
            })?;
            qadic_rational(&value, coefficients)
        })
        .collect()
}

fn qadic_rational(
    value: &Rational,
    coefficients: &QadicRing,
) -> Result<QadicElement, PointCountError> {
    let numerator = coefficients.element(value.numerator());
    let denominator = coefficients.element(value.denominator());
    let inverse = denominator.inverse().map_err(|_| {
        PointCountError::TraceFormulaFailure(
            "an Eulerian coefficient denominator was not p-adically invertible".to_owned(),
        )
    })?;
    coefficients.mul(&numerator, &inverse).map_err(Into::into)
}

fn qadic_base_residue(value: &QadicElement) -> Result<Integer, PointCountError> {
    if value.frobenius(1) != *value {
        return Err(PointCountError::TraceFormulaFailure(
            "a q-adic point-count residue was not fixed by coefficient Frobenius".to_owned(),
        ));
    }
    let coordinates = value.coefficients();
    let Some(constant) = coordinates.first().and_then(|entry| entry.representative()) else {
        return Err(PointCountError::TraceFormulaFailure(
            "a q-adic point-count residue had no integral constant coordinate".to_owned(),
        ));
    };
    if coordinates
        .iter()
        .skip(1)
        .any(|entry| entry.representative().is_none_or(|entry| !entry.is_zero()))
    {
        return Err(PointCountError::TraceFormulaFailure(
            "a Frobenius-fixed q-adic residue did not lie in the coefficient p-adic ring"
                .to_owned(),
        ));
    }
    Ok(constant)
}

fn checksum_qadic(checksum: &mut u64, value: &QadicElement) {
    for coordinate in value.coefficients() {
        let representative = coordinate
            .representative()
            .expect("a Witt-ring coordinate is an integral p-adic value");
        checksum_integer(checksum, representative.modulo_u64(CHECKSUM_MODULUS));
    }
}

fn constant_double_cover_fiber(
    field: &FiniteField,
    extension_degree: u64,
    value: &FiniteFieldElement,
) -> Result<u64, PointCountError> {
    if value.is_zero() {
        return Ok(1);
    }
    let field_order =
        field
            .order()
            .to_u64_checked()
            .ok_or(PointCountError::TraceFormulaUnavailable(
                "the constant-fiber test requires a word-sized finite base field",
            ))?;
    let is_square_over_base = value.pow((field_order - 1) / 2) == field.one();
    Ok(
        if is_square_over_base || extension_degree.is_multiple_of(2) {
            2
        } else {
            0
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn torus_character_sums(
    polynomial: &Polynomial<ResidueRing>,
    field_order: u64,
    branch_degree: u64,
    degrees: &[u64],
    precision_records: &[PrecisionRecord],
    coefficients: &[Vec<ResidueClass>],
    largest_matrix_dimension: &mut usize,
    coefficient_checksum: &mut u64,
) -> Result<Vec<ResidueClass>, PointCountError> {
    let residues = polynomial.ring().coefficient_ring();
    let half = (field_order - 1) / 2;
    let mut power = polynomial.pow(half)?;
    let step = polynomial.pow(field_order - 1)?;
    let mut sums = vec![residues.zero(); degrees.len()];

    for coefficient_index in 0..precision_records
        .iter()
        .map(|record| record.precision)
        .max()
        .unwrap_or(0)
    {
        let odd_power = 2 * coefficient_index + 1;
        let polynomial_degree = branch_degree
            .checked_mul(odd_power)
            .and_then(|degree| degree.checked_mul(half))
            .ok_or(PointCountError::ResourceLimit {
                resource: "trace polynomial degree",
                limit: u64::MAX,
            })?;
        let operator_degree = polynomial_degree.div_ceil(field_order - 1);
        let matrix = trace_operator_matrix(
            &power,
            field_order,
            polynomial_degree,
            operator_degree,
            largest_matrix_dimension,
            coefficient_checksum,
        )?;
        let first_degree_index = precision_records
            .iter()
            .position(|record| record.precision > coefficient_index)
            .ok_or_else(|| {
                PointCountError::TraceFormulaFailure(
                    "an unused trace operator was constructed".to_owned(),
                )
            })?;
        let first_power = degrees[first_degree_index];
        let mut matrix_power = matrix_pow(&matrix, first_power)?;
        let mut current_power = first_power;
        for index in first_degree_index..degrees.len() {
            let requested_power = degrees[index];
            while current_power < requested_power {
                matrix_power = matrix_power.matmul(&matrix)?;
                current_power += 1;
            }
            let trace = matrix_power.trace()?;
            let coefficient = &coefficients[precision_records[index].precision as usize - 1]
                [coefficient_index as usize];
            sums[index] = residues.add(&sums[index], &residues.mul(coefficient, &trace)?)?;
        }
        if coefficient_index + 1
            < precision_records
                .iter()
                .map(|record| record.precision)
                .max()
                .unwrap_or(0)
        {
            power = polynomial.ring().mul(&power, &step)?;
        }
    }
    Ok(sums)
}

fn trace_operator_matrix(
    power: &Polynomial<ResidueRing>,
    field_order: u64,
    polynomial_degree: u64,
    operator_degree: u64,
    largest_matrix_dimension: &mut usize,
    coefficient_checksum: &mut u64,
) -> Result<Matrix<ResidueRing>, PointCountError> {
    let polynomial_degree =
        usize::try_from(polynomial_degree).map_err(|_| PointCountError::ResourceLimit {
            resource: "dense simplex degree",
            limit: usize::MAX as u64,
        })?;
    let dense = DenseSimplex::from_polynomial(power, polynomial_degree, coefficient_checksum)?;
    let operator_degree =
        usize::try_from(operator_degree).map_err(|_| PointCountError::ResourceLimit {
            resource: "trace operator degree",
            limit: usize::MAX as u64,
        })?;
    let basis = simplex_basis(power.ring().variables().len(), operator_degree)?;
    let dimension = basis.len();
    require_trace_matrix_size(dimension)?;
    let cells = dimension * dimension;
    *largest_matrix_dimension = (*largest_matrix_dimension).max(dimension);
    Matrix::from_integer_entry_refs(
        power.ring().coefficient_ring().clone(),
        dimension,
        dimension,
        (0..cells).map(|index| {
            let row = index / dimension;
            let column = index % dimension;
            dense.operator_entry(&basis[row], &basis[column], field_order)
        }),
    )
    .map_err(Into::into)
}

struct DenseSimplex {
    arity: usize,
    maximum_degree: usize,
    coefficients: Vec<Integer>,
    zero: Integer,
}

impl DenseSimplex {
    fn from_polynomial(
        polynomial: &Polynomial<ResidueRing>,
        maximum_degree: usize,
        checksum: &mut u64,
    ) -> Result<Self, PointCountError> {
        let arity = polynomial.ring().variables().len();
        if arity > 2 {
            return Err(PointCountError::TraceFormulaUnavailable(
                "the current dense-simplex operator supports at most two torus variables",
            ));
        }
        let length = checked_simplex_size(arity, maximum_degree, "dense simplex coefficients")?;
        let mut result = Self {
            arity,
            maximum_degree,
            coefficients: vec![Integer::zero(); length],
            zero: Integer::zero(),
        };
        let mut valid = true;
        let native = polynomial.visit_native_residue_terms(|exponent, coefficient| {
            let Some(index) = result.index(exponent) else {
                valid = false;
                return;
            };
            checksum_integer(checksum, index as u64);
            checksum_integer(checksum, coefficient.modulo_u64(CHECKSUM_MODULUS));
            result.coefficients[index] = coefficient;
        });
        if !native {
            return Err(PointCountError::TraceFormulaFailure(
                "the p-adic power lost native residue-polynomial storage".to_owned(),
            ));
        }
        if !valid {
            return Err(PointCountError::TraceFormulaFailure(
                "a p-adic polynomial term escaped its declared simplex".to_owned(),
            ));
        }
        Ok(result)
    }

    fn index(&self, exponent: &[u64]) -> Option<usize> {
        simplex_index(self.arity, self.maximum_degree, exponent)
    }

    fn operator_entry(&self, output: &[u64], input: &[u64], field_order: u64) -> &Integer {
        let mut exponent = [0_u64; 2];
        for variable in 0..self.arity {
            let Some(scaled) = output[variable].checked_mul(field_order) else {
                return &self.zero;
            };
            let Some(value) = scaled.checked_sub(input[variable]) else {
                return &self.zero;
            };
            exponent[variable] = value;
        }
        self.index(&exponent[..self.arity])
            .and_then(|index| self.coefficients.get(index))
            .unwrap_or(&self.zero)
    }
}

fn simplex_basis(arity: usize, maximum_degree: usize) -> Result<Vec<Vec<u64>>, PointCountError> {
    let length = simplex_size(arity, maximum_degree).ok_or(PointCountError::ResourceLimit {
        resource: "trace matrix dimension",
        limit: MAX_TRACE_MATRIX_DIMENSION as u64,
    })?;
    let mut basis = Vec::with_capacity(length);
    match arity {
        1 => {
            for exponent in 0..=maximum_degree {
                basis.push(vec![exponent as u64]);
            }
        }
        2 => {
            for left in 0..=maximum_degree {
                for right in 0..=maximum_degree - left {
                    basis.push(vec![left as u64, right as u64]);
                }
            }
        }
        _ => {
            return Err(PointCountError::TraceFormulaUnavailable(
                "trace matrices require one or two torus variables",
            ));
        }
    }
    Ok(basis)
}

fn simplex_size(arity: usize, maximum_degree: usize) -> Option<usize> {
    match arity {
        0 => Some(1),
        1 => maximum_degree.checked_add(1),
        2 => maximum_degree
            .checked_add(1)?
            .checked_mul(maximum_degree.checked_add(2)?)?
            .checked_div(2),
        _ => None,
    }
}

fn checked_simplex_size(
    arity: usize,
    maximum_degree: usize,
    resource: &'static str,
) -> Result<usize, PointCountError> {
    let length = simplex_size(arity, maximum_degree).ok_or(PointCountError::ResourceLimit {
        resource,
        limit: MAX_DENSE_SIMPLEX_COEFFICIENTS as u64,
    })?;
    if length > MAX_DENSE_SIMPLEX_COEFFICIENTS {
        return Err(PointCountError::ResourceLimit {
            resource,
            limit: MAX_DENSE_SIMPLEX_COEFFICIENTS as u64,
        });
    }
    Ok(length)
}

fn simplex_index(arity: usize, maximum_degree: usize, exponent: &[u64]) -> Option<usize> {
    if exponent.len() != arity {
        return None;
    }
    match arity {
        0 => Some(0),
        1 => {
            let value = usize::try_from(exponent[0]).ok()?;
            (value <= maximum_degree).then_some(value)
        }
        2 => {
            let left = usize::try_from(exponent[0]).ok()?;
            let right = usize::try_from(exponent[1]).ok()?;
            if left.checked_add(right)? > maximum_degree {
                return None;
            }
            let offset = left
                .checked_mul(maximum_degree + 1)?
                .checked_sub(left.checked_mul(left.saturating_sub(1))? / 2)?;
            offset.checked_add(right)
        }
        _ => None,
    }
}

fn require_trace_matrix_size(dimension: usize) -> Result<(), PointCountError> {
    if dimension > MAX_TRACE_MATRIX_DIMENSION {
        return Err(PointCountError::ResourceLimit {
            resource: "trace matrix dimension",
            limit: MAX_TRACE_MATRIX_DIMENSION as u64,
        });
    }
    let cells = dimension
        .checked_mul(dimension)
        .ok_or(PointCountError::ResourceLimit {
            resource: "trace matrix cells",
            limit: MAX_TRACE_MATRIX_CELLS as u64,
        })?;
    if cells > MAX_TRACE_MATRIX_CELLS {
        return Err(PointCountError::ResourceLimit {
            resource: "trace matrix cells",
            limit: MAX_TRACE_MATRIX_CELLS as u64,
        });
    }
    Ok(())
}

fn matrix_pow<R: Ring>(matrix: &Matrix<R>, exponent: u64) -> Result<Matrix<R>, PointCountError> {
    if exponent == 0 {
        return Ok(Matrix::identity(matrix.ring().clone(), matrix.shape().0));
    }
    let mut result = Matrix::identity(matrix.ring().clone(), matrix.shape().0);
    let mut base = matrix.clone();
    let mut power = exponent;
    while power != 0 {
        if power & 1 == 1 {
            result = result.matmul(&base)?;
        }
        power >>= 1;
        if power != 0 {
            base = base.matmul(&base)?;
        }
    }
    Ok(result)
}

fn eulerian_coefficients(
    precision: u64,
    residues: &ResidueRing,
) -> Result<Vec<ResidueClass>, PointCountError> {
    let twice_precision_factorial = factorial(2 * precision);
    let denominator = &(Integer::one() << (2 * precision - 1) as u32)
        * &(&factorial(precision) * &factorial(precision - 1));
    let normalization = Rational::new(twice_precision_factorial, denominator).map_err(|error| {
        PointCountError::TraceFormulaFailure(format!(
            "Eulerian normalization division failed: {error}"
        ))
    })?;
    (1..=precision)
        .map(|index| {
            let mut numerator = &normalization.numerator() * &binomial(precision - 1, index - 1);
            if index % 2 == 0 {
                numerator = -numerator;
            }
            let denominator = &normalization.denominator() * &Integer::from(2 * index - 1);
            rational_residue(
                &Rational::new(numerator, denominator).map_err(|error| {
                    PointCountError::TraceFormulaFailure(format!(
                        "Eulerian coefficient division failed: {error}"
                    ))
                })?,
                residues,
            )
        })
        .collect()
}

fn rational_residue(
    value: &Rational,
    residues: &ResidueRing,
) -> Result<ResidueClass, PointCountError> {
    let numerator = residues.element(value.numerator());
    let denominator = residues.element(value.denominator());
    let inverse = denominator.inverse().map_err(|_| {
        PointCountError::TraceFormulaFailure(
            "an Eulerian coefficient denominator was not p-adically invertible".to_owned(),
        )
    })?;
    residues.mul(&numerator, &inverse).map_err(Into::into)
}

fn recover_exact_surface_count(
    field_order: u64,
    record: PrecisionRecord,
    count_residue_at_maximum_precision: &Integer,
) -> Result<PadicTraceCount, PointCountError> {
    let count_residue = count_residue_at_maximum_precision % &record.modulus;
    let baseline = &Integer::one() + &Integer::from(field_order).pow(2 * record.degree);
    let mut trace = &(&count_residue - &baseline) % &record.modulus;
    if trace.sign() < 0 {
        trace = &trace + &record.modulus;
    }
    if trace > (&record.modulus / &Integer::from(2)) {
        trace = &trace - &record.modulus;
    }
    if trace.abs() > record.trace_bound {
        return Err(PointCountError::TraceFormulaFailure(
            "the trace-formula residue has no lift inside the certified Weil bound".to_owned(),
        ));
    }
    let count = &baseline + &trace;
    Ok(PadicTraceCount {
        degree: record.degree,
        count,
        precision: record.precision,
        modulus: record.modulus,
        count_residue,
        trace_bound: record.trace_bound,
    })
}

fn factorial(value: u64) -> Integer {
    (2..=value).fold(Integer::one(), |product, factor| {
        &product * &Integer::from(factor)
    })
}

fn binomial(n: u64, k: u64) -> Integer {
    let k = k.min(n - k);
    let mut result = Integer::one();
    for index in 0..k {
        result = &(&result * &Integer::from(n - index)) / &Integer::from(index + 1);
    }
    result
}

fn checksum_integer(checksum: &mut u64, value: u64) {
    for byte in value.to_be_bytes() {
        *checksum ^= u64::from(byte);
        *checksum = checksum.wrapping_mul(FNV_PRIME);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::ProjectiveSpace;

    #[test]
    fn dense_simplex_index_matches_basis_order() {
        let simplex = DenseSimplex {
            arity: 2,
            maximum_degree: 4,
            coefficients: vec![Integer::zero(); 15],
            zero: Integer::zero(),
        };
        for (index, exponent) in simplex_basis(2, 4).unwrap().iter().enumerate() {
            assert_eq!(simplex.index(exponent), Some(index));
        }
    }

    #[test]
    fn accelerated_quadric_double_cover_matches_exhaustive_counts() {
        let field = FiniteField::prime(11_u8).unwrap();
        let plane = ProjectiveSpace::new(field, 2).unwrap();
        let ring = plane.coordinate_ring();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let z = ring.generator(2).unwrap();
        let branch = (&(&x * &z).unwrap() - &y.pow(2).unwrap()).unwrap();
        let cover = CyclicHypersurfaceCover::<FiniteField>::new(plane, 2, branch).unwrap();
        let accelerated = count_projective_double_cover(&cover, &[1, 2]).unwrap();
        let exhaustive = [1_u64, 2]
            .into_iter()
            .map(|degree| super::super::point_counts::count_cyclic_extension(&cover, degree))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            accelerated
                .counts
                .iter()
                .map(|record| record.count.clone())
                .collect::<Vec<_>>(),
            exhaustive
                .iter()
                .map(|record| record.count().clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn accelerated_k3_double_cover_matches_exhaustive_small_extensions() {
        let field = FiniteField::prime(11_u8).unwrap();
        let plane = ProjectiveSpace::new(field, 2).unwrap();
        let ring = plane.coordinate_ring();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let z = ring.generator(2).unwrap();
        let branch = ring
            .add(
                &ring.add(&x.pow(6).unwrap(), &y.pow(6).unwrap()).unwrap(),
                &z.pow(6).unwrap(),
            )
            .unwrap();
        let cover = CyclicHypersurfaceCover::<FiniteField>::new(plane, 2, branch).unwrap();
        let accelerated = count_projective_double_cover(&cover, &[1, 2]).unwrap();
        let exhaustive = [1_u64, 2]
            .into_iter()
            .map(|degree| super::super::point_counts::count_cyclic_extension(&cover, degree))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            accelerated
                .counts
                .iter()
                .map(|record| record.count.clone())
                .collect::<Vec<_>>(),
            exhaustive
                .iter()
                .map(|record| record.count().clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn unramified_trace_formula_matches_exhaustive_count_over_f9() {
        let field = FiniteField::extension(3_u64, 2).unwrap();
        let plane = ProjectiveSpace::new(field.clone(), 2).unwrap();
        let ring = plane.coordinate_ring();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let z = ring.generator(2).unwrap();
        let scaled_y_square = y.pow(2).unwrap().scalar_mul(&field.generator()).unwrap();
        let branch = (&(&x * &z).unwrap() - &scaled_y_square).unwrap();
        let cover = CyclicHypersurfaceCover::<FiniteField>::new(plane, 2, branch).unwrap();

        let accelerated = count_projective_double_cover(&cover, &[1]).unwrap();
        let exhaustive = super::super::point_counts::count_cyclic_extension(&cover, 1).unwrap();
        assert_eq!(accelerated.counts[0].count, *exhaustive.count());
        assert_eq!(accelerated.counts[0].precision, 4);
    }

    #[test]
    fn public_unramified_dispatch_verifies_affordable_degrees_and_quadric_formula() {
        let field = FiniteField::extension(5_u64, 2).unwrap();
        let plane = ProjectiveSpace::new(field.clone(), 2).unwrap();
        let ring = plane.coordinate_ring();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let z = ring.generator(2).unwrap();
        let scaled_y_square = y.pow(2).unwrap().scalar_mul(&field.generator()).unwrap();
        let branch = (&(&x * &z).unwrap() - &scaled_y_square).unwrap();
        let cover = CyclicHypersurfaceCover::<FiniteField>::new(plane, 2, branch).unwrap();

        let counts = cover.point_counts([1, 2, 3]).unwrap();
        assert!(counts.counts()[0].evidence().independently_verified());
        assert!(counts.counts()[1].evidence().independently_verified());
        assert!(!counts.counts()[2].evidence().independently_verified());

        let q = Integer::from(25);
        let first_trace = counts.count(1).unwrap() - &(&Integer::one() + &q.pow(2));
        let expected_third = &(&Integer::one() + &q.pow(6)) + &(&first_trace * &q.pow(2));
        assert_eq!(counts.count(3), Some(&expected_third));
    }
}
