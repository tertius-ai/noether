// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact rational univariate representations over generic fields.

use poly_core::{Monomial, MonomialOrder, SparsePolynomial};

use crate::fglm::{
    apply_operator, finite_quotient_algebra, zero_vector, LinearBasis, LinearDependence,
    MultiplicationOperators, SparseVector,
};
use crate::{Field, GroebnerError, GroebnerOptions};

/// Work counters from a rational univariate representation search.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RurMetrics {
    /// Dimension of the represented quotient algebra.
    pub quotient_dimension: usize,
    /// Sparse nonzeros in the reused variable multiplication operators.
    pub multiplication_operator_nonzeros: usize,
    /// Deterministic linear forms tested before finding a primitive element.
    pub linear_forms_tried: usize,
    /// Krylov vectors classified across all tested forms.
    pub krylov_vectors: usize,
    /// Source polynomial terms checked by exact univariate substitution.
    pub substitution_terms: usize,
}

/// Exactly verified rational univariate representation data.
///
/// If `f` is [`Self::equation`] and `q_i` is the `i`th coordinate, this
/// represents `x_i = q_i(t)` in `K[t]/(f)`. The selected parameter is the
/// declared linear form in the original coordinates. All denominators are one
/// because a primitive quotient-algebra element was found directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RurOutput<C> {
    /// Monic univariate equation for the selected parameter.
    pub equation: SparsePolynomial<C>,
    /// Coordinate polynomials modulo `equation`, in source-variable order.
    pub coordinates: Vec<SparsePolynomial<C>>,
    /// Coefficients of the selected linear form in source-variable order.
    pub linear_form: Vec<C>,
    /// Whether the quotient is reduced, equivalently whether `equation` is
    /// square-free for this monogenic representation.
    pub is_reduced: bool,
    /// Operation counters.
    pub metrics: RurMetrics,
}

/// Construct and exactly verify a rational univariate representation.
///
/// The source must be a reduced Gröbner basis of a zero-dimensional ideal.
/// Deterministic coordinate and small-coefficient linear forms are tested
/// until one generates the quotient algebra. Some non-monogenic algebras admit
/// no such representation; exhausting `max_rur_forms` returns a typed resource
/// failure rather than weakening the result.
pub fn rational_univariate_representation<F: Field>(
    field: &F,
    source_basis: &[SparsePolynomial<F::Element>],
    options: &GroebnerOptions,
) -> Result<RurOutput<F::Element>, GroebnerError> {
    let quotient = finite_quotient_algebra(field, source_basis, options)?;
    let arity = quotient.arity;
    if quotient.is_unit {
        return Ok(RurOutput {
            equation: univariate_polynomial(field, [field.one()]),
            coordinates: (0..arity)
                .map(|_| univariate_polynomial(field, core::iter::empty()))
                .collect(),
            linear_form: zero_vector(field, arity),
            is_reduced: true,
            metrics: RurMetrics::default(),
        });
    }

    let dimension = quotient.dimension;
    let one = Monomial::one(arity);
    let one_index = quotient
        .standard
        .iter()
        .position(|monomial| monomial == &one)
        .ok_or(GroebnerError::VerificationFailed)?;
    let mut one_vector = zero_vector(field, dimension);
    one_vector[one_index] = field.one();
    let forms = candidate_linear_forms(field, arity, options.max_rur_forms)?;
    let mut metrics = RurMetrics {
        quotient_dimension: dimension,
        multiplication_operator_nonzeros: quotient.operator_nonzeros,
        ..RurMetrics::default()
    };

    for linear_form in forms {
        metrics.linear_forms_tried = metrics.linear_forms_tried.saturating_add(1);
        let mut basis = LinearBasis::new(dimension);
        let mut power = one_vector.clone();
        let mut dependence = None;
        for degree in 0..=dimension {
            metrics.krylov_vectors = metrics.krylov_vectors.saturating_add(1);
            match basis.classify(field, &power, degree)? {
                LinearDependence::Independent => {
                    if degree == dimension {
                        return Err(GroebnerError::VerificationFailed);
                    }
                    power = apply_linear_form(
                        field,
                        &quotient.operators,
                        &linear_form,
                        &power,
                        dimension,
                    );
                }
                LinearDependence::Dependent(coefficients) => {
                    if degree == dimension {
                        dependence = Some(coefficients);
                    }
                    break;
                }
            }
        }
        let Some(dependence) = dependence else {
            continue;
        };

        let coordinates =
            coordinate_polynomials(field, &quotient.operators, one_index, dimension, &mut basis)?;
        let equation_coefficients = dependence
            .iter()
            .map(|coefficient| field.neg(coefficient))
            .chain(core::iter::once(field.one()))
            .collect::<Vec<_>>();
        verify_rur(
            field,
            source_basis,
            &linear_form,
            &dependence,
            &coordinates,
            &mut metrics,
        )?;
        let is_reduced = square_free(field, &equation_coefficients);
        return Ok(RurOutput {
            equation: univariate_polynomial(field, equation_coefficients),
            coordinates: coordinates
                .into_iter()
                .map(|coefficients| univariate_polynomial(field, coefficients))
                .collect(),
            linear_form,
            is_reduced,
            metrics,
        });
    }
    Err(GroebnerError::ResourceLimit {
        resource: "RUR primitive linear forms",
        limit: options.max_rur_forms,
    })
}

fn candidate_linear_forms<F: Field>(
    field: &F,
    arity: usize,
    limit: usize,
) -> Result<Vec<Vec<F::Element>>, GroebnerError> {
    let mut forms = Vec::new();
    for variable in 0..arity {
        let mut form = zero_vector(field, arity);
        form[variable] = field.one();
        push_distinct(&mut forms, form, limit);
    }
    let mut magnitude = field.one();
    for _ in 0..limit.saturating_mul(2) {
        if forms.len() == limit {
            break;
        }
        for scalar in [magnitude.clone(), field.neg(&magnitude)] {
            let mut coefficient = field.one();
            let mut form = Vec::with_capacity(arity);
            for _ in 0..arity {
                form.push(coefficient.clone());
                coefficient = field.mul(&coefficient, &scalar);
            }
            push_distinct(&mut forms, form, limit);
            if forms.len() == limit {
                break;
            }
        }
        magnitude = field.add(&magnitude, &field.one());
        if arity == 0 || forms.len() == limit {
            break;
        }
    }
    if forms.is_empty() {
        return Err(GroebnerError::ResourceLimit {
            resource: "RUR primitive linear forms",
            limit,
        });
    }
    Ok(forms)
}

fn push_distinct<C: Clone + Eq>(forms: &mut Vec<Vec<C>>, form: Vec<C>, limit: usize) {
    if forms.len() < limit && !forms.contains(&form) {
        forms.push(form);
    }
}

fn apply_linear_form<F: Field>(
    field: &F,
    operators: &MultiplicationOperators<F::Element>,
    coefficients: &[F::Element],
    vector: &[F::Element],
    dimension: usize,
) -> Vec<F::Element> {
    let mut output = zero_vector(field, dimension);
    for (operator, coefficient) in operators.iter().zip(coefficients) {
        if field.is_zero(coefficient) {
            continue;
        }
        let image = apply_operator(field, operator, vector, dimension);
        for (target, value) in output.iter_mut().zip(image) {
            let value = field.mul(coefficient, &value);
            *target = field.add(target, &value);
        }
    }
    output
}

fn coordinate_polynomials<F: Field>(
    field: &F,
    operators: &MultiplicationOperators<F::Element>,
    one_index: usize,
    dimension: usize,
    basis: &mut LinearBasis<F::Element>,
) -> Result<Vec<Vec<F::Element>>, GroebnerError> {
    operators
        .iter()
        .map(|operator| {
            let vector = dense_column(field, &operator[one_index], dimension);
            match basis.classify(field, &vector, dimension)? {
                LinearDependence::Dependent(coefficients) => Ok(coefficients),
                LinearDependence::Independent => Err(GroebnerError::VerificationFailed),
            }
        })
        .collect()
}

fn dense_column<F: Field>(
    field: &F,
    column: &SparseVector<F::Element>,
    dimension: usize,
) -> Vec<F::Element> {
    let mut dense = zero_vector(field, dimension);
    for (row, coefficient) in column {
        dense[*row] = coefficient.clone();
    }
    dense
}

#[allow(clippy::too_many_arguments)]
fn verify_rur<F: Field>(
    field: &F,
    source_basis: &[SparsePolynomial<F::Element>],
    linear_form: &[F::Element],
    parameter_relation: &[F::Element],
    coordinates: &[Vec<F::Element>],
    metrics: &mut RurMetrics,
) -> Result<(), GroebnerError> {
    let dimension = parameter_relation.len();
    let reconstructed_parameter = coordinates.iter().zip(linear_form).fold(
        zero_vector(field, dimension),
        |mut sum, (coordinate, coefficient)| {
            for (target, value) in sum.iter_mut().zip(coordinate) {
                *target = field.add(target, &field.mul(coefficient, value));
            }
            sum
        },
    );
    let mut parameter = zero_vector(field, dimension);
    if dimension == 1 {
        parameter[0] = parameter_relation[0].clone();
    } else {
        parameter[1] = field.one();
    }
    if parameter != reconstructed_parameter {
        return Err(GroebnerError::VerificationFailed);
    }

    for polynomial in source_basis {
        let mut value = zero_vector(field, dimension);
        for (monomial, coefficient) in polynomial.terms() {
            metrics.substitution_terms = metrics.substitution_terms.saturating_add(1);
            let mut term = zero_vector(field, dimension);
            term[0] = coefficient.clone();
            for (variable, coordinate) in coordinates.iter().enumerate() {
                let exponent = monomial.exponent(variable);
                if exponent != 0 {
                    let power = power_mod(field, coordinate, exponent, parameter_relation);
                    term = multiply_mod(field, &term, &power, parameter_relation);
                }
            }
            for (target, coefficient) in value.iter_mut().zip(term) {
                *target = field.add(target, &coefficient);
            }
        }
        if value.iter().any(|coefficient| !field.is_zero(coefficient)) {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn power_mod<F: Field>(
    field: &F,
    value: &[F::Element],
    mut exponent: u32,
    relation: &[F::Element],
) -> Vec<F::Element> {
    let mut result = zero_vector(field, relation.len());
    result[0] = field.one();
    let mut power = value.to_vec();
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = multiply_mod(field, &result, &power, relation);
        }
        exponent >>= 1;
        if exponent != 0 {
            power = multiply_mod(field, &power, &power, relation);
        }
    }
    result
}

fn multiply_mod<F: Field>(
    field: &F,
    left: &[F::Element],
    right: &[F::Element],
    relation: &[F::Element],
) -> Vec<F::Element> {
    let dimension = relation.len();
    let mut product = zero_vector(field, dimension.saturating_mul(2).saturating_sub(1));
    for (left_degree, left_coefficient) in left.iter().enumerate() {
        if field.is_zero(left_coefficient) {
            continue;
        }
        for (right_degree, right_coefficient) in right.iter().enumerate() {
            if field.is_zero(right_coefficient) {
                continue;
            }
            let term = field.mul(left_coefficient, right_coefficient);
            let degree = left_degree + right_degree;
            product[degree] = field.add(&product[degree], &term);
        }
    }
    for degree in (dimension..product.len()).rev() {
        let coefficient = product[degree].clone();
        if field.is_zero(&coefficient) {
            continue;
        }
        for (offset, relation_coefficient) in relation.iter().enumerate() {
            let term = field.mul(&coefficient, relation_coefficient);
            let target = degree - dimension + offset;
            product[target] = field.add(&product[target], &term);
        }
    }
    product.truncate(dimension);
    product
}

fn univariate_polynomial<F: Field>(
    field: &F,
    coefficients: impl IntoIterator<Item = F::Element>,
) -> SparsePolynomial<F::Element> {
    SparsePolynomial::new(
        1,
        MonomialOrder::Lexicographic,
        coefficients
            .into_iter()
            .enumerate()
            .filter_map(|(degree, coefficient)| {
                (!field.is_zero(&coefficient)).then(|| {
                    (
                        Monomial::new(
                            vec![u32::try_from(degree).expect("bounded RUR degree")]
                                .into_boxed_slice(),
                        ),
                        coefficient,
                    )
                })
            }),
        field,
    )
}

fn square_free<F: Field>(field: &F, polynomial: &[F::Element]) -> bool {
    let derivative = polynomial
        .iter()
        .enumerate()
        .skip(1)
        .map(|(degree, coefficient)| field.mul(coefficient, &usize_scalar(field, degree)))
        .collect::<Vec<_>>();
    polynomial_gcd(field, polynomial.to_vec(), derivative).len() <= 1
}

fn usize_scalar<F: Field>(field: &F, mut value: usize) -> F::Element {
    let mut result = field.zero();
    let mut power = field.one();
    while value != 0 {
        if value & 1 == 1 {
            result = field.add(&result, &power);
        }
        value >>= 1;
        if value != 0 {
            power = field.add(&power, &power);
        }
    }
    result
}

fn polynomial_gcd<F: Field>(
    field: &F,
    mut left: Vec<F::Element>,
    mut right: Vec<F::Element>,
) -> Vec<F::Element> {
    trim(field, &mut left);
    trim(field, &mut right);
    while !right.is_empty() {
        let remainder = polynomial_remainder(field, left, &right);
        left = right;
        right = remainder;
    }
    if let Some(leading) = left.last().cloned() {
        let inverse = field.inverse(&leading);
        for coefficient in &mut left {
            *coefficient = field.mul(coefficient, &inverse);
        }
    }
    left
}

fn polynomial_remainder<F: Field>(
    field: &F,
    mut dividend: Vec<F::Element>,
    divisor: &[F::Element],
) -> Vec<F::Element> {
    trim(field, &mut dividend);
    let inverse = field.inverse(divisor.last().expect("nonzero polynomial divisor"));
    while dividend.len() >= divisor.len() {
        let shift = dividend.len() - divisor.len();
        let scale = field.mul(dividend.last().expect("nonzero dividend"), &inverse);
        for (degree, coefficient) in divisor.iter().enumerate() {
            let term = field.mul(&scale, coefficient);
            let target = shift + degree;
            dividend[target] = field.add(&dividend[target], &field.neg(&term));
        }
        trim(field, &mut dividend);
    }
    dividend
}

fn trim<F: Field>(field: &F, polynomial: &mut Vec<F::Element>) {
    while polynomial.last().is_some_and(|value| field.is_zero(value)) {
        polynomial.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{groebner_basis_prime, PrimeField32};

    fn polynomial(field: &PrimeField32, terms: &[(i64, &[u32])]) -> SparsePolynomial<u32> {
        SparsePolynomial::new(
            terms.first().map_or(0, |(_, exponents)| exponents.len()),
            MonomialOrder::GradedReverseLexicographic,
            terms.iter().map(|(coefficient, exponents)| {
                (
                    Monomial::new(exponents.to_vec().into_boxed_slice()),
                    field.element(*coefficient),
                )
            }),
            field,
        )
    }

    #[test]
    fn radical_system_round_trips_through_one_parameter() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            polynomial(&field, &[(1, &[2, 0]), (-1, &[0, 1])]),
            polynomial(&field, &[(1, &[0, 2]), (-1, &[1, 0])]),
        ];
        let basis = groebner_basis_prime(&field, &generators, &GroebnerOptions::default())
            .unwrap()
            .basis;
        let output =
            rational_univariate_representation(&field, &basis, &GroebnerOptions::default())
                .unwrap();

        assert_eq!(output.metrics.quotient_dimension, 4);
        assert_eq!(output.equation.leading_monomial().unwrap().exponent(0), 4);
        assert_eq!(output.coordinates.len(), 2);
        assert!(output.is_reduced);
    }

    #[test]
    fn nonreduced_monogenic_system_is_reported_honestly() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![polynomial(&field, &[(1, &[3])])];
        let basis = groebner_basis_prime(&field, &generators, &GroebnerOptions::default())
            .unwrap()
            .basis;
        let output =
            rational_univariate_representation(&field, &basis, &GroebnerOptions::default())
                .unwrap();

        assert!(!output.is_reduced);
        assert_eq!(output.equation.leading_monomial().unwrap().exponent(0), 3);
        assert_eq!(output.coordinates[0].terms().len(), 1);
    }

    #[test]
    fn nonmonogenic_algebra_fails_closed() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            polynomial(&field, &[(1, &[2, 0])]),
            polynomial(&field, &[(1, &[1, 1])]),
            polynomial(&field, &[(1, &[0, 2])]),
        ];
        let basis = groebner_basis_prime(&field, &generators, &GroebnerOptions::default())
            .unwrap()
            .basis;
        assert!(matches!(
            rational_univariate_representation(&field, &basis, &GroebnerOptions::default(),),
            Err(GroebnerError::ResourceLimit {
                resource: "RUR primitive linear forms",
                ..
            })
        ));
    }
}
