// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integral Kedlaya--Harrison reduction in the formal `x^i dx/y^3` module.
//!
//! Harrison, arXiv:1006.4206, Lemmas 3.3--3.4 and Proposition 3.5, supplies
//! integrality for every odd prime, including p=3. The even-degree formal
//! module has a one-dimensional kernel with q-Frobenius eigenvalue 1.
//! All coefficient arithmetic stays in one explicitly allocated Witt ring.
//! Divisions carry a common p-power denominator; the work cap pays for every
//! division, and exact uniformizer division explicitly lowers it at the end.

use crate::linalg::{FrobeniusSemilinearMap, Matrix};
use crate::numthy::{FiniteField, QadicElement, QadicRing};
use crate::{Integer, Ring};

use super::{HyperellipticError, MonomialOrder, Polynomial, PolynomialRing};

type Result<T> = std::result::Result<T, HyperellipticError>;
type Poly = Polynomial<QadicRing>;

const MAX_WORK_DEGREE: usize = 250_000;
const MAX_WORK_PRECISION: u64 = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Computation {
    pub(super) precision: u64,
    pub(super) working_precision: u64,
    pub(super) terms: usize,
    pub(super) denominator_loss: u64,
    pub(super) frobenius: FrobeniusSemilinearMap<QadicRing>,
    pub(super) linear_frobenius: Matrix<QadicRing>,
}

fn log_floor(mut n: u64, p: u64) -> u64 {
    let mut result = 0;
    while n >= p {
        n /= p;
        result += 1;
    }
    result
}

fn valuation(mut n: u64, p: u64) -> u64 {
    let mut value = 0;
    while n.is_multiple_of(p) {
        n /= p;
        value += 1;
    }
    value
}

// The omitted k-th summand has valuation >=k+1. After standard expansion
// its largest pole is p(2k+3); RednA loses at most floor(log_p(p(2k+3)-2))
// digits, including the primitive giving the reduction (Harrison 3.3(i)).
// Stopping at y^-3 retains the same bound on the primitive's coefficients.
// This lower bound is nondecreasing in k and tends to infinity.
fn tail_bound(k: u64, p: u64) -> u64 {
    (k + 1).saturating_sub(log_floor(p * (2 * k + 3), p))
}

fn binomial(n: usize, k: usize) -> Integer {
    (0..k).fold(Integer::one(), |value, i| {
        (&value * &Integer::from(n - i)) / Integer::from(i + 1)
    })
}

fn lift(value: &Polynomial<FiniteField>, ring: &PolynomialRing<QadicRing>) -> Result<Poly> {
    Ok(ring.from_terms(
        value
            .terms()
            .map(|(exponent, c)| Ok((exponent.to_vec(), ring.coefficient_ring().teichmuller(c)?)))
            .collect::<Result<Vec<_>>>()?,
    )?)
}

fn inverse_derivative(f: &Poly, residue_f: &Polynomial<FiniteField>) -> Result<Poly> {
    let derivative = f.derivative(0)?;
    let residue_inverse = residue_f.derivative(0)?.inverse_mod(residue_f)?;
    let mut inverse = lift(&residue_inverse, f.ring())?;
    let two = f.ring().constant(f.ring().coefficient_ring().element(2))?;
    let mut digits = 1_u64;
    while digits < f.ring().coefficient_ring().absolute_precision() {
        inverse = (&inverse * &(&two - &(&derivative * &inverse)?)?)?
            .monic_div_rem(f)?
            .1;
        digits = digits.saturating_mul(2);
    }
    if (&derivative * &inverse)?.monic_div_rem(f)?.1 != f.ring().one() {
        return Err(HyperellipticError::InvalidEvidence(
            "lifted Bezout identity failed".into(),
        ));
    }
    Ok(inverse)
}

// B/p^e * dx/y^s reduces to (A + 2 C'/(s-2))/p^e * dx/y^(s-2),
// with C = B/f' mod f and B = A f + C f'. Writing s-2 = p^v u,
// the new numerator is p^v A + 2 C'/u and the new denominator is p^(e+v).
// The numerator ring cap never changes inside this loop.
fn reduce(mut b: Poly, mut pole: u64, f: &Poly, inverse: &Poly, p: u64) -> Result<Poly> {
    let coefficients = f.ring().coefficient_ring();
    let derivative = f.derivative(0)?;
    while pole > 3 {
        // B=Qf+R first. Then C=(R/f') mod f and
        // A=Q+(R-Cf')/f. Only the first division sees the high-degree B;
        // the product and remaining divisions have degree below 2 deg(f).
        let (quotient, residue) = b.monic_div_rem(f)?;
        let c = (&residue * inverse)?.monic_div_rem(f)?.1;
        let (correction, remainder) = (&residue - &(&c * &derivative)?)?.monic_div_rem(f)?;
        if !remainder.is_zero() {
            return Err(HyperellipticError::InvalidEvidence(
                "differential Bezout decomposition failed".into(),
            ));
        }
        let a = (&quotient + &correction)?;
        let divisor = pole - 2;
        let loss = valuation(divisor, p);
        let p_power = Integer::from(p).pow(loss);
        let unit = &Integer::from(divisor) / &p_power;
        let unit_inverse = coefficients
            .element(unit)
            .inverse()
            .map_err(|_| HyperellipticError::InvalidEvidence("unit division failed".into()))?;
        let factor = coefficients.mul(&coefficients.element(2), &unit_inverse)?;
        b = (&a.scalar_mul(&coefficients.element(p_power))?
            + &c.derivative(0)?.scalar_mul(&factor)?)?;
        pole -= 2;
    }
    Ok(b)
}

pub(super) fn compute(branch: &Polynomial<FiniteField>, precision: u64) -> Result<Computation> {
    let field = branch.ring().coefficient_ring();
    let p = field
        .characteristic()
        .to_u64_checked()
        .ok_or(HyperellipticError::Unsupported(
            "the Witt backend requires a word-sized characteristic",
        ))?;
    let d = branch
        .degree(None)?
        .and_then(|d| d.to_u64_checked())
        .ok_or(HyperellipticError::InvalidModel(
            "the branch polynomial must have finite positive degree",
        ))?;
    if p == 2 {
        return Err(HyperellipticError::Unsupported("characteristic two"));
    }
    if precision == 0 || precision > MAX_WORK_PRECISION / 4 {
        return Err(HyperellipticError::ResourceLimit {
            resource: "target p-adic precision",
            limit: (MAX_WORK_PRECISION / 4) as usize,
        });
    }
    let terms = (1..MAX_WORK_PRECISION)
        .find(|&k| {
            p.checked_mul(2 * k + 3)
                .is_some_and(|_| tail_bound(k, p) >= precision)
        })
        .ok_or(HyperellipticError::ResourceLimit {
            resource: "Frobenius expansion length",
            limit: MAX_WORK_PRECISION as usize,
        })?;
    let pole = p
        .checked_mul(2 * terms + 1)
        .ok_or(HyperellipticError::ResourceLimit {
            resource: "Frobenius pole order",
            limit: MAX_WORK_DEGREE,
        })?;
    let maximum_degree = d.checked_mul(p).and_then(|n| n.checked_mul(terms + 1));
    if maximum_degree.is_none_or(|n| n > MAX_WORK_DEGREE as u64) {
        return Err(HyperellipticError::ResourceLimit {
            resource: "Frobenius polynomial degree",
            limit: MAX_WORK_DEGREE,
        });
    }
    let denominator_loss = (3..pole).step_by(2).map(|n| valuation(n, p)).sum::<u64>();
    let working_precision =
        precision
            .checked_add(denominator_loss)
            .ok_or(HyperellipticError::ResourceLimit {
                resource: "working p-adic precision",
                limit: MAX_WORK_PRECISION as usize,
            })?;
    if working_precision > MAX_WORK_PRECISION {
        return Err(HyperellipticError::ResourceLimit {
            resource: "working p-adic precision",
            limit: MAX_WORK_PRECISION as usize,
        });
    }
    let monic = branch.monic()?;
    let coefficients = QadicRing::new(field.clone(), working_precision)?;
    let ring = PolynomialRing::new(
        coefficients.clone(),
        branch.ring().variables().to_vec(),
        MonomialOrder::default(),
    )?;
    let f = lift(&monic, &ring)?;
    let inverse = inverse_derivative(&f, &monic)?;
    let fp = f.pow(p)?;
    let substituted = ring.from_terms(
        f.terms()
            .map(|(exponent, c)| (vec![&exponent[0] * &Integer::from(p)], c.frobenius(1))),
    )?;
    let delta = (&substituted - &fp)?;
    let four_inverse = coefficients
        .element(4)
        .inverse()
        .map_err(|_| HyperellipticError::InvalidEvidence("four is not a unit".into()))?;
    let terms = terms as usize;
    let mut binomials = Vec::with_capacity(terms);
    let mut inverse_power = coefficients.one();
    for k in 0..terms {
        // binom(-3/2,k)=(-1)^k(2k+1)binom(2k,k)/4^k has no odd denominator.
        let mut numerator = &Integer::from(2 * k + 1) * &binomial(2 * k, k);
        if k % 2 == 1 {
            numerator = -numerator;
        }
        binomials.push(coefficients.mul(&coefficients.element(numerator), &inverse_power)?);
        inverse_power = coefficients.mul(&inverse_power, &four_inverse)?;
    }
    let mut numerator = ring.constant(binomials[terms - 1].clone())?;
    let mut fp_power = fp.clone();
    for k in (0..terms - 1).rev() {
        numerator = (&(&numerator * &delta)? + &fp_power.scalar_mul(&binomials[k])?)?;
        if k > 0 {
            fp_power = (&fp_power * &fp)?;
        }
    }
    numerator = numerator.scalar_mul(&coefficients.element(p))?;
    let target = QadicRing::new(field.clone(), precision)?;
    let rank = (d - 1) as usize;
    let mut entries = vec![vec![target.zero(); rank]; rank];
    for column in 0..rank {
        let shift = ring.from_terms([(
            vec![Integer::from(p * (column as u64 + 1) - 1)],
            coefficients.one(),
        )])?;
        let reduced = reduce((&numerator * &shift)?, pole, &f, &inverse, p)?;
        if reduced
            .degree(None)?
            .is_some_and(|degree| degree >= Integer::from(rank))
        {
            return Err(HyperellipticError::InvalidEvidence(
                "integral reduction exceeded the formal module rank".into(),
            ));
        }
        for (row, entries) in entries.iter_mut().enumerate() {
            let value = reduced.coefficient(&[Integer::from(row)])?;
            entries[column] = if denominator_loss == 0 {
                value
            } else {
                coefficients
                    .divide_uniformizer_exact(&value, denominator_loss)
                    .map_err(|error| HyperellipticError::Precision(error.to_string()))?
            };
        }
    }
    let frobenius = FrobeniusSemilinearMap::new(Matrix::from_rows(target.clone(), entries)?, 1);
    let mut linear = FrobeniusSemilinearMap::identity(target, rank);
    let mut power = frobenius.clone();
    let mut exponent = field.degree();
    while exponent != 0 {
        if exponent % 2 == 1 {
            linear = linear.compose(&power)?;
        }
        exponent /= 2;
        if exponent != 0 {
            power = power.compose(&power)?;
        }
    }
    Ok(Computation {
        precision,
        working_precision,
        terms,
        denominator_loss,
        frobenius,
        linear_frobenius: linear.matrix().clone(),
    })
}

pub(super) fn base_residue(value: &QadicElement) -> Result<Integer> {
    if value.frobenius(1) != *value {
        return Err(HyperellipticError::InvalidEvidence(
            "a characteristic coefficient was not Frobenius fixed".into(),
        ));
    }
    let coordinates = value.coefficients();
    if coordinates
        .iter()
        .skip(1)
        .any(|c| c.representative().is_none_or(|v| !v.is_zero()))
    {
        return Err(HyperellipticError::InvalidEvidence(
            "a characteristic coefficient did not descend to Z_p".into(),
        ));
    }
    coordinates
        .first()
        .and_then(|c| c.representative())
        .ok_or(HyperellipticError::InvalidEvidence(
            "a characteristic coefficient was not integral".into(),
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Original Bezout reduction, retaining its full high-degree products and
    // two divisions. This is the arithmetic floor for quotient reuse.
    fn direct_reduce(mut b: Poly, mut pole: u64, f: &Poly, inverse: &Poly, p: u64) -> Poly {
        let coefficients = f.ring().coefficient_ring();
        let derivative = f.derivative(0).unwrap();
        while pole > 3 {
            let c = (&b * inverse).unwrap().monic_div_rem(f).unwrap().1;
            let (a, remainder) = (&b - &(&c * &derivative).unwrap())
                .unwrap()
                .monic_div_rem(f)
                .unwrap();
            assert!(remainder.is_zero());
            let power = Integer::from(p).pow(valuation(pole - 2, p));
            let unit = &Integer::from(pole - 2) / &power;
            let factor = coefficients
                .mul(
                    &coefficients.element(2),
                    &coefficients.element(unit).inverse().unwrap(),
                )
                .unwrap();
            b = (&a.scalar_mul(&coefficients.element(power)).unwrap()
                + &c.derivative(0).unwrap().scalar_mul(&factor).unwrap())
                .unwrap();
            pole -= 2;
        }
        b
    }

    #[test]
    fn quotient_reuse_matches_direct_reduction_at_fixed_precision() {
        for (p, extension, degree, pole, precision) in [(3, 2, 5, 27, 17), (13, 1, 11, 39, 13)] {
            let field = FiniteField::extension(p, extension).unwrap();
            let residue =
                PolynomialRing::new(field.clone(), ["x"], MonomialOrder::default()).unwrap();
            let branch = residue
                .from_terms([
                    (vec![Integer::from(degree)], field.one()),
                    (vec![Integer::one()], field.generator()),
                ])
                .unwrap();
            let coefficients = QadicRing::new(field, precision).unwrap();
            let ring =
                PolynomialRing::new(coefficients.clone(), ["x"], MonomialOrder::default()).unwrap();
            let f = lift(&branch, &ring).unwrap();
            let inverse = inverse_derivative(&f, &branch).unwrap();
            let b = ring
                .from_terms((0..6 * degree + 3).map(|i| {
                    (
                        vec![Integer::from(i)],
                        coefficients
                            .add(
                                &coefficients.element(p * i * i + 2),
                                &coefficients
                                    .mul(&coefficients.generator(), &coefficients.element(i))
                                    .unwrap(),
                            )
                            .unwrap(),
                    )
                }))
                .unwrap();
            for scale in [1, p, p * p] {
                let b = b.scalar_mul(&coefficients.element(scale)).unwrap();
                for pole in [3, 7, pole] {
                    assert_eq!(
                        reduce(b.clone(), pole, &f, &inverse, p).unwrap(),
                        direct_reduce(b.clone(), pole, &f, &inverse, p),
                        "p={p}, extension={extension}, degree={degree}, pole={pole}, scale={scale}",
                    );
                }
            }
        }
    }
}
