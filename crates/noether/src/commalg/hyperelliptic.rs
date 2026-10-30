// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Smooth odd-characteristic hyperelliptic models and replayable Frobenius.

use super::{
    kedlaya, CyclicCoverSmoothnessEvidence, CyclicHypersurfaceCover, FrobeniusPolynomial,
    MonomialOrder, Polynomial, PolynomialContextError, PolynomialError, PolynomialRing,
    ProjectiveSpace, WeightedProjectiveError, ZetaFunction,
};
use crate::linalg::{FrobeniusSemilinearMap, LinalgError, Matrix, SemilinearMapError};
use crate::numthy::{FiniteField, QadicContextError, QadicRing};
use crate::{
    ContextMismatchError, Field, Integer, IntegerRing, NoetherError, Ring, SemanticType,
    TypeDescription, Z,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Failure to construct a smooth model or certify its Frobenius computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HyperellipticError {
    /// An exact polynomial operation failed.
    Polynomial(PolynomialError),
    /// A polynomial context was invalid.
    PolynomialContext(PolynomialContextError),
    /// Weighted-cover geometry failed.
    Geometry(WeightedProjectiveError),
    /// Values belonged to different coefficient contexts.
    ContextMismatch(ContextMismatchError),
    /// A Witt-ring context was unavailable.
    Qadic(QadicContextError),
    /// A matrix operation failed.
    LinearAlgebra(LinalgError),
    /// Semilinear composition failed.
    Semilinear(SemilinearMapError),
    /// The equation was not a smooth positive-genus hyperelliptic model.
    InvalidModel(&'static str),
    /// A stated regime is unsupported.
    Unsupported(&'static str),
    /// Precision was insufficient or a claimed integral division failed.
    Precision(String),
    /// Replayed evidence did not agree with its mathematical claims.
    InvalidEvidence(String),
    /// A computation exceeded an explicit allocation bound.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Maximum admitted amount.
        limit: usize,
    },
}

impl fmt::Display for HyperellipticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Polynomial(e) => e.fmt(f),
            Self::PolynomialContext(e) => e.fmt(f),
            Self::Geometry(e) => e.fmt(f),
            Self::ContextMismatch(e) => e.fmt(f),
            Self::Qadic(e) => e.fmt(f),
            Self::LinearAlgebra(e) => e.fmt(f),
            Self::Semilinear(e) => e.fmt(f),
            Self::InvalidModel(e) => write!(f, "invalid hyperelliptic model: {e}"),
            Self::Unsupported(e) => write!(f, "unsupported hyperelliptic regime: {e}"),
            Self::Precision(e) => write!(f, "insufficient hyperelliptic precision: {e}"),
            Self::InvalidEvidence(e) => write!(f, "invalid hyperelliptic evidence: {e}"),
            Self::ResourceLimit { resource, limit } => {
                write!(f, "hyperelliptic {resource} exceeded limit {limit}")
            }
        }
    }
}
impl std::error::Error for HyperellipticError {}
macro_rules! conversion {
    ($source:ty, $variant:ident) => {
        impl From<$source> for HyperellipticError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}
conversion!(PolynomialError, Polynomial);
conversion!(PolynomialContextError, PolynomialContext);
conversion!(WeightedProjectiveError, Geometry);
conversion!(ContextMismatchError, ContextMismatch);
conversion!(QadicContextError, Qadic);
conversion!(LinalgError, LinearAlgebra);
conversion!(SemilinearMapError, Semilinear);
type Result<T> = std::result::Result<T, HyperellipticError>;

/// A checked smooth projective model of `y^2 + h(x)y = f(x)` over a finite field.
///
/// The characteristic must be odd and `f+h^2/4` must be squarefree of degree
/// `2g+1` or `2g+2`, with `g>=1`. The weighted realization uses the explicit
/// coordinate `Y=y+h/2`. Infinity is retained, including a nonsplit degree-two
/// point for even-degree models with nonsquare leading coefficient.
#[derive(Clone, PartialEq, Eq)]
pub struct HyperellipticCurve {
    f: Polynomial<FiniteField>,
    h: Polynomial<FiniteField>,
    branch: Polynomial<FiniteField>,
    genus: usize,
    cover: CyclicHypersurfaceCover<FiniteField>,
}

impl HyperellipticCurve {
    /// Construct the smooth projective model of `y^2=f(x)`.
    pub fn new(f: Polynomial<FiniteField>) -> Result<Self> {
        let h = f.ring().zero();
        Self::from_equation(f, h)
    }

    /// Construct `y^2+h(x)y=f(x)`, retaining the checked square-completion map.
    pub fn from_equation(f: Polynomial<FiniteField>, h: Polynomial<FiniteField>) -> Result<Self> {
        if f.ring() != h.ring() {
            return Err(HyperellipticError::ContextMismatch(ContextMismatchError {
                expected: f.ring().to_string(),
                actual: h.ring().to_string(),
            }));
        }
        if f.ring().variables().len() != 1 {
            return Err(HyperellipticError::InvalidModel(
                "the equation must be univariate",
            ));
        }
        let field = f.ring().coefficient_ring();
        if field.characteristic() == &Integer::from(2) {
            return Err(HyperellipticError::Unsupported("characteristic two"));
        }
        let inverse_four = field
            .inverse(&field.element(4))
            .map_err(|_| HyperellipticError::InvalidModel("four is not invertible"))?;
        let branch = (&f + &h.pow(2)?.scalar_mul(&inverse_four)?)?;
        let degree = branch
            .degree(None)?
            .and_then(|d| d.to_u64_checked())
            .ok_or(HyperellipticError::InvalidModel(
                "the completed square has no positive degree",
            ))?;
        if degree < 3 {
            return Err(HyperellipticError::InvalidModel(
                "the completed square must have degree at least three",
            ));
        }
        if branch.gcd(&branch.derivative(0)?)? != branch.ring().one() {
            return Err(HyperellipticError::InvalidModel(
                "the completed square is not squarefree",
            ));
        }
        let genus =
            usize::try_from((degree - 1) / 2).map_err(|_| HyperellipticError::ResourceLimit {
                resource: "genus",
                limit: usize::MAX,
            })?;
        let homogeneous_degree = genus.checked_add(1).and_then(|n| n.checked_mul(2)).ok_or(
            HyperellipticError::ResourceLimit {
                resource: "weighted degree",
                limit: usize::MAX,
            },
        )?;
        let base = ProjectiveSpace::with_names(field.clone(), 1, ["x", "z"])
            .map_err(WeightedProjectiveError::from)?;
        let homogeneous =
            base.coordinate_ring()
                .from_terms(branch.terms().map(|(exponent, coefficient)| {
                    (
                        vec![
                            exponent[0].clone(),
                            &Integer::from(homogeneous_degree) - &exponent[0],
                        ],
                        coefficient.clone(),
                    )
                }))?;
        let cover = CyclicHypersurfaceCover::<FiniteField>::new(base, 2, homogeneous)?;
        Ok(Self {
            f,
            h,
            branch,
            genus,
            cover,
        })
    }
    /// Exact finite base field, including its presentation.
    pub fn base(&self) -> &FiniteField {
        self.f.ring().coefficient_ring()
    }
    /// Right-hand side of the original equation.
    pub fn f(&self) -> &Polynomial<FiniteField> {
        &self.f
    }
    /// Coefficient of `y` in the original equation.
    pub fn h(&self) -> &Polynomial<FiniteField> {
        &self.h
    }
    /// Branch polynomial `f+h^2/4` in the coordinate `Y=y+h/2`.
    pub fn branch_equation(&self) -> &Polynomial<FiniteField> {
        &self.branch
    }
    /// Geometric genus of the smooth projective model.
    pub fn genus(&self) -> usize {
        self.genus
    }
    /// The degree-two cover of P1 in completed-square coordinates.
    pub fn cover(&self) -> &CyclicHypersurfaceCover<FiniteField> {
        &self.cover
    }
    /// The polynomial shift in the explicit change of coordinate `Y=y+h/2`.
    pub fn completion_shift(&self) -> Result<Polynomial<FiniteField>> {
        let half = self
            .base()
            .inverse(&self.base().element(2))
            .map_err(|_| HyperellipticError::InvalidModel("two is not invertible"))?;
        Ok(self.h.scalar_mul(&half)?)
    }
    /// Residue degrees of the closed points at infinity, in ascending order.
    pub fn infinity_residue_degrees(&self) -> Vec<usize> {
        if self
            .branch
            .degree(None)
            .ok()
            .flatten()
            .is_some_and(|d| d == Integer::from(2 * self.genus + 1))
        {
            vec![1]
        } else if self.quadratic_twist_sign() == 1 {
            vec![1, 1]
        } else {
            vec![2]
        }
    }
    /// Replay the weighted-cover Jacobian and smooth-branch criteria.
    pub fn smoothness_evidence(&self) -> Result<CyclicCoverSmoothnessEvidence<FiniteField>> {
        Ok(self.cover.smoothness_evidence()?)
    }
    /// Reconstruct the original equation, square completion, and weighted model.
    pub fn verify(&self) -> Result<()> {
        if Self::from_equation(self.f.clone(), self.h.clone())? != *self {
            return Err(HyperellipticError::InvalidEvidence(
                "the weighted model changed under reconstruction".into(),
            ));
        }
        Ok(())
    }
    /// Compute the compact H1 Frobenius polynomial by integral p-adic reduction.
    pub fn frobenius_polynomial(&self) -> Result<FrobeniusPolynomial> {
        let evidence = HyperellipticFrobeniusEvidence::compute(self.clone(), None)?;
        FrobeniusPolynomial::from_hyperelliptic(evidence)
    }
    /// Compute with an explicit target p-adic precision, rejecting too few digits.
    pub fn frobenius_evidence(
        &self,
        precision: Option<u64>,
    ) -> Result<HyperellipticFrobeniusEvidence> {
        HyperellipticFrobeniusEvidence::compute(self.clone(), precision)
    }
    /// Exact zeta function of the smooth projective model.
    pub fn zeta_function(&self) -> Result<ZetaFunction> {
        self.frobenius_polynomial()?
            .zeta_function()
            .map_err(|e| HyperellipticError::InvalidEvidence(e.to_string()))
    }

    /// Count points over a relative extension without enumerating that field.
    pub fn point_count(&self, extension_degree: u64) -> Result<Integer> {
        self.frobenius_polynomial()?
            .point_count(extension_degree)
            .map_err(|e| HyperellipticError::InvalidEvidence(e.to_string()))
    }
    pub(super) fn quadratic_twist_sign(&self) -> i8 {
        let mut exponent = (&self.base().order() - &Integer::one()) / Integer::from(2);
        let mut value = self
            .branch
            .leading_coefficient()
            .expect("checked positive degree");
        let mut power = self.base().one();
        while !exponent.is_zero() {
            if exponent.modulo_u64(2) == 1 {
                power = self.base().mul(&power, &value).expect("one field");
            }
            value = self.base().mul(&value, &value).expect("one field");
            exponent = exponent.div_floor(&Integer::from(2));
        }
        if power == self.base().one() {
            1
        } else {
            -1
        }
    }
}

fn binomial(n: usize, k: usize) -> Integer {
    (0..k).fold(Integer::one(), |a, i| {
        (&a * &Integer::from(n - i)) / Integer::from(i + 1)
    })
}
fn minimum_precision(curve: &HyperellipticCurve) -> u64 {
    let magnitude_squared = curve.base().order().pow(curve.genus as u64);
    let mut magnitude = magnitude_squared.sqrt_floor();
    if magnitude.square() != magnitude_squared {
        magnitude = &magnitude + &Integer::one();
    }
    let bound = &(&Integer::from(2) * &binomial(2 * curve.genus, curve.genus)) * &magnitude;
    let mut precision = 1;
    while curve.base().characteristic().pow(precision) <= bound {
        precision += 1;
    }
    precision
}

/// Replayable integral Frobenius on the monic model's `x^i dx/y^3` module.
///
/// In even degree the formal module has one extra eigenvalue 1, which is
/// removed before the recorded quadratic twist and compact H1 reconstruction.
/// The matrices belong to the normalized monic model, while `curve()` retains
/// the original equation. `frobenius_polynomial()` restores the original twist.
#[derive(Clone, PartialEq, Eq)]
pub struct HyperellipticFrobeniusEvidence {
    curve: HyperellipticCurve,
    computation: kedlaya::Computation,
}

impl HyperellipticFrobeniusEvidence {
    fn compute(curve: HyperellipticCurve, precision: Option<u64>) -> Result<Self> {
        let rank = curve
            .branch
            .degree(None)?
            .and_then(|d| d.to_u64_checked())
            .and_then(|d| usize::try_from(d - 1).ok())
            .ok_or(HyperellipticError::ResourceLimit {
                resource: "Frobenius module rank",
                limit: usize::MAX,
            })?;
        const MAX_MATRIX_CELLS: usize = 4_194_304;
        if rank
            .checked_mul(rank)
            .is_none_or(|cells| cells > MAX_MATRIX_CELLS)
        {
            return Err(HyperellipticError::ResourceLimit {
                resource: "Frobenius matrix cells",
                limit: MAX_MATRIX_CELLS,
            });
        }
        if curve
            .genus
            .checked_mul(curve.base().degree())
            .is_none_or(|n| n > 32_768)
        {
            return Err(HyperellipticError::ResourceLimit {
                resource: "minimum p-adic precision",
                limit: 16_384,
            });
        }
        let required = minimum_precision(&curve);
        let precision = precision.unwrap_or(required);
        if precision < required {
            return Err(HyperellipticError::Precision(format!(
                "need at least {required} digits"
            )));
        }
        let computation = kedlaya::compute(&curve.branch, precision)?;
        let result = Self { curve, computation };
        result.polynomial()?;
        Ok(result)
    }
    /// Source model, retaining its original equation.
    pub fn curve(&self) -> &HyperellipticCurve {
        &self.curve
    }

    /// Monic finite-field branch polynomial whose Witt lift supplies the matrices.
    pub fn normalized_branch(&self) -> Result<Polynomial<FiniteField>> {
        Ok(self.curve.branch.monic()?)
    }

    /// Sign restoring the original curve from its normalized quadratic twist.
    pub fn quadratic_twist_sign(&self) -> i8 {
        self.curve.quadratic_twist_sign()
    }
    /// Surviving absolute p-adic precision of the matrix.
    pub fn precision(&self) -> u64 {
        self.computation.precision
    }
    /// Explicit absolute precision allocated before differential reduction.
    pub fn working_precision(&self) -> u64 {
        self.computation.working_precision
    }
    /// Number of retained binomial summands.
    pub fn binomial_terms(&self) -> usize {
        self.computation.terms
    }
    /// Digits reserved for the common denominator in exact reduction.
    pub fn denominator_loss(&self) -> u64 {
        self.computation.denominator_loss
    }
    /// Integral semilinear p-Frobenius on the normalized monic model's module.
    pub fn frobenius(&self) -> &FrobeniusSemilinearMap<QadicRing> {
        &self.computation.frobenius
    }
    /// Linear q-Frobenius of the normalized monic model, by semilinear composition.
    pub fn linear_frobenius(&self) -> &Matrix<QadicRing> {
        &self.computation.linear_frobenius
    }

    /// Reconstruct compact H1 from these precision-checked matrix residues.
    pub fn frobenius_polynomial(&self) -> Result<FrobeniusPolynomial> {
        FrobeniusPolynomial::from_hyperelliptic(self.clone())
    }
    /// Replay reduction and compare matrices and every recorded precision bound.
    pub fn verify(&self) -> Result<()> {
        self.curve.verify()?;
        let rebuilt = Self::compute(self.curve.clone(), Some(self.precision()))?;
        if rebuilt != *self {
            return Err(HyperellipticError::InvalidEvidence(
                "Frobenius replay changed the transcript".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn polynomial(&self) -> Result<Polynomial<IntegerRing>> {
        let q = self.curve.base().order();
        let g = self.curve.genus;
        let mut characteristic = self.linear_frobenius().charpoly()?;
        if self.curve.branch.degree(None)? == Some(Integer::from(2 * g + 2)) {
            let t = characteristic.ring().generator(0)?;
            let divisor = (&t - &characteristic.ring().one())?;
            let (quotient, remainder) = characteristic.monic_div_rem(&divisor)?;
            if !remainder.is_zero() {
                return Err(HyperellipticError::InvalidEvidence(
                    "even-degree kernel eigenvalue 1 was absent".into(),
                ));
            }
            characteristic = quotient;
        }
        let modulus = self.curve.base().characteristic().pow(self.precision());
        let sign = self.curve.quadratic_twist_sign();
        let mut descending = vec![Integer::zero(); 2 * g + 1];
        descending[0] = Integer::one();
        for (i, c) in descending.iter_mut().enumerate().take(g + 1).skip(1) {
            let mut residue =
                kedlaya::base_residue(&characteristic.coefficient(&[Integer::from(2 * g - i)])?)?;
            if sign == -1 && i % 2 == 1 {
                residue = &modulus - &residue;
                if residue == modulus {
                    residue = Integer::zero();
                }
            }
            *c = if &Integer::from(2) * &residue > modulus {
                &residue - &modulus
            } else {
                residue
            };
            if c.square() > &binomial(2 * g, i).square() * &q.pow(i as u64) {
                return Err(HyperellipticError::InvalidEvidence(
                    "a coefficient exceeded its Weil bound".into(),
                ));
            }
        }
        for i in 0..g {
            descending[2 * g - i] = &descending[i] * &q.pow((g - i) as u64);
        }
        for (i, c) in descending.iter().enumerate() {
            let untwisted = if sign == -1 && i % 2 == 1 {
                -c
            } else {
                c.clone()
            };
            let expected = self.frobenius().context().element(untwisted);
            if characteristic.coefficient(&[Integer::from(2 * g - i)])? != expected {
                return Err(HyperellipticError::InvalidEvidence(
                    "the characteristic polynomial violated compact duality".into(),
                ));
            }
        }
        let ring = PolynomialRing::new(Z, ["T"], MonomialOrder::Lexicographic)?;
        Ok(ring.from_terms(
            descending
                .into_iter()
                .enumerate()
                .map(|(i, c)| (vec![Integer::from(2 * g - i)], c)),
        )?)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct HyperellipticCurveWire {
    #[serde(rename = "type")]
    type_id: String,
    f: Polynomial<FiniteField>,
    h: Polynomial<FiniteField>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceWire {
    #[serde(rename = "type")]
    type_id: String,
    curve: HyperellipticCurve,
    precision: u64,
    working_precision: u64,
    binomial_terms: usize,
    denominator_loss: u64,
    frobenius: FrobeniusSemilinearMap<QadicRing>,
    linear_frobenius: Matrix<QadicRing>,
}
impl From<&HyperellipticCurve> for HyperellipticCurveWire {
    fn from(value: &HyperellipticCurve) -> Self {
        Self {
            type_id: HyperellipticCurve::TYPE_ID.into(),
            f: value.f.clone(),
            h: value.h.clone(),
        }
    }
}
impl From<&HyperellipticFrobeniusEvidence> for EvidenceWire {
    fn from(value: &HyperellipticFrobeniusEvidence) -> Self {
        Self {
            type_id: HyperellipticFrobeniusEvidence::TYPE_ID.into(),
            curve: value.curve.clone(),
            precision: value.precision(),
            working_precision: value.working_precision(),
            binomial_terms: value.binomial_terms(),
            denominator_loss: value.denominator_loss(),
            frobenius: value.frobenius().clone(),
            linear_frobenius: value.linear_frobenius().clone(),
        }
    }
}
impl HyperellipticCurve {
    /// Serialize the original coefficient context and equation.
    pub fn to_json(&self) -> std::result::Result<String, NoetherError> {
        serde_json::to_string(&HyperellipticCurveWire::from(self))
            .map_err(|e| NoetherError::Serialization(e.to_string()))
    }
    /// Restore and independently recheck smoothness and the weighted model.
    pub fn from_json(value: &str) -> std::result::Result<Self, NoetherError> {
        let wire: HyperellipticCurveWire =
            serde_json::from_str(value).map_err(|e| NoetherError::Serialization(e.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(
                "wrong hyperelliptic type".into(),
            ));
        }
        Self::from_equation(wire.f, wire.h).map_err(|e| NoetherError::Serialization(e.to_string()))
    }
}
impl HyperellipticFrobeniusEvidence {
    /// Serialize the source, precision budget and both Frobenius matrices.
    pub fn to_json(&self) -> std::result::Result<String, NoetherError> {
        serde_json::to_string(&EvidenceWire::from(self))
            .map_err(|e| NoetherError::Serialization(e.to_string()))
    }
    /// Restore and replay the full computation, rejecting altered evidence.
    pub fn from_json(value: &str) -> std::result::Result<Self, NoetherError> {
        let w: EvidenceWire =
            serde_json::from_str(value).map_err(|e| NoetherError::Serialization(e.to_string()))?;
        if w.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(
                "wrong hyperelliptic evidence type".into(),
            ));
        }
        let result = Self {
            curve: w.curve,
            computation: kedlaya::Computation {
                precision: w.precision,
                working_precision: w.working_precision,
                terms: w.binomial_terms,
                denominator_loss: w.denominator_loss,
                frobenius: w.frobenius,
                linear_frobenius: w.linear_frobenius,
            },
        };
        result
            .verify()
            .map_err(|e| NoetherError::Serialization(e.to_string()))?;
        Ok(result)
    }
}
macro_rules! semantic {
    ($name:ident, $wire:ty, $id:literal, $rust:literal, $python:literal) => {
        impl SemanticType for $name {
            const TYPE_ID: &'static str = $id;
            fn canonical_bytes(&self) -> std::result::Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }
            fn describe() -> &'static TypeDescription {
                static DESCRIPTION: TypeDescription = TypeDescription {
                    semantic_id: $id,
                    rust_path: $rust,
                    python_path: Some($python),
                };
                &DESCRIPTION
            }
        }
        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                <$wire>::from(self).serialize(serializer)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(
                deserializer: D,
            ) -> std::result::Result<Self, D::Error> {
                let value = serde_json::Value::deserialize(deserializer)?;
                Self::from_json(&value.to_string()).map_err(serde::de::Error::custom)
            }
        }
        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(concat!(stringify!($name), "(...)"))
            }
        }
        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(self, f)
            }
        }
    };
}
semantic!(
    HyperellipticCurve,
    HyperellipticCurveWire,
    "noether/commalg/HyperellipticCurve@1",
    "noether::commalg::HyperellipticCurve",
    "noether.commalg.HyperellipticCurve"
);
semantic!(
    HyperellipticFrobeniusEvidence,
    EvidenceWire,
    "noether/commalg/HyperellipticFrobeniusEvidence@1",
    "noether::commalg::HyperellipticFrobeniusEvidence",
    "noether.commalg.HyperellipticFrobeniusEvidence"
);
