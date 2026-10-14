// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Generalized Weierstrass elliptic curves and explicit descent functions.

use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use std::any::Any;
use std::sync::OnceLock;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::commalg::{
    Curve, CurveError, CurveFunction, CurvePoint, Polynomial, PolynomialFactorizationError,
    RationalFunction, RationalFunctionError, RationalFunctionField, WeilDivisor,
};
use crate::{
    ContextMismatchError, Field, Integer, InversionError, NoetherError, Rational, RationalField,
    Ring, SemanticType, TypeDescription, VerificationError, Q,
};

use super::{factor, FactorError, FiniteField, FiniteFieldElement};

static ELLIPTIC_CURVE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/EllipticCurve@1",
    rust_path: "noether::numthy::EllipticCurve",
    python_path: Some("noether.numthy.EllipticCurve"),
};

static ELLIPTIC_CURVE_POINT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/EllipticCurvePoint@1",
    rust_path: "noether::numthy::EllipticCurvePoint",
    python_path: Some("noether.numthy.EllipticCurvePoint"),
};

static MILLER_FUNCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/MillerFunction@1",
    rust_path: "noether::numthy::MillerFunction",
    python_path: Some("noether.numthy.MillerFunction"),
};

static NTH_POWER_CLASS_GROUP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/NthPowerClassGroup@1",
    rust_path: "noether::numthy::NthPowerClassGroup",
    python_path: Some("noether.numthy.NthPowerClassGroup"),
};

static NTH_POWER_CLASS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/NthPowerClass@1",
    rust_path: "noether::numthy::NthPowerClass",
    python_path: Some("noether.numthy.NthPowerClass"),
};

/// Failure to construct an elliptic curve or perform its exact group law.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EllipticCurveError {
    /// The checked underlying curve rejected the model, point, or function.
    Curve(CurveError),
    /// Operands belonged to unequal coefficient or elliptic-curve contexts.
    ContextMismatch(ContextMismatchError),
    /// An exact field denominator was zero or belonged to another context.
    Inversion(InversionError),
    /// A point at infinity was supplied to an operation requiring affine coordinates.
    PointAtInfinity,
    /// A Miller multiplier or candidate order was not positive.
    NonPositiveMultiplier,
    /// A claimed exact-order witness was false.
    IncorrectOrder,
    /// A reconstructed result failed an internal mathematical invariant.
    InvalidWitness(&'static str),
}

impl fmt::Display for EllipticCurveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Curve(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Inversion(error) => error.fmt(formatter),
            Self::PointAtInfinity => {
                formatter.write_str("this elliptic-curve operation requires an affine point")
            }
            Self::NonPositiveMultiplier => {
                formatter.write_str("an elliptic multiplier or candidate order must be positive")
            }
            Self::IncorrectOrder => formatter.write_str(
                "the supplied candidate is not the exact order of this elliptic-curve point",
            ),
            Self::InvalidWitness(claim) => write!(formatter, "invalid elliptic witness: {claim}"),
        }
    }
}

impl std::error::Error for EllipticCurveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Curve(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::Inversion(error) => Some(error),
            Self::PointAtInfinity
            | Self::NonPositiveMultiplier
            | Self::IncorrectOrder
            | Self::InvalidWitness(_) => None,
        }
    }
}

impl From<CurveError> for EllipticCurveError {
    fn from(value: CurveError) -> Self {
        Self::Curve(value)
    }
}

impl From<ContextMismatchError> for EllipticCurveError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<InversionError> for EllipticCurveError {
    fn from(value: InversionError) -> Self {
        Self::Inversion(value)
    }
}

/// A nonsingular generalized Weierstrass equation with chosen origin.
///
/// The checked [`Curve`] is the sole owner of the plane-curve semantics. This
/// value adds the distinguished origin and group law without duplicating a
/// second curve implementation.
#[derive(Clone, PartialEq, Eq)]
pub struct EllipticCurve<R: Field> {
    curve: Curve<R>,
    coefficients: [R::Element; 5],
    discriminant: R::Element,
    short_model: Option<ShortWeierstrassModel<R::Element>>,
}

#[derive(Clone, PartialEq, Eq)]
struct ShortWeierstrassModel<E> {
    identity_coordinates: bool,
    x_shift: E,
    y_x: E,
    y_constant: E,
    a: E,
}

#[derive(Clone, Copy)]
struct WordJacobianPoint {
    x: u64,
    y: u64,
    z: u64,
}

#[derive(Clone, Copy)]
struct WordPrimeArithmetic {
    modulus: u64,
}

impl WordPrimeArithmetic {
    fn add(self, left: u64, right: u64) -> u64 {
        ((u128::from(left) + u128::from(right)) % u128::from(self.modulus)) as u64
    }

    fn sub(self, left: u64, right: u64) -> u64 {
        if left >= right {
            left - right
        } else {
            self.modulus - (right - left)
        }
    }

    fn mul(self, left: u64, right: u64) -> u64 {
        ((u128::from(left) * u128::from(right)) % u128::from(self.modulus)) as u64
    }

    fn multiple(self, value: u64, factor: u64) -> u64 {
        ((u128::from(value) * u128::from(factor)) % u128::from(self.modulus)) as u64
    }

    fn inverse(self, value: u64) -> u64 {
        debug_assert_ne!(value, 0);
        let mut result = 1_u64;
        let mut base = value;
        let mut exponent = self.modulus - 2;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = self.mul(result, base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = self.mul(base, base);
            }
        }
        result
    }

    fn double(self, point: WordJacobianPoint, curve_a: u64) -> WordJacobianPoint {
        if point.z == 0 || point.y == 0 {
            return WordJacobianPoint { x: 0, y: 1, z: 0 };
        }
        let a = self.mul(point.x, point.x);
        let b = self.mul(point.y, point.y);
        let c = self.mul(b, b);
        let x_plus_b = self.add(point.x, b);
        let d = self.multiple(self.sub(self.sub(self.mul(x_plus_b, x_plus_b), a), c), 2);
        let z_squared = self.mul(point.z, point.z);
        let e = self.add(
            self.multiple(a, 3),
            self.mul(curve_a, self.mul(z_squared, z_squared)),
        );
        let x = self.sub(self.mul(e, e), self.multiple(d, 2));
        let y = self.sub(self.mul(e, self.sub(d, x)), self.multiple(c, 8));
        let z = self.multiple(self.mul(point.y, point.z), 2);
        WordJacobianPoint { x, y, z }
    }

    fn add_mixed(
        self,
        left: WordJacobianPoint,
        affine_right: WordJacobianPoint,
        curve_a: u64,
    ) -> WordJacobianPoint {
        debug_assert_eq!(affine_right.z, 1);
        if left.z == 0 {
            return affine_right;
        }
        let z1_squared = self.mul(left.z, left.z);
        let u2 = self.mul(affine_right.x, z1_squared);
        let s2 = self.mul(affine_right.y, self.mul(left.z, z1_squared));
        if left.x == u2 {
            return if left.y == s2 {
                self.double(left, curve_a)
            } else {
                WordJacobianPoint { x: 0, y: 1, z: 0 }
            };
        }
        let h = self.sub(u2, left.x);
        let h_squared = self.mul(h, h);
        let i = self.multiple(h_squared, 4);
        let j = self.mul(h, i);
        let r = self.multiple(self.sub(s2, left.y), 2);
        let v = self.mul(left.x, i);
        let x = self.sub(self.sub(self.mul(r, r), j), self.multiple(v, 2));
        let y = self.sub(
            self.mul(r, self.sub(v, x)),
            self.multiple(self.mul(left.y, j), 2),
        );
        let z = self.multiple(self.mul(left.z, h), 2);
        WordJacobianPoint { x, y, z }
    }

    fn normalize(self, point: WordJacobianPoint) -> Option<(u64, u64)> {
        if point.z == 0 {
            return None;
        }
        let inverse_z = self.inverse(point.z);
        let inverse_z_squared = self.mul(inverse_z, inverse_z);
        Some((
            self.mul(point.x, inverse_z_squared),
            self.mul(point.y, self.mul(inverse_z_squared, inverse_z)),
        ))
    }
}

impl<R: Field> EllipticCurve<R> {
    /// Construct
    /// `y^2 + a1*x*y + a3*y = x^3 + a2*x^2 + a4*x + a6`.
    pub fn new(base: R, coefficients: [R::Element; 5]) -> Result<Self, EllipticCurveError> {
        let curve = Curve::smooth_projective_weierstrass(base, coefficients.clone())?;
        let discriminant =
            curve
                .weierstrass_discriminant()?
                .ok_or(EllipticCurveError::InvalidWitness(
                    "a distinguished Weierstrass curve has a discriminant",
                ))?;
        let short_model = Self::short_weierstrass_model(curve.base(), &coefficients)?;
        Ok(Self {
            curve,
            coefficients,
            discriminant,
            short_model,
        })
    }

    fn short_weierstrass_model(
        base: &R,
        coefficients: &[R::Element; 5],
    ) -> Result<Option<ShortWeierstrassModel<R::Element>>, EllipticCurveError> {
        let [a1, a2, a3, a4, _a6] = coefficients;
        if base.is_zero(a1) && base.is_zero(a2) && base.is_zero(a3) {
            return Ok(Some(ShortWeierstrassModel {
                identity_coordinates: true,
                x_shift: base.zero(),
                y_x: base.zero(),
                y_constant: base.zero(),
                a: a4.clone(),
            }));
        }

        let one = base.one();
        let two = natural_multiple(base, &one, 2)?;
        let three = natural_multiple(base, &one, 3)?;
        if base.is_zero(&two) || base.is_zero(&three) {
            return Ok(None);
        }
        let inverse_two = base.inverse(&two)?;
        let twelve = natural_multiple(base, &one, 12)?;
        let forty_eight = natural_multiple(base, &one, 48)?;
        let inverse_twelve = base.inverse(&twelve)?;
        let inverse_forty_eight = base.inverse(&forty_eight)?;

        let a1_squared = base.mul(a1, a1)?;
        let b2 = base.add(&a1_squared, &natural_multiple(base, a2, 4)?)?;
        let b4 = base.add(&base.mul(a1, a3)?, &natural_multiple(base, a4, 2)?)?;
        let c4 = base.sub(&base.mul(&b2, &b2)?, &natural_multiple(base, &b4, 24)?)?;
        Ok(Some(ShortWeierstrassModel {
            identity_coordinates: false,
            x_shift: base.mul(&b2, &inverse_twelve)?,
            y_x: base.mul(a1, &inverse_two)?,
            y_constant: base.mul(a3, &inverse_two)?,
            a: base.neg(&base.mul(&c4, &inverse_forty_eight)?)?,
        }))
    }

    /// Exact base field context.
    pub fn base(&self) -> &R {
        self.curve.base()
    }

    /// Coefficients `(a1,a2,a3,a4,a6)`.
    pub fn coefficients(&self) -> &[R::Element; 5] {
        &self.coefficients
    }

    /// Exact nonzero Weierstrass discriminant.
    pub fn discriminant(&self) -> &R::Element {
        &self.discriminant
    }

    /// Checked underlying smooth projective curve.
    pub fn underlying_curve(&self) -> &Curve<R> {
        &self.curve
    }

    /// Chosen origin, the point `[0:1:0]`.
    pub fn origin(&self) -> Result<EllipticCurvePoint<R>, EllipticCurveError> {
        let point =
            self.curve
                .point([self.base().zero(), self.base().one(), self.base().zero()])?;
        Ok(EllipticCurvePoint {
            curve: self.clone(),
            point,
            affine: None,
            exact_order: OnceLock::new(),
        })
    }

    /// Construct and validate one affine point `(x,y)`.
    pub fn point(
        &self,
        x: R::Element,
        y: R::Element,
    ) -> Result<EllipticCurvePoint<R>, EllipticCurveError> {
        if !self.base().contains(&x) || !self.base().contains(&y) {
            return Err(ContextMismatchError {
                expected: R::TYPE_ID.to_owned(),
                actual: "elliptic point coordinate from another field context".to_owned(),
            }
            .into());
        }
        let point = self
            .curve
            .point([x.clone(), y.clone(), self.base().one()])?;
        Ok(EllipticCurvePoint {
            curve: self.clone(),
            point,
            affine: Some((x, y)),
            exact_order: OnceLock::new(),
        })
    }

    /// Whether a checked point belongs to this exact elliptic context.
    pub fn contains(&self, point: &EllipticCurvePoint<R>) -> bool {
        self == point.curve()
    }

    /// Distinguished `x` coordinate in the underlying curve function field.
    pub fn x_function(&self) -> Result<CurveFunction<R>, EllipticCurveError> {
        Ok(self.curve.function_field()?.x()?)
    }

    /// Distinguished `y` coordinate in the underlying curve function field.
    pub fn y_function(&self) -> Result<CurveFunction<R>, EllipticCurveError> {
        Ok(self.curve.function_field()?.y()?)
    }

    /// The vertical-line function `x-x(P)` through an affine point.
    pub fn vertical_line_function(
        &self,
        point: &EllipticCurvePoint<R>,
    ) -> Result<CurveFunction<R>, EllipticCurveError> {
        self.require_point(point)?;
        let (x, _) = point
            .affine_coordinates()
            .ok_or(EllipticCurveError::PointAtInfinity)?;
        let functions = self.curve.function_field()?;
        Ok((&functions.x()? - &functions.constant(x.clone())?)?)
    }

    /// The chord, tangent, or vertical function through two affine points.
    pub fn line_function(
        &self,
        left: &EllipticCurvePoint<R>,
        right: &EllipticCurvePoint<R>,
    ) -> Result<CurveFunction<R>, EllipticCurveError> {
        self.require_point(left)?;
        self.require_point(right)?;
        let ((left_x, left_y), (right_x, right_y)) =
            match (left.affine_coordinates(), right.affine_coordinates()) {
                (Some(left), Some(right)) => (left, right),
                _ => return Err(EllipticCurveError::PointAtInfinity),
            };
        if left_x == right_x && right_y == &self.negated_y(left_x, left_y)? {
            return self.vertical_line_function(left);
        }
        let (slope, intercept) = if left_x == right_x {
            self.tangent_parameters(left_x, left_y)?
        } else {
            self.chord_parameters(left_x, left_y, right_x, right_y)?
        };
        let functions = self.curve.function_field()?;
        let slope_x = (&functions.constant(slope)? * &functions.x()?)?;
        let line = (&functions.y()? - &slope_x)?;
        Ok((&line - &functions.constant(intercept)?)?)
    }

    /// Compute Miller's function with divisor
    /// `n(P) - (nP) - (n-1)(O)` for positive `n`.
    pub fn miller_function(
        &self,
        point: &EllipticCurvePoint<R>,
        multiplier: &Integer,
    ) -> Result<MillerFunction<R>, EllipticCurveError> {
        self.require_point(point)?;
        if !multiplier.is_positive() {
            return Err(EllipticCurveError::NonPositiveMultiplier);
        }
        let functions = self.curve.function_field()?;
        let mut function = functions.one();
        let mut accumulator = point.clone();
        let two = Integer::from(2_u8);
        let mut quotient = multiplier.clone();
        let mut bits = Vec::new();
        while !quotient.is_zero() {
            bits.push(quotient.mod_u(2) == 1);
            quotient = quotient.div_floor(&two);
        }
        for bit in bits.iter().rev().skip(1) {
            let step = self.miller_step_function(&accumulator, &accumulator)?;
            function = (&(&function * &function)? * &step)?;
            accumulator = accumulator.checked_add(&accumulator)?;
            if *bit {
                let step = self.miller_step_function(&accumulator, point)?;
                function = (&function * &step)?;
                accumulator = accumulator.checked_add(point)?;
            }
        }
        Ok(MillerFunction {
            curve: self.clone(),
            point: point.clone(),
            multiplier: multiplier.clone(),
            terminal_point: accumulator,
            function,
        })
    }

    /// Reconstruct the curve, chosen origin, and nonzero discriminant.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let rebuilt = Self::new(self.base().clone(), self.coefficients.clone()).map_err(|_| {
            VerificationError::InvalidWitness {
                claim: "generalized Weierstrass coefficients define a nonsingular curve",
            }
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "elliptic curve, discriminant, and chosen origin reconstruct canonically",
            })
        }
    }

    /// Canonical typed JSON; nonsingularity is recomputed on restoration.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        encode_json(
            Self::TYPE_ID,
            [
                (
                    "base",
                    serde_json::to_value(self.base()).map_err(json_error)?,
                ),
                (
                    "coefficients",
                    serde_json::to_value(&self.coefficients).map_err(json_error)?,
                ),
            ],
        )
    }

    /// Restore the coefficient context and recheck nonsingularity.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: DeserializeOwned,
        R::Element: DeserializeOwned,
    {
        let mut object = decode_json(value, Self::TYPE_ID, &["base", "coefficients"])?;
        let base = decode_json_field(&mut object, "base")?;
        let coefficients = decode_json_field(&mut object, "coefficients")?;
        Self::new(base, coefficients).map_err(elliptic_json_error)
    }

    fn require_point(&self, point: &EllipticCurvePoint<R>) -> Result<(), EllipticCurveError> {
        if self.contains(point) {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: Self::TYPE_ID.to_owned(),
                actual: "point lies on another elliptic curve".to_owned(),
            }
            .into())
        }
    }

    fn negated_y(&self, x: &R::Element, y: &R::Element) -> Result<R::Element, EllipticCurveError> {
        let a1x = self.base().mul(&self.coefficients[0], x)?;
        let value = self.base().add(y, &a1x)?;
        let value = self.base().add(&value, &self.coefficients[2])?;
        Ok(self.base().neg(&value)?)
    }

    fn chord_parameters(
        &self,
        left_x: &R::Element,
        left_y: &R::Element,
        right_x: &R::Element,
        right_y: &R::Element,
    ) -> Result<(R::Element, R::Element), EllipticCurveError> {
        let denominator = self.base().sub(right_x, left_x)?;
        let inverse = self.base().inverse(&denominator)?;
        let slope = self
            .base()
            .mul(&self.base().sub(right_y, left_y)?, &inverse)?;
        let left_term = self.base().mul(left_y, right_x)?;
        let right_term = self.base().mul(right_y, left_x)?;
        let intercept = self
            .base()
            .mul(&self.base().sub(&left_term, &right_term)?, &inverse)?;
        Ok((slope, intercept))
    }

    fn tangent_parameters(
        &self,
        x: &R::Element,
        y: &R::Element,
    ) -> Result<(R::Element, R::Element), EllipticCurveError> {
        let [a1, a2, a3, a4, a6] = &self.coefficients;
        let x_squared = self.base().mul(x, x)?;
        let x_cubed = self.base().mul(&x_squared, x)?;
        let denominator = {
            let two_y = natural_multiple(self.base(), y, 2)?;
            let a1x = self.base().mul(a1, x)?;
            let value = self.base().add(&two_y, &a1x)?;
            self.base().add(&value, a3)?
        };
        let inverse = self.base().inverse(&denominator)?;
        let numerator = {
            let three_x_squared = natural_multiple(self.base(), &x_squared, 3)?;
            let two_a2x = natural_multiple(self.base(), &self.base().mul(a2, x)?, 2)?;
            let a1y = self.base().mul(a1, y)?;
            let value = self.base().add(&three_x_squared, &two_a2x)?;
            let value = self.base().add(&value, a4)?;
            self.base().sub(&value, &a1y)?
        };
        let slope = self.base().mul(&numerator, &inverse)?;
        let intercept_numerator = {
            let minus_x_cubed = self.base().neg(&x_cubed)?;
            let a4x = self.base().mul(a4, x)?;
            let two_a6 = natural_multiple(self.base(), a6, 2)?;
            let a3y = self.base().mul(a3, y)?;
            let value = self.base().add(&minus_x_cubed, &a4x)?;
            let value = self.base().add(&value, &two_a6)?;
            self.base().sub(&value, &a3y)?
        };
        let intercept = self.base().mul(&intercept_numerator, &inverse)?;
        Ok((slope, intercept))
    }

    fn miller_step_function(
        &self,
        left: &EllipticCurvePoint<R>,
        right: &EllipticCurvePoint<R>,
    ) -> Result<CurveFunction<R>, EllipticCurveError> {
        if left.is_infinity() || right.is_infinity() {
            return Ok(self.curve.function_field()?.one());
        }
        let line = self.line_function(left, right)?;
        let sum = left.checked_add(right)?;
        if sum.is_infinity() {
            Ok(line)
        } else {
            let vertical = self.vertical_line_function(&sum)?;
            Ok((&line / &vertical)?)
        }
    }
}

impl<R: Field> fmt::Debug for EllipticCurve<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EllipticCurve")
            .field("base", &R::TYPE_ID)
            .field("model", &"generalized Weierstrass")
            .finish()
    }
}

impl<R: Field> fmt::Display for EllipticCurve<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "EllipticCurve<{}>", R::TYPE_ID)
    }
}

impl<R: Field> SemanticType for EllipticCurve<R> {
    const TYPE_ID: &'static str = "noether/numthy/EllipticCurve@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.curve.canonical_bytes()?;
        for coefficient in &self.coefficients {
            append_component(&mut output, &coefficient.canonical_bytes()?)?;
        }
        append_component(&mut output, &self.discriminant.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &ELLIPTIC_CURVE_DESCRIPTION
    }
}

/// An affine point or the chosen point at infinity on one elliptic curve.
pub struct EllipticCurvePoint<R: Field> {
    curve: EllipticCurve<R>,
    point: CurvePoint<R>,
    affine: Option<(R::Element, R::Element)>,
    exact_order: OnceLock<Integer>,
}

impl<R: Field> Clone for EllipticCurvePoint<R> {
    fn clone(&self) -> Self {
        let exact_order = OnceLock::new();
        if let Some(order) = self.exact_order.get() {
            let _ = exact_order.set(order.clone());
        }
        Self {
            curve: self.curve.clone(),
            point: self.point.clone(),
            affine: self.affine.clone(),
            exact_order,
        }
    }
}

impl<R: Field> PartialEq for EllipticCurvePoint<R> {
    fn eq(&self, other: &Self) -> bool {
        self.curve == other.curve && self.point == other.point && self.affine == other.affine
    }
}

impl<R: Field> Eq for EllipticCurvePoint<R> {}

#[derive(Clone)]
struct JacobianPoint<E> {
    x: E,
    y: E,
    z: E,
}

impl<R: Field> EllipticCurvePoint<R> {
    /// Owning elliptic-curve context.
    pub fn curve(&self) -> &EllipticCurve<R> {
        &self.curve
    }

    /// Whether this is the chosen point at infinity.
    pub fn is_infinity(&self) -> bool {
        self.affine.is_none()
    }

    /// Affine coordinates, or `None` at infinity.
    pub fn affine_coordinates(&self) -> Option<(&R::Element, &R::Element)> {
        self.affine.as_ref().map(|(x, y)| (x, y))
    }

    /// Checked projective point on the underlying commutative-algebra curve.
    pub fn underlying_curve_point(&self) -> &CurvePoint<R> {
        &self.point
    }

    /// Elliptic inverse.
    pub fn negated(&self) -> Result<Self, EllipticCurveError> {
        let Some((x, y)) = self.affine_coordinates() else {
            return self.curve.origin();
        };
        self.curve.point(x.clone(), self.curve.negated_y(x, y)?)
    }

    /// Exact generalized-Weierstrass group addition.
    pub fn checked_add(&self, other: &Self) -> Result<Self, EllipticCurveError> {
        self.curve.require_point(other)?;
        if self.is_infinity() {
            return Ok(other.clone());
        }
        if other.is_infinity() {
            return Ok(self.clone());
        }
        let (left_x, left_y) = self
            .affine_coordinates()
            .ok_or(EllipticCurveError::PointAtInfinity)?;
        let (right_x, right_y) = other
            .affine_coordinates()
            .ok_or(EllipticCurveError::PointAtInfinity)?;
        if left_x == right_x && right_y == &self.curve.negated_y(left_x, left_y)? {
            return self.curve.origin();
        }
        let (slope, intercept) = if left_x == right_x {
            self.curve.tangent_parameters(left_x, left_y)?
        } else {
            self.curve
                .chord_parameters(left_x, left_y, right_x, right_y)?
        };
        let base = self.curve.base();
        let slope_squared = base.mul(&slope, &slope)?;
        let a1_slope = base.mul(&self.curve.coefficients[0], &slope)?;
        let x3 = base.add(&slope_squared, &a1_slope)?;
        let x3 = base.sub(&x3, &self.curve.coefficients[1])?;
        let x3 = base.sub(&x3, left_x)?;
        let x3 = base.sub(&x3, right_x)?;
        let slope_plus_a1 = base.add(&slope, &self.curve.coefficients[0])?;
        let y3 = base.mul(&slope_plus_a1, &x3)?;
        let y3 = base.add(&y3, &intercept)?;
        let y3 = base.add(&y3, &self.curve.coefficients[2])?;
        self.curve.point(x3, base.neg(&y3)?)
    }

    /// Scalar multiplication by an arbitrary-precision signed integer.
    pub fn scalar_mul(&self, scalar: &Integer) -> Result<Self, EllipticCurveError> {
        if let Some(result) = self.scalar_mul_word_prime(scalar) {
            return result;
        }
        if self.curve.short_model.is_some() {
            return self.scalar_mul_projective(scalar);
        }
        self.scalar_mul_affine(scalar)
    }

    fn scalar_mul_word_prime(&self, scalar: &Integer) -> Option<Result<Self, EllipticCurveError>> {
        let field = (self.curve.base() as &dyn Any).downcast_ref::<FiniteField>()?;
        if field.degree() != 1 {
            return None;
        }
        let short_model = self.curve.short_model.as_ref()?;
        let modulus = field.characteristic().to_u64_checked()?;
        let arithmetic = WordPrimeArithmetic { modulus };
        let curve_a = finite_field_word(&short_model.a)?;
        let Some((x, y)) = self.affine_coordinates() else {
            return Some(self.curve.origin());
        };
        let affine_x = finite_field_word(x)?;
        let mut x = affine_x;
        let mut y = finite_field_word(y)?;
        if !short_model.identity_coordinates {
            let x_shift = finite_field_word(&short_model.x_shift)?;
            let y_x = finite_field_word(&short_model.y_x)?;
            let y_constant = finite_field_word(&short_model.y_constant)?;
            x = arithmetic.add(x, x_shift);
            y = arithmetic.add(y, arithmetic.mul(y_x, affine_x));
            y = arithmetic.add(y, y_constant);
        }
        let affine_point = WordJacobianPoint { x, y, z: 1 };
        let quotient = scalar.abs().to_u64_checked()?;
        if quotient == 0 {
            return Some(self.curve.origin());
        }
        let mut result = affine_point;
        let highest_bit = u64::BITS - 1 - quotient.leading_zeros();
        for bit in (0..highest_bit).rev() {
            result = arithmetic.double(result, curve_a);
            if quotient & (1_u64 << bit) != 0 {
                result = arithmetic.add_mixed(result, affine_point, curve_a);
            }
        }
        let result = match arithmetic.normalize(result) {
            None => self.curve.origin(),
            Some((short_x, short_y)) => {
                let (x, y) = if short_model.identity_coordinates {
                    (short_x, short_y)
                } else {
                    let x = arithmetic.sub(short_x, finite_field_word(&short_model.x_shift)?);
                    let y = arithmetic.sub(
                        short_y,
                        arithmetic.mul(finite_field_word(&short_model.y_x)?, x),
                    );
                    (
                        x,
                        arithmetic.sub(y, finite_field_word(&short_model.y_constant)?),
                    )
                };
                let field_x = field.element(x);
                let field_y = field.element(y);
                let x = (&field_x as &dyn Any).downcast_ref::<R::Element>()?.clone();
                let y = (&field_y as &dyn Any).downcast_ref::<R::Element>()?.clone();
                self.curve.point(x, y)
            }
        };
        Some(result.and_then(|point| {
            if scalar.is_negative() {
                point.negated()
            } else {
                Ok(point)
            }
        }))
    }

    fn scalar_mul_affine(&self, scalar: &Integer) -> Result<Self, EllipticCurveError> {
        let negative = scalar.is_negative();
        let mut result = self.curve.origin()?;
        let mut summand = self.clone();
        if let Some(mut quotient) = scalar.abs().to_u64_checked() {
            while quotient != 0 {
                if quotient & 1 == 1 {
                    result = result.checked_add(&summand)?;
                }
                quotient >>= 1;
                if quotient != 0 {
                    summand = summand.checked_add(&summand)?;
                }
            }
            return if negative {
                result.negated()
            } else {
                Ok(result)
            };
        }

        let mut quotient = scalar.abs();
        let two = Integer::from(2_u8);
        while !quotient.is_zero() {
            if quotient.mod_u(2) == 1 {
                result = result.checked_add(&summand)?;
            }
            quotient = quotient.div_floor(&two);
            if !quotient.is_zero() {
                summand = summand.checked_add(&summand)?;
            }
        }
        if negative {
            result.negated()
        } else {
            Ok(result)
        }
    }

    fn scalar_mul_projective(&self, scalar: &Integer) -> Result<Self, EllipticCurveError> {
        let base = self.curve.base();
        let short_model = self
            .curve
            .short_model
            .as_ref()
            .expect("projective scalar multiplication requires a short model");
        let mut result = JacobianPoint {
            x: base.zero(),
            y: base.one(),
            z: base.zero(),
        };
        let mut summand = match self.affine_coordinates() {
            Some((x, y)) if !short_model.identity_coordinates => {
                let short_x = base.add(x, &short_model.x_shift)?;
                let short_y = base.add(y, &base.mul(&short_model.y_x, x)?)?;
                let short_y = base.add(&short_y, &short_model.y_constant)?;
                JacobianPoint {
                    x: short_x,
                    y: short_y,
                    z: base.one(),
                }
            }
            Some((x, y)) => JacobianPoint {
                x: x.clone(),
                y: y.clone(),
                z: base.one(),
            },
            None => result.clone(),
        };
        let negative = scalar.is_negative();
        if let Some(quotient) = scalar.abs().to_u64_checked() {
            if quotient == 0 || base.is_zero(&summand.z) {
                return self.curve.origin();
            }
            result = summand.clone();
            let highest_bit = u64::BITS - 1 - quotient.leading_zeros();
            for bit in (0..highest_bit).rev() {
                result = self.jacobian_double(&result)?;
                if quotient & (1_u64 << bit) != 0 {
                    result = self.jacobian_add_mixed(&result, &summand)?;
                }
            }
            let affine = self.jacobian_to_affine(&result)?;
            return if negative {
                affine.negated()
            } else {
                Ok(affine)
            };
        }

        let mut quotient = scalar.abs();
        let two = Integer::from(2_u8);
        while !quotient.is_zero() {
            if quotient.mod_u(2) == 1 {
                result = self.jacobian_add(&result, &summand)?;
            }
            quotient = quotient.div_floor(&two);
            if !quotient.is_zero() {
                summand = self.jacobian_double(&summand)?;
            }
        }
        let affine = self.jacobian_to_affine(&result)?;
        if negative {
            affine.negated()
        } else {
            Ok(affine)
        }
    }

    fn jacobian_add_mixed(
        &self,
        left: &JacobianPoint<R::Element>,
        affine_right: &JacobianPoint<R::Element>,
    ) -> Result<JacobianPoint<R::Element>, EllipticCurveError> {
        let base = self.curve.base();
        debug_assert!(affine_right.z == base.one());
        if base.is_zero(&left.z) {
            return Ok(affine_right.clone());
        }
        let z1_squared = base.mul(&left.z, &left.z)?;
        let u2 = base.mul(&affine_right.x, &z1_squared)?;
        let s2 = base.mul(&affine_right.y, &base.mul(&left.z, &z1_squared)?)?;
        if left.x == u2 {
            if left.y == s2 {
                return self.jacobian_double(left);
            }
            return Ok(JacobianPoint {
                x: base.zero(),
                y: base.one(),
                z: base.zero(),
            });
        }
        let h = base.sub(&u2, &left.x)?;
        let h_squared = base.mul(&h, &h)?;
        let i = natural_multiple(base, &h_squared, 4)?;
        let j = base.mul(&h, &i)?;
        let r = natural_multiple(base, &base.sub(&s2, &left.y)?, 2)?;
        let v = base.mul(&left.x, &i)?;
        let x = base.sub(
            &base.sub(&base.mul(&r, &r)?, &j)?,
            &natural_multiple(base, &v, 2)?,
        )?;
        let y = base.sub(
            &base.mul(&r, &base.sub(&v, &x)?)?,
            &natural_multiple(base, &base.mul(&left.y, &j)?, 2)?,
        )?;
        let z = natural_multiple(base, &base.mul(&left.z, &h)?, 2)?;
        Ok(JacobianPoint { x, y, z })
    }

    fn jacobian_double(
        &self,
        point: &JacobianPoint<R::Element>,
    ) -> Result<JacobianPoint<R::Element>, EllipticCurveError> {
        let base = self.curve.base();
        if base.is_zero(&point.z) || base.is_zero(&point.y) {
            return Ok(JacobianPoint {
                x: base.zero(),
                y: base.one(),
                z: base.zero(),
            });
        }
        let a = base.mul(&point.x, &point.x)?;
        let b = base.mul(&point.y, &point.y)?;
        let c = base.mul(&b, &b)?;
        let x_plus_b = base.add(&point.x, &b)?;
        let x_plus_b_squared = base.mul(&x_plus_b, &x_plus_b)?;
        let d = natural_multiple(base, &base.sub(&base.sub(&x_plus_b_squared, &a)?, &c)?, 2)?;
        let z_squared = base.mul(&point.z, &point.z)?;
        let z_fourth = base.mul(&z_squared, &z_squared)?;
        let e = base.add(
            &natural_multiple(base, &a, 3)?,
            &base.mul(
                &self
                    .curve
                    .short_model
                    .as_ref()
                    .expect("Jacobian doubling requires a short model")
                    .a,
                &z_fourth,
            )?,
        )?;
        let f = base.mul(&e, &e)?;
        let x = base.sub(&f, &natural_multiple(base, &d, 2)?)?;
        let y = base.sub(
            &base.mul(&e, &base.sub(&d, &x)?)?,
            &natural_multiple(base, &c, 8)?,
        )?;
        let z = natural_multiple(base, &base.mul(&point.y, &point.z)?, 2)?;
        Ok(JacobianPoint { x, y, z })
    }

    fn jacobian_add(
        &self,
        left: &JacobianPoint<R::Element>,
        right: &JacobianPoint<R::Element>,
    ) -> Result<JacobianPoint<R::Element>, EllipticCurveError> {
        let base = self.curve.base();
        if base.is_zero(&left.z) {
            return Ok(right.clone());
        }
        if base.is_zero(&right.z) {
            return Ok(left.clone());
        }
        let z1_squared = base.mul(&left.z, &left.z)?;
        let z2_squared = base.mul(&right.z, &right.z)?;
        let u1 = base.mul(&left.x, &z2_squared)?;
        let u2 = base.mul(&right.x, &z1_squared)?;
        let s1 = base.mul(&left.y, &base.mul(&right.z, &z2_squared)?)?;
        let s2 = base.mul(&right.y, &base.mul(&left.z, &z1_squared)?)?;
        if u1 == u2 {
            if s1 == s2 {
                return self.jacobian_double(left);
            }
            return Ok(JacobianPoint {
                x: base.zero(),
                y: base.one(),
                z: base.zero(),
            });
        }
        let h = base.sub(&u2, &u1)?;
        let two_h = natural_multiple(base, &h, 2)?;
        let i = base.mul(&two_h, &two_h)?;
        let j = base.mul(&h, &i)?;
        let r = natural_multiple(base, &base.sub(&s2, &s1)?, 2)?;
        let v = base.mul(&u1, &i)?;
        let x = base.sub(
            &base.sub(&base.mul(&r, &r)?, &j)?,
            &natural_multiple(base, &v, 2)?,
        )?;
        let y = base.sub(
            &base.mul(&r, &base.sub(&v, &x)?)?,
            &natural_multiple(base, &base.mul(&s1, &j)?, 2)?,
        )?;
        let z_sum = base.add(&left.z, &right.z)?;
        let z = base.mul(
            &base.sub(
                &base.sub(&base.mul(&z_sum, &z_sum)?, &z1_squared)?,
                &z2_squared,
            )?,
            &h,
        )?;
        Ok(JacobianPoint { x, y, z })
    }

    fn jacobian_to_affine(
        &self,
        point: &JacobianPoint<R::Element>,
    ) -> Result<Self, EllipticCurveError> {
        let base = self.curve.base();
        if base.is_zero(&point.z) {
            return self.curve.origin();
        }
        let inverse_z = base.inverse(&point.z)?;
        let inverse_z_squared = base.mul(&inverse_z, &inverse_z)?;
        let short_x = base.mul(&point.x, &inverse_z_squared)?;
        let short_y = base.mul(&point.y, &base.mul(&inverse_z_squared, &inverse_z)?)?;
        let short_model = self
            .curve
            .short_model
            .as_ref()
            .expect("Jacobian normalization requires a short model");
        if short_model.identity_coordinates {
            self.curve.point(short_x, short_y)
        } else {
            let x = base.sub(&short_x, &short_model.x_shift)?;
            let y = base.sub(&short_y, &base.mul(&short_model.y_x, &x)?)?;
            let y = base.sub(&y, &short_model.y_constant)?;
            self.curve.point(x, y)
        }
    }

    /// Decide whether `candidate` is this point's exact positive order.
    pub fn has_exact_order(&self, candidate: &Integer) -> Result<bool, EllipticCurveError> {
        if !candidate.is_positive() {
            return Err(EllipticCurveError::NonPositiveMultiplier);
        }
        if let Some(exact_order) = self.exact_order.get() {
            return Ok(candidate == exact_order);
        }
        if !self.scalar_mul(candidate)?.is_infinity() {
            return Ok(false);
        }
        let factorization = factor(candidate).map_err(|_| {
            EllipticCurveError::InvalidWitness("candidate-order factorization completes")
        })?;
        for (prime, _) in factorization.factors() {
            let quotient = candidate / prime;
            if self.scalar_mul(&quotient)?.is_infinity() {
                return Ok(false);
            }
        }
        let _ = self.exact_order.set(candidate.clone());
        Ok(true)
    }

    /// Fail unless `candidate` is the exact positive order.
    pub fn verify_exact_order(&self, candidate: &Integer) -> Result<(), EllipticCurveError> {
        if self.has_exact_order(candidate)? {
            Ok(())
        } else {
            Err(EllipticCurveError::IncorrectOrder)
        }
    }

    /// Recheck incidence, affine normalization, and the chosen origin.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.curve.verify()?;
        self.point.verify()?;
        let rebuilt = match &self.affine {
            Some((x, y)) => self.curve.point(x.clone(), y.clone()),
            None => self.curve.origin(),
        }
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "elliptic point coordinates reconstruct on the owning curve",
        })?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "elliptic affine or infinity presentation is canonical",
            })
        }
    }

    /// Canonical typed JSON; point membership is recomputed.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        let coordinates = self.affine.as_ref().map(|(x, y)| [x, y]);
        encode_json(
            Self::TYPE_ID,
            [
                ("curve", nested_json(self.curve.to_json()?)?),
                (
                    "affine_coordinates",
                    serde_json::to_value(coordinates).map_err(json_error)?,
                ),
            ],
        )
    }

    /// Restore and recheck an affine point or the chosen origin.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: DeserializeOwned,
        R::Element: DeserializeOwned,
    {
        let mut object = decode_json(value, Self::TYPE_ID, &["curve", "affine_coordinates"])?;
        let curve = EllipticCurve::<R>::from_json(&nested_json_string(
            object
                .remove("curve")
                .ok_or_else(|| missing_json_field("curve"))?,
        )?)?;
        let coordinates: Option<[R::Element; 2]> =
            decode_json_field(&mut object, "affine_coordinates")?;
        match coordinates {
            Some([x, y]) => curve.point(x, y),
            None => curve.origin(),
        }
        .map_err(elliptic_json_error)
    }
}

impl<R: Field> fmt::Debug for EllipticCurvePoint<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EllipticCurvePoint")
            .field("base", &R::TYPE_ID)
            .field("infinity", &self.is_infinity())
            .finish()
    }
}

impl<R: Field> fmt::Display for EllipticCurvePoint<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinity() {
            write!(formatter, "EllipticCurvePoint<{}>(infinity)", R::TYPE_ID)
        } else {
            write!(formatter, "EllipticCurvePoint<{}>(affine)", R::TYPE_ID)
        }
    }
}

impl<R: Field> SemanticType for EllipticCurvePoint<R> {
    const TYPE_ID: &'static str = "noether/numthy/EllipticCurvePoint@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.curve.canonical_bytes()?;
        output.push(u8::from(self.is_infinity()));
        if let Some((x, y)) = &self.affine {
            append_component(&mut output, &x.canonical_bytes()?)?;
            append_component(&mut output, &y.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &ELLIPTIC_CURVE_POINT_DESCRIPTION
    }
}

impl<R: Field> Add<&EllipticCurvePoint<R>> for &EllipticCurvePoint<R> {
    type Output = Result<EllipticCurvePoint<R>, EllipticCurveError>;

    fn add(self, rhs: &EllipticCurvePoint<R>) -> Self::Output {
        self.checked_add(rhs)
    }
}

impl<R: Field> Sub<&EllipticCurvePoint<R>> for &EllipticCurvePoint<R> {
    type Output = Result<EllipticCurvePoint<R>, EllipticCurveError>;

    fn sub(self, rhs: &EllipticCurvePoint<R>) -> Self::Output {
        self.checked_add(&rhs.negated()?)
    }
}

impl<R: Field> Neg for &EllipticCurvePoint<R> {
    type Output = Result<EllipticCurvePoint<R>, EllipticCurveError>;

    fn neg(self) -> Self::Output {
        self.negated()
    }
}

/// A curve function together with its exact Miller divisor formula.
#[derive(Clone, PartialEq, Eq)]
pub struct MillerFunction<R: Field> {
    curve: EllipticCurve<R>,
    point: EllipticCurvePoint<R>,
    multiplier: Integer,
    terminal_point: EllipticCurvePoint<R>,
    function: CurveFunction<R>,
}

impl<R: Field> MillerFunction<R> {
    /// Owning elliptic curve.
    pub fn curve(&self) -> &EllipticCurve<R> {
        &self.curve
    }

    /// Source point `P`.
    pub fn point(&self) -> &EllipticCurvePoint<R> {
        &self.point
    }

    /// Positive Miller multiplier `n`.
    pub fn multiplier(&self) -> &Integer {
        &self.multiplier
    }

    /// Terminal group-law point `nP`.
    pub fn terminal_point(&self) -> &EllipticCurvePoint<R> {
        &self.terminal_point
    }

    /// Canonical function in the underlying curve function field.
    pub fn as_curve_function(&self) -> &CurveFunction<R> {
        &self.function
    }

    /// Exact formula `n(P) - (nP) - (n-1)(O)` as a Weil divisor.
    pub fn divisor_formula(&self) -> Result<WeilDivisor<R>, EllipticCurveError> {
        let point = self.point.underlying_curve_point().as_closed_point()?;
        let terminal = self
            .terminal_point
            .underlying_curve_point()
            .as_closed_point()?;
        let origin = self
            .curve
            .origin()?
            .underlying_curve_point()
            .as_closed_point()?;
        Ok(WeilDivisor::new(
            self.curve.underlying_curve().clone(),
            [
                (point, self.multiplier.clone()),
                (terminal, Integer::from(-1_i8)),
                (origin, -(&self.multiplier - &Integer::one())),
            ],
        )?)
    }

    /// Recompute the complete double-and-add function and divisor formula.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.curve.verify()?;
        self.point.verify()?;
        self.terminal_point.verify()?;
        self.function.verify()?;
        let rebuilt = self
            .curve
            .miller_function(&self.point, &self.multiplier)
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "Miller double-and-add function recomputes",
            })?;
        let divisor = self
            .divisor_formula()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "Miller divisor formula is defined on rational support",
            })?;
        divisor.verify()?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "Miller function and terminal point are canonical",
            })
        }
    }

    /// Serialize source data; the function is recomputed on restoration.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        encode_json(
            Self::TYPE_ID,
            [
                ("point", nested_json(self.point.to_json()?)?),
                (
                    "multiplier",
                    serde_json::to_value(&self.multiplier).map_err(json_error)?,
                ),
            ],
        )
    }

    /// Restore the source point and rerun Miller's algorithm.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: DeserializeOwned,
        R::Element: DeserializeOwned,
    {
        let mut object = decode_json(value, Self::TYPE_ID, &["point", "multiplier"])?;
        let point = EllipticCurvePoint::<R>::from_json(&nested_json_string(
            object
                .remove("point")
                .ok_or_else(|| missing_json_field("point"))?,
        )?)?;
        let multiplier = decode_json_field(&mut object, "multiplier")?;
        point
            .curve()
            .miller_function(&point, &multiplier)
            .map_err(elliptic_json_error)
    }
}

impl MillerFunction<RationalField> {
    /// Verify the divisor formula independently by complete factorization over `Q`.
    pub fn verify_principal_divisor(&self) -> Result<(), VerificationError> {
        let divisor = self
            .divisor_formula()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "Miller divisor formula is defined over Q",
            })?;
        self.function.verify_principal_divisor(&divisor)
    }
}

impl MillerFunction<FiniteField> {
    /// Verify the divisor formula independently by finite-field factorization.
    pub fn verify_principal_divisor(&self) -> Result<(), VerificationError> {
        let divisor = self
            .divisor_formula()
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "Miller divisor formula is defined over the finite field",
            })?;
        self.function.verify_principal_divisor(&divisor)
    }
}

impl<R: Field> fmt::Debug for MillerFunction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MillerFunction")
            .field("base", &R::TYPE_ID)
            .field("multiplier", &self.multiplier)
            .field("terminal_is_infinity", &self.terminal_point.is_infinity())
            .finish()
    }
}

impl<R: Field> fmt::Display for MillerFunction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "MillerFunction<{}>(multiplier={})",
            R::TYPE_ID,
            self.multiplier
        )
    }
}

impl<R: Field> SemanticType for MillerFunction<R> {
    const TYPE_ID: &'static str = "noether/numthy/MillerFunction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.point.canonical_bytes()?;
        append_component(&mut output, &self.multiplier.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &MILLER_FUNCTION_DESCRIPTION
    }
}

/// Failure in an effective multiplicative `K*/K*^n` computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowerClassError {
    /// The exponent was smaller than two.
    InvalidExponent,
    /// Zero was supplied even though only the multiplicative group is represented.
    Zero,
    /// A value belonged to another exact field context.
    ContextMismatch(ContextMismatchError),
    /// Complete integer factorization failed.
    IntegerFactorization(FactorError),
    /// Complete rational-polynomial factorization failed.
    PolynomialFactorization(PolynomialFactorizationError),
    /// Rational-function arithmetic failed while constructing a representative.
    RationalFunction(RationalFunctionError),
    /// A canonical representative would exceed a declared materialization bound.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed limit.
        limit: u64,
    },
    /// Revalidation found inconsistent canonical factor data.
    InvalidWitness(&'static str),
}

impl fmt::Display for PowerClassError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExponent => formatter.write_str("an nth-power class needs n >= 2"),
            Self::Zero => formatter.write_str("zero has no class in a multiplicative group"),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::IntegerFactorization(error) => error.fmt(formatter),
            Self::PolynomialFactorization(error) => error.fmt(formatter),
            Self::RationalFunction(error) => error.fmt(formatter),
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "power-class computation exceeded its {resource} limit of {limit}"
            ),
            Self::InvalidWitness(claim) => {
                write!(formatter, "invalid power-class witness: {claim}")
            }
        }
    }
}

impl std::error::Error for PowerClassError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::IntegerFactorization(error) => Some(error),
            Self::PolynomialFactorization(error) => Some(error),
            Self::RationalFunction(error) => Some(error),
            Self::InvalidExponent
            | Self::Zero
            | Self::ResourceLimit { .. }
            | Self::InvalidWitness(_) => None,
        }
    }
}

impl From<ContextMismatchError> for PowerClassError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<FactorError> for PowerClassError {
    fn from(value: FactorError) -> Self {
        Self::IntegerFactorization(value)
    }
}

impl From<PolynomialFactorizationError> for PowerClassError {
    fn from(value: PolynomialFactorizationError) -> Self {
        Self::PolynomialFactorization(value)
    }
}

impl From<RationalFunctionError> for PowerClassError {
    fn from(value: RationalFunctionError) -> Self {
        Self::RationalFunction(value)
    }
}

#[derive(Clone, PartialEq, Eq)]
enum PowerClassData {
    Rational {
        negative: bool,
        factors: Vec<(Integer, Integer)>,
    },
    RationalFunctions {
        negative: bool,
        rational_factors: Vec<(Integer, Integer)>,
        polynomial_factors: Vec<(Polynomial<RationalField>, Integer)>,
    },
}

/// An effective context for the multiplicative quotient `K*/K*^n`.
///
/// Constructors exist only for fields with a native, verified complete
/// factorization route: initially `Q` and `Q(t)`.
#[derive(Clone, PartialEq, Eq)]
pub struct NthPowerClassGroup<R: Field> {
    field: R,
    exponent: Integer,
}

impl<R: Field> NthPowerClassGroup<R> {
    /// Exact source field.
    pub fn field(&self) -> &R {
        &self.field
    }

    /// Quotient exponent `n`.
    pub fn exponent(&self) -> &Integer {
        &self.exponent
    }

    /// Recheck the admitted field context and exponent.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.exponent >= Integer::from(2_u8) {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "an nth-power class group has exponent at least two",
            })
        }
    }
}

impl NthPowerClassGroup<RationalField> {
    /// Construct `Q*/Q*^n`.
    pub fn rational(exponent: Integer) -> Result<Self, PowerClassError> {
        power_group(Q, exponent)
    }

    /// Reduce a nonzero rational by its complete prime valuations modulo `n`.
    pub fn class(&self, value: Rational) -> Result<NthPowerClass<RationalField>, PowerClassError> {
        if value.is_zero() {
            return Err(PowerClassError::Zero);
        }
        let (negative, factors) = reduce_rational(&value, &self.exponent)?;
        Ok(NthPowerClass {
            group: self.clone(),
            data: PowerClassData::Rational { negative, factors },
        })
    }

    /// Trivial class.
    pub fn identity(&self) -> NthPowerClass<RationalField> {
        NthPowerClass {
            group: self.clone(),
            data: PowerClassData::Rational {
                negative: false,
                factors: Vec::new(),
            },
        }
    }

    /// Canonical typed JSON for this effective quotient context.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        power_group_to_json(self)
    }

    /// Restore `Q*/Q*^n` and recheck the exponent.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        power_group_from_json(value, power_group)
    }
}

impl NthPowerClassGroup<RationalFunctionField<RationalField>> {
    /// Construct `Q(t)*/Q(t)*^n` in one explicit rational-function context.
    pub fn rational_functions(
        field: RationalFunctionField<RationalField>,
        exponent: Integer,
    ) -> Result<Self, PowerClassError> {
        power_group(field, exponent)
    }

    /// Reduce constant and irreducible-polynomial valuations modulo `n`.
    pub fn class(
        &self,
        value: RationalFunction<RationalField>,
    ) -> Result<NthPowerClass<RationalFunctionField<RationalField>>, PowerClassError> {
        if value.field() != &self.field {
            return Err(ContextMismatchError {
                expected: RationalFunctionField::<RationalField>::TYPE_ID.to_owned(),
                actual: "rational function belongs to another field context".to_owned(),
            }
            .into());
        }
        if value.is_zero() {
            return Err(PowerClassError::Zero);
        }
        let numerator = value.numerator().factor()?;
        let denominator = value.denominator().factor()?;
        numerator.verify()?;
        denominator.verify()?;
        let constant = numerator
            .coefficient()
            .checked_div(denominator.coefficient())
            .map_err(|_| PowerClassError::InvalidWitness("factorization coefficient is nonzero"))?;
        let (negative, rational_factors) = reduce_rational(&constant, &self.exponent)?;
        let polynomial_factors = reduce_polynomial_valuations(
            numerator.factors(),
            denominator.factors(),
            &self.exponent,
        )?;
        Ok(NthPowerClass {
            group: self.clone(),
            data: PowerClassData::RationalFunctions {
                negative,
                rational_factors,
                polynomial_factors,
            },
        })
    }

    /// Trivial class.
    pub fn identity(&self) -> NthPowerClass<RationalFunctionField<RationalField>> {
        NthPowerClass {
            group: self.clone(),
            data: PowerClassData::RationalFunctions {
                negative: false,
                rational_factors: Vec::new(),
                polynomial_factors: Vec::new(),
            },
        }
    }

    /// Canonical typed JSON for this effective quotient context.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        power_group_to_json(self)
    }

    /// Restore the exact `Q(t)` presentation and exponent.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        power_group_from_json(value, power_group)
    }
}

impl<R: Field> fmt::Debug for NthPowerClassGroup<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NthPowerClassGroup")
            .field("field", &R::TYPE_ID)
            .field("exponent", &self.exponent)
            .finish()
    }
}

impl<R: Field> fmt::Display for NthPowerClassGroup<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "NthPowerClassGroup<{}>(n={})",
            R::TYPE_ID,
            self.exponent
        )
    }
}

impl<R: Field> SemanticType for NthPowerClassGroup<R> {
    const TYPE_ID: &'static str = "noether/numthy/NthPowerClassGroup@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.field.canonical_bytes()?;
        append_component(&mut output, &self.exponent.canonical_bytes()?)?;
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &NTH_POWER_CLASS_GROUP_DESCRIPTION
    }
}

/// One canonical multiplicative class with factor valuations in `[0,n)`.
#[derive(Clone, PartialEq, Eq)]
pub struct NthPowerClass<R: Field> {
    group: NthPowerClassGroup<R>,
    data: PowerClassData,
}

impl<R: Field> NthPowerClass<R> {
    /// Owning quotient context.
    pub fn group(&self) -> &NthPowerClassGroup<R> {
        &self.group
    }

    /// Whether this is the trivial class.
    pub fn is_identity(&self) -> bool {
        match &self.data {
            PowerClassData::Rational { negative, factors } => !negative && factors.is_empty(),
            PowerClassData::RationalFunctions {
                negative,
                rational_factors,
                polynomial_factors,
            } => !negative && rational_factors.is_empty() && polynomial_factors.is_empty(),
        }
    }

    /// Recheck reduced exponent ranges and factor ordering.
    pub fn verify(&self) -> Result<(), VerificationError> {
        self.group.verify()?;
        let valid_integer_factors = |factors: &[(Integer, Integer)]| {
            factors.iter().all(|(prime, exponent)| {
                prime.is_prime() && exponent.is_positive() && exponent < &self.group.exponent
            }) && factors.windows(2).all(|pair| pair[0].0 < pair[1].0)
        };
        let valid = match &self.data {
            PowerClassData::Rational { factors, .. } => valid_integer_factors(factors),
            PowerClassData::RationalFunctions {
                rational_factors,
                polynomial_factors,
                ..
            } => {
                valid_integer_factors(rational_factors)
                    && polynomial_factors.iter().all(|(_, exponent)| {
                        exponent.is_positive() && exponent < &self.group.exponent
                    })
                    && polynomial_factors
                        .iter()
                        .all(|(factor, _)| match factor.factor() {
                            Ok(factorization) => {
                                factorization.verify().is_ok()
                                    && factorization.coefficient().is_one()
                                    && factorization.factors().len() == 1
                                    && factorization.factors()[0].0 == *factor
                                    && factorization.factors()[0].1 == 1
                            }
                            Err(_) => false,
                        })
                    && polynomial_factors.windows(2).all(|pair| {
                        match (pair[0].0.canonical_bytes(), pair[1].0.canonical_bytes()) {
                            (Ok(left), Ok(right)) => left < right,
                            _ => false,
                        }
                    })
            }
        };
        if valid {
            Ok(())
        } else {
            Err(VerificationError::InvalidWitness {
                claim: "power-class prime and polynomial valuations are canonical modulo n",
            })
        }
    }
}

impl NthPowerClass<RationalField> {
    /// Multiply and reduce prime valuations modulo `n`.
    pub fn checked_mul(&self, other: &Self) -> Result<Self, PowerClassError> {
        require_same_power_group(&self.group, &other.group)?;
        let (left_negative, left_factors) = rational_data(&self.data)?;
        let (right_negative, right_factors) = rational_data(&other.data)?;
        Ok(Self {
            group: self.group.clone(),
            data: PowerClassData::Rational {
                negative: left_negative ^ right_negative,
                factors: combine_integer_factors(left_factors, right_factors, &self.group.exponent),
            },
        })
    }

    /// Multiplicative inverse.
    pub fn inverse(&self) -> Result<Self, PowerClassError> {
        let (negative, factors) = rational_data(&self.data)?;
        Ok(Self {
            group: self.group.clone(),
            data: PowerClassData::Rational {
                negative,
                factors: scale_integer_factors(
                    factors,
                    &Integer::from(-1_i8),
                    &self.group.exponent,
                ),
            },
        })
    }

    /// Raise the class to an arbitrary-precision signed integer.
    pub fn pow(&self, exponent: &Integer) -> Result<Self, PowerClassError> {
        let (negative, factors) = rational_data(&self.data)?;
        let residue = integer_residue(exponent, &self.group.exponent);
        Ok(Self {
            group: self.group.clone(),
            data: PowerClassData::Rational {
                negative: negative && residue.mod_u(2) == 1,
                factors: scale_integer_factors(factors, &residue, &self.group.exponent),
            },
        })
    }

    /// Materialize a deterministic rational representative.
    pub fn representative(&self) -> Result<Rational, PowerClassError> {
        let (negative, factors) = rational_data(&self.data)?;
        let mut numerator = Integer::one();
        for (prime, exponent) in factors {
            let exponent = check_materialized_exponent(exponent)?;
            numerator = &numerator * &prime.pow(exponent);
        }
        if negative {
            numerator = -numerator;
        }
        Ok(Rational::from(numerator))
    }

    /// Serialize a canonical representative; restoration refactors it.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        power_class_to_json(self, self.representative().map_err(power_json_error)?)
    }

    /// Restore and recompute all rational prime valuations.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        power_class_from_json(
            value,
            power_group,
            |group: &NthPowerClassGroup<RationalField>, value| group.class(value),
        )
    }
}

impl NthPowerClass<RationalFunctionField<RationalField>> {
    /// Multiply and reduce constant and polynomial valuations modulo `n`.
    pub fn checked_mul(&self, other: &Self) -> Result<Self, PowerClassError> {
        require_same_power_group(&self.group, &other.group)?;
        let (ln, li, lp) = rational_function_data(&self.data)?;
        let (rn, ri, rp) = rational_function_data(&other.data)?;
        Ok(Self {
            group: self.group.clone(),
            data: PowerClassData::RationalFunctions {
                negative: ln ^ rn,
                rational_factors: combine_integer_factors(li, ri, &self.group.exponent),
                polynomial_factors: combine_polynomial_factors(lp, rp, &self.group.exponent)?,
            },
        })
    }

    /// Multiplicative inverse.
    pub fn inverse(&self) -> Result<Self, PowerClassError> {
        let (negative, rational_factors, polynomial_factors) = rational_function_data(&self.data)?;
        Ok(Self {
            group: self.group.clone(),
            data: PowerClassData::RationalFunctions {
                negative,
                rational_factors: scale_integer_factors(
                    rational_factors,
                    &Integer::from(-1_i8),
                    &self.group.exponent,
                ),
                polynomial_factors: scale_polynomial_factors(
                    polynomial_factors,
                    &Integer::from(-1_i8),
                    &self.group.exponent,
                ),
            },
        })
    }

    /// Raise the class to an arbitrary-precision signed integer.
    pub fn pow(&self, exponent: &Integer) -> Result<Self, PowerClassError> {
        let (negative, rational_factors, polynomial_factors) = rational_function_data(&self.data)?;
        let residue = integer_residue(exponent, &self.group.exponent);
        Ok(Self {
            group: self.group.clone(),
            data: PowerClassData::RationalFunctions {
                negative: negative && residue.mod_u(2) == 1,
                rational_factors: scale_integer_factors(
                    rational_factors,
                    &residue,
                    &self.group.exponent,
                ),
                polynomial_factors: scale_polynomial_factors(
                    polynomial_factors,
                    &residue,
                    &self.group.exponent,
                ),
            },
        })
    }

    /// Materialize a deterministic rational-function representative.
    pub fn representative(&self) -> Result<RationalFunction<RationalField>, PowerClassError> {
        let (negative, rational_factors, polynomial_factors) = rational_function_data(&self.data)?;
        let mut constant = Integer::one();
        for (prime, exponent) in rational_factors {
            let exponent = check_materialized_exponent(exponent)?;
            constant = &constant * &prime.pow(exponent);
        }
        if negative {
            constant = -constant;
        }
        let ring = self.group.field.polynomial_ring();
        let mut polynomial = ring
            .constant(Rational::from(constant))
            .map_err(|_| PowerClassError::InvalidWitness("power-class constant belongs to Q[t]"))?;
        for (factor, exponent) in polynomial_factors {
            let exponent = check_materialized_exponent(exponent)?;
            polynomial = (&polynomial
                * &factor.pow(exponent).map_err(|_| {
                    PowerClassError::InvalidWitness("power-class polynomial power is defined")
                })?)
                .map_err(|_| {
                    PowerClassError::InvalidWitness("power-class polynomial product is defined")
                })?;
        }
        self.group
            .field
            .from_polynomial(polynomial)
            .map_err(PowerClassError::from)
    }

    /// Serialize a canonical representative; restoration refactors it.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        power_class_to_json(self, self.representative().map_err(power_json_error)?)
    }

    /// Restore and recompute all constant and polynomial valuations.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        power_class_from_json(
            value,
            power_group,
            |group: &NthPowerClassGroup<RationalFunctionField<RationalField>>, value| {
                group.class(value)
            },
        )
    }
}

impl<R: Field> fmt::Debug for NthPowerClass<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NthPowerClass")
            .field("field", &R::TYPE_ID)
            .field("exponent", &self.group.exponent)
            .field("identity", &self.is_identity())
            .finish()
    }
}

impl<R: Field> fmt::Display for NthPowerClass<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "NthPowerClass<{}>(n={}, identity={})",
            R::TYPE_ID,
            self.group.exponent,
            self.is_identity()
        )
    }
}

impl<R: Field> SemanticType for NthPowerClass<R> {
    const TYPE_ID: &'static str = "noether/numthy/NthPowerClass@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.group.canonical_bytes()?;
        match &self.data {
            PowerClassData::Rational { negative, factors } => {
                output.extend_from_slice(b"Q");
                output.push(u8::from(*negative));
                append_integer_factor_bytes(&mut output, factors)?;
            }
            PowerClassData::RationalFunctions {
                negative,
                rational_factors,
                polynomial_factors,
            } => {
                output.extend_from_slice(b"Q(t)");
                output.push(u8::from(*negative));
                append_integer_factor_bytes(&mut output, rational_factors)?;
                for (factor, exponent) in polynomial_factors {
                    append_component(&mut output, &factor.canonical_bytes()?)?;
                    append_component(&mut output, &exponent.canonical_bytes()?)?;
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &NTH_POWER_CLASS_DESCRIPTION
    }
}

impl Mul<&NthPowerClass<RationalField>> for &NthPowerClass<RationalField> {
    type Output = Result<NthPowerClass<RationalField>, PowerClassError>;

    fn mul(self, rhs: &NthPowerClass<RationalField>) -> Self::Output {
        self.checked_mul(rhs)
    }
}

impl Mul<&NthPowerClass<RationalFunctionField<RationalField>>>
    for &NthPowerClass<RationalFunctionField<RationalField>>
{
    type Output = Result<NthPowerClass<RationalFunctionField<RationalField>>, PowerClassError>;

    fn mul(self, rhs: &NthPowerClass<RationalFunctionField<RationalField>>) -> Self::Output {
        self.checked_mul(rhs)
    }
}

fn finite_field_word<E: 'static>(value: &E) -> Option<u64> {
    (value as &dyn Any)
        .downcast_ref::<FiniteFieldElement>()?
        .representative()
        .to_u64_checked()
}

fn natural_multiple<R: Field>(
    field: &R,
    value: &R::Element,
    mut factor: u64,
) -> Result<R::Element, ContextMismatchError> {
    let mut result = None;
    let mut summand = value.clone();
    while factor > 0 {
        if factor & 1 == 1 {
            result = Some(match result {
                Some(current) => field.add(&current, &summand)?,
                None => summand.clone(),
            });
        }
        factor >>= 1;
        if factor > 0 {
            summand = field.add(&summand, &summand)?;
        }
    }
    Ok(result.unwrap_or_else(|| field.zero()))
}

fn power_group<R: Field>(
    field: R,
    exponent: Integer,
) -> Result<NthPowerClassGroup<R>, PowerClassError> {
    if exponent < Integer::from(2_u8) {
        return Err(PowerClassError::InvalidExponent);
    }
    Ok(NthPowerClassGroup { field, exponent })
}

fn reduce_rational(
    value: &Rational,
    exponent: &Integer,
) -> Result<(bool, Vec<(Integer, Integer)>), PowerClassError> {
    if value.is_zero() {
        return Err(PowerClassError::Zero);
    }
    let numerator = value.numerator();
    let denominator = value.denominator();
    let numerator_factorization = factor(&numerator.abs())?;
    let denominator_factorization = factor(&denominator)?;
    let mut valuations = Vec::<(Integer, Integer)>::new();
    for (prime, multiplicity) in numerator_factorization.factors() {
        valuations.push((prime.clone(), Integer::from(*multiplicity)));
    }
    for (prime, multiplicity) in denominator_factorization.factors() {
        if let Some((_, valuation)) = valuations.iter_mut().find(|(known, _)| known == prime) {
            *valuation -= &Integer::from(*multiplicity);
        } else {
            valuations.push((prime.clone(), -Integer::from(*multiplicity)));
        }
    }
    valuations.sort_by(|left, right| left.0.cmp(&right.0));
    let factors = valuations
        .into_iter()
        .filter_map(|(prime, valuation)| {
            let residue = integer_residue(&valuation, exponent);
            (!residue.is_zero()).then_some((prime, residue))
        })
        .collect();
    Ok((value.is_negative() && exponent.mod_u(2) == 0, factors))
}

fn reduce_polynomial_valuations(
    numerator: &[(Polynomial<RationalField>, u64)],
    denominator: &[(Polynomial<RationalField>, u64)],
    exponent: &Integer,
) -> Result<Vec<(Polynomial<RationalField>, Integer)>, PowerClassError> {
    let mut valuations = numerator
        .iter()
        .map(|(factor, multiplicity)| (factor.clone(), Integer::from(*multiplicity)))
        .collect::<Vec<_>>();
    for (factor, multiplicity) in denominator {
        if let Some((_, valuation)) = valuations.iter_mut().find(|(known, _)| known == factor) {
            *valuation -= &Integer::from(*multiplicity);
        } else {
            valuations.push((factor.clone(), -Integer::from(*multiplicity)));
        }
    }
    let output = valuations
        .into_iter()
        .filter_map(|(factor, valuation)| {
            let residue = integer_residue(&valuation, exponent);
            (!residue.is_zero()).then_some((factor, residue))
        })
        .collect::<Vec<_>>();
    let mut keyed = output
        .into_iter()
        .map(|(factor, residue)| {
            factor
                .canonical_bytes()
                .map(|key| (key, factor, residue))
                .map_err(|_| {
                    PowerClassError::InvalidWitness(
                        "irreducible polynomial has canonical semantic bytes",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    let output = keyed
        .into_iter()
        .map(|(_, factor, residue)| (factor, residue))
        .collect::<Vec<_>>();
    if output.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(PowerClassError::InvalidWitness(
            "factorization returns unique irreducible factors",
        ));
    }
    Ok(output)
}

fn combine_integer_factors(
    left: &[(Integer, Integer)],
    right: &[(Integer, Integer)],
    exponent: &Integer,
) -> Vec<(Integer, Integer)> {
    let mut values = left.to_vec();
    for (prime, multiplicity) in right {
        if let Some((_, known)) = values.iter_mut().find(|(known, _)| known == prime) {
            *known = add_residues(known, multiplicity, exponent);
        } else {
            values.push((prime.clone(), multiplicity.clone()));
        }
    }
    values.retain(|(_, multiplicity)| !multiplicity.is_zero());
    values.sort_by(|left, right| left.0.cmp(&right.0));
    values
}

fn combine_polynomial_factors(
    left: &[(Polynomial<RationalField>, Integer)],
    right: &[(Polynomial<RationalField>, Integer)],
    exponent: &Integer,
) -> Result<Vec<(Polynomial<RationalField>, Integer)>, PowerClassError> {
    let mut values = left.to_vec();
    for (factor, multiplicity) in right {
        if let Some((_, known)) = values.iter_mut().find(|(known, _)| known == factor) {
            *known = add_residues(known, multiplicity, exponent);
        } else {
            values.push((factor.clone(), multiplicity.clone()));
        }
    }
    values.retain(|(_, multiplicity)| !multiplicity.is_zero());
    let mut keyed = values
        .into_iter()
        .map(|(factor, residue)| {
            factor
                .canonical_bytes()
                .map(|key| (key, factor, residue))
                .map_err(|_| {
                    PowerClassError::InvalidWitness(
                        "irreducible polynomial has canonical semantic bytes",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(keyed
        .into_iter()
        .map(|(_, factor, residue)| (factor, residue))
        .collect())
}

fn scale_integer_factors(
    factors: &[(Integer, Integer)],
    scalar: &Integer,
    exponent: &Integer,
) -> Vec<(Integer, Integer)> {
    factors
        .iter()
        .filter_map(|(prime, multiplicity)| {
            let residue = integer_residue(&(multiplicity * scalar), exponent);
            (!residue.is_zero()).then_some((prime.clone(), residue))
        })
        .collect()
}

fn scale_polynomial_factors(
    factors: &[(Polynomial<RationalField>, Integer)],
    scalar: &Integer,
    exponent: &Integer,
) -> Vec<(Polynomial<RationalField>, Integer)> {
    factors
        .iter()
        .filter_map(|(factor, multiplicity)| {
            let residue = integer_residue(&(multiplicity * scalar), exponent);
            (!residue.is_zero()).then_some((factor.clone(), residue))
        })
        .collect()
}

fn add_residues(left: &Integer, right: &Integer, modulus: &Integer) -> Integer {
    integer_residue(&(left + right), modulus)
}

fn integer_residue(value: &Integer, modulus: &Integer) -> Integer {
    let mut residue = value % modulus;
    if residue.is_negative() {
        residue += modulus;
    }
    residue
}

type RationalClassData<'a> = (bool, &'a [(Integer, Integer)]);

fn rational_data(data: &PowerClassData) -> Result<RationalClassData<'_>, PowerClassError> {
    match data {
        PowerClassData::Rational { negative, factors } => Ok((*negative, factors)),
        PowerClassData::RationalFunctions { .. } => Err(PowerClassError::InvalidWitness(
            "rational class stores rational factor data",
        )),
    }
}

type RationalFunctionClassData<'a> = (
    bool,
    &'a [(Integer, Integer)],
    &'a [(Polynomial<RationalField>, Integer)],
);

fn rational_function_data(
    data: &PowerClassData,
) -> Result<RationalFunctionClassData<'_>, PowerClassError> {
    match data {
        PowerClassData::RationalFunctions {
            negative,
            rational_factors,
            polynomial_factors,
        } => Ok((*negative, rational_factors, polynomial_factors)),
        PowerClassData::Rational { .. } => Err(PowerClassError::InvalidWitness(
            "rational-function class stores polynomial factor data",
        )),
    }
}

fn require_same_power_group<R: Field>(
    left: &NthPowerClassGroup<R>,
    right: &NthPowerClassGroup<R>,
) -> Result<(), PowerClassError> {
    if left == right {
        Ok(())
    } else {
        Err(ContextMismatchError {
            expected: NthPowerClassGroup::<R>::TYPE_ID.to_owned(),
            actual: "power classes belong to unequal quotient contexts".to_owned(),
        }
        .into())
    }
}

fn check_materialized_exponent(exponent: &Integer) -> Result<u64, PowerClassError> {
    const LIMIT: u64 = 1_000_000;
    if exponent <= &Integer::from(LIMIT) {
        exponent
            .to_u64_checked()
            .ok_or(PowerClassError::InvalidWitness(
                "a nonnegative bounded exponent converts to u64",
            ))
    } else {
        Err(PowerClassError::ResourceLimit {
            resource: "representative factor exponent",
            limit: LIMIT,
        })
    }
}

fn power_group_to_json<R>(group: &NthPowerClassGroup<R>) -> Result<String, NoetherError>
where
    R: Field + Serialize,
{
    encode_json(
        NthPowerClassGroup::<R>::TYPE_ID,
        [
            (
                "field",
                serde_json::to_value(&group.field).map_err(json_error)?,
            ),
            (
                "exponent",
                serde_json::to_value(&group.exponent).map_err(json_error)?,
            ),
        ],
    )
}

fn power_group_from_json<R, F>(
    value: &str,
    construct: F,
) -> Result<NthPowerClassGroup<R>, NoetherError>
where
    R: Field + DeserializeOwned,
    F: FnOnce(R, Integer) -> Result<NthPowerClassGroup<R>, PowerClassError>,
{
    let mut object = decode_json(
        value,
        NthPowerClassGroup::<R>::TYPE_ID,
        &["field", "exponent"],
    )?;
    let field = decode_json_field(&mut object, "field")?;
    let exponent = decode_json_field(&mut object, "exponent")?;
    construct(field, exponent).map_err(power_json_error)
}

fn power_class_to_json<R, V>(
    value: &NthPowerClass<R>,
    representative: V,
) -> Result<String, NoetherError>
where
    R: Field + Serialize,
    V: Serialize,
{
    encode_json(
        NthPowerClass::<R>::TYPE_ID,
        [
            (
                "field",
                serde_json::to_value(&value.group.field).map_err(json_error)?,
            ),
            (
                "exponent",
                serde_json::to_value(&value.group.exponent).map_err(json_error)?,
            ),
            (
                "representative",
                serde_json::to_value(representative).map_err(json_error)?,
            ),
        ],
    )
}

fn power_class_from_json<R, V, G, C>(
    value: &str,
    construct_group: G,
    construct_class: C,
) -> Result<NthPowerClass<R>, NoetherError>
where
    R: Field + DeserializeOwned,
    V: DeserializeOwned,
    G: FnOnce(R, Integer) -> Result<NthPowerClassGroup<R>, PowerClassError>,
    C: FnOnce(&NthPowerClassGroup<R>, V) -> Result<NthPowerClass<R>, PowerClassError>,
{
    let mut object = decode_json(
        value,
        NthPowerClass::<R>::TYPE_ID,
        &["field", "exponent", "representative"],
    )?;
    let field = decode_json_field(&mut object, "field")?;
    let exponent = decode_json_field(&mut object, "exponent")?;
    let representative = decode_json_field(&mut object, "representative")?;
    let group = construct_group(field, exponent).map_err(power_json_error)?;
    construct_class(&group, representative).map_err(power_json_error)
}

fn append_integer_factor_bytes(
    output: &mut Vec<u8>,
    factors: &[(Integer, Integer)],
) -> Result<(), NoetherError> {
    for (prime, exponent) in factors {
        append_component(output, &prime.canonical_bytes()?)?;
        append_component(output, &exponent.canonical_bytes()?)?;
    }
    Ok(())
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(component.len())
        .map_err(|_| NoetherError::Serialization("semantic component is too large".to_owned()))?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(component);
    Ok(())
}

fn encode_json<const N: usize>(
    type_id: &'static str,
    fields: [(&'static str, serde_json::Value); N],
) -> Result<String, NoetherError> {
    let mut object = serde_json::Map::new();
    object.insert(
        "type".to_owned(),
        serde_json::Value::String(type_id.to_owned()),
    );
    for (name, value) in fields {
        object.insert(name.to_owned(), value);
    }
    serde_json::to_string(&object).map_err(json_error)
}

fn decode_json(
    value: &str,
    type_id: &'static str,
    fields: &[&str],
) -> Result<serde_json::Map<String, serde_json::Value>, NoetherError> {
    let mut object = serde_json::from_str::<serde_json::Value>(value)
        .map_err(json_error)?
        .as_object()
        .cloned()
        .ok_or_else(|| NoetherError::Serialization("typed JSON must be an object".to_owned()))?;
    let actual = object
        .remove("type")
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| NoetherError::Serialization("typed JSON needs a string type".to_owned()))?;
    if actual != type_id {
        return Err(NoetherError::Serialization(format!(
            "expected typed JSON {type_id}, got {actual}"
        )));
    }
    let expected = fields
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let actual = object
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    if actual != expected {
        return Err(NoetherError::Serialization(
            "typed JSON has missing or unknown fields".to_owned(),
        ));
    }
    Ok(object)
}

fn decode_json_field<T: DeserializeOwned>(
    object: &mut serde_json::Map<String, serde_json::Value>,
    field: &'static str,
) -> Result<T, NoetherError> {
    serde_json::from_value(
        object
            .remove(field)
            .ok_or_else(|| missing_json_field(field))?,
    )
    .map_err(json_error)
}

fn nested_json(value: String) -> Result<serde_json::Value, NoetherError> {
    serde_json::from_str(&value).map_err(json_error)
}

fn nested_json_string(value: serde_json::Value) -> Result<String, NoetherError> {
    serde_json::to_string(&value).map_err(json_error)
}

fn missing_json_field(field: &'static str) -> NoetherError {
    NoetherError::Serialization(format!("typed JSON is missing {field}"))
}

fn json_error(error: serde_json::Error) -> NoetherError {
    NoetherError::Serialization(error.to_string())
}

fn elliptic_json_error(error: EllipticCurveError) -> NoetherError {
    NoetherError::Serialization(error.to_string())
}

fn power_json_error(error: PowerClassError) -> NoetherError {
    NoetherError::Serialization(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::CurveFunctionEvaluation;

    fn rational(value: i64) -> Rational {
        Rational::from(value)
    }

    #[test]
    fn short_weierstrass_jacobian_scalar_multiplication_matches_affine_floor() {
        let field = FiniteField::prime(101_u64).unwrap();
        let curve = EllipticCurve::new(
            field.clone(),
            [
                field.zero(),
                field.zero(),
                field.zero(),
                field.element(2_u8),
                field.element(3_u8),
            ],
        )
        .unwrap();
        let point = curve
            .point(field.element(3_u8), field.element(6_u8))
            .unwrap();

        for scalar in -40_i64..=40 {
            let scalar = Integer::from(scalar);
            assert_eq!(
                point.scalar_mul(&scalar).unwrap(),
                point.scalar_mul_affine(&scalar).unwrap(),
                "Jacobian and affine scalar multiplication differed at {scalar}"
            );
        }
        assert_eq!(
            curve.origin().unwrap().scalar_mul(&17.into()).unwrap(),
            curve.origin().unwrap()
        );
        assert!(point.has_exact_order(&12.into()).unwrap());
        assert_eq!(point.exact_order.get(), Some(&Integer::from(12_u8)));
        assert!(point.has_exact_order(&12.into()).unwrap());
        assert!(!point.has_exact_order(&24.into()).unwrap());

        // Exercise both the word-sized bit walk and the arbitrary-precision
        // fallback against the independently established order-12 cycle.
        for scalar in [
            Integer::from(u64::MAX),
            Integer::from(u128::from(u64::MAX) + 1),
        ] {
            let reduced = Integer::from(scalar.mod_u(12));
            assert_eq!(
                point.scalar_mul(&scalar).unwrap(),
                point.scalar_mul(&reduced).unwrap()
            );
        }
    }

    #[test]
    fn generalized_weierstrass_projective_model_matches_affine_floor() {
        for (prime, coefficients) in [(11_u64, [1_u64, 0, 1, 0, 0]), (13_u64, [1_u64, 2, 1, 3, 4])]
        {
            let field = FiniteField::prime(prime).unwrap();
            let curve = EllipticCurve::new(
                field.clone(),
                coefficients.map(|coefficient| field.element(coefficient)),
            )
            .unwrap();
            assert!(curve.short_model.is_some());
            let point = (0..prime)
                .flat_map(|x| (0..prime).map(move |y| (x, y)))
                .find_map(|(x, y)| curve.point(field.element(x), field.element(y)).ok())
                .expect("the generalized test curve has an affine point");

            for scalar in -20_i64..=20 {
                let scalar = Integer::from(scalar);
                assert_eq!(
                    point.scalar_mul(&scalar).unwrap(),
                    point.scalar_mul_affine(&scalar).unwrap(),
                    "projective and affine multiplication differed over F_{prime} at {scalar}"
                );
            }
        }
    }

    #[test]
    fn characteristic_three_generalized_model_retains_affine_fallback() {
        let field = FiniteField::prime(3_u8).unwrap();
        let curve = EllipticCurve::new(
            field.clone(),
            [
                field.one(),
                field.zero(),
                field.one(),
                field.zero(),
                field.zero(),
            ],
        )
        .unwrap();
        assert!(curve.short_model.is_none());
        let point = (0..3_u64)
            .flat_map(|x| (0..3_u64).map(move |y| (x, y)))
            .find_map(|(x, y)| curve.point(field.element(x), field.element(y)).ok())
            .expect("the characteristic-three curve has an affine point");
        for scalar in -12_i64..=12 {
            let scalar = Integer::from(scalar);
            assert_eq!(
                point.scalar_mul(&scalar).unwrap(),
                point.scalar_mul_affine(&scalar).unwrap()
            );
        }
    }

    #[test]
    fn characteristic_two_generalized_model_retains_affine_fallback() {
        let field = FiniteField::prime(2_u8).unwrap();
        let curve = EllipticCurve::new(
            field.clone(),
            [
                field.one(),
                field.zero(),
                field.zero(),
                field.zero(),
                field.one(),
            ],
        )
        .unwrap();
        assert!(curve.short_model.is_none());
        let point = curve.point(field.zero(), field.one()).unwrap();
        for scalar in -12_i64..=12 {
            let scalar = Integer::from(scalar);
            assert_eq!(
                point.scalar_mul(&scalar).unwrap(),
                point.scalar_mul_affine(&scalar).unwrap()
            );
        }
    }

    #[test]
    fn rational_generalized_projective_model_matches_affine_floor() {
        let curve = EllipticCurve::new(
            Q,
            [
                rational(1),
                rational(0),
                rational(1),
                rational(-1),
                rational(0),
            ],
        )
        .unwrap();
        assert!(curve.short_model.is_some());
        let point = curve.point(rational(0), rational(0)).unwrap();
        for scalar in -12_i64..=12 {
            let scalar = Integer::from(scalar);
            assert_eq!(
                point.scalar_mul(&scalar).unwrap(),
                point.scalar_mul_affine(&scalar).unwrap()
            );
        }
    }

    #[test]
    fn word_prime_projective_lane_handles_full_u64_products() {
        let field = FiniteField::prime(18_446_744_073_709_551_557_u64).unwrap();
        let curve = EllipticCurve::new(
            field.clone(),
            [
                field.zero(),
                field.zero(),
                field.zero(),
                field.element(2_u8),
                field.element(3_u8),
            ],
        )
        .unwrap();
        let point = curve
            .point(field.element(3_u8), field.element(6_u8))
            .unwrap();
        for scalar in [-17_i64, -2, -1, 0, 1, 2, 17] {
            let scalar = Integer::from(scalar);
            assert_eq!(
                point.scalar_mul(&scalar).unwrap(),
                point.scalar_mul_affine(&scalar).unwrap()
            );
        }
    }

    #[test]
    fn generalized_group_law_and_miller_divisor_are_exact() {
        let curve = EllipticCurve::new(
            Q,
            [
                rational(0),
                rational(0),
                rational(0),
                rational(-1),
                rational(1),
            ],
        )
        .unwrap();
        let point = curve.point(rational(0), rational(1)).unwrap();
        let inverse = point.negated().unwrap();
        assert!(point.checked_add(&inverse).unwrap().is_infinity());
        let doubled = point.checked_add(&point).unwrap();
        assert_eq!(
            doubled.affine_coordinates(),
            Some((
                &Rational::new(Integer::from(1), Integer::from(4)).unwrap(),
                &Rational::new(Integer::from(-7), Integer::from(8)).unwrap(),
            ))
        );
        assert_eq!(point.scalar_mul(&Integer::from(2)).unwrap(), doubled);
        point.verify().unwrap();
        curve.verify().unwrap();
    }

    #[test]
    fn fisher_tate_family_has_order_five_and_matching_miller_function() {
        let coefficients = RationalFunctionField::new(Q, "lambda").unwrap();
        let lambda = coefficients.generator();
        let one = coefficients.one();
        let zero = coefficients.zero();
        let curve = EllipticCurve::new(
            coefficients.clone(),
            [
                (&one - &lambda).unwrap(),
                (-&lambda).unwrap(),
                (-&lambda).unwrap(),
                zero.clone(),
                zero.clone(),
            ],
        )
        .unwrap();
        let point = curve.point(zero.clone(), zero).unwrap();
        point.verify_exact_order(&Integer::from(5)).unwrap();

        let two = point.scalar_mul(&Integer::from(2)).unwrap();
        let three = point.scalar_mul(&Integer::from(3)).unwrap();
        let four = point.scalar_mul(&Integer::from(4)).unwrap();
        assert_eq!(
            two.affine_coordinates(),
            Some((&lambda, &lambda.pow(2).unwrap()))
        );
        assert_eq!(
            three.affine_coordinates(),
            Some((&lambda, &coefficients.zero()))
        );
        assert_eq!(
            four.affine_coordinates(),
            Some((&coefficients.zero(), &lambda))
        );

        let miller = curve.miller_function(&point, &Integer::from(5)).unwrap();
        assert!(miller.terminal_point().is_infinity());
        assert_eq!(miller.divisor_formula().unwrap().degree(), Integer::zero());
        miller.verify().unwrap();

        let point_place = point.underlying_curve_point().as_closed_point().unwrap();
        let origin_place = curve
            .origin()
            .unwrap()
            .underlying_curve_point()
            .as_closed_point()
            .unwrap();
        let divisor = WeilDivisor::new(
            curve.underlying_curve().clone(),
            [
                (origin_place, Integer::from(5)),
                (point_place, Integer::from(-5)),
            ],
        )
        .unwrap();
        let space = divisor.riemann_roch_space().unwrap();
        let quotient = space.basis()[0]
            .checked_div(miller.as_curve_function())
            .unwrap();
        assert!(quotient.as_constant().unwrap().is_some());

        let h = &space.basis()[0];
        let value_at_two = match h.evaluate_at(two.underlying_curve_point()).unwrap() {
            CurveFunctionEvaluation::Regular(value) => value,
            _ => panic!("the Fisher function is regular and nonzero at 2P"),
        };
        let value_at_four = match h.evaluate_at(four.underlying_curve_point()).unwrap() {
            CurveFunctionEvaluation::Regular(value) => value,
            _ => panic!("the Fisher function is regular and nonzero at 4P"),
        };
        let classes =
            NthPowerClassGroup::rational_functions(coefficients, Integer::from(5)).unwrap();
        let normalization = classes
            .class(value_at_two)
            .unwrap()
            .pow(&Integer::from(-2))
            .unwrap()
            .checked_mul(
                &classes
                    .class(value_at_four)
                    .unwrap()
                    .pow(&Integer::one())
                    .unwrap(),
            )
            .unwrap();
        assert!(normalization.is_identity());
        normalization.verify().unwrap();

        assert_eq!(
            EllipticCurve::<RationalFunctionField<RationalField>>::from_json(
                &curve.to_json().unwrap()
            )
            .unwrap(),
            curve
        );
        assert_eq!(
            EllipticCurvePoint::<RationalFunctionField<RationalField>>::from_json(
                &point.to_json().unwrap()
            )
            .unwrap(),
            point
        );
        assert_eq!(
            MillerFunction::<RationalFunctionField<RationalField>>::from_json(
                &miller.to_json().unwrap()
            )
            .unwrap(),
            miller
        );
        assert_eq!(
            NthPowerClass::<RationalFunctionField<RationalField>>::from_json(
                &normalization.to_json().unwrap()
            )
            .unwrap(),
            normalization
        );
    }

    #[test]
    fn rational_power_classes_reduce_sign_and_negative_valuations() {
        let fifth = NthPowerClassGroup::rational(Integer::from(5)).unwrap();
        let value = Rational::new(Integer::from(-8), Integer::from(9)).unwrap();
        let class = fifth.class(value).unwrap();
        assert_eq!(class.representative().unwrap(), Rational::from(216));
        assert_eq!(
            class
                .checked_mul(&class.pow(&Integer::from(-1)).unwrap())
                .unwrap(),
            fifth.identity()
        );

        let squares = NthPowerClassGroup::rational(Integer::from(2)).unwrap();
        assert!(!squares.class(Rational::from(-1)).unwrap().is_identity());
        assert!(fifth.class(Rational::from(-1)).unwrap().is_identity());
        assert_eq!(
            NthPowerClass::<RationalField>::from_json(&class.to_json().unwrap()).unwrap(),
            class
        );

        let huge_exponent = Integer::from(2_u8).pow(80);
        let huge = NthPowerClassGroup::rational(huge_exponent.clone()).unwrap();
        assert_eq!(huge.exponent(), &huge_exponent);
        let two = huge.class(Rational::from(2)).unwrap();
        assert_eq!(two.representative().unwrap(), Rational::from(2));
        assert_eq!(
            NthPowerClassGroup::<RationalField>::from_json(&huge.to_json().unwrap()).unwrap(),
            huge
        );
    }
}
