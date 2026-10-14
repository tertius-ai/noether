// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    ContextMismatchError, DivisionByZeroError, Integer, InversionError, NoetherError, UnitError,
    VerificationError,
};

static BEZOUT_IDENTITY_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/root/BezoutIdentity@1",
    rust_path: "noether::BezoutIdentity",
    python_path: Some("noether.BezoutIdentity"),
};

/// Static discovery metadata for one semantic type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeDescription {
    /// Versioned, language-neutral semantic identity.
    pub semantic_id: &'static str,
    /// Canonical public Rust path.
    pub rust_path: &'static str,
    /// Canonical Python path when this type is projected to Python.
    pub python_path: Option<&'static str>,
}

/// A value with one versioned identity and canonical serialization.
pub trait SemanticType: Clone + Send + Sync + 'static {
    /// Versioned, language-neutral semantic identity.
    const TYPE_ID: &'static str;

    /// Serialize without backend- or language-specific identity.
    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError>;

    /// Return static discovery metadata for the type.
    fn describe() -> &'static TypeDescription;
}

/// A concrete ring context with one associated element type.
pub trait Ring: SemanticType + Eq {
    /// Elements manufactured and validated by this context.
    type Element: SemanticType + Clone + Eq;

    /// Test whether a value belongs to this exact context.
    fn contains(&self, value: &Self::Element) -> bool;
    /// The additive identity in this context.
    fn zero(&self) -> Self::Element;
    /// Test whether a value is the additive identity.
    ///
    /// Contexts with a native zero predicate should override this default to
    /// avoid constructing and comparing a separate zero value.
    fn is_zero(&self, value: &Self::Element) -> bool {
        value == &self.zero()
    }
    /// The multiplicative identity in this context.
    fn one(&self) -> Self::Element;
    /// Add two elements after checking their context.
    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError>;
    /// Subtract two elements after checking their context.
    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError>;
    /// Multiply two elements after checking their context.
    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError>;
    /// Compute a sum of pairwise products in this context.
    ///
    /// The default is the exact ring-law implementation. Contexts with a
    /// reusable accumulator or fused native operation may override it without
    /// changing matrix, vector, or module semantics.
    fn sum_products(
        &self,
        products: &[(&Self::Element, &Self::Element)],
    ) -> Result<Self::Element, ContextMismatchError> {
        let mut sum = self.zero();
        for &(left, right) in products {
            let product = self.mul(left, right)?;
            sum = self.add(&sum, &product)?;
        }
        Ok(sum)
    }
    /// Negate an element after checking its context.
    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError>;
}

/// A ring whose multiplication is commutative.
pub trait CommutativeRing: Ring {}

/// A checked map between two explicit commutative coefficient contexts.
///
/// The source and target are part of the value. Composite objects use this
/// contract to induce coefficientwise maps without implicit coercions or
/// language-boundary loops.
pub trait CoefficientMap {
    /// Source coefficient context.
    type Source: CommutativeRing + fmt::Display;
    /// Target coefficient context.
    type Target: CommutativeRing + fmt::Display;
    /// Scalar application failure.
    type Error: fmt::Display;

    /// Exact source context accepted by this map.
    fn source(&self) -> &Self::Source;

    /// Exact target context produced by this map.
    fn target(&self) -> &Self::Target;

    /// Apply the map after checking that `value` belongs to the exact source.
    fn apply(
        &self,
        value: &<Self::Source as Ring>::Element,
    ) -> Result<<Self::Target as Ring>::Element, Self::Error>;
}

/// A coefficient context with explicit lifted Frobenius endomorphisms.
pub trait CoefficientFrobeniusContext: CommutativeRing + fmt::Display {
    /// Concrete checked Frobenius-map value returned for this context.
    type Frobenius: CoefficientMap<Source = Self, Target = Self>;

    /// Construct the declared nonnegative Frobenius power.
    fn coefficient_frobenius(&self, exponent: u64) -> Self::Frobenius;

    /// Residue characteristic whose power scales monomial exponents.
    fn residue_characteristic(&self) -> &Integer;
}

/// A commutative ring whose units can be recognized and inverted.
///
/// This is weaker than [`Field`]: residue rings, fixed-cap p-adic rings, and
/// the integers all have nonunits while still supporting useful unit
/// inversion. Generic local algebra should require this capability instead of
/// falsely requiring field laws.
pub trait UnitInversion: CommutativeRing {
    /// Invert `value` when it belongs to this context and is a unit.
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError>;
}

/// A commutative ring with no zero divisors.
pub trait IntegralDomain: CommutativeRing {}

/// A commutative ring in which every nonzero value is invertible.
pub trait Field: CommutativeRing + UnitInversion {
    /// Invert a nonzero value in this context.
    fn inverse(&self, value: &Self::Element) -> Result<Self::Element, InversionError>;
}

/// An integral domain equipped with Euclidean division.
pub trait EuclideanDomain: IntegralDomain {
    /// Ordered size used by the Euclidean algorithm.
    type Degree: Ord;

    /// Divide with remainder.
    fn div_rem(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<(Self::Element, Self::Element), DivisionByZeroError>;

    /// Return the Euclidean degree, with `None` for zero.
    fn euclidean_degree(&self, value: &Self::Element) -> Option<Self::Degree>;
}

/// A Bézout witness for two elements of a principal ideal domain.
///
/// For inputs `a` and `b`, the fields satisfy
/// `gcd = coefficient_a * a + coefficient_b * b`. The `gcd` is the
/// context's canonical associate; the coefficients need not be unique.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BezoutIdentity<R: Ring> {
    /// Canonically normalized greatest common divisor.
    pub gcd: R::Element,
    /// Coefficient multiplying the first input, `a`.
    pub coefficient_a: R::Element,
    /// Coefficient multiplying the second input, `b`.
    pub coefficient_b: R::Element,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BezoutIdentityWire {
    #[serde(rename = "type")]
    type_id: String,
    ring: String,
    gcd: String,
    coefficient_a: String,
    coefficient_b: String,
}

impl<R> fmt::Display for BezoutIdentity<R>
where
    R: Ring,
    R::Element: fmt::Display,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "gcd = {}; coefficient_a = {}; coefficient_b = {}",
            self.gcd, self.coefficient_a, self.coefficient_b
        )
    }
}

impl<R> BezoutIdentity<R>
where
    R: Ring,
    R::Element: fmt::Display,
{
    /// Serialize this witness using its semantic and coefficient-context IDs.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&BezoutIdentityWire {
            type_id: Self::TYPE_ID.to_owned(),
            ring: R::TYPE_ID.to_owned(),
            gcd: self.gcd.to_string(),
            coefficient_a: self.coefficient_a.to_string(),
            coefficient_b: self.coefficient_b.to_string(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> BezoutIdentity<R>
where
    R: Ring,
    R::Element: core::str::FromStr,
    <R::Element as core::str::FromStr>::Err: fmt::Display,
{
    /// Restore a witness whose element type has a canonical text parser.
    ///
    /// Deserialization validates the type and coefficient-context identities.
    /// The relation to particular inputs is checked separately by [`Self::verify`].
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<BezoutIdentityWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        if wire.ring != R::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected coefficient context {}, received {}",
                R::TYPE_ID,
                wire.ring
            )));
        }
        let parse = |name: &'static str, component: String| {
            component.parse::<R::Element>().map_err(|error| {
                NoetherError::Serialization(format!(
                    "invalid Bézout {name} component {component:?}: {error}"
                ))
            })
        };
        Ok(Self {
            gcd: parse("gcd", wire.gcd)?,
            coefficient_a: parse("coefficient_a", wire.coefficient_a)?,
            coefficient_b: parse("coefficient_b", wire.coefficient_b)?,
        })
    }
}

impl<R: Ring> SemanticType for BezoutIdentity<R> {
    const TYPE_ID: &'static str = "noether/root/BezoutIdentity@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let components = [
            self.gcd.canonical_bytes()?,
            self.coefficient_a.canonical_bytes()?,
            self.coefficient_b.canonical_bytes()?,
        ];
        let mut output = Vec::new();
        for component in components {
            let length = u64::try_from(component.len()).map_err(|_| {
                NoetherError::Serialization(
                    "Bézout component is too large for canonical framing".to_owned(),
                )
            })?;
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(&component);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &BEZOUT_IDENTITY_DESCRIPTION
    }
}

impl<R: PrincipalIdealDomain> BezoutIdentity<R> {
    /// Verify the normalized gcd and Bézout relation for inputs `a` and `b`.
    pub fn verify(
        &self,
        ring: &R,
        a: &R::Element,
        b: &R::Element,
    ) -> Result<(), VerificationError> {
        for (name, value) in [
            ("a", a),
            ("b", b),
            ("gcd", &self.gcd),
            ("coefficient_a", &self.coefficient_a),
            ("coefficient_b", &self.coefficient_b),
        ] {
            if !ring.contains(value) {
                return Err(ContextMismatchError {
                    expected: R::TYPE_ID.to_owned(),
                    actual: format!("{name} belongs to another context"),
                }
                .into());
            }
        }

        if ring.gcd(a, b) != self.gcd {
            return Err(VerificationError::InvalidWitness {
                claim: "gcd is not the context's canonical normalized gcd",
            });
        }

        let first = ring.mul(&self.coefficient_a, a)?;
        let second = ring.mul(&self.coefficient_b, b)?;
        if ring.add(&first, &second)? != self.gcd {
            return Err(VerificationError::InvalidWitness {
                claim: "gcd != coefficient_a * a + coefficient_b * b",
            });
        }

        Ok(())
    }
}

/// An integral domain in which every ideal is principal.
pub trait PrincipalIdealDomain: IntegralDomain {
    /// Compute a greatest common divisor.
    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Compute a Bézout identity.
    fn extended_gcd(&self, a: &Self::Element, b: &Self::Element) -> BezoutIdentity<Self>;
}

/// A context whose precision specification is fixed at construction.
pub trait PrecisionContext: Ring {
    /// Domain-specific precision shape.
    type Precision: Clone + Eq + Send + Sync + 'static;

    /// The immutable precision specification of this context.
    fn precision(&self) -> &Self::Precision;

    /// Whether `candidate` contains at least the information in `reference`.
    fn refines(&self, candidate: &Self::Precision, reference: &Self::Precision) -> bool;
}

/// A commutative finite-precision context with a valuation.
pub trait ValuedRing: CommutativeRing + PrecisionContext {
    /// A chosen uniformizer.
    fn uniformizer(&self) -> Self::Element;

    /// A proven lower bound for the valuation, or `None` for zero.
    fn valuation_lower_bound(&self, value: &Self::Element) -> Option<i64>;
}

/// A genuinely finite ring or field context.
pub trait FiniteContext: Ring {
    /// Exact cardinality without eager enumeration.
    fn cardinality(&self) -> Integer;

    /// Deterministic lazy iteration over all elements.
    ///
    /// The iterator deliberately does not promise [`ExactSizeIterator`]: a
    /// finite context may have cardinality larger than `usize`.
    fn elements(&self) -> Box<dyn Iterator<Item = Self::Element> + '_>;
}

/// Deterministic checkpoint and coverage support for long enumerations.
pub trait CheckpointableEnumeration: Iterator + SemanticType {
    /// Operation-specific checkpoint value.
    type Checkpoint: SemanticType;
    /// Operation-specific coverage value.
    type Coverage: SemanticType;

    /// Capture a resumable checkpoint.
    fn checkpoint(&self) -> Self::Checkpoint;
    /// Report the ranges covered by this enumeration.
    fn coverage(&self) -> &Self::Coverage;
}
