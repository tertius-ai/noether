// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Generic sparse multivariate polynomial rings.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};

use crate::{
    CoefficientMap, CoefficientMappingError, CommutativeRing, ContextMismatchError,
    DivisionByZeroError, EuclideanDomain, Field, Integer, IntegralDomain, NoetherError, Ring,
    SemanticType, TypeDescription, UnitError, UnitInversion, Q, Z,
};

use super::{
    Degree, FreeAbelianGrading, GradingError, PositiveGrading, ResidueRing, SeriesTruncation,
    SeriesTruncationError, SeriesTruncationMode, WeightedGrading,
};

static MONOMIAL_ORDER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/MonomialOrder@1",
    rust_path: "noether::commalg::MonomialOrder",
    python_path: Some("noether.commalg.MonomialOrder"),
};

static POLYNOMIAL_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/PolynomialRing@1",
    rust_path: "noether::commalg::PolynomialRing",
    python_path: Some("noether.commalg.PolynomialRing"),
};

static POLYNOMIAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/Polynomial@1",
    rust_path: "noether::commalg::Polynomial",
    python_path: Some("noether.commalg.Polynomial"),
};

const PACKED_TERM_PRODUCT_CROSSOVER: usize = 64;
const MAX_GRADED_PIECE_CANDIDATES: usize = 1_000_000;

pub(super) struct PolyCoreRingLane<'a, R: CommutativeRing> {
    pub(super) ring: &'a R,
}

impl<R: CommutativeRing> poly_core::CoefficientRing for PolyCoreRingLane<'_, R> {
    type Element = R::Element;

    fn zero(&self) -> Self::Element {
        self.ring.zero()
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        self.ring.is_zero(value)
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        self.ring
            .add(left, right)
            .expect("validated packed polynomial coefficient addition")
    }

    fn mul(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        self.ring
            .mul(left, right)
            .expect("validated packed polynomial coefficient multiplication")
    }

    fn neg(&self, value: &Self::Element) -> Self::Element {
        self.ring
            .neg(value)
            .expect("validated packed polynomial coefficient negation")
    }
}

/// Failure to construct an admissible block monomial order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MonomialOrderError {
    /// A block order did not contain any blocks.
    NoBlocks,
    /// One block did not contain any variables.
    EmptyBlock {
        /// Zero-based block index.
        block: usize,
    },
    /// A block component was itself a block order.
    NestedBlock {
        /// Zero-based block index.
        block: usize,
    },
    /// One variable occurred in more than one block.
    DuplicateVariable {
        /// Repeated zero-based variable index.
        variable: usize,
    },
    /// A block referred to a variable outside its polynomial context.
    VariableOutOfRange {
        /// Rejected zero-based variable index.
        variable: usize,
        /// Number of variables in the polynomial context.
        variable_count: usize,
    },
    /// A block order omitted a variable from its polynomial context.
    MissingVariable {
        /// Omitted zero-based variable index.
        variable: usize,
    },
}

impl fmt::Display for MonomialOrderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBlocks => {
                formatter.write_str("a block monomial order requires at least one block")
            }
            Self::EmptyBlock { block } => {
                write!(formatter, "monomial-order block {block} is empty")
            }
            Self::NestedBlock { block } => write!(
                formatter,
                "monomial-order block {block} uses a nested block order"
            ),
            Self::DuplicateVariable { variable } => write!(
                formatter,
                "variable index {variable} occurs more than once in the block order"
            ),
            Self::VariableOutOfRange {
                variable,
                variable_count,
            } => write!(
                formatter,
                "variable index {variable} is outside a polynomial context with {variable_count} variables"
            ),
            Self::MissingVariable { variable } => write!(
                formatter,
                "variable index {variable} is missing from the block order"
            ),
        }
    }
}

impl std::error::Error for MonomialOrderError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StandardMonomialOrder {
    Lexicographic,
    GradedLexicographic,
    GradedReverseLexicographic,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MonomialOrderBlock {
    variables: Vec<usize>,
    order: StandardMonomialOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MonomialOrderKind {
    Standard(StandardMonomialOrder),
    Block(Arc<[MonomialOrderBlock]>),
}

/// A deterministic admissible order on monomials.
///
/// Besides the three standard global orders, Noether supports block orders on
/// an ordered partition of the variable indices. Earlier blocks dominate
/// later blocks, and each block uses one standard order internally. A
/// [`PolynomialRing`] validates that the blocks contain every variable exactly
/// once.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MonomialOrder {
    kind: MonomialOrderKind,
}

impl Default for MonomialOrder {
    fn default() -> Self {
        Self::GradedReverseLexicographic
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MonomialOrderWire {
    #[serde(rename = "type")]
    type_id: String,
    name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    blocks: Vec<MonomialOrderBlockWire>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MonomialOrderBlockWire {
    variables: Vec<usize>,
    order: String,
}

impl MonomialOrder {
    /// Pure lexicographic order, with the first variable dominant.
    #[allow(non_upper_case_globals)]
    pub const Lexicographic: Self = Self {
        kind: MonomialOrderKind::Standard(StandardMonomialOrder::Lexicographic),
    };

    /// Total degree followed by lexicographic order.
    #[allow(non_upper_case_globals)]
    pub const GradedLexicographic: Self = Self {
        kind: MonomialOrderKind::Standard(StandardMonomialOrder::GradedLexicographic),
    };

    /// Total degree followed by reverse lexicographic order.
    #[allow(non_upper_case_globals)]
    pub const GradedReverseLexicographic: Self = Self {
        kind: MonomialOrderKind::Standard(StandardMonomialOrder::GradedReverseLexicographic),
    };

    /// Construct a block order on an ordered partition of variable indices.
    ///
    /// Each component order must be one of the three standard orders. Empty
    /// blocks, repeated indices, and nested block orders are rejected here;
    /// the eventual polynomial-ring context additionally checks range and
    /// coverage.
    pub fn block<I, J>(blocks: I) -> Result<Self, MonomialOrderError>
    where
        I: IntoIterator<Item = (J, Self)>,
        J: IntoIterator<Item = usize>,
    {
        let mut seen = BTreeSet::new();
        let mut result = Vec::new();
        for (block, (variables, order)) in blocks.into_iter().enumerate() {
            let variables = variables.into_iter().collect::<Vec<_>>();
            if variables.is_empty() {
                return Err(MonomialOrderError::EmptyBlock { block });
            }
            for &variable in &variables {
                if !seen.insert(variable) {
                    return Err(MonomialOrderError::DuplicateVariable { variable });
                }
            }
            let MonomialOrderKind::Standard(order) = order.kind else {
                return Err(MonomialOrderError::NestedBlock { block });
            };
            result.push(MonomialOrderBlock { variables, order });
        }
        if result.is_empty() {
            return Err(MonomialOrderError::NoBlocks);
        }
        Ok(Self {
            kind: MonomialOrderKind::Block(Arc::from(result)),
        })
    }

    /// Construct the standard two-block elimination order.
    ///
    /// The selected variables form a leading lexicographic block; all other
    /// variables form a trailing graded-reverse-lexicographic block. Variable
    /// order within each block is their ambient order.
    pub fn elimination<I>(variable_count: usize, eliminated: I) -> Result<Self, MonomialOrderError>
    where
        I: IntoIterator<Item = usize>,
    {
        let eliminated = eliminated.into_iter().collect::<BTreeSet<_>>();
        if let Some(&variable) = eliminated
            .iter()
            .find(|&&variable| variable >= variable_count)
        {
            return Err(MonomialOrderError::VariableOutOfRange {
                variable,
                variable_count,
            });
        }
        if eliminated.is_empty() {
            return Ok(Self::GradedReverseLexicographic);
        }
        if eliminated.len() == variable_count {
            return Ok(Self::Lexicographic);
        }
        let leading = eliminated.iter().copied().collect::<Vec<_>>();
        let trailing = (0..variable_count)
            .filter(|variable| !eliminated.contains(variable))
            .collect::<Vec<_>>();
        Self::block([
            (leading, Self::Lexicographic),
            (trailing, Self::GradedReverseLexicographic),
        ])
    }

    pub(crate) fn compare(&self, left: &[Integer], right: &[Integer]) -> Ordering {
        match &self.kind {
            MonomialOrderKind::Standard(order) => compare_standard_all(*order, left, right),
            MonomialOrderKind::Block(blocks) => blocks
                .iter()
                .map(|block| compare_standard(block.order, left, right, &block.variables))
                .find(|ordering| *ordering != Ordering::Equal)
                .unwrap_or(Ordering::Equal),
        }
    }

    /// Stable short name accepted by the Python constructor.
    pub fn name(&self) -> &'static str {
        match &self.kind {
            MonomialOrderKind::Standard(StandardMonomialOrder::Lexicographic) => "lex",
            MonomialOrderKind::Standard(StandardMonomialOrder::GradedLexicographic) => "grlex",
            MonomialOrderKind::Standard(StandardMonomialOrder::GradedReverseLexicographic) => {
                "grevlex"
            }
            MonomialOrderKind::Block(_) => "block",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "lex" => Some(Self::Lexicographic),
            "grlex" => Some(Self::GradedLexicographic),
            "grevlex" => Some(Self::GradedReverseLexicographic),
            _ => None,
        }
    }

    pub(super) fn standard_name(&self) -> Option<&'static str> {
        matches!(&self.kind, MonomialOrderKind::Standard(_)).then(|| self.name())
    }

    pub(super) fn is_degree_compatible(&self) -> bool {
        matches!(
            &self.kind,
            MonomialOrderKind::Standard(
                StandardMonomialOrder::GradedLexicographic
                    | StandardMonomialOrder::GradedReverseLexicographic
            )
        )
    }

    /// Ordered block specifications, or `None` for a standard global order.
    pub fn blocks(&self) -> Option<Vec<(Vec<usize>, Self)>> {
        let MonomialOrderKind::Block(blocks) = &self.kind else {
            return None;
        };
        Some(
            blocks
                .iter()
                .map(|block| {
                    (
                        block.variables.clone(),
                        Self {
                            kind: MonomialOrderKind::Standard(block.order),
                        },
                    )
                })
                .collect(),
        )
    }

    pub(super) fn validate_arity(&self, variable_count: usize) -> Result<(), MonomialOrderError> {
        let MonomialOrderKind::Block(blocks) = &self.kind else {
            return Ok(());
        };
        let mut seen = vec![false; variable_count];
        for block in blocks.iter() {
            for &variable in &block.variables {
                if variable >= variable_count {
                    return Err(MonomialOrderError::VariableOutOfRange {
                        variable,
                        variable_count,
                    });
                }
                if seen[variable] {
                    return Err(MonomialOrderError::DuplicateVariable { variable });
                }
                seen[variable] = true;
            }
        }
        if let Some(variable) = seen.iter().position(|seen| !seen) {
            return Err(MonomialOrderError::MissingVariable { variable });
        }
        Ok(())
    }

    fn without_variable(&self, variable: usize) -> Self {
        let MonomialOrderKind::Block(blocks) = &self.kind else {
            return self.clone();
        };
        let remaining = blocks
            .iter()
            .filter_map(|block| {
                let variables = block
                    .variables
                    .iter()
                    .filter(|&&candidate| candidate != variable)
                    .map(|&candidate| candidate - usize::from(candidate > variable))
                    .collect::<Vec<_>>();
                (!variables.is_empty()).then_some(MonomialOrderBlock {
                    variables,
                    order: block.order,
                })
            })
            .collect::<Vec<_>>();
        if remaining.len() == 1 {
            Self {
                kind: MonomialOrderKind::Standard(remaining[0].order),
            }
        } else {
            Self {
                kind: MonomialOrderKind::Block(Arc::from(remaining)),
            }
        }
    }

    pub(super) fn without_variables(&self, variables: &[usize]) -> Self {
        let mut variables = variables.to_vec();
        variables.sort_unstable();
        variables.dedup();
        let mut result = self.clone();
        for variable in variables.into_iter().rev() {
            result = result.without_variable(variable);
        }
        result
    }

    fn on_variable(&self, variable: usize) -> Self {
        match &self.kind {
            MonomialOrderKind::Standard(_) => self.clone(),
            MonomialOrderKind::Block(blocks) => {
                let order = blocks
                    .iter()
                    .find(|block| block.variables.contains(&variable))
                    .map(|block| block.order)
                    .expect("a validated block order contains every variable");
                Self {
                    kind: MonomialOrderKind::Standard(order),
                }
            }
        }
    }

    fn blocks_wire(&self) -> Vec<MonomialOrderBlockWire> {
        let MonomialOrderKind::Block(blocks) = &self.kind else {
            return Vec::new();
        };
        blocks
            .iter()
            .map(|block| MonomialOrderBlockWire {
                variables: block.variables.clone(),
                order: standard_name(block.order).to_owned(),
            })
            .collect()
    }

    /// Serialize this order's stable name as canonical typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate one admissible monomial order.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for MonomialOrder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        MonomialOrderWire {
            type_id: Self::TYPE_ID.to_owned(),
            name: self.name().to_owned(),
            blocks: self.blocks_wire(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MonomialOrder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MonomialOrderWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        if wire.name != "block" {
            if !wire.blocks.is_empty() {
                return Err(serde::de::Error::custom(
                    "a standard monomial order cannot contain blocks",
                ));
            }
            return Self::from_name(&wire.name).ok_or_else(|| {
                serde::de::Error::custom(format!("unknown monomial order {:?}", wire.name))
            });
        }
        let blocks = wire
            .blocks
            .into_iter()
            .map(|block| {
                let order = Self::from_name(&block.order).ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "unknown block monomial order {:?}",
                        block.order
                    ))
                })?;
                Ok((block.variables, order))
            })
            .collect::<Result<Vec<_>, D::Error>>()?;
        Self::block(blocks).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for MonomialOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl SemanticType for MonomialOrder {
    const TYPE_ID: &'static str = "noether/commalg/MonomialOrder@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = vec![match &self.kind {
            MonomialOrderKind::Standard(StandardMonomialOrder::Lexicographic) => 0,
            MonomialOrderKind::Standard(StandardMonomialOrder::GradedLexicographic) => 1,
            MonomialOrderKind::Standard(StandardMonomialOrder::GradedReverseLexicographic) => 2,
            MonomialOrderKind::Block(_) => 3,
        }];
        if let MonomialOrderKind::Block(blocks) = &self.kind {
            output.extend_from_slice(&(blocks.len() as u64).to_be_bytes());
            for block in blocks.iter() {
                output.push(standard_code(block.order));
                output.extend_from_slice(&(block.variables.len() as u64).to_be_bytes());
                for &variable in &block.variables {
                    let variable = u64::try_from(variable).map_err(|_| {
                        NoetherError::Serialization(
                            "block-order variable index does not fit canonical framing".to_owned(),
                        )
                    })?;
                    output.extend_from_slice(&variable.to_be_bytes());
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &MONOMIAL_ORDER_DESCRIPTION
    }
}

fn standard_code(order: StandardMonomialOrder) -> u8 {
    match order {
        StandardMonomialOrder::Lexicographic => 0,
        StandardMonomialOrder::GradedLexicographic => 1,
        StandardMonomialOrder::GradedReverseLexicographic => 2,
    }
}

fn standard_name(order: StandardMonomialOrder) -> &'static str {
    match order {
        StandardMonomialOrder::Lexicographic => "lex",
        StandardMonomialOrder::GradedLexicographic => "grlex",
        StandardMonomialOrder::GradedReverseLexicographic => "grevlex",
    }
}

fn compare_standard(
    order: StandardMonomialOrder,
    left: &[Integer],
    right: &[Integer],
    variables: &[usize],
) -> Ordering {
    match order {
        StandardMonomialOrder::Lexicographic => variables
            .iter()
            .map(|&variable| left[variable].cmp(&right[variable]))
            .find(|ordering| *ordering != Ordering::Equal)
            .unwrap_or(Ordering::Equal),
        StandardMonomialOrder::GradedLexicographic => variables
            .iter()
            .fold(Integer::zero(), |degree, &variable| {
                &degree + &left[variable]
            })
            .cmp(
                &variables.iter().fold(Integer::zero(), |degree, &variable| {
                    &degree + &right[variable]
                }),
            )
            .then_with(|| {
                variables
                    .iter()
                    .map(|&variable| left[variable].cmp(&right[variable]))
                    .find(|ordering| *ordering != Ordering::Equal)
                    .unwrap_or(Ordering::Equal)
            }),
        StandardMonomialOrder::GradedReverseLexicographic => {
            let degree = variables
                .iter()
                .fold(Integer::zero(), |degree, &variable| {
                    &degree + &left[variable]
                })
                .cmp(
                    &variables.iter().fold(Integer::zero(), |degree, &variable| {
                        &degree + &right[variable]
                    }),
                );
            if degree != Ordering::Equal {
                return degree;
            }
            for &variable in variables.iter().rev() {
                match left[variable].cmp(&right[variable]) {
                    Ordering::Equal => {}
                    Ordering::Less => return Ordering::Greater,
                    Ordering::Greater => return Ordering::Less,
                }
            }
            Ordering::Equal
        }
    }
}

fn compare_standard_all(
    order: StandardMonomialOrder,
    left: &[Integer],
    right: &[Integer],
) -> Ordering {
    match order {
        StandardMonomialOrder::Lexicographic => left.cmp(right),
        StandardMonomialOrder::GradedLexicographic => left
            .iter()
            .fold(Integer::zero(), |degree, power| &degree + power)
            .cmp(
                &right
                    .iter()
                    .fold(Integer::zero(), |degree, power| &degree + power),
            )
            .then_with(|| left.cmp(right)),
        StandardMonomialOrder::GradedReverseLexicographic => {
            let degree = left
                .iter()
                .fold(Integer::zero(), |degree, power| &degree + power)
                .cmp(
                    &right
                        .iter()
                        .fold(Integer::zero(), |degree, power| &degree + power),
                );
            if degree != Ordering::Equal {
                return degree;
            }
            for (left_power, right_power) in left.iter().zip(right).rev() {
                match left_power.cmp(right_power) {
                    Ordering::Equal => {}
                    Ordering::Less => return Ordering::Greater,
                    Ordering::Greater => return Ordering::Less,
                }
            }
            Ordering::Equal
        }
    }
}

/// Failure to construct a polynomial-ring context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolynomialContextError {
    /// One variable name was empty.
    EmptyVariableName {
        /// Zero-based variable index.
        variable: usize,
    },
    /// Two variables had the same name.
    DuplicateVariableName {
        /// Repeated name.
        name: String,
    },
    /// The grading was invalid or had the wrong arity.
    Grading(GradingError),
    /// The monomial order was not admissible for the variable context.
    MonomialOrder(MonomialOrderError),
}

impl fmt::Display for PolynomialContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVariableName { variable } => {
                write!(f, "polynomial variable {variable} has an empty name")
            }
            Self::DuplicateVariableName { name } => {
                write!(f, "polynomial variable name {name:?} is duplicated")
            }
            Self::Grading(error) => error.fmt(f),
            Self::MonomialOrder(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for PolynomialContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Grading(error) => Some(error),
            Self::MonomialOrder(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GradingError> for PolynomialContextError {
    fn from(value: GradingError) -> Self {
        Self::Grading(value)
    }
}

impl From<MonomialOrderError> for PolynomialContextError {
    fn from(value: MonomialOrderError) -> Self {
        Self::MonomialOrder(value)
    }
}

/// Failure to construct or operate on a polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolynomialError {
    /// Values belonged to unequal polynomial contexts.
    ContextMismatch(ContextMismatchError),
    /// An exponent vector had the wrong number of entries.
    ExponentArity {
        /// Required number of exponents.
        expected: usize,
        /// Supplied number of exponents.
        actual: usize,
    },
    /// One exponent was negative.
    NegativeExponent {
        /// Zero-based variable index.
        variable: usize,
    },
    /// A coefficient belonged to another coefficient context.
    CoefficientOutsideContext,
    /// A variable index was outside the context arity.
    VariableIndex {
        /// Rejected zero-based index.
        variable: usize,
        /// Number of variables in the context.
        arity: usize,
    },
    /// Evaluation or substitution supplied the wrong number of images.
    ImageArity {
        /// Required number of images.
        expected: usize,
        /// Supplied number of images.
        actual: usize,
    },
    /// Euclidean polynomial arithmetic was requested in a multivariate ring.
    NotUnivariate {
        /// Number of variables in the polynomial ring.
        arity: usize,
    },
    /// Selected-variable coefficients were requested from a one-variable ring.
    NoRemainingVariables {
        /// Selected zero-based variable index.
        selected: usize,
    },
    /// A checked univariate view still depended on another variable.
    DependsOnOtherVariable {
        /// Variable retained by the requested view.
        selected: usize,
        /// Another variable with a nonzero exponent.
        other: usize,
    },
    /// A bounded polynomial operation exceeded its explicit output limit.
    ResourceLimit {
        /// Bounded resource that was exhausted.
        resource: &'static str,
        /// Caller-supplied maximum.
        limit: usize,
    },
    /// A rank-one convenience was requested for a higher-rank grading.
    RankOneGradingRequired {
        /// Actual rank of the polynomial ring's grading group.
        actual: usize,
    },
    /// A supplied degree had the wrong grading-group rank.
    DegreeRank {
        /// Rank required by the grading.
        expected: usize,
        /// Rank supplied by the degree.
        actual: usize,
    },
    /// A zero polynomial was used as a divisor.
    DivisionByZero,
    /// Exact division left a nonzero remainder.
    NonExactDivision,
    /// Division over a coefficient ring requires a monic divisor.
    NonMonicDivisor,
    /// A polynomial was not invertible modulo the supplied polynomial.
    NotCoprime,
    /// An operation guaranteed by the coefficient context unexpectedly failed.
    ArithmeticInvariant,
}

impl fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(f),
            Self::ExponentArity { expected, actual } => write!(
                f,
                "polynomial exponent has length {actual}; expected {expected}"
            ),
            Self::NegativeExponent { variable } => {
                write!(
                    f,
                    "polynomial exponent for variable {variable} must be nonnegative"
                )
            }
            Self::CoefficientOutsideContext => {
                f.write_str("polynomial coefficient belongs to another ring context")
            }
            Self::VariableIndex { variable, arity } => {
                write!(
                    f,
                    "polynomial variable index {variable} is outside arity {arity}"
                )
            }
            Self::ImageArity { expected, actual } => write!(
                f,
                "polynomial operation received {actual} images; expected {expected}"
            ),
            Self::NotUnivariate { arity } => write!(
                f,
                "univariate polynomial arithmetic requires exactly one variable; ring has {arity}"
            ),
            Self::NoRemainingVariables { selected } => write!(
                f,
                "coefficients in variable {selected} cannot be represented as polynomials because no variables remain"
            ),
            Self::DependsOnOtherVariable { selected, other } => write!(
                f,
                "polynomial cannot be viewed as univariate in variable {selected}: it depends on variable {other}"
            ),
            Self::ResourceLimit { resource, limit } => write!(
                f,
                "polynomial operation exceeded its {resource} limit of {limit}"
            ),
            Self::RankOneGradingRequired { actual } => write!(
                f,
                "operation requires a rank-one grading; grading rank is {actual}"
            ),
            Self::DegreeRank { expected, actual } => write!(
                f,
                "degree has rank {actual}; polynomial grading has rank {expected}"
            ),
            Self::DivisionByZero => f.write_str("cannot divide by the zero polynomial"),
            Self::NonExactDivision => f.write_str("polynomial division has a nonzero remainder"),
            Self::NonMonicDivisor => f.write_str("polynomial divisor must be monic"),
            Self::NotCoprime => f.write_str("polynomial is not coprime to its modulus"),
            Self::ArithmeticInvariant => {
                f.write_str("coefficient arithmetic violated its declared ring capability")
            }
        }
    }
}

impl std::error::Error for PolynomialError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for PolynomialError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// A sparse multivariate polynomial-ring context over `R`.
#[derive(Clone)]
pub struct PolynomialRing<R: CommutativeRing> {
    coefficient_ring: R,
    variables: Arc<Vec<String>>,
    term_order: MonomialOrder,
    grading: Arc<FreeAbelianGrading>,
    backend: Option<Arc<super::polynomial_flint::FlintPolynomialContext>>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>")
)]
struct PolynomialRingWire<R: CommutativeRing> {
    #[serde(rename = "type")]
    type_id: String,
    coefficient_ring: R,
    variables: Vec<String>,
    term_order: MonomialOrder,
    grading: PolynomialGradingWire,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum PolynomialGradingWire {
    Weighted(WeightedGrading),
    FreeAbelian(FreeAbelianGrading),
}

impl PolynomialGradingWire {
    fn into_grading(self) -> FreeAbelianGrading {
        match self {
            Self::Weighted(grading) => grading.into(),
            Self::FreeAbelian(grading) => grading,
        }
    }
}

impl<R: CommutativeRing> PolynomialRing<R> {
    fn polynomial_from_canonical_terms(
        &self,
        terms: BTreeMap<Vec<Integer>, R::Element>,
    ) -> Polynomial<R> {
        let storage = self
            .backend
            .as_ref()
            .and_then(|context| {
                super::polynomial_flint::from_terms(
                    Arc::clone(context),
                    &self.coefficient_ring,
                    &terms,
                )
            })
            .map(|native| {
                PolynomialStorage::Flint(Arc::new(FlintBackedPolynomial {
                    native,
                    terms: OnceLock::new(),
                }))
            })
            .unwrap_or_else(|| PolynomialStorage::Generic(Arc::new(terms)));
        Polynomial {
            ring: self.clone(),
            storage,
        }
    }

    pub(super) fn polynomial_from_native(
        &self,
        native: super::polynomial_flint::FlintPolynomial,
    ) -> Polynomial<R> {
        Polynomial {
            ring: self.clone(),
            storage: PolynomialStorage::Flint(Arc::new(FlintBackedPolynomial {
                native,
                terms: OnceLock::new(),
            })),
        }
    }

    /// Copy a structurally compatible polynomial into this ring's native
    /// backend context.
    ///
    /// Equal polynomial contexts reconstructed independently (most notably by
    /// nested JSON deserialization) own distinct FLINT context pointers.  The
    /// sparse terms remain the representation-independent interchange format,
    /// so materializing them here lets native operations retain their fast
    /// path without weakening exact context equality.
    fn materialize_native(
        &self,
        value: &Polynomial<R>,
    ) -> Option<super::polynomial_flint::FlintPolynomial> {
        let context = self.backend.as_ref()?;
        super::polynomial_flint::from_terms(
            Arc::clone(context),
            &self.coefficient_ring,
            value.terms_map(),
        )
    }

    /// Construct a polynomial ring with the standard all-weights-one grading.
    pub fn new<I, S>(
        coefficient_ring: R,
        variables: I,
        term_order: MonomialOrder,
    ) -> Result<Self, PolynomialContextError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let variables = collect_variables(variables)?;
        term_order.validate_arity(variables.len())?;
        let grading = FreeAbelianGrading::standard(variables.len());
        // FLINT's multivariate contexts require a positive arity. The
        // zero-variable polynomial ring is the coefficient ring with an
        // explicit polynomial-context identity and uses the exact sparse
        // representation.
        let backend = (!variables.is_empty())
            .then(|| {
                super::polynomial_flint::context(&coefficient_ring, variables.len(), &term_order)
            })
            .flatten();
        Ok(Self {
            coefficient_ring,
            variables: Arc::new(variables),
            term_order,
            grading: Arc::new(grading),
            backend,
        })
    }

    /// Construct a polynomial ring with an explicit free-abelian grading.
    ///
    /// A [`WeightedGrading`] is accepted as its positive rank-one special
    /// case. The grading is part of ring identity.
    pub fn with_grading<I, S, G>(
        coefficient_ring: R,
        variables: I,
        term_order: MonomialOrder,
        grading: G,
    ) -> Result<Self, PolynomialContextError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
        G: Into<FreeAbelianGrading>,
    {
        let variables = collect_variables(variables)?;
        term_order.validate_arity(variables.len())?;
        let grading = grading.into();
        if grading.arity() != variables.len() {
            return Err(GradingError::ArityMismatch {
                expected: variables.len(),
                actual: grading.arity(),
            }
            .into());
        }
        let backend = (!variables.is_empty())
            .then(|| {
                super::polynomial_flint::context(&coefficient_ring, variables.len(), &term_order)
            })
            .flatten();
        Ok(Self {
            coefficient_ring,
            variables: Arc::new(variables),
            term_order,
            grading: Arc::new(grading),
            backend,
        })
    }

    /// Coefficient-ring context.
    pub fn coefficient_ring(&self) -> &R {
        &self.coefficient_ring
    }

    /// Ordered indeterminate names.
    pub fn variables(&self) -> &[String] {
        self.variables.as_slice()
    }

    /// Monomial order used for leading terms and later Gröbner algorithms.
    pub fn term_order(&self) -> MonomialOrder {
        self.term_order.clone()
    }

    /// Free-abelian grading carried by this structured polynomial context.
    pub fn grading(&self) -> &FreeAbelianGrading {
        self.grading.as_ref()
    }

    /// Positive rank-one view of this grading, when one exists.
    pub fn weighted_grading(&self) -> Option<WeightedGrading> {
        self.grading.weighted_view()
    }

    /// Serialize the complete coefficient, variable, order, and grading context.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate a complete polynomial-ring context.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Construct a sparse polynomial from arbitrary-size exponent vectors.
    ///
    /// Duplicate monomials are summed and zero coefficients are removed.
    pub fn from_terms<I>(&self, terms: I) -> Result<Polynomial<R>, PolynomialError>
    where
        I: IntoIterator<Item = (Vec<Integer>, R::Element)>,
    {
        let mut canonical = BTreeMap::new();
        for (exponent, coefficient) in terms {
            self.validate_exponent(&exponent)?;
            if !self.coefficient_ring.contains(&coefficient) {
                return Err(PolynomialError::CoefficientOutsideContext);
            }
            if coefficient == self.coefficient_ring.zero() {
                continue;
            }
            let sum = if let Some(previous) = canonical.remove(&exponent) {
                self.coefficient_ring.add(&previous, &coefficient)?
            } else {
                coefficient
            };
            if sum != self.coefficient_ring.zero() {
                canonical.insert(exponent, sum);
            }
        }
        Ok(self.polynomial_from_canonical_terms(canonical))
    }

    /// Construct from ordinary machine-word exponent vectors.
    pub fn from_u64_terms<I>(&self, terms: I) -> Result<Polynomial<R>, PolynomialError>
    where
        I: IntoIterator<Item = (Vec<u64>, R::Element)>,
    {
        if self.backend.as_ref().is_some_and(|context| {
            !super::polynomial_flint::supports_direct_u64_construction(context)
        }) {
            return self.from_terms(terms.into_iter().map(|(exponent, coefficient)| {
                (
                    exponent.into_iter().map(Integer::from).collect(),
                    coefficient,
                )
            }));
        }
        let mut canonical = BTreeMap::new();
        for (exponent, coefficient) in terms {
            if exponent.len() != self.variables.len() {
                return Err(PolynomialError::ExponentArity {
                    expected: self.variables.len(),
                    actual: exponent.len(),
                });
            }
            if !self.coefficient_ring.contains(&coefficient) {
                return Err(PolynomialError::CoefficientOutsideContext);
            }
            if coefficient == self.coefficient_ring.zero() {
                continue;
            }
            let sum = if let Some(previous) = canonical.remove(&exponent) {
                self.coefficient_ring.add(&previous, &coefficient)?
            } else {
                coefficient
            };
            if sum != self.coefficient_ring.zero() {
                canonical.insert(exponent, sum);
            }
        }
        if let Some(native) = self.backend.as_ref().and_then(|context| {
            super::polynomial_flint::from_canonical_u64_terms(
                Arc::clone(context),
                &self.coefficient_ring,
                &canonical,
            )
        }) {
            return Ok(self.polynomial_from_native(native));
        }
        Ok(self.polynomial_from_canonical_terms(
            canonical
                .into_iter()
                .map(|(exponent, coefficient)| {
                    (
                        exponent.into_iter().map(Integer::from).collect(),
                        coefficient,
                    )
                })
                .collect(),
        ))
    }

    /// Project canonical kernel terms without re-canonicalizing them.
    ///
    /// Terms must be nonzero, unique, and ordered greatest-first in this
    /// ring's monomial order. The Gröbner kernels maintain that invariant.
    pub(super) fn polynomial_from_canonical_u32_terms(
        &self,
        terms: Vec<(Vec<u32>, R::Element)>,
    ) -> Result<Polynomial<R>, PolynomialError> {
        let zero = self.coefficient_ring.zero();
        for (exponent, coefficient) in &terms {
            if exponent.len() != self.variables.len() {
                return Err(PolynomialError::ExponentArity {
                    expected: self.variables.len(),
                    actual: exponent.len(),
                });
            }
            if !self.coefficient_ring.contains(coefficient) {
                return Err(PolynomialError::CoefficientOutsideContext);
            }
            if coefficient == &zero {
                return Err(PolynomialError::ArithmeticInvariant);
            }
        }
        if let Some(native) = self.backend.as_ref().and_then(|context| {
            super::polynomial_flint::from_canonical_u32_terms(
                Arc::clone(context),
                &self.coefficient_ring,
                &terms,
            )
        }) {
            return Ok(Polynomial {
                ring: self.clone(),
                storage: PolynomialStorage::Flint(Arc::new(FlintBackedPolynomial {
                    native,
                    terms: OnceLock::new(),
                })),
            });
        }
        self.from_terms(terms.into_iter().map(|(exponent, coefficient)| {
            (
                exponent.into_iter().map(Integer::from).collect(),
                coefficient,
            )
        }))
    }

    /// Embed one coefficient as a constant polynomial.
    pub fn constant(&self, coefficient: R::Element) -> Result<Polynomial<R>, PolynomialError> {
        self.from_terms([(vec![Integer::zero(); self.variables.len()], coefficient)])
    }

    /// Return the `variable`-th indeterminate.
    pub fn generator(&self, variable: usize) -> Result<Polynomial<R>, PolynomialError> {
        if variable >= self.variables.len() {
            return Err(PolynomialError::VariableIndex {
                variable,
                arity: self.variables.len(),
            });
        }
        let mut exponent = vec![Integer::zero(); self.variables.len()];
        exponent[variable] = Integer::one();
        self.from_terms([(exponent, self.coefficient_ring.one())])
    }

    /// All indeterminates in declared order.
    pub fn generators(&self) -> Result<Vec<Polynomial<R>>, PolynomialError> {
        (0..self.variables.len())
            .map(|variable| self.generator(variable))
            .collect()
    }

    /// Enumerate the monomials of one exact degree in this ring's grading.
    ///
    /// Standard grading gives ordinary homogeneous monomials; an explicit
    /// [`WeightedGrading`] gives weighted-homogeneous monomials. Results are
    /// greatest-to-least in the ring's term order. The operation fails before
    /// retaining more than `max_count` monomials.
    pub fn homogeneous_monomials(
        &self,
        degree: u64,
        max_count: usize,
    ) -> Result<Vec<Polynomial<R>>, PolynomialError> {
        let mut exponents = Vec::new();
        let mut exponent = vec![0_u64; self.variables.len()];
        let grading = self
            .weighted_grading()
            .ok_or(PolynomialError::RankOneGradingRequired {
                actual: self.grading.rank(),
            })?;
        enumerate_exact_weighted(
            grading.weights(),
            0,
            degree,
            &mut exponent,
            &mut exponents,
            max_count,
        )?;
        self.monomials_from_exponents(exponents)
    }

    /// Enumerate one exact piece of an arbitrary free-abelian grading.
    ///
    /// `positivity` is exact evidence that the piece is finite. Its grading
    /// must equal this ring's grading. Candidate enumeration and retained
    /// output are both bounded and fail closed.
    pub fn homogeneous_monomials_of_degree(
        &self,
        degree: &Degree,
        positivity: &PositiveGrading,
        max_count: usize,
    ) -> Result<Vec<Polynomial<R>>, PolynomialError> {
        if positivity.grading() != self.grading.as_ref() {
            return Err(ContextMismatchError {
                expected: self.grading.to_string(),
                actual: positivity.grading().to_string(),
            }
            .into());
        }
        let projected = positivity.project(degree).map_err(grading_as_polynomial)?;
        if projected.is_negative() {
            return Ok(Vec::new());
        }
        let projected = projected
            .to_u64_checked()
            .ok_or(PolynomialError::ResourceLimit {
                resource: "positive-functional target degree",
                limit: u64::MAX as usize,
            })?;
        let weights = positivity
            .weights()
            .iter()
            .map(|weight| {
                weight.to_u64_checked().or_else(|| {
                    projected
                        .checked_add(1)
                        .filter(|_| weight > &Integer::from(projected))
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(PolynomialError::ResourceLimit {
                resource: "positive-functional variable weight",
                limit: u64::MAX as usize,
            })?;
        let mut candidates = Vec::new();
        let mut exponent = vec![0_u64; self.variables.len()];
        enumerate_exact_weighted(
            &weights,
            0,
            projected,
            &mut exponent,
            &mut candidates,
            MAX_GRADED_PIECE_CANDIDATES,
        )
        .map_err(|error| match error {
            PolynomialError::ResourceLimit {
                resource: "monomial count",
                limit: MAX_GRADED_PIECE_CANDIDATES,
            } => PolynomialError::ResourceLimit {
                resource: "positive-functional candidate count",
                limit: MAX_GRADED_PIECE_CANDIDATES,
            },
            error => error,
        })?;
        let mut exponents = Vec::new();
        for exponent in candidates {
            let exponent = exponent.into_iter().map(Integer::from).collect::<Vec<_>>();
            if self
                .grading
                .degree(&exponent)
                .map_err(grading_as_polynomial)?
                == *degree
            {
                if exponents.len() == max_count {
                    return Err(PolynomialError::ResourceLimit {
                        resource: "homogeneous monomial count",
                        limit: max_count,
                    });
                }
                exponents.push(exponent);
            }
        }
        exponents.sort_by(|left, right| self.term_order.compare(right, left));
        exponents
            .into_iter()
            .map(|exponent| self.from_terms([(exponent, self.coefficient_ring.one())]))
            .collect()
    }

    /// Enumerate every monomial retained by one explicit finite region.
    ///
    /// The region uses the same strict univariate, total-degree, box, and
    /// weighted-degree shapes as fixed-truncation series. Results are
    /// greatest-to-least in this ring's term order, with an explicit retained
    /// output cap.
    pub fn monomials_in_region(
        &self,
        region: &SeriesTruncation,
        max_count: usize,
    ) -> Result<Vec<Polynomial<R>>, PolynomialError> {
        region
            .validate_arity(self.variables.len())
            .map_err(truncation_as_polynomial)?;
        let mut exponents = Vec::new();
        let mut exponent = vec![0_u64; self.variables.len()];
        match region.mode() {
            SeriesTruncationMode::Univariate => enumerate_box(
                region
                    .bound()
                    .map(|bound| vec![bound])
                    .as_deref()
                    .ok_or(PolynomialError::ArithmeticInvariant)?,
                0,
                &mut exponent,
                &mut exponents,
                max_count,
            )?,
            SeriesTruncationMode::TotalDegree => enumerate_weighted_below(
                &vec![1; self.variables.len()],
                0,
                region
                    .bound()
                    .and_then(|bound| bound.checked_sub(1))
                    .ok_or(PolynomialError::ArithmeticInvariant)?,
                &mut exponent,
                &mut exponents,
                max_count,
            )?,
            SeriesTruncationMode::Box => enumerate_box(
                region
                    .bounds()
                    .ok_or(PolynomialError::ArithmeticInvariant)?,
                0,
                &mut exponent,
                &mut exponents,
                max_count,
            )?,
            SeriesTruncationMode::WeightedDegree => enumerate_weighted_below(
                region
                    .weights()
                    .ok_or(PolynomialError::ArithmeticInvariant)?,
                0,
                region
                    .bound()
                    .and_then(|bound| bound.checked_sub(1))
                    .ok_or(PolynomialError::ArithmeticInvariant)?,
                &mut exponent,
                &mut exponents,
                max_count,
            )?,
        }
        self.monomials_from_exponents(exponents)
    }

    fn monomials_from_exponents(
        &self,
        mut exponents: Vec<Vec<u64>>,
    ) -> Result<Vec<Polynomial<R>>, PolynomialError> {
        exponents.sort_by(|left, right| {
            let left = left.iter().copied().map(Integer::from).collect::<Vec<_>>();
            let right = right.iter().copied().map(Integer::from).collect::<Vec<_>>();
            self.term_order.compare(&right, &left)
        });
        exponents
            .into_iter()
            .map(|exponent| self.from_u64_terms([(exponent, self.coefficient_ring.one())]))
            .collect()
    }

    fn validate_exponent(&self, exponent: &[Integer]) -> Result<(), PolynomialError> {
        if exponent.len() != self.variables.len() {
            return Err(PolynomialError::ExponentArity {
                expected: self.variables.len(),
                actual: exponent.len(),
            });
        }
        if let Some(variable) = exponent.iter().position(|power| power.sign() < 0) {
            return Err(PolynomialError::NegativeExponent { variable });
        }
        Ok(())
    }

    fn require(&self, value: &Polynomial<R>) -> Result<(), ContextMismatchError> {
        if self == &value.ring {
            Ok(())
        } else {
            Err(ContextMismatchError {
                expected: Self::TYPE_ID.to_owned(),
                actual: "unequal polynomial context".to_owned(),
            })
        }
    }

    fn add_polynomial(
        &self,
        left: &Polynomial<R>,
        right: &Polynomial<R>,
    ) -> Result<Polynomial<R>, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        if let Some(native) = left
            .native()
            .zip(right.native())
            .and_then(|(left, right)| left.add(right))
        {
            return Ok(self.polynomial_from_native(native));
        }
        let mut terms = left.terms_map().clone();
        for (exponent, coefficient) in right.terms_map() {
            let sum = if let Some(previous) = terms.remove(exponent) {
                self.coefficient_ring.add(&previous, coefficient)?
            } else {
                coefficient.clone()
            };
            if sum != self.coefficient_ring.zero() {
                terms.insert(exponent.clone(), sum);
            }
        }
        Ok(self.polynomial_from_canonical_terms(terms))
    }

    fn neg_polynomial(&self, value: &Polynomial<R>) -> Result<Polynomial<R>, ContextMismatchError> {
        self.require(value)?;
        if let Some(native) = value.native().and_then(|value| value.neg()) {
            return Ok(self.polynomial_from_native(native));
        }
        let terms = value
            .terms_map()
            .iter()
            .map(|(exponent, coefficient)| {
                self.coefficient_ring
                    .neg(coefficient)
                    .map(|coefficient| (exponent.clone(), coefficient))
            })
            .collect::<Result<_, _>>()?;
        Ok(self.polynomial_from_canonical_terms(terms))
    }

    fn sub_polynomial(
        &self,
        left: &Polynomial<R>,
        right: &Polynomial<R>,
    ) -> Result<Polynomial<R>, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        if let Some(native) = left
            .native()
            .zip(right.native())
            .and_then(|(left, right)| left.sub(right))
        {
            return Ok(self.polynomial_from_native(native));
        }
        self.add_polynomial(left, &self.neg_polynomial(right)?)
    }

    fn mul_polynomial(
        &self,
        left: &Polynomial<R>,
        right: &Polynomial<R>,
    ) -> Result<Polynomial<R>, ContextMismatchError> {
        self.require(left)?;
        self.require(right)?;
        if let Some(native) = left
            .native()
            .zip(right.native())
            .and_then(|(left, right)| left.mul(right))
        {
            return Ok(self.polynomial_from_native(native));
        }
        let work = left
            .terms_map()
            .len()
            .saturating_mul(right.terms_map().len());
        if work >= PACKED_TERM_PRODUCT_CROSSOVER {
            if let Some(terms) = packed_term_product(self, left.terms_map(), right.terms_map()) {
                return Ok(self.polynomial_from_canonical_terms(terms));
            }
        }
        let mut terms = BTreeMap::new();
        for (left_exponent, left_coefficient) in left.terms_map() {
            for (right_exponent, right_coefficient) in right.terms_map() {
                let exponent = left_exponent
                    .iter()
                    .zip(right_exponent)
                    .map(|(left, right)| left + right)
                    .collect::<Vec<_>>();
                let product = self
                    .coefficient_ring
                    .mul(left_coefficient, right_coefficient)?;
                let coefficient = if let Some(previous) = terms.remove(&exponent) {
                    self.coefficient_ring.add(&previous, &product)?
                } else {
                    product
                };
                if coefficient != self.coefficient_ring.zero() {
                    terms.insert(exponent, coefficient);
                }
            }
        }
        Ok(self.polynomial_from_canonical_terms(terms))
    }

    fn pow_polynomial_integer(
        &self,
        value: &Polynomial<R>,
        exponent: &Integer,
    ) -> Result<Polynomial<R>, PolynomialError> {
        if exponent.sign() < 0 {
            return Err(PolynomialError::NegativeExponent { variable: 0 });
        }
        self.require(value)?;
        if let Some(native) = exponent
            .to_u64_checked()
            .and_then(|exponent| value.native()?.pow_u64(exponent))
        {
            return Ok(self.polynomial_from_native(native));
        }
        let mut result = self.one();
        let mut base = value.clone();
        for bit in bits(exponent)? {
            if bit {
                result = self.mul_polynomial(&result, &base)?;
            }
            base = self.mul_polynomial(&base, &base)?;
        }
        Ok(result)
    }

    fn natural_scale(
        &self,
        coefficient: &R::Element,
        factor: &Integer,
    ) -> Result<R::Element, PolynomialError> {
        let mut result = self.coefficient_ring.zero();
        let mut addend = coefficient.clone();
        for bit in bits(factor)? {
            if bit {
                result = self.coefficient_ring.add(&result, &addend)?;
            }
            addend = self.coefficient_ring.add(&addend, &addend)?;
        }
        Ok(result)
    }

    fn pow_coefficient_integer(
        &self,
        value: &R::Element,
        exponent: &Integer,
    ) -> Result<R::Element, PolynomialError> {
        let mut result = self.coefficient_ring.one();
        let mut base = value.clone();
        for bit in bits(exponent)? {
            if bit {
                result = self.coefficient_ring.mul(&result, &base)?;
            }
            base = self.coefficient_ring.mul(&base, &base)?;
        }
        Ok(result)
    }
}

fn packed_term_product<R: CommutativeRing>(
    ring: &PolynomialRing<R>,
    left: &BTreeMap<Vec<Integer>, R::Element>,
    right: &BTreeMap<Vec<Integer>, R::Element>,
) -> Option<BTreeMap<Vec<Integer>, R::Element>> {
    let arity = ring.variables.len();
    let order = kernel_monomial_order(&ring.term_order)?;
    let capacity = left
        .len()
        .saturating_add(right.len())
        .saturating_mul(4)
        .max(64);
    let mut arena = poly_core::MonomialArena::with_capacity(arity, order, capacity);
    let lane = PolyCoreRingLane {
        ring: &ring.coefficient_ring,
    };
    let mut convert = |terms: &BTreeMap<Vec<Integer>, R::Element>| {
        let terms = terms
            .iter()
            .map(|(exponent, coefficient)| {
                let exponent = exponent
                    .iter()
                    .map(|value| {
                        value
                            .to_u64_checked()
                            .and_then(|value| u32::try_from(value).ok())
                    })
                    .collect::<Option<Vec<_>>>()?;
                arena
                    .intern_exponents(&exponent)
                    .ok()
                    .map(|monomial| (monomial, coefficient.clone()))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(poly_core::ArenaPolynomial::new(&arena, terms, &lane))
    };
    let left = convert(left)?;
    let right = convert(right)?;
    let product = left
        .multiply_adaptive(&right, &mut arena, &lane, None)
        .ok()?;
    Some(
        product
            .iter()
            .map(|(monomial, coefficient)| {
                (
                    arena
                        .get(monomial)
                        .to_u32_vec()
                        .into_iter()
                        .map(Integer::from)
                        .collect(),
                    coefficient.clone(),
                )
            })
            .collect(),
    )
}

pub(super) fn kernel_monomial_order(order: &MonomialOrder) -> Option<poly_core::MonomialOrder> {
    match order.standard_name()? {
        "lex" => Some(poly_core::MonomialOrder::Lexicographic),
        "grlex" => Some(poly_core::MonomialOrder::GradedLexicographic),
        "grevlex" => Some(poly_core::MonomialOrder::GradedReverseLexicographic),
        _ => None,
    }
}

impl<R> Serialize for PolynomialRing<R>
where
    R: CommutativeRing + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PolynomialRing", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("coefficient_ring", &self.coefficient_ring)?;
        state.serialize_field("variables", self.variables.as_ref())?;
        state.serialize_field("term_order", &self.term_order)?;
        let grading = self
            .weighted_grading()
            .map(PolynomialGradingWire::Weighted)
            .unwrap_or_else(|| PolynomialGradingWire::FreeAbelian(self.grading.as_ref().clone()));
        state.serialize_field("grading", &grading)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for PolynomialRing<R>
where
    R: CommutativeRing + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PolynomialRingWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::with_grading(
            wire.coefficient_ring,
            wire.variables,
            wire.term_order,
            wire.grading.into_grading(),
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<R: CommutativeRing> PartialEq for PolynomialRing<R> {
    fn eq(&self, other: &Self) -> bool {
        self.coefficient_ring == other.coefficient_ring
            && self.variables == other.variables
            && self.term_order == other.term_order
            && self.grading == other.grading
    }
}

impl<R: CommutativeRing> Eq for PolynomialRing<R> {}

impl<R: CommutativeRing> fmt::Debug for PolynomialRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PolynomialRing")
            .field("coefficient_type", &R::TYPE_ID)
            .field("variables", &self.variables)
            .field("term_order", &self.term_order)
            .field("grading", &self.grading)
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Display for PolynomialRing<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PolynomialRing<{}>[{}] ({}, {})",
            R::TYPE_ID,
            self.variables.join(", "),
            self.term_order,
            self.grading
        )
    }
}

impl<R: CommutativeRing> SemanticType for PolynomialRing<R> {
    const TYPE_ID: &'static str = "noether/commalg/PolynomialRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, R::TYPE_ID.as_bytes())?;
        append_component(&mut output, &self.coefficient_ring.canonical_bytes()?)?;
        append_length(&mut output, self.variables.len())?;
        for variable in self.variables.iter() {
            append_component(&mut output, variable.as_bytes())?;
        }
        append_component(&mut output, &self.term_order.canonical_bytes()?)?;
        if let Some(grading) = self.weighted_grading() {
            // Preserve PolynomialRing@1 identity for every existing positive
            // rank-one context.
            append_component(&mut output, &grading.canonical_bytes()?)?;
        } else {
            let mut grading = b"free-abelian-grading@1\0".to_vec();
            grading.extend_from_slice(&self.grading.canonical_bytes()?);
            append_component(&mut output, &grading)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &POLYNOMIAL_RING_DESCRIPTION
    }
}

impl<R: CommutativeRing> Ring for PolynomialRing<R> {
    type Element = Polynomial<R>;

    fn contains(&self, value: &Self::Element) -> bool {
        self == &value.ring
    }

    fn zero(&self) -> Self::Element {
        self.polynomial_from_canonical_terms(BTreeMap::new())
    }

    fn is_zero(&self, value: &Self::Element) -> bool {
        value.is_zero()
    }

    fn one(&self) -> Self::Element {
        let mut terms = BTreeMap::new();
        terms.insert(
            vec![Integer::zero(); self.variables.len()],
            self.coefficient_ring.one(),
        );
        self.polynomial_from_canonical_terms(terms)
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.add_polynomial(a, b)
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.sub_polynomial(a, b)
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        self.mul_polynomial(a, b)
    }

    fn sum_products(
        &self,
        products: &[(&Self::Element, &Self::Element)],
    ) -> Result<Self::Element, ContextMismatchError> {
        for &(left, right) in products {
            self.require(left)?;
            self.require(right)?;
        }
        if products.is_empty() {
            return Ok(self.zero());
        }

        let native_products = products
            .iter()
            .map(|&(left, right)| left.native().zip(right.native()))
            .collect::<Option<Vec<_>>>();
        if let Some(native) = native_products
            .as_deref()
            .and_then(super::polynomial_flint::FlintPolynomial::sum_products)
        {
            return Ok(self.polynomial_from_native(native));
        }

        let mut sum = self.zero();
        for &(left, right) in products {
            let product = self.mul_polynomial(left, right)?;
            sum = self.add_polynomial(&sum, &product)?;
        }
        Ok(sum)
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        self.neg_polynomial(value)
    }
}

impl<R: CommutativeRing> CommutativeRing for PolynomialRing<R> {}

impl<R: IntegralDomain> IntegralDomain for PolynomialRing<R> {}

impl<R: IntegralDomain + UnitInversion> UnitInversion for PolynomialRing<R> {
    fn inverse_unit(&self, value: &Self::Element) -> Result<Self::Element, UnitError> {
        self.require(value)?;
        let terms = value.terms_map();
        if terms.is_empty() {
            return Err(DivisionByZeroError.into());
        }
        if terms.len() != 1 {
            return Err(UnitError::NotAUnit {
                context: Self::TYPE_ID,
            });
        }
        let (exponent, coefficient) = terms.first_key_value().ok_or(DivisionByZeroError)?;
        if exponent.iter().any(|power| !power.is_zero()) {
            return Err(UnitError::NotAUnit {
                context: Self::TYPE_ID,
            });
        }
        let inverse = self.coefficient_ring.inverse_unit(coefficient)?;
        self.constant(inverse).map_err(|_| UnitError::NotAUnit {
            context: Self::TYPE_ID,
        })
    }
}

struct FlintBackedPolynomial<E> {
    native: super::polynomial_flint::FlintPolynomial,
    terms: OnceLock<BTreeMap<Vec<Integer>, E>>,
}

enum PolynomialStorage<E> {
    Generic(Arc<BTreeMap<Vec<Integer>, E>>),
    Flint(Arc<FlintBackedPolynomial<E>>),
}

impl<E> Clone for PolynomialStorage<E> {
    fn clone(&self) -> Self {
        match self {
            Self::Generic(terms) => Self::Generic(Arc::clone(terms)),
            Self::Flint(polynomial) => Self::Flint(Arc::clone(polynomial)),
        }
    }
}

/// A sparse polynomial carrying its complete ring context.
///
/// Supported coefficient contexts keep an immutable FLINT multivariate
/// polynomial as their primary representation. The canonical sparse term view
/// is materialized lazily only for inspection, serialization, or a generic
/// algorithm. Unsupported contexts retain the generic sparse representation.
#[derive(Clone)]
pub struct Polynomial<R: CommutativeRing> {
    ring: PolynomialRing<R>,
    storage: PolynomialStorage<R::Element>,
}

#[derive(Serialize)]
struct PolynomialTermRef<'a, E> {
    exponent: &'a [Integer],
    coefficient: &'a E,
}

#[derive(Deserialize)]
#[serde(bound(deserialize = "E: Deserialize<'de>"))]
struct PolynomialTermWire<E> {
    exponent: Vec<Integer>,
    coefficient: E,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: CommutativeRing + Deserialize<'de>, E: Deserialize<'de>")
)]
struct PolynomialWire<R: CommutativeRing, E> {
    #[serde(rename = "type")]
    type_id: String,
    ring: PolynomialRing<R>,
    terms: Vec<PolynomialTermWire<E>>,
}

impl<R: CommutativeRing> Polynomial<R> {
    fn terms_map(&self) -> &BTreeMap<Vec<Integer>, R::Element> {
        match &self.storage {
            PolynomialStorage::Generic(terms) => terms,
            PolynomialStorage::Flint(polynomial) => polynomial
                .terms
                .get_or_init(|| polynomial.native.to_terms(&self.ring.coefficient_ring)),
        }
    }

    pub(super) fn native(&self) -> Option<&super::polynomial_flint::FlintPolynomial> {
        match &self.storage {
            PolynomialStorage::Generic(_) => None,
            PolynomialStorage::Flint(polynomial) => Some(&polynomial.native),
        }
    }

    #[cfg(test)]
    pub(super) fn uses_flint_storage(&self) -> bool {
        matches!(self.storage, PolynomialStorage::Flint(_))
    }

    #[cfg(test)]
    pub(super) fn sparse_view_is_materialized(&self) -> bool {
        match &self.storage {
            PolynomialStorage::Generic(_) => true,
            PolynomialStorage::Flint(polynomial) => polynomial.terms.get().is_some(),
        }
    }

    #[cfg(test)]
    fn shares_storage_with(&self, other: &Self) -> bool {
        match (&self.storage, &other.storage) {
            (PolynomialStorage::Generic(left), PolynomialStorage::Generic(right)) => {
                Arc::ptr_eq(left, right)
            }
            (PolynomialStorage::Flint(left), PolynomialStorage::Flint(right)) => {
                Arc::ptr_eq(left, right)
            }
            _ => false,
        }
    }

    /// Polynomial-ring context.
    pub fn ring(&self) -> &PolynomialRing<R> {
        &self.ring
    }

    /// Number of nonzero terms.
    pub fn nonzero_term_count(&self) -> usize {
        self.native()
            .map_or_else(|| self.terms_map().len(), |native| native.len())
    }

    /// Whether this polynomial is zero.
    pub fn is_zero(&self) -> bool {
        self.native()
            .map_or_else(|| self.terms_map().is_empty(), |native| native.is_zero())
    }

    /// Whether this polynomial is one.
    pub fn is_one(&self) -> bool {
        self == &self.ring.one()
    }

    /// Nonzero terms from greatest to least in the context's term order.
    pub fn terms(&self) -> impl Iterator<Item = (&[Integer], &R::Element)> {
        let mut terms = self
            .terms_map()
            .iter()
            .map(|(exponent, coefficient)| (exponent.as_slice(), coefficient))
            .collect::<Vec<_>>();
        terms.sort_by(|(left, _), (right, _)| self.ring.term_order.compare(right, left));
        terms.into_iter()
    }

    /// Apply one checked scalar map to every coefficient while preserving the
    /// variables, monomial order, grading, and exponents.
    pub fn map_coefficients<M>(
        &self,
        map: &M,
    ) -> Result<Polynomial<M::Target>, CoefficientMappingError>
    where
        M: CoefficientMap<Source = R>,
        R: fmt::Display,
    {
        if self.ring.coefficient_ring() != map.source() {
            return Err(ContextMismatchError {
                expected: map.source().to_string(),
                actual: self.ring.coefficient_ring().to_string(),
            }
            .into());
        }
        let target_ring = PolynomialRing::with_grading(
            map.target().clone(),
            self.ring.variables().iter().cloned(),
            self.ring.term_order(),
            self.ring.grading().clone(),
        )
        .map_err(|error| CoefficientMappingError::TargetContext(error.to_string()))?;
        let mut terms = Vec::with_capacity(self.nonzero_term_count());
        for (exponent, coefficient) in self.terms() {
            terms.push((
                exponent.to_vec(),
                map.apply(coefficient)
                    .map_err(|error| CoefficientMappingError::Coefficient(error.to_string()))?,
            ));
        }
        target_ring
            .from_terms(terms)
            .map_err(|error| CoefficientMappingError::Value(error.to_string()))
    }

    /// Coefficient of one exact exponent vector.
    pub fn coefficient(&self, exponent: &[Integer]) -> Result<R::Element, PolynomialError> {
        self.ring.validate_exponent(exponent)?;
        if let Some(native) = self.native() {
            return Ok(native.coefficient(&self.ring.coefficient_ring, exponent));
        }
        Ok(self
            .terms_map()
            .get(exponent)
            .cloned()
            .unwrap_or_else(|| self.ring.coefficient_ring.zero()))
    }

    /// Coefficient of one power of `variable`, as a polynomial in all other
    /// variables.
    ///
    /// The remaining variables retain their order, grading weights, and the
    /// induced monomial order. A one-variable source has no polynomial ring of
    /// remaining variables in the current context model and is rejected.
    pub fn coefficient_polynomial(
        &self,
        variable: usize,
        exponent: &Integer,
    ) -> Result<Polynomial<R>, PolynomialError> {
        let target = self.coefficient_polynomial_ring(variable)?;
        if exponent.sign() < 0 {
            return Err(PolynomialError::NegativeExponent { variable });
        }
        target.from_terms(
            self.terms_map()
                .iter()
                .filter(|(powers, _)| &powers[variable] == exponent)
                .map(|(powers, coefficient)| {
                    let mut powers = powers.clone();
                    powers.remove(variable);
                    (powers, coefficient.clone())
                }),
        )
    }

    /// Dense coefficients in `variable`, from exponent zero through its
    /// degree, as polynomials in all other variables.
    ///
    /// `max_count` is checked against the selected-variable degree before the
    /// dense result is allocated. The zero polynomial returns an empty list.
    pub fn coefficient_polynomials(
        &self,
        variable: usize,
        max_count: usize,
    ) -> Result<Vec<Polynomial<R>>, PolynomialError> {
        let target = self.coefficient_polynomial_ring(variable)?;
        let mut grouped = BTreeMap::<Integer, Vec<(Vec<Integer>, R::Element)>>::new();
        for (powers, coefficient) in self.terms_map() {
            let mut remaining = powers.clone();
            let power = remaining.remove(variable);
            grouped
                .entry(power)
                .or_default()
                .push((remaining, coefficient.clone()));
        }
        let Some(degree) = grouped.keys().next_back() else {
            return Ok(Vec::new());
        };
        let required = degree + &Integer::one();
        if required > Integer::from(max_count) {
            return Err(PolynomialError::ResourceLimit {
                resource: "selected-variable coefficient count",
                limit: max_count,
            });
        }
        let count = required
            .to_u64_checked()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or(PolynomialError::ResourceLimit {
                resource: "selected-variable coefficient count",
                limit: max_count,
            })?;
        (0..count)
            .map(|power| {
                target.from_terms(grouped.remove(&Integer::from(power)).unwrap_or_default())
            })
            .collect()
    }

    /// Checked univariate view in one selected variable.
    ///
    /// Every exponent outside `variable` must vanish. The returned polynomial
    /// has a one-variable context with the selected name, grading weight, and
    /// induced monomial order.
    pub fn univariate_view(&self, variable: usize) -> Result<Polynomial<R>, PolynomialError> {
        self.validate_variable(variable)?;
        if self.ring.variables.len() == 1 {
            return Ok(self.clone());
        }
        for powers in self.terms_map().keys() {
            if let Some(other) = powers
                .iter()
                .enumerate()
                .find_map(|(other, power)| (other != variable && !power.is_zero()).then_some(other))
            {
                return Err(PolynomialError::DependsOnOtherVariable {
                    selected: variable,
                    other,
                });
            }
        }
        let target = PolynomialRing::with_grading(
            self.ring.coefficient_ring.clone(),
            [self.ring.variables[variable].clone()],
            self.ring.term_order.on_variable(variable),
            self.ring
                .grading
                .on_variables([variable])
                .map_err(grading_as_polynomial)?,
        )
        .map_err(|_| PolynomialError::ArithmeticInvariant)?;
        target.from_terms(
            self.terms_map()
                .iter()
                .map(|(powers, coefficient)| (vec![powers[variable].clone()], coefficient.clone())),
        )
    }

    fn validate_variable(&self, variable: usize) -> Result<(), PolynomialError> {
        if variable >= self.ring.variables.len() {
            Err(PolynomialError::VariableIndex {
                variable,
                arity: self.ring.variables.len(),
            })
        } else {
            Ok(())
        }
    }

    fn coefficient_polynomial_ring(
        &self,
        variable: usize,
    ) -> Result<PolynomialRing<R>, PolynomialError> {
        self.validate_variable(variable)?;
        if self.ring.variables.len() == 1 {
            return Err(PolynomialError::NoRemainingVariables { selected: variable });
        }
        let variables = self
            .ring
            .variables
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != variable)
            .map(|(_, name)| name.clone());
        let retained_variables = self
            .ring
            .variables
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != variable)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        PolynomialRing::with_grading(
            self.ring.coefficient_ring.clone(),
            variables,
            self.ring.term_order.without_variable(variable),
            self.ring
                .grading
                .on_variables(retained_variables)
                .map_err(grading_as_polynomial)?,
        )
        .map_err(|_| PolynomialError::ArithmeticInvariant)
    }

    /// Constant coefficient.
    pub fn constant_coefficient(&self) -> R::Element {
        if let Some(native) = self.native() {
            return native.coefficient(
                &self.ring.coefficient_ring,
                &vec![Integer::zero(); self.ring.variables.len()],
            );
        }
        self.terms_map()
            .get(&vec![Integer::zero(); self.ring.variables.len()])
            .cloned()
            .unwrap_or_else(|| self.ring.coefficient_ring.zero())
    }

    /// Leading coefficient in the context's monomial order, or `None` for
    /// zero.
    pub fn leading_coefficient(&self) -> Option<R::Element> {
        self.leading_term_owned()
            .map(|(_, coefficient)| coefficient)
    }

    pub(crate) fn leading_term_owned(&self) -> Option<(Vec<Integer>, R::Element)> {
        if let Some(native) = self.native() {
            return native.leading_term(&self.ring.coefficient_ring);
        }
        self.terms_map()
            .iter()
            .max_by(|(left, _), (right, _)| self.ring.term_order.compare(left, right))
            .map(|(exponent, coefficient)| (exponent.clone(), coefficient.clone()))
    }

    /// Serialize this polynomial with its complete ring and ordered nonzero terms.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and revalidate the ring, exponents, coefficients, and term order.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Degree in one variable, or weighted total degree when `variable` is
    /// `None`. Zero has no degree.
    pub fn degree(&self, variable: Option<usize>) -> Result<Option<Integer>, PolynomialError> {
        if let Some(variable) = variable {
            if variable >= self.ring.variables.len() {
                return Err(PolynomialError::VariableIndex {
                    variable,
                    arity: self.ring.variables.len(),
                });
            }
            return Ok(self
                .terms_map()
                .keys()
                .map(|exponent| exponent[variable].clone())
                .max());
        }
        self.require_rank_one_grading()?;
        self.terms_map()
            .keys()
            .map(|exponent| self.ring.grading.rank_one_degree(exponent))
            .collect::<Result<Vec<_>, _>>()
            .map(|degrees| degrees.into_iter().max())
            .map_err(grading_as_polynomial)
    }

    /// Degree shared by every nonzero monomial, or `None` when this
    /// polynomial is zero or inhomogeneous.
    pub fn homogeneous_degree(&self) -> Result<Option<Integer>, PolynomialError> {
        self.require_rank_one_grading()?;
        let mut degrees = self
            .terms_map()
            .keys()
            .map(|exponent| self.ring.grading.rank_one_degree(exponent));
        let Some(first) = degrees.next().transpose().map_err(grading_as_polynomial)? else {
            return Ok(None);
        };
        for degree in degrees {
            if degree.map_err(grading_as_polynomial)? != first {
                return Ok(None);
            }
        }
        Ok(Some(first))
    }

    /// Split into nonzero homogeneous components in increasing degree.
    pub fn homogeneous_components(&self) -> Result<Vec<(Integer, Polynomial<R>)>, PolynomialError> {
        self.require_rank_one_grading()?;
        let mut grouped = BTreeMap::<Integer, Vec<(Vec<Integer>, R::Element)>>::new();
        for (exponent, coefficient) in self.terms_map() {
            let degree = self
                .ring
                .grading
                .rank_one_degree(exponent)
                .map_err(grading_as_polynomial)?;
            grouped
                .entry(degree)
                .or_default()
                .push((exponent.clone(), coefficient.clone()));
        }
        grouped
            .into_iter()
            .map(|(degree, terms)| self.ring.from_terms(terms).map(|value| (degree, value)))
            .collect()
    }

    /// Degree shared by every nonzero monomial in the full free-abelian
    /// grading, or `None` when this polynomial is zero or inhomogeneous.
    pub fn multihomogeneous_degree(&self) -> Result<Option<Degree>, PolynomialError> {
        let mut degrees = self
            .terms_map()
            .keys()
            .map(|exponent| self.ring.grading.degree(exponent));
        let Some(first) = degrees.next().transpose().map_err(grading_as_polynomial)? else {
            return Ok(None);
        };
        for degree in degrees {
            if degree.map_err(grading_as_polynomial)? != first {
                return Ok(None);
            }
        }
        Ok(Some(first))
    }

    /// Split into nonzero components in the full free-abelian grading.
    pub fn multihomogeneous_components(
        &self,
    ) -> Result<Vec<(Degree, Polynomial<R>)>, PolynomialError> {
        let mut grouped = BTreeMap::<Degree, Vec<(Vec<Integer>, R::Element)>>::new();
        for (exponent, coefficient) in self.terms_map() {
            let degree = self
                .ring
                .grading
                .degree(exponent)
                .map_err(grading_as_polynomial)?;
            grouped
                .entry(degree)
                .or_default()
                .push((exponent.clone(), coefficient.clone()));
        }
        grouped
            .into_iter()
            .map(|(degree, terms)| self.ring.from_terms(terms).map(|value| (degree, value)))
            .collect()
    }

    /// Multiply every coefficient by one coefficient-ring value.
    pub fn scalar_mul(&self, scalar: &R::Element) -> Result<Self, PolynomialError> {
        if !self.ring.coefficient_ring.contains(scalar) {
            return Err(PolynomialError::CoefficientOutsideContext);
        }
        if let Some(native) = self
            .native()
            .and_then(|native| native.scalar_mul(&self.ring.coefficient_ring, scalar))
        {
            return Ok(self.ring.polynomial_from_native(native));
        }
        self.ring.from_terms(
            self.terms_map()
                .iter()
                .map(|(exponent, coefficient)| {
                    self.ring
                        .coefficient_ring
                        .mul(coefficient, scalar)
                        .map(|coefficient| (exponent.clone(), coefficient))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    /// Raise to an arbitrary-size nonnegative integer power.
    pub fn pow_integer(&self, exponent: &Integer) -> Result<Self, PolynomialError> {
        self.ring.pow_polynomial_integer(self, exponent)
    }

    /// Raise to a machine-word nonnegative power.
    pub fn pow(&self, exponent: u64) -> Result<Self, PolynomialError> {
        self.pow_integer(&Integer::from(exponent))
    }

    /// Formal derivative with respect to one variable.
    pub fn derivative(&self, variable: usize) -> Result<Self, PolynomialError> {
        if variable >= self.ring.variables.len() {
            return Err(PolynomialError::VariableIndex {
                variable,
                arity: self.ring.variables.len(),
            });
        }
        if let Some(native) = self.native().and_then(|native| native.derivative(variable)) {
            return Ok(self.ring.polynomial_from_native(native));
        }
        let mut terms = Vec::new();
        for (exponent, coefficient) in self.terms_map() {
            if exponent[variable].is_zero() {
                continue;
            }
            let mut derivative_exponent = exponent.clone();
            derivative_exponent[variable] = &derivative_exponent[variable] - &Integer::one();
            let derivative_coefficient =
                self.ring.natural_scale(coefficient, &exponent[variable])?;
            terms.push((derivative_exponent, derivative_coefficient));
        }
        self.ring.from_terms(terms)
    }

    /// Evaluate at one coefficient-ring value per variable.
    pub fn evaluate(&self, values: &[R::Element]) -> Result<R::Element, PolynomialError> {
        if values.len() != self.ring.variables.len() {
            return Err(PolynomialError::ImageArity {
                expected: self.ring.variables.len(),
                actual: values.len(),
            });
        }
        if values
            .iter()
            .any(|value| !self.ring.coefficient_ring.contains(value))
        {
            return Err(PolynomialError::CoefficientOutsideContext);
        }
        if let Some(result) = self
            .native()
            .and_then(|native| native.evaluate(&self.ring.coefficient_ring, values))
        {
            return Ok(result);
        }
        let mut result = self.ring.coefficient_ring.zero();
        for (exponent, coefficient) in self.terms_map() {
            let mut term = coefficient.clone();
            for (power, value) in exponent.iter().zip(values) {
                let factor = self.ring.pow_coefficient_integer(value, power)?;
                term = self.ring.coefficient_ring.mul(&term, &factor)?;
            }
            result = self.ring.coefficient_ring.add(&result, &term)?;
        }
        Ok(result)
    }

    /// Substitute one polynomial from the same context for each variable.
    pub fn substitute(&self, images: &[Self]) -> Result<Self, PolynomialError> {
        if images.len() != self.ring.variables.len() {
            return Err(PolynomialError::ImageArity {
                expected: self.ring.variables.len(),
                actual: images.len(),
            });
        }
        for image in images {
            self.ring.require(image)?;
        }
        if let Some(input) = self.native() {
            if let Some(images) = images
                .iter()
                .map(Polynomial::native)
                .collect::<Option<Vec<_>>>()
            {
                if let Some(result) = input.compose(&images) {
                    return Ok(self.ring.polynomial_from_native(result));
                }
            }
        }
        let mut result = self.ring.zero();
        for (exponent, coefficient) in self.terms_map() {
            let mut term = self.ring.constant(coefficient.clone())?;
            for (power, image) in exponent.iter().zip(images) {
                let factor = self.ring.pow_polynomial_integer(image, power)?;
                term = self.ring.mul_polynomial(&term, &factor)?;
            }
            result = self.ring.add_polynomial(&result, &term)?;
        }
        Ok(result)
    }

    fn require_rank_one_grading(&self) -> Result<(), PolynomialError> {
        let actual = self.ring.grading.rank();
        if actual == 1 {
            Ok(())
        } else {
            Err(PolynomialError::RankOneGradingRequired { actual })
        }
    }
}

impl Polynomial<ResidueRing> {
    pub(crate) fn visit_native_residue_terms(&self, visitor: impl FnMut(&[u64], Integer)) -> bool {
        self.native()
            .and_then(|native| native.visit_residue_terms(visitor))
            .is_some()
    }
}

impl<R> Serialize for Polynomial<R>
where
    R: CommutativeRing + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let terms = self
            .terms()
            .map(|(exponent, coefficient)| PolynomialTermRef {
                exponent,
                coefficient,
            })
            .collect::<Vec<_>>();
        let mut state = serializer.serialize_struct("Polynomial", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ring", &self.ring)?;
        state.serialize_field("terms", &terms)?;
        state.end()
    }
}

impl<'de, R> Deserialize<'de> for Polynomial<R>
where
    R: CommutativeRing + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PolynomialWire::<R, R::Element>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let mut previous: Option<&[Integer]> = None;
        for term in &wire.terms {
            wire.ring
                .validate_exponent(&term.exponent)
                .map_err(serde::de::Error::custom)?;
            if !wire.ring.coefficient_ring.contains(&term.coefficient) {
                return Err(serde::de::Error::custom(
                    "polynomial coefficient lies outside its declared context",
                ));
            }
            if term.coefficient == wire.ring.coefficient_ring.zero() {
                return Err(serde::de::Error::custom(
                    "serialized polynomial terms must have nonzero coefficients",
                ));
            }
            if let Some(previous) = previous {
                if wire.ring.term_order.compare(previous, &term.exponent) != Ordering::Greater {
                    return Err(serde::de::Error::custom(
                        "serialized polynomial terms are not in strict descending term order",
                    ));
                }
            }
            previous = Some(&term.exponent);
        }
        wire.ring
            .from_terms(
                wire.terms
                    .into_iter()
                    .map(|term| (term.exponent, term.coefficient)),
            )
            .map_err(serde::de::Error::custom)
    }
}

impl<R: CommutativeRing> Polynomial<R> {
    /// Divide by a monic univariate polynomial over any commutative ring.
    ///
    /// No coefficient inversion is used. Returns `(q, r)` with `self = q*d+r`
    /// and `r = 0` or `degree(r) < degree(d)`, including over rings with zero
    /// divisors and fixed-precision coefficient contexts.
    pub fn monic_div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolynomialError> {
        self.ring.require(divisor)?;
        let arity = self.ring.variables.len();
        if arity != 1 {
            return Err(PolynomialError::NotUnivariate { arity });
        }
        let (exponent, leading) = divisor
            .leading_term_owned()
            .ok_or(PolynomialError::DivisionByZero)?;
        if leading != self.ring.coefficient_ring.one() {
            return Err(PolynomialError::NonMonicDivisor);
        }
        let coefficients = &self.ring.coefficient_ring;
        let zero = coefficients.zero();
        let mut quotient = BTreeMap::new();
        let mut remainder = self.terms_map().clone();
        // In one variable every admitted monomial order is degree order.
        // Update only the shifted divisor's support; rebuilding the entire
        // remainder and quotient per leading term makes dense division
        // quadratic in the dividend degree even for a fixed small divisor.
        while let Some((current, _)) = remainder.last_key_value() {
            if current[0] < exponent[0] {
                break;
            }
            let (current, coefficient) = remainder.pop_last().expect("nonempty remainder");
            let shift = &current[0] - &exponent[0];
            for (power, value) in divisor.terms_map() {
                if power == &exponent {
                    continue; // The monic leading term was removed by pop_last.
                }
                let product = coefficients.mul(&coefficient, value)?;
                if product == zero {
                    continue;
                }
                let power = vec![&power[0] + &shift];
                let value = match remainder.remove(&power) {
                    Some(previous) => coefficients.sub(&previous, &product)?,
                    None => coefficients.neg(&product)?,
                };
                if value != zero {
                    remainder.insert(power, value);
                }
            }
            quotient.insert(vec![shift], coefficient);
        }
        Ok((
            self.ring.polynomial_from_canonical_terms(quotient),
            self.ring.polynomial_from_canonical_terms(remainder),
        ))
    }
}

impl<R: Field> Polynomial<R> {
    /// Monic extended gcd of two univariate polynomials.
    ///
    /// Returns `(g, s, t)` satisfying `g = s*self + t*other`. For two zero
    /// inputs the canonical result is `(0, 1, 0)`.
    pub fn extended_gcd(&self, other: &Self) -> Result<(Self, Self, Self), PolynomialError> {
        self.ring.require(other)?;
        self.require_univariate()?;
        let (mut a, mut b) = (self.clone(), other.clone());
        let (mut s, mut u) = (self.ring.one(), self.ring.zero());
        let (mut t, mut v) = (self.ring.zero(), self.ring.one());
        while !b.is_zero() {
            let (q, r) = a.div_rem(&b)?;
            let next_s = (&s - &(&q * &u)?)?;
            let next_t = (&t - &(&q * &v)?)?;
            (a, b) = (b, r);
            (s, u) = (u, next_s);
            (t, v) = (v, next_t);
        }
        if let Some(leading) = a.leading_coefficient() {
            let inverse = self
                .ring
                .coefficient_ring
                .inverse(&leading)
                .map_err(|_| PolynomialError::ArithmeticInvariant)?;
            Ok((
                a.scalar_mul(&inverse)?,
                s.scalar_mul(&inverse)?,
                t.scalar_mul(&inverse)?,
            ))
        } else {
            Ok((a, s, t))
        }
    }

    /// Invert this univariate polynomial modulo a nonzero polynomial.
    ///
    /// The returned representative has degree less than the modulus. A
    /// nonconstant gcd produces [`PolynomialError::NotCoprime`].
    pub fn inverse_mod(&self, modulus: &Self) -> Result<Self, PolynomialError> {
        self.ring.require(modulus)?;
        if modulus.is_zero() {
            return Err(PolynomialError::DivisionByZero);
        }
        let (gcd, inverse, _) = self.extended_gcd(modulus)?;
        if gcd != self.ring.one() {
            return Err(PolynomialError::NotCoprime);
        }
        Ok(inverse.div_rem(modulus)?.1)
    }

    /// Normalize a nonzero polynomial to leading coefficient one.
    ///
    /// The zero polynomial is returned unchanged.
    pub fn monic(&self) -> Result<Self, PolynomialError> {
        let Some(leading) = self.leading_coefficient() else {
            return Ok(self.clone());
        };
        let inverse = self
            .ring
            .coefficient_ring
            .inverse(&leading)
            .map_err(|_| PolynomialError::ArithmeticInvariant)?;
        self.scalar_mul(&inverse)
    }

    /// Divide two univariate polynomials over a field.
    ///
    /// Returns `(quotient, remainder)` with
    /// `self = quotient * divisor + remainder` and either zero remainder or
    /// remainder degree strictly smaller than divisor degree.
    pub fn div_rem(&self, divisor: &Self) -> Result<(Self, Self), PolynomialError> {
        self.ring.require(divisor)?;
        self.require_univariate()?;
        if divisor.is_zero() {
            return Err(PolynomialError::DivisionByZero);
        }

        let (divisor_exponent, divisor_leading) = divisor
            .terms_map()
            .last_key_value()
            .ok_or(PolynomialError::DivisionByZero)?;
        let inverse_divisor_leading = self
            .ring
            .coefficient_ring
            .inverse(divisor_leading)
            .map_err(|_| PolynomialError::ArithmeticInvariant)?;
        let divisor_degree = &divisor_exponent[0];
        let mut quotient = self.ring.zero();
        let mut remainder = self.clone();

        while let Some((remainder_exponent, remainder_leading)) =
            remainder.terms_map().last_key_value()
        {
            if &remainder_exponent[0] < divisor_degree {
                break;
            }
            let quotient_exponent = &remainder_exponent[0] - divisor_degree;
            let quotient_coefficient = self
                .ring
                .coefficient_ring
                .mul(remainder_leading, &inverse_divisor_leading)?;
            let term = self
                .ring
                .from_terms([(vec![quotient_exponent], quotient_coefficient)])?;
            quotient = self.ring.add_polynomial(&quotient, &term)?;
            let multiple = self.ring.mul_polynomial(&term, divisor)?;
            remainder = self.ring.sub_polynomial(&remainder, &multiple)?;
        }

        Ok((quotient, remainder))
    }

    /// Divide exactly by one polynomial over a field.
    ///
    /// Native coefficient contexts use their multivariate divisibility
    /// kernels. The exact sparse floor performs single-divisor leading-term
    /// reduction in the ring's declared global monomial order.
    pub fn exact_div(&self, divisor: &Self) -> Result<Self, PolynomialError> {
        self.ring.require(divisor)?;
        if divisor.is_zero() {
            return Err(PolynomialError::DivisionByZero);
        }
        if let (Some(dividend), Some(divisor)) = (self.native(), divisor.native()) {
            if let Some((divides, quotient)) = dividend.divides(divisor) {
                return if divides {
                    Ok(self.ring.polynomial_from_native(quotient))
                } else {
                    Err(PolynomialError::NonExactDivision)
                };
            }
        }
        if let (Some(dividend), Some(divisor)) = (
            self.ring.materialize_native(self),
            self.ring.materialize_native(divisor),
        ) {
            if let Some((divides, quotient)) = dividend.divides(&divisor) {
                return if divides {
                    Ok(self.ring.polynomial_from_native(quotient))
                } else {
                    Err(PolynomialError::NonExactDivision)
                };
            }
        }

        let (divisor_exponent, divisor_leading) = divisor
            .leading_term_owned()
            .ok_or(PolynomialError::DivisionByZero)?;
        let inverse_divisor_leading = self
            .ring
            .coefficient_ring
            .inverse(&divisor_leading)
            .map_err(|_| PolynomialError::ArithmeticInvariant)?;
        let mut quotient = self.ring.zero();
        let mut remainder = self.clone();
        while let Some((remainder_exponent, remainder_leading)) = remainder.leading_term_owned() {
            if remainder_exponent
                .iter()
                .zip(&divisor_exponent)
                .any(|(available, required)| available < required)
            {
                return Err(PolynomialError::NonExactDivision);
            }
            let quotient_exponent = remainder_exponent
                .iter()
                .zip(&divisor_exponent)
                .map(|(available, required)| available - required)
                .collect::<Vec<_>>();
            let quotient_coefficient = self
                .ring
                .coefficient_ring
                .mul(&remainder_leading, &inverse_divisor_leading)?;
            let term = self
                .ring
                .from_terms([(quotient_exponent, quotient_coefficient)])?;
            quotient = self.ring.add_polynomial(&quotient, &term)?;
            let multiple = self.ring.mul_polynomial(&term, divisor)?;
            remainder = self.ring.sub_polynomial(&remainder, &multiple)?;
        }
        Ok(quotient)
    }

    /// Monic greatest common divisor of two univariate polynomials over a
    /// field, with `gcd(0, 0) = 0`.
    ///
    /// Native multivariate coefficient contexts also use their exact
    /// multivariate gcd kernel. The coefficient-generic sparse fallback is
    /// currently univariate and fails closed otherwise.
    pub fn gcd(&self, other: &Self) -> Result<Self, PolynomialError> {
        self.ring.require(other)?;
        if let (Some(left), Some(right)) = (self.native(), other.native()) {
            if let Some(gcd) = left.gcd(right) {
                return self.ring.polynomial_from_native(gcd).monic();
            }
        }
        if let (Some(left), Some(right)) = (
            self.ring.materialize_native(self),
            self.ring.materialize_native(other),
        ) {
            if let Some(gcd) = left.gcd(&right) {
                return self.ring.polynomial_from_native(gcd).monic();
            }
        }
        self.require_univariate()?;
        let mut left = self.clone();
        let mut right = other.clone();
        while !right.is_zero() {
            let (_, remainder) = left.div_rem(&right)?;
            left = right;
            right = remainder;
        }
        left.monic()
    }

    fn require_univariate(&self) -> Result<(), PolynomialError> {
        let arity = self.ring.variables.len();
        if arity == 1 {
            Ok(())
        } else {
            Err(PolynomialError::NotUnivariate { arity })
        }
    }
}

impl Polynomial<crate::IntegerRing> {
    /// Divide exactly by one univariate polynomial over the integers.
    ///
    /// The operation returns [`PolynomialError::NonExactDivision`] when the
    /// quotient does not have integer coefficients.
    pub fn exact_div(&self, divisor: &Self) -> Result<Self, PolynomialError> {
        self.ring.require(divisor)?;
        self.require_univariate()?;
        if divisor.is_zero() {
            return Err(PolynomialError::DivisionByZero);
        }
        if let (Some(dividend), Some(divisor)) = (self.native(), divisor.native()) {
            if let Some((divides, quotient)) = dividend.divides(divisor) {
                return if divides {
                    Ok(self.ring.polynomial_from_native(quotient))
                } else {
                    Err(PolynomialError::NonExactDivision)
                };
            }
        }

        let (divisor_exponent, divisor_leading) = divisor
            .terms_map()
            .last_key_value()
            .ok_or(PolynomialError::DivisionByZero)?;
        let divisor_degree = &divisor_exponent[0];
        let mut quotient = self.ring.zero();
        let mut remainder = self.clone();
        while let Some((remainder_exponent, remainder_leading)) =
            remainder.terms_map().last_key_value()
        {
            if &remainder_exponent[0] < divisor_degree
                || !remainder_leading.is_divisible(divisor_leading)
            {
                return Err(PolynomialError::NonExactDivision);
            }
            let quotient_exponent = &remainder_exponent[0] - divisor_degree;
            let quotient_coefficient = remainder_leading.div_floor(divisor_leading);
            let term = self
                .ring
                .from_terms([(vec![quotient_exponent], quotient_coefficient)])?;
            quotient = self.ring.add_polynomial(&quotient, &term)?;
            let multiple = self.ring.mul_polynomial(&term, divisor)?;
            remainder = self.ring.sub_polynomial(&remainder, &multiple)?;
        }
        Ok(quotient)
    }

    /// Greatest common divisor of two univariate integer polynomials.
    ///
    /// The result has positive leading coefficient and includes the gcd of
    /// the coefficient contents, with `gcd(0, 0) = 0`.
    pub fn gcd(&self, other: &Self) -> Result<Self, PolynomialError> {
        self.ring.require(other)?;
        self.require_univariate()?;
        if let (Some(left), Some(right)) = (self.native(), other.native()) {
            if let Some(gcd) = left.gcd(right) {
                return normalize_integer_polynomial(self.ring.polynomial_from_native(gcd));
            }
        }
        generic_integer_polynomial_gcd(self, other)
    }

    fn require_univariate(&self) -> Result<(), PolynomialError> {
        let arity = self.ring.variables.len();
        if arity == 1 {
            Ok(())
        } else {
            Err(PolynomialError::NotUnivariate { arity })
        }
    }
}

fn normalize_integer_polynomial(
    polynomial: Polynomial<crate::IntegerRing>,
) -> Result<Polynomial<crate::IntegerRing>, PolynomialError> {
    if polynomial
        .leading_coefficient()
        .is_some_and(|coefficient| coefficient.is_negative())
    {
        polynomial.scalar_mul(&Integer::from(-1))
    } else {
        Ok(polynomial)
    }
}

fn generic_integer_polynomial_gcd(
    left: &Polynomial<crate::IntegerRing>,
    right: &Polynomial<crate::IntegerRing>,
) -> Result<Polynomial<crate::IntegerRing>, PolynomialError> {
    if left.is_zero() && right.is_zero() {
        return Ok(left.ring.zero());
    }
    let content = |polynomial: &Polynomial<crate::IntegerRing>| {
        polynomial
            .terms_map()
            .values()
            .fold(Integer::zero(), |gcd, coefficient| gcd.gcd(coefficient))
    };
    let common_content = content(left).gcd(&content(right));
    let rational_ring = PolynomialRing::with_grading(
        Q,
        left.ring.variables.iter().cloned(),
        left.ring.term_order.clone(),
        left.ring.grading.as_ref().clone(),
    )
    .map_err(|_| PolynomialError::ArithmeticInvariant)?;
    let rational =
        |polynomial: &Polynomial<crate::IntegerRing>| {
            rational_ring.from_terms(polynomial.terms_map().iter().map(
                |(exponent, coefficient)| {
                    (exponent.clone(), crate::Rational::from(coefficient.clone()))
                },
            ))
        };
    let rational_gcd = rational(left)?.gcd(&rational(right)?)?;
    let common_denominator = rational_gcd
        .terms_map()
        .values()
        .fold(Integer::one(), |lcm, coefficient| {
            lcm.lcm(&coefficient.denominator())
        });
    let mut integer_terms = rational_gcd
        .terms_map()
        .iter()
        .map(|(exponent, coefficient)| {
            let multiplier = common_denominator.div_floor(&coefficient.denominator());
            (exponent.clone(), coefficient.numerator() * multiplier)
        })
        .collect::<Vec<_>>();
    let primitive_content = integer_terms
        .iter()
        .fold(Integer::zero(), |gcd, (_, coefficient)| {
            gcd.gcd(coefficient)
        });
    for (_, coefficient) in &mut integer_terms {
        *coefficient = coefficient.div_floor(&primitive_content) * &common_content;
    }
    normalize_integer_polynomial(left.ring.from_terms(integer_terms)?)
}

impl<R: CommutativeRing> PartialEq for Polynomial<R> {
    fn eq(&self, other: &Self) -> bool {
        if self.ring != other.ring {
            return false;
        }
        match (&self.storage, &other.storage) {
            (PolynomialStorage::Generic(left), PolynomialStorage::Generic(right)) => {
                Arc::ptr_eq(left, right) || left == right
            }
            (PolynomialStorage::Flint(left), PolynomialStorage::Flint(right)) => {
                Arc::ptr_eq(left, right)
                    || left
                        .native
                        .equal(&right.native)
                        .unwrap_or_else(|| self.terms_map() == other.terms_map())
            }
            _ => self.terms_map() == other.terms_map(),
        }
    }
}

impl<R: CommutativeRing> Eq for Polynomial<R> {}

impl<R: CommutativeRing> fmt::Debug for Polynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Polynomial")
            .field("coefficient_type", &R::TYPE_ID)
            .field("variables", &self.ring.variables)
            .field("nonzero_terms", &self.nonzero_term_count())
            .finish()
    }
}

impl<R: CommutativeRing> fmt::Display for Polynomial<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Polynomial<{}>(nonzero_terms={}, variables=[{}])",
            R::TYPE_ID,
            self.nonzero_term_count(),
            self.ring.variables.join(", ")
        )
    }
}

impl<R: CommutativeRing> SemanticType for Polynomial<R> {
    const TYPE_ID: &'static str = "noether/commalg/Polynomial@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        append_component(&mut output, &self.ring.canonical_bytes()?)?;
        append_length(&mut output, self.terms_map().len())?;
        for (exponent, coefficient) in self.terms_map() {
            append_length(&mut output, exponent.len())?;
            for power in exponent {
                append_component(&mut output, &power.canonical_bytes()?)?;
            }
            append_component(&mut output, &coefficient.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &POLYNOMIAL_DESCRIPTION
    }
}

impl<R: CommutativeRing> Add<&Polynomial<R>> for &Polynomial<R> {
    type Output = Result<Polynomial<R>, ContextMismatchError>;

    fn add(self, rhs: &Polynomial<R>) -> Self::Output {
        self.ring.add(self, rhs)
    }
}

impl<R: CommutativeRing> Sub<&Polynomial<R>> for &Polynomial<R> {
    type Output = Result<Polynomial<R>, ContextMismatchError>;

    fn sub(self, rhs: &Polynomial<R>) -> Self::Output {
        self.ring.sub(self, rhs)
    }
}

impl<R: CommutativeRing> Mul<&Polynomial<R>> for &Polynomial<R> {
    type Output = Result<Polynomial<R>, ContextMismatchError>;

    fn mul(self, rhs: &Polynomial<R>) -> Self::Output {
        self.ring.mul(self, rhs)
    }
}

impl<R: CommutativeRing> Neg for &Polynomial<R> {
    type Output = Result<Polynomial<R>, ContextMismatchError>;

    fn neg(self) -> Self::Output {
        self.ring.neg(self)
    }
}

fn collect_variables<I, S>(variables: I) -> Result<Vec<String>, PolynomialContextError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let variables = variables.into_iter().map(Into::into).collect::<Vec<_>>();
    if let Some(variable) = variables.iter().position(String::is_empty) {
        return Err(PolynomialContextError::EmptyVariableName { variable });
    }
    let mut names = BTreeSet::new();
    for name in &variables {
        if !names.insert(name.clone()) {
            return Err(PolynomialContextError::DuplicateVariableName { name: name.clone() });
        }
    }
    Ok(variables)
}

fn bits(value: &Integer) -> Result<Vec<bool>, PolynomialError> {
    if value.sign() < 0 {
        return Err(PolynomialError::NegativeExponent { variable: 0 });
    }
    let mut value = value.clone();
    let two = Integer::from(2_u8);
    let mut output = Vec::new();
    while !value.is_zero() {
        let (quotient, remainder) = Z
            .div_rem(&value, &two)
            .map_err(|_| PolynomialError::ArithmeticInvariant)?;
        output.push(!remainder.is_zero());
        value = quotient;
    }
    Ok(output)
}

fn grading_as_polynomial(error: GradingError) -> PolynomialError {
    match error {
        GradingError::NegativeExponent { variable } => {
            PolynomialError::NegativeExponent { variable }
        }
        GradingError::ArityMismatch { expected, actual } => {
            PolynomialError::ExponentArity { expected, actual }
        }
        GradingError::NoVariables
        | GradingError::ZeroWeight { .. }
        | GradingError::DegreeOverflow
        | GradingError::NonpositiveFunctional { .. } => PolynomialError::ArithmeticInvariant,
        GradingError::RankMismatch { expected, actual } => {
            PolynomialError::DegreeRank { expected, actual }
        }
    }
}

fn truncation_as_polynomial(error: SeriesTruncationError) -> PolynomialError {
    match error {
        SeriesTruncationError::ArityMismatch { expected, actual } => {
            PolynomialError::ExponentArity { expected, actual }
        }
        SeriesTruncationError::ZeroBound
        | SeriesTruncationError::EmptyVariableData
        | SeriesTruncationError::ZeroBoxBound { .. }
        | SeriesTruncationError::ZeroWeight { .. }
        | SeriesTruncationError::Grading(_) => PolynomialError::ArithmeticInvariant,
    }
}

fn retain_exponent(
    exponent: &[u64],
    output: &mut Vec<Vec<u64>>,
    max_count: usize,
) -> Result<(), PolynomialError> {
    if output.len() == max_count {
        return Err(PolynomialError::ResourceLimit {
            resource: "monomial count",
            limit: max_count,
        });
    }
    output.push(exponent.to_vec());
    Ok(())
}

fn enumerate_exact_weighted(
    weights: &[u64],
    variable: usize,
    remaining_degree: u64,
    exponent: &mut [u64],
    output: &mut Vec<Vec<u64>>,
    max_count: usize,
) -> Result<(), PolynomialError> {
    let weight = weights[variable];
    if variable + 1 == weights.len() {
        if remaining_degree.is_multiple_of(weight) {
            exponent[variable] = remaining_degree / weight;
            retain_exponent(exponent, output, max_count)?;
        }
        return Ok(());
    }
    for power in 0..=remaining_degree / weight {
        exponent[variable] = power;
        enumerate_exact_weighted(
            weights,
            variable + 1,
            remaining_degree - power * weight,
            exponent,
            output,
            max_count,
        )?;
    }
    Ok(())
}

fn enumerate_weighted_below(
    weights: &[u64],
    variable: usize,
    remaining_degree: u64,
    exponent: &mut [u64],
    output: &mut Vec<Vec<u64>>,
    max_count: usize,
) -> Result<(), PolynomialError> {
    let weight = weights[variable];
    if variable + 1 == weights.len() {
        for power in 0..=remaining_degree / weight {
            exponent[variable] = power;
            retain_exponent(exponent, output, max_count)?;
        }
        return Ok(());
    }
    for power in 0..=remaining_degree / weight {
        exponent[variable] = power;
        enumerate_weighted_below(
            weights,
            variable + 1,
            remaining_degree - power * weight,
            exponent,
            output,
            max_count,
        )?;
    }
    Ok(())
}

fn enumerate_box(
    bounds: &[u64],
    variable: usize,
    exponent: &mut [u64],
    output: &mut Vec<Vec<u64>>,
    max_count: usize,
) -> Result<(), PolynomialError> {
    if variable + 1 == bounds.len() {
        for power in 0..bounds[variable] {
            exponent[variable] = power;
            retain_exponent(exponent, output, max_count)?;
        }
        return Ok(());
    }
    for power in 0..bounds[variable] {
        exponent[variable] = power;
        enumerate_box(bounds, variable + 1, exponent, output, max_count)?;
    }
    Ok(())
}

fn append_length(output: &mut Vec<u8>, length: usize) -> Result<(), NoetherError> {
    let length = u64::try_from(length).map_err(|_| {
        NoetherError::Serialization("polynomial component count does not fit in u64".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    Ok(())
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    append_length(output, component.len())?;
    output.extend_from_slice(component);
    Ok(())
}

#[cfg(test)]
mod view_tests {
    use super::*;
    use crate::{Rational, RationalField, Q, Z};

    #[test]
    fn selected_variable_coefficients_have_the_induced_context() {
        let order = MonomialOrder::block([
            (vec![2], MonomialOrder::Lexicographic),
            (vec![0, 1], MonomialOrder::GradedReverseLexicographic),
        ])
        .unwrap();
        let ring = PolynomialRing::with_grading(
            Z,
            ["x", "y", "t"],
            order,
            WeightedGrading::new(vec![2, 3, 5]).unwrap(),
        )
        .unwrap();
        let value = ring
            .from_u64_terms(
                (0..=6).map(|power| (vec![6 - power, 0, power], Integer::from(power + 1))),
            )
            .unwrap();

        let coefficients = value.coefficient_polynomials(2, 7).unwrap();

        assert_eq!(coefficients.len(), 7);
        assert_eq!(coefficients[0].ring().variables(), &["x", "y"]);
        assert_eq!(
            coefficients[0].ring().term_order(),
            MonomialOrder::GradedReverseLexicographic
        );
        assert_eq!(
            coefficients[0]
                .ring()
                .weighted_grading()
                .expect("positive rank-one grading")
                .weights(),
            &[2, 3]
        );
        assert_eq!(
            coefficients[3]
                .coefficient(&[Integer::from(3), Integer::zero()])
                .unwrap(),
            Integer::from(4)
        );
        assert!(value
            .coefficient_polynomial(2, &Integer::from(7))
            .unwrap()
            .is_zero());
        assert!(matches!(
            value.coefficient_polynomials(2, 6),
            Err(PolynomialError::ResourceLimit {
                resource: "selected-variable coefficient count",
                limit: 6
            })
        ));
    }

    #[test]
    fn coefficient_allocation_rejects_an_arbitrary_size_degree_before_conversion() {
        let ring = PolynomialRing::new(Z, ["x", "t"], MonomialOrder::Lexicographic).unwrap();
        let huge = Integer::from(u128::from(u64::MAX) + 1);
        let value = ring
            .from_terms([(vec![Integer::zero(), huge], Integer::one())])
            .unwrap();

        assert!(matches!(
            value.coefficient_polynomials(1, 100),
            Err(PolynomialError::ResourceLimit { limit: 100, .. })
        ));
    }

    #[test]
    fn univariate_view_checks_every_other_variable() {
        let order = MonomialOrder::block([
            (vec![1], MonomialOrder::GradedLexicographic),
            (vec![0, 2], MonomialOrder::Lexicographic),
        ])
        .unwrap();
        let ring = PolynomialRing::with_grading(
            Z,
            ["x", "t", "y"],
            order,
            WeightedGrading::new(vec![2, 5, 3]).unwrap(),
        )
        .unwrap();
        let t_only = ring
            .from_u64_terms([
                (vec![0, 8, 0], Integer::from(3)),
                (vec![0, 0, 0], Integer::one()),
            ])
            .unwrap();

        let view = t_only.univariate_view(1).unwrap();

        assert_eq!(view.ring().variables(), &["t"]);
        assert_eq!(
            view.ring()
                .weighted_grading()
                .expect("positive rank-one grading")
                .weights(),
            &[5]
        );
        assert_eq!(view.ring().term_order(), MonomialOrder::GradedLexicographic);
        assert_eq!(view.degree(Some(0)).unwrap(), Some(Integer::from(8)));
        let depends_on_x = ring
            .from_u64_terms([(vec![1, 1, 0], Integer::one())])
            .unwrap();
        assert_eq!(
            depends_on_x.univariate_view(1),
            Err(PolynomialError::DependsOnOtherVariable {
                selected: 1,
                other: 0,
            })
        );

        let one_variable = PolynomialRing::new(Z, ["u"], MonomialOrder::Lexicographic).unwrap();
        let u = one_variable.generator(0).unwrap();
        assert_eq!(u.univariate_view(0).unwrap(), u);
        assert!(matches!(
            u.coefficient_polynomials(0, 2),
            Err(PolynomialError::NoRemainingVariables { selected: 0 })
        ));
    }

    #[test]
    fn homogeneous_and_region_monomials_are_deterministic_and_bounded() {
        let ring = PolynomialRing::new(
            Z,
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let degree_two = ring.homogeneous_monomials(2, 6).unwrap();
        let exponents = degree_two
            .iter()
            .map(|value| value.terms().next().unwrap().0.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(degree_two.len(), 6);
        assert!(exponents
            .windows(2)
            .all(|pair| { ring.term_order.compare(&pair[0], &pair[1]) == Ordering::Greater }));
        assert!(matches!(
            ring.homogeneous_monomials(2, 5),
            Err(PolynomialError::ResourceLimit {
                resource: "monomial count",
                limit: 5
            })
        ));

        let weighted = PolynomialRing::with_grading(
            Z,
            ["x", "y"],
            MonomialOrder::Lexicographic,
            WeightedGrading::new(vec![1, 2]).unwrap(),
        )
        .unwrap();
        assert_eq!(weighted.homogeneous_monomials(4, 3).unwrap().len(), 3);

        let box_region = SeriesTruncation::box_bounds(vec![2, 3]).unwrap();
        assert_eq!(
            weighted.monomials_in_region(&box_region, 6).unwrap().len(),
            6
        );
        assert!(matches!(
            weighted.monomials_in_region(&box_region, 5),
            Err(PolynomialError::ResourceLimit { limit: 5, .. })
        ));
        assert!(matches!(
            ring.monomials_in_region(&box_region, 10),
            Err(PolynomialError::ExponentArity {
                expected: 2,
                actual: 3
            })
        ));
    }

    #[test]
    fn zero_variable_polynomial_ring_is_the_explicit_constant_context() {
        let ring = PolynomialRing::new(
            Q,
            Vec::<String>::new(),
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        assert!(ring.variables().is_empty());
        assert!(ring.generators().unwrap().is_empty());
        assert_eq!(ring.zero().terms().count(), 0);
        let constant = ring.constant(Rational::from(7_i64)).unwrap();
        assert_eq!(constant.terms().count(), 1);
        assert_eq!(constant.constant_coefficient(), Rational::from(7_i64));
        assert_eq!(
            PolynomialRing::<RationalField>::from_json(&ring.to_json().unwrap()).unwrap(),
            ring
        );
    }

    #[test]
    fn arbitrary_gradings_preserve_context_and_use_positive_evidence() {
        let grading = FreeAbelianGrading::new(
            2,
            vec![
                Degree::new(vec![1.into(), 0.into()]),
                Degree::new(vec![1.into(), 1.into()]),
                Degree::new(vec![0.into(), 1.into()]),
            ],
        )
        .unwrap();
        let ring = PolynomialRing::with_grading(
            Z,
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
            grading.clone(),
        )
        .unwrap();
        let positivity = grading
            .certify_positive(Degree::new(vec![1.into(), 1.into()]))
            .unwrap();
        let target = Degree::new(vec![2.into(), 1.into()]);
        let monomials = ring
            .homogeneous_monomials_of_degree(&target, &positivity, 2)
            .unwrap();
        assert_eq!(monomials.len(), 2);
        assert!(monomials
            .iter()
            .all(|value| value.multihomogeneous_degree().unwrap() == Some(target.clone())));
        assert!(matches!(
            ring.homogeneous_monomials_of_degree(&target, &positivity, 1),
            Err(PolynomialError::ResourceLimit { limit: 1, .. })
        ));
        assert_eq!(
            PolynomialRing::<crate::IntegerRing>::from_json(&ring.to_json().unwrap()).unwrap(),
            ring
        );
        assert!(ring.weighted_grading().is_none());
        assert!(matches!(
            ring.homogeneous_monomials(2, 10),
            Err(PolynomialError::RankOneGradingRequired { actual: 2 })
        ));
    }
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    use crate::commalg::ResidueRing;
    use crate::numthy::FiniteField;
    use crate::{Field, Q, Z};

    fn assert_native_derivative<R: CommutativeRing>(
        ring: PolynomialRing<R>,
        leading: R::Element,
        trailing: R::Element,
    ) {
        let leading_derivative = ring.natural_scale(&leading, &Integer::from(4)).unwrap();
        let value = ring
            .from_u64_terms([(vec![4, 1], leading), (vec![1, 2], trailing.clone())])
            .unwrap();
        let expected = ring
            .from_u64_terms([(vec![3, 1], leading_derivative), (vec![0, 2], trailing)])
            .unwrap();

        assert!(value.uses_flint_storage());
        assert!(!value.sparse_view_is_materialized());
        let derivative = value.derivative(0).unwrap();
        assert!(derivative.uses_flint_storage());
        assert!(!value.sparse_view_is_materialized());
        assert!(!derivative.sparse_view_is_materialized());
        assert_eq!(derivative, expected);
    }

    fn assert_native_evaluation<R: CommutativeRing>(
        ring: PolynomialRing<R>,
        first: R::Element,
        second: R::Element,
    ) {
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let value = ring.add(&ring.mul(&x, &x).unwrap(), &y).unwrap();
        let expected = ring
            .coefficient_ring()
            .add(
                &ring.coefficient_ring().mul(&first, &first).unwrap(),
                &second,
            )
            .unwrap();

        assert!(value.uses_flint_storage());
        assert!(!value.sparse_view_is_materialized());
        assert!(value.evaluate(&[first, second]).unwrap() == expected);
        assert!(!value.sparse_view_is_materialized());
    }

    fn assert_native_substitution<R: CommutativeRing>(ring: PolynomialRing<R>) {
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let value = ring.add(&ring.mul(&x, &x).unwrap(), &y).unwrap();
        let images = [
            ring.add(&x, &ring.one()).unwrap(),
            ring.add(&y, &x).unwrap(),
        ];
        let expected = ring
            .add(&ring.mul(&images[0], &images[0]).unwrap(), &images[1])
            .unwrap();

        assert!(value.uses_flint_storage());
        assert!(!value.sparse_view_is_materialized());
        let result = value.substitute(&images).unwrap();
        assert!(result.uses_flint_storage());
        assert!(!value.sparse_view_is_materialized());
        assert!(images
            .iter()
            .all(|image| !image.sparse_view_is_materialized()));
        assert!(!result.sparse_view_is_materialized());
        assert_eq!(result, expected);
    }

    fn assert_native_sum_products<R: CommutativeRing>(ring: PolynomialRing<R>) {
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let x_plus_y = ring.add(&x, &y).unwrap();
        let x_minus_y = ring.sub(&x, &y).unwrap();
        let one = ring.one();
        let expected = ring
            .add(
                &ring
                    .add(
                        &ring.mul(&x_plus_y, &x).unwrap(),
                        &ring.mul(&x_minus_y, &y).unwrap(),
                    )
                    .unwrap(),
                &x,
            )
            .unwrap();
        let inputs = [(&x_plus_y, &x), (&x_minus_y, &y), (&one, &x)];

        assert!(inputs.iter().all(|(left, right)| {
            left.uses_flint_storage()
                && right.uses_flint_storage()
                && !left.sparse_view_is_materialized()
                && !right.sparse_view_is_materialized()
        }));
        let result = ring.sum_products(&inputs).unwrap();
        assert!(result.uses_flint_storage());
        assert!(!result.sparse_view_is_materialized());
        assert!(inputs.iter().all(|(left, right)| {
            !left.sparse_view_is_materialized() && !right.sparse_view_is_materialized()
        }));
        assert_eq!(result, expected);
    }

    fn assert_native_field_division_and_gcd<R: Field>(ring: PolynomialRing<R>) {
        let x = ring.generator(0).unwrap();
        let common = ring.add(&x, &ring.one()).unwrap();
        let left = ring.mul(&common, &x).unwrap();
        let right = ring.mul(&common, &common).unwrap();

        let quotient = left.exact_div(&common).unwrap();
        assert!(quotient.uses_flint_storage());
        assert!(!left.sparse_view_is_materialized());
        assert!(!common.sparse_view_is_materialized());
        assert!(!quotient.sparse_view_is_materialized());
        assert_eq!(quotient, x);

        let gcd = left.gcd(&right).unwrap();
        assert!(gcd.uses_flint_storage());
        assert!(!gcd.sparse_view_is_materialized());
        assert_eq!(gcd, common);
        assert!(matches!(
            x.exact_div(&common),
            Err(PolynomialError::NonExactDivision)
        ));
    }

    fn assert_sticky_native_arithmetic<R: CommutativeRing>(
        ring: PolynomialRing<R>,
        a_coefficient: R::Element,
        b_coefficient: R::Element,
        scalar: R::Element,
    ) {
        let expected_mixed_coefficient = ring
            .coefficient_ring()
            .mul(&a_coefficient, &b_coefficient)
            .unwrap();
        let left = ring
            .from_u64_terms([
                (vec![2, 0], a_coefficient),
                (vec![0, 1], ring.coefficient_ring().one()),
            ])
            .unwrap();
        let right = ring
            .from_u64_terms([
                (vec![1, 1], b_coefficient),
                (vec![0, 0], ring.coefficient_ring().one()),
            ])
            .unwrap();
        assert!(left.uses_flint_storage());
        assert!(!left.sparse_view_is_materialized());
        let cloned = left.clone();
        assert!(left.shares_storage_with(&cloned));
        assert!(!cloned.sparse_view_is_materialized());
        let _ = left
            .coefficient(&[Integer::from(2), Integer::zero()])
            .unwrap();
        assert!(!left.sparse_view_is_materialized());

        let sum = ring.add(&left, &right).unwrap();
        let difference = ring.sub(&sum, &right).unwrap();
        let product = ring.mul(&left, &right).unwrap();
        let negative = ring.neg(&left).unwrap();
        let scaled = left.scalar_mul(&scalar).unwrap();
        for value in [&sum, &difference, &product, &negative, &scaled] {
            assert!(value.uses_flint_storage());
            assert!(!value.sparse_view_is_materialized());
        }
        assert!(
            product
                .coefficient(&[Integer::from(3), Integer::from(1)])
                .unwrap()
                == expected_mixed_coefficient
        );
        assert!(!product.sparse_view_is_materialized());

        assert_eq!(difference, left);
        assert_eq!(product, ring.mul(&right, &left).unwrap());
        assert_eq!(ring.add(&left, &negative).unwrap(), ring.zero());
        for value in [
            &left,
            &right,
            &sum,
            &difference,
            &product,
            &negative,
            &scaled,
        ] {
            assert!(!value.sparse_view_is_materialized());
        }

        let generic_left = Polynomial {
            ring: ring.clone(),
            storage: PolynomialStorage::Generic(Arc::new(left.terms_map().clone())),
        };
        let generic_right = Polynomial {
            ring: ring.clone(),
            storage: PolynomialStorage::Generic(Arc::new(right.terms_map().clone())),
        };
        assert_eq!(sum, ring.add(&generic_left, &generic_right).unwrap());
        assert_eq!(
            difference,
            ring.sub(
                &ring.add(&generic_left, &generic_right).unwrap(),
                &generic_right,
            )
            .unwrap()
        );
        assert_eq!(product, ring.mul(&generic_left, &generic_right).unwrap());
        assert_eq!(negative, ring.neg(&generic_left).unwrap());
        assert_eq!(scaled, generic_left.scalar_mul(&scalar).unwrap());
    }

    fn assert_canonical_native_transfer<R: CommutativeRing>(
        ring: PolynomialRing<R>,
        coefficients: [R::Element; 6],
    ) {
        let leading_coefficient = coefficients[0].clone();
        let exponents = [[2, 0], [1, 1], [0, 2], [1, 0], [0, 1], [0, 0]];
        let terms = exponents
            .into_iter()
            .zip(coefficients)
            .map(|(exponent, coefficient)| (exponent.to_vec(), coefficient))
            .collect::<Vec<_>>();
        let expected = ring
            .from_u64_terms(terms.iter().map(|(exponent, coefficient)| {
                (
                    exponent.iter().copied().map(u64::from).collect::<Vec<_>>(),
                    coefficient.clone(),
                )
            }))
            .unwrap();
        let actual = ring.polynomial_from_canonical_u32_terms(terms).unwrap();
        assert!(actual.uses_flint_storage());
        assert!(!actual.sparse_view_is_materialized());
        assert!(!ring.is_zero(&actual));
        assert!(
            actual.leading_term_owned()
                == Some((vec![Integer::from(2), Integer::zero()], leading_coefficient))
        );
        assert!(!actual.sparse_view_is_materialized());
        assert_eq!(actual, expected);
    }

    fn assert_equal_contexts_stay_native<R: CommutativeRing>(
        left_ring: PolynomialRing<R>,
        right_ring: PolynomialRing<R>,
        left_coefficient: R::Element,
        right_coefficient: R::Element,
    ) {
        let left = left_ring
            .from_u64_terms([(vec![1, 0], left_coefficient)])
            .unwrap();
        let right = right_ring
            .from_u64_terms([(vec![0, 1], right_coefficient)])
            .unwrap();

        let sum = left_ring.add(&left, &right).unwrap();
        let product = left_ring.mul(&left, &right).unwrap();

        for value in [&left, &right, &sum, &product] {
            assert!(value.uses_flint_storage());
            assert!(!value.sparse_view_is_materialized());
        }
    }

    fn fresh_ring<R: CommutativeRing>(coefficients: R) -> PolynomialRing<R> {
        PolynomialRing::new(coefficients, ["x", "y"], MonomialOrder::GradedLexicographic).unwrap()
    }

    #[test]
    fn supported_exact_contexts_keep_flint_storage_across_arithmetic() {
        assert_sticky_native_arithmetic(
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedLexicographic).unwrap(),
            Integer::from(7),
            Integer::from(-5),
            Integer::from(3),
        );
        assert_sticky_native_arithmetic(
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap(),
            "7/3".parse().unwrap(),
            "-5/2".parse().unwrap(),
            "3/5".parse().unwrap(),
        );

        for residues in [
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            assert_sticky_native_arithmetic(
                PolynomialRing::new(
                    residues.clone(),
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
                residues.element(7_u8),
                residues.element(11_u8),
                residues.element(3_u8),
            );
        }

        for field in [
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(3, 2).unwrap(),
        ] {
            assert_sticky_native_arithmetic(
                PolynomialRing::new(
                    field.clone(),
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
                field.element(2_u8),
                field.generator(),
                field.element(2_u8),
            );
        }
    }

    #[test]
    fn native_derivative_stays_native_in_every_flint_coefficient_context() {
        assert_native_derivative(
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
            Integer::from(3),
            Integer::from(-2),
        );
        assert_native_derivative(
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
            "3/5".parse().unwrap(),
            "-2/7".parse().unwrap(),
        );
        for residues in [
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            assert_native_derivative(
                PolynomialRing::new(
                    residues.clone(),
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
                residues.element(3_u8),
                residues.element(2_u8),
            );
        }
        for field in [
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(3, 2).unwrap(),
        ] {
            assert_native_derivative(
                PolynomialRing::new(
                    field.clone(),
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
                field.element(2_u8),
                field.generator(),
            );
        }
    }

    #[test]
    fn native_evaluation_avoids_sparse_materialization_in_every_flint_context() {
        assert_native_evaluation(
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
            Integer::from(3),
            Integer::from(-2),
        );
        assert_native_evaluation(
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
            "3/5".parse().unwrap(),
            "-2/7".parse().unwrap(),
        );
        for residues in [
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            assert_native_evaluation(
                PolynomialRing::new(
                    residues.clone(),
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
                residues.element(3_u8),
                residues.element(2_u8),
            );
        }
        for field in [
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(3, 2).unwrap(),
        ] {
            assert_native_evaluation(
                PolynomialRing::new(
                    field.clone(),
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
                field.element(2_u8),
                field.generator(),
            );
        }
    }

    #[test]
    fn evaluation_retains_the_arbitrary_exponent_floor() {
        let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let exponent = Integer::from(u128::from(u64::MAX) + 1);
        let value = ring
            .from_terms([(vec![exponent], Integer::from(7))])
            .unwrap();

        assert!(!value.uses_flint_storage());
        assert_eq!(value.evaluate(&[Integer::one()]).unwrap(), Integer::from(7));
    }

    #[test]
    fn native_substitution_avoids_sparse_materialization_in_every_flint_context() {
        assert_native_substitution(
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
        );
        assert_native_substitution(
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
        );
        for residues in [
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            assert_native_substitution(
                PolynomialRing::new(
                    residues,
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
            );
        }
        for field in [
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(3, 2).unwrap(),
        ] {
            assert_native_substitution(
                PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
                    .unwrap(),
            );
        }
    }

    #[test]
    fn native_sum_products_reuses_flint_storage_in_every_coefficient_context() {
        assert_native_sum_products(
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
        );
        assert_native_sum_products(
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
        );
        for residues in [
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            assert_native_sum_products(
                PolynomialRing::new(
                    residues,
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
            );
        }
        for field in [
            FiniteField::prime(7_u8).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(3, 2).unwrap(),
        ] {
            assert_native_sum_products(
                PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
                    .unwrap(),
            );
        }
    }

    #[test]
    fn sum_products_retains_the_arbitrary_exponent_floor() {
        let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let exponent = Integer::from(u128::from(u64::MAX) + 1);
        let huge = ring
            .from_terms([(vec![exponent.clone()], Integer::from(7))])
            .unwrap();
        let one = ring.one();
        let result = ring.sum_products(&[(&huge, &one), (&one, &one)]).unwrap();

        assert!(!huge.uses_flint_storage());
        assert!(!result.uses_flint_storage());
        assert_eq!(result.coefficient(&[exponent]).unwrap(), Integer::from(7));
        assert_eq!(result.constant_coefficient(), Integer::one());
    }

    #[test]
    fn substitution_retains_the_arbitrary_exponent_floor() {
        let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let exponent = Integer::from(u128::from(u64::MAX) + 1);
        let value = ring
            .from_terms([(vec![exponent], Integer::from(7))])
            .unwrap();
        let x = ring.generator(0).unwrap();

        assert!(!value.uses_flint_storage());
        assert_eq!(value.substitute(&[x]).unwrap(), value);
    }

    #[test]
    fn native_field_exact_division_and_gcd_preserve_native_storage() {
        assert_native_field_division_and_gcd(
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap(),
        );
        for field in [
            FiniteField::prime(2_u8).unwrap(),
            FiniteField::prime(65_521_u64).unwrap(),
            FiniteField::prime(
                "170141183460469231731687303715884105727"
                    .parse::<Integer>()
                    .unwrap(),
            )
            .unwrap(),
            FiniteField::extension(2, 4).unwrap(),
        ] {
            assert_native_field_division_and_gcd(
                PolynomialRing::new(field, ["x"], MonomialOrder::GradedReverseLexicographic)
                    .unwrap(),
            );
        }
    }

    #[test]
    fn native_multivariate_gcd_rehomes_independently_deserialized_contexts() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let common = (&x + &y).unwrap();
        let left = (&common * &x).unwrap();
        let right = (&common * &y).unwrap();
        let left = Polynomial::<crate::RationalField>::from_json(&left.to_json().unwrap()).unwrap();
        let right =
            Polynomial::<crate::RationalField>::from_json(&right.to_json().unwrap()).unwrap();

        let gcd = left.gcd(&right).unwrap();

        assert_eq!(gcd, common);
        assert!(gcd.uses_flint_storage());
        assert_eq!(left.exact_div(&gcd).unwrap(), x);
        assert_eq!(right.exact_div(&gcd).unwrap(), y);
    }

    #[test]
    fn field_exact_division_is_multivariate_with_an_arbitrary_exponent_floor() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let variables = ring.generators().unwrap();
        let common = (&variables[0] + &variables[1]).unwrap();
        let cofactor = (&(&variables[0] * &variables[1]).unwrap() + &ring.one()).unwrap();
        let product = (&common * &cofactor).unwrap();

        let quotient = product.exact_div(&common).unwrap();

        assert_eq!(quotient, cofactor);
        assert!(quotient.uses_flint_storage());
        assert!(matches!(
            variables[0].exact_div(&common),
            Err(PolynomialError::NonExactDivision)
        ));

        let huge = Integer::from(u128::from(u64::MAX) + 1);
        let sparse_common = ring
            .from_terms([
                (vec![huge.clone(), Integer::zero()], Q.one()),
                (vec![Integer::zero(), Integer::one()], Q.one()),
            ])
            .unwrap();
        let sparse_product = (&sparse_common * &variables[1]).unwrap();
        assert!(!sparse_product.uses_flint_storage());
        assert_eq!(
            sparse_product.exact_div(&sparse_common).unwrap(),
            variables[1]
        );

        let order_sensitive = ring
            .from_terms([
                (vec![huge, Integer::zero()], Q.one()),
                (
                    vec![Integer::zero(), Integer::from(u128::from(u64::MAX) + 2)],
                    Q.one(),
                ),
            ])
            .unwrap();
        assert!(!order_sensitive.uses_flint_storage());
        assert_eq!(
            order_sensitive.leading_term_owned().unwrap().0,
            vec![Integer::zero(), Integer::from(u128::from(u64::MAX) + 2),]
        );
        assert_eq!(
            (&order_sensitive * &common)
                .unwrap()
                .exact_div(&order_sensitive)
                .unwrap(),
            common
        );
    }

    #[test]
    fn integer_exact_division_and_gcd_have_native_and_arbitrary_exponent_floors() {
        let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let x = ring.generator(0).unwrap();
        let common = ring
            .add(
                &x.scalar_mul(&Integer::from(2)).unwrap(),
                &ring.constant(2.into()).unwrap(),
            )
            .unwrap();
        let left_cofactor = ring
            .add(&x.scalar_mul(&Integer::from(3)).unwrap(), &ring.one())
            .unwrap();
        let right_cofactor = ring
            .sub(
                &x.scalar_mul(&Integer::from(5)).unwrap(),
                &ring.constant(3.into()).unwrap(),
            )
            .unwrap();
        let left = ring.mul(&common, &left_cofactor).unwrap();
        let right = ring.mul(&common, &right_cofactor).unwrap();

        let quotient = left.exact_div(&common).unwrap();
        let gcd = left.gcd(&right).unwrap();
        assert!(quotient.uses_flint_storage());
        assert!(gcd.uses_flint_storage());
        assert!(!quotient.sparse_view_is_materialized());
        assert!(!gcd.sparse_view_is_materialized());
        assert_eq!(quotient, left_cofactor);
        assert_eq!(gcd, common);

        let huge = Integer::from(u128::from(u64::MAX) + 1);
        let huge_common = ring
            .from_terms([
                (vec![huge], Integer::one()),
                (vec![Integer::zero()], Integer::one()),
            ])
            .unwrap();
        let huge_left = ring.mul(&huge_common, &x).unwrap();
        let huge_right = ring
            .mul(&huge_common, &ring.add(&x, &ring.one()).unwrap())
            .unwrap();
        assert!(!huge_left.uses_flint_storage());
        assert_eq!(huge_left.exact_div(&huge_common).unwrap(), x);
        assert_eq!(huge_left.gcd(&huge_right).unwrap(), huge_common);
        assert!(matches!(
            x.exact_div(&common),
            Err(PolynomialError::NonExactDivision)
        ));
    }

    #[test]
    fn canonical_kernel_terms_append_directly_to_every_native_context() {
        assert_canonical_native_transfer(
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
            [2, -3, 5, 7, -11, 13].map(Integer::from),
        );
        assert_canonical_native_transfer(
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap(),
            ["2/3", "-3/5", "5/7", "7/11", "-11/13", "13/17"].map(|value| value.parse().unwrap()),
        );
        for residues in [
            ResidueRing::new(25_u8).unwrap(),
            ResidueRing::new("18446744073709551629".parse::<Integer>().unwrap()).unwrap(),
        ] {
            assert_canonical_native_transfer(
                PolynomialRing::new(
                    residues.clone(),
                    ["x", "y"],
                    MonomialOrder::GradedReverseLexicographic,
                )
                .unwrap(),
                [2, 3, 5, 7, 11, 13].map(|value| residues.element(value)),
            );
        }
        let prime = FiniteField::prime(7_u8).unwrap();
        assert_canonical_native_transfer(
            PolynomialRing::new(
                prime.clone(),
                ["x", "y"],
                MonomialOrder::GradedReverseLexicographic,
            )
            .unwrap(),
            [1, 2, 3, 4, 5, 6].map(|value| prime.element(value)),
        );
        let extension = FiniteField::extension(3, 2).unwrap();
        let generator = extension.generator();
        assert_canonical_native_transfer(
            PolynomialRing::new(
                extension.clone(),
                ["x", "y"],
                MonomialOrder::GradedReverseLexicographic,
            )
            .unwrap(),
            [
                extension.one(),
                extension.element(2),
                generator.clone(),
                extension.add(&generator, &extension.one()).unwrap(),
                extension.mul(&generator, &generator).unwrap(),
                extension.add(&generator, &extension.element(2)).unwrap(),
            ],
        );
    }

    #[test]
    fn equal_separately_created_contexts_reuse_native_storage_without_materialization() {
        assert_equal_contexts_stay_native(fresh_ring(Z), fresh_ring(Z), 2.into(), 3.into());
        assert_equal_contexts_stay_native(
            fresh_ring(Q),
            fresh_ring(Q),
            "2/3".parse().unwrap(),
            "3/5".parse().unwrap(),
        );

        for modulus in [
            Integer::from(25),
            "18446744073709551629".parse::<Integer>().unwrap(),
        ] {
            let left_context = ResidueRing::new(modulus.clone()).unwrap();
            let right_context = ResidueRing::new(modulus).unwrap();
            assert_equal_contexts_stay_native(
                fresh_ring(left_context.clone()),
                fresh_ring(right_context.clone()),
                left_context.element(2),
                right_context.element(3),
            );
        }

        for characteristic in [
            Integer::from(7),
            "170141183460469231731687303715884105727"
                .parse::<Integer>()
                .unwrap(),
        ] {
            let left_context = FiniteField::prime(characteristic.clone()).unwrap();
            let right_context = FiniteField::prime(characteristic).unwrap();
            assert_equal_contexts_stay_native(
                fresh_ring(left_context.clone()),
                fresh_ring(right_context.clone()),
                left_context.element(2),
                right_context.element(3),
            );
        }

        let left_context = FiniteField::extension(3, 2).unwrap();
        let right_context = FiniteField::extension(3, 2).unwrap();
        assert_equal_contexts_stay_native(
            fresh_ring(left_context.clone()),
            fresh_ring(right_context.clone()),
            left_context.generator(),
            right_context.element(2),
        );
    }

    #[test]
    fn exponents_beyond_flint_word_storage_keep_the_generic_floor() {
        let ring = PolynomialRing::new(Z, ["x"], MonomialOrder::Lexicographic).unwrap();
        let exponent = Integer::from(u128::from(u64::MAX) + 1);
        let value = ring
            .from_terms([(vec![exponent.clone()], Integer::one())])
            .unwrap();
        assert!(!value.uses_flint_storage());
        assert_eq!(value.degree(Some(0)).unwrap(), Some(exponent));
    }

    #[test]
    fn packed_generic_multiplication_agrees_with_native_flint_storage() {
        let ring =
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let left = ring
            .from_u64_terms((0..8).map(|index| (vec![index, 7 - index], Integer::from(index + 1))))
            .unwrap();
        let right = ring
            .from_u64_terms(
                (0..8).map(|index| (vec![2 * index, index], Integer::from(2 * index + 1))),
            )
            .unwrap();
        let expected = ring.mul(&left, &right).unwrap();
        let generic_left = Polynomial {
            ring: ring.clone(),
            storage: PolynomialStorage::Generic(Arc::new(left.terms_map().clone())),
        };
        let generic_right = Polynomial {
            ring: ring.clone(),
            storage: PolynomialStorage::Generic(Arc::new(right.terms_map().clone())),
        };
        assert_eq!(ring.mul(&generic_left, &generic_right).unwrap(), expected);
    }

    #[test]
    fn adaptive_dense_generic_multiplication_agrees_with_native_flint_storage() {
        let ring = PolynomialRing::new(
            Z,
            ["x", "y", "z", "w"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let terms = (0..256_u64)
            .map(|index| {
                (
                    vec![index % 4, (index / 4) % 4, (index / 16) % 4, index / 64],
                    Integer::from(index + 1),
                )
            })
            .collect::<Vec<_>>();
        let native = ring.from_u64_terms(terms).unwrap();
        let expected = ring.mul(&native, &native).unwrap();
        let generic = Polynomial {
            ring: ring.clone(),
            storage: PolynomialStorage::Generic(Arc::new(native.terms_map().clone())),
        };
        assert_eq!(ring.mul(&generic, &generic).unwrap(), expected);
    }
}
