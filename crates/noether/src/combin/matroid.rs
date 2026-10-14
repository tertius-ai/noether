// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact finite matroids represented by their bases.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::VertexId;
use crate::commalg::{
    MonomialOrder, Polynomial, PolynomialContextError, PolynomialError, PolynomialRing,
};
use crate::{Integer, IntegerRing, NoetherError, SemanticType, TypeDescription, Z};

/// Largest ground-set size accepted by the exhaustive 0.1 representation.
///
/// Circuits and the Tutte polynomial inspect every subset. Keeping the bound on
/// the type, rather than only on individual methods, makes those accepted
/// operations total for every constructed matroid.
pub const MAX_MATROID_ORDER: usize = 24;

/// A failure while constructing or querying an exact finite matroid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatroidError {
    /// The ground set exceeds the bounded exhaustive representation.
    GroundSetTooLarge {
        /// Number of supplied elements.
        order: usize,
        /// Largest accepted number of elements.
        max: usize,
    },
    /// The ground set contains the same element more than once.
    DuplicateGroundElement {
        /// Repeated element identifier.
        element: VertexId,
    },
    /// A nonempty ground set was supplied without any bases.
    NoBases,
    /// Bases have unequal cardinalities.
    UnequalBasisSizes {
        /// Cardinality of the first basis.
        expected: usize,
        /// Cardinality of the rejected basis.
        actual: usize,
    },
    /// A basis contains the same element more than once.
    DuplicateBasisElement {
        /// Repeated element identifier.
        element: VertexId,
    },
    /// A supplied element does not belong to the ground set.
    ElementOutsideGroundSet {
        /// Rejected element identifier.
        element: VertexId,
    },
    /// The supplied family does not satisfy basis exchange.
    BasisExchange {
        /// First basis in a failed exchange pair.
        left: Vec<VertexId>,
        /// Second basis in a failed exchange pair.
        right: Vec<VertexId>,
        /// Element of `left ∖ right` that cannot be exchanged.
        element: VertexId,
    },
    /// The fixed integer polynomial context could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Construction of the exact polynomial result failed.
    Polynomial(PolynomialError),
}

impl fmt::Display for MatroidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GroundSetTooLarge { order, max } => {
                write!(
                    f,
                    "matroid ground-set size {order} exceeds the 0.1 limit {max}"
                )
            }
            Self::DuplicateGroundElement { element } => {
                write!(f, "matroid ground-set element {element} is duplicated")
            }
            Self::NoBases => f.write_str("a matroid needs at least one basis"),
            Self::UnequalBasisSizes { expected, actual } => write!(
                f,
                "matroid basis has size {actual}; every basis must have size {expected}"
            ),
            Self::DuplicateBasisElement { element } => {
                write!(f, "matroid basis contains element {element} more than once")
            }
            Self::ElementOutsideGroundSet { element } => {
                write!(f, "matroid element {element} is outside the ground set")
            }
            Self::BasisExchange {
                left,
                right,
                element,
            } => write!(
                f,
                "matroid bases {left:?} and {right:?} fail basis exchange at element {element}"
            ),
            Self::PolynomialContext(error) => error.fmt(f),
            Self::Polynomial(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for MatroidError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PolynomialContextError> for MatroidError {
    fn from(error: PolynomialContextError) -> Self {
        Self::PolynomialContext(error)
    }
}

impl From<PolynomialError> for MatroidError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

/// A finite labeled matroid with exact basis data.
///
/// Elements are stable nonnegative identifiers, matching the public graph and
/// poset convention. The ground set and every basis are canonicalized, and the
/// complete basis-exchange axiom is checked at construction. Internally, bases
/// use packed masks; labels remain visible across deletion and contraction.
#[derive(Clone, Debug)]
pub struct Matroid {
    ground_set: Vec<VertexId>,
    bases: Vec<u64>,
    rank: usize,
}

#[derive(Serialize, Deserialize)]
struct MatroidRepr {
    ground_set: Vec<VertexId>,
    bases: Vec<Vec<VertexId>>,
}

impl Matroid {
    /// Construct and validate a matroid from its complete family of bases.
    pub fn from_bases(
        mut ground_set: Vec<VertexId>,
        bases: Vec<Vec<VertexId>>,
    ) -> Result<Self, MatroidError> {
        if ground_set.len() > MAX_MATROID_ORDER {
            return Err(MatroidError::GroundSetTooLarge {
                order: ground_set.len(),
                max: MAX_MATROID_ORDER,
            });
        }
        ground_set.sort_unstable();
        if let Some(pair) = ground_set.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(MatroidError::DuplicateGroundElement { element: pair[0] });
        }
        if bases.is_empty() {
            return Err(MatroidError::NoBases);
        }

        let positions: BTreeMap<_, _> = ground_set
            .iter()
            .copied()
            .enumerate()
            .map(|(position, element)| (element, position))
            .collect();
        let rank = bases[0].len();
        let mut masks = BTreeSet::new();
        for basis in bases {
            if basis.len() != rank {
                return Err(MatroidError::UnequalBasisSizes {
                    expected: rank,
                    actual: basis.len(),
                });
            }
            let mut mask = 0_u64;
            for element in basis {
                let position = positions
                    .get(&element)
                    .copied()
                    .ok_or(MatroidError::ElementOutsideGroundSet { element })?;
                let bit = 1_u64 << position;
                if mask & bit != 0 {
                    return Err(MatroidError::DuplicateBasisElement { element });
                }
                mask |= bit;
            }
            masks.insert(mask);
        }

        let value = Self {
            ground_set,
            bases: masks.into_iter().collect(),
            rank,
        };
        value.verify_basis_exchange()?;
        Ok(value)
    }

    /// Canonical ordered ground set.
    #[must_use]
    pub fn ground_set(&self) -> &[VertexId] {
        &self.ground_set
    }

    /// Canonical family of bases in lexicographic element order.
    #[must_use]
    pub fn bases(&self) -> Vec<Vec<VertexId>> {
        let mut bases: Vec<_> = self
            .bases
            .iter()
            .map(|&mask| self.elements_from_mask(mask))
            .collect();
        bases.sort();
        bases
    }

    /// Rank of one subset of the ground set.
    pub fn rank(&self, subset: &[VertexId]) -> Result<usize, MatroidError> {
        let mask = self.subset_mask(subset)?;
        Ok(self.rank_mask(mask))
    }

    /// Minimal dependent subsets, in deterministic order.
    #[must_use]
    pub fn circuits(&self) -> Vec<Vec<VertexId>> {
        let mut output = Vec::new();
        for subset in 0_u64..(1_u64 << self.ground_set.len()) {
            if self.rank_mask(subset) == subset.count_ones() as usize {
                continue;
            }
            let minimal = (0..self.ground_set.len())
                .filter(|position| subset & (1_u64 << position) != 0)
                .all(|position| {
                    let smaller = subset & !(1_u64 << position);
                    self.rank_mask(smaller) == smaller.count_ones() as usize
                });
            if minimal {
                output.push(self.elements_from_mask(subset));
            }
        }
        output
    }

    /// Delete one labeled element while preserving every remaining label.
    pub fn delete(&self, element: VertexId) -> Result<Self, MatroidError> {
        let position = self.position(element)?;
        let bit = 1_u64 << position;
        let retained_rank = self
            .bases
            .iter()
            .map(|basis| (basis & !bit).count_ones() as usize)
            .max()
            .unwrap_or(0);
        let bases = self
            .bases
            .iter()
            .filter(|basis| (**basis & !bit).count_ones() as usize == retained_rank)
            .map(|basis| self.elements_from_mask(*basis & !bit))
            .collect();
        let ground_set = self
            .ground_set
            .iter()
            .copied()
            .filter(|candidate| *candidate != element)
            .collect();
        Self::from_bases(ground_set, bases)
    }

    /// Contract one labeled element while preserving every remaining label.
    pub fn contract(&self, element: VertexId) -> Result<Self, MatroidError> {
        let position = self.position(element)?;
        let bit = 1_u64 << position;
        let element_rank = self.rank_mask(bit);
        let contracted_rank = self.rank - element_rank;
        let bases = self
            .bases
            .iter()
            .filter(|basis| {
                let remainder = **basis & !bit;
                remainder.count_ones() as usize == contracted_rank
            })
            .map(|basis| self.elements_from_mask(*basis & !bit))
            .collect();
        let ground_set = self
            .ground_set
            .iter()
            .copied()
            .filter(|candidate| *candidate != element)
            .collect();
        Self::from_bases(ground_set, bases)
    }

    /// Exact Tutte polynomial in the canonical context `Z[x, y]`.
    pub fn tutte_polynomial(&self) -> Result<Polynomial<IntegerRing>, MatroidError> {
        let ring = PolynomialRing::new(Z, ["x", "y"], MonomialOrder::Lexicographic)?;
        let mut coefficients: BTreeMap<(usize, usize), Integer> = BTreeMap::new();
        for subset in 0_u64..(1_u64 << self.ground_set.len()) {
            let subset_rank = self.rank_mask(subset);
            let x_power = self.rank - subset_rank;
            let y_power = subset.count_ones() as usize - subset_rank;
            for x_degree in 0..=x_power {
                for y_degree in 0..=y_power {
                    let mut coefficient = &Integer::from(binomial(x_power, x_degree))
                        * &Integer::from(binomial(y_power, y_degree));
                    if (x_power - x_degree + y_power - y_degree) % 2 == 1 {
                        coefficient = -coefficient;
                    }
                    let entry = coefficients
                        .entry((x_degree, y_degree))
                        .or_insert_with(Integer::zero);
                    *entry = &*entry + &coefficient;
                }
            }
        }
        ring.from_u64_terms(coefficients.into_iter().filter_map(
            |((x_degree, y_degree), coefficient)| {
                (!coefficient.is_zero())
                    .then_some((vec![x_degree as u64, y_degree as u64], coefficient))
            },
        ))
        .map_err(MatroidError::from)
    }

    fn position(&self, element: VertexId) -> Result<usize, MatroidError> {
        self.ground_set
            .binary_search(&element)
            .map_err(|_| MatroidError::ElementOutsideGroundSet { element })
    }

    fn subset_mask(&self, subset: &[VertexId]) -> Result<u64, MatroidError> {
        let mut mask = 0_u64;
        for &element in subset {
            mask |= 1_u64 << self.position(element)?;
        }
        Ok(mask)
    }

    fn rank_mask(&self, subset: u64) -> usize {
        self.bases
            .iter()
            .map(|basis| (basis & subset).count_ones() as usize)
            .max()
            .unwrap_or(0)
    }

    fn elements_from_mask(&self, mask: u64) -> Vec<VertexId> {
        self.ground_set
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(position, element)| (mask & (1_u64 << position) != 0).then_some(element))
            .collect()
    }

    fn verify_basis_exchange(&self) -> Result<(), MatroidError> {
        let family: BTreeSet<_> = self.bases.iter().copied().collect();
        for &left in &self.bases {
            for &right in &self.bases {
                let left_only = left & !right;
                let right_only = right & !left;
                for position in 0..self.ground_set.len() {
                    let removed = 1_u64 << position;
                    if left_only & removed == 0 {
                        continue;
                    }
                    let exchange_exists = (0..self.ground_set.len()).any(|replacement| {
                        let added = 1_u64 << replacement;
                        right_only & added != 0 && family.contains(&((left & !removed) | added))
                    });
                    if !exchange_exists {
                        return Err(MatroidError::BasisExchange {
                            left: self.elements_from_mask(left),
                            right: self.elements_from_mask(right),
                            element: self.ground_set[position],
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

impl PartialEq for Matroid {
    fn eq(&self, other: &Self) -> bool {
        self.ground_set == other.ground_set && self.bases == other.bases
    }
}

impl Eq for Matroid {}

impl fmt::Display for Matroid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matroid(order={}, rank={}, bases={})",
            self.ground_set.len(),
            self.rank,
            self.bases.len()
        )
    }
}

impl Serialize for Matroid {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        MatroidRepr {
            ground_set: self.ground_set.clone(),
            bases: self.bases(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Matroid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = MatroidRepr::deserialize(deserializer)?;
        Self::from_bases(repr.ground_set, repr.bases).map_err(serde::de::Error::custom)
    }
}

static MATROID_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/Matroid@1",
    rust_path: "noether::combin::Matroid",
    python_path: Some("noether.combin.Matroid"),
};

impl SemanticType for Matroid {
    const TYPE_ID: &'static str = "noether/combin/Matroid@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MATROID_DESCRIPTION
    }
}

fn binomial(n: usize, k: usize) -> usize {
    let k = k.min(n - k);
    (0..k).fold(1, |value, index| value * (n - index) / (index + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(rank: usize, order: usize) -> Matroid {
        let mut bases = Vec::new();
        subsets(order, rank, 0, &mut Vec::new(), &mut bases);
        Matroid::from_bases((0..order as VertexId).collect(), bases).unwrap()
    }

    fn subsets(
        order: usize,
        size: usize,
        start: usize,
        current: &mut Vec<VertexId>,
        output: &mut Vec<Vec<VertexId>>,
    ) {
        if current.len() == size {
            output.push(current.clone());
            return;
        }
        for element in start..order {
            current.push(element as VertexId);
            subsets(order, size, element + 1, current, output);
            current.pop();
        }
    }

    #[test]
    fn validates_basis_exchange_and_canonicalizes_input() {
        let value = Matroid::from_bases(
            vec![30, 10, 20],
            vec![vec![20, 10], vec![30, 10], vec![30, 20], vec![10, 20]],
        )
        .unwrap();
        assert_eq!(value.ground_set(), &[10, 20, 30]);
        assert_eq!(
            value.bases(),
            vec![vec![10, 20], vec![10, 30], vec![20, 30]]
        );

        let error =
            Matroid::from_bases(vec![0, 1, 2, 3], vec![vec![0, 1], vec![2, 3]]).unwrap_err();
        assert!(matches!(error, MatroidError::BasisExchange { .. }));
    }

    #[test]
    fn minimal_uniform_case_is_hand_checkable() {
        let value = uniform(1, 2);
        assert_eq!(value.rank(&[]).unwrap(), 0);
        assert_eq!(value.rank(&[0, 1]).unwrap(), 1);
        assert_eq!(value.circuits(), vec![vec![0, 1]]);
    }

    #[test]
    fn minors_preserve_labels() {
        let value = Matroid::from_bases(
            vec![10, 20, 30],
            vec![vec![10, 20], vec![10, 30], vec![20, 30]],
        )
        .unwrap();
        assert_eq!(value.delete(30).unwrap().bases(), vec![vec![10, 20]]);
        assert_eq!(
            value.contract(10).unwrap().bases(),
            vec![vec![20], vec![30]]
        );
    }

    #[test]
    fn loops_and_coloops_have_correct_minors() {
        let value = Matroid::from_bases(vec![0, 1, 2], vec![vec![0, 1], vec![0, 2]]).unwrap();
        assert_eq!(value.delete(0).unwrap().bases(), vec![vec![1], vec![2]]);
        assert_eq!(value.contract(0).unwrap().bases(), vec![vec![1], vec![2]]);

        let looped = Matroid::from_bases(vec![0, 1], vec![vec![0]]).unwrap();
        assert_eq!(looped.delete(1).unwrap(), looped.contract(1).unwrap());
    }

    #[test]
    fn tutte_polynomial_is_typed_integer_polynomial() {
        let polynomial = uniform(2, 3).tutte_polynomial().unwrap();
        assert_eq!(polynomial.ring().variables(), &["x", "y"]);
        assert_eq!(
            polynomial
                .coefficient(&[Integer::from(2), Integer::zero()])
                .unwrap(),
            Integer::one()
        );
        assert_eq!(
            polynomial
                .coefficient(&[Integer::one(), Integer::zero()])
                .unwrap(),
            Integer::one()
        );
        assert_eq!(
            polynomial
                .coefficient(&[Integer::zero(), Integer::one()])
                .unwrap(),
            Integer::one()
        );
    }

    #[test]
    fn tutte_polynomial_satisfies_counting_identities() {
        for value in [
            uniform(2, 4),
            Matroid::from_bases(vec![0, 1, 2], vec![vec![0, 1], vec![0, 2]]).unwrap(),
            Matroid::from_bases(vec![0, 1], vec![vec![0]]).unwrap(),
        ] {
            let polynomial = value.tutte_polynomial().unwrap();
            let evaluate = |x: usize, y: usize| {
                let mut sum = Integer::zero();
                for x_degree in 0..=value.rank {
                    for y_degree in 0..=value.ground_set.len() - value.rank {
                        let coefficient = polynomial
                            .coefficient(&[Integer::from(x_degree), Integer::from(y_degree)])
                            .unwrap();
                        let term = &coefficient * &Integer::from(x).pow(x_degree as u64);
                        let term = &term * &Integer::from(y).pow(y_degree as u64);
                        sum = &sum + &term;
                    }
                }
                sum
            };
            assert_eq!(evaluate(1, 1), Integer::from(value.bases.len()));
            assert_eq!(
                evaluate(2, 2),
                Integer::from(2_u64.pow(value.ground_set.len() as u32))
            );
        }
    }

    #[test]
    fn serialization_revalidates_the_matroid() {
        let value = uniform(2, 4);
        let rebuilt: Matroid =
            serde_json::from_str(&serde_json::to_string(&value).unwrap()).unwrap();
        assert_eq!(rebuilt, value);
        let invalid = r#"{"ground_set":[0,1,2,3],"bases":[[0,1],[2,3]]}"#;
        assert!(serde_json::from_str::<Matroid>(invalid).is_err());
    }
}
