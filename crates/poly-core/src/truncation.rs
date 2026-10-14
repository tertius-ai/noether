// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite monomial regions for fixed-truncation series.

use ahash::AHashMap as HashMap;
use core::fmt;

use crate::{ExponentOverflow, Monomial, MonomialArena, MonomialArenaError, MonomialId};

/// A strict finite truncation policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Truncation {
    /// Retain monomials of total degree strictly below `bound`.
    TotalDegree {
        /// Variable count.
        arity: usize,
        /// Exclusive total-degree bound.
        bound: u32,
    },
    /// Retain monomials of weighted degree strictly below `bound`.
    Weighted {
        /// Positive variable weights.
        weights: Box<[u32]>,
        /// Exclusive weighted-degree bound.
        bound: u64,
    },
    /// Retain exponents componentwise below the supplied bounds.
    Box {
        /// One positive exclusive bound per variable.
        bounds: Box<[u32]>,
    },
}

impl Truncation {
    /// Number of variables.
    pub fn arity(&self) -> usize {
        match self {
            Self::TotalDegree { arity, .. } => *arity,
            Self::Weighted { weights, .. } => weights.len(),
            Self::Box { bounds } => bounds.len(),
        }
    }

    /// Whether a monomial is retained.
    pub fn retains(&self, monomial: &Monomial) -> bool {
        if monomial.arity() != self.arity() {
            return false;
        }
        match self {
            Self::TotalDegree { bound, .. } => monomial
                .total_degree()
                .is_some_and(|degree| degree < u64::from(*bound)),
            Self::Weighted { weights, bound } => weights
                .iter()
                .enumerate()
                .try_fold(0_u64, |sum, (index, weight)| {
                    sum.checked_add(u64::from(monomial.exponent(index)) * u64::from(*weight))
                })
                .is_some_and(|degree| degree < *bound),
            Self::Box { bounds } => bounds
                .iter()
                .enumerate()
                .all(|(index, bound)| monomial.exponent(index) < *bound),
        }
    }

    /// Whether an interned monomial is retained, without materializing it.
    pub fn retains_id(&self, arena: &MonomialArena, monomial: MonomialId) -> bool {
        if arena.arity() != self.arity() || !arena.contains(monomial) {
            return false;
        }
        match self {
            Self::TotalDegree { bound, .. } => arena.total_degree(monomial) < u64::from(*bound),
            Self::Weighted { weights, bound } => weights
                .iter()
                .enumerate()
                .try_fold(0_u64, |sum, (index, weight)| {
                    sum.checked_add(u64::from(arena.exponent(monomial, index)) * u64::from(*weight))
                })
                .is_some_and(|degree| degree < *bound),
            Self::Box { bounds } => bounds
                .iter()
                .enumerate()
                .all(|(index, bound)| arena.exponent(monomial, index) < *bound),
        }
    }

    /// Whether a product of interned monomials is retained.
    ///
    /// This tests the truncation before interning the product, so discarded
    /// terms do not consume arena storage.
    pub fn retains_product_ids(
        &self,
        arena: &MonomialArena,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<bool, MonomialArenaError> {
        assert_eq!(arena.arity(), self.arity(), "truncation arity mismatch");
        assert!(arena.contains(left), "monomial identifier out of range");
        assert!(arena.contains(right), "monomial identifier out of range");
        match self {
            Self::TotalDegree { bound, .. } => arena
                .total_degree(left)
                .checked_add(arena.total_degree(right))
                .map(|degree| degree < u64::from(*bound))
                .ok_or(MonomialArenaError::ExponentOverflow),
            Self::Weighted { weights, bound } => {
                let mut degree = 0_u64;
                for (index, weight) in weights.iter().enumerate() {
                    let exponent = arena
                        .exponent(left, index)
                        .checked_add(arena.exponent(right, index))
                        .ok_or(MonomialArenaError::ExponentOverflow)?;
                    degree = degree
                        .checked_add(u64::from(exponent) * u64::from(*weight))
                        .ok_or(MonomialArenaError::ExponentOverflow)?;
                    if degree >= *bound {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Self::Box { bounds } => {
                for (index, bound) in bounds.iter().enumerate() {
                    let exponent = arena
                        .exponent(left, index)
                        .checked_add(arena.exponent(right, index))
                        .ok_or(MonomialArenaError::ExponentOverflow)?;
                    if exponent >= *bound {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }
}

/// Failure to construct a finite dense monomial region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegionError {
    /// A bound or weight was zero.
    ZeroBound,
    /// The requested region exceeded its explicit term cap.
    TermLimit {
        /// Explicit cap.
        limit: usize,
    },
    /// Exponent arithmetic overflowed.
    ExponentOverflow,
}

impl fmt::Display for RegionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBound => {
                formatter.write_str("truncation bounds and weights must be positive")
            }
            Self::TermLimit { limit } => {
                write!(
                    formatter,
                    "dense monomial region exceeds term limit {limit}"
                )
            }
            Self::ExponentOverflow => formatter.write_str("monomial exponent overflow"),
        }
    }
}

impl std::error::Error for RegionError {}

impl From<ExponentOverflow> for RegionError {
    fn from(_: ExponentOverflow) -> Self {
        Self::ExponentOverflow
    }
}

/// Enumerated finite monomial region with deterministic dense indices.
#[derive(Clone, Debug)]
pub struct DenseMonomialRegion {
    truncation: Truncation,
    monomials: Vec<Monomial>,
    indices: HashMap<Monomial, usize>,
}

impl DenseMonomialRegion {
    /// Enumerate `truncation`, failing before more than `term_limit` values.
    pub fn new(truncation: Truncation, term_limit: usize) -> Result<Self, RegionError> {
        validate(&truncation)?;
        let mut monomials = Vec::new();
        let mut current = vec![0_u32; truncation.arity()];
        enumerate(&truncation, 0, &mut current, term_limit, &mut monomials)?;
        let indices = monomials
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, monomial)| (monomial, index))
            .collect();
        Ok(Self {
            truncation,
            monomials,
            indices,
        })
    }

    /// Truncation policy.
    pub fn truncation(&self) -> &Truncation {
        &self.truncation
    }

    /// Number of retained monomials.
    pub fn len(&self) -> usize {
        self.monomials.len()
    }

    /// Whether the region is empty.
    pub fn is_empty(&self) -> bool {
        self.monomials.is_empty()
    }

    /// Monomial at one dense index.
    pub fn monomial(&self, index: usize) -> &Monomial {
        &self.monomials[index]
    }

    /// Dense index of a retained monomial.
    pub fn index_of(&self, monomial: &Monomial) -> Option<usize> {
        self.indices.get(monomial).copied()
    }

    /// Dense index of a retained product.
    pub fn product_index(&self, left: usize, right: usize) -> Result<Option<usize>, RegionError> {
        let product = self.monomials[left].checked_product(&self.monomials[right])?;
        Ok(self.index_of(&product))
    }
}

fn validate(truncation: &Truncation) -> Result<(), RegionError> {
    match truncation {
        Truncation::TotalDegree { arity, bound } => {
            if *arity == 0 || *bound == 0 {
                return Err(RegionError::ZeroBound);
            }
        }
        Truncation::Weighted { weights, bound } => {
            if weights.is_empty() || *bound == 0 || weights.contains(&0) {
                return Err(RegionError::ZeroBound);
            }
        }
        Truncation::Box { bounds } => {
            if bounds.is_empty() || bounds.contains(&0) {
                return Err(RegionError::ZeroBound);
            }
        }
    }
    Ok(())
}

fn enumerate(
    truncation: &Truncation,
    variable: usize,
    current: &mut [u32],
    term_limit: usize,
    output: &mut Vec<Monomial>,
) -> Result<(), RegionError> {
    if variable == current.len() {
        let monomial = Monomial::new(current.to_vec().into_boxed_slice());
        if truncation.retains(&monomial) {
            if output.len() == term_limit {
                return Err(RegionError::TermLimit { limit: term_limit });
            }
            output.push(monomial);
        }
        return Ok(());
    }
    let cap = coordinate_cap(truncation, variable)?;
    for exponent in 0..cap {
        current[variable] = exponent;
        let partial = Monomial::new(current.to_vec().into_boxed_slice());
        if partial_can_extend(truncation, &partial, variable) {
            enumerate(truncation, variable + 1, current, term_limit, output)?;
        }
    }
    current[variable] = 0;
    Ok(())
}

fn coordinate_cap(truncation: &Truncation, variable: usize) -> Result<u32, RegionError> {
    match truncation {
        Truncation::TotalDegree { bound, .. } => Ok(*bound),
        Truncation::Weighted { weights, bound } => {
            let cap = (*bound - 1) / u64::from(weights[variable]) + 1;
            u32::try_from(cap).map_err(|_| RegionError::ExponentOverflow)
        }
        Truncation::Box { bounds } => Ok(bounds[variable]),
    }
}

fn partial_can_extend(truncation: &Truncation, monomial: &Monomial, through: usize) -> bool {
    match truncation {
        Truncation::TotalDegree { bound, .. } => {
            (0..=through)
                .map(|index| u64::from(monomial.exponent(index)))
                .sum::<u64>()
                < u64::from(*bound)
        }
        Truncation::Weighted { weights, bound } => {
            (0..=through)
                .map(|index| u64::from(monomial.exponent(index)) * u64::from(weights[index]))
                .sum::<u64>()
                < *bound
        }
        Truncation::Box { .. } => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monomial(exponents: &[u32]) -> Monomial {
        Monomial::new(exponents.to_vec().into_boxed_slice())
    }

    #[test]
    fn total_degree_region_has_expected_size() {
        let region =
            DenseMonomialRegion::new(Truncation::TotalDegree { arity: 2, bound: 4 }, 100).unwrap();
        assert_eq!(region.len(), 10);
    }

    #[test]
    fn products_are_cut_off_before_indexing() {
        let region = DenseMonomialRegion::new(
            Truncation::Box {
                bounds: vec![2, 3].into_boxed_slice(),
            },
            100,
        )
        .unwrap();
        let x = region
            .index_of(&Monomial::new(vec![1, 0].into_boxed_slice()))
            .unwrap();
        assert_eq!(region.product_index(x, x).unwrap(), None);
    }

    #[test]
    fn truncations_check_arity_boundaries_and_overflow() {
        let total = Truncation::TotalDegree { arity: 2, bound: 4 };
        assert_eq!(total.arity(), 2);
        assert!(total.retains(&monomial(&[1, 2])));
        assert!(!total.retains(&monomial(&[1, 3])));
        assert!(!total.retains(&monomial(&[1])));

        let weighted = Truncation::Weighted {
            weights: vec![1, 3].into_boxed_slice(),
            bound: 7,
        };
        assert_eq!(weighted.arity(), 2);
        assert!(weighted.retains(&monomial(&[3, 1])));
        assert!(!weighted.retains(&monomial(&[1, 2])));

        let overflowing_weighted = Truncation::Weighted {
            weights: vec![u32::MAX, u32::MAX].into_boxed_slice(),
            bound: u64::MAX,
        };
        assert!(!overflowing_weighted.retains(&monomial(&[u32::MAX, u32::MAX,])));

        let boxed = Truncation::Box {
            bounds: vec![2, 4].into_boxed_slice(),
        };
        assert_eq!(boxed.arity(), 2);
        assert!(boxed.retains(&monomial(&[1, 3])));
        assert!(!boxed.retains(&monomial(&[2, 0])));
    }

    #[test]
    fn invalid_dense_regions_fail_with_specific_errors() {
        let invalid = [
            Truncation::TotalDegree { arity: 0, bound: 1 },
            Truncation::TotalDegree { arity: 1, bound: 0 },
            Truncation::Weighted {
                weights: Box::new([]),
                bound: 1,
            },
            Truncation::Weighted {
                weights: vec![0].into_boxed_slice(),
                bound: 1,
            },
            Truncation::Weighted {
                weights: vec![1].into_boxed_slice(),
                bound: 0,
            },
            Truncation::Box {
                bounds: Box::new([]),
            },
            Truncation::Box {
                bounds: vec![1, 0].into_boxed_slice(),
            },
        ];
        for truncation in invalid {
            assert_eq!(
                DenseMonomialRegion::new(truncation, 10).unwrap_err(),
                RegionError::ZeroBound
            );
        }

        assert_eq!(
            DenseMonomialRegion::new(Truncation::TotalDegree { arity: 2, bound: 3 }, 5)
                .unwrap_err(),
            RegionError::TermLimit { limit: 5 }
        );
        assert_eq!(
            DenseMonomialRegion::new(
                Truncation::Weighted {
                    weights: vec![1].into_boxed_slice(),
                    bound: u64::MAX,
                },
                10,
            )
            .unwrap_err(),
            RegionError::ExponentOverflow
        );

        assert_eq!(
            RegionError::ZeroBound.to_string(),
            "truncation bounds and weights must be positive"
        );
        assert_eq!(
            RegionError::TermLimit { limit: 17 }.to_string(),
            "dense monomial region exceeds term limit 17"
        );
        assert_eq!(
            RegionError::from(ExponentOverflow).to_string(),
            "monomial exponent overflow"
        );
    }

    #[test]
    fn dense_regions_have_deterministic_indices_for_each_policy() {
        let weighted = DenseMonomialRegion::new(
            Truncation::Weighted {
                weights: vec![1, 2].into_boxed_slice(),
                bound: 5,
            },
            100,
        )
        .unwrap();
        assert!(!weighted.is_empty());
        assert_eq!(weighted.truncation().arity(), 2);
        let x = weighted.index_of(&monomial(&[1, 0])).unwrap();
        let y = weighted.index_of(&monomial(&[0, 1])).unwrap();
        let xy = weighted.product_index(x, y).unwrap().unwrap();
        assert_eq!(weighted.monomial(xy), &monomial(&[1, 1]));
        assert_eq!(weighted.product_index(xy, y).unwrap(), None);

        let boxed = DenseMonomialRegion::new(
            Truncation::Box {
                bounds: vec![2, 3].into_boxed_slice(),
            },
            6,
        )
        .unwrap();
        assert_eq!(boxed.len(), 6);
        assert_eq!(boxed.monomial(0), &monomial(&[0, 0]));
        assert_eq!(boxed.index_of(&monomial(&[2, 0])), None);
    }

    #[test]
    fn arena_queries_cut_off_products_without_materializing_them() {
        let mut arena = MonomialArena::new(2, crate::MonomialOrder::GradedLexicographic);
        let x = arena.intern(monomial(&[1, 0])).unwrap();
        let y = arena.intern(monomial(&[0, 1])).unwrap();
        let x_squared = arena.intern(monomial(&[2, 0])).unwrap();

        let total = Truncation::TotalDegree { arity: 2, bound: 3 };
        assert!(total.retains_id(&arena, x_squared));
        assert!(total.retains_product_ids(&arena, x, y).unwrap());
        assert!(!total.retains_product_ids(&arena, x_squared, y).unwrap());

        let weighted = Truncation::Weighted {
            weights: vec![1, 2].into_boxed_slice(),
            bound: 3,
        };
        assert!(weighted.retains_id(&arena, x));
        assert!(weighted.retains_id(&arena, y));
        assert!(!weighted.retains_product_ids(&arena, x, y).unwrap());

        let boxed = Truncation::Box {
            bounds: vec![2, 2].into_boxed_slice(),
        };
        assert!(boxed.retains_product_ids(&arena, x, y).unwrap());
        assert!(!boxed.retains_product_ids(&arena, x, x).unwrap());

        let different_arity = Truncation::TotalDegree { arity: 1, bound: 3 };
        assert!(!different_arity.retains_id(&arena, x));
    }

    #[test]
    fn arena_product_queries_report_word_exponent_overflow() {
        let mut arena = MonomialArena::new(1, crate::MonomialOrder::Lexicographic);
        let largest = arena.intern(monomial(&[u32::MAX])).unwrap();

        let weighted = Truncation::Weighted {
            weights: vec![1].into_boxed_slice(),
            bound: u64::MAX,
        };
        assert_eq!(
            weighted.retains_product_ids(&arena, largest, largest),
            Err(MonomialArenaError::ExponentOverflow)
        );

        let boxed = Truncation::Box {
            bounds: vec![u32::MAX].into_boxed_slice(),
        };
        assert_eq!(
            boxed.retains_product_ids(&arena, largest, largest),
            Err(MonomialArenaError::ExponentOverflow)
        );
    }
}
