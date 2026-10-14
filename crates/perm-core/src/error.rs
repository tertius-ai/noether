// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Typed errors for `bsgs`.
//!
//! Every fallible entry point returns [`Result<T, GroupError>`]. Malformed
//! input maps to a variant here rather than causing a panic.

use core::fmt;

/// The error type for every fallible `bsgs` operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupError {
    /// A generator (or a permutation being sifted / tested for
    /// membership) had a degree that did not match the group's
    /// declared degree. Carries the offending and expected lengths.
    DegreeMismatch {
        /// The mismatched length.
        got: usize,
        /// The expected length.
        expected: usize,
    },
    /// A would-be generator's image vector contained a point outside
    /// `0..degree`, or repeated a point — i.e. was not a permutation.
    NotAPermutation {
        /// The generator index in the input list whose images were
        /// invalid; `0` for a singleton input.
        generator: usize,
        /// A short description of *how* it failed.
        reason: NotAPermutationReason,
    },
    /// A base point passed to a routine that requires it (e.g. the
    /// orbit-along-a-base-step accessors) was not actually in the
    /// group's base.
    BasePointMissing {
        /// The point that was requested but not present.
        point: usize,
    },
    /// A subset action requested more points than occur in the action domain.
    InvalidSubsetSize {
        /// Requested subset cardinality.
        size: usize,
        /// Permutation-action degree.
        degree: usize,
    },
    /// A value presented as a set repeated one of its points.
    DuplicateSubsetPoint {
        /// Repeated point.
        point: usize,
    },
    /// Distinct pair seeds lay in the same point orbit, leaving it ambiguous
    /// whether the endpoints were meant to retain colors or be unordered.
    AmbiguousPairSeeds {
        /// First seed point.
        left: usize,
        /// Second seed point.
        right: usize,
    },
    /// A routine that requires `H ≤ G` (e.g. [`crate::coset_transversal`])
    /// was given a purported subgroup with a generator not in the group.
    NotASubgroup,
    /// A checked group operation was asked to use a permutation of the right
    /// degree that does not belong to the group.
    ElementNotInGroup,
    /// An operation requiring a prime was given a composite or subunit value.
    InvalidPrime {
        /// Rejected value.
        value: u64,
    },
    /// A coset enumeration ([`crate::coset_transversal`]) exceeded its
    /// safety cap on the index `[G:H]`. Carries the cap that was hit.
    IndexTooLarge {
        /// The maximum number of cosets the enumeration was allowed.
        cap: usize,
    },
    /// An exact group-element enumeration would exceed its public safety cap.
    /// The operation fails before materializing a partial answer.
    EnumerationLimitExceeded {
        /// Stable name of the requested enumeration.
        operation: &'static str,
        /// Maximum group order accepted by the enumeration.
        cap: usize,
    },
    /// A partition-backtrack search hit its
    /// refinement / node cap before completing. Analogous to
    /// [`IndexTooLarge`](Self::IndexTooLarge): an incomplete search
    /// returns this rather than a wrong or partial answer. Carries the
    /// node cap that was hit.
    SearchBudgetExceeded {
        /// The maximum number of search-tree nodes the backtrack was
        /// allowed to visit.
        cap: usize,
    },
    /// A candidate conjugator / normalizing generator produced by a
    /// backtrack search failed its mandatory sift-verification against
    /// the target. An unverified witness is never handed out. This is an
    /// internal-consistency invariant, not reachable
    /// from user input.
    WitnessUnverified,
}

/// The specific failure mode behind [`GroupError::NotAPermutation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotAPermutationReason {
    /// An image referenced a point `>= degree`.
    OutOfRange {
        /// The offending image.
        image: usize,
        /// The degree the image was checked against.
        degree: usize,
    },
    /// Two distinct preimages produced the same image, so the map is
    /// not a bijection on `0..degree`.
    RepeatedImage {
        /// The image that appeared twice.
        image: usize,
    },
}

impl fmt::Display for GroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GroupError::DegreeMismatch { got, expected } => {
                write!(
                    f,
                    "permutation degree mismatch: got {got}, expected {expected}"
                )
            }
            GroupError::NotAPermutation { generator, reason } => {
                write!(f, "generator {generator} is not a permutation: {reason}")
            }
            GroupError::BasePointMissing { point } => {
                write!(f, "base point {point} is not in the group's base")
            }
            GroupError::InvalidSubsetSize { size, degree } => {
                write!(f, "subset size {size} exceeds permutation degree {degree}")
            }
            GroupError::DuplicateSubsetPoint { point } => {
                write!(f, "subset point {point} occurs more than once")
            }
            GroupError::AmbiguousPairSeeds { left, right } => write!(
                f,
                "distinct pair seeds {left} and {right} lie in the same point orbit"
            ),
            GroupError::NotASubgroup => {
                write!(f, "the purported subgroup has a generator not in the group")
            }
            GroupError::ElementNotInGroup => {
                write!(f, "the permutation is not an element of the group")
            }
            GroupError::InvalidPrime { value } => {
                write!(f, "{value} is not prime")
            }
            GroupError::IndexTooLarge { cap } => {
                write!(f, "coset enumeration exceeded the index cap of {cap}")
            }
            GroupError::EnumerationLimitExceeded { operation, cap } => {
                write!(
                    f,
                    "{operation} requires enumerating more than {cap} group elements"
                )
            }
            GroupError::SearchBudgetExceeded { cap } => {
                write!(
                    f,
                    "partition-backtrack search exceeded the node cap of {cap}"
                )
            }
            GroupError::WitnessUnverified => {
                write!(
                    f,
                    "a backtrack witness failed its sift-verification against the target"
                )
            }
        }
    }
}

impl fmt::Display for NotAPermutationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NotAPermutationReason::OutOfRange { image, degree } => {
                write!(f, "image {image} is outside the domain 0..{degree}")
            }
            NotAPermutationReason::RepeatedImage { image } => {
                write!(f, "image {image} occurs more than once")
            }
        }
    }
}

impl std::error::Error for GroupError {}

/// Shorthand for `Result<T, GroupError>`.
pub type Result<T> = core::result::Result<T, GroupError>;
