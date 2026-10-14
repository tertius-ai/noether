// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! The [`Perm`] type — an image-vector permutation of `0..degree`.
//!
//! Permutations here are stored as a flat `Vec<Point>` of length
//! `degree`, with `perm[i]` the image of `i` under the permutation.
//! The composition convention is **right action**:
//!
//! ```text
//! (g * h)[i] = h[g[i]]
//! ```
//!
//! which matches GAP and most Schreier–Sims expositions. This is the
//! single arithmetic primitive every higher-level operation here
//! (orbit, sift, transversal product) composes.

use crate::error::{GroupError, NotAPermutationReason, Result};
use crate::BigUInt;

/// A point in the permutation domain `0..degree`.
///
/// Pinned to `u32` for compact orbit and transversal storage while supporting
/// domains far larger than the release workloads.
pub type Point = u32;

/// A permutation of `0..degree`, stored as an image vector.
///
/// `images[i]` is the image of `i`. The empty vector is the unique
/// permutation of the empty set (degree 0). Composition acts on the
/// right: `(g * h)[i] = h[g[i]]`.
///
/// `Perm` carries no parent reference: the same vector is a valid
/// permutation in any group of matching degree. The group-side
/// validation lives on [`crate::PermGroup`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Perm {
    images: Vec<Point>,
}

impl Perm {
    /// Build a permutation from an image vector.
    ///
    /// # Errors
    ///
    /// [`GroupError::NotAPermutation`] if `images` contains a point
    /// `>= images.len()` (out of range) or repeats any value
    /// (not a bijection).
    pub fn from_images(images: &[Point]) -> Result<Perm> {
        Self::from_images_generator(images, 0)
    }

    /// Like [`from_images`](Self::from_images), but tags any error
    /// with the supplied generator index. Used by group constructors
    /// so the error pinpoints *which* generator was wrong.
    pub fn from_images_generator(images: &[Point], generator: usize) -> Result<Perm> {
        let degree = images.len();
        let mut seen = vec![false; degree];
        for &x in images {
            let xi = x as usize;
            if xi >= degree {
                return Err(GroupError::NotAPermutation {
                    generator,
                    reason: NotAPermutationReason::OutOfRange { image: xi, degree },
                });
            }
            if seen[xi] {
                return Err(GroupError::NotAPermutation {
                    generator,
                    reason: NotAPermutationReason::RepeatedImage { image: xi },
                });
            }
            seen[xi] = true;
        }
        Ok(Perm {
            images: images.to_vec(),
        })
    }

    /// The identity permutation on `0..degree`.
    pub fn identity(degree: usize) -> Perm {
        Perm {
            images: (0..degree as Point).collect(),
        }
    }

    /// The permutation's degree — the size of its underlying domain.
    pub fn degree(&self) -> usize {
        self.images.len()
    }

    /// `true` iff this is the identity permutation on its domain.
    pub fn is_identity(&self) -> bool {
        self.images
            .iter()
            .enumerate()
            .all(|(i, &x)| x as usize == i)
    }

    /// Borrow the image vector.
    pub fn images(&self) -> &[Point] {
        &self.images
    }

    /// The image of `p` under this permutation, panicking only on the
    /// invariant violation "`p` is outside the domain". The orbit /
    /// sift hot paths call this for every step.
    #[inline]
    pub fn apply(&self, p: Point) -> Point {
        self.images[p as usize]
    }

    /// Compose two permutations: `(self * other)[i] = other[self[i]]`
    /// — right-action convention.
    ///
    /// # Errors
    ///
    /// [`GroupError::DegreeMismatch`] if the two permutations act on
    /// domains of different sizes.
    pub fn compose(&self, other: &Perm) -> Result<Perm> {
        if self.degree() != other.degree() {
            return Err(GroupError::DegreeMismatch {
                got: other.degree(),
                expected: self.degree(),
            });
        }
        let mut images = vec![0; self.degree()];
        for (i, slot) in images.iter_mut().enumerate() {
            *slot = other.images[self.images[i] as usize];
        }
        Ok(Perm { images })
    }

    /// Compose with a permutation guaranteed to have matching degree
    /// (the orbit / sift hot path).
    #[inline]
    pub(crate) fn compose_unchecked(&self, other: &Perm) -> Perm {
        debug_assert_eq!(self.degree(), other.degree());
        let mut images = vec![0; self.degree()];
        for (i, slot) in images.iter_mut().enumerate() {
            *slot = other.images[self.images[i] as usize];
        }
        Perm { images }
    }

    /// Compose `self * other` into a caller-supplied scratch buffer,
    /// avoiding the per-call `Vec<Point>` allocation
    /// [`compose_unchecked`] does. The buffer is resized to
    /// `self.degree()`; previous contents are overwritten.
    ///
    /// The result is written into `scratch`; no new `Perm` is built.
    /// Use this on inner loops where the scratch buffer can be reused
    /// (Schreier-tree extension, sift, element enumeration).
    #[inline]
    pub(crate) fn compose_into(&self, other: &Perm, scratch: &mut Vec<Point>) {
        debug_assert_eq!(self.degree(), other.degree());
        let n = self.degree();
        scratch.clear();
        scratch.resize(n, 0);
        for (i, slot) in scratch.iter_mut().enumerate() {
            *slot = other.images[self.images[i] as usize];
        }
    }

    /// Replace `self` with `self * other`, reusing `scratch` across calls.
    #[inline]
    pub(crate) fn compose_assign_unchecked(&mut self, other: &Perm, scratch: &mut Vec<Point>) {
        self.compose_into(other, scratch);
        core::mem::swap(&mut self.images, scratch);
    }

    /// Construct a `Perm` from an owned image vector without
    /// re-validation. Caller guarantees `images` is a permutation of
    /// `0..images.len()`. Hot-path constructor used by
    /// [`crate::group::ElementIter`] to hand its scratch buffer back
    /// out as a Perm without re-allocating.
    #[inline]
    pub(crate) fn from_images_unchecked(images: Vec<Point>) -> Perm {
        Perm { images }
    }

    /// The inverse permutation.
    pub fn inverse(&self) -> Perm {
        let mut images = vec![0 as Point; self.degree()];
        for (i, &x) in self.images.iter().enumerate() {
            images[x as usize] = i as Point;
        }
        Perm { images }
    }

    /// Whether this is an *even* permutation (sign `+1`, i.e. lies in `A_n`).
    /// Computed from the cycle decomposition: a length-`ℓ` cycle is `ℓ−1`
    /// transpositions, so the sign is `(−1)^Σ(ℓ−1)`.
    pub fn is_even(&self) -> bool {
        let n = self.degree();
        let mut visited = vec![false; n];
        let mut transpositions: usize = 0;
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut j = start;
            let mut len = 0usize;
            while !visited[j] {
                visited[j] = true;
                j = self.images[j] as usize;
                len += 1;
            }
            transpositions += len - 1;
        }
        transpositions.is_multiple_of(2)
    }

    /// The conjugate `self^by = by⁻¹ · self · by` (right-action convention,
    /// matching [`compose`](Self::compose)). The substrate of normal closures
    /// and the derived / lower-central series.
    ///
    /// # Errors
    /// [`GroupError::DegreeMismatch`] if `self` and `by` act on different domains.
    pub fn conjugate(&self, by: &Perm) -> Result<Perm> {
        by.inverse().compose(self)?.compose(by)
    }

    /// Conjugate using a caller-supplied inverse on an already checked common
    /// domain. Hot orbit traversals use this to avoid recomputing the inverse
    /// and allocating two intermediate permutations for every edge.
    #[inline]
    pub(crate) fn conjugate_unchecked_with_inverse(&self, by: &Perm, by_inverse: &Perm) -> Perm {
        debug_assert_eq!(self.degree(), by.degree());
        debug_assert_eq!(self.degree(), by_inverse.degree());
        let mut images = vec![0; self.degree()];
        for (point, image) in images.iter_mut().enumerate() {
            *image = by.images[self.images[by_inverse.images[point] as usize] as usize];
        }
        Perm { images }
    }

    /// The commutator `[self, other] = self⁻¹ · other⁻¹ · self · other`.
    ///
    /// # Errors
    /// [`GroupError::DegreeMismatch`] if the two permutations act on different domains.
    pub fn commutator(&self, other: &Perm) -> Result<Perm> {
        self.inverse()
            .compose(&other.inverse())?
            .compose(self)?
            .compose(other)
    }

    /// The (multiplicative) order of this permutation — the smallest
    /// `k ≥ 1` with `σ^k = identity`. Computed as the LCM of cycle
    /// lengths in the cycle decomposition; runs in `O(degree)`.
    pub fn order(&self) -> BigUInt {
        let n = self.degree();
        let mut visited = vec![false; n];
        let mut result = BigUInt::one();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut cycle_len: usize = 0;
            let mut p = start;
            while !visited[p] {
                visited[p] = true;
                p = self.images[p] as usize;
                cycle_len += 1;
            }
            // gcd(result, cycle_len) = gcd(result mod cycle_len, cycle_len),
            // so the unbounded accumulator only needs division by a word.
            let remainder = result.divmod_u64(cycle_len as u64).1 as usize;
            let (mut a, mut b) = (cycle_len, remainder);
            while b != 0 {
                let t = b;
                b = a % b;
                a = t;
            }
            result.mul_assign_u64((cycle_len / a) as u64);
        }
        result
    }
}
