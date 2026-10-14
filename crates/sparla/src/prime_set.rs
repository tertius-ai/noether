// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Prime selection for the modular HNF pipeline.
//!
//! The Storjohann modular HNF picks primes whose product exceeds a
//! determinant-magnitude bound, runs the per-prime kernel, then CRT-
//! lifts via [`crate::crt`]. This module provides the bound, the
//! prime stream, and the deterministic primality test the stream
//! filters on.
//!
//! Every selected prime is at most `2³¹`, so the per-prime kernel can run
//! inside [`crate::storage::CsrFp`].

use std::cmp::Ordering;

use crate::elim::gauss_fp::pow_mod_p;
use crate::storage::{CsrZ, SparseRingStorage};
use crate::Integer;

/// The largest prime in the 31-bit range: `2³¹ − 1 = 2_147_483_647`.
/// This is `M_31`, the 8th Mersenne prime — and the entry point for
/// [`primes_descending`].
pub const LARGEST_31_BIT_PRIME: u32 = (1u32 << 31) - 1;

/// Deterministic Miller-Rabin primality test on a 32-bit unsigned
/// integer.
///
/// Witnesses `{2, 7, 61}` are deterministic for every `n <
/// 4_759_123_141`, well past our `2³¹` cap.
pub fn is_prime_u32(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 || n == 7 || n == 61 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    // Write n - 1 = d · 2^s with d odd.
    let mut d = n - 1;
    let mut s = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    'witness: for &a in &[2u32, 7, 61] {
        if a >= n {
            continue;
        }
        let mut x = pow_mod_p(a, d, n);
        if x == 1 || x == n - 1 {
            continue 'witness;
        }
        for _ in 0..s.saturating_sub(1) {
            x = ((x as u64 * x as u64) % n as u64) as u32;
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Iterator over primes in `[2, 2³¹)` in descending order.
///
/// Starts at [`LARGEST_31_BIT_PRIME`] and walks odd candidates down,
/// yielding each one that passes [`is_prime_u32`]. Terminates with
/// `2` and then `None`.
///
/// Primes drawn from the top of the range maximise the product
/// per-prime, so fewer primes are needed to exceed any determinant
/// bound — the standard recipe for modular reconstruction.
pub fn primes_descending() -> impl Iterator<Item = u32> {
    let mut candidate: i64 = LARGEST_31_BIT_PRIME as i64;
    // The two = "yielded 2"; -1 = exhausted.
    std::iter::from_fn(move || {
        while candidate >= 3 {
            let c = candidate as u32;
            candidate -= 2;
            if is_prime_u32(c) {
                return Some(c);
            }
        }
        if candidate == 2 {
            candidate = -1;
            Some(2)
        } else {
            None
        }
    })
}

/// Select distinct primes (descending from `2³¹`) whose product
/// strictly exceeds `bound`. Always returns at least one prime,
/// even when `bound == 0`.
pub fn select_primes_for_bound(bound: &Integer) -> Vec<u32> {
    let mut selected = Vec::new();
    let mut product = Integer::from(1);
    let mut iter = primes_descending();
    loop {
        let p = iter
            .next()
            .expect("select_primes_for_bound: exhausted primes below 2^31");
        selected.push(p);
        product *= p;
        if product > *bound {
            break;
        }
    }
    selected
}

/// Hadamard upper bound on `|det(M)|` for every maximal square
/// minor `M` of `matrix`.
///
/// Computes `⌈√(∏_i Σ_j A[i,j]²)⌉`. Squared form so the per-row
/// sqrt is avoided; the global integer sqrt at the end is exact
/// (FLINT's integer square root returns the floor; we bump by 1 when the
/// product isn't a perfect square).
///
/// Loose — but every entry of `HNF(A)` is bounded above by this, so
/// the CRT lift is sound when paired with
/// [`select_primes_for_bound`].
///
/// Edge cases:
/// - Zero rows contribute factor `0` to the product → bound is `0`,
///   correctly capturing `det = 0`.
/// - Empty matrix (`0 × 0`) yields bound `1`, matching `det = 1` of
///   the empty product convention.
pub fn hadamard_bound(matrix: &CsrZ) -> Integer {
    let mut prod_squared = Integer::from(1);
    for r in 0..matrix.rows() {
        let mut row_sum = Integer::from(0);
        for (_c, v) in matrix.iter_row(r) {
            row_sum += v.square();
        }
        prod_squared *= row_sum;
    }
    if prod_squared.cmp0() == Ordering::Equal {
        return Integer::from(0);
    }
    let floor = prod_squared.sqrt_floor();
    let sq = &floor * &floor;
    if sq == prod_squared {
        floor
    } else {
        floor + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: i64) -> Integer {
        Integer::from(v)
    }

    // -- is_prime_u32 -----------------------------------------------

    #[test]
    fn is_prime_u32_small_primes() {
        for &p in &[2u32, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
            assert!(is_prime_u32(p), "{} should be prime", p);
        }
    }

    #[test]
    fn is_prime_u32_small_composites() {
        for &n in &[
            0u32, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 21, 25, 27, 49, 51, 100,
        ] {
            assert!(!is_prime_u32(n), "{} should be composite", n);
        }
    }

    #[test]
    fn is_prime_u32_carmichael_561() {
        // 561 = 3 · 11 · 17 — a Carmichael number that fools the
        // naive Fermat test but not Miller-Rabin.
        assert!(!is_prime_u32(561));
    }

    #[test]
    fn is_prime_u32_largest_31_bit_prime() {
        assert!(is_prime_u32(LARGEST_31_BIT_PRIME));
    }

    #[test]
    fn is_prime_u32_near_top_composites() {
        // 2^31 - 3 = 2_147_483_645 = 5 · 429_496_729.
        assert!(!is_prime_u32(LARGEST_31_BIT_PRIME - 2));
        // 2^31 - 5 = 2_147_483_643 = 3 · ...
        assert!(!is_prime_u32(LARGEST_31_BIT_PRIME - 4));
    }

    #[test]
    fn is_prime_u32_known_large_prime() {
        // 1_000_000_007 — a classic competitive-programming modulus.
        assert!(is_prime_u32(1_000_000_007));
    }

    // -- primes_descending ------------------------------------------

    #[test]
    fn primes_descending_starts_with_mersenne_prime() {
        let mut it = primes_descending();
        assert_eq!(it.next(), Some(LARGEST_31_BIT_PRIME));
    }

    #[test]
    fn primes_descending_is_monotonic_and_prime() {
        let primes: Vec<u32> = primes_descending().take(50).collect();
        assert_eq!(primes.len(), 50);
        // Strictly decreasing.
        for win in primes.windows(2) {
            assert!(win[0] > win[1], "not descending: {} → {}", win[0], win[1]);
        }
        // Every value is prime by independent check.
        for p in primes {
            assert!(is_prime_u32(p));
        }
    }

    // -- select_primes_for_bound ------------------------------------

    #[test]
    fn select_primes_for_zero_bound_returns_one_prime() {
        let primes = select_primes_for_bound(&z(0));
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0], LARGEST_31_BIT_PRIME);
    }

    #[test]
    fn select_primes_product_exceeds_bound() {
        // Bound = 2^62. Need at least 3 primes from the top.
        let bound = Integer::from(1u64) << 62;
        let primes = select_primes_for_bound(&bound);
        let product: Integer = primes
            .iter()
            .fold(Integer::from(1), |acc, &p| acc * Integer::from(p));
        assert!(product > bound);
        // No proper prefix should already exceed the bound: the
        // loop adds primes until it does.
        if primes.len() > 1 {
            let prefix_product: Integer = primes[..primes.len() - 1]
                .iter()
                .fold(Integer::from(1), |acc, &p| acc * Integer::from(p));
            assert!(prefix_product <= bound);
        }
    }

    #[test]
    fn select_primes_distinct() {
        let primes = select_primes_for_bound(&(Integer::from(1u64) << 100));
        let mut sorted = primes.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(primes.len(), sorted.len(), "primes should be distinct");
    }

    // -- hadamard_bound ---------------------------------------------

    fn csr_from_dense(dense: &[Vec<i64>]) -> CsrZ {
        let rows = dense.len();
        let cols = if rows == 0 { 0 } else { dense[0].len() };
        let mut triples: Vec<(usize, usize, Integer)> = Vec::new();
        for (r, row) in dense.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                if v != 0 {
                    triples.push((r, c, z(v)));
                }
            }
        }
        CsrZ::from_coo(rows, cols, &triples)
    }

    #[test]
    fn hadamard_of_empty_matrix_is_one() {
        assert_eq!(hadamard_bound(&CsrZ::zeros(0, 0)), z(1));
    }

    #[test]
    fn hadamard_of_zero_matrix_is_zero() {
        assert_eq!(hadamard_bound(&CsrZ::zeros(3, 3)), z(0));
    }

    #[test]
    fn hadamard_of_identity_is_one() {
        let a = csr_from_dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
        assert_eq!(hadamard_bound(&a), z(1));
    }

    #[test]
    fn hadamard_2x2_known() {
        // [[3, 4], [5, 12]]: row norms² = 25, 169; product = 4225;
        // sqrt = 65 exactly. |det| = |36 - 20| = 16, so 65 is a
        // (loose) upper bound. ✓
        let a = csr_from_dense(&[vec![3, 4], vec![5, 12]]);
        assert_eq!(hadamard_bound(&a), z(65));
    }

    #[test]
    fn hadamard_ceils_imperfect_square() {
        // Single row [1, 1]: sum² = 2 → product = 2 → sqrt(2) ≈ 1.41
        // → ceil = 2. ✓
        let a = csr_from_dense(&[vec![1, 1]]);
        assert_eq!(hadamard_bound(&a), z(2));
    }

    #[test]
    fn hadamard_bounds_actual_det() {
        // Sanity-check the bound on several random small inputs.
        let cases: Vec<(Vec<Vec<i64>>, i64)> = vec![
            (vec![vec![2, 3], vec![1, 4]], 5),
            (vec![vec![1, 0, 2], vec![0, 3, 0], vec![4, 0, 5]], 9),
            (vec![vec![1, 2, 3], vec![0, 4, 5], vec![0, 0, 6]], 24),
        ];
        for (dense, det_abs) in cases {
            let a = csr_from_dense(&dense);
            let bound = hadamard_bound(&a);
            assert!(
                bound >= z(det_abs),
                "Hadamard bound {} is not ≥ |det| {} for {:?}",
                bound,
                det_abs,
                dense
            );
        }
    }
}
