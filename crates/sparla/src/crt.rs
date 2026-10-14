// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Chinese Remainder Theorem reconstruction.
//!
//! Combines per-prime residues back into an integer. The modular
//! HNF pipeline runs sparse Gaussian over many primes and folds the
//! per-prime results here; [`symmetric_lift`] then maps the result
//! into `(-m/2, m/2]` so the recovered integer matches the
//! original (signed) HNF entry.

use std::cmp::Ordering;

use crate::Integer;

/// Combine two residues with coprime moduli.
///
/// Given `r₁ mod m₁` and `r₂ mod m₂` with `gcd(m₁, m₂) = 1`, returns
/// the unique `r ∈ [0, m₁·m₂)` satisfying both congruences.
///
/// # Panics
///
/// Panics if `m₁` and `m₂` are not coprime — the inverse of `m₁`
/// mod `m₂` doesn't exist in that case.
pub fn crt_pair(r1: &Integer, m1: &Integer, r2: &Integer, m2: &Integer) -> Integer {
    // r = r1 + m1 · ((r2 - r1) · m1^{-1} mod m2).
    let m1_inv: Integer = m1.clone().invert(m2).expect("crt_pair: moduli not coprime");
    let diff = r2 - r1;
    let mut coeff: Integer = diff * m1_inv;
    coeff %= m2;
    if coeff.cmp0() == Ordering::Less {
        coeff += m2;
    }
    let mut result = m1 * &coeff;
    result += r1;
    result
}

/// Fold a list of `(residue, modulus)` pairs into a single
/// `(value, product_of_moduli)`.
///
/// All moduli must be pairwise coprime — the standard use case of
/// distinct primes from [`crate::prime_set::primes_descending`]
/// satisfies this automatically.
///
/// # Panics
///
/// Panics on an empty input slice or on coprimality failure inside
/// [`crt_pair`].
pub fn crt_combine(pairs: &[(Integer, Integer)]) -> (Integer, Integer) {
    assert!(!pairs.is_empty(), "crt_combine: empty input");
    let mut value = pairs[0].0.clone();
    let mut modulus = pairs[0].1.clone();
    for (r, m) in &pairs[1..] {
        value = crt_pair(&value, &modulus, r, m);
        modulus *= m;
    }
    (value, modulus)
}

/// Map a residue from `[0, modulus)` into the symmetric range
/// `(-modulus/2, modulus/2]` — the unique representative of
/// smallest absolute value.
///
/// `value` is taken mod `modulus` first (so callers may pass values
/// outside `[0, modulus)`).
pub fn symmetric_lift(value: &Integer, modulus: &Integer) -> Integer {
    let mut v = value % modulus;
    if v.cmp0() == Ordering::Less {
        v += modulus;
    }
    let half = modulus / 2u32; // floor(modulus / 2)
    if v > half {
        v - modulus
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: i64) -> Integer {
        Integer::from(v)
    }

    #[test]
    fn crt_pair_basic() {
        // x ≡ 1 (mod 3), x ≡ 2 (mod 5) → x = 7 in [0, 15).
        let r = crt_pair(&z(1), &z(3), &z(2), &z(5));
        assert_eq!(r, z(7));
    }

    #[test]
    fn crt_pair_at_zero() {
        // x ≡ 0 (mod m1), x ≡ 0 (mod m2) → x = 0.
        let r = crt_pair(&z(0), &z(7), &z(0), &z(11));
        assert_eq!(r, z(0));
    }

    #[test]
    fn crt_pair_round_trip() {
        // Pick x = 42, moduli 5 and 13 (coprime).
        let x = z(42);
        let m1 = z(5);
        let m2 = z(13);
        let r1 = &x % &m1;
        let r2 = &x % &m2;
        let r = crt_pair(&r1, &m1, &r2, &m2);
        assert_eq!(r, z(42));
    }

    #[test]
    fn crt_pair_returns_canonical_residue() {
        // r1=2, m1=3, r2=3, m2=5. We expect x in [0, 15).
        let r = crt_pair(&z(2), &z(3), &z(3), &z(5));
        assert!(r >= 0);
        assert!(r < 15);
        assert_eq!(&r % z(3), z(2));
        assert_eq!(&r % z(5), z(3));
    }

    #[test]
    #[should_panic(expected = "moduli not coprime")]
    fn crt_pair_rejects_non_coprime() {
        // gcd(4, 6) = 2.
        let _ = crt_pair(&z(1), &z(4), &z(2), &z(6));
    }

    #[test]
    fn crt_combine_three_primes() {
        // Recover x = 100 from residues mod 7, 11, 13.
        let x = z(100);
        let pairs = vec![(&x % 7, z(7)), (&x % 11, z(11)), (&x % 13, z(13))];
        let (value, modulus) = crt_combine(&pairs);
        assert_eq!(modulus, z(7 * 11 * 13));
        assert_eq!(value, z(100));
    }

    #[test]
    fn crt_combine_single_pair_passes_through() {
        let (value, modulus) = crt_combine(&[(z(3), z(7))]);
        assert_eq!(value, z(3));
        assert_eq!(modulus, z(7));
    }

    #[test]
    #[should_panic(expected = "crt_combine: empty input")]
    fn crt_combine_rejects_empty() {
        let _ = crt_combine(&[]);
    }

    #[test]
    fn symmetric_lift_around_zero() {
        // mod 7: [0,1,2,3] map to themselves; [4,5,6] → [-3,-2,-1].
        for (raw, expected) in [(0, 0), (1, 1), (2, 2), (3, 3), (4, -3), (5, -2), (6, -1)] {
            assert_eq!(symmetric_lift(&z(raw), &z(7)), z(expected));
        }
    }

    #[test]
    fn symmetric_lift_on_even_modulus_keeps_half() {
        // mod 8: half = 4. Range (-4, 4] — value 4 stays as 4.
        assert_eq!(symmetric_lift(&z(4), &z(8)), z(4));
        assert_eq!(symmetric_lift(&z(5), &z(8)), z(-3));
    }

    #[test]
    fn symmetric_lift_reduces_input_first() {
        // value 10 mod 7 = 3 → lifts to 3.
        assert_eq!(symmetric_lift(&z(10), &z(7)), z(3));
        // value -1 mod 7 = 6 → lifts to -1.
        assert_eq!(symmetric_lift(&z(-1), &z(7)), z(-1));
    }
}
