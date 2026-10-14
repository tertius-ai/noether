// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Stable deterministic randomness for mathematical algorithms.
//!
//! The public seed mapping and bounded sampling live here so a dependency
//! upgrade cannot silently change Noether's mathematical output.

use rand::rngs::ChaCha20Rng;
use rand::{Rng, SeedableRng};

/// Expand the public 64-bit seed to ChaCha's fixed 32-byte seed.
pub(crate) fn seeded_rng(seed: u64) -> ChaCha20Rng {
    let mut state = seed;
    let mut expanded = [0_u8; 32];
    for chunk in expanded.chunks_exact_mut(8) {
        chunk.copy_from_slice(&splitmix64(&mut state).to_le_bytes());
    }
    ChaCha20Rng::from_seed(expanded)
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Sample uniformly from `0..upper` using raw words and rejection.
pub(crate) fn sample_below<R: Rng + ?Sized>(rng: &mut R, upper: u64) -> u64 {
    assert!(upper > 0, "sample_below requires a positive upper bound");
    let threshold = upper.wrapping_neg() % upper;
    loop {
        let value = rng.next_u64();
        if value >= threshold {
            return value % upper;
        }
    }
}

/// Fisher-Yates shuffle with the stable bounded sampler above.
#[cfg(test)]
pub(crate) fn shuffle<T, R: Rng + ?Sized>(values: &mut [T], rng: &mut R) {
    for end in (1..values.len()).rev() {
        let index = sample_below(rng, (end + 1) as u64) as usize;
        values.swap(end, index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_seed_mapping_is_stable() {
        let mut rng = seeded_rng(42);
        assert_eq!(rng.next_u64(), 693_385_945_204_756_564);
        assert_eq!(rng.next_u64(), 16_436_763_086_163_553_629);
    }

    #[test]
    fn bounded_sample_stays_in_range() {
        let mut rng = seeded_rng(17);
        for upper in [1, 2, 3, 17, 1_000, u32::MAX as u64 + 1] {
            for _ in 0..1_000 {
                assert!(sample_below(&mut rng, upper) < upper);
            }
        }
    }
}
