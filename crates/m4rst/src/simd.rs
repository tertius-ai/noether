// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Runtime-dispatched SIMD for fused bit-packed row combinations.
//!
//! Every architecture path operates on whole `u64` words and finishes with
//! the same scalar tail. Callers therefore preserve padding bits exactly when
//! the inputs do. The x86-64 entry checks AVX2 before entering target-feature
//! code; aarch64's baseline includes NEON.

/// Combine `N` table rows into one destination row with the best available
/// word-XOR kernel. Returns whether a vector path processed at least one block.
#[inline]
pub(crate) fn combine_n_into<const N: usize>(c: &mut [u64], tables: [&[u64]; N]) -> bool {
    debug_assert!(tables.iter().all(|table| table.len() >= c.len()));

    #[cfg(target_arch = "x86_64")]
    if c.len() >= 4 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected for this process, and the function performs
        // only unaligned in-bounds loads and stores over the checked slices.
        unsafe { combine_n_avx2(c, tables) };
        return true;
    }

    #[cfg(target_arch = "aarch64")]
    if c.len() >= 2 {
        // SAFETY: NEON is part of the aarch64 architecture baseline. Loads and
        // stores are unaligned and bounded by the loop condition.
        unsafe { combine_n_neon(c, tables) };
        return true;
    }

    combine_n_scalar(c, tables);
    false
}

#[inline(always)]
pub(crate) fn combine_n_scalar<const N: usize>(c: &mut [u64], tables: [&[u64]; N]) {
    debug_assert!(tables.iter().all(|table| table.len() >= c.len()));
    for i in 0..c.len() {
        let mut acc = c[i];
        let mut table = 0;
        while table < N {
            acc ^= tables[table][i];
            table += 1;
        }
        c[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Combine rows four words at a time with AVX2.
///
/// # Safety
///
/// The running CPU must support AVX2. Every table must contain at least
/// `c.len()` words; the runtime dispatcher establishes both conditions.
pub(crate) unsafe fn combine_n_avx2<const N: usize>(c: &mut [u64], tables: [&[u64]; N]) {
    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_xor_si256};

    let mut i = 0;
    while i + 4 <= c.len() {
        let mut acc = _mm256_loadu_si256(c.as_ptr().add(i).cast::<__m256i>());
        let mut table = 0;
        while table < N {
            let value = _mm256_loadu_si256(tables[table].as_ptr().add(i).cast::<__m256i>());
            acc = _mm256_xor_si256(acc, value);
            table += 1;
        }
        _mm256_storeu_si256(c.as_mut_ptr().add(i).cast::<__m256i>(), acc);
        i += 4;
    }
    combine_n_scalar(&mut c[i..], tables.map(|table| &table[i..]));
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
/// Combine rows two words at a time with aarch64 NEON.
///
/// # Safety
///
/// The running CPU must support NEON (guaranteed by aarch64). Every table must
/// contain at least `c.len()` words; the runtime dispatcher checks the slices.
pub(crate) unsafe fn combine_n_neon<const N: usize>(c: &mut [u64], tables: [&[u64]; N]) {
    use core::arch::aarch64::{veorq_u64, vld1q_u64, vst1q_u64};

    let mut i = 0;
    while i + 2 <= c.len() {
        let mut acc = vld1q_u64(c.as_ptr().add(i));
        let mut table = 0;
        while table < N {
            acc = veorq_u64(acc, vld1q_u64(tables[table].as_ptr().add(i)));
            table += 1;
        }
        vst1q_u64(c.as_mut_ptr().add(i), acc);
        i += 2;
    }
    combine_n_scalar(&mut c[i..], tables.map(|table| &table[i..]));
}

#[cfg(test)]
mod tests {
    use super::{combine_n_into, combine_n_scalar};

    fn inputs<const N: usize>(len: usize) -> (Vec<u64>, [Vec<u64>; N]) {
        let destination = (0..len)
            .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let tables = std::array::from_fn(|table| {
            (0..len)
                .map(|i| {
                    (i as u64)
                        .wrapping_add((table + 1) as u64)
                        .rotate_left(((i + table) % 64) as u32)
                })
                .collect()
        });
        (destination, tables)
    }

    #[test]
    fn dispatched_combine_matches_scalar_for_blocks_and_tails() {
        for len in [0, 1, 2, 3, 4, 5, 6, 17, 64, 259] {
            let (initial, tables) = inputs::<8>(len);
            let references: [&[u64]; 8] = std::array::from_fn(|i| tables[i].as_slice());
            let mut expected = initial.clone();
            let mut actual = initial;
            combine_n_scalar(&mut expected, references);
            combine_n_into(&mut actual, references);
            assert_eq!(actual, expected, "length {len}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn forced_avx2_combine_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [4, 5, 6, 16, 67, 256] {
            let (initial, tables) = inputs::<8>(len);
            let references: [&[u64]; 8] = std::array::from_fn(|i| tables[i].as_slice());
            let mut expected = initial.clone();
            let mut actual = initial;
            combine_n_scalar(&mut expected, references);
            // SAFETY: this test returns above unless AVX2 is available.
            unsafe { super::combine_n_avx2(&mut actual, references) };
            assert_eq!(actual, expected, "length {len}");
        }
    }
}
