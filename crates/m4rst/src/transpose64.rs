// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! 64×64 bit-block transpose primitive.
//!
//! Treats a `[u64; 64]` buffer as a 64-row, 64-column bit-matrix
//! laid out row-major with LSB-first within each `u64` (matching
//! M4RI's `mzd_t` convention). After [`transpose_64x64_inplace`]:
//!
//! ```text
//! bit (r, c) of output == bit (c, r) of input
//! ```
//!
//! ## Algorithm
//!
//! Six passes of masked-XOR shuffles, mirroring Warren's recursive
//! 64×64 bit-transpose:
//!
//! * Pass j = 32 swaps the upper-right 32×32 corner with the
//!   lower-left 32×32 corner of the 64×64 block.
//! * Pass j = 16 does the same swap inside each 32×32 quadrant.
//! * Pass j ∈ {8, 4, 2, 1} continue down the recursive halving.
//!
//! At each pass:
//!
//! ```text
//! for r in 0..64 step 2·j, for k in 0..j:
//!     xor       = ((rows[r+k] >> j) ^ rows[r+k+j]) & mask
//!     rows[r+k]  ^=  xor << j
//!     rows[r+k+j] ^= xor
//! ```
//!
//! The mask at each level isolates the `j`-bit-wide groups that
//! need swapping; the bit-pattern is the alternating-`j`-ones-and-
//! zeros constants below.
//!
//! ## References
//!
//! * Warren, *Hacker's Delight* 2e, Ch. 7 "Transposing a Bit Matrix"
//!   — the recursive bit-block transpose this is a direct port of.
//! * Knuth, *TAOCP* Vol. 4A pre-fascicle 1B § 7.1.3.
//! * libm4ri's `_mzd_copy_transpose_64x64` in `m4rst/mzd.c:249` is
//!   the engineering reference (uses rowstride-walked memory; we
//!   simplify because our input is a contiguous `[u64; 64]`).

/// Per-pass column-group masks. Entry `i` masks the `2^(5-i)`-bit
/// groups: 32-bit, 16-bit, 8-bit, 4-bit, 2-bit, 1-bit alternation.
const PASS_MASKS: [u64; 6] = [
    0x0000_0000_FFFF_FFFF,
    0x0000_FFFF_0000_FFFF,
    0x00FF_00FF_00FF_00FF,
    0x0F0F_0F0F_0F0F_0F0F,
    0x3333_3333_3333_3333,
    0x5555_5555_5555_5555,
];

/// Per-pass swap stride, in rows. Always `2 ^ (5 - i)` for pass `i`.
const PASS_J: [usize; 6] = [32, 16, 8, 4, 2, 1];

/// In-place transpose of a 64×64 bit-block held as `[u64; 64]`
/// (row-major, LSB-first within each word — M4RI's `mzd_t`
/// convention). Bit `(r, c)` of the output equals bit `(c, r)` of
/// the input.
pub(crate) fn transpose_64x64_inplace(rows: &mut [u64; 64]) {
    for pass in 0..6 {
        let j = PASS_J[pass];
        let mask = PASS_MASKS[pass];
        let mut r = 0;
        while r < 64 {
            for k in 0..j {
                let xor = ((rows[r + k] >> j) ^ rows[r + k + j]) & mask;
                rows[r + k] ^= xor << j;
                rows[r + k + j] ^= xor;
            }
            r += 2 * j;
        }
    }
}

/// Out-of-place 64×64 transpose. Writes the transpose of `src` into
/// `dst`. Equivalent to `dst.copy_from_slice(src); transpose_64x64_inplace(dst)`.
#[allow(dead_code)] // used by transpose.rs in the next commit
pub(crate) fn transpose_64x64_into(dst: &mut [u64; 64], src: &[u64; 64]) {
    dst.copy_from_slice(src);
    transpose_64x64_inplace(dst);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-by-bit reference transpose — the correctness floor the
    /// fast path must agree with cell-for-cell.
    fn transpose_reference(src: &[u64; 64]) -> [u64; 64] {
        let mut dst = [0u64; 64];
        for (r, &src_row) in src.iter().enumerate() {
            for (c, dst_row) in dst.iter_mut().enumerate() {
                let bit = (src_row >> c) & 1;
                *dst_row |= bit << r;
            }
        }
        dst
    }

    fn splitmix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    fn seeded_block(seed: u64) -> [u64; 64] {
        let mut rows = [0u64; 64];
        for (i, row) in rows.iter_mut().enumerate() {
            *row = splitmix(seed.wrapping_add(i as u64));
        }
        rows
    }

    #[test]
    fn transpose_inplace_matches_bit_reference_random() {
        // Six adversarially-chosen seeds covering a range of bit
        // distributions.
        for &seed in &[0x1u64, 0xDEAD, 0xBEEF, 0xCAFE, 0xFFFF_0000, 0x5555_AAAA] {
            let src = seeded_block(seed);
            let mut got = src;
            transpose_64x64_inplace(&mut got);
            let want = transpose_reference(&src);
            assert_eq!(
                got, want,
                "fast 64x64 transpose disagrees with bit-by-bit reference at seed={seed:#x}"
            );
        }
    }

    #[test]
    fn transpose_inplace_is_involution() {
        for &seed in &[0x42u64, 0xC0DE, 0x1234_5678] {
            let src = seeded_block(seed);
            let mut got = src;
            transpose_64x64_inplace(&mut got);
            transpose_64x64_inplace(&mut got);
            assert_eq!(got, src, "T(T(M)) != M at seed={seed:#x}");
        }
    }

    #[test]
    fn transpose_all_zeros() {
        let mut rows = [0u64; 64];
        transpose_64x64_inplace(&mut rows);
        assert_eq!(rows, [0u64; 64]);
    }

    #[test]
    fn transpose_all_ones() {
        let mut rows = [!0u64; 64];
        transpose_64x64_inplace(&mut rows);
        assert_eq!(rows, [!0u64; 64], "all-ones is its own transpose");
    }

    #[test]
    fn transpose_diagonal_bits() {
        // Diagonal: row r has bit (r, r) only. Self-transpose.
        let mut rows = [0u64; 64];
        for (i, row) in rows.iter_mut().enumerate() {
            *row = 1u64 << i;
        }
        let original = rows;
        transpose_64x64_inplace(&mut rows);
        assert_eq!(rows, original, "diagonal matrix is its own transpose");
    }

    #[test]
    fn transpose_offdiagonal_bits() {
        // Upper-triangle bit-mask: row r has bits set for c > r.
        // Transpose should be the strict-lower-triangle pattern.
        let mut rows = [0u64; 64];
        for (r, row) in rows.iter_mut().enumerate() {
            // Bits c in (r, 64) set — i.e., bits r+1..63 inclusive.
            if r < 63 {
                *row = (!0u64) << (r + 1);
            }
        }
        let want = transpose_reference(&rows);
        transpose_64x64_inplace(&mut rows);
        assert_eq!(rows, want);
    }

    #[test]
    fn transpose_into_matches_inplace() {
        for &seed in &[0xAAAA_BBBBu64, 0xC0FF_EE00] {
            let src = seeded_block(seed);
            let mut dst = [0u64; 64];
            transpose_64x64_into(&mut dst, &src);
            let mut want = src;
            transpose_64x64_inplace(&mut want);
            assert_eq!(dst, want);
        }
    }

    #[test]
    fn transpose_into_does_not_mutate_src() {
        let src = seeded_block(0x1A2Bu64);
        let original = src;
        let mut dst = [0u64; 64];
        transpose_64x64_into(&mut dst, &src);
        assert_eq!(src, original, "transpose_64x64_into mutated its src arg");
    }
}
