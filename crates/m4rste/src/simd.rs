// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! SIMD acceleration for the elimination inner loop on **GF(256) with
//! the AES polynomial** (`minpoly = 0x11B`).
//!
//! Large GF(256) `rank` / `rref` cases can spend most of their time in
//! per-element field multiplication when Newton-John row tables do not
//! amortise. Both supported desktop architectures provide useful GF(2^8)
//! vector primitives:
//!
//! - **x86_64**: GFNI's `_mm256_gf2p8mul_epi8` multiplies 32 bytes in
//!   GF(2^8) with the AES poly per call. Shipping on Zen 4 (Ryzen
//!   7000+) and Intel Tremont / Ice Lake +.
//! - **aarch64**: NEON `vmull_p8` does eight polynomial byte
//!   multiplications at a time; mod-`0x11B` reduction is a fixed
//!   two-step `pmull`/XOR sequence. NEON is always available on
//!   ARMv8 (Apple Silicon).
//!
//! Both paths apply *only* to GF(256) with `minpoly = 0x11B`. Other
//! fields and non-AES GF(256) minpolys keep the scalar log-table
//! path in `Mzed::echelonize`.

/// Compute `dst[i] ^= mul_aes(factor, src[i])` for every `i`, where the
/// multiplication is GF(256) with the AES polynomial `0x11B`.
///
/// `factor` is a GF(256) element (`u8`). `src` and `dst` are slices of
/// equal length holding `u16` elements whose meaningful payload is in
/// the low byte (the high byte is required to be 0, which is how
/// `Mzed::to_u16_buffer` always lays GF(256) data out).
///
/// Returns `true` if a SIMD path executed, `false` if we fell through
/// to the scalar fallback. The bit-equality is identical either way —
/// callers don't need to branch on the return value for correctness;
/// it's exposed for benchmarks / tests.
#[inline]
pub fn gf256_aes_xor_scaled_row(factor: u8, src: &[u16], dst: &mut [u16]) -> bool {
    debug_assert_eq!(src.len(), dst.len());

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("gfni") {
            // SAFETY: feature-gated; we just confirmed both extensions.
            unsafe { x86::gf256_aes_xor_row_gfni(factor, src, dst) };
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is universal on aarch64 — no runtime detect needed.
        // SAFETY: target arch is aarch64; NEON is guaranteed by ARMv8.
        unsafe { aarch64_neon::gf256_aes_xor_row_neon(factor, src, dst) };
        return true;
    }

    #[allow(unreachable_code)]
    {
        scalar_gf256_aes_xor_row(factor, src, dst);
        false
    }
}

/// Carry-less GF(256) AES-poly mult. Reference for the SIMD paths and
/// scalar fallback when no SIMD path applies.
fn gf256_aes_mul_scalar(a: u8, b: u8) -> u8 {
    let mut acc: u16 = 0;
    let mut aa = a as u16;
    let mut bb = b as u16;
    for _ in 0..8 {
        if bb & 1 != 0 {
            acc ^= aa;
        }
        bb >>= 1;
        aa <<= 1;
        if aa & 0x100 != 0 {
            aa ^= 0x11B;
        }
    }
    acc as u8
}

fn scalar_gf256_aes_xor_row(factor: u8, src: &[u16], dst: &mut [u16]) {
    for i in 0..src.len() {
        let p = src[i] as u8;
        if p != 0 {
            dst[i] ^= gf256_aes_mul_scalar(factor, p) as u16;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// AVX2 + GFNI implementation: 16 GF(256) products per `vgf2p8mulb`,
    /// XORed into 16 u16 destination cells per loop iteration.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that `avx2` and `gfni` are available on
    /// the running CPU.
    #[target_feature(enable = "avx2,gfni")]
    pub(super) unsafe fn gf256_aes_xor_row_gfni(factor: u8, src: &[u16], dst: &mut [u16]) {
        let n = src.len();
        // Broadcast `factor` to every byte (including the high byte of
        // each u16 lane, which is fine: high bytes of `src` are 0, so
        // the GFNI product of factor × 0 is 0 and the high byte of the
        // result stays 0).
        let factor_vec = _mm256_set1_epi8(factor as i8);
        let mut i = 0;
        while i + 16 <= n {
            let src_ptr = src.as_ptr().add(i) as *const __m256i;
            let dst_ptr = dst.as_mut_ptr().add(i) as *mut __m256i;
            let src_vec = _mm256_loadu_si256(src_ptr);
            let prod = _mm256_gf2p8mul_epi8(factor_vec, src_vec);
            let dst_vec = _mm256_loadu_si256(dst_ptr);
            let result = _mm256_xor_si256(dst_vec, prod);
            _mm256_storeu_si256(dst_ptr, result);
            i += 16;
        }
        // Scalar tail for elements past the last 16-element chunk.
        super::scalar_gf256_aes_xor_row(factor, &src[i..], &mut dst[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_neon {
    use std::arch::aarch64::*;

    /// NEON implementation: 8 GF(256) products per iteration via
    /// `vmull_p8` + a two-step reduction modulo the AES polynomial.
    ///
    /// Reduction: a `poly16` product `p = p_hi·x^8 + p_lo` is folded
    /// once via `x^8 ≡ 0x1B (mod 0x11B)`:
    ///
    /// ```text
    ///   t = p_hi · 0x1B               // up to 11 bits
    ///   t = t_hi · x^8 + t_lo         // t_hi has at most 3 bits
    ///   q = t_hi · 0x1B               // at most 7 bits — done
    ///   result = p_lo XOR t_lo XOR q  // 8 bits
    /// ```
    ///
    /// # Safety
    ///
    /// Caller must guarantee NEON availability (always true on
    /// aarch64).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn gf256_aes_xor_row_neon(factor: u8, src: &[u16], dst: &mut [u16]) {
        let n = src.len();
        let factor_v = vdup_n_p8(factor);
        let const_1b = vdup_n_p8(0x1B);
        let mut i = 0;
        while i + 8 <= n {
            let src_ptr = src.as_ptr().add(i);
            let dst_ptr = dst.as_mut_ptr().add(i);

            // 8 GF(256) elements, extract low byte of each u16.
            let src_u16 = vld1q_u16(src_ptr);
            let src_lo = vmovn_u16(src_u16);

            // poly mult, 8 lanes × u8 × u8 → u16.
            let prod_p16 = vmull_p8(factor_v, vreinterpret_p8_u8(src_lo));
            let prod_u16 = vreinterpretq_u16_p16(prod_p16);

            // First reduction: t = p_hi * 0x1B.
            let p_hi = vshrn_n_u16::<8>(prod_u16);
            let p_lo = vmovn_u16(prod_u16);
            let t_p16 = vmull_p8(vreinterpret_p8_u8(p_hi), const_1b);
            let t_u16 = vreinterpretq_u16_p16(t_p16);

            // Second reduction: q = t_hi * 0x1B (no further reduction
            // since q has ≤ 7 bits and fits in the low byte).
            let t_hi = vshrn_n_u16::<8>(t_u16);
            let t_lo = vmovn_u16(t_u16);
            let q_p16 = vmull_p8(vreinterpret_p8_u8(t_hi), const_1b);
            let q_lo = vmovn_u16(vreinterpretq_u16_p16(q_p16));

            // Final reduced byte and widen to u16 to XOR into dst.
            let reduced = veor_u8(veor_u8(p_lo, t_lo), q_lo);
            let reduced_u16 = vmovl_u8(reduced);
            let dst_u16 = vld1q_u16(dst_ptr);
            let combined = veorq_u16(dst_u16, reduced_u16);
            vst1q_u16(dst_ptr, combined);

            i += 8;
        }
        super::scalar_gf256_aes_xor_row(factor, &src[i..], &mut dst[i..]);
    }
}

// ── GF(16) Newton-John table-build SIMD ───────────────────────────

/// Compute `dst[i] = factor · src[i]` over GF(16) with the given
/// minimal polynomial (a degree-4 irreducible `u64` bit pattern, e.g.
/// `0b1_0011` for `x⁴ + x + 1`).
///
/// `factor` is a GF(16) element in `[0, 16)`. `src` and `dst` are
/// equal-length slices of `u16` values whose meaningful payload is in
/// the low **nibble** of each element (high bits are required to be
/// zero, matching `Mzed`'s layout for `e == 4`).
///
/// Used to accelerate `Mzed::echelonize`'s Newton-John table build for
/// the `e == 4` (GF(16)) case — there are 14 such scaled-row
/// computations per pivot, and this turns the inner loop from a
/// scalar log-table-multiply per element into one byte-shuffle
/// instruction per 16 elements.
///
/// The mechanism is identical on both architectures: precompute the
/// 16-entry multiplication table `t[c] = factor · c` (fits in one
/// 16-byte vector register), then table-lookup it via `vpshufb` on
/// x86_64 or `vqtbl1q_u8` on aarch64. The high byte of every `u16` is
/// 0, which indexes `t[0] = 0` and stays 0 in the result — matching
/// the `u16` storage convention.
///
/// Returns `true` if a SIMD path ran, `false` for scalar fallback.
#[inline]
pub fn gf16_apply_factor_row(factor: u8, minpoly: u64, src: &[u16], dst: &mut [u16]) -> bool {
    debug_assert_eq!(src.len(), dst.len());

    // Precompute the 16-byte multiplication table: t[c] = factor · c
    // in GF(16) under `minpoly`.
    let mut table = [0u8; 16];
    for c in 0..16usize {
        table[c] = gf16_mul_scalar(factor, c as u8, minpoly);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed; no out-of-bounds, slices verified above.
            unsafe { x86_gf16::apply_pshufb(&table, src, dst) };
            return true;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is universal on aarch64.
        // SAFETY: NEON guaranteed by ARMv8.
        unsafe { aarch64_gf16::apply_tbl(&table, src, dst) };
        return true;
    }

    #[allow(unreachable_code)]
    {
        gf16_apply_scalar(&table, src, dst);
        false
    }
}

/// Carry-less GF(16) multiplication with arbitrary irreducible
/// `minpoly`. Reference / scalar fallback / table-build primitive.
pub fn gf16_mul_scalar(a: u8, b: u8, minpoly: u64) -> u8 {
    let mut acc: u16 = 0;
    let mut aa = a as u16;
    for k in 0..4 {
        if (b >> k) & 1 != 0 {
            acc ^= aa;
        }
        aa <<= 1;
        if aa & 0x10 != 0 {
            aa ^= minpoly as u16;
        }
    }
    (acc & 0x0F) as u8
}

fn gf16_apply_scalar(table: &[u8; 16], src: &[u16], dst: &mut [u16]) {
    for i in 0..src.len() {
        dst[i] = table[(src[i] & 0x0F) as usize] as u16;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_gf16 {
    use std::arch::x86_64::*;

    /// 16 GF(16) products per AVX2 byte-shuffle. The low byte of each
    /// `u16` indexes the table; the high byte is 0 and selects
    /// `table[0] = 0`, so the output's high byte stays 0.
    ///
    /// `_mm256_shuffle_epi8` does the shuffle within each 128-bit
    /// lane (no cross-lane), so we broadcast the same 16-byte table
    /// to both lanes via `_mm256_broadcastsi128_si256`.
    ///
    /// # Safety
    ///
    /// Caller must guarantee AVX2 availability.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn apply_pshufb(table: &[u8; 16], src: &[u16], dst: &mut [u16]) {
        let n = src.len();
        let table_lo = _mm_loadu_si128(table.as_ptr() as *const __m128i);
        let table_v = _mm256_broadcastsi128_si256(table_lo);
        let mut i = 0;
        while i + 16 <= n {
            let src_ptr = src.as_ptr().add(i) as *const __m256i;
            let dst_ptr = dst.as_mut_ptr().add(i) as *mut __m256i;
            let src_vec = _mm256_loadu_si256(src_ptr);
            let result = _mm256_shuffle_epi8(table_v, src_vec);
            _mm256_storeu_si256(dst_ptr, result);
            i += 16;
        }
        super::gf16_apply_scalar(table, &src[i..], &mut dst[i..]);
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_gf16 {
    use std::arch::aarch64::*;

    /// 16 GF(16) products per NEON `vqtbl1q_u8`. Indexes ≥ 16 produce
    /// 0 (saturating), but our nibble indexes are bounded to `[0,16)`
    /// so this never trips.
    ///
    /// # Safety
    ///
    /// NEON guaranteed by aarch64.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn apply_tbl(table: &[u8; 16], src: &[u16], dst: &mut [u16]) {
        let n = src.len();
        let table_v = vld1q_u8(table.as_ptr());
        let mut i = 0;
        while i + 16 <= n {
            let src_ptr = src.as_ptr().add(i);
            let dst_ptr = dst.as_mut_ptr().add(i);
            // Load 16 u16s (32 bytes); narrow to 16 u8 nibble-indexes.
            let a = vld1q_u16(src_ptr);
            let b = vld1q_u16(src_ptr.add(8));
            let idx = vcombine_u8(vmovn_u16(a), vmovn_u16(b));
            let r = vqtbl1q_u8(table_v, idx);
            // Widen back to u16 (high byte = 0).
            vst1q_u16(dst_ptr, vmovl_u8(vget_low_u8(r)));
            vst1q_u16(dst_ptr.add(8), vmovl_u8(vget_high_u8(r)));
            i += 16;
        }
        super::gf16_apply_scalar(table, &src[i..], &mut dst[i..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_xor_row(factor: u8, src: &[u16], dst: &mut [u16]) {
        for i in 0..src.len() {
            dst[i] ^= gf256_aes_mul_scalar(factor, src[i] as u8) as u16;
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn next_u8(&mut self) -> u8 {
            self.next() as u8
        }
    }

    fn random_row(n: usize, seed: u64) -> Vec<u16> {
        let mut rng = Rng(seed | 1);
        (0..n).map(|_| rng.next_u8() as u16).collect()
    }

    #[test]
    fn simd_matches_scalar_across_lengths_and_factors() {
        // Hit every length class — sub-vector, exact vector multiples,
        // and tail-handling cases — and exercise the full byte space
        // of factors.
        let lengths = [0usize, 1, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 256, 257];
        for &n in &lengths {
            for &factor in &[0u8, 1, 2, 3, 0x1B, 0x7F, 0x80, 0xFE, 0xFF] {
                let src = random_row(n, 0xA11C ^ (n as u64) ^ ((factor as u64) << 32));
                let init = random_row(n, 0xB0B ^ (n as u64) ^ ((factor as u64) << 40));
                let mut dst_simd = init.clone();
                let mut dst_naive = init.clone();
                gf256_aes_xor_scaled_row(factor, &src, &mut dst_simd);
                naive_xor_row(factor, &src, &mut dst_naive);
                assert_eq!(
                    dst_simd, dst_naive,
                    "SIMD ≠ naive at n={n}, factor={factor:#x}"
                );
            }
        }
    }

    #[test]
    fn scalar_mul_matches_carryless_definition() {
        // Spot-check the scalar reference against a manual GF(256)
        // multiplication for a few known pairs.
        // 0x53 · 0xCA = 0x01 in GF(256) AES poly (a known inverse pair).
        assert_eq!(gf256_aes_mul_scalar(0x53, 0xCA), 0x01);
        // Multiplicative identity.
        for a in 0u8..=255 {
            assert_eq!(gf256_aes_mul_scalar(a, 1), a);
            assert_eq!(gf256_aes_mul_scalar(1, a), a);
        }
        // Zero annihilates.
        for a in 0u8..=255 {
            assert_eq!(gf256_aes_mul_scalar(a, 0), 0);
            assert_eq!(gf256_aes_mul_scalar(0, a), 0);
        }
    }

    fn random_nibble_row(n: usize, seed: u64) -> Vec<u16> {
        let mut rng = Rng(seed | 1);
        (0..n).map(|_| (rng.next_u8() as u16) & 0x0F).collect()
    }

    #[test]
    fn gf16_simd_matches_scalar_exhaustively() {
        // Every minimal polynomial we care about for GF(16): three
        // irreducible degree-4 polynomials over F_2.
        let minpolys = [
            0b1_0011u64, // x^4 + x + 1 (primitive)
            0b1_1001,    // x^4 + x^3 + 1 (primitive)
            0b1_1111,    // x^4 + x^3 + x^2 + x + 1 (irreducible, not primitive)
        ];
        // Hit every length class — sub-vector, vector multiples, tails.
        let lengths = [0usize, 1, 7, 8, 15, 16, 17, 31, 32, 64, 256, 257];
        for &minpoly in &minpolys {
            for &n in &lengths {
                for factor in 0u8..16 {
                    let src = random_nibble_row(n, 0xC0FFEE ^ minpoly ^ (n as u64));
                    let mut dst_simd = vec![0u16; n];
                    let mut dst_naive = vec![0u16; n];
                    gf16_apply_factor_row(factor, minpoly, &src, &mut dst_simd);
                    for i in 0..n {
                        dst_naive[i] = gf16_mul_scalar(factor, src[i] as u8, minpoly) as u16;
                    }
                    assert_eq!(
                        dst_simd, dst_naive,
                        "GF(16) SIMD ≠ naive at minpoly={minpoly:#x}, n={n}, factor={factor:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn gf16_mul_scalar_basics() {
        let mp = 0b1_0011u64; // x^4 + x + 1
                              // x · x = x² (no reduction yet).
        assert_eq!(gf16_mul_scalar(2, 2, mp), 4);
        // x² · x² = x⁴ ≡ x + 1 = 3 (under x⁴ + x + 1).
        assert_eq!(gf16_mul_scalar(4, 4, mp), 3);
        // Multiplicative identity + zero annihilation, every element.
        for a in 0u8..16 {
            assert_eq!(gf16_mul_scalar(a, 1, mp), a);
            assert_eq!(gf16_mul_scalar(1, a, mp), a);
            assert_eq!(gf16_mul_scalar(a, 0, mp), 0);
            assert_eq!(gf16_mul_scalar(0, a, mp), 0);
        }
    }
}
