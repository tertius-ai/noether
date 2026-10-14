// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Fused multi-row XOR kernels — `c ^= t[0] ^ t[1] ^ ... ^ t[N-1]`.
//!
//! Used by the NTABLES=8 variant of M4RM (`matmul_m4rm_ntables`) to
//! collapse N separate "XOR table row into C" passes into one fused
//! pass. The N=8 case is the headline kernel: it cuts C memory
//! traffic 8× by reading+writing each `c` word once per super-strip
//! instead of once per strip.
//!
//! The const-generic form lets the scalar floor unroll the inner N-XOR chain.
//! The hot N=8 path also reaches explicit runtime-dispatched AVX2 or NEON code,
//! so portable builds do not depend on LLVM choosing a vector width.

/// `c[i] ^= sum_{n in 0..N} tables[n][i]` for `i in 0..c.len()`.
///
/// All `tables[n]` slices must have length `≥ c.len()`. Vector kernels use
/// unaligned loads and stores, then share the scalar tail implementation.
#[inline(always)]
#[doc(hidden)]
pub fn combine_n_into<const N: usize>(c: &mut [u64], tables: [&[u64]; N]) {
    let _ = crate::simd::combine_n_into(c, tables);
}
