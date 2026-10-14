// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![deny(missing_docs)]

//! PWVLA — specialized packed-kernel crate for dense matmul over
//! Z/2^n (native wrap) and Z/p^n (packed Barrett), with SIMD backends.
//!
//! This crate is intentionally narrow: it does **not** provide a
//! general linear-algebra surface (PLUQ, rank, inverse, charpoly,
//! polynomials, Witt vectors, F_{p^k}). Noether's generic matrix facade
//! consumes this crate through a private adapter when a context is small
//! enough to benefit from bit-packing.
//! Rayon is optional and disabled by default; the same entry points execute
//! serially unless the `parallel` feature is selected.
//!
//! # What pwvla ships
//!
//! * **`Mat<T: Z2n>`** for native-type Z/2^n (T ∈ {u8, u16, u32, u64,
//!   u128}): wrapping arithmetic is the ring operation. The dispatched
//!   kernels use NEON for u8/u16/u32/u64 on aarch64 and runtime-dispatched
//!   AVX2 for u16 on x86-64, with scalar fallbacks.
//!
//! * **`MatZpn`** for packed Z/p^n: bit-packed elements with Barrett
//!   reducer, multiple elements per u64 word. Intended for small
//!   p with `p^n ≲ 2^32`. SIMD backends: AVX2 u16/u32 and NEON u32
//!   kernels; scalar Barrett fallback.
//!
//! # Out of scope
//!
//! Generic-ring `DenseMat<R: Ring>` (with PLUQ, charpoly, etc.),
//! polynomial surfaces, Hensel lifting, F_{p^k}, Witt vectors, FFI /
//! PyO3 / Python dispatch hooks, and external-library shims. Those belong to
//! their owning facade, transport, or binding crates.

#[cfg(not(feature = "parallel"))]
extern crate self as rayon;

#[cfg(not(feature = "parallel"))]
pub(crate) mod prelude {
    //! Serial adapters matching the small Rayon iterator surface used here.

    /// Serial replacement for `rayon::IntoParallelIterator`.
    pub(crate) trait IntoParallelIterator: IntoIterator + Sized {
        /// The ordinary serial iterator.
        type Iter: Iterator<Item = Self::Item>;

        /// Consume this value as an ordinary serial iterator.
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<T: IntoIterator> IntoParallelIterator for T {
        type Iter = T::IntoIter;

        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    /// Serial replacements for the mutable-slice methods used by the kernels.
    pub(crate) trait ParallelSliceMut<T> {
        /// Split into mutable chunks and visit them serially.
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T>;

        /// Visit mutable elements serially.
        fn par_iter_mut(&mut self) -> core::slice::IterMut<'_, T>;
    }

    impl<T> ParallelSliceMut<T> for [T] {
        fn par_chunks_mut(&mut self, chunk_size: usize) -> core::slice::ChunksMut<'_, T> {
            self.chunks_mut(chunk_size)
        }

        fn par_iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
            self.iter_mut()
        }
    }
}

/// Execute two independent closures serially when Rayon is not selected.
#[cfg(all(not(feature = "parallel"), target_arch = "aarch64"))]
pub(crate) fn join<A, B, RA, RB>(left: A, right: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB,
{
    (left(), right())
}

pub mod mat;
pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "aarch64")]
pub mod neon_wide;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod strassen;

pub mod zpn;

pub use mat::{Mat, Z2n};

/// Multiply dense matrices over `Z/2^8` with the best compiled kernel.
pub fn matmul_u8(left: &Mat<u8>, right: &Mat<u8>) -> Mat<u8> {
    #[cfg(target_arch = "aarch64")]
    {
        return neon_wide::matmul_u8(left, right);
    }
    #[allow(unreachable_code)]
    scalar::matmul(left, right)
}

/// Multiply dense matrices over `Z/2^16` with the best compiled kernel.
pub fn matmul_u16(left: &Mat<u16>, right: &Mat<u16>) -> Mat<u16> {
    #[cfg(target_arch = "x86_64")]
    {
        return avx2::matmul_u16(left, right);
    }
    #[cfg(target_arch = "aarch64")]
    {
        return neon::matmul_u16(left, right);
    }
    #[allow(unreachable_code)]
    scalar::matmul(left, right)
}

/// Multiply dense matrices over `Z/2^32` with the best compiled kernel.
pub fn matmul_u32(left: &Mat<u32>, right: &Mat<u32>) -> Mat<u32> {
    #[cfg(target_arch = "aarch64")]
    {
        return neon_wide::matmul_u32(left, right);
    }
    #[allow(unreachable_code)]
    scalar::matmul(left, right)
}

/// Multiply dense matrices over `Z/2^64` with the best compiled kernel.
pub fn matmul_u64(left: &Mat<u64>, right: &Mat<u64>) -> Mat<u64> {
    #[cfg(target_arch = "aarch64")]
    {
        return neon_wide::matmul_u64(left, right);
    }
    #[allow(unreachable_code)]
    scalar::matmul(left, right)
}

/// Multiply dense matrices over `Z/2^128` with the portable kernel.
pub fn matmul_u128(left: &Mat<u128>, right: &Mat<u128>) -> Mat<u128> {
    scalar::matmul(left, right)
}
