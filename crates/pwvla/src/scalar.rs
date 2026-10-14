// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Scalar (portable) reference implementations. Correctness baseline.
//!
//! Also hosts the ring-agnostic **streaming matmul** driver
//! (`streaming_matmul_with`) — a `T: Z2n`-generic tiling wrapper that
//! delegates per-tile matmul to an injected callable. The GPU wrapper
//! in `src/gpu.rs` composes this with its on-device kernel to unlock
//! the "matrix fits in RAM but not VRAM" regime; the helper stays here
//! (not in `gpu.rs`) so the tiling correctness is unit-testable on any
//! host without the `gpu` feature or a real adapter.

use crate::mat::{Mat, Z2n};

/// C = A * B with wrapping arithmetic in Z/2^BITS. Three-loop reference.
pub fn matmul<T: Z2n>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    assert_eq!(a.cols, b.rows, "matmul dimension mismatch");
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut c = Mat::<T>::zeros(m, n);
    for i in 0..m {
        for p in 0..k {
            let aip = a.data[i * k + p];
            let row_b = &b.data[p * n..(p + 1) * n];
            let row_c = &mut c.data[i * n..(i + 1) * n];
            for j in 0..n {
                row_c[j] = row_c[j].wrapping_add(aip.wrapping_mul(row_b[j]));
            }
        }
    }
    c
}

/// Element-wise wrapping add `C = A + B` over `Z/2^BITS`. Scalar; the
/// rayon-parallel version lives on `Mat<T>` directly.
pub fn add<T: Z2n>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols));
    let mut c = Mat::<T>::zeros(a.rows, a.cols);
    for i in 0..a.data.len() {
        c.data[i] = a.data[i].wrapping_add(b.data[i]);
    }
    c
}

/// Compute `C = A · B` over Z/2^BITS by tiling A by row-blocks, B by
/// column-blocks, and K by inner chunks — each of size at most
/// `block_size`. Per-chunk matmul is delegated to `submatmul`, so the
/// caller controls where those products happen (GPU, CPU, remote
/// accelerator, …). Partial products `A_chunk · B_chunk` are summed
/// into the running output via wrapping add, which is correct because
/// matmul over Z/2^BITS is distributive.
///
/// Working set (number of ring elements live at any one time in the
/// submatmul callable): bounded by `3 · block_size²`. For a GPU
/// submatmul this means a 50k×50k u32 matmul can run on a GPU with a
/// few tens of MB of VRAM free: pick `block_size = 2048` and the
/// per-call footprint is ≤ 48 MB rather than the 30 GB the full-size
/// matmul would need.
///
/// Edge tiles (when dimensions aren't multiples of `block_size`) are
/// passed to `submatmul` as exact-sized sub-matrices — no padding.
///
/// `submatmul` must satisfy:
///     submatmul(a_chunk, b_chunk).data[i * bn + j]
///         = Σ_{k} a_chunk[i, k] · b_chunk[k, j]  (wrapping in Z/2^BITS)
/// i.e., it's a pure matmul of its inputs. The helper has no way to
/// verify that; callers are trusted.
pub fn streaming_matmul_with<T, F>(
    a: &Mat<T>,
    b: &Mat<T>,
    block_size: usize,
    submatmul: F,
) -> Mat<T>
where
    T: Z2n,
    F: Fn(&Mat<T>, &Mat<T>) -> Mat<T>,
{
    assert!(block_size > 0, "block_size must be positive");
    assert_eq!(a.cols, b.rows, "streaming_matmul_with: dim mismatch");
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut c = Mat::<T>::zeros(m, n);

    let mut i = 0;
    while i < m {
        let bm = block_size.min(m - i);
        let mut j = 0;
        while j < n {
            let bn = block_size.min(n - j);
            let mut kk = 0;
            while kk < k {
                let bk = block_size.min(k - kk);

                // Extract A[i..i+bm, kk..kk+bk] into a contiguous Mat.
                let mut a_chunk = Mat::<T>::zeros(bm, bk);
                for r in 0..bm {
                    let src = (i + r) * a.cols + kk;
                    a_chunk.data[r * bk..(r + 1) * bk].copy_from_slice(&a.data[src..src + bk]);
                }
                // Extract B[kk..kk+bk, j..j+bn].
                let mut b_chunk = Mat::<T>::zeros(bk, bn);
                for r in 0..bk {
                    let src = (kk + r) * b.cols + j;
                    b_chunk.data[r * bn..(r + 1) * bn].copy_from_slice(&b.data[src..src + bn]);
                }

                let partial = submatmul(&a_chunk, &b_chunk);
                debug_assert_eq!((partial.rows, partial.cols), (bm, bn));

                for r in 0..bm {
                    let base_c = (i + r) * n + j;
                    let base_p = r * bn;
                    for cc in 0..bn {
                        c.data[base_c + cc] =
                            c.data[base_c + cc].wrapping_add(partial.data[base_p + cc]);
                    }
                }
                kk += bk;
            }
            j += bn;
        }
        i += bm;
    }
    c
}
