// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Dense matmul over F_2.
//!
//! The public dispatcher (`Mzd::matmul`) selects among naive, M4RM, blocked,
//! and Strassen kernels based on input shape and feature flags.

pub mod combine;
pub mod m4rm;
pub mod naive;
pub mod strassen;

use crate::mzd::Mzd;
use crate::view::MatrixWrite;

/// Dispatch threshold: use M4RM iff `rows_a ≥ M4RM_MIN_ROWS_A` **and**
/// `cols_a ≥ M4RM_MIN_COLS_A`. Below the threshold the table-build
/// overhead dominates and naive is faster.
const M4RM_MIN_ROWS_A: usize = 32;
const M4RM_MIN_COLS_A: usize = 8;

/// Dispatch threshold for the NTABLES=8 M4RM variant: kk = 64
/// columns per super-strip, so we need at least that much inner
/// dimension to enter the fused path. Below this we fall through to
/// the single-table M4RM (which is the same kernel structure with
/// `NTABLES = 1`).
const NTABLES_MIN_INNER: usize = 64;

/// Dispatch threshold for switching from plain NTABLES to M-blocked
/// NTABLES. At small N the NTABLES working set fits cache without
/// help; the per-block table-rebuild cost is pure overhead. At
/// `N ≥ MBLOCK_MIN`, the C and table working sets escape L2 and
/// the M-block keeps a row-panel hot — net win.
const MBLOCK_MIN: usize = 12288;
/// Row-panel size for the M-blocked NTABLES dispatch. Tuned to keep
/// `m_block × cols(B)` slab of C in L2.
const MBLOCK_DEFAULT: usize = 2048;

/// Above this dimension, use serial Strassen with an M4RM leaf. The parallel
/// dispatcher uses the same choice at one thread, where parallel-leaf
/// materialization has no thread-level benefit.
const STRASSEN_SERIAL_MIN: usize = 16384;
const STRASSEN_SERIAL_THRESHOLD: usize = 8192;
/// Strassen block-alignment: every dim must be multiple of 128 so
/// the n/2 split lands on a word boundary (n/2 multiple of 64).
const STRASSEN_BLOCK_MULT: usize = 128;

/// Parallel-feature dispatch thresholds, calibrated on the repository's
/// pinned benchmark host and protected by production-branch tests.
///
/// Four regimes (calibrated with explicit rayon thread budgets):
///
/// * Tall products with at most one packed output word use
///   **outer-parallel M-blocked NTABLES** once the row and inner dimensions
///   are large enough. This is the distributed SUMMA leaf regime: parallelism
///   comes from independent output-row panels even though N is narrow.
///
/// * `[PAR_MBLOCK_MIN, PAR_INNER_MIN)`: **outer-parallel M-blocked
///   NTABLES** at `mb = 1024`. Each rayon task owns a 1024-row panel
///   and runs serial NTABLES to completion while its C slab fits cache.
///
/// * `[PAR_INNER_MIN, PAR_STRASSEN_MIN)`: **inner-parallel NTABLES**.
///   Tables shared across threads (read-only); each thread processes
///   a chunk of rows per super-strip. This reduces per-thread memory pressure
///   once panels spill L2.
///
/// * `[PAR_STRASSEN_MIN, ∞)`: **Strassen with parallel NTABLES leaf**
///   at threshold `N/2` (one recursion level). 7 leaves of `N/2 × N/2`
///   each running parallel NTABLES across all threads. This combines
///   Strassen's algorithmic saving with rayon scaling.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_DIM: usize = 6144;
#[cfg(feature = "parallel")]
const PAR_MBLOCK_MIN: usize = 6144;
#[cfg(feature = "parallel")]
const PAR_MBLOCK_SIZE: usize = 1024;
#[cfg(feature = "parallel")]
const PAR_NARROW_INNER_MIN: usize = 1024;
#[cfg(feature = "parallel")]
const PAR_NARROW_MAX_COLS: usize = 64;
#[cfg(feature = "parallel")]
const PAR_INNER_MIN: usize = 24576;
#[cfg(feature = "parallel")]
const PAR_STRASSEN_MIN_HIGH_T: usize = 49152;
#[cfg(feature = "parallel")]
const PAR_STRASSEN_MIN_LOW_T: usize = 32768;

/// Per-thread-count Strassen threshold. Below 16 threads,
/// Strassen-with-parallel-leaf wins at N ≥ 32768 because each leaf
/// is still big enough that the parallel kernel inside scales. At
/// 16 threads the leaves of N=32768 split (size N/2=16384) become
/// too small to amortise rayon-task overhead, and inner-parallel
/// NTABLES wins because it spreads each super-strip across all
/// threads with table-sharing.
#[cfg(feature = "parallel")]
fn par_strassen_threshold(nthreads: usize) -> usize {
    if nthreads >= 16 {
        PAR_STRASSEN_MIN_HIGH_T
    } else {
        PAR_STRASSEN_MIN_LOW_T
    }
}

impl Mzd {
    /// `C = A · B` over F_2.
    ///
    /// Picks the fastest available kernel automatically based on
    /// the input shape and (when `--features parallel` is on) the
    /// rayon thread count.
    ///
    /// Briefly: small inputs fall through to the naive XOR-of-rows
    /// kernel; mid-sized inputs use M4RM with NTABLES=8 fusion;
    /// very large inputs (N ≥ 16384) use serial multi-level Strassen
    /// over an NTABLES leaf. With `parallel` on, the parallel
    /// kernels kick in at `N ≥ 6144` and the dispatcher picks among
    /// outer-blocked, inner-parallel, and Strassen-with-parallel-leaf
    /// based on N and thread count.
    ///
    /// # Panics
    ///
    /// On shape mismatch: `a.cols()` must equal `b.rows()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rst::Mzd;
    /// let a = Mzd::random(128, 128, 0xA);
    /// let b = Mzd::random(128, 128, 0xB);
    /// let c = Mzd::matmul(&a, &b);
    /// assert_eq!(c.rows(), 128);
    /// assert_eq!(c.cols(), 128);
    /// // A · I = A.
    /// let i = Mzd::identity(128);
    /// assert_eq!(Mzd::matmul(&a, &i).as_words(), a.as_words());
    /// // (A·B)^T = B^T · A^T.
    /// let ab_t = Mzd::matmul(&a, &b).transpose();
    /// let bt_at = Mzd::matmul(&b.transpose(), &a.transpose());
    /// assert_eq!(ab_t.as_words(), bt_at.as_words());
    /// ```
    #[inline]
    pub fn matmul(a: &Mzd, b: &Mzd) -> Mzd {
        let mut c = Mzd::new(a.rows(), b.cols());
        Self::matmul_zeroed_into(&mut c, a, b);
        c
    }

    /// Caller-owned-output counterpart to [`Mzd::matmul`]. `dst` must already be
    /// `a.rows() × b.cols()`; this fully overwrites it (zeroes it
    /// first, so a reused/stale `dst` never leaks old content into
    /// the result — every kernel tier below XOR-*accumulates*, not
    /// overwrites, hence the upfront zero).
    ///
    /// This uses [`Mzd::matmul`]'s dispatch tree. Naive and M4RM tiers do not
    /// allocate. The Strassen tiers write through [`crate::view::MzdViewMut`]
    /// straight into `dst`'s own backing store
    /// (`strassen::matmul_strassen_view` /
    /// `matmul_strassen_view_parallel_leaf` /
    /// `matmul_strassen_peeled_parallel_leaf_into`) — the M1–M7
    /// sub-products need `O(m·n/4)` scratch (three matrices reused across all
    /// seven), but the destination is never a separate
    /// `Mzd` that gets copied into `dst`; the recursive combination
    /// writes the C-quadrants of `dst` directly via aliased windows.
    pub fn matmul_into(dst: &mut Mzd, a: &Mzd, b: &Mzd) {
        assert_eq!(
            a.cols(),
            b.rows(),
            "matmul_into: inner dimension mismatch ({}×{}) · ({}×{})",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        );
        assert_eq!(dst.rows(), a.rows(), "matmul_into: dst.rows() != a.rows()");
        assert_eq!(dst.cols(), b.cols(), "matmul_into: dst.cols() != b.cols()");
        dst.fill_zero();
        Self::matmul_zeroed_into(dst, a, b);
    }

    /// Dispatch into a freshly allocated or otherwise known-zero output.
    ///
    /// Every leaf XOR-accumulates into `dst`. Keeping this boundary private
    /// lets [`Self::matmul`] avoid clearing its newly zeroed allocation a
    /// second time while [`Self::matmul_into`] continues to accept arbitrary
    /// reusable output storage safely.
    #[inline]
    fn matmul_zeroed_into(dst: &mut Mzd, a: &Mzd, b: &Mzd) {
        assert_eq!(
            a.cols(),
            b.rows(),
            "matmul_into: inner dimension mismatch ({}×{}) · ({}×{})",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        );
        assert_eq!(dst.rows(), a.rows(), "matmul_into: dst.rows() != a.rows()");
        assert_eq!(dst.cols(), b.cols(), "matmul_into: dst.cols() != b.cols()");

        let m = a.rows();
        let n = a.cols();
        let p = b.cols();
        #[cfg(feature = "parallel")]
        {
            // Only take the parallel branches when rayon has > 1
            // thread. At rayon-T=1 the parallel-leaf path's
            // materialize overhead beats the (zero) thread saving;
            // fall through to the serial paths which include a
            // pure-Strassen branch that wins at T=1.
            let nthreads = rayon::current_num_threads();
            if nthreads >= 2 {
                if m >= PARALLEL_MIN_DIM
                    && n >= PAR_NARROW_INNER_MIN
                    && (1..=PAR_NARROW_MAX_COLS).contains(&p)
                {
                    m4rm::matmul_m4rm_ntables_mblocked_parallel_into(dst, a, b, PAR_MBLOCK_SIZE);
                    return;
                }
                let strassen_min = par_strassen_threshold(nthreads);
                // (a) Aligned Strassen-par-leaf — fastest path when
                //     every dim is a multiple of 128. See `matmul`'s
                //     history for the depth-threshold rationale.
                //     Genuinely alloc-free: writes through a view
                //     straight into `dst`'s own backing store.
                if m >= strassen_min
                    && n >= strassen_min
                    && p >= strassen_min
                    && m.is_multiple_of(128)
                    && n.is_multiple_of(128)
                    && p.is_multiple_of(128)
                {
                    let depth_thr = if nthreads >= 16 { m / 2 + 1 } else { m / 4 + 1 };
                    let mut dst_view = dst.full_view_mut();
                    let a_view = a.full_view();
                    let b_view = b.full_view();
                    strassen::matmul_strassen_view_parallel_leaf(
                        &mut dst_view,
                        &a_view,
                        &b_view,
                        depth_thr,
                    );
                    return;
                }
                // (b) Peeled Strassen — also alloc-free via its own
                //     `_into` form.
                if m >= strassen_min && n >= strassen_min && p >= strassen_min {
                    strassen::matmul_strassen_peeled_parallel_leaf_into(dst, a, b);
                    return;
                }
                if m >= PAR_INNER_MIN && n >= PAR_INNER_MIN && p >= PAR_INNER_MIN {
                    let k = m4rm::ntables_auto_k(b.width());
                    m4rm::matmul_m4rm_ntables_parallel_with_k_into(
                        dst,
                        a,
                        b,
                        k,
                        m4rm::M4RM_DEFAULT_CHUNK_ROWS,
                    );
                    return;
                }
                if m >= PAR_MBLOCK_MIN && n >= PAR_MBLOCK_MIN && p >= PAR_MBLOCK_MIN {
                    m4rm::matmul_m4rm_ntables_mblocked_parallel_into(dst, a, b, PAR_MBLOCK_SIZE);
                    return;
                }
                if m >= PARALLEL_MIN_DIM && n >= PARALLEL_MIN_DIM && p >= PARALLEL_MIN_DIM {
                    let k = m4rm::ntables_auto_k(b.width());
                    m4rm::matmul_m4rm_ntables_parallel_with_k_into(
                        dst,
                        a,
                        b,
                        k,
                        m4rm::M4RM_DEFAULT_CHUNK_ROWS,
                    );
                    return;
                }
            }
        }
        // Serial / single-thread paths. Pure serial Strassen wins
        // over flat NTABLES at large N regardless of thread count.
        // Genuinely alloc-free: writes through a view straight into
        // `dst`'s own backing store (see `matmul_strassen_view`).
        if m >= STRASSEN_SERIAL_MIN
            && n >= STRASSEN_SERIAL_MIN
            && p >= STRASSEN_SERIAL_MIN
            && m.is_multiple_of(STRASSEN_BLOCK_MULT)
            && n.is_multiple_of(STRASSEN_BLOCK_MULT)
            && p.is_multiple_of(STRASSEN_BLOCK_MULT)
        {
            let mut dst_view = dst.full_view_mut();
            let a_view = a.full_view();
            let b_view = b.full_view();
            strassen::matmul_strassen_view(
                &mut dst_view,
                &a_view,
                &b_view,
                STRASSEN_SERIAL_THRESHOLD,
            );
            return;
        }
        if m >= MBLOCK_MIN && n >= MBLOCK_MIN && p >= MBLOCK_MIN {
            m4rm::matmul_m4rm_ntables_mblocked_into(dst, a, b, MBLOCK_DEFAULT);
        } else if m >= M4RM_MIN_ROWS_A && n >= NTABLES_MIN_INNER {
            m4rm::matmul_m4rm_ntables_into(dst, a, b);
        } else if m >= M4RM_MIN_ROWS_A && n >= M4RM_MIN_COLS_A {
            m4rm::matmul_m4rm_into(dst, a, b);
        } else {
            naive::matmul_naive_into(dst, a, b);
        }
    }
}
