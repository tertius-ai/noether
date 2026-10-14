// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Strassen-on-M4RM matmul, **zero-copy views** edition.
//!
//! ## Algorithm
//!
//! Standard Strassen: split each of `A`, `B`, `C` into 2×2 blocks of
//! half size, compute 7 sub-products and 18 sub-additions over F_2,
//! stitch into `C`. The 7-vs-8 multiplications saving drops the
//! exponent from 3 to `log_2(7) ≈ 2.807`. Over F_2 all subtractions
//! are XOR, so the formulas are particularly clean.
//!
//! ```text
//!   M1 = (A11 + A22)(B11 + B22)   → C11, C22
//!   M2 = (A21 + A22) ·  B11       → C21, C22
//!   M3 =  A11        · (B12+B22)  → C12, C22
//!   M4 =  A22        · (B21+B11)  → C11, C21
//!   M5 = (A11 + A12) ·  B22       → C11, C12
//!   M6 = (A21 + A11)(B11 + B12)   → C22
//!   M7 = (A12 + A22)(B21 + B22)   → C11
//!
//!   C11 = M1 + M4 + M5 + M7
//!   C12 = M3 + M5
//!   C21 = M2 + M4
//!   C22 = M1 + M2 + M3 + M6
//! ```
//!
//! ## Memory layout
//!
//! Inputs `A` and `B` are sliced **without copying** via [`MzdView`].
//! The destination `C` is split into 4 disjoint mutable quadrants via
//! [`MzdViewMut::quad_split`]; each `M_i` is computed into a *shared*
//! scratch buffer and then XOR-accumulated into the right C-quadrants.
//! Per recursion level the only allocations are three scratch
//! matrices (S: `m2×n2`, T: `n2×p2`, M: `m2×p2`), reused across all
//! 7 sub-products.
//!
//! ## Recursion gate
//!
//! Recurse when **every dim is at least `threshold` AND a multiple of
//! 128**. The multiple-of-128 requirement is to keep view splits
//! word-aligned: `n/2` must itself be a multiple of 64 for the
//! [`MzdView`] constructor to accept the column offset. Off the
//! 128-grid, fall through to M4RM at this level.

use crate::matmul::m4rm::matmul_m4rm_ntables_into;
use crate::mzd::Mzd;
use crate::view::{MatrixRead, MatrixWrite, MzdView, MzdViewMut};

/// Recurse only when every dim is at least this large. Tunable.
///
/// Set to 8192 based on the systematic bench in
/// `examples/bench_systematic.rs`: M4RM scales as `N^x` with
/// `x < log_2(7) ≈ 2.807` ("sub-Strassen") for `N ≤ 8192`, so
/// recursion loses. At `N = 16384` we hit the L3-cache wall, M4RM's
/// effective exponent rises, and **one** level of Strassen avoids excess
/// direct M4RM work at `N = 16384`. Deeper recursion does not help because the
/// per-level overhead exceeds the marginal saving past one split.
pub const STRASSEN_THRESHOLD: usize = 8192;

/// Word-alignment block size for the recursion gate. `n/2` must
/// itself be a multiple of 64 (= RADIX) so the view's column
/// offset is word-aligned, hence `n` itself must be a multiple of
/// `2 * 64 = 128`.
const STRASSEN_BLOCK: usize = 128;

/// `C = A · B` over F_2 via Strassen-on-M4RM. Returns a fresh `C`.
pub fn matmul_strassen(a: &Mzd, b: &Mzd) -> Mzd {
    matmul_strassen_with_threshold(a, b, STRASSEN_THRESHOLD)
}

/// Same as [`matmul_strassen`] but with a caller-chosen recursion
/// threshold. Exposed for testing (small thresholds let unit tests
/// drive multi-level recursion on small matrices).
#[doc(hidden)]
pub fn matmul_strassen_with_threshold(a: &Mzd, b: &Mzd, threshold: usize) -> Mzd {
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_strassen: shape mismatch ({}×{}) · ({}×{})",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
    let mut c = Mzd::new(a.rows(), b.cols());
    {
        let mut c_view = c.full_view_mut();
        let a_view = a.full_view();
        let b_view = b.full_view();
        matmul_strassen_view(&mut c_view, &a_view, &b_view, threshold);
    }
    c
}

/// Strassen with the leaf calls dispatched to the parallel NTABLES
/// kernel (rayon). Gated on `--features parallel`. The 7 sub-products
/// run sequentially; each gets the full thread pool. With one or two
/// levels of recursion at huge N, this lets us combine Strassen's
/// algorithmic saving with NTABLES's parallel scaling — the closest
/// thing we have to libm4ri's Strassen + threaded BLIS stack.
#[cfg(feature = "parallel")]
pub fn matmul_strassen_parallel_leaf(a: &Mzd, b: &Mzd) -> Mzd {
    matmul_strassen_parallel_leaf_with_threshold(a, b, STRASSEN_THRESHOLD)
}

#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn matmul_strassen_parallel_leaf_with_threshold(a: &Mzd, b: &Mzd, threshold: usize) -> Mzd {
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_strassen_parallel_leaf: shape mismatch"
    );
    let mut c = Mzd::new(a.rows(), b.cols());
    {
        let mut c_view = c.full_view_mut();
        let a_view = a.full_view();
        let b_view = b.full_view();
        matmul_strassen_view_parallel_leaf(&mut c_view, &a_view, &b_view, threshold);
    }
    c
}

/// View-typed Strassen with parallel NTABLES leaf. At the leaf the
/// inputs are MzdViews and the destination is an MzdViewMut; we
/// materialise both inputs and a fresh output Mzd, run the parallel
/// kernel, then XOR the result into the dst view. Materialisation
/// cost is O(N_leaf² / 8) bytes × 3 — at N_leaf=16384, 3 × 32 MB =
/// 96 MB copied per leaf × 49 leaves for N=65536 thr=N/4 ≈ 4.7 GB
/// of copies. At 30 GB/s memory bandwidth that's ~160 ms overhead,
/// negligible against the 16+ sec leaf compute.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn matmul_strassen_view_parallel_leaf(
    dst: &mut MzdViewMut<'_>,
    a: &MzdView<'_>,
    b: &MzdView<'_>,
    threshold: usize,
) {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();

    let recursable = m >= threshold
        && n >= threshold
        && p >= threshold
        && m.is_multiple_of(STRASSEN_BLOCK)
        && n.is_multiple_of(STRASSEN_BLOCK)
        && p.is_multiple_of(STRASSEN_BLOCK);
    if !recursable {
        // Leaf: call the view-typed parallel kernel directly. No
        // materialise copies — the kernel reads `a`, `b` through
        // MatrixRead+Sync and writes into the dst view via
        // disjoint row_split chunks. Eliminates ~270 ms of memcpy
        // overhead at T=4 N=65536 vs the old materialise path.
        crate::matmul::m4rm::matmul_m4rm_ntables_view_parallel_with_k_into(
            dst,
            a,
            b,
            crate::matmul::m4rm::M4RM_DEFAULT_K,
            crate::matmul::m4rm::M4RM_DEFAULT_CHUNK_ROWS,
        );
        return;
    }

    let m2 = m / 2;
    let n2 = n / 2;
    let p2 = p / 2;

    let a11 = a.view(0, 0, m2, n2);
    let a12 = a.view(0, n2, m2, n);
    let a21 = a.view(m2, 0, m, n2);
    let a22 = a.view(m2, n2, m, n);

    let b11 = b.view(0, 0, n2, p2);
    let b12 = b.view(0, p2, n2, p);
    let b21 = b.view(n2, 0, n, p2);
    let b22 = b.view(n2, p2, n, p);

    let (mut c11, mut c12, mut c21, mut c22) = dst.reborrow().quad_split(m2, p2);

    let mut s = Mzd::new(m2, n2);
    let mut t = Mzd::new(n2, p2);
    let mut m_scratch = Mzd::new(m2, p2);

    let recurse = |s_view: &MzdView<'_>, t_view: &MzdView<'_>, m_scratch: &mut Mzd| {
        m_scratch.fill_zero();
        let mut m_view = m_scratch.full_view_mut();
        matmul_strassen_view_parallel_leaf(&mut m_view, s_view, t_view, threshold);
    };

    // Same 7-mult / 18-add sequence as standard Strassen.
    s.assign(&a11);
    s.xor_into(&a22);
    t.assign(&b11);
    t.xor_into(&b22);
    recurse(&s.full_view(), &t.full_view(), &mut m_scratch);
    c11.xor_into(&m_scratch);
    c22.xor_into(&m_scratch);

    s.assign(&a21);
    s.xor_into(&a22);
    recurse(&s.full_view(), &b11, &mut m_scratch);
    c21.xor_into(&m_scratch);
    c22.xor_into(&m_scratch);

    t.assign(&b12);
    t.xor_into(&b22);
    recurse(&a11, &t.full_view(), &mut m_scratch);
    c12.xor_into(&m_scratch);
    c22.xor_into(&m_scratch);

    t.assign(&b21);
    t.xor_into(&b11);
    recurse(&a22, &t.full_view(), &mut m_scratch);
    c11.xor_into(&m_scratch);
    c21.xor_into(&m_scratch);

    s.assign(&a11);
    s.xor_into(&a12);
    recurse(&s.full_view(), &b22, &mut m_scratch);
    c11.xor_into(&m_scratch);
    c12.xor_into(&m_scratch);

    s.assign(&a21);
    s.xor_into(&a11);
    t.assign(&b11);
    t.xor_into(&b12);
    recurse(&s.full_view(), &t.full_view(), &mut m_scratch);
    c22.xor_into(&m_scratch);

    s.assign(&a12);
    s.xor_into(&a22);
    t.assign(&b21);
    t.xor_into(&b22);
    recurse(&s.full_view(), &t.full_view(), &mut m_scratch);
    c11.xor_into(&m_scratch);
}

/// View-typed Strassen entrypoint — the recursive workhorse.
/// `dst` is assumed pre-zeroed.
#[doc(hidden)]
pub fn matmul_strassen_view(
    dst: &mut MzdViewMut<'_>,
    a: &MzdView<'_>,
    b: &MzdView<'_>,
    threshold: usize,
) {
    let m = a.rows();
    let n = a.cols();
    let p = b.cols();

    let recursable = m >= threshold
        && n >= threshold
        && p >= threshold
        && m.is_multiple_of(STRASSEN_BLOCK)
        && n.is_multiple_of(STRASSEN_BLOCK)
        && p.is_multiple_of(STRASSEN_BLOCK);
    if !recursable {
        matmul_m4rm_ntables_into(dst, a, b);
        return;
    }

    let m2 = m / 2;
    let n2 = n / 2;
    let p2 = p / 2;

    // Zero-copy quadrant views of A and B.
    let a11 = a.view(0, 0, m2, n2);
    let a12 = a.view(0, n2, m2, n);
    let a21 = a.view(m2, 0, m, n2);
    let a22 = a.view(m2, n2, m, n);

    let b11 = b.view(0, 0, n2, p2);
    let b12 = b.view(0, p2, n2, p);
    let b21 = b.view(n2, 0, n, p2);
    let b22 = b.view(n2, p2, n, p);

    // Four disjoint mutable quadrants of C.
    let (mut c11, mut c12, mut c21, mut c22) = dst.reborrow().quad_split(m2, p2);

    // Per-level scratch.
    let mut s = Mzd::new(m2, n2);
    let mut t = Mzd::new(n2, p2);
    let mut m_scratch = Mzd::new(m2, p2);

    let recurse = |s: &MzdView<'_>, t: &MzdView<'_>, m_scratch: &mut Mzd| {
        m_scratch.fill_zero();
        let mut m_view = m_scratch.full_view_mut();
        matmul_strassen_view(&mut m_view, s, t, threshold);
    };

    // M1 = (A11 + A22)(B11 + B22)  → C11, C22
    s.assign(&a11);
    s.xor_into(&a22);
    t.assign(&b11);
    t.xor_into(&b22);
    recurse(&s.full_view(), &t.full_view(), &mut m_scratch);
    c11.xor_into(&m_scratch);
    c22.xor_into(&m_scratch);

    // M2 = (A21 + A22) · B11        → C21, C22
    s.assign(&a21);
    s.xor_into(&a22);
    recurse(&s.full_view(), &b11, &mut m_scratch);
    c21.xor_into(&m_scratch);
    c22.xor_into(&m_scratch);

    // M3 = A11 · (B12 + B22)        → C12, C22
    t.assign(&b12);
    t.xor_into(&b22);
    recurse(&a11, &t.full_view(), &mut m_scratch);
    c12.xor_into(&m_scratch);
    c22.xor_into(&m_scratch);

    // M4 = A22 · (B21 + B11)        → C11, C21
    t.assign(&b21);
    t.xor_into(&b11);
    recurse(&a22, &t.full_view(), &mut m_scratch);
    c11.xor_into(&m_scratch);
    c21.xor_into(&m_scratch);

    // M5 = (A11 + A12) · B22        → C11, C12
    s.assign(&a11);
    s.xor_into(&a12);
    recurse(&s.full_view(), &b22, &mut m_scratch);
    c11.xor_into(&m_scratch);
    c12.xor_into(&m_scratch);

    // M6 = (A21 + A11)(B11 + B12)   → C22
    s.assign(&a21);
    s.xor_into(&a11);
    t.assign(&b11);
    t.xor_into(&b12);
    recurse(&s.full_view(), &t.full_view(), &mut m_scratch);
    c22.xor_into(&m_scratch);

    // M7 = (A12 + A22)(B21 + B22)   → C11
    s.assign(&a12);
    s.xor_into(&a22);
    t.assign(&b21);
    t.xor_into(&b22);
    recurse(&s.full_view(), &t.full_view(), &mut m_scratch);
    c11.xor_into(&m_scratch);
}

// ---- Peeled Strassen (handles non-128-aligned dims) --------------------

/// Largest dim ≤ `n` that's a multiple of `STRASSEN_BLOCK = 128`.
#[cfg(feature = "parallel")]
#[inline]
fn align_down(n: usize) -> usize {
    n - (n % STRASSEN_BLOCK)
}

/// Strassen-with-peeling: handles inputs whose dims are NOT all
/// multiples of 128 by splitting each matrix into an aligned inner
/// core plus trailing strips, running Strassen on the aligned core,
/// and accumulating the trail-products via NTABLES.
///
/// At N=65536 (aligned), standard Strassen-par-leaf wins. At
/// N=65000 (not aligned), the standard kernel can't fire and we
/// fell back to inner-par for ~1.5× regression. This routine
/// closes that cliff by doing Strassen on the largest 128-aligned
/// inner core (e.g., 64896 = 507·128 for N=65000).
///
/// Decomposition: with `m'`, `k'`, `n'` the largest aligned dims
/// ≤ `m`, `k`, `n`,
///
/// ```text
///   A = [A_TL  A_TR]    B = [B_TL  B_TR]
///       [A_BL  A_BR]        [B_BL  B_BR]
///
///   C_TL = A_TL · B_TL + A_TR · B_BL    (Strassen + NTABLES-into)
///   C_TR = A_TL · B_TR + A_TR · B_BR    (NTABLES + NTABLES-into)
///   C_BL = A_BL · B_TL + A_BR · B_BL    (NTABLES + NTABLES-into)
///   C_BR = A_BL · B_TR + A_BR · B_BR    (NTABLES + NTABLES-into)
/// ```
///
/// Total work: one big Strassen on the m'×k'×n' core (which gets
/// the 7-vs-8 saving) plus 7 small NTABLES products on the trails.
/// The trails are at most 127 wide on any dim — tiny compared to
/// the core at production sizes.
#[cfg(feature = "parallel")]
pub fn matmul_strassen_peeled_parallel_leaf(a: &Mzd, b: &Mzd) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    matmul_strassen_peeled_parallel_leaf_into(&mut c, a, b);
    c
}

/// `_into` counterpart to [`matmul_strassen_peeled_parallel_leaf`] —
/// writes straight into a caller-supplied `dst` instead of allocating
/// and returning a fresh `Mzd`. `dst` must already be
/// `a.rows() × b.cols()`; this fully overwrites it (zeroes it first,
/// so a reused/stale `dst` never leaks old content into the result).
#[cfg(feature = "parallel")]
pub fn matmul_strassen_peeled_parallel_leaf_into(dst: &mut Mzd, a: &Mzd, b: &Mzd) {
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_strassen_peeled_parallel_leaf_into: shape mismatch ({}×{}) · ({}×{})",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();
    assert_eq!(
        (dst.rows(), dst.cols()),
        (m, n),
        "matmul_strassen_peeled_parallel_leaf_into: dst shape mismatch"
    );
    dst.fill_zero();

    let m_inner = align_down(m);
    let k_inner = align_down(k);
    let n_inner = align_down(n);

    // If any inner dim is zero (e.g., m < 128), there's no aligned
    // core to use Strassen on — fall through to NTABLES-parallel on
    // the whole product. The caller (Mzd::matmul) should generally
    // not have routed us here in that case.
    if m_inner == 0 || k_inner == 0 || n_inner == 0 {
        crate::matmul::m4rm::matmul_m4rm_ntables_parallel_with_k_into(
            dst,
            a,
            b,
            crate::matmul::m4rm::M4RM_DEFAULT_K,
            crate::matmul::m4rm::M4RM_DEFAULT_CHUNK_ROWS,
        );
        return;
    }

    // Build immutable view quadrants of A and B. All col offsets are
    // multiples of 128 (= 2 × RADIX), so they're word-aligned for
    // MzdView::new.
    let a_tl = a.view(0, 0, m_inner, k_inner);
    let a_tr = a.view(0, k_inner, m_inner, k);
    let a_bl = a.view(m_inner, 0, m, k_inner);
    let a_br = a.view(m_inner, k_inner, m, k);

    let b_tl = b.view(0, 0, k_inner, n_inner);
    let b_tr = b.view(0, n_inner, k_inner, n);
    let b_bl = b.view(k_inner, 0, k, n_inner);
    let b_br = b.view(k_inner, n_inner, k, n);

    // Split C into 4 disjoint mutable view quadrants.
    let (mut c_tl, mut c_tr, mut c_bl, mut c_br) = dst.full_view_mut().quad_split(m_inner, n_inner);

    // ---- Inner core: A_TL · B_TL via Strassen-par-leaf -------------
    //
    // Use threshold = m_inner / 2 so we get exactly one level of
    // recursion. m_inner is a multiple of 128, so the split is
    // word-aligned; the sub-dim m_inner/2 may NOT be a multiple
    // of 128 (e.g., 64896/2 = 32448 = 128·253 + 64), and the
    // recursable check in matmul_strassen_view_parallel_leaf
    // will then bail to the parallel NTABLES leaf — exactly what
    // we want.
    matmul_strassen_view_parallel_leaf(&mut c_tl, &a_tl, &b_tl, m_inner.max(1) / 2);

    // ---- Trail accumulations: 7 NTABLES-into calls ------------------
    //
    // Each kernel call is a no-op when any of its dims is zero, so
    // partial alignment (e.g., m % 128 == 0 but n % 128 != 0) falls
    // out automatically.
    matmul_m4rm_ntables_into(&mut c_tl, &a_tr, &b_bl);

    matmul_m4rm_ntables_into(&mut c_tr, &a_tl, &b_tr);
    matmul_m4rm_ntables_into(&mut c_tr, &a_tr, &b_br);

    matmul_m4rm_ntables_into(&mut c_bl, &a_bl, &b_tl);
    matmul_m4rm_ntables_into(&mut c_bl, &a_br, &b_bl);

    matmul_m4rm_ntables_into(&mut c_br, &a_bl, &b_tr);
    matmul_m4rm_ntables_into(&mut c_br, &a_br, &b_br);
}
