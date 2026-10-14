// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Method of Four Russians Multiplication (M4RM).
//!
//! ## Algorithm
//!
//! For `C = A · B` over F_2 with `A: m × n`, `B: n × p`:
//!
//! 1. Pick a strip width `k` (the dispatcher uses measured, cache-aware
//!    crossover bands). Walk A's columns in strips of width `k`:
//!    `j = 0, k, 2k, …`.
//!
//! 2. For each strip `[j, j+k)`, precompute a table `T[0..2^k]` of
//!    rows of `B`:
//!
//!    ```text
//!    T[0] = 0
//!    T[p] = XOR_{b ∈ set bits of p} B[j + b]
//!    ```
//!
//!    Built incrementally in O(2^k) row-XORs via the "natural order"
//!    construction: `T[p] = T[p ^ lowbit(p)] XOR B[j + log2(lowbit(p))]`.
//!
//! 3. For each row `i` of `A`, look up the table at index
//!    `A[i, j..j+k]` and XOR `T[slice]` into `C[i]`.
//!
//! ## Cost
//!
//! `O((n/k) · (2^k + m) · width(B))` word ops. The naive baseline is
//! `O(m · n · width(B))` for dense inputs, so M4RM is faster whenever
//! `m / (m + 2^k) > 1/k`, i.e. `m ≳ 2^k / (k - 1)`. For `k = 8`,
//! that's `m ≳ 37`.
//!
//! ## Generic over views
//!
//! The kernels take `impl MatrixRead` / `impl MatrixWrite` so they
//! work uniformly on owned `Mzd`s and `MzdView`/`MzdViewMut`
//! quadrant slices. That's the substrate Strassen needs to avoid
//! copy-in-copy-out at every recursion level.
//!
//! ## Reference
//!
//! Albrecht, Bard, Hart, "Algorithm 898: Efficient Multiplication of
//! Dense Matrices over GF(2)", ACM TOMS 37(1), 2010.

use crate::mzd::{Mzd, RADIX};
use crate::view::{MatrixRead, MatrixWrite};

// Natural-order codebook rationale.
//
// M4RI's `mzd_make_table` stores T in Gray-code order and uses an
// `L[]` permutation at apply time to map natural-binary slice
// values to Gray-code positions. The construction loop is one op
// shorter (no `trailing_zeros`) per table step.
//
// The apply phase dominates the kernel. Adding the `L[]` permutation there
// costs more than eliminating `trailing_zeros` during table construction, so
// this implementation keeps natural-order construction.

/// Default M4RM strip width. Matches libm4ri's `__M4RI_MAXKAY` for
/// the dense kernel; chosen so the 256-entry table fits comfortably
/// in L1 for any reasonable B width.
pub const M4RM_DEFAULT_K: usize = 8;

/// Number of parallel Gray-code tables used by the NTABLES-fused
/// kernel (`matmul_m4rm_ntables`). Matches libm4ri's
/// `__M4RI_M4RM_NTABLES = 8`. At the maximum `k = 8` this gives
/// `kk = 64` bits read per row per super-strip; smaller measured `k`
/// values trade additional super-strips for substantially smaller tables.
pub const M4RM_NTABLES: usize = 8;

#[doc(hidden)]
pub const M4RM_MAX_K: usize = 12;

#[doc(hidden)]
pub const M4RM_DEFAULT_M_BLOCK: usize = 1024;

#[doc(hidden)]
pub const M4RM_DEFAULT_N_BLOCK: usize = 512;

/// `C = A · B` over F_2 via M4RM, returning a freshly-allocated `C`.
/// Uses the default block width (`k = 8`). Panics on shape mismatch.
pub fn matmul_m4rm<A: MatrixRead, B: MatrixRead>(a: &A, b: &B) -> Mzd {
    matmul_m4rm_with_k(a, b, M4RM_DEFAULT_K)
}

/// Same as [`matmul_m4rm`] but the caller picks the table width `k`
/// (`1 ≤ k ≤ M4RM_MAX_K`). Used by the bench harness to tune the
/// L1/L2 trade-off: larger `k` builds a larger table once but
/// processes more rows per strip.
#[doc(hidden)]
pub fn matmul_m4rm_with_k<A: MatrixRead, B: MatrixRead>(a: &A, b: &B, k: usize) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    matmul_m4rm_with_k_into(&mut c, a, b, k);
    c
}

/// `dst = a · b` (overwriting `dst`'s pre-existing content with the
/// product, which equals XOR-accumulating into a pre-zeroed
/// destination). Default `k = 8`.
pub fn matmul_m4rm_into<D, A, B>(dst: &mut D, a: &A, b: &B)
where
    D: MatrixWrite,
    A: MatrixRead,
    B: MatrixRead,
{
    matmul_m4rm_with_k_into(dst, a, b, M4RM_DEFAULT_K);
}

/// Parametric M4RM into-destination kernel. The caller MUST pass a
/// zeroed `dst` of the right shape and `1 ≤ k ≤ M4RM_MAX_K`.
#[doc(hidden)]
pub fn matmul_m4rm_with_k_into<D, A, B>(dst: &mut D, a: &A, b: &B, k: usize)
where
    D: MatrixWrite,
    A: MatrixRead,
    B: MatrixRead,
{
    assert!(
        (1..=M4RM_MAX_K).contains(&k),
        "matmul_m4rm_with_k_into: k = {} out of range [1, {}]",
        k,
        M4RM_MAX_K
    );
    matmul_m4rm_inner(dst, a, b, k);
}

/// Parallel NTABLES kernel — gated on `--features parallel`.
/// Same algorithm as `matmul_m4rm_ntables_into`, but chunks the
/// per-super-strip row-apply loop across threads via rayon. Each
/// thread reads `a` + the shared NT tables (immutable, no
/// contention) and writes its own slice of `dst` (disjoint, no
/// locks). Restricted to `&mut Mzd` (not generic) because we need
/// the contiguous row-major backing slice for `par_chunks_mut`.
///
/// `chunk_rows` controls thread granularity. The default
/// (`M4RM_DEFAULT_CHUNK_ROWS = 256`) keeps each thread's working
/// set (chunk_rows × cols / 8 bytes) comfortably in L2 while
/// amortising the rayon-task dispatch cost.
#[cfg(feature = "parallel")]
pub fn matmul_m4rm_ntables_parallel(a: &Mzd, b: &Mzd) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    let k = ntables_auto_k(b.width());
    matmul_m4rm_ntables_parallel_with_k_into(&mut c, a, b, k, M4RM_DEFAULT_CHUNK_ROWS);
    c
}

/// `dst = a · b` via parallel NTABLES-fused M4RM. `dst` must be
/// pre-zeroed.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn matmul_m4rm_ntables_parallel_with_k_into(
    dst: &mut Mzd,
    a: &Mzd,
    b: &Mzd,
    k: usize,
    chunk_rows: usize,
) {
    use rayon::prelude::*;

    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_m4rm_ntables_parallel shape mismatch"
    );
    assert_eq!(dst.rows(), a.rows());
    assert_eq!(dst.cols(), b.cols());
    assert!(
        (1..=M4RM_DEFAULT_K).contains(&k),
        "matmul_m4rm_ntables_parallel_with_k_into: k out of range"
    );
    assert!(chunk_rows > 0, "chunk_rows must be > 0");

    let n = a.cols();
    let m = a.rows();
    if m == 0 || b.cols() == 0 || n == 0 {
        return;
    }
    const NT: usize = M4RM_NTABLES;
    let kk = k * NT;
    debug_assert!(kk <= RADIX);

    let w_c = dst.width();
    debug_assert_eq!(w_c, b.width());
    let table_entries: usize = 1 << k;
    let table_stride = table_entries * w_c;
    let mut tables: Vec<u64> = vec![0u64; NT * table_stride];
    let mask: u64 = (1u64 << k) - 1;

    let full_end = (n / kk) * kk;
    let mut j = 0;
    while j < full_end {
        // --- Build NT tables (single-threaded; ~3% of total work) ---
        for z in 0..NT {
            let base_b = j + z * k;
            let tbl_off = z * table_stride;
            for word in &mut tables[tbl_off..tbl_off + w_c] {
                *word = 0;
            }
            for p in 1..table_entries {
                let lowbit_idx = p.trailing_zeros() as usize;
                let prev = p ^ (1usize << lowbit_idx);
                let b_row_idx = base_b + lowbit_idx;
                let b_row = b.row_words(b_row_idx);
                let (lhs, rhs) = tables[tbl_off..tbl_off + table_stride].split_at_mut(p * w_c);
                let dst_row = &mut rhs[..w_c];
                let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] = prev_row[col] ^ b_row[col];
                }
            }
        }

        // --- Parallel apply: chunk dst rows across threads ---
        let tables_ref: &[u64] = &tables;
        let a_ref: &Mzd = a;
        let chunk_words = chunk_rows * w_c;
        dst.as_words_mut()
            .par_chunks_mut(chunk_words)
            .enumerate()
            .for_each(|(chunk_idx, dst_chunk)| {
                let row_base = chunk_idx * chunk_rows;
                let rows_here = dst_chunk.len() / w_c;
                for local_i in 0..rows_here {
                    let i = row_base + local_i;
                    if i >= m {
                        break;
                    }
                    let slice = a_ref.extract_bits(i, j, kk);
                    let mut t_rows: [&[u64]; NT] = [&[][..]; NT];
                    for (z, t) in t_rows.iter_mut().enumerate() {
                        let idx = ((slice >> (z * k)) & mask) as usize;
                        let row_start = z * table_stride + idx * w_c;
                        *t = &tables_ref[row_start..row_start + w_c];
                    }
                    let dst_row = &mut dst_chunk[local_i * w_c..(local_i + 1) * w_c];
                    crate::matmul::combine::combine_n_into::<NT>(dst_row, t_rows);
                }
            });

        j += kk;
    }

    // --- Tail (single-threaded; small) ---
    // Reuse the serial tail kernel via the generic path on the
    // remaining columns. We'd need to slice `a` and `b` to feed
    // it; simplest is to use the in-trait extract_bits + scalar
    // table-build inline, same as the serial kernel.
    if j < n {
        let mut j2 = j;
        while j2 < n {
            let strip = k.min(n - j2);
            let strip_table_size = 1usize << strip;
            for word in &mut tables[..w_c] {
                *word = 0;
            }
            for p in 1..strip_table_size {
                let lowbit_idx = p.trailing_zeros() as usize;
                let prev = p ^ (1usize << lowbit_idx);
                let b_row_idx = j2 + lowbit_idx;
                let b_row = b.row_words(b_row_idx);
                let (lhs, rhs) = tables.split_at_mut(p * w_c);
                let dst_row = &mut rhs[..w_c];
                let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] = prev_row[col] ^ b_row[col];
                }
            }
            for i in 0..m {
                let idx = a.extract_bits(i, j2, strip) as usize;
                if idx == 0 {
                    continue;
                }
                let t_row = &tables[idx * w_c..(idx + 1) * w_c];
                let dst_row = dst.row_words_mut(i);
                for col in 0..w_c {
                    dst_row[col] ^= t_row[col];
                }
            }
            j2 += strip;
        }
    }
}

/// Default per-thread row chunk for the parallel NTABLES kernel.
/// 256 rows × typical 128-word rows = 256 KB per chunk — comfortable
/// L2 residency, ~32 chunks per N=8192 matrix → spreads across 8–16
/// cores without too much rayon-task overhead per call.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub const M4RM_DEFAULT_CHUNK_ROWS: usize = 256;

/// View-typed parallel NTABLES kernel — same algorithm as
/// `matmul_m4rm_ntables_parallel_with_k_into` but takes any
/// `MatrixRead + Sync` for the inputs and writes into an
/// `MzdViewMut` destination. Used by Strassen-par-leaf at the leaf
/// to **skip the materialise copy** that the `&Mzd`-bound parallel
/// kernel forced. Eliminates ~270 ms of memcpy at T=4 N=65536.
///
/// Parallelism: dst is row-split into chunks (via
/// `MzdViewMut::row_split`) and each chunk is processed by one
/// rayon task. The chunks tile dst disjointly, so each task owns
/// its row range with no aliasing. The shared NT-table buffer and
/// the `&A`/`&B` inputs are read-only across threads.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn matmul_m4rm_ntables_view_parallel_with_k_into<A, B>(
    dst: &mut crate::view::MzdViewMut<'_>,
    a: &A,
    b: &B,
    k: usize,
    chunk_rows: usize,
) where
    A: MatrixRead + Sync,
    B: MatrixRead + Sync,
{
    use rayon::prelude::*;

    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_m4rm_ntables_view_parallel_with_k_into: shape mismatch"
    );
    assert_eq!(dst.rows(), a.rows());
    assert_eq!(dst.cols(), b.cols());
    assert!(
        (1..=M4RM_DEFAULT_K).contains(&k),
        "matmul_m4rm_ntables_view_parallel_with_k_into: k out of range"
    );
    assert!(chunk_rows > 0);

    let m = a.rows();
    let n = a.cols();
    if m == 0 || b.cols() == 0 || n == 0 {
        return;
    }
    const NT: usize = M4RM_NTABLES;
    let kk = k * NT;
    debug_assert!(kk <= RADIX);
    let w_c = dst.width();
    debug_assert_eq!(w_c, b.width());
    let table_entries: usize = 1 << k;
    let table_stride = table_entries * w_c;
    let mut tables: Vec<u64> = vec![0u64; NT * table_stride];
    let mask: u64 = (1u64 << k) - 1;
    let full_end = (n / kk) * kk;

    let mut j = 0;
    while j < full_end {
        // ---- Build NT Gray-code tables (sequential) ---------------
        for z in 0..NT {
            let base_b = j + z * k;
            let tbl_off = z * table_stride;
            for word in &mut tables[tbl_off..tbl_off + w_c] {
                *word = 0;
            }
            for p in 1..table_entries {
                let lowbit_idx = p.trailing_zeros() as usize;
                let prev = p ^ (1usize << lowbit_idx);
                let b_row = b.row_words(base_b + lowbit_idx);
                let (lhs, rhs) = tables[tbl_off..tbl_off + table_stride].split_at_mut(p * w_c);
                let dst_row = &mut rhs[..w_c];
                let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] = prev_row[col] ^ b_row[col];
                }
            }
        }

        // ---- Parallel apply: dst split by rows, one task per chunk
        let tables_ref: &[u64] = &tables;
        let a_ref: &A = a;
        let chunks = dst.reborrow().row_split(chunk_rows);
        chunks
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut chunk)| {
                let row_base = chunk_idx * chunk_rows;
                for local_i in 0..chunk.rows() {
                    let global_i = row_base + local_i;
                    let slice = a_ref.extract_bits(global_i, j, kk);
                    let mut t_rows: [&[u64]; NT] = [&[][..]; NT];
                    for (z, t) in t_rows.iter_mut().enumerate() {
                        let idx = ((slice >> (z * k)) & mask) as usize;
                        let row_start = z * table_stride + idx * w_c;
                        *t = &tables_ref[row_start..row_start + w_c];
                    }
                    let dst_row = chunk.row_words_mut(local_i);
                    crate::matmul::combine::combine_n_into::<NT>(dst_row, t_rows);
                }
            });

        j += kk;
    }

    // ---- Tail (sequential; small) -----------------------------------
    if j < n {
        let mut j2 = j;
        while j2 < n {
            let strip = k.min(n - j2);
            let strip_size = 1usize << strip;
            for word in &mut tables[..w_c] {
                *word = 0;
            }
            for p in 1..strip_size {
                let lowbit_idx = p.trailing_zeros() as usize;
                let prev = p ^ (1usize << lowbit_idx);
                let b_row = b.row_words(j2 + lowbit_idx);
                let (lhs, rhs) = tables.split_at_mut(p * w_c);
                let dst_row = &mut rhs[..w_c];
                let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] = prev_row[col] ^ b_row[col];
                }
            }
            for i in 0..m {
                let idx = a.extract_bits(i, j2, strip) as usize;
                if idx == 0 {
                    continue;
                }
                let t_row = &tables[idx * w_c..(idx + 1) * w_c];
                let dst_row = dst.row_words_mut(i);
                for col in 0..w_c {
                    dst_row[col] ^= t_row[col];
                }
            }
            j2 += strip;
        }
    }
}

/// NTABLES-fused M4RM with M-axis (giantstep) row blocking. Each
/// row-block of A reuses one set of 8 Gray-code tables for the full
/// inner-dim sweep before moving on; total table-build word XORs are
/// `m / m_block` times higher than plain `matmul_m4rm_ntables`, but
/// the C working set per block stays in L2 (1 MB at `m_block=1024`,
/// `N=8192`). Whether this wins depends on cache topology — bench
/// at the call site.
///
/// `m_block = 0` is treated as "no blocking" (= `matmul_m4rm_ntables`).
#[doc(hidden)]
pub fn matmul_m4rm_ntables_mblocked(a: &Mzd, b: &Mzd, m_block: usize) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    matmul_m4rm_ntables_mblocked_into(&mut c, a, b, m_block);
    c
}

/// Outer-parallel M-blocked NTABLES: rayon parallelism over the M-axis
/// row-blocks rather than inside each NTABLES call. Each thread takes
/// a sub-block of `m_block` rows and runs serial NTABLES on it. The
/// difference from `matmul_m4rm_ntables_parallel`: there the
/// parallelism is **inside** each super-strip (synchronisation
/// between strips); here each thread runs an independent sub-matmul
/// to completion, less synchronisation, potentially better for the
/// 7950X3D's 16 cores at large N. Compare empirically.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn matmul_m4rm_ntables_mblocked_parallel(a: &Mzd, b: &Mzd, m_block: usize) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    matmul_m4rm_ntables_mblocked_parallel_into(&mut c, a, b, m_block);
    c
}

/// `dst = a · b` via outer-parallel M-blocked NTABLES. `dst` must be
/// pre-zeroed.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn matmul_m4rm_ntables_mblocked_parallel_into(dst: &mut Mzd, a: &Mzd, b: &Mzd, m_block: usize) {
    use rayon::prelude::*;
    assert_eq!(a.cols(), b.rows());
    assert_eq!(dst.rows(), a.rows());
    assert_eq!(dst.cols(), b.cols());
    let m = a.rows();
    if m == 0 || a.cols() == 0 || b.cols() == 0 || m_block == 0 || m_block >= m {
        // Whole matrix fits in one block — fall to the inner-parallel
        // variant which spreads a single super-strip across threads.
        matmul_m4rm_ntables_into(dst, a, b);
        return;
    }
    let w = dst.width();
    let chunk_words = m_block * w;
    let a_ref: &Mzd = a;
    let b_ref: &Mzd = b;
    dst.as_words_mut()
        .par_chunks_mut(chunk_words)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let row_base = chunk_idx * m_block;
            let rows_here = dst_chunk.len() / w;
            // Construct an MzdView over a_ref's rows [row_base, row_base + rows_here)
            // and call the serial NTABLES kernel writing into the
            // matching MzdViewMut. We can build views from the
            // shared a_ref but not a fresh mut view from the
            // dst_chunk slice directly. Workaround: write to dst_chunk
            // as a slice by reusing the serial kernel structure
            // inline — we already have NTABLES's body, just call it
            // through views.
            //
            // Simpler: build a temporary `&mut [u64]`-backed `Mzd`
            // is overkill. The cleanest is to write a small
            // serial-NTABLES helper that takes (dst_slice, a, b,
            // row_base, rows_here, w). We open-code it here to keep
            // the body local.
            ntables_apply_into_slice(dst_chunk, rows_here, w, row_base, a_ref, b_ref);
        });
}

/// Serial NTABLES body — apply phase for a row-range of A into the
/// caller-owned destination slice. Internally builds the NTABLES
/// tables and iterates super-strips, writing into `dst_chunk`.
/// Used by the outer-parallel m-blocked driver above so each rayon
/// task owns its own contiguous destination slice (no aliasing).
#[cfg(feature = "parallel")]
fn ntables_apply_into_slice(
    dst_chunk: &mut [u64],
    rows_here: usize,
    w_c: usize,
    row_base: usize,
    a: &Mzd,
    b: &Mzd,
) {
    let n = a.cols();
    if rows_here == 0 || n == 0 || b.cols() == 0 {
        return;
    }
    let k_param = ntables_auto_k(b.width());
    const NT: usize = M4RM_NTABLES;
    let kk = k_param * NT;
    debug_assert!(kk <= RADIX);
    let table_entries: usize = 1 << k_param;
    let table_stride = table_entries * w_c;
    let mut tables: Vec<u64> = vec![0u64; NT * table_stride];
    let mask: u64 = (1u64 << k_param) - 1;
    let full_end = (n / kk) * kk;

    let mut j = 0;
    while j < full_end {
        // Build NT tables (single-threaded per task).
        for z in 0..NT {
            let base_b = j + z * k_param;
            let tbl_off = z * table_stride;
            for word in &mut tables[tbl_off..tbl_off + w_c] {
                *word = 0;
            }
            for p in 1..table_entries {
                let lowbit_idx = p.trailing_zeros() as usize;
                let prev = p ^ (1usize << lowbit_idx);
                let b_row = b.row_words(base_b + lowbit_idx);
                let (lhs, rhs) = tables[tbl_off..tbl_off + table_stride].split_at_mut(p * w_c);
                let dst_row = &mut rhs[..w_c];
                let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] = prev_row[col] ^ b_row[col];
                }
            }
        }
        // Apply to local rows.
        for local_i in 0..rows_here {
            let global_i = row_base + local_i;
            let slice = a.extract_bits(global_i, j, kk);
            let mut t_rows: [&[u64]; NT] = [&[][..]; NT];
            for (z, t) in t_rows.iter_mut().enumerate() {
                let idx = ((slice >> (z * k_param)) & mask) as usize;
                let row_start = z * table_stride + idx * w_c;
                *t = &tables[row_start..row_start + w_c];
            }
            let dst_row = &mut dst_chunk[local_i * w_c..(local_i + 1) * w_c];
            crate::matmul::combine::combine_n_into::<NT>(dst_row, t_rows);
        }
        j += kk;
    }
    // Tail (partial kk).
    if j < n {
        let mut j2 = j;
        while j2 < n {
            let strip = k_param.min(n - j2);
            let strip_size = 1usize << strip;
            for word in &mut tables[..w_c] {
                *word = 0;
            }
            for p in 1..strip_size {
                let lowbit_idx = p.trailing_zeros() as usize;
                let prev = p ^ (1usize << lowbit_idx);
                let b_row = b.row_words(j2 + lowbit_idx);
                let (lhs, rhs) = tables.split_at_mut(p * w_c);
                let dst_row = &mut rhs[..w_c];
                let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] = prev_row[col] ^ b_row[col];
                }
            }
            for local_i in 0..rows_here {
                let global_i = row_base + local_i;
                let idx = a.extract_bits(global_i, j2, strip) as usize;
                if idx == 0 {
                    continue;
                }
                let t_row = &tables[idx * w_c..(idx + 1) * w_c];
                let dst_row = &mut dst_chunk[local_i * w_c..(local_i + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] ^= t_row[col];
                }
            }
            j2 += strip;
        }
    }
}

/// `dst = a · b` via NTABLES-fused M4RM with M-axis row blocking.
/// `dst` must be pre-zeroed.
#[doc(hidden)]
pub fn matmul_m4rm_ntables_mblocked_into(dst: &mut Mzd, a: &Mzd, b: &Mzd, m_block: usize) {
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_m4rm_ntables_mblocked shape mismatch"
    );
    assert_eq!(dst.rows(), a.rows());
    assert_eq!(dst.cols(), b.cols());
    let m = a.rows();
    if m_block == 0 || m_block >= m {
        matmul_m4rm_ntables_into(dst, a, b);
        return;
    }
    let mut i = 0;
    while i < m {
        let i_end = (i + m_block).min(m);
        let a_view = a.view(i, 0, i_end, a.cols());
        let mut dst_view = dst.view_mut(i, 0, i_end, dst.cols());
        matmul_m4rm_ntables_into(&mut dst_view, &a_view, b);
        i = i_end;
    }
}

/// Cache-aware k for the NTABLES kernel. The small and middle widths use
/// pinned crossover measurements rather than the asymptotic L2 formula: table
/// construction dominates below 512 bits, while the eight tables' collective
/// L1 footprint dominates through 2048 bits. Wider rows use M4RI's per-table
/// L2 budget. Override via `matmul_m4rm_ntables_with_k_into` when benchmarking
/// a new architecture.
const L2_BYTES: usize = 1 << 20; // 1 MB

#[inline]
pub(crate) fn ntables_auto_k(width_b: usize) -> usize {
    match width_b {
        0..=4 => return 4,
        5..=6 => return 6,
        7..=16 => return 5,
        17..=32 => return 7,
        _ => {}
    }
    // Per-table budget: 2^k · width_b · 8 bytes ≤ L2
    //   → 2^k ≤ L2 / (8 · width_b)
    //   → k ≤ log2(L2 / (8 · width_b))
    let limit = L2_BYTES / (8 * width_b.max(1));
    let k_floor = if limit == 0 {
        0
    } else {
        (limit as u64).ilog2() as usize
    };
    k_floor.clamp(2, M4RM_DEFAULT_K)
}

/// NTABLES-fused M4RM: processes 8 Gray-code tables in parallel per
/// super-strip of `kk = NTABLES · k` columns. For each row of A the
/// 8 looked-up table rows + the destination row are XOR-fused in a
/// single pass via [`combine_n_into`][crate::matmul::combine::combine_n_into],
/// cutting C memory traffic 8× vs the single-table kernel.
///
/// `k` is auto-tuned from L2 cache size. Falls through to single-table
/// for any tail columns not covered by full kk-strips.
#[inline]
pub fn matmul_m4rm_ntables<A: MatrixRead, B: MatrixRead>(a: &A, b: &B) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    matmul_m4rm_ntables_into(&mut c, a, b);
    c
}

/// `dst = a · b` via NTABLES-fused M4RM with auto-k. `dst` must be
/// pre-zeroed.
#[inline]
pub fn matmul_m4rm_ntables_into<D, A, B>(dst: &mut D, a: &A, b: &B)
where
    D: MatrixWrite,
    A: MatrixRead,
    B: MatrixRead,
{
    let k = ntables_auto_k(b.width());
    matmul_m4rm_ntables_with_k_into(dst, a, b, k);
}

/// `dst = a · b` via NTABLES-fused M4RM with caller-chosen `k`.
/// `dst` must be pre-zeroed. `1 ≤ k ≤ M4RM_DEFAULT_K = 8`.
#[doc(hidden)]
#[inline]
pub fn matmul_m4rm_ntables_with_k_into<D, A, B>(dst: &mut D, a: &A, b: &B, k: usize)
where
    D: MatrixWrite,
    A: MatrixRead,
    B: MatrixRead,
{
    assert_eq!(a.cols(), b.rows(), "matmul_m4rm_ntables shape mismatch");
    assert_eq!(dst.rows(), a.rows(), "dst.rows() != a.rows()");
    assert_eq!(dst.cols(), b.cols(), "dst.cols() != b.cols()");
    assert!(
        (1..=M4RM_DEFAULT_K).contains(&k),
        "matmul_m4rm_ntables_with_k_into: k out of range [1, {}]",
        M4RM_DEFAULT_K
    );

    let n = a.cols();
    if a.rows() == 0 || b.cols() == 0 || n == 0 {
        return;
    }

    let k_param = k;
    const NT: usize = M4RM_NTABLES; // 8
    let kk = k_param * NT;
    debug_assert!(kk <= RADIX, "NTABLES kk must fit in one u64");

    let w_c = dst.width();
    debug_assert_eq!(w_c, b.width());

    let table_entries: usize = 1 << k_param;
    let table_stride = table_entries * w_c;
    // One contiguous buffer holding all NT tables.
    let mut tables: Vec<u64> = vec![0u64; NT * table_stride];

    let mask: u64 = (1u64 << k_param) - 1;

    // Process full super-strips of kk columns of A.
    let full_end = (n / kk) * kk;
    let mut j = 0;
    while j < full_end {
        // --- Build NT Gray-code tables, one per k-strip inside the super-strip ---
        for z in 0..NT {
            let base_b = j + z * k_param;
            // Table at offset z * table_stride; zero its first entry.
            let tbl_off = z * table_stride;
            for word in &mut tables[tbl_off..tbl_off + w_c] {
                *word = 0;
            }
            // Natural-order construction: T[p] = T[p ^ lowbit(p)] ^ B[base_b + log2(lowbit(p))]
            for p in 1..table_entries {
                let lowbit_idx = p.trailing_zeros() as usize;
                let prev = p ^ (1usize << lowbit_idx);
                let b_row_idx = base_b + lowbit_idx;
                let b_row = b.row_words(b_row_idx);
                let (lhs, rhs) = tables[tbl_off..tbl_off + table_stride].split_at_mut(p * w_c);
                let dst_row = &mut rhs[..w_c];
                let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] = prev_row[col] ^ b_row[col];
                }
            }
        }

        // --- Apply NT tables to every row of A in one fused pass ---
        for i in 0..a.rows() {
            // Read the full kk-bit super-strip with one extract_bits.
            let slice = a.extract_bits(i, j, kk);

            // Slice into NT k-bit chunks; build NT table-row references.
            let mut t_rows: [&[u64]; NT] = [&[][..]; NT];
            for (z, t) in t_rows.iter_mut().enumerate() {
                let idx = ((slice >> (z * k_param)) & mask) as usize;
                let row_start = z * table_stride + idx * w_c;
                *t = &tables[row_start..row_start + w_c];
            }

            let dst_row = dst.row_words_mut(i);
            crate::matmul::combine::combine_n_into::<NT>(dst_row, t_rows);
        }

        j += kk;
    }

    // --- Tail: any columns left after the kk-aligned region ---
    // Run the single-table kernel on the remainder so we cover the
    // last K-strip and any partial-K strip with one call.
    if j < n {
        // Build a view-like sub-region: A's columns [j..n], B's rows
        // [j..n]. We can't directly slice A or B (they're impl
        // MatrixRead), so we re-invoke the kernel piecewise via
        // single-table strips, accumulating into dst.
        let mut j2 = j;
        while j2 < n {
            let strip = k_param.min(n - j2);
            let strip_table_size = 1usize << strip;
            // Reuse table[0..strip_table_size * w_c] as the tail table.
            for word in &mut tables[..w_c] {
                *word = 0;
            }
            let strip_mask: u64 = if strip == 64 {
                u64::MAX
            } else {
                (1u64 << strip) - 1
            };
            for p in 1..strip_table_size {
                let lowbit_idx = p.trailing_zeros() as usize;
                let prev = p ^ (1usize << lowbit_idx);
                let b_row_idx = j2 + lowbit_idx;
                let b_row = b.row_words(b_row_idx);
                let (lhs, rhs) = tables.split_at_mut(p * w_c);
                let dst_row = &mut rhs[..w_c];
                let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
                for col in 0..w_c {
                    dst_row[col] = prev_row[col] ^ b_row[col];
                }
            }
            for i in 0..a.rows() {
                let idx = a.extract_bits(i, j2, strip) as usize;
                if idx == 0 {
                    continue;
                }
                let t_row = &tables[idx * w_c..(idx + 1) * w_c];
                let dst_row = dst.row_words_mut(i);
                for col in 0..w_c {
                    dst_row[col] ^= t_row[col];
                }
            }
            // strip_mask used implicitly through extract_bits' k arg.
            let _ = strip_mask;
            j2 += strip;
        }
    }
}

/// Cache-blocked M4RM: splits rows of A (and corresponding rows of C)
/// into row-panels of `m_block`, so a `m_block × cols(B)` slice of C
/// plus the M4RM Gray-code table fit together in L2. At `N ≥ 4096`
/// this can beat the unblocked kernel — the table is rebuilt
/// `M / m_block` times more often, but the L2-hit savings on C
/// dominate. Inputs are `&Mzd` (not generic) so we can build views
/// without threading a Submatrix trait through everything.
///
/// Default `m_block = M4RM_DEFAULT_M_BLOCK`. `k` is the M4RM
/// block-width (`M4RM_DEFAULT_K`).
#[doc(hidden)]
pub fn matmul_m4rm_blocked(a: &Mzd, b: &Mzd) -> Mzd {
    matmul_m4rm_blocked_with_k(a, b, M4RM_DEFAULT_K, M4RM_DEFAULT_M_BLOCK)
}

/// Cache-blocked M4RM with caller-chosen `k` and `m_block`. Returns a
/// fresh `C`.
#[doc(hidden)]
pub fn matmul_m4rm_blocked_with_k(a: &Mzd, b: &Mzd, k: usize, m_block: usize) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    matmul_m4rm_blocked_with_k_into(&mut c, a, b, k, m_block);
    c
}

/// N-axis cache-blocked M4RM: process B in column-strips of
/// `n_block`. The Gray-code table per (K-strip, N-strip) covers only
/// `n_block` columns, so it fits L1; total word-XOR work is identical
/// to the unblocked kernel. This is the actual cache win at large N.
///
/// `n_block` must be a multiple of 64 (so view column offsets stay
/// word-aligned).
#[doc(hidden)]
pub fn matmul_m4rm_nblocked(a: &Mzd, b: &Mzd) -> Mzd {
    matmul_m4rm_nblocked_with_k(a, b, M4RM_DEFAULT_K, M4RM_DEFAULT_N_BLOCK)
}

/// N-blocked M4RM with caller-chosen `k` and `n_block`.
#[doc(hidden)]
pub fn matmul_m4rm_nblocked_with_k(a: &Mzd, b: &Mzd, k: usize, n_block: usize) -> Mzd {
    let mut c = Mzd::new(a.rows(), b.cols());
    matmul_m4rm_nblocked_with_k_into(&mut c, a, b, k, n_block);
    c
}

/// `dst = a · b` via N-blocked M4RM. `dst` must be pre-zeroed.
#[doc(hidden)]
pub fn matmul_m4rm_nblocked_with_k_into(dst: &mut Mzd, a: &Mzd, b: &Mzd, k: usize, n_block: usize) {
    assert!(
        (1..=M4RM_MAX_K).contains(&k),
        "matmul_m4rm_nblocked_with_k_into: k = {} out of range",
        k
    );
    assert!(
        n_block > 0 && n_block.is_multiple_of(RADIX),
        "n_block must be a positive multiple of {}",
        RADIX
    );
    assert_eq!(a.cols(), b.rows(), "matmul shape mismatch");
    assert_eq!(dst.rows(), a.rows());
    assert_eq!(dst.cols(), b.cols());

    let n_cols = b.cols();
    let mut j = 0;
    while j < n_cols {
        let j_end = (j + n_block).min(n_cols);
        let b_view = b.view(0, j, b.rows(), j_end);
        let mut dst_view = dst.view_mut(0, j, dst.rows(), j_end);
        matmul_m4rm_with_k_into(&mut dst_view, a, &b_view, k);
        j = j_end;
    }
}

/// `dst = a · b` via cache-blocked M4RM. `dst` must be pre-zeroed.
#[doc(hidden)]
pub fn matmul_m4rm_blocked_with_k_into(dst: &mut Mzd, a: &Mzd, b: &Mzd, k: usize, m_block: usize) {
    assert!(
        (1..=M4RM_MAX_K).contains(&k),
        "matmul_m4rm_blocked_with_k_into: k = {} out of range",
        k
    );
    assert!(
        m_block > 0,
        "matmul_m4rm_blocked_with_k_into: m_block must be > 0"
    );
    assert_eq!(a.cols(), b.rows(), "matmul shape mismatch");
    assert_eq!(dst.rows(), a.rows());
    assert_eq!(dst.cols(), b.cols());

    let m = a.rows();
    if m == 0 || a.cols() == 0 || b.cols() == 0 {
        return;
    }
    // The Mzd::view APIs require column offsets that are multiples of
    // RADIX. We pass col 0, which is always word-aligned, so this is
    // safe. Row offsets are unrestricted.
    let _ = RADIX;
    let mut i = 0;
    while i < m {
        let i_end = (i + m_block).min(m);
        let a_view = a.view(i, 0, i_end, a.cols());
        let mut dst_view = dst.view_mut(i, 0, i_end, dst.cols());
        matmul_m4rm_with_k_into(&mut dst_view, &a_view, b, k);
        i = i_end;
    }
}

#[inline(always)]
fn matmul_m4rm_inner<D, A, B>(dst: &mut D, a: &A, b: &B, k_param: usize)
where
    D: MatrixWrite,
    A: MatrixRead,
    B: MatrixRead,
{
    assert_eq!(
        a.cols(),
        b.rows(),
        "matmul_m4rm_into: shape mismatch ({}×{}) · ({}×{})",
        a.rows(),
        a.cols(),
        b.rows(),
        b.cols()
    );
    assert_eq!(
        dst.rows(),
        a.rows(),
        "matmul_m4rm_into: dst.rows() != a.rows()"
    );
    assert_eq!(
        dst.cols(),
        b.cols(),
        "matmul_m4rm_into: dst.cols() != b.cols()"
    );
    let n = a.cols();
    if a.rows() == 0 || b.cols() == 0 || n == 0 {
        return;
    }
    let w_c = dst.width();
    debug_assert_eq!(w_c, b.width());

    let table_capacity = 1usize << k_param;
    let mut table: Vec<u64> = vec![0u64; table_capacity * w_c];

    let mut j = 0;
    while j < n {
        let strip = k_param.min(n - j);
        let table_size = 1usize << strip;

        // Build table — natural-order Gray-code construction.
        for word in &mut table[..w_c] {
            *word = 0;
        }
        for p in 1..table_size {
            let lowbit_idx = p.trailing_zeros() as usize;
            let prev = p ^ (1usize << lowbit_idx);
            let b_row_idx = j + lowbit_idx;
            let b_row = b.row_words(b_row_idx);
            let (lhs, rhs) = table.split_at_mut(p * w_c);
            let dst_row = &mut rhs[..w_c];
            let prev_row = &lhs[prev * w_c..(prev + 1) * w_c];
            for col in 0..w_c {
                dst_row[col] = prev_row[col] ^ b_row[col];
            }
        }

        // Apply the table to each row of A.
        for i in 0..a.rows() {
            let slice = a.extract_bits(i, j, strip) as usize;
            if slice == 0 {
                continue;
            }
            let t_row = &table[slice * w_c..(slice + 1) * w_c];
            let c_row = dst.row_words_mut(i);
            for col in 0..w_c {
                c_row[col] ^= t_row[col];
            }
        }

        j += strip;
    }
}

#[cfg(test)]
mod cost_model_tests {
    //! Cost-model invariant tests for the NTABLES dispatcher.
    //!
    //! These don't measure wall-clock time (flaky in CI). They
    //! exercise the dispatch-time auto-k selection (`ntables_auto_k`)
    //! and assert the **invariants** the cost model relies on:
    //!
    //! 1. **L2 budget**: the table working set
    //!    `2^k · width_b · 8 bytes` fits in `L2_BYTES` (1 MB).
    //! 2. **Measured crossovers**: the release-relevant packed widths retain
    //!    their pinned choices.
    //! 3. **Clamp range**: auto-selected `k` is always in
    //!    `[2, M4RM_DEFAULT_K]`.
    //!
    //! These are counting/formula tests rather than timing tests, so they are
    //! stable in CI and guard the dispatcher's predictions.

    use super::{ntables_auto_k, L2_BYTES, M4RM_DEFAULT_K};

    /// Invariant (1): the selected `k` fits the L2 budget — i.e.
    /// `2^k · width_b · 8 bytes ≤ L2_BYTES`. If a future change
    /// to the formula bumps the budget, this catches it.
    #[test]
    fn auto_k_respects_l2_budget() {
        // Sweep a range of width_b values covering small (single-word
        // rows) through large (multi-MiB rows). At every value, the
        // table working set must fit L2.
        for &width_b in &[1usize, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384] {
            let k = ntables_auto_k(width_b);
            let table_bytes = (1usize << k) * width_b * 8;
            assert!(
                table_bytes <= L2_BYTES,
                "auto_k({width_b}) = {k} → table {table_bytes} bytes > L2 {L2_BYTES}"
            );
        }
    }

    /// Invariant (2): release-relevant width bands retain the choices measured
    /// by the pinned crossover benchmark.
    #[test]
    fn auto_k_preserves_measured_crossover_bands() {
        assert_eq!(ntables_auto_k(4), 4);
        assert_eq!(ntables_auto_k(6), 6);
        assert_eq!(ntables_auto_k(8), 5);
        assert_eq!(ntables_auto_k(16), 5);
        assert_eq!(ntables_auto_k(32), 7);
    }

    /// Invariant (3): auto-selected `k` is always in
    /// `[2, M4RM_DEFAULT_K]`. The dispatcher's other branches
    /// (M4RM single-table fallback, naive at very small N) assume
    /// k is bounded.
    #[test]
    fn auto_k_clamped_to_range() {
        for &width_b in &[1usize, 16, 1024, 65536, usize::MAX / 16] {
            let k = ntables_auto_k(width_b);
            assert!(
                (2..=M4RM_DEFAULT_K).contains(&k),
                "auto_k({width_b}) = {k} out of [2, {M4RM_DEFAULT_K}]"
            );
        }
    }

    /// Invariant (4): single-word outputs avoid building 256-row tables.
    #[test]
    fn auto_k_keeps_single_word_tables_small() {
        assert_eq!(ntables_auto_k(1), 4);
    }

    /// Invariant (5): at `width_b = L2_BYTES / 16` (i.e., one row
    /// fills 1/16 of L2), `k = 2` (the floor) because 2^2 · width
    /// already touches L2/4 — anything larger would overflow.
    /// Pins the lower boundary of the auto-k formula.
    #[test]
    fn auto_k_floors_at_large_width() {
        // width_b = L2 / 8 / 4 = L2 / 32 → 2^k ≤ 32 → k ≤ 5; we'd
        // expect k around 5 here, but the clamp floor of 2 applies
        // for *huge* widths where 2^2 already saturates.
        let huge_width = L2_BYTES / 8; // 2^k ≤ 1 → k=0, clamped to 2
        assert_eq!(ntables_auto_k(huge_width), 2);
    }
}
