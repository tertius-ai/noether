// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Direct Smith Normal Form over `Z`.
//!
//! Pipeline:
//!
//! 1. [`column_reduce_z`] computes the sparse Z row echelon
//!    reduction — same kernel that drives [`crate::hnf::hnf`].
//! 2. Lift the `rank × cols` pivot block to a dense matrix.
//! 3. [`dense_snf`] performs the standard textbook SNF on the dense
//!    block.
//! 4. Pad the diagonal with zeros to length `min(rows, cols)`,
//!    matching the noether reference convention.
//!
//! The "sparse-input but dense-final-SNF" shape is the load-bearing
//! perf claim of decision 007 §"the kernel is sparse-input +
//! sparse-mod-p but dense-final-SNF" — at PH scale `rank ≪ cols`
//! so the dense step is cheap.
//!
//! [`dense_snf`] is the correctness floor. FLINT dense SNF remains a parity
//! oracle and a possible measured terminal-block route, not a second public
//! operation.

use std::cmp::Ordering;
use std::mem;

use crate::hnf::direct::{column_reduce_z, ColumnReductionZ};
use crate::storage::{CsrZ, SparseRingStorage};
use crate::Integer;

/// Smith Normal Form diagonal of `A`, padded with zeros to length
/// `min(rows, cols)`.
///
/// The returned vector `[d_0, d_1, ..., d_{r-1}, 0, 0, ..., 0]`
/// satisfies `d_0 | d_1 | ... | d_{r-1}` and `d_i > 0` for `i < r`.
/// Trailing zeros pad to `min(rows, cols)`, matching
/// `noether.playbooks.snf.snf_diagonal`'s convention.
pub fn snf(matrix: &CsrZ) -> Vec<Integer> {
    let n_diag = matrix.rows().min(matrix.cols());
    if n_diag == 0 {
        return Vec::new();
    }

    let ColumnReductionZ {
        working,
        pivot_cols: _,
        pivot_rows,
    } = column_reduce_z(matrix);
    let rank = pivot_rows.len();

    if rank == 0 {
        return vec![Integer::from(0); n_diag];
    }

    // Lift pivot rows to a dense `rank × cols` matrix.
    let ncols = matrix.cols();
    let mut dense: Vec<Vec<Integer>> = (0..rank).map(|_| vec![Integer::from(0); ncols]).collect();
    for (i, &r) in pivot_rows.iter().enumerate() {
        for (c, v) in &working[r] {
            dense[i][*c] = v.clone();
        }
    }

    let mut diag = dense_snf(dense);

    while diag.len() < n_diag {
        diag.push(Integer::from(0));
    }
    diag.truncate(n_diag);
    diag
}

/// Pure-Rust dense Smith Normal Form. Consumes the input matrix
/// and returns the diagonal of length `min(rows, cols)`.
///
/// Cohen GTM 138 §2.4.14 — standard textbook algorithm. For each
/// diagonal position `k`:
///
/// 1. Move the smallest-magnitude nonzero entry of the active block
///    `a[k..][k..]` to `(k, k)`.
/// 2. Sign-normalise `a[k][k] > 0`.
/// 3. Reduce row `k` and column `k` via Euclidean column / row
///    subtractions. If either reduction leaves a nonzero remainder
///    elsewhere in row `k` or column `k`, a strictly smaller `|val|`
///    has appeared — restart this iteration.
/// 4. Verify `a[k][k]` divides every entry of the remaining lower-
///    right block. If not, add the offending row into row `k` to
///    surface the obstruction, and restart.
///
/// Each iteration of the inner loop strictly decreases `|a[k][k]|`
/// (Euclidean descent), so it terminates. The outer loop processes
/// `min(m, n)` diagonal positions.
///
/// Returns `[d_0, ..., d_{r-1}, 0, ..., 0]` of length `min(m, n)`,
/// where `r` is the rank, `d_i > 0` for `i < r`, and
/// `d_0 | d_1 | ... | d_{r-1}`.
// Textbook SNF cross-references between `(k, k)` and `(i, j)` in
// the active block, swaps columns via `for row in a.iter_mut()`,
// and uses indices to express col-op vs row-op asymmetry — switching
// to iterators would obscure the algorithm.
#[allow(clippy::needless_range_loop)]
pub(crate) fn dense_snf(mut a: Vec<Vec<Integer>>) -> Vec<Integer> {
    let m = a.len();
    if m == 0 {
        return Vec::new();
    }
    let n = a[0].len();
    let n_diag = m.min(n);
    if n_diag == 0 {
        return Vec::new();
    }

    for k in 0..n_diag {
        loop {
            // (1) Smallest |val| > 0 in the active block.
            let found = find_min_in_block(&a, k, m, n);
            let Some((pi, pj)) = found else {
                // Active block is all zero — remaining diagonal is 0.
                return collect_diagonal_padded(&a, n_diag, k);
            };

            // (2) Move to (k, k).
            if pi != k {
                a.swap(pi, k);
            }
            if pj != k {
                for row in a.iter_mut() {
                    row.swap(pj, k);
                }
            }

            // Sign-normalise `a[k][k]` positive.
            if a[k][k].cmp0() == Ordering::Less {
                negate_row(&mut a[k]);
            }

            // (3) Reduce row k via column subtractions. The Euclidean
            // descent guarantees a strictly smaller pivot next iteration
            // when this leaves a nonzero remainder.
            let pivot = a[k][k].clone();
            let mut row_clean = true;
            for l in (k + 1)..n {
                let q = a[k][l].div_floor(&pivot);
                if q.cmp0() != Ordering::Equal {
                    // col_l -= q · col_k.
                    for i in 0..m {
                        let prod = &q * &a[i][k];
                        a[i][l] -= prod;
                    }
                }
                if a[k][l].cmp0() != Ordering::Equal {
                    row_clean = false;
                }
            }
            if !row_clean {
                continue;
            }

            // Reduce column k via row subtractions.
            let pivot = a[k][k].clone();
            let mut col_clean = true;
            for i in (k + 1)..m {
                let q = a[i][k].div_floor(&pivot);
                if q.cmp0() != Ordering::Equal {
                    // row_i -= q · row_k.
                    for j in 0..n {
                        let prod = &q * &a[k][j];
                        a[i][j] -= prod;
                    }
                }
                if a[i][k].cmp0() != Ordering::Equal {
                    col_clean = false;
                }
            }
            if !col_clean {
                continue;
            }

            // (4) Divisibility check on the lower-right block.
            let pivot = a[k][k].clone();
            let mut nondivisible_row: Option<usize> = None;
            'block: for i in (k + 1)..m {
                for l in (k + 1)..n {
                    if a[i][l].cmp0() != Ordering::Equal && !a[i][l].is_divisible(&pivot) {
                        nondivisible_row = Some(i);
                        break 'block;
                    }
                }
            }
            if let Some(i) = nondivisible_row {
                // Surface the obstruction: add row i into row k. The
                // entry that wasn't divisible now lives in row k, and
                // the next Euclidean reduction reveals a smaller pivot.
                for j in 0..n {
                    let v = a[i][j].clone();
                    a[k][j] += v;
                }
                continue;
            }

            // Done with pivot k.
            break;
        }
    }

    collect_diagonal_padded(&a, n_diag, n_diag)
}

#[allow(clippy::needless_range_loop)]
fn find_min_in_block(a: &[Vec<Integer>], k: usize, m: usize, n: usize) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, Integer)> = None;
    for i in k..m {
        for j in k..n {
            if a[i][j].cmp0() != Ordering::Equal {
                let abs_val = a[i][j].abs();
                match &best {
                    None => best = Some((i, j, abs_val)),
                    Some((_, _, cur_best)) if abs_val < *cur_best => {
                        best = Some((i, j, abs_val));
                    }
                    _ => {}
                }
            }
        }
    }
    best.map(|(i, j, _)| (i, j))
}

fn negate_row(row: &mut [Integer]) {
    for v in row.iter_mut() {
        let t = mem::take(v);
        *v = -t;
    }
}

#[allow(clippy::needless_range_loop)]
fn collect_diagonal_padded(a: &[Vec<Integer>], n_diag: usize, k_done: usize) -> Vec<Integer> {
    let mut out = Vec::with_capacity(n_diag);
    for k in 0..n_diag {
        if k < k_done {
            out.push(a[k][k].clone());
        } else {
            out.push(Integer::from(0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: i64) -> Integer {
        Integer::from(v)
    }

    fn dense(rows: &[Vec<i64>]) -> Vec<Vec<Integer>> {
        rows.iter()
            .map(|r| r.iter().map(|&v| z(v)).collect())
            .collect()
    }

    #[test]
    fn dense_snf_of_empty_matrix() {
        assert!(dense_snf(Vec::new()).is_empty());
    }

    #[test]
    fn dense_snf_of_zero_matrix() {
        let a = dense(&[vec![0, 0], vec![0, 0]]);
        assert_eq!(dense_snf(a), vec![z(0), z(0)]);
    }

    #[test]
    fn dense_snf_of_identity() {
        let a = dense(&[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
        assert_eq!(dense_snf(a), vec![z(1), z(1), z(1)]);
    }

    #[test]
    fn dense_snf_already_in_snf_form() {
        // diag(2, 6): 2 | 6, so already in SNF.
        let a = dense(&[vec![2, 0], vec![0, 6]]);
        assert_eq!(dense_snf(a), vec![z(2), z(6)]);
    }

    #[test]
    fn dense_snf_reorders_diagonal() {
        // diag(6, 2): not in SNF order (6 ∤ 2). SNF = (2, 6).
        let a = dense(&[vec![6, 0], vec![0, 2]]);
        assert_eq!(dense_snf(a), vec![z(2), z(6)]);
    }

    #[test]
    fn dense_snf_combines_coprime_diagonals() {
        // diag(2, 3): coprime, so gcd = 1, lcm = 6 → SNF = (1, 6).
        let a = dense(&[vec![2, 0], vec![0, 3]]);
        assert_eq!(dense_snf(a), vec![z(1), z(6)]);
    }

    #[test]
    fn dense_snf_with_zero_diagonal_entry() {
        // diag(2, 0): rank 1, SNF = (2, 0).
        let a = dense(&[vec![2, 0], vec![0, 0]]);
        assert_eq!(dense_snf(a), vec![z(2), z(0)]);
        let b = dense(&[vec![0, 0], vec![0, 2]]);
        assert_eq!(dense_snf(b), vec![z(2), z(0)]);
    }

    #[test]
    fn dense_snf_two_by_two_known() {
        // [[2, 4], [6, 8]]: |det| = 16 - 24 = -8 → |det| = 8.
        // SNF chain: d_0 | d_1, d_0 · d_1 = 8.
        // Worked out: gcd of all entries is 2, so d_0 = 2 and d_1 = 4. ✓
        let a = dense(&[vec![2, 4], vec![6, 8]]);
        assert_eq!(dense_snf(a), vec![z(2), z(4)]);
    }

    #[test]
    fn dense_snf_with_negative_entries() {
        // diag(-3, -5): SNF normalises signs → (1, 15) since gcd(3,5)=1, lcm=15.
        let a = dense(&[vec![-3, 0], vec![0, -5]]);
        assert_eq!(dense_snf(a), vec![z(1), z(15)]);
    }

    #[test]
    fn dense_snf_single_row() {
        // [[2, 4, 6]]: gcd = 2, rank 1, SNF = (2).
        let a = dense(&[vec![2, 4, 6]]);
        assert_eq!(dense_snf(a), vec![z(2)]);
    }

    #[test]
    fn dense_snf_single_column() {
        // [[2], [4], [6]]: rank 1, SNF = (2).
        let a = dense(&[vec![2], vec![4], vec![6]]);
        assert_eq!(dense_snf(a), vec![z(2)]);
    }

    #[test]
    fn dense_snf_3x3_known() {
        // diag(2, 3, 5): gcd(2,3,5)=1, so d_0=1.
        // d_0·d_1 = gcd of 2x2 minors = gcd(6, 10, 15) = 1, so d_1 = 1.
        // Wait: d_0·d_1 = 1 means d_1 = 1, and d_2 = product/1 = 30.
        // SNF = (1, 1, 30).
        let a = dense(&[vec![2, 0, 0], vec![0, 3, 0], vec![0, 0, 5]]);
        assert_eq!(dense_snf(a), vec![z(1), z(1), z(30)]);
    }

    #[test]
    fn dense_snf_rectangular_wide() {
        // [[2, 0, 4], [0, 3, 6]]: rank 2 (rows linearly independent),
        // SNF length = min(2, 3) = 2.
        // Det of all 2x2 minors: 6, 12, -12 → gcd = 6.
        // Product of diagonal = 6 (since d_0 | d_1 and d_0 · d_1 = 6, d_0 = gcd of entries = 1, d_1 = 6).
        // Wait gcd of all entries (2, 0, 4, 0, 3, 6) = 1.
        // So d_0 = 1, d_1 = 6. SNF = (1, 6).
        let a = dense(&[vec![2, 0, 4], vec![0, 3, 6]]);
        assert_eq!(dense_snf(a), vec![z(1), z(6)]);
    }

    #[test]
    fn dense_snf_divisibility_holds() {
        // Property check: d_i | d_{i+1} for several random small inputs.
        let cases = [
            dense(&[vec![6, 10], vec![15, 21]]),
            dense(&[vec![2, 4, 6], vec![3, 5, 7]]),
            dense(&[vec![0, 1, 2], vec![3, 0, 5], vec![6, 7, 0]]),
        ];
        for m in cases {
            let diag = dense_snf(m.clone());
            for win in diag.windows(2) {
                if win[1].cmp0() != Ordering::Equal {
                    assert!(
                        win[1].is_divisible(&win[0]),
                        "SNF divisibility failed: {} ∤ {} (diag = {:?})",
                        win[0],
                        win[1],
                        diag
                    );
                }
            }
        }
    }
}
