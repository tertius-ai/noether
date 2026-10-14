// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Z-surface derivations from HNF: `kernel`, `image`, `solve_left`.
//!
//! All three reduce to running [`crate::hnf::hnf_with_transforms`]
//! (or [`crate::hnf::hnf`] alone for `image`) on `Aᵀ` and reading
//! off structure. See Cohen GTM 138 §2.4.10 and §2.4.13.
//!
//! Setup: let `(H, V) = hnf_with_transforms(Aᵀ)` so `V · Aᵀ = H`
//! with `V` unimodular `n × n` and `H` an `n × m` row-HNF.
//! Equivalently `A · Vᵀ = Hᵀ`, with `r = rank(A)` nonzero columns
//! of `Hᵀ` followed by `n - r` zero columns.
//!
//! - **image**: the `r` nonzero rows of `H` (transposed) are a
//!   Z-basis of `image(A) ⊆ Z^m`. Output is `m × r`.
//! - **kernel**: rows `r..n` of `V` (transposed) are a Z-basis of
//!   `ker(A) ⊆ Z^n` since `A · (j-th column of Vᵀ) = j-th column of
//!   Hᵀ = 0` for `j ≥ r`. Output is `n × (n - r)`.
//! - **solve_left**: setting `x = Vᵀ · y` turns `Ax = b` into
//!   `Hᵀ y = b`. The first `r` columns of `Hᵀ` (the nonzero ones)
//!   are lower-triangular in the pivot-row positions, so `y_0, …,
//!   y_{r-1}` are determined by back-substitution; the remaining
//!   `y_i` are free and we set them to 0. The non-pivot rows of
//!   `Hᵀ` give consistency equations (linear constraints `b` must
//!   satisfy for a Z-solution to exist).

use std::cmp::Ordering;

use crate::storage::{CsrZ, SparseRingStorage};
use crate::Integer;

/// `r = rank(H)` for `H` already in canonical row HNF — the number
/// of leading nonzero rows. `O(r)` because non-pivot rows live at
/// the bottom by canonical-HNF row order.
fn rank_of_canonical_hnf(h: &CsrZ) -> usize {
    (0..h.rows())
        .take_while(|&j| h.iter_row(j).next().is_some())
        .count()
}

/// Pivot columns of `H` (canonical HNF), in ascending order. Each
/// is the column index of the leading nonzero in the corresponding
/// pivot row.
fn pivot_cols_of(h: &CsrZ) -> Vec<usize> {
    let r = rank_of_canonical_hnf(h);
    (0..r)
        .map(|j| h.iter_row(j).next().expect("pivot row is nonzero").0)
        .collect()
}

/// Z-basis of `image(A) ⊆ Z^m` as columns of an `m × r` sparse
/// matrix. The nonzero rows of `hnf(Aᵀ)` transposed.
pub fn image(matrix: &CsrZ) -> CsrZ {
    let m = matrix.rows();
    let at = matrix.transpose();
    let h = at.hnf();
    let r = rank_of_canonical_hnf(&h);

    let mut img_rows: Vec<Vec<(usize, Integer)>> = (0..m).map(|_| Vec::new()).collect();
    // For each pivot row j of H, distribute its entries into the
    // j-th column of the result. Outer loop over j is ascending, so
    // each img_rows[i] receives entries in ascending j order — the
    // CSR sorted-columns invariant holds.
    for j in 0..r {
        for (i, v) in h.iter_row(j) {
            img_rows[i].push((j, v.clone()));
        }
    }
    CsrZ::from_sorted_rows(m, r, img_rows)
}

/// Z-basis of `ker(A) ⊆ Z^n` as columns of an `n × nullity` sparse
/// matrix, where `nullity = n - rank(A)`. The "extra" rows `r..n`
/// of the HNF transform `V` (where `V · Aᵀ = H`) transposed.
pub fn kernel(matrix: &CsrZ) -> CsrZ {
    let n = matrix.cols();
    let at = matrix.transpose();
    let (h, v) = at.hnf_with_transforms();
    let r = rank_of_canonical_hnf(&h);
    let nullity = n - r;

    let mut k_rows: Vec<Vec<(usize, Integer)>> = (0..n).map(|_| Vec::new()).collect();
    for j in 0..nullity {
        let vrow = r + j;
        for (i, val) in v.iter_row(vrow) {
            k_rows[i].push((j, val.clone()));
        }
    }
    CsrZ::from_sorted_rows(n, nullity, k_rows)
}

/// Solve `A · x = b` over `Z`. Returns one particular solution as
/// an `n × 1` sparse vector, or `None` if no integer solution
/// exists (`b` not in `image(A)`).
///
/// Combine with [`kernel`] to enumerate every solution: the full
/// affine solution set is `x + ker(A)`.
///
/// # Panics
///
/// Panics if `b.rows() != matrix.rows()` or `b.cols() != 1`.
// Back-substitution is most readable in textbook index style:
// `y_j` is solved using `H[k][pc]` and previously computed `y_k`
// for `k < j`. Switching to iterators here would obscure the
// triangular-solve structure with zip+enumerate plumbing.
#[allow(clippy::needless_range_loop)]
pub fn solve_left(matrix: &CsrZ, b: &CsrZ) -> Option<CsrZ> {
    assert_eq!(
        b.rows(),
        matrix.rows(),
        "solve_left: b has {} rows, expected {}",
        b.rows(),
        matrix.rows()
    );
    assert_eq!(b.cols(), 1, "solve_left: b must be a column vector");

    let m = matrix.rows();
    let n = matrix.cols();

    let at = matrix.transpose();
    let (h, v) = at.hnf_with_transforms();
    let pivot_cols = pivot_cols_of(&h);
    let r = pivot_cols.len();

    // Densify b for indexed access (b is m × 1; one entry per row).
    let mut b_dense: Vec<Integer> = vec![Integer::new(); m];
    for r_idx in 0..m {
        if let Some(v) = b.get(r_idx, 0) {
            b_dense[r_idx] = v.clone();
        }
    }

    // Back-substitution: y_j = (b[pc_j] - Σ_{k<j} H[k][pc_j] · y_k) / H[j][pc_j].
    // Each step must divide exactly over Z; if not, no Z-solution.
    let mut y_prime: Vec<Integer> = vec![Integer::new(); r];
    for j in 0..r {
        let pc = pivot_cols[j];
        let mut residual = b_dense[pc].clone();
        for k in 0..j {
            if let Some(h_k_pc) = h.get(k, pc) {
                let prod = h_k_pc * &y_prime[k];
                residual -= prod;
            }
        }
        let pivot_val: Integer = h.get(j, pc).expect("pivot must exist").clone();
        if !residual.is_divisible(&pivot_val) {
            return None;
        }
        y_prime[j] = residual / pivot_val;
    }

    // Consistency check at the non-pivot rows of Hᵀ: those rows
    // contribute equations Σ_k H[k][i] · y_k = b[i] that must hold
    // for `b` to be in image(A).
    let pivot_col_set: std::collections::HashSet<usize> = pivot_cols.iter().copied().collect();
    for i in 0..m {
        if pivot_col_set.contains(&i) {
            continue;
        }
        let mut lhs = Integer::new();
        for k in 0..r {
            if let Some(h_k_i) = h.get(k, i) {
                let prod = h_k_i * &y_prime[k];
                lhs += prod;
            }
        }
        if lhs != b_dense[i] {
            return None;
        }
    }

    // x = Vᵀ · y_padded where y_padded = (y'_0, …, y'_{r-1}, 0, …).
    // Equivalently x[i] = Σ_{j=0..r} V[j][i] · y_j.
    let mut x_dense: Vec<Integer> = vec![Integer::new(); n];
    for j in 0..r {
        for (i, v_val) in v.iter_row(j) {
            let prod = v_val * &y_prime[j];
            x_dense[i] += prod;
        }
    }

    // Pack as sparse n × 1.
    let mut x_rows: Vec<Vec<(usize, Integer)>> = (0..n).map(|_| Vec::new()).collect();
    for (i, val) in x_dense.into_iter().enumerate() {
        if val.cmp0() != Ordering::Equal {
            x_rows[i].push((0, val));
        }
    }
    Some(CsrZ::from_sorted_rows(n, 1, x_rows))
}
