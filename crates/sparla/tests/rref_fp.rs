// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Integration tests for [`CsrFp::rref`].
//!
//! Three layers of coverage:
//!
//! 1. **Hand-checked cases.** Tiny matrices with RREF computed by
//!    hand or trivially derivable (identity, zero, single row/col).
//! 2. **Algebraic-property checks.** For every test case we verify
//!    the canonical RREF invariants: each pivot is `1`, each pivot
//!    column has exactly one nonzero (the pivot), pivot columns are
//!    strictly ascending, and rows below the last pivot are zero.
//! 3. **Dense-reference parity.** A tiny dense Gaussian elimination
//!    in this file is the oracle. By RREF uniqueness, two correct
//!    implementations must agree bit-for-bit.

use sparla::storage::{CsrFp, SparseRingStorage};

const P: u64 = 7;

/// Build a `CsrFp` from a dense `Vec<Vec<i64>>`. Helper to make tests
/// readable.
fn csr_from_dense(dense: &[Vec<i64>], prime: u64) -> CsrFp {
    let rows = dense.len();
    let cols = if rows == 0 { 0 } else { dense[0].len() };
    let mut triples = Vec::new();
    for (r, row) in dense.iter().enumerate() {
        assert_eq!(row.len(), cols, "dense rows must have uniform width");
        for (c, &v) in row.iter().enumerate() {
            if v.rem_euclid(prime as i64) != 0 {
                triples.push((r, c, v));
            }
        }
    }
    CsrFp::from_coo(rows, cols, &triples, prime)
}

/// Reduce a dense `F_p` matrix to RREF. The oracle for parity tests.
/// Returns `(rref_matrix, pivot_cols)`.
// Dense Gaussian elimination is most readable in textbook index
// style: the broadcast row-op reads `m[pivot_row][j]` and writes
// `m[r][j]` in lockstep. Switching to iterators here would obscure
// the algorithm with `split_at_mut` plumbing.
#[allow(clippy::needless_range_loop)]
fn dense_rref(mat: &[Vec<i64>], prime: u64) -> (Vec<Vec<u32>>, Vec<usize>) {
    let p = prime;
    let rows = mat.len();
    let cols = if rows == 0 { 0 } else { mat[0].len() };
    let mut m: Vec<Vec<u32>> = mat
        .iter()
        .map(|r| r.iter().map(|&v| v.rem_euclid(p as i64) as u32).collect())
        .collect();
    let mut pivot_row = 0;
    let mut pivot_cols = Vec::new();
    for c in 0..cols {
        if pivot_row >= rows {
            break;
        }
        let mut found = None;
        for r in pivot_row..rows {
            if m[r][c] != 0 {
                found = Some(r);
                break;
            }
        }
        let r = match found {
            None => continue,
            Some(r) => r,
        };
        m.swap(pivot_row, r);
        let inv = pow_mod(m[pivot_row][c] as u64, p - 2, p);
        for j in 0..cols {
            m[pivot_row][j] = ((m[pivot_row][j] as u64 * inv) % p) as u32;
        }
        for r in 0..rows {
            if r == pivot_row {
                continue;
            }
            let coeff = m[r][c];
            if coeff == 0 {
                continue;
            }
            for j in 0..cols {
                let prod = (coeff as u64 * m[pivot_row][j] as u64) % p;
                m[r][j] = ((m[r][j] as u64 + p - prod) % p) as u32;
            }
        }
        pivot_cols.push(c);
        pivot_row += 1;
    }
    (m, pivot_cols)
}

fn pow_mod(base: u64, exp: u64, p: u64) -> u64 {
    let mut result: u64 = 1;
    let mut b = base % p;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % p;
        }
        b = (b * b) % p;
        e >>= 1;
    }
    result
}

/// Assert the canonical RREF invariants on a `CsrFp` already known
/// to have the given `pivot_cols` and `rank`.
fn assert_rref_canonical(matrix: &CsrFp, pivot_cols: &[usize], rank: usize) {
    assert_eq!(pivot_cols.len(), rank);
    // Pivot columns ascending.
    for win in pivot_cols.windows(2) {
        assert!(win[0] < win[1], "pivot cols not strictly ascending");
    }
    // For each pivot row i, the pivot is at column pivot_cols[i],
    // value 1, and it's the leftmost nonzero in the row.
    for (i, &pc) in pivot_cols.iter().enumerate() {
        let mut entries: Vec<(usize, u32)> = matrix.iter_row(i).collect();
        entries.sort();
        let leading = entries
            .first()
            .expect("pivot row must have at least one entry");
        assert_eq!(
            leading.0, pc,
            "row {}: leading column is {}, expected pivot col {}",
            i, leading.0, pc
        );
        assert_eq!(
            leading.1, 1,
            "row {}: pivot value is {}, expected 1",
            i, leading.1
        );
        for (c, _) in &entries {
            assert!(*c < matrix.cols());
        }
    }
    // For each pivot column c == pivot_cols[i], only row i is
    // nonzero in column c.
    for (i, &pc) in pivot_cols.iter().enumerate() {
        for r in 0..matrix.rows() {
            let v = matrix.get(r, pc).unwrap_or(0);
            if r == i {
                assert_eq!(v, 1, "pivot at ({}, {}) should be 1", r, pc);
            } else {
                assert_eq!(v, 0, "({}, {}) should be 0 in RREF (pivot col)", r, pc);
            }
        }
    }
    // Rows below the last pivot are zero.
    for r in rank..matrix.rows() {
        let row: Vec<_> = matrix.iter_row(r).collect();
        assert!(row.is_empty(), "row {} below last pivot is non-empty", r);
    }
}

/// Compute RREF via sparse and via dense; assert they agree.
fn assert_sparse_matches_dense(dense_in: &[Vec<i64>], prime: u64) {
    let sparse_in = csr_from_dense(dense_in, prime);
    let result = sparse_in.rref();
    let (dense_out, dense_pivot_cols) = dense_rref(dense_in, prime);

    assert_eq!(
        result.pivot_cols, dense_pivot_cols,
        "pivot columns disagree"
    );
    assert_eq!(result.rank, dense_pivot_cols.len(), "rank disagrees");
    assert_eq!(result.matrix.to_dense(), dense_out, "matrices disagree");
    // Algebraic invariants on the sparse output.
    assert_rref_canonical(&result.matrix, &result.pivot_cols, result.rank);
}

// -- Hand-checked tiny cases -------------------------------------------

#[test]
fn zero_matrix_has_rank_zero() {
    let a = CsrFp::zeros(3, 4, P);
    let r = a.rref();
    assert_eq!(r.rank, 0);
    assert!(r.pivot_cols.is_empty());
    assert_eq!(r.matrix.to_dense(), vec![vec![0u32; 4]; 3]);
    assert_rref_canonical(&r.matrix, &r.pivot_cols, r.rank);
}

#[test]
fn identity_is_its_own_rref() {
    let dense = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
    let a = csr_from_dense(&dense, P);
    let r = a.rref();
    assert_eq!(r.rank, 3);
    assert_eq!(r.pivot_cols, vec![0, 1, 2]);
    assert_eq!(
        r.matrix.to_dense(),
        vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
    );
    assert_rref_canonical(&r.matrix, &r.pivot_cols, r.rank);
}

#[test]
fn permuted_identity_is_reordered_to_identity() {
    // Rows are pivot rows but in shuffled order. RREF must
    // reorder them to canonical form.
    let dense = vec![vec![0, 0, 1], vec![1, 0, 0], vec![0, 1, 0]];
    let a = csr_from_dense(&dense, P);
    let r = a.rref();
    assert_eq!(r.rank, 3);
    assert_eq!(r.pivot_cols, vec![0, 1, 2]);
    assert_eq!(
        r.matrix.to_dense(),
        vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]
    );
    assert_rref_canonical(&r.matrix, &r.pivot_cols, r.rank);
}

#[test]
fn singular_two_by_two_recovers_pivot() {
    // [[2, 4], [1, 2]] over F_7: rows are linearly dependent
    // (row 0 = 2 * row 1). Expected RREF: [[1, 2], [0, 0]], rank 1.
    let dense = vec![vec![2, 4], vec![1, 2]];
    let a = csr_from_dense(&dense, P);
    let r = a.rref();
    assert_eq!(r.rank, 1);
    assert_eq!(r.pivot_cols, vec![0]);
    assert_eq!(r.matrix.to_dense(), vec![vec![1, 2], vec![0, 0]]);
    assert_rref_canonical(&r.matrix, &r.pivot_cols, r.rank);
}

#[test]
fn three_by_three_with_known_rref() {
    // A = [[2, 3, 1], [4, 6, 2], [1, 1, 1]] over F_7.
    // Row 1 = 2*row 0, so rank is 2.
    // Row-reduce by hand:
    //   normalise row 0: scale by inv(2)=4 → [1, 5, 4]  (3*4=12≡5; 1*4=4)
    //   row 1 -= 4 * row 0 = [4 - 4, 6 - 20, 2 - 16] = [0, 6-6, 2-2] = [0, 0, 0]
    //   row 2 -= 1 * row 0 = [1-1, 1-5, 1-4] = [0, -4, -3] = [0, 3, 4]
    //   normalise row 2: scale by inv(3)=5 → [0, 1, 20] = [0, 1, 6]
    //   row 0 -= 5 * row 2 = [1, 5-5, 4-30] = [1, 0, 4-2] = [1, 0, 2]
    // → RREF = [[1, 0, 2], [0, 1, 6], [0, 0, 0]], pivots [0, 1].
    let dense = vec![vec![2, 3, 1], vec![4, 6, 2], vec![1, 1, 1]];
    let a = csr_from_dense(&dense, P);
    let r = a.rref();
    assert_eq!(r.rank, 2);
    assert_eq!(r.pivot_cols, vec![0, 1]);
    assert_eq!(
        r.matrix.to_dense(),
        vec![vec![1, 0, 2], vec![0, 1, 6], vec![0, 0, 0]]
    );
    assert_rref_canonical(&r.matrix, &r.pivot_cols, r.rank);
}

#[test]
fn rectangular_wide() {
    let dense = vec![vec![1, 2, 0, 3], vec![0, 0, 1, 4]];
    let a = csr_from_dense(&dense, P);
    let r = a.rref();
    assert_eq!(r.rank, 2);
    assert_eq!(r.pivot_cols, vec![0, 2]);
    assert_eq!(
        r.matrix.to_dense(),
        vec![vec![1, 2, 0, 3], vec![0, 0, 1, 4]]
    );
    assert_rref_canonical(&r.matrix, &r.pivot_cols, r.rank);
}

#[test]
fn rectangular_tall_with_zero_rows() {
    let dense = vec![vec![0, 0, 0], vec![1, 2, 3], vec![0, 0, 0], vec![2, 4, 6]];
    let a = csr_from_dense(&dense, P);
    let r = a.rref();
    assert_eq!(r.rank, 1);
    assert_eq!(r.pivot_cols, vec![0]);
    assert_eq!(
        r.matrix.to_dense(),
        vec![vec![1, 2, 3], vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]
    );
    assert_rref_canonical(&r.matrix, &r.pivot_cols, r.rank);
}

#[test]
fn already_in_rref_is_idempotent() {
    let dense = vec![vec![1, 0, 3, 0], vec![0, 1, 5, 0], vec![0, 0, 0, 1]];
    let a = csr_from_dense(&dense, P);
    let r1 = a.rref();
    let r2 = r1.matrix.rref();
    assert_eq!(r1.matrix.to_dense(), r2.matrix.to_dense());
    assert_eq!(r1.pivot_cols, r2.pivot_cols);
    assert_eq!(r1.rank, r2.rank);
}

#[test]
fn rank_matches_rref_rank() {
    let dense = vec![vec![2, 3, 1], vec![4, 6, 2], vec![1, 1, 1]];
    let a = csr_from_dense(&dense, P);
    assert_eq!(a.rank(), a.rref().rank);
    assert_eq!(a.rank(), 2);
}

#[test]
fn empty_matrix_has_rank_zero() {
    let a = CsrFp::zeros(0, 0, P);
    let r = a.rref();
    assert_eq!(r.rank, 0);
    assert!(r.pivot_cols.is_empty());
    assert_eq!(r.matrix.rows(), 0);
    assert_eq!(r.matrix.cols(), 0);
}

#[test]
fn matrix_with_zero_columns_has_rank_zero() {
    let a = CsrFp::zeros(3, 0, P);
    let r = a.rref();
    assert_eq!(r.rank, 0);
    assert!(r.pivot_cols.is_empty());
    assert_eq!(r.matrix.rows(), 3);
    assert_eq!(r.matrix.cols(), 0);
}

// -- Dense-reference parity -------------------------------------------

#[test]
fn parity_singular_5x5() {
    let dense = vec![
        vec![1, 2, 3, 4, 5],
        vec![2, 4, 6, 8, 10],
        vec![0, 1, 0, 0, 1],
        vec![3, 0, 3, 6, 6],
        vec![1, 1, 1, 1, 1],
    ];
    assert_sparse_matches_dense(&dense, P);
}

#[test]
fn parity_full_rank_4x4() {
    let dense = vec![
        vec![1, 2, 0, 3],
        vec![0, 1, 4, 0],
        vec![5, 0, 1, 2],
        vec![0, 6, 0, 1],
    ];
    assert_sparse_matches_dense(&dense, P);
}

#[test]
fn parity_wide_3x7() {
    let dense = vec![
        vec![1, 0, 2, 0, 3, 0, 4],
        vec![0, 1, 0, 2, 0, 3, 0],
        vec![1, 1, 2, 2, 3, 3, 4],
    ];
    assert_sparse_matches_dense(&dense, P);
}

#[test]
fn parity_tall_8x3() {
    let dense = vec![
        vec![1, 0, 0],
        vec![0, 1, 0],
        vec![0, 0, 1],
        vec![1, 1, 1],
        vec![2, 3, 4],
        vec![5, 6, 0],
        vec![0, 0, 0],
        vec![3, 3, 3],
    ];
    assert_sparse_matches_dense(&dense, P);
}

#[test]
fn parity_with_larger_prime() {
    // p = 101 — exercises wider arithmetic paths.
    let dense = vec![
        vec![17, 23, 41, 0],
        vec![0, 56, 78, 13],
        vec![91, 0, 0, 88],
        vec![34, 46, 82, 0],
    ];
    assert_sparse_matches_dense(&dense, 101);
}

#[test]
fn parity_with_31_bit_prime() {
    // Largest 31-bit prime is 2^31 - 1 = 2147483647.
    let p = (1u64 << 31) - 1;
    let dense = vec![
        vec![1_000_000, 2_000_000, 0],
        vec![0, 3, 4_000_000],
        vec![p as i64 - 1, 5, 6],
    ];
    assert_sparse_matches_dense(&dense, p);
}

#[test]
fn parity_random_pseudo_3x3_set() {
    // Twenty deterministic LCG-driven inputs. RREF is uniquely
    // determined by the input, so two correct implementations must
    // agree on every one.
    let mut state: u64 = 0xDEAD_BEEF;
    for _ in 0..20 {
        let mut dense = vec![vec![0i64; 3]; 3];
        for row in dense.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 32) as i64 % 7;
            }
        }
        assert_sparse_matches_dense(&dense, P);
    }
}
