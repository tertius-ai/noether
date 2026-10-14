// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Standalone correctness tests for `echelonize` + `rank`.

use m4rst::Mzd;

#[test]
fn identity_is_full_rank() {
    for &n in &[1usize, 8, 64, 65, 128, 200] {
        let i = Mzd::identity(n);
        assert_eq!(i.rank(), n, "identity({}) rank wrong", n);
    }
}

#[test]
fn zero_has_rank_zero() {
    let z = Mzd::new(10, 12);
    assert_eq!(z.rank(), 0);
}

#[test]
fn echelonize_idempotent() {
    let m = Mzd::random(20, 30, 0xABCD);
    let mut r1 = m.clone();
    let rk1 = r1.echelonize(true);
    let mut r2 = r1.clone();
    let rk2 = r2.echelonize(true);
    assert_eq!(rk1, rk2);
    assert_eq!(r1.as_words(), r2.as_words(), "RREF should be a fixed point");
}

#[test]
fn rref_pivot_columns_are_clean() {
    let m = Mzd::random(20, 30, 0xDEAD);
    let mut r = m.clone();
    r.echelonize(true);
    // For each row, find the first 1 (the pivot column). That column must
    // be 0 in every other row.
    for i in 0..r.rows() {
        let mut pivot = None;
        for j in 0..r.cols() {
            if r.get(i, j) == 1 {
                pivot = Some(j);
                break;
            }
        }
        if let Some(p) = pivot {
            for i2 in 0..r.rows() {
                if i2 != i {
                    assert_eq!(
                        r.get(i2, p),
                        0,
                        "RREF: column {} has another 1 at row {}, expected only row {}",
                        p,
                        i2,
                        i
                    );
                }
            }
        }
    }
}

#[test]
fn ref_is_upper_triangular_in_pivot_layout() {
    let m = Mzd::random(20, 30, 0xBEEF);
    let mut r = m.clone();
    r.echelonize(false);
    // For each non-zero row i, the first nonzero column ("pivot") must be
    // strictly increasing across rows.
    let mut last_pivot: Option<usize> = None;
    for i in 0..r.rows() {
        let mut pivot = None;
        for j in 0..r.cols() {
            if r.get(i, j) == 1 {
                pivot = Some(j);
                break;
            }
        }
        if let Some(p) = pivot {
            if let Some(lp) = last_pivot {
                assert!(
                    p > lp,
                    "REF: pivots not strictly increasing (row {}: {} ≤ {})",
                    i,
                    p,
                    lp
                );
            }
            last_pivot = Some(p);
        }
    }
}

#[test]
fn rank_matches_brute_force_for_small_matrix() {
    // 4-row matrix where the third row is the XOR of the first two —
    // rank should be 3 not 4.
    let mut m = Mzd::new(4, 8);
    for j in 0..4 {
        m.set(0, j, 1);
    }
    for j in 4..8 {
        m.set(1, j, 1);
    }
    for j in 0..8 {
        m.set(2, j, 1); // == row 0 ^ row 1
    }
    m.set(3, 0, 1);
    m.set(3, 5, 1);
    assert_eq!(m.rank(), 3);
}

#[test]
fn rank_of_nilpotent_jordan_block() {
    // J_n (super-diagonal 1s) is nilpotent of nilpotency n; its rank is n - 1.
    for &n in &[2usize, 5, 8, 33, 64, 65] {
        let mut j = Mzd::new(n, n);
        for i in 0..n - 1 {
            j.set(i, i + 1, 1);
        }
        assert_eq!(j.rank(), n - 1, "Jordan block J_{} rank wrong", n);
    }
}
