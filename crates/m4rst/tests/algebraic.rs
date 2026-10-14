// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Adversarial *algebraic* edge cases — matrices with specific
//! structure that pin behaviour the way randomised tests can't:
//!
//! - All-zero and all-one matrices.
//! - Jordan blocks (nilpotent, rank n-1).
//! - Permutation matrices (rank n, easy PLUQ).
//! - Upper / lower triangular matrices.
//! - Rank-1 outer products.
//! - Matrices where every row equals row 0 (rank 1 even for huge n).
//! - Matrices where row k = row 0 ⊕ row 1 (provable rank reduction).

use m4rst::Mzd;

fn schoolbook(a: &Mzd, b: &Mzd) -> Mzd {
    assert_eq!(a.cols(), b.rows());
    let mut c = Mzd::new(a.rows(), b.cols());
    for i in 0..a.rows() {
        for j in 0..b.cols() {
            let mut acc = 0u8;
            for k in 0..a.cols() {
                acc ^= a.get(i, k) & b.get(k, j);
            }
            if acc != 0 {
                c.set(i, j, 1);
            }
        }
    }
    c
}

// ---- All-zero / all-one matrices ---------------------------------------

#[test]
fn all_zero_full_battery() {
    for &n in &[1usize, 64, 65, 127, 128, 129, 256, 500] {
        let z = Mzd::new(n, n);
        // Rank, nullspace, inverse.
        assert_eq!(z.rank(), 0);
        let kern = Mzd::nullspace(&z);
        assert_eq!(kern.rows(), n);
        assert_eq!(kern.cols(), n);
        assert_eq!(kern.rank(), n, "nullspace of 0 should span everything");
        assert!(
            Mzd::inverse(&z).is_none(),
            "inverse(0) must be None at n={}",
            n
        );
        // PLUQ.
        let (p, l, u, _q, rank) = z.pluq();
        assert_eq!(rank, 0, "PLUQ rank of zero != 0 at n={}", n);
        let plu = Mzd::matmul(&p, &Mzd::matmul(&l, &u));
        for &w in plu.as_words() {
            assert_eq!(w, 0, "P·L·U of zero matrix not zero at n={}", n);
        }
        // 0 · X = 0.
        let x = Mzd::random(n, 3, 0xCAFE);
        let prod = Mzd::matmul(&z, &x);
        for &w in prod.as_words() {
            assert_eq!(w, 0);
        }
    }
}

#[test]
fn all_one_matrix() {
    for &n in &[2usize, 64, 65, 128, 129] {
        let mut a = Mzd::new(n, n);
        for i in 0..n {
            for j in 0..n {
                a.set(i, j, 1);
            }
        }
        // All-ones over F_2: rank = 1.
        assert_eq!(a.rank(), 1, "rank(J_{}) = 1 expected", n);
        // PLUQ round-trip.
        let (p, l, u, q, rank) = a.pluq();
        assert_eq!(rank, 1);
        let pluq = Mzd::matmul(&Mzd::matmul(&p, &Mzd::matmul(&l, &u)), &q);
        assert_eq!(
            pluq.as_words(),
            a.as_words(),
            "P·L·U·Q of J_{} != J_{}",
            n,
            n
        );
        // nullspace dim = n - 1.
        assert_eq!(Mzd::nullspace(&a).cols(), n - 1);
    }
}

// ---- Jordan / nilpotent ------------------------------------------------

/// Super-diagonal nilpotent block J_n. J_n^n = 0, rank = n-1, ker
/// spanned by the first basis vector.
#[test]
fn jordan_block_battery() {
    for &n in &[2usize, 64, 65, 128, 129, 256] {
        let mut j = Mzd::new(n, n);
        for i in 0..n - 1 {
            j.set(i, i + 1, 1);
        }
        // Rank n-1.
        assert_eq!(j.rank(), n - 1, "rank(J_{}) = n-1 expected", n);
        // PLUQ.
        let (p, l, u, q, rank) = j.pluq();
        assert_eq!(rank, n - 1);
        let pluq = Mzd::matmul(&Mzd::matmul(&p, &Mzd::matmul(&l, &u)), &q);
        assert_eq!(pluq.as_words(), j.as_words());
        // Nullspace dim = 1.
        let kern = Mzd::nullspace(&j);
        assert_eq!(kern.cols(), 1);
        // J · kern = 0.
        let jk = Mzd::matmul(&j, &kern);
        for &w in jk.as_words() {
            assert_eq!(w, 0);
        }
        // Inverse is None.
        assert!(
            Mzd::inverse(&j).is_none(),
            "inverse(J_{}) should be None",
            n
        );
    }
}

// ---- Permutation matrices ----------------------------------------------

/// Permutation matrix: cyclic shift (i → i+1 mod n).
#[test]
fn permutation_matrix_battery() {
    for &n in &[1usize, 64, 65, 128, 129, 256] {
        let mut p = Mzd::new(n, n);
        for i in 0..n {
            p.set(i, (i + 1) % n, 1);
        }
        // Rank n.
        assert_eq!(p.rank(), n);
        // Inverse exists and equals transpose (since P is orthogonal in F_2 too).
        let inv = Mzd::inverse(&p).expect("permutation is invertible");
        assert_eq!(
            inv.as_words(),
            p.transpose().as_words(),
            "P^-1 = P^T for permutations (n={})",
            n
        );
        // P · P^T = I.
        let id = Mzd::identity(n);
        let ppt = Mzd::matmul(&p, &p.transpose());
        assert_eq!(ppt.as_words(), id.as_words());
    }
}

// ---- Triangular matrices ------------------------------------------------

/// Strict upper triangular (T[i,j] = 1 iff i < j).
#[test]
fn strict_upper_triangular() {
    for &n in &[2usize, 64, 65, 128] {
        let mut t = Mzd::new(n, n);
        for i in 0..n {
            for j in (i + 1)..n {
                t.set(i, j, 1);
            }
        }
        // Strict upper triangular: nilpotent, rank ≤ n-1.
        let r = t.rank();
        assert!(r < n, "rank({}-strict-upper) should be < n, got {}", n, r);
        // Last row is zero → rank = n-1.
        assert_eq!(r, n - 1);
        // Inverse is None.
        assert!(Mzd::inverse(&t).is_none());
    }
}

/// Unit upper triangular (T[i,j] = 1 iff i ≤ j) — invertible.
#[test]
fn unit_upper_triangular_is_invertible() {
    for &n in &[1usize, 64, 65, 128, 129] {
        let mut t = Mzd::new(n, n);
        for i in 0..n {
            for j in i..n {
                t.set(i, j, 1);
            }
        }
        assert_eq!(t.rank(), n, "unit-triangular rank wrong at n={}", n);
        let inv = Mzd::inverse(&t).expect("unit upper triangular is invertible");
        let id = Mzd::identity(n);
        assert_eq!(Mzd::matmul(&t, &inv).as_words(), id.as_words());
        assert_eq!(Mzd::matmul(&inv, &t).as_words(), id.as_words());
    }
}

// ---- Rank-1 outer products + duplicate-row matrices --------------------

/// All rows equal to row 0 → rank = 1 (if row 0 non-zero).
#[test]
fn duplicated_row_is_rank_one() {
    for &(r, c) in &[(8usize, 8usize), (64, 64), (65, 130), (130, 65), (200, 200)] {
        let mut a = Mzd::new(r, c);
        // Row 0: alternating 1s.
        for j in 0..c {
            if j.is_multiple_of(2) {
                a.set(0, j, 1);
            }
        }
        // All other rows are copies.
        for i in 1..r {
            for j in 0..c {
                if a.get(0, j) == 1 {
                    a.set(i, j, 1);
                }
            }
        }
        // Expect rank 1.
        assert_eq!(
            a.rank(),
            1,
            "duplicate-row rank should be 1 at ({},{})",
            r,
            c
        );
        // Nullity = c - 1.
        assert_eq!(Mzd::nullspace(&a).cols(), c - 1);
    }
}

/// Outer product: a[i,j] = x[i] AND y[j]. Rank ≤ 1.
#[test]
fn outer_product_is_rank_one() {
    for &(r, c) in &[(8usize, 8usize), (64, 65), (65, 64), (130, 130)] {
        let x = Mzd::random(r, 1, 0xAA);
        let y = Mzd::random(1, c, 0xBB);
        let a = Mzd::matmul(&x, &y);
        // rank(xy^T) ≤ 1; equals 1 unless x or y is all-zero.
        let nonzero_x = (0..r).any(|i| x.get(i, 0) == 1);
        let nonzero_y = (0..c).any(|j| y.get(0, j) == 1);
        let expect_rank = if nonzero_x && nonzero_y { 1 } else { 0 };
        assert_eq!(a.rank(), expect_rank, "outer product rank at ({},{})", r, c);
    }
}

/// Row k = row 0 ⊕ row 1: forces a known rank reduction.
#[test]
fn linearly_dependent_third_row() {
    let mut a = Mzd::random(4, 16, 0xC0DE);
    // Set row 2 = row 0 ⊕ row 1.
    for j in 0..16 {
        let v = a.get(0, j) ^ a.get(1, j);
        a.set(2, j, v);
    }
    // Now A has at most rank 3 (row 2 is dependent).
    assert!(a.rank() <= 3, "rank should be ≤ 3 after forced dependence");
}

// ---- Matmul agrees with schoolbook on structured inputs ----------------

#[test]
fn matmul_structured_inputs() {
    for &n in &[2usize, 64, 65, 128, 129] {
        // J_n (Jordan block) × I_n = J_n.
        let mut j = Mzd::new(n, n);
        for i in 0..n - 1 {
            j.set(i, i + 1, 1);
        }
        let id = Mzd::identity(n);
        assert_eq!(Mzd::matmul(&j, &id).as_words(), j.as_words());
        assert_eq!(Mzd::matmul(&id, &j).as_words(), j.as_words());
        // J_n^2: shifts by 2. Should match schoolbook.
        let j2 = Mzd::matmul(&j, &j);
        let j2_school = schoolbook(&j, &j);
        assert_eq!(
            j2.as_words(),
            j2_school.as_words(),
            "J^2 mismatch at n={}",
            n
        );
        // (PERMUTATION)^n = I for a cyclic shift of period n.
        let mut p = Mzd::new(n, n);
        for i in 0..n {
            p.set(i, (i + 1) % n, 1);
        }
        let mut pn = Mzd::identity(n);
        for _ in 0..n {
            pn = Mzd::matmul(&pn, &p);
        }
        assert_eq!(pn.as_words(), id.as_words(), "cycle^n != I at n={}", n);
    }
}

// ---- Numerical / bit-pattern attack ------------------------------------

/// Alternating bit pattern A[i,j] = (i+j) & 1. Checkerboard.
#[test]
fn checkerboard_matmul() {
    for &n in &[8usize, 64, 65, 128, 129] {
        let mut a = Mzd::new(n, n);
        for i in 0..n {
            for j in 0..n {
                if (i + j).is_multiple_of(2) {
                    a.set(i, j, 1);
                }
            }
        }
        // a · a should match schoolbook bit-for-bit.
        let aa = Mzd::matmul(&a, &a);
        let aa_school = schoolbook(&a, &a);
        assert_eq!(
            aa.as_words(),
            aa_school.as_words(),
            "checkerboard^2 at n={}",
            n
        );
    }
}

/// A matrix where every column is identical → rank ≤ 1.
#[test]
fn duplicate_column_matrix() {
    for &n in &[8usize, 64, 65, 128] {
        let mut a = Mzd::new(n, n);
        // Column 0: alternating 1s.
        for i in 0..n {
            if i.is_multiple_of(2) {
                a.set(i, 0, 1);
            }
        }
        // Copy column 0 to all other columns.
        for j in 1..n {
            for i in 0..n {
                if a.get(i, 0) == 1 {
                    a.set(i, j, 1);
                }
            }
        }
        assert_eq!(a.rank(), 1, "duplicate-column rank at n={}", n);
    }
}
