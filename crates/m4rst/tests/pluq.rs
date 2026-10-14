// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Standalone correctness tests for PLUQ decomposition.
//!
//! The defining property is `P · L · U · Q == A`. Plus structure:
//! P, Q are permutation matrices; L is unit lower triangular in its
//! first rank columns; U is upper triangular in its first rank rows.

use m4rst::Mzd;

/// `is_permutation_matrix(m)`: exactly one 1 per row and per column.
fn is_permutation_matrix(m: &Mzd) -> bool {
    if m.rows() != m.cols() {
        return false;
    }
    let n = m.rows();
    for i in 0..n {
        let mut c = 0;
        for j in 0..n {
            if m.get(i, j) == 1 {
                c += 1;
            }
        }
        if c != 1 {
            return false;
        }
    }
    for j in 0..n {
        let mut c = 0;
        for i in 0..n {
            if m.get(i, j) == 1 {
                c += 1;
            }
        }
        if c != 1 {
            return false;
        }
    }
    true
}

/// Check L is unit lower triangular in its first `r` columns.
fn check_l_structure(l: &Mzd, r: usize) {
    for i in 0..l.rows() {
        for j in 0..l.cols() {
            if j >= r {
                assert_eq!(
                    l.get(i, j),
                    0,
                    "L[{}, {}] should be 0 (column past rank)",
                    i,
                    j
                );
            } else if j > i {
                assert_eq!(
                    l.get(i, j),
                    0,
                    "L[{}, {}] should be 0 (above diagonal)",
                    i,
                    j
                );
            } else if j == i && i < r {
                assert_eq!(
                    l.get(i, j),
                    1,
                    "L[{}, {}] should be 1 (unit diagonal)",
                    i,
                    j
                );
            }
        }
    }
}

/// Check U is upper triangular in its first `r` rows.
fn check_u_structure(u: &Mzd, r: usize) {
    for i in 0..u.rows() {
        for j in 0..u.cols() {
            if i >= r {
                assert_eq!(
                    u.get(i, j),
                    0,
                    "U[{}, {}] should be 0 (row past rank)",
                    i,
                    j
                );
            } else if j < i {
                assert_eq!(
                    u.get(i, j),
                    0,
                    "U[{}, {}] should be 0 (below diagonal)",
                    i,
                    j
                );
            }
        }
    }
}

fn assert_pluq_eq_a(a: &Mzd) {
    let (p, l, u, q, r) = a.pluq();
    assert!(is_permutation_matrix(&p), "P is not a permutation matrix");
    assert!(is_permutation_matrix(&q), "Q is not a permutation matrix");
    assert_eq!(l.rows(), a.rows());
    assert_eq!(l.cols(), a.rows().min(a.cols()));
    assert_eq!(u.rows(), a.rows().min(a.cols()));
    assert_eq!(u.cols(), a.cols());
    check_l_structure(&l, r);
    check_u_structure(&u, r);
    let lu = Mzd::matmul(&l, &u);
    let plu = Mzd::matmul(&p, &lu);
    let pluq = Mzd::matmul(&plu, &q);
    assert_eq!(pluq.as_words(), a.as_words(), "P · L · U · Q ≠ A");
    assert_eq!(r, a.rank(), "PLUQ rank disagrees with echelon rank");
}

#[test]
fn pluq_identity() {
    for &n in &[1usize, 8, 65, 128] {
        let i = Mzd::identity(n);
        assert_pluq_eq_a(&i);
    }
}

#[test]
fn pluq_zero() {
    for &(r, c) in &[(5usize, 7usize), (8, 64), (33, 130), (64, 64)] {
        let z = Mzd::new(r, c);
        assert_pluq_eq_a(&z);
    }
}

#[test]
fn pluq_random_full_rank_likely() {
    for &(r, c, seed) in &[
        (1usize, 1usize, 1u64),
        (4, 4, 2),
        (8, 8, 3),
        (16, 16, 4),
        (33, 33, 5),
        (64, 64, 6),
        (65, 65, 7),
        (128, 128, 8),
    ] {
        let a = Mzd::random(r, c, seed);
        assert_pluq_eq_a(&a);
    }
}

#[test]
fn pluq_rectangular() {
    for &(r, c, seed) in &[
        (5usize, 12usize, 11u64),
        (12, 5, 12),
        (8, 65, 13),
        (65, 8, 14),
    ] {
        let a = Mzd::random(r, c, seed);
        assert_pluq_eq_a(&a);
    }
}

#[test]
fn pluq_rank_deficient() {
    // A 4×8 matrix where row 2 = row 0 ⊕ row 1, row 3 = row 0 — rank should be 2.
    let mut a = Mzd::new(4, 8);
    for j in 0..4 {
        a.set(0, j, 1);
    }
    for j in 4..8 {
        a.set(1, j, 1);
    }
    for j in 0..8 {
        a.set(2, j, 1);
    }
    for j in 0..4 {
        a.set(3, j, 1);
    }
    let (_p, _l, _u, _q, r) = a.pluq();
    assert_eq!(
        r, 2,
        "expected rank 2 (rows 2, 3 are linear combos of 0, 1)"
    );
    assert_pluq_eq_a(&a);
}

#[test]
fn pluq_into_matches_pluq() {
    for &(r, c, seed) in &[
        (1usize, 1usize, 1u64),
        (5, 12, 2),
        (33, 33, 3),
        (12, 5, 4),
        (65, 8, 5),
    ] {
        let a = Mzd::random(r, c, seed);
        let (want_p, want_l, want_u, want_q, want_rank) = a.pluq();

        let k = r.min(c);
        let mut p = Mzd::new(r, r);
        let mut l = Mzd::new(r, k);
        let mut u = Mzd::new(k, c);
        let mut q = Mzd::new(c, c);
        let rank = a.pluq_into(&mut p, &mut l, &mut u, &mut q);

        assert_eq!(rank, want_rank);
        assert_eq!(p.as_words(), want_p.as_words());
        assert_eq!(l.as_words(), want_l.as_words());
        assert_eq!(u.as_words(), want_u.as_words());
        assert_eq!(q.as_words(), want_q.as_words());
    }
}

#[test]
fn pluq_into_on_reused_buffers_does_not_accumulate_stale_data() {
    // pluq_into must fully overwrite p/l/u/q, not leave any residue
    // from a prior unrelated call reusing the same allocations.
    let a1 = Mzd::random(8, 8, 0xA1);
    let a2 = Mzd::random(8, 8, 0xA2);

    let mut p = Mzd::new(8, 8);
    let mut l = Mzd::new(8, 8);
    let mut u = Mzd::new(8, 8);
    let mut q = Mzd::new(8, 8);

    a1.pluq_into(&mut p, &mut l, &mut u, &mut q); // first call: populates p/l/u/q
    let rank2 = a2.pluq_into(&mut p, &mut l, &mut u, &mut q); // reuse: must fully overwrite

    let (want_p, want_l, want_u, want_q, want_rank) = a2.pluq();
    assert_eq!(rank2, want_rank);
    assert_eq!(p.as_words(), want_p.as_words());
    assert_eq!(l.as_words(), want_l.as_words());
    assert_eq!(u.as_words(), want_u.as_words());
    assert_eq!(q.as_words(), want_q.as_words());
}
