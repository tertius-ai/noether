// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Standalone correctness tests for `solve_left` + `nullspace`.

use m4rst::Mzd;

#[test]
fn solve_identity_returns_b() {
    for &(n, nrhs) in &[(1usize, 1usize), (8, 3), (33, 17), (65, 8)] {
        let a = Mzd::identity(n);
        let b = Mzd::random(n, nrhs, 1);
        let x = Mzd::solve_left(&a, &b).expect("identity is always solvable");
        assert_eq!(x.as_words(), b.as_words(), "I · X = B should give X = B");
    }
}

#[test]
fn solve_round_trip_on_random_square() {
    // For a random square A (full rank likely), solve(A, A · y) = y for
    // any chosen y.
    for &(n, nrhs, seed) in &[(8usize, 3usize, 1u64), (33, 5, 2), (65, 8, 3), (128, 12, 4)] {
        let a = Mzd::random(n, n, seed);
        if a.rank() < n {
            continue; // unlikely; skip if it happens
        }
        let y = Mzd::random(n, nrhs, seed.wrapping_mul(31));
        let b = Mzd::matmul(&a, &y);
        let x = Mzd::solve_left(&a, &b).expect("full-rank A should be solvable");
        // Verify A · X == B (the solution might not equal y if A has nontrivial
        // kernel, but here we picked full-rank so X = Y).
        let ax = Mzd::matmul(&a, &x);
        assert_eq!(ax.as_words(), b.as_words(), "A · X ≠ B");
    }
}

#[test]
fn solve_consistent_rank_deficient() {
    // 4 × 4 A with rank 2: rows 2 and 3 are linear combos of rows 0, 1.
    let mut a = Mzd::new(4, 4);
    a.set(0, 0, 1);
    a.set(0, 1, 1);
    a.set(1, 2, 1);
    a.set(1, 3, 1);
    a.set(2, 0, 1);
    a.set(2, 1, 1);
    a.set(2, 2, 1);
    a.set(2, 3, 1); // row 0 ⊕ row 1
    a.set(3, 0, 1);
    a.set(3, 1, 1); // row 0
                    // Pick X = [1, 0, 1, 0]^T. Compute B = A · X.
    let mut x_true = Mzd::new(4, 1);
    x_true.set(0, 0, 1);
    x_true.set(2, 0, 1);
    let b = Mzd::matmul(&a, &x_true);
    let x = Mzd::solve_left(&a, &b).expect("consistent system should solve");
    let ax = Mzd::matmul(&a, &x);
    assert_eq!(
        ax.as_words(),
        b.as_words(),
        "A · X ≠ B for rank-deficient consistent system"
    );
}

#[test]
fn solve_returns_none_on_inconsistent() {
    // A · X = B where B is outside col-space of A.
    let mut a = Mzd::new(3, 2);
    a.set(0, 0, 1);
    a.set(1, 1, 1);
    // Row 2 of A is all-zero; if B[2] = 1, the system is inconsistent.
    let mut b = Mzd::new(3, 1);
    b.set(2, 0, 1);
    let x = Mzd::solve_left(&a, &b);
    assert!(x.is_none(), "inconsistent system should return None");
}

#[test]
fn nullspace_of_identity_is_empty() {
    for &n in &[1usize, 8, 65, 128] {
        let i = Mzd::identity(n);
        let k = Mzd::nullspace(&i);
        assert_eq!(k.rows(), n);
        assert_eq!(k.cols(), 0, "identity has trivial kernel");
    }
}

#[test]
fn nullspace_of_zero_is_full() {
    let z = Mzd::new(5, 7);
    let k = Mzd::nullspace(&z);
    assert_eq!(k.rows(), 7);
    assert_eq!(k.cols(), 7, "zero matrix kernel == whole space");
    // The columns should be a basis: full rank.
    assert_eq!(k.rank(), 7);
}

#[test]
fn nullspace_columns_are_in_kernel() {
    for &(r, c, seed) in &[
        (5usize, 8usize, 1u64),
        (8, 12, 2),
        (33, 65, 3),
        (64, 128, 4),
    ] {
        let a = Mzd::random(r, c, seed);
        let k = Mzd::nullspace(&a);
        if k.cols() == 0 {
            continue;
        }
        let ak = Mzd::matmul(&a, &k);
        // Every entry of A · K must be 0.
        for i in 0..ak.rows() {
            for j in 0..ak.cols() {
                assert_eq!(ak.get(i, j), 0, "A · K[{}, {}] ≠ 0", i, j);
            }
        }
    }
}

#[test]
fn nullspace_pluq_block_solve_matches_back_substitution_convention() {
    for &(rows, cols, seed) in &[
        (0usize, 0usize, 1u64),
        (0, 7, 2),
        (5, 8, 3),
        (8, 5, 4),
        (33, 65, 5),
        (65, 33, 6),
    ] {
        let matrix = Mzd::random(rows, cols, seed);
        let (_p, _l, u, q, rank) = matrix.pluq();
        let nullity = cols - rank;
        let leading = u.submatrix(0, 0, rank, rank);
        let trailing = u.submatrix(0, rank, rank, cols);
        let dependent = Mzd::solve_left(&leading, &trailing)
            .expect("the PLUQ unit-upper pivot block is nonsingular");
        let mut basis = Mzd::new(cols, nullity);
        basis.paste_into(0, 0, &dependent);
        for (column, free_variable) in (rank..cols).enumerate() {
            basis.set(free_variable, column, 1);
        }
        let projected = Mzd::matmul(&q.transpose(), &basis);
        assert_eq!(
            projected.as_words(),
            Mzd::nullspace(&matrix).as_words(),
            "PLUQ block solve changed the canonical basis at {rows} x {cols}"
        );
    }
}

#[test]
fn inverse_of_identity_is_identity() {
    for &n in &[1usize, 8, 64, 65, 128] {
        let i = Mzd::identity(n);
        let inv = Mzd::inverse(&i).expect("identity is invertible");
        assert_eq!(inv.as_words(), i.as_words());
    }
}

#[test]
fn inverse_singular_returns_none() {
    // A 4×4 matrix with rank 3 (row 3 = row 0 ⊕ row 1).
    let mut a = Mzd::new(4, 4);
    a.set(0, 0, 1);
    a.set(0, 1, 1);
    a.set(1, 2, 1);
    a.set(1, 3, 1);
    a.set(2, 0, 1);
    a.set(3, 0, 1);
    a.set(3, 1, 1);
    a.set(3, 2, 1);
    a.set(3, 3, 1);
    assert!(
        Mzd::inverse(&a).is_none(),
        "singular matrix should return None"
    );
}

#[test]
fn inverse_round_trip() {
    // For random square A, if invertible: A · A^{-1} == I AND A^{-1} · A == I.
    for &(n, seed) in &[(8usize, 1u64), (33, 2), (65, 3), (128, 4)] {
        let a = Mzd::random(n, n, seed);
        let inv = match Mzd::inverse(&a) {
            Some(x) => x,
            None => continue, // rank-deficient; skip
        };
        let i = Mzd::identity(n);
        let ai = Mzd::matmul(&a, &inv);
        assert_eq!(ai.as_words(), i.as_words(), "A · A^{{-1}} ≠ I at n={}", n);
        let ia = Mzd::matmul(&inv, &a);
        assert_eq!(ia.as_words(), i.as_words(), "A^{{-1}} · A ≠ I at n={}", n);
    }
}

#[test]
fn rank_nullity_theorem() {
    for &(r, c, seed) in &[(5usize, 8usize, 1u64), (8, 12, 2), (33, 65, 3), (64, 64, 4)] {
        let a = Mzd::random(r, c, seed);
        let nullity = Mzd::nullspace(&a).cols();
        assert_eq!(
            a.rank() + nullity,
            c,
            "rank-nullity violated: rank={}, nullity={}, cols={}",
            a.rank(),
            nullity,
            c
        );
    }
}
