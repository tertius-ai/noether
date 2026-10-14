// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Tests for the `MatZpn` factorization and reduction surface: PLUQ, rank,
//! determinant, RREF, solve-left, and nullspace.
//!
//! Strategy: hand-computed small cases that pin individual algorithmic
//! steps, plus algebraic invariants on parametric random inputs:
//!
//! - `pluq`: round-trip `P·L·U·Q == A` when `A` is fully reducible by
//!   unit pivots; a surviving non-unit residue is dropped (see the
//!   `pluq` doc comment and the residue regression tests below).
//! - `rank`: matches the rank derived from `rref`.
//! - `rref`: idempotent (rref(rref(A)) == rref(A)).
//! - `det`: det(I) = 1, det(diag(d_i)) = ∏ d_i; the true ring value even
//!   for non-invertible matrices (a nonzero zero-divisor determinant).
//! - `solve_left`: `A · solve_left(A, B) == B` when solvable.
//! - `nullspace`: `A · nullspace(A) == 0`; rank-nullity holds.

use pwvla::zpn::MatZpn;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn random_matzpn(p: u32, n: u32, rows: usize, cols: usize, seed: u64) -> MatZpn {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let modulus = (p as u64).pow(n);
    MatZpn::from_fn(p, n, rows, cols, |_, _| rng.random_range(0..modulus))
}

fn zero_matrix(p: u32, n: u32, rows: usize, cols: usize) -> MatZpn {
    MatZpn::zeros(p, n, rows, cols)
}

fn identity_matrix(p: u32, n: u32, dim: usize) -> MatZpn {
    MatZpn::from_fn(p, n, dim, dim, |i, j| if i == j { 1 } else { 0 })
}

// ─── PLUQ ───────────────────────────────────────────────────────────

#[test]
fn pluq_identity_round_trips() {
    for &(p, n, dim) in &[(2u32, 4u32, 4usize), (3, 3, 5), (5, 2, 8), (7, 2, 6)] {
        let i = identity_matrix(p, n, dim);
        let (pp, l, u, qq, rank) = i.pluq();
        assert_eq!(rank, dim);
        let prod = pp.matmul(&l).matmul(&u).matmul(&qq);
        assert_eq!(prod, i, "PLUQ ≠ I at p={p} n={n} dim={dim}");
    }
}

#[test]
fn pluq_round_trips_on_random_full_rank() {
    // Build random invertible matrices: A = L·U where L unit lower and
    // U unit upper, both with random unit entries off-diagonal. Then
    // pluq(A) should give rank == dim and round-trip.
    for &(p, n, dim, seed) in &[
        (2u32, 4u32, 6usize, 1u64),
        (3, 3, 5, 2),
        (5, 2, 7, 3),
        (7, 2, 4, 4),
    ] {
        let modulus = (p as u64).pow(n);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        // Lower triangular with 1's on diagonal.
        let lo = MatZpn::from_fn(p, n, dim, dim, |i, j| {
            if i == j {
                1
            } else if i > j {
                rng.random_range(0..modulus)
            } else {
                0
            }
        });
        let up = MatZpn::from_fn(p, n, dim, dim, |i, j| {
            if i == j {
                1
            } else if i < j {
                rng.random_range(0..modulus)
            } else {
                0
            }
        });
        let a = lo.matmul(&up);
        let (pp, l, u, qq, rank) = a.pluq();
        assert_eq!(rank, dim);
        let prod = pp.matmul(&l).matmul(&u).matmul(&qq);
        assert_eq!(prod, a);
    }
}

#[test]
fn pluq_rectangular_round_trips() {
    // 4×6 and 6×4 cases. To guarantee a clean round-trip over a composite
    // modulus we build A = lo · up with unit pivots on the leading
    // diagonal, so every available pivot is a unit (full unit-rank) and no
    // non-unit residue can survive — see `pluq_nonunit_residue_dropped`
    // for why a *random* composite-modulus matrix may not round-trip.
    for &(p, n, m, k, seed) in &[(5u32, 2u32, 4usize, 6usize, 10u64), (3, 3, 6, 4, 11)] {
        let modulus = (p as u64).pow(n);
        let min = m.min(k);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        // lo: m×m unit lower triangular.
        let lo = MatZpn::from_fn(p, n, m, m, |i, j| {
            if i == j {
                1
            } else if i > j {
                rng.random_range(0..modulus)
            } else {
                0
            }
        });
        // up: m×k upper triangular with unit (1) pivots on the leading
        // diagonal positions (i < min).
        let up = MatZpn::from_fn(p, n, m, k, |i, j| {
            if i == j && i < min {
                1
            } else if i < j {
                rng.random_range(0..modulus)
            } else {
                0
            }
        });
        let a = lo.matmul(&up);
        let (pp, l, u, qq, rank) = a.pluq();
        assert_eq!(rank, min, "expected full unit-rank construction");
        let prod = pp.matmul(&l).matmul(&u).matmul(&qq);
        assert_eq!(prod, a);
    }
}

#[test]
fn pluq_singular_matrix_reports_low_rank() {
    // Z/25, build a matrix where pivot search finds only non-units.
    // Easiest: a matrix all of whose entries are multiples of 5.
    let a = MatZpn::from_fn(5, 2, 3, 3, |i, j| (5 * ((i * j + 1) as u64)) % 25);
    let r = a.rank();
    assert_eq!(r, 0, "all-multiples-of-p should have rank 0 over Z/p^n");
}

#[test]
fn pluq_nonunit_residue_dropped() {
    // Boundary of the reconstruction guarantee: over Z/p^n the unit-pivot
    // PLUQ captures only the unit-pivot-reducible part of A. When a genuine
    // non-unit remainder survives elimination it is NOT reconstructed.
    // (Locking this documents the limitation; a faithful canonical form
    // needs a Howell-form algorithm, not implemented at tier-1.)
    let cases: &[(u32, u32, usize, usize, &[u64])] = &[
        // [2] over Z/4: the lone entry is a non-unit, rank 0, product [0].
        (2, 2, 1, 1, &[2]),
        // diag(1, 2) over Z/4: the unit 1 is a pivot, the 2 is dropped.
        (2, 2, 2, 2, &[1, 0, 0, 2]),
    ];
    for &(p, n, rows, cols, data) in cases {
        let a = MatZpn::from_fn(p, n, rows, cols, |i, j| data[i * cols + j]);
        let (pp, l, u, qq, rank) = a.pluq();
        let prod = pp.matmul(&l).matmul(&u).matmul(&qq);
        assert!(
            rank < rows.min(cols),
            "expected deficient unit-rank, got {rank}"
        );
        assert_ne!(prod, a, "a surviving non-unit residue must not reconstruct");
    }
}

#[test]
fn pluq_reconstructs_when_residue_cancels() {
    // The complementary case: reconstruction can hold even at rank < dim
    // when the non-unit content is eliminated to zero by a unit pivot.
    // [[3,1],[0,3]] over Z/9 has unit-rank 1 (only the 1 is a unit) yet
    // round-trips exactly, because the 3s cancel during elimination.
    let a = MatZpn::from_fn(3, 2, 2, 2, |i, j| match (i, j) {
        (0, 0) => 3,
        (0, 1) => 1,
        (1, 1) => 3,
        _ => 0,
    });
    let (pp, l, u, qq, rank) = a.pluq();
    assert_eq!(rank, 1);
    let prod = pp.matmul(&l).matmul(&u).matmul(&qq);
    assert_eq!(prod, a);
}

// ─── rank ───────────────────────────────────────────────────────────

#[test]
fn rank_of_identity_is_dim() {
    for &(p, n, dim) in &[(2u32, 8u32, 4usize), (5, 2, 6), (7, 2, 8)] {
        assert_eq!(identity_matrix(p, n, dim).rank(), dim);
    }
}

#[test]
fn rank_of_zero_is_zero() {
    for &(p, n, m, k) in &[(2u32, 4u32, 5usize, 3usize), (5, 2, 4, 4), (7, 2, 6, 8)] {
        assert_eq!(zero_matrix(p, n, m, k).rank(), 0);
    }
}

#[test]
fn rank_agrees_with_rref() {
    // The rank reported by rref should match the rank from PLUQ.
    for &(p, n, m, k, seed) in &[
        (5u32, 2u32, 4usize, 5usize, 100u64),
        (3, 3, 6, 4, 101),
        (7, 2, 5, 7, 102),
    ] {
        let a = random_matzpn(p, n, m, k, seed);
        let r_pluq = a.rank();
        let (_, r_rref) = a.rref();
        assert_eq!(r_pluq, r_rref, "rank mismatch p={p} n={n} m={m} k={k}");
    }
}

// ─── det ────────────────────────────────────────────────────────────

#[test]
fn det_of_identity_is_one() {
    for &(p, n, dim) in &[(2u32, 4u32, 3usize), (5, 2, 5), (7, 2, 4)] {
        assert_eq!(identity_matrix(p, n, dim).det(), 1);
    }
}

#[test]
fn det_of_diagonal_is_product() {
    // diag(2, 3, 7) over Z/121 — all units, det = 2·3·7 = 42.
    let d = MatZpn::from_fn(11, 2, 3, 3, |i, j| {
        if i != j {
            0
        } else {
            match i {
                0 => 2,
                1 => 3,
                2 => 7,
                _ => unreachable!(),
            }
        }
    });
    assert_eq!(d.det(), 2 * 3 * 7);
}

#[test]
fn det_of_zero_matrix_is_zero() {
    // All-zeros 3×3 over Z/25 — determinant is genuinely 0.
    let a = zero_matrix(5, 2, 3, 3);
    assert_eq!(a.det(), 0);
}

#[test]
fn det_of_swap_is_minus_one() {
    // Swap rows of I_2 over Z/25 → det = -1 = 24.
    let s = MatZpn::from_fn(5, 2, 2, 2, |i, j| match (i, j) {
        (0, 1) => 1,
        (1, 0) => 1,
        _ => 0,
    });
    assert_eq!(s.det(), 25 - 1);
}

#[test]
fn det_hand_case_z25() {
    // A = [[3, 2], [1, 5]] over Z/25, det = 3·5 − 2·1 = 13.
    let a = MatZpn::from_fn(5, 2, 2, 2, |i, j| match (i, j) {
        (0, 0) => 3,
        (0, 1) => 2,
        (1, 0) => 1,
        (1, 1) => 5,
        _ => 0,
    });
    assert_eq!(a.det(), 13);
}

#[test]
fn det_hand_case_singular_z25() {
    // A = [[3, 2], [1, 4]], det = 12 - 2 = 10 = 5·2. gcd(10, 25) = 5, so A
    // is non-invertible over Z/25 — but its determinant is the *nonzero*
    // zero-divisor 10, not 0. Unit-pivot elimination stops at rank 1 (the
    // second pivot, 20, is not coprime to 5); det() returns the true value
    // via the division-free Berkowitz fallback.
    let a = MatZpn::from_fn(5, 2, 2, 2, |i, j| match (i, j) {
        (0, 0) => 3,
        (0, 1) => 2,
        (1, 0) => 1,
        (1, 1) => 4,
        _ => 0,
    });
    assert_eq!(a.det(), 10);
}

#[test]
fn det_nonunit_is_true_ring_value() {
    // Non-invertible matrices whose determinant is a nonzero zero-divisor:
    // det() must return that value, not a 0 "singular" sentinel.
    // [2] over Z/4 → det 2.
    let a = MatZpn::from_fn(2, 2, 1, 1, |_, _| 2);
    assert_eq!(a.det(), 2);
    // diag(1, 2) over Z/4 → det 1·2 = 2.
    let a = MatZpn::from_fn(2, 2, 2, 2, |i, j| if i == j { [1u64, 2][i] } else { 0 });
    assert_eq!(a.det(), 2);
    // diag(2, 2) over Z/4 → det 2·2 = 4 ≡ 0: a genuinely-zero determinant.
    let a = MatZpn::from_fn(2, 2, 2, 2, |i, j| if i == j { 2 } else { 0 });
    assert_eq!(a.det(), 0);
    // A non-diagonal singular case over Z/9: [[3,1],[3,1]] has det
    // 3·1 - 1·3 = 0 (repeated row), confirming the zero case too.
    let a = MatZpn::from_fn(3, 2, 2, 2, |i, j| match (i, j) {
        (_, 0) => 3,
        (_, 1) => 1,
        _ => 0,
    });
    assert_eq!(a.det(), 0);
}

#[test]
fn det_berkowitz_agrees_with_pluq_on_units() {
    // The Berkowitz determinant must agree with the fast PLUQ path on
    // invertible matrices (built as L·U with unit pivots, full unit-rank).
    for &(p, n, dim, seed) in &[(5u32, 2u32, 4usize, 7u64), (3, 3, 5, 8), (7, 2, 3, 9)] {
        let modulus = (p as u64).pow(n);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let lo = MatZpn::from_fn(p, n, dim, dim, |i, j| match i.cmp(&j) {
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Greater => rng.random_range(0..modulus),
            std::cmp::Ordering::Less => 0,
        });
        let up = MatZpn::from_fn(p, n, dim, dim, |i, j| match i.cmp(&j) {
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Less => rng.random_range(0..modulus),
            std::cmp::Ordering::Greater => 0,
        });
        let a = lo.matmul(&up);
        // det(L·U) = det L · det U = 1, regardless of path.
        assert_eq!(a.det(), 1, "unit-triangular product has det 1");
    }
}

// ─── rref ───────────────────────────────────────────────────────────

#[test]
fn rref_is_idempotent() {
    for &(p, n, m, k, seed) in &[
        (5u32, 2u32, 4usize, 5usize, 200u64),
        (3, 3, 6, 4, 201),
        (7, 2, 5, 5, 202),
    ] {
        let a = random_matzpn(p, n, m, k, seed);
        let (r1, _) = a.rref();
        let (r2, _) = r1.rref();
        assert_eq!(r1, r2, "rref not idempotent at p={p} n={n}");
    }
}

#[test]
fn rref_identity_is_identity() {
    for &(p, n, dim) in &[(2u32, 4u32, 3usize), (5, 2, 4), (7, 2, 5)] {
        let i = identity_matrix(p, n, dim);
        let (r, rank) = i.rref();
        assert_eq!(r, i);
        assert_eq!(rank, dim);
    }
}

// ─── solve_left ─────────────────────────────────────────────────────

#[test]
fn solve_left_round_trip_full_rank() {
    // Build random unit-pivot lower×upper, multiply to get a regular A,
    // pick random X, set B = A·X, then check solve_left(A, B) round-trips.
    for &(p, n, dim, nrhs, seed) in &[
        (5u32, 2u32, 4usize, 3usize, 300u64),
        (3, 3, 5, 2, 301),
        (7, 2, 6, 4, 302),
    ] {
        let modulus = (p as u64).pow(n);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let lo = MatZpn::from_fn(p, n, dim, dim, |i, j| {
            if i == j {
                1
            } else if i > j {
                rng.random_range(0..modulus)
            } else {
                0
            }
        });
        let up = MatZpn::from_fn(p, n, dim, dim, |i, j| {
            if i == j {
                1
            } else if i < j {
                rng.random_range(0..modulus)
            } else {
                0
            }
        });
        let a = lo.matmul(&up);
        let x = random_matzpn(p, n, dim, nrhs, seed + 1000);
        let b = a.matmul(&x);
        let x_solved = MatZpn::solve_left(&a, &b).expect("full-rank should be solvable");
        let b_check = a.matmul(&x_solved);
        assert_eq!(b_check, b);
    }
}

#[test]
fn solve_left_inconsistent_returns_none() {
    // Build a 3×2 matrix A with rank 2 over Z/25, and a B that's not
    // in the column space. solve_left should return None.
    let a = MatZpn::from_fn(5, 2, 3, 2, |i, j| match (i, j) {
        (0, 0) => 1,
        (1, 1) => 1,
        (2, 0) => 0,
        (2, 1) => 0,
        _ => 0,
    });
    // B has a nonzero in row 2 where A's row is zero → inconsistent.
    let b = MatZpn::from_fn(5, 2, 3, 1, |i, _| if i == 2 { 1 } else { 0 });
    assert!(MatZpn::solve_left(&a, &b).is_none());
}

// ─── nullspace ──────────────────────────────────────────────────────

#[test]
fn nullspace_times_a_is_zero() {
    for &(p, n, m, k, seed) in &[
        (5u32, 2u32, 3usize, 5usize, 400u64),
        (3, 3, 4, 6, 401),
        (7, 2, 5, 7, 402),
    ] {
        let a = random_matzpn(p, n, m, k, seed);
        let nullspace = a.nullspace();
        let ak = a.matmul(&nullspace);
        for r in 0..ak.rows {
            for c in 0..ak.cols {
                assert_eq!(ak.get(r, c), 0, "A·nullspace ≠ 0 at p={p} n={n}");
            }
        }
    }
}

#[test]
fn nullspace_rank_nullity_holds() {
    for &(p, n, m, k, seed) in &[
        (5u32, 2u32, 3usize, 5usize, 500u64),
        (3, 3, 4, 7, 501),
        (7, 2, 6, 4, 502),
    ] {
        let a = random_matzpn(p, n, m, k, seed);
        let r = a.rank();
        let nullity = a.nullspace().cols;
        assert_eq!(r + nullity, k, "rank-nullity broken at p={p} n={n}");
    }
}

#[test]
fn nullspace_of_full_column_rank_is_empty() {
    // Tall matrix with linearly independent columns.
    let a = identity_matrix(5, 2, 6);
    let null = a.nullspace();
    assert_eq!(null.cols, 0);
    assert_eq!(null.rows, 6);
}
