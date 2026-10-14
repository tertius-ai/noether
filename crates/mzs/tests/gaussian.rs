// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Tier-1B Gaussian-elimination surface on `Mzs`: rref, rank,
//! nullspace, solve_left. Hand-computed small cases pin algorithmic
//! correctness; algebraic invariants (rank-nullity, `A·nullspace=0`,
//! `A·X=B` round-trip, idempotence) cover the random-input space; a
//! local schoolbook reference catches regressions.

use mzs::Mzs;
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};

fn random_mzs(rows: usize, cols: usize, density: f64, seed: u64) -> Mzs {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut m = Mzs::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            if rng.random::<f64>() < density {
                m.set(i, j, 1);
            }
        }
    }
    m
}

fn identity(n: usize) -> Mzs {
    let mut m = Mzs::new(n, n);
    for i in 0..n {
        m.set(i, i, 1);
    }
    m
}

/// Dense schoolbook rank/rref over F_2 — used as an oracle. Returns
/// (rref, rank).
fn schoolbook_rref(a: &Mzs) -> (Vec<Vec<u8>>, usize) {
    let m = a.rows();
    let n = a.cols();
    let mut work: Vec<Vec<u8>> = (0..m)
        .map(|i| (0..n).map(|j| a.get(i, j)).collect())
        .collect();
    let mut rank = 0usize;
    let mut col = 0usize;
    while rank < m && col < n {
        let mut pivot = None;
        for r in rank..m {
            if work[r][col] == 1 {
                pivot = Some(r);
                break;
            }
        }
        let pi = match pivot {
            Some(r) => r,
            None => {
                col += 1;
                continue;
            }
        };
        work.swap(pi, rank);
        for i in 0..m {
            if i == rank {
                continue;
            }
            if work[i][col] == 1 {
                for j in col..n {
                    work[i][j] ^= work[rank][j];
                }
            }
        }
        rank += 1;
        col += 1;
    }
    (work, rank)
}

// ─── rref / rank ────────────────────────────────────────────────────

#[test]
fn rref_of_identity_is_identity() {
    for n in [1usize, 2, 5, 10] {
        let i = identity(n);
        let (rref, rank) = i.rref();
        assert_eq!(rank, n);
        assert_eq!(rref, i);
    }
}

#[test]
fn rref_of_zero_is_zero() {
    for &(m, n) in &[(3usize, 5usize), (5, 3), (0, 0), (4, 0), (0, 4)] {
        let z = Mzs::new(m, n);
        let (rref, rank) = z.rref();
        assert_eq!(rank, 0);
        assert_eq!(rref, z);
    }
}

#[test]
fn rref_hand_2x3() {
    // A = [[1 1 0], [0 1 1]]
    // After RREF: row 0 += row 1 → [1 0 1], row 1 unchanged.
    let a = Mzs::from_triples(2, 3, &[(0, 0), (0, 1), (1, 1), (1, 2)]);
    let (rref, rank) = a.rref();
    assert_eq!(rank, 2);
    // Expected: [[1 0 1], [0 1 1]]
    let expected = Mzs::from_triples(2, 3, &[(0, 0), (0, 2), (1, 1), (1, 2)]);
    assert_eq!(rref, expected);
}

#[test]
fn rref_is_idempotent() {
    for &(m, n, density, seed) in &[
        (5usize, 7usize, 0.3, 100u64),
        (8, 4, 0.5, 101),
        (3, 11, 0.2, 102),
    ] {
        let a = random_mzs(m, n, density, seed);
        let (r1, k1) = a.rref();
        let (r2, k2) = r1.rref();
        assert_eq!(k1, k2);
        assert_eq!(
            r1, r2,
            "rref not idempotent at m={m} n={n} density={density}"
        );
    }
}

#[test]
fn rref_matches_schoolbook() {
    for &(m, n, density, seed) in &[
        (5usize, 7usize, 0.2, 200u64),
        (6, 6, 0.4, 201),
        (3, 11, 0.1, 202),
        (8, 4, 0.5, 203),
        (1, 20, 0.3, 204),
        (20, 1, 0.3, 205),
    ] {
        let a = random_mzs(m, n, density, seed);
        let (rref_fast, rank_fast) = a.rref();
        let (rref_ref, rank_ref) = schoolbook_rref(&a);
        assert_eq!(rank_fast, rank_ref, "rank mismatch m={m} n={n}");
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    rref_fast.get(i, j),
                    rref_ref[i][j],
                    "rref mismatch at ({i}, {j}) m={m} n={n} density={density}"
                );
            }
        }
    }
}

#[test]
fn rank_of_random_matches_schoolbook() {
    for &(m, n, density, seed) in &[
        (5usize, 7usize, 0.2, 300u64),
        (10, 10, 0.5, 301),
        (15, 7, 0.3, 302),
    ] {
        let a = random_mzs(m, n, density, seed);
        let r_fast = a.rank();
        let (_, r_ref) = schoolbook_rref(&a);
        assert_eq!(r_fast, r_ref);
    }
}

#[test]
fn rank_of_rank_one_outer_product() {
    // u · v^T where u, v ∈ F_2^n nonzero → rank 1 (assuming u ≠ 0
    // and v ≠ 0).
    let u = Mzs::from_triples(4, 1, &[(0, 0), (2, 0), (3, 0)]); // [1 0 1 1]^T
    let v = Mzs::from_triples(1, 5, &[(0, 1), (0, 3)]);
    let outer = u.matmul(&v);
    assert_eq!(outer.rank(), 1);
}

// ─── nullspace ──────────────────────────────────────────────────────

#[test]
fn nullspace_of_identity_is_empty() {
    let i = identity(5);
    let k = i.nullspace();
    assert_eq!(k.rows(), 5);
    assert_eq!(k.cols(), 0);
}

#[test]
fn nullspace_of_zero_is_full() {
    // 0 matrix m × n: every vector is in the kernel; basis is I_n.
    let z = Mzs::new(3, 5);
    let k = z.nullspace();
    assert_eq!(k.rows(), 5);
    assert_eq!(k.cols(), 5);
    let expected = identity(5);
    assert_eq!(k, expected);
}

#[test]
fn nullspace_times_a_is_zero() {
    for &(m, n, density, seed) in &[
        (3usize, 5usize, 0.3, 400u64),
        (4, 8, 0.25, 401),
        (5, 5, 0.4, 402),
        (2, 7, 0.5, 403),
    ] {
        let a = random_mzs(m, n, density, seed);
        let k = a.nullspace();
        let ak = a.matmul(&k);
        assert_eq!(
            ak.nnz(),
            0,
            "A·nullspace(A) ≠ 0 at m={m} n={n} density={density}"
        );
    }
}

#[test]
fn nullspace_rank_nullity_holds() {
    for &(m, n, density, seed) in &[
        (3usize, 5usize, 0.3, 500u64),
        (8, 4, 0.5, 501),
        (5, 7, 0.2, 502),
        (10, 10, 0.4, 503),
    ] {
        let a = random_mzs(m, n, density, seed);
        let r = a.rank();
        let k = a.nullspace();
        assert_eq!(r + k.cols(), n, "rank-nullity at m={m} n={n}");
    }
}

#[test]
fn nullspace_hand_case() {
    // A = [[1 0 1], [0 1 1]] → rank 2, nullspace spanned by [1, 1, 1]^T.
    let a = Mzs::from_triples(2, 3, &[(0, 0), (0, 2), (1, 1), (1, 2)]);
    let k = a.nullspace();
    assert_eq!(k.rows(), 3);
    assert_eq!(k.cols(), 1);
    for i in 0..3 {
        assert_eq!(k.get(i, 0), 1);
    }
}

// ─── solve_left ─────────────────────────────────────────────────────

#[test]
fn solve_left_identity() {
    let i = identity(4);
    let b = Mzs::from_triples(4, 2, &[(0, 0), (1, 1), (2, 0), (3, 1)]);
    let x = Mzs::solve_left(&i, &b).expect("identity always solvable");
    assert_eq!(x, b);
}

#[test]
fn solve_left_full_rank_round_trip() {
    for &(n, n_rhs, seed) in &[(4usize, 3usize, 600u64), (6, 1, 601), (8, 5, 602)] {
        // Build a guaranteed-full-rank A: I + strictly-lower random.
        let mut a = identity(n);
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        for i in 0..n {
            for j in 0..i {
                if rng.random::<f64>() < 0.4 {
                    a.set(i, j, 1);
                }
            }
        }
        // Random B; A is invertible so A·X=B has a unique solution.
        let b = random_mzs(n, n_rhs, 0.4, seed + 1000);
        let x = Mzs::solve_left(&a, &b).expect("full-rank should be solvable");
        let ax = a.matmul(&x);
        assert_eq!(ax, b, "A·X != B at n={n} n_rhs={n_rhs}");
    }
}

#[test]
fn solve_left_inconsistent_returns_none() {
    // A = [[1 0], [0 0]] (rank 1), B = [[0], [1]]. The row "0 = 1"
    // makes the system inconsistent.
    let a = Mzs::from_triples(2, 2, &[(0, 0)]);
    let b = Mzs::from_triples(2, 1, &[(1, 0)]);
    assert!(Mzs::solve_left(&a, &b).is_none());
}

#[test]
fn solve_left_rank_deficient_consistent() {
    // A = [[1 0], [0 0]], B = [[1], [0]]. Solvable: x = [[1], [0]]
    // (free variable x[1] = 0 by the "natural solution" convention).
    let a = Mzs::from_triples(2, 2, &[(0, 0)]);
    let b = Mzs::from_triples(2, 1, &[(0, 0)]);
    let x = Mzs::solve_left(&a, &b).expect("consistent");
    assert_eq!(x.get(0, 0), 1);
    // x[1, 0] may be 0 or 1 — we don't pin the free-variable
    // assignment beyond the contract. But A·X must equal B.
    let ax = a.matmul(&x);
    assert_eq!(ax, b);
}

#[test]
fn solve_left_against_nullspace_basis() {
    // A · X = 0 system: the basis vectors of nullspace are solutions.
    let a = random_mzs(4, 6, 0.3, 700);
    let k = a.nullspace();
    let zero = Mzs::new(4, k.cols());
    let x = Mzs::solve_left(&a, &zero).expect("zero RHS always solvable");
    // x might be the trivial zero solution since that's the natural one;
    // check it works.
    let ax = a.matmul(&x);
    assert_eq!(ax, zero);
}
