// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Standalone correctness tests for `solve_left` / `inverse` /
//! `nullspace` over GF(2^e).

use m4rste::{Gf2e, Mzed};

fn gf4() -> Gf2e {
    Gf2e::new(0b111)
}

fn gf256_aes() -> Gf2e {
    Gf2e::new(0x11B)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn random(f: Gf2e, rows: usize, cols: usize, seed: u64) -> Mzed {
    let mask: u16 = if f.degree() >= 16 {
        u16::MAX
    } else {
        (1u16 << f.degree()) - 1
    };
    let mut rng = Rng(seed | 1);
    let buf: Vec<u16> = (0..rows * cols)
        .map(|_| (rng.next() as u16) & mask)
        .collect();
    Mzed::from_u16_buffer(f, rows, cols, &buf)
}

#[test]
fn solve_identity_returns_b() {
    for &f in &[gf4(), gf256_aes()] {
        for &(n, nrhs, seed) in &[(1usize, 1usize, 1u64), (8, 3, 2), (17, 5, 3), (32, 4, 4)] {
            let a = Mzed::identity(f, n);
            let b = random(f, n, nrhs, seed ^ f.minpoly());
            let x = Mzed::solve_left(&a, &b).expect("identity is always solvable");
            assert_eq!(
                x.to_u16_buffer(),
                b.to_u16_buffer(),
                "I · X = B should give X = B"
            );
        }
    }
}

#[test]
fn solve_round_trip_on_random_square() {
    // For a random square A (full rank likely), solve(A, A·y) has a
    // solution X with A·X == B (X might differ from y if A has a
    // nontrivial kernel, but at full rank X == y is forced).
    for &f in &[gf4(), gf256_aes()] {
        for &(n, nrhs, seed) in &[(8usize, 3usize, 1u64), (17, 5, 2), (32, 4, 3)] {
            let a = random(f, n, n, seed ^ f.minpoly());
            if a.rank() < n {
                continue; // unlikely; skip if it happens
            }
            let y = random(f, n, nrhs, seed.wrapping_mul(31) ^ f.minpoly());
            let b = Mzed::matmul(&a, &y);
            let x = Mzed::solve_left(&a, &b).expect("full-rank A should be solvable");
            let ax = Mzed::matmul(&a, &x);
            assert_eq!(ax.to_u16_buffer(), b.to_u16_buffer(), "A · X ≠ B");
        }
    }
}

#[test]
fn solve_consistent_rank_deficient() {
    for &f in &[gf4(), gf256_aes()] {
        // 4×4 A with rank 2: row 2 = 3 · row 0 (a genuine field scalar
        // multiple, not just a bit duplicate), row 3 = row 0.
        let three = 3u16 & ((1u16 << f.degree()) - 1).max(1);
        let mut a = Mzed::zeros(f, 4, 4);
        a.set(0, 0, 1);
        a.set(0, 1, 2 & ((1u16 << f.degree()) - 1).max(1));
        a.set(1, 2, 1);
        a.set(1, 3, 1);
        for j in 0..4 {
            a.set(2, j, f.mul(three, a.get(0, j)));
        }
        a.set(3, 0, 1);
        // Pick X = [1, 0, 1, 0]^T. Compute B = A · X.
        let mut x_true = Mzed::zeros(f, 4, 1);
        x_true.set(0, 0, 1);
        x_true.set(2, 0, 1);
        let b = Mzed::matmul(&a, &x_true);
        let x = Mzed::solve_left(&a, &b).expect("consistent system should solve");
        let ax = Mzed::matmul(&a, &x);
        assert_eq!(
            ax.to_u16_buffer(),
            b.to_u16_buffer(),
            "A · X ≠ B for rank-deficient consistent system"
        );
    }
}

#[test]
fn solve_returns_none_on_inconsistent() {
    for &f in &[gf4(), gf256_aes()] {
        // A · X = B where B is outside col-space of A: row 2 of A is
        // all-zero, but B[2] = 1.
        let mut a = Mzed::zeros(f, 3, 2);
        a.set(0, 0, 1);
        a.set(1, 1, 1);
        let mut b = Mzed::zeros(f, 3, 1);
        b.set(2, 0, 1);
        let x = Mzed::solve_left(&a, &b);
        assert!(x.is_none(), "inconsistent system should return None");
    }
}

#[test]
fn nullspace_of_identity_is_empty() {
    for &f in &[gf4(), gf256_aes()] {
        for &n in &[1usize, 8, 17, 32] {
            let i = Mzed::identity(f, n);
            let k = Mzed::nullspace(&i);
            assert_eq!(k.rows(), n);
            assert_eq!(k.cols(), 0, "identity has trivial kernel");
        }
    }
}

#[test]
fn nullspace_of_zero_is_full() {
    for &f in &[gf4(), gf256_aes()] {
        let z = Mzed::zeros(f, 5, 7);
        let k = Mzed::nullspace(&z);
        assert_eq!(k.rows(), 7);
        assert_eq!(k.cols(), 7, "zero matrix kernel == whole space");
        assert_eq!(k.rank(), 7);
    }
}

#[test]
fn nullspace_columns_are_in_kernel() {
    for &f in &[gf4(), gf256_aes()] {
        for &(r, c, seed) in &[(5usize, 8usize, 1u64), (8, 12, 2), (17, 33, 3)] {
            let a = random(f, r, c, seed ^ f.minpoly());
            let k = Mzed::nullspace(&a);
            if k.cols() == 0 {
                continue;
            }
            let ak = Mzed::matmul(&a, &k);
            for v in ak.to_u16_buffer() {
                assert_eq!(v, 0);
            }
        }
    }
}

#[test]
fn inverse_of_identity_is_identity() {
    for &f in &[gf4(), gf256_aes()] {
        for &n in &[1usize, 8, 17, 32] {
            let i = Mzed::identity(f, n);
            let inv = Mzed::inverse(&i).expect("identity is invertible");
            assert_eq!(inv.to_u16_buffer(), i.to_u16_buffer());
        }
    }
}

#[test]
fn inverse_singular_returns_none() {
    for &f in &[gf4(), gf256_aes()] {
        // 4×4, rank 3: row 3 = row 0 + row 1 + row 2.
        let mut a = Mzed::zeros(f, 4, 4);
        a.set(0, 0, 1);
        a.set(0, 1, 1);
        a.set(1, 2, 1);
        a.set(1, 3, 1);
        a.set(2, 0, 1);
        for j in 0..4 {
            let v = a.get(0, j) ^ a.get(1, j) ^ a.get(2, j);
            a.set(3, j, v);
        }
        assert!(
            Mzed::inverse(&a).is_none(),
            "singular matrix should return None"
        );
    }
}

#[test]
fn inverse_round_trip() {
    for &f in &[gf4(), gf256_aes()] {
        for &(n, seed) in &[(8usize, 1u64), (17, 2), (32, 3)] {
            let a = random(f, n, n, seed ^ f.minpoly());
            let inv = match Mzed::inverse(&a) {
                Some(x) => x,
                None => continue, // rank-deficient; skip
            };
            let i = Mzed::identity(f, n);
            let ai = Mzed::matmul(&a, &inv);
            assert_eq!(
                ai.to_u16_buffer(),
                i.to_u16_buffer(),
                "A · A⁻¹ ≠ I at n={n}"
            );
            let ia = Mzed::matmul(&inv, &a);
            assert_eq!(
                ia.to_u16_buffer(),
                i.to_u16_buffer(),
                "A⁻¹ · A ≠ I at n={n}"
            );
        }
    }
}

#[test]
fn rank_nullity_theorem() {
    for &f in &[gf4(), gf256_aes()] {
        for &(r, c, seed) in &[(5usize, 8usize, 1u64), (8, 12, 2), (17, 33, 3)] {
            let a = random(f, r, c, seed ^ f.minpoly());
            let nullity = Mzed::nullspace(&a).cols();
            assert_eq!(
                a.rank() + nullity,
                c,
                "rank-nullity violated: rank={}, nullity={nullity}, cols={c}",
                a.rank()
            );
        }
    }
}
