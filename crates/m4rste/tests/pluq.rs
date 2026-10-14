// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Standalone correctness tests for GF(2^e) PLUQ decomposition.
//!
//! The defining property is `P · L · U · Q == A`. Plus structure: P, Q
//! are permutation matrices; L is unit lower triangular in its first
//! rank columns (diagonal exactly 1, matching `m4rst::Mzd::pluq`'s
//! contract — U carries the real pivot values instead, see
//! `src/pluq.rs`'s module doc); U is upper triangular in its first
//! rank rows.

use m4rste::{Gf2e, Mzed};

fn gf4() -> Gf2e {
    Gf2e::new(0b111)
}

fn gf256_aes() -> Gf2e {
    Gf2e::new(0x11B)
}

/// `is_permutation_matrix(m)`: exactly one nonzero (and it must be the
/// field's `1`) per row and per column.
fn is_permutation_matrix(m: &Mzed) -> bool {
    if m.rows() != m.cols() {
        return false;
    }
    let n = m.rows();
    for i in 0..n {
        let mut c = 0;
        for j in 0..n {
            let v = m.get(i, j);
            if v != 0 {
                if v != 1 {
                    return false;
                }
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
            if m.get(i, j) != 0 {
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
fn check_l_structure(l: &Mzed, r: usize) {
    for i in 0..l.rows() {
        for j in 0..l.cols() {
            if j >= r {
                assert_eq!(l.get(i, j), 0, "L[{i}, {j}] should be 0 (column past rank)");
            } else if j > i {
                assert_eq!(l.get(i, j), 0, "L[{i}, {j}] should be 0 (above diagonal)");
            } else if j == i && i < r {
                assert_eq!(l.get(i, j), 1, "L[{i}, {j}] should be 1 (unit diagonal)");
            }
        }
    }
}

/// Check U is upper triangular in its first `r` rows (diagonal may be
/// any nonzero field value — see the module's doc comment).
fn check_u_structure(u: &Mzed, r: usize) {
    for i in 0..u.rows() {
        for j in 0..u.cols() {
            if i >= r {
                assert_eq!(u.get(i, j), 0, "U[{i}, {j}] should be 0 (row past rank)");
            } else if j < i {
                assert_eq!(u.get(i, j), 0, "U[{i}, {j}] should be 0 (below diagonal)");
            } else if j == i {
                assert_ne!(u.get(i, j), 0, "U[{i}, {j}] (pivot) should be nonzero");
            }
        }
    }
}

fn assert_pluq_eq_a(a: &Mzed) {
    let (p, l, u, q, r) = a.pluq();
    assert!(is_permutation_matrix(&p), "P is not a permutation matrix");
    assert!(is_permutation_matrix(&q), "Q is not a permutation matrix");
    assert_eq!(l.rows(), a.rows());
    assert_eq!(l.cols(), a.rows().min(a.cols()));
    assert_eq!(u.rows(), a.rows().min(a.cols()));
    assert_eq!(u.cols(), a.cols());
    check_l_structure(&l, r);
    check_u_structure(&u, r);
    let lu = Mzed::matmul(&l, &u);
    let plu = Mzed::matmul(&p, &lu);
    let pluq = Mzed::matmul(&plu, &q);
    assert_eq!(pluq.to_u16_buffer(), a.to_u16_buffer(), "P · L · U · Q ≠ A");
    assert_eq!(r, a.rank(), "PLUQ rank disagrees with echelon rank");
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
fn pluq_identity() {
    for &f in &[gf4(), gf256_aes()] {
        for &n in &[1usize, 5, 17, 40] {
            let i = Mzed::identity(f, n);
            assert_pluq_eq_a(&i);
        }
    }
}

#[test]
fn pluq_zero() {
    for &f in &[gf4(), gf256_aes()] {
        for &(r, c) in &[(5usize, 7usize), (8, 20), (13, 33), (20, 20)] {
            let z = Mzed::zeros(f, r, c);
            assert_pluq_eq_a(&z);
        }
    }
}

#[test]
fn pluq_random_full_rank_likely() {
    for &f in &[gf4(), gf256_aes()] {
        for &(r, c, seed) in &[
            (1usize, 1usize, 1u64),
            (4, 4, 2),
            (8, 8, 3),
            (16, 16, 4),
            (17, 17, 5),
            (32, 32, 6),
        ] {
            let a = random(f, r, c, seed ^ f.minpoly());
            assert_pluq_eq_a(&a);
        }
    }
}

#[test]
fn pluq_rectangular() {
    for &f in &[gf4(), gf256_aes()] {
        for &(r, c, seed) in &[
            (5usize, 12usize, 11u64),
            (12, 5, 12),
            (8, 21, 13),
            (21, 8, 14),
        ] {
            let a = random(f, r, c, seed ^ f.minpoly());
            assert_pluq_eq_a(&a);
        }
    }
}

#[test]
fn pluq_rank_deficient() {
    for &f in &[gf4(), gf256_aes()] {
        // 4×8 over the field: row 2 = 3 * row 0 (a nontrivial field
        // scalar, not just a bit duplicate — exercises the elimination
        // factor being a genuine field value), row 3 is independent —
        // rank should be 3, not 4.
        let mut a = Mzed::zeros(f, 4, 8);
        for j in 0..4 {
            a.set(0, j, (j as u16 + 1) & ((1u16 << f.degree()) - 1).max(1));
        }
        for j in 4..8 {
            a.set(1, j, (j as u16) & ((1u16 << f.degree()) - 1).max(1));
        }
        for j in 0..4 {
            a.set(2, j, f.mul(3 & ((1u16 << f.degree()) - 1), a.get(0, j)));
        }
        for j in 0..8 {
            a.set(3, j, ((j as u16) + 7) & ((1u16 << f.degree()) - 1).max(1));
        }
        let (_p, _l, _u, _q, r) = a.pluq();
        assert_eq!(
            r, 3,
            "expected rank 3 (row 2 is a scalar multiple of row 0)"
        );
        assert_pluq_eq_a(&a);
    }
}

#[test]
fn pluq_nontrivial_pivot_values_reconstruct() {
    // Pivot values other than the field's 1 must reconstruct exactly.
    let f = gf256_aes();
    let a = Mzed::from_u16_buffer(
        f,
        4,
        4,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 200, 12, 13, 14, 15, 250],
    );
    assert_pluq_eq_a(&a);
}

#[test]
fn pluq_into_matches_pluq() {
    for &f in &[gf4(), gf256_aes()] {
        for &(r, c, seed) in &[(1usize, 1usize, 1u64), (5, 12, 2), (17, 17, 3), (12, 5, 4)] {
            let a = random(f, r, c, seed ^ f.minpoly());
            let (want_p, want_l, want_u, want_q, want_rank) = a.pluq();

            let k = r.min(c);
            let mut p = Mzed::zeros(f, r, r);
            let mut l = Mzed::zeros(f, r, k);
            let mut u = Mzed::zeros(f, k, c);
            let mut q = Mzed::zeros(f, c, c);
            let rank = a.pluq_into(&mut p, &mut l, &mut u, &mut q);

            assert_eq!(rank, want_rank);
            assert_eq!(p.to_u16_buffer(), want_p.to_u16_buffer());
            assert_eq!(l.to_u16_buffer(), want_l.to_u16_buffer());
            assert_eq!(u.to_u16_buffer(), want_u.to_u16_buffer());
            assert_eq!(q.to_u16_buffer(), want_q.to_u16_buffer());
        }
    }
}

#[test]
fn pluq_into_on_reused_buffers_does_not_accumulate_stale_data() {
    // pluq_into must fully overwrite p/l/u/q, not leave any residue
    // from a prior unrelated call reusing the same allocations.
    let f = gf256_aes();
    let a1 = random(f, 4, 4, 0xA1);
    let a2 = random(f, 4, 4, 0xA2);

    let mut p = Mzed::zeros(f, 4, 4);
    let mut l = Mzed::zeros(f, 4, 4);
    let mut u = Mzed::zeros(f, 4, 4);
    let mut q = Mzed::zeros(f, 4, 4);

    a1.pluq_into(&mut p, &mut l, &mut u, &mut q); // first call: populates p/l/u/q
    let rank2 = a2.pluq_into(&mut p, &mut l, &mut u, &mut q); // reuse: must fully overwrite

    let (want_p, want_l, want_u, want_q, want_rank) = a2.pluq();
    assert_eq!(rank2, want_rank);
    assert_eq!(p.to_u16_buffer(), want_p.to_u16_buffer());
    assert_eq!(l.to_u16_buffer(), want_l.to_u16_buffer());
    assert_eq!(u.to_u16_buffer(), want_u.to_u16_buffer());
    assert_eq!(q.to_u16_buffer(), want_q.to_u16_buffer());
}
