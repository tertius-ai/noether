// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Adversarial correctness sweep — every public API × every edge
//! shape. Goal: find any input that makes the library panic, return
//! the wrong answer, or behave inconsistently with its own algebraic
//! invariants.
//!
//! Cross-validation uses:
//! 1. **Schoolbook matmul** as the oracle for `Mzd::matmul`.
//! 2. **Algebraic identities** for `transpose`, `pluq`, `solve_left`,
//!    `nullspace`, `inverse`, `rank`, `echelonize` (rank-nullity,
//!    `P · L · U · Q == A`, `A · solve(A, B) == B`, etc.).
//! 3. The standalone libm4ri parity tests (in `parity_libm4ri.rs`)
//!    cover bit-equality of canonical-output ops.

use m4rst::Mzd;

/// All the shape edge cases we want to exercise. Combinations of
/// (rows, cols) covering:
/// - empty: 0×0, 0×N, N×0
/// - single: 1×1, 1×N, N×1
/// - awkward column counts: cols ∈ {1, 63, 64, 65, 127, 128, 129}
/// - thin/tall: rows or cols >> the other
/// - rectangular sizes through dispatch thresholds
const EDGE_SHAPES: &[(usize, usize)] = &[
    (0, 0),
    (0, 1),
    (1, 0),
    (0, 64),
    (64, 0),
    (0, 65),
    (65, 0),
    (1, 1),
    (1, 63),
    (1, 64),
    (1, 65),
    (1, 127),
    (1, 128),
    (1, 129),
    (1, 1000),
    (63, 1),
    (64, 1),
    (65, 1),
    (127, 1),
    (128, 1),
    (129, 1),
    (1000, 1),
    (2, 2),
    (3, 5),
    (5, 3),
    (64, 64),
    (64, 65),
    (65, 64),
    (65, 65),
    (127, 127),
    (128, 128),
    (129, 129),
    (256, 1),
    (1, 256),
    (1, 4097),
    (4097, 1),
];

/// Schoolbook matmul as oracle.
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

// ---- Construction & IO --------------------------------------------------

/// `Mzd::new(r, c)` returns a zero matrix at every edge shape.
#[test]
fn new_at_edge_shapes_is_all_zero() {
    for &(r, c) in EDGE_SHAPES {
        let m = Mzd::new(r, c);
        assert_eq!(m.rows(), r);
        assert_eq!(m.cols(), c);
        // Every word is zero.
        for &w in m.as_words() {
            assert_eq!(w, 0, "new({},{}) has nonzero word", r, c);
        }
        // Every cell reads as 0.
        for i in 0..r {
            for j in 0..c {
                assert_eq!(m.get(i, j), 0, "new({},{})[{},{}] != 0", r, c, i, j);
            }
        }
    }
}

/// `Mzd::identity` matches expected diagonal at edge sizes (square only).
#[test]
fn identity_at_edge_sizes() {
    for &n in &[0usize, 1, 63, 64, 65, 127, 128, 129] {
        let id = Mzd::identity(n);
        assert_eq!(id.rows(), n);
        assert_eq!(id.cols(), n);
        for i in 0..n {
            for j in 0..n {
                let expect = if i == j { 1 } else { 0 };
                assert_eq!(id.get(i, j), expect, "I({})[{},{}] wrong", n, i, j);
            }
        }
    }
}

/// `Mzd::random` works at every edge shape (doesn't panic, respects
/// the high-bitmask invariant).
#[test]
fn random_at_edge_shapes() {
    for &(r, c) in EDGE_SHAPES {
        let m = Mzd::random(r, c, 0x1234);
        assert_eq!(m.rows(), r);
        assert_eq!(m.cols(), c);
        // High-bitmask invariant: excess bits in the last word are zero.
        if c > 0 && r > 0 {
            let cols_mod = c % 64;
            if cols_mod != 0 {
                let mask = (1u64 << cols_mod) - 1;
                let width = c.div_ceil(64);
                for i in 0..r {
                    let row = m.row_words(i);
                    let high = row[width - 1];
                    assert_eq!(
                        high & !mask,
                        0,
                        "random({},{}) row {} has dirty excess",
                        r,
                        c,
                        i
                    );
                }
            }
        }
    }
}

/// `from_byte_buffer ∘ to_byte_buffer == identity` at every edge shape.
#[test]
fn byte_buffer_round_trip_at_edge_shapes() {
    for &(r, c) in EDGE_SHAPES {
        let m = Mzd::random(r, c, 0x9999);
        let bytes = m.to_byte_buffer();
        assert_eq!(
            bytes.len(),
            r * c,
            "to_byte_buffer({},{}) wrong length",
            r,
            c
        );
        let m2 = Mzd::from_byte_buffer(r, c, &bytes);
        assert_eq!(m.as_words(), m2.as_words(), "round-trip at ({},{})", r, c);
    }
}

// ---- Transpose ----------------------------------------------------------

#[test]
fn transpose_at_edge_shapes() {
    for &(r, c) in EDGE_SHAPES {
        let m = Mzd::random(r, c, 0x5555);
        let t = m.transpose();
        assert_eq!(t.rows(), c, "transpose({},{}) rows", r, c);
        assert_eq!(t.cols(), r, "transpose({},{}) cols", r, c);
        // Cell-by-cell oracle.
        for i in 0..r {
            for j in 0..c {
                assert_eq!(
                    m.get(i, j),
                    t.get(j, i),
                    "transpose at ({},{})[{},{}]",
                    r,
                    c,
                    i,
                    j
                );
            }
        }
        // Involution: (A^T)^T == A.
        let tt = t.transpose();
        assert_eq!(tt.as_words(), m.as_words());
    }
}

// ---- Matmul -------------------------------------------------------------

/// Matmul-compatible shape triples to exercise. Each generates inputs
/// `A: (r × k)`, `B: (k × c)` for matmul.
const MATMUL_SHAPES: &[(usize, usize, usize)] = &[
    (0, 0, 0),
    (0, 0, 5),
    (0, 5, 0),
    (5, 0, 0),
    (5, 0, 5),
    (1, 1, 1),
    (1, 5, 1),
    (5, 1, 5),
    (1, 64, 1),
    (1, 65, 1),
    (64, 1, 64),
    (65, 1, 65),
    (64, 64, 1),
    (1, 64, 64),
    (2, 3, 5),
    (5, 3, 2),
    (64, 64, 64),
    (65, 65, 65),
    (127, 129, 131),
    (1, 1000, 1),
    (1000, 1, 1000),
    (3, 1000, 5),
    (100, 1, 100), // rank-1 outer product
];

#[test]
fn matmul_matches_schoolbook_at_edge_shapes() {
    for &(r, k, c) in MATMUL_SHAPES {
        let a = Mzd::random(r, k, 0xA001 ^ (r * 31 + k) as u64);
        let b = Mzd::random(k, c, 0xB002 ^ (k * 17 + c) as u64);
        let got = Mzd::matmul(&a, &b);
        let expect = schoolbook(&a, &b);
        assert_eq!(
            got.as_words(),
            expect.as_words(),
            "matmul disagrees at ({}×{} · {}×{})",
            r,
            k,
            k,
            c
        );
        // Shape sanity.
        assert_eq!(got.rows(), r);
        assert_eq!(got.cols(), c);
    }
}

#[test]
fn matmul_a_times_zero_is_zero() {
    for &(r, k, c) in MATMUL_SHAPES {
        let a = Mzd::random(r, k, 1);
        let z = Mzd::new(k, c);
        let got = Mzd::matmul(&a, &z);
        for &w in got.as_words() {
            assert_eq!(w, 0, "A·0 != 0 at ({},{},{})", r, k, c);
        }
    }
}

#[test]
fn matmul_zero_times_b_is_zero() {
    for &(r, k, c) in MATMUL_SHAPES {
        let z = Mzd::new(r, k);
        let b = Mzd::random(k, c, 2);
        let got = Mzd::matmul(&z, &b);
        for &w in got.as_words() {
            assert_eq!(w, 0, "0·B != 0 at ({},{},{})", r, k, c);
        }
    }
}

#[test]
fn matmul_a_times_identity_is_a() {
    for &(r, c) in EDGE_SHAPES {
        if c == 0 {
            continue; // identity(0) is fine but matmul with k=0 → all zeros
        }
        let a = Mzd::random(r, c, 3);
        let i = Mzd::identity(c);
        let got = Mzd::matmul(&a, &i);
        assert_eq!(got.as_words(), a.as_words(), "A·I != A at ({},{})", r, c);
    }
}

#[test]
fn matmul_identity_times_a_is_a() {
    for &(r, c) in EDGE_SHAPES {
        if r == 0 {
            continue;
        }
        let i = Mzd::identity(r);
        let a = Mzd::random(r, c, 4);
        let got = Mzd::matmul(&i, &a);
        assert_eq!(got.as_words(), a.as_words(), "I·A != A at ({},{})", r, c);
    }
}

// ---- echelonize + rank --------------------------------------------------

#[test]
fn echelonize_at_edge_shapes() {
    for &(r, c) in EDGE_SHAPES {
        let m = Mzd::random(r, c, 5);
        let mut m_ref = m.clone();
        let ref_rank = m_ref.echelonize(false);
        let mut m_rref = m.clone();
        let rref_rank = m_rref.echelonize(true);
        // Rank is independent of full/partial echelonize.
        assert_eq!(
            ref_rank, rref_rank,
            "echelonize rank inconsistent at ({},{})",
            r, c
        );
        // Rank ≤ min(rows, cols).
        assert!(ref_rank <= r.min(c), "rank > min(r,c) at ({},{})", r, c);
        // rank() (clone+echelonize) matches.
        assert_eq!(m.rank(), ref_rank, "Mzd::rank disagrees at ({},{})", r, c);
    }
}

#[test]
fn rank_zero_matrix_is_zero() {
    for &(r, c) in EDGE_SHAPES {
        let z = Mzd::new(r, c);
        assert_eq!(z.rank(), 0, "rank(0) != 0 at ({},{})", r, c);
    }
}

#[test]
fn rank_identity_is_full() {
    for &n in &[0usize, 1, 63, 64, 65, 127, 128, 129, 256] {
        assert_eq!(Mzd::identity(n).rank(), n, "rank(I_{}) != {}", n, n);
    }
}

#[test]
fn ple_rank_clamps_a_wide_panel_to_the_remaining_rows() {
    for &(rows, cols, seed) in &[(1, 1, 1), (3, 7, 2), (6, 65, 3), (2, 257, 4)] {
        let matrix = Mzd::random(rows, cols, seed);
        assert_eq!(matrix.ple_rank(), matrix.rank());
    }
}

// ---- PLUQ ---------------------------------------------------------------

#[test]
fn pluq_at_edge_shapes() {
    for &(r, c) in EDGE_SHAPES {
        let a = Mzd::random(r, c, 6);
        let (p, l, u, q, rank) = a.pluq();
        // Shapes.
        let k = r.min(c);
        assert_eq!(p.rows(), r, "P.rows wrong at ({},{})", r, c);
        assert_eq!(p.cols(), r, "P.cols wrong at ({},{})", r, c);
        assert_eq!(l.rows(), r, "L.rows wrong at ({},{})", r, c);
        assert_eq!(l.cols(), k, "L.cols wrong at ({},{})", r, c);
        assert_eq!(u.rows(), k, "U.rows wrong at ({},{})", r, c);
        assert_eq!(u.cols(), c, "U.cols wrong at ({},{})", r, c);
        assert_eq!(q.rows(), c, "Q.rows wrong at ({},{})", r, c);
        assert_eq!(q.cols(), c, "Q.cols wrong at ({},{})", r, c);
        // Rank consistency.
        assert!(rank <= k, "rank > k at ({},{})", r, c);
        assert_eq!(rank, a.rank(), "PLUQ rank != Mzd::rank at ({},{})", r, c);
        // P · L · U · Q == A.
        let lu = Mzd::matmul(&l, &u);
        let plu = Mzd::matmul(&p, &lu);
        let pluq = Mzd::matmul(&plu, &q);
        assert_eq!(
            pluq.as_words(),
            a.as_words(),
            "P·L·U·Q != A at ({},{})",
            r,
            c
        );
    }
}

// ---- solve_left + nullspace + inverse ----------------------------------

#[test]
fn solve_consistent_at_edge_shapes() {
    // A · y is always consistent. Test solve_left(A, A·y) returns
    // some x with A·x = A·y.
    for &(r, c) in EDGE_SHAPES {
        for &nrhs in &[0usize, 1, 3] {
            let a = Mzd::random(r, c, 7);
            let y = Mzd::random(c, nrhs, 8);
            let b = Mzd::matmul(&a, &y);
            let x = match Mzd::solve_left(&a, &b) {
                Some(x) => x,
                None => panic!(
                    "solve_left returned None on consistent system at ({},{},{})",
                    r, c, nrhs
                ),
            };
            assert_eq!(x.rows(), c, "x.rows wrong");
            assert_eq!(x.cols(), nrhs, "x.cols wrong");
            let ax = Mzd::matmul(&a, &x);
            assert_eq!(
                ax.as_words(),
                b.as_words(),
                "A·x != A·y at ({},{},{})",
                r,
                c,
                nrhs
            );
        }
    }
}

#[test]
fn nullspace_at_edge_shapes() {
    for &(r, c) in EDGE_SHAPES {
        let a = Mzd::random(r, c, 9);
        let k = Mzd::nullspace(&a);
        // Shape.
        assert_eq!(k.rows(), c);
        let nullity = k.cols();
        // Rank-nullity.
        assert_eq!(a.rank() + nullity, c, "rank-nullity at ({},{})", r, c);
        // A · K == 0.
        if nullity > 0 && r > 0 {
            let ak = Mzd::matmul(&a, &k);
            for &w in ak.as_words() {
                assert_eq!(w, 0, "A·K != 0 at ({},{})", r, c);
            }
        }
    }
}

#[test]
fn inverse_at_edge_sizes() {
    for &n in &[0usize, 1, 64, 65, 127, 128, 129] {
        let i = Mzd::identity(n);
        let inv = match Mzd::inverse(&i) {
            Some(x) => x,
            None => {
                if n == 0 {
                    continue;
                }
                panic!("inverse(I_{}) returned None", n);
            }
        };
        assert_eq!(inv.as_words(), i.as_words(), "inverse(I_{}) != I", n);
    }
}

#[test]
fn inverse_of_singular_returns_none() {
    // Rank-(n-1) matrices: zero row + identity-on-rest.
    for &n in &[2usize, 64, 65, 128] {
        let mut a = Mzd::identity(n);
        // Zero out row 0 → rank n-1.
        for j in 0..n {
            a.set(0, j, 0);
        }
        assert!(
            Mzd::inverse(&a).is_none(),
            "inverse of rank-{} matrix should be None (n={})",
            n - 1,
            n
        );
    }
}
