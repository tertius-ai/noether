// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Step-2 standalone kernel tests: XOR-into, transpose, naive matmul.
//!
//! Algebraic identities only — no libm4ri oracle here. The libm4ri
//! parity layer is in `parity_libm4ri.rs` (feature-gated).

use m4rst::{MatrixWrite, Mzd};

// ---- xor_into ---------------------------------------------------------

#[test]
fn xor_into_is_involution() {
    for &(r, c) in &[(1usize, 1usize), (5, 200), (8, 64), (8, 65), (33, 130)] {
        let a0 = Mzd::random(r, c, 0x1111_2222_3333_4444);
        let b = Mzd::random(r, c, 0x5555_6666_7777_8888);
        let mut a = a0.clone();
        a.xor_into(&b);
        a.xor_into(&b);
        assert_eq!(
            a.as_words(),
            a0.as_words(),
            "xor twice ≠ identity at ({}×{})",
            r,
            c
        );
    }
}

#[test]
fn xor_into_with_zero_is_noop() {
    let a = Mzd::random(8, 130, 42);
    let z = Mzd::new(8, 130);
    let mut a2 = a.clone();
    a2.xor_into(&z);
    assert_eq!(a.as_words(), a2.as_words());
}

#[test]
fn xor_into_self_is_zero() {
    let a = Mzd::random(8, 130, 99);
    let b = a.clone();
    let mut a2 = a;
    a2.xor_into(&b);
    for &w in a2.as_words() {
        assert_eq!(w, 0);
    }
}

#[test]
#[should_panic(expected = "row mismatch")]
fn xor_into_shape_panic() {
    let mut a = Mzd::new(3, 4);
    let b = Mzd::new(4, 4);
    a.xor_into(&b);
}

// ---- transpose --------------------------------------------------------

#[test]
fn transpose_dim_swap() {
    let m = Mzd::random(7, 130, 1);
    let t = m.transpose();
    assert_eq!(t.rows(), 130);
    assert_eq!(t.cols(), 7);
}

#[test]
fn transpose_is_involution() {
    for &(r, c) in &[(1usize, 1usize), (7, 130), (64, 65), (65, 64), (33, 200)] {
        let m = Mzd::random(r, c, 0xCAFE);
        let tt = m.transpose().transpose();
        assert_eq!(m.as_words(), tt.as_words(), "({}×{})", r, c);
    }
}

#[test]
fn transpose_identity_is_identity() {
    for &n in &[1usize, 64, 65, 130] {
        let i = Mzd::identity(n);
        let t = i.transpose();
        assert_eq!(i.as_words(), t.as_words());
    }
}

#[test]
fn transpose_matches_get_oracle() {
    // Drives the transpose against the algebraic definition cell-by-cell —
    // catches off-by-one or word/bit-order bugs that the involution test alone
    // can miss (an involution test passes even if rows and cols are silently
    // swapped twice incorrectly).
    let m = Mzd::random(20, 100, 17);
    let t = m.transpose();
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            assert_eq!(m.get(i, j), t.get(j, i), "({}, {}) wrong", i, j);
        }
    }
}

#[test]
fn transpose_large_random_matches_bit_oracle() {
    // Regression guard for the tile-walked 64×64-block transpose
    // at the common large size. The proptest sweep covers
    // correctness up to ~80, libm4ri parity covers up to 1280;
    // this fills the gap at the 4096 mid-large regime where a
    // tile-boundary off-by-one could silently corrupt one
    // boundary row/col and slip past every smaller test.
    //
    // Spot-checks 200 cells (deterministic seed) for cell-level
    // agreement with the get(i, j) oracle rather than a full
    // 4096² = 16M cell scan — the involution and parity tests
    // already cover whole-matrix equality at smaller N.
    let n = 4096usize;
    let m = Mzd::random(n, n, 0x4096_4096);
    let t = m.transpose();
    assert_eq!(t.rows(), n);
    assert_eq!(t.cols(), n);
    // Deterministic spot-check sweep.
    for k in 0..200u64 {
        let h = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let i = (h as usize) % n;
        let j = ((h >> 32) as usize) % n;
        assert_eq!(m.get(i, j), t.get(j, i), "spot ({}, {}) wrong", i, j);
    }
    // Boundary spot-check: cells at the tile-boundary indices
    // (i or j divisible by 64, or close to it) are the most
    // likely failure points.
    for &(i, j) in &[
        (0usize, 0usize),
        (63, 0),
        (64, 0),
        (65, 0),
        (0, 63),
        (0, 64),
        (0, 65),
        (n - 1, 0),
        (0, n - 1),
        (n - 1, n - 1),
        (63, 63),
        (64, 64),
        (4095, 4095),
    ] {
        assert_eq!(m.get(i, j), t.get(j, i), "boundary ({}, {}) wrong", i, j);
    }
}

// ---- naive matmul -----------------------------------------------------

#[test]
fn matmul_a_times_identity_is_a() {
    for &(r, c) in &[(1usize, 1usize), (5, 200), (8, 64), (8, 65), (33, 130)] {
        let a = Mzd::random(r, c, 7);
        let i = Mzd::identity(c);
        let prod = Mzd::matmul(&a, &i);
        assert_eq!(prod.as_words(), a.as_words(), "({}×{}) · I", r, c);
    }
}

#[test]
fn matmul_identity_times_a_is_a() {
    for &(r, c) in &[(1usize, 1usize), (5, 200), (8, 64), (8, 65), (33, 130)] {
        let a = Mzd::random(r, c, 8);
        let i = Mzd::identity(r);
        let prod = Mzd::matmul(&i, &a);
        assert_eq!(prod.as_words(), a.as_words(), "I · ({}×{})", r, c);
    }
}

#[test]
fn matmul_zero_annihilates() {
    let z = Mzd::new(5, 8);
    let a = Mzd::random(8, 12, 9);
    let prod = Mzd::matmul(&z, &a);
    for &w in prod.as_words() {
        assert_eq!(w, 0);
    }
}

#[test]
fn matmul_transpose_identity() {
    // (A · B)^T == B^T · A^T over any commutative ring; F_2 qualifies.
    let a = Mzd::random(7, 130, 11);
    let b = Mzd::random(130, 65, 13);
    let ab_t = Mzd::matmul(&a, &b).transpose();
    let bt_at = Mzd::matmul(&b.transpose(), &a.transpose());
    assert_eq!(ab_t.as_words(), bt_at.as_words());
}

#[test]
fn matmul_associativity() {
    // (A · B) · C == A · (B · C). Catches a wide class of shape /
    // index bugs that the identity tests miss.
    let a = Mzd::random(5, 17, 21);
    let b = Mzd::random(17, 33, 22);
    let c = Mzd::random(33, 7, 23);
    let ab_c = Mzd::matmul(&Mzd::matmul(&a, &b), &c);
    let a_bc = Mzd::matmul(&a, &Mzd::matmul(&b, &c));
    assert_eq!(ab_c.as_words(), a_bc.as_words());
}

#[test]
fn matmul_matches_oracle_brute_force() {
    // Cross-validate against the per-cell oracle definition:
    //   C[i, j] = XOR_k A[i, k] · B[k, j].
    // Run on a small enough size that the O(rows · cols · inner) brute
    // force is fast but on a non-aligned column count to exercise the
    // high-word path.
    let (r, k_dim, c_dim) = (8usize, 33usize, 65usize);
    let a = Mzd::random(r, k_dim, 31);
    let b = Mzd::random(k_dim, c_dim, 32);
    let prod = Mzd::matmul(&a, &b);
    for i in 0..r {
        for j in 0..c_dim {
            let mut acc = 0u8;
            for k in 0..k_dim {
                acc ^= a.get(i, k) & b.get(k, j);
            }
            assert_eq!(prod.get(i, j), acc, "({}, {}) wrong", i, j);
        }
    }
}

#[test]
#[should_panic(expected = "inner dimension mismatch")]
fn matmul_shape_panic() {
    let a = Mzd::new(3, 4);
    let b = Mzd::new(5, 6);
    let _ = Mzd::matmul(&a, &b);
}

// ---- M4RM cross-check vs naive (kernel-level, not dispatch level) -----

#[test]
fn m4rm_matches_naive_across_shapes() {
    use m4rst::matmul::{m4rm::matmul_m4rm, naive::matmul_naive};
    // Cover: column count not multiple of k=8 (e.g. 33), column count not
    // multiple of 64 (e.g. 65, 130), shapes both above and below the
    // dispatch threshold so the kernel sees a wide variety.
    for &(r, k, c, seed) in &[
        (1usize, 1usize, 1usize, 1u64),
        (8, 7, 12, 2),
        (8, 8, 8, 3),
        (8, 9, 12, 4),
        (32, 33, 65, 5),
        (64, 64, 64, 6),
        (64, 65, 65, 7),
        (100, 130, 200, 8),
    ] {
        let a = Mzd::random(r, k, seed);
        let b = Mzd::random(k, c, seed.wrapping_mul(31));
        let m = matmul_m4rm(&a, &b);
        let n = matmul_naive(&a, &b);
        assert_eq!(
            m.as_words(),
            n.as_words(),
            "M4RM disagrees with naive at ({}×{} · {}×{}, seed={})",
            r,
            k,
            k,
            c,
            seed
        );
    }
}

#[test]
fn m4rm_a_times_identity_is_a() {
    use m4rst::matmul::m4rm::matmul_m4rm;
    // Force M4RM directly so we don't depend on the dispatcher's
    // threshold; exercises the strip-spanning slice extraction.
    for &(r, c) in &[(8usize, 65usize), (33, 130), (64, 200)] {
        let a = Mzd::random(r, c, 0xCAFE_BABE);
        let i = Mzd::identity(c);
        let prod = matmul_m4rm(&a, &i);
        assert_eq!(prod.as_words(), a.as_words(), "M4RM: ({}×{}) · I", r, c);
    }
}

// ---- Strassen cross-check vs M4RM at small threshold -----------------
//
// The view-based Strassen requires every dim to be a multiple of 128
// AND ≥ threshold to recurse. Tests below pick sizes that satisfy
// both at threshold=128, giving log2(N/128) levels of recursion.

#[test]
fn strassen_matches_m4rm_across_shapes() {
    use m4rst::matmul::{m4rm::matmul_m4rm, strassen::matmul_strassen_with_threshold};
    for &(r, k, c, seed) in &[
        // Below the 128-gate — Strassen falls through to M4RM.
        (8usize, 8usize, 8usize, 1u64),
        (64, 64, 64, 2),
        // Hits the 128-gate, threshold=128 → 1 recursion level.
        (128, 128, 128, 3),
        (256, 256, 256, 4),
        // 2 recursion levels at threshold=128: 512 → 256 → 128 leaf.
        (512, 512, 512, 5),
        // Rectangular but every dim multiple of 128.
        (128, 256, 384, 6),
        (256, 128, 256, 7),
        // Not all dims multiples of 128 — Strassen bails immediately.
        (130, 130, 130, 8),
        (256, 200, 256, 9),
    ] {
        let a = Mzd::random(r, k, seed);
        let b = Mzd::random(k, c, seed.wrapping_mul(31));
        let s = matmul_strassen_with_threshold(&a, &b, 128);
        let m = matmul_m4rm(&a, &b);
        assert_eq!(
            s.as_words(),
            m.as_words(),
            "Strassen disagrees with M4RM at ({}×{} · {}×{}, seed={})",
            r,
            k,
            k,
            c,
            seed
        );
    }
}

#[test]
fn strassen_a_times_identity() {
    use m4rst::matmul::strassen::matmul_strassen_with_threshold;
    // threshold=128, N=512 gives 2 levels of recursion.
    for &n in &[128usize, 256, 512] {
        let a = Mzd::random(n, n, 0xFEED);
        let i = Mzd::identity(n);
        let prod = matmul_strassen_with_threshold(&a, &i, 128);
        assert_eq!(prod.as_words(), a.as_words(), "Strassen({}×{}) · I", n, n);
    }
}

#[test]
fn ntables_m4rm_matches_unblocked() {
    use m4rst::matmul::m4rm::{matmul_m4rm_ntables, matmul_m4rm_with_k};
    // Cover the kk=64 super-strip path (>= 64 inner dim), the tail
    // (inner dim % 64 != 0), and inner dim < 64 (pure tail).
    for &(r, k_dim, c, seed) in &[
        (8usize, 7usize, 8usize, 1u64), // pure tail, kk=64 path not entered
        (8, 64, 8, 2),                  // exactly one kk super-strip
        (8, 65, 8, 3),                  // 1 kk + 1-bit tail
        (130, 130, 200, 4),             // mixed
        (200, 200, 200, 5),             // bigger non-aligned
        (256, 256, 256, 6),             // exactly multiple kk
        (300, 300, 300, 7),             // multiple kk + tail
        (33, 100, 65, 8),               // odd dims everywhere
    ] {
        let a = Mzd::random(r, k_dim, seed);
        let b = Mzd::random(k_dim, c, seed.wrapping_mul(31));
        let single = matmul_m4rm_with_k(&a, &b, 8);
        let ntables = matmul_m4rm_ntables(&a, &b);
        assert_eq!(
            single.as_words(),
            ntables.as_words(),
            "NTABLES disagrees with single-table at ({}×{} · {}×{}, seed={})",
            r,
            k_dim,
            k_dim,
            c,
            seed
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn strassen_peeled_matches_schoolbook() {
    use m4rst::matmul::strassen::matmul_strassen_peeled_parallel_leaf;
    // Test sizes that are deliberately NOT multiples of 128 so the
    // peeling path is exercised. Plus a few aligned sizes for
    // boundary/sanity coverage.
    for &(r, k_dim, c, seed) in &[
        // Non-aligned (peeling path).
        (200usize, 200usize, 200usize, 1u64),
        (300, 300, 300, 2),
        (500, 500, 500, 3),
        (200, 130, 250, 4), // mixed alignments
        (130, 200, 130, 5),
        (500, 137, 500, 6), // smallest non-aligned k
        // Aligned (should also work — falls through to inner Strassen).
        (256, 256, 256, 7),
        (384, 384, 384, 8),
        // Edge: m < 128 (inner is zero) → falls to NTABLES-parallel.
        (50, 200, 200, 9),
        (200, 50, 200, 10),
        (200, 200, 50, 11),
    ] {
        let a = Mzd::random(r, k_dim, seed);
        let b = Mzd::random(k_dim, c, seed.wrapping_mul(31));
        let peeled = matmul_strassen_peeled_parallel_leaf(&a, &b);
        // Schoolbook oracle.
        let mut oracle = Mzd::new(r, c);
        for i in 0..r {
            for j in 0..c {
                let mut acc = 0u8;
                for kk in 0..k_dim {
                    acc ^= a.get(i, kk) & b.get(kk, j);
                }
                if acc != 0 {
                    oracle.set(i, j, 1);
                }
            }
        }
        assert_eq!(
            peeled.as_words(),
            oracle.as_words(),
            "peeled Strassen disagrees with schoolbook at ({}×{} · {}×{}, seed={})",
            r,
            k_dim,
            k_dim,
            c,
            seed
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn strassen_parallel_leaf_matches_serial() {
    use m4rst::matmul::strassen::{
        matmul_strassen_parallel_leaf_with_threshold, matmul_strassen_with_threshold,
    };
    for &(n, seed, thr) in &[
        (256usize, 1u64, 128usize),
        (512, 2, 128),
        (1024, 3, 128),
        (1024, 4, 256),
        (256, 5, 4096), // threshold > N → both fall through to leaf
    ] {
        let a = Mzd::random(n, n, seed);
        let b = Mzd::random(n, n, seed.wrapping_mul(31));
        let serial = matmul_strassen_with_threshold(&a, &b, thr);
        let par = matmul_strassen_parallel_leaf_with_threshold(&a, &b, thr);
        assert_eq!(
            serial.as_words(),
            par.as_words(),
            "Strassen parallel-leaf disagrees with serial-leaf at N={}, thr={}, seed={}",
            n,
            thr,
            seed
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_mblocked_ntables_matches_serial() {
    use m4rst::matmul::m4rm::{matmul_m4rm_ntables, matmul_m4rm_ntables_mblocked_parallel};
    for &(r, k_dim, c, seed, mb) in &[
        (3usize, 0usize, 5usize, 0u64, 2usize),
        (3, 5, 0, 0, 2),
        (8usize, 64usize, 8usize, 1u64, 4usize),
        (256, 256, 256, 2, 64),
        (1024, 1024, 1024, 3, 256),
        (1000, 1000, 1000, 4, 333),
        (130, 130, 200, 5, 17),
        (2048, 2048, 2048, 6, 512),
        (8192, 4096, 64, 7, 1024),
    ] {
        let a = Mzd::random(r, k_dim, seed);
        let b = Mzd::random(k_dim, c, seed.wrapping_mul(31));
        let serial = matmul_m4rm_ntables(&a, &b);
        let parmblock = matmul_m4rm_ntables_mblocked_parallel(&a, &b, mb);
        assert_eq!(
            serial.as_words(),
            parmblock.as_words(),
            "parallel-mblock NTABLES disagrees with serial at ({}×{} · {}×{}, mb={}, seed={})",
            r,
            k_dim,
            k_dim,
            c,
            mb,
            seed
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_ntables_matches_serial() {
    use m4rst::matmul::m4rm::{matmul_m4rm_ntables, matmul_m4rm_ntables_parallel};
    for &(r, k_dim, c, seed) in &[
        (8usize, 64usize, 8usize, 1u64),
        (256, 256, 256, 2),
        (1024, 1024, 1024, 3),
        (1000, 1000, 1000, 4), // non-power-of-2
        (130, 130, 200, 5),
        (2048, 2048, 2048, 6),
    ] {
        let a = Mzd::random(r, k_dim, seed);
        let b = Mzd::random(k_dim, c, seed.wrapping_mul(31));
        let serial = matmul_m4rm_ntables(&a, &b);
        let parallel = matmul_m4rm_ntables_parallel(&a, &b);
        assert_eq!(
            serial.as_words(),
            parallel.as_words(),
            "parallel NTABLES disagrees with serial at ({}×{} · {}×{}, seed={})",
            r,
            k_dim,
            k_dim,
            c,
            seed
        );
    }
}

#[test]
fn ntables_mblocked_matches_ntables() {
    use m4rst::matmul::m4rm::{matmul_m4rm_ntables, matmul_m4rm_ntables_mblocked};
    for &(r, k_dim, c, seed, mb) in &[
        (8usize, 64usize, 8usize, 1u64, 4usize),
        (256, 256, 256, 2, 64),
        (1024, 1024, 1024, 3, 256),
        (1000, 1000, 1000, 4, 333), // non-tiling block
        (130, 130, 200, 5, 17),
    ] {
        let a = Mzd::random(r, k_dim, seed);
        let b = Mzd::random(k_dim, c, seed.wrapping_mul(31));
        let direct = matmul_m4rm_ntables(&a, &b);
        let blocked = matmul_m4rm_ntables_mblocked(&a, &b, mb);
        assert_eq!(
            direct.as_words(),
            blocked.as_words(),
            "NTABLES m-blocked disagrees with direct at ({}×{} · {}×{}, mb={}, seed={})",
            r,
            k_dim,
            k_dim,
            c,
            mb,
            seed
        );
    }
}

#[test]
fn nblocked_m4rm_matches_unblocked() {
    use m4rst::matmul::m4rm::{matmul_m4rm_nblocked_with_k, matmul_m4rm_with_k};
    // n_block must be multiple of 64 (view col-alignment); cover both
    // exactly-tiling and partial-final-strip cases.
    for &(r, k_dim, c, seed, nb) in &[
        (8usize, 8usize, 8usize, 1u64, 64usize),
        (130, 65, 256, 2, 64),
        (130, 65, 200, 3, 128), // last strip is partial
        (1000, 1000, 1024, 4, 256),
        (1024, 1024, 1024, 5, 512),
    ] {
        let a = Mzd::random(r, k_dim, seed);
        let b = Mzd::random(k_dim, c, seed.wrapping_mul(31));
        let unblocked = matmul_m4rm_with_k(&a, &b, 8);
        let blocked = matmul_m4rm_nblocked_with_k(&a, &b, 8, nb);
        assert_eq!(
            unblocked.as_words(),
            blocked.as_words(),
            "N-blocked disagrees with unblocked at ({}×{} · {}×{}, nb={}, seed={})",
            r,
            k_dim,
            k_dim,
            c,
            nb,
            seed
        );
    }
}

#[test]
fn blocked_m4rm_matches_unblocked() {
    use m4rst::matmul::m4rm::{matmul_m4rm_blocked_with_k, matmul_m4rm_with_k};
    for &(r, k_dim, c, seed, mb) in &[
        // (rows, inner, cols, seed, m_block)
        (8usize, 8usize, 8usize, 1u64, 4usize),
        (64, 64, 64, 2, 32),
        (130, 65, 200, 3, 64), // non-power-of-2 dims, odd m_block divisions
        (1000, 1000, 1000, 4, 256),
        (1024, 1024, 1024, 5, 512),
    ] {
        let a = Mzd::random(r, k_dim, seed);
        let b = Mzd::random(k_dim, c, seed.wrapping_mul(31));
        let unblocked = matmul_m4rm_with_k(&a, &b, 8);
        let blocked = matmul_m4rm_blocked_with_k(&a, &b, 8, mb);
        assert_eq!(
            unblocked.as_words(),
            blocked.as_words(),
            "cache-blocked disagrees with unblocked at ({}×{} · {}×{}, mb={}, seed={})",
            r,
            k_dim,
            k_dim,
            c,
            mb,
            seed
        );
    }
}

#[test]
fn strassen_associativity() {
    use m4rst::matmul::strassen::matmul_strassen_with_threshold;
    // 256 hits the gate at threshold=128 with 1 level of recursion.
    let a = Mzd::random(256, 256, 11);
    let b = Mzd::random(256, 256, 13);
    let c = Mzd::random(256, 256, 17);
    let ab_c =
        matmul_strassen_with_threshold(&matmul_strassen_with_threshold(&a, &b, 128), &c, 128);
    let a_bc =
        matmul_strassen_with_threshold(&a, &matmul_strassen_with_threshold(&b, &c, 128), 128);
    assert_eq!(ab_c.as_words(), a_bc.as_words());
}
