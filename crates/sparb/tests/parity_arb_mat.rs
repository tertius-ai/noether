// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Parity oracle — sparb sparse SpMV vs FLINT dense `arb_mat_mul`.
//!
//! The contract this test pins: for the same input matrix `A` and
//! vector `x`, sparb's sparse SpMV and FLINT's dense
//! `arb_mat_mul(A_dense, x_col)` both compute output balls that
//! rigorously enclose the true product `A · x`. They use *different
//! rounding orderings* (sparse walks each row's stored entries left-
//! to-right; dense walks the whole row including structural zeros)
//! so the midpoints and radii will differ — but every pair of
//! corresponding output balls must **intersect**, since both
//! rigorously enclose the same true real value.
//!
//! Two balls `[a_mid ± a_rad]` and `[b_mid ± b_rad]` intersect iff
//! `|a_mid - b_mid| ≤ a_rad + b_rad`. That's the assertion.
//!
//! The oracle answers the question every ball-arithmetic kernel
//! needs to answer before it ships: "if I rolled my own primitive,
//! does it agree with FLINT?" This is the bit-rigorous regression
//! against the reference; the property tests in
//! `tests/randomised.rs` cover structural invariants (commutativity,
//! enclosure, etc.) that don't need a second implementation.

#[path = "common/mod.rs"]
mod common;

use common::arb_mat_ffi::{flint_supports_stride_layout, ArbMat};
use sparb::{Arb, CsrArb};

// ── ball-intersection predicate ───────────────────────────────────────

/// Returns true iff the two balls share at least one real point.
fn balls_intersect(a: (f64, f64), b: (f64, f64)) -> bool {
    let (am, ar) = a;
    let (bm, br) = b;
    (am - bm).abs() <= ar + br
}

// ── deterministic fixture generators ──────────────────────────────────

/// xorshift64 PRNG — small, deterministic, good enough for fixtures.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Random `(row, col, value)` triples up to `nnz = density · n²`.
/// Values are exactly representable f64s in `[-1, 1)`.
fn random_triples(n: usize, density: f64, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut state = seed | 1;
    let target = ((n * n) as f64 * density) as usize;
    (0..target)
        .map(|_| {
            let r = (xorshift(&mut state) as usize) % n;
            let c = (xorshift(&mut state) as usize) % n;
            let v = (xorshift(&mut state) % 2000) as f64 / 1000.0 - 1.0;
            (r, c, v)
        })
        .collect()
}

fn random_x(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| (xorshift(&mut state) % 2000) as f64 / 1000.0 - 1.0)
        .collect()
}

// ── core parity-check routine ─────────────────────────────────────────

fn parity_check(n: usize, density: f64, prec: i64, seed: u64) {
    // `arb_mat_struct`'s layout changed in FLINT 3.1 (the 4th field
    // became `slong stride` instead of `arb_ptr * rows`). On older
    // FLINT this `ArbMat` impl reads garbage as `stride` and walks
    // entries off the heap. Self-skip rather than crash; the test
    // is still valuable on any host with a recent FLINT.
    if !flint_supports_stride_layout() {
        eprintln!(
            "parity_arb_mat: skipping (linked FLINT < 3.1; arb_mat_struct \
             layout incompatible)"
        );
        return;
    }
    let triples = random_triples(n, density, seed);
    let x_f64 = random_x(n, seed.wrapping_add(1));

    // sparb path: COO → CSR → sparse SpMV.
    let triples_quads: Vec<(usize, usize, f64, f64)> =
        triples.iter().map(|&(r, c, v)| (r, c, v, 0.0)).collect();
    let a_sp = CsrArb::from_coo_f64(n, n, prec, &triples_quads);
    let x_sp: Vec<Arb> = x_f64.iter().map(|&v| Arb::from_f64(v)).collect();
    let y_sp = a_sp.spmv(&x_sp);

    // FLINT path: dense arb_mat × column-vec arb_mat.
    // Build at 128 bits so construction rounding can't be the
    // difference; the comparison precision is `prec`.
    let a_dn = ArbMat::from_triples(n, n, &triples, 128);
    let x_dn = ArbMat::col_from_f64(&x_f64);
    let y_dn = a_dn.mul(&x_dn, prec);

    // Compare every output ball — they must intersect.
    for (i, sp_ball) in y_sp.iter().enumerate() {
        let sp = sp_ball.to_pair(128);
        let dn = y_dn.entry_pair(i, 0, 128);
        assert!(
            balls_intersect(sp, dn),
            "row {i}: sparb ball {sp:?} disjoint from FLINT-dense ball \
             {dn:?} (n={n}, density={density}, prec={prec}, seed={seed:#x})"
        );
    }
}

// ── the actual tests — a sweep over (shape, density, prec) ────────────

#[test]
fn parity_n64_density_10pct_prec_64() {
    parity_check(64, 0.10, 64, 0x5EED_0001);
}

#[test]
fn parity_n128_density_5pct_prec_64() {
    parity_check(128, 0.05, 64, 0x5EED_0002);
}

#[test]
fn parity_n256_density_1pct_prec_64() {
    parity_check(256, 0.01, 64, 0x5EED_0003);
}

#[test]
fn parity_n64_density_10pct_prec_24() {
    // Low precision: both sides have generous radii, intersection
    // is loose. Still must hold.
    parity_check(64, 0.10, 24, 0x5EED_0004);
}

#[test]
fn parity_n64_density_10pct_prec_128() {
    // High precision: both sides have tight radii. Intersection
    // is correspondingly tight.
    parity_check(64, 0.10, 128, 0x5EED_0005);
}

#[test]
fn parity_n64_density_10pct_prec_256() {
    parity_check(64, 0.10, 256, 0x5EED_0006);
}

#[test]
fn parity_extreme_density_50pct() {
    // Dense-ish — sparb's pattern overhead is less of a win here,
    // but the parity contract still holds.
    parity_check(32, 0.50, 64, 0x5EED_0007);
}

#[test]
fn parity_very_sparse() {
    parity_check(512, 0.002, 64, 0x5EED_0008);
}

// ── matmul parity: sparb spgemm vs FLINT arb_mat_mul ─────────────────

fn matmul_parity_check(n: usize, density: f64, prec: i64, seed: u64) {
    if !flint_supports_stride_layout() {
        eprintln!(
            "parity_arb_mat: skipping (linked FLINT < 3.1; arb_mat_struct \
             layout incompatible)"
        );
        return;
    }
    let triples_a = random_triples(n, density, seed);
    let triples_b = random_triples(n, density, seed.wrapping_add(1));

    // sparb path: sparse · sparse via Gustavson.
    let quads_a: Vec<(usize, usize, f64, f64)> =
        triples_a.iter().map(|&(r, c, v)| (r, c, v, 0.0)).collect();
    let quads_b: Vec<(usize, usize, f64, f64)> =
        triples_b.iter().map(|&(r, c, v)| (r, c, v, 0.0)).collect();
    let a_sp = CsrArb::from_coo_f64(n, n, prec, &quads_a);
    let b_sp = CsrArb::from_coo_f64(n, n, prec, &quads_b);
    let c_sp = a_sp.matmul(&b_sp);

    // FLINT path: dense arb_mat × dense arb_mat. Build at 128 bits
    // so construction rounding is negligible vs the matmul prec.
    let a_dn = ArbMat::from_triples(n, n, &triples_a, 128);
    let b_dn = ArbMat::from_triples(n, n, &triples_b, 128);
    let c_dn = a_dn.mul(&b_dn, prec);

    // Compare every cell — sparse's stored entries materialise via
    // get(); structural zeros are (0, 0) balls.
    for i in 0..n {
        for j in 0..n {
            let sp = c_sp.get(i, j).map(|b| b.to_pair(128)).unwrap_or((0.0, 0.0));
            let dn = c_dn.entry_pair(i, j, 128);
            assert!(
                balls_intersect(sp, dn),
                "C[{i},{j}]: sparb ball {sp:?} disjoint from FLINT-dense \
                 ball {dn:?} (n={n}, density={density}, prec={prec}, \
                 seed={seed:#x})"
            );
        }
    }
}

#[test]
fn matmul_parity_n32_density_10pct_prec_64() {
    matmul_parity_check(32, 0.10, 64, 0x5EED_0101);
}

#[test]
fn matmul_parity_n64_density_5pct_prec_64() {
    matmul_parity_check(64, 0.05, 64, 0x5EED_0102);
}

#[test]
fn matmul_parity_n128_density_2pct_prec_64() {
    matmul_parity_check(128, 0.02, 64, 0x5EED_0103);
}

#[test]
fn matmul_parity_n32_density_25pct_prec_128() {
    // Denser fixture stresses spgemm's fill-pattern; higher prec
    // makes the intersection tighter.
    matmul_parity_check(32, 0.25, 128, 0x5EED_0104);
}

#[test]
fn matmul_parity_n32_density_10pct_prec_24() {
    matmul_parity_check(32, 0.10, 24, 0x5EED_0105);
}
