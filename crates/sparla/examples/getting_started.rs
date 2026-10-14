// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Tutorial walkthrough of sparla's public API across both rings.
//!
//! Build matrices from COO triples, run the Tier-1 ops, and check
//! their algebraic invariants. Run with
//! `cargo run --release --example getting_started`.

use sparla::storage::{CsrFp, CsrZ, SparseRingStorage};
use sparla::Integer;

fn z(v: i64) -> Integer {
    Integer::from(v)
}

fn main() {
    println!("=== sparla getting_started ===\n");

    // ---- 1. Construction ----------------------------------------
    //
    // `CsrZ::from_coo(rows, cols, &triples)` is the canonical
    // entry. Triples are `(row, col, Integer)`; duplicates at the
    // same `(r, c)` are summed; entries that reduce to zero are
    // dropped. Same shape for `CsrFp::from_coo` with `i64` triple
    // values + a prime modulus.
    let triples_z = vec![(0, 0, z(2)), (0, 1, z(4)), (1, 0, z(6)), (1, 1, z(8))];
    let a = CsrZ::from_coo(2, 2, &triples_z);
    println!(
        "A is {}×{} with {} stored entries",
        a.rows(),
        a.cols(),
        a.nnz()
    );

    // ---- 2. Tier-1 Z ops ----------------------------------------
    //
    // rank, det, hnf, snf, kernel, image, solve_left, transpose,
    // matmul, hnf_with_transforms, elementary_divisors.
    println!("\nrank(A) = {}", a.rank());
    println!("det(A) = {}", a.det());

    let h = a.hnf();
    println!("hnf(A) =");
    for r in 0..h.rows() {
        let row: Vec<i64> = (0..h.cols())
            .map(|c| h.get(r, c).map(|v| v.to_i64().unwrap()).unwrap_or(0))
            .collect();
        println!("  {:?}", row);
    }

    println!(
        "snf(A) = {:?}",
        a.snf()
            .iter()
            .map(|d| d.to_i64().unwrap())
            .collect::<Vec<_>>()
    );
    println!(
        "elementary_divisors(A) = {:?}",
        a.elementary_divisors()
            .iter()
            .map(|d| d.to_i64().unwrap())
            .collect::<Vec<_>>()
    );

    // ---- 3. The HNF transform: U · A = H ------------------------
    let (h, u) = a.hnf_with_transforms();
    let ua = u.matmul(&a);
    assert_eq!(
        ua.to_dense(),
        h.to_dense(),
        "U · A must equal H bit-for-bit"
    );
    println!("\nU · A == H verified (det(U) = {})", u.det());

    // ---- 4. Sparse · sparse matmul ------------------------------
    let aa = a.matmul(&a);
    println!(
        "A · A is {}×{} with {} entries",
        aa.rows(),
        aa.cols(),
        aa.nnz()
    );

    // ---- 5. F_p surface ------------------------------------------
    //
    // Same shape but with `u32`-storage entries in `[0, prime)`.
    let p = 7;
    let triples_fp: Vec<(usize, usize, i64)> = vec![(0, 0, 1), (0, 1, 2), (1, 1, 1), (2, 2, 1)];
    let a_fp = CsrFp::from_coo(3, 3, &triples_fp, p);
    println!(
        "\nA (mod {}): rank = {}, det = {}",
        p,
        a_fp.rank(),
        a_fp.det()
    );

    let rref = a_fp.rref();
    println!("  pivot cols = {:?}", rref.pivot_cols);

    let k = a_fp.kernel();
    let img = a_fp.image();
    println!("  kernel basis: {} × {}", k.rows(), k.cols());
    println!("  image basis:  {} × {}", img.rows(), img.cols());

    // ---- 6. Algebraic invariant: A · K = 0 ----------------------
    let ak = a_fp.matmul(&k);
    assert_eq!(ak.nnz(), 0, "A · kernel(A) must be zero");
    println!("  A · kernel(A) == 0 verified");

    // ---- 7. Solve A · x = b over F_p ----------------------------
    let b = CsrFp::from_coo(3, 1, &[(0, 0, 3), (1, 0, 1), (2, 0, 4)], p);
    if let Some(x) = a_fp.solve(&b) {
        let ax = a_fp.matmul(&x);
        assert_eq!(ax.to_dense(), b.to_dense());
        println!("  solve(A, b) found x with A·x = b");
    }

    println!("\n=== done ===");
}
