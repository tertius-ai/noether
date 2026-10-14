// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Runnable tutorial of the `mzs` surface. Run with:
//!
//! ```bash
//! cargo run --release --example getting_started
//! ```

use mzs::Mzs;

fn main() {
    println!("mzs — sparse F_2 linear algebra getting-started\n");

    // Build a 4×6 matrix from a list of 1-positions.
    let a = Mzs::from_triples(
        4,
        6,
        &[
            (0, 0),
            (0, 2),
            (0, 5),
            (1, 1),
            (1, 3),
            (2, 0),
            (2, 2),
            (2, 5),
            (3, 4),
        ],
    );
    println!(
        "A is {} × {}, nnz = {}, density = {:.3}\n",
        a.rows(),
        a.cols(),
        a.nnz(),
        a.density()
    );

    // Densify for inspection (one byte per cell, row-major).
    println!("A as a dense grid:");
    let dense = a.to_byte_buffer();
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            print!("{} ", dense[i * a.cols() + j]);
        }
        println!();
    }
    println!();

    // Note: row 2 = row 0 (same pattern (0, 2, 5)) — so rank should be 3.
    let r = a.rank();
    println!("rank(A) = {}\n", r);

    let (rref, rref_rank) = a.rref();
    println!("RREF(A) (rank = {}):", rref_rank);
    let rref_dense = rref.to_byte_buffer();
    for i in 0..rref.rows() {
        for j in 0..rref.cols() {
            print!("{} ", rref_dense[i * rref.cols() + j]);
        }
        println!();
    }
    println!();

    // Nullspace: dimension cols - rank = 6 - 3 = 3.
    let k = a.nullspace();
    println!("nullspace basis is {} × {}", k.rows(), k.cols());
    // Verify A · k = 0.
    let ak = a.matmul(&k);
    println!("A · nullspace has nnz = {} (expect 0)\n", ak.nnz());

    // Solve A · X = b for a known-solvable b: take b = A · x for some x.
    let x = Mzs::from_triples(6, 1, &[(0, 0), (3, 0)]);
    let b = a.matmul(&x);
    println!("b = A · x, where x = [(0,0), (3,0)]:");
    let bd = b.to_byte_buffer();
    for i in 0..b.rows() {
        println!("  {}", bd[i]);
    }
    let x_solved = Mzs::solve_left(&a, &b).expect("consistent by construction");
    let b_check = a.matmul(&x_solved);
    println!("solve_left returned X' with A·X' = b: {}\n", b_check == b);

    // Byte-buffer round-trip: bridge to dense F_2 LA without a path-dep.
    let bytes = a.to_byte_buffer();
    let a_round_trip = Mzs::from_byte_buffer(a.rows(), a.cols(), &bytes);
    assert_eq!(a, a_round_trip);
    println!("Byte-buffer round-trip preserved A bit-for-bit.");
    println!("    (Use this to bridge to m4rst::Mzd or any dense F_2 type.)");
}
