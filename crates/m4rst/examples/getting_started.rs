// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Walks through the public m4rst API: build a matrix, multiply,
//! decompose, solve. Run with `cargo run --release --example
//! getting_started`.

use m4rst::Mzd;

fn main() {
    // ---- 1. construction ----------------------------------------
    //
    // `Mzd::new(r, c)` is the zero matrix; `Mzd::identity(n)` is I;
    // `Mzd::random(r, c, seed)` is a seeded uniform-random matrix.
    let n = 64;
    let a = Mzd::random(n, n, 0xA001);
    let b = Mzd::random(n, n, 0xB002);
    let i = Mzd::identity(n);
    println!("A is {}×{}", a.rows(), a.cols());

    // ---- 2. matrix multiplication -------------------------------
    //
    // `Mzd::matmul` picks the fastest available kernel (naive M4RM
    // NTABLES Strassen + the parallel variants if the `parallel`
    // feature is on). Output: a fresh `Mzd`.
    let c = Mzd::matmul(&a, &b);
    assert_eq!(c.rows(), n);
    assert_eq!(c.cols(), n);

    // A · I == A.
    assert_eq!(Mzd::matmul(&a, &i).as_words(), a.as_words());

    // ---- 3. rank + PLUQ decomposition ---------------------------
    //
    // `rank` and `echelonize` are the cheapest interfaces.
    // `pluq` returns the full P · L · U · Q decomposition.
    let r = a.rank();
    println!("rank(A) = {}/{}", r, n);

    let (p, l, u, q, rank_via_pluq) = a.pluq();
    assert_eq!(r, rank_via_pluq);
    // P · L · U · Q == A — verify.
    let lu = Mzd::matmul(&l, &u);
    let plu = Mzd::matmul(&p, &lu);
    let pluq = Mzd::matmul(&plu, &q);
    assert_eq!(pluq.as_words(), a.as_words(), "PLUQ round-trip");

    // ---- 4. inverse (if full rank) ------------------------------
    if r == n {
        let a_inv = Mzd::inverse(&a).expect("full-rank => invertible");
        assert_eq!(Mzd::matmul(&a, &a_inv).as_words(), i.as_words());
        println!("A^{{-1}} verified.");
    }

    // ---- 5. solve and nullspace ---------------------------------
    //
    // `solve_left(A, B)` returns X such that A·X = B, or None.
    // `nullspace(A)` returns a basis matrix whose columns span ker(A).
    let y = Mzd::random(n, 8, 0xC003);
    let rhs = Mzd::matmul(&a, &y);
    let x = Mzd::solve_left(&a, &rhs).expect("consistent");
    assert_eq!(Mzd::matmul(&a, &x).as_words(), rhs.as_words());

    let kernel = Mzd::nullspace(&a);
    // A · (any column of kernel) is zero.
    if kernel.cols() > 0 {
        let ak = Mzd::matmul(&a, &kernel);
        for ij in ak.as_words() {
            assert_eq!(*ij, 0);
        }
        println!("nullspace dim = {}", kernel.cols());
    } else {
        println!("nullspace dim = 0 (A is full rank)");
    }

    // ---- 6. byte-buffer interop ---------------------------------
    //
    // For exchanging matrices with non-m4rst callers (Python, C, ...)
    // use `from_byte_buffer` / `to_byte_buffer` — row-major, one
    // byte per bit, value in the low bit (`byte & 1`).
    let bytes = a.to_byte_buffer();
    let a_back = Mzd::from_byte_buffer(n, n, &bytes);
    assert_eq!(a.as_words(), a_back.as_words());

    println!("\nAll checks passed.");
}
