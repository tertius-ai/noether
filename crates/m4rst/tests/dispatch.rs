// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Coverage-driven dispatcher tests — exercise the parallel and
//! serial-Strassen branches of `Mzd::matmul` at moderate N, where
//! the standard test suite (which mostly uses N ≤ 1024) doesn't
//! reach. Bigger N is covered by the bench examples + libm4ri
//! parity at N=1280, so we stay at N=8192 here for fast `cargo test`.

use m4rst::Mzd;

/// Schoolbook oracle.
fn schoolbook(a: &Mzd, b: &Mzd) -> Mzd {
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

/// Hit the mid-tier parallel dispatcher path (PARALLEL_MIN_DIM and
/// PAR_MBLOCK_MIN both = 6144). The result is sanity-checked against
/// an algebraic identity (A · I = A) rather than schoolbook (too
/// slow at N=8192).
#[test]
fn dispatch_mid_size_parallel_paths() {
    let a = Mzd::random(8192, 8192, 0xDEAD);
    let i = Mzd::identity(8192);
    let prod = Mzd::matmul(&a, &i);
    assert_eq!(prod.as_words(), a.as_words(), "A·I = A at N=8192");
    let prod_i_a = Mzd::matmul(&i, &a);
    assert_eq!(prod_i_a.as_words(), a.as_words(), "I·A = A at N=8192");
}

/// Hit the serial-Strassen branch (STRASSEN_SERIAL_MIN = 16384) at
/// rayon-T=1 so the parallel branches don't get taken first. We use
/// a small matmul wrapped in a custom rayon pool with 1 thread to
/// force this. Sanity-checked via A · I = A.
#[cfg(feature = "parallel")]
#[cfg_attr(coverage, ignore)]
#[test]
fn dispatch_serial_strassen_branch() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build single-thread pool");
    pool.install(|| {
        let a = Mzd::random(16384, 16384, 0xABCD);
        let i = Mzd::identity(16384);
        let prod = Mzd::matmul(&a, &i);
        assert_eq!(prod.as_words(), a.as_words(), "A·I = A at N=16384");
    });
}

/// Small (N=256) dispatcher cross-check: every kernel that the
/// dispatcher might pick at this size produces the same answer.
#[test]
fn dispatch_small_size_correctness() {
    let a = Mzd::random(256, 256, 0xBEEF);
    let b = Mzd::random(256, 256, 0xCAFE);
    let got = Mzd::matmul(&a, &b);
    let expect = schoolbook(&a, &b);
    assert_eq!(got.as_words(), expect.as_words());
}
