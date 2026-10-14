// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Determinism of the parallel matmul kernels.
//!
//! m4rst's parallel matmul kernels assign disjoint output row ranges
//! to worker threads and XOR-accumulate into pre-zeroed destinations.
//! Because F_2 has no associativity hazard and the output regions are
//! disjoint by construction, the result is mathematically independent
//! of thread count and execution order.
//!
//! This test pins that property: the same `(seed, shape)` matmul under
//! pools of T ∈ {1, 2, 4, 16} threads must produce bit-equal output,
//! both at the public dispatcher entry point (`Mzd::matmul`) and at
//! the individual parallel kernels (`matmul_strassen_parallel_leaf`,
//! `matmul_m4rm_ntables_mblocked_parallel`). A future change that
//! introduces a races-but-usually-wins reduction strategy in any of
//! these kernels would fail this test on the first run.
//!
//! The tests are no-ops without `--features parallel`; the harness
//! still compiles so that `cargo test` without the feature passes
//! trivially.

#[cfg(feature = "parallel")]
use m4rst::matmul::m4rm::matmul_m4rm_ntables_mblocked_parallel;
#[cfg(feature = "parallel")]
use m4rst::matmul::strassen::matmul_strassen_parallel_leaf_with_threshold;
#[cfg(feature = "parallel")]
use m4rst::Mzd;

#[cfg(feature = "parallel")]
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 16];

/// Run `f` inside a custom rayon pool with `nthreads` worker threads.
/// Each call builds a fresh pool so the run is independent of any
/// ambient global pool state (test order, prior `cargo test`
/// invocations, etc.).
#[cfg(feature = "parallel")]
fn under_pool<F, T>(nthreads: usize, f: F) -> T
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap_or_else(|_| panic!("build rayon pool with {} threads", nthreads));
    pool.install(f)
}

/// `Mzd::matmul` at N=8192 hits the outer-parallel M-blocked NTABLES
/// branch at T ≥ 2 and the serial NTABLES branch at T = 1. All four
/// thread counts must produce the same output words.
#[cfg(feature = "parallel")]
#[cfg_attr(coverage, ignore)]
#[test]
fn dispatcher_matmul_is_thread_count_independent_at_n_8192() {
    let a = Mzd::random(8192, 8192, 0xD27E_8192);
    let b = Mzd::random(8192, 8192, 0xD53E_8192);

    let outputs: Vec<(usize, Mzd)> = THREAD_COUNTS
        .iter()
        .map(|&t| (t, under_pool(t, || Mzd::matmul(&a, &b))))
        .collect();

    let (ref_t, ref_out) = &outputs[0];
    let ref_words = ref_out.as_words();
    for (t, out) in &outputs[1..] {
        assert_eq!(
            out.as_words(),
            ref_words,
            "Mzd::matmul at T={} disagrees with T={} (N=8192)",
            t,
            ref_t,
        );
    }
}

/// The distributed SUMMA leaf shape uses the outer-row-parallel narrow-output
/// dispatch at T ≥ 2 and the serial NTABLES path at T = 1.
#[cfg(feature = "parallel")]
#[test]
fn dispatcher_narrow_output_is_thread_count_independent() {
    let a = Mzd::random(8192, 4096, 0xD27E_0064);
    let b = Mzd::random(4096, 64, 0xD53E_0064);

    let outputs: Vec<(usize, Mzd)> = THREAD_COUNTS
        .iter()
        .map(|&t| (t, under_pool(t, || Mzd::matmul(&a, &b))))
        .collect();

    let (ref_t, ref_out) = &outputs[0];
    for (t, out) in &outputs[1..] {
        assert_eq!(
            out.as_words(),
            ref_out.as_words(),
            "narrow-output Mzd::matmul at T={} disagrees with T={}",
            t,
            ref_t,
        );
    }
}

/// `Mzd::matmul` at N=16384 hits the Strassen-par-leaf path (depth 1
/// at T=16, depth 2 at T<16) at T ≥ 2 and the serial multi-level
/// Strassen branch at T = 1. The thread-count-aware Strassen-depth
/// switch is the highest-risk surface for an order-dependent bug; pin
/// it explicitly.
#[cfg(feature = "parallel")]
#[cfg_attr(coverage, ignore)]
#[test]
fn dispatcher_matmul_is_thread_count_independent_at_n_16384() {
    let a = Mzd::random(16384, 16384, 0xD27E_1638);
    let b = Mzd::random(16384, 16384, 0xD53E_1638);

    let outputs: Vec<(usize, Mzd)> = THREAD_COUNTS
        .iter()
        .map(|&t| (t, under_pool(t, || Mzd::matmul(&a, &b))))
        .collect();

    let (ref_t, ref_out) = &outputs[0];
    let ref_words = ref_out.as_words();
    for (t, out) in &outputs[1..] {
        assert_eq!(
            out.as_words(),
            ref_words,
            "Mzd::matmul at T={} disagrees with T={} (N=16384)",
            t,
            ref_t,
        );
    }
}

/// Direct-call determinism on the outer-mblocked-parallel NTABLES
/// kernel — bypasses the dispatcher so a future change that moves the
/// dispatch threshold away from this kernel can't accidentally hide a
/// kernel-level regression.
#[cfg(feature = "parallel")]
#[cfg_attr(coverage, ignore)]
#[test]
fn matmul_m4rm_ntables_mblocked_parallel_is_thread_count_independent() {
    let a = Mzd::random(8192, 8192, 0xD27E_3000);
    let b = Mzd::random(8192, 8192, 0xD53E_3000);
    let m_block = 1024;

    let outputs: Vec<(usize, Mzd)> = THREAD_COUNTS
        .iter()
        .map(|&t| {
            (
                t,
                under_pool(t, || matmul_m4rm_ntables_mblocked_parallel(&a, &b, m_block)),
            )
        })
        .collect();

    let (ref_t, ref_out) = &outputs[0];
    let ref_words = ref_out.as_words();
    for (t, out) in &outputs[1..] {
        assert_eq!(
            out.as_words(),
            ref_words,
            "matmul_m4rm_ntables_mblocked_parallel at T={} disagrees with T={} (N=8192, mb={})",
            t,
            ref_t,
            m_block,
        );
    }
}

/// Direct-call determinism on the Strassen-with-parallel-leaf kernel.
/// Same reasoning as the mblocked test: hits the kernel even if the
/// dispatcher routes elsewhere in the future.
#[cfg(feature = "parallel")]
#[cfg_attr(coverage, ignore)]
#[test]
fn matmul_strassen_parallel_leaf_is_thread_count_independent() {
    let a = Mzd::random(8192, 8192, 0xD27E_4000);
    let b = Mzd::random(8192, 8192, 0xD53E_4000);
    let threshold = 4096; // one level of recursion → 7 leaves of 4096

    let outputs: Vec<(usize, Mzd)> = THREAD_COUNTS
        .iter()
        .map(|&t| {
            (
                t,
                under_pool(t, || {
                    matmul_strassen_parallel_leaf_with_threshold(&a, &b, threshold)
                }),
            )
        })
        .collect();

    let (ref_t, ref_out) = &outputs[0];
    let ref_words = ref_out.as_words();
    for (t, out) in &outputs[1..] {
        assert_eq!(
            out.as_words(),
            ref_words,
            "matmul_strassen_parallel_leaf at T={} disagrees with T={} (N=8192, thr={})",
            t,
            ref_t,
            threshold,
        );
    }
}
