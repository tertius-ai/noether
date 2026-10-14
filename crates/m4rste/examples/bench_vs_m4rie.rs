// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shootout: m4rste's pure-Rust GF(2^e) kernels vs libm4rie.
//!
//! Times every exposed `Mzed` operation — `matmul`, `rank`, `rref`,
//! `add` — against the reference libm4rie implementation across
//! GF(4) / GF(16) / GF(256) and a sweep of square sizes. Before
//! timing each cell it bit-checks the two implementations agree, so
//! the run doubles as a parity oracle.
//!
//! Build + run (libm4rie + libm4ri must be installed — Debian/Ubuntu
//! `libm4rie-dev` / `libm4ri-dev`):
//!
//! ```text
//! cargo run --release --features validate-libm4rie --example bench_vs_m4rie
//! ```
//!
//! Env knobs:
//!   - `RUNS`   — timed runs per cell, median is reported (default 3)
//!   - `WARMUP` — untimed warmup runs per cell (default 1)
//!
//! m4rste's kernels are the bit-sliced + Karatsuba reduction onto
//! `m4rst` (matmul) and Newton-John–blocked Gaussian elimination
//! with hardware GFNI / NEON / AVX2 SIMD on the field-arithmetic
//! hotspots (rank / rref). Results are measurements for the current machine
//! and must be run under the repository CPU-pinning wrapper.

use std::os::raw::c_int;
use std::time::Instant;
use std::{ffi::c_void, ptr::NonNull};

use m4rste::{Gf2e, Mzed};

// ── libm4rie oracle FFI (see oracle/m4rie_oracle.c) ───────────────

extern "C" {
    fn m4rie_oracle_pair_new(
        minpoly: u64,
        n: usize,
        a_data: *const u16,
        b_data: *const u16,
    ) -> *mut c_void;
    fn m4rie_oracle_pair_free(pair: *mut c_void);
    fn m4rie_oracle_preloaded_matmul(pair: *mut c_void, checksum: *mut u16) -> i32;
    fn m4rie_oracle_preloaded_add(pair: *mut c_void, checksum: *mut u16) -> i32;
    fn m4rie_oracle_preloaded_rank(pair: *mut c_void) -> usize;
    fn m4rie_oracle_preloaded_rref(pair: *mut c_void, checksum: *mut u16) -> usize;
    fn m4rie_oracle_matmul_u16(
        minpoly: u64,
        nrows_a: usize,
        ncols_a: usize,
        a_data: *const u16,
        nrows_b: usize,
        ncols_b: usize,
        b_data: *const u16,
        out_data: *mut u16,
    ) -> c_int;

    fn m4rie_oracle_add_u16(
        minpoly: u64,
        nrows: usize,
        ncols: usize,
        a_data: *const u16,
        b_data: *const u16,
        out_data: *mut u16,
    ) -> c_int;

    fn m4rie_oracle_rank_u16(minpoly: u64, nrows: usize, ncols: usize, data: *const u16) -> usize;

    fn m4rie_oracle_rref_u16(
        minpoly: u64,
        nrows: usize,
        ncols: usize,
        data: *const u16,
        out_data: *mut u16,
    ) -> usize;
}

struct PreloadedPair {
    inner: NonNull<c_void>,
}

impl PreloadedPair {
    fn new(minpoly: u64, n: usize, a: &[u16], b: &[u16]) -> Self {
        assert_eq!(a.len(), n * n);
        assert_eq!(b.len(), n * n);
        // SAFETY: the input slices have exactly `n*n` initialized elements;
        // the C constructor copies them into its own opaque fixture.
        let inner = unsafe { m4rie_oracle_pair_new(minpoly, n, a.as_ptr(), b.as_ptr()) };
        Self {
            inner: NonNull::new(inner).expect("libm4rie fixture allocation failed"),
        }
    }

    fn matmul(&self) -> u16 {
        let mut checksum = 0;
        // SAFETY: `inner` remains live for `self`; checksum is writable.
        let status = unsafe { m4rie_oracle_preloaded_matmul(self.inner.as_ptr(), &mut checksum) };
        assert_eq!(status, 0, "libm4rie preloaded matmul failed");
        checksum
    }

    fn add(&self) -> u16 {
        let mut checksum = 0;
        // SAFETY: `inner` remains live for `self`; checksum is writable.
        let status = unsafe { m4rie_oracle_preloaded_add(self.inner.as_ptr(), &mut checksum) };
        assert_eq!(status, 0, "libm4rie preloaded add failed");
        checksum
    }

    fn rank(&self) -> usize {
        // SAFETY: `inner` remains live for `self`; the C function copies the
        // retained input before destructive elimination.
        let rank = unsafe { m4rie_oracle_preloaded_rank(self.inner.as_ptr()) };
        assert_ne!(rank, usize::MAX, "libm4rie preloaded rank failed");
        rank
    }

    fn rref(&self) -> (usize, u16) {
        let mut checksum = 0;
        // SAFETY: `inner` remains live for `self`; checksum is writable.
        let rank = unsafe { m4rie_oracle_preloaded_rref(self.inner.as_ptr(), &mut checksum) };
        assert_ne!(rank, usize::MAX, "libm4rie preloaded rref failed");
        (rank, checksum)
    }
}

impl Drop for PreloadedPair {
    fn drop(&mut self) {
        // SAFETY: the pointer was returned by `pair_new`, is uniquely owned,
        // and is freed exactly once here.
        unsafe { m4rie_oracle_pair_free(self.inner.as_ptr()) };
    }
}

fn m4rie_matmul(minpoly: u64, n: usize, a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut out = vec![0u16; n * n];
    let rc = unsafe {
        m4rie_oracle_matmul_u16(
            minpoly,
            n,
            n,
            a.as_ptr(),
            n,
            n,
            b.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "m4rie_oracle_matmul_u16 failed");
    out
}

fn m4rie_add(minpoly: u64, n: usize, a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut out = vec![0u16; n * n];
    let rc =
        unsafe { m4rie_oracle_add_u16(minpoly, n, n, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };
    assert_eq!(rc, 0, "m4rie_oracle_add_u16 failed");
    out
}

fn m4rie_rank(minpoly: u64, n: usize, a: &[u16]) -> usize {
    let r = unsafe { m4rie_oracle_rank_u16(minpoly, n, n, a.as_ptr()) };
    assert_ne!(r, usize::MAX, "m4rie_oracle_rank_u16 failed");
    r
}

fn m4rie_rref(minpoly: u64, n: usize, a: &[u16]) -> (usize, Vec<u16>) {
    let mut out = vec![0u16; n * n];
    let rank = unsafe { m4rie_oracle_rref_u16(minpoly, n, n, a.as_ptr(), out.as_mut_ptr()) };
    assert_ne!(rank, usize::MAX, "m4rie_oracle_rref_u16 failed");
    (rank, out)
}

// ── Helpers ───────────────────────────────────────────────────────

/// Deterministic xorshift64* — gives reproducible random matrices.
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

/// `n x n` buffer of GF(2^e) elements, each masked into `[0, 2^e)`.
fn random_buf(n: usize, e: u32, seed: u64) -> Vec<u16> {
    let mask: u16 = if e >= 16 { u16::MAX } else { (1u16 << e) - 1 };
    let mut rng = Rng(seed | 1);
    (0..n * n).map(|_| (rng.next() as u16) & mask).collect()
}

/// Median wall-clock (ms) of `runs` timed calls after `warmup` untimed.
fn time_median<T, F: FnMut() -> T>(warmup: usize, runs: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        std::hint::black_box(f());
    }
    let mut samples = (0..runs)
        .map(|_| {
            let t0 = Instant::now();
            std::hint::black_box(f());
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let warmup = env_usize("WARMUP", 1);
    let runs = env_usize("RUNS", 3).max(1);

    // (label, e, minpoly bit pattern).
    let fields: &[(&str, u32, u64)] = &[
        ("GF(4)", 2, 0b111),     // x^2 + x + 1
        ("GF(16)", 4, 0b1_0011), // x^4 + x + 1
        ("GF(256)", 8, 0x11B),   // x^8 + x^4 + x^3 + x + 1
    ];
    let sizes: &[usize] = &[64, 128, 256];

    println!(
        "m4rste vs libm4rie — warmup={warmup} runs={runs} (median ms reported)\n\
         `m4rie speedup` = t_m4rste / t_m4rie: <1× means m4rste is faster.\n\
         m4rste matmul + GF(256) elimination ship the bit-sliced + GFNI/NEON\n\
         path; small-field elimination is still in NJ-only territory.\n\
         \n\
         Both timed paths retain preloaded inputs and allocate the operation's\n\
         native result. Buffer marshalling is excluded; parity is checked\n\
         separately through the independent flat-buffer oracle.\n"
    );

    let mut parity_fails = 0usize;

    for &(fname, e, minpoly) in fields {
        println!("═══ {fname}  e={e}  minpoly={minpoly:#x} ═══════════════════════════");
        println!(
            "  {:<8} {:>5}  {:>12}  {:>12}  {:>14}  {:>7}",
            "op", "N", "m4rste (ms)", "m4rie (ms)", "m4rie speedup", "parity"
        );
        let ctx = Gf2e::new(minpoly);

        for &n in sizes {
            let a_buf = random_buf(n, e, 0xA11CE ^ (n as u64) ^ ((e as u64) << 32));
            let b_buf = random_buf(n, e, 0xB0B ^ (n as u64) ^ ((e as u64) << 40));
            let a = Mzed::from_u16_buffer(ctx, n, n, &a_buf);
            let b = Mzed::from_u16_buffer(ctx, n, n, &b_buf);
            let m4rie = PreloadedPair::new(minpoly, n, &a_buf, &b_buf);

            // ── matmul ────────────────────────────────────────────
            let rust_c = Mzed::matmul(&a, &b).to_u16_buffer();
            let m4rie_c = m4rie_matmul(minpoly, n, &a_buf, &b_buf);
            let matmul_ok = rust_c == m4rie_c;
            let t_rust = time_median(warmup, runs, || Mzed::matmul(&a, &b));
            let t_m4rie = time_median(warmup, runs, || m4rie.matmul());
            report("matmul", n, t_rust, t_m4rie, matmul_ok, &mut parity_fails);

            // ── add ───────────────────────────────────────────────
            let rust_s = Mzed::add(&a, &b).to_u16_buffer();
            let m4rie_s = m4rie_add(minpoly, n, &a_buf, &b_buf);
            let add_ok = rust_s == m4rie_s;
            let t_rust = time_median(warmup, runs, || Mzed::add(&a, &b));
            let t_m4rie = time_median(warmup, runs, || m4rie.add());
            report("add", n, t_rust, t_m4rie, add_ok, &mut parity_fails);

            // ── rank ──────────────────────────────────────────────
            let rust_rank = a.rank();
            let m4rie_rk = m4rie_rank(minpoly, n, &a_buf);
            let rank_ok = rust_rank == m4rie_rk;
            let t_rust = time_median(warmup, runs, || a.rank());
            let t_m4rie = time_median(warmup, runs, || m4rie.rank());
            report("rank", n, t_rust, t_m4rie, rank_ok, &mut parity_fails);

            // ── rref ──────────────────────────────────────────────
            // RREF is unique over a field, so m4rste and m4rie must
            // agree on both the rank and the reduced form bit-for-bit.
            // `Mzed::rref` returns `(Mzed, rank)`.
            let (rust_rref, rust_rrank) = a.rref();
            let (m4rie_rrank, m4rie_rref_buf) = m4rie_rref(minpoly, n, &a_buf);
            let rref_ok = rust_rrank == m4rie_rrank && rust_rref.to_u16_buffer() == m4rie_rref_buf;
            let t_rust = time_median(warmup, runs, || a.rref());
            let t_m4rie = time_median(warmup, runs, || m4rie.rref());
            report("rref", n, t_rust, t_m4rie, rref_ok, &mut parity_fails);
        }
        println!();
    }

    if parity_fails == 0 {
        println!("parity: all cells bit-equal to libm4rie ✓");
    } else {
        println!("parity: {parity_fails} cell(s) DISAGREE with libm4rie ✗");
        std::process::exit(1);
    }
}

fn report(op: &str, n: usize, t_rust: f64, t_m4rie: f64, parity_ok: bool, fails: &mut usize) {
    let speedup = if t_rust > 0.0 {
        t_rust / t_m4rie.max(1e-9)
    } else {
        0.0
    };
    let parity = if parity_ok {
        "ok"
    } else {
        *fails += 1;
        "MISMATCH"
    };
    println!(
        "  {:<8} {:>5}  {:>12.3}  {:>12.3}  {:>12.1}×  {:>7}",
        op, n, t_rust, t_m4rie, speedup, parity
    );
}
