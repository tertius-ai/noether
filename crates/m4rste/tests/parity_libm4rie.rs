// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! libm4rie parity tests — gated behind the `validate-libm4rie`
//! feature.
//!
//! ## Why these exist
//!
//! Mirrors m4rst's `tests/parity_libm4ri.rs`: the per-op, per-shape,
//! per-field bit-equality unit tests that pin m4rste against the GPL
//! reference (`libm4rie`) at small + dispatch-relevant sizes. The
//! existing `examples/bench_vs_m4rie.rs` does parity-during-bench
//! (catches gross regressions) but only at the three bench shapes
//! per field; a proper test suite runs faster, exercises edge shapes,
//! and lives under `cargo test` for per-commit CI.
//!
//! ## How to run
//!
//! ```text
//! M4RSTE_VALIDATE_LIBM4RIE=1 \
//! LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
//!   cargo test --release --features validate-libm4rie --test parity_libm4rie \
//!   -- --test-threads=1
//! ```
//!
//! Setting the env var is required even with the Cargo feature on —
//! same pattern as m4rst. Without the env var, every test bails
//! vacuously, letting CI machines compile the feature without
//! requiring `libm4rie`/`libm4ri` at link time on every node.
//!
//! `--test-threads=1` is required: libm4rie's `gf2e_init` /
//! `gf2e_free` share process-wide state (precomputed log/antilog
//! tables) without locking, so concurrent test invocation triggers
//! heap corruption (`malloc_consolidate(): unaligned fastbin chunk`).
//! libm4ri/libm4rie's thread-safety is `--enable-thread-safe` opt-in
//! at libm4ri's configure time; the system `libm4rie-dev` packages
//! don't get that flag.
//!
//! ## Coverage
//!
//! For each field in {GF(4), GF(16), GF(256)} and each shape in a
//! small sweep:
//!   - `Mzed::add` bit-equal to `mzed_add` (XOR over F_{2^e}).
//!   - `Mzed::matmul` bit-equal to `mzed_mul`.
//!   - `Mzed::rank` matches `mzed_echelonize`-rank (RREF unique;
//!     rank is single integer).
//!   - `Mzed::echelonize(full=true)` bit-equal to `mzed_echelonize`.
//!   - `Mzed::pluq` rank matches `mzed_ple`'s rank (rank-only — see
//!     `pluq_rank_matches_libm4rie`'s doc comment for why P/L/U/Q
//!     themselves aren't bit-compared against the oracle).
//!
//! Edge shapes include 1×1, n×1, 1×n, and the dispatch boundaries
//! (around N=64 where bit-packed slice cling crosses a word boundary
//! and the Newton-John blocked elimination kicks in for rref).

use std::os::raw::c_int;

use m4rste::{Gf2e, Mzed};

// ── libm4rie oracle FFI ───────────────────────────────────────────
//
// Same shim as `examples/bench_vs_m4rie.rs` uses; declared here too
// so the test binary links it directly under the `validate-libm4rie`
// feature.

extern "C" {
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

    fn m4rie_oracle_ple_rank_u16(
        minpoly: u64,
        nrows: usize,
        ncols: usize,
        data: *const u16,
    ) -> usize;
}

// ── Test-only helpers ─────────────────────────────────────────────

fn validation_enabled() -> bool {
    matches!(
        std::env::var("M4RSTE_VALIDATE_LIBM4RIE").as_deref(),
        Ok("1")
    )
}

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

/// `nrows × ncols` buffer of GF(2^e) elements, each masked into
/// `[0, 2^e)`. Matches `examples/bench_vs_m4rie.rs::random_buf`'s
/// layout so the FFI shim sees the same bit-pattern.
fn random_buf(nrows: usize, ncols: usize, e: u32, seed: u64) -> Vec<u16> {
    let mask: u16 = if e >= 16 { u16::MAX } else { (1u16 << e) - 1 };
    let mut rng = Rng(seed | 1);
    (0..nrows * ncols)
        .map(|_| (rng.next() as u16) & mask)
        .collect()
}

/// `(label, e, minpoly bit pattern)` — the three fields the bench
/// example covers, plus the ground-truth seeds m4rste's kernels
/// have to handle without going to the FLINT `fq_zech` backstop.
const FIELDS: &[(&str, u32, u64)] = &[
    ("GF(4)", 2, 0b111),     // x^2 + x + 1 (primitive)
    ("GF(16)", 4, 0b1_0011), // x^4 + x + 1
    ("GF(256)", 8, 0x11B),   // AES poly x^8 + x^4 + x^3 + x + 1
];

/// Shapes used by the per-op parity tests. Covers:
/// * Degenerate 1×1, tiny non-square (3×7, 7×3, 8×9).
/// * Near-word boundaries 63 / 64 / 65 (M4RIE's mzed_t packs `e` bits
///   per element so a row's bit width is `ncols × e`; the slice/cling
///   bit-deinterleave hits its first-strip boundary at column 64/e).
/// * 100×100 + 128×128 to exercise the M4RM path for matmul.
///
/// Per-op runtimes are seconds total across all (field, shape)
/// combinations; kept small enough that the suite stays under a
/// minute on a release build.
const PARITY_SHAPES: &[(usize, usize, u64)] = &[
    (1, 1, 0xE001),
    (3, 7, 0xE002),
    (7, 3, 0xE003),
    (8, 9, 0xE004),
    (16, 16, 0xE005),
    (32, 33, 0xE006),
    (63, 64, 0xE007),
    (64, 64, 0xE008),
    (64, 65, 0xE009),
    (65, 64, 0xE00A),
    (100, 100, 0xE00B),
    (128, 128, 0xE00C),
];

/// Square shapes for matmul, rank, rref. Same coverage rationale as
/// `PARITY_SHAPES`'s square entries plus a 256×256 cell that hits
/// matmul's Newton-John blocked elimination crossover and rref's
/// dispatcher.
const SQUARE_SHAPES: &[(usize, u64)] = &[
    (1, 0xF001),
    (8, 0xF002),
    (16, 0xF003),
    (32, 0xF004),
    (63, 0xF005),
    (64, 0xF006),
    (65, 0xF007),
    (100, 0xF008),
    (128, 0xF009),
    (256, 0xF00A),
];

// ── Per-op parity tests ──────────────────────────────────────────

#[test]
fn add_matches_libm4rie() {
    if !validation_enabled() {
        eprintln!("skipping (M4RSTE_VALIDATE_LIBM4RIE not set)");
        return;
    }
    for &(fname, e, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &(r, c, seed) in PARITY_SHAPES {
            let a_buf = random_buf(r, c, e, seed);
            let b_buf = random_buf(r, c, e, seed ^ 0xDEAD_BEEF);

            let a = Mzed::from_u16_buffer(ctx, r, c, &a_buf);
            let b = Mzed::from_u16_buffer(ctx, r, c, &b_buf);
            let rust = Mzed::add(&a, &b).to_u16_buffer();

            let mut m4rie_out = vec![0u16; r * c];
            let rc = unsafe {
                m4rie_oracle_add_u16(
                    minpoly,
                    r,
                    c,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    m4rie_out.as_mut_ptr(),
                )
            };
            assert_eq!(rc, 0, "m4rie add returned {rc} for {fname} {r}x{c}");
            assert_eq!(
                rust, m4rie_out,
                "{fname} add({r}×{c}, seed={seed:#x}) disagrees"
            );
        }
    }
}

#[test]
fn matmul_matches_libm4rie() {
    if !validation_enabled() {
        eprintln!("skipping (M4RSTE_VALIDATE_LIBM4RIE not set)");
        return;
    }
    for &(fname, e, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &(n, seed) in SQUARE_SHAPES {
            let a_buf = random_buf(n, n, e, seed);
            let b_buf = random_buf(n, n, e, seed ^ 0xCAFE_BABE);

            let a = Mzed::from_u16_buffer(ctx, n, n, &a_buf);
            let b = Mzed::from_u16_buffer(ctx, n, n, &b_buf);
            let rust = Mzed::matmul(&a, &b).to_u16_buffer();

            let mut m4rie_out = vec![0u16; n * n];
            let rc = unsafe {
                m4rie_oracle_matmul_u16(
                    minpoly,
                    n,
                    n,
                    a_buf.as_ptr(),
                    n,
                    n,
                    b_buf.as_ptr(),
                    m4rie_out.as_mut_ptr(),
                )
            };
            assert_eq!(rc, 0, "m4rie matmul returned {rc} for {fname} {n}x{n}");
            assert_eq!(
                rust, m4rie_out,
                "{fname} matmul({n}×{n}, seed={seed:#x}) disagrees"
            );
        }
    }
}

#[test]
fn rank_matches_libm4rie() {
    if !validation_enabled() {
        eprintln!("skipping (M4RSTE_VALIDATE_LIBM4RIE not set)");
        return;
    }
    for &(fname, e, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &(r, c, seed) in PARITY_SHAPES {
            let a_buf = random_buf(r, c, e, seed);
            let a = Mzed::from_u16_buffer(ctx, r, c, &a_buf);
            let rust_rank = a.rank();
            let m4rie_rank = unsafe { m4rie_oracle_rank_u16(minpoly, r, c, a_buf.as_ptr()) };
            assert_ne!(
                m4rie_rank,
                usize::MAX,
                "m4rie rank failed for {fname} {r}x{c}"
            );
            assert_eq!(
                rust_rank, m4rie_rank,
                "{fname} rank({r}×{c}, seed={seed:#x}) disagrees: \
                 rust={rust_rank} m4rie={m4rie_rank}"
            );
        }
    }
}

#[test]
fn rref_matches_libm4rie() {
    if !validation_enabled() {
        eprintln!("skipping (M4RSTE_VALIDATE_LIBM4RIE not set)");
        return;
    }
    for &(fname, e, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &(r, c, seed) in PARITY_SHAPES {
            let a_buf = random_buf(r, c, e, seed);

            // m4rste path — `rref()` returns the RREF matrix + rank.
            let a = Mzed::from_u16_buffer(ctx, r, c, &a_buf);
            let (rust_mat, rust_rank) = a.rref();
            let rust_out = rust_mat.to_u16_buffer();

            // libm4rie path via the oracle.
            let mut m4rie_out = vec![0u16; r * c];
            let m4rie_rank = unsafe {
                m4rie_oracle_rref_u16(minpoly, r, c, a_buf.as_ptr(), m4rie_out.as_mut_ptr())
            };
            assert_ne!(
                m4rie_rank,
                usize::MAX,
                "m4rie rref failed for {fname} {r}x{c}"
            );
            assert_eq!(
                rust_rank, m4rie_rank,
                "{fname} rref({r}×{c}, seed={seed:#x}) rank disagrees: \
                 rust={rust_rank} m4rie={m4rie_rank}"
            );
            // RREF is canonical, so the matrices should bit-equal.
            assert_eq!(
                rust_out, m4rie_out,
                "{fname} rref({r}×{c}, seed={seed:#x}) matrix disagrees"
            );
        }
    }
}

/// `Mzed::pluq` vs `mzed_ple`, rank-only — mirrors `m4rst`'s own
/// `pluq_rank_matches_libm4ri` precedent. PLUQ/PLE factorizations
/// aren't unique across implementations (pivot search order,
/// tie-breaking), so there's no bit-equality target for P/L/U/Q
/// themselves; `Mzed::pluq` is structurally self-checked against
/// `P·L·U·Q == A` elsewhere (tests/pluq.rs, tests/proptest.rs). Rank
/// is the one well-defined, checkable cross-implementation invariant.
#[test]
fn pluq_rank_matches_libm4rie() {
    if !validation_enabled() {
        eprintln!("skipping (M4RSTE_VALIDATE_LIBM4RIE not set)");
        return;
    }
    for &(fname, e, minpoly) in FIELDS {
        let ctx = Gf2e::new(minpoly);
        for &(r, c, seed) in PARITY_SHAPES {
            let a_buf = random_buf(r, c, e, seed);
            let a = Mzed::from_u16_buffer(ctx, r, c, &a_buf);
            let (_p, _l, _u, _q, rust_rank) = a.pluq();
            let m4rie_rank = unsafe { m4rie_oracle_ple_rank_u16(minpoly, r, c, a_buf.as_ptr()) };
            assert_ne!(
                m4rie_rank,
                usize::MAX,
                "m4rie ple failed for {fname} {r}x{c}"
            );
            assert_eq!(
                rust_rank, m4rie_rank,
                "{fname} pluq({r}×{c}, seed={seed:#x}) rank disagrees: \
                 rust={rust_rank} m4rie={m4rie_rank}"
            );
        }
    }
}
