// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! libm4ri parity tests — gated behind the `validate-libm4ri` feature.
//!
//! ## Why these exist
//!
//! Until m4rst is at tier 4, libm4ri is the correctness oracle (see
//! `AGENTS.md`). These tests run the same input through an `Mzd` op and
//! the corresponding libm4ri op and assert **bit-equality** of the
//! result.
//!
//! ## How the oracle is wired
//!
//! libm4ri's row + bit accessors (`mzd_row`, `mzd_read_bit`, …) are
//! `static inline` in the headers — no exported symbol a raw Rust FFI
//! boundary could call. Rather than hand-mirror the `mzd_t` struct in
//! Rust (which silently broke when libm4ri's layout drifted — the
//! `size_of == 64` assert passed for both the old and new layouts),
//! the oracle is a thin C shim, `oracle/m4ri_oracle.c`, that
//! `#include`s the real header. The C compiler therefore sees
//! whatever `mzd_t` the linked libm4ri actually uses, and `mzd_row` is
//! the genuine inline. The shim exposes flat extern-C entry points
//! taking packed `u64` row-major buffers — exactly `Mzd::as_words()`'s
//! layout (the universal F_2 bit-packing convention), pinned by
//! `byte_buffer_matches_libm4ri`.
//!
//! ## How to run
//!
//! ```text
//! M4RST_VALIDATE_LIBM4RI=1 \
//!   cargo test --release --features validate-libm4ri \
//!     --test parity_libm4ri -- --test-threads=1
//! ```
//!
//! The runtime `M4RST_VALIDATE_LIBM4RI=1` gate is intentional: enabling
//! the Cargo feature compiles + links the C shim, but each test bails
//! early (prints a skip-line, passes vacuously) unless the env var is
//! set. This lets CI compile the feature on nodes that then choose
//! whether to run the asserts. `--test-threads=1` is used because the
//! system `libm4ri-dev` is built without `--enable-thread-safe` and
//! lazily initialises process-wide tables.
//!
//! On Debian/Ubuntu the system `libm4ri-dev` package is enough; point
//! `M4RI_PREFIX` at a custom libm4ri install to override.

use std::os::raw::c_int;

use m4rst::{MatrixWrite, Mzd};

// ---- C-shim oracle FFI (see oracle/m4ri_oracle.c) ----------------------
//
// Every buffer is row-major packed: `nrows` rows, each `ceil(ncols/64)`
// u64 words. `Mzd::as_words()` already is exactly this layout.

extern "C" {
    /// `mzd_set_ui(M, value)`: value 0 → zero matrix, 1 → identity.
    fn m4ri_oracle_set_ui(nrows: usize, ncols: usize, value: u32, out: *mut u64) -> c_int;
    /// Load a packed buffer into libm4ri and read it straight back.
    fn m4ri_oracle_roundtrip(nrows: usize, ncols: usize, data: *const u64, out: *mut u64) -> c_int;
    /// `OUT = A · B`. `use_strassen`: 0 → `mzd_mul_naive`, 1 → `mzd_mul`.
    fn m4ri_oracle_matmul(
        nra: usize,
        nca: usize,
        a: *const u64,
        nrb: usize,
        ncb: usize,
        b: *const u64,
        out: *mut u64,
        use_strassen: c_int,
    ) -> c_int;
    /// `OUT = A^T` (`OUT` is `ncols × nrows`).
    fn m4ri_oracle_transpose(nrows: usize, ncols: usize, data: *const u64, out: *mut u64) -> c_int;
    /// `OUT = A + B` (element-wise XOR over F_2).
    fn m4ri_oracle_add(
        nrows: usize,
        ncols: usize,
        a: *const u64,
        b: *const u64,
        out: *mut u64,
    ) -> c_int;
    /// In-place echelonize. `full`: 1 → RREF, 0 → REF. Returns the rank
    /// (≥ 0) or -1; writes the echelon form into `out`.
    fn m4ri_oracle_echelonize(
        nrows: usize,
        ncols: usize,
        data: *const u64,
        full: c_int,
        out: *mut u64,
    ) -> c_int;
    /// Rank via libm4ri's production PLUQ. Returns the rank (≥ 0) or -1.
    fn m4ri_oracle_pluq_rank(nrows: usize, ncols: usize, data: *const u64) -> c_int;
}

/// Words per row for a matrix with `cols` columns. Matches
/// `Mzd::width()`; needed when sizing an output buffer before the
/// result `Mzd` exists.
fn wwords(cols: usize) -> usize {
    cols.div_ceil(64)
}

/// Whether the env var is set. Without it, tests skip vacuously.
fn validation_enabled() -> bool {
    matches!(std::env::var("M4RST_VALIDATE_LIBM4RI").as_deref(), Ok("1"))
}

// ---- safe wrappers over the shim entry points --------------------------

fn oracle_set_ui(rows: usize, cols: usize, value: u32) -> Vec<u64> {
    let mut out = vec![0u64; rows * wwords(cols)];
    let rc = unsafe { m4ri_oracle_set_ui(rows, cols, value, out.as_mut_ptr()) };
    assert_eq!(rc, 0, "m4ri_oracle_set_ui({rows},{cols},{value}) failed");
    out
}

fn oracle_roundtrip(m: &Mzd) -> Vec<u64> {
    let mut out = vec![0u64; m.rows() * m.width()];
    let rc = unsafe {
        m4ri_oracle_roundtrip(m.rows(), m.cols(), m.as_words().as_ptr(), out.as_mut_ptr())
    };
    assert_eq!(rc, 0, "m4ri_oracle_roundtrip failed");
    out
}

fn oracle_matmul(a: &Mzd, b: &Mzd, use_strassen: bool) -> Vec<u64> {
    let mut out = vec![0u64; a.rows() * wwords(b.cols())];
    let rc = unsafe {
        m4ri_oracle_matmul(
            a.rows(),
            a.cols(),
            a.as_words().as_ptr(),
            b.rows(),
            b.cols(),
            b.as_words().as_ptr(),
            out.as_mut_ptr(),
            use_strassen as c_int,
        )
    };
    assert_eq!(rc, 0, "m4ri_oracle_matmul failed");
    out
}

fn oracle_transpose(a: &Mzd) -> Vec<u64> {
    // The transpose is `cols × rows`.
    let mut out = vec![0u64; a.cols() * wwords(a.rows())];
    let rc = unsafe {
        m4ri_oracle_transpose(a.rows(), a.cols(), a.as_words().as_ptr(), out.as_mut_ptr())
    };
    assert_eq!(rc, 0, "m4ri_oracle_transpose failed");
    out
}

fn oracle_add(a: &Mzd, b: &Mzd) -> Vec<u64> {
    let mut out = vec![0u64; a.rows() * a.width()];
    let rc = unsafe {
        m4ri_oracle_add(
            a.rows(),
            a.cols(),
            a.as_words().as_ptr(),
            b.as_words().as_ptr(),
            out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0, "m4ri_oracle_add failed");
    out
}

/// Returns `(rank, echelon-form words)`.
fn oracle_echelonize(m: &Mzd, full: bool) -> (usize, Vec<u64>) {
    let mut out = vec![0u64; m.rows() * m.width()];
    let rc = unsafe {
        m4ri_oracle_echelonize(
            m.rows(),
            m.cols(),
            m.as_words().as_ptr(),
            full as c_int,
            out.as_mut_ptr(),
        )
    };
    assert!(rc >= 0, "m4ri_oracle_echelonize failed");
    (rc as usize, out)
}

fn oracle_pluq_rank(m: &Mzd) -> usize {
    let rc = unsafe { m4ri_oracle_pluq_rank(m.rows(), m.cols(), m.as_words().as_ptr()) };
    assert!(rc >= 0, "m4ri_oracle_pluq_rank failed");
    rc as usize
}

/// Word-level equality with a useful first-difference message. m4rst
/// and libm4ri share the packed row layout (pinned by
/// `byte_buffer_matches_libm4ri`); both zero the high tail past
/// `ncols`, so comparing every `width` word is exact.
fn assert_words_eq(label: &str, expected: &[u64], actual: &[u64]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label}: word-count mismatch ({} vs {})",
        expected.len(),
        actual.len()
    );
    if expected != actual {
        let i = expected
            .iter()
            .zip(actual.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "{label}: word {i} differs (m4rst=0x{:016x}, libm4ri=0x{:016x})",
            expected[i], actual[i]
        );
    }
}

// ---- construction parity ----------------------------------------------

#[test]
fn zero_matrix_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(r, c) in &[(1usize, 1usize), (8, 64), (8, 65), (33, 130), (100, 100)] {
        let m = Mzd::new(r, c);
        assert_words_eq(
            &format!("zero({r}, {c})"),
            m.as_words(),
            &oracle_set_ui(r, c, 0),
        );
    }
}

#[test]
fn identity_matrix_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &n in &[1usize, 64, 65, 130, 256] {
        let m = Mzd::identity(n);
        // mzd_set_ui(M, 1) yields the identity.
        assert_words_eq(
            &format!("identity({n})"),
            m.as_words(),
            &oracle_set_ui(n, n, 1),
        );
    }
}

#[test]
fn byte_buffer_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    // Build a deterministic {0,1} pattern, load it into m4rst, and run
    // the packed words through libm4ri and straight back. Equality
    // proves m4rst's `as_words()` layout is the one the oracle's
    // buffer convention assumes — the contract every op test relies on.
    for &(r, c, seed) in &[(8usize, 64usize, 1u64), (17, 130, 2), (64, 200, 3)] {
        let bytes: Vec<u8> = (0..r)
            .flat_map(|i| {
                (0..c).map(move |j| {
                    let h = (seed
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                        .wrapping_add((j as u64).wrapping_mul(0x94D0_49BB_1331_11EB)))
                        >> 33;
                    (h & 1) as u8
                })
            })
            .collect();
        let m = Mzd::from_byte_buffer(r, c, &bytes);
        assert_words_eq(
            &format!("bytes({r}, {c}, {seed})"),
            m.as_words(),
            &oracle_roundtrip(&m),
        );
        // And round-trip: to_byte_buffer must equal the input.
        assert_eq!(m.to_byte_buffer(), bytes, "to_byte_buffer round-trip");
    }
}

// ---- op parity: matmul, transpose, xor, echelonize ---------------------

/// Random-shape sweep used by the op-level parity tests below.
/// Covers cols % 64 ∈ {0, ≠ 0, near-radix} on each side, and shapes
/// hitting all three dispatch paths in `Mzd::matmul`: naive (small),
/// M4RM (mid), and Strassen-on-M4RM (every dim ≥ 1024).
const OP_SHAPES: &[(usize, usize, usize, u64)] = &[
    // (rows_a, inner, cols_b, seed)
    (1, 1, 1, 0xA001),
    (8, 64, 64, 0xA002), // below M4RM threshold (rows_a < 32) — naive path
    (8, 65, 64, 0xA003),
    (8, 64, 65, 0xA004),
    (17, 33, 130, 0xA005),
    (64, 128, 200, 0xA006), // above M4RM threshold — exercises the M4RM kernel
    (100, 100, 100, 0xA007),
    (256, 256, 256, 0xA008), // square at a size where M4RM should dominate
];

/// Larger square shapes used only by the matmul parity test, where the
/// Strassen dispatch threshold (every dim ≥ 1024) is in play. Kept out
/// of OP_SHAPES so transpose / xor / echelonize parity stays cheap.
const MATMUL_LARGE_SHAPES: &[(usize, u64)] = &[
    (1024, 0xB001), // exactly at threshold — Strassen kicks in
    (1280, 0xB002), // multi-level (1280/2 = 640 < 1024 so only 1 level)
];

/// Production-dispatch-path matmul shapes — Ns above the thresholds
/// in `src/matmul/mod.rs`. Each entry pins a distinct branch of the
/// `Mzd::matmul` parallel dispatcher when `--features parallel` is
/// enabled and rayon's default 16-thread pool is in use:
///
/// * `6400`, `8192`, `8320`, `16384` — outer-parallel M-blocked NTABLES
///   (`PAR_MBLOCK_MIN = 6144 ≤ N < PAR_INNER_MIN = 24576`).
/// * `24832` — inner-parallel NTABLES (just above `PAR_INNER_MIN`).
///
/// All five Ns are multiples of 128. The shim drives libm4ri's
/// production `mzd_mul` (Strassen-on-M4RM) here, not the O(n³)
/// `mzd_mul_naive`; the small-N tests above already pin both libraries
/// against the cubic ground truth, so `mzd_mul` is a valid oracle.
const MATMUL_DISPATCH_SHAPES: &[(usize, u64)] = &[
    (6400, 0xC001),  // PAR_MBLOCK, first N above the parallel threshold
    (8192, 0xC002),  // PAR_MBLOCK, bench-canonical size
    (8320, 0xC003),  // PAR_MBLOCK, off-square-power-of-two
    (16384, 0xC004), // PAR_MBLOCK, largest within the first dispatch tier
    (24832, 0xC005), // PAR_INNER, first N above the inner-parallel threshold
];

/// Square shapes for the `Mzd::ple_rank` ↔ `mzd_pluq` rank parity
/// sweep. Bracketing the M4R-block fast-path crossover
/// (`min(m,n) > 256`) on the low end and going up to the
/// ship-blocker sizes on the high end.
const PLE_RANK_SHAPES: &[(usize, u64)] = &[
    (256, 0xD001),
    (512, 0xD002),
    (640, 0xD003),  // first multi-super-strip size that broke my fast path
    (1024, 0xD004), // 16 super-strips
    (2048, 0xD005),
    (4096, 0xD006), // v1.0 ship-blocker lower bound
];

#[test]
fn matmul_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    // Mid-size sweep — covers naive + M4RM dispatch paths. Oracle is
    // the cubic ground truth `mzd_mul_naive`.
    for &(r, k, c, seed) in OP_SHAPES {
        let a = Mzd::random(r, k, seed);
        let b = Mzd::random(k, c, seed ^ 0xDEAD_BEEF);
        let prod = Mzd::matmul(&a, &b);
        assert_words_eq(
            &format!("matmul({r}×{k} · {k}×{c}, seed={seed})"),
            prod.as_words(),
            &oracle_matmul(&a, &b, false),
        );
    }
    // Large square sweep — drives the Strassen-on-M4RM dispatch path.
    for &(n, seed) in MATMUL_LARGE_SHAPES {
        let a = Mzd::random(n, n, seed);
        let b = Mzd::random(n, n, seed ^ 0xCAFE_BABE);
        let prod = Mzd::matmul(&a, &b);
        assert_words_eq(
            &format!("strassen-matmul({n}×{n}, seed={seed})"),
            prod.as_words(),
            &oracle_matmul(&a, &b, false),
        );
    }
}

/// Parity at production-dispatch-path sizes. Distinct from
/// `matmul_matches_libm4ri` so it can carry the heavier oracle
/// (`mzd_mul` instead of `mzd_mul_naive`) and the larger Ns.
#[test]
fn matmul_dispatch_paths_match_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(n, seed) in MATMUL_DISPATCH_SHAPES {
        let a = Mzd::random(n, n, seed);
        let b = Mzd::random(n, n, seed ^ 0xDEAD_BEEF);
        let prod = Mzd::matmul(&a, &b);
        assert_words_eq(
            &format!("dispatch-matmul({n}×{n}, seed={seed})"),
            prod.as_words(),
            &oracle_matmul(&a, &b, true),
        );
    }
}

#[test]
fn transpose_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(r, _k, c, seed) in OP_SHAPES {
        // Use (r, c) as the shape; the inner dim is irrelevant here.
        let a = Mzd::random(r, c, seed);
        assert_words_eq(
            &format!("transpose({r}×{c}, seed={seed})"),
            a.transpose().as_words(),
            &oracle_transpose(&a),
        );
    }
}

/// `Mzd::ple_rank` ↔ `mzd_pluq` rank parity across the M4R-block
/// fast-path crossover (N=256) and the v1.0 ship-blocker sizes
/// (N up to 4096). PLUQ output isn't unique across factorisation
/// orders, so only ranks are compared here — bit-equal RREF parity
/// (`rref_matches_libm4ri`) handles the reduction-side correctness.
#[test]
fn ple_rank_matches_libm4ri_pluq() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(n, seed) in PLE_RANK_SHAPES {
        let a = Mzd::random(n, n, seed);
        // Cross-check: m4rst's two rank paths should agree.
        let r_naive = a.rank();
        let r_ple_rust = a.ple_rank();
        assert_eq!(
            r_naive, r_ple_rust,
            "Mzd::rank vs Mzd::ple_rank disagree at N={n} seed={seed:#x}: \
             rank={r_naive} ple_rank={r_ple_rust}"
        );

        // libm4ri's PLUQ and echelonize must agree on the rank for the
        // same input; if not, libm4ri itself is inconsistent here.
        let r_pluq_c = oracle_pluq_rank(&a);
        let (r_echelon_c, _) = oracle_echelonize(&a, true);
        assert_eq!(
            r_pluq_c, r_echelon_c,
            "libm4ri internal: mzd_pluq vs mzd_echelonize disagree at \
             N={n} seed={seed:#x}: pluq={r_pluq_c} echelon={r_echelon_c}"
        );
        assert_eq!(
            r_naive, r_pluq_c,
            "m4rst rank disagrees with libm4ri pluq/echelonize at \
             N={n} seed={seed:#x}: m4rst={r_naive} libm4ri={r_pluq_c}"
        );
    }
}

#[test]
fn rref_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    // RREF is canonical: same input → same output for any correct
    // elimination order. Assert bit-equality of the output AND rank.
    for &(r, _k, c, seed) in OP_SHAPES {
        let m = Mzd::random(r, c, seed);
        let mut m_rust = m.clone();
        let rk_rust = m_rust.echelonize(true);

        let (rk_c, ech_c) = oracle_echelonize(&m, true);
        assert_eq!(
            rk_rust, rk_c,
            "RREF rank disagrees at ({r}×{c}, seed={seed})"
        );
        assert_words_eq(
            &format!("RREF({r}×{c}, seed={seed})"),
            m_rust.as_words(),
            &ech_c,
        );
    }
}

#[test]
fn rank_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    // REF output isn't unique across elimination orders, but the rank
    // is. Drive libm4ri with REF (full=0) to also cover that path.
    for &(r, _k, c, seed) in OP_SHAPES {
        let m = Mzd::random(r, c, seed);
        let rk_rust = m.rank();
        let (rk_c, _) = oracle_echelonize(&m, false);
        assert_eq!(rk_rust, rk_c, "rank disagrees at ({r}×{c}, seed={seed})");
    }
}

#[test]
fn xor_into_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(r, _k, c, seed) in OP_SHAPES {
        let mut a = Mzd::random(r, c, seed);
        let b = Mzd::random(r, c, seed ^ 0x5A5A_5A5A);
        // Snapshot the oracle result before `xor_into` mutates `a`.
        let oracle = oracle_add(&a, &b);
        a.xor_into(&b);
        assert_words_eq(
            &format!("xor_into({r}×{c}, seed={seed})"),
            a.as_words(),
            &oracle,
        );
    }
}

// ---- edge-shape parity (adversarial sweep) -----------------------------

/// Edge shapes spanning awkward column counts (mod 64 boundaries),
/// extreme aspect ratios, and 1×1 / 1×N / N×1 cases. libm4ri's
/// `mzd_init` is fragile around 0 rows / 0 cols, so empty shapes are
/// tested separately by the standalone adversarial tests (no FFI).
const EDGE_SHAPES_LIBM4RI: &[(usize, usize)] = &[
    (1, 1),
    (1, 63),
    (1, 64),
    (1, 65),
    (1, 127),
    (1, 128),
    (1, 129),
    (63, 1),
    (64, 1),
    (65, 1),
    (127, 1),
    (128, 1),
    (129, 1),
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
];

#[test]
fn matmul_edge_shapes_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(r, c) in EDGE_SHAPES_LIBM4RI {
        // Use a square inner dim that exercises both axes.
        let k = c;
        let a = Mzd::random(r, k, 0xED01 ^ (r * 37 + c) as u64);
        let b = Mzd::random(k, c, 0xED02 ^ (r * 13 + c) as u64);
        assert_words_eq(
            &format!("matmul edge ({r}×{k}·{k}×{c})"),
            Mzd::matmul(&a, &b).as_words(),
            &oracle_matmul(&a, &b, false),
        );
    }
}

#[test]
fn transpose_edge_shapes_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(r, c) in EDGE_SHAPES_LIBM4RI {
        let a = Mzd::random(r, c, 0xED03 ^ (r * 41 + c) as u64);
        assert_words_eq(
            &format!("transpose edge ({r}×{c})"),
            a.transpose().as_words(),
            &oracle_transpose(&a),
        );
    }
}

#[test]
fn rref_edge_shapes_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(r, c) in EDGE_SHAPES_LIBM4RI {
        let m = Mzd::random(r, c, 0xED04 ^ (r * 43 + c) as u64);
        let mut m_rust = m.clone();
        let rk_rust = m_rust.echelonize(true);
        let (rk_c, ech_c) = oracle_echelonize(&m, true);
        assert_eq!(rk_rust, rk_c, "RREF rank disagrees at edge ({r},{c})");
        assert_words_eq(&format!("RREF edge ({r}×{c})"), m_rust.as_words(), &ech_c);
    }
}

#[test]
fn rank_edge_shapes_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(r, c) in EDGE_SHAPES_LIBM4RI {
        let m = Mzd::random(r, c, 0xED05 ^ (r * 47 + c) as u64);
        let rk_rust = m.rank();
        let (rk_c, _) = oracle_echelonize(&m, false);
        assert_eq!(rk_rust, rk_c, "rank disagrees at edge ({r},{c})");
    }
}

#[test]
fn xor_edge_shapes_matches_libm4ri() {
    if !validation_enabled() {
        eprintln!("skipping (M4RST_VALIDATE_LIBM4RI not set)");
        return;
    }
    for &(r, c) in EDGE_SHAPES_LIBM4RI {
        let mut a = Mzd::random(r, c, 0xED06 ^ (r * 53 + c) as u64);
        let b = Mzd::random(r, c, 0xED07 ^ (r * 59 + c) as u64);
        let oracle = oracle_add(&a, &b);
        a.xor_into(&b);
        assert_words_eq(&format!("xor edge ({r}×{c})"), a.as_words(), &oracle);
    }
}
