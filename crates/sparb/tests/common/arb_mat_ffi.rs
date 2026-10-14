// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Safe ownership around FLINT `arb_mat` for parity tests.
//!
//! sparb itself doesn't need `arb_mat` — the kernel is sparse and
//! operates on individual arbs via `arb_addmul`. Parity tests want
//! FLINT's dense `arb_mat_mul` as the oracle for ball-arithmetic
//! correctness, reached through the workspace `flint-sys` bindings.
//!
//! Struct layout matches FLINT 3.x as confirmed by reading
//! `src/arb_mat/init.c`:
//! ```text
//! { entries: arb_ptr, r: slong, c: slong, stride: slong }   /* 32 bytes */
//! ```
//! `arb_mat_init(_, r, c)` writes `stride = c`; entries are laid
//! out row-major with that stride.

// Allow dead code — different test binaries use different subsets
// of these helpers, and Rust's compilation unit per integration-
// test file would otherwise complain about unused items in the
// shared `common/` modules.
#![allow(dead_code)]

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::os::raw::c_char;

use flint_sys::{
    arb_add, arb_get_ubound_arf, arb_mat_clear, arb_mat_init, arb_mat_mul, arb_mat_struct,
    arb_set_d, arb_struct, arf_clear, arf_get_d, arf_init, arf_rnd_t_ARF_RND_CEIL,
    arf_rnd_t_ARF_RND_NEAR, arf_struct, slong,
};
use sparb::Arb;

// FLINT's runtime version string (a global, exported as a data
// symbol). The struct layout for `arb_mat_struct` changed in FLINT
// 3.1: 3.0.x used `{ entries, r, c, rows }` (where `rows` was
// `arb_ptr *`); 3.1+ uses `{ entries, r, c, stride }` (with `stride:
// slong`). Same byte offsets, *different field semantics* — reading
// the 3.0 layout as if it were 3.1's `stride` produces a wildly
// out-of-bounds offset and a SIGSEGV. The `flint_supports_stride_layout`
// gate below lets parity tests self-skip on FLINT < 3.1 so they
// don't crash CI on Ubuntu 24.04's libflint-dev 3.0.1.
// FLINT declares it as `extern char flint_version[]` (an array,
// not a pointer), so the symbol IS the address of the string's
// first byte. Declaring it as `c_char` and taking `&flint_version`
// recovers that address. Mis-declaring it as `*const c_char` and
// reading the static would read the first 8 bytes of the string
// as a 64-bit pointer — bogus address, instant SIGSEGV.
extern "C" {
    static flint_version: c_char;
}

/// Parse `flint_version` ("X.Y.Z") as `(major, minor)`. Returns
/// `None` if the string isn't parseable.
fn flint_major_minor() -> Option<(u32, u32)> {
    // SAFETY: `&flint_version` is the address of FLINT's NUL-
    // terminated version string; the static has lived through
    // every FLINT release as a stable global.
    let ptr: *const c_char = unsafe { &flint_version };
    let s = unsafe { CStr::from_ptr(ptr) }.to_str().ok()?;
    let mut parts = s.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// `true` iff the linked FLINT is recent enough that
/// `arb_mat_struct` uses the `{ entries, r, c, stride }` layout.
/// Parity tests that read entries via the stride field gate on
/// this to self-skip on older FLINT rather than crash.
pub fn flint_supports_stride_layout() -> bool {
    flint_major_minor().is_some_and(|(major, minor)| (major, minor) >= (3, 1))
}

/// RAII wrapper around `arb_mat_struct`. `Drop` calls
/// `arb_mat_clear` exactly once; moves are safe (the inner struct
/// is plain data with one heap pointer per entry).
pub struct ArbMat {
    inner: arb_mat_struct,
}

impl ArbMat {
    /// All-zero matrix with the given shape. `arb_mat_init`
    /// zero-initialises every entry.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let mut inner = MaybeUninit::<arb_mat_struct>::uninit();
        // SAFETY: arb_mat_init fully initialises the struct in place
        // and allocates one zero arb per entry.
        unsafe {
            arb_mat_init(inner.as_mut_ptr(), rows as slong, cols as slong);
        }
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    /// Build from `(row, col, value)` triples — duplicates summed
    /// in arb arithmetic at `build_prec` bits (tests use 64 or
    /// higher; the rounding here is negligible for the fixtures
    /// we care about).
    pub fn from_triples(
        rows: usize,
        cols: usize,
        triples: &[(usize, usize, f64)],
        build_prec: slong,
    ) -> Self {
        let mat = Self::zeros(rows, cols);
        for &(r, c, v) in triples {
            let cell = mat.entry_ptr(r, c);
            let scratch = Arb::from_f64(v);
            // SAFETY: cell is a valid entry slot; scratch is a
            // valid arb; arb_add tolerates `cell` aliasing `a`.
            unsafe {
                arb_add(cell, cell, scratch.as_ptr(), build_prec);
            }
        }
        mat
    }

    /// `n × 1` column matrix from a slice of `f64` values, all
    /// stored as exact balls (`rad = 0`).
    pub fn col_from_f64(x: &[f64]) -> Self {
        let mat = Self::zeros(x.len(), 1);
        for (i, &v) in x.iter().enumerate() {
            let cell = mat.entry_ptr(i, 0);
            // SAFETY: cell is a valid arb slot; arb_set_d is exact
            // for any finite f64.
            unsafe {
                arb_set_d(cell, v);
            }
        }
        mat
    }

    /// Dense ball mat-mat product `self * other` at `prec`.
    pub fn mul(&self, other: &ArbMat, prec: slong) -> ArbMat {
        let m = self.inner.r as usize;
        let n = other.inner.c as usize;
        let mut out = ArbMat::zeros(m, n);
        // SAFETY: all three matrices are properly init'd; arb_mat_mul
        // requires distinct out / a / b which we satisfy.
        unsafe {
            arb_mat_mul(&mut out.inner, &self.inner, &other.inner, prec);
        }
        out
    }

    /// `(midpoint, radius)` at position `(i, j)` with radius a
    /// conservative `f64` upper bound at `readback_prec`.
    pub fn entry_pair(&self, i: usize, j: usize, readback_prec: slong) -> (f64, f64) {
        let ptr = self.entry_ptr(i, j) as *const arb_struct;
        // SAFETY: ptr is a valid arb; scratch arf init / use /
        // clear is balanced inside one unsafe block.
        unsafe {
            let mid = arf_get_d(&(*ptr).mid, arf_rnd_t_ARF_RND_NEAR);
            let mut u = MaybeUninit::<arf_struct>::uninit();
            arf_init(u.as_mut_ptr());
            let mut u = u.assume_init();
            arb_get_ubound_arf(&mut u, ptr, readback_prec);
            let u_d = arf_get_d(&u, arf_rnd_t_ARF_RND_CEIL);
            arf_clear(&mut u);
            (mid, (u_d - mid).max(0.0))
        }
    }

    fn entry_ptr(&self, i: usize, j: usize) -> *mut arb_struct {
        // SAFETY: caller is responsible for in-bounds (i, j);
        // arb_mat_init wrote `stride = c` so entries are row-major.
        unsafe { self.inner.entries.add(i * self.inner.stride as usize + j) }
    }
}

impl Drop for ArbMat {
    fn drop(&mut self) {
        // SAFETY: inner was arb_mat_init'd and is cleared exactly
        // once here.
        unsafe { arb_mat_clear(&mut self.inner) }
    }
}
