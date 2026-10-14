// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Safe benchmark-only ownership for opaque libm4ri matrices.

use core::ffi::{c_int, c_void};
use m4rst::Mzd;

extern "C" {
    fn m4ri_oracle_matrix_new(rows: usize, cols: usize, data: *const u64) -> *mut c_void;
    fn m4ri_oracle_matrix_free(matrix: *mut c_void);
    fn m4ri_oracle_preloaded_matmul(
        left: *const c_void,
        right: *const c_void,
        algorithm: c_int,
        checksum: *mut u64,
    ) -> c_int;
    #[allow(dead_code)]
    fn m4ri_oracle_preloaded_echelonize(
        matrix: *const c_void,
        full: c_int,
        checksum: *mut u64,
        rank: *mut usize,
    ) -> c_int;
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    fn m4ri_oracle_echelonize(
        nrows: usize,
        ncols: usize,
        data: *const u64,
        full: c_int,
        out: *mut u64,
    ) -> c_int;
}

/// Compute the complete libm4ri product for an untimed exact preflight.
#[allow(dead_code)]
pub fn matmul_words(left: &Mzd, right: &Mzd) -> Vec<u64> {
    let width = right.cols().div_ceil(64);
    let mut out = vec![0_u64; left.rows() * width];
    // SAFETY: the input slices remain live, the output has exactly the packed
    // dimensions expected by the C shim, and the shim validates conformity.
    let status = unsafe {
        m4ri_oracle_matmul(
            left.rows(),
            left.cols(),
            left.as_words().as_ptr(),
            right.rows(),
            right.cols(),
            right.as_words().as_ptr(),
            out.as_mut_ptr(),
            1,
        )
    };
    assert_eq!(status, 0, "libm4ri matrix multiplication failed");
    out
}

/// Compute the complete libm4ri echelon form for an untimed exact preflight.
#[allow(dead_code)]
pub fn echelon_words(matrix: &Mzd, full: bool) -> (usize, Vec<u64>) {
    let mut out = vec![0_u64; matrix.as_words().len()];
    // SAFETY: the input remains live and the output is the complete packed
    // matrix buffer required by the C shim.
    let status = unsafe {
        m4ri_oracle_echelonize(
            matrix.rows(),
            matrix.cols(),
            matrix.as_words().as_ptr(),
            c_int::from(full),
            out.as_mut_ptr(),
        )
    };
    assert!(status >= 0, "libm4ri matrix elimination failed");
    (status as usize, out)
}

pub struct OracleMatrix {
    inner: *mut c_void,
}

impl OracleMatrix {
    pub fn from_mzd(matrix: &Mzd) -> Self {
        // SAFETY: `as_words` is the packed layout promised by the C shim and
        // remains live for the duration of the call. The returned handle is
        // either null or uniquely owned by this value.
        let inner = unsafe {
            m4ri_oracle_matrix_new(matrix.rows(), matrix.cols(), matrix.as_words().as_ptr())
        };
        assert!(!inner.is_null(), "libm4ri matrix allocation failed");
        Self { inner }
    }

    #[allow(dead_code)]
    pub fn matmul_m4rm(&self, rhs: &Self) -> u64 {
        self.matmul_with(rhs, 0)
    }

    #[allow(dead_code)]
    pub fn matmul(&self, rhs: &Self) -> u64 {
        self.matmul_with(rhs, 1)
    }

    fn matmul_with(&self, rhs: &Self, algorithm: c_int) -> u64 {
        let mut checksum = 0;
        // SAFETY: both opaque handles are live libm4ri matrices owned by these
        // values. The shim checks dimensions and writes one initialized word.
        let status = unsafe {
            m4ri_oracle_preloaded_matmul(self.inner, rhs.inner, algorithm, &mut checksum)
        };
        assert_eq!(status, 0, "libm4ri matrix multiplication failed");
        checksum
    }

    #[allow(dead_code)]
    pub fn echelonize(&self, full: bool) -> (usize, u64) {
        let mut checksum = 0;
        let mut rank = 0;
        // SAFETY: the opaque handle is live and owned by this value. The shim
        // copies it before reduction and initializes both scalar outputs.
        let status = unsafe {
            m4ri_oracle_preloaded_echelonize(
                self.inner,
                c_int::from(full),
                &mut checksum,
                &mut rank,
            )
        };
        assert_eq!(status, 0, "libm4ri matrix elimination failed");
        (rank, checksum)
    }
}

impl Drop for OracleMatrix {
    fn drop(&mut self) {
        // SAFETY: `inner` came from `m4ri_oracle_matrix_new`, is uniquely
        // owned, and is freed exactly once here.
        unsafe { m4ri_oracle_matrix_free(self.inner) };
    }
}
