// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Bit-for-bit parity tests against FLINT.
//!
//! Gated behind the `flint` Cargo feature because the sweep is slower than
//! the ordinary unit suite. Build with
//! `FLINT_PREFIX=/path/to/flint cargo test --release --features flint
//! --test parity_flint`.
//!
//! For every op where FLINT exposes a canonical implementation we
//! assert sparla's output matches FLINT's bit-for-bit:
//!
//! - **F_p**: `det`, `rank`, `rref` vs `nmod_mat_{det,rank,rref}`.
//! - **Z**: `snf`, `hnf`, `det`, `rank` vs `fmpz_mat_{snf,hnf,det,rank}`.
//!
//! Coverage: hand-checked edge cases (empty, identity, zero,
//! singular) plus LCG-driven random sweeps across a (size, density)
//! grid. Parity is the strongest correctness signal we can get
//! short of running the noether-side bench harness end-to-end.

#![cfg(feature = "flint")]

use flint_sys::{
    fmpz, fmpz_clear, fmpz_get_si, fmpz_init, fmpz_mat_clear, fmpz_mat_det, fmpz_mat_entry,
    fmpz_mat_hnf, fmpz_mat_init, fmpz_mat_rank, fmpz_mat_snf, fmpz_mat_struct, fmpz_set_si,
    nmod_mat_clear, nmod_mat_det, nmod_mat_get_entry, nmod_mat_init, nmod_mat_rank, nmod_mat_rref,
    nmod_mat_set_entry, nmod_mat_struct, slong, ulong,
};
use sparla::storage::{CsrFp, CsrZ, SparseRingStorage};
use sparla::Integer;
use std::mem::MaybeUninit;

// =====================================================================
// RAII wrappers
// =====================================================================

struct FlintNmodMat {
    inner: nmod_mat_struct,
}

impl FlintNmodMat {
    fn new(nrows: usize, ncols: usize, modulus: u64) -> Self {
        let mut inner = MaybeUninit::<nmod_mat_struct>::uninit();
        unsafe {
            nmod_mat_init(
                inner.as_mut_ptr(),
                nrows as slong,
                ncols as slong,
                modulus as ulong,
            );
            Self {
                inner: inner.assume_init(),
            }
        }
    }
    fn as_ptr(&self) -> *const nmod_mat_struct {
        &self.inner
    }
    fn as_mut_ptr(&mut self) -> *mut nmod_mat_struct {
        &mut self.inner
    }
}

impl Drop for FlintNmodMat {
    fn drop(&mut self) {
        unsafe { nmod_mat_clear(&mut self.inner) }
    }
}

struct FlintFmpzMat {
    inner: fmpz_mat_struct,
}

impl FlintFmpzMat {
    fn new(nrows: usize, ncols: usize) -> Self {
        let mut inner = MaybeUninit::<fmpz_mat_struct>::uninit();
        unsafe {
            fmpz_mat_init(inner.as_mut_ptr(), nrows as slong, ncols as slong);
            Self {
                inner: inner.assume_init(),
            }
        }
    }
    fn as_ptr(&self) -> *const fmpz_mat_struct {
        &self.inner
    }
    fn as_mut_ptr(&mut self) -> *mut fmpz_mat_struct {
        &mut self.inner
    }
}

impl Drop for FlintFmpzMat {
    fn drop(&mut self) {
        unsafe { fmpz_mat_clear(&mut self.inner) }
    }
}

// =====================================================================
// Sparla → FLINT loaders
// =====================================================================

fn load_csr_fp(a: &CsrFp) -> FlintNmodMat {
    let mut fm = FlintNmodMat::new(a.rows(), a.cols(), a.modulus());
    for r in 0..a.rows() {
        for (c, v) in a.iter_row(r) {
            unsafe {
                nmod_mat_set_entry(fm.as_mut_ptr(), r as slong, c as slong, v as ulong);
            }
        }
    }
    fm
}

fn load_csr_z(a: &CsrZ) -> FlintFmpzMat {
    let fm = FlintFmpzMat::new(a.rows(), a.cols());
    for r in 0..a.rows() {
        for (c, v) in a.iter_row(r) {
            let val = v.to_i64().expect("test inputs fit in i64");
            unsafe {
                let entry = fmpz_mat_entry(fm.as_ptr(), r as slong, c as slong);
                fmpz_set_si(entry, val as slong);
            }
        }
    }
    fm
}

fn nmod_to_dense(fm: &FlintNmodMat, rows: usize, cols: usize) -> Vec<Vec<u32>> {
    let mut out = vec![vec![0u32; cols]; rows];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let v = unsafe { nmod_mat_get_entry(fm.as_ptr(), r as slong, c as slong) };
            *cell = v as u32;
        }
    }
    out
}

fn fmpz_mat_to_dense(fm: &FlintFmpzMat, rows: usize, cols: usize) -> Vec<Vec<Integer>> {
    let mut out: Vec<Vec<Integer>> = (0..rows).map(|_| vec![Integer::from(0); cols]).collect();
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let v = unsafe {
                let entry = fmpz_mat_entry(fm.as_ptr(), r as slong, c as slong);
                fmpz_get_si(entry)
            };
            *cell = Integer::from(v);
        }
    }
    out
}

// =====================================================================
// Generators
// =====================================================================

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 32
    }
}

fn gen_sparse_fp(rows: usize, cols: usize, density: f64, prime: u64, seed: u64) -> CsrFp {
    let mut rng = Lcg(seed);
    let target = ((rows * cols) as f64 * density) as usize;
    let p = prime as i64;
    let mut triples: Vec<(usize, usize, i64)> = Vec::with_capacity(target);
    for _ in 0..target {
        let r = (rng.next() as usize) % rows;
        let c = (rng.next() as usize) % cols;
        let v = (rng.next() as i64 % (p - 1)) + 1;
        triples.push((r, c, v));
    }
    CsrFp::from_coo(rows, cols, &triples, prime)
}

fn gen_sparse_z(rows: usize, cols: usize, density: f64, seed: u64) -> CsrZ {
    let mut rng = Lcg(seed);
    let target = ((rows * cols) as f64 * density) as usize;
    let mut triples: Vec<(usize, usize, Integer)> = Vec::with_capacity(target);
    for _ in 0..target {
        let r = (rng.next() as usize) % rows;
        let c = (rng.next() as usize) % cols;
        // Small entries so fmpz_get_si round-trips cleanly.
        let mut v = (rng.next() as i64 % 19) - 9;
        if v == 0 {
            v = 1;
        }
        triples.push((r, c, Integer::from(v)));
    }
    CsrZ::from_coo(rows, cols, &triples)
}

fn csr_z_from_dense(dense: &[Vec<i64>]) -> CsrZ {
    let rows = dense.len();
    let cols = if rows == 0 { 0 } else { dense[0].len() };
    let mut triples: Vec<(usize, usize, Integer)> = Vec::new();
    for (r, row) in dense.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if v != 0 {
                triples.push((r, c, Integer::from(v)));
            }
        }
    }
    CsrZ::from_coo(rows, cols, &triples)
}

fn csr_fp_from_dense(dense: &[Vec<i64>], prime: u64) -> CsrFp {
    let rows = dense.len();
    let cols = if rows == 0 { 0 } else { dense[0].len() };
    let mut triples = Vec::new();
    for (r, row) in dense.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if v.rem_euclid(prime as i64) != 0 {
                triples.push((r, c, v));
            }
        }
    }
    CsrFp::from_coo(rows, cols, &triples, prime)
}

// =====================================================================
// F_p parity: det
// =====================================================================

fn check_fp_det(a: &CsrFp) {
    let sparla_det = a.det();
    let flint = load_csr_fp(a);
    let flint_det = unsafe { nmod_mat_det(flint.as_ptr()) } as u32;
    assert_eq!(
        sparla_det,
        flint_det,
        "det mismatch (mod {}): {} × {} matrix",
        a.modulus(),
        a.rows(),
        a.cols()
    );
}

#[test]
fn parity_fp_det_hand_cases() {
    let p = 7u64;
    check_fp_det(&CsrFp::zeros(3, 3, p));
    check_fp_det(&csr_fp_from_dense(&[vec![1, 0], vec![0, 1]], p));
    check_fp_det(&csr_fp_from_dense(&[vec![0, 1], vec![1, 0]], p));
    check_fp_det(&csr_fp_from_dense(&[vec![2, 3], vec![4, 5]], p));
    check_fp_det(&csr_fp_from_dense(
        &[vec![1, 2, 3], vec![2, 4, 6], vec![0, 1, 0]],
        p,
    ));
}

#[test]
fn parity_fp_det_random() {
    for prime in [7u64, 101, (1u64 << 31) - 1] {
        for n in [3usize, 5, 10] {
            for &density in &[0.2, 0.5, 1.0] {
                for seed in 0..5u64 {
                    let a = gen_sparse_fp(n, n, density, prime, 0xA0FF ^ (seed << 8));
                    check_fp_det(&a);
                }
            }
        }
    }
}

// =====================================================================
// F_p parity: rank
// =====================================================================

fn check_fp_rank(a: &CsrFp) {
    let sparla_rank = a.rank();
    let flint = load_csr_fp(a);
    let flint_rank = unsafe { nmod_mat_rank(flint.as_ptr()) } as usize;
    assert_eq!(
        sparla_rank,
        flint_rank,
        "rank mismatch (mod {}): {} × {} matrix",
        a.modulus(),
        a.rows(),
        a.cols()
    );
}

#[test]
fn parity_fp_rank_hand_cases() {
    let p = 7u64;
    check_fp_rank(&CsrFp::zeros(4, 5, p));
    check_fp_rank(&csr_fp_from_dense(
        &[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]],
        p,
    ));
    // Row 1 = 2·row 0 → rank 2.
    check_fp_rank(&csr_fp_from_dense(
        &[vec![1, 2, 3], vec![2, 4, 6], vec![0, 1, 0]],
        p,
    ));
}

#[test]
fn parity_fp_rank_random() {
    for prime in [7u64, 101] {
        for &(rows, cols) in &[(3, 5), (5, 3), (10, 10)] {
            for &density in &[0.1, 0.3, 0.7] {
                for seed in 0..5u64 {
                    let a = gen_sparse_fp(rows, cols, density, prime, 0xB0FF ^ (seed << 8));
                    check_fp_rank(&a);
                }
            }
        }
    }
}

// =====================================================================
// F_p parity: rref
// =====================================================================

fn check_fp_rref(a: &CsrFp) {
    let sparla_rref = a.rref();
    let mut flint = load_csr_fp(a);
    let flint_rank = unsafe { nmod_mat_rref(flint.as_mut_ptr()) } as usize;
    assert_eq!(
        sparla_rref.rank,
        flint_rank,
        "RREF rank mismatch on {}×{}",
        a.rows(),
        a.cols()
    );
    let flint_dense = nmod_to_dense(&flint, a.rows(), a.cols());
    assert_eq!(
        sparla_rref.matrix.to_dense(),
        flint_dense,
        "RREF matrix mismatch on {}×{}",
        a.rows(),
        a.cols()
    );
}

#[test]
fn parity_fp_rref_hand_cases() {
    let p = 7u64;
    check_fp_rref(&CsrFp::zeros(3, 4, p));
    check_fp_rref(&csr_fp_from_dense(
        &[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]],
        p,
    ));
    check_fp_rref(&csr_fp_from_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4]], p));
    check_fp_rref(&csr_fp_from_dense(
        &[vec![2, 3, 1], vec![4, 6, 2], vec![1, 1, 1]],
        p,
    ));
}

#[test]
fn parity_fp_rref_random() {
    for prime in [7u64, 101] {
        for &(rows, cols) in &[(4, 4), (5, 7), (7, 5)] {
            for &density in &[0.3, 0.7] {
                for seed in 0..5u64 {
                    let a = gen_sparse_fp(rows, cols, density, prime, 0xC0FF ^ (seed << 8));
                    check_fp_rref(&a);
                }
            }
        }
    }
}

// =====================================================================
// Z parity: det
// =====================================================================

fn check_z_det(a: &CsrZ) {
    let sparla_det = a.det();
    let flint = load_csr_z(a);
    let mut det: fmpz = unsafe { std::mem::zeroed() };
    unsafe {
        fmpz_init(&mut det);
        fmpz_mat_det(&mut det, flint.as_ptr());
    }
    let flint_det = Integer::from(unsafe { fmpz_get_si(&det) });
    unsafe { fmpz_clear(&mut det) };
    assert_eq!(
        sparla_det,
        flint_det,
        "Z det mismatch on {}×{}",
        a.rows(),
        a.cols()
    );
}

#[test]
fn parity_z_det_hand_cases() {
    check_z_det(&CsrZ::zeros(3, 3));
    check_z_det(&csr_z_from_dense(&[
        vec![1, 0, 0],
        vec![0, 1, 0],
        vec![0, 0, 1],
    ]));
    check_z_det(&csr_z_from_dense(&[vec![1, 2], vec![3, 4]]));
    check_z_det(&csr_z_from_dense(&[vec![2, 4], vec![6, 8]]));
    check_z_det(&csr_z_from_dense(&[vec![0, 1], vec![1, 0]]));
    check_z_det(&csr_z_from_dense(&[
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 10],
    ]));
}

#[test]
fn parity_z_det_random() {
    for n in [3usize, 4, 5] {
        for &density in &[0.4, 0.8] {
            for seed in 0..5u64 {
                let a = gen_sparse_z(n, n, density, 0xD0FF ^ (seed << 8));
                check_z_det(&a);
            }
        }
    }
}

// =====================================================================
// Z parity: rank
// =====================================================================

fn check_z_rank(a: &CsrZ) {
    let sparla_rank = a.rank();
    let flint = load_csr_z(a);
    let flint_rank = unsafe { fmpz_mat_rank(flint.as_ptr()) } as usize;
    assert_eq!(
        sparla_rank,
        flint_rank,
        "Z rank mismatch on {}×{}",
        a.rows(),
        a.cols()
    );
}

#[test]
fn parity_z_rank_hand_cases() {
    check_z_rank(&CsrZ::zeros(4, 5));
    check_z_rank(&csr_z_from_dense(&[
        vec![1, 0, 0],
        vec![0, 1, 0],
        vec![0, 0, 1],
    ]));
    check_z_rank(&csr_z_from_dense(&[
        vec![1, 2, 3],
        vec![2, 4, 6],
        vec![3, 6, 9],
    ]));
    check_z_rank(&csr_z_from_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4]]));
}

#[test]
fn parity_z_rank_random() {
    for &(rows, cols) in &[(3, 5), (5, 3), (5, 5), (7, 4)] {
        for &density in &[0.3, 0.7] {
            for seed in 0..5u64 {
                let a = gen_sparse_z(rows, cols, density, 0xE0FF ^ (seed << 8));
                check_z_rank(&a);
            }
        }
    }
}

// =====================================================================
// Z parity: HNF
// =====================================================================

fn check_z_hnf(a: &CsrZ) {
    let sparla_hnf = a.hnf().to_dense();
    let flint_in = load_csr_z(a);
    let mut flint_h = FlintFmpzMat::new(a.rows(), a.cols());
    unsafe { fmpz_mat_hnf(flint_h.as_mut_ptr(), flint_in.as_ptr()) };
    let flint_hnf = fmpz_mat_to_dense(&flint_h, a.rows(), a.cols());
    assert_eq!(
        sparla_hnf,
        flint_hnf,
        "HNF mismatch on {}×{}",
        a.rows(),
        a.cols()
    );
}

#[test]
fn parity_z_hnf_hand_cases() {
    check_z_hnf(&CsrZ::zeros(3, 4));
    check_z_hnf(&csr_z_from_dense(&[
        vec![1, 0, 0],
        vec![0, 1, 0],
        vec![0, 0, 1],
    ]));
    check_z_hnf(&csr_z_from_dense(&[vec![2, 4], vec![6, 8]]));
    check_z_hnf(&csr_z_from_dense(&[vec![1, 2], vec![3, 4]]));
    check_z_hnf(&csr_z_from_dense(&[vec![1, 2, 0, 3], vec![0, 0, 1, 4]]));
}

#[test]
fn parity_z_hnf_random() {
    for &(rows, cols) in &[(3, 3), (4, 4), (3, 5), (5, 3)] {
        for &density in &[0.5, 0.9] {
            for seed in 0..5u64 {
                let a = gen_sparse_z(rows, cols, density, 0xF0FF ^ (seed << 8));
                check_z_hnf(&a);
            }
        }
    }
}

// =====================================================================
// Z parity: SNF
// =====================================================================

fn check_z_snf(a: &CsrZ) {
    let sparla_snf = a.snf();
    let flint_in = load_csr_z(a);
    let mut flint_s = FlintFmpzMat::new(a.rows(), a.cols());
    unsafe { fmpz_mat_snf(flint_s.as_mut_ptr(), flint_in.as_ptr()) };
    let flint_diag: Vec<Integer> = (0..a.rows().min(a.cols()))
        .map(|i| unsafe {
            let entry = fmpz_mat_entry(flint_s.as_ptr(), i as slong, i as slong);
            Integer::from(fmpz_get_si(entry))
        })
        .collect();
    assert_eq!(
        sparla_snf,
        flint_diag,
        "SNF mismatch on {}×{}",
        a.rows(),
        a.cols()
    );
}

#[test]
fn parity_z_snf_hand_cases() {
    check_z_snf(&CsrZ::zeros(3, 3));
    check_z_snf(&csr_z_from_dense(&[
        vec![1, 0, 0],
        vec![0, 1, 0],
        vec![0, 0, 1],
    ]));
    check_z_snf(&csr_z_from_dense(&[vec![2, 0], vec![0, 6]]));
    check_z_snf(&csr_z_from_dense(&[vec![6, 0], vec![0, 2]]));
    check_z_snf(&csr_z_from_dense(&[vec![2, 0], vec![0, 3]]));
    check_z_snf(&csr_z_from_dense(&[vec![2, 4], vec![6, 8]]));
    check_z_snf(&csr_z_from_dense(&[
        vec![2, 0, 0],
        vec![0, 3, 0],
        vec![0, 0, 5],
    ]));
}

#[test]
fn parity_z_snf_random() {
    for &(rows, cols) in &[(3, 3), (4, 4), (3, 5), (5, 3)] {
        for &density in &[0.5, 0.9] {
            for seed in 0..5u64 {
                let a = gen_sparse_z(rows, cols, density, 0x10FE ^ (seed << 8));
                check_z_snf(&a);
            }
        }
    }
}
