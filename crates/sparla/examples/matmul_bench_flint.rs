// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse · sparse vs FLINT-dense matmul over `F_p`.
//!
//! Production comparison against the canonical dense backend
//! (`nmod_mat_mul`), as opposed to the naive triple loop in
//! `examples/matmul_bench.rs`. Build/run with:
//!
//! ```sh
//! FLINT_PREFIX=/path/to/flint \
//!   cargo run --release --features flint --example matmul_bench_flint
//! ```
//!
//! Three columns of timing per (size, density) config:
//! - `sparse ms` — sparla's Gustavson sparse · sparse → sparse.
//! - `flint mul ms` — just `nmod_mat_mul` on already-loaded dense.
//! - `flint total ms` — load-into-nmod + `nmod_mat_mul` + read-out.
//!
//! `flint total` is what a noether user paying the densify-and-
//! delegate cost would actually see; `flint mul` is the pure
//! algorithmic-work comparison.
//!
//! Parity-checks sparse against FLINT before each timing.

use std::mem::MaybeUninit;
use std::time::Instant;

use flint_sys::{
    nmod_mat_clear, nmod_mat_get_entry, nmod_mat_init, nmod_mat_mul, nmod_mat_set_entry,
    nmod_mat_struct, slong, ulong,
};
use sparla::storage::{CsrFp, SparseRingStorage};

const PRIME: u64 = 65537;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 32
    }
}

fn gen_sparse(rows: usize, cols: usize, density: f64, prime: u64, seed: u64) -> CsrFp {
    let mut rng = Lcg::new(seed);
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

fn sparse_matmul(a: &CsrFp, b: &CsrFp) -> Vec<Vec<(usize, u32)>> {
    assert_eq!(a.cols(), b.rows());
    assert_eq!(a.modulus(), b.modulus());
    let p = a.modulus();
    let m = a.rows();
    let n = b.cols();
    let mut out_rows: Vec<Vec<(usize, u32)>> = Vec::with_capacity(m);
    let mut acc: Vec<u64> = vec![0; n];
    let mut touched: Vec<usize> = Vec::new();
    for i in 0..m {
        for (k, av) in a.iter_row(i) {
            let av = av as u64;
            for (j, bv) in b.iter_row(k) {
                if acc[j] == 0 {
                    touched.push(j);
                }
                acc[j] = (acc[j] + av * bv as u64) % p;
            }
        }
        touched.sort_unstable();
        touched.dedup();
        let mut row: Vec<(usize, u32)> = Vec::with_capacity(touched.len());
        for &j in &touched {
            let v = acc[j];
            if v != 0 {
                row.push((j, v as u32));
            }
            acc[j] = 0;
        }
        touched.clear();
        out_rows.push(row);
    }
    out_rows
}

fn densify_sparse_rows(rows: &[Vec<(usize, u32)>], ncols: usize) -> Vec<Vec<u32>> {
    rows.iter()
        .map(|row| {
            let mut r = vec![0u32; ncols];
            for &(c, v) in row {
                r[c] = v;
            }
            r
        })
        .collect()
}

/// RAII wrapper for `nmod_mat_t` — handles init/clear. Mirrors
/// noether's `FlintNmodMat` pattern in `dense_mat_nmod.rs`.
struct FlintNmodMat {
    inner: nmod_mat_struct,
}

impl FlintNmodMat {
    fn new(nrows: usize, ncols: usize, modulus: u64) -> Self {
        let mut inner = MaybeUninit::<nmod_mat_struct>::uninit();
        // Safety: nmod_mat_init initialises the struct in place.
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
        unsafe {
            nmod_mat_clear(&mut self.inner);
        }
    }
}

fn load_into_flint(a: &CsrFp) -> FlintNmodMat {
    let mut fm = FlintNmodMat::new(a.rows(), a.cols(), a.modulus());
    for r in 0..a.rows() {
        for (c, v) in a.iter_row(r) {
            // Safety: indices are in-bounds by construction.
            unsafe {
                nmod_mat_set_entry(fm.as_mut_ptr(), r as slong, c as slong, v as ulong);
            }
        }
    }
    fm
}

fn flint_result_to_dense(fc: &FlintNmodMat, rows: usize, cols: usize) -> Vec<Vec<u32>> {
    let mut out = vec![vec![0u32; cols]; rows];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            // Safety: indices are in-bounds by construction.
            let v = unsafe { nmod_mat_get_entry(fc.as_ptr(), i as slong, j as slong) };
            *cell = v as u32;
        }
    }
    out
}

fn main() {
    println!(
        "sparse · sparse vs FLINT nmod_mat_mul over F_p (p = {})",
        PRIME
    );
    println!();
    println!(
        "{:>6} {:>9} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "size",
        "density",
        "A nnz",
        "B nnz",
        "sparse ms",
        "flint mul ms",
        "flint tot ms",
        "vs mul",
        "vs total"
    );
    println!("{}", "-".repeat(108));

    let configs: &[(usize, f64)] = &[
        (50, 0.01),
        (50, 0.05),
        (50, 0.25),
        (50, 1.00),
        (100, 0.01),
        (100, 0.05),
        (100, 0.25),
        (100, 1.00),
        (200, 0.01),
        (200, 0.05),
        (200, 0.25),
        (500, 0.001),
        (500, 0.01),
        (500, 0.05),
        (500, 0.25),
        (1000, 0.001),
        (1000, 0.01),
        (1000, 0.05),
    ];

    for &(n, density) in configs {
        let a = gen_sparse(n, n, density, PRIME, 0xC0FF_EE42);
        let b = gen_sparse(n, n, density, PRIME, 0xBAAD_F00D);

        // Sparse algorithmic work only.
        let t0 = Instant::now();
        let c_sparse = sparse_matmul(&a, &b);
        let sparse_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // FLINT "total" = load CsrFp → nmod_mat, multiply, read out.
        let t_total = Instant::now();
        let fa = load_into_flint(&a);
        let fb = load_into_flint(&b);
        let mut fc = FlintNmodMat::new(n, n, PRIME);
        let t_mul = Instant::now();
        unsafe {
            nmod_mat_mul(fc.as_mut_ptr(), fa.as_ptr(), fb.as_ptr());
        }
        let flint_mul_ms = t_mul.elapsed().as_secs_f64() * 1000.0;
        let c_flint = flint_result_to_dense(&fc, n, n);
        let flint_total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(
            densify_sparse_rows(&c_sparse, n),
            c_flint,
            "sparse and FLINT disagree at n = {}, density = {}",
            n,
            density
        );

        let vs_mul = flint_mul_ms / sparse_ms.max(1e-6);
        let vs_total = flint_total_ms / sparse_ms.max(1e-6);
        println!(
            "{:>6} {:>8.2}% {:>10} {:>10} {:>10.3} {:>12.3} {:>12.3} {:>9.2}x {:>9.2}x",
            n,
            density * 100.0,
            a.nnz(),
            b.nnz(),
            sparse_ms,
            flint_mul_ms,
            flint_total_ms,
            vs_mul,
            vs_total,
        );
    }
}
