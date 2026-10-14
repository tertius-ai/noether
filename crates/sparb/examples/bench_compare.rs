// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse/dense crossover benchmark for rigorous real and complex balls.
//!
//! Each row compares `sparb` CSR arithmetic with the corresponding dense
//! FLINT `arb_mat` or `acb_mat` operation on the same entries, precision,
//! and vector/matrix fixture. Before timing, every sparse result is checked
//! to overlap the independently computed dense enclosure. A disagreement
//! aborts the benchmark rather than publishing incomparable timings.
//!
//! Run from the repository root:
//!
//! ```text
//! scripts/bench-pin.sh cargo run --release -p sparb --example bench_compare
//! ```

use std::hint::black_box;
use std::mem::MaybeUninit;
use std::time::Instant;

use flint_sys::{
    acb_add, acb_mat_clear, acb_mat_entry_ptr, acb_mat_init, acb_mat_mul, acb_mat_struct,
    acb_overlaps, arb_add, arb_mat_clear, arb_mat_entry_ptr, arb_mat_init, arb_mat_mul,
    arb_mat_struct, arb_overlaps, slong,
};
use sparb::{Acb, Arb, CsrAcb, CsrArb};

#[derive(Clone, Copy)]
struct Entry {
    row: usize,
    col: usize,
    real: f64,
    imag: f64,
}

struct FixtureRng(u64);

impl FixtureRng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, upper: u64) -> u64 {
        assert!(upper > 0);
        let cutoff = u64::MAX - u64::MAX % upper;
        loop {
            let value = self.next_u64();
            if value < cutoff {
                return value % upper;
            }
        }
    }

    fn scalar(&mut self) -> f64 {
        self.below(2_001) as f64 / 1_000.0 - 1.0
    }
}

fn random_entries(n: usize, density: f64, seed: u64) -> Vec<Entry> {
    let mut rng = FixtureRng::new(seed);
    let target_nnz = ((n * n) as f64 * density) as usize;
    (0..target_nnz)
        .map(|_| Entry {
            row: rng.below(n as u64) as usize,
            col: rng.below(n as u64) as usize,
            real: rng.scalar(),
            imag: rng.scalar(),
        })
        .collect()
}

fn vector_entries(n: usize) -> Vec<Entry> {
    (0..n)
        .map(|row| Entry {
            row,
            col: 0,
            real: row as f64 * 0.5 - 0.25,
            imag: row as f64 * -0.125 + 0.5,
        })
        .collect()
}

struct DenseArbMat {
    inner: arb_mat_struct,
}

impl DenseArbMat {
    fn zeros(rows: usize, cols: usize) -> Self {
        let mut inner = MaybeUninit::<arb_mat_struct>::uninit();
        // SAFETY: FLINT initializes every field and entry of the matrix.
        unsafe { arb_mat_init(inner.as_mut_ptr(), rows as slong, cols as slong) };
        // SAFETY: `arb_mat_init` completed successfully.
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    fn from_entries(rows: usize, cols: usize, entries: &[Entry]) -> Self {
        let mut matrix = Self::zeros(rows, cols);
        for entry in entries {
            let value = Arb::from_f64(entry.real);
            // SAFETY: fixture coordinates are within the initialized matrix;
            // FLINT permits the output to alias the left input of `arb_add`.
            unsafe {
                let slot =
                    arb_mat_entry_ptr(&mut matrix.inner, entry.row as slong, entry.col as slong);
                arb_add(slot, slot, value.as_ptr(), 64);
            }
        }
        matrix
    }

    fn mul(&self, rhs: &Self, precision: i64) -> Self {
        let mut output = Self::zeros(self.inner.r as usize, rhs.inner.c as usize);
        // SAFETY: the dimensions match by fixture construction and all three
        // FLINT matrices are initialized for the duration of the call.
        unsafe { arb_mat_mul(&mut output.inner, &self.inner, &rhs.inner, precision) };
        output
    }

    fn overlaps_vector(&mut self, sparse: &[Arb]) -> bool {
        if self.inner.r as usize != sparse.len() || self.inner.c != 1 {
            return false;
        }
        sparse.iter().enumerate().all(|(row, value)| {
            // SAFETY: `row` and column zero are in bounds, and both balls are
            // live for the duration of this read-only overlap test.
            unsafe {
                arb_overlaps(
                    arb_mat_entry_ptr(&mut self.inner, row as slong, 0),
                    value.as_ptr(),
                ) != 0
            }
        })
    }

    fn overlaps_csr(&mut self, sparse: &CsrArb, n: usize) -> bool {
        if self.inner.r as usize != n || self.inner.c as usize != n {
            return false;
        }
        let zero = Arb::zero();
        (0..n).all(|row| {
            (0..n).all(|column| {
                let value = sparse.get(row, column).unwrap_or(&zero);
                // SAFETY: the loop coordinates are in bounds, and both balls
                // are live for the duration of this read-only overlap test.
                unsafe {
                    arb_overlaps(
                        arb_mat_entry_ptr(&mut self.inner, row as slong, column as slong),
                        value.as_ptr(),
                    ) != 0
                }
            })
        })
    }
}

impl Drop for DenseArbMat {
    fn drop(&mut self) {
        // SAFETY: `inner` was initialized once and is cleared once here.
        unsafe { arb_mat_clear(&mut self.inner) };
    }
}

struct DenseAcbMat {
    inner: acb_mat_struct,
}

impl DenseAcbMat {
    fn zeros(rows: usize, cols: usize) -> Self {
        let mut inner = MaybeUninit::<acb_mat_struct>::uninit();
        // SAFETY: FLINT initializes every field and entry of the matrix.
        unsafe { acb_mat_init(inner.as_mut_ptr(), rows as slong, cols as slong) };
        // SAFETY: `acb_mat_init` completed successfully.
        Self {
            inner: unsafe { inner.assume_init() },
        }
    }

    fn from_entries(rows: usize, cols: usize, entries: &[Entry]) -> Self {
        let mut matrix = Self::zeros(rows, cols);
        for entry in entries {
            let value = Acb::from_re_im(entry.real, entry.imag);
            // SAFETY: fixture coordinates are within the initialized matrix;
            // FLINT permits the output to alias the left input of `acb_add`.
            unsafe {
                let slot =
                    acb_mat_entry_ptr(&mut matrix.inner, entry.row as slong, entry.col as slong);
                acb_add(slot, slot, value.as_ptr(), 64);
            }
        }
        matrix
    }

    fn mul(&self, rhs: &Self, precision: i64) -> Self {
        let mut output = Self::zeros(self.inner.r as usize, rhs.inner.c as usize);
        // SAFETY: the dimensions match by fixture construction and all three
        // FLINT matrices are initialized for the duration of the call.
        unsafe { acb_mat_mul(&mut output.inner, &self.inner, &rhs.inner, precision) };
        output
    }

    fn overlaps_vector(&mut self, sparse: &[Acb]) -> bool {
        if self.inner.r as usize != sparse.len() || self.inner.c != 1 {
            return false;
        }
        sparse.iter().enumerate().all(|(row, value)| {
            // SAFETY: `row` and column zero are in bounds, and both balls are
            // live for the duration of this read-only overlap test.
            unsafe {
                acb_overlaps(
                    acb_mat_entry_ptr(&mut self.inner, row as slong, 0),
                    value.as_ptr(),
                ) != 0
            }
        })
    }

    fn overlaps_csr(&mut self, sparse: &CsrAcb, n: usize) -> bool {
        if self.inner.r as usize != n || self.inner.c as usize != n {
            return false;
        }
        let zero = Acb::zero();
        (0..n).all(|row| {
            (0..n).all(|column| {
                let value = sparse.get(row, column).unwrap_or(&zero);
                // SAFETY: the loop coordinates are in bounds, and both balls
                // are live for the duration of this read-only overlap test.
                unsafe {
                    acb_overlaps(
                        acb_mat_entry_ptr(&mut self.inner, row as slong, column as slong),
                        value.as_ptr(),
                    ) != 0
                }
            })
        })
    }
}

impl Drop for DenseAcbMat {
    fn drop(&mut self) {
        // SAFETY: `inner` was initialized once and is cleared once here.
        unsafe { acb_mat_clear(&mut self.inner) };
    }
}

fn mean_us<F, T>(repetitions: u32, mut operation: F) -> f64
where
    F: FnMut() -> T,
{
    black_box(operation());
    let start = Instant::now();
    for _ in 0..repetitions {
        black_box(operation());
    }
    start.elapsed().as_secs_f64() * 1e6 / repetitions as f64
}

#[derive(Clone, Copy)]
struct Row {
    scalar: &'static str,
    operation: &'static str,
    backend: &'static str,
    n: usize,
    density: f64,
    precision: i64,
    micros: f64,
}

fn run_spmv(
    rows: &mut Vec<Row>,
    scalar: &'static str,
    n: usize,
    density: f64,
    precision: i64,
    entries: &[Entry],
    vector: &[Entry],
) {
    let repetitions = 20;
    if scalar == "Arb" {
        let sparse_entries = entries
            .iter()
            .map(|entry| (entry.row, entry.col, entry.real, 0.0))
            .collect::<Vec<_>>();
        let sparse = CsrArb::from_coo_f64(n, n, precision, &sparse_entries);
        let x = vector
            .iter()
            .map(|entry| Arb::from_f64(entry.real))
            .collect::<Vec<_>>();
        let dense = DenseArbMat::from_entries(n, n, entries);
        let dense_x = DenseArbMat::from_entries(n, 1, vector);
        let sparse_result = sparse.spmv(&x);
        let mut dense_result = dense.mul(&dense_x, precision);
        assert!(
            dense_result.overlaps_vector(&sparse_result),
            "Arb SpMV preflight disagrees with FLINT dense"
        );
        rows.push(Row {
            scalar,
            operation: "SpMV",
            backend: "sparb CSR",
            n,
            density,
            precision,
            micros: mean_us(repetitions, || sparse.spmv(&x)),
        });
        rows.push(Row {
            scalar,
            operation: "SpMV",
            backend: "FLINT dense",
            n,
            density,
            precision,
            micros: mean_us(5, || dense.mul(&dense_x, precision)),
        });
    } else {
        let sparse_entries = entries
            .iter()
            .map(|entry| (entry.row, entry.col, entry.real, 0.0, entry.imag, 0.0))
            .collect::<Vec<_>>();
        let sparse = CsrAcb::from_coo_f64(n, n, precision, &sparse_entries);
        let x = vector
            .iter()
            .map(|entry| Acb::from_re_im(entry.real, entry.imag))
            .collect::<Vec<_>>();
        let dense = DenseAcbMat::from_entries(n, n, entries);
        let dense_x = DenseAcbMat::from_entries(n, 1, vector);
        let sparse_result = sparse.spmv(&x);
        let mut dense_result = dense.mul(&dense_x, precision);
        assert!(
            dense_result.overlaps_vector(&sparse_result),
            "Acb SpMV preflight disagrees with FLINT dense"
        );
        rows.push(Row {
            scalar,
            operation: "SpMV",
            backend: "sparb CSR",
            n,
            density,
            precision,
            micros: mean_us(repetitions, || sparse.spmv(&x)),
        });
        rows.push(Row {
            scalar,
            operation: "SpMV",
            backend: "FLINT dense",
            n,
            density,
            precision,
            micros: mean_us(5, || dense.mul(&dense_x, precision)),
        });
    }
}

fn run_matmul(
    rows: &mut Vec<Row>,
    scalar: &'static str,
    n: usize,
    density: f64,
    precision: i64,
    left: &[Entry],
    right: &[Entry],
) {
    if scalar == "Arb" {
        let sparse_left = left
            .iter()
            .map(|entry| (entry.row, entry.col, entry.real, 0.0))
            .collect::<Vec<_>>();
        let sparse_right = right
            .iter()
            .map(|entry| (entry.row, entry.col, entry.real, 0.0))
            .collect::<Vec<_>>();
        let a = CsrArb::from_coo_f64(n, n, precision, &sparse_left);
        let b = CsrArb::from_coo_f64(n, n, precision, &sparse_right);
        let dense_a = DenseArbMat::from_entries(n, n, left);
        let dense_b = DenseArbMat::from_entries(n, n, right);
        let sparse_result = a.matmul(&b);
        let mut dense_result = dense_a.mul(&dense_b, precision);
        assert!(
            dense_result.overlaps_csr(&sparse_result, n),
            "Arb matmul preflight disagrees with FLINT dense"
        );
        rows.push(Row {
            scalar,
            operation: "matmul",
            backend: "sparb CSR",
            n,
            density,
            precision,
            micros: mean_us(5, || a.matmul(&b)),
        });
        rows.push(Row {
            scalar,
            operation: "matmul",
            backend: "FLINT dense",
            n,
            density,
            precision,
            micros: mean_us(3, || dense_a.mul(&dense_b, precision)),
        });
    } else {
        let sparse_left = left
            .iter()
            .map(|entry| (entry.row, entry.col, entry.real, 0.0, entry.imag, 0.0))
            .collect::<Vec<_>>();
        let sparse_right = right
            .iter()
            .map(|entry| (entry.row, entry.col, entry.real, 0.0, entry.imag, 0.0))
            .collect::<Vec<_>>();
        let a = CsrAcb::from_coo_f64(n, n, precision, &sparse_left);
        let b = CsrAcb::from_coo_f64(n, n, precision, &sparse_right);
        let dense_a = DenseAcbMat::from_entries(n, n, left);
        let dense_b = DenseAcbMat::from_entries(n, n, right);
        let sparse_result = a.matmul(&b);
        let mut dense_result = dense_a.mul(&dense_b, precision);
        assert!(
            dense_result.overlaps_csr(&sparse_result, n),
            "Acb matmul preflight disagrees with FLINT dense"
        );
        rows.push(Row {
            scalar,
            operation: "matmul",
            backend: "sparb CSR",
            n,
            density,
            precision,
            micros: mean_us(5, || a.matmul(&b)),
        });
        rows.push(Row {
            scalar,
            operation: "matmul",
            backend: "FLINT dense",
            n,
            density,
            precision,
            micros: mean_us(3, || dense_a.mul(&dense_b, precision)),
        });
    }
}

fn main() {
    let mut rows = Vec::new();
    for &(n, density) in &[(256, 0.01), (512, 0.05), (1_024, 0.10)] {
        let entries = random_entries(n, density, 0xB0FF);
        let vector = vector_entries(n);
        for &precision in &[53, 128] {
            run_spmv(&mut rows, "Arb", n, density, precision, &entries, &vector);
            run_spmv(&mut rows, "Acb", n, density, precision, &entries, &vector);
        }
    }

    for &(n, density) in &[(128, 0.05), (256, 0.10)] {
        let left = random_entries(n, density, 0xB0FF);
        let right = random_entries(n, density, 0xB101);
        for &precision in &[53, 128] {
            run_matmul(&mut rows, "Arb", n, density, precision, &left, &right);
            run_matmul(&mut rows, "Acb", n, density, precision, &left, &right);
        }
    }

    println!("| scalar | operation | backend | n | density | precision | us/op |");
    println!("|---|---|---|---:|---:|---:|---:|");
    for row in rows {
        println!(
            "| {} | {} | {} | {} | {:.3} | {} | {:.1} |",
            row.scalar, row.operation, row.backend, row.n, row.density, row.precision, row.micros
        );
    }
}
