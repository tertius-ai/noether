// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Head-to-head pwvla's dispatched kernels vs FLINT's `nmod_mat_mul` over
//! Z/2^n for n ∈ {8, 16, 32}. Run with:
//!
//! ```bash
//! FLINT_PREFIX=/usr cargo run --release --features flint \
//!     --example bench_z2n_flint
//! ```
//!
//! ## What's measured
//!
//! - **pwvla**: the same per-width dispatch functions used by Noether's
//!   residue-matrix adapter. Native wrapping arithmetic is the ring operation.
//! - **FLINT**: `nmod_mat_mul` with `nmod_t` modulus 2^8 / 2^16 / 2^32.
//! - Both sides allocate a fresh output matrix for each timed product.
//!
//! u64 / u128 aren't compared head-to-head because FLINT's `nmod_t`
//! can't represent modulus = 2^64 / 2^128 (modulus is a `ulong`).
//! pwvla supports `Mat<u64>` / `Mat<u128>` natively via `wrapping_mul`.

use flint_sys::{
    nmod_mat_clear, nmod_mat_get_entry, nmod_mat_init, nmod_mat_mul, nmod_mat_set_entry,
    nmod_mat_struct,
};
use pwvla::mat::{Mat, Z2n};
use rand::rngs::ChaCha20Rng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use std::mem::MaybeUninit;
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
const U8_ROUTE: &str = "neon";
#[cfg(not(target_arch = "aarch64"))]
const U8_ROUTE: &str = "scalar";

#[cfg(target_arch = "aarch64")]
const U16_ROUTE: &str = "neon";
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const U16_ROUTE: &str = "scalar";

#[cfg(target_arch = "x86_64")]
fn u16_route() -> &'static str {
    if std::arch::is_x86_feature_detected!("avx2") {
        "avx2"
    } else {
        "scalar"
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn u16_route() -> &'static str {
    U16_ROUTE
}

#[cfg(target_arch = "aarch64")]
const U32_ROUTE: &str = "neon";
#[cfg(not(target_arch = "aarch64"))]
const U32_ROUTE: &str = "scalar";

struct FlintMat {
    inner: nmod_mat_struct,
}

impl FlintMat {
    fn from_pwvla<T: Z2n>(m: &Mat<T>, modulus: u64) -> Self {
        unsafe {
            let mut inner = MaybeUninit::<nmod_mat_struct>::uninit();
            nmod_mat_init(inner.as_mut_ptr(), m.rows as i64, m.cols as i64, modulus);
            let mut inner = inner.assume_init();
            for i in 0..m.rows {
                for j in 0..m.cols {
                    let v = m.get(i, j).to_u64();
                    nmod_mat_set_entry(&mut inner, i as i64, j as i64, v);
                }
            }
            Self { inner }
        }
    }

    fn zeros(rows: usize, cols: usize, modulus: u64) -> Self {
        unsafe {
            let mut inner = MaybeUninit::<nmod_mat_struct>::uninit();
            nmod_mat_init(inner.as_mut_ptr(), rows as i64, cols as i64, modulus);
            Self {
                inner: inner.assume_init(),
            }
        }
    }
}

impl Drop for FlintMat {
    fn drop(&mut self) {
        unsafe { nmod_mat_clear(&mut self.inner) };
    }
}

// ---- bench harness --------------------------------------------------

fn rand_mat<T: Z2n>(n: usize, seed: u64) -> Mat<T> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    Mat::<T>::from_fn(n, n, |_, _| T::from_u64(rng.random::<u64>()))
}

fn gflops(ops: f64, dt: f64) -> f64 {
    ops / dt / 1e9
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn measure(mut product: impl FnMut()) -> f64 {
    product();
    let mut repetitions = 1usize;
    loop {
        let started = Instant::now();
        for _ in 0..repetitions {
            product();
        }
        if started.elapsed().as_secs_f64() >= 0.1 || repetitions >= 1 << 20 {
            break;
        }
        repetitions *= 2;
    }

    median(
        (0..5)
            .map(|_| {
                let started = Instant::now();
                for _ in 0..repetitions {
                    product();
                }
                started.elapsed().as_secs_f64() / repetitions as f64
            })
            .collect(),
    )
}

fn bench_ring<T: Z2n>(
    label: &str,
    route: &str,
    modulus: u64,
    n: usize,
    pwvla_matmul: fn(&Mat<T>, &Mat<T>) -> Mat<T>,
) -> (f64, f64) {
    let a = rand_mat::<T>(n, 1);
    let b = rand_mat::<T>(n, 2);
    let ops = 2.0 * (n as f64).powi(3); // 1 mul + 1 add per inner iter

    let af = FlintMat::from_pwvla(&a, modulus);
    let bf = FlintMat::from_pwvla(&b, modulus);
    let exact_product = pwvla_matmul(&a, &b);
    let mut flint_product = FlintMat::zeros(n, n, modulus);
    unsafe { nmod_mat_mul(&mut flint_product.inner, &af.inner, &bf.inner) };
    for row in 0..n {
        for column in 0..n {
            let flint_entry =
                unsafe { nmod_mat_get_entry(&flint_product.inner, row as i64, column as i64) };
            assert_eq!(
                exact_product.get(row, column).to_u64(),
                flint_entry,
                "pwvla and FLINT differ at ({row}, {column}) over {label}"
            );
        }
    }
    let dt_pwvla = measure(|| {
        black_box(pwvla_matmul(black_box(&a), black_box(&b)));
    });
    let dt_flint = measure(|| {
        let mut output = FlintMat::zeros(n, n, modulus);
        unsafe { nmod_mat_mul(&mut output.inner, &af.inner, &bf.inner) };
        black_box(output);
    });

    println!(
        "  {label:>10} N={n:<5}  pwvla/{route:<6} {:>7.2} Gops/s ({:>7.2} ms) | FLINT {:>7.2} Gops/s ({:>7.2} ms)  ratio {:>5.2}×",
        gflops(ops, dt_pwvla),
        dt_pwvla * 1e3,
        gflops(ops, dt_flint),
        dt_flint * 1e3,
        dt_flint / dt_pwvla,
    );

    (dt_pwvla, dt_flint)
}

fn main() {
    println!("pwvla vs FLINT — Z/2^n dense matmul, serial production routes");
    println!("median of five calibrated samples; fresh output storage per product");
    println!();

    for &n in &[256usize, 512, 1024, 2048] {
        bench_ring::<u8>("Z/2^8", U8_ROUTE, 1u64 << 8, n, pwvla::matmul_u8);
        bench_ring::<u16>("Z/2^16", u16_route(), 1u64 << 16, n, pwvla::matmul_u16);
        bench_ring::<u32>("Z/2^32", U32_ROUTE, 1u64 << 32, n, pwvla::matmul_u32);
        println!();
    }
}
