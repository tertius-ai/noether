// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Paired sparse/dense binary multiplication, with streaming exact checks.
//!
//! Run through `scripts/bench-pin.sh` with `TERTIUS_BENCH_CPUS=group`.
//! The default is the historical grid through order 131072. `N_VALS` and
//! `DENSITIES` select a bounded Cartesian grid. `RUNS` (default 1) and
//! `WARMUP` (default 0) control median sampling; `SKIP_SERIAL=1` omits the
//! serial sparse baseline. Fixture construction and checks are untimed.

use std::hint::black_box;
use std::str::FromStr;
use std::time::Instant;

use m4rst::Mzd;
use mzs::Mzs;
use rand::rngs::ChaCha8Rng;
use rand::{RngExt, SeedableRng};

fn build_pair(n: usize, density: f64) -> (Mzs, Mzs, Mzd, Mzd) {
    assert!(n <= u32::MAX as usize, "N exceeds sparse column storage");
    n.checked_mul(n).expect("N squared overflows usize");
    assert!(density.is_finite() && (0.0..=1.0).contains(&density));
    let mut rng_a = ChaCha8Rng::seed_from_u64(0xA);
    let mut rng_b = ChaCha8Rng::seed_from_u64(0xB);
    let mut sparse_a = Mzs::new(n, n);
    let mut sparse_b = Mzs::new(n, n);
    let mut dense_a = Mzd::new(n, n);
    let mut dense_b = Mzd::new(n, n);
    // Construct the two representations directly. An N-by-N byte staging
    // matrix would use eight times the packed storage at the largest orders.
    for i in 0..n {
        for j in 0..n {
            if rng_a.random::<f64>() < density {
                sparse_a.set(i, j, 1);
                dense_a.set(i, j, 1);
            }
            if rng_b.random::<f64>() < density {
                sparse_b.set(i, j, 1);
                dense_b.set(i, j, 1);
            }
        }
    }
    (sparse_a, sparse_b, dense_a, dense_b)
}

fn rows_equal_streaming(sparse: &Mzs, dense: &Mzd) -> bool {
    if sparse.rows() != dense.rows() || sparse.cols() != dense.cols() {
        return false;
    }
    let mut packed = vec![0_u64; dense.width()];
    for i in 0..sparse.rows() {
        packed.fill(0);
        for &j in sparse.row_indices(i) {
            let j = j as usize;
            packed[j / 64] |= 1_u64 << (j % 64);
        }
        if packed != dense.row_words(i) {
            return false;
        }
    }
    true
}

fn median_ms<T>(
    warmup: usize,
    runs: usize,
    mut operation: impl FnMut() -> T,
    mut verify: impl FnMut(&T),
) -> f64 {
    for _ in 0..warmup {
        let result = operation();
        verify(black_box(&result));
    }
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        let result = black_box(operation());
        times.push(started.elapsed().as_secs_f64() * 1_000.0);
        verify(&result);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn bench_cell(n: usize, density: f64, warmup: usize, runs: usize, skip_serial: bool) {
    let (sparse_a, sparse_b, dense_a, dense_b) = build_pair(n, density);
    let expected = Mzd::matmul(&dense_a, &dense_b);
    let check_sparse = |product: &Mzs| {
        assert!(
            rows_equal_streaming(product, &expected),
            "sparse and dense products differ at N={n}, density={density}"
        );
    };
    let serial = if skip_serial {
        String::from("—")
    } else {
        format!(
            "{:.6}",
            median_ms(
                warmup,
                runs,
                || black_box(&sparse_a).matmul(black_box(&sparse_b)),
                check_sparse,
            )
        )
    };
    let parallel = median_ms(
        warmup,
        runs,
        || black_box(&sparse_a).matmul_parallel(black_box(&sparse_b)),
        check_sparse,
    );
    let dense = median_ms(
        warmup,
        runs,
        || Mzd::matmul(black_box(&dense_a), black_box(&dense_b)),
        |product| assert_eq!(product.as_words(), expected.as_words()),
    );
    println!(
        "| {n} | {density} | {} | {} | {serial} | {parallel:.6} | {dense:.6} | equal |",
        sparse_a.nnz(),
        sparse_b.nnz(),
    );
}

fn env_value<T: FromStr>(key: &str, default: T) -> T {
    match std::env::var(key) {
        Ok(value) => value.parse().unwrap_or_else(|_| panic!("invalid {key}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("invalid {key}: {error}"),
    }
}

fn env_list<T: FromStr + Copy>(key: &str, default: &[T]) -> Vec<T> {
    match std::env::var(key) {
        Ok(value) => value
            .split(',')
            .map(|part| {
                part.trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("invalid {key}"))
            })
            .collect(),
        Err(std::env::VarError::NotPresent) => default.to_vec(),
        Err(error) => panic!("invalid {key}: {error}"),
    }
}

fn main() {
    let warmup = env_value("WARMUP", 0_usize);
    let runs = env_value("RUNS", 1_usize);
    assert!(runs > 0, "RUNS must be positive");
    let skip_serial = env_value("SKIP_SERIAL", 0_u8);
    assert!(skip_serial <= 1, "SKIP_SERIAL must be 0 or 1");
    let historical: &[(usize, &[f64])] = &[
        (256, &[0.001, 0.01, 0.05, 0.1, 0.5]),
        (1024, &[0.001, 0.005, 0.01, 0.05, 0.1]),
        (4096, &[0.001, 0.005, 0.01, 0.05]),
        (16384, &[0.001, 0.005, 0.01]),
        (65536, &[0.001, 0.005]),
        (131072, &[0.001, 0.005]),
    ];
    let cells: Vec<(usize, f64)> =
        if std::env::var_os("N_VALS").is_some() || std::env::var_os("DENSITIES").is_some() {
            let sizes = env_list("N_VALS", &[256_usize, 1024, 4096]);
            let densities = env_list("DENSITIES", &[0.001_f64, 0.005, 0.01]);
            sizes
                .into_iter()
                .flat_map(|n| densities.iter().map(move |&density| (n, density)))
                .collect()
        } else {
            historical
                .iter()
                .flat_map(|&(n, densities)| densities.iter().map(move |&density| (n, density)))
                .collect()
        };
    // Validate every fixture before starting a potentially long sweep.
    for &(n, density) in &cells {
        assert!(n <= u32::MAX as usize, "N exceeds sparse column storage");
        n.checked_mul(n).expect("N squared overflows usize");
        assert!(
            density.is_finite() && (0.0..=1.0).contains(&density),
            "DENSITIES must be finite fractions in [0, 1]"
        );
    }
    eprintln!(
        "# seeds=0xA,0xB rng=rand::ChaCha8Rng threads={} warmup={warmup} runs={runs} median ms",
        rayon::current_num_threads(),
    );
    println!(
        "| N | density | A nnz | B nnz | mzs serial ms | mzs parallel ms | m4rst ms | parity |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---|");
    for (n, density) in cells {
        bench_cell(
            n,
            density,
            warmup,
            runs,
            skip_serial == 1 || (n >= 131072 && density >= 0.005),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_check_detects_tail_and_shape_mismatches() {
        let mut sparse = Mzs::new(1, 65);
        sparse.set(0, 64, 1);
        let mut dense = Mzd::new(1, 65);
        dense.set(0, 64, 1);
        assert!(rows_equal_streaming(&sparse, &dense));
        dense.set(0, 63, 1);
        assert!(!rows_equal_streaming(&sparse, &dense));
        assert!(!rows_equal_streaming(&sparse, &Mzd::new(1, 64)));
        assert!(!rows_equal_streaming(&sparse, &Mzd::new(2, 65)));
    }

    #[test]
    fn fixtures_and_products_agree_at_empty_and_word_boundaries() {
        for n in [0, 1, 2, 63, 64, 65] {
            for density in [0.0, 0.2, 1.0] {
                let (a, b, da, db) = build_pair(n, density);
                assert!(rows_equal_streaming(&a, &da));
                assert!(rows_equal_streaming(&b, &db));
                let dense = Mzd::matmul(&da, &db);
                assert!(rows_equal_streaming(&a.matmul(&b), &dense));
                assert!(rows_equal_streaming(&a.matmul_parallel(&b), &dense));
                if density == 1.0 {
                    for i in 0..n {
                        for j in 0..n {
                            assert_eq!(dense.get(i, j), (n % 2) as u8);
                        }
                    }
                }
            }
        }
    }
}
