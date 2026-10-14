// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! FLINT-backed scalar facade overhead and dense integer-matrix boundaries.
//!
//! The direct scalar rows call `flint-core`'s owned values; the facade rows
//! call Noether's ring traits on those same values. The raw matrix floor uses
//! the same FLINT operation and fresh output allocation as `Matrix::matmul`,
//! but omits Noether's context/shape checks and result wrapper. Construction,
//! zero-copy clone, and terminal row materialization are timed separately.

use std::hint::black_box;
use std::mem::MaybeUninit;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use flint_sys::{
    fmpq_add, fmpq_mul, fmpz_add, fmpz_mat_clear, fmpz_mat_entry, fmpz_mat_init, fmpz_mat_mul,
    fmpz_mat_struct, fmpz_mul, fmpz_set,
};
use noether::linalg::Matrix;
use noether::numthy::{EllipticCurve, FiniteField};
use noether::{Field, Integer, PrincipalIdealDomain, Rational, Ring, Q, Z};

struct RawIntegerMatrix {
    inner: fmpz_mat_struct,
    rows: usize,
    cols: usize,
}

impl RawIntegerMatrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        let mut inner = MaybeUninit::<fmpz_mat_struct>::uninit();
        // SAFETY: dimensions fit `slong` in every bounded benchmark case and
        // FLINT initializes every field of the matrix in place.
        unsafe { fmpz_mat_init(inner.as_mut_ptr(), rows as i64, cols as i64) };
        Self {
            // SAFETY: `fmpz_mat_init` completed above.
            inner: unsafe { inner.assume_init() },
            rows,
            cols,
        }
    }

    fn from_rows(rows: &[Vec<Integer>]) -> Self {
        let row_count = rows.len();
        let col_count = rows.first().map_or(0, Vec::len);
        let matrix = Self::zeros(row_count, col_count);
        for (row, entries) in rows.iter().enumerate() {
            assert_eq!(entries.len(), col_count);
            for (col, entry) in entries.iter().enumerate() {
                // SAFETY: both indices are in bounds and the destination and
                // source are initialized `fmpz` values.
                unsafe { fmpz_set(matrix.entry(row, col), entry.as_ptr()) };
            }
        }
        matrix
    }

    fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows);
        let mut output = Self::zeros(self.rows, rhs.cols);
        // SAFETY: all matrices are initialized, dimensions match, and output
        // is disjoint from both inputs.
        unsafe { fmpz_mat_mul(&mut output.inner, &self.inner, &rhs.inner) };
        output
    }

    fn entry(&self, row: usize, col: usize) -> *mut flint_sys::fmpz {
        // SAFETY: callers provide in-bounds coordinates.
        unsafe { fmpz_mat_entry(&self.inner, row as i64, col as i64) }
    }
}

impl Drop for RawIntegerMatrix {
    fn drop(&mut self) {
        // SAFETY: `inner` was initialized exactly once and is cleared once.
        unsafe { fmpz_mat_clear(&mut self.inner) };
    }
}

fn integer(bits: u32, salt: u64) -> Integer {
    (Integer::one() << bits) + Integer::from(salt)
}

fn rational(bits: u32, salt: u64) -> Rational {
    Rational::new(
        integer(bits, 2 * salt + 1),
        integer(bits / 2 + 1, 2 * salt + 3),
    )
    .unwrap()
}

fn integer_rows(size: usize, bits: u32, salt: u64) -> Vec<Vec<Integer>> {
    (0..size)
        .map(|row| {
            (0..size)
                .map(|col| integer(bits, salt + (17 * row + 31 * col) as u64))
                .collect()
        })
        .collect()
}

fn scalar_benchmarks(criterion: &mut Criterion) {
    for (label, bits) in [
        ("small-30b", 30),
        ("medium-256b", 256),
        ("large-4096b", 4096),
    ] {
        let left = integer(bits, 17);
        let right = integer(bits, 43);
        let mut group = criterion.benchmark_group(format!("flint-scalar/integer/{label}"));
        group.sample_size(30);
        group.measurement_time(Duration::from_secs(2));

        group.bench_function("add/direct", |bencher| {
            bencher.iter(|| black_box(black_box(&left) + black_box(&right)))
        });
        group.bench_function("add/facade", |bencher| {
            bencher.iter(|| black_box(Z.add(black_box(&left), black_box(&right)).unwrap()))
        });
        let mut add_output = Integer::zero();
        group.bench_function("add/reused-output-floor", |bencher| {
            bencher.iter(|| {
                // SAFETY: all three values are initialized; FLINT permits a
                // caller-owned destination distinct from the inputs.
                unsafe { fmpz_add(add_output.as_mut_ptr(), left.as_ptr(), right.as_ptr()) };
                black_box(&add_output);
            })
        });

        group.bench_function("mul/direct", |bencher| {
            bencher.iter(|| black_box(black_box(&left) * black_box(&right)))
        });
        group.bench_function("mul/facade", |bencher| {
            bencher.iter(|| black_box(Z.mul(black_box(&left), black_box(&right)).unwrap()))
        });
        let mut mul_output = Integer::zero();
        group.bench_function("mul/reused-output-floor", |bencher| {
            bencher.iter(|| {
                // SAFETY: all values are initialized and output is disjoint.
                unsafe { fmpz_mul(mul_output.as_mut_ptr(), left.as_ptr(), right.as_ptr()) };
                black_box(&mul_output);
            })
        });

        group.bench_function("gcd/direct", |bencher| {
            bencher.iter(|| black_box(black_box(&left).gcd(black_box(&right))))
        });
        group.bench_function("gcd/facade", |bencher| {
            bencher.iter(|| black_box(Z.gcd(black_box(&left), black_box(&right))))
        });
        group.finish();

        let left = rational(bits, 19);
        let right = rational(bits, 47);
        let mut group = criterion.benchmark_group(format!("flint-scalar/rational/{label}"));
        group.sample_size(30);
        group.measurement_time(Duration::from_secs(2));

        group.bench_function("add/direct", |bencher| {
            bencher.iter(|| black_box(black_box(&left) + black_box(&right)))
        });
        group.bench_function("add/facade", |bencher| {
            bencher.iter(|| black_box(Q.add(black_box(&left), black_box(&right)).unwrap()))
        });
        let mut add_output = Rational::zero();
        group.bench_function("add/reused-output-floor", |bencher| {
            bencher.iter(|| {
                // SAFETY: all values are initialized and output is disjoint.
                unsafe { fmpq_add(add_output.as_mut_ptr(), left.as_ptr(), right.as_ptr()) };
                black_box(&add_output);
            })
        });

        group.bench_function("mul/direct", |bencher| {
            bencher.iter(|| black_box(black_box(&left) * black_box(&right)))
        });
        group.bench_function("mul/facade", |bencher| {
            bencher.iter(|| black_box(Q.mul(black_box(&left), black_box(&right)).unwrap()))
        });
        let mut mul_output = Rational::zero();
        group.bench_function("mul/reused-output-floor", |bencher| {
            bencher.iter(|| {
                // SAFETY: all values are initialized and output is disjoint.
                unsafe { fmpq_mul(mul_output.as_mut_ptr(), left.as_ptr(), right.as_ptr()) };
                black_box(&mul_output);
            })
        });
        group.finish();
    }
}

fn prime_field_benchmarks(criterion: &mut Criterion) {
    let field = FiniteField::prime(101_u8).unwrap();
    let left = field.element(73_u8);
    let right = field.element(89_u8);
    let short_curve = EllipticCurve::new(
        field.clone(),
        [
            field.zero(),
            field.zero(),
            field.zero(),
            field.element(2_u8),
            field.element(3_u8),
        ],
    )
    .unwrap();
    let short_point = short_curve
        .point(field.element(3_u8), field.element(6_u8))
        .unwrap();
    let generalized_curve = EllipticCurve::new(
        field.clone(),
        [
            field.one(),
            field.zero(),
            field.one(),
            field.element(2_u8),
            field.element(3_u8),
        ],
    )
    .unwrap();
    let generalized_point = generalized_curve
        .point(field.zero(), field.element(17_u8))
        .unwrap();
    let scalar = Integer::from(17_u8);
    let short_order = Integer::from(12_u8);
    let generalized_order = Integer::from(44_u8);

    assert_eq!(
        short_point
            .scalar_mul(&scalar)
            .unwrap()
            .affine_coordinates()
            .map(|(x, y)| (x.representative().clone(), y.representative().clone())),
        Some((Integer::from(76_u8), Integer::from(65_u8)))
    );
    assert_eq!(
        generalized_point
            .scalar_mul(&scalar)
            .unwrap()
            .affine_coordinates()
            .map(|(x, y)| (x.representative().clone(), y.representative().clone())),
        Some((Integer::from(78_u8), Integer::from(54_u8)))
    );
    assert!(short_point.has_exact_order(&short_order).unwrap());
    assert!(generalized_point
        .has_exact_order(&generalized_order)
        .unwrap());

    let mut group = criterion.benchmark_group("prime-field/f101");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function("scalar-mul", |bencher| {
        bencher.iter(|| black_box(field.mul(black_box(&left), black_box(&right)).unwrap()))
    });
    group.bench_function("scalar-inverse", |bencher| {
        bencher.iter(|| black_box(field.inverse(black_box(&left)).unwrap()))
    });
    group.bench_function("elliptic-short-17p", |bencher| {
        bencher.iter(|| black_box(short_point.scalar_mul(black_box(&scalar)).unwrap()))
    });
    group.bench_function("elliptic-generalized-17p", |bencher| {
        bencher.iter(|| black_box(generalized_point.scalar_mul(black_box(&scalar)).unwrap()))
    });
    group.bench_function("elliptic-short-scorecard", |bencher| {
        bencher.iter(|| {
            let multiple = short_point.scalar_mul(black_box(&scalar)).unwrap();
            let exact = short_point
                .has_exact_order(black_box(&short_order))
                .unwrap();
            black_box((multiple, exact))
        })
    });
    group.bench_function("elliptic-generalized-scorecard", |bencher| {
        bencher.iter(|| {
            let multiple = generalized_point.scalar_mul(black_box(&scalar)).unwrap();
            let exact = generalized_point
                .has_exact_order(black_box(&generalized_order))
                .unwrap();
            black_box((multiple, exact))
        })
    });
    group.finish();
}

fn matrix_benchmarks(criterion: &mut Criterion) {
    for (label, size, bits) in [
        ("small-2x2-30b", 2, 30),
        ("medium-32x32-128b", 32, 128),
        ("large-128x128-512b", 128, 512),
    ] {
        let left_rows = integer_rows(size, bits, 11);
        let right_rows = integer_rows(size, bits, 101);
        let left = Matrix::from_rows(Z, left_rows.clone()).unwrap();
        let right = Matrix::from_rows(Z, right_rows.clone()).unwrap();
        let raw_left = RawIntegerMatrix::from_rows(&left_rows);
        let raw_right = RawIntegerMatrix::from_rows(&right_rows);

        let facade_product = left.matmul(&right).unwrap();
        let raw_product = raw_left.matmul(&raw_right);
        for row in 0..size {
            for col in 0..size {
                let facade_entry = facade_product.get(row, col).unwrap();
                // SAFETY: both entries are initialized and live for this
                // pre-timing differential comparison.
                assert_ne!(
                    unsafe {
                        flint_sys::fmpz_equal(facade_entry.as_ptr(), raw_product.entry(row, col))
                    },
                    0
                );
            }
        }

        let mut group = criterion.benchmark_group(format!("flint-matrix/integer/{label}"));
        group.sample_size(20);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements((size * size) as u64));

        group.bench_function("matmul/raw-flint-floor", |bencher| {
            bencher.iter(|| black_box(raw_left.matmul(black_box(&raw_right))))
        });
        group.bench_function("matmul/noether-facade", |bencher| {
            bencher.iter(|| black_box(left.matmul(black_box(&right)).unwrap()))
        });
        group.bench_with_input(
            BenchmarkId::new("construction/noether", size),
            &left_rows,
            |bencher, rows| {
                bencher.iter_batched(
                    || rows.clone(),
                    |rows| black_box(Matrix::from_rows(Z, rows).unwrap()),
                    BatchSize::SmallInput,
                )
            },
        );
        group.bench_with_input(
            BenchmarkId::new("construction/raw-flint-floor", size),
            &left_rows,
            |bencher, rows| bencher.iter(|| black_box(RawIntegerMatrix::from_rows(rows))),
        );
        group.bench_function("clone/persistent-storage", |bencher| {
            bencher.iter(|| black_box(black_box(&left).clone()))
        });
        group.bench_function("materialize/to-rows", |bencher| {
            bencher.iter(|| black_box(black_box(&left).to_rows()))
        });
        group.finish();
    }
}

fn arithmetic_boundaries(criterion: &mut Criterion) {
    scalar_benchmarks(criterion);
    prime_field_benchmarks(criterion);
    matrix_benchmarks(criterion);
}

criterion_group!(benches, arithmetic_boundaries);
criterion_main!(benches);
