// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Sparse matrix–vector product over real balls.
//!
//! `y = A · x` where `A` is a [`CsrArb`] and `x` a dense slice of
//! [`Arb`] balls. Every multiply and accumulate runs in FLINT ball
//! arithmetic at the matrix's working precision, so each output
//! `y[i]` is a *rigorous enclosure* of the true `Σ_j A[i,j] · x[j]`.
//!
//! This is the load-bearing kernel for the PDE verification rail:
//! the residual enclosure `K·u − b` is one SpMV plus a ball
//! subtraction, and the a-posteriori error bound is built on that.

use crate::ball::Arb;
use crate::storage::csr_arb::CsrArb;
use crate::storage::SparseRingStorage;

/// Compute `y = A · x` in ball arithmetic — sparb's headline
/// primitive.
///
/// # Contract
///
/// `x` must have length `a.cols()`. The result has length
/// `a.rows()`; each output ball `y[i]` rigorously encloses the
/// true mathematical sum `Σ_j A[i,j] · x[j]`. Empty rows of `A`
/// yield the exact zero ball.
///
/// # Precision
///
/// All intermediate operations run at `a.prec()` (the matrix's
/// working precision). The output ball at row `i` has radius
/// bounded above by
///
/// ```text
/// rad(y[i])  ≲   Σ_j  |A_mid[i,j]| · rad(x[j])
///             +  Σ_j  rad(A[i,j])  · |x_mid[j]|
///             +  Σ_j  rad(A[i,j])  · rad(x[j])
///             +  O(nnz_in_row · |y[i]| · 2⁻ᵃ·ᵖʳᵉᶜ)
/// ```
///
/// — the per-non-zero cross-term sum plus the fused-multiply-add
/// rounding error accumulated over the row. For exact-input SpMV
/// (`rad = 0` everywhere on input) the bound collapses to the
/// rightmost term, which goes to zero as `a.prec() → ∞`. This is
/// the contract `tests/instability_demo.rs` pins on a worked
/// catastrophic-cancellation case.
///
/// # Cost
///
/// One FLINT `arb_addmul` per non-zero (one FFI call, one
/// mantissa update); pre-allocated output `Vec<Arb>` so no
/// per-nz allocation. Wall-clock per nnz ~30 ns at `prec=53` on
/// the reference dev box — see `README.md` § SpMV benchmarks.
///
/// # Panics
///
/// Panics if `x.len() != a.cols()`.
pub fn spmv(a: &CsrArb, x: &[Arb]) -> Vec<Arb> {
    assert_eq!(
        x.len(),
        a.cols(),
        "spmv: x has length {} but A has {} columns",
        x.len(),
        a.cols()
    );
    let prec = a.prec();
    // Pre-allocate the result so each row's accumulator is held in
    // place (one mantissa allocation per row, reused across all
    // non-zeros in that row), and arb_addmul writes directly into
    // it — no per-nz temporary.
    let mut y: Vec<Arb> = (0..a.rows()).map(|_| Arb::zero()).collect();
    for (r, acc) in y.iter_mut().enumerate() {
        for (c, v) in a.iter_row(r) {
            acc.addmul_assign(v, &x[c], prec);
        }
    }
    y
}

/// Parallel sparse matrix–vector product `y = A · x` — same
/// mathematical contract as [`spmv`], rows distributed across rayon
/// worker threads.
///
/// # Determinism
///
/// **Bit-for-bit identical to [`spmv`]** on the same input,
/// asserted by `par_spmv_matches_serial` in the test suite. Each
/// row's accumulation order is fixed (left-to-right within the
/// row); only the assignment of rows to threads varies, and the
/// per-row computation is independent. No race in the radius
/// arithmetic.
///
/// # When it pays off
///
/// Intended for matrices large enough to amortize Rayon's scheduling cost;
/// benchmark the workload before selecting it directly.
///
/// # Cost
///
/// Same `O(nnz)` `arb_addmul` count as [`spmv`], divided by the
/// effective thread parallelism, plus a constant rayon fork/join
/// overhead.
///
/// Gated on the `parallel` Cargo feature, which pulls in `rayon`.
///
/// # Panics
///
/// Panics if `x.len() != a.cols()`.
#[cfg(feature = "parallel")]
pub fn par_spmv(a: &CsrArb, x: &[Arb]) -> Vec<Arb> {
    use rayon::prelude::*;
    assert_eq!(
        x.len(),
        a.cols(),
        "par_spmv: x has length {} but A has {} columns",
        x.len(),
        a.cols()
    );
    let prec = a.prec();
    let mut y: Vec<Arb> = (0..a.rows()).map(|_| Arb::zero()).collect();
    // Each row's work is independent and writes only into `y[r]`,
    // so `par_iter_mut().enumerate()` is the natural shape. `a` and
    // `x` cross to worker threads as shared (`Send + Sync`).
    y.par_iter_mut().enumerate().for_each(|(r, acc)| {
        for (c, v) in a.iter_row(r) {
            acc.addmul_assign(v, &x[c], prec);
        }
    });
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(r: usize, c: usize, v: f64) -> (usize, usize, f64, f64) {
        (r, c, v, 0.0)
    }

    fn vec_arb(vals: &[f64]) -> Vec<Arb> {
        vals.iter().map(|&v| Arb::from_f64(v)).collect()
    }

    #[test]
    fn identity_is_a_no_op() {
        let id = CsrArb::from_coo_f64(3, 3, 53, &[e(0, 0, 1.0), e(1, 1, 1.0), e(2, 2, 1.0)]);
        let x = vec_arb(&[2.0, -5.0, 7.0]);
        let y = id.spmv(&x);
        let mids: Vec<f64> = y.iter().map(|b| b.midpoint()).collect();
        assert_eq!(mids, vec![2.0, -5.0, 7.0]);
    }

    #[test]
    fn dense_2x2_product() {
        // [[1, 2], [3, 4]] · [5, 6] = [17, 39].
        let a = CsrArb::from_coo_f64(
            2,
            2,
            53,
            &[e(0, 0, 1.0), e(0, 1, 2.0), e(1, 0, 3.0), e(1, 1, 4.0)],
        );
        let y = a.spmv(&vec_arb(&[5.0, 6.0]));
        assert_eq!(y[0].midpoint(), 17.0);
        assert_eq!(y[1].midpoint(), 39.0);
        // Exact integer arithmetic: radii stay zero.
        assert_eq!(y[0].radius(53), 0.0);
        assert_eq!(y[1].radius(53), 0.0);
    }

    #[test]
    fn empty_row_yields_zero_ball() {
        // Row 1 has no stored entries.
        let a = CsrArb::from_coo_f64(2, 2, 53, &[e(0, 0, 9.0)]);
        let y = a.spmv(&vec_arb(&[1.0, 1.0]));
        assert_eq!(y[0].midpoint(), 9.0);
        assert!(y[1].is_exact_zero());
    }

    #[test]
    fn enclosure_contains_true_value() {
        // A has an inexact entry; the SpMV output must still
        // rigorously enclose the true product.
        // A = [[1/3 (as a ball)]], x = [3]  →  true y = 1.
        // Build 1/3 as a ball by dividing nothing — instead use a
        // known inexact ball: [0.333... ± 1e-3].
        let third = 1.0_f64 / 3.0;
        let a = CsrArb::from_coo_f64(1, 1, 64, &[(0, 0, third, 1e-3)]);
        let y = a.spmv(&vec_arb(&[3.0]));
        let (mid, rad) = y[0].to_pair(64);
        // The true value 3 * (1/3) = 1 lies within [mid - rad, mid + rad].
        assert!((mid - rad) <= 1.0 + 1e-12 && 1.0 <= (mid + rad) + 1e-12);
        assert!(rad >= 3.0 * 1e-3 - 1e-9); // radius scaled by |x| = 3
    }

    #[test]
    fn zero_matrix_yields_all_zero() {
        let z = CsrArb::zeros(3, 2, 53);
        let y = z.spmv(&vec_arb(&[1.0, 2.0]));
        assert_eq!(y.len(), 3);
        assert!(y.iter().all(|b| b.is_exact_zero()));
    }

    #[test]
    #[should_panic(expected = "x has length 2 but A has 3 columns")]
    fn dimension_mismatch_panics() {
        let a = CsrArb::zeros(2, 3, 53);
        let _ = a.spmv(&vec_arb(&[1.0, 2.0]));
    }

    /// `par_spmv` must agree with `spmv` on every input — same
    /// inner loop just fanned out across rayon workers.
    #[cfg(feature = "parallel")]
    #[test]
    fn par_spmv_matches_serial() {
        let a = CsrArb::from_coo_f64(
            3,
            4,
            64,
            &[
                e(0, 0, 1.0),
                e(0, 3, 2.0),
                e(1, 1, 3.0),
                e(2, 0, -1.0),
                e(2, 2, 4.0),
                e(2, 3, 5.0),
            ],
        );
        let x = vec_arb(&[2.0, 1.0, -1.0, 0.5]);
        let serial = a.spmv(&x);
        let parallel = crate::par_spmv(&a, &x);
        assert_eq!(serial.len(), parallel.len());
        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert_eq!(s.midpoint(), p.midpoint());
            assert_eq!(s.radius(64), p.radius(64));
        }
    }
}
