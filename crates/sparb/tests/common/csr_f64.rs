// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Plain-`f64` sparse CSR matrix for tests that compare sparb's
//! rigorous-ball SpMV against the naive floating-point baseline.
//!
//! This is *not* a sparb primitive — it lives in `tests/common/`
//! so the instability and parity tests can show the failure modes
//! plain `f64` falls into. The bench example (`examples/bench_compare.rs`)
//! keeps its own inline copy because `examples/` and `tests/` are
//! separate cargo build units and can't share a `mod`.

// Allow dead code — each integration-test binary uses a different
// subset of the common helpers, and Rust would otherwise flag the
// unused items in the per-binary compilation.
#![allow(dead_code)]

/// Sparse `rows × cols` matrix over `f64` in compressed-sparse-row
/// layout. Built from `(row, col, value)` triples with the same
/// dedupe-sum-drop-exact-zero rules sparb's `CsrArb` uses, so the
/// two are structurally comparable.
pub struct CsrF64 {
    rows: usize,
    cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl CsrF64 {
    /// Build from `(row, col, value)` triples. Duplicates are
    /// summed; entries summing to exactly `0.0` are dropped.
    pub fn from_triples(rows: usize, cols: usize, mut triples: Vec<(usize, usize, f64)>) -> Self {
        triples.sort_by_key(|&(r, c, _)| (r, c));
        let mut indptr = vec![0usize; rows + 1];
        let mut indices = Vec::with_capacity(triples.len());
        let mut data: Vec<f64> = Vec::with_capacity(triples.len());
        let mut iter = triples.into_iter();
        let Some((mut cr, mut cc, mut acc)) = iter.next() else {
            return Self {
                rows,
                cols,
                indptr,
                indices,
                data,
            };
        };
        for (r, c, v) in iter {
            if r == cr && c == cc {
                acc += v;
            } else {
                if acc != 0.0 {
                    indices.push(cc);
                    data.push(acc);
                    indptr[cr + 1] += 1;
                }
                cr = r;
                cc = c;
                acc = v;
            }
        }
        if acc != 0.0 {
            indices.push(cc);
            data.push(acc);
            indptr[cr + 1] += 1;
        }
        for r in 0..rows {
            indptr[r + 1] += indptr[r];
        }
        Self {
            rows,
            cols,
            indptr,
            indices,
            data,
        }
    }

    /// Plain `f64` SpMV `y = A · x` — no precision tracking, no
    /// enclosure. Naive left-to-right accumulation, which is what
    /// most production codes do unless they go out of their way to
    /// use compensated summation.
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols, "spmv: x.len() != cols");
        let mut y = vec![0.0_f64; self.rows];
        for (r, yr) in y.iter_mut().enumerate() {
            let s = self.indptr[r];
            let e = self.indptr[r + 1];
            let mut acc = 0.0_f64;
            for off in s..e {
                acc += self.data[off] * x[self.indices[off]];
            }
            *yr = acc;
        }
        y
    }
}
