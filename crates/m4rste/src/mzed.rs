// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! [`Mzed`] — a dense **bit-packed** GF(2^e) matrix.
//!
//! The layout mirrors M4RIE's `mzed_t` exactly: an `m × n` matrix over
//! GF(2^e) is stored as an `m4rst::Mzd` (a dense F_2 matrix) with
//! `n · e` columns — element `(i, j)`'s `e` polynomial-coefficient bits
//! occupy F_2-columns `[j·e, j·e + e)` of row `i`. So `m4rste` reuses
//! `m4rst`'s proven bit-packed F_2 storage rather than rolling its own.
//!
//! - `add` is a straight word-XOR over the packed backing store.
//! - `matmul` goes through the **bit-sliced** path in [`crate::matmul`]:
//!   slice into `e` F_2 bit-planes, run `m4rst`'s F_2 matmul kernel,
//!   cling back. This is "m4rste uses m4rst the way M4RIE uses M4RI".
//! - `rank` / `rref` use Gaussian elimination with log/antilog field
//!   tables and an M4RIE-style Newton-John–blocked inner loop.

use m4rst::Mzd;

use crate::gf2e::Gf2e;

/// A dense matrix over GF(2^e), bit-packed M4RIE-style.
///
/// Backed by an `m4rst::Mzd` of shape `rows × (cols · e)`. The field
/// context [`Gf2e`] travels with the matrix so binary operations can
/// assert the operands share a field.
#[derive(Debug, Clone)]
pub struct Mzed {
    pub(crate) ctx: Gf2e,
    /// Logical column count (the `m4rst::Mzd` has `cols · e` columns).
    pub(crate) cols: usize,
    /// Bit-packed F_2 backing store: `rows × (cols · e)`.
    pub(crate) packed: Mzd,
}

impl Mzed {
    /// All-zero `rows × cols` matrix over `ctx`.
    pub fn zeros(ctx: Gf2e, rows: usize, cols: usize) -> Self {
        let e = ctx.degree() as usize;
        Self {
            ctx,
            cols,
            packed: Mzd::new(rows, cols * e),
        }
    }

    /// `n × n` identity over `ctx`.
    pub fn identity(ctx: Gf2e, n: usize) -> Self {
        let mut m = Self::zeros(ctx, n, n);
        for i in 0..n {
            m.set(i, i, 1);
        }
        m
    }

    /// Build from a row-major `u16` buffer. Each element's low `e` bits
    /// are packed into the backing store; higher bits are ignored.
    ///
    /// # Panics
    ///
    /// Panics if `buf.len() != rows * cols`.
    pub fn from_u16_buffer(ctx: Gf2e, rows: usize, cols: usize, buf: &[u16]) -> Self {
        assert_eq!(
            buf.len(),
            rows * cols,
            "Mzed::from_u16_buffer: expected {} elements, got {}",
            rows * cols,
            buf.len()
        );
        let mut m = Self::zeros(ctx, rows, cols);
        let degree = ctx.degree() as usize;
        let mask = (1_u64 << degree) - 1;
        // Pack complete elements into the zeroed row storage. Writing one or
        // two word fragments per element avoids the former per-bit `set`
        // loop, which was a material terminal cost after elimination.
        for row in 0..rows {
            let packed_row = m.packed.row_words_mut(row);
            for col in 0..cols {
                let value = u64::from(buf[row * cols + col]) & mask;
                let bit = col * degree;
                let word = bit / 64;
                let offset = bit % 64;
                packed_row[word] |= value << offset;
                if offset + degree > 64 {
                    packed_row[word + 1] |= value >> (64 - offset);
                }
            }
        }
        m
    }

    /// Materialise to a row-major `u16` buffer (one `u16` per element).
    pub fn to_u16_buffer(&self) -> Vec<u16> {
        let rows = self.rows();
        let cols = self.cols;
        let mut out = vec![0u16; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] = self.get(i, j);
            }
        }
        out
    }

    /// The field context.
    pub fn ctx(&self) -> Gf2e {
        self.ctx
    }

    /// Row count.
    pub fn rows(&self) -> usize {
        self.packed.rows()
    }

    /// Column count.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Read element `(i, j)` — the `e`-bit group at packed columns
    /// `[j·e, j·e + e)` of row `i`.
    pub fn get(&self, i: usize, j: usize) -> u16 {
        let e = self.ctx.degree() as usize;
        self.packed.extract_bits(i, j * e, e) as u16
    }

    /// Write element `(i, j)`. Only the low `e` bits of `v` are stored.
    pub fn set(&mut self, i: usize, j: usize, v: u16) {
        let e = self.ctx.degree() as usize;
        for k in 0..e {
            self.packed.set(i, j * e + k, ((v >> k) & 1) as u8);
        }
    }

    /// Element-wise sum — a straight XOR over the packed backing store
    /// (addition in characteristic 2). Shape and field must match.
    pub fn add(a: &Mzed, b: &Mzed) -> Mzed {
        assert_eq!(a.ctx, b.ctx, "Mzed::add: field mismatch");
        assert_eq!(
            (a.rows(), a.cols),
            (b.rows(), b.cols),
            "Mzed::add: shape mismatch"
        );
        let mut out = a.packed.clone();
        for (o, &w) in out.as_words_mut().iter_mut().zip(b.packed.as_words()) {
            *o ^= w;
        }
        Mzed {
            ctx: a.ctx,
            cols: a.cols,
            packed: out,
        }
    }

    /// Allocation-free in-place counterpart to [`Mzed::add`]. `self += other`
    /// is a
    /// straight XOR over the packed backing store. Shape and field
    /// must match.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rste::{Gf2e, Mzed};
    /// let f = Gf2e::new(0x11B);
    /// let mut a = Mzed::from_u16_buffer(f, 1, 3, &[10, 20, 30]);
    /// let b = Mzed::from_u16_buffer(f, 1, 3, &[1, 2, 3]);
    /// a.add_into(&b);
    /// assert_eq!(a.to_u16_buffer(), Mzed::add(&Mzed::from_u16_buffer(f, 1, 3, &[10, 20, 30]), &b).to_u16_buffer());
    /// ```
    pub fn add_into(&mut self, other: &Mzed) {
        assert_eq!(self.ctx, other.ctx, "Mzed::add_into: field mismatch");
        assert_eq!(
            (self.rows(), self.cols),
            (other.rows(), other.cols),
            "Mzed::add_into: shape mismatch"
        );
        for (o, &w) in self
            .packed
            .as_words_mut()
            .iter_mut()
            .zip(other.packed.as_words())
        {
            *o ^= w;
        }
    }

    /// GF(2^e) matmul `C = A · B` via the bit-sliced path — see
    /// [`crate::matmul`].
    ///
    /// # Panics
    ///
    /// Panics on field mismatch or inner-dimension mismatch.
    pub fn matmul(a: &Mzed, b: &Mzed) -> Mzed {
        crate::matmul::matmul(a, b)
    }

    /// Caller-owned-output counterpart to [`Mzed::matmul`]. `dst` must already
    /// be `a.rows() × b.cols()`
    /// over the same field; this fully overwrites it.
    ///
    /// Writes the result straight into `dst.packed` (no top-level
    /// allocation, no copy) and computes every leaf F_2 product via
    /// `m4rst::Mzd::matmul_into` against a shared scratch buffer
    /// rather than allocating one per leaf. See
    /// [`crate::matmul::matmul_into`] for the internal format-conversion and
    /// Karatsuba scratch allocations.
    ///
    /// # Panics
    ///
    /// On field or shape mismatch.
    pub fn matmul_into(dst: &mut Mzed, a: &Mzed, b: &Mzed) {
        crate::matmul::matmul_into(dst, a, b)
    }

    /// Gaussian elimination over GF(2^e). Materialises the elements to
    /// a `Vec<u16>`, eliminates, repacks. Returns `(echelon_matrix,
    /// rank)`; `full = true` yields RREF.
    ///
    /// The field arithmetic uses `Gf2eTables` (log/antilog lookups,
    /// built once per call). The elimination inner loop uses M4RIE-style
    /// **Newton-John tables**: for each pivot row a table
    /// `nj[c] = c · pivot_row` is precomputed, so eliminating a row is a
    /// straight `row ^= nj[factor]` XOR rather than a column of field
    /// multiplies. That trade pays off when the field is small relative
    /// to the row count (`2^e <= rows` — which also bounds the table to
    /// no larger than the input matrix); otherwise the inner loop falls
    /// back to direct table-multiplies — with a **hardware-GF(2^8) SIMD
    /// path on GF(256) with the AES polynomial** (x86_64 GFNI or
    /// aarch64 NEON, see `crate::simd`) that closes the otherwise large
    /// GF(256) large-N gap.
    fn echelonize_buffer(&self, full: bool) -> (Vec<u16>, usize) {
        let f = self.ctx;
        let rows = self.rows();
        let cols = self.cols;
        let tables = f.tables();
        let field_size = 1usize << f.degree();
        // Newton-John trades `rows·cols` field-multiplies for
        // `rows·cols` XORs plus a `field_size·cols` table build per
        // pivot. It only pays once the field is comfortably smaller
        // than the row count — at the `field_size == rows` break-even
        // the table build alone eats the saving. Require a 2× margin;
        // that also keeps the table no bigger than half the input.
        let newton_john = 2 * field_size <= rows;
        // Hardware GF(2^8) is wired in for the AES polynomial only;
        // other GF(256) representations retain the scalar path.
        let simd_gf256_aes = f.degree() == 8 && f.minpoly() == 0x11B;
        // GF(16): SIMD-accelerate the Newton-John table build via a
        // 16-byte multiplication table + `vpshufb`/`vqtbl`. Works for
        // any GF(16) minpoly (the table absorbs the minpoly).
        let simd_gf16 = f.degree() == 4;

        let mut m = self.to_u16_buffer();
        // Reusable Newton-John table: `nj[c * cols + j] = c · pivot[j]`.
        let mut nj = if newton_john {
            vec![0u16; field_size * cols]
        } else {
            Vec::new()
        };
        let mut rank = 0usize;
        for col in 0..cols {
            if rank == rows {
                break;
            }
            // Find a pivot at or below `rank` in this column.
            let mut pivot = None;
            for r in rank..rows {
                if m[r * cols + col] != 0 {
                    pivot = Some(r);
                    break;
                }
            }
            let Some(pr) = pivot else { continue };
            // Swap the pivot row up to `rank`.
            if pr != rank {
                for c in 0..cols {
                    m.swap(pr * cols + c, rank * cols + c);
                }
            }
            // Normalise the pivot row so the pivot entry is 1.
            let inv = tables.inv(m[rank * cols + col]);
            // Entries before `col` are already zero in every remaining
            // pivot candidate.  Restricting the arithmetic to the live row
            // suffix is especially material for rank/RREF: the old whole-row
            // loops did almost twice the necessary field work on a dense
            // square matrix.
            for c in col..cols {
                let v = m[rank * cols + c];
                if v != 0 {
                    m[rank * cols + c] = tables.mul(v, inv);
                }
            }

            let elim_start = if full { 0 } else { rank + 1 };

            if newton_john {
                // Build nj[c][·] = c · (normalised pivot row). Row c=0
                // is all-zero (already), c=1 is the pivot row itself.
                // For GF(16) the c=2..15 scaled-row computations go
                // through SIMD byte-shuffle (one instruction per 16
                // GF(16) elements vs a scalar log-table mul each);
                // other fields use the scalar table-multiply path.
                let pr_base = rank * cols;
                nj[cols + col..2 * cols].copy_from_slice(&m[pr_base + col..pr_base + cols]);
                if simd_gf16 {
                    let minpoly = f.minpoly();
                    let pivot_slice: &[u16] = &m[pr_base + col..pr_base + cols];
                    for c in 2..field_size {
                        let base = c * cols;
                        let dst_slice = &mut nj[base + col..base + cols];
                        crate::simd::gf16_apply_factor_row(
                            c as u8,
                            minpoly,
                            pivot_slice,
                            dst_slice,
                        );
                    }
                } else {
                    for c in 2..field_size {
                        let cu = c as u16;
                        let base = c * cols;
                        for j in col..cols {
                            nj[base + j] = tables.mul(cu, m[pr_base + j]);
                        }
                    }
                }
                // Eliminate: each row is a single XOR against nj[factor].
                for r in elim_start..rows {
                    if r == rank {
                        continue;
                    }
                    let factor = m[r * cols + col] as usize;
                    if factor == 0 {
                        continue;
                    }
                    let (rb, nb) = (r * cols, factor * cols);
                    for j in col..cols {
                        m[rb + j] ^= nj[nb + j];
                    }
                }
            } else {
                // Direct table-multiply (or SIMD, for GF(256)/AES)
                // inner loop. The pivot row at `rank * cols` and the
                // target row at `r * cols` are disjoint slices of `m`
                // (since `r != rank`); split via `split_at_mut` to
                // hand both to the inner kernel.
                let pr_base = rank * cols;
                for r in elim_start..rows {
                    if r == rank {
                        continue;
                    }
                    let factor = m[r * cols + col];
                    if factor == 0 {
                        continue;
                    }
                    let target_base = r * cols;
                    // Whole rows whose length is a multiple of the vector
                    // width avoid the scalar tail in the GFNI/NEON kernel.
                    // Earlier columns of the pivot row are zero, so retaining
                    // them costs cheap vector XORs and is materially faster
                    // than a short scalar GF(256) tail on every eliminated
                    // row. Other fields still use the exact live suffix.
                    let row_start = if simd_gf256_aes { 0 } else { col };
                    let (pivot_slice, target_slice): (&[u16], &mut [u16]) = if target_base < pr_base
                    {
                        let (lo, hi) = m.split_at_mut(pr_base);
                        (
                            &hi[row_start..cols],
                            &mut lo[target_base + row_start..target_base + cols],
                        )
                    } else {
                        let (lo, hi) = m.split_at_mut(target_base);
                        (
                            &lo[pr_base + row_start..pr_base + cols],
                            &mut hi[row_start..cols],
                        )
                    };
                    if simd_gf256_aes {
                        crate::simd::gf256_aes_xor_scaled_row(
                            factor as u8,
                            pivot_slice,
                            target_slice,
                        );
                    } else {
                        for c in 0..pivot_slice.len() {
                            let p = pivot_slice[c];
                            if p != 0 {
                                target_slice[c] ^= tables.mul(factor, p);
                            }
                        }
                    }
                }
            }
            rank += 1;
        }
        (m, rank)
    }

    fn echelonize(&self, full: bool) -> (Mzed, usize) {
        let (buffer, rank) = self.echelonize_buffer(full);
        (
            Mzed::from_u16_buffer(self.ctx, self.rows(), self.cols, &buffer),
            rank,
        )
    }

    /// Rank over GF(2^e).
    pub fn rank(&self) -> usize {
        // Rank needs no materialized echelon form. Keep elimination in its
        // contiguous element buffer and avoid repacking an output that the
        // caller would immediately discard.
        self.echelonize_buffer(false).1
    }

    /// Reduced row echelon form. Returns `(rref_matrix, rank)`.
    pub fn rref(&self) -> (Mzed, usize) {
        self.echelonize(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gf4() -> Gf2e {
        Gf2e::new(0b111)
    }

    #[test]
    fn get_set_round_trip_bit_packed() {
        // Exercise the e-bit packing across element boundaries.
        let f = Gf2e::new(0x11B); // GF(256), e = 8
        let mut m = Mzed::zeros(f, 3, 5);
        for i in 0..3 {
            for j in 0..5 {
                m.set(i, j, ((i * 5 + j) as u16 * 7 + 1) & 0xFF);
            }
        }
        for i in 0..3 {
            for j in 0..5 {
                assert_eq!(m.get(i, j), ((i * 5 + j) as u16 * 7 + 1) & 0xFF);
            }
        }
    }

    #[test]
    fn u16_buffer_round_trip() {
        let f = Gf2e::new(0b1_0011); // GF(16), e = 4
        let buf: Vec<u16> = (0..12).map(|x| (x as u16 * 3) & 0xF).collect();
        let m = Mzed::from_u16_buffer(f, 3, 4, &buf);
        assert_eq!(m.to_u16_buffer(), buf);
    }

    #[test]
    fn identity_matmul_is_noop() {
        let f = gf4();
        let id = Mzed::identity(f, 2);
        let a = Mzed::from_u16_buffer(f, 2, 2, &[2, 3, 1, 2]);
        assert_eq!(Mzed::matmul(&id, &a).to_u16_buffer(), a.to_u16_buffer());
        assert_eq!(Mzed::matmul(&a, &id).to_u16_buffer(), a.to_u16_buffer());
    }

    #[test]
    fn identity_rank_full() {
        let f = gf4();
        assert_eq!(Mzed::identity(f, 5).rank(), 5);
    }

    #[test]
    fn rank_drops_on_dependent_rows() {
        let f = gf4();
        // Row 2 = Row 0 (literal duplicate) -> rank 2, not 3.
        let m = Mzed::from_u16_buffer(f, 3, 3, &[1, 2, 3, 0, 1, 2, 1, 2, 3]);
        assert_eq!(m.rank(), 2);
    }

    #[test]
    fn rref_of_identity_is_identity() {
        let f = gf4();
        let id = Mzed::identity(f, 4);
        let (r, rank) = id.rref();
        assert_eq!(rank, 4);
        assert_eq!(r.to_u16_buffer(), id.to_u16_buffer());
    }

    #[test]
    fn add_is_self_inverse() {
        let f = Gf2e::new(0x11B);
        let a = Mzed::from_u16_buffer(f, 2, 3, &[10, 20, 30, 40, 50, 60]);
        let z = Mzed::add(&a, &a);
        assert_eq!(z.to_u16_buffer(), vec![0u16; 6]);
    }

    #[test]
    fn add_into_matches_add() {
        let f = Gf2e::new(0x11B);
        let a = Mzed::from_u16_buffer(f, 2, 3, &[10, 20, 30, 40, 50, 60]);
        let b = Mzed::from_u16_buffer(f, 2, 3, &[1, 2, 3, 4, 5, 6]);
        let want = Mzed::add(&a, &b).to_u16_buffer();
        let mut a_mut = a;
        a_mut.add_into(&b);
        assert_eq!(a_mut.to_u16_buffer(), want);
    }

    #[test]
    fn add_into_on_reused_buffer_does_not_accumulate_stale_data() {
        // add_into must fully overwrite via XOR, not leave any residue
        // from a prior unrelated call on the same allocation.
        let f = Gf2e::new(0x11B);
        let mut acc = Mzed::from_u16_buffer(f, 1, 2, &[99, 200]); // stale content
        acc.add_into(&Mzed::from_u16_buffer(f, 1, 2, &[99, 200])); // self-cancel: now zero
        assert_eq!(acc.to_u16_buffer(), vec![0, 0]);
        acc.add_into(&Mzed::from_u16_buffer(f, 1, 2, &[7, 8]));
        assert_eq!(acc.to_u16_buffer(), vec![7, 8]);
    }

    #[test]
    fn matmul_into_matches_matmul() {
        let f = Gf2e::new(0x11B);
        let a = Mzed::from_u16_buffer(f, 2, 2, &[2, 1, 1, 3]);
        let b = Mzed::from_u16_buffer(f, 2, 2, &[3, 0, 2, 2]);
        let want = Mzed::matmul(&a, &b).to_u16_buffer();
        let mut got = Mzed::zeros(f, 2, 2);
        Mzed::matmul_into(&mut got, &a, &b);
        assert_eq!(got.to_u16_buffer(), want);
    }

    #[test]
    fn matmul_into_on_reused_buffer_does_not_accumulate_stale_data() {
        let f = Gf2e::new(0x11B);
        let a1 = Mzed::from_u16_buffer(f, 2, 2, &[2, 1, 1, 3]);
        let b1 = Mzed::from_u16_buffer(f, 2, 2, &[3, 0, 2, 2]);
        let a2 = Mzed::from_u16_buffer(f, 2, 2, &[5, 6, 7, 8]);
        let b2 = Mzed::from_u16_buffer(f, 2, 2, &[9, 10, 200, 12]);

        let mut dst = Mzed::zeros(f, 2, 2);
        Mzed::matmul_into(&mut dst, &a1, &b1); // first call: populates dst
        Mzed::matmul_into(&mut dst, &a2, &b2); // reuse: must fully overwrite

        let want = Mzed::matmul(&a2, &b2).to_u16_buffer();
        assert_eq!(dst.to_u16_buffer(), want);
    }

    /// Naive RREF straight off `Gf2e::{mul, inv}` (carry-less) — the
    /// independent oracle for `echelonize`. RREF is unique over a
    /// field, so this must match bit-for-bit regardless of which inner
    /// loop (Newton-John or table-multiply) `echelonize` chose.
    fn naive_rref(a: &Mzed) -> (Vec<u16>, usize) {
        let f = a.ctx();
        let (rows, cols) = (a.rows(), a.cols());
        let mut m = a.to_u16_buffer();
        let mut rank = 0;
        for col in 0..cols {
            if rank == rows {
                break;
            }
            let Some(pr) = (rank..rows).find(|&r| m[r * cols + col] != 0) else {
                continue;
            };
            if pr != rank {
                for c in 0..cols {
                    m.swap(pr * cols + c, rank * cols + c);
                }
            }
            let inv = f.inv(m[rank * cols + col]);
            for c in 0..cols {
                let v = m[rank * cols + c];
                m[rank * cols + c] = f.mul(v, inv);
            }
            for r in 0..rows {
                if r == rank {
                    continue;
                }
                let factor = m[r * cols + col];
                if factor != 0 {
                    for c in 0..cols {
                        let p = m[rank * cols + c];
                        m[r * cols + c] ^= f.mul(factor, p);
                    }
                }
            }
            rank += 1;
        }
        (m, rank)
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    fn random(f: Gf2e, rows: usize, cols: usize, seed: u64) -> Mzed {
        let mask: u16 = if f.degree() >= 16 {
            u16::MAX
        } else {
            (1u16 << f.degree()) - 1
        };
        let mut rng = Rng(seed | 1);
        let buf: Vec<u16> = (0..rows * cols)
            .map(|_| (rng.next() as u16) & mask)
            .collect();
        Mzed::from_u16_buffer(f, rows, cols, &buf)
    }

    #[test]
    fn echelonize_matches_naive_both_inner_loops() {
        // GF(16) 20×24: field_size 16 ≤ rows 20 → Newton-John path.
        // GF(256) 12×15: field_size 256 > rows 12 → table-multiply path.
        // GF(4) 8×8: field_size 4 ≤ rows 8 → Newton-John, square.
        let cases: &[(u64, usize, usize)] = &[(0b1_0011, 20, 24), (0x11B, 12, 15), (0b111, 8, 8)];
        for &(minpoly, rows, cols) in cases {
            let f = Gf2e::new(minpoly);
            let a = random(f, rows, cols, 0xE12 ^ minpoly ^ (rows as u64));
            let (rref_mat, rank) = a.rref();
            let (want_buf, want_rank) = naive_rref(&a);
            assert_eq!(
                rank, want_rank,
                "rank mismatch for minpoly={minpoly:#x} {rows}x{cols}"
            );
            assert_eq!(
                rref_mat.to_u16_buffer(),
                want_buf,
                "RREF mismatch for minpoly={minpoly:#x} {rows}x{cols}"
            );
            assert_eq!(a.rank(), want_rank, "rank() disagrees with rref rank");
        }
    }
}
