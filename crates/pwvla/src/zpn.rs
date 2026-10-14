// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Guard-bit packed storage for Z/p^n with odd p.
//!
//! For p = 2 the native `Mat<uN>` layout works because wrapping
//! arithmetic on uN is already Z/2^N. For odd p, we need explicit
//! modular reduction, which is fast IF multiple elements are packed
//! per machine word so SIMD lanes work on many at once.
//!
//! ## Layout
//!
//! Each element of Z/p^n fits in `ceil(log_2(p^n))` bits. We allocate
//! `slot_bits = element_bits + guard_bits` per element, where the
//! guard zone absorbs carries from accumulation without leaking into
//! neighboring slots. A u64 holds `64 / slot_bits` packed elements.
//!
//! For schoolbook matmul that sums k products per output cell, we need
//! `guard_bits ≥ ceil(log_2(k) + 1)` to accumulate k terms before a
//! reduction step is mandatory. Smaller guard → larger pack ratio →
//! more reductions; larger guard → fewer reductions at the cost of
//! throughput.
//!
//! ## Example
//!
//! For `Z/5` (p = 5, n = 1): element_bits = 3. With guard = 2, slot = 5.
//! 12 elements per u64 — a ~4× throughput win over one-per-u64.

/// A dense vector of Z/p^n elements stored with guard bits between
/// slots so accumulation can proceed without immediate reduction.
#[derive(Debug, Clone)]
pub struct PackedZpn {
    /// The prime `p`.
    pub p: u32,
    /// The exponent `n` (so the ring is Z/p^n).
    pub n: u32,
    /// Pre-computed `p^n`.
    pub p_pow_n: u64,
    /// Number of logical elements.
    pub length: usize,
    /// Bits per slot (element bits + guard bits).
    pub slot_bits: u32,
    /// Number of slots per u64 word.
    pub slots_per_word: u32,
    /// Mask of `slot_bits` ones.
    pub slot_mask: u64,
    /// Packed storage.
    pub data: Vec<u64>,
}

impl PackedZpn {
    /// Construct a packed vector of `length` zeros, with a default
    /// guard of 2 bits (supports accumulation of up to ~4 terms before
    /// reduction). Use `with_guard` for custom guard sizing.
    pub fn new(p: u32, n: u32, length: usize) -> Self {
        Self::with_guard(p, n, length, 2)
    }

    /// Construct a packed vector of `length` zeros with caller-chosen
    /// `guard_bits` (extra bits per slot reserved for accumulation
    /// headroom before a reduction must be issued).
    pub fn with_guard(p: u32, n: u32, length: usize, guard_bits: u32) -> Self {
        assert!(p >= 2, "p must be at least 2");
        assert!(p % 2 == 1 || p == 2, "p must be prime");
        assert!(n >= 1, "n must be at least 1");

        let p_pow_n: u64 = (p as u64).checked_pow(n).expect("p^n overflow u64");
        let element_bits = if p_pow_n <= 1 {
            0
        } else {
            64 - (p_pow_n - 1).leading_zeros()
        };
        let slot_bits = element_bits + guard_bits;
        assert!(slot_bits <= 32, "slot too wide ({slot_bits} bits)");
        assert!(slot_bits >= 1, "slot must be at least 1 bit");
        let slots_per_word = 64 / slot_bits;
        let num_words = length.div_ceil(slots_per_word as usize);
        let slot_mask = if slot_bits >= 64 {
            !0u64
        } else {
            (1u64 << slot_bits) - 1
        };
        Self {
            p,
            n,
            p_pow_n,
            length,
            slot_bits,
            slots_per_word,
            slot_mask,
            data: vec![0u64; num_words],
        }
    }

    /// Read element at logical index `i`. Returns the **raw slot value**
    /// (may exceed `p^n` if there was accumulation without reduction).
    pub fn get_raw(&self, i: usize) -> u64 {
        debug_assert!(i < self.length);
        let word_idx = i / self.slots_per_word as usize;
        let slot_idx = i % self.slots_per_word as usize;
        let shift = (slot_idx as u32) * self.slot_bits;
        (self.data[word_idx] >> shift) & self.slot_mask
    }

    /// Read element at logical index `i` modulo `p^n`.
    pub fn get(&self, i: usize) -> u64 {
        self.get_raw(i) % self.p_pow_n
    }

    /// Write element `v` at logical index `i`. `v` is assumed to fit
    /// in `slot_bits` (caller reduces modulo `p^n` if needed).
    pub fn set(&mut self, i: usize, v: u64) {
        debug_assert!(i < self.length);
        debug_assert!(v <= self.slot_mask, "v = {v} exceeds slot mask");
        let word_idx = i / self.slots_per_word as usize;
        let slot_idx = i % self.slots_per_word as usize;
        let shift = (slot_idx as u32) * self.slot_bits;
        let cleared = self.data[word_idx] & !(self.slot_mask << shift);
        self.data[word_idx] = cleared | ((v & self.slot_mask) << shift);
    }

    /// Reduce slot `i` modulo `p^n` in place.
    pub fn reduce_slot(&mut self, i: usize) {
        let v = self.get_raw(i);
        if v >= self.p_pow_n {
            self.set(i, v % self.p_pow_n);
        }
    }

    /// Reduce every slot modulo `p^n`.
    pub fn reduce_all(&mut self) {
        for i in 0..self.length {
            self.reduce_slot(i);
        }
    }

    /// Build a `PackedZpn` from a slice of u64 values (each assumed
    /// already in `[0, p^n)`).
    pub fn from_slice(p: u32, n: u32, data: &[u64]) -> Self {
        let mut out = Self::new(p, n, data.len());
        for (i, &v) in data.iter().enumerate() {
            debug_assert!(v < out.p_pow_n, "input {v} ≥ p^n = {}", out.p_pow_n);
            out.set(i, v);
        }
        out
    }

    /// Extract the elements as a `Vec<u64>` in [0, p^n).
    pub fn to_vec(&self) -> Vec<u64> {
        (0..self.length).map(|i| self.get(i)).collect()
    }

    /// How many u64 words back the storage. Useful for bench / sanity.
    pub fn num_words(&self) -> usize {
        self.data.len()
    }

    /// Build a word with `p^n` tiled across all slots. Used as the
    /// "subtract me" constant for packed subtraction / negation.
    fn p_pow_n_word(&self) -> u64 {
        let mut w = 0u64;
        for i in 0..self.slots_per_word {
            w |= self.p_pow_n << (i * self.slot_bits);
        }
        w
    }

    /// Element-wise `self + other` in Z/p^n — word-parallel, guard-bit
    /// absorbing the per-slot carry. Each slot of the sum is at most
    /// `2·(p^n − 1) < 2^{slot_bits}` (requires guard ≥ 1), so no
    /// cross-slot contamination; `reduce_all` then normalizes.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!(self.length, other.length);
        let mut out = self.clone();
        for i in 0..self.data.len() {
            out.data[i] = self.data[i].wrapping_add(other.data[i]);
        }
        out.reduce_all();
        out
    }

    /// Element-wise `self − other` in Z/p^n.
    ///
    /// Uses the identity `a − b ≡ a + (p^n − b) (mod p^n)`. For reduced
    /// `b ∈ [0, p^n)`, `p^n − b ∈ [1, p^n]` always, so subtracting
    /// word-parallel from the `p_pow_n_word` tile never borrows
    /// across slots. The sum `a + (p^n − b)` lies in `[1, 2·p^n)`,
    /// absorbed by 1 guard bit; `reduce_all` finalizes.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!(self.length, other.length);
        let p_word = self.p_pow_n_word();
        let mut out = self.clone();
        for i in 0..self.data.len() {
            let neg_b = p_word.wrapping_sub(other.data[i]);
            out.data[i] = self.data[i].wrapping_add(neg_b);
        }
        out.reduce_all();
        out
    }

    /// Element-wise `−self` in Z/p^n. Each slot: `(p^n − s) mod p^n`.
    /// The `reduce_all` step normalizes the `p^n → 0` case.
    pub fn neg(&self) -> Self {
        let p_word = self.p_pow_n_word();
        let mut out = self.clone();
        for i in 0..self.data.len() {
            out.data[i] = p_word.wrapping_sub(self.data[i]);
        }
        out.reduce_all();
        out
    }

    /// Broadcast scalar multiply. `s` is assumed to be in `[0, p^n)`.
    /// Uses Barrett reduction per element — avoids a hardware divide.
    pub fn scalar_mul(&self, s: u64) -> Self {
        assert!(s < self.p_pow_n, "scalar must be in [0, p^n)");
        let reducer = BarrettReducer::new(self.p_pow_n);
        let mut out = Self::new(self.p, self.n, self.length);
        for i in 0..self.length {
            let prod = self.get(i).wrapping_mul(s);
            out.set(i, reducer.reduce(prod));
        }
        out
    }

    /// Prime-field scalar inverse of every non-zero element
    /// (requires `self.n == 1`, i.e., F_p). Zero elements stay zero;
    /// non-zero elements become their multiplicative inverse in F_p.
    pub fn fp_inverse_elementwise(&self) -> Self {
        assert_eq!(
            self.n, 1,
            "fp_inverse_elementwise: only defined for F_p (n=1)"
        );
        let p = self.p as u64;
        let mut out = Self::new(self.p, self.n, self.length);
        for i in 0..self.length {
            let v = self.get(i);
            let inv = if v == 0 {
                0
            } else {
                fp_scalar_inverse(v, p).expect("non-zero in F_p must have inverse")
            };
            out.set(i, inv);
        }
        out
    }

    /// Element-wise multiplication `self ∘ other` in Z/p^n using
    /// Barrett reduction per slot.
    pub fn mul_elementwise(&self, other: &Self) -> Self {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!(self.length, other.length);
        let reducer = BarrettReducer::new(self.p_pow_n);
        let mut out = Self::new(self.p, self.n, self.length);
        for i in 0..self.length {
            let prod = self.get(i).wrapping_mul(other.get(i));
            out.set(i, reducer.reduce(prod));
        }
        out
    }
}

// ─── Dense matrix over Z/p^n ───────────────────────────────────────

/// Dense row-major matrix over Z/p^n, backed by `PackedZpn` storage.
#[derive(Debug, Clone)]
pub struct MatZpn {
    /// Prime base of the ring.
    pub p: u32,
    /// Exponent: the ring is `Z/p^n`.
    pub n: u32,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row-major packed-element backing storage of length `rows * cols`.
    pub data: PackedZpn,
}

impl MatZpn {
    /// Construct an all-zero `rows × cols` matrix over `Z/p^n`.
    pub fn zeros(p: u32, n: u32, rows: usize, cols: usize) -> Self {
        Self {
            p,
            n,
            rows,
            cols,
            data: PackedZpn::new(p, n, rows * cols),
        }
    }

    /// Construct a `rows × cols` matrix over `Z/p^n` with cell `(i, j) =
    /// f(i, j) mod p^n`.
    pub fn from_fn<F: FnMut(usize, usize) -> u64>(
        p: u32,
        n: u32,
        rows: usize,
        cols: usize,
        mut f: F,
    ) -> Self {
        let mut m = Self::zeros(p, n, rows, cols);
        let modulus = m.data.p_pow_n;
        for i in 0..rows {
            for j in 0..cols {
                // Reduce mod p^n as documented: `set` packs into slot_bits
                // and requires `v <= slot_mask`, so an out-of-range `f(i, j)`
                // (e.g. an unreduced product) would otherwise corrupt
                // adjacent slots in release and trip the debug assert.
                m.set(i, j, f(i, j) % modulus);
            }
        }
        m
    }

    /// Construct the `size × size` identity matrix over `Z/p^n`.
    pub fn identity(p: u32, n: u32, size: usize) -> Self {
        Self::from_fn(p, n, size, size, |i, j| u64::from(i == j))
    }

    /// Alias for [`MatZpn::identity`] — the NumPy/MATLAB spelling (`eye`
    /// ≈ the letter "I"). Provided for callers who reach for that name.
    pub fn eye(p: u32, n: u32, size: usize) -> Self {
        Self::identity(p, n, size)
    }

    /// Read cell `(i, j)`. Panics on out-of-bounds in debug builds.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> u64 {
        debug_assert!(i < self.rows && j < self.cols);
        self.data.get(i * self.cols + j)
    }

    /// Write `v mod p^n` to cell `(i, j)`. Panics on out-of-bounds in
    /// debug builds.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: u64) {
        debug_assert!(i < self.rows && j < self.cols);
        self.data.set(i * self.cols + j, v);
    }

    /// Element-wise `-A` in Z/p^n.
    pub fn neg(&self) -> Self {
        Self {
            p: self.p,
            n: self.n,
            rows: self.rows,
            cols: self.cols,
            data: self.data.neg(),
        }
    }

    /// Element-wise `A + B` in Z/p^n.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Self {
            p: self.p,
            n: self.n,
            rows: self.rows,
            cols: self.cols,
            data: self.data.add(&other.data),
        }
    }

    /// Element-wise `A − B` in Z/p^n.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Self {
            p: self.p,
            n: self.n,
            rows: self.rows,
            cols: self.cols,
            data: self.data.sub(&other.data),
        }
    }

    /// Matrix-vector product `A · v` over Z/p^n with Barrett
    /// reduction per inner-product step. Parallel over output rows
    /// via rayon. Primitive behind Wiedemann / Lanczos iteration.
    /// Expects `v.len() == self.cols` and entries `v[j] ∈ [0, p^n)`.
    pub fn mat_vec(&self, v: &[u64]) -> Vec<u64> {
        use rayon::prelude::*;
        assert_eq!(self.cols, v.len(), "mat_vec: A.cols != v.len()");
        let p_pow_n = self.data.p_pow_n;
        let reducer = BarrettReducer::new(p_pow_n);
        let (rows, cols) = (self.rows, self.cols);
        let a_unpacked = self.data.to_vec();
        (0..rows)
            .into_par_iter()
            .map(|i| {
                let mut acc: u64 = 0;
                for j in 0..cols {
                    let a_ij = a_unpacked[i * cols + j];
                    if a_ij == 0 {
                        continue;
                    }
                    let term = reducer.reduce(a_ij.wrapping_mul(v[j]));
                    let s = acc + term;
                    acc = if s >= p_pow_n { s - p_pow_n } else { s };
                }
                acc
            })
            .collect()
    }

    /// Full BLAS `gemv`: `y' = α · op(A) · x + β · y`.
    ///
    /// Generalizes `mat_vec` (which is the `α=1, β=0, op=NoTrans`
    /// special case) to the complete Level-2 signature: scaling
    /// factor on the matrix–vector product, transpose toggle, and
    /// a β-scaled accumulation onto the incoming `y`. Returns the
    /// updated `y` as a new `Vec<u64>` (Rust-immutable-builder
    /// style, matching `matmul`/`trmm`/`trsm`).
    ///
    /// Dimension rules:
    /// - `op = NoTrans`: `A` is `m × n`, `x.len() == n`, `y.len() == m`.
    /// - `op = Trans`:   `A` is `m × n`, `x.len() == m`, `y.len() == n`.
    ///
    /// Scalar Barrett with a `u128` accumulator per output row;
    /// α and β fold in once at the end of each row. Parallel over
    /// output rows via rayon.
    pub fn gemv(&self, alpha: u64, op: TransA, x: &[u64], beta: u64, y: &[u64]) -> Vec<u64> {
        use rayon::prelude::*;
        let modulus = self.data.p_pow_n;
        assert!(alpha < modulus, "gemv: alpha must be in [0, p^n)");
        assert!(beta < modulus, "gemv: beta must be in [0, p^n)");

        let (out_len, in_len) = match op {
            TransA::NoTrans => (self.rows, self.cols),
            TransA::Trans => (self.cols, self.rows),
        };
        assert_eq!(x.len(), in_len, "gemv: x length mismatch");
        assert_eq!(y.len(), out_len, "gemv: y length mismatch");

        let a_unpacked = self.data.to_vec();
        let a_cols = self.cols;
        let md = modulus as u128;
        let alpha128 = alpha as u128;
        let beta128 = beta as u128;

        (0..out_len)
            .into_par_iter()
            .map(|i| {
                let mut acc: u128 = 0;
                for j in 0..in_len {
                    let a_val = match op {
                        TransA::NoTrans => a_unpacked[i * a_cols + j],
                        TransA::Trans => a_unpacked[j * a_cols + i],
                    };
                    acc = (acc + (a_val as u128) * (x[j] as u128)) % md;
                }
                let scaled = (acc * alpha128) % md;
                let beta_term = (beta128 * y[i] as u128) % md;
                ((scaled + beta_term) % md) as u64
            })
            .collect()
    }

    /// BLAS `trmv`: triangular matrix–vector product `y = op(A) · x`.
    ///
    /// Single-RHS specialization of `trmm` with a direct `O(N²)`
    /// access pattern (no extra allocations beyond the output). `A`
    /// is `dim × dim`; `x.len() == dim`; the returned vector has
    /// length `dim`. `uplo` picks which triangle of `A` is read,
    /// `op` toggles transpose, and `diag == Unit` masks the stored
    /// diagonal to 1.
    ///
    /// The BLAS signature is in-place (`x := op(A)·x`); we return a
    /// new `Vec<u64>` to match the Rust-immutable-builder style used
    /// across the rest of `MatZpn`.
    pub fn trmv(&self, uplo: Uplo, op: TransA, diag: Diag, x: &[u64]) -> Vec<u64> {
        let a = self;
        assert_eq!(a.rows, a.cols, "trmv: A must be square");
        let dim = a.rows;
        assert_eq!(x.len(), dim, "trmv: x length must match A.dim");
        let modulus = a.data.p_pow_n;
        let md = modulus as u128;
        let eff_upper = (uplo == Uplo::Upper) != (op == TransA::Trans);

        let get_opa = |i: usize, k: usize| -> u64 {
            match op {
                TransA::NoTrans => a.get(i, k),
                TransA::Trans => a.get(k, i),
            }
        };

        let mut y = vec![0u64; dim];
        for i in 0..dim {
            // Inner k-range: the half of indices where op(A)[i, k] is
            // inside the effective triangle, including the diagonal.
            let (lo, hi) = if eff_upper { (i, dim) } else { (0, i + 1) };
            let mut acc: u128 = 0;
            for k in lo..hi {
                let a_val = if i == k && matches!(diag, Diag::Unit) {
                    1
                } else {
                    get_opa(i, k)
                };
                acc = (acc + (a_val as u128) * (x[k] as u128)) % md;
            }
            y[i] = acc as u64;
        }
        y
    }

    /// BLAS `trsv`: single-RHS triangular solve `op(A) · y = x`.
    ///
    /// Single-column specialization of `trsm`. Returns `None` when
    /// `diag == NonUnit` and some diagonal entry is in the maximal
    /// ideal of `Z/p^n` (i.e. shares a factor with `p`) — same
    /// contract as `trsm`. The BLAS signature is in-place (`x :=
    /// op(A)⁻¹·x`); we return a new `Vec<u64>` so the call stays
    /// side-effect free.
    ///
    /// Direct `O(N²)` forward / back substitution (no nrhs loop
    /// overhead). Pivot inverses are computed once up-front when
    /// `diag == NonUnit`.
    pub fn trsv(&self, uplo: Uplo, op: TransA, diag: Diag, x: &[u64]) -> Option<Vec<u64>> {
        let a = self;
        assert_eq!(a.rows, a.cols, "trsv: A must be square");
        let dim = a.rows;
        assert_eq!(x.len(), dim, "trsv: x length must match A.dim");
        let modulus = a.data.p_pow_n;
        let md = modulus as u128;

        let mut pivots_inv: Vec<u64> = vec![1; dim];
        if matches!(diag, Diag::NonUnit) {
            for i in 0..dim {
                pivots_inv[i] = zpn_scalar_inverse(a.get(i, i), modulus)?;
            }
        }

        let eff_upper = (uplo == Uplo::Upper) != (op == TransA::Trans);

        let get_opa = |i: usize, k: usize| -> u64 {
            match op {
                TransA::NoTrans => a.get(i, k),
                TransA::Trans => a.get(k, i),
            }
        };

        let mod_sub = |x: u128, y: u128| -> u128 {
            if x >= y {
                x - y
            } else {
                x + md - y
            }
        };

        let mut y = x.to_vec();

        if eff_upper {
            // Back substitution.
            for i in (0..dim).rev() {
                let mut acc = y[i] as u128;
                for k in (i + 1)..dim {
                    let term = (get_opa(i, k) as u128 * y[k] as u128) % md;
                    acc = mod_sub(acc, term);
                }
                y[i] = ((acc * pivots_inv[i] as u128) % md) as u64;
            }
        } else {
            // Forward substitution.
            for i in 0..dim {
                let mut acc = y[i] as u128;
                for k in 0..i {
                    let term = (get_opa(i, k) as u128 * y[k] as u128) % md;
                    acc = mod_sub(acc, term);
                }
                y[i] = ((acc * pivots_inv[i] as u128) % md) as u64;
            }
        }

        Some(y)
    }

    /// Out-of-place transpose: returns a new `rows × cols → cols × rows`
    /// matrix with `out.get(j, i) == self.get(i, j)`. No restriction on
    /// shape. O(m·n) — one pass over the source cells.
    pub fn transposed(&self) -> Self {
        let (m, n) = (self.rows, self.cols);
        let mut out = Self::zeros(self.p, self.n, n, m);
        for i in 0..m {
            for j in 0..n {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    /// In-place transpose: swap entries across the diagonal. Only valid
    /// for **square** matrices; panics otherwise (non-square in-place
    /// transpose would need cycle-leader bookkeeping or a full second
    /// buffer, which is the wrong shape to support on a bit-packed
    /// `PackedZpn` backing). Rectangular callers should go through
    /// `transposed` and assign back.
    ///
    /// O(N²/2) cell reads + writes; touches only cells strictly above
    /// the diagonal.
    pub fn transpose_inplace(&mut self) {
        assert_eq!(
            self.rows, self.cols,
            "transpose_inplace: matrix must be square"
        );
        let n = self.rows;
        for i in 0..n {
            for j in (i + 1)..n {
                let a = self.get(i, j);
                let b = self.get(j, i);
                self.set(i, j, b);
                self.set(j, i, a);
            }
        }
    }

    /// Kronecker (tensor) product `C = A ⊗ B`.
    ///
    /// `A` is `m × n`, `B` is `p × q`; `C` is `m·p × n·q` with
    /// `C[i·p + k, j·q + l] = A[i, j] · B[k, l]`.
    ///
    /// Not in BLAS proper, but a natural add for tensor-structured
    /// linear systems (e.g. stiffness matrices on product grids,
    /// periodic decompositions, multi-index ring arithmetic).
    ///
    /// Sparse-skip on `A[i, j] == 0`: if the `A` block is all zero
    /// for some `(i, j)`, skip the entire `p × q` sub-block fill,
    /// saving an O(p·q) mul loop per zeroed cell of `A`.
    pub fn kron(&self, other: &Self) -> Self {
        assert_eq!(self.p, other.p, "kron: p mismatch");
        assert_eq!(self.n, other.n, "kron: n mismatch");
        let modulus = self.data.p_pow_n;
        let md = modulus as u128;
        let (m, n) = (self.rows, self.cols);
        let (p_rows, q) = (other.rows, other.cols);

        let mut out = Self::zeros(self.p, self.n, m * p_rows, n * q);
        for i in 0..m {
            for j in 0..n {
                let a_ij = self.get(i, j);
                if a_ij == 0 {
                    continue;
                }
                let a128 = a_ij as u128;
                for k in 0..p_rows {
                    for l in 0..q {
                        let b_kl = other.get(k, l);
                        let prod = (a128 * b_kl as u128) % md;
                        out.set(i * p_rows + k, j * q + l, prod as u64);
                    }
                }
            }
        }
        out
    }

    /// BLAS Level-1 `axpy`: `y' = α · x + y` over `Z/p^n`.
    ///
    /// Associated (not instance) function because no matrix is
    /// involved — `axpy` is a pure vector op. Takes the ring
    /// parameters `(p, n)` explicitly so callers don't need a
    /// dummy `MatZpn` on hand. Returns a new `Vec<u64>`.
    pub fn axpy(p: u32, n: u32, alpha: u64, x: &[u64], y: &[u64]) -> Vec<u64> {
        let modulus = (p as u64).pow(n);
        assert_eq!(x.len(), y.len(), "axpy: x and y must match length");
        assert!(alpha < modulus, "axpy: alpha must be in [0, p^n)");
        let md = modulus as u128;
        let alpha128 = alpha as u128;
        x.iter()
            .zip(y.iter())
            .map(|(&xv, &yv)| {
                let term = (alpha128 * xv as u128) % md;
                ((term + yv as u128) % md) as u64
            })
            .collect()
    }

    /// BLAS Level-1 `dot`: `Σᵢ x[i]·y[i]` over `Z/p^n`. Inner product,
    /// no conjugation (the ring is commutative and has no natural
    /// involution here). Associated function; same rationale as `axpy`.
    pub fn dot(p: u32, n: u32, x: &[u64], y: &[u64]) -> u64 {
        let modulus = (p as u64).pow(n);
        assert_eq!(x.len(), y.len(), "dot: x and y must match length");
        let md = modulus as u128;
        let mut acc: u128 = 0;
        for (&xv, &yv) in x.iter().zip(y.iter()) {
            acc = (acc + xv as u128 * yv as u128) % md;
        }
        acc as u64
    }

    /// BLAS `ger`: rank-1 update `C = A + α · x · yᵀ`.
    ///
    /// `self` is the existing `A` (shape `m × n`); `x.len() == m`
    /// and `y.len() == n`. Returns the updated matrix as a new
    /// `MatZpn` (the BLAS in-place signature `A := A + α·x·yᵀ` is
    /// deferred — matches the Rust-immutable-builder convention the
    /// rest of `MatZpn` uses).
    ///
    /// The building block behind blocked LU's trailing-submatrix
    /// update: after a pivot step picks out a row and column, the
    /// remainder gets a rank-1 subtract.
    pub fn ger(&self, alpha: u64, x: &[u64], y: &[u64]) -> Self {
        let modulus = self.data.p_pow_n;
        assert!(alpha < modulus, "ger: alpha must be in [0, p^n)");
        assert_eq!(x.len(), self.rows, "ger: x.len() must equal A.rows");
        assert_eq!(y.len(), self.cols, "ger: y.len() must equal A.cols");

        let md = modulus as u128;
        let alpha128 = alpha as u128;
        let (m, n) = (self.rows, self.cols);

        let mut out = Self::zeros(self.p, self.n, m, n);
        for i in 0..m {
            let ax_i = (alpha128 * x[i] as u128) % md;
            for j in 0..n {
                let term = (ax_i * y[j] as u128) % md;
                let sum = self.get(i, j) as u128 + term;
                let v = if sum >= md { sum - md } else { sum };
                out.set(i, j, v as u64);
            }
        }
        out
    }

    /// Dense matmul over Z/p^n using the best available implementation.
    /// On x86-64 this runtime-dispatches to AVX2 for supported u16/u32
    /// moduli. On aarch64 it dispatches to NEON for supported u32 moduli.
    /// Every other case uses the scalar Barrett correctness floor.
    pub fn matmul(&self, other: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if self.data.p_pow_n <= 32_768 {
                if let Some(product) = self.matmul_avx2(other) {
                    return product;
                }
            } else if self.data.p_pow_n <= (1u64 << 31) {
                if let Some(product) = self.matmul_avx2_u32(other) {
                    return product;
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(product) = self.matmul_neon_u32(other) {
            return product;
        }
        self.matmul_scalar(other)
    }

    /// Dense scalar Barrett matmul over Z/p^n.
    ///
    /// This is the always-present correctness floor and the differential-test
    /// reference for architecture-specific kernels. Ordinary callers should
    /// use [`Self::matmul`] so runtime dispatch remains available.
    pub fn matmul_scalar(&self, other: &Self) -> Self {
        use rayon::prelude::*;

        assert_eq!(self.p, other.p, "p mismatch");
        assert_eq!(self.n, other.n, "n mismatch");
        assert_eq!(self.cols, other.rows, "matmul dim mismatch");
        let p_pow_n = self.data.p_pow_n;
        let reducer = BarrettReducer::new(p_pow_n);
        let (m, k, n) = (self.rows, self.cols, other.cols);

        // Build intermediate dense u64 output, then pack at the end.
        // Parallel per output row.
        let a_unpacked = self.data.to_vec();
        let b_unpacked = other.data.to_vec();

        let c_unpacked: Vec<u64> = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                let mut row = vec![0u64; n];
                for kk in 0..k {
                    let a_ik = a_unpacked[i * k + kk];
                    if a_ik == 0 {
                        continue;
                    }
                    for j in 0..n {
                        let b_kj = b_unpacked[kk * n + j];
                        let term = reducer.reduce(a_ik.wrapping_mul(b_kj));
                        let sum = row[j].wrapping_add(term);
                        row[j] = if sum >= p_pow_n { sum - p_pow_n } else { sum };
                    }
                }
                row
            })
            .collect();

        let mut c = Self::zeros(self.p, self.n, m, n);
        for i in 0..m {
            for j in 0..n {
                c.set(i, j, c_unpacked[i * n + j]);
            }
        }
        c
    }

    /// BLAS-style triangular matrix–matrix product.
    ///
    /// - `Side::Left`:  `C = alpha · op(A) · B`  (A is m×m, B is m×n)
    /// - `Side::Right`: `C = alpha · B · op(A)`  (A is n×n, B is m×n)
    ///
    /// `uplo` selects which triangle of the stored `A` is read;
    /// entries outside that triangle are treated as zero. `op`
    /// selects whether `A` or its transpose is used. The diagonal
    /// of `A` is read as stored (no implicit-unit-diag option — the
    /// BLAS `diag` parameter is intentionally omitted; callers that
    /// need unit diagonal should zero the diagonal before calling).
    ///
    /// Output is returned as a new matrix rather than overwriting
    /// `B` — matching the Rust-style immutable-builder convention
    /// the rest of `MatZpn` uses.
    ///
    /// Scalar Barrett in this iteration; the triangular sweep skips
    /// the half of the inner range that would hit zeros, so cost is
    /// `~m · n · dim/2` instead of the `m · n · dim` a materialize-
    /// then-gemm reference would pay. Sharing the AVX2 4×8 micro-
    /// kernel lands in a later iteration alongside the rest of the
    /// Level-3 vectorization work.
    pub fn trmm(&self, side: Side, uplo: Uplo, op: TransA, alpha: u64, b: &Self) -> Self {
        let a = self;
        assert_eq!(a.p, b.p, "p mismatch");
        assert_eq!(a.n, b.n, "n mismatch");
        assert_eq!(a.rows, a.cols, "trmm: A must be square");

        let modulus = a.data.p_pow_n;
        assert!(alpha < modulus, "trmm: alpha must be in [0, p^n)");
        let dim = a.rows;

        let (m, n_out) = match side {
            Side::Left => {
                assert_eq!(dim, b.rows, "trmm Left: A.cols != B.rows");
                (b.rows, b.cols)
            }
            Side::Right => {
                assert_eq!(dim, b.cols, "trmm Right: B.cols != A.rows");
                (b.rows, b.cols)
            }
        };

        let mut c = Self::zeros(a.p, a.n, m, n_out);
        let md = modulus as u128;
        let alpha128 = alpha as u128;

        match side {
            Side::Left => {
                // C[i,j] = alpha · Σ_k op(A)[i,k] · B[k,j]
                for i in 0..m {
                    // Inner range is the half of k-indices where
                    // op(A)[i, k] is inside the specified triangle.
                    let (lo, hi): (usize, usize) = match (op, uplo) {
                        (TransA::NoTrans, Uplo::Upper) => (i, dim),
                        (TransA::NoTrans, Uplo::Lower) => (0, i + 1),
                        (TransA::Trans, Uplo::Upper) => (0, i + 1),
                        (TransA::Trans, Uplo::Lower) => (i, dim),
                    };
                    for j in 0..n_out {
                        let mut acc: u128 = 0;
                        for k in lo..hi {
                            let a_val = match op {
                                TransA::NoTrans => a.get(i, k),
                                TransA::Trans => a.get(k, i),
                            };
                            let b_val = b.get(k, j);
                            acc = (acc + (a_val as u128) * (b_val as u128)) % md;
                        }
                        let scaled = (acc * alpha128) % md;
                        c.set(i, j, scaled as u64);
                    }
                }
            }
            Side::Right => {
                // C[i,j] = alpha · Σ_k B[i,k] · op(A)[k,j]
                for j in 0..n_out {
                    let (lo, hi): (usize, usize) = match (op, uplo) {
                        (TransA::NoTrans, Uplo::Upper) => (0, j + 1),
                        (TransA::NoTrans, Uplo::Lower) => (j, dim),
                        (TransA::Trans, Uplo::Upper) => (j, dim),
                        (TransA::Trans, Uplo::Lower) => (0, j + 1),
                    };
                    for i in 0..m {
                        let mut acc: u128 = 0;
                        for k in lo..hi {
                            let b_val = b.get(i, k);
                            let a_val = match op {
                                TransA::NoTrans => a.get(k, j),
                                TransA::Trans => a.get(j, k),
                            };
                            acc = (acc + (b_val as u128) * (a_val as u128)) % md;
                        }
                        let scaled = (acc * alpha128) % md;
                        c.set(i, j, scaled as u64);
                    }
                }
            }
        }
        c
    }

    /// BLAS-style triangular solve with multiple right-hand sides.
    ///
    /// - `Side::Left`:  solves `op(A) · X = alpha · B`  (A is m×m)
    /// - `Side::Right`: solves `X · op(A) = alpha · B`  (A is n×n)
    ///
    /// `uplo` picks which triangle of `A` is used. `diag == Unit`
    /// treats the diagonal as implicit ones (stored diagonal
    /// ignored); `diag == NonUnit` reads the stored diagonal and
    /// requires every pivot to be a unit in `Z/p^n` (coprime to the
    /// prime `p` — `Z/p^n` is a local ring, so non-coprime entries
    /// are the maximal ideal and non-invertible).
    ///
    /// Returns `None` when `diag == NonUnit` and some diagonal entry
    /// lies in the maximal ideal, so callers can fall back without
    /// a panic.
    ///
    /// Output `X` is returned as a new matrix (Rust-immutable-builder
    /// style, matching `trmm`). Scalar Barrett inside the inner
    /// update loop; pivot inversion via `ext_gcd` once per diagonal.
    pub fn trsm(
        &self,
        side: Side,
        uplo: Uplo,
        op: TransA,
        diag: Diag,
        alpha: u64,
        b: &Self,
    ) -> Option<Self> {
        let a = self;
        assert_eq!(a.p, b.p, "p mismatch");
        assert_eq!(a.n, b.n, "n mismatch");
        assert_eq!(a.rows, a.cols, "trsm: A must be square");

        let modulus = a.data.p_pow_n;
        assert!(alpha < modulus, "trsm: alpha must be in [0, p^n)");
        let dim = a.rows;

        let (m, n_out) = match side {
            Side::Left => {
                assert_eq!(dim, b.rows, "trsm Left: A.cols != B.rows");
                (b.rows, b.cols)
            }
            Side::Right => {
                assert_eq!(dim, b.cols, "trsm Right: B.cols != A.rows");
                (b.rows, b.cols)
            }
        };

        // Precompute pivot inverses. For Unit diag we keep the
        // default 1 — division by 1 is a no-op.
        let mut pivots_inv: Vec<u64> = vec![1; dim];
        if matches!(diag, Diag::NonUnit) {
            for i in 0..dim {
                pivots_inv[i] = zpn_scalar_inverse(a.get(i, i), modulus)?;
            }
        }

        // The effective triangular shape of op(A).
        let eff_upper = (uplo == Uplo::Upper) != (op == TransA::Trans);
        let md = modulus as u128;
        let alpha128 = alpha as u128;

        // Seed X with `alpha · B`, then overwrite in substitution order.
        let mut x = Self::zeros(a.p, a.n, m, n_out);
        for i in 0..m {
            for j in 0..n_out {
                let v = (alpha128 * b.get(i, j) as u128) % md;
                x.set(i, j, v as u64);
            }
        }

        let get_opa = |i: usize, k: usize| -> u64 {
            match op {
                TransA::NoTrans => a.get(i, k),
                TransA::Trans => a.get(k, i),
            }
        };

        // Modular subtract, both args in [0, md).
        let mod_sub = |x: u128, y: u128| -> u128 {
            if x >= y {
                x - y
            } else {
                x + md - y
            }
        };

        match side {
            Side::Left => {
                // Solve op(A) · X = αB, one B-column at a time.
                for j in 0..n_out {
                    if eff_upper {
                        // Back substitution in i.
                        for i in (0..m).rev() {
                            let mut acc = x.get(i, j) as u128;
                            for k in (i + 1)..m {
                                let term = (get_opa(i, k) as u128 * x.get(k, j) as u128) % md;
                                acc = mod_sub(acc, term);
                            }
                            let val = (acc * pivots_inv[i] as u128) % md;
                            x.set(i, j, val as u64);
                        }
                    } else {
                        // Forward substitution in i.
                        for i in 0..m {
                            let mut acc = x.get(i, j) as u128;
                            for k in 0..i {
                                let term = (get_opa(i, k) as u128 * x.get(k, j) as u128) % md;
                                acc = mod_sub(acc, term);
                            }
                            let val = (acc * pivots_inv[i] as u128) % md;
                            x.set(i, j, val as u64);
                        }
                    }
                }
            }
            Side::Right => {
                // Solve X · op(A) = αB, one B-row at a time.
                for i in 0..m {
                    if eff_upper {
                        // Forward substitution in j.
                        for j in 0..n_out {
                            let mut acc = x.get(i, j) as u128;
                            for k in 0..j {
                                let term = (x.get(i, k) as u128 * get_opa(k, j) as u128) % md;
                                acc = mod_sub(acc, term);
                            }
                            let val = (acc * pivots_inv[j] as u128) % md;
                            x.set(i, j, val as u64);
                        }
                    } else {
                        // Back substitution in j.
                        for j in (0..n_out).rev() {
                            let mut acc = x.get(i, j) as u128;
                            for k in (j + 1)..n_out {
                                let term = (x.get(i, k) as u128 * get_opa(k, j) as u128) % md;
                                acc = mod_sub(acc, term);
                            }
                            let val = (acc * pivots_inv[j] as u128) % md;
                            x.set(i, j, val as u64);
                        }
                    }
                }
            }
        }

        Some(x)
    }
}

/// Which side of `B` the triangular `A` sits on in a `trmm`/`trsm`
/// call. Matches BLAS `SIDE` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// `alpha · op(A) · B` — A on the left.
    Left,
    /// `alpha · B · op(A)` — A on the right.
    Right,
}

/// Which triangle of `A` is read. Matches BLAS `UPLO` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uplo {
    /// Read the upper triangle of `A` (entries with row ≤ col).
    Upper,
    /// Read the lower triangle of `A` (entries with row ≥ col).
    Lower,
}

/// Whether `A` or its transpose is used. Matches BLAS `TRANSA`
/// parameter (no conjugate-transpose variant — our rings are not
/// complex).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransA {
    /// Use `A` as stored.
    NoTrans,
    /// Use `Aᵀ`.
    Trans,
}

/// Whether the diagonal of `A` is treated as the stored values or as
/// implicit ones. Matches BLAS `DIAG` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diag {
    /// Diagonal is implicit 1 — stored values are ignored.
    Unit,
    /// Diagonal is read from storage; each pivot must be a unit in
    /// `Z/p^n` (i.e. coprime to `p`) for `trsm` to succeed.
    NonUnit,
}

/// CSR-format sparse matrix over Z/p^n. Parallel analogue of
/// `SparseCsrMat<T: Z2n>` in `src/mat.rs`, with Barrett reduction on
/// the inner products. Used for block Wiedemann and the sparse-F_p →
/// Z/p^n lift step.
#[derive(Clone, Debug)]
pub struct SparseCsrMatZpn {
    /// Prime base of the ring.
    pub p: u32,
    /// Exponent: the ring is `Z/p^n`.
    pub n: u32,
    /// Precomputed `p^n` for fast modular reduction.
    pub p_pow_n: u64,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Starting index in `col_ind`/`values` for each row (length `rows + 1`).
    pub row_ptr: Vec<usize>,
    /// Column index of each stored nonzero, in row-major order.
    pub col_ind: Vec<usize>,
    /// Value of each stored nonzero (already reduced mod `p^n`).
    pub values: Vec<u64>,
}

impl SparseCsrMatZpn {
    /// Build a CSR from a dense MatZpn.
    pub fn from_dense(m: &MatZpn) -> Self {
        let mut row_ptr = Vec::with_capacity(m.rows + 1);
        row_ptr.push(0);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();
        for i in 0..m.rows {
            for j in 0..m.cols {
                let v = m.get(i, j);
                if v != 0 {
                    col_ind.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(col_ind.len());
        }
        Self {
            p: m.p,
            n: m.n,
            p_pow_n: m.data.p_pow_n,
            rows: m.rows,
            cols: m.cols,
            row_ptr,
            col_ind,
            values,
        }
    }

    /// Number of nonzero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse A · dense B → dense C over Z/p^n. Walks A row by row,
    /// accumulating each nonzero's contribution to the output row
    /// with Barrett reduction. Parallel across output rows.
    pub fn matmul_dense(&self, b: &MatZpn) -> MatZpn {
        use rayon::prelude::*;
        assert_eq!(self.p, b.p, "sparse matmul: p mismatch");
        assert_eq!(self.n, b.n, "sparse matmul: n mismatch");
        assert_eq!(
            self.cols, b.rows,
            "sparse matmul: A.cols ({}) != B.rows ({})",
            self.cols, b.rows
        );
        let (m, n) = (self.rows, b.cols);
        let p_pow_n = self.p_pow_n;
        let reducer = BarrettReducer::new(p_pow_n);
        let b_unpacked = b.data.to_vec();

        // Compute output rows in parallel.
        let out_flat: Vec<u64> = (0..m)
            .into_par_iter()
            .flat_map(|i| {
                let mut row = vec![0u64; n];
                let row_start = self.row_ptr[i];
                let row_end = self.row_ptr[i + 1];
                for p in row_start..row_end {
                    let col = self.col_ind[p];
                    let val = self.values[p];
                    let b_row = &b_unpacked[col * n..(col + 1) * n];
                    for j in 0..n {
                        let term = reducer.reduce(val.wrapping_mul(b_row[j]));
                        let sum = row[j] + term;
                        row[j] = if sum >= p_pow_n { sum - p_pow_n } else { sum };
                    }
                }
                row
            })
            .collect();

        let mut c = MatZpn::zeros(self.p, self.n, m, n);
        for i in 0..m {
            for j in 0..n {
                c.set(i, j, out_flat[i * n + j]);
            }
        }
        c
    }
}

impl PartialEq for MatZpn {
    fn eq(&self, other: &Self) -> bool {
        self.p == other.p
            && self.n == other.n
            && self.rows == other.rows
            && self.cols == other.cols
            && self.data == other.data
    }
}

// ─── Hensel-lifted inverse over Z/p^n ──────────────────────────────

impl MatZpn {
    /// Matrix inverse over Z/p^n via Hensel lifting.
    ///
    /// 1. Reduce `A` mod `p` to get `Ā` in `F_p`.
    /// 2. Compute `X_0 = Ā⁻¹` over `F_p` by Gauss-Jordan (returns `None` if singular).
    /// 3. Newton iteration `X_{k+1} = X_k · (2·I − A·X_k)` doubles the
    ///    p-adic precision per step; `⌈log_2(n)⌉` iterations suffice
    ///    to reach mod-p^n precision.
    ///
    /// `A` is invertible over Z/p^n iff its reduction mod p is invertible
    /// over F_p (iff `det(A) ≢ 0 (mod p)`), which this method detects.
    pub fn inverse(&self) -> Option<Self> {
        let n = self.rows;
        assert_eq!(n, self.cols, "inverse: matrix must be square");

        let p = self.p as u64;
        let p_pow_n = self.data.p_pow_n;

        // Step 1 + 2: F_p inverse (reduce-and-eliminate internally).
        let mut x = fp_matrix_inverse(self, p)?;

        // Step 3: Newton iteration. Precision after k steps: p^{2^k}.
        // Need 2^k ≥ self.n → k = ⌈log_2(self.n)⌉.
        let iterations = if self.n <= 1 {
            0
        } else {
            (self.n - 1).next_power_of_two().trailing_zeros() + 1
        };

        for _ in 0..iterations {
            // 2·I − A·X
            let ax = self.matmul(&x);
            let mut two_i_minus_ax = ax.neg();
            for i in 0..n {
                let v = two_i_minus_ax.get(i, i);
                let sum = v.wrapping_add(2);
                let sum = if sum >= p_pow_n { sum - p_pow_n } else { sum };
                two_i_minus_ax.set(i, i, sum);
            }
            x = x.matmul(&two_i_minus_ax);
        }

        Some(x)
    }
}

/// F_p matrix inverse via Gauss-Jordan elimination on the augmented
/// matrix `[A | I]`. Returns `None` if `Ā` is singular (a pivot column
/// has no non-zero entry).
fn fp_matrix_inverse(a: &MatZpn, p: u64) -> Option<MatZpn> {
    let n = a.rows;
    // Augmented [A mod p | I], stored as n rows of length 2n.
    let mut aug: Vec<Vec<u64>> = (0..n)
        .map(|i| {
            let mut row = vec![0u64; 2 * n];
            for j in 0..n {
                row[j] = a.get(i, j) % p;
            }
            row[n + i] = 1;
            row
        })
        .collect();

    for k in 0..n {
        // Find a non-zero pivot in column k at or below row k.
        let pivot_row = (k..n).find(|&r| aug[r][k] != 0)?;
        if pivot_row != k {
            aug.swap(k, pivot_row);
        }
        // Normalize pivot row so aug[k][k] = 1.
        let pivot = aug[k][k];
        let pivot_inv = fp_scalar_inverse(pivot, p)?;
        for j in 0..(2 * n) {
            aug[k][j] = (aug[k][j] * pivot_inv) % p;
        }
        // Eliminate column k in every other row.
        for i in 0..n {
            if i == k || aug[i][k] == 0 {
                continue;
            }
            let factor = aug[i][k];
            for j in 0..(2 * n) {
                let sub = (factor * aug[k][j]) % p;
                aug[i][j] = if aug[i][j] >= sub {
                    aug[i][j] - sub
                } else {
                    aug[i][j] + p - sub
                };
            }
        }
    }

    let mut result = MatZpn::zeros(a.p, a.n, n, n);
    for i in 0..n {
        for j in 0..n {
            result.set(i, j, aug[i][n + j]);
        }
    }
    Some(result)
}

/// Modular inverse of `a` mod prime `p`, via extended Euclidean.
fn fp_scalar_inverse(a: u64, p: u64) -> Option<u64> {
    if a == 0 {
        return None;
    }
    let (g, x, _) = ext_gcd(a as i64, p as i64);
    if g != 1 {
        return None;
    }
    Some(((x % p as i64 + p as i64) as u64) % p)
}

/// Modular inverse of `a` in `Z/modulus` where `modulus = p^n`.
/// Returns `None` when `a` shares a factor with the prime `p` (and
/// is therefore not a unit in the local ring). Used by `trsm` to
/// invert triangular pivots. Caller must ensure `modulus` fits in
/// the u32 packed range so `ext_gcd` stays inside i64.
fn zpn_scalar_inverse(a: u64, modulus: u64) -> Option<u64> {
    if a == 0 {
        return None;
    }
    let (g, x, _) = ext_gcd(a as i64, modulus as i64);
    if g != 1 {
        return None;
    }
    Some(((x % modulus as i64 + modulus as i64) as u64) % modulus)
}

fn ext_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x1, y1) = ext_gcd(b, a % b);
        (g, y1, x1 - (a / b) * y1)
    }
}

// ─── Barrett reduction ──────────────────────────────────────────────

/// Barrett reducer for a fixed modulus `m = p^n < 2^32`.
///
/// Precomputed constant `μ = ⌊2^64 / m⌋`. Given `a` with
/// `a < m² ≤ 2^64`, the quotient
/// `q = ⌊a·μ / 2^64⌋` is within 1 of the true `⌊a/m⌋`, so one
/// comparison + conditional subtract yields `a mod m`.
///
/// Faster than `%` when many reductions are done with the same `m` —
/// multiplies and shifts are cheaper than a full 64/32-bit divide on
/// most architectures. Used for packed Z/p^n multiplies and scalar
/// multiplications.
#[derive(Debug, Clone, Copy)]
pub struct BarrettReducer {
    /// The modulus (`reduce(a)` returns `a mod m`).
    pub m: u64,
    mu: u64,
}

impl BarrettReducer {
    /// Build a Barrett reducer for modulus `m`. Requires `m ≥ 2` and
    /// `m < 2^32` (so inputs `a < m²` fit in u64).
    pub fn new(m: u64) -> Self {
        assert!(m >= 2, "Barrett: modulus must be at least 2");
        assert!(m < (1u64 << 32), "Barrett: modulus must be < 2^32");
        let mu = ((1u128 << 64) / (m as u128)) as u64;
        Self { m, mu }
    }

    /// Reduce `a` mod `m`. Requires `a < m² ≤ 2^64`, which is
    /// `a ≤ 2^64 − 1`.
    #[inline]
    pub fn reduce(&self, a: u64) -> u64 {
        // q ≈ ⌊a · μ / 2^64⌋  (the upper half of the u128 product).
        let q = ((a as u128).wrapping_mul(self.mu as u128) >> 64) as u64;
        let r = a.wrapping_sub(q.wrapping_mul(self.m));
        // r ∈ [0, 2m), so one conditional subtract suffices.
        if r >= self.m {
            r - self.m
        } else {
            r
        }
    }
}

impl PartialEq for PackedZpn {
    fn eq(&self, other: &Self) -> bool {
        if self.p != other.p || self.n != other.n || self.length != other.length {
            return false;
        }
        // Compare reduced values (raw data may differ due to unreduced guard state).
        for i in 0..self.length {
            if self.get(i) != other.get(i) {
                return false;
            }
        }
        true
    }
}

// ─── PackedRing impl for MatZpn ────────────────────────────────────
//
// Lets the `fp_pluq_packed` / `fp_rank_packed` generics in
// `src/pluq_gen.rs` be instantiated on MatZpn. See the `PackedRing`
// trait docs there for the broader unification plan.

/// `PackedRing` instance for `Z/p^n` with guard-bit packed storage.
/// Carries `(p, p_pow_n)` as runtime state.
pub struct ZpnRing {
    /// Prime base of the ring.
    pub p: u32,
    /// Modulus `p^n` (cached so callers can hand it to Barrett directly).
    pub p_pow_n: u64,
}

impl ZpnRing {
    /// Build a `ZpnRing` from a MatZpn's `(p, p_pow_n)` pair. Typical
    /// usage: `let ring = ZpnRing::from_mat(&m); pluq_gen::
    /// fp_pluq_packed(&ring, &m)`.
    pub fn from_mat(m: &MatZpn) -> Self {
        Self {
            p: m.p,
            p_pow_n: m.data.p_pow_n,
        }
    }
}

// ─── NEON microkernel for MatZpn u16 ──────────────────────────────────
//
// A 4×8 register tile parallelling `src/neon.rs`: C is held as four
// `uint16x8_t` accumulators (one per row of the 4-row tile), K is
// iterated with per-product Barrett reduction on widened u32 lanes,
// and wrapping u16 add + conditional subtract keeps each accumulator
// in `[0, p^n)` between K steps.
//
// Modulus constraint: `p^n ≤ 2^15` so `acc + new_prod < 2·p^n` fits in
// u16 without wrap. For `p^n ∈ (2^15, 2^16)` the dispatcher falls back
// to the scalar `MatZpn::matmul` path. Widening every accumulator to
// uint32x4_t (cutting the output tile width in half) is the natural
// fix and is listed as deferred future work.

#[cfg(target_arch = "aarch64")]
mod neon_kernel {
    use core::arch::aarch64::*;

    /// Barrett-reduce each lane of `x` (all lanes `< m_v32²`) into
    /// `[0, modulus)`. `mu_v` is `μ = ⌊2^32 / modulus⌋` broadcast across
    /// all lanes; `m_v32` is `modulus` broadcast across all lanes.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn barrett_u32x4(x: uint32x4_t, mu_v: uint32x4_t, m_v32: uint32x4_t) -> uint32x4_t {
        // `q = ⌊x · μ / 2^32⌋` — high 32 bits of each u32 × u32 product.
        let x_lo = vget_low_u32(x);
        let x_hi = vget_high_u32(x);
        let mu_lo = vget_low_u32(mu_v);
        let mu_hi = vget_high_u32(mu_v);
        let prod_lo = vmull_u32(x_lo, mu_lo);
        let prod_hi = vmull_u32(x_hi, mu_hi);
        let q_lo = vshrn_n_u64::<32>(prod_lo);
        let q_hi = vshrn_n_u64::<32>(prod_hi);
        let q = vcombine_u32(q_lo, q_hi);
        let r = vsubq_u32(x, vmulq_u32(q, m_v32));
        // r ∈ [0, 2·modulus); conditional subtract.
        let over = vcgeq_u32(r, m_v32);
        vsubq_u32(r, vandq_u32(over, m_v32))
    }

    /// Compute C[i0..i0+4, j0..j0+8] += Σ_k A[i0..i0+4, k] · B[k, j0..j0+8]
    /// for a single 4×8 tile, mod `modulus`.
    #[target_feature(enable = "neon")]
    unsafe fn kernel_4x8(
        a: *const u16,
        b: *const u16,
        c: *mut u16,
        i0: usize,
        j0: usize,
        k: usize,
        n: usize,
        mu: u32,
        modulus: u16,
    ) {
        let mu_v = vdupq_n_u32(mu);
        let m_v32 = vdupq_n_u32(modulus as u32);
        let m_v16 = vdupq_n_u16(modulus);

        let mut acc0 = vdupq_n_u16(0);
        let mut acc1 = vdupq_n_u16(0);
        let mut acc2 = vdupq_n_u16(0);
        let mut acc3 = vdupq_n_u16(0);

        for p in 0..k {
            let b_vec = vld1q_u16(b.add(p * n + j0));
            // 4 a-values for this K step, one per output row.
            let a_v0 = vdupq_n_u16(*a.add(i0 * k + p));
            let a_v1 = vdupq_n_u16(*a.add((i0 + 1) * k + p));
            let a_v2 = vdupq_n_u16(*a.add((i0 + 2) * k + p));
            let a_v3 = vdupq_n_u16(*a.add((i0 + 3) * k + p));

            // Expand into u32 products, Barrett-reduce to [0, modulus),
            // narrow back to u16, accumulate with conditional subtract.
            macro_rules! fma_row {
                ($acc:ident, $a_v:ident) => {{
                    let p_lo = vmull_u16(vget_low_u16($a_v), vget_low_u16(b_vec));
                    let p_hi = vmull_high_u16($a_v, b_vec);
                    let r_lo = barrett_u32x4(p_lo, mu_v, m_v32);
                    let r_hi = barrett_u32x4(p_hi, mu_v, m_v32);
                    let prod = vcombine_u16(vmovn_u32(r_lo), vmovn_u32(r_hi));
                    let sum = vaddq_u16($acc, prod);
                    let mask = vcgeq_u16(sum, m_v16);
                    $acc = vsubq_u16(sum, vandq_u16(mask, m_v16));
                }};
            }
            fma_row!(acc0, a_v0);
            fma_row!(acc1, a_v1);
            fma_row!(acc2, a_v2);
            fma_row!(acc3, a_v3);
        }

        vst1q_u16(c.add(i0 * n + j0), acc0);
        vst1q_u16(c.add((i0 + 1) * n + j0), acc1);
        vst1q_u16(c.add((i0 + 2) * n + j0), acc2);
        vst1q_u16(c.add((i0 + 3) * n + j0), acc3);
    }

    /// `C = A · B` in Z/modulus for u16 inputs and `modulus ≤ 2^15`.
    /// Expects `a` to be `m × k` and `b` to be `k × n`, both row-major,
    /// all entries already in `[0, modulus)`. Returns `m × n` row-major.
    ///
    /// Main loop runs the 4×8 register tile; tail rows (`m mod 4`) and
    /// tail cols (`n mod 8`) fall back to a scalar Barrett loop.
    pub fn matmul_u16_mod(
        a: &[u16],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
        modulus: u16,
    ) -> Vec<u16> {
        debug_assert!(modulus >= 2);
        debug_assert!(
            modulus <= 32768,
            "neon matmul: modulus > 2^15 not yet supported"
        );
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);

        let mu: u32 = ((1u64 << 32) / modulus as u64) as u32;
        let mut c = vec![0u16; m * n];

        const TILE_M: usize = 4;
        const TILE_N: usize = 8;
        let m_tiled = m - (m % TILE_M);
        let n_tiled = n - (n % TILE_N);

        // Main 4×8 tiles.
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let c_ptr = c.as_mut_ptr();
            let mut i0 = 0;
            while i0 < m_tiled {
                let mut j0 = 0;
                while j0 < n_tiled {
                    kernel_4x8(a_ptr, b_ptr, c_ptr, i0, j0, k, n, mu, modulus);
                    j0 += TILE_N;
                }
                i0 += TILE_M;
            }
        }

        // Tail rows (full n).
        for i in m_tiled..m {
            for j in 0..n {
                let mut acc: u32 = 0;
                let md = modulus as u32;
                for p in 0..k {
                    let prod = (a[i * k + p] as u32) * (b[p * n + j] as u32);
                    acc = (acc + prod % md) % md;
                }
                c[i * n + j] = acc as u16;
            }
        }
        // Tail cols in the main-row region (j_tiled..n for rows 0..m_tiled).
        for i in 0..m_tiled {
            for j in n_tiled..n {
                let mut acc: u32 = 0;
                let md = modulus as u32;
                for p in 0..k {
                    let prod = (a[i * k + p] as u32) * (b[p * n + j] as u32);
                    acc = (acc + prod % md) % md;
                }
                c[i * n + j] = acc as u16;
            }
        }

        c
    }
}

// ─── NEON microkernel for MatZpn u32 ──────────────────────────────────
//
// Same BLIS skeleton as the u16 kernel, widened: products are computed
// as u64 (`vmull_u32` / `vmull_high_u32`) and the Barrett constant is
// `μ = ⌊2^64 / modulus⌋`. Because NEON has no `uint64 × uint64 → uint128`
// intrinsic, the Barrett high-word computation falls through to scalar
// per-lane u128 arithmetic — honest slice: the SIMD win is from the
// widening multiply + broadcast + accumulator vector add, not from the
// reduction step. A proper SIMD Barrett via long-multiplication of the
// two u32 halves of each u64 product is listed as deferred future work.
//
// Modulus range: `[2, 2^31]`. The output-accumulator headroom check is
// `acc + product < 2^32`, satisfied when each operand is in
// `[0, 2^31)` (sum < 2^32).

#[cfg(target_arch = "aarch64")]
mod neon_kernel_u32 {
    use core::arch::aarch64::*;

    /// Scalar Barrett: reduce a u64 `x < modulus²` to `[0, modulus)`
    /// using `μ = ⌊2^64 / modulus⌋`. Used per lane after the SIMD
    /// `vmull` since NEON lacks a 64 × 64 → 128 multiply.
    #[inline(always)]
    fn barrett_u64_to_u32(x: u64, mu: u64, modulus: u32) -> u32 {
        let q = ((x as u128 * mu as u128) >> 64) as u64;
        let r = x.wrapping_sub(q.wrapping_mul(modulus as u64)) as u32;
        if r >= modulus {
            r - modulus
        } else {
            r
        }
    }

    /// Reduce a `uint64x2_t` of two u64 products to two u32 lanes in
    /// `[0, modulus)` via per-lane scalar Barrett extract/insert.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn barrett_uint64x2_to_uint32x2(x: uint64x2_t, mu: u64, modulus: u32) -> uint32x2_t {
        let x0 = vgetq_lane_u64::<0>(x);
        let x1 = vgetq_lane_u64::<1>(x);
        let r0 = barrett_u64_to_u32(x0, mu, modulus);
        let r1 = barrett_u64_to_u32(x1, mu, modulus);
        let out = vdup_n_u32(0);
        let out = vset_lane_u32::<0>(r0, out);
        vset_lane_u32::<1>(r1, out)
    }

    /// 4×4 register-tile microkernel: four `uint32x4_t` accumulators,
    /// one per output row, each holding 4 u32 lanes. Per-K step loads
    /// one `uint32x4_t` b-vector and broadcasts four a-scalars,
    /// computes 4 u64 products per row via `vmull_u32` + `vmull_high_u32`,
    /// scalar-Barrett-reduces to u32, accumulates with conditional
    /// subtract.
    #[target_feature(enable = "neon")]
    unsafe fn kernel_4x4(
        a: *const u32,
        b: *const u32,
        c: *mut u32,
        i0: usize,
        j0: usize,
        k: usize,
        n: usize,
        mu: u64,
        modulus: u32,
    ) {
        let m_v = vdupq_n_u32(modulus);

        let mut acc0 = vdupq_n_u32(0);
        let mut acc1 = vdupq_n_u32(0);
        let mut acc2 = vdupq_n_u32(0);
        let mut acc3 = vdupq_n_u32(0);

        for p in 0..k {
            let b_vec = vld1q_u32(b.add(p * n + j0));
            let a_v0 = vdupq_n_u32(*a.add(i0 * k + p));
            let a_v1 = vdupq_n_u32(*a.add((i0 + 1) * k + p));
            let a_v2 = vdupq_n_u32(*a.add((i0 + 2) * k + p));
            let a_v3 = vdupq_n_u32(*a.add((i0 + 3) * k + p));

            macro_rules! fma_row {
                ($acc:ident, $a_v:ident) => {{
                    let p_lo = vmull_u32(vget_low_u32($a_v), vget_low_u32(b_vec));
                    let p_hi = vmull_high_u32($a_v, b_vec);
                    let r_lo = barrett_uint64x2_to_uint32x2(p_lo, mu, modulus);
                    let r_hi = barrett_uint64x2_to_uint32x2(p_hi, mu, modulus);
                    let prod = vcombine_u32(r_lo, r_hi);
                    let sum = vaddq_u32($acc, prod);
                    let mask = vcgeq_u32(sum, m_v);
                    $acc = vsubq_u32(sum, vandq_u32(mask, m_v));
                }};
            }
            fma_row!(acc0, a_v0);
            fma_row!(acc1, a_v1);
            fma_row!(acc2, a_v2);
            fma_row!(acc3, a_v3);
        }

        vst1q_u32(c.add(i0 * n + j0), acc0);
        vst1q_u32(c.add((i0 + 1) * n + j0), acc1);
        vst1q_u32(c.add((i0 + 2) * n + j0), acc2);
        vst1q_u32(c.add((i0 + 3) * n + j0), acc3);
    }

    /// `C = A · B` in Z/modulus for u32 inputs and `modulus ≤ 2^31`.
    /// Expects all entries in `[0, modulus)`. Returns `m × n` row-major.
    pub fn matmul_u32_mod(
        a: &[u32],
        b: &[u32],
        m: usize,
        k: usize,
        n: usize,
        modulus: u32,
    ) -> Vec<u32> {
        debug_assert!(modulus >= 2);
        debug_assert!(
            modulus <= (1u32 << 31),
            "neon u32 matmul: modulus > 2^31 not yet supported"
        );
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);

        let mu: u64 = ((1u128 << 64) / modulus as u128) as u64;
        let mut c = vec![0u32; m * n];

        const TILE_M: usize = 4;
        const TILE_N: usize = 4;
        let m_tiled = m - (m % TILE_M);
        let n_tiled = n - (n % TILE_N);

        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let c_ptr = c.as_mut_ptr();
            let mut i0 = 0;
            while i0 < m_tiled {
                let mut j0 = 0;
                while j0 < n_tiled {
                    kernel_4x4(a_ptr, b_ptr, c_ptr, i0, j0, k, n, mu, modulus);
                    j0 += TILE_N;
                }
                i0 += TILE_M;
            }
        }

        // Tail rows.
        for i in m_tiled..m {
            for j in 0..n {
                let mut acc: u64 = 0;
                let md = modulus as u64;
                for p in 0..k {
                    let prod = (a[i * k + p] as u64) * (b[p * n + j] as u64);
                    acc = (acc + prod % md) % md;
                }
                c[i * n + j] = acc as u32;
            }
        }
        // Tail cols in main-row region.
        for i in 0..m_tiled {
            for j in n_tiled..n {
                let mut acc: u64 = 0;
                let md = modulus as u64;
                for p in 0..k {
                    let prod = (a[i * k + p] as u64) * (b[p * n + j] as u64);
                    acc = (acc + prod % md) % md;
                }
                c[i * n + j] = acc as u32;
            }
        }

        c
    }
}

// ─── AVX2 microkernel for MatZpn u16 / u32 ───────────────────────────
//
// Compiled for every x86-64 target and entered only after runtime AVX2
// detection. Output rows are processed one at a time; within each row,
// output cells are computed 8 at a time via
// AVX2 (YMM = 8 u32 lanes). Per-K step: broadcast a, load 8 b values,
// widen to u32, multiply, Barrett-reduce each product, then update the
// accumulator via SIMD add + unsigned min-trick `acc = min(sum, sum −
// modulus)` for the conditional subtract.
//
// Barrett itself stays scalar per lane (8 calls per K step) — SIMD
// Barrett over u32 × u32 → high-word via paired `_mm256_mul_epu32` on
// even / shifted-odd lanes is a reasonable next optimization; the
// current path prioritizes clear correctness and gets its SIMD win
// from the multiply-store-load and the accumulator update.

#[cfg(target_arch = "x86_64")]
mod avx2_kernel {
    use core::arch::x86_64::*;

    #[inline(always)]
    fn barrett_u32_to_u16(x: u32, mu: u32, modulus: u16) -> u16 {
        let q = ((x as u64 * mu as u64) >> 32) as u32;
        let r = x.wrapping_sub(q.wrapping_mul(modulus as u32)) as u16;
        if r >= modulus {
            r - modulus
        } else {
            r
        }
    }

    #[inline(always)]
    fn barrett_u64_to_u32(x: u64, mu: u64, modulus: u32) -> u32 {
        let q = ((x as u128 * mu as u128) >> 64) as u64;
        let r = x.wrapping_sub(q.wrapping_mul(modulus as u64)) as u32;
        if r >= modulus {
            r - modulus
        } else {
            r
        }
    }

    /// K-slab size for the BLIS-style outer tile loop. Picked so an
    /// (kc × 8) u16 B-panel slice (= `KC_U16 * 8 * 2` bytes) comfortably
    /// fits in L1 alongside a (4 × kc) A-strip, leaving headroom for
    /// prefetched neighbors.
    const KC_U16: usize = 256;

    /// K-slab variant of the 4×8 BLIS-style register-blocked u16
    /// micro-kernel. Loads four u16x8 output rows from `C`, adds the
    /// A[ib..ib+4, k_start..k_start+kc] · B[k_start..k_start+kc, jb..jb+8]
    /// contribution with per-K scalar Barrett + min-sub reductions,
    /// stores the updated rows back. Callers stream multiple slabs
    /// across K without touching other tiles; C is zero-initialized
    /// on the first slab so the accumulator begins in [0, modulus).
    ///
    /// Shared by `matmul_u16_mod` and its triangular/factorization kernels.
    #[target_feature(enable = "avx2")]
    unsafe fn kernel_4x8_u16_slab(
        a: &[u16],
        b: &[u16],
        c: &mut [u16],
        ib: usize,
        jb: usize,
        k_start: usize,
        kc: usize,
        k: usize,
        n: usize,
        mu: u32,
        modulus: u16,
    ) {
        let m_v16 = _mm_set1_epi16(modulus as i16);
        let mut acc = [
            _mm_loadu_si128(c.as_ptr().add((ib) * n + jb) as *const __m128i),
            _mm_loadu_si128(c.as_ptr().add((ib + 1) * n + jb) as *const __m128i),
            _mm_loadu_si128(c.as_ptr().add((ib + 2) * n + jb) as *const __m128i),
            _mm_loadu_si128(c.as_ptr().add((ib + 3) * n + jb) as *const __m128i),
        ];
        let mut scratch: [u32; 8] = [0; 8];

        for p in k_start..k_start + kc {
            let b_u16 = _mm_loadu_si128(b.as_ptr().add(p * n + jb) as *const __m128i);
            let b_u32 = _mm256_cvtepu16_epi32(b_u16);

            for sub_i in 0..4 {
                let a_val = a[(ib + sub_i) * k + p];
                let a_bcast = _mm256_set1_epi32(a_val as i32);
                let prod = _mm256_mullo_epi32(a_bcast, b_u32);
                _mm256_storeu_si256(scratch.as_mut_ptr() as *mut __m256i, prod);

                let mut reds: [u16; 8] = [0; 8];
                for idx in 0..8 {
                    reds[idx] = barrett_u32_to_u16(scratch[idx], mu, modulus);
                }
                let red_vec = _mm_loadu_si128(reds.as_ptr() as *const __m128i);
                let sum = _mm_add_epi16(acc[sub_i], red_vec);
                let sub = _mm_sub_epi16(sum, m_v16);
                acc[sub_i] = _mm_min_epu16(sum, sub);
            }
        }

        for sub_i in 0..4 {
            _mm_storeu_si128(
                c.as_mut_ptr().add((ib + sub_i) * n + jb) as *mut __m128i,
                acc[sub_i],
            );
        }
    }

    /// K-slab variant of the 1×8 u16 tail kernel. Same slab semantics
    /// as `kernel_4x8_u16_slab`, specialized to one output row for the
    /// `m % 4` row stragglers.
    #[target_feature(enable = "avx2")]
    unsafe fn kernel_1x8_u16_slab(
        a: &[u16],
        b: &[u16],
        c: &mut [u16],
        i: usize,
        jb: usize,
        k_start: usize,
        kc: usize,
        k: usize,
        n: usize,
        mu: u32,
        modulus: u16,
    ) {
        let m_v16 = _mm_set1_epi16(modulus as i16);
        let mut acc = _mm_loadu_si128(c.as_ptr().add(i * n + jb) as *const __m128i);
        let mut scratch: [u32; 8] = [0; 8];

        for p in k_start..k_start + kc {
            let a_val = a[i * k + p];
            let a_bcast = _mm256_set1_epi32(a_val as i32);
            let b_u16 = _mm_loadu_si128(b.as_ptr().add(p * n + jb) as *const __m128i);
            let b_u32 = _mm256_cvtepu16_epi32(b_u16);
            let prod = _mm256_mullo_epi32(a_bcast, b_u32);
            _mm256_storeu_si256(scratch.as_mut_ptr() as *mut __m256i, prod);

            let mut reds: [u16; 8] = [0; 8];
            for idx in 0..8 {
                reds[idx] = barrett_u32_to_u16(scratch[idx], mu, modulus);
            }
            let red_vec = _mm_loadu_si128(reds.as_ptr() as *const __m128i);
            let sum = _mm_add_epi16(acc, red_vec);
            let sub = _mm_sub_epi16(sum, m_v16);
            acc = _mm_min_epu16(sum, sub);
        }
        _mm_storeu_si128(c.as_mut_ptr().add(i * n + jb) as *mut __m128i, acc);
    }

    /// `C = A · B` over Z/modulus for u16 inputs, `modulus ≤ 2^15`.
    /// BLIS-style outer nest: K is sliced into `KC_U16`-wide slabs,
    /// the `jb` (N-panel) loop sits outside the `ib` (M-strip) loop
    /// per slab so an 8-column B-panel stays L1-resident while A rows
    /// are streamed through it. Inner work runs the slab-variant 4×8
    /// u16 micro-kernel for full (m/4, n/8) tiles; 1×8 slab kernels
    /// cover `m % 4` row tails, and a scalar full-K column loop covers
    /// `n % 8` column tails. Barrett μ = ⌊2^32 / modulus⌋ precomputed
    /// once; per-K Barrett remains scalar.
    #[target_feature(enable = "avx2")]
    pub unsafe fn matmul_u16_mod(
        a: &[u16],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
        modulus: u16,
    ) -> Vec<u16> {
        debug_assert!((2..=32768).contains(&modulus));
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);

        let mu: u32 = ((1u64 << 32) / modulus as u64) as u32;
        let mut c = vec![0u16; m * n];
        let m4 = m - (m % 4);
        let n8 = n - (n % 8);

        // K-slab outer loop. Inside each slab: jb (N-panel) → ib
        // (M-strip) so the 8-col B slice stays L1-resident while we
        // sweep A's rows.
        let mut k_start = 0;
        while k_start < k {
            let kc = core::cmp::min(KC_U16, k - k_start);

            let mut jb = 0;
            while jb < n8 {
                let mut ib = 0;
                while ib < m4 {
                    kernel_4x8_u16_slab(a, b, &mut c, ib, jb, k_start, kc, k, n, mu, modulus);
                    ib += 4;
                }
                for i in m4..m {
                    kernel_1x8_u16_slab(a, b, &mut c, i, jb, k_start, kc, k, n, mu, modulus);
                }
                jb += 8;
            }

            k_start += kc;
        }

        // Column tail (n % 8) — full-K scalar Barrett, run once.
        if n8 < n {
            let md = modulus as u32;
            for i in 0..m {
                for j in n8..n {
                    let mut acc: u32 = 0;
                    for p in 0..k {
                        let prod = (a[i * k + p] as u32) * (b[p * n + j] as u32);
                        acc = (acc + prod % md) % md;
                    }
                    c[i * n + j] = acc as u16;
                }
            }
        }
        c
    }

    /// K-slab size for the u32 BLIS-style outer tile loop. An
    /// (kc × 8) u32 B-panel slice is `KC_U32 * 8 * 4` bytes — picked
    /// to match `KC_U16 * 16` so both kernels have the same L1 budget.
    const KC_U32: usize = 128;

    /// K-slab variant of the 4×8 u32 register-blocked micro-kernel
    /// (`modulus ≤ 2^31`). Four `__m256i` u32x8 accumulators (one per
    /// output row) are loaded from `C`, accumulate the
    /// A[ib..ib+4, k_start..k_start+kc] · B[k_start..k_start+kc, jb..jb+8]
    /// contribution via even/shifted-odd `_mm256_mul_epu32` + scalar
    /// Barrett + min-sub conditional subtract, then written back.
    #[target_feature(enable = "avx2")]
    unsafe fn kernel_4x8_u32_slab(
        a: &[u32],
        b: &[u32],
        c: &mut [u32],
        ib: usize,
        jb: usize,
        k_start: usize,
        kc: usize,
        k: usize,
        n: usize,
        mu: u64,
        modulus: u32,
    ) {
        let m_v32 = _mm256_set1_epi32(modulus as i32);
        let mut acc = [
            _mm256_loadu_si256(c.as_ptr().add((ib) * n + jb) as *const __m256i),
            _mm256_loadu_si256(c.as_ptr().add((ib + 1) * n + jb) as *const __m256i),
            _mm256_loadu_si256(c.as_ptr().add((ib + 2) * n + jb) as *const __m256i),
            _mm256_loadu_si256(c.as_ptr().add((ib + 3) * n + jb) as *const __m256i),
        ];
        let mut scratch_even: [u64; 4] = [0; 4];
        let mut scratch_odd: [u64; 4] = [0; 4];

        for p in k_start..k_start + kc {
            let b_vec = _mm256_loadu_si256(b.as_ptr().add(p * n + jb) as *const __m256i);
            let b_shifted = _mm256_srli_epi64::<32>(b_vec);

            for sub_i in 0..4 {
                let a_val = a[(ib + sub_i) * k + p];
                let a_bcast = _mm256_set1_epi32(a_val as i32);
                let prod_even = _mm256_mul_epu32(a_bcast, b_vec);
                let prod_odd = _mm256_mul_epu32(a_bcast, b_shifted);

                _mm256_storeu_si256(scratch_even.as_mut_ptr() as *mut __m256i, prod_even);
                _mm256_storeu_si256(scratch_odd.as_mut_ptr() as *mut __m256i, prod_odd);

                let mut reds: [u32; 8] = [0; 8];
                for idx in 0..4 {
                    reds[idx * 2] = barrett_u64_to_u32(scratch_even[idx], mu, modulus);
                    reds[idx * 2 + 1] = barrett_u64_to_u32(scratch_odd[idx], mu, modulus);
                }
                let red_vec = _mm256_loadu_si256(reds.as_ptr() as *const __m256i);
                let sum = _mm256_add_epi32(acc[sub_i], red_vec);
                let sub = _mm256_sub_epi32(sum, m_v32);
                acc[sub_i] = _mm256_min_epu32(sum, sub);
            }
        }

        for sub_i in 0..4 {
            _mm256_storeu_si256(
                c.as_mut_ptr().add((ib + sub_i) * n + jb) as *mut __m256i,
                acc[sub_i],
            );
        }
    }

    /// K-slab variant of the 1×8 u32 tail kernel for `m % 4` row
    /// stragglers.
    #[target_feature(enable = "avx2")]
    unsafe fn kernel_1x8_u32_slab(
        a: &[u32],
        b: &[u32],
        c: &mut [u32],
        i: usize,
        jb: usize,
        k_start: usize,
        kc: usize,
        k: usize,
        n: usize,
        mu: u64,
        modulus: u32,
    ) {
        let m_v32 = _mm256_set1_epi32(modulus as i32);
        let mut acc = _mm256_loadu_si256(c.as_ptr().add(i * n + jb) as *const __m256i);
        let mut scratch_even: [u64; 4] = [0; 4];
        let mut scratch_odd: [u64; 4] = [0; 4];

        for p in k_start..k_start + kc {
            let a_val = a[i * k + p];
            let a_bcast = _mm256_set1_epi32(a_val as i32);
            let b_vec = _mm256_loadu_si256(b.as_ptr().add(p * n + jb) as *const __m256i);
            let prod_even = _mm256_mul_epu32(a_bcast, b_vec);
            let b_shifted = _mm256_srli_epi64::<32>(b_vec);
            let prod_odd = _mm256_mul_epu32(a_bcast, b_shifted);

            _mm256_storeu_si256(scratch_even.as_mut_ptr() as *mut __m256i, prod_even);
            _mm256_storeu_si256(scratch_odd.as_mut_ptr() as *mut __m256i, prod_odd);

            let mut reds: [u32; 8] = [0; 8];
            for idx in 0..4 {
                reds[idx * 2] = barrett_u64_to_u32(scratch_even[idx], mu, modulus);
                reds[idx * 2 + 1] = barrett_u64_to_u32(scratch_odd[idx], mu, modulus);
            }
            let red_vec = _mm256_loadu_si256(reds.as_ptr() as *const __m256i);
            let sum = _mm256_add_epi32(acc, red_vec);
            let sub = _mm256_sub_epi32(sum, m_v32);
            acc = _mm256_min_epu32(sum, sub);
        }
        _mm256_storeu_si256(c.as_mut_ptr().add(i * n + jb) as *mut __m256i, acc);
    }

    /// `C = A · B` over Z/modulus for u32 inputs, `modulus ≤ 2^31`.
    /// K-slab outer nest (see `matmul_u16_mod` for the u16 analogue):
    /// slab by `KC_U32`, with `k_slab → jb → ib` nesting so an 8-col
    /// u32 B-panel stays L1-resident while A rows stream through it.
    #[target_feature(enable = "avx2")]
    pub unsafe fn matmul_u32_mod(
        a: &[u32],
        b: &[u32],
        m: usize,
        k: usize,
        n: usize,
        modulus: u32,
    ) -> Vec<u32> {
        debug_assert!((2..=(1u32 << 31)).contains(&modulus));
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);

        let mu: u64 = ((1u128 << 64) / modulus as u128) as u64;
        let mut c = vec![0u32; m * n];
        let m4 = m - (m % 4);
        let n8 = n - (n % 8);

        let mut k_start = 0;
        while k_start < k {
            let kc = core::cmp::min(KC_U32, k - k_start);

            let mut jb = 0;
            while jb < n8 {
                let mut ib = 0;
                while ib < m4 {
                    kernel_4x8_u32_slab(a, b, &mut c, ib, jb, k_start, kc, k, n, mu, modulus);
                    ib += 4;
                }
                for i in m4..m {
                    kernel_1x8_u32_slab(a, b, &mut c, i, jb, k_start, kc, k, n, mu, modulus);
                }
                jb += 8;
            }

            k_start += kc;
        }

        if n8 < n {
            let md = modulus as u64;
            for i in 0..m {
                for j in n8..n {
                    let mut acc: u64 = 0;
                    for p in 0..k {
                        let prod = (a[i * k + p] as u64) * (b[p * n + j] as u64);
                        acc = (acc + prod % md) % md;
                    }
                    c[i * n + j] = acc as u32;
                }
            }
        }
        c
    }
}

/// Strassen-Winograd O(N^{log₂ 7}) ≈ O(N^2.807) recursion on top of
/// the tuned AVX2 gemm kernels. Works over any ring with subtraction
/// (Z/p^n qualifies — additive inverse always exists), so none of
/// the 7 subproducts + 18 add/subs need division.
///
/// Layout: operates on unpacked row-major u16/u32 dense arrays. At
/// each recursion level, if any of (m, k, n) is odd or below the
/// per-call threshold, we fall through to the tuned `matmul_{u16,u32}
/// _mod` leaf kernel. Otherwise we split each input into a 2×2 block
/// grid, launch 7 recursive calls on sub-blocks of half the dimension,
/// and combine the results via 18 element-wise add/sub operations.
///
/// Variant used: **standard Strassen** (7 muls, 18 adds), not the
/// Winograd variant (7 muls, 15 adds) — Winograd saves 3 adds per
/// level but needs more intermediate buffers; the extra allocations
/// swamp the saving at the sizes that reach this code.
#[cfg(target_arch = "x86_64")]
mod strassen {
    use super::avx2_kernel;

    /// Production threshold for u16 Strassen: below this N, the
    /// recursive overhead (7 allocations + 18 add/subs per level)
    /// exceeds what the sub-cubic scaling buys you against the
    /// tuned GEMM leaf.
    pub const STRASSEN_THRESHOLD_U16: usize = 512;

    /// Production threshold for u32 Strassen.
    pub const STRASSEN_THRESHOLD_U32: usize = 512;

    fn extract_u16(
        src: &[u16],
        row_start: usize,
        col_start: usize,
        rows: usize,
        cols: usize,
        src_cols: usize,
    ) -> Vec<u16> {
        let mut out = vec![0u16; rows * cols];
        for i in 0..rows {
            let src_off = (row_start + i) * src_cols + col_start;
            let dst_off = i * cols;
            out[dst_off..dst_off + cols].copy_from_slice(&src[src_off..src_off + cols]);
        }
        out
    }

    fn extract_u32(
        src: &[u32],
        row_start: usize,
        col_start: usize,
        rows: usize,
        cols: usize,
        src_cols: usize,
    ) -> Vec<u32> {
        let mut out = vec![0u32; rows * cols];
        for i in 0..rows {
            let src_off = (row_start + i) * src_cols + col_start;
            let dst_off = i * cols;
            out[dst_off..dst_off + cols].copy_from_slice(&src[src_off..src_off + cols]);
        }
        out
    }

    fn insert_u16(
        dst: &mut [u16],
        row_start: usize,
        col_start: usize,
        rows: usize,
        cols: usize,
        dst_cols: usize,
        src: &[u16],
    ) {
        for i in 0..rows {
            let dst_off = (row_start + i) * dst_cols + col_start;
            let src_off = i * cols;
            dst[dst_off..dst_off + cols].copy_from_slice(&src[src_off..src_off + cols]);
        }
    }

    fn insert_u32(
        dst: &mut [u32],
        row_start: usize,
        col_start: usize,
        rows: usize,
        cols: usize,
        dst_cols: usize,
        src: &[u32],
    ) {
        for i in 0..rows {
            let dst_off = (row_start + i) * dst_cols + col_start;
            let src_off = i * cols;
            dst[dst_off..dst_off + cols].copy_from_slice(&src[src_off..src_off + cols]);
        }
    }

    fn add_mod_u16(a: &[u16], b: &[u16], modulus: u16) -> Vec<u16> {
        debug_assert_eq!(a.len(), b.len());
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            // modulus ≤ 2^15 guarantees a[i] + b[i] < 2^16.
            let s = a[i] + b[i];
            out.push(if s >= modulus { s - modulus } else { s });
        }
        out
    }

    fn sub_mod_u16(a: &[u16], b: &[u16], modulus: u16) -> Vec<u16> {
        debug_assert_eq!(a.len(), b.len());
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            out.push(if a[i] >= b[i] {
                a[i] - b[i]
            } else {
                (a[i] + modulus) - b[i]
            });
        }
        out
    }

    fn add_mod_u32(a: &[u32], b: &[u32], modulus: u32) -> Vec<u32> {
        debug_assert_eq!(a.len(), b.len());
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            // modulus ≤ 2^31 so a[i] + b[i] < 2^32 (fits in u32).
            let s = a[i] + b[i];
            out.push(if s >= modulus { s - modulus } else { s });
        }
        out
    }

    fn sub_mod_u32(a: &[u32], b: &[u32], modulus: u32) -> Vec<u32> {
        debug_assert_eq!(a.len(), b.len());
        let mut out = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            out.push(if a[i] >= b[i] {
                a[i] - b[i]
            } else {
                (a[i] + modulus) - b[i]
            });
        }
        out
    }

    /// Strassen `C = A · B` over Z/modulus for u16 inputs. If any of
    /// `m, k, n` is odd or below `threshold`, falls through to the
    /// tuned `matmul_u16_mod` leaf; otherwise recursively computes 7
    /// half-size subproducts and combines them with 18 mod-add/subs.
    pub fn strassen_u16_mod_with_threshold(
        a: &[u16],
        b: &[u16],
        m: usize,
        k: usize,
        n: usize,
        modulus: u16,
        threshold: usize,
    ) -> Vec<u16> {
        if !m.is_multiple_of(2)
            || !k.is_multiple_of(2)
            || !n.is_multiple_of(2)
            || m < threshold
            || k < threshold
            || n < threshold
        {
            return unsafe { avx2_kernel::matmul_u16_mod(a, b, m, k, n, modulus) };
        }

        let m2 = m / 2;
        let k2 = k / 2;
        let n2 = n / 2;

        let a11 = extract_u16(a, 0, 0, m2, k2, k);
        let a12 = extract_u16(a, 0, k2, m2, k2, k);
        let a21 = extract_u16(a, m2, 0, m2, k2, k);
        let a22 = extract_u16(a, m2, k2, m2, k2, k);
        let b11 = extract_u16(b, 0, 0, k2, n2, n);
        let b12 = extract_u16(b, 0, n2, k2, n2, n);
        let b21 = extract_u16(b, k2, 0, k2, n2, n);
        let b22 = extract_u16(b, k2, n2, k2, n2, n);

        // M1 = (A11 + A22)(B11 + B22)
        let s1 = add_mod_u16(&a11, &a22, modulus);
        let t1 = add_mod_u16(&b11, &b22, modulus);
        let mm1 = strassen_u16_mod_with_threshold(&s1, &t1, m2, k2, n2, modulus, threshold);

        // M2 = (A21 + A22) B11
        let s2 = add_mod_u16(&a21, &a22, modulus);
        let mm2 = strassen_u16_mod_with_threshold(&s2, &b11, m2, k2, n2, modulus, threshold);

        // M3 = A11 (B12 - B22)
        let t3 = sub_mod_u16(&b12, &b22, modulus);
        let mm3 = strassen_u16_mod_with_threshold(&a11, &t3, m2, k2, n2, modulus, threshold);

        // M4 = A22 (B21 - B11)
        let t4 = sub_mod_u16(&b21, &b11, modulus);
        let mm4 = strassen_u16_mod_with_threshold(&a22, &t4, m2, k2, n2, modulus, threshold);

        // M5 = (A11 + A12) B22
        let s5 = add_mod_u16(&a11, &a12, modulus);
        let mm5 = strassen_u16_mod_with_threshold(&s5, &b22, m2, k2, n2, modulus, threshold);

        // M6 = (A21 - A11)(B11 + B12)
        let s6 = sub_mod_u16(&a21, &a11, modulus);
        let t6 = add_mod_u16(&b11, &b12, modulus);
        let mm6 = strassen_u16_mod_with_threshold(&s6, &t6, m2, k2, n2, modulus, threshold);

        // M7 = (A12 - A22)(B21 + B22)
        let s7 = sub_mod_u16(&a12, &a22, modulus);
        let t7 = add_mod_u16(&b21, &b22, modulus);
        let mm7 = strassen_u16_mod_with_threshold(&s7, &t7, m2, k2, n2, modulus, threshold);

        // C11 = M1 + M4 - M5 + M7
        let mut c11 = add_mod_u16(&mm1, &mm4, modulus);
        c11 = sub_mod_u16(&c11, &mm5, modulus);
        c11 = add_mod_u16(&c11, &mm7, modulus);

        // C12 = M3 + M5
        let c12 = add_mod_u16(&mm3, &mm5, modulus);

        // C21 = M2 + M4
        let c21 = add_mod_u16(&mm2, &mm4, modulus);

        // C22 = M1 - M2 + M3 + M6
        let mut c22 = sub_mod_u16(&mm1, &mm2, modulus);
        c22 = add_mod_u16(&c22, &mm3, modulus);
        c22 = add_mod_u16(&c22, &mm6, modulus);

        let mut c = vec![0u16; m * n];
        insert_u16(&mut c, 0, 0, m2, n2, n, &c11);
        insert_u16(&mut c, 0, n2, m2, n2, n, &c12);
        insert_u16(&mut c, m2, 0, m2, n2, n, &c21);
        insert_u16(&mut c, m2, n2, m2, n2, n, &c22);
        c
    }

    /// u32 analogue of `strassen_u16_mod_with_threshold`, bottoming
    /// out in the tuned `matmul_u32_mod` leaf.
    pub fn strassen_u32_mod_with_threshold(
        a: &[u32],
        b: &[u32],
        m: usize,
        k: usize,
        n: usize,
        modulus: u32,
        threshold: usize,
    ) -> Vec<u32> {
        if !m.is_multiple_of(2)
            || !k.is_multiple_of(2)
            || !n.is_multiple_of(2)
            || m < threshold
            || k < threshold
            || n < threshold
        {
            return unsafe { avx2_kernel::matmul_u32_mod(a, b, m, k, n, modulus) };
        }

        let m2 = m / 2;
        let k2 = k / 2;
        let n2 = n / 2;

        let a11 = extract_u32(a, 0, 0, m2, k2, k);
        let a12 = extract_u32(a, 0, k2, m2, k2, k);
        let a21 = extract_u32(a, m2, 0, m2, k2, k);
        let a22 = extract_u32(a, m2, k2, m2, k2, k);
        let b11 = extract_u32(b, 0, 0, k2, n2, n);
        let b12 = extract_u32(b, 0, n2, k2, n2, n);
        let b21 = extract_u32(b, k2, 0, k2, n2, n);
        let b22 = extract_u32(b, k2, n2, k2, n2, n);

        let s1 = add_mod_u32(&a11, &a22, modulus);
        let t1 = add_mod_u32(&b11, &b22, modulus);
        let mm1 = strassen_u32_mod_with_threshold(&s1, &t1, m2, k2, n2, modulus, threshold);

        let s2 = add_mod_u32(&a21, &a22, modulus);
        let mm2 = strassen_u32_mod_with_threshold(&s2, &b11, m2, k2, n2, modulus, threshold);

        let t3 = sub_mod_u32(&b12, &b22, modulus);
        let mm3 = strassen_u32_mod_with_threshold(&a11, &t3, m2, k2, n2, modulus, threshold);

        let t4 = sub_mod_u32(&b21, &b11, modulus);
        let mm4 = strassen_u32_mod_with_threshold(&a22, &t4, m2, k2, n2, modulus, threshold);

        let s5 = add_mod_u32(&a11, &a12, modulus);
        let mm5 = strassen_u32_mod_with_threshold(&s5, &b22, m2, k2, n2, modulus, threshold);

        let s6 = sub_mod_u32(&a21, &a11, modulus);
        let t6 = add_mod_u32(&b11, &b12, modulus);
        let mm6 = strassen_u32_mod_with_threshold(&s6, &t6, m2, k2, n2, modulus, threshold);

        let s7 = sub_mod_u32(&a12, &a22, modulus);
        let t7 = add_mod_u32(&b21, &b22, modulus);
        let mm7 = strassen_u32_mod_with_threshold(&s7, &t7, m2, k2, n2, modulus, threshold);

        let mut c11 = add_mod_u32(&mm1, &mm4, modulus);
        c11 = sub_mod_u32(&c11, &mm5, modulus);
        c11 = add_mod_u32(&c11, &mm7, modulus);

        let c12 = add_mod_u32(&mm3, &mm5, modulus);

        let c21 = add_mod_u32(&mm2, &mm4, modulus);

        let mut c22 = sub_mod_u32(&mm1, &mm2, modulus);
        c22 = add_mod_u32(&c22, &mm3, modulus);
        c22 = add_mod_u32(&c22, &mm6, modulus);

        let mut c = vec![0u32; m * n];
        insert_u32(&mut c, 0, 0, m2, n2, n, &c11);
        insert_u32(&mut c, 0, n2, m2, n2, n, &c12);
        insert_u32(&mut c, m2, 0, m2, n2, n, &c21);
        insert_u32(&mut c, m2, n2, m2, n2, n, &c22);
        c
    }
}

impl MatZpn {
    /// AVX2-accelerated matmul for u16-range moduli (`p^n ≤ 2^15`).
    /// The entry point is safe on every x86-64 CPU: it detects AVX2 at runtime
    /// and uses the scalar floor when the instruction set is absent. Returns
    /// `Some` in the supported modulus range and `None` otherwise.
    #[cfg(target_arch = "x86_64")]
    pub fn matmul_avx2(&self, other: &Self) -> Option<Self> {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!(self.cols, other.rows);
        let modulus_u64 = self.data.p_pow_n;
        if !(2..=32768).contains(&modulus_u64) {
            return None;
        }
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Some(self.matmul_scalar(other));
        }
        let modulus = modulus_u64 as u16;
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let a_u64 = self.data.to_vec();
        let b_u64 = other.data.to_vec();
        let a: Vec<u16> = a_u64.into_iter().map(|v| v as u16).collect();
        let b: Vec<u16> = b_u64.into_iter().map(|v| v as u16).collect();
        let c_u16 = unsafe { avx2_kernel::matmul_u16_mod(&a, &b, m, k, n, modulus) };
        let mut out = MatZpn::zeros(self.p, self.n, m, n);
        for i in 0..m {
            for j in 0..n {
                out.set(i, j, c_u16[i * n + j] as u64);
            }
        }
        Some(out)
    }

    /// AVX2-accelerated matmul for u32-range moduli (`modulus ≤ 2^31`).
    #[cfg(target_arch = "x86_64")]
    pub fn matmul_avx2_u32(&self, other: &Self) -> Option<Self> {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!(self.cols, other.rows);
        let modulus_u64 = self.data.p_pow_n;
        if !(2..=(1u64 << 31)).contains(&modulus_u64) {
            return None;
        }
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Some(self.matmul_scalar(other));
        }
        let modulus = modulus_u64 as u32;
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let a_u64 = self.data.to_vec();
        let b_u64 = other.data.to_vec();
        let a: Vec<u32> = a_u64.into_iter().map(|v| v as u32).collect();
        let b: Vec<u32> = b_u64.into_iter().map(|v| v as u32).collect();
        let c_u32 = unsafe { avx2_kernel::matmul_u32_mod(&a, &b, m, k, n, modulus) };
        let mut out = MatZpn::zeros(self.p, self.n, m, n);
        for i in 0..m {
            for j in 0..n {
                out.set(i, j, c_u32[i * n + j] as u64);
            }
        }
        Some(out)
    }

    /// Strassen-Winograd O(N^2.807) matmul for u16-range moduli
    /// (`p^n ≤ 2^15`), crossing over to the tuned AVX2 GEMM leaf at
    /// `strassen::STRASSEN_THRESHOLD_U16` (default 512). Returns
    /// `None` if the modulus is outside the u16-supported range so
    /// callers can dispatch cleanly.
    #[cfg(target_arch = "x86_64")]
    pub fn matmul_strassen_avx2(&self, other: &Self) -> Option<Self> {
        self.matmul_strassen_avx2_with_threshold(other, strassen::STRASSEN_THRESHOLD_U16)
    }

    /// Strassen u16 variant with an explicit recursion threshold. Used
    /// by tests to force deep recursion on small matrices and
    /// cross-validate bit-for-bit against the tuned GEMM leaf.
    #[cfg(target_arch = "x86_64")]
    pub fn matmul_strassen_avx2_with_threshold(
        &self,
        other: &Self,
        threshold: usize,
    ) -> Option<Self> {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!(self.cols, other.rows);
        let modulus_u64 = self.data.p_pow_n;
        if !(2..=32768).contains(&modulus_u64) {
            return None;
        }
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Some(self.matmul_scalar(other));
        }
        let modulus = modulus_u64 as u16;
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let a_u64 = self.data.to_vec();
        let b_u64 = other.data.to_vec();
        let a: Vec<u16> = a_u64.into_iter().map(|v| v as u16).collect();
        let b: Vec<u16> = b_u64.into_iter().map(|v| v as u16).collect();
        let c_u16 = strassen::strassen_u16_mod_with_threshold(&a, &b, m, k, n, modulus, threshold);
        let mut out = MatZpn::zeros(self.p, self.n, m, n);
        for i in 0..m {
            for j in 0..n {
                out.set(i, j, c_u16[i * n + j] as u64);
            }
        }
        Some(out)
    }

    /// Strassen-Winograd matmul for u32-range moduli (`modulus ≤ 2^31`),
    /// crossing over to the tuned AVX2 GEMM leaf at
    /// `strassen::STRASSEN_THRESHOLD_U32` (default 512).
    #[cfg(target_arch = "x86_64")]
    pub fn matmul_strassen_avx2_u32(&self, other: &Self) -> Option<Self> {
        self.matmul_strassen_avx2_u32_with_threshold(other, strassen::STRASSEN_THRESHOLD_U32)
    }

    /// Strassen u32 variant with explicit threshold, for tests.
    #[cfg(target_arch = "x86_64")]
    pub fn matmul_strassen_avx2_u32_with_threshold(
        &self,
        other: &Self,
        threshold: usize,
    ) -> Option<Self> {
        assert_eq!(self.p, other.p);
        assert_eq!(self.n, other.n);
        assert_eq!(self.cols, other.rows);
        let modulus_u64 = self.data.p_pow_n;
        if !(2..=(1u64 << 31)).contains(&modulus_u64) {
            return None;
        }
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Some(self.matmul_scalar(other));
        }
        let modulus = modulus_u64 as u32;
        let (m, k, n) = (self.rows, self.cols, other.cols);
        let a_u64 = self.data.to_vec();
        let b_u64 = other.data.to_vec();
        let a: Vec<u32> = a_u64.into_iter().map(|v| v as u32).collect();
        let b: Vec<u32> = b_u64.into_iter().map(|v| v as u32).collect();
        let c_u32 = strassen::strassen_u32_mod_with_threshold(&a, &b, m, k, n, modulus, threshold);
        let mut out = MatZpn::zeros(self.p, self.n, m, n);
        for i in 0..m {
            for j in 0..n {
                out.set(i, j, c_u32[i * n + j] as u64);
            }
        }
        Some(out)
    }
}

impl MatZpn {
    /// NEON-accelerated matmul for u32-range moduli (p^n up to 2^31).
    /// aarch64 only. Returns `None` for moduli outside `[2, 2^31]`, or
    /// for non-aarch64 targets, so callers can dispatch cleanly to
    /// `matmul`.
    ///
    /// When this method returns `Some(c)`, `c` equals the output of
    /// the scalar Barrett path (`MatZpn::matmul`) bit-for-bit — see
    /// `tests/neon_matzpn_matmul.rs` for the cross-validation.
    #[cfg(target_arch = "aarch64")]
    pub fn matmul_neon_u32(&self, other: &Self) -> Option<Self> {
        assert_eq!(self.p, other.p, "p mismatch");
        assert_eq!(self.n, other.n, "n mismatch");
        assert_eq!(self.cols, other.rows, "matmul dim mismatch");
        let modulus_u64 = self.data.p_pow_n;
        if !(2..=(1u64 << 31)).contains(&modulus_u64) {
            return None;
        }
        let modulus = modulus_u64 as u32;
        let (m, k, n) = (self.rows, self.cols, other.cols);

        let a_u64 = self.data.to_vec();
        let b_u64 = other.data.to_vec();
        let a: Vec<u32> = a_u64.into_iter().map(|v| v as u32).collect();
        let b: Vec<u32> = b_u64.into_iter().map(|v| v as u32).collect();

        let c_u32 = neon_kernel_u32::matmul_u32_mod(&a, &b, m, k, n, modulus);
        let mut out = MatZpn::zeros(self.p, self.n, m, n);
        for i in 0..m {
            for j in 0..n {
                out.set(i, j, c_u32[i * n + j] as u64);
            }
        }
        Some(out)
    }
}

impl MatZpn {
    /// NEON-accelerated matmul for u16-range moduli. Available on
    /// aarch64 (the Apple Silicon / ARM Linux target). For `p^n >
    /// 2^15` or non-aarch64, `matmul` stays the canonical path — this
    /// method returns `None` on those cases so callers can dispatch
    /// cleanly.
    ///
    /// When this method returns `Some(c)`, `c.matmul(b)` (scalar
    /// Barrett) would produce the same MatZpn — the two paths are
    /// cross-validated in `tests/neon_matzpn_matmul.rs`.
    #[cfg(target_arch = "aarch64")]
    pub fn matmul_neon(&self, other: &Self) -> Option<Self> {
        assert_eq!(self.p, other.p, "p mismatch");
        assert_eq!(self.n, other.n, "n mismatch");
        assert_eq!(self.cols, other.rows, "matmul dim mismatch");
        let modulus_u64 = self.data.p_pow_n;
        if !(2..=32768).contains(&modulus_u64) {
            return None;
        }
        let modulus = modulus_u64 as u16;
        let (m, k, n) = (self.rows, self.cols, other.cols);

        // Unpack into dense u16 slices (Mat arithmetic then writes back).
        let a_u64 = self.data.to_vec();
        let b_u64 = other.data.to_vec();
        let a: Vec<u16> = a_u64.into_iter().map(|v| v as u16).collect();
        let b: Vec<u16> = b_u64.into_iter().map(|v| v as u16).collect();

        let c_u16 = neon_kernel::matmul_u16_mod(&a, &b, m, k, n, modulus);
        let mut out = MatZpn::zeros(self.p, self.n, m, n);
        for i in 0..m {
            for j in 0..n {
                out.set(i, j, c_u16[i * n + j] as u64);
            }
        }
        Some(out)
    }
}

// ─── Factorizations and reductions over Z/p^n ──────────────────────
//
// PLUQ as the foundation; rank, det, rref, solve_left, inverse, and
// nullspace built on top. Mirrors FLINT's `nmod_mat_*` surface for
// composite moduli: pivot search requires a *unit* (element coprime
// to the prime `p`). When no unit pivot can be found in the remaining
// submatrix, the algorithm stops and reports the count of unit pivots
// as the rank. For matrices needing a true canonical form over the
// quotient ring (including non-unit pivot rows), a Howell-form algorithm is
// required; this type does not implement one.
//
// All arithmetic uses a `u128` intermediate plus `% modulus` reduction. For
// `p^n ≤ 2^30`, the packed range pwvla supports, `u128 = (u64 · u64)` never
// overflows.

impl MatZpn {
    /// Swap rows `i` and `j` in place. O(cols) cell-by-cell swaps
    /// through the packed storage. Private helper for the
    /// factorization paths.
    #[inline]
    fn swap_rows(&mut self, i: usize, j: usize) {
        debug_assert!(i < self.rows);
        debug_assert!(j < self.rows);
        if i == j {
            return;
        }
        for c in 0..self.cols {
            let a = self.get(i, c);
            let b = self.get(j, c);
            self.set(i, c, b);
            self.set(j, c, a);
        }
    }

    /// Partial PLUQ decomposition over `Z/p^n`. Returns `(P, L, U, Q, rank)`.
    ///
    /// - `P`: `m × m` row permutation matrix.
    /// - `L`: `m × k` (k = min(m, n)) unit lower triangular in its
    ///   first `rank` columns; trailing `k - rank` columns are zero.
    /// - `U`: `k × n` upper triangular in its first `rank` rows
    ///   (pivots on the diagonal); trailing `k - rank` rows are zero.
    /// - `Q`: `n × n` column permutation matrix.
    /// - `rank`: number of unit pivots found.
    ///
    /// `Z/p^n` is a local ring: an element is invertible iff coprime
    /// to the prime `p`. Pivot search at each step looks for such a
    /// unit; when none exists in the remaining submatrix, the
    /// algorithm stops. `rank` is the number of unit pivots, matching
    /// FLINT's `nmod_mat_rank` notion of rank over composite moduli.
    ///
    /// # Reconstruction
    ///
    /// `P · L · U · Q == self` holds **iff** the unit-pivot elimination
    /// leaves no residual — i.e. the trailing submatrix has reduced to
    /// zero by the time the algorithm stops. That is always the case
    /// when `self` is invertible (`rank == min(m, n)`), and can also
    /// hold at lower rank when the non-unit content happens to cancel
    /// during elimination. But when a genuine non-unit *remainder*
    /// survives — e.g. `[2]` over `Z/4`, which yields `rank == 0` and a
    /// zero product — that content is **not** captured by `U`: only the
    /// unit-pivot-reducible part of `self` is reconstructed. A faithful
    /// canonical form over the quotient ring (capturing non-unit pivot
    /// rows) requires a Howell-form algorithm, which this type does not
    /// implement.
    ///
    /// # Examples
    ///
    /// ```
    /// use pwvla::zpn::MatZpn;
    /// // I_3 over Z/25 — full unit-rank, so it reconstructs exactly.
    /// let a = MatZpn::from_fn(5, 2, 3, 3, |i, j| if i == j { 1 } else { 0 });
    /// let (p, l, u, q, rank) = a.pluq();
    /// assert_eq!(rank, 3);
    /// let pluq = p.matmul(&l).matmul(&u).matmul(&q);
    /// assert_eq!(pluq, a);
    /// ```
    pub fn pluq(&self) -> (MatZpn, MatZpn, MatZpn, MatZpn, usize) {
        let m = self.rows;
        let n = self.cols;
        let k = m.min(n);
        let modulus = self.data.p_pow_n;
        let md = modulus as u128;

        let mut work = self.clone();
        let mut row_perm: Vec<usize> = (0..m).collect();
        let mut col_perm: Vec<usize> = (0..n).collect();
        let mut l = MatZpn::zeros(self.p, self.n, m, k);

        let mut r = 0usize;
        while r < k {
            // Find an invertible pivot in work[r..m, r..n]. Lex order:
            // smallest column with a unit, then smallest row in that
            // column. Matches m4rst::Mzd::pluq's pivot rule.
            let mut pivot: Option<(usize, usize)> = None;
            'outer: for c in r..n {
                for i in r..m {
                    let v = work.get(i, c);
                    if v != 0 && zpn_scalar_inverse(v, modulus).is_some() {
                        pivot = Some((i, c));
                        break 'outer;
                    }
                }
            }
            let (pr, pc) = match pivot {
                Some(p) => p,
                None => break,
            };

            // Row swap (in work + row_perm + L's already-built cols 0..r).
            if pr != r {
                work.swap_rows(pr, r);
                row_perm.swap(pr, r);
                for j in 0..r {
                    let a = l.get(pr, j);
                    let b = l.get(r, j);
                    l.set(pr, j, b);
                    l.set(r, j, a);
                }
            }

            // Column swap (in work + col_perm).
            if pc != r {
                for i in 0..m {
                    let a = work.get(i, pc);
                    let b = work.get(i, r);
                    work.set(i, pc, b);
                    work.set(i, r, a);
                }
                col_perm.swap(pc, r);
            }

            // Pivot is now at (r, r) and is invertible.
            let pivot_val = work.get(r, r);
            let pivot_inv = zpn_scalar_inverse(pivot_val, modulus)
                .expect("invariant: pivot was selected as a unit");
            l.set(r, r, 1);
            for i in (r + 1)..m {
                let entry = work.get(i, r);
                if entry == 0 {
                    continue;
                }
                let factor = ((entry as u128 * pivot_inv as u128) % md) as u64;
                l.set(i, r, factor);
                // work[i, j] -= factor · work[r, j], for j in r..n
                for j in r..n {
                    let v_rj = work.get(r, j);
                    let term = (factor as u128 * v_rj as u128) % md;
                    let v_ij = work.get(i, j) as u128;
                    let new_val = if v_ij >= term {
                        v_ij - term
                    } else {
                        v_ij + md - term
                    };
                    work.set(i, j, new_val as u64);
                }
            }

            r += 1;
        }

        // Build P: P[row_perm[i], i] = 1 (column i tells us where row i
        // landed in the permuted matrix, so the inverse permutation maps
        // it back).
        let mut p = MatZpn::zeros(self.p, self.n, m, m);
        for (i, &pi) in row_perm.iter().enumerate() {
            p.set(pi, i, 1);
        }

        // Build Q: Q[i, col_perm[i]] = 1.
        let mut q = MatZpn::zeros(self.p, self.n, n, n);
        for (i, &qi) in col_perm.iter().enumerate() {
            q.set(i, qi, 1);
        }

        // Build U from the upper triangle of the reduced work matrix.
        let mut u = MatZpn::zeros(self.p, self.n, k, n);
        for i in 0..r {
            for j in i..n {
                u.set(i, j, work.get(i, j));
            }
        }

        (p, l, u, q, r)
    }

    /// Rank of `self` over `Z/p^n`: the number of unit pivots in the
    /// PLUQ decomposition. `self` is unmodified.
    ///
    /// # Examples
    ///
    /// ```
    /// use pwvla::zpn::MatZpn;
    /// assert_eq!(MatZpn::from_fn(5, 2, 4, 4, |i, j| if i == j { 1 } else { 0 }).rank(), 4);
    /// assert_eq!(MatZpn::zeros(5, 2, 3, 5).rank(), 0);
    /// ```
    pub fn rank(&self) -> usize {
        let (_, _, _, _, r) = self.pluq();
        r
    }

    /// Determinant over `Z/p^n`. Panics if `self` is not square.
    ///
    /// Returns the true ring determinant, *including* for non-invertible
    /// matrices whose determinant is a nonzero zero-divisor — e.g.
    /// `det([2]) == 2` over `Z/4`, not `0`. Invertible matrices (full
    /// unit-rank) take the fast PLUQ path: the product of `U`'s unit
    /// pivots, signed by the row/column permutation parities. Rank-
    /// deficient matrices fall back to the division-free Samuelson–
    /// Berkowitz determinant, since unit-pivot elimination then has no
    /// pivot to divide by. (A matrix is invertible over `Z/p^n` iff its
    /// determinant is a unit, i.e. coprime to `p`.)
    pub fn det(&self) -> u64 {
        assert_eq!(self.rows, self.cols, "det: matrix must be square");
        let n = self.rows;
        if n == 0 {
            return 1;
        }
        let modulus = self.data.p_pow_n;
        let md = modulus as u128;
        let (p, _, u, q, rank) = self.pluq();
        if rank < n {
            // Non-invertible over Z/p^n: det is a non-unit, possibly a
            // *nonzero* zero-divisor. Unit-pivot elimination can't recover
            // it (no pivot to divide by), so use the division-free path.
            return self.det_berkowitz();
        }
        let mut det: u128 = 1;
        for i in 0..n {
            det = (det * u.get(i, i) as u128) % md;
        }
        // Sign is +1 iff P and Q have the same parity. perm_mat_sign
        // returns true for +1, false for -1.
        let positive = perm_mat_sign(&p) == perm_mat_sign(&q);
        if positive {
            det as u64
        } else if det == 0 {
            0
        } else {
            (md - det) as u64
        }
    }

    /// Determinant over `Z/p^n` via the division-free Samuelson–Berkowitz
    /// algorithm. Correct for every square matrix over the quotient ring,
    /// including singular ones whose determinant is a nonzero zero-divisor
    /// (where the unit-pivot PLUQ has no pivot to divide by). `O(n^4)`
    /// ring operations, all `u128`-intermediate `% p^n` — no division by
    /// ring elements, so it is valid over the non-domain `Z/p^n`.
    ///
    /// Builds the characteristic-polynomial coefficient vector of each
    /// leading principal submatrix via the Berkowitz Toeplitz recurrence;
    /// the determinant is `(-1)^n` times the constant term.
    fn det_berkowitz(&self) -> u64 {
        let n = self.rows;
        let md = self.data.p_pow_n as u128;
        let neg = |x: u128| if x == 0 { 0 } else { md - x };
        let at = |i: usize, j: usize| self.get(i, j) as u128 % md;

        // Char-poly coefficients of the 0×0 submatrix: the constant 1.
        let mut p_prev: Vec<u128> = vec![1];
        for k in 1..=n {
            let m = k - 1; // dimension of the previous leading submatrix
            let a_kk = at(k - 1, k - 1);

            // c[0..=k]: c0 = 1, c1 = -a_kk, c_j = -(S · M^{j-2} · R).
            let mut c = vec![0u128; k + 1];
            c[0] = 1;
            c[1] = neg(a_kk);
            if m > 0 {
                // v walks the powers M^{j-2}·R, starting at R (M^0·R).
                let mut v: Vec<u128> = (0..m).map(|i| at(i, k - 1)).collect();
                for j in 2..=k {
                    // S · v = sum_i A[k-1][i] · v[i]
                    let mut s = 0u128;
                    for i in 0..m {
                        s = (s + at(k - 1, i) * v[i]) % md;
                    }
                    c[j] = neg(s);
                    if j < k {
                        // v <- M · v for the next power
                        let mut nv = vec![0u128; m];
                        for (r, slot) in nv.iter_mut().enumerate() {
                            let mut acc = 0u128;
                            for (cc, &vc) in v.iter().enumerate() {
                                acc = (acc + at(r, cc) * vc) % md;
                            }
                            *slot = acc;
                        }
                        v = nv;
                    }
                }
            }

            // p_k = T · p_prev, with T the (k+1)×k lower Toeplitz of c:
            // p_k[r] = sum_col c[r-col] · p_prev[col].
            let mut p_k = vec![0u128; k + 1];
            for (r, slot) in p_k.iter_mut().enumerate() {
                let mut acc = 0u128;
                for (col, &pc) in p_prev.iter().enumerate() {
                    if r >= col {
                        acc = (acc + c[r - col] * pc) % md;
                    }
                }
                *slot = acc;
            }
            p_prev = p_k;
        }

        // p_prev has length n+1; constant term = (-1)^n · det.
        let const_term = p_prev[n];
        let det = if n.is_multiple_of(2) {
            const_term
        } else {
            neg(const_term)
        };
        det as u64
    }

    /// Reduced row echelon form over `Z/p^n`. Returns `(rref, rank)`
    /// where `rref` is the row-equivalent canonical form (every
    /// pivot column has a single `1` in the pivot row) and `rank`
    /// is the number of unit pivots found.
    ///
    /// As with `pluq` and `rank`, this uses unit-pivot semantics
    /// — see [`pluq`](Self::pluq) for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use pwvla::zpn::MatZpn;
    /// let a = MatZpn::from_fn(5, 2, 3, 3, |i, j| if i == j { 1 } else { 0 });
    /// let (rref, rank) = a.rref();
    /// assert_eq!(rank, 3);
    /// assert_eq!(rref, a); // I is its own RREF.
    /// ```
    pub fn rref(&self) -> (MatZpn, usize) {
        let m = self.rows;
        let n = self.cols;
        let modulus = self.data.p_pow_n;
        let md = modulus as u128;

        let mut work = self.clone();
        let mut rank = 0usize;
        let mut col = 0usize;

        while rank < m && col < n {
            // Unit pivot search in rows [rank, m) of column col.
            let mut pivot_row: Option<usize> = None;
            for i in rank..m {
                let v = work.get(i, col);
                if v != 0 && zpn_scalar_inverse(v, modulus).is_some() {
                    pivot_row = Some(i);
                    break;
                }
            }
            let pi = match pivot_row {
                Some(i) => i,
                None => {
                    col += 1;
                    continue;
                }
            };
            if pi != rank {
                work.swap_rows(pi, rank);
            }
            // Scale pivot row so pivot = 1.
            let pivot_val = work.get(rank, col);
            let pivot_inv = zpn_scalar_inverse(pivot_val, modulus)
                .expect("invariant: pivot was selected as a unit");
            for j in col..n {
                let v = work.get(rank, j) as u128;
                work.set(rank, j, ((v * pivot_inv as u128) % md) as u64);
            }
            // Eliminate column in every other row.
            for i in 0..m {
                if i == rank {
                    continue;
                }
                let factor = work.get(i, col);
                if factor == 0 {
                    continue;
                }
                for j in col..n {
                    let v_rj = work.get(rank, j);
                    let term = (factor as u128 * v_rj as u128) % md;
                    let v_ij = work.get(i, j) as u128;
                    let new_val = if v_ij >= term {
                        v_ij - term
                    } else {
                        v_ij + md - term
                    };
                    work.set(i, j, new_val as u64);
                }
            }
            rank += 1;
            col += 1;
        }
        (work, rank)
    }

    /// Solve `A · X = B` over `Z/p^n`. Returns `Some(X)` if a
    /// consistent solution exists, `None` otherwise.
    ///
    /// `A` is `m × n`, `B` is `m × n_rhs`, `X` is `n × n_rhs`. When
    /// `A` is rank-deficient and the system is solvable, returns the
    /// "natural" solution with free variables set to zero; combine
    /// with [`nullspace`](Self::nullspace) to enumerate the full
    /// affine solution set.
    ///
    /// # Panics
    ///
    /// On shape mismatch: `a.rows == b.rows` is required.
    /// On modulus mismatch: `a.p == b.p && a.n == b.n` is required.
    pub fn solve_left(a: &MatZpn, b: &MatZpn) -> Option<MatZpn> {
        assert_eq!(a.p, b.p, "solve_left: ring p mismatch");
        assert_eq!(a.n, b.n, "solve_left: ring n mismatch");
        assert_eq!(
            a.rows, b.rows,
            "solve_left: A.rows ({}) != B.rows ({})",
            a.rows, b.rows
        );
        let m = a.rows;
        let n = a.cols;
        let n_rhs = b.cols;
        let k = m.min(n);
        let modulus = a.data.p_pow_n;
        let md = modulus as u128;

        let (p, l, u, q, rank) = a.pluq();

        // PT_B = P^T · B
        let pt = p.transposed();
        let pt_b = pt.matmul(b);

        // Forward sub: L · Y = PT_B, solving for Y[0..rank].
        let mut y = MatZpn::zeros(a.p, a.n, k, n_rhs);
        for i in 0..rank {
            for c in 0..n_rhs {
                y.set(i, c, pt_b.get(i, c));
            }
            for j in 0..i {
                let l_ij = l.get(i, j);
                if l_ij == 0 {
                    continue;
                }
                for c in 0..n_rhs {
                    let y_jc = y.get(j, c);
                    let term = (l_ij as u128 * y_jc as u128) % md;
                    let y_ic = y.get(i, c) as u128;
                    let new_val = if y_ic >= term {
                        y_ic - term
                    } else {
                        y_ic + md - term
                    };
                    y.set(i, c, new_val as u64);
                }
            }
        }

        // Consistency check: rows [rank, m) of PT_B must be expressible
        // as L[rank..m, 0..rank] · Y[0..rank]; otherwise the system has
        // no solution.
        if rank < m && n_rhs > 0 {
            for i in rank..m {
                for c in 0..n_rhs {
                    let mut acc = pt_b.get(i, c) as u128;
                    for j in 0..rank {
                        let l_ij = l.get(i, j);
                        if l_ij == 0 {
                            continue;
                        }
                        let y_jc = y.get(j, c);
                        let term = (l_ij as u128 * y_jc as u128) % md;
                        acc = if acc >= term {
                            acc - term
                        } else {
                            acc + md - term
                        };
                    }
                    if acc != 0 {
                        return None;
                    }
                }
            }
        }

        // Back sub: U · Z = Y. U[i, i] is the (invertible) pivot.
        // Z[i] = (Y[i] - sum_{j > i} U[i, j] · Z[j]) / U[i, i].
        let mut z = MatZpn::zeros(a.p, a.n, n, n_rhs);
        for i in (0..rank).rev() {
            let u_ii_inv = zpn_scalar_inverse(u.get(i, i), modulus)
                .expect("invariant: U[i, i] is the unit pivot from PLUQ");
            for c in 0..n_rhs {
                let mut acc = y.get(i, c) as u128;
                for j in (i + 1)..rank {
                    let u_ij = u.get(i, j);
                    if u_ij == 0 {
                        continue;
                    }
                    let z_jc = z.get(j, c);
                    let term = (u_ij as u128 * z_jc as u128) % md;
                    acc = if acc >= term {
                        acc - term
                    } else {
                        acc + md - term
                    };
                }
                let val = (acc * u_ii_inv as u128) % md;
                z.set(i, c, val as u64);
            }
        }

        // X = Q^T · Z.
        let qt = q.transposed();
        Some(qt.matmul(&z))
    }

    // NB: `inverse` is implemented elsewhere in this file via Hensel
    // lifting (Newton iteration over F_p → Z/p^n). That approach is
    // asymptotically the same but much better-constant for `n > 1`
    // than a naive `solve_left(A, I)`, so we don't redefine inverse
    // here. See the `pub fn inverse` near `fp_matrix_inverse`.

    /// Compute a basis for the right nullspace of `self`. Returns an
    /// `n × (n - rank)` matrix whose columns span
    /// `ker(self) = { v : self · v = 0 }`. If `self` has full column
    /// rank (rank == n), returns an `n × 0` matrix.
    ///
    /// Rank-nullity: `a.rank() + a.nullspace().cols == a.cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pwvla::zpn::MatZpn;
    /// // A simple 2×3 matrix with a 1-dimensional nullspace.
    /// let a = MatZpn::from_fn(5, 2, 2, 3, |i, j| match (i, j) {
    ///     (0, 0) => 1, (0, 1) => 0, (0, 2) => 7,
    ///     (1, 0) => 0, (1, 1) => 1, (1, 2) => 11,
    ///     _ => 0,
    /// });
    /// let k = a.nullspace();
    /// let ak = a.matmul(&k);
    /// for r in 0..ak.rows {
    ///     for c in 0..ak.cols {
    ///         assert_eq!(ak.get(r, c), 0);
    ///     }
    /// }
    /// ```
    pub fn nullspace(&self) -> MatZpn {
        let n = self.cols;
        let (_, _, u, q, rank) = self.pluq();
        let nullity = n - rank;
        if nullity == 0 {
            return MatZpn::zeros(self.p, self.n, n, 0);
        }
        let modulus = self.data.p_pow_n;
        let md = modulus as u128;

        // Set the "free variable" coordinates: for each k ∈ [rank, n),
        // build a basis vector with z[k] = 1.
        let mut zs = MatZpn::zeros(self.p, self.n, n, nullity);
        for (col, free_var) in (rank..n).enumerate() {
            zs.set(free_var, col, 1);
        }
        // Back-substitute through U for the pivot coordinates:
        // U[i, i] · z[i] + sum_{j > i} U[i, j] · z[j] = 0
        // → z[i] = -inv(U[i, i]) · sum_{j > i} U[i, j] · z[j].
        for i in (0..rank).rev() {
            let u_ii_inv = zpn_scalar_inverse(u.get(i, i), modulus)
                .expect("invariant: U[i, i] is a unit pivot");
            for col in 0..nullity {
                let mut acc: u128 = 0;
                for j in (i + 1)..n {
                    let u_ij = u.get(i, j);
                    if u_ij == 0 {
                        continue;
                    }
                    let z_jc = zs.get(j, col);
                    let term = (u_ij as u128 * z_jc as u128) % md;
                    acc = (acc + term) % md;
                }
                let val = (acc * u_ii_inv as u128) % md;
                let neg = if val == 0 { 0 } else { (md - val) as u64 };
                zs.set(i, col, neg);
            }
        }
        // The basis vectors are in PLUQ-permuted column order; apply
        // Q^T to map back to the original coordinate system.
        let qt = q.transposed();
        qt.matmul(&zs)
    }
}

/// Sign of a permutation matrix: returns `true` for even (+1) and
/// `false` for odd (-1). Recovers the underlying permutation from
/// the matrix layout and counts inversions in O(n^2).
fn perm_mat_sign(p: &MatZpn) -> bool {
    let n = p.rows;
    let mut sigma = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if p.get(j, i) == 1 {
                sigma[i] = j;
                break;
            }
        }
    }
    let mut inv = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if sigma[i] > sigma[j] {
                inv += 1;
            }
        }
    }
    inv.is_multiple_of(2)
}
