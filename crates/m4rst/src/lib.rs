// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![deny(missing_docs)]

//! Pure-Rust dense linear algebra over **F_2** (`GF(2)`, bit-packed).
//!
//! # Getting started
//!
//! The headline type is [`Mzd`], an owned bit-packed matrix over
//! F_2. The headline operation is [`Mzd::matmul`], which auto-picks
//! the fastest available kernel based on shape (and on
//! `rayon::current_num_threads()` when the [`parallel`][feat]
//! feature is on).
//!
//! [feat]: #features
//!
//! ```
//! use m4rst::Mzd;
//!
//! let a = Mzd::random(64, 64, 0xA);
//! let b = Mzd::random(64, 64, 0xB);
//! let c = Mzd::matmul(&a, &b);
//! assert_eq!(c.rows(), 64);
//!
//! // Round-trip: A · I == A.
//! let i = Mzd::identity(64);
//! assert_eq!(Mzd::matmul(&a, &i).as_words(), a.as_words());
//! ```
//!
//! The full surface includes multiplication, transpose, XOR, echelon form,
//! rank, PLUQ, solve, nullspace, and inverse. See
//! `examples/getting_started.rs` for a walk-through.
//!
//! # Guarantee class
//!
//! All operations are **exact over F_2**: every output bit is
//! mathematically determined by the input bits, with no precision,
//! rounding, or numerical-stability surface. The only failure modes
//! are `Option::None` from [`Mzd::solve_left`][solve_left] on an
//! inconsistent system and [`Mzd::inverse`][inverse] on a singular
//! input (both shape-mismatch panics excepted). Parallel kernels are
//! deterministic by construction — disjoint output rows + XOR
//! accumulation, no order-dependent reduction — and the bit-equal
//! output across thread counts is pinned by `tests/determinism.rs`.
//!
//! [solve_left]: crate::Mzd::solve_left
//! [inverse]: crate::Mzd::inverse
//!
//! # Features
//!
//! - **`parallel`** (off by default): opt-in CPU parallelism via
//!   rayon. Pulls in `rayon = "1"` (Apache-2.0/MIT, no GPL). With
//!   this on, [`Mzd::matmul`] dispatches to outer-row-parallel,
//!   outer-blocked, inner-parallel, or Strassen-with-parallel-leaf kernels
//!   chosen by shape and the rayon thread count. The outer-row path includes
//!   tall products with a single packed output word.
//!
//! - **`validate-libm4ri`** (off by default; for testing only):
//!   links libm4ri to run bit-equality parity tests against. Never
//!   enable in production — this is the only path that pulls in a
//!   GPL dep.
//!
//! # Layout
//!
//! - [`mzd`] — the [`Mzd`] type + construction + bit IO.
//! - [`view`] — zero-copy submatrix views ([`MzdView`],
//!   [`MzdViewMut`]) and the [`MatrixRead`] / [`MatrixWrite`]
//!   traits that the kernels are generic over.
//! - [`matmul`] — every matmul kernel: naive, M4RM, NTABLES=8,
//!   Strassen, and parallel variants. [`Mzd::matmul`] is the
//!   dispatcher you usually want.
//! - [`transpose`] — out-of-place transpose.
//! - [`xor`] — row-level XOR primitives (mostly used internally by
//!   the elimination kernels).
//! - [`echelon`] — `echelonize` (REF + RREF) and `rank`.
//! - [`pluq`] — PLUQ decomposition.
//! - [`solve`] — `solve_left` (linear system solve), `nullspace`,
//!   and `inverse`.
//!
//! # Bit layout
//!
//! `Mzd` is bit-packed in row-major `u64` words, **LSB-first
//! within each word** — matching libm4ri's `mzd_t` convention
//! exactly. Column `c` of row `r` lives at bit `c % 64` of word
//! `c / 64` of row `r`. See [`mzd`] module docs for details.
//!
//! # Versioning
//!
//! - **MSRV**: Rust 1.87 (uses `usize::is_multiple_of`).
//! - **API stability**: `0.x`; names and module shapes may change.

pub mod echelon;
pub mod matmul;
pub mod mzd;
pub mod pluq;
pub(crate) mod pluq_kernels;
pub(crate) mod simd;
pub mod solve;
pub mod transpose;
pub(crate) mod transpose64;
pub mod view;
pub mod xor;

pub use mzd::{Mzd, MzdAllocationError};
pub use view::{MatrixRead, MatrixWrite, MzdView, MzdViewMut};
