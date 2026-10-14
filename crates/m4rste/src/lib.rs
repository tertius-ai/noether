// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
#![deny(missing_docs)]
// `needless_range_loop`: this crate's bit-packed kernels use index
// variables that participate in derived arithmetic (`w * 8` byte
// offsets, `bit_pos / 64` word offsets, etc.). The
// `.iter().enumerate()` rewrite the lint suggests is harder to read
// than the indexed loops we have.
#![allow(clippy::needless_range_loop)]

//! Pure-Rust dense linear algebra over **GF(2^e)** (`e <= 16`).
//!
//! # Design — bit-packed, built on m4rst
//!
//! - [`Gf2e`] — a GF(2^e) field context built from a minimal
//!   polynomial in bit-pattern encoding.
//! - [`Mzed`] — a dense, bit-packed GF(2^e) matrix: it
//!   wraps an `m4rst::Mzd` (a dense F_2 matrix) of width `cols · e`,
//!   so element `(i, j)`'s `e` bits sit in F_2-columns `[j·e, j·e+e)`.
//!
//! - `matmul` goes through the **bit-sliced** path in [`matmul`] —
//!   slice the operands into `e` F_2 bit-planes, run `m4rst`'s F_2
//!   matmul kernel on them (schoolbook `e²` products + a minpoly
//!   reduction), cling the result back. The GF(2^e) field multiply
//!   disappears entirely into `m4rst`.
//! - `add` is a straight word-XOR over the packed backing store.
//! - `rank` / `rref` use Newton-John–blocked elimination over the
//!   field. Their relative performance varies with the extension
//!   degree, so a multi-backend consumer may select another
//!   elimination kernel independently of its matmul storage.
//!
//! # Element encoding
//!
//! A GF(2^e) element is a polynomial over F_2 of degree `< e`, stored
//! in the low `e` bits of a `u16` at the API boundary (so `e <= 16`)
//! and bit-packed `e` bits per element inside [`Mzed`]. Addition is
//! `XOR`. Multiplication is carry-less polynomial multiplication
//! followed by reduction modulo the field's minimal polynomial.
//!
//! The minimal polynomial is passed as a `u64` bit pattern with the
//! degree-`e` term set: e.g. GF(4) = `x^2 + x + 1` is `0b111 = 7`,
//! GF(256) with the AES polynomial `x^8+x^4+x^3+x+1` is `0x11B`.
//!
//! ```
//! use m4rste::{Gf2e, Mzed};
//!
//! // GF(4), minpoly x^2 + x + 1.
//! let gf4 = Gf2e::new(0b111);
//! assert_eq!(gf4.degree(), 2);
//!
//! // 2x2 identity over GF(4) is rank 2, and I * A == A.
//! let id = Mzed::identity(gf4, 2);
//! let a = Mzed::from_u16_buffer(gf4, 2, 2, &[2, 3, 1, 2]);
//! assert_eq!(id.rank(), 2);
//! assert_eq!(Mzed::matmul(&id, &a).to_u16_buffer(), a.to_u16_buffer());
//! ```

pub mod gf2e;
pub mod matmul;
pub mod mzed;
pub mod pluq;
pub mod simd;
pub mod solve;
pub mod transpose;

pub use gf2e::{Gf2e, Gf2eTables};
pub use mzed::Mzed;
