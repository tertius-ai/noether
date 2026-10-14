// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Safe owned values over FLINT's raw C ABI.

#![deny(missing_docs)]
#![deny(unsafe_code)]

mod acb;
mod ball;
mod integer;
mod rational;

pub use acb::Acb;
pub use ball::{Arb, BallEncodingError};
pub use integer::{Integer, IntegerRadixError, ParseIntegerError};
pub use rational::{
    rational_matrix_rref, rational_matrix_rref_owned, DivisionByZeroError, ParseRationalError,
    Rational, RationalMatrixShapeError,
};
