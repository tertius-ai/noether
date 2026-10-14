// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Transpose over GF(2^e).
//!
//! The implementation copies elements directly into transposed positions.

use crate::mzed::Mzed;

impl Mzed {
    /// `Aᵀ`: swap rows and columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use m4rste::{Gf2e, Mzed};
    /// let f = Gf2e::new(0x11B);
    /// let a = Mzed::from_u16_buffer(f, 2, 3, &[1, 2, 3, 4, 5, 6]);
    /// let at = a.transpose();
    /// assert_eq!(at.rows(), 3);
    /// assert_eq!(at.cols(), 2);
    /// assert_eq!(at.to_u16_buffer(), vec![1, 4, 2, 5, 3, 6]);
    /// assert_eq!(at.transpose().to_u16_buffer(), a.to_u16_buffer());
    /// ```
    pub fn transpose(&self) -> Mzed {
        let f = self.ctx();
        let (rows, cols) = (self.rows(), self.cols());
        let mut out = Mzed::zeros(f, cols, rows);
        for i in 0..rows {
            for j in 0..cols {
                let v = self.get(i, j);
                if v != 0 {
                    out.set(j, i, v);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf2e::Gf2e;

    #[test]
    fn transpose_identity_is_noop() {
        let f = Gf2e::new(0b111);
        let id = Mzed::identity(f, 5);
        assert_eq!(id.transpose().to_u16_buffer(), id.to_u16_buffer());
    }

    #[test]
    fn transpose_twice_is_identity() {
        let f = Gf2e::new(0b111);
        let a = Mzed::from_u16_buffer(f, 3, 5, &[1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(a.transpose().transpose().to_u16_buffer(), a.to_u16_buffer());
    }

    #[test]
    fn transpose_rectangular_shape() {
        let f = Gf2e::new(0b111);
        let a = Mzed::zeros(f, 2, 5);
        let at = a.transpose();
        assert_eq!((at.rows(), at.cols()), (5, 2));
    }
}
