// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Contiguous round-local term storage.

use core::ops::Range;

use crate::MonomialId;

/// One coefficient-bearing term using an interned monomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term<C> {
    /// Interned monomial.
    pub monomial: MonomialId,
    /// Nonzero coefficient.
    pub coefficient: C,
}

/// Contiguous storage for many short-lived polynomial rows.
#[derive(Clone, Debug, Default)]
pub struct TermArena<C> {
    terms: Vec<Term<C>>,
    rows: Vec<Range<usize>>,
}

impl<C> TermArena<C> {
    /// Construct an empty term arena.
    pub fn new() -> Self {
        Self {
            terms: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// Append one complete row and return its zero-based identifier.
    pub fn push_row(&mut self, terms: impl IntoIterator<Item = Term<C>>) -> usize {
        let start = self.terms.len();
        self.terms.extend(terms);
        let end = self.terms.len();
        self.rows.push(start..end);
        self.rows.len() - 1
    }

    /// Borrow one row.
    pub fn row(&self, row: usize) -> &[Term<C>] {
        &self.terms[self.rows[row].clone()]
    }

    /// Number of stored rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Total number of stored terms.
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Drop all rows while retaining allocations.
    pub fn clear(&mut self) {
        self.terms.clear();
        self.rows.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::{Monomial, MonomialArena, MonomialOrder};

    use super::*;

    #[test]
    fn rows_share_one_allocation_owner() {
        let mut monomials = MonomialArena::new(1, MonomialOrder::Lexicographic);
        let x = monomials
            .intern(Monomial::new(vec![1].into_boxed_slice()))
            .unwrap();
        let mut terms = TermArena::new();
        terms.push_row([Term {
            monomial: x,
            coefficient: 3_u32,
        }]);
        assert_eq!(terms.row(0)[0].coefficient, 3);
        assert_eq!(terms.term_count(), 1);
    }
}
