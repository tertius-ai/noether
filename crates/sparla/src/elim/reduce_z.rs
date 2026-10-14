// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Z row-reduction primitives shared by the HNF / SNF kernels.
//!
//! Working representation is `Vec<(usize, crate::Integer)>` per row,
//! sorted by column index, with no zero values. The primitives here
//! preserve that invariant. The actual Euclidean drive-to-single-
//! pivot loop lives in [`crate::hnf::direct`]; this module just
//! exposes the per-pair row operations it composes.

use std::cmp::Ordering;
use std::mem;

use crate::Integer;

/// Look up the value at `col` in a sorted sparse row. `None` if not
/// stored (i.e. structural zero).
#[inline]
pub(crate) fn row_get(row: &[(usize, Integer)], col: usize) -> Option<&Integer> {
    row.binary_search_by_key(&col, |e| e.0)
        .ok()
        .map(|off| &row[off].1)
}

/// `target -= coeff * src` over `Z`, preserving the sorted-columns /
/// no-zero invariant on `target`. Merge-style walk; cost
/// `O(|target| + |src|)` in entry count plus the per-entry GMP work.
///
/// Move-based: ownership of `target`'s entries flows into `result`
/// without cloning the `Integer`s. New entries (in columns only
/// `src` has) cost one multiply + negate per cell.
pub(crate) fn row_axpy(
    target: &mut Vec<(usize, Integer)>,
    src: &[(usize, Integer)],
    coeff: &Integer,
) {
    if coeff.cmp0() == Ordering::Equal {
        return;
    }
    let old = mem::take(target);
    let mut result: Vec<(usize, Integer)> = Vec::with_capacity(old.len() + src.len());
    let mut t_iter = old.into_iter().peekable();
    let mut j = 0;
    loop {
        let tc = t_iter.peek().map(|(c, _)| *c);
        let sc = src.get(j).map(|(c, _)| *c);
        match (tc, sc) {
            (None, None) => break,
            (Some(_), None) => {
                result.push(t_iter.next().unwrap());
            }
            (None, Some(sc)) => {
                let prod = coeff * &src[j].1;
                let neg = -prod;
                if neg.cmp0() != Ordering::Equal {
                    result.push((sc, neg));
                }
                j += 1;
            }
            (Some(tc), Some(sc)) => match tc.cmp(&sc) {
                Ordering::Less => {
                    result.push(t_iter.next().unwrap());
                }
                Ordering::Greater => {
                    let prod = coeff * &src[j].1;
                    let neg = -prod;
                    if neg.cmp0() != Ordering::Equal {
                        result.push((sc, neg));
                    }
                    j += 1;
                }
                Ordering::Equal => {
                    let (_, mut tv) = t_iter.next().unwrap();
                    let prod = coeff * &src[j].1;
                    tv -= prod;
                    if tv.cmp0() != Ordering::Equal {
                        result.push((tc, tv));
                    }
                    j += 1;
                }
            },
        }
    }
    *target = result;
}

/// Flip the sign of every entry in `row`. Sparsity pattern is
/// preserved (negation of a nonzero is nonzero).
pub(crate) fn row_negate(row: &mut [(usize, Integer)]) {
    for entry in row.iter_mut() {
        let v = mem::take(&mut entry.1);
        entry.1 = -v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(v: i64) -> Integer {
        Integer::from(v)
    }

    fn row(entries: &[(usize, i64)]) -> Vec<(usize, Integer)> {
        entries.iter().map(|&(c, v)| (c, z(v))).collect()
    }

    #[test]
    fn row_get_finds_stored_entries() {
        let r = row(&[(0, 5), (3, -2), (7, 11)]);
        assert_eq!(row_get(&r, 0), Some(&z(5)));
        assert_eq!(row_get(&r, 3), Some(&z(-2)));
        assert_eq!(row_get(&r, 7), Some(&z(11)));
        assert_eq!(row_get(&r, 2), None);
        assert_eq!(row_get(&r, 8), None);
    }

    #[test]
    fn axpy_disjoint_columns() {
        let mut t = row(&[(0, 1), (2, 3)]);
        let s = row(&[(1, 5), (3, 7)]);
        // t -= 1 * s
        row_axpy(&mut t, &s, &z(1));
        assert_eq!(t, row(&[(0, 1), (1, -5), (2, 3), (3, -7)]));
    }

    #[test]
    fn axpy_overlapping_columns_subtract() {
        let mut t = row(&[(0, 10), (1, 20)]);
        let s = row(&[(0, 3), (1, 4)]);
        // t -= 2 * s = [10 - 6, 20 - 8] = [4, 12]
        row_axpy(&mut t, &s, &z(2));
        assert_eq!(t, row(&[(0, 4), (1, 12)]));
    }

    #[test]
    fn axpy_drops_cells_that_zero_out() {
        let mut t = row(&[(0, 6), (1, 9)]);
        let s = row(&[(0, 2), (1, 3)]);
        // t -= 3 * s = [0, 0]
        row_axpy(&mut t, &s, &z(3));
        assert!(t.is_empty());
    }

    #[test]
    fn axpy_with_zero_coefficient_is_noop() {
        let mut t = row(&[(0, 7), (5, 11)]);
        let original = t.clone();
        row_axpy(&mut t, &row(&[(0, 1), (5, 1)]), &z(0));
        assert_eq!(t, original);
    }

    #[test]
    fn axpy_extends_when_target_shorter() {
        let mut t = row(&[(0, 1)]);
        let s = row(&[(0, 1), (1, 1), (2, 1)]);
        // t -= 1 * s: col 0 → 0 (drop), col 1 → -1, col 2 → -1.
        row_axpy(&mut t, &s, &z(1));
        assert_eq!(t, row(&[(1, -1), (2, -1)]));
    }

    #[test]
    fn axpy_with_negative_coefficient() {
        let mut t = row(&[(0, 5)]);
        let s = row(&[(0, 2), (1, 3)]);
        // t -= -1 * s: col 0 → 5 + 2 = 7, col 1 → 0 + 3 = 3.
        row_axpy(&mut t, &s, &z(-1));
        assert_eq!(t, row(&[(0, 7), (1, 3)]));
    }

    #[test]
    fn axpy_with_large_integer_values() {
        // Bigger than i64 to exercise multi-limb GMP arithmetic.
        let big = Integer::from_str_radix("123456789012345678901234567890", 10).unwrap();
        let s = vec![(0, big.clone()), (2, z(1))];
        let mut t = vec![(0, big.clone() * z(2)), (1, z(7))];
        // t -= 1 * s: col 0 → big, col 1 → 7, col 2 → -1.
        row_axpy(&mut t, &s, &z(1));
        assert_eq!(t, vec![(0, big), (1, z(7)), (2, z(-1))]);
    }

    #[test]
    fn negate_flips_signs() {
        let mut r = row(&[(0, 3), (5, -7), (8, 11)]);
        row_negate(&mut r);
        assert_eq!(r, row(&[(0, -3), (5, 7), (8, -11)]));
    }

    #[test]
    fn negate_empty_row() {
        let mut r: Vec<(usize, Integer)> = Vec::new();
        row_negate(&mut r);
        assert!(r.is_empty());
    }
}
