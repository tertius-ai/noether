// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Representation vectors — [`representation_vectors_collect`]: all integer
//! `v` with `vᵀGv = k` (theta-series coefficient support).

use crate::error::{EnumError, EnumResult};
use crate::gso::exact_quadratic;
use crate::svp::svp_enumerate;
use flint_core::Integer;

/// All integer `v` with `vᵀGv = k`, one representative per ± pair (canonical
/// sign). A convenience filter over [`svp_enumerate`] with `bound = k`.
///
/// `k = 0` yields the empty list (only the excluded zero vector has norm 0).
///
/// # Errors
///
/// [`EnumError::InvalidBound`] (`k < 0`), plus every error
/// [`svp_enumerate`] can raise (bad/indefinite Gram, enumeration overflow).
pub fn representation_vectors_collect(
    gram: &[Vec<Integer>],
    k: Integer,
) -> EnumResult<Vec<Vec<i64>>> {
    if k.is_negative() {
        return Err(EnumError::InvalidBound { bound: k });
    }
    let sv = svp_enumerate(gram, Some(k.clone()), None)?;
    let mut out = Vec::new();
    for v in sv.vectors {
        if exact_quadratic(gram, &v)? == k {
            out.push(v);
        }
    }
    Ok(out)
}
