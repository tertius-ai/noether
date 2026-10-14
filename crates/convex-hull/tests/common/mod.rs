// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shared corpus generators for the integration tests.
#![allow(dead_code)] // not every generator is used by every test binary

use convex_hull::{q, Q};

pub fn cube(n: usize) -> Vec<Vec<Q>> {
    (0..(1usize << n))
        .map(|m| (0..n).map(|i| q(((m >> i) & 1) as i64, 1)).collect())
        .collect()
}

pub fn cross(n: usize) -> Vec<Vec<Q>> {
    let mut v = vec![];
    for i in 0..n {
        let mut a = vec![q(0, 1); n];
        a[i] = q(1, 1);
        v.push(a.clone());
        a[i] = q(-1, 1);
        v.push(a);
    }
    v
}

pub fn simplex(n: usize) -> Vec<Vec<Q>> {
    let mut v = vec![vec![q(0, 1); n]];
    for i in 0..n {
        let mut a = vec![q(0, 1); n];
        a[i] = q(1, 1);
        v.push(a);
    }
    v
}

pub fn cyclic(n: usize, d: usize) -> Vec<Vec<Q>> {
    (1..=n as i64)
        .map(|t| (1..=d as u32).map(|k| q(t.pow(k), 1)).collect())
        .collect()
}
