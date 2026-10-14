// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Barrett reduction + packed multiply tests.

use pwvla::zpn::{BarrettReducer, PackedZpn};

#[test]
fn barrett_matches_naive_modulo() {
    for &m in &[2u64, 3, 5, 7, 17, 257, 65521, 100003] {
        let r = BarrettReducer::new(m);
        // Sample inputs covering boundary cases.
        for a in [
            0u64,
            1,
            m - 1,
            m,
            m + 1,
            m * m - 1,
            (m * m).saturating_sub(1),
        ] {
            assert_eq!(r.reduce(a), a % m, "Barrett(m={m}) failed at a={a}");
        }
    }
}

#[test]
fn barrett_large_inputs() {
    let m = 65537u64;
    let r = BarrettReducer::new(m);
    for a in [
        0u64,
        100,
        65536,
        65537,
        65538,
        1_000_000,
        4_294_967_295,
        u64::MAX / 2,
    ] {
        assert_eq!(r.reduce(a), a % m, "Barrett(m={m}) failed at a={a}");
    }
}

#[test]
fn barrett_max_input_near_2_64() {
    // For m < 2^32, reducer handles a up to m² which may be large.
    let m = (1u64 << 31) - 1; // Mersenne 2^31 − 1 = 2147483647
    let r = BarrettReducer::new(m);
    let a = m * m - 1;
    assert_eq!(r.reduce(a), a % m);
}

#[test]
fn scalar_mul_uses_barrett_correctly() {
    // Z/11: 3 · 4 = 12 ≡ 1. Test after the switch to Barrett-backed scalar_mul.
    let a = PackedZpn::from_slice(11, 1, &[3, 7, 10, 1]);
    let c = a.scalar_mul(4);
    assert_eq!(c.to_vec(), vec![1, 6, 7, 4]);
}

#[test]
fn mul_elementwise_small() {
    let a = PackedZpn::from_slice(7, 1, &[1, 2, 3, 4, 5, 6]);
    let b = PackedZpn::from_slice(7, 1, &[6, 5, 4, 3, 2, 1]);
    let c = a.mul_elementwise(&b);
    // (1·6=6)%7=6, (2·5=10)%7=3, (3·4=12)%7=5, (4·3=12)%7=5, (5·2=10)%7=3, (6·1=6)%7=6
    assert_eq!(c.to_vec(), vec![6, 3, 5, 5, 3, 6]);
}

#[test]
fn mul_elementwise_identity() {
    let a = PackedZpn::from_slice(13, 1, &[3, 7, 12, 1, 9]);
    let ones = PackedZpn::from_slice(13, 1, &[1, 1, 1, 1, 1]);
    assert_eq!(a.mul_elementwise(&ones), a);
}

#[test]
fn mul_elementwise_commutative() {
    let a = PackedZpn::from_slice(17, 1, &[1, 5, 9, 13, 16]);
    let b = PackedZpn::from_slice(17, 1, &[7, 11, 2, 8, 4]);
    assert_eq!(a.mul_elementwise(&b), b.mul_elementwise(&a));
}

#[test]
fn mul_elementwise_z_p_squared() {
    // Z/25 = Z/5^2. 7 · 4 = 28 ≡ 3 (mod 25).
    let a = PackedZpn::from_slice(5, 2, &[7, 6, 24, 0]);
    let b = PackedZpn::from_slice(5, 2, &[4, 10, 24, 17]);
    let c = a.mul_elementwise(&b);
    // (7·4=28)%25=3, (6·10=60)%25=10, (24·24=576)%25=576-23·25=1, (0·17)%25=0
    assert_eq!(c.to_vec(), vec![3, 10, 1, 0]);
}

#[test]
fn mul_elementwise_distributes_over_add() {
    let a = PackedZpn::from_slice(11, 1, &[1, 3, 5, 7]);
    let b = PackedZpn::from_slice(11, 1, &[2, 4, 6, 8]);
    let c = PackedZpn::from_slice(11, 1, &[9, 10, 3, 1]);
    let lhs = a.mul_elementwise(&b.add(&c));
    let rhs = a.mul_elementwise(&b).add(&a.mul_elementwise(&c));
    assert_eq!(lhs, rhs);
}
