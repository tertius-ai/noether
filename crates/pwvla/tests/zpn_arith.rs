// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Packed Z/p^n arithmetic: add, sub, neg, scalar_mul.

use pwvla::zpn::PackedZpn;

#[test]
fn add_of_zeros_is_zero() {
    let a = PackedZpn::new(5, 1, 10);
    let b = PackedZpn::new(5, 1, 10);
    let c = a.add(&b);
    for i in 0..10 {
        assert_eq!(c.get(i), 0);
    }
}

#[test]
fn add_reduces_modulo() {
    // In Z/5: 3 + 4 = 7 ≡ 2.
    let a = PackedZpn::from_slice(5, 1, &[3, 4, 2, 1, 0]);
    let b = PackedZpn::from_slice(5, 1, &[4, 3, 2, 1, 0]);
    let c = a.add(&b);
    assert_eq!(c.to_vec(), vec![2, 2, 4, 2, 0]);
}

#[test]
fn add_across_many_slots() {
    // Z/7 with 10 elements (spans multiple slots/words).
    let a: Vec<u64> = (0..10).map(|i| i as u64 % 7).collect();
    let b: Vec<u64> = vec![6, 5, 4, 3, 2, 1, 0, 6, 5, 4];
    let expected: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| (x + y) % 7).collect();

    let pa = PackedZpn::from_slice(7, 1, &a);
    let pb = PackedZpn::from_slice(7, 1, &b);
    let pc = pa.add(&pb);
    assert_eq!(pc.to_vec(), expected);
}

#[test]
fn add_is_commutative() {
    let a = PackedZpn::from_slice(11, 1, &[3, 7, 9, 1, 2]);
    let b = PackedZpn::from_slice(11, 1, &[8, 4, 2, 10, 9]);
    assert_eq!(a.add(&b), b.add(&a));
}

#[test]
fn sub_matches_add_of_neg() {
    let a = PackedZpn::from_slice(13, 1, &[1, 5, 9, 12]);
    let b = PackedZpn::from_slice(13, 1, &[7, 3, 2, 11]);
    let via_sub = a.sub(&b);
    let via_add_neg = a.add(&b.neg());
    assert_eq!(via_sub, via_add_neg);
}

#[test]
fn sub_reduces_correctly() {
    // In Z/5: 2 - 4 = -2 ≡ 3 (mod 5).
    let a = PackedZpn::from_slice(5, 1, &[2, 1, 0, 3]);
    let b = PackedZpn::from_slice(5, 1, &[4, 3, 2, 4]);
    let c = a.sub(&b);
    assert_eq!(c.to_vec(), vec![3, 3, 3, 4]);
}

#[test]
fn sub_self_is_zero() {
    let a = PackedZpn::from_slice(17, 1, &[3, 11, 16, 1, 9, 7]);
    let z = a.sub(&a);
    for i in 0..6 {
        assert_eq!(z.get(i), 0);
    }
}

#[test]
fn neg_is_self_inverse() {
    let a = PackedZpn::from_slice(17, 1, &[3, 11, 16, 1, 9, 7]);
    assert_eq!(a.neg().neg(), a);
}

#[test]
fn neg_of_zero_is_zero() {
    let a = PackedZpn::from_slice(7, 1, &[0, 0, 0, 0, 0]);
    let n = a.neg();
    for i in 0..5 {
        assert_eq!(n.get(i), 0);
    }
}

#[test]
fn add_neg_is_zero() {
    let a = PackedZpn::from_slice(23, 1, &[1, 2, 3, 5, 8, 13, 21]);
    let s = a.add(&a.neg());
    for i in 0..7 {
        assert_eq!(s.get(i), 0);
    }
}

#[test]
fn scalar_mul_by_zero_is_zero() {
    let a = PackedZpn::from_slice(11, 1, &[1, 5, 9, 3]);
    let z = a.scalar_mul(0);
    for i in 0..4 {
        assert_eq!(z.get(i), 0);
    }
}

#[test]
fn scalar_mul_by_one_is_identity() {
    let a = PackedZpn::from_slice(11, 1, &[1, 5, 9, 3]);
    assert_eq!(a.scalar_mul(1), a);
}

#[test]
fn scalar_mul_reduces() {
    // Z/7: 3 · 4 = 12 ≡ 5.
    let a = PackedZpn::from_slice(7, 1, &[3, 2, 6, 1]);
    let c = a.scalar_mul(4);
    assert_eq!(c.to_vec(), vec![5, 1, 3, 4]);
}

#[test]
fn scalar_mul_distributes_over_add() {
    let a = PackedZpn::from_slice(11, 1, &[1, 5, 9, 3]);
    let b = PackedZpn::from_slice(11, 1, &[7, 2, 10, 4]);
    let s: u64 = 6;
    let lhs = a.add(&b).scalar_mul(s);
    let rhs = a.scalar_mul(s).add(&b.scalar_mul(s));
    assert_eq!(lhs, rhs);
}

#[test]
fn add_in_z_p_squared() {
    // Z/25 = Z/5^2. element_bits = 5, default guard = 2, slot = 7.
    let a = PackedZpn::from_slice(5, 2, &[0, 24, 13, 7]);
    let b = PackedZpn::from_slice(5, 2, &[1, 2, 12, 19]);
    let c = a.add(&b);
    assert_eq!(c.to_vec(), vec![1, 1, 0, 1]); // (0+1)%25=1, (24+2)%25=1, (13+12)%25=0, (7+19)%25=1
}
