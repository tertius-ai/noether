// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! F_p prime-field arithmetic over `PackedZpn` (n = 1).
//!
//! F_p is the special case Z/p^1 of our general Z/p^n. All ops
//! (add, sub, neg, mul_elementwise, scalar_mul) inherit; this file
//! exercises specifically field semantics.

use pwvla::zpn::PackedZpn;

#[test]
fn fp_every_nonzero_has_inverse() {
    // In F_p (p prime), every non-zero element has a multiplicative inverse.
    let p = 17;
    let values: Vec<u64> = (0..p as u64).collect();
    let pv = PackedZpn::from_slice(p, 1, &values);
    let inv = pv.fp_inverse_elementwise();

    // Check: a · inv(a) = 1 for a != 0; inv(0) = 0 by convention.
    for i in 0..values.len() {
        let a = values[i];
        let a_inv = inv.get(i);
        if a == 0 {
            assert_eq!(a_inv, 0);
        } else {
            assert_eq!((a * a_inv) % p as u64, 1, "inv({a}) = {a_inv} failed");
        }
    }
}

#[test]
fn fp_mul_is_commutative() {
    let p = 13;
    let a = PackedZpn::from_slice(p, 1, &[1, 5, 7, 11, 12, 3]);
    let b = PackedZpn::from_slice(p, 1, &[2, 8, 9, 4, 6, 10]);
    assert_eq!(a.mul_elementwise(&b), b.mul_elementwise(&a));
}

#[test]
fn fp_inverse_involution() {
    // inv(inv(a)) = a for non-zero a.
    let p = 23;
    let a = PackedZpn::from_slice(p, 1, &[1, 5, 7, 22, 11, 3]);
    let inv_inv = a.fp_inverse_elementwise().fp_inverse_elementwise();
    assert_eq!(inv_inv, a);
}

#[test]
fn fp_fermat_little_theorem_witness() {
    // a^p ≡ a (mod p). We verify via repeated multiplication for small p.
    let p = 7u32;
    let a = PackedZpn::from_slice(p, 1, &[1, 2, 3, 4, 5, 6]);
    // Compute a^p via p successive multiplications with a.
    let mut ap = a.clone();
    for _ in 1..p {
        ap = ap.mul_elementwise(&a);
    }
    assert_eq!(ap, a, "Fermat's little theorem: a^p = a in F_p");
}

#[test]
fn fp_add_distributes_over_mul() {
    // (a + b) · c = a·c + b·c in F_p.
    let p = 19;
    let a = PackedZpn::from_slice(p, 1, &[1, 5, 18, 10, 7]);
    let b = PackedZpn::from_slice(p, 1, &[2, 4, 6, 8, 11]);
    let c = PackedZpn::from_slice(p, 1, &[3, 9, 15, 12, 17]);
    let lhs = a.add(&b).mul_elementwise(&c);
    let rhs = a.mul_elementwise(&c).add(&b.mul_elementwise(&c));
    assert_eq!(lhs, rhs);
}

#[test]
fn fp_inverse_of_zero_is_zero_convention() {
    let p = 11;
    let v = PackedZpn::from_slice(p, 1, &[0, 0, 3, 0]);
    let inv = v.fp_inverse_elementwise();
    assert_eq!(inv.get(0), 0);
    assert_eq!(inv.get(1), 0);
    assert_ne!(inv.get(2), 0); // 3^{-1} ≠ 0
    assert_eq!(inv.get(3), 0);
}

#[test]
fn fp_all_field_axioms_on_small_prime() {
    let p = 5;
    // Try every pair of elements to exercise all op combinations.
    for ai in 0..p {
        for bi in 0..p {
            for ci in 0..p {
                let a = PackedZpn::from_slice(p, 1, &[ai as u64]);
                let b = PackedZpn::from_slice(p, 1, &[bi as u64]);
                let c = PackedZpn::from_slice(p, 1, &[ci as u64]);

                // Associativity of add.
                assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
                // Associativity of mul.
                assert_eq!(
                    a.mul_elementwise(&b).mul_elementwise(&c),
                    a.mul_elementwise(&b.mul_elementwise(&c))
                );
                // Distributivity.
                let lhs = a.mul_elementwise(&b.add(&c));
                let rhs = a.mul_elementwise(&b).add(&a.mul_elementwise(&c));
                assert_eq!(lhs, rhs);
            }
        }
    }
}

#[test]
fn fp_large_prime_basic_ops() {
    // Test with a 16-bit prime to ensure no overflow issues.
    let p = 65521; // largest prime < 2^16
    let a = PackedZpn::from_slice(p, 1, &[100, 500, 65520, 1]);
    let b = PackedZpn::from_slice(p, 1, &[999, 12345, 50000, 65520]);

    let sum = a.add(&b);
    assert_eq!(sum.get(0), (100 + 999) % p as u64);
    assert_eq!(sum.get(1), (500 + 12345) % p as u64);
    assert_eq!(sum.get(2), (65520 + 50000) % p as u64);
    assert_eq!(sum.get(3), (1 + 65520) % p as u64);

    let prod = a.mul_elementwise(&b);
    assert_eq!(prod.get(0), (100 * 999) % p as u64);
}
