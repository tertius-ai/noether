// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Packed Z/p^n storage type tests.

use pwvla::zpn::PackedZpn;

#[test]
fn new_zeros() {
    let v = PackedZpn::new(5, 1, 20);
    for i in 0..20 {
        assert_eq!(v.get(i), 0);
    }
}

#[test]
fn roundtrip_small_prime() {
    let data: Vec<u64> = (0..12).map(|i| (i % 5) as u64).collect();
    let v = PackedZpn::from_slice(5, 1, &data);
    assert_eq!(v.to_vec(), data);
}

#[test]
fn roundtrip_larger_prime() {
    // All values must be in [0, 31).
    let data: Vec<u64> = vec![0, 1, 7, 10, 16, 22, 30, 4, 9, 25];
    let v = PackedZpn::from_slice(31, 1, &data);
    assert_eq!(v.to_vec(), data);
}

#[test]
fn roundtrip_z_p_squared() {
    // Z/9 = Z/3^2.
    let data: Vec<u64> = (0..9).collect();
    let v = PackedZpn::from_slice(3, 2, &data);
    assert_eq!(v.to_vec(), data);
}

#[test]
fn pack_ratio_for_z_3() {
    // Z/3, element_bits=2, default guard=2, slot=4 → 16/u64.
    let v = PackedZpn::new(3, 1, 100);
    assert_eq!(v.slots_per_word, 16);
    assert_eq!(v.slot_bits, 4);
    assert_eq!(v.num_words(), 100 / 16 + if 100 % 16 != 0 { 1 } else { 0 });
}

#[test]
fn pack_ratio_for_z_p_large() {
    // Z/65537 (a prime), element_bits=17, default guard=2, slot=19 → 3/u64.
    let v = PackedZpn::new(65537, 1, 10);
    assert_eq!(v.slot_bits, 19);
    assert_eq!(v.slots_per_word, 3);
}

#[test]
fn custom_guard_bits() {
    // Z/7 with high guard (8 bits) for big accumulations.
    // element_bits = 3, slot = 11, slots/word = 64/11 = 5.
    let v = PackedZpn::with_guard(7, 1, 50, 8);
    assert_eq!(v.slot_bits, 11);
    assert_eq!(v.slots_per_word, 5);
}

#[test]
fn set_independent_slots() {
    // Setting one slot shouldn't disturb its neighbors.
    let mut v = PackedZpn::new(5, 1, 16);
    v.set(0, 3);
    v.set(5, 1);
    v.set(10, 4);
    assert_eq!(v.get(0), 3);
    assert_eq!(v.get(5), 1);
    assert_eq!(v.get(10), 4);
    for i in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15] {
        assert_eq!(v.get(i), 0, "slot {i} should still be zero");
    }
}

#[test]
fn reduce_removes_guard_overflow() {
    // Set a raw slot value above p^n and verify reduce normalizes.
    let mut v = PackedZpn::new(5, 1, 3);
    // Default: slot_bits = 5 (elem=3, guard=2), max slot = 31.
    // Directly set a slot to 17 — exceeds p^n=5.
    v.set(0, 17);
    assert_eq!(v.get_raw(0), 17);
    assert_eq!(v.get(0), 17 % 5); // = 2
    v.reduce_slot(0);
    assert_eq!(v.get_raw(0), 2);
}

#[test]
fn equality_after_unreduced_mutation() {
    let a = PackedZpn::from_slice(7, 1, &[1, 2, 3, 4, 5, 6]);
    let mut b = PackedZpn::new(7, 1, 6);
    for i in 0..6 {
        // Store (i+1) + 7 — which reduces to (i+1) mod 7.
        b.set(i, (i as u64 + 1) + 7);
    }
    // a and b should compare equal because PartialEq uses reduced values.
    assert_eq!(a, b);
}

#[test]
fn length_and_word_count_large() {
    let v = PackedZpn::new(11, 1, 1_000);
    // Z/11: element_bits=4, guard=2, slot=6, slots/word=10. 1000/10 = 100 words.
    assert_eq!(v.slot_bits, 6);
    assert_eq!(v.slots_per_word, 10);
    assert_eq!(v.num_words(), 100);
}

#[test]
fn works_for_p_2_as_base_case() {
    // p=2 with n=5 → Z/32. Elements fit in 5 bits. Can still be packed this way.
    let data: Vec<u64> = (0..32).collect();
    let v = PackedZpn::from_slice(2, 5, &data);
    assert_eq!(v.to_vec(), data);
}
