// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! A minimal big unsigned integer for group orders.
//!
//! `|S_n|` blows past 64 bits at n = 21, so the kernel cannot
//! represent its own group orders in a primitive integer. We keep a
//! tiny, dependency-free big-unsigned type for exactly this purpose:
//! multiplications of transversal lengths into a product, comparison,
//! and decimal rendering.
//!
//! This is **not** a general-purpose big integer — we ship just what
//! the kernel needs, so the kernel stays pure-Rust and free of any
//! GPL or LGPL runtime dep.

use core::cmp::Ordering;
use core::fmt;

const LIMB_BITS: usize = 32;

/// A non-negative big integer.
///
/// The internal representation is base-2^32 little-endian limbs with
/// no trailing zero limbs (so `BigUInt::zero().limbs` is `[]`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BigUInt {
    /// Little-endian limbs, base 2^32, no trailing zeros.
    limbs: Vec<u32>,
}

impl BigUInt {
    /// The integer 0.
    pub fn zero() -> BigUInt {
        BigUInt { limbs: Vec::new() }
    }

    /// The integer 1.
    pub fn one() -> BigUInt {
        BigUInt { limbs: vec![1] }
    }

    /// `true` iff this is 0.
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// `true` iff this is 1.
    pub fn is_one(&self) -> bool {
        self.limbs.len() == 1 && self.limbs[0] == 1
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    /// Multiply in place by a small primitive `u64`.
    pub fn mul_assign_u64(&mut self, n: u64) {
        if n == 0 {
            self.limbs.clear();
            return;
        }
        if self.is_zero() {
            return;
        }
        let lo = n & 0xFFFF_FFFF;
        let hi = n >> 32;
        if hi == 0 {
            let mut carry: u64 = 0;
            for l in self.limbs.iter_mut() {
                let prod = (*l as u64) * lo + carry;
                *l = prod as u32;
                carry = prod >> 32;
            }
            while carry != 0 {
                self.limbs.push(carry as u32);
                carry >>= 32;
            }
        } else {
            // Full 64-bit multiplier: do it via two 32-bit passes.
            // result = self * lo + (self * hi) << 32
            let term_lo = self.clone();
            let mut term_hi = self.clone();
            term_hi.shl_assign_limbs(1);
            self.limbs.clear();
            let mut a = term_lo;
            a.mul_assign_u64(lo);
            let mut b = term_hi;
            b.mul_assign_u64(hi);
            *self = a.add(&b);
        }
    }

    /// Shift left by `k` limbs (i.e. by `k * 32` bits).
    fn shl_assign_limbs(&mut self, k: usize) {
        if self.is_zero() || k == 0 {
            return;
        }
        let mut new = vec![0u32; self.limbs.len() + k];
        new[k..].copy_from_slice(&self.limbs);
        self.limbs = new;
    }

    /// Add two big-uints (allocating a fresh result).
    pub fn add(&self, other: &BigUInt) -> BigUInt {
        let n = self.limbs.len().max(other.limbs.len());
        let mut out = Vec::with_capacity(n + 1);
        let mut carry: u64 = 0;
        for i in 0..n {
            let a = *self.limbs.get(i).unwrap_or(&0) as u64;
            let b = *other.limbs.get(i).unwrap_or(&0) as u64;
            let s = a + b + carry;
            out.push(s as u32);
            carry = s >> 32;
        }
        if carry != 0 {
            out.push(carry as u32);
        }
        let mut r = BigUInt { limbs: out };
        r.trim();
        r
    }

    /// Multiply two big-uints (schoolbook). The kernel's usage —
    /// products of small transversal lengths — never stresses
    /// performance here.
    pub fn mul(&self, other: &BigUInt) -> BigUInt {
        if self.is_zero() || other.is_zero() {
            return BigUInt::zero();
        }
        let mut out = vec![0u32; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0_u64;
            for (j, &b) in other.limbs.iter().enumerate() {
                let index = i + j;
                let value = (a as u64) * (b as u64) + out[index] as u64 + carry;
                out[index] = value as u32;
                carry = value >> 32;
            }
            let mut index = i + other.limbs.len();
            while carry != 0 {
                let value = out[index] as u64 + carry;
                out[index] = value as u32;
                carry = value >> 32;
                index += 1;
            }
        }
        let mut limbs = out;
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        BigUInt { limbs }
    }

    /// Divide by a small `u64`, returning `(quotient, remainder)`.
    pub fn divmod_u64(&self, d: u64) -> (BigUInt, u64) {
        assert!(d != 0, "BigUInt: division by zero");
        if d <= (u32::MAX as u64) {
            return self.divmod_u32(d as u32);
        }
        // General case: shift–subtract long division on bytes.
        let mut q = BigUInt::zero();
        let mut r: u128 = 0;
        for i in (0..self.bit_length()).rev() {
            r = (r << 1) | (self.bit(i) as u128);
            if r >= d as u128 {
                r -= d as u128;
                q.set_bit(i);
            }
        }
        (q, r as u64)
    }

    fn divmod_u32(&self, d: u32) -> (BigUInt, u64) {
        let mut q = vec![0u32; self.limbs.len()];
        let mut r: u64 = 0;
        for (i, &limb) in self.limbs.iter().enumerate().rev() {
            let cur = (r << 32) | (limb as u64);
            q[i] = (cur / (d as u64)) as u32;
            r = cur % (d as u64);
        }
        let mut q = BigUInt { limbs: q };
        q.trim();
        (q, r)
    }

    fn bit_length(&self) -> usize {
        self.limbs.last().map_or(0, |top| {
            (self.limbs.len() - 1) * LIMB_BITS + LIMB_BITS - top.leading_zeros() as usize
        })
    }

    fn bit(&self, index: usize) -> u32 {
        let limb = index / LIMB_BITS;
        let offset = index % LIMB_BITS;
        self.limbs
            .get(limb)
            .map_or(0, |value| (value >> offset) & 1)
    }

    fn set_bit(&mut self, index: usize) {
        let limb = index / LIMB_BITS;
        let offset = index % LIMB_BITS;
        if limb >= self.limbs.len() {
            self.limbs.resize(limb + 1, 0);
        }
        self.limbs[limb] |= 1 << offset;
    }

    /// Decimal rendering — repeatedly divides by 10^9 and emits.
    pub fn to_decimal(&self) -> String {
        if self.is_zero() {
            return "0".to_string();
        }
        let mut n = self.clone();
        let mut chunks: Vec<u64> = Vec::new();
        let chunk_div: u64 = 1_000_000_000;
        while !n.is_zero() {
            let (q, r) = n.divmod_u64(chunk_div);
            chunks.push(r);
            n = q;
        }
        let mut out = String::new();
        let last = chunks.pop().unwrap();
        out.push_str(&last.to_string());
        for c in chunks.into_iter().rev() {
            out.push_str(&format!("{:09}", c));
        }
        out
    }
}

impl From<u64> for BigUInt {
    fn from(value: u64) -> Self {
        if value == 0 {
            BigUInt::zero()
        } else if value <= u32::MAX as u64 {
            BigUInt {
                limbs: vec![value as u32],
            }
        } else {
            BigUInt {
                limbs: vec![value as u32, (value >> 32) as u32],
            }
        }
    }
}

impl fmt::Debug for BigUInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigUInt({})", self.to_decimal())
    }
}

impl fmt::Display for BigUInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_decimal())
    }
}

impl Ord for BigUInt {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.limbs.len() != other.limbs.len() {
            return self.limbs.len().cmp(&other.limbs.len());
        }
        for (a, b) in self.limbs.iter().zip(other.limbs.iter()).rev() {
            match a.cmp(b) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for BigUInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_constructors() {
        assert!(BigUInt::zero().is_zero());
        assert!(BigUInt::one().is_one());
        assert_eq!(BigUInt::from(42u64).to_decimal(), "42");
        assert_eq!(BigUInt::from(u64::MAX).to_decimal(), u64::MAX.to_string());
    }

    #[test]
    fn factorial_to_decimal() {
        let mut n = BigUInt::one();
        for i in 1..=20u64 {
            n.mul_assign_u64(i);
        }
        // 20!
        assert_eq!(n.to_decimal(), "2432902008176640000");
        for i in 21..=30u64 {
            n.mul_assign_u64(i);
        }
        // 30! = 265252859812191058636308480000000
        assert_eq!(n.to_decimal(), "265252859812191058636308480000000");
    }

    #[test]
    fn ordering_and_mul() {
        let a = BigUInt::from(1_000_000u64);
        let b = a.mul(&a);
        assert_eq!(b.to_decimal(), "1000000000000");
        assert!(a < b);
    }
}
