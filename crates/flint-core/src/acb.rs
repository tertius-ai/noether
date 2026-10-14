// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! [`Acb`] — an owned complex ball shared by FLINT-backed consumers.
//!
//! Each [`Acb`] wraps a FLINT `acb_struct` (real and imaginary
//! components, each an `arb_struct` with midpoint + radius) with
//! RAII: `acb_init` on construction, `acb_clear` on `Drop`,
//! `acb_set` for `Clone`. A complex ball encloses a *rectangle* in
//! ℂ — `[re_mid ± re_rad] × i [im_mid ± im_rad]` — that provably
//! contains the true complex value, with every arithmetic operation
//! widening the component radii to keep the guarantee.
//!
//! Round-trip with the outside world uses 4-tuples
//! `(re_mid, re_rad, im_mid, im_rad)` of `f64`s: exact complex
//! inputs come in with both radii at 0; the materialised radii are
//! always conservative upper bounds.
//!
//! `Acb` is the complex sister of [`crate::Arb`]; the storage and
//! algorithm shapes mirror the real side closely. Where the math
//! genuinely differs, the method signature reflects it: [`abs`]
//! returns an [`Arb`] (the modulus is real), and the squared-
//! magnitude that goes into `frobenius_norm` is real-valued.
//!
//! [`abs`]: Acb::abs

#![allow(unsafe_code)]

use std::fmt;
use std::mem::MaybeUninit;

use crate::{Arb, Rational};
use flint_sys as ffi;

/// An owned complex ball — a rigorous enclosure
/// `[re_mid ± re_rad] + i [im_mid ± im_rad]`, backed by FLINT's
/// `acb`.
///
/// Not `Copy` (owns FLINT-allocated mantissa storage); `Clone`
/// (deep copy via `acb_set`); `Send` (an owned ball has no shared
/// state) and `Sync` (FLINT `acb_*` arithmetic treats `*const`
/// inputs as read-only, no lazy state).
pub struct Acb {
    inner: ffi::acb_struct,
}

// SAFETY: same rationale as the `Arb` impl — FLINT `acb_*`
// arithmetic doesn't lazily mutate `*const` inputs.
unsafe impl Send for Acb {}
unsafe impl Sync for Acb {}

impl Acb {
    /// The exact zero complex ball (`0 + 0i`, radii zero).
    ///
    /// `acb_init` zero-initialises both components.
    pub fn zero() -> Self {
        let mut inner = MaybeUninit::<ffi::acb_struct>::uninit();
        // SAFETY: acb_init fully initialises the struct in place.
        unsafe {
            ffi::acb_init(inner.as_mut_ptr());
            Self {
                inner: inner.assume_init(),
            }
        }
    }

    /// The exact point one.
    pub fn one() -> Self {
        Self::from_re_im(1.0, 0.0)
    }

    /// An exact complex ball at `re + i·im` (both radii 0).
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Acb;
    /// let z = Acb::from_re_im(3.0, 4.0);
    /// assert_eq!(z.re_midpoint(), 3.0);
    /// assert_eq!(z.im_midpoint(), 4.0);
    /// ```
    pub fn from_re_im(re: f64, im: f64) -> Self {
        let mut b = Self::zero();
        // SAFETY: b.inner is a valid acb_struct.
        unsafe {
            ffi::acb_set_d_d(&mut b.inner, re, im);
        }
        b
    }

    /// Enclose exact rational real and imaginary parts at `prec` bits.
    pub fn from_rationals(real: &Rational, imaginary: &Rational, prec: i64) -> Self {
        let real = Arb::from_rational(real, prec);
        let imaginary = Arb::from_rational(imaginary, prec);
        Self::from_arb_parts(&real, &imaginary)
    }

    /// Construct from already-enclosed real and imaginary components.
    pub fn from_arb_parts(real: &Arb, imaginary: &Arb) -> Self {
        let mut output = Self::zero();
        unsafe { ffi::acb_set_arb_arb(output.as_mut_ptr(), real.as_ptr(), imaginary.as_ptr()) };
        output
    }

    /// The complex ball
    /// `[re_mid ± re_rad] + i [im_mid ± im_rad]`. Both radii must
    /// be finite and `>= 0`; both `0` yields an exact complex ball
    /// (identical to [`Acb::from_re_im`]).
    ///
    /// # Panics
    ///
    /// Panics if either radius is negative or non-finite.
    pub fn from_re_im_rad(re_mid: f64, re_rad: f64, im_mid: f64, im_rad: f64) -> Self {
        assert!(
            re_rad >= 0.0 && re_rad.is_finite(),
            "Acb::from_re_im_rad: re_rad must be finite and >= 0; got {re_rad}"
        );
        assert!(
            im_rad >= 0.0 && im_rad.is_finite(),
            "Acb::from_re_im_rad: im_rad must be finite and >= 0; got {im_rad}"
        );
        let mut b = Self::from_re_im(re_mid, im_mid);
        unsafe {
            // Widen real component if needed.
            if re_rad > 0.0 {
                let mut err = MaybeUninit::<ffi::arf_struct>::uninit();
                ffi::arf_init(err.as_mut_ptr());
                let mut err = err.assume_init();
                ffi::arf_set_d(&mut err, re_rad);
                ffi::arb_add_error_arf(&mut b.inner.real, &err);
                ffi::arf_clear(&mut err);
            }
            if im_rad > 0.0 {
                let mut err = MaybeUninit::<ffi::arf_struct>::uninit();
                ffi::arf_init(err.as_mut_ptr());
                let mut err = err.assume_init();
                ffi::arf_set_d(&mut err, im_rad);
                ffi::arb_add_error_arf(&mut b.inner.imag, &err);
                ffi::arf_clear(&mut err);
            }
        }
        b
    }

    /// The real component as an [`Arb`] (a cloned copy).
    pub fn real(&self) -> Arb {
        unsafe { Arb::from_arb_ptr(&self.inner.real) }
    }

    /// The imaginary component as an [`Arb`] (a cloned copy).
    pub fn imag(&self) -> Arb {
        unsafe { Arb::from_arb_ptr(&self.inner.imag) }
    }

    /// The real component's midpoint as an `f64`. Convenience —
    /// equivalent to `self.real().midpoint()` but doesn't clone.
    pub fn re_midpoint(&self) -> f64 {
        // SAFETY: self.inner.real is a valid arb_struct.
        unsafe { ffi::arf_get_d(&self.inner.real.mid, ffi::arf_rnd_t_ARF_RND_NEAR) }
    }

    /// The imaginary component's midpoint as an `f64`.
    pub fn im_midpoint(&self) -> f64 {
        // SAFETY: self.inner.imag is a valid arb_struct.
        unsafe { ffi::arf_get_d(&self.inner.imag.mid, ffi::arf_rnd_t_ARF_RND_NEAR) }
    }

    /// `(re_mid, re_rad, im_mid, im_rad)` `f64` 4-tuple — the
    /// canonical materialisation. Radii are conservative upper
    /// bounds at `prec`.
    pub fn to_4tuple(&self, prec: i64) -> (f64, f64, f64, f64) {
        let (re_mid, re_rad) = self.real().to_pair(prec);
        let (im_mid, im_rad) = self.imag().to_pair(prec);
        (re_mid, re_rad, im_mid, im_rad)
    }

    /// `true` iff this is *exactly* the zero complex ball (`0 + 0i`
    /// with zero radii on both components). A ball whose components
    /// merely contain zero is not exact-zero by this test —
    /// dropping such a ball would lose its rigorous enclosure.
    pub fn is_exact_zero(&self) -> bool {
        // SAFETY: self.inner is a valid acb.
        unsafe { ffi::acb_is_zero(&self.inner) != 0 }
    }

    /// Whether both components have radius zero.
    pub fn is_exact(&self) -> bool {
        unsafe {
            ffi::arb_is_exact(&self.inner.real) != 0 && ffi::arb_is_exact(&self.inner.imag) != 0
        }
    }

    /// Whether this is exactly the point one.
    pub fn is_one(&self) -> bool {
        unsafe { ffi::acb_is_one(self.as_ptr()) != 0 }
    }

    /// Whether this rectangle contains zero.
    pub fn contains_zero(&self) -> bool {
        unsafe {
            ffi::arb_contains_zero(&self.inner.real) != 0
                && ffi::arb_contains_zero(&self.inner.imag) != 0
        }
    }

    /// Whether this rectangle intersects `other` componentwise.
    pub fn overlaps(&self, other: &Self) -> bool {
        unsafe {
            ffi::arb_overlaps(&self.inner.real, &other.inner.real) != 0
                && ffi::arb_overlaps(&self.inner.imag, &other.inner.imag) != 0
        }
    }

    /// `self += other`, midpoints rounded to `prec` bits, radii
    /// widened componentwise by the sum of the input radii plus the
    /// per-component rounding error.
    pub fn add_assign(&mut self, other: &Acb, prec: i64) {
        // SAFETY: acb_add tolerates aliasing of res with an input.
        unsafe {
            ffi::acb_add(&mut self.inner, &self.inner, &other.inner, prec);
        }
    }

    /// `self + other` as a new complex ball at `prec` bits.
    pub fn add(&self, other: &Acb, prec: i64) -> Acb {
        let mut output = self.clone();
        output.add_assign(other, prec);
        output
    }

    /// `self * other` as a new complex ball, midpoint rounded to
    /// `prec` bits. The radius widening follows the standard
    /// complex-ball-multiply rule on each component, which expands
    /// into four real-ball cross-terms — sparb leaves the details
    /// to FLINT.
    pub fn mul(&self, other: &Acb, prec: i64) -> Acb {
        let mut out = Acb::zero();
        unsafe {
            ffi::acb_mul(&mut out.inner, &self.inner, &other.inner, prec);
        }
        out
    }

    /// `self += a * b` rounded to `prec` bits — the fused complex
    /// multiply-add primitive. Same role for `CsrAcb::spmv` /
    /// `matmul` that [`crate::Arb::addmul_assign`] plays for the
    /// real side: one FFI call, no temporary.
    pub fn addmul_assign(&mut self, a: &Acb, b: &Acb, prec: i64) {
        unsafe {
            ffi::acb_addmul(&mut self.inner, &a.inner, &b.inner, prec);
        }
    }

    /// `self - other` as a new complex ball, midpoint rounded to
    /// `prec`. Radii add (interval subtraction).
    pub fn sub(&self, other: &Acb, prec: i64) -> Acb {
        let mut out = Acb::zero();
        unsafe {
            ffi::acb_sub(&mut out.inner, &self.inner, &other.inner, prec);
        }
        out
    }

    /// `-self` as a new complex ball. Negation is *exact* — flips
    /// the sign of both components' midpoints, preserves radii.
    pub fn neg(&self) -> Acb {
        let mut out = Acb::zero();
        unsafe {
            ffi::acb_neg(&mut out.inner, &self.inner);
        }
        out
    }

    /// `|self|` — the modulus, returned as a real [`Arb`] ball.
    /// Rigorous: encloses `√(re² + im²)` for every (re, im) in the
    /// complex ball's rectangle.
    ///
    /// Takes `prec` because the square root introduces rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Acb;
    /// // |3 + 4i| = 5.
    /// let m = Acb::from_re_im(3.0, 4.0).abs(64);
    /// assert_eq!(m.midpoint(), 5.0);
    /// assert_eq!(m.radius(64), 0.0);
    /// ```
    pub fn abs(&self, prec: i64) -> Arb {
        let mut out = Arb::zero();
        // SAFETY: out.as_mut_ptr() is a valid arb_struct (init'd via
        // Arb::zero); self.inner is a valid acb_struct.
        unsafe {
            ffi::acb_abs(out.as_mut_ptr(), &self.inner, prec);
        }
        out
    }

    /// Principal complex square root at `prec`.
    pub fn sqrt(&self, prec: i64) -> Acb {
        let mut out = Acb::zero();
        unsafe {
            ffi::acb_sqrt(&mut out.inner, &self.inner, prec);
        }
        out
    }

    /// `true` iff `self` and `other` are the identical complex
    /// ball — both real and imaginary parts equal as `arb` balls
    /// (same midpoint *and* same radius). This is ball identity,
    /// not "encloses the same complex values".
    pub fn same_ball(&self, other: &Acb) -> bool {
        unsafe { ffi::acb_equal(&self.inner, &other.inner) != 0 }
    }

    /// Raw const pointer to the wrapped `acb_struct` — for callers
    /// that bridge sparb to other FLINT FFI.
    pub fn as_ptr(&self) -> *const ffi::acb_struct {
        &self.inner
    }

    /// Mutable raw pointer to the wrapped FLINT complex ball.
    #[doc(hidden)]
    pub fn as_mut_ptr(&mut self) -> *mut ffi::acb_struct {
        &mut self.inner
    }

    /// Deep-copy a valid initialized FLINT complex ball.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a live initialized `acb_struct` for this call.
    #[doc(hidden)]
    pub unsafe fn clone_from_ptr(ptr: *const ffi::acb_struct) -> Self {
        let mut output = Self::zero();
        unsafe { ffi::acb_set(output.as_mut_ptr(), ptr) };
        output
    }
}

impl Clone for Acb {
    fn clone(&self) -> Self {
        let mut out = Acb::zero();
        unsafe {
            ffi::acb_set(&mut out.inner, &self.inner);
        }
        out
    }
}

impl Drop for Acb {
    fn drop(&mut self) {
        unsafe {
            ffi::acb_clear(&mut self.inner);
        }
    }
}

impl Default for Acb {
    fn default() -> Self {
        Acb::zero()
    }
}

impl fmt::Debug for Acb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re_mid, re_rad, im_mid, im_rad) = self.to_4tuple(53);
        write!(f, "Acb({re_mid} ± {re_rad} + i({im_mid} ± {im_rad}))")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_exact_zero() {
        let z = Acb::zero();
        assert!(z.is_exact_zero());
        assert_eq!(z.re_midpoint(), 0.0);
        assert_eq!(z.im_midpoint(), 0.0);
    }

    #[test]
    fn from_re_im_is_exact() {
        let z = Acb::from_re_im(2.5, -1.5);
        assert_eq!(z.re_midpoint(), 2.5);
        assert_eq!(z.im_midpoint(), -1.5);
        let (rm, rr, im, ir) = z.to_4tuple(53);
        assert_eq!((rm, rr, im, ir), (2.5, 0.0, -1.5, 0.0));
        assert!(!z.is_exact_zero());
    }

    #[test]
    fn from_re_im_rad_carries_both_radii() {
        let z = Acb::from_re_im_rad(1.0, 0.1, 2.0, 0.05);
        let (rm, rr, im, ir) = z.to_4tuple(64);
        assert_eq!(rm, 1.0);
        assert_eq!(im, 2.0);
        assert!(rr >= 0.1);
        assert!(ir >= 0.05);
    }

    #[test]
    #[should_panic(expected = "re_rad must be finite and >= 0")]
    fn from_re_im_rad_rejects_negative_re_radius() {
        let _ = Acb::from_re_im_rad(0.0, -0.1, 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "im_rad must be finite and >= 0")]
    fn from_re_im_rad_rejects_nan_im_radius() {
        let _ = Acb::from_re_im_rad(0.0, 0.0, 0.0, f64::NAN);
    }

    #[test]
    fn add_complex_components_independently() {
        let a = Acb::from_re_im(1.0, 2.0);
        let b = Acb::from_re_im(3.0, 5.0);
        let mut c = a.clone();
        c.add_assign(&b, 64);
        assert_eq!(c.re_midpoint(), 4.0);
        assert_eq!(c.im_midpoint(), 7.0);
    }

    #[test]
    fn complex_multiply_classic() {
        // (1 + 2i) · (3 + 4i) = (3 - 8) + (4 + 6)i = -5 + 10i.
        let a = Acb::from_re_im(1.0, 2.0);
        let b = Acb::from_re_im(3.0, 4.0);
        let c = a.mul(&b, 64);
        assert_eq!(c.re_midpoint(), -5.0);
        assert_eq!(c.im_midpoint(), 10.0);
    }

    #[test]
    fn complex_multiply_with_i_rotates_by_90_degrees() {
        // (a + bi) · i = -b + ai.
        let z = Acb::from_re_im(3.0, 4.0);
        let i = Acb::from_re_im(0.0, 1.0);
        let zi = z.mul(&i, 64);
        assert_eq!(zi.re_midpoint(), -4.0);
        assert_eq!(zi.im_midpoint(), 3.0);
    }

    #[test]
    fn addmul_complex_accumulator() {
        // start: 1+0i. += (2+0i)(3+4i) = (6+8i). result: 7+8i.
        let mut acc = Acb::from_re_im(1.0, 0.0);
        let a = Acb::from_re_im(2.0, 0.0);
        let b = Acb::from_re_im(3.0, 4.0);
        acc.addmul_assign(&a, &b, 64);
        assert_eq!(acc.re_midpoint(), 7.0);
        assert_eq!(acc.im_midpoint(), 8.0);
    }

    #[test]
    fn sub_complex_subtracts_components() {
        let a = Acb::from_re_im(5.0, 7.0);
        let b = Acb::from_re_im(2.0, 3.0);
        let d = a.sub(&b, 64);
        assert_eq!(d.re_midpoint(), 3.0);
        assert_eq!(d.im_midpoint(), 4.0);
    }

    #[test]
    fn neg_flips_both_components() {
        let z = Acb::from_re_im(3.0, -4.0);
        let n = z.neg();
        assert_eq!(n.re_midpoint(), -3.0);
        assert_eq!(n.im_midpoint(), 4.0);
        // Negation is exact — radii stay zero.
        let (rm, rr, im, ir) = n.to_4tuple(64);
        let _ = (rm, im);
        assert_eq!(rr, 0.0);
        assert_eq!(ir, 0.0);
    }

    #[test]
    fn abs_of_3_4i_is_5() {
        // |3 + 4i| = √(9 + 16) = 5 (exactly representable).
        let m = Acb::from_re_im(3.0, 4.0).abs(64);
        assert_eq!(m.midpoint(), 5.0);
        assert_eq!(m.radius(64), 0.0);
    }

    #[test]
    fn abs_of_zero_is_zero() {
        assert!(Acb::zero().abs(64).is_exact_zero());
    }

    #[test]
    fn sqrt_of_minus_one_is_i() {
        // √(-1) = i (principal branch).
        let r = Acb::from_re_im(-1.0, 0.0).sqrt(64);
        assert_eq!(r.re_midpoint(), 0.0);
        assert_eq!(r.im_midpoint(), 1.0);
    }

    #[test]
    fn clone_is_independent_deep_copy() {
        let a = Acb::from_re_im_rad(1.0, 0.1, 2.0, 0.05);
        let b = a.clone();
        assert!(a.same_ball(&b));
        drop(a);
        assert_eq!(b.re_midpoint(), 1.0);
        assert_eq!(b.im_midpoint(), 2.0);
    }

    #[test]
    fn real_imag_extract_components() {
        let z = Acb::from_re_im_rad(3.0, 0.1, 4.0, 0.2);
        let re = z.real();
        let im = z.imag();
        assert_eq!(re.midpoint(), 3.0);
        assert!(re.radius(64) >= 0.1);
        assert_eq!(im.midpoint(), 4.0);
        assert!(im.radius(64) >= 0.2);
    }

    #[test]
    fn same_ball_distinguishes_radius() {
        let exact = Acb::from_re_im(1.0, 0.0);
        let fuzzy = Acb::from_re_im_rad(1.0, 0.1, 0.0, 0.0);
        assert!(!exact.same_ball(&fuzzy));
        assert!(exact.same_ball(&Acb::from_re_im(1.0, 0.0)));
    }
}
