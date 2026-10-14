// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! [`Arb`] — an owned real ball shared by FLINT-backed consumers.
//!
//! Each [`Arb`] wraps a FLINT `arb_struct` (midpoint + radius)
//! with RAII: `arb_init` on construction, `arb_clear` on `Drop`,
//! `arb_set` for `Clone`. A ball `[mid ± rad]` is a *rigorous
//! enclosure* — it provably contains the true real value, and every
//! arithmetic operation widens the radius to keep that guarantee.
//!
//! Raw lifecycle and arithmetic calls live in this module; storage and
//! algorithms using [`Arb`] remain safe Rust.
//!
//! Round-trip with the outside world uses `(mid, rad)` `f64` pairs:
//! exact inputs come in with `rad == 0`; the materialised `rad` is
//! always a conservative upper bound, so `[mid - rad, mid + rad]`
//! always contains the true value.

#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::fmt;
use std::mem::MaybeUninit;

use flint_sys as ffi;

use crate::Rational;

/// Failure to load or emit an exact Arb encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BallEncodingError;

impl fmt::Display for BallEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid exact Arb ball encoding")
    }
}

impl std::error::Error for BallEncodingError {}

/// An owned real ball — a rigorous enclosure `[midpoint ± radius]`
/// of a real number, backed by FLINT's `arb`.
///
/// `Arb` is not `Copy` (it owns FLINT-allocated mantissa storage);
/// it is `Clone` (deep copy via `arb_set`) and `Send` (an owned ball
/// has no shared state). It is intentionally **not** `Sync` — the
/// underlying C type carries no interior-mutability guarantees we
/// want to vouch for.
pub struct Arb {
    inner: ffi::arb_struct,
}

// An owned `Arb` has no aliasing into shared state — moving one
// to another thread is sound (`Send`). FLINT's `arb_*` arithmetic
// functions treat their `*const arb_struct` inputs as read-only —
// no lazy mutation, no reference counting, no memoisation — so
// sharing `&Arb` across threads is also sound (`Sync`). This is
// load-bearing for `par_spmv`, which fans `&CsrArb` and `&[Arb]`
// out across worker threads.
unsafe impl Send for Arb {}
unsafe impl Sync for Arb {}

impl Arb {
    /// The exact zero ball `{0}` (midpoint 0, radius 0).
    ///
    /// `arb_init` zero-initialises, so this is just an initialised
    /// `arb_struct`.
    pub fn zero() -> Self {
        let mut inner = MaybeUninit::<ffi::arb_struct>::uninit();
        // SAFETY: arb_init fully initialises the struct in place.
        unsafe {
            ffi::arb_init(inner.as_mut_ptr());
            Self {
                inner: inner.assume_init(),
            }
        }
    }

    /// The exact point one.
    pub fn one() -> Self {
        Self::from_f64(1.0)
    }

    /// An exact ball at `mid` (radius 0). The midpoint is set via
    /// `arb_set_d`, which is exact for any finite `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// let a = Arb::from_f64(3.25);
    /// assert_eq!(a.midpoint(), 3.25);
    /// assert_eq!(a.radius(53), 0.0);
    /// ```
    pub fn from_f64(mid: f64) -> Self {
        let mut b = Self::zero();
        // SAFETY: `b.inner` is an initialised arb_struct.
        unsafe {
            ffi::arb_set_d(&mut b.inner, mid);
        }
        b
    }

    /// Enclose an exact rational at `prec` bits.
    pub fn from_rational(value: &Rational, prec: i64) -> Self {
        let mut output = Self::zero();
        unsafe { ffi::arb_set_fmpq(output.as_mut_ptr(), value.as_ptr(), prec) };
        output
    }

    /// The ball `[mid ± rad]`. `rad` must be finite and `>= 0`;
    /// `rad == 0` yields an exact ball (identical to
    /// [`Arb::from_f64`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// let a = Arb::from_mid_rad(1.0, 0.1);
    /// assert_eq!(a.midpoint(), 1.0);
    /// // The materialised radius is a conservative upper bound on 0.1.
    /// assert!(a.radius(53) >= 0.1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `rad` is negative or non-finite.
    pub fn from_mid_rad(mid: f64, rad: f64) -> Self {
        assert!(
            rad >= 0.0 && rad.is_finite(),
            "Arb::from_mid_rad: radius must be finite and >= 0; got {rad}"
        );
        let mut b = Self::from_f64(mid);
        if rad > 0.0 {
            // SAFETY: scratch arf is init'd, set, used, cleared in
            // a single unsafe block; `b.inner` is a valid arb.
            unsafe {
                let mut err = MaybeUninit::<ffi::arf_struct>::uninit();
                ffi::arf_init(err.as_mut_ptr());
                let mut err = err.assume_init();
                ffi::arf_set_d(&mut err, rad);
                ffi::arb_add_error_arf(&mut b.inner, &err);
                ffi::arf_clear(&mut err);
            }
        }
        b
    }

    /// The ball's midpoint as an `f64` (round-to-nearest).
    pub fn midpoint(&self) -> f64 {
        // SAFETY: `self.inner` is a valid arb; arb_mid_ptr returns a
        // valid arf pointer into it.
        unsafe { ffi::arf_get_d(&self.inner.mid, ffi::arf_rnd_t_ARF_RND_NEAR) }
    }

    /// A conservative `f64` upper bound on the ball's radius.
    ///
    /// `[midpoint() - radius(prec), midpoint() + radius(prec)]`
    /// always contains the true value the ball encloses. The
    /// internal `arb` carries a 30-bit `mag` radius (exact); this
    /// method *materialises* that radius as an `f64`, with two
    /// sources of looseness:
    ///
    /// - **`prec` is the *readback* precision**, not the matrix's
    ///   working precision (they need not agree). Higher `prec`
    ///   tightens the intermediate `arf` upper bound used to derive
    ///   the radius; the `f64` cast at the end can still lose bits.
    /// - **`f64` floor**: a true radius larger than `f64::MAX` is
    ///   reported as `f64::INFINITY`; a true radius smaller than
    ///   `f64::MIN_POSITIVE` may round to a subnormal or to `0.0`.
    ///   For "essentially zero" radii (high-precision computation
    ///   on small balls) the materialised radius is at the f64-ulp
    ///   floor of `|midpoint|`, even when the internal `mag` is
    ///   genuinely tighter.
    /// - **`prec < 2`** is FLINT's documented minimum; passing it
    ///   here doesn't panic but the returned bound is at FLINT's
    ///   discretion. Stay at `prec ≥ 2` for predictable readback.
    ///
    /// # Implementation
    ///
    /// FLINT's `arb_get_ubound_arf(u, x, prec)` sets `u` to an
    /// upper bound on the signed value `x` (the ball's *upper
    /// endpoint*, i.e. `u ≥ mid + rad`). Subtracting the signed
    /// midpoint recovers an upper bound on `rad`. The
    /// `.max(0.0)` guards against subtraction underflow when both
    /// readbacks happen to round in opposing directions for a
    /// near-exact ball.
    pub fn radius(&self, prec: i64) -> f64 {
        let mid = self.midpoint();
        // SAFETY: scratch arf init'd / used / cleared in one block.
        unsafe {
            let mut u = MaybeUninit::<ffi::arf_struct>::uninit();
            ffi::arf_init(u.as_mut_ptr());
            let mut u = u.assume_init();
            ffi::arb_get_ubound_arf(&mut u, &self.inner, prec);
            let u_d = ffi::arf_get_d(&u, ffi::arf_rnd_t_ARF_RND_CEIL);
            ffi::arf_clear(&mut u);
            (u_d - mid).max(0.0)
        }
    }

    /// The ball as a `(midpoint, radius)` `f64` pair — convenience
    /// wrapper around [`Self::midpoint`] and [`Self::radius`].
    ///
    /// `prec` is the *readback* precision for the radius materialisation;
    /// see [`Self::radius`] for the f64-floor / overflow caveats. The
    /// midpoint readback is exact round-to-nearest f64 and ignores
    /// `prec`. The returned pair satisfies `[mid - rad, mid + rad]`
    /// rigorously enclosing the true value, modulo a possible
    /// `f64::INFINITY` for the radius if it exceeds `f64::MAX`.
    pub fn to_pair(&self, prec: i64) -> (f64, f64) {
        (self.midpoint(), self.radius(prec))
    }

    /// `true` iff this is *exactly* the zero ball (`{0}`, radius 0).
    /// A ball that merely contains zero is not zero by this test —
    /// dropping such a ball would lose its rigorous enclosure.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// assert!(Arb::zero().is_exact_zero());
    /// // [0 ± 0.1] *contains* zero but isn't the exact zero ball:
    /// assert!(!Arb::from_mid_rad(0.0, 0.1).is_exact_zero());
    /// ```
    pub fn is_exact_zero(&self) -> bool {
        // SAFETY: `self.inner` is a valid arb.
        unsafe { ffi::arb_is_zero(&self.inner) != 0 }
    }

    /// Whether this ball has radius zero.
    pub fn is_exact(&self) -> bool {
        unsafe { ffi::arb_is_exact(self.as_ptr()) != 0 }
    }

    /// Whether zero belongs to this enclosure.
    pub fn contains_zero(&self) -> bool {
        unsafe { ffi::arb_contains_zero(self.as_ptr()) != 0 }
    }

    /// Whether this is exactly the point one.
    pub fn is_one(&self) -> bool {
        unsafe { ffi::arb_is_one(self.as_ptr()) != 0 }
    }

    /// Whether this enclosure intersects `other`.
    pub fn overlaps(&self, other: &Self) -> bool {
        unsafe { ffi::arb_overlaps(self.as_ptr(), other.as_ptr()) != 0 }
    }

    /// Emit FLINT's exact, round-trippable ball encoding.
    pub fn dump(&self) -> Result<String, BallEncodingError> {
        let pointer = unsafe { ffi::arb_dump_str(self.as_ptr()) };
        if pointer.is_null() {
            return Err(BallEncodingError);
        }
        let value = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::flint_free(pointer.cast()) };
        Ok(value)
    }

    /// Load FLINT's exact ball encoding.
    pub fn from_dump(value: &str) -> Result<Self, BallEncodingError> {
        let value = CString::new(value).map_err(|_| BallEncodingError)?;
        let mut output = Self::zero();
        if unsafe { ffi::arb_load_str(output.as_mut_ptr(), value.as_ptr()) } == 0 {
            Ok(output)
        } else {
            Err(BallEncodingError)
        }
    }

    /// Bounded human-readable decimal rendering.
    pub fn display_decimal(&self, precision: u64) -> String {
        let digits = (precision.saturating_mul(302).div_ceil(1000).clamp(16, 40)) as i64;
        let pointer = unsafe { ffi::arb_get_str(self.as_ptr(), digits, 0) };
        if pointer.is_null() {
            return "<ball formatting error>".to_owned();
        }
        let value = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::flint_free(pointer.cast()) };
        value
    }

    /// `self += other`, with the result midpoint rounded to `prec`
    /// bits of mantissa and the result radius widened by:
    ///
    /// - the sum of the two input radii (interval addition), and
    /// - the rounding error of materialising the new midpoint at
    ///   `prec` bits — `≲ |new_midpoint| · 2⁻ᵖʳᵉᶜ`.
    ///
    /// Aliasing of `self` with `other` is permitted (FLINT's
    /// `arb_add` tolerates it). Mantissa allocation is reused
    /// in place when `self` already has space at this precision,
    /// so accumulator-style use across many calls is allocation-
    /// free in steady state.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// let mut acc = Arb::from_f64(2.0);
    /// acc.add_assign(&Arb::from_f64(3.0), 53);
    /// assert_eq!(acc.midpoint(), 5.0);
    /// ```
    pub fn add_assign(&mut self, other: &Arb, prec: i64) {
        // SAFETY: arb_add tolerates aliasing of res with an input;
        // all three pointers are valid arbs.
        unsafe {
            ffi::arb_add(&mut self.inner, &self.inner, &other.inner, prec);
        }
    }

    /// `self + other` as a new ball at `prec` bits.
    pub fn add(&self, other: &Arb, prec: i64) -> Arb {
        let mut output = self.clone();
        output.add_assign(other, prec);
        output
    }

    /// `self * other` as a new ball, midpoint rounded to `prec`
    /// bits.
    ///
    /// The result radius is widened to capture **interval-multiply
    /// cross terms** plus rounding error:
    ///
    /// `rad(self * other) ≲ |mid(self)|·rad(other)
    ///                     + |mid(other)|·rad(self)
    ///                     + rad(self)·rad(other)
    ///                     + |result_mid| · 2⁻ᵖʳᵉᶜ`
    ///
    /// — the standard ball-multiply rule. Exact × exact at any
    /// `prec` produces an exact result iff the product fits the
    /// available mantissa; otherwise the radius captures the
    /// rounding error.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// // Exact integer product — the radius stays zero.
    /// let p = Arb::from_f64(6.0).mul(&Arb::from_f64(7.0), 53);
    /// assert_eq!(p.midpoint(), 42.0);
    /// assert_eq!(p.radius(53), 0.0);
    ///
    /// // Uncertain · exact propagates the uncertainty.
    /// let q = Arb::from_mid_rad(2.0, 0.1).mul(&Arb::from_f64(5.0), 53);
    /// assert_eq!(q.midpoint(), 10.0);
    /// assert!(q.radius(53) >= 0.5);  // |5| · 0.1
    /// ```
    pub fn mul(&self, other: &Arb, prec: i64) -> Arb {
        let mut out = Arb::zero();
        // SAFETY: all three pointers are valid, distinct arbs.
        unsafe {
            ffi::arb_mul(&mut out.inner, &self.inner, &other.inner, prec);
        }
        out
    }

    /// `self += a * b` rounded to `prec` bits — the fused
    /// multiply-add primitive.
    ///
    /// Semantically identical to (and faster than)
    /// `self.add_assign(&a.mul(b, prec), prec)`: one FFI call and
    /// one in-place mantissa update instead of two FFI calls plus
    /// an allocate/free pair for the temporary product. The radius
    /// is widened by the same combination of cross-terms (see
    /// [`Self::mul`]) plus the single fused rounding error,
    /// instead of two separate roundings.
    ///
    /// This is the SpMV inner-loop primitive: about half of sparb's
    /// per-non-zero cost on the bench is in this single call.
    /// Aliasing of `self` with `a` or `b` is permitted.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// let mut acc = Arb::from_f64(1.0);
    /// acc.addmul_assign(&Arb::from_f64(2.0), &Arb::from_f64(3.0), 53);
    /// assert_eq!(acc.midpoint(), 7.0);  // 1 + 2 · 3
    /// ```
    pub fn addmul_assign(&mut self, a: &Arb, b: &Arb, prec: i64) {
        // SAFETY: all three pointers are valid arbs; arb_addmul
        // tolerates `res` aliasing `a` or `b` (per FLINT docs).
        unsafe {
            ffi::arb_addmul(&mut self.inner, &a.inner, &b.inner, prec);
        }
    }

    /// `self - other` as a new ball, midpoint rounded to `prec`
    /// bits.
    ///
    /// Subtraction in ball arithmetic follows the worst-case
    /// interval rule: **radii add**, the midpoint subtracts.
    ///
    /// `rad(self - other) = rad(self) + rad(other) + O(|new_mid| · 2⁻ᵖʳᵉᶜ)`
    ///
    /// This is the load-bearing fact behind catastrophic
    /// cancellation: subtracting two balls whose midpoints are
    /// near-equal produces a midpoint near zero, but the radius
    /// still carries the *sum* of input uncertainties — sparb
    /// honestly reports the resulting wide ball where plain f64
    /// would silently return a tiny meaningless number.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// let d = Arb::from_f64(10.0).sub(&Arb::from_f64(3.0), 53);
    /// assert_eq!(d.midpoint(), 7.0);
    /// // Radii *add* under subtraction (worst-case uncertainty).
    /// let d = Arb::from_mid_rad(5.0, 0.1).sub(&Arb::from_mid_rad(2.0, 0.05), 53);
    /// assert_eq!(d.midpoint(), 3.0);
    /// assert!(d.radius(53) >= 0.15);
    /// ```
    pub fn sub(&self, other: &Arb, prec: i64) -> Arb {
        let mut out = Arb::zero();
        // SAFETY: all three pointers are valid, distinct arbs.
        unsafe {
            ffi::arb_sub(&mut out.inner, &self.inner, &other.inner, prec);
        }
        out
    }

    /// `|self|` as a new ball — the rigorous absolute value.
    ///
    /// For a ball that doesn't straddle zero this is exact (just
    /// flips the sign of the midpoint if needed; same radius).
    /// For a ball that contains zero, the result is the hull
    /// `[0, max(|lo|, |hi|)]` — wider than the original because
    /// the sign of the true value isn't determined.
    ///
    /// Exact: no precision argument, no new rounding error.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// // Negative midpoint flips, radius preserved.
    /// let a = Arb::from_mid_rad(-3.0, 0.1).abs();
    /// assert_eq!(a.midpoint(), 3.0);
    /// assert!(a.radius(53) >= 0.1);
    /// ```
    pub fn abs(&self) -> Arb {
        let mut out = Arb::zero();
        // SAFETY: both pointers are valid, distinct arbs.
        unsafe {
            ffi::arb_abs(&mut out.inner, &self.inner);
        }
        out
    }

    /// `self / other` rounded to `prec` bits.
    ///
    /// If `other` provably doesn't contain zero the result is a tight
    /// enclosure of the true quotient with radius widened to capture
    /// interval-division stretching plus rounding error. If `other`
    /// contains zero FLINT returns a NaN-tagged ball (the principal-
    /// indeterminate enclosure); callers should guard `other` upstream
    /// — sparb's spectral Rayleigh-quotient code guards via
    /// `is_exact_zero` on the denominator.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// let q = Arb::from_f64(7.0).div(&Arb::from_f64(2.0), 53);
    /// assert_eq!(q.midpoint(), 3.5);
    /// ```
    pub fn div(&self, other: &Arb, prec: i64) -> Arb {
        let mut out = Arb::zero();
        // SAFETY: all three pointers are valid, distinct arbs.
        unsafe {
            ffi::arb_div(&mut out.inner, &self.inner, &other.inner, prec);
        }
        out
    }

    /// The smallest ball (up to `prec`-bit rounding) containing both
    /// `self` and `other` — the interval hull as a single `Arb`.
    ///
    /// Used as the rigorous "enclose two real-number candidates in a
    /// single ball" primitive in the spectrum sandwich. Differs from
    /// [`Self::add_assign`] in semantics: union is the hull of the
    /// two ball supports, addition is the Minkowski sum (a ball at
    /// `mid_a + mid_b`).
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// // union of two exact points is the interval between them.
    /// let u = Arb::from_f64(1.0).union(&Arb::from_f64(5.0), 53);
    /// assert_eq!(u.midpoint(), 3.0);
    /// assert!(u.radius(53) >= 2.0);
    /// ```
    pub fn union(&self, other: &Arb, prec: i64) -> Arb {
        let mut out = Arb::zero();
        // SAFETY: all three pointers are valid, distinct arbs.
        unsafe {
            ffi::arb_union(&mut out.inner, &self.inner, &other.inner, prec);
        }
        out
    }

    /// Rigorous `f64` upper endpoint — a value `u` such that the true
    /// ball lies entirely in `(−∞, u]`. Computed via `arb_get_ubound_arf`
    /// then `arf_get_d` with **ceiling** rounding so any f64 loss is
    /// absorbed into a (very small) widening on the high side.
    ///
    /// Equivalent to `midpoint() + radius(prec)`, but spelled
    /// explicitly here because the spectrum sandwich needs *symmetric*
    /// endpoint readback (`upper_endpoint` + `lower_endpoint`) and
    /// mixing midpoint/radius f64 arithmetic with FLINT's bounds
    /// would lose the conservative-rounding guarantee.
    pub fn upper_endpoint(&self, prec: i64) -> f64 {
        // SAFETY: scratch arf init'd / used / cleared in one block.
        unsafe {
            let mut u = MaybeUninit::<ffi::arf_struct>::uninit();
            ffi::arf_init(u.as_mut_ptr());
            let mut u = u.assume_init();
            ffi::arb_get_ubound_arf(&mut u, &self.inner, prec);
            let u_d = ffi::arf_get_d(&u, ffi::arf_rnd_t_ARF_RND_CEIL);
            ffi::arf_clear(&mut u);
            u_d
        }
    }

    /// Rigorous `f64` lower endpoint — a value `l` such that the true
    /// ball lies entirely in `[l, ∞)`. Computed via `arb_get_lbound_arf`
    /// then `arf_get_d` with **floor** rounding so any f64 loss is
    /// absorbed into a (very small) widening on the low side.
    pub fn lower_endpoint(&self, prec: i64) -> f64 {
        // SAFETY: scratch arf init'd / used / cleared in one block.
        unsafe {
            let mut l = MaybeUninit::<ffi::arf_struct>::uninit();
            ffi::arf_init(l.as_mut_ptr());
            let mut l = l.assume_init();
            ffi::arb_get_lbound_arf(&mut l, &self.inner, prec);
            let l_d = ffi::arf_get_d(&l, ffi::arf_rnd_t_ARF_RND_FLOOR);
            ffi::arf_clear(&mut l);
            l_d
        }
    }

    /// `√self` rounded to `prec` bits — principal square root.
    ///
    /// On a ball provably containing only non-negative values, the
    /// result is a tight enclosure of `√x` for every `x` in `self`.
    /// On a ball that may contain negative values, FLINT either
    /// returns a wider enclosure capturing the indeterminate case
    /// or a NaN-tagged ball — callers responsible for guarding.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// let r = Arb::from_f64(4.0).sqrt(53);
    /// assert_eq!(r.midpoint(), 2.0);
    /// assert_eq!(r.radius(53), 0.0);
    /// ```
    pub fn sqrt(&self, prec: i64) -> Arb {
        let mut out = Arb::zero();
        // SAFETY: both pointers are valid, distinct arbs.
        unsafe {
            ffi::arb_sqrt(&mut out.inner, &self.inner, prec);
        }
        out
    }

    /// `-self` as a new ball. Negation is *exact* — it flips the
    /// midpoint's sign and copies the radius. No precision argument
    /// because no mantissa rounding is involved.
    ///
    /// # Examples
    ///
    /// ```
    /// use flint_core::Arb;
    /// let n = Arb::from_mid_rad(2.5, 0.1).neg();
    /// assert_eq!(n.midpoint(), -2.5);
    /// assert!(n.radius(53) >= 0.1);  // radius preserved exactly
    /// ```
    pub fn neg(&self) -> Arb {
        let mut out = Arb::zero();
        // SAFETY: both pointers are valid, distinct arbs.
        unsafe {
            ffi::arb_neg(&mut out.inner, &self.inner);
        }
        out
    }

    /// `true` iff `self` and `other` are the identical ball — same
    /// midpoint *and* same radius. This is ball identity, not
    /// "encloses the same values".
    pub fn same_ball(&self, other: &Arb) -> bool {
        // SAFETY: both are valid arbs.
        unsafe { ffi::arb_equal(&self.inner, &other.inner) != 0 }
    }

    /// Raw const pointer to the wrapped `arb_struct` — for callers
    /// that bridge sparb to other FLINT FFI (e.g. a pyo3 glue layer).
    pub fn as_ptr(&self) -> *const ffi::arb_struct {
        &self.inner
    }

    /// Mutable raw pointer to the wrapped `arb_struct`. Crate-
    /// private: used by `Acb` (in `crate::acb`) to write real-
    /// valued FLINT results — e.g. `acb_abs` — directly into a
    /// fresh `Arb`'s storage without going through a clone.
    #[doc(hidden)]
    pub fn as_mut_ptr(&mut self) -> *mut ffi::arb_struct {
        &mut self.inner
    }

    /// Build by cloning the `arb_struct` at `ptr`. Crate-private:
    /// used by `Acb::real` / `Acb::imag` to extract real / imaginary
    /// components (each of which is an `arb_struct` embedded inside
    /// the `acb_struct`).
    ///
    /// # Safety
    ///
    /// `ptr` must point to a valid `arb_struct` (init'd, owned by
    /// some caller that outlives this call). The caller-side
    /// invariant we rely on is that `Acb` always holds valid
    /// `arb_struct` components.
    pub(crate) unsafe fn from_arb_ptr(ptr: *const ffi::arb_struct) -> Self {
        let mut out = Self::zero();
        // SAFETY: per the function's safety contract; out.inner is
        // a valid arb_struct (init'd via Self::zero) so arb_set's
        // both pointers are valid.
        unsafe { ffi::arb_set(&mut out.inner, ptr) };
        out
    }

    /// Deep-copy a valid initialized FLINT ball.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a live initialized `arb_struct` for this call.
    #[doc(hidden)]
    pub unsafe fn clone_from_ptr(ptr: *const ffi::arb_struct) -> Self {
        unsafe { Self::from_arb_ptr(ptr) }
    }
}

impl Clone for Arb {
    fn clone(&self) -> Self {
        let mut out = Arb::zero();
        // SAFETY: both pointers are valid, distinct arbs.
        unsafe {
            ffi::arb_set(&mut out.inner, &self.inner);
        }
        out
    }
}

impl Drop for Arb {
    fn drop(&mut self) {
        // SAFETY: `self.inner` was arb_init'd and is cleared exactly
        // once here.
        unsafe {
            ffi::arb_clear(&mut self.inner);
        }
    }
}

impl Default for Arb {
    fn default() -> Self {
        Arb::zero()
    }
}

impl fmt::Debug for Arb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 53-bit radius readback is plenty for a Debug rendering.
        let (mid, rad) = self.to_pair(53);
        write!(f, "Arb({mid} ± {rad})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_exact_zero() {
        let z = Arb::zero();
        assert!(z.is_exact_zero());
        assert_eq!(z.midpoint(), 0.0);
        assert_eq!(z.radius(53), 0.0);
    }

    #[test]
    fn from_f64_is_exact() {
        let a = Arb::from_f64(3.25);
        assert_eq!(a.midpoint(), 3.25);
        assert_eq!(a.radius(53), 0.0);
        assert!(!a.is_exact_zero());
    }

    #[test]
    fn from_mid_rad_carries_radius() {
        let a = Arb::from_mid_rad(1.0, 0.5);
        assert_eq!(a.midpoint(), 1.0);
        assert!(a.radius(53) >= 0.5);
        // A ball containing zero is NOT exact-zero.
        let containing_zero = Arb::from_mid_rad(0.0, 0.1);
        assert!(!containing_zero.is_exact_zero());
    }

    #[test]
    #[should_panic(expected = "radius must be finite and >= 0")]
    fn from_mid_rad_rejects_negative_radius() {
        let _ = Arb::from_mid_rad(0.0, -1.0);
    }

    #[test]
    fn add_assign_accumulates() {
        let mut acc = Arb::zero();
        acc.add_assign(&Arb::from_f64(2.0), 53);
        acc.add_assign(&Arb::from_f64(3.0), 53);
        assert_eq!(acc.midpoint(), 5.0);
        assert_eq!(acc.radius(53), 0.0); // small integers: still exact
    }

    #[test]
    fn add_assign_cancelling_yields_exact_zero() {
        let mut acc = Arb::from_f64(4.0);
        acc.add_assign(&Arb::from_f64(-4.0), 53);
        assert!(acc.is_exact_zero());
    }

    #[test]
    fn mul_is_exact_on_small_integers() {
        let p = Arb::from_f64(6.0).mul(&Arb::from_f64(7.0), 53);
        assert_eq!(p.midpoint(), 42.0);
        assert_eq!(p.radius(53), 0.0);
    }

    #[test]
    fn sub_is_exact_on_small_integers() {
        let d = Arb::from_f64(10.0).sub(&Arb::from_f64(3.0), 53);
        assert_eq!(d.midpoint(), 7.0);
        assert_eq!(d.radius(53), 0.0);
    }

    #[test]
    fn sub_widens_under_uncertainty() {
        let a = Arb::from_mid_rad(5.0, 0.1);
        let b = Arb::from_mid_rad(2.0, 0.05);
        let d = a.sub(&b, 53);
        assert_eq!(d.midpoint(), 3.0);
        // Radii add under subtraction — exact uncertainty propagation.
        assert!(d.radius(53) >= 0.15);
    }

    #[test]
    fn neg_flips_midpoint_keeps_radius() {
        let a = Arb::from_mid_rad(2.5, 0.1);
        let n = a.neg();
        assert_eq!(n.midpoint(), -2.5);
        // Negation is exact — radius unchanged.
        assert_eq!(n.radius(53), a.radius(53));
        // Double-neg is the original ball.
        let n2 = n.neg();
        assert!(n2.same_ball(&a));
    }

    #[test]
    fn neg_of_zero_is_zero() {
        let z = Arb::zero();
        assert!(z.neg().is_exact_zero());
    }

    #[test]
    fn abs_of_negative_midpoint_flips_sign_exactly() {
        let a = Arb::from_mid_rad(-7.0, 0.25).abs();
        assert_eq!(a.midpoint(), 7.0);
        // Exact — abs introduces no rounding.
        assert!(a.radius(64) >= 0.25);
    }

    #[test]
    fn abs_of_ball_straddling_zero_is_hull() {
        // [-0.5, 1.0] → |·| ⊆ [0, 1.0]. The result must be the
        // ball [0.5 ± 0.5] (or wider, never narrower).
        let a = Arb::from_mid_rad(0.25, 0.75).abs();
        let (mid, rad) = a.to_pair(64);
        // True |·| max over the input ball is 1.0; result must
        // enclose 0 (the minimum) and 1.0 (the maximum).
        assert!(mid - rad <= 0.0 && 0.0 <= mid + rad);
        assert!(mid - rad <= 1.0 && 1.0 <= mid + rad);
    }

    #[test]
    fn abs_of_zero_is_zero() {
        assert!(Arb::zero().abs().is_exact_zero());
    }

    #[test]
    fn sqrt_of_perfect_squares_is_exact() {
        for k in 0..10 {
            let r = Arb::from_f64((k * k) as f64).sqrt(64);
            assert_eq!(r.midpoint(), k as f64);
            assert_eq!(r.radius(64), 0.0);
        }
    }

    #[test]
    fn sqrt_widens_under_uncertainty() {
        // √(4 ± 0.4): true range [√3.6, √4.4] ≈ [1.897, 2.098].
        // Midpoint ≈ 2.0; radius captures both endpoints.
        let r = Arb::from_mid_rad(4.0, 0.4).sqrt(80);
        let (mid, rad) = r.to_pair(80);
        assert!(mid - rad <= (3.6_f64).sqrt() + 1e-12);
        assert!((4.4_f64).sqrt() - 1e-12 <= mid + rad);
    }

    #[test]
    fn sqrt_of_zero_is_zero() {
        assert!(Arb::zero().sqrt(64).is_exact_zero());
    }

    #[test]
    fn mul_widens_on_inexact() {
        // 1/3 is not representable: the product carries a radius.
        let third = Arb::from_f64(1.0).mul(&Arb::from_f64(1.0), 8);
        // exact here; instead check an inexact product genuinely widens
        let inexact = Arb::from_mid_rad(1.0, 0.1).mul(&Arb::from_f64(2.0), 53);
        assert_eq!(inexact.midpoint(), 2.0);
        assert!(inexact.radius(53) >= 0.2);
        let _ = third;
    }

    #[test]
    fn clone_is_independent_deep_copy() {
        let a = Arb::from_mid_rad(2.0, 0.25);
        let b = a.clone();
        assert!(a.same_ball(&b));
        drop(a);
        // b still valid after a is dropped.
        assert_eq!(b.midpoint(), 2.0);
    }

    #[test]
    fn div_of_small_integers_is_exact() {
        let q = Arb::from_f64(8.0).div(&Arb::from_f64(2.0), 53);
        assert_eq!(q.midpoint(), 4.0);
        assert_eq!(q.radius(53), 0.0);
    }

    #[test]
    fn div_propagates_numerator_uncertainty() {
        // (2 ± 0.1) / 2 = (1 ± 0.05).
        let q = Arb::from_mid_rad(2.0, 0.1).div(&Arb::from_f64(2.0), 53);
        assert_eq!(q.midpoint(), 1.0);
        assert!(q.radius(53) >= 0.05);
    }

    #[test]
    fn union_of_two_exact_points_is_the_interval() {
        let u = Arb::from_f64(1.0).union(&Arb::from_f64(5.0), 53);
        // Hull is [1, 5]: midpoint 3, radius 2.
        assert_eq!(u.midpoint(), 3.0);
        assert!(u.radius(53) >= 2.0);
        // Containment of both inputs is the load-bearing property.
        assert!(u.lower_endpoint(53) <= 1.0);
        assert!(u.upper_endpoint(53) >= 5.0);
    }

    #[test]
    fn union_with_self_is_idempotent() {
        let a = Arb::from_mid_rad(3.0, 0.1);
        let u = a.union(&a, 53);
        assert_eq!(u.midpoint(), 3.0);
        // Radius can be (very slightly) wider due to FLINT's
        // conservative rounding, but never narrower.
        assert!(u.radius(53) >= a.radius(53));
    }

    #[test]
    fn upper_endpoint_is_conservative() {
        // [3 ± 0.25]: upper = 3.25. Exact at prec=64.
        let b = Arb::from_mid_rad(3.0, 0.25);
        let u = b.upper_endpoint(64);
        assert!(u >= 3.25);
    }

    #[test]
    fn lower_endpoint_is_conservative() {
        // [3 ± 0.25]: lower = 2.75.
        let b = Arb::from_mid_rad(3.0, 0.25);
        let l = b.lower_endpoint(64);
        assert!(l <= 2.75);
    }

    #[test]
    fn endpoints_sandwich_true_value() {
        // Every value in [mid - rad, mid + rad] must lie in
        // [lower_endpoint, upper_endpoint] — the rigorous-readback
        // contract.
        let cases = [(0.0, 0.0), (1.0, 0.0), (-2.5, 0.1), (1e6, 1e-3)];
        for (m, r) in cases {
            let b = Arb::from_mid_rad(m, r);
            assert!(b.lower_endpoint(64) <= m - r);
            assert!(b.upper_endpoint(64) >= m + r);
        }
    }

    #[test]
    fn same_ball_distinguishes_radius() {
        let exact = Arb::from_f64(1.0);
        let fuzzy = Arb::from_mid_rad(1.0, 0.1);
        assert!(!exact.same_ball(&fuzzy));
        assert!(exact.same_ball(&Arb::from_f64(1.0)));
    }
}
