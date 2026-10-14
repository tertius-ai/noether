// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Linkage and ABI smoke tests against the selected FLINT build.

use flint_sys::*;
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;

/// Helper: build a canonical `fmpq_t` representing `p/q` and return
/// it on the stack as a ready-to-use `fmpq`. Uses the unsigned-`q`
/// signature of `fmpq_set_si`.
unsafe fn make_fmpq(p: slong, q: ulong) -> fmpq {
    let mut x = MaybeUninit::<fmpq>::uninit();
    fmpq_init(x.as_mut_ptr());
    fmpq_set_si(x.as_mut_ptr(), p, q);
    x.assume_init()
}

unsafe fn clear_fmpq(x: &mut fmpq) {
    fmpq_clear(x as *mut fmpq);
}

#[test]
fn padic_matrix_lifecycle_entries_and_multiplication() {
    unsafe {
        let mut prime = MaybeUninit::<fmpz>::uninit();
        fmpz_init(prime.as_mut_ptr());
        fmpz_set_ui(prime.as_mut_ptr(), 5);
        let mut prime = prime.assume_init();

        let mut context = MaybeUninit::<padic_ctx_struct>::uninit();
        padic_ctx_init(
            context.as_mut_ptr(),
            &prime,
            0,
            8,
            padic_print_mode_PADIC_TERSE,
        );
        let mut context = context.assume_init();

        let mut matrix = MaybeUninit::<padic_mat_struct>::uninit();
        padic_mat_init2(matrix.as_mut_ptr(), 2, 2, 8);
        let mut matrix = matrix.assume_init();
        padic_mat_one(&mut matrix);

        let mut three = MaybeUninit::<padic_struct>::uninit();
        padic_init2(three.as_mut_ptr(), 8);
        padic_set_ui(three.as_mut_ptr(), 3, &context);
        let mut three = three.assume_init();
        padic_mat_set_entry_padic(&mut matrix, 0, 1, &three, &context);

        let mut identity = MaybeUninit::<padic_mat_struct>::uninit();
        padic_mat_init2(identity.as_mut_ptr(), 2, 2, 8);
        let mut identity = identity.assume_init();
        padic_mat_one(&mut identity);

        let mut product = MaybeUninit::<padic_mat_struct>::uninit();
        padic_mat_init2(product.as_mut_ptr(), 2, 2, 8);
        let mut product = product.assume_init();
        padic_mat_mul(&mut product, &matrix, &identity, &context);

        let mut entry = MaybeUninit::<padic_struct>::uninit();
        padic_init2(entry.as_mut_ptr(), 8);
        let mut entry = entry.assume_init();
        padic_mat_get_entry_padic(&mut entry, &product, 0, 1, &context);
        assert_eq!(padic_equal(&entry, &three), 1);

        padic_clear(&mut entry);
        padic_mat_clear(&mut product);
        padic_mat_clear(&mut identity);
        padic_clear(&mut three);
        padic_mat_clear(&mut matrix);
        padic_ctx_clear(&mut context);
        fmpz_clear(&mut prime);
    }
}

#[test]
fn fmpq_lifecycle_and_identity() {
    unsafe {
        let mut z = MaybeUninit::<fmpq>::uninit();
        fmpq_init(z.as_mut_ptr());
        fmpq_zero(z.as_mut_ptr());
        assert_eq!(fmpq_is_zero(z.as_ptr()), 1);
        assert_eq!(fmpq_is_one(z.as_ptr()), 0);
        let mut z = z.assume_init();

        let mut o = MaybeUninit::<fmpq>::uninit();
        fmpq_init(o.as_mut_ptr());
        fmpq_one(o.as_mut_ptr());
        assert_eq!(fmpq_is_one(o.as_ptr()), 1);
        assert_eq!(fmpq_is_zero(o.as_ptr()), 0);
        let mut o = o.assume_init();

        clear_fmpq(&mut z);
        clear_fmpq(&mut o);
    }
}

#[test]
fn fmpq_add_sub_mul_div() {
    unsafe {
        // x = 2/3, y = 1/4, expected x + y = 11/12
        let x = make_fmpq(2, 3);
        let y = make_fmpq(1, 4);
        let mut res = MaybeUninit::<fmpq>::uninit();
        fmpq_init(res.as_mut_ptr());
        fmpq_add(res.as_mut_ptr(), &x as *const _, &y as *const _);
        let mut res = res.assume_init();

        let expected = make_fmpq(11, 12);
        assert_eq!(fmpq_equal(&res as *const _, &expected as *const _), 1);

        // x - y = 5/12
        fmpq_sub(&mut res as *mut _, &x as *const _, &y as *const _);
        let s = make_fmpq(5, 12);
        assert_eq!(fmpq_equal(&res as *const _, &s as *const _), 1);

        // x * y = 1/6
        fmpq_mul(&mut res as *mut _, &x as *const _, &y as *const _);
        let p = make_fmpq(1, 6);
        assert_eq!(fmpq_equal(&res as *const _, &p as *const _), 1);

        // x / y = 8/3
        fmpq_div(&mut res as *mut _, &x as *const _, &y as *const _);
        let q = make_fmpq(8, 3);
        assert_eq!(fmpq_equal(&res as *const _, &q as *const _), 1);

        for mut v in [x, y, res, expected, s, p, q] {
            clear_fmpq(&mut v);
        }
    }
}

#[test]
fn fmpq_neg_and_sgn() {
    unsafe {
        let x = make_fmpq(-3, 5);
        assert_eq!(fmpq_sgn(&x as *const _), -1);
        let mut res = MaybeUninit::<fmpq>::uninit();
        fmpq_init(res.as_mut_ptr());
        fmpq_neg(res.as_mut_ptr(), &x as *const _);
        let res = res.assume_init();
        assert_eq!(fmpq_sgn(&res as *const _), 1);
        let expected = make_fmpq(3, 5);
        assert_eq!(fmpq_equal(&res as *const _, &expected as *const _), 1);
        for mut v in [x, res, expected] {
            clear_fmpq(&mut v);
        }
    }
}

#[test]
fn fmpq_inv_roundtrip() {
    unsafe {
        let x = make_fmpq(7, 11);
        let mut inv = MaybeUninit::<fmpq>::uninit();
        fmpq_init(inv.as_mut_ptr());
        fmpq_inv(inv.as_mut_ptr(), &x as *const _);
        let inv = inv.assume_init();
        let expected = make_fmpq(11, 7);
        assert_eq!(fmpq_equal(&inv as *const _, &expected as *const _), 1);

        // x * inv(x) == 1
        let mut prod = MaybeUninit::<fmpq>::uninit();
        fmpq_init(prod.as_mut_ptr());
        fmpq_mul(prod.as_mut_ptr(), &x as *const _, &inv as *const _);
        let prod = prod.assume_init();
        assert_eq!(fmpq_is_one(&prod as *const _), 1);

        for mut v in [x, inv, expected, prod] {
            clear_fmpq(&mut v);
        }
    }
}

#[test]
fn fmpq_cmp_ordering() {
    unsafe {
        let a = make_fmpq(1, 2); // 0.5
        let b = make_fmpq(2, 3); // 0.666...
        assert!(fmpq_cmp(&a as *const _, &b as *const _) < 0);
        assert!(fmpq_cmp(&b as *const _, &a as *const _) > 0);
        assert_eq!(fmpq_cmp(&a as *const _, &a as *const _), 0);
        for mut v in [a, b] {
            clear_fmpq(&mut v);
        }
    }
}

#[test]
fn fmpq_str_roundtrip() {
    unsafe {
        let x = make_fmpq(-17, 23);
        let s = fmpq_get_str(std::ptr::null_mut(), 10, &x as *const _);
        assert!(!s.is_null());
        let rust_str = CStr::from_ptr(s).to_string_lossy().into_owned();
        flint_free(s as *mut std::ffi::c_void);
        assert_eq!(rust_str, "-17/23");

        // Parse it back.
        let c = CString::new(rust_str).unwrap();
        let mut parsed = MaybeUninit::<fmpq>::uninit();
        fmpq_init(parsed.as_mut_ptr());
        let rc = fmpq_set_str(parsed.as_mut_ptr(), c.as_ptr(), 10);
        assert_eq!(rc, 0, "fmpq_set_str failed");
        let parsed = parsed.assume_init();
        assert_eq!(fmpq_equal(&parsed as *const _, &x as *const _), 1);

        for mut v in [x, parsed] {
            clear_fmpq(&mut v);
        }
    }
}

#[test]
fn fmpq_set_fmpz_frac_and_canonicalise() {
    unsafe {
        // Build 4/6 from fmpz components, then canonicalise to 2/3.
        let mut num = MaybeUninit::<fmpz>::uninit();
        let mut den = MaybeUninit::<fmpz>::uninit();
        fmpz_init(num.as_mut_ptr());
        fmpz_init(den.as_mut_ptr());
        fmpz_set_si(num.as_mut_ptr(), 4);
        fmpz_set_si(den.as_mut_ptr(), 6);
        let mut num = num.assume_init();
        let mut den = den.assume_init();

        let mut x = MaybeUninit::<fmpq>::uninit();
        fmpq_init(x.as_mut_ptr());
        fmpq_set_fmpz_frac(x.as_mut_ptr(), &num as *const _, &den as *const _);
        fmpq_canonicalise(x.as_mut_ptr());
        let x = x.assume_init();

        let expected = make_fmpq(2, 3);
        assert_eq!(fmpq_equal(&x as *const _, &expected as *const _), 1);

        fmpz_clear(&mut num as *mut fmpz);
        fmpz_clear(&mut den as *mut fmpz);
        let mut x = x;
        clear_fmpq(&mut x);
        let mut e = expected;
        clear_fmpq(&mut e);
    }
}

// ── Scalar nmod linkage tests ────────────────────────────────

#[test]
fn n_preinvert_limb_roundtrips_with_mulmod() {
    // 7 · 8 mod 15 = 11 using n_mulmod2_preinv.
    unsafe {
        let n: u64 = 15;
        let ninv = n_preinvert_limb(n);
        let r = n_mulmod2_preinv(7, 8, n, ninv);
        assert_eq!(r, 11);
    }
}

#[test]
fn n_mulmod2_without_preinv_matches() {
    // The non-preinv variant computes ninv internally; verify
    // both give the same result on a handful of values.
    unsafe {
        let n: u64 = 97;
        let ninv = n_preinvert_limb(n);
        for a in [0u64, 1, 5, 42, 96] {
            for b in [0u64, 1, 11, 73, 96] {
                let r1 = n_mulmod2(a, b, n);
                let r2 = n_mulmod2_preinv(a, b, n, ninv);
                assert_eq!(r1, r2, "a={} b={} n={}", a, b, n);
                // Cross-check against u128 reference.
                let expected = ((a as u128) * (b as u128) % (n as u128)) as u64;
                assert_eq!(r1, expected);
            }
        }
    }
}

#[test]
fn n_gcdinv_as_inverse() {
    // n_gcdinv(out, x, y) returns g = gcd(x, y) and writes an `out`
    // such that out * x ≡ g (mod y). When g == 1, out is the
    // modular inverse; when g > 1, x is not invertible. Use this
    // instead of n_invmod (which aborts the process on non-unit
    // input).
    unsafe {
        let mut inv: u64 = 0;
        // 3 · 5 ≡ 1 (mod 7), so 3^{-1} mod 7 = 5.
        let g = n_gcdinv(&mut inv, 3, 7);
        assert_eq!(g, 1);
        assert_eq!(inv, 5);

        let g = n_gcdinv(&mut inv, 5, 7);
        assert_eq!(g, 1);
        assert_eq!(inv, 3);

        // 1 is self-inverse.
        let g = n_gcdinv(&mut inv, 1, 7);
        assert_eq!(g, 1);
        assert_eq!(inv, 1);

        // gcd(4, 6) = 2, so 4 is not invertible mod 6.
        let g = n_gcdinv(&mut inv, 4, 6);
        assert_eq!(g, 2);
        // inv is unspecified when g > 1; we do not assert its value.

        // gcd(2, 4) = 2, not invertible.
        let g = n_gcdinv(&mut inv, 2, 4);
        assert_eq!(g, 2);
    }
}

#[test]
fn n_powmod2_ui_preinv_basic() {
    unsafe {
        let n: u64 = 1_000_003; // a prime
        let ninv = n_preinvert_limb(n);
        // Fermat's little theorem: a^(n-1) ≡ 1 (mod n) for a coprime to n.
        for a in [2u64, 3, 7, 999_999] {
            let r = n_powmod2_ui_preinv(a, n - 1, n, ninv);
            assert_eq!(r, 1, "Fermat failed for a={}", a);
        }
        // 2^10 = 1024, trivially.
        let small_n: u64 = 10_000;
        let small_ninv = n_preinvert_limb(small_n);
        assert_eq!(n_powmod2_ui_preinv(2, 10, small_n, small_ninv), 1024);
    }
}

// ── fmpz gap-fill linkage tests ──────────────────────────────

/// Helper: initialize an fmpz and set it from a decimal string.
unsafe fn make_fmpz_from_str(s: &str) -> fmpz {
    let mut z = MaybeUninit::<fmpz>::uninit();
    fmpz_init(z.as_mut_ptr());
    let c = CString::new(s).unwrap();
    let rc = fmpz_set_str(z.as_mut_ptr(), c.as_ptr(), 10);
    assert_eq!(rc, 0, "fmpz_set_str failed for {:?}", s);
    z.assume_init()
}

unsafe fn clear_fmpz(z: &mut fmpz) {
    fmpz_clear(z as *mut fmpz);
}

unsafe fn fmpz_to_string_10(f: *const fmpz) -> String {
    let ptr = fmpz_get_str(std::ptr::null_mut(), 10, f);
    assert!(!ptr.is_null());
    let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    flint_free(ptr as *mut std::ffi::c_void);
    s
}

#[test]
fn fmpz_sub_neg_cmp_equal() {
    unsafe {
        let a = make_fmpz_from_str("100");
        let b = make_fmpz_from_str("37");
        let mut diff = MaybeUninit::<fmpz>::uninit();
        fmpz_init(diff.as_mut_ptr());
        fmpz_sub(diff.as_mut_ptr(), &a, &b);
        let diff = diff.assume_init();
        assert_eq!(fmpz_to_string_10(&diff), "63");

        let mut neg = MaybeUninit::<fmpz>::uninit();
        fmpz_init(neg.as_mut_ptr());
        fmpz_neg(neg.as_mut_ptr(), &a);
        let neg = neg.assume_init();
        assert_eq!(fmpz_to_string_10(&neg), "-100");

        assert!(fmpz_cmp(&a, &b) > 0);
        assert!(fmpz_cmp(&b, &a) < 0);
        assert_eq!(fmpz_cmp(&a, &a), 0);
        assert_eq!(fmpz_equal(&a, &a), 1);
        assert_eq!(fmpz_equal(&a, &b), 0);

        for mut v in [a, b, diff, neg] {
            clear_fmpz(&mut v);
        }
    }
}

#[test]
fn fmpz_gcd_and_mod() {
    unsafe {
        // gcd(42, 56) = 14
        let a = make_fmpz_from_str("42");
        let b = make_fmpz_from_str("56");
        let mut g = MaybeUninit::<fmpz>::uninit();
        fmpz_init(g.as_mut_ptr());
        fmpz_gcd(g.as_mut_ptr(), &a, &b);
        let g = g.assume_init();
        assert_eq!(fmpz_to_string_10(&g), "14");

        // 100 mod 17 = 15 (Euclidean)
        let hundred = make_fmpz_from_str("100");
        let seventeen = make_fmpz_from_str("17");
        let mut r = MaybeUninit::<fmpz>::uninit();
        fmpz_init(r.as_mut_ptr());
        fmpz_mod(r.as_mut_ptr(), &hundred, &seventeen);
        let r = r.assume_init();
        assert_eq!(fmpz_to_string_10(&r), "15");

        // -100 mod 17: Euclidean remainder is in [0, 17), so = 2.
        let neg_hundred = make_fmpz_from_str("-100");
        let mut r2 = MaybeUninit::<fmpz>::uninit();
        fmpz_init(r2.as_mut_ptr());
        fmpz_mod(r2.as_mut_ptr(), &neg_hundred, &seventeen);
        let r2 = r2.assume_init();
        assert_eq!(fmpz_to_string_10(&r2), "2");

        for mut v in [a, b, g, hundred, seventeen, r, neg_hundred, r2] {
            clear_fmpz(&mut v);
        }
    }
}

#[test]
fn fmpz_is_one_and_set_ui() {
    unsafe {
        let mut z = MaybeUninit::<fmpz>::uninit();
        fmpz_init(z.as_mut_ptr());
        fmpz_set_ui(z.as_mut_ptr(), 1);
        let mut z = z.assume_init();
        assert_eq!(fmpz_is_one(&z), 1);
        assert_eq!(fmpz_is_zero(&z), 0);

        fmpz_set_ui(&mut z as *mut fmpz, 0);
        assert_eq!(fmpz_is_one(&z), 0);
        assert_eq!(fmpz_is_zero(&z), 1);

        fmpz_set_ui(&mut z as *mut fmpz, 42);
        assert_eq!(fmpz_is_one(&z), 0);
        assert_eq!(fmpz_is_zero(&z), 0);
        assert_eq!(fmpz_to_string_10(&z), "42");

        clear_fmpz(&mut z);
    }
}

#[test]
fn fmpz_divexact_basic() {
    unsafe {
        // 100 / 5 = 20 (exact).
        let a = make_fmpz_from_str("100");
        let b = make_fmpz_from_str("5");
        let mut q = MaybeUninit::<fmpz>::uninit();
        fmpz_init(q.as_mut_ptr());
        fmpz_divexact(q.as_mut_ptr(), &a, &b);
        let q = q.assume_init();
        assert_eq!(fmpz_to_string_10(&q), "20");

        // Large: 10^20 / 10^10 = 10^10.
        let big = make_fmpz_from_str("100000000000000000000");
        let ten_ten = make_fmpz_from_str("10000000000");
        let mut q2 = MaybeUninit::<fmpz>::uninit();
        fmpz_init(q2.as_mut_ptr());
        fmpz_divexact(q2.as_mut_ptr(), &big, &ten_ten);
        let q2 = q2.assume_init();
        assert_eq!(fmpz_to_string_10(&q2), "10000000000");

        for mut v in [a, b, q, big, ten_ten, q2] {
            clear_fmpz(&mut v);
        }
    }
}

// ── fmpz_mod_ctx linkage tests ───────────────────────────────

/// Helper: initialise an fmpz_mod_ctx around modulus n.
///
/// The caller must pair this with ``fmpz_mod_ctx_clear`` before the
/// ``fmpz_mod_ctx_struct`` goes out of scope. We use ``std::mem::zeroed``
/// because FLINT's ``fmpz_mod_ctx_init`` overwrites every byte it cares
/// about from the caller's modulus; a zero buffer is a valid starting
/// point and matches what an OS-cleared stack slot would look like.
unsafe fn init_ctx(n: *const fmpz) -> fmpz_mod_ctx_struct {
    let mut ctx: fmpz_mod_ctx_struct = std::mem::zeroed();
    fmpz_mod_ctx_init(&mut ctx as *mut _, n);
    ctx
}

#[test]
fn fmpz_mod_ctx_add_mul_neg() {
    unsafe {
        let n = make_fmpz_from_str("7");
        let mut ctx = init_ctx(&n);

        let three = make_fmpz_from_str("3");
        let five = make_fmpz_from_str("5");

        // 3 + 5 ≡ 1 (mod 7)
        let mut r = MaybeUninit::<fmpz>::uninit();
        fmpz_init(r.as_mut_ptr());
        fmpz_mod_add(r.as_mut_ptr(), &three, &five, &ctx);
        let r = r.assume_init();
        assert_eq!(fmpz_to_string_10(&r), "1");

        // 3 * 5 ≡ 1 (mod 7)
        let mut p = MaybeUninit::<fmpz>::uninit();
        fmpz_init(p.as_mut_ptr());
        fmpz_mod_mul(p.as_mut_ptr(), &three, &five, &ctx);
        let p = p.assume_init();
        assert_eq!(fmpz_to_string_10(&p), "1");

        // -3 ≡ 4 (mod 7)
        let mut ng = MaybeUninit::<fmpz>::uninit();
        fmpz_init(ng.as_mut_ptr());
        fmpz_mod_neg(ng.as_mut_ptr(), &three, &ctx);
        let ng = ng.assume_init();
        assert_eq!(fmpz_to_string_10(&ng), "4");

        // 5 - 3 ≡ 2 (mod 7)
        let mut d = MaybeUninit::<fmpz>::uninit();
        fmpz_init(d.as_mut_ptr());
        fmpz_mod_sub(d.as_mut_ptr(), &five, &three, &ctx);
        let d = d.assume_init();
        assert_eq!(fmpz_to_string_10(&d), "2");

        fmpz_mod_ctx_clear(&mut ctx as *mut _);
        for mut v in [n, three, five, r, p, ng, d] {
            clear_fmpz(&mut v);
        }
    }
}

#[test]
fn fmpz_mod_ctx_set_fmpz_reduces() {
    unsafe {
        // Reduce 100 mod 17 via the ctx-aware path.
        let n = make_fmpz_from_str("17");
        let mut ctx = init_ctx(&n);

        let big = make_fmpz_from_str("100");
        let mut r = MaybeUninit::<fmpz>::uninit();
        fmpz_init(r.as_mut_ptr());
        fmpz_mod_set_fmpz(r.as_mut_ptr(), &big, &ctx);
        let r = r.assume_init();
        assert_eq!(fmpz_to_string_10(&r), "15");

        fmpz_mod_ctx_clear(&mut ctx as *mut _);
        for mut v in [n, big, r] {
            clear_fmpz(&mut v);
        }
    }
}

#[test]
fn fmpz_mod_mat_arithmetic_and_rref_linkage() {
    unsafe {
        // Use a modulus larger than one machine word so this exercises the
        // fmpz_mod_mat family rather than merely duplicating nmod_mat.
        let modulus = make_fmpz_from_str("18446744073709551629");
        let mut ctx = init_ctx(&modulus);
        let one = make_fmpz_from_str("1");
        let two = make_fmpz_from_str("2");
        let three = make_fmpz_from_str("3");

        let mut a = MaybeUninit::<fmpz_mod_mat_struct>::uninit();
        let mut b = MaybeUninit::<fmpz_mod_mat_struct>::uninit();
        let mut product = MaybeUninit::<fmpz_mod_mat_struct>::uninit();
        let mut reduced = MaybeUninit::<fmpz_mod_mat_struct>::uninit();
        fmpz_mod_mat_init(a.as_mut_ptr(), 2, 2, &ctx);
        fmpz_mod_mat_init(b.as_mut_ptr(), 2, 2, &ctx);
        fmpz_mod_mat_init(product.as_mut_ptr(), 2, 2, &ctx);
        fmpz_mod_mat_init(reduced.as_mut_ptr(), 2, 2, &ctx);
        let mut a = a.assume_init();
        let mut b = b.assume_init();
        let mut product = product.assume_init();
        let mut reduced = reduced.assume_init();

        fmpz_mod_mat_set_entry(&mut a, 0, 0, &one, &ctx);
        fmpz_mod_mat_set_entry(&mut a, 0, 1, &two, &ctx);
        fmpz_mod_mat_set_entry(&mut a, 1, 0, &three, &ctx);
        fmpz_mod_mat_set_entry(&mut a, 1, 1, &one, &ctx);
        fmpz_mod_mat_set(&mut b, &a, &ctx);
        fmpz_mod_mat_mul(&mut product, &a, &b, &ctx);

        let mut entry = MaybeUninit::<fmpz>::uninit();
        fmpz_init(entry.as_mut_ptr());
        let mut entry = entry.assume_init();
        fmpz_mod_mat_get_entry(&mut entry, &product, 0, 0, &ctx);
        assert_eq!(fmpz_to_string_10(&entry), "7");
        assert_eq!(fmpz_mod_mat_rref(&mut reduced, &a, &ctx), 2);

        fmpz_mod_mat_clear(&mut reduced, &ctx);
        fmpz_mod_mat_clear(&mut product, &ctx);
        fmpz_mod_mat_clear(&mut b, &ctx);
        fmpz_mod_mat_clear(&mut a, &ctx);
        fmpz_mod_ctx_clear(&mut ctx);
        for mut value in [modulus, one, two, three, entry] {
            clear_fmpz(&mut value);
        }
    }
}

#[test]
fn fmpz_mod_ctx_bignum_modulus() {
    unsafe {
        // M_127 = 2^127 - 1, a Mersenne prime. Verify basic arithmetic
        // works at a bignum modulus through the ctx path.
        let n = make_fmpz_from_str("170141183460469231731687303715884105727");
        let mut ctx = init_ctx(&n);

        let two = make_fmpz_from_str("2");
        let three = make_fmpz_from_str("3");

        // 2 * 3 = 6
        let mut r = MaybeUninit::<fmpz>::uninit();
        fmpz_init(r.as_mut_ptr());
        fmpz_mod_mul(r.as_mut_ptr(), &two, &three, &ctx);
        let r = r.assume_init();
        assert_eq!(fmpz_to_string_10(&r), "6");

        // (n - 1) + 1 ≡ 0 (mod n)
        let n_minus_1 = make_fmpz_from_str("170141183460469231731687303715884105726");
        let one = make_fmpz_from_str("1");
        let mut s = MaybeUninit::<fmpz>::uninit();
        fmpz_init(s.as_mut_ptr());
        fmpz_mod_add(s.as_mut_ptr(), &n_minus_1, &one, &ctx);
        let s = s.assume_init();
        assert_eq!(fmpz_to_string_10(&s), "0");

        fmpz_mod_ctx_clear(&mut ctx as *mut _);
        for mut v in [n, two, three, r, n_minus_1, one, s] {
            clear_fmpz(&mut v);
        }
    }
}

// ── arb / acb linkage tests ──────────────────────────────────

/// Helper: initialise an `arb_struct` (stack-allocated, opaque 48
/// bytes). The caller must pair this with `arb_clear` before the
/// value goes out of scope.
unsafe fn init_arb() -> arb_struct {
    let mut x: arb_struct = std::mem::zeroed();
    arb_init(&mut x as *mut _);
    x
}

unsafe fn clear_arb(x: &mut arb_struct) {
    arb_clear(x as *mut _);
}

unsafe fn arb_to_string_default(x: *const arb_struct) -> String {
    let ptr = arb_get_str(x, 15, 0);
    assert!(!ptr.is_null());
    let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    flint_free(ptr as *mut std::ffi::c_void);
    s
}

#[test]
fn arb_struct_size_asserts() {
    // The const _ blocks at module scope already enforce these,
    // but we mirror them in a test for visibility.
    assert_eq!(std::mem::size_of::<arb_struct>(), 48);
    assert_eq!(std::mem::size_of::<acb_struct>(), 96);
    // acb_struct field offsets: real at 0, imag at 48.
    let acb: acb_struct = unsafe { std::mem::zeroed() };
    let base = &acb as *const _ as usize;
    let real = &acb.real as *const _ as usize;
    let imag = &acb.imag as *const _ as usize;
    assert_eq!(real - base, 0);
    assert_eq!(imag - base, 48);
}

#[test]
fn arb_init_zero_and_set_si() {
    unsafe {
        let mut x = init_arb();
        assert_eq!(arb_is_zero(&x), 1);
        assert_eq!(arb_is_one(&x), 0);
        arb_set_si(&mut x as *mut _, 42);
        assert_eq!(arb_is_zero(&x), 0);
        let s = arb_to_string_default(&x);
        assert!(s.starts_with("42"), "expected '42…', got {}", s);
        clear_arb(&mut x);
    }
}

#[test]
fn arb_add_mul_sqrt() {
    unsafe {
        let mut a = init_arb();
        let mut b = init_arb();
        arb_set_si(&mut a as *mut _, 4);
        arb_set_si(&mut b as *mut _, 3);

        // 4 + 3 = 7 at 64 bits precision.
        let mut sum = init_arb();
        arb_add(&mut sum as *mut _, &a, &b, 64);
        let s = arb_to_string_default(&sum);
        assert!(s.starts_with("7"), "expected 7, got {}", s);

        // 4 * 3 = 12.
        let mut prod = init_arb();
        arb_mul(&mut prod as *mut _, &a, &b, 64);
        let s = arb_to_string_default(&prod);
        assert!(s.starts_with("12"), "expected 12, got {}", s);

        // sqrt(4) = 2 (exact ball since 4 is a perfect square).
        let mut root = init_arb();
        arb_sqrt(&mut root as *mut _, &a, 200);
        let s = arb_to_string_default(&root);
        assert!(s.starts_with("2"), "expected 2, got {}", s);

        for mut v in [a, b, sum, prod, root] {
            clear_arb(&mut v);
        }
    }
}

#[test]
fn arb_pi_at_200_bits_starts_with_known_digits() {
    unsafe {
        let mut pi = init_arb();
        arb_const_pi(&mut pi as *mut _, 200);
        let s = arb_to_string_default(&pi);
        // Arb's default format wraps the value in "[midpoint +/- radius]",
        // so the digits of π appear starting at index 1.
        assert!(
            s.contains("3.141592653"),
            "expected string to contain 3.141592653, got {}",
            s
        );
        clear_arb(&mut pi);
    }
}

#[test]
fn arb_contains_zero_and_overlaps() {
    unsafe {
        let mut zero = init_arb();
        arb_zero(&mut zero as *mut _);
        assert_eq!(arb_contains_zero(&zero), 1);

        let mut one = init_arb();
        arb_one(&mut one as *mut _);
        assert_eq!(arb_contains_zero(&one), 0);

        // zero and zero overlap; zero and one don't.
        let mut zero2 = init_arb();
        arb_zero(&mut zero2 as *mut _);
        assert_eq!(arb_overlaps(&zero, &zero2), 1);
        assert_eq!(arb_overlaps(&zero, &one), 0);

        for mut v in [zero, one, zero2] {
            clear_arb(&mut v);
        }
    }
}

unsafe fn init_acb() -> acb_struct {
    let mut x: acb_struct = std::mem::zeroed();
    acb_init(&mut x as *mut _);
    x
}

unsafe fn clear_acb(x: &mut acb_struct) {
    acb_clear(x as *mut _);
}

#[test]
fn acb_set_arb_arb_and_arithmetic() {
    unsafe {
        // Build a = 3 + 4i, b = 1 - 2i. Verify a * b = 11 - 2i.
        // (3 + 4i)(1 - 2i) = 3 - 6i + 4i - 8i² = 3 - 2i + 8 = 11 - 2i.
        let mut re_a = init_arb();
        let mut im_a = init_arb();
        arb_set_si(&mut re_a as *mut _, 3);
        arb_set_si(&mut im_a as *mut _, 4);
        let mut a = init_acb();
        acb_set_arb_arb(&mut a as *mut _, &re_a, &im_a);

        let mut re_b = init_arb();
        let mut im_b = init_arb();
        arb_set_si(&mut re_b as *mut _, 1);
        arb_set_si(&mut im_b as *mut _, -2);
        let mut b = init_acb();
        acb_set_arb_arb(&mut b as *mut _, &re_b, &im_b);

        let mut prod = init_acb();
        acb_mul(&mut prod as *mut _, &a, &b, 64);

        // Extract real and imaginary parts of prod. Since acb_struct
        // has concrete real / imag fields, we can take refs.
        let real_s = arb_to_string_default(&prod.real);
        let imag_s = arb_to_string_default(&prod.imag);
        assert!(real_s.starts_with("11"), "expected real=11, got {}", real_s);
        assert!(imag_s.starts_with("-2"), "expected imag=-2, got {}", imag_s);

        // |3 + 4i| = 5 (exact because 3² + 4² = 25, sqrt(25) = 5).
        let mut absval = init_arb();
        acb_abs(&mut absval as *mut _, &a, 200);
        let abs_s = arb_to_string_default(&absval);
        assert!(abs_s.starts_with("5"), "expected |3+4i|=5, got {}", abs_s);

        // Conjugate of 3 + 4i is 3 - 4i.
        let mut conj = init_acb();
        acb_conj(&mut conj as *mut _, &a);
        let conj_re_s = arb_to_string_default(&conj.real);
        let conj_im_s = arb_to_string_default(&conj.imag);
        assert!(conj_re_s.starts_with("3"));
        assert!(conj_im_s.starts_with("-4"));

        for mut v in [re_a, im_a, re_b, im_b, absval] {
            clear_arb(&mut v);
        }
        for mut v in [a, b, prod, conj] {
            clear_acb(&mut v);
        }
    }
}

// ── Calcium (ca) linkage tests ───────────────────────────────

#[test]
fn ca_struct_size_asserts() {
    // These are the layouts reported by the pinned FLINT 3.6 headers.
    assert_eq!(std::mem::size_of::<ca_struct>(), 40);
    assert_eq!(std::mem::size_of::<ca_ctx_struct>(), 120);
    assert_eq!(std::mem::align_of::<ca_struct>(), 8);
    assert_eq!(std::mem::align_of::<ca_ctx_struct>(), 8);
}

#[test]
fn ca_truth_t_values_match_flint() {
    // T_TRUE=0, T_FALSE=1, T_UNKNOWN=2.
    assert_eq!(truth_t_T_TRUE as i32, 0);
    assert_eq!(truth_t_T_FALSE as i32, 1);
    assert_eq!(truth_t_T_UNKNOWN as i32, 2);
    assert_eq!(std::mem::size_of::<truth_t>(), 4);
}

unsafe fn init_ca_ctx() -> ca_ctx_struct {
    let mut ctx: ca_ctx_struct = std::mem::zeroed();
    ca_ctx_init(&mut ctx as *mut _);
    ctx
}

unsafe fn init_ca(ctx: *mut ca_ctx_struct) -> ca_struct {
    let mut x: ca_struct = std::mem::zeroed();
    ca_init(&mut x as *mut _, ctx);
    x
}

unsafe fn ca_to_string(x: *const ca_struct, ctx: *mut ca_ctx_struct) -> String {
    let ptr = ca_get_str(x, ctx);
    assert!(!ptr.is_null());
    let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    flint_free(ptr as *mut std::ffi::c_void);
    s
}

#[test]
fn ca_ctx_lifecycle_and_set_si() {
    unsafe {
        let mut ctx = init_ca_ctx();
        let mut x = init_ca(&mut ctx as *mut _);
        ca_set_si(&mut x as *mut _, 42, &mut ctx as *mut _);
        assert_eq!(ca_check_is_zero(&x, &mut ctx as *mut _), truth_t_T_FALSE);
        assert_eq!(ca_check_is_integer(&x, &mut ctx as *mut _), truth_t_T_TRUE);
        let s = ca_to_string(&x, &mut ctx as *mut _);
        assert!(s.contains("42"), "expected '42' in {}", s);

        ca_clear(&mut x as *mut _, &mut ctx as *mut _);
        ca_ctx_clear(&mut ctx as *mut _);
    }
}

#[test]
fn ca_add_mul_exact_equality() {
    unsafe {
        let mut ctx = init_ca_ctx();
        let mut a = init_ca(&mut ctx as *mut _);
        let mut b = init_ca(&mut ctx as *mut _);
        let mut c = init_ca(&mut ctx as *mut _);
        ca_set_si(&mut a as *mut _, 6, &mut ctx as *mut _);
        ca_set_si(&mut b as *mut _, 7, &mut ctx as *mut _);

        // 6 + 7 = 13
        ca_add(
            &mut c as *mut _,
            &a as *const _,
            &b as *const _,
            &mut ctx as *mut _,
        );
        let mut thirteen = init_ca(&mut ctx as *mut _);
        ca_set_si(&mut thirteen as *mut _, 13, &mut ctx as *mut _);
        assert_eq!(
            ca_check_equal(&c, &thirteen, &mut ctx as *mut _),
            truth_t_T_TRUE
        );

        // 6 * 7 = 42
        ca_mul(
            &mut c as *mut _,
            &a as *const _,
            &b as *const _,
            &mut ctx as *mut _,
        );
        let mut forty_two = init_ca(&mut ctx as *mut _);
        ca_set_si(&mut forty_two as *mut _, 42, &mut ctx as *mut _);
        assert_eq!(
            ca_check_equal(&c, &forty_two, &mut ctx as *mut _),
            truth_t_T_TRUE
        );

        for mut v in [a, b, c, thirteen, forty_two] {
            ca_clear(&mut v as *mut _, &mut ctx as *mut _);
        }
        ca_ctx_clear(&mut ctx as *mut _);
    }
}

#[test]
fn ca_sqrt_two_squared_is_two() {
    // The motivating Calcium test: (√2)² = 2 decided exactly,
    // not as a ball comparison. `ca_check_equal` must return
    // `T_TRUE` here or Calcium is not doing its job.
    unsafe {
        let mut ctx = init_ca_ctx();
        let mut two = init_ca(&mut ctx as *mut _);
        ca_set_si(&mut two as *mut _, 2, &mut ctx as *mut _);

        let mut root = init_ca(&mut ctx as *mut _);
        ca_sqrt(&mut root as *mut _, &two, &mut ctx as *mut _);

        let mut sq = init_ca(&mut ctx as *mut _);
        ca_mul(
            &mut sq as *mut _,
            &root as *const _,
            &root as *const _,
            &mut ctx as *mut _,
        );

        assert_eq!(
            ca_check_equal(&sq, &two, &mut ctx as *mut _),
            truth_t_T_TRUE,
            "Calcium failed to decide (√2)² = 2"
        );

        for mut v in [two, root, sq] {
            ca_clear(&mut v as *mut _, &mut ctx as *mut _);
        }
        ca_ctx_clear(&mut ctx as *mut _);
    }
}

#[test]
fn ca_eulers_identity_is_decided() {
    // e^{iπ} + 1 = 0. Calcium should decide this exactly via its
    // closed-form recognition of the exponential/π tower.
    unsafe {
        let mut ctx = init_ca_ctx();

        let mut pi_i = init_ca(&mut ctx as *mut _);
        ca_pi_i(&mut pi_i as *mut _, &mut ctx as *mut _);

        let mut e_pi_i = init_ca(&mut ctx as *mut _);
        ca_exp(&mut e_pi_i as *mut _, &pi_i, &mut ctx as *mut _);

        let mut one = init_ca(&mut ctx as *mut _);
        ca_one(&mut one as *mut _, &mut ctx as *mut _);

        let mut lhs = init_ca(&mut ctx as *mut _);
        ca_add(
            &mut lhs as *mut _,
            &e_pi_i as *const _,
            &one as *const _,
            &mut ctx as *mut _,
        );

        assert_eq!(
            ca_check_is_zero(&lhs, &mut ctx as *mut _),
            truth_t_T_TRUE,
            "Calcium failed to decide Euler's identity"
        );

        for mut v in [pi_i, e_pi_i, one, lhs] {
            ca_clear(&mut v as *mut _, &mut ctx as *mut _);
        }
        ca_ctx_clear(&mut ctx as *mut _);
    }
}

#[test]
fn ca_pi_is_real_and_not_rational() {
    unsafe {
        let mut ctx = init_ca_ctx();
        let mut pi = init_ca(&mut ctx as *mut _);
        ca_pi(&mut pi as *mut _, &mut ctx as *mut _);

        assert_eq!(ca_check_is_real(&pi, &mut ctx as *mut _), truth_t_T_TRUE);
        // π is not rational (Lambert, 1761). Calcium may report
        // False here (if it knows) or Unknown (if it does not).
        // We accept either as "not a positive rationality claim."
        let r = ca_check_is_rational(&pi, &mut ctx as *mut _);
        assert_ne!(r, truth_t_T_TRUE, "Calcium wrongly claimed π is rational");

        ca_clear(&mut pi as *mut _, &mut ctx as *mut _);
        ca_ctx_clear(&mut ctx as *mut _);
    }
}

#[test]
fn ca_set_fmpq_and_get_fmpq_roundtrip() {
    unsafe {
        let mut ctx = init_ca_ctx();
        let q = make_fmpq(22, 7);
        let mut x = init_ca(&mut ctx as *mut _);
        ca_set_fmpq(&mut x as *mut _, &q as *const _, &mut ctx as *mut _);

        assert_eq!(ca_check_is_rational(&x, &mut ctx as *mut _), truth_t_T_TRUE);

        let mut back = MaybeUninit::<fmpq>::uninit();
        fmpq_init(back.as_mut_ptr());
        let rc = ca_get_fmpq(back.as_mut_ptr(), &x, &mut ctx as *mut _);
        assert_eq!(rc, 1, "ca_get_fmpq failed for 22/7");
        let back = back.assume_init();
        assert_eq!(fmpq_equal(&back as *const _, &q as *const _), 1);

        ca_clear(&mut x as *mut _, &mut ctx as *mut _);
        ca_ctx_clear(&mut ctx as *mut _);
        let mut q = q;
        let mut back = back;
        clear_fmpq(&mut q);
        clear_fmpq(&mut back);
    }
}

// ─── fq_nmod linkage + smoke tests ────────────────────────────────

#[test]
fn fq_nmod_ctx_sizes_match_flint() {
    // Covered by the const assert in the main module, but re-check
    // here so a mismatch shows up in `cargo test` output.
    assert_eq!(std::mem::size_of::<fq_nmod_ctx_struct>(), 160);
    assert_eq!(std::mem::size_of::<fq_nmod_struct>(), 48);
    assert_eq!(std::mem::size_of::<fmpz_mod_poly_struct>(), 24);
    assert_eq!(std::mem::align_of::<fmpz_mod_poly_struct>(), 8);
    assert_eq!(std::mem::size_of::<fq_nmod_poly_struct>(), 24);
    assert_eq!(std::mem::align_of::<fq_nmod_poly_struct>(), 8);
}

#[test]
fn fq_nmod_construct_small_field() {
    // F_{25} = F_5[x] / (Conway polynomial)
    unsafe {
        let mut ctx: fq_nmod_ctx_t = [std::mem::zeroed()];
        let var = std::ffi::CString::new("a").unwrap();
        fq_nmod_ctx_init_conway_ui(ctx.as_mut_ptr(), 5, 2, var.as_ptr());
        assert_eq!(fq_nmod_ctx_degree(ctx.as_ptr()), 2);

        let mut order = MaybeUninit::<fmpz>::uninit();
        fmpz_init(order.as_mut_ptr());
        fq_nmod_ctx_order(order.as_mut_ptr(), ctx.as_ptr());
        let mut order = order.assume_init();
        assert_eq!(fmpz_get_si(&order as *const _), 25);
        fmpz_clear(&mut order as *mut _);

        fq_nmod_ctx_clear(ctx.as_mut_ptr());
    }
}

#[test]
fn fq_nmod_checked_conway_and_explicit_modulus_construction() {
    unsafe {
        let var = std::ffi::CString::new("a").unwrap();

        let mut conway_ctx: fq_nmod_ctx_t = [std::mem::zeroed()];
        assert_eq!(
            _fq_nmod_ctx_init_conway_ui(conway_ctx.as_mut_ptr(), 5, 2, var.as_ptr()),
            1
        );
        assert_eq!(fq_nmod_ctx_prime(conway_ctx.as_ptr()), 5);
        let conway_modulus = fq_nmod_ctx_modulus(conway_ctx.as_ptr());
        assert!(!conway_modulus.is_null());
        assert_eq!(nmod_poly_degree(conway_modulus), 2);
        fq_nmod_ctx_clear(conway_ctx.as_mut_ptr());

        let mut modulus: nmod_poly_t = [std::mem::zeroed()];
        nmod_poly_init(modulus.as_mut_ptr(), 5);
        // x^2 + 2 is irreducible over F_5.
        nmod_poly_set_coeff_ui(modulus.as_mut_ptr(), 0, 2);
        nmod_poly_set_coeff_ui(modulus.as_mut_ptr(), 2, 1);
        assert_eq!(nmod_poly_is_irreducible(modulus.as_ptr()), 1);

        let mut explicit_ctx: fq_nmod_ctx_t = [std::mem::zeroed()];
        fq_nmod_ctx_init_modulus(explicit_ctx.as_mut_ptr(), modulus.as_ptr(), var.as_ptr());
        assert_eq!(fq_nmod_ctx_prime(explicit_ctx.as_ptr()), 5);
        assert_eq!(fq_nmod_ctx_degree(explicit_ctx.as_ptr()), 2);
        fq_nmod_ctx_clear(explicit_ctx.as_mut_ptr());
        nmod_poly_clear(modulus.as_mut_ptr());
    }
}

#[test]
fn fq_nmod_arithmetic_mod_conway() {
    // In F_{25} with Conway poly x^2 + 4x + 2:
    // (x + 1) * (x + 1) = x^2 + 2x + 1
    // = (4x + 3) + 2x + 1 (substitute x^2 = -4x - 2 = x + 3)
    // = 3x + 4
    // Just verify the ring axioms numerically: compute a*a^{-1}=1
    // and (a+b)-b=a for a random-ish element.
    unsafe {
        let mut ctx: fq_nmod_ctx_t = [std::mem::zeroed()];
        let var = std::ffi::CString::new("a").unwrap();
        fq_nmod_ctx_init_conway_ui(ctx.as_mut_ptr(), 5, 2, var.as_ptr());

        let mut a: fq_nmod_t = [std::mem::zeroed()];
        let mut b: fq_nmod_t = [std::mem::zeroed()];
        let mut tmp: fq_nmod_t = [std::mem::zeroed()];
        let mut tmp2: fq_nmod_t = [std::mem::zeroed()];
        let mut one: fq_nmod_t = [std::mem::zeroed()];
        fq_nmod_init(a.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_init(b.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_init(tmp.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_init(tmp2.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_init(one.as_mut_ptr(), ctx.as_ptr());

        fq_nmod_set_ui(a.as_mut_ptr(), 7, ctx.as_ptr()); // a = 7 ≡ 2
        fq_nmod_set_ui(b.as_mut_ptr(), 3, ctx.as_ptr());
        fq_nmod_one(one.as_mut_ptr(), ctx.as_ptr());

        // (a + b) - b = a
        fq_nmod_add(tmp.as_mut_ptr(), a.as_ptr(), b.as_ptr(), ctx.as_ptr());
        fq_nmod_sub(tmp2.as_mut_ptr(), tmp.as_ptr(), b.as_ptr(), ctx.as_ptr());
        assert_eq!(fq_nmod_equal(tmp2.as_ptr(), a.as_ptr(), ctx.as_ptr()), 1);

        // a * a^{-1} = 1 (a is nonzero: 7 ≡ 2 in F_25)
        fq_nmod_inv(tmp.as_mut_ptr(), a.as_ptr(), ctx.as_ptr());
        fq_nmod_mul(tmp2.as_mut_ptr(), a.as_ptr(), tmp.as_ptr(), ctx.as_ptr());
        assert_eq!(fq_nmod_equal(tmp2.as_ptr(), one.as_ptr(), ctx.as_ptr()), 1);
        assert_eq!(fq_nmod_is_one(tmp2.as_ptr(), ctx.as_ptr()), 1);

        fq_nmod_clear(a.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_clear(b.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_clear(tmp.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_clear(tmp2.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_clear(one.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_ctx_clear(ctx.as_mut_ptr());
    }
}

#[test]
fn fq_nmod_frobenius_fixes_f_p() {
    // In F_{p^f}, the Frobenius x -> x^p fixes F_p.
    // Concretely: for x ∈ F_5 ⊂ F_25, frob(x) = x.
    unsafe {
        let mut ctx: fq_nmod_ctx_t = [std::mem::zeroed()];
        let var = std::ffi::CString::new("a").unwrap();
        fq_nmod_ctx_init_conway_ui(ctx.as_mut_ptr(), 5, 2, var.as_ptr());

        let mut x: fq_nmod_t = [std::mem::zeroed()];
        let mut fx: fq_nmod_t = [std::mem::zeroed()];
        fq_nmod_init(x.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_init(fx.as_mut_ptr(), ctx.as_ptr());

        fq_nmod_set_ui(x.as_mut_ptr(), 3, ctx.as_ptr()); // 3 ∈ F_5
        fq_nmod_frobenius(fx.as_mut_ptr(), x.as_ptr(), 1, ctx.as_ptr());
        assert_eq!(fq_nmod_equal(fx.as_ptr(), x.as_ptr(), ctx.as_ptr()), 1);

        fq_nmod_clear(x.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_clear(fx.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_ctx_clear(ctx.as_mut_ptr());
    }
}

#[test]
fn fq_default_mat_rref_uses_output_input_context_abi() {
    unsafe {
        let mut modulus: nmod_poly_t = [std::mem::zeroed()];
        nmod_poly_init(modulus.as_mut_ptr(), 5);
        nmod_poly_set_coeff_ui(modulus.as_mut_ptr(), 0, 2);
        nmod_poly_set_coeff_ui(modulus.as_mut_ptr(), 2, 1);

        let mut ctx: fq_default_ctx_t = [std::mem::zeroed()];
        let var = std::ffi::CString::new("a").unwrap();
        fq_default_ctx_init_modulus_nmod(ctx.as_mut_ptr(), modulus.as_ptr(), var.as_ptr());

        let mut one: fq_default_t = [std::mem::zeroed()];
        fq_default_init(one.as_mut_ptr(), ctx.as_ptr());
        fq_default_one(one.as_mut_ptr(), ctx.as_ptr());

        let mut input = MaybeUninit::<fq_default_mat_struct>::uninit();
        let mut output = MaybeUninit::<fq_default_mat_struct>::uninit();
        fq_default_mat_init(input.as_mut_ptr(), 2, 2, ctx.as_ptr());
        fq_default_mat_init(output.as_mut_ptr(), 2, 2, ctx.as_ptr());
        let mut input = input.assume_init();
        let mut output = output.assume_init();
        fq_default_mat_entry_set(&mut input, 0, 0, one.as_ptr(), ctx.as_ptr());
        fq_default_mat_entry_set(&mut input, 1, 1, one.as_ptr(), ctx.as_ptr());

        assert_eq!(fq_default_mat_rref(&mut output, &input, ctx.as_ptr()), 2);
        let mut diagonal: fq_default_t = [std::mem::zeroed()];
        fq_default_init(diagonal.as_mut_ptr(), ctx.as_ptr());
        fq_default_mat_entry(diagonal.as_mut_ptr(), &output, 1, 1, ctx.as_ptr());
        assert_eq!(fq_default_is_one(diagonal.as_ptr(), ctx.as_ptr()), 1);

        fq_default_clear(diagonal.as_mut_ptr(), ctx.as_ptr());
        fq_default_mat_clear(&mut output, ctx.as_ptr());
        fq_default_mat_clear(&mut input, ctx.as_ptr());
        fq_default_clear(one.as_mut_ptr(), ctx.as_ptr());
        fq_default_ctx_clear(ctx.as_mut_ptr());
        nmod_poly_clear(modulus.as_mut_ptr());
    }
}

// ─── fq_zech linkage + smoke tests ──────────────────────────────

#[test]
fn fq_zech_ctx_sizes_match_flint() {
    assert_eq!(std::mem::size_of::<fq_zech_ctx_struct>(), 88);
    assert_eq!(std::mem::size_of::<fq_zech_struct>(), 8);
}

#[test]
fn fq_zech_arithmetic_in_f256() {
    // F_{256} = F_{2^8}. Exercise context creation, element ops,
    // and the a * a^{-1} = 1 ring axiom.
    unsafe {
        let mut ctx: fq_zech_ctx_t = [std::mem::zeroed()];
        let var = std::ffi::CString::new("a").unwrap();
        fq_zech_ctx_init_conway_ui(ctx.as_mut_ptr(), 2, 8, var.as_ptr());

        assert_eq!(fq_zech_ctx_degree(ctx.as_ptr()), 8);

        let mut order = MaybeUninit::<fmpz>::uninit();
        fmpz_init(order.as_mut_ptr());
        fq_zech_ctx_order(order.as_mut_ptr(), ctx.as_ptr());
        let mut order = order.assume_init();
        assert_eq!(fmpz_get_si(&order as *const _), 256);
        fmpz_clear(&mut order as *mut _);

        let mut a: fq_zech_t = [fq_zech_struct { value: 0 }];
        let mut b: fq_zech_t = [fq_zech_struct { value: 0 }];
        let mut tmp: fq_zech_t = [fq_zech_struct { value: 0 }];
        let mut tmp2: fq_zech_t = [fq_zech_struct { value: 0 }];
        let mut one: fq_zech_t = [fq_zech_struct { value: 0 }];
        fq_zech_init(a.as_mut_ptr(), ctx.as_ptr());
        fq_zech_init(b.as_mut_ptr(), ctx.as_ptr());
        fq_zech_init(tmp.as_mut_ptr(), ctx.as_ptr());
        fq_zech_init(tmp2.as_mut_ptr(), ctx.as_ptr());
        fq_zech_init(one.as_mut_ptr(), ctx.as_ptr());

        fq_zech_set_ui(a.as_mut_ptr(), 7, ctx.as_ptr());
        fq_zech_set_ui(b.as_mut_ptr(), 3, ctx.as_ptr());
        fq_zech_one(one.as_mut_ptr(), ctx.as_ptr());

        // (a + b) - b = a
        fq_zech_add(tmp.as_mut_ptr(), a.as_ptr(), b.as_ptr(), ctx.as_ptr());
        fq_zech_sub(tmp2.as_mut_ptr(), tmp.as_ptr(), b.as_ptr(), ctx.as_ptr());
        assert_eq!(fq_zech_equal(tmp2.as_ptr(), a.as_ptr(), ctx.as_ptr()), 1);

        // a * a^{-1} = 1 (7 ≠ 0 in F_2, but 7 ≡ 1 in F_2 — pick
        // a non-trivial value via set_ui(5) which gives the generator^k)
        fq_zech_set_ui(a.as_mut_ptr(), 1, ctx.as_ptr());
        // 1 is certainly nonzero
        fq_zech_inv(tmp.as_mut_ptr(), a.as_ptr(), ctx.as_ptr());
        fq_zech_mul(tmp2.as_mut_ptr(), a.as_ptr(), tmp.as_ptr(), ctx.as_ptr());
        assert_eq!(fq_zech_is_one(tmp2.as_ptr(), ctx.as_ptr()), 1);

        fq_zech_clear(a.as_mut_ptr(), ctx.as_ptr());
        fq_zech_clear(b.as_mut_ptr(), ctx.as_ptr());
        fq_zech_clear(tmp.as_mut_ptr(), ctx.as_ptr());
        fq_zech_clear(tmp2.as_mut_ptr(), ctx.as_ptr());
        fq_zech_clear(one.as_mut_ptr(), ctx.as_ptr());
        fq_zech_ctx_clear(ctx.as_mut_ptr());
    }
}

// ── qadic linkage tests ─────────────────────────────────────────

#[test]
fn qadic_ctx_sizes_match_flint() {
    assert_eq!(std::mem::size_of::<qadic_struct>(), 40);
    assert_eq!(std::mem::size_of::<qadic_ctx_struct>(), 80);
}

#[test]
fn qadic_basic_arithmetic() {
    // Z_{5^2} at precision 50: test (a+b)-b=a and a*a^{-1}=1 for a unit.
    unsafe {
        let mut ctx: qadic_ctx_t = [std::mem::zeroed()];
        let var = CString::new("x").unwrap();
        let rc = _qadic_ctx_init_conway_ui(
            ctx.as_mut_ptr(),
            5,
            2,
            0,
            50,
            var.as_ptr(),
            padic_print_mode_PADIC_TERSE,
        );
        assert_ne!(rc, 0, "Conway poly not found for (5,2)");
        assert_eq!(qadic_ctx_degree(ctx.as_ptr()), 2);

        let mut a: qadic_t = [std::mem::zeroed()];
        let mut b: qadic_t = [std::mem::zeroed()];
        let mut tmp: qadic_t = [std::mem::zeroed()];
        let mut tmp2: qadic_t = [std::mem::zeroed()];
        let mut one_elt: qadic_t = [std::mem::zeroed()];
        qadic_init2(a.as_mut_ptr(), 50);
        qadic_init2(b.as_mut_ptr(), 50);
        qadic_init2(tmp.as_mut_ptr(), 50);
        qadic_init2(tmp2.as_mut_ptr(), 50);
        qadic_init2(one_elt.as_mut_ptr(), 50);

        qadic_set_ui(a.as_mut_ptr(), 3, ctx.as_ptr());
        qadic_set_ui(b.as_mut_ptr(), 4, ctx.as_ptr());
        qadic_set_ui(one_elt.as_mut_ptr(), 1, ctx.as_ptr());

        // (a + b) - b = a
        qadic_add(tmp.as_mut_ptr(), a.as_ptr(), b.as_ptr(), ctx.as_ptr());
        qadic_sub(tmp2.as_mut_ptr(), tmp.as_ptr(), b.as_ptr(), ctx.as_ptr());
        assert_eq!(qadic_equal(tmp2.as_ptr(), a.as_ptr()), 1, "(a+b)-b != a");

        // a * a^{-1} = 1 (a=3 is a unit in Z_{5^2})
        qadic_inv(tmp.as_mut_ptr(), a.as_ptr(), ctx.as_ptr());
        qadic_mul(tmp2.as_mut_ptr(), a.as_ptr(), tmp.as_ptr(), ctx.as_ptr());
        assert_eq!(qadic_is_one(tmp2.as_ptr()), 1, "a * a^{{-1}} != 1");

        qadic_clear(a.as_mut_ptr());
        qadic_clear(b.as_mut_ptr());
        qadic_clear(tmp.as_mut_ptr());
        qadic_clear(tmp2.as_mut_ptr());
        qadic_clear(one_elt.as_mut_ptr());
        qadic_ctx_clear(ctx.as_mut_ptr());
    }
}

// ─── fmpz_poly linkage smoke tests ────────────────────────────

#[test]
fn fmpz_poly_add_and_degree() {
    unsafe {
        let mut a: fmpz_poly_t = [std::mem::zeroed()];
        let mut b: fmpz_poly_t = [std::mem::zeroed()];
        let mut c: fmpz_poly_t = [std::mem::zeroed()];
        fmpz_poly_init(a.as_mut_ptr());
        fmpz_poly_init(b.as_mut_ptr());
        fmpz_poly_init(c.as_mut_ptr());

        // a = 1 + 2x + 3x^2
        fmpz_poly_set_coeff_si(a.as_mut_ptr(), 0, 1);
        fmpz_poly_set_coeff_si(a.as_mut_ptr(), 1, 2);
        fmpz_poly_set_coeff_si(a.as_mut_ptr(), 2, 3);
        // b = 4 + 5x
        fmpz_poly_set_coeff_si(b.as_mut_ptr(), 0, 4);
        fmpz_poly_set_coeff_si(b.as_mut_ptr(), 1, 5);

        fmpz_poly_add(c.as_mut_ptr(), a.as_ptr(), b.as_ptr());
        // c should be 5 + 7x + 3x^2
        assert_eq!(fmpz_poly_get_coeff_si(c.as_ptr(), 0), 5);
        assert_eq!(fmpz_poly_get_coeff_si(c.as_ptr(), 1), 7);
        assert_eq!(fmpz_poly_get_coeff_si(c.as_ptr(), 2), 3);
        assert_eq!(fmpz_poly_degree(c.as_ptr()), 2);
        assert_eq!(fmpz_poly_length(c.as_ptr()), 3);

        fmpz_poly_clear(a.as_mut_ptr());
        fmpz_poly_clear(b.as_mut_ptr());
        fmpz_poly_clear(c.as_mut_ptr());
    }
}

#[test]
fn fmpz_poly_mul_and_evaluate() {
    unsafe {
        let mut a: fmpz_poly_t = [std::mem::zeroed()];
        let mut b: fmpz_poly_t = [std::mem::zeroed()];
        let mut c: fmpz_poly_t = [std::mem::zeroed()];
        fmpz_poly_init(a.as_mut_ptr());
        fmpz_poly_init(b.as_mut_ptr());
        fmpz_poly_init(c.as_mut_ptr());

        // (1 + x)(1 - x) = 1 - x^2
        fmpz_poly_set_coeff_si(a.as_mut_ptr(), 0, 1);
        fmpz_poly_set_coeff_si(a.as_mut_ptr(), 1, 1);
        fmpz_poly_set_coeff_si(b.as_mut_ptr(), 0, 1);
        fmpz_poly_set_coeff_si(b.as_mut_ptr(), 1, -1);
        fmpz_poly_mul(c.as_mut_ptr(), a.as_ptr(), b.as_ptr());
        assert_eq!(fmpz_poly_get_coeff_si(c.as_ptr(), 0), 1);
        assert_eq!(fmpz_poly_get_coeff_si(c.as_ptr(), 1), 0);
        assert_eq!(fmpz_poly_get_coeff_si(c.as_ptr(), 2), -1);

        // Evaluate c at x=3: 1 - 9 = -8
        let mut x = MaybeUninit::<fmpz>::uninit();
        let mut out = MaybeUninit::<fmpz>::uninit();
        fmpz_init(x.as_mut_ptr());
        fmpz_init(out.as_mut_ptr());
        fmpz_set_si(x.as_mut_ptr(), 3);
        fmpz_poly_evaluate_fmpz(out.as_mut_ptr(), c.as_ptr(), x.as_ptr());
        assert_eq!(fmpz_get_si(out.as_ptr()), -8);
        fmpz_clear(x.as_mut_ptr());
        fmpz_clear(out.as_mut_ptr());

        fmpz_poly_clear(a.as_mut_ptr());
        fmpz_poly_clear(b.as_mut_ptr());
        fmpz_poly_clear(c.as_mut_ptr());
    }
}

// ─── fmpz_mpoly linkage smoke tests ──────────────────────────

#[test]
fn fmpz_mpoly_add_and_mul() {
    unsafe {
        let mut ctx: fmpz_mpoly_ctx_t = [std::mem::zeroed()];
        fmpz_mpoly_ctx_init(ctx.as_mut_ptr(), 2, ordering_t_ORD_DEGREVLEX);
        assert_eq!(fmpz_mpoly_ctx_nvars(ctx.as_ptr()), 2);

        let mut a: fmpz_mpoly_t = [std::mem::zeroed()];
        let mut b: fmpz_mpoly_t = [std::mem::zeroed()];
        let mut c: fmpz_mpoly_t = [std::mem::zeroed()];
        fmpz_mpoly_init(a.as_mut_ptr(), ctx.as_ptr());
        fmpz_mpoly_init(b.as_mut_ptr(), ctx.as_ptr());
        fmpz_mpoly_init(c.as_mut_ptr(), ctx.as_ptr());

        // Build a = x + y.
        let mut coeff = MaybeUninit::<fmpz>::uninit();
        fmpz_init(coeff.as_mut_ptr());
        fmpz_set_si(coeff.as_mut_ptr(), 1);
        // x = x^1 y^0
        let exp_x: [ulong; 2] = [1, 0];
        fmpz_mpoly_set_coeff_fmpz_ui(a.as_mut_ptr(), coeff.as_ptr(), exp_x.as_ptr(), ctx.as_ptr());
        // y = x^0 y^1
        let exp_y: [ulong; 2] = [0, 1];
        fmpz_mpoly_set_coeff_fmpz_ui(a.as_mut_ptr(), coeff.as_ptr(), exp_y.as_ptr(), ctx.as_ptr());

        // b = x - y
        fmpz_set_si(coeff.as_mut_ptr(), 1);
        fmpz_mpoly_set_coeff_fmpz_ui(b.as_mut_ptr(), coeff.as_ptr(), exp_x.as_ptr(), ctx.as_ptr());
        fmpz_set_si(coeff.as_mut_ptr(), -1);
        fmpz_mpoly_set_coeff_fmpz_ui(b.as_mut_ptr(), coeff.as_ptr(), exp_y.as_ptr(), ctx.as_ptr());

        // (x + y)(x - y) = x^2 - y^2.
        fmpz_mpoly_mul(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), ctx.as_ptr());
        assert_eq!(fmpz_mpoly_length(c.as_ptr(), ctx.as_ptr()), 2);

        let exp_x2: [ulong; 2] = [2, 0];
        let exp_y2: [ulong; 2] = [0, 2];
        let mut cx2 = MaybeUninit::<fmpz>::uninit();
        let mut cy2 = MaybeUninit::<fmpz>::uninit();
        fmpz_init(cx2.as_mut_ptr());
        fmpz_init(cy2.as_mut_ptr());
        fmpz_mpoly_get_coeff_fmpz_ui(cx2.as_mut_ptr(), c.as_ptr(), exp_x2.as_ptr(), ctx.as_ptr());
        fmpz_mpoly_get_coeff_fmpz_ui(cy2.as_mut_ptr(), c.as_ptr(), exp_y2.as_ptr(), ctx.as_ptr());
        assert_eq!(fmpz_get_si(cx2.as_ptr()), 1);
        assert_eq!(fmpz_get_si(cy2.as_ptr()), -1);

        fmpz_clear(cx2.as_mut_ptr());
        fmpz_clear(cy2.as_mut_ptr());
        fmpz_clear(coeff.as_mut_ptr());
        fmpz_mpoly_clear(a.as_mut_ptr(), ctx.as_ptr());
        fmpz_mpoly_clear(b.as_mut_ptr(), ctx.as_ptr());
        fmpz_mpoly_clear(c.as_mut_ptr(), ctx.as_ptr());
        fmpz_mpoly_ctx_clear(ctx.as_mut_ptr());
    }
}

// ─── nmod_mpoly linkage smoke tests ──────────────────────────

#[test]
fn nmod_mpoly_basic_arith() {
    unsafe {
        let mut ctx: nmod_mpoly_ctx_t = [std::mem::zeroed()];
        nmod_mpoly_ctx_init(ctx.as_mut_ptr(), 3, ordering_t_ORD_DEGREVLEX, 17);
        assert_eq!(nmod_mpoly_ctx_nvars(ctx.as_ptr()), 3);
        assert_eq!(nmod_mpoly_ctx_modulus(ctx.as_ptr()), 17);

        let mut a: nmod_mpoly_t = [std::mem::zeroed()];
        let mut b: nmod_mpoly_t = [std::mem::zeroed()];
        let mut c: nmod_mpoly_t = [std::mem::zeroed()];
        nmod_mpoly_init(a.as_mut_ptr(), ctx.as_ptr());
        nmod_mpoly_init(b.as_mut_ptr(), ctx.as_ptr());
        nmod_mpoly_init(c.as_mut_ptr(), ctx.as_ptr());

        // a = 5, b = 13 ⇒ a + b ≡ 1 (mod 17).
        let exp0: [ulong; 3] = [0, 0, 0];
        nmod_mpoly_set_coeff_ui_ui(a.as_mut_ptr(), 5, exp0.as_ptr(), ctx.as_ptr());
        nmod_mpoly_set_coeff_ui_ui(b.as_mut_ptr(), 13, exp0.as_ptr(), ctx.as_ptr());
        nmod_mpoly_add(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), ctx.as_ptr());
        let coeff = nmod_mpoly_get_coeff_ui_ui(c.as_ptr(), exp0.as_ptr(), ctx.as_ptr());
        assert_eq!(coeff, 1);

        nmod_mpoly_clear(a.as_mut_ptr(), ctx.as_ptr());
        nmod_mpoly_clear(b.as_mut_ptr(), ctx.as_ptr());
        nmod_mpoly_clear(c.as_mut_ptr(), ctx.as_ptr());
        nmod_mpoly_ctx_clear(ctx.as_mut_ptr());
    }
}

// ─── fmpq_mpoly linkage smoke tests ──────────────────────────

#[test]
fn fmpz_mod_mpoly_large_modulus_mul() {
    unsafe {
        let mut modulus = make_fmpz_from_str("18446744073709551629");
        let mut ctx: fmpz_mod_mpoly_ctx_t = [std::mem::zeroed()];
        fmpz_mod_mpoly_ctx_init(ctx.as_mut_ptr(), 2, ordering_t_ORD_DEGREVLEX, &modulus);
        let mut a: fmpz_mod_mpoly_t = [std::mem::zeroed()];
        let mut b: fmpz_mod_mpoly_t = [std::mem::zeroed()];
        let mut c: fmpz_mod_mpoly_t = [std::mem::zeroed()];
        fmpz_mod_mpoly_init(a.as_mut_ptr(), ctx.as_ptr());
        fmpz_mod_mpoly_init(b.as_mut_ptr(), ctx.as_ptr());
        fmpz_mod_mpoly_init(c.as_mut_ptr(), ctx.as_ptr());

        let mut coefficient = make_fmpz_from_str("18446744073709551628");
        let exp_x: [ulong; 2] = [1, 0];
        let exp_y: [ulong; 2] = [0, 1];
        fmpz_mod_mpoly_set_coeff_fmpz_ui(
            a.as_mut_ptr(),
            &coefficient,
            exp_x.as_ptr(),
            ctx.as_ptr(),
        );
        fmpz_set_si(&mut coefficient, 2);
        fmpz_mod_mpoly_set_coeff_fmpz_ui(
            b.as_mut_ptr(),
            &coefficient,
            exp_y.as_ptr(),
            ctx.as_ptr(),
        );
        fmpz_mod_mpoly_mul(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), ctx.as_ptr());

        let mut got = MaybeUninit::<fmpz>::uninit();
        fmpz_init(got.as_mut_ptr());
        let mut exponent = [0_u64; 2];
        fmpz_mod_mpoly_get_term_coeff_fmpz(got.as_mut_ptr(), c.as_ptr(), 0, ctx.as_ptr());
        fmpz_mod_mpoly_get_term_exp_ui(exponent.as_mut_ptr(), c.as_ptr(), 0, ctx.as_ptr());
        let mut got = got.assume_init();
        let mut expected = make_fmpz_from_str("18446744073709551627");
        assert_eq!(fmpz_mod_mpoly_length(c.as_ptr(), ctx.as_ptr()), 1);
        assert_eq!(fmpz_equal(&got, &expected), 1);
        assert_eq!(exponent, [1, 1]);

        fmpz_clear(&mut expected);
        fmpz_clear(&mut got);
        fmpz_clear(&mut coefficient);
        fmpz_mod_mpoly_clear(a.as_mut_ptr(), ctx.as_ptr());
        fmpz_mod_mpoly_clear(b.as_mut_ptr(), ctx.as_ptr());
        fmpz_mod_mpoly_clear(c.as_mut_ptr(), ctx.as_ptr());
        fmpz_mod_mpoly_ctx_clear(ctx.as_mut_ptr());
        fmpz_clear(&mut modulus);
    }
}

#[test]
fn fq_nmod_mpoly_extension_coefficients_mul() {
    unsafe {
        let mut field_ctx: fq_nmod_ctx_t = [std::mem::zeroed()];
        let variable = std::ffi::CString::new("a").unwrap();
        fq_nmod_ctx_init_conway_ui(field_ctx.as_mut_ptr(), 3, 2, variable.as_ptr());
        let mut ctx: fq_nmod_mpoly_ctx_t = [std::mem::zeroed()];
        fq_nmod_mpoly_ctx_init(
            ctx.as_mut_ptr(),
            2,
            ordering_t_ORD_DEGLEX,
            field_ctx.as_ptr(),
        );
        let mut a: fq_nmod_mpoly_t = [std::mem::zeroed()];
        let mut b: fq_nmod_mpoly_t = [std::mem::zeroed()];
        let mut c: fq_nmod_mpoly_t = [std::mem::zeroed()];
        fq_nmod_mpoly_init(a.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_mpoly_init(b.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_mpoly_init(c.as_mut_ptr(), ctx.as_ptr());

        let mut two: fq_nmod_t = [std::mem::zeroed()];
        fq_nmod_init(two.as_mut_ptr(), field_ctx.as_ptr());
        fq_nmod_set_ui(two.as_mut_ptr(), 2, field_ctx.as_ptr());
        let exp_x: [ulong; 2] = [1, 0];
        let exp_y: [ulong; 2] = [0, 1];
        fq_nmod_mpoly_set_coeff_fq_nmod_ui(
            a.as_mut_ptr(),
            two.as_ptr(),
            exp_x.as_ptr(),
            ctx.as_ptr(),
        );
        fq_nmod_mpoly_set_coeff_fq_nmod_ui(
            b.as_mut_ptr(),
            two.as_ptr(),
            exp_y.as_ptr(),
            ctx.as_ptr(),
        );
        fq_nmod_mpoly_mul(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), ctx.as_ptr());

        let mut got: fq_nmod_t = [std::mem::zeroed()];
        let mut one: fq_nmod_t = [std::mem::zeroed()];
        fq_nmod_init(got.as_mut_ptr(), field_ctx.as_ptr());
        fq_nmod_init(one.as_mut_ptr(), field_ctx.as_ptr());
        fq_nmod_one(one.as_mut_ptr(), field_ctx.as_ptr());
        let mut exponent = [0_u64; 2];
        fq_nmod_mpoly_get_term_coeff_fq_nmod(got.as_mut_ptr(), c.as_ptr(), 0, ctx.as_ptr());
        fq_nmod_mpoly_get_term_exp_ui(exponent.as_mut_ptr(), c.as_ptr(), 0, ctx.as_ptr());
        assert_eq!(fq_nmod_mpoly_length(c.as_ptr(), ctx.as_ptr()), 1);
        assert_eq!(
            fq_nmod_equal(got.as_ptr(), one.as_ptr(), field_ctx.as_ptr()),
            1
        );
        assert_eq!(exponent, [1, 1]);

        fq_nmod_clear(one.as_mut_ptr(), field_ctx.as_ptr());
        fq_nmod_clear(got.as_mut_ptr(), field_ctx.as_ptr());
        fq_nmod_clear(two.as_mut_ptr(), field_ctx.as_ptr());
        fq_nmod_mpoly_clear(a.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_mpoly_clear(b.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_mpoly_clear(c.as_mut_ptr(), ctx.as_ptr());
        fq_nmod_mpoly_ctx_clear(ctx.as_mut_ptr());
        fq_nmod_ctx_clear(field_ctx.as_mut_ptr());
    }
}

#[test]
fn fmpq_mpoly_basic_arith() {
    unsafe {
        let mut ctx: fmpq_mpoly_ctx_t = [std::mem::zeroed()];
        fmpq_mpoly_ctx_init(ctx.as_mut_ptr(), 2, ordering_t_ORD_LEX);
        assert_eq!(fmpq_mpoly_ctx_nvars(ctx.as_ptr()), 2);

        let mut a: fmpq_mpoly_t = [std::mem::zeroed()];
        let mut b: fmpq_mpoly_t = [std::mem::zeroed()];
        let mut c: fmpq_mpoly_t = [std::mem::zeroed()];
        fmpq_mpoly_init(a.as_mut_ptr(), ctx.as_ptr());
        fmpq_mpoly_init(b.as_mut_ptr(), ctx.as_ptr());
        fmpq_mpoly_init(c.as_mut_ptr(), ctx.as_ptr());

        // a = 1/2 * x, b = 2/3 * y — two terms each at a single monomial.
        let half = make_fmpq(1, 2);
        let two_thirds = make_fmpq(2, 3);
        let exp_x: [ulong; 2] = [1, 0];
        let exp_y: [ulong; 2] = [0, 1];
        fmpq_mpoly_set_coeff_fmpq_ui(a.as_mut_ptr(), &half, exp_x.as_ptr(), ctx.as_ptr());
        fmpq_mpoly_set_coeff_fmpq_ui(b.as_mut_ptr(), &two_thirds, exp_y.as_ptr(), ctx.as_ptr());

        // a * b = 1/3 * xy.
        fmpq_mpoly_mul(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), ctx.as_ptr());
        let exp_xy: [ulong; 2] = [1, 1];
        let mut got = MaybeUninit::<fmpq>::uninit();
        fmpq_init(got.as_mut_ptr());
        fmpq_mpoly_get_coeff_fmpq_ui(got.as_mut_ptr(), c.as_ptr(), exp_xy.as_ptr(), ctx.as_ptr());
        let one_third = make_fmpq(1, 3);
        assert_eq!(fmpq_equal(got.as_ptr(), &one_third), 1);
        fmpq_clear(got.as_mut_ptr());

        let mut half_m = half;
        let mut two_thirds_m = two_thirds;
        let mut one_third_m = one_third;
        clear_fmpq(&mut half_m);
        clear_fmpq(&mut two_thirds_m);
        clear_fmpq(&mut one_third_m);

        fmpq_mpoly_clear(a.as_mut_ptr(), ctx.as_ptr());
        fmpq_mpoly_clear(b.as_mut_ptr(), ctx.as_ptr());
        fmpq_mpoly_clear(c.as_mut_ptr(), ctx.as_ptr());
        fmpq_mpoly_ctx_clear(ctx.as_mut_ptr());
    }
}
