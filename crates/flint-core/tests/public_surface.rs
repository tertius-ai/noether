// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use flint_core::{Acb, Arb, Integer, Rational};

#[test]
fn owned_exact_and_ball_values_work_through_the_public_surface() {
    let large: Integer = "18446744073709551617".parse().unwrap();
    assert_eq!((&large * &large).sqrt_floor(), large);

    let rational: Rational = "-6/8".parse().unwrap();
    assert_eq!(rational.to_string(), "-3/4");
    assert_eq!((&rational + &rational).to_string(), "-3/2");

    let two = Arb::from_f64(2.0);
    assert!(two.mul(&two, 128).same_ball(&Arb::from_f64(4.0)));

    let imaginary_unit = Acb::from_re_im(0.0, 1.0);
    assert!(imaginary_unit
        .mul(&imaginary_unit, 128)
        .same_ball(&Acb::from_re_im(-1.0, 0.0)));
}
