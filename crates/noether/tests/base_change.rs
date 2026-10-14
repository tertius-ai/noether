// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{AffineSpace, MonomialOrder, PolynomialRing, ProjectiveSpace};
use noether::numthy::FiniteField;
use noether::{Integer, Ring, Q, Z};

#[test]
fn public_polynomial_base_change_is_one_checked_induced_map() {
    let source = PolynomialRing::new(Z, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
    let [x, y] = source.generators().unwrap().try_into().unwrap();
    let value = source
        .add(
            &source.mul(&x, &x).unwrap(),
            &y.scalar_mul(&Integer::from(3)).unwrap(),
        )
        .unwrap();
    let map = Z.rational_embedding();
    let target = source.base_change(&map).unwrap();
    let induced = source.base_change_homomorphism(map).unwrap();

    assert_eq!(target.coefficient_ring(), &Q);
    assert_eq!(target.variables(), &["x", "y"]);
    assert_eq!(induced.target(), &target);
    assert_eq!(
        induced.apply(&value).unwrap(),
        value.base_change(&map).unwrap()
    );
}

#[test]
fn induced_ideal_and_projective_base_change_preserve_presentations() {
    let prime = FiniteField::prime(5_u8).unwrap();
    let extension = FiniteField::with_modulus(5, vec![2, 0, 1]).unwrap();
    let embedding = extension.prime_subfield_embedding();
    let source =
        PolynomialRing::new(prime.clone(), ["x", "y"], MonomialOrder::Lexicographic).unwrap();
    let [x, y] = source.generators().unwrap().try_into().unwrap();
    let ideal = source
        .ideal([source.sub(&y, &source.mul(&x, &x).unwrap()).unwrap()])
        .unwrap();
    let target = PolynomialRing::new(
        extension.clone(),
        ["u", "t"],
        MonomialOrder::GradedReverseLexicographic,
    )
    .unwrap();
    let [u, t] = target.generators().unwrap().try_into().unwrap();
    let homomorphism = source
        .homomorphism(&target, embedding.clone(), vec![u, t])
        .unwrap();
    let extended = homomorphism.extend_ideal(&ideal).unwrap();
    assert_eq!(extended.ring(), &target);
    extended.verify().unwrap();

    let projective = ProjectiveSpace::with_names(prime, 1, ["X", "Y"]).unwrap();
    let projective_changed = projective.base_change(&embedding).unwrap();
    assert_eq!(projective_changed.base(), &extension);
    assert_eq!(projective_changed.coordinates(), &["X", "Y"]);

    let affine = AffineSpace::with_names(FiniteField::prime(5_u8).unwrap(), 2, ["x", "y"]).unwrap();
    let affine_ring = affine.coordinate_ring();
    let [x, y] = affine_ring.generators().unwrap().try_into().unwrap();
    let curve = affine
        .closed_subscheme(
            affine_ring
                .ideal([affine_ring
                    .sub(&y, &affine_ring.mul(&x, &x).unwrap())
                    .unwrap()])
                .unwrap(),
        )
        .unwrap();
    let point = curve
        .point([affine.base().element(2), affine.base().element(4)])
        .unwrap();
    let changed_point = point.base_change(&embedding).unwrap();
    changed_point.verify().unwrap();
    assert_eq!(
        changed_point.scheme(),
        &curve.base_change(&embedding).unwrap()
    );
}
