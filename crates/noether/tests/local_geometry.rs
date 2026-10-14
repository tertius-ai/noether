// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::{
    AffineSpace, BlowUp, BlowUpChart, ProjectiveSpace, SchemeCompletion, SeriesTruncation,
};
use noether::numthy::FiniteField;
use noether::{RationalField, Ring, SemanticType, Q};

fn projective_cusp() -> (
    noether::commalg::Scheme<FiniteField>,
    noether::commalg::SchemePoint<FiniteField>,
) {
    let field = FiniteField::prime(5_u64).unwrap();
    let plane = ProjectiveSpace::with_names(field.clone(), 2, ["x", "y", "z"]).unwrap();
    let ring = plane.coordinate_ring();
    let coordinates = ring.generators().unwrap();
    let y_squared_z = ring
        .mul(&coordinates[1].pow(2).unwrap(), &coordinates[2])
        .unwrap();
    let equation = ring
        .sub(&y_squared_z, &coordinates[0].pow(3).unwrap())
        .unwrap();
    let cusp = plane
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();
    let center = cusp
        .point([field.zero(), field.zero(), field.one()])
        .unwrap();
    (cusp, center)
}

#[test]
fn cusp_point_blow_up_keeps_all_chartwise_transform_evidence() {
    let (cusp, center) = projective_cusp();
    assert!(!cusp.is_smooth().unwrap());

    let blow_up = cusp.blow_up_at(&center).unwrap();
    assert_eq!(blow_up.source_affine_chart(), Some(2));
    assert_eq!(blow_up.charts().len(), 2);
    blow_up.verify().unwrap();

    let chart = &blow_up.charts()[0];
    let ring = chart.strict_transform().coordinate_ring();
    let variables = ring.generators().unwrap();
    let expected_strict = ring
        .ideal([ring
            .sub(&variables[0], &variables[1].pow(2).unwrap())
            .unwrap()])
        .unwrap();
    assert_eq!(chart.weak_transform().defining_ideal(), &expected_strict);
    assert_eq!(chart.strict_transform().defining_ideal(), &expected_strict);
    assert!(chart.strict_transform().is_smooth().unwrap());

    let expected_exceptional = ring
        .ideal([variables[0].clone(), variables[1].pow(2).unwrap()])
        .unwrap();
    assert_eq!(
        chart.exceptional_divisor().defining_ideal(),
        &expected_exceptional
    );
    assert_eq!(chart.exceptional_divisor().dimension().unwrap(), 0);
    assert_eq!(
        chart.exceptional_divisor().degree().unwrap().to_string(),
        "2"
    );

    let map = chart.source_coordinate_map();
    assert_eq!(map[0], variables[0]);
    assert_eq!(map[1], ring.mul(&variables[0], &variables[1]).unwrap());
    assert_eq!(map[2], ring.one());

    let chart_encoded = chart.to_json().unwrap();
    assert_eq!(
        BlowUpChart::from_json(&chart_encoded).unwrap(),
        chart.clone()
    );

    let encoded = blow_up.to_json().unwrap();
    assert_eq!(BlowUp::from_json(&encoded).unwrap(), blow_up);
    let mut payload: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    payload["charts"][0]["chart_index"] = serde_json::json!(1);
    assert!(BlowUp::from_json(&serde_json::to_string(&payload).unwrap()).is_err());
    assert_eq!(BlowUp::<FiniteField>::TYPE_ID, "noether/commalg/BlowUp@1");
    assert_eq!(
        BlowUpChart::<FiniteField>::TYPE_ID,
        "noether/commalg/BlowUpChart@1"
    );
}

#[test]
fn cusp_tangent_cone_and_residual_completion_are_fixed_and_exact() {
    let (cusp, center) = projective_cusp();
    let tangent = cusp.tangent_cone(&center).unwrap();
    let tangent_ring = tangent.coordinate_ring();
    let tangent_variables = tangent_ring.generators().unwrap();
    let expected_tangent = tangent_ring
        .ideal([tangent_variables[1].pow(2).unwrap()])
        .unwrap();
    assert_eq!(tangent.defining_ideal(), &expected_tangent);
    assert_eq!(tangent.dimension().unwrap(), 1);
    assert_eq!(tangent.degree().unwrap().to_string(), "2");

    let blow_up = cusp.blow_up_at(&center).unwrap();
    let strict = blow_up.charts()[0].strict_transform();
    let field = cusp.base();
    let residual = strict.point([field.zero(), field.zero()]).unwrap();
    let completion = strict
        .completion(&residual, SeriesTruncation::total_degree(4).unwrap())
        .unwrap();
    completion.verify().unwrap();
    assert_eq!(completion.equations().len(), 1);
    let series_variables = completion.ambient_ring().generators().unwrap();
    let expected_equation = (&series_variables[1].pow(2).unwrap() - &series_variables[0]).unwrap();
    assert_eq!(completion.equations()[0], expected_equation);
    assert_eq!(completion.truncation().bound(), Some(4));

    let encoded = completion.to_json().unwrap();
    assert_eq!(SchemeCompletion::from_json(&encoded).unwrap(), completion);
    let mut payload: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    payload["equations"] = serde_json::json!([]);
    assert!(SchemeCompletion::from_json(&serde_json::to_string(&payload).unwrap()).is_err());
    assert_eq!(
        SchemeCompletion::<FiniteField>::TYPE_ID,
        "noether/commalg/SchemeCompletion@1"
    );
}

#[test]
fn rational_points_use_the_same_generic_rust_local_geometry() {
    let plane = ProjectiveSpace::with_names(Q, 2, ["x", "y", "z"]).unwrap();
    let ring = plane.coordinate_ring();
    let coordinates = ring.generators().unwrap();
    let equation = ring
        .sub(
            &ring
                .mul(&coordinates[1].pow(2).unwrap(), &coordinates[2])
                .unwrap(),
            &coordinates[0].pow(3).unwrap(),
        )
        .unwrap();
    let cusp = plane
        .closed_subscheme(ring.ideal([equation]).unwrap())
        .unwrap();
    let center = cusp.point([Q.zero(), Q.zero(), Q.one()]).unwrap();
    center.verify().unwrap();
    let encoded = center.to_json().unwrap();
    assert!(noether::commalg::SchemePoint::<RationalField>::from_json(&encoded).unwrap() == center);
    assert_eq!(cusp.blow_up_at(&center).unwrap().charts().len(), 2);
    assert_eq!(
        cusp.tangent_cone(&center)
            .unwrap()
            .degree()
            .unwrap()
            .to_string(),
        "2"
    );
}

#[test]
fn nonprincipal_tangent_cone_uses_a_local_standard_basis() {
    let space = AffineSpace::with_names(Q, 3, ["x", "y", "z"]).unwrap();
    let ring = space.coordinate_ring();
    let variables = ring.generators().unwrap();
    let curve = space
        .closed_subscheme(
            ring.ideal([
                ring.sub(&variables[1], &variables[0].pow(2).unwrap())
                    .unwrap(),
                ring.sub(&variables[2], &variables[0].pow(3).unwrap())
                    .unwrap(),
            ])
            .unwrap(),
        )
        .unwrap();
    let center = curve.point([Q.zero(), Q.zero(), Q.zero()]).unwrap();

    let tangent = curve.tangent_cone(&center).unwrap();
    let tangent_variables = tangent.coordinate_ring().generators().unwrap();

    assert_eq!(
        tangent.defining_ideal(),
        &tangent
            .coordinate_ring()
            .ideal([tangent_variables[1].clone(), tangent_variables[2].clone()])
            .unwrap()
    );
    assert_eq!(tangent.dimension().unwrap(), 1);
    assert_eq!(tangent.degree().unwrap(), 1);
}
