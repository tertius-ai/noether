// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::commalg::ResidueRing;
use noether::linalg::{LinearOperator, Matrix, Vector};
use noether::numthy::{FiniteField, PadicRing};
use noether::{
    CommutativeRing, ContextMismatchError, Integer, NoetherError, Rational, RealBallField, Ring,
    SemanticType, TypeDescription, Q, Z,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModSeven;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ModSevenElement(u8);

static MOD_SEVEN_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "test/ModSeven@1",
    rust_path: "vectors::ModSeven",
    python_path: None,
};

static MOD_SEVEN_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "test/ModSevenElement@1",
    rust_path: "vectors::ModSevenElement",
    python_path: None,
};

impl SemanticType for ModSeven {
    const TYPE_ID: &'static str = "test/ModSeven@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(vec![7])
    }

    fn describe() -> &'static TypeDescription {
        &MOD_SEVEN_DESCRIPTION
    }
}

impl SemanticType for ModSevenElement {
    const TYPE_ID: &'static str = "test/ModSevenElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(vec![self.0])
    }

    fn describe() -> &'static TypeDescription {
        &MOD_SEVEN_ELEMENT_DESCRIPTION
    }
}

impl Ring for ModSeven {
    type Element = ModSevenElement;

    fn contains(&self, value: &Self::Element) -> bool {
        value.0 < 7
    }

    fn zero(&self) -> Self::Element {
        ModSevenElement(0)
    }

    fn one(&self) -> Self::Element {
        ModSevenElement(1)
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(ModSevenElement((a.0 + b.0) % 7))
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(ModSevenElement((a.0 + 7 - b.0) % 7))
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(ModSevenElement((a.0 * b.0) % 7))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        Ok(ModSevenElement((7 - value.0) % 7))
    }
}

impl CommutativeRing for ModSeven {}

#[test]
fn every_ring_receives_generic_vector_arithmetic() {
    let x = Vector::from_values(
        ModSeven,
        [ModSevenElement(6), ModSevenElement(2), ModSevenElement(4)],
    )
    .unwrap();
    let y = Vector::from_values(
        ModSeven,
        [ModSevenElement(3), ModSevenElement(5), ModSevenElement(1)],
    )
    .unwrap();

    assert_eq!(x.dimension(), 3);
    assert_eq!(x.add(&y).unwrap().get(0).unwrap(), ModSevenElement(2));
    assert_eq!(x.sub(&y).unwrap().get(1).unwrap(), ModSevenElement(4));
    assert_eq!(x.neg().unwrap().get(2).unwrap(), ModSevenElement(3));
    assert_eq!(x.dot(&y).unwrap(), ModSevenElement(4));
    assert_eq!(x.to_column_matrix().shape(), (3, 1));
    assert_eq!(x.to_row_matrix().shape(), (1, 3));
}

#[test]
fn exact_vectors_reuse_matrix_semantics() {
    let large_integer = "100000000000000000000000000000000000000";
    let integer =
        Vector::from_values(Z, [Integer::from(1), large_integer.parse().unwrap()]).unwrap();
    let rational =
        Vector::from_values(Q, ["1/2".parse::<Rational>().unwrap(), Rational::from(3)]).unwrap();

    assert_eq!(integer.get(1).unwrap().to_string(), large_integer);
    assert_eq!(rational.dot(&rational).unwrap().to_string(), "37/4");
    assert_eq!(
        Vector::<noether::IntegerRing>::TYPE_ID,
        "noether/linalg/Vector@1"
    );
    assert_eq!(
        Vector::<noether::IntegerRing>::describe().python_path,
        Some("noether.linalg.Vector")
    );
    assert_eq!(
        Vector::<noether::IntegerRing>::from_json(&integer.to_json().unwrap()).unwrap(),
        integer
    );
    assert_eq!(
        Vector::<noether::RationalField>::from_json(&rational.to_json().unwrap()).unwrap(),
        rational
    );
}

#[test]
fn specialized_context_vectors_reuse_matrix_arithmetic() {
    let field = FiniteField::extension(3, 2).unwrap();
    let generator = field.generator();
    let finite_left = Vector::from_values(field.clone(), [field.one(), generator.clone()]).unwrap();
    let finite_right =
        Vector::from_values(field.clone(), [generator.clone(), field.one()]).unwrap();
    assert_eq!(
        finite_left.dot(&finite_right).unwrap(),
        field.add(&generator, &generator).unwrap()
    );

    let residues = ResidueRing::new(25_u8).unwrap();
    let residue_left = Vector::from_values(
        residues.clone(),
        [residues.element(3_u8), residues.element(4_u8)],
    )
    .unwrap();
    let residue_right = Vector::from_values(
        residues.clone(),
        [residues.element(5_u8), residues.element(6_u8)],
    )
    .unwrap();
    assert_eq!(
        residue_left.dot(&residue_right).unwrap(),
        residues.element(14_u8)
    );

    let padics = PadicRing::new(5_u8, 3).unwrap();
    let padic_left =
        Vector::from_values(padics.clone(), [padics.element(3_u8), padics.element(4_u8)]).unwrap();
    let padic_right =
        Vector::from_values(padics.clone(), [padics.element(5_u8), padics.element(6_u8)]).unwrap();
    assert_eq!(padic_left.dot(&padic_right).unwrap(), padics.element(39_u8));

    let balls = RealBallField::new(96).unwrap();
    let ball_left = Vector::from_values(balls, [balls.element(1_u8), balls.element(2_u8)]).unwrap();
    let ball_right =
        Vector::from_values(balls, [balls.element(3_u8), balls.element(4_u8)]).unwrap();
    assert!(ball_left
        .dot(&ball_right)
        .unwrap()
        .overlaps(&balls.element(11_u8))
        .unwrap());
}

#[test]
fn matrices_are_linear_operators_on_vectors_and_blocks() {
    let matrix = Matrix::from_rows(
        Z,
        [
            [Integer::from(1), Integer::from(2), Integer::from(3)],
            [Integer::from(4), Integer::from(5), Integer::from(6)],
        ],
    )
    .unwrap();
    let vector =
        Vector::from_values(Z, [Integer::from(7), Integer::from(8), Integer::from(9)]).unwrap();
    let block = Matrix::identity(Z, 3);

    assert_eq!(matrix.input_dimension(), 3);
    assert_eq!(matrix.output_dimension(), 2);
    assert_eq!(
        matrix.apply(&vector).unwrap().get(0).unwrap(),
        Integer::from(50)
    );
    assert_eq!(
        matrix.apply(&vector).unwrap().get(1).unwrap(),
        Integer::from(122)
    );
    assert_eq!(matrix.apply_block(&block).unwrap(), matrix);
}

#[test]
fn zero_dimensions_and_mismatches_fail_explicitly() {
    let empty = Vector::zeros(Z, 0);
    assert_eq!(empty.dimension(), 0);
    assert_eq!(empty.values().count(), 0);
    assert_eq!(empty.to_column_matrix().shape(), (0, 1));
    assert_eq!(empty.to_row_matrix().shape(), (1, 0));
    assert_eq!(empty.dot(&empty).unwrap(), Integer::zero());

    let two = Vector::zeros(Z, 2);
    let three = Vector::zeros(Z, 3);
    assert!(matches!(
        two.add(&three),
        Err(noether::linalg::LinalgError::DimensionMismatch {
            operation: "vector addition",
            left: 2,
            right: 3,
        })
    ));

    let matrix = Matrix::zeros(Z, 2, 3);
    assert!(matches!(
        matrix.apply(&two),
        Err(noether::linalg::LinalgError::DimensionMismatch {
            operation: "linear-operator application",
            left: 3,
            right: 2,
        })
    ));
}
