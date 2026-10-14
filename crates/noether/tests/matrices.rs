// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use noether::linalg::{LinalgError, Matrix};
use noether::{
    ContextMismatchError, Integer, NoetherError, Rational, Ring, SemanticType, TypeDescription, Q,
    Z,
};

static MOD_ELEMENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "test/ModElement@1",
    rust_path: "matrices::ModElement",
    python_path: None,
};

static MOD_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "test/ModRing@1",
    rust_path: "matrices::ModRing",
    python_path: None,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModElement(u8);

impl SemanticType for ModElement {
    const TYPE_ID: &'static str = "test/ModElement@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(vec![self.0])
    }

    fn describe() -> &'static TypeDescription {
        &MOD_ELEMENT_DESCRIPTION
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModRing(u8);

impl SemanticType for ModRing {
    const TYPE_ID: &'static str = "test/ModRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(vec![self.0])
    }

    fn describe() -> &'static TypeDescription {
        &MOD_RING_DESCRIPTION
    }
}

impl Ring for ModRing {
    type Element = ModElement;

    fn contains(&self, value: &Self::Element) -> bool {
        self.0 >= 2 && value.0 < self.0
    }

    fn zero(&self) -> Self::Element {
        ModElement(0)
    }

    fn one(&self) -> Self::Element {
        ModElement(1)
    }

    fn add(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(ModElement(
            ((a.0 as u16 + b.0 as u16) % self.0 as u16) as u8,
        ))
    }

    fn sub(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(ModElement((a.0 + self.0 - b.0) % self.0))
    }

    fn mul(
        &self,
        a: &Self::Element,
        b: &Self::Element,
    ) -> Result<Self::Element, ContextMismatchError> {
        Ok(ModElement(
            ((a.0 as u16 * b.0 as u16) % self.0 as u16) as u8,
        ))
    }

    fn neg(&self, value: &Self::Element) -> Result<Self::Element, ContextMismatchError> {
        Ok(ModElement((self.0 - value.0) % self.0))
    }
}

fn m(value: u8) -> ModElement {
    ModElement(value)
}

fn values<R: Ring>(matrix: &Matrix<R>) -> Vec<Vec<R::Element>> {
    matrix.rows().collect()
}

#[test]
fn every_ring_receives_the_generic_matrix_operations() {
    let ring = ModRing(7);
    let a = Matrix::from_rows(ring.clone(), [[m(1), m(2), m(3)], [m(4), m(5), m(6)]]).unwrap();
    let b = Matrix::from_rows(ring.clone(), [[m(6), m(5)], [m(4), m(3)], [m(2), m(1)]]).unwrap();

    assert_eq!(a.shape(), (2, 3));
    assert_eq!(a.ring(), &ring);
    assert_eq!(a.get(1, 2).unwrap(), m(6));
    assert_eq!(
        values(&a.transpose()),
        vec![vec![m(1), m(4)], vec![m(2), m(5)], vec![m(3), m(6)]]
    );

    let product = a.matmul(&b).unwrap();
    assert_eq!(values(&product), vec![vec![m(6), m(0)], vec![m(0), m(6)]]);

    let c = Matrix::from_rows(ring.clone(), [[m(6), m(1), m(0)], [m(2), m(4), m(3)]]).unwrap();
    assert_eq!(
        values(&(&a + &c).unwrap()),
        vec![vec![m(0), m(3), m(3)], vec![m(6), m(2), m(2)]]
    );
    assert_eq!(
        values(&(&a - &c).unwrap()),
        vec![vec![m(2), m(1), m(3)], vec![m(2), m(1), m(3)]]
    );
    assert_eq!(
        values(&(-&a).unwrap()),
        vec![vec![m(6), m(5), m(4)], vec![m(3), m(2), m(1)]]
    );
    assert_eq!(
        values(&(&a * &m(3)).unwrap()),
        vec![vec![m(3), m(6), m(2)], vec![m(5), m(1), m(4)]]
    );
}

#[test]
fn integer_matrices_keep_arbitrary_precision_exact() {
    let large = "100000000000000000000000000000000000003"
        .parse::<Integer>()
        .unwrap();
    let a = Matrix::from_rows(
        Z,
        [
            [large.clone(), Integer::from(2)],
            [Integer::from(-3), Integer::from(4)],
        ],
    )
    .unwrap();
    let b = Matrix::from_rows(
        Z,
        [
            [Integer::one(), Integer::from(5)],
            [Integer::from(6), Integer::from(-2)],
        ],
    )
    .unwrap();

    let product = a.matmul(&b).unwrap();
    assert_eq!(product.get(0, 0).unwrap(), &large + &Integer::from(12));
    assert_eq!(
        product.get(0, 1).unwrap(),
        &(&large * &Integer::from(5)) - &Integer::from(4)
    );
    assert_eq!(product.get(1, 0).unwrap(), Integer::from(21));
    assert_eq!(product.get(1, 1).unwrap(), Integer::from(-23));

    let scaled = a.scalar_mul(&Integer::from(-2)).unwrap();
    assert_eq!(scaled.get(0, 0).unwrap(), -(&large * &Integer::from(2)));
    assert_eq!(a.transpose().transpose(), a);
}

#[test]
fn rational_matrices_use_exact_field_arithmetic() {
    let a = Matrix::from_rows(
        Q,
        [
            ["1/2".parse::<Rational>().unwrap(), "1/3".parse().unwrap()],
            [Rational::from(2), "-3/4".parse().unwrap()],
        ],
    )
    .unwrap();
    let b = Matrix::from_rows(
        Q,
        [
            [Rational::from(2), Rational::from(3)],
            [Rational::from(4), Rational::from(5)],
        ],
    )
    .unwrap();

    let product = a.matmul(&b).unwrap();
    assert_eq!(product.get(0, 0).unwrap().to_string(), "7/3");
    assert_eq!(product.get(0, 1).unwrap().to_string(), "19/6");
    assert_eq!(product.get(1, 0).unwrap().to_string(), "1");
    assert_eq!(product.get(1, 1).unwrap().to_string(), "9/4");

    let scaled = a.scalar_mul(&"3/2".parse().unwrap()).unwrap();
    assert_eq!(scaled.get(0, 0).unwrap().to_string(), "3/4");
    assert_eq!(values(&(&a - &a).unwrap()), values(&Matrix::zeros(Q, 2, 2)));
}

#[test]
fn construction_and_operations_reject_invalid_shapes_and_contexts() {
    assert!(matches!(
        Matrix::from_rows(ModRing(7), [vec![m(1)], vec![m(1), m(2)]]),
        Err(LinalgError::RaggedRows { row: 1, .. })
    ));
    assert!(matches!(
        Matrix::from_rows(ModRing(5), [[m(6)]]),
        Err(LinalgError::ElementOutsideContext { row: 0, col: 0 })
    ));

    let two_by_two = Matrix::identity(ModRing(7), 2);
    let three_by_three = Matrix::identity(ModRing(7), 3);
    assert!(matches!(
        two_by_two.add(&three_by_three),
        Err(LinalgError::ShapeMismatch { .. })
    ));
    assert!(two_by_two.get(2, 0).is_err());

    let other_context = Matrix::identity(ModRing(5), 2);
    assert!(matches!(
        two_by_two.add(&other_context),
        Err(LinalgError::ContextMismatch(_))
    ));
}

#[test]
fn zero_dimensions_identity_serialization_and_repr_are_stable() {
    let left = Matrix::zeros(Z, 2, 0);
    let right = Matrix::zeros(Z, 0, 3);
    let product = left.matmul(&right).unwrap();
    assert_eq!(product.shape(), (2, 3));
    assert_eq!(values(&product), vec![vec![Integer::zero(); 3]; 2]);

    let matrix = Matrix::from_rows(Z, [[Integer::from(1), Integer::from(2)]]).unwrap();
    assert_eq!(matrix.matmul(&Matrix::identity(Z, 2)).unwrap(), matrix);
    assert_eq!(
        matrix.canonical_bytes().unwrap(),
        matrix.clone().canonical_bytes().unwrap()
    );
    assert_ne!(
        matrix.canonical_bytes().unwrap(),
        Matrix::zeros(Z, 2, 1).canonical_bytes().unwrap()
    );
    assert_eq!(
        Matrix::<noether::IntegerRing>::TYPE_ID,
        "noether/linalg/Matrix@1"
    );
    assert_eq!(
        Matrix::<noether::IntegerRing>::describe().python_path,
        Some("noether.linalg.Matrix")
    );
    assert_eq!(
        matrix.to_string(),
        "Matrix<noether/root/IntegerRing@1>(1x2)"
    );

    assert_eq!(
        Matrix::<noether::IntegerRing>::from_json(&matrix.to_json().unwrap()).unwrap(),
        matrix
    );
    let zero_rows = Matrix::zeros(Z, 0, 3);
    assert_eq!(
        Matrix::<noether::IntegerRing>::from_json(&zero_rows.to_json().unwrap()).unwrap(),
        zero_rows
    );
    let mut wrong_shape: serde_json::Value =
        serde_json::from_str(&matrix.to_json().unwrap()).unwrap();
    wrong_shape["shape"] = serde_json::json!([2, 2]);
    assert!(Matrix::<noether::IntegerRing>::from_json(&wrong_shape.to_string()).is_err());
}

#[test]
fn matrices_are_safe_to_share_between_threads() {
    fn assert_thread_safe<T: Send + Sync>() {}
    assert_thread_safe::<Matrix<noether::IntegerRing>>();
    assert_thread_safe::<Matrix<noether::RationalField>>();

    let matrix = Matrix::identity(Z, 4);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| assert_eq!(matrix.matmul(&matrix).unwrap(), matrix));
        }
    });
}
