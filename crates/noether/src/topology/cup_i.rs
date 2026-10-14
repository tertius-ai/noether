// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Normalized cup-i products and Steenrod squares at the prime two.

use crate::linalg::{LinearOperator, Vector};
use crate::numthy::FiniteField;
use crate::{Integer, Ring};

use super::{Simplex, SimplicialComplex, SimplicialSet, TopologyError};

const CUP_I_METHOD: &str = "medina-mardones-subset-formula/v1";

impl SimplicialComplex {
    /// Chain-level cup-`i` product on normalized cochains in characteristic 2.
    ///
    /// This uses the subset formula of Medina-Mardones. Inputs need not be
    /// cocycles. The output has degree `left_degree + right_degree - i`.
    pub fn cup_i(
        &self,
        cup_index: usize,
        left_degree: usize,
        left: &Vector<FiniteField>,
        right_degree: usize,
        right: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        cup_i(
            cup_index,
            CochainInput::new(left_degree, left, self.count(left_degree)),
            CochainInput::new(right_degree, right, self.count(right_degree)),
            |output_degree, simplex, removed| {
                let value = self.simplices(output_degree).get(simplex)?;
                let face = Simplex::new(
                    value
                        .vertices()
                        .iter()
                        .enumerate()
                        .filter_map(|(position, vertex)| {
                            removed.binary_search(&position).is_err().then_some(*vertex)
                        })
                        .collect(),
                );
                self.position(&face)
            },
            self.count(
                left_degree
                    .checked_add(right_degree)
                    .and_then(|degree| degree.checked_sub(cup_index))
                    .unwrap_or(usize::MAX),
            ),
        )
    }

    /// Steenrod square of a checked normalized cocycle over a field of
    /// characteristic 2.
    pub fn steenrod_square(
        &self,
        square: usize,
        degree: usize,
        cocycle: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        let upper = degree
            .checked_add(1)
            .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
        let target = degree
            .checked_add(square)
            .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
        require_cocycle(
            cocycle,
            degree,
            self.count(degree),
            self.boundary_matrix(cocycle.ring().clone(), upper)?,
        )?;
        steenrod_square(
            square,
            degree,
            cocycle,
            |cup_index| self.cup_i(cup_index, degree, cocycle, degree, cocycle),
            self.count(target),
        )
    }
}

impl SimplicialSet {
    /// Chain-level cup-`i` product on normalized cochains in characteristic 2.
    ///
    /// Degenerate iterated faces evaluate to zero in the normalized complex.
    /// Inputs need not be cocycles. The output has degree
    /// `left_degree + right_degree - i`.
    pub fn cup_i(
        &self,
        cup_index: usize,
        left_degree: usize,
        left: &Vector<FiniteField>,
        right_degree: usize,
        right: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        cup_i(
            cup_index,
            CochainInput::new(left_degree, left, self.count(left_degree)),
            CochainInput::new(right_degree, right, self.count(right_degree)),
            |output_degree, simplex, removed| {
                self.normalized_iterated_face(output_degree, simplex, removed)
            },
            self.count(
                left_degree
                    .checked_add(right_degree)
                    .and_then(|degree| degree.checked_sub(cup_index))
                    .unwrap_or(usize::MAX),
            ),
        )
    }

    /// Steenrod square of a checked normalized cocycle over a field of
    /// characteristic 2.
    pub fn steenrod_square(
        &self,
        square: usize,
        degree: usize,
        cocycle: &Vector<FiniteField>,
    ) -> Result<Vector<FiniteField>, TopologyError> {
        let upper = degree
            .checked_add(1)
            .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
        let target = degree
            .checked_add(square)
            .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
        require_cocycle(
            cocycle,
            degree,
            self.count(degree),
            self.boundary_matrix(cocycle.ring().clone(), upper)?,
        )?;
        steenrod_square(
            square,
            degree,
            cocycle,
            |cup_index| self.cup_i(cup_index, degree, cocycle, degree, cocycle),
            self.count(target),
        )
    }
}

struct CochainInput<'a> {
    degree: usize,
    value: &'a Vector<FiniteField>,
    expected: usize,
}

impl<'a> CochainInput<'a> {
    const fn new(degree: usize, value: &'a Vector<FiniteField>, expected: usize) -> Self {
        Self {
            degree,
            value,
            expected,
        }
    }
}

fn cup_i<F>(
    cup_index: usize,
    left: CochainInput<'_>,
    right: CochainInput<'_>,
    mut face_index: F,
    output_count: usize,
) -> Result<Vector<FiniteField>, TopologyError>
where
    F: FnMut(usize, usize, &[usize]) -> Option<usize>,
{
    require_characteristic_two(left.value.ring())?;
    if left.value.ring() != right.value.ring() {
        return Err(invalid("cup-i inputs use different coefficient contexts"));
    }
    if left.value.dimension() != left.expected {
        return Err(invalid(format!(
            "left degree-{} cochain has {} coefficients; expected {}",
            left.degree,
            left.value.dimension(),
            left.expected
        )));
    }
    if right.value.dimension() != right.expected {
        return Err(invalid(format!(
            "right degree-{} cochain has {} coefficients; expected {}",
            right.degree,
            right.value.dimension(),
            right.expected
        )));
    }
    let total_degree =
        left.degree
            .checked_add(right.degree)
            .ok_or(TopologyError::DimensionTooLarge {
                dimension: usize::MAX,
            })?;
    let output_degree = total_degree.checked_sub(cup_index).ok_or_else(|| {
        invalid(format!(
            "cup index {cup_index} exceeds total input degree {total_degree}"
        ))
    })?;
    let Some(subset_size) = output_degree.checked_sub(cup_index) else {
        return Ok(Vector::zeros(left.value.ring().clone(), output_count));
    };

    let mut subsets = Vec::new();
    combinations(
        output_degree.saturating_add(1),
        subset_size,
        0,
        &mut Vec::new(),
        &mut subsets,
    );
    let field = left.value.ring().clone();
    let mut output = Vec::with_capacity(output_count);
    for simplex in 0..output_count {
        let mut value = field.zero();
        for subset in &subsets {
            let (left_removed, right_removed) = parity_partition(subset);
            if output_degree.saturating_sub(left_removed.len()) != left.degree
                || output_degree.saturating_sub(right_removed.len()) != right.degree
            {
                continue;
            }
            let Some(left_simplex) = face_index(output_degree, simplex, &left_removed) else {
                continue;
            };
            let Some(right_simplex) = face_index(output_degree, simplex, &right_removed) else {
                continue;
            };
            let left_value = left
                .value
                .get(left_simplex)
                .map_err(|_| invalid("cup-i left iterated face escaped its checked cochain"))?;
            let right_value = right
                .value
                .get(right_simplex)
                .map_err(|_| invalid("cup-i right iterated face escaped its checked cochain"))?;
            let product = field.mul(&left_value, &right_value)?;
            value = field.add(&value, &product)?;
        }
        output.push(value);
    }
    Vector::from_values(field, output).map_err(Into::into)
}

fn require_cocycle(
    cocycle: &Vector<FiniteField>,
    degree: usize,
    expected: usize,
    upper_boundary: crate::linalg::Matrix<FiniteField>,
) -> Result<(), TopologyError> {
    require_characteristic_two(cocycle.ring())?;
    if cocycle.dimension() != expected {
        return Err(invalid(format!(
            "degree-{degree} cocycle has {} coefficients; expected {expected}",
            cocycle.dimension()
        )));
    }
    let coboundary = upper_boundary.transpose().apply(cocycle)?;
    if coboundary
        .values()
        .any(|coefficient| !coefficient.is_zero())
    {
        return Err(invalid("Steenrod-square input is not a cocycle"));
    }
    Ok(())
}

fn steenrod_square<F>(
    square: usize,
    degree: usize,
    cocycle: &Vector<FiniteField>,
    cup: F,
    target_count: usize,
) -> Result<Vector<FiniteField>, TopologyError>
where
    F: FnOnce(usize) -> Result<Vector<FiniteField>, TopologyError>,
{
    if square > degree {
        return Ok(Vector::zeros(cocycle.ring().clone(), target_count));
    }
    cup(degree - square)
}

fn require_characteristic_two(field: &FiniteField) -> Result<(), TopologyError> {
    if field.characteristic() != &Integer::from(2) {
        return Err(invalid(format!(
            "cup-i method {CUP_I_METHOD} requires characteristic 2"
        )));
    }
    Ok(())
}

fn invalid(reason: impl Into<String>) -> TopologyError {
    TopologyError::InvalidCochain {
        reason: reason.into(),
    }
}

fn parity_partition(subset: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut even = Vec::new();
    let mut odd = Vec::new();
    for (position, &value) in subset.iter().enumerate() {
        if value % 2 == (position + 1) % 2 {
            even.push(value);
        } else {
            odd.push(value);
        }
    }
    (even, odd)
}

fn combinations(
    universe: usize,
    size: usize,
    next: usize,
    current: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
) {
    if current.len() == size {
        output.push(current.clone());
        return;
    }
    let needed = size - current.len();
    if universe.saturating_sub(next) < needed {
        return;
    }
    for value in next..=universe - needed {
        current.push(value);
        combinations(universe, size, value + 1, current, output);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numthy::FiniteField;
    use crate::topology::SimplicialFace;

    fn basis(field: &FiniteField, dimension: usize, selected: usize) -> Vector<FiniteField> {
        Vector::from_values(
            field.clone(),
            (0..dimension).map(|index| {
                if index == selected {
                    field.one()
                } else {
                    field.zero()
                }
            }),
        )
        .unwrap()
    }

    #[test]
    fn cup_zero_is_alexander_whitney() {
        let complex = SimplicialComplex::from_facets(&[vec![0, 1, 2]]);
        let field = FiniteField::prime(2_u8).unwrap();
        let left = basis(&field, complex.count(1), 0);
        let right = basis(&field, complex.count(1), 2);
        assert_eq!(
            complex.cup_i(0, 1, &left, 1, &right).unwrap(),
            complex.cup_product(1, &left, 1, &right).unwrap()
        );
    }

    #[test]
    fn simplicial_set_cup_zero_is_normalized_alexander_whitney() {
        let simplex = SimplicialSet::new(vec![
            vec![vec![], vec![], vec![]],
            vec![
                vec![SimplicialFace::Cell(1), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(2), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(2), SimplicialFace::Cell(1)],
            ],
            vec![vec![
                SimplicialFace::Cell(2),
                SimplicialFace::Cell(1),
                SimplicialFace::Cell(0),
            ]],
        ])
        .unwrap();
        let field = FiniteField::prime(2_u8).unwrap();
        let left = basis(&field, simplex.count(1), 0);
        let right = basis(&field, simplex.count(1), 2);
        assert_eq!(
            simplex.cup_i(0, 1, &left, 1, &right).unwrap(),
            simplex.cup_product(1, &left, 1, &right).unwrap()
        );
    }

    #[test]
    fn top_cup_index_is_the_diagonal() {
        let complex = SimplicialComplex::from_facets(&[vec![0, 1, 2]]);
        let field = FiniteField::prime(2_u8).unwrap();
        let cochain = basis(&field, complex.count(2), 0);
        assert_eq!(complex.cup_i(2, 2, &cochain, 2, &cochain).unwrap(), cochain);
        assert_eq!(complex.steenrod_square(0, 2, &cochain).unwrap(), cochain);
    }

    #[test]
    fn normalized_simplicial_set_faces_discard_degeneracies() {
        let sphere = SimplicialSet::sphere(2);
        let field = FiniteField::prime(2_u8).unwrap();
        let top = basis(&field, 1, 0);
        let square_zero = sphere.steenrod_square(0, 2, &top).unwrap();
        assert_eq!(square_zero, top);
        assert!(sphere
            .steenrod_square(1, 2, &top)
            .unwrap()
            .values()
            .all(|value| value.is_zero()));
    }

    #[test]
    fn cup_i_satisfies_the_defining_chain_homotopy_identity() {
        let complex = SimplicialComplex::from_facets(&[vec![0, 1, 2, 3]]);
        let field = FiniteField::prime(2_u8).unwrap();
        for left_degree in 0..=2 {
            for right_degree in 0..=2 {
                let left = basis(&field, complex.count(left_degree), 0);
                let right = basis(
                    &field,
                    complex.count(right_degree),
                    complex.count(right_degree) - 1,
                );
                for cup_index in 0..=left_degree.min(right_degree) {
                    let output_degree = left_degree + right_degree - cup_index;
                    if output_degree + 1 > 3 {
                        continue;
                    }
                    let product = complex
                        .cup_i(cup_index, left_degree, &left, right_degree, &right)
                        .unwrap();
                    let lhs = complex
                        .boundary_matrix(field.clone(), output_degree + 1)
                        .unwrap()
                        .transpose()
                        .apply(&product)
                        .unwrap();
                    let delta_left = complex
                        .boundary_matrix(field.clone(), left_degree + 1)
                        .unwrap()
                        .transpose()
                        .apply(&left)
                        .unwrap();
                    let delta_right = complex
                        .boundary_matrix(field.clone(), right_degree + 1)
                        .unwrap()
                        .transpose()
                        .apply(&right)
                        .unwrap();
                    let mut rhs = complex
                        .cup_i(
                            cup_index,
                            left_degree + 1,
                            &delta_left,
                            right_degree,
                            &right,
                        )
                        .unwrap()
                        .add(
                            &complex
                                .cup_i(
                                    cup_index,
                                    left_degree,
                                    &left,
                                    right_degree + 1,
                                    &delta_right,
                                )
                                .unwrap(),
                        )
                        .unwrap();
                    if let Some(lower) = cup_index.checked_sub(1) {
                        rhs = rhs
                            .add(
                                &complex
                                    .cup_i(lower, left_degree, &left, right_degree, &right)
                                    .unwrap(),
                            )
                            .unwrap()
                            .add(
                                &complex
                                    .cup_i(lower, right_degree, &right, left_degree, &left)
                                    .unwrap(),
                            )
                            .unwrap();
                    }
                    assert_eq!(
                        lhs, rhs,
                        "degrees ({left_degree},{right_degree}), i={cup_index}"
                    );
                }
            }
        }
    }

    #[test]
    fn cup_i_rejects_odd_characteristic() {
        let complex = SimplicialComplex::from_facets(&[vec![0, 1]]);
        let field = FiniteField::prime(3_u8).unwrap();
        let vertex = basis(&field, complex.count(0), 0);
        assert!(matches!(
            complex.cup_i(0, 0, &vertex, 0, &vertex),
            Err(TopologyError::InvalidCochain { .. })
        ));
    }
}
