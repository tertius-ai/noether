// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact finite-dimensional cohomology algebras of finite simplicial sets.

use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::GradedCommutativePresentation;
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{Integer, NoetherError, Ring, SemanticType, TypeDescription};

use super::cohomology::{
    cohomology_coordinates, connected_presentation, odd_primary_target_degree, terms_in_degree,
    SimplicialCohomologyBasisElement, SimplicialCohomologyProduct, SimplicialCohomologyProductTerm,
};
use super::{AdmissibleCohomologyOperation, SimplicialSet, TopologyError};

static SIMPLICIAL_SET_COHOMOLOGY_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/SimplicialSetCohomologyRing@1",
    rust_path: "noether::topology::SimplicialSetCohomologyRing",
    python_path: Some("noether.topology.SimplicialSetCohomologyRing"),
};

/// Complete cohomology algebra of one finite simplicial set over a finite
/// field.
///
/// Cocycles use the normalized nondegenerate-simplex bases. Products are
/// stored for `left <= right`; the remaining products follow from the Koszul
/// sign rule. Connected nonempty simplicial sets also carry a complete
/// graded-commutative quotient presentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplicialSetCohomologyRing {
    simplicial_set: SimplicialSet,
    field: FiniteField,
    basis: Vec<SimplicialCohomologyBasisElement>,
    products: Vec<SimplicialCohomologyProduct>,
    presentation: Option<GradedCommutativePresentation>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimplicialSetCohomologyRingWire {
    #[serde(rename = "type")]
    type_id: String,
    simplicial_set: SimplicialSet,
    field: FiniteField,
    basis: Vec<SimplicialCohomologyBasisElement>,
    products: Vec<SimplicialCohomologyProduct>,
    presentation: Option<GradedCommutativePresentation>,
}

impl SimplicialSet {
    /// Compute the complete normalized cohomology algebra over `field`,
    /// including canonical cocycles and every basis product.
    pub fn cohomology_ring(
        &self,
        field: FiniteField,
    ) -> Result<SimplicialSetCohomologyRing, TopologyError> {
        SimplicialSetCohomologyRing::compute(self, field)
    }
}

impl SimplicialSetCohomologyRing {
    fn compute(simplicial_set: &SimplicialSet, field: FiniteField) -> Result<Self, TopologyError> {
        let top_degree = usize::try_from(simplicial_set.dimension().max(0)).map_err(|_| {
            TopologyError::DimensionTooLarge {
                dimension: usize::MAX,
            }
        })?;
        let cochains = simplicial_set.cochain_complex(field.clone())?;
        let groups = (0..=top_degree)
            .map(|degree| {
                let signed = i32::try_from(degree)
                    .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
                cochains.cohomology(signed).map_err(Into::into)
            })
            .collect::<Result<Vec<_>, TopologyError>>()?;

        let mut basis = Vec::new();
        let mut degree_offsets = Vec::with_capacity(groups.len());
        for (degree, group) in groups.iter().enumerate() {
            degree_offsets.push(basis.len());
            basis.extend(group.representatives().iter().enumerate().map(
                |(local_index, cocycle)| SimplicialCohomologyBasisElement {
                    id: format!("h{degree}_{local_index}"),
                    degree: u32::try_from(degree).expect("checked cohomological degree fits u32"),
                    cocycle: cocycle.clone(),
                },
            ));
        }

        let mut products = Vec::new();
        for left in 0..basis.len() {
            for right in left..basis.len() {
                let degree = usize::try_from(basis[left].degree)
                    .expect("stored degree fits usize")
                    .checked_add(
                        usize::try_from(basis[right].degree).expect("stored degree fits usize"),
                    )
                    .ok_or(TopologyError::DimensionTooLarge {
                        dimension: usize::MAX,
                    })?;
                let terms = if degree > top_degree {
                    Vec::new()
                } else {
                    let cochain = simplicial_set.cup_product(
                        basis[left].degree as usize,
                        &basis[left].cocycle,
                        basis[right].degree as usize,
                        &basis[right].cocycle,
                    )?;
                    cohomology_coordinates(&groups[degree], &cochain)?
                        .into_iter()
                        .enumerate()
                        .filter_map(|(local_index, coefficient)| {
                            (!coefficient.is_zero()).then_some(SimplicialCohomologyProductTerm {
                                basis: degree_offsets[degree] + local_index,
                                coefficient,
                            })
                        })
                        .collect()
                };
                products.push(SimplicialCohomologyProduct { left, right, terms });
            }
        }

        let presentation = connected_presentation(&field, &basis, &products, top_degree)?;
        let result = Self {
            simplicial_set: simplicial_set.clone(),
            field,
            basis,
            products,
            presentation,
        };
        result.verify_internal()?;
        Ok(result)
    }

    /// Source simplicial set in its fixed normalized bases.
    pub fn simplicial_set(&self) -> &SimplicialSet {
        &self.simplicial_set
    }

    /// Exact coefficient field.
    pub fn field(&self) -> &FiniteField {
        &self.field
    }

    /// Canonical basis, ordered first by degree and then by quotient pivot.
    pub fn basis(&self) -> &[SimplicialCohomologyBasisElement] {
        &self.basis
    }

    /// Ordered basis products with `left <= right`.
    pub fn products(&self) -> &[SimplicialCohomologyProduct] {
        &self.products
    }

    /// Complete connected quotient presentation, when `H^0` is the
    /// coefficient field.
    pub fn presentation(&self) -> Option<&GradedCommutativePresentation> {
        self.presentation.as_ref()
    }

    /// Return a basis product in either order, applying its Koszul sign.
    pub fn product(
        &self,
        left: usize,
        right: usize,
    ) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError> {
        let left_basis = self
            .basis
            .get(left)
            .ok_or_else(|| TopologyError::InvalidCochain {
                reason: format!("cohomology basis index {left} is out of range"),
            })?;
        let right_basis = self
            .basis
            .get(right)
            .ok_or_else(|| TopologyError::InvalidCochain {
                reason: format!("cohomology basis index {right} is out of range"),
            })?;
        let (ordered_left, ordered_right, reversed) = if left <= right {
            (left, right, false)
        } else {
            (right, left, true)
        };
        let product = self
            .products
            .iter()
            .find(|product| product.left == ordered_left && product.right == ordered_right)
            .expect("complete triangular product table");
        let negative = reversed
            && left_basis.degree % 2 == 1
            && right_basis.degree % 2 == 1
            && self.field.characteristic() != &Integer::from(2_u8);
        product
            .terms
            .iter()
            .map(|term| {
                Ok(SimplicialCohomologyProductTerm {
                    basis: term.basis,
                    coefficient: if negative {
                        self.field.neg(&term.coefficient)?
                    } else {
                        term.coefficient.clone()
                    },
                })
            })
            .collect()
    }

    /// Steenrod square of one canonical cohomology-basis class.
    pub fn steenrod_square(
        &self,
        square: usize,
        source_index: usize,
    ) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError> {
        if self.field.characteristic() != &Integer::from(2_u8) {
            return Err(TopologyError::InvalidCochain {
                reason: "Steenrod squares require characteristic 2".to_owned(),
            });
        }
        let source = self
            .basis
            .get(source_index)
            .ok_or_else(|| TopologyError::InvalidCochain {
                reason: format!("cohomology basis index {source_index} is out of range"),
            })?;
        let degree =
            usize::try_from(source.degree).map_err(|_| TopologyError::DimensionTooLarge {
                dimension: usize::MAX,
            })?;
        if square > degree {
            return Ok(Vec::new());
        }
        if square == 0 {
            return Ok(vec![SimplicialCohomologyProductTerm {
                basis: source_index,
                coefficient: self.field.one(),
            }]);
        }
        if square == degree {
            return self.product(source_index, source_index);
        }
        let target_degree = degree
            .checked_add(square)
            .ok_or(TopologyError::DimensionTooLarge { dimension: degree })?;
        let cochain = self
            .simplicial_set
            .steenrod_square(square, degree, &source.cocycle)?;
        let top_degree = usize::try_from(self.simplicial_set.dimension().max(0)).map_err(|_| {
            TopologyError::DimensionTooLarge {
                dimension: target_degree,
            }
        })?;
        if target_degree > top_degree {
            return Ok(Vec::new());
        }
        let target = self
            .simplicial_set
            .cochain_complex(self.field.clone())?
            .cohomology(i32::try_from(target_degree).map_err(|_| {
                TopologyError::DimensionTooLarge {
                    dimension: target_degree,
                }
            })?)?;
        let target_degree_u32 =
            u32::try_from(target_degree).map_err(|_| TopologyError::DimensionTooLarge {
                dimension: target_degree,
            })?;
        let offset = self
            .basis
            .iter()
            .take_while(|element| element.degree < target_degree_u32)
            .count();
        Ok(cohomology_coordinates(&target, &cochain)?
            .into_iter()
            .enumerate()
            .filter_map(|(local, coefficient)| {
                (!coefficient.is_zero()).then_some(SimplicialCohomologyProductTerm {
                    basis: offset + local,
                    coefficient,
                })
            })
            .collect())
    }

    /// Odd-primary Steenrod reduced power `P^power` of one canonical
    /// cohomology-basis class.
    pub fn steenrod_reduced_power(
        &self,
        power: usize,
        source_index: usize,
    ) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError> {
        self.odd_primary_operation(power, source_index, false)
    }

    /// Odd-primary operation `beta P^power` on one canonical
    /// cohomology-basis class.
    pub fn bockstein_reduced_power(
        &self,
        power: usize,
        source_index: usize,
    ) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError> {
        self.odd_primary_operation(power, source_index, true)
    }

    /// Apply one typed admissible Steenrod word to a canonical basis class.
    ///
    /// The rightmost operation acts first, and every intermediate cocycle is
    /// reduced into the canonical normalized cohomology basis.
    pub fn apply_operation(
        &self,
        operation: &AdmissibleCohomologyOperation,
        source_index: usize,
    ) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError> {
        let characteristic = self
            .field
            .characteristic()
            .to_u64_checked()
            .ok_or_else(|| TopologyError::InvalidCohomologyOperation {
                reason: "finite-field characteristic does not fit u64".to_owned(),
            })?;
        if characteristic != operation.prime() {
            return Err(TopologyError::InvalidCohomologyOperation {
                reason: format!(
                    "operation at prime {} cannot act over characteristic {characteristic}",
                    operation.prime()
                ),
            });
        }
        self.basis
            .get(source_index)
            .ok_or_else(|| TopologyError::InvalidCochain {
                reason: format!("cohomology basis index {source_index} is out of range"),
            })?;
        let mut terms = vec![SimplicialCohomologyProductTerm {
            basis: source_index,
            coefficient: self.field.one(),
        }];
        if let Some(sequence) = operation.square_sequence() {
            for square in sequence.iter().rev() {
                let square = usize::try_from(*square).map_err(|_| {
                    TopologyError::InvalidCohomologyOperation {
                        reason: "Steenrod-square index exceeds usize".to_owned(),
                    }
                })?;
                terms = self.apply_to_terms(&terms, |basis| self.steenrod_square(square, basis))?;
            }
            return Ok(terms);
        }

        let bocksteins = operation
            .bocksteins()
            .expect("an odd-primary operation has Bockstein exponents");
        let powers = operation
            .reduced_powers()
            .expect("an odd-primary operation has reduced powers");
        if bocksteins.last().copied().unwrap_or(false) {
            terms = self.apply_to_terms(&terms, |basis| self.bockstein_reduced_power(0, basis))?;
        }
        for index in (0..powers.len()).rev() {
            let power = usize::try_from(powers[index]).map_err(|_| {
                TopologyError::InvalidCohomologyOperation {
                    reason: "reduced-power index exceeds usize".to_owned(),
                }
            })?;
            terms = if bocksteins[index] {
                self.apply_to_terms(&terms, |basis| self.bockstein_reduced_power(power, basis))?
            } else {
                self.apply_to_terms(&terms, |basis| self.steenrod_reduced_power(power, basis))?
            };
        }
        Ok(terms)
    }

    fn apply_to_terms<F>(
        &self,
        source: &[SimplicialCohomologyProductTerm],
        mut operation: F,
    ) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError>
    where
        F: FnMut(usize) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError>,
    {
        let mut target = BTreeMap::<usize, FiniteFieldElement>::new();
        for source_term in source {
            for operation_term in operation(source_term.basis)? {
                let coefficient = self
                    .field
                    .mul(&source_term.coefficient, &operation_term.coefficient)?;
                let entry = target
                    .entry(operation_term.basis)
                    .or_insert_with(|| self.field.zero());
                *entry = self.field.add(entry, &coefficient)?;
            }
        }
        Ok(target
            .into_iter()
            .filter_map(|(basis, coefficient)| {
                (!coefficient.is_zero())
                    .then_some(SimplicialCohomologyProductTerm { basis, coefficient })
            })
            .collect())
    }

    fn odd_primary_operation(
        &self,
        power: usize,
        source_index: usize,
        bockstein: bool,
    ) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError> {
        let source = self
            .basis
            .get(source_index)
            .ok_or_else(|| TopologyError::InvalidCochain {
                reason: format!("cohomology basis index {source_index} is out of range"),
            })?;
        let degree =
            usize::try_from(source.degree).map_err(|_| TopologyError::DimensionTooLarge {
                dimension: usize::MAX,
            })?;
        let cochain = if bockstein {
            self.simplicial_set
                .bockstein_reduced_power(power, degree, &source.cocycle)?
        } else {
            self.simplicial_set
                .steenrod_reduced_power(power, degree, &source.cocycle)?
        };
        let target_degree = odd_primary_target_degree(&self.field, power, degree, bockstein)?;
        let top_degree = usize::try_from(self.simplicial_set.dimension().max(0)).map_err(|_| {
            TopologyError::DimensionTooLarge {
                dimension: target_degree,
            }
        })?;
        if target_degree > top_degree {
            return Ok(Vec::new());
        }
        let target = self
            .simplicial_set
            .cochain_complex(self.field.clone())?
            .cohomology(i32::try_from(target_degree).map_err(|_| {
                TopologyError::DimensionTooLarge {
                    dimension: target_degree,
                }
            })?)?;
        terms_in_degree(&self.basis, target_degree, &target, &cochain)
    }

    /// Recompute and compare every canonical cocycle, product, and
    /// presentation.
    pub fn verify(&self) -> Result<(), TopologyError> {
        let rebuilt = Self::compute(&self.simplicial_set, self.field.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(TopologyError::InvalidCochain {
                reason: "recomputation changed the stored simplicial-set cohomology algebra"
                    .to_owned(),
            })
        }
    }

    /// Serialize the source, field, cocycles, products, and presentation.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly revalidate a simplicial-set cohomology ring.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn verify_internal(&self) -> Result<(), TopologyError> {
        let expected_products = self
            .basis
            .len()
            .checked_mul(self.basis.len().saturating_add(1))
            .and_then(|value| value.checked_div(2))
            .ok_or(TopologyError::InvalidCochain {
                reason: "cohomology product-table size overflowed".to_owned(),
            })?;
        if self.products.len() != expected_products
            || self
                .products
                .iter()
                .any(|product| product.left > product.right || product.right >= self.basis.len())
        {
            return Err(TopologyError::InvalidCochain {
                reason: "cohomology product table is not complete and triangular".to_owned(),
            });
        }
        if let Some(presentation) = &self.presentation {
            presentation.verify()?;
            let top = u32::try_from(self.simplicial_set.dimension().max(0)).map_err(|_| {
                TopologyError::DimensionTooLarge {
                    dimension: usize::MAX,
                }
            })?;
            for degree in 0..=top.saturating_add(1) {
                let expected = self
                    .basis
                    .iter()
                    .filter(|basis| basis.degree == degree)
                    .count();
                if presentation.dimension(degree)? != expected {
                    return Err(TopologyError::InvalidCochain {
                        reason: format!(
                            "connected presentation has the wrong dimension in degree {degree}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

impl Serialize for SimplicialSetCohomologyRing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SimplicialSetCohomologyRingWire {
            type_id: Self::TYPE_ID.to_owned(),
            simplicial_set: self.simplicial_set.clone(),
            field: self.field.clone(),
            basis: self.basis.clone(),
            products: self.products.clone(),
            presentation: self.presentation.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialSetCohomologyRing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SimplicialSetCohomologyRingWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let value = Self {
            simplicial_set: wire.simplicial_set,
            field: wire.field,
            basis: wire.basis,
            products: wire.products,
            presentation: wire.presentation,
        };
        value.verify().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl SemanticType for SimplicialSetCohomologyRing {
    const TYPE_ID: &'static str = "noether/topology/SimplicialSetCohomologyRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &SIMPLICIAL_SET_COHOMOLOGY_RING_DESCRIPTION
    }
}

impl fmt::Display for SimplicialSetCohomologyRing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "H^*({}, {}; basis={})",
            self.simplicial_set,
            self.field,
            self.basis.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::SimplicialFace;

    fn mod_three_moore_space() -> SimplicialSet {
        SimplicialSet::new(vec![
            vec![vec![]],
            vec![
                vec![SimplicialFace::Cell(0), SimplicialFace::Cell(0)],
                vec![SimplicialFace::Cell(0), SimplicialFace::Cell(0)],
            ],
            vec![
                vec![
                    SimplicialFace::Cell(0),
                    SimplicialFace::Cell(1),
                    SimplicialFace::Cell(0),
                ],
                vec![
                    SimplicialFace::Cell(0),
                    SimplicialFace::degenerate(0, 0, vec![0, 0]),
                    SimplicialFace::Cell(1),
                ],
            ],
        ])
        .unwrap()
    }

    #[test]
    fn minimal_circle_has_complete_odd_characteristic_ring() {
        let circle = SimplicialSet::sphere(1);
        let ring = circle
            .cohomology_ring(FiniteField::prime(3_u8).unwrap())
            .unwrap();
        assert_eq!(
            ring.basis()
                .iter()
                .map(SimplicialCohomologyBasisElement::degree)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(ring.product(1, 1).unwrap().is_empty());
        assert_eq!(
            ring.presentation().unwrap().hilbert_function(2).unwrap(),
            vec![1, 1, 0]
        );
        ring.verify().unwrap();
    }

    #[test]
    fn minimal_sphere_has_normalized_mod_two_operations_and_round_trip() {
        let sphere = SimplicialSet::sphere(2);
        let ring = sphere
            .cohomology_ring(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(
            ring.basis()
                .iter()
                .map(SimplicialCohomologyBasisElement::degree)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(ring.steenrod_square(0, 1).unwrap()[0].basis(), 1);
        assert!(ring.steenrod_square(2, 1).unwrap().is_empty());
        let restored = SimplicialSetCohomologyRing::from_json(&ring.to_json().unwrap()).unwrap();
        assert_eq!(restored, ring);
    }

    #[test]
    fn moore_space_ring_records_nonzero_mod_three_bockstein() {
        let ring = mod_three_moore_space()
            .cohomology_ring(FiniteField::prime(3_u8).unwrap())
            .unwrap();
        assert_eq!(
            ring.basis()
                .iter()
                .map(SimplicialCohomologyBasisElement::degree)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(ring.steenrod_reduced_power(0, 1).unwrap()[0].basis(), 1);
        assert_eq!(ring.bockstein_reduced_power(0, 1).unwrap()[0].basis(), 2);
        assert_eq!(
            ring.apply_operation(
                &AdmissibleCohomologyOperation::bockstein_reduced_power(3, 0).unwrap(),
                1,
            )
            .unwrap()[0]
                .basis(),
            2
        );
    }
}
