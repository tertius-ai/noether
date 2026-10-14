// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact finite-dimensional simplicial cohomology algebras.

use core::fmt;
use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{
    GradedCommutativePolynomialRing, GradedCommutativePresentation, GradedGenerator,
};
use crate::linalg::{Matrix, Vector};
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{Integer, NoetherError, Ring, SemanticType, TypeDescription};

use super::{AdmissibleCohomologyOperation, SimplicialComplex, TopologyError};

static SIMPLICIAL_COHOMOLOGY_RING_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/SimplicialCohomologyRing@1",
    rust_path: "noether::topology::SimplicialCohomologyRing",
    python_path: Some("noether.topology.SimplicialCohomologyRing"),
};

/// One deterministic cohomology-basis class with its canonical cocycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimplicialCohomologyBasisElement {
    pub(super) id: String,
    pub(super) degree: u32,
    pub(super) cocycle: Vector<FiniteField>,
}

impl SimplicialCohomologyBasisElement {
    /// Stable degree-local identifier such as `h1_0`.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Cohomological degree.
    pub fn degree(&self) -> u32 {
        self.degree
    }

    /// Canonical cocycle in the lexicographic simplex basis.
    pub fn cocycle(&self) -> &Vector<FiniteField> {
        &self.cocycle
    }
}

/// One nonzero basis coefficient in a cup product.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimplicialCohomologyProductTerm {
    pub(super) basis: usize,
    pub(super) coefficient: FiniteFieldElement,
}

impl SimplicialCohomologyProductTerm {
    /// Global basis index in [`SimplicialCohomologyRing::basis`].
    pub fn basis(&self) -> usize {
        self.basis
    }

    /// Exact nonzero coefficient.
    pub fn coefficient(&self) -> &FiniteFieldElement {
        &self.coefficient
    }
}

/// Cup product of an ordered pair of basis classes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimplicialCohomologyProduct {
    pub(super) left: usize,
    pub(super) right: usize,
    pub(super) terms: Vec<SimplicialCohomologyProductTerm>,
}

impl SimplicialCohomologyProduct {
    /// Left global basis index.
    pub fn left(&self) -> usize {
        self.left
    }

    /// Right global basis index.
    pub fn right(&self) -> usize {
        self.right
    }

    /// Sparse result in the canonical cohomology basis.
    pub fn terms(&self) -> &[SimplicialCohomologyProductTerm] {
        &self.terms
    }
}

/// Complete cohomology algebra of one finite simplicial complex over a finite
/// field.
///
/// The basis is deterministic and carries chain-level cocycles. Products are
/// stored for `left <= right`; the remaining products follow from the Koszul
/// sign rule. A connected nonempty complex also carries a complete quotient
/// presentation. For a disconnected complex the full multiplication table is
/// still available, but its degree-zero algebra is not the coefficient field,
/// so no misleading connected presentation is manufactured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimplicialCohomologyRing {
    complex: SimplicialComplex,
    field: FiniteField,
    basis: Vec<SimplicialCohomologyBasisElement>,
    products: Vec<SimplicialCohomologyProduct>,
    presentation: Option<GradedCommutativePresentation>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimplicialCohomologyRingWire {
    #[serde(rename = "type")]
    type_id: String,
    complex: SimplicialComplex,
    field: FiniteField,
    basis: Vec<SimplicialCohomologyBasisElement>,
    products: Vec<SimplicialCohomologyProduct>,
    presentation: Option<GradedCommutativePresentation>,
}

impl SimplicialComplex {
    /// Compute the complete finite-dimensional cohomology algebra over
    /// `field`, including canonical cocycles and every basis product.
    pub fn cohomology_ring(
        &self,
        field: FiniteField,
    ) -> Result<SimplicialCohomologyRing, TopologyError> {
        SimplicialCohomologyRing::compute(self, field)
    }
}

impl SimplicialCohomologyRing {
    fn compute(complex: &SimplicialComplex, field: FiniteField) -> Result<Self, TopologyError> {
        let top_degree = usize::try_from(complex.dimension().max(0)).map_err(|_| {
            TopologyError::DimensionTooLarge {
                dimension: usize::MAX,
            }
        })?;
        let cochains = complex.cochain_complex(field.clone())?;
        let groups = (0..=top_degree)
            .map(|degree| {
                let degree = i32::try_from(degree)
                    .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
                cochains.cohomology(degree).map_err(Into::into)
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
                    let cochain = complex.cup_product(
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
            complex: complex.clone(),
            field,
            basis,
            products,
            presentation,
        };
        result.verify_internal()?;
        Ok(result)
    }

    /// Source complex in its fixed simplex bases.
    pub fn complex(&self) -> &SimplicialComplex {
        &self.complex
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

    /// Complete connected quotient presentation, when `H^0` is exactly the
    /// coefficient field.
    pub fn presentation(&self) -> Option<&GradedCommutativePresentation> {
        self.presentation.as_ref()
    }

    /// Return a basis product in either input order, applying its Koszul sign.
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
    ///
    /// The operation is computed on the stored canonical cocycle with the
    /// normalized cup-i formula and then reduced into the canonical
    /// cohomology basis. It is available exactly in characteristic 2.
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
            .complex
            .steenrod_square(square, degree, &source.cocycle)?;
        let top_degree = usize::try_from(self.complex.dimension().max(0)).map_err(|_| {
            TopologyError::DimensionTooLarge {
                dimension: target_degree,
            }
        })?;
        if target_degree > top_degree {
            return Ok(Vec::new());
        }
        let target = self
            .complex
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
    /// The rightmost operation acts first. Every intermediate result is
    /// reduced into the retained canonical basis, so the returned coordinates
    /// are independently checkable by replaying the single-operation methods.
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
            self.complex
                .bockstein_reduced_power(power, degree, &source.cocycle)?
        } else {
            self.complex
                .steenrod_reduced_power(power, degree, &source.cocycle)?
        };
        let target_degree = odd_primary_target_degree(&self.field, power, degree, bockstein)?;
        let top_degree = usize::try_from(self.complex.dimension().max(0)).map_err(|_| {
            TopologyError::DimensionTooLarge {
                dimension: target_degree,
            }
        })?;
        if target_degree > top_degree {
            return Ok(Vec::new());
        }
        let target = self
            .complex
            .cochain_complex(self.field.clone())?
            .cohomology(i32::try_from(target_degree).map_err(|_| {
                TopologyError::DimensionTooLarge {
                    dimension: target_degree,
                }
            })?)?;
        terms_in_degree(&self.basis, target_degree, &target, &cochain)
    }

    /// Recompute the algebra from its source and compare every cocycle,
    /// product, and presentation.
    pub fn verify(&self) -> Result<(), TopologyError> {
        let rebuilt = Self::compute(&self.complex, self.field.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(TopologyError::InvalidCochain {
                reason: "recomputation changed the stored cohomology algebra".to_owned(),
            })
        }
    }

    /// Serialize the source, coefficient context, cocycles, products, and
    /// connected presentation.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly revalidate a cohomology algebra.
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
            let top = u32::try_from(self.complex.dimension().max(0)).map_err(|_| {
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

impl Serialize for SimplicialCohomologyRing {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SimplicialCohomologyRingWire {
            type_id: Self::TYPE_ID.to_owned(),
            complex: self.complex.clone(),
            field: self.field.clone(),
            basis: self.basis.clone(),
            products: self.products.clone(),
            presentation: self.presentation.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SimplicialCohomologyRing {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SimplicialCohomologyRingWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let value = Self {
            complex: wire.complex,
            field: wire.field,
            basis: wire.basis,
            products: wire.products,
            presentation: wire.presentation,
        };
        value.verify().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

impl SemanticType for SimplicialCohomologyRing {
    const TYPE_ID: &'static str = "noether/topology/SimplicialCohomologyRing@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &SIMPLICIAL_COHOMOLOGY_RING_DESCRIPTION
    }
}

impl fmt::Display for SimplicialCohomologyRing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "H^*({}, {}; basis={})",
            self.complex,
            self.field,
            self.basis.len()
        )
    }
}

pub(super) fn odd_primary_target_degree(
    field: &FiniteField,
    power: usize,
    degree: usize,
    bockstein: bool,
) -> Result<usize, TopologyError> {
    if field.degree() != 1 || field.characteristic() == &Integer::from(2_u8) {
        return Err(TopologyError::InvalidCochain {
            reason: "odd-primary Steenrod operations require a prime field of odd characteristic"
                .to_owned(),
        });
    }
    let prime =
        field
            .characteristic()
            .to_u64_checked()
            .ok_or_else(|| TopologyError::InvalidCochain {
                reason: "finite-field characteristic does not fit u64".to_owned(),
            })?;
    let prime_minus_one = usize::try_from(prime - 1)
        .map_err(|_| TopologyError::DimensionTooLarge { dimension: degree })?;
    power
        .checked_mul(2)
        .and_then(|value| value.checked_mul(prime_minus_one))
        .and_then(|value| value.checked_add(usize::from(bockstein)))
        .and_then(|shift| degree.checked_add(shift))
        .ok_or(TopologyError::DimensionTooLarge { dimension: degree })
}

pub(super) fn terms_in_degree(
    basis: &[SimplicialCohomologyBasisElement],
    target_degree: usize,
    target: &crate::homalg::HomologyGroup<FiniteField>,
    cocycle: &Vector<FiniteField>,
) -> Result<Vec<SimplicialCohomologyProductTerm>, TopologyError> {
    let target_degree_u32 =
        u32::try_from(target_degree).map_err(|_| TopologyError::DimensionTooLarge {
            dimension: target_degree,
        })?;
    let offset = basis
        .iter()
        .take_while(|element| element.degree < target_degree_u32)
        .count();
    Ok(cohomology_coordinates(target, cocycle)?
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

pub(super) fn cohomology_coordinates(
    group: &crate::homalg::HomologyGroup<FiniteField>,
    cocycle: &Vector<FiniteField>,
) -> Result<Vec<FiniteFieldElement>, TopologyError> {
    let boundary_count = group.boundaries().basis().len();
    let mut columns = group.boundaries().basis().to_vec();
    columns.extend(group.representatives().iter().cloned());
    if columns.is_empty() {
        if cocycle.values().any(|coefficient| !coefficient.is_zero()) {
            return Err(TopologyError::InvalidCochain {
                reason: "a cup product is not a cocycle in the computed cycle space".to_owned(),
            });
        }
        return Ok(Vec::new());
    }
    let matrix = Matrix::from_rows(
        group.coefficient_context().clone(),
        (0..cocycle.dimension()).map(|row| {
            columns
                .iter()
                .map(|column| {
                    column
                        .get(row)
                        .expect("cohomology columns share one dimension")
                })
                .collect::<Vec<_>>()
        }),
    )?;
    let solution = matrix.solve(cocycle)?;
    if solution.dimension() != 0 {
        return Err(TopologyError::InvalidCochain {
            reason: "cohomology coordinates are not unique".to_owned(),
        });
    }
    Ok(solution
        .particular()
        .values()
        .skip(boundary_count)
        .collect())
}

pub(super) fn connected_presentation(
    field: &FiniteField,
    basis: &[SimplicialCohomologyBasisElement],
    products: &[SimplicialCohomologyProduct],
    top_degree: usize,
) -> Result<Option<GradedCommutativePresentation>, TopologyError> {
    if basis.iter().filter(|basis| basis.degree == 0).count() != 1 {
        return Ok(None);
    }
    let positive = basis
        .iter()
        .enumerate()
        .filter(|(_, basis)| basis.degree > 0)
        .collect::<Vec<_>>();
    let ring = GradedCommutativePolynomialRing::new(
        field.clone(),
        positive
            .iter()
            .map(|(_, basis)| GradedGenerator::new(basis.id.clone(), basis.degree))
            .collect(),
    )?;
    let mut generator_for_basis = vec![None; basis.len()];
    for (generator, (basis_index, _)) in positive.iter().enumerate() {
        generator_for_basis[*basis_index] = Some(generator);
    }
    let mut relations = Vec::new();
    for (left_generator, (left_basis, _)) in positive.iter().enumerate() {
        for (right_generator, (right_basis, _)) in positive.iter().enumerate().skip(left_generator)
        {
            let left = ring.generator(left_generator)?;
            let right = ring.generator(right_generator)?;
            let product = products
                .iter()
                .find(|product| product.left == *left_basis && product.right == *right_basis)
                .expect("the complete product table contains every positive pair");
            let rhs = ring.polynomial(product.terms.iter().map(|term| {
                let target = generator_for_basis[term.basis]
                    .expect("positive-degree products have positive-degree targets");
                let mut exponents = vec![0; ring.arity()];
                exponents[target] = 1;
                (term.coefficient.clone(), exponents)
            }))?;
            relations.push(left.mul(&right)?.sub(&rhs)?);
        }
    }
    let presentation = GradedCommutativePresentation::new(ring, relations)?;
    let top = u32::try_from(top_degree).map_err(|_| TopologyError::DimensionTooLarge {
        dimension: top_degree,
    })?;
    for degree in 0..=top.saturating_add(1) {
        let expected = basis.iter().filter(|basis| basis.degree == degree).count();
        if presentation.dimension(degree)? != expected {
            return Err(TopologyError::InvalidCochain {
                reason: format!(
                    "multiplication-table presentation has the wrong dimension in degree {degree}"
                ),
            });
        }
    }
    Ok(Some(presentation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_has_complete_exterior_style_presentation() {
        let circle = SimplicialComplex::from_facets(&[vec![0, 1], vec![1, 2], vec![0, 2]]);
        let ring = circle
            .cohomology_ring(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(
            ring.basis()
                .iter()
                .map(SimplicialCohomologyBasisElement::degree)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        let presentation = ring.presentation().unwrap();
        assert_eq!(presentation.hilbert_function(2).unwrap(), vec![1, 1, 0]);
        assert!(ring.product(1, 1).unwrap().is_empty());
        ring.verify().unwrap();
    }

    #[test]
    fn projective_plane_detects_the_degree_one_square() {
        let projective_plane = SimplicialComplex::from_facets(&[
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 4],
            vec![0, 3, 5],
            vec![0, 4, 5],
            vec![1, 2, 5],
            vec![1, 3, 4],
            vec![1, 4, 5],
            vec![2, 3, 4],
            vec![2, 3, 5],
        ]);
        let ring = projective_plane
            .cohomology_ring(FiniteField::prime(2_u8).unwrap())
            .unwrap();
        assert_eq!(
            ring.basis()
                .iter()
                .map(SimplicialCohomologyBasisElement::degree)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(ring.product(1, 1).unwrap()[0].basis(), 2);
        assert_eq!(ring.steenrod_square(0, 1).unwrap()[0].basis(), 1);
        assert_eq!(ring.steenrod_square(1, 1).unwrap()[0].basis(), 2);
        assert_eq!(
            ring.apply_operation(&AdmissibleCohomologyOperation::steenrod_square(1), 1)
                .unwrap()[0]
                .basis(),
            2
        );
        assert!(ring.steenrod_square(2, 1).unwrap().is_empty());
        assert_eq!(
            ring.presentation().unwrap().hilbert_function(3).unwrap(),
            vec![1, 1, 1, 0]
        );
    }

    #[test]
    fn disconnected_degree_zero_algebra_keeps_its_table_without_a_presentation() {
        let points = SimplicialComplex::from_facets(&[vec![0], vec![1]]);
        let ring = points
            .cohomology_ring(FiniteField::prime(3_u8).unwrap())
            .unwrap();
        assert_eq!(ring.basis().len(), 2);
        assert!(ring.presentation().is_none());
        assert!(matches!(
            ring.steenrod_square(0, 0),
            Err(TopologyError::InvalidCochain { .. })
        ));
        let restored = SimplicialCohomologyRing::from_json(&ring.to_json().unwrap()).unwrap();
        assert_eq!(restored, ring);
    }

    #[test]
    fn coefficient_field_is_the_empty_generator_presentation() {
        let point = SimplicialComplex::from_facets(&[vec![0]]);
        let ring = point
            .cohomology_ring(FiniteField::prime(5_u8).unwrap())
            .unwrap();
        let presentation = ring.presentation().unwrap();
        assert!(presentation.ring().generators().is_empty());
        assert_eq!(presentation.hilbert_function(2).unwrap(), vec![1, 0, 0]);
        assert_eq!(ring.steenrod_reduced_power(0, 0).unwrap()[0].basis(), 0);
        assert!(ring.bockstein_reduced_power(0, 0).unwrap().is_empty());
    }
}
