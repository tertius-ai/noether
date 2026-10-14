// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Groups, actions, and representation theory.

use core::fmt;
use std::collections::BTreeMap;
use std::str::FromStr;

use crate::{Integer, NoetherError, SemanticType, TypeDescription};

mod representation;

pub use representation::{
    PolynomialGlRepresentation, RepresentationError, WeightBasisConvention, WeightCharacter,
    WeightLattice, WeightLatticeMap, MAX_EXPLICIT_WEIGHT_CHARACTER_DIMENSION,
};

pub use perm_core::{
    base_and_strong_generating_set, derived_subgroup, double_coset_triangle, double_cosets,
    element_membership_test, group_order, is_nilpotent, is_perfect, is_solvable, iterate_elements,
    orbit, orbit_stabilizer, pointwise_stabilizer, schreier_generators, schreier_tree,
    stabilizer_chain, transversal, uniform_random_element, visit_double_coset_triangle, BigUInt,
    ConjugacyClass, DoubleCoset, GroupError, NotAPermutationReason, OrbitStabilizer, PairOrbit,
    Perm as Permutation, PermGroup as PermutationGroup, Point, RandomElement, SchreierGenerator,
    SchreierTree, StabilizerChain, SubsetOrbit, Transversal, SUBGROUP_ENUMERATION_CAP,
};

const ABELIAN_INVARIANT_ELEMENT_LIMIT: usize = 1_000_000;

/// Failure to recover the invariant factors of a concrete finite abelian group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteAbelianInvariantError {
    /// The supplied generators do not commute.
    NonAbelian,
    /// Exact element enumeration exceeds the implementation safety ceiling.
    EnumerationLimitExceeded {
        /// Maximum group order accepted by this implementation.
        limit: usize,
    },
    /// The exact group order or one element order does not fit the enumerator.
    OrderTooLarge,
    /// Element-order counts contradicted the finite-abelian classification.
    InvalidOrderStatistics,
    /// A checked permutation operation failed.
    Group(GroupError),
}

impl fmt::Display for FiniteAbelianInvariantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonAbelian => formatter.write_str("the supplied finite group is not abelian"),
            Self::EnumerationLimitExceeded { limit } => write!(
                formatter,
                "finite-abelian invariant factors require enumerating at most {limit} elements"
            ),
            Self::OrderTooLarge => formatter
                .write_str("a finite-group order does not fit the invariant-factor enumerator"),
            Self::InvalidOrderStatistics => formatter.write_str(
                "element-order counts contradict the classification of finite abelian groups",
            ),
            Self::Group(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for FiniteAbelianInvariantError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Group(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GroupError> for FiniteAbelianInvariantError {
    fn from(value: GroupError) -> Self {
        Self::Group(value)
    }
}

/// Exact invariant-factor decomposition of a concrete finite abelian group.
pub trait FiniteAbelianInvariants {
    /// Return `d_1 | ... | d_r` such that the group is isomorphic to
    /// `C_{d_1} x ... x C_{d_r}`. The trivial group returns an empty family.
    fn abelian_invariant_factors(&self) -> Result<Vec<Integer>, FiniteAbelianInvariantError>;
}

/// Exact invariant factors of the abelianization of a concrete finite group.
pub trait FiniteAbelianization {
    /// Return the invariant factors of `G/[G,G]`.
    ///
    /// The quotient is constructed as the faithful regular action on cosets
    /// of the derived subgroup and is then passed to the canonical finite-
    /// abelian invariant-factor calculation.
    fn abelianization_invariant_factors(&self)
        -> Result<Vec<Integer>, FiniteAbelianInvariantError>;
}

impl FiniteAbelianization for PermutationGroup {
    fn abelianization_invariant_factors(
        &self,
    ) -> Result<Vec<Integer>, FiniteAbelianInvariantError> {
        let order = self
            .order()
            .to_decimal()
            .parse::<usize>()
            .map_err(|_| FiniteAbelianInvariantError::OrderTooLarge)?;
        if order > ABELIAN_INVARIANT_ELEMENT_LIMIT {
            return Err(FiniteAbelianInvariantError::EnumerationLimitExceeded {
                limit: ABELIAN_INVARIANT_ELEMENT_LIMIT,
            });
        }
        let derived = derived_subgroup(self)?;
        let mut elements = iterate_elements(self).collect::<Vec<_>>();
        elements.sort_by(|left, right| left.images().cmp(right.images()));
        let mut derived_elements = iterate_elements(&derived).collect::<Vec<_>>();
        derived_elements.sort_by(|left, right| left.images().cmp(right.images()));

        let mut element_cosets = BTreeMap::<Vec<u32>, usize>::new();
        let mut representatives = Vec::new();
        for element in &elements {
            if element_cosets.contains_key(element.images()) {
                continue;
            }
            let coset = representatives.len();
            representatives.push(element.clone());
            for commutator in &derived_elements {
                let member = commutator.compose(element)?;
                element_cosets.insert(member.images().to_vec(), coset);
            }
        }
        if element_cosets.len() != elements.len() {
            return Err(FiniteAbelianInvariantError::InvalidOrderStatistics);
        }

        let quotient_generators = self
            .generators()
            .iter()
            .map(|generator| {
                let images = representatives
                    .iter()
                    .map(|representative| {
                        let image = representative.compose(generator)?;
                        let coset = element_cosets
                            .get(image.images())
                            .copied()
                            .ok_or(FiniteAbelianInvariantError::InvalidOrderStatistics)?;
                        u32::try_from(coset).map_err(|_| FiniteAbelianInvariantError::OrderTooLarge)
                    })
                    .collect::<Result<Vec<_>, FiniteAbelianInvariantError>>()?;
                Permutation::from_images(&images).map_err(Into::into)
            })
            .collect::<Result<Vec<_>, FiniteAbelianInvariantError>>()?;
        let quotient =
            PermutationGroup::from_generators(representatives.len(), &quotient_generators)?;
        quotient.abelian_invariant_factors()
    }
}

impl FiniteAbelianInvariants for PermutationGroup {
    fn abelian_invariant_factors(&self) -> Result<Vec<Integer>, FiniteAbelianInvariantError> {
        for (left_index, left) in self.generators().iter().enumerate() {
            for right in self.generators().iter().skip(left_index + 1) {
                if left.compose(right)? != right.compose(left)? {
                    return Err(FiniteAbelianInvariantError::NonAbelian);
                }
            }
        }

        let order = self
            .order()
            .to_decimal()
            .parse::<usize>()
            .map_err(|_| FiniteAbelianInvariantError::OrderTooLarge)?;
        if order > ABELIAN_INVARIANT_ELEMENT_LIMIT {
            return Err(FiniteAbelianInvariantError::EnumerationLimitExceeded {
                limit: ABELIAN_INVARIANT_ELEMENT_LIMIT,
            });
        }
        if order == 1 {
            return Ok(Vec::new());
        }
        let element_orders = iterate_elements(self)
            .map(|element| {
                element
                    .order()
                    .to_decimal()
                    .parse::<usize>()
                    .map_err(|_| FiniteAbelianInvariantError::OrderTooLarge)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if element_orders.len() != order {
            return Err(FiniteAbelianInvariantError::InvalidOrderStatistics);
        }

        let mut primary_factors = Vec::<Vec<usize>>::new();
        for (prime, group_exponent) in factor_word(order) {
            let mut killed_logarithms = Vec::with_capacity(group_exponent);
            let mut prime_power = 1usize;
            for _ in 0..group_exponent {
                prime_power = prime_power
                    .checked_mul(prime)
                    .ok_or(FiniteAbelianInvariantError::OrderTooLarge)?;
                let killed = element_orders
                    .iter()
                    .filter(|element_order| prime_power.is_multiple_of(**element_order))
                    .count();
                killed_logarithms.push(exact_logarithm(killed, prime)?);
            }
            let mut ranks = Vec::with_capacity(group_exponent + 1);
            let mut previous = 0usize;
            for logarithm in killed_logarithms {
                let rank = logarithm
                    .checked_sub(previous)
                    .ok_or(FiniteAbelianInvariantError::InvalidOrderStatistics)?;
                ranks.push(rank);
                previous = logarithm;
            }
            ranks.push(0);
            let mut factors = Vec::new();
            let mut power = 1usize;
            for exponent in 1..=group_exponent {
                power = power
                    .checked_mul(prime)
                    .ok_or(FiniteAbelianInvariantError::OrderTooLarge)?;
                let multiplicity = ranks[exponent - 1]
                    .checked_sub(ranks[exponent])
                    .ok_or(FiniteAbelianInvariantError::InvalidOrderStatistics)?;
                factors.extend(std::iter::repeat_n(power, multiplicity));
            }
            primary_factors.push(factors);
        }

        let rank = primary_factors.iter().map(Vec::len).max().unwrap_or(0);
        let mut invariant_factors = vec![1usize; rank];
        for factors in primary_factors {
            let offset = rank - factors.len();
            for (index, factor) in factors.into_iter().enumerate() {
                invariant_factors[offset + index] = invariant_factors[offset + index]
                    .checked_mul(factor)
                    .ok_or(FiniteAbelianInvariantError::OrderTooLarge)?;
            }
        }
        if invariant_factors.iter().product::<usize>() != order
            || invariant_factors
                .windows(2)
                .any(|pair| !pair[1].is_multiple_of(pair[0]))
        {
            return Err(FiniteAbelianInvariantError::InvalidOrderStatistics);
        }
        Ok(invariant_factors.into_iter().map(Integer::from).collect())
    }
}

fn factor_word(mut value: usize) -> Vec<(usize, usize)> {
    let mut factors = Vec::new();
    let mut prime = 2usize;
    while prime <= value / prime {
        if value.is_multiple_of(prime) {
            let mut exponent = 0usize;
            while value.is_multiple_of(prime) {
                value /= prime;
                exponent += 1;
            }
            factors.push((prime, exponent));
        }
        prime += usize::from(prime == 2) + 2 * usize::from(prime != 2);
    }
    if value > 1 {
        factors.push((value, 1));
    }
    factors
}

fn exact_logarithm(mut value: usize, base: usize) -> Result<usize, FiniteAbelianInvariantError> {
    let mut exponent = 0usize;
    while value > 1 && value.is_multiple_of(base) {
        value /= base;
        exponent += 1;
    }
    if value == 1 {
        Ok(exponent)
    } else {
        Err(FiniteAbelianInvariantError::InvalidOrderStatistics)
    }
}

static PERMUTATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/Permutation@1",
    rust_path: "noether::groups::Permutation",
    python_path: Some("noether.groups.Permutation"),
};

static PERMUTATION_GROUP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/PermutationGroup@1",
    rust_path: "noether::groups::PermutationGroup",
    python_path: Some("noether.groups.PermutationGroup"),
};

static CONJUGACY_CLASS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/ConjugacyClass@1",
    rust_path: "noether::groups::ConjugacyClass",
    python_path: Some("noether.groups.ConjugacyClass"),
};

static SUBSET_ORBIT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/SubsetOrbit@1",
    rust_path: "noether::groups::SubsetOrbit",
    python_path: Some("noether.groups.SubsetOrbit"),
};

static PAIR_ORBIT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/PairOrbit@1",
    rust_path: "noether::groups::PairOrbit",
    python_path: Some("noether.groups.PairOrbit"),
};

static DOUBLE_COSET_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/groups/DoubleCoset@1",
    rust_path: "noether::groups::DoubleCoset",
    python_path: Some("noether.groups.DoubleCoset"),
};

fn push_frame(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), NoetherError> {
    let length = u64::try_from(bytes.len()).map_err(|_| {
        NoetherError::Serialization("group component is too large to serialize".to_owned())
    })?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}

impl SemanticType for Permutation {
    const TYPE_ID: &'static str = "noether/groups/Permutation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let degree = u64::try_from(self.degree()).map_err(|_| {
            NoetherError::Serialization("permutation degree exceeds u64".to_owned())
        })?;
        let mut output = Vec::new();
        output.extend_from_slice(&degree.to_be_bytes());
        for image in self.images() {
            output.extend_from_slice(&image.to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PERMUTATION_DESCRIPTION
    }
}

impl SemanticType for PermutationGroup {
    const TYPE_ID: &'static str = "noether/groups/PermutationGroup@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let degree = u64::try_from(self.degree()).map_err(|_| {
            NoetherError::Serialization("permutation-group degree exceeds u64".to_owned())
        })?;
        let generator_count = u64::try_from(self.generators().len()).map_err(|_| {
            NoetherError::Serialization("permutation-group generator count exceeds u64".to_owned())
        })?;
        let mut output = Vec::new();
        output.extend_from_slice(&degree.to_be_bytes());
        output.extend_from_slice(&generator_count.to_be_bytes());
        for generator in self.generators() {
            push_frame(&mut output, &generator.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &PERMUTATION_GROUP_DESCRIPTION
    }
}

impl SemanticType for ConjugacyClass {
    const TYPE_ID: &'static str = "noether/groups/ConjugacyClass@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        push_frame(&mut output, &self.group().canonical_bytes()?)?;
        push_frame(&mut output, &self.representative().canonical_bytes()?)?;
        let member_count = u64::try_from(self.elements().len()).map_err(|_| {
            NoetherError::Serialization("conjugacy class is too large to serialize".to_owned())
        })?;
        output.extend_from_slice(&member_count.to_be_bytes());
        for member in self.elements() {
            push_frame(&mut output, &member.canonical_bytes()?)?;
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &CONJUGACY_CLASS_DESCRIPTION
    }
}

impl SemanticType for SubsetOrbit {
    const TYPE_ID: &'static str = "noether/groups/SubsetOrbit@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.degree(),
            self.subset_size(),
            self.representative(),
            self.elements(),
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &SUBSET_ORBIT_DESCRIPTION
    }
}

impl SemanticType for PairOrbit {
    const TYPE_ID: &'static str = "noether/groups/PairOrbit@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.degree(),
            self.left_seed(),
            self.right_seed(),
            self.representative(),
            self.anchored_suborbits(),
            self.size(),
            self.same_point_orbit(),
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &PAIR_ORBIT_DESCRIPTION
    }
}

impl SemanticType for DoubleCoset {
    const TYPE_ID: &'static str = "noether/groups/DoubleCoset@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.degree(),
            self.representative().images(),
            self.inverse_representative().map(Permutation::images),
            self.right_coset_count(),
            self.size().to_decimal(),
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &DOUBLE_COSET_DESCRIPTION
    }
}

/// A finite group with a concrete element type.
pub trait FiniteGroup: crate::SemanticType {
    /// Elements of this group.
    type Element: crate::SemanticType + Clone + Eq;

    /// Identity element.
    fn identity(&self) -> Self::Element;
    /// Checked group multiplication.
    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Result<Self::Element, GroupError>;
    /// Checked group inversion.
    fn inverse(&self, value: &Self::Element) -> Result<Self::Element, GroupError>;
    /// Exact group order.
    fn order(&self) -> crate::Integer;
    /// Membership in this concrete group.
    fn contains(&self, value: &Self::Element) -> Result<bool, GroupError>;
}

impl FiniteGroup for PermutationGroup {
    type Element = Permutation;

    fn identity(&self) -> Self::Element {
        Permutation::identity(self.degree())
    }

    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Result<Self::Element, GroupError> {
        if !PermutationGroup::contains(self, a)? || !PermutationGroup::contains(self, b)? {
            return Err(GroupError::ElementNotInGroup);
        }
        a.compose(b)
    }

    fn inverse(&self, value: &Self::Element) -> Result<Self::Element, GroupError> {
        if !PermutationGroup::contains(self, value)? {
            return Err(GroupError::ElementNotInGroup);
        }
        Ok(value.inverse())
    }

    fn order(&self) -> Integer {
        Integer::from_str(&PermutationGroup::order(self).to_decimal())
            .expect("perm-core renders every group order as a decimal integer")
    }

    fn contains(&self, value: &Self::Element) -> Result<bool, GroupError> {
        PermutationGroup::contains(self, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permutation(images: &[u32]) -> Permutation {
        Permutation::from_images(images).unwrap()
    }

    fn group(degree: usize, generators: &[&[u32]]) -> PermutationGroup {
        PermutationGroup::from_generators(
            degree,
            &generators
                .iter()
                .map(|images| permutation(images))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn finite_abelian_invariants_distinguish_order_four_groups() {
        let cyclic = group(4, &[&[1, 2, 3, 0]]);
        assert_eq!(
            cyclic.abelian_invariant_factors().unwrap(),
            vec![Integer::from(4_u8)]
        );

        let klein = group(4, &[&[1, 0, 3, 2], &[2, 3, 0, 1]]);
        assert_eq!(
            klein.abelian_invariant_factors().unwrap(),
            vec![Integer::from(2_u8), Integer::from(2_u8)]
        );
    }

    #[test]
    fn finite_abelian_invariants_combine_primary_factors() {
        let cyclic_six = group(6, &[&[1, 2, 3, 4, 5, 0]]);
        assert_eq!(
            cyclic_six.abelian_invariant_factors().unwrap(),
            vec![Integer::from(6_u8)]
        );
        let c2_times_c4 = group(6, &[&[1, 0, 2, 3, 4, 5], &[0, 1, 3, 4, 5, 2]]);
        assert_eq!(
            c2_times_c4.abelian_invariant_factors().unwrap(),
            vec![Integer::from(2_u8), Integer::from(4_u8)]
        );
        let trivial = group(0, &[]);
        assert!(trivial.abelian_invariant_factors().unwrap().is_empty());
    }

    #[test]
    fn finite_abelian_invariants_reject_nonabelian_groups() {
        let symmetric_three = group(3, &[&[1, 0, 2], &[1, 2, 0]]);
        assert_eq!(
            symmetric_three.abelian_invariant_factors(),
            Err(FiniteAbelianInvariantError::NonAbelian)
        );
    }

    #[test]
    fn abelianization_invariants_handle_nonabelian_groups() {
        let symmetric_three = group(3, &[&[1, 0, 2], &[1, 2, 0]]);
        assert_eq!(
            symmetric_three.abelianization_invariant_factors().unwrap(),
            vec![Integer::from(2_u8)]
        );
        let alternating_four = group(4, &[&[1, 2, 0, 3], &[1, 0, 3, 2]]);
        assert_eq!(
            alternating_four.abelianization_invariant_factors().unwrap(),
            vec![Integer::from(3_u8)]
        );
    }
}
