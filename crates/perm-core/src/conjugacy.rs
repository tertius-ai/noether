// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact, compact conjugacy classes for permutation groups of bounded order.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::sync::OnceLock;

use crate::group::symmetric_group_order;
use crate::{BigUInt, GroupError, Perm, PermGroup, Point, Result};

/// Largest group order accepted by exact conjugacy enumeration.
///
/// This makes the direct method fail closed on groups for which materializing
/// every element would be surprising. Larger-group algorithms can be added
/// behind the same mathematical method without changing this correctness
/// floor.
pub const CONJUGACY_ENUMERATION_CAP: usize = 1_000_000;

/// One exact conjugacy class in a concrete permutation group.
#[derive(Debug, Clone)]
pub struct ConjugacyClass {
    group: PermGroup,
    representative: Perm,
    size: BigUInt,
    elements: OnceLock<Vec<Perm>>,
}

impl ConjugacyClass {
    /// The group in which conjugacy is being computed.
    pub fn group(&self) -> &PermGroup {
        &self.group
    }

    /// Lexicographically least element of the class.
    pub fn representative(&self) -> &Perm {
        &self.representative
    }

    /// Class elements in lexicographic image-vector order, materialized lazily.
    pub fn elements(&self) -> &[Perm] {
        self.elements
            .get_or_init(|| conjugates(&self.group, &self.representative))
    }

    /// Exact class size.
    pub fn size(&self) -> BigUInt {
        self.size.clone()
    }

    /// Centralizer of the representative inside the source group.
    pub fn centralizer(&self) -> Result<PermGroup> {
        let mut elements = Vec::new();
        for element in self.group.iter_elements() {
            if element.compose(&self.representative)? == self.representative.compose(&element)? {
                elements.push(element);
            }
        }
        PermGroup::from_generators(self.group.degree(), &elements)
    }

    /// Recompute the orbit under conjugation and its centralizer identity.
    pub fn verify(&self) -> Result<()> {
        let elements = self.elements();
        if elements.is_empty()
            || self.representative != elements[0]
            || BigUInt::from(elements.len() as u64) != self.size
            || elements
                .windows(2)
                .any(|pair| pair[0].images() >= pair[1].images())
            || elements
                .iter()
                .any(|element| !self.group.contains(element).unwrap_or(false))
        {
            return Err(GroupError::WitnessUnverified);
        }

        let expected = conjugates(&self.group, &self.representative);
        if expected != elements {
            return Err(GroupError::WitnessUnverified);
        }

        let centralizer = self.centralizer()?;
        let mut product = centralizer.order();
        product.mul_assign_u64(elements.len() as u64);
        if product != self.group.order() {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(())
    }
}

impl PartialEq for ConjugacyClass {
    fn eq(&self, other: &Self) -> bool {
        self.group.degree() == other.group.degree()
            && self.group.generators() == other.group.generators()
            && self.representative == other.representative
            && self.size == other.size
    }
}

impl Eq for ConjugacyClass {}

impl fmt::Display for ConjugacyClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConjugacyClass(degree={}, size={})",
            self.group.degree(),
            self.size
        )
    }
}

impl PermGroup {
    /// Enumerate every conjugacy class exactly for a bounded-order group.
    ///
    /// The result order is deterministic: representatives and lazily
    /// materialized members use lexicographic image-vector order. If the group order exceeds
    /// [`CONJUGACY_ENUMERATION_CAP`], this returns a typed error before
    /// materializing any elements.
    pub fn conjugacy_classes(&self) -> Result<Vec<ConjugacyClass>> {
        if self.order() > BigUInt::from(CONJUGACY_ENUMERATION_CAP as u64) {
            return Err(GroupError::EnumerationLimitExceeded {
                operation: "conjugacy-class enumeration",
                cap: CONJUGACY_ENUMERATION_CAP,
            });
        }
        if symmetric_group_order(self.degree()).as_ref() == Some(&self.order()) {
            return symmetric_group_conjugacy_classes(self);
        }
        if self.is_full_alternating() {
            return alternating_group_conjugacy_classes(self);
        }

        let mut elements: Vec<Perm> = self.iter_elements().collect();
        elements.sort_by(|left, right| left.images().cmp(right.images()));
        let mut assigned = HashSet::with_capacity(elements.len());
        let mut classes = Vec::new();

        for element in &elements {
            if assigned.contains(element) {
                continue;
            }
            let members = conjugates(self, element);
            assigned.extend(members.iter().cloned());
            classes.push(ConjugacyClass {
                group: self.clone(),
                representative: members[0].clone(),
                size: BigUInt::from(members.len() as u64),
                elements: OnceLock::from(members),
            });
        }

        if assigned.len() != elements.len() {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(classes)
    }
}

fn integer_partitions(
    remaining: usize,
    minimum: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if remaining == 0 {
        result.push(current.clone());
        return;
    }
    for part in minimum..=remaining {
        current.push(part);
        integer_partitions(remaining - part, part, current, result);
        current.pop();
    }
}

fn partition_representative(degree: usize, partition: &[usize]) -> Result<Perm> {
    let degree_point = Point::try_from(degree).map_err(|_| GroupError::WitnessUnverified)?;
    let mut images = (0..degree_point).collect::<Vec<_>>();
    let mut offset = 0usize;
    for &length in partition {
        if length > 1 {
            for point in 0..length - 1 {
                images[offset + point] = Point::try_from(offset + point + 1)
                    .map_err(|_| GroupError::WitnessUnverified)?;
            }
            images[offset + length - 1] =
                Point::try_from(offset).map_err(|_| GroupError::WitnessUnverified)?;
        }
        offset += length;
    }
    Perm::from_images(&images)
}

fn symmetric_class_size(order: u64, partition: &[usize]) -> Option<BigUInt> {
    let mut centralizer_order = 1u64;
    let mut cursor = 0usize;
    while cursor < partition.len() {
        let length = partition[cursor];
        let mut multiplicity = 1usize;
        while cursor + multiplicity < partition.len() && partition[cursor + multiplicity] == length
        {
            multiplicity += 1;
        }
        for _ in 0..multiplicity {
            centralizer_order = centralizer_order.checked_mul(u64::try_from(length).ok()?)?;
        }
        for factor in 2..=multiplicity {
            centralizer_order = centralizer_order.checked_mul(u64::try_from(factor).ok()?)?;
        }
        cursor += multiplicity;
    }
    order
        .is_multiple_of(centralizer_order)
        .then(|| BigUInt::from(order / centralizer_order))
}

fn symmetric_group_conjugacy_classes(group: &PermGroup) -> Result<Vec<ConjugacyClass>> {
    // Conjugacy classes in S_n are exactly the integer partitions of n. The
    // ambient-order equality proves this concrete subgroup is all of S_n, so
    // store a canonical representative and exact size now and materialize the
    // members only if `elements()` or `verify()` asks for them.
    let order = group
        .order()
        .to_decimal()
        .parse::<u64>()
        .map_err(|_| GroupError::WitnessUnverified)?;
    let mut partitions = Vec::new();
    integer_partitions(group.degree(), 1, &mut Vec::new(), &mut partitions);
    let mut classes = partitions
        .into_iter()
        .map(|partition| {
            let representative = partition_representative(group.degree(), &partition)?;
            let size =
                symmetric_class_size(order, &partition).ok_or(GroupError::WitnessUnverified)?;
            Ok(ConjugacyClass {
                group: group.clone(),
                representative,
                size,
                elements: OnceLock::new(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    classes.sort_by(|left, right| {
        left.representative
            .images()
            .cmp(right.representative.images())
    });
    if classes
        .iter()
        .fold(BigUInt::zero(), |sum, class| sum.add(&class.size))
        != group.order()
    {
        return Err(GroupError::WitnessUnverified);
    }
    Ok(classes)
}

fn alternating_group_conjugacy_classes(group: &PermGroup) -> Result<Vec<ConjugacyClass>> {
    // An S_n class lies in A_n exactly when its cycle partition has even
    // sign. It remains one A_n class unless all cycle lengths are distinct
    // and odd, in which case it splits into two equal classes. Exact order
    // has already proved that this concrete group is A_n.
    let symmetric_order = symmetric_group_order(group.degree())
        .and_then(|order| order.to_decimal().parse::<u64>().ok())
        .ok_or(GroupError::WitnessUnverified)?;
    let mut partitions = Vec::new();
    integer_partitions(group.degree(), 1, &mut Vec::new(), &mut partitions);
    let mut classes = Vec::new();
    for partition in partitions {
        if !(group.degree() - partition.len()).is_multiple_of(2) {
            continue;
        }
        let representative = partition_representative(group.degree(), &partition)?;
        let symmetric_size = symmetric_class_size(symmetric_order, &partition)
            .ok_or(GroupError::WitnessUnverified)?;
        let splits = partition.iter().all(|part| part % 2 == 1)
            && partition.windows(2).all(|pair| pair[0] != pair[1]);
        if !splits {
            classes.push(ConjugacyClass {
                group: group.clone(),
                representative,
                size: symmetric_size,
                elements: OnceLock::new(),
            });
            continue;
        }

        let (class_size, remainder) = symmetric_size.divmod_u64(2);
        if remainder != 0 || group.degree() < 2 {
            return Err(GroupError::WitnessUnverified);
        }
        let mut odd_images: Vec<Point> = (0..group.degree() as Point).collect();
        odd_images.swap(0, 1);
        let odd = Perm::from_images(&odd_images)?;
        for seed in [representative.clone(), representative.conjugate(&odd)?] {
            let members = conjugates(group, &seed);
            if BigUInt::from(members.len() as u64) != class_size {
                return Err(GroupError::WitnessUnverified);
            }
            classes.push(ConjugacyClass {
                group: group.clone(),
                representative: members[0].clone(),
                size: class_size.clone(),
                elements: OnceLock::from(members),
            });
        }
    }
    classes.sort_by(|left, right| {
        left.representative
            .images()
            .cmp(right.representative.images())
    });
    if classes
        .iter()
        .fold(BigUInt::zero(), |sum, class| sum.add(&class.size))
        != group.order()
    {
        return Err(GroupError::WitnessUnverified);
    }
    Ok(classes)
}

fn conjugates(group: &PermGroup, representative: &Perm) -> Vec<Perm> {
    // A conjugacy class is the orbit under conjugation by any generating set.
    // Walking that orbit visits each member once; conjugating by every ambient
    // group element for every class repeats the same work |G|/|C_G(x)| times.
    let mut conjugators: Vec<Perm> = Vec::with_capacity(group.generators().len() * 2);
    for generator in group.generators() {
        if !conjugators.iter().any(|candidate| candidate == generator) {
            conjugators.push(generator.clone());
        }
        let inverse = generator.inverse();
        if !conjugators.iter().any(|candidate| candidate == &inverse) {
            conjugators.push(inverse);
        }
    }
    let conjugators = conjugators
        .into_iter()
        .map(|conjugator| {
            let inverse = conjugator.inverse();
            (conjugator, inverse)
        })
        .collect::<Vec<_>>();
    let mut members = HashSet::from([representative.clone()]);
    let mut queue = VecDeque::from([representative.clone()]);
    while let Some(member) = queue.pop_front() {
        for (conjugator, inverse) in &conjugators {
            let image = member.conjugate_unchecked_with_inverse(conjugator, inverse);
            if members.insert(image.clone()) {
                queue.push_back(image);
            }
        }
    }
    let mut members: Vec<Perm> = members.into_iter().collect();
    members.sort_by(|left, right| left.images().cmp(right.images()));
    members
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_group(degree: usize) -> PermGroup {
        let transposition = {
            let mut images: Vec<_> = (0..degree as u32).collect();
            images.swap(0, 1);
            Perm::from_images(&images).unwrap()
        };
        let cycle = Perm::from_images(
            &(1..degree as u32)
                .chain(core::iter::once(0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        PermGroup::from_generators(degree, &[transposition, cycle]).unwrap()
    }

    #[test]
    fn s4_classes_have_the_cycle_type_sizes() {
        let classes = symmetric_group(4).conjugacy_classes().unwrap();
        let mut sizes: Vec<_> = classes
            .iter()
            .map(|class| class.size().to_decimal().parse::<usize>().unwrap())
            .collect();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 3, 6, 6, 8]);
        for class in classes {
            class.verify().unwrap();
        }
    }

    #[test]
    fn s8_classes_have_the_partition_cycle_type_sizes() {
        let classes = symmetric_group(8).conjugacy_classes().unwrap();
        assert!(classes.iter().all(|class| class.elements.get().is_none()));
        let mut sizes: Vec<_> = classes
            .iter()
            .map(|class| class.size().to_decimal().parse::<usize>().unwrap())
            .collect();
        sizes.sort_unstable();
        assert_eq!(
            sizes,
            vec![
                1, 28, 105, 112, 210, 420, 420, 1120, 1120, 1120, 1260, 1260, 1344, 1680, 2520,
                2688, 3360, 3360, 3360, 4032, 5040, 5760
            ]
        );
        assert!(classes.iter().all(|class| class.verify().is_ok()));
    }

    #[test]
    fn nonsymmetric_group_uses_exact_conjugation_orbits() {
        let alternating_five = PermGroup::from_generators(
            5,
            &[
                Perm::from_images(&[1, 2, 0, 3, 4]).unwrap(),
                Perm::from_images(&[1, 2, 3, 4, 0]).unwrap(),
            ],
        )
        .unwrap();
        let classes = alternating_five.conjugacy_classes().unwrap();
        let mut sizes = classes
            .iter()
            .map(|class| class.size().to_decimal().parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 12, 12, 15, 20]);
        assert!(classes.iter().all(|class| class.verify().is_ok()));
    }

    #[test]
    fn alternating_six_classes_use_partition_splitting() {
        let alternating_six = PermGroup::from_generators(
            6,
            &[
                Perm::from_images(&[1, 2, 0, 3, 4, 5]).unwrap(),
                Perm::from_images(&[0, 2, 3, 4, 5, 1]).unwrap(),
            ],
        )
        .unwrap();
        assert!(alternating_six.is_full_alternating());
        let classes = alternating_six.conjugacy_classes().unwrap();
        let mut sizes = classes
            .iter()
            .map(|class| class.size().to_decimal().parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![1, 40, 40, 45, 72, 72, 90]);
        assert!(classes.iter().all(|class| class.verify().is_ok()));
    }

    #[test]
    fn conjugacy_enumeration_fails_before_materializing_s10() {
        let error = symmetric_group(10).conjugacy_classes().unwrap_err();
        assert_eq!(
            error,
            GroupError::EnumerationLimitExceeded {
                operation: "conjugacy-class enumeration",
                cap: CONJUGACY_ENUMERATION_CAP,
            }
        );
    }
}
