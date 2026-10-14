// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Subgroup-relative operations on the Schreier–Sims substrate ([`PermGroup`]):
//! literal containment, left-coset transversals, and the subdegree sequence.
//!
//! The transversal supplies coset representatives for subgroup and resolvent
//! computations. Subdegrees are a conjugacy invariant finer than cycle type
//! for separating some cycle-type-equivalent groups.
//!
//! Right-action convention throughout (matching [`Perm::compose`]): the group
//! product `s·r` is `s.compose(r)`, and `g⁻¹·g'` is `g.inverse().compose(g')`.

use std::collections::{HashMap, VecDeque};

use crate::bigint::BigUInt;
use crate::error::{GroupError, Result};
use crate::group::{symmetric_group_order, PermGroup};
use crate::perm::{Perm, Point};

/// Safety cap on the index `[G:H]` that [`coset_transversal`] will enumerate, so
/// an accidentally-tiny subgroup cannot exhaust memory. Stauduhar uses
/// small-index (large) subgroups, well under this.
const MAX_INDEX: usize = 1 << 20;

/// Maximum group order materialized by exact subgroup-enumeration operations.
pub const SUBGROUP_ENUMERATION_CAP: usize = 1_000_000;

fn checked_elements(group: &PermGroup, operation: &'static str) -> Result<Vec<Perm>> {
    if group.order() > crate::BigUInt::from(SUBGROUP_ENUMERATION_CAP as u64) {
        return Err(GroupError::EnumerationLimitExceeded {
            operation,
            cap: SUBGROUP_ENUMERATION_CAP,
        });
    }
    let mut elements = group.iter_elements().collect::<Vec<_>>();
    elements.sort_by(|left, right| left.images().cmp(right.images()));
    Ok(elements)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor <= value / divisor {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn is_power_of(order: &crate::BigUInt, prime: u64) -> bool {
    let mut quotient = order.clone();
    while !quotient.is_one() {
        let (next, remainder) = quotient.divmod_u64(prime);
        if remainder != 0 {
            return false;
        }
        quotient = next;
    }
    true
}

fn permutation_power(element: &Perm, mut exponent: u64) -> Perm {
    let mut result = Perm::identity(element.degree());
    let mut base = element.clone();
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result.compose_unchecked(&base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base.compose_unchecked(&base);
        }
    }
    result
}

fn normalizes_subgroup(element: &Perm, subgroup: &PermGroup) -> Result<bool> {
    for generator in subgroup.generators() {
        if !subgroup.contains(&generator.conjugate(element)?)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn symmetric_sylow_generators(degree: usize, prime: u64) -> Option<Vec<Perm>> {
    let prime = usize::try_from(prime).ok()?;
    let degree_point = Point::try_from(degree).ok()?;
    let mut digits = Vec::new();
    let mut quotient = degree;
    while quotient != 0 {
        digits.push(quotient % prime);
        quotient /= prime;
    }

    let mut block_size = 1usize;
    let mut offset = 0usize;
    let mut generators = Vec::new();
    for (level, digit) in digits.into_iter().enumerate() {
        for _ in 0..digit {
            let mut subblock_size = 1usize;
            for _ in 0..level {
                let span = subblock_size.checked_mul(prime)?;
                let mut images = (0..degree_point).collect::<Vec<_>>();
                for subblock in 0..prime {
                    for within in 0..subblock_size {
                        let source = offset + subblock * subblock_size + within;
                        let target = offset + ((subblock + 1) % prime) * subblock_size + within;
                        images[source] = Point::try_from(target).ok()?;
                    }
                }
                debug_assert!(span <= block_size);
                generators.push(Perm::from_images(&images).ok()?);
                subblock_size = span;
            }
            offset = offset.checked_add(block_size)?;
        }
        block_size = block_size.checked_mul(prime)?;
    }
    (offset == degree).then_some(generators)
}

impl PermGroup {
    /// Kernel of the sign homomorphism restricted to this group.
    ///
    /// When an odd generator exists, `{1, t}` is a right transversal for the
    /// even subgroup.  The following Reidemeister--Schreier generators write
    /// down that index-two kernel directly; exact order and containment checks
    /// replay the construction before it is returned.
    fn even_subgroup(&self) -> Result<PermGroup> {
        let Some(odd) = self
            .generators()
            .iter()
            .find(|generator| !generator.is_even())
        else {
            return Ok(self.clone());
        };
        let odd_inverse = odd.inverse();
        let mut generators = Vec::with_capacity(self.generators().len() * 2);
        for generator in self.generators() {
            if generator.is_even() {
                generators.push(generator.clone());
                generators.push(odd.compose(generator)?.compose(&odd_inverse)?);
            } else {
                generators.push(generator.compose(&odd_inverse)?);
                generators.push(odd.compose(generator)?);
            }
        }
        generators.retain(|generator| !generator.is_identity());
        let kernel = PermGroup::from_generators(self.degree(), &generators)?;
        let (expected_order, remainder) = self.order().divmod_u64(2);
        if remainder != 0
            || kernel.order() != expected_order
            || kernel
                .generators()
                .iter()
                .any(|generator| !generator.is_even())
            || !is_subgroup(self, &kernel)?
        {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(kernel)
    }

    /// A deterministic Sylow `p`-subgroup in this concrete permutation group.
    ///
    /// The returned subgroup acts on the same domain and is literally
    /// contained in `self`. The exact native lane enumerates groups of order
    /// at most [`SUBGROUP_ENUMERATION_CAP`] and fails before partial work above
    /// that bound. Full symmetric groups use their standard iterated-wreath
    /// Sylow subgroup and do not enumerate the ambient group.
    ///
    /// # Errors
    /// - [`GroupError::InvalidPrime`] when `prime` is not prime;
    /// - [`GroupError::EnumerationLimitExceeded`] above the exact native cap;
    /// - construction or membership errors from the permutation-group kernel.
    pub fn sylow_subgroup(&self, prime: u64) -> Result<PermGroup> {
        if !is_prime(prime) {
            return Err(GroupError::InvalidPrime { value: prime });
        }
        let mut quotient = self.order();
        let mut target = crate::BigUInt::one();
        loop {
            let (next, remainder) = quotient.divmod_u64(prime);
            if remainder != 0 {
                break;
            }
            quotient = next;
            target.mul_assign_u64(prime);
        }
        if target.is_one() {
            return PermGroup::from_generators(self.degree(), &[]);
        }

        if symmetric_group_order(self.degree()).as_ref() == Some(&self.order()) {
            let generators = symmetric_sylow_generators(self.degree(), prime)
                .ok_or(GroupError::WitnessUnverified)?;
            let subgroup = PermGroup::from_generators(self.degree(), &generators)?;
            if subgroup.order() != target || !is_subgroup(self, &subgroup)? {
                return Err(GroupError::WitnessUnverified);
            }
            return Ok(subgroup);
        }

        if self.is_full_alternating() && prime != 2 {
            let generators = symmetric_sylow_generators(self.degree(), prime)
                .ok_or(GroupError::WitnessUnverified)?;
            let subgroup = PermGroup::from_generators(self.degree(), &generators)?;
            if subgroup.order() != target || !is_subgroup(self, &subgroup)? {
                return Err(GroupError::WitnessUnverified);
            }
            return Ok(subgroup);
        }

        // A candidate that extends a p-group to another p-group must itself
        // have p-power order. Filtering once removes the usually much larger
        // complement of elements that can never be admitted.
        let elements = checked_elements(self, "Sylow-subgroup construction")?
            .into_iter()
            .filter(|element| is_power_of(&element.order(), prime))
            .collect::<Vec<_>>();
        let mut subgroup = PermGroup::from_generators(self.degree(), &[])?;
        while subgroup.order() != target {
            let mut extension = None;
            for element in &elements {
                if subgroup.contains(element)? {
                    continue;
                }
                // If g normalizes P and g^p lies in P, then gP has order p
                // in N_G(P)/P. Hence <P,g> is a p-subgroup of order p|P|.
                // Conversely, a proper p-subgroup P of a Sylow subgroup has
                // a strictly larger normalizer there, so such a g exists
                // until the full p-part of |G| has been reached.
                if !normalizes_subgroup(element, &subgroup)?
                    || !subgroup.contains(&permutation_power(element, prime))?
                {
                    continue;
                }
                let generators = subgroup
                    .generators()
                    .iter()
                    .cloned()
                    .chain(core::iter::once(element.clone()))
                    .collect::<Vec<_>>();
                let candidate = PermGroup::from_generators(self.degree(), &generators)?;
                let mut expected_order = subgroup.order();
                expected_order.mul_assign_u64(prime);
                if candidate.order() != expected_order {
                    return Err(GroupError::WitnessUnverified);
                }
                extension = Some(candidate);
                break;
            }
            subgroup = extension.ok_or(GroupError::WitnessUnverified)?;
        }
        if !is_subgroup(self, &subgroup)? || subgroup.order() != target {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(subgroup)
    }

    /// Literal intersection with another subgroup of the same symmetric
    /// group.
    ///
    /// The result acts on the same point set and contains exactly the
    /// permutations lying in both inputs. Exact generator containment returns
    /// a known contained input directly; otherwise the smaller input is
    /// enumerated, subject to [`SUBGROUP_ENUMERATION_CAP`].
    ///
    /// # Errors
    /// - [`GroupError::DegreeMismatch`] for different permutation domains;
    /// - [`GroupError::EnumerationLimitExceeded`] above the exact native cap;
    /// - construction or membership errors from the permutation-group kernel.
    pub fn intersection(&self, other: &PermGroup) -> Result<PermGroup> {
        if self.degree() != other.degree() {
            return Err(GroupError::DegreeMismatch {
                got: other.degree(),
                expected: self.degree(),
            });
        }
        let (smaller, larger) = if self.order() <= other.order() {
            (self, other)
        } else {
            (other, self)
        };
        if is_subgroup(larger, smaller)? {
            return Ok(smaller.clone());
        }
        if larger.is_full_alternating() {
            return smaller.even_subgroup();
        }
        let mut intersection = PermGroup::from_generators(self.degree(), &[])?;
        for element in checked_elements(smaller, "permutation-group intersection")? {
            if larger.contains(&element)? && !intersection.contains(&element)? {
                intersection = intersection.adjoin_generator(&element)?;
            }
        }
        if !is_subgroup(self, &intersection)? || !is_subgroup(other, &intersection)? {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(intersection)
    }
}

/// Largest ambient group order accepted by exact double-coset enumeration.
///
/// The implementation materializes one compact table of ambient elements and
/// reuses it for every subgroup pair. This cap bounds memory and running time
/// before enumeration starts; [`DOUBLE_COSET_QUOTIENT_CAP`] independently
/// bounds each right-coset quotient.
pub const DOUBLE_COSET_AMBIENT_CAP: usize = 12_000_000;

/// Largest right-coset quotient materialized by exact double-coset enumeration.
pub const DOUBLE_COSET_QUOTIENT_CAP: usize = 12_000_000;

/// One exact double coset `H g K` inside a concrete permutation group.
///
/// `right_coset_count` is the number of right `K`-cosets in `H g K`, so
/// `size = right_coset_count * |K|`.  The representative is the
/// lexicographically least Schreier-sift normal form among those right cosets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DoubleCoset {
    degree: usize,
    representative: Perm,
    inverse_representative: Option<Perm>,
    right_coset_count: usize,
    size: BigUInt,
}

impl DoubleCoset {
    /// Degree of the common permutation action.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Canonical element representing this double coset.
    pub fn representative(&self) -> &Perm {
        &self.representative
    }

    /// Canonical representative of the inverse double coset when the left and
    /// right subgroups are equal.
    pub fn inverse_representative(&self) -> Option<&Perm> {
        self.inverse_representative.as_ref()
    }

    /// Number of right-subgroup cosets contained in this double coset.
    pub fn right_coset_count(&self) -> usize {
        self.right_coset_count
    }

    /// Exact number of ambient group elements in this double coset.
    pub fn size(&self) -> &BigUInt {
        &self.size
    }

    /// Recompute and verify this double coset exactly.
    pub fn verify(&self, ambient: &PermGroup, left: &PermGroup, right: &PermGroup) -> Result<()> {
        if ambient.degree() != self.degree {
            return Err(GroupError::DegreeMismatch {
                got: ambient.degree(),
                expected: self.degree,
            });
        }
        if !ambient.contains(&self.representative)? {
            return Err(GroupError::ElementNotInGroup);
        }
        let rebuilt = double_cosets(ambient, left, right)?;
        if !rebuilt.contains(self) {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(())
    }
}

/// Is every generator of `subgroup` a member of `group`? — literal containment
/// `H ≤ G` (same point set, no conjugation). `O(|gens(H)|)` membership sifts.
///
/// # Errors
/// [`GroupError::DegreeMismatch`] if the two groups act on different domains.
pub fn is_subgroup(group: &PermGroup, subgroup: &PermGroup) -> Result<bool> {
    if group.degree() != subgroup.degree() {
        return Err(GroupError::DegreeMismatch {
            got: subgroup.degree(),
            expected: group.degree(),
        });
    }
    for g in subgroup.generators() {
        if !group.contains(g)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A left transversal of `subgroup` `H` in `group` `G`: one representative per
/// left coset `gH`, the identity representing `H` itself. `result.len()` is the
/// index `[G:H]`.
///
/// BFS the coset graph: from a known representative `r`, each generator `s` of
/// `G` reaches the coset `(s·r)H`; a candidate `c` opens a new coset iff
/// `p⁻¹·c ∉ H` for every representative `p` collected so far. The orbit of the
/// trivial coset under left multiplication is all of `G/H`, so the result is a
/// complete, irredundant transversal.
///
/// # Errors
/// - [`GroupError::DegreeMismatch`] on a domain-size mismatch.
/// - [`GroupError::NotASubgroup`] if `H ⊄ G`.
/// - [`GroupError::IndexTooLarge`] if the index exceeds the internal cap.
pub fn coset_transversal(group: &PermGroup, subgroup: &PermGroup) -> Result<Vec<Perm>> {
    let degree = group.degree();
    if subgroup.degree() != degree {
        return Err(GroupError::DegreeMismatch {
            got: subgroup.degree(),
            expected: degree,
        });
    }
    if !is_subgroup(group, subgroup)? {
        return Err(GroupError::NotASubgroup);
    }
    let id = Perm::identity(degree);
    let mut reps: Vec<Perm> = vec![id.clone()];
    let mut queue: Vec<Perm> = vec![id];
    while let Some(r) = queue.pop() {
        for s in group.generators() {
            let cand = s.compose(&r)?; // s · r
            let mut new_coset = true;
            for p in &reps {
                // same left coset iff p⁻¹·cand ∈ H
                if subgroup.contains(&p.inverse().compose(&cand)?)? {
                    new_coset = false;
                    break;
                }
            }
            if new_coset {
                if reps.len() >= MAX_INDEX {
                    return Err(GroupError::IndexTooLarge { cap: MAX_INDEX });
                }
                reps.push(cand.clone());
                queue.push(cand);
            }
        }
    }
    Ok(reps)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ElementKey {
    Packed(u128),
    Images(Perm),
}

fn packed_image_bits(degree: usize) -> Option<u32> {
    let bits = usize::BITS - degree.saturating_sub(1).leading_zeros();
    (degree.checked_mul(bits as usize)? <= u128::BITS as usize).then_some(bits)
}

fn element_key(element: &Perm, packed_bits: Option<u32>) -> ElementKey {
    let Some(bits) = packed_bits else {
        return ElementKey::Images(element.clone());
    };
    let mut packed = 0u128;
    for (index, image) in element.images().iter().copied().enumerate() {
        packed |= (image as u128) << (index as u32 * bits);
    }
    ElementKey::Packed(packed)
}

struct AmbientElementTable {
    degree: usize,
    packed_bits: Option<u32>,
    elements: Vec<Perm>,
    element_index: HashMap<ElementKey, usize>,
}

impl AmbientElementTable {
    fn build(ambient: &PermGroup) -> Result<Self> {
        let degree = ambient.degree();
        if ambient.order() > BigUInt::from(DOUBLE_COSET_AMBIENT_CAP as u64) {
            return Err(GroupError::EnumerationLimitExceeded {
                operation: "double-coset ambient enumeration",
                cap: DOUBLE_COSET_AMBIENT_CAP,
            });
        }
        let ambient_order = ambient.order().to_decimal().parse::<usize>().map_err(|_| {
            GroupError::EnumerationLimitExceeded {
                operation: "double-coset ambient enumeration",
                cap: DOUBLE_COSET_AMBIENT_CAP,
            }
        })?;
        let packed_bits = packed_image_bits(degree);
        let mut elements = Vec::with_capacity(ambient_order);
        let mut element_index = HashMap::with_capacity(ambient_order);
        for element in ambient.iter_elements() {
            let index = elements.len();
            if element_index
                .insert(element_key(&element, packed_bits), index)
                .is_some()
            {
                return Err(GroupError::WitnessUnverified);
            }
            elements.push(element);
        }
        if elements.len() != ambient_order || element_index.len() != ambient_order {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(Self {
            degree,
            packed_bits,
            elements,
            element_index,
        })
    }

    fn index(&self, element: &Perm) -> Result<usize> {
        self.element_index
            .get(&element_key(element, self.packed_bits))
            .copied()
            .ok_or(GroupError::WitnessUnverified)
    }
}

struct RightCosetSpace<'a> {
    ambient: &'a AmbientElementTable,
    right: PermGroup,
    right_order: usize,
    representatives: Vec<usize>,
    element_to_coset: Vec<u32>,
}

impl<'a> RightCosetSpace<'a> {
    fn build(
        ambient_group: &PermGroup,
        ambient: &'a AmbientElementTable,
        right: &PermGroup,
    ) -> Result<Self> {
        if right.degree() != ambient.degree {
            return Err(GroupError::DegreeMismatch {
                got: right.degree(),
                expected: ambient.degree,
            });
        }
        if !is_subgroup(ambient_group, right)? {
            return Err(GroupError::NotASubgroup);
        }
        let right_order = right
            .order()
            .to_decimal()
            .parse::<usize>()
            .map_err(|_| GroupError::WitnessUnverified)?;
        if right_order == 0 || !ambient.elements.len().is_multiple_of(right_order) {
            return Err(GroupError::WitnessUnverified);
        }
        let quotient_size = ambient.elements.len() / right_order;
        if quotient_size > DOUBLE_COSET_QUOTIENT_CAP {
            return Err(GroupError::EnumerationLimitExceeded {
                operation: "double-coset right-quotient enumeration",
                cap: DOUBLE_COSET_QUOTIENT_CAP,
            });
        }
        let mut representatives = Vec::with_capacity(quotient_size);
        let mut element_to_coset = vec![u32::MAX; ambient.elements.len()];
        for seed in 0..ambient.elements.len() {
            if element_to_coset[seed] != u32::MAX {
                continue;
            }
            let coset_index =
                u32::try_from(representatives.len()).map_err(|_| GroupError::WitnessUnverified)?;
            let element = &ambient.elements[seed];
            let mut representative = seed;
            for member in right.iter_elements() {
                let coset_member = element.compose(&member)?;
                let member_index = ambient.index(&coset_member)?;
                if element_to_coset[member_index] != u32::MAX {
                    return Err(GroupError::WitnessUnverified);
                }
                element_to_coset[member_index] = coset_index;
                if coset_member.images() < ambient.elements[representative].images() {
                    representative = member_index;
                }
            }
            representatives.push(representative);
        }
        if representatives.len() != quotient_size || element_to_coset.contains(&u32::MAX) {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(Self {
            ambient,
            right: right.clone(),
            right_order,
            representatives,
            element_to_coset,
        })
    }

    fn coset_index(&self, element: &Perm) -> Result<usize> {
        let ambient_index = self.ambient.index(element)?;
        let coset_index = self.element_to_coset[ambient_index];
        usize::try_from(coset_index).map_err(|_| GroupError::WitnessUnverified)
    }

    fn double_cosets(&self, ambient: &PermGroup, left: &PermGroup) -> Result<Vec<DoubleCoset>> {
        if left.degree() != self.ambient.degree {
            return Err(GroupError::DegreeMismatch {
                got: left.degree(),
                expected: self.ambient.degree,
            });
        }
        if !is_subgroup(ambient, left)? {
            return Err(GroupError::NotASubgroup);
        }
        let mut assignment = vec![usize::MAX; self.representatives.len()];
        let mut result = Vec::new();
        for seed in 0..self.representatives.len() {
            if assignment[seed] != usize::MAX {
                continue;
            }
            let double_coset_index = result.len();
            assignment[seed] = double_coset_index;
            let mut queue = VecDeque::from([seed]);
            let mut representative = self.representatives[seed];
            let mut right_coset_count = 0usize;
            while let Some(current) = queue.pop_front() {
                right_coset_count += 1;
                let current_representative = &self.ambient.elements[self.representatives[current]];
                if current_representative.images() < self.ambient.elements[representative].images()
                {
                    representative = self.representatives[current];
                }
                for generator in left.generators() {
                    let image = generator.compose(current_representative)?;
                    let target = self.coset_index(&image)?;
                    if assignment[target] == usize::MAX {
                        assignment[target] = double_coset_index;
                        queue.push_back(target);
                    } else if assignment[target] != double_coset_index {
                        return Err(GroupError::WitnessUnverified);
                    }
                }
            }
            result.push(DoubleCoset {
                degree: self.ambient.degree,
                representative: self.ambient.elements[representative].clone(),
                inverse_representative: None,
                right_coset_count,
                size: BigUInt::from(self.right_order as u64)
                    .mul(&BigUInt::from(right_coset_count as u64)),
            });
        }
        if same_subgroup(left, &self.right)? {
            let inverse_representatives = result
                .iter()
                .map(|double_coset| {
                    let inverse = double_coset.representative.inverse();
                    let inverse_coset = self.coset_index(&inverse)?;
                    let partner = assignment[inverse_coset];
                    result
                        .get(partner)
                        .map(|double_coset| double_coset.representative.clone())
                        .ok_or(GroupError::WitnessUnverified)
                })
                .collect::<Result<Vec<_>>>()?;
            for (double_coset, inverse_representative) in
                result.iter_mut().zip(inverse_representatives)
            {
                double_coset.inverse_representative = Some(inverse_representative);
            }
        }
        result.sort_by(|left, right| {
            left.representative
                .images()
                .cmp(right.representative.images())
        });
        Ok(result)
    }
}

fn same_subgroup(left: &PermGroup, right: &PermGroup) -> Result<bool> {
    if left.degree() != right.degree() || left.order() != right.order() {
        return Ok(false);
    }
    Ok(is_subgroup(left, right)? && is_subgroup(right, left)?)
}

/// Enumerate the exact double cosets `left \ ambient / right`.
///
/// The algorithm acts with `left` on the right-coset quotient
/// `ambient / right`.  It never constructs a permutation whose degree is the
/// quotient size: one compact exact ambient-element table is shared across
/// all right-coset quotients, and each quotient is released after its row.
/// This is the bounded-memory form needed when a small faithful action
/// describes a group whose geometric orbit has millions of points.
///
/// Results are sorted by the image vector of their canonical representative.
/// Their sizes partition `|ambient|` exactly.
///
/// # Errors
///
/// - [`GroupError::DegreeMismatch`] if the three actions have different degrees.
/// - [`GroupError::NotASubgroup`] unless both subgroups lie in `ambient`.
/// - [`GroupError::EnumerationLimitExceeded`] before an ambient group larger
///   than [`DOUBLE_COSET_AMBIENT_CAP`] is materialized, or before more than
///   [`DOUBLE_COSET_QUOTIENT_CAP`] right cosets are retained.
pub fn double_cosets(
    ambient: &PermGroup,
    left: &PermGroup,
    right: &PermGroup,
) -> Result<Vec<DoubleCoset>> {
    let degree = ambient.degree();
    for subgroup in [left, right] {
        if subgroup.degree() != degree {
            return Err(GroupError::DegreeMismatch {
                got: subgroup.degree(),
                expected: degree,
            });
        }
        if !is_subgroup(ambient, subgroup)? {
            return Err(GroupError::NotASubgroup);
        }
    }
    let ambient_elements = AmbientElementTable::build(ambient)?;
    let result =
        RightCosetSpace::build(ambient, &ambient_elements, right)?.double_cosets(ambient, left)?;
    let covered_order = result.iter().fold(BigUInt::zero(), |sum, double_coset| {
        sum.add(double_coset.size())
    });
    if covered_order != ambient.order() {
        return Err(GroupError::WitnessUnverified);
    }
    Ok(result)
}

/// Exact lower-triangular table of double cosets for several subgroups.
///
/// Entry `result[j][i]`, for `i <= j`, is `subgroups[i] \ ambient /
/// subgroups[j]`.  The ambient element table is enumerated once and reused
/// across every subgroup pair, avoiding repeated materialization of the same
/// small faithful group action.
pub fn double_coset_triangle(
    ambient: &PermGroup,
    subgroups: &[PermGroup],
) -> Result<Vec<Vec<Vec<DoubleCoset>>>> {
    let mut table = (0..subgroups.len())
        .map(|right_index| (0..=right_index).map(|_| Vec::new()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    visit_double_coset_triangle(ambient, subgroups, |left, right, double_coset| {
        table[right][left].push(double_coset);
    })?;
    Ok(table)
}

/// Visit an exact lower-triangular double-coset table without retaining it.
///
/// The callback receives `(left_index, right_index, double_coset)` in
/// deterministic row-major order. Each subgroup-pair partition is verified
/// against the exact ambient order before any of its values are emitted. This
/// is the bounded-memory interface for applications that can terminally
/// classify or aggregate each double coset as it arrives.
///
/// The callback is deliberately infallible. Applications that need to record
/// their own failure can retain it externally and ignore later visits; all
/// group-theoretic failures still return as [`GroupError`].
pub fn visit_double_coset_triangle<F>(
    ambient: &PermGroup,
    subgroups: &[PermGroup],
    mut visit: F,
) -> Result<()>
where
    F: FnMut(usize, usize, DoubleCoset),
{
    for subgroup in subgroups {
        if subgroup.degree() != ambient.degree() {
            return Err(GroupError::DegreeMismatch {
                got: subgroup.degree(),
                expected: ambient.degree(),
            });
        }
        if !is_subgroup(ambient, subgroup)? {
            return Err(GroupError::NotASubgroup);
        }
    }
    let ambient_elements = AmbientElementTable::build(ambient)?;
    for (right_index, right) in subgroups.iter().enumerate() {
        let quotient = RightCosetSpace::build(ambient, &ambient_elements, right)?;
        for (left_index, left) in subgroups[..=right_index].iter().enumerate() {
            let double_cosets = quotient.double_cosets(ambient, left)?;
            let covered_order = double_cosets
                .iter()
                .fold(BigUInt::zero(), |sum, double_coset| {
                    sum.add(double_coset.size())
                });
            if covered_order != ambient.order() {
                return Err(GroupError::WitnessUnverified);
            }
            for double_coset in double_cosets {
                visit(left_index, right_index, double_coset);
            }
        }
    }
    Ok(())
}

impl PermGroup {
    /// Exact double cosets of two literal subgroups of `self`.
    pub fn double_cosets(&self, left: &PermGroup, right: &PermGroup) -> Result<Vec<DoubleCoset>> {
        double_cosets(self, left, right)
    }

    /// Exact lower-triangular double-coset table for several subgroups.
    pub fn double_coset_triangle(
        &self,
        subgroups: &[PermGroup],
    ) -> Result<Vec<Vec<Vec<DoubleCoset>>>> {
        double_coset_triangle(self, subgroups)
    }
}

/// The **subdegrees** of `group`: the ascending multiset of orbit lengths of the
/// stabilizer of point `0` acting on `0..degree`. For a transitive group the
/// smallest is `1` (the fixed point) and the lengths sum to `degree`.
///
/// A conjugacy invariant: it depends only on the conjugacy class of `G ≤ S_n`,
/// and can separate cycle-type-equivalent transitive groups whose point
/// stabilizers have different orbit structure. Returns `[]` for `degree 0`.
///
/// # Errors
/// Propagates [`PermGroup::pointwise_stabilizer`] / [`PermGroup::orbit`] errors.
pub fn subdegrees(group: &PermGroup) -> Result<Vec<usize>> {
    let degree = group.degree();
    if degree == 0 {
        return Ok(Vec::new());
    }
    let stab = group.pointwise_stabilizer(&[0])?;
    let mut seen = vec![false; degree];
    let mut lengths: Vec<usize> = Vec::new();
    for start in 0..degree as Point {
        if seen[start as usize] {
            continue;
        }
        let orb = stab.orbit(start)?;
        for &x in &orb {
            seen[x as usize] = true;
        }
        lengths.push(orb.len());
    }
    lengths.sort_unstable();
    Ok(lengths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perm::Point;

    fn p(img: &[usize]) -> Perm {
        Perm::from_images(&img.iter().map(|&x| x as Point).collect::<Vec<_>>()).unwrap()
    }
    fn grp(degree: usize, gens: &[&[usize]]) -> PermGroup {
        PermGroup::from_generators(degree, &gens.iter().map(|g| p(g)).collect::<Vec<_>>()).unwrap()
    }
    fn s_n(n: usize) -> PermGroup {
        // (0 1) and the n-cycle (0 1 … n-1) generate S_n.
        let transp: Vec<usize> = (0..n)
            .map(|i| {
                if i == 0 {
                    1
                } else if i == 1 {
                    0
                } else {
                    i
                }
            })
            .collect();
        let cycle: Vec<usize> = (0..n).map(|i| (i + 1) % n).collect();
        grp(n, &[&transp, &cycle])
    }
    fn a4() -> PermGroup {
        grp(4, &[&[1, 2, 0, 3], &[0, 2, 3, 1]])
    }
    fn d4() -> PermGroup {
        // <(0 1 2 3), (0 2)> — dihedral of order 8 on 4 points.
        grp(4, &[&[1, 2, 3, 0], &[2, 1, 0, 3]])
    }

    #[test]
    fn is_subgroup_basic() {
        let s4 = s_n(4);
        assert!(is_subgroup(&s4, &a4()).unwrap()); // A4 ≤ S4
        assert!(is_subgroup(&s4, &d4()).unwrap()); // D4 ≤ S4
        assert!(!is_subgroup(&a4(), &d4()).unwrap()); // D4 ⊄ A4 (D4 has odd perms)
        assert!(is_subgroup(&s4, &s4).unwrap()); // reflexive
    }

    #[test]
    fn transversal_index_is_group_quotient() {
        // [S4 : A4] = 2, [S4 : D4] = 3, [S4 : 1] = 24.
        let s4 = s_n(4);
        assert_eq!(coset_transversal(&s4, &a4()).unwrap().len(), 2);
        assert_eq!(coset_transversal(&s4, &d4()).unwrap().len(), 3);
        let triv = grp(4, &[]);
        assert_eq!(coset_transversal(&s4, &triv).unwrap().len(), 24);
        // [S5 : A5] = 2.
        assert_eq!(
            coset_transversal(&s_n(5), &grp(5, &[&[1, 2, 3, 4, 0], &[0, 1, 3, 4, 2]]))
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn transversal_reps_are_distinct_cosets_and_cover_g() {
        // The reps are pairwise in different left cosets, and every group element
        // lies in exactly one rep's coset. Check on S4 / D4 (index 3).
        let s4 = s_n(4);
        let d4 = d4();
        let reps = coset_transversal(&s4, &d4).unwrap();
        // pairwise distinct cosets: r_i^{-1} r_j ∉ D4 for i != j
        for (i, ri) in reps.iter().enumerate() {
            for (j, rj) in reps.iter().enumerate() {
                let in_h = d4.contains(&ri.inverse().compose(rj).unwrap()).unwrap();
                assert_eq!(in_h, i == j);
            }
        }
        // |reps| * |H| == |G|
        assert_eq!(reps.len() as u64 * 8, 24);
    }

    #[test]
    fn coset_transversal_rejects_non_subgroup() {
        // D4 ⊄ A4: a transversal of D4 in A4 is undefined.
        assert_eq!(
            coset_transversal(&a4(), &d4()).unwrap_err(),
            GroupError::NotASubgroup
        );
    }

    #[test]
    fn right_coset_space_partitions_group_elements() {
        let ambient = s_n(4);
        let subgroup = d4();
        let elements = AmbientElementTable::build(&ambient).unwrap();
        let space = RightCosetSpace::build(&ambient, &elements, &subgroup).unwrap();
        assert_eq!(space.representatives.len(), 3);
        for element in ambient.iter_elements() {
            let expected = space.coset_index(&element).unwrap();
            for member in subgroup.iter_elements() {
                let same_coset = element.compose(&member).unwrap();
                assert_eq!(space.coset_index(&same_coset).unwrap(), expected);
            }
        }
    }

    #[test]
    fn exact_double_cosets_partition_the_ambient_group() {
        let ambient = s_n(4);
        let left = d4();
        let right = a4();
        let cosets = ambient.double_cosets(&left, &right).unwrap();
        assert_eq!(cosets.len(), 1);
        assert_eq!(cosets[0].right_coset_count(), 2);
        assert_eq!(cosets[0].size(), &BigUInt::from(24));
        cosets[0].verify(&ambient, &left, &right).unwrap();

        let trivial = grp(4, &[]);
        let regular = grp(4, &[&[1, 2, 3, 0]]);
        let cosets = regular.double_cosets(&trivial, &trivial).unwrap();
        assert_eq!(cosets.len(), 4);
        assert!(cosets.iter().all(|coset| coset.size() == &BigUInt::one()));
        for coset in &cosets {
            assert!(coset.inverse_representative().is_some());
            let inverse = coset.representative().inverse();
            assert_eq!(coset.inverse_representative(), Some(&inverse));
        }
    }

    #[test]
    fn triangular_batch_matches_individual_double_cosets() {
        let ambient = s_n(4);
        let subgroups = vec![grp(4, &[]), d4(), a4()];
        let triangle = ambient.double_coset_triangle(&subgroups).unwrap();
        assert_eq!(triangle.len(), subgroups.len());
        for (right_index, row) in triangle.iter().enumerate() {
            assert_eq!(row.len(), right_index + 1);
            for (left_index, batched) in row.iter().enumerate() {
                assert_eq!(
                    batched,
                    &ambient
                        .double_cosets(&subgroups[left_index], &subgroups[right_index])
                        .unwrap()
                );
            }
        }
    }

    #[test]
    fn streaming_triangle_matches_materialized_triangle() {
        let ambient = s_n(4);
        let subgroups = vec![a4(), d4(), grp(4, &[])];
        let expected = double_coset_triangle(&ambient, &subgroups).unwrap();
        let mut visited = (0..subgroups.len())
            .map(|right_index| (0..=right_index).map(|_| Vec::new()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        visit_double_coset_triangle(&ambient, &subgroups, |left, right, double_coset| {
            visited[right][left].push(double_coset);
        })
        .unwrap();
        assert_eq!(visited, expected);
    }

    #[test]
    fn subdegrees_textbook() {
        // S4 is 2-transitive: point stabilizer (≅ S3) is transitive on the other
        // 3 points → subdegrees [1, 3].
        assert_eq!(subdegrees(&s_n(4)).unwrap(), vec![1, 3]);
        // S5 likewise 2-transitive → [1, 4].
        assert_eq!(subdegrees(&s_n(5)).unwrap(), vec![1, 4]);
        // C4 = <(0 1 2 3)> regular on 4 points: stabilizer of 0 is trivial,
        // every point a singleton orbit → [1, 1, 1, 1].
        assert_eq!(
            subdegrees(&grp(4, &[&[1, 2, 3, 0]])).unwrap(),
            vec![1, 1, 1, 1]
        );
        // D4 on 4 points: stabilizer of 0 has order 2, fixing 0 and 2, swapping
        // 1,3 → orbits {0},{2},{1,3} → subdegrees [1, 1, 2].
        assert_eq!(subdegrees(&d4()).unwrap(), vec![1, 1, 2]);
    }

    #[test]
    fn subdegrees_sum_to_degree() {
        for g in [s_n(4), s_n(5), a4(), d4()] {
            assert_eq!(subdegrees(&g).unwrap().iter().sum::<usize>(), g.degree());
        }
    }

    #[test]
    fn sylow_subgroups_have_the_full_prime_part_and_are_literal_subgroups() {
        let s4 = s_n(4);
        let sylow_two = s4.sylow_subgroup(2).unwrap();
        let sylow_three = s4.sylow_subgroup(3).unwrap();
        let sylow_five = s4.sylow_subgroup(5).unwrap();
        assert_eq!(sylow_two.order(), crate::BigUInt::from(8_u64));
        assert_eq!(sylow_three.order(), crate::BigUInt::from(3_u64));
        assert_eq!(sylow_five.order(), crate::BigUInt::one());
        assert!(is_subgroup(&s4, &sylow_two).unwrap());
        assert!(is_subgroup(&s4, &sylow_three).unwrap());
        assert_eq!(
            s4.sylow_subgroup(4).unwrap_err(),
            GroupError::InvalidPrime { value: 4 }
        );

        let s8 = s_n(8);
        let sylow_two = s8.sylow_subgroup(2).unwrap();
        assert_eq!(sylow_two.order(), crate::BigUInt::from(128_u64));
        assert!(is_subgroup(&s8, &sylow_two).unwrap());

        let alternating = a4();
        assert_eq!(
            alternating.sylow_subgroup(2).unwrap().order(),
            BigUInt::from(4_u64)
        );
        assert_eq!(
            alternating.sylow_subgroup(3).unwrap().order(),
            BigUInt::from(3_u64)
        );
    }

    #[test]
    fn symmetric_sylow_wreath_construction_matches_legendre_orders() {
        for degree in 2..=10 {
            let symmetric = s_n(degree);
            for prime in [2usize, 3, 5, 7] {
                let mut expected = 1u64;
                for factor in 2..=degree {
                    let mut quotient = factor;
                    while quotient.is_multiple_of(prime) {
                        expected *= prime as u64;
                        quotient /= prime;
                    }
                }
                let sylow = symmetric.sylow_subgroup(prime as u64).unwrap();
                assert_eq!(sylow.order(), BigUInt::from(expected));
                assert!(is_subgroup(&symmetric, &sylow).unwrap());
            }
        }
    }

    #[test]
    fn literal_intersection_is_symmetric_and_exact() {
        let alternating = a4();
        let dihedral = d4();
        let left = alternating.intersection(&dihedral).unwrap();
        let right = dihedral.intersection(&alternating).unwrap();
        assert_eq!(left.order(), crate::BigUInt::from(4_u64));
        assert_eq!(right.order(), left.order());
        let left_elements = left
            .iter_elements()
            .map(|element| element.images().to_vec())
            .collect::<std::collections::BTreeSet<_>>();
        let right_elements = right
            .iter_elements()
            .map(|element| element.images().to_vec())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(left_elements, right_elements);
        for element in left.iter_elements() {
            assert!(alternating.contains(&element).unwrap());
            assert!(dihedral.contains(&element).unwrap());
        }
    }

    #[test]
    fn literal_intersection_covers_scorecard_and_dihedral_neighbor() {
        let alternating_five = grp(5, &[&[1, 2, 0, 3, 4], &[1, 2, 3, 4, 0]]);
        let symmetric_four_fixing_last = grp(5, &[&[1, 0, 2, 3, 4], &[1, 2, 3, 0, 4]]);
        let alternating_four = alternating_five
            .intersection(&symmetric_four_fixing_last)
            .unwrap();
        assert_eq!(alternating_four.order(), BigUInt::from(12_u64));
        assert!(alternating_four.iter_elements().all(|element| {
            alternating_five.contains(&element).unwrap()
                && symmetric_four_fixing_last.contains(&element).unwrap()
        }));

        let symmetric_three_fixing_two_points = grp(5, &[&[1, 0, 2, 3, 4], &[1, 2, 0, 3, 4]]);
        let cyclic_three = alternating_five
            .intersection(&symmetric_three_fixing_two_points)
            .unwrap();
        assert_eq!(cyclic_three.order(), BigUInt::from(3_u64));
        assert!(cyclic_three.iter_elements().all(|element| {
            alternating_five.contains(&element).unwrap()
                && symmetric_three_fixing_two_points
                    .contains(&element)
                    .unwrap()
        }));
    }

    #[test]
    fn sign_kernel_handles_symmetric_dihedral_and_odd_order_groups() {
        let alternating_four = s_n(4).even_subgroup().unwrap();
        assert_eq!(alternating_four.order(), BigUInt::from(12_u64));
        assert!(alternating_four
            .iter_elements()
            .all(|element| element.is_even()));

        let even_dihedral = d4().even_subgroup().unwrap();
        assert_eq!(even_dihedral.order(), BigUInt::from(4_u64));
        assert!(even_dihedral
            .iter_elements()
            .all(|element| element.is_even()));

        let cyclic_five = grp(5, &[&[1, 2, 3, 4, 0]]);
        let unchanged = cyclic_five.even_subgroup().unwrap();
        assert_eq!(unchanged.order(), BigUInt::from(5_u64));
        assert_eq!(
            unchanged
                .iter_elements()
                .map(|element| element.images().to_vec())
                .collect::<std::collections::BTreeSet<_>>(),
            cyclic_five
                .iter_elements()
                .map(|element| element.images().to_vec())
                .collect::<std::collections::BTreeSet<_>>()
        );
    }

    #[test]
    fn subgroup_enumeration_operations_fail_before_materializing_large_groups() {
        let symmetric_ten = s_n(10);
        let sylow_two = symmetric_ten.sylow_subgroup(2).unwrap();
        assert_eq!(sylow_two.order(), BigUInt::from(256_u64));
        assert!(is_subgroup(&symmetric_ten, &sylow_two).unwrap());

        // A_10 is above the enumeration cap but is not the full-symmetric fast
        // path: (0 1 2) together with (1 2 ... 9) generates A_10.
        let alternating_ten = grp(
            10,
            &[
                &[1, 2, 0, 3, 4, 5, 6, 7, 8, 9],
                &[0, 2, 3, 4, 5, 6, 7, 8, 9, 1],
            ],
        );
        assert_eq!(alternating_ten.order(), BigUInt::from(1_814_400_u64));
        assert!(matches!(
            alternating_ten.sylow_subgroup(2),
            Err(GroupError::EnumerationLimitExceeded {
                operation: "Sylow-subgroup construction",
                cap: SUBGROUP_ENUMERATION_CAP,
            })
        ));
        assert_eq!(
            symmetric_ten.intersection(&symmetric_ten).unwrap().order(),
            symmetric_ten.order()
        );

        let alternating_ten_fixing_last = grp(
            11,
            &[
                &[1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10],
                &[0, 2, 3, 4, 5, 6, 7, 8, 9, 1, 10],
            ],
        );
        let alternating_ten_fixing_first = grp(
            11,
            &[
                &[0, 2, 3, 1, 4, 5, 6, 7, 8, 9, 10],
                &[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 2],
            ],
        );
        assert!(matches!(
            alternating_ten_fixing_last.intersection(&alternating_ten_fixing_first),
            Err(GroupError::EnumerationLimitExceeded {
                operation: "permutation-group intersection",
                cap: SUBGROUP_ENUMERATION_CAP,
            })
        ));
    }
}
