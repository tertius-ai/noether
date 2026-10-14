// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact induced actions on finite subsets of the permutation domain.

use std::collections::{BTreeSet, VecDeque};

use crate::error::{GroupError, Result};
use crate::group::PermGroup;
use crate::perm::Point;

const SUBSET_ENUMERATION_CAP: usize = 1_000_000;

/// One exact orbit under the induced action on fixed-cardinality subsets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubsetOrbit {
    degree: usize,
    subset_size: usize,
    representative: Vec<Point>,
    elements: Vec<Vec<Point>>,
}

/// One compact exact orbit of point pairs under a diagonal group action.
///
/// The two seed points name invariant point orbits.  When they are distinct
/// group orbits, the pair is color-preserving: the first endpoint remains in
/// the first seed orbit and the second remains in the second.  Passing the
/// same seed twice instead enumerates unordered pairs of distinct points in
/// that one orbit.  `anchored_suborbits` records the one or two
/// `Stab_G(seed)`-orbits which form the corresponding double-coset class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairOrbit {
    degree: usize,
    left_seed: Point,
    right_seed: Point,
    representative: [Point; 2],
    anchored_suborbits: Vec<Vec<Point>>,
    size: usize,
    same_point_orbit: bool,
}

impl PairOrbit {
    /// Degree of the underlying point action.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Seed of the first invariant point orbit.
    pub fn left_seed(&self) -> Point {
        self.left_seed
    }

    /// Seed of the second invariant point orbit.
    pub fn right_seed(&self) -> Point {
        self.right_seed
    }

    /// Canonical endpoint pair representing this orbit.
    pub fn representative(&self) -> &[Point; 2] {
        &self.representative
    }

    /// `Stab_G(left_seed)` suborbits defining this double-coset class.
    ///
    /// Cross-orbit pairs have one suborbit.  An unordered pair orbit inside
    /// one point orbit has one suborbit when its ordered orbital is
    /// self-paired and two when transposition exchanges two orbitals.
    pub fn anchored_suborbits(&self) -> &[Vec<Point>] {
        &self.anchored_suborbits
    }

    /// Number of unordered/color-preserving pairs in this orbit.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Whether both endpoints range through the same invariant point orbit.
    pub fn same_point_orbit(&self) -> bool {
        self.same_point_orbit
    }

    /// Recompute the compact double-coset partition and this orbit exactly.
    pub fn verify(&self, group: &PermGroup) -> Result<()> {
        if group.degree() != self.degree {
            return Err(GroupError::DegreeMismatch {
                got: group.degree(),
                expected: self.degree,
            });
        }
        let rebuilt = pair_orbits(group, self.left_seed, self.right_seed)?;
        if !rebuilt.contains(self) {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(())
    }
}

impl SubsetOrbit {
    /// Degree of the underlying point action.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Common cardinality of the subsets in the orbit.
    pub fn subset_size(&self) -> usize {
        self.subset_size
    }

    /// Lexicographically least subset in the orbit.
    pub fn representative(&self) -> &[Point] {
        &self.representative
    }

    /// Every subset in the orbit, in stable lexical order.
    pub fn elements(&self) -> &[Vec<Point>] {
        &self.elements
    }

    /// Exact orbit cardinality.
    pub fn size(&self) -> usize {
        self.elements.len()
    }

    /// Recompute the orbit and the orbit-stabilizer identity exactly.
    pub fn verify(&self, group: &PermGroup) -> Result<()> {
        if group.degree() != self.degree {
            return Err(GroupError::DegreeMismatch {
                got: group.degree(),
                expected: self.degree,
            });
        }
        let rebuilt = orbit_of_subset(group, &self.representative)?;
        if rebuilt != *self {
            return Err(GroupError::WitnessUnverified);
        }
        let stabilizer = group.set_stabilizer(&self.representative)?;
        let expected_order = stabilizer
            .order()
            .mul(&crate::BigUInt::from(self.size() as u64));
        if expected_order != group.order() {
            return Err(GroupError::WitnessUnverified);
        }
        Ok(())
    }
}

fn canonical_subset(degree: usize, subset: &[Point]) -> Result<Vec<Point>> {
    let mut canonical = subset.to_vec();
    canonical.sort_unstable();
    canonical.dedup();
    if canonical.len() != subset.len() {
        let point = subset
            .iter()
            .copied()
            .find(|point| subset.iter().filter(|item| *item == point).count() > 1)
            .unwrap_or(0);
        return Err(GroupError::DuplicateSubsetPoint {
            point: point as usize,
        });
    }
    if let Some(point) = canonical
        .iter()
        .copied()
        .find(|point| *point as usize >= degree)
    {
        return Err(GroupError::DegreeMismatch {
            got: point as usize,
            expected: degree,
        });
    }
    Ok(canonical)
}

fn image_of_subset(generator: &crate::Perm, subset: &[Point]) -> Vec<Point> {
    let mut image = subset
        .iter()
        .map(|point| generator.apply(*point))
        .collect::<Vec<_>>();
    image.sort_unstable();
    image
}

/// Exact orbit of one subset under `group`.
pub fn orbit_of_subset(group: &PermGroup, subset: &[Point]) -> Result<SubsetOrbit> {
    let representative_candidate = canonical_subset(group.degree(), subset)?;
    let subset_size = representative_candidate.len();
    let mut seen = BTreeSet::from([representative_candidate.clone()]);
    let mut queue = VecDeque::from([representative_candidate]);
    while let Some(current) = queue.pop_front() {
        for generator in group.generators() {
            let image = image_of_subset(generator, &current);
            if seen.insert(image.clone()) {
                if seen.len() > SUBSET_ENUMERATION_CAP {
                    return Err(GroupError::EnumerationLimitExceeded {
                        operation: "orbits_on_subsets",
                        cap: SUBSET_ENUMERATION_CAP,
                    });
                }
                queue.push_back(image);
            }
        }
    }
    let elements = seen.into_iter().collect::<Vec<_>>();
    let representative = elements
        .first()
        .cloned()
        .expect("a subset orbit contains its seed");
    Ok(SubsetOrbit {
        degree: group.degree(),
        subset_size,
        representative,
        elements,
    })
}

fn all_subsets(degree: usize, size: usize) -> Result<BTreeSet<Vec<Point>>> {
    if size > degree {
        return Err(GroupError::InvalidSubsetSize { size, degree });
    }
    let mut count = 1usize;
    for factor in 1..=size.min(degree - size) {
        count = count
            .checked_mul(degree - size.min(degree - size) + factor)
            .and_then(|value| value.checked_div(factor))
            .ok_or(GroupError::EnumerationLimitExceeded {
                operation: "orbits_on_subsets",
                cap: SUBSET_ENUMERATION_CAP,
            })?;
        if count > SUBSET_ENUMERATION_CAP {
            return Err(GroupError::EnumerationLimitExceeded {
                operation: "orbits_on_subsets",
                cap: SUBSET_ENUMERATION_CAP,
            });
        }
    }
    let mut subsets = BTreeSet::new();
    if size == 0 {
        subsets.insert(Vec::new());
        return Ok(subsets);
    }
    let mut current = (0..size as Point).collect::<Vec<_>>();
    loop {
        subsets.insert(current.clone());
        let Some(position) = (0..size)
            .rev()
            .find(|index| current[*index] as usize != degree - size + *index)
        else {
            break;
        };
        current[position] += 1;
        for index in position + 1..size {
            current[index] = current[index - 1] + 1;
        }
    }
    Ok(subsets)
}

/// Every orbit on `size`-subsets, ordered by canonical representative.
pub fn orbits_on_subsets(group: &PermGroup, size: usize) -> Result<Vec<SubsetOrbit>> {
    let mut unseen = all_subsets(group.degree(), size)?;
    let mut orbits = Vec::new();
    while let Some(seed) = unseen.first().cloned() {
        let orbit = orbit_of_subset(group, &seed)?;
        for element in orbit.elements() {
            if !unseen.remove(element) {
                return Err(GroupError::WitnessUnverified);
            }
        }
        orbits.push(orbit);
    }
    Ok(orbits)
}

fn checked_point(group: &PermGroup, point: Point) -> Result<()> {
    if point as usize >= group.degree() {
        return Err(GroupError::DegreeMismatch {
            got: point as usize,
            expected: group.degree(),
        });
    }
    Ok(())
}

fn point_orbit_partition(group: &PermGroup, points: &BTreeSet<Point>) -> Result<Vec<Vec<Point>>> {
    let mut unseen = points.clone();
    let mut blocks = Vec::new();
    while let Some(seed) = unseen.first().copied() {
        let mut orbit = group.orbit(seed)?;
        orbit.sort_unstable();
        for point in &orbit {
            if !unseen.remove(point) {
                return Err(GroupError::WitnessUnverified);
            }
        }
        blocks.push(orbit);
    }
    blocks.sort();
    Ok(blocks)
}

/// Compact diagonal pair orbits between two invariant point orbits.
///
/// If `left_seed` and `right_seed` lie in distinct `G`-orbits, this returns
/// the color-preserving pair orbits in their Cartesian product.  If the two
/// seeds are equal, it returns the orbits of unordered pairs of distinct
/// points in that one `G`-orbit.  Distinct seeds in the same point orbit are
/// rejected because they do not specify whether endpoint colors are meant to
/// be retained.
///
/// The computation is the standard double-coset reduction: after anchoring
/// the first endpoint, pair orbits are suborbits of its point stabilizer.  In
/// the unordered same-orbit case, an orbital and its transpose are merged.
pub fn pair_orbits(
    group: &PermGroup,
    left_seed: Point,
    right_seed: Point,
) -> Result<Vec<PairOrbit>> {
    checked_point(group, left_seed)?;
    checked_point(group, right_seed)?;
    let mut left_orbit = group.orbit(left_seed)?;
    left_orbit.sort_unstable();
    let left_points = left_orbit.iter().copied().collect::<BTreeSet<_>>();
    let same_point_orbit = left_points.contains(&right_seed);
    if same_point_orbit && left_seed != right_seed {
        return Err(GroupError::AmbiguousPairSeeds {
            left: left_seed as usize,
            right: right_seed as usize,
        });
    }

    let stabilizer = group.pointwise_stabilizer(&[left_seed])?;
    if !same_point_orbit {
        let right_points = group
            .orbit(right_seed)?
            .into_iter()
            .collect::<BTreeSet<_>>();
        let blocks = point_orbit_partition(&stabilizer, &right_points)?;
        let mut result = blocks
            .into_iter()
            .map(|block| {
                Ok(PairOrbit {
                    degree: group.degree(),
                    left_seed,
                    right_seed,
                    representative: [left_seed, block[0]],
                    size: left_orbit.len().checked_mul(block.len()).ok_or(
                        GroupError::EnumerationLimitExceeded {
                            operation: "pair_orbits",
                            cap: usize::MAX,
                        },
                    )?,
                    anchored_suborbits: vec![block],
                    same_point_orbit: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        result.sort_by_key(|orbit| orbit.representative);
        let covered = result.iter().try_fold(0usize, |sum, orbit| {
            sum.checked_add(orbit.size)
                .ok_or(GroupError::EnumerationLimitExceeded {
                    operation: "pair_orbits",
                    cap: usize::MAX,
                })
        })?;
        let expected = left_orbit
            .len()
            .checked_mul(group.orbit(right_seed)?.len())
            .ok_or(GroupError::EnumerationLimitExceeded {
                operation: "pair_orbits",
                cap: usize::MAX,
            })?;
        if covered != expected {
            return Err(GroupError::WitnessUnverified);
        }
        return Ok(result);
    }

    let mut other_points = left_points;
    other_points.remove(&left_seed);
    let blocks = point_orbit_partition(&stabilizer, &other_points)?;
    let mut point_to_block = vec![usize::MAX; group.degree()];
    for (index, block) in blocks.iter().enumerate() {
        for point in block {
            point_to_block[*point as usize] = index;
        }
    }
    let transversal = group.transversal(left_seed)?;
    let mut rep_for_point = vec![None; group.degree()];
    for (point, representative) in transversal.orbit.iter().zip(transversal.reps) {
        rep_for_point[*point as usize] = Some(representative);
    }

    let mut consumed = vec![false; blocks.len()];
    let mut result = Vec::new();
    for index in 0..blocks.len() {
        if consumed[index] {
            continue;
        }
        let endpoint = blocks[index][0];
        let transport = rep_for_point[endpoint as usize]
            .as_ref()
            .ok_or(GroupError::WitnessUnverified)?;
        let transposed_endpoint = transport.inverse().apply(left_seed);
        let partner = point_to_block[transposed_endpoint as usize];
        if partner == usize::MAX {
            return Err(GroupError::WitnessUnverified);
        }
        let partner_endpoint = blocks[partner][0];
        let back_transport = rep_for_point[partner_endpoint as usize]
            .as_ref()
            .ok_or(GroupError::WitnessUnverified)?;
        let back = back_transport.inverse().apply(left_seed);
        if point_to_block[back as usize] != index {
            return Err(GroupError::WitnessUnverified);
        }

        consumed[index] = true;
        consumed[partner] = true;
        let mut anchored_suborbits = if partner == index {
            vec![blocks[index].clone()]
        } else {
            vec![blocks[index].clone(), blocks[partner].clone()]
        };
        anchored_suborbits.sort();
        let representative_endpoint = anchored_suborbits
            .iter()
            .flat_map(|block| block.iter())
            .copied()
            .min()
            .ok_or(GroupError::WitnessUnverified)?;
        let ordered_size = left_orbit.len().checked_mul(blocks[index].len()).ok_or(
            GroupError::EnumerationLimitExceeded {
                operation: "pair_orbits",
                cap: usize::MAX,
            },
        )?;
        let size = if partner == index {
            if !ordered_size.is_multiple_of(2) {
                return Err(GroupError::WitnessUnverified);
            }
            ordered_size / 2
        } else {
            if blocks[index].len() != blocks[partner].len() {
                return Err(GroupError::WitnessUnverified);
            }
            ordered_size
        };
        result.push(PairOrbit {
            degree: group.degree(),
            left_seed,
            right_seed,
            representative: [left_seed, representative_endpoint],
            anchored_suborbits,
            size,
            same_point_orbit: true,
        });
    }
    result.sort_by_key(|orbit| orbit.representative);
    let covered = result.iter().try_fold(0usize, |sum, orbit| {
        sum.checked_add(orbit.size)
            .ok_or(GroupError::EnumerationLimitExceeded {
                operation: "pair_orbits",
                cap: usize::MAX,
            })
    })?;
    let expected = left_orbit
        .len()
        .checked_mul(left_orbit.len().saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or(GroupError::EnumerationLimitExceeded {
            operation: "pair_orbits",
            cap: usize::MAX,
        })?;
    if covered != expected {
        return Err(GroupError::WitnessUnverified);
    }
    Ok(result)
}

impl PermGroup {
    /// Exact orbit of one set of points under the induced set action.
    pub fn orbit_of_subset(&self, subset: &[Point]) -> Result<SubsetOrbit> {
        orbit_of_subset(self, subset)
    }

    /// Exact orbit partition of all subsets of a fixed cardinality.
    pub fn orbits_on_subsets(&self, size: usize) -> Result<Vec<SubsetOrbit>> {
        orbits_on_subsets(self, size)
    }

    /// Exact orbit partition of unordered pairs of distinct points.
    pub fn orbits_on_pairs(&self) -> Result<Vec<SubsetOrbit>> {
        orbits_on_subsets(self, 2)
    }

    /// Compact double-coset reduction of pairs between two point orbits.
    pub fn pair_orbits(&self, left_seed: Point, right_seed: Point) -> Result<Vec<PairOrbit>> {
        pair_orbits(self, left_seed, right_seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Perm, PermGroup};

    fn group(degree: usize, generators: &[&[Point]]) -> PermGroup {
        let generators = generators
            .iter()
            .map(|images| Perm::from_images(images).unwrap())
            .collect::<Vec<_>>();
        PermGroup::from_generators(degree, &generators).unwrap()
    }

    #[test]
    fn cyclic_four_has_two_orbits_on_pairs() {
        let cyclic = group(4, &[&[1, 2, 3, 0]]);
        let orbits = cyclic.orbits_on_pairs().unwrap();
        assert_eq!(orbits.len(), 2);
        assert_eq!(orbits[0].representative(), &[0, 1]);
        assert_eq!(orbits[0].size(), 4);
        assert_eq!(orbits[1].representative(), &[0, 2]);
        assert_eq!(orbits[1].size(), 2);
        for orbit in orbits {
            orbit.verify(&cyclic).unwrap();
        }

        let compact = cyclic.pair_orbits(0, 0).unwrap();
        assert_eq!(compact.len(), 2);
        assert_eq!(compact[0].representative(), &[0, 1]);
        assert_eq!(compact[0].size(), 4);
        assert_eq!(compact[0].anchored_suborbits(), &[vec![1], vec![3]]);
        assert_eq!(compact[1].representative(), &[0, 2]);
        assert_eq!(compact[1].size(), 2);
        assert_eq!(compact[1].anchored_suborbits(), &[vec![2]]);
        for orbit in compact {
            orbit.verify(&cyclic).unwrap();
        }
    }

    #[test]
    fn symmetric_group_is_transitive_on_fixed_size_subsets() {
        let symmetric = group(5, &[&[1, 0, 2, 3, 4], &[1, 2, 3, 4, 0]]);
        let orbits = symmetric.orbits_on_subsets(3).unwrap();
        assert_eq!(orbits.len(), 1);
        assert_eq!(orbits[0].size(), 10);
        orbits[0].verify(&symmetric).unwrap();
    }

    #[test]
    fn subset_validation_fails_closed() {
        let cyclic = group(4, &[&[1, 2, 3, 0]]);
        assert!(matches!(
            cyclic.orbits_on_subsets(5),
            Err(GroupError::InvalidSubsetSize { .. })
        ));
        assert!(cyclic.orbit_of_subset(&[0, 0]).is_err());
        assert!(cyclic.orbit_of_subset(&[4]).is_err());
        assert!(matches!(
            cyclic.pair_orbits(0, 1),
            Err(GroupError::AmbiguousPairSeeds { .. })
        ));
    }

    #[test]
    fn compact_cross_pair_orbits_are_double_cosets() {
        // C3 acts regularly and in parallel on two colored triples.  The
        // relative displacement gives three diagonal pair orbits of size 3.
        let parallel = group(6, &[&[1, 2, 0, 4, 5, 3]]);
        let orbits = parallel.pair_orbits(0, 3).unwrap();
        assert_eq!(orbits.len(), 3);
        assert_eq!(
            orbits
                .iter()
                .map(|orbit| (*orbit.representative(), orbit.size()))
                .collect::<Vec<_>>(),
            vec![([0, 3], 3), ([0, 4], 3), ([0, 5], 3)]
        );
        assert_eq!(orbits.iter().map(PairOrbit::size).sum::<usize>(), 9);
        for orbit in orbits {
            orbit.verify(&parallel).unwrap();
        }
    }
}
