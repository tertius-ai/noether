// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact degree-preserving canonicalization and atomic symmetry.

use crate::{
    combin::{Graph, GraphSymmetry},
    groups::{Permutation, PermutationGroup, Point},
};

use super::model::{CategoryError, SteinerCategory, SteinerCell};

const NEGATIVE_INCIDENCE_WEIGHT: i64 = -1;
const POSITIVE_INCIDENCE_WEIGHT: i64 = 1;
const DEGREE_COLOUR_BASE: i64 = 3;

/// Exact canonical representative and automorphism evidence.
///
/// The labeling maps flattened atoms of the submitted category to flattened
/// atoms of the canonical category. The automorphism group instead acts on the
/// submitted flattened atoms; the labeling need not be an automorphism.
#[derive(Debug, Clone)]
pub struct SteinerCanonicalLabel {
    category: SteinerCategory,
    encoding: Vec<u64>,
    labeling: Permutation,
    local_images: Vec<Vec<usize>>,
    automorphism_group: PermutationGroup,
}

impl SteinerCanonicalLabel {
    pub(crate) fn compute(category: &SteinerCategory) -> Result<Self, CategoryError> {
        let graph = signed_incidence_graph(category)?;
        let graph_label = graph.canonical_label()?;
        let local_images = degreewise_images(category, graph_label.labeling())?;
        let canonical = category.relabel(&local_images)?;
        let identity = canonical
            .atom_counts()
            .iter()
            .map(|&count| (0..count).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let encoding = key_under(&canonical, &identity);
        let labeling = flattened_permutation(category, &local_images)?;
        let result = Self {
            category: canonical,
            encoding,
            labeling,
            local_images,
            automorphism_group: graph_label.automorphism_group().clone(),
        };
        if !result.verify(category) {
            return Err(CategoryError::Canonicalization(
                "canonical labeling failed exact verification",
            ));
        }
        Ok(result)
    }

    /// Canonical category.
    #[must_use]
    pub fn category(&self) -> &SteinerCategory {
        &self.category
    }

    /// Version-one canonical unsigned-word encoding.
    ///
    /// The sequence begins with the number of occupied atomic degrees and the
    /// degreewise atom counts. It then stores, column by column, each support
    /// length followed by its sorted degree-local atom indices, negative side
    /// first. Atom order is induced by the exact canonical labeling of the
    /// degree-coloured signed incidence graph.
    #[must_use]
    pub fn encoding(&self) -> &[u64] {
        &self.encoding
    }

    /// Permutation from submitted flattened atom ids to canonical ids.
    #[must_use]
    pub fn labeling(&self) -> &Permutation {
        &self.labeling
    }

    /// Complete automorphism group on submitted flattened atom ids.
    #[must_use]
    pub fn automorphism_group(&self) -> &PermutationGroup {
        &self.automorphism_group
    }

    /// Degree-local atom images from the submitted category to the canonical one.
    #[must_use]
    pub fn degreewise_labeling(&self) -> &[Vec<usize>] {
        &self.local_images
    }

    /// Transport one submitted cell into the canonical category.
    pub fn transport_cell(&self, cell: &SteinerCell) -> Result<SteinerCell, CategoryError> {
        cell.relabel(&self.category, &self.local_images)
    }

    /// Verify the relabeling and every stored group generator.
    #[must_use]
    pub fn verify(&self, original: &SteinerCategory) -> bool {
        if self.labeling.degree() != original.atom_count()
            || original.atom_counts() != self.category.atom_counts()
        {
            return false;
        }
        let identity = self
            .category
            .atom_counts()
            .iter()
            .map(|&count| (0..count).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        if self.encoding != key_under(&self.category, &identity) {
            return false;
        }
        let Ok(relabeled) = original.relabel(&self.local_images) else {
            return false;
        };
        if relabeled != self.category {
            return false;
        }
        self.automorphism_group
            .generators()
            .iter()
            .all(|generator| is_automorphism(original, generator))
    }
}

fn signed_incidence_graph(category: &SteinerCategory) -> Result<Graph, CategoryError> {
    let mut edges = Vec::new();
    let mut offsets = Vec::with_capacity(category.atom_counts().len());
    let mut offset = 0usize;
    for (degree, &count) in category.atom_counts().iter().enumerate() {
        offsets.push(offset);
        let colour = DEGREE_COLOUR_BASE
            .checked_add(i64::try_from(degree).map_err(|_| CategoryError::SizeOverflow)?)
            .ok_or(CategoryError::SizeOverflow)?;
        for atom in 0..count {
            let vertex = u32::try_from(offset + atom).map_err(|_| CategoryError::SizeOverflow)?;
            edges.push((vertex, vertex, colour));
        }
        offset = offset
            .checked_add(count)
            .ok_or(CategoryError::SizeOverflow)?;
    }
    for degree in 1..category.atom_counts().len() {
        for atom in 0..category.atom_count_in_degree(degree) {
            let upper =
                u32::try_from(offsets[degree] + atom).map_err(|_| CategoryError::SizeOverflow)?;
            let boundary = category.boundary(super::model::AtomId::new(degree, atom))?;
            for &lower in boundary.negative() {
                edges.push((
                    u32::try_from(offsets[degree - 1] + lower)
                        .map_err(|_| CategoryError::SizeOverflow)?,
                    upper,
                    NEGATIVE_INCIDENCE_WEIGHT,
                ));
            }
            for &lower in boundary.positive() {
                edges.push((
                    u32::try_from(offsets[degree - 1] + lower)
                        .map_err(|_| CategoryError::SizeOverflow)?,
                    upper,
                    POSITIVE_INCIDENCE_WEIGHT,
                ));
            }
        }
    }
    Graph::from_edges(category.atom_count(), &edges, true).map_err(|_| {
        CategoryError::Canonicalization("could not build the signed atomic incidence graph")
    })
}

fn degreewise_images(
    category: &SteinerCategory,
    graph_labeling: &Permutation,
) -> Result<Vec<Vec<usize>>, CategoryError> {
    let mut output = Vec::with_capacity(category.atom_counts().len());
    let mut offset = 0usize;
    for &count in category.atom_counts() {
        let mut atoms = (0..count)
            .map(|atom| {
                let flattened =
                    u32::try_from(offset + atom).map_err(|_| CategoryError::SizeOverflow)?;
                Ok((graph_labeling.apply(flattened), atom))
            })
            .collect::<Result<Vec<_>, CategoryError>>()?;
        atoms.sort_unstable();
        let mut images = vec![0usize; count];
        for (canonical, (_, original)) in atoms.into_iter().enumerate() {
            images[original] = canonical;
        }
        output.push(images);
        offset = offset
            .checked_add(count)
            .ok_or(CategoryError::SizeOverflow)?;
    }
    Ok(output)
}

fn key_under(category: &SteinerCategory, images: &[Vec<usize>]) -> Vec<u64> {
    let mut key = Vec::new();
    key.push(category.atom_counts().len() as u64);
    key.extend(category.atom_counts().iter().map(|&count| count as u64));
    for degree in 1..category.atom_counts().len() {
        let mut columns = vec![None; category.atom_count_in_degree(degree)];
        for old in 0..category.atom_count_in_degree(degree) {
            let boundary = category
                .boundary(super::model::AtomId::new(degree, old))
                .expect("canonical search uses valid atoms");
            let mut negative = boundary
                .negative()
                .iter()
                .map(|&atom| images[degree - 1][atom])
                .collect::<Vec<_>>();
            let mut positive = boundary
                .positive()
                .iter()
                .map(|&atom| images[degree - 1][atom])
                .collect::<Vec<_>>();
            negative.sort_unstable();
            positive.sort_unstable();
            columns[images[degree][old]] = Some((negative, positive));
        }
        for (negative, positive) in columns
            .into_iter()
            .map(|column| column.expect("a degreewise permutation fills every boundary column"))
        {
            key.push(negative.len() as u64);
            key.extend(negative.into_iter().map(|atom| atom as u64));
            key.push(positive.len() as u64);
            key.extend(positive.into_iter().map(|atom| atom as u64));
        }
    }
    key
}

fn flattened_permutation(
    category: &SteinerCategory,
    images: &[Vec<usize>],
) -> Result<Permutation, CategoryError> {
    let mut flattened = Vec::with_capacity(category.atom_count());
    let mut offset = 0usize;
    for (degree, local) in images.iter().enumerate() {
        if local.len() != category.atom_count_in_degree(degree) {
            return Err(CategoryError::Canonicalization(
                "degreewise permutation has the wrong length",
            ));
        }
        flattened.extend(local.iter().map(|&image| (offset + image) as Point));
        offset += local.len();
    }
    Ok(Permutation::from_images(&flattened)?)
}

pub(crate) fn is_automorphism(category: &SteinerCategory, permutation: &Permutation) -> bool {
    if permutation.degree() != category.atom_count() {
        return false;
    }
    let mut images = category
        .atom_counts()
        .iter()
        .map(|&count| vec![0usize; count])
        .collect::<Vec<_>>();
    for flattened in 0..category.atom_count() {
        let Ok(atom) = category.unflatten_atom(flattened) else {
            return false;
        };
        let image = permutation.apply(flattened as Point) as usize;
        let Ok(image) = category.unflatten_atom(image) else {
            return false;
        };
        if atom.degree() != image.degree() {
            return false;
        }
        images[atom.degree()][atom.index()] = image.index();
    }
    category
        .relabel(&images)
        .is_ok_and(|image| image == *category)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::groups::FiniteGroup;

    fn permutations(count: usize) -> Vec<Vec<usize>> {
        fn visit(
            position: usize,
            permutation: &mut [usize],
            used: &mut [bool],
            output: &mut Vec<Vec<usize>>,
        ) {
            if position == permutation.len() {
                output.push(permutation.to_vec());
                return;
            }
            for image in 0..permutation.len() {
                if !used[image] {
                    used[image] = true;
                    permutation[position] = image;
                    visit(position + 1, permutation, used, output);
                    used[image] = false;
                }
            }
        }
        let mut output = Vec::new();
        visit(0, &mut vec![0; count], &mut vec![false; count], &mut output);
        output
    }

    fn degreewise_labelings(category: &SteinerCategory) -> Vec<Vec<Vec<usize>>> {
        let mut labelings = vec![Vec::new()];
        for &count in category.atom_counts() {
            let local = permutations(count);
            labelings = labelings
                .into_iter()
                .flat_map(|prefix| {
                    local.iter().cloned().map(move |permutation| {
                        let mut labeling = prefix.clone();
                        labeling.push(permutation);
                        labeling
                    })
                })
                .collect();
        }
        labelings
    }

    fn check_exhaustive(category: &SteinerCategory) {
        let canonical = category.canonical_form().unwrap();
        let labelings = degreewise_labelings(category);
        let automorphisms = labelings
            .iter()
            .filter(|labeling| category.relabel(labeling).unwrap() == *category)
            .count();
        assert_eq!(
            FiniteGroup::order(&category.automorphism_group().unwrap()).to_string(),
            automorphisms.to_string()
        );
        for labeling in labelings {
            let relabeled = category.relabel(&labeling).unwrap();
            assert_eq!(relabeled.canonical_form().unwrap(), canonical);
        }
    }

    #[test]
    fn discrete_objects_have_full_symmetric_automorphisms() {
        let category = SteinerCategory::from_boundaries(vec![4], vec![vec![]]).unwrap();
        let label = category.canonical_label().unwrap();
        assert_eq!(label.category(), &category);
        assert_eq!(
            FiniteGroup::order(label.automorphism_group()).to_string(),
            "24"
        );
        assert!(label.verify(&category));
    }

    #[test]
    fn oriental_canonicalization_is_idempotent() {
        let category = SteinerCategory::oriental(2).unwrap();
        let label = category.canonical_label().unwrap();
        let canonical = label.category().clone();
        let identity = canonical
            .atom_counts()
            .iter()
            .map(|&count| (0..count).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert_eq!(label.encoding(), key_under(&canonical, &identity));
        assert_eq!(canonical.canonical_form().unwrap(), canonical);
        assert!(category.is_isomorphic(&canonical).unwrap());
    }

    #[test]
    fn canonicalization_and_automorphisms_match_exhaustive_small_searches() {
        check_exhaustive(&SteinerCategory::oriental(2).unwrap());
        check_exhaustive(&SteinerCategory::cube(2).unwrap());
    }
}
