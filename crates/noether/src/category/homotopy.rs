// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Iterated mapping categories, oriented basepoints, and homotopy Posets.

use crate::combin::Poset;

use super::model::{CategoryError, SignedBoundary, SteinerCategory, SteinerCell};

/// A finite iterated mapping omega-category inside a strong Steiner category.
///
/// Mapping categories need not carry a preferred strong-Steiner atomic basis,
/// so this value retains the ambient category and the boundary prefix instead
/// of pretending to own another based directed complex.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteOmegaCategory {
    ambient: SteinerCategory,
    prefix: Vec<(SteinerCell, SteinerCell)>,
}

impl FiniteOmegaCategory {
    pub(crate) fn new(
        ambient: SteinerCategory,
        prefix: Vec<(SteinerCell, SteinerCell)>,
    ) -> Result<Self, CategoryError> {
        validate_pairs(&ambient, &prefix)?;
        Ok(Self { ambient, prefix })
    }

    /// Ambient strong Steiner category.
    #[must_use]
    pub fn ambient(&self) -> &SteinerCategory {
        &self.ambient
    }

    /// Number of mapping iterations used to form this category.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.prefix.len()
    }

    /// Relative nonidentity dimension, or `-1` when the mapping category is empty.
    pub fn dimension(&self) -> Result<i64, CategoryError> {
        let counts = self.exact_cell_counts()?;
        Ok(counts
            .iter()
            .rposition(|&count| count != 0)
            .map_or(-1, |degree| degree as i64))
    }

    /// Objects of this mapping category as ambient cells.
    pub fn objects(&self) -> Result<Vec<SteinerCell>, CategoryError> {
        fillers(&self.ambient, &self.prefix, self.prefix.len())
    }

    /// Relative exact-dimensional nonidentity cells.
    pub fn exact_cells(
        &self,
        relative_dimension: usize,
    ) -> Result<Vec<SteinerCell>, CategoryError> {
        let mut cells = fillers(
            &self.ambient,
            &self.prefix,
            self.prefix
                .len()
                .checked_add(relative_dimension)
                .ok_or(CategoryError::SizeOverflow)?,
        )?;
        if relative_dimension > 0 {
            cells.retain(|cell| !cell.is_identity());
        }
        Ok(cells)
    }

    /// Relative exact-dimensional cell counts through this mapping category's dimension.
    pub fn exact_cell_counts(&self) -> Result<Vec<usize>, CategoryError> {
        let start = self.prefix.len();
        let maximum = self.ambient.dimension();
        if maximum < start as i64 {
            return Ok(Vec::new());
        }
        let mut counts = (start..=maximum as usize)
            .enumerate()
            .map(|(relative, dimension)| {
                fillers(&self.ambient, &self.prefix, dimension).map(|mut cells| {
                    if relative > 0 {
                        cells.retain(|cell| !cell.is_identity());
                    }
                    cells.len()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        while counts.last() == Some(&0) {
            counts.pop();
        }
        Ok(counts)
    }

    /// Recover a strong-Steiner presentation of this mapping category when one
    /// is supplied by its indecomposable ambient fillers.
    ///
    /// Mapping objects form the degree-zero basis. In positive relative degree,
    /// the basis consists of fillers with one top-dimensional ambient atom.
    /// Full ambient cell realizations and their boundaries are checked
    /// bijectively in every degree, retaining whiskering. Mapping categories of
    /// free strict higher categories need not themselves be free because
    /// interchange can identify composites; those return `None`.
    pub fn strong_steiner_model(&self) -> Result<Option<SteinerCategory>, CategoryError> {
        verified_mapping_model(self)
    }

    /// Mapping category between two objects of this mapping category.
    pub fn mapping_category(
        &self,
        source: &SteinerCell,
        target: &SteinerCell,
    ) -> Result<Self, CategoryError> {
        let objects = self.objects()?;
        if !objects.contains(source) || !objects.contains(target) {
            return Err(CategoryError::InvalidBasepoint(
                "mapping endpoints are not objects of this mapping category",
            ));
        }
        let mut prefix = self.prefix.clone();
        prefix.push((source.clone(), target.clone()));
        Self::new(self.ambient.clone(), prefix)
    }

    /// Poset of components, ordered by existence of a morphism.
    pub fn component_poset(&self) -> Result<Poset, CategoryError> {
        component_poset(&self.ambient, &self.prefix)
    }
}

/// A finite oriented basepoint represented by a globe-boundary functor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrientedBasepoint {
    pairs: Vec<(SteinerCell, SteinerCell)>,
}

impl OrientedBasepoint {
    /// Construct and validate an oriented basepoint.
    pub fn new(
        category: &SteinerCategory,
        pairs: Vec<(SteinerCell, SteinerCell)>,
    ) -> Result<Self, CategoryError> {
        if pairs.is_empty() {
            return Err(CategoryError::InvalidBasepoint(
                "an oriented basepoint has nonnegative dimension",
            ));
        }
        validate_pairs(category, &pairs)?;
        Ok(Self { pairs })
    }

    /// Basepoint dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.pairs.len() - 1
    }

    /// Homotopy-Poset degree associated to this basepoint.
    #[must_use]
    pub fn homotopy_degree(&self) -> usize {
        self.pairs.len()
    }

    /// Recursive pairs defining the globe-boundary map.
    #[must_use]
    pub fn pairs(&self) -> &[(SteinerCell, SteinerCell)] {
        &self.pairs
    }

    /// Whether the globe-boundary functor is monomorphic.
    ///
    /// In the finite gaunt strict setting, recursive cell injectivity is
    /// equivalent to requiring the two globe faces to remain distinct in every
    /// dimension.
    #[must_use]
    pub fn is_monomorphic(&self) -> bool {
        self.pairs.iter().all(|(source, target)| source != target)
    }

    /// Iterated mapping category selected by this basepoint.
    pub fn iterated_mapping_category(
        &self,
        category: &SteinerCategory,
    ) -> Result<FiniteOmegaCategory, CategoryError> {
        FiniteOmegaCategory::new(category.clone(), self.pairs.clone())
    }

    /// Gepner--Heine homotopy Poset at this basepoint.
    pub fn homotopy_poset(&self, category: &SteinerCategory) -> Result<Poset, CategoryError> {
        validate_pairs(category, &self.pairs)?;
        component_poset(category, &self.pairs)
    }

    pub(crate) fn enumerate_monomorphic(
        category: &SteinerCategory,
        dimension: usize,
    ) -> Result<Vec<Self>, CategoryError> {
        let mut output = Vec::new();
        enumerate_prefixes(category, dimension + 1, &mut Vec::new(), &mut output)?;
        Ok(output)
    }
}

fn enumerate_prefixes(
    category: &SteinerCategory,
    target_length: usize,
    prefix: &mut Vec<(SteinerCell, SteinerCell)>,
    output: &mut Vec<OrientedBasepoint>,
) -> Result<(), CategoryError> {
    let dimension = prefix.len();
    let objects = fillers(category, prefix, dimension)?;
    for source in &objects {
        for target in &objects {
            if source == target {
                continue;
            }
            prefix.push((source.clone(), target.clone()));
            if prefix.len() == target_length {
                output.push(OrientedBasepoint {
                    pairs: prefix.clone(),
                });
            } else {
                enumerate_prefixes(category, target_length, prefix, output)?;
            }
            prefix.pop();
        }
    }
    Ok(())
}

fn validate_pairs(
    category: &SteinerCategory,
    pairs: &[(SteinerCell, SteinerCell)],
) -> Result<(), CategoryError> {
    for (dimension, (source, target)) in pairs.iter().enumerate() {
        if source.category() != category || target.category() != category {
            return Err(CategoryError::CellContextMismatch);
        }
        if source.dimension() != dimension || target.dimension() != dimension {
            return Err(CategoryError::InvalidBasepoint(
                "basepoint pair has the wrong ambient dimension",
            ));
        }
        if dimension > 0 {
            let previous = &pairs[dimension - 1];
            if source.source()? != previous.0
                || source.target()? != previous.1
                || target.source()? != previous.0
                || target.target()? != previous.1
            {
                return Err(CategoryError::InvalidBasepoint(
                    "successive basepoint cells are not parallel over the preceding pair",
                ));
            }
        }
    }
    Ok(())
}

fn fillers(
    category: &SteinerCategory,
    prefix: &[(SteinerCell, SteinerCell)],
    dimension: usize,
) -> Result<Vec<SteinerCell>, CategoryError> {
    if prefix.len() > dimension {
        return Err(CategoryError::InvalidBasepoint(
            "mapping prefix is longer than the requested ambient cell dimension",
        ));
    }
    let mut cells = category.cells(dimension)?;
    cells.retain(|cell| {
        prefix.iter().enumerate().all(|(boundary_dimension, pair)| {
            let source = cell.source_at(boundary_dimension);
            let target = cell.target_at(boundary_dimension);
            source.is_ok_and(|source| source == pair.0)
                && target.is_ok_and(|target| target == pair.1)
        })
    });
    Ok(cells)
}

fn component_poset(
    category: &SteinerCategory,
    prefix: &[(SteinerCell, SteinerCell)],
) -> Result<Poset, CategoryError> {
    validate_pairs(category, prefix)?;
    if let [(source, target)] = prefix {
        return fundamental_poset(category, source, target);
    }
    component_poset_generic(category, prefix)
}

/// Poset of objects, ordered by the existence of a 1-cell.
pub(crate) fn object_component_poset(category: &SteinerCategory) -> Result<Poset, CategoryError> {
    let objects = category.exact_cells(0)?;
    let mut relations = Vec::new();
    for arrow in category.exact_cells(1)? {
        let source = arrow.source()?;
        let target = arrow.target()?;
        if source == target {
            continue;
        }
        let source_index = objects.iter().position(|object| object == &source).ok_or(
            CategoryError::Canonicalization("a 1-cell source was absent from the object basis"),
        )?;
        let target_index = objects.iter().position(|object| object == &target).ok_or(
            CategoryError::Canonicalization("a 1-cell target was absent from the object basis"),
        )?;
        relations.push((
            u32::try_from(source_index).map_err(|_| CategoryError::SizeOverflow)?,
            u32::try_from(target_index).map_err(|_| CategoryError::SizeOverflow)?,
        ));
    }
    Ok(Poset::new(objects.len(), &relations)?)
}

pub(crate) fn fundamental_poset(
    category: &SteinerCategory,
    source: &SteinerCell,
    target: &SteinerCell,
) -> Result<Poset, CategoryError> {
    if source.category() != category || target.category() != category {
        return Err(CategoryError::CellContextMismatch);
    }
    if source.dimension() != 0 || target.dimension() != 0 {
        return Err(CategoryError::InvalidBasepoint(
            "a fundamental Poset is based at two objects",
        ));
    }
    let objects = category
        .cells(1)?
        .into_iter()
        .filter(|cell| {
            cell.source()
                .is_ok_and(|cell_source| cell_source == *source)
                && cell
                    .target()
                    .is_ok_and(|cell_target| cell_target == *target)
        })
        .collect::<Vec<_>>();
    let mut relations = Vec::new();
    for arrow in category.cells(2)? {
        let arrow_source = arrow.source()?;
        let arrow_target = arrow.target()?;
        if arrow_source == arrow_target {
            continue;
        }
        let Some(source_index) = objects.iter().position(|object| object == &arrow_source) else {
            continue;
        };
        let Some(target_index) = objects.iter().position(|object| object == &arrow_target) else {
            continue;
        };
        relations.push((
            u32::try_from(source_index).map_err(|_| CategoryError::SizeOverflow)?,
            u32::try_from(target_index).map_err(|_| CategoryError::SizeOverflow)?,
        ));
    }
    Ok(Poset::new(objects.len(), &relations)?)
}

fn component_poset_generic(
    category: &SteinerCategory,
    prefix: &[(SteinerCell, SteinerCell)],
) -> Result<Poset, CategoryError> {
    let object_dimension = prefix.len();
    let objects = fillers(category, prefix, object_dimension)?;
    let arrows = fillers(category, prefix, object_dimension + 1)?;
    let mut relations = Vec::new();
    for arrow in arrows {
        let source = arrow.source()?;
        let target = arrow.target()?;
        if source == target {
            continue;
        }
        let Some(source_index) = objects.iter().position(|object| object == &source) else {
            return Err(CategoryError::InvalidBasepoint(
                "mapping arrow source was not an object of the iterated mapping category",
            ));
        };
        let Some(target_index) = objects.iter().position(|object| object == &target) else {
            return Err(CategoryError::InvalidBasepoint(
                "mapping arrow target was not an object of the iterated mapping category",
            ));
        };
        relations.push((
            u32::try_from(source_index).map_err(|_| CategoryError::SizeOverflow)?,
            u32::try_from(target_index).map_err(|_| CategoryError::SizeOverflow)?,
        ));
    }
    Ok(Poset::new(objects.len(), &relations)?)
}

/// Recover a model together with an exhaustive degreewise boundary check.
/// Unsupported nonfree mapping categories return `None`.
fn verified_mapping_model(
    mapping: &FiniteOmegaCategory,
) -> Result<Option<SteinerCategory>, CategoryError> {
    let counts = mapping.exact_cell_counts()?;
    if counts.is_empty() || counts[0] == 0 {
        return Ok(Some(SteinerCategory::empty()));
    }
    let mut generators = vec![mapping.exact_cells(0)?];
    for degree in 1..counts.len() {
        generators.push(
            mapping
                .exact_cells(degree)?
                .into_iter()
                .filter(|cell| cell.top_atoms().len() == 1)
                .collect(),
        );
    }
    let atom_counts: Vec<_> = generators.iter().map(Vec::len).collect();
    let mut boundaries = vec![Vec::new()];
    let mut model = SteinerCategory::from_boundaries(vec![atom_counts[0]], boundaries.clone())?;
    let mut inverse: Vec<(SteinerCell, SteinerCell)> = Vec::new();
    for degree in 0..counts.len() {
        if degree > 0 {
            let mut boundary = Vec::new();
            for generator in &generators[degree] {
                let source = generator.source()?;
                let target = generator.target()?;
                let Some((_, source_model)) =
                    inverse.iter().find(|(ambient, _)| ambient == &source)
                else {
                    return Ok(None);
                };
                let Some((_, target_model)) =
                    inverse.iter().find(|(ambient, _)| ambient == &target)
                else {
                    return Ok(None);
                };
                let source_support = &source_model.negative_supports()[degree - 1];
                let target_support = &target_model.negative_supports()[degree - 1];
                let negative: Vec<_> = source_support
                    .iter()
                    .copied()
                    .filter(|atom| !target_support.contains(atom))
                    .collect();
                let positive: Vec<_> = target_support
                    .iter()
                    .copied()
                    .filter(|atom| !source_support.contains(atom))
                    .collect();
                boundary.push(SignedBoundary::new(&negative, &positive)?);
            }
            boundaries.push(boundary);
            let Ok(next) = SteinerCategory::from_boundaries(
                atom_counts[..=degree].to_vec(),
                boundaries.clone(),
            ) else {
                return Ok(None);
            };
            model = next;
        }
        let expected = mapping.exact_cells(degree)?;
        let mut observed = Vec::new();
        let mut next_inverse = Vec::new();
        for cell in model.cells(degree)? {
            let Some(ambient) = realize(mapping, &generators, &cell)? else {
                return Ok(None);
            };
            if next_inverse
                .iter()
                .any(|(previous, _)| previous == &ambient)
            {
                return Ok(None);
            }
            if degree == 0 || !cell.is_identity() {
                if !expected.contains(&ambient) {
                    return Ok(None);
                }
                observed.push(ambient.clone());
            }
            if degree > 0 {
                for (model_boundary, actual_boundary) in [
                    (cell.source()?, ambient.source()?),
                    (cell.target()?, ambient.target()?),
                ] {
                    let Some((_, prior_model)) = inverse
                        .iter()
                        .find(|(actual, _)| actual == &actual_boundary)
                    else {
                        return Ok(None);
                    };
                    // The earlier skeleton has another category context; compare
                    // double sequences, which retain every whiskering component.
                    if prior_model.negative_supports() != model_boundary.negative_supports()
                        || prior_model.positive_supports() != model_boundary.positive_supports()
                    {
                        return Ok(None);
                    }
                }
            }
            next_inverse.push((ambient, cell));
        }
        if observed.len() != expected.len() {
            return Ok(None);
        }
        inverse = next_inverse;
    }
    Ok(Some(model))
}

fn realize(
    mapping: &FiniteOmegaCategory,
    generators: &[Vec<SteinerCell>],
    cell: &SteinerCell,
) -> Result<Option<SteinerCell>, CategoryError> {
    let depth = mapping.depth();
    let base = &generators[0][0];
    let mut negative = base.negative_supports()[..depth].to_vec();
    let mut positive = base.positive_supports()[..depth].to_vec();
    for degree in 0..=cell.dimension() {
        for (supports, result) in [
            (cell.negative_supports(), &mut negative),
            (cell.positive_supports(), &mut positive),
        ] {
            let mut chain: Vec<_> = supports[degree]
                .iter()
                .flat_map(|&index| {
                    generators[degree][index].negative_supports()[depth + degree]
                        .iter()
                        .copied()
                })
                .collect();
            chain.sort_unstable();
            if chain.windows(2).any(|pair| pair[0] == pair[1]) {
                return Ok(None);
            }
            result.push(chain);
        }
    }
    Ok(mapping
        .ambient()
        .cell_from_supports(negative, positive)
        .ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::AtomId;

    #[test]
    fn two_oriental_fundamental_poset_is_the_two_chain() {
        let category = SteinerCategory::oriental(2).unwrap();
        let source = category.atom(AtomId::new(0, 0)).unwrap();
        let target = category.atom(AtomId::new(0, 2)).unwrap();
        let basepoint = OrientedBasepoint::new(&category, vec![(source, target)]).unwrap();
        assert!(basepoint.is_monomorphic());
        let poset = basepoint.homotopy_poset(&category).unwrap();
        assert_eq!(poset.order(), 2);
        assert!(poset.leq(0, 1).unwrap() || poset.leq(1, 0).unwrap());
    }

    #[test]
    fn mapping_category_recovers_a_strong_steiner_model() {
        let category = SteinerCategory::oriental(2).unwrap();
        let source = category.atom(AtomId::new(0, 0)).unwrap();
        let target = category.atom(AtomId::new(0, 2)).unwrap();
        let mapping = category.mapping_category(&source, &target).unwrap();
        let model = mapping.strong_steiner_model().unwrap().unwrap();
        assert_eq!(model.atom_counts(), &[2, 1]);
        assert_eq!(model.exact_cell_counts().unwrap(), vec![2, 1]);
        assert_eq!(
            model.canonical_form().unwrap(),
            SteinerCategory::globe(1).unwrap()
        );

        let reverse = category.mapping_category(&target, &source).unwrap();
        assert_eq!(
            reverse.strong_steiner_model().unwrap().unwrap(),
            SteinerCategory::empty()
        );
    }

    #[test]
    fn parallel_higher_cells_recover_the_mapping_model() {
        let edge = SignedBoundary::new(&[0], &[1]).unwrap();
        let higher = SignedBoundary::new(&[0], &[1]).unwrap();
        let category = SteinerCategory::from_boundaries(
            vec![2, 2, 6],
            vec![vec![], vec![edge.clone(), edge], vec![higher; 6]],
        )
        .unwrap();
        let objects = category.exact_cells(0).unwrap();
        let mapping = category.mapping_category(&objects[0], &objects[1]).unwrap();
        assert_eq!(mapping.exact_cell_counts().unwrap(), vec![2, 6]);
        let model = mapping.strong_steiner_model().unwrap().unwrap();
        assert_eq!(model.atom_counts(), &[2, 6]);
        assert_eq!(model.exact_cell_counts().unwrap(), vec![2, 6]);
    }

    #[test]
    fn higher_mapping_model_cancels_shared_boundary_atoms() {
        let edge = SignedBoundary::new(&[0], &[1]).unwrap();
        let category = SteinerCategory::from_boundaries(
            vec![2, 4, 4, 1],
            vec![
                vec![],
                vec![edge.clone(); 4],
                vec![
                    SignedBoundary::new(&[0], &[1]).unwrap(),
                    SignedBoundary::new(&[0], &[1]).unwrap(),
                    SignedBoundary::new(&[2], &[0]).unwrap(),
                    SignedBoundary::new(&[1], &[3]).unwrap(),
                ],
                vec![SignedBoundary::new(&[0], &[1]).unwrap()],
            ],
        )
        .unwrap();
        let objects = category.exact_cells(0).unwrap();
        let mapping = category.mapping_category(&objects[0], &objects[1]).unwrap();
        let _ = mapping.strong_steiner_model().unwrap();
    }

    #[test]
    fn fundamental_fast_path_matches_generic_mapping_components() {
        for category in [
            SteinerCategory::oriental(2).unwrap(),
            SteinerCategory::cube(2).unwrap(),
        ] {
            let objects = category.exact_cells(0).unwrap();
            for source in &objects {
                for target in &objects {
                    let prefix = vec![(source.clone(), target.clone())];
                    let fast = fundamental_poset(&category, source, target).unwrap();
                    let generic = component_poset_generic(&category, &prefix).unwrap();
                    assert_eq!(
                        fast.canonical_form().unwrap(),
                        generic.canonical_form().unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn monomorphic_basepoints_form_a_finite_recursive_tree() {
        let category = SteinerCategory::oriental(2).unwrap();
        let zero = category.monomorphic_oriented_basepoints(0).unwrap();
        assert_eq!(zero.len(), 6);
        let one = category.monomorphic_oriented_basepoints(1).unwrap();
        assert_eq!(one.len(), 2);
        assert!(one.iter().all(OrientedBasepoint::is_monomorphic));
    }
}

#[cfg(test)]
mod mapping_model_regressions {
    use super::*;
    use crate::category::AtomId;

    #[test]
    fn whiskered_globes_have_distinct_boundaries() {
        let xy = SignedBoundary::new(&[0], &[1]).unwrap();
        let yz = SignedBoundary::new(&[1], &[2]).unwrap();
        let parallel = SignedBoundary::new(&[0], &[1]).unwrap();
        let category = SteinerCategory::from_boundaries(
            vec![3, 4, 2, 1],
            vec![
                vec![],
                vec![xy.clone(), xy, yz.clone(), yz],
                vec![parallel.clone(); 2],
                vec![parallel],
            ],
        )
        .unwrap();
        let mapping = category
            .mapping_category(
                &category.atom(AtomId::new(0, 0)).unwrap(),
                &category.atom(AtomId::new(0, 2)).unwrap(),
            )
            .unwrap();
        let model = mapping
            .strong_steiner_model()
            .unwrap()
            .expect("two disjoint globes are free");
        let globe = SteinerCategory::globe(2).unwrap();
        assert!(model
            .is_isomorphic(&globe.disjoint_union(&globe).unwrap())
            .unwrap());
        assert_ne!(
            model.boundary(AtomId::new(2, 0)).unwrap(),
            model.boundary(AtomId::new(2, 1)).unwrap()
        );
    }

    #[test]
    fn whiskered_composite_boundaries_recover_two_orientals() {
        let xy = SignedBoundary::new(&[0], &[1]).unwrap();
        let yz = SignedBoundary::new(&[1], &[2]).unwrap();
        let category = SteinerCategory::from_boundaries(
            vec![3, 5, 3, 1],
            vec![
                vec![],
                vec![xy.clone(), xy.clone(), xy, yz.clone(), yz],
                vec![
                    SignedBoundary::new(&[0], &[1]).unwrap(),
                    SignedBoundary::new(&[1], &[2]).unwrap(),
                    SignedBoundary::new(&[0], &[2]).unwrap(),
                ],
                vec![SignedBoundary::new(&[2], &[0, 1]).unwrap()],
            ],
        )
        .unwrap();
        let mapping = category
            .mapping_category(
                &category.atom(AtomId::new(0, 0)).unwrap(),
                &category.atom(AtomId::new(0, 2)).unwrap(),
            )
            .unwrap();
        let model = mapping
            .strong_steiner_model()
            .unwrap()
            .expect("two disjoint orientals are free");
        let oriental = SteinerCategory::oriental(2).unwrap();
        assert!(model
            .is_isomorphic(&oriental.disjoint_union(&oriental).unwrap())
            .unwrap());
    }

    #[test]
    fn empty_identity_and_single_globe_models() {
        for dimension in 0..=4 {
            let category = SteinerCategory::globe(dimension).unwrap();
            for source in category.cells(0).unwrap() {
                for target in category.cells(0).unwrap() {
                    let mapping = category.mapping_category(&source, &target).unwrap();
                    let model = mapping.strong_steiner_model().unwrap().unwrap();
                    assert_eq!(
                        model.exact_cell_counts().unwrap(),
                        mapping.exact_cell_counts().unwrap()
                    );
                }
            }
        }
    }
}
