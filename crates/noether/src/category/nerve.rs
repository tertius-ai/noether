// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Street nerve and the exact combinatorial presentation of categorical homology.

use std::collections::BTreeMap;

use crate::combin::{Poset, PosetError, MAX_POSET_ORDER};

use super::{
    homotopy::OrientedBasepoint,
    model::{AtomId, CategoryError, SteinerCategory, SteinerCell},
};

/// Largest number of Street simplices materialized in one simplicial degree.
pub const MAX_STREET_NERVE_SIMPLICES_PER_DEGREE: usize = 1_000_000;

/// Largest number of permutation sectors inspected in one categorical-homology
/// fundamental Poset.
pub const MAX_CATEGORICAL_HOMOLOGY_PERMUTATIONS: usize = 1_000_000;

/// One simplex of the Street nerve, represented as a positive chain map.
///
/// `images[q][i]` is the top atomic support of the image of degree-`q` atom
/// `i` of the corresponding oriental. An empty support represents an identity
/// image in positive degree.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StreetSimplex {
    dimension: usize,
    images: Vec<Vec<Vec<usize>>>,
}

impl StreetSimplex {
    /// Simplicial dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Degreewise images of oriental atoms.
    #[must_use]
    pub fn images(&self) -> &[Vec<Vec<usize>>] {
        &self.images
    }

    /// Whether the simplex is thin in the stratified Street nerve.
    #[must_use]
    pub fn is_thin(&self) -> bool {
        self.dimension > 0
            && self
                .images
                .last()
                .and_then(|images| images.first())
                .is_some_and(Vec::is_empty)
    }
}

/// A certified finite skeleton of the stratified Street nerve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreetNerveSkeleton {
    source: SteinerCategory,
    degrees: Vec<Vec<StreetSimplex>>,
    faces: Vec<Vec<Vec<usize>>>,
    degeneracies: Vec<Vec<Vec<usize>>>,
}

impl StreetNerveSkeleton {
    pub(crate) fn compute(
        source: &SteinerCategory,
        max_simplicial_degree: usize,
    ) -> Result<Self, CategoryError> {
        if max_simplicial_degree > 8 {
            return Err(CategoryError::StreetNerve(
                "Street-nerve materialization accepts simplicial degree at most 8",
            ));
        }
        let mut degrees = Vec::with_capacity(max_simplicial_degree + 1);
        for dimension in 0..=max_simplicial_degree {
            degrees.push(enumerate_simplices(source, dimension)?);
        }
        let indices = degrees
            .iter()
            .map(|simplices| {
                simplices
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(|(index, simplex)| (simplex, index))
                    .collect::<BTreeMap<_, _>>()
            })
            .collect::<Vec<_>>();

        let mut faces = vec![Vec::new(); max_simplicial_degree + 1];
        for dimension in 1..=max_simplicial_degree {
            for simplex in &degrees[dimension] {
                let simplex_faces = (0..=dimension)
                    .map(|omitted| {
                        let face = restrict_face(simplex, omitted);
                        indices[dimension - 1].get(&face).copied().ok_or(
                            CategoryError::StreetNerve(
                                "a Street face was absent from the preceding degree",
                            ),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                faces[dimension].push(simplex_faces);
            }
        }

        let mut degeneracies = vec![Vec::new(); max_simplicial_degree + 1];
        for dimension in 0..max_simplicial_degree {
            for simplex in &degrees[dimension] {
                let simplex_degeneracies = (0..=dimension)
                    .map(|repeated| {
                        let degeneracy = extend_degeneracy(simplex, repeated);
                        indices[dimension + 1].get(&degeneracy).copied().ok_or(
                            CategoryError::StreetNerve(
                                "a Street degeneracy was absent from the next degree",
                            ),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                degeneracies[dimension].push(simplex_degeneracies);
            }
        }

        Ok(Self {
            source: source.clone(),
            degrees,
            faces,
            degeneracies,
        })
    }

    /// Source category.
    #[must_use]
    pub fn source(&self) -> &SteinerCategory {
        &self.source
    }

    /// Largest materialized simplicial degree.
    #[must_use]
    pub fn max_simplicial_degree(&self) -> usize {
        self.degrees.len().saturating_sub(1)
    }

    /// Street simplices in one materialized degree.
    #[must_use]
    pub fn simplices(&self, degree: usize) -> Option<&[StreetSimplex]> {
        self.degrees.get(degree).map(Vec::as_slice)
    }

    /// Numbers of simplices by degree.
    #[must_use]
    pub fn simplex_counts(&self) -> Vec<usize> {
        self.degrees.iter().map(Vec::len).collect()
    }

    /// Numbers of thin simplices by degree.
    #[must_use]
    pub fn thin_simplex_counts(&self) -> Vec<usize> {
        self.degrees
            .iter()
            .map(|simplices| simplices.iter().filter(|simplex| simplex.is_thin()).count())
            .collect()
    }

    /// Face indices of one simplex, ordered by the omitted vertex.
    #[must_use]
    pub fn faces(&self, degree: usize, simplex: usize) -> Option<&[usize]> {
        self.faces
            .get(degree)
            .and_then(|degree_faces| degree_faces.get(simplex))
            .map(Vec::as_slice)
    }

    /// Degeneracy indices of one simplex, ordered by the repeated vertex.
    #[must_use]
    pub fn degeneracies(&self, degree: usize, simplex: usize) -> Option<&[usize]> {
        self.degeneracies
            .get(degree)
            .and_then(|degree_degeneracies| degree_degeneracies.get(simplex))
            .map(Vec::as_slice)
    }

    /// Verify the elementary simplicial identities within the stored range.
    #[must_use]
    pub fn verify(&self) -> bool {
        // d_i d_j = d_{j-1} d_i for i < j.
        for degree in 2..self.degrees.len() {
            for simplex in 0..self.degrees[degree].len() {
                for left in 0..degree {
                    for right in left + 1..=degree {
                        let Some(first) = self.faces(degree, simplex) else {
                            return false;
                        };
                        let a = first[right];
                        let b = first[left];
                        let Some(a_faces) = self.faces(degree - 1, a) else {
                            return false;
                        };
                        let Some(b_faces) = self.faces(degree - 1, b) else {
                            return false;
                        };
                        if a_faces[left] != b_faces[right - 1] {
                            return false;
                        }
                    }
                }
            }
        }

        // d_i s_j, including the two identity cases.  The nonidentity cases
        // only occur once the preceding degree is available.
        for degree in 0..self.degrees.len().saturating_sub(1) {
            for simplex in 0..self.degrees[degree].len() {
                let Some(simplex_degeneracies) = self.degeneracies(degree, simplex) else {
                    return false;
                };
                for repeated in 0..=degree {
                    let degenerate = simplex_degeneracies[repeated];
                    let Some(degenerate_faces) = self.faces(degree + 1, degenerate) else {
                        return false;
                    };
                    for face in 0..=degree + 1 {
                        let actual = degenerate_faces[face];
                        let expected = if face == repeated || face == repeated + 1 {
                            simplex
                        } else if face < repeated {
                            if degree == 0 {
                                return false;
                            }
                            let Some(simplex_faces) = self.faces(degree, simplex) else {
                                return false;
                            };
                            let lower = simplex_faces[face];
                            let Some(lower_degeneracies) = self.degeneracies(degree - 1, lower)
                            else {
                                return false;
                            };
                            lower_degeneracies[repeated - 1]
                        } else {
                            if degree == 0 {
                                return false;
                            }
                            let Some(simplex_faces) = self.faces(degree, simplex) else {
                                return false;
                            };
                            let lower = simplex_faces[face - 1];
                            let Some(lower_degeneracies) = self.degeneracies(degree - 1, lower)
                            else {
                                return false;
                            };
                            lower_degeneracies[repeated]
                        };
                        if actual != expected {
                            return false;
                        }
                    }
                }
            }
        }

        // s_i s_j = s_{j+1} s_i for i <= j.
        for degree in 0..self.degrees.len().saturating_sub(2) {
            for simplex in 0..self.degrees[degree].len() {
                let Some(first) = self.degeneracies(degree, simplex) else {
                    return false;
                };
                for left in 0..=degree {
                    for right in left..=degree {
                        let after_right = first[right];
                        let after_left = first[left];
                        let Some(right_degeneracies) = self.degeneracies(degree + 1, after_right)
                        else {
                            return false;
                        };
                        let Some(left_degeneracies) = self.degeneracies(degree + 1, after_left)
                        else {
                            return false;
                        };
                        if right_degeneracies[left] != left_degeneracies[right + 1] {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

/// Exact lazy combinatorial presentation of Heine's categorical N-homology.
///
/// The represented value is the free stratified simplicial commutative monoid
/// on the Street nerve. A finite skeleton stores its monoid generators and
/// simplicial operators without attempting to enumerate the infinite free
/// commutative monoids themselves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CategoricalHomologyPresentation {
    source: SteinerCategory,
    basepoint: Option<SteinerCell>,
}

/// One finite particle-number slice of the zeroth categorical-homology Poset.
///
/// The full Poset of components is usually infinite.  Its non-reduced
/// categorical-homology model is canonically graded by particle number, and
/// each slice is finite.  Element `i` of `poset` is the multiset at index `i`.
#[derive(Clone, Debug)]
pub struct CategoricalHomologyComponentSlice {
    particle_number: usize,
    object_multisets: Vec<Vec<usize>>,
    poset: Poset,
}

impl CategoricalHomologyComponentSlice {
    /// Number of particles in every element of the slice.
    #[must_use]
    pub fn particle_number(&self) -> usize {
        self.particle_number
    }

    /// Object multisets, represented by nondecreasing object indices.
    #[must_use]
    pub fn object_multisets(&self) -> &[Vec<usize>] {
        &self.object_multisets
    }

    /// Finite Poset carried by this particle-number slice.
    #[must_use]
    pub fn poset(&self) -> &Poset {
        &self.poset
    }
}

impl CategoricalHomologyPresentation {
    pub(crate) fn new(source: SteinerCategory, basepoint: Option<SteinerCell>) -> Self {
        Self { source, basepoint }
    }

    /// Stable coefficient monoid name.
    #[must_use]
    pub const fn coefficient_monoid(&self) -> &'static str {
        "N"
    }

    /// Stable combinatorial model name.
    #[must_use]
    pub const fn model(&self) -> &'static str {
        "free-stratified-simplicial-commutative-monoid-on-Street-nerve"
    }

    /// Stable categorical spectrum represented by this presentation.
    #[must_use]
    pub const fn spectrum_model(&self) -> &'static str {
        "H(N)-Gray-smash-categorical-suspension-spectrum"
    }

    /// Infinite-loop category in which the homotopy Posets are computed.
    #[must_use]
    pub const fn infinite_loop_model(&self) -> &'static str {
        "free-strict-symmetric-monoidal-infinity-category"
    }

    /// Source category.
    #[must_use]
    pub fn source(&self) -> &SteinerCategory {
        &self.source
    }

    /// Distinguished object for reduced categorical homology, when present.
    #[must_use]
    pub fn basepoint(&self) -> Option<&SteinerCell> {
        self.basepoint.as_ref()
    }

    /// Whether this is the reduced pointed presentation.
    #[must_use]
    pub fn is_reduced(&self) -> bool {
        self.basepoint.is_some()
    }

    /// Materialize the exact finite generator presentation through one degree.
    pub fn skeleton(&self, max_degree: usize) -> Result<StreetNerveSkeleton, CategoryError> {
        self.source.street_nerve_skeleton(max_degree)
    }

    /// Finite particle-number slice of the zeroth homotopy Poset.
    ///
    /// For non-reduced categorical homology this is the `particle_number`-th
    /// symmetric power of `pi_0(source)`.  The full zeroth Poset is the
    /// coproduct of these slices over all particle numbers.  Reduced
    /// categorical homology is a filtered colimit in which particle number is
    /// not generally preserved, so it deliberately has no such slice API.
    pub fn component_poset_slice(
        &self,
        particle_number: usize,
    ) -> Result<CategoricalHomologyComponentSlice, CategoryError> {
        self.require_non_reduced_particle_model()?;
        let source_poset = self.source.component_poset()?;
        let object_multisets = multisets(source_poset.order(), particle_number)?;
        let indices = object_multisets
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, multiset)| (multiset, index))
            .collect::<BTreeMap<_, _>>();
        let mut relations = Vec::new();
        for (lower_index, lower) in object_multisets.iter().enumerate() {
            for position in 0..lower.len() {
                if position > 0 && lower[position - 1] == lower[position] {
                    continue;
                }
                let object =
                    u32::try_from(lower[position]).map_err(|_| CategoryError::SizeOverflow)?;
                for edge in source_poset
                    .covers()
                    .edges()
                    .iter()
                    .filter(|edge| edge.source == object)
                {
                    let mut upper = lower.clone();
                    upper[position] =
                        usize::try_from(edge.target).map_err(|_| CategoryError::SizeOverflow)?;
                    upper.sort_unstable();
                    let upper_index =
                        indices
                            .get(&upper)
                            .copied()
                            .ok_or(CategoryError::Canonicalization(
                                "a categorical-homology multiset successor was absent",
                            ))?;
                    relations.push((
                        u32::try_from(lower_index).map_err(|_| CategoryError::SizeOverflow)?,
                        u32::try_from(upper_index).map_err(|_| CategoryError::SizeOverflow)?,
                    ));
                }
            }
        }
        let poset = Poset::new(object_multisets.len(), &relations)?;
        Ok(CategoricalHomologyComponentSlice {
            particle_number,
            object_multisets,
            poset,
        })
    }

    /// Fundamental Poset in a fixed finite-particle sector.
    ///
    /// A morphism between two words in the free symmetric-monoidal category
    /// first chooses a permutation matching their particles.  Its component
    /// Poset is the cartesian product of the corresponding fundamental Posets
    /// in the source.  This method returns the disjoint union over every
    /// permutation sector.  Words of different lengths have the empty Poset.
    pub fn fundamental_poset(
        &self,
        source_word: &[SteinerCell],
        target_word: &[SteinerCell],
    ) -> Result<Poset, CategoryError> {
        self.require_non_reduced_particle_model()?;
        self.validate_object_word(source_word)?;
        self.validate_object_word(target_word)?;
        if source_word.len() != target_word.len() {
            return Ok(Poset::new(0, &[])?);
        }
        let length = source_word.len();
        if checked_factorial(length)? > MAX_CATEGORICAL_HOMOLOGY_PERMUTATIONS {
            return Err(CategoryError::ConstructionTooLarge {
                construction: "categorical-homology permutation sectors",
                parameter: length,
            });
        }

        let mut permutation = (0..length).collect::<Vec<_>>();
        let mut components = Vec::new();
        loop {
            let factors = source_word
                .iter()
                .enumerate()
                .map(|(index, source)| {
                    self.source
                        .fundamental_poset(source, &target_word[permutation[index]])
                })
                .collect::<Result<Vec<_>, _>>()?;
            let product = cartesian_product(&factors)?;
            if product.order() != 0 {
                components.push(product);
            }
            if !next_permutation(&mut permutation) {
                break;
            }
        }
        disjoint_union(&components)
    }

    /// Higher homotopy Poset in one fixed permutation/particle sector.
    ///
    /// The sector is specified by one source-category oriented basepoint per
    /// particle.  All factors must have the same homotopy degree.  The answer
    /// is their cartesian product; varying the first-level permutation gives
    /// the other disjoint sectors of the free symmetric-monoidal category.
    pub fn particle_homotopy_poset(
        &self,
        factors: &[OrientedBasepoint],
    ) -> Result<Poset, CategoryError> {
        self.require_non_reduced_particle_model()?;
        if let Some(first) = factors.first() {
            if factors
                .iter()
                .any(|factor| factor.homotopy_degree() != first.homotopy_degree())
            {
                return Err(CategoryError::InvalidBasepoint(
                    "categorical-homology particle basepoints have different dimensions",
                ));
            }
        }
        let posets = factors
            .iter()
            .map(|factor| factor.homotopy_poset(&self.source))
            .collect::<Result<Vec<_>, _>>()?;
        cartesian_product(&posets)
    }

    fn require_non_reduced_particle_model(&self) -> Result<(), CategoryError> {
        if self.is_reduced() {
            return Err(CategoryError::InvalidBasepoint(
                "particle-number slices apply to non-reduced categorical homology of C_+",
            ));
        }
        Ok(())
    }

    fn validate_object_word(&self, word: &[SteinerCell]) -> Result<(), CategoryError> {
        if word.iter().any(|object| object.category() != &self.source) {
            return Err(CategoryError::CellContextMismatch);
        }
        if word.iter().any(|object| object.dimension() != 0) {
            return Err(CategoryError::InvalidBasepoint(
                "a categorical-homology object word contains a positive-dimensional cell",
            ));
        }
        Ok(())
    }
}

fn multisets(
    object_count: usize,
    particle_number: usize,
) -> Result<Vec<Vec<usize>>, CategoryError> {
    fn visit(
        object_count: usize,
        particle_number: usize,
        minimum: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) -> Result<(), CategoryError> {
        if current.len() == particle_number {
            if output.len() == MAX_POSET_ORDER {
                return Err(CategoryError::Poset(PosetError::OrderTooLarge {
                    order: MAX_POSET_ORDER.saturating_add(1),
                    max: MAX_POSET_ORDER,
                }));
            }
            output.push(current.clone());
            return Ok(());
        }
        for object in minimum..object_count {
            current.push(object);
            visit(object_count, particle_number, object, current, output)?;
            current.pop();
        }
        Ok(())
    }

    if object_count == 0 && particle_number != 0 {
        return Ok(Vec::new());
    }
    let mut output = Vec::new();
    visit(
        object_count,
        particle_number,
        0,
        &mut Vec::new(),
        &mut output,
    )?;
    Ok(output)
}

fn cartesian_product(factors: &[Poset]) -> Result<Poset, CategoryError> {
    if factors.iter().any(|factor| factor.order() == 0) {
        return Ok(Poset::new(0, &[])?);
    }
    let mut order = 1usize;
    for factor in factors {
        order = order
            .checked_mul(factor.order())
            .ok_or(CategoryError::SizeOverflow)?;
        if order > MAX_POSET_ORDER {
            return Err(CategoryError::Poset(PosetError::OrderTooLarge {
                order,
                max: MAX_POSET_ORDER,
            }));
        }
    }
    let mut strides = vec![1usize; factors.len()];
    let mut stride = 1usize;
    for index in (0..factors.len()).rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(factors[index].order())
            .ok_or(CategoryError::SizeOverflow)?;
    }
    let mut relations = Vec::new();
    for vertex in 0..order {
        for (factor_index, factor) in factors.iter().enumerate() {
            let factor_vertex = (vertex / strides[factor_index]) % factor.order();
            for edge in factor
                .covers()
                .edges()
                .iter()
                .filter(|edge| edge.source as usize == factor_vertex)
            {
                let target = vertex - factor_vertex * strides[factor_index]
                    + edge.target as usize * strides[factor_index];
                relations.push((
                    u32::try_from(vertex).map_err(|_| CategoryError::SizeOverflow)?,
                    u32::try_from(target).map_err(|_| CategoryError::SizeOverflow)?,
                ));
            }
        }
    }
    Ok(Poset::new(order, &relations)?)
}

fn disjoint_union(components: &[Poset]) -> Result<Poset, CategoryError> {
    let mut order = 0usize;
    let mut relations = Vec::new();
    for component in components {
        let offset = order;
        order = order
            .checked_add(component.order())
            .ok_or(CategoryError::SizeOverflow)?;
        if order > MAX_POSET_ORDER {
            return Err(CategoryError::Poset(PosetError::OrderTooLarge {
                order,
                max: MAX_POSET_ORDER,
            }));
        }
        for edge in component.covers().edges() {
            relations.push((
                u32::try_from(offset + edge.source as usize)
                    .map_err(|_| CategoryError::SizeOverflow)?,
                u32::try_from(offset + edge.target as usize)
                    .map_err(|_| CategoryError::SizeOverflow)?,
            ));
        }
    }
    Ok(Poset::new(order, &relations)?)
}

fn checked_factorial(value: usize) -> Result<usize, CategoryError> {
    (1..=value).try_fold(1usize, |product, factor| {
        product
            .checked_mul(factor)
            .ok_or(CategoryError::SizeOverflow)
    })
}

fn next_permutation(values: &mut [usize]) -> bool {
    let Some(pivot) = values.windows(2).rposition(|pair| pair[0] < pair[1]) else {
        return false;
    };
    let Some(successor) = (pivot + 1..values.len()).rfind(|&index| values[pivot] < values[index])
    else {
        return false;
    };
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

fn enumerate_simplices(
    target: &SteinerCategory,
    dimension: usize,
) -> Result<Vec<StreetSimplex>, CategoryError> {
    let domain = SteinerCategory::oriental(dimension)?;
    let mut images = domain
        .atom_counts()
        .iter()
        .map(|&count| vec![Vec::new(); count])
        .collect::<Vec<_>>();
    let mut output = Vec::new();
    enumerate_atom_images(target, &domain, 0, 0, &mut images, &mut output, dimension)?;
    output.sort();
    output.dedup();
    Ok(output)
}

fn enumerate_atom_images(
    target: &SteinerCategory,
    domain: &SteinerCategory,
    degree: usize,
    atom: usize,
    images: &mut [Vec<Vec<usize>>],
    output: &mut Vec<StreetSimplex>,
    simplex_dimension: usize,
) -> Result<(), CategoryError> {
    if output.len() >= MAX_STREET_NERVE_SIMPLICES_PER_DEGREE {
        return Err(CategoryError::StreetNerve(
            "Street-nerve simplex count exceeded 1,000,000 in one degree",
        ));
    }
    if degree == domain.atom_counts().len() {
        output.push(StreetSimplex {
            dimension: simplex_dimension,
            images: images.to_vec(),
        });
        return Ok(());
    }
    if atom == domain.atom_count_in_degree(degree) {
        return enumerate_atom_images(
            target,
            domain,
            degree + 1,
            0,
            images,
            output,
            simplex_dimension,
        );
    }

    let candidates = if degree == 0 {
        target
            .exact_cells(0)?
            .into_iter()
            .map(|cell| cell.top_atoms().to_vec())
            .collect::<Vec<_>>()
    } else {
        let mut candidates = vec![Vec::new()];
        candidates.extend(
            target
                .exact_cells(degree)?
                .into_iter()
                .map(|cell| cell.top_atoms().to_vec()),
        );
        candidates
    };

    for candidate in candidates {
        if degree > 0
            && !chain_map_boundary_matches(target, domain, degree, atom, &candidate, images)?
        {
            continue;
        }
        images[degree][atom] = candidate;
        enumerate_atom_images(
            target,
            domain,
            degree,
            atom + 1,
            images,
            output,
            simplex_dimension,
        )?;
    }
    Ok(())
}

fn chain_map_boundary_matches(
    target: &SteinerCategory,
    domain: &SteinerCategory,
    degree: usize,
    atom: usize,
    candidate: &[usize],
    images: &[Vec<Vec<usize>>],
) -> Result<bool, CategoryError> {
    let mut left = vec![0i64; target.atom_count_in_degree(degree - 1)];
    if !candidate.is_empty() {
        let (negative, positive) = target.boundary_of_top(degree, candidate)?;
        for atom in negative {
            left[atom] -= 1;
        }
        for atom in positive {
            left[atom] += 1;
        }
    }
    let mut right = vec![0i64; target.atom_count_in_degree(degree - 1)];
    let boundary = domain.boundary(AtomId::new(degree, atom))?;
    for &domain_atom in boundary.negative() {
        for &target_atom in &images[degree - 1][domain_atom] {
            right[target_atom] -= 1;
        }
    }
    for &domain_atom in boundary.positive() {
        for &target_atom in &images[degree - 1][domain_atom] {
            right[target_atom] += 1;
        }
    }
    Ok(left == right)
}

fn restrict_face(simplex: &StreetSimplex, omitted: usize) -> StreetSimplex {
    let smaller = simplex.dimension - 1;
    let mut images = Vec::with_capacity(smaller + 1);
    for degree in 0..=smaller {
        let small_faces = combinations(smaller + 1, degree + 1);
        let large_faces = combinations(simplex.dimension + 1, degree + 1)
            .into_iter()
            .enumerate()
            .map(|(index, face)| (face, index))
            .collect::<BTreeMap<_, _>>();
        images.push(
            small_faces
                .into_iter()
                .map(|face| {
                    let embedded = face
                        .into_iter()
                        .map(|vertex| if vertex < omitted { vertex } else { vertex + 1 })
                        .collect::<Vec<_>>();
                    simplex.images[degree][large_faces[&embedded]].clone()
                })
                .collect(),
        );
    }
    StreetSimplex {
        dimension: smaller,
        images,
    }
}

fn extend_degeneracy(simplex: &StreetSimplex, repeated: usize) -> StreetSimplex {
    let larger = simplex.dimension + 1;
    let mut images = Vec::with_capacity(larger + 1);
    for degree in 0..=larger {
        let small_faces = combinations(simplex.dimension + 1, degree + 1)
            .into_iter()
            .enumerate()
            .map(|(index, face)| (face, index))
            .collect::<BTreeMap<_, _>>();
        let large_faces = combinations(larger + 1, degree + 1);
        images.push(
            large_faces
                .into_iter()
                .map(|face| {
                    let mapped = face
                        .into_iter()
                        .map(|vertex| {
                            if vertex <= repeated {
                                vertex
                            } else {
                                vertex - 1
                            }
                        })
                        .collect::<Vec<_>>();
                    if mapped.windows(2).any(|pair| pair[0] == pair[1]) {
                        Vec::new()
                    } else {
                        simplex.images[degree][small_faces[&mapped]].clone()
                    }
                })
                .collect(),
        );
    }
    StreetSimplex {
        dimension: larger,
        images,
    }
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn visit(
        n: usize,
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            output.push(current.clone());
            return;
        }
        let remaining = k - current.len();
        for value in start..=n - remaining {
            current.push(value);
            visit(n, k, value + 1, current, output);
            current.pop();
        }
    }
    if k > n {
        return Vec::new();
    }
    let mut output = Vec::new();
    visit(n, k, 0, &mut Vec::new(), &mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_nerve_has_one_simplex_in_every_degree() {
        let skeleton = SteinerCategory::point().street_nerve_skeleton(4).unwrap();
        assert_eq!(skeleton.simplex_counts(), vec![1, 1, 1, 1, 1]);
        assert_eq!(skeleton.thin_simplex_counts(), vec![0, 1, 1, 1, 1]);
        assert!(skeleton.verify());
    }

    #[test]
    fn arrow_nerve_begins_with_two_vertices_and_three_edges() {
        let skeleton = SteinerCategory::globe(1)
            .unwrap()
            .street_nerve_skeleton(2)
            .unwrap();
        assert_eq!(skeleton.simplex_counts()[0], 2);
        assert_eq!(skeleton.simplex_counts()[1], 3);
        assert!(skeleton.verify());
    }

    #[test]
    fn categorical_homology_keeps_its_categorical_value_shape() {
        let category = SteinerCategory::globe(2).unwrap();
        let presentation = category.categorical_homology();
        assert_eq!(presentation.coefficient_monoid(), "N");
        assert!(!presentation.is_reduced());
        assert!(presentation.skeleton(2).unwrap().verify());
    }

    #[test]
    fn component_slices_are_exact_symmetric_powers() {
        let category = SteinerCategory::globe(1).unwrap();
        let presentation = category.categorical_homology();
        let zeroth = presentation.component_poset_slice(0).unwrap();
        assert_eq!(zeroth.object_multisets(), &[Vec::<usize>::new()]);
        assert_eq!(zeroth.poset().order(), 1);

        let second = presentation.component_poset_slice(2).unwrap();
        assert_eq!(
            second.object_multisets(),
            &[vec![0, 0], vec![0, 1], vec![1, 1]]
        );
        assert_eq!(second.poset().order(), 3);
        assert_eq!(second.poset().height(), 3);
    }

    #[test]
    fn point_fundamental_particle_sector_is_a_symmetric_group() {
        let category = SteinerCategory::point();
        let object = category.atom(AtomId::new(0, 0)).unwrap();
        let word = vec![object; 3];
        let poset = category
            .categorical_homology()
            .fundamental_poset(&word, &word)
            .unwrap();
        assert_eq!(poset.order(), 6);
        assert_eq!(poset.width(), 6);
        assert_eq!(poset.height(), 1);
    }

    #[test]
    fn higher_particle_sectors_are_cartesian_products() {
        let category = SteinerCategory::oriental(2).unwrap();
        let source = category.atom(AtomId::new(0, 0)).unwrap();
        let target = category.atom(AtomId::new(0, 2)).unwrap();
        let basepoint = OrientedBasepoint::new(&category, vec![(source, target)]).unwrap();
        let poset = category
            .categorical_homology()
            .particle_homotopy_poset(&[basepoint.clone(), basepoint])
            .unwrap();
        assert_eq!(poset.order(), 4);
        assert_eq!(poset.height(), 3);
        assert_eq!(poset.width(), 2);
    }
}
