// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Packed deterministic kernels for Schreyer resolutions.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::fmt;
use std::collections::BTreeMap;

use groebner_kernel::{
    module_groebner_basis, module_groebner_basis_binary_extension, module_groebner_basis_f2,
    module_groebner_basis_prime, module_leading_terms, module_schreyer_syzygies,
    module_schreyer_syzygies_binary_extension,
    module_schreyer_syzygies_binary_extension_with_hilbert_target, module_schreyer_syzygies_f2,
    module_schreyer_syzygies_f2_with_hilbert_target, module_schreyer_syzygies_prime,
    module_schreyer_syzygies_prime_with_hilbert_target,
    module_schreyer_syzygies_with_hilbert_target, BinaryExtensionField16, BinaryField, Field,
    GroebnerError, GroebnerOptions, ModuleBasisOutput, ModuleSyzygyOutput, PrimeField32,
    SparseModuleElement, SparseModuleOrder, SparseModuleTerm, StandardModuleHilbertTarget,
};
use module_core::ModuleTerm;
use poly_core::{MonomialArena, MonomialArenaError, MonomialId};

const MIN_HILBERT_GUIDANCE_LATER_PAIRS: usize = 16;

trait StandardGradedBackend<F: Field> {
    fn basis(
        field: &F,
        generators: &[SparseModuleElement<F::Element>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleBasisOutput<F::Element>, GroebnerError>;

    fn syzygies(
        field: &F,
        basis: &[SparseModuleElement<F::Element>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleSyzygyOutput<F::Element>, GroebnerError>;

    fn guided_syzygies(
        field: &F,
        basis: &[SparseModuleElement<F::Element>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
        target: &StandardModuleHilbertTarget,
    ) -> Result<ModuleSyzygyOutput<F::Element>, GroebnerError>;
}

struct GenericBackend;

impl<F: Field> StandardGradedBackend<F> for GenericBackend {
    fn basis(
        field: &F,
        generators: &[SparseModuleElement<F::Element>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleBasisOutput<F::Element>, GroebnerError> {
        module_groebner_basis(field, generators, order, options)
    }

    fn syzygies(
        field: &F,
        basis: &[SparseModuleElement<F::Element>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleSyzygyOutput<F::Element>, GroebnerError> {
        module_schreyer_syzygies(field, basis, order, options)
    }

    fn guided_syzygies(
        field: &F,
        basis: &[SparseModuleElement<F::Element>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
        target: &StandardModuleHilbertTarget,
    ) -> Result<ModuleSyzygyOutput<F::Element>, GroebnerError> {
        module_schreyer_syzygies_with_hilbert_target(field, basis, order, options, target)
    }
}

struct PrimeBackend;

impl StandardGradedBackend<PrimeField32> for PrimeBackend {
    fn basis(
        field: &PrimeField32,
        generators: &[SparseModuleElement<u32>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleBasisOutput<u32>, GroebnerError> {
        module_groebner_basis_prime(field, generators, order, options)
    }

    fn syzygies(
        field: &PrimeField32,
        basis: &[SparseModuleElement<u32>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleSyzygyOutput<u32>, GroebnerError> {
        module_schreyer_syzygies_prime(field, basis, order, options)
    }

    fn guided_syzygies(
        field: &PrimeField32,
        basis: &[SparseModuleElement<u32>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
        target: &StandardModuleHilbertTarget,
    ) -> Result<ModuleSyzygyOutput<u32>, GroebnerError> {
        module_schreyer_syzygies_prime_with_hilbert_target(field, basis, order, options, target)
    }
}

struct F2Backend;

impl StandardGradedBackend<BinaryField> for F2Backend {
    fn basis(
        _: &BinaryField,
        generators: &[SparseModuleElement<u8>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleBasisOutput<u8>, GroebnerError> {
        module_groebner_basis_f2(generators, order, options)
    }

    fn syzygies(
        _: &BinaryField,
        basis: &[SparseModuleElement<u8>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleSyzygyOutput<u8>, GroebnerError> {
        module_schreyer_syzygies_f2(basis, order, options)
    }

    fn guided_syzygies(
        _: &BinaryField,
        basis: &[SparseModuleElement<u8>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
        target: &StandardModuleHilbertTarget,
    ) -> Result<ModuleSyzygyOutput<u8>, GroebnerError> {
        module_schreyer_syzygies_f2_with_hilbert_target(basis, order, options, target)
    }
}

struct BinaryExtensionBackend;

impl StandardGradedBackend<BinaryExtensionField16> for BinaryExtensionBackend {
    fn basis(
        field: &BinaryExtensionField16,
        generators: &[SparseModuleElement<u16>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleBasisOutput<u16>, GroebnerError> {
        module_groebner_basis_binary_extension(field, generators, order, options)
    }

    fn syzygies(
        field: &BinaryExtensionField16,
        basis: &[SparseModuleElement<u16>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
    ) -> Result<ModuleSyzygyOutput<u16>, GroebnerError> {
        module_schreyer_syzygies_binary_extension(field, basis, order, options)
    }

    fn guided_syzygies(
        field: &BinaryExtensionField16,
        basis: &[SparseModuleElement<u16>],
        order: &SparseModuleOrder,
        options: &GroebnerOptions,
        target: &StandardModuleHilbertTarget,
    ) -> Result<ModuleSyzygyOutput<u16>, GroebnerError> {
        module_schreyer_syzygies_binary_extension_with_hilbert_target(
            field, basis, order, options, target,
        )
    }
}

/// One exact standard-graded Betti multiplicity from a packed resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandardGradedBettiEntry {
    /// Homological degree.
    pub homological_degree: usize,
    /// Standard internal degree.
    pub internal_degree: usize,
    /// Multiplicity in this bidegree.
    pub multiplicity: usize,
}

/// Exact standard-graded Betti data computed without materializing public
/// polynomial differential matrices between Schreyer steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandardGradedBettiOutput {
    /// Nonzero entries in homological-degree then internal-degree order.
    pub entries: Vec<StandardGradedBettiEntry>,
    /// Largest homological degree with a nonzero entry.
    pub length: usize,
}

/// One certified differential in a packed standard-graded resolution.
///
/// `columns` are the images of the ordered source basis in the preceding
/// free module. `source_generator_degrees` records their exact homogeneous
/// degrees, so a facade can project the complete frame without repeating
/// homogeneity scans between Schreyer steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandardGradedResolutionFrame<C> {
    /// Sparse columns of this differential.
    pub columns: Vec<SparseModuleElement<C>>,
    /// Internal degree of each ordered source generator.
    pub source_generator_degrees: Vec<usize>,
}

/// Complete certified packed Schreyer resolution of a standard-graded
/// polynomial module.
///
/// The first frame maps to the supplied degree-zero free module. Every later
/// frame maps to the source of the preceding frame. An empty vector therefore
/// represents a free module, while a nonempty result terminates exactly when
/// the final kernel is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StandardGradedResolutionOutput<C> {
    /// Differentials in increasing positive homological degree.
    pub differentials: Vec<StandardGradedResolutionFrame<C>>,
}

impl<C: Clone + Eq> StandardGradedResolutionOutput<C> {
    /// Cancel every scalar unit in homological degrees at least two.
    ///
    /// These cancellations do not change the degree-zero presentation or its
    /// augmentation. They are performed on the retained sparse packed frame,
    /// before a facade materializes coefficient-library polynomial objects.
    /// Scalar units in the first differential are intentionally left to the
    /// facade because cancelling them also changes the represented module's
    /// augmentation.
    pub fn minimalize_higher_differentials<F: Field<Element = C>>(
        &mut self,
        field: &F,
    ) -> Result<(), StandardGradedBettiError> {
        while let Some((index, pivot_row, pivot_column, pivot_inverse)) =
            find_higher_unit(&self.differentials, field)?
        {
            cancel_higher_unit(
                &mut self.differentials,
                field,
                index,
                pivot_row,
                pivot_column,
                &pivot_inverse,
            )?;
        }
        if let Some(first_empty) = self
            .differentials
            .iter()
            .position(|frame| frame.columns.is_empty())
        {
            if self.differentials[first_empty..]
                .iter()
                .any(|frame| !frame.columns.is_empty())
            {
                return Err(StandardGradedBettiError::Grading);
            }
            self.differentials.truncate(first_empty);
        }
        Ok(())
    }
}

fn find_higher_unit<C: Clone + Eq, F: Field<Element = C>>(
    frames: &[StandardGradedResolutionFrame<C>],
    field: &F,
) -> Result<Option<(usize, usize, usize, C)>, StandardGradedBettiError> {
    for index in 1..frames.len() {
        let target_degrees = &frames[index - 1].source_generator_degrees;
        let frame = &frames[index];
        for (column, value) in frame.columns.iter().enumerate() {
            if value.rank() != target_degrees.len() {
                return Err(StandardGradedBettiError::Grading);
            }
            let source_degree = frame
                .source_generator_degrees
                .get(column)
                .ok_or(StandardGradedBettiError::Grading)?;
            for (row, polynomial) in value.coordinates(field).iter().enumerate() {
                if target_degrees[row] != *source_degree || polynomial.terms().len() != 1 {
                    continue;
                }
                let (monomial, coefficient) = &polynomial.terms()[0];
                if monomial.total_degree() == Some(0) && !field.is_zero(coefficient) {
                    return Ok(Some((index, row, column, field.inverse(coefficient))));
                }
            }
        }
    }
    Ok(None)
}

fn cancel_higher_unit<C: Clone + Eq, F: Field<Element = C>>(
    frames: &mut [StandardGradedResolutionFrame<C>],
    field: &F,
    index: usize,
    pivot_row: usize,
    pivot_column: usize,
    pivot_inverse: &C,
) -> Result<(), StandardGradedBettiError> {
    if index == 0 || index >= frames.len() {
        return Err(StandardGradedBettiError::Grading);
    }
    let current = frames[index]
        .columns
        .iter()
        .map(|column| column.coordinates(field))
        .collect::<Vec<_>>();
    let target_rank = frames[index - 1].source_generator_degrees.len();
    let source_rank = frames[index].source_generator_degrees.len();
    if current.len() != source_rank
        || current.iter().any(|column| column.len() != target_rank)
        || pivot_row >= target_rank
        || pivot_column >= source_rank
    {
        return Err(StandardGradedBettiError::Grading);
    }

    let retained_rows = (0..target_rank)
        .filter(|row| *row != pivot_row)
        .collect::<Vec<_>>();
    let retained_columns = (0..source_rank)
        .filter(|column| *column != pivot_column)
        .collect::<Vec<_>>();
    let mut reduced = Vec::with_capacity(retained_columns.len());
    for column in retained_columns {
        let mut coordinates = Vec::with_capacity(retained_rows.len());
        for &row in &retained_rows {
            let left = scale_polynomial(&current[pivot_column][row], pivot_inverse, field);
            let correction = left.multiply(&current[column][pivot_row], field, None);
            coordinates.push(current[column][row].sub(&correction, field));
        }
        reduced.push(SparseModuleElement::from_coordinates(coordinates, field));
    }
    frames[index].columns = reduced;
    frames[index].source_generator_degrees.remove(pivot_column);

    frames[index - 1].columns.remove(pivot_row);
    frames[index - 1].source_generator_degrees.remove(pivot_row);

    if index + 1 < frames.len() {
        frames[index + 1].columns = frames[index + 1]
            .columns
            .iter()
            .map(|column| {
                let mut coordinates = column.coordinates(field);
                if coordinates.len() != source_rank {
                    return Err(StandardGradedBettiError::Grading);
                }
                coordinates.remove(pivot_column);
                Ok(SparseModuleElement::from_coordinates(coordinates, field))
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    Ok(())
}

fn scale_polynomial<C: Clone + Eq, F: Field<Element = C>>(
    polynomial: &groebner_kernel::SparsePolynomial<C>,
    scalar: &C,
    field: &F,
) -> groebner_kernel::SparsePolynomial<C> {
    groebner_kernel::SparsePolynomial::from_sorted_terms(
        polynomial.arity(),
        polynomial.order(),
        polynomial
            .terms()
            .iter()
            .filter_map(|(monomial, coefficient)| {
                let coefficient = field.mul(coefficient, scalar);
                (!field.is_zero(&coefficient)).then(|| (monomial.clone(), coefficient))
            }),
        field,
    )
}

/// Failure of the packed standard-graded Betti lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StandardGradedBettiError {
    /// A Gröbner or Schreyer kernel rejected the computation.
    Groebner(GroebnerError),
    /// A supplied or computed frame was not standard-graded homogeneous.
    Grading,
    /// A standard degree did not fit the bounded packed lane.
    DegreeOverflow,
    /// A residue differential exceeded its homogeneous block.
    ResidueRank,
    /// Schreyer iteration exceeded Hilbert's syzygy bound.
    ProjectiveDimension {
        /// Number of polynomial variables and admitted bound.
        variables: usize,
    },
}

impl fmt::Display for StandardGradedBettiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Groebner(error) => error.fmt(formatter),
            Self::Grading => formatter.write_str("packed Betti frame is not standard-graded"),
            Self::DegreeOverflow => formatter.write_str("packed Betti degree overflow"),
            Self::ResidueRank => formatter.write_str("packed residue rank exceeds its degree block"),
            Self::ProjectiveDimension { variables } => write!(
                formatter,
                "packed Schreyer iteration exceeds the {variables}-variable projective-dimension bound"
            ),
        }
    }
}

impl std::error::Error for StandardGradedBettiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Groebner(error) => Some(error),
            Self::Grading
            | Self::DegreeOverflow
            | Self::ResidueRank
            | Self::ProjectiveDimension { .. } => None,
        }
    }
}

impl From<GroebnerError> for StandardGradedBettiError {
    fn from(value: GroebnerError) -> Self {
        Self::Groebner(value)
    }
}

/// Compute exact standard-graded Betti multiplicities while retaining packed
/// certified module frames across Schreyer iterations.
///
/// `target_generator_degrees` are the internal degrees of the basis of
/// `F_0`, and `presentation_generator_degrees` are those of the declared
/// presentation source. Every nonzero presentation column is checked against
/// its declared degree. The generic field lane is the correctness floor.
pub fn standard_graded_betti<F: Field>(
    field: &F,
    presentation: &[SparseModuleElement<F::Element>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedBettiOutput, StandardGradedBettiError> {
    standard_graded_betti_with::<F, GenericBackend>(
        field,
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
        order,
        options,
    )
}

/// Compute packed exact standard-graded Betti data over a word-sized prime
/// field using the specialized module row reducers.
pub fn standard_graded_betti_prime(
    field: &PrimeField32,
    presentation: &[SparseModuleElement<u32>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedBettiOutput, StandardGradedBettiError> {
    standard_graded_betti_with::<PrimeField32, PrimeBackend>(
        field,
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
        order,
        options,
    )
}

/// Compute packed exact standard-graded Betti data over `F_2` using bit-packed
/// module row reduction.
pub fn standard_graded_betti_f2(
    presentation: &[SparseModuleElement<u8>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedBettiOutput, StandardGradedBettiError> {
    standard_graded_betti_with::<BinaryField, F2Backend>(
        &BinaryField,
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
        order,
        options,
    )
}

/// Compute packed exact standard-graded Betti data over `F_{2^e}` using the
/// declared polynomial basis and bit-packed extension-field rows.
pub fn standard_graded_betti_binary_extension(
    field: &BinaryExtensionField16,
    presentation: &[SparseModuleElement<u16>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedBettiOutput, StandardGradedBettiError> {
    standard_graded_betti_with::<BinaryExtensionField16, BinaryExtensionBackend>(
        field,
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
        order,
        options,
    )
}

/// Compute a complete certified standard-graded Schreyer resolution while
/// retaining every sparse packed differential.
pub fn standard_graded_resolution<F: Field>(
    field: &F,
    presentation: &[SparseModuleElement<F::Element>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedResolutionOutput<F::Element>, StandardGradedBettiError> {
    standard_graded_resolution_with::<F, GenericBackend>(
        field,
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
        order,
        options,
    )
}

/// Prime-field specialization of [`standard_graded_resolution`].
pub fn standard_graded_resolution_prime(
    field: &PrimeField32,
    presentation: &[SparseModuleElement<u32>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedResolutionOutput<u32>, StandardGradedBettiError> {
    standard_graded_resolution_with::<PrimeField32, PrimeBackend>(
        field,
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
        order,
        options,
    )
}

/// Binary-field specialization of [`standard_graded_resolution`].
pub fn standard_graded_resolution_f2(
    presentation: &[SparseModuleElement<u8>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedResolutionOutput<u8>, StandardGradedBettiError> {
    standard_graded_resolution_with::<BinaryField, F2Backend>(
        &BinaryField,
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
        order,
        options,
    )
}

/// Binary-extension-field specialization of
/// [`standard_graded_resolution`].
pub fn standard_graded_resolution_binary_extension(
    field: &BinaryExtensionField16,
    presentation: &[SparseModuleElement<u16>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedResolutionOutput<u16>, StandardGradedBettiError> {
    standard_graded_resolution_with::<BinaryExtensionField16, BinaryExtensionBackend>(
        field,
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
        order,
        options,
    )
}

fn standard_graded_resolution_with<F, B>(
    field: &F,
    presentation: &[SparseModuleElement<F::Element>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedResolutionOutput<F::Element>, StandardGradedBettiError>
where
    F: Field,
    B: StandardGradedBackend<F>,
{
    if presentation.len() != presentation_generator_degrees.len() {
        return Err(StandardGradedBettiError::Grading);
    }
    if presentation.is_empty() {
        return Ok(StandardGradedResolutionOutput {
            differentials: Vec::new(),
        });
    }
    let first = &presentation[0];
    if first.rank() != target_generator_degrees.len()
        || order.base_order() != first.polynomial_order()
    {
        return Err(StandardGradedBettiError::Grading);
    }
    verify_presentation_degrees(
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
    )?;

    let mut current_order = order.clone();
    let mut basis = B::basis(field, presentation, &current_order, options)?.basis;
    if basis.is_empty() {
        return Ok(StandardGradedResolutionOutput {
            differentials: Vec::new(),
        });
    }
    let arity = first.arity();
    let mut ambient_degrees = target_generator_degrees.to_vec();
    let mut differentials = Vec::new();

    loop {
        let basis_degrees = homogeneous_generator_degrees(&basis, &ambient_degrees)?;
        let leaders = module_leading_terms(field, &basis, &current_order)?;
        if !has_component_compatible_pair(&leaders) {
            differentials.push(StandardGradedResolutionFrame {
                columns: basis,
                source_generator_degrees: basis_degrees,
            });
            break;
        }
        if differentials.len() + 1 >= arity {
            return Err(StandardGradedBettiError::ProjectiveDimension { variables: arity });
        }
        let syzygies = if should_use_hilbert_guidance(&leaders)? {
            let target = StandardModuleHilbertTarget::from_submodule_leading_terms(
                arity,
                &ambient_degrees,
                basis_degrees.clone(),
                &leaders
                    .iter()
                    .map(|leader| (leader.monomial().clone(), leader.component()))
                    .collect::<Vec<_>>(),
            )?;
            B::guided_syzygies(field, &basis, &current_order, options, &target).or_else(
                |error| match error {
                    GroebnerError::VerificationFailed => {
                        B::syzygies(field, &basis, &current_order, options)
                    }
                    _ => Err(error),
                },
            )?
        } else {
            B::syzygies(field, &basis, &current_order, options)?
        };
        differentials.push(StandardGradedResolutionFrame {
            columns: basis,
            source_generator_degrees: basis_degrees.clone(),
        });
        if syzygies.generators.is_empty() {
            break;
        }
        current_order = SparseModuleOrder::schreyer(current_order, leaders);
        ambient_degrees = basis_degrees;
        basis = syzygies.generators;
    }

    Ok(StandardGradedResolutionOutput { differentials })
}

fn standard_graded_betti_with<F, B>(
    field: &F,
    presentation: &[SparseModuleElement<F::Element>],
    target_generator_degrees: &[usize],
    presentation_generator_degrees: &[usize],
    order: &SparseModuleOrder,
    options: &GroebnerOptions,
) -> Result<StandardGradedBettiOutput, StandardGradedBettiError>
where
    F: Field,
    B: StandardGradedBackend<F>,
{
    let mut module_degrees = vec![degree_counts(target_generator_degrees)?];
    let mut differential_ranks = vec![BTreeMap::<usize, usize>::new()];
    if presentation.len() != presentation_generator_degrees.len() {
        return Err(StandardGradedBettiError::Grading);
    }
    if presentation.is_empty() {
        return finish_betti(module_degrees, differential_ranks);
    }
    let first = &presentation[0];
    if first.rank() != target_generator_degrees.len()
        || order.base_order() != first.polynomial_order()
    {
        return Err(StandardGradedBettiError::Grading);
    }
    verify_presentation_degrees(
        presentation,
        target_generator_degrees,
        presentation_generator_degrees,
    )?;
    let mut current_order = order.clone();
    let mut basis = B::basis(field, presentation, &current_order, options)?.basis;
    if basis.is_empty() {
        return finish_betti(module_degrees, differential_ranks);
    }
    let arity = first.arity();
    let mut ambient_degrees = target_generator_degrees.to_vec();

    loop {
        let basis_degrees = homogeneous_generator_degrees(&basis, &ambient_degrees)?;
        module_degrees.push(degree_counts(&basis_degrees)?);
        differential_ranks.push(residue_ranks(
            field,
            &basis,
            &ambient_degrees,
            &basis_degrees,
        )?);

        let leaders = module_leading_terms(field, &basis, &current_order)?;
        if !has_component_compatible_pair(&leaders) {
            break;
        }
        let homological_degree = differential_ranks.len() - 1;
        if homological_degree >= arity {
            return Err(StandardGradedBettiError::ProjectiveDimension { variables: arity });
        }
        let syzygies = if should_use_hilbert_guidance(&leaders)? {
            let target = StandardModuleHilbertTarget::from_submodule_leading_terms(
                arity,
                &ambient_degrees,
                basis_degrees.clone(),
                &leaders
                    .iter()
                    .map(|leader| (leader.monomial().clone(), leader.component()))
                    .collect::<Vec<_>>(),
            )?;
            B::guided_syzygies(field, &basis, &current_order, options, &target).or_else(
                |error| match error {
                    GroebnerError::VerificationFailed => {
                        B::syzygies(field, &basis, &current_order, options)
                    }
                    _ => Err(error),
                },
            )?
        } else {
            B::syzygies(field, &basis, &current_order, options)?
        };
        if syzygies.generators.is_empty() {
            break;
        }
        let next_order = SparseModuleOrder::schreyer(current_order, leaders);
        ambient_degrees = basis_degrees;
        basis = syzygies.generators;
        current_order = next_order;
    }

    finish_betti(module_degrees, differential_ranks)
}

fn verify_presentation_degrees<C: Clone + Eq>(
    presentation: &[SparseModuleElement<C>],
    target_degrees: &[usize],
    source_degrees: &[usize],
) -> Result<(), StandardGradedBettiError> {
    for (column, declared_degree) in presentation.iter().zip(source_degrees) {
        if column.rank() != target_degrees.len() {
            return Err(StandardGradedBettiError::Grading);
        }
        let mut actual_degree = None;
        for (term, _) in column.terms() {
            let monomial_degree = term
                .monomial()
                .total_degree()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(StandardGradedBettiError::DegreeOverflow)?;
            let term_degree = target_degrees
                .get(term.component())
                .and_then(|target| target.checked_add(monomial_degree))
                .ok_or(StandardGradedBettiError::DegreeOverflow)?;
            match actual_degree {
                None => actual_degree = Some(term_degree),
                Some(known) if known == term_degree => {}
                Some(_) => return Err(StandardGradedBettiError::Grading),
            }
        }
        if actual_degree.is_some_and(|degree| degree != *declared_degree) {
            return Err(StandardGradedBettiError::Grading);
        }
    }
    Ok(())
}

fn homogeneous_generator_degrees<C: Clone + Eq>(
    basis: &[SparseModuleElement<C>],
    ambient_degrees: &[usize],
) -> Result<Vec<usize>, StandardGradedBettiError> {
    basis
        .iter()
        .map(|generator| {
            if generator.rank() != ambient_degrees.len() {
                return Err(StandardGradedBettiError::Grading);
            }
            let mut degree = None;
            for (term, _) in generator.terms() {
                let monomial_degree = term
                    .monomial()
                    .total_degree()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or(StandardGradedBettiError::DegreeOverflow)?;
                let term_degree = ambient_degrees
                    .get(term.component())
                    .and_then(|ambient| ambient.checked_add(monomial_degree))
                    .ok_or(StandardGradedBettiError::DegreeOverflow)?;
                match degree {
                    None => degree = Some(term_degree),
                    Some(known) if known == term_degree => {}
                    Some(_) => return Err(StandardGradedBettiError::Grading),
                }
            }
            degree.ok_or(StandardGradedBettiError::Grading)
        })
        .collect()
}

fn degree_counts(degrees: &[usize]) -> Result<BTreeMap<usize, usize>, StandardGradedBettiError> {
    let mut counts = BTreeMap::new();
    for degree in degrees {
        let count = counts.entry(*degree).or_insert(0_usize);
        *count = count
            .checked_add(1)
            .ok_or(StandardGradedBettiError::DegreeOverflow)?;
    }
    Ok(counts)
}

fn residue_ranks<F: Field>(
    field: &F,
    columns: &[SparseModuleElement<F::Element>],
    target_degrees: &[usize],
    source_degrees: &[usize],
) -> Result<BTreeMap<usize, usize>, StandardGradedBettiError> {
    let mut source_blocks = BTreeMap::<usize, Vec<usize>>::new();
    let mut target_blocks = BTreeMap::<usize, Vec<usize>>::new();
    for (column, degree) in source_degrees.iter().enumerate() {
        source_blocks.entry(*degree).or_default().push(column);
    }
    for (row, degree) in target_degrees.iter().enumerate() {
        target_blocks.entry(*degree).or_default().push(row);
    }
    let mut ranks = BTreeMap::new();
    for (degree, block_columns) in source_blocks {
        let Some(block_rows) = target_blocks.get(&degree) else {
            continue;
        };
        let mut matrix = vec![vec![field.zero(); block_columns.len()]; block_rows.len()];
        for (block_column, column) in block_columns.iter().enumerate() {
            for (term, coefficient) in columns[*column].terms() {
                if term.monomial().total_degree() == Some(0) {
                    let block_row = block_rows
                        .iter()
                        .position(|row| *row == term.component())
                        .ok_or(StandardGradedBettiError::Grading)?;
                    matrix[block_row][block_column] = coefficient.clone();
                }
            }
        }
        let rank = field_rank(field, &mut matrix);
        if rank != 0 {
            ranks.insert(degree, rank);
        }
    }
    Ok(ranks)
}

fn field_rank<F: Field>(field: &F, matrix: &mut [Vec<F::Element>]) -> usize {
    let columns = matrix.first().map_or(0, Vec::len);
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..matrix.len()).find(|row| !field.is_zero(&matrix[*row][column]))
        else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse = field.inverse(&matrix[rank][column]);
        for entry in &mut matrix[rank][column..] {
            *entry = field.mul(entry, &inverse);
        }
        let pivot_row = matrix[rank].clone();
        for row in matrix.iter_mut().skip(rank + 1) {
            let factor = field.neg(&row[column]);
            if field.is_zero(&factor) {
                continue;
            }
            for index in column..columns {
                row[index] = field.add(&row[index], &field.mul(&factor, &pivot_row[index]));
            }
        }
        rank += 1;
        if rank == matrix.len() {
            break;
        }
    }
    rank
}

fn has_component_compatible_pair(leaders: &[SparseModuleTerm]) -> bool {
    leaders.iter().enumerate().any(|(left, leader)| {
        leaders[(left + 1)..]
            .iter()
            .any(|other| leader.component() == other.component())
    })
}

fn should_use_hilbert_guidance(
    leaders: &[SparseModuleTerm],
) -> Result<bool, StandardGradedBettiError> {
    let mut degree_counts = BTreeMap::<u64, usize>::new();
    let mut total_pairs = 0_usize;
    for left in 0..leaders.len() {
        for right in (left + 1)..leaders.len() {
            if leaders[left].component() != leaders[right].component() {
                continue;
            }
            let degree = leaders[left]
                .monomial()
                .lcm(leaders[right].monomial())
                .map_err(|_| StandardGradedBettiError::DegreeOverflow)?
                .total_degree()
                .ok_or(StandardGradedBettiError::DegreeOverflow)?;
            *degree_counts.entry(degree).or_default() += 1;
            total_pairs += 1;
        }
    }
    Ok(degree_counts.first_key_value().is_some_and(|(_, first)| {
        total_pairs.saturating_sub(*first) >= MIN_HILBERT_GUIDANCE_LATER_PAIRS
    }))
}

fn finish_betti(
    module_degrees: Vec<BTreeMap<usize, usize>>,
    differential_ranks: Vec<BTreeMap<usize, usize>>,
) -> Result<StandardGradedBettiOutput, StandardGradedBettiError> {
    let mut entries = Vec::new();
    for (homological_degree, degrees) in module_degrees.iter().enumerate() {
        for (internal_degree, dimension) in degrees {
            let outgoing = differential_ranks[homological_degree]
                .get(internal_degree)
                .copied()
                .unwrap_or(0);
            let incoming = differential_ranks
                .get(homological_degree + 1)
                .and_then(|ranks| ranks.get(internal_degree))
                .copied()
                .unwrap_or(0);
            let multiplicity = dimension
                .checked_sub(
                    outgoing
                        .checked_add(incoming)
                        .ok_or(StandardGradedBettiError::ResidueRank)?,
                )
                .ok_or(StandardGradedBettiError::ResidueRank)?;
            if multiplicity != 0 {
                entries.push(StandardGradedBettiEntry {
                    homological_degree,
                    internal_degree: *internal_degree,
                    multiplicity,
                });
            }
        }
    }
    let length = entries
        .iter()
        .map(|entry| entry.homological_degree)
        .max()
        .unwrap_or(0);
    Ok(StandardGradedBettiOutput { entries, length })
}

/// Failure to construct a bounded Schreyer pair schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolutionKernelError {
    /// A monomial operation failed in the shared arena.
    Monomial(MonomialArenaError),
    /// The number of component-compatible critical pairs exceeded its cap.
    PairLimit {
        /// Maximum admitted pair count.
        limit: usize,
    },
}

impl fmt::Display for ResolutionKernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Monomial(error) => error.fmt(formatter),
            Self::PairLimit { limit } => {
                write!(formatter, "Schreyer pair count exceeds limit {limit}")
            }
        }
    }
}

impl std::error::Error for ResolutionKernelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Monomial(error) => Some(error),
            Self::PairLimit { .. } => None,
        }
    }
}

impl From<MonomialArenaError> for ResolutionKernelError {
    fn from(value: MonomialArenaError) -> Self {
        Self::Monomial(value)
    }
}

/// One component-compatible critical pair in a Schreyer frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SchreyerPair {
    left: usize,
    right: usize,
    lcm: MonomialId,
}

impl SchreyerPair {
    /// Zero-based left leading-term position.
    pub const fn left(self) -> usize {
        self.left
    }

    /// Zero-based right leading-term position.
    pub const fn right(self) -> usize {
        self.right
    }

    /// Interned least common multiple of the polynomial monomials.
    pub const fn lcm(self) -> MonomialId {
        self.lcm
    }
}

/// Critical pairs sharing one total LCM degree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchreyerDegreeCell {
    degree: u64,
    pairs: Vec<SchreyerPair>,
}

impl SchreyerDegreeCell {
    /// Total polynomial degree of every pair LCM in this cell.
    pub const fn degree(&self) -> u64 {
        self.degree
    }

    /// Deterministically ordered pairs in this degree cell.
    pub fn pairs(&self) -> &[SchreyerPair] {
        &self.pairs
    }
}

/// Group all component-compatible Schreyer pairs into ascending degree cells.
///
/// Pairs inside a cell retain lexicographic `(left, right)` order. The routine
/// interns each LCM once in the caller's shared monomial arena, making the
/// result directly reusable by later symbolic preprocessing.
pub fn schreyer_degree_cells(
    arena: &mut MonomialArena,
    leaders: &[ModuleTerm],
    max_pairs: usize,
) -> Result<Vec<SchreyerDegreeCell>, ResolutionKernelError> {
    let mut cells = BTreeMap::<u64, Vec<SchreyerPair>>::new();
    let mut pair_count = 0_usize;
    for left in 0..leaders.len() {
        for right in (left + 1)..leaders.len() {
            if leaders[left].component() != leaders[right].component() {
                continue;
            }
            pair_count = pair_count
                .checked_add(1)
                .ok_or(ResolutionKernelError::PairLimit { limit: max_pairs })?;
            if pair_count > max_pairs {
                return Err(ResolutionKernelError::PairLimit { limit: max_pairs });
            }
            let lcm = arena.lcm(leaders[left].monomial(), leaders[right].monomial())?;
            cells
                .entry(arena.total_degree(lcm))
                .or_default()
                .push(SchreyerPair { left, right, lcm });
        }
    }
    Ok(cells
        .into_iter()
        .map(|(degree, pairs)| SchreyerDegreeCell { degree, pairs })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use groebner_kernel::{
        BinaryExtensionField16, BinaryField, Monomial, MonomialOrder, PrimeField32,
    };

    fn module_monomial_over<F: Field>(
        field: &F,
        arity: usize,
        rank: usize,
        exponents: &[u32],
        component: usize,
    ) -> SparseModuleElement<F::Element> {
        SparseModuleElement::new(
            arity,
            rank,
            MonomialOrder::GradedReverseLexicographic,
            [(
                SparseModuleTerm::new(
                    Monomial::new(exponents.to_vec().into_boxed_slice()),
                    component,
                ),
                field.one(),
            )],
            field,
        )
    }

    fn module_monomial(
        field: &PrimeField32,
        arity: usize,
        rank: usize,
        exponents: &[u32],
        component: usize,
    ) -> SparseModuleElement<u32> {
        module_monomial_over(field, arity, rank, exponents, component)
    }

    #[test]
    fn pairs_are_component_filtered_and_grouped_by_lcm_degree() {
        let mut arena = MonomialArena::new(3, MonomialOrder::GradedReverseLexicographic);
        let x2 = arena.intern_exponents(&[2, 0, 0]).unwrap();
        let xy = arena.intern_exponents(&[1, 1, 0]).unwrap();
        let yz2 = arena.intern_exponents(&[0, 1, 2]).unwrap();
        let other = arena.intern_exponents(&[0, 0, 1]).unwrap();
        let leaders = [
            ModuleTerm::new(x2, 0),
            ModuleTerm::new(xy, 0),
            ModuleTerm::new(yz2, 0),
            ModuleTerm::new(other, 1),
        ];

        let cells = schreyer_degree_cells(&mut arena, &leaders, 3).unwrap();
        assert_eq!(cells.len(), 3);
        assert_eq!(cells[0].degree(), 3);
        assert_eq!(
            cells[0].pairs(),
            &[SchreyerPair {
                left: 0,
                right: 1,
                lcm: arena.lcm(x2, xy).unwrap()
            }]
        );
        assert_eq!(cells[1].degree(), 4);
        assert_eq!(cells[1].pairs().len(), 1);
        assert_eq!(
            (cells[1].pairs()[0].left(), cells[1].pairs()[0].right()),
            (1, 2)
        );
        assert_eq!(cells[2].degree(), 5);
        assert_eq!(
            (cells[2].pairs()[0].left(), cells[2].pairs()[0].right()),
            (0, 2)
        );
    }

    #[test]
    fn pair_cap_fails_before_returning_a_partial_schedule() {
        let mut arena = MonomialArena::new(1, MonomialOrder::Lexicographic);
        let x = arena.intern_exponents(&[1]).unwrap();
        let leaders = [ModuleTerm::new(x, 0); 3];
        assert_eq!(
            schreyer_degree_cells(&mut arena, &leaders, 2).unwrap_err(),
            ResolutionKernelError::PairLimit { limit: 2 }
        );
    }

    #[test]
    fn packed_betti_stream_resolves_a_seventh_power_without_public_maps() {
        let field = PrimeField32::new(65_521).unwrap();
        let presentation = (0..=7)
            .map(|y_degree| module_monomial(&field, 2, 1, &[7 - y_degree, y_degree], 0))
            .collect::<Vec<_>>();
        let output = standard_graded_betti(
            &field,
            &presentation,
            &[0],
            &[7; 8],
            &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
            &GroebnerOptions::default(),
        )
        .unwrap();
        assert_eq!(
            standard_graded_betti_prime(
                &field,
                &presentation,
                &[0],
                &[7; 8],
                &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic,),
                &GroebnerOptions::default(),
            )
            .unwrap(),
            output
        );
        assert_eq!(
            output.entries,
            vec![
                StandardGradedBettiEntry {
                    homological_degree: 0,
                    internal_degree: 0,
                    multiplicity: 1,
                },
                StandardGradedBettiEntry {
                    homological_degree: 1,
                    internal_degree: 7,
                    multiplicity: 8,
                },
                StandardGradedBettiEntry {
                    homological_degree: 2,
                    internal_degree: 8,
                    multiplicity: 7,
                },
            ]
        );
        assert_eq!(output.length, 2);
    }

    #[test]
    fn packed_betti_stream_computes_residue_cancellation_exactly() {
        let field = PrimeField32::new(101).unwrap();
        let presentation = vec![module_monomial(&field, 1, 1, &[0], 0)];
        let output = standard_graded_betti(
            &field,
            &presentation,
            &[0],
            &[0],
            &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
            &GroebnerOptions::default(),
        )
        .unwrap();
        assert!(output.entries.is_empty());
        assert_eq!(output.length, 0);
        assert_eq!(
            standard_graded_betti(
                &field,
                &presentation,
                &[0],
                &[1],
                &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic,),
                &GroebnerOptions::default(),
            )
            .unwrap_err(),
            StandardGradedBettiError::Grading
        );
    }

    #[test]
    fn packed_betti_stream_iterates_a_complete_intersection() {
        let field = PrimeField32::new(101).unwrap();
        let presentation = (0..3)
            .map(|variable| {
                let mut exponents = vec![0; 3];
                exponents[variable] = 2;
                module_monomial(&field, 3, 1, &exponents, 0)
            })
            .collect::<Vec<_>>();
        let output = standard_graded_betti(
            &field,
            &presentation,
            &[0],
            &[2; 3],
            &SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic),
            &GroebnerOptions::default(),
        )
        .unwrap();
        assert_eq!(
            output
                .entries
                .iter()
                .map(|entry| (
                    entry.homological_degree,
                    entry.internal_degree,
                    entry.multiplicity,
                ))
                .collect::<Vec<_>>(),
            vec![(0, 0, 1), (1, 2, 3), (2, 4, 3), (3, 6, 1)]
        );
        assert_eq!(output.length, 3);
    }

    #[test]
    fn binary_betti_backends_match_the_generic_field_floor() {
        let order = SparseModuleOrder::TermOverPosition(MonomialOrder::GradedReverseLexicographic);
        let options = GroebnerOptions::default();

        let f2 = BinaryField;
        let f2_presentation = (0..3)
            .map(|variable| {
                let mut exponents = vec![0; 3];
                exponents[variable] = 2;
                module_monomial_over(&f2, 3, 1, &exponents, 0)
            })
            .collect::<Vec<_>>();
        let generic =
            standard_graded_betti(&f2, &f2_presentation, &[0], &[2; 3], &order, &options).unwrap();
        assert_eq!(
            standard_graded_betti_f2(&f2_presentation, &[0], &[2; 3], &order, &options).unwrap(),
            generic
        );

        let extension = BinaryExtensionField16::new(0b111).unwrap();
        let extension_presentation = (0..3)
            .map(|variable| {
                let mut exponents = vec![0; 3];
                exponents[variable] = 2;
                module_monomial_over(&extension, 3, 1, &exponents, 0)
            })
            .collect::<Vec<_>>();
        let generic = standard_graded_betti(
            &extension,
            &extension_presentation,
            &[0],
            &[2; 3],
            &order,
            &options,
        )
        .unwrap();
        assert_eq!(
            standard_graded_betti_binary_extension(
                &extension,
                &extension_presentation,
                &[0],
                &[2; 3],
                &order,
                &options,
            )
            .unwrap(),
            generic
        );
    }
}
