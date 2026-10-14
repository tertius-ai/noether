// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Verified primary decompositions of zero submodules in presented modules.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{
    AssociatedPrimeError, GroebnerError, Ideal, MinimalPrimeCoefficientField, PolynomialError,
    PolynomialPrimeCertificate, PolynomialRing,
};
use crate::numthy::FiniteField;
use crate::{
    ContextMismatchError, Field, NoetherError, RationalField, Ring, SemanticType, TypeDescription,
};

use super::associated_primes::{
    construct_module_associated_primes, verify_module_associated_primes,
};
use super::module_groebner::map_columns;
use super::{
    FreeModule, ModuleAssociatedPrimeSet, ModuleError, ModuleGroebnerError, ModuleHomomorphism,
    ModuleMonomialOrder, PresentedModule, PresentedModuleHomomorphism, Submodule,
};

const MAX_PRIMARY_EXPONENT: u64 = 8;
const MAX_PRIMARY_COMBINATIONS: usize = 4_096;

type PolynomialPresentedMap<R> = PresentedModuleHomomorphism<PolynomialRing<R>>;
type PrimaryComponentMaps<R> = (PolynomialPresentedMap<R>, PolynomialPresentedMap<R>);

static MODULE_PRIMARY_COMPONENT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModulePrimaryComponent@1",
    rust_path: "noether::homalg::ModulePrimaryComponent",
    python_path: Some("noether.homalg.ModulePrimaryComponent"),
};

static MODULE_PRIMARY_DECOMPOSITION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModulePrimaryDecomposition@1",
    rust_path: "noether::homalg::ModulePrimaryDecomposition",
    python_path: Some("noether.homalg.ModulePrimaryDecomposition"),
};

/// Failure to construct or verify a primary decomposition of a module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModulePrimaryDecompositionError {
    /// Complete associated-prime discovery or admission failed.
    AssociatedPrimes(AssociatedPrimeError),
    /// Exact module Groebner arithmetic failed.
    Groebner(ModuleGroebnerError),
    /// A based or presented module construction failed.
    Module(ModuleError),
    /// Exact polynomial-ideal arithmetic failed.
    Ideal(GroebnerError),
    /// Exact polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// An operation mixed unequal polynomial contexts.
    ContextMismatch(ContextMismatchError),
    /// The bounded native recovery floor found no complete decomposition.
    Unsupported(String),
    /// Retained evidence failed an exact identity.
    Invalid(String),
    /// A bounded candidate family exceeded its explicit native limit.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed native limit.
        limit: usize,
    },
}

impl fmt::Display for ModulePrimaryDecompositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AssociatedPrimes(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Module(error) => error.fmt(formatter),
            Self::Ideal(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Unsupported(detail) => {
                write!(
                    formatter,
                    "module primary decomposition is unavailable: {detail}"
                )
            }
            Self::Invalid(detail) => {
                write!(
                    formatter,
                    "invalid module primary-decomposition evidence: {detail}"
                )
            }
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "module primary decomposition exceeded its {resource} limit of {limit}"
            ),
        }
    }
}

impl std::error::Error for ModulePrimaryDecompositionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AssociatedPrimes(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::Module(error) => Some(error),
            Self::Ideal(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::Unsupported(_) | Self::Invalid(_) | Self::ResourceLimit { .. } => None,
        }
    }
}

impl From<AssociatedPrimeError> for ModulePrimaryDecompositionError {
    fn from(value: AssociatedPrimeError) -> Self {
        Self::AssociatedPrimes(value)
    }
}

impl From<ModuleGroebnerError> for ModulePrimaryDecompositionError {
    fn from(value: ModuleGroebnerError) -> Self {
        match value {
            ModuleGroebnerError::ResourceLimit { resource, limit } => {
                Self::ResourceLimit { resource, limit }
            }
            other => Self::Groebner(other),
        }
    }
}

impl From<ModuleError> for ModulePrimaryDecompositionError {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}

impl From<GroebnerError> for ModulePrimaryDecompositionError {
    fn from(value: GroebnerError) -> Self {
        match value {
            GroebnerError::ResourceLimit { resource, limit } => {
                Self::ResourceLimit { resource, limit }
            }
            other => Self::Ideal(other),
        }
    }
}

impl From<PolynomialError> for ModulePrimaryDecompositionError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<ContextMismatchError> for ModulePrimaryDecompositionError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// One primary submodule, retained as an exact kernel inside its source.
///
/// If `M = F/N`, `preimage` is the submodule `C` of `F` containing `N`.
/// The actual component is `C/N`, represented by `inclusion`, and its quotient
/// is `F/C`, represented by `quotient_projection`. Complete EHV evidence proves
/// that this quotient has exactly one associated prime.
#[derive(Clone)]
pub struct ModulePrimaryComponent<R: Field> {
    preimage: Submodule<PolynomialRing<R>>,
    inclusion: PresentedModuleHomomorphism<PolynomialRing<R>>,
    quotient_projection: PresentedModuleHomomorphism<PolynomialRing<R>>,
    quotient_associated_primes: ModuleAssociatedPrimeSet<R>,
}

impl<R: Field> ModulePrimaryComponent<R> {
    /// Preimage `C` in the retained generator free module `F` of `M = F/N`.
    pub fn preimage(&self) -> &Submodule<PolynomialRing<R>> {
        &self.preimage
    }

    /// Exact inclusion of the primary submodule `C/N` into the source module.
    pub fn inclusion(&self) -> &PresentedModuleHomomorphism<PolynomialRing<R>> {
        &self.inclusion
    }

    /// Exact quotient projection `M -> M/(C/N) = F/C`.
    pub fn quotient_projection(&self) -> &PresentedModuleHomomorphism<PolynomialRing<R>> {
        &self.quotient_projection
    }

    /// Complete singleton associated-prime evidence for the quotient `F/C`.
    pub fn quotient_associated_primes(&self) -> &ModuleAssociatedPrimeSet<R> {
        &self.quotient_associated_primes
    }

    /// The unique associated prime of the quotient, proving primaryness.
    pub fn associated_prime(&self) -> &PolynomialPrimeCertificate<R> {
        &self.quotient_associated_primes.prime_certificates()[0]
    }

    /// Exact definition of primaryness used by this certificate.
    pub const fn theorem(&self) -> &'static str {
        "a submodule Q of M is P-primary when Ass(M/Q) is the singleton {P}"
    }
}

impl<R: Field> PartialEq for ModulePrimaryComponent<R> {
    fn eq(&self, other: &Self) -> bool {
        self.preimage.generator_map() == other.preimage.generator_map()
            && self.inclusion == other.inclusion
            && self.inclusion.relation_lift() == other.inclusion.relation_lift()
            && self.quotient_projection == other.quotient_projection
            && self.quotient_projection.relation_lift() == other.quotient_projection.relation_lift()
            && self.quotient_associated_primes == other.quotient_associated_primes
    }
}

impl<R: Field> Eq for ModulePrimaryComponent<R> {}

impl<R: Field> fmt::Debug for ModulePrimaryComponent<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModulePrimaryComponent")
            .field("preimage_generators", &self.preimage.generator_count())
            .field("associated_prime", self.associated_prime().ideal())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for ModulePrimaryComponent<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ModulePrimaryComponent(generators={}, associated_prime={})",
            self.preimage.generator_count(),
            self.associated_prime().ideal()
        )
    }
}

/// Complete verified primary decomposition of the zero submodule of `source`.
#[derive(Clone)]
pub struct ModulePrimaryDecomposition<R: Field> {
    source: PresentedModule<PolynomialRing<R>>,
    source_associated_primes: ModuleAssociatedPrimeSet<R>,
    components: Vec<ModulePrimaryComponent<R>>,
}

impl<R: Field> ModulePrimaryDecomposition<R> {
    /// Retained source module whose zero submodule was decomposed.
    pub fn source(&self) -> &PresentedModule<PolynomialRing<R>> {
        &self.source
    }

    /// Complete EHV associated-prime evidence for the source module.
    pub fn source_associated_primes(&self) -> &ModuleAssociatedPrimeSet<R> {
        &self.source_associated_primes
    }

    /// Primary submodules in the canonical order of the source associated primes.
    pub fn components(&self) -> &[ModulePrimaryComponent<R>] {
        &self.components
    }

    /// Canonically ordered distinct associated-prime certificates.
    pub fn associated_primes(&self) -> Vec<&PolynomialPrimeCertificate<R>> {
        self.components
            .iter()
            .map(ModulePrimaryComponent::associated_prime)
            .collect()
    }

    /// Exact reconstruction and primaryness statement certified by the result.
    pub const fn theorem(&self) -> &'static str {
        "the primary submodules intersect to zero and their quotient associated primes are distinct singletons"
    }
}

impl<R: Field> PartialEq for ModulePrimaryDecomposition<R> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.source_associated_primes == other.source_associated_primes
            && self.components == other.components
    }
}

impl<R: Field> Eq for ModulePrimaryDecomposition<R> {}

impl<R: Field> fmt::Debug for ModulePrimaryDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModulePrimaryDecomposition")
            .field("components", &self.components.len())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for ModulePrimaryDecomposition<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ModulePrimaryDecomposition(components={})",
            self.components.len()
        )
    }
}

fn ideal_is_subset<R: Field>(
    left: &Ideal<PolynomialRing<R>>,
    right: &Ideal<PolynomialRing<R>>,
) -> Result<bool, ModulePrimaryDecompositionError> {
    if left.ring() != right.ring() {
        return Ok(false);
    }
    for generator in left.generators() {
        if !right.contains(generator)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn localization_contraction<R>(
    source: &Submodule<PolynomialRing<R>>,
    primes: &[PolynomialPrimeCertificate<R>],
    target: &Ideal<PolynomialRing<R>>,
    order: ModuleMonomialOrder,
) -> Result<(Submodule<PolynomialRing<R>>, bool), ModulePrimaryDecompositionError>
where
    R: Field,
{
    let mut separator = source.ambient().ring().one();
    let mut has_factor = false;
    for other in primes {
        if ideal_is_subset(other.ideal(), target)? {
            continue;
        }
        let mut factor = None;
        for generator in other.ideal().generators() {
            if !target.contains(generator)? {
                factor = Some(generator);
                break;
            }
        }
        let factor = factor.ok_or_else(|| {
            ModulePrimaryDecompositionError::Invalid(
                "associated-prime inclusion disagrees with a localization separator".to_owned(),
            )
        })?;
        separator = (&separator * factor)?;
        has_factor = true;
    }
    if has_factor {
        Ok((source.saturate_by_element(&separator, order)?, true))
    } else {
        Ok((source.clone(), false))
    }
}

fn bracket_power_multiple<R: Field>(
    ambient: &FreeModule<PolynomialRing<R>>,
    prime: &Ideal<PolynomialRing<R>>,
    exponent: u64,
) -> Result<Submodule<PolynomialRing<R>>, ModulePrimaryDecompositionError> {
    let powers = prime
        .generators()
        .iter()
        .map(|generator| generator.pow(exponent))
        .collect::<Result<Vec<_>, _>>()?;
    let mut generators = Vec::with_capacity(ambient.rank().saturating_mul(powers.len()));
    for component in 0..ambient.rank() {
        let basis = ambient.basis_element(component)?;
        for power in &powers {
            generators.push(basis.scalar_mul(power)?);
        }
    }
    Submodule::from_generators(ambient.clone(), generators).map_err(Into::into)
}

fn component_maps<R: Field>(
    source: &PresentedModule<PolynomialRing<R>>,
    preimage: &Submodule<PolynomialRing<R>>,
    order: ModuleMonomialOrder,
) -> Result<PrimaryComponentMaps<R>, ModulePrimaryDecompositionError> {
    if preimage.ambient() != source.generator_module() {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "a primary preimage belongs to another generator free module".to_owned(),
        ));
    }
    let basis = preimage.groebner_basis(order.clone())?;
    let relation_columns = map_columns(source.presentation())?
        .into_iter()
        .map(|relation| {
            basis
                .lift(&relation)?
                .ok_or_else(|| {
                    ModulePrimaryDecompositionError::Invalid(
                        "a primary preimage does not contain every source relation".to_owned(),
                    )
                })
                .map(|lift| lift.coordinates().values().collect())
        })
        .collect::<Result<Vec<Vec<_>>, ModulePrimaryDecompositionError>>()?;
    let quotient = PresentedModule::cokernel(preimage.generator_map().clone());
    let relation_lift = ModuleHomomorphism::from_coordinate_columns(
        source.relation_module().clone(),
        preimage.generator_map().source().clone(),
        relation_columns,
    )?;
    let quotient_projection = PresentedModuleHomomorphism::from_lifts(
        source.clone(),
        quotient,
        ModuleHomomorphism::identity(source.generator_module().clone())?,
        relation_lift,
    )?;
    let inclusion = quotient_projection.kernel(order)?;
    Ok((inclusion, quotient_projection))
}

fn construct_component<R>(
    source: &PresentedModule<PolynomialRing<R>>,
    preimage: Submodule<PolynomialRing<R>>,
    quotient_associated_primes: ModuleAssociatedPrimeSet<R>,
    target: &Ideal<PolynomialRing<R>>,
    order: ModuleMonomialOrder,
) -> Result<Option<ModulePrimaryComponent<R>>, ModulePrimaryDecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let [associated_prime] = quotient_associated_primes.prime_certificates() else {
        return Ok(None);
    };
    if associated_prime.ideal() != target {
        return Ok(None);
    }
    let (inclusion, quotient_projection) = component_maps(source, &preimage, order)?;
    if quotient_projection.target() != quotient_associated_primes.source() {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "a primary quotient and its associated-prime evidence use different presentations"
                .to_owned(),
        ));
    }
    Ok(Some(ModulePrimaryComponent {
        preimage,
        inclusion,
        quotient_projection,
        quotient_associated_primes,
    }))
}

fn primary_candidate<R>(
    source: &PresentedModule<PolynomialRing<R>>,
    preimage: Submodule<PolynomialRing<R>>,
    target: &Ideal<PolynomialRing<R>>,
    order: ModuleMonomialOrder,
) -> Result<Option<ModulePrimaryComponent<R>>, ModulePrimaryDecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let quotient = PresentedModule::cokernel(preimage.generator_map().clone());
    let associated_primes = construct_module_associated_primes(&quotient)?;
    construct_component(source, preimage, associated_primes, target, order)
}

fn push_distinct_component<R: Field>(
    choices: &mut Vec<ModulePrimaryComponent<R>>,
    candidate: ModulePrimaryComponent<R>,
    order: ModuleMonomialOrder,
) -> Result<(), ModulePrimaryDecompositionError> {
    for existing in choices.iter() {
        if existing
            .preimage()
            .is_equal(candidate.preimage(), order.clone())?
        {
            return Ok(());
        }
    }
    choices.push(candidate);
    Ok(())
}

fn whole_submodule<R: Field>(
    ambient: &FreeModule<PolynomialRing<R>>,
) -> Result<Submodule<PolynomialRing<R>>, ModulePrimaryDecompositionError> {
    Submodule::from_generator_map(
        ambient.clone(),
        ModuleHomomorphism::identity(ambient.clone())?,
    )
    .map_err(Into::into)
}

fn intersection_of_components<R: Field>(
    components: &[ModulePrimaryComponent<R>],
    ambient: &FreeModule<PolynomialRing<R>>,
    order: ModuleMonomialOrder,
) -> Result<Submodule<PolynomialRing<R>>, ModulePrimaryDecompositionError> {
    let Some(first) = components.first() else {
        return whole_submodule(ambient);
    };
    let mut intersection = first.preimage().clone();
    for component in &components[1..] {
        intersection = intersection.intersection(component.preimage(), order.clone())?;
    }
    Ok(intersection)
}

fn components_are_irredundant<R: Field>(
    components: &[ModulePrimaryComponent<R>],
    ambient: &FreeModule<PolynomialRing<R>>,
    order: ModuleMonomialOrder,
) -> Result<bool, ModulePrimaryDecompositionError> {
    for omitted in 0..components.len() {
        let others = components
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != omitted)
            .map(|(_, component)| component.clone())
            .collect::<Vec<_>>();
        let intersection = intersection_of_components(&others, ambient, order.clone())?;
        if intersection.is_subset_of(components[omitted].preimage(), order.clone())? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn search_component_product<R: Field>(
    choices: &[Vec<ModulePrimaryComponent<R>>],
    depth: usize,
    selected: &mut Vec<ModulePrimaryComponent<R>>,
    current: Option<Submodule<PolynomialRing<R>>>,
    source_relations: &Submodule<PolynomialRing<R>>,
    order: ModuleMonomialOrder,
    attempts: &mut usize,
) -> Result<Option<Vec<ModulePrimaryComponent<R>>>, ModulePrimaryDecompositionError> {
    if depth == choices.len() {
        *attempts = attempts.saturating_add(1);
        if *attempts > MAX_PRIMARY_COMBINATIONS {
            return Err(ModulePrimaryDecompositionError::ResourceLimit {
                resource: "primary-component combination count",
                limit: MAX_PRIMARY_COMBINATIONS,
            });
        }
        let Some(intersection) = current else {
            return Ok(source_relations
                .is_equal(&whole_submodule(source_relations.ambient())?, order.clone())?
                .then(Vec::new));
        };
        return Ok((intersection.is_equal(source_relations, order.clone())?
            && components_are_irredundant(selected, source_relations.ambient(), order)?)
        .then(|| selected.clone()));
    }

    for candidate in &choices[depth] {
        let next = if let Some(current) = &current {
            current.intersection(candidate.preimage(), order.clone())?
        } else {
            candidate.preimage().clone()
        };
        selected.push(candidate.clone());
        if let Some(result) = search_component_product(
            choices,
            depth + 1,
            selected,
            Some(next),
            source_relations,
            order.clone(),
            attempts,
        )? {
            return Ok(Some(result));
        }
        selected.pop();
    }
    Ok(None)
}

fn construct_module_primary_decomposition<R>(
    source: &PresentedModule<PolynomialRing<R>>,
) -> Result<ModulePrimaryDecomposition<R>, ModulePrimaryDecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    let ring = source.generator_module().ring();
    let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
    let source_associated_primes = construct_module_associated_primes(source)?;
    let source_relations = Submodule::from_generator_map(
        source.generator_module().clone(),
        source.presentation().clone(),
    )?;
    let primes = source_associated_primes.prime_certificates();
    if primes.is_empty() {
        if !source_relations
            .is_equal(&whole_submodule(source.generator_module())?, order.clone())?
        {
            return Err(ModulePrimaryDecompositionError::Invalid(
                "a nonzero module has an empty associated-prime set".to_owned(),
            ));
        }
        return Ok(ModulePrimaryDecomposition {
            source: source.clone(),
            source_associated_primes,
            components: Vec::new(),
        });
    }

    let mut all_choices = Vec::with_capacity(primes.len());
    for (target_index, target) in primes.iter().enumerate() {
        let mut choices = Vec::new();
        let (isolated, was_localized) =
            localization_contraction(&source_relations, primes, target.ideal(), order.clone())?;
        let isolated_component = if was_localized {
            primary_candidate(source, isolated, target.ideal(), order.clone())?
        } else {
            construct_component(
                source,
                isolated,
                source_associated_primes.clone(),
                target.ideal(),
                order.clone(),
            )?
        };
        if let Some(component) = isolated_component {
            all_choices.push(vec![component]);
            if target_index + 1 == primes.len() {
                let mut attempts = 0;
                if let Some(components) = search_component_product(
                    &all_choices,
                    0,
                    &mut Vec::with_capacity(all_choices.len()),
                    None,
                    &source_relations,
                    order.clone(),
                    &mut attempts,
                )? {
                    return Ok(ModulePrimaryDecomposition {
                        source: source.clone(),
                        source_associated_primes: source_associated_primes.clone(),
                        components,
                    });
                }
            }
            continue;
        }

        let mut previous_power: Option<Submodule<PolynomialRing<R>>> = None;
        for exponent in 1..=MAX_PRIMARY_EXPONENT {
            let multiple =
                bracket_power_multiple(source.generator_module(), target.ideal(), exponent)?;
            let powered_source = source_relations.sum(&multiple)?;
            if let Some(previous) = &previous_power {
                if previous.is_equal(&powered_source, order.clone())? {
                    break;
                }
            }
            previous_power = Some(powered_source.clone());
            let powered_quotient =
                PresentedModule::cokernel(powered_source.generator_map().clone());
            let powered_associated_primes = construct_module_associated_primes(&powered_quotient)?;
            let (localized, was_localized) = localization_contraction(
                &powered_source,
                powered_associated_primes.prime_certificates(),
                target.ideal(),
                order.clone(),
            )?;
            let candidate = if was_localized {
                primary_candidate(source, localized, target.ideal(), order.clone())?
            } else {
                construct_component(
                    source,
                    localized,
                    powered_associated_primes,
                    target.ideal(),
                    order.clone(),
                )?
            };
            if let Some(component) = candidate {
                push_distinct_component(&mut choices, component, order.clone())?;
                if target_index + 1 == primes.len() {
                    all_choices.push(choices.clone());
                    let mut attempts = 0;
                    let complete = search_component_product(
                        &all_choices,
                        0,
                        &mut Vec::with_capacity(all_choices.len()),
                        None,
                        &source_relations,
                        order.clone(),
                        &mut attempts,
                    )?;
                    all_choices.pop();
                    if let Some(components) = complete {
                        return Ok(ModulePrimaryDecomposition {
                            source: source.clone(),
                            source_associated_primes: source_associated_primes.clone(),
                            components,
                        });
                    }
                }
            }
        }
        if choices.is_empty() {
            return Err(ModulePrimaryDecompositionError::Unsupported(format!(
                "no verified component was found for associated prime {} within exponent {}",
                target.ideal(),
                MAX_PRIMARY_EXPONENT
            )));
        }
        all_choices.push(choices);
    }

    let mut attempts = 0;
    let components = search_component_product(
        &all_choices,
        0,
        &mut Vec::with_capacity(all_choices.len()),
        None,
        &source_relations,
        order,
        &mut attempts,
    )?
    .ok_or_else(|| {
        ModulePrimaryDecompositionError::Unsupported(
            "verified primary candidates did not reconstruct the zero submodule".to_owned(),
        )
    })?;
    Ok(ModulePrimaryDecomposition {
        source: source.clone(),
        source_associated_primes,
        components,
    })
}

fn verify_component<R>(
    value: &ModulePrimaryComponent<R>,
) -> Result<(), ModulePrimaryDecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    verify_module_associated_primes(&value.quotient_associated_primes)?;
    let [associated_prime] = value.quotient_associated_primes.prime_certificates() else {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "a primary quotient does not have exactly one associated prime".to_owned(),
        ));
    };
    if associated_prime.ideal().ring() != value.preimage.ambient().ring() {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "a primary component and its associated prime use different contexts".to_owned(),
        ));
    }
    let source = value.quotient_projection.source();
    let order =
        ModuleMonomialOrder::TermOverPosition(source.generator_module().ring().term_order());
    let (expected_inclusion, expected_projection) =
        component_maps(source, &value.preimage, order.clone())?;
    if value.inclusion != expected_inclusion
        || value.inclusion.relation_lift() != expected_inclusion.relation_lift()
        || value.quotient_projection != expected_projection
        || value.quotient_projection.relation_lift() != expected_projection.relation_lift()
        || value.quotient_projection.target() != value.quotient_associated_primes.source()
    {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "the retained inclusion or quotient projection changed on exact replay".to_owned(),
        ));
    }
    value.inclusion.verify()?;
    value.quotient_projection.verify()?;
    if !value
        .quotient_projection
        .is_exact_at_source(&value.inclusion, order)?
    {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "the retained primary submodule is not the exact quotient kernel".to_owned(),
        ));
    }
    Ok(())
}

fn verify_decomposition<R>(
    value: &ModulePrimaryDecomposition<R>,
) -> Result<(), ModulePrimaryDecompositionError>
where
    R: MinimalPrimeCoefficientField + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    verify_module_associated_primes(&value.source_associated_primes)?;
    if value.source_associated_primes.source() != &value.source {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "the retained associated-prime evidence belongs to another source module".to_owned(),
        ));
    }
    if value.components.len() != value.source_associated_primes.prime_certificates().len() {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "the component count differs from the complete source associated-prime count"
                .to_owned(),
        ));
    }
    for (component, source_prime) in value
        .components
        .iter()
        .zip(value.source_associated_primes.prime_certificates())
    {
        verify_component(component)?;
        if component.quotient_projection.source() != &value.source
            || component.associated_prime().ideal() != source_prime.ideal()
        {
            return Err(ModulePrimaryDecompositionError::Invalid(
                "a primary component is not paired with its source associated prime".to_owned(),
            ));
        }
    }
    let order =
        ModuleMonomialOrder::TermOverPosition(value.source.generator_module().ring().term_order());
    let source_relations = Submodule::from_generator_map(
        value.source.generator_module().clone(),
        value.source.presentation().clone(),
    )?;
    let reconstructed = intersection_of_components(
        &value.components,
        value.source.generator_module(),
        order.clone(),
    )?;
    if !reconstructed.is_equal(&source_relations, order.clone())? {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "the primary preimages do not intersect to the source relation submodule".to_owned(),
        ));
    }
    if !components_are_irredundant(&value.components, value.source.generator_module(), order)? {
        return Err(ModulePrimaryDecompositionError::Invalid(
            "the retained primary decomposition is redundant".to_owned(),
        ));
    }
    Ok(())
}

impl PresentedModule<PolynomialRing<RationalField>> {
    /// Compute a verified primary decomposition of the zero submodule.
    pub fn primary_decomposition(
        &self,
    ) -> Result<ModulePrimaryDecomposition<RationalField>, ModulePrimaryDecompositionError> {
        construct_module_primary_decomposition(self)
    }
}

impl PresentedModule<PolynomialRing<FiniteField>> {
    /// Compute a verified primary decomposition of the zero submodule.
    pub fn primary_decomposition(
        &self,
    ) -> Result<ModulePrimaryDecomposition<FiniteField>, ModulePrimaryDecompositionError> {
        construct_module_primary_decomposition(self)
    }
}

macro_rules! impl_verified_primary_values {
    ($field:ty) => {
        impl ModulePrimaryComponent<$field> {
            /// Independently replay the primary-submodule evidence.
            pub fn verify(&self) -> Result<(), ModulePrimaryDecompositionError> {
                verify_component(self)
            }

            /// Deserialize and exactly reverify a primary-submodule certificate.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl ModulePrimaryDecomposition<$field> {
            /// Independently replay every component and reconstruction identity.
            pub fn verify(&self) -> Result<(), ModulePrimaryDecompositionError> {
                verify_decomposition(self)
            }

            /// Deserialize and exactly reverify a complete module decomposition.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
    };
}

impl_verified_primary_values!(RationalField);
impl_verified_primary_values!(FiniteField);

macro_rules! impl_primary_json {
    ($type:ident, $description:expr) => {
        impl<R> $type<R>
        where
            R: Field + Serialize,
            R::Element: Serialize,
            Ideal<PolynomialRing<R>>: Serialize,
        {
            /// Serialize the complete exact evidence as typed JSON.
            pub fn to_json(&self) -> Result<String, NoetherError> {
                serde_json::to_string(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }

        impl<R> SemanticType for $type<R>
        where
            R: Field + Serialize,
            R::Element: Serialize,
            Ideal<PolynomialRing<R>>: Serialize,
        {
            const TYPE_ID: &'static str = concat!("noether/homalg/", stringify!($type), "@1");

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                self.to_json().map(String::into_bytes)
            }

            fn describe() -> &'static TypeDescription {
                $description
            }
        }
    };
}

impl_primary_json!(
    ModulePrimaryComponent,
    &MODULE_PRIMARY_COMPONENT_DESCRIPTION
);
impl_primary_json!(
    ModulePrimaryDecomposition,
    &MODULE_PRIMARY_DECOMPOSITION_DESCRIPTION
);

impl<R> Serialize for ModulePrimaryComponent<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ModulePrimaryComponent", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("preimage", &self.preimage)?;
        state.serialize_field("inclusion", &self.inclusion)?;
        state.serialize_field("quotient_projection", &self.quotient_projection)?;
        state.serialize_field(
            "quotient_associated_primes",
            &self.quotient_associated_primes,
        )?;
        state.end()
    }
}

impl<R> Serialize for ModulePrimaryDecomposition<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ModulePrimaryDecomposition", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("source_associated_primes", &self.source_associated_primes)?;
        state.serialize_field("components", &self.components)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, ModuleAssociatedPrimeSet<R>: Deserialize<'de>"
    )
)]
struct ModulePrimaryComponentWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    preimage: Submodule<PolynomialRing<R>>,
    inclusion: PresentedModuleHomomorphism<PolynomialRing<R>>,
    quotient_projection: PresentedModuleHomomorphism<PolynomialRing<R>>,
    quotient_associated_primes: ModuleAssociatedPrimeSet<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, ModuleAssociatedPrimeSet<R>: Deserialize<'de>, ModulePrimaryComponent<R>: Deserialize<'de>"
    )
)]
struct ModulePrimaryDecompositionWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: PresentedModule<PolynomialRing<R>>,
    source_associated_primes: ModuleAssociatedPrimeSet<R>,
    components: Vec<ModulePrimaryComponent<R>>,
}

fn deserialize_component<'de, R, D>(deserializer: D) -> Result<ModulePrimaryComponent<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    ModuleAssociatedPrimeSet<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = ModulePrimaryComponentWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/homalg/ModulePrimaryComponent@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type noether/homalg/ModulePrimaryComponent@1, received {}",
            wire.type_id
        )));
    }
    let result = ModulePrimaryComponent {
        preimage: wire.preimage,
        inclusion: wire.inclusion,
        quotient_projection: wire.quotient_projection,
        quotient_associated_primes: wire.quotient_associated_primes,
    };
    verify_component(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

fn deserialize_decomposition<'de, R, D>(
    deserializer: D,
) -> Result<ModulePrimaryDecomposition<R>, D::Error>
where
    R: MinimalPrimeCoefficientField + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize,
    ModuleAssociatedPrimeSet<R>: Deserialize<'de>,
    ModulePrimaryComponent<R>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = ModulePrimaryDecompositionWire::<R>::deserialize(deserializer)?;
    if wire.type_id != "noether/homalg/ModulePrimaryDecomposition@1" {
        return Err(serde::de::Error::custom(format!(
            "expected type noether/homalg/ModulePrimaryDecomposition@1, received {}",
            wire.type_id
        )));
    }
    let result = ModulePrimaryDecomposition {
        source: wire.source,
        source_associated_primes: wire.source_associated_primes,
        components: wire.components,
    };
    verify_decomposition(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

macro_rules! impl_primary_deserialize {
    ($field:ty) => {
        impl<'de> Deserialize<'de> for ModulePrimaryComponent<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserialize_component(deserializer)
            }
        }

        impl<'de> Deserialize<'de> for ModulePrimaryDecomposition<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserialize_decomposition(deserializer)
            }
        }
    };
}

impl_primary_deserialize!(RationalField);
impl_primary_deserialize!(FiniteField);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::linalg::Matrix;
    use crate::Q;

    fn embedded_module() -> PresentedModule<PolynomialRing<RationalField>> {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let linear = (&variables[0] + &variables[1]).unwrap();
        PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                FreeModule::new(ring.clone(), 3),
                FreeModule::new(ring.clone(), 2),
                Matrix::from_rows(
                    ring.clone(),
                    [
                        [linear.clone(), ring.zero(), ring.zero()],
                        [ring.zero(), linear, variables[2].clone()],
                    ],
                )
                .unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn noncyclic_module_primary_components_reconstruct_zero() {
        let source = embedded_module();
        let ring = source.generator_module().ring();
        let variables = ring.generators().unwrap();
        let linear = (&variables[0] + &variables[1]).unwrap();

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 2);
        let minimal = ring.ideal([linear.clone()]).unwrap();
        let embedded = ring.ideal([linear, variables[2].clone()]).unwrap();
        assert_eq!(
            decomposition
                .associated_primes()
                .into_iter()
                .map(PolynomialPrimeCertificate::ideal)
                .collect::<Vec<_>>(),
            vec![&minimal, &embedded]
        );
        for component in decomposition.components() {
            component.verify().unwrap();
            assert_eq!(component.inclusion().target(), &source);
            assert_eq!(component.quotient_projection().source(), &source);
            assert_eq!(component.quotient_associated_primes().primes().len(), 1);
        }

        let round_trip = ModulePrimaryDecomposition::<RationalField>::from_json(
            &decomposition.to_json().unwrap(),
        )
        .unwrap();
        assert_eq!(round_trip, decomposition);
        let mut payload = serde_json::to_value(&decomposition).unwrap();
        payload["components"].as_array_mut().unwrap().pop();
        assert!(
            serde_json::from_value::<ModulePrimaryDecomposition<RationalField>>(payload).is_err()
        );
    }

    #[test]
    fn zero_module_has_the_empty_primary_decomposition() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let free = FreeModule::new(ring, 1);
        let zero = PresentedModule::cokernel(ModuleHomomorphism::identity(free).unwrap());

        let decomposition = zero.primary_decomposition().unwrap();

        assert!(decomposition.components().is_empty());
        decomposition.verify().unwrap();
    }

    #[test]
    fn finite_field_module_primary_decomposition_uses_the_same_exact_contract() {
        let field = FiniteField::prime(65_521_u64).unwrap();
        let ring = PolynomialRing::new(field, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let linear = (&variables[0] + &variables[1]).unwrap();
        let source = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                FreeModule::new(ring.clone(), 3),
                FreeModule::new(ring.clone(), 2),
                Matrix::from_rows(
                    ring.clone(),
                    [
                        [linear.clone(), ring.zero(), ring.zero()],
                        [ring.zero(), linear, variables[2].clone()],
                    ],
                )
                .unwrap(),
            )
            .unwrap(),
        );

        let decomposition = source.primary_decomposition().unwrap();

        assert_eq!(decomposition.components().len(), 2);
        decomposition.verify().unwrap();
        let round_trip =
            ModulePrimaryDecomposition::<FiniteField>::from_json(&decomposition.to_json().unwrap())
                .unwrap();
        assert_eq!(round_trip, decomposition);
    }

    #[test]
    fn localization_separates_incomparable_embedded_module_primes() {
        let ring = PolynomialRing::new(Q, ["x", "y", "z"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let linear = (&variables[0] + &variables[1]).unwrap();
        let zero = ring.zero();
        let source = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                FreeModule::new(ring.clone(), 5),
                FreeModule::new(ring.clone(), 3),
                Matrix::from_rows(
                    ring.clone(),
                    [
                        [
                            linear.clone(),
                            zero.clone(),
                            zero.clone(),
                            zero.clone(),
                            zero.clone(),
                        ],
                        [
                            zero.clone(),
                            linear.clone(),
                            variables[1].clone(),
                            zero.clone(),
                            zero.clone(),
                        ],
                        [
                            zero.clone(),
                            zero.clone(),
                            zero,
                            linear.clone(),
                            variables[2].clone(),
                        ],
                    ],
                )
                .unwrap(),
            )
            .unwrap(),
        );

        let decomposition = source.primary_decomposition().unwrap();

        decomposition.verify().unwrap();
        assert_eq!(decomposition.components().len(), 3);
        let expected = [
            ring.ideal([linear.clone()]).unwrap(),
            ring.ideal([linear.clone(), variables[1].clone()]).unwrap(),
            ring.ideal([linear, variables[2].clone()]).unwrap(),
        ];
        let actual = decomposition
            .associated_primes()
            .into_iter()
            .map(PolynomialPrimeCertificate::ideal)
            .collect::<Vec<_>>();
        assert!(expected.iter().all(|prime| actual.contains(&prime)));
    }
}
