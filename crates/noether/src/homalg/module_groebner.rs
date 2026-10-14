// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact module Groebner bases, normal forms, lifts, and first syzygies.

use core::cmp::Ordering;
use core::fmt;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use module_core::ModuleTerm as PackedModuleTerm;
use poly_core::{MonomialArena, MonomialOrder as PackedMonomialOrder};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{
    exponent_to_kernel, from_kernel_basis, from_prime_kernel_polynomial,
    from_rational_kernel_terms, kernel_groebner_basis_with_transform, to_kernel_order,
    to_kernel_polynomial, to_rational_kernel_polynomial, Degree, FreeAbelianGrading,
    GroebnerError as PolynomialGroebnerError, HilbertSeries, NoetherFieldLane, Polynomial,
    PolynomialError, PolynomialRing, QuotientElement, QuotientRing, RationalKernelLane,
};
use crate::linalg::{Matrix, SparseMatrix, Vector};
use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{
    ContextMismatchError, Field, Integer, NoetherError, RationalField, Ring, SemanticType,
    TypeDescription,
};

use super::{
    FreeModule, ModuleElement, ModuleError, ModuleHomomorphism, ModuleMonomialOrder,
    PresentedModule, PresentedModuleHomology, PresentedModuleHomomorphism, Submodule,
};

mod integer;

pub use integer::{StrongModuleGroebnerBasis, StrongModuleReduction, StrongModuleTerm};

const MAX_BASIS_SIZE: usize = 128;
const MAX_PENDING_PAIRS: usize = 16_384;
const MAX_REDUCTION_STEPS: usize = 1_000_000;
const MAX_MODULE_HILBERT_TERMS: usize = 100_000;
const MAX_STANDARD_MODULE_HILBERT_DEGREE: usize = 4_096;
const MAX_SUBMODULE_SATURATION_STEPS: usize = 128;
const MIN_HILBERT_GUIDANCE_LATER_PAIRS: usize = 16;

type PolynomialModule<C> = ModuleElement<PolynomialRing<C>>;
type ModuleNormalForm<C> = (PolynomialModule<C>, Vec<Polynomial<C>>);
type ModuleSPair<C> = (PolynomialModule<C>, Polynomial<C>, Polynomial<C>);
type MonicTrackedPair<C> = (PolynomialModule<C>, PolynomialModule<C>);

struct SchreyerSyzygyData<C: Field> {
    relations: Vec<PolynomialModule<C>>,
    reduced_groebner_basis: bool,
}

static MODULE_GROEBNER_BASIS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModuleGroebnerBasis@1",
    rust_path: "noether::homalg::ModuleGroebnerBasis",
    python_path: Some("noether.homalg.ModuleGroebnerBasis"),
};

static MODULE_REDUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModuleReduction@1",
    rust_path: "noether::homalg::ModuleReduction",
    python_path: Some("noether.homalg.ModuleReduction"),
};

static MODULE_MONOMIAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ModuleMonomial@1",
    rust_path: "noether::homalg::ModuleMonomial",
    python_path: Some("noether.homalg.ModuleMonomial"),
};

/// A monomial times one ordered basis vector of a free module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleMonomial {
    exponent: Vec<Integer>,
    component: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModuleMonomialWire {
    #[serde(rename = "type")]
    type_id: String,
    exponent: Vec<Integer>,
    component: usize,
}

impl ModuleMonomial {
    /// Construct a module monomial from an exact exponent vector and a
    /// zero-based basis component.
    fn new(exponent: Vec<Integer>, component: usize) -> Self {
        Self {
            exponent,
            component,
        }
    }

    /// Polynomial exponent vector.
    pub fn exponent(&self) -> &[Integer] {
        &self.exponent
    }

    /// Zero-based free-module basis component.
    pub const fn component(&self) -> usize {
        self.component
    }

    /// Serialize this exact exponent vector and basis component.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize an exact module monomial.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl SemanticType for ModuleMonomial {
    const TYPE_ID: &'static str = "noether/homalg/ModuleMonomial@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &MODULE_MONOMIAL_DESCRIPTION
    }
}

impl Serialize for ModuleMonomial {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ModuleMonomialWire {
            type_id: Self::TYPE_ID.to_owned(),
            exponent: self.exponent.clone(),
            component: self.component,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ModuleMonomial {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModuleMonomialWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        if wire.exponent.iter().any(|power| power.sign() < 0) {
            return Err(serde::de::Error::custom(
                "module monomial exponents must be nonnegative",
            ));
        }
        Ok(Self::new(wire.exponent, wire.component))
    }
}

/// Failure to compute or verify an exact module Groebner result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleGroebnerError {
    /// A based-module operation failed.
    Module(ModuleError),
    /// Polynomial construction or arithmetic failed.
    Polynomial(PolynomialError),
    /// Polynomial arithmetic mixed unequal contexts.
    ContextMismatch(ContextMismatchError),
    /// The supplied module order does not extend the ambient polynomial
    /// ring's exact monomial order.
    OrderMismatch,
    /// A Schreyer frame has the wrong number of entries for its source
    /// module.
    SchreyerFrameRank {
        /// Source module rank.
        rank: usize,
        /// Number of retained frame leaders.
        frame_len: usize,
    },
    /// A nested Schreyer leader names no component in its target frame.
    SchreyerComponent {
        /// Rejected target component.
        component: usize,
        /// Target frame length.
        target_rank: usize,
    },
    /// Inverting a nonzero coefficient contradicted the declared field laws.
    CoefficientInvariant,
    /// A native correctness-floor computation reached an explicit bound.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed limit.
        limit: usize,
    },
    /// Supplied or computed evidence failed an exact identity.
    InvalidEvidence(&'static str),
}

impl fmt::Display for ModuleGroebnerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::OrderMismatch => formatter
                .write_str("the module order does not extend the polynomial ring's monomial order"),
            Self::SchreyerFrameRank { rank, frame_len } => write!(
                formatter,
                "Schreyer frame has {frame_len} leaders for a source module of rank {rank}"
            ),
            Self::SchreyerComponent {
                component,
                target_rank,
            } => write!(
                formatter,
                "Schreyer leader component {component} is outside target frame rank {target_rank}"
            ),
            Self::CoefficientInvariant => formatter.write_str(
                "coefficient inversion failed despite a nonzero value in a declared field",
            ),
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "native module Groebner computation exceeded its {resource} limit of {limit}"
            ),
            Self::InvalidEvidence(reason) => {
                write!(formatter, "invalid module Groebner evidence: {reason}")
            }
        }
    }
}

impl std::error::Error for ModuleGroebnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Module(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModuleError> for ModuleGroebnerError {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}

impl From<PolynomialError> for ModuleGroebnerError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<ContextMismatchError> for ModuleGroebnerError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

/// A complete reduced Groebner basis of one generated submodule over a field.
///
/// The retained transformation map has source the ordered basis-generator
/// module and target the original generator module. Consequently
/// `input.generator_map() * transformation == basis.generator_map()` is an
/// exact certificate that every basis generator belongs to the input.
#[derive(Clone)]
pub struct ModuleGroebnerBasis<R: Ring> {
    input: Submodule<R>,
    order: ModuleMonomialOrder,
    basis: Submodule<R>,
    transformation: ModuleHomomorphism<R>,
    leading_monomials: Vec<ModuleMonomial>,
}

/// Exact normal-form and lift evidence against a module Groebner basis.
///
/// It retains both identities
/// `value = basis_map(quotient) + remainder` and
/// `original_combination = transformation(quotient)`. Therefore a zero
/// remainder gives coordinates in the original input generators directly.
#[derive(Clone)]
pub struct ModuleReduction<R: Ring> {
    basis: ModuleGroebnerBasis<R>,
    value: ModuleElement<R>,
    quotient: ModuleElement<R>,
    original_combination: ModuleElement<R>,
    remainder: ModuleElement<R>,
}

impl<R: Ring> ModuleGroebnerBasis<R> {
    /// Original generated submodule.
    pub fn input(&self) -> &Submodule<R> {
        &self.input
    }

    /// Exact module-monomial order.
    pub fn order(&self) -> &ModuleMonomialOrder {
        &self.order
    }

    /// Canonically ordered monic reduced basis as a generator map.
    pub fn basis(&self) -> &Submodule<R> {
        &self.basis
    }

    /// Map expressing every basis generator in the original generators.
    pub fn transformation(&self) -> &ModuleHomomorphism<R> {
        &self.transformation
    }

    /// Ordered leading module monomials of the reduced basis.
    pub fn leading_monomials(&self) -> &[ModuleMonomial] {
        &self.leading_monomials
    }

    /// This value is complete. Resource-bounded computations fail instead of
    /// returning a partial result that could be mistaken for complete.
    pub const fn is_complete(&self) -> bool {
        true
    }

    /// Serialize the exact input, order, basis, transformation, and leaders.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> ModuleGroebnerBasis<PolynomialRing<C>> {
    /// Exact Schreyer order induced on the ordered basis-generator module.
    pub fn schreyer_order(&self) -> ModuleMonomialOrder {
        ModuleMonomialOrder::schreyer(self.order.clone(), self.leading_monomials.clone())
    }

    /// Monomial submodule generated by the leading module monomials.
    pub fn initial_submodule(&self) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
        let ring = self.input.ambient().ring();
        let generators = self
            .leading_monomials
            .iter()
            .map(|leading| {
                let mut coordinates = vec![ring.zero(); self.input.ambient().rank()];
                coordinates[leading.component] =
                    ring.from_terms([(leading.exponent.clone(), ring.coefficient_ring().one())])?;
                element_from_coordinates(self.input.ambient().clone(), coordinates)
                    .map_err(ModuleGroebnerError::from)
            })
            .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
        Submodule::from_generators(self.input.ambient().clone(), generators).map_err(Into::into)
    }

    /// Multigraded Hilbert numerator of the quotient of the ambient free
    /// module by the input submodule.
    ///
    /// The ambient module must carry explicit algebraic shifts, with the
    /// convention that `R(a)` contributes `T^-a`. The computed reduced basis
    /// must be homogeneous for the polynomial-ring grading. These checks keep
    /// an arbitrary filtered module from being mislabeled as multigraded.
    pub fn multigraded_hilbert_numerator(
        &self,
    ) -> Result<Vec<(Degree, Integer)>, ModuleGroebnerError> {
        let ambient = self.input.ambient();
        let ring = ambient.ring();
        let shifts = ambient
            .degree_shifts()
            .ok_or_else(|| ModuleError::GradingMismatch {
                operation: "module Hilbert numerator",
            })?;
        if ambient.grading_rank() != Some(ring.grading().rank()) {
            return Err(ModuleError::GradingMismatch {
                operation: "module Hilbert numerator",
            }
            .into());
        }
        for generator in map_columns(self.basis.generator_map())? {
            homogeneous_module_degree(&generator, ring.grading(), shifts)?;
        }

        module_hilbert_numerator(ambient, &self.leading_monomials, shifts)
    }

    /// Exact Hilbert series of the ambient free-module quotient by the input.
    pub fn hilbert_series(&self) -> Result<HilbertSeries, ModuleGroebnerError> {
        HilbertSeries::new(
            self.input.ambient().ring().grading().clone(),
            self.multigraded_hilbert_numerator()?,
        )
        .map_err(|_| ModuleGroebnerError::InvalidEvidence("the Hilbert series is not canonical"))
    }

    /// Reduce an ambient element and retain quotient and original-generator
    /// coordinates.
    pub fn reduce(
        &self,
        value: &ModuleElement<PolynomialRing<C>>,
    ) -> Result<ModuleReduction<PolynomialRing<C>>, ModuleGroebnerError> {
        if value.module() != self.input.ambient() {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the reduced value belongs to another ambient module",
            ));
        }
        let generators = map_columns(self.basis.generator_map())?;
        let mut steps = 0;
        let (remainder, quotients) = normal_form(value, &generators, &self.order, &mut steps)?;
        let quotient =
            element_from_coordinates(self.basis.generator_map().source().clone(), quotients)?;
        let original_combination = self.transformation.apply(&quotient)?;
        let result = ModuleReduction {
            basis: self.clone(),
            value: value.clone(),
            quotient,
            original_combination,
            remainder,
        };
        result.verify()?;
        Ok(result)
    }

    /// Canonical normal form of an ambient element.
    pub fn normal_form(
        &self,
        value: &ModuleElement<PolynomialRing<C>>,
    ) -> Result<ModuleElement<PolynomialRing<C>>, ModuleGroebnerError> {
        self.normal_form_without_witness(value)
    }

    /// Whether an ambient element belongs to the input submodule.
    pub fn contains(
        &self,
        value: &ModuleElement<PolynomialRing<C>>,
    ) -> Result<bool, ModuleGroebnerError> {
        self.contains_all(std::slice::from_ref(value))
    }

    pub(crate) fn contains_all(
        &self,
        values: &[ModuleElement<PolynomialRing<C>>],
    ) -> Result<bool, ModuleGroebnerError> {
        if values
            .iter()
            .any(|value| value.module() != self.input.ambient())
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "a tested value belongs to another ambient module",
            ));
        }
        let generators = map_columns(self.basis.generator_map())?;
        match kernel_module_membership(&generators, values, self.input.ambient(), &self.order) {
            Ok(Some(results)) => return Ok(results.into_iter().all(|member| member)),
            Ok(None) | Err(ModuleGroebnerError::InvalidEvidence(_)) => {}
            Err(error) => return Err(error),
        }
        for value in values {
            let mut steps = 0;
            let (remainder, _) = normal_form(value, &generators, &self.order, &mut steps)?;
            if !is_zero(&remainder) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn normal_form_without_witness(
        &self,
        value: &ModuleElement<PolynomialRing<C>>,
    ) -> Result<ModuleElement<PolynomialRing<C>>, ModuleGroebnerError> {
        if value.module() != self.input.ambient() {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the reduced value belongs to another ambient module",
            ));
        }
        let generators = map_columns(self.basis.generator_map())?;
        let mut steps = 0;
        let (remainder, _) = normal_form(value, &generators, &self.order, &mut steps)?;
        Ok(remainder)
    }

    /// Original-generator coordinates when `value` belongs to the submodule.
    pub fn lift(
        &self,
        value: &ModuleElement<PolynomialRing<C>>,
    ) -> Result<Option<ModuleElement<PolynomialRing<C>>>, ModuleGroebnerError> {
        let reduction = self.reduce(value)?;
        Ok(reduction
            .is_member()
            .then(|| reduction.original_combination.clone()))
    }

    /// Compute generators for all relations among the original input
    /// generators using Schreyer's critical-pair theorem and the retained
    /// change map.
    pub fn syzygies(&self) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
        if let Some(relations) = primitive_rank_two_three_kernel(self.input.generator_map())? {
            return Ok(relations);
        }
        if let Some(relations) = self.packed_binary_original_syzygies()? {
            return Ok(relations);
        }
        if let Some(relations) = self.packed_prime_original_syzygies()? {
            return Ok(relations);
        }
        let original_source = self.input.generator_map().source();
        let basis_syzygies = self.schreyer_syzygies()?;
        let transformed_syzygies = self
            .transformation
            .compose(basis_syzygies.generator_map())?;
        let mut correction_relations = Vec::new();

        // If Q expresses each original generator through B and T expresses B
        // through the originals, every column of I-TQ is an original syzygy.
        // A sparse monomial matrix with unit entries is already an explicit
        // basis isomorphism, so I-TQ is identically zero and none of those
        // reductions need to be repeated.
        if !is_sparse_unit_monomial_isomorphism(&self.transformation) {
            let input_generators = map_columns(self.input.generator_map())?;
            let basis_generators = map_columns(self.basis.generator_map())?;
            let mut steps = 0;
            for (index, generator) in input_generators.iter().enumerate() {
                let (remainder, quotients) =
                    normal_form(generator, &basis_generators, &self.order, &mut steps)?;
                if !is_zero(&remainder) {
                    return Err(ModuleGroebnerError::InvalidEvidence(
                        "an original generator does not reduce through its computed basis",
                    ));
                }
                let quotient = element_from_coordinates(
                    self.basis.generator_map().source().clone(),
                    quotients,
                )?;
                let reconstructed = self.transformation.apply(&quotient)?;
                let relation = original_source.basis_element(index)?.sub(&reconstructed)?;
                if !is_zero(&relation) {
                    correction_relations.push(relation);
                }
            }
        }

        let syzygies = if correction_relations.is_empty()
            && sparse_map_has_no_zero_columns(&transformed_syzygies)
        {
            Submodule::from_generator_map(original_source.clone(), transformed_syzygies)?
        } else {
            let mut relations = map_columns(&transformed_syzygies)?
                .into_iter()
                .filter(|relation| !is_zero(relation))
                .collect::<Vec<_>>();
            relations.append(&mut correction_relations);
            Submodule::from_generators(original_source.clone(), relations)?
        };
        #[cfg(debug_assertions)]
        for relation in map_columns(syzygies.generator_map())? {
            if !is_zero(&self.input.generator_map().apply(&relation)?) {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "a computed syzygy does not map to zero",
                ));
            }
        }
        Ok(syzygies)
    }

    /// Schreyer generators for the kernel of the canonical basis map.
    ///
    /// Unlike [`Self::syzygies`], these relations stay in the ordered
    /// canonical basis frame. Resolution construction uses that frame so the
    /// induced Schreyer order can be retained without translating through a
    /// generally rectangular change map.
    pub(crate) fn schreyer_syzygies(
        &self,
    ) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
        if let Some(relations) = self.packed_binary_schreyer_submodule()? {
            return Ok(relations);
        }
        if let Some(relations) = self.packed_prime_schreyer_submodule()? {
            return Ok(relations);
        }
        let data = self.schreyer_syzygy_data()?;
        Submodule::from_generators(self.basis.generator_map().source().clone(), data.relations)
            .map_err(Into::into)
    }

    fn packed_binary_schreyer_submodule(
        &self,
    ) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
        let ring = self.input.ambient().ring();
        let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>()
        else {
            return Ok(None);
        };
        if field.characteristic() != &Integer::from(2) {
            return Ok(None);
        }
        if !has_component_compatible_pair(&self.leading_monomials) {
            return Submodule::from_generators(
                self.basis.generator_map().source().clone(),
                Vec::new(),
            )
            .map(Some)
            .map_err(Into::into);
        }
        let Some(kernel_order) = to_kernel_module_order(&self.order) else {
            return Ok(None);
        };
        let basis = map_columns(self.basis.generator_map())?;
        let source = self.basis.generator_map().source();
        let hilbert_target = should_use_schreyer_hilbert_guidance(&self.leading_monomials)
            .then(|| standard_schreyer_hilbert_target(self))
            .flatten();
        let relations = if field.degree() == 1 {
            let Some(output) = specialized_f2_schreyer_output(
                &basis,
                ring,
                &kernel_order,
                hilbert_target.as_ref(),
            )?
            else {
                return Ok(None);
            };
            project_specialized_schreyer_relations(
                output,
                source,
                &groebner_kernel::BinaryField,
                |coefficient| Ok(field.element(u64::from(*coefficient))),
            )?
        } else {
            let Some(binary_field) = binary_extension_kernel_field(field) else {
                return Ok(None);
            };
            let Some(output) = specialized_binary_extension_schreyer_output(
                &basis,
                ring,
                &kernel_order,
                &binary_field,
                hilbert_target.as_ref(),
            )?
            else {
                return Ok(None);
            };
            project_specialized_schreyer_relations(output, source, &binary_field, |coefficient| {
                decode_binary_extension_coefficient(field, *coefficient)
            })?
        };
        Submodule::from_generators(source.clone(), relations)
            .map(Some)
            .map_err(Into::into)
    }

    fn packed_binary_original_syzygies(
        &self,
    ) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
        let ring = self.input.ambient().ring();
        let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>()
        else {
            return Ok(None);
        };
        if field.characteristic() != &Integer::from(2) {
            return Ok(None);
        }
        let Some(kernel_order) = to_kernel_module_order(&self.order) else {
            return Ok(None);
        };
        let basis = map_columns(self.basis.generator_map())?;
        let transformations = map_columns(&self.transformation)?;
        let values = map_columns(self.input.generator_map())?;
        let options = groebner_kernel::GroebnerOptions::default();
        if field.degree() == 1 {
            let kernel_field = groebner_kernel::BinaryField;
            let encode = |coefficient: &FiniteFieldElement| Some(u8::from(!coefficient.is_zero()));
            let Some(basis) = to_specialized_kernel_module_elements(
                &basis,
                ring,
                &kernel_order,
                &kernel_field,
                encode,
            ) else {
                return Ok(None);
            };
            let Some(transformations) = to_specialized_kernel_module_elements(
                &transformations,
                ring,
                &kernel_order,
                &kernel_field,
                encode,
            ) else {
                return Ok(None);
            };
            let Some(values) = to_specialized_kernel_module_elements(
                &values,
                ring,
                &kernel_order,
                &kernel_field,
                encode,
            ) else {
                return Ok(None);
            };
            let output = match groebner_kernel::module_relations_from_groebner_data_f2(
                &basis,
                &transformations,
                &values,
                &kernel_order,
                &options,
            ) {
                Ok(output) => output,
                Err(groebner_kernel::GroebnerError::ContextMismatch)
                | Err(groebner_kernel::GroebnerError::ExponentOverflow)
                | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
                Err(error) => return Err(map_kernel_module_error(error)),
            };
            return specialized_module_relations_submodule(
                output.generators,
                self.input.generator_map().source(),
                ring,
                &kernel_field,
                |coefficient| Ok(field.element(u64::from(*coefficient))),
            )
            .map(Some);
        }

        let Some(kernel_field) = binary_extension_kernel_field(field) else {
            return Ok(None);
        };
        let Some(basis) = to_specialized_kernel_module_elements(
            &basis,
            ring,
            &kernel_order,
            &kernel_field,
            encode_binary_extension_coefficient,
        ) else {
            return Ok(None);
        };
        let Some(transformations) = to_specialized_kernel_module_elements(
            &transformations,
            ring,
            &kernel_order,
            &kernel_field,
            encode_binary_extension_coefficient,
        ) else {
            return Ok(None);
        };
        let Some(values) = to_specialized_kernel_module_elements(
            &values,
            ring,
            &kernel_order,
            &kernel_field,
            encode_binary_extension_coefficient,
        ) else {
            return Ok(None);
        };
        let output = match groebner_kernel::module_relations_from_groebner_data_binary_extension(
            &kernel_field,
            &basis,
            &transformations,
            &values,
            &kernel_order,
            &options,
        ) {
            Ok(output) => output,
            Err(groebner_kernel::GroebnerError::ContextMismatch)
            | Err(groebner_kernel::GroebnerError::ExponentOverflow)
            | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
            Err(error) => return Err(map_kernel_module_error(error)),
        };
        specialized_module_relations_submodule(
            output.generators,
            self.input.generator_map().source(),
            ring,
            &kernel_field,
            |coefficient| decode_binary_extension_coefficient(field, *coefficient),
        )
        .map(Some)
    }

    fn packed_prime_schreyer_submodule(
        &self,
    ) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
        if !has_component_compatible_pair(&self.leading_monomials) {
            return Submodule::from_generators(
                self.basis.generator_map().source().clone(),
                Vec::new(),
            )
            .map(Some)
            .map_err(Into::into);
        }
        let ring = self.input.ambient().ring();
        let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>()
        else {
            return Ok(None);
        };
        if field.degree() != 1 {
            return Ok(None);
        }
        let Some(characteristic) = field
            .characteristic()
            .to_u64_checked()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value < (1_u32 << 31))
        else {
            return Ok(None);
        };
        let Some(kernel_order) = to_kernel_module_order(&self.order) else {
            return Ok(None);
        };
        let basis = map_columns(self.basis.generator_map())?;
        let hilbert_target = should_use_schreyer_hilbert_guidance(&self.leading_monomials)
            .then(|| standard_schreyer_hilbert_target(self))
            .flatten();
        specialized_prime_schreyer_submodule(
            &basis,
            self.basis.generator_map().source(),
            &kernel_order,
            field,
            characteristic,
            hilbert_target.as_ref(),
        )
    }

    fn packed_prime_original_syzygies(
        &self,
    ) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
        let ring = self.input.ambient().ring();
        let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>()
        else {
            return Ok(None);
        };
        if field.degree() != 1 {
            return Ok(None);
        }
        let Some(characteristic) = field
            .characteristic()
            .to_u64_checked()
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value < (1_u32 << 31))
        else {
            return Ok(None);
        };
        let Some(kernel_order) = to_kernel_module_order(&self.order) else {
            return Ok(None);
        };
        let prime = groebner_kernel::PrimeField32::new(characteristic)
            .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
        let basis = map_columns(self.basis.generator_map())?;
        let transformations = map_columns(&self.transformation)?;
        let values = map_columns(self.input.generator_map())?;
        let Some(basis) = to_prime_kernel_module_elements(&basis, ring, &kernel_order, &prime)
        else {
            return Ok(None);
        };
        let Some(transformations) =
            to_prime_kernel_module_elements(&transformations, ring, &kernel_order, &prime)
        else {
            return Ok(None);
        };
        let Some(values) = to_prime_kernel_module_elements(&values, ring, &kernel_order, &prime)
        else {
            return Ok(None);
        };
        let output = match groebner_kernel::module_relations_from_groebner_data_prime(
            &prime,
            &basis,
            &transformations,
            &values,
            &kernel_order,
            &groebner_kernel::GroebnerOptions::default(),
        ) {
            Ok(output) => output,
            Err(groebner_kernel::GroebnerError::ContextMismatch)
            | Err(groebner_kernel::GroebnerError::ExponentOverflow)
            | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
            Err(error) => return Err(map_kernel_module_error(error)),
        };
        prime_module_relations_submodule(
            output.generators,
            self.input.generator_map().source(),
            ring,
            &prime,
            field,
        )
        .map(Some)
    }

    pub(crate) fn next_schreyer_basis(
        &self,
    ) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
        let data = self.schreyer_syzygy_data()?;
        if data.relations.is_empty() {
            return Ok(None);
        }
        let order = self.schreyer_order();
        if data.reduced_groebner_basis {
            // The packed kernel has already certified this exact ordered set
            // as reduced.  Build its canonical graded module once and retain
            // the resulting identity change map for the next frame.
            module_basis_from_reduced_generators(
                self.basis.generator_map().source(),
                data.relations,
                order,
            )
            .map(Some)
        } else {
            Submodule::from_generators(self.basis.generator_map().source().clone(), data.relations)?
                .groebner_basis(order)
                .map(Some)
        }
    }

    fn schreyer_syzygy_data(&self) -> Result<SchreyerSyzygyData<C>, ModuleGroebnerError> {
        if !has_component_compatible_pair(&self.leading_monomials) {
            return Ok(SchreyerSyzygyData {
                relations: Vec::new(),
                reduced_groebner_basis: true,
            });
        }
        let ring = self.input.ambient().ring();
        let basis_generators = map_columns(self.basis.generator_map())?;
        let basis_source = self.basis.generator_map().source();
        let hilbert_target = should_use_schreyer_hilbert_guidance(&self.leading_monomials)
            .then(|| standard_schreyer_hilbert_target(self))
            .flatten();
        match kernel_schreyer_relations(
            &basis_generators,
            basis_source,
            &self.order,
            hilbert_target.as_ref(),
        ) {
            Ok(Some(relations)) => {
                // The packed kernel checks every relation against the input
                // basis both before and after interreduction. Projection is a
                // lossless in-process representation conversion; repeat the
                // multiplication only in debug builds so release resolution
                // construction does not prove the same identity twice.
                #[cfg(debug_assertions)]
                for relation in &relations {
                    if !is_zero(&self.basis.generator_map().apply(relation)?) {
                        return Err(ModuleGroebnerError::InvalidEvidence(
                            "a packed Schreyer relation does not map to zero",
                        ));
                    }
                }
                return Ok(SchreyerSyzygyData {
                    relations,
                    reduced_groebner_basis: true,
                });
            }
            Ok(None) | Err(ModuleGroebnerError::InvalidEvidence(_)) => {}
            Err(error) => return Err(error),
        }
        let mut relations = Vec::new();
        let mut steps = 0;

        for (left, right) in ordered_schreyer_pairs(&self.leading_monomials)? {
            let Some((s, left_factor, right_factor)) = s_element(
                &basis_generators[left],
                &basis_generators[right],
                &self.order,
            )?
            else {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "the Schreyer scheduler paired unequal target components",
                ));
            };
            let (remainder, quotients) =
                normal_form(&s, &basis_generators, &self.order, &mut steps)?;
            if !is_zero(&remainder) {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "a retained basis critical pair has nonzero remainder",
                ));
            }
            let mut coordinates = quotients
                .into_iter()
                .map(|value| ring.neg(&value))
                .collect::<Result<Vec<_>, _>>()?;
            coordinates[left] = ring.add(&coordinates[left], &left_factor)?;
            coordinates[right] = ring.sub(&coordinates[right], &right_factor)?;
            let relation = element_from_coordinates(basis_source.clone(), coordinates)?;
            if !is_zero(&relation) {
                relations.push(relation);
            }
        }

        for relation in &relations {
            if !is_zero(&self.basis.generator_map().apply(relation)?) {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "a computed Schreyer relation does not map to zero",
                ));
            }
        }
        Ok(SchreyerSyzygyData {
            relations,
            reduced_groebner_basis: false,
        })
    }

    /// Recheck normalization, change-map identities, input equivalence, and
    /// Buchberger's criterion without trusting construction metadata.
    pub fn verify(&self) -> Result<(), ModuleGroebnerError> {
        validate_order(self.input.ambient(), &self.order)?;
        if self.basis.ambient() != self.input.ambient()
            || self.transformation.source() != self.basis.generator_map().source()
            || self.transformation.target() != self.input.generator_map().source()
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "basis and transformation modules are incompatible",
            ));
        }
        let reconstructed = self.input.generator_map().compose(&self.transformation)?;
        if &reconstructed != self.basis.generator_map() {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the transformation does not reconstruct the basis map",
            ));
        }
        let generators = map_columns(self.basis.generator_map())?;
        let leaders = generators
            .iter()
            .map(|generator| {
                leading_term(generator, &self.order)
                    .map(|(monomial, _)| monomial)
                    .ok_or(ModuleGroebnerError::InvalidEvidence(
                        "a module Groebner basis contains zero",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if leaders != self.leading_monomials {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the retained leading module monomials are stale",
            ));
        }
        if leaders
            .windows(2)
            .any(|pair| compare_module_monomials(&pair[0], &pair[1], &self.order) == Ordering::Less)
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the basis is not in canonical leading-term order",
            ));
        }
        for generator in &generators {
            let (_, coefficient) = leading_term(generator, &self.order).ok_or(
                ModuleGroebnerError::InvalidEvidence("a module Groebner basis contains zero"),
            )?;
            if coefficient != ring_of(generator).coefficient_ring().one() {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "a module Groebner basis generator is not monic",
                ));
            }
        }
        verify_reduced(&generators, &leaders)?;
        let mut steps = 0;
        for left in 0..generators.len() {
            for right in (left + 1)..generators.len() {
                let Some((s, _, _)) =
                    s_element(&generators[left], &generators[right], &self.order)?
                else {
                    continue;
                };
                let (remainder, _) = normal_form(&s, &generators, &self.order, &mut steps)?;
                if !is_zero(&remainder) {
                    return Err(ModuleGroebnerError::InvalidEvidence(
                        "Buchberger's criterion failed",
                    ));
                }
            }
        }
        for generator in map_columns(self.input.generator_map())? {
            let (remainder, _) = normal_form(&generator, &generators, &self.order, &mut steps)?;
            if !is_zero(&remainder) {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "the computed basis does not generate the input submodule",
                ));
            }
        }
        Ok(())
    }

    /// Deserialize and independently reverify a complete result.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

fn has_component_compatible_pair(leaders: &[ModuleMonomial]) -> bool {
    leaders.iter().enumerate().any(|(left, leader)| {
        leaders[(left + 1)..]
            .iter()
            .any(|other| leader.component == other.component)
    })
}

impl<R: Ring> ModuleReduction<R> {
    /// Groebner basis used for this reduction.
    pub fn basis(&self) -> &ModuleGroebnerBasis<R> {
        &self.basis
    }

    /// Ambient value that was reduced.
    pub fn value(&self) -> &ModuleElement<R> {
        &self.value
    }

    /// Quotient coordinates in the canonical basis generators.
    pub fn quotient(&self) -> &ModuleElement<R> {
        &self.quotient
    }

    /// Combination of the original input generators contributing to `value`.
    pub fn original_combination(&self) -> &ModuleElement<R> {
        &self.original_combination
    }

    /// Canonical module remainder.
    pub fn remainder(&self) -> &ModuleElement<R> {
        &self.remainder
    }

    /// Whether this reduction proves input-submodule membership.
    pub fn is_member(&self) -> bool {
        is_zero(&self.remainder)
    }

    /// Original-generator lift when this reduction proves membership.
    pub fn lift(&self) -> Option<&ModuleElement<R>> {
        self.is_member().then_some(&self.original_combination)
    }

    /// Verify both exact reconstruction identities.
    pub fn verify(&self) -> Result<(), ModuleGroebnerError> {
        if self.value.module() != self.basis.input.ambient()
            || self.remainder.module() != self.basis.input.ambient()
            || self.quotient.module() != self.basis.basis.generator_map().source()
            || self.original_combination.module() != self.basis.input.generator_map().source()
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "module reduction components have incompatible modules",
            ));
        }
        let basis_part = self.basis.basis.generator_map().apply(&self.quotient)?;
        if basis_part.add(&self.remainder)? != self.value {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "quotient and remainder do not reconstruct the reduced value",
            ));
        }
        if self.basis.transformation.apply(&self.quotient)? != self.original_combination {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the original-generator combination does not match the change map",
            ));
        }
        let original_part = self
            .basis
            .input
            .generator_map()
            .apply(&self.original_combination)?;
        if original_part.add(&self.remainder)? != self.value {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the original lift and remainder do not reconstruct the value",
            ));
        }
        Ok(())
    }

    /// Serialize the complete normal-form and lift evidence as typed JSON.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and independently verify normal-form and lift evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        Self: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> Submodule<PolynomialRing<C>> {
    /// Decide exact containment in another submodule of the same free module.
    pub fn is_subset_of(
        &self,
        rhs: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<bool, ModuleGroebnerError> {
        require_portable_presented_order(&order)?;
        if self.ambient() != rhs.ambient() {
            return Err(ModuleError::ModuleMismatch {
                operation: "submodule containment",
            }
            .into());
        }
        rhs.groebner_basis(order)?
            .contains_all(&map_columns(self.generator_map())?)
    }

    /// Decide equality of generated submodules, independently of generators.
    pub fn is_equal(
        &self,
        rhs: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<bool, ModuleGroebnerError> {
        Ok(self.is_subset_of(rhs, order.clone())? && rhs.is_subset_of(self, order)?)
    }

    /// Exact intersection with another submodule of the same free module.
    ///
    /// The kernel of `[A,-B]` records every equality `Au=Bv`. Projecting its
    /// generators to `u` and applying `A` therefore generates precisely the
    /// intersection of the two images.
    pub fn intersection(
        &self,
        rhs: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<Self, ModuleGroebnerError> {
        require_portable_presented_order(&order)?;
        if self.ambient() != rhs.ambient() {
            return Err(ModuleError::ModuleMismatch {
                operation: "submodule intersection",
            }
            .into());
        }
        let negative_rhs =
            ModuleHomomorphism::zero(rhs.generator_map().source().clone(), rhs.ambient().clone())?
                .sub(rhs.generator_map())?;
        let graph = self.generator_map().copair(&negative_rhs)?;
        let kernel = graph.kernel(order)?;
        let left_rank = self.generator_count();
        let generators = map_columns(kernel.generator_map())?
            .into_iter()
            .map(|solution| {
                element_from_coordinates(
                    self.generator_map().source().clone(),
                    solution.coordinates().values().take(left_rank).collect(),
                )
                .and_then(|coordinates| self.generator_map().apply(&coordinates))
            })
            .collect::<Result<Vec<_>, ModuleError>>()?
            .into_iter()
            .filter(|generator| !is_zero(generator))
            .collect();
        Self::from_generators(self.ambient().clone(), generators).map_err(Into::into)
    }

    /// Exact module quotient `self : scalar` inside the ambient free module.
    ///
    /// This is the preimage of `self` under scalar multiplication. The kernel
    /// of `[scalar * identity,-A]` retains a complete witness for every
    /// relation `scalar * u = Av`; projecting to `u` gives the colon module.
    pub fn colon_by_element(
        &self,
        scalar: &Polynomial<C>,
        order: ModuleMonomialOrder,
    ) -> Result<Self, ModuleGroebnerError> {
        require_portable_presented_order(&order)?;
        if scalar.ring() != self.ambient().ring() {
            return Err(ContextMismatchError {
                expected: "the submodule's exact polynomial context".to_owned(),
                actual: "an unequal polynomial context".to_owned(),
            }
            .into());
        }
        if scalar.is_zero() {
            return Self::from_generator_map(
                self.ambient().clone(),
                ModuleHomomorphism::identity(self.ambient().clone())?,
            )
            .map_err(Into::into);
        }
        if scalar.is_one() {
            return Ok(self.clone());
        }
        let rank = self.ambient().rank();
        let multiplication_columns = (0..rank)
            .map(|column| {
                let mut values = vec![self.ambient().ring().zero(); rank];
                values[column] = scalar.clone();
                values
            })
            .collect();
        let multiplication = ModuleHomomorphism::from_coordinate_columns(
            self.ambient().clone(),
            self.ambient().clone(),
            multiplication_columns,
        )?;
        let negative_generators = ModuleHomomorphism::zero(
            self.generator_map().source().clone(),
            self.ambient().clone(),
        )?
        .sub(self.generator_map())?;
        let graph = multiplication.copair(&negative_generators)?;
        let kernel = graph.kernel(order)?;
        let generators = map_columns(kernel.generator_map())?
            .into_iter()
            .map(|solution| {
                element_from_coordinates(
                    self.ambient().clone(),
                    solution.coordinates().values().take(rank).collect(),
                )
            })
            .collect::<Result<Vec<_>, ModuleError>>()?
            .into_iter()
            .filter(|generator| !is_zero(generator))
            .collect();
        Self::from_generators(self.ambient().clone(), generators).map_err(Into::into)
    }

    /// Saturation `self : scalar^infinity`, stopping only at exact equality.
    ///
    /// Iterated colons form an ascending chain and therefore stabilize over
    /// the Noetherian polynomial ring. The implementation fails closed if the
    /// explicit native resource bound is reached before equality is proved.
    pub fn saturate_by_element(
        &self,
        scalar: &Polynomial<C>,
        order: ModuleMonomialOrder,
    ) -> Result<Self, ModuleGroebnerError> {
        let mut current = self.clone();
        for _ in 0..MAX_SUBMODULE_SATURATION_STEPS {
            let next = current.colon_by_element(scalar, order.clone())?;
            if current.is_equal(&next, order.clone())? {
                return Ok(next);
            }
            current = next;
        }
        Err(resource_limit(
            "submodule saturation step count",
            MAX_SUBMODULE_SATURATION_STEPS,
        ))
    }

    /// Remove redundant homogeneous generators over a positively graded
    /// polynomial field.
    ///
    /// The returned generators are a deterministic subset of the nonzero
    /// input generators.  Their image is either unchanged or independently
    /// checked to equal the original submodule.  This operation requires
    /// explicit shifts on the ambient free module and a positive rank-one
    /// polynomial grading. Those hypotheses make a nonzero constant
    /// coefficient in a homogeneous
    /// relation a scalar unit, so exact field elimination detects precisely
    /// the redundant generators.
    pub fn minimal_homogeneous_generators(
        &self,
    ) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
        let ambient = self.ambient();
        let ring = ambient.ring();
        let ambient_shifts = ambient
            .degree_shifts()
            .ok_or(ModuleError::GradingMismatch {
                operation: "minimal homogeneous generators",
            })?;
        if ambient.grading_rank() != Some(ring.grading().rank())
            || ring.grading().rank() != 1
            || ring
                .grading()
                .variable_degrees()
                .iter()
                .any(|degree| !degree.coordinates()[0].is_positive())
        {
            return Err(ModuleError::GradingMismatch {
                operation: "minimal homogeneous generators",
            }
            .into());
        }

        let original_labels = self.generator_map().source().basis_labels();
        let mut generators = Vec::new();
        let mut generator_degrees = Vec::new();
        let mut labels = original_labels.map(|_| Vec::new());
        for (index, generator) in map_columns(self.generator_map())?.into_iter().enumerate() {
            if is_zero(&generator) {
                continue;
            }
            let degree = homogeneous_module_degree(&generator, ring.grading(), ambient_shifts)?
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "a nonzero homogeneous generator has no degree",
                ))?;
            generators.push(generator);
            generator_degrees.push(degree);
            if let (Some(output), Some(input)) = (&mut labels, original_labels) {
                output.push(input[index].clone());
            }
        }
        let working = graded_submodule_from_homogeneous_generators(
            ambient,
            generators,
            labels,
            "minimal homogeneous generators",
        )?;
        if working.generator_count() <= 1 {
            return Ok(working);
        }
        if generator_degrees
            .first()
            .is_some_and(|first| generator_degrees.iter().all(|degree| degree == first))
            && homogeneous_generators_are_field_independent(&working)?
        {
            return Ok(working);
        }

        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let relations = working.groebner_basis(order.clone())?.syzygies()?;
        let relation_generators = map_columns(relations.generator_map())?;
        let relation_shifts =
            relations
                .ambient()
                .degree_shifts()
                .ok_or(ModuleError::GradingMismatch {
                    operation: "minimal homogeneous generators",
                })?;
        let field = ring.coefficient_ring().clone();
        let constant_relations = if relation_generators.is_empty() {
            Matrix::zeros(field, 0, working.generator_count())
        } else {
            let rows = relation_generators
                .iter()
                .map(|relation| {
                    homogeneous_module_degree(relation, ring.grading(), relation_shifts)?.ok_or(
                        ModuleGroebnerError::InvalidEvidence(
                            "a nonzero homogeneous relation has no degree",
                        ),
                    )?;
                    let mut row = relation
                        .coordinates()
                        .values()
                        .map(|coefficient| coefficient.constant_coefficient())
                        .collect::<Vec<_>>();
                    row.reverse();
                    Ok(row)
                })
                .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
            Matrix::from_rows(field, rows).map_err(ModuleError::from)?
        };
        let redundant = constant_relations
            .pivot_columns_from_elimination()
            .map_err(ModuleError::from)?
            .into_iter()
            .map(|reversed| working.generator_count() - 1 - reversed)
            .collect::<Vec<_>>();
        if redundant.is_empty() {
            return Ok(working);
        }

        let redundant = redundant.into_iter().collect::<BTreeSet<_>>();
        let working_labels = working.generator_map().source().basis_labels();
        let retained = map_columns(working.generator_map())?
            .into_iter()
            .enumerate()
            .filter(|(index, _)| !redundant.contains(index))
            .collect::<Vec<_>>();
        let labels = working_labels.map(|input| {
            retained
                .iter()
                .map(|(index, _)| input[*index].clone())
                .collect()
        });
        let result = graded_submodule_from_homogeneous_generators(
            ambient,
            retained.into_iter().map(|(_, value)| value).collect(),
            labels,
            "minimal homogeneous generators",
        )?;

        let result_basis = result.groebner_basis(order)?;
        let original_generators = map_columns(working.generator_map())?;
        if !result_basis.contains_all(&original_generators)? {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "minimal generators do not span the original submodule",
            ));
        }
        let remaining_relations = result_basis.syzygies()?;
        let remaining_shifts =
            remaining_relations
                .ambient()
                .degree_shifts()
                .ok_or(ModuleError::GradingMismatch {
                    operation: "minimal homogeneous generators",
                })?;
        for relation in map_columns(remaining_relations.generator_map())? {
            homogeneous_module_degree(&relation, ring.grading(), remaining_shifts)?.ok_or(
                ModuleGroebnerError::InvalidEvidence(
                    "a nonzero homogeneous relation has no degree",
                ),
            )?;
            if relation.coordinates().values().any(|coefficient| {
                coefficient.constant_coefficient() != ring.coefficient_ring().zero()
            }) {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "minimal generators retain a scalar-unit relation",
                ));
            }
        }
        Ok(result)
    }

    /// Compute the complete canonical reduced module Groebner basis.
    pub fn groebner_basis(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<ModuleGroebnerBasis<PolynomialRing<C>>, ModuleGroebnerError> {
        self.groebner_basis_with_target(order, None)
    }

    /// Compute the complete canonical reduced basis using an exact Hilbert
    /// series as a construction-completion target.
    ///
    /// The input must be multihomogeneous in an explicitly shifted ambient
    /// module. A target match can discard remaining construction pairs, but
    /// the returned basis still passes the independent Buchberger and
    /// change-map verifier. A bad target fails closed after an ordinary retry.
    pub fn groebner_basis_with_hilbert_series(
        &self,
        order: ModuleMonomialOrder,
        target: &HilbertSeries,
    ) -> Result<ModuleGroebnerBasis<PolynomialRing<C>>, ModuleGroebnerError> {
        validate_order(self.ambient(), &order)?;
        let ambient = self.ambient();
        let shifts = ambient
            .degree_shifts()
            .ok_or_else(|| ModuleError::GradingMismatch {
                operation: "module Hilbert target",
            })?;
        if target.grading() != ambient.ring().grading()
            || ambient.grading_rank() != Some(target.grading().rank())
        {
            return Err(ModuleError::GradingMismatch {
                operation: "module Hilbert target",
            }
            .into());
        }
        for generator in map_columns(self.generator_map())? {
            homogeneous_module_degree(&generator, ambient.ring().grading(), shifts)?;
        }
        let guided = self.groebner_basis_with_target(order.clone(), Some(target));
        let result = match guided {
            Ok(result) => result,
            Err(_) => self.groebner_basis_with_target(order, None)?,
        };
        if &result.hilbert_series()? != target {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the computed module quotient does not have the supplied Hilbert series",
            ));
        }
        Ok(result)
    }

    fn groebner_basis_with_target(
        &self,
        order: ModuleMonomialOrder,
        target: Option<&HilbertSeries>,
    ) -> Result<ModuleGroebnerBasis<PolynomialRing<C>>, ModuleGroebnerError> {
        validate_order(self.ambient(), &order)?;
        match monomial_module_basis(self, &order) {
            Ok(Some(result)) => return Ok(result),
            Ok(None) => {}
            Err(error) => return Err(error),
        }
        if target.is_none() {
            match higher_rank_kernel_basis(self, &order) {
                Ok(Some(result)) => return Ok(result),
                Ok(None) | Err(ModuleGroebnerError::InvalidEvidence(_)) => {}
                Err(error) => return Err(error),
            }
            match rank_one_kernel_basis(self, &order) {
                Ok(Some(result)) => return Ok(result),
                Ok(None) | Err(ModuleGroebnerError::InvalidEvidence(_)) => {}
                Err(error) => return Err(error),
            }
        }
        let original_source = self.generator_map().source();
        let mut tracked = Vec::new();
        for (index, generator) in map_columns(self.generator_map())?.into_iter().enumerate() {
            if is_zero(&generator) {
                continue;
            }
            let representation = original_source.basis_element(index)?;
            let (generator, representation) = make_monic_pair(generator, representation, &order)?;
            if tracked
                .iter()
                .any(|entry: &Tracked<C>| entry.value == generator)
            {
                continue;
            }
            tracked.push(Tracked {
                value: generator,
                representation,
            });
        }
        if tracked.len() > MAX_BASIS_SIZE {
            return Err(resource_limit("basis size", MAX_BASIS_SIZE));
        }
        let mut pairs = VecDeque::new();
        for left in 0..tracked.len() {
            for right in (left + 1)..tracked.len() {
                if leading_components_agree(&tracked[left].value, &tracked[right].value, &order) {
                    pairs.push_back((left, right));
                }
            }
        }
        require_pair_bound(pairs.len())?;
        if target.is_some_and(|target| {
            tracked_hilbert_series(self, &tracked, &order).is_ok_and(|series| &series == target)
        }) {
            pairs.clear();
        }
        let mut steps = 0;
        while let Some((left, right)) = pairs.pop_front() {
            let Some((s, left_factor, right_factor)) =
                s_element(&tracked[left].value, &tracked[right].value, &order)?
            else {
                continue;
            };
            let left_representation = tracked[left].representation.scalar_mul(&left_factor)?;
            let right_representation = tracked[right].representation.scalar_mul(&right_factor)?;
            let representation = left_representation.sub(&right_representation)?;
            let values = tracked
                .iter()
                .map(|entry| entry.value.clone())
                .collect::<Vec<_>>();
            let (remainder, quotients) = normal_form(&s, &values, &order, &mut steps)?;
            if is_zero(&remainder) {
                continue;
            }
            let mut remainder_representation = representation;
            for (quotient, entry) in quotients.iter().zip(&tracked) {
                remainder_representation =
                    remainder_representation.sub(&entry.representation.scalar_mul(quotient)?)?;
            }
            let (remainder, remainder_representation) =
                make_monic_pair(remainder, remainder_representation, &order)?;
            if tracked.iter().any(|entry| entry.value == remainder) {
                continue;
            }
            if tracked.len() == MAX_BASIS_SIZE {
                return Err(resource_limit("basis size", MAX_BASIS_SIZE));
            }
            let next = tracked.len();
            for (previous, entry) in tracked.iter().enumerate().take(next) {
                if leading_components_agree(&entry.value, &remainder, &order) {
                    pairs.push_back((previous, next));
                }
            }
            require_pair_bound(pairs.len())?;
            tracked.push(Tracked {
                value: remainder,
                representation: remainder_representation,
            });
            if target.is_some_and(|target| {
                tracked_hilbert_series(self, &tracked, &order).is_ok_and(|series| &series == target)
            }) {
                pairs.clear();
            }
        }
        tracked = interreduce(tracked, &order, &mut steps)?;
        let result = module_basis_from_tracked(self, order, tracked)?;
        result.verify()?;
        Ok(result)
    }
}

fn monomial_module_basis<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
    let original_source = input.generator_map().source();
    let ring = input.ambient().ring();
    let mut tracked = Vec::new();
    for (index, generator) in map_columns(input.generator_map())?.into_iter().enumerate() {
        if is_zero(&generator) {
            continue;
        }
        let mut terms = 0_usize;
        for coordinate in generator.coordinates().values() {
            terms = terms.saturating_add(coordinate.terms().take(2).count());
            if terms > 1 {
                return Ok(None);
            }
        }
        if terms != 1 {
            return Ok(None);
        }
        let (_, coefficient) = leading_term(&generator, order).ok_or(
            ModuleGroebnerError::InvalidEvidence("a monomial module basis contains zero"),
        )?;
        let coefficient_one = ring.coefficient_ring().one();
        let (value, scalar) = if coefficient == coefficient_one {
            (generator, ring.one())
        } else {
            let inverse = ring
                .coefficient_ring()
                .inverse(&coefficient)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let scalar = ring.constant(inverse)?;
            (generator.scalar_mul(&scalar)?, scalar)
        };
        tracked.push((value, index, scalar));
    }
    tracked.sort_by(|left, right| {
        let left = leading_term(&left.0, order)
            .expect("nonzero monomial basis invariant")
            .0;
        let right = leading_term(&right.0, order)
            .expect("nonzero monomial basis invariant")
            .0;
        compare_module_monomials(&right, &left, order)
    });
    let leaders = tracked
        .iter()
        .map(|entry| {
            leading_term(&entry.0, order)
                .map(|(leading, _)| leading)
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "a monomial module basis contains zero",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let keep = leaders
        .iter()
        .enumerate()
        .map(|(index, leading)| {
            !leaders.iter().enumerate().any(|(other_index, other)| {
                index != other_index
                    && module_monomial_divides(other, leading)
                    && (other != leading || other_index < index)
            })
        })
        .collect::<Vec<_>>();
    let retained = tracked
        .into_iter()
        .zip(leaders)
        .zip(keep)
        .filter_map(|(((value, original_index, scalar), leading), keep)| {
            keep.then_some((value, original_index, scalar, leading))
        })
        .collect::<Vec<_>>();
    let mut basis_values = Vec::with_capacity(retained.len());
    let mut transformation_entries = Vec::with_capacity(retained.len());
    let mut leading_monomials = Vec::with_capacity(retained.len());
    for (column, (value, row, scalar, leading)) in retained.into_iter().enumerate() {
        basis_values.push(value);
        transformation_entries.push((row, column, scalar));
        leading_monomials.push(leading);
    }
    let basis = submodule_from_homogeneous_generators(input.ambient(), basis_values)?;
    let transformation = ModuleHomomorphism::from_sparse_matrix(
        basis.generator_map().source().clone(),
        original_source.clone(),
        SparseMatrix::from_triplets(
            ring.clone(),
            original_source.rank(),
            transformation_entries.len(),
            transformation_entries,
        )
        .map_err(ModuleError::from)?,
    )?;
    let result = ModuleGroebnerBasis {
        input: input.clone(),
        order: order.clone(),
        basis,
        transformation,
        leading_monomials,
    };
    debug_assert!(result.verify().is_ok());
    Ok(Some(result))
}

fn higher_rank_kernel_basis<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
    if input.ambient().rank() <= 1 {
        return Ok(None);
    }
    let ring = input.ambient().ring();
    let lane = NoetherFieldLane {
        field: ring.coefficient_ring(),
    };
    let Some(kernel_order) = to_kernel_module_order(order) else {
        return Ok(None);
    };
    if (ring.coefficient_ring() as &dyn Any).is::<RationalField>() {
        return specialized_rational_module_basis(input, order, kernel_order);
    }
    if let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() {
        if field.characteristic() == &Integer::from(2) {
            if field.degree() == 1 {
                return specialized_f2_module_basis(input, order, kernel_order, field);
            }
            if let Some(binary_field) = binary_extension_kernel_field(field) {
                return specialized_binary_extension_module_basis(
                    input,
                    order,
                    kernel_order,
                    field,
                    &binary_field,
                );
            }
        }
        if field.degree() == 1 {
            if let Some(characteristic) = field
                .characteristic()
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value < (1_u32 << 31))
            {
                return specialized_prime_module_basis(
                    input,
                    order,
                    kernel_order,
                    field,
                    characteristic,
                );
            }
        }
    }
    let Some(generators) = map_columns(input.generator_map())?
        .into_iter()
        .map(|generator| {
            generator
                .coordinates()
                .values()
                .map(|coordinate| to_kernel_polynomial(&coordinate))
                .collect::<Option<Vec<_>>>()
                .map(|coordinates| {
                    groebner_kernel::SparseModuleElement::from_coordinates(coordinates, &lane)
                })
        })
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let output = match groebner_kernel::module_groebner_basis_with_transform(
        &lane,
        &generators,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    if output.basis.len() != output.transformations.len()
        || output
            .transformations
            .iter()
            .any(|row| row.rank() != generators.len())
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "the higher-rank module kernel returned an invalid change-matrix shape",
        ));
    }
    let original_source = input.generator_map().source().clone();
    let tracked = output
        .basis
        .into_iter()
        .zip(output.transformations)
        .map(|(value, representation)| {
            let value = element_from_coordinates(
                input.ambient().clone(),
                from_kernel_basis(ring, value.coordinates(&lane))
                    .map_err(map_polynomial_groebner_error)?,
            )?;
            let representation = element_from_coordinates(
                original_source.clone(),
                from_kernel_basis(ring, representation.coordinates(&lane))
                    .map_err(map_polynomial_groebner_error)?,
            )?;
            Ok(Tracked {
                value,
                representation,
            })
        })
        .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
    let result = module_basis_from_tracked(input, order.clone(), tracked)?;
    debug_assert!(result.verify().is_ok());
    Ok(Some(result))
}

fn sparse_kernel_basis_source<C, E>(
    ambient: &FreeModule<PolynomialRing<C>>,
    basis: &[groebner_kernel::SparseModuleElement<E>],
) -> Result<FreeModule<PolynomialRing<C>>, ModuleGroebnerError>
where
    C: Field,
    E: Clone + Eq,
{
    let ring = ambient.ring();
    let Some(ambient_shifts) = ambient.degree_shifts() else {
        return Ok(FreeModule::new(ring.clone(), basis.len()));
    };
    if ambient.grading_rank() != Some(ring.grading().rank()) {
        return Err(ModuleError::GradingMismatch {
            operation: "homogeneous generator source",
        }
        .into());
    }
    let inferred_shifts = basis
        .iter()
        .map(|value| {
            let mut homogeneous_degree = None;
            for (term, _) in value.terms() {
                let shift = ambient_shifts.get(term.component()).ok_or(
                    ModuleGroebnerError::InvalidEvidence(
                        "the ambient module has incomplete degree shifts",
                    ),
                )?;
                let exponent = term
                    .monomial()
                    .to_u32_vec()
                    .into_iter()
                    .map(Integer::from)
                    .collect::<Vec<_>>();
                let degree = ring
                    .grading()
                    .degree(&exponent)
                    .and_then(|degree| degree.sub(shift))
                    .map_err(|_| {
                        ModuleGroebnerError::InvalidEvidence(
                            "a module term has an incompatible grading degree",
                        )
                    })?;
                match &homogeneous_degree {
                    None => homogeneous_degree = Some(degree),
                    Some(known) if known == &degree => {}
                    Some(_) => {
                        return Err(ModuleGroebnerError::InvalidEvidence(
                            "the computed module basis is not homogeneous",
                        ));
                    }
                }
            }
            homogeneous_degree.map(|degree| degree.neg()).ok_or(
                ModuleGroebnerError::InvalidEvidence("a canonical module basis contains zero"),
            )
        })
        .collect::<Result<Vec<_>, _>>();
    match inferred_shifts {
        Ok(shifts) => Ok(FreeModule::new(ring.clone(), basis.len())
            .with_degree_shifts(ring.grading().rank(), shifts)?),
        Err(_) => Ok(FreeModule::new(ring.clone(), basis.len())),
    }
}

fn rational_kernel_matrix_entries<C: Field>(
    ring: &PolynomialRing<C>,
    elements: Vec<groebner_kernel::SparseModuleElement<crate::Rational>>,
    expected_rank: usize,
    expected_order: poly_core::MonomialOrder,
) -> Result<Vec<(usize, usize, Polynomial<C>)>, ModuleGroebnerError> {
    let mut entries = Vec::new();
    for (column, element) in elements.into_iter().enumerate() {
        if element.arity() != ring.variables().len()
            || element.rank() != expected_rank
            || element.polynomial_order() != expected_order
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "a rational module kernel result has an incompatible shape",
            ));
        }
        let mut terms = element.into_terms().into_iter().peekable();
        while let Some((term, coefficient)) = terms.next() {
            let (monomial, row) = term.into_parts();
            let mut coordinate_terms = vec![(monomial, coefficient)];
            while terms
                .peek()
                .is_some_and(|(next, _)| next.component() == row)
            {
                let (term, coefficient) =
                    terms.next().expect("a peeked rational module term exists");
                let (monomial, _) = term.into_parts();
                coordinate_terms.push((monomial, coefficient));
            }
            entries.push((
                row,
                column,
                from_rational_kernel_terms(ring, coordinate_terms)
                    .map_err(map_polynomial_groebner_error)?,
            ));
        }
    }
    Ok(entries)
}

fn rational_kernel_generators_from_sparse_map<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
    order: poly_core::MonomialOrder,
) -> Option<Vec<groebner_kernel::SparseModuleElement<crate::Rational>>> {
    let matrix = map.sparse_matrix()?;
    let ring = map.target().ring();
    let mut columns = vec![Vec::new(); map.source().rank()];
    for (row, offsets) in matrix.row_offsets().windows(2).enumerate() {
        for index in offsets[0]..offsets[1] {
            let column = matrix.column_indices()[index];
            for (exponent, coefficient) in matrix.values()[index].terms() {
                let coefficient = (coefficient as &dyn Any)
                    .downcast_ref::<crate::Rational>()?
                    .clone();
                let monomial = exponent_to_kernel(exponent)?;
                columns[column].push((
                    groebner_kernel::SparseModuleTerm::new(monomial, row),
                    coefficient,
                ));
            }
        }
    }
    Some(
        columns
            .into_iter()
            .map(|terms| {
                groebner_kernel::SparseModuleElement::from_sorted_terms(
                    ring.variables().len(),
                    map.target().rank(),
                    order,
                    terms,
                    &RationalKernelLane,
                )
            })
            .collect(),
    )
}

fn specialized_rational_module_basis<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
    kernel_order: groebner_kernel::SparseModuleOrder,
) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
    let ring = input.ambient().ring();
    let generators = if let Some(generators) =
        rational_kernel_generators_from_sparse_map(input.generator_map(), kernel_order.base_order())
    {
        Some(generators)
    } else {
        map_columns(input.generator_map())?
            .into_iter()
            .map(|generator| {
                generator
                    .coordinates()
                    .values()
                    .map(|coordinate| {
                        to_rational_kernel_polynomial(&coordinate, kernel_order.base_order())
                    })
                    .collect::<Option<Vec<_>>>()
                    .map(|coordinates| {
                        groebner_kernel::SparseModuleElement::from_coordinates(
                            coordinates,
                            &RationalKernelLane,
                        )
                    })
            })
            .collect::<Option<Vec<_>>>()
    };
    let Some(generators) = generators else {
        return Ok(None);
    };
    let output = match groebner_kernel::module_groebner_basis_with_transform_rational(
        &generators,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    if output.basis.len() != output.transformations.len()
        || output
            .transformations
            .iter()
            .any(|row| row.rank() != generators.len())
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "the rational module kernel returned an invalid change-matrix shape",
        ));
    }
    let leading_monomials =
        groebner_kernel::module_leading_terms(&RationalKernelLane, &output.basis, &kernel_order)
            .map_err(map_kernel_module_error)?
            .into_iter()
            .map(|term| {
                ModuleMonomial::new(
                    term.monomial()
                        .to_u32_vec()
                        .into_iter()
                        .map(Integer::from)
                        .collect(),
                    term.component(),
                )
            })
            .collect::<Vec<_>>();
    let basis_source = sparse_kernel_basis_source(input.ambient(), &output.basis)?;
    let basis_count = output.basis.len();
    let basis_entries = rational_kernel_matrix_entries(
        ring,
        output.basis,
        input.ambient().rank(),
        kernel_order.base_order(),
    )?;
    let basis_map = ModuleHomomorphism::from_sparse_matrix(
        basis_source.clone(),
        input.ambient().clone(),
        SparseMatrix::from_triplets(
            ring.clone(),
            input.ambient().rank(),
            basis_count,
            basis_entries,
        )
        .map_err(ModuleError::from)?,
    )?;
    let basis = Submodule::from_generator_map(input.ambient().clone(), basis_map)?;
    let original_source = input.generator_map().source().clone();
    let transformation_entries = rational_kernel_matrix_entries(
        ring,
        output.transformations,
        original_source.rank(),
        kernel_order.base_order(),
    )?;
    let transformation = ModuleHomomorphism::from_sparse_matrix(
        basis_source,
        original_source.clone(),
        SparseMatrix::from_triplets(
            ring.clone(),
            original_source.rank(),
            basis_count,
            transformation_entries,
        )
        .map_err(ModuleError::from)?,
    )?;
    let result = ModuleGroebnerBasis {
        input: input.clone(),
        order: order.clone(),
        basis,
        transformation,
        leading_monomials,
    };
    debug_assert!(result.verify().is_ok());
    Ok(Some(result))
}

fn kernel_schreyer_relations<C: Field>(
    basis: &[PolynomialModule<C>],
    source: &FreeModule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
    hilbert_target: Option<&groebner_kernel::StandardModuleHilbertTarget>,
) -> Result<Option<Vec<PolynomialModule<C>>>, ModuleGroebnerError> {
    if basis.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let ring = source.ring();
    let Some(kernel_order) = to_kernel_module_order(order) else {
        return Ok(None);
    };
    if let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() {
        if field.characteristic() == &Integer::from(2) {
            if field.degree() == 1 {
                let Some(output) =
                    specialized_f2_schreyer_output(basis, ring, &kernel_order, hilbert_target)?
                else {
                    return Ok(None);
                };
                return project_specialized_schreyer_relations(
                    output,
                    source,
                    &groebner_kernel::BinaryField,
                    |coefficient| Ok(field.element(u64::from(*coefficient))),
                )
                .map(Some);
            }
            if let Some(binary_field) = binary_extension_kernel_field(field) {
                let Some(output) = specialized_binary_extension_schreyer_output(
                    basis,
                    ring,
                    &kernel_order,
                    &binary_field,
                    hilbert_target,
                )?
                else {
                    return Ok(None);
                };
                return project_specialized_schreyer_relations(
                    output,
                    source,
                    &binary_field,
                    |coefficient| decode_binary_extension_coefficient(field, *coefficient),
                )
                .map(Some);
            }
        }
        if field.degree() == 1 {
            if let Some(characteristic) = field
                .characteristic()
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value < (1_u32 << 31))
            {
                return specialized_prime_schreyer_relations(
                    basis,
                    source,
                    &kernel_order,
                    field,
                    characteristic,
                    hilbert_target,
                );
            }
        }
    }

    let lane = NoetherFieldLane {
        field: ring.coefficient_ring(),
    };
    let Some(basis) = to_kernel_module_elements(basis, &lane) else {
        return Ok(None);
    };
    let options = groebner_kernel::GroebnerOptions::default();
    let output = match hilbert_target.map_or_else(
        || groebner_kernel::module_schreyer_syzygies(&lane, &basis, &kernel_order, &options),
        |target| {
            groebner_kernel::module_schreyer_syzygies_with_hilbert_target(
                &lane,
                &basis,
                &kernel_order,
                &options,
                target,
            )
            .or_else(|error| match error {
                groebner_kernel::GroebnerError::VerificationFailed => {
                    groebner_kernel::module_schreyer_syzygies(
                        &lane,
                        &basis,
                        &kernel_order,
                        &options,
                    )
                }
                _ => Err(error),
            })
        },
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    output
        .generators
        .into_iter()
        .map(|relation| {
            element_from_coordinates(
                source.clone(),
                from_kernel_basis(ring, relation.coordinates(&lane))
                    .map_err(map_polynomial_groebner_error)?,
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>, ModuleGroebnerError>>()
        .map(Some)
}

fn standard_schreyer_hilbert_target<C: Field>(
    basis: &ModuleGroebnerBasis<PolynomialRing<C>>,
) -> Option<groebner_kernel::StandardModuleHilbertTarget> {
    let previous_ambient = basis.input().ambient();
    let ring = previous_ambient.ring();
    if ring.grading() != &FreeAbelianGrading::standard(ring.variables().len()) {
        return None;
    }
    let previous_degrees = standard_free_generator_degrees(previous_ambient)?;
    let source_degrees = standard_free_generator_degrees(basis.basis().generator_map().source())?;
    let leading_terms = basis
        .leading_monomials
        .iter()
        .map(|leading| Some((exponent_to_kernel(&leading.exponent)?, leading.component)))
        .collect::<Option<Vec<_>>>()?;
    groebner_kernel::StandardModuleHilbertTarget::from_submodule_leading_terms(
        ring.variables().len(),
        &previous_degrees,
        source_degrees,
        &leading_terms,
    )
    .ok()
}

pub(crate) fn packed_standard_graded_betti<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<resolution_kernel::StandardGradedBettiOutput>, ModuleGroebnerError> {
    let ring = input.ambient().ring();
    if ring.grading() != &FreeAbelianGrading::standard(ring.variables().len()) {
        return Ok(None);
    }
    let Some(kernel_order) = to_kernel_module_order(order) else {
        return Ok(None);
    };
    let Some(target_degrees) = standard_free_generator_degrees(input.ambient()) else {
        return Ok(None);
    };
    let Some(presentation_degrees) =
        standard_free_generator_degrees(input.generator_map().source())
    else {
        return Ok(None);
    };
    let columns = map_columns(input.generator_map())?;
    if let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() {
        if field.characteristic() == &Integer::from(2) {
            if field.degree() == 1 {
                let Some(presentation) = to_specialized_kernel_module_elements(
                    &columns,
                    ring,
                    &kernel_order,
                    &groebner_kernel::BinaryField,
                    |coefficient| Some(u8::from(!coefficient.is_zero())),
                ) else {
                    return Ok(None);
                };
                return match resolution_kernel::standard_graded_betti_f2(
                    &presentation,
                    &target_degrees,
                    &presentation_degrees,
                    &kernel_order,
                    &groebner_kernel::GroebnerOptions::default(),
                ) {
                    Ok(output) => Ok(Some(output)),
                    Err(_) => Ok(None),
                };
            }
            if let Some(binary_field) = binary_extension_kernel_field(field) {
                let Some(presentation) = to_specialized_kernel_module_elements(
                    &columns,
                    ring,
                    &kernel_order,
                    &binary_field,
                    encode_binary_extension_coefficient,
                ) else {
                    return Ok(None);
                };
                return match resolution_kernel::standard_graded_betti_binary_extension(
                    &binary_field,
                    &presentation,
                    &target_degrees,
                    &presentation_degrees,
                    &kernel_order,
                    &groebner_kernel::GroebnerOptions::default(),
                ) {
                    Ok(output) => Ok(Some(output)),
                    Err(_) => Ok(None),
                };
            }
        }
        if field.degree() == 1 {
            let Some(characteristic) = field
                .characteristic()
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value < (1_u32 << 31))
            else {
                return Ok(None);
            };
            let prime = groebner_kernel::PrimeField32::new(characteristic)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let Some(presentation) =
                to_prime_kernel_module_elements(&columns, ring, &kernel_order, &prime)
            else {
                return Ok(None);
            };
            return match resolution_kernel::standard_graded_betti_prime(
                &prime,
                &presentation,
                &target_degrees,
                &presentation_degrees,
                &kernel_order,
                &groebner_kernel::GroebnerOptions::default(),
            ) {
                Ok(output) => Ok(Some(output)),
                Err(_) => Ok(None),
            };
        }
    }
    let lane = NoetherFieldLane {
        field: ring.coefficient_ring(),
    };
    let Some(presentation) = to_kernel_module_elements(&columns, &lane) else {
        return Ok(None);
    };
    match resolution_kernel::standard_graded_betti(
        &lane,
        &presentation,
        &target_degrees,
        &presentation_degrees,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => Ok(Some(output)),
        Err(_) => Ok(None),
    }
}

pub(crate) fn packed_standard_graded_resolution<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<Vec<ModuleHomomorphism<PolynomialRing<C>>>>, ModuleGroebnerError> {
    let ring = input.ambient().ring();
    if ring.grading() != &FreeAbelianGrading::standard(ring.variables().len()) {
        return Ok(None);
    }
    let Some(kernel_order) = to_kernel_module_order(order) else {
        return Ok(None);
    };
    let Some(target_degrees) = standard_free_generator_degrees(input.ambient()) else {
        return Ok(None);
    };
    let Some(presentation_degrees) =
        standard_free_generator_degrees(input.generator_map().source())
    else {
        return Ok(None);
    };
    let columns = map_columns(input.generator_map())?;
    let options = groebner_kernel::GroebnerOptions::default();

    if let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() {
        if field.characteristic() == &Integer::from(2) {
            if field.degree() == 1 {
                let Some(presentation) = to_specialized_kernel_module_elements(
                    &columns,
                    ring,
                    &kernel_order,
                    &groebner_kernel::BinaryField,
                    |coefficient| Some(u8::from(!coefficient.is_zero())),
                ) else {
                    return Ok(None);
                };
                let mut output = match resolution_kernel::standard_graded_resolution_f2(
                    &presentation,
                    &target_degrees,
                    &presentation_degrees,
                    &kernel_order,
                    &options,
                ) {
                    Ok(output) => output,
                    Err(_) => return Ok(None),
                };
                output
                    .minimalize_higher_differentials(&groebner_kernel::BinaryField)
                    .map_err(|_| {
                        ModuleGroebnerError::InvalidEvidence(
                            "a packed binary resolution could not be minimalized",
                        )
                    })?;
                return project_specialized_resolution(
                    output,
                    input.ambient(),
                    &groebner_kernel::BinaryField,
                    |coefficient| Ok(field.element(u64::from(*coefficient))),
                )
                .map(Some);
            }
            if let Some(binary_field) = binary_extension_kernel_field(field) {
                let Some(presentation) = to_specialized_kernel_module_elements(
                    &columns,
                    ring,
                    &kernel_order,
                    &binary_field,
                    encode_binary_extension_coefficient,
                ) else {
                    return Ok(None);
                };
                let mut output =
                    match resolution_kernel::standard_graded_resolution_binary_extension(
                        &binary_field,
                        &presentation,
                        &target_degrees,
                        &presentation_degrees,
                        &kernel_order,
                        &options,
                    ) {
                        Ok(output) => output,
                        Err(_) => return Ok(None),
                    };
                output
                    .minimalize_higher_differentials(&binary_field)
                    .map_err(|_| {
                        ModuleGroebnerError::InvalidEvidence(
                            "a packed binary-extension resolution could not be minimalized",
                        )
                    })?;
                return project_specialized_resolution(
                    output,
                    input.ambient(),
                    &binary_field,
                    |coefficient| decode_binary_extension_coefficient(field, *coefficient),
                )
                .map(Some);
            }
        }
        if field.degree() == 1 {
            let Some(characteristic) = field
                .characteristic()
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value < (1_u32 << 31))
            else {
                return Ok(None);
            };
            let prime = groebner_kernel::PrimeField32::new(characteristic)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let Some(presentation) =
                to_prime_kernel_module_elements(&columns, ring, &kernel_order, &prime)
            else {
                return Ok(None);
            };
            let mut output = match resolution_kernel::standard_graded_resolution_prime(
                &prime,
                &presentation,
                &target_degrees,
                &presentation_degrees,
                &kernel_order,
                &options,
            ) {
                Ok(output) => output,
                Err(_) => return Ok(None),
            };
            output
                .minimalize_higher_differentials(&prime)
                .map_err(|_| {
                    ModuleGroebnerError::InvalidEvidence(
                        "a packed prime-field resolution could not be minimalized",
                    )
                })?;
            return project_prime_resolution(output, input.ambient(), &prime, field).map(Some);
        }
    }

    let lane = NoetherFieldLane {
        field: ring.coefficient_ring(),
    };
    let Some(presentation) = to_kernel_module_elements(&columns, &lane) else {
        return Ok(None);
    };
    let mut output = match resolution_kernel::standard_graded_resolution(
        &lane,
        &presentation,
        &target_degrees,
        &presentation_degrees,
        &kernel_order,
        &options,
    ) {
        Ok(output) => output,
        Err(_) => return Ok(None),
    };
    output.minimalize_higher_differentials(&lane).map_err(|_| {
        ModuleGroebnerError::InvalidEvidence(
            "a packed exact-field resolution could not be minimalized",
        )
    })?;
    project_generic_resolution(output, input.ambient(), &lane).map(Some)
}

fn project_generic_resolution<C: Field>(
    output: resolution_kernel::StandardGradedResolutionOutput<C::Element>,
    initial_target: &FreeModule<PolynomialRing<C>>,
    field: &NoetherFieldLane<'_, C>,
) -> Result<Vec<ModuleHomomorphism<PolynomialRing<C>>>, ModuleGroebnerError> {
    let mut target = initial_target.clone();
    let mut differentials = Vec::with_capacity(output.differentials.len());
    for frame in output.differentials {
        let source =
            standard_graded_resolution_source(target.ring(), &frame.source_generator_degrees)?;
        let column_count = frame.columns.len();
        if source.rank() != column_count {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "a packed resolution frame has incompatible source degrees",
            ));
        }
        let mut entries = Vec::new();
        for (column_index, column) in frame.columns.into_iter().enumerate() {
            let coordinates = from_kernel_basis(target.ring(), column.coordinates(field))
                .map_err(map_polynomial_groebner_error)?;
            if coordinates.len() != target.rank() {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "a packed resolution frame has incompatible target rank",
                ));
            }
            entries.extend(
                coordinates
                    .into_iter()
                    .enumerate()
                    .filter(|(_, value)| !target.ring().is_zero(value))
                    .map(|(row, value)| (row, column_index, value)),
            );
        }
        let matrix = SparseMatrix::from_triplets(
            target.ring().clone(),
            target.rank(),
            column_count,
            entries,
        )
        .map_err(ModuleError::from)?;
        let differential = ModuleHomomorphism::from_sparse_matrix(source, target.clone(), matrix)?;
        target = differential.source().clone();
        differentials.push(differential);
    }
    Ok(differentials)
}

fn project_prime_resolution<C: Field>(
    output: resolution_kernel::StandardGradedResolutionOutput<u32>,
    initial_target: &FreeModule<PolynomialRing<C>>,
    prime: &groebner_kernel::PrimeField32,
    field: &FiniteField,
) -> Result<Vec<ModuleHomomorphism<PolynomialRing<C>>>, ModuleGroebnerError> {
    let mut target = initial_target.clone();
    let mut differentials = Vec::with_capacity(output.differentials.len());
    for frame in output.differentials {
        let source =
            standard_graded_resolution_source(target.ring(), &frame.source_generator_degrees)?;
        let column_count = frame.columns.len();
        if source.rank() != column_count {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "a packed prime-field resolution frame has incompatible source degrees",
            ));
        }
        let mut entries = Vec::new();
        for (column_index, column) in frame.columns.into_iter().enumerate() {
            if column.rank() != target.rank() || column.arity() != target.ring().variables().len() {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "a packed prime-field resolution frame has incompatible target rank",
                ));
            }
            let polynomial_order = column.polynomial_order();
            let mut coordinates = BTreeMap::<usize, Vec<(poly_core::Monomial, u32)>>::new();
            for (term, coefficient) in column.terms() {
                coordinates
                    .entry(term.component())
                    .or_default()
                    .push((term.monomial().clone(), *coefficient));
            }
            for (row, terms) in coordinates {
                let polynomial = poly_core::SparsePolynomial::from_sorted_terms(
                    target.ring().variables().len(),
                    polynomial_order,
                    terms,
                    prime,
                );
                entries.push((
                    row,
                    column_index,
                    from_prime_kernel_polynomial(target.ring(), polynomial, field)
                        .map_err(map_polynomial_groebner_error)?,
                ));
            }
        }
        let matrix = SparseMatrix::from_triplets(
            target.ring().clone(),
            target.rank(),
            column_count,
            entries,
        )
        .map_err(ModuleError::from)?;
        let differential = ModuleHomomorphism::from_sparse_matrix(source, target.clone(), matrix)?;
        target = differential.source().clone();
        differentials.push(differential);
    }
    Ok(differentials)
}

fn project_specialized_resolution<C, K, Decode>(
    output: resolution_kernel::StandardGradedResolutionOutput<K::Element>,
    initial_target: &FreeModule<PolynomialRing<C>>,
    field: &K,
    decode: Decode,
) -> Result<Vec<ModuleHomomorphism<PolynomialRing<C>>>, ModuleGroebnerError>
where
    C: Field,
    K: groebner_kernel::Field,
    Decode: Fn(&K::Element) -> Result<FiniteFieldElement, PolynomialGroebnerError> + Copy,
{
    let mut target = initial_target.clone();
    let mut differentials = Vec::with_capacity(output.differentials.len());
    for frame in output.differentials {
        let source =
            standard_graded_resolution_source(target.ring(), &frame.source_generator_degrees)?;
        let column_count = frame.columns.len();
        if source.rank() != column_count {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "a specialized packed resolution frame has incompatible source degrees",
            ));
        }
        let mut entries = Vec::new();
        for (column_index, column) in frame.columns.into_iter().enumerate() {
            let coordinates =
                from_specialized_kernel_coordinates(target.ring(), column, field, decode)
                    .map_err(map_polynomial_groebner_error)?;
            if coordinates.len() != target.rank() {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "a specialized packed resolution frame has incompatible target rank",
                ));
            }
            entries.extend(
                coordinates
                    .into_iter()
                    .enumerate()
                    .filter(|(_, value)| !target.ring().is_zero(value))
                    .map(|(row, value)| (row, column_index, value)),
            );
        }
        let matrix = SparseMatrix::from_triplets(
            target.ring().clone(),
            target.rank(),
            column_count,
            entries,
        )
        .map_err(ModuleError::from)?;
        let differential = ModuleHomomorphism::from_sparse_matrix(source, target.clone(), matrix)?;
        target = differential.source().clone();
        differentials.push(differential);
    }
    Ok(differentials)
}

fn standard_graded_resolution_source<C: Field>(
    ring: &PolynomialRing<C>,
    generator_degrees: &[usize],
) -> Result<FreeModule<PolynomialRing<C>>, ModuleGroebnerError> {
    if ring.grading().rank() != 1
        || generator_degrees
            .iter()
            .any(|degree| *degree > MAX_STANDARD_MODULE_HILBERT_DEGREE)
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "a packed standard-graded frame has an invalid internal degree",
        ));
    }
    let shifts = generator_degrees
        .iter()
        .map(|degree| Degree::new(vec![-Integer::from(*degree as u64)]))
        .collect();
    FreeModule::new(ring.clone(), generator_degrees.len())
        .with_degree_shifts(1, shifts)
        .map_err(Into::into)
}

fn should_use_schreyer_hilbert_guidance(leaders: &[ModuleMonomial]) -> bool {
    let mut degree_counts = BTreeMap::<Integer, usize>::new();
    let mut total_pairs = 0_usize;
    for left in 0..leaders.len() {
        for right in (left + 1)..leaders.len() {
            if leaders[left].component != leaders[right].component {
                continue;
            }
            let degree = leaders[left]
                .exponent
                .iter()
                .zip(&leaders[right].exponent)
                .fold(Integer::zero(), |sum, (left, right)| {
                    &sum + if left >= right { left } else { right }
                });
            let count = degree_counts.entry(degree).or_default();
            *count = count.saturating_add(1);
            total_pairs = total_pairs.saturating_add(1);
        }
    }
    degree_counts
        .first_key_value()
        .is_some_and(|(_, first_count)| {
            total_pairs.saturating_sub(*first_count) >= MIN_HILBERT_GUIDANCE_LATER_PAIRS
        })
}

fn standard_free_generator_degrees<C: Field>(
    module: &FreeModule<PolynomialRing<C>>,
) -> Option<Vec<usize>> {
    module
        .degree_shifts()?
        .iter()
        .map(|shift| {
            shift
                .neg()
                .coordinates()
                .first()?
                .to_u64_checked()
                .and_then(|degree| usize::try_from(degree).ok())
                .filter(|degree| *degree <= MAX_STANDARD_MODULE_HILBERT_DEGREE)
        })
        .collect()
}

fn kernel_module_membership<C: Field>(
    basis: &[PolynomialModule<C>],
    values: &[PolynomialModule<C>],
    ambient: &FreeModule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<Vec<bool>>, ModuleGroebnerError> {
    if values.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let ring = ambient.ring();
    let Some(kernel_order) = to_kernel_module_order(order) else {
        return Ok(None);
    };
    if let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() {
        if field.degree() == 1 {
            if let Some(characteristic) = field
                .characteristic()
                .to_u64_checked()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value < (1_u32 << 31))
            {
                return specialized_prime_module_membership(
                    basis,
                    values,
                    ring,
                    &kernel_order,
                    characteristic,
                );
            }
        }
    }

    let lane = NoetherFieldLane {
        field: ring.coefficient_ring(),
    };
    let Some(basis) = to_kernel_module_elements(basis, &lane) else {
        return Ok(None);
    };
    let Some(values) = to_kernel_module_elements(values, &lane) else {
        return Ok(None);
    };
    match groebner_kernel::module_membership(
        &lane,
        &basis,
        &values,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(results) => Ok(Some(results)),
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => Ok(None),
        Err(error) => Err(map_kernel_module_error(error)),
    }
}

fn to_kernel_module_elements<C: Field>(
    values: &[PolynomialModule<C>],
    lane: &NoetherFieldLane<'_, C>,
) -> Option<Vec<groebner_kernel::SparseModuleElement<C::Element>>> {
    values
        .iter()
        .map(|value| {
            value
                .coordinates()
                .values()
                .map(|coordinate| to_kernel_polynomial(&coordinate))
                .collect::<Option<Vec<_>>>()
                .map(|coordinates| {
                    groebner_kernel::SparseModuleElement::from_coordinates(coordinates, lane)
                })
        })
        .collect()
}

fn binary_extension_kernel_field(
    field: &FiniteField,
) -> Option<groebner_kernel::BinaryExtensionField16> {
    if field.characteristic() != &Integer::from(2) || !(2..=16).contains(&field.degree()) {
        return None;
    }
    let minpoly = field
        .defining_modulus()?
        .iter()
        .copied()
        .enumerate()
        .try_fold(0_u64, |bits, (index, coefficient)| {
            (coefficient <= 1 && index < 64).then_some(bits | (coefficient << index))
        })?;
    groebner_kernel::BinaryExtensionField16::new(minpoly).ok()
}

fn encode_binary_extension_coefficient(value: &FiniteFieldElement) -> Option<u16> {
    value
        .coefficients()
        .into_iter()
        .enumerate()
        .try_fold(0_u16, |bits, (index, coefficient)| {
            coefficient.to_u64_checked().and_then(|coefficient| {
                (coefficient <= 1 && index < 16).then_some(bits | ((coefficient as u16) << index))
            })
        })
}

fn decode_binary_extension_coefficient(
    field: &FiniteField,
    value: u16,
) -> Result<FiniteFieldElement, PolynomialGroebnerError> {
    let coordinates = (0..field.degree())
        .map(|index| u64::from((value >> index) & 1))
        .collect::<Vec<_>>();
    field
        .element_from_coefficients(&coordinates)
        .map_err(|_| PolynomialGroebnerError::CoefficientInvariant)
}

fn to_specialized_kernel_module_elements<C, K, Encode>(
    values: &[PolynomialModule<C>],
    ring: &PolynomialRing<C>,
    kernel_order: &groebner_kernel::SparseModuleOrder,
    kernel_field: &K,
    encode: Encode,
) -> Option<Vec<groebner_kernel::SparseModuleElement<K::Element>>>
where
    C: Field,
    K: groebner_kernel::Field,
    Encode: Fn(&FiniteFieldElement) -> Option<K::Element> + Copy,
{
    values
        .iter()
        .map(|value| {
            value
                .coordinates()
                .values()
                .map(|coordinate| {
                    let terms = coordinate
                        .terms()
                        .map(|(exponent, coefficient)| {
                            let coefficient =
                                (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                            exponent_to_kernel(exponent).and_then(|monomial| {
                                encode(coefficient).map(|value| (monomial, value))
                            })
                        })
                        .collect::<Option<Vec<_>>>()?;
                    Some(poly_core::SparsePolynomial::new(
                        ring.variables().len(),
                        kernel_order.base_order(),
                        terms,
                        kernel_field,
                    ))
                })
                .collect::<Option<Vec<_>>>()
                .map(|coordinates| {
                    groebner_kernel::SparseModuleElement::from_coordinates(
                        coordinates,
                        kernel_field,
                    )
                })
        })
        .collect()
}

fn from_specialized_kernel_polynomial<C, E, Decode>(
    ring: &PolynomialRing<C>,
    polynomial: poly_core::SparsePolynomial<E>,
    decode: Decode,
) -> Result<Polynomial<C>, PolynomialGroebnerError>
where
    C: Field,
    E: Clone + Eq,
    Decode: Fn(&E) -> Result<FiniteFieldElement, PolynomialGroebnerError>,
{
    let terms = polynomial
        .terms()
        .iter()
        .map(|(monomial, coefficient)| {
            let value = decode(coefficient)?;
            let value = (Box::new(value) as Box<dyn Any>)
                .downcast::<C::Element>()
                .map(|value| *value)
                .map_err(|_| PolynomialGroebnerError::CoefficientInvariant)?;
            Ok((
                monomial
                    .to_u32_vec()
                    .into_iter()
                    .map(Integer::from)
                    .collect::<Vec<_>>(),
                value,
            ))
        })
        .collect::<Result<Vec<_>, PolynomialGroebnerError>>()?;
    ring.from_terms(terms).map_err(Into::into)
}

fn from_specialized_kernel_coordinates<C, K, Decode>(
    ring: &PolynomialRing<C>,
    value: groebner_kernel::SparseModuleElement<K::Element>,
    kernel_field: &K,
    decode: Decode,
) -> Result<Vec<Polynomial<C>>, PolynomialGroebnerError>
where
    C: Field,
    K: groebner_kernel::Field,
    Decode: Fn(&K::Element) -> Result<FiniteFieldElement, PolynomialGroebnerError> + Copy,
{
    value
        .coordinates(kernel_field)
        .into_iter()
        .map(|coordinate| from_specialized_kernel_polynomial(ring, coordinate, decode))
        .collect()
}

fn specialized_prime_module_membership<C: Field>(
    basis: &[PolynomialModule<C>],
    values: &[PolynomialModule<C>],
    ring: &PolynomialRing<C>,
    kernel_order: &groebner_kernel::SparseModuleOrder,
    characteristic: u32,
) -> Result<Option<Vec<bool>>, ModuleGroebnerError> {
    let prime = groebner_kernel::PrimeField32::new(characteristic)
        .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
    let Some(basis) = to_prime_kernel_module_elements(basis, ring, kernel_order, &prime) else {
        return Ok(None);
    };
    let Some(values) = to_prime_kernel_module_elements(values, ring, kernel_order, &prime) else {
        return Ok(None);
    };
    match groebner_kernel::module_membership(
        &prime,
        &basis,
        &values,
        kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(results) => Ok(Some(results)),
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => Ok(None),
        Err(error) => Err(map_kernel_module_error(error)),
    }
}

fn to_prime_kernel_module_elements<C: Field>(
    values: &[PolynomialModule<C>],
    ring: &PolynomialRing<C>,
    kernel_order: &groebner_kernel::SparseModuleOrder,
    prime: &groebner_kernel::PrimeField32,
) -> Option<Vec<groebner_kernel::SparseModuleElement<u32>>> {
    values
        .iter()
        .map(|value| {
            value
                .coordinates()
                .values()
                .map(|coordinate| {
                    let terms = coordinate
                        .terms()
                        .map(|(exponent, coefficient)| {
                            let coefficient =
                                (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                            let representative = coefficient.representative().to_u64_checked()?;
                            exponent_to_kernel(exponent)
                                .map(|monomial| (monomial, representative as u32))
                        })
                        .collect::<Option<Vec<_>>>()?;
                    Some(poly_core::SparsePolynomial::new(
                        ring.variables().len(),
                        kernel_order.base_order(),
                        terms,
                        prime,
                    ))
                })
                .collect::<Option<Vec<_>>>()
                .map(|coordinates| {
                    groebner_kernel::SparseModuleElement::from_coordinates(coordinates, prime)
                })
        })
        .collect()
}

fn specialized_f2_schreyer_output<C: Field>(
    basis: &[PolynomialModule<C>],
    ring: &PolynomialRing<C>,
    kernel_order: &groebner_kernel::SparseModuleOrder,
    hilbert_target: Option<&groebner_kernel::StandardModuleHilbertTarget>,
) -> Result<Option<groebner_kernel::ModuleSyzygyOutput<u8>>, ModuleGroebnerError> {
    let Some(basis) = to_specialized_kernel_module_elements(
        basis,
        ring,
        kernel_order,
        &groebner_kernel::BinaryField,
        |coefficient| Some(u8::from(!coefficient.is_zero())),
    ) else {
        return Ok(None);
    };
    let options = groebner_kernel::GroebnerOptions::default();
    let output = match hilbert_target.map_or_else(
        || groebner_kernel::module_schreyer_syzygies_f2(&basis, kernel_order, &options),
        |target| {
            groebner_kernel::module_schreyer_syzygies_f2_with_hilbert_target(
                &basis,
                kernel_order,
                &options,
                target,
            )
            .or_else(|error| match error {
                groebner_kernel::GroebnerError::VerificationFailed => {
                    groebner_kernel::module_schreyer_syzygies_f2(&basis, kernel_order, &options)
                }
                _ => Err(error),
            })
        },
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    Ok(Some(output))
}

fn specialized_binary_extension_schreyer_output<C: Field>(
    basis: &[PolynomialModule<C>],
    ring: &PolynomialRing<C>,
    kernel_order: &groebner_kernel::SparseModuleOrder,
    binary_field: &groebner_kernel::BinaryExtensionField16,
    hilbert_target: Option<&groebner_kernel::StandardModuleHilbertTarget>,
) -> Result<Option<groebner_kernel::ModuleSyzygyOutput<u16>>, ModuleGroebnerError> {
    let Some(basis) = to_specialized_kernel_module_elements(
        basis,
        ring,
        kernel_order,
        binary_field,
        encode_binary_extension_coefficient,
    ) else {
        return Ok(None);
    };
    let options = groebner_kernel::GroebnerOptions::default();
    let output = match hilbert_target.map_or_else(
        || {
            groebner_kernel::module_schreyer_syzygies_binary_extension(
                binary_field,
                &basis,
                kernel_order,
                &options,
            )
        },
        |target| {
            groebner_kernel::module_schreyer_syzygies_binary_extension_with_hilbert_target(
                binary_field,
                &basis,
                kernel_order,
                &options,
                target,
            )
            .or_else(|error| match error {
                groebner_kernel::GroebnerError::VerificationFailed => {
                    groebner_kernel::module_schreyer_syzygies_binary_extension(
                        binary_field,
                        &basis,
                        kernel_order,
                        &options,
                    )
                }
                _ => Err(error),
            })
        },
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    Ok(Some(output))
}

fn project_specialized_schreyer_relations<C, K, Decode>(
    output: groebner_kernel::ModuleSyzygyOutput<K::Element>,
    source: &FreeModule<PolynomialRing<C>>,
    kernel_field: &K,
    decode: Decode,
) -> Result<Vec<PolynomialModule<C>>, ModuleGroebnerError>
where
    C: Field,
    K: groebner_kernel::Field,
    Decode: Fn(&K::Element) -> Result<FiniteFieldElement, PolynomialGroebnerError> + Copy,
{
    let ring = source.ring();
    output
        .generators
        .into_iter()
        .map(|relation| {
            from_specialized_kernel_coordinates(ring, relation, kernel_field, decode)
                .map_err(map_polynomial_groebner_error)
                .and_then(|coordinates| {
                    element_from_coordinates(source.clone(), coordinates).map_err(Into::into)
                })
        })
        .collect()
}

fn specialized_prime_schreyer_relations<C: Field>(
    basis: &[PolynomialModule<C>],
    source: &FreeModule<PolynomialRing<C>>,
    kernel_order: &groebner_kernel::SparseModuleOrder,
    field: &FiniteField,
    characteristic: u32,
    hilbert_target: Option<&groebner_kernel::StandardModuleHilbertTarget>,
) -> Result<Option<Vec<PolynomialModule<C>>>, ModuleGroebnerError> {
    let Some((prime, output)) = specialized_prime_schreyer_output(
        basis,
        source.ring(),
        kernel_order,
        characteristic,
        hilbert_target,
    )?
    else {
        return Ok(None);
    };
    let ring = source.ring();
    output
        .generators
        .into_iter()
        .map(|relation| {
            let coordinates = relation
                .coordinates(&prime)
                .into_iter()
                .map(|coordinate| from_prime_kernel_polynomial(ring, coordinate, field))
                .collect::<Result<Vec<_>, _>>()
                .map_err(map_polynomial_groebner_error)?;
            element_from_coordinates(source.clone(), coordinates).map_err(Into::into)
        })
        .collect::<Result<Vec<_>, ModuleGroebnerError>>()
        .map(Some)
}

fn specialized_prime_schreyer_submodule<C: Field>(
    basis: &[PolynomialModule<C>],
    ambient: &FreeModule<PolynomialRing<C>>,
    kernel_order: &groebner_kernel::SparseModuleOrder,
    field: &FiniteField,
    characteristic: u32,
    hilbert_target: Option<&groebner_kernel::StandardModuleHilbertTarget>,
) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
    let Some((prime, output)) = specialized_prime_schreyer_output(
        basis,
        ambient.ring(),
        kernel_order,
        characteristic,
        hilbert_target,
    )?
    else {
        return Ok(None);
    };
    prime_module_relations_submodule(output.generators, ambient, ambient.ring(), &prime, field)
        .map(Some)
}

fn prime_module_relations_submodule<C: Field>(
    relations: Vec<groebner_kernel::SparseModuleElement<u32>>,
    ambient: &FreeModule<PolynomialRing<C>>,
    ring: &PolynomialRing<C>,
    prime: &groebner_kernel::PrimeField32,
    field: &FiniteField,
) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
    let generator_count = relations.len();
    let mut entries = Vec::new();
    for (column, relation) in relations.into_iter().enumerate() {
        if relation.rank() != ambient.rank() || relation.arity() != ring.variables().len() {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the packed Schreyer relation has an incompatible module shape",
            ));
        }
        let mut coordinates = BTreeMap::<usize, Vec<(poly_core::Monomial, u32)>>::new();
        for (term, coefficient) in relation.terms() {
            coordinates
                .entry(term.component())
                .or_default()
                .push((term.monomial().clone(), *coefficient));
        }
        for (row, terms) in coordinates {
            let polynomial = poly_core::SparsePolynomial::from_sorted_terms(
                ring.variables().len(),
                relation.polynomial_order(),
                terms,
                prime,
            );
            entries.push((
                row,
                column,
                from_prime_kernel_polynomial(ring, polynomial, field)
                    .map_err(map_polynomial_groebner_error)?,
            ));
        }
    }
    let relation_source = FreeModule::new(ring.clone(), generator_count);
    let generator_map = ModuleHomomorphism::from_sparse_matrix(
        relation_source,
        ambient.clone(),
        SparseMatrix::from_triplets(ring.clone(), ambient.rank(), generator_count, entries)
            .map_err(ModuleError::from)?,
    )?;
    Submodule::from_generator_map(ambient.clone(), generator_map).map_err(Into::into)
}

fn specialized_module_relations_submodule<C, K, Decode>(
    relations: Vec<groebner_kernel::SparseModuleElement<K::Element>>,
    ambient: &FreeModule<PolynomialRing<C>>,
    ring: &PolynomialRing<C>,
    kernel_field: &K,
    decode: Decode,
) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError>
where
    C: Field,
    K: groebner_kernel::Field,
    Decode: Fn(&K::Element) -> Result<FiniteFieldElement, PolynomialGroebnerError> + Copy,
{
    let generator_count = relations.len();
    let mut entries = Vec::new();
    for (column, relation) in relations.into_iter().enumerate() {
        if relation.rank() != ambient.rank() || relation.arity() != ring.variables().len() {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the specialized finite-field relation has an incompatible module shape",
            ));
        }
        for (row, polynomial) in relation.coordinates(kernel_field).into_iter().enumerate() {
            if polynomial.is_zero() {
                continue;
            }
            entries.push((
                row,
                column,
                from_specialized_kernel_polynomial(ring, polynomial, decode)
                    .map_err(map_polynomial_groebner_error)?,
            ));
        }
    }
    let source = FreeModule::new(ring.clone(), generator_count);
    let generator_map = ModuleHomomorphism::from_sparse_matrix(
        source,
        ambient.clone(),
        SparseMatrix::from_triplets(ring.clone(), ambient.rank(), generator_count, entries)
            .map_err(ModuleError::from)?,
    )?;
    Submodule::from_generator_map(ambient.clone(), generator_map).map_err(Into::into)
}

fn specialized_prime_schreyer_output<C: Field>(
    basis: &[PolynomialModule<C>],
    ring: &PolynomialRing<C>,
    kernel_order: &groebner_kernel::SparseModuleOrder,
    characteristic: u32,
    hilbert_target: Option<&groebner_kernel::StandardModuleHilbertTarget>,
) -> Result<
    Option<(
        groebner_kernel::PrimeField32,
        groebner_kernel::ModuleSyzygyOutput<u32>,
    )>,
    ModuleGroebnerError,
> {
    let prime = groebner_kernel::PrimeField32::new(characteristic)
        .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
    let Some(basis) = to_prime_kernel_module_elements(basis, ring, kernel_order, &prime) else {
        return Ok(None);
    };
    let options = groebner_kernel::GroebnerOptions::default();
    let output = match hilbert_target.map_or_else(
        || groebner_kernel::module_schreyer_syzygies_prime(&prime, &basis, kernel_order, &options),
        |target| {
            groebner_kernel::module_schreyer_syzygies_prime_with_hilbert_target(
                &prime,
                &basis,
                kernel_order,
                &options,
                target,
            )
            .or_else(|error| match error {
                groebner_kernel::GroebnerError::VerificationFailed => {
                    groebner_kernel::module_schreyer_syzygies_prime(
                        &prime,
                        &basis,
                        kernel_order,
                        &options,
                    )
                }
                _ => Err(error),
            })
        },
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    Ok(Some((prime, output)))
}

fn module_basis_from_specialized_output<C, K, Decode>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
    kernel_field: &K,
    output: groebner_kernel::ModuleGroebnerOutput<K::Element>,
    input_generator_count: usize,
    decode: Decode,
) -> Result<ModuleGroebnerBasis<PolynomialRing<C>>, ModuleGroebnerError>
where
    C: Field,
    K: groebner_kernel::Field,
    Decode: Fn(&K::Element) -> Result<FiniteFieldElement, PolynomialGroebnerError> + Copy,
{
    if output.basis.len() != output.transformations.len()
        || output
            .transformations
            .iter()
            .any(|row| row.rank() != input_generator_count)
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "the specialized finite-field module kernel returned an invalid change-matrix shape",
        ));
    }
    let ring = input.ambient().ring();
    let basis_values = output
        .basis
        .into_iter()
        .map(|value| {
            from_specialized_kernel_coordinates(ring, value, kernel_field, decode)
                .map_err(map_polynomial_groebner_error)
                .and_then(|coordinates| {
                    element_from_coordinates(input.ambient().clone(), coordinates)
                        .map_err(Into::into)
                })
        })
        .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
    let leading_monomials = basis_values
        .iter()
        .map(|value| {
            leading_term(value, order)
                .map(|(monomial, _)| monomial)
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "the specialized finite-field basis projected a zero generator",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let basis = submodule_from_homogeneous_generators(input.ambient(), basis_values)?;
    let original_source = input.generator_map().source().clone();
    let basis_source = basis.generator_map().source().clone();
    let mut transformation_entries = Vec::new();
    for (column, coordinates) in output.transformations.into_iter().enumerate() {
        for (row, value) in coordinates
            .coordinates(kernel_field)
            .into_iter()
            .enumerate()
        {
            if value.is_zero() {
                continue;
            }
            transformation_entries.push((
                row,
                column,
                from_specialized_kernel_polynomial(ring, value, decode)
                    .map_err(map_polynomial_groebner_error)?,
            ));
        }
    }
    let transformation = ModuleHomomorphism::from_sparse_matrix(
        basis_source,
        original_source.clone(),
        SparseMatrix::from_triplets(
            ring.clone(),
            original_source.rank(),
            basis.generator_count(),
            transformation_entries,
        )
        .map_err(ModuleError::from)?,
    )?;
    let result = ModuleGroebnerBasis {
        input: input.clone(),
        order: order.clone(),
        basis,
        transformation,
        leading_monomials,
    };
    debug_assert!(result.verify().is_ok());
    Ok(result)
}

fn specialized_f2_module_basis<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
    kernel_order: groebner_kernel::SparseModuleOrder,
    field: &FiniteField,
) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
    let ring = input.ambient().ring();
    let columns = map_columns(input.generator_map())?;
    let Some(generators) = to_specialized_kernel_module_elements(
        &columns,
        ring,
        &kernel_order,
        &groebner_kernel::BinaryField,
        |coefficient| Some(u8::from(!coefficient.is_zero())),
    ) else {
        return Ok(None);
    };
    let output = match groebner_kernel::module_groebner_basis_with_transform_f2(
        &generators,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    module_basis_from_specialized_output(
        input,
        order,
        &groebner_kernel::BinaryField,
        output,
        generators.len(),
        |coefficient| Ok(field.element(u64::from(*coefficient))),
    )
    .map(Some)
}

fn specialized_binary_extension_module_basis<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
    kernel_order: groebner_kernel::SparseModuleOrder,
    field: &FiniteField,
    binary_field: &groebner_kernel::BinaryExtensionField16,
) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
    let ring = input.ambient().ring();
    let columns = map_columns(input.generator_map())?;
    let Some(generators) = to_specialized_kernel_module_elements(
        &columns,
        ring,
        &kernel_order,
        binary_field,
        encode_binary_extension_coefficient,
    ) else {
        return Ok(None);
    };
    let output = match groebner_kernel::module_groebner_basis_with_transform_binary_extension(
        binary_field,
        &generators,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    module_basis_from_specialized_output(
        input,
        order,
        binary_field,
        output,
        generators.len(),
        |coefficient| decode_binary_extension_coefficient(field, *coefficient),
    )
    .map(Some)
}

fn specialized_prime_module_basis<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
    kernel_order: groebner_kernel::SparseModuleOrder,
    field: &FiniteField,
    characteristic: u32,
) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
    let prime = groebner_kernel::PrimeField32::new(characteristic)
        .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
    let ring = input.ambient().ring();
    let generators = map_columns(input.generator_map())?
        .into_iter()
        .map(|generator| {
            generator
                .coordinates()
                .values()
                .map(|coordinate| {
                    let terms = coordinate
                        .terms()
                        .map(|(exponent, coefficient)| {
                            let coefficient =
                                (coefficient as &dyn Any).downcast_ref::<FiniteFieldElement>()?;
                            let representative = coefficient.representative().to_u64_checked()?;
                            exponent_to_kernel(exponent)
                                .map(|monomial| (monomial, representative as u32))
                        })
                        .collect::<Option<Vec<_>>>()?;
                    Some(poly_core::SparsePolynomial::new(
                        ring.variables().len(),
                        kernel_order.base_order(),
                        terms,
                        &prime,
                    ))
                })
                .collect::<Option<Vec<_>>>()
                .map(|coordinates| {
                    groebner_kernel::SparseModuleElement::from_coordinates(coordinates, &prime)
                })
        })
        .collect::<Option<Vec<_>>>();
    let Some(generators) = generators else {
        return Ok(None);
    };
    let output = match groebner_kernel::module_groebner_basis_with_transform_prime(
        &prime,
        &generators,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    if output.basis.len() != output.transformations.len()
        || output
            .transformations
            .iter()
            .any(|row| row.rank() != generators.len())
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "the prime module kernel returned an invalid change-matrix shape",
        ));
    }
    if input.ambient().degree_shifts().is_none() {
        return project_ungraded_prime_module_basis(input, order, field, &prime, output);
    }
    let basis_values = output
        .basis
        .into_iter()
        .map(|value| {
            value
                .coordinates(&prime)
                .into_iter()
                .map(|value| from_prime_kernel_polynomial(ring, value, field))
                .collect::<Result<Vec<_>, _>>()
                .map_err(map_polynomial_groebner_error)
                .and_then(|coordinates| {
                    element_from_coordinates(input.ambient().clone(), coordinates)
                        .map_err(Into::into)
                })
        })
        .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
    let leading_monomials = basis_values
        .iter()
        .map(|value| {
            leading_term(value, order)
                .map(|(monomial, _)| monomial)
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "the packed prime basis projected a zero generator",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let basis = submodule_from_homogeneous_generators(input.ambient(), basis_values)?;
    let original_source = input.generator_map().source().clone();
    let basis_source = basis.generator_map().source().clone();
    let mut transformation_entries = Vec::new();
    for (column, coordinates) in output.transformations.into_iter().enumerate() {
        for (row, value) in coordinates.coordinates(&prime).into_iter().enumerate() {
            if value.is_zero() {
                continue;
            }
            transformation_entries.push((
                row,
                column,
                from_prime_kernel_polynomial(ring, value, field)
                    .map_err(map_polynomial_groebner_error)?,
            ));
        }
    }
    let transformation = ModuleHomomorphism::from_sparse_matrix(
        basis_source,
        original_source.clone(),
        SparseMatrix::from_triplets(
            ring.clone(),
            original_source.rank(),
            basis.generator_count(),
            transformation_entries,
        )
        .map_err(ModuleError::from)?,
    )?;
    let result = ModuleGroebnerBasis {
        input: input.clone(),
        order: order.clone(),
        basis,
        transformation,
        leading_monomials,
    };
    debug_assert!(result.verify().is_ok());
    Ok(Some(result))
}

/// Project a packed prime-field basis directly into the sparse module maps.
///
/// An ungraded free module needs no degree inference, so materializing an
/// intermediate dense coordinate vector for every basis element only creates
/// zero polynomials and then immediately removes them again. The packed
/// module terms are already canonical and grouped deterministically here.
fn project_ungraded_prime_module_basis<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
    field: &FiniteField,
    prime: &groebner_kernel::PrimeField32,
    output: groebner_kernel::ModuleGroebnerOutput<u32>,
) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
    let ring = input.ambient().ring();
    let basis_count = output.basis.len();
    let mut basis_entries = Vec::new();
    let mut leading_monomials = Vec::with_capacity(basis_count);
    for (column, value) in output.basis.into_iter().enumerate() {
        if value.rank() != input.ambient().rank() || value.arity() != ring.variables().len() {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the packed prime basis has an incompatible module shape",
            ));
        }
        let mut leading = None;
        let mut previous_component = None;
        for (term, _) in value.terms() {
            if previous_component == Some(term.component()) {
                continue;
            }
            previous_component = Some(term.component());
            let candidate = ModuleMonomial {
                exponent: term
                    .monomial()
                    .to_u32_vec()
                    .into_iter()
                    .map(Integer::from)
                    .collect(),
                component: term.component(),
            };
            if leading.as_ref().is_none_or(|current| {
                compare_module_monomials(&candidate, current, order) == Ordering::Greater
            }) {
                leading = Some(candidate);
            }
        }
        let leading = leading.ok_or(ModuleGroebnerError::InvalidEvidence(
            "the packed prime basis projected a zero generator",
        ))?;
        leading_monomials.push(leading);
        let polynomial_order = value.polynomial_order();
        let mut coordinates = BTreeMap::<usize, Vec<(poly_core::Monomial, u32)>>::new();
        for (term, coefficient) in value.terms() {
            coordinates
                .entry(term.component())
                .or_default()
                .push((term.monomial().clone(), *coefficient));
        }
        for (row, terms) in coordinates {
            let polynomial = poly_core::SparsePolynomial::from_sorted_terms(
                ring.variables().len(),
                polynomial_order,
                terms,
                prime,
            );
            basis_entries.push((
                row,
                column,
                from_prime_kernel_polynomial(ring, polynomial, field)
                    .map_err(map_polynomial_groebner_error)?,
            ));
        }
    }
    let basis_source = FreeModule::new(ring.clone(), basis_count);
    let basis_map = ModuleHomomorphism::from_sparse_matrix(
        basis_source.clone(),
        input.ambient().clone(),
        SparseMatrix::from_triplets(
            ring.clone(),
            input.ambient().rank(),
            basis_count,
            basis_entries,
        )
        .map_err(ModuleError::from)?,
    )?;
    let basis = Submodule::from_generator_map(input.ambient().clone(), basis_map)?;

    let original_source = input.generator_map().source().clone();
    let mut transformation_entries = Vec::new();
    for (column, coordinates) in output.transformations.into_iter().enumerate() {
        for (row, value) in coordinates.coordinates(prime).into_iter().enumerate() {
            if value.is_zero() {
                continue;
            }
            transformation_entries.push((
                row,
                column,
                from_prime_kernel_polynomial(ring, value, field)
                    .map_err(map_polynomial_groebner_error)?,
            ));
        }
    }
    let transformation = ModuleHomomorphism::from_sparse_matrix(
        basis_source,
        original_source.clone(),
        SparseMatrix::from_triplets(
            ring.clone(),
            original_source.rank(),
            basis_count,
            transformation_entries,
        )
        .map_err(ModuleError::from)?,
    )?;
    let result = ModuleGroebnerBasis {
        input: input.clone(),
        order: order.clone(),
        basis,
        transformation,
        leading_monomials,
    };
    debug_assert!(result.verify().is_ok());
    Ok(Some(result))
}

fn to_kernel_module_order(
    order: &ModuleMonomialOrder,
) -> Option<groebner_kernel::SparseModuleOrder> {
    match order {
        ModuleMonomialOrder::TermOverPosition(order) => Some(
            groebner_kernel::SparseModuleOrder::TermOverPosition(to_kernel_order(order)?),
        ),
        ModuleMonomialOrder::PositionOverTerm(order) => Some(
            groebner_kernel::SparseModuleOrder::PositionOverTerm(to_kernel_order(order)?),
        ),
        ModuleMonomialOrder::Schreyer {
            target_order,
            leading_monomials,
        } => Some(groebner_kernel::SparseModuleOrder::schreyer(
            to_kernel_module_order(target_order)?,
            leading_monomials
                .iter()
                .map(|leading| {
                    exponent_to_kernel(leading.exponent()).map(|monomial| {
                        groebner_kernel::SparseModuleTerm::new(monomial, leading.component())
                    })
                })
                .collect::<Option<Vec<_>>>()?,
        )),
    }
}

fn map_kernel_module_error(error: groebner_kernel::GroebnerError) -> ModuleGroebnerError {
    match error {
        groebner_kernel::GroebnerError::ResourceLimit { resource, limit } => {
            ModuleGroebnerError::ResourceLimit { resource, limit }
        }
        groebner_kernel::GroebnerError::ContextMismatch
        | groebner_kernel::GroebnerError::ExponentOverflow
        | groebner_kernel::GroebnerError::NotZeroDimensional
        | groebner_kernel::GroebnerError::VerificationFailed => {
            ModuleGroebnerError::InvalidEvidence("the packed module kernel rejected its result")
        }
    }
}

fn rank_one_kernel_basis<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<ModuleGroebnerBasis<PolynomialRing<C>>>, ModuleGroebnerError> {
    if input.ambient().rank() != 1 || order.kernel_position_order().is_none() {
        return Ok(None);
    }
    let generators = map_columns(input.generator_map())?
        .into_iter()
        .map(|generator| {
            generator
                .coordinates()
                .values()
                .next()
                .expect("a rank-one module element has one coordinate")
        })
        .collect::<Vec<_>>();
    let ring = input.ambient().ring();
    let Some(output) = kernel_groebner_basis_with_transform(ring, &generators) else {
        return Ok(None);
    };
    let output = output.map_err(map_polynomial_groebner_error)?;
    let basis = output.basis;
    let transformations = output.transformations;
    if basis.len() != transformations.len()
        || transformations
            .iter()
            .any(|row| row.len() != generators.len())
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "the rank-one polynomial kernel returned an invalid change-matrix shape",
        ));
    }
    let original_source = input.generator_map().source().clone();
    let tracked = basis
        .into_iter()
        .zip(transformations)
        .map(|(value, representation)| {
            Ok(Tracked {
                value: element_from_coordinates(input.ambient().clone(), vec![value])?,
                representation: element_from_coordinates(original_source.clone(), representation)?,
            })
        })
        .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
    let result = module_basis_from_tracked(input, order.clone(), tracked)?;

    // The kernel has already checked exact construction, reducedness, input
    // containment, and every polynomial change identity. The conversion is a
    // direct basis-major-to-column-major projection; debug builds replay its
    // orientation, while callers retain the full independent verify() method.
    debug_assert!(result
        .input
        .generator_map()
        .compose(&result.transformation)
        .is_ok_and(|reconstructed| &reconstructed == result.basis.generator_map()));
    Ok(Some(result))
}

fn map_polynomial_groebner_error(error: PolynomialGroebnerError) -> ModuleGroebnerError {
    match error {
        PolynomialGroebnerError::ContextMismatch(error) => {
            ModuleGroebnerError::ContextMismatch(error)
        }
        PolynomialGroebnerError::Polynomial(error) => ModuleGroebnerError::Polynomial(error),
        PolynomialGroebnerError::CoefficientInvariant => ModuleGroebnerError::CoefficientInvariant,
        PolynomialGroebnerError::ResourceLimit { resource, limit } => {
            ModuleGroebnerError::ResourceLimit { resource, limit }
        }
        PolynomialGroebnerError::Ideal(_)
        | PolynomialGroebnerError::InvalidBasis(_)
        | PolynomialGroebnerError::UnsupportedInvariant(_)
        | PolynomialGroebnerError::QuotientCoordinateDimension { .. }
        | PolynomialGroebnerError::QuotientVariableOutsideRange { .. }
        | PolynomialGroebnerError::QuotientCoefficientOutsideContext
        | PolynomialGroebnerError::ExternalCandidate { .. } => {
            ModuleGroebnerError::InvalidEvidence("the rank-one polynomial kernel failed")
        }
    }
}

fn homogeneous_module_degree<C: Field>(
    value: &ModuleElement<PolynomialRing<C>>,
    grading: &FreeAbelianGrading,
    shifts: &[Degree],
) -> Result<Option<Degree>, ModuleGroebnerError> {
    let mut homogeneous_degree = None;
    for (component, polynomial) in value.coordinates().values().enumerate() {
        let shift = shifts
            .get(component)
            .ok_or(ModuleGroebnerError::InvalidEvidence(
                "the ambient module has incomplete degree shifts",
            ))?;
        for (exponent, _) in polynomial.terms() {
            let degree = grading
                .degree(exponent)
                .and_then(|degree| degree.sub(shift))
                .map_err(|_| {
                    ModuleGroebnerError::InvalidEvidence(
                        "a module term has an incompatible grading degree",
                    )
                })?;
            match &homogeneous_degree {
                None => homogeneous_degree = Some(degree),
                Some(known) if known == &degree => {}
                Some(_) => {
                    return Err(ModuleGroebnerError::InvalidEvidence(
                        "the computed module basis is not homogeneous",
                    ));
                }
            }
        }
    }
    Ok(homogeneous_degree)
}

fn module_hilbert_numerator<C: Field>(
    ambient: &FreeModule<PolynomialRing<C>>,
    leading_monomials: &[ModuleMonomial],
    shifts: &[Degree],
) -> Result<Vec<(Degree, Integer)>, ModuleGroebnerError> {
    let ring = ambient.ring();
    let mut by_component = vec![Vec::new(); ambient.rank()];
    for leading in leading_monomials {
        let generators =
            by_component
                .get_mut(leading.component)
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "a leading module monomial is outside the ambient rank",
                ))?;
        generators.push(leading.exponent.clone());
    }

    let mut numerator = BTreeMap::<Degree, Integer>::new();
    for (component, generators) in by_component.into_iter().enumerate() {
        let ideal = ring.monomial_ideal(generators).map_err(|_| {
            ModuleGroebnerError::InvalidEvidence(
                "the leading module monomials do not define a monomial ideal",
            )
        })?;
        for (degree, coefficient) in ideal
            .multigraded_hilbert_numerator()
            .map_err(map_polynomial_groebner_error)?
        {
            let degree = degree.sub(&shifts[component]).map_err(|_| {
                ModuleGroebnerError::InvalidEvidence("a module shift has the wrong grading rank")
            })?;
            add_hilbert_term(&mut numerator, degree, coefficient);
            if numerator.len() > MAX_MODULE_HILBERT_TERMS {
                return Err(resource_limit(
                    "module Hilbert numerator term count",
                    MAX_MODULE_HILBERT_TERMS,
                ));
            }
        }
    }
    Ok(numerator.into_iter().collect())
}

fn tracked_hilbert_series<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    tracked: &[Tracked<C>],
    order: &ModuleMonomialOrder,
) -> Result<HilbertSeries, ModuleGroebnerError> {
    let ambient = input.ambient();
    let shifts = ambient
        .degree_shifts()
        .ok_or_else(|| ModuleError::GradingMismatch {
            operation: "module Hilbert target",
        })?;
    let leaders = tracked
        .iter()
        .map(|entry| {
            leading_term(&entry.value, order)
                .map(|(leading, _)| leading)
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "a working module basis contains zero",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    HilbertSeries::new(
        ambient.ring().grading().clone(),
        module_hilbert_numerator(ambient, &leaders, shifts)?,
    )
    .map_err(|_| ModuleGroebnerError::InvalidEvidence("the Hilbert series is not canonical"))
}

fn add_hilbert_term(
    numerator: &mut BTreeMap<Degree, Integer>,
    degree: Degree,
    coefficient: Integer,
) {
    let coefficient = numerator
        .remove(&degree)
        .map_or(coefficient.clone(), |known| &known + &coefficient);
    if !coefficient.is_zero() {
        numerator.insert(degree, coefficient);
    }
}

fn module_basis_from_tracked<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
    order: ModuleMonomialOrder,
    tracked: Vec<Tracked<C>>,
) -> Result<ModuleGroebnerBasis<PolynomialRing<C>>, ModuleGroebnerError> {
    let original_source = input.generator_map().source();
    let leading_monomials = tracked
        .iter()
        .map(|entry| {
            leading_term(&entry.value, &order)
                .map(|(monomial, _)| monomial)
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "interreduction retained zero",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (basis_values, transformation_columns): (Vec<_>, Vec<_>) = tracked
        .into_iter()
        .map(|entry| (entry.value, entry.representation.into_coordinates()))
        .unzip();
    let basis = submodule_from_homogeneous_generators(input.ambient(), basis_values)?;
    let transformation = ModuleHomomorphism::from_coordinate_columns(
        basis.generator_map().source().clone(),
        original_source.clone(),
        transformation_columns,
    )?;
    Ok(ModuleGroebnerBasis {
        input: input.clone(),
        order,
        basis,
        transformation,
        leading_monomials,
    })
}

fn module_basis_from_reduced_generators<C: Field>(
    ambient: &FreeModule<PolynomialRing<C>>,
    generators: Vec<ModuleElement<PolynomialRing<C>>>,
    order: ModuleMonomialOrder,
) -> Result<ModuleGroebnerBasis<PolynomialRing<C>>, ModuleGroebnerError> {
    let leading_monomials = generators
        .iter()
        .map(|value| {
            leading_term(value, &order)
                .map(|(monomial, _)| monomial)
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "interreduction retained zero",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let basis = submodule_from_homogeneous_generators(ambient, generators)?;
    let transformation = ModuleHomomorphism::identity(basis.generator_map().source().clone())?;
    let result = ModuleGroebnerBasis {
        input: basis.clone(),
        order,
        basis,
        transformation,
        leading_monomials,
    };
    debug_assert!(result.verify().is_ok());
    Ok(result)
}

fn submodule_from_homogeneous_generators<C: Field>(
    ambient: &FreeModule<PolynomialRing<C>>,
    generators: Vec<ModuleElement<PolynomialRing<C>>>,
) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
    let ring = ambient.ring();
    let source = if let Some(ambient_shifts) = ambient.degree_shifts() {
        if ambient.grading_rank() != Some(ring.grading().rank()) {
            return Err(ModuleError::GradingMismatch {
                operation: "homogeneous generator source",
            }
            .into());
        }
        let inferred_shifts = generators
            .iter()
            .map(|generator| {
                homogeneous_module_degree(generator, ring.grading(), ambient_shifts)?
                    .map(|degree| degree.neg())
                    .ok_or(ModuleGroebnerError::InvalidEvidence(
                        "a canonical module basis contains zero",
                    ))
            })
            .collect::<Result<Vec<_>, _>>();
        match inferred_shifts {
            Ok(shifts) => FreeModule::new(ring.clone(), generators.len())
                .with_degree_shifts(ring.grading().rank(), shifts)?,
            Err(_) => FreeModule::new(ring.clone(), generators.len()),
        }
    } else {
        FreeModule::new(ring.clone(), generators.len())
    };
    let columns = generators
        .into_iter()
        .map(ModuleElement::into_coordinates)
        .collect();
    let map = ModuleHomomorphism::from_coordinate_columns(source, ambient.clone(), columns)?;
    Submodule::from_generator_map(ambient.clone(), map).map_err(Into::into)
}

fn graded_submodule_from_homogeneous_generators<C: Field>(
    ambient: &FreeModule<PolynomialRing<C>>,
    generators: Vec<ModuleElement<PolynomialRing<C>>>,
    labels: Option<Vec<String>>,
    operation: &'static str,
) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
    let ring = ambient.ring();
    let ambient_shifts = ambient
        .degree_shifts()
        .ok_or(ModuleError::GradingMismatch { operation })?;
    if ambient.grading_rank() != Some(ring.grading().rank()) {
        return Err(ModuleError::GradingMismatch { operation }.into());
    }
    let shifts = generators
        .iter()
        .map(|generator| {
            homogeneous_module_degree(generator, ring.grading(), ambient_shifts)?
                .map(|degree| degree.neg())
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "a nonzero homogeneous generator has no degree",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source = FreeModule::with_basis_data(
        ring.clone(),
        generators.len(),
        labels,
        Some(ring.grading().rank()),
        Some(shifts),
    )?;
    let columns = generators
        .into_iter()
        .map(ModuleElement::into_coordinates)
        .collect();
    let map = ModuleHomomorphism::from_coordinate_columns(source, ambient.clone(), columns)?;
    Submodule::from_generator_map(ambient.clone(), map).map_err(Into::into)
}

fn homogeneous_generators_are_field_independent<C: Field>(
    input: &Submodule<PolynomialRing<C>>,
) -> Result<bool, ModuleGroebnerError> {
    let ring = input.ambient().ring();
    let field = ring.coefficient_ring().clone();
    let columns = input.generator_count();
    let mut coefficient_rows = BTreeMap::<(usize, Vec<Integer>), Vec<C::Element>>::new();
    for (column, generator) in map_columns(input.generator_map())?.into_iter().enumerate() {
        for (component, polynomial) in generator.coordinates().values().enumerate() {
            for (exponent, coefficient) in polynomial.terms() {
                coefficient_rows
                    .entry((component, exponent.to_vec()))
                    .or_insert_with(|| vec![field.zero(); columns])[column] = coefficient.clone();
            }
        }
    }
    let coefficients =
        Matrix::from_rows(field, coefficient_rows.into_values()).map_err(ModuleError::from)?;
    Ok(coefficients.rank().map_err(ModuleError::from)? == columns)
}

fn ordered_schreyer_pairs(
    leaders: &[ModuleMonomial],
) -> Result<Vec<(usize, usize)>, ModuleGroebnerError> {
    if let Some(packed) = packed_schreyer_pairs(leaders)? {
        return Ok(packed);
    }
    let mut pairs = Vec::<(Integer, usize, usize)>::new();
    for left in 0..leaders.len() {
        for right in (left + 1)..leaders.len() {
            if leaders[left].component != leaders[right].component {
                continue;
            }
            let degree = leaders[left]
                .exponent
                .iter()
                .zip(&leaders[right].exponent)
                .fold(Integer::zero(), |sum, (left, right)| {
                    &sum + if left >= right { left } else { right }
                });
            pairs.push((degree, left, right));
            require_pair_bound(pairs.len())?;
        }
    }
    pairs.sort();
    Ok(pairs
        .into_iter()
        .map(|(_, left, right)| (left, right))
        .collect())
}

fn packed_schreyer_pairs(
    leaders: &[ModuleMonomial],
) -> Result<Option<Vec<(usize, usize)>>, ModuleGroebnerError> {
    let Some(arity) = leaders.first().map(|leader| leader.exponent.len()) else {
        return Ok(Some(Vec::new()));
    };
    if leaders.iter().any(|leader| leader.exponent.len() != arity) {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "leading module monomials have inconsistent arity",
        ));
    }
    let mut arena = MonomialArena::with_capacity(
        arity,
        PackedMonomialOrder::GradedReverseLexicographic,
        leaders.len(),
    );
    let mut packed = Vec::with_capacity(leaders.len());
    for leader in leaders {
        let Some(exponents) = leader
            .exponent
            .iter()
            .map(|exponent| {
                exponent
                    .to_u64_checked()
                    .and_then(|value| u32::try_from(value).ok())
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let monomial = arena.intern_exponents(&exponents).map_err(|_| {
            ModuleGroebnerError::InvalidEvidence("packed Schreyer exponent overflow")
        })?;
        packed.push(PackedModuleTerm::new(monomial, leader.component));
    }
    let cells = resolution_kernel::schreyer_degree_cells(&mut arena, &packed, MAX_PENDING_PAIRS)
        .map_err(|error| match error {
            resolution_kernel::ResolutionKernelError::PairLimit { limit } => {
                resource_limit("pending Schreyer pair count", limit)
            }
            resolution_kernel::ResolutionKernelError::Monomial(_) => {
                ModuleGroebnerError::InvalidEvidence("packed Schreyer monomial operation failed")
            }
        })?;
    Ok(Some(
        cells
            .into_iter()
            .flat_map(|cell| {
                cell.pairs()
                    .iter()
                    .map(|pair| (pair.left(), pair.right()))
                    .collect::<Vec<_>>()
            })
            .collect(),
    ))
}

impl<C: Field> ModuleHomomorphism<PolynomialRing<C>> {
    /// Exact kernel as a generated submodule of this map's source.
    pub fn kernel(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
        validate_order(self.target(), &order)?;
        if let Some(kernel) = primitive_rank_two_three_kernel(self)? {
            return Ok(kernel);
        }
        if let Some(kernel) = monomial_map_kernel(self, &order)? {
            return Ok(kernel);
        }
        if let Some(kernel) = monomial_map_kernel_prime(self, &order)? {
            return Ok(kernel);
        }
        if let Some(kernel) = prime_map_kernel(self, &order)? {
            return Ok(kernel);
        }
        Submodule::from_generator_map(self.target().clone(), self.clone())?
            .groebner_basis(order)?
            .syzygies()
    }
}

/// Primitive cross-product kernel for a full-rank map `R^3 -> R^2`.
///
/// Over the UFD `R = K[x_1, ..., x_n]`, the signed maximal minors give a
/// fraction-field generator of the kernel. Dividing by their monic gcd makes
/// that vector primitive, so Gauss's lemma brings every polynomial kernel
/// vector back to an `R`-multiple of it. Native polynomial contexts provide
/// the exact multivariate gcd needed by this lane; all other contexts and
/// rank-deficient matrices retain the general Schreyer fallback.
fn primitive_rank_two_three_kernel<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
    if map.source().rank() != 3 || map.target().rank() != 2 {
        return Ok(None);
    }
    let coefficient_ring = map.source().ring().coefficient_ring();
    if (coefficient_ring as &dyn Any)
        .downcast_ref::<FiniteField>()
        .is_none()
        && (coefficient_ring as &dyn Any)
            .downcast_ref::<RationalField>()
            .is_none()
    {
        return Ok(None);
    }

    let ring = map.source().ring();
    let columns = map_columns(map)?;
    let entries = columns
        .iter()
        .map(|column| column.coordinates().values().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let determinant = |left: usize, right: usize| {
        let diagonal = ring.mul(&entries[left][0], &entries[right][1])?;
        let antidiagonal = ring.mul(&entries[right][0], &entries[left][1])?;
        ring.sub(&diagonal, &antidiagonal)
    };

    // This is the negative of the ordinary row cross product. The fixed sign
    // makes the displayed generator deterministic without affecting its span.
    let minor_12 = determinant(1, 2)?;
    let minor_02 = ring.neg(&determinant(0, 2)?)?;
    let minor_01 = determinant(0, 1)?;
    let raw = vec![
        ring.neg(&minor_12)?,
        ring.neg(&minor_02)?,
        ring.neg(&minor_01)?,
    ];
    let Some(first) = raw.iter().find(|minor| !minor.is_zero()) else {
        return Ok(None);
    };
    let nonzero = raw
        .iter()
        .filter(|minor| !minor.is_zero())
        .collect::<Vec<_>>();
    let common = if nonzero.iter().all(|minor| minor.nonzero_term_count() == 1) {
        let mut exponent = nonzero[0]
            .terms()
            .next()
            .expect("a nonzero monomial has one term")
            .0
            .to_vec();
        for minor in nonzero.iter().skip(1) {
            let minor_exponent = minor
                .terms()
                .next()
                .expect("a nonzero monomial has one term")
                .0;
            for (common, candidate) in exponent.iter_mut().zip(minor_exponent) {
                *common = (&*common).min(candidate).clone();
            }
        }
        ring.from_terms([(exponent, coefficient_ring.one())])?
    } else {
        let mut common = first.clone();
        for minor in nonzero.iter().skip(1) {
            common = common.gcd(minor)?;
            if common.is_one() {
                break;
            }
        }
        common
    };
    let coordinates = if common.is_one() {
        raw
    } else {
        raw.into_iter()
            .map(|minor| minor.exact_div(&common))
            .collect::<Result<Vec<_>, _>>()?
    };
    let relation = element_from_coordinates(map.source().clone(), coordinates)?;
    // The signed-minor identity is formal in every commutative ring. Keep an
    // exact multiplication check in debug builds and in the differential
    // tests without charging release callers for replaying that identity.
    #[cfg(debug_assertions)]
    if !is_zero(&map.apply(&relation)?) {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "a primitive maximal-minor relation does not map to zero",
        ));
    }
    Submodule::from_generators(map.source().clone(), vec![relation])
        .map(Some)
        .map_err(Into::into)
}

/// Direct syzygies for a module map whose columns are module monomials.
///
/// For each target component this is the usual pairwise lcm presentation of
/// a monomial ideal.  It is valid over every exact field, retains arbitrary
/// exponents, and avoids sending diagonal and block-monomial maps through a
/// general module Gröbner calculation.
fn monomial_map_kernel<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
    let ring = map.source().ring();
    let columns = map_columns(map)?;
    let mut leading = Vec::with_capacity(columns.len());
    for column in &columns {
        let term_count = column
            .coordinates()
            .values()
            .map(|coordinate| coordinate.nonzero_term_count())
            .sum::<usize>();
        if term_count > 1 {
            return Ok(None);
        }
        leading.push(leading_term(column, order));
    }

    let mut relations = Vec::new();
    for (index, term) in leading.iter().enumerate() {
        if term.is_none() {
            relations.push(map.source().basis_element(index)?);
        }
    }
    for left in 0..leading.len() {
        let Some((left_monomial, left_coefficient)) = &leading[left] else {
            continue;
        };
        for right in (left + 1)..leading.len() {
            let Some((right_monomial, right_coefficient)) = &leading[right] else {
                continue;
            };
            if left_monomial.component != right_monomial.component {
                continue;
            }
            require_pair_bound(relations.len().saturating_add(1))?;
            let lcm = monomial_lcm(&left_monomial.exponent, &right_monomial.exponent);
            let left_inverse = ring
                .coefficient_ring()
                .inverse(left_coefficient)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let right_inverse = ring
                .coefficient_ring()
                .inverse(right_coefficient)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let left_factor = ring.from_terms([(
                monomial_quotient(&lcm, &left_monomial.exponent),
                left_inverse,
            )])?;
            let right_factor = ring.from_terms([(
                monomial_quotient(&lcm, &right_monomial.exponent),
                right_inverse,
            )])?;
            let mut coordinates = vec![ring.zero(); map.source().rank()];
            coordinates[left] = left_factor;
            coordinates[right] = ring.neg(&right_factor)?;
            relations.push(element_from_coordinates(map.source().clone(), coordinates)?);
        }
    }
    Submodule::from_generators(map.source().clone(), relations)
        .map(Some)
        .map_err(Into::into)
}

fn polynomial_monomial_map_image<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<Vec<ModuleMonomial>>, ModuleGroebnerError> {
    let mut image = Vec::new();
    for column in map_columns(map)? {
        let term_count = column
            .coordinates()
            .values()
            .map(|coordinate| coordinate.nonzero_term_count())
            .sum::<usize>();
        if term_count > 1 {
            return Ok(None);
        }
        if let Some((monomial, _)) = leading_term(&column, order) {
            image.push(monomial);
        }
    }
    Ok(Some(image))
}

fn polynomial_monomial_map_image_contained<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
    generators: &ModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<bool>, ModuleGroebnerError> {
    let Some(image) = polynomial_monomial_map_image(map, order)? else {
        return Ok(None);
    };
    let Some(ambient) = polynomial_monomial_map_image(generators, order)? else {
        return Ok(None);
    };
    Ok(Some(monomial_generators_contained(&image, &ambient)))
}

fn polynomial_monomial_map_images_equal<C: Field>(
    left: &ModuleHomomorphism<PolynomialRing<C>>,
    right: &ModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<bool>, ModuleGroebnerError> {
    let Some(left) = polynomial_monomial_map_image(left, order)? else {
        return Ok(None);
    };
    let Some(right) = polynomial_monomial_map_image(right, order)? else {
        return Ok(None);
    };
    Ok(Some(
        monomial_generators_contained(&left, &right)
            && monomial_generators_contained(&right, &left),
    ))
}

fn monomial_generators_contained(left: &[ModuleMonomial], right: &[ModuleMonomial]) -> bool {
    left.iter().all(|generator| {
        right
            .iter()
            .any(|candidate| module_monomial_divides(candidate, generator))
    })
}

struct PresentedPreimageData<C: Field> {
    preimage_map: ModuleHomomorphism<PolynomialRing<C>>,
    preimage_basis: ModuleGroebnerBasis<PolynomialRing<C>>,
    source_relation_lift: ModuleHomomorphism<PolynomialRing<C>>,
    syzygies: Submodule<PolynomialRing<C>>,
}

struct PresentedPreimageGraph<C: Field> {
    preimage_map: ModuleHomomorphism<PolynomialRing<C>>,
    target_relation_map: ModuleHomomorphism<PolynomialRing<C>>,
}

struct PresentedPreimageLifts<C: Field> {
    graph: PresentedPreimageGraph<C>,
    preimage_basis: ModuleGroebnerBasis<PolynomialRing<C>>,
    source_relation_lift: ModuleHomomorphism<PolynomialRing<C>>,
}

impl<C: Field> PresentedModuleHomomorphism<PolynomialRing<C>> {
    /// Construct the unique retained presented-module map induced by a map on
    /// generators, computing an exact target-relation certificate.
    ///
    /// Every source presentation relation is reduced against the target
    /// presentation. Failure means that the supplied generator map does not
    /// descend to the cokernels.
    pub fn from_generator_lift(
        source: PresentedModule<PolynomialRing<C>>,
        target: PresentedModule<PolynomialRing<C>>,
        generator_lift: ModuleHomomorphism<PolynomialRing<C>>,
        order: ModuleMonomialOrder,
    ) -> Result<Self, ModuleGroebnerError> {
        if generator_lift.source() != source.generator_module()
            || generator_lift.target() != target.generator_module()
        {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module map from generator lift",
            }
            .into());
        }
        let target_relations = Submodule::from_generator_map(
            target.generator_module().clone(),
            target.presentation().clone(),
        )?
        .groebner_basis(order)?;
        let lifted_relations = generator_lift.compose(source.presentation())?;
        let columns = map_columns(&lifted_relations)?
            .into_iter()
            .map(|relation| {
                target_relations
                    .lift(&relation)?
                    .ok_or(ModuleGroebnerError::InvalidEvidence(
                        "a generator lift does not carry source relations into target relations",
                    ))
                    .map(|lift| lift.coordinates().values().collect())
            })
            .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
        let relation_lift = ModuleHomomorphism::from_coordinate_columns(
            source.relation_module().clone(),
            target.relation_module().clone(),
            columns,
        )?;
        Self::from_lifts(source, target, generator_lift, relation_lift).map_err(Into::into)
    }

    /// Decide equality of the induced quotient maps by reducing the difference
    /// of their generator lifts modulo the target relations.
    ///
    /// Structural [`PartialEq`] deliberately compares the retained lifts. This
    /// operation answers the stronger mathematical question and requires a
    /// portable term-over-position or position-over-term module order.
    pub fn is_equal(
        &self,
        rhs: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<bool, ModuleGroebnerError> {
        require_portable_presented_order(&order)?;
        if self.source() != rhs.source() || self.target() != rhs.target() {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module homomorphism equality",
            }
            .into());
        }
        let difference = self.generator_lift().sub(rhs.generator_lift())?;
        let relations = Submodule::from_generator_map(
            self.target().generator_module().clone(),
            self.target().presentation().clone(),
        )?
        .groebner_basis(order)?;
        relations.contains_all(&map_columns(&difference)?)
    }

    /// Decide whether the induced map is zero modulo the target relations.
    pub fn is_zero(&self, order: ModuleMonomialOrder) -> Result<bool, ModuleGroebnerError> {
        if map_columns(self.generator_lift())?.iter().all(is_zero) {
            return Ok(true);
        }
        require_portable_presented_order(&order)?;
        if let Some(contained) = polynomial_monomial_map_image_contained(
            self.generator_lift(),
            self.target().presentation(),
            &order,
        )? {
            return Ok(contained);
        }
        let zero = Self::zero(self.source().clone(), self.target().clone())?;
        self.is_equal(&zero, order)
    }

    /// Kernel as its exact inclusion into the source presented module.
    ///
    /// The returned map has source `ker(self)` and target `self.source()`. Its
    /// retained presentation includes both syzygies of the computed preimage
    /// and the original source relations, so it presents
    /// `A^-1(im Q) / im P` rather than the unrelated ambient preimage.
    pub fn kernel(&self, order: ModuleMonomialOrder) -> Result<Self, ModuleGroebnerError> {
        let data = presented_preimage_data(self, &order)?;
        let relations = data
            .syzygies
            .generator_map()
            .copair(&data.source_relation_lift)?;
        let kernel = PresentedModule::cokernel(relations);
        let zero = ModuleHomomorphism::zero(
            data.syzygies.generator_map().source().clone(),
            self.source().relation_module().clone(),
        )?;
        let relation_lift = zero.copair(&ModuleHomomorphism::identity(
            self.source().relation_module().clone(),
        )?)?;
        Self::from_lifts(
            kernel,
            self.source().clone(),
            data.preimage_map,
            relation_lift,
        )
        .map_err(Into::into)
    }

    /// Image as its exact inclusion into the target presented module.
    ///
    /// Its source presentation is the canonical coimage presentation
    /// `F/A^-1(im Q)`, and the relation lift proves that the inducing map lands
    /// in the target quotient.
    pub fn image(&self, order: ModuleMonomialOrder) -> Result<Self, ModuleGroebnerError> {
        let data = presented_preimage_graph(self, &order)?;
        Self::from_lifts(
            PresentedModule::cokernel(data.preimage_map),
            self.target().clone(),
            self.generator_lift().clone(),
            data.target_relation_map,
        )
        .map_err(Into::into)
    }

    /// Coimage as the exact quotient projection from the source.
    ///
    /// The target presentation is identical to the source of [`Self::image`]
    /// when both are computed with the same order.
    pub fn coimage(&self, order: ModuleMonomialOrder) -> Result<Self, ModuleGroebnerError> {
        let data = presented_preimage_lifts(self, &order)?;
        let coimage = PresentedModule::cokernel(data.graph.preimage_map);
        Self::from_lifts(
            self.source().clone(),
            coimage,
            ModuleHomomorphism::identity(self.source().generator_module().clone())?,
            data.source_relation_lift,
        )
        .map_err(Into::into)
    }

    /// Decide exactness in `incoming -> self.source() -> self.target()`.
    ///
    /// This compares the lifted incoming image plus the source relations with
    /// the exact preimage of the target relations, in both directions.
    pub fn is_exact_at_source(
        &self,
        incoming: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<bool, ModuleGroebnerError> {
        require_portable_presented_order(&order)?;
        if incoming.target() != self.source() {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module exactness",
            }
            .into());
        }
        if !self.compose(incoming)?.is_zero(order.clone())? {
            return Ok(false);
        }
        let kernel_preimage = presented_preimage_graph(self, &order)?.preimage_map;
        let incoming_preimage = self
            .source()
            .presentation()
            .copair(incoming.generator_lift())?;
        if kernel_preimage == incoming_preimage {
            return Ok(true);
        }
        if let Some(equal) =
            polynomial_monomial_map_images_equal(&kernel_preimage, &incoming_preimage, &order)?
        {
            return Ok(equal);
        }
        let kernel_basis = Submodule::from_generator_map(
            self.source().generator_module().clone(),
            kernel_preimage.clone(),
        )?
        .groebner_basis(order.clone())?;
        let image_basis = Submodule::from_generator_map(
            self.source().generator_module().clone(),
            incoming_preimage.clone(),
        )?
        .groebner_basis(order)?;
        Ok(
            kernel_basis.contains_all(&map_columns(&incoming_preimage)?)?
                && image_basis.contains_all(&map_columns(&kernel_preimage)?)?,
        )
    }

    /// Module-valued homology at `self.source()` for the complex
    /// `incoming -> self.source() -> self.target()`.
    pub fn homology(
        &self,
        incoming: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<PresentedModuleHomology<PolynomialRing<C>>, ModuleGroebnerError> {
        if incoming.target() != self.source() {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module homology",
            }
            .into());
        }
        if !self.compose(incoming)?.is_zero(order.clone())? {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "presented-module homology requires consecutive maps to compose to zero",
            ));
        }
        let data = presented_preimage_data(self, &order)?;
        let cycle_relations = data
            .syzygies
            .generator_map()
            .copair(&data.source_relation_lift)?;
        let cycles = PresentedModule::cokernel(cycle_relations);
        let zero = ModuleHomomorphism::zero(
            data.syzygies.generator_map().source().clone(),
            self.source().relation_module().clone(),
        )?;
        let cycle_relation_lift = zero.copair(&ModuleHomomorphism::identity(
            self.source().relation_module().clone(),
        )?)?;
        let cycle_inclusion = Self::from_lifts(
            cycles.clone(),
            self.source().clone(),
            data.preimage_map.clone(),
            cycle_relation_lift,
        )?;
        let boundary_generator_lift = if incoming.generator_lift() == &data.preimage_map {
            ModuleHomomorphism::identity(incoming.source().generator_module().clone())?
        } else {
            let boundary_columns = map_columns(incoming.generator_lift())?
                .into_iter()
                .map(|boundary| {
                    data.preimage_basis
                        .lift(&boundary)?
                        .ok_or(ModuleGroebnerError::InvalidEvidence(
                            "a boundary was absent from the computed cycle preimage",
                        ))
                        .map(|lift| lift.coordinates().values().collect())
                })
                .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
            ModuleHomomorphism::from_coordinate_columns(
                incoming.source().generator_module().clone(),
                cycles.generator_module().clone(),
                boundary_columns,
            )?
        };
        let direct_boundary_relation_lift = if incoming.generator_lift() == &data.preimage_map {
            let candidate = ModuleHomomorphism::zero(
                incoming.source().relation_module().clone(),
                data.syzygies.generator_map().source().clone(),
            )?
            .pair(incoming.relation_lift())?;
            (cycles.presentation().compose(&candidate)? == *incoming.source().presentation())
                .then_some(candidate)
        } else {
            None
        };
        let boundary_relation_lift = if incoming.source().relation_count() == 0 {
            ModuleHomomorphism::zero(
                incoming.source().relation_module().clone(),
                cycles.relation_module().clone(),
            )?
        } else if let Some(candidate) = direct_boundary_relation_lift {
            candidate
        } else {
            let lifted_boundary_relations =
                boundary_generator_lift.compose(incoming.source().presentation())?;
            let cycle_relation_basis = Submodule::from_generator_map(
                cycles.generator_module().clone(),
                cycles.presentation().clone(),
            )?
            .groebner_basis(order.clone())?;
            let boundary_relation_columns = map_columns(&lifted_boundary_relations)?
                .into_iter()
                .map(|relation| {
                    cycle_relation_basis
                        .lift(&relation)?
                        .ok_or(ModuleGroebnerError::InvalidEvidence(
                            "a lifted boundary relation was absent from the cycle presentation",
                        ))
                        .map(|lift| lift.coordinates().values().collect())
                })
                .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
            ModuleHomomorphism::from_coordinate_columns(
                incoming.source().relation_module().clone(),
                cycles.relation_module().clone(),
                boundary_relation_columns,
            )?
        };
        let boundary_lift = Self::from_lifts(
            incoming.source().clone(),
            cycles,
            boundary_generator_lift,
            boundary_relation_lift,
        )?;
        let quotient_projection = boundary_lift.cokernel_projection()?;
        PresentedModuleHomology::from_evidence(
            incoming.clone(),
            self.clone(),
            cycle_inclusion,
            boundary_lift,
            quotient_projection,
        )
        .map_err(Into::into)
    }
}

impl<C: Field> PresentedModule<PolynomialRing<C>> {
    /// Exact inclusion of `Hom(self, target)` into `Hom(F_0, target)`.
    ///
    /// The source is the canonical retained presentation of the `Hom` module;
    /// the inclusion and its relation lift are replayable kernel evidence.
    pub fn hom_inclusion(
        &self,
        target: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<PresentedModuleHomomorphism<PolynomialRing<C>>, ModuleGroebnerError> {
        self.hom_presentation_map(target)?.kernel(order)
    }

    /// Canonical presented module `Hom(self, target)`.
    ///
    /// Use [`Self::hom_inclusion`] when the explicit inclusion certificate is
    /// required by a larger functorial construction.
    pub fn hom(
        &self,
        target: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<Self, ModuleGroebnerError> {
        Ok(self.hom_inclusion(target, order)?.source().clone())
    }
}

impl<C: Field> PresentedModuleHomology<PolynomialRing<C>> {
    /// Induce the exact homology map of a three-term chain map.
    ///
    /// `left`, `middle`, and `right` map the incoming source, middle term, and
    /// outgoing target of `self` to the corresponding terms of `target`.
    /// Both chain squares are checked. The returned map retains a relation
    /// certificate proving that cycles map to cycles and boundaries to
    /// boundaries.
    pub fn induced_map(
        &self,
        target: &Self,
        left: &PresentedModuleHomomorphism<PolynomialRing<C>>,
        middle: &PresentedModuleHomomorphism<PolynomialRing<C>>,
        right: &PresentedModuleHomomorphism<PolynomialRing<C>>,
        order: ModuleMonomialOrder,
    ) -> Result<PresentedModuleHomomorphism<PolynomialRing<C>>, ModuleGroebnerError> {
        if left.source() != self.incoming().source()
            || left.target() != target.incoming().source()
            || middle.source() != self.outgoing().source()
            || middle.target() != target.outgoing().source()
            || right.source() != self.outgoing().target()
            || right.target() != target.outgoing().target()
        {
            return Err(ModuleError::ModuleMismatch {
                operation: "induced presented-module homology map",
            }
            .into());
        }
        let incoming_left = middle.compose(self.incoming())?;
        let incoming_right = target.incoming().compose(left)?;
        let outgoing_left = right.compose(self.outgoing())?;
        let outgoing_right = target.outgoing().compose(middle)?;
        if !incoming_left.is_equal(&incoming_right, order.clone())?
            || !outgoing_left.is_equal(&outgoing_right, order.clone())?
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the supplied maps do not form a three-term chain map",
            ));
        }

        let cycle_span = target
            .cycle_inclusion()
            .generator_lift()
            .copair(target.outgoing().source().presentation())?;
        let cycle_basis = Submodule::from_generator_map(
            target.outgoing().source().generator_module().clone(),
            cycle_span,
        )?
        .groebner_basis(order.clone())?;
        let source_cycles_to_target_middle = middle
            .compose(self.cycle_inclusion())?
            .generator_lift()
            .clone();
        let target_cycle_generators = target.cycle_inclusion().source().generator_count();
        let cycle_columns = map_columns(&source_cycles_to_target_middle)?
            .into_iter()
            .map(|cycle| {
                cycle_basis
                    .lift(&cycle)?
                    .ok_or(ModuleGroebnerError::InvalidEvidence(
                        "a chain map failed to carry a source cycle into target cycles",
                    ))
                    .map(|lift| {
                        lift.coordinates()
                            .values()
                            .take(target_cycle_generators)
                            .collect()
                    })
            })
            .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
        let cycle_generator_lift = ModuleHomomorphism::from_coordinate_columns(
            self.cycle_inclusion().source().generator_module().clone(),
            target.cycle_inclusion().source().generator_module().clone(),
            cycle_columns,
        )?;
        let homology_generator_lift = target
            .quotient_projection()
            .generator_lift()
            .compose(&cycle_generator_lift)?;
        PresentedModuleHomomorphism::<PolynomialRing<C>>::from_generator_lift(
            self.module().clone(),
            target.module().clone(),
            homology_generator_lift,
            order,
        )
    }

    /// Recompute the two exactness statements defining this homology module.
    pub fn verify(&self, order: ModuleMonomialOrder) -> Result<(), ModuleGroebnerError> {
        self.verify_structure()?;
        if !self
            .outgoing()
            .is_exact_at_source(self.cycle_inclusion(), order.clone())?
            || !self
                .cycle_inclusion()
                .compose(self.boundary_lift())?
                .is_equal(self.incoming(), order.clone())?
            || !self
                .quotient_projection()
                .is_exact_at_source(self.boundary_lift(), order)?
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "presented-module homology evidence failed exact replay",
            ));
        }
        Ok(())
    }
}

fn require_portable_presented_order(
    order: &ModuleMonomialOrder,
) -> Result<(), ModuleGroebnerError> {
    if matches!(order, ModuleMonomialOrder::Schreyer { .. }) {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "presented-module subquotient operations require a term-over-position or position-over-term order",
        ));
    }
    Ok(())
}

fn presented_preimage_graph<C: Field>(
    map: &PresentedModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<PresentedPreimageGraph<C>, ModuleGroebnerError> {
    require_portable_presented_order(order)?;
    let target_presentation = map.target().presentation();
    let negative_target_presentation = ModuleHomomorphism::zero(
        target_presentation.source().clone(),
        target_presentation.target().clone(),
    )?
    .sub(target_presentation)?;
    let graph = map.generator_lift().copair(&negative_target_presentation)?;
    let graph_kernel = graph.kernel(order.clone())?;
    let graph_generators = map_columns(graph_kernel.generator_map())?;
    let source_rank = map.source().generator_count();
    let preimage_columns = graph_generators
        .iter()
        .map(|generator| {
            generator
                .coordinates()
                .values()
                .take(source_rank)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let target_relation_columns = graph_generators
        .iter()
        .map(|generator| {
            generator
                .coordinates()
                .values()
                .skip(source_rank)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let preimage_map = ModuleHomomorphism::from_coordinate_columns(
        graph_kernel.generator_map().source().clone(),
        map.source().generator_module().clone(),
        preimage_columns,
    )?;
    let target_relation_map = ModuleHomomorphism::from_coordinate_columns(
        graph_kernel.generator_map().source().clone(),
        map.target().relation_module().clone(),
        target_relation_columns,
    )?;
    Ok(PresentedPreimageGraph {
        preimage_map,
        target_relation_map,
    })
}

fn presented_preimage_lifts<C: Field>(
    map: &PresentedModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<PresentedPreimageLifts<C>, ModuleGroebnerError> {
    let graph = presented_preimage_graph(map, order)?;
    let preimage = Submodule::from_generator_map(
        map.source().generator_module().clone(),
        graph.preimage_map.clone(),
    )?;
    let preimage_basis = preimage.groebner_basis(order.clone())?;
    let source_relation_columns = map_columns(map.source().presentation())?
        .into_iter()
        .map(|relation| {
            preimage_basis
                .lift(&relation)?
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "a source presentation relation was absent from the computed kernel preimage",
                ))
                .map(|lift| lift.coordinates().values().collect())
        })
        .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
    let source_relation_lift = ModuleHomomorphism::from_coordinate_columns(
        map.source().relation_module().clone(),
        graph.preimage_map.source().clone(),
        source_relation_columns,
    )?;
    Ok(PresentedPreimageLifts {
        graph,
        preimage_basis,
        source_relation_lift,
    })
}

fn presented_preimage_data<C: Field>(
    map: &PresentedModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<PresentedPreimageData<C>, ModuleGroebnerError> {
    let data = presented_preimage_lifts(map, order)?;
    let syzygies = data.preimage_basis.syzygies()?;
    Ok(PresentedPreimageData {
        preimage_map: data.graph.preimage_map,
        preimage_basis: data.preimage_basis,
        source_relation_lift: data.source_relation_lift,
        syzygies,
    })
}

type PolynomialQuotient<C> = QuotientRing<PolynomialRing<C>>;

impl<C: Field> Submodule<PolynomialQuotient<C>> {
    /// Complete reduced module Groebner basis over a polynomial quotient.
    ///
    /// Internally this computes the preimage in the ambient polynomial free
    /// module after adjoining every defining-ideal relation `I e_j`. Returned
    /// generators and transformations live in the quotient context; ambient
    /// ideal relations that project to zero are not exposed as fake module
    /// generators.
    pub fn groebner_basis(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<ModuleGroebnerBasis<PolynomialQuotient<C>>, ModuleGroebnerError> {
        let result = quotient_module_groebner_basis(self, order)?;
        #[cfg(debug_assertions)]
        result.verify()?;
        Ok(result)
    }
}

impl<C: Field> ModuleGroebnerBasis<PolynomialQuotient<C>> {
    /// Exact Schreyer order induced on the ordered quotient-basis generators.
    pub fn schreyer_order(&self) -> ModuleMonomialOrder {
        ModuleMonomialOrder::schreyer(self.order.clone(), self.leading_monomials.clone())
    }

    /// Reduce an ambient quotient-module element and retain both basis and
    /// original-generator coordinates.
    pub fn reduce(
        &self,
        value: &ModuleElement<PolynomialQuotient<C>>,
    ) -> Result<ModuleReduction<PolynomialQuotient<C>>, ModuleGroebnerError> {
        if value.module() != self.input.ambient() {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the reduced value belongs to another ambient module",
            ));
        }
        let lifted_basis = lift_quotient_submodule(&self.basis)?;
        let lifted_reduction = lifted_basis
            .groebner_basis(self.order.clone())?
            .reduce(&lift_quotient_element(value)?)?;
        let basis_count = self.basis.generator_count();
        let quotient = project_polynomial_coordinates(
            self.basis.generator_map().source().clone(),
            lifted_reduction
                .original_combination()
                .coordinates()
                .values()
                .take(basis_count)
                .collect(),
        )?;
        let remainder =
            project_polynomial_element(self.input.ambient().clone(), lifted_reduction.remainder())?;
        let original_combination = self.transformation.apply(&quotient)?;
        let result = ModuleReduction {
            basis: self.clone(),
            value: value.clone(),
            quotient,
            original_combination,
            remainder,
        };
        result.verify()?;
        Ok(result)
    }

    /// Canonical normal form in the ambient quotient free module.
    pub fn normal_form(
        &self,
        value: &ModuleElement<PolynomialQuotient<C>>,
    ) -> Result<ModuleElement<PolynomialQuotient<C>>, ModuleGroebnerError> {
        Ok(self.reduce(value)?.remainder().clone())
    }

    /// Whether an ambient value belongs to the generated quotient submodule.
    pub fn contains(
        &self,
        value: &ModuleElement<PolynomialQuotient<C>>,
    ) -> Result<bool, ModuleGroebnerError> {
        Ok(self.reduce(value)?.is_member())
    }

    fn contains_all(
        &self,
        values: &[ModuleElement<PolynomialQuotient<C>>],
    ) -> Result<bool, ModuleGroebnerError> {
        for value in values {
            if !self.contains(value)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Original-generator coordinates when `value` belongs to the submodule.
    pub fn lift(
        &self,
        value: &ModuleElement<PolynomialQuotient<C>>,
    ) -> Result<Option<ModuleElement<PolynomialQuotient<C>>>, ModuleGroebnerError> {
        Ok(self.reduce(value)?.lift().cloned())
    }

    /// Relations among the original quotient-module generators.
    pub fn syzygies(&self) -> Result<Submodule<PolynomialQuotient<C>>, ModuleGroebnerError> {
        // The retained Schreyer frame is indexed by the reduced basis, while
        // this kernel is taken on the original input generator map. Quotient
        // relations can make those ranks differ, so the original ambient
        // module order is the exact compatible order for this map kernel.
        self.input.generator_map().kernel(self.order.clone())
    }

    /// Independently reconstruct the lifted polynomial calculation and check
    /// the projected canonical basis and transformation.
    pub fn verify(&self) -> Result<(), ModuleGroebnerError> {
        let expected = quotient_module_groebner_basis(&self.input, self.order.clone())?;
        if self.basis.ambient() != expected.basis.ambient()
            || self.basis.generator_map() != expected.basis.generator_map()
            || self.transformation != expected.transformation
            || self.leading_monomials != expected.leading_monomials
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "a quotient-module Groebner basis failed lifted verification",
            ));
        }
        if self.input.generator_map().compose(&self.transformation)? != *self.basis.generator_map()
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the quotient-module basis transformation does not replay",
            ));
        }
        Ok(())
    }

    /// Deserialize and independently reverify a complete quotient-module
    /// basis.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

struct QuotientPresentedPreimageData<C: Field> {
    preimage_map: ModuleHomomorphism<PolynomialQuotient<C>>,
    preimage_basis: ModuleGroebnerBasis<PolynomialQuotient<C>>,
    source_relation_lift: ModuleHomomorphism<PolynomialQuotient<C>>,
    syzygies: Submodule<PolynomialQuotient<C>>,
}

struct QuotientPresentedPreimageGraph<C: Field> {
    preimage_map: ModuleHomomorphism<PolynomialQuotient<C>>,
    target_relation_map: ModuleHomomorphism<PolynomialQuotient<C>>,
}

struct QuotientPresentedPreimageLifts<C: Field> {
    graph: QuotientPresentedPreimageGraph<C>,
    preimage_basis: ModuleGroebnerBasis<PolynomialQuotient<C>>,
    source_relation_lift: ModuleHomomorphism<PolynomialQuotient<C>>,
}

impl<C: Field> PresentedModuleHomomorphism<PolynomialQuotient<C>> {
    /// Construct a quotient-module map from its generator lift, computing the
    /// exact relation certificate in the quotient context.
    pub fn from_generator_lift(
        source: PresentedModule<PolynomialQuotient<C>>,
        target: PresentedModule<PolynomialQuotient<C>>,
        generator_lift: ModuleHomomorphism<PolynomialQuotient<C>>,
        order: ModuleMonomialOrder,
    ) -> Result<Self, ModuleGroebnerError> {
        if generator_lift.source() != source.generator_module()
            || generator_lift.target() != target.generator_module()
        {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented quotient-module map from generator lift",
            }
            .into());
        }
        let target_relations = Submodule::from_generator_map(
            target.generator_module().clone(),
            target.presentation().clone(),
        )?
        .groebner_basis(order)?;
        let lifted_relations = generator_lift.compose(source.presentation())?;
        let columns = map_columns(&lifted_relations)?
            .into_iter()
            .map(|relation| {
                target_relations
                    .lift(&relation)?
                    .ok_or(ModuleGroebnerError::InvalidEvidence(
                        "a generator lift does not carry source relations into target relations",
                    ))
                    .map(|lift| lift.coordinates().values().collect())
            })
            .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
        let relation_lift = ModuleHomomorphism::from_coordinate_columns(
            source.relation_module().clone(),
            target.relation_module().clone(),
            columns,
        )?;
        Self::from_lifts(source, target, generator_lift, relation_lift).map_err(Into::into)
    }

    /// Decide equality of induced maps over a polynomial quotient by reducing
    /// their generator-lift difference modulo the target relations.
    pub fn is_equal(
        &self,
        rhs: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<bool, ModuleGroebnerError> {
        require_portable_presented_order(&order)?;
        if self.source() != rhs.source() || self.target() != rhs.target() {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module homomorphism equality",
            }
            .into());
        }
        let difference = self.generator_lift().sub(rhs.generator_lift())?;
        let relations = Submodule::from_generator_map(
            self.target().generator_module().clone(),
            self.target().presentation().clone(),
        )?
        .groebner_basis(order)?;
        relations.contains_all(&map_columns(&difference)?)
    }

    /// Decide whether this induced quotient-module map is zero.
    pub fn is_zero(&self, order: ModuleMonomialOrder) -> Result<bool, ModuleGroebnerError> {
        if map_columns(self.generator_lift())?.iter().all(is_zero) {
            return Ok(true);
        }
        require_portable_presented_order(&order)?;
        if let Some(contained) = quotient_monomial_map_image_contained(
            self.generator_lift(),
            self.target().presentation(),
            &order,
        )? {
            return Ok(contained);
        }
        let zero = Self::zero(self.source().clone(), self.target().clone())?;
        self.is_equal(&zero, order)
    }

    /// Kernel as its exact inclusion into the source quotient module.
    pub fn kernel(&self, order: ModuleMonomialOrder) -> Result<Self, ModuleGroebnerError> {
        let data = quotient_presented_preimage_data(self, &order)?;
        let relations = data
            .syzygies
            .generator_map()
            .copair(&data.source_relation_lift)?;
        let kernel = PresentedModule::cokernel(relations);
        let zero = ModuleHomomorphism::zero(
            data.syzygies.generator_map().source().clone(),
            self.source().relation_module().clone(),
        )?;
        let relation_lift = zero.copair(&ModuleHomomorphism::identity(
            self.source().relation_module().clone(),
        )?)?;
        Self::from_lifts(
            kernel,
            self.source().clone(),
            data.preimage_map,
            relation_lift,
        )
        .map_err(Into::into)
    }

    /// Image as its exact inclusion into the target quotient module.
    pub fn image(&self, order: ModuleMonomialOrder) -> Result<Self, ModuleGroebnerError> {
        let data = quotient_presented_preimage_graph(self, &order)?;
        Self::from_lifts(
            PresentedModule::cokernel(data.preimage_map),
            self.target().clone(),
            self.generator_lift().clone(),
            data.target_relation_map,
        )
        .map_err(Into::into)
    }

    /// Coimage as the exact quotient projection from the source.
    pub fn coimage(&self, order: ModuleMonomialOrder) -> Result<Self, ModuleGroebnerError> {
        let data = quotient_presented_preimage_lifts(self, &order)?;
        let coimage = PresentedModule::cokernel(data.graph.preimage_map);
        Self::from_lifts(
            self.source().clone(),
            coimage,
            ModuleHomomorphism::identity(self.source().generator_module().clone())?,
            data.source_relation_lift,
        )
        .map_err(Into::into)
    }

    /// Decide exactness in `incoming -> self.source() -> self.target()` over
    /// the quotient context.
    pub fn is_exact_at_source(
        &self,
        incoming: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<bool, ModuleGroebnerError> {
        require_portable_presented_order(&order)?;
        if incoming.target() != self.source() {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module exactness",
            }
            .into());
        }
        if !self.compose(incoming)?.is_zero(order.clone())? {
            return Ok(false);
        }
        let kernel_preimage = quotient_presented_preimage_graph(self, &order)?.preimage_map;
        let incoming_preimage = self
            .source()
            .presentation()
            .copair(incoming.generator_lift())?;
        if kernel_preimage == incoming_preimage {
            return Ok(true);
        }
        if let Some(equal) =
            quotient_monomial_map_images_equal(&kernel_preimage, &incoming_preimage, &order)?
        {
            return Ok(equal);
        }
        let kernel_basis = Submodule::from_generator_map(
            self.source().generator_module().clone(),
            kernel_preimage.clone(),
        )?
        .groebner_basis(order.clone())?;
        let image_basis = Submodule::from_generator_map(
            self.source().generator_module().clone(),
            incoming_preimage.clone(),
        )?
        .groebner_basis(order)?;
        Ok(
            kernel_basis.contains_all(&map_columns(&incoming_preimage)?)?
                && image_basis.contains_all(&map_columns(&kernel_preimage)?)?,
        )
    }

    /// Module-valued homology at `self.source()` over the polynomial quotient.
    pub fn homology(
        &self,
        incoming: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<PresentedModuleHomology<PolynomialQuotient<C>>, ModuleGroebnerError> {
        if incoming.target() != self.source() {
            return Err(ModuleError::ModuleMismatch {
                operation: "presented-module homology",
            }
            .into());
        }
        if !self.compose(incoming)?.is_zero(order.clone())? {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "presented-module homology requires consecutive maps to compose to zero",
            ));
        }
        let data = quotient_presented_preimage_data(self, &order)?;
        let cycle_relations = data
            .syzygies
            .generator_map()
            .copair(&data.source_relation_lift)?;
        let cycles = PresentedModule::cokernel(cycle_relations);
        let zero = ModuleHomomorphism::zero(
            data.syzygies.generator_map().source().clone(),
            self.source().relation_module().clone(),
        )?;
        let cycle_relation_lift = zero.copair(&ModuleHomomorphism::identity(
            self.source().relation_module().clone(),
        )?)?;
        let cycle_inclusion = Self::from_lifts(
            cycles.clone(),
            self.source().clone(),
            data.preimage_map.clone(),
            cycle_relation_lift,
        )?;
        let boundary_generator_lift = if incoming.generator_lift() == &data.preimage_map {
            ModuleHomomorphism::identity(incoming.source().generator_module().clone())?
        } else {
            let boundary_columns = map_columns(incoming.generator_lift())?
                .into_iter()
                .map(|boundary| {
                    data.preimage_basis
                        .lift(&boundary)?
                        .ok_or(ModuleGroebnerError::InvalidEvidence(
                            "a boundary was absent from the computed cycle preimage",
                        ))
                        .map(|lift| lift.coordinates().values().collect())
                })
                .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
            ModuleHomomorphism::from_coordinate_columns(
                incoming.source().generator_module().clone(),
                cycles.generator_module().clone(),
                boundary_columns,
            )?
        };
        let direct_boundary_relation_lift = if incoming.generator_lift() == &data.preimage_map {
            let candidate = ModuleHomomorphism::zero(
                incoming.source().relation_module().clone(),
                data.syzygies.generator_map().source().clone(),
            )?
            .pair(incoming.relation_lift())?;
            (cycles.presentation().compose(&candidate)? == *incoming.source().presentation())
                .then_some(candidate)
        } else {
            None
        };
        let boundary_relation_lift = if incoming.source().relation_count() == 0 {
            ModuleHomomorphism::zero(
                incoming.source().relation_module().clone(),
                cycles.relation_module().clone(),
            )?
        } else if let Some(candidate) = direct_boundary_relation_lift {
            candidate
        } else {
            let lifted_boundary_relations =
                boundary_generator_lift.compose(incoming.source().presentation())?;
            let cycle_relation_basis = Submodule::from_generator_map(
                cycles.generator_module().clone(),
                cycles.presentation().clone(),
            )?
            .groebner_basis(order.clone())?;
            let boundary_relation_columns = map_columns(&lifted_boundary_relations)?
                .into_iter()
                .map(|relation| {
                    cycle_relation_basis
                        .lift(&relation)?
                        .ok_or(ModuleGroebnerError::InvalidEvidence(
                            "a lifted boundary relation was absent from the cycle presentation",
                        ))
                        .map(|lift| lift.coordinates().values().collect())
                })
                .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
            ModuleHomomorphism::from_coordinate_columns(
                incoming.source().relation_module().clone(),
                cycles.relation_module().clone(),
                boundary_relation_columns,
            )?
        };
        let boundary_lift = Self::from_lifts(
            incoming.source().clone(),
            cycles,
            boundary_generator_lift,
            boundary_relation_lift,
        )?;
        let quotient_projection = boundary_lift.cokernel_projection()?;
        PresentedModuleHomology::from_evidence(
            incoming.clone(),
            self.clone(),
            cycle_inclusion,
            boundary_lift,
            quotient_projection,
        )
        .map_err(Into::into)
    }
}

impl<C: Field> PresentedModule<PolynomialQuotient<C>> {
    /// Exact inclusion of `Hom(self, target)` over a polynomial quotient into
    /// `Hom(F_0, target)`.
    pub fn hom_inclusion(
        &self,
        target: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<PresentedModuleHomomorphism<PolynomialQuotient<C>>, ModuleGroebnerError> {
        self.hom_presentation_map(target)?.kernel(order)
    }

    /// Canonical presented module `Hom(self, target)` over a polynomial
    /// quotient.
    pub fn hom(
        &self,
        target: &Self,
        order: ModuleMonomialOrder,
    ) -> Result<Self, ModuleGroebnerError> {
        Ok(self.hom_inclusion(target, order)?.source().clone())
    }
}

impl<C: Field> PresentedModuleHomology<PolynomialQuotient<C>> {
    /// Induce the exact quotient-module homology map of a checked three-term
    /// chain map.
    pub fn induced_map(
        &self,
        target: &Self,
        left: &PresentedModuleHomomorphism<PolynomialQuotient<C>>,
        middle: &PresentedModuleHomomorphism<PolynomialQuotient<C>>,
        right: &PresentedModuleHomomorphism<PolynomialQuotient<C>>,
        order: ModuleMonomialOrder,
    ) -> Result<PresentedModuleHomomorphism<PolynomialQuotient<C>>, ModuleGroebnerError> {
        if left.source() != self.incoming().source()
            || left.target() != target.incoming().source()
            || middle.source() != self.outgoing().source()
            || middle.target() != target.outgoing().source()
            || right.source() != self.outgoing().target()
            || right.target() != target.outgoing().target()
        {
            return Err(ModuleError::ModuleMismatch {
                operation: "induced presented quotient-module homology map",
            }
            .into());
        }
        let incoming_left = middle.compose(self.incoming())?;
        let incoming_right = target.incoming().compose(left)?;
        let outgoing_left = right.compose(self.outgoing())?;
        let outgoing_right = target.outgoing().compose(middle)?;
        if !incoming_left.is_equal(&incoming_right, order.clone())?
            || !outgoing_left.is_equal(&outgoing_right, order.clone())?
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the supplied maps do not form a three-term chain map",
            ));
        }

        let cycle_span = target
            .cycle_inclusion()
            .generator_lift()
            .copair(target.outgoing().source().presentation())?;
        let cycle_basis = Submodule::from_generator_map(
            target.outgoing().source().generator_module().clone(),
            cycle_span,
        )?
        .groebner_basis(order.clone())?;
        let source_cycles_to_target_middle = middle
            .compose(self.cycle_inclusion())?
            .generator_lift()
            .clone();
        let target_cycle_generators = target.cycle_inclusion().source().generator_count();
        let cycle_columns = map_columns(&source_cycles_to_target_middle)?
            .into_iter()
            .map(|cycle| {
                cycle_basis
                    .lift(&cycle)?
                    .ok_or(ModuleGroebnerError::InvalidEvidence(
                        "a chain map failed to carry a source cycle into target cycles",
                    ))
                    .map(|lift| {
                        lift.coordinates()
                            .values()
                            .take(target_cycle_generators)
                            .collect()
                    })
            })
            .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
        let cycle_generator_lift = ModuleHomomorphism::from_coordinate_columns(
            self.cycle_inclusion().source().generator_module().clone(),
            target.cycle_inclusion().source().generator_module().clone(),
            cycle_columns,
        )?;
        let homology_generator_lift = target
            .quotient_projection()
            .generator_lift()
            .compose(&cycle_generator_lift)?;
        PresentedModuleHomomorphism::<PolynomialQuotient<C>>::from_generator_lift(
            self.module().clone(),
            target.module().clone(),
            homology_generator_lift,
            order,
        )
    }

    /// Recompute the two exactness statements defining this quotient-module
    /// homology value.
    pub fn verify(&self, order: ModuleMonomialOrder) -> Result<(), ModuleGroebnerError> {
        self.verify_structure()?;
        if !self
            .outgoing()
            .is_exact_at_source(self.cycle_inclusion(), order.clone())?
            || !self
                .cycle_inclusion()
                .compose(self.boundary_lift())?
                .is_equal(self.incoming(), order.clone())?
            || !self
                .quotient_projection()
                .is_exact_at_source(self.boundary_lift(), order)?
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "presented-module homology evidence failed exact replay",
            ));
        }
        Ok(())
    }
}

fn quotient_presented_preimage_graph<C: Field>(
    map: &PresentedModuleHomomorphism<PolynomialQuotient<C>>,
    order: &ModuleMonomialOrder,
) -> Result<QuotientPresentedPreimageGraph<C>, ModuleGroebnerError> {
    require_portable_presented_order(order)?;
    let target_presentation = map.target().presentation();
    let negative_target_presentation = ModuleHomomorphism::zero(
        target_presentation.source().clone(),
        target_presentation.target().clone(),
    )?
    .sub(target_presentation)?;
    let graph = map.generator_lift().copair(&negative_target_presentation)?;
    let graph_kernel = graph.kernel(order.clone())?;
    let graph_generators = map_columns(graph_kernel.generator_map())?;
    let source_rank = map.source().generator_count();
    let preimage_columns = graph_generators
        .iter()
        .map(|generator| {
            generator
                .coordinates()
                .values()
                .take(source_rank)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let target_relation_columns = graph_generators
        .iter()
        .map(|generator| {
            generator
                .coordinates()
                .values()
                .skip(source_rank)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let preimage_map = ModuleHomomorphism::from_coordinate_columns(
        graph_kernel.generator_map().source().clone(),
        map.source().generator_module().clone(),
        preimage_columns,
    )?;
    let target_relation_map = ModuleHomomorphism::from_coordinate_columns(
        graph_kernel.generator_map().source().clone(),
        map.target().relation_module().clone(),
        target_relation_columns,
    )?;
    Ok(QuotientPresentedPreimageGraph {
        preimage_map,
        target_relation_map,
    })
}

fn quotient_presented_preimage_lifts<C: Field>(
    map: &PresentedModuleHomomorphism<PolynomialQuotient<C>>,
    order: &ModuleMonomialOrder,
) -> Result<QuotientPresentedPreimageLifts<C>, ModuleGroebnerError> {
    let graph = quotient_presented_preimage_graph(map, order)?;
    let preimage = Submodule::from_generator_map(
        map.source().generator_module().clone(),
        graph.preimage_map.clone(),
    )?;
    let preimage_basis = preimage.groebner_basis(order.clone())?;
    let source_relation_columns = map_columns(map.source().presentation())?
        .into_iter()
        .map(|relation| {
            preimage_basis
                .lift(&relation)?
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "a source presentation relation was absent from the computed kernel preimage",
                ))
                .map(|lift| lift.coordinates().values().collect())
        })
        .collect::<Result<Vec<Vec<_>>, ModuleGroebnerError>>()?;
    let source_relation_lift = ModuleHomomorphism::from_coordinate_columns(
        map.source().relation_module().clone(),
        graph.preimage_map.source().clone(),
        source_relation_columns,
    )?;
    Ok(QuotientPresentedPreimageLifts {
        graph,
        preimage_basis,
        source_relation_lift,
    })
}

fn quotient_presented_preimage_data<C: Field>(
    map: &PresentedModuleHomomorphism<PolynomialQuotient<C>>,
    order: &ModuleMonomialOrder,
) -> Result<QuotientPresentedPreimageData<C>, ModuleGroebnerError> {
    let data = quotient_presented_preimage_lifts(map, order)?;
    let syzygies = data.preimage_basis.syzygies()?;
    Ok(QuotientPresentedPreimageData {
        preimage_map: data.graph.preimage_map,
        preimage_basis: data.preimage_basis,
        source_relation_lift: data.source_relation_lift,
        syzygies,
    })
}

impl<C: Field> ModuleHomomorphism<PolynomialQuotient<C>> {
    /// Exact kernel over a polynomial quotient, computed through the lifted
    /// graph `[lift(self), -I e_j]` in the ambient polynomial ring.
    pub fn kernel(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<Submodule<PolynomialQuotient<C>>, ModuleGroebnerError> {
        if let Some(kernel) = quotient_monomial_map_kernel(self, &order)? {
            return Ok(kernel);
        }
        let lifted_map = lift_quotient_map(self)?;
        let ideal_relations = polynomial_ideal_relation_map(self.target())?;
        let negative_relations = ModuleHomomorphism::zero(
            ideal_relations.source().clone(),
            ideal_relations.target().clone(),
        )?
        .sub(&ideal_relations)?;
        let graph = lifted_map.copair(&negative_relations)?;
        let graph_kernel = graph.kernel(order.clone())?;
        let source_rank = self.source().rank();
        let generators = map_columns(graph_kernel.generator_map())?
            .into_iter()
            .map(|generator| {
                project_polynomial_coordinates(
                    self.source().clone(),
                    generator.coordinates().values().take(source_rank).collect(),
                )
            })
            .collect::<Result<Vec<_>, ModuleGroebnerError>>()?
            .into_iter()
            .filter(|generator| !is_zero(generator))
            .collect();
        Submodule::from_generators(self.source().clone(), generators).map_err(Into::into)
    }
}

/// Direct kernel for a module-monomial map over a monomial quotient.
///
/// Pairwise lcm relations account for cancellation between source columns;
/// the monomial colon factors `q/gcd(q,m)` account for every defining
/// relation `q` that annihilates a column monomial `m`.  Target components
/// are independent.  This is the exact monomial preimage calculation, not a
/// special case tied to a particular quotient presentation or rank.
fn quotient_monomial_map_kernel<C: Field>(
    map: &ModuleHomomorphism<PolynomialQuotient<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<Submodule<PolynomialQuotient<C>>>, ModuleGroebnerError> {
    let quotient = map.source().ring();
    if order.base_order() != &quotient.base_ring().term_order() {
        return Err(ModuleGroebnerError::OrderMismatch);
    }
    validate_schreyer_frame(
        order,
        map.source().rank(),
        quotient.base_ring().variables().len(),
    )?;
    let Some(defining_monomials) = quotient_defining_monomials(quotient) else {
        return Ok(None);
    };

    let columns = map_columns(map)?;
    let mut leading = Vec::with_capacity(columns.len());
    for column in &columns {
        let lifted = lift_quotient_element(column)?;
        let term_count = lifted
            .coordinates()
            .values()
            .map(|coordinate| coordinate.nonzero_term_count())
            .sum::<usize>();
        if term_count > 1 {
            return Ok(None);
        }
        leading.push(leading_term(&lifted, order));
    }

    let mut relations = Vec::new();
    for (index, term) in leading.iter().enumerate() {
        let Some((monomial, _)) = term else {
            relations.push(map.source().basis_element(index)?);
            continue;
        };
        for defining in &defining_monomials {
            require_pair_bound(relations.len().saturating_add(1))?;
            let exponent = defining
                .iter()
                .zip(&monomial.exponent)
                .map(|(relation, value)| {
                    if relation > value {
                        relation - value
                    } else {
                        Integer::zero()
                    }
                })
                .collect::<Vec<_>>();
            let factor = quotient_monomial_element(
                quotient,
                exponent,
                quotient.base_ring().coefficient_ring().one(),
                &defining_monomials,
            )?;
            if !factor.is_zero() {
                let mut coordinates = vec![quotient.zero(); map.source().rank()];
                coordinates[index] = factor;
                relations.push(element_from_coordinates(map.source().clone(), coordinates)?);
            }
        }
    }
    for left in 0..leading.len() {
        let Some((left_monomial, left_coefficient)) = &leading[left] else {
            continue;
        };
        for right in (left + 1)..leading.len() {
            let Some((right_monomial, right_coefficient)) = &leading[right] else {
                continue;
            };
            if left_monomial.component != right_monomial.component {
                continue;
            }
            require_pair_bound(relations.len().saturating_add(1))?;
            let lcm = monomial_lcm(&left_monomial.exponent, &right_monomial.exponent);
            let left_inverse = quotient
                .base_ring()
                .coefficient_ring()
                .inverse(left_coefficient)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let right_inverse = quotient
                .base_ring()
                .coefficient_ring()
                .inverse(right_coefficient)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let left_factor = quotient_monomial_element(
                quotient,
                monomial_quotient(&lcm, &left_monomial.exponent),
                left_inverse,
                &defining_monomials,
            )?;
            let right_factor = quotient_monomial_element(
                quotient,
                monomial_quotient(&lcm, &right_monomial.exponent),
                right_inverse,
                &defining_monomials,
            )?;
            let mut coordinates = vec![quotient.zero(); map.source().rank()];
            coordinates[left] = left_factor;
            coordinates[right] = quotient.neg(&right_factor)?;
            relations.push(element_from_coordinates(map.source().clone(), coordinates)?);
        }
    }
    Submodule::from_generators(map.source().clone(), relations)
        .map(Some)
        .map_err(Into::into)
}

fn quotient_monomial_map_image<C: Field>(
    map: &ModuleHomomorphism<PolynomialQuotient<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<Vec<ModuleMonomial>>, ModuleGroebnerError> {
    let mut image = Vec::new();
    for column in map_columns(map)? {
        let lifted = lift_quotient_element(&column)?;
        let term_count = lifted
            .coordinates()
            .values()
            .map(|coordinate| coordinate.nonzero_term_count())
            .sum::<usize>();
        if term_count > 1 {
            return Ok(None);
        }
        if let Some((monomial, _)) = leading_term(&lifted, order) {
            image.push(monomial);
        }
    }
    Ok(Some(image))
}

fn quotient_monomial_map_image_contained<C: Field>(
    map: &ModuleHomomorphism<PolynomialQuotient<C>>,
    generators: &ModuleHomomorphism<PolynomialQuotient<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<bool>, ModuleGroebnerError> {
    let Some(image) = quotient_monomial_map_image(map, order)? else {
        return Ok(None);
    };
    let Some(ambient) = quotient_monomial_map_image(generators, order)? else {
        return Ok(None);
    };
    Ok(Some(monomial_generators_contained(&image, &ambient)))
}

fn quotient_monomial_map_images_equal<C: Field>(
    left: &ModuleHomomorphism<PolynomialQuotient<C>>,
    right: &ModuleHomomorphism<PolynomialQuotient<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<bool>, ModuleGroebnerError> {
    let Some(left) = quotient_monomial_map_image(left, order)? else {
        return Ok(None);
    };
    let Some(right) = quotient_monomial_map_image(right, order)? else {
        return Ok(None);
    };
    Ok(Some(
        monomial_generators_contained(&left, &right)
            && monomial_generators_contained(&right, &left),
    ))
}

fn quotient_defining_monomials<C: Field>(
    quotient: &PolynomialQuotient<C>,
) -> Option<Vec<Vec<Integer>>> {
    quotient
        .defining_ideal()
        .generators()
        .iter()
        .map(|relation| {
            if relation.nonzero_term_count() != 1 {
                return None;
            }
            relation
                .terms()
                .next()
                .map(|(exponent, _)| exponent.to_vec())
        })
        .collect()
}

fn quotient_monomial_element<C: Field>(
    quotient: &PolynomialQuotient<C>,
    exponent: Vec<Integer>,
    coefficient: C::Element,
    defining_monomials: &[Vec<Integer>],
) -> Result<QuotientElement<PolynomialRing<C>>, ModuleGroebnerError> {
    if defining_monomials.iter().any(|defining| {
        defining
            .iter()
            .zip(&exponent)
            .all(|(relation, value)| relation <= value)
    }) {
        return Ok(quotient.zero());
    }
    let value = quotient.base_ring().from_terms([(exponent, coefficient)])?;
    quotient
        .element_from_normal_form(value)
        .map_err(ModuleError::from)
        .map_err(Into::into)
}

fn quotient_module_groebner_basis<C: Field>(
    input: &Submodule<PolynomialQuotient<C>>,
    order: ModuleMonomialOrder,
) -> Result<ModuleGroebnerBasis<PolynomialQuotient<C>>, ModuleGroebnerError> {
    if let Some(basis) = quotient_monomial_module_basis(input, &order)? {
        return Ok(basis);
    }
    let original_count = input.generator_count();
    let lifted = lift_quotient_submodule(input)?;
    let lifted_basis = lifted.groebner_basis(order.clone())?;
    let lifted_values = map_columns(lifted_basis.basis().generator_map())?;
    let lifted_transformations = map_columns(lifted_basis.transformation())?;
    let mut basis_values = Vec::new();
    let mut transformation_columns = Vec::new();
    let mut leading_monomials = Vec::new();
    for (value, transformation) in lifted_values
        .into_iter()
        .zip(lifted_transformations.into_iter())
    {
        let value = project_polynomial_element(input.ambient().clone(), &value)?;
        if is_zero(&value) {
            continue;
        }
        let transformation = project_polynomial_coordinates(
            input.generator_map().source().clone(),
            transformation
                .coordinates()
                .values()
                .take(original_count)
                .collect(),
        )?;
        let leading = leading_term(&lift_quotient_element(&value)?, &order)
            .map(|(leading, _)| leading)
            .ok_or(ModuleGroebnerError::InvalidEvidence(
                "a projected quotient-module basis contains zero",
            ))?;
        basis_values.push(value);
        transformation_columns.push(transformation.coordinates().values().collect());
        leading_monomials.push(leading);
    }
    let basis = Submodule::from_generators(input.ambient().clone(), basis_values)?;
    let transformation = ModuleHomomorphism::from_coordinate_columns(
        basis.generator_map().source().clone(),
        input.generator_map().source().clone(),
        transformation_columns,
    )?;
    if input.generator_map().compose(&transformation)? != *basis.generator_map() {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "projected quotient-module transformations do not replay",
        ));
    }
    Ok(ModuleGroebnerBasis {
        input: input.clone(),
        order,
        basis,
        transformation,
        leading_monomials,
    })
}

fn quotient_monomial_module_basis<C: Field>(
    input: &Submodule<PolynomialQuotient<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<ModuleGroebnerBasis<PolynomialQuotient<C>>>, ModuleGroebnerError> {
    let quotient = input.ambient().ring();
    if quotient_defining_monomials(quotient).is_none() {
        return Ok(None);
    }
    if order.base_order() != &quotient.base_ring().term_order() {
        return Err(ModuleGroebnerError::OrderMismatch);
    }
    validate_schreyer_frame(
        order,
        input.ambient().rank(),
        quotient.base_ring().variables().len(),
    )?;
    let generators = map_columns(input.generator_map())?;
    let mut data = Vec::with_capacity(generators.len());
    for generator in generators {
        let lifted = lift_quotient_element(&generator)?;
        let term_count = lifted
            .coordinates()
            .values()
            .map(|coordinate| coordinate.nonzero_term_count())
            .sum::<usize>();
        if term_count != 1 {
            return Ok(None);
        }
        let (leading, coefficient) = leading_term(&lifted, order).ok_or(
            ModuleGroebnerError::InvalidEvidence("a nonzero module monomial lost its leader"),
        )?;
        data.push((generator, leading, coefficient));
    }

    let mut retained = (0..data.len())
        .filter(|index| {
            !(0..data.len()).any(|other| {
                if *index == other || !module_monomial_divides(&data[other].1, &data[*index].1) {
                    return false;
                }
                data[other].1 != data[*index].1 || other < *index
            })
        })
        .collect::<Vec<_>>();
    retained
        .sort_by(|left, right| compare_module_monomials(&data[*right].1, &data[*left].1, order));

    let mut basis_values = Vec::with_capacity(retained.len());
    let mut transformation_entries = Vec::with_capacity(retained.len());
    let mut leading_monomials = Vec::with_capacity(retained.len());
    for (column, index) in retained.into_iter().enumerate() {
        let inverse = quotient
            .base_ring()
            .coefficient_ring()
            .inverse(&data[index].2)
            .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
        let scalar = quotient.element_from_normal_form(quotient.base_ring().constant(inverse)?)?;
        basis_values.push(data[index].0.scalar_mul(&scalar)?);
        transformation_entries.push((index, column, scalar));
        leading_monomials.push(data[index].1.clone());
    }
    let basis = Submodule::from_generators(input.ambient().clone(), basis_values)?;
    let transformation = ModuleHomomorphism::from_sparse_matrix(
        basis.generator_map().source().clone(),
        input.generator_map().source().clone(),
        SparseMatrix::from_triplets(
            quotient.clone(),
            data.len(),
            transformation_entries.len(),
            transformation_entries,
        )
        .map_err(ModuleError::from)?,
    )?;
    Ok(Some(ModuleGroebnerBasis {
        input: input.clone(),
        order: order.clone(),
        basis,
        transformation,
        leading_monomials,
    }))
}

fn lift_quotient_free_module<C: Field>(
    module: &FreeModule<PolynomialQuotient<C>>,
) -> Result<FreeModule<PolynomialRing<C>>, ModuleError> {
    FreeModule::with_basis_data(
        module.ring().base_ring().clone(),
        module.rank(),
        module.basis_labels().map(<[String]>::to_vec),
        module.grading_rank(),
        module.degree_shifts().map(<[Degree]>::to_vec),
    )
}

fn lift_quotient_element<C: Field>(
    value: &ModuleElement<PolynomialQuotient<C>>,
) -> Result<ModuleElement<PolynomialRing<C>>, ModuleError> {
    element_from_coordinates(
        lift_quotient_free_module(value.module())?,
        value
            .coordinates()
            .values()
            .map(|coordinate| coordinate.lift().clone())
            .collect(),
    )
}

fn project_polynomial_element<C: Field>(
    module: FreeModule<PolynomialQuotient<C>>,
    value: &ModuleElement<PolynomialRing<C>>,
) -> Result<ModuleElement<PolynomialQuotient<C>>, ModuleGroebnerError> {
    project_polynomial_coordinates(module, value.coordinates().values().collect())
}

fn project_polynomial_coordinates<C: Field>(
    module: FreeModule<PolynomialQuotient<C>>,
    coordinates: Vec<Polynomial<C>>,
) -> Result<ModuleElement<PolynomialQuotient<C>>, ModuleGroebnerError> {
    let quotient = module.ring();
    let coordinates = coordinates
        .iter()
        .map(|coordinate| quotient.element(coordinate))
        .collect::<Result<Vec<_>, _>>()?;
    element_from_coordinates(module, coordinates).map_err(Into::into)
}

fn lift_quotient_map<C: Field>(
    map: &ModuleHomomorphism<PolynomialQuotient<C>>,
) -> Result<ModuleHomomorphism<PolynomialRing<C>>, ModuleGroebnerError> {
    let columns = map_columns(map)?
        .into_iter()
        .map(|column| {
            column
                .coordinates()
                .values()
                .map(|coordinate| coordinate.lift().clone())
                .collect()
        })
        .collect();
    ModuleHomomorphism::from_coordinate_columns(
        lift_quotient_free_module(map.source())?,
        lift_quotient_free_module(map.target())?,
        columns,
    )
    .map_err(Into::into)
}

fn polynomial_ideal_relation_map<C: Field>(
    module: &FreeModule<PolynomialQuotient<C>>,
) -> Result<ModuleHomomorphism<PolynomialRing<C>>, ModuleGroebnerError> {
    let ambient = lift_quotient_free_module(module)?;
    let ideal = module.ring().defining_ideal();
    let mut generators =
        Vec::with_capacity(ambient.rank().saturating_mul(ideal.generators().len()));
    for component in 0..ambient.rank() {
        for relation in ideal.generators() {
            let mut coordinates = vec![ambient.ring().zero(); ambient.rank()];
            coordinates[component] = relation.clone();
            generators.push(element_from_coordinates(ambient.clone(), coordinates)?);
        }
    }
    Ok(Submodule::from_generators(ambient, generators)?
        .generator_map()
        .clone())
}

fn lift_quotient_submodule<C: Field>(
    input: &Submodule<PolynomialQuotient<C>>,
) -> Result<Submodule<PolynomialRing<C>>, ModuleGroebnerError> {
    let ambient = lift_quotient_free_module(input.ambient())?;
    let mut generators = map_columns(input.generator_map())?
        .into_iter()
        .map(|generator| lift_quotient_element(&generator))
        .collect::<Result<Vec<_>, _>>()?;
    generators.extend(map_columns(&polynomial_ideal_relation_map(
        input.ambient(),
    )?)?);
    Submodule::from_generators(ambient, generators).map_err(Into::into)
}

fn prime_map_kernel<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
    let ring = map.target().ring();
    let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() else {
        return Ok(None);
    };
    if field.degree() != 1 {
        return Ok(None);
    }
    let Some(characteristic) = field
        .characteristic()
        .to_u64_checked()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value < (1_u32 << 31))
    else {
        return Ok(None);
    };
    if characteristic == 2 {
        return Ok(None);
    }
    let Some(kernel_order) = to_kernel_module_order(order) else {
        return Ok(None);
    };
    let prime = groebner_kernel::PrimeField32::new(characteristic)
        .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
    let values = map_columns(map)?;
    let Some(values) = to_prime_kernel_module_elements(&values, ring, &kernel_order, &prime) else {
        return Ok(None);
    };
    let output = match groebner_kernel::module_groebner_basis_with_transform_prime(
        &prime,
        &values,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => output,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    if output.basis.len() != output.transformations.len()
        || output
            .transformations
            .iter()
            .any(|coordinates| coordinates.rank() != values.len())
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "the prime module kernel returned an invalid change-matrix shape",
        ));
    }
    let transformations = output.transformations;
    let relations = match groebner_kernel::module_relations_from_groebner_data_prime(
        &prime,
        &output.basis,
        &transformations,
        &values,
        &kernel_order,
        &groebner_kernel::GroebnerOptions::default(),
    ) {
        Ok(output) => output.generators,
        Err(groebner_kernel::GroebnerError::ContextMismatch)
        | Err(groebner_kernel::GroebnerError::ExponentOverflow)
        | Err(groebner_kernel::GroebnerError::VerificationFailed) => return Ok(None),
        Err(error) => return Err(map_kernel_module_error(error)),
    };
    let kernel = prime_module_relations_submodule(relations, map.source(), ring, &prime, field)?;
    #[cfg(debug_assertions)]
    for relation in map_columns(kernel.generator_map())? {
        if !is_zero(&map.apply(&relation)?) {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "a packed prime-map relation does not map to zero",
            ));
        }
    }
    Ok(Some(kernel))
}

fn monomial_map_kernel_prime<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<Submodule<PolynomialRing<C>>>, ModuleGroebnerError> {
    let ring = map.target().ring();
    let Some(field) = (ring.coefficient_ring() as &dyn Any).downcast_ref::<FiniteField>() else {
        return Ok(None);
    };
    if field.degree() != 1 {
        return Ok(None);
    }
    let Some(characteristic) = field
        .characteristic()
        .to_u64_checked()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value < (1_u32 << 31))
    else {
        return Ok(None);
    };
    if characteristic == 2 {
        return Ok(None);
    }
    let Some(kernel_order) = to_kernel_module_order(order) else {
        return Ok(None);
    };
    let coefficient_one = ring.coefficient_ring().one();
    let mut basis = Vec::with_capacity(map.source().rank());
    let mut leaders = Vec::with_capacity(map.source().rank());
    let mut diagonal = Vec::with_capacity(map.source().rank());
    let mut requires_scaling = false;
    for generator in map_columns(map)? {
        let mut term_count = 0_usize;
        for coordinate in generator.coordinates().values() {
            term_count = term_count.saturating_add(coordinate.terms().take(2).count());
            if term_count > 1 {
                break;
            }
        }
        if term_count != 1 {
            return Ok(None);
        }
        let (leading, coefficient) = leading_term(&generator, order).ok_or(
            ModuleGroebnerError::InvalidEvidence("a monomial map generator is zero"),
        )?;
        let (generator, scalar) = if coefficient == coefficient_one {
            (generator, None)
        } else {
            requires_scaling = true;
            let inverse = ring
                .coefficient_ring()
                .inverse(&coefficient)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let scalar = ring.constant(inverse)?;
            (generator.scalar_mul(&scalar)?, Some(scalar))
        };
        basis.push(generator);
        leaders.push(leading);
        diagonal.push(scalar);
    }
    let Some(relations) = specialized_prime_schreyer_submodule(
        &basis,
        map.source(),
        &kernel_order,
        field,
        characteristic,
        None,
    )?
    else {
        return Ok(None);
    };
    let kernel = if requires_scaling {
        let transformation = ModuleHomomorphism::from_sparse_matrix(
            map.source().clone(),
            map.source().clone(),
            SparseMatrix::from_triplets(
                ring.clone(),
                map.source().rank(),
                map.source().rank(),
                diagonal
                    .into_iter()
                    .enumerate()
                    .map(|(index, scalar)| (index, index, scalar.unwrap_or_else(|| ring.one()))),
            )
            .map_err(ModuleError::from)?,
        )?;
        let generator_map = transformation.compose(relations.generator_map())?;
        Submodule::from_generator_map(map.source().clone(), generator_map)?
    } else {
        relations
    };
    #[cfg(debug_assertions)]
    for relation in map_columns(kernel.generator_map())? {
        if !is_zero(&map.apply(&relation)?) {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "a direct monomial-map relation does not map to zero",
            ));
        }
    }
    Ok(Some(kernel))
}

impl<C: Field> ModuleElement<PolynomialRing<C>> {
    /// Leading module monomial and coefficient in the supplied exact order.
    pub fn leading_module_term(
        &self,
        order: &ModuleMonomialOrder,
    ) -> Result<Option<(ModuleMonomial, C::Element)>, ModuleGroebnerError> {
        validate_order(self.module(), order)?;
        Ok(leading_term(self, order))
    }
}

#[derive(Clone)]
struct Tracked<C: Field> {
    value: ModuleElement<PolynomialRing<C>>,
    representation: ModuleElement<PolynomialRing<C>>,
}

fn validate_order<C: Field>(
    module: &FreeModule<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<(), ModuleGroebnerError> {
    if order.base_order() != &module.ring().term_order() {
        return Err(ModuleGroebnerError::OrderMismatch);
    }
    validate_schreyer_frame(order, module.rank(), module.ring().variables().len())
}

fn validate_schreyer_frame(
    order: &ModuleMonomialOrder,
    source_rank: usize,
    arity: usize,
) -> Result<(), ModuleGroebnerError> {
    let ModuleMonomialOrder::Schreyer {
        target_order,
        leading_monomials,
    } = order
    else {
        return Ok(());
    };
    if leading_monomials.len() != source_rank {
        return Err(ModuleGroebnerError::SchreyerFrameRank {
            rank: source_rank,
            frame_len: leading_monomials.len(),
        });
    }
    if leading_monomials.iter().any(|leading| {
        leading.exponent.len() != arity || leading.exponent.iter().any(|x| x.sign() < 0)
    }) {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "a Schreyer leader has an invalid polynomial exponent vector",
        ));
    }
    if let ModuleMonomialOrder::Schreyer {
        leading_monomials: target_leaders,
        ..
    } = target_order.as_ref()
    {
        if let Some(leading) = leading_monomials
            .iter()
            .find(|leading| leading.component >= target_leaders.len())
        {
            return Err(ModuleGroebnerError::SchreyerComponent {
                component: leading.component,
                target_rank: target_leaders.len(),
            });
        }
        validate_schreyer_frame(target_order, target_leaders.len(), arity)?;
    }
    Ok(())
}

fn leading_term<C: Field>(
    value: &ModuleElement<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Option<(ModuleMonomial, C::Element)> {
    let mut leading: Option<(ModuleMonomial, C::Element)> = None;
    for (component, polynomial) in value.coordinates().values().enumerate() {
        let Some((exponent, coefficient)) = polynomial.leading_term_owned() else {
            continue;
        };
        let candidate = ModuleMonomial::new(exponent, component);
        if leading.as_ref().is_none_or(|(current, _)| {
            compare_module_monomials(&candidate, current, order) == Ordering::Greater
        }) {
            leading = Some((candidate, coefficient));
        }
    }
    leading
}

fn compare_module_monomials(
    left: &ModuleMonomial,
    right: &ModuleMonomial,
    order: &ModuleMonomialOrder,
) -> Ordering {
    let position = || right.component.cmp(&left.component);
    match order {
        ModuleMonomialOrder::TermOverPosition(base) => base
            .compare(&left.exponent, &right.exponent)
            .then_with(position),
        ModuleMonomialOrder::PositionOverTerm(base) => {
            position().then_with(|| base.compare(&left.exponent, &right.exponent))
        }
        ModuleMonomialOrder::Schreyer {
            target_order,
            leading_monomials,
        } => {
            let left_leader = &leading_monomials[left.component];
            let right_leader = &leading_monomials[right.component];
            let left_induced = ModuleMonomial::new(
                left.exponent
                    .iter()
                    .zip(&left_leader.exponent)
                    .map(|(power, leader)| power + leader)
                    .collect(),
                left_leader.component,
            );
            let right_induced = ModuleMonomial::new(
                right
                    .exponent
                    .iter()
                    .zip(&right_leader.exponent)
                    .map(|(power, leader)| power + leader)
                    .collect(),
                right_leader.component,
            );
            compare_module_monomials(&left_induced, &right_induced, target_order)
                .then_with(position)
        }
    }
}

fn module_monomial_divides(divisor: &ModuleMonomial, dividend: &ModuleMonomial) -> bool {
    divisor.component == dividend.component
        && divisor
            .exponent
            .iter()
            .zip(&dividend.exponent)
            .all(|(left, right)| left <= right)
}

fn monomial_quotient(dividend: &[Integer], divisor: &[Integer]) -> Vec<Integer> {
    dividend
        .iter()
        .zip(divisor)
        .map(|(left, right)| left - right)
        .collect()
}

fn monomial_lcm(left: &[Integer], right: &[Integer]) -> Vec<Integer> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.max(right).clone())
        .collect()
}

fn normal_form<C: Field>(
    value: &ModuleElement<PolynomialRing<C>>,
    basis: &[ModuleElement<PolynomialRing<C>>],
    order: &ModuleMonomialOrder,
    steps: &mut usize,
) -> Result<ModuleNormalForm<C>, ModuleGroebnerError> {
    let ring = ring_of(value);
    let mut work = value.clone();
    let mut remainder = ModuleElement::zero(value.module().clone());
    let mut quotients = vec![ring.zero(); basis.len()];
    while let Some((work_monomial, work_coefficient)) = leading_term(&work, order) {
        *steps = steps.saturating_add(1);
        if *steps > MAX_REDUCTION_STEPS {
            return Err(resource_limit("reduction step count", MAX_REDUCTION_STEPS));
        }
        let divisor = basis.iter().enumerate().find_map(|(index, generator)| {
            leading_term(generator, order).and_then(|(leading, coefficient)| {
                module_monomial_divides(&leading, &work_monomial).then_some((
                    index,
                    generator,
                    leading,
                    coefficient,
                ))
            })
        });
        if let Some((index, generator, leading, coefficient)) = divisor {
            let inverse = ring
                .coefficient_ring()
                .inverse(&coefficient)
                .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
            let factor_coefficient = ring.coefficient_ring().mul(&work_coefficient, &inverse)?;
            let exponent = monomial_quotient(&work_monomial.exponent, &leading.exponent);
            let factor = ring.from_terms([(exponent, factor_coefficient)])?;
            quotients[index] = ring.add(&quotients[index], &factor)?;
            work = work.sub(&generator.scalar_mul(&factor)?)?;
        } else {
            let polynomial = ring.from_terms([(work_monomial.exponent, work_coefficient)])?;
            let mut coordinates = vec![ring.zero(); value.module().rank()];
            coordinates[work_monomial.component] = polynomial;
            let term = element_from_coordinates(value.module().clone(), coordinates)?;
            remainder = remainder.add(&term)?;
            work = work.sub(&term)?;
        }
    }
    Ok((remainder, quotients))
}

fn s_element<C: Field>(
    left: &ModuleElement<PolynomialRing<C>>,
    right: &ModuleElement<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<Option<ModuleSPair<C>>, ModuleGroebnerError> {
    let Some((left_leading, left_coefficient)) = leading_term(left, order) else {
        return Ok(None);
    };
    let Some((right_leading, right_coefficient)) = leading_term(right, order) else {
        return Ok(None);
    };
    if left_leading.component != right_leading.component {
        return Ok(None);
    }
    let ring = ring_of(left);
    let lcm = monomial_lcm(&left_leading.exponent, &right_leading.exponent);
    let left_inverse = ring
        .coefficient_ring()
        .inverse(&left_coefficient)
        .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
    let right_inverse = ring
        .coefficient_ring()
        .inverse(&right_coefficient)
        .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
    let left_factor = ring.from_terms([(
        monomial_quotient(&lcm, &left_leading.exponent),
        left_inverse,
    )])?;
    let right_factor = ring.from_terms([(
        monomial_quotient(&lcm, &right_leading.exponent),
        right_inverse,
    )])?;
    let value = left
        .scalar_mul(&left_factor)?
        .sub(&right.scalar_mul(&right_factor)?)?;
    Ok(Some((value, left_factor, right_factor)))
}

fn make_monic_pair<C: Field>(
    value: ModuleElement<PolynomialRing<C>>,
    representation: ModuleElement<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> Result<MonicTrackedPair<C>, ModuleGroebnerError> {
    let (_, coefficient) = leading_term(&value, order).ok_or(
        ModuleGroebnerError::InvalidEvidence("cannot normalize a zero module element"),
    )?;
    let ring = ring_of(&value);
    let inverse = ring
        .coefficient_ring()
        .inverse(&coefficient)
        .map_err(|_| ModuleGroebnerError::CoefficientInvariant)?;
    let scalar = ring.constant(inverse)?;
    Ok((
        value.scalar_mul(&scalar)?,
        representation.scalar_mul(&scalar)?,
    ))
}

fn interreduce<C: Field>(
    mut basis: Vec<Tracked<C>>,
    order: &ModuleMonomialOrder,
    steps: &mut usize,
) -> Result<Vec<Tracked<C>>, ModuleGroebnerError> {
    sort_tracked(&mut basis, order)?;
    let mut minimal = Vec::new();
    for (index, entry) in basis.iter().enumerate() {
        let (leading, _) = leading_term(&entry.value, order).ok_or(
            ModuleGroebnerError::InvalidEvidence("a working module basis contains zero"),
        )?;
        let redundant = basis.iter().enumerate().any(|(other_index, other)| {
            if index == other_index {
                return false;
            }
            leading_term(&other.value, order).is_some_and(|(other_leading, _)| {
                module_monomial_divides(&other_leading, &leading)
                    && (other_leading != leading || other_index < index)
            })
        });
        if !redundant {
            minimal.push(entry.clone());
        }
    }
    let mut reduced: Vec<Tracked<C>> = Vec::new();
    for index in 0..minimal.len() {
        let others = minimal
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, entry)| entry.value.clone())
            .collect::<Vec<_>>();
        let (remainder, quotients) = normal_form(&minimal[index].value, &others, order, steps)?;
        if is_zero(&remainder) {
            continue;
        }
        let mut representation = minimal[index].representation.clone();
        for (quotient, other) in quotients.iter().zip(
            minimal
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, entry)| entry),
        ) {
            representation = representation.sub(&other.representation.scalar_mul(quotient)?)?;
        }
        let (remainder, representation) = make_monic_pair(remainder, representation, order)?;
        if !reduced.iter().any(|entry| entry.value == remainder) {
            reduced.push(Tracked {
                value: remainder,
                representation,
            });
        }
    }
    sort_tracked(&mut reduced, order)?;
    Ok(reduced)
}

fn sort_tracked<C: Field>(
    basis: &mut [Tracked<C>],
    order: &ModuleMonomialOrder,
) -> Result<(), ModuleGroebnerError> {
    if basis
        .iter()
        .any(|entry| leading_term(&entry.value, order).is_none())
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "a working module basis contains zero",
        ));
    }
    basis.sort_by(|left, right| {
        let left = leading_term(&left.value, order)
            .expect("nonzero basis invariant")
            .0;
        let right = leading_term(&right.value, order)
            .expect("nonzero basis invariant")
            .0;
        compare_module_monomials(&right, &left, order)
    });
    Ok(())
}

fn verify_reduced<C: Field>(
    basis: &[ModuleElement<PolynomialRing<C>>],
    leaders: &[ModuleMonomial],
) -> Result<(), ModuleGroebnerError> {
    for (index, generator) in basis.iter().enumerate() {
        for (term, _) in module_terms(generator) {
            if leaders
                .iter()
                .enumerate()
                .any(|(other, leader)| other != index && module_monomial_divides(leader, &term))
            {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "a basis term is reducible by another leading term",
                ));
            }
        }
    }
    Ok(())
}

fn module_terms<C: Field>(
    value: &ModuleElement<PolynomialRing<C>>,
) -> Vec<(ModuleMonomial, C::Element)> {
    value
        .coordinates()
        .values()
        .enumerate()
        .flat_map(|(component, polynomial)| {
            polynomial
                .terms()
                .map(|(exponent, coefficient)| {
                    (
                        ModuleMonomial::new(exponent.to_vec(), component),
                        coefficient.clone(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn leading_components_agree<C: Field>(
    left: &ModuleElement<PolynomialRing<C>>,
    right: &ModuleElement<PolynomialRing<C>>,
    order: &ModuleMonomialOrder,
) -> bool {
    leading_term(left, order)
        .zip(leading_term(right, order))
        .is_some_and(|((left, _), (right, _))| left.component == right.component)
}

pub(super) fn map_columns<R: Ring>(
    map: &ModuleHomomorphism<R>,
) -> Result<Vec<ModuleElement<R>>, ModuleError> {
    if let Some(matrix) = map.sparse_matrix() {
        let mut columns =
            vec![vec![map.target().ring().zero(); map.target().rank()]; map.source().rank()];
        for (row, offsets) in matrix.row_offsets().windows(2).enumerate() {
            for index in offsets[0]..offsets[1] {
                columns[matrix.column_indices()[index]][row] = matrix.values()[index].clone();
            }
        }
        return columns
            .into_iter()
            .map(|coordinates| element_from_coordinates(map.target().clone(), coordinates))
            .collect();
    }
    let rows = map.matrix().rows().collect::<Vec<_>>();
    (0..map.source().rank())
        .map(|column| {
            element_from_coordinates(
                map.target().clone(),
                rows.iter().map(|row| row[column].clone()).collect(),
            )
        })
        .collect()
}

fn is_sparse_unit_monomial_isomorphism<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
) -> bool {
    if map.source().rank() != map.target().rank() {
        return false;
    }
    let Some(matrix) = map.sparse_matrix() else {
        return false;
    };
    let rank = map.source().rank();
    if matrix.nnz() != rank {
        return false;
    }
    let mut seen_columns = vec![false; rank];
    for row in 0..rank {
        let start = matrix.row_offsets()[row];
        let end = matrix.row_offsets()[row + 1];
        if end != start + 1 {
            return false;
        }
        let column = matrix.column_indices()[start];
        let value = &matrix.values()[start];
        let is_unit = value.nonzero_term_count() == 1
            && value
                .terms()
                .next()
                .is_some_and(|(exponent, _)| exponent.iter().all(Integer::is_zero));
        if seen_columns[column] || !is_unit {
            return false;
        }
        seen_columns[column] = true;
    }
    true
}

fn sparse_map_has_no_zero_columns<C: Field>(map: &ModuleHomomorphism<PolynomialRing<C>>) -> bool {
    let Some(matrix) = map.sparse_matrix() else {
        return false;
    };
    let mut seen = vec![false; map.source().rank()];
    for column in matrix.column_indices() {
        seen[*column] = true;
    }
    seen.into_iter().all(|column| column)
}

fn element_from_coordinates<R: Ring>(
    module: FreeModule<R>,
    coordinates: Vec<R::Element>,
) -> Result<ModuleElement<R>, ModuleError> {
    let vector = Vector::from_values(module.ring().clone(), coordinates)?;
    ModuleElement::from_coordinates(module, vector)
}

fn is_zero<R: Ring>(value: &ModuleElement<R>) -> bool {
    value
        .coordinates()
        .values()
        .all(|coordinate| value.module().ring().is_zero(&coordinate))
}

fn ring_of<C: Field>(value: &ModuleElement<PolynomialRing<C>>) -> &PolynomialRing<C> {
    value.module().ring()
}

fn require_pair_bound(count: usize) -> Result<(), ModuleGroebnerError> {
    if count > MAX_PENDING_PAIRS {
        Err(resource_limit("pending pair count", MAX_PENDING_PAIRS))
    } else {
        Ok(())
    }
}

fn resource_limit(resource: &'static str, limit: usize) -> ModuleGroebnerError {
    ModuleGroebnerError::ResourceLimit { resource, limit }
}

#[derive(Serialize)]
#[serde(bound(serialize = "R: Ring + Serialize, R::Element: Serialize"))]
struct ModuleGroebnerBasisWireRef<'a, R: Ring> {
    #[serde(rename = "type")]
    type_id: &'static str,
    input: &'a Submodule<R>,
    order: &'a ModuleMonomialOrder,
    basis: &'a Submodule<R>,
    transformation: &'a ModuleHomomorphism<R>,
    leading_monomials: &'a [ModuleMonomial],
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct ModuleGroebnerBasisWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    input: Submodule<R>,
    order: ModuleMonomialOrder,
    basis: Submodule<R>,
    transformation: ModuleHomomorphism<R>,
    leading_monomials: Vec<ModuleMonomial>,
}

impl<R> Serialize for ModuleGroebnerBasis<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ModuleGroebnerBasisWireRef {
            type_id: Self::TYPE_ID,
            input: &self.input,
            order: &self.order,
            basis: &self.basis,
            transformation: &self.transformation,
            leading_monomials: &self.leading_monomials,
        }
        .serialize(serializer)
    }
}

impl<'de, C> Deserialize<'de> for ModuleGroebnerBasis<PolynomialRing<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModuleGroebnerBasisWire::<PolynomialRing<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/ModuleGroebnerBasis@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type noether/homalg/ModuleGroebnerBasis@1, received {}",
                wire.type_id
            )));
        }
        let result = Self {
            input: wire.input,
            order: wire.order,
            basis: wire.basis,
            transformation: wire.transformation,
            leading_monomials: wire.leading_monomials,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<'de, C> Deserialize<'de> for ModuleGroebnerBasis<PolynomialQuotient<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModuleGroebnerBasisWire::<PolynomialQuotient<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/ModuleGroebnerBasis@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type noether/homalg/ModuleGroebnerBasis@1, received {}",
                wire.type_id
            )));
        }
        let result = Self {
            input: wire.input,
            order: wire.order,
            basis: wire.basis,
            transformation: wire.transformation,
            leading_monomials: wire.leading_monomials,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

#[derive(Serialize)]
#[serde(bound(serialize = "R: Ring + Serialize, R::Element: Serialize"))]
struct ModuleReductionWireRef<'a, R: Ring> {
    #[serde(rename = "type")]
    type_id: &'static str,
    basis: &'a ModuleGroebnerBasis<R>,
    value: &'a ModuleElement<R>,
    quotient: &'a ModuleElement<R>,
    original_combination: &'a ModuleElement<R>,
    remainder: &'a ModuleElement<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>, ModuleGroebnerBasis<R>: Deserialize<'de>"
    )
)]
struct ModuleReductionWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    basis: ModuleGroebnerBasis<R>,
    value: ModuleElement<R>,
    quotient: ModuleElement<R>,
    original_combination: ModuleElement<R>,
    remainder: ModuleElement<R>,
}

impl<R> Serialize for ModuleReduction<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ModuleReductionWireRef {
            type_id: Self::TYPE_ID,
            basis: &self.basis,
            value: &self.value,
            quotient: &self.quotient,
            original_combination: &self.original_combination,
            remainder: &self.remainder,
        }
        .serialize(serializer)
    }
}

impl<'de, C> Deserialize<'de> for ModuleReduction<PolynomialRing<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModuleReductionWire::<PolynomialRing<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/ModuleReduction@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type noether/homalg/ModuleReduction@1, received {}",
                wire.type_id
            )));
        }
        let result = Self {
            basis: wire.basis,
            value: wire.value,
            quotient: wire.quotient,
            original_combination: wire.original_combination,
            remainder: wire.remainder,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<'de, C> Deserialize<'de> for ModuleReduction<PolynomialQuotient<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModuleReductionWire::<PolynomialQuotient<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/ModuleReduction@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type noether/homalg/ModuleReduction@1, received {}",
                wire.type_id
            )));
        }
        let result = Self {
            basis: wire.basis,
            value: wire.value,
            quotient: wire.quotient,
            original_combination: wire.original_combination,
            remainder: wire.remainder,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<R: Ring> fmt::Debug for ModuleGroebnerBasis<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModuleGroebnerBasis")
            .field("input_generators", &self.input.generator_count())
            .field("basis_generators", &self.basis.generator_count())
            .field("order", &self.order)
            .finish()
    }
}

impl<R: Ring> PartialEq for ModuleGroebnerBasis<R> {
    fn eq(&self, other: &Self) -> bool {
        self.input.ambient() == other.input.ambient()
            && self.input.generator_map() == other.input.generator_map()
            && self.order == other.order
            && self.basis.ambient() == other.basis.ambient()
            && self.basis.generator_map() == other.basis.generator_map()
            && self.transformation == other.transformation
            && self.leading_monomials == other.leading_monomials
    }
}

impl<R: Ring> Eq for ModuleGroebnerBasis<R> {}

impl<R: Ring> fmt::Display for ModuleGroebnerBasis<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "ModuleGroebnerBasis(input_generators={}, basis_generators={}, order={})",
            self.input.generator_count(),
            self.basis.generator_count(),
            self.order
        )
    }
}

impl<R> SemanticType for ModuleGroebnerBasis<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = "noether/homalg/ModuleGroebnerBasis@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MODULE_GROEBNER_BASIS_DESCRIPTION
    }
}

impl<R: Ring> fmt::Debug for ModuleReduction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModuleReduction")
            .field("member", &self.is_member())
            .finish()
    }
}

impl<R: Ring> PartialEq for ModuleReduction<R> {
    fn eq(&self, other: &Self) -> bool {
        self.basis == other.basis
            && self.value == other.value
            && self.quotient == other.quotient
            && self.original_combination == other.original_combination
            && self.remainder == other.remainder
    }
}

impl<R: Ring> Eq for ModuleReduction<R> {}

impl<R: Ring> fmt::Display for ModuleReduction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "ModuleReduction(member={})", self.is_member())
    }
}

impl<R> SemanticType for ModuleReduction<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = "noether/homalg/ModuleReduction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &MODULE_REDUCTION_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use crate::commalg::{
        Degree, FreeAbelianGrading, MonomialOrder, PolynomialRing, WeightedGrading,
    };
    use crate::{Rational, Q};

    use super::*;

    fn polynomial(
        ring: &PolynomialRing<crate::RationalField>,
        terms: &[(i64, i64, i64)],
    ) -> Polynomial<crate::RationalField> {
        ring.from_terms(terms.iter().map(|(x, y, coefficient)| {
            (
                vec![Integer::from(*x), Integer::from(*y)],
                Rational::from(*coefficient),
            )
        }))
        .unwrap()
    }

    fn element(
        module: &FreeModule<PolynomialRing<crate::RationalField>>,
        coordinates: Vec<Polynomial<crate::RationalField>>,
    ) -> ModuleElement<PolynomialRing<crate::RationalField>> {
        element_from_coordinates(module.clone(), coordinates).unwrap()
    }

    fn cyclic_polynomial_module(
        ring: &PolynomialRing<crate::RationalField>,
        relation: Polynomial<crate::RationalField>,
    ) -> PresentedModule<PolynomialRing<crate::RationalField>> {
        PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                FreeModule::new(ring.clone(), 1),
                FreeModule::new(ring.clone(), 1),
                Matrix::from_rows(ring.clone(), [[relation]]).unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn submodule_sums_intersections_and_containment_are_exact() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let variables = ring.generators().unwrap();
        let x = variables[0].clone();
        let y = variables[1].clone();
        let ambient = FreeModule::new(ring.clone(), 2);
        let left = Submodule::from_generators(
            ambient.clone(),
            vec![
                element(&ambient, vec![x.clone(), ring.zero()]),
                element(&ambient, vec![ring.zero(), y.clone()]),
            ],
        )
        .unwrap();
        let right = Submodule::from_generators(
            ambient.clone(),
            vec![
                element(&ambient, vec![y.clone(), ring.zero()]),
                element(&ambient, vec![ring.zero(), x.clone()]),
            ],
        )
        .unwrap();
        let xy = (&x * &y).unwrap();
        let expected_intersection = Submodule::from_generators(
            ambient.clone(),
            vec![
                element(&ambient, vec![xy.clone(), ring.zero()]),
                element(&ambient, vec![ring.zero(), xy]),
            ],
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let sum = left.sum(&right).unwrap();
        assert!(left.is_subset_of(&sum, order.clone()).unwrap());
        assert!(right.is_subset_of(&sum, order.clone()).unwrap());
        assert!(!left.is_subset_of(&right, order.clone()).unwrap());
        let intersection = left.intersection(&right, order.clone()).unwrap();
        assert!(intersection
            .is_equal(&expected_intersection, order.clone())
            .unwrap());
        assert!(intersection.is_subset_of(&left, order.clone()).unwrap());
        assert!(intersection.is_subset_of(&right, order).unwrap());
    }

    #[test]
    fn submodule_colons_and_saturation_stop_at_exact_equality() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let variables = ring.generators().unwrap();
        let x = variables[0].clone();
        let y = variables[1].clone();
        let x_squared = x.pow(2).unwrap();
        let ambient = FreeModule::new(ring.clone(), 2);
        let source = Submodule::from_generators(
            ambient.clone(),
            vec![
                element(&ambient, vec![x_squared, ring.zero()]),
                element(&ambient, vec![ring.zero(), y.clone()]),
            ],
        )
        .unwrap();
        let first_colon = Submodule::from_generators(
            ambient.clone(),
            vec![
                element(&ambient, vec![x.clone(), ring.zero()]),
                element(&ambient, vec![ring.zero(), y.clone()]),
            ],
        )
        .unwrap();
        let saturation = Submodule::from_generators(
            ambient.clone(),
            vec![
                ambient.basis_element(0).unwrap(),
                element(&ambient, vec![ring.zero(), y]),
            ],
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        assert!(source
            .colon_by_element(&x, order.clone())
            .unwrap()
            .is_equal(&first_colon, order.clone())
            .unwrap());
        assert!(source
            .saturate_by_element(&x, order.clone())
            .unwrap()
            .is_equal(&saturation, order.clone())
            .unwrap());
        assert!(source
            .colon_by_element(&ring.one(), order.clone())
            .unwrap()
            .is_equal(&source, order.clone())
            .unwrap());
        let whole_module = source
            .colon_by_element(&ring.zero(), order.clone())
            .unwrap();
        assert!(whole_module
            .is_equal(
                &Submodule::from_generator_map(
                    ambient.clone(),
                    ModuleHomomorphism::identity(ambient).unwrap(),
                )
                .unwrap(),
                order,
            )
            .unwrap());
    }

    #[test]
    fn hom_of_cyclic_polynomial_modules_is_the_verified_annihilator_kernel() {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = ring.generators().unwrap().remove(0);
        let x2 = (&x * &x).unwrap();
        let x3 = (&x2 * &x).unwrap();
        let source = cyclic_polynomial_module(&ring, x2.clone());
        let target = cyclic_polynomial_module(&ring, x3);
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let inclusion = source.hom_inclusion(&target, order.clone()).unwrap();
        inclusion.verify().unwrap();
        assert_eq!(inclusion.target(), &target);
        assert_eq!(
            inclusion.generator_lift().matrix(),
            Matrix::from_rows(ring.clone(), [[x]]).unwrap()
        );
        assert_eq!(
            inclusion.source().presentation().matrix(),
            Matrix::from_rows(ring.clone(), [[x2]]).unwrap()
        );
        assert!(source
            .hom_presentation_map(&target)
            .unwrap()
            .is_exact_at_source(&inclusion, order.clone())
            .unwrap());
        assert_eq!(
            source.hom(&target, order).unwrap(),
            inclusion.source().clone()
        );
    }

    #[test]
    fn presented_map_kernel_image_coimage_and_exactness_replay() {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = ring.generators().unwrap().remove(0);
        let x2 = (&x * &x).unwrap();
        let source = cyclic_polynomial_module(&ring, x2.clone());
        let target = cyclic_polynomial_module(&ring, x.clone());
        let one = ring.one();
        let map = PresentedModuleHomomorphism::from_lifts(
            source.clone(),
            target.clone(),
            ModuleHomomorphism::from_matrix(
                source.generator_module().clone(),
                target.generator_module().clone(),
                Matrix::from_rows(ring.clone(), [[one.clone()]]).unwrap(),
            )
            .unwrap(),
            ModuleHomomorphism::from_matrix(
                source.relation_module().clone(),
                target.relation_module().clone(),
                Matrix::from_rows(ring.clone(), [[x.clone()]]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let one_plus_x = (&one + &x).unwrap();
        let x_plus_x2 = (&x + &x2).unwrap();
        let alternative = PresentedModuleHomomorphism::from_lifts(
            source.clone(),
            target.clone(),
            ModuleHomomorphism::from_matrix(
                source.generator_module().clone(),
                target.generator_module().clone(),
                Matrix::from_rows(ring.clone(), [[one_plus_x]]).unwrap(),
            )
            .unwrap(),
            ModuleHomomorphism::from_matrix(
                source.relation_module().clone(),
                target.relation_module().clone(),
                Matrix::from_rows(ring.clone(), [[x_plus_x2]]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_ne!(map, alternative);
        assert!(map.is_equal(&alternative, order.clone()).unwrap());

        let kernel = map.kernel(order.clone()).unwrap();
        let image = map.image(order.clone()).unwrap();
        let coimage = map.coimage(order.clone()).unwrap();
        assert_eq!(kernel.target(), &source);
        assert_eq!(image.target(), &target);
        assert_eq!(coimage.source(), &source);
        assert_eq!(coimage.target(), image.source());
        assert!(map
            .compose(&kernel)
            .unwrap()
            .is_zero(order.clone())
            .unwrap());
        assert!(map
            .is_equal(&image.compose(&coimage).unwrap(), order.clone())
            .unwrap());
        assert!(map.is_exact_at_source(&kernel, order.clone()).unwrap());
        let zero_incoming =
            PresentedModuleHomomorphism::zero(source.clone(), source.clone()).unwrap();
        assert!(!map
            .is_exact_at_source(&zero_incoming, order.clone())
            .unwrap());

        let cokernel_projection = map.cokernel_projection().unwrap();
        assert_eq!(cokernel_projection.source(), &target);
        assert!(cokernel_projection
            .compose(&map)
            .unwrap()
            .is_zero(order)
            .unwrap());
    }

    #[test]
    fn presented_module_homology_retains_cycles_boundaries_and_quotient_evidence() {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = ring.generators().unwrap().remove(0);
        let relation_module = FreeModule::new(ring.clone(), 0);
        let generator_module = FreeModule::new(ring.clone(), 1);
        let free = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                relation_module.clone(),
                generator_module.clone(),
                Matrix::zeros(ring.clone(), 1, 0),
            )
            .unwrap(),
        );
        let incoming = PresentedModuleHomomorphism::from_lifts(
            free.clone(),
            free.clone(),
            ModuleHomomorphism::from_matrix(
                generator_module,
                free.generator_module().clone(),
                Matrix::from_rows(ring.clone(), [[x.clone()]]).unwrap(),
            )
            .unwrap(),
            ModuleHomomorphism::zero(relation_module.clone(), relation_module).unwrap(),
        )
        .unwrap();
        let outgoing = PresentedModuleHomomorphism::zero(free.clone(), free.clone()).unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let homology = outgoing.homology(&incoming, order.clone()).unwrap();
        let expected = cyclic_polynomial_module(&ring, x.clone());
        assert_eq!(homology.module(), &expected);
        assert_eq!(homology.cycle_inclusion().target(), &free);
        assert_eq!(homology.boundary_lift().source(), &free);
        homology.verify(order.clone()).unwrap();
        let round_trip =
            PresentedModuleHomology::<PolynomialRing<crate::RationalField>>::from_json(
                &homology.to_json().unwrap(),
            )
            .unwrap();
        round_trip.verify(order.clone()).unwrap();
        assert_eq!(round_trip, homology);

        let reconstructed = PresentedModuleHomomorphism::<
            PolynomialRing<crate::RationalField>,
        >::from_generator_lift(
            free.clone(),
            free.clone(),
            incoming.generator_lift().clone(),
            order.clone(),
        )
        .unwrap();
        assert!(reconstructed.is_equal(&incoming, order.clone()).unwrap());
        let identity = PresentedModuleHomomorphism::identity(free.clone()).unwrap();
        let induced_identity = homology
            .induced_map(&homology, &identity, &identity, &identity, order.clone())
            .unwrap();
        assert!(induced_identity
            .is_equal(
                &PresentedModuleHomomorphism::identity(homology.module().clone()).unwrap(),
                order.clone(),
            )
            .unwrap());

        let not_a_complex = incoming.homology(&incoming, order);
        assert!(matches!(
            not_a_complex,
            Err(ModuleGroebnerError::InvalidEvidence(
                "presented-module homology requires consecutive maps to compose to zero"
            ))
        ));
    }

    #[test]
    fn quotient_module_bases_membership_lifts_and_syzygies_stay_in_the_quotient() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let variables = ring.generators().unwrap();
        let x = variables[0].clone();
        let y = variables[1].clone();
        let x2 = (&x * &x).unwrap();
        let ideal = ring.ideal([x2]).unwrap();
        let quotient = ring.quotient(&ideal).unwrap();
        let ambient = FreeModule::new(quotient.clone(), 1);
        let x_bar = quotient.element(&x).unwrap();
        let y_bar = quotient.element(&y).unwrap();
        let input = Submodule::from_generators(
            ambient.clone(),
            vec![
                element_from_coordinates(ambient.clone(), vec![x_bar.clone()]).unwrap(),
                element_from_coordinates(ambient.clone(), vec![y_bar.clone()]).unwrap(),
            ],
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let basis = input.groebner_basis(order.clone()).unwrap();
        basis.verify().unwrap();
        assert!(map_columns(basis.basis().generator_map())
            .unwrap()
            .iter()
            .all(|generator| !is_zero(generator)));

        let xy = quotient.element(&(&x * &y).unwrap()).unwrap();
        let member = element_from_coordinates(ambient.clone(), vec![xy]).unwrap();
        let reduction = basis.reduce(&member).unwrap();
        assert!(reduction.is_member());
        assert_eq!(
            input
                .generator_map()
                .apply(reduction.lift().unwrap())
                .unwrap(),
            member
        );
        let reduction_round_trip = ModuleReduction::<
            QuotientRing<PolynomialRing<crate::RationalField>>,
        >::from_json(&reduction.to_json().unwrap())
        .unwrap();
        assert_eq!(reduction_round_trip, reduction);
        let one = element_from_coordinates(ambient.clone(), vec![quotient.one()]).unwrap();
        assert!(!basis.contains(&one).unwrap());

        let kernel = input.generator_map().kernel(order.clone()).unwrap();
        for relation in map_columns(kernel.generator_map()).unwrap() {
            assert!(is_zero(&input.generator_map().apply(&relation).unwrap()));
        }
        let annihilator_relation = element_from_coordinates(
            input.generator_map().source().clone(),
            vec![x_bar, quotient.zero()],
        )
        .unwrap();
        assert!(kernel
            .groebner_basis(order)
            .unwrap()
            .contains(&annihilator_relation)
            .unwrap());

        let round_trip =
            ModuleGroebnerBasis::<QuotientRing<PolynomialRing<crate::RationalField>>>::from_json(
                &basis.to_json().unwrap(),
            )
            .unwrap();
        assert_eq!(
            round_trip.canonical_bytes().unwrap(),
            basis.canonical_bytes().unwrap()
        );
    }

    #[test]
    fn monomial_quotient_module_basis_removes_redundancy_with_a_sparse_witness() {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = ring.generators().unwrap().remove(0);
        let x2 = (&x * &x).unwrap();
        let x4 = (&x2 * &x2).unwrap();
        let quotient = ring.quotient(&ring.ideal([x4]).unwrap()).unwrap();
        let ambient = FreeModule::new(quotient.clone(), 1);
        let input = Submodule::from_generators(
            ambient.clone(),
            vec![
                element_from_coordinates(ambient.clone(), vec![quotient.element(&x).unwrap()])
                    .unwrap(),
                element_from_coordinates(ambient.clone(), vec![quotient.element(&x2).unwrap()])
                    .unwrap(),
            ],
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let basis = input.groebner_basis(order).unwrap();
        basis.verify().unwrap();
        assert_eq!(basis.basis().generator_count(), 1);
        assert!(basis.transformation().is_sparse());
        assert_eq!(
            input
                .generator_map()
                .compose(basis.transformation())
                .unwrap(),
            *basis.basis().generator_map()
        );
    }

    #[test]
    fn presented_subquotients_over_a_nonreduced_polynomial_quotient_replay() {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = ring.generators().unwrap().remove(0);
        let ideal = ring.ideal([(&x * &x).unwrap()]).unwrap();
        let quotient = ring.quotient(&ideal).unwrap();
        let relations = FreeModule::new(quotient.clone(), 0);
        let generators = FreeModule::new(quotient.clone(), 1);
        let free = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                relations.clone(),
                generators.clone(),
                Matrix::zeros(quotient.clone(), 1, 0),
            )
            .unwrap(),
        );
        let multiplication_by_x = PresentedModuleHomomorphism::from_lifts(
            free.clone(),
            free.clone(),
            ModuleHomomorphism::from_matrix(
                generators,
                free.generator_module().clone(),
                Matrix::from_rows(quotient.clone(), [[quotient.element(&x).unwrap()]]).unwrap(),
            )
            .unwrap(),
            ModuleHomomorphism::zero(relations.clone(), relations).unwrap(),
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let kernel = multiplication_by_x.kernel(order.clone()).unwrap();
        let image = multiplication_by_x.image(order.clone()).unwrap();
        let coimage = multiplication_by_x.coimage(order.clone()).unwrap();
        assert!(kernel.source().generator_count() > 0);
        assert!(multiplication_by_x
            .compose(&kernel)
            .unwrap()
            .is_zero(order.clone())
            .unwrap());
        assert_eq!(coimage.target(), image.source());
        assert!(multiplication_by_x
            .is_equal(&image.compose(&coimage).unwrap(), order.clone())
            .unwrap());
        assert!(multiplication_by_x
            .is_exact_at_source(&kernel, order.clone())
            .unwrap());
        assert!(multiplication_by_x
            .cokernel_projection()
            .unwrap()
            .compose(&multiplication_by_x)
            .unwrap()
            .is_zero(order)
            .unwrap());
    }

    #[test]
    fn nilpotent_quotient_complex_has_zero_module_valued_homology() {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = ring.generators().unwrap().remove(0);
        let ideal = ring.ideal([(&x * &x).unwrap()]).unwrap();
        let quotient = ring.quotient(&ideal).unwrap();
        let relations = FreeModule::new(quotient.clone(), 0);
        let generators = FreeModule::new(quotient.clone(), 1);
        let free = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                relations.clone(),
                generators.clone(),
                Matrix::zeros(quotient.clone(), 1, 0),
            )
            .unwrap(),
        );
        let multiplication_by_x = PresentedModuleHomomorphism::from_lifts(
            free.clone(),
            free.clone(),
            ModuleHomomorphism::from_matrix(
                generators,
                free.generator_module().clone(),
                Matrix::from_rows(quotient.clone(), [[quotient.element(&x).unwrap()]]).unwrap(),
            )
            .unwrap(),
            ModuleHomomorphism::zero(relations.clone(), relations).unwrap(),
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let preimage = quotient_presented_preimage_graph(&multiplication_by_x, &order)
            .unwrap()
            .preimage_map;
        assert_eq!(preimage, multiplication_by_x.generator_lift().clone());
        let homology = multiplication_by_x
            .homology(&multiplication_by_x, order.clone())
            .unwrap();
        homology.verify(order.clone()).unwrap();
        let zero_incoming = PresentedModuleHomomorphism::zero(free.clone(), free.clone()).unwrap();
        assert!(!multiplication_by_x
            .is_exact_at_source(&zero_incoming, order.clone())
            .unwrap());
        let homology_round_trip = PresentedModuleHomology::<
            QuotientRing<PolynomialRing<crate::RationalField>>,
        >::from_json(&homology.to_json().unwrap())
        .unwrap();
        assert_eq!(homology_round_trip, homology);
        let identity = PresentedModuleHomomorphism::identity(free).unwrap();
        let induced_identity = homology
            .induced_map(&homology, &identity, &identity, &identity, order.clone())
            .unwrap();
        assert!(induced_identity
            .is_equal(
                &PresentedModuleHomomorphism::identity(homology.module().clone()).unwrap(),
                order.clone(),
            )
            .unwrap());
        let homology_relations = Submodule::from_generator_map(
            homology.module().generator_module().clone(),
            homology.module().presentation().clone(),
        )
        .unwrap()
        .groebner_basis(order)
        .unwrap();
        for basis_index in 0..homology.module().generator_count() {
            assert!(homology_relations
                .contains(
                    &homology
                        .module()
                        .generator_module()
                        .basis_element(basis_index)
                        .unwrap()
                )
                .unwrap());
        }
    }

    #[test]
    fn sparse_rational_map_projects_directly_to_canonical_kernel_columns() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let variables = ring.generators().unwrap();
        let ambient = FreeModule::new(ring.clone(), 3);
        let source = FreeModule::new(ring.clone(), 2);
        let matrix = SparseMatrix::from_triplets(
            ring.clone(),
            3,
            2,
            [
                (0, 0, (&variables[0] + &variables[1]).unwrap()),
                (2, 1, variables[1].clone()),
            ],
        )
        .unwrap();
        let map = ModuleHomomorphism::from_sparse_matrix(source, ambient, matrix).unwrap();
        let order = to_kernel_order(&ring.term_order()).unwrap();

        let direct = rational_kernel_generators_from_sparse_map(&map, order).unwrap();
        let materialized = map_columns(&map)
            .unwrap()
            .into_iter()
            .map(|generator| {
                let coordinates = generator
                    .coordinates()
                    .values()
                    .map(|coordinate| to_rational_kernel_polynomial(&coordinate, order).unwrap())
                    .collect();
                groebner_kernel::SparseModuleElement::from_coordinates(
                    coordinates,
                    &RationalKernelLane,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(direct, materialized);
    }

    #[test]
    fn minimal_homogeneous_generators_retain_an_exact_early_subset() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let variables = ring.generators().unwrap();
        let x = variables[0].clone();
        let y = variables[1].clone();
        let ambient = FreeModule::new(ring.clone(), 2)
            .with_degree_shifts(1, vec![Degree::zero(1); 2])
            .unwrap();
        let source = FreeModule::new(ring.clone(), 5)
            .with_basis_labels(vec![
                "x-e0".to_owned(),
                "y-e1".to_owned(),
                "sum".to_owned(),
                "multiple".to_owned(),
                "zero".to_owned(),
            ])
            .unwrap();
        let map = ModuleHomomorphism::from_matrix(
            source,
            ambient.clone(),
            Matrix::from_rows(
                ring.clone(),
                [
                    [
                        x.clone(),
                        ring.zero(),
                        x.clone(),
                        x.pow(2).unwrap(),
                        ring.zero(),
                    ],
                    [ring.zero(), y.clone(), y, ring.zero(), ring.zero()],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let input = Submodule::from_generator_map(ambient, map).unwrap();

        let minimal = input.minimal_homogeneous_generators().unwrap();

        assert_eq!(minimal.generator_count(), 2);
        assert_eq!(
            minimal.generator_map().source().basis_labels(),
            Some(["x-e0".to_owned(), "y-e1".to_owned()].as_slice())
        );
        assert_eq!(
            minimal.generator_map().source().degree_shifts(),
            Some(
                [
                    Degree::new(vec![Integer::from(-1)]),
                    Degree::new(vec![Integer::from(-1)]),
                ]
                .as_slice()
            )
        );
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let basis = minimal.groebner_basis(order).unwrap();
        assert!(basis
            .contains_all(&map_columns(input.generator_map()).unwrap())
            .unwrap());
    }

    #[test]
    fn minimal_homogeneous_generators_support_weights_and_reject_inhomogeneity() {
        let ring = PolynomialRing::with_grading(
            Q,
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
            WeightedGrading::new(vec![2, 3]).unwrap(),
        )
        .unwrap();
        let variables = ring.generators().unwrap();
        let x = variables[0].clone();
        let y = variables[1].clone();
        let ambient = FreeModule::new(ring.clone(), 1)
            .with_degree_shifts(1, vec![Degree::zero(1)])
            .unwrap();
        let valid = Submodule::from_generators(
            ambient.clone(),
            vec![
                element(&ambient, vec![x.clone()]),
                element(&ambient, vec![y.clone()]),
                element(&ambient, vec![x.pow(2).unwrap()]),
            ],
        )
        .unwrap();
        assert_eq!(
            valid
                .minimal_homogeneous_generators()
                .unwrap()
                .generator_count(),
            2
        );

        let invalid = Submodule::from_generators(
            ambient.clone(),
            vec![element(&ambient, vec![(&x + &y).unwrap()])],
        )
        .unwrap();
        assert!(matches!(
            invalid.minimal_homogeneous_generators(),
            Err(ModuleGroebnerError::InvalidEvidence(
                "the computed module basis is not homogeneous"
            ))
        ));

        let ungraded = FreeModule::new(ring, 1);
        let invalid =
            Submodule::from_generators(ungraded.clone(), vec![element(&ungraded, vec![x])])
                .unwrap();
        assert!(matches!(
            invalid.minimal_homogeneous_generators(),
            Err(ModuleGroebnerError::Module(ModuleError::GradingMismatch {
                operation: "minimal homogeneous generators"
            }))
        ));
    }

    #[test]
    fn module_basis_reduction_and_original_lift_verify() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let ambient = FreeModule::new(ring.clone(), 2);
        let zero = ring.zero();
        let g0 = element(
            &ambient,
            vec![
                polynomial(&ring, &[(1, 0, 1)]),
                polynomial(&ring, &[(0, 1, 1)]),
            ],
        );
        let g1 = element(
            &ambient,
            vec![
                polynomial(&ring, &[(0, 1, 1)]),
                polynomial(&ring, &[(0, 0, 1)]),
            ],
        );
        let input =
            Submodule::from_generators(ambient.clone(), vec![g0.clone(), g1.clone()]).unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let basis = input.groebner_basis(order).unwrap();
        basis.verify().unwrap();
        assert!(basis.is_complete());
        assert_eq!(
            ModuleGroebnerBasis::<PolynomialRing<crate::RationalField>>::from_json(
                &basis.to_json().unwrap()
            )
            .unwrap(),
            basis
        );
        assert_eq!(
            basis.initial_submodule().unwrap().generator_count(),
            basis.basis().generator_count()
        );
        let schreyer = basis.schreyer_order();
        assert_eq!(
            ModuleMonomialOrder::from_json(&schreyer.to_json().unwrap()).unwrap(),
            schreyer
        );
        assert_eq!(
            schreyer.schreyer_data().unwrap().1.len(),
            basis.basis().generator_count()
        );

        let x = polynomial(&ring, &[(1, 0, 1)]);
        let value = g0.add(&g1.scalar_mul(&x).unwrap()).unwrap();
        let reduction = basis.reduce(&value).unwrap();
        assert!(reduction.is_member());
        reduction.verify().unwrap();
        assert_eq!(
            ModuleReduction::<PolynomialRing<crate::RationalField>>::from_json(
                &reduction.to_json().unwrap()
            )
            .unwrap(),
            reduction
        );
        assert_eq!(
            input
                .generator_map()
                .apply(reduction.lift().unwrap())
                .unwrap(),
            value
        );

        let outside = element(&ambient, vec![zero, ring.one()]);
        assert!(!basis.contains(&outside).unwrap());
    }

    #[test]
    fn homogeneous_module_quotients_have_exact_multigraded_numerators() {
        let grading = FreeAbelianGrading::new(
            2,
            vec![
                Degree::new(vec![1.into(), 0.into()]),
                Degree::new(vec![0.into(), 1.into()]),
            ],
        )
        .unwrap();
        let ring = PolynomialRing::with_grading(
            Q,
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
            grading,
        )
        .unwrap();
        let ambient = FreeModule::new(ring.clone(), 1)
            .with_degree_shifts(2, vec![Degree::new(vec![(-1).into(), 0.into()])])
            .unwrap();
        let x_squared = element(
            &ambient,
            vec![ring
                .from_terms([(vec![2.into(), 0.into()], Rational::one())])
                .unwrap()],
        );
        let xy = element(
            &ambient,
            vec![ring
                .from_terms([(vec![1.into(), 1.into()], Rational::one())])
                .unwrap()],
        );
        let input = Submodule::from_generators(ambient, vec![x_squared, xy]).unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let basis = input.groebner_basis(order.clone()).unwrap();
        assert_eq!(
            basis.basis().generator_map().source().degree_shifts(),
            Some(
                [
                    Degree::new(vec![(-3).into(), 0.into()]),
                    Degree::new(vec![(-2).into(), (-1).into()]),
                ]
                .as_slice()
            )
        );
        let schreyer = basis.schreyer_syzygies().unwrap();
        assert_eq!(schreyer.ambient(), basis.basis().generator_map().source());
        assert!(map_columns(schreyer.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&basis.basis().generator_map().apply(relation).unwrap())));
        let next = basis.next_schreyer_basis().unwrap().unwrap();
        next.verify().unwrap();
        assert_eq!(next.input().ambient(), next.basis().ambient());
        assert_eq!(next.input().generator_map(), next.basis().generator_map());
        assert_eq!(
            next.transformation(),
            &ModuleHomomorphism::identity(next.basis().generator_map().source().clone()).unwrap()
        );
        assert!(next
            .basis()
            .generator_map()
            .source()
            .degree_shifts()
            .is_some());

        assert_eq!(
            basis.multigraded_hilbert_numerator().unwrap(),
            vec![
                (Degree::new(vec![1.into(), 0.into()]), 1.into()),
                (Degree::new(vec![2.into(), 1.into()]), (-1).into()),
                (Degree::new(vec![3.into(), 0.into()]), (-1).into()),
                (Degree::new(vec![3.into(), 1.into()]), 1.into()),
            ]
        );
        let target = basis.hilbert_series().unwrap();
        let guided = input
            .groebner_basis_with_hilbert_series(order, &target)
            .unwrap();
        assert_eq!(guided, basis);
        assert_eq!(guided.hilbert_series().unwrap(), target);

        let ambient = FreeModule::new(ring.clone(), 1)
            .with_degree_shifts(2, vec![Degree::zero(2)])
            .unwrap();
        let mixed = element(&ambient, vec![polynomial(&ring, &[(1, 0, 1), (0, 1, 1)])]);
        let mixed_basis = Submodule::from_generators(ambient, vec![mixed])
            .unwrap()
            .groebner_basis(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        assert!(matches!(
            mixed_basis.multigraded_hilbert_numerator(),
            Err(ModuleGroebnerError::InvalidEvidence(
                "the computed module basis is not homogeneous"
            ))
        ));
    }

    #[test]
    fn standard_schreyer_targets_follow_the_exact_sequence() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let ambient = FreeModule::new(ring.clone(), 1)
            .with_degree_shifts(1, vec![Degree::zero(1)])
            .unwrap();
        let generators = vec![
            element(&ambient, vec![polynomial(&ring, &[(2, 0, 1)])]),
            element(&ambient, vec![polynomial(&ring, &[(1, 1, 1)])]),
            element(&ambient, vec![polynomial(&ring, &[(0, 2, 1)])]),
        ];
        let basis = Submodule::from_generators(ambient, generators)
            .unwrap()
            .groebner_basis(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();

        assert!(!should_use_schreyer_hilbert_guidance(
            &basis.leading_monomials
        ));
        let target = standard_schreyer_hilbert_target(&basis).unwrap();
        assert_eq!(target.generator_degrees(), &[2, 2, 2]);
        assert_eq!(
            target.quotient_numerator(),
            &[
                Integer::zero(),
                Integer::zero(),
                Integer::from(3),
                Integer::from(-2),
            ]
        );
        basis
            .next_schreyer_basis()
            .unwrap()
            .unwrap()
            .verify()
            .unwrap();

        let ambient = FreeModule::new(ring.clone(), 1)
            .with_degree_shifts(1, vec![Degree::zero(1)])
            .unwrap();
        let seventh_power = (0..=7)
            .map(|y_degree| {
                element(
                    &ambient,
                    vec![polynomial(&ring, &[(7 - y_degree, y_degree, 1)])],
                )
            })
            .collect();
        let larger = Submodule::from_generators(ambient, seventh_power)
            .unwrap()
            .groebner_basis(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        assert!(should_use_schreyer_hilbert_guidance(
            &larger.leading_monomials
        ));
        let larger_target = standard_schreyer_hilbert_target(&larger).unwrap();
        assert_eq!(larger_target.generator_degrees(), &[7; 8]);
        assert_eq!(
            larger_target.quotient_numerator(),
            &[
                Integer::zero(),
                Integer::zero(),
                Integer::zero(),
                Integer::zero(),
                Integer::zero(),
                Integer::zero(),
                Integer::zero(),
                Integer::from(8),
                Integer::from(-7),
            ]
        );
        let next = larger.next_schreyer_basis().unwrap().unwrap();
        assert_eq!(next.basis().generator_count(), 7);
        next.verify().unwrap();
    }

    #[test]
    fn monomial_module_basis_normalizes_and_tracks_redundant_inputs() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        let ambient = FreeModule::new(ring.clone(), 2);
        let input = Submodule::from_generators(
            ambient.clone(),
            vec![
                element(&ambient, vec![polynomial(&ring, &[(1, 0, 2)]), ring.zero()]),
                element(&ambient, vec![polynomial(&ring, &[(2, 0, 5)]), ring.zero()]),
                element(
                    &ambient,
                    vec![ring.zero(), polynomial(&ring, &[(0, 1, -3)])],
                ),
            ],
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let direct = monomial_module_basis(&input, &order).unwrap().unwrap();
        let public = input.groebner_basis(order).unwrap();

        assert_eq!(public, direct);
        assert_eq!(direct.basis().generator_count(), 2);
        direct.verify().unwrap();
        assert!(map_columns(input.generator_map())
            .unwrap()
            .iter()
            .all(|generator| direct.contains(generator).unwrap()));
    }

    #[test]
    fn packed_prime_schreyer_projection_matches_element_projection() {
        let field = FiniteField::prime(65_521_u64).unwrap();
        let ring = PolynomialRing::new(
            field.clone(),
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let ambient = FreeModule::new(ring.clone(), 1);
        let generators = [[2_u64, 0_u64], [1, 1], [0, 2]]
            .into_iter()
            .map(|exponent| {
                element_from_coordinates(
                    ambient.clone(),
                    vec![ring
                        .from_u64_terms([(exponent.to_vec(), field.one())])
                        .unwrap()],
                )
                .unwrap()
            })
            .collect();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let basis = Submodule::from_generators(ambient, generators)
            .unwrap()
            .groebner_basis(order)
            .unwrap();

        let packed = basis.packed_prime_schreyer_submodule().unwrap().unwrap();
        let element_data = basis.schreyer_syzygy_data().unwrap();
        let elements = Submodule::from_generators(
            basis.basis().generator_map().source().clone(),
            element_data.relations,
        )
        .unwrap();

        assert!(packed.generator_map().is_sparse());
        assert_eq!(packed.generator_map(), elements.generator_map());
        assert!(map_columns(packed.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&basis.basis().generator_map().apply(relation).unwrap())));
    }

    #[test]
    fn packed_binary_module_dispatch_round_trips_f2_and_extension_coefficients() {
        let fields = [
            FiniteField::prime(2_u64).unwrap(),
            FiniteField::with_modulus(2, vec![1, 1, 1]).unwrap(),
        ];
        for field in fields {
            let ring = PolynomialRing::new(
                field.clone(),
                ["x", "y"],
                MonomialOrder::GradedReverseLexicographic,
            )
            .unwrap();
            let ambient = FreeModule::new(ring.clone(), 2);
            let a = if field.degree() == 1 {
                field.one()
            } else {
                field.element_from_coefficients(&[0, 1]).unwrap()
            };
            let b = if field.degree() == 1 {
                field.one()
            } else {
                field.add(&a, &field.one()).unwrap()
            };
            let coefficient = |value: &FiniteFieldElement| value.clone();
            let term = |exponent: Vec<u64>, value: &FiniteFieldElement| {
                ring.from_u64_terms([(exponent, coefficient(value))])
                    .unwrap()
            };
            let generators = vec![
                element_from_coordinates(
                    ambient.clone(),
                    vec![term(vec![2, 0], &a), term(vec![0, 1], &field.one())],
                )
                .unwrap(),
                element_from_coordinates(
                    ambient.clone(),
                    vec![term(vec![1, 1], &b), term(vec![1, 0], &a)],
                )
                .unwrap(),
                element_from_coordinates(ambient.clone(), vec![ring.zero(), term(vec![2, 0], &b)])
                    .unwrap(),
            ];
            let input = Submodule::from_generators(ambient, generators).unwrap();
            let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
            let basis = higher_rank_kernel_basis(&input, &order)
                .unwrap()
                .expect("a binary finite field has a packed higher-rank lane");
            basis.verify().unwrap();
            let relations = basis
                .packed_binary_original_syzygies()
                .unwrap()
                .expect("the packed binary relation lane is admitted");
            assert!(map_columns(relations.generator_map())
                .unwrap()
                .iter()
                .all(|relation| is_zero(&input.generator_map().apply(relation).unwrap())));
            let direct_kernel = input.generator_map().kernel(order.clone()).unwrap();
            assert!(map_columns(direct_kernel.generator_map())
                .unwrap()
                .iter()
                .all(|relation| is_zero(&input.generator_map().apply(relation).unwrap())));

            let graded_target = FreeModule::new(ring.clone(), 1)
                .with_degree_shifts(1, vec![Degree::zero(1)])
                .unwrap();
            let graded_source = FreeModule::new(ring.clone(), 2)
                .with_degree_shifts(
                    1,
                    vec![
                        Degree::new(vec![Integer::from(-2)]),
                        Degree::new(vec![Integer::from(-2)]),
                    ],
                )
                .unwrap();
            let x2 = term(vec![2, 0], &field.one());
            let y2 = term(vec![0, 2], &field.one());
            let presentation = ModuleHomomorphism::from_matrix(
                graded_source,
                graded_target.clone(),
                Matrix::from_rows(ring.clone(), [[x2, y2]]).unwrap(),
            )
            .unwrap();
            let graded = Submodule::from_generator_map(graded_target, presentation).unwrap();
            let betti = packed_standard_graded_betti(&graded, &order)
                .unwrap()
                .expect("the packed binary Betti lane is admitted");
            assert_eq!(
                betti
                    .entries
                    .iter()
                    .map(|entry| (
                        entry.homological_degree,
                        entry.internal_degree,
                        entry.multiplicity,
                    ))
                    .collect::<Vec<_>>(),
                vec![(0, 0, 1), (1, 2, 2), (2, 4, 1)]
            );
        }
    }

    #[test]
    fn primitive_rank_two_three_kernel_removes_a_common_factor() {
        let field = FiniteField::prime(32_003_u64).unwrap();
        let ring = PolynomialRing::new(
            field,
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let [x, y, z] = ring.generators().unwrap().try_into().unwrap();
        let zero = ring.zero();
        let source = FreeModule::new(ring.clone(), 3);
        let target = FreeModule::new(ring.clone(), 2);
        let map = ModuleHomomorphism::from_matrix(
            source.clone(),
            target.clone(),
            Matrix::from_rows(
                ring.clone(),
                [
                    [(&z * &x).unwrap(), (&z * &y).unwrap(), zero.clone()],
                    [zero.clone(), (&z * &x).unwrap(), (&z * &y).unwrap()],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let expected = element_from_coordinates(
            source.clone(),
            vec![
                ring.neg(&y.pow(2).unwrap()).unwrap(),
                (&x * &y).unwrap(),
                ring.neg(&x.pow(2).unwrap()).unwrap(),
            ],
        )
        .unwrap();

        let direct = primitive_rank_two_three_kernel(&map)
            .unwrap()
            .expect("the native maximal-minor lane is admitted");
        assert_eq!(map_columns(direct.generator_map()).unwrap(), [expected]);
        assert!(map_columns(direct.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&map.apply(relation).unwrap())));

        let image = Submodule::from_generator_map(target, map).unwrap();
        let basis = image
            .groebner_basis(ModuleMonomialOrder::TermOverPosition(
                MonomialOrder::GradedReverseLexicographic,
            ))
            .unwrap();
        assert_eq!(basis.syzygies().unwrap().generator_count(), 1);
    }

    #[test]
    fn primitive_rank_two_three_kernel_supports_q_and_falls_back_on_rank_drop() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let zero = ring.zero();
        let source = FreeModule::new(ring.clone(), 3);
        let target = FreeModule::new(ring.clone(), 2);
        let full_rank = ModuleHomomorphism::from_matrix(
            source.clone(),
            target.clone(),
            Matrix::from_rows(
                ring.clone(),
                [
                    [x.clone(), y.clone(), zero.clone()],
                    [zero.clone(), x.clone(), y.clone()],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let direct = primitive_rank_two_three_kernel(&full_rank)
            .unwrap()
            .expect("the rational maximal-minor lane is admitted");
        assert_eq!(direct.generator_count(), 1);
        assert!(map_columns(direct.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&full_rank.apply(relation).unwrap())));

        let rank_one = ModuleHomomorphism::from_matrix(
            source,
            target,
            Matrix::from_rows(
                ring.clone(),
                [
                    [x.clone(), y.clone(), zero.clone()],
                    [x.clone(), y.clone(), zero],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        assert!(primitive_rank_two_three_kernel(&rank_one)
            .unwrap()
            .is_none());
        let fallback = rank_one
            .kernel(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        assert!(fallback.generator_count() >= 2);
        assert!(map_columns(fallback.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&rank_one.apply(relation).unwrap())));
    }

    #[test]
    fn direct_monomial_map_kernel_matches_canonical_fallback() {
        let field = FiniteField::prime(65_521_u64).unwrap();
        let ring = PolynomialRing::new(
            field.clone(),
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let source = FreeModule::new(ring.clone(), 3);
        let target = FreeModule::new(ring.clone(), 1);
        let generators = [
            (vec![0_u64, 2], field.element(2_u64)),
            (vec![2_u64, 0], field.element(3_u64)),
            (vec![1_u64, 1], field.one()),
        ]
        .map(|(exponent, coefficient)| ring.from_u64_terms([(exponent, coefficient)]).unwrap());
        let map = ModuleHomomorphism::from_matrix(
            source,
            target.clone(),
            crate::linalg::Matrix::from_rows(ring.clone(), [generators]).unwrap(),
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let direct = map.kernel(order.clone()).unwrap();
        let fallback = Submodule::from_generator_map(target, map.clone())
            .unwrap()
            .groebner_basis(order.clone())
            .unwrap()
            .syzygies()
            .unwrap();
        let direct_basis = direct.groebner_basis(order.clone()).unwrap();
        let fallback_basis = fallback.groebner_basis(order).unwrap();

        assert_eq!(
            direct_basis.basis().generator_map(),
            fallback_basis.basis().generator_map()
        );
        assert!(map_columns(direct.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&map.apply(relation).unwrap())));
    }

    #[test]
    fn direct_monomial_map_kernel_handles_divisible_and_zero_columns() {
        let field = FiniteField::prime(65_521_u64).unwrap();
        let ring = PolynomialRing::new(
            field.clone(),
            ["x"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let source = FreeModule::new(ring.clone(), 3);
        let target = FreeModule::new(ring.clone(), 1);
        let x2 = ring.from_u64_terms([(vec![2], field.one())]).unwrap();
        let x5 = ring.from_u64_terms([(vec![5], field.one())]).unwrap();
        let map = ModuleHomomorphism::from_matrix(
            source.clone(),
            target,
            crate::linalg::Matrix::from_rows(ring.clone(), [[x2, x5, ring.zero()]]).unwrap(),
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let kernel = map.kernel(order.clone()).unwrap();
        assert_eq!(kernel.generator_count(), 2);
        let basis = kernel.groebner_basis(order).unwrap();
        let x3 = ring.from_u64_terms([(vec![3], field.one())]).unwrap();
        let expected_pair = element_from_coordinates(
            source.clone(),
            vec![x3, ring.neg(&ring.one()).unwrap(), ring.zero()],
        )
        .unwrap();
        assert!(basis.contains(&expected_pair).unwrap());
        assert!(basis.contains(&source.basis_element(2).unwrap()).unwrap());
        assert!(map_columns(kernel.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&map.apply(relation).unwrap())));
    }

    #[test]
    fn packed_prime_original_syzygies_track_a_nonmonomial_source_change() {
        let field = FiniteField::prime(65_521_u64).unwrap();
        let ring = PolynomialRing::new(
            field.clone(),
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let source = FreeModule::new(ring.clone(), 3);
        let target = FreeModule::new(ring.clone(), 2);
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let x_plus_y = ring.add(&x, &y).unwrap();
        let map = ModuleHomomorphism::from_matrix(
            source.clone(),
            target.clone(),
            crate::linalg::Matrix::from_rows(
                ring.clone(),
                [
                    vec![x.clone(), x_plus_y.clone(), y.clone()],
                    vec![ring.zero(), x.clone(), x_plus_y],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let basis = Submodule::from_generator_map(target, map.clone())
            .unwrap()
            .groebner_basis(order.clone())
            .unwrap();

        let packed = basis.packed_prime_original_syzygies().unwrap().unwrap();
        assert!(packed.generator_map().is_sparse());
        assert!(map_columns(packed.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&map.apply(relation).unwrap())));

        // The displayed map is A Q for
        // A = [[x,y,0],[0,x,y]] and
        // Q = [[1,1,0],[0,1,1],[0,0,1]].  Thus Q^-1 carries the known
        // generator (y^2,-xy,x^2) of ker(A) to this exact relation.
        let one = field.one();
        let minus_one = field.element(65_520_u64);
        let expected_relation = element_from_coordinates(
            source.clone(),
            vec![
                ring.from_u64_terms([
                    (vec![2, 0], one.clone()),
                    (vec![1, 1], one.clone()),
                    (vec![0, 2], one.clone()),
                ])
                .unwrap(),
                ring.from_u64_terms([(vec![2, 0], minus_one.clone()), (vec![1, 1], minus_one)])
                    .unwrap(),
                ring.from_u64_terms([(vec![2, 0], one)]).unwrap(),
            ],
        )
        .unwrap();
        let expected = Submodule::from_generators(source, vec![expected_relation])
            .unwrap()
            .groebner_basis(order.clone())
            .unwrap();
        let actual = packed.groebner_basis(order.clone()).unwrap();
        assert_eq!(
            actual.basis().generator_map(),
            expected.basis().generator_map()
        );

        let direct = map.kernel(order.clone()).unwrap();
        let direct = direct.groebner_basis(order).unwrap();
        assert_eq!(
            direct.basis().generator_map(),
            expected.basis().generator_map()
        );
    }

    #[test]
    fn packed_prime_sparse_projection_handles_four_generator_modules() {
        let field = FiniteField::prime(32_003_u64).unwrap();
        let ring = PolynomialRing::new(
            field,
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let [x, y, z]: [_; 3] = ring.generators().unwrap().try_into().unwrap();
        let source = FreeModule::new(ring.clone(), 4);
        let target = FreeModule::new(ring.clone(), 2);
        let map = ModuleHomomorphism::from_matrix(
            source,
            target.clone(),
            crate::linalg::Matrix::from_rows(
                ring.clone(),
                [
                    vec![x.clone(), y.clone(), ring.zero(), z],
                    vec![ring.zero(), x.clone(), y.clone(), (&x + &y).unwrap()],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let input = Submodule::from_generator_map(target, map.clone()).unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let basis = input.groebner_basis(order).unwrap();
        basis.verify().unwrap();
        assert!(basis.basis().generator_map().is_sparse());
        assert!(basis.transformation().is_sparse());
        let syzygies = basis.syzygies().unwrap();
        assert!(syzygies.generator_map().is_sparse());
        assert!(map_columns(syzygies.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&map.apply(relation).unwrap())));
    }

    #[test]
    fn kernel_relations_are_complete_on_a_nontrivial_row_map() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        let source = FreeModule::new(ring.clone(), 3);
        let target = FreeModule::new(ring.clone(), 1);
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let map = ModuleHomomorphism::from_matrix(
            source.clone(),
            target,
            crate::linalg::Matrix::from_rows(
                ring.clone(),
                [vec![x.clone(), y.clone(), ring.zero()]],
            )
            .unwrap(),
        )
        .unwrap();
        let kernel = map
            .kernel(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        let relations = map_columns(kernel.generator_map()).unwrap();
        assert!(relations.len() >= 2);
        assert!(relations
            .iter()
            .all(|relation| is_zero(&map.apply(relation).unwrap())));
        let expected = element(&source, vec![y, ring.neg(&x).unwrap(), ring.zero()]);
        let relation_basis = kernel
            .groebner_basis(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        assert!(relation_basis.contains(&expected).unwrap());
        assert!(relation_basis
            .contains(&source.basis_element(2).unwrap())
            .unwrap());
    }

    #[test]
    fn position_order_changes_the_leading_component_exactly() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        let module = FreeModule::new(ring.clone(), 2);
        let value = element(
            &module,
            vec![ring.generator(1).unwrap(), ring.generator(0).unwrap()],
        );
        let top = value
            .leading_module_term(&ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap()
            .unwrap()
            .0;
        let pot = value
            .leading_module_term(&ModuleMonomialOrder::PositionOverTerm(ring.term_order()))
            .unwrap()
            .unwrap()
            .0;
        assert_eq!(top.component(), 1);
        assert_eq!(pot.component(), 0);
    }

    #[test]
    fn zero_and_duplicate_generators_preserve_complete_kernel_evidence() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::Lexicographic).unwrap();
        let ambient = FreeModule::new(ring.clone(), 1);
        let x = element(&ambient, vec![ring.generator(0).unwrap()]);
        let zero = ModuleElement::zero(ambient.clone());
        let input = Submodule::from_generators(ambient.clone(), vec![zero, x.clone(), x]).unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let basis = input.groebner_basis(order).unwrap();
        assert_eq!(basis.basis().generator_count(), 1);
        let syzygies = basis.syzygies().unwrap();
        assert_eq!(syzygies.ambient().rank(), 3);
        assert!(map_columns(syzygies.generator_map())
            .unwrap()
            .iter()
            .all(|relation| is_zero(&input.generator_map().apply(relation).unwrap())));

        let zero_map = ModuleHomomorphism::zero(
            FreeModule::new(ring.clone(), 2),
            FreeModule::new(ring.clone(), 1),
        )
        .unwrap();
        let kernel = zero_map
            .kernel(ModuleMonomialOrder::TermOverPosition(ring.term_order()))
            .unwrap();
        assert_eq!(kernel.generator_count(), 2);
    }
}
