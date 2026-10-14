// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Public strong module Gröbner values over integral polynomial rings.

use core::fmt;
use std::collections::VecDeque;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{
    from_integer_kernel_polynomial, FreeAbelianGrading, Polynomial, PolynomialRing,
};
use crate::{
    Integer, IntegerRing, NoetherError, PrincipalIdealDomain, Ring, SemanticType, TypeDescription,
};

use super::{
    compare_module_monomials, element_from_coordinates, exponent_to_kernel, is_zero, map_columns,
    map_kernel_module_error, module_monomial_divides, monomial_lcm, monomial_quotient,
    resource_limit, to_kernel_module_order, validate_schreyer_frame, ModuleElement, ModuleError,
    ModuleGroebnerError, ModuleHomomorphism, ModuleMonomial, ModuleMonomialOrder, Submodule,
    MAX_REDUCTION_STEPS,
};
use crate::homalg::FreeModule;

static STRONG_MODULE_TERM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/StrongModuleTerm@1",
    rust_path: "noether::homalg::StrongModuleTerm",
    python_path: Some("noether.homalg.StrongModuleTerm"),
};

static STRONG_MODULE_GROEBNER_BASIS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/StrongModuleGroebnerBasis@1",
    rust_path: "noether::homalg::StrongModuleGroebnerBasis",
    python_path: Some("noether.homalg.StrongModuleGroebnerBasis"),
};

static STRONG_MODULE_REDUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/StrongModuleReduction@1",
    rust_path: "noether::homalg::StrongModuleReduction",
    python_path: Some("noether.homalg.StrongModuleReduction"),
};

/// A positive integer coefficient times one leading module monomial.
///
/// Strong bases over `Z` retain this coefficient because divisibility of
/// leading coefficients is part of the initial module and of reduction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrongModuleTerm {
    monomial: ModuleMonomial,
    coefficient: Integer,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrongModuleTermWire {
    #[serde(rename = "type")]
    type_id: String,
    monomial: ModuleMonomial,
    coefficient: Integer,
}

impl StrongModuleTerm {
    fn new(monomial: ModuleMonomial, coefficient: Integer) -> Self {
        Self {
            monomial,
            coefficient,
        }
    }

    /// Leading polynomial monomial and free-module component.
    pub fn monomial(&self) -> &ModuleMonomial {
        &self.monomial
    }

    /// Positive leading coefficient.
    pub fn coefficient(&self) -> &Integer {
        &self.coefficient
    }

    /// Serialize this exact strong leading term.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and validate a strong leading term.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Serialize for StrongModuleTerm {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        StrongModuleTermWire {
            type_id: Self::TYPE_ID.to_owned(),
            monomial: self.monomial.clone(),
            coefficient: self.coefficient.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StrongModuleTerm {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StrongModuleTermWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        if !wire.coefficient.is_positive() {
            return Err(serde::de::Error::custom(
                "a strong module leading coefficient must be positive",
            ));
        }
        Ok(Self::new(wire.monomial, wire.coefficient))
    }
}

impl SemanticType for StrongModuleTerm {
    const TYPE_ID: &'static str = "noether/homalg/StrongModuleTerm@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &STRONG_MODULE_TERM_DESCRIPTION
    }
}

/// A verified normalized strong Gröbner basis for a submodule of
/// `Z[x_1,...,x_n]^r`.
///
/// Unlike a field basis, its generators need not be monic. The retained
/// transformation map proves that the basis lies in the original submodule;
/// strong S/G-pair verification and reduction of every input generator prove
/// the reverse containment and completeness.
#[derive(Clone)]
pub struct StrongModuleGroebnerBasis {
    input: Submodule<PolynomialRing<IntegerRing>>,
    order: ModuleMonomialOrder,
    basis: Submodule<PolynomialRing<IntegerRing>>,
    transformation: ModuleHomomorphism<PolynomialRing<IntegerRing>>,
    leading_terms: Vec<StrongModuleTerm>,
}

/// Exact Euclidean reduction and lift evidence against an integral strong
/// module basis.
#[derive(Clone)]
pub struct StrongModuleReduction {
    basis: StrongModuleGroebnerBasis,
    value: ModuleElement<PolynomialRing<IntegerRing>>,
    quotient: ModuleElement<PolynomialRing<IntegerRing>>,
    original_combination: ModuleElement<PolynomialRing<IntegerRing>>,
    remainder: ModuleElement<PolynomialRing<IntegerRing>>,
}

impl StrongModuleGroebnerBasis {
    /// Original generated submodule.
    pub fn input(&self) -> &Submodule<PolynomialRing<IntegerRing>> {
        &self.input
    }

    /// Exact module-monomial order.
    pub fn order(&self) -> &ModuleMonomialOrder {
        &self.order
    }

    /// Canonically ordered positive-leading-coefficient strong basis.
    pub fn basis(&self) -> &Submodule<PolynomialRing<IntegerRing>> {
        &self.basis
    }

    /// Map expressing every strong-basis generator in the original family.
    pub fn transformation(&self) -> &ModuleHomomorphism<PolynomialRing<IntegerRing>> {
        &self.transformation
    }

    /// Ordered coefficient-bearing leading terms.
    pub fn leading_terms(&self) -> &[StrongModuleTerm] {
        &self.leading_terms
    }

    /// Initial submodule generated by the full coefficient-bearing leading
    /// terms.
    pub fn initial_submodule(
        &self,
    ) -> Result<Submodule<PolynomialRing<IntegerRing>>, ModuleGroebnerError> {
        let ring = self.input.ambient().ring();
        let generators = self
            .leading_terms
            .iter()
            .map(|leading| {
                let mut coordinates = vec![ring.zero(); self.input.ambient().rank()];
                coordinates[leading.monomial.component()] = ring.from_terms([(
                    leading.monomial.exponent().to_vec(),
                    leading.coefficient.clone(),
                )])?;
                element_from_coordinates(self.input.ambient().clone(), coordinates)
                    .map_err(ModuleGroebnerError::from)
            })
            .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
        integral_submodule_from_generators(self.input.ambient(), generators)
    }

    /// This value is complete; bounded failures do not return partial bases.
    pub const fn is_complete(&self) -> bool {
        true
    }

    /// Reduce an ambient value, retaining canonical-basis quotients and an
    /// exact lift into the original generators.
    pub fn reduce(
        &self,
        value: &ModuleElement<PolynomialRing<IntegerRing>>,
    ) -> Result<StrongModuleReduction, ModuleGroebnerError> {
        if value.module() != self.input.ambient() {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the reduced value belongs to another ambient module",
            ));
        }
        let ring = self.input.ambient().ring();
        let basis_values = map_columns(self.basis.generator_map())?;
        let (remainder, quotients) = match (
            to_kernel_module_order(&self.order),
            to_integer_kernel_module_elements(&basis_values, ring, self.input.ambient().rank())?,
            to_integer_kernel_module_elements(
                std::slice::from_ref(value),
                ring,
                self.input.ambient().rank(),
            )?,
        ) {
            (Some(kernel_order), Some(basis_kernel), Some(mut value_kernel)) => {
                let value_kernel = value_kernel
                    .pop()
                    .expect("one input projects to one kernel value");
                let output = groebner_kernel::strong_module_normal_form_with_quotients(
                    &value_kernel,
                    &basis_kernel,
                    &kernel_order,
                    MAX_REDUCTION_STEPS,
                )
                .map_err(map_kernel_module_error)?;
                let remainder =
                    from_integer_kernel_module_element(self.input.ambient(), output.remainder)?;
                let quotients = output
                    .quotients
                    .into_iter()
                    .map(|polynomial| {
                        from_integer_kernel_polynomial(ring, polynomial)
                            .map_err(super::map_polynomial_groebner_error)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                (remainder, quotients)
            }
            _ => strong_normal_form_arbitrary(
                value,
                &basis_values,
                &self.order,
                MAX_REDUCTION_STEPS,
            )?,
        };
        let quotient =
            element_from_coordinates(self.basis.generator_map().source().clone(), quotients)?;
        let original_combination = self.transformation.apply(&quotient)?;
        let result = StrongModuleReduction {
            basis: self.clone(),
            value: value.clone(),
            quotient,
            original_combination,
            remainder,
        };
        result.verify()?;
        Ok(result)
    }

    /// Canonical Euclidean strong normal form.
    pub fn normal_form(
        &self,
        value: &ModuleElement<PolynomialRing<IntegerRing>>,
    ) -> Result<ModuleElement<PolynomialRing<IntegerRing>>, ModuleGroebnerError> {
        Ok(self.reduce(value)?.remainder)
    }

    /// Whether an ambient value belongs to the original submodule.
    pub fn contains(
        &self,
        value: &ModuleElement<PolynomialRing<IntegerRing>>,
    ) -> Result<bool, ModuleGroebnerError> {
        Ok(self.reduce(value)?.is_member())
    }

    /// Original-generator coordinates for a member.
    pub fn lift(
        &self,
        value: &ModuleElement<PolynomialRing<IntegerRing>>,
    ) -> Result<Option<ModuleElement<PolynomialRing<IntegerRing>>>, ModuleGroebnerError> {
        let reduction = self.reduce(value)?;
        Ok(reduction
            .is_member()
            .then(|| reduction.original_combination.clone()))
    }

    /// Complete relation module among the original integral generators.
    pub fn syzygies(&self) -> Result<Submodule<PolynomialRing<IntegerRing>>, ModuleGroebnerError> {
        strong_relations_submodule(&self.input)
    }

    /// Independently recompute and compare the canonical basis,
    /// transformation, and coefficient-bearing leaders.
    pub fn verify(&self) -> Result<(), ModuleGroebnerError> {
        validate_integral_order(self.input.ambient(), &self.order)?;
        if self.basis.ambient() != self.input.ambient()
            || self.transformation.source() != self.basis.generator_map().source()
            || self.transformation.target() != self.input.generator_map().source()
            || self.input.generator_map().compose(&self.transformation)?
                != *self.basis.generator_map()
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the integral strong basis has incompatible or invalid change data",
            ));
        }
        let expected = compute_strong_module_basis(&self.input, self.order.clone())?;
        if expected.basis.generator_map() != self.basis.generator_map()
            || expected.transformation != self.transformation
            || expected.leading_terms != self.leading_terms
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "recomputation changed the normalized integral strong module basis",
            ));
        }
        Ok(())
    }

    /// Serialize the complete integral strong-basis evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and independently recompute the strong basis.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl StrongModuleReduction {
    /// Strong basis used for reduction.
    pub fn basis(&self) -> &StrongModuleGroebnerBasis {
        &self.basis
    }

    /// Ambient value that was reduced.
    pub fn value(&self) -> &ModuleElement<PolynomialRing<IntegerRing>> {
        &self.value
    }

    /// Coordinates in the canonical strong-basis generators.
    pub fn quotient(&self) -> &ModuleElement<PolynomialRing<IntegerRing>> {
        &self.quotient
    }

    /// Corresponding combination of the original input generators.
    pub fn original_combination(&self) -> &ModuleElement<PolynomialRing<IntegerRing>> {
        &self.original_combination
    }

    /// Canonical Euclidean remainder.
    pub fn remainder(&self) -> &ModuleElement<PolynomialRing<IntegerRing>> {
        &self.remainder
    }

    /// Whether this reduction proves membership.
    pub fn is_member(&self) -> bool {
        is_zero(&self.remainder)
    }

    /// Original-generator lift when the remainder vanishes.
    pub fn lift(&self) -> Option<&ModuleElement<PolynomialRing<IntegerRing>>> {
        self.is_member().then_some(&self.original_combination)
    }

    /// Verify both reconstruction identities.
    pub fn verify(&self) -> Result<(), ModuleGroebnerError> {
        if self.value.module() != self.basis.input.ambient()
            || self.remainder.module() != self.basis.input.ambient()
            || self.quotient.module() != self.basis.basis.generator_map().source()
            || self.original_combination.module() != self.basis.input.generator_map().source()
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "integral module reduction components have incompatible modules",
            ));
        }
        let basis_part = self.basis.basis.generator_map().apply(&self.quotient)?;
        if basis_part.add(&self.remainder)? != self.value {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "integral quotient and remainder do not reconstruct the value",
            ));
        }
        if self.basis.transformation.apply(&self.quotient)? != self.original_combination {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the integral original-generator combination does not match the change map",
            ));
        }
        let original_part = self
            .basis
            .input
            .generator_map()
            .apply(&self.original_combination)?;
        if original_part.add(&self.remainder)? != self.value {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the integral lift and remainder do not reconstruct the value",
            ));
        }
        Ok(())
    }

    /// Serialize exact integral reduction evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and verify exact integral reduction evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl Submodule<PolynomialRing<IntegerRing>> {
    /// Compute the complete normalized strong module Gröbner basis over `Z`.
    pub fn strong_groebner_basis(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<StrongModuleGroebnerBasis, ModuleGroebnerError> {
        compute_strong_module_basis(self, order)
    }
}

impl ModuleHomomorphism<PolynomialRing<IntegerRing>> {
    /// Exact kernel as a generated integral polynomial submodule.
    pub fn kernel(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<Submodule<PolynomialRing<IntegerRing>>, ModuleGroebnerError> {
        validate_integral_order(self.target(), &order)?;
        if self.source().rank() == 0 {
            return Submodule::from_generators(self.source().clone(), Vec::new())
                .map_err(Into::into);
        }
        strong_relations_submodule(&Submodule::from_generator_map(
            self.target().clone(),
            self.clone(),
        )?)
    }
}

impl ModuleElement<PolynomialRing<IntegerRing>> {
    /// Leading coefficient-bearing module term in the supplied order.
    pub fn leading_module_term(
        &self,
        order: &ModuleMonomialOrder,
    ) -> Result<Option<StrongModuleTerm>, ModuleGroebnerError> {
        validate_integral_order(self.module(), order)?;
        Ok(integral_leading_term(self, order)
            .map(|(monomial, coefficient)| StrongModuleTerm::new(monomial, coefficient)))
    }
}

fn compute_strong_module_basis(
    input: &Submodule<PolynomialRing<IntegerRing>>,
    order: ModuleMonomialOrder,
) -> Result<StrongModuleGroebnerBasis, ModuleGroebnerError> {
    validate_integral_order(input.ambient(), &order)?;
    let values = map_columns(input.generator_map())?;
    if values.is_empty() {
        return project_strong_module_basis(input, order, Vec::new(), Vec::new());
    }
    let ring = input.ambient().ring();
    if let (Some(kernel_order), Some(kernel_values)) = (
        to_kernel_module_order(&order),
        to_integer_kernel_module_elements(&values, ring, input.ambient().rank())?,
    ) {
        let output = groebner_kernel::strong_module_groebner_basis(
            &kernel_values,
            &kernel_order,
            &groebner_kernel::StrongGroebnerOptions::default(),
        )
        .map_err(map_kernel_module_error)?;
        return project_strong_module_basis(input, order, output.basis, output.transformations);
    }
    let tracked = strong_module_basis_arbitrary(
        input,
        &order,
        &groebner_kernel::StrongGroebnerOptions::default(),
    )?;
    project_arbitrary_strong_module_basis(input, order, tracked)
}

type IntegralPolynomial = Polynomial<IntegerRing>;
type IntegralModuleElement = ModuleElement<PolynomialRing<IntegerRing>>;

#[derive(Clone, PartialEq, Eq)]
struct ArbitraryTrackedIntegerModuleElement {
    value: IntegralModuleElement,
    representation: IntegralModuleElement,
}

fn strong_module_basis_arbitrary(
    input: &Submodule<PolynomialRing<IntegerRing>>,
    order: &ModuleMonomialOrder,
    options: &groebner_kernel::StrongGroebnerOptions,
) -> Result<Vec<ArbitraryTrackedIntegerModuleElement>, ModuleGroebnerError> {
    validate_integral_order(input.ambient(), order)?;
    let source = input.generator_map().source();
    let inputs = map_columns(input.generator_map())?;
    let mut basis = Vec::new();
    for (index, value) in inputs.iter().cloned().enumerate() {
        if is_zero(&value) {
            continue;
        }
        let entry = normalize_arbitrary_tracked_sign(
            ArbitraryTrackedIntegerModuleElement {
                value,
                representation: source.basis_element(index)?,
            },
            order,
        )?;
        if !basis
            .iter()
            .any(|known: &ArbitraryTrackedIntegerModuleElement| known.value == entry.value)
        {
            basis.push(entry);
        }
    }
    if basis.len() > options.max_basis_size {
        return Err(resource_limit(
            "strong arbitrary-exponent module basis size",
            options.max_basis_size,
        ));
    }

    let mut pending = VecDeque::new();
    for right in 1..basis.len() {
        install_arbitrary_pairs(
            &basis,
            right,
            &mut pending,
            order,
            options.max_pending_pairs,
        )?;
    }
    let mut processed = 0_usize;
    while let Some((left, right)) = pending.pop_front() {
        if processed == options.max_processed_pairs {
            return Err(resource_limit(
                "strong arbitrary-exponent module processed pairs",
                options.max_processed_pairs,
            ));
        }
        processed += 1;
        for candidate in arbitrary_integer_module_pairs(&basis[left], &basis[right], order)? {
            let remainder = tracked_normal_form_arbitrary(
                candidate,
                &basis,
                order,
                options.max_reduction_steps,
            )?;
            if is_zero(&remainder.value) {
                continue;
            }
            let remainder = normalize_arbitrary_tracked_sign(remainder, order)?;
            if basis.iter().any(|known| known.value == remainder.value) {
                continue;
            }
            if basis.len() == options.max_basis_size {
                return Err(resource_limit(
                    "strong arbitrary-exponent module basis size",
                    options.max_basis_size,
                ));
            }
            let next = basis.len();
            basis.push(remainder);
            install_arbitrary_pairs(&basis, next, &mut pending, order, options.max_pending_pairs)?;
        }
    }

    autoreduce_arbitrary_tracked(&mut basis, order, options.max_reduction_steps)?;
    verify_arbitrary_strong_module_basis(
        input,
        &inputs,
        &basis,
        order,
        options.max_reduction_steps,
    )?;
    Ok(basis)
}

fn project_arbitrary_strong_module_basis(
    input: &Submodule<PolynomialRing<IntegerRing>>,
    order: ModuleMonomialOrder,
    tracked: Vec<ArbitraryTrackedIntegerModuleElement>,
) -> Result<StrongModuleGroebnerBasis, ModuleGroebnerError> {
    let leading_terms = tracked
        .iter()
        .map(|entry| {
            integral_leading_term(&entry.value, &order)
                .map(|(monomial, coefficient)| StrongModuleTerm::new(monomial, coefficient))
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "the arbitrary-exponent strong basis contains zero",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (basis_values, representation_values): (Vec<_>, Vec<_>) = tracked
        .into_iter()
        .map(|entry| (entry.value, entry.representation))
        .unzip();
    let basis = integral_submodule_from_generators(input.ambient(), basis_values)?;
    let transformation = ModuleHomomorphism::from_coordinate_columns(
        basis.generator_map().source().clone(),
        input.generator_map().source().clone(),
        representation_values
            .into_iter()
            .map(|value| value.coordinates().values().collect())
            .collect(),
    )?;
    Ok(StrongModuleGroebnerBasis {
        input: input.clone(),
        order,
        basis,
        transformation,
        leading_terms,
    })
}

fn install_arbitrary_pairs(
    basis: &[ArbitraryTrackedIntegerModuleElement],
    new_index: usize,
    pending: &mut VecDeque<(usize, usize)>,
    order: &ModuleMonomialOrder,
    max_pending_pairs: usize,
) -> Result<(), ModuleGroebnerError> {
    let new_component = integral_leading_term(&basis[new_index].value, order)
        .expect("an installed arbitrary-exponent strong basis member is nonzero")
        .0
        .component();
    for (old, entry) in basis.iter().enumerate().take(new_index) {
        if integral_leading_term(&entry.value, order)
            .expect("an installed arbitrary-exponent strong basis member is nonzero")
            .0
            .component()
            != new_component
        {
            continue;
        }
        if pending.len() == max_pending_pairs {
            return Err(resource_limit(
                "strong arbitrary-exponent module pending pairs",
                max_pending_pairs,
            ));
        }
        pending.push_back((old, new_index));
    }
    Ok(())
}

fn arbitrary_integer_module_pairs(
    left: &ArbitraryTrackedIntegerModuleElement,
    right: &ArbitraryTrackedIntegerModuleElement,
    order: &ModuleMonomialOrder,
) -> Result<[ArbitraryTrackedIntegerModuleElement; 2], ModuleGroebnerError> {
    let (left_term, left_coefficient) = integral_leading_term(&left.value, order).ok_or(
        ModuleGroebnerError::InvalidEvidence("an arbitrary-exponent strong pair contains zero"),
    )?;
    let (right_term, right_coefficient) = integral_leading_term(&right.value, order).ok_or(
        ModuleGroebnerError::InvalidEvidence("an arbitrary-exponent strong pair contains zero"),
    )?;
    if left_term.component() != right_term.component() {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "a strong module pair has unequal leading components",
        ));
    }
    let ring = left.value.module().ring();
    let lcm = monomial_lcm(left_term.exponent(), right_term.exponent());
    let coefficient_lcm = left_coefficient.abs().lcm(&right_coefficient.abs());
    let left_exponent = monomial_quotient(&lcm, left_term.exponent());
    let right_exponent = monomial_quotient(&lcm, right_term.exponent());
    let s_left =
        ring.from_terms([(left_exponent.clone(), &coefficient_lcm / &left_coefficient)])?;
    let s_right = ring.from_terms([(
        right_exponent.clone(),
        &coefficient_lcm / &right_coefficient,
    )])?;
    let s = combine_arbitrary_tracked(left, &s_left, right, &s_right, false)?;

    let bezout = IntegerRing.extended_gcd(&left_coefficient, &right_coefficient);
    let g_left = ring.from_terms([(left_exponent, bezout.coefficient_a)])?;
    let g_right = ring.from_terms([(right_exponent, bezout.coefficient_b)])?;
    let g = combine_arbitrary_tracked(left, &g_left, right, &g_right, true)?;
    Ok([s, g])
}

fn combine_arbitrary_tracked(
    left: &ArbitraryTrackedIntegerModuleElement,
    left_factor: &IntegralPolynomial,
    right: &ArbitraryTrackedIntegerModuleElement,
    right_factor: &IntegralPolynomial,
    add_right: bool,
) -> Result<ArbitraryTrackedIntegerModuleElement, ModuleGroebnerError> {
    let left_value = left.value.scalar_mul(left_factor)?;
    let right_value = right.value.scalar_mul(right_factor)?;
    let left_representation = left.representation.scalar_mul(left_factor)?;
    let right_representation = right.representation.scalar_mul(right_factor)?;
    Ok(ArbitraryTrackedIntegerModuleElement {
        value: if add_right {
            left_value.add(&right_value)?
        } else {
            left_value.sub(&right_value)?
        },
        representation: if add_right {
            left_representation.add(&right_representation)?
        } else {
            left_representation.sub(&right_representation)?
        },
    })
}

fn strong_normal_form_arbitrary(
    value: &IntegralModuleElement,
    basis: &[IntegralModuleElement],
    order: &ModuleMonomialOrder,
    max_steps: usize,
) -> Result<(IntegralModuleElement, Vec<IntegralPolynomial>), ModuleGroebnerError> {
    validate_integral_order(value.module(), order)?;
    if basis.iter().any(|entry| entry.module() != value.module()) {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "arbitrary-exponent strong reducers belong to another module",
        ));
    }
    require_arbitrary_positive_leaders(basis, order)?;
    let ring = value.module().ring();
    let mut work = value.clone();
    let mut remainder = ModuleElement::zero(value.module().clone());
    let mut quotients = vec![ring.zero(); basis.len()];
    let mut steps = 0_usize;
    while let Some((work_term, work_coefficient)) = integral_leading_term(&work, order) {
        if steps == max_steps {
            return Err(resource_limit(
                "strong arbitrary-exponent module normal-form steps",
                max_steps,
            ));
        }
        let reduction = basis.iter().enumerate().find_map(|(index, reducer)| {
            let (reducer_term, reducer_coefficient) = integral_leading_term(reducer, order)?;
            if !module_monomial_divides(&reducer_term, &work_term) {
                return None;
            }
            let quotient = work_coefficient.div_floor(&reducer_coefficient.abs());
            (!quotient.is_zero()).then_some((index, reducer, reducer_term, quotient))
        });
        if let Some((index, reducer, reducer_term, quotient)) = reduction {
            let factor = ring.from_terms([(
                monomial_quotient(work_term.exponent(), reducer_term.exponent()),
                quotient,
            )])?;
            quotients[index] = ring.add(&quotients[index], &factor)?;
            work = work.sub(&reducer.scalar_mul(&factor)?)?;
        } else {
            let term = integral_module_term(value.module(), work_term, work_coefficient)?;
            remainder = remainder.add(&term)?;
            work = work.sub(&term)?;
        }
        steps += 1;
    }
    Ok((remainder, quotients))
}

fn tracked_normal_form_arbitrary(
    mut current: ArbitraryTrackedIntegerModuleElement,
    basis: &[ArbitraryTrackedIntegerModuleElement],
    order: &ModuleMonomialOrder,
    max_steps: usize,
) -> Result<ArbitraryTrackedIntegerModuleElement, ModuleGroebnerError> {
    let values = basis
        .iter()
        .map(|entry| entry.value.clone())
        .collect::<Vec<_>>();
    let (remainder, quotients) =
        strong_normal_form_arbitrary(&current.value, &values, order, max_steps)?;
    for (quotient, reducer) in quotients.iter().zip(basis) {
        if quotient.is_zero() {
            continue;
        }
        current.representation = current
            .representation
            .sub(&reducer.representation.scalar_mul(quotient)?)?;
    }
    current.value = remainder;
    Ok(current)
}

fn normalize_arbitrary_tracked_sign(
    mut value: ArbitraryTrackedIntegerModuleElement,
    order: &ModuleMonomialOrder,
) -> Result<ArbitraryTrackedIntegerModuleElement, ModuleGroebnerError> {
    if integral_leading_term(&value.value, order)
        .is_some_and(|(_, coefficient)| coefficient.is_negative())
    {
        let negative_one = value.value.module().ring().constant(Integer::from(-1))?;
        value.value = value.value.scalar_mul(&negative_one)?;
        value.representation = value.representation.scalar_mul(&negative_one)?;
    }
    Ok(value)
}

fn autoreduce_arbitrary_tracked(
    basis: &mut Vec<ArbitraryTrackedIntegerModuleElement>,
    order: &ModuleMonomialOrder,
    max_steps: usize,
) -> Result<(), ModuleGroebnerError> {
    let mut changed = true;
    let mut passes = 0_usize;
    while changed && passes <= basis.len().saturating_add(1) {
        changed = false;
        let mut index = 0_usize;
        while index < basis.len() {
            let reducers = basis
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, entry)| entry.clone())
                .collect::<Vec<_>>();
            let reduced = normalize_arbitrary_tracked_sign(
                tracked_normal_form_arbitrary(basis[index].clone(), &reducers, order, max_steps)?,
                order,
            )?;
            if is_zero(&reduced.value) {
                basis.remove(index);
                changed = true;
            } else {
                if reduced != basis[index] {
                    basis[index] = reduced;
                    changed = true;
                }
                index += 1;
            }
        }
        deduplicate_arbitrary_tracked(basis);
        passes += 1;
    }
    if changed {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "arbitrary-exponent strong module autoreduction did not stabilize",
        ));
    }
    sort_arbitrary_tracked(basis, order)
}

fn deduplicate_arbitrary_tracked(basis: &mut Vec<ArbitraryTrackedIntegerModuleElement>) {
    let mut unique = Vec::with_capacity(basis.len());
    for entry in basis.drain(..) {
        if !unique
            .iter()
            .any(|known: &ArbitraryTrackedIntegerModuleElement| known.value == entry.value)
        {
            unique.push(entry);
        }
    }
    *basis = unique;
}

fn sort_arbitrary_tracked(
    basis: &mut [ArbitraryTrackedIntegerModuleElement],
    order: &ModuleMonomialOrder,
) -> Result<(), ModuleGroebnerError> {
    if basis
        .iter()
        .any(|entry| integral_leading_term(&entry.value, order).is_none())
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "an arbitrary-exponent strong working basis contains zero",
        ));
    }
    basis.sort_by(|left, right| {
        let (left_term, left_coefficient) =
            integral_leading_term(&left.value, order).expect("nonzero strong basis invariant");
        let (right_term, right_coefficient) =
            integral_leading_term(&right.value, order).expect("nonzero strong basis invariant");
        compare_module_monomials(&right_term, &left_term, order)
            .then_with(|| left_coefficient.cmp(&right_coefficient))
    });
    Ok(())
}

fn require_arbitrary_positive_leaders(
    basis: &[IntegralModuleElement],
    order: &ModuleMonomialOrder,
) -> Result<(), ModuleGroebnerError> {
    if basis.iter().any(|entry| {
        integral_leading_term(entry, order)
            .is_none_or(|(_, coefficient)| !coefficient.is_positive())
    }) {
        Err(ModuleGroebnerError::InvalidEvidence(
            "an arbitrary-exponent strong basis has a nonpositive leader",
        ))
    } else {
        Ok(())
    }
}

fn verify_arbitrary_strong_module_basis(
    input: &Submodule<PolynomialRing<IntegerRing>>,
    inputs: &[IntegralModuleElement],
    basis: &[ArbitraryTrackedIntegerModuleElement],
    order: &ModuleMonomialOrder,
    max_steps: usize,
) -> Result<(), ModuleGroebnerError> {
    let values = basis
        .iter()
        .map(|entry| entry.value.clone())
        .collect::<Vec<_>>();
    require_arbitrary_positive_leaders(&values, order)?;
    for entry in basis {
        if entry.representation.module() != input.generator_map().source()
            || input.generator_map().apply(&entry.representation)? != entry.value
        {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "an arbitrary-exponent strong transformation does not reconstruct its basis value",
            ));
        }
    }
    for value in inputs {
        let (remainder, _) = strong_normal_form_arbitrary(value, &values, order, max_steps)?;
        if !is_zero(&remainder) {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "an input does not reduce to zero by the arbitrary-exponent strong basis",
            ));
        }
    }
    for right in 1..basis.len() {
        for left in 0..right {
            if integral_leading_term(&basis[left].value, order)
                .zip(integral_leading_term(&basis[right].value, order))
                .is_none_or(|((left, _), (right, _))| left.component() != right.component())
            {
                continue;
            }
            for pair in arbitrary_integer_module_pairs(&basis[left], &basis[right], order)? {
                let (remainder, _) =
                    strong_normal_form_arbitrary(&pair.value, &values, order, max_steps)?;
                if !is_zero(&remainder) {
                    return Err(ModuleGroebnerError::InvalidEvidence(
                        "an arbitrary-exponent strong S/G pair has nonzero remainder",
                    ));
                }
            }
        }
    }
    for index in 0..values.len() {
        let reducers = values
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        let (remainder, _) =
            strong_normal_form_arbitrary(&values[index], &reducers, order, max_steps)?;
        if remainder != values[index] {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "the arbitrary-exponent strong basis is not autoreduced",
            ));
        }
    }
    Ok(())
}

fn integral_module_term(
    module: &FreeModule<PolynomialRing<IntegerRing>>,
    monomial: ModuleMonomial,
    coefficient: Integer,
) -> Result<IntegralModuleElement, ModuleGroebnerError> {
    if monomial.component() >= module.rank() {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "an arbitrary-exponent module term lies outside the ambient rank",
        ));
    }
    let mut coordinates = vec![module.ring().zero(); module.rank()];
    coordinates[monomial.component()] = module
        .ring()
        .from_terms([(monomial.exponent().to_vec(), coefficient)])?;
    element_from_coordinates(module.clone(), coordinates).map_err(Into::into)
}

fn project_strong_module_basis(
    input: &Submodule<PolynomialRing<IntegerRing>>,
    order: ModuleMonomialOrder,
    basis: Vec<groebner_kernel::SparseModuleElement<Integer>>,
    transformations: Vec<Vec<poly_core::SparsePolynomial<Integer>>>,
) -> Result<StrongModuleGroebnerBasis, ModuleGroebnerError> {
    if basis.len() != transformations.len()
        || transformations
            .iter()
            .any(|row| row.len() != input.generator_count())
    {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "the integral module kernel returned an invalid change-matrix shape",
        ));
    }
    let ring = input.ambient().ring();
    let basis_values = basis
        .into_iter()
        .map(|value| from_integer_kernel_module_element(input.ambient(), value))
        .collect::<Result<Vec<_>, _>>()?;
    let leading_terms = basis_values
        .iter()
        .map(|value| {
            integral_leading_term(value, &order)
                .map(|(monomial, coefficient)| StrongModuleTerm::new(monomial, coefficient))
                .ok_or(ModuleGroebnerError::InvalidEvidence(
                    "the integral strong basis projected a zero generator",
                ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let basis = integral_submodule_from_generators(input.ambient(), basis_values)?;
    let transformation_columns = transformations
        .into_iter()
        .map(|column| {
            column
                .into_iter()
                .map(|polynomial| {
                    from_integer_kernel_polynomial(ring, polynomial)
                        .map_err(super::map_polynomial_groebner_error)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let transformation = ModuleHomomorphism::from_coordinate_columns(
        basis.generator_map().source().clone(),
        input.generator_map().source().clone(),
        transformation_columns,
    )?;
    Ok(StrongModuleGroebnerBasis {
        input: input.clone(),
        order,
        basis,
        transformation,
        leading_terms,
    })
}

fn strong_relations_submodule(
    input: &Submodule<PolynomialRing<IntegerRing>>,
) -> Result<Submodule<PolynomialRing<IntegerRing>>, ModuleGroebnerError> {
    let source = input.generator_map().source();
    if source.rank() == 0 {
        return Submodule::from_generators(source.clone(), Vec::new()).map_err(Into::into);
    }
    let ring = input.ambient().ring();
    let values = map_columns(input.generator_map())?;
    let relations = if let Some(kernel_values) =
        to_integer_kernel_module_elements(&values, ring, input.ambient().rank())?
    {
        let output = groebner_kernel::strong_module_relations(
            &kernel_values,
            &groebner_kernel::StrongGroebnerOptions::default(),
        )
        .map_err(map_kernel_module_error)?;
        output
            .generators
            .into_iter()
            .map(|value| from_integer_kernel_module_element(source, value))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        strong_relations_arbitrary(input)?
    };
    let result = integral_submodule_from_generators(source, relations)?;
    for relation in map_columns(result.generator_map())? {
        if !is_zero(&input.generator_map().apply(&relation)?) {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "an integral graph-elimination relation does not map to zero",
            ));
        }
    }
    Ok(result)
}

fn strong_relations_arbitrary(
    input: &Submodule<PolynomialRing<IntegerRing>>,
) -> Result<Vec<IntegralModuleElement>, ModuleGroebnerError> {
    let source = input.generator_map().source();
    let target_rank = input.ambient().rank();
    let source_rank = source.rank();
    let ring = input.ambient().ring();
    let graph_ambient = FreeModule::new(ring.clone(), target_rank.saturating_add(source_rank));
    let graph_values = map_columns(input.generator_map())?
        .into_iter()
        .enumerate()
        .map(|(source_component, value)| {
            let mut coordinates = value.coordinates().values().collect::<Vec<_>>();
            coordinates.extend((0..source_rank).map(|_| ring.zero()));
            coordinates[target_rank + source_component] = ring.one();
            element_from_coordinates(graph_ambient.clone(), coordinates).map_err(Into::into)
        })
        .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
    let graph = Submodule::from_generators(graph_ambient.clone(), graph_values)?;
    let elimination_order = ModuleMonomialOrder::PositionOverTerm(ring.term_order());
    let options = groebner_kernel::StrongGroebnerOptions::default();
    let graph_basis = strong_module_basis_arbitrary(&graph, &elimination_order, &options)?;
    let mut relations = graph_basis
        .into_iter()
        .filter(|entry| {
            entry
                .value
                .coordinates()
                .values()
                .take(target_rank)
                .all(|polynomial| polynomial.is_zero())
        })
        .map(|entry| {
            element_from_coordinates(
                source.clone(),
                entry
                    .value
                    .coordinates()
                    .values()
                    .skip(target_rank)
                    .collect(),
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>, ModuleGroebnerError>>()?;
    if !relations.is_empty() {
        let relation_input = Submodule::from_generators(source.clone(), relations)?;
        relations = strong_module_basis_arbitrary(&relation_input, &elimination_order, &options)?
            .into_iter()
            .map(|entry| entry.value)
            .collect();
    }
    for relation in &relations {
        if !is_zero(&input.generator_map().apply(relation)?) {
            return Err(ModuleGroebnerError::InvalidEvidence(
                "an arbitrary-exponent graph-elimination relation does not map to zero",
            ));
        }
    }
    Ok(relations)
}

fn to_integer_kernel_module_elements(
    values: &[ModuleElement<PolynomialRing<IntegerRing>>],
    ring: &PolynomialRing<IntegerRing>,
    rank: usize,
) -> Result<Option<Vec<groebner_kernel::SparseModuleElement<Integer>>>, ModuleGroebnerError> {
    let Some(order) = crate::commalg::to_kernel_order(&ring.term_order()) else {
        return Ok(None);
    };
    let projected = values
        .iter()
        .map(|value| {
            if value.module().ring() != ring || value.module().rank() != rank {
                return Err(ModuleGroebnerError::InvalidEvidence(
                    "an integral module value has an incompatible ambient module",
                ));
            }
            let terms = value
                .coordinates()
                .values()
                .enumerate()
                .flat_map(|(component, polynomial)| {
                    polynomial
                        .terms()
                        .map(move |(exponent, coefficient)| {
                            (component, exponent.to_vec(), coefficient.clone())
                        })
                        .collect::<Vec<_>>()
                })
                .map(|(component, exponent, coefficient)| {
                    exponent_to_kernel(&exponent).map(|monomial| {
                        (
                            groebner_kernel::SparseModuleTerm::new(monomial, component),
                            coefficient,
                        )
                    })
                })
                .collect::<Option<Vec<_>>>();
            Ok(terms.map(|terms| {
                groebner_kernel::SparseModuleElement::new(
                    ring.variables().len(),
                    rank,
                    order,
                    terms,
                    &groebner_kernel::IntegerCoefficientRing,
                )
            }))
        })
        .collect::<Result<Option<Vec<_>>, ModuleGroebnerError>>()?;
    Ok(projected)
}

fn from_integer_kernel_module_element(
    ambient: &FreeModule<PolynomialRing<IntegerRing>>,
    value: groebner_kernel::SparseModuleElement<Integer>,
) -> Result<ModuleElement<PolynomialRing<IntegerRing>>, ModuleGroebnerError> {
    if value.rank() != ambient.rank() || value.arity() != ambient.ring().variables().len() {
        return Err(ModuleGroebnerError::InvalidEvidence(
            "the integral module kernel returned an incompatible value shape",
        ));
    }
    let coordinates = value
        .coordinates(&groebner_kernel::IntegerCoefficientRing)
        .into_iter()
        .map(|polynomial| {
            from_integer_kernel_polynomial(ambient.ring(), polynomial)
                .map_err(super::map_polynomial_groebner_error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    element_from_coordinates(ambient.clone(), coordinates).map_err(Into::into)
}

fn validate_integral_order(
    module: &FreeModule<PolynomialRing<IntegerRing>>,
    order: &ModuleMonomialOrder,
) -> Result<(), ModuleGroebnerError> {
    if order.base_order() != &module.ring().term_order() {
        return Err(ModuleGroebnerError::OrderMismatch);
    }
    validate_schreyer_frame(order, module.rank(), module.ring().variables().len())
}

fn integral_leading_term(
    value: &ModuleElement<PolynomialRing<IntegerRing>>,
    order: &ModuleMonomialOrder,
) -> Option<(ModuleMonomial, Integer)> {
    let mut leading = None;
    for (component, polynomial) in value.coordinates().values().enumerate() {
        let Some((exponent, coefficient)) = polynomial.terms().next() else {
            continue;
        };
        let candidate = ModuleMonomial::new(exponent.to_vec(), component);
        if leading.as_ref().is_none_or(|(current, _)| {
            compare_module_monomials(&candidate, current, order) == core::cmp::Ordering::Greater
        }) {
            leading = Some((candidate, coefficient.clone()));
        }
    }
    leading
}

fn integral_submodule_from_generators(
    ambient: &FreeModule<PolynomialRing<IntegerRing>>,
    generators: Vec<ModuleElement<PolynomialRing<IntegerRing>>>,
) -> Result<Submodule<PolynomialRing<IntegerRing>>, ModuleGroebnerError> {
    let ring = ambient.ring();
    let source = if let Some(ambient_shifts) = ambient.degree_shifts() {
        if ambient.grading_rank() != Some(ring.grading().rank()) {
            return Err(ModuleError::GradingMismatch {
                operation: "integral homogeneous generator source",
            }
            .into());
        }
        let inferred = generators
            .iter()
            .map(|generator| {
                integral_homogeneous_degree(generator, ring.grading(), ambient_shifts)?
                    .map(|degree| degree.neg())
                    .ok_or(ModuleGroebnerError::InvalidEvidence(
                        "an integral strong basis contains zero",
                    ))
            })
            .collect::<Result<Vec<_>, _>>();
        match inferred {
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

fn integral_homogeneous_degree(
    value: &ModuleElement<PolynomialRing<IntegerRing>>,
    grading: &FreeAbelianGrading,
    shifts: &[crate::commalg::Degree],
) -> Result<Option<crate::commalg::Degree>, ModuleGroebnerError> {
    let mut homogeneous_degree = None;
    for (component, polynomial) in value.coordinates().values().enumerate() {
        let shift = shifts
            .get(component)
            .ok_or(ModuleGroebnerError::InvalidEvidence(
                "the ambient integral module has incomplete degree shifts",
            ))?;
        for (exponent, _) in polynomial.terms() {
            let degree = grading
                .degree(exponent)
                .and_then(|degree| degree.sub(shift))
                .map_err(|_| {
                    ModuleGroebnerError::InvalidEvidence(
                        "an integral module term has an incompatible grading degree",
                    )
                })?;
            match &homogeneous_degree {
                None => homogeneous_degree = Some(degree),
                Some(known) if known == &degree => {}
                Some(_) => {
                    return Err(ModuleGroebnerError::InvalidEvidence(
                        "the integral module basis is not homogeneous",
                    ));
                }
            }
        }
    }
    Ok(homogeneous_degree)
}

#[derive(Serialize)]
struct StrongModuleGroebnerBasisWireRef<'a> {
    #[serde(rename = "type")]
    type_id: &'static str,
    input: &'a Submodule<PolynomialRing<IntegerRing>>,
    order: &'a ModuleMonomialOrder,
    basis: &'a Submodule<PolynomialRing<IntegerRing>>,
    transformation: &'a ModuleHomomorphism<PolynomialRing<IntegerRing>>,
    leading_terms: &'a [StrongModuleTerm],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrongModuleGroebnerBasisWire {
    #[serde(rename = "type")]
    type_id: String,
    input: Submodule<PolynomialRing<IntegerRing>>,
    order: ModuleMonomialOrder,
    basis: Submodule<PolynomialRing<IntegerRing>>,
    transformation: ModuleHomomorphism<PolynomialRing<IntegerRing>>,
    leading_terms: Vec<StrongModuleTerm>,
}

impl Serialize for StrongModuleGroebnerBasis {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        StrongModuleGroebnerBasisWireRef {
            type_id: Self::TYPE_ID,
            input: &self.input,
            order: &self.order,
            basis: &self.basis,
            transformation: &self.transformation,
            leading_terms: &self.leading_terms,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StrongModuleGroebnerBasis {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StrongModuleGroebnerBasisWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            input: wire.input,
            order: wire.order,
            basis: wire.basis,
            transformation: wire.transformation,
            leading_terms: wire.leading_terms,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

#[derive(Serialize)]
struct StrongModuleReductionWireRef<'a> {
    #[serde(rename = "type")]
    type_id: &'static str,
    basis: &'a StrongModuleGroebnerBasis,
    value: &'a ModuleElement<PolynomialRing<IntegerRing>>,
    quotient: &'a ModuleElement<PolynomialRing<IntegerRing>>,
    original_combination: &'a ModuleElement<PolynomialRing<IntegerRing>>,
    remainder: &'a ModuleElement<PolynomialRing<IntegerRing>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrongModuleReductionWire {
    #[serde(rename = "type")]
    type_id: String,
    basis: StrongModuleGroebnerBasis,
    value: ModuleElement<PolynomialRing<IntegerRing>>,
    quotient: ModuleElement<PolynomialRing<IntegerRing>>,
    original_combination: ModuleElement<PolynomialRing<IntegerRing>>,
    remainder: ModuleElement<PolynomialRing<IntegerRing>>,
}

impl Serialize for StrongModuleReduction {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        StrongModuleReductionWireRef {
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

impl<'de> Deserialize<'de> for StrongModuleReduction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StrongModuleReductionWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
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

impl PartialEq for StrongModuleGroebnerBasis {
    fn eq(&self, other: &Self) -> bool {
        self.input.ambient() == other.input.ambient()
            && self.input.generator_map() == other.input.generator_map()
            && self.order == other.order
            && self.basis.ambient() == other.basis.ambient()
            && self.basis.generator_map() == other.basis.generator_map()
            && self.transformation == other.transformation
            && self.leading_terms == other.leading_terms
    }
}

impl Eq for StrongModuleGroebnerBasis {}

impl fmt::Debug for StrongModuleGroebnerBasis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StrongModuleGroebnerBasis")
            .field("input_generators", &self.input.generator_count())
            .field("basis_generators", &self.basis.generator_count())
            .field("order", &self.order)
            .finish()
    }
}

impl fmt::Display for StrongModuleGroebnerBasis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "StrongModuleGroebnerBasis(input_generators={}, basis_generators={}, order={})",
            self.input.generator_count(),
            self.basis.generator_count(),
            self.order
        )
    }
}

impl SemanticType for StrongModuleGroebnerBasis {
    const TYPE_ID: &'static str = "noether/homalg/StrongModuleGroebnerBasis@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &STRONG_MODULE_GROEBNER_BASIS_DESCRIPTION
    }
}

impl PartialEq for StrongModuleReduction {
    fn eq(&self, other: &Self) -> bool {
        self.basis == other.basis
            && self.value == other.value
            && self.quotient == other.quotient
            && self.original_combination == other.original_combination
            && self.remainder == other.remainder
    }
}

impl Eq for StrongModuleReduction {}

impl fmt::Debug for StrongModuleReduction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StrongModuleReduction")
            .field("member", &self.is_member())
            .finish()
    }
}

impl fmt::Display for StrongModuleReduction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "StrongModuleReduction(member={})",
            self.is_member()
        )
    }
}

impl SemanticType for StrongModuleReduction {
    const TYPE_ID: &'static str = "noether/homalg/StrongModuleReduction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.to_json()?.into_bytes())
    }

    fn describe() -> &'static TypeDescription {
        &STRONG_MODULE_REDUCTION_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use crate::commalg::{MonomialOrder, Polynomial, PolynomialRing};
    use crate::{Integer, Z};

    use super::*;

    fn polynomial(
        ring: &PolynomialRing<IntegerRing>,
        terms: &[(i64, i64, i64)],
    ) -> Polynomial<IntegerRing> {
        ring.from_terms(terms.iter().map(|(x, y, coefficient)| {
            (
                vec![Integer::from(*x), Integer::from(*y)],
                Integer::from(*coefficient),
            )
        }))
        .unwrap()
    }

    fn element(
        module: &FreeModule<PolynomialRing<IntegerRing>>,
        coordinates: Vec<Polynomial<IntegerRing>>,
    ) -> ModuleElement<PolynomialRing<IntegerRing>> {
        element_from_coordinates(module.clone(), coordinates).unwrap()
    }

    #[test]
    fn strong_integral_module_basis_reductions_and_lifts_verify() {
        let ring =
            PolynomialRing::new(Z, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let ambient = FreeModule::new(ring.clone(), 2);
        let first = element(&ambient, vec![polynomial(&ring, &[(1, 0, 2)]), ring.zero()]);
        let second = element(&ambient, vec![polynomial(&ring, &[(1, 0, 3)]), ring.zero()]);
        let third = element(&ambient, vec![ring.zero(), polynomial(&ring, &[(0, 1, 5)])]);
        let input =
            Submodule::from_generators(ambient.clone(), vec![first.clone(), second.clone(), third])
                .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let basis = input.strong_groebner_basis(order.clone()).unwrap();
        basis.verify().unwrap();
        assert_eq!(basis.leading_terms()[0].coefficient(), &Integer::one());
        assert_eq!(
            StrongModuleGroebnerBasis::from_json(&basis.to_json().unwrap()).unwrap(),
            basis
        );
        let value = first.add(&second).unwrap();
        let reduction = basis.reduce(&value).unwrap();
        assert!(reduction.is_member());
        reduction.verify().unwrap();
        assert_eq!(
            StrongModuleReduction::from_json(&reduction.to_json().unwrap()).unwrap(),
            reduction
        );
        assert_eq!(
            input
                .generator_map()
                .apply(reduction.lift().unwrap())
                .unwrap(),
            value
        );

        let outside = element(&ambient, vec![ring.zero(), ring.one()]);
        assert!(!basis.contains(&outside).unwrap());
        assert_eq!(
            basis.initial_submodule().unwrap().generator_count(),
            basis.basis().generator_count()
        );
        assert_eq!(
            first
                .leading_module_term(&order)
                .unwrap()
                .unwrap()
                .coefficient(),
            &Integer::from(2)
        );
    }

    #[test]
    fn integral_syzygies_and_map_kernel_use_graph_elimination() {
        let ring = PolynomialRing::new(Z, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        let ambient = FreeModule::new(ring.clone(), 1);
        let first = element(&ambient, vec![polynomial(&ring, &[(1, 0, 2)])]);
        let second = element(&ambient, vec![polynomial(&ring, &[(1, 0, 3)])]);
        let input =
            Submodule::from_generators(ambient, vec![first.clone(), second.clone()]).unwrap();
        let order = ModuleMonomialOrder::PositionOverTerm(ring.term_order());
        let basis = input.strong_groebner_basis(order.clone()).unwrap();
        let relations = basis.syzygies().unwrap();
        assert_eq!(relations.generator_count(), 1);
        for relation in map_columns(relations.generator_map()).unwrap() {
            assert!(is_zero(&input.generator_map().apply(&relation).unwrap()));
        }

        let kernel = input.generator_map().kernel(order).unwrap();
        assert_eq!(kernel.generator_map(), relations.generator_map());
    }

    #[test]
    fn arbitrary_exponent_floor_agrees_with_the_packed_strong_basis() {
        let ring = PolynomialRing::new(Z, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        let ambient = FreeModule::new(ring.clone(), 2);
        let generators = vec![
            element(&ambient, vec![polynomial(&ring, &[(1, 0, 2)]), ring.zero()]),
            element(&ambient, vec![polynomial(&ring, &[(1, 0, 3)]), ring.zero()]),
            element(&ambient, vec![ring.zero(), polynomial(&ring, &[(0, 1, 5)])]),
        ];
        let input = Submodule::from_generators(ambient, generators.clone()).unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let packed = input.strong_groebner_basis(order.clone()).unwrap();
        let arbitrary = project_arbitrary_strong_module_basis(
            &input,
            order.clone(),
            strong_module_basis_arbitrary(
                &input,
                &order,
                &groebner_kernel::StrongGroebnerOptions::default(),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            arbitrary.basis().generator_map(),
            packed.basis().generator_map()
        );
        assert_eq!(arbitrary.leading_terms(), packed.leading_terms());
        let packed_remainder = packed.normal_form(&generators[0]).unwrap();
        let arbitrary_values = map_columns(arbitrary.basis().generator_map()).unwrap();
        let (arbitrary_remainder, _) = strong_normal_form_arbitrary(
            &generators[0],
            &arbitrary_values,
            &order,
            MAX_REDUCTION_STEPS,
        )
        .unwrap();
        assert_eq!(arbitrary_remainder, packed_remainder);
    }

    #[test]
    fn arbitrary_exponents_keep_strong_bases_reductions_and_relations_exact() {
        let ring = PolynomialRing::new(Z, ["x", "y"], MonomialOrder::Lexicographic).unwrap();
        let ambient = FreeModule::new(ring.clone(), 2);
        let huge = Integer::from(u64::from(u32::MAX) + 1);
        let huge_term = |coefficient: i64| {
            ring.from_terms([(
                vec![huge.clone(), Integer::zero()],
                Integer::from(coefficient),
            )])
            .unwrap()
        };
        let first = element(&ambient, vec![huge_term(2), ring.zero()]);
        let second = element(&ambient, vec![huge_term(3), ring.zero()]);
        let third = element(&ambient, vec![ring.zero(), polynomial(&ring, &[(0, 1, 5)])]);
        let input =
            Submodule::from_generators(ambient.clone(), vec![first.clone(), second.clone(), third])
                .unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let basis = input.strong_groebner_basis(order.clone()).unwrap();
        basis.verify().unwrap();
        assert!(basis.leading_terms().iter().any(|leading| {
            leading.monomial().exponent() == [huge.clone(), Integer::zero()]
                && leading.coefficient() == &Integer::one()
        }));

        let value = first.add(&second).unwrap();
        let reduction = basis.reduce(&value).unwrap();
        assert!(reduction.is_member());
        reduction.verify().unwrap();
        assert_eq!(
            input
                .generator_map()
                .apply(reduction.lift().expect("a member has an exact lift"))
                .unwrap(),
            value
        );
        let outside = element(&ambient, vec![ring.one(), ring.zero()]);
        assert!(!basis.contains(&outside).unwrap());

        let relations = basis.syzygies().unwrap();
        assert!(relations.generator_count() > 0);
        for relation in map_columns(relations.generator_map()).unwrap() {
            assert!(is_zero(&input.generator_map().apply(&relation).unwrap()));
        }
        let kernel = input.generator_map().kernel(order).unwrap();
        assert_eq!(kernel.generator_map(), relations.generator_map());
    }
}
