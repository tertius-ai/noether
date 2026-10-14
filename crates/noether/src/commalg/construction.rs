// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Symmetric, Rees, and associated-graded algebra presentations.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::homalg::{ModuleError, PresentedModule};
use crate::{
    ContextMismatchError, Field, Integer, NoetherError, Ring, SemanticType, TypeDescription,
};

use super::{
    Degree, FreeAbelianGrading, GroebnerError, Ideal, MonomialOrder, Polynomial,
    PolynomialContextError, PolynomialError, PolynomialRing, QuotientElement, QuotientRing,
};

const SYMMETRIC_ALGEBRA_TYPE_ID: &str = "noether/commalg/SymmetricAlgebra@1";
const REES_ALGEBRA_TYPE_ID: &str = "noether/commalg/ReesAlgebra@1";
const ASSOCIATED_GRADED_ALGEBRA_TYPE_ID: &str = "noether/commalg/AssociatedGradedAlgebra@1";
const MAX_CONSTRUCTION_VARIABLES: usize = 256;

static SYMMETRIC_ALGEBRA_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: SYMMETRIC_ALGEBRA_TYPE_ID,
    rust_path: "noether::commalg::SymmetricAlgebra",
    python_path: Some("noether.commalg.SymmetricAlgebra"),
};
static REES_ALGEBRA_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: REES_ALGEBRA_TYPE_ID,
    rust_path: "noether::commalg::ReesAlgebra",
    python_path: Some("noether.commalg.ReesAlgebra"),
};
static ASSOCIATED_GRADED_ALGEBRA_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: ASSOCIATED_GRADED_ALGEBRA_TYPE_ID,
    rust_path: "noether::commalg::AssociatedGradedAlgebra",
    python_path: Some("noether.commalg.AssociatedGradedAlgebra"),
};

/// Failure to construct or verify a finitely presented construction algebra.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstructionAlgebraError {
    /// Polynomial-context construction failed.
    Context(PolynomialContextError),
    /// Polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Exact ideal or elimination arithmetic failed.
    Groebner(GroebnerError),
    /// Quotient endpoints did not agree.
    ContextMismatch(ContextMismatchError),
    /// Module presentation arithmetic failed.
    Module(ModuleError),
    /// A deterministic structural resource limit was exceeded.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// Retained structural evidence did not replay.
    InvalidEvidence(String),
}

impl fmt::Display for ConstructionAlgebraError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Module(error) => error.fmt(formatter),
            Self::ResourceLimit { resource, limit } => {
                write!(
                    formatter,
                    "construction algebra exceeded its {resource} limit of {limit}"
                )
            }
            Self::InvalidEvidence(detail) => {
                write!(formatter, "invalid construction-algebra evidence: {detail}")
            }
        }
    }
}

impl std::error::Error for ConstructionAlgebraError {}

macro_rules! impl_error_from {
    ($source:ty, $variant:ident) => {
        impl From<$source> for ConstructionAlgebraError {
            fn from(value: $source) -> Self {
                Self::$variant(value)
            }
        }
    };
}
impl_error_from!(PolynomialContextError, Context);
impl_error_from!(PolynomialError, Polynomial);
impl_error_from!(GroebnerError, Groebner);
impl_error_from!(ContextMismatchError, ContextMismatch);
impl_error_from!(ModuleError, Module);

/// The symmetric algebra of a finitely presented module over `k[x_1,...,x_n]`.
///
/// If `R^s -> R^m -> M -> 0` has matrix `(a_ij)`, this is
/// `R[T_1,...,T_m] / (sum_i a_ij T_i : 1 <= j <= s)`. Base variables have
/// construction degree zero and the `T_i` have degree one.
#[derive(Clone, PartialEq, Eq)]
pub struct SymmetricAlgebra<R: Field> {
    module: PresentedModule<PolynomialRing<R>>,
    ambient_ring: PolynomialRing<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
    algebra: QuotientRing<PolynomialRing<R>>,
}

impl<R: Field> SymmetricAlgebra<R> {
    /// Construct the canonical quotient presentation.
    pub fn new(
        module: PresentedModule<PolynomialRing<R>>,
    ) -> Result<Self, ConstructionAlgebraError> {
        let base = module.generator_module().ring();
        check_variable_count(base.variables().len(), module.generator_count(), 0)?;
        let names = construction_names(base.variables(), "sym", module.generator_count());
        let ambient_ring = construction_ring(
            base.coefficient_ring().clone(),
            names,
            base.variables().len(),
        )?;
        let variables = ambient_ring.generators()?;
        let matrix = module.presentation().matrix().rows().collect::<Vec<_>>();
        let relations = (0..module.relation_count())
            .map(|column| {
                (0..module.generator_count()).try_fold(ambient_ring.zero(), |relation, row| {
                    let coefficient = lift_polynomial(&matrix[row][column], &ambient_ring)?;
                    let term = (&coefficient * &variables[base.variables().len() + row])?;
                    (&relation + &term).map_err(Into::into)
                })
            })
            .collect::<Result<Vec<_>, ConstructionAlgebraError>>()?;
        let defining_ideal = ambient_ring.ideal(relations)?;
        let algebra = ambient_ring.quotient(&defining_ideal)?;
        Ok(Self {
            module,
            ambient_ring,
            defining_ideal,
            algebra,
        })
    }

    /// Source presented module.
    pub fn module(&self) -> &PresentedModule<PolynomialRing<R>> {
        &self.module
    }
    /// Flat polynomial presentation ring `k[x,T]`.
    pub fn ambient_ring(&self) -> &PolynomialRing<R> {
        &self.ambient_ring
    }
    /// Canonical ideal of linear module relations.
    pub fn defining_ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.defining_ideal
    }
    /// Exact quotient-algebra context.
    pub fn algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.algebra
    }
    /// Images of the base polynomial variables.
    pub fn base_images(
        &self,
    ) -> Result<Vec<QuotientElement<PolynomialRing<R>>>, ConstructionAlgebraError> {
        quotient_variables(
            &self.ambient_ring,
            &self.algebra,
            0,
            self.module.generator_module().ring().variables().len(),
        )
    }
    /// Degree-one images of the presented module generators.
    pub fn generator_images(
        &self,
    ) -> Result<Vec<QuotientElement<PolynomialRing<R>>>, ConstructionAlgebraError> {
        let base_count = self.module.generator_module().ring().variables().len();
        quotient_variables(
            &self.ambient_ring,
            &self.algebra,
            base_count,
            self.module.generator_count(),
        )
    }
    /// Recompute the presentation and every defining relation.
    pub fn verify(&self) -> Result<(), ConstructionAlgebraError> {
        if Self::new(self.module.clone())? != *self {
            return Err(ConstructionAlgebraError::InvalidEvidence(
                "recomputation changed the symmetric algebra".to_owned(),
            ));
        }
        Ok(())
    }
    /// Serialize the complete verified presentation.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
        PresentedModule<PolynomialRing<R>>: Serialize,
        Ideal<PolynomialRing<R>>: Serialize,
        QuotientRing<PolynomialRing<R>>: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> PresentedModule<PolynomialRing<R>> {
    /// Construct this module's symmetric algebra.
    pub fn symmetric_algebra(&self) -> Result<SymmetricAlgebra<R>, ConstructionAlgebraError> {
        SymmetricAlgebra::new(self.clone())
    }
}

/// The Rees algebra `R[It]` of an ideal in a field polynomial ring.
///
/// For canonical ideal generators `g_i`, the defining ideal is obtained by
/// eliminating `t` from `(T_i-g_i t)` in `k[x,T,t]`.
#[derive(Clone, PartialEq, Eq)]
pub struct ReesAlgebra<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    ambient_ring: PolynomialRing<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
    algebra: QuotientRing<PolynomialRing<R>>,
}

impl<R: Field> ReesAlgebra<R> {
    /// Construct the canonical elimination presentation.
    pub fn new(source: Ideal<PolynomialRing<R>>) -> Result<Self, ConstructionAlgebraError> {
        let base = source.ring();
        let generator_count = source.generators().len();
        check_variable_count(base.variables().len(), generator_count, 1)?;
        let names = construction_names(base.variables(), "rees", generator_count);
        let ambient_ring = construction_ring(
            base.coefficient_ring().clone(),
            names.clone(),
            base.variables().len(),
        )?;
        let mut graph_names = names;
        graph_names.push(fresh_name(&graph_names, "rees_t"));
        let graph_ring = construction_ring(
            base.coefficient_ring().clone(),
            graph_names,
            base.variables().len(),
        )?;
        let graph_variables = graph_ring.generators()?;
        let t = graph_variables.last().ok_or_else(|| {
            ConstructionAlgebraError::InvalidEvidence(
                "the Rees graph ring has no homogenizing variable".to_owned(),
            )
        })?;
        let base_count = base.variables().len();
        let graph_relations = source
            .generators()
            .iter()
            .enumerate()
            .map(|(index, generator)| {
                let lifted = lift_polynomial(generator, &graph_ring)?;
                let product = (&lifted * t)?;
                (&graph_variables[base_count + index] - &product).map_err(Into::into)
            })
            .collect::<Result<Vec<_>, ConstructionAlgebraError>>()?;
        let graph_ideal = graph_ring.ideal(graph_relations)?;
        let eliminated = graph_ideal.eliminate(&[graph_ring.variables().len() - 1])?;
        let relations = eliminated
            .generators()
            .iter()
            .map(|value| project_last_variable(value, &ambient_ring))
            .collect::<Result<Vec<_>, _>>()?;
        let defining_ideal = ambient_ring.ideal(relations)?;
        let algebra = ambient_ring.quotient(&defining_ideal)?;
        Ok(Self {
            source,
            ambient_ring,
            defining_ideal,
            algebra,
        })
    }

    /// Source ideal whose blowup algebra is represented.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }
    /// Flat polynomial presentation ring `k[x,T]`.
    pub fn ambient_ring(&self) -> &PolynomialRing<R> {
        &self.ambient_ring
    }
    /// Canonical Rees ideal obtained by elimination.
    pub fn defining_ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.defining_ideal
    }
    /// Exact quotient-algebra context.
    pub fn algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.algebra
    }
    /// Images of the base polynomial variables.
    pub fn base_images(
        &self,
    ) -> Result<Vec<QuotientElement<PolynomialRing<R>>>, ConstructionAlgebraError> {
        quotient_variables(
            &self.ambient_ring,
            &self.algebra,
            0,
            self.source.ring().variables().len(),
        )
    }
    /// Degree-one images corresponding to the canonical ideal generators.
    pub fn generator_images(
        &self,
    ) -> Result<Vec<QuotientElement<PolynomialRing<R>>>, ConstructionAlgebraError> {
        let base_count = self.source.ring().variables().len();
        quotient_variables(
            &self.ambient_ring,
            &self.algebra,
            base_count,
            self.source.generators().len(),
        )
    }
    /// Recompute the elimination ideal and quotient presentation.
    pub fn verify(&self) -> Result<(), ConstructionAlgebraError> {
        if Self::new(self.source.clone())? != *self {
            return Err(ConstructionAlgebraError::InvalidEvidence(
                "recomputation changed the Rees algebra".to_owned(),
            ));
        }
        Ok(())
    }
    /// Serialize the complete verified presentation.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
        Ideal<PolynomialRing<R>>: Serialize,
        QuotientRing<PolynomialRing<R>>: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> Ideal<PolynomialRing<R>> {
    /// Construct the Rees algebra of this ideal.
    pub fn rees_algebra(&self) -> Result<ReesAlgebra<R>, ConstructionAlgebraError> {
        ReesAlgebra::new(self.clone())
    }
    /// Construct the associated graded algebra `gr_I(R)`.
    pub fn associated_graded_algebra(
        &self,
    ) -> Result<AssociatedGradedAlgebra<R>, ConstructionAlgebraError> {
        AssociatedGradedAlgebra::new(self.clone())
    }
}

/// The associated graded algebra `gr_I(R) = Rees_I(R) / I Rees_I(R)`.
#[derive(Clone, PartialEq, Eq)]
pub struct AssociatedGradedAlgebra<R: Field> {
    rees_algebra: ReesAlgebra<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
    algebra: QuotientRing<PolynomialRing<R>>,
}

impl<R: Field> AssociatedGradedAlgebra<R> {
    /// Construct from the canonical Rees presentation and extended source ideal.
    pub fn new(source: Ideal<PolynomialRing<R>>) -> Result<Self, ConstructionAlgebraError> {
        let rees_algebra = ReesAlgebra::new(source)?;
        let extended = rees_algebra
            .source
            .generators()
            .iter()
            .map(|value| lift_polynomial(value, &rees_algebra.ambient_ring))
            .collect::<Result<Vec<_>, _>>()?;
        let extended = rees_algebra.ambient_ring.ideal(extended)?;
        let defining_ideal = rees_algebra.defining_ideal.sum(&extended)?;
        let algebra = rees_algebra.ambient_ring.quotient(&defining_ideal)?;
        Ok(Self {
            rees_algebra,
            defining_ideal,
            algebra,
        })
    }
    /// Canonical Rees algebra before quotienting by the source ideal.
    pub fn rees_algebra(&self) -> &ReesAlgebra<R> {
        &self.rees_algebra
    }
    /// Defining ideal of the flat associated-graded presentation.
    pub fn defining_ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.defining_ideal
    }
    /// Exact quotient-algebra context.
    pub fn algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.algebra
    }
    /// Images of base and degree-one variables in presentation order.
    pub fn generator_images(
        &self,
    ) -> Result<Vec<QuotientElement<PolynomialRing<R>>>, ConstructionAlgebraError> {
        quotient_variables(
            &self.rees_algebra.ambient_ring,
            &self.algebra,
            0,
            self.rees_algebra.ambient_ring.variables().len(),
        )
    }
    /// Recompute the Rees presentation and quotient by the source ideal.
    pub fn verify(&self) -> Result<(), ConstructionAlgebraError> {
        if Self::new(self.rees_algebra.source.clone())? != *self {
            return Err(ConstructionAlgebraError::InvalidEvidence(
                "recomputation changed the associated graded algebra".to_owned(),
            ));
        }
        Ok(())
    }
    /// Serialize the complete verified presentation.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
        Ideal<PolynomialRing<R>>: Serialize,
        QuotientRing<PolynomialRing<R>>: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

fn check_variable_count(
    base: usize,
    generators: usize,
    auxiliary: usize,
) -> Result<(), ConstructionAlgebraError> {
    let count = base
        .checked_add(generators)
        .and_then(|value| value.checked_add(auxiliary));
    if count.is_none_or(|value| value > MAX_CONSTRUCTION_VARIABLES) {
        return Err(ConstructionAlgebraError::ResourceLimit {
            resource: "variable count",
            limit: MAX_CONSTRUCTION_VARIABLES,
        });
    }
    Ok(())
}

fn construction_names(base: &[String], stem: &str, count: usize) -> Vec<String> {
    let mut names = base.to_vec();
    for index in 0..count {
        let name = fresh_name(&names, &format!("{stem}_{index}"));
        names.push(name);
    }
    names
}

fn fresh_name(names: &[String], preferred: &str) -> String {
    if !names.iter().any(|name| name == preferred) {
        return preferred.to_owned();
    }
    let mut suffix = 1_usize;
    loop {
        let candidate = format!("{preferred}_{suffix}");
        if !names.iter().any(|name| name == &candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

fn construction_ring<R: Field>(
    field: R,
    names: Vec<String>,
    base_count: usize,
) -> Result<PolynomialRing<R>, ConstructionAlgebraError> {
    let degrees = (0..names.len())
        .map(|index| {
            Degree::new(vec![if index < base_count {
                Integer::zero()
            } else {
                Integer::one()
            }])
        })
        .collect();
    let grading = FreeAbelianGrading::new(1, degrees).map_err(PolynomialContextError::from)?;
    PolynomialRing::with_grading(field, names, MonomialOrder::default(), grading)
        .map_err(Into::into)
}

fn lift_polynomial<R: Field>(
    value: &Polynomial<R>,
    target: &PolynomialRing<R>,
) -> Result<Polynomial<R>, ConstructionAlgebraError> {
    let extra = target
        .variables()
        .len()
        .checked_sub(value.ring().variables().len())
        .ok_or_else(|| {
            ConstructionAlgebraError::InvalidEvidence(
                "construction target has fewer variables than its source".to_owned(),
            )
        })?;
    target
        .from_terms(value.terms().map(|(exponent, coefficient)| {
            let mut exponent = exponent.to_vec();
            exponent.extend((0..extra).map(|_| Integer::zero()));
            (exponent, coefficient.clone())
        }))
        .map_err(Into::into)
}

fn project_last_variable<R: Field>(
    value: &Polynomial<R>,
    target: &PolynomialRing<R>,
) -> Result<Polynomial<R>, ConstructionAlgebraError> {
    if value
        .terms()
        .any(|(exponent, _)| exponent.last().is_some_and(|power| !power.is_zero()))
    {
        return Err(ConstructionAlgebraError::InvalidEvidence(
            "elimination retained the Rees auxiliary variable".to_owned(),
        ));
    }
    target
        .from_terms(value.terms().map(|(exponent, coefficient)| {
            (
                exponent[..target.variables().len()].to_vec(),
                coefficient.clone(),
            )
        }))
        .map_err(Into::into)
}

fn quotient_variables<R: Field>(
    ambient: &PolynomialRing<R>,
    quotient: &QuotientRing<PolynomialRing<R>>,
    skip: usize,
    take: usize,
) -> Result<Vec<QuotientElement<PolynomialRing<R>>>, ConstructionAlgebraError> {
    ambient
        .generators()?
        .into_iter()
        .skip(skip)
        .take(take)
        .map(|value| quotient.element(&value).map_err(Into::into))
        .collect()
}

macro_rules! impl_construction_value {
    ($name:ident, $type_id:expr, $description:ident, $display:literal) => {
        impl<R: Field> fmt::Debug for $name<R> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct($display)
                    .field("relations", &self.defining_ideal.generators().len())
                    .finish()
            }
        }
        impl<R: Field> fmt::Display for $name<R> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    concat!($display, "({} variables, {} relations)"),
                    self.algebra.base_ring().variables().len(),
                    self.defining_ideal.generators().len()
                )
            }
        }
        impl<R> SemanticType for $name<R>
        where
            R: Field + Serialize,
            R::Element: Serialize,
            Self: Serialize,
        {
            const TYPE_ID: &'static str = $type_id;
            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                serde_json::to_vec(self)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
            fn describe() -> &'static TypeDescription {
                &$description
            }
        }
    };
}
impl_construction_value!(
    SymmetricAlgebra,
    SYMMETRIC_ALGEBRA_TYPE_ID,
    SYMMETRIC_ALGEBRA_DESCRIPTION,
    "SymmetricAlgebra"
);
impl_construction_value!(
    ReesAlgebra,
    REES_ALGEBRA_TYPE_ID,
    REES_ALGEBRA_DESCRIPTION,
    "ReesAlgebra"
);
impl_construction_value!(
    AssociatedGradedAlgebra,
    ASSOCIATED_GRADED_ALGEBRA_TYPE_ID,
    ASSOCIATED_GRADED_ALGEBRA_DESCRIPTION,
    "AssociatedGradedAlgebra"
);

impl<R> Serialize for SymmetricAlgebra<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    PresentedModule<PolynomialRing<R>>: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SymmetricAlgebra", 5)?;
        state.serialize_field("type", SYMMETRIC_ALGEBRA_TYPE_ID)?;
        state.serialize_field("module", &self.module)?;
        state.serialize_field("ambient_ring", &self.ambient_ring)?;
        state.serialize_field("defining_ideal", &self.defining_ideal)?;
        state.serialize_field("algebra", &self.algebra)?;
        state.end()
    }
}

impl<R> Serialize for ReesAlgebra<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ReesAlgebra", 5)?;
        state.serialize_field("type", REES_ALGEBRA_TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("ambient_ring", &self.ambient_ring)?;
        state.serialize_field("defining_ideal", &self.defining_ideal)?;
        state.serialize_field("algebra", &self.algebra)?;
        state.end()
    }
}

impl<R> Serialize for AssociatedGradedAlgebra<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("AssociatedGradedAlgebra", 4)?;
        state.serialize_field("type", ASSOCIATED_GRADED_ALGEBRA_TYPE_ID)?;
        state.serialize_field("rees_algebra", &self.rees_algebra)?;
        state.serialize_field("defining_ideal", &self.defining_ideal)?;
        state.serialize_field("algebra", &self.algebra)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, PresentedModule<PolynomialRing<R>>: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>, QuotientRing<PolynomialRing<R>>: Deserialize<'de>"
    )
)]
struct SymmetricAlgebraWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    module: PresentedModule<PolynomialRing<R>>,
    ambient_ring: PolynomialRing<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
    algebra: QuotientRing<PolynomialRing<R>>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>, QuotientRing<PolynomialRing<R>>: Deserialize<'de>"
    )
)]
struct ReesAlgebraWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    ambient_ring: PolynomialRing<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
    algebra: QuotientRing<PolynomialRing<R>>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, ReesAlgebra<R>: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>, QuotientRing<PolynomialRing<R>>: Deserialize<'de>"
    )
)]
struct AssociatedGradedAlgebraWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    rees_algebra: ReesAlgebra<R>,
    defining_ideal: Ideal<PolynomialRing<R>>,
    algebra: QuotientRing<PolynomialRing<R>>,
}

fn deserialize_symmetric<'de, R, D>(deserializer: D) -> Result<SymmetricAlgebra<R>, D::Error>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
    PresentedModule<PolynomialRing<R>>: Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Deserialize<'de>,
    QuotientRing<PolynomialRing<R>>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = SymmetricAlgebraWire::<R>::deserialize(deserializer)?;
    if wire.type_id != SYMMETRIC_ALGEBRA_TYPE_ID {
        return Err(serde::de::Error::custom(
            "wrong symmetric-algebra type identity",
        ));
    }
    let rebuilt = SymmetricAlgebra::new(wire.module).map_err(serde::de::Error::custom)?;
    if rebuilt.ambient_ring != wire.ambient_ring
        || rebuilt.defining_ideal != wire.defining_ideal
        || rebuilt.algebra != wire.algebra
    {
        return Err(serde::de::Error::custom(
            "serialized symmetric algebra was not canonical",
        ));
    }
    Ok(rebuilt)
}

fn deserialize_rees<'de, R, D>(deserializer: D) -> Result<ReesAlgebra<R>, D::Error>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Deserialize<'de>,
    QuotientRing<PolynomialRing<R>>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = ReesAlgebraWire::<R>::deserialize(deserializer)?;
    if wire.type_id != REES_ALGEBRA_TYPE_ID {
        return Err(serde::de::Error::custom("wrong Rees-algebra type identity"));
    }
    let rebuilt = ReesAlgebra::new(wire.source).map_err(serde::de::Error::custom)?;
    if rebuilt.ambient_ring != wire.ambient_ring
        || rebuilt.defining_ideal != wire.defining_ideal
        || rebuilt.algebra != wire.algebra
    {
        return Err(serde::de::Error::custom(
            "serialized Rees algebra was not canonical",
        ));
    }
    Ok(rebuilt)
}

fn deserialize_associated_graded<'de, R, D>(
    deserializer: D,
) -> Result<AssociatedGradedAlgebra<R>, D::Error>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
    ReesAlgebra<R>: Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Deserialize<'de>,
    QuotientRing<PolynomialRing<R>>: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = AssociatedGradedAlgebraWire::<R>::deserialize(deserializer)?;
    if wire.type_id != ASSOCIATED_GRADED_ALGEBRA_TYPE_ID {
        return Err(serde::de::Error::custom(
            "wrong associated-graded-algebra type identity",
        ));
    }
    let rebuilt = AssociatedGradedAlgebra::new(wire.rees_algebra.source.clone())
        .map_err(serde::de::Error::custom)?;
    if rebuilt.rees_algebra != wire.rees_algebra
        || rebuilt.defining_ideal != wire.defining_ideal
        || rebuilt.algebra != wire.algebra
    {
        return Err(serde::de::Error::custom(
            "serialized associated graded algebra was not canonical",
        ));
    }
    Ok(rebuilt)
}

macro_rules! impl_construction_json {
    ($field:ty) => {
        impl<'de> Deserialize<'de> for SymmetricAlgebra<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserialize_symmetric(deserializer)
            }
        }
        impl SymmetricAlgebra<$field> {
            /// Deserialize and independently revalidate the presentation.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
        impl<'de> Deserialize<'de> for ReesAlgebra<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserialize_rees(deserializer)
            }
        }
        impl ReesAlgebra<$field> {
            /// Deserialize and independently revalidate the presentation.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
        impl<'de> Deserialize<'de> for AssociatedGradedAlgebra<$field> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                deserialize_associated_graded(deserializer)
            }
        }
        impl AssociatedGradedAlgebra<$field> {
            /// Deserialize and independently revalidate the presentation.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
    };
}
impl_construction_json!(crate::RationalField);
impl_construction_json!(crate::numthy::FiniteField);

#[cfg(test)]
mod tests {
    use crate::homalg::{FreeModule, ModuleHomomorphism};
    use crate::linalg::Matrix;
    use crate::Q;

    use super::*;

    #[test]
    fn symmetric_algebra_uses_linear_presentation_relations() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let x = ring.generator(0).unwrap();
        let relations = FreeModule::new(ring.clone(), 1);
        let generators = FreeModule::new(ring.clone(), 2);
        let matrix = Matrix::from_rows(ring.clone(), [[x], [ring.one()]]).unwrap();
        let map = ModuleHomomorphism::from_matrix(relations, generators, matrix).unwrap();
        let symmetric = PresentedModule::cokernel(map).symmetric_algebra().unwrap();
        let variables = symmetric.ambient_ring().generators().unwrap();
        let expected_relation =
            (&(&variables[0] * &variables[1]).unwrap() + &variables[2]).unwrap();
        let expected = symmetric.ambient_ring().ideal([expected_relation]).unwrap();
        assert_eq!(symmetric.defining_ideal(), &expected);
        symmetric.verify().unwrap();
        assert_eq!(
            SymmetricAlgebra::<crate::RationalField>::from_json(&symmetric.to_json().unwrap())
                .unwrap(),
            symmetric
        );
    }

    #[test]
    fn rees_and_associated_graded_present_the_coordinate_maximal_ideal() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let ideal = ring.ideal([x, y]).unwrap();
        let rees = ideal.rees_algebra().unwrap();
        assert_eq!(rees.defining_ideal().generators().len(), 1);
        rees.verify().unwrap();
        let graded = ideal.associated_graded_algebra().unwrap();
        assert_eq!(graded.defining_ideal().dimension().unwrap(), 2);
        graded.verify().unwrap();
        assert_eq!(
            ReesAlgebra::<crate::RationalField>::from_json(&rees.to_json().unwrap()).unwrap(),
            rees
        );
        assert_eq!(
            AssociatedGradedAlgebra::<crate::RationalField>::from_json(&graded.to_json().unwrap())
                .unwrap(),
            graded
        );
    }
}
