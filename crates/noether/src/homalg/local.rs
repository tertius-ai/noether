// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Mora standard bases for submodules over polynomial local rings.

use core::cmp::Ordering;
use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{
    LocalAlgebraError, LocalMonomialOrder, LocalPolynomial, LocalPolynomialRing, Polynomial,
    PolynomialError,
};
use crate::linalg::Vector;
use crate::{Field, Integer, NoetherError, Ring, SemanticType, TypeDescription};

use super::{FreeModule, ModuleElement, ModuleError, Submodule};

const MAX_LOCAL_MODULE_REDUCTION_STEPS: usize = 1_000_000;
const MAX_LOCAL_MODULE_PAIRS: usize = 1_000_000;

static LOCAL_MODULE_ORDER_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/LocalModuleOrder@1",
    rust_path: "noether::homalg::LocalModuleOrder",
    python_path: Some("noether.homalg.LocalModuleOrder"),
};

static LOCAL_MODULE_STANDARD_BASIS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/LocalModuleStandardBasis@1",
    rust_path: "noether::homalg::LocalModuleStandardBasis",
    python_path: Some("noether.homalg.LocalModuleStandardBasis"),
};

static LOCAL_MODULE_REDUCTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/LocalModuleReduction@1",
    rust_path: "noether::homalg::LocalModuleReduction",
    python_path: Some("noether.homalg.LocalModuleReduction"),
};

/// Whether polynomial terms or basis positions are compared first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LocalModuleOrder {
    /// Compare the local polynomial term before the basis position.
    #[default]
    TermOverPosition,
    /// Compare the basis position before the local polynomial term.
    PositionOverTerm,
}

impl LocalModuleOrder {
    /// Stable short name used by JSON and Python.
    pub const fn name(self) -> &'static str {
        match self {
            Self::TermOverPosition => "term_over_position",
            Self::PositionOverTerm => "position_over_term",
        }
    }

    /// Serialize this exact module order.
    pub fn to_json(self) -> Result<String, NoetherError> {
        serde_json::to_string(&self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize one exact module order.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl fmt::Display for LocalModuleOrder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl SemanticType for LocalModuleOrder {
    const TYPE_ID: &'static str = "noether/homalg/LocalModuleOrder@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(self.name().as_bytes().to_vec())
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_MODULE_ORDER_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalModuleOrderWire {
    #[serde(rename = "type")]
    type_id: String,
    name: String,
}

impl Serialize for LocalModuleOrder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        LocalModuleOrderWire {
            type_id: Self::TYPE_ID.to_owned(),
            name: self.name().to_owned(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LocalModuleOrder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalModuleOrderWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        match wire.name.as_str() {
            "term_over_position" => Ok(Self::TermOverPosition),
            "position_over_term" => Ok(Self::PositionOverTerm),
            _ => Err(serde::de::Error::custom("unknown local module order")),
        }
    }
}

/// Failure to compute or replay a local module standard basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LocalModuleError {
    /// Generic based-module construction failed.
    Module(ModuleError),
    /// Local-ring arithmetic failed.
    Local(LocalAlgebraError),
    /// Coordinate-polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// A coordinate list had the wrong rank.
    CoordinateRank {
        /// Required ambient rank.
        expected: usize,
        /// Received coordinate count.
        actual: usize,
    },
    /// A deterministic exact work bound was exceeded.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed implementation limit.
        limit: usize,
    },
    /// Retained standard-basis evidence did not replay.
    InvalidEvidence(String),
}

impl fmt::Display for LocalModuleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module(error) => error.fmt(formatter),
            Self::Local(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::CoordinateRank { expected, actual } => write!(
                formatter,
                "local module vector has {actual} coordinates; expected {expected}"
            ),
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "local module standard basis exceeded its {resource} limit of {limit}"
            ),
            Self::InvalidEvidence(detail) => {
                write!(formatter, "invalid local-module evidence: {detail}")
            }
        }
    }
}

impl std::error::Error for LocalModuleError {}

impl From<ModuleError> for LocalModuleError {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}

impl From<LocalAlgebraError> for LocalModuleError {
    fn from(value: LocalAlgebraError) -> Self {
        Self::Local(value)
    }
}

impl From<PolynomialError> for LocalModuleError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

/// A replayable Mora standard basis of a submodule of a finite free module.
///
/// Coordinate fractions are cleared only by a common local unit. The retained
/// polynomial vectors therefore generate the same submodule after
/// localization, while all hot reduction arithmetic stays in the native
/// coordinate polynomial ring.
#[derive(Clone, PartialEq, Eq)]
pub struct LocalModuleStandardBasis<R: Field> {
    ambient: FreeModule<LocalPolynomialRing<R>>,
    order: LocalModuleOrder,
    generators: Vec<Vec<Polynomial<R>>>,
    basis: Vec<Vec<Polynomial<R>>>,
}

impl<R: Field> LocalModuleStandardBasis<R> {
    /// Compute a local module standard basis from exact ambient elements.
    pub fn from_generators(
        ambient: FreeModule<LocalPolynomialRing<R>>,
        generators: &[ModuleElement<LocalPolynomialRing<R>>],
        order: LocalModuleOrder,
    ) -> Result<Self, LocalModuleError> {
        if generators
            .iter()
            .any(|generator| generator.module() != &ambient)
        {
            return Err(LocalModuleError::InvalidEvidence(
                "a local module generator belongs to another ambient module".to_owned(),
            ));
        }
        let mut cleared = generators
            .iter()
            .map(clear_local_vector)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|(_, value)| value)
            .filter(|value| !vector_is_zero(value))
            .collect::<Vec<_>>();
        sort_vectors(ambient.ring(), order, &mut cleared);
        cleared.dedup();
        let basis = construct_standard_basis(&ambient, order, &cleared)?;
        Ok(Self {
            ambient,
            order,
            generators: cleared,
            basis,
        })
    }

    /// Ambient finite based free module over the local ring.
    pub fn ambient(&self) -> &FreeModule<LocalPolynomialRing<R>> {
        &self.ambient
    }

    /// Module-monomial order extending the local order of the coefficient ring.
    pub const fn order(&self) -> LocalModuleOrder {
        self.order
    }

    /// Unit-cleared input generators as ambient local elements.
    pub fn generators(
        &self,
    ) -> Result<Vec<ModuleElement<LocalPolynomialRing<R>>>, LocalModuleError> {
        self.generators
            .iter()
            .map(|value| embed_vector(&self.ambient, value))
            .collect()
    }

    /// Deterministic monic Mora standard basis as ambient local elements.
    pub fn basis(&self) -> Result<Vec<ModuleElement<LocalPolynomialRing<R>>>, LocalModuleError> {
        self.basis
            .iter()
            .map(|value| embed_vector(&self.ambient, value))
            .collect()
    }

    /// Canonical generated submodule represented by this standard basis.
    pub fn submodule(&self) -> Result<Submodule<LocalPolynomialRing<R>>, LocalModuleError> {
        Submodule::from_generators(self.ambient.clone(), self.basis()?).map_err(Into::into)
    }

    /// Certified Mora weak normal form of one ambient local element.
    pub fn reduce(
        &self,
        value: &ModuleElement<LocalPolynomialRing<R>>,
    ) -> Result<LocalModuleReduction<R>, LocalModuleError> {
        if value.module() != &self.ambient {
            return Err(LocalModuleError::InvalidEvidence(
                "a reduced vector belongs to another ambient local module".to_owned(),
            ));
        }
        let (denominator, polynomial_value) = clear_local_vector(value)?;
        let tracked = mora_reduction(&self.ambient, self.order, &polynomial_value, &self.basis)?;
        let ring = self.ambient.ring().coordinate_ring();
        let multiplier = ring
            .mul(&tracked.multiplier, &denominator)
            .map_err(|error| {
                LocalModuleError::InvalidEvidence(format!(
                    "local module multiplier construction failed: {error}"
                ))
            })?;
        let result = LocalModuleReduction {
            standard_basis: self.clone(),
            value: value.clone(),
            multiplier,
            quotients: tracked.quotients,
            remainder: tracked.value,
        };
        result.verify_identity()?;
        Ok(result)
    }

    /// Decide submodule membership with retained weak-normal-form evidence.
    pub fn contains(
        &self,
        value: &ModuleElement<LocalPolynomialRing<R>>,
    ) -> Result<bool, LocalModuleError> {
        Ok(self.reduce(value)?.is_member())
    }

    /// Recompute every Mora pair and compare the retained standard basis.
    pub fn verify(&self) -> Result<(), LocalModuleError> {
        let generators = self.generators()?;
        let rebuilt = Self::from_generators(self.ambient.clone(), &generators, self.order)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(LocalModuleError::InvalidEvidence(
                "local module standard basis changed under recomputation".to_owned(),
            ))
        }
    }

    /// Serialize the ambient module, order, inputs, and standard basis.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly recompute a local module standard basis.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> fmt::Debug for LocalModuleStandardBasis<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalModuleStandardBasis")
            .field("ambient_rank", &self.ambient.rank())
            .field("basis", &self.basis.len())
            .field("order", &self.order)
            .finish()
    }
}

impl<R: Field> fmt::Display for LocalModuleStandardBasis<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "LocalModuleStandardBasis(rank={}, order={}, generators={})",
            self.ambient.rank(),
            self.order,
            self.basis.len()
        )
    }
}

impl<R: Field> SemanticType for LocalModuleStandardBasis<R> {
    const TYPE_ID: &'static str = "noether/homalg/LocalModuleStandardBasis@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.ambient.canonical_bytes()?;
        append_component(&mut output, &self.order.canonical_bytes()?);
        for value in self.generators.iter().chain(&self.basis) {
            append_component(&mut output, &usize_bytes(value.len())?);
            for coordinate in value {
                append_component(&mut output, &coordinate.canonical_bytes()?);
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_MODULE_STANDARD_BASIS_DESCRIPTION
    }
}

impl<R> Serialize for LocalModuleStandardBasis<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LocalModuleStandardBasis", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("ambient", &self.ambient)?;
        state.serialize_field("order", &self.order)?;
        state.serialize_field("generators", &self.generators)?;
        state.serialize_field("basis", &self.basis)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct LocalModuleStandardBasisWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    ambient: FreeModule<LocalPolynomialRing<R>>,
    order: LocalModuleOrder,
    generators: Vec<Vec<Polynomial<R>>>,
    basis: Vec<Vec<Polynomial<R>>>,
}

impl<'de, R> Deserialize<'de> for LocalModuleStandardBasis<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalModuleStandardBasisWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        if wire
            .generators
            .iter()
            .chain(&wire.basis)
            .any(|value| value.len() != wire.ambient.rank())
        {
            return Err(serde::de::Error::custom(
                "serialized local module vector has the wrong rank",
            ));
        }
        let generators = wire
            .generators
            .iter()
            .map(|value| embed_vector(&wire.ambient, value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::de::Error::custom)?;
        let rebuilt = Self::from_generators(wire.ambient, &generators, wire.order)
            .map_err(serde::de::Error::custom)?;
        if rebuilt.basis != wire.basis {
            return Err(serde::de::Error::custom(
                "serialized local module standard basis changed under recomputation",
            ));
        }
        Ok(rebuilt)
    }
}

/// Replayable weak-normal-form evidence for one local module reduction.
///
/// The retained identity is
/// `multiplier * value = sum(quotients[i] * basis[i]) + remainder`, where the
/// multiplier is a unit of the local ring.
#[derive(Clone, PartialEq, Eq)]
pub struct LocalModuleReduction<R: Field> {
    standard_basis: LocalModuleStandardBasis<R>,
    value: ModuleElement<LocalPolynomialRing<R>>,
    multiplier: Polynomial<R>,
    quotients: Vec<Polynomial<R>>,
    remainder: Vec<Polynomial<R>>,
}

impl<R: Field> LocalModuleReduction<R> {
    /// Standard basis used by the reduction.
    pub fn standard_basis(&self) -> &LocalModuleStandardBasis<R> {
        &self.standard_basis
    }

    /// Ambient local module element that was reduced.
    pub fn value(&self) -> &ModuleElement<LocalPolynomialRing<R>> {
        &self.value
    }

    /// Unit multiplier in the weak-normal-form identity.
    pub fn multiplier(&self) -> Result<LocalPolynomial<R>, LocalModuleError> {
        embed_polynomial(self.standard_basis.ambient.ring(), self.multiplier.clone())
    }

    /// Coefficients of the ordered standard-basis vectors.
    pub fn quotients(&self) -> Result<Vec<LocalPolynomial<R>>, LocalModuleError> {
        self.quotients
            .iter()
            .cloned()
            .map(|value| embed_polynomial(self.standard_basis.ambient.ring(), value))
            .collect()
    }

    /// Weak normal-form remainder.
    pub fn remainder(&self) -> Result<ModuleElement<LocalPolynomialRing<R>>, LocalModuleError> {
        embed_vector(&self.standard_basis.ambient, &self.remainder)
    }

    /// Whether the retained remainder is zero.
    pub fn is_member(&self) -> bool {
        vector_is_zero(&self.remainder)
    }

    /// Replay the exact identity and deterministic Mora reduction.
    pub fn verify(&self) -> Result<(), LocalModuleError> {
        self.verify_identity()?;
        let rebuilt = self.standard_basis.reduce(&self.value)?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(LocalModuleError::InvalidEvidence(
                "local module reduction changed under recomputation".to_owned(),
            ))
        }
    }

    fn verify_identity(&self) -> Result<(), LocalModuleError> {
        let ring = self.standard_basis.ambient.ring();
        if self.quotients.len() != self.standard_basis.basis.len()
            || self.multiplier.constant_coefficient()
                == ring.ambient_ring().coefficient_ring().zero()
            || self.remainder.len() != self.standard_basis.ambient.rank()
        {
            return Err(LocalModuleError::InvalidEvidence(
                "local module reduction has a nonunit multiplier or wrong arity".to_owned(),
            ));
        }
        let left = self.value.scalar_mul(&self.multiplier()?)?;
        let mut right = self.remainder()?;
        for (quotient, basis) in self
            .quotients()?
            .iter()
            .zip(self.standard_basis.basis()?.iter())
        {
            right = right.add(&basis.scalar_mul(quotient)?)?;
        }
        if left == right {
            Ok(())
        } else {
            Err(LocalModuleError::InvalidEvidence(
                "retained local module weak-normal-form identity does not hold".to_owned(),
            ))
        }
    }

    /// Serialize exact weak-normal-form evidence.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly replay weak-normal-form evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Field> fmt::Debug for LocalModuleReduction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalModuleReduction")
            .field("quotients", &self.quotients.len())
            .field("is_member", &self.is_member())
            .finish()
    }
}

impl<R: Field> fmt::Display for LocalModuleReduction<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "LocalModuleReduction(member={})",
            self.is_member()
        )
    }
}

impl<R: Field> SemanticType for LocalModuleReduction<R> {
    const TYPE_ID: &'static str = "noether/homalg/LocalModuleReduction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = self.standard_basis.canonical_bytes()?;
        append_component(&mut output, &self.value.canonical_bytes()?);
        append_component(&mut output, &self.multiplier.canonical_bytes()?);
        for quotient in &self.quotients {
            append_component(&mut output, &quotient.canonical_bytes()?);
        }
        for coordinate in &self.remainder {
            append_component(&mut output, &coordinate.canonical_bytes()?);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &LOCAL_MODULE_REDUCTION_DESCRIPTION
    }
}

impl<R> Serialize for LocalModuleReduction<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("LocalModuleReduction", 6)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("standard_basis", &self.standard_basis)?;
        state.serialize_field("value", &self.value)?;
        state.serialize_field("multiplier", &self.multiplier)?;
        state.serialize_field("quotients", &self.quotients)?;
        state.serialize_field("remainder", &self.remainder)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct LocalModuleReductionWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    standard_basis: LocalModuleStandardBasis<R>,
    value: ModuleElement<LocalPolynomialRing<R>>,
    multiplier: Polynomial<R>,
    quotients: Vec<Polynomial<R>>,
    remainder: Vec<Polynomial<R>>,
}

impl<'de, R> Deserialize<'de> for LocalModuleReduction<R>
where
    R: Field + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = LocalModuleReductionWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let retained = Self {
            standard_basis: wire.standard_basis,
            value: wire.value,
            multiplier: wire.multiplier,
            quotients: wire.quotients,
            remainder: wire.remainder,
        };
        retained
            .verify_identity()
            .map_err(serde::de::Error::custom)?;
        let rebuilt = retained
            .standard_basis
            .reduce(&retained.value)
            .map_err(serde::de::Error::custom)?;
        if rebuilt != retained {
            return Err(serde::de::Error::custom(
                "serialized local module reduction changed under recomputation",
            ));
        }
        Ok(retained)
    }
}

impl<R: Field> Submodule<LocalPolynomialRing<R>> {
    /// Compute a Mora standard basis in the declared local module order.
    pub fn local_standard_basis(
        &self,
        order: LocalModuleOrder,
    ) -> Result<LocalModuleStandardBasis<R>, LocalModuleError> {
        let matrix = self.generator_map().matrix();
        let mut generators = Vec::with_capacity(self.generator_count());
        for column in 0..self.generator_count() {
            let mut coordinates = Vec::with_capacity(self.ambient().rank());
            for row in 0..self.ambient().rank() {
                coordinates.push(matrix.get(row, column).map_err(|error| {
                    LocalModuleError::InvalidEvidence(format!(
                        "local submodule generator matrix was invalid: {error}"
                    ))
                })?);
            }
            generators.push(ModuleElement::from_coordinates(
                self.ambient().clone(),
                Vector::from_values(self.ambient().ring().clone(), coordinates).map_err(
                    |error| {
                        LocalModuleError::InvalidEvidence(format!(
                            "local submodule vector construction failed: {error}"
                        ))
                    },
                )?,
            )?);
        }
        LocalModuleStandardBasis::from_generators(self.ambient().clone(), &generators, order)
    }
}

#[derive(Clone)]
struct TrackedVector<R: Field> {
    value: Vec<Polynomial<R>>,
    multiplier: Polynomial<R>,
    quotients: Vec<Polynomial<R>>,
}

#[derive(Clone)]
struct LeadingModuleTerm<E> {
    exponent: Vec<Integer>,
    component: usize,
    coefficient: E,
}

fn construct_standard_basis<R: Field>(
    ambient: &FreeModule<LocalPolynomialRing<R>>,
    order: LocalModuleOrder,
    generators: &[Vec<Polynomial<R>>],
) -> Result<Vec<Vec<Polynomial<R>>>, LocalModuleError> {
    let mut basis = generators
        .iter()
        .map(|value| vector_monic(ambient.ring(), order, value))
        .collect::<Result<Vec<_>, _>>()?;
    basis.retain(|value| !vector_is_zero(value));
    sort_vectors(ambient.ring(), order, &mut basis);
    basis.dedup();
    let mut pending = Vec::new();
    for right in 0..basis.len() {
        for left in 0..right {
            if same_leading_component(ambient.ring(), order, &basis[left], &basis[right]) {
                pending.push((left, right));
            }
        }
    }
    let mut resolved = 0_usize;
    while let Some((left, right)) = pending.pop() {
        resolved += 1;
        if resolved > MAX_LOCAL_MODULE_PAIRS {
            return Err(LocalModuleError::ResourceLimit {
                resource: "Mora module pairs",
                limit: MAX_LOCAL_MODULE_PAIRS,
            });
        }
        let s = s_vector(ambient.ring(), order, &basis[left], &basis[right])?;
        let remainder = mora_reduction(ambient, order, &s, &basis)?.value;
        if vector_is_zero(&remainder) {
            continue;
        }
        let new_member = vector_monic(ambient.ring(), order, &remainder)?;
        let new_index = basis.len();
        for (old, member) in basis.iter().enumerate().take(new_index) {
            if same_leading_component(ambient.ring(), order, member, &new_member) {
                pending.push((old, new_index));
            }
        }
        basis.push(new_member);
    }
    sort_vectors(ambient.ring(), order, &mut basis);
    basis.dedup();
    Ok(basis)
}

fn mora_reduction<R: Field>(
    ambient: &FreeModule<LocalPolynomialRing<R>>,
    order: LocalModuleOrder,
    value: &[Polynomial<R>],
    basis: &[Vec<Polynomial<R>>],
) -> Result<TrackedVector<R>, LocalModuleError> {
    if value.len() != ambient.rank()
        || basis
            .iter()
            .any(|generator| generator.len() != ambient.rank())
    {
        return Err(LocalModuleError::CoordinateRank {
            expected: ambient.rank(),
            actual: value.len(),
        });
    }
    let polynomial_ring = ambient.ring().coordinate_ring();
    let mut current = TrackedVector {
        value: value.to_vec(),
        multiplier: polynomial_ring.one(),
        quotients: vec![polynomial_ring.zero(); basis.len()],
    };
    let mut reducers = basis
        .iter()
        .enumerate()
        .map(|(index, generator)| {
            let mut quotients = vec![polynomial_ring.zero(); basis.len()];
            quotients[index] = polynomial_ring
                .neg(&polynomial_ring.one())
                .map_err(|error| {
                    LocalModuleError::InvalidEvidence(format!(
                        "coefficient negation failed: {error}"
                    ))
                })?;
            Ok(TrackedVector {
                value: generator.clone(),
                multiplier: polynomial_ring.zero(),
                quotients,
            })
        })
        .collect::<Result<Vec<_>, LocalModuleError>>()?;
    let mut steps = 0_usize;
    while let Some(leading) = leading_term(ambient.ring(), order, &current.value) {
        steps += 1;
        if steps > MAX_LOCAL_MODULE_REDUCTION_STEPS {
            return Err(LocalModuleError::ResourceLimit {
                resource: "Mora module reduction steps",
                limit: MAX_LOCAL_MODULE_REDUCTION_STEPS,
            });
        }
        let remainder_ecart = vector_ecart(ambient.ring(), order, &current.value)?;
        let mut candidates = Vec::new();
        for (index, reducer) in reducers.iter().enumerate() {
            let Some(reducer_leading) = leading_term(ambient.ring(), order, &reducer.value) else {
                continue;
            };
            if module_monomial_divides(&reducer_leading, &leading) {
                candidates.push((
                    vector_ecart(ambient.ring(), order, &reducer.value)?,
                    index,
                    reducer_leading,
                ));
            }
        }
        let candidate = candidates
            .into_iter()
            .min_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let Some((reducer_ecart, reducer_index, reducer_leading)) = candidate else {
            break;
        };
        if reducer_ecart > remainder_ecart {
            reducers.push(current.clone());
        }
        let inverse = ambient
            .ring()
            .ambient_ring()
            .coefficient_ring()
            .inverse(&reducer_leading.coefficient)
            .map_err(|_| {
                LocalModuleError::InvalidEvidence(
                    "a local module reducer has a noninvertible leader".to_owned(),
                )
            })?;
        let coefficient = ambient
            .ring()
            .ambient_ring()
            .coefficient_ring()
            .mul(&leading.coefficient, &inverse)
            .map_err(|error| {
                LocalModuleError::InvalidEvidence(format!(
                    "local module coefficient multiplication failed: {error}"
                ))
            })?;
        let exponent = leading
            .exponent
            .iter()
            .zip(&reducer_leading.exponent)
            .map(|(available, required)| available - required)
            .collect::<Vec<_>>();
        let monomial = polynomial_ring.from_terms([(exponent, coefficient)])?;
        let correction =
            vector_scalar_mul(polynomial_ring, &reducers[reducer_index].value, &monomial)?;
        current.value = vector_sub(polynomial_ring, &current.value, &correction)?;
        let multiplier_correction = polynomial_ring
            .mul(&monomial, &reducers[reducer_index].multiplier)
            .map_err(|error| {
                LocalModuleError::InvalidEvidence(format!(
                    "local module multiplier update failed: {error}"
                ))
            })?;
        current.multiplier = polynomial_ring
            .sub(&current.multiplier, &multiplier_correction)
            .map_err(|error| {
                LocalModuleError::InvalidEvidence(format!(
                    "local module multiplier subtraction failed: {error}"
                ))
            })?;
        for (current_quotient, reducer_quotient) in current
            .quotients
            .iter_mut()
            .zip(&reducers[reducer_index].quotients)
        {
            let correction = polynomial_ring
                .mul(&monomial, reducer_quotient)
                .map_err(|error| {
                    LocalModuleError::InvalidEvidence(format!(
                        "local module quotient update failed: {error}"
                    ))
                })?;
            *current_quotient =
                polynomial_ring
                    .sub(current_quotient, &correction)
                    .map_err(|error| {
                        LocalModuleError::InvalidEvidence(format!(
                            "local module quotient subtraction failed: {error}"
                        ))
                    })?;
        }
    }
    Ok(current)
}

fn s_vector<R: Field>(
    ring: &LocalPolynomialRing<R>,
    order: LocalModuleOrder,
    left: &[Polynomial<R>],
    right: &[Polynomial<R>],
) -> Result<Vec<Polynomial<R>>, LocalModuleError> {
    let left_leading = leading_term(ring, order, left).ok_or_else(|| {
        LocalModuleError::InvalidEvidence("zero local module basis vector".to_owned())
    })?;
    let right_leading = leading_term(ring, order, right).ok_or_else(|| {
        LocalModuleError::InvalidEvidence("zero local module basis vector".to_owned())
    })?;
    if left_leading.component != right_leading.component {
        return Err(LocalModuleError::InvalidEvidence(
            "an S-vector pair has different leading components".to_owned(),
        ));
    }
    let lcm = left_leading
        .exponent
        .iter()
        .zip(&right_leading.exponent)
        .map(|(left, right)| left.max(right).clone())
        .collect::<Vec<_>>();
    let field = ring.ambient_ring().coefficient_ring();
    let left_inverse = field.inverse(&left_leading.coefficient).map_err(|_| {
        LocalModuleError::InvalidEvidence("noninvertible local module leader".to_owned())
    })?;
    let right_inverse = field.inverse(&right_leading.coefficient).map_err(|_| {
        LocalModuleError::InvalidEvidence("noninvertible local module leader".to_owned())
    })?;
    let polynomial_ring = ring.coordinate_ring();
    let left_multiplier = polynomial_ring.from_terms([(
        lcm.iter()
            .zip(&left_leading.exponent)
            .map(|(lcm, exponent)| lcm - exponent)
            .collect(),
        left_inverse,
    )])?;
    let right_multiplier = polynomial_ring.from_terms([(
        lcm.iter()
            .zip(&right_leading.exponent)
            .map(|(lcm, exponent)| lcm - exponent)
            .collect(),
        right_inverse,
    )])?;
    vector_sub(
        polynomial_ring,
        &vector_scalar_mul(polynomial_ring, left, &left_multiplier)?,
        &vector_scalar_mul(polynomial_ring, right, &right_multiplier)?,
    )
}

fn same_leading_component<R: Field>(
    ring: &LocalPolynomialRing<R>,
    order: LocalModuleOrder,
    left: &[Polynomial<R>],
    right: &[Polynomial<R>],
) -> bool {
    leading_term(ring, order, left)
        .zip(leading_term(ring, order, right))
        .is_some_and(|(left, right)| left.component == right.component)
}

fn leading_term<R: Field>(
    ring: &LocalPolynomialRing<R>,
    order: LocalModuleOrder,
    value: &[Polynomial<R>],
) -> Option<LeadingModuleTerm<R::Element>> {
    let mut leading = None;
    for (component, polynomial) in value.iter().enumerate() {
        for (exponent, coefficient) in polynomial.terms() {
            let candidate = LeadingModuleTerm {
                exponent: exponent.to_vec(),
                component,
                coefficient: coefficient.clone(),
            };
            if leading.as_ref().is_none_or(|current| {
                compare_module_terms(ring.order(), order, &candidate, current) == Ordering::Greater
            }) {
                leading = Some(candidate);
            }
        }
    }
    leading
}

fn compare_module_terms<E>(
    base: LocalMonomialOrder,
    order: LocalModuleOrder,
    left: &LeadingModuleTerm<E>,
    right: &LeadingModuleTerm<E>,
) -> Ordering {
    let position = || right.component.cmp(&left.component);
    match order {
        LocalModuleOrder::TermOverPosition => base
            .compare(&left.exponent, &right.exponent)
            .then_with(position),
        LocalModuleOrder::PositionOverTerm => {
            position().then_with(|| base.compare(&left.exponent, &right.exponent))
        }
    }
}

fn module_monomial_divides<E>(
    divisor: &LeadingModuleTerm<E>,
    dividend: &LeadingModuleTerm<E>,
) -> bool {
    divisor.component == dividend.component
        && divisor
            .exponent
            .iter()
            .zip(&dividend.exponent)
            .all(|(required, available)| required <= available)
}

fn vector_ecart<R: Field>(
    ring: &LocalPolynomialRing<R>,
    order: LocalModuleOrder,
    value: &[Polynomial<R>],
) -> Result<Integer, LocalModuleError> {
    let Some(leading) = leading_term(ring, order, value) else {
        return Ok(Integer::zero());
    };
    let maximum = value
        .iter()
        .map(|coordinate| coordinate.degree(None))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .max()
        .unwrap_or_else(Integer::zero);
    Ok(maximum - total_degree(&leading.exponent))
}

fn vector_monic<R: Field>(
    ring: &LocalPolynomialRing<R>,
    order: LocalModuleOrder,
    value: &[Polynomial<R>],
) -> Result<Vec<Polynomial<R>>, LocalModuleError> {
    let Some(leading) = leading_term(ring, order, value) else {
        return Ok(value.to_vec());
    };
    let inverse = ring
        .ambient_ring()
        .coefficient_ring()
        .inverse(&leading.coefficient)
        .map_err(|_| {
            LocalModuleError::InvalidEvidence("noninvertible local module leader".to_owned())
        })?;
    value
        .iter()
        .map(|coordinate| coordinate.scalar_mul(&inverse).map_err(Into::into))
        .collect()
}

fn vector_scalar_mul<R: Field>(
    ring: &crate::commalg::PolynomialRing<R>,
    value: &[Polynomial<R>],
    scalar: &Polynomial<R>,
) -> Result<Vec<Polynomial<R>>, LocalModuleError> {
    value
        .iter()
        .map(|coordinate| {
            ring.mul(coordinate, scalar).map_err(|error| {
                LocalModuleError::InvalidEvidence(format!(
                    "local module polynomial multiplication failed: {error}"
                ))
            })
        })
        .collect()
}

fn vector_sub<R: Field>(
    ring: &crate::commalg::PolynomialRing<R>,
    left: &[Polynomial<R>],
    right: &[Polynomial<R>],
) -> Result<Vec<Polynomial<R>>, LocalModuleError> {
    if left.len() != right.len() {
        return Err(LocalModuleError::CoordinateRank {
            expected: left.len(),
            actual: right.len(),
        });
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            ring.sub(left, right).map_err(|error| {
                LocalModuleError::InvalidEvidence(format!(
                    "local module polynomial subtraction failed: {error}"
                ))
            })
        })
        .collect()
}

fn vector_is_zero<R: Field>(value: &[Polynomial<R>]) -> bool {
    value.iter().all(Polynomial::is_zero)
}

fn sort_vectors<R: Field>(
    ring: &LocalPolynomialRing<R>,
    order: LocalModuleOrder,
    values: &mut [Vec<Polynomial<R>>],
) {
    values.sort_by(|left, right| {
        match (
            leading_term(ring, order, left),
            leading_term(ring, order, right),
        ) {
            (Some(left), Some(right)) => compare_module_terms(ring.order(), order, &right, &left),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
        .then_with(|| vector_text(left).cmp(&vector_text(right)))
    });
}

fn vector_text<R: Field>(value: &[Polynomial<R>]) -> String {
    value
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("|")
}

fn total_degree(exponent: &[Integer]) -> Integer {
    exponent
        .iter()
        .fold(Integer::zero(), |degree, power| degree + power)
}

fn clear_local_vector<R: Field>(
    value: &ModuleElement<LocalPolynomialRing<R>>,
) -> Result<(Polynomial<R>, Vec<Polynomial<R>>), LocalModuleError> {
    let local_ring = value.module().ring();
    let polynomial_ring = local_ring.coordinate_ring();
    let coordinates = value.coordinates().values().collect::<Vec<_>>();
    let mut denominator = polynomial_ring.one();
    for coordinate in &coordinates {
        denominator = polynomial_ring
            .mul(&denominator, coordinate.denominator())
            .map_err(|error| {
                LocalModuleError::InvalidEvidence(format!(
                    "local module denominator product failed: {error}"
                ))
            })?;
    }
    let mut cleared = Vec::with_capacity(coordinates.len());
    for (index, coordinate) in coordinates.iter().enumerate() {
        let mut complement = polynomial_ring.one();
        for (other, factor) in coordinates.iter().enumerate() {
            if other != index {
                complement = polynomial_ring
                    .mul(&complement, factor.denominator())
                    .map_err(|error| {
                        LocalModuleError::InvalidEvidence(format!(
                            "local module denominator clearing failed: {error}"
                        ))
                    })?;
            }
        }
        cleared.push(
            polynomial_ring
                .mul(coordinate.numerator(), &complement)
                .map_err(|error| {
                    LocalModuleError::InvalidEvidence(format!(
                        "local module numerator clearing failed: {error}"
                    ))
                })?,
        );
    }
    Ok((denominator, cleared))
}

fn embed_polynomial<R: Field>(
    ring: &LocalPolynomialRing<R>,
    value: Polynomial<R>,
) -> Result<LocalPolynomial<R>, LocalModuleError> {
    ring.fraction(value, ring.coordinate_ring().one())
        .map_err(Into::into)
}

fn embed_vector<R: Field>(
    ambient: &FreeModule<LocalPolynomialRing<R>>,
    value: &[Polynomial<R>],
) -> Result<ModuleElement<LocalPolynomialRing<R>>, LocalModuleError> {
    if value.len() != ambient.rank() {
        return Err(LocalModuleError::CoordinateRank {
            expected: ambient.rank(),
            actual: value.len(),
        });
    }
    let coordinates = value
        .iter()
        .cloned()
        .map(|coordinate| embed_polynomial(ambient.ring(), coordinate))
        .collect::<Result<Vec<_>, _>>()?;
    let vector = Vector::from_values(ambient.ring().clone(), coordinates).map_err(|error| {
        LocalModuleError::InvalidEvidence(format!(
            "local module vector construction failed: {error}"
        ))
    })?;
    ModuleElement::from_coordinates(ambient.clone(), vector).map_err(Into::into)
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) {
    output.extend_from_slice(&(component.len() as u64).to_le_bytes());
    output.extend_from_slice(component);
}

fn usize_bytes(value: usize) -> Result<Vec<u8>, NoetherError> {
    let value = u64::try_from(value)
        .map_err(|_| NoetherError::Serialization("length does not fit u64".to_owned()))?;
    Ok(value.to_le_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use crate::commalg::{LocalMonomialOrder, MonomialOrder, PolynomialRing};
    use crate::linalg::Vector;
    use crate::{Ring, Q};

    use super::*;

    fn element<R: Field>(
        module: &FreeModule<LocalPolynomialRing<R>>,
        coordinates: Vec<LocalPolynomial<R>>,
    ) -> ModuleElement<LocalPolynomialRing<R>> {
        ModuleElement::from_coordinates(
            module.clone(),
            Vector::from_values(module.ring().clone(), coordinates).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn local_module_membership_retains_a_weak_normal_form_identity() {
        let ambient_ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        let x = ambient_ring.generator(0).unwrap();
        let local_ring = LocalPolynomialRing::at_origin(
            ambient_ring.clone(),
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )
        .unwrap();
        let module = FreeModule::new(local_ring.clone(), 2);
        let zero = local_ring.zero();
        let generator = element(
            &module,
            vec![
                local_ring
                    .from_polynomial(&ambient_ring.add(&x, &x.pow(2).unwrap()).unwrap())
                    .unwrap(),
                zero.clone(),
            ],
        );
        let value = element(&module, vec![local_ring.from_polynomial(&x).unwrap(), zero]);
        let basis = LocalModuleStandardBasis::from_generators(
            module,
            &[generator],
            LocalModuleOrder::TermOverPosition,
        )
        .unwrap();
        let reduction = basis.reduce(&value).unwrap();

        assert!(reduction.is_member());
        assert!(reduction.multiplier().unwrap().is_unit());
        reduction.verify().unwrap();
        basis.verify().unwrap();
        assert_eq!(
            LocalModuleStandardBasis::<crate::RationalField>::from_json(&basis.to_json().unwrap())
                .unwrap(),
            basis
        );
        assert_eq!(
            LocalModuleReduction::<crate::RationalField>::from_json(&reduction.to_json().unwrap())
                .unwrap(),
            reduction
        );
    }

    #[test]
    fn position_order_changes_leaders_but_not_generated_membership() {
        let ambient_ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let local_ring = LocalPolynomialRing::at_origin(
            ambient_ring,
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )
        .unwrap();
        let module = FreeModule::new(local_ring.clone(), 2);
        let [x, y] = local_ring.generators().unwrap().try_into().unwrap();
        let first = element(&module, vec![x.clone(), y.clone()]);
        let second = element(&module, vec![local_ring.zero(), x]);
        let submodule = Submodule::from_generators(module, vec![first.clone(), second]).unwrap();

        let basis = submodule
            .local_standard_basis(LocalModuleOrder::PositionOverTerm)
            .unwrap();

        assert!(basis.contains(&first).unwrap());
        basis.verify().unwrap();
    }
}
