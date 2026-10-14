// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Derived modules constructed from verified finite free resolutions.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{PolynomialRing, QuotientRing};
use crate::{CommutativeRing, Field, NoetherError, Ring, SemanticType, TypeDescription};

use super::{
    FreeModule, FreeResolution, ModuleError, ModuleGroebnerError, ModuleHomomorphism,
    ModuleMonomialOrder, PresentedModule, PresentedModuleHomology, PresentedModuleHomomorphism,
    ResolutionCompleteness, ResolutionError,
};

static DERIVED_MODULE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/DerivedModule@1",
    rust_path: "noether::homalg::DerivedModule",
    python_path: Some("noether.homalg.DerivedModule"),
};

static DERIVED_MODULE_HOMOMORPHISM_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/DerivedModuleHomomorphism@1",
    rust_path: "noether::homalg::DerivedModuleHomomorphism",
    python_path: Some("noether.homalg.DerivedModuleHomomorphism"),
};

/// Derived functor represented by a [`DerivedModule`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivedFunctor {
    /// Left-derived tensor product.
    Tor,
    /// Right-derived `Hom` in the first variable.
    Ext,
}

impl DerivedFunctor {
    /// Stable lowercase name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Tor => "tor",
            Self::Ext => "ext",
        }
    }
}

impl fmt::Display for DerivedFunctor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Failure to construct or verify a derived module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DerivedError {
    /// The supplied resolution failed or could not be constructed.
    Resolution(ResolutionError),
    /// A tensor, dual, or presentation operation failed.
    Module(ModuleError),
    /// Module-valued homology or its exact evidence failed.
    Groebner(ModuleGroebnerError),
    /// A truncated resolution does not contain the look-ahead differential
    /// required to justify the requested homological degree.
    DegreeNotJustified {
        /// Requested derived degree.
        requested: usize,
        /// Greatest degree justified by the supplied truncated resolution.
        justified_through: Option<usize>,
    },
    /// Stored construction evidence does not match the stated functor.
    InvalidEvidence(&'static str),
}

impl fmt::Display for DerivedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resolution(error) => error.fmt(formatter),
            Self::Module(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::DegreeNotJustified {
                requested,
                justified_through,
            } => match justified_through {
                Some(degree) => write!(
                    formatter,
                    "derived degree {requested} is not justified by a resolution through degree {degree}"
                ),
                None => write!(
                    formatter,
                    "derived degree {requested} is not justified by the supplied resolution"
                ),
            },
            Self::InvalidEvidence(reason) => {
                write!(formatter, "invalid derived-module evidence: {reason}")
            }
        }
    }
}

impl std::error::Error for DerivedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Resolution(error) => Some(error),
            Self::Module(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::DegreeNotJustified { .. } | Self::InvalidEvidence(_) => None,
        }
    }
}

impl From<ResolutionError> for DerivedError {
    fn from(value: ResolutionError) -> Self {
        Self::Resolution(value)
    }
}

impl From<ModuleError> for DerivedError {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}

impl From<ModuleGroebnerError> for DerivedError {
    fn from(value: ModuleGroebnerError) -> Self {
        Self::Groebner(value)
    }
}

/// One exact `Tor` or `Ext` module with its complete construction evidence.
///
/// The value retains the verified resolution, the other module argument, the
/// module order used for homology, and the full cycle/boundary quotient
/// diagram. A truncated resolution through degree `d` justifies derived
/// modules only through degree `d - 1`, because degree `i` needs the
/// differential from degree `i + 1`.
#[derive(Clone, PartialEq, Eq)]
pub struct DerivedModule<R: Ring> {
    functor: DerivedFunctor,
    degree: usize,
    resolution: FreeResolution<R>,
    coefficient_module: PresentedModule<R>,
    order: ModuleMonomialOrder,
    homology: PresentedModuleHomology<R>,
}

/// A functorial `Tor` or `Ext` map induced by a homomorphism of the second
/// module argument.
///
/// The value retains both derived modules, the coefficient-module
/// homomorphism, and the induced presented-module map. Verification rebuilds
/// the three-term chain map and independently reinduces its homology map.
#[derive(Clone, PartialEq, Eq)]
pub struct DerivedModuleHomomorphism<R: Ring> {
    source: DerivedModule<R>,
    target: DerivedModule<R>,
    coefficient_homomorphism: PresentedModuleHomomorphism<R>,
    homomorphism: PresentedModuleHomomorphism<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Deserialize<'de>, R::Element: Deserialize<'de>, FreeResolution<R>: Deserialize<'de>"
    )
)]
struct DerivedModuleWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    functor: DerivedFunctor,
    degree: usize,
    resolution: FreeResolution<R>,
    coefficient_module: PresentedModule<R>,
    order: ModuleMonomialOrder,
    homology: PresentedModuleHomology<R>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Deserialize<'de>, R::Element: Deserialize<'de>, FreeResolution<R>: Deserialize<'de>, DerivedModule<R>: Deserialize<'de>"
    )
)]
struct DerivedModuleHomomorphismWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    source: DerivedModule<R>,
    target: DerivedModule<R>,
    coefficient_homomorphism: PresentedModuleHomomorphism<R>,
    homomorphism: PresentedModuleHomomorphism<R>,
}

type DerivedComplexMaps<R> = (
    PresentedModuleHomomorphism<R>,
    PresentedModuleHomomorphism<R>,
);
type DerivedChainMaps<R> = (
    PresentedModuleHomomorphism<R>,
    PresentedModuleHomomorphism<R>,
    PresentedModuleHomomorphism<R>,
);

impl<R: Ring> DerivedModule<R> {
    /// Derived functor represented by this value.
    pub const fn functor(&self) -> DerivedFunctor {
        self.functor
    }

    /// Homological or cohomological degree.
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Verified resolution used in the construction.
    pub fn resolution(&self) -> &FreeResolution<R> {
        &self.resolution
    }

    /// Second module argument: the tensor factor for `Tor` or the target for
    /// `Ext`.
    pub fn coefficient_module(&self) -> &PresentedModule<R> {
        &self.coefficient_module
    }

    /// Exact module-monomial order used to construct and verify homology.
    pub fn order(&self) -> &ModuleMonomialOrder {
        &self.order
    }

    /// Complete cycle/boundary quotient evidence.
    pub fn homology_evidence(&self) -> &PresentedModuleHomology<R> {
        &self.homology
    }

    /// The resulting presented module.
    pub fn module(&self) -> &PresentedModule<R> {
        self.homology.module()
    }

    /// Greatest derived degree justified by a truncated resolution. `None`
    /// means the resolution is complete and every degree is justified.
    pub fn justified_through_degree(&self) -> Option<usize> {
        match self.resolution.completeness() {
            ResolutionCompleteness::Complete => None,
            ResolutionCompleteness::ThroughHomologicalDegree { degree } => degree.checked_sub(1),
        }
    }

    /// Serialize the resolution, finite range, and complete homology diagram.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Ring> DerivedModuleHomomorphism<R> {
    /// Source derived module.
    pub fn source(&self) -> &DerivedModule<R> {
        &self.source
    }

    /// Target derived module.
    pub fn target(&self) -> &DerivedModule<R> {
        &self.target
    }

    /// Homomorphism of coefficient modules inducing this derived map.
    pub fn coefficient_homomorphism(&self) -> &PresentedModuleHomomorphism<R> {
        &self.coefficient_homomorphism
    }

    /// Exact induced homomorphism between the resulting presented modules.
    pub fn homomorphism(&self) -> &PresentedModuleHomomorphism<R> {
        &self.homomorphism
    }

    /// Serialize both derived evidence objects and the inducing and induced
    /// homomorphisms.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: CommutativeRing> DerivedModule<R> {
    fn from_construction(
        functor: DerivedFunctor,
        degree: usize,
        resolution: FreeResolution<R>,
        coefficient_module: PresentedModule<R>,
        order: ModuleMonomialOrder,
        maps: DerivedComplexMaps<R>,
        homology: PresentedModuleHomology<R>,
    ) -> Result<Self, DerivedError> {
        require_justified_degree(&resolution, degree)?;
        let (incoming, outgoing) = maps;
        if homology.incoming() != &incoming || homology.outgoing() != &outgoing {
            return Err(DerivedError::InvalidEvidence(
                "constructed homology maps do not match the derived complex",
            ));
        }
        let result = Self {
            functor,
            degree,
            resolution,
            coefficient_module,
            order,
            homology,
        };
        Ok(result)
    }
}

fn verify_derived_structure<R: CommutativeRing>(
    value: &DerivedModule<R>,
) -> Result<(), DerivedError> {
    require_justified_degree(&value.resolution, value.degree)?;
    if value.coefficient_module.generator_module().ring()
        != value.resolution.module().generator_module().ring()
    {
        return Err(DerivedError::InvalidEvidence(
            "the resolution and coefficient module have unequal rings",
        ));
    }
    let (incoming, outgoing) = derived_maps(
        value.functor,
        &value.resolution,
        &value.coefficient_module,
        value.degree,
    )?;
    if value.homology.incoming() != &incoming || value.homology.outgoing() != &outgoing {
        return Err(DerivedError::InvalidEvidence(
            "the retained complex is not the stated derived-functor complex",
        ));
    }
    Ok(())
}

impl<C: Field> DerivedModule<PolynomialRing<C>> {
    /// Independently replay the polynomial-ring resolution, functorial
    /// differential, and module-valued homology evidence.
    pub fn verify(&self) -> Result<(), DerivedError> {
        self.resolution.verify()?;
        verify_derived_structure(self)?;
        self.homology.verify(self.order.clone())?;
        Ok(())
    }

    /// Deserialize and independently revalidate polynomial-ring evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Apply `Tor` or `Ext` covariantly to a homomorphism of the second module
    /// argument and retain the complete induced-homology evidence.
    pub fn map_coefficients(
        &self,
        homomorphism: &PresentedModuleHomomorphism<PolynomialRing<C>>,
    ) -> Result<DerivedModuleHomomorphism<PolynomialRing<C>>, DerivedError> {
        if homomorphism.source() != self.coefficient_module() {
            return Err(DerivedError::InvalidEvidence(
                "a coefficient homomorphism has the wrong source module",
            ));
        }
        let target = self.resolution.derived_module(
            self.functor,
            homomorphism.target(),
            self.degree,
            self.order.clone(),
        )?;
        let (left, middle, right) = derived_coefficient_chain_maps(
            self.functor,
            &self.resolution,
            homomorphism,
            self.degree,
        )?;
        let induced = self.homology.induced_map(
            target.homology_evidence(),
            &left,
            &middle,
            &right,
            self.order.clone(),
        )?;
        let result = DerivedModuleHomomorphism {
            source: self.clone(),
            target,
            coefficient_homomorphism: homomorphism.clone(),
            homomorphism: induced,
        };
        result.verify()?;
        Ok(result)
    }
}

impl<C: Field> DerivedModuleHomomorphism<PolynomialRing<C>> {
    /// Independently replay the coefficient chain map and its induced
    /// polynomial-module homology map.
    pub fn verify(&self) -> Result<(), DerivedError> {
        self.source.verify()?;
        self.target.verify()?;
        self.coefficient_homomorphism.verify()?;
        verify_derived_homomorphism_structure(self)?;
        let (left, middle, right) = derived_coefficient_chain_maps(
            self.source.functor,
            &self.source.resolution,
            &self.coefficient_homomorphism,
            self.source.degree,
        )?;
        let expected = self.source.homology.induced_map(
            &self.target.homology,
            &left,
            &middle,
            &right,
            self.source.order.clone(),
        )?;
        if !self
            .homomorphism
            .is_equal(&expected, self.source.order.clone())?
        {
            return Err(DerivedError::InvalidEvidence(
                "the retained map is not induced by the coefficient homomorphism",
            ));
        }
        Ok(())
    }

    /// Deserialize and independently revalidate a polynomial derived map.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

type PolynomialQuotient<C> = QuotientRing<PolynomialRing<C>>;

impl<C: Field> DerivedModule<PolynomialQuotient<C>> {
    /// Independently replay the quotient-ring resolution, functorial
    /// differential, and module-valued homology evidence.
    pub fn verify(&self) -> Result<(), DerivedError> {
        self.resolution.verify()?;
        verify_derived_structure(self)?;
        self.homology.verify(self.order.clone())?;
        Ok(())
    }

    /// Deserialize and independently revalidate quotient-ring evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Apply quotient-ring `Tor` or `Ext` covariantly to a homomorphism of the
    /// second module argument.
    pub fn map_coefficients(
        &self,
        homomorphism: &PresentedModuleHomomorphism<PolynomialQuotient<C>>,
    ) -> Result<DerivedModuleHomomorphism<PolynomialQuotient<C>>, DerivedError> {
        if homomorphism.source() != self.coefficient_module() {
            return Err(DerivedError::InvalidEvidence(
                "a coefficient homomorphism has the wrong source module",
            ));
        }
        let target = self.resolution.derived_module(
            self.functor,
            homomorphism.target(),
            self.degree,
            self.order.clone(),
        )?;
        let (left, middle, right) = derived_coefficient_chain_maps(
            self.functor,
            &self.resolution,
            homomorphism,
            self.degree,
        )?;
        let induced = self.homology.induced_map(
            target.homology_evidence(),
            &left,
            &middle,
            &right,
            self.order.clone(),
        )?;
        let result = DerivedModuleHomomorphism {
            source: self.clone(),
            target,
            coefficient_homomorphism: homomorphism.clone(),
            homomorphism: induced,
        };
        result.verify()?;
        Ok(result)
    }
}

impl<C: Field> DerivedModuleHomomorphism<PolynomialQuotient<C>> {
    /// Independently replay the coefficient chain map and induced quotient
    /// homology map.
    pub fn verify(&self) -> Result<(), DerivedError> {
        self.source.verify()?;
        self.target.verify()?;
        self.coefficient_homomorphism.verify()?;
        verify_derived_homomorphism_structure(self)?;
        let (left, middle, right) = derived_coefficient_chain_maps(
            self.source.functor,
            &self.source.resolution,
            &self.coefficient_homomorphism,
            self.source.degree,
        )?;
        let expected = self.source.homology.induced_map(
            &self.target.homology,
            &left,
            &middle,
            &right,
            self.source.order.clone(),
        )?;
        if !self
            .homomorphism
            .is_equal(&expected, self.source.order.clone())?
        {
            return Err(DerivedError::InvalidEvidence(
                "the retained quotient map is not induced by the coefficient homomorphism",
            ));
        }
        Ok(())
    }

    /// Deserialize and independently revalidate a quotient-ring derived map.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> FreeResolution<PolynomialRing<C>> {
    /// Compute `Tor_degree(module(), coefficient_module)` from this verified
    /// resolution.
    pub fn tor(
        &self,
        coefficient_module: &PresentedModule<PolynomialRing<C>>,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialRing<C>>, DerivedError> {
        self.derived_module(DerivedFunctor::Tor, coefficient_module, degree, order)
    }

    /// Compute `Ext^degree(module(), coefficient_module)` from this verified
    /// resolution of the first argument.
    pub fn ext(
        &self,
        coefficient_module: &PresentedModule<PolynomialRing<C>>,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialRing<C>>, DerivedError> {
        self.derived_module(DerivedFunctor::Ext, coefficient_module, degree, order)
    }

    fn derived_module(
        &self,
        functor: DerivedFunctor,
        coefficient_module: &PresentedModule<PolynomialRing<C>>,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialRing<C>>, DerivedError> {
        require_justified_degree(self, degree)?;
        let (incoming, outgoing) = derived_maps(functor, self, coefficient_module, degree)?;
        let homology = if functor == DerivedFunctor::Tor && degree == 0 {
            PresentedModuleHomology::from_cokernel_complex(&incoming, &outgoing)?
        } else {
            outgoing.homology(&incoming, order.clone())?
        };
        DerivedModule::from_construction(
            functor,
            degree,
            self.clone(),
            coefficient_module.clone(),
            order,
            (incoming, outgoing),
            homology,
        )
    }
}

impl<C: Field> PresentedModule<PolynomialRing<C>> {
    /// Compute `Tor_degree(self, rhs)` from a verified resolution through the
    /// required look-ahead degree.
    pub fn tor(
        &self,
        rhs: &Self,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialRing<C>>, DerivedError> {
        if degree == 0 {
            return FreeResolution::presentation_through_one(self)?.tor(rhs, degree, order);
        }
        let through = degree
            .checked_add(1)
            .ok_or(DerivedError::DegreeNotJustified {
                requested: degree,
                justified_through: None,
            })?;
        self.free_resolution_through(order.clone(), through)?
            .tor(rhs, degree, order)
    }

    /// Compute `Ext^degree(self, target)` from a verified resolution through
    /// the required look-ahead degree.
    pub fn ext(
        &self,
        target: &Self,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialRing<C>>, DerivedError> {
        let through = degree
            .checked_add(1)
            .ok_or(DerivedError::DegreeNotJustified {
                requested: degree,
                justified_through: None,
            })?;
        self.free_resolution_through(order.clone(), through)?
            .ext(target, degree, order)
    }
}

impl<C: Field> FreeResolution<PolynomialQuotient<C>> {
    /// Compute `Tor_degree(module(), coefficient_module)` from an honestly
    /// bounded quotient-ring resolution.
    pub fn tor(
        &self,
        coefficient_module: &PresentedModule<PolynomialQuotient<C>>,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialQuotient<C>>, DerivedError> {
        self.derived_module(DerivedFunctor::Tor, coefficient_module, degree, order)
    }

    /// Compute `Ext^degree(module(), coefficient_module)` from an honestly
    /// bounded quotient-ring resolution of the first argument.
    pub fn ext(
        &self,
        coefficient_module: &PresentedModule<PolynomialQuotient<C>>,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialQuotient<C>>, DerivedError> {
        self.derived_module(DerivedFunctor::Ext, coefficient_module, degree, order)
    }

    fn derived_module(
        &self,
        functor: DerivedFunctor,
        coefficient_module: &PresentedModule<PolynomialQuotient<C>>,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialQuotient<C>>, DerivedError> {
        require_justified_degree(self, degree)?;
        let (incoming, outgoing) = derived_maps(functor, self, coefficient_module, degree)?;
        let homology = if functor == DerivedFunctor::Tor && degree == 0 {
            PresentedModuleHomology::from_cokernel_complex(&incoming, &outgoing)?
        } else {
            outgoing.homology(&incoming, order.clone())?
        };
        DerivedModule::from_construction(
            functor,
            degree,
            self.clone(),
            coefficient_module.clone(),
            order,
            (incoming, outgoing),
            homology,
        )
    }
}

impl<C: Field> PresentedModule<PolynomialQuotient<C>> {
    /// Compute `Tor_degree(self, rhs)` through the required explicit
    /// quotient-resolution look-ahead degree.
    pub fn tor(
        &self,
        rhs: &Self,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialQuotient<C>>, DerivedError> {
        if degree == 0 {
            return FreeResolution::presentation_through_one(self)?.tor(rhs, degree, order);
        }
        let through = degree
            .checked_add(1)
            .ok_or(DerivedError::DegreeNotJustified {
                requested: degree,
                justified_through: None,
            })?;
        self.free_resolution_through(order.clone(), through)?
            .tor(rhs, degree, order)
    }

    /// Compute `Ext^degree(self, target)` through the required explicit
    /// quotient-resolution look-ahead degree.
    pub fn ext(
        &self,
        target: &Self,
        degree: usize,
        order: ModuleMonomialOrder,
    ) -> Result<DerivedModule<PolynomialQuotient<C>>, DerivedError> {
        let through = degree
            .checked_add(1)
            .ok_or(DerivedError::DegreeNotJustified {
                requested: degree,
                justified_through: None,
            })?;
        self.free_resolution_through(order.clone(), through)?
            .ext(target, degree, order)
    }
}

fn require_justified_degree<R: Ring>(
    resolution: &FreeResolution<R>,
    degree: usize,
) -> Result<(), DerivedError> {
    match resolution.completeness() {
        ResolutionCompleteness::Complete => Ok(()),
        ResolutionCompleteness::ThroughHomologicalDegree { degree: through }
            if degree < *through =>
        {
            Ok(())
        }
        ResolutionCompleteness::ThroughHomologicalDegree { degree: through } => {
            Err(DerivedError::DegreeNotJustified {
                requested: degree,
                justified_through: through.checked_sub(1),
            })
        }
    }
}

fn verify_derived_homomorphism_structure<R: Ring>(
    value: &DerivedModuleHomomorphism<R>,
) -> Result<(), DerivedError> {
    if value.source.functor != value.target.functor
        || value.source.degree != value.target.degree
        || value.source.resolution != value.target.resolution
        || value.source.order != value.target.order
        || value.coefficient_homomorphism.source() != value.source.coefficient_module()
        || value.coefficient_homomorphism.target() != value.target.coefficient_module()
        || value.homomorphism.source() != value.source.module()
        || value.homomorphism.target() != value.target.module()
    {
        return Err(DerivedError::InvalidEvidence(
            "a derived homomorphism has incompatible retained endpoints",
        ));
    }
    Ok(())
}

fn zero_presented<R: Ring>(ring: &R) -> Result<PresentedModule<R>, ModuleError> {
    PresentedModule::free(FreeModule::new(ring.clone(), 0))
}

fn zero_coefficient_term_map<R: Ring>(
    ring: &R,
) -> Result<PresentedModuleHomomorphism<R>, ModuleError> {
    PresentedModuleHomomorphism::zero(zero_presented(ring)?, zero_presented(ring)?)
}

fn derived_coefficient_term_map<R: CommutativeRing>(
    functor: DerivedFunctor,
    resolution: &FreeResolution<R>,
    coefficient_homomorphism: &PresentedModuleHomomorphism<R>,
    degree: usize,
) -> Result<PresentedModuleHomomorphism<R>, ModuleError> {
    let Some(free) = resolution.free_module(degree) else {
        return zero_coefficient_term_map(resolution.module().generator_module().ring());
    };
    let free = match functor {
        DerivedFunctor::Tor => free.clone(),
        DerivedFunctor::Ext => free.dual()?,
    };
    PresentedModuleHomomorphism::from_free_map(ModuleHomomorphism::identity(free)?)?
        .tensor(coefficient_homomorphism)
}

fn derived_coefficient_chain_maps<R: CommutativeRing>(
    functor: DerivedFunctor,
    resolution: &FreeResolution<R>,
    coefficient_homomorphism: &PresentedModuleHomomorphism<R>,
    degree: usize,
) -> Result<DerivedChainMaps<R>, DerivedError> {
    let ring = resolution.module().generator_module().ring();
    let adjacent = |degree| {
        derived_coefficient_term_map(functor, resolution, coefficient_homomorphism, degree)
    };
    let middle = adjacent(degree)?;
    let (left, right) = match functor {
        DerivedFunctor::Tor => (
            degree
                .checked_add(1)
                .map_or_else(|| zero_coefficient_term_map(ring), adjacent)?,
            degree
                .checked_sub(1)
                .map_or_else(|| zero_coefficient_term_map(ring), adjacent)?,
        ),
        DerivedFunctor::Ext => (
            degree
                .checked_sub(1)
                .map_or_else(|| zero_coefficient_term_map(ring), adjacent)?,
            degree
                .checked_add(1)
                .map_or_else(|| zero_coefficient_term_map(ring), adjacent)?,
        ),
    };
    Ok((left, middle, right))
}

fn tensor_term<R: CommutativeRing>(
    resolution: &FreeResolution<R>,
    coefficient_module: &PresentedModule<R>,
    degree: usize,
) -> Result<PresentedModule<R>, ModuleError> {
    match resolution.free_module(degree) {
        Some(module) => PresentedModule::free(module.clone())?.tensor(coefficient_module),
        None => zero_presented(resolution.module().generator_module().ring()),
    }
}

fn hom_term<R: CommutativeRing>(
    resolution: &FreeResolution<R>,
    coefficient_module: &PresentedModule<R>,
    degree: usize,
) -> Result<PresentedModule<R>, ModuleError> {
    match resolution.free_module(degree) {
        Some(module) => PresentedModule::free(module.dual()?)?.tensor(coefficient_module),
        None => zero_presented(resolution.module().generator_module().ring()),
    }
}

fn tensor_differential<R: CommutativeRing>(
    differential: &ModuleHomomorphism<R>,
    coefficient_module: &PresentedModule<R>,
) -> Result<PresentedModuleHomomorphism<R>, ModuleError> {
    PresentedModuleHomomorphism::from_free_map(differential.clone())?.tensor(
        &PresentedModuleHomomorphism::identity(coefficient_module.clone())?,
    )
}

fn hom_differential<R: CommutativeRing>(
    differential: &ModuleHomomorphism<R>,
    coefficient_module: &PresentedModule<R>,
) -> Result<PresentedModuleHomomorphism<R>, ModuleError> {
    PresentedModuleHomomorphism::from_free_map(differential.dual()?)?.tensor(
        &PresentedModuleHomomorphism::identity(coefficient_module.clone())?,
    )
}

fn derived_maps<R: CommutativeRing>(
    functor: DerivedFunctor,
    resolution: &FreeResolution<R>,
    coefficient_module: &PresentedModule<R>,
    degree: usize,
) -> Result<DerivedComplexMaps<R>, DerivedError> {
    let ring = resolution.module().generator_module().ring();
    let zero = || zero_presented(ring);
    let lookahead = degree
        .checked_add(1)
        .ok_or(DerivedError::DegreeNotJustified {
            requested: degree,
            justified_through: None,
        })?;
    match functor {
        DerivedFunctor::Tor => {
            let middle = tensor_term(resolution, coefficient_module, degree)?;
            let incoming = match resolution.differential(lookahead) {
                Some(differential) => tensor_differential(differential, coefficient_module)?,
                None => PresentedModuleHomomorphism::zero(zero()?, middle.clone())?,
            };
            let outgoing = match degree
                .checked_sub(1)
                .and_then(|_| resolution.differential(degree))
            {
                Some(differential) => tensor_differential(differential, coefficient_module)?,
                None => PresentedModuleHomomorphism::zero(middle, zero()?)?,
            };
            Ok((incoming, outgoing))
        }
        DerivedFunctor::Ext => {
            let middle = hom_term(resolution, coefficient_module, degree)?;
            let incoming = match degree
                .checked_sub(1)
                .and_then(|_| resolution.differential(degree))
            {
                Some(differential) => hom_differential(differential, coefficient_module)?,
                None => PresentedModuleHomomorphism::zero(zero()?, middle.clone())?,
            };
            let outgoing = match resolution.differential(lookahead) {
                Some(differential) => hom_differential(differential, coefficient_module)?,
                None => PresentedModuleHomomorphism::zero(middle, zero()?)?,
            };
            Ok((incoming, outgoing))
        }
    }
}

impl<R> Serialize for DerivedModule<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("DerivedModule", 7)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("functor", &self.functor)?;
        state.serialize_field("degree", &self.degree)?;
        state.serialize_field("resolution", &self.resolution)?;
        state.serialize_field("coefficient_module", &self.coefficient_module)?;
        state.serialize_field("order", &self.order)?;
        state.serialize_field("homology", &self.homology)?;
        state.end()
    }
}

impl<R> Serialize for DerivedModuleHomomorphism<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("DerivedModuleHomomorphism", 5)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("target", &self.target)?;
        state.serialize_field("coefficient_homomorphism", &self.coefficient_homomorphism)?;
        state.serialize_field("homomorphism", &self.homomorphism)?;
        state.end()
    }
}

impl<'de, C> Deserialize<'de> for DerivedModule<PolynomialRing<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DerivedModuleWire::<PolynomialRing<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/DerivedModule@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                "noether/homalg/DerivedModule@1", wire.type_id
            )));
        }
        let result = Self {
            functor: wire.functor,
            degree: wire.degree,
            resolution: wire.resolution,
            coefficient_module: wire.coefficient_module,
            order: wire.order,
            homology: wire.homology,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<'de, C> Deserialize<'de> for DerivedModuleHomomorphism<PolynomialRing<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DerivedModuleHomomorphismWire::<PolynomialRing<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/DerivedModuleHomomorphism@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                "noether/homalg/DerivedModuleHomomorphism@1", wire.type_id
            )));
        }
        let result = Self {
            source: wire.source,
            target: wire.target,
            coefficient_homomorphism: wire.coefficient_homomorphism,
            homomorphism: wire.homomorphism,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<'de, C> Deserialize<'de> for DerivedModule<PolynomialQuotient<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DerivedModuleWire::<PolynomialQuotient<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/DerivedModule@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                "noether/homalg/DerivedModule@1", wire.type_id
            )));
        }
        let result = Self {
            functor: wire.functor,
            degree: wire.degree,
            resolution: wire.resolution,
            coefficient_module: wire.coefficient_module,
            order: wire.order,
            homology: wire.homology,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<'de, C> Deserialize<'de> for DerivedModuleHomomorphism<PolynomialQuotient<C>>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire =
            DerivedModuleHomomorphismWire::<PolynomialQuotient<C>>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/DerivedModuleHomomorphism@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                "noether/homalg/DerivedModuleHomomorphism@1", wire.type_id
            )));
        }
        let result = Self {
            source: wire.source,
            target: wire.target,
            coefficient_homomorphism: wire.coefficient_homomorphism,
            homomorphism: wire.homomorphism,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<R: Ring> fmt::Debug for DerivedModule<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DerivedModule")
            .field("functor", &self.functor)
            .field("degree", &self.degree)
            .field("generators", &self.module().generator_count())
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Debug for DerivedModuleHomomorphism<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DerivedModuleHomomorphism")
            .field("functor", &self.source.functor)
            .field("degree", &self.source.degree)
            .finish_non_exhaustive()
    }
}

impl<R: Ring> fmt::Display for DerivedModule<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} degree {} ({} generators)",
            self.functor,
            self.degree,
            self.module().generator_count()
        )
    }
}

impl<R: Ring> fmt::Display for DerivedModuleHomomorphism<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "induced {} map in degree {}",
            self.source.functor, self.source.degree
        )
    }
}

impl<R> SemanticType for DerivedModule<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = "noether/homalg/DerivedModule@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &DERIVED_MODULE_DESCRIPTION
    }
}

impl<R> SemanticType for DerivedModuleHomomorphism<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    const TYPE_ID: &'static str = "noether/homalg/DerivedModuleHomomorphism@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &DERIVED_MODULE_HOMOMORPHISM_DESCRIPTION
    }
}

#[cfg(test)]
mod tests {
    use crate::commalg::{MonomialOrder, PolynomialRing};
    use crate::homalg::{ModuleElement, Submodule};
    use crate::linalg::Matrix;
    use crate::Q;

    use super::*;

    fn residue_field_module() -> PresentedModule<PolynomialRing<crate::RationalField>> {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = ring.generators().unwrap().remove(0);
        PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                FreeModule::new(ring.clone(), 1),
                FreeModule::new(ring.clone(), 1),
                Matrix::from_rows(ring, [[x]]).unwrap(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn tor_and_ext_of_a_residue_field_retain_verified_homology_diagrams() {
        let module = residue_field_module();
        let ring = module.generator_module().ring();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let resolution = module.free_resolution(order.clone()).unwrap();

        for functor in [DerivedFunctor::Tor, DerivedFunctor::Ext] {
            for degree in 0..=1 {
                let result = match functor {
                    DerivedFunctor::Tor => resolution.tor(&module, degree, order.clone()).unwrap(),
                    DerivedFunctor::Ext => resolution.ext(&module, degree, order.clone()).unwrap(),
                };
                result.verify().unwrap();
                assert_eq!(result.functor(), functor);
                assert_eq!(result.degree(), degree);
                assert_eq!(result.module().generator_count(), 1);
                let canonical_relations = Submodule::from_generator_map(
                    result.module().generator_module().clone(),
                    result.module().presentation().clone(),
                )
                .unwrap()
                .groebner_basis(order.clone())
                .unwrap();
                assert_eq!(canonical_relations.basis().generator_count(), 1);
                assert_eq!(
                    canonical_relations.basis().generator_map().matrix(),
                    module.presentation().matrix()
                );
                let round_trip = DerivedModule::<PolynomialRing<crate::RationalField>>::from_json(
                    &result.to_json().unwrap(),
                )
                .unwrap();
                assert_eq!(round_trip, result);
            }
        }

        let zero = resolution.tor(&module, 2, order).unwrap();
        zero.verify().unwrap();
        assert_eq!(zero.module().generator_count(), 0);
    }

    #[test]
    fn a_truncated_resolution_does_not_claim_its_last_derived_degree() {
        let module = residue_field_module();
        let ring = module.generator_module().ring();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let truncated = module.free_resolution_through(order.clone(), 0).unwrap();

        assert!(matches!(
            truncated.tor(&module, 0, order),
            Err(DerivedError::DegreeNotJustified {
                requested: 0,
                justified_through: None,
            })
        ));
    }

    #[test]
    fn coefficient_maps_induce_functorial_tor_and_ext_exact_sequence_fragments() {
        let module = residue_field_module();
        let ring = module.generator_module().ring();
        let x = ring.generators().unwrap().remove(0);
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let free_module = FreeModule::new(ring.clone(), 1);
        let free = PresentedModule::free(free_module.clone()).unwrap();
        let multiplication_by_x = PresentedModuleHomomorphism::from_free_map(
            ModuleHomomorphism::from_matrix(
                free_module.clone(),
                free_module,
                Matrix::from_rows(ring.clone(), [[x]]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let quotient = multiplication_by_x.cokernel_projection().unwrap();

        for (functor, degree) in [(DerivedFunctor::Tor, 0), (DerivedFunctor::Ext, 1)] {
            let source = match functor {
                DerivedFunctor::Tor => module.tor(&free, degree, order.clone()).unwrap(),
                DerivedFunctor::Ext => module.ext(&free, degree, order.clone()).unwrap(),
            };
            let first = source.map_coefficients(&multiplication_by_x).unwrap();
            let second = first.target().map_coefficients(&quotient).unwrap();
            first.verify().unwrap();
            second.verify().unwrap();
            assert!(second
                .homomorphism()
                .is_exact_at_source(first.homomorphism(), order.clone())
                .unwrap());

            let identity = PresentedModuleHomomorphism::identity(free.clone()).unwrap();
            let induced_identity = source.map_coefficients(&identity).unwrap();
            assert!(induced_identity
                .homomorphism()
                .is_equal(
                    &PresentedModuleHomomorphism::identity(source.module().clone()).unwrap(),
                    order.clone(),
                )
                .unwrap());
            let round_trip =
                DerivedModuleHomomorphism::<PolynomialRing<crate::RationalField>>::from_json(
                    &first.to_json().unwrap(),
                )
                .unwrap();
            assert_eq!(round_trip, first);
        }
    }

    #[test]
    fn quotient_ring_tor_and_ext_use_honest_periodic_truncations() {
        let polynomial_ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let x = polynomial_ring.generators().unwrap().remove(0);
        let quotient = polynomial_ring
            .quotient(&polynomial_ring.ideal([x.pow(2).unwrap()]).unwrap())
            .unwrap();
        let x_bar = quotient.element(&x).unwrap();
        let free = FreeModule::new(quotient.clone(), 1);
        let module = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                free.clone(),
                free,
                Matrix::from_rows(quotient.clone(), [[x_bar.clone()]]).unwrap(),
            )
            .unwrap(),
        );
        let order = ModuleMonomialOrder::TermOverPosition(polynomial_ring.term_order());

        for degree in 0..=2 {
            for functor in [DerivedFunctor::Tor, DerivedFunctor::Ext] {
                let derived = match functor {
                    DerivedFunctor::Tor => module.tor(&module, degree, order.clone()).unwrap(),
                    DerivedFunctor::Ext => module.ext(&module, degree, order.clone()).unwrap(),
                };
                derived.verify().unwrap();
                assert_eq!(derived.functor(), functor);
                assert_eq!(derived.degree(), degree);
                assert_eq!(derived.justified_through_degree(), Some(degree));
                let identity = PresentedModuleHomomorphism::identity(module.clone()).unwrap();
                let induced_identity = derived.map_coefficients(&identity).unwrap();
                induced_identity.verify().unwrap();
                assert!(induced_identity
                    .homomorphism()
                    .is_equal(
                        &PresentedModuleHomomorphism::identity(derived.module().clone()).unwrap(),
                        order.clone(),
                    )
                    .unwrap());
                let map_round_trip = DerivedModuleHomomorphism::<
                    PolynomialQuotient<crate::RationalField>,
                >::from_json(
                    &induced_identity.to_json().unwrap()
                )
                .unwrap();
                assert_eq!(map_round_trip, induced_identity);
                let relations = Submodule::from_generator_map(
                    derived.module().generator_module().clone(),
                    derived.module().presentation().clone(),
                )
                .unwrap()
                .groebner_basis(order.clone())
                .unwrap();
                let zero = ModuleElement::zero(derived.module().generator_module().clone());
                let mut surviving_normal_forms = Vec::new();
                for position in 0..derived.module().generator_count() {
                    let normal = relations
                        .normal_form(
                            &derived
                                .module()
                                .generator_module()
                                .basis_element(position)
                                .unwrap(),
                        )
                        .unwrap();
                    if normal != zero && !surviving_normal_forms.contains(&normal) {
                        surviving_normal_forms.push(normal);
                    }
                }
                assert_eq!(surviving_normal_forms.len(), 1);
                assert_eq!(
                    relations
                        .normal_form(&surviving_normal_forms[0].scalar_mul(&x_bar).unwrap(),)
                        .unwrap(),
                    zero
                );
                let round_trip =
                    DerivedModule::<PolynomialQuotient<crate::RationalField>>::from_json(
                        &derived.to_json().unwrap(),
                    )
                    .unwrap();
                assert_eq!(round_trip, derived);
            }
        }
    }
}
