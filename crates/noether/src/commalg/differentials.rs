// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Kähler differentials and exact cotangent constructions for affine algebras.

use core::fmt;
use core::ops::{Add, Neg, Sub};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::homalg::{
    FreeModule, ModuleElement, ModuleError, ModuleGroebnerBasis, ModuleGroebnerError,
    ModuleHomomorphism, ModuleMonomialOrder, PresentedModule, PresentedModuleHomomorphism,
    Submodule,
};
use crate::linalg::{LinalgError, Matrix, Vector};
use crate::{
    ContextMismatchError, Field, NoetherError, Ring, SemanticType, TypeDescription,
    VerificationError,
};

use super::{
    Degree, GroebnerError, Polynomial, PolynomialError, PolynomialRing, QuotientElement,
    QuotientRing,
};

type AffineAlgebra<C> = QuotientRing<PolynomialRing<C>>;
type AffineAlgebraElement<C> = QuotientElement<PolynomialRing<C>>;

static KAHLER_DIFFERENTIALS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/KahlerDifferentials@1",
    rust_path: "noether::commalg::KahlerDifferentials",
    python_path: Some("noether.commalg.KahlerDifferentials"),
};

static KAHLER_DIFFERENTIAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/KahlerDifferential@1",
    rust_path: "noether::commalg::KahlerDifferential",
    python_path: Some("noether.commalg.KahlerDifferential"),
};

static CONORMAL_SEQUENCE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/ConormalSequenceEvidence@1",
    rust_path: "noether::commalg::ConormalSequenceEvidence",
    python_path: Some("noether.commalg.ConormalSequenceEvidence"),
};

static COTANGENT_COMPLEX_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CotangentComplexPresentation@1",
    rust_path: "noether::commalg::CotangentComplexPresentation",
    python_path: Some("noether.commalg.CotangentComplexPresentation"),
};

static COTANGENT_SPACE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/CotangentSpace@1",
    rust_path: "noether::commalg::CotangentSpace",
    python_path: Some("noether.commalg.CotangentSpace"),
};

/// Failure to construct or verify an exact differential object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KahlerDifferentialError {
    /// An input belongs to another exact polynomial or quotient context.
    ContextMismatch(ContextMismatchError),
    /// Polynomial arithmetic or differentiation failed.
    Polynomial(PolynomialError),
    /// Ideal construction, dimension, or reduction failed.
    Groebner(GroebnerError),
    /// A based-module construction failed.
    Module(ModuleError),
    /// A module Gröbner computation or exact replay failed.
    ModuleGroebner(ModuleGroebnerError),
    /// Exact field linear algebra failed.
    LinearAlgebra(LinalgError),
    /// A point has the wrong number of coordinates.
    PointArity {
        /// Required affine coordinate count.
        expected: usize,
        /// Supplied coordinate count.
        actual: usize,
    },
    /// One point coordinate is outside the coefficient field context.
    PointCoordinateOutsideField {
        /// Zero-based coordinate position.
        coordinate: usize,
    },
    /// The supplied point does not annihilate one defining equation.
    PointNotOnAlgebra {
        /// Zero-based canonical defining-equation position.
        relation: usize,
    },
    /// Explicit generators do not generate the algebra's defining ideal.
    GeneratorsDoNotPresentAlgebra,
    /// The bounded cotangent-complex lane could not prove a regular sequence.
    CompleteIntersectionRequired {
        /// Number of chosen equations.
        generator_count: usize,
        /// Exactly computed codimension when the algebra is nonzero.
        codimension: Option<usize>,
    },
    /// A zero algebra has no supported affine complete-intersection lane.
    ZeroAlgebra,
    /// A machine-word dimension conversion overflowed.
    DimensionOverflow,
    /// Retained evidence did not replay exactly.
    Verification(VerificationError),
}

impl fmt::Display for KahlerDifferentialError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Module(error) => error.fmt(formatter),
            Self::ModuleGroebner(error) => error.fmt(formatter),
            Self::LinearAlgebra(error) => error.fmt(formatter),
            Self::PointArity { expected, actual } => write!(
                formatter,
                "affine algebra has {expected} coordinates; point has {actual}"
            ),
            Self::PointCoordinateOutsideField { coordinate } => write!(
                formatter,
                "point coordinate {coordinate} is outside the coefficient field"
            ),
            Self::PointNotOnAlgebra { relation } => write!(
                formatter,
                "point does not annihilate defining relation {relation}"
            ),
            Self::GeneratorsDoNotPresentAlgebra => formatter.write_str(
                "chosen equations do not generate the algebra's canonical defining ideal",
            ),
            Self::CompleteIntersectionRequired {
                generator_count,
                codimension: Some(codimension),
            } => write!(
                formatter,
                "chosen presentation has {generator_count} equations but codimension {codimension}; a proved complete intersection is required"
            ),
            Self::CompleteIntersectionRequired {
                generator_count,
                codimension: None,
            } => write!(
                formatter,
                "the codimension of the presentation with {generator_count} equations is unavailable"
            ),
            Self::ZeroAlgebra => formatter.write_str(
                "the zero algebra is outside the complete-intersection cotangent-complex lane",
            ),
            Self::DimensionOverflow => {
                formatter.write_str("an affine dimension does not fit the supported machine word")
            }
            Self::Verification(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for KahlerDifferentialError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::Module(error) => Some(error),
            Self::ModuleGroebner(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::Verification(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ContextMismatchError> for KahlerDifferentialError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

impl From<PolynomialError> for KahlerDifferentialError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<GroebnerError> for KahlerDifferentialError {
    fn from(error: GroebnerError) -> Self {
        Self::Groebner(error)
    }
}

impl From<ModuleError> for KahlerDifferentialError {
    fn from(error: ModuleError) -> Self {
        Self::Module(error)
    }
}

impl From<ModuleGroebnerError> for KahlerDifferentialError {
    fn from(error: ModuleGroebnerError) -> Self {
        Self::ModuleGroebner(error)
    }
}

impl From<LinalgError> for KahlerDifferentialError {
    fn from(error: LinalgError) -> Self {
        Self::LinearAlgebra(error)
    }
}

impl From<VerificationError> for KahlerDifferentialError {
    fn from(error: VerificationError) -> Self {
        Self::Verification(error)
    }
}

/// Exact replayable evidence for the conormal sequence of an affine algebra.
///
/// For `A = C[x_1,...,x_n]/I`, this value retains chosen generators of `I`,
/// an exact syzygy presentation of `I/I^2`, and the induced map
/// `I/I^2 -> A^n` sending an equation to its gradient. The cokernel of that
/// map is `Omega_{A/C}`. The source is not silently replaced by a free module:
/// redundant equations and non-complete intersections retain their exact
/// conormal relations.
#[derive(Clone)]
pub struct ConormalSequenceEvidence<C: Field> {
    algebra: AffineAlgebra<C>,
    chosen_generators: Vec<Polynomial<C>>,
    conormal_module: PresentedModule<AffineAlgebra<C>>,
    ambient_differentials: PresentedModule<AffineAlgebra<C>>,
    conormal_map: PresentedModuleHomomorphism<AffineAlgebra<C>>,
    jacobian: Matrix<AffineAlgebra<C>>,
}

impl<C: Field> ConormalSequenceEvidence<C> {
    /// Affine algebra `A` in the conormal sequence.
    pub fn algebra(&self) -> &AffineAlgebra<C> {
        &self.algebra
    }

    /// Ordered equations whose exact syzygies present `I/I^2`.
    pub fn chosen_generators(&self) -> &[Polynomial<C>] {
        &self.chosen_generators
    }

    /// Exact presented `A`-module `I/I^2`.
    pub fn conormal_module(&self) -> &PresentedModule<AffineAlgebra<C>> {
        &self.conormal_module
    }

    /// Relation-free module `Omega_{C[x]/C} tensor A` with basis `dx_i`.
    pub fn ambient_differentials(&self) -> &PresentedModule<AffineAlgebra<C>> {
        &self.ambient_differentials
    }

    /// Checked conormal map `I/I^2 -> Omega_{C[x]/C} tensor A`.
    pub fn conormal_map(&self) -> &PresentedModuleHomomorphism<AffineAlgebra<C>> {
        &self.conormal_map
    }

    /// Jacobian with one row per chosen equation and one column per variable.
    pub fn jacobian(&self) -> &Matrix<AffineAlgebra<C>> {
        &self.jacobian
    }

    /// Recompute the syzygy presentation and Jacobian and compare every map.
    pub fn verify(&self) -> Result<(), KahlerDifferentialError> {
        let expected = build_conormal_evidence(&self.algebra, &self.chosen_generators)?;
        if self.chosen_generators != expected.chosen_generators
            || self.conormal_module != expected.conormal_module
            || self.ambient_differentials != expected.ambient_differentials
            || self.conormal_map.source() != expected.conormal_map.source()
            || self.conormal_map.target() != expected.conormal_map.target()
            || self.conormal_map.generator_lift() != expected.conormal_map.generator_lift()
            || self.conormal_map.relation_lift() != expected.conormal_map.relation_lift()
            || self.jacobian != expected.jacobian
        {
            return Err(VerificationError::InvalidWitness {
                claim: "conormal sequence replays from its chosen equations",
            }
            .into());
        }
        self.conormal_map.verify()?;
        Ok(())
    }

    /// Serialize the algebra and chosen equations; deserialization rebuilds
    /// and exactly reverifies all derived evidence.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        C: Serialize,
        C::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly reconstruct the conormal evidence.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> PartialEq for ConormalSequenceEvidence<C> {
    fn eq(&self, other: &Self) -> bool {
        self.algebra == other.algebra
            && self.chosen_generators == other.chosen_generators
            && self.conormal_module == other.conormal_module
            && self.ambient_differentials == other.ambient_differentials
            && self.conormal_map.source() == other.conormal_map.source()
            && self.conormal_map.target() == other.conormal_map.target()
            && self.conormal_map.generator_lift() == other.conormal_map.generator_lift()
            && self.conormal_map.relation_lift() == other.conormal_map.relation_lift()
            && self.jacobian == other.jacobian
    }
}

impl<C: Field> Eq for ConormalSequenceEvidence<C> {}

impl<C: Field> fmt::Debug for ConormalSequenceEvidence<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConormalSequenceEvidence")
            .field("variables", &self.algebra.base_ring().variables().len())
            .field("equations", &self.chosen_generators.len())
            .finish_non_exhaustive()
    }
}

impl<C: Field> SemanticType for ConormalSequenceEvidence<C> {
    const TYPE_ID: &'static str = "noether/commalg/ConormalSequenceEvidence@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut bytes = self.algebra.canonical_bytes()?;
        for generator in &self.chosen_generators {
            append_component(&mut bytes, &generator.canonical_bytes()?)?;
        }
        Ok(bytes)
    }

    fn describe() -> &'static TypeDescription {
        &CONORMAL_SEQUENCE_DESCRIPTION
    }
}

/// The exact module of Kähler differentials `Omega_{A/C}`.
///
/// The module is the cokernel of the verified conormal map. Its element
/// constructor and universal derivation reduce through a complete module
/// Gröbner basis, so equality of returned [`KahlerDifferential`] values is
/// equality in `Omega_{A/C}`, not equality of arbitrary lifts in `A^n`.
///
/// # Examples
///
/// ```
/// use noether::commalg::{MonomialOrder, PolynomialRing};
/// use noether::Q;
///
/// let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic)?;
/// let x = ring.generator(0)?;
/// let omega = ring.kahler_differentials()?;
/// let dx = omega.differentiate_polynomial(&x)?;
/// assert_eq!(omega.module().generator_count(), 1);
/// assert!(!dx.is_zero());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct KahlerDifferentials<C: Field> {
    algebra: AffineAlgebra<C>,
    module: PresentedModule<AffineAlgebra<C>>,
    conormal: ConormalSequenceEvidence<C>,
    relation_basis: ModuleGroebnerBasis<AffineAlgebra<C>>,
}

impl<C: Field> KahlerDifferentials<C> {
    /// Construct `Omega_{A/C}` from the canonical defining ideal of `A`.
    pub fn new(algebra: AffineAlgebra<C>) -> Result<Self, KahlerDifferentialError> {
        let chosen_generators = algebra.defining_ideal().generators().to_vec();
        let conormal = build_conormal_evidence(&algebra, &chosen_generators)?;
        Self::from_conormal(conormal)
    }

    fn from_conormal(
        conormal: ConormalSequenceEvidence<C>,
    ) -> Result<Self, KahlerDifferentialError> {
        let algebra = conormal.algebra.clone();
        let relation_map = conormal.conormal_map.generator_lift().clone();
        let module = PresentedModule::cokernel(relation_map.clone());
        let relation_submodule = Submodule::from_generator_map(
            conormal.ambient_differentials.generator_module().clone(),
            relation_map,
        )?;
        let relation_basis = relation_submodule.groebner_basis(module_order(&algebra))?;
        let result = Self {
            algebra,
            module,
            conormal,
            relation_basis,
        };
        #[cfg(debug_assertions)]
        result.verify()?;
        Ok(result)
    }

    /// Exact affine-algebra coefficient context.
    pub fn algebra(&self) -> &AffineAlgebra<C> {
        &self.algebra
    }

    /// Retained finite presentation of `Omega_{A/C}`.
    pub fn module(&self) -> &PresentedModule<AffineAlgebra<C>> {
        &self.module
    }

    /// Exact conormal and Jacobian evidence behind the presentation.
    pub fn conormal_evidence(&self) -> &ConormalSequenceEvidence<C> {
        &self.conormal
    }

    /// Number of ambient coordinate differentials `dx_i`.
    pub fn ambient_generator_count(&self) -> usize {
        self.algebra.base_ring().variables().len()
    }

    /// Universal derivation of one quotient element.
    pub fn differentiate(
        &self,
        value: &AffineAlgebraElement<C>,
    ) -> Result<KahlerDifferential<C>, KahlerDifferentialError> {
        if value.ring() != &self.algebra {
            return Err(
                context_mismatch("universal derivation", &self.algebra, value.ring()).into(),
            );
        }
        self.differentiate_polynomial(value.lift())
    }

    /// Derive an ambient polynomial and project the result to `Omega_{A/C}`.
    pub fn differentiate_polynomial(
        &self,
        value: &Polynomial<C>,
    ) -> Result<KahlerDifferential<C>, KahlerDifferentialError> {
        if value.ring() != self.algebra.base_ring() {
            return Err(ContextMismatchError {
                expected: self.algebra.base_ring().to_string(),
                actual: value.ring().to_string(),
            }
            .into());
        }
        let coordinates = (0..self.ambient_generator_count())
            .map(|variable| {
                value
                    .derivative(variable)
                    .map_err(KahlerDifferentialError::from)
                    .and_then(|derivative| {
                        self.algebra
                            .element(&derivative)
                            .map_err(KahlerDifferentialError::from)
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let representative = ModuleElement::from_coordinates(
            self.module.generator_module().clone(),
            Vector::from_values(self.algebra.clone(), coordinates)?,
        )?;
        self.element(representative)
    }

    /// Canonicalize an ambient `A^n` representative as a differential class.
    pub fn element(
        &self,
        representative: ModuleElement<AffineAlgebra<C>>,
    ) -> Result<KahlerDifferential<C>, KahlerDifferentialError> {
        if representative.module() != self.module.generator_module() {
            return Err(ModuleError::ModuleMismatch {
                operation: "Kahler differential construction",
            }
            .into());
        }
        let representative = self.relation_basis.normal_form(&representative)?;
        Ok(KahlerDifferential {
            differentials: self.clone(),
            representative,
        })
    }

    /// Add two canonical differential classes.
    pub fn add(
        &self,
        left: &KahlerDifferential<C>,
        right: &KahlerDifferential<C>,
    ) -> Result<KahlerDifferential<C>, KahlerDifferentialError> {
        self.require_element(left)?;
        self.require_element(right)?;
        self.element(left.representative.add(&right.representative)?)
    }

    /// Subtract two canonical differential classes.
    pub fn sub(
        &self,
        left: &KahlerDifferential<C>,
        right: &KahlerDifferential<C>,
    ) -> Result<KahlerDifferential<C>, KahlerDifferentialError> {
        self.require_element(left)?;
        self.require_element(right)?;
        self.element(left.representative.sub(&right.representative)?)
    }

    /// Negate one canonical differential class.
    pub fn neg(
        &self,
        value: &KahlerDifferential<C>,
    ) -> Result<KahlerDifferential<C>, KahlerDifferentialError> {
        self.require_element(value)?;
        self.element(value.representative.neg()?)
    }

    /// Multiply a differential class by one scalar in `A`.
    pub fn scalar_mul(
        &self,
        scalar: &AffineAlgebraElement<C>,
        value: &KahlerDifferential<C>,
    ) -> Result<KahlerDifferential<C>, KahlerDifferentialError> {
        if !self.algebra.contains(scalar) {
            return Err(context_mismatch(
                "Kahler differential scalar multiplication",
                &self.algebra,
                scalar.ring(),
            )
            .into());
        }
        self.require_element(value)?;
        self.element(value.representative.scalar_mul(scalar)?)
    }

    /// The zero differential class.
    pub fn zero(&self) -> KahlerDifferential<C> {
        KahlerDifferential {
            differentials: self.clone(),
            representative: ModuleElement::zero(self.module.generator_module().clone()),
        }
    }

    /// Fiber `Omega_{A/C} tensor_A C` at an explicit rational or finite-field
    /// point. The point is checked against every defining equation first.
    pub fn cotangent_space_at(
        &self,
        point: Vec<C::Element>,
    ) -> Result<CotangentSpace<C>, KahlerDifferentialError> {
        CotangentSpace::new(self.algebra.clone(), point)
    }

    /// The exact two-term cotangent complex for the canonical equations when
    /// their count equals the exactly computed codimension.
    ///
    /// Polynomial rings over fields are Cohen--Macaulay, so a proper ideal
    /// generated by `codimension` elements is a regular sequence. Outside that
    /// proved complete-intersection lane this method fails closed.
    pub fn cotangent_complex(
        &self,
    ) -> Result<CotangentComplexPresentation<C>, KahlerDifferentialError> {
        CotangentComplexPresentation::from_evidence(self.conormal.clone())
    }

    /// Recompute the conormal presentation, module basis, and the fact that
    /// every chosen defining equation derives to zero in the cokernel.
    pub fn verify(&self) -> Result<(), KahlerDifferentialError> {
        self.conormal.verify()?;
        let expected_module =
            PresentedModule::cokernel(self.conormal.conormal_map.generator_lift().clone());
        if self.module != expected_module
            || self.relation_basis.input().ambient() != self.module.generator_module()
            || self.relation_basis.input().generator_map()
                != self.conormal.conormal_map.generator_lift()
        {
            return Err(VerificationError::InvalidWitness {
                claim: "Kahler module presentation matches its conormal map",
            }
            .into());
        }
        self.relation_basis.verify()?;
        for generator in &self.conormal.chosen_generators {
            if !self.differentiate_polynomial(generator)?.is_zero() {
                return Err(VerificationError::InvalidWitness {
                    claim: "each defining equation derives to zero in the presented cokernel",
                }
                .into());
            }
        }
        Ok(())
    }

    /// Serialize the affine algebra; deserialization reconstructs and
    /// reverifies all conormal and module-Gröbner evidence.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        C: Serialize,
        C::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and exactly reconstruct `Omega_{A/C}`.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn require_element(
        &self,
        value: &KahlerDifferential<C>,
    ) -> Result<(), KahlerDifferentialError> {
        if value.differentials == *self {
            Ok(())
        } else {
            Err(ModuleError::ModuleMismatch {
                operation: "Kahler differential operation",
            }
            .into())
        }
    }
}

impl<C: Field> PartialEq for KahlerDifferentials<C> {
    fn eq(&self, other: &Self) -> bool {
        self.algebra == other.algebra
            && self.module == other.module
            && self.conormal == other.conormal
            && self.relation_basis.input().generator_map()
                == other.relation_basis.input().generator_map()
            && self.relation_basis.basis().generator_map()
                == other.relation_basis.basis().generator_map()
    }
}

impl<C: Field> Eq for KahlerDifferentials<C> {}

impl<C: Field> fmt::Debug for KahlerDifferentials<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KahlerDifferentials")
            .field("generators", &self.module.generator_count())
            .field("relations", &self.module.relation_count())
            .finish_non_exhaustive()
    }
}

impl<C: Field> fmt::Display for KahlerDifferentials<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "KahlerDifferentials({} generators, {} relations)",
            self.module.generator_count(),
            self.module.relation_count()
        )
    }
}

impl<C: Field> SemanticType for KahlerDifferentials<C> {
    const TYPE_ID: &'static str = "noether/commalg/KahlerDifferentials@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.algebra.canonical_bytes()
    }

    fn describe() -> &'static TypeDescription {
        &KAHLER_DIFFERENTIALS_DESCRIPTION
    }
}

/// One canonical class in a [`KahlerDifferentials`] module.
#[derive(Clone)]
pub struct KahlerDifferential<C: Field> {
    differentials: KahlerDifferentials<C>,
    representative: ModuleElement<AffineAlgebra<C>>,
}

impl<C: Field> KahlerDifferential<C> {
    /// Module of Kähler differentials containing this class.
    pub fn differentials(&self) -> &KahlerDifferentials<C> {
        &self.differentials
    }

    /// Canonical normal-form coordinates in the ordered `dx_i` basis.
    pub fn coordinates(&self) -> &ModuleElement<AffineAlgebra<C>> {
        &self.representative
    }

    /// Whether this class is zero in `Omega_{A/C}`.
    pub fn is_zero(&self) -> bool {
        self.representative
            .coordinates()
            .values()
            .all(|coordinate| self.differentials.algebra.is_zero(&coordinate))
    }

    /// Add two classes in the same Kähler module.
    pub fn add(&self, rhs: &Self) -> Result<Self, KahlerDifferentialError> {
        self.differentials.add(self, rhs)
    }

    /// Subtract another class in the same Kähler module.
    pub fn sub(&self, rhs: &Self) -> Result<Self, KahlerDifferentialError> {
        self.differentials.sub(self, rhs)
    }

    /// Negate this differential class.
    pub fn neg(&self) -> Result<Self, KahlerDifferentialError> {
        self.differentials.neg(self)
    }

    /// Multiply this class by one element of its affine algebra.
    pub fn scalar_mul(
        &self,
        scalar: &AffineAlgebraElement<C>,
    ) -> Result<Self, KahlerDifferentialError> {
        self.differentials.scalar_mul(scalar, self)
    }

    /// Reverify the ambient Kähler module and replay this canonical class.
    pub fn verify(&self) -> Result<(), KahlerDifferentialError> {
        self.differentials.verify()?;
        self.differentials.require_element(self)?;
        let expected = self.differentials.element(self.representative.clone())?;
        if expected.representative != self.representative {
            return Err(VerificationError::InvalidWitness {
                claim: "Kahler differential is in canonical module normal form",
            }
            .into());
        }
        Ok(())
    }

    /// Serialize the module and canonical representative.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        C: Serialize,
        C::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reject a noncanonical or mismatched representative.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> PartialEq for KahlerDifferential<C> {
    fn eq(&self, other: &Self) -> bool {
        self.differentials == other.differentials && self.representative == other.representative
    }
}

impl<C: Field> Eq for KahlerDifferential<C> {}

impl<C: Field> fmt::Debug for KahlerDifferential<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KahlerDifferential")
            .field(
                "coordinates",
                &self.representative.coordinates().dimension(),
            )
            .finish_non_exhaustive()
    }
}

impl<C: Field> SemanticType for KahlerDifferential<C> {
    const TYPE_ID: &'static str = "noether/commalg/KahlerDifferential@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut bytes = self.differentials.canonical_bytes()?;
        append_component(&mut bytes, &self.representative.canonical_bytes()?)?;
        Ok(bytes)
    }

    fn describe() -> &'static TypeDescription {
        &KAHLER_DIFFERENTIAL_DESCRIPTION
    }
}

impl<C: Field> Add<&KahlerDifferential<C>> for &KahlerDifferential<C> {
    type Output = Result<KahlerDifferential<C>, KahlerDifferentialError>;

    fn add(self, rhs: &KahlerDifferential<C>) -> Self::Output {
        KahlerDifferential::add(self, rhs)
    }
}

impl<C: Field> Sub<&KahlerDifferential<C>> for &KahlerDifferential<C> {
    type Output = Result<KahlerDifferential<C>, KahlerDifferentialError>;

    fn sub(self, rhs: &KahlerDifferential<C>) -> Self::Output {
        KahlerDifferential::sub(self, rhs)
    }
}

impl<C: Field> Neg for &KahlerDifferential<C> {
    type Output = Result<KahlerDifferential<C>, KahlerDifferentialError>;

    fn neg(self) -> Self::Output {
        KahlerDifferential::neg(self)
    }
}

/// An exact two-term presentation of `L_{A/C}` in the proved affine
/// complete-intersection lane.
///
/// Degree `-1` is the retained syzygy presentation of `I/I^2`; degree `0` is
/// the free module with basis `dx_i`. The differential is the conormal map.
/// The amplitude claim `[-1, 0]` is made only after exact dimension and
/// equation-count checks prove that the chosen equations form a regular
/// sequence in the polynomial ring over a field.
#[derive(Clone, PartialEq, Eq)]
pub struct CotangentComplexPresentation<C: Field> {
    evidence: ConormalSequenceEvidence<C>,
    hypothesis: CotangentComplexHypothesis,
}

/// Typed cohomological amplitude of a bounded cotangent-complex result.
///
/// This metadata is issued only by a verified [`CotangentComplexPresentation`]
/// and deliberately has no standalone deserialization path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CotangentComplexAmplitude {
    lowest_degree: i8,
    highest_degree: i8,
}

impl CotangentComplexAmplitude {
    /// The amplitude of the cotangent complex of an affine complete
    /// intersection over a field.
    pub const MINUS_ONE_TO_ZERO: Self = Self {
        lowest_degree: -1,
        highest_degree: 0,
    };

    /// Lowest cohomological degree that may be nonzero.
    pub const fn lowest_degree(self) -> i8 {
        self.lowest_degree
    }

    /// Highest cohomological degree that may be nonzero.
    pub const fn highest_degree(self) -> i8 {
        self.highest_degree
    }
}

/// Exact hypothesis proved before publishing a bounded cotangent complex.
///
/// This metadata is issued only by a verified [`CotangentComplexPresentation`].
/// Its proof-bearing variant cannot be constructed outside this crate and the
/// type deliberately has no standalone deserialization path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CotangentComplexHypothesis {
    /// The chosen equations generate a proper ideal in a polynomial ring over
    /// a field (checked by exact ideal equality), and their number equals
    /// ambient dimension minus the quotient dimension computed from the
    /// defining ideal's exact Gröbner leading ideal. Cohen--Macaulayness of
    /// the polynomial ring then proves that they are a regular sequence.
    #[non_exhaustive]
    ProvedAffineCompleteIntersection {
        /// Dimension of the ambient polynomial algebra.
        ambient_dimension: usize,
        /// Exactly computed dimension of the quotient algebra.
        quotient_dimension: usize,
        /// Ambient dimension minus quotient dimension.
        codimension: usize,
        /// Number of retained chosen equations.
        equation_count: usize,
    },
}

impl CotangentComplexHypothesis {
    /// Dimension of the ambient polynomial algebra used in the proof.
    pub const fn ambient_dimension(self) -> usize {
        match self {
            Self::ProvedAffineCompleteIntersection {
                ambient_dimension, ..
            } => ambient_dimension,
        }
    }

    /// Quotient dimension computed exactly from the defining ideal's Gröbner
    /// leading ideal.
    pub const fn quotient_dimension(self) -> usize {
        match self {
            Self::ProvedAffineCompleteIntersection {
                quotient_dimension, ..
            } => quotient_dimension,
        }
    }

    /// Codimension computed as ambient dimension minus quotient dimension.
    pub const fn codimension(self) -> usize {
        match self {
            Self::ProvedAffineCompleteIntersection { codimension, .. } => codimension,
        }
    }

    /// Number of chosen equations compared exactly with the computed
    /// codimension.
    pub const fn equation_count(self) -> usize {
        match self {
            Self::ProvedAffineCompleteIntersection { equation_count, .. } => equation_count,
        }
    }
}

impl<C: Field> CotangentComplexPresentation<C> {
    fn from_evidence(
        evidence: ConormalSequenceEvidence<C>,
    ) -> Result<Self, KahlerDifferentialError> {
        let hypothesis =
            complete_intersection_hypothesis(&evidence.algebra, evidence.chosen_generators.len())?;
        let result = Self {
            evidence,
            hypothesis,
        };
        #[cfg(debug_assertions)]
        result.verify()?;
        Ok(result)
    }

    /// Exact affine algebra represented by this complex.
    pub fn algebra(&self) -> &AffineAlgebra<C> {
        self.evidence.algebra()
    }

    /// Presented degree-`-1` term `I/I^2`.
    pub fn conormal_module(&self) -> &PresentedModule<AffineAlgebra<C>> {
        self.evidence.conormal_module()
    }

    /// Free degree-zero term `Omega_{C[x]/C} tensor A`.
    pub fn ambient_differentials(&self) -> &PresentedModule<AffineAlgebra<C>> {
        self.evidence.ambient_differentials()
    }

    /// Differential from degree `-1` to degree `0`.
    pub fn differential(&self) -> &PresentedModuleHomomorphism<AffineAlgebra<C>> {
        self.evidence.conormal_map()
    }

    /// Complete-intersection codimension and rank of `I/I^2`.
    pub fn codimension(&self) -> usize {
        match self.hypothesis {
            CotangentComplexHypothesis::ProvedAffineCompleteIntersection {
                codimension, ..
            } => codimension,
        }
    }

    /// Proved cohomological amplitude `[-1, 0]`.
    pub const fn amplitude(&self) -> CotangentComplexAmplitude {
        CotangentComplexAmplitude::MINUS_ONE_TO_ZERO
    }

    /// Exact complete-intersection hypothesis established by construction.
    pub const fn hypothesis(&self) -> CotangentComplexHypothesis {
        self.hypothesis
    }

    /// Chosen regular-sequence equations and exact conormal evidence.
    pub fn evidence(&self) -> &ConormalSequenceEvidence<C> {
        &self.evidence
    }

    /// Recompute both the conormal evidence and complete-intersection proof.
    pub fn verify(&self) -> Result<(), KahlerDifferentialError> {
        self.evidence.verify()?;
        let expected = complete_intersection_hypothesis(
            &self.evidence.algebra,
            self.evidence.chosen_generators.len(),
        )?;
        if expected != self.hypothesis {
            return Err(VerificationError::InvalidWitness {
                claim: "cotangent-complex codimension replays",
            }
            .into());
        }
        Ok(())
    }

    /// Serialize the exact chosen presentation; deserialization reverifies it.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        C: Serialize,
        C::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reprove the complete-intersection lane.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> fmt::Debug for CotangentComplexPresentation<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CotangentComplexPresentation")
            .field("hypothesis", &self.hypothesis)
            .field("amplitude", &self.amplitude())
            .finish_non_exhaustive()
    }
}

impl<C: Field> SemanticType for CotangentComplexPresentation<C> {
    const TYPE_ID: &'static str = "noether/commalg/CotangentComplexPresentation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut bytes = self.evidence.canonical_bytes()?;
        match self.hypothesis {
            CotangentComplexHypothesis::ProvedAffineCompleteIntersection {
                ambient_dimension,
                quotient_dimension,
                codimension,
                equation_count,
            } => {
                append_usize(&mut bytes, ambient_dimension)?;
                append_usize(&mut bytes, quotient_dimension)?;
                append_usize(&mut bytes, codimension)?;
                append_usize(&mut bytes, equation_count)?;
            }
        }
        Ok(bytes)
    }

    fn describe() -> &'static TypeDescription {
        &COTANGENT_COMPLEX_DESCRIPTION
    }
}

/// Exact cotangent fiber at an explicit coefficient-field point of an affine
/// algebra.
///
/// This is `Omega_{A/C} tensor_A C`, presented as the cokernel of the
/// evaluated transposed Jacobian. `embedding_dimension` is its vector-space
/// dimension; it is deliberately distinct from the number of ambient affine
/// coordinates.
#[derive(Clone, PartialEq, Eq)]
pub struct CotangentSpace<C: Field> {
    algebra: AffineAlgebra<C>,
    point: Vec<C::Element>,
    evaluated_jacobian: Matrix<C>,
    module: PresentedModule<C>,
    jacobian_rank: usize,
    embedding_dimension: usize,
}

impl<C: Field> CotangentSpace<C> {
    fn new(
        algebra: AffineAlgebra<C>,
        point: Vec<C::Element>,
    ) -> Result<Self, KahlerDifferentialError> {
        validate_point(&algebra, &point)?;
        let ring = algebra.base_ring();
        let field = ring.coefficient_ring();
        let rows = algebra
            .defining_ideal()
            .generators()
            .iter()
            .map(|generator| {
                (0..ring.variables().len())
                    .map(|variable| {
                        generator
                            .derivative(variable)
                            .and_then(|derivative| derivative.evaluate(&point))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let evaluated_jacobian = if rows.is_empty() {
            Matrix::zeros(field.clone(), 0, ring.variables().len())
        } else {
            Matrix::from_rows(field.clone(), rows)?
        };
        let jacobian_rank = evaluated_jacobian.rank()?;
        let embedding_dimension = ring.variables().len().checked_sub(jacobian_rank).ok_or(
            VerificationError::InvalidWitness {
                claim: "Jacobian rank is at most the ambient coordinate count",
            },
        )?;
        let target = FreeModule::new(field.clone(), ring.variables().len())
            .with_basis_labels(differential_labels(ring))?;
        let source = FreeModule::new(field.clone(), rows_len(&evaluated_jacobian));
        let presentation =
            ModuleHomomorphism::from_matrix(source, target, evaluated_jacobian.transpose())?;
        let result = Self {
            algebra,
            point,
            evaluated_jacobian,
            module: PresentedModule::cokernel(presentation),
            jacobian_rank,
            embedding_dimension,
        };
        #[cfg(debug_assertions)]
        result.verify()?;
        Ok(result)
    }

    /// Affine algebra whose point is being studied.
    pub fn algebra(&self) -> &AffineAlgebra<C> {
        &self.algebra
    }

    /// Explicit point coordinates in polynomial-variable order.
    pub fn point(&self) -> &[C::Element] {
        &self.point
    }

    /// Evaluated Jacobian, with equations as rows and variables as columns.
    pub fn evaluated_jacobian(&self) -> &Matrix<C> {
        &self.evaluated_jacobian
    }

    /// Presentation matrix `J(p)^T` whose cokernel is the cotangent fiber.
    pub fn presentation_matrix(&self) -> Matrix<C> {
        self.evaluated_jacobian.transpose()
    }

    /// Exact finite presentation of the cotangent vector space.
    pub fn module(&self) -> &PresentedModule<C> {
        &self.module
    }

    /// Rank of the evaluated Jacobian over the exact coefficient field.
    pub fn jacobian_rank(&self) -> usize {
        self.jacobian_rank
    }

    /// Number of affine coordinates before imposing differential relations.
    pub fn ambient_coordinate_count(&self) -> usize {
        self.algebra.base_ring().variables().len()
    }

    /// Exact dimension of the cotangent fiber over the coefficient field.
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    /// Whether the exact cotangent fiber has the supplied embedding dimension.
    ///
    /// This is only an equality test on the computed vector-space dimension.
    /// It does not claim smoothness or regularity: such a conclusion also needs
    /// independently established local dimension and the appropriate field
    /// hypotheses.
    pub fn has_embedding_dimension(&self, expected_embedding_dimension: usize) -> bool {
        self.embedding_dimension == expected_embedding_dimension
    }

    /// Generators `x_i-a_i` of the explicit point's maximal ideal.
    pub fn maximal_ideal_generators(&self) -> Result<Vec<Polynomial<C>>, PolynomialError> {
        let ring = self.algebra.base_ring();
        ring.generators()?
            .into_iter()
            .zip(&self.point)
            .map(|(variable, coordinate)| {
                let constant = ring.constant(coordinate.clone())?;
                ring.sub(&variable, &constant)
                    .map_err(PolynomialError::from)
            })
            .collect()
    }

    /// Re-evaluate the point and Jacobian and compare the exact presentation.
    pub fn verify(&self) -> Result<(), KahlerDifferentialError> {
        let expected = Self::new_unchecked_replay(self.algebra.clone(), self.point.clone())?;
        if self.evaluated_jacobian != expected.evaluated_jacobian
            || self.module != expected.module
            || self.jacobian_rank != expected.jacobian_rank
            || self.embedding_dimension != expected.embedding_dimension
        {
            return Err(VerificationError::InvalidWitness {
                claim: "cotangent-space Jacobian and dimension replay",
            }
            .into());
        }
        Ok(())
    }

    fn new_unchecked_replay(
        algebra: AffineAlgebra<C>,
        point: Vec<C::Element>,
    ) -> Result<Self, KahlerDifferentialError> {
        validate_point(&algebra, &point)?;
        let ring = algebra.base_ring();
        let field = ring.coefficient_ring();
        let rows = algebra
            .defining_ideal()
            .generators()
            .iter()
            .map(|generator| {
                (0..ring.variables().len())
                    .map(|variable| {
                        generator
                            .derivative(variable)
                            .and_then(|derivative| derivative.evaluate(&point))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let evaluated_jacobian = if rows.is_empty() {
            Matrix::zeros(field.clone(), 0, ring.variables().len())
        } else {
            Matrix::from_rows(field.clone(), rows)?
        };
        let jacobian_rank = evaluated_jacobian.rank()?;
        let embedding_dimension = ring.variables().len() - jacobian_rank;
        let target = FreeModule::new(field.clone(), ring.variables().len())
            .with_basis_labels(differential_labels(ring))?;
        let source = FreeModule::new(field.clone(), rows_len(&evaluated_jacobian));
        let presentation =
            ModuleHomomorphism::from_matrix(source, target, evaluated_jacobian.transpose())?;
        Ok(Self {
            algebra,
            point,
            evaluated_jacobian,
            module: PresentedModule::cokernel(presentation),
            jacobian_rank,
            embedding_dimension,
        })
    }

    /// Serialize the algebra and point; deserialization exactly reevaluates
    /// the Jacobian, rank, and presentation.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        C: Serialize,
        C::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and recheck an explicit point and its cotangent fiber.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<C: Field> fmt::Debug for CotangentSpace<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CotangentSpace")
            .field("ambient_coordinates", &self.ambient_coordinate_count())
            .field("embedding_dimension", &self.embedding_dimension)
            .finish_non_exhaustive()
    }
}

impl<C: Field> SemanticType for CotangentSpace<C> {
    const TYPE_ID: &'static str = "noether/commalg/CotangentSpace@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut bytes = self.algebra.canonical_bytes()?;
        for coordinate in &self.point {
            append_component(&mut bytes, &coordinate.canonical_bytes()?)?;
        }
        Ok(bytes)
    }

    fn describe() -> &'static TypeDescription {
        &COTANGENT_SPACE_DESCRIPTION
    }
}

impl<C: Field> PolynomialRing<C> {
    /// Kähler differentials of this polynomial algebra over its coefficient
    /// field. Internally this is the exact zero-ideal quotient presentation.
    pub fn kahler_differentials(&self) -> Result<KahlerDifferentials<C>, KahlerDifferentialError> {
        let ideal = self.ideal(Vec::<Polynomial<C>>::new())?;
        let algebra = self.quotient(&ideal)?;
        KahlerDifferentials::new(algebra)
    }

    /// Cotangent space of this polynomial algebra at an explicit field point.
    pub fn cotangent_space_at(
        &self,
        point: Vec<C::Element>,
    ) -> Result<CotangentSpace<C>, KahlerDifferentialError> {
        self.kahler_differentials()?.cotangent_space_at(point)
    }
}

impl<C: Field> QuotientRing<PolynomialRing<C>> {
    /// Construct the exact module `Omega_{A/C}` for this affine algebra.
    pub fn kahler_differentials(&self) -> Result<KahlerDifferentials<C>, KahlerDifferentialError> {
        KahlerDifferentials::new(self.clone())
    }

    /// Cotangent fiber at one explicit rational or finite-field point.
    pub fn cotangent_space_at(
        &self,
        point: Vec<C::Element>,
    ) -> Result<CotangentSpace<C>, KahlerDifferentialError> {
        CotangentSpace::new(self.clone(), point)
    }

    /// Construct the complete-intersection cotangent complex using the
    /// canonical reduced ideal generators.
    pub fn cotangent_complex(
        &self,
    ) -> Result<CotangentComplexPresentation<C>, KahlerDifferentialError> {
        self.kahler_differentials()?.cotangent_complex()
    }

    /// Construct the complete-intersection cotangent complex from an explicit
    /// chosen equation list.
    ///
    /// The equations are first proved to generate this algebra's canonical
    /// ideal. Redundant lists then fail the regular-sequence count check rather
    /// than producing a falsely free conormal term.
    pub fn cotangent_complex_from_generators(
        &self,
        generators: &[Polynomial<C>],
    ) -> Result<CotangentComplexPresentation<C>, KahlerDifferentialError> {
        let evidence = build_conormal_evidence(self, generators)?;
        CotangentComplexPresentation::from_evidence(evidence)
    }

    /// Present `I/I^2` and its conormal map using an explicit equation list.
    ///
    /// Unlike the complete-intersection constructor this accepts redundant
    /// generators: their exact syzygies remain visible in the returned
    /// conormal module. The equations must generate this algebra's defining
    /// ideal exactly.
    pub fn conormal_sequence_from_generators(
        &self,
        generators: &[Polynomial<C>],
    ) -> Result<ConormalSequenceEvidence<C>, KahlerDifferentialError> {
        build_conormal_evidence(self, generators)
    }
}

fn build_conormal_evidence<C: Field>(
    algebra: &AffineAlgebra<C>,
    chosen_generators: &[Polynomial<C>],
) -> Result<ConormalSequenceEvidence<C>, KahlerDifferentialError> {
    let ring = algebra.base_ring();
    if chosen_generators
        .iter()
        .any(|generator| generator.ring() != ring)
    {
        return Err(ContextMismatchError {
            expected: ring.to_string(),
            actual: "chosen conormal equation from another polynomial context".to_owned(),
        }
        .into());
    }
    if ring.ideal(chosen_generators.iter().cloned())? != *algebra.defining_ideal() {
        return Err(KahlerDifferentialError::GeneratorsDoNotPresentAlgebra);
    }

    let ideal_source = ideal_generator_module(ring, chosen_generators)?;
    let ideal_target = if algebra.is_graded() {
        FreeModule::new(ring.clone(), 1).with_degree_shifts(
            ring.grading().rank(),
            vec![Degree::zero(ring.grading().rank())],
        )?
    } else {
        FreeModule::new(ring.clone(), 1)
    };
    let ideal_matrix = Matrix::from_rows(ring.clone(), [chosen_generators.to_vec()])?;
    let ideal_map =
        ModuleHomomorphism::from_matrix(ideal_source, ideal_target.clone(), ideal_matrix)?;
    let ideal_submodule = Submodule::from_generator_map(ideal_target, ideal_map)?;
    let syzygies = ideal_submodule
        .groebner_basis(module_order_for_ring(ring))?
        .syzygies()?;
    let conormal_relation_map = project_module_map(syzygies.generator_map(), algebra)?;
    let conormal_module = PresentedModule::cokernel(conormal_relation_map);

    let ambient_free = ambient_differential_module(algebra)?;
    let ambient_differentials = PresentedModule::free(ambient_free.clone())?;
    let jacobian = quotient_jacobian(algebra, chosen_generators)?;
    let conormal_generator_lift = ModuleHomomorphism::from_matrix(
        conormal_module.generator_module().clone(),
        ambient_free,
        jacobian.transpose(),
    )?;
    let relation_lift = ModuleHomomorphism::zero(
        conormal_module.relation_module().clone(),
        ambient_differentials.relation_module().clone(),
    )?;
    let conormal_map = PresentedModuleHomomorphism::from_lifts(
        conormal_module.clone(),
        ambient_differentials.clone(),
        conormal_generator_lift,
        relation_lift,
    )?;

    Ok(ConormalSequenceEvidence {
        algebra: algebra.clone(),
        chosen_generators: chosen_generators.to_vec(),
        conormal_module,
        ambient_differentials,
        conormal_map,
        jacobian,
    })
}

fn ideal_generator_module<C: Field>(
    ring: &PolynomialRing<C>,
    generators: &[Polynomial<C>],
) -> Result<FreeModule<PolynomialRing<C>>, KahlerDifferentialError> {
    let labels = (0..generators.len())
        .map(|index| format!("relation_{index}"))
        .collect();
    let module = FreeModule::new(ring.clone(), generators.len()).with_basis_labels(labels)?;
    if generators
        .iter()
        .all(|generator| generator.multihomogeneous_degree().ok().flatten().is_some())
    {
        let shifts = generators
            .iter()
            .map(|generator| {
                generator
                    .multihomogeneous_degree()
                    .map_err(KahlerDifferentialError::from)
                    .and_then(|degree| {
                        degree.map(|degree| degree.neg()).ok_or_else(|| {
                            VerificationError::InvalidWitness {
                                claim: "a homogeneous equation has a degree",
                            }
                            .into()
                        })
                    })
            })
            .collect::<Result<Vec<_>, KahlerDifferentialError>>()?;
        Ok(module.with_degree_shifts(ring.grading().rank(), shifts)?)
    } else {
        Ok(module)
    }
}

fn ambient_differential_module<C: Field>(
    algebra: &AffineAlgebra<C>,
) -> Result<FreeModule<AffineAlgebra<C>>, KahlerDifferentialError> {
    let ring = algebra.base_ring();
    let module = FreeModule::new(algebra.clone(), ring.variables().len())
        .with_basis_labels(differential_labels(ring))?;
    if algebra.is_graded() {
        Ok(module.with_degree_shifts(
            ring.grading().rank(),
            ring.grading()
                .variable_degrees()
                .iter()
                .map(Degree::neg)
                .collect(),
        )?)
    } else {
        Ok(module)
    }
}

fn differential_labels<C: Field>(ring: &PolynomialRing<C>) -> Vec<String> {
    ring.variables()
        .iter()
        .map(|variable| format!("d{variable}"))
        .collect()
}

fn quotient_jacobian<C: Field>(
    algebra: &AffineAlgebra<C>,
    generators: &[Polynomial<C>],
) -> Result<Matrix<AffineAlgebra<C>>, KahlerDifferentialError> {
    let variable_count = algebra.base_ring().variables().len();
    if generators.is_empty() {
        return Ok(Matrix::zeros(algebra.clone(), 0, variable_count));
    }
    let rows = generators
        .iter()
        .map(|generator| {
            (0..variable_count)
                .map(|variable| {
                    generator
                        .derivative(variable)
                        .map_err(KahlerDifferentialError::from)
                        .and_then(|derivative| {
                            algebra
                                .element(&derivative)
                                .map_err(KahlerDifferentialError::from)
                        })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Matrix::from_rows(algebra.clone(), rows)?)
}

fn project_module_map<C: Field>(
    map: &ModuleHomomorphism<PolynomialRing<C>>,
    algebra: &AffineAlgebra<C>,
) -> Result<ModuleHomomorphism<AffineAlgebra<C>>, KahlerDifferentialError> {
    let source = project_free_module(map.source(), algebra)?;
    let target = project_free_module(map.target(), algebra)?;
    let matrix = map.matrix();
    let (rows, columns) = matrix.shape();
    let projected = if rows == 0 {
        Matrix::zeros(algebra.clone(), 0, columns)
    } else {
        Matrix::from_rows(
            algebra.clone(),
            matrix
                .rows()
                .map(|row| {
                    row.into_iter()
                        .map(|value| algebra.element(&value))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?,
        )?
    };
    Ok(ModuleHomomorphism::from_matrix(source, target, projected)?)
}

fn project_free_module<C: Field>(
    module: &FreeModule<PolynomialRing<C>>,
    algebra: &AffineAlgebra<C>,
) -> Result<FreeModule<AffineAlgebra<C>>, ModuleError> {
    FreeModule::with_basis_data(
        algebra.clone(),
        module.rank(),
        module.basis_labels().map(<[String]>::to_vec),
        module.grading_rank(),
        module.degree_shifts().map(<[Degree]>::to_vec),
    )
}

fn complete_intersection_hypothesis<C: Field>(
    algebra: &AffineAlgebra<C>,
    generator_count: usize,
) -> Result<CotangentComplexHypothesis, KahlerDifferentialError> {
    let dimension = algebra.dimension()?;
    if dimension < 0 {
        return Err(KahlerDifferentialError::ZeroAlgebra);
    }
    let ambient = i64::try_from(algebra.base_ring().variables().len())
        .map_err(|_| KahlerDifferentialError::DimensionOverflow)?;
    let codimension = ambient
        .checked_sub(dimension)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(KahlerDifferentialError::DimensionOverflow)?;
    if generator_count != codimension {
        return Err(KahlerDifferentialError::CompleteIntersectionRequired {
            generator_count,
            codimension: Some(codimension),
        });
    }
    let quotient_dimension =
        usize::try_from(dimension).map_err(|_| KahlerDifferentialError::DimensionOverflow)?;
    Ok(
        CotangentComplexHypothesis::ProvedAffineCompleteIntersection {
            ambient_dimension: algebra.base_ring().variables().len(),
            quotient_dimension,
            codimension,
            equation_count: generator_count,
        },
    )
}

fn validate_point<C: Field>(
    algebra: &AffineAlgebra<C>,
    point: &[C::Element],
) -> Result<(), KahlerDifferentialError> {
    let ring = algebra.base_ring();
    if point.len() != ring.variables().len() {
        return Err(KahlerDifferentialError::PointArity {
            expected: ring.variables().len(),
            actual: point.len(),
        });
    }
    if let Some(coordinate) = point
        .iter()
        .position(|value| !ring.coefficient_ring().contains(value))
    {
        return Err(KahlerDifferentialError::PointCoordinateOutsideField { coordinate });
    }
    for (relation, generator) in algebra.defining_ideal().generators().iter().enumerate() {
        if !ring.coefficient_ring().is_zero(&generator.evaluate(point)?) {
            return Err(KahlerDifferentialError::PointNotOnAlgebra { relation });
        }
    }
    Ok(())
}

fn module_order<C: Field>(algebra: &AffineAlgebra<C>) -> ModuleMonomialOrder {
    module_order_for_ring(algebra.base_ring())
}

fn module_order_for_ring<C: Field>(ring: &PolynomialRing<C>) -> ModuleMonomialOrder {
    ModuleMonomialOrder::TermOverPosition(ring.term_order())
}

fn rows_len<R: Ring>(matrix: &Matrix<R>) -> usize {
    matrix.shape().0
}

fn context_mismatch<C: Field>(
    operation: &'static str,
    expected: &AffineAlgebra<C>,
    actual: &AffineAlgebra<C>,
) -> ContextMismatchError {
    ContextMismatchError {
        expected: format!("{operation}: {expected}"),
        actual: actual.to_string(),
    }
}

fn append_component(output: &mut Vec<u8>, component: &[u8]) -> Result<(), NoetherError> {
    append_usize(output, component.len())?;
    output.extend_from_slice(component);
    Ok(())
}

fn append_usize(output: &mut Vec<u8>, value: usize) -> Result<(), NoetherError> {
    let value = u64::try_from(value)
        .map_err(|_| NoetherError::Serialization("component length exceeds u64".to_owned()))?;
    output.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "C: Field + Deserialize<'de>, C::Element: Deserialize<'de>")
)]
struct ConormalSequenceWire<C: Field> {
    #[serde(rename = "type")]
    type_id: String,
    algebra: AffineAlgebra<C>,
    chosen_generators: Vec<Polynomial<C>>,
}

impl<C> Serialize for ConormalSequenceEvidence<C>
where
    C: Field + Serialize,
    C::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ConormalSequenceEvidence", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("algebra", &self.algebra)?;
        state.serialize_field("chosen_generators", &self.chosen_generators)?;
        state.end()
    }
}

impl<'de, C> Deserialize<'de> for ConormalSequenceEvidence<C>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ConormalSequenceWire::<C>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        build_conormal_evidence(&wire.algebra, &wire.chosen_generators)
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "C: Field + Deserialize<'de>, C::Element: Deserialize<'de>")
)]
struct KahlerDifferentialsWire<C: Field> {
    #[serde(rename = "type")]
    type_id: String,
    algebra: AffineAlgebra<C>,
}

impl<C> Serialize for KahlerDifferentials<C>
where
    C: Field + Serialize,
    C::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("KahlerDifferentials", 2)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("algebra", &self.algebra)?;
        state.end()
    }
}

impl<'de, C> Deserialize<'de> for KahlerDifferentials<C>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = KahlerDifferentialsWire::<C>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.algebra).map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "C: Field + Deserialize<'de>, C::Element: Deserialize<'de>")
)]
struct KahlerDifferentialWire<C: Field> {
    #[serde(rename = "type")]
    type_id: String,
    differentials: KahlerDifferentials<C>,
    representative: ModuleElement<AffineAlgebra<C>>,
}

impl<C> Serialize for KahlerDifferential<C>
where
    C: Field + Serialize,
    C::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("KahlerDifferential", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("differentials", &self.differentials)?;
        state.serialize_field("representative", &self.representative)?;
        state.end()
    }
}

impl<'de, C> Deserialize<'de> for KahlerDifferential<C>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = KahlerDifferentialWire::<C>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let canonical = wire
            .differentials
            .element(wire.representative.clone())
            .map_err(serde::de::Error::custom)?;
        if canonical.representative != wire.representative {
            return Err(serde::de::Error::custom(
                "Kahler differential representative is not canonical",
            ));
        }
        Ok(canonical)
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "C: Field + Deserialize<'de>, C::Element: Deserialize<'de>")
)]
struct CotangentComplexWire<C: Field> {
    #[serde(rename = "type")]
    type_id: String,
    evidence: ConormalSequenceEvidence<C>,
}

impl<C> Serialize for CotangentComplexPresentation<C>
where
    C: Field + Serialize,
    C::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("CotangentComplexPresentation", 2)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("evidence", &self.evidence)?;
        state.end()
    }
}

impl<'de, C> Deserialize<'de> for CotangentComplexPresentation<C>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CotangentComplexWire::<C>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::from_evidence(wire.evidence).map_err(serde::de::Error::custom)
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "C: Field + Deserialize<'de>, C::Element: Deserialize<'de>")
)]
struct CotangentSpaceWire<C: Field> {
    #[serde(rename = "type")]
    type_id: String,
    algebra: AffineAlgebra<C>,
    point: Vec<C::Element>,
}

impl<C> Serialize for CotangentSpace<C>
where
    C: Field + Serialize,
    C::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("CotangentSpace", 3)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("algebra", &self.algebra)?;
        state.serialize_field("point", &self.point)?;
        state.end()
    }
}

impl<'de, C> Deserialize<'de> for CotangentSpace<C>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CotangentSpaceWire::<C>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        Self::new(wire.algebra, wire.point).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::numthy::FiniteField;
    use crate::{Rational, RationalField};

    use super::*;
    use crate::commalg::{FreeAbelianGrading, MonomialOrder};

    fn property_config(default_cases: u32) -> ProptestConfig {
        let configured = ProptestConfig::default();
        if std::env::var_os("PROPTEST_CASES").is_some() {
            configured
        } else {
            ProptestConfig {
                cases: default_cases,
                ..configured
            }
        }
    }

    fn rational_polynomial_ring(variables: &[&str]) -> PolynomialRing<RationalField> {
        PolynomialRing::new(
            RationalField,
            variables.iter().copied(),
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap()
    }

    fn univariate(
        ring: &PolynomialRing<RationalField>,
        coefficients: &[i16],
    ) -> Polynomial<RationalField> {
        ring.from_u64_terms(
            coefficients
                .iter()
                .enumerate()
                .filter(|(_, coefficient)| **coefficient != 0)
                .map(|(power, coefficient)| {
                    (
                        vec![u64::try_from(power).unwrap()],
                        Rational::from(*coefficient),
                    )
                }),
        )
        .unwrap()
    }

    #[test]
    fn polynomial_algebra_has_free_differentials_and_universal_derivation() {
        let ring = rational_polynomial_ring(&["x"]);
        let x = ring.generator(0).unwrap();
        let omega = ring.kahler_differentials().unwrap();
        assert_eq!(omega.module().generator_count(), 1);
        assert_eq!(omega.module().relation_count(), 0);
        assert_eq!(
            omega.module().generator_module().basis_labels().unwrap(),
            ["dx"]
        );

        let polynomial = (&x.pow(3).unwrap() + &ring.constant(Rational::from(2)).unwrap()).unwrap();
        let differential = omega.differentiate_polynomial(&polynomial).unwrap();
        let coefficient = differential.coordinates().coordinates().get(0).unwrap();
        assert_eq!(
            coefficient.lift(),
            &x.pow(2).unwrap().scalar_mul(&Rational::from(3)).unwrap()
        );
        assert!(omega
            .differentiate_polynomial(&ring.constant(Rational::from(7)).unwrap())
            .unwrap()
            .is_zero());
        omega.verify().unwrap();
    }

    #[test]
    fn dual_numbers_retain_the_conormal_relation_and_complex() {
        let ring = rational_polynomial_ring(&["x"]);
        let x = ring.generator(0).unwrap();
        let equation = x.pow(2).unwrap();
        let ideal = ring.ideal([equation]).unwrap();
        let algebra = ring.quotient(&ideal).unwrap();
        let omega = algebra.kahler_differentials().unwrap();

        assert_eq!(omega.module().generator_count(), 1);
        assert_eq!(omega.module().relation_count(), 1);
        let jacobian_entry = omega.conormal_evidence().jacobian().get(0, 0).unwrap();
        assert_eq!(
            jacobian_entry.lift(),
            &x.scalar_mul(&Rational::from(2)).unwrap()
        );
        assert!(omega
            .differentiate_polynomial(&x.pow(2).unwrap())
            .unwrap()
            .is_zero());
        assert!(omega.differentiate(&algebra.zero()).unwrap().is_zero());

        let complex = omega.cotangent_complex().unwrap();
        assert_eq!(complex.codimension(), 1);
        assert_eq!(
            complex.amplitude(),
            CotangentComplexAmplitude::MINUS_ONE_TO_ZERO
        );
        assert_eq!(complex.amplitude().lowest_degree(), -1);
        assert_eq!(complex.amplitude().highest_degree(), 0);
        let hypothesis = complex.hypothesis();
        assert_eq!(hypothesis.ambient_dimension(), 1);
        assert_eq!(hypothesis.quotient_dimension(), 0);
        assert_eq!(hypothesis.codimension(), 1);
        assert_eq!(hypothesis.equation_count(), 1);
        assert_eq!(complex.conormal_module().generator_count(), 1);
        complex.verify().unwrap();
    }

    #[test]
    fn smooth_and_singular_cusp_points_have_different_embedding_dimensions() {
        let ring = rational_polynomial_ring(&["x", "y"]);
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let cusp = (&y.pow(2).unwrap() - &x.pow(3).unwrap()).unwrap();
        let ideal = ring.ideal([cusp]).unwrap();
        let algebra = ring.quotient(&ideal).unwrap();

        let singular = algebra
            .cotangent_space_at(vec![Rational::from(0), Rational::from(0)])
            .unwrap();
        assert_eq!(singular.ambient_coordinate_count(), 2);
        assert_eq!(singular.jacobian_rank(), 0);
        assert_eq!(singular.embedding_dimension(), 2);
        assert!(!singular.has_embedding_dimension(1));

        let smooth = algebra
            .cotangent_space_at(vec![Rational::from(1), Rational::from(1)])
            .unwrap();
        assert_eq!(smooth.jacobian_rank(), 1);
        assert_eq!(smooth.embedding_dimension(), 1);
        assert!(smooth.has_embedding_dimension(1));
        for generator in smooth.maximal_ideal_generators().unwrap() {
            assert_eq!(
                generator.evaluate(smooth.point()).unwrap(),
                Rational::from(0)
            );
        }
        smooth.verify().unwrap();
    }

    #[test]
    fn characteristic_two_keeps_the_vanishing_derivative_of_x_squared() {
        let field = FiniteField::prime(2_u8).unwrap();
        let ring = PolynomialRing::new(
            field.clone(),
            ["x"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let x = ring.generator(0).unwrap();
        let ideal = ring.ideal([x.pow(2).unwrap()]).unwrap();
        let algebra = ring.quotient(&ideal).unwrap();
        let omega = algebra.kahler_differentials().unwrap();
        assert!(omega
            .conormal_evidence()
            .jacobian()
            .get(0, 0)
            .unwrap()
            .is_zero());
        assert_eq!(
            algebra
                .cotangent_space_at(vec![field.element(0)])
                .unwrap()
                .embedding_dimension(),
            1
        );
        omega.verify().unwrap();
    }

    #[test]
    fn a_reduced_zero_dimensional_point_has_zero_cotangent_space() {
        let ring = rational_polynomial_ring(&["x"]);
        let x = ring.generator(0).unwrap();
        let algebra = ring.quotient(&ring.ideal([x]).unwrap()).unwrap();
        let fiber = algebra.cotangent_space_at(vec![Rational::from(0)]).unwrap();
        assert_eq!(fiber.ambient_coordinate_count(), 1);
        assert_eq!(fiber.jacobian_rank(), 1);
        assert_eq!(fiber.embedding_dimension(), 0);
        assert!(fiber.has_embedding_dimension(0));
        fiber.verify().unwrap();
    }

    #[test]
    fn weighted_gradings_put_differentials_and_relations_in_exact_degrees() {
        let grading = FreeAbelianGrading::new(
            1,
            vec![Degree::new(vec![1.into()]), Degree::new(vec![2.into()])],
        )
        .unwrap();
        let ring = PolynomialRing::with_grading(
            RationalField,
            ["x", "y"],
            MonomialOrder::GradedReverseLexicographic,
            grading,
        )
        .unwrap();
        let x = ring.generator(0).unwrap();
        let y = ring.generator(1).unwrap();
        let relation = (&x.pow(2).unwrap() + &y).unwrap();
        let algebra = ring.quotient(&ring.ideal([relation]).unwrap()).unwrap();
        let omega = algebra.kahler_differentials().unwrap();

        assert_eq!(
            omega.module().generator_module().degree_shifts().unwrap(),
            [
                Degree::new(vec![(-1).into()]),
                Degree::new(vec![(-2).into()]),
            ]
        );
        assert_eq!(
            omega
                .conormal_evidence()
                .conormal_module()
                .generator_module()
                .degree_shifts()
                .unwrap(),
            [Degree::new(vec![(-2).into()])]
        );
        omega.verify().unwrap();
    }

    #[test]
    fn redundant_equations_fail_closed_for_the_cotangent_complex() {
        let ring = rational_polynomial_ring(&["x"]);
        let x = ring.generator(0).unwrap();
        let square = x.pow(2).unwrap();
        let cube = x.pow(3).unwrap();
        let ideal = ring.ideal([square.clone(), cube.clone()]).unwrap();
        let algebra = ring.quotient(&ideal).unwrap();
        let conormal = algebra
            .conormal_sequence_from_generators(&[square.clone(), cube.clone()])
            .unwrap();
        assert_eq!(conormal.conormal_module().generator_count(), 2);
        assert!(conormal.conormal_module().relation_count() >= 1);
        conormal.verify().unwrap();
        let error = algebra
            .cotangent_complex_from_generators(&[square, cube])
            .unwrap_err();
        assert_eq!(
            error,
            KahlerDifferentialError::CompleteIntersectionRequired {
                generator_count: 2,
                codimension: Some(1),
            }
        );
    }

    #[test]
    fn points_and_contexts_are_checked_before_differentiation() {
        let ring = rational_polynomial_ring(&["x"]);
        let x = ring.generator(0).unwrap();
        let ideal = ring.ideal([x.clone()]).unwrap();
        let algebra = ring.quotient(&ideal).unwrap();
        assert_eq!(
            algebra
                .cotangent_space_at(vec![Rational::from(1)])
                .unwrap_err(),
            KahlerDifferentialError::PointNotOnAlgebra { relation: 0 }
        );
        assert_eq!(
            algebra.cotangent_space_at(Vec::new()).unwrap_err(),
            KahlerDifferentialError::PointArity {
                expected: 1,
                actual: 0,
            }
        );

        let other = rational_polynomial_ring(&["t"]);
        let omega = algebra.kahler_differentials().unwrap();
        assert!(matches!(
            omega.differentiate_polynomial(&other.generator(0).unwrap()),
            Err(KahlerDifferentialError::ContextMismatch(_))
        ));
        assert_eq!(
            algebra.conormal_sequence_from_generators(&[]).unwrap_err(),
            KahlerDifferentialError::GeneratorsDoNotPresentAlgebra
        );
    }

    #[test]
    fn typed_json_rebuilds_and_reverifies_every_derived_object() {
        let ring = rational_polynomial_ring(&["x"]);
        let x = ring.generator(0).unwrap();
        let algebra = ring
            .quotient(&ring.ideal([x.pow(2).unwrap()]).unwrap())
            .unwrap();
        let omega = algebra.kahler_differentials().unwrap();
        let dx = omega.differentiate(&algebra.element(&x).unwrap()).unwrap();
        let fiber = algebra.cotangent_space_at(vec![Rational::from(0)]).unwrap();
        let complex = omega.cotangent_complex().unwrap();

        assert_eq!(
            ConormalSequenceEvidence::<RationalField>::from_json(
                &omega.conormal_evidence().to_json().unwrap()
            )
            .unwrap(),
            *omega.conormal_evidence()
        );
        assert_eq!(
            KahlerDifferentials::<RationalField>::from_json(&omega.to_json().unwrap()).unwrap(),
            omega
        );
        assert_eq!(
            KahlerDifferential::<RationalField>::from_json(&dx.to_json().unwrap()).unwrap(),
            dx
        );
        assert_eq!(
            CotangentSpace::<RationalField>::from_json(&fiber.to_json().unwrap()).unwrap(),
            fiber
        );
        assert_eq!(
            CotangentComplexPresentation::<RationalField>::from_json(&complex.to_json().unwrap())
                .unwrap(),
            complex
        );

        let forged_amplitude = complex.to_json().unwrap().replacen(
            '{',
            "{\"amplitude\":{\"lowest_degree\":-99,\"highest_degree\":99},",
            1,
        );
        assert!(
            CotangentComplexPresentation::<RationalField>::from_json(&forged_amplitude).is_err()
        );
        let forged_hypothesis = complex.to_json().unwrap().replacen(
            '{',
            "{\"hypothesis\":{\"ProvedAffineCompleteIntersection\":{\"ambient_dimension\":0,\"quotient_dimension\":99,\"codimension\":0,\"equation_count\":0}},",
            1,
        );
        assert!(
            CotangentComplexPresentation::<RationalField>::from_json(&forged_hypothesis).is_err()
        );

        let tampered = omega.to_json().unwrap().replace(
            KahlerDifferentials::<RationalField>::TYPE_ID,
            "wrong/type@1",
        );
        assert!(KahlerDifferentials::<RationalField>::from_json(&tampered).is_err());
    }

    #[test]
    fn affine_coordinate_translation_obeys_the_chain_rule() {
        let ring = rational_polynomial_ring(&["x"]);
        let x = ring.generator(0).unwrap();
        let one = ring.constant(Rational::from(1)).unwrap();
        let translated_coordinate = (&x + &one).unwrap();
        let cubic = x.pow(3).unwrap();
        let translated_cubic = cubic
            .substitute(std::slice::from_ref(&translated_coordinate))
            .unwrap();
        let expected_coefficient = translated_coordinate
            .pow(2)
            .unwrap()
            .scalar_mul(&Rational::from(3))
            .unwrap();
        let omega = ring.kahler_differentials().unwrap();
        let actual = omega
            .differentiate_polynomial(&translated_cubic)
            .unwrap()
            .coordinates()
            .coordinates()
            .get(0)
            .unwrap();
        assert_eq!(actual.lift(), &expected_coefficient);
    }

    proptest! {
        #[test]
        fn universal_derivation_is_additive_and_satisfies_leibniz(
            left in prop::collection::vec(-5_i16..=5, 0..5),
            right in prop::collection::vec(-5_i16..=5, 0..5),
        ) {
            let ring = rational_polynomial_ring(&["x"]);
            let algebra = ring
                .quotient(&ring.ideal(Vec::<Polynomial<RationalField>>::new()).unwrap())
                .unwrap();
            let omega = algebra.kahler_differentials().unwrap();
            let f = univariate(&ring, &left);
            let g = univariate(&ring, &right);

            let sum = (&f + &g).unwrap();
            let product = (&f * &g).unwrap();
            let df = omega.differentiate_polynomial(&f).unwrap();
            let dg = omega.differentiate_polynomial(&g).unwrap();
            prop_assert_eq!(
                omega.differentiate_polynomial(&sum).unwrap(),
                df.add(&dg).unwrap()
            );

            let f_scalar = algebra.element(&f).unwrap();
            let g_scalar = algebra.element(&g).unwrap();
            let product_rule = dg
                .scalar_mul(&f_scalar)
                .unwrap()
                .add(&df.scalar_mul(&g_scalar).unwrap())
                .unwrap();
            prop_assert_eq!(omega.differentiate_polynomial(&product).unwrap(), product_rule);
        }
    }

    proptest! {
        #![proptest_config(property_config(16))]

        #[test]
        fn invertible_changes_of_equation_generators_preserve_the_conormal_claim(
            coefficient in -8_i8..=8,
        ) {
            let ring = rational_polynomial_ring(&["x", "y"]);
            let x = ring.generator(0).unwrap();
            let y = ring.generator(1).unwrap();
            let square = x.pow(2).unwrap();
            let ideal = ring.ideal([square.clone(), y.clone()]).unwrap();
            let algebra = ring.quotient(&ideal).unwrap();
            let changed_first = (&square
                + &y.scalar_mul(&Rational::from(coefficient)).unwrap())
                .unwrap();
            let changed = [changed_first, y];

            let evidence = algebra
                .conormal_sequence_from_generators(&changed)
                .unwrap();
            evidence.verify().unwrap();
            let complex = algebra
                .cotangent_complex_from_generators(&changed)
                .unwrap();
            prop_assert_eq!(complex.codimension(), 2);
            complex.verify().unwrap();

            let omega = algebra.kahler_differentials().unwrap();
            for relation in &changed {
                prop_assert!(omega
                    .differentiate_polynomial(relation)
                    .unwrap()
                    .is_zero());
            }
        }
    }
}
