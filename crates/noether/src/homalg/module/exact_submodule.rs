// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical submodules over `Z` and arbitrary `Z/nZ`.

use core::any::{Any, TypeId};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::ResidueRing;
use crate::homalg::FinitelyGeneratedAbelianGroup;
use crate::linalg::{
    canonical_congruence_columns, canonical_integer_columns, CongruenceSystemResult,
    IntegerLinearSystemResult, Matrix, Vector,
};
use crate::{
    Integer, IntegerRing, NoetherError, Ring, SemanticType, TypeDescription, VerificationError, Z,
};

use super::{
    FreeModule, ModuleElement, ModuleError, ModuleHomomorphism, PresentedModule, Submodule,
};

static SUBMODULE_MEMBERSHIP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/SubmoduleMembership@1",
    rust_path: "noether::homalg::SubmoduleMembership",
    python_path: Some("noether.homalg.SubmoduleMembership"),
};

static KERNEL_SUBMODULE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/KernelSubmodule@1",
    rust_path: "noether::homalg::KernelSubmodule",
    python_path: Some("noether.homalg.KernelSubmodule"),
};

static IMAGE_SUBMODULE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/ImageSubmodule@1",
    rust_path: "noether::homalg::ImageSubmodule",
    python_path: Some("noether.homalg.ImageSubmodule"),
};

/// A map-bound, replayable certificate for its complete exact kernel.
///
/// Verification checks that every retained generator maps to zero and then
/// independently recomputes the canonical kernel.  Thus this result proves
/// completeness, not only that the displayed generators lie in the kernel.
#[derive(Clone)]
pub struct KernelSubmodule<R: Ring> {
    homomorphism: ModuleHomomorphism<R>,
    submodule: Submodule<R>,
}

/// A map-bound, replayable certificate for its complete exact image.
///
/// Verification checks the images of the source basis and independently
/// recomputes the canonical image submodule.
#[derive(Clone)]
pub struct ImageSubmodule<R: Ring> {
    homomorphism: ModuleHomomorphism<R>,
    submodule: Submodule<R>,
}

impl<R: Ring> KernelSubmodule<R> {
    /// Homomorphism whose kernel is certified.
    pub fn homomorphism(&self) -> &ModuleHomomorphism<R> {
        &self.homomorphism
    }

    /// Canonical submodule of the source proved to be the full kernel.
    pub fn submodule(&self) -> &Submodule<R> {
        &self.submodule
    }

    /// Serialize the map together with its canonical kernel evidence.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: Ring> ImageSubmodule<R> {
    /// Homomorphism whose image is certified.
    pub fn homomorphism(&self) -> &ModuleHomomorphism<R> {
        &self.homomorphism
    }

    /// Canonical submodule of the target proved to be the full image.
    pub fn submodule(&self) -> &Submodule<R> {
        &self.submodule
    }

    /// Serialize the map together with its canonical image evidence.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

macro_rules! serialize_bound_submodule {
    ($kind:ident) => {
        impl<R> Serialize for $kind<R>
        where
            R: Ring + Serialize,
            R::Element: Serialize,
        {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let mut state = serializer.serialize_struct(stringify!($kind), 3)?;
                state.serialize_field("type", Self::TYPE_ID)?;
                state.serialize_field("homomorphism", &self.homomorphism)?;
                state.serialize_field("submodule", &self.submodule)?;
                state.end()
            }
        }

        impl<R: Ring> SemanticType for $kind<R> {
            const TYPE_ID: &'static str = concat!("noether/homalg/", stringify!($kind), "@1");

            fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
                let mut output = Vec::new();
                for component in [
                    self.homomorphism.canonical_bytes()?,
                    self.submodule.canonical_bytes()?,
                ] {
                    let length = u64::try_from(component.len()).map_err(|_| {
                        NoetherError::Serialization("canonical component too large".into())
                    })?;
                    output.extend_from_slice(&length.to_be_bytes());
                    output.extend_from_slice(&component);
                }
                Ok(output)
            }

            fn describe() -> &'static TypeDescription {
                match stringify!($kind) {
                    "KernelSubmodule" => &KERNEL_SUBMODULE_DESCRIPTION,
                    _ => &IMAGE_SUBMODULE_DESCRIPTION,
                }
            }
        }
    };
}

serialize_bound_submodule!(KernelSubmodule);
serialize_bound_submodule!(ImageSubmodule);

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct BoundSubmoduleWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    homomorphism: ModuleHomomorphism<R>,
    submodule: Submodule<R>,
}

/// Replayable coefficients proving that an element belongs to a submodule.
///
/// The coefficient vector is expressed in the canonical generator map retained
/// by [`Submodule`], so `generator_map * coefficients = element` can be checked
/// without trusting the membership computation.
#[derive(Clone)]
pub struct SubmoduleMembership<R: Ring> {
    submodule: Submodule<R>,
    element: ModuleElement<R>,
    coefficients: Vector<R>,
}

impl<R: Ring> SubmoduleMembership<R> {
    /// Submodule whose membership is certified.
    pub fn submodule(&self) -> &Submodule<R> {
        &self.submodule
    }

    /// Ambient element certified to lie in the submodule.
    pub fn element(&self) -> &ModuleElement<R> {
        &self.element
    }

    /// Coefficients in the retained canonical generators.
    pub fn coefficients(&self) -> &Vector<R> {
        &self.coefficients
    }

    /// Reconstruct the represented element from the canonical generator map.
    pub fn verify(&self) -> Result<(), VerificationError> {
        if self.element.module() != self.submodule.ambient()
            || self.coefficients.ring() != self.submodule.ambient().ring()
            || self.coefficients.dimension() != self.submodule.generator_count()
        {
            return Err(VerificationError::InvalidWitness {
                claim: "submodule membership witness has compatible modules and dimensions",
            });
        }
        let coefficient_element = ModuleElement::from_coordinates(
            self.submodule.generator_map().source().clone(),
            self.coefficients.clone(),
        )
        .map_err(|_| VerificationError::InvalidWitness {
            claim: "submodule membership coefficients define a source element",
        })?;
        let reconstructed = self
            .submodule
            .generator_map()
            .apply(&coefficient_element)
            .map_err(|_| VerificationError::InvalidWitness {
                claim: "submodule membership coefficients can be replayed",
            })?;
        if reconstructed != self.element {
            return Err(VerificationError::InvalidWitness {
                claim: "submodule membership coefficients reconstruct the claimed element",
            });
        }
        Ok(())
    }

    /// Serialize this proof with exact context identity and replay data.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        R: Serialize,
        R::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and replay a membership proof.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        R: for<'de> Deserialize<'de>,
        R::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> Serialize for SubmoduleMembership<R>
where
    R: Ring + Serialize,
    R::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("SubmoduleMembership", 4)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("submodule", &self.submodule)?;
        state.serialize_field("element", &self.element)?;
        state.serialize_field("coefficients", &self.coefficients)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "R: Ring + Deserialize<'de>, R::Element: Deserialize<'de>")
)]
struct SubmoduleMembershipWire<R: Ring> {
    #[serde(rename = "type")]
    type_id: String,
    submodule: Submodule<R>,
    element: ModuleElement<R>,
    coefficients: Vector<R>,
}

impl<'de, R> Deserialize<'de> for SubmoduleMembership<R>
where
    R: Ring + Deserialize<'de>,
    R::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SubmoduleMembershipWire::<R>::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            submodule: wire.submodule,
            element: wire.element,
            coefficients: wire.coefficients,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<R: Ring> SemanticType for SubmoduleMembership<R> {
    const TYPE_ID: &'static str = "noether/homalg/SubmoduleMembership@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        for component in [
            self.submodule.canonical_bytes()?,
            self.element.canonical_bytes()?,
            self.coefficients.canonical_bytes()?,
        ] {
            let length = u64::try_from(component.len())
                .map_err(|_| NoetherError::Serialization("canonical component too large".into()))?;
            output.extend_from_slice(&length.to_be_bytes());
            output.extend_from_slice(&component);
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &SUBMODULE_MEMBERSHIP_DESCRIPTION
    }
}

pub(super) fn canonicalize_supported<R: Ring>(
    submodule: Submodule<R>,
) -> Result<Submodule<R>, ModuleError> {
    if TypeId::of::<R>() == TypeId::of::<IntegerRing>() {
        let concrete = *as_any_box(submodule)
            .downcast::<Submodule<IntegerRing>>()
            .expect("integer submodule type-id invariant");
        let canonical = canonicalize_integer(concrete)?;
        return Ok(*as_any_box(canonical)
            .downcast::<Submodule<R>>()
            .expect("integer submodule reverse type-id invariant"));
    }
    if TypeId::of::<R>() == TypeId::of::<ResidueRing>() {
        let concrete = *as_any_box(submodule)
            .downcast::<Submodule<ResidueRing>>()
            .expect("residue submodule type-id invariant");
        let canonical = canonicalize_congruence(concrete)?;
        return Ok(*as_any_box(canonical)
            .downcast::<Submodule<R>>()
            .expect("residue submodule reverse type-id invariant"));
    }
    Ok(submodule)
}

impl Submodule<IntegerRing> {
    /// Canonical ambient generators in column-Hermite convention.
    pub fn generators(&self) -> Vec<ModuleElement<IntegerRing>> {
        module_generators(self)
    }

    /// Decide exact membership in this integer submodule.
    pub fn contains(&self, element: &ModuleElement<IntegerRing>) -> Result<bool, ModuleError> {
        Ok(self.membership_witness(element)?.is_some())
    }

    /// Return replayable generator coefficients exactly when membership holds.
    pub fn membership_witness(
        &self,
        element: &ModuleElement<IntegerRing>,
    ) -> Result<Option<SubmoduleMembership<IntegerRing>>, ModuleError> {
        if element.module() != self.ambient() {
            return Err(ModuleError::ModuleMismatch {
                operation: "integer submodule membership",
            });
        }
        match self
            .generator_map()
            .matrix()
            .solve_exact(element.coordinates())?
        {
            IntegerLinearSystemResult::Solutions(solution) => {
                let witness = SubmoduleMembership {
                    submodule: self.clone(),
                    element: element.clone(),
                    coefficients: solution.particular_solution().clone(),
                };
                witness.verify().map_err(|error| {
                    ModuleError::LinearAlgebra(crate::linalg::LinalgError::Verification(error))
                })?;
                Ok(Some(witness))
            }
            IntegerLinearSystemResult::Inconsistent(_) => Ok(None),
        }
    }

    /// Decide inclusion using replayable membership computations on canonical generators.
    pub fn is_subset_of(&self, rhs: &Self) -> Result<bool, ModuleError> {
        require_same_ambient(self, rhs, "integer submodule containment")?;
        for generator in self.generators() {
            if rhs.membership_witness(&generator)?.is_none() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Exact intersection inside the common ambient free abelian group.
    pub fn intersection(&self, rhs: &Self) -> Result<Self, ModuleError> {
        require_same_ambient(self, rhs, "integer submodule intersection")?;
        let left = self.generator_map().matrix();
        let right = rhs.generator_map().matrix();
        let relation_operator = joined_difference(&left, &right)?;
        let zero = Vector::zeros(Z, self.ambient().rank());
        let IntegerLinearSystemResult::Solutions(relations) =
            relation_operator.solve_exact(&zero)?
        else {
            return Err(ModuleError::LinearAlgebra(
                crate::linalg::LinalgError::InconsistentSystem,
            ));
        };
        let coefficients = leading_rows(relations.homogeneous_generators(), self.generator_count());
        let generators = left.matmul(&coefficients)?;
        from_generator_matrix(self.ambient().clone(), generators)
    }

    /// Cokernel presentation `Z^r / self` retaining the canonical relation map.
    pub fn quotient(&self) -> PresentedModule<IntegerRing> {
        PresentedModule::cokernel(self.generator_map().clone())
    }

    /// Canonical invariant factors of the underlying quotient abelian group.
    pub fn quotient_invariants(&self) -> FinitelyGeneratedAbelianGroup {
        let smith = self.generator_map().matrix().smith_normal_form();
        let torsion = smith
            .invariant_factors()
            .iter()
            .filter(|factor| !factor.is_one())
            .cloned()
            .collect();
        FinitelyGeneratedAbelianGroup::from_invariants(
            self.ambient().rank() - smith.rank(),
            torsion,
        )
    }

    /// Index in the ambient lattice, failing closed when the quotient is infinite.
    pub fn finite_index(&self) -> Result<Integer, ModuleError> {
        let quotient = self.quotient_invariants();
        if quotient.free_rank() != 0 {
            return Err(ModuleError::InfiniteIndex {
                operation: "integer submodule finite index",
            });
        }
        Ok(product(quotient.torsion_coefficients()))
    }

    /// Primitive closure `(self tensor Q) intersect Z^r` in canonical form.
    pub fn saturation(&self) -> Result<Self, ModuleError> {
        let matrix = self.generator_map().matrix();
        let smith = matrix.smith_normal_form_with_transforms()?;
        let rank = smith.smith().rank();
        let left = smith.left();
        let mut columns = Vec::with_capacity(rank);
        for position in 0..rank {
            let basis = Vector::from_values(
                Z,
                (0..self.ambient().rank()).map(|row| {
                    if row == position {
                        Integer::one()
                    } else {
                        Integer::zero()
                    }
                }),
            )?;
            let IntegerLinearSystemResult::Solutions(solution) = left.solve_exact(&basis)? else {
                return Err(ModuleError::LinearAlgebra(
                    crate::linalg::LinalgError::InconsistentSystem,
                ));
            };
            columns.push(solution.particular_solution().to_values());
        }
        from_generator_matrix(
            self.ambient().clone(),
            Matrix::from_validated_columns(Z, self.ambient().rank(), columns),
        )
    }

    /// Recheck the unique column-Hermite generator convention.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let canonical = canonical_integer_columns(&self.generator_map().matrix());
        if canonical != self.generator_map().matrix() {
            return Err(VerificationError::InvalidWitness {
                claim: "integer submodule generators are in canonical column-Hermite form",
            });
        }
        Ok(())
    }
}

impl PartialEq for Submodule<IntegerRing> {
    fn eq(&self, other: &Self) -> bool {
        self.ambient == other.ambient && self.generator_map.matrix() == other.generator_map.matrix()
    }
}

impl Eq for Submodule<IntegerRing> {}

impl Submodule<ResidueRing> {
    /// Canonical ambient generators in column-Howell convention.
    pub fn generators(&self) -> Vec<ModuleElement<ResidueRing>> {
        module_generators(self)
    }

    /// Decide exact membership, including for composite moduli.
    pub fn contains(&self, element: &ModuleElement<ResidueRing>) -> Result<bool, ModuleError> {
        Ok(self.membership_witness(element)?.is_some())
    }

    /// Return replayable generator coefficients exactly when membership holds.
    pub fn membership_witness(
        &self,
        element: &ModuleElement<ResidueRing>,
    ) -> Result<Option<SubmoduleMembership<ResidueRing>>, ModuleError> {
        if element.module() != self.ambient() {
            return Err(ModuleError::ModuleMismatch {
                operation: "residue submodule membership",
            });
        }
        match self
            .generator_map()
            .matrix()
            .solve_exact(element.coordinates())?
        {
            CongruenceSystemResult::Solutions(solution) => {
                let witness = SubmoduleMembership {
                    submodule: self.clone(),
                    element: element.clone(),
                    coefficients: solution.particular_solution().clone(),
                };
                witness.verify().map_err(|error| {
                    ModuleError::LinearAlgebra(crate::linalg::LinalgError::Verification(error))
                })?;
                Ok(Some(witness))
            }
            CongruenceSystemResult::Inconsistent(_) => Ok(None),
        }
    }

    /// Decide inclusion, including over composite moduli.
    pub fn is_subset_of(&self, rhs: &Self) -> Result<bool, ModuleError> {
        require_same_ambient(self, rhs, "residue submodule containment")?;
        for generator in self.generators() {
            if rhs.membership_witness(&generator)?.is_none() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Exact intersection inside the common free `Z/nZ`-module.
    pub fn intersection(&self, rhs: &Self) -> Result<Self, ModuleError> {
        require_same_ambient(self, rhs, "residue submodule intersection")?;
        let left = self.generator_map().matrix();
        let right = rhs.generator_map().matrix();
        let relation_operator = joined_difference(&left, &right)?;
        let zero = Vector::zeros(self.ambient().ring().clone(), self.ambient().rank());
        let CongruenceSystemResult::Solutions(relations) = relation_operator.solve_exact(&zero)?
        else {
            return Err(ModuleError::LinearAlgebra(
                crate::linalg::LinalgError::InconsistentSystem,
            ));
        };
        let coefficients = leading_rows(relations.homogeneous_generators(), self.generator_count());
        let generators = left.matmul(&coefficients)?;
        from_generator_matrix(self.ambient().clone(), generators)
    }

    /// Cokernel presentation `(Z/nZ)^r / self` retaining canonical relations.
    pub fn quotient(&self) -> PresentedModule<ResidueRing> {
        PresentedModule::cokernel(self.generator_map().clone())
    }

    /// Canonical invariant factors of the quotient's underlying finite abelian group.
    pub fn quotient_invariants(&self) -> FinitelyGeneratedAbelianGroup {
        let rank = self.ambient().rank();
        let generators = self.generator_map().matrix();
        let modulus = self.ambient().ring().modulus().clone();
        let lifted = (0..rank)
            .map(|row| {
                let mut entries = (0..generators.shape().1)
                    .map(|column| {
                        generators
                            .get(row, column)
                            .expect("generator entry is in bounds")
                            .representative()
                            .clone()
                    })
                    .collect::<Vec<_>>();
                entries.extend((0..rank).map(|column| {
                    if row == column {
                        modulus.clone()
                    } else {
                        Integer::zero()
                    }
                }));
                entries
            })
            .collect::<Vec<_>>();
        let presentation = if rank == 0 {
            Matrix::zeros(Z, 0, generators.shape().1)
        } else {
            Matrix::from_rows(Z, lifted).expect("lifted presentation rows are rectangular")
        };
        let smith = presentation.smith_normal_form();
        debug_assert_eq!(smith.rank(), rank);
        FinitelyGeneratedAbelianGroup::from_invariants(
            0,
            smith
                .invariant_factors()
                .iter()
                .filter(|factor| !factor.is_one())
                .cloned()
                .collect(),
        )
    }

    /// Finite cardinal index in the ambient residue module.
    pub fn finite_index(&self) -> Integer {
        product(self.quotient_invariants().torsion_coefficients())
    }

    /// Recheck the unique column-Howell generator convention.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let canonical =
            canonical_congruence_columns(&self.generator_map().matrix()).map_err(|_| {
                VerificationError::InvalidWitness {
                    claim: "residue submodule generators admit Howell reduction",
                }
            })?;
        if canonical != self.generator_map().matrix() {
            return Err(VerificationError::InvalidWitness {
                claim: "residue submodule generators are in canonical column-Howell form",
            });
        }
        Ok(())
    }
}

impl PartialEq for Submodule<ResidueRing> {
    fn eq(&self, other: &Self) -> bool {
        self.ambient == other.ambient && self.generator_map.matrix() == other.generator_map.matrix()
    }
}

impl Eq for Submodule<ResidueRing> {}

impl ModuleHomomorphism<IntegerRing> {
    /// Canonical image lattice with map-bound completeness evidence.
    pub fn image(&self) -> Result<ImageSubmodule<IntegerRing>, ModuleError> {
        Ok(ImageSubmodule {
            homomorphism: self.clone(),
            submodule: integer_image_submodule(self)?,
        })
    }

    /// Canonical complete kernel lattice with map-bound evidence.
    pub fn kernel(&self) -> Result<KernelSubmodule<IntegerRing>, ModuleError> {
        Ok(KernelSubmodule {
            homomorphism: self.clone(),
            submodule: integer_kernel_submodule(self)?,
        })
    }
}

impl ModuleHomomorphism<ResidueRing> {
    /// Canonical image submodule with map-bound completeness evidence.
    pub fn image(&self) -> Result<ImageSubmodule<ResidueRing>, ModuleError> {
        Ok(ImageSubmodule {
            homomorphism: self.clone(),
            submodule: residue_image_submodule(self)?,
        })
    }

    /// Canonical complete kernel with evidence over a possibly composite modulus.
    pub fn kernel(&self) -> Result<KernelSubmodule<ResidueRing>, ModuleError> {
        Ok(KernelSubmodule {
            homomorphism: self.clone(),
            submodule: residue_kernel_submodule(self)?,
        })
    }
}

impl KernelSubmodule<IntegerRing> {
    /// Replay zero-containment and canonical completeness of this kernel.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_kernel_generators(&self.homomorphism, &self.submodule)?;
        let expected = integer_kernel_submodule(&self.homomorphism)
            .map_err(|_| invalid_kernel("integer kernel can be recomputed"))?;
        if expected != self.submodule {
            return Err(invalid_kernel(
                "integer kernel equals the independently recomputed canonical kernel",
            ));
        }
        Ok(())
    }

    /// Deserialize and fully replay a kernel certificate.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: BoundSubmoduleWire<IntegerRing> = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            homomorphism: wire.homomorphism,
            submodule: wire.submodule,
        };
        result.verify().map_err(NoetherError::Verification)?;
        Ok(result)
    }
}

impl KernelSubmodule<ResidueRing> {
    /// Replay zero-containment and canonical completeness over `Z/nZ`.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_kernel_generators(&self.homomorphism, &self.submodule)?;
        let expected = residue_kernel_submodule(&self.homomorphism)
            .map_err(|_| invalid_kernel("residue kernel can be recomputed"))?;
        if expected != self.submodule {
            return Err(invalid_kernel(
                "residue kernel equals the independently recomputed canonical kernel",
            ));
        }
        Ok(())
    }

    /// Deserialize and fully replay a kernel certificate.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: BoundSubmoduleWire<ResidueRing> = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            homomorphism: wire.homomorphism,
            submodule: wire.submodule,
        };
        result.verify().map_err(NoetherError::Verification)?;
        Ok(result)
    }
}

impl ImageSubmodule<IntegerRing> {
    /// Replay generator containment and canonical completeness of this image.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_image_generators(&self.homomorphism, &self.submodule)?;
        let expected = integer_image_submodule(&self.homomorphism)
            .map_err(|_| invalid_image("integer image can be recomputed"))?;
        if expected != self.submodule {
            return Err(invalid_image(
                "integer image equals the independently recomputed canonical image",
            ));
        }
        Ok(())
    }

    /// Deserialize and fully replay an image certificate.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: BoundSubmoduleWire<IntegerRing> = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            homomorphism: wire.homomorphism,
            submodule: wire.submodule,
        };
        result.verify().map_err(NoetherError::Verification)?;
        Ok(result)
    }
}

impl ImageSubmodule<ResidueRing> {
    /// Replay generator containment and canonical completeness over `Z/nZ`.
    pub fn verify(&self) -> Result<(), VerificationError> {
        verify_image_generators(&self.homomorphism, &self.submodule)?;
        let expected = residue_image_submodule(&self.homomorphism)
            .map_err(|_| invalid_image("residue image can be recomputed"))?;
        if expected != self.submodule {
            return Err(invalid_image(
                "residue image equals the independently recomputed canonical image",
            ));
        }
        Ok(())
    }

    /// Deserialize and fully replay an image certificate.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire: BoundSubmoduleWire<ResidueRing> = serde_json::from_str(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            homomorphism: wire.homomorphism,
            submodule: wire.submodule,
        };
        result.verify().map_err(NoetherError::Verification)?;
        Ok(result)
    }
}

fn integer_kernel_submodule(
    homomorphism: &ModuleHomomorphism<IntegerRing>,
) -> Result<Submodule<IntegerRing>, ModuleError> {
    let zero = Vector::zeros(Z, homomorphism.target().rank());
    let IntegerLinearSystemResult::Solutions(solution) =
        homomorphism.matrix().solve_exact(&zero)?
    else {
        return Err(ModuleError::LinearAlgebra(
            crate::linalg::LinalgError::InconsistentSystem,
        ));
    };
    from_generator_matrix(
        homomorphism.source().clone(),
        solution.homogeneous_generators().clone(),
    )
}

fn residue_kernel_submodule(
    homomorphism: &ModuleHomomorphism<ResidueRing>,
) -> Result<Submodule<ResidueRing>, ModuleError> {
    let zero = Vector::zeros(
        homomorphism.target().ring().clone(),
        homomorphism.target().rank(),
    );
    let CongruenceSystemResult::Solutions(solution) = homomorphism.matrix().solve_exact(&zero)?
    else {
        return Err(ModuleError::LinearAlgebra(
            crate::linalg::LinalgError::InconsistentSystem,
        ));
    };
    from_generator_matrix(
        homomorphism.source().clone(),
        solution.homogeneous_generators().clone(),
    )
}

fn integer_image_submodule(
    homomorphism: &ModuleHomomorphism<IntegerRing>,
) -> Result<Submodule<IntegerRing>, ModuleError> {
    Submodule::from_generator_map(homomorphism.target().clone(), homomorphism.clone())
}

fn residue_image_submodule(
    homomorphism: &ModuleHomomorphism<ResidueRing>,
) -> Result<Submodule<ResidueRing>, ModuleError> {
    Submodule::from_generator_map(homomorphism.target().clone(), homomorphism.clone())
}

fn verify_kernel_generators<R: Ring>(
    homomorphism: &ModuleHomomorphism<R>,
    submodule: &Submodule<R>,
) -> Result<(), VerificationError> {
    if submodule.ambient() != homomorphism.source() {
        return Err(invalid_kernel("kernel is a submodule of the map source"));
    }
    for generator in module_generators(submodule) {
        let value = homomorphism
            .apply(&generator)
            .map_err(|_| invalid_kernel("kernel generator can be mapped"))?;
        if value != ModuleElement::zero(homomorphism.target().clone()) {
            return Err(invalid_kernel("every kernel generator maps to zero"));
        }
    }
    Ok(())
}

fn verify_image_generators<R: Ring>(
    homomorphism: &ModuleHomomorphism<R>,
    submodule: &Submodule<R>,
) -> Result<(), VerificationError> {
    if submodule.ambient() != homomorphism.target() {
        return Err(invalid_image("image is a submodule of the map target"));
    }
    for position in 0..homomorphism.source().rank() {
        let basis = homomorphism
            .source()
            .basis_element(position)
            .map_err(|_| invalid_image("source basis can be reconstructed"))?;
        let value = homomorphism
            .apply(&basis)
            .map_err(|_| invalid_image("source basis image can be reconstructed"))?;
        let source = FreeModule::new(homomorphism.target().ring().clone(), 1);
        let generator_map = ModuleHomomorphism::from_coordinate_columns(
            source,
            homomorphism.target().clone(),
            vec![value.coordinates().to_values()],
        )
        .map_err(|_| invalid_image("basis image defines a one-generator submodule"))?;
        let candidate = Submodule {
            ambient: homomorphism.target().clone(),
            generator_map,
        };
        if !submodule_contains_supported(submodule, &candidate)? {
            return Err(invalid_image("every source basis image lies in the image"));
        }
    }
    Ok(())
}

fn submodule_contains_supported<R: Ring>(
    submodule: &Submodule<R>,
    candidate: &Submodule<R>,
) -> Result<bool, VerificationError> {
    if TypeId::of::<R>() == TypeId::of::<IntegerRing>() {
        let left = (submodule as &dyn Any)
            .downcast_ref::<Submodule<IntegerRing>>()
            .expect("integer submodule type-id invariant");
        let right = (candidate as &dyn Any)
            .downcast_ref::<Submodule<IntegerRing>>()
            .expect("integer submodule type-id invariant");
        return right
            .is_subset_of(left)
            .map_err(|_| invalid_image("integer image membership can be recomputed"));
    }
    if TypeId::of::<R>() == TypeId::of::<ResidueRing>() {
        let left = (submodule as &dyn Any)
            .downcast_ref::<Submodule<ResidueRing>>()
            .expect("residue submodule type-id invariant");
        let right = (candidate as &dyn Any)
            .downcast_ref::<Submodule<ResidueRing>>()
            .expect("residue submodule type-id invariant");
        return right
            .is_subset_of(left)
            .map_err(|_| invalid_image("residue image membership can be recomputed"));
    }
    Err(invalid_image("image coefficient context is supported"))
}

fn invalid_kernel(claim: &'static str) -> VerificationError {
    VerificationError::InvalidWitness { claim }
}

fn invalid_image(claim: &'static str) -> VerificationError {
    VerificationError::InvalidWitness { claim }
}

fn product(values: &[Integer]) -> Integer {
    values
        .iter()
        .fold(Integer::one(), |product, value| product * value)
}

fn canonicalize_integer(
    submodule: Submodule<IntegerRing>,
) -> Result<Submodule<IntegerRing>, ModuleError> {
    let canonical = canonical_integer_columns(&submodule.generator_map.matrix());
    from_canonical_matrix(submodule.ambient, canonical)
}

fn canonicalize_congruence(
    submodule: Submodule<ResidueRing>,
) -> Result<Submodule<ResidueRing>, ModuleError> {
    let canonical = canonical_congruence_columns(&submodule.generator_map.matrix())?;
    from_canonical_matrix(submodule.ambient, canonical)
}

fn from_generator_matrix<R: Ring>(
    ambient: FreeModule<R>,
    matrix: Matrix<R>,
) -> Result<Submodule<R>, ModuleError> {
    let source = FreeModule::new(ambient.ring().clone(), matrix.shape().1);
    let map = ModuleHomomorphism::from_matrix(source, ambient.clone(), matrix)?;
    Submodule::from_generator_map(ambient, map)
}

fn from_canonical_matrix<R: Ring>(
    ambient: FreeModule<R>,
    matrix: Matrix<R>,
) -> Result<Submodule<R>, ModuleError> {
    let source = FreeModule::new(ambient.ring().clone(), matrix.shape().1);
    let generator_map = ModuleHomomorphism::from_matrix(source, ambient.clone(), matrix)?;
    Ok(Submodule {
        ambient,
        generator_map,
    })
}

fn module_generators<R: Ring>(submodule: &Submodule<R>) -> Vec<ModuleElement<R>> {
    let matrix = submodule.generator_map.matrix();
    (0..matrix.shape().1)
        .map(|column| {
            let coordinates = Vector::from_values(
                submodule.ambient.ring().clone(),
                (0..matrix.shape().0)
                    .map(|row| {
                        matrix
                            .get(row, column)
                            .expect("generator coordinate is in bounds")
                    })
                    .collect::<Vec<_>>(),
            )
            .expect("canonical generator column has the ambient dimension");
            ModuleElement::from_coordinates(submodule.ambient.clone(), coordinates)
                .expect("canonical generator belongs to its ambient module")
        })
        .collect()
}

fn require_same_ambient<R: Ring>(
    left: &Submodule<R>,
    right: &Submodule<R>,
    operation: &'static str,
) -> Result<(), ModuleError> {
    if left.ambient() == right.ambient() {
        Ok(())
    } else {
        Err(ModuleError::ModuleMismatch { operation })
    }
}

fn joined_difference<R: Ring>(
    left: &Matrix<R>,
    right: &Matrix<R>,
) -> Result<Matrix<R>, ModuleError> {
    let (rows, left_columns) = left.shape();
    if right.shape().0 != rows || left.ring() != right.ring() {
        return Err(ModuleError::ModuleMismatch {
            operation: "submodule intersection relation matrix",
        });
    }
    let negated = right.neg()?;
    let columns = (0..left_columns)
        .map(|column| {
            (0..rows)
                .map(|row| left.get(row, column).expect("left column is in bounds"))
                .collect::<Vec<_>>()
        })
        .chain((0..right.shape().1).map(|column| {
            (0..rows)
                .map(|row| negated.get(row, column).expect("right column is in bounds"))
                .collect::<Vec<_>>()
        }))
        .collect();
    Ok(Matrix::from_validated_columns(
        left.ring().clone(),
        rows,
        columns,
    ))
}

fn leading_rows<R: Ring>(matrix: &Matrix<R>, row_count: usize) -> Matrix<R> {
    let columns = (0..matrix.shape().1)
        .map(|column| {
            (0..row_count)
                .map(|row| {
                    matrix
                        .get(row, column)
                        .expect("projected relation coordinate is in bounds")
                })
                .collect::<Vec<_>>()
        })
        .collect();
    Matrix::from_validated_columns(matrix.ring().clone(), row_count, columns)
}

fn as_any_box<T: Any>(value: T) -> Box<dyn Any> {
    Box::new(value)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::commalg::ResidueRing;
    use crate::homalg::{FreeModule, ModuleElement, ModuleError, ModuleHomomorphism, Submodule};
    use crate::linalg::{Matrix, Vector};
    use crate::{Integer, IntegerRing, Z};

    fn integer_element(
        ambient: &FreeModule<IntegerRing>,
        values: impl IntoIterator<Item = i64>,
    ) -> ModuleElement<IntegerRing> {
        ModuleElement::from_coordinates(
            ambient.clone(),
            Vector::from_values(Z, values.into_iter().map(Integer::from).collect::<Vec<_>>())
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn integer_submodules_are_canonical_and_support_exact_set_operations() {
        let ambient = FreeModule::new(Z, 2);
        let submodule = Submodule::from_generators(
            ambient.clone(),
            vec![
                integer_element(&ambient, [2, 0]),
                integer_element(&ambient, [0, 3]),
                integer_element(&ambient, [4, 6]),
            ],
        )
        .unwrap();
        assert_eq!(submodule.generator_count(), 2);
        submodule.verify().unwrap();
        assert_eq!(
            Submodule::<IntegerRing>::from_json(&submodule.to_json().unwrap()).unwrap(),
            submodule
        );
        assert!(submodule
            .contains(&integer_element(&ambient, [6, 9]))
            .unwrap());
        assert!(!submodule
            .contains(&integer_element(&ambient, [1, 0]))
            .unwrap());
        assert_eq!(
            submodule.quotient_invariants().torsion_coefficients(),
            [Integer::from(6)]
        );

        let rank_one = FreeModule::new(Z, 1);
        let twice =
            Submodule::from_generators(rank_one.clone(), vec![integer_element(&rank_one, [2])])
                .unwrap();
        let thrice =
            Submodule::from_generators(rank_one.clone(), vec![integer_element(&rank_one, [3])])
                .unwrap();
        let intersection = twice.intersection(&thrice).unwrap();
        assert!(intersection
            .contains(&integer_element(&rank_one, [6]))
            .unwrap());
        assert!(!intersection
            .contains(&integer_element(&rank_one, [3]))
            .unwrap());
        let sum = twice.sum(&thrice).unwrap();
        assert!(sum.contains(&integer_element(&rank_one, [1])).unwrap());
        assert_eq!(sum.generator_count(), 1);
    }

    #[test]
    fn composite_residue_submodules_use_howell_not_field_reduction() {
        let ring = ResidueRing::new(6).unwrap();
        let ambient = FreeModule::new(ring.clone(), 1);
        let element = |value| {
            ModuleElement::from_coordinates(
                ambient.clone(),
                Vector::from_values(ring.clone(), [ring.element(value)]).unwrap(),
            )
            .unwrap()
        };
        let evens =
            Submodule::from_generators(ambient.clone(), vec![element(2), element(4)]).unwrap();
        let threes = Submodule::from_generators(ambient.clone(), vec![element(3)]).unwrap();
        evens.verify().unwrap();
        assert_eq!(evens.generator_count(), 1);
        assert!(evens.contains(&element(4)).unwrap());
        assert!(!evens.contains(&element(3)).unwrap());
        assert!(evens.intersection(&threes).unwrap().generators().is_empty());
        let whole = evens.sum(&threes).unwrap();
        assert!(whole.contains(&element(1)).unwrap());
        assert_eq!(
            evens.quotient_invariants().torsion_coefficients(),
            [Integer::from(2)]
        );
    }

    #[test]
    fn power_of_two_submodule_intersection_handles_zero_generator_matrices() {
        let ring = ResidueRing::new(4).unwrap();
        let ambient = FreeModule::new(ring.clone(), 1);
        let element = |value| {
            ModuleElement::from_coordinates(
                ambient.clone(),
                Vector::from_values(ring.clone(), [ring.element(value)]).unwrap(),
            )
            .unwrap()
        };
        let nonzero = Submodule::from_generators(ambient.clone(), vec![element(2)]).unwrap();
        let zero = Submodule::from_generators(ambient, vec![]).unwrap();

        let left = nonzero.intersection(&zero).unwrap();
        let right = zero.intersection(&nonzero).unwrap();
        assert!(left.generators().is_empty());
        assert!(right.generators().is_empty());
        left.verify().unwrap();
        right.verify().unwrap();
    }

    #[test]
    fn zero_and_full_submodules_have_correct_quotients() {
        let ambient = FreeModule::new(Z, 3);
        let zero = Submodule::from_generators(ambient.clone(), vec![]).unwrap();
        assert_eq!(zero.quotient_invariants().free_rank(), 3);
        let full = Submodule::from_generators(
            ambient.clone(),
            (0..3)
                .map(|index| {
                    integer_element(
                        &ambient,
                        (0..3).map(|position| i64::from(position == index)),
                    )
                })
                .collect(),
        )
        .unwrap();
        assert!(full.quotient_invariants().is_trivial());
    }

    #[test]
    fn membership_witnesses_reconstruct_and_reject_tampering() {
        let ambient = FreeModule::new(Z, 2);
        let submodule = Submodule::from_generators(
            ambient.clone(),
            vec![
                integer_element(&ambient, [2, 0]),
                integer_element(&ambient, [1, 3]),
            ],
        )
        .unwrap();
        let element = integer_element(&ambient, [5, 9]);
        let witness = submodule.membership_witness(&element).unwrap().unwrap();
        witness.verify().unwrap();
        assert_eq!(
            super::SubmoduleMembership::<IntegerRing>::from_json(&witness.to_json().unwrap())
                .unwrap()
                .element(),
            &element
        );

        let mut forged: serde_json::Value =
            serde_json::from_str(&witness.to_json().unwrap()).unwrap();
        forged["coefficients"] =
            serde_json::to_value(Vector::zeros(Z, submodule.generator_count())).unwrap();
        assert!(super::SubmoduleMembership::<IntegerRing>::from_json(&forged.to_string()).is_err());
    }

    #[test]
    fn integer_saturation_and_finite_index_obey_lattice_laws() {
        let ambient = FreeModule::new(Z, 2);
        let lattice = Submodule::from_generators(
            ambient.clone(),
            vec![
                integer_element(&ambient, [2, 0]),
                integer_element(&ambient, [0, 3]),
            ],
        )
        .unwrap();
        assert_eq!(lattice.finite_index().unwrap(), Integer::from(6));
        let saturated = lattice.saturation().unwrap();
        assert!(saturated
            .is_subset_of(&saturated.saturation().unwrap())
            .unwrap());
        assert_eq!(saturated, saturated.saturation().unwrap());
        assert_eq!(saturated.finite_index().unwrap(), Integer::one());

        let line =
            Submodule::from_generators(ambient.clone(), vec![integer_element(&ambient, [6, 0])])
                .unwrap();
        assert_eq!(
            line,
            line.saturation().unwrap().intersection(&line).unwrap()
        );
        assert!(matches!(
            line.finite_index(),
            Err(ModuleError::InfiniteIndex { .. })
        ));

        let rank_one = FreeModule::new(Z, 1);
        let twice =
            Submodule::from_generators(rank_one.clone(), vec![integer_element(&rank_one, [2])])
                .unwrap();
        let six =
            Submodule::from_generators(rank_one.clone(), vec![integer_element(&rank_one, [6])])
                .unwrap();
        assert_eq!(six.finite_index().unwrap(), Integer::from(6));
        assert_eq!(twice.finite_index().unwrap(), Integer::from(2));
        let relative_generator = twice
            .membership_witness(&integer_element(&rank_one, [6]))
            .unwrap()
            .unwrap()
            .coefficients()
            .get(0)
            .unwrap();
        assert_eq!(
            six.finite_index().unwrap(),
            twice.finite_index().unwrap() * relative_generator.abs()
        );
    }

    #[test]
    fn exact_kernel_and_image_are_canonical_submodules() {
        let source = FreeModule::new(Z, 3);
        let target = FreeModule::new(Z, 2);
        let map = ModuleHomomorphism::from_matrix(
            source.clone(),
            target.clone(),
            Matrix::from_rows(
                Z,
                [
                    [Integer::from(2), Integer::from(4), Integer::zero()],
                    [Integer::zero(), Integer::zero(), Integer::from(3)],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let kernel = map.kernel().unwrap();
        let image = map.image().unwrap();
        kernel.verify().unwrap();
        image.verify().unwrap();
        for generator in kernel.submodule().generators() {
            assert_eq!(
                map.apply(&generator).unwrap(),
                ModuleElement::zero(target.clone())
            );
        }
        for position in 0..source.rank() {
            let basis = source.basis_element(position).unwrap();
            assert!(image
                .submodule()
                .contains(&map.apply(&basis).unwrap())
                .unwrap());
        }

        let ring = ResidueRing::new(6).unwrap();
        let source = FreeModule::new(ring.clone(), 2);
        let target = FreeModule::new(ring.clone(), 1);
        let map = ModuleHomomorphism::from_matrix(
            source.clone(),
            target.clone(),
            Matrix::from_rows(ring.clone(), [[ring.element(2), ring.element(3)]]).unwrap(),
        )
        .unwrap();
        let kernel = map.kernel().unwrap();
        let image = map.image().unwrap();
        for generator in kernel.submodule().generators() {
            assert_eq!(
                map.apply(&generator).unwrap(),
                ModuleElement::zero(target.clone())
            );
        }
        for position in 0..source.rank() {
            let basis = source.basis_element(position).unwrap();
            assert!(image
                .submodule()
                .contains(&map.apply(&basis).unwrap())
                .unwrap());
        }
        assert_eq!(image.submodule().finite_index(), Integer::one());

        let kernel_json = kernel.to_json().unwrap();
        super::KernelSubmodule::<ResidueRing>::from_json(&kernel_json)
            .unwrap()
            .verify()
            .unwrap();
        let mut forged: serde_json::Value = serde_json::from_str(&kernel_json).unwrap();
        forged["submodule"] = serde_json::from_str(
            &Submodule::from_generators(source.clone(), vec![])
                .unwrap()
                .to_json()
                .unwrap(),
        )
        .unwrap();
        assert!(super::KernelSubmodule::<ResidueRing>::from_json(&forged.to_string()).is_err());

        let image_json = image.to_json().unwrap();
        super::ImageSubmodule::<ResidueRing>::from_json(&image_json)
            .unwrap()
            .verify()
            .unwrap();
        let mut forged: serde_json::Value = serde_json::from_str(&image_json).unwrap();
        forged["submodule"] = serde_json::from_str(
            &Submodule::from_generators(target, vec![])
                .unwrap()
                .to_json()
                .unwrap(),
        )
        .unwrap();
        assert!(super::ImageSubmodule::<ResidueRing>::from_json(&forged.to_string()).is_err());
    }

    proptest! {
        #[test]
        fn rank_one_integer_membership_matches_divisibility(
            first in -30i64..=30,
            second in -30i64..=30,
            value in -100i64..=100,
        ) {
            let ambient = FreeModule::new(Z, 1);
            let submodule = Submodule::from_generators(
                ambient.clone(),
                vec![integer_element(&ambient, [first]), integer_element(&ambient, [second])],
            ).unwrap();
            let gcd = Integer::from(first).gcd(&Integer::from(second));
            let expected = if gcd.is_zero() {
                value == 0
            } else {
                Integer::from(value).is_divisible(&gcd)
            };
            prop_assert_eq!(submodule.contains(&integer_element(&ambient, [value])).unwrap(), expected);
            prop_assert!(submodule.verify().is_ok());
            if let Some(witness) = submodule.membership_witness(&integer_element(&ambient, [value])).unwrap() {
                prop_assert!(witness.verify().is_ok());
            }
            let saturation = submodule.saturation().unwrap();
            prop_assert_eq!(saturation.saturation().unwrap(), saturation);
            if !gcd.is_zero() {
                prop_assert_eq!(submodule.finite_index().unwrap(), gcd.abs());
            }
        }

        #[test]
        fn rank_one_residue_membership_matches_brute_force(
            modulus in 2u32..20,
            generator in 0u32..20,
            value in 0u32..20,
        ) {
            let ring = ResidueRing::new(modulus).unwrap();
            let ambient = FreeModule::new(ring.clone(), 1);
            let element = |entry| ModuleElement::from_coordinates(
                ambient.clone(),
                Vector::from_values(ring.clone(), [ring.element(entry)]).unwrap(),
            ).unwrap();
            let submodule =
                Submodule::from_generators(ambient.clone(), vec![element(generator)]).unwrap();
            let expected = (0..modulus).any(|coefficient| {
                (coefficient * generator) % modulus == value % modulus
            });
            prop_assert_eq!(submodule.contains(&element(value)).unwrap(), expected);
            prop_assert!(submodule.verify().is_ok());
            if let Some(witness) = submodule.membership_witness(&element(value)).unwrap() {
                prop_assert!(witness.verify().is_ok());
            }
            let generated_count = (0..modulus)
                .filter(|coefficient| (coefficient * generator) % modulus == 0)
                .count();
            let image_size = usize::try_from(modulus).unwrap() / generated_count;
            prop_assert_eq!(submodule.finite_index(), Integer::from(modulus as usize / image_size));
        }

        #[test]
        fn integer_kernel_image_saturation_and_normal_forms_are_stable(
            entries in prop::collection::vec(-12i64..=12, 4),
        ) {
            let source = FreeModule::new(Z, 2);
            let target = FreeModule::new(Z, 2);
            let matrix = Matrix::from_rows(
                Z,
                [
                    [Integer::from(entries[0]), Integer::from(entries[1])],
                    [Integer::from(entries[2]), Integer::from(entries[3])],
                ],
            ).unwrap();
            let map = ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix).unwrap();
            let kernel = map.kernel().unwrap();
            let image = map.image().unwrap();
            for generator in kernel.submodule().generators() {
                prop_assert_eq!(map.apply(&generator).unwrap(), ModuleElement::zero(target.clone()));
            }
            for position in 0..source.rank() {
                let value = map.apply(&source.basis_element(position).unwrap()).unwrap();
                prop_assert!(image.submodule().membership_witness(&value).unwrap().is_some());
            }

            let generated = Submodule::from_generators(
                target.clone(),
                vec![
                    integer_element(&target, [entries[0], entries[2]]),
                    integer_element(&target, [entries[1], entries[3]]),
                ],
            ).unwrap();
            let redundant = Submodule::from_generators(
                target.clone(),
                vec![
                    integer_element(&target, [entries[0], entries[2]]),
                    integer_element(&target, [entries[1], entries[3]]),
                    integer_element(&target, [entries[0] + entries[1], entries[2] + entries[3]]),
                ],
            ).unwrap();
            prop_assert_eq!(&generated, &redundant);
            let saturated = generated.saturation().unwrap();
            prop_assert_eq!(saturated.saturation().unwrap(), saturated.clone());
            prop_assert!(generated.is_subset_of(&saturated).unwrap());
            prop_assert!(saturated.verify().is_ok());
            if entries[0] * entries[3] - entries[1] * entries[2] != 0 {
                prop_assert_eq!(saturated.finite_index().unwrap(), Integer::one());
            } else {
                let has_infinite_index = matches!(
                    saturated.finite_index(),
                    Err(ModuleError::InfiniteIndex { .. })
                );
                prop_assert!(has_infinite_index);
            }
        }

        #[test]
        fn integer_index_is_multiplicative_for_nested_rank_one_lattices(
            outer_scale in 1i64..=30,
            relative_scale in 1i64..=30,
        ) {
            let ambient = FreeModule::new(Z, 1);
            let outer = Submodule::from_generators(
                ambient.clone(),
                vec![integer_element(&ambient, [outer_scale])],
            ).unwrap();
            let inner = Submodule::from_generators(
                ambient.clone(),
                vec![integer_element(&ambient, [outer_scale * relative_scale])],
            ).unwrap();
            let inner_generator = inner.generators().into_iter().next().unwrap();
            let relative_witness = outer.membership_witness(&inner_generator).unwrap().unwrap();
            prop_assert!(relative_witness.verify().is_ok());
            let relative_index = relative_witness.coefficients().get(0).unwrap().abs();
            prop_assert_eq!(
                inner.finite_index().unwrap(),
                outer.finite_index().unwrap() * relative_index,
            );
        }

        #[test]
        fn composite_residue_kernel_is_complete_by_exhaustion(
            modulus in 2u32..=8,
            entries in prop::collection::vec(0u32..16, 4),
        ) {
            let ring = ResidueRing::new(modulus).unwrap();
            let source = FreeModule::new(ring.clone(), 2);
            let target = FreeModule::new(ring.clone(), 2);
            let matrix = Matrix::from_rows(
                ring.clone(),
                [
                    [ring.element(entries[0]), ring.element(entries[1])],
                    [ring.element(entries[2]), ring.element(entries[3])],
                ],
            ).unwrap();
            let map = ModuleHomomorphism::from_matrix(source.clone(), target.clone(), matrix).unwrap();
            let kernel = map.kernel().unwrap();
            let image = map.image().unwrap();
            for first in 0..modulus {
                for second in 0..modulus {
                    let element = ModuleElement::from_coordinates(
                        source.clone(),
                        Vector::from_values(
                            ring.clone(),
                            [ring.element(first), ring.element(second)],
                        ).unwrap(),
                    ).unwrap();
                    let mapped = map.apply(&element).unwrap();
                    prop_assert_eq!(
                        kernel.submodule().contains(&element).unwrap(),
                        mapped == ModuleElement::zero(target.clone()),
                    );
                    prop_assert!(image.submodule().contains(&mapped).unwrap());
                }
            }
            prop_assert!(kernel.verify().is_ok());
            prop_assert!(image.verify().is_ok());
        }
    }
}
