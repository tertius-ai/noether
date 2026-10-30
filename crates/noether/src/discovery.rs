// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Rust-owned discovery metadata for accepted semantic wire types.
//!
//! The public facade owns semantic identity, wire versions, and canonical JSON
//! schemas. Language transports may add conversion functions for these types,
//! but may not invent another accepted identity or schema.

// These private structs are type-only descriptions consumed by schemars;
// values of them are never constructed.
#![allow(dead_code)]

use schemars::JsonSchema;

use crate::category::SteinerCategory;
use crate::combin::{
    Composition, Graph, KashiwaraCrystal, Matroid, Partition, Poset, SymmetricFunction, Tableau,
    Tree, VertexId,
};
use crate::commalg::{
    FrobeniusPolynomial, HyperellipticCurve, HyperellipticFrobeniusEvidence, PolynomialRing,
    ZetaFunction,
};
use crate::commalg::{PolynomialHomomorphism, RationalFunctionFieldMap};
use crate::groups::{PolynomialGlRepresentation, WeightCharacter, WeightLattice, WeightLatticeMap};
use crate::homalg::{
    ExteriorAlgebra, ExteriorElement, FinitelyGeneratedAbelianGroup, ImageSubmodule,
    KernelSubmodule, ModulePowerHomomorphism, PresentedModulePowerHomomorphism,
};
use crate::numthy::{FiniteField, FiniteFieldEmbedding};
use crate::topology::{
    EilenbergMacLaneIntegralCohomology, EilenbergMacLaneIntegralCohomologyPresentation,
    EilenbergMacLaneIntegralHomology, EilenbergMacLaneModularCohomology,
    EilenbergMacLaneModularCohomologyTruncation, EilenbergMacLaneSpace,
};
use crate::{
    IntegerFiniteFieldMap, IntegerRationalEmbedding, IntegerReduction, IntegerRing,
    PrimeSubfieldEmbedding, RationalField, SemanticType, TypeDescription,
};

type RegisteredPolynomialHomomorphism =
    PolynomialHomomorphism<IntegerRing, RationalField, IntegerRationalEmbedding>;
type RegisteredRationalFunctionFieldMap =
    RationalFunctionFieldMap<FiniteField, FiniteField, PrimeSubfieldEmbedding>;
type RegisteredExteriorAlgebra = ExteriorAlgebra<PolynomialRing<RationalField>>;
type RegisteredExteriorElement = ExteriorElement<PolynomialRing<RationalField>>;
type RegisteredModulePowerHomomorphism = ModulePowerHomomorphism<PolynomialRing<RationalField>>;
type RegisteredPresentedModulePowerHomomorphism =
    PresentedModulePowerHomomorphism<PolynomialRing<RationalField>>;

/// One accepted semantic type in Noether's canonical JSON wire vocabulary.
#[derive(Clone, Copy)]
pub struct TypeSchemaRegistration {
    description: fn() -> &'static TypeDescription,
    /// Canonical JSON shape version.
    pub version: u32,
    schema: fn() -> serde_json::Value,
}

impl TypeSchemaRegistration {
    /// Static semantic identity and public language paths.
    #[must_use]
    pub fn description(self) -> &'static TypeDescription {
        (self.description)()
    }

    /// JSON Schema for the canonical serialized representation.
    #[must_use]
    pub fn schema(self) -> serde_json::Value {
        (self.schema)()
    }
}

fn describe<T: SemanticType>() -> &'static TypeDescription {
    T::describe()
}

fn schema<T: JsonSchema>() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(serde_json::Value::Null)
}

macro_rules! registration {
    ($semantic:ty, $wire:ty) => {
        TypeSchemaRegistration {
            description: describe::<$semantic>,
            version: 1,
            schema: schema::<$wire>,
        }
    };
}

#[derive(JsonSchema)]
#[schemars(rename = "KashiwaraCrystal")]
struct KashiwaraCrystalWire {
    cartan_matrix: Vec<Vec<i64>>,
    weights: Vec<Vec<i64>>,
    f_edges: Vec<Vec<Option<VertexId>>>,
}

#[derive(JsonSchema)]
#[schemars(rename = "Graph")]
struct GraphWire {
    order: usize,
    edges: Vec<(VertexId, VertexId, i64)>,
    directed: bool,
}

#[derive(JsonSchema)]
#[schemars(rename = "Matroid")]
struct MatroidWire {
    ground_set: Vec<VertexId>,
    bases: Vec<Vec<VertexId>>,
}

#[derive(JsonSchema)]
#[schemars(rename = "Partition")]
struct PartitionWire {
    parts: Vec<usize>,
}

#[derive(JsonSchema)]
#[schemars(rename = "Composition")]
struct CompositionWire {
    parts: Vec<usize>,
}

#[derive(JsonSchema)]
#[schemars(rename = "Poset")]
struct PosetWire {
    order: usize,
    covers: Vec<(VertexId, VertexId)>,
}

#[derive(JsonSchema)]
struct SignedBoundaryWire {
    negative: Vec<usize>,
    positive: Vec<usize>,
}

#[derive(JsonSchema)]
#[schemars(rename = "SteinerCategory")]
struct SteinerCategoryWire {
    #[schemars(rename = "type")]
    type_id: String,
    atom_counts: Vec<usize>,
    boundaries: Vec<Vec<SignedBoundaryWire>>,
}

#[derive(JsonSchema)]
struct SymmetricTermWire {
    partition: PartitionWire,
    coefficient: String,
}

#[derive(JsonSchema)]
#[schemars(rename = "SymmetricFunction")]
struct SymmetricFunctionWire {
    basis: String,
    terms: Vec<SymmetricTermWire>,
}

#[derive(JsonSchema)]
#[schemars(rename = "Tableau")]
struct TableauWire {
    rows: Vec<Vec<String>>,
}

#[derive(JsonSchema)]
#[serde(rename_all = "snake_case")]
enum WeightBasisConventionWire {
    Standard,
    TypeAFundamental { degree: usize },
}

#[derive(JsonSchema)]
#[schemars(rename = "WeightLattice")]
struct WeightLatticeSchema {
    #[schemars(rename = "type")]
    type_id: String,
    basis_names: Vec<String>,
    convention: WeightBasisConventionWire,
}

#[derive(JsonSchema)]
#[schemars(rename = "WeightLatticeMap")]
struct WeightLatticeMapSchema {
    #[schemars(rename = "type")]
    type_id: String,
    source: WeightLatticeSchema,
    target: WeightLatticeSchema,
    basis_images: Vec<Vec<String>>,
}

#[derive(JsonSchema)]
struct WeightMultiplicitySchema {
    weight: Vec<String>,
    multiplicity: String,
}

#[derive(JsonSchema)]
#[schemars(rename = "WeightCharacter")]
struct WeightCharacterSchema {
    #[schemars(rename = "type")]
    type_id: String,
    lattice: WeightLatticeSchema,
    terms: Vec<WeightMultiplicitySchema>,
}

#[derive(JsonSchema)]
#[schemars(rename = "PolynomialGlRepresentation")]
struct PolynomialGlRepresentationSchema {
    #[schemars(rename = "type")]
    type_id: String,
    rank: usize,
    partition: PartitionWire,
}

#[derive(JsonSchema)]
#[schemars(rename = "Tree")]
struct TreeWire {
    order: usize,
    edges: Vec<(VertexId, VertexId)>,
    root: VertexId,
}

#[derive(JsonSchema)]
#[schemars(rename = "FinitelyGeneratedAbelianGroup")]
struct FinitelyGeneratedAbelianGroupWire {
    #[schemars(rename = "type")]
    type_id: String,
    free_rank: usize,
    torsion_coefficients: Vec<String>,
}

#[derive(JsonSchema)]
#[schemars(rename = "ExteriorAlgebra")]
struct ExteriorAlgebraSchema {
    #[schemars(rename = "type")]
    type_id: String,
    source: serde_json::Value,
}

#[derive(JsonSchema)]
struct ExteriorTermSchema {
    basis: Vec<usize>,
    coefficient: serde_json::Value,
}

#[derive(JsonSchema)]
#[schemars(rename = "ExteriorElement")]
struct ExteriorElementSchema {
    #[schemars(rename = "type")]
    type_id: String,
    algebra: ExteriorAlgebraSchema,
    terms: Vec<ExteriorTermSchema>,
}

#[derive(JsonSchema)]
#[schemars(rename = "ModulePowerHomomorphism")]
struct ModulePowerHomomorphismSchema {
    #[schemars(rename = "type")]
    type_id: String,
    source_map: serde_json::Value,
    kind: String,
    degree: usize,
    source_power: serde_json::Value,
    target_power: serde_json::Value,
    induced: serde_json::Value,
}

#[derive(JsonSchema)]
#[schemars(rename = "PresentedModulePowerHomomorphism")]
struct PresentedModulePowerHomomorphismSchema {
    #[schemars(rename = "type")]
    type_id: String,
    source_map: serde_json::Value,
    kind: String,
    degree: usize,
    source_power: serde_json::Value,
    target_power: serde_json::Value,
    induced: serde_json::Value,
}

#[derive(JsonSchema)]
struct SingletonMapWire {
    #[schemars(rename = "type")]
    type_id: String,
}

#[derive(JsonSchema)]
struct ResidueRingWire {
    #[schemars(rename = "type")]
    type_id: String,
    modulus: String,
}

#[derive(JsonSchema)]
struct FiniteFieldWire {
    #[schemars(rename = "type")]
    type_id: String,
    characteristic: String,
    modulus: Option<Vec<u64>>,
}

#[derive(JsonSchema)]
struct IntegerReductionWire {
    #[schemars(rename = "type")]
    type_id: String,
    target: ResidueRingWire,
}

#[derive(JsonSchema)]
struct IntegerFiniteFieldMapWire {
    #[schemars(rename = "type")]
    type_id: String,
    target: FiniteFieldWire,
}

#[derive(JsonSchema)]
struct PrimeSubfieldEmbeddingWire {
    #[schemars(rename = "type")]
    type_id: String,
    source: FiniteFieldWire,
    target: FiniteFieldWire,
}

#[derive(JsonSchema)]
struct FiniteFieldElementWire {
    #[schemars(rename = "type")]
    type_id: String,
    field: FiniteFieldWire,
    coefficients: Vec<String>,
}

#[derive(JsonSchema)]
struct FiniteFieldEmbeddingWire {
    #[schemars(rename = "type")]
    type_id: String,
    source: FiniteFieldWire,
    target: FiniteFieldWire,
    generator_image: FiniteFieldElementWire,
}

#[derive(JsonSchema)]
struct PolynomialHomomorphismWire {
    #[schemars(rename = "type")]
    type_id: String,
    source: serde_json::Value,
    target: serde_json::Value,
    coefficient_map: serde_json::Value,
    images: Vec<serde_json::Value>,
}

#[derive(JsonSchema)]
struct FiniteRationalFunctionFieldWire {
    #[schemars(rename = "type")]
    type_id: String,
    coefficient_ring: FiniteFieldWire,
    variable: String,
}

#[derive(JsonSchema)]
struct RationalFunctionFieldMapWire {
    #[schemars(rename = "type")]
    type_id: String,
    source: FiniteRationalFunctionFieldWire,
    target: FiniteRationalFunctionFieldWire,
    coefficient_map: PrimeSubfieldEmbeddingWire,
}

#[derive(JsonSchema)]
#[schemars(rename = "EilenbergMacLaneSpace")]
struct EilenbergMacLaneSpaceWire {
    #[schemars(rename = "type")]
    type_id: String,
    group: FinitelyGeneratedAbelianGroupWire,
    delooping_degree: usize,
}

#[derive(JsonSchema)]
#[schemars(rename = "EilenbergMacLaneIntegralHomology")]
struct EilenbergMacLaneIntegralHomologyWire {
    #[schemars(rename = "type")]
    type_id: String,
    space: EilenbergMacLaneSpaceWire,
    max_degree: usize,
    groups: Vec<FinitelyGeneratedAbelianGroupWire>,
    strategy: String,
}

#[derive(JsonSchema)]
#[schemars(rename = "EilenbergMacLaneIntegralCohomology")]
struct EilenbergMacLaneIntegralCohomologyWire {
    #[schemars(rename = "type")]
    type_id: String,
    homology: EilenbergMacLaneIntegralHomologyWire,
    groups: Vec<FinitelyGeneratedAbelianGroupWire>,
}

#[derive(JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EilenbergMacLaneIntegralGeneratorOriginWire {
    FreeFundamentalClass {
        component: usize,
    },
    IntegralBockstein {
        component: usize,
        modulus: String,
    },
    CyclicTorClass {
        left_component: usize,
        right_component: usize,
        order: String,
    },
    SecondDeloopingFundamentalClass {
        component: usize,
    },
}

#[derive(JsonSchema)]
struct EilenbergMacLaneIntegralCohomologyGeneratorWire {
    id: String,
    degree: u32,
    display_name: String,
    origin: EilenbergMacLaneIntegralGeneratorOriginWire,
}

#[derive(JsonSchema)]
struct EilenbergMacLaneIntegralCohomologyRelationTermWire {
    coefficient: String,
    exponents: Vec<u32>,
}

#[derive(JsonSchema)]
struct EilenbergMacLaneIntegralCohomologyRelationWire {
    terms: Vec<EilenbergMacLaneIntegralCohomologyRelationTermWire>,
}

#[derive(JsonSchema)]
#[schemars(rename = "EilenbergMacLaneIntegralCohomologyPresentation")]
struct EilenbergMacLaneIntegralCohomologyPresentationWire {
    #[schemars(rename = "type")]
    type_id: String,
    space: EilenbergMacLaneSpaceWire,
    coefficient_ring: String,
    graded_commutative: bool,
    generators: Vec<EilenbergMacLaneIntegralCohomologyGeneratorWire>,
    relations: Vec<EilenbergMacLaneIntegralCohomologyRelationWire>,
    strategy: String,
}

#[derive(JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum AdmissibleCohomologyOperationWire {
    ModTwo {
        sequence: Vec<u32>,
    },
    OddPrimary {
        prime: u64,
        bocksteins: Vec<bool>,
        reduced_powers: Vec<u32>,
    },
}

#[derive(JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum EilenbergMacLaneGeneratorOriginWire {
    FundamentalClass {
        component: usize,
    },
    Bockstein {
        source_generator: usize,
    },
    PeriodicCyclicClass {
        component: usize,
        primary_exponent: usize,
    },
    AdmissibleSteenrodSquare {
        component: usize,
        operation: AdmissibleCohomologyOperationWire,
        fundamental_class_lifts_integrally: bool,
    },
    AdmissibleOddPrimaryOperation {
        component: usize,
        operation: AdmissibleCohomologyOperationWire,
        fundamental_class_lifts_integrally: bool,
    },
    HigherBockstein {
        component: usize,
        primary_exponent: usize,
    },
    AdmissibleSteenrodSquareOnHigherPrimaryClass {
        component: usize,
        primary_exponent: usize,
        source_is_higher_bockstein: bool,
        operation: AdmissibleCohomologyOperationWire,
    },
    AdmissibleOddPrimaryOperationOnHigherPrimaryClass {
        component: usize,
        primary_exponent: usize,
        source_is_higher_bockstein: bool,
        operation: AdmissibleCohomologyOperationWire,
    },
}

#[derive(JsonSchema)]
struct EilenbergMacLaneCohomologyGeneratorWire {
    id: String,
    degree: u32,
    display_name: String,
    origin: EilenbergMacLaneGeneratorOriginWire,
}

#[derive(JsonSchema)]
struct EilenbergMacLaneCohomologyOperationWire {
    operation: AdmissibleCohomologyOperationWire,
    source_generator: usize,
    target: serde_json::Value,
}

#[derive(JsonSchema)]
#[schemars(rename = "EilenbergMacLaneModularCohomology")]
struct EilenbergMacLaneModularCohomologyWire {
    #[schemars(rename = "type")]
    type_id: String,
    space: EilenbergMacLaneSpaceWire,
    field: serde_json::Value,
    presentation: serde_json::Value,
    generators: Vec<EilenbergMacLaneCohomologyGeneratorWire>,
    operations: Vec<EilenbergMacLaneCohomologyOperationWire>,
    strategy: String,
}

#[derive(JsonSchema)]
#[schemars(rename = "EilenbergMacLaneModularCohomologyTruncation")]
struct EilenbergMacLaneModularCohomologyTruncationWire {
    #[schemars(rename = "type")]
    type_id: String,
    space: EilenbergMacLaneSpaceWire,
    field: serde_json::Value,
    exact_through_degree: usize,
    presentation: serde_json::Value,
    generators: Vec<EilenbergMacLaneCohomologyGeneratorWire>,
    strategy: String,
}

#[derive(JsonSchema)]
#[schemars(rename = "KernelSubmodule")]
struct KernelSubmoduleWire {
    #[schemars(rename = "type")]
    type_id: String,
    homomorphism: serde_json::Value,
    submodule: serde_json::Value,
}

#[derive(JsonSchema)]
#[schemars(rename = "ImageSubmodule")]
struct ImageSubmoduleWire {
    #[schemars(rename = "type")]
    type_id: String,
    homomorphism: serde_json::Value,
    submodule: serde_json::Value,
}

#[derive(JsonSchema)]
#[schemars(rename = "HyperellipticCurve")]
struct HyperellipticCurveSchema {
    #[schemars(rename = "type")]
    type_id: String,
    f: serde_json::Value,
    h: serde_json::Value,
}
#[derive(JsonSchema)]
#[schemars(rename = "HyperellipticFrobeniusEvidence")]
struct HyperellipticEvidenceSchema {
    #[schemars(rename = "type")]
    type_id: String,
    curve: HyperellipticCurveSchema,
    precision: u64,
    working_precision: u64,
    binomial_terms: usize,
    denominator_loss: u64,
    frobenius: serde_json::Value,
    linear_frobenius: serde_json::Value,
}
#[derive(JsonSchema)]
#[schemars(rename = "FrobeniusPolynomial")]
struct FrobeniusPolynomialSchema {
    #[schemars(rename = "type")]
    type_id: String,
    polynomial: serde_json::Value,
    field_order: String,
    cohomological_degree: u32,
    cohomological_weight: u32,
    cohomology: Option<serde_json::Value>,
    /// Exact extension counts retained by H1 or H2 Newton reconstruction.
    point_counts: Option<serde_json::Value>,
    hyperelliptic: Option<HyperellipticEvidenceSchema>,
}
#[derive(JsonSchema)]
#[schemars(rename = "ZetaFunction")]
struct ZetaFunctionSchema {
    #[schemars(rename = "type")]
    type_id: String,
    numerator: serde_json::Value,
    denominator: serde_json::Value,
    frobenius: FrobeniusPolynomialSchema,
}

static ACCEPTED_TYPE_SCHEMAS: &[TypeSchemaRegistration] = &[
    registration!(HyperellipticCurve, HyperellipticCurveSchema),
    registration!(HyperellipticFrobeniusEvidence, HyperellipticEvidenceSchema),
    registration!(FrobeniusPolynomial, FrobeniusPolynomialSchema),
    registration!(ZetaFunction, ZetaFunctionSchema),
    registration!(IntegerRationalEmbedding, SingletonMapWire),
    registration!(IntegerReduction, IntegerReductionWire),
    registration!(IntegerFiniteFieldMap, IntegerFiniteFieldMapWire),
    registration!(PrimeSubfieldEmbedding, PrimeSubfieldEmbeddingWire),
    registration!(FiniteFieldEmbedding, FiniteFieldEmbeddingWire),
    registration!(RegisteredPolynomialHomomorphism, PolynomialHomomorphismWire),
    registration!(
        RegisteredRationalFunctionFieldMap,
        RationalFunctionFieldMapWire
    ),
    registration!(SteinerCategory, SteinerCategoryWire),
    registration!(KashiwaraCrystal, KashiwaraCrystalWire),
    registration!(Graph, GraphWire),
    registration!(Matroid, MatroidWire),
    registration!(Partition, PartitionWire),
    registration!(Composition, CompositionWire),
    registration!(Poset, PosetWire),
    registration!(SymmetricFunction, SymmetricFunctionWire),
    registration!(Tableau, TableauWire),
    registration!(WeightLattice, WeightLatticeSchema),
    registration!(WeightLatticeMap, WeightLatticeMapSchema),
    registration!(WeightCharacter, WeightCharacterSchema),
    registration!(PolynomialGlRepresentation, PolynomialGlRepresentationSchema),
    registration!(Tree, TreeWire),
    registration!(ImageSubmodule<IntegerRing>, ImageSubmoduleWire),
    registration!(KernelSubmodule<IntegerRing>, KernelSubmoduleWire),
    registration!(RegisteredExteriorAlgebra, ExteriorAlgebraSchema),
    registration!(RegisteredExteriorElement, ExteriorElementSchema),
    registration!(
        FinitelyGeneratedAbelianGroup,
        FinitelyGeneratedAbelianGroupWire
    ),
    registration!(
        RegisteredModulePowerHomomorphism,
        ModulePowerHomomorphismSchema
    ),
    registration!(
        RegisteredPresentedModulePowerHomomorphism,
        PresentedModulePowerHomomorphismSchema
    ),
    registration!(EilenbergMacLaneSpace, EilenbergMacLaneSpaceWire),
    registration!(
        EilenbergMacLaneIntegralHomology,
        EilenbergMacLaneIntegralHomologyWire
    ),
    registration!(
        EilenbergMacLaneIntegralCohomology,
        EilenbergMacLaneIntegralCohomologyWire
    ),
    registration!(
        EilenbergMacLaneIntegralCohomologyPresentation,
        EilenbergMacLaneIntegralCohomologyPresentationWire
    ),
    registration!(
        EilenbergMacLaneModularCohomology,
        EilenbergMacLaneModularCohomologyWire
    ),
    registration!(
        EilenbergMacLaneModularCohomologyTruncation,
        EilenbergMacLaneModularCohomologyTruncationWire
    ),
];

/// Accepted semantic wire types, in a stable facade-owned order.
#[must_use]
pub const fn accepted_type_schemas() -> &'static [TypeSchemaRegistration] {
    ACCEPTED_TYPE_SCHEMAS
}

/// Look up an accepted semantic type by its versioned identity.
#[must_use]
pub fn type_schema(type_id: &str) -> Option<TypeSchemaRegistration> {
    ACCEPTED_TYPE_SCHEMAS
        .iter()
        .copied()
        .find(|entry| entry.description().semantic_id == type_id)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn accepted_type_ids_and_paths_are_unique() {
        let mut ids = HashSet::new();
        let mut rust_paths = HashSet::new();
        let mut python_paths = HashSet::new();
        for registration in accepted_type_schemas() {
            let description = registration.description();
            assert!(ids.insert(description.semantic_id));
            assert!(rust_paths.insert(description.rust_path));
            assert!(python_paths.insert(description.python_path));
            assert_eq!(description.semantic_id.rsplit_once('@').unwrap().1, "1");
            assert!(registration.schema().is_object());
        }
    }

    #[test]
    fn lookup_is_exact_and_rejects_unaccepted_semantic_types() {
        assert!(type_schema(Graph::TYPE_ID).is_some());
        assert!(type_schema(FinitelyGeneratedAbelianGroup::TYPE_ID).is_some());
        assert!(type_schema(EilenbergMacLaneSpace::TYPE_ID).is_some());
        assert!(type_schema("noether/combin/Graph").is_none());
        assert!(type_schema("noether/numthy/PadicRing@1").is_none());
    }
}
