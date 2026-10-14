// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! F4, signature, modular, and integral strong Gröbner kernels.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod engine;
mod fglm;
mod field;
mod hilbert;
mod integer;
mod module;
mod primitive;
mod rational;
mod reducer;
mod rur;

pub use engine::{
    groebner_basis, groebner_basis_binary_extension, groebner_basis_f2, groebner_basis_prime,
    groebner_basis_prime_profiled, groebner_basis_with_transform, polynomial_normal_forms,
    unit_ideal_certificate, Algorithm, GroebnerError, GroebnerEvidence, GroebnerMetrics,
    GroebnerOptions, GroebnerOutput, GroebnerPhaseTimings, GroebnerTransformOutput,
    PreparedPolynomialReducer, RationalBackend, RationalTransformationBackend, RowBackend,
    UnitIdealCertificateOutput,
};
pub use fglm::{
    fglm_basis, finite_quotient_algebra, FglmMetrics, FglmOutput, FiniteQuotientAlgebraOutput,
};
pub use field::{
    BinaryExtensionField16, BinaryExtensionFieldError, BinaryField, Field, PrimeField32,
    PrimeFieldError,
};
pub use flint_core::Integer;
pub use flint_core::Rational;
pub use integer::{
    strong_groebner_basis, strong_normal_form, strong_normal_form_with_quotients,
    IntegerCoefficientRing, StrongGroebnerMetrics, StrongGroebnerOptions, StrongGroebnerOutput,
    StrongNormalFormOutput,
};
pub use module::{
    module_groebner_basis, module_groebner_basis_binary_extension, module_groebner_basis_f2,
    module_groebner_basis_prime, module_groebner_basis_rational,
    module_groebner_basis_with_transform, module_groebner_basis_with_transform_binary_extension,
    module_groebner_basis_with_transform_f2, module_groebner_basis_with_transform_prime,
    module_groebner_basis_with_transform_rational, module_leading_terms, module_membership,
    module_normal_forms, module_relations_from_groebner_data,
    module_relations_from_groebner_data_binary_extension, module_relations_from_groebner_data_f2,
    module_relations_from_groebner_data_prime, module_schreyer_syzygies,
    module_schreyer_syzygies_binary_extension,
    module_schreyer_syzygies_binary_extension_with_hilbert_target, module_schreyer_syzygies_f2,
    module_schreyer_syzygies_f2_with_hilbert_target, module_schreyer_syzygies_prime,
    module_schreyer_syzygies_prime_with_hilbert_target,
    module_schreyer_syzygies_with_hilbert_target, strong_module_groebner_basis,
    strong_module_normal_form, strong_module_normal_form_with_quotients, strong_module_relations,
    ModuleBasisOutput, ModuleGroebnerOutput, ModuleKernelOutput, ModuleNormalFormOutput,
    ModuleSyzygyOutput, SparseModuleElement, SparseModuleOrder, SparseModuleTerm,
    StandardModuleHilbertTarget, StrongModuleGroebnerOutput, StrongModuleKernelOutput,
    StrongModuleNormalFormOutput,
};
pub use poly_core::{Monomial, MonomialOrder, SparsePolynomial};
pub use rational::{
    groebner_basis_rational, groebner_basis_rational_profiled,
    groebner_basis_with_transform_rational, macaulay_unit_ideal_certificate_rational,
    rational_macaulay_certificate_degree, unit_ideal_certificate_rational,
};
pub use rur::{rational_univariate_representation, RurMetrics, RurOutput};
