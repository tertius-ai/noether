// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Build facts consumed by the typed public discovery surface.

use pyo3::prelude::*;

/// Translate the Rust facade's complete accepted public-name index.
#[pyfunction(name = "_accepted_surface")]
pub fn accepted_surface() -> (
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
) {
    (
        noether::accepted_domains().to_vec(),
        noether::accepted_public_types().to_vec(),
        noether::accepted_operations().to_vec(),
        noether::accepted_public_values().to_vec(),
    )
}

/// Translate the public-name index available in this compiled artifact.
#[pyfunction(name = "_available_surface")]
pub fn available_surface() -> (
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
    Vec<&'static str>,
) {
    (
        noether::available_domains(),
        noether::available_public_types(),
        noether::available_operations(),
        noether::available_public_values(),
    )
}

/// Return every optional engine feature and whether this artifact contains it.
///
/// Keeping this in the compiled transport makes `capabilities()` describe the
/// installed wheel rather than the source checkout that happens to surround it.
#[pyfunction(name = "_facade_build_features")]
pub fn build_features() -> Vec<(&'static str, bool)> {
    let mut backends = noether::COMPILED_BACKENDS
        .iter()
        .copied()
        .filter(|name| *name != "m4rst-distributed")
        .map(|name| (name, true))
        .collect::<Vec<_>>();
    backends.extend([
        ("m4rst-distributed", cfg!(feature = "distributed")),
        ("msolve", cfg!(feature = "msolve-subprocess")),
        ("macaulay2", cfg!(feature = "macaulay2-subprocess")),
        ("singular", cfg!(feature = "singular-subprocess")),
    ]);
    backends
}

/// Translate the Rust facade's compiled optional-feature index.
#[pyfunction(name = "_compiled_features")]
pub fn compiled_features() -> Vec<&'static str> {
    noether::COMPILED_FEATURES.to_vec()
}

/// Return MPI runtime facts only after an explicit world has been initialized.
#[pyfunction(name = "_mpi_runtime")]
pub fn mpi_runtime() -> Option<(String, usize, usize, &'static str)> {
    #[cfg(feature = "distributed")]
    {
        let runtime = noether::linalg::active_mpi_runtime()?;
        let support = match runtime.thread_support {
            noether::linalg::MpiThreadSupport::Single => "single",
            noether::linalg::MpiThreadSupport::Funneled => "funneled",
            noether::linalg::MpiThreadSupport::Serialized => "serialized",
            noether::linalg::MpiThreadSupport::Multiple => "multiple",
        };
        Some((
            runtime.library_version,
            runtime.world_size,
            runtime.mpi_rank,
            support,
        ))
    }
    #[cfg(not(feature = "distributed"))]
    {
        None
    }
}

/// Translate the facade-owned process-local concurrency policy.
#[pyfunction(name = "_local_concurrency")]
pub fn local_concurrency() -> (
    usize,
    Option<usize>,
    usize,
    bool,
    &'static str,
    &'static str,
    &'static str,
) {
    let policy = noether::local_concurrency();
    (
        policy.schedulable_threads(),
        policy.requested_threads(),
        policy.effective_threads(),
        policy.override_valid(),
        policy.affinity_source(),
        policy.override_name(),
        policy.phase_owner(),
    )
}

/// Register private transport facts used by `noether.capabilities()`.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(accepted_surface, module)?)?;
    module.add_function(wrap_pyfunction!(available_surface, module)?)?;
    module.add_function(wrap_pyfunction!(build_features, module)?)?;
    module.add_function(wrap_pyfunction!(compiled_features, module)?)?;
    module.add_function(wrap_pyfunction!(mpi_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(local_concurrency, module)?)
}

#[cfg(test)]
mod tests {
    use super::{
        accepted_surface, available_surface, build_features, compiled_features, local_concurrency,
    };

    #[test]
    fn every_feature_name_is_unique() {
        let features = build_features();
        for (index, (name, _)) in features.iter().enumerate() {
            assert!(!features[..index].iter().any(|(earlier, _)| earlier == name));
        }
    }

    #[test]
    fn accepted_surface_is_a_literal_projection_of_the_facade() {
        let (domains, types, operations, values) = accepted_surface();
        assert_eq!(domains, noether::accepted_domains());
        assert_eq!(types, noether::accepted_public_types());
        assert_eq!(operations, noether::accepted_operations());
        assert_eq!(values, noether::accepted_public_values());
    }

    #[test]
    fn available_surface_is_a_literal_projection_of_the_facade() {
        let (domains, types, operations, values) = available_surface();
        assert_eq!(domains, noether::available_domains());
        assert_eq!(types, noether::available_public_types());
        assert_eq!(operations, noether::available_operations());
        assert_eq!(values, noether::available_public_values());
    }

    #[test]
    fn compiled_features_are_a_literal_projection_of_the_facade() {
        assert_eq!(compiled_features(), noether::COMPILED_FEATURES);
    }

    #[test]
    fn concurrency_is_a_literal_projection_of_the_facade() {
        let policy = noether::local_concurrency();
        let projected = local_concurrency();
        assert_eq!(projected.0, policy.schedulable_threads());
        assert_eq!(projected.1, policy.requested_threads());
        assert_eq!(projected.2, policy.effective_threads());
        assert_eq!(projected.3, policy.override_valid());
        assert_eq!(projected.4, policy.affinity_source());
        assert_eq!(projected.5, policy.override_name());
        assert_eq!(projected.6, policy.phase_owner());
    }
}
