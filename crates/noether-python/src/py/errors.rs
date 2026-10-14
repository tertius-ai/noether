// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Canonical Python exception hierarchy for the public facade.

use noether::{
    ContextMismatchError as RustContextMismatchError, NoetherError as RustNoetherError,
    UnsupportedOperationError as RustUnsupportedOperationError,
    VerificationError as RustVerificationError,
};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

create_exception!(
    noether,
    NoetherError,
    PyException,
    "Base exception for failures reported by Noether's public facade."
);
#[cfg(feature = "distributed")]
create_exception!(
    noether,
    DistributedError,
    NoetherError,
    "A distributed Noether operation failed."
);
create_exception!(
    noether,
    UnsupportedOperationError,
    NoetherError,
    "The requested operation is unavailable for a mathematical context."
);
create_exception!(
    noether,
    ContextMismatchError,
    NoetherError,
    "Values from incompatible mathematical contexts were combined."
);
create_exception!(
    noether,
    VerificationError,
    NoetherError,
    "A mathematical witness failed explicit verification."
);

/// Register the public exception classes on the private transport module.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let python = module.py();
    module.add("NoetherError", python.get_type::<NoetherError>())?;
    module.add(
        "UnsupportedOperationError",
        python.get_type::<UnsupportedOperationError>(),
    )?;
    module.add(
        "ContextMismatchError",
        python.get_type::<ContextMismatchError>(),
    )?;
    module.add("VerificationError", python.get_type::<VerificationError>())?;
    #[cfg(feature = "distributed")]
    {
        let error = python.get_type::<DistributedError>();
        error.setattr("__module__", "noether.linalg")?;
        module.add("DistributedError", error)?;
    }
    Ok(())
}

/// Translate a distributed facade failure without exposing MPI implementation
/// objects through Python.
#[cfg(feature = "distributed")]
pub fn distributed(error: noether::linalg::DistributedError) -> PyErr {
    use noether::linalg::DistributedError as RustDistributedError;

    match error {
        RustDistributedError::UnsupportedCoefficientContext { context } => unsupported_operation(
            RustUnsupportedOperationError::new("noether/linalg/DistributedMatrix@1", context)
                .with_supported_contexts(["noether/numthy/FiniteField/F2@1"]),
        ),
        RustDistributedError::ContextMismatch { expected, actual } => {
            context_mismatch(RustContextMismatchError { expected, actual })
        }
        RustDistributedError::Kernel(error) => DistributedError::new_err(error.to_string()),
    }
}

/// Translate a public-facade error without losing its typed Python category.
pub fn noether_error(error: RustNoetherError) -> PyErr {
    match error {
        RustNoetherError::ContextMismatch(error) => context_mismatch(error),
        RustNoetherError::Verification(error) => verification(error),
        RustNoetherError::UnsupportedOperation(error) => unsupported_operation(error),
        RustNoetherError::Serialization(message) => {
            NoetherError::new_err(format!("serialization failed: {message}"))
        }
    }
}

/// Translate a context mismatch to its canonical Python exception.
pub fn context_mismatch(error: RustContextMismatchError) -> PyErr {
    let python_error = ContextMismatchError::new_err(error.to_string());
    let attributes = Python::attach(|python| {
        let value = python_error.value(python);
        value.setattr("expected", &error.expected)?;
        value.setattr("actual", &error.actual)
    });
    attributes.err().unwrap_or(python_error)
}

/// Translate a verification failure to its canonical Python exception.
pub fn verification(error: RustVerificationError) -> PyErr {
    let python_error = VerificationError::new_err(error.to_string());
    let attributes = Python::attach(|python| {
        let value = python_error.value(python);
        match &error {
            RustVerificationError::ContextMismatch(mismatch) => {
                value.setattr("kind", "context_mismatch")?;
                value.setattr("expected", &mismatch.expected)?;
                value.setattr("actual", &mismatch.actual)
            }
            RustVerificationError::InvalidWitness { claim } => {
                value.setattr("kind", "invalid_witness")?;
                value.setattr("claim", claim)
            }
        }
    });
    attributes.err().unwrap_or(python_error)
}

/// Translate an unavailable capability to its canonical Python exception.
pub fn unsupported_operation(error: RustUnsupportedOperationError) -> PyErr {
    let python_error = UnsupportedOperationError::new_err(error.to_string());
    let attributes = Python::attach(|python| {
        let value = python_error.value(python);
        value.setattr("operation", &error.operation)?;
        value.setattr("context", &error.context)?;
        value.setattr("detail", &error.detail)?;
        value.setattr(
            "supported_contexts",
            PyTuple::new(python, &error.supported_contexts)?,
        )
    });
    attributes.err().unwrap_or(python_error)
}

#[cfg(test)]
mod tests {
    use noether::{NoetherError as RustNoetherError, UnsupportedOperationError as RustError};
    use pyo3::types::PyAnyMethods;
    use pyo3::Python;

    use super::{noether_error, NoetherError, UnsupportedOperationError};

    #[test]
    fn unsupported_operation_keeps_the_python_error_category() {
        Python::initialize();
        Python::attach(|python| {
            let error = noether_error(RustNoetherError::from(
                RustError::new("noether/linalg/Matrix/svd@1", "noether/root/IntegerRing@1")
                    .with_supported_contexts(["noether/root/RealBallField@1"]),
            ));
            assert!(error.is_instance_of::<UnsupportedOperationError>(python));
            assert!(error.is_instance_of::<NoetherError>(python));
            let value = error.value(python);
            assert!(matches!(
                value
                    .getattr("operation")
                    .and_then(|value| value.extract::<String>()),
                Ok(operation) if operation == "noether/linalg/Matrix/svd@1"
            ));
            assert!(matches!(
                value
                    .getattr("supported_contexts")
                    .and_then(|value| value.extract::<Vec<String>>()),
                Ok(contexts) if contexts == ["noether/root/RealBallField@1"]
            ));
            assert!(matches!(
                value
                    .getattr("detail")
                    .and_then(|value| value.extract::<Option<String>>()),
                Ok(None)
            ));
        });
    }
}
