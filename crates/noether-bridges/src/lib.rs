// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! `noether-bridges` — arm's-length backends to external GPL math tools.
//!
//! This crate is the **single audit point** for Tertius's calls into GPL
//! software (Macaulay2, Singular, and msolve). The engine core
//! links and spawns no GPL; every GPL touch lives here, behind a feature
//! gate, so a `no-gpl` build simply does not compile it in. See the GPL
//! posture in the repo-root `AGENTS.md` (cross-cutting invariant 1).
//!
//! Each backend is feature-gated and spawns a user-installed executable over
//! a textual interface. The crate links no GPL library and is therefore clean
//! to convey; a build with all features disabled contains only the shared
//! error vocabulary.
//!
//! ## Composition
//!
//! Native-vs-GPL *selection* is the caller's job (the engine's dispatch /
//! the caller's playbook), not this crate's — these functions are raw process
//! backends. Results are returned in the tool's native convention; the Noether
//! Rust facade owns operation-specific parsing, canonicalization, and exact
//! verification. Independent subprocesses are parallel-safe.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::fmt;
#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
use std::process::Command;

pub mod macaulay2;
pub mod msolve;
pub mod singular;

/// Failure modes common to every subprocess bridge.
#[derive(Debug)]
pub enum BridgeError {
    /// The external tool is not installed / not on `PATH`. Callers treat
    /// this as "this backend is unavailable in this environment", not as a
    /// computation failure — the engine surfaces it as the typed
    /// unavailable error and the dispatch falls through.
    Unavailable(String),
    /// Failed to spawn the subprocess or to read/write its pipes.
    Spawn(std::io::Error),
    /// The subprocess ran but exited non-zero.
    NonZeroExit {
        /// Process exit status, or `None` when the adapter killed it after a
        /// timeout.
        code: Option<i32>,
        /// Captured diagnostic output from the external tool.
        stderr: String,
    },
    /// The subprocess output could not be parsed into the expected shape.
    Parse(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::Unavailable(s) => write!(f, "bridge unavailable: {s}"),
            BridgeError::Spawn(e) => write!(f, "bridge subprocess I/O error: {e}"),
            BridgeError::NonZeroExit { code, stderr } => {
                write!(f, "bridge subprocess exited {code:?}: {stderr}")
            }
            BridgeError::Parse(s) => write!(f, "bridge output parse error: {s}"),
        }
    }
}

impl std::error::Error for BridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BridgeError::Spawn(e) => Some(e),
            _ => None,
        }
    }
}

/// Prevent an in-process native stack from changing the libraries resolved by
/// an arm's-length executable. The external program must use the ABI stack it
/// was installed against, not Noether's build or preload paths.
#[cfg(any(
    feature = "msolve-subprocess",
    feature = "macaulay2-subprocess",
    feature = "singular-subprocess",
    test
))]
pub(crate) fn isolate_native_libraries(command: &mut Command) {
    command
        .env_remove("LD_PRELOAD")
        .env_remove("LD_LIBRARY_PATH")
        .env_remove("DYLD_INSERT_LIBRARIES")
        .env_remove("DYLD_LIBRARY_PATH");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subprocess_native_library_injection_is_removed() {
        let mut command = Command::new("unused");
        command
            .env("LD_PRELOAD", "preload")
            .env("LD_LIBRARY_PATH", "linux-path")
            .env("DYLD_INSERT_LIBRARIES", "insert")
            .env("DYLD_LIBRARY_PATH", "macos-path");
        isolate_native_libraries(&mut command);
        let environment = command
            .get_envs()
            .map(|(key, value)| (key.to_owned(), value.map(ToOwned::to_owned)))
            .collect::<std::collections::HashMap<_, _>>();
        for key in [
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
        ] {
            assert_eq!(environment.get(std::ffi::OsStr::new(key)), Some(&None));
        }
    }
}
