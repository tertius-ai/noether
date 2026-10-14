// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Development loader path for the optional FLINT comparison benchmark.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_FLINT").is_some() {
        let dir = tertius_build::flint_library_dir();
        tertius_build::emit_development_rpath(&dir);
    }
}
