// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Development loader path for FLINT-backed exact polytope arithmetic.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let dir = tertius_build::flint_library_dir();
    tertius_build::emit_development_rpath(&dir);
}
