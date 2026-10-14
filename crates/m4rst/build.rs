// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Build script for m4rst.
//!
//! Default builds do nothing — m4rst is pure Rust with no native link
//! dependency.
//!
//! When the `validate-libm4ri` feature is enabled, this compiles the
//! `oracle/m4ri_oracle.c` shim and links libm4ri so the
//! `parity_libm4ri` test (and the `bench_*` examples) can bit-check
//! and time m4rst against the reference implementation. libm4ri is
//! GPL-2 — this is the single point that links it, and it never
//! enters a shipped build.
//!
//! On Debian/Ubuntu the system `libm4ri-dev` package puts the header
//! and library on the default search paths. Point `M4RI_PREFIX` at a
//! custom install (expects `$M4RI_PREFIX/include` + `$M4RI_PREFIX/lib`)
//! — e.g. a thread-safe libm4ri build — to override.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(coverage)");
    println!("cargo:rerun-if-env-changed=M4RI_PREFIX");
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_VALIDATE_LIBM4RI").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=oracle/m4ri_oracle.c");

    let mut build = cc::Build::new();
    build.file("oracle/m4ri_oracle.c");
    build.flag_if_supported("-Wno-unused-parameter");

    if let Ok(prefix) = std::env::var("M4RI_PREFIX") {
        build.include(format!("{prefix}/include"));
        println!("cargo:rustc-link-search=native={prefix}/lib");
    }
    // System headers (Debian/Ubuntu `libm4ri-dev`) are on the default
    // include path; the `.include` above only adds a custom prefix
    // when one is set.
    build.compile("m4rst_m4ri_oracle");

    // On Debian/Ubuntu libm4ri lives in the default search path, so no
    // extra `-L` is needed unless M4RI_PREFIX was set above.
    println!("cargo:rustc-link-lib=dylib=m4ri");
}
