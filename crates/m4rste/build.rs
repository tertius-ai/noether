// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Build script for m4rste.
//!
//! Default builds do nothing — m4rste is pure Rust with no native
//! link dependency.
//!
//! When the `validate-libm4rie` feature is enabled, this compiles the
//! `oracle/m4rie_oracle.c` shim and links libm4rie + libm4ri so the
//! `bench_vs_m4rie` example can time and bit-check m4rste against the
//! reference implementation. libm4rie is GPL-2 — this is the single
//! point that links it, and it never enters a shipped build.
//!
//! Header / library locations default to the Debian/Ubuntu system
//! layout (`/usr/include/m4rie`, `/usr/lib/x86_64-linux-gnu`).
//! Override the lib search path with `M4RIE_PREFIX` (expects
//! `$M4RIE_PREFIX/lib` and `$M4RIE_PREFIX/include`).

fn main() {
    println!("cargo:rerun-if-env-changed=M4RIE_PREFIX");
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_VALIDATE_LIBM4RIE").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=oracle/m4rie_oracle.c");

    let mut build = cc::Build::new();
    build.file("oracle/m4rie_oracle.c");
    build.flag_if_supported("-Wno-unused-parameter");

    if let Ok(prefix) = std::env::var("M4RIE_PREFIX") {
        build.include(format!("{prefix}/include"));
        println!("cargo:rustc-link-search=native={prefix}/lib");
    }
    // System headers (Debian/Ubuntu `libm4rie-dev` / `libm4ri-dev`)
    // are on the default include path; the `.include` above only adds
    // a custom prefix when one is set.
    build.compile("m4rste_m4rie_oracle");

    // libm4rie depends on libm4ri; link both. On Debian/Ubuntu these
    // live in the default search path, so no extra `-L` is needed
    // unless M4RIE_PREFIX was set above.
    println!("cargo:rustc-link-lib=dylib=m4rie");
    println!("cargo:rustc-link-lib=dylib=m4ri");
}
