// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Build script for the Noether Python extension.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Wheels place audited native libraries beside the extension. Local
    // non-relocatable builds additionally use the configured FLINT prefix.
    let origin = if cfg!(target_os = "macos") {
        "@loader_path"
    } else {
        "$ORIGIN"
    };
    println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{origin}/../tertius_noether.libs");

    if let (false, Ok(flint_prefix)) = (
        cfg!(feature = "extension-module"),
        std::env::var("FLINT_PREFIX"),
    ) {
        println!("cargo:rerun-if-env-changed=FLINT_PREFIX");
        let flint_lib = format!("{flint_prefix}/lib");
        if std::path::Path::new(&flint_lib).is_dir() {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{flint_lib}");
        }
        let so_at_root = std::fs::read_dir(&flint_prefix)
            .map(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .file_name()
                        .to_str()
                        .is_some_and(|name| name.starts_with("libflint.so"))
                })
            })
            .unwrap_or(false);
        if so_at_root {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{flint_prefix}");
        }
    }
}
