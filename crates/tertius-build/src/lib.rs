// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shared build-script helpers for Tertius crates that link native libraries.
//!
//! This crate is a **build-dependency only** — never a runtime/shipped
//! dependency, so adopting it does not add to a crate's shipped dependency set
//! (it is compatible with the pure-Rust kernels' "no Rust deps in the shipped
//! crate" invariant). Its job is to give every native-linking `build.rs` ONE
//! implementation of prefix resolution + link emission, so empty-`*_PREFIX`
//! handling can't drift between crates.
//!
//! `.cargo/config.toml` may declare an empty native prefix, which means unset.
//! [`resolve_prefix`] makes that rule the single source of truth.

use std::env;
use std::path::{Path, PathBuf};

/// How a native library is linked.
pub enum LinkKind {
    /// Dynamic linkage (`-l dylib=...`).
    Dylib,
    /// Static linkage (`-l static=...`).
    Static,
}

impl LinkKind {
    fn as_str(&self) -> &'static str {
        match self {
            LinkKind::Dylib => "dylib",
            LinkKind::Static => "static",
        }
    }
}

/// Resolve a `*_PREFIX` env var, treating an **empty** value the same as unset
/// and falling back to `default_prefix`. Emits the matching
/// `cargo:rerun-if-env-changed`.
///
/// The empty-as-unset rule is the point: `.cargo/config.toml` declares e.g.
/// `FLINT_PREFIX = ""`, so an `env::var(..).unwrap_or_else(|_| default)` — which
/// only falls back on the `Err` (unset) case — keeps the empty string and emits
/// an invalid empty `-L`.
pub fn resolve_prefix(env_var: &str, default_prefix: &str) -> String {
    println!("cargo:rerun-if-env-changed={env_var}");
    env::var(env_var)
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_prefix.to_string())
}

/// Emit the standard link directives for a native library found under a
/// `*_PREFIX`: a search path, the link-lib line, the `$ORIGIN`-relative wheel
/// layout (`<pkg>/../tertius_noether.libs`, populated by
/// `scripts/install-native-deps.sh`), and an absolute development rpath unless
/// `TERTIUS_RELOCATABLE` is set. Relocatable artifacts therefore contain no
/// build-host path, while in-tree `cargo test`/`bench` binaries resolve the
/// shared object without `LD_LIBRARY_PATH`.
///
/// The caller owns the fallback *policy* (`default_prefix`): a dev-box path, a
/// platform default, whatever fits. This helper only unifies the *mechanics*
/// (empty-guard + emission), so it never silently changes which library a
/// crate links — it just fixes the empty-prefix handling everywhere.
///
/// - `env_var`: prefix env var, e.g. `"FLINT_PREFIX"`.
/// - `default_prefix`: fallback when `env_var` is unset/empty.
/// - `lib`: library name for `-l`, e.g. `"flint"`.
/// - `kind`: [`LinkKind::Dylib`] or [`LinkKind::Static`].
/// - `lib_in_subdir`: `true` → the lib is in `{prefix}/lib`, else in `{prefix}`.
///
/// Returns the resolved link-search directory.
pub fn link_native(
    env_var: &str,
    default_prefix: &str,
    lib: &str,
    kind: LinkKind,
    lib_in_subdir: bool,
) -> String {
    let prefix = resolve_prefix(env_var, default_prefix);
    let dir = if lib_in_subdir {
        format!("{prefix}/lib")
    } else {
        prefix
    };
    emit_link(&dir, &[(lib, kind)]);
    dir
}

/// Emit the standard link directives for one or more native libraries that
/// live in an **already-resolved** directory `dir`: a search path, the
/// `$ORIGIN`-relative wheel rpath, an absolute development rpath unless
/// `TERTIUS_RELOCATABLE` is set, and one `-l` line per `(lib, kind)` in `libs`.
///
/// This is the emission half of [`link_native`] — [`link_native`] resolves a
/// `*_PREFIX` env var into a `dir` and calls this. Callers whose prefix /
/// lib-dir *resolution* is bespoke (a non-`*_PREFIX` env var, a feature gate, a
/// distro/Homebrew fallback ladder, a GPL warning) keep that resolution but
/// share this emission, so the search/rpath/link mechanics never drift between
/// crates. Several native deps ship
/// more than one shared object from the same dir (SuiteSparse's
/// cholmod/amd/colamd/suitesparseconfig, MFEM's mfem/HYPRE/mpi/…), hence the
/// slice of libs.
///
/// The rpaths are emitted once regardless of how many libs are linked; the
/// `-l` lines follow, matching the order the multi-lib crates already used.
pub fn emit_link(dir: &str, libs: &[(&str, LinkKind)]) {
    println!("cargo:rerun-if-env-changed=TERTIUS_RELOCATABLE");
    println!("cargo:rustc-link-search=native={dir}");
    let origin = if cfg!(target_os = "macos") {
        "@loader_path"
    } else {
        "$ORIGIN"
    };
    println!("cargo:rustc-link-arg=-Wl,-rpath,{origin}/../tertius_noether.libs");
    if env::var_os("TERTIUS_RELOCATABLE").is_none() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    for (lib, kind) in libs {
        println!("cargo:rustc-link-lib={}={lib}", kind.as_str());
    }
}

/// Emit only the development rpath for an already-linked native library.
///
/// This is for top-level consumers of a `*-sys` crate. The sys crate owns the
/// link declaration; the final binary owns the loader policy. Relocatable
/// artifacts omit the build-host path.
pub fn emit_development_rpath(dir: &Path) {
    println!("cargo:rerun-if-env-changed=TERTIUS_RELOCATABLE");
    if env::var_os("TERTIUS_RELOCATABLE").is_none() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

/// The FLINT prefix fallback when `FLINT_PREFIX` is unset/empty.
///
/// Prefers the **managed install** `scripts/install-flint.sh` records: it walks
/// up from the building crate to the repo root and reads `.deps/flint/prefix.txt`
/// (then the `.deps/flint` dir) — so a plain `cargo` build follows the same FLINT
/// the `make`/CI/gate flow uses, rather than a hardcoded outside folder. Falls
/// back to the platform system prefix (Homebrew on macOS, `/usr` on Linux) when
/// no managed install is found.
pub fn flint_default_prefix() -> String {
    // Walk up from the building crate (or the invocation dir) to find the
    // in-repo FLINT install under `.deps/flint`.
    let start = env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| env::current_dir().ok());
    if let Some(start) = start {
        for dir in start.ancestors() {
            let prefix = dir.join(".deps").join("flint");
            if let Ok(s) = std::fs::read_to_string(prefix.join("prefix.txt")) {
                let s = s.trim();
                if !s.is_empty() {
                    return s.to_string();
                }
            }
            if prefix.exists() {
                return prefix.to_string_lossy().into_owned();
            }
        }
    }
    if cfg!(target_os = "macos") {
        env::var("HOMEBREW_PREFIX").unwrap_or_else(|_| "/opt/homebrew".to_string())
    } else {
        "/usr".to_string()
    }
}

/// Resolve + link libflint robustly, independent of any single outside folder.
///
/// Prefix = `FLINT_PREFIX` (if non-empty) else [`flint_default_prefix`]. The lib
/// **directory** is then probed (`{prefix}/lib`, then `{prefix}`) so both the
/// install layout (lib in `/lib`) and an in-tree FLINT build (lib at the prefix
/// root) link without the caller committing to one. When called by a consumer
/// of `flint3-sys`, its `DEP_FLINT_LIB_DIR` metadata is authoritative.
pub fn flint_library_dir() -> PathBuf {
    println!("cargo:rerun-if-env-changed=DEP_FLINT_LIB_DIR");
    if let Some(dir) = env::var_os("DEP_FLINT_LIB_DIR").filter(|s| !s.is_empty()) {
        return PathBuf::from(dir);
    }
    let prefix = resolve_prefix("FLINT_PREFIX", &flint_default_prefix());
    let p = Path::new(&prefix);
    [p.join("lib"), p.to_path_buf()]
        .into_iter()
        .find(|d| dir_has_libflint(d))
        .unwrap_or_else(|| p.join("lib"))
}

/// Resolve + link libflint robustly, independent of any single outside folder.
pub fn link_flint() {
    let dir = flint_library_dir();
    emit_link(&dir.to_string_lossy(), &[("flint", LinkKind::Dylib)]);
}

fn dir_has_libflint(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .any(|e| e.file_name().to_string_lossy().starts_with("libflint."))
        })
        .unwrap_or(false)
}
