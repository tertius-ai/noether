#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
"""Check every workspace crate under ``crates/`` against its release rules.

    scripts/check-crates.py            # report
    scripts/check-crates.py --strict   # exit 1 on any violation

The normative policies are ``AGENTS.md`` and ``crates/AGENTS.md``. This script
covers only their mechanical, source-verifiable subset and distinguishes the
public facade and private PyO3 transport from kernel crates.
"""

from __future__ import annotations

import argparse
import pathlib
import re

import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
CRATES = REPO / "crates"

# Binding and build-only crates are exempt from public-rustdoc and integration
# test rules. Every crate still carries a concise human README.
SYS_SUFFIX = "-sys"
BUILD_ONLY = {"tertius-build"}
PRIVATE_TRANSPORT = {"noether-python"}
AGENT_OPTIONAL = {"spar-core", "lattice-enum", "tertius-build"}
WORKSPACE_PACKAGE = (
    tomllib.loads((REPO / "Cargo.toml").read_text()).get("workspace", {}).get("package", {})
)


def resolved_package_value(value: object, key: str) -> object | None:
    """Resolve a concrete package value or its workspace inheritance."""

    if not isinstance(value, dict):
        return value
    if value != {"workspace": True}:
        return None
    return WORKSPACE_PACKAGE.get(key)


def rules(name: str, cargo: dict, root: pathlib.Path) -> list[str]:
    pkg = cargo.get("package", {})
    is_sys = name.endswith(SYS_SUFFIX) or name in BUILD_ONLY
    is_transport = name in PRIVATE_TRANSPORT
    lib = root / "src" / "lib.rs"
    lib_text = lib.read_text(errors="ignore") if lib.exists() else ""
    bad: list[str] = []

    # 1. Says what it is, in one line.
    if not pkg.get("description"):
        bad.append("no `description` — a reader must learn what it is from Cargo.toml")

    # 2. Workspace identity, toolchain floor, licence and publication posture
    #    are declared, never inferred from the member's location.
    for key in ("version", "edition", "rust-version"):
        if resolved_package_value(pkg.get(key), key) is None:
            bad.append(f"no `{key}` — declare it or inherit it from `[workspace.package]`")
    if not (resolved_package_value(pkg.get("license"), "license") or pkg.get("license-file")):
        bad.append("no `license` — add `license.workspace = true` (inheritance is opt-in)")
    publish = resolved_package_value(pkg.get("publish"), "publish")
    if publish is None:
        bad.append("no explicit `publish` — use `publish.workspace = true` or `publish = false`")
    elif publish is not False:
        bad.append("package publication contradicts the unpublished workspace policy")

    # 3. Shipped Rust dependencies are the exception and must be justified.
    #    Dev- and build-dependencies are unrestricted.
    for dep, spec in cargo.get("dependencies", {}).items():
        # An optional dependency is opt-in: absent unless a feature turns it
        # on, so it costs the default build nothing.
        if isinstance(spec, dict) and spec.get("optional"):
            continue
        if dep in ALLOWED_RUNTIME_DEPS:
            continue
        bad.append(
            f"undeclared runtime dependency `{dep}` — add it to ALLOWED_RUNTIME_DEPS with a reason, or drop it"
        )

    # 4. Public items are documented.
    if not is_sys and not is_transport and "deny(missing_docs)" not in lib_text:
        bad.append("no `#![deny(missing_docs)]`")

    # 5. There is an integration test, not only inline ones. Inline tests share
    #    the crate's privacy; integration tests exercise the surface a caller
    #    actually gets.
    if not is_sys and not is_transport:
        tests = root / "tests"
        if not tests.exists() or not any(tests.glob("*.rs")):
            bad.append("no `tests/` — inline tests do not exercise the public surface")
    elif is_transport:
        projection_test = REPO / "validation" / "unit" / "infra" / "test_surface_projection.py"
        if not projection_test.exists():
            bad.append("no Python surface-projection integration test for the private transport")

    # 6. Documented for humans. Every crate explains why it exists. Local
    #    agent guidance is required where the parent rules are insufficient;
    #    shared contracts, build helpers and simple kernels may stay README-only.
    if not (root / "README.md").exists():
        bad.append("no `README.md`")
    if not is_sys and name not in AGENT_OPTIONAL and not (root / "AGENTS.md").exists():
        bad.append("no `AGENTS.md`")

    # 7. Every feature is wired. A declared feature nothing reads is a promise
    #    of a capability that does not exist.
    feats = set(cargo.get("features", {})) - {"default"}
    if feats:
        # Tests and benches gate on features too; a feature read only there
        # is still wired.
        src = " ".join(
            f.read_text(errors="ignore")
            for sub in ("src", "tests", "benches")
            for f in (root / sub).rglob("*.rs")
            if (root / sub).exists()
        )
        toml_text = (root / "Cargo.toml").read_text(errors="ignore")
        for feat in sorted(feats):
            if re.search(rf'feature\s*=\s*"{re.escape(feat)}"', src):
                continue
            # A feature may exist only to pull an optional dependency or
            # forward a dependency feature (for example pyo3/abi3-py310).
            values = cargo.get("features", {}).get(feat, [])
            if any(item.startswith("dep:") or "/" in item for item in values):
                continue
            # Cargo target tables can gate an oracle/test/example without a
            # source-level cfg.
            if re.search(
                rf'required-features\s*=\s*\[[^\]]*"{re.escape(feat)}"',
                toml_text,
                re.S,
            ):
                continue
            bad.append(f"feature `{feat}` is declared but no `cfg` reads it")

    return bad


# Nonoptional runtime dependencies a retained crate may ship, with the reason
# each is allowed. Optional dependencies cost the default graph nothing and are
# audited when their feature is accepted. Anything else needs a decision.
ALLOWED_RUNTIME_DEPS = {
    "ahash": "internal exact hash tables with canonical ordering at observable boundaries",
    "convex-hull": "exact rational polytope implementation behind the facade",
    "flint-core": "shared safe ownership for exact and rigorous FLINT values",
    "flint-sys": "reviewed upstream flint3-sys declarations for the FLINT 3.6 ABI",
    "geom-core": "exact Euclidean predicates and triangulations behind the facade",
    "graph-core": "graph storage and algorithms behind the facade",
    "groebner-kernel": "verified field and integral Groebner engines behind commalg",
    "lattice-enum": "exact-verified SVP/CVP implementation behind the facade",
    "m4rst": "dense F_2 kernels and the substrate for F_{2^e}",
    "m4rste": "dense F_{2^e} kernels behind finite-field matrix dispatch",
    "mzs": "sparse F_2 kernels behind sparse-matrix dispatch",
    "mpi": "system-MPI communication for m4rst-distributed",
    "module-core": "shared packed module terms and Schreyer-order contracts",
    "noether": "the Rust facade translated by the private PyO3 transport",
    "perm-core": "permutation groups and Schreier-Sims behind the facade",
    "poly-core": "packed monomial, term, truncation, and F_2 polynomial kernels",
    "pwvla": "packed Z/p^n kernels behind residue-ring matrix dispatch",
    "pyo3": "the sole Rust/Python transport boundary",
    "rand": "seeded permutation-group algorithms with Noether-owned sampling",
    "rayon": "the facade-owned affinity-bounded local computation pool",
    "resolution-kernel": "packed standard-graded Betti and Schreyer-resolution kernels",
    "schemars": "Rust-owned JSON schemas for accepted semantic types",
    "serde": "typed deterministic wire formats at kernel/facade boundaries",
    "serde_json": "canonical JSON transport and schema materialization",
    "spar-core": "the shared SparseRingStorage contract",
    "sparb": "rigorous sparse Arb/Acb kernels behind sparse-matrix dispatch",
    "sparla": "exact sparse Z and prime-field kernels behind sparse-matrix dispatch",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    findings: dict[str, list[str]] = {}
    used_runtime_dependencies: set[str] = set()
    for root in sorted(CRATES.iterdir()):
        manifest = root / "Cargo.toml"
        if not manifest.is_dir() and manifest.exists():
            cargo = tomllib.loads(manifest.read_text())
            name = cargo.get("package", {}).get("name", root.name)
            used_runtime_dependencies.update(
                dependency
                for dependency, spec in cargo.get("dependencies", {}).items()
                if not (isinstance(spec, dict) and spec.get("optional"))
            )
            bad = rules(name, cargo, root)
            if bad:
                findings[name] = bad

    stale_allowlist = sorted(ALLOWED_RUNTIME_DEPS.keys() - used_runtime_dependencies)
    if stale_allowlist:
        findings["dependency allowlist"] = [
            f"unused allowed dependency `{dependency}` — remove the stale exception"
            for dependency in stale_allowlist
        ]

    total = sum(len(v) for v in findings.values())
    n_crates = len([d for d in CRATES.iterdir() if (d / "Cargo.toml").exists()])
    if not findings:
        print(f"all {n_crates} crates satisfy the standard")
        return 0

    for name in sorted(findings):
        print(f"\n{name}")
        for line in findings[name]:
            print(f"  - {line}")
    print(f"\n{total} violations across {len(findings)} of {n_crates} crates")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
