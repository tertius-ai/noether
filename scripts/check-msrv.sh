#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Compile every target and conveyed/source-only feature on the declared Rust
# floor. Native validation-oracle linkage belongs to its dedicated parity lane.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
toolchain=1.89.0

if ! rustup toolchain list | grep -q "^${toolchain}-"; then
    echo "FAIL: Rust $toolchain is required; run: rustup toolchain install $toolchain --profile minimal"
    exit 2
fi

cd "$repo"

production_features=(
    convex-hull/parallel
    flint-core/serde
    graph-core/parallel
    groebner-kernel/parallel
    lattice-enum/parallel
    m4rst/parallel
    m4rst-distributed/parallel
    m4rste/parallel
    mzs/parallel
    noether/distributed
    noether/macaulay2-subprocess
    noether/msolve-subprocess
    noether/singular-subprocess
    noether-bridges/macaulay2-subprocess
    noether-bridges/msolve-subprocess
    noether-bridges/singular-subprocess
    noether-python/abi3-py310
    noether-python/distributed
    noether-python/extension-module
    noether-python/macaulay2-subprocess
    noether-python/msolve-subprocess
    noether-python/singular-subprocess
    poly-core/parallel
    pwvla/parallel
    pwvla/flint
    sparb/parallel
    sparla/flint
)
feature_list=$(IFS=,; echo "${production_features[*]}")
validation_features=(
    m4rst/validate-libm4ri
    m4rste/validate-libm4rie
)
validation_feature_list=$(IFS=,; echo "${validation_features[*]}")

# Fail when a new feature is added without being classified as conveyed or as
# native validation-only. This keeps the explicit profile from going stale.
python3 - "$feature_list" "$validation_feature_list" <<'PY'
import pathlib
import sys
import tomllib

classified = set(filter(None, (sys.argv[1] + "," + sys.argv[2]).split(",")))
declared = set()
for manifest in pathlib.Path("crates").glob("*/Cargo.toml"):
    package = tomllib.loads(manifest.read_text())
    name = package["package"]["name"]
    declared.update(
        f"{name}/{feature}"
        for feature in package.get("features", {})
        if feature != "default"
    )
if classified != declared:
    print(f"FAIL: unclassified Cargo features: {sorted(declared - classified)}", file=sys.stderr)
    print(f"FAIL: stale Cargo feature classifications: {sorted(classified - declared)}", file=sys.stderr)
    raise SystemExit(1)
PY

cargo "+$toolchain" check \
    --workspace --all-targets --locked --features "$feature_list"
