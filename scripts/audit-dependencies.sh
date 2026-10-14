#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Audit the locked Rust graph for advisories, licenses, sources, and policy.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)

audit_version=0.22.2
deny_version=0.20.2
actual_audit=$(cargo audit --version 2>/dev/null | awk '{print $2}')
actual_deny=$(cargo deny --version 2>/dev/null | awk '{print $2}')
if [[ "$actual_audit" != "$audit_version" ]]; then
    echo "FAIL: cargo-audit $audit_version is required, found ${actual_audit:-none}"
    exit 2
fi
if [[ "$actual_deny" != "$deny_version" ]]; then
    echo "FAIL: cargo-deny $deny_version is required, found ${actual_deny:-none}"
    exit 2
fi

# rsmpi 0.8.2 uses `conv` for checked integer conversions, and `conv` pulls
# the unmaintained `custom_derive` macro crate. RustSec classifies this as
# unmaintained rather than vulnerable and reports that no safe upgrade exists.
# Keep the exemption narrower than the advisory ID alone: it is valid only
# while the complete reverse dependency path is the accepted MPI feature.
custom_derive_path=$(cargo tree --locked --all-features --edges normal \
    --invert custom_derive --prefix none)
workspace_version=$(cargo pkgid --locked -p noether | sed 's/.*[#@]//')
expected_custom_derive_path=$(printf \
    'custom_derive v0.1.7\nconv v0.3.3\nmpi v0.8.2\nm4rst-distributed v%s\nnoether v%s\nnoether-python v%s' \
    "$workspace_version" "$workspace_version" "$workspace_version")
# Workspace paths vary by checkout, so compare a normalized package-only
# chain. A second caller or a version change makes the audit fail closed and
# forces this exception to be reconsidered.
normalized_custom_derive_path=$(printf '%s\n' "$custom_derive_path" | sed 's# (/.*)##')
if [[ "$normalized_custom_derive_path" != "$expected_custom_derive_path" ]]; then
    echo "FAIL: RUSTSEC-2025-0058 exception no longer has its reviewed rsmpi-only path"
    printf '%s\n' "$custom_derive_path"
    exit 2
fi

cargo audit --file "$repo/Cargo.lock" --deny warnings \
    --ignore RUSTSEC-2025-0058
# cargo-audit owns the advisory database above. Keep cargo-deny on the three
# graph policies it adds so one audit does not fetch and scan RustSec twice.
cargo deny --manifest-path "$repo/Cargo.toml" --locked check \
    bans licenses sources
