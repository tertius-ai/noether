#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Audit the clean source-install archive independently of its generator.
set -euo pipefail

archive=${1:-}
if [[ -z "$archive" || ! -f "$archive" ]]; then
    echo "usage: scripts/check-sdist.sh path/to/tertius_noether-*.tar.gz" >&2
    exit 2
fi

entries=$(tar -tzf "$archive")
if printf '%s\n' "$entries" | grep -Eq '(^/|(^|/)\.\.(/|$))'; then
    echo "FAIL: source distribution contains an unsafe archive path" >&2
    exit 1
fi

root=$(printf '%s\n' "$entries" | sed -n '1s#/.*##p')
[[ "$root" == tertius_noether-* ]] || {
    echo "FAIL: unexpected source-distribution root: $root" >&2
    exit 1
}

for required in Cargo.toml Cargo.lock pyproject.toml LICENSE README.md; do
    grep -Fxq "$root/$required" <<<"$entries" || {
        echo "FAIL: source distribution is missing $required" >&2
        exit 1
    }
done

forbidden=$(printf '%s\n' "$entries" \
    | grep -E '(^|/)(AGENTS\.md|__pycache__|\.pytest_cache|\.hypothesis|validation|tests?|benches|examples|oracle|docs/planning|Cargo\.lock|\.gitignore)(/|$)|\.pyc$|\.proptest-regressions$' \
    | grep -v -F "$root/Cargo.lock" \
    || true)
if [[ -n "$forbidden" ]]; then
    echo "FAIL: source distribution contains development or historical payload:" >&2
    printf '  %s\n' "$forbidden" >&2
    exit 1
fi

unexpected_top=$(printf '%s\n' "$entries" \
    | sed "s#^$root/##" \
    | awk -F/ 'NF && $1 !~ /^(Cargo\.lock|Cargo\.toml|LICENSE|PKG-INFO|README\.md|crates|licenses|pyproject\.toml|python)$/ { print }' \
    || true)
if [[ -n "$unexpected_top" ]]; then
    echo "FAIL: source distribution contains an unexpected top-level payload:" >&2
    printf '  %s\n' "$unexpected_top" >&2
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
tar -xzf "$archive" -C "$tmp"
cargo metadata --manifest-path "$tmp/$root/Cargo.toml" --no-deps --format-version 1 >/dev/null

files=$(printf '%s\n' "$entries" | grep -vc '/$' || true)
bytes=$(wc -c <"$archive")
echo "SDIST GATE PASSED — archive=$archive files=$files compressed_bytes=$bytes"
