#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Clean-install one abi3 wheel across every supported CPython minor.
set -euo pipefail

wheel=${1:-}
output=${2:-}
if [[ -z "$wheel" || ! -f "$wheel" ]]; then
    echo "usage: scripts/check-wheel-pythons.sh path/to/tertius_noether-*.whl [metrics-dir]"
    exit 2
fi
if ! command -v uv >/dev/null; then
    echo "FAIL: uv is required for the multi-Python wheel gate"
    exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
wheel_dir=$(cd "$(dirname "$wheel")" && pwd -P)
wheel="$wheel_dir/$(basename "$wheel")"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/noether-wheel-pythons.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
if [[ -n "$output" ]]; then
    mkdir -p "$output"
fi

versions=${NOETHER_PYTHONS:-"3.10 3.11 3.12 3.13 3.14"}
for version in $versions; do
    environment="$scratch/cp${version/./}"
    uv venv --python "$version" "$environment" >/dev/null
    if [[ -x "$environment/bin/python" ]]; then
        python="$environment/bin/python"
    elif [[ -x "$environment/Scripts/python.exe" ]]; then
        python="$environment/Scripts/python.exe"
    else
        echo "FAIL: no Python executable in $environment"
        exit 1
    fi
    uv pip install --python "$python" --no-deps "$wheel" >/dev/null
    if [[ -n "$output" ]]; then
        "$python" "$repo/scripts/measure-installed-wheel.py" --wheel "$wheel" \
            >"$output/cpython-${version}.json"
        echo "CPython $version: PASS ($output/cpython-${version}.json)"
    else
        "$python" "$repo/scripts/measure-installed-wheel.py" --wheel "$wheel"
        echo "CPython $version: PASS"
    fi
done
