#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Build and smoke-test the conveyed Linux wheel at the supported glibc floor.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
if ! git -C "$repo" diff --quiet || ! git -C "$repo" diff --cached --quiet; then
    echo "FAIL: the release-container input must be a committed tree"
    exit 2
fi
if ! command -v docker >/dev/null; then
    echo "FAIL: Docker is required for the Linux release-container build"
    exit 2
fi

image=${NOETHER_MANYLINUX_IMAGE:-quay.io/pypa/manylinux2014_x86_64@sha256:95440e0e72dd3a81dc8d2cf59a84d57af661456620f5bc821ff92048d0e54ff9}
output=${WHEEL_OUTPUT:-$repo/target/release-wheels/linux-x86_64}
source_revision=$(git -C "$repo" rev-parse HEAD)
archive=$(mktemp "${TMPDIR:-/tmp}/noether-release.XXXXXX.tar")
trap 'rm -f "$archive"' EXIT
mkdir -p "$output"
git -C "$repo" archive --format=tar HEAD >"$archive"

docker_args=(
    --rm -i
    -e OUTPUT_UID="$(id -u)" \
    -e OUTPUT_GID="$(id -g)" \
    -e MANYLINUX_IMAGE="$image" \
    -e SOURCE_REVISION="$source_revision" \
    -v "$archive:/tmp/noether-source.tar:ro" \
    -v "$output:/wheelhouse"
)

if [[ -n "${NOETHER_FLINT_CACHE:-}" ]]; then
    mkdir -p "$NOETHER_FLINT_CACHE"
    flint_cache=$(cd "$NOETHER_FLINT_CACHE" && pwd -P)
    docker_args+=(
        -e NOETHER_CACHED_FLINT=1
        -v "$flint_cache:/opt/noether-native"
    )
fi

if [[ -n "${NOETHER_SCCACHE_BIN:-}" ]]; then
    [[ -x "$NOETHER_SCCACHE_BIN" ]] || {
        echo "FAIL: NOETHER_SCCACHE_BIN is not executable: $NOETHER_SCCACHE_BIN"
        exit 2
    }
    sccache_bin=$(cd "$(dirname "$NOETHER_SCCACHE_BIN")" && pwd -P)/$(basename "$NOETHER_SCCACHE_BIN")
    docker_args+=(
        -e RUSTC_WRAPPER=/usr/local/bin/sccache
        -e SCCACHE_GHA_ENABLED
        -e SCCACHE_GHA_VERSION
        -e ACTIONS_CACHE_SERVICE_V2
        -e ACTIONS_CACHE_URL
        -e ACTIONS_RESULTS_URL
        -e ACTIONS_RUNTIME_TOKEN
        -v "$sccache_bin:/usr/local/bin/sccache:ro"
    )
fi

docker run "${docker_args[@]}" "$image" /bin/bash <<'CONTAINER'
set -euo pipefail

mkdir -p /work/noether
tar -xf /tmp/noether-source.tar -C /work/noether
cd /work/noether

rust_version=$(sed -n 's/^channel = "\([^"]*\)"/\1/p' rust-toolchain.toml 2>/dev/null || true)
[[ -n "$rust_version" ]] || {
    echo "FAIL: rust-toolchain.toml has no pinned channel"
    exit 2
}
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "$rust_version"
export PATH="/root/.cargo/bin:$PATH"

python=/opt/python/cp310-cp310/bin/python
"$python" -m venv /tmp/noether-bootstrap
/tmp/noether-bootstrap/bin/python -m pip install --disable-pip-version-check \
    'uv==0.11.16'
export PATH="/tmp/noether-bootstrap/bin:$PATH"

uv venv --python "$python" .venv
VIRTUAL_ENV=/work/noether/.venv uv sync --active --python "$python" \
    --locked --inexact --no-install-project --group dev
make sdist WHEEL_DIR=/tmp/noether-source-audit
if [[ "${NOETHER_CACHED_FLINT:-0}" == 1 ]]; then
    FLINT_PREFIX=/opt/noether-native \
        PREFIX_NOTE_FILE=/work/noether/.deps/flint/prefix.txt \
        scripts/install-flint.sh
    export FLINT_PREFIX=/opt/noether-native
    chown -R "$OUTPUT_UID:$OUTPUT_GID" /opt/noether-native
else
    scripts/install-flint.sh
    export FLINT_PREFIX=/work/noether/.deps/flint
fi
make wheel WHEEL_DIR=/wheelhouse \
    MATURIN_WHEEL_ARGS='--compatibility manylinux2014'

wheel=$(find /wheelhouse -maxdepth 1 -type f \
    -name 'tertius_noether-*.whl' -print)
test "$(printf '%s\n' "$wheel" | grep -c .)" -eq 1
set -o pipefail
scripts/check-wheel.sh "$wheel" | tee /wheelhouse/wheel-audit.txt
scripts/check-wheel-pythons.sh "$wheel" /wheelhouse/python-metrics
scripts/install-flint.sh --copy-sources /wheelhouse/native-sources

{
    echo "source_revision=$SOURCE_REVISION"
    echo "cpu_policy=x86-64-v3"
    echo "runtime=linux/x86_64"
    echo "manylinux_image=$MANYLINUX_IMAGE"
    echo "[rustc]"
    rustc --version --verbose
    echo "[cargo]"
    cargo --version
    echo "[python]"
    for version in 310 311 312 313 314; do
        py="/opt/python/cp${version}-cp${version}/bin/python"
        if [[ -x "$py" ]]; then "$py" --version; fi
    done
    echo "[uv]"
    /tmp/noether-bootstrap/bin/uv --version
    echo "[cc]"
    cc --version | head -n 1
    if command -v sccache >/dev/null; then
        echo "[sccache]"
        sccache --show-stats
    fi
} >/wheelhouse/build-environment.txt
sha256sum "$wheel" >/wheelhouse/SHA256SUMS
chown -R "$OUTPUT_UID:$OUTPUT_GID" /wheelhouse
CONTAINER

echo "Linux release candidate: $output"
