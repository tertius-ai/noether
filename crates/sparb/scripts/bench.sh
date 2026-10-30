#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Run the current sparse/dense ball comparison with host/cache provenance.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

log=${LOG:-bench-runs/sparb/comparison.log}
mkdir -p -- "$(dirname -- "$log")"
exec > >(tee "$log") 2>&1

if [[ -z "${FLINT_PREFIX:-}" ]]; then
    if [[ -r .deps/flint/prefix.txt ]]; then
        FLINT_PREFIX=$(<.deps/flint/prefix.txt)
    else
        FLINT_PREFIX="$repo_root/.deps/flint"
    fi
fi
export FLINT_PREFIX
export PKG_CONFIG_PATH="$FLINT_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$FLINT_PREFIX/lib:$FLINT_PREFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="$FLINT_PREFIX/lib:$FLINT_PREFIX${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

printf '# UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '# revision: %s\n' "$(git rev-parse HEAD)"
printf '# working tree (empty means clean):\n'
git status --short
uname -srm
if command -v lscpu >/dev/null 2>&1; then
    LC_ALL=C lscpu | sed -n '/^Model name:/p'
fi
rustc -Vv
printf '# FLINT_PREFIX=%s FLINT_REVISION=%s\n' \
    "$FLINT_PREFIX" "${FLINT_REVISION:-unrecorded}"
if command -v pkg-config >/dev/null 2>&1; then
    printf '# FLINT version: %s\n' "$(pkg-config --modversion flint)"
fi
printf '# fixture seeds=0xB0FF,0xB101; precision=53,128 bits\n'

cargo build --locked --release -p sparb --example bench_compare
scripts/bench-pin.sh cargo run --quiet --locked --release \
    -p sparb --example bench_compare
printf '# log: %s\n' "$log"
