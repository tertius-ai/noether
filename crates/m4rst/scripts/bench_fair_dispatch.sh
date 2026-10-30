#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Sweep the public dispatcher against libm4ri, with equal requested threads.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

read -r -a sizes <<< "${NS:-32768 65536}"
read -r -a threads <<< "${THREADS:-1 4 8 16}"
for value in "${sizes[@]}" "${threads[@]}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo 'NS and THREADS must contain positive integers' >&2
        exit 2
    fi
done
if [[ ${#sizes[@]} == 0 || ${#threads[@]} == 0 ]]; then
    echo 'NS and THREADS must not be empty' >&2
    exit 2
fi
log=${LOG:-bench-runs/m4rst/fair-dispatch.log}
mkdir -p -- "$(dirname -- "$log")"
exec > >(tee "$log") 2>&1

export TERTIUS_BENCH_CPUS=${TERTIUS_BENCH_CPUS:-group}
if [[ -n "${M4RI_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$M4RI_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

printf '# UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '# revision: %s\n' "$(git rev-parse HEAD)"
printf '# working tree (empty means clean):\n'
git status --short
uname -srm
if command -v lscpu >/dev/null 2>&1; then
    LC_ALL=C lscpu | sed -n '/^Model name:/p'
fi
rustc -Vv
printf '# M4RI_PREFIX=%s M4RI_REVISION=%s\n' \
    "${M4RI_PREFIX:-system}" "${M4RI_REVISION:-unrecorded}"
printf '# seeds=0xA,0xB NS=%s THREADS=%s WARMUP=%s RUNS=%s\n' \
    "${sizes[*]}" "${threads[*]}" "${WARMUP:-1}" "${RUNS:-5}"

cargo build --locked --release -p m4rst \
    --features validate-libm4ri,parallel --example bench_fair_dispatch

for size in "${sizes[@]}"; do
    for count in "${threads[@]}"; do
        printf '# N=%s threads=%s\n' "$size" "$count"
        scripts/bench-pin.sh env \
            N="$size" OMP_NUM_THREADS="$count" OMP_DYNAMIC=FALSE \
            RAYON_NUM_THREADS="$count" \
            cargo run --quiet --locked --release -p m4rst \
            --features validate-libm4ri,parallel --example bench_fair_dispatch
    done
done
printf '# log: %s\n' "$log"
