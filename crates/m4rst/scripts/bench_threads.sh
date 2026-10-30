#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
# Sweep raw serial/parallel eight-table kernels against libm4ri; write CSV.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

read -r -a threads <<< "${THREAD_LIST:-1 2 4 8 16}"
sizes=${N_VALS:-4096,8192,16384,32768,65536,131072}
if [[ ! "$sizes" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ || ${#threads[@]} == 0 ]]; then
    echo 'N_VALS must be positive integers separated by commas; THREAD_LIST must not be empty' >&2
    exit 2
fi
for count in "${threads[@]}"; do
    if [[ ! "$count" =~ ^[1-9][0-9]*$ ]]; then
        echo 'THREAD_LIST must contain positive integers' >&2
        exit 2
    fi
done
out_csv=${OUT_CSV:-bench-runs/m4rst/threads.csv}
log=${LOG:-${out_csv%.csv}.log}
if [[ "$out_csv" == "$log" ]]; then
    echo 'OUT_CSV and LOG must name different files' >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$out_csv")" "$(dirname -- "$log")"
exec 2> >(tee "$log" >&2)

export TERTIUS_BENCH_CPUS=${TERTIUS_BENCH_CPUS:-group}
if [[ -n "${M4RI_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$M4RI_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

{
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
    printf '# seeds=0xA,0xB N_VALS=%s THREAD_LIST=%s\n' "$sizes" "${threads[*]}"
    cargo build --locked --release -p m4rst \
        --features validate-libm4ri,parallel --example bench_threads
} >&2

printf '# N,threads,m4rst_serial_ms,m4rst_parallel_ms,libm4ri_ms\n' > "$out_csv"
first=1
for count in "${threads[@]}"; do
    printf '# threads=%s\n' "$count" >&2
    skip_serial=()
    if [[ "$first" == 0 ]]; then
        skip_serial=(SKIP_SERIAL=1)
    fi
    scripts/bench-pin.sh env -u SKIP_SERIAL "${skip_serial[@]}" \
        N_VALS="$sizes" NUM_THREADS="$count" OMP_NUM_THREADS="$count" \
        OMP_DYNAMIC=FALSE RAYON_NUM_THREADS="$count" \
        cargo run --quiet --locked --release -p m4rst \
        --features validate-libm4ri,parallel --example bench_threads \
        | tee -a "$out_csv"
    first=0
done
printf '# CSV: %s; provenance: %s\n' "$out_csv" "$log" >&2
