#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-all}

if [[ -n ${CARGO_TARGET_DIR:-} ]]; then
    if [[ "$CARGO_TARGET_DIR" == /* ]]; then
        cargo_target_dir=$CARGO_TARGET_DIR
    else
        cargo_target_dir="$repo/$CARGO_TARGET_DIR"
    fi
else
    cargo_target_dir="$repo/target"
fi

if [[ "$mode" != representation && "$mode" != operations && "$mode" != factorization && "$mode" != facade && "$mode" != benchmark && "$mode" != all ]]; then
    echo "usage: $0 {representation|operations|factorization|facade|benchmark|all}" >&2
    exit 2
fi

command -v mpirun >/dev/null || {
    echo "ERROR: mpirun is required for the distributed acceptance workload" >&2
    exit 1
}

# Some minimal systems install libclang itself without Clang's builtin headers.
# GCC's matching freestanding headers give bindgen the standard C definitions;
# a complete clang/libclang-dev installation needs no fallback.
if [[ -z ${BINDGEN_EXTRA_CLANG_ARGS:-} ]] && ! command -v clang >/dev/null && command -v gcc >/dev/null; then
    export BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)"
fi

mpi_args=(--oversubscribe --bind-to none)
if [[ $(id -u) -eq 0 ]]; then
    mpi_args+=(--allow-run-as-root)
fi

expect_mpi_failure() {
    local label=$1
    shift
    local output status
    output=$(mktemp "${TMPDIR:-/tmp}/noether-mpi-failure.XXXXXX")
    set +e
    timeout 30s mpirun "${mpi_args[@]}" "$@" >"$output" 2>&1
    status=$?
    set -e
    if [[ $status -eq 0 ]]; then
        echo "ERROR: $label unexpectedly succeeded" >&2
        cat "$output" >&2
        rm -f "$output"
        exit 1
    fi
    if [[ $status -eq 124 ]]; then
        echo "ERROR: $label hung instead of failing closed" >&2
        cat "$output" >&2
        rm -f "$output"
        exit 1
    fi
    echo "m4rst-distributed expected failure: $label (status=$status)"
    rm -f "$output"
}

if [[ "$mode" == representation || "$mode" == all ]]; then
    cargo build --release -p m4rst-distributed --example representation_acceptance
    binary="$cargo_target_dir/release/examples/representation_acceptance"
    work=$(mktemp -d "${TMPDIR:-/tmp}/noether-mpi.XXXXXX")
    trap 'rm -rf "$work"' EXIT
    mpirun "${mpi_args[@]}" -n 1 "$binary" write "$work/matrix"
    mpirun "${mpi_args[@]}" -n 2 "$binary" open "$work/matrix"
    mpirun "${mpi_args[@]}" -n 4 "$binary" open "$work/matrix"

    cp -a "$work/matrix" "$work/corrupt-manifest"
    python3 - "$work/corrupt-manifest/manifest.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["format"] = "not-packed-f2"
path.write_text(json.dumps(manifest), encoding="utf-8")
PY
    expect_mpi_failure "corrupt manifest" -n 2 "$binary" open "$work/corrupt-manifest"

    cp -a "$work/matrix" "$work/corrupt-tile"
    python3 - "$work/corrupt-tile" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
tile = directory / manifest["tiles"][0]["file"]
payload = bytearray(tile.read_bytes())
payload[0] ^= 1
tile.write_bytes(payload)
PY
    expect_mpi_failure "corrupt packed tile" -n 2 "$binary" open "$work/corrupt-tile"

    cargo build --release -p m4rst-distributed --example rank_failure_acceptance
    failure_binary="$cargo_target_dir/release/examples/rank_failure_acceptance"
    expect_mpi_failure "abrupt MPI rank loss" -n 2 "$failure_binary"
fi

if [[ "$mode" == operations || "$mode" == all ]]; then
    cargo build --release -p m4rst-distributed --example operations_acceptance
    binary="$cargo_target_dir/release/examples/operations_acceptance"
    timeout 120s mpirun "${mpi_args[@]}" -n 1 "$binary"
    timeout 120s mpirun "${mpi_args[@]}" -n 2 "$binary"
    timeout 120s mpirun "${mpi_args[@]}" -n 4 "$binary"
fi

if [[ "$mode" == factorization || "$mode" == all ]]; then
    cargo build --release -p m4rst-distributed --example factorization_acceptance
    binary="$cargo_target_dir/release/examples/factorization_acceptance"
    mpirun "${mpi_args[@]}" -n 1 "$binary"
    mpirun "${mpi_args[@]}" -n 2 "$binary"
    mpirun "${mpi_args[@]}" -n 4 "$binary"
fi

if [[ "$mode" == facade || "$mode" == all ]]; then
    cargo build --release -p noether --features distributed --example distributed_acceptance
    binary="$cargo_target_dir/release/examples/distributed_acceptance"
    mpirun "${mpi_args[@]}" -n 1 "$binary"
    mpirun "${mpi_args[@]}" -n 2 "$binary"
    mpirun "${mpi_args[@]}" -n 4 "$binary"
fi

if [[ "$mode" == benchmark || "$mode" == all ]]; then
    cargo build --release -p m4rst-distributed --example benchmark
    binary="$cargo_target_dir/release/examples/benchmark"
    benchmark_output=
    for processes in 1 2 4; do
        for operation in matmul rank echelon rref pluq solve nullspace inverse; do
            benchmark_args=(
                "$operation"
                --rows 65 --cols 65 --inner 65 --rhs-cols 3 --repetitions 1
                --tile-rows 32 --tile-cols 64
            )
            if [[ "$operation" == matmul ]]; then
                benchmark_args+=(
                    --freivalds-rounds 8
                    --freivalds-seed 15111065706836454659
                    --verification-max-bytes 1048576
                )
            fi
            output=$(mpirun "${mpi_args[@]}" -n "$processes" "$binary" \
                "${benchmark_args[@]}")
            printf '%s\n' "$output"
            benchmark_output+="$output"$'\n'
        done
    done
    result_count=$(grep -c '"kind":"noether-mpi-benchmark-result"' <<<"$benchmark_output")
    [[ "$result_count" -eq 24 ]] || {
        echo "ERROR: expected 24 MPI benchmark results, found $result_count" >&2
        exit 1
    }
    verified_matmuls=$(grep '"kind":"noether-mpi-benchmark-result"' <<<"$benchmark_output" \
        | grep '"operation":"matmul"' \
        | grep '"probabilistic_verification":{' \
        | grep -c '"passed":true')
    [[ "$verified_matmuls" -eq 3 ]] || {
        echo "ERROR: expected 3 passing probabilistically verified matmul results, found $verified_matmuls" >&2
        exit 1
    }
    verified_exact=$(grep '"kind":"noether-mpi-benchmark-result"' <<<"$benchmark_output" \
        | grep -E '"operation":"(rank|echelon|rref|pluq|solve|nullspace|inverse)"' \
        | grep -c '"exact_verification":true')
    [[ "$verified_exact" -eq 21 ]] || {
        echo "ERROR: expected 21 exactly verified non-matmul results, found $verified_exact" >&2
        exit 1
    }
    for operation in matmul rank echelon rref pluq solve nullspace inverse; do
        unique_hashes=$(grep '"kind":"noether-mpi-benchmark-result"' <<<"$benchmark_output" \
            | grep "\"operation\":\"$operation\"" \
            | sed -n 's/.*"logical_hash":"\([^"]*\)".*/\1/p' \
            | sort -u | wc -l)
        [[ "$unique_hashes" -eq 1 ]] || {
            echo "ERROR: $operation logical hashes differ across 1/2/4 MPI ranks" >&2
            exit 1
        }
    done
fi
