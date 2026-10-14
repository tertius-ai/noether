#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
#SBATCH --job-name=noether-f2-bench
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

# Build target/release/examples/benchmark with `--features parallel` against
# the site's active MPI module on the shared filesystem before submitting.
# Account, partition, modules and filesystem paths are deliberately site-owned.
set -euo pipefail

repo=${NOETHER_REPO:-$SLURM_SUBMIT_DIR}
binary=${NOETHER_MPI_BENCH_BIN:-$repo/target/release/examples/benchmark}
operation=${NOETHER_MPI_BENCH_OP:-matmul}
rows=${NOETHER_MPI_BENCH_ROWS:-4096}
cols=${NOETHER_MPI_BENCH_COLS:-4096}
inner=${NOETHER_MPI_BENCH_INNER:-4096}
rhs_cols=${NOETHER_MPI_BENCH_RHS_COLS:-64}
repetitions=${NOETHER_MPI_BENCH_REPETITIONS:-3}
seed=${NOETHER_MPI_BENCH_SEED:-5929}
tile_rows=${NOETHER_MPI_BENCH_TILE_ROWS:-1024}
tile_cols=${NOETHER_MPI_BENCH_TILE_COLS:-4096}
right_tile_rows=${NOETHER_MPI_BENCH_RIGHT_TILE_ROWS:-$tile_rows}
right_tile_cols=${NOETHER_MPI_BENCH_RIGHT_TILE_COLS:-$tile_cols}
freivalds_rounds=${NOETHER_MPI_BENCH_FREIVALDS_ROUNDS:-64}
freivalds_seed=${NOETHER_MPI_BENCH_FREIVALDS_SEED:-15111065706836454659}
verification_max_bytes=${NOETHER_MPI_BENCH_VERIFICATION_MAX_BYTES:-536870912}

if [[ ! -x "$binary" ]]; then
    echo "ERROR: benchmark binary is absent: $binary" >&2
    echo "Build it first with: cargo build --locked --release -p m4rst-distributed --features parallel --example benchmark" >&2
    exit 2
fi
if [[ -n $(git -C "$repo" status --porcelain) ]]; then
    echo "ERROR: release evidence requires a clean source tree: $repo" >&2
    exit 2
fi
command -v sha256sum >/dev/null || {
    echo "ERROR: sha256sum is required to identify the measured binary" >&2
    exit 2
}

export NOETHER_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NOETHER_SOURCE_REV
NOETHER_SOURCE_REV=$(git -C "$repo" rev-parse HEAD)
export NOETHER_BINARY_SHA256
NOETHER_BINARY_SHA256=$(sha256sum "$binary" | awk '{print $1}')
arguments=(
    "$operation"
    --rows "$rows"
    --cols "$cols"
    --inner "$inner"
    --rhs-cols "$rhs_cols"
    --repetitions "$repetitions"
    --seed "$seed"
    --tile-rows "$tile_rows"
    --tile-cols "$tile_cols"
)
if [[ "$operation" == matmul ]]; then
    arguments+=(
        --right-tile-rows "$right_tile_rows"
        --right-tile-cols "$right_tile_cols"
        --freivalds-rounds "$freivalds_rounds"
        --freivalds-seed "$freivalds_seed"
        --verification-max-bytes "$verification_max_bytes"
    )
fi
srun --cpu-bind=cores "$binary" "${arguments[@]}"
