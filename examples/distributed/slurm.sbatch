#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Benjamin Antieau and Deven Manam
#SBATCH --job-name=noether-f2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00

# Add the site's account, partition, modules, environment activation, and
# shared-filesystem paths above or in the submitting shell. Each MPI process
# receives one disjoint CPU budget; Noether does not create nested thread pools.
set -euo pipefail

export NOETHER_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun --cpu-bind=cores python examples/distributed/f2_linear_algebra.py
