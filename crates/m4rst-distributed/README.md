# m4rst-distributed

`m4rst-distributed` is Noether's standalone system-MPI kernel for dense,
bit-packed linear algebra over `F_2`. A logical matrix is split into
word-aligned rectangular tiles, assigned block-cyclically across a near-square
process grid, and stored locally as ordinary `m4rst::Mzd` values.

The crate provides explicit MPI-world metadata, deterministic zero/identity/
seeded-random construction, bounded root-buffer construction and collection,
process-grid-independent manifest persistence, elementary operations,
two-dimensional SUMMA multiplication with bounded process-row panel batching
and one process-column broadcast per shared right subpanel, deterministic echelon/PLUQ,
rank, solve, nullspace, and inverse. These operate on the distributed
representation without changing portable `m4rst` or introducing an
orchestration layer.

The algorithms do not assume a particular node count or process grid. Tile
geometry is explicit, ownership is derived from the active MPI world, and a
persisted matrix may be reopened under a different valid world size. Scale
campaigns may deliberately bound the matrices they measure; those measurement
bounds are not library limits or topology-specific tuning contracts.

Bulk transpose, submatrix and paste stream packed source tiles and delegate
each rank-owned overlap to `m4rst` block kernels. Arbitrary column offsets are
stitched and merged a packed word at a time, including offsets that cross tile
or owner boundaries. Bounded bit extraction likewise broadcasts only the one
or two tiles containing its at-most-64-bit interval.

Seeded construction remains bit-identical to `m4rst::Mzd::random` for every
tile and process grid. The distributed layer selects each tile's global bounds;
`m4rst` owns the deterministic stream and generates the corresponding
word-aligned random window without materializing the global matrix. With the
`parallel` feature, independent rank-local tiles are generated across the
existing Rayon thread budget; MPI is still called only by the serialized
calling thread.

Default constructors choose word-aligned tile dimensions from the logical
shape and process grid, while explicit-tile constructors retain caller-selected
geometry. Functional updates clone only a touched rank-owned tile: unchanged
tiles share immutable packed storage. Elimination may subdivide coarse tile
rows into shape/world-aware row blocks and reassembles the canonical tile
layout exactly, so small and rectangular problems do not leave usable ranks
idle solely because of the storage tile height.

The no-witness rank route is specialized for its scalar result: it moves each
packed input tile once into rank-owned full-row blocks, eliminates them in
place one packed-word panel at a time, and returns the replicated pivot count
without constructing factors or redistributing a matrix result. A rotating
source rank uses the shared `m4rst` packed-word basis/update kernels on its
local `Mzd` blocks and broadcasts the basis as one batch, replacing
scalar-pivot reductions and broadcasts.

Echelon form and RREF use rank-owned full-row `Mzd` blocks. REF retains the
exact first-admissible-row convention: MPI selects the global row and exchanges
it only when ownership changes, while shared `m4rst` helpers perform the local
pivot scan and packed pivot update.
RREF exploits the uniqueness of the canonical result. A forward sweep rotates
rank-local packed-word bases in batches; a reverse sweep broadcasts one
at-most-64-row panel at a time and reduces it with `m4rst`. Basis rows remain
distributed between sweeps, so no process accumulates a complete matrix. The
completed result is emitted once into the existing canonical tiles.

PLUQ uses the same operation-local full-row representation while retaining its
explicit witnesses. Collective agreement chooses the exact global pivot;
`m4rst::Mzd` performs each rank's packed pivot scan, column swap, row swap and
row XOR. Only rows crossing owners are exchanged, each packed pivot row is
broadcast once, and the completed `L` and `U` factors are emitted once into
the caller-owned canonical 2D tiles. `P` and `Q` are generated from replicated
permutation vectors. No input or factor is gathered.

Solve and inverse use packed row-distributed augmented elimination. Each input
segment moves once into an aligned rank-owned row block. Shared `m4rst`
helpers build packed coefficient bases of at most 64 pivots while carrying the
matching right-hand-side rows through the same XOR operations. MPI broadcasts
separate bounded coefficient and companion panels in a forward and reverse
sweep, and the canonical free-variables-zero solution is emitted once into the
2D layout.
This avoids element-wise MPI reads and never gathers a complete operand or
result. Nullspace reuses the explicit distributed PLUQ witnesses.

Building requires a system MPI development installation and libclang's
builtin C headers because the upstream `mpi-sys` crate generates bindings for
the selected MPI ABI. On Debian/Ubuntu, install `libopenmpi-dev`, `openmpi-bin`,
`libclang-dev`, and `clang`; HPC sites normally provide the equivalent MPI and
LLVM modules.

```bash
scripts/test-m4rst-distributed.sh all
```

This compiles the representation and arithmetic acceptance examples once and
runs them under one, two, and four local Open MPI processes. The MPI
implementation is the system installation; site builds are expected to relink
against the cluster's own MPI. The same gate rejects corrupt manifests and
packed tiles, incompatible shapes, invalid roots, undersized collection
budgets, and an abruptly failed rank. The rank-loss case proves fail-stop
launcher behavior; v0.2 does not promise MPI fault recovery.

The release-scale Rust workload emits one JSON header, one rank-local input
initialization sample per process, one rank-local operation sample per process,
and one result per repetition. Initialization, timed operation, verification,
and logical hashing have separate wall-time and application-level
communication telemetry. The counters cover reductions, broadcasts, barriers,
communicator splits, point-to-point calls, and the buffers presented to MPI;
they are not protocol-packet or physical-wire measurements. Rank-local JSON
records travel to rank zero through a separately bounded metadata gather and
only rank zero writes stdout, so launcher aggregation cannot interleave long
records. The workload covers `matmul`,
no-witness `rank`, `echelon`, `rref`, `pluq`, constructed-consistent `solve`,
`nullspace`, and constructed-invertible `inverse` without collecting a matrix.
Seeded random operands use the same index-addressable SplitMix64 stream as
`Mzd::random`; inverse uses the explicitly reported dense block factorization
`[I R; S I + S R]`, which is nonsingular for every `R` and `S`. Matmul samples
also snapshot rank-local elapsed
nanoseconds and call counts for output preparation, communicator setup, both
panel preparation and broadcast paths, local `m4rst` kernels, and output accumulation. Rank-local
phase times may overlap across processes and are summarized as rank maxima and
means rather than added into a fictitious cluster duration. For example:

```bash
cargo build --locked --release -p m4rst-distributed --features parallel --example benchmark
mpirun -n 4 target/release/examples/benchmark matmul \
  --rows 8192 --inner 8192 --cols 8192 --repetitions 3 \
  --tile-rows 1024 --tile-cols 4096 \
  --right-tile-rows 1024 --right-tile-cols 4096 \
  --freivalds-rounds 64 --freivalds-seed 15111065706836454659 \
  --verification-max-bytes 536870912
```

Matmul may tile its right input independently. This matters for strongly
rectangular aggregate-memory workloads: coarse left tiles bound metadata and
preserve large local kernels, while narrower right tile columns distribute the
output across every process-grid column. The result layout uses the left tile
rows and right tile columns. When the right-side options are omitted, they
inherit the common `--tile-rows` and `--tile-cols` values.

With a positive round count, every matmul repetition is checked by seeded
Freivalds verification over `F_2`. Up to 64 deterministic pseudorandom vectors share one
packed word lane, so the verifier independently computes `B R`, `A(B R)` and
`C R` from rank-local `Mzd` tiles with one operand scan and three XOR reductions
per batch; it neither calls distributed matmul nor gathers a matrix.
The JSON evidence records its generator, seed, rounds, and pass result, and
explicitly records no unconditional false-accept bound: a reproducible 64-bit
seed does not provide independently uniform challenge bits. Exact bounded
differential acceptance remains the correctness floor.

That local invocation is a correctness smoke unless the MPI launcher also
records process affinity. Release timing evidence uses explicit launcher CPU
binding; the Slurm template below requests `--cpu-bind=cores`, assigns one
Rayon budget per MPI process from `SLURM_CPUS_PER_TASK`, and builds the
standalone benchmark with the `parallel` feature. `NOETHER_NUM_THREADS` remains
the public facade setting; `RAYON_NUM_THREADS` is set here because this kernel
benchmark deliberately does not initialize the Noether facade's pool.

On Slurm, build against the site's active MPI and submit
`examples/distributed/slurm-benchmark.sbatch`. Its `NOETHER_MPI_BENCH_*`
environment variables choose the operation and geometry without editing the
tracked template. Compare `logical_hash` or mathematical rank across process
counts; every non-matmul sample also performs an untimed exact distributed
verification before it is accepted. Rank is cross-checked against verified
PLUQ; REF checks packed structural form and canonical row equivalence; RREF
additionally checks pivot-unit columns and idempotence. Nullspace verification
checks annihilation, basis independence, and rank-nullity; inverse verification
checks an exact distributed identity product.

Keep one JSON-lines file per `srun` step, then validate the complete scale
claim rather than inspecting it by eye:

```bash
scripts/check-mpi-scale.py \
  --require-operation matmul --require-operation rank \
  --require-operation echelon --require-operation rref \
  --require-operation pluq --require-operation solve \
  --require-operation nullspace --require-operation inverse \
  --require-node-count 1 --require-node-count 2 --require-node-count 4 \
  --require-speedup matmul --require-speedup rank \
  --aggregate-operation matmul --aggregate-operation rank \
  --aggregate-operation solve --node-memory-bytes "$NODE_RAM_BYTES" \
  --require-balanced-input --max-input-imbalance 1.5 \
  --require-freivalds-rounds 64 \
  --require-release-metadata results/*.jsonl
```

The gate checks rank and host participation, affinity, local/global packed-byte
accounting, active-rank and load balance for canonical input tiles, matmul's
output-owner compute layout, and the row-distributed
rank/echelon/RREF/solve/nullspace/inverse working layout, phase communication
intensity, per-process peak RSS, exact non-matmul evidence, required probabilistic matmul
evidence, cross-grid hashes, source and binary identity, measured speedups, and
that one matrix—not merely the sum of several operands—exceeds one node's
physical memory. `scripts/slurm-handoff.py submit`
enforces the same input-ownership condition before submitting jobs; its explicit
`--allow-idle-input-ranks` escape is for correctness experiments, not scaling
claims.
