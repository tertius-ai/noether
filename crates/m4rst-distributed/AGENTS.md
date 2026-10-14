# m4rst-distributed guidance

Follow the repository and `crates/AGENTS.md` rules. This crate owns the
system-MPI representation and algorithms for dense distributed `F_2`
matrices. Rank-local tiles are always ordinary `m4rst::Mzd` values; portable
`m4rst` never depends on this crate.

## Invariants

- Logical storage is row-major packed `u64`, LSB-first, with zero tail bits.
- `m4rst-distributed` owns global partitioning, communication, collective
  agreement and distributed algorithm structure. Rank-local arithmetic calls
  public `m4rst::Mzd` operations or shared `m4rst` kernels; if a needed local
  primitive is missing, add it to `m4rst` instead of duplicating packed
  arithmetic here.
- Tile-column boundaries are word-aligned and ownership is block-cyclic over
  a deterministic near-square process grid.
- Manifests never record a process grid. Reopening under another valid world
  size reads only the newly owned tiles.
- Scale operations never gather a complete input or result. `collect` is the
  sole gather and rejects the request before communication when its explicit
  byte ceiling is too small.
- SUMMA broadcasts each intersecting right subpanel once per process column.
  Consecutive left tile rows with the same process-row owner are packed into
  bounded batches, broadcast once per batch, and multiplied as one rank-local
  `Mzd`; the byte cap covers both the packed input and largest batch product.
  A loop that rebroadcasts a left panel per output tile, or fails to reuse a
  right panel across the local row batches, is a performance defect even when
  its result is exact.
- Scale matmul verification uses seeded Freivalds projections over `F_2`.
  Up to 64 deterministic pseudorandom vectors share one packed `Mzd` word lane, so each batch
  computes `B R`, `A(B R)` and `C R` from ordinary rank-local `Mzd` tile
  products and performs exactly three packed XOR reductions. It never calls
  distributed matmul, gathers an operand, or retains a matrix-scale replica.
  Exact bounded differential tests remain the correctness floor; the explicit
  generator, seed, rounds and pass result are recorded as probabilistic
  evidence, with a null unconditional false-accept bound.
- Transpose, submatrix, paste and bounded bit extraction stream packed tiles;
  rank-local block transforms use `Mzd::transpose`, `Mzd::submatrix`,
  `Mzd::paste_into` or `Mzd::extract_bits`. Per-bit loops in the distributed
  layer are performance defects. An extraction crossing one tile boundary
  broadcasts exactly two tiles, not one collective per bit.
- Identity tiles and root byte windows use shared `Mzd` constructors. Same-tile
  row XOR/swaps and column swaps call the corresponding `Mzd` kernel without
  communication; cross-tile operations broadcast each required source tile
  once and use shared `Mzd` row/column primitives locally.
- No-witness rank may make one 2D-tile-to-row-block redistribution because its
  terminal result is scalar. Each packed tile moves once, no process receives
  the complete matrix, and no matrix is redistributed back. MPI rotates and
  broadcasts the packed bases; shared `m4rst` packed-word basis/update kernels
  own the local elimination.
- Echelon and RREF may use rank-owned full-row `Mzd` blocks, then emit the
  result once into the existing canonical tile allocation. REF preserves the
  exact first-row pivot convention through collective pivot agreement and
  cross-owner row exchange; shared `m4rst` helpers own each local pivot scan
  and packed pivot update. RREF uses forward packed-word basis batches and a
  reverse panel sweep because its canonical result is unique. `m4rst` owns
  local basis construction, panel reduction and row updates. A process may
  replicate one at-most-64-row panel, never the matrix; scalar-pivot RREF
  reductions or broadcasts are performance defects.
- PLUQ may make the same one-way input redistribution into full-row `Mzd`
  blocks. Collective pivot selection owns only global agreement; packed pivot
  search, column swaps and elimination use `m4rst`. Cross-owner row swaps move
  one packed augmented row, each pivot row is broadcast once, and `L`/`U` are
  emitted once into caller-owned canonical tiles. Per-tile broadcasts for
  every scalar pivot are a performance defect.
- Solve may make the same one-way input redistribution for aligned packed
  augmented elimination, then emit its result once into canonical 2D tiles.
  Shared `m4rst` helpers build at-most-64-pivot coefficient bases while
  carrying the matching right-hand-side rows through the same XOR operations.
  MPI broadcasts the separate bounded coefficient and companion payloads in a
  forward sweep, then one bounded pair at a time in reverse to obtain the
  canonical free-variables-zero solution. Per-pivot broadcasts, per-element
  collective reads and a row-block-to-row-block return trip are performance
  defects. Inverse uses this solve route.
- Nullspace reuses PLUQ, solves the packed `U11 X = U12` block system through
  the distributed solve route, and applies `Q` with distributed multiplication.
  Scalar collective reads of `U` or distributed row-by-row back-substitution
  are performance defects.
- Every MPI method is collective unless its rustdoc explicitly says
  rank-local. Collective arguments are checked for agreement before data
  movement, and a local failure becomes a failure on every rank.
- MPI is initialized with serialized thread support. This crate does not
  request `MPI_THREAD_MULTIPLE` and does not call MPI concurrently.
- Random construction is process-grid-independent and bit-identical to
  `Mzd::random` without rank-zero generation. The shared generator is the
  index-addressable SplitMix64 stream; do not replace it with a sequential
  linear-recurrence stream that introduces large-matrix rank collapse.
- Every MPI call site updates the per-rank application-level communication
  counters. Counters describe buffers presented to MPI, not protocol packets;
  direct MPI calls without the matching counter update are a defect.

The crate forbids unsafe code. MPI unsafety is owned by the upstream `mpi`
binding. Filesystem persistence uses only versioned JSON manifests and
little-endian packed tile files.

## Commands

```bash
cargo test -p m4rst-distributed
scripts/test-m4rst-distributed.sh all
cargo clippy -p m4rst-distributed --all-targets -- -D warnings
cargo build --locked --release -p m4rst-distributed --features parallel --example benchmark
mpirun -n 4 target/release/examples/benchmark rank --rows 4096 --cols 4096
```
