// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Reproducible MPI-scale dense `F_2` acceptance and timing workload.

use std::fs;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use m4rst_distributed::{
    DistributedMzd, DistributedPLUQFactorization, World, DEFAULT_TILE_COLS, DEFAULT_TILE_ROWS,
};
use serde_json::json;

const DEFAULT_SIZE: usize = 8192;
const DEFAULT_RHS_COLS: usize = 64;
const DEFAULT_REPETITIONS: usize = 3;
const DEFAULT_FREIVALDS_SEED: u64 = 0xD1B5_4A32_D192_ED03;
const DEFAULT_VERIFICATION_MAX_BYTES: usize = 512 * 1024 * 1024;
const REPETITION_SEED_STEP: u64 = 0x94D0_49BB_1331_11EB;
const MAX_EVIDENCE_RECORD_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug)]
enum Operation {
    Matmul,
    Rank,
    Echelon,
    Rref,
    Pluq,
    Solve,
    Nullspace,
    Inverse,
}

impl Operation {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "matmul" => Ok(Self::Matmul),
            "rank" => Ok(Self::Rank),
            "echelon" => Ok(Self::Echelon),
            "rref" => Ok(Self::Rref),
            "pluq" => Ok(Self::Pluq),
            "solve" => Ok(Self::Solve),
            "nullspace" => Ok(Self::Nullspace),
            "inverse" => Ok(Self::Inverse),
            _ => Err(format!("unknown operation {value:?}")),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::Rank => "rank",
            Self::Echelon => "echelon",
            Self::Rref => "rref",
            Self::Pluq => "pluq",
            Self::Solve => "solve",
            Self::Nullspace => "nullspace",
            Self::Inverse => "inverse",
        }
    }
}

struct Config {
    operation: Operation,
    rows: usize,
    cols: usize,
    inner: usize,
    rhs_cols: usize,
    repetitions: usize,
    seed: u64,
    tile_rows: usize,
    tile_cols: usize,
    right_tile_rows: Option<usize>,
    right_tile_cols: Option<usize>,
    freivalds_rounds: usize,
    freivalds_seed: u64,
    verification_max_bytes: usize,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut arguments = std::env::args().skip(1);
        let operation = Operation::parse(
            &arguments
                .next()
                .ok_or_else(|| format!("missing operation; {}", usage()))?,
        )?;
        let mut config = Self {
            operation,
            rows: DEFAULT_SIZE,
            cols: DEFAULT_SIZE,
            inner: DEFAULT_SIZE,
            rhs_cols: DEFAULT_RHS_COLS,
            repetitions: DEFAULT_REPETITIONS,
            seed: 0x1729,
            tile_rows: DEFAULT_TILE_ROWS,
            tile_cols: DEFAULT_TILE_COLS,
            right_tile_rows: None,
            right_tile_cols: None,
            freivalds_rounds: 0,
            freivalds_seed: DEFAULT_FREIVALDS_SEED,
            verification_max_bytes: DEFAULT_VERIFICATION_MAX_BYTES,
        };
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("missing value after {flag:?}; {}", usage()))?;
            match flag.as_str() {
                "--rows" => config.rows = parse_positive(&value, "rows")?,
                "--cols" => config.cols = parse_positive(&value, "cols")?,
                "--inner" => config.inner = parse_positive(&value, "inner")?,
                "--rhs-cols" => config.rhs_cols = parse_positive(&value, "rhs-cols")?,
                "--repetitions" => {
                    config.repetitions = parse_positive(&value, "repetitions")?;
                }
                "--seed" => {
                    config.seed = value
                        .parse()
                        .map_err(|_| format!("seed must be an unsigned integer, got {value:?}"))?;
                }
                "--tile-rows" => config.tile_rows = parse_positive(&value, "tile-rows")?,
                "--tile-cols" => config.tile_cols = parse_positive(&value, "tile-cols")?,
                "--right-tile-rows" => {
                    config.right_tile_rows = Some(parse_positive(&value, "right-tile-rows")?);
                }
                "--right-tile-cols" => {
                    config.right_tile_cols = Some(parse_positive(&value, "right-tile-cols")?);
                }
                "--freivalds-rounds" => {
                    config.freivalds_rounds = value.parse().map_err(|_| {
                        format!("freivalds-rounds must be an unsigned integer, got {value:?}")
                    })?;
                }
                "--freivalds-seed" => {
                    config.freivalds_seed = value.parse().map_err(|_| {
                        format!("freivalds-seed must be an unsigned integer, got {value:?}")
                    })?;
                }
                "--verification-max-bytes" => {
                    config.verification_max_bytes =
                        parse_positive(&value, "verification-max-bytes")?;
                }
                _ => return Err(format!("unknown option {flag:?}; {}", usage())),
            }
        }
        if !matches!(config.operation, Operation::Matmul)
            && (config.right_tile_rows.is_some() || config.right_tile_cols.is_some())
        {
            return Err("right-hand tile geometry is only valid for matmul".to_owned());
        }
        if !matches!(config.operation, Operation::Matmul) && config.freivalds_rounds != 0 {
            return Err("Freivalds verification is only valid for matmul".to_owned());
        }
        if matches!(config.operation, Operation::Inverse) && config.rows != config.cols {
            return Err(format!(
                "inverse requires a square matrix, got {} x {}",
                config.rows, config.cols
            ));
        }
        Ok(config)
    }

    fn right_tile_rows(&self) -> usize {
        self.right_tile_rows.unwrap_or(self.tile_rows)
    }

    fn right_tile_cols(&self) -> usize {
        self.right_tile_cols.unwrap_or(self.tile_cols)
    }
}

fn usage() -> &'static str {
    "usage: benchmark {matmul|rank|echelon|rref|pluq|solve|nullspace|inverse} \
     [--rows N] [--cols N] \
     [--inner N] [--rhs-cols N] [--repetitions N] [--seed N] \
     [--tile-rows N] [--tile-cols N] \
     [--right-tile-rows N] [--right-tile-cols N] \
     [--freivalds-rounds N] [--freivalds-seed N] \
     [--verification-max-bytes N]"
}

fn parse_positive(value: &str, name: &str) -> Result<usize, String> {
    let parsed = value
        .parse()
        .map_err(|_| format!("{name} must be a positive integer, got {value:?}"))?;
    if parsed == 0 {
        Err(format!("{name} must be positive"))
    } else {
        Ok(parsed)
    }
}

enum Inputs {
    Matmul {
        left: DistributedMzd,
        right: DistributedMzd,
    },
    Factorization {
        matrix: DistributedMzd,
    },
    Solve {
        matrix: DistributedMzd,
        rhs: DistributedMzd,
    },
    Inverse {
        matrix: DistributedMzd,
    },
}

struct ProbabilisticVerification {
    rounds: usize,
    seed: u64,
    passed: bool,
}

struct VerificationEvidence {
    exact: Option<bool>,
    probabilistic: Option<ProbabilisticVerification>,
}

impl VerificationEvidence {
    fn failed(&self) -> bool {
        self.exact == Some(false)
            || self
                .probabilistic
                .as_ref()
                .is_some_and(|verification| !verification.passed)
    }
}

impl Inputs {
    fn new(world: &World, config: &Config) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(match config.operation {
            Operation::Matmul => Self::Matmul {
                left: DistributedMzd::random_with_tiles(
                    world,
                    config.rows,
                    config.inner,
                    config.seed,
                    config.tile_rows,
                    config.tile_cols,
                )?,
                right: DistributedMzd::random_with_tiles(
                    world,
                    config.inner,
                    config.cols,
                    config.seed ^ 0x2026,
                    config.right_tile_rows(),
                    config.right_tile_cols(),
                )?,
            },
            Operation::Rank
            | Operation::Echelon
            | Operation::Rref
            | Operation::Pluq
            | Operation::Nullspace => Self::Factorization {
                matrix: DistributedMzd::random_with_tiles(
                    world,
                    config.rows,
                    config.cols,
                    config.seed,
                    config.tile_rows,
                    config.tile_cols,
                )?,
            },
            Operation::Solve => {
                let matrix = DistributedMzd::random_with_tiles(
                    world,
                    config.rows,
                    config.cols,
                    config.seed,
                    config.tile_rows,
                    config.tile_cols,
                )?;
                let witness = DistributedMzd::random_with_tiles(
                    world,
                    config.cols,
                    config.rhs_cols,
                    config.seed ^ 0x2026,
                    config.tile_rows,
                    config.tile_cols,
                )?;
                let rhs = DistributedMzd::matmul(&matrix, &witness)?;
                Self::Solve { matrix, rhs }
            }
            Operation::Inverse => Self::Inverse {
                matrix: block_invertible_matrix(world, config)?,
            },
        })
    }

    fn local_bytes(&self) -> usize {
        match self {
            Self::Matmul { left, right } => left.local_memory_bytes() + right.local_memory_bytes(),
            Self::Factorization { matrix } => matrix.local_memory_bytes(),
            Self::Solve { matrix, rhs } => matrix.local_memory_bytes() + rhs.local_memory_bytes(),
            Self::Inverse { matrix } => matrix.local_memory_bytes(),
        }
    }

    fn global_bytes(&self) -> Option<usize> {
        match self {
            Self::Matmul { left, right } => left
                .global_memory_bytes()?
                .checked_add(right.global_memory_bytes()?),
            Self::Factorization { matrix } => matrix.global_memory_bytes(),
            Self::Solve { matrix, rhs } => matrix
                .global_memory_bytes()?
                .checked_add(rhs.global_memory_bytes()?),
            Self::Inverse { matrix } => matrix.global_memory_bytes(),
        }
    }

    fn run(&self, operation: Operation) -> Result<Outcome, Box<dyn std::error::Error>> {
        Ok(match (operation, self) {
            (Operation::Matmul, Self::Matmul { left, right }) => {
                Outcome::Matrix(DistributedMzd::matmul(left, right)?)
            }
            (Operation::Rank, Self::Factorization { matrix }) => Outcome::Rank(matrix.rank()?),
            (Operation::Echelon, Self::Factorization { matrix }) => {
                let (result, rank) = matrix.echelon()?;
                Outcome::Echelon {
                    matrix: result,
                    rank,
                }
            }
            (Operation::Rref, Self::Factorization { matrix }) => {
                let (result, rank) = matrix.rref()?;
                Outcome::Echelon {
                    matrix: result,
                    rank,
                }
            }
            (Operation::Pluq, Self::Factorization { matrix }) => {
                Outcome::Pluq(Box::new(matrix.pluq()?))
            }
            (Operation::Solve, Self::Solve { matrix, rhs }) => {
                let solution = DistributedMzd::solve_left(matrix, rhs)?
                    .ok_or("constructed consistent system was reported inconsistent")?;
                Outcome::Solution(solution)
            }
            (Operation::Nullspace, Self::Factorization { matrix }) => {
                Outcome::Nullspace(matrix.nullspace()?)
            }
            (Operation::Inverse, Self::Inverse { matrix }) => Outcome::Inverse(
                matrix
                    .inverse()?
                    .ok_or("constructed invertible matrix was reported singular")?,
            ),
            _ => return Err("operation/input mismatch".into()),
        })
    }

    fn verify(
        &self,
        outcome: &Outcome,
        config: &Config,
        repetition: usize,
    ) -> Result<VerificationEvidence, Box<dyn std::error::Error>> {
        let mut evidence = VerificationEvidence {
            exact: None,
            probabilistic: None,
        };
        match (self, outcome) {
            (Self::Matmul { left, right }, Outcome::Matrix(candidate)) => {
                if config.freivalds_rounds != 0 {
                    let repetition_seed = config.freivalds_seed.wrapping_add(
                        REPETITION_SEED_STEP.wrapping_mul(
                            u64::try_from(repetition)
                                .map_err(|_| "repetition does not fit an unsigned 64-bit seed")?,
                        ),
                    );
                    evidence.probabilistic = Some(ProbabilisticVerification {
                        rounds: config.freivalds_rounds,
                        seed: repetition_seed,
                        passed: DistributedMzd::verify_matmul_freivalds(
                            left,
                            right,
                            candidate,
                            config.freivalds_rounds,
                            repetition_seed,
                            config.verification_max_bytes,
                        )?,
                    });
                }
            }
            (Self::Factorization { matrix }, Outcome::Rank(reported_rank)) => {
                let factorization = matrix.pluq()?;
                evidence.exact =
                    Some(factorization.rank() == *reported_rank && factorization.verify(matrix)?);
            }
            (
                Self::Factorization { matrix: source },
                Outcome::Echelon {
                    matrix: output,
                    rank: reported_rank,
                },
            ) => {
                let source_factorization = source.pluq()?;
                let (source_rref, source_rref_rank) = source.rref()?;
                let (output_rref, output_rref_rank) = output.rref()?;
                let reduced = matches!(config.operation, Operation::Rref);
                evidence.exact = Some(
                    source_factorization.verify(source)?
                        && source_factorization.rank() == *reported_rank
                        && source_rref_rank == *reported_rank
                        && output_rref_rank == *reported_rank
                        && output.verify_echelon_structure(reduced)?
                        && source_rref.equals(&output_rref)?
                        && (!reduced || output.equals(&output_rref)?),
                );
            }
            (Self::Factorization { matrix }, Outcome::Pluq(factorization)) => {
                evidence.exact = Some(factorization.verify(matrix)?);
            }
            (Self::Solve { matrix, rhs }, Outcome::Solution(solution)) => {
                evidence.exact = Some(DistributedMzd::matmul(matrix, solution)?.equals(rhs)?);
            }
            (Self::Factorization { matrix }, Outcome::Nullspace(kernel)) => {
                let product = DistributedMzd::matmul(matrix, kernel)?;
                let zero = DistributedMzd::with_tiles(
                    matrix.world(),
                    matrix.rows(),
                    kernel.cols(),
                    product.layout().tile_rows(),
                    product.layout().tile_cols(),
                )?;
                let source_rank = matrix.rank()?;
                let kernel_rank = kernel.rank()?;
                evidence.exact = Some(
                    kernel.rows() == matrix.cols()
                        && source_rank.checked_add(kernel.cols()) == Some(matrix.cols())
                        && kernel_rank == kernel.cols()
                        && product.equals(&zero)?,
                );
            }
            (Self::Inverse { matrix }, Outcome::Inverse(inverse)) => {
                let product = DistributedMzd::matmul(matrix, inverse)?;
                let identity = DistributedMzd::identity_with_tiles(
                    matrix.world(),
                    matrix.rows(),
                    product.layout().tile_rows(),
                    product.layout().tile_cols(),
                )?;
                evidence.exact = Some(product.equals(&identity)?);
            }
            _ => evidence.exact = Some(false),
        }
        Ok(evidence)
    }
}

enum Outcome {
    Matrix(DistributedMzd),
    Rank(usize),
    Echelon { matrix: DistributedMzd, rank: usize },
    Pluq(Box<DistributedPLUQFactorization>),
    Solution(DistributedMzd),
    Nullspace(DistributedMzd),
    Inverse(DistributedMzd),
}

impl Outcome {
    fn local_bytes(&self) -> usize {
        match self {
            Self::Matrix(matrix)
            | Self::Echelon { matrix, .. }
            | Self::Solution(matrix)
            | Self::Nullspace(matrix)
            | Self::Inverse(matrix) => matrix.local_memory_bytes(),
            Self::Rank(_) => 0,
            Self::Pluq(factorization) => [
                factorization.p(),
                factorization.l(),
                factorization.u(),
                factorization.q(),
            ]
            .iter()
            .map(|matrix| matrix.local_memory_bytes())
            .sum(),
        }
    }

    fn global_bytes(&self) -> Option<usize> {
        match self {
            Self::Matrix(matrix)
            | Self::Echelon { matrix, .. }
            | Self::Solution(matrix)
            | Self::Nullspace(matrix)
            | Self::Inverse(matrix) => matrix.global_memory_bytes(),
            Self::Rank(_) => Some(0),
            Self::Pluq(factorization) => [
                factorization.p(),
                factorization.l(),
                factorization.u(),
                factorization.q(),
            ]
            .iter()
            .try_fold(0_usize, |total, matrix| {
                total.checked_add(matrix.global_memory_bytes()?)
            }),
        }
    }

    fn logical_hash(&self) -> Result<u64, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::Matrix(matrix)
            | Self::Echelon { matrix, .. }
            | Self::Solution(matrix)
            | Self::Nullspace(matrix)
            | Self::Inverse(matrix) => matrix.logical_hash()?,
            Self::Rank(rank) => *rank as u64,
            Self::Pluq(factorization) => {
                let mut hash = factorization.rank() as u64;
                for matrix in [
                    factorization.p(),
                    factorization.l(),
                    factorization.u(),
                    factorization.q(),
                ] {
                    hash = hash.rotate_left(13) ^ matrix.logical_hash()?;
                }
                hash
            }
        })
    }

    fn mathematical_rank(&self) -> Option<usize> {
        match self {
            Self::Rank(rank) | Self::Echelon { rank, .. } => Some(*rank),
            Self::Pluq(factorization) => Some(factorization.rank()),
            Self::Matrix(_) | Self::Solution(_) | Self::Nullspace(_) | Self::Inverse(_) => None,
        }
    }

    fn mathematical_nullity(&self) -> Option<usize> {
        match self {
            Self::Nullspace(matrix) => Some(matrix.cols()),
            Self::Matrix(_)
            | Self::Rank(_)
            | Self::Echelon { .. }
            | Self::Pluq(_)
            | Self::Solution(_)
            | Self::Inverse(_) => None,
        }
    }
}

fn block_invertible_matrix(
    world: &World,
    config: &Config,
) -> Result<DistributedMzd, Box<dyn std::error::Error>> {
    let leading_size = config.rows / 2;
    let trailing_size = config.rows - leading_size;
    let upper_right = DistributedMzd::random_with_tiles(
        world,
        leading_size,
        trailing_size,
        config.seed,
        config.tile_rows,
        config.tile_cols,
    )?;
    let lower_left = DistributedMzd::random_with_tiles(
        world,
        trailing_size,
        leading_size,
        config.seed ^ 0x2026,
        config.tile_rows,
        config.tile_cols,
    )?;
    let mut lower_right = DistributedMzd::matmul(&lower_left, &upper_right)?;
    lower_right.xor_assign(&DistributedMzd::identity_with_tiles(
        world,
        trailing_size,
        config.tile_rows,
        config.tile_cols,
    )?)?;

    let mut matrix = DistributedMzd::with_tiles(
        world,
        config.rows,
        config.cols,
        config.tile_rows,
        config.tile_cols,
    )?;
    matrix.paste_into(
        0,
        0,
        &DistributedMzd::identity_with_tiles(
            world,
            leading_size,
            config.tile_rows,
            config.tile_cols,
        )?,
    )?;
    matrix.paste_into(0, leading_size, &upper_right)?;
    matrix.paste_into(leading_size, 0, &lower_left)?;
    matrix.paste_into(leading_size, leading_size, &lower_right)?;
    Ok(matrix)
}

fn peak_rss_kib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn physical_memory_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

fn allowed_cpus() -> Option<String> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status
        .lines()
        .find(|line| line.starts_with("Cpus_allowed_list:"))?;
    Some(line.split_once(':')?.1.trim().to_owned())
}

fn hostname() -> Option<String> {
    // MPI launchers may copy the batch shell's HOSTNAME to every rank. Read
    // node-local operating-system state first so multi-node evidence retains
    // the physical host on which each rank actually ran.
    ["/proc/sys/kernel/hostname", "/etc/hostname"]
        .into_iter()
        .find_map(|path| {
            fs::read_to_string(path)
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
        .or_else(|| std::env::var("HOSTNAME").ok())
}

fn synchronized_elapsed(world: &World, started: Instant) -> Duration {
    // The all-reduce is also the end barrier, so rank zero's observation
    // includes time spent waiting for the slowest process.
    assert!(world.collective_all(true));
    started.elapsed()
}

fn write_json_line(value: &serde_json::Value) -> io::Result<()> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    writeln!(output, "{value}")?;
    output.flush()
}

fn write_rank_json_line(
    world: &World,
    value: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = value.to_string();
    if let Some(records) =
        world.gather_metadata_bytes(encoded.as_bytes(), 0, MAX_EVIDENCE_RECORD_BYTES)?
    {
        let stdout = io::stdout();
        let mut output = stdout.lock();
        for record in records {
            output.write_all(&record)?;
            output.write_all(b"\n")?;
        }
        output.flush()?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::parse()?;
    let world = World::initialize()?;
    let initialization_communication_started = world.communication_counters();
    let initialization_started = Instant::now();
    let inputs = Inputs::new(&world, &config)?;
    assert!(world.collective_all(true));
    let initialization_elapsed = initialization_started.elapsed();
    let initialization_communication = world
        .communication_counters()
        .since(initialization_communication_started);
    let process_peak_after_initialization_kib = peak_rss_kib();

    if world.is_root() {
        write_json_line(&json!({
            "kind": "noether-mpi-benchmark-header",
            "operation": config.operation.name(),
            "rows": config.rows,
            "cols": config.cols,
            "inner": matches!(config.operation, Operation::Matmul).then_some(config.inner),
            "rhs_cols": matches!(config.operation, Operation::Solve).then_some(config.rhs_cols),
            "input_generator": "splitmix64-counter-v1",
            "input_construction": match config.operation {
                Operation::Solve => "random-consistent-via-witness-v1",
                Operation::Inverse => "block-lu-invertible-v1",
                _ => "random-v1",
            },
            "repetitions": config.repetitions,
            "seed": config.seed,
            "tile_rows": config.tile_rows,
            "tile_cols": config.tile_cols,
            "right_tile_rows": matches!(config.operation, Operation::Matmul)
                .then_some(config.right_tile_rows()),
            "right_tile_cols": matches!(config.operation, Operation::Matmul)
                .then_some(config.right_tile_cols()),
            "output_tile_rows": matches!(config.operation, Operation::Matmul)
                .then_some(config.tile_rows),
            "output_tile_cols": matches!(config.operation, Operation::Matmul)
                .then_some(config.right_tile_cols()),
            "matmul_phase_telemetry": matches!(config.operation, Operation::Matmul)
                .then_some("rank-local-nanoseconds-v1"),
            "freivalds_rounds": matches!(config.operation, Operation::Matmul)
                .then_some(config.freivalds_rounds),
            "freivalds_seed": matches!(config.operation, Operation::Matmul)
                .then_some(config.freivalds_seed),
            "verification_max_bytes": matches!(config.operation, Operation::Matmul)
                .then_some(config.verification_max_bytes),
            "processes": world.size(),
            "process_grid_rows": match &inputs {
                Inputs::Matmul { left, .. } => left.layout().process_grid().rows(),
                Inputs::Factorization { matrix }
                | Inputs::Solve { matrix, .. }
                | Inputs::Inverse { matrix } => {
                    matrix.layout().process_grid().rows()
                }
            },
            "process_grid_cols": match &inputs {
                Inputs::Matmul { left, .. } => left.layout().process_grid().cols(),
                Inputs::Factorization { matrix }
                | Inputs::Solve { matrix, .. }
                | Inputs::Inverse { matrix } => {
                    matrix.layout().process_grid().cols()
                }
            },
            "global_input_packed_bytes": inputs.global_bytes(),
            "input_initialization_ms": initialization_elapsed.as_secs_f64() * 1_000.0,
            "rank_zero_input_initialization_communication": {
                "reduction_calls": initialization_communication.reduction_calls,
                "reduction_payload_bytes": initialization_communication.reduction_payload_bytes,
                "broadcast_calls": initialization_communication.broadcast_calls,
                "broadcast_payload_bytes": initialization_communication.broadcast_payload_bytes,
                "communicator_split_calls": initialization_communication.communicator_split_calls,
                "barrier_calls": initialization_communication.barrier_calls,
                "sent_messages": initialization_communication.sent_messages,
                "sent_payload_bytes": initialization_communication.sent_payload_bytes,
                "received_messages": initialization_communication.received_messages,
                "received_payload_bytes": initialization_communication.received_payload_bytes,
            },
            "package_version": env!("CARGO_PKG_VERSION"),
            "source_revision": std::env::var("NOETHER_SOURCE_REV").ok(),
            "binary_sha256": std::env::var("NOETHER_BINARY_SHA256").ok(),
            "mpi": world.library_version(),
            "hostname": hostname(),
            "slurm_job_id": std::env::var("SLURM_JOB_ID").ok(),
            "slurm_job_num_nodes": std::env::var("SLURM_JOB_NUM_NODES").ok(),
            "slurm_step_num_nodes": std::env::var("SLURM_STEP_NUM_NODES").ok(),
            "slurm_ntasks": std::env::var("SLURM_NTASKS").ok(),
            "slurm_step_num_tasks": std::env::var("SLURM_STEP_NUM_TASKS").ok(),
            "slurm_job_nodelist": std::env::var("SLURM_JOB_NODELIST").ok(),
            "slurm_cpus_per_task": std::env::var("SLURM_CPUS_PER_TASK").ok(),
            "noether_num_threads": std::env::var("NOETHER_NUM_THREADS").ok(),
        }))?;
    }

    write_rank_json_line(
        &world,
        &json!({
            "kind": "noether-mpi-initialization-sample",
            "operation": config.operation.name(),
            "mpi_rank": world.mpi_rank(),
            "local_input_packed_bytes": inputs.local_bytes(),
            "elapsed_ms": initialization_elapsed.as_secs_f64() * 1_000.0,
            "process_peak_rss_kib": process_peak_after_initialization_kib,
            "node_physical_memory_bytes": physical_memory_bytes(),
            "cpus_allowed_list": allowed_cpus(),
            "hostname": hostname(),
            "communication": {
                "reduction_calls": initialization_communication.reduction_calls,
                "reduction_payload_bytes": initialization_communication.reduction_payload_bytes,
                "broadcast_calls": initialization_communication.broadcast_calls,
                "broadcast_payload_bytes": initialization_communication.broadcast_payload_bytes,
                "communicator_split_calls": initialization_communication.communicator_split_calls,
                "barrier_calls": initialization_communication.barrier_calls,
                "sent_messages": initialization_communication.sent_messages,
                "sent_payload_bytes": initialization_communication.sent_payload_bytes,
                "received_messages": initialization_communication.received_messages,
                "received_payload_bytes": initialization_communication.received_payload_bytes,
            },
        }),
    )?;

    for repetition in 0..config.repetitions {
        assert!(world.collective_all(true));
        let communication_started = world.communication_counters();
        let matmul_phases_started = world.matmul_phase_counters();
        let started = Instant::now();
        let outcome = inputs.run(config.operation)?;
        let local_operation_elapsed = started.elapsed();
        let elapsed = synchronized_elapsed(&world, started);
        let communication = world.communication_counters().since(communication_started);
        let matmul_phases = world.matmul_phase_counters().since(matmul_phases_started);
        let process_peak_after_operation_kib = peak_rss_kib();

        let verification_communication_started = world.communication_counters();
        let verification_started = Instant::now();
        let verification = inputs.verify(&outcome, &config, repetition)?;
        let local_verification_elapsed = verification_started.elapsed();
        let verification_elapsed = synchronized_elapsed(&world, verification_started);
        let verification_communication = world
            .communication_counters()
            .since(verification_communication_started);
        let process_peak_after_verification_kib = peak_rss_kib();

        let hash_communication_started = world.communication_counters();
        let hash_started = Instant::now();
        let hash = outcome.logical_hash()?;
        let local_hash_elapsed = hash_started.elapsed();
        let hash_elapsed = synchronized_elapsed(&world, hash_started);
        let hash_communication = world
            .communication_counters()
            .since(hash_communication_started);
        let local_output_bytes = outcome.local_bytes();

        write_rank_json_line(
            &world,
            &json!({
                "kind": "noether-mpi-rank-sample",
                "operation": config.operation.name(),
                "repetition": repetition,
                "mpi_rank": world.mpi_rank(),
                "local_input_packed_bytes": inputs.local_bytes(),
                "local_output_packed_bytes": local_output_bytes,
                "local_operation_ms": local_operation_elapsed.as_secs_f64() * 1_000.0,
                "local_verification_ms": local_verification_elapsed.as_secs_f64() * 1_000.0,
                "local_logical_hash_ms": local_hash_elapsed.as_secs_f64() * 1_000.0,
                "process_peak_rss_after_operation_kib": process_peak_after_operation_kib,
                "process_peak_rss_after_verification_kib": process_peak_after_verification_kib,
                "cpus_allowed_list": allowed_cpus(),
                "hostname": hostname(),
                "timed_operation_communication": {
                    "reduction_calls": communication.reduction_calls,
                    "reduction_payload_bytes": communication.reduction_payload_bytes,
                    "broadcast_calls": communication.broadcast_calls,
                    "broadcast_payload_bytes": communication.broadcast_payload_bytes,
                    "communicator_split_calls": communication.communicator_split_calls,
                    "barrier_calls": communication.barrier_calls,
                    "sent_messages": communication.sent_messages,
                    "sent_payload_bytes": communication.sent_payload_bytes,
                    "received_messages": communication.received_messages,
                    "received_payload_bytes": communication.received_payload_bytes,
                },
                "timed_matmul_phases": {
                    "output_preparation": {
                        "calls": matmul_phases.output_preparation.calls,
                        "elapsed_ns": matmul_phases.output_preparation.elapsed_ns,
                    },
                    "communicator_setup": {
                        "calls": matmul_phases.communicator_setup.calls,
                        "elapsed_ns": matmul_phases.communicator_setup.elapsed_ns,
                    },
                    "right_panel_preparation": {
                        "calls": matmul_phases.right_panel_preparation.calls,
                        "elapsed_ns": matmul_phases.right_panel_preparation.elapsed_ns,
                    },
                    "right_panel_broadcast": {
                        "calls": matmul_phases.right_panel_broadcast.calls,
                        "elapsed_ns": matmul_phases.right_panel_broadcast.elapsed_ns,
                    },
                    "left_panel_preparation": {
                        "calls": matmul_phases.left_panel_preparation.calls,
                        "elapsed_ns": matmul_phases.left_panel_preparation.elapsed_ns,
                    },
                    "left_panel_broadcast": {
                        "calls": matmul_phases.left_panel_broadcast.calls,
                        "elapsed_ns": matmul_phases.left_panel_broadcast.elapsed_ns,
                    },
                    "local_kernel": {
                        "calls": matmul_phases.local_kernel.calls,
                        "elapsed_ns": matmul_phases.local_kernel.elapsed_ns,
                    },
                    "output_accumulation": {
                        "calls": matmul_phases.output_accumulation.calls,
                        "elapsed_ns": matmul_phases.output_accumulation.elapsed_ns,
                    },
                },
                "verification_communication": {
                    "reduction_calls": verification_communication.reduction_calls,
                    "reduction_payload_bytes": verification_communication.reduction_payload_bytes,
                    "broadcast_calls": verification_communication.broadcast_calls,
                    "broadcast_payload_bytes": verification_communication.broadcast_payload_bytes,
                    "communicator_split_calls": verification_communication.communicator_split_calls,
                    "barrier_calls": verification_communication.barrier_calls,
                    "sent_messages": verification_communication.sent_messages,
                    "sent_payload_bytes": verification_communication.sent_payload_bytes,
                    "received_messages": verification_communication.received_messages,
                    "received_payload_bytes": verification_communication.received_payload_bytes,
                },
                "logical_hash_communication": {
                    "reduction_calls": hash_communication.reduction_calls,
                    "reduction_payload_bytes": hash_communication.reduction_payload_bytes,
                    "broadcast_calls": hash_communication.broadcast_calls,
                    "broadcast_payload_bytes": hash_communication.broadcast_payload_bytes,
                    "communicator_split_calls": hash_communication.communicator_split_calls,
                    "barrier_calls": hash_communication.barrier_calls,
                    "sent_messages": hash_communication.sent_messages,
                    "sent_payload_bytes": hash_communication.sent_payload_bytes,
                    "received_messages": hash_communication.received_messages,
                    "received_payload_bytes": hash_communication.received_payload_bytes,
                },
            }),
        )?;
        if world.is_root() {
            write_json_line(&json!({
                "kind": "noether-mpi-benchmark-result",
                "operation": config.operation.name(),
                "repetition": repetition,
                "elapsed_ms": elapsed.as_secs_f64() * 1_000.0,
                "verification_elapsed_ms": verification_elapsed.as_secs_f64() * 1_000.0,
                "logical_hash_elapsed_ms": hash_elapsed.as_secs_f64() * 1_000.0,
                "logical_hash": format!("{hash:016x}"),
                "mathematical_rank": outcome.mathematical_rank(),
                "mathematical_nullity": outcome.mathematical_nullity(),
                "exact_verification": verification.exact,
                "probabilistic_verification": verification.probabilistic.as_ref().map(
                    |evidence| json!({
                        "algorithm": "freivalds-f2-splitmix64-v2",
                        "challenge_source": "deterministic-splitmix64",
                        "rounds": evidence.rounds,
                        "seed": evidence.seed,
                        "unconditional_false_accept_bound": null,
                        "passed": evidence.passed,
                    })
                ),
                "global_output_packed_bytes": outcome.global_bytes(),
            }))?;
        }
        if verification.failed() {
            return Err("distributed result failed verification".into());
        }
    }
    Ok(())
}
