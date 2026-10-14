// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! One/two/four-rank representation and persistence acceptance workload.

use std::env;
use std::path::Path;

use m4rst::Mzd;
use m4rst_distributed::{DistributedError, DistributedMzd, World};

const ROWS: usize = 13;
const COLS: usize = 257;
const TILE_ROWS: usize = 4;
const TILE_COLS: usize = 128;
const SEED: u64 = 0xD15_7B1B;
const LIMIT: usize = 1 << 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let mode = arguments.next().ok_or("expected write or open")?;
    let directory = arguments.next().ok_or("expected a manifest directory")?;
    if arguments.next().is_some() {
        return Err("unexpected extra arguments".into());
    }

    let world = World::initialize()?;
    let matrix = match mode.as_str() {
        "write" => {
            let matrix =
                DistributedMzd::random_with_tiles(&world, ROWS, COLS, SEED, TILE_ROWS, TILE_COLS)?;
            matrix.write(Path::new(&directory))?;
            matrix
        }
        "open" => DistributedMzd::open(&world, Path::new(&directory))?,
        _ => return Err("mode must be write or open".into()),
    };

    let expected = Mzd::random(ROWS, COLS, SEED);
    assert_eq!(
        matrix.global_memory_bytes(),
        Some(ROWS * COLS.div_ceil(64) * size_of::<u64>())
    );
    assert!(matches!(
        matrix.collect(0, 1),
        Err(DistributedError::MaterializationLimit { .. })
    ));
    let collected = matrix.collect(0, LIMIT)?;
    if world.is_root() {
        assert_eq!(collected.as_ref().unwrap().as_words(), expected.as_words());
    } else {
        assert!(collected.is_none());
    }

    let bytes = expected.to_byte_buffer();
    let scattered = DistributedMzd::from_root_byte_buffer(
        &world,
        ROWS,
        COLS,
        world.is_root().then_some(bytes.as_slice()),
        0,
        LIMIT,
        TILE_ROWS,
        TILE_COLS,
    )?;
    assert_eq!(matrix.logical_hash()?, scattered.logical_hash()?);

    if world.is_root() {
        println!(
            "m4rst-distributed representation: mode={mode} processes={} hash={:016x} mpi={}",
            world.size(),
            matrix.logical_hash()?,
            world.library_version().trim()
        );
    } else {
        let _ = matrix.logical_hash()?;
    }
    Ok(())
}
