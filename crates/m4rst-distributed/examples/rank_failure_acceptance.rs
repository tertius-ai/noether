// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Launcher-level acceptance case for an abruptly failed MPI process.

use std::thread;
use std::time::Duration;

use m4rst_distributed::World;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = World::initialize()?;
    if world.size() < 2 {
        return Err("rank-failure acceptance requires at least two MPI processes".into());
    }
    if world.mpi_rank() == 1 {
        std::process::exit(73);
    }

    // Give the launcher time to observe the failed peer. A conforming run
    // terminates nonzero; this process must never publish a mathematical
    // success marker that could be mistaken for a partial result.
    thread::sleep(Duration::from_secs(2));
    Ok(())
}
