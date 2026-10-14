// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Exact differential workload for elimination, PLUQ, and derived operations.

use m4rst::Mzd;
use m4rst_distributed::{DistributedMzd, World};

const LIMIT: usize = 1 << 20;

fn assert_matches(
    world: &World,
    distributed: &DistributedMzd,
    expected: &Mzd,
) -> Result<(), Box<dyn std::error::Error>> {
    let local = distributed.collect(0, LIMIT)?;
    if world.is_root() {
        let local = local.as_ref().unwrap();
        assert_eq!(
            (local.rows(), local.cols()),
            (expected.rows(), expected.cols())
        );
        assert_eq!(local.as_words(), expected.as_words());
    } else {
        assert!(local.is_none());
    }
    Ok(())
}

fn check_factorization(
    world: &World,
    rows: usize,
    cols: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = Mzd::random(rows, cols, seed);
    let distributed = DistributedMzd::random_with_tiles(world, rows, cols, seed, 3, 64)?;
    let (expected_p, expected_l, expected_u, expected_q, expected_rank) = expected.pluq();
    let factors = distributed.pluq()?;
    assert_eq!(factors.rank(), expected_rank);
    assert!(factors.verify(&distributed)?);
    assert_matches(world, factors.p(), &expected_p)?;
    assert_matches(world, factors.l(), &expected_l)?;
    assert_matches(world, factors.u(), &expected_u)?;
    assert_matches(world, factors.q(), &expected_q)?;
    assert_eq!(distributed.rank()?, expected.rank());
    assert_eq!(distributed.ple_rank()?, expected.ple_rank());

    let (echelon, echelon_rank) = distributed.echelon()?;
    let mut expected_echelon = expected.clone();
    assert_eq!(echelon_rank, expected_echelon.echelonize(false));
    assert_matches(world, &echelon, &expected_echelon)?;
    let rref_communication_started = world.communication_counters();
    let (rref, rref_rank) = distributed.rref()?;
    let rref_communication = world
        .communication_counters()
        .since(rref_communication_started);
    let mut expected_rref = expected.clone();
    assert_eq!(rref_rank, expected_rref.echelonize(true));
    assert_matches(world, &rref, &expected_rref)?;
    let packed_words = cols.div_ceil(m4rst::mzd::RADIX);
    let maximum_panel_broadcasts =
        3_u64 * u64::try_from(world.size()).unwrap() * u64::try_from(packed_words).unwrap();
    assert!(
        rref_communication.broadcast_calls <= maximum_panel_broadcasts,
        "RREF must use bounded multi-pivot panel broadcasts"
    );

    let inner = rows.min(cols);
    let mut p = DistributedMzd::with_tiles(world, rows, rows, 3, 64)?;
    let mut l = DistributedMzd::with_tiles(world, rows, inner, 3, 64)?;
    let mut u = DistributedMzd::with_tiles(world, inner, cols, 3, 64)?;
    let mut q = DistributedMzd::with_tiles(world, cols, cols, 3, 64)?;
    assert_eq!(
        distributed.pluq_into(&mut p, &mut l, &mut u, &mut q)?,
        expected_rank
    );
    assert_matches(world, &p, &expected_p)?;
    assert_matches(world, &l, &expected_l)?;
    assert_matches(world, &u, &expected_u)?;
    assert_matches(world, &q, &expected_q)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = World::initialize()?;
    for &(rows, cols, seed) in &[
        (0, 0, 1),
        (1, 1, 2),
        (3, 7, 3),
        (7, 3, 4),
        (7, 9, 5),
        (6, 65, 6),
        (70, 130, 7),
    ] {
        check_factorization(&world, rows, cols, seed)?;
    }

    let mut cross_tile_bytes = vec![0_u8; 6 * 130];
    cross_tile_bytes[4 * 130 + 64] = 1;
    cross_tile_bytes[5 * 130 + 129] = 1;
    let cross_tile_local = Mzd::from_byte_buffer(6, 130, &cross_tile_bytes);
    let cross_tile = DistributedMzd::from_root_byte_buffer(
        &world,
        6,
        130,
        world.is_root().then_some(cross_tile_bytes.as_slice()),
        0,
        LIMIT,
        3,
        64,
    )?;
    let (expected_p, expected_l, expected_u, expected_q, expected_rank) = cross_tile_local.pluq();
    let factors = cross_tile.pluq()?;
    assert_eq!(expected_rank, 2);
    assert_eq!(factors.rank(), expected_rank);
    assert!(factors.verify(&cross_tile)?);
    assert_matches(&world, factors.p(), &expected_p)?;
    assert_matches(&world, factors.l(), &expected_l)?;
    assert_matches(&world, factors.u(), &expected_u)?;
    assert_matches(&world, factors.q(), &expected_q)?;
    let (cross_tile_rref, cross_tile_rref_rank) = cross_tile.rref()?;
    let mut expected_cross_tile_rref = cross_tile_local.clone();
    assert_eq!(
        cross_tile_rref_rank,
        expected_cross_tile_rref.echelonize(true)
    );
    assert_matches(&world, &cross_tile_rref, &expected_cross_tile_rref)?;

    let a_local = Mzd::random(70, 130, 0x501);
    let y_local = Mzd::random(130, 3, 0x502);
    let b_local = Mzd::matmul(&a_local, &y_local);
    let a = DistributedMzd::random_with_tiles(&world, 70, 130, 0x501, 3, 64)?;
    let y = DistributedMzd::random_with_tiles(&world, 130, 3, 0x502, 3, 64)?;
    let b = DistributedMzd::matmul(&a, &y)?;
    let solve_communication_started = world.communication_counters();
    let solution = DistributedMzd::solve_left(&a, &b)?.expect("constructed system is consistent");
    let solve_communication = world
        .communication_counters()
        .since(solve_communication_started);
    let solve_panel_broadcast_bound = 5_u64
        * u64::try_from(world.size()).unwrap()
        * u64::try_from(a_local.cols().div_ceil(m4rst::mzd::RADIX)).unwrap();
    assert!(
        solve_communication.broadcast_calls <= solve_panel_broadcast_bound,
        "solve must use bounded coefficient/RHS panel broadcasts"
    );
    let b_bytes = world.is_root().then(|| b_local.to_byte_buffer());
    let differently_tiled_b = DistributedMzd::from_root_byte_buffer(
        &world,
        b_local.rows(),
        b_local.cols(),
        b_bytes.as_deref(),
        0,
        LIMIT,
        2,
        64,
    )?;
    let tiled_solve_communication_started = world.communication_counters();
    let differently_tiled_solution = DistributedMzd::solve_left(&a, &differently_tiled_b)?
        .expect("constructed differently tiled system is consistent");
    let tiled_solve_communication = world
        .communication_counters()
        .since(tiled_solve_communication_started);
    assert_eq!(
        tiled_solve_communication.broadcast_calls,
        solve_communication.broadcast_calls
    );
    assert_matches(
        &world,
        &solution,
        &Mzd::solve_left(&a_local, &b_local).unwrap(),
    )?;
    assert!(differently_tiled_solution.equals(&solution)?);
    assert!(DistributedMzd::matmul(&a, &solution)?.equals(&b)?);

    let empty_rhs = DistributedMzd::with_tiles(&world, a.rows(), 0, 3, 64)?;
    let empty_solution = DistributedMzd::solve_left(&a, &empty_rhs)?
        .expect("a zero-column RHS is always consistent");
    assert_matches(&world, &empty_solution, &Mzd::new(a.cols(), 0))?;

    let no_coefficients = DistributedMzd::with_tiles(&world, 3, 0, 2, 64)?;
    let zero_rhs = DistributedMzd::with_tiles(&world, 3, 2, 2, 64)?;
    let zero_variable_solution = DistributedMzd::solve_left(&no_coefficients, &zero_rhs)?
        .expect("the zero RHS is consistent with no variables");
    assert_matches(&world, &zero_variable_solution, &Mzd::new(0, 2))?;
    let impossible_rhs = DistributedMzd::identity_with_tiles(&world, 3, 2, 64)?;
    assert!(DistributedMzd::solve_left(&no_coefficients, &impossible_rhs)?.is_none());

    let no_equations = DistributedMzd::with_tiles(&world, 0, 5, 2, 64)?;
    let no_equation_rhs = DistributedMzd::with_tiles(&world, 0, 2, 2, 64)?;
    let unconstrained_solution = DistributedMzd::solve_left(&no_equations, &no_equation_rhs)?
        .expect("a system with no equations is consistent");
    assert_matches(&world, &unconstrained_solution, &Mzd::new(5, 2))?;

    let kernel = a.nullspace()?;
    assert_matches(&world, &kernel, &Mzd::nullspace(&a_local))?;
    assert_eq!(a.rank()? + kernel.cols(), a.cols());
    let zero = DistributedMzd::matmul(&a, &kernel)?;
    assert!(zero.equals(&DistributedMzd::with_tiles(
        &world,
        a.rows(),
        kernel.cols(),
        3,
        64,
    )?)?);

    let wide_local = Mzd::random(3, 193, 0xB10C);
    let wide = DistributedMzd::random_with_tiles(&world, 3, 193, 0xB10C, 64, 64)?;
    let wide_nullspace_started = world.communication_counters();
    let wide_kernel = wide.nullspace()?;
    let wide_nullspace_communication = world.communication_counters().since(wide_nullspace_started);
    assert!(
        wide_nullspace_communication.broadcast_calls <= 128,
        "nullspace must solve packed PLUQ blocks rather than read U element by element"
    );
    assert_matches(&world, &wide_kernel, &Mzd::nullspace(&wide_local))?;

    let identity = DistributedMzd::identity_with_tiles(&world, 7, 3, 64)?;
    assert!(identity.inverse()?.unwrap().equals(&identity)?);
    let singular = DistributedMzd::with_tiles(&world, 3, 3, 2, 64)?;
    assert!(singular.inverse()?.is_none());
    let inconsistent_rhs = DistributedMzd::identity_with_tiles(&world, 3, 2, 64)?;
    assert!(DistributedMzd::solve_left(&singular, &inconsistent_rhs)?.is_none());

    if world.is_root() {
        println!(
            "m4rst-distributed factorization: processes={} exact PLUQ+derived",
            world.size()
        );
    }
    Ok(())
}
