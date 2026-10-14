// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

//! Exact differential workload for elementary and multiplication operations.

use m4rst::{MatrixWrite, Mzd};
use m4rst_distributed::{DistributedError, DistributedMzd, World};

const LIMIT: usize = 1 << 20;
const MATMUL_LEFT_PANEL_BATCH_BYTES: usize = 8 * 1024 * 1024;

fn expected_summa_broadcasts(left: &DistributedMzd, right: &DistributedMzd) -> u64 {
    let process_grid = left.layout().process_grid();
    let mpi_rank = usize::try_from(left.world().mpi_rank()).unwrap();
    let process_row = mpi_rank / process_grid.cols();
    let process_col = mpi_rank % process_grid.cols();
    let mut shared_boundaries = vec![0, left.cols()];
    shared_boundaries
        .extend((left.layout().tile_cols()..left.cols()).step_by(left.layout().tile_cols()));
    shared_boundaries
        .extend((right.layout().tile_rows()..right.rows()).step_by(right.layout().tile_rows()));
    shared_boundaries.sort_unstable();
    shared_boundaries.dedup();
    let mut broadcasts = 0;
    for shared in shared_boundaries.windows(2) {
        let local_right_cols: Vec<_> = (0..right.layout().tile_col_count())
            .filter(|tile_col| tile_col % process_grid.cols() == process_col)
            .map(|tile_col| {
                let (_, _, col_start, col_end) = right
                    .layout()
                    .bounds(m4rst_distributed::TileCoord {
                        row: shared[0] / right.layout().tile_rows(),
                        col: tile_col,
                    })
                    .unwrap();
                col_end - col_start
            })
            .collect();
        broadcasts += local_right_cols.len();
        let bytes_per_row = (shared[1] - shared[0])
            .div_ceil(64)
            .max(right.layout().tile_cols().min(right.cols()).div_ceil(64))
            * 8;
        let row_budget = (MATMUL_LEFT_PANEL_BATCH_BYTES / bytes_per_row).max(1);
        let mut batch_rows = 0;
        for tile_row in (process_row..left.layout().tile_row_count()).step_by(process_grid.rows()) {
            let (row_start, row_end, _, _) = left
                .layout()
                .bounds(m4rst_distributed::TileCoord {
                    row: tile_row,
                    col: 0,
                })
                .unwrap();
            let rows = row_end - row_start;
            if batch_rows != 0 && batch_rows + rows > row_budget {
                broadcasts += 1;
                batch_rows = 0;
            }
            batch_rows += rows;
        }
        broadcasts += usize::from(batch_rows != 0);
    }
    u64::try_from(broadcasts).unwrap()
}

fn assert_matches(
    world: &World,
    distributed: &DistributedMzd,
    expected: &Mzd,
) -> Result<(), Box<dyn std::error::Error>> {
    let local = distributed.collect(0, LIMIT)?;
    if world.is_root() {
        assert_eq!(
            (
                local.as_ref().unwrap().rows(),
                local.as_ref().unwrap().cols()
            ),
            (expected.rows(), expected.cols())
        );
        assert_eq!(local.as_ref().unwrap().as_words(), expected.as_words());
    } else {
        assert!(local.is_none());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = World::initialize()?;

    let expected_a = Mzd::random(9, 193, 0xA11CE);
    let expected_b = Mzd::random(9, 193, 0xB0B);
    let a = DistributedMzd::random_with_tiles(&world, 9, 193, 0xA11CE, 4, 128)?;
    let b = DistributedMzd::random_with_tiles(&world, 9, 193, 0xB0B, 4, 128)?;
    let differently_tiled_a = DistributedMzd::random_with_tiles(&world, 9, 193, 0xA11CE, 3, 64)?;
    assert!(a.equals(&differently_tiled_a)?);
    assert_matches(&world, &a.add(&differently_tiled_a)?, &Mzd::new(9, 193))?;

    let mut expected_sum = expected_a.clone();
    expected_sum.xor_into(&expected_b);
    assert_matches(&world, &a.add(&b)?, &expected_sum)?;
    assert_eq!(a.get(8, 192)?, expected_a.get(8, 192));
    assert_eq!(
        a.extract_bits(3, 61, 64)?,
        expected_a.extract_bits(3, 61, 64)
    );
    let extraction_communication_started = world.communication_counters();
    assert_eq!(
        a.extract_bits(3, 100, 64)?,
        expected_a.extract_bits(3, 100, 64)
    );
    assert_eq!(
        world
            .communication_counters()
            .since(extraction_communication_started)
            .broadcast_calls,
        2,
        "packed extraction crossing one tile boundary must broadcast two tiles"
    );
    assert_eq!(a.extract_bits(0, a.cols(), 0)?, 0);
    assert_eq!(
        a.extract_bits(0, 128, 64)?,
        expected_a.extract_bits(0, 128, 64)
    );
    for &(row, col_start, count) in &[
        (a.rows(), 0, 0),
        (0, a.cols() + 1, 0),
        (0, 0, 65),
        (0, 0, a.cols().min(64) + 1),
        (0, a.cols() - 1, 2),
        (0, 130, 64),
    ] {
        assert!(matches!(
            a.extract_bits(row, col_start, count),
            Err(DistributedError::OutOfBounds(_))
        ));
    }

    for &cols in &[1, 63, 64, 65, 127, 128, 129] {
        let boundary = DistributedMzd::random_with_tiles(&world, 1, cols, cols as u64, 1, 64)?;
        assert_eq!(boundary.extract_bits(0, cols, 0)?, 0);
        assert_eq!(
            boundary.extract_bits(0, 0, cols.min(64))?,
            Mzd::random(1, cols, cols as u64).extract_bits(0, 0, cols.min(64))
        );
        if cols > 64 {
            assert_eq!(
                boundary.extract_bits(0, cols - 64, 64)?,
                Mzd::random(1, cols, cols as u64).extract_bits(0, cols - 64, 64)
            );
        }
    }

    let mut edited = a.clone();
    assert!(a
        .local_tiles()
        .iter()
        .zip(edited.local_tiles())
        .all(|(source, clone)| std::ptr::eq(source.matrix(), clone.matrix())));
    let mut expected_edited = expected_a.clone();
    edited.set(8, 192, 1)?;
    for (source, clone) in a.local_tiles().iter().zip(edited.local_tiles()) {
        let owns_edited_cell = clone.coord().row == 2 && clone.coord().col == 1;
        assert_eq!(
            std::ptr::eq(source.matrix(), clone.matrix()),
            !owns_edited_cell,
            "functional update must detach only the touched rank-owned tile"
        );
    }
    expected_edited.set(8, 192, 1);
    edited.flip(0, 0)?;
    expected_edited.flip(0, 0);
    let local_row_xor_started = world.communication_counters();
    edited.xor_row(2, 0)?;
    expected_edited.xor_row_self(2, 0);
    assert_eq!(
        world
            .communication_counters()
            .since(local_row_xor_started)
            .broadcast_calls,
        0,
        "rows in the same tile must use the rank-local m4rst XOR kernel"
    );
    let cross_tile_row_xor_started = world.communication_counters();
    edited.xor_row(7, 2)?;
    expected_edited.xor_row_self(7, 2);
    assert_eq!(
        world
            .communication_counters()
            .since(cross_tile_row_xor_started)
            .broadcast_calls,
        u64::try_from(edited.layout().tile_col_count()).unwrap(),
        "cross-tile row XOR must broadcast each source tile once"
    );
    let local_row_swap_started = world.communication_counters();
    edited.swap_rows(1, 3)?;
    expected_edited.swap_rows(1, 3);
    assert_eq!(
        world
            .communication_counters()
            .since(local_row_swap_started)
            .broadcast_calls,
        0,
        "rows in the same tile must use the rank-local m4rst swap kernel"
    );
    let cross_tile_row_swap_started = world.communication_counters();
    edited.swap_rows(1, 8)?;
    expected_edited.swap_rows(1, 8);
    assert_eq!(
        world
            .communication_counters()
            .since(cross_tile_row_swap_started)
            .broadcast_calls,
        u64::try_from(2 * edited.layout().tile_col_count()).unwrap(),
        "cross-tile row swap must broadcast each source tile once"
    );
    let local_column_swap_started = world.communication_counters();
    edited.swap_cols(3, 64)?;
    expected_edited.swap_cols(3, 64);
    assert_eq!(
        world
            .communication_counters()
            .since(local_column_swap_started)
            .broadcast_calls,
        0,
        "columns in the same tile must use the rank-local m4rst swap kernel"
    );
    let cross_tile_column_swap_started = world.communication_counters();
    edited.swap_cols(3, 192)?;
    expected_edited.swap_cols(3, 192);
    assert_eq!(
        world
            .communication_counters()
            .since(cross_tile_column_swap_started)
            .broadcast_calls,
        u64::try_from(2 * edited.layout().tile_row_count()).unwrap(),
        "cross-tile column swap must broadcast each source tile once per tile row"
    );
    assert_matches(&world, &edited, &expected_edited)?;

    if world.size() > 1 {
        let local = Mzd::identity(2);
        let error = DistributedMzd::distribute_packed(&world, Some(&local), world.mpi_rank(), 16)
            .expect_err("rank-dependent roots must fail collectively before broadcast");
        assert_eq!(error, DistributedError::CollectiveMismatch("root"));

        let partially_invalid_root = if world.mpi_rank() == 0 {
            world.size()
        } else {
            0
        };
        assert!(matches!(
            DistributedMzd::distribute_packed(&world, None, partially_invalid_root, 16),
            Err(DistributedError::InvalidMpiRank { .. })
        ));

        let agreed_left = DistributedMzd::with_tiles(&world, 2, 2, 1, 64)?;
        let agreed_right = DistributedMzd::with_tiles(&world, 2, 2, 1, 64)?;
        let divergent_right = DistributedMzd::with_tiles(&world, 3, 2, 1, 64)?;
        let rank_selected = if world.mpi_rank() == 0 {
            &agreed_right
        } else {
            &divergent_right
        };
        assert!(matches!(
            agreed_left.add(rank_selected),
            Err(DistributedError::CollectiveMismatch(_))
        ));
        assert!(matches!(
            agreed_left.equals(rank_selected),
            Err(DistributedError::CollectiveMismatch(_))
        ));
    }

    let incompatible_sum = DistributedMzd::with_tiles(&world, 8, 193, 4, 128)?;
    assert!(matches!(
        a.add(&incompatible_sum),
        Err(DistributedError::ShapeMismatch(_))
    ));
    let incompatible_product = DistributedMzd::with_tiles(&world, 192, 4, 4, 128)?;
    assert!(matches!(
        DistributedMzd::matmul(&a, &incompatible_product),
        Err(DistributedError::ShapeMismatch(_))
    ));
    assert!(matches!(
        a.submatrix(0, 0, 10, 193),
        Err(DistributedError::OutOfBounds(_))
    ));
    assert!(matches!(
        a.collect(world.size(), LIMIT),
        Err(DistributedError::InvalidMpiRank { .. })
    ));

    assert_matches(&world, &a.transpose()?, &expected_a.transpose())?;
    let submatrix = a.submatrix(2, 63, 9, 191)?;
    let expected_submatrix = expected_a.submatrix(2, 63, 9, 191);
    assert_matches(&world, &submatrix, &expected_submatrix)?;
    let zeros = DistributedMzd::with_tiles(&world, 12, 257, 4, 128)?;
    let pasted = zeros.paste(3, 64, &submatrix)?;
    let mut expected_pasted = Mzd::new(12, 257);
    expected_pasted.paste_into(3, 64, &expected_submatrix);
    assert_matches(&world, &pasted, &expected_pasted)?;
    let unaligned_pasted = zeros.paste(3, 7, &submatrix)?;
    let mut expected_unaligned_pasted = Mzd::new(12, 257);
    expected_unaligned_pasted.paste_into(3, 7, &expected_submatrix);
    assert_matches(&world, &unaligned_pasted, &expected_unaligned_pasted)?;

    let oversized_source = DistributedMzd::with_tiles(&world, 13, 257, 4, 128)?;
    let mut unchanged_after_failed_paste = zeros.clone();
    assert!(matches!(
        unchanged_after_failed_paste.paste_into(0, 0, &oversized_source),
        Err(DistributedError::OutOfBounds(_))
    ));
    assert!(unchanged_after_failed_paste.equals(&zeros)?);
    assert!(matches!(
        unchanged_after_failed_paste.paste_into(usize::MAX, 0, &submatrix),
        Err(DistributedError::OutOfBounds(_))
    ));
    assert!(unchanged_after_failed_paste.equals(&zeros)?);

    for &(row_start, col_start, row_end, col_end) in &[
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, 0, a.rows() + 1, 1),
        (0, 0, 1, a.cols() + 1),
    ] {
        assert!(matches!(
            a.submatrix(row_start, col_start, row_end, col_end),
            Err(DistributedError::OutOfBounds(_))
        ));
    }

    for &(rows, cols) in &[(0, 0), (0, 65), (65, 0), (1, 1), (64, 64), (65, 65)] {
        let boundary = DistributedMzd::with_tiles(&world, rows, cols, 64, 64)?;
        assert_matches(&world, &boundary, &Mzd::new(rows, cols))?;
    }

    let expected_right = Mzd::random(193, 17, 0xC0FFEE);
    let right = DistributedMzd::random_with_tiles(&world, 193, 17, 0xC0FFEE, 64, 128)?;
    let expected_product = Mzd::matmul(&expected_a, &expected_right);
    let communication_started = world.communication_counters();
    let product = DistributedMzd::matmul(&a, &right)?;
    let communication = world.communication_counters().since(communication_started);
    assert_eq!(
        communication.broadcast_calls,
        expected_summa_broadcasts(&a, &right),
        "SUMMA must broadcast each input panel once per process row or column"
    );
    assert_eq!(communication.communicator_split_calls, 2);
    assert_matches(&world, &product, &expected_product)?;

    let freivalds_seed = 0xF2E1_7A1D_5A11_DA7A;
    let verification_started = world.communication_counters();
    assert!(DistributedMzd::verify_matmul_freivalds(
        &a,
        &right,
        &product,
        8,
        freivalds_seed,
        LIMIT,
    )?);
    let verification_communication = world.communication_counters().since(verification_started);
    // Two operand agreements each perform one collective world check plus
    // min/max agreement for eight shape/layout values. Verifier arguments
    // contribute three min/max pairs, the three packed projections each
    // preflight and reduce one output buffer, and the final pass bit is one
    // collective reduction.
    assert_eq!(
        verification_communication.reduction_calls,
        2 * (1 + 8 * 2) + 3 * 2 + 3 * 2 + 1
    );
    assert_eq!(verification_communication.broadcast_calls, 0);
    assert_eq!(verification_communication.sent_messages, 0);
    assert_eq!(verification_communication.received_messages, 0);

    let verification_vector = Mzd::random(1, right.cols(), freivalds_seed);
    let corrupted_col = verification_vector
        .first_one_in_row(0, 0)
        .expect("fixed Freivalds vector must have a set bit");
    let mut corrupted = product.clone();
    corrupted.flip(0, corrupted_col)?;
    assert!(!DistributedMzd::verify_matmul_freivalds(
        &a,
        &right,
        &corrupted,
        1,
        freivalds_seed,
        LIMIT,
    )?);
    assert!(matches!(
        DistributedMzd::verify_matmul_freivalds(&a, &right, &product, 1, freivalds_seed, 0,),
        Err(DistributedError::WorkspaceLimit { .. })
    ));
    assert!(matches!(
        DistributedMzd::verify_matmul_freivalds(&a, &right, &product, 0, freivalds_seed, LIMIT,),
        Err(DistributedError::OutOfBounds(_))
    ));

    let mut reused = DistributedMzd::random_with_tiles(&world, 9, 17, 42, 4, 128)?;
    DistributedMzd::matmul_into(&mut reused, &a, &right)?;
    assert_matches(&world, &reused, &expected_product)?;

    for &(rows, shared, cols, left_seed, right_seed) in &[
        (0, 0, 0, 1, 2),
        (0, 65, 3, 3, 4),
        (3, 0, 5, 5, 6),
        (1, 1, 1, 7, 8),
        (2, 65, 3, 9, 10),
        (7, 129, 5, 11, 12),
        (5, 257, 67, 13, 14),
    ] {
        let expected_left = Mzd::random(rows, shared, left_seed);
        let expected_right = Mzd::random(shared, cols, right_seed);
        let distributed_left =
            DistributedMzd::random_with_tiles(&world, rows, shared, left_seed, 3, 128)?;
        let distributed_right =
            DistributedMzd::random_with_tiles(&world, shared, cols, right_seed, 5, 64)?;
        assert_matches(
            &world,
            &DistributedMzd::matmul(&distributed_left, &distributed_right)?,
            &Mzd::matmul(&expected_left, &expected_right),
        )?;
    }

    if world.is_root() {
        println!(
            "m4rst-distributed operations: processes={} exact elementary+matmul",
            world.size()
        );
    }
    Ok(())
}
