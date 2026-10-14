// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! End-to-end Rust-façade acceptance workload for distributed `F_2` matrices.

use noether::linalg::Matrix;
use noether::linalg::{DistributedMatrix, MpiThreadSupport, World};
use noether::numthy::FiniteField;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = World::initialize()?;
    assert!(matches!(
        world.thread_support(),
        MpiThreadSupport::Serialized | MpiThreadSupport::Multiple
    ));
    let field = FiniteField::prime(2_u64)?;

    // Cross both default tile dimensions so every rank in the 2x2 acceptance
    // world owns real packed payload, without ever constructing a root copy.
    let scale = DistributedMatrix::random(&world, field.clone(), 1_050, 4_100, 0x1729)?;
    assert_eq!(scale.shape(), (1_050, 4_100));
    assert!(scale.local_memory_bytes() > 0);
    assert_eq!(
        scale.global_memory_bytes(),
        Some(1_050 * 4_100_usize.div_ceil(64) * 8)
    );

    // The explicit local/distributed boundary retains packed m4rst storage in
    // both directions and is guarded by a packed-byte ceiling.
    let local = world.is_root().then(|| Matrix::identity(field.clone(), 7));
    let distributed = DistributedMatrix::distribute(&world, 0, local.as_ref(), 56)?;
    assert_eq!(distributed.get(0, 0)?.representative().to_string(), "1");
    assert_eq!(distributed.extract_bits(0, 0, 7)?, 1);
    let restored = distributed.set(0, 1, &field.element(1_u8))?.flip(0, 1)?;
    assert!(restored.equals(&distributed)?);
    let row_ops = distributed
        .xor_row(1, 0)?
        .swap_rows(1, 2)?
        .swap_cols(0, 3)?;
    assert_eq!(row_ops.get(2, 1)?.representative().to_string(), "1");
    assert_eq!(row_ops.get(2, 3)?.representative().to_string(), "1");
    assert!(distributed
        .add(&distributed)?
        .equals(&DistributedMatrix::zeros(&world, field.clone(), 7, 7,)?)?);
    assert!(distributed
        .sub(&distributed)?
        .equals(&DistributedMatrix::zeros(&world, field.clone(), 7, 7,)?)?);
    assert!(distributed.transpose()?.equals(&distributed)?);
    let corner = distributed.submatrix(0, 0, 3, 3)?;
    assert!(corner.equals(&DistributedMatrix::identity(&world, field.clone(), 3)?)?);
    let pasted = DistributedMatrix::zeros(&world, field.clone(), 7, 7)?.paste(0, 0, &corner)?;
    assert_eq!(pasted.rank()?, 3);
    assert!(distributed.matmul(&distributed)?.equals(&distributed)?);
    assert!(distributed
        .solve_left(&distributed)?
        .is_some_and(|solution| solution.equals(&distributed).unwrap_or(false)));
    assert!(distributed
        .inverse()?
        .is_some_and(|inverse| inverse.equals(&distributed).unwrap_or(false)));
    let collected = distributed.collect(0, 56)?;
    if world.is_root() {
        assert_eq!(collected.as_ref(), local.as_ref());
    } else {
        assert!(collected.is_none());
    }

    let matrix = DistributedMatrix::random(&world, field, 7, 7, 0x2026)?;
    let factorization = matrix.pluq()?;
    assert!(factorization.verify(&matrix)?);
    let (echelon, echelon_rank) = matrix.echelon()?;
    let (reduced, reduced_rank) = matrix.rref()?;
    assert_eq!(echelon.shape(), matrix.shape());
    assert_eq!(reduced.shape(), matrix.shape());
    assert_eq!(echelon_rank, reduced_rank);
    assert_eq!(matrix.rank()?, matrix.ple_rank()?);
    assert_eq!(matrix.nullspace()?.shape().0, matrix.cols());

    if world.is_root() {
        println!(
            "noether distributed facade: processes={} packed boundary+PLUQ exact mpi={}",
            world.size(),
            world.library_version(),
        );
    }
    Ok(())
}
