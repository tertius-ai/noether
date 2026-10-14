// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Deterministic distributed elimination, PLUQ, and derived operations.

use std::mem::{size_of, size_of_val};

use m4rst::{
    echelon::{
        apply_packed_pivot_to_rows, apply_packed_word_basis, apply_packed_word_basis_to_rows,
        apply_packed_word_basis_with_companion_to_rows, first_pivot_row, packed_word_basis,
        packed_word_basis_with_companion, rref_packed_word_basis_with_companion,
    },
    Mzd,
};
use mpi::traits::*;

use crate::matrix::{try_mzd, try_reserve_items, try_zeroed_words};
use crate::{DistributedError, DistributedMzd};

/// Owned distributed PLUQ decomposition with convention `P L U Q = A`.
#[derive(Clone, Debug)]
pub struct DistributedPLUQFactorization {
    p: DistributedMzd,
    l: DistributedMzd,
    u: DistributedMzd,
    q: DistributedMzd,
    rank: usize,
}

impl DistributedPLUQFactorization {
    /// Distributed row-permutation factor.
    #[must_use]
    pub const fn p(&self) -> &DistributedMzd {
        &self.p
    }

    /// Distributed unit-lower factor.
    #[must_use]
    pub const fn l(&self) -> &DistributedMzd {
        &self.l
    }

    /// Distributed upper factor.
    #[must_use]
    pub const fn u(&self) -> &DistributedMzd {
        &self.u
    }

    /// Distributed column-permutation factor.
    #[must_use]
    pub const fn q(&self) -> &DistributedMzd {
        &self.q
    }

    /// Mathematical rank/pivot count.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Consume the factorization into `(P, L, U, Q, rank)` without copying
    /// any rank-local tile payload.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        DistributedMzd,
        DistributedMzd,
        DistributedMzd,
        DistributedMzd,
        usize,
    ) {
        (self.p, self.l, self.u, self.q, self.rank)
    }

    /// Collectively verify the exact convention `P L U Q = source` without
    /// gathering any factor or product.
    pub fn verify(&self, source: &DistributedMzd) -> Result<bool, DistributedError> {
        let lu = DistributedMzd::matmul(&self.l, &self.u)?;
        let plu = DistributedMzd::matmul(&self.p, &lu)?;
        let pluq = DistributedMzd::matmul(&plu, &self.q)?;
        pluq.equals(source)
    }
}

impl DistributedMzd {
    /// Collective in-place row echelonization. With `full = true`, clears
    /// above and below each pivot and returns reduced row echelon form.
    ///
    /// Input tiles move once into rank-owned full-row [`Mzd`] blocks. Plain
    /// echelon form uses exact global pivot agreement and row exchange. RREF
    /// batches rank-local packed-word bases in a forward sweep, then reduces
    /// one bounded panel at a time in reverse. Packed local arithmetic remains
    /// ordinary `m4rst` work, no process holds the complete matrix, and the
    /// completed result is emitted once into the existing canonical tiles.
    pub fn echelonize(&mut self, full: bool) -> Result<usize, DistributedError> {
        self.world().agree_u64(u64::from(full), "full")?;
        echelonize_row_distributed(self, full)
    }

    /// Collective owned row echelon form and its rank.
    pub fn echelon(&self) -> Result<(Self, usize), DistributedError> {
        let mut result = self.clone();
        let rank = result.echelonize(false)?;
        Ok((result, rank))
    }

    /// Collective owned reduced row echelon form and its rank.
    pub fn rref(&self) -> Result<(Self, usize), DistributedError> {
        let mut result = self.clone();
        let rank = result.echelonize(true)?;
        Ok((result, rank))
    }

    /// Collective no-witness mathematical rank. Input tiles make one
    /// rank-to-rank redistribution into operation-local full-row blocks. One
    /// packed-word panel at a time, a rotating source rank broadcasts a local
    /// row-space basis as one batch; the scalar result requires no
    /// redistribution back and retains no factors.
    pub fn rank(&self) -> Result<usize, DistributedError> {
        rank_row_distributed(self)
    }

    /// Collective no-witness PLE rank. This shares the row-distributed scalar
    /// elimination route and does not allocate explicit factors.
    pub fn ple_rank(&self) -> Result<usize, DistributedError> {
        self.rank()
    }

    /// Collective deterministic PLUQ decomposition with convention
    /// `P L U Q = self`.
    ///
    /// Input tiles move once into rank-owned full-row [`Mzd`] blocks. Global
    /// pivot agreement remains collective, while packed pivot search, column
    /// swaps and elimination are ordinary `m4rst` operations. `L` and `U` are
    /// emitted once into canonical two-dimensional tiles; no rank gathers the
    /// source or any factor.
    pub fn pluq(&self) -> Result<DistributedPLUQFactorization, DistributedError> {
        let rows = self.rows();
        let cols = self.cols();
        let inner = rows.min(cols);
        let mut p = Self::with_tiles(
            self.world(),
            rows,
            rows,
            self.layout().tile_rows(),
            self.layout().tile_cols(),
        )?;
        let mut l = Self::with_tiles(
            self.world(),
            rows,
            inner,
            self.layout().tile_rows(),
            self.layout().tile_cols(),
        )?;
        let mut u = Self::with_tiles(
            self.world(),
            inner,
            cols,
            self.layout().tile_rows(),
            self.layout().tile_cols(),
        )?;
        let mut q = Self::with_tiles(
            self.world(),
            cols,
            cols,
            self.layout().tile_rows(),
            self.layout().tile_cols(),
        )?;
        let rank = self.pluq_into(&mut p, &mut l, &mut u, &mut q)?;
        Ok(DistributedPLUQFactorization { p, l, u, q, rank })
    }

    /// Caller-owned-output counterpart to [`DistributedMzd::pluq`]. All
    /// output tiles are overwritten without gathering the source. Existing
    /// output allocations are retained when row-block factors are emitted.
    pub fn pluq_into(
        &self,
        p: &mut Self,
        l: &mut Self,
        u: &mut Self,
        q: &mut Self,
    ) -> Result<usize, DistributedError> {
        for output in [&*p, &*l, &*u, &*q] {
            self.ensure_same_world(output)?;
        }
        let rows = self.rows();
        let cols = self.cols();
        let inner = rows.min(cols);
        let tile_shape = (self.layout().tile_rows(), self.layout().tile_cols());
        if p.shape() != (rows, rows)
            || l.shape() != (rows, inner)
            || u.shape() != (inner, cols)
            || q.shape() != (cols, cols)
            || [p.layout(), l.layout(), u.layout(), q.layout()]
                .iter()
                .any(|layout| (layout.tile_rows(), layout.tile_cols()) != tile_shape)
        {
            return Err(DistributedError::ShapeMismatch(
                "PLUQ outputs have incompatible shape or tile geometry".to_owned(),
            ));
        }

        p.clear();
        l.clear();
        u.clear();
        q.clear();
        pluq_row_distributed(self, p, l, u, q)
    }

    /// Collective solve of `A X = B`. Returns `None` uniformly when the
    /// system is inconsistent; free variables are set to zero.
    ///
    /// The packed inputs are redistributed once into aligned rank-owned row
    /// blocks for augmented elimination. Coefficient panels and their packed
    /// right-hand-side companions are broadcast in bounded batches. The
    /// result is then emitted once in the canonical two-dimensional layout;
    /// the solve never performs element-wise collective reads or gathers a
    /// complete matrix.
    pub fn solve_left(a: &Self, b: &Self) -> Result<Option<Self>, DistributedError> {
        a.ensure_same_world(b)?;
        if a.rows() != b.rows() {
            return Err(DistributedError::ShapeMismatch(format!(
                "solve requires equal row counts, got {} and {}",
                a.rows(),
                b.rows()
            )));
        }
        solve_row_distributed(a, b)
    }

    /// Collective right-nullspace basis with vectors stored as columns.
    /// The PLUQ upper factor is split into packed pivot and free-variable
    /// blocks, solved through [`DistributedMzd::solve_left`], and finally
    /// mapped through the column permutation. No scalar entries are read
    /// collectively.
    pub fn nullspace(&self) -> Result<Self, DistributedError> {
        let factorization = self.pluq()?;
        let rank = factorization.rank;
        let nullity = self.cols() - rank;
        let leading = factorization.u.submatrix(0, 0, rank, rank)?;
        let trailing = factorization.u.submatrix(0, rank, rank, self.cols())?;
        let dependent = Self::solve_left(&leading, &trailing)?.ok_or_else(|| {
            DistributedError::CollectiveFailure(
                "unit-upper PLUQ pivot block was unexpectedly singular".to_owned(),
            )
        })?;
        let mut basis = Self::row_shifted_identity_with_tiles(
            self.world(),
            self.cols(),
            nullity,
            rank,
            self.layout().tile_rows(),
            self.layout().tile_cols(),
        )?;
        basis.paste_into(0, 0, &dependent)?;
        let q_transpose = factorization.q.transpose()?;
        Self::matmul(&q_transpose, &basis)
    }

    /// Collective multiplicative inverse, or `None` uniformly when singular.
    pub fn inverse(&self) -> Result<Option<Self>, DistributedError> {
        if self.rows() != self.cols() {
            return Err(DistributedError::ShapeMismatch(format!(
                "inverse requires a square matrix, got {} x {}",
                self.rows(),
                self.cols()
            )));
        }
        let identity = Self::identity_with_tiles(
            self.world(),
            self.rows(),
            self.layout().tile_rows(),
            self.layout().tile_cols(),
        )?;
        Self::solve_left(self, &identity)
    }

    /// Exact structural validation hook used by scale acceptance.
    ///
    /// This scans redistributed packed row blocks without gathering the
    /// matrix. When `reduced` is true it additionally checks that every pivot
    /// column is a unit column.
    #[doc(hidden)]
    pub fn verify_echelon_structure(&self, reduced: bool) -> Result<bool, DistributedError> {
        let block_rows = operation_block_rows(self);
        let blocks = redistribute_full_rows(self, block_rows)?;
        let mut local_pivots = collective_zeroed_words(self.world(), self.rows())?;
        local_pivots.fill(u64::MAX);
        for block in &blocks {
            for local_row in 0..block.matrix.rows() {
                let pivot =
                    block
                        .matrix
                        .first_one_in_row(local_row, 0)
                        .map_or(Ok(0_u64), |col| {
                            col.checked_add(1)
                                .and_then(|value| u64::try_from(value).ok())
                                .ok_or(DistributedError::SizeOverflow)
                        })?;
                local_pivots[block.row_start + local_row] = pivot;
            }
        }
        let pivots = self.world().min_u64s(&local_pivots)?;
        let mut structurally_valid = true;
        let mut previous = None;
        let mut saw_zero_row = false;
        for &encoded in &pivots {
            if encoded == 0 {
                saw_zero_row = true;
                continue;
            }
            if encoded == u64::MAX || saw_zero_row {
                structurally_valid = false;
                break;
            }
            let pivot = usize::try_from(encoded - 1).map_err(|_| DistributedError::SizeOverflow)?;
            if previous.is_some_and(|prior| pivot <= prior) {
                structurally_valid = false;
                break;
            }
            previous = Some(pivot);
        }

        if reduced && structurally_valid {
            let mut pivot_mask = collective_mzd(self.world(), 1, self.cols())?;
            for &encoded in &pivots {
                if encoded != 0 {
                    pivot_mask.set(
                        0,
                        usize::try_from(encoded - 1).map_err(|_| DistributedError::SizeOverflow)?,
                        1,
                    );
                }
            }
            for block in &blocks {
                for local_row in 0..block.matrix.rows() {
                    let global_row = block.row_start + local_row;
                    let encoded = pivots[global_row];
                    for (word_index, (&row_word, &mask_word)) in block
                        .matrix
                        .row_words(local_row)
                        .iter()
                        .zip(pivot_mask.row_words(0))
                        .enumerate()
                    {
                        let expected = if encoded == 0 {
                            0
                        } else {
                            let pivot = usize::try_from(encoded - 1)
                                .map_err(|_| DistributedError::SizeOverflow)?;
                            if pivot / m4rst::mzd::RADIX == word_index {
                                1_u64 << (pivot % m4rst::mzd::RADIX)
                            } else {
                                0
                            }
                        };
                        structurally_valid &= row_word & mask_word == expected;
                    }
                }
            }
        }
        Ok(self.world().collective_all(structurally_valid))
    }
}

struct RankRowBlock {
    tile_row: usize,
    row_start: usize,
    matrix: Mzd,
}

fn operation_block_rows(matrix: &DistributedMzd) -> usize {
    if matrix.rows() == 0 {
        return 1;
    }
    let workers = usize::try_from(matrix.world().size()).expect("MPI world sizes are positive");
    matrix
        .layout()
        .tile_rows()
        .min(matrix.rows().div_ceil(workers))
        .max(1)
}

fn copy_word_rows(payload: &mut [u64], rows: &[Vec<u64>]) -> Result<(), DistributedError> {
    let mut offset = 0_usize;
    for row in rows {
        let end = offset
            .checked_add(row.len())
            .ok_or(DistributedError::SizeOverflow)?;
        let Some(destination) = payload.get_mut(offset..end) else {
            return Err(DistributedError::CollectiveFailure(
                "packed factorization workspace had an inconsistent word count".to_owned(),
            ));
        };
        destination.copy_from_slice(row);
        offset = end;
    }
    if offset != payload.len() {
        return Err(DistributedError::CollectiveFailure(
            "packed factorization workspace had an inconsistent word count".to_owned(),
        ));
    }
    Ok(())
}

fn collective_zeroed_words(
    world: &crate::World,
    words: usize,
) -> Result<Vec<u64>, DistributedError> {
    let payload = try_zeroed_words(words);
    if !world.collective_all(payload.is_ok()) {
        return Err(DistributedError::CollectiveFailure(
            "at least one rank could not allocate a packed factorization panel".to_owned(),
        ));
    }
    payload
}

fn collective_mzd(world: &crate::World, rows: usize, cols: usize) -> Result<Mzd, DistributedError> {
    let matrix = try_mzd(rows, cols);
    if !world.collective_all(matrix.is_ok()) {
        return Err(DistributedError::CollectiveFailure(
            "at least one rank could not allocate a packed factorization matrix".to_owned(),
        ));
    }
    matrix
}

fn echelonize_row_distributed(
    matrix: &mut DistributedMzd,
    full: bool,
) -> Result<usize, DistributedError> {
    if full {
        return rref_panel_row_distributed(matrix);
    }
    let world = matrix.world().clone();
    let rows = matrix.rows();
    let cols = matrix.cols();
    let block_rows = operation_block_rows(matrix);
    let mut blocks = redistribute_full_rows(matrix, block_rows)?;
    let mut pivot_row = 0;

    for col in 0..cols {
        if pivot_row == rows {
            break;
        }
        let Some(selected) = global_pivot_in_column(&world, &blocks, pivot_row, col)? else {
            continue;
        };
        if selected != pivot_row {
            swap_row_block_rows(&world, &mut blocks, selected, pivot_row, block_rows)?;
        }
        let pivot = broadcast_row(&world, &blocks, pivot_row, cols, block_rows)?;
        for block in &mut blocks {
            let local_start = (pivot_row + 1)
                .saturating_sub(block.row_start)
                .min(block.matrix.rows());
            let local_end = block.matrix.rows();
            apply_packed_pivot_to_rows(
                &mut block.matrix,
                local_start..local_end,
                col,
                &pivot,
                |_| {},
            );
        }
        pivot_row += 1;
    }

    row_blocks_into_distributed(&blocks, matrix, block_rows)?;
    Ok(pivot_row)
}

#[derive(Clone, Copy)]
struct RrefPanel {
    panel_word: usize,
    row_start: usize,
    row_end: usize,
}

fn rref_panel_row_distributed(matrix: &mut DistributedMzd) -> Result<usize, DistributedError> {
    let world = matrix.world().clone();
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let block_rows = operation_block_rows(matrix);
    let packed_words = matrix.cols().div_ceil(m4rst::mzd::RADIX);
    let mut work_blocks = redistribute_full_rows(matrix, block_rows)?;
    let mut result_blocks = empty_row_blocks(
        &world,
        matrix.rows(),
        matrix.cols(),
        block_rows,
        world_size,
        mpi_rank,
    )?;
    let mut panels = Vec::new();
    let mut rank = 0;

    for panel_word in 0..packed_words {
        if rank == matrix.rows() {
            break;
        }
        let panel_start = panel_word * m4rst::mzd::RADIX;
        let panel_width = (matrix.cols() - panel_start).min(m4rst::mzd::RADIX);
        let suffix_words = packed_words - panel_word;
        let row_start = rank;
        let mut panel_basis = Vec::new();

        for source_offset in 0..world_size {
            let source = (panel_word + source_offset) % world_size;
            let local_basis = if mpi_rank == source {
                packed_word_basis(
                    work_blocks.iter().map(|block| &block.matrix),
                    panel_word,
                    panel_width,
                )
            } else {
                Vec::new()
            };
            let mut basis_count = if mpi_rank == source {
                u64::try_from(local_basis.len()).map_err(|_| DistributedError::SizeOverflow)?
            } else {
                0
            };
            {
                let communicator = world.communicator();
                communicator
                    .process_at_rank(
                        i32::try_from(source).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .broadcast_into(&mut basis_count);
                world.record_broadcast(size_of::<u64>());
            }
            let basis_count =
                usize::try_from(basis_count).map_err(|_| DistributedError::SizeOverflow)?;
            let payload_words = basis_count
                .checked_mul(suffix_words)
                .ok_or(DistributedError::SizeOverflow)?;
            let mut payload = collective_zeroed_words(&world, payload_words)?;
            let copied = if mpi_rank == source {
                copy_word_rows(&mut payload, &local_basis)
            } else {
                Ok(())
            };
            if !world.collective_all(copied.is_ok()) {
                return Err(DistributedError::CollectiveFailure(
                    "the RREF panel owner produced an inconsistent payload".to_owned(),
                ));
            }
            copied?;
            debug_assert_eq!(payload.len(), payload_words);
            if !payload.is_empty() {
                let communicator = world.communicator();
                communicator
                    .process_at_rank(
                        i32::try_from(source).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .broadcast_into(payload.as_mut_slice());
                world.record_broadcast(size_of_val(&payload[..]));
            }

            for block in &mut work_blocks {
                apply_packed_word_basis(&mut block.matrix, panel_word, &payload);
            }
            panel_basis.extend(payload.chunks_exact(suffix_words).map(<[u64]>::to_vec));
            rank += basis_count;
            if rank == matrix.rows() || panel_basis.len() == panel_width {
                break;
            }
        }

        panel_basis.sort_unstable_by_key(|row| row[0].trailing_zeros());
        debug_assert!(panel_basis
            .windows(2)
            .all(|rows| { rows[0][0].trailing_zeros() < rows[1][0].trailing_zeros() }));
        assign_panel_suffix_rows(&mut result_blocks, row_start, panel_word, &panel_basis);
        if !panel_basis.is_empty() {
            panels.push(RrefPanel {
                panel_word,
                row_start,
                row_end: rank,
            });
        }
    }

    for panel in panels.into_iter().rev() {
        let mut basis =
            broadcast_result_panel(&world, &result_blocks, panel, matrix.cols(), block_rows)?;
        let panel_rank = basis.echelonize(true);
        if panel_rank != panel.row_end - panel.row_start {
            return Err(DistributedError::CollectiveFailure(
                "RREF panel lost rank during reverse reduction".to_owned(),
            ));
        }
        assign_panel_full_rows(&mut result_blocks, panel.row_start, &basis);
        let suffix_words = packed_words - panel.panel_word;
        let payload_words = basis
            .rows()
            .checked_mul(suffix_words)
            .ok_or(DistributedError::SizeOverflow)?;
        let mut payload = collective_zeroed_words(&world, payload_words)?;
        for row in 0..basis.rows() {
            let start = row * suffix_words;
            payload[start..start + suffix_words]
                .copy_from_slice(&basis.row_words(row)[panel.panel_word..]);
        }
        for block in &mut result_blocks {
            let target_end = panel
                .row_start
                .saturating_sub(block.row_start)
                .min(block.matrix.rows());
            if target_end != 0 {
                apply_packed_word_basis_to_rows(
                    &mut block.matrix,
                    0..target_end,
                    panel.panel_word,
                    &payload,
                );
            }
        }
    }

    row_blocks_into_distributed(&result_blocks, matrix, block_rows)?;
    Ok(rank)
}

fn assign_panel_suffix_rows(
    blocks: &mut [RankRowBlock],
    row_start: usize,
    panel_word: usize,
    basis: &[Vec<u64>],
) {
    for (offset, words) in basis.iter().enumerate() {
        let row = row_start + offset;
        if let Some(block) = blocks
            .iter_mut()
            .find(|block| block.row_start <= row && row < block.row_start + block.matrix.rows())
        {
            block.matrix.row_words_mut(row - block.row_start)[panel_word..].copy_from_slice(words);
        }
    }
}

fn assign_panel_full_rows(blocks: &mut [RankRowBlock], row_start: usize, basis: &Mzd) {
    for row in 0..basis.rows() {
        let global_row = row_start + row;
        if let Some(block) = blocks.iter_mut().find(|block| {
            block.row_start <= global_row && global_row < block.row_start + block.matrix.rows()
        }) {
            block
                .matrix
                .row_words_mut(global_row - block.row_start)
                .copy_from_slice(basis.row_words(row));
        }
    }
}

fn broadcast_result_panel(
    world: &crate::World,
    blocks: &[RankRowBlock],
    panel: RrefPanel,
    cols: usize,
    block_rows: usize,
) -> Result<Mzd, DistributedError> {
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let suffix_words = cols.div_ceil(m4rst::mzd::RADIX) - panel.panel_word;
    let mut result = collective_mzd(world, panel.row_end - panel.row_start, cols)?;
    if suffix_words == 0 {
        return Ok(result);
    }
    for owner in 0..world_size {
        let owner_rows: Vec<_> = (panel.row_start..panel.row_end)
            .filter(|row| (row / block_rows) % world_size == owner)
            .collect();
        if owner_rows.is_empty() {
            continue;
        }
        let payload_words = owner_rows
            .len()
            .checked_mul(suffix_words)
            .ok_or(DistributedError::SizeOverflow)?;
        let mut payload = collective_zeroed_words(world, payload_words)?;
        if mpi_rank == owner {
            for (offset, &row) in owner_rows.iter().enumerate() {
                payload[offset * suffix_words..(offset + 1) * suffix_words]
                    .copy_from_slice(&owned_row_words(blocks, row)[panel.panel_word..]);
            }
        }
        if !payload.is_empty() {
            let communicator = world.communicator();
            communicator
                .process_at_rank(i32::try_from(owner).map_err(|_| DistributedError::SizeOverflow)?)
                .broadcast_into(payload.as_mut_slice());
            world.record_broadcast(size_of_val(&payload[..]));
        }
        for (&row, words) in owner_rows.iter().zip(payload.chunks_exact(suffix_words)) {
            result.row_words_mut(row - panel.row_start)[panel.panel_word..].copy_from_slice(words);
        }
    }
    Ok(result)
}

fn global_pivot_in_column(
    world: &crate::World,
    blocks: &[RankRowBlock],
    start_row: usize,
    col: usize,
) -> Result<Option<usize>, DistributedError> {
    let mut local = u64::MAX;
    for block in blocks {
        let local_start = start_row
            .saturating_sub(block.row_start)
            .min(block.matrix.rows());
        if let Some(local_row) =
            first_pivot_row(&block.matrix, local_start..block.matrix.rows(), col)
        {
            let global_row = block.row_start + local_row;
            local =
                local.min(u64::try_from(global_row).map_err(|_| DistributedError::SizeOverflow)?);
        }
    }
    let selected = world.min_u64(local);
    if selected == u64::MAX {
        Ok(None)
    } else {
        Ok(Some(
            usize::try_from(selected).map_err(|_| DistributedError::SizeOverflow)?,
        ))
    }
}

fn swap_row_block_rows(
    world: &crate::World,
    blocks: &mut [RankRowBlock],
    first: usize,
    second: usize,
    block_rows: usize,
) -> Result<(), DistributedError> {
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let first_owner = (first / block_rows) % world_size;
    let second_owner = (second / block_rows) % world_size;
    if first_owner == second_owner {
        let swapped = if mpi_rank == first_owner {
            swap_owned_rows(blocks, first, second)
        } else {
            Ok(())
        };
        if !world.collective_all(swapped.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "the row owner could not allocate a swap workspace".to_owned(),
            ));
        }
        return swapped;
    }

    let local_width = blocks.first().map_or(u64::MAX, |block| {
        u64::try_from(block.matrix.width()).unwrap_or(u64::MAX)
    });
    let width =
        usize::try_from(world.min_u64(local_width)).map_err(|_| DistributedError::SizeOverflow)?;
    let mut payload = collective_zeroed_words(world, width)?;
    if mpi_rank == first_owner || mpi_rank == second_owner {
        let owned_row = if mpi_rank == first_owner {
            first
        } else {
            second
        };
        let partner = if mpi_rank == first_owner {
            second_owner
        } else {
            first_owner
        };
        payload.copy_from_slice(owned_row_words(blocks, owned_row));
        let communicator = world.communicator();
        let process = communicator
            .process_at_rank(i32::try_from(partner).map_err(|_| DistributedError::SizeOverflow)?);
        mpi::point_to_point::send_receive_replace_into(payload.as_mut_slice(), &process, &process);
        world.record_send(size_of_val(payload.as_slice()));
        world.record_receive(size_of_val(payload.as_slice()));
        drop(communicator);
        assign_owned_row(blocks, owned_row, &payload);
    }
    Ok(())
}

fn pluq_row_distributed(
    source: &DistributedMzd,
    p: &mut DistributedMzd,
    l: &mut DistributedMzd,
    u: &mut DistributedMzd,
    q: &mut DistributedMzd,
) -> Result<usize, DistributedError> {
    let world = source.world();
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let rows = source.rows();
    let cols = source.cols();
    let inner = rows.min(cols);
    let block_rows = operation_block_rows(source);
    let mut work_blocks = redistribute_full_rows(source, block_rows)?;
    let mut l_blocks = empty_row_blocks(world, rows, inner, block_rows, world_size, mpi_rank)?;
    let permutations = (|| -> Result<(Vec<usize>, Vec<usize>), DistributedError> {
        let mut row_permutation = Vec::new();
        try_reserve_items(&mut row_permutation, rows)?;
        row_permutation.extend(0..rows);
        let mut col_permutation = Vec::new();
        try_reserve_items(&mut col_permutation, cols)?;
        col_permutation.extend(0..cols);
        Ok((row_permutation, col_permutation))
    })();
    if !world.collective_all(permutations.is_ok()) {
        return Err(DistributedError::CollectiveFailure(
            "at least one rank could not allocate PLUQ permutations".to_owned(),
        ));
    }
    let (mut row_permutation, mut col_permutation) = permutations?;
    let mut rank = 0;

    while rank < inner {
        let Some((pivot_row, pivot_col)) =
            global_pivot_row_blocks(world, &work_blocks, rank, rows)?
        else {
            break;
        };
        if pivot_row != rank {
            swap_augmented_rows(
                world,
                &mut work_blocks,
                &mut l_blocks,
                pivot_row,
                rank,
                block_rows,
            )?;
            row_permutation.swap(pivot_row, rank);
        }
        if pivot_col != rank {
            for block in &mut work_blocks {
                block.matrix.swap_cols(pivot_col, rank);
            }
            col_permutation.swap(pivot_col, rank);
        }

        let pivot = broadcast_row(world, &work_blocks, rank, cols, block_rows)?;
        for (work_block, l_block) in work_blocks.iter_mut().zip(&mut l_blocks) {
            debug_assert_eq!(work_block.tile_row, l_block.tile_row);
            debug_assert_eq!(work_block.row_start, l_block.row_start);
            if work_block.row_start <= rank
                && rank < work_block.row_start + work_block.matrix.rows()
            {
                l_block.matrix.set(rank - work_block.row_start, rank, 1);
            }
            let local_start = (rank + 1)
                .saturating_sub(work_block.row_start)
                .min(work_block.matrix.rows());
            let local_end = work_block.matrix.rows();
            apply_packed_pivot_to_rows(
                &mut work_block.matrix,
                local_start..local_end,
                rank,
                &pivot,
                |local_row| l_block.matrix.set(local_row, rank, 1),
            );
        }
        rank += 1;
    }

    row_blocks_into_distributed(&l_blocks, l, block_rows)?;
    row_blocks_into_distributed(&work_blocks, u, block_rows)?;
    for (index, source_row) in row_permutation.into_iter().enumerate() {
        p.set_local_if_owned(source_row, index, 1)?;
    }
    for (index, destination) in col_permutation.into_iter().enumerate() {
        q.set_local_if_owned(index, destination, 1)?;
    }
    Ok(rank)
}

fn global_pivot_row_blocks(
    world: &crate::World,
    blocks: &[RankRowBlock],
    start: usize,
    rows: usize,
) -> Result<Option<(usize, usize)>, DistributedError> {
    let mut local = u64::MAX;
    for block in blocks {
        for local_row in 0..block.matrix.rows() {
            let global_row = block.row_start + local_row;
            if global_row < start {
                continue;
            }
            if let Some(col) = block.matrix.first_one_in_row(local_row, start) {
                let encoded = col
                    .checked_mul(rows)
                    .and_then(|value| value.checked_add(global_row))
                    .ok_or(DistributedError::SizeOverflow)?;
                local =
                    local.min(u64::try_from(encoded).map_err(|_| DistributedError::SizeOverflow)?);
            }
        }
    }
    let selected = world.min_u64(local);
    if selected == u64::MAX {
        Ok(None)
    } else {
        let selected = usize::try_from(selected).map_err(|_| DistributedError::SizeOverflow)?;
        Ok(Some((selected % rows, selected / rows)))
    }
}

fn swap_augmented_rows(
    world: &crate::World,
    work_blocks: &mut [RankRowBlock],
    l_blocks: &mut [RankRowBlock],
    first: usize,
    second: usize,
    block_rows: usize,
) -> Result<(), DistributedError> {
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let first_owner = (first / block_rows) % world_size;
    let second_owner = (second / block_rows) % world_size;
    if first_owner == second_owner {
        let swapped = if mpi_rank == first_owner {
            swap_owned_rows(work_blocks, first, second)
                .and_then(|()| swap_owned_rows(l_blocks, first, second))
        } else {
            Ok(())
        };
        if !world.collective_all(swapped.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "the row owner could not allocate PLUQ swap workspaces".to_owned(),
            ));
        }
        return swapped;
    }

    let local_work_width = work_blocks.first().map_or(u64::MAX, |block| {
        u64::try_from(block.matrix.width()).unwrap_or(u64::MAX)
    });
    let local_l_width = l_blocks.first().map_or(u64::MAX, |block| {
        u64::try_from(block.matrix.width()).unwrap_or(u64::MAX)
    });
    let work_width = usize::try_from(world.min_u64(local_work_width))
        .map_err(|_| DistributedError::SizeOverflow)?;
    let l_width = usize::try_from(world.min_u64(local_l_width))
        .map_err(|_| DistributedError::SizeOverflow)?;
    let payload_words = work_width
        .checked_add(l_width)
        .ok_or(DistributedError::SizeOverflow)?;
    let mut payload = collective_zeroed_words(world, payload_words)?;
    if mpi_rank == first_owner || mpi_rank == second_owner {
        let owned_row = if mpi_rank == first_owner {
            first
        } else {
            second
        };
        let partner = if mpi_rank == first_owner {
            second_owner
        } else {
            first_owner
        };
        payload[..work_width].copy_from_slice(owned_row_words(work_blocks, owned_row));
        payload[work_width..].copy_from_slice(owned_row_words(l_blocks, owned_row));
        let communicator = world.communicator();
        let process = communicator
            .process_at_rank(i32::try_from(partner).map_err(|_| DistributedError::SizeOverflow)?);
        mpi::point_to_point::send_receive_replace_into(payload.as_mut_slice(), &process, &process);
        world.record_send(size_of_val(payload.as_slice()));
        world.record_receive(size_of_val(payload.as_slice()));
        drop(communicator);
        assign_owned_row(work_blocks, owned_row, &payload[..work_width]);
        assign_owned_row(l_blocks, owned_row, &payload[work_width..]);
    }
    Ok(())
}

fn owned_row_words(blocks: &[RankRowBlock], row: usize) -> &[u64] {
    let block = blocks
        .iter()
        .find(|block| block.row_start <= row && row < block.row_start + block.matrix.rows())
        .expect("row owner retains its assigned block");
    block.matrix.row_words(row - block.row_start)
}

fn assign_owned_row(blocks: &mut [RankRowBlock], row: usize, words: &[u64]) {
    let block = blocks
        .iter_mut()
        .find(|block| block.row_start <= row && row < block.row_start + block.matrix.rows())
        .expect("row owner retains its assigned block");
    block
        .matrix
        .row_words_mut(row - block.row_start)
        .copy_from_slice(words);
}

fn swap_owned_rows(
    blocks: &mut [RankRowBlock],
    first: usize,
    second: usize,
) -> Result<(), DistributedError> {
    let first_block = blocks
        .iter()
        .position(|block| block.row_start <= first && first < block.row_start + block.matrix.rows())
        .expect("row owner retains its first assigned row");
    let second_block = blocks
        .iter()
        .position(|block| {
            block.row_start <= second && second < block.row_start + block.matrix.rows()
        })
        .expect("row owner retains its second assigned row");
    if first_block == second_block {
        let block = &mut blocks[first_block];
        block
            .matrix
            .swap_rows(first - block.row_start, second - block.row_start);
    } else {
        let mut first_words = Vec::new();
        try_reserve_items(&mut first_words, owned_row_words(blocks, first).len())?;
        first_words.extend_from_slice(owned_row_words(blocks, first));
        let mut second_words = Vec::new();
        try_reserve_items(&mut second_words, owned_row_words(blocks, second).len())?;
        second_words.extend_from_slice(owned_row_words(blocks, second));
        assign_owned_row(blocks, first, &second_words);
        assign_owned_row(blocks, second, &first_words);
    }
    Ok(())
}

fn broadcast_row(
    world: &crate::World,
    blocks: &[RankRowBlock],
    row: usize,
    cols: usize,
    block_rows: usize,
) -> Result<Mzd, DistributedError> {
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let owner = (row / block_rows) % world_size;
    let mut result = collective_mzd(world, 1, cols)?;
    if mpi_rank == owner {
        result
            .row_words_mut(0)
            .copy_from_slice(owned_row_words(blocks, row));
    }
    if !result.as_words().is_empty() {
        let communicator = world.communicator();
        communicator
            .process_at_rank(i32::try_from(owner).map_err(|_| DistributedError::SizeOverflow)?)
            .broadcast_into(result.as_words_mut());
        world.record_broadcast(size_of_val(result.as_words()));
    }
    Ok(result)
}

fn rank_row_distributed(matrix: &DistributedMzd) -> Result<usize, DistributedError> {
    let world = matrix.world();
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let block_rows = operation_block_rows(matrix);
    let mut blocks = redistribute_full_rows(matrix, block_rows)?;
    let packed_words = matrix.cols().div_ceil(m4rst::mzd::RADIX);
    let mut rank = 0;

    for panel_word in 0..packed_words {
        if rank == matrix.rows() {
            break;
        }
        let panel_start = panel_word * m4rst::mzd::RADIX;
        let panel_width = (matrix.cols() - panel_start).min(m4rst::mzd::RADIX);
        let suffix_words = packed_words - panel_word;
        let mut panel_rank = 0;

        for source_offset in 0..world_size {
            let source = (panel_word + source_offset) % world_size;
            let local_basis = if mpi_rank == source {
                packed_word_basis(
                    blocks.iter().map(|block| &block.matrix),
                    panel_word,
                    panel_width,
                )
            } else {
                Vec::new()
            };
            let mut basis_count = if mpi_rank == source {
                u64::try_from(local_basis.len()).map_err(|_| DistributedError::SizeOverflow)?
            } else {
                0
            };
            {
                let communicator = world.communicator();
                communicator
                    .process_at_rank(
                        i32::try_from(source).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .broadcast_into(&mut basis_count);
                world.record_broadcast(size_of::<u64>());
            }
            let basis_count =
                usize::try_from(basis_count).map_err(|_| DistributedError::SizeOverflow)?;
            let payload_words = basis_count
                .checked_mul(suffix_words)
                .ok_or(DistributedError::SizeOverflow)?;
            let mut payload = collective_zeroed_words(world, payload_words)?;
            let copied = if mpi_rank == source {
                copy_word_rows(&mut payload, &local_basis)
            } else {
                Ok(())
            };
            if !world.collective_all(copied.is_ok()) {
                return Err(DistributedError::CollectiveFailure(
                    "the rank panel owner produced an inconsistent payload".to_owned(),
                ));
            }
            copied?;
            debug_assert_eq!(payload.len(), payload_words);
            if !payload.is_empty() {
                let communicator = world.communicator();
                communicator
                    .process_at_rank(
                        i32::try_from(source).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .broadcast_into(&mut payload);
                world.record_broadcast(size_of_val(&payload[..]));
            }

            for block in &mut blocks {
                apply_packed_word_basis(&mut block.matrix, panel_word, &payload);
            }
            rank += basis_count;
            panel_rank += basis_count;
            if rank == matrix.rows() || panel_rank == panel_width {
                break;
            }
        }
    }
    Ok(rank)
}

fn redistribute_full_rows(
    matrix: &DistributedMzd,
    block_rows: usize,
) -> Result<Vec<RankRowBlock>, DistributedError> {
    let world = matrix.world();
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let block_count = matrix.rows().div_ceil(block_rows);
    let local_count = (mpi_rank..block_count).step_by(world_size).count();
    let blocks = (|| -> Result<Vec<RankRowBlock>, DistributedError> {
        let mut blocks = Vec::new();
        try_reserve_items(&mut blocks, local_count)?;
        for tile_row in 0..block_count {
            if tile_row % world_size == mpi_rank {
                let row_start = tile_row
                    .checked_mul(block_rows)
                    .ok_or(DistributedError::SizeOverflow)?;
                blocks.push(RankRowBlock {
                    tile_row,
                    row_start,
                    matrix: try_mzd(
                        row_start.saturating_add(block_rows).min(matrix.rows()) - row_start,
                        matrix.cols(),
                    )?,
                });
            }
        }
        Ok(blocks)
    })();
    if !world.collective_all(blocks.is_ok()) {
        return Err(DistributedError::CollectiveFailure(
            "at least one rank could not allocate row-distributed input workspaces".to_owned(),
        ));
    }
    let mut blocks = blocks?;

    let communicator = world.communicator();
    for coord in matrix.layout().coordinate_iter() {
        let source_rank = matrix.layout().owner(coord);
        let (source_row_start, source_row_end, col_start, col_end) =
            matrix.layout().bounds(coord)?;
        let source_width = (col_end - col_start).div_ceil(m4rst::mzd::RADIX);
        let mut segment_start = source_row_start;
        while segment_start < source_row_end {
            let destination_block = segment_start / block_rows;
            let segment_end = source_row_end.min((destination_block + 1) * block_rows);
            let destination_rank = destination_block % world_size;
            let row_offset = segment_start - source_row_start;
            let row_count = segment_end - segment_start;
            if source_rank == destination_rank {
                if mpi_rank == source_rank {
                    let tile = matrix
                        .local_tile(coord)
                        .expect("layout owner stores its canonical tile");
                    let start = row_offset * source_width;
                    let end = start + row_count * source_width;
                    copy_row_segment(
                        destination_block,
                        segment_start,
                        col_start,
                        col_end,
                        &tile.matrix().as_words()[start..end],
                        &mut blocks,
                    )?;
                }
            } else if mpi_rank == source_rank {
                let tile = matrix
                    .local_tile(coord)
                    .expect("layout owner stores its canonical tile");
                let start = row_offset * source_width;
                let end = start + row_count * source_width;
                let words = &tile.matrix().as_words()[start..end];
                communicator
                    .process_at_rank(
                        i32::try_from(destination_rank)
                            .map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .send(words);
                world.record_send(size_of_val(words));
            } else if mpi_rank == destination_rank {
                let (words, _) = communicator
                    .process_at_rank(
                        i32::try_from(source_rank).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .receive_vec::<u64>();
                world.record_receive(size_of_val(&words[..]));
                copy_row_segment(
                    destination_block,
                    segment_start,
                    col_start,
                    col_end,
                    &words,
                    &mut blocks,
                )?;
            }
            segment_start = segment_end;
        }
    }
    communicator.barrier();
    world.record_barrier();
    Ok(blocks)
}

fn copy_row_segment(
    destination_block: usize,
    segment_start: usize,
    col_start: usize,
    col_end: usize,
    words: &[u64],
    blocks: &mut [RankRowBlock],
) -> Result<(), DistributedError> {
    let block = blocks
        .iter_mut()
        .find(|block| block.tile_row == destination_block)
        .expect("row-layout owner retains its assigned block");
    let source_width = (col_end - col_start).div_ceil(m4rst::mzd::RADIX);
    if source_width == 0 || !words.len().is_multiple_of(source_width) {
        return Err(DistributedError::CollectiveFailure(
            "row redistribution received a truncated packed segment".to_owned(),
        ));
    }
    let destination_word = col_start / m4rst::mzd::RADIX;
    for (row, source) in words.chunks_exact(source_width).enumerate() {
        let destination = block
            .matrix
            .row_words_mut(segment_start - block.row_start + row);
        destination[destination_word..destination_word + source_width].copy_from_slice(source);
    }
    Ok(())
}

fn solve_row_distributed(
    coefficients: &DistributedMzd,
    rhs: &DistributedMzd,
) -> Result<Option<DistributedMzd>, DistributedError> {
    let world = coefficients.world();
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let block_rows = operation_block_rows(coefficients);
    let mut coefficient_blocks = redistribute_full_rows(coefficients, block_rows)?;
    let mut rhs_blocks = redistribute_full_rows(rhs, block_rows)?;
    debug_assert!(coefficient_blocks
        .iter()
        .zip(&rhs_blocks)
        .all(|(left, right)| left.tile_row == right.tile_row && left.row_start == right.row_start));
    let mut result_coefficient_blocks = empty_row_blocks(
        world,
        coefficients.rows(),
        coefficients.cols(),
        block_rows,
        world_size,
        mpi_rank,
    )?;
    let mut result_rhs_blocks = empty_row_blocks(
        world,
        coefficients.rows(),
        rhs.cols(),
        block_rows,
        world_size,
        mpi_rank,
    )?;
    let packed_words = coefficients.cols().div_ceil(m4rst::mzd::RADIX);
    let rhs_words = rhs.cols().div_ceil(m4rst::mzd::RADIX);
    let mut panels = Vec::new();
    let mut pivot_cols = Vec::new();
    let mut rank = 0;

    for panel_word in 0..packed_words {
        if rank == coefficients.rows() {
            break;
        }
        let panel_start = panel_word * m4rst::mzd::RADIX;
        let panel_width = (coefficients.cols() - panel_start).min(m4rst::mzd::RADIX);
        let coefficient_suffix_words = packed_words - panel_word;
        let row_start = rank;
        let mut panel_basis = Vec::new();

        for source_offset in 0..world_size {
            let source = (panel_word + source_offset) % world_size;
            let (local_coefficients, local_rhs) = if mpi_rank == source {
                packed_word_basis_with_companion(
                    coefficient_blocks.iter().zip(&rhs_blocks).map(
                        |(coefficient_block, rhs_block)| {
                            (&coefficient_block.matrix, &rhs_block.matrix)
                        },
                    ),
                    panel_word,
                    panel_width,
                )
            } else {
                (Vec::new(), Vec::new())
            };
            let mut basis_count = if mpi_rank == source {
                u64::try_from(local_coefficients.len())
                    .map_err(|_| DistributedError::SizeOverflow)?
            } else {
                0
            };
            {
                let communicator = world.communicator();
                communicator
                    .process_at_rank(
                        i32::try_from(source).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .broadcast_into(&mut basis_count);
                world.record_broadcast(size_of::<u64>());
            }
            let basis_count =
                usize::try_from(basis_count).map_err(|_| DistributedError::SizeOverflow)?;
            let coefficient_payload_words = basis_count
                .checked_mul(coefficient_suffix_words)
                .ok_or(DistributedError::SizeOverflow)?;
            let rhs_payload_words = basis_count
                .checked_mul(rhs_words)
                .ok_or(DistributedError::SizeOverflow)?;
            let mut coefficient_payload =
                collective_zeroed_words(world, coefficient_payload_words)?;
            let mut rhs_payload = collective_zeroed_words(world, rhs_payload_words)?;
            let copied = if mpi_rank == source {
                copy_word_rows(&mut coefficient_payload, &local_coefficients)
                    .and_then(|()| copy_word_rows(&mut rhs_payload, &local_rhs))
            } else {
                Ok(())
            };
            if !world.collective_all(copied.is_ok()) {
                return Err(DistributedError::CollectiveFailure(
                    "the solve panel owner produced inconsistent payloads".to_owned(),
                ));
            }
            copied?;
            debug_assert_eq!(coefficient_payload.len(), coefficient_payload_words);
            debug_assert_eq!(rhs_payload.len(), rhs_payload_words);
            if !coefficient_payload.is_empty() {
                let communicator = world.communicator();
                communicator
                    .process_at_rank(
                        i32::try_from(source).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .broadcast_into(coefficient_payload.as_mut_slice());
                world.record_broadcast(size_of_val(&coefficient_payload[..]));
            }
            if !rhs_payload.is_empty() {
                let communicator = world.communicator();
                communicator
                    .process_at_rank(
                        i32::try_from(source).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .broadcast_into(rhs_payload.as_mut_slice());
                world.record_broadcast(size_of_val(&rhs_payload[..]));
            }

            for (coefficient_block, rhs_block) in coefficient_blocks.iter_mut().zip(&mut rhs_blocks)
            {
                let local_rows = coefficient_block.matrix.rows();
                apply_packed_word_basis_with_companion_to_rows(
                    &mut coefficient_block.matrix,
                    &mut rhs_block.matrix,
                    0..local_rows,
                    panel_word,
                    &coefficient_payload,
                    &rhs_payload,
                );
            }
            for basis_index in 0..basis_count {
                let coefficient_start = basis_index * coefficient_suffix_words;
                let rhs_start = basis_index * rhs_words;
                panel_basis.push((
                    coefficient_payload
                        [coefficient_start..coefficient_start + coefficient_suffix_words]
                        .to_vec(),
                    rhs_payload[rhs_start..rhs_start + rhs_words].to_vec(),
                ));
            }
            rank += basis_count;
            if rank == coefficients.rows() || panel_basis.len() == panel_width {
                break;
            }
        }

        panel_basis
            .sort_unstable_by_key(|(coefficient_row, _)| coefficient_row[0].trailing_zeros());
        debug_assert!(panel_basis
            .windows(2)
            .all(|rows| { rows[0].0[0].trailing_zeros() < rows[1].0[0].trailing_zeros() }));
        pivot_cols.extend(panel_basis.iter().map(|(coefficient_row, _)| {
            panel_start + coefficient_row[0].trailing_zeros() as usize
        }));
        assign_paired_panel_rows(
            &mut result_coefficient_blocks,
            &mut result_rhs_blocks,
            row_start,
            panel_word,
            &panel_basis,
        );
        if !panel_basis.is_empty() {
            panels.push(RrefPanel {
                panel_word,
                row_start,
                row_end: rank,
            });
        }
    }

    let locally_consistent =
        coefficient_blocks
            .iter()
            .zip(&rhs_blocks)
            .all(|(coefficient_block, rhs_block)| {
                (0..coefficient_block.matrix.rows()).all(|row| {
                    coefficient_block.matrix.first_one_in_row(row, 0).is_some()
                        || rhs_block.matrix.first_one_in_row(row, 0).is_none()
                })
            });
    if !world.collective_all(locally_consistent) {
        return Ok(None);
    }

    for panel in panels.into_iter().rev() {
        let mut coefficient_basis = broadcast_result_panel(
            world,
            &result_coefficient_blocks,
            panel,
            coefficients.cols(),
            block_rows,
        )?;
        let mut rhs_basis = broadcast_result_panel(
            world,
            &result_rhs_blocks,
            RrefPanel {
                panel_word: 0,
                ..panel
            },
            rhs.cols(),
            block_rows,
        )?;
        let reduced_pivots = rref_packed_word_basis_with_companion(
            &mut coefficient_basis,
            &mut rhs_basis,
            panel.panel_word,
        );
        if reduced_pivots != pivot_cols[panel.row_start..panel.row_end] {
            return Err(DistributedError::CollectiveFailure(
                "solve panel changed pivots during reverse reduction".to_owned(),
            ));
        }
        assign_panel_full_rows(
            &mut result_coefficient_blocks,
            panel.row_start,
            &coefficient_basis,
        );
        assign_panel_full_rows(&mut result_rhs_blocks, panel.row_start, &rhs_basis);

        let coefficient_suffix_words = packed_words - panel.panel_word;
        let coefficient_payload_words = coefficient_basis
            .rows()
            .checked_mul(coefficient_suffix_words)
            .ok_or(DistributedError::SizeOverflow)?;
        let rhs_payload_words = rhs_basis
            .rows()
            .checked_mul(rhs_words)
            .ok_or(DistributedError::SizeOverflow)?;
        let mut coefficient_payload = collective_zeroed_words(world, coefficient_payload_words)?;
        let mut rhs_payload = collective_zeroed_words(world, rhs_payload_words)?;
        for row in 0..coefficient_basis.rows() {
            coefficient_payload
                [row * coefficient_suffix_words..(row + 1) * coefficient_suffix_words]
                .copy_from_slice(&coefficient_basis.row_words(row)[panel.panel_word..]);
            rhs_payload[row * rhs_words..(row + 1) * rhs_words]
                .copy_from_slice(rhs_basis.row_words(row));
        }
        for (coefficient_block, rhs_block) in result_coefficient_blocks
            .iter_mut()
            .zip(&mut result_rhs_blocks)
        {
            let target_end = panel
                .row_start
                .saturating_sub(coefficient_block.row_start)
                .min(coefficient_block.matrix.rows());
            if target_end != 0 {
                apply_packed_word_basis_with_companion_to_rows(
                    &mut coefficient_block.matrix,
                    &mut rhs_block.matrix,
                    0..target_end,
                    panel.panel_word,
                    &coefficient_payload,
                    &rhs_payload,
                );
            }
        }
    }

    let mut solution_blocks = empty_row_blocks(
        world,
        coefficients.cols(),
        rhs.cols(),
        block_rows,
        world_size,
        mpi_rank,
    )?;
    {
        let communicator = world.communicator();
        for (equation_row, solution_row) in pivot_cols.into_iter().enumerate() {
            let source_rank = (equation_row / block_rows) % world_size;
            let destination_rank = (solution_row / block_rows) % world_size;
            if source_rank == destination_rank {
                if mpi_rank == source_rank {
                    let source_block = result_rhs_blocks
                        .iter()
                        .find(|block| block.tile_row == equation_row / block_rows)
                        .expect("equation row owner retains its right-hand-side block");
                    copy_solution_row(
                        solution_row,
                        source_block
                            .matrix
                            .row_words(equation_row - source_block.row_start),
                        &mut solution_blocks,
                    );
                }
            } else if mpi_rank == source_rank {
                let source_block = result_rhs_blocks
                    .iter()
                    .find(|block| block.tile_row == equation_row / block_rows)
                    .expect("equation row owner retains its right-hand-side block");
                let words = source_block
                    .matrix
                    .row_words(equation_row - source_block.row_start);
                communicator
                    .process_at_rank(
                        i32::try_from(destination_rank)
                            .map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .send(words);
                world.record_send(size_of_val(words));
            } else if mpi_rank == destination_rank {
                let (words, _) = communicator
                    .process_at_rank(
                        i32::try_from(source_rank).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .receive_vec::<u64>();
                world.record_receive(size_of_val(&words[..]));
                copy_solution_row(solution_row, &words, &mut solution_blocks);
            }
        }
        communicator.barrier();
        world.record_barrier();
    }
    Ok(Some(row_blocks_to_distributed(
        world,
        &solution_blocks,
        coefficients.cols(),
        rhs.cols(),
        block_rows,
        coefficients.layout().tile_rows(),
        rhs.layout().tile_cols(),
    )?))
}

fn assign_paired_panel_rows(
    coefficient_blocks: &mut [RankRowBlock],
    companion_blocks: &mut [RankRowBlock],
    row_start: usize,
    panel_word: usize,
    basis: &[(Vec<u64>, Vec<u64>)],
) {
    for (offset, (coefficient_words, companion_words)) in basis.iter().enumerate() {
        let row = row_start + offset;
        if let Some((coefficient_block, companion_block)) = coefficient_blocks
            .iter_mut()
            .zip(companion_blocks.iter_mut())
            .find(|(coefficient_block, companion_block)| {
                debug_assert_eq!(coefficient_block.row_start, companion_block.row_start);
                coefficient_block.row_start <= row
                    && row < coefficient_block.row_start + coefficient_block.matrix.rows()
            })
        {
            let local_row = row - coefficient_block.row_start;
            coefficient_block.matrix.row_words_mut(local_row)[panel_word..]
                .copy_from_slice(coefficient_words);
            companion_block
                .matrix
                .row_words_mut(local_row)
                .copy_from_slice(companion_words);
        }
    }
}

fn empty_row_blocks(
    world: &crate::World,
    rows: usize,
    cols: usize,
    block_rows: usize,
    world_size: usize,
    mpi_rank: usize,
) -> Result<Vec<RankRowBlock>, DistributedError> {
    let blocks = (|| -> Result<Vec<RankRowBlock>, DistributedError> {
        let block_count = rows.div_ceil(block_rows);
        let local_count = (mpi_rank..block_count).step_by(world_size).count();
        let mut blocks = Vec::new();
        try_reserve_items(&mut blocks, local_count)?;
        for tile_row in (mpi_rank..block_count).step_by(world_size) {
            let row_start = tile_row
                .checked_mul(block_rows)
                .ok_or(DistributedError::SizeOverflow)?;
            blocks.push(RankRowBlock {
                tile_row,
                row_start,
                matrix: try_mzd(
                    row_start.saturating_add(block_rows).min(rows) - row_start,
                    cols,
                )?,
            });
        }
        Ok(blocks)
    })();
    if !world.collective_all(blocks.is_ok()) {
        return Err(DistributedError::CollectiveFailure(
            "at least one rank could not allocate row-distributed output workspaces".to_owned(),
        ));
    }
    blocks
}

fn copy_solution_row(row: usize, words: &[u64], blocks: &mut [RankRowBlock]) {
    let block = blocks
        .iter_mut()
        .find(|block| block.row_start <= row && row < block.row_start + block.matrix.rows())
        .expect("solution row owner retains its assigned block");
    block
        .matrix
        .row_words_mut(row - block.row_start)
        .copy_from_slice(words);
}

fn row_blocks_to_distributed(
    world: &crate::World,
    blocks: &[RankRowBlock],
    rows: usize,
    cols: usize,
    block_rows: usize,
    tile_rows: usize,
    tile_cols: usize,
) -> Result<DistributedMzd, DistributedError> {
    let mut output = DistributedMzd::with_tiles(world, rows, cols, tile_rows, tile_cols)?;
    row_blocks_into_distributed(blocks, &mut output, block_rows)?;
    Ok(output)
}

fn row_blocks_into_distributed(
    blocks: &[RankRowBlock],
    output: &mut DistributedMzd,
    block_rows: usize,
) -> Result<(), DistributedError> {
    output.clear();
    let world = output.world().clone();
    let layout = output.layout().clone();
    let world_size = usize::try_from(world.size()).expect("MPI world sizes are positive");
    let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
    let transfer_words = layout
        .tile_rows()
        .min(block_rows)
        .checked_mul(layout.tile_cols().div_ceil(m4rst::mzd::RADIX))
        .ok_or(DistributedError::SizeOverflow)?;
    let mut transfer = collective_zeroed_words(&world, transfer_words)?;
    let communicator = world.communicator();
    for coord in layout.coordinate_iter() {
        let destination_rank = layout.owner(coord);
        let (row_start, row_end, col_start, col_end) = layout.bounds(coord)?;
        let word_start = col_start / m4rst::mzd::RADIX;
        let word_count = (col_end - col_start).div_ceil(m4rst::mzd::RADIX);
        let mut segment_start = row_start;
        while segment_start < row_end {
            let block_index = segment_start / block_rows;
            let source_rank = block_index % world_size;
            let segment_end = row_end.min(
                block_index
                    .checked_add(1)
                    .and_then(|index| index.checked_mul(block_rows))
                    .ok_or(DistributedError::SizeOverflow)?,
            );
            let segment_rows = segment_end - segment_start;
            let payload_words = segment_rows
                .checked_mul(word_count)
                .ok_or(DistributedError::SizeOverflow)?;
            if source_rank == destination_rank {
                if mpi_rank == source_rank {
                    let source = blocks
                        .iter()
                        .find(|block| block.tile_row == block_index)
                        .expect("row owner retains its assigned output block");
                    let destination = output
                        .local_tile_mut(coord)
                        .expect("layout owner stores its canonical output tile")
                        .matrix_mut();
                    for row in segment_start..segment_end {
                        destination.row_words_mut(row - row_start).copy_from_slice(
                            &source.matrix.row_words(row - source.row_start)
                                [word_start..word_start + word_count],
                        );
                    }
                }
            } else if mpi_rank == source_rank {
                let source = blocks
                    .iter()
                    .find(|block| block.tile_row == block_index)
                    .expect("row owner retains its assigned output block");
                for (offset, row) in (segment_start..segment_end).enumerate() {
                    transfer[offset * word_count..(offset + 1) * word_count].copy_from_slice(
                        &source.matrix.row_words(row - source.row_start)
                            [word_start..word_start + word_count],
                    );
                }
                communicator
                    .process_at_rank(
                        i32::try_from(destination_rank)
                            .map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .send(&transfer[..payload_words]);
                world.record_send(size_of_val(&transfer[..payload_words]));
            } else if mpi_rank == destination_rank {
                communicator
                    .process_at_rank(
                        i32::try_from(source_rank).map_err(|_| DistributedError::SizeOverflow)?,
                    )
                    .receive_into(&mut transfer[..payload_words]);
                world.record_receive(size_of_val(&transfer[..payload_words]));
                let destination = output
                    .local_tile_mut(coord)
                    .expect("layout owner stores its canonical output tile")
                    .matrix_mut();
                for (offset, words) in transfer[..payload_words]
                    .chunks_exact(word_count)
                    .enumerate()
                {
                    destination
                        .row_words_mut(segment_start - row_start + offset)
                        .copy_from_slice(words);
                }
            }
            segment_start = segment_end;
        }
    }
    communicator.barrier();
    world.record_barrier();
    drop(communicator);
    Ok(())
}

#[cfg(test)]
mod tests {
    use m4rst::{
        echelon::{apply_packed_word_basis, packed_word_basis},
        Mzd,
    };

    use super::RankRowBlock;

    fn simulated_panel_rank(matrix: &Mzd, processes: usize, block_rows: usize) -> usize {
        let mut process_blocks: Vec<Vec<RankRowBlock>> =
            (0..processes).map(|_| Vec::new()).collect();
        for (tile_row, row_start) in (0..matrix.rows()).step_by(block_rows).enumerate() {
            let row_end = (row_start + block_rows).min(matrix.rows());
            process_blocks[tile_row % processes].push(RankRowBlock {
                tile_row,
                row_start,
                matrix: matrix.submatrix(row_start, 0, row_end, matrix.cols()),
            });
        }

        let packed_words = matrix.cols().div_ceil(m4rst::mzd::RADIX);
        let mut rank = 0;
        for panel_word in 0..packed_words {
            let panel_start = panel_word * m4rst::mzd::RADIX;
            let panel_width = (matrix.cols() - panel_start).min(m4rst::mzd::RADIX);
            let mut panel_rank = 0;
            for source_offset in 0..processes {
                let source = (panel_word + source_offset) % processes;
                let basis = packed_word_basis(
                    process_blocks[source].iter().map(|block| &block.matrix),
                    panel_word,
                    panel_width,
                );
                let basis_count = basis.len();
                let payload: Vec<u64> = basis.into_iter().flatten().collect();
                for blocks in &mut process_blocks {
                    for block in blocks {
                        apply_packed_word_basis(&mut block.matrix, panel_word, &payload);
                    }
                }
                rank += basis_count;
                panel_rank += basis_count;
                if rank == matrix.rows() || panel_rank == panel_width {
                    break;
                }
            }
        }
        rank
    }

    #[test]
    fn packed_word_panel_rank_matches_local_echelonization() {
        let matrix = Mzd::random(97, 130, 0x1729);
        let expected = matrix.rank();
        assert!(expected > m4rst::mzd::RADIX);
        for processes in [1, 3, 5] {
            assert_eq!(simulated_panel_rank(&matrix, processes, 11), expected);
        }

        let zero = Mzd::new(23, 129);
        for processes in [1, 2, 7] {
            assert_eq!(simulated_panel_rank(&zero, processes, 4), 0);
        }
    }
}
