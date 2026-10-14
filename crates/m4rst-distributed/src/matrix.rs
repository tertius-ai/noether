// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Distributed packed matrix representation, bounded collection, and persistence.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use m4rst::{MatrixWrite, Mzd, MzdAllocationError};
use mpi::topology::{Color, Communicator};
use mpi::traits::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::layout::default_tile_shape;
use crate::manifest::{
    bytes_to_words, checksum, checksum_words, tile_filename, words_to_bytes, MatrixManifest,
    TileRecord, FORMAT_VERSION, MANIFEST_NAME,
};
use crate::random::random_tile;
use crate::world::MatmulPhase;
use crate::{DistributedError, Layout, TileCoord, World};

const MATMUL_LEFT_PANEL_BATCH_BYTES: usize = 8 * 1024 * 1024;

/// One rank-owned packed tile.
#[derive(Clone, Debug)]
pub struct LocalTile {
    coord: TileCoord,
    matrix: Arc<Mzd>,
}

impl LocalTile {
    /// Logical tile-grid coordinate.
    #[must_use]
    pub const fn coord(&self) -> TileCoord {
        self.coord
    }

    /// Rank-local packed matrix value.
    #[must_use]
    pub fn matrix(&self) -> &Mzd {
        &self.matrix
    }

    pub(crate) fn matrix_mut(&mut self) -> &mut Mzd {
        Arc::make_mut(&mut self.matrix)
    }
}

/// A dense packed `F_2` matrix distributed block-cyclically across MPI ranks.
///
/// This type retains its explicit [`World`]. Matrix-scale storage consists only
/// of rank-owned [`Mzd`] tiles; no rank holds a hidden complete matrix.
#[derive(Clone, Debug)]
pub struct DistributedMzd {
    world: World,
    layout: Layout,
    local_tiles: Vec<LocalTile>,
    seed: Option<u64>,
}

impl DistributedMzd {
    /// Construct a zero matrix with default tile geometry.
    pub fn new(world: &World, rows: usize, cols: usize) -> Result<Self, DistributedError> {
        let (tile_rows, tile_cols) = default_tile_shape(
            rows,
            cols,
            usize::try_from(world.size()).map_err(|_| DistributedError::SizeOverflow)?,
        )?;
        Self::with_tiles(world, rows, cols, tile_rows, tile_cols)
    }

    /// Construct a zero matrix with explicit word-aligned tile geometry.
    pub fn with_tiles(
        world: &World,
        rows: usize,
        cols: usize,
        tile_rows: usize,
        tile_cols: usize,
    ) -> Result<Self, DistributedError> {
        let layout = collective_layout(world, rows, cols, tile_rows, tile_cols)?;
        Self::from_generator(world, layout, None, |layout, coord| {
            let (row_start, row_end, col_start, col_end) = layout
                .bounds(coord)
                .expect("canonical tile coordinate is valid");
            try_mzd(row_end - row_start, col_end - col_start)
        })
    }

    /// Construct a square identity matrix with default tile geometry.
    pub fn identity(world: &World, size: usize) -> Result<Self, DistributedError> {
        let (tile_rows, tile_cols) = default_tile_shape(
            size,
            size,
            usize::try_from(world.size()).map_err(|_| DistributedError::SizeOverflow)?,
        )?;
        Self::identity_with_tiles(world, size, tile_rows, tile_cols)
    }

    /// Construct a square identity matrix with explicit tile geometry.
    pub fn identity_with_tiles(
        world: &World,
        size: usize,
        tile_rows: usize,
        tile_cols: usize,
    ) -> Result<Self, DistributedError> {
        let layout = collective_layout(world, size, size, tile_rows, tile_cols)?;
        Self::from_generator(world, layout, None, |layout, coord| {
            let (row_start, row_end, col_start, col_end) = layout
                .bounds(coord)
                .expect("canonical tile coordinate is valid");
            Mzd::try_identity_window(row_start, row_end, col_start, col_end)
                .map_err(map_mzd_allocation)
        })
    }

    pub(crate) fn row_shifted_identity_with_tiles(
        world: &World,
        rows: usize,
        cols: usize,
        row_shift: usize,
        tile_rows: usize,
        tile_cols: usize,
    ) -> Result<Self, DistributedError> {
        let layout = collective_layout(world, rows, cols, tile_rows, tile_cols)?;
        world.agree_u64(
            u64::try_from(row_shift).map_err(|_| DistributedError::SizeOverflow)?,
            "row_shift",
        )?;
        if row_shift > rows || cols > rows - row_shift {
            return Err(DistributedError::ShapeMismatch(format!(
                "a {rows} x {cols} row-shifted identity cannot start at row {row_shift}"
            )));
        }
        Self::from_generator(world, layout, None, |layout, coord| {
            let (row_start, row_end, col_start, col_end) = layout
                .bounds(coord)
                .expect("canonical tile coordinate is valid");
            Mzd::try_identity_window(
                row_start,
                row_end,
                row_shift + col_start,
                row_shift + col_end,
            )
            .map_err(map_mzd_allocation)
        })
    }

    /// Construct a deterministic matrix bit-identical to [`Mzd::random`]
    /// without generating the complete value on rank zero.
    pub fn random(
        world: &World,
        rows: usize,
        cols: usize,
        seed: u64,
    ) -> Result<Self, DistributedError> {
        let (tile_rows, tile_cols) = default_tile_shape(
            rows,
            cols,
            usize::try_from(world.size()).map_err(|_| DistributedError::SizeOverflow)?,
        )?;
        Self::random_with_tiles(world, rows, cols, seed, tile_rows, tile_cols)
    }

    /// Deterministic distributed construction with explicit tile geometry.
    pub fn random_with_tiles(
        world: &World,
        rows: usize,
        cols: usize,
        seed: u64,
        tile_rows: usize,
        tile_cols: usize,
    ) -> Result<Self, DistributedError> {
        world.agree_u64(seed, "seed")?;
        let layout = collective_layout(world, rows, cols, tile_rows, tile_cols)?;
        Self::from_generator(world, layout, Some(seed), |layout, coord| {
            random_tile(layout, coord, seed)
        })
    }

    /// Construct from a bounded row-major byte buffer supplied only on
    /// `root`. Each byte contributes its low bit, matching
    /// [`Mzd::from_byte_buffer`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_root_byte_buffer(
        world: &World,
        rows: usize,
        cols: usize,
        root_buffer: Option<&[u8]>,
        root: i32,
        max_bytes: usize,
        tile_rows: usize,
        tile_cols: usize,
    ) -> Result<Self, DistributedError> {
        world.agree_rank(root, "root")?;
        world.agree_u64(
            u64::try_from(max_bytes).map_err(|_| DistributedError::SizeOverflow)?,
            "max_bytes",
        )?;
        let layout = collective_layout(world, rows, cols, tile_rows, tile_cols)?;
        let required = rows
            .checked_mul(cols)
            .ok_or(DistributedError::SizeOverflow)?;
        if required > max_bytes {
            return Err(DistributedError::MaterializationLimit {
                required,
                limit: max_bytes,
            });
        }
        let role_is_valid = if world.mpi_rank() == root {
            root_buffer.is_some()
        } else {
            root_buffer.is_none()
        };
        if !world.collective_all(role_is_valid) {
            return Err(DistributedError::RootBuffer(
                "exactly the declared root must supply the buffer".to_owned(),
            ));
        }
        let root_error = if world.mpi_rank() == root {
            let buffer = root_buffer.expect("root role was checked");
            (buffer.len() != required)
                .then(|| format!("buffer has {} bytes, expected {required}", buffer.len()))
        } else {
            None
        };
        let error_payload = world.broadcast_bytes(
            root,
            (world.mpi_rank() == root).then(|| root_error.as_deref().unwrap_or("").as_bytes()),
        )?;
        if !error_payload.is_empty() {
            return Err(DistributedError::RootBuffer(
                String::from_utf8_lossy(&error_payload).into_owned(),
            ));
        }

        Self::from_root_tiles(world, layout, root, |layout, coord| {
            pack_root_tile(layout, coord, root_buffer.expect("root role was checked"))
        })
    }

    /// Construct from one root-owned packed [`Mzd`] under an explicit packed
    /// byte ceiling. Tiles are copied directly from packed words; no
    /// byte-per-entry or coefficient-object intermediate is created.
    #[allow(clippy::too_many_arguments)]
    pub fn from_root_packed(
        world: &World,
        rows: usize,
        cols: usize,
        root_matrix: Option<&Mzd>,
        root: i32,
        max_bytes: usize,
        tile_rows: usize,
        tile_cols: usize,
    ) -> Result<Self, DistributedError> {
        world.agree_rank(root, "root")?;
        world.agree_u64(
            u64::try_from(max_bytes).map_err(|_| DistributedError::SizeOverflow)?,
            "max_bytes",
        )?;
        let layout = collective_layout(world, rows, cols, tile_rows, tile_cols)?;
        let required = rows
            .checked_mul(cols.div_ceil(m4rst::mzd::RADIX))
            .and_then(|words| words.checked_mul(size_of::<u64>()))
            .ok_or(DistributedError::SizeOverflow)?;
        if required > max_bytes {
            return Err(DistributedError::MaterializationLimit {
                required,
                limit: max_bytes,
            });
        }
        let role_is_valid = if world.mpi_rank() == root {
            root_matrix.is_some()
        } else {
            root_matrix.is_none()
        };
        if !world.collective_all(role_is_valid) {
            return Err(DistributedError::RootBuffer(
                "exactly the declared root must supply the packed matrix".to_owned(),
            ));
        }
        let root_error = if world.mpi_rank() == root {
            let matrix = root_matrix.expect("root role was checked");
            (matrix.rows(), matrix.cols()).ne(&(rows, cols)).then(|| {
                format!(
                    "packed root matrix is {} x {}, expected {rows} x {cols}",
                    matrix.rows(),
                    matrix.cols()
                )
            })
        } else {
            None
        };
        let error_payload = world.broadcast_bytes(
            root,
            (world.mpi_rank() == root).then(|| root_error.as_deref().unwrap_or("").as_bytes()),
        )?;
        if !error_payload.is_empty() {
            return Err(DistributedError::RootBuffer(
                String::from_utf8_lossy(&error_payload).into_owned(),
            ));
        }

        Self::from_root_tiles(world, layout, root, |layout, coord| {
            let (row_start, row_end, col_start, col_end) = layout.bounds(coord)?;
            root_matrix
                .expect("root role was checked")
                .try_submatrix(row_start, col_start, row_end, col_end)
                .map_err(map_mzd_allocation)
        })
    }

    fn from_root_tiles(
        world: &World,
        layout: Layout,
        root: i32,
        mut generate: impl FnMut(&Layout, TileCoord) -> Result<Mzd, DistributedError>,
    ) -> Result<Self, DistributedError> {
        let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
        let local_count = layout
            .coordinate_iter()
            .filter(|coord| layout.owner(*coord) == mpi_rank)
            .count();
        let mut local_tiles = Vec::new();
        let local_allocation = try_reserve_items(&mut local_tiles, local_count);
        if !world.collective_all(local_allocation.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not allocate its root-distributed tile table".to_owned(),
            ));
        }
        local_allocation?;

        for coord in layout.coordinate_iter() {
            let owner =
                i32::try_from(layout.owner(coord)).map_err(|_| DistributedError::SizeOverflow)?;
            let prepared = if world.mpi_rank() == root {
                generate(&layout, coord).map(Some)
            } else if world.mpi_rank() == owner {
                let (row_start, row_end, col_start, col_end) = layout.bounds(coord)?;
                try_mzd(row_end - row_start, col_end - col_start).map(Some)
            } else {
                Ok(None)
            };
            if !world.collective_all(prepared.is_ok()) {
                return Err(DistributedError::CollectiveFailure(
                    "at least one rank could not allocate a root-distributed tile".to_owned(),
                ));
            }
            let mut tile = prepared?;
            if owner == root {
                if world.mpi_rank() == root {
                    local_tiles.push(LocalTile {
                        coord,
                        matrix: Arc::new(tile.take().expect("root generated its owned tile")),
                    });
                }
            } else if world.mpi_rank() == root {
                let tile = tile.as_ref().expect("root generated the outgoing tile");
                let communicator = world.communicator();
                communicator.process_at_rank(owner).send(tile.as_words());
                world.record_send(size_of_val(tile.as_words()));
            } else if world.mpi_rank() == owner {
                let mut tile = tile.take().expect("owner allocated its incoming tile");
                let communicator = world.communicator();
                communicator
                    .process_at_rank(root)
                    .receive_into(tile.as_words_mut());
                world.record_receive(size_of_val(tile.as_words()));
                local_tiles.push(LocalTile {
                    coord,
                    matrix: Arc::new(tile),
                });
            }
        }
        Ok(Self {
            world: world.clone(),
            layout,
            local_tiles,
            seed: None,
        })
    }

    fn from_generator(
        world: &World,
        layout: Layout,
        seed: Option<u64>,
        generator: impl Fn(&Layout, TileCoord) -> Result<Mzd, DistributedError> + Sync,
    ) -> Result<Self, DistributedError> {
        let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
        let local_count = layout
            .coordinate_iter()
            .filter(|coord| layout.owner(*coord) == mpi_rank)
            .count();
        let local_tiles = (|| -> Result<Vec<LocalTile>, DistributedError> {
            let mut local_coords = Vec::new();
            try_reserve_items(&mut local_coords, local_count)?;
            local_coords.extend(
                layout
                    .coordinate_iter()
                    .filter(|coord| layout.owner(*coord) == mpi_rank),
            );
            #[cfg(feature = "parallel")]
            {
                let mut generated = Vec::new();
                try_reserve_items(&mut generated, local_count)?;
                generated.resize_with(local_count, || None);
                generated
                    .par_iter_mut()
                    .zip(local_coords.par_iter())
                    .for_each(|(slot, &coord)| {
                        *slot = Some(generator(&layout, coord).map(|matrix| LocalTile {
                            coord,
                            matrix: Arc::new(matrix),
                        }));
                    });
                let mut local_tiles = Vec::new();
                try_reserve_items(&mut local_tiles, local_count)?;
                for tile in generated {
                    local_tiles.push(tile.expect("every parallel tile slot was filled")?);
                }
                Ok(local_tiles)
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut local_tiles = Vec::new();
                try_reserve_items(&mut local_tiles, local_count)?;
                for coord in local_coords {
                    local_tiles.push(LocalTile {
                        coord,
                        matrix: Arc::new(generator(&layout, coord)?),
                    });
                }
                Ok(local_tiles)
            }
        })();
        if !world.collective_all(local_tiles.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not allocate its canonical matrix tiles".to_owned(),
            ));
        }
        let local_tiles = local_tiles?;
        Ok(Self {
            world: world.clone(),
            layout,
            local_tiles,
            seed,
        })
    }

    /// Explicit MPI world retained by this value.
    #[must_use]
    pub const fn world(&self) -> &World {
        &self.world
    }

    /// Logical row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.layout.rows()
    }

    /// Logical column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.layout.cols()
    }

    /// Logical `(rows, columns)` shape.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Current process-grid and tile geometry.
    #[must_use]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Rank-local packed tiles in canonical coordinate order.
    #[must_use]
    pub fn local_tiles(&self) -> &[LocalTile] {
        &self.local_tiles
    }

    /// Bytes of packed matrix payload owned by this rank.
    #[must_use]
    pub fn local_memory_bytes(&self) -> usize {
        self.local_tiles
            .iter()
            .map(|tile| size_of_val(tile.matrix.as_words()))
            .sum()
    }

    /// Total packed payload bytes across all ranks, excluding bounded
    /// communication buffers and metadata.
    #[must_use]
    pub fn global_memory_bytes(&self) -> Option<usize> {
        self.rows()
            .checked_mul(self.cols().div_ceil(m4rst::mzd::RADIX))?
            .checked_mul(size_of::<u64>())
    }

    /// Collective read of one matrix entry.
    pub fn get(&self, row: usize, col: usize) -> Result<u8, DistributedError> {
        self.agree_cell(row, col)?;
        let coord = TileCoord {
            row: row / self.layout.tile_rows(),
            col: col / self.layout.tile_cols(),
        };
        let owner =
            i32::try_from(self.layout.owner(coord)).map_err(|_| DistributedError::SizeOverflow)?;
        let (row_start, _, col_start, _) = self.layout.bounds(coord)?;
        let value = if self.world.mpi_rank() == owner {
            self.local_tile(coord)
                .expect("owner stores its canonical tile")
                .matrix
                .get(row - row_start, col - col_start)
        } else {
            0
        };
        Ok(self
            .world
            .broadcast_bytes(owner, (self.world.mpi_rank() == owner).then_some(&[value]))?[0])
    }

    /// Collective write of one matrix entry. The input is reduced modulo two.
    pub fn set(&mut self, row: usize, col: usize, value: u8) -> Result<(), DistributedError> {
        self.agree_cell(row, col)?;
        self.world.agree_u64(u64::from(value & 1), "value")?;
        let coord = TileCoord {
            row: row / self.layout.tile_rows(),
            col: col / self.layout.tile_cols(),
        };
        if self.layout.owner(coord)
            == usize::try_from(self.world.mpi_rank()).expect("MPI ranks are nonnegative")
        {
            let (row_start, _, col_start, _) = self.layout.bounds(coord)?;
            self.local_tile_mut(coord)
                .expect("owner stores its canonical tile")
                .matrix_mut()
                .set(row - row_start, col - col_start, value);
        }
        self.seed = None;
        Ok(())
    }

    /// Collective in-place flip of one matrix entry.
    pub fn flip(&mut self, row: usize, col: usize) -> Result<(), DistributedError> {
        self.agree_cell(row, col)?;
        let coord = TileCoord {
            row: row / self.layout.tile_rows(),
            col: col / self.layout.tile_cols(),
        };
        if self.layout.owner(coord)
            == usize::try_from(self.world.mpi_rank()).expect("MPI ranks are nonnegative")
        {
            let (row_start, _, col_start, _) = self.layout.bounds(coord)?;
            self.local_tile_mut(coord)
                .expect("owner stores its canonical tile")
                .matrix_mut()
                .flip(row - row_start, col - col_start);
        }
        self.seed = None;
        Ok(())
    }

    /// Collective extraction of at most 64 consecutive row bits, packed
    /// least-significant bit first.
    pub fn extract_bits(
        &self,
        row: usize,
        col_start: usize,
        count: usize,
    ) -> Result<u64, DistributedError> {
        for (value, name) in [(row, "row"), (col_start, "col_start"), (count, "count")] {
            self.world.agree_u64(
                u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
                name,
            )?;
        }
        if row >= self.rows()
            || count > 64
            || col_start > self.cols()
            || count > self.cols() - col_start
        {
            return Err(DistributedError::OutOfBounds(format!(
                "cannot extract {count} bits at ({row}, {col_start}) from {} x {} matrix",
                self.rows(),
                self.cols()
            )));
        }
        let mut result = 0_u64;
        let mut copied = 0;
        while copied < count {
            let global_col = col_start + copied;
            let coord = TileCoord {
                row: row / self.layout.tile_rows(),
                col: global_col / self.layout.tile_cols(),
            };
            let (tile_row_start, _, tile_col_start, tile_col_end) = self.layout.bounds(coord)?;
            let segment = (count - copied).min(tile_col_end - global_col);
            let tile = self.broadcast_tile(coord)?;
            result |= tile.extract_bits(row - tile_row_start, global_col - tile_col_start, segment)
                << copied;
            copied += segment;
        }
        Ok(result)
    }

    /// Exact element-wise addition/XOR. The result retains rank-local packed
    /// storage and performs no communication after collective validation.
    pub fn add(&self, other: &Self) -> Result<Self, DistributedError> {
        self.ensure_compatible(other, "addition")?;
        let mut result = self.clone();
        result.xor_assign(other)?;
        Ok(result)
    }

    /// Exact element-wise subtraction. Over `F_2` this is identical to
    /// [`DistributedMzd::add`].
    pub fn sub(&self, other: &Self) -> Result<Self, DistributedError> {
        self.add(other)
    }

    /// In-place exact element-wise addition/XOR.
    pub fn xor_assign(&mut self, other: &Self) -> Result<(), DistributedError> {
        self.ensure_compatible(other, "addition")?;
        if self.layout == other.layout {
            for (destination, source) in self.local_tiles.iter_mut().zip(&other.local_tiles) {
                debug_assert_eq!(destination.coord, source.coord);
                destination.matrix_mut().xor_into(source.matrix());
            }
        } else {
            let destination_layout = self.layout.clone();
            for source_coord in other.layout.coordinate_iter() {
                let source = other.broadcast_tile(source_coord)?;
                for destination in &mut self.local_tiles {
                    xor_tile_overlap(
                        &destination_layout,
                        destination,
                        &other.layout,
                        source_coord,
                        &source,
                    )?;
                }
            }
        }
        self.seed = None;
        Ok(())
    }

    /// Collective exact equality without gathering either operand.
    ///
    /// Storage tile geometry is not mathematical identity. Differently tiled
    /// operands are compared by streaming each tile of `other` once and
    /// checking its word-aligned intersections with rank-owned tiles of
    /// `self`.
    pub fn equals(&self, other: &Self) -> Result<bool, DistributedError> {
        self.ensure_same_world(other)?;
        if self.shape() != other.shape() {
            return Ok(false);
        }
        let mut local = true;
        if self.layout == other.layout {
            local = self
                .local_tiles
                .iter()
                .zip(&other.local_tiles)
                .all(|(left, right)| {
                    left.coord == right.coord && left.matrix.as_words() == right.matrix.as_words()
                });
        } else {
            for source_coord in other.layout.coordinate_iter() {
                let source = other.broadcast_tile(source_coord)?;
                for destination in &self.local_tiles {
                    local &= tile_overlap_equal(
                        &self.layout,
                        destination,
                        &other.layout,
                        source_coord,
                        &source,
                    )?;
                }
            }
        }
        Ok(self.world.collective_all(local))
    }

    /// Collective transpose. Packed source tiles are streamed one at a time;
    /// no process materializes the complete operand.
    pub fn transpose(&self) -> Result<Self, DistributedError> {
        let mut result = Self::with_tiles(
            &self.world,
            self.cols(),
            self.rows(),
            self.layout.tile_rows(),
            self.layout.tile_cols(),
        )?;
        for coord in self.layout.coordinate_iter() {
            let source = self.broadcast_tile(coord)?;
            let (row_start, _, col_start, _) = self.layout.bounds(coord)?;
            let transposed = source.transpose();
            result.paste_local_block(col_start, row_start, &transposed)?;
        }
        Ok(result)
    }

    /// Collective owned half-open submatrix. Source tiles are streamed and
    /// only intersecting bits are written to rank-owned output tiles.
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        row_end: usize,
        col_end: usize,
    ) -> Result<Self, DistributedError> {
        for (value, name) in [
            (row_start, "row_start"),
            (col_start, "col_start"),
            (row_end, "row_end"),
            (col_end, "col_end"),
        ] {
            self.world.agree_u64(
                u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
                name,
            )?;
        }
        if row_start > row_end
            || row_end > self.rows()
            || col_start > col_end
            || col_end > self.cols()
        {
            return Err(DistributedError::OutOfBounds(format!(
                "invalid half-open range ({row_start}..{row_end}, {col_start}..{col_end}) for {} x {} matrix",
                self.rows(), self.cols()
            )));
        }
        let mut result = Self::with_tiles(
            &self.world,
            row_end - row_start,
            col_end - col_start,
            self.layout.tile_rows(),
            self.layout.tile_cols(),
        )?;
        for coord in self.layout.coordinate_iter() {
            let (source_row_start, source_row_end, source_col_start, source_col_end) =
                self.layout.bounds(coord)?;
            if source_row_end <= row_start
                || source_row_start >= row_end
                || source_col_end <= col_start
                || source_col_start >= col_end
            {
                continue;
            }
            let source = self.broadcast_tile(coord)?;
            let overlap_row_start = source_row_start.max(row_start);
            let overlap_row_end = source_row_end.min(row_end);
            let overlap_col_start = source_col_start.max(col_start);
            let overlap_col_end = source_col_end.min(col_end);
            let block = source
                .try_submatrix(
                    overlap_row_start - source_row_start,
                    overlap_col_start - source_col_start,
                    overlap_row_end - source_row_start,
                    overlap_col_end - source_col_start,
                )
                .map_err(map_mzd_allocation)?;
            result.paste_local_block(
                overlap_row_start - row_start,
                overlap_col_start - col_start,
                &block,
            )?;
        }
        Ok(result)
    }

    /// Collective paste into a cloned destination. Only streamed source tiles
    /// and rank-owned result tiles are materialized.
    pub fn paste(
        &self,
        row_start: usize,
        col_start: usize,
        source: &Self,
    ) -> Result<Self, DistributedError> {
        let mut result = self.clone();
        result.paste_into(row_start, col_start, source)?;
        Ok(result)
    }

    /// Collective caller-owned paste. Existing rank-local allocations are
    /// retained and overwritten only inside the requested rectangle.
    pub fn paste_into(
        &mut self,
        row_start: usize,
        col_start: usize,
        source: &Self,
    ) -> Result<(), DistributedError> {
        self.ensure_same_world(source)?;
        for (value, name) in [(row_start, "row_start"), (col_start, "col_start")] {
            self.world.agree_u64(
                u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
                name,
            )?;
        }
        let row_end = row_start.checked_add(source.rows());
        let col_end = col_start.checked_add(source.cols());
        if row_end.is_none_or(|end| end > self.rows())
            || col_end.is_none_or(|end| end > self.cols())
        {
            return Err(DistributedError::OutOfBounds(format!(
                "{} x {} source at ({row_start}, {col_start}) does not fit {} x {} destination",
                source.rows(),
                source.cols(),
                self.rows(),
                self.cols()
            )));
        }
        self.seed = None;
        for coord in source.layout.coordinate_iter() {
            let tile = source.broadcast_tile(coord)?;
            let (source_row_start, _, source_col_start, _) = source.layout.bounds(coord)?;
            self.paste_local_block(
                row_start + source_row_start,
                col_start + source_col_start,
                &tile,
            )?;
        }
        Ok(())
    }

    /// Collective row XOR, `row[dst] ^= row[src]`, without gathering either
    /// row. Rows in the same tile use the local packed kernel without a
    /// broadcast. Equal row indices are rejected because they would zero the
    /// row.
    pub fn xor_row(&mut self, dst: usize, src: usize) -> Result<(), DistributedError> {
        for (value, name) in [(dst, "dst_row"), (src, "src_row")] {
            self.world.agree_u64(
                u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
                name,
            )?;
        }
        if dst >= self.rows() || src >= self.rows() || dst == src {
            return Err(DistributedError::OutOfBounds(format!(
                "row XOR requires distinct rows below {}",
                self.rows()
            )));
        }
        self.xor_row_unchecked(dst, src)
    }

    pub(crate) fn xor_row_unchecked(
        &mut self,
        dst: usize,
        src: usize,
    ) -> Result<(), DistributedError> {
        let src_tile_row = src / self.layout.tile_rows();
        let dst_tile_row = dst / self.layout.tile_rows();
        for tile_col in 0..self.layout.tile_col_count() {
            let src_coord = TileCoord {
                row: src_tile_row,
                col: tile_col,
            };
            let dst_coord = TileCoord {
                row: dst_tile_row,
                col: tile_col,
            };
            if src_coord == dst_coord {
                if self.owns(src_coord) {
                    let (row_start, _, _, _) = self.layout.bounds(src_coord)?;
                    self.local_tile_mut(src_coord)
                        .expect("owner stores its canonical tile")
                        .matrix_mut()
                        .xor_row_self(dst - row_start, src - row_start);
                }
                continue;
            }
            let source = self.broadcast_tile(src_coord)?;
            let (source_row_start, _, _, _) = self.layout.bounds(src_coord)?;
            if self.owns(dst_coord) {
                let (destination_row_start, _, _, _) = self.layout.bounds(dst_coord)?;
                self.local_tile_mut(dst_coord)
                    .expect("owner stores its canonical tile")
                    .matrix_mut()
                    .xor_row_from(dst - destination_row_start, &source, src - source_row_start);
            }
        }
        self.seed = None;
        Ok(())
    }

    /// Collective row interchange without gathering either row. Rows in the
    /// same tile use the local packed kernel without a broadcast.
    pub fn swap_rows(&mut self, first: usize, second: usize) -> Result<(), DistributedError> {
        for (value, name) in [(first, "first_row"), (second, "second_row")] {
            self.world.agree_u64(
                u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
                name,
            )?;
        }
        if first >= self.rows() || second >= self.rows() {
            return Err(DistributedError::OutOfBounds(format!(
                "row swap requires rows below {}",
                self.rows()
            )));
        }
        if first == second {
            return Ok(());
        }
        self.swap_rows_unchecked(first, second)
    }

    pub(crate) fn swap_rows_unchecked(
        &mut self,
        first: usize,
        second: usize,
    ) -> Result<(), DistributedError> {
        let first_tile_row = first / self.layout.tile_rows();
        let second_tile_row = second / self.layout.tile_rows();
        for tile_col in 0..self.layout.tile_col_count() {
            let first_coord = TileCoord {
                row: first_tile_row,
                col: tile_col,
            };
            let second_coord = TileCoord {
                row: second_tile_row,
                col: tile_col,
            };
            if first_coord == second_coord {
                if self.owns(first_coord) {
                    let (row_start, _, _, _) = self.layout.bounds(first_coord)?;
                    self.local_tile_mut(first_coord)
                        .expect("owner stores its canonical tile")
                        .matrix_mut()
                        .swap_rows(first - row_start, second - row_start);
                }
                continue;
            }
            let first_tile = self.broadcast_tile(first_coord)?;
            let second_tile = self.broadcast_tile(second_coord)?;
            let (first_start, _, _, _) = self.layout.bounds(first_coord)?;
            let (second_start, _, _, _) = self.layout.bounds(second_coord)?;
            let first_words = first_tile.row_words(first - first_start).to_vec();
            let second_words = second_tile.row_words(second - second_start).to_vec();
            if self.owns(first_coord) {
                self.local_tile_mut(first_coord)
                    .expect("owner stores its canonical tile")
                    .matrix_mut()
                    .row_words_mut(first - first_start)
                    .copy_from_slice(&second_words);
            }
            if self.owns(second_coord) {
                self.local_tile_mut(second_coord)
                    .expect("owner stores its canonical tile")
                    .matrix_mut()
                    .row_words_mut(second - second_start)
                    .copy_from_slice(&first_words);
            }
        }
        self.seed = None;
        Ok(())
    }

    /// Collective column interchange without gathering either column.
    /// Rank-local work uses the packed `m4rst` column kernels; source tiles
    /// are broadcast only when the columns occupy different tile columns.
    pub fn swap_cols(&mut self, first: usize, second: usize) -> Result<(), DistributedError> {
        for (value, name) in [(first, "first_col"), (second, "second_col")] {
            self.world.agree_u64(
                u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
                name,
            )?;
        }
        if first >= self.cols() || second >= self.cols() {
            return Err(DistributedError::OutOfBounds(format!(
                "column swap requires columns below {}",
                self.cols()
            )));
        }
        if first == second {
            return Ok(());
        }
        self.swap_cols_unchecked(first, second)
    }

    pub(crate) fn swap_cols_unchecked(
        &mut self,
        first: usize,
        second: usize,
    ) -> Result<(), DistributedError> {
        let first_tile_col = first / self.layout.tile_cols();
        let second_tile_col = second / self.layout.tile_cols();
        for tile_row in 0..self.layout.tile_row_count() {
            let first_coord = TileCoord {
                row: tile_row,
                col: first_tile_col,
            };
            let second_coord = TileCoord {
                row: tile_row,
                col: second_tile_col,
            };
            if first_coord == second_coord {
                if self.owns(first_coord) {
                    let (_, _, col_start, _) = self.layout.bounds(first_coord)?;
                    self.local_tile_mut(first_coord)
                        .expect("owner stores its canonical tile")
                        .matrix_mut()
                        .swap_cols(first - col_start, second - col_start);
                }
                continue;
            }
            let first_tile = self.broadcast_tile(first_coord)?;
            let second_tile = self.broadcast_tile(second_coord)?;
            let (_, _, first_start, _) = self.layout.bounds(first_coord)?;
            let (_, _, second_start, _) = self.layout.bounds(second_coord)?;
            if self.owns(first_coord) {
                self.local_tile_mut(first_coord)
                    .expect("owner stores its canonical tile")
                    .matrix_mut()
                    .copy_col_from(first - first_start, &second_tile, second - second_start);
            }
            if self.owns(second_coord) {
                self.local_tile_mut(second_coord)
                    .expect("owner stores its canonical tile")
                    .matrix_mut()
                    .copy_col_from(second - second_start, &first_tile, first - first_start);
            }
        }
        self.seed = None;
        Ok(())
    }

    /// Collective streamed-panel multiplication. Each shared-dimension step
    /// retains one process-column strip of right panels, streams one left
    /// panel at a time, and updates rank-owned result tiles in place.
    pub fn matmul(left: &Self, right: &Self) -> Result<Self, DistributedError> {
        left.ensure_same_world(right)?;
        if left.cols() != right.rows() {
            return Err(DistributedError::ShapeMismatch(format!(
                "cannot multiply {} x {} by {} x {}",
                left.rows(),
                left.cols(),
                right.rows(),
                right.cols()
            )));
        }
        let output_started = Instant::now();
        let mut result = Self::with_tiles(
            &left.world,
            left.rows(),
            right.cols(),
            left.layout.tile_rows(),
            right.layout.tile_cols(),
        )?;
        result
            .world
            .record_matmul_phase(MatmulPhase::OutputPreparation, output_started.elapsed());
        Self::matmul_into(&mut result, left, right)?;
        Ok(result)
    }

    /// Caller-owned-output counterpart to [`DistributedMzd::matmul`]. The
    /// output is fully overwritten and retains its rank-local allocations.
    pub fn matmul_into(
        output: &mut Self,
        left: &Self,
        right: &Self,
    ) -> Result<(), DistributedError> {
        left.ensure_same_world(right)?;
        left.ensure_same_world(output)?;
        if left.cols() != right.rows()
            || output.shape() != (left.rows(), right.cols())
            || output.layout.tile_rows() != left.layout.tile_rows()
            || output.layout.tile_cols() != right.layout.tile_cols()
        {
            return Err(DistributedError::ShapeMismatch(
                "matmul_into operands or output tile geometry are incompatible".to_owned(),
            ));
        }
        let output_started = Instant::now();
        for tile in &mut output.local_tiles {
            tile.matrix_mut().fill_zero();
        }
        output
            .world
            .record_matmul_phase(MatmulPhase::OutputPreparation, output_started.elapsed());
        if right.cols() == 0 {
            output.seed = None;
            return Ok(());
        }
        let communicator_started = Instant::now();
        let process_grid = output.layout.process_grid();
        let mpi_rank = usize::try_from(output.world.mpi_rank()).expect("MPI ranks are nonnegative");
        let process_row = mpi_rank / process_grid.cols();
        let process_col = mpi_rank % process_grid.cols();
        let serialized_world = output.world.clone();
        let communicator = serialized_world.communicator();
        let row_communicator = communicator
            .split_by_color_with_key(
                Color::with_value(
                    i32::try_from(process_row).map_err(|_| DistributedError::SizeOverflow)?,
                ),
                i32::try_from(process_col).map_err(|_| DistributedError::SizeOverflow)?,
            )
            .expect("every process joins one process row");
        output.world.record_communicator_split();
        let column_communicator = communicator
            .split_by_color_with_key(
                Color::with_value(
                    i32::try_from(process_col).map_err(|_| DistributedError::SizeOverflow)?,
                ),
                i32::try_from(process_row).map_err(|_| DistributedError::SizeOverflow)?,
            )
            .expect("every process joins one process column");
        output.world.record_communicator_split();
        let mut shared_boundaries = vec![0, left.cols()];
        shared_boundaries
            .extend((left.layout.tile_cols()..left.cols()).step_by(left.layout.tile_cols()));
        shared_boundaries
            .extend((right.layout.tile_rows()..right.rows()).step_by(right.layout.tile_rows()));
        shared_boundaries.sort_unstable();
        shared_boundaries.dedup();
        output.world.record_matmul_phase(
            MatmulPhase::CommunicatorSetup,
            communicator_started.elapsed(),
        );
        let mut product_scratch = None;
        let mut left_panel_scratch = None;

        for shared in shared_boundaries.windows(2) {
            let shared_start = shared[0];
            let shared_end = shared[1];
            let left_tile_col = shared_start / left.layout.tile_cols();
            let right_tile_row = shared_start / right.layout.tile_rows();
            let mut right_panels = Vec::new();
            try_reserve_items(&mut right_panels, right.layout.tile_col_count())?;
            for right_tile_col in 0..right.layout.tile_col_count() {
                if right_tile_col % process_grid.cols() != process_col {
                    continue;
                }
                let coord = TileCoord {
                    row: right_tile_row,
                    col: right_tile_col,
                };
                let (row_start, _, col_start, col_end) = right.layout.bounds(coord)?;
                let (panel, preparation_elapsed, broadcast_elapsed) = right
                    .broadcast_submatrix_on(
                        coord,
                        shared_start - row_start,
                        0,
                        shared_end - row_start,
                        col_end - col_start,
                        &column_communicator,
                        i32::try_from(right_tile_row % process_grid.rows())
                            .map_err(|_| DistributedError::SizeOverflow)?,
                    )?;
                output
                    .world
                    .record_matmul_phase(MatmulPhase::RightPanelPreparation, preparation_elapsed);
                output
                    .world
                    .record_matmul_phase(MatmulPhase::RightPanelBroadcast, broadcast_elapsed);
                right_panels.push((right_tile_col, panel));
            }

            // Every rank in a process row must enter the same left-panel
            // broadcasts. Derive batching from global geometry, not from the
            // right panels owned by this process column (which may be empty).
            let max_product_cols = right.layout.tile_cols().min(right.cols());
            let left_panel_batches = matmul_left_panel_batches(
                &left.layout,
                process_row,
                shared_end - shared_start,
                max_product_cols,
            )?;
            let left_root = i32::try_from(left_tile_col % process_grid.cols())
                .map_err(|_| DistributedError::SizeOverflow)?;
            for left_tile_rows in left_panel_batches {
                let (mut left_panel, preparation_elapsed) = prepare_matmul_left_panel(
                    left,
                    left_tile_col,
                    shared_start,
                    shared_end,
                    &left_tile_rows,
                    row_communicator.rank() == left_root,
                    left_panel_scratch.take(),
                )?;
                output
                    .world
                    .record_matmul_phase(MatmulPhase::LeftPanelPreparation, preparation_elapsed);
                let broadcast_started = Instant::now();
                row_communicator
                    .process_at_rank(left_root)
                    .broadcast_into(left_panel.as_words_mut());
                output
                    .world
                    .record_broadcast(std::mem::size_of_val(left_panel.as_words()));
                output.world.record_matmul_phase(
                    MatmulPhase::LeftPanelBroadcast,
                    broadcast_started.elapsed(),
                );
                accumulate_matmul_left_panel(
                    output,
                    left,
                    &left_tile_rows,
                    &left_panel,
                    &right_panels,
                    &mut product_scratch,
                )?;
                left_panel_scratch = Some(left_panel);
            }
        }
        output.seed = None;
        Ok(())
    }

    /// Probabilistically verify `candidate = left * right` with Freivalds'
    /// algorithm over `F_2`.
    ///
    /// An explicit deterministic seed constructs the random binary vectors.
    /// Up to 64 rounds share one packed `Mzd` word lane so each operand is
    /// scanned once per batch. The matrix-vector projections follow an
    /// independent all-reduce route: every rank multiplies only its ordinary
    /// local `Mzd` tiles, and packed projection batches are XOR-reduced across
    /// the world. This does not call distributed [`DistributedMzd::matmul`],
    /// gather an operand, or retain a matrix-scale replica.
    ///
    /// A correct product always passes. Challenges are generated
    /// deterministically from the recorded 64-bit seed so runs are exactly
    /// reproducible; this API therefore does not claim the unconditional
    /// `2^-rounds` bound that requires independently uniform challenge bits.
    /// `max_workspace_bytes` is a per-process ceiling for the verifier's
    /// explicit packed projection buffers; local `m4rst` kernel scratch
    /// remains governed by the selected rank-local implementation.
    pub fn verify_matmul_freivalds(
        left: &Self,
        right: &Self,
        candidate: &Self,
        rounds: usize,
        seed: u64,
        max_workspace_bytes: usize,
    ) -> Result<bool, DistributedError> {
        left.ensure_same_world(right)?;
        left.ensure_same_world(candidate)?;
        for (value, name) in [
            (rounds, "freivalds_rounds"),
            (max_workspace_bytes, "freivalds_max_workspace_bytes"),
        ] {
            left.world.agree_u64(
                u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
                name,
            )?;
        }
        left.world.agree_u64(seed, "freivalds_seed")?;
        if rounds == 0 {
            return Err(DistributedError::OutOfBounds(
                "Freivalds verification requires at least one round".to_owned(),
            ));
        }
        if left.cols() != right.rows()
            || candidate.rows() != left.rows()
            || candidate.cols() != right.cols()
        {
            return Err(DistributedError::ShapeMismatch(format!(
                "cannot verify {} x {} times {} x {} against {} x {}",
                left.rows(),
                left.cols(),
                right.rows(),
                right.cols(),
                candidate.rows(),
                candidate.cols()
            )));
        }
        let required = freivalds_workspace_bytes(left, right, candidate, rounds)?;
        if required > max_workspace_bytes {
            return Err(DistributedError::WorkspaceLimit {
                required,
                limit: max_workspace_bytes,
            });
        }

        let mut passed = true;
        for (batch, round_start) in (0..rounds).step_by(m4rst::mzd::RADIX).enumerate() {
            let batch_rounds = (rounds - round_start).min(m4rst::mzd::RADIX);
            let batch_seed =
                seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(
                    u64::try_from(batch).map_err(|_| DistributedError::SizeOverflow)?,
                ));
            let random = Mzd::try_random(batch_rounds, right.cols(), batch_seed)
                .map_err(map_mzd_allocation)?;
            let right_projection = right.project_replicated_rows(&random)?;
            let expected = left.project_replicated_rows(&right_projection)?;
            let observed = candidate.project_replicated_rows(&random)?;
            passed &= expected.as_words() == observed.as_words();
        }
        Ok(left.world.collective_all(passed))
    }

    /// Process-grid-independent logical hash derived from canonical tile
    /// coordinates, shapes, and packed checksums. This is collective and
    /// returns the same value on every rank.
    pub fn logical_hash(&self) -> Result<u64, DistributedError> {
        let checksums = self.collective_checksums()?;
        let tile_count = self
            .layout
            .tile_row_count()
            .checked_mul(self.layout.tile_col_count())
            .ok_or(DistributedError::SizeOverflow)?;
        let byte_count = tile_count
            .checked_mul(3 * size_of::<u64>())
            .and_then(|bytes| bytes.checked_add(4 * size_of::<u64>()))
            .ok_or(DistributedError::SizeOverflow)?;
        let mut bytes = Vec::new();
        try_reserve_items(&mut bytes, byte_count)?;
        for value in [
            self.rows(),
            self.cols(),
            self.layout.tile_rows(),
            self.layout.tile_cols(),
        ] {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        for (coord, value) in self.layout.coordinate_iter().zip(checksums) {
            bytes.extend_from_slice(&(coord.row as u64).to_le_bytes());
            bytes.extend_from_slice(&(coord.col as u64).to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(checksum(&bytes))
    }

    /// Collect a small matrix on `root` after checking a packed-byte ceiling
    /// on every process. The root receives `Some(Mzd)`; other ranks receive
    /// `None`. No communication begins when the ceiling is too small.
    pub fn collect(&self, root: i32, max_bytes: usize) -> Result<Option<Mzd>, DistributedError> {
        self.world.agree_rank(root, "root")?;
        self.world.agree_u64(
            u64::try_from(max_bytes).map_err(|_| DistributedError::SizeOverflow)?,
            "max_bytes",
        )?;
        let required = self
            .rows()
            .checked_mul(self.cols().div_ceil(m4rst::mzd::RADIX))
            .and_then(|words| words.checked_mul(size_of::<u64>()))
            .ok_or(DistributedError::SizeOverflow)?;
        if required > max_bytes {
            return Err(DistributedError::MaterializationLimit {
                required,
                limit: max_bytes,
            });
        }

        let output = if self.world.mpi_rank() == root {
            try_mzd(self.rows(), self.cols()).map(Some)
        } else {
            Ok(None)
        };
        if !self.world.collective_all(output.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "the collection root could not allocate the packed output".to_owned(),
            ));
        }
        let mut output = output?;
        let transfer_words = self
            .layout
            .tile_rows()
            .checked_mul(self.layout.tile_cols().div_ceil(m4rst::mzd::RADIX))
            .ok_or(DistributedError::SizeOverflow)?;
        let transfer = try_zeroed_words(transfer_words);
        if !self.world.collective_all(transfer.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not allocate a collection transfer buffer".to_owned(),
            ));
        }
        let mut transfer = transfer?;
        let communicator = self.world.communicator();
        for coord in self.layout.coordinate_iter() {
            let owner = i32::try_from(self.layout.owner(coord))
                .map_err(|_| DistributedError::SizeOverflow)?;
            if self.world.mpi_rank() == owner {
                let tile = self
                    .local_tile(coord)
                    .expect("owner stores its canonical tile");
                if owner == root {
                    paste_tile(&self.layout, coord, tile.matrix(), output.as_mut().unwrap());
                } else {
                    communicator
                        .process_at_rank(root)
                        .send(tile.matrix().as_words());
                    self.world
                        .record_send(size_of_val(tile.matrix().as_words()));
                }
            } else if self.world.mpi_rank() == root {
                let (row_start, row_end, col_start, col_end) = self.layout.bounds(coord)?;
                let words = (row_end - row_start)
                    .checked_mul((col_end - col_start).div_ceil(m4rst::mzd::RADIX))
                    .ok_or(DistributedError::SizeOverflow)?;
                communicator
                    .process_at_rank(owner)
                    .receive_into(&mut transfer[..words]);
                self.world.record_receive(size_of_val(&transfer[..words]));
                paste_tile_words(
                    &self.layout,
                    coord,
                    &transfer[..words],
                    output.as_mut().expect("root allocated the output"),
                )?;
            }
        }
        Ok(output)
    }

    /// Scatter one root-owned packed matrix without expanding it to one byte
    /// per coefficient. Non-root ranks pass `None`; the root passes `Some`.
    /// Shape is broadcast from the root before rank-local tiles are sent.
    pub fn distribute_packed(
        world: &World,
        root_matrix: Option<&Mzd>,
        root: i32,
        max_bytes: usize,
    ) -> Result<Self, DistributedError> {
        world.agree_rank(root, "root")?;
        let role_is_valid = if world.mpi_rank() == root {
            root_matrix.is_some()
        } else {
            root_matrix.is_none()
        };
        if !world.collective_all(role_is_valid) {
            return Err(DistributedError::RootBuffer(
                "the root must provide one packed matrix and every non-root must pass None"
                    .to_owned(),
            ));
        }
        let shape_bytes = root_matrix.map(|matrix| {
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&(matrix.rows() as u64).to_le_bytes());
            bytes.extend_from_slice(&(matrix.cols() as u64).to_le_bytes());
            bytes
        });
        let shape_bytes = world.broadcast_bytes(root, shape_bytes.as_deref())?;
        let shape: [u8; 16] = shape_bytes.try_into().map_err(|_| {
            DistributedError::CollectiveFailure(
                "root broadcast an invalid packed-matrix shape".to_owned(),
            )
        })?;
        let rows = usize::try_from(u64::from_le_bytes(
            shape[..8].try_into().expect("eight bytes"),
        ))
        .map_err(|_| DistributedError::SizeOverflow)?;
        let cols = usize::try_from(u64::from_le_bytes(
            shape[8..].try_into().expect("eight bytes"),
        ))
        .map_err(|_| DistributedError::SizeOverflow)?;
        let (tile_rows, tile_cols) = default_tile_shape(
            rows,
            cols,
            usize::try_from(world.size()).map_err(|_| DistributedError::SizeOverflow)?,
        )?;
        Self::from_root_packed(
            world,
            rows,
            cols,
            root_matrix,
            root,
            max_bytes,
            tile_rows,
            tile_cols,
        )
    }

    /// Collectively write canonical rank-owned tile files and a process-grid-
    /// independent manifest into `directory`.
    pub fn write(&self, directory: impl AsRef<Path>) -> Result<MatrixManifest, DistributedError> {
        let directory = directory.as_ref();
        self.agree_path(directory)?;
        let create_result = if self.world.is_root() {
            fs::create_dir_all(directory).map_err(|error| error.to_string())
        } else {
            Ok(())
        };
        if !self.world.collective_all(create_result.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "manifest directory creation failed".to_owned(),
            ));
        }
        self.world.communicator().barrier();
        self.world.record_barrier();

        let local_write = self.local_tiles.iter().try_for_each(|tile| {
            write_tile_file(directory, tile).map_err(|error| error.to_string())
        });
        if !self.world.collective_all(local_write.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not write its packed tiles".to_owned(),
            ));
        }
        self.world.communicator().barrier();
        self.world.record_barrier();

        let manifest = self.manifest()?;
        let manifest_write = if self.world.is_root() {
            write_manifest_file(directory, &manifest).map_err(|error| error.to_string())
        } else {
            Ok(())
        };
        if !self.world.collective_all(manifest_write.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "rank zero could not commit the manifest".to_owned(),
            ));
        }
        self.world.communicator().barrier();
        self.world.record_barrier();
        Ok(manifest)
    }

    /// Collectively reopen a manifest. Each process reads only the tile files
    /// assigned to it by the current process grid.
    pub fn open(world: &World, directory: impl AsRef<Path>) -> Result<Self, DistributedError> {
        let directory = directory.as_ref();
        agree_path(world, directory)?;
        let root_payload = if world.is_root() {
            match fs::read(directory.join(MANIFEST_NAME)) {
                Ok(bytes) => {
                    let mut payload = vec![1];
                    payload.extend(bytes);
                    payload
                }
                Err(error) => {
                    let mut payload = vec![0];
                    payload.extend(error.to_string().as_bytes());
                    payload
                }
            }
        } else {
            Vec::new()
        };
        let payload =
            world.broadcast_bytes(0, world.is_root().then_some(root_payload.as_slice()))?;
        let Some((&status, body)) = payload.split_first() else {
            return Err(DistributedError::Manifest(
                "rank zero broadcast an empty manifest response".to_owned(),
            ));
        };
        if status == 0 {
            return Err(DistributedError::Io(
                String::from_utf8_lossy(body).into_owned(),
            ));
        }
        let manifest: MatrixManifest = serde_json::from_slice(body)
            .map_err(|error| DistributedError::Manifest(error.to_string()))?;
        let layout = manifest
            .validate(usize::try_from(world.size()).map_err(|_| DistributedError::SizeOverflow)?)?;
        let mpi_rank = usize::try_from(world.mpi_rank()).expect("MPI ranks are nonnegative");
        let mut local_tiles = Vec::new();
        let local_read = (|| -> Result<(), DistributedError> {
            for record in manifest
                .tiles
                .iter()
                .filter(|record| layout.owner(record.coord) == mpi_rank)
            {
                let bytes = fs::read(directory.join(&record.file))
                    .map_err(|error| DistributedError::Io(error.to_string()))?;
                if checksum(&bytes) != record.checksum {
                    return Err(DistributedError::Manifest(format!(
                        "tile ({}, {}) checksum mismatch",
                        record.coord.row, record.coord.col
                    )));
                }
                let words = bytes_to_words(&bytes)?;
                let matrix = matrix_from_words(&layout, record.coord, &words)?;
                local_tiles.push(LocalTile {
                    coord: record.coord,
                    matrix: Arc::new(matrix),
                });
            }
            Ok(())
        })();
        if !world.collective_all(local_read.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank rejected an owned tile file".to_owned(),
            ));
        }
        Ok(Self {
            world: world.clone(),
            layout,
            local_tiles,
            seed: manifest.seed,
        })
    }

    pub(crate) fn local_tile(&self, coord: TileCoord) -> Option<&LocalTile> {
        self.local_tiles.iter().find(|tile| tile.coord == coord)
    }

    pub(crate) fn local_tile_mut(&mut self, coord: TileCoord) -> Option<&mut LocalTile> {
        self.local_tiles.iter_mut().find(|tile| tile.coord == coord)
    }

    pub(crate) fn owns(&self, coord: TileCoord) -> bool {
        self.layout.owner(coord)
            == usize::try_from(self.world.mpi_rank()).expect("MPI ranks are nonnegative")
    }

    pub(crate) fn ensure_same_world(&self, other: &Self) -> Result<(), DistributedError> {
        if !self
            .world
            .collective_all(self.world.same_world(&other.world))
        {
            return Err(DistributedError::UnequalWorlds);
        }
        for (value, name) in [
            (self.rows(), "left matrix rows"),
            (self.cols(), "left matrix columns"),
            (self.layout.tile_rows(), "left matrix tile rows"),
            (self.layout.tile_cols(), "left matrix tile columns"),
            (other.rows(), "right matrix rows"),
            (other.cols(), "right matrix columns"),
            (other.layout.tile_rows(), "right matrix tile rows"),
            (other.layout.tile_cols(), "right matrix tile columns"),
        ] {
            self.world.agree_u64(
                u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
                name,
            )?;
        }
        Ok(())
    }

    fn ensure_compatible(&self, other: &Self, operation: &str) -> Result<(), DistributedError> {
        self.ensure_same_world(other)?;
        if self.shape() != other.shape() {
            return Err(DistributedError::ShapeMismatch(format!(
                "{operation} requires identical matrix shapes"
            )));
        }
        Ok(())
    }

    fn agree_cell(&self, row: usize, col: usize) -> Result<(), DistributedError> {
        self.world.agree_u64(
            u64::try_from(row).map_err(|_| DistributedError::SizeOverflow)?,
            "row",
        )?;
        self.world.agree_u64(
            u64::try_from(col).map_err(|_| DistributedError::SizeOverflow)?,
            "col",
        )?;
        if row >= self.rows() || col >= self.cols() {
            return Err(DistributedError::OutOfBounds(format!(
                "cell ({row}, {col}) is outside {} x {} matrix",
                self.rows(),
                self.cols()
            )));
        }
        Ok(())
    }

    pub(crate) fn set_local_if_owned(
        &mut self,
        row: usize,
        col: usize,
        value: u8,
    ) -> Result<(), DistributedError> {
        let coord = TileCoord {
            row: row / self.layout.tile_rows(),
            col: col / self.layout.tile_cols(),
        };
        if self.owns(coord) {
            let (row_start, _, col_start, _) = self.layout.bounds(coord)?;
            self.local_tile_mut(coord)
                .expect("owner stores its canonical tile")
                .matrix_mut()
                .set(row - row_start, col - col_start, value);
        }
        Ok(())
    }

    fn paste_local_block(
        &mut self,
        row_start: usize,
        col_start: usize,
        source: &Mzd,
    ) -> Result<(), DistributedError> {
        if source.rows() == 0 || source.cols() == 0 {
            return Ok(());
        }
        let row_end = row_start
            .checked_add(source.rows())
            .ok_or(DistributedError::SizeOverflow)?;
        let col_end = col_start
            .checked_add(source.cols())
            .ok_or(DistributedError::SizeOverflow)?;
        if row_end > self.rows() || col_end > self.cols() {
            return Err(DistributedError::OutOfBounds(format!(
                "{} x {} local block at ({row_start}, {col_start}) does not fit {} x {} destination",
                source.rows(),
                source.cols(),
                self.rows(),
                self.cols()
            )));
        }
        let first_tile_row = row_start / self.layout.tile_rows();
        let last_tile_row = (row_end - 1) / self.layout.tile_rows();
        let first_tile_col = col_start / self.layout.tile_cols();
        let last_tile_col = (col_end - 1) / self.layout.tile_cols();
        for tile_row in first_tile_row..=last_tile_row {
            for tile_col in first_tile_col..=last_tile_col {
                let coord = TileCoord {
                    row: tile_row,
                    col: tile_col,
                };
                if !self.owns(coord) {
                    continue;
                }
                let (tile_row_start, tile_row_end, tile_col_start, tile_col_end) =
                    self.layout.bounds(coord)?;
                let overlap_row_start = row_start.max(tile_row_start);
                let overlap_row_end = row_end.min(tile_row_end);
                let overlap_col_start = col_start.max(tile_col_start);
                let overlap_col_end = col_end.min(tile_col_end);
                let block = source
                    .try_submatrix(
                        overlap_row_start - row_start,
                        overlap_col_start - col_start,
                        overlap_row_end - row_start,
                        overlap_col_end - col_start,
                    )
                    .map_err(map_mzd_allocation)?;
                self.local_tile_mut(coord)
                    .expect("owner stores its canonical tile")
                    .matrix_mut()
                    .paste_into(
                        overlap_row_start - tile_row_start,
                        overlap_col_start - tile_col_start,
                        &block,
                    );
            }
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        for tile in &mut self.local_tiles {
            tile.matrix_mut().fill_zero();
        }
        self.seed = None;
    }

    fn broadcast_tile(&self, coord: TileCoord) -> Result<Mzd, DistributedError> {
        let owner =
            i32::try_from(self.layout.owner(coord)).map_err(|_| DistributedError::SizeOverflow)?;
        let communicator = self.world.communicator();
        self.broadcast_tile_on(coord, &*communicator, owner)
    }

    fn broadcast_tile_on(
        &self,
        coord: TileCoord,
        communicator: &impl Communicator,
        root: i32,
    ) -> Result<Mzd, DistributedError> {
        let (row_start, row_end, col_start, col_end) = self.layout.bounds(coord)?;
        let word_count = (row_end - row_start)
            .checked_mul((col_end - col_start).div_ceil(m4rst::mzd::RADIX))
            .ok_or(DistributedError::SizeOverflow)?;
        let mut words = if communicator.rank() == root {
            let source = self
                .local_tile(coord)
                .expect("owner stores its canonical tile")
                .matrix
                .as_words();
            let mut words = Vec::new();
            try_reserve_items(&mut words, source.len())?;
            words.extend_from_slice(source);
            words
        } else {
            try_zeroed_words(word_count)?
        };
        if !words.is_empty() {
            communicator
                .process_at_rank(root)
                .broadcast_into(&mut words[..]);
            self.world.record_broadcast(size_of_val(&words[..]));
        }
        matrix_from_words(&self.layout, coord, &words)
    }

    #[allow(clippy::too_many_arguments)]
    fn broadcast_submatrix_on(
        &self,
        coord: TileCoord,
        row_start: usize,
        col_start: usize,
        row_end: usize,
        col_end: usize,
        communicator: &impl Communicator,
        root: i32,
    ) -> Result<(Mzd, Duration, Duration), DistributedError> {
        let preparation_started = Instant::now();
        let mut panel = if communicator.rank() == root {
            self.local_tile(coord)
                .expect("subcommunicator root owns its canonical tile")
                .matrix
                .try_submatrix(row_start, col_start, row_end, col_end)
                .map_err(map_mzd_allocation)?
        } else {
            try_mzd(row_end - row_start, col_end - col_start)?
        };
        let preparation_elapsed = preparation_started.elapsed();
        let broadcast_started = Instant::now();
        if !panel.as_words().is_empty() {
            communicator
                .process_at_rank(root)
                .broadcast_into(panel.as_words_mut());
            self.world.record_broadcast(size_of_val(panel.as_words()));
        }
        Ok((panel, preparation_elapsed, broadcast_started.elapsed()))
    }

    fn collective_checksums(&self) -> Result<Vec<u64>, DistributedError> {
        let tile_count = self
            .layout
            .tile_row_count()
            .checked_mul(self.layout.tile_col_count())
            .ok_or(DistributedError::SizeOverflow)?;
        let local = try_zeroed_words(tile_count);
        if !self.world.collective_all(local.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not allocate tile checksums".to_owned(),
            ));
        }
        let mut local = local?;
        for tile in &self.local_tiles {
            let index = tile.coord.row * self.layout.tile_col_count() + tile.coord.col;
            local[index] = checksum_words(tile.matrix.as_words());
        }
        self.world.xor_u64s(&local)
    }

    fn project_replicated_rows(&self, vectors: &Mzd) -> Result<Mzd, DistributedError> {
        debug_assert!((1..=m4rst::mzd::RADIX).contains(&vectors.rows()));
        debug_assert_eq!(vectors.cols(), self.cols());
        let mut partial_rows: Vec<Option<Mzd>> = Vec::new();
        try_reserve_items(&mut partial_rows, self.layout.tile_row_count())?;
        partial_rows.resize_with(self.layout.tile_row_count(), || None);
        for tile in &self.local_tiles {
            let (_, _, col_start, col_end) = self.layout.bounds(tile.coord)?;
            let vector_segment = vectors
                .try_submatrix(0, col_start, vectors.rows(), col_end)
                .map_err(map_mzd_allocation)?
                .transpose();
            let projection = Mzd::matmul(&tile.matrix, &vector_segment).transpose();
            if let Some(partial) = &mut partial_rows[tile.coord.row] {
                partial.xor_into(&projection);
            } else {
                partial_rows[tile.coord.row] = Some(projection);
            }
        }

        let mut local = try_mzd(vectors.rows(), self.rows())?;
        for (tile_row, partial) in partial_rows.into_iter().enumerate() {
            if let Some(partial) = partial {
                local.paste_into(0, tile_row * self.layout.tile_rows(), &partial);
            }
        }
        if local.as_words().is_empty() {
            return Ok(local);
        }
        let reduced = self.world.xor_u64s(local.as_words())?;
        local.as_words_mut().copy_from_slice(&reduced);
        Ok(local)
    }

    fn manifest(&self) -> Result<MatrixManifest, DistributedError> {
        let checksums = self.collective_checksums()?;
        let mut tiles = Vec::new();
        try_reserve_items(&mut tiles, checksums.len())?;
        for (coord, checksum) in self.layout.coordinate_iter().zip(checksums) {
            let (row_start, row_end, col_start, col_end) = self.layout.bounds(coord)?;
            let rows = row_end - row_start;
            let cols = col_end - col_start;
            tiles.push(TileRecord {
                coord,
                rows,
                cols,
                words: rows
                    .checked_mul(cols.div_ceil(m4rst::mzd::RADIX))
                    .ok_or(DistributedError::SizeOverflow)?,
                checksum,
                file: tile_filename(coord),
            });
        }
        Ok(MatrixManifest {
            version: FORMAT_VERSION,
            field: "F2".to_owned(),
            rows: self.rows(),
            cols: self.cols(),
            word_bits: 64,
            bit_order: "lsb-first".to_owned(),
            tile_rows: self.layout.tile_rows(),
            tile_cols: self.layout.tile_cols(),
            seed: self.seed,
            tiles,
        })
    }

    fn agree_path(&self, path: &Path) -> Result<(), DistributedError> {
        agree_path(&self.world, path)
    }
}

fn collective_layout(
    world: &World,
    rows: usize,
    cols: usize,
    tile_rows: usize,
    tile_cols: usize,
) -> Result<Layout, DistributedError> {
    for (value, name) in [
        (rows, "rows"),
        (cols, "cols"),
        (tile_rows, "tile_rows"),
        (tile_cols, "tile_cols"),
    ] {
        world.agree_u64(
            u64::try_from(value).map_err(|_| DistributedError::SizeOverflow)?,
            name,
        )?;
    }
    let layout = Layout::new(
        rows,
        cols,
        tile_rows,
        tile_cols,
        usize::try_from(world.size()).map_err(|_| DistributedError::SizeOverflow)?,
    )?;
    let tile_preflight = try_mzd(rows.min(tile_rows), cols.min(tile_cols));
    if !world.collective_all(tile_preflight.is_ok()) {
        return Err(DistributedError::CollectiveFailure(
            "at least one rank could not reserve one canonical matrix tile".to_owned(),
        ));
    }
    drop(tile_preflight?);
    Ok(layout)
}

fn prepare_matmul_left_panel(
    left: &DistributedMzd,
    left_tile_col: usize,
    shared_start: usize,
    shared_end: usize,
    left_tile_rows: &[usize],
    is_root: bool,
    scratch: Option<Mzd>,
) -> Result<(Mzd, Duration), DistributedError> {
    let preparation_started = Instant::now();
    let total_rows = left_tile_rows.iter().try_fold(0_usize, |total, tile_row| {
        let (row_start, row_end, _, _) = left.layout.bounds(TileCoord {
            row: *tile_row,
            col: left_tile_col,
        })?;
        total
            .checked_add(row_end - row_start)
            .ok_or(DistributedError::SizeOverflow)
    })?;
    let panel_cols = shared_end - shared_start;
    // Reuse is safe without clearing: roots overwrite every logical cell
    // below while canonical source tails preserve zero padding, and nonroots
    // receive every packed word from the broadcast.
    let mut left_panel = match scratch {
        Some(panel) if panel.rows() == total_rows && panel.cols() == panel_cols => panel,
        _ => try_mzd(total_rows, panel_cols)?,
    };
    if is_root {
        let mut batch_row_start = 0;
        for left_tile_row in left_tile_rows {
            let coord = TileCoord {
                row: *left_tile_row,
                col: left_tile_col,
            };
            let (row_start, row_end, col_start, col_end) = left.layout.bounds(coord)?;
            let rows = row_end - row_start;
            let local_col_start = shared_start - col_start;
            let local_col_end = shared_end - col_start;
            let source = &left
                .local_tile(coord)
                .expect("subcommunicator root owns its canonical tile")
                .matrix;
            if local_col_start == 0 && local_col_end == col_end - col_start {
                left_panel
                    .view_mut(
                        batch_row_start,
                        0,
                        batch_row_start + rows,
                        shared_end - shared_start,
                    )
                    .assign(source.as_ref());
            } else {
                let panel = source
                    .try_submatrix(0, local_col_start, rows, local_col_end)
                    .map_err(map_mzd_allocation)?;
                left_panel.paste_into(batch_row_start, 0, &panel);
            }
            batch_row_start += rows;
        }
    }
    Ok((left_panel, preparation_started.elapsed()))
}

fn accumulate_matmul_left_panel(
    output: &mut DistributedMzd,
    left: &DistributedMzd,
    left_tile_rows: &[usize],
    left_panel: &Mzd,
    right_panels: &[(usize, Mzd)],
    product_scratch: &mut Option<Mzd>,
) -> Result<(), DistributedError> {
    for (right_tile_col, right_panel) in right_panels {
        let needs_product = product_scratch.as_ref().is_none_or(|product| {
            product.rows() != left_panel.rows() || product.cols() != right_panel.cols()
        });
        if needs_product {
            *product_scratch = Some(try_mzd(left_panel.rows(), right_panel.cols())?);
        }
        let product = product_scratch
            .as_mut()
            .expect("product scratch was initialized");
        let kernel_started = Instant::now();
        Mzd::matmul_into(product, left_panel, right_panel);
        output
            .world
            .record_matmul_phase(MatmulPhase::LocalKernel, kernel_started.elapsed());
        let accumulation_started = Instant::now();
        let mut batch_row_start = 0;
        for left_tile_row in left_tile_rows {
            let (row_start, row_end, _, _) = left.layout.bounds(TileCoord {
                row: *left_tile_row,
                col: 0,
            })?;
            let rows = row_end - row_start;
            let product_rows =
                product.view(batch_row_start, 0, batch_row_start + rows, product.cols());
            output
                .local_tile_mut(TileCoord {
                    row: *left_tile_row,
                    col: *right_tile_col,
                })
                .expect("process-grid owner stores its canonical output tile")
                .matrix_mut()
                .xor_into(&product_rows);
            batch_row_start += rows;
        }
        output.world.record_matmul_phase(
            MatmulPhase::OutputAccumulation,
            accumulation_started.elapsed(),
        );
    }
    Ok(())
}

fn matmul_left_panel_batches(
    layout: &Layout,
    process_row: usize,
    shared_cols: usize,
    product_cols: usize,
) -> Result<Vec<Vec<usize>>, DistributedError> {
    let process_rows = layout.process_grid().rows();
    let bytes_per_row = packed_row_bytes(shared_cols)?.max(packed_row_bytes(product_cols)?);
    let row_budget = (MATMUL_LEFT_PANEL_BATCH_BYTES / bytes_per_row).max(1);
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_rows = 0_usize;
    for tile_row in (process_row..layout.tile_row_count()).step_by(process_rows) {
        let (row_start, row_end, _, _) = layout.bounds(TileCoord {
            row: tile_row,
            col: 0,
        })?;
        let rows = row_end - row_start;
        if !current.is_empty() && current_rows.saturating_add(rows) > row_budget {
            batches.push(std::mem::take(&mut current));
            current_rows = 0;
        }
        current.push(tile_row);
        current_rows = current_rows
            .checked_add(rows)
            .ok_or(DistributedError::SizeOverflow)?;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    Ok(batches)
}

fn freivalds_workspace_bytes(
    left: &DistributedMzd,
    right: &DistributedMzd,
    candidate: &DistributedMzd,
    rounds: usize,
) -> Result<usize, DistributedError> {
    let batch_rows = rounds.min(m4rst::mzd::RADIX);
    let random = packed_matrix_bytes(batch_rows, right.cols())?;
    let inner = packed_matrix_bytes(batch_rows, right.rows())?;
    let output = packed_matrix_bytes(batch_rows, left.rows())?;

    let right_peak = checked_bytes(&[random, projection_workspace_bytes(right, batch_rows)?])?;
    let left_peak = checked_bytes(&[random, inner, projection_workspace_bytes(left, batch_rows)?])?;
    let candidate_peak = checked_bytes(&[
        random,
        inner,
        output,
        projection_workspace_bytes(candidate, batch_rows)?,
    ])?;
    let retained_peak = checked_bytes(&[random, inner, output, output])?;
    Ok(right_peak
        .max(left_peak)
        .max(candidate_peak)
        .max(retained_peak))
}

fn projection_workspace_bytes(
    matrix: &DistributedMzd,
    batch_rows: usize,
) -> Result<usize, DistributedError> {
    let tile_rows = matrix.layout.tile_rows().min(matrix.rows());
    let tile_cols = matrix.layout.tile_cols().min(matrix.cols());
    let tile_row_count = matrix.layout.tile_row_count();
    let partial_row = packed_matrix_bytes(batch_rows, tile_rows)?;
    let partial_data = tile_row_count
        .checked_mul(partial_row)
        .ok_or(DistributedError::SizeOverflow)?;
    let partial_metadata = tile_row_count
        .checked_mul(std::mem::size_of::<Option<Mzd>>())
        .ok_or(DistributedError::SizeOverflow)?;
    let retained_partials = partial_data
        .checked_add(partial_metadata)
        .ok_or(DistributedError::SizeOverflow)?;

    let segment_row = packed_matrix_bytes(batch_rows, tile_cols)?;
    let segment_column = packed_matrix_bytes(tile_cols, batch_rows)?;
    let product_column = packed_matrix_bytes(tile_rows, batch_rows)?;
    let product_row = packed_matrix_bytes(batch_rows, tile_rows)?;
    let iteration_peak = checked_bytes(&[
        retained_partials,
        segment_row,
        segment_column,
        product_column,
        product_row,
    ])?;

    let packed_projection = packed_matrix_bytes(batch_rows, matrix.rows())?;
    let reduction_peak = checked_bytes(&[retained_partials, packed_projection, packed_projection])?;
    Ok(iteration_peak.max(reduction_peak))
}

fn packed_row_bytes(bits: usize) -> Result<usize, DistributedError> {
    bits.div_ceil(m4rst::mzd::RADIX)
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or(DistributedError::SizeOverflow)
}

fn packed_matrix_bytes(rows: usize, cols: usize) -> Result<usize, DistributedError> {
    rows.checked_mul(packed_row_bytes(cols)?)
        .ok_or(DistributedError::SizeOverflow)
}

fn checked_bytes(parts: &[usize]) -> Result<usize, DistributedError> {
    parts.iter().try_fold(0_usize, |total, part| {
        total
            .checked_add(*part)
            .ok_or(DistributedError::SizeOverflow)
    })
}

pub(crate) fn map_mzd_allocation(error: MzdAllocationError) -> DistributedError {
    match error {
        MzdAllocationError::SizeOverflow => DistributedError::SizeOverflow,
        MzdAllocationError::AllocationFailed { requested_words } => {
            DistributedError::AllocationFailure {
                requested_bytes: requested_words.checked_mul(size_of::<u64>()),
            }
        }
    }
}

pub(crate) fn try_mzd(rows: usize, cols: usize) -> Result<Mzd, DistributedError> {
    Mzd::try_new(rows, cols).map_err(map_mzd_allocation)
}

pub(crate) fn try_reserve_items<T>(
    values: &mut Vec<T>,
    additional: usize,
) -> Result<(), DistributedError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| DistributedError::AllocationFailure {
            requested_bytes: values
                .len()
                .checked_add(additional)
                .and_then(|items| items.checked_mul(size_of::<T>())),
        })
}

pub(crate) fn try_zeroed_words(words: usize) -> Result<Vec<u64>, DistributedError> {
    let mut values = Vec::new();
    try_reserve_items(&mut values, words)?;
    values.resize(words, 0);
    Ok(values)
}

fn path_hash(path: &Path) -> u64 {
    checksum(path.as_os_str().as_encoded_bytes())
}

fn agree_path(world: &World, path: &Path) -> Result<(), DistributedError> {
    world.agree_u64(path_hash(path), "path")
}

fn pack_root_tile(
    layout: &Layout,
    coord: TileCoord,
    buffer: &[u8],
) -> Result<Mzd, DistributedError> {
    let (row_start, row_end, col_start, col_end) = layout
        .bounds(coord)
        .expect("canonical tile coordinate is valid");
    let start = row_start * layout.cols() + col_start;
    Mzd::try_from_strided_byte_buffer(
        row_end - row_start,
        col_end - col_start,
        layout.cols(),
        &buffer[start..],
    )
    .map_err(map_mzd_allocation)
}

fn matrix_from_words(
    layout: &Layout,
    coord: TileCoord,
    words: &[u64],
) -> Result<Mzd, DistributedError> {
    let (row_start, row_end, col_start, col_end) = layout.bounds(coord)?;
    let mut matrix = try_mzd(row_end - row_start, col_end - col_start)?;
    if matrix.as_words().len() != words.len() {
        return Err(DistributedError::Manifest(format!(
            "tile ({}, {}) has {} words, expected {}",
            coord.row,
            coord.col,
            words.len(),
            matrix.as_words().len()
        )));
    }
    matrix.as_words_mut().copy_from_slice(words);
    if matrix.width() != 0 && matrix.high_bitmask() != u64::MAX {
        let last = matrix.width() - 1;
        for row in 0..matrix.rows() {
            if matrix.row_words(row)[last] & !matrix.high_bitmask() != 0 {
                return Err(DistributedError::Manifest(format!(
                    "tile ({}, {}) has nonzero padding bits",
                    coord.row, coord.col
                )));
            }
        }
    }
    Ok(matrix)
}

fn paste_tile(layout: &Layout, coord: TileCoord, tile: &Mzd, output: &mut Mzd) {
    let (row_start, _, col_start, _) = layout
        .bounds(coord)
        .expect("canonical tile coordinate is valid");
    output.paste_into(row_start, col_start, tile);
}

fn paste_tile_words(
    layout: &Layout,
    coord: TileCoord,
    words: &[u64],
    output: &mut Mzd,
) -> Result<(), DistributedError> {
    let (row_start, row_end, col_start, col_end) = layout.bounds(coord)?;
    let word_start = col_start / m4rst::mzd::RADIX;
    let word_count = (col_end - col_start).div_ceil(m4rst::mzd::RADIX);
    let expected = (row_end - row_start)
        .checked_mul(word_count)
        .ok_or(DistributedError::SizeOverflow)?;
    if words.len() != expected {
        return Err(DistributedError::CollectiveFailure(format!(
            "tile ({}, {}) transferred {} words, expected {expected}",
            coord.row,
            coord.col,
            words.len()
        )));
    }
    for (row_offset, source) in words.chunks_exact(word_count).enumerate() {
        output.row_words_mut(row_start + row_offset)[word_start..word_start + word_count]
            .copy_from_slice(source);
    }
    Ok(())
}

fn xor_tile_overlap(
    destination_layout: &Layout,
    destination: &mut LocalTile,
    source_layout: &Layout,
    source_coord: TileCoord,
    source: &Mzd,
) -> Result<(), DistributedError> {
    let Some((row_start, row_end, col_start, col_end, destination_bounds, source_bounds)) =
        tile_intersection(
            destination_layout,
            destination.coord,
            source_layout,
            source_coord,
        )?
    else {
        return Ok(());
    };
    let (_, _, destination_col_start, _) = destination_bounds;
    let (source_row_start, _, source_col_start, _) = source_bounds;
    let destination_row_start = destination_bounds.0;
    let destination_word_start = (col_start - destination_col_start) / m4rst::mzd::RADIX;
    let source_word_start = (col_start - source_col_start) / m4rst::mzd::RADIX;
    let word_count = (col_end - col_start).div_ceil(m4rst::mzd::RADIX);
    let trailing_bits = (col_end - col_start) % m4rst::mzd::RADIX;
    let destination = destination.matrix_mut();
    for row in row_start..row_end {
        let destination_words = &mut destination.row_words_mut(row - destination_row_start)
            [destination_word_start..destination_word_start + word_count];
        let source_words = &source.row_words(row - source_row_start)
            [source_word_start..source_word_start + word_count];
        for (word_index, (destination, source)) in
            destination_words.iter_mut().zip(source_words).enumerate()
        {
            let mask = if word_index + 1 == word_count && trailing_bits != 0 {
                (1_u64 << trailing_bits) - 1
            } else {
                u64::MAX
            };
            *destination ^= source & mask;
        }
    }
    Ok(())
}

fn tile_overlap_equal(
    destination_layout: &Layout,
    destination: &LocalTile,
    source_layout: &Layout,
    source_coord: TileCoord,
    source: &Mzd,
) -> Result<bool, DistributedError> {
    let Some((row_start, row_end, col_start, col_end, destination_bounds, source_bounds)) =
        tile_intersection(
            destination_layout,
            destination.coord,
            source_layout,
            source_coord,
        )?
    else {
        return Ok(true);
    };
    let (destination_row_start, _, destination_col_start, _) = destination_bounds;
    let (source_row_start, _, source_col_start, _) = source_bounds;
    let destination_word_start = (col_start - destination_col_start) / m4rst::mzd::RADIX;
    let source_word_start = (col_start - source_col_start) / m4rst::mzd::RADIX;
    let word_count = (col_end - col_start).div_ceil(m4rst::mzd::RADIX);
    let trailing_bits = (col_end - col_start) % m4rst::mzd::RADIX;
    for row in row_start..row_end {
        let destination_words = &destination.matrix.row_words(row - destination_row_start)
            [destination_word_start..destination_word_start + word_count];
        let source_words = &source.row_words(row - source_row_start)
            [source_word_start..source_word_start + word_count];
        for (word_index, (destination, source)) in
            destination_words.iter().zip(source_words).enumerate()
        {
            let mask = if word_index + 1 == word_count && trailing_bits != 0 {
                (1_u64 << trailing_bits) - 1
            } else {
                u64::MAX
            };
            if (destination ^ source) & mask != 0 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[allow(clippy::type_complexity)]
fn tile_intersection(
    left_layout: &Layout,
    left_coord: TileCoord,
    right_layout: &Layout,
    right_coord: TileCoord,
) -> Result<
    Option<(
        usize,
        usize,
        usize,
        usize,
        (usize, usize, usize, usize),
        (usize, usize, usize, usize),
    )>,
    DistributedError,
> {
    let left = left_layout.bounds(left_coord)?;
    let right = right_layout.bounds(right_coord)?;
    let row_start = left.0.max(right.0);
    let row_end = left.1.min(right.1);
    let col_start = left.2.max(right.2);
    let col_end = left.3.min(right.3);
    if row_start >= row_end || col_start >= col_end {
        Ok(None)
    } else {
        debug_assert!(col_start.is_multiple_of(m4rst::mzd::RADIX));
        Ok(Some((row_start, row_end, col_start, col_end, left, right)))
    }
}

fn write_tile_file(directory: &Path, tile: &LocalTile) -> Result<(), DistributedError> {
    let filename = tile_filename(tile.coord);
    let final_path = directory.join(&filename);
    let temporary = directory.join(format!(".{filename}.tmp-{}", std::process::id()));
    fs::write(&temporary, words_to_bytes(tile.matrix.as_words()))
        .map_err(|error| DistributedError::Io(error.to_string()))?;
    fs::rename(&temporary, final_path).map_err(|error| DistributedError::Io(error.to_string()))
}

fn write_manifest_file(
    directory: &Path,
    manifest: &MatrixManifest,
) -> Result<(), DistributedError> {
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| DistributedError::Manifest(error.to_string()))?;
    let final_path = directory.join(MANIFEST_NAME);
    let temporary: PathBuf = directory.join(format!(".{MANIFEST_NAME}.tmp"));
    fs::write(&temporary, bytes).map_err(|error| DistributedError::Io(error.to_string()))?;
    fs::rename(&temporary, final_path).map_err(|error| DistributedError::Io(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{matmul_left_panel_batches, Layout};

    #[test]
    fn left_panel_batches_are_byte_bounded_and_topology_neutral() {
        let layout = Layout::new(1_310_720, 65_536, 8_192, 4_096, 8).unwrap();
        let batches = matmul_left_panel_batches(&layout, 0, 4_096, 64).unwrap();
        assert_eq!(batches.len(), 40);
        assert!(batches.iter().all(|batch| batch.len() == 2));
        assert_eq!(batches[0], vec![0, 2]);

        let wide_product = matmul_left_panel_batches(&layout, 0, 4_096, 65_536).unwrap();
        assert_eq!(wide_product.len(), 80);
        assert!(wide_product.iter().all(|batch| batch.len() == 1));
    }
}
