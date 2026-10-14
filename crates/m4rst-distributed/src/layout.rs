// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Deterministic tile and process-grid layout.

use serde::{Deserialize, Serialize};

use crate::DistributedError;

/// Default number of rows in one distributed tile.
pub const DEFAULT_TILE_ROWS: usize = 1024;
/// Default number of columns in one distributed tile.
pub const DEFAULT_TILE_COLS: usize = 4096;

/// Choose word-aligned defaults that expose at least one tile per process-grid
/// row and column when the logical shape permits it.
pub(crate) fn default_tile_shape(
    rows: usize,
    cols: usize,
    world_size: usize,
) -> Result<(usize, usize), DistributedError> {
    let grid = ProcessGrid::new(world_size)?;
    let tile_rows = if rows == 0 {
        DEFAULT_TILE_ROWS
    } else {
        DEFAULT_TILE_ROWS.min(rows.div_ceil(grid.rows())).max(1)
    };
    let tile_cols = if cols == 0 {
        DEFAULT_TILE_COLS
    } else {
        let columns_per_process = cols.div_ceil(grid.cols());
        let aligned = columns_per_process
            .div_ceil(m4rst::mzd::RADIX)
            .checked_mul(m4rst::mzd::RADIX)
            .ok_or(DistributedError::SizeOverflow)?;
        DEFAULT_TILE_COLS.min(aligned).max(m4rst::mzd::RADIX)
    };
    Ok((tile_rows, tile_cols))
}

/// A tile coordinate in the logical row/column tile grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TileCoord {
    /// Zero-based tile-row index.
    pub row: usize,
    /// Zero-based tile-column index.
    pub col: usize,
}

/// A deterministic near-square two-dimensional MPI process grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessGrid {
    rows: usize,
    cols: usize,
}

impl ProcessGrid {
    /// Choose the factorization with the greatest row count not exceeding the
    /// square root. Prime world sizes therefore use a `1 x size` grid.
    pub fn new(world_size: usize) -> Result<Self, DistributedError> {
        if world_size == 0 {
            return Err(DistributedError::InvalidLayout(
                "world size must be positive".to_owned(),
            ));
        }
        let mut rows = 1;
        let mut candidate = 1;
        while candidate <= world_size / candidate {
            if world_size.is_multiple_of(candidate) {
                rows = candidate;
            }
            candidate += 1;
        }
        Ok(Self {
            rows,
            cols: world_size / rows,
        })
    }

    /// Number of process rows.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Number of process columns.
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// World rank owning one block-cyclic tile coordinate.
    #[must_use]
    pub const fn owner(self, coord: TileCoord) -> usize {
        (coord.row % self.rows) * self.cols + coord.col % self.cols
    }
}

/// Complete local interpretation of a logical tiled matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    rows: usize,
    cols: usize,
    tile_rows: usize,
    tile_cols: usize,
    tile_row_count: usize,
    tile_col_count: usize,
    process_grid: ProcessGrid,
}

impl Layout {
    /// Validate matrix, tile, and world geometry.
    pub fn new(
        rows: usize,
        cols: usize,
        tile_rows: usize,
        tile_cols: usize,
        world_size: usize,
    ) -> Result<Self, DistributedError> {
        if tile_rows == 0 {
            return Err(DistributedError::InvalidLayout(
                "tile_rows must be positive".to_owned(),
            ));
        }
        if tile_cols == 0 || !tile_cols.is_multiple_of(m4rst::mzd::RADIX) {
            return Err(DistributedError::InvalidLayout(format!(
                "tile_cols must be a positive multiple of {}",
                m4rst::mzd::RADIX
            )));
        }
        let tile_row_count = rows.div_ceil(tile_rows);
        let tile_col_count = cols.div_ceil(tile_cols);
        tile_row_count
            .checked_mul(tile_col_count)
            .ok_or(DistributedError::SizeOverflow)?;
        rows.checked_mul(cols.div_ceil(m4rst::mzd::RADIX))
            .and_then(|words| words.checked_mul(std::mem::size_of::<u64>()))
            .ok_or(DistributedError::SizeOverflow)?;
        Ok(Self {
            rows,
            cols,
            tile_rows,
            tile_cols,
            tile_row_count,
            tile_col_count,
            process_grid: ProcessGrid::new(world_size)?,
        })
    }

    /// Logical row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Logical column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Nominal tile row count.
    #[must_use]
    pub const fn tile_rows(&self) -> usize {
        self.tile_rows
    }

    /// Nominal word-aligned tile column count.
    #[must_use]
    pub const fn tile_cols(&self) -> usize {
        self.tile_cols
    }

    /// Number of logical tile rows.
    #[must_use]
    pub const fn tile_row_count(&self) -> usize {
        self.tile_row_count
    }

    /// Number of logical tile columns.
    #[must_use]
    pub const fn tile_col_count(&self) -> usize {
        self.tile_col_count
    }

    /// Process grid selected for the current world.
    #[must_use]
    pub const fn process_grid(&self) -> ProcessGrid {
        self.process_grid
    }

    /// Canonical row-major tile coordinates.
    #[must_use]
    pub fn coordinates(&self) -> Vec<TileCoord> {
        self.coordinate_iter().collect()
    }

    /// Iterate canonical row-major tile coordinates without allocating a
    /// matrix-scale coordinate table.
    pub fn coordinate_iter(&self) -> impl Iterator<Item = TileCoord> + '_ {
        (0..self.tile_row_count)
            .flat_map(|row| (0..self.tile_col_count).map(move |col| TileCoord { row, col }))
    }

    /// Rank owning one tile.
    #[must_use]
    pub const fn owner(&self, coord: TileCoord) -> usize {
        self.process_grid.owner(coord)
    }

    /// Global half-open row/column bounds and local shape of a tile.
    pub fn bounds(
        &self,
        coord: TileCoord,
    ) -> Result<(usize, usize, usize, usize), DistributedError> {
        if coord.row >= self.tile_row_count || coord.col >= self.tile_col_count {
            return Err(DistributedError::InvalidLayout(format!(
                "tile ({}, {}) is outside {} x {} tile grid",
                coord.row, coord.col, self.tile_row_count, self.tile_col_count
            )));
        }
        let row_start = coord
            .row
            .checked_mul(self.tile_rows)
            .ok_or(DistributedError::SizeOverflow)?;
        let col_start = coord
            .col
            .checked_mul(self.tile_cols)
            .ok_or(DistributedError::SizeOverflow)?;
        Ok((
            row_start,
            row_start.saturating_add(self.tile_rows).min(self.rows),
            col_start,
            col_start.saturating_add(self.tile_cols).min(self.cols),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{default_tile_shape, Layout, ProcessGrid, TileCoord};

    #[test]
    fn default_tiles_activate_the_process_grid_when_shape_permits() {
        assert_eq!(default_tile_shape(100, 100, 4).unwrap(), (50, 64));
        assert_eq!(default_tile_shape(1, 1, 4).unwrap(), (1, 64));
        assert_eq!(default_tile_shape(0, 0, 4).unwrap(), (1024, 4096));
    }

    #[test]
    fn process_grid_is_near_square_and_deterministic() {
        assert_eq!(
            (
                ProcessGrid::new(1).unwrap().rows(),
                ProcessGrid::new(1).unwrap().cols()
            ),
            (1, 1)
        );
        assert_eq!(
            (
                ProcessGrid::new(4).unwrap().rows(),
                ProcessGrid::new(4).unwrap().cols()
            ),
            (2, 2)
        );
        assert_eq!(
            (
                ProcessGrid::new(6).unwrap().rows(),
                ProcessGrid::new(6).unwrap().cols()
            ),
            (2, 3)
        );
        assert_eq!(
            (
                ProcessGrid::new(7).unwrap().rows(),
                ProcessGrid::new(7).unwrap().cols()
            ),
            (1, 7)
        );
    }

    #[test]
    fn block_cyclic_ownership_is_total() {
        let layout = Layout::new(9, 257, 4, 128, 4).unwrap();
        assert_eq!(layout.coordinates().len(), 9);
        assert_eq!(layout.owner(TileCoord { row: 0, col: 0 }), 0);
        assert_eq!(layout.owner(TileCoord { row: 0, col: 1 }), 1);
        assert_eq!(layout.owner(TileCoord { row: 1, col: 0 }), 2);
        assert_eq!(layout.owner(TileCoord { row: 2, col: 2 }), 0);
        assert_eq!(
            layout.bounds(TileCoord { row: 2, col: 2 }).unwrap(),
            (8, 9, 256, 257)
        );
    }
}
