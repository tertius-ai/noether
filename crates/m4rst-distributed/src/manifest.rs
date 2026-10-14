// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Process-grid-independent distributed matrix manifests.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{DistributedError, Layout, TileCoord};

pub(crate) const FORMAT_VERSION: u32 = 1;
pub(crate) const MANIFEST_NAME: &str = "manifest.json";

/// One canonical packed tile record in a distributed matrix manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileRecord {
    /// Tile-grid coordinate.
    pub coord: TileCoord,
    /// Actual row count, including tail-tile truncation.
    pub rows: usize,
    /// Actual column count, including tail-tile truncation.
    pub cols: usize,
    /// Packed `u64` word count.
    pub words: usize,
    /// FNV-1a checksum of the little-endian packed file.
    pub checksum: u64,
    /// Canonical relative tile filename.
    pub file: String,
}

/// Versioned, process-grid-independent dense `F_2` matrix manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixManifest {
    /// File format version.
    pub version: u32,
    /// Stable coefficient-field identity.
    pub field: String,
    /// Logical row count.
    pub rows: usize,
    /// Logical column count.
    pub cols: usize,
    /// Packed word bit width.
    pub word_bits: u32,
    /// Bit order within each word.
    pub bit_order: String,
    /// Nominal tile row count.
    pub tile_rows: usize,
    /// Nominal word-aligned tile column count.
    pub tile_cols: usize,
    /// Optional deterministic construction seed.
    pub seed: Option<u64>,
    /// Canonical row-major tile records.
    pub tiles: Vec<TileRecord>,
}

impl MatrixManifest {
    pub(crate) fn validate(&self, world_size: usize) -> Result<Layout, DistributedError> {
        if self.version != FORMAT_VERSION {
            return Err(DistributedError::Manifest(format!(
                "unsupported format version {}",
                self.version
            )));
        }
        if self.field != "F2" || self.word_bits != 64 || self.bit_order != "lsb-first" {
            return Err(DistributedError::Manifest(
                "manifest does not describe canonical packed F2 data".to_owned(),
            ));
        }
        let layout = Layout::new(
            self.rows,
            self.cols,
            self.tile_rows,
            self.tile_cols,
            world_size,
        )?;
        let expected = layout.coordinates();
        if self.tiles.len() != expected.len() {
            return Err(DistributedError::Manifest(format!(
                "manifest has {} tiles, expected {}",
                self.tiles.len(),
                expected.len()
            )));
        }
        let mut seen = BTreeSet::new();
        for (record, expected_coord) in self.tiles.iter().zip(expected) {
            if record.coord != expected_coord || !seen.insert(record.coord) {
                return Err(DistributedError::Manifest(
                    "tile records are not unique canonical row-major coordinates".to_owned(),
                ));
            }
            let (row_start, row_end, col_start, col_end) = layout.bounds(record.coord)?;
            let rows = row_end - row_start;
            let cols = col_end - col_start;
            let words = rows
                .checked_mul(cols.div_ceil(m4rst::mzd::RADIX))
                .ok_or(DistributedError::SizeOverflow)?;
            if (record.rows, record.cols, record.words) != (rows, cols, words) {
                return Err(DistributedError::Manifest(format!(
                    "tile ({}, {}) shape/word count is inconsistent",
                    record.coord.row, record.coord.col
                )));
            }
            if record.file != tile_filename(record.coord) {
                return Err(DistributedError::Manifest(format!(
                    "tile ({}, {}) has noncanonical filename",
                    record.coord.row, record.coord.col
                )));
            }
        }
        Ok(layout)
    }
}

pub(crate) fn tile_filename(coord: TileCoord) -> String {
    format!("tile-{:08}-{:08}.bin", coord.row, coord.col)
}

pub(crate) fn checksum(bytes: &[u8]) -> u64 {
    let mut result = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        result ^= u64::from(*byte);
        result = result.wrapping_mul(0x0000_0100_0000_01b3);
    }
    result
}

pub(crate) fn checksum_words(words: &[u64]) -> u64 {
    let mut result = 0xcbf2_9ce4_8422_2325_u64;
    for word in words {
        for byte in word.to_le_bytes() {
            result ^= u64::from(byte);
            result = result.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    result
}

pub(crate) fn words_to_bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

pub(crate) fn bytes_to_words(bytes: &[u8]) -> Result<Vec<u64>, DistributedError> {
    if !bytes.len().is_multiple_of(size_of::<u64>()) {
        return Err(DistributedError::Manifest(
            "packed tile byte length is not a multiple of eight".to_owned(),
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("chunk length is eight")))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{bytes_to_words, checksum, words_to_bytes};

    #[test]
    fn packed_words_round_trip_in_declared_endianness() {
        let words = [0, 1, u64::MAX, 0x0123_4567_89ab_cdef];
        let bytes = words_to_bytes(&words);
        assert_eq!(&bytes[24..32], &0x0123_4567_89ab_cdef_u64.to_le_bytes());
        assert_eq!(bytes_to_words(&bytes).unwrap(), words);
        assert_ne!(checksum(&bytes), 0);
    }
}
