// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Process-grid-independent random tile generation.

use m4rst::Mzd;

use crate::{DistributedError, Layout, TileCoord};

pub(crate) fn random_tile(
    layout: &Layout,
    coord: TileCoord,
    seed: u64,
) -> Result<Mzd, DistributedError> {
    let (row_start, row_end, col_start, col_end) = layout
        .bounds(coord)
        .expect("a coordinate produced by Layout is valid");
    Mzd::try_random_word_aligned_submatrix(
        layout.rows(),
        layout.cols(),
        row_start,
        row_end,
        col_start,
        col_end,
        seed,
    )
    .map_err(super::matrix::map_mzd_allocation)
}

#[cfg(test)]
mod tests {
    use m4rst::Mzd;

    use crate::{random::random_tile, Layout};

    #[test]
    fn tiled_random_is_bit_identical_to_mzd_for_every_grid() {
        for &(rows, cols, tile_rows, tile_cols, seed) in &[
            (0, 0, 3, 64, 0),
            (1, 1, 1, 64, 1),
            (7, 193, 3, 128, 0),
            (11, 257, 4, 64, 0xDEAD_BEEF),
            (129, 1089, 17, 256, 0x1729),
        ] {
            let layout = Layout::new(rows, cols, tile_rows, tile_cols, 6).unwrap();
            let mut assembled = Mzd::new(rows, cols);
            for coord in layout.coordinates() {
                let (row_start, _, col_start, _) = layout.bounds(coord).unwrap();
                assembled.paste_into(
                    row_start,
                    col_start,
                    &random_tile(&layout, coord, seed).unwrap(),
                );
            }
            assert_eq!(
                assembled.as_words(),
                Mzd::random(rows, cols, seed).as_words()
            );
        }
    }
}
