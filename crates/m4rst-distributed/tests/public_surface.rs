// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use m4rst_distributed::{Layout, ProcessGrid, TileCoord};

#[test]
fn public_layout_is_process_count_specific_but_data_coordinates_are_not() {
    let one = Layout::new(9, 257, 4, 128, 1).unwrap();
    let six = Layout::new(9, 257, 4, 128, 6).unwrap();

    assert_eq!(one.coordinates(), six.coordinates());
    assert_eq!(six.process_grid(), ProcessGrid::new(6).unwrap());
    assert_eq!(six.owner(TileCoord { row: 2, col: 2 }), 2);
    assert_eq!(
        six.bounds(TileCoord { row: 2, col: 2 }).unwrap(),
        (8, 9, 256, 257)
    );
}
