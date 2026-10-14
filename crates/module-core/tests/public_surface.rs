// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use core::cmp::Ordering;

use module_core::{ModuleOrder, ModuleOrderError, ModuleTerm, PositionModuleOrder};
use poly_core::{MonomialArena, MonomialOrder};

#[test]
fn public_orders_retain_positions_and_nested_schreyer_frames() {
    let mut arena = MonomialArena::new(2, MonomialOrder::Lexicographic);
    let one = arena.intern_exponents(&[0, 0]).unwrap();
    let x = arena.intern_exponents(&[1, 0]).unwrap();
    let y = arena.intern_exponents(&[0, 1]).unwrap();

    assert_eq!(
        PositionModuleOrder::TermOverPosition.compare(
            ModuleTerm::new(x, 1),
            ModuleTerm::new(y, 0),
            &arena,
        ),
        Ordering::Greater,
    );

    let first = ModuleOrder::schreyer(
        PositionModuleOrder::TermOverPosition.into(),
        vec![ModuleTerm::new(x, 0), ModuleTerm::new(y, 0)],
    );
    let nested = ModuleOrder::schreyer(first, vec![ModuleTerm::new(y, 0), ModuleTerm::new(x, 1)]);
    assert_eq!(
        nested
            .compare(ModuleTerm::new(x, 0), ModuleTerm::new(y, 1), &mut arena,)
            .unwrap(),
        Ordering::Greater,
    );
    assert_eq!(
        nested
            .compare(ModuleTerm::new(one, 2), ModuleTerm::new(one, 0), &mut arena,)
            .unwrap_err(),
        ModuleOrderError::ComponentOutOfRange {
            component: 2,
            frame_len: 2,
        },
    );
}
