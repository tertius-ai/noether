// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Packed module terms and exact module-monomial orders.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::cmp::Ordering;
use core::fmt;

use poly_core::{MonomialArena, MonomialArenaError, MonomialId};

/// A monomial times one basis vector of a free module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModuleTerm {
    monomial: MonomialId,
    component: usize,
}

impl ModuleTerm {
    /// Construct `monomial * e_component`.
    pub const fn new(monomial: MonomialId, component: usize) -> Self {
        Self {
            monomial,
            component,
        }
    }

    /// Polynomial monomial identifier.
    pub const fn monomial(self) -> MonomialId {
        self.monomial
    }

    /// Zero-based free-module basis position.
    pub const fn component(self) -> usize {
        self.component
    }
}

/// The two standard extensions of a polynomial monomial order to module
/// terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PositionModuleOrder {
    /// Compare polynomial monomials first, then basis positions.
    TermOverPosition,
    /// Compare basis positions first, then polynomial monomials.
    PositionOverTerm,
}

/// An exact module-monomial order over one caller-owned monomial arena.
///
/// Standard position orders are the recursion base. A Schreyer order retains
/// the target order and the ordered target leaders that induce it, so nested
/// resolution frames do not need to unpack exponent vectors between stages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleOrder {
    /// A standard term/position extension of the arena's polynomial order.
    Position(PositionModuleOrder),
    /// The order induced from an ordered leading frame in the target module.
    Schreyer {
        /// Exact order on the target module.
        target: Box<Self>,
        /// Ordered leading terms of the source generators in the target.
        leaders: Vec<ModuleTerm>,
    },
}

impl ModuleOrder {
    /// Construct a standard module order.
    pub const fn position(order: PositionModuleOrder) -> Self {
        Self::Position(order)
    }

    /// Construct the Schreyer order induced by `leaders` in `target`.
    pub fn schreyer(target: Self, leaders: Vec<ModuleTerm>) -> Self {
        Self::Schreyer {
            target: Box::new(target),
            leaders,
        }
    }

    /// Ordered leading frame when this is an induced Schreyer order.
    pub fn schreyer_data(&self) -> Option<(&Self, &[ModuleTerm])> {
        match self {
            Self::Position(_) => None,
            Self::Schreyer { target, leaders } => Some((target, leaders)),
        }
    }

    /// Compare two terms using this exact, possibly nested order.
    ///
    /// A smaller component wins every position tie, including equal induced
    /// target terms in a Schreyer frame.
    pub fn compare(
        &self,
        left: ModuleTerm,
        right: ModuleTerm,
        arena: &mut MonomialArena,
    ) -> Result<Ordering, ModuleOrderError> {
        match self {
            Self::Position(order) => Ok(order.compare(left, right, arena)),
            Self::Schreyer { target, leaders } => {
                let left_leader = leader(leaders, left.component)?;
                let right_leader = leader(leaders, right.component)?;
                let left_induced = ModuleTerm::new(
                    arena.product(left.monomial, left_leader.monomial)?,
                    left_leader.component,
                );
                let right_induced = ModuleTerm::new(
                    arena.product(right.monomial, right_leader.monomial)?,
                    right_leader.component,
                );
                Ok(target
                    .compare(left_induced, right_induced, arena)?
                    .then_with(|| right.component.cmp(&left.component)))
            }
        }
    }
}

impl From<PositionModuleOrder> for ModuleOrder {
    fn from(value: PositionModuleOrder) -> Self {
        Self::position(value)
    }
}

impl PositionModuleOrder {
    /// Compare two module terms in one monomial arena.
    ///
    /// On a position tie, a smaller component index is greater. Thus `e_0`
    /// precedes `e_1` in a descending canonical term sequence.
    pub fn compare(self, left: ModuleTerm, right: ModuleTerm, arena: &MonomialArena) -> Ordering {
        let position = || right.component.cmp(&left.component);
        match self {
            Self::TermOverPosition => arena
                .compare(left.monomial, right.monomial)
                .then_with(position),
            Self::PositionOverTerm => {
                position().then_with(|| arena.compare(left.monomial, right.monomial))
            }
        }
    }
}

/// Failure to compare terms in an induced Schreyer order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleOrderError {
    /// A source component did not occur in the ordered leading frame.
    ComponentOutOfRange {
        /// Rejected component index.
        component: usize,
        /// Number of entries in the frame.
        frame_len: usize,
    },
    /// Multiplying a source monomial by a frame leader failed.
    Monomial(MonomialArenaError),
}

impl fmt::Display for ModuleOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ComponentOutOfRange {
                component,
                frame_len,
            } => write!(
                f,
                "module component {component} is outside Schreyer frame length {frame_len}"
            ),
            Self::Monomial(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ModuleOrderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Monomial(error) => Some(error),
            Self::ComponentOutOfRange { .. } => None,
        }
    }
}

impl From<MonomialArenaError> for ModuleOrderError {
    fn from(value: MonomialArenaError) -> Self {
        Self::Monomial(value)
    }
}

/// Ordered leading module terms inducing a Schreyer order on a source module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchreyerFrame {
    target_order: PositionModuleOrder,
    leaders: Vec<ModuleTerm>,
}

impl SchreyerFrame {
    /// Construct an induced order from the ordered leading terms of source
    /// generators in their target module.
    pub fn new(target_order: PositionModuleOrder, leaders: Vec<ModuleTerm>) -> Self {
        Self {
            target_order,
            leaders,
        }
    }

    /// Module order used in the target frame.
    pub const fn target_order(&self) -> PositionModuleOrder {
        self.target_order
    }

    /// Ordered leading module terms inducing this frame.
    pub fn leaders(&self) -> &[ModuleTerm] {
        &self.leaders
    }

    /// Compare source terms using their induced target leaders.
    ///
    /// If induced target terms agree, the smaller source component wins. This
    /// tie rule is part of the exact Schreyer order and depends on the retained
    /// generator ordering.
    pub fn compare(
        &self,
        left: ModuleTerm,
        right: ModuleTerm,
        arena: &mut MonomialArena,
    ) -> Result<Ordering, ModuleOrderError> {
        let left_leader = self.leader(left.component)?;
        let right_leader = self.leader(right.component)?;
        let left_induced = ModuleTerm::new(
            arena.product(left.monomial, left_leader.monomial)?,
            left_leader.component,
        );
        let right_induced = ModuleTerm::new(
            arena.product(right.monomial, right_leader.monomial)?,
            right_leader.component,
        );
        Ok(self
            .target_order
            .compare(left_induced, right_induced, arena)
            .then_with(|| right.component.cmp(&left.component)))
    }

    fn leader(&self, component: usize) -> Result<ModuleTerm, ModuleOrderError> {
        leader(&self.leaders, component)
    }
}

fn leader(leaders: &[ModuleTerm], component: usize) -> Result<ModuleTerm, ModuleOrderError> {
    leaders
        .get(component)
        .copied()
        .ok_or(ModuleOrderError::ComponentOutOfRange {
            component,
            frame_len: leaders.len(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use poly_core::MonomialOrder;

    #[test]
    fn standard_module_orders_distinguish_term_and_position_priority() {
        let mut arena = MonomialArena::new(2, MonomialOrder::Lexicographic);
        let x = arena.intern_exponents(&[1, 0]).unwrap();
        let y = arena.intern_exponents(&[0, 1]).unwrap();
        let x_e1 = ModuleTerm::new(x, 1);
        let y_e0 = ModuleTerm::new(y, 0);

        assert_eq!(
            PositionModuleOrder::TermOverPosition.compare(x_e1, y_e0, &arena),
            Ordering::Greater
        );
        assert_eq!(
            PositionModuleOrder::PositionOverTerm.compare(x_e1, y_e0, &arena),
            Ordering::Less
        );
    }

    #[test]
    fn schreyer_order_compares_induced_target_leaders_and_breaks_ties_by_source_position() {
        let mut arena = MonomialArena::new(2, MonomialOrder::Lexicographic);
        let one = arena.intern_exponents(&[0, 0]).unwrap();
        let x = arena.intern_exponents(&[1, 0]).unwrap();
        let y = arena.intern_exponents(&[0, 1]).unwrap();
        let frame = SchreyerFrame::new(
            PositionModuleOrder::TermOverPosition,
            vec![ModuleTerm::new(x, 0), ModuleTerm::new(y, 0)],
        );

        assert_eq!(
            frame
                .compare(ModuleTerm::new(y, 0), ModuleTerm::new(x, 1), &mut arena)
                .unwrap(),
            Ordering::Greater
        );
        assert_eq!(
            frame
                .compare(ModuleTerm::new(one, 2), ModuleTerm::new(one, 0), &mut arena,)
                .unwrap_err(),
            ModuleOrderError::ComponentOutOfRange {
                component: 2,
                frame_len: 2
            }
        );
    }

    #[test]
    fn recursive_order_retains_nested_schreyer_frames() {
        let mut arena = MonomialArena::new(2, MonomialOrder::Lexicographic);
        let one = arena.intern_exponents(&[0, 0]).unwrap();
        let x = arena.intern_exponents(&[1, 0]).unwrap();
        let y = arena.intern_exponents(&[0, 1]).unwrap();
        let first = ModuleOrder::schreyer(
            PositionModuleOrder::TermOverPosition.into(),
            vec![ModuleTerm::new(x, 0), ModuleTerm::new(y, 0)],
        );
        let nested =
            ModuleOrder::schreyer(first, vec![ModuleTerm::new(y, 0), ModuleTerm::new(x, 1)]);

        assert_eq!(
            nested
                .compare(ModuleTerm::new(x, 0), ModuleTerm::new(y, 1), &mut arena)
                .unwrap(),
            Ordering::Greater
        );
        assert_eq!(
            nested
                .compare(ModuleTerm::new(one, 2), ModuleTerm::new(one, 0), &mut arena)
                .unwrap_err(),
            ModuleOrderError::ComponentOutOfRange {
                component: 2,
                frame_len: 2,
            }
        );
    }
}
