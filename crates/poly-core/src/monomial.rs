// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Packed exact monomials and round-local interning.

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

/// A supported admissible monomial order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MonomialOrder {
    /// Lexicographic order, with the first variable dominant.
    Lexicographic,
    /// Total degree followed by lexicographic order.
    GradedLexicographic,
    /// Total degree followed by reverse lexicographic order.
    #[default]
    GradedReverseLexicographic,
}

#[derive(Clone, PartialEq, Eq)]
enum Exponents {
    U16(Arc<[u16]>),
    U32(Arc<[u32]>),
}

/// One exact fixed-arity monomial using the narrowest word representation.
#[derive(Clone, Eq)]
pub struct Monomial {
    exponents: Exponents,
}

impl Monomial {
    /// Construct from nonnegative word-sized exponents.
    pub fn new(exponents: impl Into<Box<[u32]>>) -> Self {
        let exponents = exponents.into();
        if exponents.iter().all(|&value| u16::try_from(value).is_ok()) {
            Self {
                exponents: Exponents::U16(
                    exponents
                        .iter()
                        .map(|&value| value as u16)
                        .collect::<Arc<_>>(),
                ),
            }
        } else {
            Self {
                exponents: Exponents::U32(Arc::from(exponents)),
            }
        }
    }

    /// The multiplicative identity in `arity` variables.
    pub fn one(arity: usize) -> Self {
        Self::new(vec![0; arity].into_boxed_slice())
    }

    /// Number of variables.
    pub fn arity(&self) -> usize {
        match &self.exponents {
            Exponents::U16(values) => values.len(),
            Exponents::U32(values) => values.len(),
        }
    }

    /// Exponent of one variable.
    pub fn exponent(&self, variable: usize) -> u32 {
        match &self.exponents {
            Exponents::U16(values) => u32::from(values[variable]),
            Exponents::U32(values) => values[variable],
        }
    }

    /// Whether this monomial uses 16-bit exponents.
    pub fn is_packed_u16(&self) -> bool {
        matches!(self.exponents, Exponents::U16(_))
    }

    /// Materialize exponents in the common `u32` interchange form.
    pub fn to_u32_vec(&self) -> Vec<u32> {
        (0..self.arity())
            .map(|index| self.exponent(index))
            .collect()
    }

    /// Total degree, or `None` if it does not fit `u64`.
    pub fn total_degree(&self) -> Option<u64> {
        (0..self.arity()).try_fold(0_u64, |sum, index| {
            sum.checked_add(u64::from(self.exponent(index)))
        })
    }

    /// Test monomial divisibility.
    pub fn divides(&self, other: &Self) -> bool {
        self.arity() == other.arity()
            && (0..self.arity()).all(|index| self.exponent(index) <= other.exponent(index))
    }

    /// Multiply monomials, detecting exponent overflow.
    pub fn checked_product(&self, other: &Self) -> Result<Self, ExponentOverflow> {
        self.require_same_arity(other)?;
        let exponents = (0..self.arity())
            .map(|index| {
                self.exponent(index)
                    .checked_add(other.exponent(index))
                    .ok_or(ExponentOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(exponents.into_boxed_slice()))
    }

    /// Least common multiple.
    pub fn lcm(&self, other: &Self) -> Result<Self, ExponentOverflow> {
        self.require_same_arity(other)?;
        Ok(Self::new(
            (0..self.arity())
                .map(|index| self.exponent(index).max(other.exponent(index)))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ))
    }

    /// Return `self / divisor` when divisible.
    pub fn quotient(&self, divisor: &Self) -> Option<Self> {
        divisor.divides(self).then(|| {
            Self::new(
                (0..self.arity())
                    .map(|index| self.exponent(index) - divisor.exponent(index))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        })
    }

    /// Compare in one admissible order.
    pub fn compare(&self, other: &Self, order: MonomialOrder) -> Ordering {
        assert_eq!(self.arity(), other.arity(), "monomial arity mismatch");
        match order {
            MonomialOrder::Lexicographic => lex_compare(self, other),
            MonomialOrder::GradedLexicographic => self
                .total_degree()
                .cmp(&other.total_degree())
                .then_with(|| lex_compare(self, other)),
            MonomialOrder::GradedReverseLexicographic => {
                let degree = self.total_degree().cmp(&other.total_degree());
                if degree != Ordering::Equal {
                    return degree;
                }
                for index in (0..self.arity()).rev() {
                    match self.exponent(index).cmp(&other.exponent(index)) {
                        Ordering::Equal => {}
                        Ordering::Less => return Ordering::Greater,
                        Ordering::Greater => return Ordering::Less,
                    }
                }
                Ordering::Equal
            }
        }
    }

    fn require_same_arity(&self, other: &Self) -> Result<(), ExponentOverflow> {
        if self.arity() == other.arity() {
            Ok(())
        } else {
            Err(ExponentOverflow)
        }
    }
}

fn lex_compare(left: &Monomial, right: &Monomial) -> Ordering {
    (0..left.arity())
        .map(|index| left.exponent(index).cmp(&right.exponent(index)))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

impl PartialEq for Monomial {
    fn eq(&self, other: &Self) -> bool {
        self.arity() == other.arity()
            && (0..self.arity()).all(|index| self.exponent(index) == other.exponent(index))
    }
}

impl Hash for Monomial {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.arity().hash(state);
        for index in 0..self.arity() {
            self.exponent(index).hash(state);
        }
    }
}

impl fmt::Debug for Monomial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Monomial")
            .field(&self.to_u32_vec())
            .finish()
    }
}

/// A monomial operation exceeded the word-exponent kernel regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExponentOverflow;

impl fmt::Display for ExponentOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("monomial exponent overflow or arity mismatch")
    }
}

impl std::error::Error for ExponentOverflow {}

/// Stable identifier of one monomial within a [`MonomialArena`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MonomialId(u32);

impl MonomialId {
    /// Zero-based arena index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Identity of one compatible family of interned monomial identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MonomialArenaId(u64);

/// Failure to intern a monomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MonomialArenaError {
    /// The monomial has the wrong number of variables.
    ArityMismatch {
        /// Arena arity.
        expected: usize,
        /// Received arity.
        actual: usize,
    },
    /// More than `u32::MAX` monomials were requested.
    CapacityExceeded,
    /// Exponent arithmetic overflowed.
    ExponentOverflow,
}

impl fmt::Display for MonomialArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArityMismatch { expected, actual } => {
                write!(formatter, "monomial arity {actual}; expected {expected}")
            }
            Self::CapacityExceeded => formatter.write_str("monomial arena exceeds u32 capacity"),
            Self::ExponentOverflow => formatter.write_str("monomial exponent overflow"),
        }
    }
}

impl std::error::Error for MonomialArenaError {}

#[derive(Clone, Debug)]
enum ArenaExponents {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl ArenaExponents {
    fn exponent(&self, offset: usize) -> u32 {
        match self {
            Self::U16(values) => u32::from(values[offset]),
            Self::U32(values) => values[offset],
        }
    }

    fn append(&mut self, exponents: &[u32]) {
        if matches!(self, Self::U16(_))
            && exponents
                .iter()
                .any(|&exponent| u16::try_from(exponent).is_err())
        {
            let Self::U16(values) = core::mem::replace(self, Self::U16(Vec::new())) else {
                unreachable!();
            };
            *self = Self::U32(values.into_iter().map(u32::from).collect());
        }
        match self {
            Self::U16(values) => values.extend(exponents.iter().map(|&value| value as u16)),
            Self::U32(values) => values.extend_from_slice(exponents),
        }
    }

    fn is_packed_u16(&self) -> bool {
        matches!(self, Self::U16(_))
    }

    fn allocated_bytes(&self) -> usize {
        match self {
            Self::U16(values) => values
                .capacity()
                .saturating_mul(core::mem::size_of::<u16>()),
            Self::U32(values) => values
                .capacity()
                .saturating_mul(core::mem::size_of::<u32>()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MonomialMetadata {
    degree: u64,
    fingerprint: u64,
    divisor_mask: u64,
}

#[derive(Clone, Debug)]
struct MonomialTable {
    /// Zero denotes an empty slot; an occupied slot stores `MonomialId + 1`.
    slots: Vec<u32>,
}

impl MonomialTable {
    fn with_capacity(capacity: usize) -> Self {
        let slots = capacity
            .saturating_mul(2)
            .max(8)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX / 2 + 1);
        Self {
            slots: vec![0; slots],
        }
    }

    fn needs_growth(&self, entries: usize) -> bool {
        entries.saturating_mul(2) > self.slots.len()
    }

    fn rebuild(&mut self, metadata: &[MonomialMetadata]) {
        let mut replacement = Self::with_capacity(metadata.len().saturating_mul(2));
        for (index, value) in metadata.iter().enumerate() {
            replacement.insert_vacant(
                value.fingerprint,
                MonomialId(u32::try_from(index).expect("monomial table index fits u32")),
            );
        }
        *self = replacement;
    }

    fn insert_vacant(&mut self, fingerprint: u64, id: MonomialId) {
        let mask = self.slots.len() - 1;
        let mut slot = fingerprint_slot(fingerprint) & mask;
        loop {
            if self.slots[slot] == 0 {
                self.slots[slot] = id.0 + 1;
                return;
            }
            slot = (slot + 1) & mask;
        }
    }
}

static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);

/// Deterministic monomial interner backed by one contiguous exponent slab.
///
/// Identifiers remain stable if the arena upgrades its entire exponent slab
/// from `u16` to `u32`. Hash collisions are resolved by exact exponent-vector
/// comparison.
#[derive(Debug)]
pub struct MonomialArena {
    identity: MonomialArenaId,
    arity: usize,
    order: MonomialOrder,
    fingerprint_weights: Box<[u64]>,
    mask_variables: usize,
    mask_bits_per_variable: usize,
    exponents: ArenaExponents,
    metadata: Vec<MonomialMetadata>,
    ids: MonomialTable,
    scratch: Vec<u32>,
}

/// Reusable generation-marked set for IDs from one monomial arena.
///
/// Clearing advances a generation rather than touching every allocated slot.
/// The backing array grows automatically as its arena interns new monomials.
#[derive(Clone, Debug)]
pub struct MonomialMarks {
    arena: MonomialArenaId,
    generation: u32,
    marks: Vec<u32>,
}

impl MonomialMarks {
    /// Construct an empty set associated with `arena`.
    pub fn new(arena: &MonomialArena) -> Self {
        Self {
            arena: arena.identity(),
            generation: 1,
            marks: vec![0; arena.len()],
        }
    }

    /// Bytes reserved by the generation-mark buffer.
    pub fn allocated_bytes(&self) -> usize {
        self.marks
            .capacity()
            .saturating_mul(core::mem::size_of::<u32>())
    }

    /// Remove every ID while retaining storage.
    pub fn clear(&mut self, arena: &MonomialArena) {
        self.require_arena(arena);
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    /// Insert an ID, returning whether it was absent.
    pub fn insert(&mut self, arena: &MonomialArena, id: MonomialId) -> bool {
        self.require_arena(arena);
        assert!(arena.contains(id), "monomial identifier out of range");
        if self.marks.len() < arena.len() {
            self.marks.resize(arena.len(), 0);
        }
        let previous = self.marks[id.index()];
        self.marks[id.index()] = self.generation;
        previous != self.generation
    }

    /// Whether an ID is in the current generation.
    pub fn contains(&self, arena: &MonomialArena, id: MonomialId) -> bool {
        self.require_arena(arena);
        arena.contains(id) && self.marks.get(id.index()).copied() == Some(self.generation)
    }

    fn require_arena(&self, arena: &MonomialArena) {
        assert_eq!(self.arena, arena.identity(), "monomial mark arena mismatch");
    }
}

impl MonomialArena {
    /// Construct an empty arena.
    pub fn new(arity: usize, order: MonomialOrder) -> Self {
        Self::with_capacity(arity, order, 0)
    }

    /// Construct an empty arena with space reserved for monomial IDs.
    pub fn with_capacity(arity: usize, order: MonomialOrder, capacity: usize) -> Self {
        let raw_identity = NEXT_ARENA_ID.fetch_add(1, AtomicOrdering::Relaxed);
        assert_ne!(raw_identity, 0, "monomial arena identity space exhausted");
        let exponent_capacity = capacity.checked_mul(arity).unwrap_or(0);
        let mask_variables = arity.min(64);
        Self {
            identity: MonomialArenaId(raw_identity),
            arity,
            order,
            fingerprint_weights: (0..arity).map(variable_fingerprint).collect::<Box<_>>(),
            mask_variables,
            mask_bits_per_variable: if mask_variables == 0 {
                0
            } else {
                64 / mask_variables
            },
            exponents: ArenaExponents::U16(Vec::with_capacity(exponent_capacity)),
            metadata: Vec::with_capacity(capacity),
            ids: MonomialTable::with_capacity(capacity),
            scratch: Vec::with_capacity(arity),
        }
    }

    /// Deep-copy an arena while preserving every existing monomial ID.
    ///
    /// The fork receives a fresh arena identity, so polynomials owned by the
    /// original cannot be mixed with values subsequently built in the fork.
    /// Existing numeric IDs nevertheless name the same exponent vectors in
    /// both arenas. This supports independent deterministic work from one
    /// frozen monomial vocabulary.
    pub fn fork(&self) -> Self {
        let raw_identity = NEXT_ARENA_ID.fetch_add(1, AtomicOrdering::Relaxed);
        assert_ne!(raw_identity, 0, "monomial arena identity space exhausted");
        Self {
            identity: MonomialArenaId(raw_identity),
            arity: self.arity,
            order: self.order,
            fingerprint_weights: self.fingerprint_weights.clone(),
            mask_variables: self.mask_variables,
            mask_bits_per_variable: self.mask_bits_per_variable,
            exponents: self.exponents.clone(),
            metadata: self.metadata.clone(),
            ids: self.ids.clone(),
            scratch: self.scratch.clone(),
        }
    }

    /// Stable identity of this arena.
    pub fn identity(&self) -> MonomialArenaId {
        self.identity
    }

    /// Number of variables.
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Declared monomial order.
    pub fn order(&self) -> MonomialOrder {
        self.order
    }

    /// Number of interned monomials.
    pub fn len(&self) -> usize {
        self.metadata.len()
    }

    /// Whether no monomial has been interned.
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
    }

    /// Whether an identifier is present in this arena.
    pub fn contains(&self, id: MonomialId) -> bool {
        id.index() < self.metadata.len()
    }

    /// Intern one monomial, preserving the first-insertion identifier.
    pub fn intern(&mut self, monomial: Monomial) -> Result<MonomialId, MonomialArenaError> {
        if monomial.arity() != self.arity {
            return Err(MonomialArenaError::ArityMismatch {
                expected: self.arity,
                actual: monomial.arity(),
            });
        }
        let mut exponents = core::mem::take(&mut self.scratch);
        exponents.clear();
        exponents.extend((0..self.arity).map(|index| monomial.exponent(index)));
        let result = self.intern_exponents(&exponents);
        self.scratch = exponents;
        result
    }

    /// Intern one exponent vector without first allocating a [`Monomial`].
    pub fn intern_exponents(
        &mut self,
        exponents: &[u32],
    ) -> Result<MonomialId, MonomialArenaError> {
        if exponents.len() != self.arity {
            return Err(MonomialArenaError::ArityMismatch {
                expected: self.arity,
                actual: exponents.len(),
            });
        }
        let metadata = exponent_metadata(
            exponents,
            &self.fingerprint_weights,
            self.mask_variables,
            self.mask_bits_per_variable,
        )?;
        self.intern_with_metadata(exponents, metadata)
    }

    fn intern_with_metadata(
        &mut self,
        exponents: &[u32],
        metadata: MonomialMetadata,
    ) -> Result<MonomialId, MonomialArenaError> {
        if let Some(id) = self.find_id(metadata.fingerprint, exponents) {
            return Ok(id);
        }
        if self.metadata.len() == u32::MAX as usize {
            return Err(MonomialArenaError::CapacityExceeded);
        }
        if self.ids.needs_growth(self.metadata.len() + 1) {
            self.ids.rebuild(&self.metadata);
        }
        let raw =
            u32::try_from(self.metadata.len()).map_err(|_| MonomialArenaError::CapacityExceeded)?;
        let id = MonomialId(raw);
        self.exponents.append(exponents);
        self.metadata.push(metadata);
        self.ids
            .insert_vacant(self.metadata[id.index()].fingerprint, id);
        Ok(id)
    }

    fn find_id(&self, fingerprint: u64, exponents: &[u32]) -> Option<MonomialId> {
        let mask = self.ids.slots.len() - 1;
        let mut slot = fingerprint_slot(fingerprint) & mask;
        loop {
            let encoded = self.ids.slots[slot];
            if encoded == 0 {
                return None;
            }
            let id = MonomialId(encoded - 1);
            if self.metadata[id.index()].fingerprint == fingerprint
                && self.exponents_equal(id, exponents)
            {
                return Some(id);
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Materialize one standalone monomial.
    pub fn get(&self, id: MonomialId) -> Monomial {
        self.require_id(id);
        Monomial::new(
            (0..self.arity)
                .map(|variable| self.exponent_unchecked(id, variable))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }

    /// Exponent of one variable in an interned monomial.
    pub fn exponent(&self, id: MonomialId, variable: usize) -> u32 {
        self.require_id(id);
        assert!(variable < self.arity, "monomial variable out of range");
        self.exponent_unchecked(id, variable)
    }

    /// Cached total degree of one interned monomial.
    pub fn total_degree(&self, id: MonomialId) -> u64 {
        self.require_id(id);
        self.metadata[id.index()].degree
    }

    /// Cached short exponent-threshold mask for divisibility rejection.
    ///
    /// Mask containment is necessary but not sufficient for divisibility;
    /// exact exponent comparison remains authoritative.
    pub fn divisor_mask(&self, id: MonomialId) -> u64 {
        self.require_id(id);
        self.metadata[id.index()].divisor_mask
    }

    /// Cached exact-exponent fingerprint of one interned monomial.
    ///
    /// Equal fingerprints are only a candidate equality: callers must still
    /// compare exact exponents, as the arena itself does while interning.
    pub fn fingerprint(&self, id: MonomialId) -> u64 {
        self.require_id(id);
        self.metadata[id.index()].fingerprint
    }

    /// Whether the arena currently stores all exponents in `u16` words.
    pub fn is_packed_u16(&self) -> bool {
        self.exponents.is_packed_u16()
    }

    /// Bytes owned by the arena's allocated exponent, metadata, hash-table,
    /// fingerprint, and scratch buffers.
    ///
    /// This reports Rust-owned buffer capacity rather than only live entries,
    /// so callers can account for retained workspace after a large round. It
    /// deliberately excludes the fixed-size arena value itself.
    pub fn allocated_bytes(&self) -> usize {
        self.exponents
            .allocated_bytes()
            .saturating_add(
                self.metadata
                    .capacity()
                    .saturating_mul(core::mem::size_of::<MonomialMetadata>()),
            )
            .saturating_add(
                self.ids
                    .slots
                    .capacity()
                    .saturating_mul(core::mem::size_of::<u32>()),
            )
            .saturating_add(
                self.fingerprint_weights
                    .len()
                    .saturating_mul(core::mem::size_of::<u64>()),
            )
            .saturating_add(
                self.scratch
                    .capacity()
                    .saturating_mul(core::mem::size_of::<u32>()),
            )
    }

    /// Test exact divisibility using a short-mask rejection first.
    pub fn divides(&self, divisor: MonomialId, dividend: MonomialId) -> bool {
        self.require_id(divisor);
        self.require_id(dividend);
        let divisor_mask = self.metadata[divisor.index()].divisor_mask;
        let dividend_mask = self.metadata[dividend.index()].divisor_mask;
        self.metadata[divisor.index()].degree <= self.metadata[dividend.index()].degree
            && divisor_mask & !dividend_mask == 0
            && (0..self.arity).all(|variable| {
                self.exponent_unchecked(divisor, variable)
                    <= self.exponent_unchecked(dividend, variable)
            })
    }

    /// Intern the product of two existing monomials.
    pub fn product(
        &mut self,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<MonomialId, MonomialArenaError> {
        self.require_id(left);
        self.require_id(right);
        let Some(degree) = self.metadata[left.index()]
            .degree
            .checked_add(self.metadata[right.index()].degree)
        else {
            return Err(MonomialArenaError::ExponentOverflow);
        };
        let fingerprint = self.metadata[left.index()]
            .fingerprint
            .wrapping_add(self.metadata[right.index()].fingerprint);
        if let Some(id) = self.find_product(fingerprint, left, right)? {
            return Ok(id);
        }

        let mut exponents = core::mem::take(&mut self.scratch);
        exponents.clear();
        let left_offset = left.index() * self.arity;
        let right_offset = right.index() * self.arity;
        let mut divisor_mask = 0_u64;
        match &self.exponents {
            ArenaExponents::U16(values) => {
                for variable in 0..self.arity {
                    let exponent = u32::from(values[left_offset + variable])
                        + u32::from(values[right_offset + variable]);
                    divisor_mask |= exponent_divisor_bits(
                        variable,
                        exponent,
                        self.mask_variables,
                        self.mask_bits_per_variable,
                    );
                    exponents.push(exponent);
                }
            }
            ArenaExponents::U32(values) => {
                for variable in 0..self.arity {
                    let Some(exponent) =
                        values[left_offset + variable].checked_add(values[right_offset + variable])
                    else {
                        self.scratch = exponents;
                        return Err(MonomialArenaError::ExponentOverflow);
                    };
                    divisor_mask |= exponent_divisor_bits(
                        variable,
                        exponent,
                        self.mask_variables,
                        self.mask_bits_per_variable,
                    );
                    exponents.push(exponent);
                }
            }
        }
        let metadata = MonomialMetadata {
            degree,
            fingerprint,
            divisor_mask,
        };
        let result = self.intern_with_metadata(&exponents, metadata);
        self.scratch = exponents;
        result
    }

    fn find_product(
        &self,
        fingerprint: u64,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<Option<MonomialId>, MonomialArenaError> {
        let mask = self.ids.slots.len() - 1;
        let mut slot = fingerprint_slot(fingerprint) & mask;
        loop {
            let encoded = self.ids.slots[slot];
            if encoded == 0 {
                return Ok(None);
            }
            let candidate = MonomialId(encoded - 1);
            if self.metadata[candidate.index()].fingerprint == fingerprint
                && self.product_exponents_equal(candidate, left, right)?
            {
                return Ok(Some(candidate));
            }
            slot = (slot + 1) & mask;
        }
    }

    fn product_exponents_equal(
        &self,
        candidate: MonomialId,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<bool, MonomialArenaError> {
        let candidate_offset = candidate.index() * self.arity;
        let left_offset = left.index() * self.arity;
        let right_offset = right.index() * self.arity;
        match &self.exponents {
            ArenaExponents::U16(values) => Ok((0..self.arity).all(|variable| {
                u32::from(values[candidate_offset + variable])
                    == u32::from(values[left_offset + variable])
                        + u32::from(values[right_offset + variable])
            })),
            ArenaExponents::U32(values) => {
                for variable in 0..self.arity {
                    let Some(product) =
                        values[left_offset + variable].checked_add(values[right_offset + variable])
                    else {
                        return Err(MonomialArenaError::ExponentOverflow);
                    };
                    if values[candidate_offset + variable] != product {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    /// Intern the least common multiple of two existing monomials.
    pub fn lcm(
        &mut self,
        left: MonomialId,
        right: MonomialId,
    ) -> Result<MonomialId, MonomialArenaError> {
        self.require_id(left);
        self.require_id(right);
        let mut exponents = core::mem::take(&mut self.scratch);
        exponents.clear();
        let left_offset = left.index() * self.arity;
        let right_offset = right.index() * self.arity;
        let mut degree = 0_u64;
        let mut fingerprint = 0_u64;
        let mut divisor_mask = 0_u64;
        for variable in 0..self.arity {
            let exponent = match &self.exponents {
                ArenaExponents::U16(values) => {
                    u32::from(values[left_offset + variable].max(values[right_offset + variable]))
                }
                ArenaExponents::U32(values) => {
                    values[left_offset + variable].max(values[right_offset + variable])
                }
            };
            let Some(next_degree) = degree.checked_add(u64::from(exponent)) else {
                self.scratch = exponents;
                return Err(MonomialArenaError::ExponentOverflow);
            };
            degree = next_degree;
            fingerprint = fingerprint
                .wrapping_add(self.fingerprint_weights[variable].wrapping_mul(u64::from(exponent)));
            divisor_mask |= exponent_divisor_bits(
                variable,
                exponent,
                self.mask_variables,
                self.mask_bits_per_variable,
            );
            exponents.push(exponent);
        }
        let result = self.intern_with_metadata(
            &exponents,
            MonomialMetadata {
                degree,
                fingerprint,
                divisor_mask,
            },
        );
        self.scratch = exponents;
        result
    }

    /// Intern `dividend / divisor` when exact monomial divisibility holds.
    pub fn quotient(
        &mut self,
        dividend: MonomialId,
        divisor: MonomialId,
    ) -> Result<Option<MonomialId>, MonomialArenaError> {
        if !self.divides(divisor, dividend) {
            return Ok(None);
        }
        let mut exponents = core::mem::take(&mut self.scratch);
        exponents.clear();
        let dividend_offset = dividend.index() * self.arity;
        let divisor_offset = divisor.index() * self.arity;
        let mut divisor_mask = 0_u64;
        for variable in 0..self.arity {
            let exponent = match &self.exponents {
                ArenaExponents::U16(values) => u32::from(
                    values[dividend_offset + variable] - values[divisor_offset + variable],
                ),
                ArenaExponents::U32(values) => {
                    values[dividend_offset + variable] - values[divisor_offset + variable]
                }
            };
            divisor_mask |= exponent_divisor_bits(
                variable,
                exponent,
                self.mask_variables,
                self.mask_bits_per_variable,
            );
            exponents.push(exponent);
        }
        let metadata = MonomialMetadata {
            degree: self.metadata[dividend.index()].degree - self.metadata[divisor.index()].degree,
            fingerprint: self.metadata[dividend.index()]
                .fingerprint
                .wrapping_sub(self.metadata[divisor.index()].fingerprint),
            divisor_mask,
        };
        let result = self.intern_with_metadata(&exponents, metadata).map(Some);
        self.scratch = exponents;
        result
    }

    /// Compare two identifiers in the arena's order.
    pub fn compare(&self, left: MonomialId, right: MonomialId) -> Ordering {
        self.require_id(left);
        self.require_id(right);
        match self.order {
            MonomialOrder::Lexicographic => self.lex_compare(left, right),
            MonomialOrder::GradedLexicographic => self.metadata[left.index()]
                .degree
                .cmp(&self.metadata[right.index()].degree)
                .then_with(|| self.lex_compare(left, right)),
            MonomialOrder::GradedReverseLexicographic => {
                let degree = self.metadata[left.index()]
                    .degree
                    .cmp(&self.metadata[right.index()].degree);
                if degree != Ordering::Equal {
                    return degree;
                }
                for variable in (0..self.arity).rev() {
                    match self
                        .exponent_unchecked(left, variable)
                        .cmp(&self.exponent_unchecked(right, variable))
                    {
                        Ordering::Equal => {}
                        Ordering::Less => return Ordering::Greater,
                        Ordering::Greater => return Ordering::Less,
                    }
                }
                Ordering::Equal
            }
        }
    }

    fn lex_compare(&self, left: MonomialId, right: MonomialId) -> Ordering {
        (0..self.arity)
            .map(|variable| {
                self.exponent_unchecked(left, variable)
                    .cmp(&self.exponent_unchecked(right, variable))
            })
            .find(|ordering| *ordering != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    }

    fn exponents_equal(&self, id: MonomialId, exponents: &[u32]) -> bool {
        let offset = id.index() * self.arity;
        match &self.exponents {
            ArenaExponents::U16(values) => values[offset..offset + self.arity]
                .iter()
                .zip(exponents)
                .all(|(&left, &right)| u32::from(left) == right),
            ArenaExponents::U32(values) => values[offset..offset + self.arity] == *exponents,
        }
    }

    #[inline]
    fn exponent_unchecked(&self, id: MonomialId, variable: usize) -> u32 {
        self.exponents.exponent(id.index() * self.arity + variable)
    }

    fn require_id(&self, id: MonomialId) {
        assert!(
            id.index() < self.metadata.len(),
            "monomial identifier out of range"
        );
    }
}

fn exponent_metadata(
    exponents: &[u32],
    fingerprint_weights: &[u64],
    mask_variables: usize,
    mask_bits_per_variable: usize,
) -> Result<MonomialMetadata, MonomialArenaError> {
    let mut fingerprint = 0_u64;
    let mut degree = 0_u64;
    let mut divisor_mask = 0_u64;

    for (variable, &exponent) in exponents.iter().enumerate() {
        degree = degree
            .checked_add(u64::from(exponent))
            .ok_or(MonomialArenaError::ExponentOverflow)?;
        fingerprint = fingerprint
            .wrapping_add(fingerprint_weights[variable].wrapping_mul(u64::from(exponent)));
        divisor_mask |=
            exponent_divisor_bits(variable, exponent, mask_variables, mask_bits_per_variable);
    }

    Ok(MonomialMetadata {
        degree,
        fingerprint,
        divisor_mask,
    })
}

fn exponent_divisor_bits(
    variable: usize,
    exponent: u32,
    mask_variables: usize,
    bits_per_variable: usize,
) -> u64 {
    if variable >= mask_variables {
        return 0;
    }
    let occupied = usize::try_from(exponent)
        .unwrap_or(usize::MAX)
        .min(bits_per_variable);
    let thresholds = if occupied == 64 {
        u64::MAX
    } else {
        (1_u64 << occupied) - 1
    };
    thresholds << (variable * bits_per_variable)
}

fn variable_fingerprint(variable: usize) -> u64 {
    let mut value = (variable as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (value ^ (value >> 31)) | 1
}

fn fingerprint_slot(mut fingerprint: u64) -> usize {
    fingerprint = (fingerprint ^ (fingerprint >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    fingerprint = (fingerprint ^ (fingerprint >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (fingerprint ^ (fingerprint >> 31)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_and_arithmetic_are_exact() {
        let a = Monomial::new(vec![2, 1, 0].into_boxed_slice());
        let b = Monomial::new(vec![0, 3, 1].into_boxed_slice());
        assert!(a.is_packed_u16());
        assert_eq!(a.checked_product(&b).unwrap().to_u32_vec(), [2, 4, 1]);
        assert_eq!(a.lcm(&b).unwrap().to_u32_vec(), [2, 3, 1]);
        assert_eq!(a.checked_product(&b).unwrap().quotient(&a).unwrap(), b);
    }

    #[test]
    fn arena_ids_are_stable() {
        let mut arena = MonomialArena::new(2, MonomialOrder::GradedReverseLexicographic);
        let value = Monomial::new(vec![1, 2].into_boxed_slice());
        let first = arena.intern(value.clone()).unwrap();
        assert_eq!(arena.intern(value).unwrap(), first);
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn arena_forks_preserve_existing_ids_with_independent_identity_and_growth() {
        let mut arena = MonomialArena::new(2, MonomialOrder::GradedReverseLexicographic);
        let x = arena.intern_exponents(&[1, 0]).unwrap();
        let mut fork = arena.fork();

        assert_ne!(arena.identity(), fork.identity());
        assert_eq!(fork.get(x), arena.get(x));
        assert_eq!(fork.intern_exponents(&[1, 0]).unwrap(), x);
        fork.intern_exponents(&[0, 1]).unwrap();
        assert_eq!(arena.len(), 1);
        assert_eq!(fork.len(), 2);
    }

    #[test]
    fn arena_table_growth_preserves_exact_ids() {
        let mut arena =
            MonomialArena::with_capacity(3, MonomialOrder::GradedReverseLexicographic, 1);
        let expected = (0..2_048_u32)
            .map(|index| {
                let exponents = [index, index * 3, index * 7];
                (exponents, arena.intern_exponents(&exponents).unwrap())
            })
            .collect::<Vec<_>>();

        assert_eq!(arena.len(), expected.len());
        for (exponents, id) in expected {
            assert_eq!(arena.intern_exponents(&exponents).unwrap(), id);
        }
        assert_eq!(arena.len(), 2_048);
    }

    #[test]
    fn arena_product_finds_an_existing_exact_sum() {
        let mut arena = MonomialArena::new(3, MonomialOrder::GradedReverseLexicographic);
        let left = arena.intern_exponents(&[12, 4, 7]).unwrap();
        let right = arena.intern_exponents(&[3, 10, 2]).unwrap();
        let expected = arena.intern_exponents(&[15, 14, 9]).unwrap();
        let len = arena.len();

        for _ in 0..100 {
            assert_eq!(arena.product(left, right).unwrap(), expected);
            assert_eq!(arena.product(right, left).unwrap(), expected);
        }
        assert_eq!(arena.len(), len);
        assert_eq!(
            arena.fingerprint(expected),
            arena
                .fingerprint(left)
                .wrapping_add(arena.fingerprint(right))
        );
    }

    #[test]
    fn arena_product_rejects_u32_exponent_overflow() {
        let mut arena = MonomialArena::new(2, MonomialOrder::Lexicographic);
        let largest = arena.intern_exponents(&[u32::MAX, 0]).unwrap();
        let one = arena.intern_exponents(&[1, 0]).unwrap();

        assert_eq!(
            arena.product(largest, one),
            Err(MonomialArenaError::ExponentOverflow)
        );
    }

    #[test]
    fn arena_upgrade_preserves_ids_and_metadata() {
        let mut arena = MonomialArena::new(3, MonomialOrder::GradedReverseLexicographic);
        let initial_bytes = arena.allocated_bytes();
        let small = arena.intern_exponents(&[2, 0, 3]).unwrap();
        assert!(arena.is_packed_u16());
        let packed_bytes = arena.allocated_bytes();
        assert!(packed_bytes > initial_bytes);
        let large = arena.intern_exponents(&[70_000, 1, 0]).unwrap();
        assert!(!arena.is_packed_u16());
        assert!(arena.allocated_bytes() >= packed_bytes);
        assert_eq!(arena.get(small).to_u32_vec(), [2, 0, 3]);
        assert_eq!(arena.total_degree(small), 5);
        assert_eq!(arena.get(large).to_u32_vec(), [70_000, 1, 0]);
        assert_eq!(arena.intern_exponents(&[2, 0, 3]).unwrap(), small);
    }

    #[test]
    fn arena_operations_match_standalone_monomials() {
        for order in [
            MonomialOrder::Lexicographic,
            MonomialOrder::GradedLexicographic,
            MonomialOrder::GradedReverseLexicographic,
        ] {
            let mut arena = MonomialArena::new(3, order);
            let left_value = Monomial::new(vec![2, 1, 0].into_boxed_slice());
            let right_value = Monomial::new(vec![0, 3, 1].into_boxed_slice());
            let left = arena.intern(left_value.clone()).unwrap();
            let right = arena.intern(right_value.clone()).unwrap();
            let product = arena.product(left, right).unwrap();
            let lcm = arena.lcm(left, right).unwrap();
            assert_eq!(
                arena.get(product),
                left_value.checked_product(&right_value).unwrap()
            );
            assert_eq!(arena.get(lcm), left_value.lcm(&right_value).unwrap());
            assert_eq!(arena.quotient(product, left).unwrap(), Some(right));
            assert_eq!(
                arena.compare(left, right),
                left_value.compare(&right_value, order)
            );
        }
    }

    #[test]
    fn generation_marks_clear_without_losing_new_arena_ids() {
        let mut arena = MonomialArena::new(1, MonomialOrder::Lexicographic);
        let one = arena.intern_exponents(&[0]).unwrap();
        let mut marks = MonomialMarks::new(&arena);
        assert!(marks.insert(&arena, one));
        assert!(!marks.insert(&arena, one));
        marks.clear(&arena);
        assert!(!marks.contains(&arena, one));
        let x = arena.intern_exponents(&[1]).unwrap();
        assert!(marks.insert(&arena, x));
        assert!(marks.contains(&arena, x));
    }

    #[test]
    fn short_masks_never_reject_exact_divisibility() {
        let mut arena = MonomialArena::new(3, MonomialOrder::Lexicographic);
        let mut values = Vec::new();
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..5 {
                    let monomial = Monomial::new(vec![x, y, z].into_boxed_slice());
                    let id = arena.intern(monomial.clone()).unwrap();
                    values.push((monomial, id));
                }
            }
        }
        for (left, left_id) in &values {
            for (right, right_id) in &values {
                assert_eq!(arena.divides(*left_id, *right_id), left.divides(right));
            }
        }
    }
}
