// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Shared test helpers — types and harnesses that more than one
//! integration-test file uses. Each module is a small, focused
//! utility that doesn't belong in `src/` (because it's a
//! reference / comparison shape, not part of the kernel's surface).

pub mod arb_mat_ffi;
pub mod csr_f64;
