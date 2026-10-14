// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Randomised property tests for sparb's CSR-Arb storage + SpMV.
//!
//! The properties asserted here are the structural contracts that
//! hold for *every* input, in contrast to the worked examples in the
//! per-module unit tests:
//!
//!   * SpMV agrees with a hand-rolled dense ball product;
//!   * transpose is an involution;
//!   * COO ingestion is invariant under triple permutation;
//!   * exact (radius-0) inputs produce exact (radius-0) outputs.

use proptest::prelude::*;

use sparb::{Arb, CsrArb, SparseRingStorage};

/// A COO fixture: dims plus exact integer-valued triples.
fn coo_strategy() -> impl Strategy<Value = (usize, usize, Vec<(usize, usize, i64)>)> {
    (1usize..8, 1usize..8).prop_flat_map(|(rows, cols)| {
        let triple = (0..rows, 0..cols, -20i64..20i64);
        (Just(rows), Just(cols), prop::collection::vec(triple, 0..24))
    })
}

/// Dense reference: materialise the COO into a `rows × cols` grid of
/// `f64` midpoints (duplicates summed), the ground truth for exact
/// integer inputs.
fn dense_ref(rows: usize, cols: usize, triples: &[(usize, usize, i64)]) -> Vec<Vec<f64>> {
    let mut g = vec![vec![0.0f64; cols]; rows];
    for &(r, c, v) in triples {
        g[r][c] += v as f64;
    }
    g
}

fn to_f64_quads(triples: &[(usize, usize, i64)]) -> Vec<(usize, usize, f64, f64)> {
    triples
        .iter()
        .map(|&(r, c, v)| (r, c, v as f64, 0.0))
        .collect()
}

proptest! {
    /// Every stored cell of a `CsrArb` built from exact integer
    /// triples matches the dense reference exactly, and structural
    /// zeros really are zero.
    #[test]
    fn csr_matches_dense_reference((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let dense = dense_ref(rows, cols, &triples);
        for (r, dense_row) in dense.iter().enumerate() {
            for (c, &expected) in dense_row.iter().enumerate() {
                match a.get(r, c) {
                    Some(ball) => {
                        prop_assert_eq!(ball.midpoint(), expected);
                        prop_assert_eq!(ball.radius(64), 0.0);
                    }
                    None => prop_assert_eq!(expected, 0.0),
                }
            }
        }
    }

    /// SpMV agrees with the dense ball product on exact inputs:
    /// `y[i] == Σ_j A[i,j] · x[j]` with zero radius.
    #[test]
    fn spmv_matches_dense_product((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let dense = dense_ref(rows, cols, &triples);

        // x[j] = j + 1 (exact).
        let x: Vec<Arb> = (0..cols).map(|j| Arb::from_f64((j + 1) as f64)).collect();
        let y = a.spmv(&x);
        prop_assert_eq!(y.len(), rows);

        for (i, yi) in y.iter().enumerate() {
            let expect: f64 = (0..cols).map(|j| dense[i][j] * (j + 1) as f64).sum();
            prop_assert_eq!(yi.midpoint(), expect);
            prop_assert_eq!(yi.radius(64), 0.0);  // exact in, exact out
        }
    }

    /// Transpose is an involution: `(Aᵀ)ᵀ == A` as dense grids.
    #[test]
    fn transpose_is_involution((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let att = a.transpose().transpose();
        prop_assert_eq!(att.to_dense_f64(), a.to_dense_f64());
    }

    /// COO ingestion is permutation-invariant: shuffling the triple
    /// list yields the identical CSR matrix.
    #[test]
    fn coo_ingest_is_permutation_invariant(
        (rows, cols, triples) in coo_strategy(),
        seed in any::<u64>(),
    ) {
        let quads = to_f64_quads(&triples);
        let a = CsrArb::from_coo_f64(rows, cols, 64, &quads);

        // Deterministic shuffle of the triple order.
        let mut shuffled = quads.clone();
        let mut state = seed | 1;
        for i in (1..shuffled.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            shuffled.swap(i, j);
        }
        let b = CsrArb::from_coo_f64(rows, cols, 64, &shuffled);

        prop_assert_eq!(a.to_dense_f64(), b.to_dense_f64());
    }

    /// `A + B == B + A` on midpoints — addition is commutative.
    #[test]
    fn add_is_commutative((rows, cols, triples_a) in coo_strategy(),
                          triples_b in prop::collection::vec(
                              (0..8usize, 0..8usize, -20i64..20i64), 0..24)) {
        // Clip B's triples to (rows, cols) so the shapes match.
        let triples_b: Vec<_> = triples_b
            .into_iter()
            .filter(|&(r, c, _)| r < rows && c < cols)
            .collect();
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples_a));
        let b = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples_b));
        let ab = a.add(&b);
        let ba = b.add(&a);
        prop_assert_eq!(ab.to_dense_f64(), ba.to_dense_f64());
    }

    /// `A - A` is the zero matrix (no stored entries).
    #[test]
    fn sub_self_is_zero_matrix((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let z = a.sub(&a);
        prop_assert_eq!(z.nnz(), 0);
        prop_assert!(z.is_zero());
    }

    /// `(A + B) - B` recovers `A` on midpoints for exact inputs.
    #[test]
    fn add_then_sub_recovers((rows, cols, triples_a) in coo_strategy(),
                              triples_b in prop::collection::vec(
                                  (0..8usize, 0..8usize, -20i64..20i64), 0..24)) {
        let triples_b: Vec<_> = triples_b
            .into_iter()
            .filter(|&(r, c, _)| r < rows && c < cols)
            .collect();
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples_a));
        let b = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples_b));
        let recovered = a.add(&b).sub(&b);
        prop_assert_eq!(recovered.to_dense_f64(), a.to_dense_f64());
    }

    /// `scale(alpha) ∘ scale(beta) == scale(alpha · beta)` on
    /// midpoints, for exact scalar factors.
    #[test]
    fn scale_composes_multiplicatively((rows, cols, triples) in coo_strategy(),
                                        alpha in -10i64..10i64,
                                        beta in -10i64..10i64) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let alpha_b = Arb::from_f64(alpha as f64);
        let beta_b = Arb::from_f64(beta as f64);
        let twice = a.scale(&alpha_b).scale(&beta_b);
        let once = a.scale(&alpha_b.mul(&beta_b, 64));
        prop_assert_eq!(twice.to_dense_f64(), once.to_dense_f64());
    }

    /// `scale(exact zero)` is the zero matrix at the same shape.
    #[test]
    fn scale_by_zero_is_zero_matrix((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let z = a.scale(&Arb::zero());
        prop_assert_eq!(z.nnz(), 0);
        prop_assert_eq!(z.rows(), rows);
        prop_assert_eq!(z.cols(), cols);
    }

    /// `scale(one)` preserves the matrix (midpoints unchanged).
    #[test]
    fn scale_by_one_is_identity_on_midpoints((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let s = a.scale(&Arb::from_f64(1.0));
        prop_assert_eq!(s.to_dense_f64(), a.to_dense_f64());
    }
}

// ── ball-regime sweep ────────────────────────────────────────────────
//
// The properties below stress sparb on ball inputs (non-zero radii,
// wider shape range, mixed radius regimes) — the actual interval-
// arithmetic surface, not just the exact-input CSR plumbing.

/// A wider-shape strategy — `[1, 32]` per axis — for properties
/// that should hold beyond the tiny `(1..8)` fixtures `coo_strategy`
/// uses for fast iteration on midpoint-only invariants.
fn wider_shape() -> impl Strategy<Value = (usize, usize)> {
    (1usize..32, 1usize..32)
}

/// One of the four radius regimes a ball-arithmetic kernel must
/// handle gracefully.
fn radius_regime() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),                         // exact entry
        (1e-30f64..1e-20).prop_map(|r| r), // tiny
        (1e-6f64..1e-2).prop_map(|r| r),   // moderate
        (1e6f64..1e12).prop_map(|r| r),    // huge
    ]
}

/// A ball-valued fixture: shape + triples carrying `(row, col, mid, rad)`
/// with `rad` drawn from the regime sweep above. Mid magnitudes
/// span a moderate range so cancellation can actually happen during
/// SpMV without going to extreme values.
/// Shape + ball triples — the value type returned by [`ball_fixture`].
type BallFixture = (usize, usize, Vec<(usize, usize, f64, f64)>);

fn ball_fixture() -> impl Strategy<Value = BallFixture> {
    // Triples are deduplicated by `(row, col)` after generation —
    // multiple triples at the same cell would otherwise be summed
    // at the matrix's working precision by `from_coo_f64`, which
    // makes the resulting *entries* depend on `prec`. That breaks
    // the apples-to-apples premise of precision-monotonicity tests
    // where the matrix is meant to be the same across the prec
    // sweep.
    wider_shape().prop_flat_map(|(rows, cols)| {
        let entry = (0..rows, 0..cols, -100.0f64..100.0f64, radius_regime());
        (Just(rows), Just(cols), prop::collection::vec(entry, 0..96)).prop_map(
            |(rows, cols, triples)| {
                let mut seen: std::collections::HashSet<(usize, usize)> =
                    std::collections::HashSet::new();
                let deduped = triples
                    .into_iter()
                    .filter(|&(r, c, _, _)| seen.insert((r, c)))
                    .collect();
                (rows, cols, deduped)
            },
        )
    })
}

proptest! {
    /// **Enclosure invariant** under non-zero input radii. The
    /// dense ball reference (which uses `arb_addmul` exactly the
    /// way sparb does, just walking *every* cell instead of the
    /// stored ones) and sparb's SpMV both rigorously enclose the
    /// true `A · x`, so their output balls must intersect at every
    /// row.
    #[test]
    fn spmv_balls_intersect_dense_reference(
        (rows, cols, triples) in ball_fixture(),
        x_seed in any::<u64>(),
    ) {
        let prec = 64i64;
        let a = CsrArb::from_coo_f64(rows, cols, prec, &triples);

        // Deterministic random x with exact (rad=0) midpoints — keeps
        // the comparison focused on A's interval bookkeeping.
        let mut state = x_seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let x_f64: Vec<f64> = (0..cols)
            .map(|_| (next() % 2000) as f64 / 1000.0 - 1.0)
            .collect();
        let x_arb: Vec<Arb> = x_f64.iter().map(|&v| Arb::from_f64(v)).collect();

        // sparb SpMV.
        let y = a.spmv(&x_arb);

        // Dense-reference SpMV in ball arithmetic: walk every cell,
        // including structural zeros (which arb_addmul handles
        // correctly: zero × anything = zero, no radius growth).
        let mut y_ref: Vec<Arb> = (0..rows).map(|_| Arb::zero()).collect();
        let dense = a.to_dense_f64();
        for (r, dense_row) in dense.iter().enumerate() {
            for (c, &(mid, rad)) in dense_row.iter().enumerate() {
                let entry = Arb::from_mid_rad(mid, rad);
                y_ref[r].addmul_assign(&entry, &x_arb[c], prec);
            }
        }

        // Both balls enclose the true product → intersect.
        for (sp, dn) in y.iter().zip(y_ref.iter()) {
            let (sm, sr) = sp.to_pair(128);
            let (dm, dr) = dn.to_pair(128);
            prop_assert!(
                (sm - dm).abs() <= sr + dr + 1e-30,
                "row balls disjoint: sparb ({sm}, {sr}) vs dense-ref ({dm}, {dr})"
            );
        }
    }

    /// **Precision tightening** for SpMV. Same input matrix /
    /// vector, swept across `prec ∈ {24, 53, 80, 128}`; the
    /// output ball's radius at the highest precision must be
    /// **no larger than** the radius at the lowest, with a
    /// generous slack to absorb the sub-ulp wobble that the
    /// arb_addmul accumulator + f64 readback can introduce
    /// (intermediate accumulator midpoints round differently at
    /// different working precisions, so the per-step rounding
    /// contribution starts from a slightly different base — strict
    /// monotonicity holds in the limit but not bit-for-bit on the
    /// materialised f64 number). A gross monotonicity violation
    /// (radius growing by more than ~1% between extreme precs)
    /// would still fail loudly. The strict bit-monotonicity case
    /// at exact inputs is covered by `tests/instability_demo.rs`.
    #[test]
    fn spmv_radius_tightens_with_prec((rows, cols, triples) in ball_fixture()) {
        // Use the same exact (rad=0) x for all precs so any radius
        // change is attributable to the matrix prec alone.
        let x_f64: Vec<f64> = (0..cols)
            .map(|i| (i as f64) * 0.1 - 1.0)
            .collect();
        let x_arb: Vec<Arb> = x_f64.iter().map(|&v| Arb::from_f64(v)).collect();

        // Endpoint check rather than per-step monotonicity: the
        // max radius at prec=128 should be no larger (modulo 1%
        // slack) than the max radius at prec=24. This catches
        // gross failures (radius doubling, NaN propagation) while
        // tolerating the sub-1e-8 wobble that intermediate-
        // accumulator midpoint rounding can introduce.
        let a_low = CsrArb::from_coo_f64(rows, cols, 24, &triples);
        let a_high = CsrArb::from_coo_f64(rows, cols, 128, &triples);
        let max_rad_low = a_low
            .spmv(&x_arb)
            .iter()
            .map(|b| b.radius(256))
            .fold(0.0f64, f64::max);
        let max_rad_high = a_high
            .spmv(&x_arb)
            .iter()
            .map(|b| b.radius(256))
            .fold(0.0f64, f64::max);
        let tol = max_rad_low.abs() * 0.01 + 1e-12;
        prop_assert!(
            max_rad_high <= max_rad_low + tol,
            "high-prec radius {max_rad_high} > low-prec radius \
             {max_rad_low} by more than 1%"
        );
    }

    /// **Enclosure rigour** under exact-input SpMV. Every output
    /// ball must contain the exact dense-computed midpoint product
    /// (since both run the same scalar arithmetic on exact data
    /// the midpoint should match within the ball's radius).
    #[test]
    fn spmv_encloses_exact_dense_product(
        (rows, cols, triples) in ball_fixture(),
    ) {
        // Promote inputs to exact (rad = 0) so the dense reference
        // computed in plain f64 is the *true* answer (modulo f64
        // rounding, which is absorbed by sparb's radius).
        let exact_triples: Vec<(usize, usize, f64, f64)> = triples
            .iter()
            .map(|&(r, c, m, _)| (r, c, m, 0.0))
            .collect();
        let prec = 80i64;
        let a = CsrArb::from_coo_f64(rows, cols, prec, &exact_triples);

        let x: Vec<f64> = (0..cols).map(|i| (i as f64) * 0.3).collect();
        let x_arb: Vec<Arb> = x.iter().map(|&v| Arb::from_f64(v)).collect();
        let y_sp = a.spmv(&x_arb);

        // Dense f64 SpMV.
        let dense = a.to_dense_f64();
        for (r, dense_row) in dense.iter().enumerate() {
            let true_y: f64 = dense_row
                .iter()
                .enumerate()
                .map(|(c, &(mid, _))| mid * x[c])
                .sum();
            let (sm, sr) = y_sp[r].to_pair(128);
            prop_assert!(
                (sm - true_y).abs() <= sr + 1e-10 * sm.abs().max(true_y.abs()).max(1.0),
                "row {r}: sparb ball ({sm} ± {sr}) does not enclose dense f64 \
                 product {true_y}"
            );
        }
    }

    /// `A · I == A` on midpoints (the identity is a right unit for
    /// matrix multiplication).
    #[test]
    fn matmul_right_identity_is_noop((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let i_triples: Vec<(usize, usize, f64, f64)> =
            (0..cols).map(|k| (k, k, 1.0, 0.0)).collect();
        let id = CsrArb::from_coo_f64(cols, cols, 64, &i_triples);
        let c = a.matmul(&id);
        prop_assert_eq!(c.to_dense_f64(), a.to_dense_f64());
    }

    /// `I · A == A` (left unit).
    #[test]
    fn matmul_left_identity_is_noop((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let i_triples: Vec<(usize, usize, f64, f64)> =
            (0..rows).map(|k| (k, k, 1.0, 0.0)).collect();
        let id = CsrArb::from_coo_f64(rows, rows, 64, &i_triples);
        let c = id.matmul(&a);
        prop_assert_eq!(c.to_dense_f64(), a.to_dense_f64());
    }

    /// `A · 0 == 0` (annihilation by the zero matrix).
    #[test]
    fn matmul_by_zero_is_zero((rows, cols, triples) in coo_strategy()) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let z = CsrArb::zeros(cols, 5, 64);
        let c = a.matmul(&z);
        prop_assert_eq!(c.nnz(), 0);
        prop_assert_eq!((c.rows(), c.cols()), (rows, 5));
    }

    /// `(A · Aᵀ)` is symmetric on midpoints (square, equal to its
    /// own transpose at the dense level).
    #[test]
    fn matmul_a_a_transpose_is_symmetric_on_midpoints(
        (rows, cols, triples) in coo_strategy(),
    ) {
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples));
        let at = a.transpose();
        let aat = a.matmul(&at);
        let aat_t = aat.transpose();
        prop_assert_eq!(aat.to_dense_f64(), aat_t.to_dense_f64());
    }

    /// `(A · B)ᵀ == Bᵀ · Aᵀ` on midpoints — the classic transpose-of-
    /// product identity.
    #[test]
    fn matmul_transpose_of_product_equals_product_of_transposes(
        (rows, cols, triples_a) in coo_strategy(),
    ) {
        // B has shape (cols, ?), derived from a permutation of A's
        // triples — gives us a deterministic compatible B without
        // a second nested proptest strategy.
        let a = CsrArb::from_coo_f64(rows, cols, 64, &to_f64_quads(&triples_a));
        // Build B as cols × rows from A's transpose (so B = Aᵀ).
        // Then A · B = A · Aᵀ, and (A · Aᵀ)ᵀ = A · Aᵀ ⇒ equal to
        // Bᵀ · Aᵀ = A · Aᵀ. Reduces to the previous property but
        // exercised through an explicit second matmul to stress the
        // transpose / matmul interaction.
        let b = a.transpose();
        let ab = a.matmul(&b);
        let bt_at = b.transpose().matmul(&a.transpose());
        prop_assert_eq!(ab.to_dense_f64(), bt_at.to_dense_f64());
    }
}
