// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! F4 and conservative signature-assisted Gröbner computation.

use core::cmp::Ordering;
use core::fmt;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ahash::AHashSet as HashSet;
use flint_core::Integer;

use poly_core::{
    ArenaPolynomial, MonomialArena, MonomialId, MonomialMarks, MonomialOrder,
    PolynomialAccumulator, PolynomialWorkspace, SparsePolynomial,
};

use crate::field::{BinaryExtensionField16, BinaryField, Field, PrimeField32};
use crate::reducer::{
    BinaryExtensionReducer, BinaryReducer, F4Reduction, F4RowWork, GenericReducer, PrimeReducer,
    RowReducer,
};

/// Field-basis implementation requested from the fixed kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Algorithm {
    /// Ordinary degree-batched F4.
    #[default]
    F4,
    /// F4 with conservative learned-signature syzygy pruning and verified F4
    /// fallback.
    SignatureF4,
}

/// Specialized row representation selected in the most recent F4 round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowBackend {
    /// Generic coefficient-bearing dense rows.
    GenericDense,
    /// Dense rational rows through FLINT's fraction-free matrix reduction.
    RationalFlint,
    /// Sparse rows over F₂ through `mzs`.
    BinarySparse,
    /// Dense bit-packed rows over F₂ through `m4rst`.
    BinaryDense,
    /// Dense bit-packed rows over `F_{2^e}` through `m4rste`.
    BinaryExtensionDense,
    /// Sparse word-prime rows through `sparla`.
    PrimeSparse,
    /// F4-specific sparse `A|B` and dense `C|D` word-prime reduction.
    PrimeAbCd,
    /// Dense word-prime rows with delayed modular arithmetic.
    PrimeDense,
}

/// Private rational-basis construction regime selected by kernel dispatch.
///
/// This is benchmark instrumentation, not mathematical identity and is not
/// projected through Noether's public facade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RationalBackend {
    /// Primitive integer polynomials with fraction-free pseudo-reduction.
    PrimitiveInteger,
    /// Prime images followed by CRT and rational reconstruction.
    Modular,
    /// Coefficient-generic direct arithmetic with exact rational values.
    DirectField,
}

/// Private rational change-matrix construction regime selected by dispatch.
///
/// This is benchmark instrumentation, not mathematical identity and is not
/// projected through Noether's public facade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RationalTransformationBackend {
    /// Primitive-integer basis arithmetic with rational change vectors.
    PrimitiveInteger,
    /// Word-prime tracked images followed by CRT and rational reconstruction.
    Modular,
    /// Coefficient-generic direct arithmetic with exact rational values.
    DirectField,
}

/// Explicit resource bounds for one kernel invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroebnerOptions {
    /// F4 or signature-assisted F4.
    pub algorithm: Algorithm,
    /// Maximum number of basis elements.
    pub max_basis_size: usize,
    /// Maximum number of simultaneously pending pairs.
    pub max_pending_pairs: usize,
    /// Maximum number of F4 degree rounds.
    pub max_rounds: usize,
    /// Maximum rows in one F4 matrix.
    pub max_matrix_rows: usize,
    /// Maximum columns in one F4 matrix.
    pub max_matrix_columns: usize,
    /// Maximum Rust-owned workspace bytes retained at an observed F4 phase.
    ///
    /// This cap is checked before row-reducer dense allocations and after
    /// symbolic construction. Coefficient values with context-owned native
    /// allocations can use additional memory not visible to the generic
    /// kernel.
    pub max_workspace_bytes: usize,
    /// Maximum selected pairs per round; zero means every pair of minimum
    /// degree.
    pub max_pairs_per_round: usize,
    /// Maximum polynomial-reduction steps in one normal form.
    pub max_reduction_steps: usize,
    /// Maximum usable prime images in one modular rational reconstruction.
    pub max_modular_primes: usize,
    /// Maximum usable prime images when reconstructing a polynomial change
    /// matrix, whose coefficients can be substantially larger than its
    /// reduced basis.
    pub max_transformation_primes: usize,
    /// Maximum quotient dimension admitted by classical FGLM.
    pub max_quotient_dimension: usize,
    /// Maximum target-order monomials visited by one FGLM conversion.
    pub max_fglm_candidates: usize,
    /// Maximum deterministic linear forms tested while constructing an RUR.
    pub max_rur_forms: usize,
    /// Optional exact standard-graded Hilbert numerator of the quotient.
    ///
    /// A matching leading ideal may stop pair construction early, but it is
    /// never accepted as correctness evidence: the candidate receives an
    /// independent Buchberger sweep and automatically falls back to ordinary
    /// F4 if that verification fails.
    pub hilbert_numerator_target: Option<Vec<Integer>>,
}

impl Default for GroebnerOptions {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::F4,
            max_basis_size: 16_384,
            max_pending_pairs: 4_000_000,
            max_rounds: 10_000,
            max_matrix_rows: 2_000_000,
            max_matrix_columns: 4_000_000,
            max_workspace_bytes: 2_147_483_648,
            max_pairs_per_round: 0,
            max_reduction_steps: 10_000_000,
            max_modular_primes: 16,
            max_transformation_primes: 64,
            max_quotient_dimension: 2_048,
            max_fglm_candidates: 1_000_000,
            max_rur_forms: 64,
            hilbert_numerator_target: None,
        }
    }
}

/// Phase and representation counters from a completed kernel invocation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GroebnerMetrics {
    /// F4 degree rounds.
    pub rounds: usize,
    /// Smallest sugar degree selected for an F4 round.
    pub minimum_round_degree: Option<u64>,
    /// Largest sugar degree selected for an F4 round.
    pub maximum_round_degree: Option<u64>,
    /// Sum of selected round sugar degrees, for an allocation-free mean.
    pub round_degree_sum: u128,
    /// Critical pairs constructed.
    pub pairs_generated: usize,
    /// Critical pairs selected into rounds.
    pub pairs_selected: usize,
    /// Pairs discarded by the relatively-prime leading-term criterion.
    pub product_criterion_pairs: usize,
    /// Pairs discarded by Buchberger's chain criterion after both supporting
    /// pairs were resolved.
    pub chain_criterion_pairs: usize,
    /// Pairs discarded by the Gebauer--Moeller installation criteria.
    pub gebauer_moeller_pairs: usize,
    /// Duplicate equal-LCM pairs removed during basis updates.
    pub equal_lcm_pairs: usize,
    /// Schreyer pairs omitted because an earlier pair generates a divisor of
    /// the same leading source-component colon monomial.
    pub schreyer_colon_pairs: usize,
    /// Pairs discarded by learned signature syzygies.
    pub signature_pruned_pairs: usize,
    /// Rows assembled across all rounds.
    pub matrix_rows: usize,
    /// Columns assembled across all rounds.
    pub matrix_columns: usize,
    /// Nonzero entries assembled across all rounds.
    pub matrix_nonzeros: usize,
    /// Largest row count in one round.
    pub peak_matrix_rows: usize,
    /// Largest column count in one round.
    pub peak_matrix_columns: usize,
    /// Largest nonzero count in one assembled matrix.
    pub peak_matrix_nonzeros: usize,
    /// Largest allocated monomial-arena footprint observed.
    pub peak_monomial_bytes: usize,
    /// Largest allocated critical-pair and resolved-pair footprint observed.
    pub peak_pair_bytes: usize,
    /// Largest allocated sparse polynomial and matrix-row structural footprint
    /// observed, excluding coefficient slots reported separately.
    pub peak_sparse_row_bytes: usize,
    /// Largest dense row-reducer and backend scratch footprint observed.
    pub peak_dense_row_bytes: usize,
    /// Largest inline coefficient-slot footprint in retained sparse values
    /// observed. Context-owned allocations behind a coefficient are excluded.
    pub peak_coefficient_bytes: usize,
    /// Largest total Rust-owned workspace footprint observed at one phase.
    pub peak_workspace_bytes: usize,
    /// Generic dense reductions.
    pub generic_row_reductions: usize,
    /// Dense fraction-free rational reductions through FLINT.
    pub rational_dense_row_reductions: usize,
    /// Sparse binary reductions.
    pub binary_sparse_row_reductions: usize,
    /// Dense binary reductions.
    pub binary_dense_row_reductions: usize,
    /// Dense binary-extension reductions.
    pub binary_extension_dense_row_reductions: usize,
    /// Sparse word-prime reductions.
    pub prime_sparse_row_reductions: usize,
    /// F4-specific sparse/dense `A|B / C|D` reductions.
    pub prime_abcd_row_reductions: usize,
    /// Dense word-prime reductions.
    pub prime_dense_row_reductions: usize,
    /// Left (known-pivot) columns processed by `A|B / C|D` reductions.
    pub abcd_left_columns: usize,
    /// Right (new-pivot) columns processed by `A|B / C|D` reductions.
    pub abcd_right_columns: usize,
    /// Dense right-block cells materialized by `A|B / C|D` reductions.
    pub abcd_dense_cells: usize,
    /// Full-width target cells that the `A|B / C|D` splice avoided retaining.
    pub abcd_full_dense_cells: usize,
    /// `A|B / C|D` rounds whose independent `C -> D` projection used private
    /// worker parallelism.
    pub abcd_parallel_projection_rounds: usize,
    /// Target rows projected by those private parallel rounds.
    pub abcd_parallel_projection_rows: usize,
    /// Independent final Schreyer critical pairs verified in private worker
    /// arenas. Zero means the deterministic serial verifier was used.
    pub schreyer_parallel_verification_pairs: usize,
    /// Independent Schreyer degree cells constructed in private worker
    /// arenas and merged in canonical degree order.
    pub schreyer_parallel_cells: usize,
    /// Backend used by the last nonempty matrix.
    pub last_row_backend: Option<RowBackend>,
    /// Distinct monomials interned while constructing the retained result.
    /// Private verification-worker vocabulary is deliberately excluded.
    pub monomials_interned: usize,
    /// Whether signature mode failed exact verification and reran ordinary F4.
    pub signature_fallback: bool,
    /// Leading ideals compared with an exact target Hilbert numerator.
    pub hilbert_target_checks: usize,
    /// Pending critical pairs skipped after the target Hilbert numerator matched.
    pub hilbert_pruned_pairs: usize,
    /// Whether a target-guided candidate failed verification and ordinary F4 reran.
    pub hilbert_target_fallback: bool,
    /// Prime images computed during modular rational reconstruction.
    pub modular_primes: usize,
    /// Prime images rejected because their leading support was inconsistent.
    pub unlucky_primes: usize,
    /// Rational reconstruction attempts followed by exact verification.
    pub reconstruction_attempts: usize,
    /// Degree rounds recorded while learning reusable prime-field structure.
    pub trace_learning_rounds: usize,
    /// Degree rounds replayed without pair selection or symbolic preprocessing.
    pub trace_application_rounds: usize,
    /// Target and reducer rows reconstructed from checked trace instructions.
    pub trace_replayed_rows: usize,
    /// Prime images whose coefficients were incompatible with the learned
    /// structure and therefore used ordinary F4.
    pub trace_rejections: usize,
    /// Rational construction regime, absent for non-rational computations.
    pub rational_backend: Option<RationalBackend>,
    /// Rational change-matrix regime, absent when no such matrix was built.
    pub rational_transformation_backend: Option<RationalTransformationBackend>,
}

/// Wall-clock phase timings produced only by the explicit profiling entry
/// points used by benchmark drivers.
///
/// These measurements are deliberately separate from [`GroebnerOutput`]:
/// they are neither mathematical evidence nor stable serialized state.
#[derive(Clone, Debug, Default)]
pub struct GroebnerPhaseTimings {
    /// Input validation, packing, and initial canonicalization.
    pub input_conversion: Duration,
    /// Critical-pair installation, selection, and criteria.
    pub pair_selection: Duration,
    /// F4 symbolic preprocessing.
    pub symbolic_preprocessing: Duration,
    /// Monomial-column construction, workspace checks, and row assembly.
    pub matrix_construction: Duration,
    /// Backend row reduction.
    pub row_reduction: Duration,
    /// Reduced-row conversion and new-basis extraction.
    pub basis_extraction: Duration,
    /// CRT and rational reconstruction; zero for prime-field profiles.
    pub rational_reconstruction: Duration,
    /// Complete modular-image construction in a rational profile.
    ///
    /// Prime-field profiles split this work into the finer F4 phase fields;
    /// the rational trace-replay lane records it here to avoid changing the
    /// measured algorithm merely for instrumentation.
    pub modular_image_construction: Duration,
    /// Primitive-integer, fraction-free rational basis construction.
    pub primitive_integer_construction: Duration,
    /// Final interreduction, exact verification, and materialization.
    pub final_canonicalization_and_verification: Duration,
}

#[derive(Clone, Copy)]
pub(crate) enum ProfilePhase {
    InputConversion,
    PairSelection,
    SymbolicPreprocessing,
    MatrixConstruction,
    RowReduction,
    BasisExtraction,
    ModularImageConstruction,
    PrimitiveIntegerConstruction,
    RationalReconstruction,
    FinalCanonicalizationAndVerification,
}

impl GroebnerPhaseTimings {
    pub(crate) fn observe(&mut self, phase: ProfilePhase, elapsed: Duration) {
        let target = match phase {
            ProfilePhase::InputConversion => &mut self.input_conversion,
            ProfilePhase::PairSelection => &mut self.pair_selection,
            ProfilePhase::SymbolicPreprocessing => &mut self.symbolic_preprocessing,
            ProfilePhase::MatrixConstruction => &mut self.matrix_construction,
            ProfilePhase::RowReduction => &mut self.row_reduction,
            ProfilePhase::BasisExtraction => &mut self.basis_extraction,
            ProfilePhase::ModularImageConstruction => &mut self.modular_image_construction,
            ProfilePhase::PrimitiveIntegerConstruction => &mut self.primitive_integer_construction,
            ProfilePhase::RationalReconstruction => &mut self.rational_reconstruction,
            ProfilePhase::FinalCanonicalizationAndVerification => {
                &mut self.final_canonicalization_and_verification
            }
        };
        *target += elapsed;
    }
}

pub(crate) fn profile_phase<T>(
    profile: &mut Option<&mut GroebnerPhaseTimings>,
    phase: ProfilePhase,
    operation: impl FnOnce() -> T,
) -> T {
    let started = profile.is_some().then(Instant::now);
    let output = operation();
    if let (Some(profile), Some(started)) = (profile.as_deref_mut(), started) {
        profile.observe(phase, started.elapsed());
    }
    output
}

impl GroebnerMetrics {
    fn observe_round_degree(&mut self, degree: u64) {
        self.minimum_round_degree = Some(
            self.minimum_round_degree
                .map_or(degree, |minimum| minimum.min(degree)),
        );
        self.maximum_round_degree = Some(
            self.maximum_round_degree
                .map_or(degree, |maximum| maximum.max(degree)),
        );
        self.round_degree_sum = self.round_degree_sum.saturating_add(u128::from(degree));
    }

    fn observe_workspace(
        &mut self,
        monomial_bytes: usize,
        pair_bytes: usize,
        sparse_row_bytes: usize,
        dense_row_bytes: usize,
        coefficient_bytes: usize,
    ) {
        self.peak_monomial_bytes = self.peak_monomial_bytes.max(monomial_bytes);
        self.peak_pair_bytes = self.peak_pair_bytes.max(pair_bytes);
        self.peak_sparse_row_bytes = self.peak_sparse_row_bytes.max(sparse_row_bytes);
        self.peak_dense_row_bytes = self.peak_dense_row_bytes.max(dense_row_bytes);
        self.peak_coefficient_bytes = self.peak_coefficient_bytes.max(coefficient_bytes);
        let total = monomial_bytes
            .saturating_add(pair_bytes)
            .saturating_add(sparse_row_bytes)
            .saturating_add(dense_row_bytes)
            .saturating_add(coefficient_bytes);
        self.peak_workspace_bytes = self.peak_workspace_bytes.max(total);
    }
}

/// Exact evidence established for a returned field Gröbner basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroebnerEvidence {
    /// Monomial generators were normalized and reduced to their divisibility
    /// antichain; every same-component S-polynomial vanishes identically.
    MonomialDivisibilityTheorem,
    /// Schreyer's theorem was applied to a monomial module basis: exact LCM
    /// relations from minimal colon generators were interreduced without an
    /// F4 matrix, so completeness follows directly from the colon ideals.
    MonomialSchreyerTheorem,
    /// The exact F4 construction exhausted its critical-pair set using
    /// Buchberger and Gebauer--Moeller criteria; final reducedness and source
    /// generator containment were checked.
    ExactConstruction,
    /// A separate final Buchberger sweep additionally reduced every required
    /// critical pair to zero.
    IndependentBuchbergerSweep,
}

/// Verified field Gröbner output from a fixed kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroebnerOutput<C> {
    /// Reduced monic basis, sorted by descending leading monomial.
    pub basis: Vec<SparsePolynomial<C>>,
    /// Operation counters.
    pub metrics: GroebnerMetrics,
    /// Exact evidence established by this invocation.
    pub evidence: GroebnerEvidence,
}

/// Verified field Gröbner basis together with its input-generator change matrix.
///
/// Row `j` of [`Self::transformations`] expresses basis element `j` as a
/// polynomial linear combination of the supplied generators, in their exact
/// input order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroebnerTransformOutput<C> {
    /// Reduced monic basis, sorted by descending leading monomial.
    pub basis: Vec<SparsePolynomial<C>>,
    /// Basis-major polynomial change matrix from input generators.
    pub transformations: Vec<Vec<SparsePolynomial<C>>>,
    /// Operation counters, including transformation-aware row work.
    pub metrics: GroebnerMetrics,
    /// Exact evidence established by this invocation.
    pub evidence: GroebnerEvidence,
}

/// Verified expression of one in a supplied field-polynomial generator family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnitIdealCertificateOutput<C> {
    /// Multipliers in the exact input-generator order.
    pub multipliers: Vec<SparsePolynomial<C>>,
    /// F4 phase and representation counters.
    pub metrics: GroebnerMetrics,
}

/// Fixed-kernel failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroebnerError {
    /// Input polynomials had incompatible arity or monomial order.
    ContextMismatch,
    /// Monomial exponent arithmetic left the word-sized kernel regime.
    ExponentOverflow,
    /// FGLM was requested for a quotient that is not zero-dimensional.
    NotZeroDimensional,
    /// An explicit resource bound was reached.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Configured limit.
        limit: usize,
    },
    /// Final exact Buchberger verification failed.
    VerificationFailed,
}

impl fmt::Display for GroebnerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch => formatter.write_str("incompatible polynomial kernel contexts"),
            Self::ExponentOverflow => formatter.write_str("monomial exponent overflow"),
            Self::NotZeroDimensional => {
                formatter.write_str("FGLM requires a zero-dimensional quotient")
            }
            Self::ResourceLimit { resource, limit } => {
                write!(
                    formatter,
                    "Groebner kernel exceeded {resource} limit {limit}"
                )
            }
            Self::VerificationFailed => {
                formatter.write_str("computed basis failed exact Buchberger verification")
            }
        }
    }
}

impl std::error::Error for GroebnerError {}

/// Compute using the generic coefficient-bearing row lane.
pub fn groebner_basis<F: Field>(
    field: &F,
    generators: &[SparsePolynomial<F::Element>],
    options: &GroebnerOptions,
) -> Result<GroebnerOutput<F::Element>, GroebnerError> {
    run_with_signature_fallback(field, generators, options, &GenericReducer, None)
}

/// Reduce several polynomials by one ordered reducer family.
///
/// Inputs share one compact monomial arena and one packed reducer family. If
/// `basis` is a Gröbner basis, each output is the canonical normal form in the
/// declared monomial order. Otherwise the result is the deterministic ordered
/// remainder. The reduction-step bound applies independently to each value.
pub fn polynomial_normal_forms<F: Field>(
    field: &F,
    basis: &[SparsePolynomial<F::Element>],
    values: &[SparsePolynomial<F::Element>],
    options: &GroebnerOptions,
) -> Result<Vec<SparsePolynomial<F::Element>>, GroebnerError> {
    let Some(first) = values.first().or_else(|| basis.first()) else {
        return Ok(Vec::new());
    };
    let arity = first.arity();
    let order = first.order();
    if basis
        .iter()
        .chain(values)
        .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
    {
        return Err(GroebnerError::ContextMismatch);
    }
    if basis.iter().any(SparsePolynomial::is_zero) {
        return Err(GroebnerError::VerificationFailed);
    }
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let input_terms = basis
        .iter()
        .chain(values)
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let mut arena = MonomialArena::with_capacity(
        arity,
        order,
        input_terms
            .max(input_terms.saturating_mul(2).min(65_536))
            .max(64),
    );
    let basis = basis
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        let value = ArenaPolynomial::from_sparse(&mut arena, value, field)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        output.push(
            normal_form(
                field,
                value,
                &basis,
                options.max_reduction_steps,
                &mut arena,
            )?
            .to_sparse(&arena, field),
        );
    }
    Ok(output)
}

/// One immutable reducer family with pooled mutable normal-form workspaces.
///
/// Preparation interns the basis support and constructs its leading-divisor
/// index once. Independent callers borrow a workspace from an internal pool;
/// concurrent calls receive separate monomial arenas, while sequential calls
/// retain their arena vocabulary and allocations. The mathematical result is
/// the same deterministic ordered remainder as [`polynomial_normal_forms`].
pub struct PreparedPolynomialReducer<C: Clone + Eq + Send + Sync> {
    arity: usize,
    order: MonomialOrder,
    basis: Vec<SparsePolynomial<C>>,
    pool: Mutex<Vec<PolynomialReducerWorkspace<C>>>,
    max_cached_workspaces: usize,
}

struct PolynomialReducerWorkspace<C> {
    arena: MonomialArena,
    basis: Vec<ArenaPolynomial<C>>,
    divisor_index: LeadingDivisorIndex,
}

impl<C: Clone + Eq + Send + Sync> PreparedPolynomialReducer<C> {
    /// Validate and prepare one ordered reducer family.
    ///
    /// `arity` and `order` remain explicit so the empty family is a valid
    /// prepared identity reduction for that polynomial context.
    pub fn new<F: Field<Element = C>>(
        field: &F,
        arity: usize,
        order: MonomialOrder,
        basis: &[SparsePolynomial<C>],
    ) -> Result<Self, GroebnerError> {
        if basis
            .iter()
            .any(|polynomial| polynomial.arity() != arity || polynomial.order() != order)
        {
            return Err(GroebnerError::ContextMismatch);
        }
        if basis.iter().any(SparsePolynomial::is_zero) {
            return Err(GroebnerError::VerificationFailed);
        }
        let workspace = PolynomialReducerWorkspace::new(field, arity, order, basis)?;
        Ok(Self {
            arity,
            order,
            basis: basis.to_vec(),
            pool: Mutex::new(vec![workspace]),
            max_cached_workspaces: std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get),
        })
    }

    /// Reduce several values while reusing a prepared basis and workspace.
    ///
    /// The reduction-step limit applies independently to each input value.
    pub fn normal_forms<F: Field<Element = C>>(
        &self,
        field: &F,
        values: &[SparsePolynomial<C>],
        max_reduction_steps: usize,
    ) -> Result<Vec<SparsePolynomial<C>>, GroebnerError> {
        if values
            .iter()
            .any(|polynomial| polynomial.arity() != self.arity || polynomial.order() != self.order)
        {
            return Err(GroebnerError::ContextMismatch);
        }
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let mut workspace = self
            .pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop()
            .map(Ok)
            .unwrap_or_else(|| {
                PolynomialReducerWorkspace::new(field, self.arity, self.order, &self.basis)
            })?;
        let result = workspace.normal_forms(field, values, max_reduction_steps);
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if pool.len() < self.max_cached_workspaces {
            pool.push(workspace);
        }
        result
    }
}

impl<C: Clone + Eq + Send + Sync> fmt::Debug for PreparedPolynomialReducer<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedPolynomialReducer")
            .field("arity", &self.arity)
            .field("order", &self.order)
            .field("basis_polynomials", &self.basis.len())
            .finish()
    }
}

impl<C: Clone + Eq + Send + Sync> PolynomialReducerWorkspace<C> {
    fn new<F: Field<Element = C>>(
        field: &F,
        arity: usize,
        order: MonomialOrder,
        basis: &[SparsePolynomial<C>],
    ) -> Result<Self, GroebnerError> {
        let input_terms = basis
            .iter()
            .map(|polynomial| polynomial.terms().len())
            .sum::<usize>();
        let mut arena = MonomialArena::with_capacity(
            arity,
            order,
            input_terms
                .max(input_terms.saturating_mul(2).min(65_536))
                .max(64),
        );
        let basis = basis
            .iter()
            .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, field))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        let divisor_index = LeadingDivisorIndex::new(&basis, &arena);
        Ok(Self {
            arena,
            basis,
            divisor_index,
        })
    }

    fn normal_forms<F: Field<Element = C>>(
        &mut self,
        field: &F,
        values: &[SparsePolynomial<C>],
        max_reduction_steps: usize,
    ) -> Result<Vec<SparsePolynomial<C>>, GroebnerError> {
        values
            .iter()
            .map(|value| {
                let value = ArenaPolynomial::from_sparse(&mut self.arena, value, field)
                    .map_err(|_| GroebnerError::ExponentOverflow)?;
                normal_form_with_divisor_index(
                    field,
                    value,
                    &self.basis,
                    &self.divisor_index,
                    max_reduction_steps,
                    &mut self.arena,
                )
                .map(|remainder| remainder.to_sparse(&self.arena, field))
            })
            .collect()
    }
}

/// Compute a reduced basis and exact expressions in the supplied generators.
///
/// This transformation-aware path uses ordinary F4 regardless of the
/// signature preference in `options`: signatures identify leading module
/// terms, while this result deliberately tracks complete polynomial vectors.
pub fn groebner_basis_with_transform<F: Field>(
    field: &F,
    generators: &[SparsePolynomial<F::Element>],
    options: &GroebnerOptions,
) -> Result<GroebnerTransformOutput<F::Element>, GroebnerError> {
    let (arity, order) = validate_inputs(generators)?;
    let input_terms = generators
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let arena_capacity = input_terms
        .max(input_terms.saturating_mul(8).min(65_536))
        .max(256);
    let mut arena = MonomialArena::with_capacity(arity, order, arena_capacity);
    let arena_generators = generators
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let (basis, transformations, mut metrics) =
        run_tracked_basis(field, &arena_generators, options, &mut arena)?;
    verify_construction_output(field, &arena_generators, &basis, options, &mut arena)?;
    verify_basis_transform(
        field,
        &arena_generators,
        &basis,
        &transformations,
        &mut arena,
    )?;
    metrics.monomials_interned = arena.len();
    Ok(GroebnerTransformOutput {
        basis: basis
            .into_iter()
            .map(|polynomial| polynomial.to_sparse(&arena, field))
            .collect(),
        transformations: transformations
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|polynomial| polynomial.to_sparse(&arena, field))
                    .collect()
            })
            .collect(),
        metrics,
        evidence: GroebnerEvidence::ExactConstruction,
    })
}

/// Express one in a field-polynomial generator family with transformation-
/// aware F4 row reduction, or return `None` when the generated ideal is not
/// the unit ideal.
pub fn unit_ideal_certificate<F: Field>(
    field: &F,
    generators: &[SparsePolynomial<F::Element>],
    options: &GroebnerOptions,
) -> Result<Option<UnitIdealCertificateOutput<F::Element>>, GroebnerError> {
    let (arity, order) = validate_inputs(generators)?;
    let input_terms = generators
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let arena_capacity = input_terms
        .max(input_terms.saturating_mul(8).min(65_536))
        .max(256);
    let mut arena = MonomialArena::with_capacity(arity, order, arena_capacity);
    let arena_generators = generators
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let (basis, transformations, mut metrics) =
        run_tracked_basis(field, &arena_generators, options, &mut arena)?;
    if basis.len() != 1
        || basis[0]
            .leading_term()
            .is_none_or(|(leading, _)| arena.total_degree(leading) != 0)
    {
        return Ok(None);
    }
    let multipliers = transformations
        .into_iter()
        .next()
        .ok_or(GroebnerError::VerificationFailed)?;
    verify_unit_ideal_transform(field, &arena_generators, &multipliers, &mut arena)?;
    metrics.monomials_interned = arena.len();
    Ok(Some(UnitIdealCertificateOutput {
        multipliers: multipliers
            .into_iter()
            .map(|polynomial| polynomial.to_sparse(&arena, field))
            .collect(),
        metrics,
    }))
}

/// Compute over a word-sized prime field using sparse/dense specialized rows.
pub fn groebner_basis_prime(
    field: &PrimeField32,
    generators: &[SparsePolynomial<u32>],
    options: &GroebnerOptions,
) -> Result<GroebnerOutput<u32>, GroebnerError> {
    run_with_signature_fallback(field, generators, options, &PrimeReducer, None)
}

/// Compute over a word-sized prime field while collecting benchmark-only
/// phase timings separately from the mathematical result.
pub fn groebner_basis_prime_profiled(
    field: &PrimeField32,
    generators: &[SparsePolynomial<u32>],
    options: &GroebnerOptions,
) -> Result<(GroebnerOutput<u32>, GroebnerPhaseTimings), GroebnerError> {
    let mut profile = GroebnerPhaseTimings::default();
    let output = run_with_signature_fallback(
        field,
        generators,
        options,
        &PrimeReducer,
        Some(&mut profile),
    )?;
    Ok((output, profile))
}

/// Internal coefficient-independent structure learned from one ordinary F4
/// prime image. The trace never crosses the kernel boundary.
pub(crate) struct PrimeTraceSession {
    arena: MonomialArena,
    trace: F4Trace,
}

impl PrimeTraceSession {
    pub(crate) fn fork(&self) -> Self {
        Self {
            arena: self.arena.fork(),
            trace: self.trace.clone(),
        }
    }
}

/// Internal prime image produced by checked trace replay. It is a modular
/// reconstruction input, not independently certified public output.
pub(crate) struct PrimeTraceImage {
    pub(crate) basis: Vec<SparsePolynomial<u32>>,
    pub(crate) metrics: GroebnerMetrics,
}

pub(crate) fn learn_prime_trace(
    field: &PrimeField32,
    generators: &[SparsePolynomial<u32>],
    options: &GroebnerOptions,
) -> Result<(GroebnerOutput<u32>, PrimeTraceSession), GroebnerError> {
    debug_assert_eq!(options.algorithm, Algorithm::F4);
    let (arity, order) = validate_inputs(generators)?;
    let input_terms = generators
        .iter()
        .map(|polynomial| polynomial.terms().len())
        .sum::<usize>();
    let arena_capacity = input_terms
        .max(input_terms.saturating_mul(8).min(65_536))
        .max(256);
    let mut arena = MonomialArena::with_capacity(arity, order, arena_capacity);
    let compact_generators = generators
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let mut trace = F4Trace::default();
    let mut output = run(
        field,
        &compact_generators,
        options,
        &PrimeReducer,
        &mut arena,
        Some(&mut trace),
        None,
    )?;
    if output.hilbert_guided_stop {
        verify_output(
            field,
            &compact_generators,
            &output.basis,
            options,
            &PrimeReducer,
            &mut arena,
        )?;
        output.evidence = GroebnerEvidence::IndependentBuchbergerSweep;
    } else {
        verify_construction_output(
            field,
            &compact_generators,
            &output.basis,
            options,
            &mut arena,
        )?;
    }
    output.metrics.trace_learning_rounds = trace.rounds.len();
    let output = materialize_output(field, output, &arena);
    Ok((output, PrimeTraceSession { arena, trace }))
}

pub(crate) fn apply_prime_trace(
    session: &mut PrimeTraceSession,
    field: &PrimeField32,
    generators: &[SparsePolynomial<u32>],
    options: &GroebnerOptions,
) -> Result<Option<PrimeTraceImage>, GroebnerError> {
    debug_assert_eq!(options.algorithm, Algorithm::F4);
    let (arity, order) = validate_inputs(generators)?;
    let PrimeTraceSession { arena, trace } = session;
    if arena.arity() != arity || arena.order() != order {
        return Ok(None);
    }
    let generators = generators
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let mut basis = normalized_generators(field, generators, arena);
    if !supports_match(&basis, &trace.initial_supports) {
        return Ok(None);
    }

    let mut metrics = GroebnerMetrics {
        trace_application_rounds: trace.rounds.len(),
        ..GroebnerMetrics::default()
    };
    for round in &trace.rounds {
        let Some(mut rows) = rows_from_trace(&basis, &round.targets, arena)? else {
            return Ok(None);
        };
        let target_count = rows.len();
        let Some(reducers) = rows_from_trace(&basis, &round.reducers, arena)? else {
            return Ok(None);
        };
        metrics.trace_replayed_rows = metrics
            .trace_replayed_rows
            .saturating_add(target_count + reducers.len());
        rows.extend(reducers);
        if rows.len() > options.max_matrix_rows
            || round.columns.len() > options.max_matrix_columns
            || !columns_cover_rows(&round.columns, &rows, arena)
        {
            return Ok(None);
        }
        let pair_bytes = trace_workspace_bytes(Some(trace));
        if check_matrix_assembly_workspace(
            arena,
            &basis,
            &rows,
            &round.columns,
            pair_bytes,
            0,
            0,
            options,
            &mut metrics,
        )
        .is_err()
        {
            return Ok(None);
        }
        let matrix = assemble_f4_work(&rows, &round.columns, target_count, arena);
        let nonzeros = matrix.nonzeros();
        metrics.rounds += 1;
        metrics.observe_round_degree(round.degree);
        metrics.matrix_rows += matrix.len();
        metrics.matrix_columns += round.columns.len();
        metrics.matrix_nonzeros += nonzeros;
        metrics.peak_matrix_rows = metrics.peak_matrix_rows.max(matrix.len());
        metrics.peak_matrix_columns = metrics.peak_matrix_columns.max(round.columns.len());
        metrics.peak_matrix_nonzeros = metrics.peak_matrix_nonzeros.max(nonzeros);
        if observe_f4_workspace(
            field,
            arena,
            &basis,
            &rows,
            &round.columns,
            &matrix,
            pair_bytes,
            0,
            0,
            &PrimeReducer,
            options,
            &mut metrics,
        )
        .is_err()
        {
            return Ok(None);
        }
        let F4Reduction {
            rows: reduced,
            reduced_tails,
        } = PrimeReducer.reduce_f4(field, round.columns.len(), matrix, &mut metrics);

        let mut new_values = Vec::new();
        let mut round_reducers = basis.clone();
        for row in reduced {
            let polynomial = polynomial_from_row(field, &round.columns, row, arena);
            if polynomial.is_zero()
                || polynomial.leading_term().is_some_and(|(leading, _)| {
                    basis.iter().any(|known| {
                        known
                            .leading_term()
                            .is_some_and(|(known_leading, _)| arena.divides(known_leading, leading))
                    })
                })
            {
                continue;
            }
            let value = if reduced_tails {
                polynomial
            } else {
                normal_form(
                    field,
                    polynomial,
                    &round_reducers,
                    options.max_reduction_steps,
                    arena,
                )?
            };
            if value.is_zero() {
                continue;
            }
            let value = monic(field, value, arena);
            if basis.iter().chain(&new_values).any(|known| known == &value) {
                continue;
            }
            round_reducers.push(value.clone());
            new_values.push(value);
        }
        let leading = new_values
            .iter()
            .filter_map(|polynomial| polynomial.leading_term().map(|term| term.0))
            .collect::<Vec<_>>();
        if leading != round.new_leading
            || basis.len().saturating_add(new_values.len()) > options.max_basis_size
        {
            return Ok(None);
        }
        basis.extend(new_values);
    }

    basis = interreduce(field, basis, options.max_reduction_steps, arena)?;
    if !supports_match(&basis, &trace.final_supports) {
        return Ok(None);
    }
    metrics.monomials_interned = arena.len();
    Ok(Some(PrimeTraceImage {
        basis: basis
            .into_iter()
            .map(|polynomial| polynomial.to_sparse(arena, field))
            .collect(),
        metrics,
    }))
}

/// Compute over `F_{2^e}`, `2 <= e <= 16`, using packed `m4rste` rows.
pub fn groebner_basis_binary_extension(
    field: &BinaryExtensionField16,
    generators: &[SparsePolynomial<u16>],
    options: &GroebnerOptions,
) -> Result<GroebnerOutput<u16>, GroebnerError> {
    run_with_signature_fallback(field, generators, options, &BinaryExtensionReducer, None)
}

/// Compute over F₂ using coefficient-free sparse terms and packed F4 rows.
pub fn groebner_basis_f2(
    generators: &[SparsePolynomial<u8>],
    options: &GroebnerOptions,
) -> Result<GroebnerOutput<u8>, GroebnerError> {
    run_with_signature_fallback(&BinaryField, generators, options, &BinaryReducer, None)
}

fn run_with_signature_fallback<F, R>(
    field: &F,
    generators: &[SparsePolynomial<F::Element>],
    options: &GroebnerOptions,
    reducer: &R,
    mut profile: Option<&mut GroebnerPhaseTimings>,
) -> Result<GroebnerOutput<F::Element>, GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    let (mut arena, generators) =
        profile_phase(&mut profile, ProfilePhase::InputConversion, || {
            let (arity, order) = validate_inputs(generators)?;
            let input_terms = generators
                .iter()
                .map(|polynomial| polynomial.terms().len())
                .sum::<usize>();
            let arena_capacity = input_terms
                .max(input_terms.saturating_mul(8).min(65_536))
                .max(256);
            let mut arena = MonomialArena::with_capacity(arity, order, arena_capacity);
            let generators = generators
                .iter()
                .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, field))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            Ok((arena, generators))
        })?;
    let mut output = run(
        field,
        &generators,
        options,
        reducer,
        &mut arena,
        None,
        profile.as_deref_mut(),
    )?;
    if options.algorithm == Algorithm::F4 && !output.hilbert_guided_stop {
        profile_phase(
            &mut profile,
            ProfilePhase::FinalCanonicalizationAndVerification,
            || verify_construction_output(field, &generators, &output.basis, options, &mut arena),
        )?;
        return Ok(profile_phase(
            &mut profile,
            ProfilePhase::FinalCanonicalizationAndVerification,
            || materialize_output(field, output, &arena),
        ));
    }
    if profile_phase(
        &mut profile,
        ProfilePhase::FinalCanonicalizationAndVerification,
        || {
            verify_output(
                field,
                &generators,
                &output.basis,
                options,
                reducer,
                &mut arena,
            )
        },
    )
    .is_ok()
    {
        output.evidence = GroebnerEvidence::IndependentBuchbergerSweep;
        return Ok(profile_phase(
            &mut profile,
            ProfilePhase::FinalCanonicalizationAndVerification,
            || materialize_output(field, output, &arena),
        ));
    }
    if options.algorithm != Algorithm::SignatureF4 && !output.hilbert_guided_stop {
        return Err(GroebnerError::VerificationFailed);
    }
    let guided_metrics = output.metrics.clone();
    let mut plain = options.clone();
    plain.algorithm = Algorithm::F4;
    plain.hilbert_numerator_target = None;
    let mut output = run(
        field,
        &generators,
        &plain,
        reducer,
        &mut arena,
        None,
        profile.as_deref_mut(),
    )?;
    profile_phase(
        &mut profile,
        ProfilePhase::FinalCanonicalizationAndVerification,
        || verify_construction_output(field, &generators, &output.basis, options, &mut arena),
    )?;
    output.metrics.signature_fallback = options.algorithm == Algorithm::SignatureF4;
    output.metrics.hilbert_target_fallback = guided_metrics.hilbert_target_checks != 0;
    output.metrics.hilbert_target_checks = guided_metrics.hilbert_target_checks;
    output.metrics.hilbert_pruned_pairs = guided_metrics.hilbert_pruned_pairs;
    Ok(profile_phase(
        &mut profile,
        ProfilePhase::FinalCanonicalizationAndVerification,
        || materialize_output(field, output, &arena),
    ))
}

struct CompactGroebnerOutput<C> {
    basis: Vec<ArenaPolynomial<C>>,
    metrics: GroebnerMetrics,
    evidence: GroebnerEvidence,
    hilbert_guided_stop: bool,
}

fn materialize_output<F: Field>(
    field: &F,
    mut output: CompactGroebnerOutput<F::Element>,
    arena: &MonomialArena,
) -> GroebnerOutput<F::Element> {
    output.metrics.monomials_interned = arena.len();
    GroebnerOutput {
        basis: output
            .basis
            .into_iter()
            .map(|polynomial| polynomial.to_sparse(arena, field))
            .collect(),
        metrics: output.metrics,
        evidence: output.evidence,
    }
}

fn polynomial_workspace_bytes<C: Clone + Eq>(polynomials: &[ArenaPolynomial<C>]) -> (usize, usize) {
    let outer = polynomials
        .len()
        .saturating_mul(core::mem::size_of::<ArenaPolynomial<C>>());
    polynomials
        .iter()
        .fold((outer, 0), |(total, coefficients), polynomial| {
            (
                total.saturating_add(polynomial.allocated_support_bytes()),
                coefficients.saturating_add(polynomial.allocated_coefficient_bytes()),
            )
        })
}

fn projected_matrix_workspace_bytes<C: Clone + Eq>(rows: &[ArenaPolynomial<C>]) -> (usize, usize) {
    let outer = F4RowWork::<C>::projected_structural_bytes(rows.len());
    let entry_bytes = core::mem::size_of::<(u32, C)>();
    let coefficient_bytes = core::mem::size_of::<C>();
    let structural_entry_bytes = entry_bytes.saturating_sub(coefficient_bytes);
    rows.iter().fold((outer, 0), |(total, coefficients), row| {
        (
            total.saturating_add(row.len().saturating_mul(structural_entry_bytes)),
            coefficients.saturating_add(row.len().saturating_mul(coefficient_bytes)),
        )
    })
}

fn tracked_workspace_bytes<C: Clone + Eq>(
    representations: &[Vec<ArenaPolynomial<C>>],
) -> (usize, usize) {
    let mut sparse = representations
        .len()
        .saturating_mul(core::mem::size_of::<Vec<ArenaPolynomial<C>>>());
    let mut coefficients = 0_usize;
    for representation in representations {
        sparse = sparse.saturating_add(
            representation
                .capacity()
                .saturating_mul(core::mem::size_of::<ArenaPolynomial<C>>()),
        );
        for polynomial in representation {
            sparse = sparse.saturating_add(polynomial.allocated_support_bytes());
            coefficients = coefficients.saturating_add(polynomial.allocated_coefficient_bytes());
        }
    }
    (sparse, coefficients)
}

fn pair_workspace_bytes(
    pending: &Vec<Pair>,
    resolved: &HashSet<(usize, usize)>,
    signatures: &Vec<Signature>,
    syzygies: &Vec<Signature>,
) -> usize {
    // hashbrown admits roughly seven live entries per eight buckets. Recover
    // a conservative bucket count from the exposed usable capacity and add a
    // control byte per bucket.
    let resolved_buckets = resolved
        .capacity()
        .saturating_add(resolved.capacity().div_ceil(7));
    pending
        .capacity()
        .saturating_mul(core::mem::size_of::<Pair>())
        .saturating_add(
            resolved_buckets
                .saturating_mul(core::mem::size_of::<(usize, usize)>().saturating_add(1)),
        )
        .saturating_add(
            signatures
                .capacity()
                .saturating_mul(core::mem::size_of::<Signature>()),
        )
        .saturating_add(
            syzygies
                .capacity()
                .saturating_mul(core::mem::size_of::<Signature>()),
        )
}

fn vector_workspace_bytes<T>(capacity: usize) -> usize {
    capacity.saturating_mul(core::mem::size_of::<T>())
}

fn trace_workspace_bytes(trace: Option<&F4Trace>) -> usize {
    let Some(trace) = trace else {
        return 0;
    };
    let nested_support_bytes = |supports: &[Vec<MonomialId>]| {
        supports
            .len()
            .saturating_mul(core::mem::size_of::<Vec<MonomialId>>())
            .saturating_add(
                supports
                    .iter()
                    .map(|support| vector_workspace_bytes::<MonomialId>(support.capacity()))
                    .fold(0_usize, usize::saturating_add),
            )
    };
    nested_support_bytes(&trace.initial_supports)
        .saturating_add(nested_support_bytes(&trace.final_supports))
        .saturating_add(vector_workspace_bytes::<F4RoundTrace>(
            trace.rounds.capacity(),
        ))
        .saturating_add(
            trace
                .rounds
                .iter()
                .map(|round| {
                    vector_workspace_bytes::<F4RowPlan>(round.targets.capacity())
                        .saturating_add(vector_workspace_bytes::<F4RowPlan>(
                            round.reducers.capacity(),
                        ))
                        .saturating_add(vector_workspace_bytes::<MonomialId>(
                            round.columns.capacity(),
                        ))
                        .saturating_add(vector_workspace_bytes::<MonomialId>(
                            round.new_leading.capacity(),
                        ))
                })
                .fold(0_usize, usize::saturating_add),
        )
}

#[allow(clippy::too_many_arguments)]
fn observe_f4_workspace<F, R>(
    field: &F,
    arena: &MonomialArena,
    basis: &[ArenaPolynomial<F::Element>],
    rows: &[ArenaPolynomial<F::Element>],
    columns: &[MonomialId],
    matrix: &F4RowWork<F::Element>,
    pair_bytes: usize,
    extra_sparse_row_bytes: usize,
    extra_coefficient_bytes: usize,
    reducer: &R,
    options: &GroebnerOptions,
    metrics: &mut GroebnerMetrics,
) -> Result<(), GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    let (basis_bytes, basis_coefficients) = polynomial_workspace_bytes(basis);
    let (row_bytes, row_coefficients) = polynomial_workspace_bytes(rows);
    let (matrix_bytes, matrix_coefficients) = matrix.allocated_workspace_bytes();
    let sparse_row_bytes = basis_bytes
        .saturating_add(row_bytes)
        .saturating_add(matrix_bytes)
        .saturating_add(extra_sparse_row_bytes)
        .saturating_add(
            columns
                .len()
                .saturating_mul(core::mem::size_of::<MonomialId>()),
        )
        // `assemble_f4_work` owns this reverse-index scratch while the sparse
        // matrix grows. Retaining its full size here is a conservative peak.
        .saturating_add(arena.len().saturating_mul(core::mem::size_of::<u32>()));
    let coefficient_bytes = basis_coefficients
        .saturating_add(row_coefficients)
        .saturating_add(matrix_coefficients)
        .saturating_add(extra_coefficient_bytes);
    let dense_row_bytes = reducer.f4_dense_workspace_bytes(field, columns.len(), matrix);
    metrics.observe_workspace(
        arena.allocated_bytes(),
        pair_bytes,
        sparse_row_bytes,
        dense_row_bytes,
        coefficient_bytes,
    );
    if metrics.peak_workspace_bytes == usize::MAX
        || metrics.peak_workspace_bytes > options.max_workspace_bytes
    {
        return Err(resource("workspace bytes", options.max_workspace_bytes));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_matrix_assembly_workspace<C: Clone + Eq>(
    arena: &MonomialArena,
    basis: &[ArenaPolynomial<C>],
    rows: &[ArenaPolynomial<C>],
    columns: &[MonomialId],
    pair_bytes: usize,
    extra_sparse_row_bytes: usize,
    extra_coefficient_bytes: usize,
    options: &GroebnerOptions,
    metrics: &mut GroebnerMetrics,
) -> Result<(), GroebnerError> {
    let (basis_bytes, basis_coefficients) = polynomial_workspace_bytes(basis);
    let (row_bytes, row_coefficients) = polynomial_workspace_bytes(rows);
    let (matrix_bytes, matrix_coefficients) = projected_matrix_workspace_bytes(rows);
    let sparse_row_bytes = basis_bytes
        .saturating_add(row_bytes)
        .saturating_add(matrix_bytes)
        .saturating_add(extra_sparse_row_bytes)
        .saturating_add(
            columns
                .len()
                .saturating_mul(core::mem::size_of::<MonomialId>()),
        )
        .saturating_add(arena.len().saturating_mul(core::mem::size_of::<u32>()));
    let coefficient_bytes = basis_coefficients
        .saturating_add(row_coefficients)
        .saturating_add(matrix_coefficients)
        .saturating_add(extra_coefficient_bytes);
    metrics.observe_workspace(
        arena.allocated_bytes(),
        pair_bytes,
        sparse_row_bytes,
        0,
        coefficient_bytes,
    );
    if metrics.peak_workspace_bytes == usize::MAX
        || metrics.peak_workspace_bytes > options.max_workspace_bytes
    {
        return Err(resource("workspace bytes", options.max_workspace_bytes));
    }
    Ok(())
}

#[derive(Clone)]
struct Signature {
    generator: usize,
    monomial: MonomialId,
}

impl Signature {
    fn multiplied(
        &self,
        monomial: MonomialId,
        arena: &mut MonomialArena,
    ) -> Result<Self, GroebnerError> {
        Ok(Self {
            generator: self.generator,
            monomial: arena
                .product(self.monomial, monomial)
                .map_err(|_| GroebnerError::ExponentOverflow)?,
        })
    }

    fn divides(&self, other: &Self, arena: &MonomialArena) -> bool {
        self.generator == other.generator && arena.divides(self.monomial, other.monomial)
    }

    fn compare(&self, other: &Self, arena: &MonomialArena) -> Ordering {
        self.generator
            .cmp(&other.generator)
            .then_with(|| arena.compare(self.monomial, other.monomial))
    }
}

#[derive(Clone)]
struct Pair {
    left: usize,
    right: usize,
    lcm: MonomialId,
    degree: u64,
    signature: Signature,
}

struct SymbolicRows<E> {
    targets: Vec<ArenaPolynomial<E>>,
    reducers: Vec<ArenaPolynomial<E>>,
    reducer_plans: Vec<F4RowPlan>,
}

#[derive(Clone, Copy)]
struct F4RowPlan {
    basis_index: u32,
    multiplier: MonomialId,
}

#[derive(Clone, Default)]
struct F4Trace {
    initial_supports: Vec<Vec<MonomialId>>,
    rounds: Vec<F4RoundTrace>,
    final_supports: Vec<Vec<MonomialId>>,
}

#[derive(Clone)]
struct F4RoundTrace {
    degree: u64,
    targets: Vec<F4RowPlan>,
    reducers: Vec<F4RowPlan>,
    columns: Vec<MonomialId>,
    new_leading: Vec<MonomialId>,
}

fn supports_match<C: Clone + Eq>(
    basis: &[ArenaPolynomial<C>],
    expected: &[Vec<MonomialId>],
) -> bool {
    basis.len() == expected.len()
        && basis
            .iter()
            .zip(expected)
            .all(|(polynomial, support)| polynomial.support() == support)
}

fn rows_from_trace(
    basis: &[ArenaPolynomial<u32>],
    plans: &[F4RowPlan],
    arena: &mut MonomialArena,
) -> Result<Option<Vec<ArenaPolynomial<u32>>>, GroebnerError> {
    plans
        .iter()
        .map(|plan| {
            let Some(polynomial) = basis.get(plan.basis_index as usize) else {
                return Ok(None);
            };
            polynomial
                .monomial_mul(plan.multiplier, arena)
                .map(Some)
                .map_err(|_| GroebnerError::ExponentOverflow)
        })
        .collect()
}

fn columns_cover_rows<C: Clone + Eq>(
    columns: &[MonomialId],
    rows: &[ArenaPolynomial<C>],
    arena: &MonomialArena,
) -> bool {
    let mut included = MonomialMarks::new(arena);
    for &column in columns {
        included.insert(arena, column);
    }
    rows.iter()
        .flat_map(ArenaPolynomial::support)
        .all(|monomial| included.contains(arena, *monomial))
}

fn hilbert_target_reached<C: Clone + Eq>(
    basis: &[ArenaPolynomial<C>],
    options: &GroebnerOptions,
    arena: &MonomialArena,
    metrics: &mut GroebnerMetrics,
) -> bool {
    let Some(target) = options.hilbert_numerator_target.as_deref() else {
        return false;
    };
    metrics.hilbert_target_checks = metrics.hilbert_target_checks.saturating_add(1);
    let leading = basis
        .iter()
        .filter_map(|polynomial| {
            polynomial
                .leading_term()
                .map(|(monomial, _)| arena.get(monomial))
        })
        .collect::<Vec<_>>();
    crate::hilbert::standard_hilbert_numerator(arena.arity(), &leading)
        .is_ok_and(|numerator| equal_trimmed_polynomials(&numerator, target))
}

fn equal_trimmed_polynomials(left: &[Integer], right: &[Integer]) -> bool {
    fn trimmed_length(values: &[Integer]) -> usize {
        values
            .iter()
            .rposition(|value| !value.is_zero())
            .map_or(1, |index| index + 1)
    }
    let left_length = trimmed_length(left);
    let right_length = trimmed_length(right);
    left_length == right_length
        && (0..left_length).all(|index| {
            left.get(index).cloned().unwrap_or_else(Integer::zero)
                == right.get(index).cloned().unwrap_or_else(Integer::zero)
        })
}

fn run<F, R>(
    field: &F,
    generators: &[ArenaPolynomial<F::Element>],
    options: &GroebnerOptions,
    reducer: &R,
    arena: &mut MonomialArena,
    mut trace: Option<&mut F4Trace>,
    mut profile: Option<&mut GroebnerPhaseTimings>,
) -> Result<CompactGroebnerOutput<F::Element>, GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    let mut basis = profile_phase(&mut profile, ProfilePhase::InputConversion, || {
        normalized_generators(field, generators.to_vec(), arena)
    });
    if let Some(trace) = trace.as_deref_mut() {
        trace.initial_supports = basis
            .iter()
            .map(|polynomial| polynomial.support().to_vec())
            .collect();
    }
    if basis.is_empty() {
        if let Some(trace) = trace.as_deref_mut() {
            trace.final_supports.clear();
        }
        return Ok(CompactGroebnerOutput {
            basis,
            metrics: GroebnerMetrics::default(),
            evidence: GroebnerEvidence::ExactConstruction,
            hilbert_guided_stop: false,
        });
    }
    let one_monomial = arena
        .intern_exponents(&vec![0; arena.arity()])
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let mut signatures = basis
        .iter()
        .enumerate()
        .map(|(generator, _)| Signature {
            generator,
            monomial: one_monomial,
        })
        .collect::<Vec<_>>();
    let mut known_syzygies = Vec::<Signature>::new();
    let mut pending = Vec::<Pair>::new();
    let mut resolved_pairs = HashSet::<(usize, usize)>::new();
    let mut metrics = GroebnerMetrics::default();
    let mut hilbert_guided_stop = hilbert_target_reached(&basis, options, arena, &mut metrics);
    if !hilbert_guided_stop {
        profile_phase(
            &mut profile,
            ProfilePhase::PairSelection,
            || -> Result<(), GroebnerError> {
                for right in 1..basis.len() {
                    update_pairs(
                        &basis,
                        &signatures,
                        right,
                        options,
                        &known_syzygies,
                        &mut resolved_pairs,
                        &mut pending,
                        &mut metrics,
                        arena,
                    )?;
                }
                Ok(())
            },
        )?;
    }

    while !pending.is_empty() {
        if metrics.rounds == options.max_rounds {
            return Err(resource("rounds", options.max_rounds));
        }
        let (minimum_degree, targets, target_signatures, target_plans, processed_pairs) =
            profile_phase(
                &mut profile,
                ProfilePhase::PairSelection,
                || -> Result<_, GroebnerError> {
                    let minimum_degree = pending
                        .iter()
                        .map(|pair| pair.degree)
                        .min()
                        .expect("nonempty pending pairs");
                    let mut selected = Vec::new();
                    let mut retained = Vec::with_capacity(pending.len());
                    for pair in pending.drain(..) {
                        let room = options.max_pairs_per_round == 0
                            || selected.len() < options.max_pairs_per_round;
                        if pair.degree == minimum_degree && room {
                            selected.push(pair);
                        } else {
                            retained.push(pair);
                        }
                    }
                    pending = retained;
                    metrics.rounds += 1;
                    metrics.observe_round_degree(minimum_degree);
                    metrics.pairs_selected += selected.len();

                    let mut eligible = Vec::new();
                    let mut processed_pairs = Vec::new();
                    for pair in selected {
                        let pair_key = ordered_pair(pair.left, pair.right);
                        if chain_criterion(&basis, &pair, &resolved_pairs, arena) {
                            metrics.chain_criterion_pairs += 1;
                            resolved_pairs.insert(pair_key);
                            continue;
                        }
                        if options.algorithm == Algorithm::SignatureF4
                            && known_syzygies
                                .iter()
                                .any(|syzygy| syzygy.divides(&pair.signature, arena))
                        {
                            metrics.signature_pruned_pairs += 1;
                            continue;
                        }
                        processed_pairs.push(pair_key);
                        eligible.push(pair);
                    }
                    let (targets, target_signatures, target_plans) = selected_pair_rows(
                        field,
                        &basis,
                        &signatures,
                        eligible,
                        options.algorithm,
                        arena,
                    )?;
                    Ok((
                        minimum_degree,
                        targets,
                        target_signatures,
                        target_plans,
                        processed_pairs,
                    ))
                },
            )?;
        if targets.is_empty() {
            resolved_pairs.extend(processed_pairs);
            continue;
        }

        let SymbolicRows {
            targets: mut rows,
            reducers,
            reducer_plans,
        } = profile_phase(&mut profile, ProfilePhase::SymbolicPreprocessing, || {
            symbolic_preprocess_split(field, &basis, targets, options.max_workspace_bytes, arena)
        })?;
        let (columns, matrix) = profile_phase(
            &mut profile,
            ProfilePhase::MatrixConstruction,
            || -> Result<_, GroebnerError> {
                let target_count = rows.len();
                rows.extend(reducers);
                if rows.len() > options.max_matrix_rows {
                    return Err(resource("matrix rows", options.max_matrix_rows));
                }
                let columns = matrix_columns(&rows, arena);
                if columns.len() > options.max_matrix_columns {
                    return Err(resource("matrix columns", options.max_matrix_columns));
                }
                let pair_bytes =
                    pair_workspace_bytes(&pending, &resolved_pairs, &signatures, &known_syzygies)
                        .saturating_add(vector_workspace_bytes::<(usize, usize)>(
                            processed_pairs.capacity(),
                        ))
                        .saturating_add(vector_workspace_bytes::<Signature>(
                            target_signatures.capacity(),
                        ))
                        .saturating_add(vector_workspace_bytes::<F4RowPlan>(
                            target_plans.capacity(),
                        ))
                        .saturating_add(vector_workspace_bytes::<F4RowPlan>(
                            reducer_plans.capacity(),
                        ))
                        .saturating_add(trace_workspace_bytes(trace.as_deref()));
                check_matrix_assembly_workspace(
                    arena,
                    &basis,
                    &rows,
                    &columns,
                    pair_bytes,
                    0,
                    0,
                    options,
                    &mut metrics,
                )?;
                let matrix = assemble_f4_work(&rows, &columns, target_count, arena);
                let nonzeros = matrix.nonzeros();
                metrics.matrix_rows += matrix.len();
                metrics.matrix_columns += columns.len();
                metrics.matrix_nonzeros += nonzeros;
                metrics.peak_matrix_rows = metrics.peak_matrix_rows.max(matrix.len());
                metrics.peak_matrix_columns = metrics.peak_matrix_columns.max(columns.len());
                metrics.peak_matrix_nonzeros = metrics.peak_matrix_nonzeros.max(nonzeros);
                observe_f4_workspace(
                    field,
                    arena,
                    &basis,
                    &rows,
                    &columns,
                    &matrix,
                    pair_bytes,
                    0,
                    0,
                    reducer,
                    options,
                    &mut metrics,
                )?;
                Ok((columns, matrix))
            },
        )?;
        let F4Reduction {
            rows: reduced,
            reduced_tails,
        } = profile_phase(&mut profile, ProfilePhase::RowReduction, || {
            reducer.reduce_f4(field, columns.len(), matrix, &mut metrics)
        });

        let new_values = profile_phase(
            &mut profile,
            ProfilePhase::BasisExtraction,
            || -> Result<_, GroebnerError> {
                let mut new_values = Vec::new();
                let mut round_reducers = basis.clone();
                for row in reduced {
                    let polynomial = polynomial_from_row(field, &columns, row, arena);
                    if polynomial.is_zero() {
                        continue;
                    }
                    if polynomial.leading_term().is_some_and(|(leading, _)| {
                        basis.iter().any(|known| {
                            known.leading_term().is_some_and(|(known_leading, _)| {
                                arena.divides(known_leading, leading)
                            })
                        })
                    }) {
                        continue;
                    }
                    let value = if reduced_tails {
                        polynomial
                    } else {
                        normal_form(
                            field,
                            polynomial,
                            &round_reducers,
                            options.max_reduction_steps,
                            arena,
                        )?
                    };
                    if value.is_zero() {
                        continue;
                    }
                    let value = monic(field, value, arena);
                    if basis.iter().chain(&new_values).any(|known| known == &value) {
                        continue;
                    }
                    round_reducers.push(value.clone());
                    new_values.push(value);
                }
                Ok(new_values)
            },
        )?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.rounds.push(F4RoundTrace {
                degree: minimum_degree,
                targets: target_plans,
                reducers: reducer_plans,
                columns: columns.clone(),
                new_leading: new_values
                    .iter()
                    .filter_map(|polynomial| polynomial.leading_term().map(|term| term.0))
                    .collect(),
            });
        }
        if new_values.is_empty() {
            if options.algorithm == Algorithm::SignatureF4 {
                known_syzygies.extend(target_signatures);
            }
            resolved_pairs.extend(processed_pairs);
            continue;
        }

        profile_phase(
            &mut profile,
            ProfilePhase::PairSelection,
            || -> Result<(), GroebnerError> {
                for (new_offset, value) in new_values.into_iter().enumerate() {
                    if basis.len() == options.max_basis_size {
                        return Err(resource("basis size", options.max_basis_size));
                    }
                    let new_index = basis.len();
                    let signature = if options.algorithm == Algorithm::SignatureF4 {
                        target_signatures
                            .get(new_offset)
                            .cloned()
                            .unwrap_or(Signature {
                                generator: signatures.len(),
                                monomial: one_monomial,
                            })
                    } else {
                        Signature {
                            generator: signatures.len(),
                            monomial: one_monomial,
                        }
                    };
                    basis.push(value);
                    signatures.push(signature);
                    update_pairs(
                        &basis,
                        &signatures,
                        new_index,
                        options,
                        &known_syzygies,
                        &mut resolved_pairs,
                        &mut pending,
                        &mut metrics,
                        arena,
                    )?;
                }
                Ok(())
            },
        )?;
        resolved_pairs.extend(processed_pairs);
        if hilbert_target_reached(&basis, options, arena, &mut metrics) {
            metrics.hilbert_pruned_pairs =
                metrics.hilbert_pruned_pairs.saturating_add(pending.len());
            pending.clear();
            hilbert_guided_stop = true;
        }
    }

    basis = profile_phase(
        &mut profile,
        ProfilePhase::FinalCanonicalizationAndVerification,
        || interreduce(field, basis, options.max_reduction_steps, arena),
    )?;
    if let Some(trace) = trace {
        trace.final_supports = basis
            .iter()
            .map(|polynomial| polynomial.support().to_vec())
            .collect();
    }
    Ok(CompactGroebnerOutput {
        basis,
        metrics,
        evidence: GroebnerEvidence::ExactConstruction,
        hilbert_guided_stop,
    })
}

type TrackedRepresentation<C> = Vec<ArenaPolynomial<C>>;
type TrackedPolynomial<C> = (ArenaPolynomial<C>, TrackedRepresentation<C>);
type TrackedBasis<C> = (Vec<ArenaPolynomial<C>>, Vec<TrackedRepresentation<C>>);
type TrackedBasisResult<C> = (
    Vec<ArenaPolynomial<C>>,
    Vec<TrackedRepresentation<C>>,
    GroebnerMetrics,
);
type TrackedReducedRow<C> = (Vec<(u32, C)>, TrackedRepresentation<C>);

fn run_tracked_basis<F: Field>(
    field: &F,
    generators: &[ArenaPolynomial<F::Element>],
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<TrackedBasisResult<F::Element>, GroebnerError> {
    let one_monomial = arena
        .intern_exponents(&vec![0; arena.arity()])
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let (mut basis, mut representations) =
        normalized_tracked_generators(field, generators, one_monomial, arena);
    let mut metrics = GroebnerMetrics::default();
    if let Some(index) = basis.iter().position(|polynomial| {
        polynomial
            .leading_term()
            .is_some_and(|(leading, _)| leading == one_monomial)
    }) {
        return Ok((
            vec![basis.swap_remove(index)],
            vec![representations.swap_remove(index)],
            metrics,
        ));
    }
    if basis.is_empty() {
        return Ok((basis, representations, metrics));
    }

    let mut plain_options = options.clone();
    plain_options.algorithm = Algorithm::F4;
    let mut signatures = basis
        .iter()
        .enumerate()
        .map(|(generator, _)| Signature {
            generator,
            monomial: one_monomial,
        })
        .collect::<Vec<_>>();
    let known_syzygies = Vec::<Signature>::new();
    let mut pending = Vec::<Pair>::new();
    let mut resolved_pairs = HashSet::<(usize, usize)>::new();
    for right in 1..basis.len() {
        update_pairs(
            &basis,
            &signatures,
            right,
            &plain_options,
            &known_syzygies,
            &mut resolved_pairs,
            &mut pending,
            &mut metrics,
            arena,
        )?;
    }

    while !pending.is_empty() {
        if metrics.rounds == plain_options.max_rounds {
            return Err(resource("certificate rounds", plain_options.max_rounds));
        }
        let minimum_degree = pending
            .iter()
            .map(|pair| pair.degree)
            .min()
            .expect("nonempty pending certificate pairs");
        let mut selected = Vec::new();
        let mut retained = Vec::with_capacity(pending.len());
        for pair in pending.drain(..) {
            let room = plain_options.max_pairs_per_round == 0
                || selected.len() < plain_options.max_pairs_per_round;
            if pair.degree == minimum_degree && room {
                selected.push(pair);
            } else {
                retained.push(pair);
            }
        }
        pending = retained;
        metrics.rounds += 1;
        metrics.observe_round_degree(minimum_degree);
        metrics.pairs_selected += selected.len();

        let mut eligible = Vec::new();
        let mut processed_pairs = Vec::new();
        for pair in selected {
            let pair_key = ordered_pair(pair.left, pair.right);
            if chain_criterion(&basis, &pair, &resolved_pairs, arena) {
                metrics.chain_criterion_pairs += 1;
                resolved_pairs.insert(pair_key);
            } else {
                processed_pairs.push(pair_key);
                eligible.push(pair);
            }
        }
        let (targets, _, target_plans) =
            selected_pair_rows(field, &basis, &signatures, eligible, Algorithm::F4, arena)?;
        if targets.is_empty() {
            resolved_pairs.extend(processed_pairs);
            continue;
        }
        let SymbolicRows {
            targets: mut rows,
            reducers,
            reducer_plans,
        } = symbolic_preprocess_split(
            field,
            &basis,
            targets,
            plain_options.max_workspace_bytes,
            arena,
        )?;
        let mut row_representations =
            tracked_rows_from_plans(&representations, &target_plans, arena)?;
        row_representations.extend(tracked_rows_from_plans(
            &representations,
            &reducer_plans,
            arena,
        )?);
        rows.extend(reducers);
        if rows.len() > plain_options.max_matrix_rows {
            return Err(resource(
                "certificate matrix rows",
                plain_options.max_matrix_rows,
            ));
        }
        let columns = matrix_columns(&rows, arena);
        if columns.len() > plain_options.max_matrix_columns {
            return Err(resource(
                "certificate matrix columns",
                plain_options.max_matrix_columns,
            ));
        }
        let pair_bytes =
            pair_workspace_bytes(&pending, &resolved_pairs, &signatures, &known_syzygies)
                .saturating_add(vector_workspace_bytes::<(usize, usize)>(
                    processed_pairs.capacity(),
                ))
                .saturating_add(vector_workspace_bytes::<F4RowPlan>(target_plans.capacity()))
                .saturating_add(vector_workspace_bytes::<F4RowPlan>(
                    reducer_plans.capacity(),
                ));
        let (basis_representation_bytes, basis_representation_coefficients) =
            tracked_workspace_bytes(&representations);
        let (row_representation_bytes, row_representation_coefficients) =
            tracked_workspace_bytes(&row_representations);
        let extra_sparse_row_bytes =
            basis_representation_bytes.saturating_add(row_representation_bytes);
        let extra_coefficient_bytes =
            basis_representation_coefficients.saturating_add(row_representation_coefficients);
        check_matrix_assembly_workspace(
            arena,
            &basis,
            &rows,
            &columns,
            pair_bytes,
            extra_sparse_row_bytes,
            extra_coefficient_bytes,
            &plain_options,
            &mut metrics,
        )?;
        let matrix = assemble_f4_work(&rows, &columns, rows.len(), arena);
        metrics.matrix_rows += matrix.len();
        metrics.matrix_columns += columns.len();
        let nonzeros = matrix.nonzeros();
        metrics.matrix_nonzeros += nonzeros;
        metrics.peak_matrix_rows = metrics.peak_matrix_rows.max(matrix.len());
        metrics.peak_matrix_columns = metrics.peak_matrix_columns.max(columns.len());
        metrics.peak_matrix_nonzeros = metrics.peak_matrix_nonzeros.max(nonzeros);
        observe_f4_workspace(
            field,
            arena,
            &basis,
            &rows,
            &columns,
            &matrix,
            pair_bytes,
            extra_sparse_row_bytes,
            extra_coefficient_bytes,
            &GenericReducer,
            &plain_options,
            &mut metrics,
        )?;
        let matrix = matrix.into_rows();
        let reduced = reduce_tracked_matrix(
            field,
            columns.len(),
            matrix,
            row_representations,
            &mut metrics,
            arena,
        );

        let mut new_values = Vec::new();
        let mut new_representations = Vec::new();
        for (row, representation) in reduced {
            let polynomial = polynomial_from_row(field, &columns, row, arena);
            if polynomial.is_zero() {
                continue;
            }
            if polynomial.leading_term().is_some_and(|(leading, _)| {
                basis.iter().any(|known| {
                    known
                        .leading_term()
                        .is_some_and(|(known_leading, _)| arena.divides(known_leading, leading))
                })
            }) {
                continue;
            }
            let (value, representation) = monic_tracked(field, polynomial, representation, arena);
            if value
                .leading_term()
                .is_some_and(|(leading, _)| leading == one_monomial)
            {
                resolved_pairs.extend(processed_pairs);
                return Ok((vec![value], vec![representation], metrics));
            }
            if basis.iter().chain(&new_values).any(|known| known == &value) {
                continue;
            }
            new_values.push(value);
            new_representations.push(representation);
        }
        if new_values.is_empty() {
            resolved_pairs.extend(processed_pairs);
            continue;
        }
        for (value, representation) in new_values.into_iter().zip(new_representations) {
            if basis.len() == plain_options.max_basis_size {
                return Err(resource(
                    "certificate basis size",
                    plain_options.max_basis_size,
                ));
            }
            let new_index = basis.len();
            basis.push(value);
            representations.push(representation);
            signatures.push(Signature {
                generator: signatures.len(),
                monomial: one_monomial,
            });
            update_pairs(
                &basis,
                &signatures,
                new_index,
                &plain_options,
                &known_syzygies,
                &mut resolved_pairs,
                &mut pending,
                &mut metrics,
                arena,
            )?;
        }
        resolved_pairs.extend(processed_pairs);
    }
    let (basis, representations) = interreduce_tracked(
        field,
        basis,
        representations,
        plain_options.max_reduction_steps,
        arena,
    )?;
    Ok((basis, representations, metrics))
}

fn normalized_tracked_generators<F: Field>(
    field: &F,
    generators: &[ArenaPolynomial<F::Element>],
    one_monomial: MonomialId,
    arena: &MonomialArena,
) -> TrackedBasis<F::Element> {
    let generator_count = generators.len();
    let mut entries = Vec::new();
    for (index, generator) in generators.iter().enumerate() {
        let Some((_, leading)) = generator.leading_term() else {
            continue;
        };
        let inverse = field.inverse(leading);
        let polynomial = generator.scalar_mul(&inverse, arena, field);
        if entries
            .iter()
            .any(|(known, _): &(ArenaPolynomial<F::Element>, _)| known == &polynomial)
        {
            continue;
        }
        let mut representation = vec![ArenaPolynomial::zero(arena); generator_count];
        representation[index] = ArenaPolynomial::new(arena, [(one_monomial, inverse)], field);
        entries.push((polynomial, representation));
    }
    entries.sort_by(|(left, _), (right, _)| {
        arena.compare(
            right.leading_term().expect("nonzero").0,
            left.leading_term().expect("nonzero").0,
        )
    });
    entries.into_iter().unzip()
}

fn tracked_rows_from_plans<C: Clone + Eq>(
    representations: &[TrackedRepresentation<C>],
    plans: &[F4RowPlan],
    arena: &mut MonomialArena,
) -> Result<Vec<TrackedRepresentation<C>>, GroebnerError> {
    plans
        .iter()
        .map(|plan| {
            let representation = representations
                .get(plan.basis_index as usize)
                .ok_or(GroebnerError::VerificationFailed)?;
            representation
                .iter()
                .map(|polynomial| {
                    polynomial
                        .monomial_mul(plan.multiplier, arena)
                        .map_err(|_| GroebnerError::ExponentOverflow)
                })
                .collect()
        })
        .collect()
}

fn monic_tracked<F: Field>(
    field: &F,
    polynomial: ArenaPolynomial<F::Element>,
    representation: TrackedRepresentation<F::Element>,
    arena: &MonomialArena,
) -> (
    ArenaPolynomial<F::Element>,
    TrackedRepresentation<F::Element>,
) {
    let leading = polynomial
        .leading_term()
        .expect("a tracked F4 row is nonzero")
        .1;
    if leading == &field.one() {
        return (polynomial, representation);
    }
    let inverse = field.inverse(leading);
    (
        polynomial.scalar_mul(&inverse, arena, field),
        representation
            .into_iter()
            .map(|value| value.scalar_mul(&inverse, arena, field))
            .collect(),
    )
}

#[allow(clippy::too_many_arguments)]
fn normal_form_tracked_excluding<F: Field>(
    field: &F,
    value: ArenaPolynomial<F::Element>,
    representation: TrackedRepresentation<F::Element>,
    basis: &[ArenaPolynomial<F::Element>],
    basis_representations: &[TrackedRepresentation<F::Element>],
    excluded_reducer: Option<usize>,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<TrackedPolynomial<F::Element>, GroebnerError> {
    if basis.len() != basis_representations.len()
        || basis_representations
            .iter()
            .any(|row| row.len() != representation.len())
    {
        return Err(GroebnerError::VerificationFailed);
    }
    let mut remainder_terms = Vec::new();
    let mut steps = 0;
    let one = field.one();
    let divisor_index = LeadingDivisorIndex::new_excluding(basis, excluded_reducer, arena);
    let mut value = PolynomialAccumulator::from_polynomial(&value, arena);
    let mut representation = representation;
    let mut representation_workspace = PolynomialWorkspace::new();
    let mut representation_combined = ArenaPolynomial::zero(arena);
    while let Some((leading_monomial, leading_coefficient)) = value.leading_term(arena) {
        if steps == max_steps {
            return Err(resource("tracked normal-form steps", max_steps));
        }
        if let Some(reducer_index) = divisor_index.find(leading_monomial, arena) {
            let reducer = &basis[reducer_index];
            let (reducer_monomial, reducer_coefficient) =
                reducer.leading_term().expect("nonzero tracked reducer");
            let multiplier = arena
                .quotient(leading_monomial, reducer_monomial)
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .expect("selected tracked reducer divisibility");
            let scalar = if reducer_coefficient == &one {
                leading_coefficient
            } else {
                field.mul(&leading_coefficient, &field.inverse(reducer_coefficient))
            };
            value
                .subtract_multiple(reducer, multiplier, &scalar, arena, field)
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            for (coordinate, reducer_coordinate) in representation
                .iter_mut()
                .zip(&basis_representations[reducer_index])
            {
                coordinate
                    .sub_term_into(
                        reducer_coordinate,
                        multiplier,
                        &scalar,
                        arena,
                        field,
                        &mut representation_workspace,
                        &mut representation_combined,
                    )
                    .map_err(|_| GroebnerError::ExponentOverflow)?;
                core::mem::swap(coordinate, &mut representation_combined);
            }
        } else {
            remainder_terms.push(
                value
                    .pop_leading(arena)
                    .expect("the observed tracked leading term remains present"),
            );
        }
        steps += 1;
    }
    Ok((
        ArenaPolynomial::from_sorted_terms(arena, remainder_terms, field),
        representation,
    ))
}

pub(crate) fn interreduce_tracked<F: Field>(
    field: &F,
    basis: Vec<ArenaPolynomial<F::Element>>,
    representations: Vec<TrackedRepresentation<F::Element>>,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<TrackedBasis<F::Element>, GroebnerError> {
    if basis.len() != representations.len() {
        return Err(GroebnerError::VerificationFailed);
    }
    let mut minimal_indices = Vec::with_capacity(basis.len());
    for (index, polynomial) in basis.iter().enumerate() {
        let leading = polynomial
            .leading_term()
            .map(|(monomial, _)| monomial)
            .ok_or(GroebnerError::VerificationFailed)?;
        let redundant = basis.iter().enumerate().any(|(other_index, other)| {
            if other_index == index {
                return false;
            }
            let other_leading = other
                .leading_term()
                .map(|(monomial, _)| monomial)
                .expect("tracked bases contain no zero polynomial");
            arena.divides(other_leading, leading)
                && (other_leading != leading || other_index < index)
        });
        if !redundant {
            minimal_indices.push(index);
        }
    }
    let minimal_basis = minimal_indices
        .iter()
        .map(|&index| basis[index].clone())
        .collect::<Vec<_>>();
    let minimal_representations = minimal_indices
        .iter()
        .map(|&index| representations[index].clone())
        .collect::<Vec<_>>();

    let mut reduced = Vec::with_capacity(minimal_basis.len());
    let mut reduced_representations = Vec::with_capacity(minimal_basis.len());
    for index in 0..minimal_basis.len() {
        let (remainder, representation) = normal_form_tracked_excluding(
            field,
            minimal_basis[index].clone(),
            minimal_representations[index].clone(),
            &minimal_basis,
            &minimal_representations,
            Some(index),
            max_steps,
            arena,
        )?;
        if remainder.is_zero() {
            continue;
        }
        let (remainder, representation) = monic_tracked(field, remainder, representation, arena);
        if reduced.contains(&remainder) {
            continue;
        }
        reduced.push(remainder);
        reduced_representations.push(representation);
    }
    let mut entries = reduced
        .into_iter()
        .zip(reduced_representations)
        .collect::<Vec<_>>();
    entries.sort_by(|(left, _), (right, _)| {
        arena.compare(
            right.leading_term().expect("nonzero").0,
            left.leading_term().expect("nonzero").0,
        )
    });
    Ok(entries.into_iter().unzip())
}

fn reduce_tracked_matrix<F: Field>(
    field: &F,
    columns: usize,
    rows: Vec<Vec<(u32, F::Element)>>,
    mut representations: Vec<TrackedRepresentation<F::Element>>,
    metrics: &mut GroebnerMetrics,
    arena: &MonomialArena,
) -> Vec<TrackedReducedRow<F::Element>> {
    debug_assert_eq!(rows.len(), representations.len());
    metrics.generic_row_reductions += 1;
    metrics.last_row_backend = Some(RowBackend::GenericDense);
    let mut rows = rows
        .into_iter()
        .map(|row| {
            let mut dense = vec![field.zero(); columns];
            for (column, value) in row {
                dense[column as usize] = value;
            }
            dense
        })
        .collect::<Vec<_>>();
    let mut rank = 0;
    for column in 0..columns {
        let Some(pivot) = (rank..rows.len()).find(|&row| !field.is_zero(&rows[row][column])) else {
            continue;
        };
        rows.swap(rank, pivot);
        representations.swap(rank, pivot);
        let inverse = field.inverse(&rows[rank][column]);
        for entry in &mut rows[rank][column..] {
            *entry = field.mul(entry, &inverse);
        }
        for value in &mut representations[rank] {
            *value = value.scalar_mul(&inverse, arena, field);
        }
        let pivot_row = rows[rank].clone();
        let pivot_representation = representations[rank].clone();
        for row_index in 0..rows.len() {
            if row_index == rank || field.is_zero(&rows[row_index][column]) {
                continue;
            }
            let factor = rows[row_index][column].clone();
            for index in column..columns {
                let product = field.mul(&factor, &pivot_row[index]);
                rows[row_index][index] = field.add(&rows[row_index][index], &field.neg(&product));
            }
            for (value, pivot_value) in representations[row_index]
                .iter_mut()
                .zip(&pivot_representation)
            {
                let correction = pivot_value.scalar_mul(&factor, arena, field);
                *value = value.sub(&correction, arena, field);
            }
        }
        rank += 1;
        if rank == rows.len() {
            break;
        }
    }
    rows.into_iter()
        .zip(representations)
        .map(|(row, representation)| {
            let row = row
                .into_iter()
                .enumerate()
                .filter(|(_, value)| !field.is_zero(value))
                .map(|(column, value)| {
                    (
                        u32::try_from(column)
                            .expect("a tracked F4 column fits the monomial-ID address space"),
                        value,
                    )
                })
                .collect();
            (row, representation)
        })
        .collect()
}

fn verify_unit_ideal_transform<F: Field>(
    field: &F,
    generators: &[ArenaPolynomial<F::Element>],
    multipliers: &[ArenaPolynomial<F::Element>],
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    if generators.len() != multipliers.len() {
        return Err(GroebnerError::VerificationFailed);
    }
    let mut represented = ArenaPolynomial::zero(arena);
    for (multiplier, generator) in multipliers.iter().zip(generators) {
        let product = multiplier
            .multiply(generator, arena, field, None)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        represented = represented.add(&product, arena, field);
    }
    let one_monomial = arena
        .intern_exponents(&vec![0; arena.arity()])
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    let one = ArenaPolynomial::new(arena, [(one_monomial, field.one())], field);
    if represented == one {
        Ok(())
    } else {
        Err(GroebnerError::VerificationFailed)
    }
}

pub(crate) fn verify_basis_transform<F: Field>(
    field: &F,
    generators: &[ArenaPolynomial<F::Element>],
    basis: &[ArenaPolynomial<F::Element>],
    transformations: &[TrackedRepresentation<F::Element>],
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    if basis.len() != transformations.len()
        || transformations
            .iter()
            .any(|row| row.len() != generators.len())
    {
        return Err(GroebnerError::VerificationFailed);
    }
    let mut workspace = PolynomialWorkspace::new();
    for (expected, row) in basis.iter().zip(transformations) {
        let matches = expected
            .equals_sum_products(
                row.iter()
                    .zip(generators)
                    .filter(|(multiplier, generator)| {
                        !multiplier.is_zero() && !generator.is_zero()
                    }),
                arena,
                field,
                &mut workspace,
            )
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        if !matches {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

pub(crate) fn validate_inputs<C: Clone + Eq>(
    generators: &[SparsePolynomial<C>],
) -> Result<(usize, MonomialOrder), GroebnerError> {
    let Some(first) = generators.first() else {
        return Err(GroebnerError::ContextMismatch);
    };
    let arity = first.arity();
    let order = first.order();
    if generators
        .iter()
        .any(|value| value.arity() != arity || value.order() != order)
    {
        Err(GroebnerError::ContextMismatch)
    } else {
        Ok((arity, order))
    }
}

#[derive(Clone)]
struct PairCandidate {
    left: usize,
    lcm: MonomialId,
    degree: u64,
    product: bool,
    active: bool,
}

fn compare_pair_candidates(left: &PairCandidate, right: &PairCandidate) -> Ordering {
    left.degree
        .cmp(&right.degree)
        .then_with(|| left.lcm.cmp(&right.lcm))
        .then_with(|| left.left.cmp(&right.left))
}

#[allow(clippy::too_many_arguments)]
fn update_pairs<C: Clone + Eq>(
    basis: &[ArenaPolynomial<C>],
    signatures: &[Signature],
    new_index: usize,
    options: &GroebnerOptions,
    known_syzygies: &[Signature],
    resolved_pairs: &mut HashSet<(usize, usize)>,
    pending: &mut Vec<Pair>,
    metrics: &mut GroebnerMetrics,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    let new_lm = basis[new_index]
        .leading_term()
        .map(|(monomial, _)| monomial)
        .expect("basis members are nonzero");
    let mut candidates = Vec::with_capacity(new_index);
    for (left, polynomial) in basis.iter().enumerate().take(new_index) {
        let left_lm = polynomial
            .leading_term()
            .map(|(monomial, _)| monomial)
            .expect("basis members are nonzero");
        let lcm = arena
            .lcm(left_lm, new_lm)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        let degree = arena.total_degree(lcm);
        let product = Some(degree)
            == arena
                .total_degree(left_lm)
                .checked_add(arena.total_degree(new_lm));
        candidates.push(PairCandidate {
            left,
            lcm,
            degree,
            product,
            active: true,
        });
    }

    // Gebauer--Moeller's old-pair criterion. The candidate vector is indexed
    // by the old basis position, so the two new LCMs are direct lookups.
    pending.retain(|pair| {
        let removed = arena.divides(new_lm, pair.lcm)
            && candidates[pair.left].lcm != pair.lcm
            && candidates[pair.right].lcm != pair.lcm;
        if removed {
            metrics.gebauer_moeller_pairs += 1;
        }
        !removed
    });

    // A proper monomial divisor has strictly smaller total degree. Sorting by
    // degree is therefore sufficient for the minimal-LCM sweep, while the
    // stable arena ID groups equal LCMs without repeated exponent comparisons.
    candidates.sort_unstable_by(compare_pair_candidates);

    // Keep only divisibility-minimal new LCMs.
    for index in 0..candidates.len() {
        if !candidates[index].active {
            continue;
        }
        for earlier in 0..index {
            if candidates[earlier].active
                && candidates[earlier].lcm != candidates[index].lcm
                && arena.divides(candidates[earlier].lcm, candidates[index].lcm)
            {
                candidates[index].active = false;
                metrics.gebauer_moeller_pairs += 1;
                break;
            }
        }
    }

    // Equal LCMs need one representative. If one edge is relatively prime,
    // the product criterion resolves the entire equal-LCM group.
    let mut group_start = 0;
    while group_start < candidates.len() {
        let lcm = candidates[group_start].lcm;
        let mut group_end = group_start + 1;
        while group_end < candidates.len() && candidates[group_end].lcm == lcm {
            group_end += 1;
        }
        let product_group = candidates[group_start..group_end]
            .iter()
            .any(|candidate| candidate.active && candidate.product);
        let mut retained_one = false;
        for candidate in &mut candidates[group_start..group_end] {
            if !candidate.active {
                continue;
            }
            if product_group {
                candidate.active = false;
                metrics.product_criterion_pairs += 1;
                if candidate.product {
                    resolved_pairs.insert(ordered_pair(candidate.left, new_index));
                }
            } else if retained_one {
                candidate.active = false;
                metrics.equal_lcm_pairs += 1;
            } else {
                retained_one = true;
            }
        }
        group_start = group_end;
    }

    for candidate in candidates.into_iter().filter(|candidate| candidate.active) {
        let signature = if options.algorithm == Algorithm::SignatureF4 {
            let left_lm = basis[candidate.left]
                .leading_term()
                .map(|(monomial, _)| monomial)
                .expect("basis members are nonzero");
            let left_multiplier = arena
                .quotient(candidate.lcm, left_lm)
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .expect("lcm is divisible by left leading monomial");
            let right_multiplier = arena
                .quotient(candidate.lcm, new_lm)
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .expect("lcm is divisible by right leading monomial");
            let left_signature = signatures[candidate.left].multiplied(left_multiplier, arena)?;
            let right_signature = signatures[new_index].multiplied(right_multiplier, arena)?;
            if left_signature.compare(&right_signature, arena) == Ordering::Less {
                right_signature
            } else {
                left_signature
            }
        } else {
            signatures[new_index].clone()
        };
        if options.algorithm == Algorithm::SignatureF4
            && known_syzygies
                .iter()
                .any(|syzygy| syzygy.divides(&signature, arena))
        {
            metrics.signature_pruned_pairs += 1;
            continue;
        }
        if pending.len() == options.max_pending_pairs {
            return Err(resource("pending pairs", options.max_pending_pairs));
        }
        pending.push(Pair {
            left: candidate.left,
            right: new_index,
            lcm: candidate.lcm,
            degree: candidate.degree,
            signature,
        });
        metrics.pairs_generated += 1;
    }
    Ok(())
}

fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn chain_criterion<C: Clone + Eq>(
    basis: &[ArenaPolynomial<C>],
    pair: &Pair,
    resolved_pairs: &HashSet<(usize, usize)>,
    arena: &MonomialArena,
) -> bool {
    basis.iter().enumerate().any(|(middle, polynomial)| {
        middle != pair.left
            && middle != pair.right
            && polynomial
                .leading_term()
                .is_some_and(|(leading, _)| arena.divides(leading, pair.lcm))
            && resolved_pairs.contains(&ordered_pair(pair.left, middle))
            && resolved_pairs.contains(&ordered_pair(middle, pair.right))
    })
}

fn monic<F: Field>(
    field: &F,
    polynomial: ArenaPolynomial<F::Element>,
    arena: &MonomialArena,
) -> ArenaPolynomial<F::Element> {
    let Some((_, leading)) = polynomial.leading_term() else {
        return polynomial;
    };
    if leading == &field.one() {
        return polynomial;
    }
    polynomial.scalar_mul(&field.inverse(leading), arena, field)
}

fn s_polynomial<F: Field>(
    field: &F,
    left: &ArenaPolynomial<F::Element>,
    right: &ArenaPolynomial<F::Element>,
    lcm: MonomialId,
    arena: &mut MonomialArena,
) -> Result<ArenaPolynomial<F::Element>, GroebnerError> {
    let (left_lm, left_lc) = left.leading_term().expect("nonzero basis member");
    let (right_lm, right_lc) = right.leading_term().expect("nonzero basis member");
    let left_multiplier = arena
        .quotient(lcm, left_lm)
        .map_err(|_| GroebnerError::ExponentOverflow)?
        .expect("lcm left divisibility");
    let right_multiplier = arena
        .quotient(lcm, right_lm)
        .map_err(|_| GroebnerError::ExponentOverflow)?
        .expect("lcm right divisibility");
    let one = field.one();
    let left_value = if left_lc == &one {
        left.monomial_mul(left_multiplier, arena)
    } else {
        left.term_mul(left_multiplier, &field.inverse(left_lc), arena, field)
    }
    .map_err(|_| GroebnerError::ExponentOverflow)?;
    let right_value = if right_lc == &one {
        right.monomial_mul(right_multiplier, arena)
    } else {
        right.term_mul(right_multiplier, &field.inverse(right_lc), arena, field)
    }
    .map_err(|_| GroebnerError::ExponentOverflow)?;
    Ok(left_value.sub(&right_value, arena, field))
}

type SelectedRows<E> = (Vec<ArenaPolynomial<E>>, Vec<Signature>, Vec<F4RowPlan>);

fn selected_pair_rows<F: Field>(
    field: &F,
    basis: &[ArenaPolynomial<F::Element>],
    signatures: &[Signature],
    mut pairs: Vec<Pair>,
    algorithm: Algorithm,
    arena: &mut MonomialArena,
) -> Result<SelectedRows<F::Element>, GroebnerError> {
    pairs.sort_unstable_by(|left, right| {
        arena
            .compare(left.lcm, right.lcm)
            .then_with(|| left.left.cmp(&right.left))
            .then_with(|| left.right.cmp(&right.right))
    });
    let mut rows = Vec::new();
    let mut row_signatures = Vec::new();
    let mut row_plans = Vec::new();
    let mut group_start = 0;
    while group_start < pairs.len() {
        let lcm = pairs[group_start].lcm;
        let mut group_end = group_start + 1;
        while group_end < pairs.len() && pairs[group_end].lcm == lcm {
            group_end += 1;
        }
        let mut generators = pairs[group_start..group_end]
            .iter()
            .flat_map(|pair| [pair.left, pair.right])
            .collect::<Vec<_>>();
        generators.sort_unstable();
        generators.dedup();

        // One generator multiple is the implicit upper row introduced by
        // symbolic preprocessing. Every other distinct generator multiple is
        // a target. Their differences span exactly the selected S-polynomials
        // while avoiding eager construction of two multiples per pair.
        for generator in generators.into_iter().skip(1) {
            let (leading, leading_coefficient) = basis[generator]
                .leading_term()
                .expect("nonzero basis member");
            let multiplier = arena
                .quotient(lcm, leading)
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .expect("pair lcm is divisible by its generator leading term");
            debug_assert!(leading_coefficient == &field.one());
            rows.push(
                basis[generator]
                    .monomial_mul(multiplier, arena)
                    .map_err(|_| GroebnerError::ExponentOverflow)?,
            );
            if algorithm == Algorithm::SignatureF4 {
                row_signatures.push(signatures[generator].multiplied(multiplier, arena)?);
            }
            row_plans.push(F4RowPlan {
                basis_index: u32::try_from(generator)
                    .expect("an F4 basis index fits the monomial-ID address space"),
                multiplier,
            });
        }
        group_start = group_end;
    }
    Ok((rows, row_signatures, row_plans))
}

pub(crate) fn normal_form<F: Field>(
    field: &F,
    value: ArenaPolynomial<F::Element>,
    basis: &[ArenaPolynomial<F::Element>],
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<ArenaPolynomial<F::Element>, GroebnerError> {
    normal_form_excluding(field, value, basis, None, max_steps, arena)
}

fn normal_form_excluding<F: Field>(
    field: &F,
    value: ArenaPolynomial<F::Element>,
    basis: &[ArenaPolynomial<F::Element>],
    excluded_reducer: Option<usize>,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<ArenaPolynomial<F::Element>, GroebnerError> {
    let divisor_index = LeadingDivisorIndex::new_excluding(basis, excluded_reducer, arena);
    normal_form_with_divisor_index(field, value, basis, &divisor_index, max_steps, arena)
}

fn normal_form_with_divisor_index<F: Field>(
    field: &F,
    value: ArenaPolynomial<F::Element>,
    basis: &[ArenaPolynomial<F::Element>],
    divisor_index: &LeadingDivisorIndex,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<ArenaPolynomial<F::Element>, GroebnerError> {
    let mut remainder_terms = Vec::new();
    let mut steps = 0;
    let one = field.one();
    let mut value = PolynomialAccumulator::from_polynomial(&value, arena);
    while let Some((leading_monomial, leading_coefficient)) = value.leading_term(arena) {
        if steps == max_steps {
            return Err(resource("normal-form steps", max_steps));
        }
        if let Some(reducer_index) = divisor_index.find(leading_monomial, arena) {
            let reducer = &basis[reducer_index];
            let (reducer_monomial, reducer_coefficient) =
                reducer.leading_term().expect("nonzero reducer");
            let multiplier = arena
                .quotient(leading_monomial, reducer_monomial)
                .map_err(|_| GroebnerError::ExponentOverflow)?
                .expect("selected reducer divisibility");
            let scalar = if reducer_coefficient == &one {
                leading_coefficient
            } else {
                field.mul(&leading_coefficient, &field.inverse(reducer_coefficient))
            };
            value
                .subtract_multiple(reducer, multiplier, &scalar, arena, field)
                .map_err(|_| GroebnerError::ExponentOverflow)?;
        } else {
            remainder_terms.push(
                value
                    .pop_leading(arena)
                    .expect("the observed leading accumulator term remains present"),
            );
        }
        steps += 1;
    }
    Ok(ArenaPolynomial::from_sorted_terms(
        arena,
        remainder_terms,
        field,
    ))
}

fn normalized_generators<F: Field>(
    field: &F,
    generators: Vec<ArenaPolynomial<F::Element>>,
    arena: &MonomialArena,
) -> Vec<ArenaPolynomial<F::Element>> {
    let mut normalized = Vec::with_capacity(generators.len());
    for generator in generators {
        if generator.is_zero() {
            continue;
        }
        let monic = monic(field, generator, arena);
        if !normalized.contains(&monic) {
            normalized.push(monic);
        }
    }
    sort_basis(&mut normalized, arena);
    normalized
}

pub(crate) fn interreduce<F: Field>(
    field: &F,
    basis: Vec<ArenaPolynomial<F::Element>>,
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<Vec<ArenaPolynomial<F::Element>>, GroebnerError> {
    let basis = normalized_generators(field, basis, arena);
    let mut minimal = Vec::with_capacity(basis.len());
    for (index, polynomial) in basis.iter().enumerate() {
        let leading = polynomial
            .leading_term()
            .map(|(monomial, _)| monomial)
            .expect("normalized generators are nonzero");
        let redundant = basis.iter().enumerate().any(|(other_index, other)| {
            if other_index == index {
                return false;
            }
            let other_leading = other
                .leading_term()
                .map(|(monomial, _)| monomial)
                .expect("normalized generators are nonzero");
            arena.divides(other_leading, leading)
                && (other_leading != leading || other_index < index)
        });
        if !redundant {
            minimal.push(polynomial.clone());
        }
    }

    let mut reduced = Vec::with_capacity(minimal.len());
    for index in 0..minimal.len() {
        let remainder = normal_form_excluding(
            field,
            minimal[index].clone(),
            &minimal,
            Some(index),
            max_steps,
            arena,
        )?;
        if !remainder.is_zero() {
            let monic = monic(field, remainder, arena);
            if !reduced.contains(&monic) {
                reduced.push(monic);
            }
        }
    }
    sort_basis(&mut reduced, arena);
    Ok(reduced)
}

fn sort_basis<C: Clone + Eq>(basis: &mut [ArenaPolynomial<C>], arena: &MonomialArena) {
    basis.sort_by(|left, right| {
        arena.compare(
            right.leading_term().expect("nonzero").0,
            left.leading_term().expect("nonzero").0,
        )
    });
}

struct LeadingDivisorIndex {
    entries: Vec<(MonomialId, usize, u64, u64)>,
}

impl LeadingDivisorIndex {
    fn new<C: Clone + Eq>(basis: &[ArenaPolynomial<C>], arena: &MonomialArena) -> Self {
        Self::new_excluding(basis, None, arena)
    }

    fn new_excluding<C: Clone + Eq>(
        basis: &[ArenaPolynomial<C>],
        excluded: Option<usize>,
        arena: &MonomialArena,
    ) -> Self {
        Self {
            entries: basis
                .iter()
                .enumerate()
                .filter(|(index, _)| Some(*index) != excluded)
                .filter_map(|(index, polynomial)| {
                    polynomial.leading_term().map(|(leading, _)| {
                        (
                            leading,
                            index,
                            arena.total_degree(leading),
                            arena.divisor_mask(leading),
                        )
                    })
                })
                .collect(),
        }
    }

    fn find(&self, monomial: MonomialId, arena: &MonomialArena) -> Option<usize> {
        let degree = arena.total_degree(monomial);
        let mask = arena.divisor_mask(monomial);
        self.entries
            .iter()
            .find(|(leading, _, leading_degree, leading_mask)| {
                *leading_degree <= degree
                    && *leading_mask & !mask == 0
                    && arena.divides(*leading, monomial)
            })
            .map(|(_, index, _, _)| *index)
    }
}

fn symbolic_preprocess_split<F: Field>(
    _field: &F,
    basis: &[ArenaPolynomial<F::Element>],
    targets: Vec<ArenaPolynomial<F::Element>>,
    max_workspace_bytes: usize,
    arena: &mut MonomialArena,
) -> Result<SymbolicRows<F::Element>, GroebnerError> {
    let mut visited = MonomialMarks::new(arena);
    let mut queue = VecDeque::new();
    for monomial in targets.iter().flat_map(ArenaPolynomial::support) {
        if visited.insert(arena, *monomial) {
            queue.push_back(*monomial);
        }
    }
    let mut reducers = Vec::new();
    let mut reducer_plans = Vec::new();
    let divisor_index = LeadingDivisorIndex::new(basis, arena);
    let (basis_structure_bytes, basis_coefficient_bytes) = polynomial_workspace_bytes(basis);
    let (target_structure_bytes, target_coefficient_bytes) = polynomial_workspace_bytes(&targets);
    let retained_bytes = basis_structure_bytes
        .saturating_add(basis_coefficient_bytes)
        .saturating_add(target_structure_bytes)
        .saturating_add(target_coefficient_bytes);
    let mut reducer_buffer_bytes = 0_usize;
    let initial_symbolic_bytes = retained_bytes
        .saturating_add(visited.allocated_bytes())
        .saturating_add(
            queue
                .capacity()
                .saturating_mul(core::mem::size_of::<MonomialId>()),
        )
        .saturating_add(
            divisor_index
                .entries
                .capacity()
                .saturating_mul(core::mem::size_of::<(MonomialId, usize, u64, u64)>()),
        )
        .saturating_add(arena.allocated_bytes());
    if initial_symbolic_bytes == usize::MAX || initial_symbolic_bytes > max_workspace_bytes {
        return Err(resource("workspace bytes", max_workspace_bytes));
    }
    while let Some(monomial) = queue.pop_front() {
        let Some(index) = divisor_index.find(monomial, arena) else {
            continue;
        };
        let polynomial = &basis[index];
        let multiplier = arena
            .quotient(
                monomial,
                polynomial.leading_term().expect("nonzero reducer").0,
            )
            .map_err(|_| GroebnerError::ExponentOverflow)?
            .expect("selected reducer divides monomial");
        let row = polynomial
            .monomial_mul(multiplier, arena)
            .map_err(|_| GroebnerError::ExponentOverflow)?;
        for term in row.support() {
            if visited.insert(arena, *term) {
                queue.push_back(*term);
            }
        }
        reducer_buffer_bytes = reducer_buffer_bytes.saturating_add(row.allocated_bytes());
        reducers.push(row);
        reducer_plans.push(F4RowPlan {
            basis_index: u32::try_from(index)
                .expect("an F4 basis index fits the monomial-ID address space"),
            multiplier,
        });
        let symbolic_bytes = retained_bytes
            .saturating_add(reducer_buffer_bytes)
            .saturating_add(
                reducers
                    .capacity()
                    .saturating_mul(core::mem::size_of::<ArenaPolynomial<F::Element>>()),
            )
            .saturating_add(
                reducer_plans
                    .capacity()
                    .saturating_mul(core::mem::size_of::<F4RowPlan>()),
            )
            .saturating_add(visited.allocated_bytes())
            .saturating_add(
                queue
                    .capacity()
                    .saturating_mul(core::mem::size_of::<MonomialId>()),
            )
            .saturating_add(
                divisor_index
                    .entries
                    .capacity()
                    .saturating_mul(core::mem::size_of::<(MonomialId, usize, u64, u64)>()),
            )
            .saturating_add(arena.allocated_bytes());
        if symbolic_bytes == usize::MAX || symbolic_bytes > max_workspace_bytes {
            return Err(resource("workspace bytes", max_workspace_bytes));
        }
    }
    Ok(SymbolicRows {
        targets,
        reducers,
        reducer_plans,
    })
}

fn matrix_columns<C: Clone + Eq>(
    rows: &[ArenaPolynomial<C>],
    arena: &MonomialArena,
) -> Vec<MonomialId> {
    let mut seen = MonomialMarks::new(arena);
    let mut columns = Vec::new();
    for monomial in rows.iter().flat_map(ArenaPolynomial::support) {
        if seen.insert(arena, *monomial) {
            columns.push(*monomial);
        }
    }
    columns.sort_unstable_by(|left, right| arena.compare(*right, *left));
    columns
}

fn assemble_f4_work<C: Clone + Eq>(
    rows: &[ArenaPolynomial<C>],
    columns: &[MonomialId],
    target_count: usize,
    arena: &MonomialArena,
) -> F4RowWork<C> {
    debug_assert!(target_count <= rows.len());
    let mut indices = vec![u32::MAX; arena.len()];
    for (index, monomial) in columns.iter().enumerate() {
        indices[monomial.index()] =
            u32::try_from(index).expect("an F4 column fits the monomial-ID address space");
    }
    let mut work = F4RowWork::with_expected_rows(target_count, rows.len() - target_count);
    for (row_index, polynomial) in rows.iter().enumerate() {
        let row = polynomial
            .iter()
            .map(|(monomial, coefficient)| (indices[monomial.index()], coefficient.clone()))
            .collect();
        if row_index < target_count {
            work.push_target(row);
        } else {
            work.push_reducer(row);
        }
    }
    work
}

fn polynomial_from_row<F: Field>(
    field: &F,
    columns: &[MonomialId],
    row: Vec<(u32, F::Element)>,
    arena: &MonomialArena,
) -> ArenaPolynomial<F::Element> {
    ArenaPolynomial::from_sorted_terms(
        arena,
        row.into_iter()
            .filter(|(_, coefficient)| !field.is_zero(coefficient))
            .map(|(column, coefficient)| (columns[column as usize], coefficient)),
        field,
    )
}

#[cfg(test)]
fn verify<F: Field>(
    field: &F,
    basis: &[SparsePolynomial<F::Element>],
    max_steps: usize,
) -> Result<(), GroebnerError> {
    let (arity, order) = validate_inputs(basis)?;
    let mut arena = MonomialArena::new(arity, order);
    let basis = basis
        .iter()
        .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, field))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| GroebnerError::ExponentOverflow)?;
    verify_compact(field, &basis, max_steps, &mut arena)
}

pub(crate) fn verify_compact<F: Field>(
    field: &F,
    basis: &[ArenaPolynomial<F::Element>],
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    verify_monic_and_reduced(field, basis, arena)?;
    let pairs = verification_pairs(basis, arena)?;
    verify_compact_pairs(field, basis, &pairs, max_steps, arena)
}

fn verify_compact_pairs<F: Field>(
    field: &F,
    basis: &[ArenaPolynomial<F::Element>],
    pairs: &[(u64, usize, usize, MonomialId)],
    max_steps: usize,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    for (_, left, right, lcm) in pairs {
        let s = s_polynomial(field, &basis[*left], &basis[*right], *lcm, arena)?;
        if !normal_form(field, s, basis, max_steps, arena)?.is_zero() {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

pub(crate) fn verify_monic_and_reduced<F: Field>(
    field: &F,
    basis: &[ArenaPolynomial<F::Element>],
    arena: &MonomialArena,
) -> Result<(), GroebnerError> {
    for polynomial in basis {
        let (_, leading) = polynomial
            .leading_term()
            .ok_or(GroebnerError::VerificationFailed)?;
        if leading != &field.one() {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    for (index, polynomial) in basis.iter().enumerate() {
        for monomial in polynomial.support() {
            if basis.iter().enumerate().any(|(other, candidate)| {
                other != index
                    && candidate
                        .leading_term()
                        .is_some_and(|(leading, _)| arena.divides(leading, *monomial))
            }) {
                return Err(GroebnerError::VerificationFailed);
            }
        }
    }
    Ok(())
}

pub(crate) fn verification_pairs<C: Clone + Eq>(
    basis: &[ArenaPolynomial<C>],
    arena: &mut MonomialArena,
) -> Result<Vec<(u64, usize, usize, MonomialId)>, GroebnerError> {
    let mut candidates = Vec::with_capacity(basis.len().saturating_mul(basis.len()) / 2);
    for (left, left_polynomial) in basis.iter().enumerate() {
        let left_leading = left_polynomial
            .leading_term()
            .expect("nonzero basis member")
            .0;
        for (right, right_polynomial) in basis.iter().enumerate().skip(left + 1) {
            let right_leading = right_polynomial
                .leading_term()
                .expect("nonzero basis member")
                .0;
            let lcm = arena
                .lcm(left_leading, right_leading)
                .map_err(|_| GroebnerError::ExponentOverflow)?;
            candidates.push((arena.total_degree(lcm), left, right, lcm));
        }
    }
    candidates.sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| arena.compare(left.3, right.3))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let mut resolved = HashSet::<(usize, usize)>::default();
    let mut required = Vec::with_capacity(candidates.len());
    for candidate @ (_, left, right, lcm) in candidates {
        let left_leading = basis[left].leading_term().expect("nonzero basis member").0;
        let right_leading = basis[right].leading_term().expect("nonzero basis member").0;
        let product = Some(arena.total_degree(lcm))
            == arena
                .total_degree(left_leading)
                .checked_add(arena.total_degree(right_leading));
        let key = ordered_pair(left, right);
        if product || verification_chain_criterion(left, right, lcm, basis, &resolved, arena) {
            resolved.insert(key);
            continue;
        }
        required.push(candidate);
        // Later chain decisions may depend on this pair. The complete
        // verifier succeeds only after every retained predecessor succeeds,
        // so recording the dependency here is an induction over this order,
        // not an assumption about its coefficient values.
        resolved.insert(key);
    }
    Ok(required)
}

fn verification_chain_criterion<C: Clone + Eq>(
    left: usize,
    right: usize,
    lcm: MonomialId,
    basis: &[ArenaPolynomial<C>],
    resolved: &HashSet<(usize, usize)>,
    arena: &MonomialArena,
) -> bool {
    basis.iter().enumerate().any(|(middle, polynomial)| {
        middle != left
            && middle != right
            && polynomial
                .leading_term()
                .is_some_and(|(leading, _)| arena.divides(leading, lcm))
            && resolved.contains(&ordered_pair(left, middle))
            && resolved.contains(&ordered_pair(middle, right))
    })
}

fn verify_batched<F, R>(
    field: &F,
    basis: &[ArenaPolynomial<F::Element>],
    pairs: &[(u64, usize, usize, MonomialId)],
    options: &GroebnerOptions,
    reducer: &R,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    for batch in pairs.chunks(512) {
        verify_pair_batch(field, basis, batch, options, reducer, arena)?;
    }
    Ok(())
}

fn verify_pair_batch<F, R>(
    field: &F,
    basis: &[ArenaPolynomial<F::Element>],
    pairs: &[(u64, usize, usize, MonomialId)],
    options: &GroebnerOptions,
    reducer: &R,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    let targets = pairs
        .iter()
        .map(|(_, left, right, lcm)| {
            s_polynomial(field, &basis[*left], &basis[*right], *lcm, arena)
        })
        .filter_map(|result| match result {
            Ok(polynomial) if polynomial.is_zero() => None,
            other => Some(other),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if targets.is_empty() {
        return Ok(());
    }
    let symbolic = match symbolic_preprocess_split(
        field,
        basis,
        targets,
        options.max_workspace_bytes,
        arena,
    ) {
        Ok(symbolic) => symbolic,
        Err(GroebnerError::ResourceLimit { .. }) if pairs.len() > 1 => {
            return split_verification_batch(field, basis, pairs, options, reducer, arena);
        }
        Err(error) => return Err(error),
    };
    let SymbolicRows {
        targets: mut rows,
        reducers,
        ..
    } = symbolic;
    let target_count = rows.len();
    let reducer_count = reducers.len();
    let distinct_leading = reducers
        .iter()
        .filter_map(|polynomial| polynomial.leading_term().map(|(monomial, _)| monomial))
        .collect::<HashSet<_>>()
        .len();
    if distinct_leading != reducer_count {
        return Err(GroebnerError::VerificationFailed);
    }
    rows.extend(reducers);
    if rows.len() > options.max_matrix_rows {
        if pairs.len() == 1 {
            return Err(resource(
                "verification matrix rows",
                options.max_matrix_rows,
            ));
        }
        return split_verification_batch(field, basis, pairs, options, reducer, arena);
    }
    let columns = matrix_columns(&rows, arena);
    if columns.len() > options.max_matrix_columns {
        if pairs.len() == 1 {
            return Err(resource(
                "verification matrix columns",
                options.max_matrix_columns,
            ));
        }
        return split_verification_batch(field, basis, pairs, options, reducer, arena);
    }
    let pair_bytes = vector_workspace_bytes::<(u64, usize, usize, MonomialId)>(pairs.len());
    let mut workspace = GroebnerMetrics::default();
    if let Err(error) = check_matrix_assembly_workspace(
        arena,
        basis,
        &rows,
        &columns,
        pair_bytes,
        0,
        0,
        options,
        &mut workspace,
    ) {
        if pairs.len() > 1 {
            return split_verification_batch(field, basis, pairs, options, reducer, arena);
        }
        return Err(error);
    }
    let matrix = assemble_f4_work(&rows, &columns, target_count, arena);
    if let Err(error) = observe_f4_workspace(
        field,
        arena,
        basis,
        &rows,
        &columns,
        &matrix,
        pair_bytes,
        0,
        0,
        reducer,
        options,
        &mut workspace,
    ) {
        if pairs.len() > 1 {
            return split_verification_batch(field, basis, pairs, options, reducer, arena);
        }
        return Err(error);
    }
    if reducer.targets_in_span(field, columns.len(), matrix) {
        Ok(())
    } else {
        Err(GroebnerError::VerificationFailed)
    }
}

fn split_verification_batch<F, R>(
    field: &F,
    basis: &[ArenaPolynomial<F::Element>],
    pairs: &[(u64, usize, usize, MonomialId)],
    options: &GroebnerOptions,
    reducer: &R,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    debug_assert!(pairs.len() > 1);
    let middle = pairs.len() / 2;
    verify_pair_batch(field, basis, &pairs[..middle], options, reducer, arena)?;
    verify_pair_batch(field, basis, &pairs[middle..], options, reducer, arena)
}

pub(crate) fn verify_output<F, R>(
    field: &F,
    generators: &[ArenaPolynomial<F::Element>],
    basis: &[ArenaPolynomial<F::Element>],
    options: &GroebnerOptions,
    reducer: &R,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError>
where
    F: Field,
    R: RowReducer<F>,
{
    verify_monic_and_reduced(field, basis, arena)?;
    let pairs = verification_pairs(basis, arena)?;
    if pairs.len() < 24 {
        verify_compact_pairs(field, basis, &pairs, options.max_reduction_steps, arena)?;
    } else {
        verify_batched(field, basis, &pairs, options, reducer, arena)?;
    }
    for generator in generators {
        if !normal_form(
            field,
            generator.clone(),
            basis,
            options.max_reduction_steps,
            arena,
        )?
        .is_zero()
        {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

pub(crate) fn verify_construction_output<F: Field>(
    field: &F,
    generators: &[ArenaPolynomial<F::Element>],
    basis: &[ArenaPolynomial<F::Element>],
    options: &GroebnerOptions,
    arena: &mut MonomialArena,
) -> Result<(), GroebnerError> {
    verify_monic_and_reduced(field, basis, arena)?;
    for generator in generators {
        if !normal_form(
            field,
            generator.clone(),
            basis,
            options.max_reduction_steps,
            arena,
        )?
        .is_zero()
        {
            return Err(GroebnerError::VerificationFailed);
        }
    }
    Ok(())
}

fn resource(resource: &'static str, limit: usize) -> GroebnerError {
    GroebnerError::ResourceLimit { resource, limit }
}

#[cfg(test)]
mod tests {
    use flint_core::Rational;
    use poly_core::Monomial;
    use proptest::prelude::*;

    use super::*;

    fn property_config(default_cases: u32) -> ProptestConfig {
        let configured = ProptestConfig::default();
        if std::env::var_os("PROPTEST_CASES").is_some() {
            configured
        } else {
            ProptestConfig {
                cases: default_cases,
                ..configured
            }
        }
    }

    fn monomial(values: &[u32]) -> Monomial {
        Monomial::new(values.to_vec().into_boxed_slice())
    }

    fn prime_polynomial(field: &PrimeField32, terms: &[(u32, &[u32])]) -> SparsePolynomial<u32> {
        SparsePolynomial::new(
            2,
            MonomialOrder::GradedReverseLexicographic,
            terms
                .iter()
                .map(|(coefficient, exponent)| (monomial(exponent), *coefficient)),
            field,
        )
    }

    fn cyclic(size: usize, field: &PrimeField32) -> Vec<SparsePolynomial<u32>> {
        let mut generators = Vec::with_capacity(size);
        for degree in 1..size {
            generators.push(SparsePolynomial::new(
                size,
                MonomialOrder::GradedReverseLexicographic,
                (0..size).map(|start| {
                    let mut exponent = vec![0_u32; size];
                    for offset in 0..degree {
                        exponent[(start + offset) % size] += 1;
                    }
                    (Monomial::new(exponent.into_boxed_slice()), 1_u32)
                }),
                field,
            ));
        }
        generators.push(SparsePolynomial::new(
            size,
            MonomialOrder::GradedReverseLexicographic,
            [
                (Monomial::new(vec![1_u32; size].into_boxed_slice()), 1_u32),
                (Monomial::one(size), field.modulus() - 1),
            ],
            field,
        ));
        generators
    }

    #[test]
    fn prime_f4_computes_and_verifies_a_nontrivial_basis() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[1, 1]), (1, &[0, 0])]),
        ];
        let output =
            groebner_basis_prime(&field, &generators, &GroebnerOptions::default()).unwrap();
        let (profiled, profile) =
            groebner_basis_prime_profiled(&field, &generators, &GroebnerOptions::default())
                .unwrap();
        assert_eq!(profiled, output);
        assert_eq!(profile.rational_reconstruction, Duration::ZERO);
        assert!(!output.basis.is_empty());
        assert!(output.metrics.rounds > 0);
        assert!(output.metrics.minimum_round_degree.is_some());
        assert!(output.metrics.maximum_round_degree.is_some());
        assert!(output.metrics.round_degree_sum > 0);
        assert!(output.metrics.peak_matrix_nonzeros > 0);
        assert!(output.metrics.peak_monomial_bytes > 0);
        assert!(output.metrics.peak_pair_bytes > 0);
        assert!(output.metrics.peak_sparse_row_bytes > 0);
        assert!(output.metrics.peak_dense_row_bytes > 0);
        assert!(output.metrics.peak_coefficient_bytes > 0);
        assert!(output.metrics.peak_workspace_bytes > 0);
        assert_eq!(output.evidence, GroebnerEvidence::ExactConstruction);
        verify(&field, &output.basis, 1_000_000).unwrap();
    }

    #[test]
    fn polynomial_normal_forms_share_reducers_and_match_hand_remainders() {
        let field = PrimeField32::new(101).unwrap();
        let basis = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (100, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[0, 2])]),
        ];
        let values = vec![
            prime_polynomial(&field, &[(1, &[4, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[2, 0]), (3, &[0, 0])]),
        ];
        let expected = vec![
            prime_polynomial(&field, &[(1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[0, 1]), (3, &[0, 0])]),
        ];

        assert_eq!(
            polynomial_normal_forms(&field, &basis, &values, &GroebnerOptions::default()).unwrap(),
            expected
        );
        assert_eq!(
            polynomial_normal_forms(&field, &[], &values, &GroebnerOptions::default()).unwrap(),
            values
        );
    }

    #[test]
    fn prepared_polynomial_reducers_reuse_workspaces_and_support_concurrent_callers() {
        let field = PrimeField32::new(101).unwrap();
        let basis = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (100, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[0, 2])]),
        ];
        let values = vec![
            prime_polynomial(&field, &[(1, &[4, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[2, 0]), (3, &[0, 0])]),
        ];
        let expected =
            polynomial_normal_forms(&field, &basis, &values, &GroebnerOptions::default()).unwrap();
        let prepared = std::sync::Arc::new(
            PreparedPolynomialReducer::new(
                &field,
                2,
                MonomialOrder::GradedReverseLexicographic,
                &basis,
            )
            .unwrap(),
        );

        assert_eq!(
            prepared.normal_forms(&field, &values, 1_000_000).unwrap(),
            expected
        );
        assert_eq!(
            prepared.normal_forms(&field, &values, 1_000_000).unwrap(),
            expected
        );
        let workers = (0..4)
            .map(|_| {
                let prepared = std::sync::Arc::clone(&prepared);
                let values = values.clone();
                let expected = expected.clone();
                std::thread::spawn(move || {
                    let field = PrimeField32::new(101).unwrap();
                    assert_eq!(
                        prepared.normal_forms(&field, &values, 1_000_000).unwrap(),
                        expected
                    );
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }

        let empty = PreparedPolynomialReducer::new(
            &field,
            2,
            MonomialOrder::GradedReverseLexicographic,
            &[],
        )
        .unwrap();
        assert_eq!(empty.normal_forms(&field, &values, 10).unwrap(), values);
    }

    #[test]
    fn hilbert_guidance_is_independently_verified_and_falls_back_when_wrong() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[1, 1]), (1, &[0, 0])]),
        ];
        let ordinary =
            groebner_basis_prime(&field, &generators, &GroebnerOptions::default()).unwrap();
        let final_leaders = ordinary
            .basis
            .iter()
            .map(|polynomial| polynomial.terms()[0].0.clone())
            .collect::<Vec<_>>();
        let target = crate::hilbert::standard_hilbert_numerator(2, &final_leaders).unwrap();
        let guided = groebner_basis_prime(
            &field,
            &generators,
            &GroebnerOptions {
                hilbert_numerator_target: Some(target),
                ..GroebnerOptions::default()
            },
        )
        .unwrap();
        assert_eq!(guided.basis, ordinary.basis);
        assert_eq!(
            guided.evidence,
            GroebnerEvidence::IndependentBuchbergerSweep
        );
        assert!(guided.metrics.hilbert_target_checks > 1);
        assert!(!guided.metrics.hilbert_target_fallback);

        let input_leaders = generators
            .iter()
            .map(|polynomial| polynomial.terms()[0].0.clone())
            .collect::<Vec<_>>();
        let wrong_target = crate::hilbert::standard_hilbert_numerator(2, &input_leaders).unwrap();
        let recovered = groebner_basis_prime(
            &field,
            &generators,
            &GroebnerOptions {
                hilbert_numerator_target: Some(wrong_target),
                ..GroebnerOptions::default()
            },
        )
        .unwrap();
        assert_eq!(recovered.basis, ordinary.basis);
        assert!(recovered.metrics.hilbert_target_fallback);
    }

    #[test]
    fn f4_workspace_cap_fails_before_row_reduction() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[1, 1]), (1, &[0, 0])]),
        ];
        let options = GroebnerOptions {
            max_workspace_bytes: 1,
            ..GroebnerOptions::default()
        };
        assert_eq!(
            groebner_basis_prime(&field, &generators, &options),
            Err(GroebnerError::ResourceLimit {
                resource: "workspace bytes",
                limit: 1,
            })
        );
    }

    #[test]
    fn tracked_f4_returns_verified_input_multipliers_for_one() {
        let field = PrimeField32::new(7).unwrap();
        let x = prime_polynomial(&field, &[(1, &[1, 0]), (0, &[0, 0])]);
        let x_plus_one = prime_polynomial(&field, &[(1, &[1, 0]), (1, &[0, 0])]);
        let output = unit_ideal_certificate(&field, &[x, x_plus_one], &GroebnerOptions::default())
            .unwrap()
            .expect("x and x + 1 generate one");
        assert_eq!(output.multipliers.len(), 2);
        assert!(output.metrics.generic_row_reductions > 0);
    }

    #[test]
    fn tracked_f4_distinguishes_a_nonunit_ideal() {
        let field = PrimeField32::new(7).unwrap();
        let x = prime_polynomial(&field, &[(1, &[1, 0])]);
        assert!(
            unit_ideal_certificate(&field, &[x], &GroebnerOptions::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn tracked_f4_returns_the_same_reduced_prime_basis_with_a_change_matrix() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[1, 1]), (1, &[0, 0])]),
        ];
        let options = GroebnerOptions::default();
        let plain = groebner_basis(&field, &generators, &options).unwrap();
        let tracked = groebner_basis_with_transform(&field, &generators, &options).unwrap();
        assert_eq!(tracked.basis, plain.basis);
        assert_eq!(tracked.transformations.len(), tracked.basis.len());
        assert!(tracked
            .transformations
            .iter()
            .all(|row| row.len() == generators.len()));
        assert!(tracked.metrics.generic_row_reductions > 0);
        assert_eq!(tracked.evidence, GroebnerEvidence::ExactConstruction);
    }

    #[test]
    fn tracked_f4_returns_the_same_reduced_rational_basis() {
        let field = crate::rational::RationalCoefficientField;
        let polynomial = |terms: &[(i64, &[u32])]| {
            SparsePolynomial::new(
                2,
                MonomialOrder::GradedReverseLexicographic,
                terms.iter().map(|(coefficient, exponent)| {
                    (monomial(exponent), Rational::from(*coefficient))
                }),
                &field,
            )
        };
        let generators = vec![
            polynomial(&[(1, &[2, 0]), (1, &[0, 1])]),
            polynomial(&[(1, &[1, 1]), (1, &[0, 0])]),
        ];
        let options = GroebnerOptions::default();
        let plain = groebner_basis(&field, &generators, &options).unwrap();
        let tracked = groebner_basis_with_transform(&field, &generators, &options).unwrap();
        assert_eq!(tracked.basis, plain.basis);
        assert_eq!(tracked.transformations.len(), tracked.basis.len());
    }

    #[test]
    fn cyclic_six_reaches_the_abcd_backend() {
        let field = PrimeField32::new(65_521).unwrap();
        let output =
            groebner_basis_prime(&field, &cyclic(6, &field), &GroebnerOptions::default()).unwrap();
        assert!(output.metrics.prime_abcd_row_reductions > 0);
        assert!(output.metrics.abcd_left_columns > 0);
        assert!(output.metrics.abcd_right_columns > 0);
        assert!(output.metrics.abcd_dense_cells > 0);
        assert!(output.metrics.abcd_dense_cells < output.metrics.abcd_full_dense_cells);
    }

    #[test]
    fn learned_prime_trace_replays_and_rejects_changed_support() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[1, 1]), (1, &[0, 0])]),
        ];
        let options = GroebnerOptions::default();
        let (learned, mut session) = learn_prime_trace(&field, &generators, &options).unwrap();
        let replayed = apply_prime_trace(&mut session, &field, &generators, &options)
            .unwrap()
            .expect("the identical support is compatible with its learned trace");
        assert_eq!(replayed.basis, learned.basis);
        assert!(replayed.metrics.trace_application_rounds > 0);
        assert!(replayed.metrics.trace_replayed_rows > 0);

        let changed = vec![
            prime_polynomial(&field, &[(1, &[2, 0])]),
            generators[1].clone(),
        ];
        assert!(apply_prime_trace(&mut session, &field, &changed, &options)
            .unwrap()
            .is_none());
    }

    #[test]
    fn batched_prime_verification_agrees_with_direct_normal_forms() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[1, 1]), (1, &[0, 0])]),
        ];
        let options = GroebnerOptions::default();
        let output = groebner_basis_prime(&field, &generators, &options).unwrap();
        verify(&field, &output.basis, options.max_reduction_steps).unwrap();
        let mut arena = MonomialArena::new(2, MonomialOrder::GradedReverseLexicographic);
        let compact_basis = output
            .basis
            .iter()
            .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field).unwrap())
            .collect::<Vec<_>>();
        let pairs = verification_pairs(&compact_basis, &mut arena).unwrap();
        verify_batched(
            &field,
            &compact_basis,
            &pairs,
            &options,
            &PrimeReducer,
            &mut arena,
        )
        .unwrap();

        assert_eq!(
            verify(&field, &generators, options.max_reduction_steps),
            Err(GroebnerError::VerificationFailed)
        );
        let mut invalid_arena = MonomialArena::new(2, MonomialOrder::GradedReverseLexicographic);
        let compact_generators = generators
            .iter()
            .map(|polynomial| {
                ArenaPolynomial::from_sparse(&mut invalid_arena, polynomial, &field).unwrap()
            })
            .collect::<Vec<_>>();
        let invalid_pairs = verification_pairs(&compact_generators, &mut invalid_arena).unwrap();
        assert_eq!(
            verify_batched(
                &field,
                &compact_generators,
                &invalid_pairs,
                &options,
                &PrimeReducer,
                &mut invalid_arena,
            ),
            Err(GroebnerError::VerificationFailed)
        );
    }

    #[test]
    fn f2_lane_uses_coefficient_free_inputs() {
        let order = MonomialOrder::GradedReverseLexicographic;
        let generators = vec![
            SparsePolynomial::new(
                2,
                order,
                [(monomial(&[2, 0]), 1_u8), (monomial(&[0, 1]), 1_u8)],
                &BinaryField,
            ),
            SparsePolynomial::new(
                2,
                order,
                [(monomial(&[0, 2]), 1_u8), (monomial(&[1, 0]), 1_u8)],
                &BinaryField,
            ),
        ];
        let output = groebner_basis_f2(&generators, &GroebnerOptions::default()).unwrap();
        assert!(!output.basis.is_empty());
        assert!(output
            .basis
            .iter()
            .all(|polynomial| polynomial.arity() == 2 && polynomial.order() == order));
    }

    #[test]
    fn generic_and_specialized_prime_lanes_agree() {
        let field = PrimeField32::new(7).unwrap();
        let generators = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[0, 2]), (1, &[1, 0])]),
        ];
        let generic = groebner_basis(&field, &generators, &GroebnerOptions::default()).unwrap();
        let specialized =
            groebner_basis_prime(&field, &generators, &GroebnerOptions::default()).unwrap();
        assert_eq!(generic.basis, specialized.basis);
    }

    #[test]
    fn pair_candidate_order_places_every_proper_divisor_first() {
        let exponents = [
            [0_u32, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [2, 0, 0],
            [2, 1, 0],
            [1, 1, 1],
            [3, 2, 1],
        ];
        for order in [
            MonomialOrder::Lexicographic,
            MonomialOrder::GradedLexicographic,
            MonomialOrder::GradedReverseLexicographic,
        ] {
            let mut arena = MonomialArena::new(3, order);
            let mut candidates = Vec::new();
            for (left, values) in exponents.iter().enumerate() {
                let lcm = arena.intern_exponents(values).unwrap();
                candidates.push(PairCandidate {
                    left,
                    lcm,
                    degree: arena.total_degree(lcm),
                    product: false,
                    active: true,
                });
            }
            candidates.sort_unstable_by(compare_pair_candidates);
            for (divisor_position, divisor) in candidates.iter().enumerate() {
                for (dividend_position, dividend) in candidates.iter().enumerate() {
                    if divisor.lcm != dividend.lcm && arena.divides(divisor.lcm, dividend.lcm) {
                        assert!(divisor_position < dividend_position);
                    }
                }
            }
        }
    }

    #[test]
    fn verification_pairs_apply_product_and_dependency_safe_chain_criteria() {
        let field = PrimeField32::new(101).unwrap();
        let sparse = [
            prime_polynomial(&field, &[(1, &[2, 0])]),
            prime_polynomial(&field, &[(1, &[1, 1])]),
            prime_polynomial(&field, &[(1, &[0, 2])]),
        ];
        let mut arena = MonomialArena::new(2, MonomialOrder::GradedReverseLexicographic);
        let basis = sparse
            .iter()
            .map(|polynomial| ArenaPolynomial::from_sparse(&mut arena, polynomial, &field).unwrap())
            .collect::<Vec<_>>();

        let required = verification_pairs(&basis, &mut arena).unwrap();
        assert_eq!(required.len(), 2);
        assert!(required
            .iter()
            .all(|(_, left, right, _)| { (*left, *right) == (0, 1) || (*left, *right) == (1, 2) }));

        let relatively_prime = vec![basis[0].clone(), basis[2].clone()];
        assert!(verification_pairs(&relatively_prime, &mut arena)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn signature_mode_is_verified() {
        let field = PrimeField32::new(101).unwrap();
        let generators = vec![
            prime_polynomial(&field, &[(1, &[2, 0]), (1, &[0, 1])]),
            prime_polynomial(&field, &[(1, &[0, 2]), (1, &[1, 0])]),
            prime_polynomial(&field, &[(1, &[1, 1]), (1, &[0, 0])]),
        ];
        let ordinary =
            groebner_basis_prime(&field, &generators, &GroebnerOptions::default()).unwrap();
        let options = GroebnerOptions {
            algorithm: Algorithm::SignatureF4,
            ..GroebnerOptions::default()
        };
        let output = groebner_basis_prime(&field, &generators, &options).unwrap();
        assert_eq!(output.basis, ordinary.basis);
        assert_eq!(
            output.evidence,
            GroebnerEvidence::IndependentBuchbergerSweep
        );
        verify(&field, &output.basis, 1_000_000).unwrap();
    }

    #[test]
    fn binary_extension_lane_agrees_with_the_generic_small_matrix_dispatch() {
        let field = BinaryExtensionField16::new(0b111).unwrap();
        let generators = vec![
            SparsePolynomial::new(
                2,
                MonomialOrder::GradedReverseLexicographic,
                [(monomial(&[2, 0]), 1_u16), (monomial(&[0, 1]), 2_u16)],
                &field,
            ),
            SparsePolynomial::new(
                2,
                MonomialOrder::GradedReverseLexicographic,
                [(monomial(&[1, 1]), 1_u16), (monomial(&[0, 0]), 3_u16)],
                &field,
            ),
        ];
        let generic = groebner_basis(&field, &generators, &GroebnerOptions::default()).unwrap();
        let specialized =
            groebner_basis_binary_extension(&field, &generators, &GroebnerOptions::default())
                .unwrap();
        assert_eq!(generic.basis, specialized.basis);
        assert!(specialized.metrics.generic_row_reductions > 0);
    }

    proptest! {
        #![proptest_config(property_config(96))]

        #[test]
        fn random_small_prime_systems_match_the_generic_lane_and_all_pair_check(
            raw_generators in prop::collection::vec(
                prop::collection::vec((0_u32..7, 0_u32..4, 0_u32..4), 1..5),
                1..5,
            )
        ) {
            let field = PrimeField32::new(7).unwrap();
            let generators = raw_generators
                .into_iter()
                .map(|terms| {
                    SparsePolynomial::new(
                        2,
                        MonomialOrder::GradedReverseLexicographic,
                        terms.into_iter().map(|(coefficient, x, y)| {
                            (monomial(&[x, y]), coefficient)
                        }),
                        &field,
                    )
                })
                .collect::<Vec<_>>();
            let options = GroebnerOptions::default();
            let specialized = groebner_basis_prime(&field, &generators, &options).unwrap();
            let generic = groebner_basis(&field, &generators, &options).unwrap();

            prop_assert_eq!(&specialized.basis, &generic.basis);
            prop_assert_eq!(specialized.evidence, GroebnerEvidence::ExactConstruction);
            if !specialized.basis.is_empty() {
                verify(&field, &specialized.basis, options.max_reduction_steps).unwrap();
            }
        }
    }
}
