// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! PyO3 transport for Noether's permutation-group values.
//!
//! Transports the kernel surface through the private extension for projection
//! from `noether.groups`:
//!
//! * `Permutation` — a [`noether::groups::Permutation`], an image-vector permutation
//!   of `0..degree` with right-action composition (`(g * h)[i] =
//!   h[g[i]]`, matching GAP).
//! * `PermutationGroup` — a permutation group with its full BSGS
//!   pre-computed. Carries the original generators, the base, the
//!   strong generating set, the stabilizer chain, and the group order.
//! * `Transversal` / `SchreierTree` / `StabilizerChain` /
//!   `SchreierGenerator` / `OrbitStabilizer` / `RandomElement` —
//!   structured result types returned by group methods, so
//!   no method on the boundary returns a stringified or untyped
//!   tuple.
//!
//! The pure-Rust kernel does not bind FLINT or any other native
//! library, so this binding stays import-time free of platform gates.

use pyo3::exceptions::{PyIndexError, PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use noether::groups::{
    base_and_strong_generating_set as kernel_bsgs, element_membership_test as kernel_membership,
    group_order as kernel_order, iterate_elements as kernel_iter, orbit as kernel_orbit,
    orbit_stabilizer as kernel_orbit_stabilizer,
    pointwise_stabilizer as kernel_pointwise_stabilizer,
    schreier_generators as kernel_schreier_gens, schreier_tree as kernel_schreier_tree,
    stabilizer_chain as kernel_stabilizer_chain, transversal as kernel_transversal,
    uniform_random_element as kernel_uniform, BigUInt, ConjugacyClass as KernelConjugacyClass,
    DoubleCoset as KernelDoubleCoset, FiniteAbelianInvariantError, FiniteAbelianInvariants,
    FiniteAbelianization, GroupError, NotAPermutationReason, OrbitStabilizer,
    PairOrbit as KernelPairOrbit, Permutation as Perm, PermutationGroup as PermGroup, Point,
    RandomElement, SchreierGenerator, SchreierTree, StabilizerChain,
    SubsetOrbit as KernelSubsetOrbit, Transversal,
};
use noether::homalg::{
    FiniteAbelianGroupCohomology, FiniteAbelianGroupIntegralCohomology,
    FiniteCyclicGroupIntegralCohomology, FiniteGroupCohomology, FiniteGroupIntegralCohomology,
};

use super::facade_scalar::integer_to_python;

/// Convert a kernel [`GroupError`] into the appropriate Python
/// exception. `BasePointMissing` is an index error; `DegreeMismatch`,
/// `NotAPermutation`, nonmembership, `NotASubgroup`, and `IndexTooLarge` are
/// value errors.
fn group_err(e: GroupError) -> PyErr {
    match e {
        GroupError::DegreeMismatch { got, expected } => PyValueError::new_err(format!(
            "permutation degree mismatch: got {got}, expected {expected}"
        )),
        GroupError::NotAPermutation { generator, reason } => match reason {
            NotAPermutationReason::OutOfRange { image, degree } => PyValueError::new_err(format!(
                "generator {generator}: image {image} outside domain 0..{degree}"
            )),
            NotAPermutationReason::RepeatedImage { image } => PyValueError::new_err(format!(
                "generator {generator}: image {image} occurs more than once"
            )),
        },
        GroupError::BasePointMissing { point } => {
            PyIndexError::new_err(format!("base point {point} is not in the group's base"))
        }
        GroupError::InvalidSubsetSize { size, degree } => PyValueError::new_err(format!(
            "subset size {size} exceeds permutation degree {degree}"
        )),
        GroupError::DuplicateSubsetPoint { point } => {
            PyValueError::new_err(format!("subset point {point} occurs more than once"))
        }
        GroupError::AmbiguousPairSeeds { left, right } => PyValueError::new_err(format!(
            "distinct pair seeds {left} and {right} lie in the same point orbit"
        )),
        GroupError::NotASubgroup => {
            PyValueError::new_err("the purported subgroup has a generator not in the group")
        }
        GroupError::ElementNotInGroup => {
            PyValueError::new_err("the permutation is not an element of the group")
        }
        GroupError::InvalidPrime { value } => {
            PyValueError::new_err(format!("{value} is not prime"))
        }
        GroupError::IndexTooLarge { cap } => {
            PyOverflowError::new_err(format!("coset enumeration exceeded the index cap of {cap}"))
        }
        GroupError::EnumerationLimitExceeded { operation, cap } => PyOverflowError::new_err(
            format!("{operation} requires enumerating more than {cap} group elements"),
        ),
        GroupError::SearchBudgetExceeded { cap } => PyOverflowError::new_err(format!(
            "partition-backtrack search exceeded the node cap of {cap}"
        )),
        GroupError::WitnessUnverified => PyRuntimeError::new_err(
            "a backtrack witness failed its sift-verification against the target",
        ),
    }
}

fn finite_abelian_invariant_error(error: FiniteAbelianInvariantError) -> PyErr {
    match error {
        FiniteAbelianInvariantError::EnumerationLimitExceeded { .. }
        | FiniteAbelianInvariantError::OrderTooLarge => PyOverflowError::new_err(error.to_string()),
        FiniteAbelianInvariantError::Group(error) => group_err(error),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn big_uint_to_python(value: BigUInt) -> PyResult<Py<PyAny>> {
    Python::attach(|py| {
        Ok(py
            .import("builtins")?
            .getattr("int")?
            .call1((value.to_decimal(),))?
            .unbind())
    })
}

// ── Permutation ────────────────────────────────────────────────────

/// A permutation of `0..degree`, stored as an image vector.
///
/// Construct via `Permutation.from_images`. The composition
/// convention is **right action** — `(g * h).apply(i) = h.apply(
/// g.apply(i))` — matching GAP and the Schreier–Sims literature.
#[pyclass(module = "noether.groups", name = "Permutation", from_py_object)]
#[derive(Clone)]
pub struct PyPermutation {
    pub(crate) inner: Perm,
}

#[pymethods]
impl PyPermutation {
    /// The stable semantic identity shared with the public Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        <Perm as noether::SemanticType>::TYPE_ID
    }

    /// Build a permutation from its image vector.
    ///
    /// `images[i]` is the image of `i` under the permutation, so
    /// `len(images)` is the degree. Raises `ValueError` if `images`
    /// is not a permutation of `0..len(images)` (out-of-range or
    /// repeated image).
    #[staticmethod]
    fn from_images(images: Vec<Point>) -> PyResult<PyPermutation> {
        Ok(PyPermutation {
            inner: Perm::from_images(&images).map_err(group_err)?,
        })
    }

    /// The identity permutation on `0..degree`.
    #[staticmethod]
    fn identity(degree: usize) -> PyPermutation {
        PyPermutation {
            inner: Perm::identity(degree),
        }
    }

    /// The permutation's degree — the size of its domain.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// `True` iff this is the identity permutation on its domain.
    #[getter]
    fn is_identity(&self) -> bool {
        self.inner.is_identity()
    }

    /// The (multiplicative) order of this permutation — the smallest
    /// `k ≥ 1` with `σ^k = identity`. Equals the LCM of cycle lengths
    /// in the cycle decomposition.
    #[getter]
    fn order(&self) -> PyResult<Py<PyAny>> {
        big_uint_to_python(self.inner.order())
    }

    /// The image vector as a `list[int]` of length `degree`.
    #[getter]
    fn images(&self) -> Vec<Point> {
        self.inner.images().to_vec()
    }

    /// The image of `p` under this permutation. Raises `IndexError`
    /// if `p >= degree`.
    fn apply(&self, p: Point) -> PyResult<Point> {
        if (p as usize) >= self.inner.degree() {
            return Err(PyIndexError::new_err(format!(
                "Permutation.apply: point {p} >= degree {}",
                self.inner.degree()
            )));
        }
        Ok(self.inner.apply(p))
    }

    /// Apply this permutation to a point.
    fn __call__(&self, p: Point) -> PyResult<Point> {
        self.apply(p)
    }

    /// The composition `self * other` under the right-action
    /// convention. Raises `ValueError` on a degree mismatch.
    fn compose(&self, other: &PyPermutation) -> PyResult<PyPermutation> {
        Ok(PyPermutation {
            inner: self.inner.compose(&other.inner).map_err(group_err)?,
        })
    }

    /// Compose two permutations using the kernel's right-action convention.
    fn __mul__(&self, other: &PyPermutation) -> PyResult<PyPermutation> {
        self.compose(other)
    }

    /// The inverse permutation.
    fn inverse(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.inverse(),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "Permutation(degree={}, images={:?})",
                self.inner.degree(),
                self.inner.images()
            )
        })
    }

    fn __eq__(&self, other: &PyPermutation) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> i64 {
        // Cheap deterministic hash off the image vector.
        let mut h: i64 = 1469598103934665603i64.wrapping_neg();
        for &x in self.inner.images() {
            h = h.wrapping_mul(1099511628211).wrapping_add(x as i64);
        }
        h
    }
}

// ── PermGroup ──────────────────────────────────────────────────────

/// A permutation group given by generators, carrying its BSGS.
///
/// Construct via `PermGroup.from_generators` (deterministic
/// Schreier–Sims) or `PermGroup.from_generators_randomized`
/// (Monte-Carlo BSGS, **verified** by a deterministic final sweep, so an
/// unverified probabilistic result is never returned). The constructor computes the base, strong generating set,
/// stabilizer chain, per-level transversals, and group order; every
/// query downstream is a method on this object and runs in
/// constant-time-per-step.
#[pyclass(module = "noether.groups", name = "PermutationGroup", from_py_object)]
#[derive(Clone)]
pub struct PyPermGroup {
    pub(crate) inner: PermGroup,
}

#[pymethods]
impl PyPermGroup {
    /// The stable semantic identity shared with the public Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        <PermGroup as noether::SemanticType>::TYPE_ID
    }

    /// Construct a permutation group on `0..degree` from a generator
    /// list, running deterministic Schreier–Sims.
    ///
    /// Raises `ValueError` on a degree mismatch or a non-permutation
    /// generator.
    #[staticmethod]
    fn from_generators(
        py: Python<'_>,
        degree: usize,
        generators: Vec<PyPermutation>,
    ) -> PyResult<PyPermGroup> {
        let gens: Vec<Perm> = generators.into_iter().map(|p| p.inner).collect();
        py.detach(move || {
            Ok(PyPermGroup {
                inner: PermGroup::from_generators(degree, &gens).map_err(group_err)?,
            })
        })
    }

    /// Construct via the verified randomized Schreier–Sims variant.
    ///
    /// Sims-style product replacement seeded with `seed` supplies
    /// random group elements; the chain stops growing after
    /// `stabilize` consecutive sift-to-identity events; the result
    /// is then verified by the deterministic Schreier-generator
    /// sweep. The `seed` is recorded on the result so the build is
    /// reproducible.
    #[staticmethod]
    fn from_generators_randomized(
        py: Python<'_>,
        degree: usize,
        generators: Vec<PyPermutation>,
        seed: u64,
        stabilize: u32,
    ) -> PyResult<PyPermGroup> {
        let gens: Vec<Perm> = generators.into_iter().map(|p| p.inner).collect();
        py.detach(move || {
            Ok(PyPermGroup {
                inner: PermGroup::from_generators_randomized(degree, &gens, seed, stabilize)
                    .map_err(group_err)?,
            })
        })
    }

    /// The degree — the size of the permutation domain `0..degree`.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// The original generators, in input order, as a `list[
    /// Permutation]`.
    #[getter]
    fn generators(&self) -> Vec<PyPermutation> {
        self.inner
            .generators()
            .iter()
            .map(|p| PyPermutation { inner: p.clone() })
            .collect()
    }

    /// The base — the ordered list of base points the stabilizer
    /// chain stabilises one level at a time. Empty `list[int]` for
    /// the trivial group.
    #[getter]
    fn base(&self) -> Vec<Point> {
        self.inner.base()
    }

    /// The strong generating set, returned as `list[Permutation]`
    /// in the order the generators were added (top of the chain
    /// first).
    #[getter]
    fn strong_generating_set(&self) -> Vec<PyPermutation> {
        self.inner
            .strong_generating_set()
            .into_iter()
            .map(|p| PyPermutation { inner: p.clone() })
            .collect()
    }

    /// `|G|` as a Python `int` — computed once at construction as
    /// the product of the transversal lengths. Returned as a string
    /// the Python side parses to `int`, since the order can exceed
    /// any primitive integer (e.g. `|S_512|` is ~1170 digits).
    #[getter]
    fn order(&self) -> PyResult<Py<PyAny>> {
        big_uint_to_python(self.inner.order())
    }

    /// `True` iff `g` lies in the group. Sifts `g` down the stabilizer
    /// chain and checks the residue is the identity. Raises
    /// `ValueError` on a degree mismatch.
    fn contains(&self, py: Python<'_>, g: &PyPermutation) -> PyResult<bool> {
        py.detach(|| kernel_membership(&self.inner, &g.inner).map_err(group_err))
    }

    /// Support Python's `element in group` membership protocol.
    fn __contains__(&self, py: Python<'_>, g: &PyPermutation) -> PyResult<bool> {
        // NOETHER_RELEASES_GIL: contains delegates to the detached membership implementation.
        self.contains(py, g)
    }

    /// The identity element in this permutation action.
    fn identity(&self) -> PyPermutation {
        PyPermutation {
            inner: Perm::identity(self.inner.degree()),
        }
    }

    /// Checked multiplication of two elements of this group.
    fn multiply(&self, left: &PyPermutation, right: &PyPermutation) -> PyResult<PyPermutation> {
        if !self.inner.contains(&left.inner).map_err(group_err)?
            || !self.inner.contains(&right.inner).map_err(group_err)?
        {
            return Err(group_err(GroupError::ElementNotInGroup));
        }
        Ok(PyPermutation {
            inner: left.inner.compose(&right.inner).map_err(group_err)?,
        })
    }

    /// Checked inverse of an element of this group.
    fn inverse(&self, value: &PyPermutation) -> PyResult<PyPermutation> {
        if !self.inner.contains(&value.inner).map_err(group_err)? {
            return Err(group_err(GroupError::ElementNotInGroup));
        }
        Ok(PyPermutation {
            inner: value.inner.inverse(),
        })
    }

    /// Compute the exact mod-p cohomology ring through `max_degree`.
    fn modular_cohomology(
        &self,
        py: Python<'_>,
        prime: u64,
        max_degree: usize,
    ) -> PyResult<super::homalg::PyFiniteGroupCohomologyTruncation> {
        py.detach(|| {
            Ok(super::homalg::PyFiniteGroupCohomologyTruncation {
                inner: self
                    .inner
                    .modular_cohomology(prime, max_degree)
                    .map_err(super::homalg::finite_group_cohomology_error)?,
            })
        })
    }

    /// Compute the exact integral cohomology ring through `max_degree`.
    fn integral_cohomology(
        &self,
        py: Python<'_>,
        max_degree: usize,
    ) -> PyResult<super::homalg::PyFiniteGroupIntegralCohomologyTruncation> {
        py.detach(|| {
            Ok(super::homalg::PyFiniteGroupIntegralCohomologyTruncation {
                inner: self
                    .inner
                    .integral_cohomology(max_degree)
                    .map_err(super::homalg::finite_group_integral_cohomology_error)?,
            })
        })
    }

    /// Canonical invariant factors when this concrete group is finite abelian.
    fn abelian_invariant_factors(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        py.detach(|| self.inner.abelian_invariant_factors())
            .map_err(finite_abelian_invariant_error)?
            .iter()
            .map(|factor| integer_to_python(py, factor))
            .collect()
    }

    /// Canonical invariant factors of this finite group's abelianization.
    fn abelianization_invariant_factors(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        py.detach(|| self.inner.abelianization_invariant_factors())
            .map_err(finite_abelian_invariant_error)?
            .iter()
            .map(|factor| integer_to_python(py, factor))
            .collect()
    }

    /// Complete closed-form H*(BG; F_p) for a finite abelian group.
    fn complete_abelian_modular_cohomology(
        &self,
        py: Python<'_>,
        field: &super::numthy::PyFiniteField,
    ) -> PyResult<super::homalg::PyFiniteAbelianGroupCohomologyPresentation> {
        py.detach(|| {
            Ok(super::homalg::PyFiniteAbelianGroupCohomologyPresentation {
                inner: self
                    .inner
                    .complete_abelian_modular_cohomology(field.inner.clone())
                    .map_err(super::homalg::finite_abelian_group_cohomology_error)?,
            })
        })
    }

    /// Complete closed-form H*(BG; Z) for a supported finite abelian group.
    fn complete_abelian_integral_cohomology(
        &self,
        py: Python<'_>,
    ) -> PyResult<super::homalg::PyFiniteAbelianGroupIntegralCohomologyPresentation> {
        py.detach(|| {
            Ok(
                super::homalg::PyFiniteAbelianGroupIntegralCohomologyPresentation {
                    inner: self
                        .inner
                        .complete_abelian_integral_cohomology()
                        .map_err(super::homalg::finite_abelian_group_cohomology_error)?,
                },
            )
        })
    }

    /// Complete closed-form H*(BG; Z) for a finite cyclic group.
    fn complete_cyclic_integral_cohomology(
        &self,
        py: Python<'_>,
    ) -> PyResult<super::homalg::PyFiniteCyclicGroupIntegralCohomologyPresentation> {
        py.detach(|| {
            Ok(
                super::homalg::PyFiniteCyclicGroupIntegralCohomologyPresentation {
                    inner: self
                        .inner
                        .complete_cyclic_integral_cohomology()
                        .map_err(super::homalg::finite_cyclic_group_integral_cohomology_error)?,
                },
            )
        })
    }

    /// A deterministic Sylow `p`-subgroup in this concrete action.
    fn sylow_subgroup(&self, py: Python<'_>, prime: u64) -> PyResult<PyPermGroup> {
        py.detach(|| {
            Ok(PyPermGroup {
                inner: self.inner.sylow_subgroup(prime).map_err(group_err)?,
            })
        })
    }

    /// Literal intersection with another group on the same point set.
    fn intersection(&self, py: Python<'_>, other: &PyPermGroup) -> PyResult<PyPermGroup> {
        py.detach(|| {
            Ok(PyPermGroup {
                inner: self.inner.intersection(&other.inner).map_err(group_err)?,
            })
        })
    }

    /// Sift `g` through the stabilizer chain. Returns the sift
    /// residue and a `bool` indicating membership (residue is the
    /// identity iff `g ∈ G`).
    fn sift(&self, g: &PyPermutation) -> PyResult<(PyPermutation, bool)> {
        let (residue, ok) = self.inner.sift(&g.inner).map_err(group_err)?;
        Ok((PyPermutation { inner: residue }, ok))
    }

    /// The orbit of `point` under the **full group**, returned as
    /// `list[int]` in BFS-from-`point` order.
    /// Raises `ValueError` if `point >= degree`.
    fn orbit(&self, py: Python<'_>, point: Point) -> PyResult<Vec<Point>> {
        py.detach(|| kernel_orbit(&self.inner, point).map_err(group_err))
    }

    /// Exact orbit of one subset under the induced set action.
    fn orbit_of_subset(&self, py: Python<'_>, points: Vec<Point>) -> PyResult<PySubsetOrbit> {
        py.detach(|| {
            Ok(PySubsetOrbit {
                inner: self.inner.orbit_of_subset(&points).map_err(group_err)?,
            })
        })
    }

    /// Exact orbit partition of all subsets with the requested cardinality.
    fn orbits_on_subsets(&self, py: Python<'_>, size: usize) -> PyResult<Vec<PySubsetOrbit>> {
        py.detach(|| {
            Ok(self
                .inner
                .orbits_on_subsets(size)
                .map_err(group_err)?
                .into_iter()
                .map(|inner| PySubsetOrbit { inner })
                .collect())
        })
    }

    /// Exact orbit partition of unordered pairs of distinct points.
    fn orbits_on_pairs(&self, py: Python<'_>) -> PyResult<Vec<PySubsetOrbit>> {
        py.detach(|| {
            Ok(self
                .inner
                .orbits_on_pairs()
                .map_err(group_err)?
                .into_iter()
                .map(|inner| PySubsetOrbit { inner })
                .collect())
        })
    }

    /// Compact exact double-coset reduction of point pairs.
    fn pair_orbits(
        &self,
        py: Python<'_>,
        left_seed: Point,
        right_seed: Point,
    ) -> PyResult<Vec<PyPairOrbit>> {
        py.detach(|| {
            Ok(self
                .inner
                .pair_orbits(left_seed, right_seed)
                .map_err(group_err)?
                .into_iter()
                .map(|inner| PyPairOrbit { inner })
                .collect())
        })
    }

    /// Exact double cosets of two literal subgroups.
    fn double_cosets(
        &self,
        py: Python<'_>,
        left: &PyPermGroup,
        right: &PyPermGroup,
    ) -> PyResult<Vec<PyDoubleCoset>> {
        py.detach(|| {
            Ok(self
                .inner
                .double_cosets(&left.inner, &right.inner)
                .map_err(group_err)?
                .into_iter()
                .map(|inner| PyDoubleCoset { inner })
                .collect())
        })
    }

    /// Lower-triangular exact double-coset table for several subgroups.
    fn double_coset_triangle(
        &self,
        py: Python<'_>,
        subgroups: Vec<PyPermGroup>,
    ) -> PyResult<Vec<Vec<Vec<PyDoubleCoset>>>> {
        let subgroups = subgroups
            .into_iter()
            .map(|subgroup| subgroup.inner)
            .collect::<Vec<_>>();
        py.detach(|| {
            Ok(self
                .inner
                .double_coset_triangle(&subgroups)
                .map_err(group_err)?
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|entry| {
                            entry
                                .into_iter()
                                .map(|inner| PyDoubleCoset { inner })
                                .collect()
                        })
                        .collect()
                })
                .collect())
        })
    }

    /// Whether this action has one orbit. The empty action is not transitive.
    #[getter]
    fn is_transitive(&self, py: Python<'_>) -> PyResult<bool> {
        if self.inner.degree() == 0 {
            return Ok(false);
        }
        // NOETHER_RELEASES_GIL: orbit owns and detaches the group traversal.
        Ok(self.orbit(py, 0)?.len() == self.inner.degree())
    }

    /// A transversal of the **point stabilizer** of `point` in `G`.
    /// Returns a [`Transversal`] whose
    /// `reps[i]` maps `point` to `orbit[i]`.
    fn transversal(&self, py: Python<'_>, point: Point) -> PyResult<PyTransversal> {
        py.detach(|| {
            let inner = kernel_transversal(&self.inner, point).map_err(group_err)?;
            Ok(PyTransversal { inner })
        })
    }

    /// The Schreier tree of `point` under the full group. Returns a
    /// [`SchreierTree`] (BFS
    /// orbit + predecessor table).
    fn schreier_tree(&self, py: Python<'_>, point: Point) -> PyResult<PySchreierTree> {
        py.detach(|| {
            let inner = kernel_schreier_tree(&self.inner, point).map_err(group_err)?;
            Ok(PySchreierTree { inner })
        })
    }

    /// The Schreier generators of `Stab_G(point)`. Returns
    /// `list[SchreierGenerator]`,
    /// each carrying the coset index, generator index, and the
    /// Schreier-generator permutation itself.
    fn schreier_generators(
        &self,
        py: Python<'_>,
        point: Point,
    ) -> PyResult<Vec<PySchreierGenerator>> {
        py.detach(|| {
            let generators = kernel_schreier_gens(&self.inner, point).map_err(group_err)?;
            Ok(generators
                .into_iter()
                .map(|inner| PySchreierGenerator { inner })
                .collect())
        })
    }

    /// The pointwise stabilizer of `points` in `G`. Returns a new
    /// `PermGroup` (the
    /// stabilizer itself, with its own BSGS pre-computed). Raises
    /// `ValueError` if any `points` entry is `>= degree`.
    fn pointwise_stabilizer(&self, py: Python<'_>, points: Vec<Point>) -> PyResult<PyPermGroup> {
        py.detach(|| {
            let inner = kernel_pointwise_stabilizer(&self.inner, &points).map_err(group_err)?;
            Ok(PyPermGroup { inner })
        })
    }

    /// The stabilizer of one point.
    fn stabilizer(&self, py: Python<'_>, point: Point) -> PyResult<PyPermGroup> {
        // NOETHER_RELEASES_GIL: pointwise_stabilizer owns and detaches the group search.
        self.pointwise_stabilizer(py, vec![point])
    }

    /// The (orbit, stabilizer) pair of `point` under `G`. Returns an
    /// [`OrbitStabilizer`]
    /// carrying the orbit (`list[int]`) and the stabilizer
    /// (`PermGroup`).
    fn orbit_stabilizer(&self, py: Python<'_>, point: Point) -> PyResult<PyOrbitStabilizer> {
        py.detach(|| {
            let inner = kernel_orbit_stabilizer(&self.inner, point).map_err(group_err)?;
            Ok(PyOrbitStabilizer { inner })
        })
    }

    /// The full stabilizer chain. Returns a [`StabilizerChain`] whose
    /// `steps` exposes per-level
    /// base point, generators, orbit, and transversal.
    fn stabilizer_chain(&self, py: Python<'_>) -> PyStabilizerChain {
        py.detach(|| PyStabilizerChain {
            inner: kernel_stabilizer_chain(&self.inner),
        })
    }

    /// A lazy odometer iterator over every group element. Each iteration
    /// yields a fresh
    /// `Permutation`. The total number of yields equals
    /// `order`.
    fn iterate_elements(slf: PyRef<'_, Self>) -> PyResult<Py<PyElementIter>> {
        // Materialise the iteration to break the lifetime from the
        // borrow chain — the kernel iterator borrows the group, but
        // the Python wrapper needs an owned snapshot. We pre-compute
        // the full list; `iterate_elements` is intended for groups
        // with manageable element counts.
        let py = slf.py();
        let inner = slf.inner.clone();
        let order = inner.order();
        let max_chunk = BigUInt::from(1_000_000u64);
        if order > max_chunk {
            return Err(PyOverflowError::new_err(format!(
                "PermutationGroup.iterate_elements: refusing to materialise {} elements eagerly; use a streaming sampler",
                order.to_decimal()
            )));
        }
        drop(slf);
        let elts = py.detach(move || {
            kernel_iter(&inner)
                .map(|inner| PyPermutation { inner })
                .collect()
        });
        Py::new(py, PyElementIter { elts, cursor: 0 })
    }

    /// Exact conjugacy classes, ordered by their canonical representatives.
    ///
    /// The native correctness floor enumerates groups of order at most one
    /// million and fails before returning a partial result above that bound.
    fn conjugacy_classes(&self, py: Python<'_>) -> PyResult<Vec<PyConjugacyClass>> {
        py.detach(|| {
            Ok(self
                .inner
                .conjugacy_classes()
                .map_err(group_err)?
                .into_iter()
                .map(|inner| PyConjugacyClass { inner })
                .collect())
        })
    }

    /// A uniform random element via per-level transversal sampling. Exact
    /// uniform draw
    /// (not a walk). Returns a [`RandomElement`] carrying the
    /// element and the `seed` it was generated from.
    fn uniform_random_element(&self, seed: u64) -> PyRandomElement {
        let r = kernel_uniform(&self.inner, seed);
        PyRandomElement { inner: r }
    }

    /// Whether the group is solvable, computed from its derived series.
    #[getter]
    fn is_solvable(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| noether::groups::is_solvable(&self.inner).map_err(group_err))
    }

    /// Whether the group is nilpotent, computed from its lower central series.
    #[getter]
    fn is_nilpotent(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| noether::groups::is_nilpotent(&self.inner).map_err(group_err))
    }

    /// Whether the group equals its derived subgroup.
    #[getter]
    fn is_perfect(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| noether::groups::is_perfect(&self.inner).map_err(group_err))
    }

    /// The base and strong generating set as a `(list[int],
    /// list[Permutation])` pair, useful for callers
    /// that want the BSGS without constructing the full
    /// `PermGroup` API on it twice.
    #[staticmethod]
    fn base_and_strong_generating_set(
        py: Python<'_>,
        degree: usize,
        generators: Vec<PyPermutation>,
    ) -> PyResult<(Vec<Point>, Vec<PyPermutation>)> {
        let gens: Vec<Perm> = generators.into_iter().map(|p| p.inner).collect();
        py.detach(move || {
            let (base, sgs) = kernel_bsgs(degree, &gens).map_err(group_err)?;
            Ok((
                base,
                sgs.into_iter()
                    .map(|p| PyPermutation { inner: p })
                    .collect(),
            ))
        })
    }

    /// `|G|` as a Python `int`. Conversion through decimal text preserves
    /// arbitrarily large
    /// orders without imposing a Rust primitive-integer limit.
    #[staticmethod]
    fn group_order(
        py: Python<'_>,
        degree: usize,
        generators: Vec<PyPermutation>,
    ) -> PyResult<Py<PyAny>> {
        let gens: Vec<Perm> = generators.into_iter().map(|p| p.inner).collect();
        let order = py.detach(move || kernel_order(degree, &gens).map_err(group_err))?;
        let order = order.to_decimal();
        Ok(py
            .import("builtins")?
            .getattr("int")?
            .call1((order,))?
            .unbind())
    }

    // ── partition backtracking ─────────────────────────────────────

    /// A conjugating element `g` with `selfᵍ = other` (i.e.
    /// `g⁻¹ · self · g = other`), or `None` if the two groups are not
    /// conjugate in `S_n`. The witness is sift-verified before it is
    /// returned. Raises `ValueError` on a degree mismatch, `OverflowError`
    /// if the backtrack exceeds its search budget.
    fn conjugating_element(
        &self,
        py: Python<'_>,
        other: &PyPermGroup,
    ) -> PyResult<Option<PyPermutation>> {
        py.detach(|| {
            Ok(self
                .inner
                .conjugating_element(&other.inner)
                .map_err(group_err)?
                .map(|inner| PyPermutation { inner }))
        })
    }

    /// A witness `g` with `selfᵍ ≤ over` (some `S_n`-conjugate of `self`
    /// is a subgroup of `over`), or `None`. The witness is verified
    /// before it is returned. Raises `ValueError` on a degree mismatch,
    /// `OverflowError` on an exhausted search budget.
    fn embeds_up_to_conjugacy(
        &self,
        py: Python<'_>,
        over: &PyPermGroup,
    ) -> PyResult<Option<PyPermutation>> {
        py.detach(|| {
            Ok(self
                .inner
                .embeds_up_to_conjugacy(&over.inner)
                .map_err(group_err)?
                .map(|inner| PyPermutation { inner }))
        })
    }

    /// The normalizer of `self` in the full symmetric group,
    /// `N_{S_n}(self)`, returned as a `PermGroup`.
    fn normalizer_in_symmetric(&self, py: Python<'_>) -> PyResult<PyPermGroup> {
        py.detach(|| {
            Ok(PyPermGroup {
                inner: self.inner.normalizer_in_symmetric().map_err(group_err)?,
            })
        })
    }

    /// The normalizer of `h` inside `self`, `N_self(h) = { g ∈ self :
    /// hᵍ = h }`, as a `PermGroup`. Raises `ValueError` on a degree
    /// mismatch.
    fn normalizer(&self, py: Python<'_>, h: &PyPermGroup) -> PyResult<PyPermGroup> {
        py.detach(|| {
            Ok(PyPermGroup {
                inner: self.inner.normalizer(&h.inner).map_err(group_err)?,
            })
        })
    }

    /// The set stabilizer `Stab_self(points)` under the `OnSets`
    /// action — the subgroup fixing `points` setwise — as a
    /// `PermGroup`. Duplicate points are canonicalized (sort + dedup);
    /// a point `>= degree` raises `ValueError`.
    fn set_stabilizer(&self, py: Python<'_>, points: Vec<Point>) -> PyResult<PyPermGroup> {
        py.detach(|| {
            Ok(PyPermGroup {
                inner: self.inner.set_stabilizer(&points).map_err(group_err)?,
            })
        })
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "PermutationGroup(degree={}, generators={}, order={})",
                self.inner.degree(),
                self.inner.generators().len(),
                self.inner.order().to_decimal()
            )
        })
    }
}

/// One exact orbit under a permutation group's action on subsets.
#[pyclass(
    module = "noether.groups",
    name = "SubsetOrbit",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PySubsetOrbit {
    inner: KernelSubsetOrbit,
}

#[pymethods]
impl PySubsetOrbit {
    /// Stable semantic identity shared with the Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        <KernelSubsetOrbit as noether::SemanticType>::TYPE_ID
    }

    /// Degree of the underlying point action.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Common cardinality of the subsets in this orbit.
    #[getter]
    fn subset_size(&self) -> usize {
        self.inner.subset_size()
    }

    /// Lexicographically least subset in the orbit.
    #[getter]
    fn representative(&self) -> Vec<Point> {
        self.inner.representative().to_vec()
    }

    /// Every subset in stable lexical order.
    #[getter]
    fn elements(&self) -> Vec<Vec<Point>> {
        self.inner.elements().to_vec()
    }

    /// Exact orbit cardinality.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Recompute the orbit and orbit-stabilizer identity exactly.
    fn verify(&self, group: &PyPermGroup) -> PyResult<()> {
        self.inner.verify(&group.inner).map_err(group_err)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "SubsetOrbit(degree={}, subset_size={}, size={})",
            self.inner.degree(),
            self.inner.subset_size(),
            self.inner.size()
        ))
    }
}

/// One compact exact diagonal orbit of point pairs.
#[pyclass(module = "noether.groups", name = "PairOrbit", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyPairOrbit {
    inner: KernelPairOrbit,
}

/// One exact double coset of two subgroups in a concrete ambient group.
#[pyclass(
    module = "noether.groups",
    name = "DoubleCoset",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyDoubleCoset {
    inner: KernelDoubleCoset,
}

#[pymethods]
impl PyDoubleCoset {
    /// Stable semantic identity shared with the Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        <KernelDoubleCoset as noether::SemanticType>::TYPE_ID
    }

    /// Degree of the common permutation action.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// Canonical ambient-group representative.
    #[getter]
    fn representative(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.representative().clone(),
        }
    }

    /// Canonical representative of the inverse double coset for equal subgroups.
    #[getter]
    fn inverse_representative(&self) -> Option<PyPermutation> {
        self.inner
            .inverse_representative()
            .cloned()
            .map(|inner| PyPermutation { inner })
    }

    /// Number of right-subgroup cosets in this double coset.
    #[getter]
    fn right_coset_count(&self) -> usize {
        self.inner.right_coset_count()
    }

    /// Exact number of ambient elements in this double coset.
    #[getter]
    fn size(&self) -> PyResult<Py<PyAny>> {
        big_uint_to_python(self.inner.size().clone())
    }

    /// Recompute this double coset and verify its evidence.
    fn verify(
        &self,
        ambient: &PyPermGroup,
        left: &PyPermGroup,
        right: &PyPermGroup,
    ) -> PyResult<()> {
        self.inner
            .verify(&ambient.inner, &left.inner, &right.inner)
            .map_err(group_err)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "DoubleCoset(right_cosets={}, size={})",
            self.inner.right_coset_count(),
            self.inner.size()
        ))
    }
}

#[pymethods]
impl PyPairOrbit {
    /// Stable semantic identity shared with the Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        <KernelPairOrbit as noether::SemanticType>::TYPE_ID
    }

    /// Degree of the underlying action.
    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree()
    }

    /// First invariant-orbit seed.
    #[getter]
    fn left_seed(&self) -> Point {
        self.inner.left_seed()
    }

    /// Second invariant-orbit seed.
    #[getter]
    fn right_seed(&self) -> Point {
        self.inner.right_seed()
    }

    /// Canonical endpoint pair.
    #[getter]
    fn representative(&self) -> Vec<Point> {
        self.inner.representative().to_vec()
    }

    /// Stabilizer suborbits encoding the double-coset class.
    #[getter]
    fn anchored_suborbits(&self) -> Vec<Vec<Point>> {
        self.inner.anchored_suborbits().to_vec()
    }

    /// Exact orbit cardinality.
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Whether both endpoints range through one point orbit.
    #[getter]
    fn same_point_orbit(&self) -> bool {
        self.inner.same_point_orbit()
    }

    /// Recompute and verify this compact pair orbit.
    fn verify(&self, group: &PyPermGroup) -> PyResult<()> {
        self.inner.verify(&group.inner).map_err(group_err)
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(format!(
            "PairOrbit(representative={:?}, size={})",
            self.inner.representative(),
            self.inner.size()
        ))
    }
}

/// One exact conjugacy class in a concrete permutation group.
#[pyclass(
    module = "noether.groups",
    name = "ConjugacyClass",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub struct PyConjugacyClass {
    inner: KernelConjugacyClass,
}

#[pymethods]
impl PyConjugacyClass {
    /// Stable semantic identity shared with the public Rust facade.
    #[classattr]
    fn type_id() -> &'static str {
        <KernelConjugacyClass as noether::SemanticType>::TYPE_ID
    }

    /// The source permutation group.
    #[getter]
    fn group(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.group().clone(),
        }
    }

    /// Canonical representative of the class.
    #[getter]
    fn representative(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.representative().clone(),
        }
    }

    /// Class elements in deterministic order, materialized lazily.
    #[getter]
    fn elements(&self, py: Python<'_>) -> Vec<PyPermutation> {
        py.detach(|| {
            self.inner
                .elements()
                .iter()
                .cloned()
                .map(|inner| PyPermutation { inner })
                .collect()
        })
    }

    /// Exact class size as a Python integer.
    #[getter]
    fn size(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let size = self.inner.size().to_decimal();
        Ok(py
            .import("builtins")?
            .getattr("int")?
            .call1((size,))?
            .unbind())
    }

    /// Centralizer of the representative inside the source group.
    fn centralizer(&self, py: Python<'_>) -> PyResult<PyPermGroup> {
        py.detach(|| {
            Ok(PyPermGroup {
                inner: self.inner.centralizer().map_err(group_err)?,
            })
        })
    }

    /// Recompute the conjugation orbit and orbit-centralizer identity.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.verify().map_err(group_err))
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text(self.inner.to_string())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn _tertius_summary_(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let summary = PyDict::new(py);
        summary.set_item("type", Self::type_id())?;
        summary.set_item("degree", self.inner.group().degree())?;
        summary.set_item("size", self.size(py)?)?;
        Ok(summary.unbind())
    }
}

// ── Transversal ────────────────────────────────────────────────────

/// A transversal of `Stab_G(point)` in `G`, paired with the orbit it indexes.
#[pyclass(module = "noether.groups", name = "Transversal")]
pub struct PyTransversal {
    pub(crate) inner: Transversal,
}

#[pymethods]
impl PyTransversal {
    /// The base point of this transversal.
    #[getter]
    fn base(&self) -> Point {
        self.inner.base
    }

    /// The orbit `base^G`, in BFS-from-`base` order. `list[int]`.
    #[getter]
    fn orbit(&self) -> Vec<Point> {
        self.inner.orbit.clone()
    }

    /// The coset representatives, `list[Permutation]` of the same
    /// length as `orbit`. `reps[i]` maps `base` to `orbit[i]`.
    #[getter]
    fn reps(&self) -> Vec<PyPermutation> {
        self.inner
            .reps
            .iter()
            .map(|p| PyPermutation { inner: p.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "Transversal(base={}, orbit_len={})",
                self.inner.base,
                self.inner.orbit.len()
            )
        })
    }
}

// ── SchreierTree ───────────────────────────────────────────────────

/// The Schreier tree of a base point under the full group.
#[pyclass(module = "noether.groups", name = "SchreierTree")]
pub struct PySchreierTree {
    pub(crate) inner: SchreierTree,
}

#[pymethods]
impl PySchreierTree {
    /// Root of the tree.
    #[getter]
    fn root(&self) -> Point {
        self.inner.root
    }

    /// Orbit in BFS-from-`root` discovery order. `list[int]`.
    #[getter]
    fn orbit(&self) -> Vec<Point> {
        self.inner.orbit.clone()
    }

    /// Predecessor table — `pred[p]` is `None` for the root and any
    /// non-orbit point, else `(parent, generator_index)` such that
    /// `generators[generator_index].apply(parent) == p`. Returned
    /// as `list[tuple[int, int] | None]` of length `degree`.
    #[getter]
    fn pred(&self) -> Vec<Option<(Point, u32)>> {
        self.inner.pred.clone()
    }

    /// `True` iff `p` is in the orbit.
    fn contains(&self, p: Point) -> bool {
        self.inner.contains(p)
    }

    /// The generators the `generator_index` slots refer to —
    /// `list[Permutation]`.
    #[getter]
    fn generators_used(&self) -> Vec<PyPermutation> {
        self.inner
            .generators_used
            .iter()
            .map(|p| PyPermutation { inner: p.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "SchreierTree(root={}, orbit_len={})",
                self.inner.root,
                self.inner.orbit.len()
            )
        })
    }
}

// ── StabilizerChain ────────────────────────────────────────────────

/// One step of a stabilizer chain — a row of `StabilizerChain.steps`.
#[pyclass(module = "noether.groups", name = "StabilizerChainStep")]
pub struct PyStabilizerChainStep {
    pub(crate) base: Point,
    pub(crate) generators: Vec<Perm>,
    pub(crate) orbit: Vec<Point>,
    pub(crate) transversal: Vec<Perm>,
}

#[pymethods]
impl PyStabilizerChainStep {
    /// Base point at this level.
    #[getter]
    fn base(&self) -> Point {
        self.base
    }

    /// Strong generators at this level — those stabilising every
    /// earlier base point. `list[Permutation]`.
    #[getter]
    fn generators(&self) -> Vec<PyPermutation> {
        self.generators
            .iter()
            .map(|p| PyPermutation { inner: p.clone() })
            .collect()
    }

    /// Orbit of `base` at this level, BFS order. `list[int]`.
    #[getter]
    fn orbit(&self) -> Vec<Point> {
        self.orbit.clone()
    }

    /// Coset representatives, parallel to `orbit`. `list[Permutation]`.
    #[getter]
    fn transversal(&self) -> Vec<PyPermutation> {
        self.transversal
            .iter()
            .map(|p| PyPermutation { inner: p.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "StabilizerChainStep(base={}, generators={}, orbit_len={})",
                self.base,
                self.generators.len(),
                self.orbit.len()
            )
        })
    }
}

/// The full stabilizer chain.
#[pyclass(module = "noether.groups", name = "StabilizerChain")]
pub struct PyStabilizerChain {
    pub(crate) inner: StabilizerChain,
}

#[pymethods]
impl PyStabilizerChain {
    /// The chain steps, top-of-chain (`G`) first. Each step is a
    /// [`StabilizerChainStep`] with `base`, `generators`, `orbit`,
    /// and `transversal`.
    #[getter]
    fn steps(&self) -> Vec<PyStabilizerChainStep> {
        self.inner
            .steps
            .iter()
            .map(|s| PyStabilizerChainStep {
                base: s.base,
                generators: s.generators.clone(),
                orbit: s.orbit.clone(),
                transversal: s.transversal.clone(),
            })
            .collect()
    }

    /// Number of levels.
    fn __len__(&self) -> usize {
        self.inner.steps.len()
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!("StabilizerChain(levels={})", self.inner.steps.len())
        })
    }
}

// ── SchreierGenerator ──────────────────────────────────────────────

/// One Schreier generator of a point stabilizer.
#[pyclass(module = "noether.groups", name = "SchreierGenerator")]
pub struct PySchreierGenerator {
    pub(crate) inner: SchreierGenerator,
}

#[pymethods]
impl PySchreierGenerator {
    /// Index of the coset representative used (into the transversal
    /// of the base point).
    #[getter]
    fn coset_index(&self) -> usize {
        self.inner.coset_index
    }

    /// Index of the group generator used.
    #[getter]
    fn generator_index(&self) -> usize {
        self.inner.generator_index
    }

    /// The Schreier-generator permutation itself.
    #[getter]
    fn perm(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.perm.clone(),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "SchreierGenerator(coset_index={}, generator_index={})",
                self.inner.coset_index, self.inner.generator_index
            )
        })
    }
}

// ── OrbitStabilizer ────────────────────────────────────────────────

/// The (orbit, stabilizer) pair returned by `PermGroup.orbit_stabilizer`,
/// directly encoding the orbit–stabilizer identity `|orbit| *
/// |stab| = |G|`.
#[pyclass(module = "noether.groups", name = "OrbitStabilizer")]
pub struct PyOrbitStabilizer {
    pub(crate) inner: OrbitStabilizer,
}

#[pymethods]
impl PyOrbitStabilizer {
    /// The orbit, `list[int]`.
    #[getter]
    fn orbit(&self) -> Vec<Point> {
        self.inner.orbit.clone()
    }

    /// The point stabilizer, as a `PermGroup`.
    #[getter]
    fn stabilizer(&self) -> PyPermGroup {
        PyPermGroup {
            inner: self.inner.stabilizer.clone(),
        }
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "OrbitStabilizer(orbit_len={}, stab_order={})",
                self.inner.orbit.len(),
                self.inner.stabilizer.order().to_decimal()
            )
        })
    }
}

// ── RandomElement ──────────────────────────────────────────────────

/// A seeded uniform random group element. Carries the element and
/// the `seed` that produced it, so the draw is fully reproducible.
#[pyclass(module = "noether.groups", name = "RandomElement")]
pub struct PyRandomElement {
    pub(crate) inner: RandomElement,
}

#[pymethods]
impl PyRandomElement {
    /// The element itself.
    #[getter]
    fn element(&self) -> PyPermutation {
        PyPermutation {
            inner: self.inner.element.clone(),
        }
    }

    /// The seed the element was generated from.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        crate::py::repr::bounded_text({
            format!(
                "RandomElement(seed={}, element_degree={})",
                self.inner.seed,
                self.inner.element.degree()
            )
        })
    }
}

// ── ElementIter ────────────────────────────────────────────────────

/// Lazy iterator over the elements of a [`PermGroup`]. Snapshotted at construction
/// so it is independent of the parent group's lifetime.
#[pyclass(module = "noether._engine", name = "PermGroupElementIter")]
pub struct PyElementIter {
    elts: Vec<PyPermutation>,
    cursor: usize,
}

#[pymethods]
impl PyElementIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Return the next element, or raise `StopIteration` when
    /// exhausted.
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyPermutation> {
        if slf.cursor >= slf.elts.len() {
            None
        } else {
            let i = slf.cursor;
            slf.cursor += 1;
            Some(slf.elts[i].clone())
        }
    }

    /// Number of elements remaining.
    fn __len__(&self) -> usize {
        self.elts.len().saturating_sub(self.cursor)
    }
}
