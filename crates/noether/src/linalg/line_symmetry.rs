// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact integral actions and finite stabilizers of line configurations.

use core::fmt;
use std::collections::{BTreeSet, VecDeque};
use std::str::FromStr;

use crate::groups::{Permutation, PermutationGroup, Point};
use crate::{Integer, IntegerRing, NoetherError, Rational, SemanticType, TypeDescription, Q, Z};

use super::{IntegralLineConfiguration, LineConfigurationError, Matrix, Vector};

const SYMMETRY_SEARCH_NODE_CAP: usize = 10_000_000;
const EXTERNAL_LINE_ORBIT_CAP: usize = 1_000_000;

static INTEGRAL_LINE_AUTOMORPHISM_GROUP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/IntegralLineAutomorphismGroup@1",
    rust_path: "noether::linalg::IntegralLineAutomorphismGroup",
    python_path: Some("noether.linalg.IntegralLineAutomorphismGroup"),
};

static INTEGRAL_LINE_EQUIVALENCE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/IntegralLineEquivalence@1",
    rust_path: "noether::linalg::IntegralLineEquivalence",
    python_path: Some("noether.linalg.IntegralLineEquivalence"),
};

static INTEGRAL_LINE_ORBIT_ACTION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/IntegralLineOrbitAction@1",
    rust_path: "noether::linalg::IntegralLineOrbitAction",
    python_path: Some("noether.linalg.IntegralLineOrbitAction"),
};

static INTEGRAL_LINE_SUBGROUP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/linalg/IntegralLineSubgroup@1",
    rust_path: "noether::linalg::IntegralLineSubgroup",
    python_path: Some("noether.linalg.IntegralLineSubgroup"),
};

/// A verified unimodular equivalence between embedded line configurations.
#[derive(Clone)]
pub struct IntegralLineEquivalence {
    source: IntegralLineConfiguration,
    target: IntegralLineConfiguration,
    transformation: Matrix<IntegerRing>,
    line_permutation: Permutation,
}

impl IntegralLineEquivalence {
    /// Source configuration.
    pub fn source(&self) -> &IntegralLineConfiguration {
        &self.source
    }

    /// Target configuration.
    pub fn target(&self) -> &IntegralLineConfiguration {
        &self.target
    }

    /// Unimodular matrix carrying the source to the target.
    pub fn transformation(&self) -> &Matrix<IntegerRing> {
        &self.transformation
    }

    /// Induced bijection from source-line indices to target-line indices.
    pub fn line_permutation(&self) -> &Permutation {
        &self.line_permutation
    }

    /// Recheck the transformation and its induced line bijection exactly.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        let (line, _) = equivalence_actions(&self.source, &self.target, &self.transformation)?;
        if line != self.line_permutation {
            return Err(symmetry_error(
                "equivalence witness has the wrong induced line permutation",
            ));
        }
        Ok(())
    }
}

impl SemanticType for IntegralLineEquivalence {
    const TYPE_ID: &'static str = "noether/linalg/IntegralLineEquivalence@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let matrix = self
            .transformation
            .rows()
            .map(|row| row.iter().map(ToString::to_string).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.source.canonical_bytes()?,
            self.target.canonical_bytes()?,
            matrix,
            self.line_permutation.images(),
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_LINE_EQUIVALENCE_DESCRIPTION
    }
}

impl fmt::Debug for IntegralLineEquivalence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegralLineEquivalence")
            .field("ambient_rank", &self.source.ambient_rank())
            .field("line_count", &self.source.line_count())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for IntegralLineEquivalence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "IntegralLineEquivalence(rank={}, lines={})",
            self.source.ambient_rank(),
            self.source.line_count()
        )
    }
}

/// A verified finite subgroup of the integral stabilizer of a spanning line
/// configuration.
///
/// Unlike [`IntegralLineAutomorphismGroup`], this value need not be the full
/// stabilizer.  It is the exact object needed for face stabilizers, restriction
/// to invariant subconfigurations, and coordinate transport.
#[derive(Clone)]
pub struct IntegralLineSubgroup {
    configuration: IntegralLineConfiguration,
    matrix_generators: Vec<Matrix<IntegerRing>>,
    signed_action: PermutationGroup,
    line_action: PermutationGroup,
    order: Integer,
}

impl IntegralLineSubgroup {
    /// Configuration preserved by every subgroup element.
    pub fn configuration(&self) -> &IntegralLineConfiguration {
        &self.configuration
    }

    /// Unimodular matrix generators.
    pub fn matrix_generators(&self) -> &[Matrix<IntegerRing>] {
        &self.matrix_generators
    }

    /// Faithful action on signed line representatives.
    pub fn signed_action(&self) -> &PermutationGroup {
        &self.signed_action
    }

    /// Induced action on unoriented lines.
    pub fn line_action(&self) -> &PermutationGroup {
        &self.line_action
    }

    /// Exact subgroup order.
    pub fn order(&self) -> &Integer {
        &self.order
    }

    /// Restrict the same matrix group to an invariant spanning subconfiguration.
    pub fn restrict(
        &self,
        configuration: &IntegralLineConfiguration,
    ) -> Result<Self, LineConfigurationError> {
        if configuration.ambient_rank() != self.configuration.ambient_rank() {
            return Err(LineConfigurationError::DimensionMismatch {
                line: 0,
                expected: self.configuration.ambient_rank(),
                actual: configuration.ambient_rank(),
            });
        }
        if !configuration.is_spanning() {
            return Err(LineConfigurationError::NonspanningStabilizer);
        }
        let restricted =
            build_line_subgroup(configuration.clone(), self.matrix_generators.clone())?;
        if restricted.order != self.order {
            return Err(symmetry_error(
                "restriction lost faithfulness on a spanning subconfiguration",
            ));
        }
        Ok(restricted)
    }

    /// Conjugate this matrix subgroup through a unimodular coordinate map.
    pub fn conjugate(
        &self,
        transformation: &Matrix<IntegerRing>,
    ) -> Result<Self, LineConfigurationError> {
        let target = self.configuration.transform(transformation)?;
        let inverse = unimodular_inverse(transformation)?;
        let generators = self
            .matrix_generators
            .iter()
            .map(|generator| {
                transformation
                    .matmul(generator)
                    .and_then(|left| left.matmul(&inverse))
                    .map_err(linalg_error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let transported = build_line_subgroup(target, generators)?;
        if transported.order != self.order {
            return Err(symmetry_error(
                "coordinate transport changed the subgroup order",
            ));
        }
        Ok(transported)
    }

    /// Recheck all matrix actions, generated actions, and the exact order.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        let rebuilt =
            build_line_subgroup(self.configuration.clone(), self.matrix_generators.clone())?;
        if rebuilt.signed_action.generators() != self.signed_action.generators()
            || rebuilt.line_action.generators() != self.line_action.generators()
            || rebuilt.order != self.order
        {
            return Err(symmetry_error("integral line subgroup did not verify"));
        }
        Ok(())
    }
}

impl SemanticType for IntegralLineSubgroup {
    const TYPE_ID: &'static str = "noether/linalg/IntegralLineSubgroup@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let matrices = self
            .matrix_generators
            .iter()
            .map(|matrix| {
                matrix
                    .rows()
                    .map(|row| row.iter().map(ToString::to_string).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.configuration.canonical_bytes()?,
            matrices,
            self.order.to_string(),
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_LINE_SUBGROUP_DESCRIPTION
    }
}

impl fmt::Debug for IntegralLineSubgroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegralLineSubgroup")
            .field("ambient_rank", &self.configuration.ambient_rank())
            .field("line_count", &self.configuration.line_count())
            .field("generator_count", &self.matrix_generators.len())
            .field("order", &self.order)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for IntegralLineSubgroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "IntegralLineSubgroup(rank={}, lines={}, generators={}, order={})",
            self.configuration.ambient_rank(),
            self.configuration.line_count(),
            self.matrix_generators.len(),
            self.order
        )
    }
}

/// Exact finite action of a configuration stabilizer on tagged external lines.
///
/// Each seed defines its own invariant color class, even when two seeds happen
/// to lie in the same geometric stabilizer orbit.  Points in `action` index the
/// flattened `lines`; `seed_points[i]` is the point occupied by seed `i`, and
/// `orbit_ranges[i]` is its half-open tagged orbit range.  This makes several
/// coface types composable in one faithful-to-the-request finite action without
/// confusing equal vector representatives with equal application-level types.
#[derive(Clone)]
pub struct IntegralLineOrbitAction {
    automorphisms: IntegralLineAutomorphismGroup,
    seeds: Vec<Vec<Integer>>,
    lines: Vec<Vec<Integer>>,
    seed_points: Vec<Point>,
    orbit_ranges: Vec<(usize, usize)>,
    action: PermutationGroup,
}

impl IntegralLineOrbitAction {
    /// Stabilizer whose matrix generators induce this action.
    pub fn automorphisms(&self) -> &IntegralLineAutomorphismGroup {
        &self.automorphisms
    }

    /// Canonical primitive line used to seed each tagged orbit.
    pub fn seeds(&self) -> &[Vec<Integer>] {
        &self.seeds
    }

    /// Flattened canonical primitive lines indexed by the action domain.
    pub fn lines(&self) -> &[Vec<Integer>] {
        &self.lines
    }

    /// Point occupied by each seed in the flattened action domain.
    pub fn seed_points(&self) -> &[Point] {
        &self.seed_points
    }

    /// Half-open flattened point range for each tagged seed orbit.
    pub fn orbit_ranges(&self) -> &[(usize, usize)] {
        &self.orbit_ranges
    }

    /// Simultaneous permutation action induced by the matrix generators.
    pub fn action(&self) -> &PermutationGroup {
        &self.action
    }

    /// Induced action of a verified subgroup on the same external-line domain.
    pub fn subgroup_action(
        &self,
        subgroup: &IntegralLineSubgroup,
    ) -> Result<PermutationGroup, LineConfigurationError> {
        if subgroup.configuration() != self.automorphisms.configuration() {
            return Err(symmetry_error(
                "external-line action and subgroup use different configurations",
            ));
        }
        for generator in subgroup.signed_action().generators() {
            if !self
                .automorphisms
                .signed_action()
                .contains(generator)
                .map_err(group_error)?
            {
                return Err(symmetry_error(
                    "external-line subgroup is not contained in the full stabilizer",
                ));
            }
        }
        induced_external_action(&subgroup.matrix_generators, &self.lines, &self.orbit_ranges)
    }

    /// Canonical line at one action point.
    pub fn line(&self, point: Point) -> Result<&[Integer], LineConfigurationError> {
        self.lines
            .get(point as usize)
            .map(Vec::as_slice)
            .ok_or_else(|| symmetry_error(format!("external-line point {point} is out of range")))
    }

    /// Rebuild every tagged orbit and induced generator action exactly.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        let rebuilt = build_external_line_action(&self.automorphisms, self.seeds.clone())?;
        if rebuilt.seeds != self.seeds
            || rebuilt.lines != self.lines
            || rebuilt.seed_points != self.seed_points
            || rebuilt.orbit_ranges != self.orbit_ranges
            || rebuilt.action.generators() != self.action.generators()
            || rebuilt.action.order() != self.action.order()
        {
            return Err(symmetry_error(
                "external integral-line action did not verify",
            ));
        }
        Ok(())
    }
}

impl SemanticType for IntegralLineOrbitAction {
    const TYPE_ID: &'static str = "noether/linalg/IntegralLineOrbitAction@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let seeds = self
            .seeds
            .iter()
            .map(|line| line.iter().map(ToString::to_string).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.automorphisms.configuration().canonical_bytes()?,
            seeds,
            self.action
                .generators()
                .iter()
                .map(|generator| generator.images())
                .collect::<Vec<_>>(),
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_LINE_ORBIT_ACTION_DESCRIPTION
    }
}

impl fmt::Debug for IntegralLineOrbitAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegralLineOrbitAction")
            .field(
                "ambient_rank",
                &self.automorphisms.configuration().ambient_rank(),
            )
            .field("seed_count", &self.seeds.len())
            .field("degree", &self.action.degree())
            .finish_non_exhaustive()
    }
}

impl fmt::Display for IntegralLineOrbitAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "IntegralLineOrbitAction(rank={}, seeds={}, degree={})",
            self.automorphisms.configuration().ambient_rank(),
            self.seeds.len(),
            self.action.degree()
        )
    }
}

/// Exact finite stabilizer of a spanning integral-line configuration.
///
/// Matrix generators act on column vectors in `Z^N`. `line_action` is their
/// possibly nonfaithful action on unoriented lines; `order` is the order of
/// the faithful action on both signed representatives.
#[derive(Clone)]
pub struct IntegralLineAutomorphismGroup {
    configuration: IntegralLineConfiguration,
    matrix_generators: Vec<Matrix<IntegerRing>>,
    signed_action: PermutationGroup,
    line_action: PermutationGroup,
    orientation_signs: Vec<i8>,
    order: Integer,
    lift_basis: Vec<usize>,
    lift_source_inverse: Matrix<crate::RationalField>,
}

impl IntegralLineAutomorphismGroup {
    /// Configuration stabilized by this group.
    pub fn configuration(&self) -> &IntegralLineConfiguration {
        &self.configuration
    }

    /// Unimodular matrix generators acting on column vectors.
    pub fn matrix_generators(&self) -> &[Matrix<IntegerRing>] {
        &self.matrix_generators
    }

    /// Induced permutation group on unoriented lines.
    pub fn line_action(&self) -> &PermutationGroup {
        &self.line_action
    }

    /// Faithful induced action on the two signed representatives of each line.
    pub fn signed_action(&self) -> &PermutationGroup {
        &self.signed_action
    }

    /// Recover and verify the unique integral matrix inducing a signed action.
    pub fn lift_signed_action(
        &self,
        action: &Permutation,
    ) -> Result<Matrix<IntegerRing>, LineConfigurationError> {
        if !self.signed_action.contains(action).map_err(group_error)? {
            return Err(symmetry_error(
                "signed permutation is not in the configuration stabilizer",
            ));
        }
        let transformation = self.lift_member(action)?;
        let (_, rebuilt_signed) = self.configuration.induced_actions(&transformation)?;
        if rebuilt_signed != *action {
            return Err(symmetry_error(
                "lifted matrix induces a different signed permutation",
            ));
        }
        Ok(transformation)
    }

    /// Apply one stabilizer element to an external primitive integral line.
    ///
    /// The signed action is faithful, so its images on a cached rational basis
    /// determine the unique integral matrix. This operation reuses that basis
    /// factorization and avoids constructing or verifying an orbit containing
    /// every image of the external line.
    pub fn apply_signed_action_to_external_line(
        &self,
        action: &Permutation,
        line: Vec<Integer>,
    ) -> Result<Vec<Integer>, LineConfigurationError> {
        if !self.signed_action.contains(action).map_err(group_error)? {
            return Err(symmetry_error(
                "signed permutation is not in the configuration stabilizer",
            ));
        }
        let checked = IntegralLineConfiguration::new(self.configuration.ambient_rank(), [line])?;
        let transformation = self.lift_member(action)?;
        let rows = transformation.rows().collect::<Vec<_>>();
        Ok(canonicalize_line(apply_matrix(&rows, &checked.vectors()[0])).0)
    }

    fn lift_member(
        &self,
        action: &Permutation,
    ) -> Result<Matrix<IntegerRing>, LineConfigurationError> {
        let images = self
            .lift_basis
            .iter()
            .map(|source_line| {
                point(2 * source_line).map(|source_point| action.apply(source_point) as usize)
            })
            .collect::<Result<Vec<_>, _>>()?;
        recover_transformation_with_inverse(
            &self.configuration,
            &images,
            &self.lift_source_inverse,
        )?
        .ok_or_else(|| symmetry_error("signed action did not lift to an integral matrix"))
    }

    /// Simultaneous exact action on tagged orbits of external primitive lines.
    pub fn action_on_line_orbits(
        &self,
        seeds: impl IntoIterator<Item = Vec<Integer>>,
    ) -> Result<IntegralLineOrbitAction, LineConfigurationError> {
        build_external_line_action(self, seeds.into_iter().collect())
    }

    /// Exact subgroup stabilizing one unoriented configuration line.
    pub fn line_stabilizer(
        &self,
        line_index: usize,
    ) -> Result<IntegralLineSubgroup, LineConfigurationError> {
        if line_index >= self.configuration.line_count() {
            return Err(symmetry_error(format!(
                "line index {line_index} is outside the configuration"
            )));
        }
        let signed_points = [point(2 * line_index)?, point(2 * line_index + 1)?];
        let stabilizer = self
            .signed_action
            .set_stabilizer(&signed_points)
            .map_err(group_error)?;
        let matrices = stabilizer
            .generators()
            .iter()
            .map(|generator| self.lift_signed_action(generator))
            .collect::<Result<Vec<_>, _>>()?;
        let result = build_line_subgroup(self.configuration.clone(), matrices)?;
        if result.signed_action.order() != stabilizer.order() {
            return Err(symmetry_error("lifted line stabilizer has the wrong order"));
        }
        Ok(result)
    }

    /// Exact order of the faithful signed-line action.
    pub fn order(&self) -> &Integer {
        &self.order
    }

    /// Orientation character on the returned matrix generators.
    pub fn orientation_signs(&self) -> &[i8] {
        &self.orientation_signs
    }

    /// Whether some configuration automorphism reverses the rank-one span.
    pub fn has_orientation_reversing(&self) -> bool {
        self.orientation_signs.contains(&-1)
    }

    /// Recheck every matrix action, generated group, order, and orientation sign.
    pub fn verify(&self) -> Result<(), LineConfigurationError> {
        if self.matrix_generators.len() != self.orientation_signs.len() {
            return Err(symmetry_error(
                "matrix-generator and orientation-sign counts differ",
            ));
        }
        let mut signed_generators = Vec::with_capacity(self.matrix_generators.len());
        let mut line_generators = Vec::with_capacity(self.matrix_generators.len());
        for (matrix, claimed_sign) in self.matrix_generators.iter().zip(&self.orientation_signs) {
            let (line, signed) = self.configuration.induced_actions(matrix)?;
            if self.configuration.orientation_sign(matrix)? != *claimed_sign {
                return Err(symmetry_error(
                    "a matrix generator has the wrong orientation character",
                ));
            }
            line_generators.push(line);
            signed_generators.push(signed);
        }
        let signed = PermutationGroup::from_generators(
            2 * self.configuration.line_count(),
            &signed_generators,
        )
        .map_err(group_error)?;
        let line =
            PermutationGroup::from_generators(self.configuration.line_count(), &line_generators)
                .map_err(group_error)?;
        if signed.order() != self.signed_action.order()
            || line.order() != self.line_action.order()
            || Integer::from_str(&signed.order().to_decimal()).map_err(|error| {
                symmetry_error(format!("could not parse verified group order: {error}"))
            })? != self.order
        {
            return Err(symmetry_error(
                "reconstructed permutation actions have different orders",
            ));
        }
        Ok(())
    }
}

impl SemanticType for IntegralLineAutomorphismGroup {
    const TYPE_ID: &'static str = "noether/linalg/IntegralLineAutomorphismGroup@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let matrices = self
            .matrix_generators
            .iter()
            .map(|matrix| {
                matrix
                    .rows()
                    .map(|row| row.iter().map(ToString::to_string).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        serde_json::to_vec(&(
            Self::TYPE_ID,
            self.configuration.canonical_bytes()?,
            matrices,
            &self.orientation_signs,
            self.order.to_string(),
        ))
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &INTEGRAL_LINE_AUTOMORPHISM_GROUP_DESCRIPTION
    }
}

impl fmt::Debug for IntegralLineAutomorphismGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IntegralLineAutomorphismGroup")
            .field("ambient_rank", &self.configuration.ambient_rank())
            .field("line_count", &self.configuration.line_count())
            .field("generator_count", &self.matrix_generators.len())
            .field("order", &self.order)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for IntegralLineAutomorphismGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "IntegralLineAutomorphismGroup(rank={}, lines={}, generators={}, order={})",
            self.configuration.ambient_rank(),
            self.configuration.line_count(),
            self.matrix_generators.len(),
            self.order
        )
    }
}

impl IntegralLineConfiguration {
    /// Permutation induced on unoriented lines by a preserving unimodular matrix.
    pub fn induced_permutation(
        &self,
        transformation: &Matrix<IntegerRing>,
    ) -> Result<Permutation, LineConfigurationError> {
        self.induced_actions(transformation).map(|(line, _)| line)
    }

    /// Faithful permutation induced on both signed representatives of each line.
    pub fn induced_signed_permutation(
        &self,
        transformation: &Matrix<IntegerRing>,
    ) -> Result<Permutation, LineConfigurationError> {
        self.induced_actions(transformation)
            .map(|(_, signed)| signed)
    }

    /// Orientation character of a preserving matrix on the rank-one span.
    ///
    /// This computes the determinant of the exact induced rational action. It
    /// therefore remains correct for non-simplicial configurations, where the
    /// sign of the full ray permutation need not be the orientation character.
    pub fn orientation_sign(
        &self,
        transformation: &Matrix<IntegerRing>,
    ) -> Result<i8, LineConfigurationError> {
        let (line_permutation, _) = self.induced_actions(transformation)?;
        let rank = self.perfection_rank();
        if rank == 0 {
            return Ok(1);
        }
        let rank_one_rows = self.rank_one_matrix().rows().collect::<Vec<_>>();
        let basis_indices = independent_row_indices(&rank_one_rows, rank);
        if basis_indices.len() != rank {
            return Err(symmetry_error(
                "could not extract a basis of the rank-one span",
            ));
        }
        let basis_rows = basis_indices
            .iter()
            .map(|index| rank_one_rows[*index].clone())
            .collect::<Vec<_>>();
        let basis_transpose = rational_matrix(&basis_rows).transpose();
        let mut columns = Vec::with_capacity(rank);
        for source in &basis_indices {
            let target = line_permutation.images()[*source] as usize;
            let rhs =
                Vector::from_values(Q, rank_one_rows[target].iter().cloned().map(Rational::from))
                    .map_err(linalg_error)?;
            let solution = basis_transpose.solve(&rhs).map_err(linalg_error)?;
            if solution.dimension() != 0 {
                return Err(symmetry_error("rank-one basis coordinates were not unique"));
            }
            columns.push(solution.particular().values().collect::<Vec<_>>());
        }
        let action = Matrix::from_rows(
            Q,
            (0..rank).map(|row| {
                (0..rank)
                    .map(|column| columns[column][row].clone())
                    .collect::<Vec<_>>()
            }),
        )
        .map_err(linalg_error)?;
        let determinant = action.det().map_err(linalg_error)?;
        if determinant.numerator().abs() != determinant.denominator() {
            return Err(LineConfigurationError::InvalidOrientationDeterminant { determinant });
        }
        Ok(if determinant.is_positive() { 1 } else { -1 })
    }

    /// Compute the exact finite integral stabilizer of a spanning configuration.
    pub fn automorphism_group(
        &self,
    ) -> Result<IntegralLineAutomorphismGroup, LineConfigurationError> {
        if !self.is_spanning() {
            return Err(LineConfigurationError::NonspanningStabilizer);
        }
        ensure_permutation_domain(self.line_count())?;
        let mut search = SymmetrySearch::new(self)?;
        search.visit(&mut Vec::new(), &mut vec![false; self.line_count()])?;
        let order =
            Integer::from_str(&search.signed_group.order().to_decimal()).map_err(|error| {
                symmetry_error(format!(
                    "could not parse configuration group order: {error}"
                ))
            })?;
        let orientation_signs = search
            .matrix_generators
            .iter()
            .map(|matrix| self.orientation_sign(matrix))
            .collect::<Result<Vec<_>, _>>()?;
        let (lift_basis, lift_source_inverse) = lift_context(self)?;
        let result = IntegralLineAutomorphismGroup {
            configuration: self.clone(),
            matrix_generators: search.matrix_generators,
            signed_action: search.signed_group,
            line_action: search.line_group,
            orientation_signs,
            order,
            lift_basis,
            lift_source_inverse,
        };
        result.verify()?;
        Ok(result)
    }

    /// Find a verified unimodular equivalence to another spanning configuration.
    pub fn equivalence_to(
        &self,
        target: &Self,
    ) -> Result<Option<IntegralLineEquivalence>, LineConfigurationError> {
        if !self.equivalence_invariants_match(target)? {
            return Ok(None);
        }
        self.ensure_spanning_equivalence(target)?;
        if let Some(identity) = self.identity_equivalence(target)? {
            return Ok(Some(identity));
        }
        let target_automorphisms = target.automorphism_group()?;
        self.search_equivalence(target, &target_automorphisms)
    }

    /// Find an equivalence while reusing a previously computed target stabilizer.
    ///
    /// This has the same exact semantics as [`Self::equivalence_to`]. It avoids
    /// recomputing the target automorphism group when many configurations are
    /// classified against one orbit catalog.
    pub fn equivalence_to_with_target_automorphisms(
        &self,
        target: &Self,
        target_automorphisms: &IntegralLineAutomorphismGroup,
    ) -> Result<Option<IntegralLineEquivalence>, LineConfigurationError> {
        if !self.equivalence_invariants_match(target)? {
            return Ok(None);
        }
        self.ensure_spanning_equivalence(target)?;
        if target_automorphisms.configuration() != target {
            return Err(symmetry_error(
                "the supplied automorphism group belongs to a different target configuration",
            ));
        }
        if let Some(identity) = self.identity_equivalence(target)? {
            return Ok(Some(identity));
        }
        self.search_equivalence(target, target_automorphisms)
    }

    fn equivalence_invariants_match(&self, target: &Self) -> Result<bool, LineConfigurationError> {
        Ok(self.ambient_rank() == target.ambient_rank()
            && self.line_count() == target.line_count()
            && self.span_rank() == target.span_rank()
            && self.perfection_rank() == target.perfection_rank()
            && self.barycenter_form().det().map_err(linalg_error)?
                == target.barycenter_form().det().map_err(linalg_error)?)
    }

    fn ensure_spanning_equivalence(&self, target: &Self) -> Result<(), LineConfigurationError> {
        if !self.is_spanning() || !target.is_spanning() {
            Err(LineConfigurationError::NonspanningEquivalence)
        } else {
            Ok(())
        }
    }

    fn identity_equivalence(
        &self,
        target: &Self,
    ) -> Result<Option<IntegralLineEquivalence>, LineConfigurationError> {
        if self != target {
            return Ok(None);
        }
        let transformation = Matrix::identity(Z, self.ambient_rank());
        let (line_permutation, _) = equivalence_actions(self, target, &transformation)?;
        Ok(Some(IntegralLineEquivalence {
            source: self.clone(),
            target: target.clone(),
            transformation,
            line_permutation,
        }))
    }

    fn search_equivalence(
        &self,
        target: &Self,
        target_automorphisms: &IntegralLineAutomorphismGroup,
    ) -> Result<Option<IntegralLineEquivalence>, LineConfigurationError> {
        let mut search =
            EquivalenceSearch::new(self, target, target_automorphisms.signed_action())?;
        search.visit(&mut Vec::new(), &mut vec![false; target.line_count()])?;
        if let Some((transformation, line_permutation)) = search.found {
            let result = IntegralLineEquivalence {
                source: self.clone(),
                target: target.clone(),
                transformation,
                line_permutation,
            };
            result.verify()?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    fn induced_actions(
        &self,
        transformation: &Matrix<IntegerRing>,
    ) -> Result<(Permutation, Permutation), LineConfigurationError> {
        equivalence_actions(self, self, transformation)
    }
}

struct EquivalenceSearch<'a> {
    source: &'a IntegralLineConfiguration,
    target: &'a IntegralLineConfiguration,
    source_basis: Vec<usize>,
    source_dual_products: Vec<Vec<Rational>>,
    target_dual_products: Vec<Vec<Rational>>,
    target_group: &'a PermutationGroup,
    nodes: usize,
    found: Option<(Matrix<IntegerRing>, Permutation)>,
}

impl<'a> EquivalenceSearch<'a> {
    fn new(
        source: &'a IntegralLineConfiguration,
        target: &'a IntegralLineConfiguration,
        target_group: &'a PermutationGroup,
    ) -> Result<Self, LineConfigurationError> {
        let source_basis = independent_row_indices(source.vectors(), source.ambient_rank());
        if source_basis.len() != source.ambient_rank() {
            return Err(LineConfigurationError::NonspanningEquivalence);
        }
        Ok(Self {
            source,
            target,
            source_basis,
            source_dual_products: dual_inner_products(source)?,
            target_dual_products: dual_inner_products(target)?,
            target_group,
            nodes: 0,
            found: None,
        })
    }

    fn visit(
        &mut self,
        images: &mut Vec<usize>,
        used_lines: &mut [bool],
    ) -> Result<(), LineConfigurationError> {
        if self.found.is_some() {
            return Ok(());
        }
        self.nodes += 1;
        if self.nodes > SYMMETRY_SEARCH_NODE_CAP {
            return Err(LineConfigurationError::SymmetrySearchBudgetExceeded {
                cap: SYMMETRY_SEARCH_NODE_CAP,
            });
        }
        if images.len() == self.source_basis.len() {
            let Some(transformation) =
                recover_transformation(self.source, self.target, &self.source_basis, images)?
            else {
                return Ok(());
            };
            if let Ok((line, _)) = equivalence_actions(self.source, self.target, &transformation) {
                self.found = Some((transformation, line));
            }
            return Ok(());
        }

        let depth = images.len();
        let source_line = self.source_basis[depth];
        let mut explored = Vec::<Point>::new();
        for candidate in 0..2 * self.target.line_count() {
            let target_line = candidate / 2;
            if used_lines[target_line] || !self.compatible(source_line, candidate, images) {
                continue;
            }
            let candidate_point = point(candidate)?;
            let fixed = images
                .iter()
                .copied()
                .map(point)
                .collect::<Result<Vec<_>, _>>()?;
            let stabilizer = self
                .target_group
                .pointwise_stabilizer(&fixed)
                .map_err(group_error)?;
            let mut covered = false;
            for representative in &explored {
                if stabilizer
                    .orbit(*representative)
                    .map_err(group_error)?
                    .contains(&candidate_point)
                {
                    covered = true;
                    break;
                }
            }
            if covered {
                continue;
            }
            images.push(candidate);
            used_lines[target_line] = true;
            self.visit(images, used_lines)?;
            used_lines[target_line] = false;
            images.pop();
            explored.push(candidate_point);
            if self.found.is_some() {
                break;
            }
        }
        Ok(())
    }

    fn compatible(&self, source: usize, candidate: usize, images: &[usize]) -> bool {
        let target = candidate / 2;
        let sign = signed_sign(candidate);
        if self.source_dual_products[source][source]
            != signed_rational(&self.target_dual_products[target][target], sign * sign)
        {
            return false;
        }
        images.iter().enumerate().all(|(position, previous)| {
            let previous_source = self.source_basis[position];
            let previous_target = previous / 2;
            let previous_sign = signed_sign(*previous);
            self.source_dual_products[source][previous_source]
                == signed_rational(
                    &self.target_dual_products[target][previous_target],
                    sign * previous_sign,
                )
        })
    }
}

struct SymmetrySearch<'a> {
    configuration: &'a IntegralLineConfiguration,
    source_basis: Vec<usize>,
    dual_products: Vec<Vec<Rational>>,
    nodes: usize,
    matrix_generators: Vec<Matrix<IntegerRing>>,
    signed_generators: Vec<Permutation>,
    line_generators: Vec<Permutation>,
    signed_group: PermutationGroup,
    line_group: PermutationGroup,
}

impl<'a> SymmetrySearch<'a> {
    fn new(configuration: &'a IntegralLineConfiguration) -> Result<Self, LineConfigurationError> {
        let source_basis =
            independent_row_indices(configuration.vectors(), configuration.ambient_rank());
        if source_basis.len() != configuration.ambient_rank() {
            return Err(LineConfigurationError::NonspanningStabilizer);
        }
        let signed_group = PermutationGroup::from_generators(2 * configuration.line_count(), &[])
            .map_err(group_error)?;
        let line_group = PermutationGroup::from_generators(configuration.line_count(), &[])
            .map_err(group_error)?;
        Ok(Self {
            configuration,
            source_basis,
            dual_products: dual_inner_products(configuration)?,
            nodes: 0,
            matrix_generators: Vec::new(),
            signed_generators: Vec::new(),
            line_generators: Vec::new(),
            signed_group,
            line_group,
        })
    }

    fn visit(
        &mut self,
        images: &mut Vec<usize>,
        used_lines: &mut [bool],
    ) -> Result<(), LineConfigurationError> {
        self.nodes += 1;
        if self.nodes > SYMMETRY_SEARCH_NODE_CAP {
            return Err(LineConfigurationError::SymmetrySearchBudgetExceeded {
                cap: SYMMETRY_SEARCH_NODE_CAP,
            });
        }
        if images.len() == self.source_basis.len() {
            return self.visit_leaf(images);
        }

        let depth = images.len();
        let source = self.source_basis[depth];
        let mut explored = Vec::<Point>::new();
        for candidate in 0..2 * self.configuration.line_count() {
            let target_line = candidate / 2;
            if used_lines[target_line] || !self.compatible(source, candidate, images) {
                continue;
            }
            let candidate_point = point(candidate)?;
            let fixed = images
                .iter()
                .copied()
                .map(point)
                .collect::<Result<Vec<_>, _>>()?;
            let stabilizer = self
                .signed_group
                .pointwise_stabilizer(&fixed)
                .map_err(group_error)?;
            let mut covered = false;
            for representative in &explored {
                if stabilizer
                    .orbit(*representative)
                    .map_err(group_error)?
                    .contains(&candidate_point)
                {
                    covered = true;
                    break;
                }
            }
            if covered {
                continue;
            }

            images.push(candidate);
            used_lines[target_line] = true;
            self.visit(images, used_lines)?;
            used_lines[target_line] = false;
            images.pop();
            explored.push(candidate_point);
        }
        Ok(())
    }

    fn compatible(&self, source: usize, candidate: usize, images: &[usize]) -> bool {
        let target = candidate / 2;
        let sign = signed_sign(candidate);
        if self.dual_products[source][source]
            != signed_rational(&self.dual_products[target][target], sign * sign)
        {
            return false;
        }
        images.iter().enumerate().all(|(position, previous)| {
            let previous_source = self.source_basis[position];
            let previous_target = previous / 2;
            let previous_sign = signed_sign(*previous);
            self.dual_products[source][previous_source]
                == signed_rational(
                    &self.dual_products[target][previous_target],
                    sign * previous_sign,
                )
        })
    }

    fn visit_leaf(&mut self, images: &[usize]) -> Result<(), LineConfigurationError> {
        let Some(transformation) = recover_transformation(
            self.configuration,
            self.configuration,
            &self.source_basis,
            images,
        )?
        else {
            return Ok(());
        };
        let Ok((line, signed)) = self.configuration.induced_actions(&transformation) else {
            return Ok(());
        };
        if self.signed_group.contains(&signed).map_err(group_error)? {
            return Ok(());
        }
        self.matrix_generators.push(transformation);
        self.signed_generators.push(signed);
        self.line_generators.push(line);
        self.signed_group = PermutationGroup::from_generators(
            2 * self.configuration.line_count(),
            &self.signed_generators,
        )
        .map_err(group_error)?;
        self.line_group = PermutationGroup::from_generators(
            self.configuration.line_count(),
            &self.line_generators,
        )
        .map_err(group_error)?;
        Ok(())
    }
}

fn independent_row_indices(rows: &[Vec<Integer>], target_rank: usize) -> Vec<usize> {
    let column_count = rows.first().map_or(0, Vec::len);
    let mut selected = Vec::new();
    let mut selected_rows = Vec::new();
    let mut rank = 0;
    for (index, row) in rows.iter().enumerate() {
        selected_rows.push(row.clone());
        let matrix = if selected_rows.is_empty() {
            Matrix::zeros(Z, 0, column_count)
        } else {
            Matrix::from_rows(Z, selected_rows.clone())
                .expect("candidate independent rows have one checked dimension")
        };
        let next_rank = matrix.rank_backend();
        if next_rank > rank {
            selected.push(index);
            rank = next_rank;
            if rank == target_rank {
                break;
            }
        } else {
            selected_rows.pop();
        }
    }
    selected
}

fn dual_inner_products(
    configuration: &IntegralLineConfiguration,
) -> Result<Vec<Vec<Rational>>, LineConfigurationError> {
    let barycenter = rational_matrix(&configuration.barycenter_form().rows().collect::<Vec<_>>());
    let inverse_images = configuration
        .vectors()
        .iter()
        .map(|line| {
            let rhs = Vector::from_values(Q, line.iter().cloned().map(Rational::from))
                .map_err(linalg_error)?;
            let solution = barycenter.solve(&rhs).map_err(linalg_error)?;
            if solution.dimension() != 0 {
                return Err(symmetry_error(
                    "barycenter form did not have a unique solve",
                ));
            }
            Ok(solution.particular().values().collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, LineConfigurationError>>()?;
    Ok(configuration
        .vectors()
        .iter()
        .map(|left| {
            inverse_images
                .iter()
                .map(|right| {
                    left.iter()
                        .zip(right)
                        .fold(Rational::zero(), |sum, (a, b)| {
                            &sum + &(&Rational::from(a.clone()) * b)
                        })
                })
                .collect()
        })
        .collect())
}

fn recover_transformation(
    source: &IntegralLineConfiguration,
    target: &IntegralLineConfiguration,
    source_basis: &[usize],
    images: &[usize],
) -> Result<Option<Matrix<IntegerRing>>, LineConfigurationError> {
    let rank = source.ambient_rank();
    let source_columns = Matrix::from_rows(
        Q,
        (0..rank).map(|coordinate| {
            source_basis
                .iter()
                .map(|source_line| {
                    Rational::from(source.vectors()[*source_line][coordinate].clone())
                })
                .collect::<Vec<_>>()
        }),
    )
    .map_err(linalg_error)?;
    let source_transpose = source_columns.transpose();
    let mut rows = Vec::with_capacity(rank);
    for output_coordinate in 0..rank {
        let rhs = Vector::from_values(
            Q,
            images.iter().map(|image| {
                let value = Rational::from(target.vectors()[image / 2][output_coordinate].clone());
                signed_rational(&value, signed_sign(*image))
            }),
        )
        .map_err(linalg_error)?;
        let solution = source_transpose.solve(&rhs).map_err(linalg_error)?;
        if solution.dimension() != 0 {
            return Ok(None);
        }
        let mut row = Vec::with_capacity(rank);
        for value in solution.particular().values() {
            if !value.denominator().is_one() {
                return Ok(None);
            }
            row.push(value.numerator());
        }
        rows.push(row);
    }
    let transformation = Matrix::from_rows(Z, rows).map_err(linalg_error)?;
    let determinant = transformation.det().map_err(linalg_error)?;
    if !determinant.abs().is_one() {
        return Ok(None);
    }
    Ok(Some(transformation))
}

fn lift_context(
    configuration: &IntegralLineConfiguration,
) -> Result<(Vec<usize>, Matrix<crate::RationalField>), LineConfigurationError> {
    let rank = configuration.ambient_rank();
    let basis = independent_row_indices(configuration.vectors(), rank);
    if basis.len() != rank {
        return Err(symmetry_error(
            "could not extract a basis for signed-action lifting",
        ));
    }
    let source_columns = Matrix::from_rows(
        Q,
        (0..rank).map(|coordinate| {
            basis
                .iter()
                .map(|source_line| {
                    Rational::from(configuration.vectors()[*source_line][coordinate].clone())
                })
                .collect::<Vec<_>>()
        }),
    )
    .map_err(linalg_error)?;
    let mut inverse_columns = Vec::with_capacity(rank);
    for column in 0..rank {
        let rhs = Vector::from_values(
            Q,
            (0..rank).map(|row| Rational::from(i64::from(row == column))),
        )
        .map_err(linalg_error)?;
        let solution = source_columns.solve(&rhs).map_err(linalg_error)?;
        if solution.dimension() != 0 {
            return Err(symmetry_error(
                "signed-action lift basis did not have a unique inverse",
            ));
        }
        inverse_columns.push(solution.particular().values().collect::<Vec<_>>());
    }
    let inverse = Matrix::from_rows(
        Q,
        (0..rank).map(|row| {
            (0..rank)
                .map(|column| inverse_columns[column][row].clone())
                .collect::<Vec<_>>()
        }),
    )
    .map_err(linalg_error)?;
    if source_columns.matmul(&inverse).map_err(linalg_error)? != Matrix::identity(Q, rank) {
        return Err(symmetry_error(
            "signed-action lift basis inverse failed verification",
        ));
    }
    Ok((basis, inverse))
}

fn recover_transformation_with_inverse(
    target: &IntegralLineConfiguration,
    images: &[usize],
    source_inverse: &Matrix<crate::RationalField>,
) -> Result<Option<Matrix<IntegerRing>>, LineConfigurationError> {
    let rank = target.ambient_rank();
    if images.len() != rank || source_inverse.shape() != (rank, rank) {
        return Err(symmetry_error(
            "signed-action lift context has the wrong rank",
        ));
    }
    let target_columns = Matrix::from_rows(
        Q,
        (0..rank).map(|coordinate| {
            images
                .iter()
                .map(|image| {
                    let value = Rational::from(target.vectors()[image / 2][coordinate].clone());
                    signed_rational(&value, signed_sign(*image))
                })
                .collect::<Vec<_>>()
        }),
    )
    .map_err(linalg_error)?;
    let rational = target_columns
        .matmul(source_inverse)
        .map_err(linalg_error)?;
    let mut rows = Vec::with_capacity(rank);
    for row in rational.rows() {
        let mut integer_row = Vec::with_capacity(rank);
        for value in row {
            if !value.denominator().is_one() {
                return Ok(None);
            }
            integer_row.push(value.numerator());
        }
        rows.push(integer_row);
    }
    Matrix::from_rows(Z, rows).map(Some).map_err(linalg_error)
}

fn rational_matrix(rows: &[Vec<Integer>]) -> Matrix<crate::RationalField> {
    if rows.is_empty() {
        Matrix::zeros(Q, 0, 0)
    } else {
        Matrix::from_rows(
            Q,
            rows.iter()
                .map(|row| row.iter().cloned().map(Rational::from)),
        )
        .expect("integer rows retain their checked shape over the rationals")
    }
}

fn equivalence_actions(
    source: &IntegralLineConfiguration,
    target: &IntegralLineConfiguration,
    transformation: &Matrix<IntegerRing>,
) -> Result<(Permutation, Permutation), LineConfigurationError> {
    if source.ambient_rank() != target.ambient_rank() || source.line_count() != target.line_count()
    {
        return Err(LineConfigurationError::TransformationDoesNotPreserveConfiguration);
    }
    ensure_permutation_domain(source.line_count())?;
    if source.transform(transformation)? != *target {
        return Err(LineConfigurationError::TransformationDoesNotPreserveConfiguration);
    }
    let transformation_rows = transformation.rows().collect::<Vec<_>>();
    let mut line_images = Vec::with_capacity(source.line_count());
    let mut signed_images = vec![0; 2 * source.line_count()];
    for (source_index, line) in source.vectors().iter().enumerate() {
        let image = apply_matrix(&transformation_rows, line);
        let (canonical, negative) = canonicalize_line(image);
        let target_index = target
            .vectors()
            .binary_search(&canonical)
            .map_err(|_| LineConfigurationError::TransformationDoesNotPreserveConfiguration)?;
        line_images.push(point(target_index)?);
        signed_images[2 * source_index] = point(2 * target_index + usize::from(negative))?;
        signed_images[2 * source_index + 1] = point(2 * target_index + usize::from(!negative))?;
    }
    let line = Permutation::from_images(&line_images).map_err(group_error)?;
    let signed = Permutation::from_images(&signed_images).map_err(group_error)?;
    Ok((line, signed))
}

fn build_line_subgroup(
    configuration: IntegralLineConfiguration,
    matrix_generators: Vec<Matrix<IntegerRing>>,
) -> Result<IntegralLineSubgroup, LineConfigurationError> {
    if !configuration.is_spanning() {
        return Err(LineConfigurationError::NonspanningStabilizer);
    }
    let mut line_generators = Vec::with_capacity(matrix_generators.len());
    let mut signed_generators = Vec::with_capacity(matrix_generators.len());
    for matrix in &matrix_generators {
        let (line, signed) = configuration.induced_actions(matrix)?;
        line_generators.push(line);
        signed_generators.push(signed);
    }
    let signed_action =
        PermutationGroup::from_generators(2 * configuration.line_count(), &signed_generators)
            .map_err(group_error)?;
    let line_action =
        PermutationGroup::from_generators(configuration.line_count(), &line_generators)
            .map_err(group_error)?;
    let order = Integer::from_str(&signed_action.order().to_decimal()).map_err(|error| {
        symmetry_error(format!("could not parse verified subgroup order: {error}"))
    })?;
    Ok(IntegralLineSubgroup {
        configuration,
        matrix_generators,
        signed_action,
        line_action,
        order,
    })
}

fn unimodular_inverse(
    transformation: &Matrix<IntegerRing>,
) -> Result<Matrix<IntegerRing>, LineConfigurationError> {
    let (rows, columns) = transformation.shape();
    if rows != columns {
        return Err(LineConfigurationError::TransformationShape {
            expected: (rows, rows),
            actual: (rows, columns),
        });
    }
    let determinant = transformation.det().map_err(linalg_error)?;
    if !determinant.abs().is_one() {
        return Err(LineConfigurationError::NonUnimodularTransformation { determinant });
    }
    let rational = rational_matrix(&transformation.rows().collect::<Vec<_>>());
    let mut inverse_columns = Vec::with_capacity(rows);
    for column in 0..rows {
        let rhs = Vector::from_values(
            Q,
            (0..rows).map(|row| Rational::from(i64::from(row == column))),
        )
        .map_err(linalg_error)?;
        let solution = rational.solve(&rhs).map_err(linalg_error)?;
        if solution.dimension() != 0 {
            return Err(symmetry_error("unimodular inverse was not unique"));
        }
        let entries = solution
            .particular()
            .values()
            .map(|entry| {
                if !entry.denominator().is_one() {
                    return Err(symmetry_error(
                        "unimodular inverse unexpectedly had a denominator",
                    ));
                }
                Ok(entry.numerator())
            })
            .collect::<Result<Vec<_>, _>>()?;
        inverse_columns.push(entries);
    }
    let inverse = Matrix::from_rows(
        Z,
        (0..rows).map(|row| {
            (0..rows)
                .map(|column| inverse_columns[column][row].clone())
                .collect::<Vec<_>>()
        }),
    )
    .map_err(linalg_error)?;
    if transformation.matmul(&inverse).map_err(linalg_error)? != Matrix::identity(Z, rows)
        || inverse.matmul(transformation).map_err(linalg_error)? != Matrix::identity(Z, rows)
    {
        return Err(symmetry_error("unimodular inverse failed verification"));
    }
    Ok(inverse)
}

fn induced_external_action(
    matrix_generators: &[Matrix<IntegerRing>],
    lines: &[Vec<Integer>],
    orbit_ranges: &[(usize, usize)],
) -> Result<PermutationGroup, LineConfigurationError> {
    let generator_rows = matrix_generators
        .iter()
        .map(|matrix| matrix.rows().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut generator_images = vec![Vec::<Point>::with_capacity(lines.len()); generator_rows.len()];
    for (start, end) in orbit_ranges {
        let orbit = &lines[*start..*end];
        for (generator_index, rows) in generator_rows.iter().enumerate() {
            for line in orbit {
                let (image, _) = canonicalize_line(apply_matrix(rows, line));
                let target = orbit.binary_search(&image).map_err(|_| {
                    symmetry_error("external line orbit is not closed under a subgroup generator")
                })?;
                generator_images[generator_index].push(point(start + target)?);
            }
        }
    }
    let generators = generator_images
        .iter()
        .map(|images| Permutation::from_images(images).map_err(group_error))
        .collect::<Result<Vec<_>, _>>()?;
    PermutationGroup::from_generators(lines.len(), &generators).map_err(group_error)
}

fn build_external_line_action(
    automorphisms: &IntegralLineAutomorphismGroup,
    seeds: Vec<Vec<Integer>>,
) -> Result<IntegralLineOrbitAction, LineConfigurationError> {
    let ambient_rank = automorphisms.configuration().ambient_rank();
    let seeds = seeds
        .into_iter()
        .enumerate()
        .map(|(index, seed)| {
            let singleton = IntegralLineConfiguration::new(ambient_rank, [seed]).map_err(
                |error| match error {
                    LineConfigurationError::DimensionMismatch {
                        expected, actual, ..
                    } => LineConfigurationError::DimensionMismatch {
                        line: index,
                        expected,
                        actual,
                    },
                    LineConfigurationError::ZeroVector { .. } => {
                        LineConfigurationError::ZeroVector { line: index }
                    }
                    LineConfigurationError::NonPrimitive { content, .. } => {
                        LineConfigurationError::NonPrimitive {
                            line: index,
                            content,
                        }
                    }
                    other => other,
                },
            )?;
            Ok(singleton.vectors()[0].clone())
        })
        .collect::<Result<Vec<_>, LineConfigurationError>>()?;
    let generator_rows = automorphisms
        .matrix_generators()
        .iter()
        .map(|matrix| matrix.rows().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut lines = Vec::new();
    let mut seed_points = Vec::with_capacity(seeds.len());
    let mut orbit_ranges = Vec::with_capacity(seeds.len());
    let mut generator_images = vec![Vec::<Point>::new(); generator_rows.len()];

    for seed in &seeds {
        let mut seen = BTreeSet::from([seed.clone()]);
        let mut queue = VecDeque::from([seed.clone()]);
        while let Some(current) = queue.pop_front() {
            for rows in &generator_rows {
                let (image, _) = canonicalize_line(apply_matrix(rows, &current));
                if seen.insert(image.clone()) {
                    if lines.len() + seen.len() > EXTERNAL_LINE_ORBIT_CAP {
                        return Err(LineConfigurationError::LineOrbitEnumerationLimitExceeded {
                            cap: EXTERNAL_LINE_ORBIT_CAP,
                        });
                    }
                    queue.push_back(image);
                }
            }
        }
        let orbit = seen.into_iter().collect::<Vec<_>>();
        let start = lines.len();
        let end = start + orbit.len();
        if end > Point::MAX as usize {
            return Err(LineConfigurationError::PermutationDomainTooLarge { line_count: end });
        }
        let seed_offset = orbit
            .binary_search(seed)
            .map_err(|_| symmetry_error("external line seed disappeared from its orbit"))?;
        seed_points.push(point(start + seed_offset)?);
        orbit_ranges.push((start, end));
        for (generator_index, rows) in generator_rows.iter().enumerate() {
            for line in &orbit {
                let (image, _) = canonicalize_line(apply_matrix(rows, line));
                let target = orbit.binary_search(&image).map_err(|_| {
                    symmetry_error("external line orbit is not closed under a generator")
                })?;
                generator_images[generator_index].push(point(start + target)?);
            }
        }
        lines.extend(orbit);
    }

    let generators = generator_images
        .iter()
        .map(|images| Permutation::from_images(images).map_err(group_error))
        .collect::<Result<Vec<_>, _>>()?;
    let action =
        PermutationGroup::from_generators(lines.len(), &generators).map_err(group_error)?;
    Ok(IntegralLineOrbitAction {
        automorphisms: automorphisms.clone(),
        seeds,
        lines,
        seed_points,
        orbit_ranges,
        action,
    })
}

fn apply_matrix(rows: &[Vec<Integer>], vector: &[Integer]) -> Vec<Integer> {
    rows.iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .fold(Integer::zero(), |sum, (left, right)| &sum + &(left * right))
        })
        .collect()
}

fn canonicalize_line(mut line: Vec<Integer>) -> (Vec<Integer>, bool) {
    let negative = line
        .iter()
        .find(|entry| !entry.is_zero())
        .is_some_and(Integer::is_negative);
    if negative {
        for entry in &mut line {
            *entry = -entry.clone();
        }
    }
    (line, negative)
}

fn signed_sign(signed_point: usize) -> i8 {
    if signed_point.is_multiple_of(2) {
        1
    } else {
        -1
    }
}

fn signed_rational(value: &Rational, sign: i8) -> Rational {
    if sign < 0 {
        -value.clone()
    } else {
        value.clone()
    }
}

fn ensure_permutation_domain(line_count: usize) -> Result<(), LineConfigurationError> {
    let signed_count = line_count
        .checked_mul(2)
        .ok_or(LineConfigurationError::PermutationDomainTooLarge { line_count })?;
    if signed_count > Point::MAX as usize {
        return Err(LineConfigurationError::PermutationDomainTooLarge { line_count });
    }
    Ok(())
}

fn point(index: usize) -> Result<Point, LineConfigurationError> {
    Point::try_from(index).map_err(|_| LineConfigurationError::PermutationDomainTooLarge {
        line_count: index / 2,
    })
}

fn symmetry_error(message: impl Into<String>) -> LineConfigurationError {
    LineConfigurationError::SymmetryComputation {
        message: message.into(),
    }
}

fn group_error(error: crate::groups::GroupError) -> LineConfigurationError {
    symmetry_error(error.to_string())
}

fn linalg_error(error: super::LinalgError) -> LineConfigurationError {
    symmetry_error(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integers(rows: &[&[i64]]) -> Vec<Vec<Integer>> {
        rows.iter()
            .map(|row| row.iter().copied().map(Integer::from).collect())
            .collect()
    }

    #[test]
    fn signed_coordinate_axes_have_the_hyperoctahedral_stabilizer() {
        let configuration =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[0, 1]])).unwrap();
        let group = configuration.automorphism_group().unwrap();
        assert_eq!(group.order(), &Integer::from(8));
        assert_eq!(group.line_action().order().to_decimal(), "2");
        assert!(group.has_orientation_reversing());
        for action in group.signed_action().generators() {
            let matrix = group.lift_signed_action(action).unwrap();
            assert_eq!(
                configuration.induced_signed_permutation(&matrix).unwrap(),
                *action
            );
            let external_line = integers(&[&[1, 2]])[0].clone();
            let expected = canonicalize_line(apply_matrix(
                &matrix.rows().collect::<Vec<_>>(),
                &external_line,
            ))
            .0;
            assert_eq!(
                group
                    .apply_signed_action_to_external_line(action, external_line)
                    .unwrap(),
                expected
            );
        }
        group.verify().unwrap();
    }

    #[test]
    fn stabilizer_builds_a_tagged_exact_action_on_external_lines() {
        let configuration =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[0, 1]])).unwrap();
        let group = configuration.automorphism_group().unwrap();
        let action = group
            .action_on_line_orbits(integers(&[&[1, 1], &[1, 2]]))
            .unwrap();

        assert_eq!(action.seeds(), integers(&[&[1, 1], &[1, 2]]));
        assert_eq!(action.orbit_ranges(), &[(0, 2), (2, 6)]);
        assert_eq!(action.seed_points().len(), 2);
        assert_eq!(action.action().degree(), 6);
        assert_eq!(action.action().order().to_decimal(), "4");
        assert_eq!(
            action
                .action()
                .pair_orbits(action.seed_points()[0], action.seed_points()[1])
                .unwrap()
                .len(),
            2
        );
        action.verify().unwrap();
    }

    #[test]
    fn line_stabilizers_restrict_and_transport_exactly() {
        let face = IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[0, 1]])).unwrap();
        let upper =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[0, 1], &[1, 1]])).unwrap();
        let omitted = upper
            .vectors()
            .binary_search(&integers(&[&[1, 1]])[0])
            .unwrap();
        let stabilizer = upper
            .automorphism_group()
            .unwrap()
            .line_stabilizer(omitted)
            .unwrap();
        let restricted = stabilizer.restrict(&face).unwrap();
        assert_eq!(restricted.order(), &Integer::from(4));
        let transformation = Matrix::from_rows(Z, integers(&[&[1, 1], &[0, 1]])).unwrap();
        let transported = restricted.conjugate(&transformation).unwrap();
        assert_eq!(
            transported.configuration(),
            &face.transform(&transformation).unwrap()
        );
        assert_eq!(transported.order(), restricted.order());
        transported.verify().unwrap();
    }

    #[test]
    fn nontrivial_index_span_filters_nonintegral_signed_symmetries() {
        let configuration =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[1, 3]])).unwrap();
        let group = configuration.automorphism_group().unwrap();
        assert_eq!(group.order(), &Integer::from(4));
        group.verify().unwrap();
    }

    #[test]
    fn nonsimplicial_orientation_uses_the_rank_one_span() {
        let configuration =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[0, 1], &[1, 1], &[1, -1]]))
                .unwrap();
        assert_eq!(configuration.line_count(), 4);
        assert_eq!(configuration.perfection_rank(), 3);
        let reflection = Matrix::from_rows(Z, integers(&[&[0, 1], &[1, 0]])).unwrap();
        let permutation = configuration.induced_permutation(&reflection).unwrap();
        assert_eq!(permutation.degree(), 4);
        assert_eq!(configuration.orientation_sign(&reflection).unwrap(), -1);
    }

    #[test]
    fn nonspanning_configuration_has_no_finite_line_stabilizer() {
        let configuration = IntegralLineConfiguration::new(2, integers(&[&[1, 0]])).unwrap();
        assert!(matches!(
            configuration.automorphism_group(),
            Err(LineConfigurationError::NonspanningStabilizer)
        ));
    }

    #[test]
    fn equivalence_returns_a_verified_unimodular_witness() {
        let source =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[0, 1], &[1, 1]])).unwrap();
        let transformation = Matrix::from_rows(Z, integers(&[&[1, 1], &[0, 1]])).unwrap();
        let target = source.transform(&transformation).unwrap();
        assert_ne!(source, target);

        let equivalence = source.equivalence_to(&target).unwrap().unwrap();
        assert_eq!(
            source.transform(equivalence.transformation()).unwrap(),
            target
        );
        equivalence.verify().unwrap();

        let target_automorphisms = target.automorphism_group().unwrap();
        let reused = source
            .equivalence_to_with_target_automorphisms(&target, &target_automorphisms)
            .unwrap()
            .unwrap();
        reused.verify().unwrap();

        let wrong_automorphisms = source.automorphism_group().unwrap();
        assert!(source
            .equivalence_to_with_target_automorphisms(&target, &wrong_automorphisms)
            .is_err());

        let different_index =
            IntegralLineConfiguration::new(2, integers(&[&[1, 0], &[1, 2], &[1, 1]])).unwrap();
        assert!(source.equivalence_to(&different_index).unwrap().is_none());
    }
}
