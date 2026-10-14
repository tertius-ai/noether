// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Structural invariants of finite presentations.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::commalg::{
    FreeAbelianGrading, GroebnerError, HilbertSeries, HilbertSeriesError, Ideal, PolynomialRing,
};
use crate::linalg::{LinalgError, Matrix};
use crate::{CommutativeRing, Field, Integer, NoetherError, Ring, SemanticType, TypeDescription};

use super::{
    FreeModule, FreeResolution, GradedBettiTable, GradedBettiTableError, ModuleError,
    ModuleGroebnerError, ModuleHomomorphism, ModuleMonomialOrder, PresentedModule,
    PresentedModuleHomomorphism, ResolutionCompleteness, ResolutionError, ResolutionMinimality,
    Submodule,
};

const MAX_FITTING_MINORS: usize = 100_000;

static GRADED_MODULE_INVARIANTS_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/homalg/GradedModuleInvariants@1",
    rust_path: "noether::homalg::GradedModuleInvariants",
    python_path: Some("noether.homalg.GradedModuleInvariants"),
};

/// Failure to construct or verify a structural module invariant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleInvariantError {
    /// A based-module construction failed.
    Module(ModuleError),
    /// A module kernel or membership calculation failed.
    Groebner(ModuleGroebnerError),
    /// A canonical polynomial ideal could not be constructed.
    Ideal(GroebnerError),
    /// Exact determinant construction failed.
    LinearAlgebra(LinalgError),
    /// A free resolution could not be constructed or verified.
    Resolution(ResolutionError),
    /// Exact Betti data could not be derived from the resolution.
    Betti(GradedBettiTableError),
    /// Exact Hilbert-series data could not be specialized.
    Hilbert(HilbertSeriesError),
    /// The requested family of minors exceeds the fixed native bound.
    MinorCountLimit {
        /// Maximum number of minors admitted in one result.
        limit: usize,
        /// Number of minors requested, when it fits a machine word.
        requested: Option<usize>,
    },
    /// Independently replayed evidence contradicted the claimed invariant.
    InvalidEvidence(&'static str),
}

impl fmt::Display for ModuleInvariantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Ideal(error) => error.fmt(formatter),
            Self::LinearAlgebra(error) => error.fmt(formatter),
            Self::Resolution(error) => error.fmt(formatter),
            Self::Betti(error) => error.fmt(formatter),
            Self::Hilbert(error) => error.fmt(formatter),
            Self::MinorCountLimit { limit, requested } => match requested {
                Some(requested) => write!(
                    formatter,
                    "the requested {requested} minors exceed the limit {limit}"
                ),
                None => write!(
                    formatter,
                    "the requested minor family exceeds the limit {limit}"
                ),
            },
            Self::InvalidEvidence(reason) => {
                write!(formatter, "invalid module-invariant evidence: {reason}")
            }
        }
    }
}

impl std::error::Error for ModuleInvariantError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Module(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::Ideal(error) => Some(error),
            Self::LinearAlgebra(error) => Some(error),
            Self::Resolution(error) => Some(error),
            Self::Betti(error) => Some(error),
            Self::Hilbert(error) => Some(error),
            Self::MinorCountLimit { .. } | Self::InvalidEvidence(_) => None,
        }
    }
}

impl From<ModuleError> for ModuleInvariantError {
    fn from(value: ModuleError) -> Self {
        Self::Module(value)
    }
}

impl From<ModuleGroebnerError> for ModuleInvariantError {
    fn from(value: ModuleGroebnerError) -> Self {
        Self::Groebner(value)
    }
}

impl From<GroebnerError> for ModuleInvariantError {
    fn from(value: GroebnerError) -> Self {
        Self::Ideal(value)
    }
}

impl From<LinalgError> for ModuleInvariantError {
    fn from(value: LinalgError) -> Self {
        Self::LinearAlgebra(value)
    }
}

impl From<ResolutionError> for ModuleInvariantError {
    fn from(value: ResolutionError) -> Self {
        Self::Resolution(value)
    }
}

impl From<GradedBettiTableError> for ModuleInvariantError {
    fn from(value: GradedBettiTableError) -> Self {
        Self::Betti(value)
    }
}

impl From<HilbertSeriesError> for ModuleInvariantError {
    fn from(value: HilbertSeriesError) -> Self {
        Self::Hilbert(value)
    }
}

/// Exact structural invariants of a standard-graded polynomial module.
///
/// The value retains a complete minimal free resolution, its Betti table, and
/// the exact Hilbert series. Projective dimension and regularity are read from
/// the Betti table; depth is certified by Auslander--Buchsbaum over the
/// regular polynomial ring; dimension is certified by Hilbert--Serre. The
/// zero module has `None` for invariants whose conventional value would be an
/// infinity rather than an ordinary integer.
#[derive(Clone, PartialEq, Eq)]
pub struct GradedModuleInvariants<C: Field> {
    module: PresentedModule<PolynomialRing<C>>,
    order: ModuleMonomialOrder,
    resolution: FreeResolution<PolynomialRing<C>>,
    betti_table: GradedBettiTable,
    hilbert_series: HilbertSeries,
    projective_dimension: Option<usize>,
    regularity: Option<Integer>,
    depth: Option<usize>,
    dimension: Option<usize>,
    cohen_macaulay: Option<bool>,
    perfect: Option<bool>,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "C: Field + Deserialize<'de>, C::Element: Deserialize<'de>")
)]
struct GradedModuleInvariantsWire<C: Field> {
    #[serde(rename = "type")]
    type_id: String,
    module: PresentedModule<PolynomialRing<C>>,
    order: ModuleMonomialOrder,
    resolution: FreeResolution<PolynomialRing<C>>,
    betti_table: GradedBettiTable,
    hilbert_series: HilbertSeries,
    projective_dimension: Option<usize>,
    regularity: Option<Integer>,
    depth: Option<usize>,
    dimension: Option<usize>,
    cohen_macaulay: Option<bool>,
    perfect: Option<bool>,
}

impl<C: Field> GradedModuleInvariants<C> {
    /// Standard-graded module whose invariants were certified.
    pub fn module(&self) -> &PresentedModule<PolynomialRing<C>> {
        &self.module
    }

    /// Module-monomial order used for the exact construction.
    pub fn order(&self) -> &ModuleMonomialOrder {
        &self.order
    }

    /// Complete minimal resolution proving the homological invariants.
    pub fn resolution(&self) -> &FreeResolution<PolynomialRing<C>> {
        &self.resolution
    }

    /// Exact multigraded Betti table of the retained resolution.
    pub fn betti_table(&self) -> &GradedBettiTable {
        &self.betti_table
    }

    /// Exact Hilbert series proving dimension.
    pub fn hilbert_series(&self) -> &HilbertSeries {
        &self.hilbert_series
    }

    /// Projective dimension, or `None` for the zero module.
    pub const fn projective_dimension(&self) -> Option<usize> {
        self.projective_dimension
    }

    /// Castelnuovo--Mumford regularity, or `None` for the zero module.
    pub fn regularity(&self) -> Option<&Integer> {
        self.regularity.as_ref()
    }

    /// Depth certified by Auslander--Buchsbaum, or `None` for zero.
    pub const fn depth(&self) -> Option<usize> {
        self.depth
    }

    /// Krull dimension certified by Hilbert--Serre, or `None` for zero.
    pub const fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Cohen--Macaulay predicate, or `None` for the zero module.
    pub const fn is_cohen_macaulay(&self) -> Option<bool> {
        self.cohen_macaulay
    }

    /// Perfect-module predicate, or `None` for the zero module.
    pub const fn is_perfect(&self) -> Option<bool> {
        self.perfect
    }

    /// Stable theorem label for orchestration and display.
    pub const fn theorem(&self) -> &'static str {
        "minimal Betti table + Hilbert-Serre + Auslander-Buchsbaum"
    }

    /// Independently replay every retained resolution, Betti, Hilbert, and
    /// theorem-derived claim.
    pub fn verify(&self) -> Result<(), ModuleInvariantError> {
        if self.resolution.module() != &self.module
            || self.resolution.completeness() != &ResolutionCompleteness::Complete
            || self.resolution.minimality() != &ResolutionMinimality::Minimal
        {
            return Err(ModuleInvariantError::InvalidEvidence(
                "graded invariants require a complete minimal resolution of the stated module",
            ));
        }
        require_standard_grading(&self.module)?;
        self.resolution.verify()?;
        let betti_table = self.resolution.graded_betti_table()?;
        if betti_table != self.betti_table {
            return Err(ModuleInvariantError::InvalidEvidence(
                "the retained Betti table does not come from the retained resolution",
            ));
        }
        let hilbert_series = module_hilbert_series(&self.module, self.order.clone())?;
        if hilbert_series != self.hilbert_series {
            return Err(ModuleInvariantError::InvalidEvidence(
                "the retained Hilbert series does not come from the stated module",
            ));
        }
        let values = structural_values(
            self.module.generator_module().ring().variables().len(),
            &self.betti_table,
            &self.hilbert_series,
        )?;
        if values
            != (
                self.projective_dimension,
                self.regularity.clone(),
                self.depth,
                self.dimension,
                self.cohen_macaulay,
                self.perfect,
            )
        {
            return Err(ModuleInvariantError::InvalidEvidence(
                "the stored structural values do not follow from the retained evidence",
            ));
        }
        Ok(())
    }

    /// Serialize the module, exact evidence, and theorem-derived values.
    pub fn to_json(&self) -> Result<String, NoetherError>
    where
        C: Serialize,
        C::Element: Serialize,
    {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and independently revalidate every retained claim.
    pub fn from_json(value: &str) -> Result<Self, NoetherError>
    where
        C: for<'de> Deserialize<'de>,
        C::Element: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(value).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R: CommutativeRing> ModuleHomomorphism<R> {
    fn determinantal_generators(
        &self,
        size: usize,
    ) -> Result<Vec<R::Element>, ModuleInvariantError> {
        if size == 0 {
            return Ok(vec![self.target().ring().one()]);
        }
        if size > self.target().rank() || size > self.source().rank() {
            return Ok(Vec::new());
        }
        let row_count = bounded_binomial(self.target().rank(), size)?;
        let column_count = bounded_binomial(self.source().rank(), size)?;
        let requested = row_count.checked_mul(column_count);
        if requested.is_none_or(|count| count > MAX_FITTING_MINORS) {
            return Err(ModuleInvariantError::MinorCountLimit {
                limit: MAX_FITTING_MINORS,
                requested,
            });
        }
        let entries = self.matrix().rows().collect::<Vec<_>>();
        let rows = combinations(self.target().rank(), size);
        let columns = combinations(self.source().rank(), size);
        let mut minors = Vec::with_capacity(requested.unwrap_or(0));
        for row_set in &rows {
            for column_set in &columns {
                let values = row_set.iter().map(|row| {
                    column_set
                        .iter()
                        .map(|column| entries[*row][*column].clone())
                        .collect::<Vec<_>>()
                });
                minors.push(Matrix::from_rows(self.target().ring().clone(), values)?.det()?);
            }
        }
        Ok(minors)
    }
}

impl<C: Field> PresentedModule<PolynomialRing<C>> {
    /// Annihilator ideal of this finitely presented module.
    ///
    /// The implementation intersects the exact kernels of the maps
    /// `R -> M` sending `1` to each retained presentation generator, then
    /// independently checks that every returned ideal generator kills every
    /// module generator modulo the presentation relations.
    pub fn annihilator(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<Ideal<PolynomialRing<C>>, ModuleInvariantError> {
        let ring = self.generator_module().ring();
        let mut result = ring.ideal([ring.one()])?;
        let relation_basis = Submodule::from_generator_map(
            self.generator_module().clone(),
            self.presentation().clone(),
        )?
        .groebner_basis(order.clone())?;

        for position in 0..self.generator_count() {
            let free = PresentedModule::free(FreeModule::new(ring.clone(), 1))?;
            let generator_lift = ModuleHomomorphism::from_coordinate_columns(
                free.generator_module().clone(),
                self.generator_module().clone(),
                vec![self
                    .generator_module()
                    .basis_element(position)?
                    .coordinates()
                    .values()
                    .collect()],
            )?;
            let relation_lift = ModuleHomomorphism::zero(
                free.relation_module().clone(),
                self.relation_module().clone(),
            )?;
            let map = PresentedModuleHomomorphism::from_lifts(
                free,
                self.clone(),
                generator_lift,
                relation_lift,
            )?;
            let kernel = map.kernel(order.clone())?;
            let generators = kernel
                .generator_lift()
                .matrix()
                .rows()
                .flatten()
                .collect::<Vec<_>>();
            result = result.intersection(&ring.ideal(generators)?)?;
        }

        for annihilator in result.generators() {
            for position in 0..self.generator_count() {
                let value = self
                    .generator_module()
                    .basis_element(position)?
                    .scalar_mul(annihilator)?;
                if !relation_basis.contains(&value)? {
                    return Err(ModuleInvariantError::InvalidEvidence(
                        "an annihilator generator does not kill the presented module",
                    ));
                }
            }
        }
        Ok(result)
    }

    /// The `index`-th Fitting ideal of this finite presentation.
    ///
    /// For a presentation `R^m -> R^n -> M`, this is generated by the
    /// `(n-index)`-minors. The standard conventions are retained: sizes above
    /// the matrix shape give the zero ideal, while `index >= n` gives the unit
    /// ideal. The result is canonicalized by the owning polynomial ring.
    pub fn fitting_ideal(
        &self,
        index: usize,
    ) -> Result<Ideal<PolynomialRing<C>>, ModuleInvariantError> {
        let size = self.generator_count().saturating_sub(index);
        Ok(self
            .generator_module()
            .ring()
            .ideal(self.presentation().determinantal_generators(size)?)?)
    }

    /// Compute the standard-graded projective dimension, regularity, depth,
    /// dimension, Cohen--Macaulay predicate, and perfect predicate with one
    /// shared exact evidence object.
    pub fn graded_invariants(
        &self,
        order: ModuleMonomialOrder,
    ) -> Result<GradedModuleInvariants<C>, ModuleInvariantError> {
        require_standard_grading(self)?;
        let resolution = self.minimal_free_resolution(order.clone())?;
        let betti_table = resolution.graded_betti_table()?;
        let hilbert_series = module_hilbert_series(self, order.clone())?;
        let (projective_dimension, regularity, depth, dimension, cohen_macaulay, perfect) =
            structural_values(
                self.generator_module().ring().variables().len(),
                &betti_table,
                &hilbert_series,
            )?;
        let result = GradedModuleInvariants {
            module: self.clone(),
            order,
            resolution,
            betti_table,
            hilbert_series,
            projective_dimension,
            regularity,
            depth,
            dimension,
            cohen_macaulay,
            perfect,
        };
        result.verify()?;
        Ok(result)
    }
}

type StructuralValues = (
    Option<usize>,
    Option<Integer>,
    Option<usize>,
    Option<usize>,
    Option<bool>,
    Option<bool>,
);

fn require_standard_grading<C: Field>(
    module: &PresentedModule<PolynomialRing<C>>,
) -> Result<(), ModuleInvariantError> {
    let ring = module.generator_module().ring();
    if ring.grading() != &FreeAbelianGrading::standard(ring.variables().len()) {
        return Err(ModuleInvariantError::InvalidEvidence(
            "graded structural invariants currently require the standard grading",
        ));
    }
    Ok(())
}

fn module_hilbert_series<C: Field>(
    module: &PresentedModule<PolynomialRing<C>>,
    order: ModuleMonomialOrder,
) -> Result<HilbertSeries, ModuleInvariantError> {
    Ok(Submodule::from_generator_map(
        module.generator_module().clone(),
        module.presentation().clone(),
    )?
    .groebner_basis(order)?
    .hilbert_series()?)
}

fn structural_values(
    variables: usize,
    betti_table: &GradedBettiTable,
    hilbert_series: &HilbertSeries,
) -> Result<StructuralValues, ModuleInvariantError> {
    let dimension = hilbert_dimension(hilbert_series, variables)?;
    let Some(dimension) = dimension else {
        return Ok((None, None, None, None, None, None));
    };
    let projective_dimension =
        betti_table
            .projective_dimension()
            .ok_or(ModuleInvariantError::InvalidEvidence(
                "a complete minimal Betti table did not determine projective dimension",
            ))?;
    let depth = variables.checked_sub(projective_dimension).ok_or(
        ModuleInvariantError::InvalidEvidence(
            "projective dimension exceeds the regular polynomial-ring dimension",
        ),
    )?;
    let codimension =
        variables
            .checked_sub(dimension)
            .ok_or(ModuleInvariantError::InvalidEvidence(
                "module dimension exceeds the polynomial-ring dimension",
            ))?;
    let regularity = betti_regularity(betti_table)?.ok_or(
        ModuleInvariantError::InvalidEvidence("a nonzero module has an empty Betti table"),
    )?;
    Ok((
        Some(projective_dimension),
        Some(regularity),
        Some(depth),
        Some(dimension),
        Some(depth == dimension),
        Some(projective_dimension == codimension),
    ))
}

fn betti_regularity(
    betti_table: &GradedBettiTable,
) -> Result<Option<Integer>, ModuleInvariantError> {
    let mut regularity: Option<Integer> = None;
    for (homological_degree, internal_degree, _) in betti_table.entries() {
        let degree =
            internal_degree
                .coordinates()
                .first()
                .ok_or(ModuleInvariantError::InvalidEvidence(
                    "standard-graded Betti degree has no coordinate",
                ))?;
        let homological_degree = u64::try_from(*homological_degree).map_err(|_| {
            ModuleInvariantError::InvalidEvidence("homological degree does not fit an integer")
        })?;
        let candidate = degree - &Integer::from(homological_degree);
        if regularity.as_ref().is_none_or(|known| &candidate > known) {
            regularity = Some(candidate);
        }
    }
    Ok(regularity)
}

fn hilbert_dimension(
    hilbert_series: &HilbertSeries,
    variables: usize,
) -> Result<Option<usize>, ModuleInvariantError> {
    let numerator = hilbert_series.numerator();
    if numerator.is_empty() {
        return Ok(None);
    }
    for vanishing_order in 0..=variables {
        let derivative_at_one =
            numerator
                .iter()
                .try_fold(Integer::zero(), |sum, (degree, coefficient)| {
                    let exponent = degree.coordinates().first().ok_or(
                        ModuleInvariantError::InvalidEvidence(
                            "standard-graded Hilbert degree has no coordinate",
                        ),
                    )?;
                    let mut falling_factorial = Integer::one();
                    for offset in 0..vanishing_order {
                        let offset = u64::try_from(offset).map_err(|_| {
                            ModuleInvariantError::InvalidEvidence(
                                "Hilbert derivative order does not fit an integer",
                            )
                        })?;
                        falling_factorial =
                            &falling_factorial * &(exponent - &Integer::from(offset));
                    }
                    Ok::<_, ModuleInvariantError>(&sum + &(coefficient * &falling_factorial))
                })?;
        if !derivative_at_one.is_zero() {
            return variables.checked_sub(vanishing_order).map(Some).ok_or(
                ModuleInvariantError::InvalidEvidence(
                    "Hilbert dimension underflowed the polynomial-ring dimension",
                ),
            );
        }
    }
    Err(ModuleInvariantError::InvalidEvidence(
        "Hilbert numerator has impossible vanishing order at one",
    ))
}

impl<C> Serialize for GradedModuleInvariants<C>
where
    C: Field + Serialize,
    C::Element: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("GradedModuleInvariants", 12)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("module", &self.module)?;
        state.serialize_field("order", &self.order)?;
        state.serialize_field("resolution", &self.resolution)?;
        state.serialize_field("betti_table", &self.betti_table)?;
        state.serialize_field("hilbert_series", &self.hilbert_series)?;
        state.serialize_field("projective_dimension", &self.projective_dimension)?;
        state.serialize_field("regularity", &self.regularity)?;
        state.serialize_field("depth", &self.depth)?;
        state.serialize_field("dimension", &self.dimension)?;
        state.serialize_field("cohen_macaulay", &self.cohen_macaulay)?;
        state.serialize_field("perfect", &self.perfect)?;
        state.end()
    }
}

impl<'de, C> Deserialize<'de> for GradedModuleInvariants<C>
where
    C: Field + Deserialize<'de>,
    C::Element: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = GradedModuleInvariantsWire::<C>::deserialize(deserializer)?;
        if wire.type_id != "noether/homalg/GradedModuleInvariants@1" {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                "noether/homalg/GradedModuleInvariants@1", wire.type_id
            )));
        }
        let result = Self {
            module: wire.module,
            order: wire.order,
            resolution: wire.resolution,
            betti_table: wire.betti_table,
            hilbert_series: wire.hilbert_series,
            projective_dimension: wire.projective_dimension,
            regularity: wire.regularity,
            depth: wire.depth,
            dimension: wire.dimension,
            cohen_macaulay: wire.cohen_macaulay,
            perfect: wire.perfect,
        };
        result.verify().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

impl<C: Field> fmt::Debug for GradedModuleInvariants<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GradedModuleInvariants")
            .field("projective_dimension", &self.projective_dimension)
            .field("regularity", &self.regularity)
            .field("depth", &self.depth)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl<C: Field> fmt::Display for GradedModuleInvariants<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GradedModuleInvariants(pd={:?}, reg={:?}, depth={:?}, dim={:?})",
            self.projective_dimension, self.regularity, self.depth, self.dimension
        )
    }
}

impl<C> SemanticType for GradedModuleInvariants<C>
where
    C: Field + Serialize,
    C::Element: Serialize,
{
    const TYPE_ID: &'static str = "noether/homalg/GradedModuleInvariants@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &GRADED_MODULE_INVARIANTS_DESCRIPTION
    }
}

fn bounded_binomial(n: usize, k: usize) -> Result<usize, ModuleInvariantError> {
    let k = k.min(n.saturating_sub(k));
    let mut result = 1_usize;
    for index in 0..k {
        result = result
            .checked_mul(n - index)
            .and_then(|value| value.checked_div(index + 1))
            .ok_or(ModuleInvariantError::MinorCountLimit {
                limit: MAX_FITTING_MINORS,
                requested: None,
            })?;
        if result > MAX_FITTING_MINORS {
            return Err(ModuleInvariantError::MinorCountLimit {
                limit: MAX_FITTING_MINORS,
                requested: Some(result),
            });
        }
    }
    Ok(result)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    let mut current = (0..k).collect::<Vec<_>>();
    let mut result = Vec::new();
    loop {
        result.push(current.clone());
        let Some(position) = (0..k)
            .rev()
            .find(|position| current[*position] < n - k + *position)
        else {
            break;
        };
        current[position] += 1;
        for next in position + 1..k {
            current[next] = current[next - 1] + 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use crate::commalg::{Degree, MonomialOrder, PolynomialRing};
    use crate::linalg::Matrix;
    use crate::Q;

    use super::*;

    #[test]
    fn fitting_ideals_and_annihilators_follow_the_defining_presentations() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let [x, y]: [_; 2] = ring.generators().unwrap().try_into().unwrap();
        let free = FreeModule::new(ring.clone(), 2);
        let module = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                free.clone(),
                free,
                Matrix::from_rows(
                    ring.clone(),
                    [[x.clone(), ring.zero()], [ring.zero(), y.clone()]],
                )
                .unwrap(),
            )
            .unwrap(),
        );
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        assert_eq!(
            module.fitting_ideal(0).unwrap(),
            ring.ideal([(&x * &y).unwrap()]).unwrap()
        );
        assert_eq!(
            module.fitting_ideal(1).unwrap(),
            ring.ideal([x.clone(), y.clone()]).unwrap()
        );
        assert_eq!(
            module.fitting_ideal(2).unwrap(),
            ring.ideal([ring.one()]).unwrap()
        );
        assert_eq!(
            module.annihilator(order).unwrap(),
            ring.ideal([(&x * &y).unwrap()]).unwrap()
        );
    }

    #[test]
    fn fitting_conventions_cover_free_and_zero_modules() {
        let ring =
            PolynomialRing::new(Q, ["x"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let free = PresentedModule::free(FreeModule::new(ring.clone(), 2)).unwrap();
        let zero = PresentedModule::free(FreeModule::new(ring.clone(), 0)).unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        assert_eq!(free.fitting_ideal(0).unwrap(), ring.ideal([]).unwrap());
        assert_eq!(free.fitting_ideal(1).unwrap(), ring.ideal([]).unwrap());
        assert_eq!(
            free.fitting_ideal(2).unwrap(),
            ring.ideal([ring.one()]).unwrap()
        );
        assert_eq!(
            free.annihilator(order.clone()).unwrap(),
            ring.ideal([]).unwrap()
        );
        assert_eq!(
            zero.annihilator(order).unwrap(),
            ring.ideal([ring.one()]).unwrap()
        );
    }

    #[test]
    fn graded_invariants_retain_the_resolution_and_theorem_evidence() {
        let ring = PolynomialRing::new(
            Q,
            ["x", "y", "z"],
            MonomialOrder::GradedReverseLexicographic,
        )
        .unwrap();
        let [x, y, _]: [_; 3] = ring.generators().unwrap().try_into().unwrap();
        let target = FreeModule::new(ring.clone(), 1)
            .with_degree_shifts(1, vec![Degree::zero(1)])
            .unwrap();
        let source = FreeModule::new(ring.clone(), 2)
            .with_degree_shifts(
                1,
                vec![
                    Degree::new(vec![(-1).into()]),
                    Degree::new(vec![(-1).into()]),
                ],
            )
            .unwrap();
        let module = PresentedModule::cokernel(
            ModuleHomomorphism::from_matrix(
                source,
                target,
                Matrix::from_rows(ring.clone(), [[x, y]]).unwrap(),
            )
            .unwrap(),
        );
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());

        let invariants = module.graded_invariants(order).unwrap();
        invariants.verify().unwrap();
        assert_eq!(invariants.projective_dimension(), Some(2));
        assert_eq!(invariants.regularity(), Some(&Integer::zero()));
        assert_eq!(invariants.depth(), Some(1));
        assert_eq!(invariants.dimension(), Some(1));
        assert_eq!(invariants.is_cohen_macaulay(), Some(true));
        assert_eq!(invariants.is_perfect(), Some(true));
        assert_eq!(invariants.resolution().length(), 2);
        assert_eq!(invariants.betti_table().total(1), 2);
        assert_eq!(
            GradedModuleInvariants::<crate::RationalField>::from_json(
                &invariants.to_json().unwrap(),
            )
            .unwrap(),
            invariants
        );
    }

    #[test]
    fn graded_invariants_handle_laurent_shifts_and_the_zero_module() {
        let ring =
            PolynomialRing::new(Q, ["x", "y"], MonomialOrder::GradedReverseLexicographic).unwrap();
        let order = ModuleMonomialOrder::TermOverPosition(ring.term_order());
        let shifted = PresentedModule::free(
            FreeModule::new(ring.clone(), 1)
                .with_degree_shifts(1, vec![Degree::new(vec![3.into()])])
                .unwrap(),
        )
        .unwrap();
        let shifted_invariants = shifted.graded_invariants(order.clone()).unwrap();
        assert_eq!(shifted_invariants.projective_dimension(), Some(0));
        assert_eq!(shifted_invariants.regularity(), Some(&Integer::from(-3)));
        assert_eq!(shifted_invariants.depth(), Some(2));
        assert_eq!(shifted_invariants.dimension(), Some(2));

        let zero = PresentedModule::free(
            FreeModule::new(ring, 0)
                .with_degree_shifts(1, Vec::new())
                .unwrap(),
        )
        .unwrap();
        let zero_invariants = zero.graded_invariants(order).unwrap();
        assert_eq!(zero_invariants.projective_dimension(), None);
        assert_eq!(zero_invariants.regularity(), None);
        assert_eq!(zero_invariants.depth(), None);
        assert_eq!(zero_invariants.dimension(), None);
        assert_eq!(zero_invariants.is_cohen_macaulay(), None);
        assert_eq!(zero_invariants.is_perfect(), None);
    }
}
