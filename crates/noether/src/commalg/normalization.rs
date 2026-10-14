// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite polynomial-subring maps witnessing Noether normalization.

use core::fmt;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::numthy::{FiniteField, FiniteFieldElement};
use crate::{
    CoefficientMap, ContextMismatchError, Field, FiniteContext, IdentityCoefficientMap, Integer,
    NoetherError, Rational, RationalField, Ring, SemanticType, TypeDescription,
};

use super::{
    GroebnerError, Ideal, MonomialOrder, Polynomial, PolynomialContextError, PolynomialError,
    PolynomialHomomorphism, PolynomialHomomorphismError, PolynomialRing, QuotientElement,
    QuotientRing,
};

static NOETHER_NORMALIZATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/NoetherNormalization@1",
    rust_path: "noether::commalg::NoetherNormalization",
    python_path: Some("noether.commalg.NoetherNormalization"),
};

const MAX_SEARCH_CANDIDATES: usize = 512;
const MAX_SEARCH_SCALARS: usize = 5;
const MAX_MATERIALIZED_MODULE_GENERATORS: usize = 1_000_000;

/// Failure to construct, apply, or verify a Noether normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoetherNormalizationError {
    /// The zero algebra has no injective polynomial-subring normalization.
    ZeroAlgebra,
    /// No finite linear normalization was found in the deterministic search.
    SearchExhausted {
        /// Number of invertible coordinate changes tested.
        candidates: usize,
    },
    /// Materializing the finite module generating family exceeded its bound.
    ResourceLimit {
        /// Name of the bounded resource.
        resource: &'static str,
        /// Exact requested cardinality.
        requested: Integer,
        /// Maximum materialized cardinality.
        limit: usize,
    },
    /// Polynomial-context construction failed.
    PolynomialContext(String),
    /// Polynomial construction or arithmetic failed.
    Polynomial(PolynomialError),
    /// Gröbner-basis or dimension computation failed.
    Groebner(GroebnerError),
    /// A checked polynomial homomorphism failed.
    Homomorphism(PolynomialHomomorphismError),
    /// Exact context identity was violated.
    ContextMismatch(ContextMismatchError),
    /// Retained normalization evidence failed exact replay.
    InvalidEvidence(String),
}

impl fmt::Display for NoetherNormalizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAlgebra => {
                formatter.write_str("the zero algebra has no Noether normalization")
            }
            Self::SearchExhausted { candidates } => write!(
                formatter,
                "no finite linear Noether normalization was found after {candidates} deterministic coordinate changes"
            ),
            Self::ResourceLimit {
                resource,
                requested,
                limit,
            } => write!(
                formatter,
                "{resource} requires {requested} values, exceeding the materialization limit {limit}"
            ),
            Self::PolynomialContext(detail) => {
                write!(formatter, "normalization polynomial context failed: {detail}")
            }
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Homomorphism(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::InvalidEvidence(detail) => {
                write!(formatter, "invalid Noether-normalization evidence: {detail}")
            }
        }
    }
}

impl std::error::Error for NoetherNormalizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Polynomial(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::Homomorphism(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::ZeroAlgebra
            | Self::SearchExhausted { .. }
            | Self::ResourceLimit { .. }
            | Self::PolynomialContext(_)
            | Self::InvalidEvidence(_) => None,
        }
    }
}

impl From<PolynomialError> for NoetherNormalizationError {
    fn from(value: PolynomialError) -> Self {
        Self::Polynomial(value)
    }
}

impl From<GroebnerError> for NoetherNormalizationError {
    fn from(value: GroebnerError) -> Self {
        Self::Groebner(value)
    }
}

impl From<PolynomialHomomorphismError> for NoetherNormalizationError {
    fn from(value: PolynomialHomomorphismError) -> Self {
        Self::Homomorphism(value)
    }
}

impl From<ContextMismatchError> for NoetherNormalizationError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<PolynomialContextError> for NoetherNormalizationError {
    fn from(value: PolynomialContextError) -> Self {
        Self::PolynomialContext(value.to_string())
    }
}

/// A verified finite injective polynomial-subring map into an affine algebra.
///
/// Let `A = k[x_1,...,x_n]/I` and `d = dim(A)`. The retained source is a
/// polynomial ring `k[t_1,...,t_d]`, and the parameter images are independent
/// linear forms in `A`. An invertible shear writes the ambient coordinates as
/// parameter and dependent coordinates. In a block order with the dependent
/// block first, the retained reduced Gröbner basis has a pure leading power of
/// every dependent coordinate. The bounded dependent monomials therefore span
/// `A` over the parameter ring. Since a finite image of a `d`-dimensional
/// polynomial ring also has dimension `d`, the kernel is zero.
#[derive(Clone, PartialEq, Eq)]
pub struct NoetherNormalization<R: Field> {
    source: Ideal<PolynomialRing<R>>,
    dimension: usize,
    parameter_indices: Vec<usize>,
    dependent_indices: Vec<usize>,
    shear_coefficients: Vec<Vec<R::Element>>,
    transformed_ideal: Ideal<PolynomialRing<R>>,
    pure_power_bounds: Vec<Integer>,
    parameter_ring: PolynomialRing<R>,
    algebra: QuotientRing<PolynomialRing<R>>,
    coordinate_images: Vec<Polynomial<R>>,
    parameter_images: Vec<QuotientElement<PolynomialRing<R>>>,
}

impl<R: Field> NoetherNormalization<R> {
    /// Defining ideal of the affine algebra being normalized.
    pub fn source(&self) -> &Ideal<PolynomialRing<R>> {
        &self.source
    }

    /// Krull dimension of the target affine algebra.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Polynomial parameter ring `k[t_1,...,t_d]`.
    pub fn parameter_ring(&self) -> &PolynomialRing<R> {
        &self.parameter_ring
    }

    /// Target affine algebra `k[x_1,...,x_n]/I`.
    pub fn algebra(&self) -> &QuotientRing<PolynomialRing<R>> {
        &self.algebra
    }

    /// Ambient pivot coordinates used for the parameter shear chart.
    pub fn parameter_indices(&self) -> &[usize] {
        &self.parameter_indices
    }

    /// Complementary coordinates proved finite over the parameters.
    pub fn dependent_indices(&self) -> &[usize] {
        &self.dependent_indices
    }

    /// Shear coefficients, with rows aligned to parameters and columns to dependents.
    pub fn shear_coefficients(&self) -> &[Vec<R::Element>] {
        &self.shear_coefficients
    }

    /// Images of all sheared coordinates as ambient polynomials before quotienting.
    pub fn coordinate_images(&self) -> &[Polynomial<R>] {
        &self.coordinate_images
    }

    /// Images of the polynomial parameters in the target affine algebra.
    pub fn parameter_images(&self) -> &[QuotientElement<PolynomialRing<R>>] {
        &self.parameter_images
    }

    /// Defining ideal after the retained invertible coordinate change and block order.
    pub fn transformed_ideal(&self) -> &Ideal<PolynomialRing<R>> {
        &self.transformed_ideal
    }

    /// Strict exponent bounds for dependent monomials spanning the finite module.
    pub fn pure_power_bounds(&self) -> &[Integer] {
        &self.pure_power_bounds
    }

    /// Exact cardinality of the retained rectangular module generating family.
    pub fn module_generator_count(&self) -> Integer {
        self.pure_power_bounds
            .iter()
            .fold(Integer::one(), |product, bound| product * bound)
    }

    /// Ambient exponent vectors for a finite spanning family over the parameter ring.
    pub fn module_generator_exponents(
        &self,
    ) -> Result<Vec<Vec<Integer>>, NoetherNormalizationError> {
        let requested = self.module_generator_count();
        let count = integer_to_usize(&requested).ok_or_else(|| {
            NoetherNormalizationError::ResourceLimit {
                resource: "Noether-normalization module generators",
                requested: requested.clone(),
                limit: MAX_MATERIALIZED_MODULE_GENERATORS,
            }
        })?;
        if count > MAX_MATERIALIZED_MODULE_GENERATORS {
            return Err(NoetherNormalizationError::ResourceLimit {
                resource: "Noether-normalization module generators",
                requested,
                limit: MAX_MATERIALIZED_MODULE_GENERATORS,
            });
        }
        let bounds = self
            .pure_power_bounds
            .iter()
            .map(|bound| {
                integer_to_usize(bound).ok_or_else(|| {
                    NoetherNormalizationError::InvalidEvidence(
                        "a pure-power bound does not fit machine-sized enumeration".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut output = Vec::with_capacity(count);
        let mut dependent_exponents = vec![0_usize; bounds.len()];
        enumerate_rectangular_exponents(
            0,
            &bounds,
            &mut dependent_exponents,
            &self.dependent_indices,
            self.source.ring().variables().len(),
            &mut output,
        );
        Ok(output)
    }

    /// A finite family of target elements spanning the algebra over the parameters.
    pub fn module_generators(
        &self,
    ) -> Result<Vec<QuotientElement<PolynomialRing<R>>>, NoetherNormalizationError> {
        let coordinate_classes = self
            .coordinate_images
            .iter()
            .map(|coordinate| self.algebra.element(coordinate))
            .collect::<Result<Vec<_>, _>>()?;
        self.module_generator_exponents()?
            .into_iter()
            .map(|exponents| {
                let mut value = self.algebra.one();
                for &coordinate in &self.dependent_indices {
                    let exponent = &exponents[coordinate];
                    let exponent = integer_to_u64(exponent).ok_or_else(|| {
                        NoetherNormalizationError::InvalidEvidence(
                            "a module-generator exponent does not fit exact powering".to_owned(),
                        )
                    })?;
                    value = self
                        .algebra
                        .mul(&value, &coordinate_classes[coordinate].pow(exponent))?;
                }
                Ok(value)
            })
            .collect()
    }

    /// Apply the retained injective finite map to one parameter polynomial.
    pub fn apply(
        &self,
        value: &Polynomial<R>,
    ) -> Result<QuotientElement<PolynomialRing<R>>, NoetherNormalizationError>
    where
        R: fmt::Display,
    {
        let coefficient_map = QuotientCoefficientEmbedding {
            target: self.algebra.clone(),
        };
        let map = PolynomialHomomorphism::new(
            self.parameter_ring.clone(),
            self.algebra.clone(),
            coefficient_map,
            self.parameter_images.clone(),
        )?;
        map.apply(value).map_err(Into::into)
    }

    /// The exact commutative-algebra theorem certified by this value.
    pub const fn theorem(&self) -> &'static str {
        "pure dependent leading powers make the target finite over the parameter ring; equal Krull dimension then forces the parameter map to be injective"
    }
}

impl<R: Field> fmt::Debug for NoetherNormalization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NoetherNormalization")
            .field("dimension", &self.dimension)
            .field("parameters", &self.parameter_indices)
            .field("dependents", &self.dependent_indices)
            .field("module_generators", &self.module_generator_count())
            .finish_non_exhaustive()
    }
}

impl<R: Field> fmt::Display for NoetherNormalization<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "NoetherNormalization(dimension={}, module_generators={})",
            self.dimension,
            self.module_generator_count()
        )
    }
}

#[derive(Clone)]
struct QuotientCoefficientEmbedding<R: Field> {
    target: QuotientRing<PolynomialRing<R>>,
}

impl<R> CoefficientMap for QuotientCoefficientEmbedding<R>
where
    R: Field + fmt::Display,
{
    type Source = R;
    type Target = QuotientRing<PolynomialRing<R>>;
    type Error = NoetherNormalizationError;

    fn source(&self) -> &Self::Source {
        self.target.base_ring().coefficient_ring()
    }

    fn target(&self) -> &Self::Target {
        &self.target
    }

    fn apply(&self, value: &R::Element) -> Result<<Self::Target as Ring>::Element, Self::Error> {
        let constant = self.target.base_ring().constant(value.clone())?;
        self.target.element(&constant).map_err(Into::into)
    }
}

fn construct_normalization<R>(
    source: &Ideal<PolynomialRing<R>>,
    search_scalars: &[R::Element],
) -> Result<NoetherNormalization<R>, NoetherNormalizationError>
where
    R: Field + fmt::Display,
{
    source.verify().map_err(|error| {
        NoetherNormalizationError::InvalidEvidence(format!("source ideal did not verify: {error}"))
    })?;
    let dimension = source.dimension()?;
    if dimension < 0 {
        return Err(NoetherNormalizationError::ZeroAlgebra);
    }
    let dimension = usize::try_from(dimension).map_err(|_| {
        NoetherNormalizationError::InvalidEvidence(
            "the source dimension does not fit the ambient arity".to_owned(),
        )
    })?;
    let arity = source.ring().variables().len();
    if dimension > arity {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "the source dimension exceeds its polynomial-ring arity".to_owned(),
        ));
    }
    let parameter_subsets = combinations(arity, dimension);
    let zero = source.ring().coefficient_ring().zero();
    let mut tested = 0_usize;

    // Coordinate normalizations are common and avoid constructing any shear.
    for parameters in &parameter_subsets {
        if tested == MAX_SEARCH_CANDIDATES {
            break;
        }
        let dependents = complement(arity, parameters);
        let coefficients = vec![vec![zero.clone(); dependents.len()]; parameters.len()];
        tested += 1;
        if let Some(result) = normalization_from_choice(source, parameters, &coefficients)? {
            return Ok(result);
        }
    }

    // Every linear parameter subspace lies in a shear chart for some pivot
    // subset. The bounded deterministic grid is a correctness-preserving
    // search: failure is typed and never fabricates finiteness evidence.
    for parameters in &parameter_subsets {
        let dependents = complement(arity, parameters);
        let entries = parameters.len().saturating_mul(dependents.len());
        if entries == 0 || search_scalars.len() < 2 {
            continue;
        }
        let total = bounded_power(search_scalars.len(), entries, MAX_SEARCH_CANDIDATES);
        for code in 1..total {
            if tested == MAX_SEARCH_CANDIDATES {
                break;
            }
            let coefficients =
                matrix_from_code(code, parameters.len(), dependents.len(), search_scalars);
            if coefficients
                .iter()
                .all(|row| row.iter().all(|coefficient| coefficient == &zero))
            {
                continue;
            }
            tested += 1;
            if let Some(result) = normalization_from_choice(source, parameters, &coefficients)? {
                return Ok(result);
            }
        }
        if tested == MAX_SEARCH_CANDIDATES {
            break;
        }
    }
    Err(NoetherNormalizationError::SearchExhausted { candidates: tested })
}

fn normalization_from_choice<R>(
    source: &Ideal<PolynomialRing<R>>,
    parameter_indices: &[usize],
    shear_coefficients: &[Vec<R::Element>],
) -> Result<Option<NoetherNormalization<R>>, NoetherNormalizationError>
where
    R: Field + fmt::Display,
{
    let arity = source.ring().variables().len();
    let dependent_indices = complement(arity, parameter_indices);
    validate_choice(
        source,
        parameter_indices,
        &dependent_indices,
        shear_coefficients,
    )?;
    let block_order = normalization_order(arity, &dependent_indices, parameter_indices)?;
    let transformed_ring = PolynomialRing::new(
        source.ring().coefficient_ring().clone(),
        source.ring().variables().iter().cloned(),
        block_order,
    )?;
    let transformed_coordinates = transformed_ring.generators()?;
    let inverse_images = inverse_coordinate_images(
        &transformed_ring,
        &transformed_coordinates,
        parameter_indices,
        &dependent_indices,
        shear_coefficients,
    )?;
    let inverse = source.ring().homomorphism(
        &transformed_ring,
        IdentityCoefficientMap::new(source.ring().coefficient_ring().clone()),
        inverse_images,
    )?;
    let transformed_generators = source
        .generators()
        .iter()
        .map(|generator| inverse.apply(generator))
        .collect::<Result<Vec<_>, _>>()?;
    let transformed_ideal = transformed_ring.ideal(transformed_generators)?;
    let Some(pure_power_bounds) = pure_power_bounds(&transformed_ideal, &dependent_indices) else {
        return Ok(None);
    };

    let dimension = usize::try_from(source.dimension()?).map_err(|_| {
        NoetherNormalizationError::InvalidEvidence(
            "a proper affine algebra has a negative dimension".to_owned(),
        )
    })?;
    if dimension != parameter_indices.len() {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "the parameter count changed from the exact quotient dimension".to_owned(),
        ));
    }
    let parameter_ring = PolynomialRing::new(
        source.ring().coefficient_ring().clone(),
        (0..dimension).map(|index| format!("t{index}")),
        MonomialOrder::GradedReverseLexicographic,
    )?;
    let ambient_coordinates = source.ring().generators()?;
    let coordinate_images = forward_coordinate_images(
        source.ring(),
        &ambient_coordinates,
        parameter_indices,
        &dependent_indices,
        shear_coefficients,
    )?;
    let algebra = source.ring().quotient(source)?;
    let parameter_images = parameter_indices
        .iter()
        .map(|index| algebra.element(&coordinate_images[*index]))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(NoetherNormalization {
        source: source.clone(),
        dimension,
        parameter_indices: parameter_indices.to_vec(),
        dependent_indices,
        shear_coefficients: shear_coefficients.to_vec(),
        transformed_ideal,
        pure_power_bounds,
        parameter_ring,
        algebra,
        coordinate_images,
        parameter_images,
    }))
}

fn validate_choice<R: Field>(
    source: &Ideal<PolynomialRing<R>>,
    parameters: &[usize],
    dependents: &[usize],
    coefficients: &[Vec<R::Element>],
) -> Result<(), NoetherNormalizationError> {
    let arity = source.ring().variables().len();
    if !strictly_increasing(parameters)
        || !strictly_increasing(dependents)
        || parameters.iter().any(|index| *index >= arity)
        || dependents.iter().any(|index| *index >= arity)
        || parameters.len() + dependents.len() != arity
        || parameters.iter().any(|index| dependents.contains(index))
    {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "parameter and dependent indices do not partition the ambient variables".to_owned(),
        ));
    }
    if coefficients.len() != parameters.len()
        || coefficients.iter().any(|row| row.len() != dependents.len())
        || coefficients
            .iter()
            .flatten()
            .any(|coefficient| !source.ring().coefficient_ring().contains(coefficient))
    {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "the shear matrix has the wrong shape or coefficient context".to_owned(),
        ));
    }
    Ok(())
}

fn normalization_order(
    arity: usize,
    dependents: &[usize],
    parameters: &[usize],
) -> Result<MonomialOrder, NoetherNormalizationError> {
    if arity == 0 || dependents.is_empty() {
        return Ok(MonomialOrder::GradedReverseLexicographic);
    }
    if parameters.is_empty() {
        return Ok(MonomialOrder::Lexicographic);
    }
    MonomialOrder::block([
        (dependents.to_vec(), MonomialOrder::Lexicographic),
        (
            parameters.to_vec(),
            MonomialOrder::GradedReverseLexicographic,
        ),
    ])
    .map_err(|error| NoetherNormalizationError::PolynomialContext(error.to_string()))
}

fn inverse_coordinate_images<R: Field>(
    ring: &PolynomialRing<R>,
    coordinates: &[Polynomial<R>],
    parameters: &[usize],
    dependents: &[usize],
    coefficients: &[Vec<R::Element>],
) -> Result<Vec<Polynomial<R>>, NoetherNormalizationError> {
    let mut images = coordinates.to_vec();
    for (row, &parameter) in parameters.iter().enumerate() {
        let mut image = coordinates[parameter].clone();
        for (column, &dependent) in dependents.iter().enumerate() {
            let term = coordinates[dependent].scalar_mul(&coefficients[row][column])?;
            image = ring.sub(&image, &term)?;
        }
        images[parameter] = image;
    }
    Ok(images)
}

fn forward_coordinate_images<R: Field>(
    ring: &PolynomialRing<R>,
    coordinates: &[Polynomial<R>],
    parameters: &[usize],
    dependents: &[usize],
    coefficients: &[Vec<R::Element>],
) -> Result<Vec<Polynomial<R>>, NoetherNormalizationError> {
    let mut images = coordinates.to_vec();
    for (row, &parameter) in parameters.iter().enumerate() {
        let mut image = coordinates[parameter].clone();
        for (column, &dependent) in dependents.iter().enumerate() {
            let term = coordinates[dependent].scalar_mul(&coefficients[row][column])?;
            image = ring.add(&image, &term)?;
        }
        images[parameter] = image;
    }
    Ok(images)
}

fn pure_power_bounds<R: Field>(
    ideal: &Ideal<PolynomialRing<R>>,
    dependents: &[usize],
) -> Option<Vec<Integer>> {
    dependents
        .iter()
        .map(|&dependent| {
            ideal
                .generators()
                .iter()
                .filter_map(|generator| generator.terms().next().map(|(powers, _)| powers))
                .filter(|powers| {
                    !powers[dependent].is_zero()
                        && powers
                            .iter()
                            .enumerate()
                            .all(|(index, exponent)| index == dependent || exponent.is_zero())
                })
                .map(|powers| powers[dependent].clone())
                .min()
        })
        .collect()
}

fn verify_normalization<R>(value: &NoetherNormalization<R>) -> Result<(), NoetherNormalizationError>
where
    R: Field + fmt::Display,
{
    value.source.verify().map_err(|error| {
        NoetherNormalizationError::InvalidEvidence(format!("source ideal did not verify: {error}"))
    })?;
    value.transformed_ideal.verify().map_err(|error| {
        NoetherNormalizationError::InvalidEvidence(format!(
            "transformed ideal did not verify: {error}"
        ))
    })?;
    let source_dimension = value.source.dimension()?;
    if source_dimension < 0 || usize::try_from(source_dimension).ok() != Some(value.dimension) {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "retained dimension differs from the exact affine-algebra dimension".to_owned(),
        ));
    }
    validate_choice(
        &value.source,
        &value.parameter_indices,
        &value.dependent_indices,
        &value.shear_coefficients,
    )?;
    if value.parameter_indices.len() != value.dimension
        || value.pure_power_bounds.len() != value.dependent_indices.len()
    {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "parameter dimension or pure-power evidence has the wrong arity".to_owned(),
        ));
    }
    let Some(recomputed) = normalization_from_choice(
        &value.source,
        &value.parameter_indices,
        &value.shear_coefficients,
    )?
    else {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "the retained coordinate change is not module-finite".to_owned(),
        ));
    };
    if &recomputed != value {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "recomputation changed the parameter map or finite-module evidence".to_owned(),
        ));
    }
    // Reconstructing the full rectangular family is deliberately optional;
    // the pure leading powers are the compact exact spanning certificate.
    if value
        .pure_power_bounds
        .iter()
        .any(|bound| bound <= &Integer::zero())
    {
        return Err(NoetherNormalizationError::InvalidEvidence(
            "a pure-power module bound is not positive".to_owned(),
        ));
    }
    Ok(())
}

impl Ideal<PolynomialRing<RationalField>> {
    /// Compute a deterministic finite polynomial-subring normalization over `Q`.
    pub fn noether_normalization(
        &self,
    ) -> Result<NoetherNormalization<RationalField>, NoetherNormalizationError> {
        let scalars = [0_i64, 1, -1, 2, -2]
            .into_iter()
            .map(Rational::from)
            .collect::<Vec<_>>();
        construct_normalization(self, &scalars)
    }
}

impl Ideal<PolynomialRing<FiniteField>> {
    /// Compute a deterministic finite polynomial-subring normalization over a finite field.
    pub fn noether_normalization(
        &self,
    ) -> Result<NoetherNormalization<FiniteField>, NoetherNormalizationError> {
        let scalars = self
            .ring()
            .coefficient_ring()
            .elements()
            .take(MAX_SEARCH_SCALARS)
            .collect::<Vec<FiniteFieldElement>>();
        construct_normalization(self, &scalars)
    }
}

macro_rules! concrete_methods {
    ($coefficient:ty) => {
        impl NoetherNormalization<$coefficient> {
            /// Independently replay the coordinate change and finite-module proof.
            pub fn verify(&self) -> Result<(), NoetherNormalizationError> {
                verify_normalization(self)
            }

            /// Deserialize and exactly reverify a Noether normalization.
            pub fn from_json(value: &str) -> Result<Self, NoetherError> {
                serde_json::from_str(value)
                    .map_err(|error| NoetherError::Serialization(error.to_string()))
            }
        }
    };
}

concrete_methods!(RationalField);
concrete_methods!(FiniteField);

impl<R> NoetherNormalization<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    Polynomial<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
    QuotientElement<PolynomialRing<R>>: Serialize,
{
    /// Serialize the source, finite map, coordinate change, and spanning evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }
}

impl<R> SemanticType for NoetherNormalization<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    Polynomial<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
    QuotientElement<PolynomialRing<R>>: Serialize,
{
    const TYPE_ID: &'static str = "noether/commalg/NoetherNormalization@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &NOETHER_NORMALIZATION_DESCRIPTION
    }
}

impl<R> Serialize for NoetherNormalization<R>
where
    R: Field + Serialize,
    R::Element: Serialize,
    Ideal<PolynomialRing<R>>: Serialize,
    Polynomial<R>: Serialize,
    QuotientRing<PolynomialRing<R>>: Serialize,
    QuotientElement<PolynomialRing<R>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("NoetherNormalization", 12)?;
        state.serialize_field("type", Self::TYPE_ID)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("dimension", &self.dimension)?;
        state.serialize_field("parameter_indices", &self.parameter_indices)?;
        state.serialize_field("dependent_indices", &self.dependent_indices)?;
        state.serialize_field("shear_coefficients", &self.shear_coefficients)?;
        state.serialize_field("transformed_ideal", &self.transformed_ideal)?;
        state.serialize_field("pure_power_bounds", &self.pure_power_bounds)?;
        state.serialize_field("parameter_ring", &self.parameter_ring)?;
        state.serialize_field("algebra", &self.algebra)?;
        state.serialize_field("coordinate_images", &self.coordinate_images)?;
        state.serialize_field("parameter_images", &self.parameter_images)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        deserialize = "R: Field + Deserialize<'de>, R::Element: Deserialize<'de>, Ideal<PolynomialRing<R>>: Deserialize<'de>, Polynomial<R>: Deserialize<'de>, QuotientRing<PolynomialRing<R>>: Deserialize<'de>, QuotientElement<PolynomialRing<R>>: Deserialize<'de>"
    )
)]
struct NoetherNormalizationWire<R: Field> {
    #[serde(rename = "type")]
    type_id: String,
    source: Ideal<PolynomialRing<R>>,
    dimension: usize,
    parameter_indices: Vec<usize>,
    dependent_indices: Vec<usize>,
    shear_coefficients: Vec<Vec<R::Element>>,
    transformed_ideal: Ideal<PolynomialRing<R>>,
    pure_power_bounds: Vec<Integer>,
    parameter_ring: PolynomialRing<R>,
    algebra: QuotientRing<PolynomialRing<R>>,
    coordinate_images: Vec<Polynomial<R>>,
    parameter_images: Vec<QuotientElement<PolynomialRing<R>>>,
}

fn deserialize_normalization<'de, R, D>(
    deserializer: D,
) -> Result<NoetherNormalization<R>, D::Error>
where
    R: Field + fmt::Display + Serialize + Deserialize<'de>,
    R::Element: Serialize + Deserialize<'de>,
    Ideal<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    Polynomial<R>: Serialize + Deserialize<'de>,
    QuotientRing<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    QuotientElement<PolynomialRing<R>>: Serialize + Deserialize<'de>,
    D: Deserializer<'de>,
{
    let wire = NoetherNormalizationWire::<R>::deserialize(deserializer)?;
    if wire.type_id != NoetherNormalization::<R>::TYPE_ID {
        return Err(serde::de::Error::custom(format!(
            "expected type {}, received {}",
            NoetherNormalization::<R>::TYPE_ID,
            wire.type_id
        )));
    }
    let result = NoetherNormalization {
        source: wire.source,
        dimension: wire.dimension,
        parameter_indices: wire.parameter_indices,
        dependent_indices: wire.dependent_indices,
        shear_coefficients: wire.shear_coefficients,
        transformed_ideal: wire.transformed_ideal,
        pure_power_bounds: wire.pure_power_bounds,
        parameter_ring: wire.parameter_ring,
        algebra: wire.algebra,
        coordinate_images: wire.coordinate_images,
        parameter_images: wire.parameter_images,
    };
    verify_normalization(&result).map_err(serde::de::Error::custom)?;
    Ok(result)
}

impl<'de> Deserialize<'de> for NoetherNormalization<RationalField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_normalization(deserializer)
    }
}

impl<'de> Deserialize<'de> for NoetherNormalization<FiniteField> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_normalization(deserializer)
    }
}

fn combinations(arity: usize, count: usize) -> Vec<Vec<usize>> {
    fn recurse(
        next: usize,
        arity: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<Vec<usize>>,
    ) {
        if remaining == 0 {
            output.push(current.clone());
            return;
        }
        if next >= arity || arity - next < remaining {
            return;
        }
        for index in next..=arity - remaining {
            current.push(index);
            recurse(index + 1, arity, remaining - 1, current, output);
            current.pop();
        }
    }
    let mut output = Vec::new();
    recurse(0, arity, count, &mut Vec::new(), &mut output);
    output
}

fn complement(arity: usize, selected: &[usize]) -> Vec<usize> {
    (0..arity)
        .filter(|index| !selected.contains(index))
        .collect()
}

fn strictly_increasing(values: &[usize]) -> bool {
    values.windows(2).all(|window| window[0] < window[1])
}

fn bounded_power(base: usize, exponent: usize, limit: usize) -> usize {
    (0..exponent).fold(1_usize, |value, _| value.saturating_mul(base).min(limit))
}

fn matrix_from_code<E: Clone>(
    mut code: usize,
    rows: usize,
    columns: usize,
    scalars: &[E],
) -> Vec<Vec<E>> {
    let mut matrix = vec![vec![scalars[0].clone(); columns]; rows];
    for row in &mut matrix {
        for entry in row {
            *entry = scalars[code % scalars.len()].clone();
            code /= scalars.len();
        }
    }
    matrix
}

fn integer_to_usize(value: &Integer) -> Option<usize> {
    value.to_string().parse().ok()
}

fn integer_to_u64(value: &Integer) -> Option<u64> {
    value.to_string().parse().ok()
}

fn enumerate_rectangular_exponents(
    position: usize,
    bounds: &[usize],
    current: &mut [usize],
    dependent_indices: &[usize],
    arity: usize,
    output: &mut Vec<Vec<Integer>>,
) {
    if position == bounds.len() {
        let mut exponent = vec![Integer::zero(); arity];
        for (&index, &value) in dependent_indices.iter().zip(current.iter()) {
            exponent[index] = Integer::from(value);
        }
        output.push(exponent);
        return;
    }
    for exponent in 0..bounds[position] {
        current[position] = exponent;
        enumerate_rectangular_exponents(
            position + 1,
            bounds,
            current,
            dependent_indices,
            arity,
            output,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numthy::FiniteField;
    use crate::Q;

    #[test]
    fn coordinate_parameter_normalizes_a_parabola() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([ring
                .sub(&variables[1], &variables[0].pow(2).unwrap())
                .unwrap()])
            .unwrap();

        let normalization = source.noether_normalization().unwrap();
        assert_eq!(normalization.dimension(), 1);
        assert_eq!(normalization.parameter_indices(), &[0]);
        assert_eq!(normalization.dependent_indices(), &[1]);
        assert_eq!(normalization.pure_power_bounds(), &[Integer::one()]);
        assert_eq!(normalization.module_generator_count(), Integer::one());
        let parameter = normalization.parameter_ring().generator(0).unwrap();
        assert_eq!(
            normalization.apply(&parameter.pow(2).unwrap()).unwrap(),
            normalization.algebra().element(&variables[1]).unwrap()
        );
        normalization.verify().unwrap();
        assert_eq!(
            NoetherNormalization::<RationalField>::from_json(&normalization.to_json().unwrap())
                .unwrap(),
            normalization
        );
    }

    #[test]
    fn deterministic_shear_normalizes_the_hyperbola() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([ring
                .sub(
                    &ring.mul(&variables[0], &variables[1]).unwrap(),
                    &ring.one(),
                )
                .unwrap()])
            .unwrap();

        let normalization = source.noether_normalization().unwrap();
        assert_eq!(normalization.dimension(), 1);
        assert_eq!(normalization.parameter_indices(), &[0]);
        assert_eq!(normalization.dependent_indices(), &[1]);
        assert_eq!(normalization.shear_coefficients(), &[vec![Rational::one()]]);
        assert_eq!(normalization.pure_power_bounds(), &[Integer::from(2_u8)]);
        assert_eq!(normalization.module_generators().unwrap().len(), 2);
        assert_eq!(
            normalization.parameter_images()[0],
            normalization
                .algebra()
                .element(&ring.add(&variables[0], &variables[1]).unwrap())
                .unwrap()
        );
        normalization.verify().unwrap();

        let mut payload = serde_json::to_value(&normalization).unwrap();
        payload["pure_power_bounds"][0] = serde_json::Value::String("3".to_owned());
        assert!(serde_json::from_value::<NoetherNormalization<RationalField>>(payload).is_err());
    }

    #[test]
    fn finite_field_shear_has_the_same_exact_evidence() {
        let field = FiniteField::prime(2_u8).unwrap();
        let ring =
            PolynomialRing::new(field, ["x", "y"], MonomialOrder::GradedReverseLexicographic)
                .unwrap();
        let variables = ring.generators().unwrap();
        let source = ring
            .ideal([ring
                .sub(
                    &ring.mul(&variables[0], &variables[1]).unwrap(),
                    &ring.one(),
                )
                .unwrap()])
            .unwrap();

        let normalization = source.noether_normalization().unwrap();
        assert_eq!(normalization.dimension(), 1);
        assert_eq!(normalization.pure_power_bounds(), &[Integer::from(2_u8)]);
        normalization.verify().unwrap();
        assert_eq!(
            NoetherNormalization::<FiniteField>::from_json(&normalization.to_json().unwrap())
                .unwrap(),
            normalization
        );
    }

    #[test]
    fn polynomial_ring_and_zero_dimensional_algebra_are_normalized() {
        let ring = PolynomialRing::new(Q, ["x", "y"], MonomialOrder::default()).unwrap();
        let polynomial_ring = ring.ideal([]).unwrap().noether_normalization().unwrap();
        assert_eq!(polynomial_ring.dimension(), 2);
        assert!(polynomial_ring.dependent_indices().is_empty());
        assert_eq!(polynomial_ring.module_generator_count(), Integer::one());
        polynomial_ring.verify().unwrap();

        let variables = ring.generators().unwrap();
        let finite = ring
            .ideal([variables[0].pow(2).unwrap(), variables[1].pow(3).unwrap()])
            .unwrap()
            .noether_normalization()
            .unwrap();
        assert_eq!(finite.dimension(), 0);
        assert!(finite.parameter_indices().is_empty());
        assert_eq!(finite.module_generator_count(), Integer::from(6_u8));
        finite.verify().unwrap();
    }

    #[test]
    fn zero_algebra_fails_closed() {
        let ring = PolynomialRing::new(Q, ["x"], MonomialOrder::default()).unwrap();
        assert_eq!(
            ring.ideal([ring.one()])
                .unwrap()
                .noether_normalization()
                .unwrap_err(),
            NoetherNormalizationError::ZeroAlgebra
        );
    }
}
