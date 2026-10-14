// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Verified local algebra around exact scheme points.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::numthy::FiniteField;
use crate::{
    ContextMismatchError, Field, Integer, NoetherError, Ring, SemanticType, TypeDescription,
};

use super::scheme::affine_closed_subscheme;
use super::{
    AffineSpace, GroebnerError, LocalAlgebraError, LocalMonomialOrder, LocalPolynomialRing,
    Polynomial, PolynomialContextError, PolynomialError, PolynomialRing, PowerSeries,
    PowerSeriesError, PowerSeriesRing, Scheme, SchemeError, SchemePoint, SeriesContextError,
    SeriesTruncation,
};

const MAX_BLOW_UP_CHARTS: usize = 8;

static BLOW_UP_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/BlowUp@1",
    rust_path: "noether::commalg::BlowUp",
    python_path: Some("noether.commalg.BlowUp"),
};

static BLOW_UP_CHART_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/BlowUpChart@1",
    rust_path: "noether::commalg::BlowUpChart",
    python_path: Some("noether.commalg.BlowUpChart"),
};

static SCHEME_COMPLETION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/commalg/SchemeCompletion@1",
    rust_path: "noether::commalg::SchemeCompletion",
    python_path: Some("noether.commalg.SchemeCompletion"),
};

/// Failure to construct or verify bounded exact local geometry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeometryError {
    /// An ambient space, scheme, or point was invalid.
    Scheme(SchemeError),
    /// A polynomial-ring context could not be constructed.
    PolynomialContext(PolynomialContextError),
    /// Exact polynomial arithmetic failed.
    Polynomial(PolynomialError),
    /// Canonical ideal arithmetic failed.
    Groebner(GroebnerError),
    /// Polynomial-localization or Mora standard-basis arithmetic failed.
    LocalAlgebra(LocalAlgebraError),
    /// Values or maps belonged to unequal exact contexts.
    ContextMismatch(ContextMismatchError),
    /// A fixed-truncation series context could not be constructed.
    SeriesContext(SeriesContextError),
    /// Fixed-truncation series arithmetic failed.
    PowerSeries(PowerSeriesError),
    /// The requested exact case lies outside the deliberately bounded floor.
    Unsupported(String),
    /// A fixed native work bound was exceeded.
    ResourceLimit {
        /// Bounded resource.
        resource: &'static str,
        /// Fixed limit.
        limit: usize,
    },
    /// Supplied or computed geometric evidence failed verification.
    Invalid(String),
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheme(error) => error.fmt(formatter),
            Self::PolynomialContext(error) => error.fmt(formatter),
            Self::Polynomial(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::LocalAlgebra(error) => error.fmt(formatter),
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::SeriesContext(error) => error.fmt(formatter),
            Self::PowerSeries(error) => error.fmt(formatter),
            Self::Unsupported(reason) => write!(formatter, "unsupported local geometry: {reason}"),
            Self::ResourceLimit { resource, limit } => write!(
                formatter,
                "local geometry exceeded its {resource} limit of {limit}"
            ),
            Self::Invalid(reason) => write!(formatter, "invalid local geometry: {reason}"),
        }
    }
}

impl std::error::Error for GeometryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Scheme(error) => Some(error),
            Self::PolynomialContext(error) => Some(error),
            Self::Polynomial(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::LocalAlgebra(error) => Some(error),
            Self::ContextMismatch(error) => Some(error),
            Self::SeriesContext(error) => Some(error),
            Self::PowerSeries(error) => Some(error),
            Self::Unsupported(_) | Self::ResourceLimit { .. } | Self::Invalid(_) => None,
        }
    }
}

impl From<SchemeError> for GeometryError {
    fn from(error: SchemeError) -> Self {
        Self::Scheme(error)
    }
}

impl From<PolynomialContextError> for GeometryError {
    fn from(error: PolynomialContextError) -> Self {
        Self::PolynomialContext(error)
    }
}

impl From<PolynomialError> for GeometryError {
    fn from(error: PolynomialError) -> Self {
        Self::Polynomial(error)
    }
}

impl From<GroebnerError> for GeometryError {
    fn from(error: GroebnerError) -> Self {
        Self::Groebner(error)
    }
}

impl From<LocalAlgebraError> for GeometryError {
    fn from(error: LocalAlgebraError) -> Self {
        Self::LocalAlgebra(error)
    }
}

impl From<ContextMismatchError> for GeometryError {
    fn from(error: ContextMismatchError) -> Self {
        Self::ContextMismatch(error)
    }
}

impl From<SeriesContextError> for GeometryError {
    fn from(error: SeriesContextError) -> Self {
        Self::SeriesContext(error)
    }
}

impl From<PowerSeriesError> for GeometryError {
    fn from(error: PowerSeriesError) -> Self {
        Self::PowerSeries(error)
    }
}

/// One affine chart of the point blow-up, with all transform evidence.
#[derive(Clone, PartialEq, Eq)]
pub struct BlowUpChart<R: Field> {
    source: Scheme<R>,
    center: SchemePoint<R>,
    chart_index: usize,
    source_coordinate_map: Vec<Polynomial<R>>,
    total_transform: Scheme<R>,
    weak_transform: Scheme<R>,
    strict_transform: Scheme<R>,
    exceptional_divisor: Scheme<R>,
}

impl<R: Field> BlowUpChart<R> {
    /// Scheme being blown up.
    pub fn source(&self) -> &Scheme<R> {
        &self.source
    }

    /// Rational point used as the center.
    pub fn center(&self) -> &SchemePoint<R> {
        &self.center
    }

    /// Index of the translated local coordinate chosen as exceptional coordinate.
    pub fn chart_index(&self) -> usize {
        self.chart_index
    }

    /// Pullbacks of the source ambient coordinates in this affine chart.
    pub fn source_coordinate_map(&self) -> &[Polynomial<R>] {
        &self.source_coordinate_map
    }

    /// Scheme-theoretic total transform before removing exceptional factors.
    pub fn total_transform(&self) -> &Scheme<R> {
        &self.total_transform
    }

    /// Transform after dividing by the common exceptional order.
    pub fn weak_transform(&self) -> &Scheme<R> {
        &self.weak_transform
    }

    /// Saturation of the total transform by the exceptional coordinate.
    pub fn strict_transform(&self) -> &Scheme<R> {
        &self.strict_transform
    }

    /// Scheme-theoretic intersection of the strict transform with the exceptional divisor.
    pub fn exceptional_divisor(&self) -> &Scheme<R> {
        &self.exceptional_divisor
    }

    /// Recheck the saturation and exceptional-intersection identities in this chart.
    pub fn verify(&self) -> Result<(), GeometryError> {
        let ring = self.total_transform.coordinate_ring();
        if self.chart_index >= ring.variables().len()
            || self.weak_transform.coordinate_ring() != ring
            || self.strict_transform.coordinate_ring() != ring
            || self.exceptional_divisor.coordinate_ring() != ring
            || self
                .source_coordinate_map
                .iter()
                .any(|coordinate| coordinate.ring() != ring)
        {
            return Err(GeometryError::Invalid(
                "a blow-up chart contains incompatible coordinate contexts".to_owned(),
            ));
        }
        let exceptional = ring.ideal([ring.generator(self.chart_index)?])?;
        let saturated = self
            .total_transform
            .defining_ideal()
            .saturate(&exceptional)?;
        if &saturated != self.strict_transform.defining_ideal() {
            return Err(GeometryError::Invalid(
                "the strict transform is not the exceptional saturation of the total transform"
                    .to_owned(),
            ));
        }
        if self
            .weak_transform
            .defining_ideal()
            .saturate(&exceptional)?
            != saturated
        {
            return Err(GeometryError::Invalid(
                "the weak transform does not saturate to the strict transform".to_owned(),
            ));
        }
        let mut exceptional_generators = saturated.generators().to_vec();
        exceptional_generators.push(ring.generator(self.chart_index)?);
        if ring.ideal(exceptional_generators)? != *self.exceptional_divisor.defining_ideal() {
            return Err(GeometryError::Invalid(
                "the exceptional scheme is not the strict-transform intersection".to_owned(),
            ));
        }
        let rebuilt = build_blow_up(&self.source, &self.center)?;
        if rebuilt.charts.get(self.chart_index) != Some(self) {
            return Err(GeometryError::Invalid(
                "the chart does not reconstruct from its source and center".to_owned(),
            ));
        }
        Ok(())
    }
}

impl<R: Field> fmt::Debug for BlowUpChart<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlowUpChart")
            .field("chart_index", &self.chart_index)
            .field("source_coordinates", &self.source_coordinate_map.len())
            .finish()
    }
}

/// Canonical affine charts over a point blow-up's exceptional neighbourhood.
///
/// For a projective source, `source_affine_chart` is the first nonzero
/// homogeneous coordinate of the normalized center. The charts cover the
/// inverse image of that canonical affine neighbourhood; away from the center
/// the blow-up is canonically the unchanged source.
#[derive(Clone, PartialEq, Eq)]
pub struct BlowUp<R: Field> {
    source: Scheme<R>,
    center: SchemePoint<R>,
    source_affine_chart: Option<usize>,
    local_source: Scheme<R>,
    charts: Vec<BlowUpChart<R>>,
}

impl<R: Field> BlowUp<R> {
    /// Scheme being blown up.
    pub fn source(&self) -> &Scheme<R> {
        &self.source
    }

    /// Rational point used as the center.
    pub fn center(&self) -> &SchemePoint<R> {
        &self.center
    }

    /// Dehomogenizing coordinate for a projective source; `None` for an affine source.
    pub fn source_affine_chart(&self) -> Option<usize> {
        self.source_affine_chart
    }

    /// Source restricted to translated affine coordinates centered at the point.
    pub fn local_source(&self) -> &Scheme<R> {
        &self.local_source
    }

    /// Canonically ordered point-blow-up charts, one per translated local coordinate.
    pub fn charts(&self) -> &[BlowUpChart<R>] {
        &self.charts
    }

    /// Reconstruct every chart and recheck total/weak/strict/exceptional evidence.
    pub fn verify(&self) -> Result<(), GeometryError> {
        let rebuilt = build_blow_up(&self.source, &self.center)?;
        if &rebuilt != self {
            return Err(GeometryError::Invalid(
                "the point blow-up does not match its source and center".to_owned(),
            ));
        }
        Ok(())
    }
}

impl<R: Field> fmt::Debug for BlowUp<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlowUp")
            .field("source_affine_chart", &self.source_affine_chart)
            .field("charts", &self.charts.len())
            .finish()
    }
}

/// A fixed-order presentation of a completed local scheme.
///
/// The ambient power-series ring and truncated defining equations stay
/// together: this is the quotient presentation `R[[u_1,...,u_n]]/(f_i)`
/// modulo the context's explicit truncation, not a bare series context.
#[derive(Clone, PartialEq, Eq)]
pub struct SchemeCompletion<R: Field> {
    source: Scheme<R>,
    center: SchemePoint<R>,
    ambient_ring: PowerSeriesRing<R>,
    equations: Vec<PowerSeries<R>>,
}

impl<R: Field> SchemeCompletion<R> {
    /// Scheme whose completed local presentation is recorded.
    pub fn source(&self) -> &Scheme<R> {
        &self.source
    }

    /// Rational center of the completion.
    pub fn center(&self) -> &SchemePoint<R> {
        &self.center
    }

    /// Fixed-truncation ambient power-series context.
    pub fn ambient_ring(&self) -> &PowerSeriesRing<R> {
        &self.ambient_ring
    }

    /// Truncated equations generating the completed local quotient presentation.
    pub fn equations(&self) -> &[PowerSeries<R>] {
        &self.equations
    }

    /// Immutable absolute monomial truncation.
    pub fn truncation(&self) -> &SeriesTruncation {
        self.ambient_ring.truncation()
    }

    /// Reconstruct and compare the translated fixed-order local equations.
    pub fn verify(&self) -> Result<(), GeometryError> {
        let rebuilt = build_completion(&self.source, &self.center, self.truncation().clone())?;
        if &rebuilt == self {
            Ok(())
        } else {
            Err(GeometryError::Invalid(
                "the completion does not match its source, center, and truncation".to_owned(),
            ))
        }
    }
}

impl<R: Field> fmt::Debug for SchemeCompletion<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SchemeCompletion")
            .field("variables", &self.ambient_ring.variables())
            .field("equations", &self.equations.len())
            .finish()
    }
}

impl<R: Field> Scheme<R> {
    /// Construct verified affine charts for the blow-up at one rational point.
    pub fn blow_up_at(&self, center: &SchemePoint<R>) -> Result<BlowUp<R>, GeometryError> {
        let blow_up = build_blow_up(self, center)?;
        blow_up.verify()?;
        Ok(blow_up)
    }

    /// Form the tangent cone from a degree-compatible Mora standard basis.
    pub fn tangent_cone(&self, point: &SchemePoint<R>) -> Result<Scheme<R>, GeometryError> {
        let local = local_presentation(self, point)?;
        let ring = local.scheme.coordinate_ring();
        let localization = LocalPolynomialRing::at_origin(
            ring.clone(),
            LocalMonomialOrder::LocalDegreeReverseLexicographic,
        )?;
        let generators = local
            .scheme
            .defining_ideal()
            .generators()
            .iter()
            .map(|generator| localization.from_polynomial(generator))
            .collect::<Result<Vec<_>, _>>()?;
        let standard_basis = localization.standard_basis(&generators)?;
        let tangent_ideal = standard_basis.tangent_cone_ideal()?;
        let tangent_names = localization.coordinate_ring().variables().to_vec();
        let tangent_space =
            AffineSpace::with_names(self.base().clone(), tangent_names.len(), tangent_names)?;
        affine_closed_subscheme(&tangent_space, tangent_ideal).map_err(Into::into)
    }

    /// Complete at a rational point to one explicit finite monomial order.
    pub fn completion(
        &self,
        point: &SchemePoint<R>,
        truncation: SeriesTruncation,
    ) -> Result<SchemeCompletion<R>, GeometryError> {
        build_completion(self, point, truncation)
    }
}

struct LocalPresentation<R: Field> {
    source_affine_chart: Option<usize>,
    scheme: Scheme<R>,
    source_coordinate_map: Vec<Polynomial<R>>,
}

fn local_presentation<R: Field>(
    source: &Scheme<R>,
    point: &SchemePoint<R>,
) -> Result<LocalPresentation<R>, GeometryError> {
    point.verify()?;
    if point.scheme() != source {
        return Err(GeometryError::Invalid(
            "the local point belongs to a different scheme".to_owned(),
        ));
    }
    let source_ring = source.coordinate_ring();
    let source_coordinate_count = source_ring.variables().len();
    let source_affine_chart = if source.is_projective() {
        let zero = source.base().zero();
        Some(
            point
                .coordinates()
                .iter()
                .position(|coordinate| coordinate != &zero)
                .ok_or(SchemeError::ZeroProjectiveCoordinates)?,
        )
    } else {
        None
    };
    let local_names = source_ring
        .variables()
        .iter()
        .enumerate()
        .filter(|(index, _)| Some(*index) != source_affine_chart)
        .map(|(_, name)| name.clone())
        .collect::<Vec<_>>();
    if local_names.is_empty() {
        return Err(GeometryError::Unsupported(
            "a zero-dimensional ambient space has no point blow-up charts".to_owned(),
        ));
    }
    let local_space =
        AffineSpace::with_names(source.base().clone(), local_names.len(), local_names)?;
    let local_ring = local_space.coordinate_ring();
    let local_variables = local_ring.generators()?;
    let mut local_position = 0_usize;
    let mut source_coordinate_map = Vec::with_capacity(source_coordinate_count);
    for source_index in 0..source_coordinate_count {
        if Some(source_index) == source_affine_chart {
            source_coordinate_map.push(local_ring.one());
        } else {
            let translated = local_ring.add(
                &local_variables[local_position],
                &local_ring.constant(point.coordinates()[source_index].clone())?,
            )?;
            source_coordinate_map.push(translated);
            local_position += 1;
        }
    }
    let equations = source
        .defining_ideal()
        .generators()
        .iter()
        .map(|equation| map_polynomial(equation, local_ring, &source_coordinate_map))
        .collect::<Result<Vec<_>, _>>()?;
    let ideal = local_ring.ideal(equations)?;
    let scheme = affine_closed_subscheme(&local_space, ideal)?;
    Ok(LocalPresentation {
        source_affine_chart,
        scheme,
        source_coordinate_map,
    })
}

fn build_blow_up<R: Field>(
    source: &Scheme<R>,
    center: &SchemePoint<R>,
) -> Result<BlowUp<R>, GeometryError> {
    let local = local_presentation(source, center)?;
    let local_ring = local.scheme.coordinate_ring();
    if local.scheme.defining_ideal().generators().is_empty()
        || local
            .scheme
            .defining_ideal()
            .generators()
            .iter()
            .all(Polynomial::is_zero)
    {
        return Err(GeometryError::Unsupported(
            "the whole ambient space has no bounded transform presentation".to_owned(),
        ));
    }
    let local_count = local_ring.variables().len();
    if local_count > MAX_BLOW_UP_CHARTS {
        return Err(GeometryError::ResourceLimit {
            resource: "point-blow-up chart count",
            limit: MAX_BLOW_UP_CHARTS,
        });
    }
    let mut charts = Vec::with_capacity(local_count);
    for chart_index in 0..local_count {
        let exceptional_name = &local_ring.variables()[chart_index];
        let chart_names = local_ring
            .variables()
            .iter()
            .enumerate()
            .map(|(index, name)| {
                if index == chart_index {
                    format!("{name}_exceptional")
                } else {
                    format!("{name}_over_{exceptional_name}")
                }
            })
            .collect::<Vec<_>>();
        let chart_space = AffineSpace::with_names(source.base().clone(), local_count, chart_names)?;
        let chart_ring = chart_space.coordinate_ring();
        let chart_variables = chart_ring.generators()?;
        let exceptional_coordinate = &chart_variables[chart_index];
        let local_coordinate_map = chart_variables
            .iter()
            .enumerate()
            .map(|(index, coordinate)| {
                if index == chart_index {
                    Ok(coordinate.clone())
                } else {
                    chart_ring
                        .mul(exceptional_coordinate, coordinate)
                        .map_err(PolynomialError::from)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let source_coordinate_map = local
            .source_coordinate_map
            .iter()
            .map(|coordinate| map_polynomial(coordinate, chart_ring, &local_coordinate_map))
            .collect::<Result<Vec<_>, _>>()?;
        let total_generators = local
            .scheme
            .defining_ideal()
            .generators()
            .iter()
            .map(|equation| map_polynomial(equation, chart_ring, &local_coordinate_map))
            .collect::<Result<Vec<_>, _>>()?;
        let common_order = exceptional_order(&total_generators, chart_index)?;
        let weak_generators = total_generators
            .iter()
            .map(|equation| divide_exceptional_power(equation, chart_index, &common_order))
            .collect::<Result<Vec<_>, _>>()?;
        let total_ideal = chart_ring.ideal(total_generators)?;
        let weak_ideal = chart_ring.ideal(weak_generators)?;
        let exceptional_ideal = chart_ring.ideal([exceptional_coordinate.clone()])?;
        let strict_ideal = total_ideal.saturate(&exceptional_ideal)?;
        let mut exceptional_generators = strict_ideal.generators().to_vec();
        exceptional_generators.push(exceptional_coordinate.clone());
        let exceptional_intersection = chart_ring.ideal(exceptional_generators)?;
        charts.push(BlowUpChart {
            source: source.clone(),
            center: center.clone(),
            chart_index,
            source_coordinate_map,
            total_transform: affine_closed_subscheme(&chart_space, total_ideal)?,
            weak_transform: affine_closed_subscheme(&chart_space, weak_ideal)?,
            strict_transform: affine_closed_subscheme(&chart_space, strict_ideal)?,
            exceptional_divisor: affine_closed_subscheme(&chart_space, exceptional_intersection)?,
        });
    }
    Ok(BlowUp {
        source: source.clone(),
        center: center.clone(),
        source_affine_chart: local.source_affine_chart,
        local_source: local.scheme,
        charts,
    })
}

fn build_completion<R: Field>(
    source: &Scheme<R>,
    center: &SchemePoint<R>,
    truncation: SeriesTruncation,
) -> Result<SchemeCompletion<R>, GeometryError> {
    let local = local_presentation(source, center)?;
    let polynomial_ring = local.scheme.coordinate_ring();
    let ambient_ring = PowerSeriesRing::new(
        source.base().clone(),
        polynomial_ring.variables().to_vec(),
        truncation,
    )?;
    let equations = local
        .scheme
        .defining_ideal()
        .generators()
        .iter()
        .map(|equation| polynomial_to_series(equation, &ambient_ring))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SchemeCompletion {
        source: source.clone(),
        center: center.clone(),
        ambient_ring,
        equations,
    })
}

fn map_polynomial<R: Field>(
    source: &Polynomial<R>,
    target: &PolynomialRing<R>,
    images: &[Polynomial<R>],
) -> Result<Polynomial<R>, GeometryError> {
    if images.len() != source.ring().variables().len()
        || images.iter().any(|image| image.ring() != target)
    {
        return Err(GeometryError::Invalid(
            "a polynomial coordinate map has incompatible arity or context".to_owned(),
        ));
    }
    let mut result = target.zero();
    for (exponent, coefficient) in source.terms() {
        let mut term = target.constant(coefficient.clone())?;
        for (power, image) in exponent.iter().zip(images) {
            if !power.is_zero() {
                term = target.mul(&term, &image.pow_integer(power)?)?;
            }
        }
        result = target.add(&result, &term)?;
    }
    Ok(result)
}

fn exceptional_order<R: Field>(
    equations: &[Polynomial<R>],
    variable: usize,
) -> Result<Integer, GeometryError> {
    equations
        .iter()
        .flat_map(Polynomial::terms)
        .map(|(exponent, _)| exponent[variable].clone())
        .min()
        .filter(|order| !order.is_zero())
        .ok_or_else(|| {
            GeometryError::Invalid(
                "the center does not contribute a positive exceptional order".to_owned(),
            )
        })
}

fn divide_exceptional_power<R: Field>(
    equation: &Polynomial<R>,
    variable: usize,
    order: &Integer,
) -> Result<Polynomial<R>, GeometryError> {
    equation
        .ring()
        .from_terms(equation.terms().map(|(exponent, coefficient)| {
            let mut quotient_exponent = exponent.to_vec();
            quotient_exponent[variable] = &quotient_exponent[variable] - order;
            (quotient_exponent, coefficient.clone())
        }))
        .map_err(Into::into)
}

fn polynomial_to_series<R: Field>(
    polynomial: &Polynomial<R>,
    ring: &PowerSeriesRing<R>,
) -> Result<PowerSeries<R>, GeometryError> {
    let mut terms = Vec::new();
    for (exponent, coefficient) in polynomial.terms() {
        let exponent = exponent
            .iter()
            .map(|power| {
                power.to_string().parse::<u64>().map_err(|_| {
                    GeometryError::Unsupported(
                        "a polynomial exponent does not fit the bounded series context".to_owned(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ring.truncation().contains_exponent(&exponent) {
            terms.push((exponent, coefficient.clone()));
        }
    }
    ring.from_terms(terms).map_err(Into::into)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteBlowUpChartWire {
    #[serde(rename = "type")]
    type_id: String,
    source: Scheme<FiniteField>,
    center: SchemePoint<FiniteField>,
    chart_index: usize,
    source_coordinate_map: Vec<Polynomial<FiniteField>>,
    total_transform: Scheme<FiniteField>,
    weak_transform: Scheme<FiniteField>,
    strict_transform: Scheme<FiniteField>,
    exceptional_divisor: Scheme<FiniteField>,
}

impl BlowUpChart<FiniteField> {
    /// Serialize this chart's transform and exceptional evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&FiniteBlowUpChartWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            center: self.center.clone(),
            chart_index: self.chart_index,
            source_coordinate_map: self.source_coordinate_map.clone(),
            total_transform: self.total_transform.clone(),
            weak_transform: self.weak_transform.clone(),
            strict_transform: self.strict_transform.clone(),
            exceptional_divisor: self.exceptional_divisor.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and recheck the chartwise saturation identities.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<FiniteBlowUpChartWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            source: wire.source,
            center: wire.center,
            chart_index: wire.chart_index,
            source_coordinate_map: wire.source_coordinate_map,
            total_transform: wire.total_transform,
            weak_transform: wire.weak_transform,
            strict_transform: wire.strict_transform,
            exceptional_divisor: wire.exceptional_divisor,
        };
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl SemanticType for BlowUpChart<FiniteField> {
    const TYPE_ID: &'static str = "noether/commalg/BlowUpChart@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &BLOW_UP_CHART_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteBlowUpWire {
    #[serde(rename = "type")]
    type_id: String,
    source: Scheme<FiniteField>,
    center: SchemePoint<FiniteField>,
    source_affine_chart: Option<usize>,
    local_source: Scheme<FiniteField>,
    charts: Vec<FiniteBlowUpChartWire>,
}

impl BlowUp<FiniteField> {
    /// Serialize every chart and its verification evidence.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&FiniteBlowUpWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            center: self.center.clone(),
            source_affine_chart: self.source_affine_chart,
            local_source: self.local_source.clone(),
            charts: self
                .charts
                .iter()
                .map(|chart| FiniteBlowUpChartWire {
                    type_id: BlowUpChart::<FiniteField>::TYPE_ID.to_owned(),
                    source: chart.source.clone(),
                    center: chart.center.clone(),
                    chart_index: chart.chart_index,
                    source_coordinate_map: chart.source_coordinate_map.clone(),
                    total_transform: chart.total_transform.clone(),
                    weak_transform: chart.weak_transform.clone(),
                    strict_transform: chart.strict_transform.clone(),
                    exceptional_divisor: chart.exceptional_divisor.clone(),
                })
                .collect(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and reconstruct the blow-up from its source and center.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<FiniteBlowUpWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        if wire
            .charts
            .iter()
            .any(|chart| chart.type_id != BlowUpChart::<FiniteField>::TYPE_ID)
        {
            return Err(NoetherError::Serialization(
                "a blow-up chart has the wrong type identity".to_owned(),
            ));
        }
        let result = Self {
            source: wire.source,
            center: wire.center,
            source_affine_chart: wire.source_affine_chart,
            local_source: wire.local_source,
            charts: wire
                .charts
                .into_iter()
                .map(|chart| BlowUpChart {
                    source: chart.source,
                    center: chart.center,
                    chart_index: chart.chart_index,
                    source_coordinate_map: chart.source_coordinate_map,
                    total_transform: chart.total_transform,
                    weak_transform: chart.weak_transform,
                    strict_transform: chart.strict_transform,
                    exceptional_divisor: chart.exceptional_divisor,
                })
                .collect(),
        };
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl SemanticType for BlowUp<FiniteField> {
    const TYPE_ID: &'static str = "noether/commalg/BlowUp@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &BLOW_UP_DESCRIPTION
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FiniteCompletionWire {
    #[serde(rename = "type")]
    type_id: String,
    source: Scheme<FiniteField>,
    center: SchemePoint<FiniteField>,
    ambient_ring: PowerSeriesRing<FiniteField>,
    equations: Vec<PowerSeries<FiniteField>>,
}

impl SchemeCompletion<FiniteField> {
    /// Serialize the local quotient presentation and fixed truncation.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&FiniteCompletionWire {
            type_id: Self::TYPE_ID.to_owned(),
            source: self.source.clone(),
            center: self.center.clone(),
            ambient_ring: self.ambient_ring.clone(),
            equations: self.equations.clone(),
        })
        .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Deserialize and recompute the translated truncated equations.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<FiniteCompletionWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let result = Self {
            source: wire.source,
            center: wire.center,
            ambient_ring: wire.ambient_ring,
            equations: wire.equations,
        };
        result
            .verify()
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        Ok(result)
    }
}

impl SemanticType for SchemeCompletion<FiniteField> {
    const TYPE_ID: &'static str = "noether/commalg/SchemeCompletion@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &SCHEME_COMPLETION_DESCRIPTION
    }
}
