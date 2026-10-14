// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use core::fmt;
use std::collections::BTreeMap;

use crate::{Field, NoetherError, Ring, SemanticType, TypeDescription};

use super::{Simplex, SimplicialComplex, TopologyError};

static FILTERED_COMPLEX_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/FilteredComplex@1",
    rust_path: "noether::topology::FilteredComplex",
    python_path: Some("noether.topology.FilteredComplex"),
};

static BARCODE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/topology/Barcode@1",
    rust_path: "noether::topology::Barcode",
    python_path: Some("noether.topology.Barcode"),
};

/// A finite simplicial complex equipped with a finite real-valued filtration.
///
/// Every nonempty simplex is stored once. Construction orders cells by
/// `(value, dimension, vertices)` and verifies that every face is present no
/// later than its coface. Filtration values are finite `f64` values: persistent
/// homology is stable under perturbation of those values, while `NaN` and
/// infinity do not define a sortable finite filtration and are rejected.
#[derive(Clone, PartialEq)]
pub struct FilteredComplex {
    simplices: Vec<Simplex>,
    values: Vec<f64>,
    index: BTreeMap<Simplex, usize>,
}

impl FilteredComplex {
    /// Construct and verify a simplex-wise filtration.
    pub fn new(mut entries: Vec<(Simplex, f64)>) -> Result<Self, TopologyError> {
        for (position, (simplex, value)) in entries.iter().enumerate() {
            if simplex.vertices().is_empty() {
                return Err(TopologyError::InvalidFiltration {
                    reason: format!("entry {position} is the empty simplex"),
                });
            }
            if !value.is_finite() {
                return Err(TopologyError::InvalidFiltration {
                    reason: format!(
                        "entry {position} for {:?} has non-finite value {value}",
                        simplex.vertices()
                    ),
                });
            }
        }

        entries.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| a.0.dimension().cmp(&b.0.dimension()))
                .then_with(|| a.0.cmp(&b.0))
        });
        let simplices: Vec<Simplex> = entries.iter().map(|entry| entry.0.clone()).collect();
        let values: Vec<f64> = entries.iter().map(|entry| entry.1).collect();
        let mut index = BTreeMap::new();
        for (position, simplex) in simplices.iter().enumerate() {
            if let Some(previous) = index.insert(simplex.clone(), position) {
                return Err(TopologyError::InvalidFiltration {
                    reason: format!(
                        "simplex {:?} occurs more than once (positions {previous} and {position})",
                        simplex.vertices()
                    ),
                });
            }
        }

        for (position, simplex) in simplices.iter().enumerate() {
            for face in simplex.faces() {
                let Some(&face_position) = index.get(&face) else {
                    return Err(TopologyError::InvalidFiltration {
                        reason: format!(
                            "simplex {:?} is present but its face {:?} is missing",
                            simplex.vertices(),
                            face.vertices()
                        ),
                    });
                };
                if values[face_position] > values[position] {
                    return Err(TopologyError::InvalidFiltration {
                        reason: format!(
                            "simplex {:?} enters at {} before its face {:?} enters at {}",
                            simplex.vertices(),
                            values[position],
                            face.vertices(),
                            values[face_position]
                        ),
                    });
                }
            }
        }

        Ok(Self {
            simplices,
            values,
            index,
        })
    }

    /// Give every simplex of `complex` the same finite filtration value.
    pub fn constant(complex: &SimplicialComplex, value: f64) -> Result<Self, TopologyError> {
        let entries = complex
            .top_dimension()
            .into_iter()
            .flat_map(|top| 0..=top)
            .flat_map(|dimension| {
                complex
                    .simplices(dimension)
                    .iter()
                    .cloned()
                    .map(move |simplex| (simplex, value))
            })
            .collect();
        Self::new(entries)
    }

    /// Whether no nonempty simplex is present.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// Number of stored nonempty simplices.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Largest simplex dimension, or `-1` for the empty filtered complex.
    pub fn dimension(&self) -> i64 {
        self.top_dimension().map_or(-1, |dimension| {
            i64::try_from(dimension).expect("filtered-complex dimensions fit i64")
        })
    }

    fn top_dimension(&self) -> Option<usize> {
        self.simplices.iter().map(Simplex::dimension).max()
    }

    /// Simplices in canonical filtration order.
    pub fn simplices(&self) -> &[Simplex] {
        &self.simplices
    }

    /// Finite filtration values aligned with [`simplices`](Self::simplices).
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Position of `simplex` in canonical filtration order.
    pub fn position(&self, simplex: &Simplex) -> Option<usize> {
        self.index.get(simplex).copied()
    }

    /// The sublevel simplicial complex containing cells with value at most `scale`.
    pub fn slice_at(&self, scale: f64) -> Result<SimplicialComplex, TopologyError> {
        if !scale.is_finite() {
            return Err(TopologyError::InvalidFiltration {
                reason: format!("slice scale must be finite; received {scale}"),
            });
        }
        let facets = self
            .simplices
            .iter()
            .zip(&self.values)
            .filter(|(_, value)| **value <= scale)
            .map(|(simplex, _)| simplex.vertices().to_vec())
            .collect::<Vec<_>>();
        Ok(SimplicialComplex::from_facets(&facets))
    }

    /// Compute persistent homology over an exact field.
    ///
    /// This generic reduction is the correctness floor. Implementations may
    /// dispatch to a packed or sparse reducer for a particular field and
    /// workload, but must return the same checked [`Barcode`].
    pub fn persistent_homology<R: Field>(&self, field: R) -> Result<Barcode<R>, TopologyError> {
        let pairs = reduce(self.boundary_columns(&field)?, &field)?;
        Barcode::new(field, bars_from_pairs(self, &pairs))
    }

    fn boundary_columns<R: Ring>(&self, ring: &R) -> Result<Vec<Column<R>>, TopologyError> {
        let positive = ring.one();
        let negative = ring.neg(&positive)?;
        self.simplices
            .iter()
            .map(|simplex| {
                let mut column = simplex
                    .faces()
                    .iter()
                    .enumerate()
                    .filter_map(|(omitted, face)| {
                        self.position(face).map(|row| {
                            (
                                row as u32,
                                if omitted % 2 == 0 {
                                    positive.clone()
                                } else {
                                    negative.clone()
                                },
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                column.sort_unstable_by_key(|entry| entry.0);
                Ok(column)
            })
            .collect()
    }
}

impl fmt::Debug for FilteredComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilteredComplex")
            .field("dimension", &self.dimension())
            .field("cells", &self.len())
            .field("value_range", &self.values.first().zip(self.values.last()))
            .finish()
    }
}

impl fmt::Display for FilteredComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FilteredComplex(dimension={}, cells={})",
            self.dimension(),
            self.len()
        )
    }
}

impl SemanticType for FilteredComplex {
    const TYPE_ID: &'static str = "noether/topology/FilteredComplex@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = Vec::new();
        output.extend_from_slice(&(self.simplices.len() as u64).to_be_bytes());
        for (simplex, value) in self.simplices.iter().zip(&self.values) {
            output.extend_from_slice(&(simplex.vertices().len() as u64).to_be_bytes());
            for vertex in simplex.vertices() {
                output.extend_from_slice(&vertex.to_be_bytes());
            }
            output.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &FILTERED_COMPLEX_DESCRIPTION
    }
}

/// Persistent-homology intervals together with their exact coefficient field.
///
/// Each bar is `(dimension, birth, death)`, with `None` death for an essential
/// class. Equality includes the field context: equal intervals over `F_2` and
/// `F_3` answer different mathematical questions and are not equal barcodes.
#[derive(Clone, PartialEq)]
pub struct Barcode<R: Field> {
    field: R,
    bars: Vec<(usize, f64, Option<f64>)>,
}

impl<R: Field> Barcode<R> {
    /// Construct a barcode after validating and canonically ordering its bars.
    pub fn new(field: R, mut bars: Vec<(usize, f64, Option<f64>)>) -> Result<Self, TopologyError> {
        for (position, (dimension, birth, death)) in bars.iter().enumerate() {
            if !birth.is_finite() {
                return Err(TopologyError::InvalidBarcode {
                    reason: format!("bar {position} in dimension {dimension} has non-finite birth"),
                });
            }
            if let Some(death) = death {
                if !death.is_finite() {
                    return Err(TopologyError::InvalidBarcode {
                        reason: format!(
                            "bar {position} in dimension {dimension} has non-finite death; use None for an essential class"
                        ),
                    });
                }
                if death < birth {
                    return Err(TopologyError::InvalidBarcode {
                        reason: format!(
                            "bar {position} in dimension {dimension} dies at {death} before birth at {birth}"
                        ),
                    });
                }
            }
        }
        bars.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.total_cmp(&b.1))
                .then_with(|| match (a.2, b.2) {
                    (None, None) => core::cmp::Ordering::Equal,
                    (None, Some(_)) => core::cmp::Ordering::Greater,
                    (Some(_), None) => core::cmp::Ordering::Less,
                    (Some(a), Some(b)) => a.total_cmp(&b),
                })
        });
        Ok(Self { field, bars })
    }

    /// Coefficient field used for this computation.
    pub fn field(&self) -> &R {
        &self.field
    }

    /// Canonically ordered `(dimension, birth, death)` intervals.
    pub fn bars(&self) -> &[(usize, f64, Option<f64>)] {
        &self.bars
    }

    /// Number of nonzero-length or essential intervals.
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// Whether there are no intervals.
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// Dimensions carrying at least one interval, in increasing order.
    pub fn dimensions(&self) -> Vec<usize> {
        let mut dimensions = self
            .bars
            .iter()
            .map(|(dimension, _, _)| *dimension)
            .collect::<Vec<_>>();
        dimensions.dedup();
        dimensions
    }

    /// Intervals in one homological dimension, retaining the field context.
    pub fn in_dimension(&self, dimension: usize) -> Self {
        Self {
            field: self.field.clone(),
            bars: self
                .bars
                .iter()
                .filter(|(candidate, _, _)| *candidate == dimension)
                .copied()
                .collect(),
        }
    }

    /// Essential intervals, whose death is `None`.
    pub fn essential(&self) -> Vec<(usize, f64, Option<f64>)> {
        self.bars
            .iter()
            .filter(|(_, _, death)| death.is_none())
            .copied()
            .collect()
    }

    /// Betti numbers at one finite filtration value.
    pub fn betti_at(&self, value: f64) -> Result<Vec<usize>, TopologyError> {
        if !value.is_finite() {
            return Err(TopologyError::InvalidBarcode {
                reason: format!("Betti evaluation value must be finite; received {value}"),
            });
        }
        let top = self.dimensions().last().copied().unwrap_or(0);
        Ok((0..=top)
            .map(|dimension| {
                self.bars
                    .iter()
                    .filter(|(candidate, birth, death)| {
                        *candidate == dimension
                            && *birth <= value
                            && death.is_none_or(|death| death > value)
                    })
                    .count()
            })
            .collect())
    }

    /// Interval lengths, with `None` for essential classes.
    pub fn lifetimes(&self) -> Vec<Option<f64>> {
        self.bars
            .iter()
            .map(|(_, birth, death)| death.map(|death| death - birth))
            .collect()
    }

    /// Bottleneck distance to `other`, respecting homological dimensions.
    ///
    /// Intervals may match the diagonal but never an interval in another
    /// dimension. Different coefficient contexts are rejected: comparing
    /// intervals over `F_2` with intervals over `F_3` would compare answers to
    /// different mathematical questions.
    pub fn bottleneck_distance(&self, other: &Self) -> Result<f64, TopologyError> {
        self.require_same_field(other)?;
        Ok(all_dimensions(self, other)
            .into_iter()
            .map(|dimension| {
                bottleneck(
                    &diagram_in_dimension(self, dimension),
                    &diagram_in_dimension(other, dimension),
                )
            })
            .fold(0.0, f64::max))
    }

    /// `power`-Wasserstein distance to `other`, respecting dimensions.
    ///
    /// `power` must be finite and at least one. Costs from separate dimensions
    /// are combined by the same `L^power` norm used within each diagram.
    pub fn wasserstein_distance(&self, other: &Self, power: f64) -> Result<f64, TopologyError> {
        self.require_same_field(other)?;
        if !power.is_finite() || power < 1.0 {
            return Err(TopologyError::InvalidBarcode {
                reason: format!(
                    "Wasserstein power must be finite and at least one; received {power}"
                ),
            });
        }
        let powered_cost = all_dimensions(self, other)
            .into_iter()
            .map(|dimension| {
                wasserstein_powered(
                    &diagram_in_dimension(self, dimension),
                    &diagram_in_dimension(other, dimension),
                    power,
                )
            })
            .sum::<f64>();
        Ok(powered_cost.powf(1.0 / power))
    }

    fn require_same_field(&self, other: &Self) -> Result<(), TopologyError> {
        if self.field == other.field {
            return Ok(());
        }
        Err(crate::ContextMismatchError {
            expected: format!("{} coefficient context", R::TYPE_ID),
            actual: format!("a different {} coefficient context", R::TYPE_ID),
        }
        .into())
    }
}

impl<R: Field> fmt::Debug for Barcode<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Barcode")
            .field("field", &R::TYPE_ID)
            .field("bars", &self.bars)
            .finish()
    }
}

impl<R: Field> fmt::Display for Barcode<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Barcode(field={}, bars={}, dimensions={:?})",
            R::TYPE_ID,
            self.len(),
            self.dimensions()
        )
    }
}

impl<R: Field> SemanticType for Barcode<R> {
    const TYPE_ID: &'static str = "noether/topology/Barcode@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let field = self.field.canonical_bytes()?;
        let mut output = Vec::new();
        output.extend_from_slice(&(field.len() as u64).to_be_bytes());
        output.extend_from_slice(&field);
        output.extend_from_slice(&(self.bars.len() as u64).to_be_bytes());
        for (dimension, birth, death) in &self.bars {
            output.extend_from_slice(&(*dimension as u64).to_be_bytes());
            output.extend_from_slice(&birth.to_bits().to_be_bytes());
            match death {
                None => output.push(0),
                Some(death) => {
                    output.push(1);
                    output.extend_from_slice(&death.to_bits().to_be_bytes());
                }
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &BARCODE_DESCRIPTION
    }
}

type Column<R> = Vec<(u32, <R as Ring>::Element)>;

fn reduce<R: Field>(
    mut columns: Vec<Column<R>>,
    field: &R,
) -> Result<Vec<Option<u32>>, TopologyError> {
    let mut owner = BTreeMap::<u32, usize>::new();
    let mut pairs = vec![None; columns.len()];
    for column in 0..columns.len() {
        loop {
            let Some((row, coefficient)) = columns[column].last().cloned() else {
                break;
            };
            let Some(&earlier) = owner.get(&row) else {
                owner.insert(row, column);
                pairs[column] = Some(row);
                break;
            };
            let pivot = columns[earlier]
                .last()
                .expect("an owner column has a nonzero pivot")
                .1
                .clone();
            let inverse = field.inverse(&pivot)?;
            let quotient = field.mul(&coefficient, &inverse)?;
            let scale = field.neg(&quotient)?;
            let source = columns[earlier].clone();
            axpy(&mut columns[column], &source, &scale, field)?;
        }
    }
    Ok(pairs)
}

fn axpy<R: Field>(
    target: &mut Column<R>,
    source: &Column<R>,
    scale: &R::Element,
    field: &R,
) -> Result<(), TopologyError> {
    let mut output = Vec::with_capacity(target.len() + source.len());
    let zero = field.zero();
    let (mut left, mut right) = (0, 0);
    while left < target.len() || right < source.len() {
        match (target.get(left), source.get(right)) {
            (Some(a), Some(b)) if a.0 == b.0 => {
                let scaled = field.mul(scale, &b.1)?;
                let value = field.add(&a.1, &scaled)?;
                if value != zero {
                    output.push((a.0, value));
                }
                left += 1;
                right += 1;
            }
            (Some(a), Some(b)) if a.0 < b.0 => {
                output.push(a.clone());
                left += 1;
            }
            (_, Some(b)) => {
                let value = field.mul(scale, &b.1)?;
                if value != zero {
                    output.push((b.0, value));
                }
                right += 1;
            }
            (Some(a), None) => {
                output.push(a.clone());
                left += 1;
            }
            (None, None) => break,
        }
    }
    *target = output;
    Ok(())
}

fn bars_from_pairs(
    filtration: &FilteredComplex,
    pairs: &[Option<u32>],
) -> Vec<(usize, f64, Option<f64>)> {
    let mut killed = vec![false; filtration.len()];
    let mut bars = Vec::new();
    for (death_position, birth_position) in pairs.iter().enumerate() {
        if let Some(birth_position) = birth_position {
            let birth_position = *birth_position as usize;
            killed[birth_position] = true;
            let birth = filtration.values[birth_position];
            let death = filtration.values[death_position];
            if death > birth {
                bars.push((
                    filtration.simplices[birth_position].dimension(),
                    birth,
                    Some(death),
                ));
            }
        }
    }
    for (position, was_killed) in killed.into_iter().enumerate() {
        if !was_killed && pairs[position].is_none() {
            bars.push((
                filtration.simplices[position].dimension(),
                filtration.values[position],
                None,
            ));
        }
    }
    bars
}

type DiagramPoint = (f64, f64);

fn all_dimensions<R: Field>(left: &Barcode<R>, right: &Barcode<R>) -> Vec<usize> {
    let mut dimensions = left.dimensions();
    dimensions.extend(right.dimensions());
    dimensions.sort_unstable();
    dimensions.dedup();
    dimensions
}

fn diagram_in_dimension<R: Field>(barcode: &Barcode<R>, dimension: usize) -> Vec<DiagramPoint> {
    barcode
        .bars
        .iter()
        .filter(|(candidate, _, _)| *candidate == dimension)
        .map(|(_, birth, death)| (*birth, death.unwrap_or(f64::INFINITY)))
        .collect()
}

fn linf(left: DiagramPoint, right: DiagramPoint) -> f64 {
    (left.0 - right.0).abs().max((left.1 - right.1).abs())
}

fn diagonal_cost(point: DiagramPoint) -> f64 {
    (point.1 - point.0).abs() / 2.0
}

fn split_diagram(diagram: &[DiagramPoint]) -> (Vec<DiagramPoint>, Vec<f64>) {
    let mut finite = Vec::new();
    let mut essential = Vec::new();
    for &(birth, death) in diagram {
        if death.is_finite() {
            finite.push((birth, death));
        } else {
            essential.push(birth);
        }
    }
    essential.sort_by(f64::total_cmp);
    (finite, essential)
}

fn cost_matrix(left: &[DiagramPoint], right: &[DiagramPoint]) -> Vec<Vec<f64>> {
    let (left_len, right_len) = (left.len(), right.len());
    let size = left_len + right_len;
    (0..size)
        .map(|row| {
            (0..size)
                .map(|column| match (row < left_len, column < right_len) {
                    (true, true) => linf(left[row], right[column]),
                    (true, false) => diagonal_cost(left[row]),
                    (false, true) => diagonal_cost(right[column]),
                    (false, false) => 0.0,
                })
                .collect()
        })
        .collect()
}

fn has_perfect_matching(costs: &[Vec<f64>], maximum: f64) -> bool {
    let size = costs.len();
    let adjacency = (0..size)
        .map(|row| {
            (0..size)
                .filter(|&column| costs[row][column] <= maximum)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut right_owner = vec![None; size];

    fn augment(
        row: usize,
        adjacency: &[Vec<usize>],
        seen: &mut [bool],
        right_owner: &mut [Option<usize>],
    ) -> bool {
        for &column in &adjacency[row] {
            if seen[column] {
                continue;
            }
            seen[column] = true;
            if right_owner[column].is_none_or(|owner| augment(owner, adjacency, seen, right_owner))
            {
                right_owner[column] = Some(row);
                return true;
            }
        }
        false
    }

    (0..size).all(|row| augment(row, &adjacency, &mut vec![false; size], &mut right_owner))
}

fn bottleneck(left: &[DiagramPoint], right: &[DiagramPoint]) -> f64 {
    let (left_finite, left_essential) = split_diagram(left);
    let (right_finite, right_essential) = split_diagram(right);
    if left_essential.len() != right_essential.len() {
        return f64::INFINITY;
    }
    let essential_cost = left_essential
        .iter()
        .zip(&right_essential)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    if left_finite.is_empty() && right_finite.is_empty() {
        return essential_cost;
    }
    let costs = cost_matrix(&left_finite, &right_finite);
    let mut candidates = costs
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();
    let (mut lower, mut upper) = (0, candidates.len() - 1);
    while lower < upper {
        let middle = (lower + upper) / 2;
        if has_perfect_matching(&costs, candidates[middle]) {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    candidates[lower].max(essential_cost)
}

fn hungarian(costs: &[Vec<f64>]) -> f64 {
    let size = costs.len();
    if size == 0 {
        return 0.0;
    }
    let mut left_potential = vec![0.0; size + 1];
    let mut right_potential = vec![0.0; size + 1];
    let mut assignment = vec![0; size + 1];
    let mut predecessor = vec![0; size + 1];

    for row in 1..=size {
        assignment[0] = row;
        let mut column = 0;
        let mut minimum = vec![f64::INFINITY; size + 1];
        let mut used = vec![false; size + 1];
        loop {
            used[column] = true;
            let assigned_row = assignment[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0;
            for candidate in 1..=size {
                if !used[candidate] {
                    let reduced = costs[assigned_row - 1][candidate - 1]
                        - left_potential[assigned_row]
                        - right_potential[candidate];
                    if reduced < minimum[candidate] {
                        minimum[candidate] = reduced;
                        predecessor[candidate] = column;
                    }
                    if minimum[candidate] < delta {
                        delta = minimum[candidate];
                        next_column = candidate;
                    }
                }
            }
            for candidate in 0..=size {
                if used[candidate] {
                    left_potential[assignment[candidate]] += delta;
                    right_potential[candidate] -= delta;
                } else {
                    minimum[candidate] -= delta;
                }
            }
            column = next_column;
            if assignment[column] == 0 {
                break;
            }
        }
        loop {
            let previous = predecessor[column];
            assignment[column] = assignment[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    (1..=size)
        .map(|column| costs[assignment[column] - 1][column - 1])
        .sum()
}

fn wasserstein_powered(left: &[DiagramPoint], right: &[DiagramPoint], power: f64) -> f64 {
    let (left_finite, left_essential) = split_diagram(left);
    let (right_finite, right_essential) = split_diagram(right);
    if left_essential.len() != right_essential.len() {
        return f64::INFINITY;
    }
    let essential = left_essential
        .iter()
        .zip(&right_essential)
        .map(|(left, right)| (left - right).abs().powf(power))
        .sum::<f64>();
    let powered = cost_matrix(&left_finite, &right_finite)
        .into_iter()
        .map(|row| row.into_iter().map(|cost| cost.powf(power)).collect())
        .collect::<Vec<Vec<_>>>();
    hungarian(&powered) + essential
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numthy::FiniteField;
    use crate::{SemanticType, Q};

    fn s(vertices: &[u32]) -> Simplex {
        Simplex::new(vertices.to_vec())
    }

    fn filling_circle() -> FilteredComplex {
        FilteredComplex::new(vec![
            (s(&[0]), 0.0),
            (s(&[1]), 0.0),
            (s(&[2]), 0.0),
            (s(&[0, 1]), 1.0),
            (s(&[0, 2]), 1.0),
            (s(&[1, 2]), 2.0),
            (s(&[0, 1, 2]), 3.0),
        ])
        .unwrap()
    }

    fn rp2() -> SimplicialComplex {
        SimplicialComplex::from_facets(&[
            vec![0, 1, 2],
            vec![0, 1, 4],
            vec![0, 2, 3],
            vec![0, 3, 5],
            vec![0, 4, 5],
            vec![1, 2, 5],
            vec![1, 3, 4],
            vec![1, 3, 5],
            vec![2, 3, 4],
            vec![2, 4, 5],
        ])
    }

    #[test]
    fn invalid_filtrations_fail_closed() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(FilteredComplex::new(vec![(s(&[0]), value)]).is_err());
        }
        assert!(FilteredComplex::new(vec![(s(&[0]), 0.0), (s(&[0]), 1.0)]).is_err());
        assert!(FilteredComplex::new(vec![(s(&[0]), 0.0), (s(&[0, 1]), 1.0)]).is_err());
        assert!(
            FilteredComplex::new(vec![(s(&[0]), 2.0), (s(&[1]), 0.0), (s(&[0, 1]), 1.0),]).is_err()
        );
    }

    #[test]
    fn empty_and_zero_dimensional_filtrations_have_distinct_dimensions() {
        let empty = FilteredComplex::new(Vec::new()).unwrap();
        let point = FilteredComplex::new(vec![(s(&[0]), 0.0)]).unwrap();

        assert!(empty.is_empty());
        assert_eq!(empty.dimension(), -1);
        assert!(!point.is_empty());
        assert_eq!(point.dimension(), 0);
    }

    #[test]
    fn a_circle_fills_at_the_declared_time_over_several_fields() {
        let filtration = filling_circle();
        let f2 = FiniteField::prime(2).unwrap();
        let f9 = FiniteField::extension(3, 2).unwrap();
        for bars in [
            filtration.persistent_homology(Q).unwrap().bars().to_vec(),
            filtration.persistent_homology(f2).unwrap().bars().to_vec(),
            filtration.persistent_homology(f9).unwrap().bars().to_vec(),
        ] {
            assert!(bars.contains(&(1, 2.0, Some(3.0))));
            assert_eq!(
                bars.iter()
                    .filter(|(dimension, _, death)| *dimension == 0 && death.is_none())
                    .count(),
                1
            );
        }
    }

    #[test]
    fn constant_rp2_distinguishes_f2_and_q() {
        let filtration = FilteredComplex::constant(&rp2(), 0.0).unwrap();
        let f2 = filtration
            .persistent_homology(FiniteField::prime(2).unwrap())
            .unwrap();
        let rational = filtration.persistent_homology(Q).unwrap();
        assert_eq!(f2.betti_at(0.0).unwrap(), vec![1, 1, 1]);
        assert_eq!(rational.betti_at(0.0).unwrap(), vec![1]);
    }

    #[test]
    fn field_context_is_part_of_barcode_identity() {
        let bars = vec![(0, 0.0, None)];
        let f2 = Barcode::new(FiniteField::prime(2).unwrap(), bars.clone()).unwrap();
        let f3 = Barcode::new(FiniteField::prime(3).unwrap(), bars).unwrap();
        assert_ne!(f2, f3);
        assert_ne!(f2.canonical_bytes().unwrap(), f3.canonical_bytes().unwrap());
        assert_eq!(
            Barcode::<FiniteField>::TYPE_ID,
            "noether/topology/Barcode@1"
        );
    }

    #[test]
    fn slicing_is_checked_and_preserves_isolated_vertices() {
        let filtration = filling_circle();
        let early = filtration.slice_at(0.0).unwrap();
        assert_eq!(early.count(0), 3);
        assert_eq!(early.count(1), 0);
        assert!(filtration.slice_at(f64::NAN).is_err());
    }

    #[test]
    fn barcode_validation_rejects_false_intervals() {
        assert!(Barcode::new(Q, vec![(0, f64::NAN, None)]).is_err());
        assert!(Barcode::new(Q, vec![(0, 1.0, Some(0.0))]).is_err());
        assert!(Barcode::new(Q, vec![(0, 0.0, Some(f64::INFINITY))]).is_err());
    }

    #[test]
    fn barcode_distances_respect_dimension_and_field() {
        let left = Barcode::new(
            FiniteField::prime(2).unwrap(),
            vec![(0, 0.0, None), (1, 0.0, Some(1.0)), (2, 0.0, Some(2.0))],
        )
        .unwrap();
        let right = Barcode::new(
            FiniteField::prime(2).unwrap(),
            vec![(0, 0.25, None), (1, 0.0, Some(1.2))],
        )
        .unwrap();
        assert!((left.bottleneck_distance(&right).unwrap() - 1.0).abs() < 1e-12);
        let expected = (0.25f64.powi(2) + 0.2f64.powi(2) + 1.0).sqrt();
        assert!((left.wasserstein_distance(&right, 2.0).unwrap() - expected).abs() < 1e-12);

        let wrong_field = Barcode::new(FiniteField::prime(3).unwrap(), vec![]).unwrap();
        assert!(left.bottleneck_distance(&wrong_field).is_err());
        assert!(left.wasserstein_distance(&right, 0.5).is_err());
    }
}
