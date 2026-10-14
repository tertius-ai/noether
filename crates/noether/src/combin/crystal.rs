// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Finite type-A Kashiwara crystals.

use core::fmt;
use std::collections::{BTreeMap, HashMap};

use graph_core::VertexId;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{CombinError, Partition};
use crate::{NoetherError, SemanticType, TypeDescription};

/// Largest Cartan rank accepted by the explicit 0.1 crystal representation.
pub const MAX_CRYSTAL_RANK: usize = 32;

/// Largest explicit vertex set accepted by the 0.1 crystal representation.
pub const MAX_CRYSTAL_VERTICES: usize = 200_000;

/// Largest tableau shape accepted by the recursive type-A builder.
pub const MAX_CRYSTAL_TABLEAU_CELLS: usize = 4_096;

const MAX_CRYSTAL_TABLEAU_ENTRIES: usize = 2_000_000;

/// A finite type-A Kashiwara crystal represented by its colored graph.
///
/// Vertices are stable [`VertexId`] values `0..vertex_count`. A zero-based
/// index `i` selects the partial raising and lowering operators `e_i` and
/// `f_i`. Weights are stored in fundamental-weight coordinates. The complete
/// Cartan matrix is defining context, so equal-rank but mathematically
/// different crystal families cannot be combined accidentally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KashiwaraCrystal {
    cartan_matrix: Vec<Vec<i64>>,
    weights: Vec<Vec<i64>>,
    f_edges: Vec<Vec<Option<VertexId>>>,
    e_edges: Vec<Vec<Option<VertexId>>>,
}

#[derive(Serialize, Deserialize)]
struct KashiwaraCrystalRepr {
    cartan_matrix: Vec<Vec<i64>>,
    weights: Vec<Vec<i64>>,
    f_edges: Vec<Vec<Option<VertexId>>>,
}

impl KashiwaraCrystal {
    /// Construct the standard letter crystal `B(Lambda_1)` of type `A_rank`.
    pub fn type_a_letters(rank: usize) -> Result<Self, CombinError> {
        let cartan_matrix = type_a_cartan_matrix(rank)?;
        let vertex_count = rank + 1;
        let mut weights = Vec::with_capacity(vertex_count);
        let mut f_edges = vec![vec![None; rank]; vertex_count];
        for letter in 0..vertex_count {
            let mut content = vec![0_i64; vertex_count];
            content[letter] = 1;
            weights.push(type_a_weight(&content));
            if letter < rank {
                f_edges[letter][letter] = Some(to_vertex_id(letter + 1)?);
            }
        }
        Self::from_parts(cartan_matrix, weights, f_edges)
    }

    /// Construct the semistandard-tableau crystal `B(shape)` of type `A_rank`.
    ///
    /// The alphabet is `1..=rank+1`, rows are weakly increasing, and columns
    /// are strictly increasing. The shape and exact representation-size bounds
    /// are checked before the tableaux are materialized.
    pub fn type_a_tableaux(rank: usize, shape: &Partition) -> Result<Self, CombinError> {
        let cartan_matrix = type_a_cartan_matrix(rank)?;
        let alphabet = rank + 1;
        if shape.parts().len() > alphabet {
            return Err(CombinError::CrystalShapeTooTall {
                rows: shape.parts().len(),
                alphabet,
            });
        }
        if shape.size() > MAX_CRYSTAL_TABLEAU_CELLS {
            return Err(CombinError::CrystalShapeTooLarge {
                cells: shape.size(),
                max: MAX_CRYSTAL_TABLEAU_CELLS,
            });
        }

        let vertex_count = type_a_tableau_count(rank, shape)?;
        let entry_limit = if shape.size() == 0 {
            MAX_CRYSTAL_VERTICES
        } else {
            MAX_CRYSTAL_TABLEAU_ENTRIES / shape.size()
        };
        let accepted_vertices = MAX_CRYSTAL_VERTICES.min(entry_limit.max(1));
        if vertex_count > accepted_vertices {
            return Err(CombinError::CrystalTooLarge {
                vertices: vertex_count,
                max: accepted_vertices,
            });
        }

        let tableaux = enumerate_ssyt(shape.parts(), alphabet, vertex_count);
        if tableaux.len() != vertex_count {
            return Err(invalid_crystal(format!(
                "tableau enumeration produced {} vertices after the dimension formula predicted {vertex_count}",
                tableaux.len()
            )));
        }
        let index = tableaux
            .iter()
            .enumerate()
            .map(|(position, tableau)| Ok((flatten(tableau), to_vertex_id(position)?)))
            .collect::<Result<HashMap<_, _>, CombinError>>()?;

        let weights = tableaux
            .iter()
            .map(|tableau| {
                let mut content = vec![0_i64; alphabet];
                for &entry in tableau.iter().flatten() {
                    content[entry - 1] += 1;
                }
                type_a_weight(&content)
            })
            .collect::<Vec<_>>();
        let mut f_edges = vec![vec![None; rank]; vertex_count];
        for (source, tableau) in tableaux.iter().enumerate() {
            let word = column_reading_word(tableau);
            for (crystal_index, edge) in f_edges[source].iter_mut().enumerate() {
                let Some(target) = apply_tableau_f(tableau, &word, crystal_index) else {
                    continue;
                };
                *edge = index.get(&flatten(&target)).copied();
                if edge.is_none() {
                    return Err(invalid_crystal(
                        "a tableau lowering operator left the enumerated crystal".to_owned(),
                    ));
                }
            }
        }
        Self::from_parts(cartan_matrix, weights, f_edges)
    }

    /// Cartan rank, equal to the number of Kashiwara colors.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.cartan_matrix.len()
    }

    /// Number of vertices in the explicit colored graph.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.weights.len()
    }

    /// Defining Cartan matrix for the crystal context.
    #[must_use]
    pub fn cartan_matrix(&self) -> &[Vec<i64>] {
        &self.cartan_matrix
    }

    /// Weight of a vertex in fundamental-weight coordinates.
    pub fn weight(&self, vertex: VertexId) -> Result<&[i64], CombinError> {
        let position = self.vertex_position(vertex)?;
        Ok(&self.weights[position])
    }

    /// Apply the Kashiwara raising operator `e_index`.
    pub fn e(&self, index: usize, vertex: VertexId) -> Result<Option<VertexId>, CombinError> {
        let position = self.checked_position(index, vertex)?;
        Ok(self.e_edges[position][index])
    }

    /// Apply the Kashiwara lowering operator `f_index`.
    pub fn f(&self, index: usize, vertex: VertexId) -> Result<Option<VertexId>, CombinError> {
        let position = self.checked_position(index, vertex)?;
        Ok(self.f_edges[position][index])
    }

    /// Return `(epsilon_index(vertex), phi_index(vertex))`, the upward and
    /// downward distances along one colored string.
    pub fn epsilon_phi(
        &self,
        index: usize,
        vertex: VertexId,
    ) -> Result<(usize, usize), CombinError> {
        let position = self.checked_position(index, vertex)?;
        Ok(self.string_lengths()[position][index])
    }

    /// Vertices killed by every raising operator.
    #[must_use]
    pub fn highest_weight_vertices(&self) -> Vec<VertexId> {
        self.e_edges
            .iter()
            .enumerate()
            .filter(|(_, row)| row.iter().all(Option::is_none))
            .filter_map(|(vertex, _)| VertexId::try_from(vertex).ok())
            .collect()
    }

    /// Weight-multiplicity character in deterministic weight order.
    #[must_use]
    pub fn character(&self) -> BTreeMap<Vec<i64>, u64> {
        let mut character = BTreeMap::new();
        for weight in &self.weights {
            *character.entry(weight.clone()).or_insert(0) += 1;
        }
        character
    }

    /// Form the right-to-left Kashiwara tensor product.
    ///
    /// Vertex `(left, right)` has identifier
    /// `left * rhs.vertex_count() + right`. The Cartan matrices must agree
    /// exactly, and the complete allocation is checked before materialization.
    pub fn tensor(&self, rhs: &Self) -> Result<Self, CombinError> {
        if self.cartan_matrix != rhs.cartan_matrix {
            return Err(CombinError::CrystalContextMismatch);
        }
        let vertex_count = self
            .vertex_count()
            .checked_mul(rhs.vertex_count())
            .ok_or(CombinError::CrystalSizeOverflow)?;
        if vertex_count > MAX_CRYSTAL_VERTICES {
            return Err(CombinError::CrystalTooLarge {
                vertices: vertex_count,
                max: MAX_CRYSTAL_VERTICES,
            });
        }

        let rank = self.rank();
        let right_count = rhs.vertex_count();
        let vertex_id = |left: usize, right: usize| -> Result<VertexId, CombinError> {
            let position = left
                .checked_mul(right_count)
                .and_then(|base| base.checked_add(right))
                .ok_or(CombinError::CrystalSizeOverflow)?;
            to_vertex_id(position)
        };
        let left_lengths = self.string_lengths();
        let right_lengths = rhs.string_lengths();
        let mut weights = Vec::with_capacity(vertex_count);
        let mut f_edges = vec![vec![None; rank]; vertex_count];
        for (left, left_string_lengths) in left_lengths.iter().enumerate() {
            for (right, right_string_lengths) in right_lengths.iter().enumerate() {
                let source = usize::try_from(vertex_id(left, right)?)
                    .map_err(|_| CombinError::CrystalSizeOverflow)?;
                weights.push(
                    self.weights[left]
                        .iter()
                        .zip(&rhs.weights[right])
                        .map(|(left, right)| {
                            left.checked_add(*right).ok_or_else(|| {
                                invalid_crystal("tensor-product weight overflowed i64".to_owned())
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
                for index in 0..rank {
                    let epsilon_left = left_string_lengths[index].0;
                    let phi_right = right_string_lengths[index].1;
                    f_edges[source][index] = if phi_right > epsilon_left {
                        rhs.f_edges[right][index]
                            .map(|target| vertex_id(left, target as usize))
                            .transpose()?
                    } else {
                        self.f_edges[left][index]
                            .map(|target| vertex_id(target as usize, right))
                            .transpose()?
                    };
                }
            }
        }
        Self::from_parts(self.cartan_matrix.clone(), weights, f_edges)
    }

    fn from_parts(
        cartan_matrix: Vec<Vec<i64>>,
        weights: Vec<Vec<i64>>,
        f_edges: Vec<Vec<Option<VertexId>>>,
    ) -> Result<Self, CombinError> {
        let rank = cartan_matrix.len();
        let expected = type_a_cartan_matrix(rank)?;
        if cartan_matrix != expected {
            return Err(invalid_crystal(
                "0.1 accepts only the standard finite type-A Cartan matrix".to_owned(),
            ));
        }
        let vertex_count = weights.len();
        if vertex_count == 0 {
            return Err(invalid_crystal(
                "a finite crystal must contain at least one vertex".to_owned(),
            ));
        }
        if vertex_count > MAX_CRYSTAL_VERTICES {
            return Err(CombinError::CrystalTooLarge {
                vertices: vertex_count,
                max: MAX_CRYSTAL_VERTICES,
            });
        }
        if weights.iter().any(|weight| weight.len() != rank) {
            return Err(invalid_crystal(
                "every weight vector must have Cartan-rank coordinates".to_owned(),
            ));
        }
        if f_edges.len() != vertex_count || f_edges.iter().any(|row| row.len() != rank) {
            return Err(invalid_crystal(
                "the lowering table must have vertex_count by rank entries".to_owned(),
            ));
        }

        let mut e_edges = vec![vec![None; rank]; vertex_count];
        for (source, row) in f_edges.iter().enumerate() {
            for (index, target) in row.iter().enumerate() {
                let Some(target) = target else { continue };
                let target_position =
                    usize::try_from(*target).map_err(|_| CombinError::CrystalSizeOverflow)?;
                if target_position >= vertex_count {
                    return Err(CombinError::CrystalVertexOutOfRange {
                        vertex: *target,
                        vertices: vertex_count,
                    });
                }
                if e_edges[target_position][index]
                    .replace(to_vertex_id(source)?)
                    .is_some()
                {
                    return Err(invalid_crystal(format!(
                        "two f_{index} edges enter vertex {target}"
                    )));
                }
                for coordinate in 0..rank {
                    let expected_weight = weights[source][coordinate]
                        .checked_sub(cartan_matrix[index][coordinate])
                        .ok_or_else(|| {
                            invalid_crystal("a crystal weight transition overflowed i64".to_owned())
                        })?;
                    if weights[target_position][coordinate] != expected_weight {
                        return Err(invalid_crystal(format!(
                            "f_{index} does not subtract the corresponding simple root"
                        )));
                    }
                }
            }
        }

        let crystal = Self {
            cartan_matrix,
            weights,
            f_edges,
            e_edges,
        };
        crystal.validate_strings()?;
        Ok(crystal)
    }

    #[allow(clippy::needless_range_loop)]
    fn validate_strings(&self) -> Result<(), CombinError> {
        let mut visited = vec![vec![false; self.rank()]; self.vertex_count()];
        for index in 0..self.rank() {
            for head in 0..self.vertex_count() {
                if self.e_edges[head][index].is_some() {
                    continue;
                }
                let mut string = Vec::new();
                let mut current = Some(to_vertex_id(head)?);
                while let Some(vertex) = current {
                    let position =
                        usize::try_from(vertex).map_err(|_| CombinError::CrystalSizeOverflow)?;
                    if visited[position][index] {
                        return Err(invalid_crystal(format!(
                            "the color-{index} graph contains a repeated vertex"
                        )));
                    }
                    visited[position][index] = true;
                    string.push(position);
                    current = self.f_edges[position][index];
                }
                for (epsilon, &vertex) in string.iter().enumerate() {
                    let phi = string.len() - epsilon - 1;
                    let pairing = i64::try_from(phi)
                        .and_then(|phi| i64::try_from(epsilon).map(|epsilon| phi - epsilon))
                        .map_err(|_| CombinError::CrystalSizeOverflow)?;
                    if self.weights[vertex][index] != pairing {
                        return Err(invalid_crystal(format!(
                            "phi_{index} - epsilon_{index} disagrees with vertex {vertex}'s weight"
                        )));
                    }
                }
            }
        }
        if visited.iter().any(|row| row.iter().any(|seen| !seen)) {
            return Err(invalid_crystal(
                "a colored Kashiwara string contains a cycle".to_owned(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn string_lengths(&self) -> Vec<Vec<(usize, usize)>> {
        let mut lengths = vec![vec![(0, 0); self.rank()]; self.vertex_count()];
        for index in 0..self.rank() {
            for head in 0..self.vertex_count() {
                if self.e_edges[head][index].is_some() {
                    continue;
                }
                let mut string = Vec::new();
                let mut current = Some(head);
                while let Some(vertex) = current {
                    string.push(vertex);
                    current = self.f_edges[vertex][index].map(|target| target as usize);
                }
                for (epsilon, vertex) in string.iter().copied().enumerate() {
                    lengths[vertex][index] = (epsilon, string.len() - epsilon - 1);
                }
            }
        }
        lengths
    }

    fn checked_position(&self, index: usize, vertex: VertexId) -> Result<usize, CombinError> {
        if index >= self.rank() {
            return Err(CombinError::CrystalIndexOutOfRange {
                index,
                rank: self.rank(),
            });
        }
        self.vertex_position(vertex)
    }

    fn vertex_position(&self, vertex: VertexId) -> Result<usize, CombinError> {
        let position = vertex as usize;
        if position >= self.vertex_count() {
            Err(CombinError::CrystalVertexOutOfRange {
                vertex,
                vertices: self.vertex_count(),
            })
        } else {
            Ok(position)
        }
    }
}

impl fmt::Display for KashiwaraCrystal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KashiwaraCrystal(type=A{}, vertices={})",
            self.rank(),
            self.vertex_count()
        )
    }
}

impl Serialize for KashiwaraCrystal {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        KashiwaraCrystalRepr {
            cartan_matrix: self.cartan_matrix.clone(),
            weights: self.weights.clone(),
            f_edges: self.f_edges.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for KashiwaraCrystal {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = KashiwaraCrystalRepr::deserialize(deserializer)?;
        Self::from_parts(repr.cartan_matrix, repr.weights, repr.f_edges)
            .map_err(serde::de::Error::custom)
    }
}

static KASHIWARA_CRYSTAL_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/combin/KashiwaraCrystal@1",
    rust_path: "noether::combin::KashiwaraCrystal",
    python_path: Some("noether.combin.KashiwaraCrystal"),
};

impl SemanticType for KashiwaraCrystal {
    const TYPE_ID: &'static str = "noether/combin/KashiwaraCrystal@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        serde_json::to_vec(self).map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &KASHIWARA_CRYSTAL_DESCRIPTION
    }
}

fn type_a_cartan_matrix(rank: usize) -> Result<Vec<Vec<i64>>, CombinError> {
    if !(1..=MAX_CRYSTAL_RANK).contains(&rank) {
        return Err(CombinError::InvalidCrystalRank {
            rank,
            max: MAX_CRYSTAL_RANK,
        });
    }
    let mut matrix = vec![vec![0_i64; rank]; rank];
    for index in 0..rank {
        matrix[index][index] = 2;
        if index > 0 {
            matrix[index][index - 1] = -1;
        }
        if index + 1 < rank {
            matrix[index][index + 1] = -1;
        }
    }
    Ok(matrix)
}

fn type_a_weight(content: &[i64]) -> Vec<i64> {
    content.windows(2).map(|pair| pair[0] - pair[1]).collect()
}

fn type_a_tableau_count(rank: usize, shape: &Partition) -> Result<usize, CombinError> {
    let alphabet = rank + 1;
    let mut parts = shape.parts().to_vec();
    parts.resize(alphabet, 0);
    let mut numerators = Vec::<u128>::new();
    let mut denominators = Vec::<u128>::new();
    for left in 0..alphabet {
        for right in left + 1..alphabet {
            numerators.push((parts[left] - parts[right]) as u128 + (right - left) as u128);
            denominators.push((right - left) as u128);
        }
    }
    for mut denominator in denominators {
        for numerator in &mut numerators {
            let divisor = gcd(*numerator, denominator);
            *numerator /= divisor;
            denominator /= divisor;
            if denominator == 1 {
                break;
            }
        }
        if denominator != 1 {
            return Err(invalid_crystal(
                "the type-A dimension formula did not simplify integrally".to_owned(),
            ));
        }
    }
    let mut count = 1_usize;
    for numerator in numerators {
        let factor = usize::try_from(numerator).map_err(|_| CombinError::CrystalTooLarge {
            vertices: usize::MAX,
            max: MAX_CRYSTAL_VERTICES,
        })?;
        count = count
            .checked_mul(factor)
            .ok_or(CombinError::CrystalTooLarge {
                vertices: usize::MAX,
                max: MAX_CRYSTAL_VERTICES,
            })?;
        if count > MAX_CRYSTAL_VERTICES {
            return Err(CombinError::CrystalTooLarge {
                vertices: count,
                max: MAX_CRYSTAL_VERTICES,
            });
        }
    }
    Ok(count)
}

fn gcd(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn enumerate_ssyt(shape: &[usize], alphabet: usize, expected: usize) -> Vec<Vec<Vec<usize>>> {
    let cells = shape
        .iter()
        .enumerate()
        .flat_map(|(row, &width)| (0..width).map(move |column| (row, column)))
        .collect::<Vec<_>>();
    let mut tableau = shape
        .iter()
        .map(|&width| vec![0_usize; width])
        .collect::<Vec<_>>();
    let mut output = Vec::with_capacity(expected);
    fill_ssyt(0, &cells, alphabet, &mut tableau, &mut output);
    output
}

fn fill_ssyt(
    position: usize,
    cells: &[(usize, usize)],
    alphabet: usize,
    tableau: &mut [Vec<usize>],
    output: &mut Vec<Vec<Vec<usize>>>,
) {
    if position == cells.len() {
        output.push(tableau.to_vec());
        return;
    }
    let (row, column) = cells[position];
    let row_minimum = if column == 0 {
        1
    } else {
        tableau[row][column - 1]
    };
    let column_minimum = if row == 0 {
        1
    } else {
        tableau[row - 1][column] + 1
    };
    for entry in row_minimum.max(column_minimum)..=alphabet {
        tableau[row][column] = entry;
        fill_ssyt(position + 1, cells, alphabet, tableau, output);
    }
    tableau[row][column] = 0;
}

fn flatten(tableau: &[Vec<usize>]) -> Vec<usize> {
    tableau.iter().flatten().copied().collect()
}

fn column_reading_word(tableau: &[Vec<usize>]) -> Vec<(usize, usize, usize)> {
    let width = tableau.iter().map(Vec::len).max().unwrap_or(0);
    let mut word = Vec::new();
    for column in 0..width {
        for row in (0..tableau.len()).rev() {
            if column < tableau[row].len() {
                word.push((row, column, tableau[row][column]));
            }
        }
    }
    word
}

fn apply_tableau_f(
    tableau: &[Vec<usize>],
    word: &[(usize, usize, usize)],
    index: usize,
) -> Option<Vec<Vec<usize>>> {
    let low = index + 1;
    let high = index + 2;
    let mut signature = Vec::<(usize, bool)>::new();
    for (position, &(_, _, entry)) in word.iter().enumerate() {
        if entry == high {
            signature.push((position, false));
        } else if entry == low {
            if signature.last().is_some_and(|(_, plus)| !plus) {
                signature.pop();
            } else {
                signature.push((position, true));
            }
        }
    }
    let position = signature
        .iter()
        .rev()
        .find(|(_, plus)| *plus)
        .map(|(position, _)| *position)?;
    let (row, column, _) = word[position];
    let mut output = tableau.to_vec();
    output[row][column] = high;
    Some(output)
}

fn to_vertex_id(position: usize) -> Result<VertexId, CombinError> {
    VertexId::try_from(position).map_err(|_| CombinError::CrystalSizeOverflow)
}

fn invalid_crystal(detail: String) -> CombinError {
    CombinError::InvalidCrystal { detail }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a2_letters_have_checked_weights_and_strings() {
        let crystal = KashiwaraCrystal::type_a_letters(2).unwrap();

        assert_eq!(crystal.vertex_count(), 3);
        assert_eq!(crystal.cartan_matrix(), &[vec![2, -1], vec![-1, 2]]);
        assert_eq!(crystal.weight(0).unwrap(), &[1, 0]);
        assert_eq!(crystal.f(0, 0).unwrap(), Some(1));
        assert_eq!(crystal.e(0, 1).unwrap(), Some(0));
        assert_eq!(crystal.highest_weight_vertices(), vec![0]);
    }

    #[test]
    fn a2_tableau_adjoint_has_the_expected_character() {
        let shape = Partition::new(vec![2, 1]).unwrap();
        let crystal = KashiwaraCrystal::type_a_tableaux(2, &shape).unwrap();

        assert_eq!(crystal.vertex_count(), 8);
        assert_eq!(crystal.highest_weight_vertices().len(), 1);
        assert_eq!(crystal.character().get(&vec![0, 0]), Some(&2));
        for vertex in 0..crystal.vertex_count() as VertexId {
            for index in 0..crystal.rank() {
                let (epsilon, phi) = crystal.epsilon_phi(index, vertex).unwrap();
                assert_eq!(
                    phi as i64 - epsilon as i64,
                    crystal.weight(vertex).unwrap()[index]
                );
            }
        }
    }

    #[test]
    fn tensor_letters_decomposes_three_times_three() {
        let letters = KashiwaraCrystal::type_a_letters(2).unwrap();
        let tensor = letters.tensor(&letters).unwrap();
        let mut highest_weights = tensor
            .highest_weight_vertices()
            .into_iter()
            .map(|vertex| tensor.weight(vertex).unwrap().to_vec())
            .collect::<Vec<_>>();
        highest_weights.sort();

        assert_eq!(tensor.vertex_count(), 9);
        assert_eq!(highest_weights, vec![vec![0, 1], vec![2, 0]]);
    }

    #[test]
    fn serde_revalidates_the_colored_graph() {
        let crystal = KashiwaraCrystal::type_a_letters(2).unwrap();
        let json = serde_json::to_string(&crystal).unwrap();
        assert_eq!(
            serde_json::from_str::<KashiwaraCrystal>(&json).unwrap(),
            crystal
        );

        let malformed = json.replace("[1,null]", "[0,null]");
        assert!(serde_json::from_str::<KashiwaraCrystal>(&malformed).is_err());
    }

    #[test]
    fn bounds_fail_closed() {
        assert!(matches!(
            KashiwaraCrystal::type_a_letters(0),
            Err(CombinError::InvalidCrystalRank { .. })
        ));
        let crystal = KashiwaraCrystal::type_a_letters(2).unwrap();
        assert!(matches!(
            crystal.f(2, 0),
            Err(CombinError::CrystalIndexOutOfRange { .. })
        ));
        assert!(matches!(
            crystal.weight(3),
            Err(CombinError::CrystalVertexOutOfRange { .. })
        ));
    }
}
