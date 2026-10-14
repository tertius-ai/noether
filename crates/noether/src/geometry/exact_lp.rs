// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Small exact-rational simplex kernel used by mathematical playbooks.
//!
//! This is deliberately crate-private: public operations own their mathematical
//! result and evidence types instead of exposing an implementation-planning API.

use crate::Rational;

/// Check a proposed dual as an exact nonnegative upper-bound certificate.
pub(crate) fn certified_upper_bound_from_dual_free(
    a: &[Vec<Rational>],
    b: &[Rational],
    c: &[Rational],
    upper_bound: &Rational,
    dual: &[Rational],
) -> Option<Vec<Rational>> {
    for multipliers in [dual.to_vec(), dual.iter().map(|value| -value).collect()] {
        if multipliers.iter().any(Rational::is_negative) {
            continue;
        }
        let columns = (0..c.len())
            .map(|column| {
                a.iter()
                    .zip(&multipliers)
                    .fold(Rational::zero(), |sum, (row, multiplier)| {
                        &sum + &(&row[column] * multiplier)
                    })
            })
            .collect::<Vec<_>>();
        let bound = b
            .iter()
            .zip(&multipliers)
            .fold(Rational::zero(), |sum, (rhs, multiplier)| {
                &sum + &(rhs * multiplier)
            });
        if columns == c && bound <= *upper_bound {
            return Some(multipliers);
        }
    }
    None
}

/// A reusable native proposal engine for `max c*x` subject to `A*x <= b`.
///
/// The floating phase solves the dual `min b*y`, `A^T*y = c`, `y >= 0` with
/// revised simplex.  Its basis has the number of primal variables rather than
/// the potentially much larger number of inequalities.  A successful basis
/// is mapped by exact constraint identity into the next cutting-plane LP and
/// warm-started after new cuts arrive.
///
/// Floating values are proposals only.  The selected basis is reconstructed
/// over [`Rational`], and both primal feasibility and dual strong equality are
/// checked exactly before a solution is returned.
#[derive(Default)]
pub(crate) struct NativeLpProposal {
    basis: Vec<ConstraintKey>,
    last_warm_start: bool,
    consecutive_failures: u32,
    cooldown: usize,
}

#[derive(Clone)]
struct ConstraintKey {
    row: Vec<Rational>,
    rhs: Rational,
}

impl NativeLpProposal {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn optimized_solution_free(
        &mut self,
        a: &[Vec<Rational>],
        b: &[Rational],
        c: &[Rational],
    ) -> Option<(Vec<Rational>, Vec<Rational>)> {
        self.last_warm_start = false;
        if a.len() != b.len()
            || a.iter().any(|row| row.len() != c.len())
            || c.is_empty()
            || a.len() < c.len()
        {
            self.basis.clear();
            return None;
        }
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return None;
        }
        let solution = self.try_optimized_solution_free(a, b, c);
        if solution.is_some() {
            self.consecutive_failures = 0;
            self.cooldown = 0;
        } else {
            self.consecutive_failures = self.consecutive_failures.saturating_add(1);
            let exponent = self.consecutive_failures.saturating_sub(1).min(5);
            self.cooldown = 1_usize << exponent;
        }
        solution
    }

    fn try_optimized_solution_free(
        &mut self,
        a: &[Vec<Rational>],
        b: &[Rational],
        c: &[Rational],
    ) -> Option<(Vec<Rational>, Vec<Rational>)> {
        let problem = FloatingDual::new(a, b, c)?;

        if let Some(mut basis) = self.map_basis(a, b, c.len()) {
            if problem.optimize_from_feasible_basis(&mut basis) {
                if let Some(solution) = reconstruct_exact_solution(a, b, c, &basis) {
                    self.remember_basis(a, b, &basis);
                    self.last_warm_start = true;
                    return Some(solution);
                }
            }
        }

        let basis = problem.cold_start_and_optimize()?;
        let solution = reconstruct_exact_solution(a, b, c, &basis)?;
        self.remember_basis(a, b, &basis);
        Some(solution)
    }

    fn map_basis(
        &self,
        a: &[Vec<Rational>],
        b: &[Rational],
        dimension: usize,
    ) -> Option<Vec<usize>> {
        if self.basis.len() != dimension {
            return None;
        }
        let mut used = vec![false; a.len()];
        self.basis
            .iter()
            .map(|key| {
                a.iter()
                    .zip(b)
                    .enumerate()
                    .find(|(index, (row, rhs))| {
                        !used[*index] && *row == &key.row && *rhs == &key.rhs
                    })
                    .map(|(index, _)| {
                        used[index] = true;
                        index
                    })
            })
            .collect()
    }

    fn remember_basis(&mut self, a: &[Vec<Rational>], b: &[Rational], basis: &[usize]) {
        self.basis = basis
            .iter()
            .map(|index| ConstraintKey {
                row: a[*index].clone(),
                rhs: b[*index].clone(),
            })
            .collect();
    }

    #[cfg(test)]
    fn last_solve_was_warm(&self) -> bool {
        self.last_warm_start
    }
}

struct FloatingDual {
    columns: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    phase_one_costs: Vec<f64>,
    phase_two_costs: Vec<f64>,
    original_columns: usize,
    equations: usize,
}

impl FloatingDual {
    fn new(a: &[Vec<Rational>], b: &[Rational], c: &[Rational]) -> Option<Self> {
        let original_columns = a.len();
        let equations = c.len();
        let raw_a = a
            .iter()
            .map(|row| row.iter().map(Rational::to_f64).collect::<Option<Vec<_>>>())
            .collect::<Option<Vec<_>>>()?;
        let raw_b = b.iter().map(Rational::to_f64).collect::<Option<Vec<_>>>()?;
        let raw_c = c.iter().map(Rational::to_f64).collect::<Option<Vec<_>>>()?;

        let equation_scales = (0..equations)
            .map(|equation| {
                raw_a
                    .iter()
                    .map(|row| row[equation].abs())
                    .fold(raw_c[equation].abs().max(1.0), f64::max)
            })
            .collect::<Vec<_>>();
        if equation_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return None;
        }

        let mut columns = raw_a
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&equation_scales)
                    .map(|(value, scale)| value / scale)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let rhs = raw_c
            .iter()
            .zip(&equation_scales)
            .map(|(value, scale)| value / scale)
            .collect::<Vec<_>>();
        for equation in 0..equations {
            let mut artificial = vec![0.0; equations];
            artificial[equation] = if rhs[equation] < 0.0 { -1.0 } else { 1.0 };
            columns.push(artificial);
        }

        let mut phase_one_costs = vec![0.0; original_columns];
        phase_one_costs.extend(core::iter::repeat_n(1.0, equations));
        let objective_scale = raw_b.iter().map(|value| value.abs()).fold(1.0, f64::max);
        if !objective_scale.is_finite() {
            return None;
        }
        let mut phase_two_costs = raw_b
            .iter()
            .map(|value| value / objective_scale)
            .collect::<Vec<_>>();
        phase_two_costs.extend(core::iter::repeat_n(0.0, equations));

        Some(Self {
            columns,
            rhs,
            phase_one_costs,
            phase_two_costs,
            original_columns,
            equations,
        })
    }

    fn cold_start_and_optimize(&self) -> Option<Vec<usize>> {
        let mut basis = (0..self.equations)
            .map(|equation| self.original_columns + equation)
            .collect::<Vec<_>>();
        revised_minimize(
            &self.columns,
            &self.rhs,
            &self.phase_one_costs,
            self.columns.len(),
            &mut basis,
        )?;
        let inverse = floating_basis_inverse(&self.columns, &basis)?;
        let values = multiply_f64(&inverse, &self.rhs);
        let phase_one_value = basis
            .iter()
            .zip(&values)
            .map(|(column, value)| self.phase_one_costs[*column] * value)
            .sum::<f64>();
        if phase_one_value > 1.0e-7 {
            return None;
        }
        self.remove_artificial_basis(&mut basis)?;
        self.optimize_from_feasible_basis(&mut basis)
            .then_some(basis)
    }

    fn remove_artificial_basis(&self, basis: &mut [usize]) -> Option<()> {
        for row in 0..basis.len() {
            if basis[row] < self.original_columns {
                continue;
            }
            let inverse = floating_basis_inverse(&self.columns, basis)?;
            let basic = basis
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            let entering = (0..self.original_columns)
                .filter(|column| !basic.contains(column))
                .filter_map(|column| {
                    let direction = multiply_f64(&inverse, &self.columns[column]);
                    let pivot = direction[row].abs();
                    (pivot > 1.0e-10).then_some((column, pivot))
                })
                .max_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| right.0.cmp(&left.0))
                })
                .map(|(column, _)| column)?;
            basis[row] = entering;
        }
        Some(())
    }

    fn optimize_from_feasible_basis(&self, basis: &mut [usize]) -> bool {
        if basis.len() != self.equations
            || basis.iter().any(|column| *column >= self.original_columns)
        {
            return false;
        }
        let Some(inverse) = floating_basis_inverse(&self.columns, basis) else {
            return false;
        };
        if multiply_f64(&inverse, &self.rhs)
            .iter()
            .any(|value| *value < -1.0e-8)
        {
            return false;
        }
        revised_minimize(
            &self.columns,
            &self.rhs,
            &self.phase_two_costs,
            self.original_columns,
            basis,
        )
        .is_some()
    }
}

fn revised_minimize(
    columns: &[Vec<f64>],
    rhs: &[f64],
    costs: &[f64],
    allowed_columns: usize,
    basis: &mut [usize],
) -> Option<()> {
    let iteration_limit = 50_000_usize.checked_add(allowed_columns.checked_mul(100)?)?;
    for _ in 0..iteration_limit {
        let inverse = floating_basis_inverse(columns, basis)?;
        let values = multiply_f64(&inverse, rhs);
        if values.iter().any(|value| *value < -1.0e-8) {
            return None;
        }
        let mut prices = vec![0.0; basis.len()];
        for (basis_row, column) in basis.iter().enumerate() {
            for equation in 0..basis.len() {
                prices[equation] += costs[*column] * inverse[basis_row][equation];
            }
        }
        let basic = basis
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let entering = (0..allowed_columns)
            .filter(|column| !basic.contains(column))
            .map(|column| {
                let reduced = costs[column]
                    - prices
                        .iter()
                        .zip(&columns[column])
                        .map(|(price, entry)| price * entry)
                        .sum::<f64>();
                (column, reduced)
            })
            .filter(|(_, reduced)| *reduced < -1.0e-10)
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
        let Some((entering, _)) = entering else {
            return Some(());
        };
        let direction = multiply_f64(&inverse, &columns[entering]);
        let leaving = (0..basis.len())
            .filter(|row| direction[*row] > 1.0e-10)
            .map(|row| (row, values[row].max(0.0) / direction[row]))
            .min_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| basis[left.0].cmp(&basis[right.0]))
            })
            .map(|(row, _)| row)?;
        basis[leaving] = entering;
    }
    None
}

fn floating_basis_inverse(columns: &[Vec<f64>], basis: &[usize]) -> Option<Vec<Vec<f64>>> {
    let dimension = basis.len();
    let matrix = (0..dimension)
        .map(|row| {
            basis
                .iter()
                .map(|column| columns[*column][row])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    invert_f64(matrix)
}

fn invert_f64(matrix: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let dimension = matrix.len();
    if matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    let mut augmented = matrix
        .into_iter()
        .enumerate()
        .map(|(row, mut entries)| {
            entries.extend((0..dimension).map(|column| f64::from(row == column)));
            entries
        })
        .collect::<Vec<_>>();
    for column in 0..dimension {
        let pivot_row = (column..dimension).max_by(|left, right| {
            augmented[*left][column]
                .abs()
                .total_cmp(&augmented[*right][column].abs())
        })?;
        let pivot = augmented[pivot_row][column];
        if !pivot.is_finite() || pivot.abs() <= 1.0e-12 {
            return None;
        }
        augmented.swap(column, pivot_row);
        for entry in &mut augmented[column] {
            *entry /= pivot;
        }
        let pivot_entries = augmented[column].clone();
        for (row, entries) in augmented.iter_mut().enumerate() {
            if row == column {
                continue;
            }
            let factor = entries[column];
            if factor == 0.0 {
                continue;
            }
            for (entry, pivot_entry) in entries.iter_mut().zip(&pivot_entries) {
                *entry -= factor * pivot_entry;
            }
        }
    }
    let inverse = augmented
        .into_iter()
        .map(|row| row[dimension..].to_vec())
        .collect::<Vec<_>>();
    inverse
        .iter()
        .flatten()
        .all(|entry| entry.is_finite())
        .then_some(inverse)
}

fn multiply_f64(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect()
}

fn reconstruct_exact_solution(
    a: &[Vec<Rational>],
    b: &[Rational],
    c: &[Rational],
    basis: &[usize],
) -> Option<(Vec<Rational>, Vec<Rational>)> {
    if basis.len() != c.len() || basis.iter().any(|index| *index >= a.len()) {
        return None;
    }
    let dual_basis_matrix = (0..c.len())
        .map(|equation| {
            basis
                .iter()
                .map(|constraint| a[*constraint][equation].clone())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dual_basis = solve_square_exact(dual_basis_matrix, c.to_vec())?;
    let mut dual = vec![Rational::zero(); a.len()];
    for (constraint, value) in basis.iter().zip(dual_basis) {
        dual[*constraint] = value;
    }
    let primal_matrix = basis
        .iter()
        .map(|constraint| a[*constraint].clone())
        .collect::<Vec<_>>();
    let primal_rhs = basis
        .iter()
        .map(|constraint| b[*constraint].clone())
        .collect::<Vec<_>>();
    let primal = solve_square_exact(primal_matrix, primal_rhs)?;
    exact_primal_dual_match(a, b, c, &primal, &dual).then_some((primal, dual))
}

fn solve_square_exact(matrix: Vec<Vec<Rational>>, rhs: Vec<Rational>) -> Option<Vec<Rational>> {
    let dimension = matrix.len();
    if rhs.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    let mut augmented = matrix
        .into_iter()
        .zip(rhs)
        .map(|(mut row, value)| {
            row.push(value);
            row
        })
        .collect::<Vec<_>>();
    for column in 0..dimension {
        let pivot_row = (column..dimension).find(|row| !augmented[*row][column].is_zero())?;
        augmented.swap(column, pivot_row);
        let pivot = augmented[column][column].clone();
        for entry in &mut augmented[column][column..] {
            *entry = &*entry / &pivot;
        }
        let pivot_entries = augmented[column].clone();
        for (row, entries) in augmented.iter_mut().enumerate() {
            if row == column || entries[column].is_zero() {
                continue;
            }
            let factor = entries[column].clone();
            for index in column..=dimension {
                entries[index] = &entries[index] - &(&factor * &pivot_entries[index]);
            }
        }
    }
    Some(
        augmented
            .into_iter()
            .map(|row| row[dimension].clone())
            .collect(),
    )
}

fn exact_primal_dual_match(
    a: &[Vec<Rational>],
    b: &[Rational],
    c: &[Rational],
    primal: &[Rational],
    dual: &[Rational],
) -> bool {
    primal.len() == c.len()
        && dual.len() == a.len()
        && dual.iter().all(|entry| !entry.is_negative())
        && a.iter()
            .zip(b)
            .all(|(row, rhs)| dot_exact(row, primal) <= *rhs)
        && (0..c.len()).all(|column| {
            a.iter()
                .zip(dual)
                .fold(Rational::zero(), |sum, (row, multiplier)| {
                    &sum + &(&row[column] * multiplier)
                })
                == c[column]
        })
        && dot_exact(c, primal) == dot_exact(b, dual)
}

fn dot_exact(left: &[Rational], right: &[Rational]) -> Rational {
    left.iter()
        .zip(right)
        .fold(Rational::zero(), |sum, (left, right)| {
            &sum + &(left * right)
        })
}

/// Exact outcome of maximizing a linear functional over weak inequalities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LinearProgramResult {
    /// A primal optimum and a feasible dual attaining the same value.
    Optimal {
        /// Optimal values of the caller's variables.
        primal: Vec<Rational>,
        /// Exact optimal objective value.
        optimum: Rational,
        /// Nonnegative multiplier for every input inequality.
        dual: Vec<Rational>,
    },
    /// Farkas multipliers proving that the inequalities are inconsistent.
    Infeasible {
        /// Nonnegative `y` with `A^T y = 0` and `b^T y < 0`.
        certificate: Vec<Rational>,
    },
    /// The objective is unbounded above.
    Unbounded,
}

/// Maximize `c*x` subject to `A*x <= b`, with every `x_j` free in `Q`.
pub(crate) fn maximize_free(
    a: &[Vec<Rational>],
    b: &[Rational],
    c: &[Rational],
) -> Result<LinearProgramResult, String> {
    if a.len() != b.len() || a.iter().any(|row| row.len() != c.len()) {
        return Err("linear-program dimensions are inconsistent".to_owned());
    }
    let columns = c.len();
    let expanded_a = a
        .iter()
        .map(|row| {
            row.iter()
                .cloned()
                .chain(row.iter().map(|entry| -entry))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expanded_c = c
        .iter()
        .cloned()
        .chain(c.iter().map(|entry| -entry))
        .collect::<Vec<_>>();
    match SimplexTableau::new(&expanded_a, b, &expanded_c)?.solve() {
        LinearProgramResult::Optimal {
            primal,
            optimum,
            dual,
        } => Ok(LinearProgramResult::Optimal {
            primal: (0..columns)
                .map(|index| &primal[index] - &primal[index + columns])
                .collect(),
            optimum,
            dual,
        }),
        other => Ok(other),
    }
}

/// Two-phase simplex tableau for `max c*x`, `A*x <= b`, `x >= 0`.
struct SimplexTableau {
    constraints: usize,
    variables: usize,
    basic: Vec<isize>,
    nonbasic: Vec<isize>,
    entries: Vec<Vec<Rational>>,
}

impl SimplexTableau {
    fn new(a: &[Vec<Rational>], b: &[Rational], c: &[Rational]) -> Result<Self, String> {
        let constraints = b.len();
        let variables = c.len();
        if a.len() != constraints || a.iter().any(|row| row.len() != variables) {
            return Err("nonnegative linear-program dimensions are inconsistent".to_owned());
        }
        let mut entries = vec![vec![Rational::zero(); variables + 2]; constraints + 2];
        let mut basic = vec![0_isize; constraints];
        let mut nonbasic = vec![0_isize; variables + 1];
        for row in 0..constraints {
            for column in 0..variables {
                entries[row][column] = a[row][column].clone();
            }
            basic[row] = isize::try_from(variables + row)
                .map_err(|_| "linear program exceeds the simplex index domain".to_owned())?;
            entries[row][variables] = -Rational::one();
            entries[row][variables + 1] = b[row].clone();
        }
        for column in 0..variables {
            nonbasic[column] = isize::try_from(column)
                .map_err(|_| "linear program exceeds the simplex index domain".to_owned())?;
            entries[constraints][column] = -&c[column];
        }
        nonbasic[variables] = -1;
        entries[constraints + 1][variables] = Rational::one();
        Ok(Self {
            constraints,
            variables,
            basic,
            nonbasic,
            entries,
        })
    }

    fn pivot(&mut self, row: usize, column: usize) {
        let pivot = self.entries[row][column].clone();
        let pivot_row = self.entries[row].clone();
        let pivot_column = self
            .entries
            .iter()
            .map(|entry_row| entry_row[column].clone())
            .collect::<Vec<_>>();
        for (i, (entry_row, pivot_column_value)) in self
            .entries
            .iter_mut()
            .zip(&pivot_column)
            .enumerate()
            .take(self.constraints + 2)
        {
            if i == row {
                continue;
            }
            for (j, (entry, pivot_row_value)) in entry_row
                .iter_mut()
                .zip(&pivot_row)
                .enumerate()
                .take(self.variables + 2)
            {
                if j != column {
                    *entry = &*entry - &(pivot_row_value * pivot_column_value / &pivot);
                }
            }
        }
        for (j, entry) in self.entries[row]
            .iter_mut()
            .enumerate()
            .take(self.variables + 2)
        {
            if j != column {
                *entry = &*entry / &pivot;
            }
        }
        for (i, value) in pivot_column.iter().enumerate().take(self.constraints + 2) {
            if i != row {
                self.entries[i][column] = -value / &pivot;
            }
        }
        self.entries[row][column] = Rational::one() / pivot;
        core::mem::swap(&mut self.basic[row], &mut self.nonbasic[column]);
    }

    fn simplex(&mut self, phase_one: bool) -> bool {
        let objective_row = if phase_one {
            self.constraints + 1
        } else {
            self.constraints
        };
        loop {
            let mut entering = None;
            for column in 0..=self.variables {
                if !phase_one && self.nonbasic[column] == -1 {
                    continue;
                }
                let replace = entering.is_none_or(|old: usize| {
                    self.entries[objective_row][column] < self.entries[objective_row][old]
                        || (self.entries[objective_row][column] == self.entries[objective_row][old]
                            && self.nonbasic[column] < self.nonbasic[old])
                });
                if replace {
                    entering = Some(column);
                }
            }
            let Some(column) = entering else {
                return true;
            };
            if !self.entries[objective_row][column].is_negative() {
                return true;
            }
            let mut leaving = None;
            for row in 0..self.constraints {
                if !self.entries[row][column].is_positive() {
                    continue;
                }
                let replace = leaving.is_none_or(|old: usize| {
                    let candidate =
                        &self.entries[row][self.variables + 1] / &self.entries[row][column];
                    let current =
                        &self.entries[old][self.variables + 1] / &self.entries[old][column];
                    candidate < current
                        || (candidate == current && self.basic[row] < self.basic[old])
                });
                if replace {
                    leaving = Some(row);
                }
            }
            let Some(row) = leaving else {
                return false;
            };
            self.pivot(row, column);
        }
    }

    fn dual(&self, objective_row: usize) -> Vec<Rational> {
        (0..self.constraints)
            .map(|constraint| {
                let slack = isize::try_from(self.variables + constraint).ok();
                self.nonbasic
                    .iter()
                    .position(|entry| Some(*entry) == slack)
                    .map_or_else(Rational::zero, |column| {
                        self.entries[objective_row][column].clone()
                    })
            })
            .collect()
    }

    fn solve(mut self) -> LinearProgramResult {
        let rhs = self.variables + 1;
        let first_negative = (0..self.constraints).min_by(|left, right| {
            self.entries[*left][rhs]
                .cmp(&self.entries[*right][rhs])
                .then_with(|| self.basic[*left].cmp(&self.basic[*right]))
        });
        if let Some(row) = first_negative {
            if self.entries[row][rhs].is_negative() {
                self.pivot(row, self.variables);
                let feasible = self.simplex(true);
                if !feasible || self.entries[self.constraints + 1][rhs].is_negative() {
                    return LinearProgramResult::Infeasible {
                        certificate: self.dual(self.constraints + 1),
                    };
                }
                if let Some(artificial_row) = self.basic.iter().position(|entry| *entry == -1) {
                    let entering = (0..=self.variables)
                        .filter(|column| !self.entries[artificial_row][*column].is_zero())
                        .min_by_key(|column| self.nonbasic[*column]);
                    if let Some(column) = entering {
                        self.pivot(artificial_row, column);
                    }
                }
            }
        }
        if !self.simplex(false) {
            return LinearProgramResult::Unbounded;
        }
        let mut primal = vec![Rational::zero(); self.variables];
        for row in 0..self.constraints {
            if self.basic[row] >= 0 {
                if let Ok(index) = usize::try_from(self.basic[row]) {
                    if index < self.variables {
                        primal[index] = self.entries[row][rhs].clone();
                    }
                }
            }
        }
        LinearProgramResult::Optimal {
            primal,
            optimum: self.entries[self.constraints][rhs].clone(),
            dual: self.dual(self.constraints),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(value: i64) -> Rational {
        Rational::from(value)
    }

    #[test]
    fn solves_a_bounded_free_problem_with_dual_evidence() {
        // max x subject to x <= 3 and -x <= 2.
        let result = maximize_free(&[vec![q(1)], vec![q(-1)]], &[q(3), q(2)], &[q(1)]).unwrap();
        let LinearProgramResult::Optimal {
            primal,
            optimum,
            dual,
        } = result
        else {
            panic!("expected a finite optimum")
        };
        assert_eq!(primal, vec![q(3)]);
        assert_eq!(optimum, q(3));
        assert_eq!(dual, vec![q(1), q(0)]);
    }

    #[test]
    fn native_revised_proposal_reconstructs_an_exact_optimum() {
        // max x + y subject to x <= 3, y <= 2, -x <= 1, -y <= 1.
        let a = vec![
            vec![q(1), q(0)],
            vec![q(0), q(1)],
            vec![q(-1), q(0)],
            vec![q(0), q(-1)],
        ];
        let b = vec![q(3), q(2), q(1), q(1)];
        let c = vec![q(1), q(1)];
        let mut proposal = NativeLpProposal::new();

        let (primal, dual) = proposal.optimized_solution_free(&a, &b, &c).unwrap();

        assert_eq!(primal, vec![q(3), q(2)]);
        assert_eq!(dot_exact(&c, &primal), q(5));
        assert!(exact_primal_dual_match(&a, &b, &c, &primal, &dual));
        assert!(!proposal.last_solve_was_warm());
    }

    #[test]
    fn native_revised_proposal_warm_starts_after_a_cut_and_row_reordering() {
        let mut proposal = NativeLpProposal::new();
        let first_a = vec![vec![q(1)], vec![q(-1)]];
        let first_b = vec![q(3), q(2)];
        let c = vec![q(1)];
        let (first, _) = proposal
            .optimized_solution_free(&first_a, &first_b, &c)
            .unwrap();
        assert_eq!(first, vec![q(3)]);

        // The old active constraint moves to the middle and a stronger cut is
        // inserted. Exact row identity maps the cached basis before reoptimization.
        let second_a = vec![vec![q(-1)], vec![q(1)], vec![q(1)]];
        let second_b = vec![q(2), q(3), q(2)];
        let (second, dual) = proposal
            .optimized_solution_free(&second_a, &second_b, &c)
            .unwrap();

        assert_eq!(second, vec![q(2)]);
        assert!(exact_primal_dual_match(
            &second_a, &second_b, &c, &second, &dual
        ));
        assert!(proposal.last_solve_was_warm());

        let repeated = proposal
            .optimized_solution_free(&second_a, &second_b, &c)
            .unwrap();
        assert_eq!(repeated, (second, dual));
        assert!(proposal.last_solve_was_warm());
    }

    #[test]
    fn native_revised_proposal_recovers_rational_basis_values_exactly() {
        // max x subject to 3x <= 2 and -x <= 1.
        let a = vec![vec![q(3)], vec![q(-1)]];
        let b = vec![q(2), q(1)];
        let c = vec![q(1)];
        let mut proposal = NativeLpProposal::new();

        let (primal, dual) = proposal.optimized_solution_free(&a, &b, &c).unwrap();

        assert_eq!(primal, vec![Rational::new(2.into(), 3.into()).unwrap()]);
        assert_eq!(dual, vec![Rational::new(1.into(), 3.into()).unwrap(), q(0)]);
        assert!(exact_primal_dual_match(&a, &b, &c, &primal, &dual));
    }

    #[test]
    fn native_revised_proposal_fails_closed_when_no_exact_primal_matches() {
        // x <= 0 and x >= 1. Floating dual work must not manufacture a
        // proposal for an infeasible primal; the exact tableau owns Farkas.
        let a = vec![vec![q(1)], vec![q(-1)]];
        let b = vec![q(0), q(-1)];
        let mut proposal = NativeLpProposal::new();

        assert!(proposal.optimized_solution_free(&a, &b, &[q(0)]).is_none());
        assert!(matches!(
            maximize_free(&a, &b, &[q(0)]).unwrap(),
            LinearProgramResult::Infeasible { .. }
        ));
    }

    #[test]
    fn returns_a_checked_farkas_shape_for_infeasibility() {
        // x <= 0 and x >= 1.
        let a = vec![vec![q(1)], vec![q(-1)]];
        let b = vec![q(0), q(-1)];
        let result = maximize_free(&a, &b, &[q(0)]).unwrap();
        let LinearProgramResult::Infeasible { certificate } = result else {
            panic!("expected infeasibility")
        };
        assert!(certificate.iter().all(|entry| !entry.is_negative()));
        assert_eq!(&certificate[0] - &certificate[1], q(0));
        assert!((&b[0] * &certificate[0] + &b[1] * &certificate[1]).is_negative());
    }

    #[test]
    fn detects_an_unbounded_free_problem() {
        assert_eq!(
            maximize_free(&[], &[], &[q(1)]).unwrap(),
            LinearProgramResult::Unbounded
        );
    }
}
