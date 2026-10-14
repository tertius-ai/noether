// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam

use std::collections::{BTreeMap, HashMap};

use geom_core::ball::min_enclosing_ball;
use geom_core::{Metric, PointCloud, Triangulation};

use super::{FilteredComplex, Simplex, TopologyError};

/// Default ceiling on face slots walked by an alpha-filtration build.
///
/// A rank-`d` maximal simplex has `2^(d + 1) - 1` nonempty faces.  The
/// triangulation is already available when this guard is checked, so this is
/// a bound on the remaining work rather than a prediction or planner.
pub const DEFAULT_ALPHA_FACE_SLOTS: usize = 2_000_000;

impl FilteredComplex {
    /// Construct the Vietoris–Rips filtration of any checked finite metric.
    ///
    /// `max_scale` is a distance. `max_dimension` is the largest simplex
    /// dimension materialized; homology in degree `k` needs it to be at least
    /// `k + 1` so deaths can be detected.
    #[allow(clippy::needless_range_loop)]
    pub fn vietoris_rips<M: Metric + ?Sized>(
        metric: &M,
        max_scale: f64,
        max_dimension: usize,
    ) -> Result<Self, TopologyError> {
        validate_scale("Vietoris–Rips", max_scale)?;
        validate_vertex_count(metric.len())?;

        let n = metric.len();
        let mut entries = (0..n)
            .map(|index| (Simplex::new(vec![index as u32]), 0.0))
            .collect::<Vec<_>>();
        if n == 0 || max_dimension == 0 {
            return Self::new(entries);
        }

        let mut edge = vec![vec![None; n]; n];
        let mut current = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let distance = checked_distance(metric, i, j, "Vietoris–Rips")?;
                if distance <= max_scale {
                    edge[i][j] = Some(distance);
                    edge[j][i] = Some(distance);
                    entries.push((Simplex::new(vec![i as u32, j as u32]), distance));
                    current.push((vec![i as u32, j as u32], distance));
                }
            }
        }

        for _ in 2..=max_dimension {
            let mut next = Vec::new();
            for (vertices, value) in &current {
                let Some(&last) = vertices.last() else {
                    continue;
                };
                for vertex in (last as usize + 1)..n {
                    let mut worst = *value;
                    let mut is_clique = true;
                    for existing in vertices {
                        match edge[*existing as usize][vertex] {
                            Some(distance) => worst = worst.max(distance),
                            None => {
                                is_clique = false;
                                break;
                            }
                        }
                    }
                    if is_clique {
                        let mut simplex = vertices.clone();
                        simplex.push(vertex as u32);
                        entries.push((Simplex::new(simplex.clone()), worst));
                        next.push((simplex, worst));
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            current = next;
        }
        Self::new(entries)
    }

    /// Construct the Čech filtration of a Euclidean point cloud.
    ///
    /// `max_scale` is a ball radius, so an edge of length `d` enters at
    /// `d / 2`. This differs deliberately from Vietoris–Rips' distance axis.
    #[allow(clippy::needless_range_loop)]
    pub fn cech(
        points: &PointCloud,
        max_scale: f64,
        max_dimension: usize,
    ) -> Result<Self, TopologyError> {
        validate_scale("Čech", max_scale)?;
        validate_vertex_count(points.len())?;

        let n = points.len();
        let mut entries = (0..n)
            .map(|index| (Simplex::new(vec![index as u32]), 0.0))
            .collect::<Vec<_>>();
        if n == 0 || max_dimension == 0 {
            return Self::new(entries);
        }

        let mut values = BTreeMap::new();
        for (simplex, value) in &entries {
            values.insert(simplex.clone(), *value);
        }

        let mut adjacent = vec![vec![false; n]; n];
        let mut current = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let radius = checked_distance(points, i, j, "Čech")? / 2.0;
                if radius <= max_scale {
                    adjacent[i][j] = true;
                    adjacent[j][i] = true;
                    let simplex = Simplex::new(vec![i as u32, j as u32]);
                    entries.push((simplex.clone(), radius));
                    values.insert(simplex, radius);
                    current.push(vec![i as u32, j as u32]);
                }
            }
        }

        for _ in 2..=max_dimension {
            let mut next = Vec::new();
            for vertices in &current {
                let Some(&last) = vertices.last() else {
                    continue;
                };
                for vertex in (last as usize + 1)..n {
                    if !vertices
                        .iter()
                        .all(|existing| adjacent[*existing as usize][vertex])
                    {
                        continue;
                    }
                    let mut simplex_vertices = vertices.clone();
                    simplex_vertices.push(vertex as u32);
                    let rows = simplex_vertices
                        .iter()
                        .map(|index| points.point(*index as usize).to_vec())
                        .collect::<Vec<_>>();
                    let ball = min_enclosing_ball(&rows).ok_or_else(|| {
                        TopologyError::InvalidFiltration {
                            reason: format!(
                                "Čech could not construct the enclosing ball of simplex {simplex_vertices:?}"
                            ),
                        }
                    })?;
                    let simplex = Simplex::new(simplex_vertices.clone());
                    let radius = simplex
                        .faces()
                        .iter()
                        .filter_map(|face| values.get(face))
                        .fold(ball.radius(), |current, face| current.max(*face));
                    if radius <= max_scale {
                        entries.push((simplex.clone(), radius));
                        values.insert(simplex, radius);
                        next.push(simplex_vertices);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            current = next;
        }
        Self::new(entries)
    }

    /// Construct the alpha filtration carried by a Delaunay triangulation.
    ///
    /// `max_scale` is a ball radius, matching [`Self::cech`].
    /// `max_dimension` is the largest simplex dimension returned.  The
    /// already-built triangulation is consumed as geometry but remains
    /// reusable by the caller; Noether does not triangulate the points again.
    ///
    /// ```
    /// use noether::geometry::{delaunay, PointCloud};
    /// use noether::topology::FilteredComplex;
    ///
    /// let points = PointCloud::new(&[
    ///     vec![0.0, 0.0], vec![3.0, 0.0], vec![0.0, 4.0],
    /// ])?;
    /// let triangulation = delaunay(&points)?;
    /// let filtration = FilteredComplex::alpha(&triangulation, 3.0, 2)?;
    /// assert!(filtration.values().contains(&2.5));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn alpha(
        triangulation: &Triangulation,
        max_scale: f64,
        max_dimension: usize,
    ) -> Result<Self, TopologyError> {
        Self::alpha_with_max_face_slots(
            triangulation,
            max_scale,
            max_dimension,
            DEFAULT_ALPHA_FACE_SLOTS,
        )
    }

    /// Construct an alpha filtration with an explicit face-slot ceiling.
    ///
    /// This is the opt-in route when the deterministic default guard in
    /// [`Self::alpha`] is too small.  It changes only the permitted cost, not
    /// the mathematical result, and exceeding it is an error rather than a
    /// truncated filtration.
    pub fn alpha_with_max_face_slots(
        triangulation: &Triangulation,
        max_scale: f64,
        max_dimension: usize,
        max_face_slots: usize,
    ) -> Result<Self, TopologyError> {
        validate_scale("alpha", max_scale)?;
        validate_vertex_count(triangulation.points().len())?;
        if triangulation.weights().is_some() {
            return invalid_filtration(
                "alpha requires an ordinary Delaunay triangulation; weighted alpha values are not implemented",
            );
        }

        let rank = triangulation.dim();
        let cells = triangulation.cells().collect::<Vec<_>>();
        let slots_per_cell = 1usize
            .checked_shl((rank + 1) as u32)
            .and_then(|slots| slots.checked_sub(1))
            .ok_or_else(|| TopologyError::InvalidFiltration {
                reason: format!(
                    "alpha cannot represent the 2^{} - 1 nonempty faces of one maximal simplex",
                    rank + 1
                ),
            })?;
        let face_slots = cells.len().checked_mul(slots_per_cell).ok_or_else(|| {
            TopologyError::InvalidFiltration {
                reason: "alpha face-slot count overflowed usize".to_owned(),
            }
        })?;
        if face_slots > max_face_slots {
            return invalid_filtration(format!(
                "alpha would walk {face_slots} face slots from {} rank-{rank} Delaunay cells, past the {max_face_slots} ceiling; raise max_face_slots explicitly to opt in",
                cells.len()
            ));
        }
        if u32::try_from(cells.len()).is_err() {
            return invalid_filtration(format!(
                "alpha cannot index {} Delaunay cells with u32 coface labels",
                cells.len()
            ));
        }

        struct Face {
            radius_squared: f64,
            value: f64,
            attached: bool,
            cofaces: Vec<u32>,
        }

        let points = triangulation.points();
        // Keys are vertex labels generated by the checked triangulation.  A
        // hash table keeps the face walk linear; `FilteredComplex::new`
        // canonicalizes the final order, so randomized iteration cannot leak
        // into the semantic value.
        let mut faces = HashMap::<Vec<u32>, Face>::new();
        for (cell_index, cell) in cells.iter().enumerate() {
            let cell_len = cell.len();
            if cell_len != rank + 1 {
                return invalid_filtration(format!(
                    "alpha received a rank-{rank} triangulation with a {cell_len}-vertex maximal simplex"
                ));
            }
            for mask in 1usize..(1usize << cell_len) {
                let mut vertices = (0..cell_len)
                    .filter(|index| mask >> index & 1 == 1)
                    .map(|index| cell[index])
                    .collect::<Vec<_>>();
                vertices.sort_unstable();
                if let Some(face) = faces.get_mut(&vertices) {
                    face.cofaces.push(cell_index as u32);
                    continue;
                }
                let rows = vertices
                    .iter()
                    .map(|vertex| &points[*vertex as usize][..])
                    .collect::<Vec<_>>();
                let radius_squared = checked_alpha_ball(&rows, &vertices)?.radius_sq;
                faces.insert(
                    vertices,
                    Face {
                        radius_squared,
                        value: f64::INFINITY,
                        attached: false,
                        cofaces: vec![cell_index as u32],
                    },
                );
            }
        }

        let keys = faces.keys().cloned().collect::<Vec<_>>();
        for vertices in &keys {
            let rows = vertices
                .iter()
                .map(|vertex| &points[*vertex as usize][..])
                .collect::<Vec<_>>();
            let ball = checked_alpha_ball(&rows, vertices)?;
            let radius_squared = ball.radius_sq;
            let cofaces = faces
                .get(vertices)
                .map(|face| face.cofaces.clone())
                .unwrap_or_default();
            let mut attached = false;
            for cell_index in cofaces {
                for vertex in cells[cell_index as usize] {
                    if vertices.contains(vertex) {
                        continue;
                    }
                    let distance_squared = points[*vertex as usize]
                        .iter()
                        .zip(&ball.center)
                        .map(|(coordinate, centre)| (coordinate - centre) * (coordinate - centre))
                        .sum::<f64>();
                    if !distance_squared.is_finite() {
                        return invalid_filtration(format!(
                            "alpha distance from vertex {vertex} to the circumsphere of {vertices:?} is not finite"
                        ));
                    }
                    // The comparison is scale-relative.  An absolute floor
                    // makes the same cloud answer differently after a change
                    // of units, especially below unit scale.
                    let tolerance = 1e-12 * radius_squared.abs().max(distance_squared.abs());
                    if distance_squared < radius_squared - tolerance {
                        attached = true;
                        break;
                    }
                }
                if attached {
                    break;
                }
            }
            if let Some(face) = faces.get_mut(vertices) {
                face.attached = attached;
            }
        }

        let top_size = keys.iter().map(Vec::len).max().unwrap_or(1);
        let mut by_size = vec![Vec::<Vec<u32>>::new(); top_size + 1];
        for vertices in keys {
            by_size[vertices.len()].push(vertices);
        }
        for size in (1..by_size.len()).rev() {
            let level = std::mem::take(&mut by_size[size]);
            for vertices in &level {
                let Some(face) = faces.get(vertices) else {
                    continue;
                };
                let value = if !face.attached {
                    face.radius_squared.min(face.value)
                } else if face.value.is_finite() {
                    face.value
                } else {
                    face.radius_squared
                };
                if let Some(face) = faces.get_mut(vertices) {
                    face.value = value;
                }
                if size > 1 {
                    for removed in 0..vertices.len() {
                        let mut facet = vertices.clone();
                        facet.remove(removed);
                        if let Some(face) = faces.get_mut(&facet) {
                            face.value = face.value.min(value);
                        }
                    }
                }
            }
            by_size[size] = level;
        }

        let max_scale_squared = max_scale * max_scale;
        let entries = faces
            .into_iter()
            .filter(|(vertices, face)| {
                vertices.len().saturating_sub(1) <= max_dimension
                    && face.value.is_finite()
                    && face.value <= max_scale_squared
            })
            .map(|(vertices, face)| (Simplex::new(vertices), face.value.max(0.0).sqrt()))
            .collect();
        Self::new(entries)
    }
}

fn checked_alpha_ball(rows: &[&[f64]], vertices: &[u32]) -> Result<geom_core::Ball, TopologyError> {
    let ball = geom_core::ball::circumsphere(rows, None).ok_or_else(|| {
        TopologyError::InvalidFiltration {
            reason: format!(
                "alpha could not construct the circumsphere of simplex {vertices:?} from a checked triangulation"
            ),
        }
    })?;
    if ball.radius_sq.is_finite()
        && ball.radius_sq >= 0.0
        && ball.center.iter().all(|coordinate| coordinate.is_finite())
    {
        Ok(ball)
    } else {
        invalid_filtration(format!(
            "alpha constructed a non-finite circumsphere for simplex {vertices:?}"
        ))
    }
}

fn invalid_filtration<T>(reason: impl Into<String>) -> Result<T, TopologyError> {
    Err(TopologyError::InvalidFiltration {
        reason: reason.into(),
    })
}

fn validate_scale(name: &str, max_scale: f64) -> Result<(), TopologyError> {
    if max_scale.is_finite() && max_scale >= 0.0 {
        return Ok(());
    }
    Err(TopologyError::InvalidFiltration {
        reason: format!("{name} max_scale must be finite and non-negative; received {max_scale}"),
    })
}

fn validate_vertex_count(count: usize) -> Result<(), TopologyError> {
    if u32::try_from(count).is_ok() {
        return Ok(());
    }
    Err(TopologyError::InvalidFiltration {
        reason: format!("{count} vertices cannot be represented by u32 simplex labels"),
    })
}

fn checked_distance<M: Metric + ?Sized>(
    metric: &M,
    i: usize,
    j: usize,
    construction: &str,
) -> Result<f64, TopologyError> {
    let distance = metric.distance(i, j);
    if distance.is_finite() && distance >= 0.0 {
        return Ok(distance);
    }
    Err(TopologyError::InvalidFiltration {
        reason: format!(
            "{construction} received invalid distance d({i}, {j}) = {distance}; distances must be finite and non-negative"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct ThreePointMetric;

    impl Metric for ThreePointMetric {
        fn len(&self) -> usize {
            3
        }

        fn distance(&self, i: usize, j: usize) -> f64 {
            if i == j {
                0.0
            } else if (i, j) == (0, 2) || (i, j) == (2, 0) {
                2.0
            } else {
                1.0
            }
        }
    }

    #[test]
    fn rips_needs_only_the_metric_contract() {
        let filtration = FilteredComplex::vietoris_rips(&ThreePointMetric, 1.0, 2).unwrap();
        assert_eq!(filtration.len(), 5);
        assert_eq!(filtration.dimension(), 1);
        assert!(filtration
            .simplices()
            .iter()
            .any(|simplex| simplex.vertices() == [0, 1]));
        assert!(!filtration
            .simplices()
            .iter()
            .any(|simplex| simplex.vertices() == [0, 2]));
    }

    #[test]
    fn cech_uses_radius_and_the_true_enclosing_ball() {
        let edge = PointCloud::new(&[vec![0.0], vec![2.0]]).unwrap();
        let filtration = FilteredComplex::cech(&edge, 1.0, 1).unwrap();
        assert_eq!(filtration.values().last(), Some(&1.0));

        let obtuse = PointCloud::new(&[vec![0.0, 0.0], vec![4.0, 0.0], vec![2.0, 0.1]]).unwrap();
        let filtration = FilteredComplex::cech(&obtuse, 2.0, 2).unwrap();
        let triangle = Simplex::new(vec![0, 1, 2]);
        assert_eq!(
            filtration.values()[filtration.position(&triangle).unwrap()],
            2.0
        );
    }

    #[test]
    fn invalid_scales_and_distances_fail_closed() {
        assert!(FilteredComplex::vietoris_rips(&ThreePointMetric, f64::NAN, 2).is_err());
        assert!(
            FilteredComplex::cech(&PointCloud::new(&[vec![0.0], vec![1.0]]).unwrap(), -1.0, 1)
                .is_err()
        );

        struct InfiniteMetric;
        impl Metric for InfiniteMetric {
            fn len(&self) -> usize {
                2
            }
            fn distance(&self, _i: usize, _j: usize) -> f64 {
                f64::INFINITY
            }
        }
        assert!(FilteredComplex::vietoris_rips(&InfiniteMetric, 1.0, 1).is_err());
    }

    #[test]
    fn alpha_consumes_the_existing_triangulation() {
        let points = PointCloud::new(&[vec![0.0, 0.0], vec![3.0, 0.0], vec![0.0, 4.0]]).unwrap();
        let triangulation = geom_core::delaunay(&points).unwrap();
        let filtration = FilteredComplex::alpha(&triangulation, 3.0, 2).unwrap();
        let triangle = Simplex::new(vec![0, 1, 2]);
        assert_eq!(
            filtration.values()[filtration.position(&triangle).unwrap()],
            2.5
        );
    }

    #[test]
    fn alpha_is_similarity_equivariant_across_scales() {
        for scale in [1e-4, 1.0, 1e4] {
            let points = PointCloud::new(&[
                vec![0.0, 0.0],
                vec![3.0 * scale, 0.0],
                vec![0.0, 4.0 * scale],
            ])
            .unwrap();
            let triangulation = geom_core::delaunay(&points).unwrap();
            let filtration = FilteredComplex::alpha(&triangulation, 3.0 * scale, 2).unwrap();
            let triangle = Simplex::new(vec![0, 1, 2]);
            let value = filtration.values()[filtration.position(&triangle).unwrap()];
            assert!((value / scale - 2.5).abs() < 1e-9, "scale {scale}: {value}");
        }
    }

    #[test]
    fn alpha_rejects_weighted_input_and_cost_overruns() {
        let points = PointCloud::new(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let weighted = geom_core::regular(&points, &[0.0; 3]).unwrap();
        assert!(FilteredComplex::alpha(&weighted, 1.0, 2).is_err());

        let triangulation = geom_core::delaunay(&points).unwrap();
        let error =
            FilteredComplex::alpha_with_max_face_slots(&triangulation, 1.0, 2, 0).unwrap_err();
        assert!(error.to_string().contains("face slots"));
    }
}
