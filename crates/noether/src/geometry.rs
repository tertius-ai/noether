// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Exact predicates, polytopes, and Euclidean constructions.

use core::fmt;
use core::str::FromStr;

pub use convex_hull::{Face, FaceLattice, Facet};
pub use geom_core::{circumsphere, DistanceMatrix, Metric, PointCloud, Triangulation};
use serde::{Deserialize, Serialize};

use crate::{NoetherError, Rational, SemanticType, TypeDescription};

pub(crate) mod exact_lp;

static POINT_CLOUD_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/geometry/PointCloud@1",
    rust_path: "noether::geometry::PointCloud",
    python_path: Some("noether.geometry.PointCloud"),
};

static TRIANGULATION_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/geometry/Triangulation@1",
    rust_path: "noether::geometry::Triangulation",
    python_path: Some("noether.geometry.Triangulation"),
};

static POLYTOPE_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/geometry/Polytope@1",
    rust_path: "noether::geometry::Polytope",
    python_path: Some("noether.geometry.Polytope"),
};

/// Failure to construct or verify a geometric object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryError {
    reason: String,
}

impl GeometryError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Precise failed geometric precondition.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "geometry error: {}", self.reason)
    }
}

impl std::error::Error for GeometryError {}

/// A nonempty bounded convex polytope over the rational numbers.
///
/// `convex-hull` owns the exact V/H conversion, incidence, and face-lattice
/// algorithms and stores the same shared FLINT-backed [`Rational`] value that
/// the facade exposes.
#[derive(Clone)]
pub struct Polytope {
    inner: convex_hull::Polytope,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolytopeRepr {
    #[serde(rename = "type")]
    type_id: String,
    vertices: Vec<Vec<String>>,
}

impl Polytope {
    /// Construct the convex hull of exact rational points.
    pub fn from_vertices(points: &[Vec<Rational>]) -> Result<Self, GeometryError> {
        let points = points.to_vec();
        let inner = convex_hull::convex_hull(&points).map_err(geometry_kernel_error)?;
        Self::from_kernel(inner)
    }

    /// Construct a bounded polytope from inequalities `normal_i · x <= rhs_i`.
    ///
    /// Empty, infeasible, and unbounded regions fail rather than being
    /// represented by a bounded type.
    pub fn from_halfspaces(
        normals: &[Vec<Rational>],
        rhs: &[Rational],
    ) -> Result<Self, GeometryError> {
        let dimension = normals.first().map_or(0, Vec::len);
        let normals = normals.to_vec();
        let rhs = rhs.to_vec();
        let inner = convex_hull::Polytope::from_inequalities(&normals, &rhs, dimension)
            .map_err(geometry_kernel_error)?;
        Self::from_kernel(inner)
    }

    fn from_kernel(inner: convex_hull::Polytope) -> Result<Self, GeometryError> {
        Ok(Self { inner })
    }

    /// Ambient coordinate dimension.
    pub fn ambient_dimension(&self) -> usize {
        self.inner.ambient_dim()
    }

    /// Affine dimension of the polytope.
    pub fn dimension(&self) -> usize {
        self.inner.dim() as usize
    }

    /// Whether the affine hull is the whole ambient space.
    pub fn is_full_dimensional(&self) -> bool {
        self.inner.is_full_dimensional()
    }

    /// Lexicographically ordered vertices in ambient coordinates.
    pub fn vertices(&self) -> &[Vec<Rational>] {
        self.inner.vertices()
    }

    /// Exact ambient facet inequalities.
    ///
    /// The kernel's inequalities for a lower-dimensional polytope live in a
    /// deterministic affine coordinate frame, not the original ambient
    /// coordinates. This method fails in that case instead of mislabelling
    /// them; incidence and the face lattice remain available.
    pub fn facets(&self) -> Result<&[Facet], GeometryError> {
        if self.inner.is_full_dimensional() {
            Ok(self.inner.facets())
        } else {
            Err(GeometryError::new(
                "ambient facet inequalities require a full-dimensional polytope",
            ))
        }
    }

    /// Vertex indices lying on each facet.
    pub fn vertex_facet_incidence(&self) -> &[Vec<usize>] {
        self.inner.vertex_facet_incidence()
    }

    /// Face counts `(f_-1, f_0, ..., f_dimension)`.
    pub fn f_vector(&self) -> Vec<usize> {
        self.inner.f_vector()
    }

    /// Complete face lattice and its Hasse covering relation.
    pub fn face_lattice(&self) -> FaceLattice {
        self.inner.face_lattice()
    }

    /// Whether an ambient rational point is a vertex.
    pub fn is_vertex(&self, point: &[Rational]) -> Result<bool, GeometryError> {
        Ok(self.inner.is_vertex(point))
    }

    /// Whether an ambient inequality defines a facet.
    pub fn is_facet(&self, normal: &[Rational], rhs: &Rational) -> Result<bool, GeometryError> {
        Ok(self.inner.is_facet(normal, rhs))
    }

    /// Reconstruct from the defining vertices and compare every derived value.
    pub fn verify(&self) -> Result<(), GeometryError> {
        let rebuilt = Self::from_vertices(self.inner.vertices())?;
        if rebuilt.inner.vertices() == self.inner.vertices()
            && kernel_facet_signature(&rebuilt.inner) == kernel_facet_signature(&self.inner)
            && rebuilt.f_vector() == self.f_vector()
        {
            Ok(())
        } else {
            Err(GeometryError::new(
                "polytope differs from its exact reconstruction",
            ))
        }
    }

    /// Serialize the type identity and exact defining vertices.
    pub fn to_json(&self) -> Result<String, GeometryError> {
        let repr = PolytopeRepr {
            type_id: Self::TYPE_ID.to_owned(),
            vertices: self
                .inner
                .vertices()
                .iter()
                .map(|row| row.iter().map(ToString::to_string).collect())
                .collect(),
        };
        serde_json::to_string(&repr).map_err(|error| GeometryError::new(error.to_string()))
    }

    /// Deserialize and recompute every derived polytope invariant.
    pub fn from_json(value: &str) -> Result<Self, GeometryError> {
        let repr: PolytopeRepr =
            serde_json::from_str(value).map_err(|error| GeometryError::new(error.to_string()))?;
        if repr.type_id != Self::TYPE_ID {
            return Err(GeometryError::new(format!(
                "expected type {}, got {}",
                Self::TYPE_ID,
                repr.type_id
            )));
        }
        let vertices = repr
            .vertices
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| {
                        Rational::from_str(value)
                            .map_err(|error| GeometryError::new(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_vertices(&vertices)
    }
}

impl PartialEq for Polytope {
    fn eq(&self, other: &Self) -> bool {
        self.inner.vertices() == other.inner.vertices()
    }
}

impl Eq for Polytope {}

impl fmt::Debug for Polytope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Polytope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Polytope(dimension={}, ambient_dimension={}, vertices={}, facets={})",
            self.dimension(),
            self.ambient_dimension(),
            self.inner.n_vertices(),
            self.inner.n_facets()
        )
    }
}

impl SemanticType for Polytope {
    const TYPE_ID: &'static str = "noether/geometry/Polytope@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json()
            .map(String::into_bytes)
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    fn describe() -> &'static TypeDescription {
        &POLYTOPE_DESCRIPTION
    }
}

fn geometry_kernel_error(error: convex_hull::HullError) -> GeometryError {
    GeometryError::new(error.to_string())
}

fn kernel_facet_signature(
    polytope: &convex_hull::Polytope,
) -> Vec<(Vec<convex_hull::Q>, Vec<usize>)> {
    let mut signature = polytope
        .facets()
        .iter()
        .zip(polytope.vertex_facet_incidence())
        .map(|(facet, incidence)| {
            let mut inequality = facet.normal.clone();
            inequality.push(facet.rhs.clone());
            (inequality, incidence.clone())
        })
        .collect::<Vec<_>>();
    signature.sort();
    signature
}

/// Compute the exact rational convex hull of a finite point set.
pub fn convex_hull(points: &[Vec<Rational>]) -> Result<Polytope, GeometryError> {
    Polytope::from_vertices(points)
}

/// Construct the Delaunay triangulation of a checked point cloud.
pub fn delaunay(points: &PointCloud) -> Result<Triangulation, GeometryError> {
    geom_core::delaunay(points).map_err(GeometryError::new)
}

/// Construct the regular triangulation with one finite weight per site.
pub fn regular_triangulation(
    points: &PointCloud,
    weights: &[f64],
) -> Result<Triangulation, GeometryError> {
    geom_core::regular(points, weights).map_err(GeometryError::new)
}

impl SemanticType for PointCloud {
    const TYPE_ID: &'static str = "noether/geometry/PointCloud@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        Ok(point_rows_bytes(&self.to_rows()))
    }

    fn describe() -> &'static TypeDescription {
        &POINT_CLOUD_DESCRIPTION
    }
}

impl SemanticType for Triangulation {
    const TYPE_ID: &'static str = "noether/geometry/Triangulation@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        let mut output = point_rows_bytes(self.points());
        output.extend_from_slice(&(self.dim() as u64).to_be_bytes());
        match self.weights() {
            None => output.push(0),
            Some(weights) => {
                output.push(1);
                output.extend_from_slice(&(weights.len() as u64).to_be_bytes());
                for weight in weights {
                    output.extend_from_slice(&weight.to_bits().to_be_bytes());
                }
            }
        }

        let mut cells = self
            .cells()
            .map(|cell| {
                let mut cell = cell.to_vec();
                cell.sort_unstable();
                cell
            })
            .collect::<Vec<_>>();
        cells.sort_unstable();
        output.extend_from_slice(&(cells.len() as u64).to_be_bytes());
        for cell in cells {
            output.extend_from_slice(&(cell.len() as u64).to_be_bytes());
            for vertex in cell {
                output.extend_from_slice(&vertex.to_be_bytes());
            }
        }
        Ok(output)
    }

    fn describe() -> &'static TypeDescription {
        &TRIANGULATION_DESCRIPTION
    }
}

fn point_rows_bytes(points: &[Vec<f64>]) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(&(points.len() as u64).to_be_bytes());
    let dimension = points.first().map_or(0, Vec::len);
    output.extend_from_slice(&(dimension as u64).to_be_bytes());
    for point in points {
        for coordinate in point {
            output.extend_from_slice(&coordinate.to_bits().to_be_bytes());
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_cloud_has_one_facade_identity_and_affine_rank() {
        let points = PointCloud::new(&[
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ])
        .unwrap();

        assert_eq!(points.affine_rank(), 2);
        assert_eq!(PointCloud::TYPE_ID, "noether/geometry/PointCloud@1");
        assert_eq!(
            PointCloud::describe().python_path,
            Some("noether.geometry.PointCloud")
        );
        assert_eq!(points.canonical_bytes(), points.clone().canonical_bytes());
    }

    #[test]
    fn triangulations_are_typed_facade_results() {
        let points = PointCloud::new(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ])
        .unwrap();
        let ordinary = delaunay(&points).unwrap();
        let weighted = regular_triangulation(&points, &[0.0; 4]).unwrap();

        assert_eq!(ordinary.n_cells(), 2);
        assert!(ordinary.weights().is_none());
        assert_eq!(weighted.weights(), Some(&[0.0; 4][..]));
        assert_eq!(Triangulation::TYPE_ID, "noether/geometry/Triangulation@1");
        assert_ne!(
            ordinary.canonical_bytes().unwrap(),
            weighted.canonical_bytes().unwrap()
        );
        assert_eq!(
            serde_json::from_str::<Triangulation>(&serde_json::to_string(&ordinary).unwrap())
                .unwrap(),
            ordinary
        );
        assert_eq!(
            serde_json::from_str::<PointCloud>(&serde_json::to_string(&points).unwrap()).unwrap(),
            points
        );
    }

    #[test]
    fn exact_polytopes_use_flint_rationals_and_recompute_json() {
        let rational = |value: &str| value.parse::<Rational>().unwrap();
        let square = vec![
            vec![rational("0"), rational("0")],
            vec![rational("1"), rational("0")],
            vec![rational("0"), rational("1")],
            vec![rational("1"), rational("1")],
            vec![rational("1/2"), rational("1/2")],
        ];
        let polytope = convex_hull(&square).unwrap();

        assert_eq!(polytope.dimension(), 2);
        assert_eq!(polytope.vertices().len(), 4);
        assert_eq!(polytope.facets().unwrap().len(), 4);
        assert_eq!(polytope.f_vector(), vec![1, 4, 4, 1]);
        assert_eq!(Polytope::TYPE_ID, "noether/geometry/Polytope@1");
        polytope.verify().unwrap();
        assert_eq!(
            Polytope::from_json(&polytope.to_json().unwrap()).unwrap(),
            polytope
        );

        let normals = vec![
            vec![rational("-1"), rational("0")],
            vec![rational("1"), rational("0")],
            vec![rational("0"), rational("-1")],
            vec![rational("0"), rational("1")],
        ];
        let rhs = vec![rational("0"), rational("1"), rational("0"), rational("1")];
        let from_halfspaces = Polytope::from_halfspaces(&normals, &rhs).unwrap();
        assert_eq!(from_halfspaces, polytope);
    }

    #[test]
    fn lower_dimensional_polytopes_do_not_mislabel_reduced_facets() {
        let points = vec![
            vec![Rational::from(0), Rational::from(0), Rational::from(0)],
            vec![Rational::from(1), Rational::from(0), Rational::from(0)],
        ];
        let segment = convex_hull(&points).unwrap();

        assert_eq!(segment.dimension(), 1);
        assert!(!segment.is_full_dimensional());
        assert!(segment
            .facets()
            .unwrap_err()
            .reason()
            .contains("full-dimensional"));
        assert_eq!(segment.f_vector(), vec![1, 2, 1]);
    }

    #[test]
    fn invalid_weighted_geometry_fails_closed() {
        let points = PointCloud::new(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let error = regular_triangulation(&points, &[0.0, f64::NAN, 0.0]).unwrap_err();
        assert!(error.reason().contains("finite"));
    }
}
