// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checked scalar extension for polynomial and scheme presentations.

use core::fmt;

use super::scheme::{affine_closed_subscheme, projective_closed_subscheme};
use super::{
    AffineSpace, GroebnerError, Ideal, Polynomial, PolynomialHomomorphism,
    PolynomialHomomorphismError, PolynomialRing, PolynomialRingHomomorphism, ProjectiveSpace,
    Scheme, SchemeError, SchemePoint,
};
use crate::{
    CoefficientMap, CoefficientMappingError, CommutativeRing, ContextMismatchError, Field,
    IntegerRing,
};

/// Failure to extend a polynomial or geometric presentation along a checked
/// coefficient map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseChangeError {
    /// The coefficient map did not start at the presented base context.
    ContextMismatch(ContextMismatchError),
    /// Mapping or reconstructing coefficients failed.
    CoefficientMapping(CoefficientMappingError),
    /// Constructing the induced polynomial-ring homomorphism failed.
    PolynomialHomomorphism(PolynomialHomomorphismError),
    /// Recanonicalizing an extended polynomial ideal failed.
    Groebner(GroebnerError),
    /// Reconstructing an ambient space, scheme, or point failed.
    Scheme(SchemeError),
}

impl fmt::Display for BaseChangeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextMismatch(error) => error.fmt(formatter),
            Self::CoefficientMapping(error) => error.fmt(formatter),
            Self::PolynomialHomomorphism(error) => error.fmt(formatter),
            Self::Groebner(error) => error.fmt(formatter),
            Self::Scheme(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for BaseChangeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextMismatch(error) => Some(error),
            Self::CoefficientMapping(error) => Some(error),
            Self::PolynomialHomomorphism(error) => Some(error),
            Self::Groebner(error) => Some(error),
            Self::Scheme(error) => Some(error),
        }
    }
}

impl From<ContextMismatchError> for BaseChangeError {
    fn from(value: ContextMismatchError) -> Self {
        Self::ContextMismatch(value)
    }
}

impl From<CoefficientMappingError> for BaseChangeError {
    fn from(value: CoefficientMappingError) -> Self {
        Self::CoefficientMapping(value)
    }
}

impl From<PolynomialHomomorphismError> for BaseChangeError {
    fn from(value: PolynomialHomomorphismError) -> Self {
        Self::PolynomialHomomorphism(value)
    }
}

impl From<GroebnerError> for BaseChangeError {
    fn from(value: GroebnerError) -> Self {
        Self::Groebner(value)
    }
}

impl From<SchemeError> for BaseChangeError {
    fn from(value: SchemeError) -> Self {
        Self::Scheme(value)
    }
}

impl<R: CommutativeRing + fmt::Display> PolynomialRing<R> {
    /// Extend the coefficient context while preserving variables, term order,
    /// and grading exactly.
    pub fn base_change<M>(&self, map: &M) -> Result<PolynomialRing<M::Target>, BaseChangeError>
    where
        M: CoefficientMap<Source = R>,
    {
        if self.coefficient_ring() != map.source() {
            return Err(ContextMismatchError {
                expected: self.coefficient_ring().to_string(),
                actual: map.source().to_string(),
            }
            .into());
        }
        PolynomialRing::with_grading(
            map.target().clone(),
            self.variables().iter().cloned(),
            self.term_order(),
            self.grading().clone(),
        )
        .map_err(|error| CoefficientMappingError::TargetContext(error.to_string()).into())
    }

    /// The canonical coefficientwise homomorphism into the base-changed
    /// polynomial ring.
    pub fn base_change_homomorphism<M>(
        &self,
        map: M,
    ) -> Result<PolynomialRingHomomorphism<R, M::Target, M>, BaseChangeError>
    where
        M: CoefficientMap<Source = R>,
    {
        let target = self.base_change(&map)?;
        let images = target
            .generators()
            .map_err(PolynomialHomomorphismError::from)?;
        self.homomorphism(&target, map, images).map_err(Into::into)
    }
}

impl<R: CommutativeRing + fmt::Display> Polynomial<R> {
    /// Extend every coefficient while preserving the complete polynomial-ring
    /// presentation.
    pub fn base_change<M>(&self, map: &M) -> Result<Polynomial<M::Target>, BaseChangeError>
    where
        M: CoefficientMap<Source = R>,
    {
        self.map_coefficients(map).map_err(Into::into)
    }
}

impl<R: Field + fmt::Display> Ideal<PolynomialRing<R>> {
    /// Extend generators coefficientwise and recompute the canonical reduced
    /// Groebner basis over the target field.
    pub fn base_change<M>(
        &self,
        map: &M,
    ) -> Result<Ideal<PolynomialRing<M::Target>>, BaseChangeError>
    where
        M: CoefficientMap<Source = R>,
        M::Target: Field,
    {
        let target = self.ring().base_change(map)?;
        let generators = self
            .generators()
            .iter()
            .map(|generator| generator.base_change(map))
            .collect::<Result<Vec<_>, _>>()?;
        target.ideal(generators).map_err(Into::into)
    }
}

impl Ideal<PolynomialRing<IntegerRing>> {
    /// Extend an integral polynomial ideal to an exact field and recompute
    /// the target's canonical reduced Gröbner basis.
    pub fn base_change<M>(
        &self,
        map: &M,
    ) -> Result<Ideal<PolynomialRing<M::Target>>, BaseChangeError>
    where
        M: CoefficientMap<Source = IntegerRing>,
        M::Target: Field,
    {
        let target = self.ring().base_change(map)?;
        let generators = self
            .generators()
            .iter()
            .map(|generator| generator.base_change(map))
            .collect::<Result<Vec<_>, _>>()?;
        target.ideal(generators).map_err(Into::into)
    }
}

impl<S, T, M> PolynomialHomomorphism<S, PolynomialRing<T>, M>
where
    S: CommutativeRing + fmt::Display,
    T: Field,
    M: CoefficientMap<Source = S, Target = PolynomialRing<T>>,
{
    /// Extend an ideal along this homomorphism and recanonicalize its image in
    /// the target polynomial ring.
    pub fn extend_ideal(
        &self,
        ideal: &Ideal<PolynomialRing<S>>,
    ) -> Result<Ideal<PolynomialRing<T>>, BaseChangeError> {
        if ideal.ring() != self.source() {
            return Err(ContextMismatchError {
                expected: self.source().to_string(),
                actual: ideal.ring().to_string(),
            }
            .into());
        }
        let generators = ideal
            .generators()
            .iter()
            .map(|generator| self.apply(generator).map_err(BaseChangeError::from))
            .collect::<Result<Vec<_>, _>>()?;
        self.target().ideal(generators).map_err(Into::into)
    }
}

impl<R: CommutativeRing + fmt::Display> AffineSpace<R> {
    /// Extend the base while preserving the exact affine coordinate-ring
    /// presentation.
    pub fn base_change<M>(&self, map: &M) -> Result<AffineSpace<M::Target>, BaseChangeError>
    where
        M: CoefficientMap<Source = R>,
    {
        Ok(AffineSpace::from_coordinate_ring(
            self.coordinate_ring().base_change(map)?,
        ))
    }
}

impl<R: CommutativeRing + fmt::Display> ProjectiveSpace<R> {
    /// Extend the base while preserving the exact homogeneous coordinate-ring
    /// presentation.
    pub fn base_change<M>(&self, map: &M) -> Result<ProjectiveSpace<M::Target>, BaseChangeError>
    where
        M: CoefficientMap<Source = R>,
    {
        ProjectiveSpace::from_coordinate_ring(self.coordinate_ring().base_change(map)?)
            .map_err(Into::into)
    }
}

impl<R: Field + fmt::Display> Scheme<R> {
    /// Extend a closed affine or projective presentation along a checked field
    /// map, preserving its ambient kind and presentation data.
    pub fn base_change<M>(&self, map: &M) -> Result<Scheme<M::Target>, BaseChangeError>
    where
        M: CoefficientMap<Source = R>,
        M::Target: Field,
    {
        let defining_ideal = self.defining_ideal().base_change(map)?;
        if let Some(ambient) = self.affine_ambient() {
            let ambient = ambient.base_change(map)?;
            return affine_closed_subscheme(&ambient, defining_ideal).map_err(Into::into);
        }
        let ambient = self
            .projective_ambient()
            .ok_or(SchemeError::UnsupportedGeometry(
                "a scheme has no affine or projective ambient presentation",
            ))?
            .base_change(map)?;
        projective_closed_subscheme(&ambient, defining_ideal).map_err(Into::into)
    }
}

impl<R: Field + fmt::Display> SchemePoint<R> {
    /// Extend the point and its complete scheme presentation along a checked
    /// field map.
    pub fn base_change<M>(&self, map: &M) -> Result<SchemePoint<M::Target>, BaseChangeError>
    where
        M: CoefficientMap<Source = R>,
        M::Target: Field,
    {
        let scheme = self.scheme().base_change(map)?;
        let coordinates = self
            .coordinates()
            .iter()
            .map(|coordinate| {
                map.apply(coordinate)
                    .map_err(|error| CoefficientMappingError::Coefficient(error.to_string()))
                    .map_err(BaseChangeError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        scheme.point(coordinates).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commalg::MonomialOrder;
    use crate::numthy::FiniteField;
    use crate::{Integer, Ring, Q, Z};

    #[test]
    fn polynomial_base_change_preserves_the_complete_presentation() {
        let source = PolynomialRing::with_grading(
            Z,
            ["x", "y"],
            MonomialOrder::elimination(2, [0]).unwrap(),
            super::super::WeightedGrading::new(vec![2_u64, 3_u64]).unwrap(),
        )
        .unwrap();
        let [x, y] = source.generators().unwrap().try_into().unwrap();
        let value = source
            .add(
                &source.mul(&x, &x).unwrap(),
                &y.scalar_mul(&Integer::from(2)).unwrap(),
            )
            .unwrap();
        let map = Z.rational_embedding();
        let target = source.base_change(&map).unwrap();
        let mapped = value.base_change(&map).unwrap();

        assert_eq!(target.coefficient_ring(), &Q);
        assert_eq!(target.variables(), source.variables());
        assert_eq!(target.term_order(), source.term_order());
        assert_eq!(target.grading(), source.grading());
        assert_eq!(mapped.ring(), &target);
        assert_eq!(
            source
                .base_change_homomorphism(map)
                .unwrap()
                .apply(&value)
                .unwrap(),
            mapped
        );
    }

    #[test]
    fn ideals_schemes_and_points_recanonicalize_over_the_target_field() {
        let prime = FiniteField::prime(5_u8).unwrap();
        let extension = FiniteField::with_modulus(5_u64, vec![2_u64, 0_u64, 1_u64]).unwrap();
        let map = extension.prime_subfield_embedding();
        let ambient = AffineSpace::with_names(prime.clone(), 2, ["x", "y"]).unwrap();
        let ring = ambient.coordinate_ring();
        let [x, y] = ring.generators().unwrap().try_into().unwrap();
        let equation = ring.sub(&y, &ring.mul(&x, &x).unwrap()).unwrap();
        let scheme = ambient
            .closed_subscheme(ring.ideal([equation]).unwrap())
            .unwrap();
        let point = scheme.point([prime.element(2), prime.element(4)]).unwrap();

        let changed = scheme.base_change(&map).unwrap();
        let changed_point = point.base_change(&map).unwrap();
        assert_eq!(changed.base(), &extension);
        assert_eq!(changed.coordinate_ring().variables(), &["x", "y"]);
        assert!(changed.is_affine());
        changed.defining_ideal().verify().unwrap();
        assert_eq!(changed_point.scheme(), &changed);
        changed_point.verify().unwrap();
    }
}
