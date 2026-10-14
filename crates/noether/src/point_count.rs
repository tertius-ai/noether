// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Checkable transcripts for deterministic finite-field point counts.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::commalg::{Scheme, SchemeError};
use crate::numthy::FiniteField;
use crate::{Integer, NoetherError, SemanticType, TypeDescription};

pub(crate) const CHECKSUM_OFFSET: u64 = 0xcbf29ce484222325;
const CHECKSUM_PRIME: u64 = 0x00000100000001b3;

static POINT_COUNT_DESCRIPTION: TypeDescription = TypeDescription {
    semantic_id: "noether/numthy/PointCountCertificate@1",
    rust_path: "noether::numthy::PointCountCertificate",
    python_path: Some("noether.numthy.PointCountCertificate"),
};

/// Deterministic coverage and checksum transcript for a finite-field point count.
#[derive(Clone, PartialEq, Eq)]
pub struct PointCountCertificate {
    model_checksum: u64,
    candidate_count: u64,
    shard_candidate_counts: Vec<u64>,
    shard_point_counts: Vec<u64>,
    shard_checksums: Vec<u64>,
}

impl PointCountCertificate {
    pub(crate) fn from_verified_parts(
        model_checksum: u64,
        candidate_count: u64,
        shard_candidate_counts: Vec<u64>,
        shard_point_counts: Vec<u64>,
        shard_checksums: Vec<u64>,
    ) -> Result<Self, SchemeError> {
        let certificate = Self {
            model_checksum,
            candidate_count,
            shard_candidate_counts,
            shard_point_counts,
            shard_checksums,
        };
        certificate
            .verify_structure()
            .map_err(SchemeError::InvalidEncoding)?;
        Ok(certificate)
    }

    /// Stable checksum of the exact scheme presentation.
    pub fn model_checksum(&self) -> u64 {
        self.model_checksum
    }

    /// Number of canonical ambient candidates covered.
    pub fn candidate_count(&self) -> u64 {
        self.candidate_count
    }

    /// Number of deterministic disjoint shards.
    pub fn shard_count(&self) -> usize {
        self.shard_checksums.len()
    }

    /// Candidate counts scanned by each shard.
    pub fn shard_candidate_counts(&self) -> &[u64] {
        &self.shard_candidate_counts
    }

    /// Rational-point counts found by each shard.
    pub fn shard_point_counts(&self) -> &[u64] {
        &self.shard_point_counts
    }

    /// Stable evaluation checksums for each shard.
    pub fn shard_checksums(&self) -> &[u64] {
        &self.shard_checksums
    }

    /// Total rational-point count certified by the shards.
    pub fn point_count(&self) -> Integer {
        Integer::from(
            checked_sum(&self.shard_point_counts)
                .expect("a validated point-count transcript has a bounded total"),
        )
    }

    /// Re-run the same deterministic partition and compare every transcript field.
    pub fn verify(&self, scheme: &Scheme<FiniteField>) -> Result<(), SchemeError> {
        self.verify_structure()
            .map_err(SchemeError::InvalidEncoding)?;
        let (_, rebuilt) = scheme.point_count_partitioned(self.shard_count())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(SchemeError::InvalidEncoding(
                "point-count transcript does not match the supplied scheme".to_owned(),
            ))
        }
    }

    /// Serialize the deterministic transcript.
    pub fn to_json(&self) -> Result<String, NoetherError> {
        serde_json::to_string(&PointCountCertificateWire::from(self))
            .map_err(|error| NoetherError::Serialization(error.to_string()))
    }

    /// Restore and structurally validate a deterministic transcript.
    pub fn from_json(value: &str) -> Result<Self, NoetherError> {
        let wire = serde_json::from_str::<PointCountCertificateWire>(value)
            .map_err(|error| NoetherError::Serialization(error.to_string()))?;
        if wire.type_id != Self::TYPE_ID {
            return Err(NoetherError::Serialization(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let certificate = Self {
            model_checksum: wire.model_checksum,
            candidate_count: wire.candidate_count,
            shard_candidate_counts: wire.shard_candidate_counts,
            shard_point_counts: wire.shard_point_counts,
            shard_checksums: wire.shard_checksums,
        };
        certificate
            .verify_structure()
            .map_err(NoetherError::Serialization)?;
        Ok(certificate)
    }

    fn verify_structure(&self) -> Result<(), String> {
        let shard_count = self.shard_checksums.len();
        if shard_count == 0
            || self.shard_candidate_counts.len() != shard_count
            || self.shard_point_counts.len() != shard_count
        {
            return Err("point-count shard arrays have incompatible lengths".to_owned());
        }
        if checked_sum(&self.shard_candidate_counts) != Some(self.candidate_count) {
            return Err("point-count shards do not cover every canonical candidate".to_owned());
        }
        if self
            .shard_point_counts
            .iter()
            .zip(&self.shard_candidate_counts)
            .any(|(points, candidates)| points > candidates)
        {
            return Err("a shard contains more points than candidates".to_owned());
        }
        Ok(())
    }
}

fn checked_sum(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(0_u64, |sum, value| sum.checked_add(*value))
}

impl SemanticType for PointCountCertificate {
    const TYPE_ID: &'static str = "noether/numthy/PointCountCertificate@1";

    fn canonical_bytes(&self) -> Result<Vec<u8>, NoetherError> {
        self.to_json().map(String::into_bytes)
    }

    fn describe() -> &'static TypeDescription {
        &POINT_COUNT_DESCRIPTION
    }
}

impl fmt::Debug for PointCountCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PointCountCertificate")
            .field("candidate_count", &self.candidate_count)
            .field("point_count", &self.point_count())
            .field("shard_count", &self.shard_count())
            .finish()
    }
}

impl fmt::Display for PointCountCertificate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PointCountCertificate(points={}, candidates={}, shards={})",
            self.point_count(),
            self.candidate_count,
            self.shard_count()
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PointCountCertificateWire {
    #[serde(rename = "type")]
    type_id: String,
    model_checksum: u64,
    candidate_count: u64,
    shard_candidate_counts: Vec<u64>,
    shard_point_counts: Vec<u64>,
    shard_checksums: Vec<u64>,
}

impl From<&PointCountCertificate> for PointCountCertificateWire {
    fn from(certificate: &PointCountCertificate) -> Self {
        Self {
            type_id: PointCountCertificate::TYPE_ID.to_owned(),
            model_checksum: certificate.model_checksum,
            candidate_count: certificate.candidate_count,
            shard_candidate_counts: certificate.shard_candidate_counts.clone(),
            shard_point_counts: certificate.shard_point_counts.clone(),
            shard_checksums: certificate.shard_checksums.clone(),
        }
    }
}

impl Serialize for PointCountCertificate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        PointCountCertificateWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PointCountCertificate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = PointCountCertificateWire::deserialize(deserializer)?;
        if wire.type_id != Self::TYPE_ID {
            return Err(serde::de::Error::custom(format!(
                "expected type {}, received {}",
                Self::TYPE_ID,
                wire.type_id
            )));
        }
        let certificate = Self {
            model_checksum: wire.model_checksum,
            candidate_count: wire.candidate_count,
            shard_candidate_counts: wire.shard_candidate_counts,
            shard_point_counts: wire.shard_point_counts,
            shard_checksums: wire.shard_checksums,
        };
        certificate
            .verify_structure()
            .map_err(serde::de::Error::custom)?;
        Ok(certificate)
    }
}

pub(crate) fn checksum_bytes(bytes: &[u8]) -> u64 {
    let mut checksum = CHECKSUM_OFFSET;
    for byte in bytes {
        checksum ^= u64::from(*byte);
        checksum = checksum.wrapping_mul(CHECKSUM_PRIME);
    }
    checksum
}

pub(crate) fn checksum_candidate(checksum: &mut u64, index: u64, is_point: bool) {
    for byte in index.to_be_bytes() {
        *checksum ^= u64::from(byte);
        *checksum = checksum.wrapping_mul(CHECKSUM_PRIME);
    }
    *checksum ^= u64::from(is_point);
    *checksum = checksum.wrapping_mul(CHECKSUM_PRIME);
}
