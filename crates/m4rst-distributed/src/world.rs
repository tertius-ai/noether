// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Explicit MPI world ownership and collective agreement helpers.

use std::fmt;
use std::mem::{size_of, size_of_val};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Duration;

use mpi::collective::SystemOperation;
use mpi::environment::{self, Universe};
use mpi::topology::{Communicator, SimpleCommunicator};
use mpi::traits::*;

use crate::DistributedError;

struct WorldInner {
    universe: Universe,
    mpi_rank: i32,
    size: i32,
    library_version: String,
    thread_support: MpiThreadSupport,
    mpi_lock: Mutex<()>,
    communication: CommunicationTelemetry,
    matmul_phases: MatmulPhaseTelemetry,
}

#[derive(Default)]
struct CommunicationTelemetry {
    reduction_calls: AtomicU64,
    reduction_payload_bytes: AtomicU64,
    broadcast_calls: AtomicU64,
    broadcast_payload_bytes: AtomicU64,
    communicator_split_calls: AtomicU64,
    barrier_calls: AtomicU64,
    sent_messages: AtomicU64,
    sent_payload_bytes: AtomicU64,
    received_messages: AtomicU64,
    received_payload_bytes: AtomicU64,
}

#[derive(Default)]
struct TimedPhaseTelemetry {
    calls: AtomicU64,
    elapsed_ns: AtomicU64,
}

#[derive(Default)]
struct MatmulPhaseTelemetry {
    output_preparation: TimedPhaseTelemetry,
    communicator_setup: TimedPhaseTelemetry,
    right_panel_preparation: TimedPhaseTelemetry,
    right_panel_broadcast: TimedPhaseTelemetry,
    left_panel_preparation: TimedPhaseTelemetry,
    left_panel_broadcast: TimedPhaseTelemetry,
    local_kernel: TimedPhaseTelemetry,
    output_accumulation: TimedPhaseTelemetry,
}

static ACTIVE_WORLD: OnceLock<Mutex<Weak<WorldInner>>> = OnceLock::new();

fn active_world_slot() -> &'static Mutex<Weak<WorldInner>> {
    ACTIVE_WORLD.get_or_init(|| Mutex::new(Weak::new()))
}

/// Read-only facts about the currently retained MPI world.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveMpiRuntime {
    /// MPI implementation/version string.
    pub library_version: String,
    /// Number of processes in the world.
    pub world_size: usize,
    /// This process's MPI rank.
    pub mpi_rank: usize,
    /// Thread support supplied by MPI.
    pub thread_support: MpiThreadSupport,
}

/// Per-process application-level MPI communication totals.
///
/// Payload bytes describe buffers passed to MPI by this crate, not MPI
/// protocol overhead or physical network traffic. Collective calls are
/// counted once per participating process.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommunicationCounters {
    /// Reduction calls made by this process.
    pub reduction_calls: u64,
    /// Input payload bytes passed to reductions by this process.
    pub reduction_payload_bytes: u64,
    /// Broadcast calls entered by this process.
    pub broadcast_calls: u64,
    /// Payload bytes passed to broadcasts by this process.
    pub broadcast_payload_bytes: u64,
    /// Communicator-split collectives entered by this process.
    pub communicator_split_calls: u64,
    /// Barrier calls entered by this process.
    pub barrier_calls: u64,
    /// Point-to-point messages sent by this process.
    pub sent_messages: u64,
    /// Point-to-point payload bytes sent by this process.
    pub sent_payload_bytes: u64,
    /// Point-to-point messages received by this process.
    pub received_messages: u64,
    /// Point-to-point payload bytes received by this process.
    pub received_payload_bytes: u64,
}

impl CommunicationCounters {
    /// Return the saturating difference from an earlier snapshot.
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            reduction_calls: self.reduction_calls.saturating_sub(earlier.reduction_calls),
            reduction_payload_bytes: self
                .reduction_payload_bytes
                .saturating_sub(earlier.reduction_payload_bytes),
            broadcast_calls: self.broadcast_calls.saturating_sub(earlier.broadcast_calls),
            broadcast_payload_bytes: self
                .broadcast_payload_bytes
                .saturating_sub(earlier.broadcast_payload_bytes),
            communicator_split_calls: self
                .communicator_split_calls
                .saturating_sub(earlier.communicator_split_calls),
            barrier_calls: self.barrier_calls.saturating_sub(earlier.barrier_calls),
            sent_messages: self.sent_messages.saturating_sub(earlier.sent_messages),
            sent_payload_bytes: self
                .sent_payload_bytes
                .saturating_sub(earlier.sent_payload_bytes),
            received_messages: self
                .received_messages
                .saturating_sub(earlier.received_messages),
            received_payload_bytes: self
                .received_payload_bytes
                .saturating_sub(earlier.received_payload_bytes),
        }
    }
}

/// Rank-local call and elapsed-time totals for one timed phase.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimedPhaseCounters {
    /// Number of phase executions by this process.
    pub calls: u64,
    /// Saturating elapsed nanoseconds observed by this process.
    pub elapsed_ns: u64,
}

impl TimedPhaseCounters {
    fn since(self, earlier: Self) -> Self {
        Self {
            calls: self.calls.saturating_sub(earlier.calls),
            elapsed_ns: self.elapsed_ns.saturating_sub(earlier.elapsed_ns),
        }
    }
}

/// Per-process elapsed-time totals for the distributed matmul phases.
///
/// These diagnostic counters are rank-local. Values from different ranks may
/// overlap in wall-clock time and therefore must not be summed as an operation
/// duration.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatmulPhaseCounters {
    /// Output allocation, initialization, or reset work.
    pub output_preparation: TimedPhaseCounters,
    /// Process-row and process-column communicator setup.
    pub communicator_setup: TimedPhaseCounters,
    /// Right-panel allocation and root-side extraction.
    pub right_panel_preparation: TimedPhaseCounters,
    /// Blocking right-panel broadcasts.
    pub right_panel_broadcast: TimedPhaseCounters,
    /// Left-panel allocation and root-side extraction.
    pub left_panel_preparation: TimedPhaseCounters,
    /// Blocking left-panel broadcasts.
    pub left_panel_broadcast: TimedPhaseCounters,
    /// Rank-local packed matrix multiplication kernels.
    pub local_kernel: TimedPhaseCounters,
    /// Rank-local XOR accumulation into output tiles.
    pub output_accumulation: TimedPhaseCounters,
}

impl MatmulPhaseCounters {
    /// Return the saturating difference from an earlier snapshot.
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            output_preparation: self.output_preparation.since(earlier.output_preparation),
            communicator_setup: self.communicator_setup.since(earlier.communicator_setup),
            right_panel_preparation: self
                .right_panel_preparation
                .since(earlier.right_panel_preparation),
            right_panel_broadcast: self
                .right_panel_broadcast
                .since(earlier.right_panel_broadcast),
            left_panel_preparation: self
                .left_panel_preparation
                .since(earlier.left_panel_preparation),
            left_panel_broadcast: self
                .left_panel_broadcast
                .since(earlier.left_panel_broadcast),
            local_kernel: self.local_kernel.since(earlier.local_kernel),
            output_accumulation: self.output_accumulation.since(earlier.output_accumulation),
        }
    }
}

pub(crate) enum MatmulPhase {
    OutputPreparation,
    CommunicatorSetup,
    RightPanelPreparation,
    RightPanelBroadcast,
    LeftPanelPreparation,
    LeftPanelBroadcast,
    LocalKernel,
    OutputAccumulation,
}

pub(crate) struct SerializedCommunicator<'a> {
    _guard: MutexGuard<'a, ()>,
    communicator: SimpleCommunicator,
}

impl Deref for SerializedCommunicator<'_> {
    type Target = SimpleCommunicator;

    fn deref(&self) -> &Self::Target {
        &self.communicator
    }
}

/// MPI thread-support level supplied for this process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpiThreadSupport {
    /// Only one application thread may exist.
    Single,
    /// Only the initializing thread may make MPI calls.
    Funneled,
    /// Any thread may make MPI calls, but calls must be serialized.
    Serialized,
    /// Multiple threads may make MPI calls concurrently.
    Multiple,
}

impl From<mpi::environment::Threading> for MpiThreadSupport {
    fn from(value: mpi::environment::Threading) -> Self {
        match value {
            mpi::environment::Threading::Single => Self::Single,
            mpi::environment::Threading::Funneled => Self::Funneled,
            mpi::environment::Threading::Serialized => Self::Serialized,
            mpi::environment::Threading::Multiple => Self::Multiple,
        }
    }
}

/// Explicit owner of one initialized MPI world.
///
/// Initialization requests `MPI_THREAD_SERIALIZED`: MPI calls may originate
/// from different application threads but never concurrently. The world and
/// every matrix retaining it are process-local values.
#[derive(Clone)]
pub struct World {
    inner: Arc<WorldInner>,
}

impl World {
    /// Initialize MPI exactly once for this process.
    pub fn initialize() -> Result<Self, DistributedError> {
        let (universe, provided) =
            mpi::initialize_with_threading(mpi::environment::Threading::Serialized)
                .ok_or(DistributedError::MpiAlreadyInitialized)?;
        if provided < mpi::environment::Threading::Serialized {
            return Err(DistributedError::InsufficientThreadSupport);
        }
        let library_version = environment::library_version()
            .map_err(|error| DistributedError::CollectiveFailure(error.to_string()))?
            .trim_end_matches(['\0', '\n', '\r'])
            .to_owned();
        let communicator = universe.world();
        let mpi_rank = communicator.rank();
        let size = communicator.size();
        let inner = Arc::new(WorldInner {
            universe,
            mpi_rank,
            size,
            library_version,
            thread_support: provided.into(),
            mpi_lock: Mutex::new(()),
            communication: CommunicationTelemetry::default(),
            matmul_phases: MatmulPhaseTelemetry::default(),
        });
        *active_world_slot()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::downgrade(&inner);
        Ok(Self { inner })
    }

    pub(crate) fn communicator(&self) -> SerializedCommunicator<'_> {
        let guard = self
            .inner
            .mpi_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        SerializedCommunicator {
            _guard: guard,
            communicator: self.inner.universe.world(),
        }
    }

    /// This process's MPI rank.
    #[must_use]
    pub fn mpi_rank(&self) -> i32 {
        self.inner.mpi_rank
    }

    /// Number of MPI processes in this world.
    #[must_use]
    pub fn size(&self) -> i32 {
        self.inner.size
    }

    /// Whether this process is rank zero.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.mpi_rank() == 0
    }

    /// MPI implementation/version string reported by the linked runtime.
    #[must_use]
    pub fn library_version(&self) -> &str {
        &self.inner.library_version
    }

    /// Thread support supplied by the initialized MPI runtime.
    #[must_use]
    pub fn thread_support(&self) -> MpiThreadSupport {
        self.inner.thread_support
    }

    /// Snapshot this process's application-level MPI communication totals.
    #[doc(hidden)]
    #[must_use]
    pub fn communication_counters(&self) -> CommunicationCounters {
        let counters = &self.inner.communication;
        CommunicationCounters {
            reduction_calls: counters.reduction_calls.load(Ordering::Relaxed),
            reduction_payload_bytes: counters.reduction_payload_bytes.load(Ordering::Relaxed),
            broadcast_calls: counters.broadcast_calls.load(Ordering::Relaxed),
            broadcast_payload_bytes: counters.broadcast_payload_bytes.load(Ordering::Relaxed),
            communicator_split_calls: counters.communicator_split_calls.load(Ordering::Relaxed),
            barrier_calls: counters.barrier_calls.load(Ordering::Relaxed),
            sent_messages: counters.sent_messages.load(Ordering::Relaxed),
            sent_payload_bytes: counters.sent_payload_bytes.load(Ordering::Relaxed),
            received_messages: counters.received_messages.load(Ordering::Relaxed),
            received_payload_bytes: counters.received_payload_bytes.load(Ordering::Relaxed),
        }
    }

    /// Snapshot this process's distributed-matmul phase totals.
    #[doc(hidden)]
    #[must_use]
    pub fn matmul_phase_counters(&self) -> MatmulPhaseCounters {
        let phases = &self.inner.matmul_phases;
        MatmulPhaseCounters {
            output_preparation: snapshot_phase(&phases.output_preparation),
            communicator_setup: snapshot_phase(&phases.communicator_setup),
            right_panel_preparation: snapshot_phase(&phases.right_panel_preparation),
            right_panel_broadcast: snapshot_phase(&phases.right_panel_broadcast),
            left_panel_preparation: snapshot_phase(&phases.left_panel_preparation),
            left_panel_broadcast: snapshot_phase(&phases.left_panel_broadcast),
            local_kernel: snapshot_phase(&phases.local_kernel),
            output_accumulation: snapshot_phase(&phases.output_accumulation),
        }
    }

    pub(crate) fn record_matmul_phase(&self, phase: MatmulPhase, elapsed: Duration) {
        let phases = &self.inner.matmul_phases;
        let counters = match phase {
            MatmulPhase::OutputPreparation => &phases.output_preparation,
            MatmulPhase::CommunicatorSetup => &phases.communicator_setup,
            MatmulPhase::RightPanelPreparation => &phases.right_panel_preparation,
            MatmulPhase::RightPanelBroadcast => &phases.right_panel_broadcast,
            MatmulPhase::LeftPanelPreparation => &phases.left_panel_preparation,
            MatmulPhase::LeftPanelBroadcast => &phases.left_panel_broadcast,
            MatmulPhase::LocalKernel => &phases.local_kernel,
            MatmulPhase::OutputAccumulation => &phases.output_accumulation,
        };
        add_counter(&counters.calls, 1);
        add_u64_counter(
            &counters.elapsed_ns,
            u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        );
    }

    pub(crate) fn record_reduction(&self, payload_bytes: usize) {
        add_counter(&self.inner.communication.reduction_calls, 1);
        add_counter(
            &self.inner.communication.reduction_payload_bytes,
            payload_bytes,
        );
    }

    pub(crate) fn record_broadcast(&self, payload_bytes: usize) {
        add_counter(&self.inner.communication.broadcast_calls, 1);
        add_counter(
            &self.inner.communication.broadcast_payload_bytes,
            payload_bytes,
        );
    }

    pub(crate) fn record_barrier(&self) {
        add_counter(&self.inner.communication.barrier_calls, 1);
    }

    pub(crate) fn record_communicator_split(&self) {
        add_counter(&self.inner.communication.communicator_split_calls, 1);
    }

    pub(crate) fn record_send(&self, payload_bytes: usize) {
        add_counter(&self.inner.communication.sent_messages, 1);
        add_counter(&self.inner.communication.sent_payload_bytes, payload_bytes);
    }

    pub(crate) fn record_receive(&self, payload_bytes: usize) {
        add_counter(&self.inner.communication.received_messages, 1);
        add_counter(
            &self.inner.communication.received_payload_bytes,
            payload_bytes,
        );
    }

    pub(crate) fn same_world(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn validate_rank(&self, rank: i32) -> Result<(), DistributedError> {
        if (0..self.size()).contains(&rank) {
            Ok(())
        } else {
            Err(DistributedError::InvalidMpiRank {
                rank,
                world_size: self.size(),
            })
        }
    }

    /// Collectively validate and agree on one MPI rank before any rooted
    /// collective or point-to-point movement begins.
    pub(crate) fn agree_rank(&self, rank: i32, name: &'static str) -> Result<(), DistributedError> {
        if !self.collective_all((0..self.size()).contains(&rank)) {
            return Err(DistributedError::InvalidMpiRank {
                rank,
                world_size: self.size(),
            });
        }
        self.agree_u64(
            u64::try_from(rank).map_err(|_| DistributedError::SizeOverflow)?,
            name,
        )
    }

    pub(crate) fn agree_u64(&self, value: u64, name: &'static str) -> Result<(), DistributedError> {
        let communicator = self.communicator();
        // rsmpi permits distinct send/receive buffer element types at the
        // Rust type level, while MPI requires them to match. Keep these
        // annotations explicit: an inferred i32 destination would let MPI
        // write eight bytes into four.
        let mut minimum = 0_u64;
        let mut maximum = 0_u64;
        communicator.all_reduce_into(&value, &mut minimum, SystemOperation::min());
        communicator.all_reduce_into(&value, &mut maximum, SystemOperation::max());
        self.record_reduction(size_of::<u64>());
        self.record_reduction(size_of::<u64>());
        if minimum == maximum {
            Ok(())
        } else {
            Err(DistributedError::CollectiveMismatch(name))
        }
    }

    /// Return whether `condition` is true on every rank.
    ///
    /// This is a low-level façade integration hook. Mathematical users should
    /// prefer the typed collective constructors, which perform their own
    /// agreement checks.
    #[doc(hidden)]
    pub fn collective_all(&self, condition: bool) -> bool {
        let communicator = self.communicator();
        let value = u8::from(condition);
        let mut result = 0_u8;
        communicator.all_reduce_into(&value, &mut result, SystemOperation::min());
        self.record_reduction(size_of::<u8>());
        result == 1
    }

    /// Gather bounded rank-local metadata bytes to `root` in MPI-rank order.
    ///
    /// This hidden integration hook is for telemetry and evidence records,
    /// not matrix data. The collective fails uniformly before point-to-point
    /// movement when any rank's record exceeds `max_bytes_per_rank`.
    #[doc(hidden)]
    pub fn gather_metadata_bytes(
        &self,
        local: &[u8],
        root: i32,
        max_bytes_per_rank: usize,
    ) -> Result<Option<Vec<Vec<u8>>>, DistributedError> {
        self.agree_rank(root, "metadata root")?;
        self.agree_u64(
            u64::try_from(max_bytes_per_rank).map_err(|_| DistributedError::SizeOverflow)?,
            "metadata byte limit",
        )?;

        let local_bytes = u64::try_from(local.len()).map_err(|_| DistributedError::SizeOverflow)?;
        let communicator = self.communicator();
        let mut maximum_bytes = 0_u64;
        communicator.all_reduce_into(&local_bytes, &mut maximum_bytes, SystemOperation::max());
        self.record_reduction(size_of::<u64>());
        drop(communicator);
        let required =
            usize::try_from(maximum_bytes).map_err(|_| DistributedError::SizeOverflow)?;
        if required > max_bytes_per_rank {
            return Err(DistributedError::MaterializationLimit {
                required,
                limit: max_bytes_per_rank,
            });
        }

        let capacity = usize::try_from(self.size()).expect("MPI world sizes are positive");
        let mut lengths = Vec::new();
        let lengths_allocation = try_reserve_items(&mut lengths, capacity);
        if !self.collective_all(lengths_allocation.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not allocate metadata lengths".to_owned(),
            ));
        }
        lengths_allocation?;
        lengths.resize(capacity, 0_u64);
        let communicator = self.communicator();
        communicator.all_gather_into(&local_bytes, &mut lengths[..]);
        self.record_reduction(size_of::<u64>());
        drop(communicator);

        let prepared = if self.mpi_rank() == root {
            (|| -> Result<Option<Vec<Vec<u8>>>, DistributedError> {
                let mut gathered = Vec::new();
                try_reserve_items(&mut gathered, capacity)?;
                for length in &lengths {
                    let length =
                        usize::try_from(*length).map_err(|_| DistributedError::SizeOverflow)?;
                    let mut bytes = Vec::new();
                    try_reserve_items(&mut bytes, length)?;
                    bytes.resize(length, 0);
                    gathered.push(bytes);
                }
                gathered[usize::try_from(root).expect("validated MPI roots are nonnegative")]
                    .copy_from_slice(local);
                Ok(Some(gathered))
            })()
        } else {
            Ok(None)
        };
        if !self.collective_all(prepared.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "the metadata root could not allocate gather buffers".to_owned(),
            ));
        }
        let mut gathered = prepared?;
        let communicator = self.communicator();
        if let Some(gathered) = &mut gathered {
            for source in 0..self.size() {
                if source != root && !gathered[source as usize].is_empty() {
                    communicator
                        .process_at_rank(source)
                        .receive_into(&mut gathered[source as usize][..]);
                    self.record_receive(gathered[source as usize].len());
                }
            }
        } else if !local.is_empty() {
            communicator.process_at_rank(root).send(local);
            self.record_send(local.len());
        }
        Ok(gathered)
    }

    pub(crate) fn xor_u64s(&self, local: &[u64]) -> Result<Vec<u64>, DistributedError> {
        let mut result = Vec::new();
        let allocation = try_reserve_items(&mut result, local.len());
        if !self.collective_all(allocation.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not allocate a reduction buffer".to_owned(),
            ));
        }
        allocation?;
        result.resize(local.len(), 0);
        let communicator = self.communicator();
        communicator.all_reduce_into(local, &mut result[..], SystemOperation::bitwise_xor());
        self.record_reduction(size_of_val(local));
        Ok(result)
    }

    pub(crate) fn min_u64(&self, local: u64) -> u64 {
        let communicator = self.communicator();
        let mut result = u64::MAX;
        communicator.all_reduce_into(&local, &mut result, SystemOperation::min());
        self.record_reduction(size_of::<u64>());
        result
    }

    pub(crate) fn min_u64s(&self, local: &[u64]) -> Result<Vec<u64>, DistributedError> {
        let mut result = Vec::new();
        let allocation = try_reserve_items(&mut result, local.len());
        if !self.collective_all(allocation.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not allocate a reduction buffer".to_owned(),
            ));
        }
        allocation?;
        result.resize(local.len(), u64::MAX);
        let communicator = self.communicator();
        communicator.all_reduce_into(local, &mut result[..], SystemOperation::min());
        self.record_reduction(size_of_val(local));
        Ok(result)
    }

    pub(crate) fn broadcast_bytes(
        &self,
        root: i32,
        bytes: Option<&[u8]>,
    ) -> Result<Vec<u8>, DistributedError> {
        self.validate_rank(root)?;
        let communicator = self.communicator();
        let root_process = communicator.process_at_rank(root);
        let mut length = if self.mpi_rank() == root {
            u64::try_from(bytes.map_or(0, <[u8]>::len))
                .map_err(|_| DistributedError::SizeOverflow)?
        } else {
            0
        };
        root_process.broadcast_into(&mut length);
        self.record_broadcast(size_of::<u64>());
        drop(communicator);
        let length = usize::try_from(length).map_err(|_| DistributedError::SizeOverflow)?;
        let result = if self.mpi_rank() == root {
            try_copy_bytes(bytes.unwrap_or_default())
        } else {
            let mut result = Vec::new();
            try_reserve_items(&mut result, length).map(|()| {
                result.resize(length, 0);
                result
            })
        };
        if !self.collective_all(result.is_ok()) {
            return Err(DistributedError::CollectiveFailure(
                "at least one rank could not allocate a broadcast buffer".to_owned(),
            ));
        }
        let mut result = result?;
        if length != 0 {
            let communicator = self.communicator();
            let root_process = communicator.process_at_rank(root);
            root_process.broadcast_into(&mut result[..]);
            self.record_broadcast(result.len());
        }
        Ok(result)
    }
}

fn try_reserve_items<T>(values: &mut Vec<T>, additional: usize) -> Result<(), DistributedError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| DistributedError::AllocationFailure {
            requested_bytes: values
                .len()
                .checked_add(additional)
                .and_then(|items| items.checked_mul(size_of::<T>())),
        })
}

fn try_copy_bytes(bytes: &[u8]) -> Result<Vec<u8>, DistributedError> {
    let mut result = Vec::new();
    try_reserve_items(&mut result, bytes.len())?;
    result.extend_from_slice(bytes);
    Ok(result)
}

fn add_counter(counter: &AtomicU64, value: usize) {
    let value = u64::try_from(value).unwrap_or(u64::MAX);
    add_u64_counter(counter, value);
}

fn add_u64_counter(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

fn snapshot_phase(phase: &TimedPhaseTelemetry) -> TimedPhaseCounters {
    TimedPhaseCounters {
        calls: phase.calls.load(Ordering::Relaxed),
        elapsed_ns: phase.elapsed_ns.load(Ordering::Relaxed),
    }
}

/// Return installed MPI facts when an explicit [`World`] is currently alive.
#[must_use]
pub fn active_mpi_runtime() -> Option<ActiveMpiRuntime> {
    let inner = active_world_slot()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .upgrade()?;
    Some(ActiveMpiRuntime {
        library_version: inner.library_version.clone(),
        world_size: usize::try_from(inner.size).expect("MPI world sizes are positive"),
        mpi_rank: usize::try_from(inner.mpi_rank).expect("MPI ranks are nonnegative"),
        thread_support: inner.thread_support,
    })
}

impl fmt::Debug for World {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("World")
            .field("mpi_rank", &self.mpi_rank())
            .field("size", &self.size())
            .field("library_version", &self.library_version())
            .field("thread_support", &self.thread_support())
            .finish_non_exhaustive()
    }
}
