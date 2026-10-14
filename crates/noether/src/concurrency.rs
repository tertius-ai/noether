// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Benjamin Antieau and Deven Manam
//! Process-local concurrency policy for the public facade.

use std::num::NonZeroUsize;
use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Environment variable that may lower Noether's process-local thread budget.
pub const THREAD_BUDGET_ENV: &str = "NOETHER_NUM_THREADS";

/// The process-local CPU budget fixed for this Noether instance.
///
/// Noether defaults to [`std::thread::available_parallelism`], which accounts
/// for process affinity and container CPU limits on supported platforms. A
/// positive `NOETHER_NUM_THREADS` value may lower, but never exceed, that
/// schedulable budget. The setting is read once, before the first capability
/// query or parallel facade operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalConcurrency {
    schedulable_threads: usize,
    requested_threads: Option<usize>,
    effective_threads: usize,
    override_valid: bool,
}

impl LocalConcurrency {
    /// CPUs available to this process according to the standard-library probe.
    #[must_use]
    pub const fn schedulable_threads(&self) -> usize {
        self.schedulable_threads
    }

    /// Explicit requested budget, when `NOETHER_NUM_THREADS` was a positive integer.
    #[must_use]
    pub const fn requested_threads(&self) -> Option<usize> {
        self.requested_threads
    }

    /// Threads in Noether's shared Rayon pool.
    #[must_use]
    pub const fn effective_threads(&self) -> usize {
        self.effective_threads
    }

    /// Whether the optional `NOETHER_NUM_THREADS` setting was absent or valid.
    #[must_use]
    pub const fn override_valid(&self) -> bool {
        self.override_valid
    }

    /// Stable description of the default CPU-budget probe.
    #[must_use]
    pub const fn affinity_source(&self) -> &'static str {
        "std::thread::available_parallelism"
    }

    /// Name of the process-level override.
    #[must_use]
    pub const fn override_name(&self) -> &'static str {
        THREAD_BUDGET_ENV
    }

    /// Owner of parallel Rust phases in the ordinary facade build.
    #[must_use]
    pub const fn phase_owner(&self) -> &'static str {
        "noether-rayon"
    }
}

static LOCAL_CONCURRENCY: OnceLock<LocalConcurrency> = OnceLock::new();
static LOCAL_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// Return the process-local concurrency policy fixed for this Noether instance.
#[must_use]
pub fn local_concurrency() -> &'static LocalConcurrency {
    LOCAL_CONCURRENCY.get_or_init(|| {
        let available = std::thread::available_parallelism()
            .unwrap_or(NonZeroUsize::MIN)
            .get();
        derive_policy(available, std::env::var(THREAD_BUDGET_ENV).ok().as_deref())
    })
}

pub(crate) fn install<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    let pool = LOCAL_POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(local_concurrency().effective_threads())
            .thread_name(|index| format!("noether-{index}"))
            .build()
            .unwrap_or_else(|error| panic!("failed to initialize Noether's Rayon pool: {error}"))
    });
    pool.install(operation)
}

fn derive_policy(available: usize, override_value: Option<&str>) -> LocalConcurrency {
    let available = available.max(1);
    let parsed = override_value.and_then(|value| value.parse::<usize>().ok());
    let requested_threads = parsed.filter(|value| *value > 0);
    LocalConcurrency {
        schedulable_threads: available,
        requested_threads,
        effective_threads: requested_threads.map_or(available, |value| value.min(available)),
        override_valid: override_value.is_none() || requested_threads.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::{derive_policy, install, local_concurrency};

    #[test]
    fn explicit_budget_can_lower_but_not_exceed_affinity() {
        let lowered = derive_policy(8, Some("3"));
        assert_eq!(lowered.requested_threads(), Some(3));
        assert_eq!(lowered.effective_threads(), 3);
        assert!(lowered.override_valid());

        let clamped = derive_policy(8, Some("64"));
        assert_eq!(clamped.requested_threads(), Some(64));
        assert_eq!(clamped.effective_threads(), 8);
    }

    #[test]
    fn invalid_budget_is_visible_and_falls_back_safely() {
        for value in ["", "0", "many"] {
            let policy = derive_policy(4, Some(value));
            assert_eq!(policy.requested_threads(), None);
            assert_eq!(policy.effective_threads(), 4);
            assert!(!policy.override_valid());
        }
    }

    #[test]
    fn nested_phases_share_the_one_pool() {
        let expected = local_concurrency().effective_threads();
        let (outer, inner) = install(|| {
            let outer = rayon::current_num_threads();
            let inner = install(rayon::current_num_threads);
            (outer, inner)
        });
        assert_eq!((outer, inner), (expected, expected));
    }
}
