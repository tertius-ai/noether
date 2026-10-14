# Noether validation

This is the single test, oracle-validation and benchmark root for the Rust
facade's Python projection. Production code never imports from it.

- `unit/` contains deterministic Noether tests and is the default pytest root.
- `corpus/` contains curated, self-contained public-API programs exercised by
  `unit/infra/test_corpus_acceptance.py`; it is validation input, never package
  data.
- `parity/` compares accepted operations with independent implementations.
- `backends/` contains oracle adapters used by the retained parity lanes.
- `bench/` records the requirements for Python boundary benchmarks; kernel and
  facade benchmarks live beside their Rust owners.
- `harness/` contains shared parity support.

The fast local gate is:

```bash
make test-py
```

The 40 numbered files under `examples/` are the narrative documentation
workflows. The broader curated acceptance corpus is a separate test-only layer:

```bash
make test-corpus
```

Its manifest pins the source revision, attribution, public operations, and
execution bounds. The gate requires every program to use only the standard
library and public `noether` imports, contain mathematical assertions, execute
deterministically, and produce a result. The wheel audit rejects all
`validation/` content, so neither the corpus nor its source metadata enters the
runtime package.

Oracle parity is deliberately opt-in:

```bash
make validate
NOETHER_VALIDATE_GAP=1 make validate
NOETHER_VALIDATE_GP=1 make validate
NOETHER_VALIDATE_MAGMA=1 make validate
NOETHER_VALIDATE_SAGE=1 make validate
make validate-sage
```

These Make targets first refresh the in-tree Rust extension from the checked-out
sources. A pull followed by validation is therefore one reproducible command;
manual removal of extension artifacts is neither required nor supported as a
normal workflow. The engine recipe removes every prior in-tree `_engine*.so`
or `_engine*.pyd` before Maturin writes the new extension, so Python ABI suffixes
cannot accumulate. The general and Sage gates use the portable CPU build, while
the release-only Magma contract deliberately uses the native CPU build.

An unavailable oracle whose gate is off is reported as a skip. Enabling a gate
or passing `--require-oracle NAME` requires its executable, at least one
selected test with that oracle's marker, and no collection skips, skips,
expected failures, or unexpected passes in that required lane. Thus a required
oracle cannot produce a green release evaluation without running.
`make validate-sage` runs every Sage-gated pytest comparison and requires Sage
agreement on the U1 benchmark suite. The GAP benchmark suite and GP parity
lane are strict in the same way.
Oracle tools are never imported into the Noether process and are never runtime
backends for the installed package.

The public-release Magma gate is one command on a machine with a licensed
Magma installation:

```bash
NOETHER_MAGMA_BIN=/path/to/magma make validate-magma-release
```

It builds a native engine, executes 71 deterministic paired correctness and
timing workloads, and then requires the full surface-Frobenius oracle. The
required suite covers integer, rational, prime-field, residue-ring and extension-
field arithmetic; dense linear algebra; polynomial arithmetic and
factorization; grevlex and lex ideal Groebner bases; module Groebner bases,
syzygies and graded resolutions; number fields and orders; elliptic curves;
multiple permutation-group families and subgroup operations, including fast
and fallback stabilizer/normalizer regimes; graph
automorphisms; degree-zero and higher integer and finite-field homology; and
Kahler differentials. Every workload is timed on both systems: there is no
correctness-only case in the release contract.

The paired workers import and materialize fixtures outside their timers,
compare serialized results before speed, and record process/setup costs, raw
samples, variance, versions, binary fingerprints, hardware, source revision
and input hashes. The tracked contract also fixes Magma's random seed,
disables per-user Magma startup files, and rejects dirty tracked source. Any
mismatch, missing case, skipped oracle, material slowdown or noisy timing is
red. The untracked report is `target/magma-validation/core.json`; the required
surface log is beside it.

The release contract is intentionally readable in three layers:

- `data/magma_release_v1.json` names every mathematical workload, frozen input
  and timing boundary.
- `bench/magma_release.py` contains the paired Noether and Magma operations,
  exact normalizations, calibration, comparison and report generation.
- `unit/infra/test_magma_release_protocol.py` freezes the corpus and verifies
  completeness, fail-closed protocol behavior and the local mathematical
  results.

Generated API/reference JSON is disposable wheel-build output. Tests should
exercise the live typed surface or run a generator smoke test, never compare a
working tree against a committed generated snapshot.
