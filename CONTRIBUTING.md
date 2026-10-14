# Contributing to Noether

Noether welcomes focused contributions to its mathematics, kernels, Python
interface, documentation, and validation corpus. Search the existing issues
before starting, and open an issue before a substantial API or architectural
change so its mathematical scope and unsupported boundary are explicit.

## Development setup

From a source checkout:

```bash
make sync
scripts/install-flint.sh
make engine
```

See `AGENTS.md` and the nearest component `AGENTS.md` for the complete design,
correctness, performance, and repository contracts.

## Correctness and performance

A mathematical change should include a hand-checkable minimal case, tests of
the relevant laws or certificates, and an independent comparison when a
suitable oracle exists. Unsupported inputs must fail closed. A bug or
performance fix keeps its triggering regression and adds a nearby case that
does not depend on the same shortcut.

Run benchmarks through `scripts/bench-pin.sh` and report the CPU placement,
problem size, build profile, and exact command. Unpinned timings are not
performance evidence.

## Before opening a pull request

Run the local gate:

```bash
make test-all
```

Use `make test-stress` for changes to randomized or expensive mathematics and
an explicitly enabled `make validate` lane when an external CAS is part of the
claim. Update living documentation with the code, but do not commit generated
API references, build products, or benchmark output.

Keep pull requests narrow and state which commands ran. Contributions are
accepted under the repository's MIT license.
